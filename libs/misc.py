import polars as pl
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm
from torch_geometric.data import Data
import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.loader import DataLoader as GeoDataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn as nn
import torch.nn.functional as F
from rdkit.Chem import rdFingerprintGenerator
from rdkit import Chem
import numpy as np
from tqdm import tqdm


def compute_pIC50(df: pl.DataFrame) -> pl.DataFrame:
    """
    Oblicza pIC50 = -log10(M) dla wartości w nM.
    Dodaje kolumnę pIC50.
    """
    # Zamieniamy tylko rekordy, które mają jednostki "nM" i standard_value nie null
    df = df.with_columns(
        pl.when(
            (pl.col("standard_units") == "nM") & pl.col("standard_value").is_not_null()
        )
        .then(-(pl.col("standard_value") * 1e-9).log10())
        .otherwise(None)
        .alias("pIC50")
    )
    return df


def impute_units(df: pl.DataFrame, value_col="standard_value", units_col="standard_units") -> pl.DataFrame:
    """
    Imputuje brakujące standard_units na podstawie zakresu standard_value
    i dodaje kolumnę units_imputed.
    """
    df = df.with_columns([
        pl.lit(False).alias("units_imputed")
    ])

    # Maski: brak jednostki i wartość istnieje
    mask_missing = pl.col(units_col).is_null() & pl.col(value_col).is_not_null()
    mask_range = (pl.col(value_col) >= 0.01) & (pl.col(value_col) <= 1e6)

    df = df.with_columns([
        pl.when(mask_missing & mask_range)
        .then(pl.lit("nM"))
        .otherwise(pl.col(units_col))
        .alias(units_col),
        pl.when(mask_missing & mask_range)
        .then(pl.lit(True))
        .otherwise(pl.lit(False))
        .alias("units_imputed")
    ])

    return df

def load_and_clean_data(path):
    print("Ładowanie i czyszczenie danych...")
    df = pl.scan_parquet(path)
    
    # 1. Pobierz próbkę (lub całość) i wykonaj operacje z Twojego notebooka
    df = df.collect() # .sample(n=50000, seed=RANDOM_SEED) # Odkomentuj dla testów na mniejszej próbce
    df = impute_units(df)
    df = compute_pIC50(df)
    
    # 2. Filtrowanie
    # Usuwamy rekordy bez SMILES, bez pIC50 (target) oraz duplikaty
    df_clean = df.filter(
        pl.col("canonical_smiles").is_not_null() &
        pl.col("pIC50").is_not_null() & 
        (pl.col("pIC50").is_infinite().not_()) # Usuwamy inf
    ).unique(subset=["canonical_smiles"]) # Jeden pIC50 na jeden SMILES (uproszczenie)
    
    print(f"Dane po oczyszczeniu: {df_clean.shape}")
    return df_clean


def generate_fingerprints(smiles_list, radius=2, n_bits=1024):
    """
    Generuje fingerprinty Morgana używając nowego API RDKit (rdFingerprintGenerator).
    """
    # 1. Tworzymy generator raz przed pętlą (jest to wydajniejsze)
    # fpSize zastępuje nBits, radius pozostaje ten sam
    mfgen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    
    fps = []
    valid_indices = []
    
    for idx, smile in enumerate(tqdm(smiles_list, desc="Generowanie fingerprintów")):
        mol = Chem.MolFromSmiles(smile)
        if mol:
            # 2. Używamy metody GetFingerprintAsNumPy
            # Zwraca ona bezpośrednio tablicę numpy (zazwyczaj uint8 lub int)
            # co oszczędza nam ręcznej konwersji z BitVect
            fp_np = mfgen.GetFingerprintAsNumPy(mol)
            
            fps.append(fp_np)
            valid_indices.append(idx)
            
    # Zwracamy jako float32, bo tego oczekuje PyTorch w warstwach Linear
    return np.array(fps, dtype=np.float32), valid_indices


def smile_to_graph(smile, target_val):
    mol = Chem.MolFromSmiles(smile)
    if not mol:
        return None

    # Cechy węzłów (Atomy) - uproszczone: tylko liczba atomowa (Atomic Num)
    # W produkcji dodaje się też hybrydyzację, ładunek itp.
    node_feats = []
    for atom in mol.GetAtoms():
        node_feats.append(atom.GetAtomicNum())
    
    x = torch.tensor(node_feats, dtype=torch.float).view(-1, 1)

    # Krawędzie (Wiązania)
    edge_indices = []
    edge_attrs = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        
        # Grafy są nieskierowane, dodajemy (i, j) oraz (j, i)
        edge_indices.append([i, j])
        edge_indices.append([j, i])
        
        # Typ wiązania jako cecha (1.0 = Single, 1.5 = Aromatic, 2.0 = Double, 3.0 = Triple)
        b_type = bond.GetBondTypeAsDouble()
        edge_attrs.append(b_type)
        edge_attrs.append(b_type)

    if not edge_indices: # Samotny atom bez wiązań
        return None

    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attrs, dtype=torch.float).view(-1, 1)
    
    y = torch.tensor([target_val], dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)


class MoleculeDatasetMLP(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# --- MODEL MLP ---
class BioActivityMLP(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=256):
        super(BioActivityMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, 1) # Regresja: 1 wyjście
        
    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# --- MODEL GNN ---

class BioActivityGNN(torch.nn.Module):
    def __init__(self, node_features_dim=1, hidden_dim=64):
        super(BioActivityGNN, self).__init__()
        # Graph Convolution Layers
        self.conv1 = GCNConv(node_features_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim * 2)
        self.conv3 = GCNConv(hidden_dim * 2, hidden_dim * 2)
        
        # Readout layer
        self.lin = nn.Linear(hidden_dim * 2, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        batch = data.batch # batch vector informujący, który węzeł należy do którego grafu

        # 1. Uzyskanie reprezentacji węzłów
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)

        # 2. Agregacja węzłów do reprezentacji całego grafu (Global Pooling)
        x = global_mean_pool(x, batch) 

        # 3. Klasyfikator / Regresor
        x = self.lin(x)
        return x