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
from torch_geometric.nn import GATConv


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

def load_and_clean_data(path:str, n_value:int, random_seed:int)->pl.DataFrame:
    """
    Ładuje dane z pliku Parquet, wykonuje wstępne przetwarzanie chemiczne 
    oraz filtruje rekordy pod kątem jakości i unikalności.

    Proces działania:
    1. Leniwe ładowanie (lazy loading) danych za pomocą Polars.
    2. Pobranie określonej liczby próbek (n_value) dla optymalizacji.
    3. Normalizacja jednostek oraz obliczenie wartości pIC50.
    4. Usunięcie brakujących danych (SMILES, pIC50) oraz wartości nieskończonych.
    5. Deduplikacja na podstawie struktury SMILES, aby zapewnić jeden wynik na związek.

    Args:
        path (str): Ścieżka do pliku .parquet z danymi chemicznymi.
        n_value (int): Liczba wierszy do wylosowania z całego zbioru.
        random_seed (int): Ziarno losowości dla powtarzalności próbkowania.

    Returns:
        polars.DataFrame: Oczyszczona ramka danych gotowa do modelowania.
    """
    print("Ładowanie i czyszczenie danych...")
    df = pl.scan_parquet(path)
    
    # 1. Pobierz próbkę (lub całość) i wykonaj operacje z Twojego notebooka
    df = df.collect().sample(n=n_value, seed=random_seed) 
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


def generate_fingerprints(smiles_list:list, radius:int=2, n_bits:int=1024)->tuple:
    """
    Konwertuje listę struktur SMILES na numeryczne reprezentacje wektorowe (Morgan Fingerprints).
    
    Funkcja wykorzystuje nowoczesne API RDKit (FingerprintGenerator) do transformacji 
    struktur chemicznych w wektory bitowe, które są następnie rzutowane na format 
    macierzowy zgodny z wymaganiami sieci neuronowych.

    Proces działania:
    1. Inicjalizacja generatora Morgana o zadanym promieniu (radius) i długości wektora (n_bits).
    2. Iteracja przez listę SMILES z wizualizacją postępu (tqdm).
    3. Walidacja struktur: jeśli SMILES jest niepoprawny, cząsteczka jest pomijana.
    4. Ekstrakcja fingerprintu bezpośrednio do formatu NumPy (optymalizacja pamięciowa).
    5. Konwersja końcowej macierzy na typ float32, zapewniająca kompatybilność z PyTorch.

    Args:
        smiles_list (list/Iterable): Lista ciągów tekstowych SMILES reprezentujących cząsteczki.
        radius (int): Promień otoczenia atomowego (np. 2 odpowiada ECFP4, 3 odpowiada ECFP6).
        n_bits (int): Długość wynikowego wektora bitowego (rozmiar fingerprintu).

    Returns:
        tuple: (
            np.ndarray: Macierz o kształcie (liczba_poprawnych_mol, n_bits) typu float32,
            list: Indeksy cząsteczek z listy wejściowej, które udało się poprawnie przetworzyć.
        )
    """
    # 1. Tworzymy generator raz przed pętlą (jest to wydajniejsze)
    mfgen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    
    fps = []
    valid_indices = []
    
    for idx, smile in enumerate(tqdm(smiles_list, desc="Generowanie fingerprintów")):
        mol = Chem.MolFromSmiles(smile)
        if mol:
            # 2. Używamy metody GetFingerprintAsNumPy
            # Zwraca ona bezpośrednio tablicę numpy, co oszczędza ręcznej konwersji
            fp_np = mfgen.GetFingerprintAsNumPy(mol)
            
            fps.append(fp_np)
            valid_indices.append(idx)
            
    # Zwracamy jako float32, bo tego oczekuje PyTorch w warstwach Linear
    return np.array(fps, dtype=np.float32), valid_indices

def smile_to_graph(smile:str, target_val:float)->Data:
    """
    Konwertuje pojedynczy ciąg SMILES na obiekt grafowy PyTorch Geometric (Data).

    Funkcja mapuje strukturę molekularną na graf, gdzie atomy stają się węzłami, 
    a wiązania krawędziami. Jest to format wejściowy niezbędny dla sieci 
    grafowych (GNN - Graph Neural Networks).

    Proces transformacji:
    1. Konwersja SMILES na obiekt cząsteczki RDKit.
    2. Generowanie cech węzłów (x): Używa liczby atomowej jako głównej cechy.
    3. Budowa macierzy incydencji (edge_index): Tworzy listę par połączonych atomów. 
       Wiązania są traktowane jako nieskierowane (dodawane w obu kierunkach).
    4. Generowanie cech krawędzi (edge_attr): Koduje rząd wiązania (np. pojedyncze, aromatyczne).
    5. Pakowanie danych: Tworzy obiekt `torch_geometric.data.Data` z etykietą celu (y).

    Args:
        smile (str): Reprezentacja tekstowa cząsteczki w formacie SMILES.
        target_val (float): Wartość docelowa (np. pIC50) przypisana do tej cząsteczki.

    Returns:
        torch_geometric.data.Data: Obiekt grafu zawierający tensory x, edge_index, 
                                   edge_attr oraz y. Zwraca None, jeśli SMILES 
                                   jest niepoprawny lub cząsteczka nie ma wiązań.
    """
    mol = Chem.MolFromSmiles(smile)
    if not mol:
        return None

    # Lista najważniejszych atomów w lekach
    permitted_atoms = [5, 6, 7, 8, 9, 15, 16, 17, 35, 53] # B, C, N, O, F, P, S, Cl, Br, I
    
    node_feats = []
    for atom in mol.GetAtoms():
        # 1. Typ atomu (One-Hot, 10 + 1 "Other" = 11 cech)
        atom_num = atom.GetAtomicNum()
        atom_vec = [int(atom_num == x) for x in permitted_atoms]
        atom_vec.append(int(atom_num not in permitted_atoms)) # "Other"
        
        # 2. Stopień węzła (Liczba sąsiadów)
        degree = atom.GetDegree()
        atom_vec.append(degree)
        
        # 3. Aromatyczność
        atom_vec.append(int(atom.GetIsAromatic()))
        
        # Razem: 11 + 1 + 1 = 13 cech
        node_feats.append(atom_vec)
    
    x = torch.tensor(node_feats, dtype=torch.float) # PyTorch sam obsłuży shape (N, 13)

    edge_indices = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_indices.append([i, j])
        edge_indices.append([j, i])

    if not edge_indices:
        return None

    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    # GCNConv nie obsługuje edge_attr w najprostszej wersji, więc je pomijamy
    y = torch.tensor([target_val], dtype=torch.float)

    return Data(x=x, edge_index=edge_index, y=y)

class MoleculeDatasetMLP(Dataset):
    """
    Standardowy Dataset PyTorch do obsługi wektorowych reprezentacji cząsteczek.

    Klasa ta przygotowuje dane w formacie stałej długości wektorów (np. fingerprintów Morgana),
    które są podawane do klasycznych sieci gęstych (MLP - Multi-Layer Perceptron). 
    Zapewnia automatyczną konwersję danych z formatu NumPy/list na tensory PyTorch 
    oraz dba o poprawny kształt (shape) etykiet celu.

    Atrybuty:
        X (torch.Tensor): Macierz cech o kształcie [liczba_próbek, n_bits].
        y (torch.Tensor): Wektor wartości docelowych o kształcie [liczba_próbek, 1].

    Metody:
        __len__: Zwraca całkowitą liczbę cząsteczek w zbiorze.
        __getitem__: Pobiera pojedynczą parę (cechy, cel) dla danego indeksu.
    """
    def __init__(self, X, y):
        """
        Inicjalizuje dataset, konwertując dane wejściowe na tensory.

        Args:
            X (np.ndarray / list): Macierz fingerprintów lub innych cech numerycznych.
            y (np.ndarray / list): Lista wartości docelowych (np. pIC50).
        """
        # Konwersja na float32 (standard w PyTorch dla warstw Linear)
        self.X = torch.tensor(X, dtype=torch.float32)
        
        # .view(-1, 1) zapewnia, że y jest macierzą kolumnową (wymagane przez MSELoss)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)
    
    def __len__(self):
        """Zwraca liczbę próbek w zbiorze danych."""
        return len(self.y)
    
    def __getitem__(self, idx):
        """
        Pobiera próbkę o wskazanym indeksie.

        Args:
            idx (int): Indeks próbki.

        Returns:
            tuple: (tensor_cech, tensor_celu)
        """
        return self.X[idx], self.y[idx]

# --- MODEL MLP ---
class BioActivityMLP(nn.Module):
    """
    Architektura sieci neuronowej typu MLP (Multi-Layer Perceptron) 
    zoptymalizowana pod kątem regresji aktywności biologicznej.

    Model przyjmuje wektorowy opis cząsteczki (np. fingerprinty) i zwraca 
    ciągłą wartość przewidywanej aktywności (np. pIC50). Zawiera mechanizmy 
    stabilizujące proces uczenia (Batch Normalization) oraz zapobiegające 
    przeuczeniu (Dropout).

    Struktura warstw:
    1. Warstwa wejściowa: Liniowa (input_dim -> hidden_dim).
    2. Batch Normalization: Stabilizuje aktywacje i przyspiesza trening.
    3. Aktywacja ReLU: Wprowadza nieliniowość.
    4. Dropout: Losowo wyłącza 30% neuronów dla lepszej generalizacji.
    5. Warstwa ukryta: Redukuje wymiarowość (hidden_dim -> hidden_dim // 2).
    6. Warstwa wyjściowa: Pojedynczy neuron zwracający wynik regresji.

    Args:
        input_dim (int): Rozmiar wektora wejściowego (domyślnie 1024 bity fingerprintu).
        hidden_dim (int): Liczba neuronów w pierwszej warstwie ukrytej.
    """
    def __init__(self, input_dim=1024, hidden_dim=256):
        super(BioActivityMLP, self).__init__()
        # Pierwsza warstwa gęsta
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        # Normalizacja wsadowa (pomaga przy dużych różnicach w wagach)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        # Regularyzacja (p=0.3)
        self.dropout = nn.Dropout(0.3)
        # Druga warstwa gęsta ze zwężeniem szerokości sieci
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        # Warstwa wyjściowa dla regresji (brak funkcji aktywacji na końcu)
        self.fc3 = nn.Linear(hidden_dim // 2, 1) 
        
    def forward(self, x:torch.Tensor)->torch.Tensor:
        """
        Definiuje przepływ danych przez sieć.

        Args:
            x (torch.Tensor): Batch danych o kształcie [batch_size, input_dim].

        Returns:
            torch.Tensor: Przewidywana wartość pIC50 o kształcie [batch_size, 1].
        """
        # Przejście przez pierwszą warstwę z BN i ReLU
        x = F.relu(self.bn1(self.fc1(x)))
        # Aplikacja Dropoutu tylko w fazie treningu
        x = self.dropout(x)
        # Przejście przez drugą warstwę ukrytą
        x = F.relu(self.fc2(x))
        # Finalny wynik (regresja liniowa na wyjściu)
        return self.fc3(x)

# --- MODEL GNN ---

class BioActivityGNN(torch.nn.Module):
    """
    Grafowa Sieć Neuronowa (GNN) oparta na architekturze GCN (Graph Convolutional Network)
    przeznaczona do predykcji aktywności biologicznej cząsteczek.

    W przeciwieństwie do MLP, model ten operuje bezpośrednio na strukturze grafu molekularnego,
    wykorzystując konektywność atomów do nauki reprezentacji przestrzennej związku.

    Proces przetwarzania:
    1. Konwolucje Grafowe (GCNConv): Agregacja cech z sąsiednich atomów w celu wytworzenia 
       lokalnych reprezentacji chemicznych.
    2. Nieliniowość (ReLU): Wprowadzenie złożoności do wyuczonych cech.
    3. Global Pooling (Readout): Redukcja cech wszystkich atomów do jednego wektora 
       reprezentującego całą cząsteczkę (uśrednianie cech węzłów).
    4. Warstwa Liniowa: Mapowanie globalnej reprezentacji grafu na wynik regresji.

    Args:
        node_features_dim (int): Liczba cech wejściowych dla każdego atomu (np. liczba atomowa).
        hidden_dim (int): Wymiarowość przestrzeni ukrytej w warstwach konwolucyjnych.
    """
    def __init__(self, node_features_dim=1, hidden_dim=64):
        super(BioActivityGNN, self).__init__()
        
        # Pierwsza warstwa: ekstrakcja cech z atomów
        self.conv1 = GCNConv(node_features_dim, hidden_dim)
        # Druga warstwa: zwiększenie wymiarowości i głębokości pola recepcyjnego
        self.conv2 = GCNConv(hidden_dim, hidden_dim * 2)
        # Trzecia warstwa: dalsza propagacja informacji w grafie
        self.conv3 = GCNConv(hidden_dim * 2, hidden_dim * 2)
        
        # Warstwa wyjściowa po agregacji globalnej (Readout)
        self.lin = nn.Linear(hidden_dim * 2, 1)

    def forward(self, data):
        """
        Definiuje przepływ danych (message passing) przez graf.

        Args:
            data (torch_geometric.data.Batch): Obiekt zawierający połączone grafy (x, edge_index, batch).

        Returns:
            torch.Tensor: Przewidywana aktywność (pIC50) o kształcie [batch_size, 1].
        """
        x, edge_index = data.x, data.edge_index
        batch = data.batch # Wektor przypisujący każdy węzeł do konkretnego grafu w paczce

        # 1. Uzyskanie reprezentacji węzłów (Message Passing)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)

        # 2. Agregacja węzłów do reprezentacji całego grafu (Global Pooling)
        # Przejście z wymiaru [suma_atomów, features] do [batch_size, features]
        x = global_mean_pool(x, batch) 

        # 3. Warstwa wyjściowa (Regresor)
        x = self.lin(x)
        return x