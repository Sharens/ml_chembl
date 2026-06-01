from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem
from torch_geometric.data import Data
from torch_geometric.nn import BatchNorm, GINEConv, global_add_pool, global_mean_pool

MODEL_CACHE_DIR = Path("processed_data/model_cache")
GRAPH_CACHE_DIR = Path("processed_data/graph_cache")

ATOMIC_NUM_LIST = [1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 34, 35, 53]
HYBRIDIZATION_TYPES = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]
BOND_TYPES = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]
BOND_STEREO_TYPES = [
    Chem.rdchem.BondStereo.STEREONONE,
    Chem.rdchem.BondStereo.STEREOZ,
    Chem.rdchem.BondStereo.STEREOE,
    Chem.rdchem.BondStereo.STEREOCIS,
    Chem.rdchem.BondStereo.STEREOTRANS,
]
CHIRAL_TAGS = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
]


def one_hot_encode(value, categories):
    return [1 if value == cat else 0 for cat in categories]


def mol_to_graph(smiles: str) -> Data | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    node_feats = []
    for atom in mol.GetAtoms():
        atomic_num = atom.GetAtomicNum()
        one_hot_atomic = (
            one_hot_encode(atomic_num, ATOMIC_NUM_LIST)
            if atomic_num in ATOMIC_NUM_LIST
            else [0] * len(ATOMIC_NUM_LIST)
        )

        hybridization = atom.GetHybridization()
        one_hot_hybrid = (
            one_hot_encode(hybridization, HYBRIDIZATION_TYPES)
            if hybridization in HYBRIDIZATION_TYPES
            else [0] * len(HYBRIDIZATION_TYPES)
        )

        chiral_tag = atom.GetChiralTag()
        one_hot_chiral = (
            one_hot_encode(chiral_tag, CHIRAL_TAGS)
            if chiral_tag in CHIRAL_TAGS
            else [0] * len(CHIRAL_TAGS)
        )

        degree = atom.GetTotalDegree() / 4.0
        formal_charge = (atom.GetFormalCharge() + 4) / 8.0
        num_hs = atom.GetTotalNumHs() / 4.0
        total_valence = atom.GetTotalValence() / 8.0
        num_radical = atom.GetNumRadicalElectrons() / 2.0
        is_aromatic = float(atom.GetIsAromatic())
        is_in_ring = float(atom.IsInRing())

        node_feats.append(
            one_hot_atomic
            + one_hot_hybrid
            + one_hot_chiral
            + [
                degree,
                formal_charge,
                num_hs,
                total_valence,
                num_radical,
                is_aromatic,
                is_in_ring,
            ]
        )

    x = torch.tensor(node_feats, dtype=torch.float)

    edge_index_list = []
    edge_attr_list = []
    for bond in mol.GetBonds():
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()

        bond_type = bond.GetBondType()
        one_hot_bond = (
            one_hot_encode(bond_type, BOND_TYPES)
            if bond_type in BOND_TYPES
            else [0] * len(BOND_TYPES)
        )

        stereo = bond.GetStereo()
        one_hot_stereo = (
            one_hot_encode(stereo, BOND_STEREO_TYPES)
            if stereo in BOND_STEREO_TYPES
            else [0] * len(BOND_STEREO_TYPES)
        )

        bond_feats = (
            one_hot_bond
            + one_hot_stereo
            + [float(bond.GetIsConjugated()), float(bond.IsInRing())]
        )
        edge_index_list.extend([[u, v], [v, u]])
        edge_attr_list.extend([bond_feats, bond_feats])

    edge_dim = len(BOND_TYPES) + len(BOND_STEREO_TYPES) + 2
    if edge_index_list:
        edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr_list, dtype=torch.float)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, edge_dim), dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


class GNNRegressor(torch.nn.Module):
    def __init__(
        self,
        node_features: int,
        edge_features: int,
        hidden_dim: int = 128,
        num_layers: int = 4,
        dropout: float = 0.15,
        pooling: str = "mean",
    ):
        super().__init__()
        if pooling not in {"mean", "add"}:
            raise ValueError("pooling must be 'mean' or 'add'")

        self.pooling = pooling
        self.dropout = dropout
        self.node_proj = nn.Linear(node_features, hidden_dim)
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.convs.append(GINEConv(mlp, edge_dim=hidden_dim))
            self.norms.append(BatchNorm(hidden_dim))

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def _pool(self, x, batch):
        if self.pooling == "add":
            return global_add_pool(x, batch)
        return global_mean_pool(x, batch)

    def forward(self, data):
        x, edge_index, edge_attr, batch = (
            data.x,
            data.edge_index,
            data.edge_attr,
            data.batch,
        )
        x = self.node_proj(x)
        edge_attr = self.edge_encoder(edge_attr)

        for conv, norm in zip(self.convs, self.norms):
            residual = x
            x = conv(x, edge_index, edge_attr)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = x + residual

        x = self._pool(x, batch)
        return self.head(x)


def _infer_architecture(state_dict: dict) -> dict:
    node_features = state_dict["node_proj.weight"].shape[1]
    edge_features = state_dict["edge_encoder.0.weight"].shape[1]
    hidden_dim = state_dict["node_proj.weight"].shape[0]
    num_layers = sum(
        1 for k in state_dict if k.startswith("convs.") and k.endswith(".eps")
    )
    return {
        "node_features": node_features,
        "edge_features": edge_features,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
    }


def _get_best_model_path() -> Path | None:
    if not MODEL_CACHE_DIR.exists():
        return None
    best_path = None
    best_r2 = -float("inf")
    for fpath in sorted(MODEL_CACHE_DIR.glob("default_*.pt")):
        try:
            ckpt = torch.load(fpath, map_location="cpu", weights_only=False)
            result = ckpt.get("result", {})
            if result.get("model") == "GNN" and result.get("split") == "scaffold":
                r2 = result.get("r2_val", -float("inf"))
                if r2 > best_r2:
                    best_r2 = r2
                    best_path = fpath
        except Exception:
            continue
    return best_path


def load_model(
    model_path: Path | None = None, device: torch.device | None = None
) -> tuple[nn.Module, dict] | None:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_path is None:
        model_path = _get_best_model_path()

    if model_path is None or not model_path.exists():
        return None

    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    result = ckpt.get("result", {})
    arch = _infer_architecture(ckpt["model_state_dict"])
    pooling = result.get("pooling", "mean")
    dropout = result.get("gnn_dropout", 0.15)

    model = GNNRegressor(
        node_features=arch["node_features"],
        edge_features=arch["edge_features"],
        hidden_dim=arch["hidden_dim"],
        num_layers=arch["num_layers"],
        dropout=dropout,
        pooling=pooling,
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, result


def predict_pic50(
    smiles: str,
    model: nn.Module | None = None,
    device: torch.device | None = None,
) -> dict:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {
            "smiles": smiles,
            "pIC50": None,
            "valid": False,
            "error": "Invalid SMILES string",
        }

    data = mol_to_graph(smiles)
    if data is None:
        return {
            "smiles": smiles,
            "pIC50": None,
            "valid": False,
            "error": "Could not convert molecule to graph",
        }

    if model is None:
        loaded = load_model(device=device)
        if loaded is None:
            return {
                "smiles": smiles,
                "pIC50": None,
                "valid": False,
                "error": "No trained model found. Run training first.",
            }
        model, _ = loaded

    data = data.to(device)
    with torch.no_grad():
        pred = model(data).cpu().item()

    return {
        "smiles": smiles,
        "pIC50": round(float(pred), 4),
        "valid": True,
        "error": None,
    }
