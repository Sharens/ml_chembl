from __future__ import annotations

from copy import deepcopy

import numpy as np
import polars as pl
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as GeoDataLoader

from src.models.config import (
    ATOMIC_NUM_LIST,
    BOND_STEREO_TYPES,
    BOND_TYPES,
    CHIRAL_TAGS,
    GRAPH_CACHE_DIR,
    GRAPH_DATA_CACHE,
    HYBRIDIZATION_INTS,
    MFP_N_BITS,
    MFP_RADIUS,
)
from src.models.training import compute_num_workers, get_device

RDLogger.logger().setLevel(RDLogger.ERROR)

# ---------------------------------------------------------------------------
# Morgan fingerprint generator (inicjalizowany raz na poziomie modulu)
# ---------------------------------------------------------------------------
_mfpgen = rdFingerprintGenerator.GetMorganGenerator(
    radius=MFP_RADIUS, fpSize=MFP_N_BITS
)


def fp_from_smiles(smiles: str) -> np.ndarray | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        return _mfpgen.GetFingerprintAsNumPy(mol).astype(np.float32)
    except Exception:
        return None


def get_scaffold_smiles(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)


# ---------------------------------------------------------------------------
# Splity danych
# ---------------------------------------------------------------------------


def random_split_df(df_fp: pl.DataFrame, test_size=0.1, val_size=0.1, seed=42):
    df_shuffled = df_fp.sample(fraction=1.0, seed=seed)
    n = len(df_shuffled)
    n_test = int(test_size * n)
    n_val = int(val_size * n)
    test_df = df_shuffled[:n_test]
    val_df = df_shuffled[n_test : n_test + n_val]
    train_df = df_shuffled[n_test + n_val :]
    return train_df, val_df, test_df


def scaffold_split(df_fp: pl.DataFrame, test_size=0.1, val_size=0.1, seed=42):
    df_scaff = df_fp.with_row_index("row_id").with_columns(
        pl.col("canonical_smiles")
        .map_elements(get_scaffold_smiles, return_dtype=pl.Utf8)
        .alias("scaffold")
    )
    df_scaff = df_scaff.filter(pl.col("scaffold").is_not_null())

    stats = (
        df_scaff.group_by("scaffold")
        .agg(pl.len().alias("count"))
        .sort("count", descending=True)
    )

    rng = np.random.default_rng(seed)
    scaffold_rows = stats.to_dicts()
    rng.shuffle(scaffold_rows)
    scaffold_rows.sort(key=lambda r: r["count"], reverse=True)

    total = int(df_scaff.height)
    target_test = int(total * test_size)
    target_val = int(total * val_size)

    split_counts = {"train": 0, "val": 0, "test": 0}
    scaff_to_split = {}

    for row in scaffold_rows:
        scaf = row["scaffold"]
        count = int(row["count"])
        need_test = target_test - split_counts["test"]
        need_val = target_val - split_counts["val"]

        if need_test > 0 and (need_test >= need_val):
            split = "test"
        elif need_val > 0:
            split = "val"
        else:
            split = "train"

        scaff_to_split[scaf] = split
        split_counts[split] += count

    df_labeled = df_scaff.with_columns(
        pl.col("scaffold")
        .replace_strict(scaff_to_split, default="train")
        .alias("split")
    )

    return (
        df_labeled.filter(pl.col("split") == "train"),
        df_labeled.filter(pl.col("split") == "val"),
        df_labeled.filter(pl.col("split") == "test"),
    )


def get_split_dfs(df_fp: pl.DataFrame, split_type="random", seed=42):
    if split_type == "scaffold":
        return scaffold_split(df_fp, seed=seed)
    if split_type == "random":
        return random_split_df(df_fp, seed=seed)
    if split_type == "family":
        return family_split(df_fp, seed=seed)
    raise ValueError("split_type must be 'random', 'scaffold', or 'family'")


def family_split(
    df_fp: pl.DataFrame,
    n_families: int = 4,
    n_train_families: int = 2,
    n_val_families: int = 1,
    seed: int = 42,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Hold-out split based on molecular families (unsupervised clustering).

    1. Computes Morgan fingerprints.
    2. Reduces to 50 PCA dimensions.
    3. Clusters molecules into n_families using KMeans.
    4. Assigns families to train/val/test, ensuring families are different.

    The test set is always one family (the remaining one).
    """
    from scipy.spatial.distance import cdist
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA

    df_idx = df_fp.with_row_index("row_id")

    fps = np.stack(df_idx["fp"].to_list()).astype(np.float64)
    pca = PCA(n_components=min(50, fps.shape[0] - 1, fps.shape[1]), random_state=seed)
    fps_reduced = pca.fit_transform(fps)

    kmeans = KMeans(n_clusters=n_families, random_state=seed, n_init=10)
    labels = kmeans.fit_predict(fps_reduced)

    centroids = kmeans.cluster_centers_
    inter_dists = cdist(centroids, centroids)
    np.fill_diagonal(inter_dists, float("inf"))
    min_inter = inter_dists.min()

    intra_dists = []
    for c in range(n_families):
        mask = labels == c
        if mask.sum() > 1:
            intra = cdist(fps_reduced[mask], fps_reduced[mask]).mean()
        else:
            intra = 0.0
        intra_dists.append(intra)
    mean_intra = np.mean(intra_dists)

    if min_inter <= mean_intra:
        print(
            f"WARNING: min inter-family dist ({min_inter:.3f}) <= "
            f"mean intra-family dist ({mean_intra:.3f}). "
            "Families may not be well separated. Consider increasing n_families."
        )

    rng = np.random.default_rng(seed)
    family_ids = np.arange(n_families)
    rng.shuffle(family_ids)

    train_ids = set(family_ids[:n_train_families])
    val_ids = set(family_ids[n_train_families : n_train_families + n_val_families])
    test_ids = set(family_ids[n_train_families + n_val_families :])

    df_labeled = df_idx.with_columns(pl.Series("_family", labels).cast(pl.Int64))

    return (
        df_labeled.filter(pl.col("_family").is_in(list(train_ids))),
        df_labeled.filter(pl.col("_family").is_in(list(val_ids))),
        df_labeled.filter(pl.col("_family").is_in(list(test_ids))),
    )


# ---------------------------------------------------------------------------
# Loadery dla MLP
# ---------------------------------------------------------------------------


def build_mlp_loaders(df_fp: pl.DataFrame, split_type="random", batch_size=64, seed=42):
    device = get_device()
    train_df, val_df, test_df = get_split_dfs(df_fp, split_type=split_type, seed=seed)

    X_train = np.stack(train_df["fp"].to_list()).astype(np.float32)
    y_train = np.array(train_df["pIC50"].to_list(), dtype=np.float32)
    X_val = np.stack(val_df["fp"].to_list()).astype(np.float32)
    y_val = np.array(val_df["pIC50"].to_list(), dtype=np.float32)
    X_test = np.stack(test_df["fp"].to_list()).astype(np.float32)
    y_test = np.array(test_df["pIC50"].to_list(), dtype=np.float32)

    num_workers = compute_num_workers()
    pin = device.type == "cuda"

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)),
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=(num_workers > 0),
    )
    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test)),
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=(num_workers > 0),
    )
    return train_loader, val_loader, test_loader


# ---------------------------------------------------------------------------
# GNN – graph building i loadery
# ---------------------------------------------------------------------------


def one_hot_encode(value, categories):
    return [1 if value == cat else 0 for cat in categories]


def mol_to_graph(smiles: str, target: float):
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

        hybridization = int(atom.GetHybridization())
        one_hot_hybrid = (
            one_hot_encode(hybridization, HYBRIDIZATION_INTS)
            if hybridization in HYBRIDIZATION_INTS
            else [0] * len(HYBRIDIZATION_INTS)
        )

        chiral_tag = atom.GetChiralTag()
        one_hot_chiral = (
            one_hot_encode(chiral_tag, CHIRAL_TAGS)
            if chiral_tag in CHIRAL_TAGS
            else [0] * len(CHIRAL_TAGS)
        )

        degree = atom.GetTotalDegree() / 6.0
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
            + [
                float(bond.GetIsConjugated()),
                float(bond.IsInRing()),
            ]
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

    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=torch.tensor([target], dtype=torch.float),
        smiles=smiles,
    )


def _build_all_graphs(df_fp: pl.DataFrame):
    graphs = []
    smiles_to_idx = {}

    for row in df_fp.select(["canonical_smiles", "pIC50"]).iter_rows(named=True):
        smiles = row["canonical_smiles"]
        if smiles in smiles_to_idx:
            continue
        g = mol_to_graph(smiles, float(row["pIC50"]))
        if g is not None:
            smiles_to_idx[smiles] = len(graphs)
            graphs.append(g)

    return graphs, smiles_to_idx


def load_or_build_graph_cache(df_fp: pl.DataFrame):
    GRAPH_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if GRAPH_DATA_CACHE.exists():
        try:
            cached = torch.load(GRAPH_DATA_CACHE, weights_only=False)
            return cached["graphs"], cached["smiles_to_idx"]
        except Exception as exc:
            print(f"Cache load failed ({exc}); rebuilding graph cache...")

    graphs, smiles_to_idx = _build_all_graphs(df_fp)
    torch.save({"graphs": graphs, "smiles_to_idx": smiles_to_idx}, GRAPH_DATA_CACHE)
    return graphs, smiles_to_idx


def _subset_graphs(df_part: pl.DataFrame, all_graphs, smiles_to_idx):
    graphs = []
    for smiles, target in df_part.select(["canonical_smiles", "pIC50"]).iter_rows():
        idx = smiles_to_idx.get(smiles)
        if idx is None:
            continue
        g = deepcopy(all_graphs[idx])
        g.y = torch.tensor(float(target), dtype=torch.float)
        graphs.append(g)
    return graphs


def build_gnn_loaders(df_fp: pl.DataFrame, split_type="random", batch_size=64, seed=42):
    device = get_device()
    train_df, val_df, test_df = get_split_dfs(df_fp, split_type=split_type, seed=seed)

    all_graphs, smiles_to_idx = load_or_build_graph_cache(df_fp)

    train_graphs = _subset_graphs(train_df, all_graphs, smiles_to_idx)
    val_graphs = _subset_graphs(val_df, all_graphs, smiles_to_idx)
    test_graphs = _subset_graphs(test_df, all_graphs, smiles_to_idx)

    num_workers = compute_num_workers()
    pin = device.type == "cuda"

    train_loader = GeoDataLoader(
        train_graphs,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=(num_workers > 0),
    )
    val_loader = GeoDataLoader(
        val_graphs,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=(num_workers > 0),
    )
    test_loader = GeoDataLoader(
        test_graphs,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=(num_workers > 0),
    )
    return train_loader, val_loader, test_loader
