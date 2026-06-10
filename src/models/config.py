from rdkit import Chem

from src._config import MODEL_CACHE

EPOCHS_DEFAULT = 200
LR_DEFAULT = 3e-4
BATCH_SIZE_DEFAULT = 64
SEED_DEFAULT = 42

EARLY_STOPPING_PATIENCE_DEFAULT = 15
MIN_DELTA_DEFAULT = 1e-4
WEIGHT_DECAY_DEFAULT = 1e-5
POOLING_DEFAULT = "mean"
LOSS_FN_DEFAULT = "huber"

MFP_RADIUS = 2
MFP_N_BITS = 2048

ATOMIC_NUM_LIST = [1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 34, 35, 53]

HYBRIDIZATION_TYPES = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]
HYBRIDIZATION_INTS = [int(h) for h in HYBRIDIZATION_TYPES]

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

GRAPH_CACHE_DIR = MODEL_CACHE.parent / "graph_cache"
GRAPH_DATA_CACHE = GRAPH_CACHE_DIR / "all_graphs_v2.pt"
GRAPH_SPLIT_CACHE_TEMPLATE = "split_{split}_seed{seed}.pt"
