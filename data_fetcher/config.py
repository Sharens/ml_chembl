from pathlib import Path
from dataclasses import dataclass

PROJECT_ROOT = Path(__file__).resolve().parent.parent

@dataclass
class Config:
    """Central configuration class for the data fetching and processing pipeline."""
    db_path: Path = Path(f"{PROJECT_ROOT}/data_fetcher/raw/chembl_36.db")
    raw_data_path: Path = Path(f"{PROJECT_ROOT}/data_fetcher/raw/")
    row_limit: int = 1000000
    download_url: str = "https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/chembl_36_sqlite.tar.gz"
    archive_name: str = "chembl_36_sqlite.tar.gz"
    internal_db_path: str = "chembl_36/chembl_36_sqlite/chembl_36.db"

CONFIG = Config()
