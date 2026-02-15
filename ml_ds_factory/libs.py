import sys
import os
from pathlib import Path

from attr import dataclass

project_root = '~/projects/python/ml_chembl'
sys.path.append(os.path.expanduser(project_root))
import logging
import polars as pl
from data_fetcher.config import CONFIG


@dataclass 
class Config:
# Sciezka do plikow parquetowych
    data_path: Path = Path("/home/sharens/projects/python/ml_chembl/data_fetcher/raw")


# Wczytywanie danych z plików Parquet
class DataLoader:
    def __init__(self, config: Config):
        self.config = config

    def load_data(self):
        data_path = self.config.data_path
        logging.info(f"Loading data from {data_path}")
        activities = pl.scan_parquet(data_path / "raw_activities.parquet")
        assays = pl.scan_parquet(data_path / "raw_assays.parquet")
        target_dictionary = pl.scan_parquet(data_path / "raw_target_dictionary.parquet")
        compound_structures = pl.scan_parquet(data_path / "raw_compound_structures.parquet")
        compound_properties = pl.scan_parquet(data_path / "raw_compound_properties.parquet")
        
        return activities, assays, target_dictionary, compound_structures, compound_properties

class DataProcessor:
    def __init__(self, config: Config):
        self.config = config

    def process_data(self, activities, assays, target_dictionary, compound_structures, compound_properties):
        # Przygotujmy target_dictionary wcześniej, żeby uniknąć konfliktów nazw
        td_prepared = target_dictionary.select([
            pl.col("tid"),
            pl.col("organism"),
            pl.col("chembl_id").alias("target_chembl_id"), # Od razu zmieniamy nazwę
            pl.col("pref_name").alias("target_name")
        ])

        query = (
            activities
            .join(assays, on="assay_id")
            .join(td_prepared, on="tid") # Teraz nie ma konfliktu nazw
            .join(compound_structures, on="molregno")
            .join(compound_properties, on="molregno")
            .filter(
                (pl.col("organism").str.to_lowercase() == "homo sapiens") &
                (pl.col("pchembl_value").is_not_null()) &
                (pl.col("canonical_smiles").is_not_null()) &
                ((pl.col("potential_duplicate").is_null()) | (pl.col("potential_duplicate") == 0))
            )
            .select([
                "activity_id",
                "molregno",
                "canonical_smiles",
                "mw_freebase",
                "alogp",
                "hba",
                "hbd",
                "psa",
                "rtb",
                "aromatic_rings",
                "qed_weighted",
                "standard_value",
                "standard_units",
                "standard_type",
                "standard_relation",
                "pchembl_value",
                "target_chembl_id", 
                "target_name",
                "confidence_score"
            ])
        )


        df = query.collect()

        return df