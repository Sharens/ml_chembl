import polars as pl
from pathlib import Path
from attr import dataclass
import logging

@dataclass 
class Config:
    data_path: Path = Path(__file__).resolve().parents[1] / "data_fetcher" / "raw"

class DataMisc:
    @staticmethod
    def compute_pIC50(df: pl.DataFrame) -> pl.DataFrame:
        """Computes pIC50 = -log10(M) for values in nM."""
        return df.with_columns(
            pl.when(
                (pl.col("standard_units") == "nM") & pl.col("standard_value").is_not_null()
            )
            .then(-(pl.col("standard_value") * 1e-9).log10())
            .otherwise(pl.col("pchembl_value")) # If no calculation is possible, use the ready pchembl_value
            .alias("pIC50")
        )

    @staticmethod
    def impute_units(df: pl.DataFrame) -> pl.DataFrame:
        """Imputes missing nM units for sensible value ranges."""
        mask_missing = pl.col("standard_units").is_null() & pl.col("standard_value").is_not_null()
        mask_range = (pl.col("standard_value") >= 0.01) & (pl.col("standard_value") <= 1e6)

        return df.with_columns(
            pl.when(mask_missing & mask_range)
            .then(pl.lit("nM"))
            .otherwise(pl.col("standard_units"))
            .alias("standard_units")
        )

class DataLoader:
    def __init__(self, config: Config):
        self.config = config

    def load_from_sqlite(self) -> pl.DataFrame:
        """Fetches an optimized dataset directly from the SQLite database using the ADBC engine."""
        
        # Create an absolute path to avoid problems with .db file location
        db_path = self.config.data_path / "chembl_36.db"
        uri = f"sqlite:///{db_path.absolute()}"
        
        # Explicit type casting (CAST) solves the problem with NULLs in connectorx/adbc
        query = """
            SELECT 
                CAST(act.activity_id AS INT) AS activity_id,
                CAST(act.molregno AS INT) AS molregno,
                CAST(cs.canonical_smiles AS TEXT) AS canonical_smiles,
                CAST(cp.mw_freebase AS REAL) AS mw_freebase,
                CAST(cp.alogp AS REAL) AS alogp,
                CAST(cp.hba AS INT) AS hba,
                CAST(cp.hbd AS INT) AS hbd,
                CAST(cp.psa AS REAL) AS psa,
                CAST(cp.rtb AS INT) AS rtb,
                CAST(cp.aromatic_rings AS INT) AS aromatic_rings,
                CAST(cp.qed_weighted AS REAL) AS qed_weighted,
                CAST(act.standard_value AS REAL) AS standard_value,
                CAST(act.standard_units AS TEXT) AS standard_units,
                CAST(act.standard_type  AS TEXT) AS standard_type,
                CAST(act.standard_relation AS TEXT) AS standard_relation,
                CAST(act.pchembl_value AS REAL) AS pchembl_value,
                CAST(td.chembl_id AS TEXT) AS target_chembl_id,
                CAST(td.pref_name AS TEXT) AS target_name,
                CAST(ass.confidence_score AS INT) AS confidence_score
            FROM activities act
            JOIN assays ass ON act.assay_id = ass.assay_id
            JOIN target_dictionary td ON ass.tid = td.tid
            JOIN compound_structures cs ON act.molregno = cs.molregno
            JOIN compound_properties cp ON act.molregno = cp.molregno
            WHERE LOWER(td.organism) = 'homo sapiens'
                AND cs.canonical_smiles IS NOT NULL
                AND (act.potential_duplicate IS NULL OR act.potential_duplicate = 0)
                AND (act.pchembl_value IS NOT NULL OR act.standard_value IS NOT NULL);
        """
        
        logging.info("Fetching data from the SQLite database...")
        return pl.read_database_uri(query=query, uri=uri, engine="adbc")

class DataProcessor:
    def __init__(self, config: Config):
        self.config = config

    def process_data(self, df: pl.DataFrame, n_value: int = None, seed: int = 42) -> pl.DataFrame:
        # 1. Sampling (optional)
        if n_value and n_value < df.height:
            df = df.sample(n=n_value, seed=seed)

        # 2. Transformations (Imputation -> Computations)
        df = DataMisc.impute_units(df)
        df = DataMisc.compute_pIC50(df)

        # 3. Final cleaning
        df_clean = (
            df.filter(
                pl.col("pIC50").is_not_null() & 
                pl.col("pIC50").is_infinite().not_()
            )
            .unique(subset=["canonical_smiles"]) # Remove duplicate structures
        )

        print(f"Processed records: {df_clean.shape[0]}")
        return df_clean
