import logging
import os
import tarfile
from pathlib import Path

import httpx
import polars as pl
from attr import dataclass
from tqdm import tqdm


@dataclass
class Config:
    data_path: Path = Path(__file__).resolve().parent / "raw"
    download_url: str = "https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/chembl_36_sqlite.tar.gz"
    archive_name: str = "chembl_36_sqlite.tar.gz"
    internal_db_path: str = "chembl_36/chembl_36_sqlite/chembl_36.db"


class DataDownloader:
    def __init__(self, config: Config):
        self.config = config
        self.output_path = config.data_path / "chembl_36.db"

    def download_sqlite_archive(self) -> None:
        if os.path.exists(self.config.archive_name):
            logging.info("Archiwum już znajduje się na dysku.")
            return

        logging.info(
            f"Downloading ChEMBL SQLite archive from: {self.config.download_url}"
        )

        with httpx.Client(http2=True, timeout=None, follow_redirects=True) as client:
            with client.stream("GET", self.config.download_url) as r:
                r.raise_for_status()
                total_size = int(r.headers.get("content-length", 0))

                with open(self.config.archive_name, "wb") as f:
                    with tqdm(
                        total=total_size, unit="B", unit_scale=True, desc="Downloading"
                    ) as pbar:
                        for chunk in r.iter_bytes(chunk_size=131072):
                            f.write(chunk)
                            pbar.update(len(chunk))

        logging.info("Download completed.")

    def extract(self) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        logging.info("Database extraction started.")

        try:
            with tarfile.open(self.config.archive_name, "r:gz") as tar:
                member = tar.getmember(self.config.internal_db_path)
                with tqdm(
                    total=member.size, unit="B", unit_scale=True, desc="Extracting"
                ) as pbar:
                    source = tar.extractfile(member)
                    if source:
                        with open(self.output_path, "wb") as target:
                            while True:
                                chunk = source.read(128 * 1024)
                                if not chunk:
                                    break
                                target.write(chunk)
                                pbar.update(len(chunk))

            logging.info(f"Database saved to: {self.output_path}")
            self._cleanup()
        except KeyError:
            logging.error(f"Could not find {self.config.internal_db_path} in archive.")
        except Exception as e:
            logging.error(f"Error during extraction: {e}")
            raise

    def _cleanup(self) -> None:
        if os.path.exists(self.config.archive_name):
            os.remove(self.config.archive_name)
            logging.info("Temporary archive file removed.")

    def download_and_extract(self) -> None:
        self.download_sqlite_archive()
        self.extract()


class DataMisc:
    @staticmethod
    def compute_pIC50(df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(
            pl.when(
                (pl.col("standard_units") == "nM")
                & pl.col("standard_value").is_not_null()
            )
            .then(-(pl.col("standard_value") * 1e-9).log10())
            .otherwise(pl.col("pchembl_value"))
            .alias("pIC50")
        )

    @staticmethod
    def impute_units(df: pl.DataFrame) -> pl.DataFrame:
        mask_missing = (
            pl.col("standard_units").is_null() & pl.col("standard_value").is_not_null()
        )
        mask_range = (pl.col("standard_value") >= 0.01) & (
            pl.col("standard_value") <= 1e6
        )

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
        db_path = self.config.data_path / "chembl_36.db"
        uri = f"sqlite:///{db_path.absolute()}"

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
            WHERE td.chembl_id IN ('CHEMBL220', 'CHEMBL4822', 'CHEMBL3177')
                AND cs.canonical_smiles IS NOT NULL
                AND act.pchembl_value IS NOT NULL
                AND (act.potential_duplicate IS NULL OR act.potential_duplicate = 0)
            ;
        """

        logging.info("Fetching data from the SQLite database...")
        return pl.read_database_uri(query=query, uri=uri, engine="adbc")


class DataProcessor:
    def __init__(self, config: Config):
        self.config = config

    def process_data(
        self, df: pl.DataFrame, n_value: int = None, seed: int = 42
    ) -> pl.DataFrame:
        if n_value and n_value < df.height:
            df = df.sample(n=n_value, seed=seed)

        df = DataMisc.impute_units(df)
        df = DataMisc.compute_pIC50(df)

        df_clean = df.filter(
            pl.col("pIC50").is_not_null() & pl.col("pIC50").is_infinite().not_()
        ).unique(subset=["canonical_smiles"])

        print(f"Processed records: {df_clean.shape[0]}")
        return df_clean
