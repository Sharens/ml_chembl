from data_fetcher.config import CONFIG
from data_fetcher.database.connection import DatabaseConnection
from data_fetcher.database.repository import TableRepository
from pathlib import Path
from pathlib import Path
from typing import Set
import logging
import logging
import polars as pl

logger = logging.getLogger(__name__)

class ParquetExporter:
    """Data exporter class for writing Polars DataFrames to Parquet files."""
    
    def __init__(self, output_path: Path):
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
    
    def export_dataframe(self, df: pl.DataFrame, table_name: str) -> Path:
        """Exports a Polars DataFrame to a Parquet file."""
        output_file = self.output_path / f"raw_{table_name}.parquet"
        
        df.write_parquet(str(output_file), compression="zstd")
        logger.info(f"Data from table {table_name} saved into {output_file}")
        
        return output_file
    
    def export_multiple_tables(self, repo, table_names: Set[str], row_limit: int) -> None:
        """Exports multiple tables from a repository."""
        for table_name in table_names:
            try:
                df = repo.fetch_table_as_dataframe(table_name, row_limit)
                self.export_dataframe(df, table_name)
            except Exception as e:
                logger.error(f"Error during export of table {table_name}: {e}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    db_conn = DatabaseConnection(CONFIG.db_path)
    repo = TableRepository(db_conn)
    exporter = ParquetExporter(CONFIG.raw_data_path)

    logging.info("Starting export of all tables to Parquet...")
    table_names = repo.get_table_names()
    logging.info(f"Found {len(table_names)} tables to export")

    exporter.export_multiple_tables(repo, table_names, CONFIG.row_limit)
    logging.info("Export finished")