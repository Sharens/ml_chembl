from pathlib import Path
from typing import Set
import polars as pl
from .connection import DatabaseConnection

class TableRepository:
    """Repository class for fetching table names and data from the SQLite database."""
    
    def __init__(self, db_connection: DatabaseConnection):
        self.db_connection = db_connection
    
    def get_table_names(self) -> Set[str]:
        """Fetches the names of all tables in the database."""
        query = "SELECT * FROM sqlite_master WHERE type='table';"
        
        with self.db_connection.get_cursor() as cursor:
            cursor.execute(query)
            return set(row[1] for row in cursor.fetchall())
    
    def fetch_table_as_dataframe(self, table_name: str, limit: int) -> pl.DataFrame:
        """Fetches a table as a Polars DataFrame"""
        query = f"SELECT * FROM {table_name} LIMIT {limit};"
        
        with self.db_connection.get_connection() as conn:
            df = pl.read_database(
                connection=conn,
                infer_schema_length=limit,
                query=query
            )
        return df