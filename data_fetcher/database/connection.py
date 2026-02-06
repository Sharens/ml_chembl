import sqlite3
from contextlib import contextmanager
from pathlib import Path

class DatabaseConnection:
    """SQLite database connection manager."""
    
    def __init__(self, db_path: Path):
        self.db_path = db_path
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(str(self.db_path))
        try:
            yield conn
        finally:
            conn.close()
    
    @contextmanager
    def get_cursor(self):
        """Context manager for cursors."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            try:
                yield cursor
            finally:
                cursor.close()