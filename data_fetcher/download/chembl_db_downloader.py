import httpx
import tarfile
import os
import logging
from pathlib import Path
from tqdm import tqdm
from data_fetcher.config import CONFIG

logger = logging.getLogger()

class ChEMBLDownloader:
    """Downloader class for fetching and extracting the ChEMBL database."""
    
    def __init__(self, url: str, archive_name: str, internal_path: str, output_path: Path):
        self.url = url
        self.archive_name = archive_name
        self.internal_path = internal_path
        self.output_path = Path(output_path)
    
    def download_sqlite_archive(self) -> None:
        """Downloads the ChEMBL SQLite archive if it doesn't already exist."""
        if os.path.exists(self.archive_name):
            logger.info("Archiwum już znajduje się na dysku.")
            return
        
        logger.info(f"Downloading ChEMBL SQLite archive from: {self.url}")
        
        with httpx.Client(http2=True, timeout=None, follow_redirects=True) as client:
            with client.stream("GET", self.url) as r:
                r.raise_for_status()
                total_size = int(r.headers.get('content-length', 0))
                
                with open(self.archive_name, 'wb') as f:
                    with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading") as pbar:
                        for chunk in r.iter_bytes(chunk_size=131072):
                            f.write(chunk)
                            pbar.update(len(chunk))
        
        logger.info("Download completed.")
    
    def extract(self) -> None:
        """Extracts the SQLite database from the downloaded archive."""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info("Database extraction started.")
        
        try:
            with tarfile.open(self.archive_name, "r:gz") as tar:
                with tqdm(total=tar.getmember(self.internal_path).size, unit='B', unit_scale=True, desc="Extracting") as pbar:
                    for member in tar.getmembers():
                        if member.name == self.internal_path:
                            tar.extract(member, path=self.output_path.parent)
                            break
                        pbar.update(member.size)
            
            logger.info(f"Database saved to: {self.output_path}")
            self._cleanup()
        except KeyError:
            logger.error(f"Could not find {self.internal_path} in archive.")
        except Exception as e:
            logger.error(f"Error during extraction: {e}")
            raise
    
    def _cleanup(self) -> None:
        """Removes the downloaded archive to save space."""
        if os.path.exists(self.archive_name):
            os.remove(self.archive_name)
            logger.info("Temporary archive file removed.")
    
    def download_and_extract(self) -> None:
        """Downloads and extracts the database."""
        self.download_sqlite_archive()
        self.extract()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    downloader = ChEMBLDownloader(
        CONFIG.download_url,
        CONFIG.archive_name,
        CONFIG.internal_db_path,
        CONFIG.db_path
    )
    downloader.download_and_extract()