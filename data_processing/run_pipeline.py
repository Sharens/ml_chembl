import logging
from pathlib import Path
from components import Config, DataLoader, DataProcessor

# Konfiguracja logowania (wyświetla czas i status w konsoli)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

config = Config()

loader = DataLoader(config)
raw_df = loader.load_from_sqlite()
    
processor = DataProcessor(config)
    
final_df = processor.process_data(raw_df)

output_dir = Path("processed_data")
output_dir.mkdir(exist_ok=True)

output_path = output_dir / "ChEMBL_processed.parquet"
final_df.write_parquet(output_path, compression="zstd")

logging.info(f"ETL succeeded")
logging.info(f"Saved {final_df.height} records into: {output_path}")
