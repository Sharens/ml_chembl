import logging
import os
from libs import Config, DataLoader, DataProcessor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logging.info("Starting ETL process.")

# 1. Configuration
config = Config()

# 2. Load data
logging.info("Loading data...")
try:
    activities, assays, target_dictionary, compound_structures, compound_properties = DataLoader(config).load_data()
    logging.info("Data loaded successfully.")
except Exception as e:
    logging.error(f"Error during data loading: {e}")
    exit(1) # Exit with an error code

# 3. Process data
logging.info("Processing data...")
try:
    processed_df = DataProcessor(config).process_data(
        activities,
        assays,
        target_dictionary,
        compound_structures,
        compound_properties
    )
    logging.info("Data processed successfully.")
    logging.info(f"Number of rows in the resulting dataset: {len(processed_df)}")
except Exception as e:
    logging.error(f"Error during data processing: {e}")
    exit(1) # Exit with an error code

# 4. Display results (optional)
logging.info("Displaying example processed data:")
print(processed_df.head())

# Save processed data to a file
output_directory = "processed_data"
output_path = os.path.join(output_directory, "processed_data.parquet")

os.makedirs(output_directory, exist_ok=True)
processed_df.write_parquet(output_path)
logging.info(f"Processed data saved to: {output_path}")

logging.info("ETL process completed successfully.")
