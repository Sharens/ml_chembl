from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, log10, lit, round as spark_round
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml import Pipeline

def create_spark_session():
    # Konfiguracja pod Twój klaster z pliku DockerCompose.yml
    return SparkSession.builder \
        .appName("ChEMBL_ML_Preprocessing") \
        .master("spark://spark-master:7077") \
        .config("spark.driver.memory", "2g") \
        .config("spark.executor.memory", "1g") \
        .getOrCreate()

def process_chembl_data(input_path, output_path):
    spark = create_spark_session()
    
    print(f"Reading data from {input_path}...")
    # 1. Wczytanie danych (Parquet jest natywny dla Sparka)
    df = spark.read.parquet(input_path)

    # 2. Wstępne czyszczenie (odpowiednik dropna i filtrów z load_and_clean_data)
    # Filtrujemy tylko rekordy z wartościami aktywności i prawidłowymi jednostkami
    df_clean = df.filter(
        (col("standard_value").isNotNull()) & 
        (col("canonical_smiles").isNotNull()) &
        (col("standard_type") == "IC50") &  # Skupiamy się na IC50 jak w notebooku
        (col("standard_units") == "nM")     # Dla uproszczenia (imputacja w Sparku jest możliwa, ale tu filtrujemy)
    )

    # 3. Obliczenie pIC50
    # Wzór: pIC50 = 9 - log10(IC50_nM)
    # Odpowiednik funkcji compute_pIC50 z Twojego notebooka
    df_calculated = df_clean.withColumn(
        "pIC50_raw", 
        9 - log10(col("standard_value"))
    ).withColumn(
        "pIC50", 
        spark_round(col("pIC50_raw"), 2) # Zaokrąglanie jak w Twoim kodzie Polars
    )

    # 4. Feature Engineering (SMILES -> Fingerprints)
    # UWAGA: Spark standardowo nie ma RDKit. Tutaj przygotowujemy dane tabelaryczne.
    # W środowisku produkcyjnym użylibyśmy UDF z RDKit, ale wymaga to instalacji na workerach.
    # Tutaj skupimy się na przygotowaniu targetu (y) i cech numerycznych.

    # 5. Normalizacja (StandardScaler)
    # Spark MLlib wymaga kolumny typu 'Vector'
    assembler = VectorAssembler(
        inputCols=["pIC50"], # Tutaj normalizujemy target, ale normalnie normalizowalibyśmy input features
        outputCol="pIC50_vec"
    )
    
    scaler = StandardScaler(
        inputCol="pIC50_vec",
        outputCol="pIC50_scaled",
        withStd=True,
        withMean=True
    )

    # Budowa Pipeline'u
    pipeline = Pipeline(stages=[assembler, scaler])
    model = pipeline.fit(df_calculated)
    df_transformed = model.transform(df_calculated)

    # 6. Zapis przetworzonych danych
    # Wybieramy kluczowe kolumny do treningu ML
    final_cols = ["activity_id", "canonical_smiles", "pIC50", "pIC50_scaled"]
    
    print(f"Saving processed data to {output_path}...")
    df_transformed.select(final_cols).write.mode("overwrite").parquet(output_path)
    
    print("Done!")
    spark.stop()

if __name__ == "__main__":
    # Ścieżki muszą być widoczne wewnątrz kontenera Docker
    INPUT_FILE = "/opt/workspace/libs/datasets/chembl_selected_ds.parquet"
    OUTPUT_DIR = "/opt/workspace/libs/datasets/chembl_spark_processed.parquet"
    
    process_chembl_data(INPUT_FILE, OUTPUT_DIR)