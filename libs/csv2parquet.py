import polars as pl

csv_path = 'libs/datasets/raw_lim_export.csv'

chembl_schema = {
    "activity_id": pl.UInt32,          # ID aktywności (liczba całkowita)
    "molregno": pl.UInt32,             # ID cząsteczki (liczba całkowita)
    "canonical_smiles": pl.String,     # Notacja SMILES (tekst)
    "mw_freebase": pl.Float64,         # Masa cząsteczkowa (wysoka precyzja)
    "alogp": pl.Float32,               # LogP (pojedyncza precyzja wystarczy)
    "hba": pl.UInt8,                   # Akceptory wiązań wodorowych (mała liczba całkowita)
    "hbd": pl.UInt8,                   # Donory wiązań wodorowych (mała liczba całkowita)
    "psa": pl.Float32,                 # Powierzchnia polarna (pojedyncza precyzja)
    "rtb": pl.UInt8,                   # Wiązania rotacyjne (mała liczba całkowita)
    "aromatic_rings": pl.UInt8,        # Pierścienie aromatyczne (mała liczba całkowita)
    "qed_weighted": pl.Float32,        # QED (pojedyncza precyzja)
    "standard_value": pl.Float32,
    "standard_units": pl.String,
    "standard_type": pl.String,
    "pchembl_value": pl.Float32,       # Wartość pChEMBL (pojedyncza precyzja)
    "target_chembl_id": pl.String,     # ID celu (tekst)
    "target_name": pl.String           # Nazwa celu (tekst)
}


df = pl.read_csv(
    csv_path,
    separator=';',
    schema=chembl_schema
)


print(f"Pomyślnie wczytano plik: {csv_path}")
print(df.head())

parquet_path = 'libs/datasets/chembl_selected_ds.parquet'

df.write_parquet(parquet_path, compression='brotli')

print(f"\nPlik został pomyślnie zapisany jako: {parquet_path}")