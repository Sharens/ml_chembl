import polars as pl
import sqlite3

DB_PATH = "libs/chembl_36/chembl_36_sqlite/chembl_36.db"


query = """
SELECT
    act.activity_id,
    act.assay_id,
    act.doc_id,
    act.record_id,
    act.molregno,
    act.standard_relation,
    act.standard_value,
    act.standard_units,
    act.standard_flag,
    act.standard_type,
    -- act.activity_comment, -- NOT NEEDED
    -- act.data_validity_comment, -- NOT NEEDED
    act.potential_duplicate,
    act.pchembl_value,
    act.bao_endpoint,
    act.uo_units,
    act.qudt_units,
    act.toid,
    act.upper_value,
    act.standard_upper_value,
    act.src_id,
    act.type,
    act.relation,
    act.value,
    act.units,
    act.text_value,
    act.standard_text_value
    -- CAST(act.action_type AS STRING) AS action_type -- Wywołuje błąd
FROM activities act
JOIN assays ass ON act.assay_id = ass.assay_id
JOIN target_dictionary trg ON ass.tid = trg.tid
WHERE trg.organism = 'Homo sapiens'
    AND act.standard_value IS NOT NULL
LIMIT 2000000 -- hard limit dla mojego WSLa
"""

polars_schema = {
    "ACTIVITY_ID": pl.Int64,
    "ASSAY_ID": pl.Int64,
    "DOC_ID": pl.Int64,
    "RECORD_ID": pl.Int64,
    "MOLREGNO": pl.Int64,
    "SRC_ID": pl.Int64,
    "TOID": pl.Int64,
    "STANDARD_RELATION": pl.String,
    "STANDARD_VALUE": pl.Float64,
    "STANDARD_UPPER_VALUE": pl.Float64,
    "STANDARD_UNITS": pl.String,
    "STANDARD_FLAG": pl.Int8,
    "STANDARD_TYPE": pl.String,
    "PCHEMBL_VALUE": pl.Float64,
    "RELATION": pl.String,
    "VALUE": pl.Float64,
    "UPPER_VALUE": pl.Float64,
    "UNITS": pl.String,
    "TYPE": pl.String,
    "TEXT_VALUE": pl.String,
    "STANDARD_TEXT_VALUE": pl.String,
    # "ACTIVITY_COMMENT": pl.String,
    # "DATA_VALIDITY_COMMENT": pl.String,
    "POTENTIAL_DUPLICATE": pl.Int8,
    # "ACTION_TYPE": pl.String,
    "BAO_ENDPOINT": pl.String,
    "UO_UNITS": pl.String,
    "QUDT_UNITS": pl.String
}

def extract_chembl_from_sqlite():
    print("DEBUG: Nawiązywanie połączenia z bazą SQLite...")
    conn = sqlite3.connect(DB_PATH)

    print("DEBUG: Wykonywanie zapytania i ładowanie do Polars (to może chwilę potrwać)...")


    df = pl.read_database(query, connection=conn, schema_overrides=polars_schema)

    conn.close()
    return df

if __name__ == "__main__":

    df = extract_chembl_from_sqlite()

    print(f"Pobrano {df.height} rekordów.")
    print(df.head())
    df.write_parquet("chembl_human_activities.parquet", compression="brotli")