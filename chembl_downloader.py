import polars as pl
import sqlite3
import tomli

with open("config.toml", "rb") as f:
    config = tomli.load(f)
    DB_PATH = config['sqldb']['path']


def get_query(human: bool, chemblid: list[str] = None, limit: int = 200000) -> tuple[str, list]:
    """
    Generuje bezpieczne zapytanie SQL i listę parametrów dla bazy ChEMBL.
    """
    where_clauses = ["act.standard_value IS NOT NULL"]
    limit_query = ''
    params = []
    if human:
        where_clauses.append("trg.organism = 'Homo sapiens'")

    if chemblid:
        placeholders = ", ".join(["?"] * len(chemblid))
        where_clauses.append(f"trg.chembl_id IN ({placeholders})")
        params.extend(chemblid)
    if limit:
        limit_query = f"LIMIT {limit}"

    where_string = " AND ".join(where_clauses)

    query = f"""
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
        act.standard_text_value,
        cs.canonical_smiles,
        cp.mw_freebase,
        cp.alogp,
        cp.hbd,
        cp.hba,
        cp.psa,
        cp.heavy_atoms,
        a.confidence_score
    FROM activities act
    JOIN assays ass ON act.assay_id = ass.assay_id
    JOIN target_dictionary trg ON ass.tid = trg.tid
    JOIN compound_structures cs ON act.molregno = cs.molregno
    JOIN compound_properties cp ON act.molregno = cp.molregno
    JOIN assays a ON act.assay_id = a.assay_id
    WHERE {where_string}
    {limit_query}
    """
    return query, params

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
    "CANONICAL_SMILES": pl.String,
    "MW_FREEBASE": pl.Float64,
    "ALOGP": pl.Float64,
    "HBD": pl.Int64,
    "HBA": pl.Int64,
    "PSA": pl.Float64,
    "HEAVY_ATOMS": pl.Int64,
    "CONFIDENCE_SCORE": pl.Int64,
    # "ACTIVITY_COMMENT": pl.String,
    # "DATA_VALIDITY_COMMENT": pl.String,
    "POTENTIAL_DUPLICATE": pl.Int8,
    # "ACTION_TYPE": pl.String,
    "BAO_ENDPOINT": pl.String,
    "UO_UNITS": pl.String,
    "QUDT_UNITS": pl.String
}


def extract_chembl_from_sqlite(human: bool, chemblid: list[str] = None, limit: int = 20000):
    print("DEBUG: Nawiązywanie połączenia z bazą SQLite...")
    conn = sqlite3.connect(DB_PATH)

    print("DEBUG: Wykonywanie zapytania i ładowanie do Polars (to może chwilę potrwać)...")
    query, params = get_query(human, chemblid, limit)
    df = pl.read_database(query, connection=conn, execute_options={"parameters": params}, schema_overrides=polars_schema)

    conn.close()
    return df


if __name__ == "__main__":
    df = extract_chembl_from_sqlite(False,None)
    filename = "libs/datasets/chembl_activities.parquet"
    print(f"Pobrano {df.height} rekordów do {filename}.")
    print(df.head())
    df.write_parquet(filename, compression="brotli")

    df = extract_chembl_from_sqlite(True,None)
    filename = "libs/datasets/chembl_human_activities.parquet"
    print(f"Pobrano {df.height} rekordów do {filename}.")
    print(df.head())
    df.write_parquet(filename, compression="brotli")

    df = extract_chembl_from_sqlite(True,['CHEMBL203'])
    filename = "libs/datasets/chembl_human_CHEMBL203_activities.parquet"
    print(f"Pobrano {df.height} rekordów do {filename}.")
    print(df.head())
    df.write_parquet(filename, compression="brotli")


