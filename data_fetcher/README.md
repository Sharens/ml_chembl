Data Fetcher — README
======================

What this service does
----------------------
- Downloads the ChEMBL SQLite archive, extracts the database, and exports selected tables to Parquet files.
- Intended to produce ready-to-use Parquet files in the `data_fetcher/raw` directory.

How it works (short)
--------------------
- `download` module fetches and extracts the SQLite DB.
- `database` layer provides a small repository to read tables.
- `export` module converts tables to Parquet using Polars.
- An entrypoint (`python -m data_fetcher`) orchestrates these steps.

Prerequisites
-------------
- Python 3.9+ (match your virtualenv)
- A virtual environment with project dependencies installed (see `requirements.txt`).

Quick start (recommended)
-------------------------
1. Activate the project's virtual environment:

```bash
source .venv/bin/activate
```
1. Run download + export:
```bash
python -m data_fetcher
```

1. Run only download (download + extract):

```bash
python -m data_fetcher download
```

4. Run only export (reads existing SQLite and writes Parquet). You can set a row limit:

```bash
python -m data_fetcher export
python -m data_fetcher export --limit 5000
```

Where outputs go
----------------
- Parquet files are written to the directory defined by `CONFIG.raw_data_path` (default: `data_fetcher/raw`).
- Files are named like `raw_<table_name>.parquet`.
