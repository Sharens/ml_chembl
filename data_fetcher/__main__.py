"""CLI entrypoint for the data_fetcher package.

Usage:
  python -m data_fetcher [download|export|all]
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from data_fetcher.config import CONFIG

logger = logging.getLogger(__name__)

# Public API when imported from other scripts
__all__ = ["main", "run_download", "run_export", "get_parser", "_setup_logging"]


def _setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="# %(levelname)s - %(message)s",
    )


def get_parser() -> argparse.ArgumentParser:
    """Return an ArgumentParser configured like the CLI uses.

    This lets other scripts import the parser, extend it, or reuse the
    same arguments when embedding the CLI.
    """
    parser = argparse.ArgumentParser(description="Data fetcher orchestration CLI")
    parser.add_argument(
        "action",
        nargs="?",
        choices=["download", "export", "all"],
        default="all",
        help="Action to perform: download | export | all (default=all)",
    )
    parser.add_argument("--limit", type=int, default=None, help="Row limit for export (overrides config)")
    return parser


def run_download() -> None:
    from data_fetcher.download.chembl_db_downloader import ChEMBLDownloader

    dl = ChEMBLDownloader(
        CONFIG.download_url,
        CONFIG.archive_name,
        CONFIG.internal_db_path,
        CONFIG.db_path,
    )
    logger.info("Running download_and_extract()")
    dl.download_and_extract()


def run_export(limit: int | None = None) -> None:
    from data_fetcher.database.connection import DatabaseConnection
    from data_fetcher.database.repository import TableRepository
    from data_fetcher.export.parquet_exporter import ParquetExporter

    db_conn = DatabaseConnection(CONFIG.db_path)
    repo = TableRepository(db_conn)
    exporter = ParquetExporter(CONFIG.raw_data_path)

    table_names = repo.get_table_names()
    logger.info(f"Found {len(table_names)} tables to export")

    row_limit = CONFIG.row_limit if limit is None else limit
    exporter.export_multiple_tables(repo, table_names, row_limit)


def main(argv: list[str] | None = None) -> int:
    _setup_logging()

    parser = get_parser()
    args = parser.parse_args(argv)

    try:
        if args.action == "download":
            run_download()
        elif args.action == "export":
            run_export(limit=args.limit)
        else:
            # all
            run_download()
            run_export(limit=args.limit)

        logger.info("Action completed successfully")
        return 0
    except Exception as exc:  # noqa: BLE001 (keep top-level handler)
        logger.exception("Fatal error during execution: %s", exc)
        return 2


if __name__ == "__main__":
    sys.exit(main())
