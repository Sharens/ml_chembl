from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DATA = ROOT / "processed_data"
MODEL_CACHE = PROCESSED_DATA / "model_cache"
GRAPH_CACHE = PROCESSED_DATA / "graph_cache"
MLFLOW_DB_URI = "sqlite:///" + str((ROOT / "mlflow.db").as_posix())
MLRUNS = ROOT / "mlruns"
