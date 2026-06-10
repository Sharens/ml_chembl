from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any

import mlflow

from src._config import MLFLOW_DB_URI, MLRUNS

DEFAULT_TRACKING_URI = MLFLOW_DB_URI
DEFAULT_EXPERIMENT_NAME = "ml_chembl_baselines"


def _normalize_tracking_uri(tracking_uri: str) -> str:
    if tracking_uri.startswith("sqlite:///") and not tracking_uri.startswith(
        "sqlite:////"
    ):
        relative_path = tracking_uri.removeprefix("sqlite:///")
        absolute_path = Path(relative_path).resolve()
        return f"sqlite:////{absolute_path.as_posix()}"
    return tracking_uri


def configure_mlflow(
    tracking_uri: str = DEFAULT_TRACKING_URI,
    experiment_name: str = DEFAULT_EXPERIMENT_NAME,
    artifact_root: str | Path = MLRUNS,
) -> str:
    artifact_path = Path(artifact_root)
    artifact_path.mkdir(parents=True, exist_ok=True)

    tracking_uri = _normalize_tracking_uri(tracking_uri)
    mlflow.set_tracking_uri(tracking_uri)

    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(
            name=experiment_name,
            artifact_location=artifact_path.resolve().as_uri(),
        )
    else:
        experiment_id = experiment.experiment_id

    mlflow.set_experiment(experiment_name)
    return str(experiment_id)


@contextmanager
def start_training_run(
    *,
    run_name: str,
    model_type: str,
    split_type: str,
    seed: int,
    extra_tags: dict[str, str] | None = None,
):
    with mlflow.start_run(run_name=run_name):
        tags = {
            "project": "ml_chembl",
            "task": "bioactivity_regression",
            "target": "pIC50",
            "model_type": model_type,
            "split_type": split_type,
            "seed": str(seed),
        }
        if extra_tags:
            tags.update(extra_tags)
        mlflow.set_tags(tags)
        yield


def log_split_sizes(train_size: int, val_size: int, test_size: int) -> None:
    mlflow.log_metrics(
        {
            "n_train": float(train_size),
            "n_val": float(val_size),
            "n_test": float(test_size),
        }
    )


def log_epoch_metrics(history: list[dict[str, Any]]) -> None:
    for row in history:
        step = int(row["epoch"])
        for key in ("train_loss", "val_loss", "val_r2", "lr"):
            value = row.get(key)
            if value is not None:
                mlflow.log_metric(key, float(value), step=step)


def log_final_metrics(
    *,
    best_val_loss: float,
    r2_val: float | None = None,
    rmse_val: float | None = None,
    mae_val: float | None = None,
    r2_test: float | None = None,
    rmse_test: float | None = None,
    mae_test: float | None = None,
) -> None:
    mlflow.log_metric("best_val_loss", float(best_val_loss))
    if r2_val is not None:
        mlflow.log_metric("r2_val", float(r2_val))
    if rmse_val is not None:
        mlflow.log_metric("rmse_val", float(rmse_val))
    if mae_val is not None:
        mlflow.log_metric("mae_val", float(mae_val))
    if r2_test is not None:
        mlflow.log_metric("r2_test", float(r2_test))
    if rmse_test is not None:
        mlflow.log_metric("rmse_test", float(rmse_test))
    if mae_test is not None:
        mlflow.log_metric("mae_test", float(mae_test))
