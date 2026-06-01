from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any

import mlflow


DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"
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
    artifact_root: str | Path = "mlruns",
) -> str:
    """Configure local MLflow backend and return experiment ID."""
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
    """Start MLflow run with project-level tags."""
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
    """Log per-epoch metrics from a list of dicts.

    Each row can contain keys like: epoch, train_loss, val_loss, val_r2, lr.
    """
    for row in history:
        step = int(row["epoch"])
        for key in ("train_loss", "val_loss", "val_r2", "lr"):
            value = row.get(key)
            if value is not None:
                mlflow.log_metric(key, float(value), step=step)


def log_final_metrics(
    *, best_val_loss: float, r2_val: float, r2_test: float | None = None
) -> None:
    mlflow.log_metric("best_val_loss", float(best_val_loss))
    mlflow.log_metric("r2_val", float(r2_val))
    if r2_test is not None:
        mlflow.log_metric("r2_test", float(r2_test))
