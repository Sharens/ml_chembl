from __future__ import annotations

import json

import numpy as np
import optuna
import polars as pl

from src._config import PROCESSED_DATA
from src.models.config import (
    BATCH_SIZE_DEFAULT,
    EARLY_STOPPING_PATIENCE_DEFAULT,
    EPOCHS_DEFAULT,
    LR_DEFAULT,
    MIN_DELTA_DEFAULT,
    POOLING_DEFAULT,
    WEIGHT_DECAY_DEFAULT,
)
from src.models.pipeline import train_and_score

OPTUNA_STUDIES_DIR = PROCESSED_DATA / "optuna_studies"


def _gnn_params(trial, search_space: dict) -> dict:
    params = {}
    for key, config in search_space.items():
        typ = config.get("type", "categorical")
        if typ == "categorical":
            params[key] = trial.suggest_categorical(key, config["values"])
        elif typ == "float":
            params[key] = trial.suggest_float(
                key, config["low"], config["high"], log=config.get("log", False)
            )
        elif typ == "int":
            params[key] = trial.suggest_int(key, config["low"], config["high"])
    return params


def tune_gnn_optuna(
    split_type: str,
    df_fp: pl.DataFrame,
    search_space: dict,
    n_trials: int = 30,
    seeds: list[int] | None = None,
    epochs: int = EPOCHS_DEFAULT,
    batch_size: int = BATCH_SIZE_DEFAULT,
    early_stopping_patience: int = EARLY_STOPPING_PATIENCE_DEFAULT,
    min_delta: float = MIN_DELTA_DEFAULT,
    log_mlflow: bool = False,
    prefer_cuda: bool = True,
    study_name: str | None = None,
    storage: str | None = None,
    load_if_exists: bool = True,
) -> optuna.Study:
    OPTUNA_STUDIES_DIR.mkdir(parents=True, exist_ok=True)

    if seeds is None:
        seeds = [42, 123]

    if study_name is None:
        study_name = f"gnn_{split_type}"

    if storage is None:
        storage = f"sqlite:///{OPTUNA_STUDIES_DIR / f'{study_name}.db'}"

    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5, n_warmup_steps=10, interval_steps=1
    )
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        load_if_exists=load_if_exists,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=pruner,
    )

    print(f"Study '{study_name}' ({storage}) — existing trials: {len(study.trials)}")

    for trial_num in range(len(study.trials), n_trials):
        trial = study.ask()
        params = _gnn_params(trial, search_space)
        trial.set_user_attr("split_type", split_type)
        trial.set_user_attr("seeds", seeds)

        per_seed_r2 = []
        for seed in seeds:
            res = train_and_score(
                model_type="GNN",
                split_type=split_type,
                df_fp=df_fp,
                epochs=epochs,
                lr=params.get("lr", LR_DEFAULT),
                batch_size=batch_size,
                seed=seed,
                log_mlflow=log_mlflow,
                replace_existing=False,
                evaluate_test=False,
                early_stopping_patience=early_stopping_patience,
                min_delta=min_delta,
                weight_decay=params.get("weight_decay", WEIGHT_DECAY_DEFAULT),
                pooling=params.get("pooling", POOLING_DEFAULT),
                gnn_hidden_dim=params.get("gnn_hidden_dim", 128),
                gnn_num_layers=params.get("gnn_num_layers", 4),
                gnn_dropout=params.get("gnn_dropout", 0.15),
                prefer_cuda=prefer_cuda,
                deterministic=False,
                use_amp=True,
                use_model_cache=True,
                force_retrain=False,
                cache_namespace=f"optuna_{split_type}",
            )
            per_seed_r2.append(res["r2_val"])

            trial.report(res["r2_val"], seed)
            if trial.should_prune():
                print(
                    f"  Trial {trial_num + 1}/{n_trials} PRUNED at seed={seed} "
                    f"(R²={res['r2_val']:.3f})"
                )
                break

        mean_r2 = float(np.mean(per_seed_r2))
        std_r2 = float(np.std(per_seed_r2)) if len(per_seed_r2) > 1 else 0.0
        trial.set_user_attr("r2_val_std", std_r2)
        trial.set_user_attr("params", params)
        study.tell(trial, mean_r2)

        print(
            f"  Trial {trial_num + 1}/{n_trials} | "
            f"R²={mean_r2:.4f} ± {std_r2:.4f} | "
            f"params={params}"
        )

    print(f"\nBest trial ({study.best_trial.number}):")
    print(f"  R² = {study.best_trial.value:.4f}")
    print(f"  params = {study.best_trial.user_attrs['params']}")
    return study


def best_params_from_study(study_name: str, split_type: str) -> dict:
    storage = f"sqlite:///{OPTUNA_STUDIES_DIR / f'{study_name}.db'}"
    study = optuna.load_study(study_name=study_name, storage=storage)
    return study.best_trial.user_attrs["params"]


def study_summary(study_name: str) -> pl.DataFrame:
    storage = f"sqlite:///{OPTUNA_STUDIES_DIR / f'{study_name}.db'}"
    study = optuna.load_study(study_name=study_name, storage=storage)
    rows = []
    for t in study.trials:
        if t.state == optuna.trial.TrialState.COMPLETE:
            rows.append(
                {
                    "trial": t.number,
                    "r2_val_mean": t.value,
                    "r2_val_std": t.user_attrs.get("r2_val_std", 0.0),
                    "params": json.dumps(t.user_attrs.get("params", {})),
                    "state": str(t.state),
                }
            )
    df = pl.DataFrame(rows).sort("r2_val_mean", descending=True)
    return df
