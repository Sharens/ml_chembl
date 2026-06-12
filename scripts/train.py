#!/usr/bin/env python3
"""CLI do trenowania modeli MLP/GNN i tuningu hiperparametrow.

Uzycie:
  # pojedynczy trening
  python scripts/train.py --model MLP --split random

  # trening z deskryptorami i MACCS
  python scripts/train.py --model MLP --split scaffold --use-descriptors --use-maccs --batch-norm

  # tuning GNN z Optuna (wznawialny)
  python scripts/train.py tune --split scaffold --n-trials 50

  # pokaz podsumowanie studium
  python scripts/train.py study-summary gnn_scaffold

  # kampania z pliku YAML (wiele treningow)
  python scripts/train.py campaign experiments/campaign.yaml
"""

from __future__ import annotations

import argparse
import sys

import polars as pl

from src._config import PROCESSED_DATA
from src.models.config import MLP_DESCRIPTOR_COLS as _DESC_COLS
from src.models.pipeline import run_parallel_trainings, train_and_score
from src.models.tuning import study_summary, tune_gnn_optuna


def _load_data() -> pl.DataFrame:
    path = PROCESSED_DATA / "ChEMBL_processed.parquet"
    if not path.exists():
        print(f"ERROR: {path} not found. Run the data pipeline first.")
        sys.exit(1)
    df = pl.read_parquet(path)
    return df.with_columns(
        pl.col("canonical_smiles")
        .map_elements(
            lambda s: __import__(
                "src.models.data", fromlist=["fp_from_smiles"]
            ).fp_from_smiles(s),
            return_dtype=pl.Object,
        )
        .alias("fp")
    ).filter(
        pl.col("fp").is_not_null()
        & pl.col("pIC50").is_not_null()
        & pl.col("pIC50").is_not_nan()
    )


def cmd_train(args):
    df = _load_data()
    spec = {
        "model_type": args.model,
        "split_type": args.split,
        "df_fp": df,
        "epochs": args.epochs,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "log_mlflow": not args.no_mlflow,
        "evaluate_test": args.evaluate_test,
        "prefer_cuda": not args.cpu,
        "mlp_descriptor_cols": _DESC_COLS if args.use_descriptors else None,
        "mlp_use_maccs": args.use_maccs,
        "mlp_use_batch_norm": args.batch_norm,
    }
    result = train_and_score(**spec)
    print(f"\nResult: R²={result['r2_val']:.4f}  RMSE={result['rmse_val']:.4f}")


def cmd_train_all(args):
    df = _load_data()
    splits = args.splits or ["random", "scaffold"]
    models = args.models or ["MLP", "GNN"]
    specs = []
    for model_type in models:
        for split in splits:
            spec = {
                "model_type": model_type,
                "split_type": split,
                "df_fp": df,
                "epochs": args.epochs,
                "seed": args.seed,
                "batch_size": args.batch_size,
                "log_mlflow": not args.no_mlflow,
                "evaluate_test": args.evaluate_test,
                "prefer_cuda": not args.cpu,
            }
            if model_type == "MLP":
                spec["mlp_descriptor_cols"] = (
                    _DESC_COLS if args.use_descriptors else None
                )
                spec["mlp_use_maccs"] = args.use_maccs
                spec["mlp_use_batch_norm"] = args.batch_norm
            specs.append(spec)
    results = run_parallel_trainings(specs, max_workers=args.workers)
    print(f"\nDone: {len(results)} trainings")


def cmd_tune(args):
    df = _load_data()
    search_space = {
        "lr": {"type": "float", "low": 1e-4, "high": 1e-3, "log": True},
        "weight_decay": {"type": "categorical", "values": [1e-5, 5e-5, 1e-4]},
        "pooling": {"type": "categorical", "values": ["mean", "add", "attention"]},
        "gnn_hidden_dim": {"type": "int", "low": 128, "high": 320, "step": 32},
        "gnn_num_layers": {"type": "int", "low": 3, "high": 6},
        "gnn_dropout": {"type": "float", "low": 0.0, "high": 0.3},
    }
    study = tune_gnn_optuna(
        split_type=args.split,
        df_fp=df,
        search_space=search_space,
        n_trials=args.n_trials,
        seeds=args.seeds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        log_mlflow=not args.no_mlflow,
        prefer_cuda=not args.cpu,
        study_name=args.study_name or f"gnn_{args.split}",
    )
    print(f"\nBest R²: {study.best_trial.value:.4f}")
    print(f"Best params: {study.best_trial.user_attrs['params']}")


def cmd_study_summary(args):
    df = study_summary(args.study_name)
    print(df)


def cmd_campaign(args):
    import yaml

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    df = _load_data()

    if "train" in cfg:
        for spec in cfg["train"]:
            spec["df_fp"] = df
            train_and_score(**spec)

    if "tune" in cfg:
        for tune_cfg in cfg["tune"]:
            tune_gnn_optuna(
                split_type=tune_cfg["split"],
                df_fp=df,
                search_space=tune_cfg["search_space"],
                n_trials=tune_cfg.get("n_trials", 30),
                seeds=tune_cfg.get("seeds", [42, 123]),
                epochs=tune_cfg.get("epochs", 200),
            )

    print("Campaign complete.")


def _add_common_args(p):
    p.add_argument("--cpu", action="store_true", help="Force CPU")
    p.add_argument("--no-mlflow", action="store_true", help="Skip MLflow logging")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=2)


def main():
    parser = argparse.ArgumentParser(description="Train ML models for pIC50 prediction")

    sub = parser.add_subparsers(dest="command", required=True)

    # train
    p_train = sub.add_parser("train", help="Single training run")
    _add_common_args(p_train)
    p_train.add_argument("--model", required=True, choices=["MLP", "GNN"])
    p_train.add_argument(
        "--split", required=True, choices=["random", "scaffold", "family"]
    )
    p_train.add_argument("--evaluate-test", action="store_true")
    p_train.add_argument(
        "--use-descriptors", action="store_true", help="Add molecular descriptors"
    )
    p_train.add_argument(
        "--use-maccs", action="store_true", help="Add MACCS fingerprints"
    )
    p_train.add_argument(
        "--batch-norm", action="store_true", help="Use BatchNorm in MLP"
    )
    p_train.set_defaults(func=cmd_train)

    # train-all
    p_all = sub.add_parser("train-all", help="Train all model/split combinations")
    _add_common_args(p_all)
    p_all.add_argument("--splits", nargs="+", choices=["random", "scaffold", "family"])
    p_all.add_argument("--models", nargs="+", choices=["MLP", "GNN"])
    p_all.add_argument("--evaluate-test", action="store_true")
    p_all.add_argument("--use-descriptors", action="store_true")
    p_all.add_argument("--use-maccs", action="store_true")
    p_all.add_argument("--batch-norm", action="store_true")
    p_all.set_defaults(func=cmd_train_all)

    # tune
    p_tune = sub.add_parser("tune", help="Optuna hyperparameter tuning (resumable)")
    _add_common_args(p_tune)
    p_tune.add_argument("--split", required=True, choices=["random", "scaffold"])
    p_tune.add_argument("--n-trials", type=int, default=30)
    p_tune.add_argument("--seeds", type=int, nargs="+", default=[42, 123])
    p_tune.add_argument("--study-name")
    p_tune.set_defaults(func=cmd_tune)

    # study-summary
    p_ss = sub.add_parser("study-summary", help="Show Optuna study results")
    _add_common_args(p_ss)
    p_ss.add_argument("study_name")
    p_ss.set_defaults(func=cmd_study_summary)

    # campaign
    p_camp = sub.add_parser("campaign", help="Run training campaign from YAML")
    _add_common_args(p_camp)
    p_camp.add_argument("config")
    p_camp.set_defaults(func=cmd_campaign)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
