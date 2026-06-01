from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import mlflow
import numpy as np
import polars as pl
import torch

from src._config import MODEL_CACHE
from src.mlflow_utils import (
    configure_mlflow,
    log_epoch_metrics,
    log_final_metrics,
    log_split_sizes,
    start_training_run,
)
from src.models.config import (
    BATCH_SIZE_DEFAULT,
    EARLY_STOPPING_PATIENCE_DEFAULT,
    EPOCHS_DEFAULT,
    LR_DEFAULT,
    MIN_DELTA_DEFAULT,
    POOLING_DEFAULT,
    SEED_DEFAULT,
    WEIGHT_DECAY_DEFAULT,
)
from src.models.data import (
    build_gnn_loaders,
    build_mlp_loaders,
)
from src.models.models import GNNRegressor, MLPBaseline
from src.models.training import (
    evaluate_loss,
    evaluate_r2,
    get_device,
    seed_everything,
    train_one_epoch,
)

# ---------------------------------------------------------------------------
# Globalny rejestr wynikow
# ---------------------------------------------------------------------------
results_table: list[dict] = []


# ---------------------------------------------------------------------------
# Cache pomocniczy
# ---------------------------------------------------------------------------
def _make_cache_key(config: dict) -> str:
    payload = json.dumps(config, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def get_model_cache_path(config: dict, namespace: str = "default") -> Path:
    key = _make_cache_key(config)
    return MODEL_CACHE / f"{namespace}_{key}.pt"


def _upsert_result(
    result: dict,
    replace_existing: bool,
    model_type: str,
    split_type: str,
    seed: int,
    pooling: str,
    gnn_hidden_dim: int,
    gnn_num_layers: int,
    gnn_dropout: float,
    lr: float,
    weight_decay: float,
):
    if replace_existing:
        results_table[:] = [
            r
            for r in results_table
            if not (
                r["model"] == model_type
                and r["split"] == split_type
                and r.get("seed") == seed
                and r.get("pooling") == pooling
                and r.get("gnn_hidden_dim") == gnn_hidden_dim
                and r.get("gnn_num_layers") == gnn_num_layers
                and r.get("gnn_dropout") == gnn_dropout
                and r.get("lr") == lr
                and r.get("weight_decay") == weight_decay
            )
        ]
    results_table.append(result)


# ---------------------------------------------------------------------------
# Glowna funkcja treningowa
# ---------------------------------------------------------------------------


def train_and_score(
    model_type: str,
    split_type: str,
    df_fp: pl.DataFrame | None = None,
    epochs: int = EPOCHS_DEFAULT,
    lr: float = LR_DEFAULT,
    batch_size: int = BATCH_SIZE_DEFAULT,
    seed: int = SEED_DEFAULT,
    log_mlflow: bool = False,
    replace_existing: bool = True,
    evaluate_test: bool = False,
    early_stopping_patience: int = EARLY_STOPPING_PATIENCE_DEFAULT,
    min_delta: float = MIN_DELTA_DEFAULT,
    weight_decay: float = WEIGHT_DECAY_DEFAULT,
    pooling: str = POOLING_DEFAULT,
    gnn_hidden_dim: int = 128,
    gnn_num_layers: int = 4,
    gnn_dropout: float = 0.15,
    prefer_cuda: bool = True,
    deterministic: bool = False,
    use_amp: bool = True,
    use_model_cache: bool = True,
    force_retrain: bool = False,
    cache_namespace: str = "default",
    mlflow_tracking_uri: str = "sqlite:///mlflow.db",
    mlflow_experiment_name: str = "ml_chembl_baselines",
    mlflow_artifact_root: str = "mlruns",
):
    seed_everything(seed, deterministic=deterministic)
    device = get_device(prefer_cuda=prefer_cuda)
    amp_enabled = bool(use_amp and device.type == "cuda")
    scaler = torch.amp.GradScaler(device="cuda", enabled=amp_enabled)

    if log_mlflow:
        configure_mlflow(
            tracking_uri=mlflow_tracking_uri,
            experiment_name=mlflow_experiment_name,
            artifact_root=mlflow_artifact_root,
        )

    if model_type == "MLP":
        train_loader, val_loader, test_loader = build_mlp_loaders(
            df_fp, split_type=split_type, batch_size=batch_size, seed=seed
        )
        model = MLPBaseline().to(device)
        is_gnn_flag = False
    elif model_type == "GNN":
        train_loader, val_loader, test_loader = build_gnn_loaders(
            df_fp, split_type=split_type, batch_size=batch_size, seed=seed
        )
        sample_graph = train_loader.dataset[0]
        model = GNNRegressor(
            node_features=sample_graph.num_node_features,
            edge_features=sample_graph.edge_attr.shape[1],
            hidden_dim=gnn_hidden_dim,
            num_layers=gnn_num_layers,
            dropout=gnn_dropout,
            pooling=pooling,
        ).to(device)
        is_gnn_flag = True
    else:
        raise ValueError("model_type must be 'MLP' or 'GNN'")

    cache_config = {
        "model_type": model_type,
        "split_type": split_type,
        "seed": seed,
        "epochs": epochs,
        "lr": lr,
        "batch_size": batch_size,
        "early_stopping_patience": early_stopping_patience,
        "min_delta": min_delta,
        "weight_decay": weight_decay,
        "pooling": pooling,
        "gnn_hidden_dim": gnn_hidden_dim,
        "gnn_num_layers": gnn_num_layers,
        "gnn_dropout": gnn_dropout,
    }
    model_cache_path = get_model_cache_path(cache_config, namespace=cache_namespace)

    if use_model_cache and model_cache_path.exists() and not force_retrain:
        try:
            cached = torch.load(
                model_cache_path, map_location=device, weights_only=False
            )
            model.load_state_dict(cached["model_state_dict"])

            r2_val = evaluate_r2(
                model, val_loader, device, is_gnn=is_gnn_flag, use_amp=amp_enabled
            )
            r2_test = (
                evaluate_r2(
                    model,
                    test_loader,
                    device,
                    is_gnn=is_gnn_flag,
                    use_amp=amp_enabled,
                )
                if evaluate_test
                else None
            )

            result = cached.get("result", {}).copy()
            result.update(
                {
                    "model": model_type,
                    "split": split_type,
                    "seed": seed,
                    "r2_val": r2_val,
                    "r2_test": r2_test,
                    "device": str(device),
                    "amp_enabled": amp_enabled,
                    "from_cache": True,
                    "cache_path": str(model_cache_path),
                }
            )

            if log_mlflow:
                _log_to_mlflow(
                    model=model,
                    result=result,
                    model_type=model_type,
                    split_type=split_type,
                    seed=seed,
                    epochs=epochs,
                    lr=lr,
                    batch_size=batch_size,
                    evaluate_test=evaluate_test,
                    early_stopping_patience=early_stopping_patience,
                    min_delta=min_delta,
                    weight_decay=weight_decay,
                    pooling=pooling,
                    gnn_hidden_dim=gnn_hidden_dim,
                    gnn_num_layers=gnn_num_layers,
                    gnn_dropout=gnn_dropout,
                    device=device,
                    amp_enabled=amp_enabled,
                    deterministic=deterministic,
                    cache_namespace=cache_namespace,
                    use_model_cache=use_model_cache,
                    force_retrain=force_retrain,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    test_loader=test_loader,
                    from_cache=True,
                )

            _upsert_result(
                result=result,
                replace_existing=replace_existing,
                model_type=model_type,
                split_type=split_type,
                seed=seed,
                pooling=pooling,
                gnn_hidden_dim=gnn_hidden_dim,
                gnn_num_layers=gnn_num_layers,
                gnn_dropout=gnn_dropout,
                lr=lr,
                weight_decay=weight_decay,
            )
            print(f"Loaded cached model: {model_cache_path.name} | val R2={r2_val:.3f}")
            return result
        except Exception as exc:
            print(f"Model cache load failed ({exc}); training from scratch...")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=max(2, early_stopping_patience // 3),
        min_lr=1e-6,
    )
    criterion = torch.nn.MSELoss()

    train_losses = []
    val_losses = []
    val_r2_history = []
    lr_history = []
    best_val_loss = float("inf")
    best_epoch = -1
    best_state = None
    no_improve_epochs = 0

    for epoch in range(epochs):
        epoch_train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            is_gnn=is_gnn_flag,
            clip_grad=1.0,
            scaler=scaler,
            use_amp=amp_enabled,
        )
        epoch_val_loss = evaluate_loss(
            model,
            val_loader,
            criterion,
            device,
            is_gnn=is_gnn_flag,
            use_amp=amp_enabled,
        )
        epoch_val_r2 = evaluate_r2(
            model, val_loader, device, is_gnn=is_gnn_flag, use_amp=amp_enabled
        )
        current_lr = optimizer.param_groups[0]["lr"]

        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)
        val_r2_history.append(epoch_val_r2)
        lr_history.append(current_lr)

        scheduler.step(epoch_val_loss)

        if epoch_val_loss < (best_val_loss - min_delta):
            best_val_loss = epoch_val_loss
            best_epoch = epoch + 1
            best_state = deepcopy(model.state_dict())
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1

        if no_improve_epochs >= early_stopping_patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    avg_train_loss = float(np.mean(train_losses))
    avg_val_loss = float(np.mean(val_losses))

    r2_val = evaluate_r2(
        model, val_loader, device, is_gnn=is_gnn_flag, use_amp=amp_enabled
    )
    r2_test = (
        evaluate_r2(model, test_loader, device, is_gnn=is_gnn_flag, use_amp=amp_enabled)
        if evaluate_test
        else None
    )

    if log_mlflow:
        _log_to_mlflow(
            model=model,
            result=None,
            model_type=model_type,
            split_type=split_type,
            seed=seed,
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            evaluate_test=evaluate_test,
            early_stopping_patience=early_stopping_patience,
            min_delta=min_delta,
            weight_decay=weight_decay,
            pooling=pooling,
            gnn_hidden_dim=gnn_hidden_dim,
            gnn_num_layers=gnn_num_layers,
            gnn_dropout=gnn_dropout,
            device=device,
            amp_enabled=amp_enabled,
            deterministic=deterministic,
            cache_namespace=cache_namespace,
            use_model_cache=use_model_cache,
            force_retrain=force_retrain,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            from_cache=False,
            train_losses=train_losses,
            val_losses=val_losses,
            val_r2_history=val_r2_history,
            lr_history=lr_history,
            best_val_loss=best_val_loss,
            best_epoch=best_epoch,
            avg_train_loss=avg_train_loss,
            avg_val_loss=avg_val_loss,
            r2_val=r2_val,
            r2_test=r2_test,
        )

    result = {
        "model": model_type,
        "split": split_type,
        "seed": seed,
        "epochs": epochs,
        "epochs_trained": len(train_losses),
        "best_epoch": best_epoch,
        "lr": lr,
        "batch_size": batch_size,
        "weight_decay": weight_decay,
        "pooling": pooling,
        "gnn_hidden_dim": gnn_hidden_dim,
        "gnn_num_layers": gnn_num_layers,
        "gnn_dropout": gnn_dropout,
        "avg_train_loss": avg_train_loss,
        "avg_val_loss": avg_val_loss,
        "best_val_loss": float(best_val_loss),
        "r2_val": r2_val,
        "r2_test": r2_test,
        "device": str(device),
        "amp_enabled": amp_enabled,
        "from_cache": False,
        "cache_path": str(model_cache_path),
    }

    if use_model_cache:
        try:
            torch.save(
                {"model_state_dict": model.state_dict(), "result": result},
                model_cache_path,
            )
        except Exception as exc:
            print(f"Warning: could not save model cache ({exc})")

    _upsert_result(
        result=result,
        replace_existing=replace_existing,
        model_type=model_type,
        split_type=split_type,
        seed=seed,
        pooling=pooling,
        gnn_hidden_dim=gnn_hidden_dim,
        gnn_num_layers=gnn_num_layers,
        gnn_dropout=gnn_dropout,
        lr=lr,
        weight_decay=weight_decay,
    )

    if r2_test is None:
        print(
            f"{model_type} | {split_type} | device={device}: "
            f"avg train loss={avg_train_loss:.4f}, avg val loss={avg_val_loss:.4f}, "
            f"best val loss={best_val_loss:.4f}, best epoch={best_epoch}, "
            f"val R2={r2_val:.3f}"
        )
    else:
        print(
            f"{model_type} | {split_type} | device={device}: "
            f"avg train loss={avg_train_loss:.4f}, avg val loss={avg_val_loss:.4f}, "
            f"best val loss={best_val_loss:.4f}, best epoch={best_epoch}, "
            f"val R2={r2_val:.3f}, test R2={r2_test:.3f}"
        )
    return result


# ---------------------------------------------------------------------------
# Wewnetrzna funkcja do logowania MLflow
# ---------------------------------------------------------------------------


def _log_to_mlflow(
    *,
    model,
    result,
    model_type,
    split_type,
    seed,
    epochs,
    lr,
    batch_size,
    evaluate_test,
    early_stopping_patience,
    min_delta,
    weight_decay,
    pooling,
    gnn_hidden_dim,
    gnn_num_layers,
    gnn_dropout,
    device,
    amp_enabled,
    deterministic,
    cache_namespace,
    use_model_cache,
    force_retrain,
    train_loader,
    val_loader,
    test_loader,
    from_cache,
    train_losses=None,
    val_losses=None,
    val_r2_history=None,
    lr_history=None,
    best_val_loss=None,
    best_epoch=None,
    avg_train_loss=None,
    avg_val_loss=None,
    r2_val=None,
    r2_test=None,
):
    run_suffix = "_cache" if from_cache else ""
    with start_training_run(
        run_name=f"{model_type}_{split_type}_seed{seed}{run_suffix}",
        model_type=model_type,
        split_type=split_type,
        seed=seed,
        extra_tags={
            "framework": "pytorch",
            "split_strategy": split_type,
            "from_cache": str(from_cache).lower(),
        },
    ):
        params = {
            "model_type": model_type,
            "split_type": split_type,
            "epochs": epochs,
            "lr": lr,
            "batch_size": batch_size,
            "seed": seed,
            "evaluate_test": evaluate_test,
            "early_stopping_patience": early_stopping_patience,
            "min_delta": min_delta,
            "weight_decay": weight_decay,
            "pooling": pooling,
            "gnn_hidden_dim": gnn_hidden_dim,
            "gnn_num_layers": gnn_num_layers,
            "gnn_dropout": gnn_dropout,
            "device": str(device),
            "amp_enabled": amp_enabled,
            "deterministic": deterministic,
            "cache_namespace": cache_namespace,
            "used_model_cache": use_model_cache,
            "force_retrain": force_retrain,
            "from_cache": from_cache,
        }

        if from_cache and result is not None:
            params["epochs_trained"] = result.get("epochs_trained")
            params["best_epoch"] = result.get("best_epoch")

        mlflow.log_params(params)
        log_split_sizes(
            len(train_loader.dataset),
            len(val_loader.dataset),
            len(test_loader.dataset),
        )

        if from_cache and result is not None:
            cached_avg_train_loss = result.get("avg_train_loss")
            cached_avg_val_loss = result.get("avg_val_loss")
            cached_best_val_loss = result.get("best_val_loss")
            if cached_avg_train_loss is not None:
                mlflow.log_metric("avg_train_loss", float(cached_avg_train_loss))
            if cached_avg_val_loss is not None:
                mlflow.log_metric("avg_val_loss", float(cached_avg_val_loss))
            if cached_best_val_loss is not None:
                log_final_metrics(
                    best_val_loss=float(cached_best_val_loss),
                    r2_val=r2_val,
                    r2_test=r2_test,
                )
            else:
                mlflow.log_metric("r2_val", float(r2_val))
                if r2_test is not None:
                    mlflow.log_metric("r2_test", float(r2_test))
        else:
            mlflow.log_metric("avg_train_loss", float(avg_train_loss))
            mlflow.log_metric("avg_val_loss", float(avg_val_loss))
            log_final_metrics(
                best_val_loss=float(best_val_loss),
                r2_val=r2_val,
                r2_test=r2_test,
            )
            epoch_history = [
                {
                    "epoch": i,
                    "train_loss": tr,
                    "val_loss": va,
                    "val_r2": va_r2,
                    "lr": lri,
                }
                for i, (tr, va, va_r2, lri) in enumerate(
                    zip(train_losses, val_losses, val_r2_history, lr_history),
                    start=1,
                )
            ]
            log_epoch_metrics(epoch_history)
            mlflow.pytorch.log_model(model, "model")


# ---------------------------------------------------------------------------
# Grid search dla GNN
# ---------------------------------------------------------------------------


def tune_gnn(
    split_type: str,
    search_space: dict,
    seeds: list[int],
    df_fp: pl.DataFrame | None = None,
    epochs: int = 50,
    batch_size: int = BATCH_SIZE_DEFAULT,
    early_stopping_patience: int = EARLY_STOPPING_PATIENCE_DEFAULT,
    min_delta: float = MIN_DELTA_DEFAULT,
    log_mlflow: bool = False,
    prefer_cuda: bool = True,
):
    import itertools

    import pandas as pd

    keys = [
        "lr",
        "weight_decay",
        "pooling",
        "gnn_hidden_dim",
        "gnn_num_layers",
        "gnn_dropout",
    ]
    combos = list(itertools.product(*(search_space[k] for k in keys)))

    tuning_rows = []
    for i, (lr, wd, pooling, hidden, layers, dropout) in enumerate(combos, start=1):
        print(
            f"[{split_type}] config {i}/{len(combos)} | "
            f"lr={lr} wd={wd} pool={pooling} "
            f"hidden={hidden} layers={layers} dropout={dropout}"
        )
        per_seed = []
        for seed in seeds:
            res = train_and_score(
                model_type="GNN",
                split_type=split_type,
                df_fp=df_fp,
                epochs=epochs,
                lr=lr,
                batch_size=batch_size,
                seed=seed,
                log_mlflow=log_mlflow,
                replace_existing=False,
                evaluate_test=False,
                early_stopping_patience=early_stopping_patience,
                min_delta=min_delta,
                weight_decay=wd,
                pooling=pooling,
                gnn_hidden_dim=hidden,
                gnn_num_layers=layers,
                gnn_dropout=dropout,
                prefer_cuda=prefer_cuda,
                deterministic=False,
                use_amp=True,
                use_model_cache=True,
                force_retrain=False,
                cache_namespace="gnn_tuning",
            )
            per_seed.append(res)

        val_r2 = [r["r2_val"] for r in per_seed]
        val_loss = [r["best_val_loss"] for r in per_seed]
        tuning_rows.append(
            {
                "model": "GNN",
                "split": split_type,
                "lr": lr,
                "weight_decay": wd,
                "pooling": pooling,
                "gnn_hidden_dim": hidden,
                "gnn_num_layers": layers,
                "gnn_dropout": dropout,
                "seeds": ",".join(str(s) for s in seeds),
                "epochs_selection": epochs,
                "r2_val_mean": float(np.mean(val_r2)),
                "r2_val_std": float(np.std(val_r2)),
                "best_val_loss_mean": float(np.mean(val_loss)),
                "best_val_loss_std": float(np.std(val_loss)),
            }
        )

    df_tuning = (
        pd.DataFrame(tuning_rows)
        .sort_values(["r2_val_mean", "best_val_loss_mean"], ascending=[False, True])
        .reset_index(drop=True)
    )
    return df_tuning
