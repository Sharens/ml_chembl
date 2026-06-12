from __future__ import annotations

import concurrent.futures
import hashlib
import io
import json
import multiprocessing
import pickle
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
    LOSS_FN_DEFAULT,
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
    evaluate_all_metrics,
    evaluate_loss,
    evaluate_r2,
    get_device,
    seed_everything,
    train_one_epoch,
)

# ---------------------------------------------------------------------------
# Pomocnicza serializacja DataFrame z kolumna Object (fp) dla workerow
# ---------------------------------------------------------------------------


def _serialize_df(df: pl.DataFrame) -> tuple[bytes, bytes]:
    fp_col = df.select("fp").to_series().to_list()
    fp_bytes = pickle.dumps(fp_col)
    buf = io.BytesIO()
    df.drop("fp").write_parquet(buf)
    return buf.getvalue(), fp_bytes


def _deserialize_df(df_bytes: bytes, fp_bytes: bytes) -> pl.DataFrame:
    df = pl.read_parquet(io.BytesIO(df_bytes))
    fp_series = pl.Series("fp", pickle.loads(fp_bytes))
    return df.with_columns(fp_series)


def _get_loss_fn(name: str) -> torch.nn.Module:
    if name == "huber":
        return torch.nn.HuberLoss(delta=1.0)
    elif name == "mse":
        return torch.nn.MSELoss()
    elif name == "mae":
        return torch.nn.L1Loss()
    raise ValueError(f"Unknown loss function: {name}")


# ---------------------------------------------------------------------------
# Worker functions dla zrownoleglenia (module-level = picklable)
# ---------------------------------------------------------------------------


def _tune_config_worker(args: tuple) -> dict:
    (
        lr,
        wd,
        pooling,
        hidden,
        layers,
        dropout,
        seeds,
        df_bytes,
        fp_bytes,
        common_kwargs,
    ) = args

    import torch.utils.data
    from src.models.training import compute_num_workers

    compute_num_workers.__defaults__ = (0,)

    df_fp = _deserialize_df(df_bytes, fp_bytes)

    per_seed = []
    for seed in seeds:
        res = train_and_score(
            model_type="GNN",
            split_type=common_kwargs["split_type"],
            df_fp=df_fp,
            epochs=common_kwargs["epochs"],
            lr=lr,
            batch_size=common_kwargs["batch_size"],
            seed=seed,
            log_mlflow=False,
            replace_existing=False,
            evaluate_test=False,
            early_stopping_patience=common_kwargs["early_stopping_patience"],
            min_delta=common_kwargs["min_delta"],
            weight_decay=wd,
            pooling=pooling,
            gnn_hidden_dim=hidden,
            gnn_num_layers=layers,
            gnn_dropout=dropout,
            prefer_cuda=common_kwargs["prefer_cuda"],
            deterministic=False,
            use_amp=True,
            use_model_cache=True,
            force_retrain=False,
            cache_namespace="gnn_tuning",
        )
        per_seed.append(res)

    val_r2 = [r["r2_val"] for r in per_seed]
    val_rmse = [r["rmse_val"] for r in per_seed]
    val_loss = [r["best_val_loss"] for r in per_seed]

    return {
        "lr": lr,
        "weight_decay": wd,
        "pooling": pooling,
        "gnn_hidden_dim": hidden,
        "gnn_num_layers": layers,
        "gnn_dropout": dropout,
        "per_seed_results": per_seed,
        "r2_val_mean": float(np.mean(val_r2)),
        "r2_val_std": float(np.std(val_r2)),
        "rmse_val_mean": float(np.mean(val_rmse)),
        "rmse_val_std": float(np.std(val_rmse)),
        "best_val_loss_mean": float(np.mean(val_loss)),
        "best_val_loss_std": float(np.std(val_loss)),
    }


def _single_train_worker(args: tuple) -> dict:
    kwargs_bytes, df_bytes, fp_bytes = args

    kwargs = pickle.loads(kwargs_bytes)
    kwargs["df_fp"] = _deserialize_df(df_bytes, fp_bytes)
    kwargs["log_mlflow"] = False

    return train_and_score(**kwargs)


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
    loss_fn: str = LOSS_FN_DEFAULT,
    prefer_cuda: bool = True,
    deterministic: bool = False,
    use_amp: bool = True,
    use_model_cache: bool = True,
    force_retrain: bool = False,
    cache_namespace: str = "default",
    mlflow_tracking_uri: str = "sqlite:///mlflow.db",
    mlflow_experiment_name: str = "ml_chembl_baselines",
    mlflow_artifact_root: str = "mlruns",
    mlp_descriptor_cols: list[str] | None = None,
    mlp_use_maccs: bool = False,
    mlp_use_batch_norm: bool = False,
    mlp_hidden_sizes: list[int] | None = None,
    mlp_dropout: float = 0.2,
    scheduler_type: str = "plateau",
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
            df_fp,
            split_type=split_type,
            batch_size=batch_size,
            seed=seed,
            descriptor_cols=mlp_descriptor_cols,
            use_maccs=mlp_use_maccs,
        )
        sample = next(iter(train_loader))[0]
        input_size = sample.shape[1]
        model = MLPBaseline(
            input_size=input_size,
            use_batch_norm=mlp_use_batch_norm,
            hidden_sizes=mlp_hidden_sizes,
            dropout=mlp_dropout,
        ).to(device)
        is_gnn_flag = False
    elif model_type == "GNN":
        train_loader, val_loader, test_loader = build_gnn_loaders(
            df_fp,
            split_type=split_type,
            batch_size=batch_size,
            seed=seed,
            descriptor_cols=mlp_descriptor_cols,
        )
        sample_graph = train_loader.dataset[0]
        descriptor_dim = (
            len(mlp_descriptor_cols) if mlp_descriptor_cols is not None else 0
        )
        model = GNNRegressor(
            node_features=sample_graph.num_node_features,
            edge_features=sample_graph.edge_attr.shape[1],
            hidden_dim=gnn_hidden_dim,
            num_layers=gnn_num_layers,
            dropout=gnn_dropout,
            pooling=pooling,
            descriptor_dim=descriptor_dim,
            use_jk=True,
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
        "loss_fn": loss_fn,
        "mlp_descriptor_cols": mlp_descriptor_cols,
        "mlp_use_maccs": mlp_use_maccs,
        "mlp_use_batch_norm": mlp_use_batch_norm,
        "mlp_hidden_sizes": mlp_hidden_sizes,
        "mlp_dropout": mlp_dropout,
        "scheduler_type": scheduler_type,
        "gnn_descriptor_dim": len(mlp_descriptor_cols)
        if mlp_descriptor_cols is not None
        else 0,
        "gnn_use_jk": True,
    }
    model_cache_path = get_model_cache_path(cache_config, namespace=cache_namespace)

    if use_model_cache and model_cache_path.exists() and not force_retrain:
        try:
            cached = torch.load(
                model_cache_path, map_location=device, weights_only=False
            )
            model.load_state_dict(cached["model_state_dict"])

            results = evaluate_all_metrics(
                model, val_loader, device, is_gnn=is_gnn_flag, use_amp=amp_enabled
            )
            r2_val = results["r2"]
            rmse_val = results["rmse"]
            mae_val = results["mae"]

            test_results = {}
            if evaluate_test:
                test_results = evaluate_all_metrics(
                    model,
                    test_loader,
                    device,
                    is_gnn=is_gnn_flag,
                    use_amp=amp_enabled,
                )

            result = cached.get("result", {}).copy()
            result.update(
                {
                    "model": model_type,
                    "split": split_type,
                    "seed": seed,
                    "r2_val": r2_val,
                    "rmse_val": rmse_val,
                    "mae_val": mae_val,
                    "r2_test": test_results.get("r2"),
                    "rmse_test": test_results.get("rmse"),
                    "mae_test": test_results.get("mae"),
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
                    loss_fn=loss_fn,
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
            print(
                f"Loaded cached model: {model_cache_path.name} | "
                f"val R²={r2_val:.3f} RMSE={rmse_val:.3f} MAE={mae_val:.3f}"
            )
            return result
        except Exception as exc:
            print(f"Model cache load failed ({exc}); training from scratch...")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=1e-6
        )
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=6,
            min_lr=1e-6,
        )
    criterion = _get_loss_fn(loss_fn)

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

        if scheduler_type == "cosine":
            scheduler.step()
        else:
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

    results = evaluate_all_metrics(
        model, val_loader, device, is_gnn=is_gnn_flag, use_amp=amp_enabled
    )
    r2_val = results["r2"]
    rmse_val = results["rmse"]
    mae_val = results["mae"]

    test_results = {}
    if evaluate_test:
        test_results = evaluate_all_metrics(
            model, test_loader, device, is_gnn=is_gnn_flag, use_amp=amp_enabled
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
            loss_fn=loss_fn,
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
            rmse_val=rmse_val,
            mae_val=mae_val,
            r2_test=test_results.get("r2"),
            rmse_test=test_results.get("rmse"),
            mae_test=test_results.get("mae"),
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
        "loss_fn": loss_fn,
        "mlp_descriptor_cols": mlp_descriptor_cols,
        "mlp_use_maccs": mlp_use_maccs,
        "mlp_use_batch_norm": mlp_use_batch_norm,
        "mlp_hidden_sizes": mlp_hidden_sizes,
        "mlp_dropout": mlp_dropout,
        "scheduler_type": scheduler_type,
        "avg_train_loss": avg_train_loss,
        "avg_val_loss": avg_val_loss,
        "best_val_loss": float(best_val_loss),
        "r2_val": r2_val,
        "rmse_val": rmse_val,
        "mae_val": mae_val,
        "r2_test": test_results.get("r2"),
        "rmse_test": test_results.get("rmse"),
        "mae_test": test_results.get("mae"),
        "device": str(device),
        "amp_enabled": amp_enabled,
        "from_cache": False,
        "cache_path": str(model_cache_path),
    }

    if use_model_cache:
        try:
            model_cache_path.parent.mkdir(parents=True, exist_ok=True)
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

    test_suffix = ""
    if evaluate_test:
        test_suffix = (
            f", test R²={test_results.get('r2', 0):.3f} "
            f"RMSE={test_results.get('rmse', 0):.3f}"
        )
    print(
        f"{model_type} | {split_type} | device={device}: "
        f"avg train loss={avg_train_loss:.4f}, avg val loss={avg_val_loss:.4f}, "
        f"best val loss={best_val_loss:.4f}, best epoch={best_epoch}, "
        f"val R²={r2_val:.3f} RMSE={rmse_val:.3f} MAE={mae_val:.3f}"
        f"{test_suffix}"
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
    loss_fn,
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
    rmse_val=None,
    mae_val=None,
    r2_test=None,
    rmse_test=None,
    mae_test=None,
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
            "loss_fn": loss_fn,
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
            _log_cached_metrics(
                result, r2_val, rmse_val, mae_val, r2_test, rmse_test, mae_test
            )
        else:
            mlflow.log_metric("avg_train_loss", float(avg_train_loss))
            mlflow.log_metric("avg_val_loss", float(avg_val_loss))
            log_final_metrics(
                best_val_loss=float(best_val_loss),
                r2_val=r2_val,
                rmse_val=rmse_val,
                mae_val=mae_val,
                r2_test=r2_test,
                rmse_test=rmse_test,
                mae_test=mae_test,
            )
            epoch_history = [
                {
                    "epoch": i,
                    "train_loss": tr,
                    "val_loss": va,
                    "val_r2": vr2,
                    "lr": lri,
                }
                for i, (tr, va, vr2, lri) in enumerate(
                    zip(train_losses, val_losses, val_r2_history, lr_history),
                    start=1,
                )
            ]
            log_epoch_metrics(epoch_history)
            mlflow.pytorch.log_model(model, name="model")


def _log_cached_metrics(
    result, r2_val, rmse_val, mae_val, r2_test, rmse_test, mae_test
):
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
            rmse_val=rmse_val,
            mae_val=mae_val,
            r2_test=r2_test,
            rmse_test=rmse_test,
            mae_test=mae_test,
        )
    else:
        mlflow.log_metric("r2_val", float(r2_val))
        mlflow.log_metric("rmse_val", float(rmse_val))
        mlflow.log_metric("mae_val", float(mae_val))
        if r2_test is not None:
            mlflow.log_metric("r2_test", float(r2_test))
        if rmse_test is not None:
            mlflow.log_metric("rmse_test", float(rmse_test))


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
    max_workers: int = 1,
):
    import itertools

    if df_fp is None:
        raise ValueError("df_fp must be provided")

    keys = [
        "lr",
        "weight_decay",
        "pooling",
        "gnn_hidden_dim",
        "gnn_num_layers",
        "gnn_dropout",
    ]
    combos = list(itertools.product(*(search_space[k] for k in keys)))

    common_kwargs = {
        "split_type": split_type,
        "epochs": epochs,
        "batch_size": batch_size,
        "early_stopping_patience": early_stopping_patience,
        "min_delta": min_delta,
        "prefer_cuda": prefer_cuda,
    }

    # --- sciezka rownolegla ---
    if max_workers > 1:
        try:
            from src.models.data import load_or_build_graph_cache

            load_or_build_graph_cache(df_fp)
        except Exception as exc:
            print(f"Graph cache pre-build warning: {exc}")

        df_bytes, fp_bytes = _serialize_df(df_fp)

        worker_args = [
            (
                lr,
                wd,
                pooling,
                hidden,
                layers,
                dropout,
                seeds,
                df_bytes,
                fp_bytes,
                common_kwargs,
            )
            for lr, wd, pooling, hidden, layers, dropout in combos
        ]

        ctx = multiprocessing.get_context("spawn")
        worker_results: list[dict] = []
        with concurrent.futures.ProcessPoolExecutor(
            mp_context=ctx, max_workers=max_workers
        ) as executor:
            futures = {
                executor.submit(_tune_config_worker, arg): i
                for i, arg in enumerate(worker_args)
            }
            for future in concurrent.futures.as_completed(futures):
                idx = futures[future]
                try:
                    wr = future.result()
                    worker_results.append(wr)
                    print(
                        f"[{split_type}] config {idx + 1}/{len(combos)} done | "
                        f"R² mean={wr['r2_val_mean']:.3f} ± {wr['r2_val_std']:.3f}"
                    )
                except Exception as exc:
                    print(
                        f"[{split_type}] config {idx + 1}/{len(combos)} FAILED: {exc}"
                    )

        tuning_rows = []
        for wr in worker_results:
            tuning_rows.append(
                {
                    "model": "GNN",
                    "split": split_type,
                    "lr": wr["lr"],
                    "weight_decay": wr["weight_decay"],
                    "pooling": wr["pooling"],
                    "gnn_hidden_dim": wr["gnn_hidden_dim"],
                    "gnn_num_layers": wr["gnn_num_layers"],
                    "gnn_dropout": wr["gnn_dropout"],
                    "seeds": ",".join(str(s) for s in seeds),
                    "epochs_selection": epochs,
                    "r2_val_mean": wr["r2_val_mean"],
                    "r2_val_std": wr["r2_val_std"],
                    "rmse_val_mean": wr["rmse_val_mean"],
                    "rmse_val_std": wr["rmse_val_std"],
                    "best_val_loss_mean": wr["best_val_loss_mean"],
                    "best_val_loss_std": wr["best_val_loss_std"],
                }
            )
            for seed_res in wr["per_seed_results"]:
                _upsert_result(
                    result=seed_res,
                    replace_existing=False,
                    model_type=seed_res["model"],
                    split_type=seed_res["split"],
                    seed=seed_res["seed"],
                    pooling=seed_res.get("pooling", POOLING_DEFAULT),
                    gnn_hidden_dim=seed_res.get("gnn_hidden_dim", 128),
                    gnn_num_layers=seed_res.get("gnn_num_layers", 4),
                    gnn_dropout=seed_res.get("gnn_dropout", 0.15),
                    lr=seed_res["lr"],
                    weight_decay=seed_res["weight_decay"],
                )

        df_tuning = pl.DataFrame(tuning_rows).sort(
            ["r2_val_mean", "rmse_val_mean"], descending=[True, False]
        )

        if log_mlflow:
            _log_tuning_batch(df_tuning, split_type, search_space, seeds, epochs)

        return df_tuning

    # --- sciezka sekwencyjna (oryginalna) ---
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
        val_rmse = [r["rmse_val"] for r in per_seed]
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
                "rmse_val_mean": float(np.mean(val_rmse)),
                "rmse_val_std": float(np.std(val_rmse)),
                "best_val_loss_mean": float(np.mean(val_loss)),
                "best_val_loss_std": float(np.std(val_loss)),
            }
        )

    df_tuning = pl.DataFrame(tuning_rows).sort(
        ["r2_val_mean", "rmse_val_mean"], descending=[True, False]
    )
    return df_tuning


# ---------------------------------------------------------------------------
# Batch MLflow logging dla grid searcha
# ---------------------------------------------------------------------------


def _log_tuning_batch(
    df_tuning: pl.DataFrame,
    split_type: str,
    search_space: dict,
    seeds: list[int],
    epochs: int,
):
    import tempfile

    configure_mlflow()

    with mlflow.start_run(
        run_name=f"GNN_tune_{split_type}_{len(df_tuning)}configs",
        tags={
            "project": "ml_chembl",
            "task": "grid_search",
            "split_type": split_type,
        },
    ):
        mlflow.log_params(
            {
                "split_type": split_type,
                "n_configs": len(df_tuning),
                "n_seeds": len(seeds),
                "seeds": ",".join(str(s) for s in seeds),
                "epochs": epochs,
                "search_space": json.dumps(search_space),
            }
        )

        if not df_tuning.is_empty():
            best = df_tuning.row(0, named=True)
            mlflow.log_metrics(
                {
                    "best_r2_val_mean": float(best["r2_val_mean"]),
                    "best_r2_val_std": float(best["r2_val_std"]),
                    "best_rmse_val_mean": float(best["rmse_val_mean"]),
                    "best_val_loss_mean": float(best["best_val_loss_mean"]),
                }
            )

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            df_tuning.write_csv(f.name)
            mlflow.log_artifact(f.name, "tuning_results")


# ---------------------------------------------------------------------------
# Zrownoleglone wywolywanie wielu train_and_score
# ---------------------------------------------------------------------------


def run_parallel_trainings(
    train_specs: list[dict],
    max_workers: int = 2,
    log_mlflow: bool = True,
):
    if max_workers <= 1:
        results = {}
        for i, spec in enumerate(train_specs):
            results[i] = train_and_score(**spec)
        return results

    ctx = multiprocessing.get_context("spawn")

    worker_args = []
    for spec in train_specs:
        spec_copy = dict(spec)
        df_fp = spec_copy.pop("df_fp", None)

        if df_fp is not None:
            df_bytes, fp_bytes = _serialize_df(df_fp)
        else:
            df_bytes = b""
            fp_bytes = b""

        kwargs_bytes = pickle.dumps(spec_copy)
        worker_args.append((kwargs_bytes, df_bytes, fp_bytes))

    results: dict[int, dict] = {}
    with concurrent.futures.ProcessPoolExecutor(
        mp_context=ctx, max_workers=max_workers
    ) as executor:
        futures = {
            executor.submit(_single_train_worker, arg): i
            for i, arg in enumerate(worker_args)
        }
        for future in concurrent.futures.as_completed(futures):
            idx = futures[future]
            try:
                result = future.result()
                results[idx] = result

                _upsert_result(
                    result=result,
                    replace_existing=False,
                    model_type=result["model"],
                    split_type=result["split"],
                    seed=result["seed"],
                    pooling=result.get("pooling", POOLING_DEFAULT),
                    gnn_hidden_dim=result.get("gnn_hidden_dim", 128),
                    gnn_num_layers=result.get("gnn_num_layers", 4),
                    gnn_dropout=result.get("gnn_dropout", 0.15),
                    lr=result["lr"],
                    weight_decay=result["weight_decay"],
                )
            except Exception as exc:
                print(f"Worker {idx} failed: {exc}")

    if log_mlflow:
        _log_parallel_trainings_batch(results)

    return results


def _log_parallel_trainings_batch(
    results: dict[int, dict],
):
    for idx, result in results.items():
        run_name = f"{result['model']}_{result['split']}_seed{result['seed']}"
        if result.get("from_cache"):
            run_name += "_cache"

        configure_mlflow()
        with mlflow.start_run(
            run_name=run_name,
            tags={
                "project": "ml_chembl",
                "task": "bioactivity_regression",
                "target": "pIC50",
                "model_type": result["model"],
                "split_type": result["split"],
                "seed": str(result["seed"]),
                "from_cache": str(result.get("from_cache", False)).lower(),
                "parallel": "true",
            },
        ):
            params = {
                k: result[k]
                for k in [
                    "model",
                    "split",
                    "seed",
                    "epochs",
                    "epochs_trained",
                    "best_epoch",
                    "lr",
                    "batch_size",
                    "weight_decay",
                    "pooling",
                    "gnn_hidden_dim",
                    "gnn_num_layers",
                    "gnn_dropout",
                    "loss_fn",
                    "device",
                    "amp_enabled",
                    "from_cache",
                    "cache_path",
                ]
                if k in result
            }
            mlflow.log_params(params)

            for key in (
                "avg_train_loss",
                "avg_val_loss",
                "best_val_loss",
                "r2_val",
                "rmse_val",
                "mae_val",
                "r2_test",
                "rmse_test",
                "mae_test",
            ):
                if key in result and result[key] is not None:
                    mlflow.log_metric(key, float(result[key]))
