from __future__ import annotations

import os
import random

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def seed_everything(seed: int = 42, deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def get_device(prefer_cuda: bool = True):
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def compute_num_workers(max_workers: int = 8):
    cpu_count = os.cpu_count() or 1
    return max(0, min(max_workers, cpu_count // 2))


def _forward_batch(model, batch, device, is_gnn=False):
    if is_gnn:
        batch = batch.to(device, non_blocking=True)
        target = batch.y
        out = model(batch)
    else:
        x, y = batch
        x = x.to(device, non_blocking=True).float()
        target = y.to(device, non_blocking=True)
        out = model(x)
    return out, target


def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
    is_gnn=False,
    clip_grad=1.0,
    scaler=None,
    use_amp=False,
):
    model.train()
    total_loss = 0.0

    for batch in loader:
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            out, target = _forward_batch(model, batch, device, is_gnn=is_gnn)
            loss = criterion(out.squeeze(-1), target.float())

        if torch.isnan(loss):
            print("Loss exploded to NaN! Stopping...")
            return float("nan")

        if scaler is not None:
            scaler.scale(loss).backward()
            if clip_grad is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)
            optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.inference_mode()
def evaluate_loss(model, loader, criterion, device, is_gnn=False, use_amp=False):
    model.eval()
    total_loss = 0.0

    for batch in loader:
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            out, target = _forward_batch(model, batch, device, is_gnn=is_gnn)
            loss = criterion(out.squeeze(-1), target.float())
        total_loss += loss.item()

    return total_loss / len(loader)


@torch.inference_mode()
def predict_arrays(model, loader, device, is_gnn=False, use_amp=False):
    model.eval()
    all_preds = []
    all_targets = []

    for batch in loader:
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            out, target = _forward_batch(model, batch, device, is_gnn=is_gnn)
        all_preds.append(out.squeeze(-1).cpu().numpy())
        all_targets.append(target.float().cpu().numpy())

    preds = np.concatenate(all_preds).ravel()
    targets = np.concatenate(all_targets).ravel()
    return targets, preds


def evaluate_r2(model, loader, device, is_gnn=False, use_amp=False):
    targets, preds = predict_arrays(
        model, loader, device, is_gnn=is_gnn, use_amp=use_amp
    )
    return float(r2_score(targets, preds))


def evaluate_rmse(model, loader, device, is_gnn=False, use_amp=False):
    targets, preds = predict_arrays(
        model, loader, device, is_gnn=is_gnn, use_amp=use_amp
    )
    return float(np.sqrt(mean_squared_error(targets, preds)))


def evaluate_mae(model, loader, device, is_gnn=False, use_amp=False):
    targets, preds = predict_arrays(
        model, loader, device, is_gnn=is_gnn, use_amp=use_amp
    )
    return float(mean_absolute_error(targets, preds))


def evaluate_all_metrics(model, loader, device, is_gnn=False, use_amp=False):
    targets, preds = predict_arrays(
        model, loader, device, is_gnn=is_gnn, use_amp=use_amp
    )
    return {
        "r2": float(r2_score(targets, preds)),
        "rmse": float(np.sqrt(mean_squared_error(targets, preds))),
        "mae": float(mean_absolute_error(targets, preds)),
    }


class WarmupCosineScheduler:
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        base_lr: float,
        min_lr: float = 1e-6,
        warmup_start_lr: float = 1e-7,
    ):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.warmup_start_lr = warmup_start_lr
        self.current_epoch = 0

    def step(self, epoch=None):
        if epoch is not None:
            self.current_epoch = epoch
        else:
            self.current_epoch += 1

        if self.current_epoch < self.warmup_epochs:
            progress = self.current_epoch / max(self.warmup_epochs, 1)
            lr = self.warmup_start_lr + (self.base_lr - self.warmup_start_lr) * progress
        else:
            progress = (self.current_epoch - self.warmup_epochs) / max(
                self.total_epochs - self.warmup_epochs, 1
            )
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (
                1.0 + np.cos(np.pi * progress)
            )

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        return lr


def train_one_epoch_with_accumulation(
    model,
    loader,
    optimizer,
    criterion,
    device,
    is_gnn=False,
    clip_grad=1.0,
    scaler=None,
    use_amp=False,
    accumulation_steps=1,
):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad(set_to_none=True)

    for batch_idx, batch in enumerate(loader):
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            out, target = _forward_batch(model, batch, device, is_gnn=is_gnn)
            loss = criterion(out.squeeze(-1), target.float())
            loss = loss / accumulation_steps

        if torch.isnan(loss):
            print("Loss exploded to NaN! Stopping...")
            return float("nan")

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(loader):
            if scaler is not None:
                if clip_grad is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=clip_grad
                    )
                scaler.step(optimizer)
                scaler.update()
            else:
                if clip_grad is not None:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=clip_grad
                    )
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item() * accumulation_steps

    return total_loss / len(loader)


__all__ = [
    "seed_everything",
    "get_device",
    "compute_num_workers",
    "train_one_epoch",
    "train_one_epoch_with_accumulation",
    "evaluate_loss",
    "predict_arrays",
    "evaluate_r2",
    "evaluate_rmse",
    "evaluate_mae",
    "evaluate_all_metrics",
    "WarmupCosineScheduler",
]
