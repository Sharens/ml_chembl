from __future__ import annotations

import os
import random

import numpy as np
import torch
from sklearn.metrics import r2_score


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
        target = batch.y.view(-1, 1)
        out = model(batch)
    else:
        x, y = batch
        x = x.to(device, non_blocking=True)
        target = y.to(device, non_blocking=True).view(-1, 1)
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
            loss = criterion(out, target)

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
def predict_arrays(model, loader, device, is_gnn=False, use_amp=False):
    model.eval()
    all_preds = []
    all_targets = []

    for batch in loader:
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            out, target = _forward_batch(model, batch, device, is_gnn=is_gnn)
        all_preds.append(out.cpu().numpy())
        all_targets.append(target.cpu().numpy())

    preds = np.concatenate(all_preds).ravel()
    targets = np.concatenate(all_targets).ravel()
    return targets, preds


@torch.inference_mode()
def evaluate_loss(model, loader, criterion, device, is_gnn=False, use_amp=False):
    model.eval()
    total_loss = 0.0

    for batch in loader:
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            out, target = _forward_batch(model, batch, device, is_gnn=is_gnn)
            loss = criterion(out, target)
        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate_r2(model, loader, device, is_gnn=False, use_amp=False):
    targets, preds = predict_arrays(
        model, loader, device, is_gnn=is_gnn, use_amp=use_amp
    )
    if np.isnan(preds).any():
        print(f"BLAD: Predykcje modelu zawieraja NaN! (Pierwsze 5: {preds[:5]})")
    if np.isnan(targets).any():
        print("BLAD: Etykiety zawieraja NaN!")
    return r2_score(targets, preds)
