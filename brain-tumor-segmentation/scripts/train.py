#!/usr/bin/env python3
"""Deterministic 3D BraTS 2023 GLI segmentation training (PyTorch + MONAI).

Reads data/manifest.json (300 verified subjects, 4 modalities, label 0..3).
Splits deterministically (seed 42) into 240 train / 60 validation.
Trains a 3D U-Net on the Tesla V100 with mixed-precision DiceCE and
tumor-aware patch sampling; evaluates per-case Dice with sliding-window
inference; saves best + final checkpoints and per-epoch CSV/JSON metrics.

Nothing here runs at import time.

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config config/config.yaml
    CUDA_VISIBLE_DEVICES=0 python scripts/smoke_test.py           # tiny 1-subject, 1-epoch smoke

The smoke test lives in scripts/smoke_test.py (not executed here).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import monai
from monai.data import Dataset, list_data_collate, decollate_batch
from monai.utils import set_determinism

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (  # noqa: E402
    build_loss,
    build_model,
    build_split,
    build_train_transforms,
    build_val_transforms,
    device_from_cfg,
    load_config,
    per_case_dice,
    predict_map,
    read_manifest,
    root_path,
    save_split,
    to_one_hot,
)


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def make_dataset(items: Dict[str, Dict[str, Any]], ids: List[str], train: bool, cfg):
    transform = build_train_transforms(cfg) if train else build_val_transforms(cfg)
    data = [items[sid] for sid in ids]
    return Dataset(data=data, transform=transform)


def move_batch(batch, device):
    # batch from dict Dataset: list of dicts {image: (4,D,H,W) | [1,4,D,H,W], label: (D,H,W) | [1,D,H,W]}
    outs = {}
    for k, v in batch.items():
        v = list(v) if isinstance(v, (list, tuple)) else [v]
        v = torch.stack(v).to(device)
        outs[k] = v
    return outs


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def train_one_epoch(
    model, loader, optimizer, scaler, loss_fn, cfg, device, epoch
) -> float:
    model.train()
    tcfg = cfg["train"]
    num_classes = cfg["data"]["num_classes"]
    amp_on = bool(tcfg.get("amp", True)) and device.type == "cuda"
    max_grad_norm = float(tcfg.get("clip_grad_norm", 12.0))

    total, n = 0.0, 0
    n_skipped = 0
    n_step_skipped = 0
    for batch in loader:
        image = batch["image"].to(device).float()           # [B, 4, D, H, W]
        label = batch["label"].to(device).long()            # [B, D, H, W]
        if image.dim() == 4:
            image = image.unsqueeze(0)
        if label.dim() == 4:
            label = label.unsqueeze(0)

        # AMP
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_on):
            logits = model(image)
            target = to_one_hot(label, num_classes)
            loss = loss_fn(logits, target)

        # Non-finite loss guard: skip this batch before backward
        if not torch.isfinite(loss):
            n_skipped += 1
            continue

        optimizer.zero_grad(set_to_none=True)
        if amp_on:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            # Check gradients after unscaling for non-finiteness
            grad_ok = True
            for p in model.parameters():
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    grad_ok = False
                    break
            if grad_ok:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
            else:
                n_step_skipped += 1
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

        total += float(loss.item())
        n += 1

    if n_skipped or n_step_skipped:
        print(f"  [train] epoch {epoch}: skipped {n_skipped} non-finite batches, "
              f"{n_step_skipped} non-finite gradient steps")
    return total / max(n, 1)


@torch.no_grad()
def evaluate_val(
    model, loader, cfg, device
) -> Dict[str, Any]:
    """
    Sliding-window inference per subject; per-case Dice over foreground classes
    (1..3), averaged across cases.
    """
    model.eval()
    num_classes = cfg["data"]["num_classes"]
    amp_on = bool(cfg["train"].get("amp", True)) and device.type == "cuda"

    per_case_dices: List[float] = []
    per_class_sums = torch.zeros(num_classes)
    n_cases = 0
    per_subject = []

    for batch in loader:
        image = batch["image"].to(device).float()           # [1,4,D,H,W]
        label = batch["label"].to(device).long()            # [1,D,H,W]
        if image.dim() == 4:
            image = image.unsqueeze(0)
        if label.dim() == 4:
            label = label.unsqueeze(0)

        # predict per-case
        if amp_on:
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                pred = predict_map(model, image, cfg, device)   # [1, D, H, W]
        else:
            pred = predict_map(model, image, cfg, device)
        pred = pred.to("cpu")
        label = label.to("cpu")

        # per-case Dice
        start = 0 if cfg["eval"]["include_background_in_dice"] else 1
        case_d: List[float] = []
        for c in range(start, num_classes):
            pm = (pred[0] == c).flatten()
            gm = (label[0] == c).flatten()
            tp = float((pm & gm).float().sum())
            den = float(pm.float().sum()) + float(gm.float().sum())
            d = 1.0 if den == 0 else 2.0 * tp / den
            case_d.append(d)
            per_class_sums[c] += d
        per_case_dices.append(float(sum(case_d) / len(case_d)))
        per_subject.append(float(sum(case_d) / len(case_d)))
        n_cases += 1

    per_case = per_case_dices
    mean_dice = float(sum(per_case_dices) / len(per_case_dices)) if per_case_dices else float("nan")
    per_class = {f"class_{c}": float(per_class_sums[c] / n_cases) if n_cases else float("nan")
                 for c in range(1, num_classes)}
    return {
        "mean_foreground_dice": mean_dice,
        "per_class": per_class,
        "per_case": per_subject,
        "n_cases": n_cases,
    }


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--epochs", type=int, default=None, help="Override config's train.epochs.")
    ap.add_argument("--limit", type=int, default=None,
                    help="Cap subjects per split (for smoke test only).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Parse config, build everything, do NOT start the training loop.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    if args.limit is not None:
        cfg["_smoke_limit"] = args.limit

    if cfg.get("deterministic"):
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        set_determinism(seed=int(cfg["seed"]), use_deterministic_algorithms=True)

    device = device_from_cfg(cfg)
    if device.type == "cuda":
        nm = torch.cuda.get_device_name(0)
        mb = torch.cuda.get_device_properties(0).total_memory / 1e6
        print(f"[train] CUDA device: {nm}  {mb:,.0f} MB")
    else:
        print("[train] CPU device (AMP off)")
    print(f"[train] torch {torch.__version__} | monai {monai.__version__} | device {device}")

    # manifest + split
    meta = read_manifest(cfg)
    print(f"[train] manifest: {meta['source']}\n"
          f"        subjects: {meta['num_subjects']}   "
          f"modalities: {'/'.join(meta['modalities'])}   "
          f"n_classes: {meta['num_classes']}")
    split = build_split(cfg, meta)
    if args.limit is not None:
        split["train"] = split["train"][: args.limit]
        split["validation"] = split["validation"][: args.limit] or split["train"][:2]
    split_path = save_split(cfg, split)
    print(f"[train] split: train={len(split['train'])}  val={len(split['validation'])}")
    print(f"[train] split recorded -> {split_path}")
    if not split["train"]:
        raise SystemExit("no training subjects after --limit")

    # datasets + loaders
    train_set = make_dataset(meta["items"], split["train"], train=True, cfg=cfg)
    val_set = make_dataset(meta["items"], split["validation"], train=False, cfg=cfg)
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True,
                              num_workers=int(cfg["train"].get("workers", 0)),
                              collate_fn=list_data_collate)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False,
                            num_workers=int(cfg["eval"].get("workers", 0)),
                            collate_fn=list_data_collate)

    # model, optimizer, loss, scaler
    torch.manual_seed(int(cfg["seed"]))
    model = build_model(cfg).to(device)
    loss_fn = build_loss(cfg)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    amp_on = bool(cfg["train"].get("amp", True)) and device.type == "cuda"
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=float(cfg["train"]["learning_rate"]),
                                  weight_decay=float(cfg["train"]["weight_decay"]))
    scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(
                     optimizer, T_max=max(1, int(cfg["train"]["epochs"])))
                 if cfg["train"].get("scheduler") == "cosine" else None)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_on)
    print(f"[train] {type(model).__name__}: {n_params:,} params")
    print(f"[train] loss = {type(loss_fn).__name__}")
    print(f"[train] AMP = {amp_on}")

    if args.dry_run:
        print("[train] --dry-run: all objects constructed. NOT entering training loop.")
        return

    # checkpoint dirs (honour configurable paths)
    ckpt_dir = root_path(cfg, cfg["paths"]["checkpoints"])
    met_dir = root_path(cfg, cfg["paths"]["metrics"])
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(met_dir, exist_ok=True)

    now = lambda: datetime.now(timezone.utc).isoformat()
    csv_path = os.path.join(met_dir, "train_metrics.csv")
    epochs = int(cfg["train"]["epochs"])
    best = float("-inf")
    best_path = None
    t_start = time.time()

    with open(csv_path, "w", newline="") as csvf:
        w = csv.writer(csvf)
        w.writerow(["epoch", "train_loss", "val_dice_fg", "class_1", "class_2", "class_3",
                    "lr", "n_params", "best", "timestamp_utc"])

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            loss = train_one_epoch(model, train_loader, optimizer, scaler,
                                   loss_fn, cfg, device, epoch)
            ev = evaluate_val(model, val_loader, cfg, device)
            dice = ev["mean_foreground_dice"]
            if dice > best:
                best = dice
                best_path = os.path.join(ckpt_dir, "best.pt")
                torch.save({
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "val_dice": dice,
                    "per_class": ev["per_class"],
                    "n_params": n_params,
                    "config": cfg,
                    "amp": amp_on,
                }, best_path)
            if scheduler is not None:
                scheduler.step()

            ckpt = os.path.join(ckpt_dir, f"epoch-{epoch:03d}-valdice-{dice:.4f}.pt")
            torch.save({
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scaler_state": (scaler.state_dict() if hasattr(scaler, "state_dict") else None),
                "epoch": epoch,
                "val_dice": dice,
                "per_class": ev["per_class"],
                "n_params": n_params,
                "config": cfg,
                "amp": amp_on,
                "timestamp_utc": now(),
            }, ckpt)

            row = {
                "epoch": epoch, "train_loss": f"{loss:.6f}",
                "val_dice_fg": f"{dice:.6f}",
                "class_1": f"{ev['per_class']['class_1']:.6f}",
                "class_2": f"{ev['per_class']['class_2']:.6f}",
                "class_3": f"{ev['per_class']['class_3']:.6f}",
                "lr": f"{optimizer.param_groups[0]['lr']:.8e}",
                "n_params": int(n_params),
                "best": f"{best:.6f}",
                "timestamp_utc": now(),
            }
            w.writerow([row[k] for k in
                        ["epoch", "train_loss", "val_dice_fg", "class_1", "class_2",
                         "class_3", "lr", "n_params", "best", "timestamp_utc"]])
            csvf.flush()
            print(f"  epoch {epoch:02d}: loss={loss:.5f}  dice_fg={dice:.4f}  "
                  f"c1={ev['per_class']['class_1']:.4f} c2={ev['per_class']['class_2']:.4f} "
                  f"c3={ev['per_class']['class_3']:.4f}  [{time.time()-t0:.1f}s]  "
                  f"ckpt={os.path.basename(ckpt)}")

        # ---- final checkpoint ----
        final_ckpt = os.path.join(ckpt_dir, "final.pt")
        torch.save({
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scaler_state": (scaler.state_dict() if hasattr(scaler, "state_dict") else None),
            "epoch": epochs,
            "best_val_dice": best,
            "n_params": n_params,
            "config": cfg,
            "amp": amp_on,
        }, final_ckpt)

    # ---- final machine-readable metrics ----
    final_ev = evaluate_val(model, val_loader, cfg, device)
    final_metrics_json = os.path.join(met_dir, "final_metrics.json")
    with open(final_metrics_json, "w") as f:
        json.dump({
            "val_mean_foreground_dice": final_ev["mean_foreground_dice"],
            "val_per_class_dice": final_ev["per_class"],
            "best_val_dice": best,
            "epochs_trained": epochs,
            "train_subjects": len(split["train"]),
            "val_subjects": len(split["validation"]),
            "n_params": n_params,
            "amp": bool(amp_on),
            "seed": int(cfg["seed"]),
            "torch_version": torch.__version__,
            "monai_version": monai.__version__,
            "device": str(device),
            "dataset": meta["source"],
            "license": meta.get("license"),
        "duration_sec": round(time.time() - t_start, 3),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }, f, indent=2)

    print("\n[train] " + "=" * 60)
    print(f"[train] DONE  best={best:.4f}   final={final_ev['mean_foreground_dice']:.4f}")
    print(f"[train] per-class (final): {final_ev['per_class']}")
    print(f"[train] CSV  -> {csv_path}")
    print(f"[train] JSON -> {final_metrics_json}")
    print(f"[train] best ckpt -> {best_path}")
    print(f"[train] final ckpt -> {final_ckpt}")
    print(f"[train] duration: {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()
