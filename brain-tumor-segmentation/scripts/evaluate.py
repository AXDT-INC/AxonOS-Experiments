#!/usr/bin/env python3
"""Deterministic evaluation / per-case Dice for BraTS 2023 GLI.

Loads a `.pt` checkpoint from train.py and evaluates on a deterministic split or
an explicit subject list with sliding-window inference. Reports per-case Dice
then averages across cases (BraTS convention). Saves a machine-readable JSON.

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/evaluate.py --config config/config.yaml \
        --checkpoint outputs/checkpoints/best.pt --split validation
    CUDA_VISIBLE_DEVICES=0 python scripts/evaluate.py --config config/config.yaml \
        --split validation --subjects BraTS-GLI-01000-000 BraTS-GLI-01100-000
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List

import torch
from torch.utils.data import DataLoader

import monai
from monai.data import Dataset, list_data_collate
from monai.utils import set_determinism

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (  # noqa: E402
    build_model,
    build_split,
    build_val_transforms,
    device_from_cfg,
    load_config,
    predict_map,
    read_manifest,
    root_path,
)


def load_state(model, path, device):
    st = torch.load(path, map_location="cpu", weights_only=False)
    if "model_state" in st:
        st = st["model_state"]
    model.load_state_dict(st)
    return st


@torch.no_grad()
def to_eval_batch(batch, device):
    image = batch["image"].to(device).float()
    label = batch["label"].to(device).long()
    if image.dim() == 4:
        image = image.unsqueeze(0)
    if label.dim() == 4:
        label = label.unsqueeze(0)
    return image, label


def run(dataset, cfg, device, ckpt_path: str):
    st_full = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = build_model(cfg).to(device)
    st = st_full.get("model_state", st_full)
    model.load_state_dict(st)
    model.eval()
    num_classes = cfg["data"]["num_classes"]
    amp_on = bool(cfg["train"].get("amp", True)) and device.type == "cuda"

    per_case = []
    per_class_sums = torch.zeros(num_classes)
    n = 0
    loader = DataLoader(dataset, batch_size=1, shuffle=False,
                        num_workers=int(cfg["eval"].get("workers", 0)),
                        collate_fn=list_data_collate)
    for batch in loader:
        image, label = to_eval_batch(batch, device)
        if amp_on:
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                pred = predict_map(model, image, cfg, device)
        else:
            pred = predict_map(model, image, cfg, device)
        pred = pred.cpu().squeeze(0)   # [D,H,W]
        label = label.cpu().squeeze(0)# [D,H,W]
        start = 0 if cfg["eval"]["include_background_in_dice"] else 1
        vals = []
        for c in range(start, num_classes):
            pm = (pred == c).flatten(); gm = (label == c).flatten()
            tp = float((pm & gm).float().sum()); den = float(pm.float().sum()) + float(gm.float().sum())
            d = 1.0 if den == 0 else 2.0 * tp / den
            vals.append(d); per_class_sums[c] += d
        per_case.append(float(sum(vals) / len(vals)))
        n += 1
    mean_dice = sum(per_case) / len(per_case) if per_case else float("nan")
    return {
        "st": {k: (v if not isinstance(v, torch.Tensor) else v.tolist()) for k, v in st.items()
               if k not in ("model_state", "optimizer_state", "scaler_state", "config")},
        "n_cases": n,
        "mean_foreground_dice": mean_dice,
        "per_class": {f"class_{c}": float(per_class_sums[c] / n) if n else float("nan")
                      for c in range(1, num_classes)},
        "per_case": per_case,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--checkpoint", default="")
    ap.add_argument("--split", default="validation")
    ap.add_argument("--subjects", nargs="*", default=[])
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if cfg.get("deterministic"):
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        set_determinism(seed=int(cfg["seed"]), use_deterministic_algorithms=True)
    device = device_from_cfg(cfg)

    if args.checkpoint:
        ckpt_path = args.checkpoint
    else:
        cdir = root_path(cfg, cfg["paths"]["checkpoints"])
        files = sorted(glob.glob(os.path.join(cdir, "*.pt")))
        if not files:
            raise SystemExit(f"No checkpoint in {cdir}; run scripts/train.py first or pass --checkpoint")
        ckpt_path = files[-1]
    print(f"[eval] checkpoint -> {ckpt_path}")

    meta = read_manifest(cfg)
    if args.subjects:
        ids = list(args.subjects)
    else:
        split = build_split(cfg, meta)
        if args.split not in split:
            raise SystemExit(f"Unknown split '{args.split}'. Available: {list(split.keys())}")
        ids = split[args.split]
    print(f"[eval] split={args.split or 'custom'}  n={len(ids)}")
    dataset = Dataset(data=[meta["items"][sid] for sid in ids],
                      transform=build_val_transforms(cfg))

    ev = run(dataset, cfg, device, ckpt_path)

    out = args.out or os.path.join(root_path(cfg, cfg["paths"]["metrics"]), "eval_metrics.json")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    payload = {
        "checkpoint": os.path.abspath(ckpt_path),
        "split": args.split,
        "n_cases": ev["n_cases"],
        "mean_foreground_dice": ev["mean_foreground_dice"],
        "per_class": ev["per_class"],
        "include_background": bool(cfg["eval"]["include_background_in_dice"]),
        "ckpt_epoch": ev["st"].get("epoch"),
        "ckpt_best_val_dice": ev["st"].get("best_val_dice"),
        "device": str(device),
        "torch_version": torch.__version__,
        "monai_version": monai.__version__,
        "dataset": meta["source"],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[eval] n_cases={ev['n_cases']}  mean_fg_dice={ev['mean_foreground_dice']:.4f}")
    print(f"[eval] per_class={ev['per_class']}")
    print(f"[eval] -> {out}")


if __name__ == "__main__":
    main()
