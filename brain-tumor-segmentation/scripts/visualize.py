#!/usr/bin/env python3
"""Deterministic 3D brain-tumor MRI segmentation visualization.

For each subject, renders:
  * a mid-slice of each BraTS modality (FLAIR, T1, T1ce, T2),
  * a mid-slice of the ground truth label map,
  * a mid-slice of the model's segmentation prediction,
and saves a PNG side-by-side comparison plus a machine-readable JSON summary.

Uses matplotlib (Agg backend) — no GUI required, safe under headless CUDA.

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/visualize.py --config config/config.yaml \
        --checkpoint outputs/checkpoints/epoch-XXX.pt --split validation
    CUDA_VISIBLE_DEVICES=0 python scripts/visualize.py --config config/config.yaml \
        --subjects PRAD_00201 PRAD_00269
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
from typing import Optional

import monai
from monai.utils import set_determinism

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (  # noqa: E402
    build_model,
    build_split,
    device_from_cfg,
    load_config,
    read_manifest,
    root_path,
)
from common import predict_map  # noqa: E402

NUM_CLASSES = 4

# BraTS tumor palette: 1 = NCR, 2 = ED, 3 = ET (background 0 intentionally left transparent)
CLASS_COLOR_HEX = {1: "#fcae1d", 2: "#f17cb0", 3: "#00a495"}
CLASS_LABELS = {
    1: "1 = NCR (necrotic core)",
    2: "2 = ED (peritumoral edema)",
    3: "3 = ET (enhancing tumor)",
}


def _fg_dice(pred: np.ndarray, lab: np.ndarray) -> float:
    """Subject foreground Dice = mean over classes 1..3 (class with no pred & no GT -> 1.0).

    Matches the per-case foreground Dice convention used by train/evaluate.
    """
    vals = []
    for c in range(1, NUM_CLASSES):
        pm = pred == c
        gm = lab == c
        den = int(pm.sum()) + int(gm.sum())
        if den == 0:
            vals.append(1.0)
        else:
            vals.append(2.0 * int((pm & gm).sum()) / den)
    return float(np.mean(vals))


def _build_overlay(label_slice: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """(H, W) integer label map -> (H, W, 4) RGBA overlay (tumor classes only, bg transparent)."""
    from matplotlib.colors import to_rgba
    h, w = label_slice.shape
    rgba = np.zeros((h, w, 4), dtype=np.float64)
    for c, hexc in CLASS_COLOR_HEX.items():
        r, g, b, _a = to_rgba(hexc)
        mask = label_slice == c
        if mask.any():
            rgba[mask] = [r, g, b, alpha]
    return rgba


def _draw_pred_contours(ax, pred_slice: np.ndarray) -> None:
    """Outline each predicted tumor class (1..3) in its own colour."""
    for c, hexc in CLASS_COLOR_HEX.items():
        mask = (pred_slice == c).astype(float)
        if mask.sum() == 0:
            continue
        ax.contour(mask, levels=[0.5], colors=[hexc], linewidths=1.8, antialiased=True)


@torch.no_grad()
def render_subject(model, sample: Dict[str, Any], cfg: Dict[str, Any],
                   device: torch.device, out_dir: str) -> str:
    # 4-modality model input (order: t1n, t1c, t2w, t2f) — kept identical to before.
    chans = []
    for path in list(sample["image"]):
        a = np.asarray(nib.load(path).get_fdata(), dtype=np.float32)
        hi, lo = float(a.max()), float(a.min())
        if hi - lo <= 0:
            a = a * 0.0
        else:
            a = (a - lo) / (hi - lo)
        chans.append(np.clip(a, 0.0, 1.0))

    lab = np.asarray(nib.load(sample["label"]).get_fdata(), dtype=int).astype(int)

    # build model input (1, C, D, H, W)
    if len(chans) >= 4:
        x = np.stack(chans[:4], axis=0)
    else:
        x = np.stack(chans * (4 // max(1, len(chans))), axis=0)
    x_t = torch.from_numpy(x).float().unsqueeze(0)
    pred = predict_map(model, x_t, cfg, device).squeeze().numpy().astype(int)

    # FLAIR / T2-FLAIR is the final modality (t2f) in the manifest order.
    flair = chans[-1]

    # Axial slice with the largest non-background ground-truth tumor area.
    fg_per_slice = (lab > 0).sum(axis=(1, 2))
    z = int(np.argmax(fg_per_slice))

    mri = flair[z]
    gt = lab[z].astype(int)
    pr = pred[z].astype(int)
    dice = _fg_dice(pred, lab)

    title = f"Subject {sample.get('_id', 'unknown')}  ·  axial slice {z}"
    title += f"  ·  FG Dice {dice:.3f}"

    # Single clean row, exactly four panels.
    fig, axes = plt.subplots(1, 4, figsize=(16.0, 4.4), constrained_layout=True)
    fig.suptitle(title, fontsize=14)

    # 1) MRI alone
    axes[0].imshow(mri, cmap="gray")
    axes[0].set_title("T2-FLAIR (FLAIR)")

    # 2) Ground-truth overlay on MRI
    axes[1].imshow(mri, cmap="gray")
    axes[1].imshow(_build_overlay(gt, alpha=0.55))
    axes[1].set_title("Ground truth")

    # 3) Predicted overlay on MRI
    axes[2].imshow(mri, cmap="gray")
    axes[2].imshow(_build_overlay(pr, alpha=0.5))
    axes[2].set_title("Prediction")

    # 4) GT vs prediction comparison (GT fill + prediction outline) on MRI
    axes[3].imshow(mri, cmap="gray")
    axes[3].imshow(_build_overlay(gt, alpha=0.4))
    _draw_pred_contours(axes[3], pr)
    axes[3].set_title("GT (fill) vs Pred (outline)")

    for ax in axes:
        ax.set_xlim(0, mri.shape[1])
        ax.set_ylim(mri.shape[0] - 1, 0)
        ax.axis("off")

    # Compact legend explaining the BraTS tumor classes (consistent colours).
    handles = [
        matplotlib.patches.Patch(facecolor=CLASS_COLOR_HEX[1], edgecolor="k", label=CLASS_LABELS[1]),
        matplotlib.patches.Patch(facecolor=CLASS_COLOR_HEX[2], edgecolor="k", label=CLASS_LABELS[2]),
        matplotlib.patches.Patch(facecolor=CLASS_COLOR_HEX[3], edgecolor="k", label=CLASS_LABELS[3]),
        matplotlib.patches.Patch(facecolor="#d0d0d0", edgecolor="k", label="Grey = MRI (T2-FLAIR)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=11,
               framealpha=0.95, borderpad=0.4, handletextpad=0.6)

    out_path = os.path.join(out_dir, f"subject-{sample.get('_id', 'unknown')}.png")
    fig.savefig(out_path, dpi=int(cfg.get("viz", {}).get("dpi", 150)))
    plt.close(fig)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--checkpoint", default="")
    ap.add_argument("--split", default="validation")
    ap.add_argument("--subjects", nargs="*", default=[])
    ap.add_argument("--max", type=int, default=4)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if cfg.get("deterministic"):
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        set_determinism(seed=int(cfg["seed"]), use_deterministic_algorithms=True)

    device = device_from_cfg(cfg)
    if device.type == "cuda":
        try:
            nm = torch.cuda.get_device_name(0)
            mb = torch.cuda.get_device_properties(0).total_memory / 1e6
            print(f"[viz] CUDA ready: {nm}  {mb:.0f} MB")
        except Exception as e:
            print(f"[viz] CUDA query warning: {e}")
    print(f"[viz] torch {torch.__version__} | monai {monai.__version__} | device {device}")

    model = build_model(cfg).to(device)
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        if "model_state" in state:
            model.load_state_dict(state["model_state"])
        else:
            model.load_state_dict(state["state_dict"])
        print(f"[viz] loaded checkpoint -> {args.checkpoint}")
    else:
        print("[viz] WARNING: default (untrained) weights used. Pass --checkpoint for "
              "a meaningful visualization.")
    model.eval()

    manifest_meta = read_manifest(cfg)
    index = manifest_meta["items"]
    split = build_split(cfg, manifest_meta)
    if args.subjects:
        ids = list(args.subjects)
        split_name = "custom"
    else:
        if args.split not in split:
            raise SystemExit(f"Unknown split '{args.split}'. Available: {list(split.keys())}")
        ids = split[args.split][: max(1, args.max)]
        split_name = args.split
    print(f"[viz] split={split_name}   subjects={len(ids)}")

    out_dir = args.out or root_path(cfg, cfg["paths"]["visualizations"])
    os.makedirs(out_dir, exist_ok=True)

    rendered = []
    for sid in ids:
        if sid not in index:
            print(f"[viz] subject {sid} not in manifest; skipping")
            continue
        s = dict(index[sid]); s["_id"] = sid
        path = render_subject(model, s, cfg, device, out_dir)
        rendered.append(path)
        print(f"  -> {path}")

    with open(os.path.join(out_dir, "viz_manifest.json"), "w") as f:
        json.dump(
            {
                "split": split_name,
                "subjects": ids,
                "rendered": rendered,
                "checkpoint": args.checkpoint or "(default/untrained)",
                "device": str(device),
                "torch_version": torch.__version__,
                "monai_version": monai.__version__,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            },
            f,
            indent=2,
        )
    print(f"[viz] summary -> {os.path.join(out_dir, 'viz_manifest.json')}")
    print("[viz] done")


if __name__ == "__main__":
    main()
