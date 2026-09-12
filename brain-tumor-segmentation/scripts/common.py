"""Shared, import-safe utilities for the BraTS 2023 GLI workload.

Adapted to the verified 300-subject BraTS2023 GLI dataset (MedOtter/brats2023-gli-dataset):
- 4 input modalities: t1n, t1c, t2w, t2f
- 4 output classes:   0=background  1=NCR  2=ED  3=ET (already contiguous on disk)
- manifest:           data/manifest.json (source of truth)
- volumes:            240 x 240 x 155 at 1 mm isotropic

No GPU work is done at import time.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F

import monai
from monai.utils import set_determinism

import yaml  # pyyaml


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    cfg["_config_path"] = os.path.abspath(path)
    cfg["_root"] = os.path.dirname(os.path.dirname(os.path.abspath(path)))
    return cfg


def root_path(cfg: Dict[str, Any], rel: str) -> str:
    if os.path.isabs(rel):
        return rel
    return os.path.join(cfg["_root"], rel)


def device_from_cfg(cfg: Dict[str, Any]) -> torch.device:
    if cfg.get("device") == "cuda" and torch.cuda.is_available():
        return torch.device("cuda", 0)
    return torch.device("cpu")


# --------------------------------------------------------------------------- #
# Manifest
# --------------------------------------------------------------------------- #
def read_manifest(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Read data/manifest.json; return metadata + absolute per-subject file paths."""
    mpath = root_path(cfg, cfg["data"]["manifest"])
    with open(mpath) as f:
        m = json.load(f)
    data_root = root_path(cfg, cfg["data"]["data_root"])

    modalities: List[str] = list(cfg["data"]["modalities"])  # [t1n, t1c, t2w, t2f]

    items: Dict[str, Dict[str, Any]] = {}
    for s in m["subjects"]:
        sid = s["id"]
        items[sid] = {
            "image": [os.path.join(data_root, s["files"][mod]["path"]) for mod in modalities],
            "label": os.path.join(data_root, s["files"]["seg"]["path"]),
        }

    return {
        "source": m.get("source"),
        "license": m.get("license"),
        "num_subjects": m.get("num_subjects_selected", len(items)),
        "items": items,
        "modalities": modalities,
        "num_input_channels": len(modalities),
        "num_classes": int(cfg["data"]["num_classes"]),
        "manifest_path": mpath,
        "data_root": data_root,
    }


# --------------------------------------------------------------------------- #
# Deterministic split
# --------------------------------------------------------------------------- #
def build_split(cfg: Dict[str, Any], manifest_meta: Dict[str, Any]) -> Dict[str, list]:
    from monai.data import partition_dataset

    split_cfg = cfg["split"]
    subject_ids: List[str] = sorted(manifest_meta["items"].keys())
    ratios = [float(r) for r in split_cfg["ratios"]]
    while len(ratios) > 2 and ratios[-1] == 0.0:
        ratios.pop()
    parts = partition_dataset(
        data=subject_ids,
        ratios=ratios,
        shuffle=bool(split_cfg.get("shuffle", True)),
        seed=int(split_cfg.get("seed", cfg.get("seed", 42))),
    )
    return {
        "train": parts[0],
        "validation": parts[1],
        "test": parts[2] if len(parts) > 2 else [],
    }


def save_split(cfg: Dict[str, Any], split: Dict[str, list]) -> str:
    path = root_path(cfg, "data/splits/split_ids.json")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "deterministic": True,
        "method": "monai.data.partition_dataset",
        "seed": cfg["split"]["seed"],
        "shuffle": cfg["split"]["shuffle"],
        "ratios": [float(r) for r in cfg["split"]["ratios"]],
        "train": split["train"],
        "validation": split["validation"],
        "test": split["test"],
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path


# --------------------------------------------------------------------------- #
# Transforms
# --------------------------------------------------------------------------- #
def _per_image_minmax(t: torch.Tensor) -> torch.Tensor:
    hi, lo = float(t.max()), float(t.min())
    if hi - lo <= 0:
        return (t / (hi + 1e-8)).clamp(0.0, 1.0).float()
    return ((t - lo) / (hi - lo)).clamp(0.0, 1.0).float()


def build_train_transforms(cfg: Dict[str, Any]):
    """Load 4 modalities + mask -> normalize -> augment -> fixed tumor-aware crop.

    Every output patch is exactly ``train.patch_size`` regardless of the native
    240x240x155 shape, so batch collation can never fail on size mismatches.
    """
    from monai.transforms import (
        Compose, EnsureChannelFirstd, LoadImaged, Lambdad,
        RandAffined, RandCropByPosNegLabeld,
    )

    d = cfg["data"]
    tcfg = cfg["train"]
    keys = [d["image_key"], d["label_key"]]
    patch = list(tcfg["patch_size"])
    pos, neg = (list(tcfg.get("crop_pos_neg", [0.7, 0.3])))[:2]

    return Compose([
        LoadImaged(keys=keys),
        EnsureChannelFirstd(keys=keys),
        # mask -> integer class indices (0..3)
        Lambdad(keys=[d["label_key"]], func=lambda x: x.long()),
        # per-subject per-modality min-max to [0,1]
        Lambdad(keys=[d["image_key"]], func=_per_image_minmax),
        RandAffined(
            keys=keys,
            prob=0.9,
            rotate_range=(-0.25, 0.25),
            shear_range=(-0.1, 0.1),
            scale_range=(0.85, 0.85, 0.95, 1.05, 1.05, 1.05),
            mode=["bilinear", "nearest"],
            padding_mode=["border", "nearest"],
        ),
        # tumor-aware fixed-size crop (70% tumor-positive / 30% background)
        RandCropByPosNegLabeld(
            keys=keys,
            label_key=d["label_key"],
            image_key=d["image_key"],
            spatial_size=patch,
            pos=pos,
            neg=neg,
            num_samples=1,
            allow_smaller=False,
        ),
    ])


def build_val_transforms(cfg: Dict[str, Any]):
    """Load 4 modalities + mask -> normalize.

    Volume keeps its native 240x240x155 shape; sliding-window inference handles tiling so
    heterogeneous native sizes still work uniformly.
    """
    from monai.transforms import (
        Compose, EnsureChannelFirstd, LoadImaged, Lambdad,
    )

    d = cfg["data"]
    keys = [d["image_key"], d["label_key"]]
    return Compose([
        LoadImaged(keys=keys),
        EnsureChannelFirstd(keys=keys),
        Lambdad(keys=[d["label_key"]], func=lambda x: x.long()),
        Lambdad(keys=[d["image_key"]], func=_per_image_minmax),
    ])


# --------------------------------------------------------------------------- #
# Loss target: one-hot (B, C, D, H, W)
# --------------------------------------------------------------------------- #
def to_one_hot(label: torch.Tensor, num_classes: int) -> torch.Tensor:
    """[B, D, H, W] integer labels -> [B, C, D, H, W] one-hot float."""
    label = label.long()
    if label.dim() == 5 and label.shape[1] == 1:
        label = label.squeeze(1)
    return F.one_hot(label, num_classes).permute(0, -1, 1, 2, 3).float()


# --------------------------------------------------------------------------- #
# Per-case Dice (BraTS convention: per-case foreground Dice, then average)
# --------------------------------------------------------------------------- #
def per_case_dice(
    logits: torch.Tensor,   # [B, C, D, H, W] (logits) OR [B, D, H, W] (argmax map)
    label: torch.Tensor,    # [B, D, H, W] integer
    include_background: bool,
    num_classes: int,
) -> Dict[str, Any]:
    """
    For each case in the batch, Dice over foreground classes (1..num_classes-1),
    averaged over classes -> one per-case Dice value.

    Returns {"mean": float, "per_class": {class_name: float}, "per_case": [float]}.
    """
    # shape contract: logits are [B,C,D,H,W] (5-D); pre-argmax maps are [B,D,H,W] (4-D)
    if logits.dim() == 5:
        pred = torch.argmax(logits, dim=1)
    else:
        pred = logits
    pred = pred.reshape(-1, *pred.shape[-3:]).long()
    label = label.reshape(-1, *label.shape[-3:]).long() if label.dim() == 5 else label
    B = pred.shape[0]
    start = 0 if include_background else 1
    per_case_dices: List[float] = []
    per_class_sums = np.zeros(num_classes)
    for b in range(B):
        per: List[float] = []
        for c in range(start, num_classes):
            pm = (pred[b] == c).flatten()
            gm = (label[b] == c).flatten()
            tp = float((pm & gm).float().sum())
            den = float(pm.float().sum()) + float(gm.float().sum())
            d = 1.0 if den == 0 else 2.0 * tp / den
            per.append(d)
            per_class_sums[c] += d
        per_case_dices.append(float(np.mean(per)) if per else float("nan"))
    mean_dice = float(np.nanmean(per_case_dices)) if per_case_dices else float("nan")
    per_class = {
        f"class_{c}": float(per_class_sums[c] / B) if B > 0 else float("nan")
        for c in range(start, num_classes)
    }
    return {"mean": mean_dice, "per_class": per_class, "per_case": per_case_dices}


# --------------------------------------------------------------------------- #
# Model + loss
# --------------------------------------------------------------------------- #
def build_model(cfg: Dict[str, Any]) -> torch.nn.Module:
    from monai.networks.nets import UNet

    return UNet(
        spatial_dims=3,
        in_channels=int(cfg["data"]["num_input_channels"]),
        out_channels=int(cfg["data"]["num_classes"]),
        channels=list(cfg["model"]["channels"]),
        strides=list(cfg["model"]["strides"]),
        kernel_size=int(cfg["model"].get("kernel_size", 3)),
        num_res_units=int(cfg["model"].get("num_res_units", 1)),
    )


def build_loss(cfg: Dict[str, Any]) -> torch.nn.Module:
    """
    DiceCELoss with softmax over model logits.
    Target must be **one-hot (B, C, D, H, W)** in this MONAI version; to_onehot_y=False
    keeps MONAI from forcing its own one-hot (see `to_one_hot` helper above).
    """
    from monai.losses import DiceCELoss

    return DiceCELoss(
        softmax=True,
        to_onehot_y=False,
        include_background=True,
        squared_pred=True,
        jaccard=False,
        lambda_dice=1.0,
        lambda_ce=1.0,
    )


# --------------------------------------------------------------------------- #
# Sliding-window inference -> per-case argmax label map
# --------------------------------------------------------------------------- #
def predict_map(model, image: torch.Tensor, cfg: Dict[str, Any], device: torch.device) -> torch.Tensor:
    """
    image: [1, C, D, H, W] -> returns [1, D, H, W] argmax label map on CPU.
    Uses monai.inferers.sliding_window_inference so native 240x240x155 volumes are
    segmented tile-by-tile (configurable ROI + overlap) without ever exceeding
    the GPU memory budget.
    """
    from monai.inferers import sliding_window_inference

    ev = cfg["eval"]
    roi = list(ev["roi_size"])
    model.eval()
    with torch.no_grad():
        if device.type == "cuda" and cfg["train"].get("amp", True):
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                _run = lambda: sliding_window_inference(
                    image.to(device).float(), roi_size=roi,
                    sw_batch_size=int(ev.get("sw_batch_size", 2)),
                    predictor=model, overlap=float(ev.get("overlap", 0.25)),
                    mode="gaussian", sw_device=device,
                )
                logits = _run()
        else:
            logits = sliding_window_inference(
                image.to(device).float(), roi_size=roi,
                sw_batch_size=int(ev.get("sw_batch_size", 2)),
                predictor=model, overlap=float(ev.get("overlap", 0.25)),
                mode="gaussian", sw_device=device,
            )
    return torch.argmax(logits, dim=1, keepdim=True).cpu()
