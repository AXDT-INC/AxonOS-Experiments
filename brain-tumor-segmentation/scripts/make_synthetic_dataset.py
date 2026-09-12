#!/usr/bin/env python3
"""Generate a small, deterministic synthetic BraTS-like NIfTI dataset.

This exists so the GPU smoke-test can be run end-to-end WITHOUT first
downloading the multi-GB real BraTS/TCIA dataset (which requires explicit
approval and a restricted network route to download it). It produces
``num_subjects`` subjects, each with:
  * 4 modality volumes (FLAIR, T1, T1ce, T2), isotropic 96x96x96 at default,
  * 1 label volume with raw BraTS codes in {0,1,2,4} (4 = enhancing) so the
    standard BraTS label remap (4 -> 3) is exercised end-to-end,
  * a manifest (``data/manifest.json``) matching the schema consumed by
    ``train``/``evaluate``/``visualize``.

Everything is seeded (``--seed``); the same seed produces bit-identical files.

NOTE: This is a pipeline validator. Real BraTS/TCIA data is preferred once it
has been downloaded with the commands in ``docs/DATASET_SELECTION.md``.

Usage:
    python scripts/make_synthetic_dataset.py --num-subjects 16 --size 96 \
        --outdir data/raw/synthetic
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import numpy as np

import nibabel as nib


def _ellipsoid(center: np.ndarray, radii: np.ndarray, grid: tuple) -> np.ndarray:
    grid = tuple(grid)
    axis = [np.linspace(-0.5, 0.5, s, endpoint=True) for s in grid]
    zz, yy, xx = np.meshgrid(*axis, indexing="ij")
    r2 = ((xx - center[0]) / max(radii[0], 1e-8)) ** 2 + \
         ((yy - center[1]) / max(radii[1], 1e-8)) ** 2 + \
         ((zz - center[2]) / max(radii[2], 1e-8)) ** 2
    return (r2 <= 1.0).astype(np.float32)


def _make_subject(rng: np.random.Generator, size: int):
    S = size
    label = np.zeros((S, S, S), dtype=np.float32)
    center = np.array([S * 0.42, S * 0.5, S * 0.5])
    core = _ellipsoid(center, np.array([S * 0.07, S * 0.08, S * 0.06]), (S, S, S))
    shell = _ellipsoid(center, np.array([S * 0.14, S * 0.15, S * 0.12]), (S, S, S))
    enh = _ellipsoid(center, np.array([S * 0.20, S * 0.21, S * 0.19]), (S, S, S))
    label[enh > 0] = 4.0
    label[(shell > 0) & (label == 0)] = 1.0
    label[core > 0] = 2.0

    tissue = rng.normal(0.5, 0.05, (S, S, S)).astype(np.float32)
    cls_to_intensity = {0: 0.40, 1: 0.55, 2: 0.30, 4: 0.80}
    base = np.zeros((S, S, S), dtype=np.float32)
    for k, v in cls_to_intensity.items():
        base[(label == k)] = v
    def channel(shift: float, gain: float) -> np.ndarray:
        v = tissue + gain * base + shift + rng.normal(0, 0.02, (S, S, S)).astype(np.float32)
        return np.clip(v, 0.0, 1.0).astype(np.float32)
    return (channel(0.00, 1.00), channel(0.05, 0.85),
            channel(0.10, 1.10), channel(0.03, 0.95)), label


def _write_subject(outdir: str, root_sub: str, sid: str, chans, label, affine, cfg):
    sub = os.path.relpath(os.path.join(outdir, sid), cfg["_root"])
    os.makedirs(os.path.join(outdir, sid), exist_ok=True)
    rel = {}
    for name, arr in [("FLAIR", chans[0]), ("T1", chans[1]),
                      ("T1ce", chans[2]), ("T2", chans[3]), ("label", label)]:
        p = os.path.join(outdir, sid, f"{sid}_{name}.nii.gz")
        nib.save(nib.Nifti1Image(arr.astype(np.float32), affine), p)
        rel[name] = os.path.join(sub, f"{sid}_{name}.nii.gz")
    return {"id": sid, **rel}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--num-subjects", type=int, default=16)
    ap.add_argument("--size", type=int, default=96)
    ap.add_argument("--spacing", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", default="data/raw/synthetic")
    ap.add_argument("--manifest", default="data/manifest.json")
    ap.add_argument("--source-tag", default="synthetic-braTS-like")
    ap.add_argument("--allow-override", action="store_true")
    args = ap.parse_args()

    sys_path = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, sys_path)
    from common import load_config
    try:
        cfg = load_config(args.config)
    except Exception:
        cfg = {"_root": os.getcwd()}
    root = cfg["_root"]

    outdir = args.outdir if os.path.isabs(args.outdir) else os.path.join(root, args.outdir)
    manifest_path = args.manifest if os.path.isabs(args.manifest) else os.path.join(root, args.manifest)
    if os.path.exists(outdir) and os.listdir(outdir) and not args.allow_override:
        raise SystemExit(f"outdir {outdir} has content. Pass --allow-override to replace.")
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(os.path.dirname(manifest_path) or ".", exist_ok=True)

    rng = np.random.default_rng(args.seed)
    affine = np.diag([args.spacing] * 3 + [1.0])

    samples = []
    images: dict = {"FLAIR": [], "T1": [], "T1ce": [], "T2": [], "label": []}
    for i in range(args.num_subjects):
        sid = f"SYN{i:04d}"
        (fl, t1, t1ce, t2), label = _make_subject(rng, args.size)
        sample = _write_subject(outdir, "raw/synthetic", sid, (fl, t1, t1ce, t2), label, affine, cfg)
        samples.append(sample)
        images["FLAIR"].append(sample["FLAIR"])
        images["T1"].append(sample["T1"])
        images["T1ce"].append(sample["T1ce"])
        images["T2"].append(sample["T2"])
        images["label"].append(sample["label"])
        print(f"  -> {sid}")

    manifest = {
        "source": args.source_tag,
        "access": "local (generated on disk)",
        "license": "synthetic BraTS-style codes {0,1,2,4}; intended for pipeline validation only",
        "num_subjects": len(samples),
        "images": images,
        "samples": samples,
        "modalities": ["FLAIR", "T1", "T1ce", "T2"],
        "note": "Synthetic data produced by scripts/make_synthetic_dataset.py for pipeline validation only.",
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[make-synthetic] wrote {len(samples)} subjects -> {outdir}")
    print(f"[make-synthetic] manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
