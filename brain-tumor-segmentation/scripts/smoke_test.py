#!/usr/bin/env python3
"""End-to-end GPU-0 smoke test for the brain-tumor segmentation workload.

Purpose: verify that the full pipeline (data -> transforms -> model -> loss ->
gradient step -> checkpoint -> Dice evaluation -> PNG visualization) runs and
terminates cleanly on GPU 0, using a tiny synthetic dataset (16 subjects,
48^3 each) generated locally. No multi-GB download is required.

This script does NOT train a useful model — it is only a plumbing check.

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/smoke_test.py
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

def run(cmd: list[str], **kw) -> int:
    print("+ " + " ".join(cmd))
    r = subprocess.run(cmd, **kw)
    return r.returncode

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--num-subjects", type=int, default=16)
    ap.add_argument("--size", type=int, default=48,
                    help="Voxel grid size. 48 is small and fast for smoke.")
    ap.add_argument("--epochs", type=int, default=1, help="Number of training epochs for smoke.")
    ap.add_argument("--limit-per-split", type=int, default=2,
                    help="Subjects per split used for smoke. Keep small.")
    args = ap.parse_args()

    here = Path(__file__).resolve().parent
    root = here.parent
    cfg = args.config
    py = sys.executable
    t0 = time.time()

    print(f"[smoke] starting  torch={__import__('torch').__version__}  monai={__import__('monai').__version__}  pid={os.getpid()}")
    if __import__('torch').cuda.is_available():
        dev = __import__('torch').cuda.get_device_name(0)
        print(f"[smoke] cuda device -> {dev}")
    else:
        print("[smoke] WARNING: CUDA not visible. smoke will still run on CPU.")

    # ---- 1. generate a small synthetic dataset (local, deterministic) ---- #
    manifest = root / "data" / "manifest.json"
    synthetic_dir = root / "data" / "raw" / "synthetic"
    if not manifest.exists():
        print("\n[smoke] no manifest; generating synthetic dataset ...")
        r = run([py, str(here / "make_synthetic_dataset.py"),
                 "--config", cfg,
                 "--num-subjects", str(args.num_subjects),
                 "--size", str(args.size),
                 "--seed", "42",
                 "--outdir", "data/raw/synthetic",
                 "--manifest", "data/manifest.json",
                 "--allow-override"])
        if r != 0:
            print(f"[smoke] data generation failed (code {r})")
            return 1

    # ---- 2. one-epoch GPU training (smoke) ---- #
    print("\n[smoke] one-step train (smoke) ...")
    r = run([py, str(here / "train.py"),
             "--config", cfg,
             "--epochs", str(args.epochs),
             "--limit", str(args.limit_per_split)])
    if r != 0:
        print(f"[smoke] train failed (code {r})")
        return 1

    # ---- 3. evaluate on a single val subject ---- #
    print("\n[smoke] evaluate (smoke) ...")
    import glob, json
    ckpts = sorted(glob.glob(str(root / "outputs" / "checkpoints" / "*.pt")))
    if not ckpts:
        print("[smoke] no checkpoint produced; skipping evaluate")
        return 1
    latest = ckpts[-1]
    r = run([py, str(here / "evaluate.py"),
             "--config", cfg,
             "--checkpoint", latest,
             "--split", "validation"])
    if r != 0:
        print(f"[smoke] evaluate failed (code {r})")
        return 1

    # ---- 4. visualise one subject ---- #
    print("\n[smoke] visualise (smoke) ...")
    r = run([py, str(here / "visualize.py"),
             "--config", cfg,
             "--checkpoint", latest,
             "--split", "validation",
             "--max", "1"])
    if r != 0:
        print(f"[smoke] visualize failed (code {r})")
        return 1

    print("\n" + "=" * 60)
    print(f"[smoke] SMOKE-TEST PASSED in {time.time()-t0:.2f}s")
    print(f"[smoke] last checkpoint   : {latest}")
    print(f"[smoke] metrics           : {str(root)}/outputs/metrics/final_metrics.json")
    print(f"[smoke] visualisations    : {str(root)}/outputs/visualizations/")
    print(f"[smoke] synthetic data    : {synthetic_dir}/")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
