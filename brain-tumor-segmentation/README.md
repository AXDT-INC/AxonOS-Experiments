# BraTS 2023 GLI — Reproducibility Package

Reproducible 3D glioma segmentation experiment on **BraTS 2023 GLI** MRI data,
running a compact 3D U-Net. This package accompanies an experimental case study
and contains everything needed to reproduce the workflow, configuration, data
references, recorded metrics, and visual results — **without** redistributing the
raw MRI dataset or large model checkpoints.

**Status: Experimental case study. Not clinical validation, not a diagnostic
system, not a medical device claim, not a hardware benchmark.**

---

## Package layout

```
reproducibility/
├── README.md                  This file
├── MANIFEST.md                File-by-file SHA-256 checksums (human-readable)
├── MANIFEST.sha256            Machine-verifiable SHA-256 checksums (sha256sum -c)
├── environment.lock.txt       Exact recorded Python/runtime versions
├── config/
│   ├── config.yaml            Original 5-epoch experiment configuration
│   └── config-50epoch.yaml    Definitive 50-epoch follow-up configuration
├── scripts/
│   ├── common.py              Shared utilities (model, loss, transforms, Dice)
│   ├── train.py               3D training loop (deterministic settings)
│   ├── evaluate.py            Standalone evaluation
│   ├── visualize.py           Per-subject visualization
│   ├── download_braTS2023_gli.py   Dataset downloader
│   ├── make_synthetic_dataset.py   Synthetic data generator (pipeline validation)
│   ├── smoke_test.py          End-to-end pipeline smoke test
│   └── make_figures.py        Reproduces the figures from recorded artifacts
├── data/
│   ├── manifest.json          300-subject manifest (paths, sizes, URLs, license)
│   ├── manifest-subject-ids.csv  The exact 300 subject IDs
│   ├── _download_report.json  Download integrity report (300/300, 1500/1500)
│   └── splits/
│       └── split_ids.json     Deterministic 240/60 train/validation split (seed 42)
├── metrics/
│   ├── train_metrics.csv      Per-epoch training log (50 epochs)
│   ├── final_metrics.json     Follow-up final training summary
│   └── eval_metrics.json      Independent evaluation of best checkpoint
├── results/
│   ├── figures/               learning_curve.png, per_class_dice.png, gpu_telemetry.png
│   └── visualizations/        5 validation-subject slice overlays + manifest
└── telemetry/
    └── gpu0.csv               Raw GPU telemetry, 6,297 samples (~2 s interval)
```

---

## Recorded results (source of truth)

These are the results recorded in the experiment artifacts. They are preserved
verbatim and are **not** modified by this package.

### Original 5-epoch experiment (September 3, 2026)

- 300 subjects, 240/60 split, seed 42, compact 3D U-Net (601,985 params)
- 5 epochs, duration = 1,323.975 s (~22 min 4 s)
- Mean foreground Dice = **0.533161** (`final_metrics.json`)
- Per-class: NCR 0.270380, ED 0.664768, ET 0.664335

Artifacts preserved at `outputs/original-5epoch-backup/` in the original project.

### 50-epoch follow-up (September 8, 2026)

- Same scientific configuration (identical hyperparameters, model, split,
  augmentation); separate fresh completed run
- 50 epochs, duration = 12,813.172 s (~3 h 33 min)
- Best epoch = 50 (final); best/final mean foreground Dice = **0.705055**
- Independent evaluation of `best.pt` reproduced mean foreground Dice = **0.7051**
- Final independently reproduced per-class: NCR 0.608520, ED 0.746048, ET 0.760596

The independent-evaluation figures are **checkpoint-evaluation
reproducibility** — the post-training evaluator re-run on the saved best
checkpoint — and are not an independent reproduction of the complete training
trajectory.

### Derived comparison (vs original recorded 5-epoch run)

- Absolute improvement: 0.7051 - 0.5332 = **+0.1719**
- Relative improvement: **+32.24%**

### Important: separate training trajectories

The original 5-epoch run is an **independently recorded prior run**, NOT epoch 5
of the 50-epoch run. The follow-up run's own epoch-5 Dice was 0.495, while the
original independent 5-epoch experiment was 0.533. These are separate runs with
different trajectories. Do not present the original 0.5332 as a point on the
50-epoch learning curve.

---

## Environment and runtime

`environment.lock.txt` records the reference runtime used for both runs:

- Platform: Linux 5.15.0-186-generic x86_64 (glibc 2.35)
- Python 3.10.12
- PyTorch 2.5.1+cu121 (CUDA runtime 12.1, cuDNN 90100)
- MONAI 1.6.0
- numpy 1.26.4, scipy 1.15.3, nibabel 5.4.2, matplotlib 3.10.9, pyyaml 5.4.1
- GPU: 1x NVIDIA Tesla V100-SXM2-32GB (driver 580.173.02, CUDA 13.0 driver)

No package was installed, upgraded, or removed by this project.
`environment.lock.txt` records the reference environment used for the
experiment. For the closest reproduction, match the recorded software,
driver/runtime and hardware versions as closely as practical.

---

## Dataset acquisition

The raw MRI dataset is **not redistributed** in this package (it is ~2.88 GB).

**Source:** `MedOtter/brats2023-gli-dataset` on Hugging Face
(https://huggingface.co/datasets/MedOtter/brats2023-gli-dataset), a public
redistribution of the BraTS 2023 GLI challenge training data (Synapse
syn51156910). **License: CC-BY-4.0.**

`data/manifest.json` contains the exact per-file relative paths, byte sizes, and
download URLs for all 300 subjects and 1,500 NIfTI files, plus the
`subset_id_hash_sha256`, so the identical subset can be fetched and verified.

`scripts/download_braTS2023_gli.py` is the resumable, multi-threaded downloader
used to obtain the data. After downloading, `data/_download_report.json`
documents the integrity verification (300/300 subjects, 1500/1500 files,
3,094,014,357 bytes, 0 failures).

**License/attribution requirement:** If you use the dataset, you must comply
with the CC-BY-4.0 license and attribute the BraTS 2023 challenge. See
`data/manifest.json` for the canonical origin and license URL.

---

## Commands

All commands are relative to a project root containing `config/`, `scripts/`,
`data/`, and the dataset under `data/raw/brats2023-gli/`.

### Pre-flight

```bash
cat environment.lock.txt
CUDA_VISIBLE_DEVICES=0 python -c "import torch; print(torch.cuda.get_device_name(0))"
```

### Verify dataset integrity (after download)

```bash
python3 -c "
import json, os
m = json.load(open('data/manifest.json'))
missing = 0
for s in m['subjects']:
    for k in ('t1n','t1c','t2w','t2f','seg'):
        p = os.path.join('data/raw/brats2023-gli', s['files'][k]['path'])
        if not os.path.exists(p): missing += 1
print('subjects:', len(m['subjects']), 'missing files:', missing)
"
```

### Train — definitive 50-epoch run

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train.py \
    --config config/config-50epoch.yaml --epochs 50
```

Outputs go to `outputs/50epoch/` (checkpoints, metrics, visualizations).

### Train — original 5-epoch run

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train.py \
    --config config/config.yaml --epochs 5
```

### Evaluate

The evaluation command assumes the training step above has already completed
and produced `outputs/50epoch/checkpoints/best.pt`. This package
intentionally does **not** distribute model checkpoints, so `best.pt` must be
produced locally by training before it can be evaluated.

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate.py \
    --config config/config-50epoch.yaml \
    --checkpoint outputs/50epoch/checkpoints/best.pt \
    --split validation \
    --out outputs/50epoch/metrics/eval_metrics.json
```

### Visualize

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/visualize.py \
    --config config/config-50epoch.yaml --split validation --max 5 \
    --checkpoint outputs/50epoch/checkpoints/best.pt
```

### Smoke test (synthetic data; no real dataset needed)

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/smoke_test.py
```

### Regenerate figures from recorded artifacts

`scripts/make_figures.py` regenerates `learning_curve.png`,
`per_class_dice.png`, and `gpu_telemetry.png` from the recorded metric/telemetry
artifacts. It is included as reference/provenance. Note that it expects the
**original project layout** (paths under `outputs/50epoch/...` and
`outputs/original-5epoch-backup/...`), so it runs cleanly inside the original
project tree rather than standalone from inside this package:

```bash
# From the ORIGINAL project root (not the standalone package):
python outputs/50epoch/figures/make_figures.py
```

The three source figures are already committed in `results/figures/` for
convenience and are unchanged reproductions of the recorded data.

---

## Telemetry methodology

GPU telemetry was collected by sampling `nvidia-smi` at approximately
2-second intervals during the 50-epoch run (`telemetry/gpu0.csv`, 6,297
samples). Fields: timestamp, GPU index, name, utilization %, memory used MB,
memory total MB, power draw W, temperature C.

**Interpretation cautions:**

- The whole-window mean GPU utilization is **24.15%** (median 6%, min 3%,
  max 100%). This includes periods before/after/between epochs when the GPU was
  idle or doing lightweight work. This is **NOT** the average utilization during
  active training compute.
- **Conditional** active-sample statistics (utilization >= 10%, n = 1,975):
  mean utilization **64.11%** (median 63%). Labeled clearly as conditional.
- Safe headline: "GPU utilization reached 100% during compute-intensive phases."
- The workload alternates between GPU-heavy computation (training steps) and
  data preparation/validation phases.
- VRAM mean 3,249 MB, peak 3,433 MB; power mean 86.95 W, peak 216.39 W;
  temperature mean 38.83 C, peak 48 C.
- **No cost claim** is made; no billing artifact was recorded.

---

## Expected stochastic variability

GPU-based deep learning is not bitwise reproducible across separate stochastic
training runs, even with the same nominal seed and configuration. Contributing
factors include:

- Floating-point non-determinism in certain CUDA operations (thread scheduling,
  memory allocation patterns, hardware parallelism).
- Data-loading order effects across multiple DataLoader workers and async I/O.
- AMP dynamic loss scaling and float16/float32 casting.

The configuration fixes the random seed (seed 42) and the deterministic 240/60
train/validation split, and the project requests deterministic behavior where
the underlying libraries support it. The recorded configuration, split
metadata, environment information, scripts, metrics, and checksums support
reproducibility-oriented execution and verification.

However, separate fresh runs can follow different trajectories even when started
with the same nominal seed and configuration; exact numerical or bitwise
equality across fresh GPU training runs is **not** guaranteed. The original
5-epoch run's Dice of 0.5332 and the 50-epoch follow-up's own epoch-5 Dice of
0.495302 demonstrate that separate trajectories occurred.

---

## AMP repair (engineering provenance)

During development of the follow-up, an initial longer run exposed an AMP
correctness issue: gradient clipping was applied to *scaled* AMP gradients
before unscaling, causing a GradScaler collapse (scale dropped from 2048 at
epoch 5 to ~7.28e-12 at epoch 6) and NaN loss from epoch 6 onward.

The repair added `scaler.unscale_(optimizer)` before `clip_grad_norm_`, plus
finite-loss and finite-gradient safeguards that skip and report non-finite
steps. **No scientific hyperparameters were changed.** The definitive 50-epoch
run completed without NaN collapse. This is engineering/reproducibility
provenance, not a medical finding.

---

## Reproducibility of this package

- Verify file integrity: `sha256sum -c MANIFEST.sha256`
- The split (`data/splits/split_ids.json`) is derived deterministically from
  `data/manifest.json` via `monai.data.partition_dataset(seed=42, shuffle=True)`
  with ratios 0.8/0.2 → 240 train / 60 validation.

---

## What is intentionally excluded

- **Raw MRI dataset** (~2.88 GB) — not redistributed; see `data/manifest.json`
  and `scripts/download_braTS2023_gli.py` to obtain it. See the dataset's
  CC-BY-4.0 license and attribution requirements.
- **Model checkpoints** (`.pt` files) — large binary model weights. The
  scientific results (metrics, figures) are included; the weights live in the
  original project at `outputs/original-5epoch-backup/checkpoints/` and
  `outputs/50epoch/checkpoints/`.
- **Secrets, credentials, tokens, private paths, wallet information, IP
  addresses.** None are present.

---

## License / attribution

- Dataset: CC-BY-4.0 (BraTS 2023). Attribute the BraTS 2023 challenge
  (Synapse syn51156910) and the Hugging Face redistributor `MedOtter`.
- Code: see original project for license terms; standard open-source tools
  (PyTorch, MONAI) are used.

---

*Prepared from experimental artifacts recorded September 3, 2026 (original run)
and September 8, 2026 (50-epoch follow-up). Original 5-epoch artifacts are
preserved unchanged in the original project at `outputs/original-5epoch-backup/`.*
