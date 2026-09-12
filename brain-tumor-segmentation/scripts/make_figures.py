#!/usr/bin/env python3
"""Generate publication-ready evidence figures for the BraTS 50-epoch follow-up.

Reads ONLY recorded experiment artifacts. Does not train, evaluate, infer,
download, or alter any existing scientific data.
"""
import json
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from datetime import datetime

BASE = "outputs/50epoch"
FIG = f"{BASE}/figures"
MET = f"{BASE}/metrics"
TEL = f"{BASE}/telemetry/gpu0.csv"

matplotlib.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "figure.dpi": 160,
    "savefig.bbox": "tight",
})

# --------------------------------------------------------------------------- #
# Load sources of truth
# --------------------------------------------------------------------------- #
m = pd.read_csv(f"{MET}/train_metrics.csv")
fj = json.load(open(f"{MET}/final_metrics.json"))
ej = json.load(open(f"{MET}/eval_metrics.json"))
oj = json.load(open("outputs/original-5epoch-backup/metrics/final_metrics.json"))

BEST_DICE = fj["best_val_dice"]
BEST_EPOCH = 50
ORIG_5EP = oj["val_mean_foreground_dice"]

# --------------------------------------------------------------------------- #
# 1. learning_curve.png
# --------------------------------------------------------------------------- #
fig, ax1 = plt.subplots(figsize=(9.5, 6.0))

ax1.plot(m["epoch"], m["val_dice_fg"], "-o", ms=4, lw=2,
         color="#1f77b4", label="Validation mean foreground Dice")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Validation mean foreground Dice (1 = perfect overlap)")
ax1.set_xlim(1, 51)
ax1.set_ylim(0.2, 0.85)
ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
ax1.grid(True, which="major", alpha=0.3, linestyle="--")

ax2 = ax1.twinx()
ax2.plot(m["epoch"], m["train_loss"], "-o", ms=4, lw=1.8,
         color="#d62728", alpha=0.85, label="Training loss")
ax2.set_ylabel("Training loss")
ax2.tick_params(axis="y", labelcolor="#d62728")

# Best point marker at epoch 50
ax1.scatter([BEST_EPOCH], [BEST_DICE], s=90, zorder=5,
            facecolor="none", edgecolor="#1f77b4", linewidths=2.2)
ax1.annotate(f"Best Dice = {BEST_DICE:.4f}\n(epoch {BEST_EPOCH})",
             xy=(BEST_EPOCH, BEST_DICE), xytext=(36, 0.56),
             arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=1.2),
             fontsize=10, color="#1f77b4", fontweight="bold")

# Original independently recorded 5-epoch reference (NOT epoch 5 of this run)
ax1.axhline(y=ORIG_5EP, color="#2ca02c", linestyle=":", lw=1.8)
ax1.annotate(f"Independent 5-epoch run reference\nDice = {ORIG_5EP:.4f}",
             xy=(50, ORIG_5EP), xytext=(8, 0.34),
             color="#2ca02c", fontsize=10)

h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1 + h2, l1 + l2, loc="upper left", framealpha=0.95)

ax1.set_title("BraTS 2023 GLI 3D U-Net - 50-epoch training curve\n"
              "(validation Dice and training loss, per recorded epoch)")

plt.savefig(f"{FIG}/learning_curve.png")
plt.close()
print("wrote learning_curve.png")

# --------------------------------------------------------------------------- #
# 2. per_class_dice.png
# --------------------------------------------------------------------------- #
class_labels = {
    "class_1": "NCR (Necrotic / non-enhancing tumor core)",
    "class_2": "ED (Peritumoral edema / invaded tissue)",
    "class_3": "ET (Enhancing tumor)",
}
colors = {"class_1": "#9467bd", "class_2": "#ff7f0e", "class_3": "#17becf"}

fig, ax = plt.subplots(figsize=(9.5, 6.0))
for key, lab in class_labels.items():
    vals = m[key].astype(float)
    final_val = float(vals.iloc[-1])
    ax.plot(m["epoch"], vals, "-o", ms=4, lw=2, color=colors[key],
            label=f"{lab}  (final = {final_val:.4f})")
    ax.scatter([50], [final_val], s=80, facecolor="none",
               edgecolor=colors[key], linewidths=2)

ax.set_xlabel("Epoch")
ax.set_ylabel("Per-class Dice (foreground, 1 = perfect overlap)")
ax.set_xlim(1, 51)
ax.set_ylim(0.0, 1.0)
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.grid(True, which="major", alpha=0.3, linestyle="--")
ax.legend(loc="upper left", framealpha=0.95)
ax.set_title("BraTS 2023 GLI 3D U-Net - per-class validation Dice by epoch\n"
             "(no smoothing; recorded values)")

plt.savefig(f"{FIG}/per_class_dice.png")
plt.close()
print("wrote per_class_dice.png")

# --------------------------------------------------------------------------- #
# 3. gpu_telemetry.png
# --------------------------------------------------------------------------- #
tel = pd.read_csv(TEL, skipinitialspace=True)
t0 = datetime.strptime(tel["timestamp"].iloc[0], "%Y/%m/%d %H:%M:%S.%f")
times_min = [((datetime.strptime(ts, "%Y/%m/%d %H:%M:%S.%f") - t0).total_seconds()) / 60.0
             for ts in tel["timestamp"]]
tel["t_min"] = times_min

mem_gb = tel["memory_used_mb"] / 1024.0

# Downsampling for rendering only (documented below): use all samples, but to
# keep the file manageable the line renderer handles 6297 points natively.
DSTEP = 1
print(f"[gpu] plotting all {len(tel)} samples (no downsampling)")

fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

# (a) utilization %
ax = axes[0]
ax.plot(tel["t_min"], tel["utilization_gpu_pct"], lw=0.8, color="#1f77b4")
ax.set_ylabel("GPU utilization (%)")
ax.set_ylim(0, 105)
ax.grid(True, alpha=0.3, ls="--")

# (b) VRAM used
ax = axes[1]
ax.plot(tel["t_min"], mem_gb, lw=0.8, color="#2ca02c")
ax.set_ylabel("VRAM used (GB)")
ax.grid(True, alpha=0.3, ls="--")

# (c) power draw
ax = axes[2]
ax.plot(tel["t_min"], tel["power_draw_w"], lw=0.8, color="#d62728")
ax.set_ylabel("Power draw (W)")
ax.grid(True, alpha=0.3, ls="--")

# (d) temperature
ax = axes[3]
ax.plot(tel["t_min"], tel["temperature_c"], lw=0.8, color="#ff7f0e")
ax.set_ylabel("Temperature (°C)")
ax.set_xlabel("Time since first telemetry sample (minutes)")
ax.grid(True, alpha=0.3, ls="--")

axes[0].set_title("GPU telemetry - Tesla V100-SXM2-32GB, 50-epoch BraTS 2023 GLI run\n"
                  f"({len(tel)} recorded samples, time relative to first sample; "
                  "separate panels, each with its own units)")

for a in axes:
    a.autoscale(enable=True, axis="x", tight=True)

plt.tight_layout()
plt.savefig(f"{FIG}/gpu_telemetry.png")
plt.close()
print("wrote gpu_telemetry.png")
