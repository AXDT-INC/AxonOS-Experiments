#!/usr/bin/env python3
"""Render figures from analyze.py outputs; requires matplotlib."""
import csv,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/figures';OUT.mkdir(parents=True,exist_ok=True)
s=json.loads((ROOT/'metrics/comparison.json').read_text())
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
fig,ax=plt.subplots(figsize=(9,5.5)); vals=[s[k]['reported_interval_seconds']/60 for k in ('single','two')]
bars=ax.bar(['1 GPU\nLegacy single-device path','2 GPUs\nDistributed path'],vals,color=['#126782','#d77a27'],width=.55)
ax.bar_label(bars,labels=[f'{v:.2f} min' for v in vals],padding=6);ax.set_ylim(0,max(vals)*1.18);ax.set_ylabel('Sum of recorded intervals (minutes)');ax.set_title('Both runs reached and held 100,000 cells',loc='left',fontweight='bold')
fig.text(.10,.015,'One run per configuration • Setup excluded • Historical commit hashes not recorded',fontsize=9)
fig.tight_layout(rect=[0,.04,1,1]);fig.savefig(OUT/'runtime_comparison.png',dpi=180);plt.close(fig)
with (ROOT/'metrics/intervals.csv').open() as f: rows=list(csv.DictReader(f))
fig,ax=plt.subplots(figsize=(9,5.5))
for label,color,title in [('single','#126782','1 GPU'),('two','#d77a27','2 GPUs')]:
 r=[v for v in rows if v['run']==label];ax.plot([int(v['cells']) for v in r],[float(v['seconds_per_step']) for v in r],label=title,color=color,linewidth=2)
ax.set_xlabel('Cells at end of reporting interval');ax.set_ylabel('Seconds per simulation step');ax.set_title('Early timing baseline for throughput optimization',loc='left',fontweight='bold');ax.legend();ax.grid(alpha=.2)
fig.text(.10,.015,'Different simulation trajectories; reporting intervals are not identical workloads.',fontsize=9)
fig.tight_layout(rect=[0,.04,1,1]);fig.savefig(OUT/'scaling.png',dpi=180)
