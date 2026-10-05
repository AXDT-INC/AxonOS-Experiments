#!/usr/bin/env python3
"""Recompute recorded metrics without GPUs. Python standard library only."""
import csv,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def load(name):
 rows=[json.loads(s) for s in (ROOT/'metrics'/name).read_text().splitlines() if s.strip()]
 start=next(r for r in rows if r.get('event')=='start')
 end=next(r for r in rows if r.get('event')=='end')
 reports=[]; prev=0; totals={}
 for r in rows:
  if 'interval_seconds' not in r: continue
  n=r['step']-prev
  if n<=0: raise ValueError('Non-increasing steps')
  reports.append(dict(step=r['step'],cells=r['cells'],steps=n,seconds=r['interval_seconds'],seconds_per_step=r['interval_seconds']/n))
  prev=r['step']
  for stage,values in r['work_items'].items():
   total=totals.setdefault(stage,[0]*len(values))
   for i,v in enumerate(values):total[i]+=v
 at=[r for r in reports if r['cells']==start['config']['max_cells']]
 return {'config':start,'end':end,'final_step':prev,'reported_interval_seconds':sum(r['seconds'] for r in reports),
         'seconds_per_step_intervals_ending_at_target':sum(r['seconds'] for r in at)/sum(r['steps'] for r in at),
         'work_items_sum':totals},reports
def telemetry(name,cutoff):
 result={}
 with (ROOT/'telemetry'/name).open() as f:
  for raw in csv.DictReader(f):
   r={k.strip():v.strip() for k,v in raw.items()}
   if r['timestamp']>=cutoff:continue
   idx=r['index']; u=float(r['utilization.gpu [%]'].split()[0]);mem=float(r['memory.used [MiB]'].split()[0])
   a=result.setdefault(idx,{'samples':0,'utilization_sum':0,'max_gpu_utilization_percent':0,'max_memory_mib':0})
   a['samples']+=1;a['utilization_sum']+=u;a['max_gpu_utilization_percent']=max(a['max_gpu_utilization_percent'],u);a['max_memory_mib']=max(a['max_memory_mib'],mem)
 for a in result.values():a['mean_gpu_utilization_percent']=a.pop('utilization_sum')/a['samples']
 return result
config=json.loads((ROOT/'config/benchmark.json').read_text())
single,rs=load('stress-one-gpu.jsonl');two,rt=load('stress-two-gpus.jsonl')
assert single['config']['config']==two['config']['config']
summary={'single':single,'two':two,'two_over_single_elapsed_ratio':two['reported_interval_seconds']/single['reported_interval_seconds'],
 'timing_definition':'Sum of logged reporting intervals, excluding setup. One run per configuration; exact historical commits unknown; trajectories differ.',
 'telemetry_window_definition':config['telemetry_cutoff_method'],
 'telemetry':{k:telemetry(k+'.csv',v) for k,v in config['telemetry_approximate_cutoffs'].items()}}
(ROOT/'metrics/comparison.json').write_text(json.dumps(summary,indent=2)+'\n')
with (ROOT/'metrics/intervals.csv').open('w',newline='') as f:
 writer=csv.DictWriter(f,lineterminator='\n',fieldnames=['run','step','cells','steps','seconds','seconds_per_step']);writer.writeheader()
 for label,rows in [('single',rs),('two',rt)]:
  writer.writerows(dict(run=label,**r) for r in rows)
print(json.dumps({'single_seconds':single['reported_interval_seconds'],'two_seconds':two['reported_interval_seconds'],'ratio':summary['two_over_single_elapsed_ratio']},indent=2))
