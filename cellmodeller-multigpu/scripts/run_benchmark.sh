#!/usr/bin/env bash
# Usage: bash scripts/run_benchmark.sh /opt/CellModeller 0 /absolute/new/output-dir
# Repeat with 0,1 and a DIFFERENT output directory, on the same unchanged checkout.
set -euo pipefail
checkout=${1:?CellModeller checkout required}; devices=${2:?device indices required}; output=${3:?new output directory required}
[[ "$output" = /* ]] || { echo 'Use an absolute output path' >&2; exit 1; }
mkdir "$output"  # Refuse an existing directory; preserve previous evidence.
cd "$checkout"
git rev-parse HEAD > "$output/commit.txt"
git status --short > "$output/git-status.txt"
git diff > "$output/working-tree.patch"
python -m pip freeze > "$output/python-packages.txt"
python -V > "$output/python-version.txt" 2>&1
nvidia-smi -q > "$output/nvidia-smi.txt"
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,power.draw --format=csv -l 1 > "$output/gpu-telemetry.csv" &
telemetry_pid=$!
trap 'kill "$telemetry_pid" 2>/dev/null || true; wait "$telemetry_pid" 2>/dev/null || true' EXIT
python Examples/multigpu_stress.py --platform 0 --devices "$devices" \
 --gpu-memory partitioned --max-cells 100000 --initial-cells 1 \
 --max-contacts 32 --max-sqs 400000 --species 4 --seed 12345 \
 --growth-rate 2.0 --memory-fraction 0.5 --report-every 10 --dt 0.025 \
 --steps 10000 --hold-steps 100 --log "$output/stress.jsonl" 2>&1 | tee "$output/console.log"
