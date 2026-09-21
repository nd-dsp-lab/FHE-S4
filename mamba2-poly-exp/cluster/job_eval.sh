#!/bin/bash
#$ -q gpu@@jung_gpu
#$ -l gpu=1
#$ -pe smp 1
#$ -N fhemamba_eval
#$ -cwd
#
#$ -o logs/$JOB_NAME.$JOB_ID.out
#$ -e logs/$JOB_NAME.$JOB_ID.err
#
# NO `#$ -M` / `#$ -m` DIRECTIVE ON PURPOSE.
# CRC's qsub is a site wrapper that VALIDATES the address inside the script, so a
# placeholder like YOUR_NETID@nd.edu is rejected outright and a command-line -M
# does NOT override it. Editing the line in place works, but then every later
# `git pull` aborts with "local changes would be overwritten" -- which is exactly
# what happened, on the pull carrying a fix. With no directive here there is
# nothing to validate and nothing to edit, so pass it at submit time:
#
#     qsub -M you@nd.edu -m abe cluster/<this script>
#
# JOB 2 -- reproduce Parts 6/7/8 on the GPU, with the OFFICIAL backend and real
# memory numbers. On the laptop these ran on CPU, so every peak_gpu_memory_gb in
# the repo is currently nan.
#
#   qsub cluster/job_eval.sh

source cluster/env.sh
source cluster/job_common.sh

BACKEND="${BACKEND:-official}"

echo "--- disk check (home was 95% full on 2026-09-19) ---"
df -h "$HOME" | tail -1
AVAIL_GB=$(df -BG --output=avail "$HOME" | tail -1 | tr -dc '0-9')
if [ "${AVAIL_GB:-0}" -lt 2 ]; then
  echo "ABORTING: only ${AVAIL_GB}G free in \$HOME. Free space before running this."
  exit 1
fi

echo "--- Part 6: measure the real z distribution (this picks the intervals) ---"
# Keep the laptop-derived stats: the per-head intervals in the committed results
# came from them, and overwriting silently would make old and new numbers
# incomparable without anyone noticing.
if [ -f runs/part6_transition_stats/transition_stats.json ] && \
   [ ! -f runs/part6_transition_stats/transition_stats.laptop.json ]; then
  cp runs/part6_transition_stats/transition_stats.json \
     runs/part6_transition_stats/transition_stats.laptop.json
  echo "    (kept the previous stats as transition_stats.laptop.json)"
fi
python collect_transition_stats.py --backend "$BACKEND" --device cuda \
    --blocks 64 --seq-len 1024

echo "--- Part 7: polynomial in, zero training, full validation set ---"
python eval_poly_exp.py --sweep --backend "$BACKEND" --device cuda \
    --seq-len 1024 --batch-size 4

echo "--- Part 8: stability up to 2048 ---"
python stability_check.py --sweep --backend "$BACKEND" --device cuda \
    --lengths 128 512 1024 2048 --blocks 8

python make_results_summary.py
echo "eval job finished at $(date)"
