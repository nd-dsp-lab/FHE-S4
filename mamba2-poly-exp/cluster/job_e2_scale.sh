#!/bin/bash
#$ -q gpu@@jung_gpu
#$ -l gpu=1
#$ -pe smp 1
#$ -N fhe_e2_scale
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
# E2 -- does the per-head story hold at other model scales, or is 130m special?
# For each size: measure z, then run the full zero-training sweep. The claim to
# test is that (a) |A| spans orders of magnitude at EVERY scale, (b) a single
# global interval fails at every scale, (c) per-head works at every scale.
source cluster/env.sh
source cluster/job_common.sh

for SIZE in 130m 370m 780m 1.3b; do
  echo
  echo "############### mamba2-$SIZE ###############"
  python collect_transition_stats.py \
      --model "state-spaces/mamba2-$SIZE" --backend official --device cuda \
      --data wikitext2 --split validation --blocks 24 --seq-len 1024 \
      --outdir "runs/e2_scale/$SIZE/stats" || { echo "STATS FAILED for $SIZE"; continue; }
  python eval_poly_exp.py --sweep \
      --model "state-spaces/mamba2-$SIZE" --backend official --device cuda \
      --seq-len 1024 --blocks 100 --batch-size 2 \
      --stats-json "runs/e2_scale/$SIZE/stats/transition_stats.json" \
      --outdir "runs/e2_scale/$SIZE/sweep" || echo "SWEEP FAILED for $SIZE"
done
echo "E2 finished at $(date)"
