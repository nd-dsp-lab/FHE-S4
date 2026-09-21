#!/bin/bash
#$ -q gpu@@jung_gpu
#$ -l gpu=1
#$ -pe smp 1
#$ -N fhe_norm_p1
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
# PHASE 1 -- what the three RMSNorm instances actually see. RUN THIS FIRST:
# both Path A (c init) and Path B (prescale) read its output, and it may kill
# Path B outright. The Newton convergence basin was measured at v/s in
# [0.25, 2.0] -- only 8x wide -- so if the WITHIN-layer p99/p1 of v exceeds 8,
# a static per-layer prescale cannot work regardless of how well it is trained.
# qsub does not forward your shell environment unless you pass -V, so give the
# two paths a default here. Override by exporting them and submitting with -V.
: "${FHEMAMBA_CONDA_ENV:=$HOME/.conda/envs/pdpo}"
: "${HF_HOME:=/groups/tjung/$USER/FHEMAMBA/hf-cache}"
export FHEMAMBA_CONDA_ENV HF_HOME

source cluster/env.sh
source cluster/job_common.sh

python norm/collect_norm_stats.py --backend official --device cuda \
    --lengths 512 2048 --blocks 48 --out norm/norm_stats.json
echo "phase 1 finished at $(date)"
