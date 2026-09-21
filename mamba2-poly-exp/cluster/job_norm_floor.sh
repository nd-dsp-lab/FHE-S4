#!/bin/bash
#$ -q gpu@@jung_gpu
#$ -l gpu=1
#$ -pe smp 1
#$ -N fhe_norm_floor
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
# PHASE 0 / 0b -- the eval noise floor, and the PAIRED floor that is the correct
# test for an operator swap. Independent of phase 1, so it can run in parallel.
#
# Already measured on CPU at 8 shards: unpaired 2 sigma = 4.92 @512, 3.46 @2048.
# Under that yardstick three of the four gate results in GATES.md are within
# noise. But those comparisons are PAIRED, so the unpaired spread is too lenient
# -- this job measures the paired standard error at both lengths so the gate is
# applied correctly.
# qsub does not forward your shell environment unless you pass -V, so give the
# two paths a default here. Override by exporting them and submitting with -V.
: "${FHEMAMBA_CONDA_ENV:=$HOME/.conda/envs/pdpo}"
: "${HF_HOME:=/groups/tjung/$USER/FHEMAMBA/hf-cache}"
export FHEMAMBA_CONDA_ENV HF_HOME

source cluster/env.sh
source cluster/job_common.sh

echo "=== unpaired floor, more shards than CPU could manage ==="
python eval/noise_floor.py --backend official --device cuda \
    --shards 16 --lengths 512 2048 --batch-size 4 --out eval/noise_floor_gpu.json

echo "=== paired floor: every gate, both lengths ==="
python eval/paired_significance.py --backend official --device cuda \
    --gate exp softplus silu_norm silu_conv --shards 16 --lengths 512 2048 \
    --out eval/paired_significance_gpu.json
echo "floor job finished at $(date)"
