#!/bin/bash
#$ -q gpu@@jung_gpu
#$ -l gpu=1
#$ -pe smp 1
#$ -N fhe_norm_pathB
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
# PATH B -- prescaled Newton inverse sqrt. Depends on job_norm_phase1.sh.
#
# DEPTH, DERIVED FROM THE GRAPH (not the brief's estimate of 2/step):
#   t=1 -> 2 levels   the first step is FREE, because y0 is a plaintext
#                     constant so y0^2 is plaintext and the whole step is ct x pt
#   t=2 -> 5 levels   each later step costs 3: y^2, then v'*y^2, then y*(...)
#   t=3 -> 8 levels
#
# CONVERGENCE BASIN, MEASURED: v/s in [0.25, 2.0] with y0=1. Outside it the
# iteration DIVERGES and MORE steps diverge FASTER (at v'=5: 54 -> 1.3e5 ->
# 7.9e14). So this is a range problem, not an accuracy problem, and the range
# penalty is the mechanism -- not the iteration count.
# qsub does not forward your shell environment unless you pass -V, so give the
# two paths a default here. Override by exporting them and submitting with -V.
: "${FHEMAMBA_CONDA_ENV:=$HOME/.conda/envs/pdpo}"
: "${HF_HOME:=/groups/tjung/$USER/FHEMAMBA/hf-cache}"
export FHEMAMBA_CONDA_ENV HF_HOME

source cluster/env.sh
source cluster/job_common.sh

COMMON="--backend official --device cuda --dtype float32 \
        --norm-stats norm/norm_stats.json --stats-length 512 \
        --train-data wikitext103 --train-split train \
        --eval-data wikitext2 --eval-split validation \
        --eval-lengths 512 2048 --eval-blocks 60 \
        --micro-batch 2 --grad-accum 8 --precision-bits 20 \
        --token-budget 1000000"

for T in 1 2 3; do
  for LAM in 0.0 0.1 1.0; do
    echo; echo "########### Path B t=$T lambda_range=$LAM ###########"
    python norm/distill_norm.py --path B --stage A3 --t-steps "$T" \
        --lambda-range "$LAM" $COMMON \
        --outdir "runs/norm_pathB/t${T}_lam${LAM}"
  done
done
echo "path B finished at $(date)"
