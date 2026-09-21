#!/bin/bash
#$ -q gpu@@jung_gpu
#$ -l gpu=1
#$ -pe smp 1
#$ -N fhe_norm_pathA
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
# PATH A -- learned constant divisor. ZERO ciphertext levels.
# Depends on job_norm_phase1.sh: submit with -hold_jid.
#
# DELIBERATE DEVIATION FROM THE BRIEF: fp32, not BF16.
# The brief asks for BF16. This node is a Quadro RTX 6000 (sm_75), which has NO
# bf16 tensor cores -- and torch.cuda.is_bf16_supported() returns True anyway,
# reporting driver support rather than hardware support. `auto` would resolve to
# fp16, and these trainers have no GradScaler, which is exactly how a run with
# 49 tiny log_c scalars silently learns nothing. fp32 costs us nothing here: the
# trainable set is ~56k parameters and peak memory was 11 GB of 22 GB.
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
        --micro-batch 2 --grad-accum 8 --precision-bits 20"

# Progressive: one operator group at a time, distilled between stages. Replacing
# all 49 at once from the pretrained checkpoint is the thing most likely to fail
# while telling you nothing about which site caused it.
for BUDGET in 100000 1000000; do
  for STAGE in A1 A2 A3; do
    echo; echo "############ Path A $STAGE, $BUDGET tokens ############"
    python norm/distill_norm.py --path A --stage "$STAGE" \
        --token-budget "$BUDGET" $COMMON \
        --outdir "runs/norm_pathA/${STAGE}_${BUDGET}tok"
  done
done

echo; echo "############ A3 with the auxiliary norm-matching term ############"
python norm/distill_norm.py --path A --stage A3 --token-budget 1000000 \
    --lambda-aux 1.0 $COMMON --outdir runs/norm_pathA/A3_1000000tok_aux1

echo "path A finished at $(date)"
