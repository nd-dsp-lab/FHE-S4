#!/bin/bash
#$ -q gpu@@jung_gpu
#$ -l gpu=1
#$ -pe smp 1
#$ -N fhemamba_ft
#$ -M YOUR_NETID@nd.edu
#$ -m abe
#$ -cwd
#$ -o logs/$JOB_NAME.$JOB_ID.out
#$ -e logs/$JOB_NAME.$JOB_ID.err
#
# JOB 3 -- Part 9 at the budgets the laptop could not reach, plus MODE C, plus
# the exact-exp control at every budget. The control is not optional: without it
# you cannot separate "adapted to the polynomial" from "any 1,728 parameters help
# on this eval set".
#
#   qsub cluster/job_finetune.sh                  # 1M tokens
#   qsub -v BUDGET=5000000 cluster/job_finetune.sh

source cluster/env.sh
source cluster/job_common.sh

BUDGET="${BUDGET:-1000000}"
# wikitext-2 train is 2,407,424 tokens and is already cached, so a 1M-token
# budget is 0.42 epochs -- no repetition, no download. Budgets above ~2M should
# switch to wikitext103 (pre-fetch it from crcfe01 first; $HOME was 95% full on
# 2026-09-19, so check `df -h $HOME` before pulling ~250 MB of parquet).
TRAIN_DATA="${TRAIN_DATA:-wikitext2}"
# --dtype float32, deliberately, NOT auto:
#   * `auto` resolves to fp16 on this sm_75 card, and these scripts have no
#     GradScaler. MODE A trains 1,728 parameters with small gradients, which is
#     exactly the regime where fp16 gradients underflow to zero and the run
#     silently learns nothing.
#   * we are not memory-bound: only ~1.7k-3.5k parameters are trainable, so the
#     optimiser state is negligible and fp32 weights are 516 MB of a 22 GB card.
# fp16 is only forced on us for the OFFICIAL Triton SSD kernel, which the patched
# forward never calls -- see real_mamba/model.recommended_dtype.
# micro-batch 2 (not 4): the reference SSD retains a (B,H,nchunks,Q,Q) segprod
# tensor per layer for backward, so activation memory scales with batch.
COMMON="--backend official --device cuda --dtype float32 \
        --train-data "$TRAIN_DATA" --train-split train \
        --eval-data wikitext2 --eval-split validation \
        --eval-blocks 120 --micro-batch 2 --grad-accum 8 \
        --token-budget $BUDGET"

for MODE in A B C; do
  echo "=== Part 9 MODE $MODE, $BUDGET tokens ==="
  python finetune_poly_exp.py --transition poly2 --interval-mode per-head --pin-zero \
      --mode "$MODE" $COMMON
done

# ONE CONTROL PER MODE THAT CAN HAVE ONE.
# A single MODE-A control is not enough. MODE C trains 1.24M LoRA parameters, and
# comparing that against a 1,728-parameter control would credit the polynomial for
# a drop that LoRA produced by ordinary domain adaptation. MODE B has no control:
# its extra parameters ARE the polynomial coefficients, which do not exist for
# exact exp, so MODE B is compared against MODE A instead.
for CMODE in A C; do
  echo "=== THE CONTROL: exact exp, MODE $CMODE, trained identically ==="
  python finetune_poly_exp.py --transition exact --mode "$CMODE" $COMMON \
      --outdir "runs/part9_finetune/control_exact_mode${CMODE}_${BUDGET}tok"
done

python make_results_summary.py
echo "finetune job finished at $(date)"
