#!/bin/bash
#$ -q gpu@@jung_gpu
#$ -l gpu=1
#$ -pe smp 1
#$ -N fhemamba_kd
#$ -M YOUR_NETID@nd.edu
#$ -m abe
#$ -cwd
#$ -o logs/$JOB_NAME.$JOB_ID.out
#$ -e logs/$JOB_NAME.$JOB_ID.err
#
# JOB 4 -- Part 10. Run only after job_finetune.sh has produced numbers.
#
#   qsub cluster/job_distill.sh

source cluster/env.sh
source cluster/job_common.sh

BUDGET="${BUDGET:-1000000}"
# wikitext-2 train is 2,407,424 tokens and is already cached, so a 1M-token
# budget is 0.42 epochs -- no repetition, no download. Budgets above ~2M should
# switch to wikitext103 (pre-fetch it from crcfe01 first; $HOME was 95% full on
# 2026-09-19, so check `df -h $HOME` before pulling ~250 MB of parquet).
TRAIN_DATA="${TRAIN_DATA:-wikitext2}"
# fp32 for the same reason as job_finetune.sh: no GradScaler, tiny trainable set,
# and `auto` would pick fp16 on this sm_75 card.
COMMON="--backend official --device cuda --dtype float32 \
        --train-data "$TRAIN_DATA" --train-split train --eval-blocks 120 \
        --micro-batch 2 --grad-accum 8 --token-budget $BUDGET"

echo "=== logit distillation only ==="
python distill_poly_exp.py --transition poly2 --interval-mode per-head --pin-zero \
    --mode A $COMMON --lambda-kd 1.0 --temperature 2.0 --lambda-transition 0.0

echo "=== logit distillation + transition matching ==="
python distill_poly_exp.py --transition poly2 --interval-mode per-head --pin-zero \
    --mode A $COMMON --lambda-kd 1.0 --temperature 2.0 --lambda-transition 1.0

python make_results_summary.py
echo "distill job finished at $(date)"
