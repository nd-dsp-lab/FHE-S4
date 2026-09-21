#!/bin/bash
#$ -q gpu@@jung_gpu
#$ -l gpu=1
#$ -pe smp 1
#$ -N fhemamba_ctrlC
#$ -M YOUR_NETID@nd.edu
#$ -m abe
#$ -cwd
#$ -o logs/$JOB_NAME.$JOB_ID.out
#$ -e logs/$JOB_NAME.$JOB_ID.err
#
# The control that job 1458347 was missing: exact exp + MODE C (LoRA), trained
# identically to the poly2 MODE C run. Without it, poly2/MODE C's 16.78 perplexity
# cannot be attributed -- 1.24M LoRA parameters adapting to wikitext would produce
# a large drop whether or not the transition is a polynomial.
source cluster/env.sh
source cluster/job_common.sh

BUDGET="${BUDGET:-1000000}"
python finetune_poly_exp.py --transition exact --mode C \
    --backend official --device cuda --dtype float32 \
    --train-data wikitext2 --train-split train \
    --eval-data wikitext2 --eval-split validation \
    --eval-blocks 120 --micro-batch 2 --grad-accum 8 \
    --token-budget "$BUDGET" \
    --outdir "runs/part9_finetune/control_exact_modeC_${BUDGET}tok"
python make_results_summary.py
echo "mode-C control finished at $(date)"
