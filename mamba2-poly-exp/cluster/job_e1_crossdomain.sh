#!/bin/bash
#$ -q gpu@@jung_gpu
#$ -l gpu=1
#$ -pe smp 1
#$ -N fhe_e1_crossdom
#$ -M YOUR_NETID@nd.edu
#$ -m abe
#$ -cwd
#$ -o logs/$JOB_NAME.$JOB_ID.out
#$ -e logs/$JOB_NAME.$JOB_ID.err
#
# E1 -- THE experiment the whole claim rests on. Per-head intervals are measured
# on wikitext-103 ONLY, then evaluated on corpora they have never seen. If the
# intervals are a property of the model they will transfer; if they are a
# property of wikitext, this is where the result falls apart.
source cluster/env.sh
source cluster/job_common.sh

python cross_domain_eval.py \
    --model state-spaces/mamba2-130m --backend official --device cuda \
    --fit-data wikitext103 --fit-split train --fit-blocks 64 \
    --eval-data wikitext2 pile lambada --eval-blocks 100 \
    --degrees 2 3 4 --margin 0.25 --pin-zero \
    --outdir runs/e1_cross_domain
echo "E1 finished at $(date)"
