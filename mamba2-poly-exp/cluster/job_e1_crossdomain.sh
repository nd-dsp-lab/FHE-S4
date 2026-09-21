#!/bin/bash
#$ -q gpu@@jung_gpu
#$ -l gpu=1
#$ -pe smp 1
#$ -N fhe_e1_crossdom
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
