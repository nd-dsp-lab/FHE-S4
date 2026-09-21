#!/bin/bash
#$ -q gpu@@jung_gpu
#$ -l gpu=1
#$ -pe smp 1
#$ -N fhe_e3456_robust
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
# E3 long context, E4 LAMBADA accuracy, E5 margin sensitivity, E6 bootstrap CI.
# Uses the intervals E1 fitted on wikitext-103, so it must run AFTER E1.
source cluster/env.sh
source cluster/job_common.sh

python robustness_suite.py \
    --model state-spaces/mamba2-130m --backend official --device cuda \
    --stats-json runs/e1_cross_domain/stats_fitted_on_wikitext103.json \
    --experiments longctx lambada bootstrap margin \
    --degrees 2 3 4 --lengths 1024 2048 4096 8192 \
    --margins 0.0 0.1 0.25 0.5 1.0 \
    --eval-blocks 100 --batch-size 2 --lambada-examples 800 \
    --outdir runs/e3456_robustness
echo "E3-E6 finished at $(date)"
