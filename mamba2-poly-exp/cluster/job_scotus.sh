#!/bin/bash
#$ -q gpu@@jung_gpu
#$ -l gpu=1
#$ -pe smp 1
#$ -N fhe_scotus
#$ -cwd
#
#$ -o logs/$JOB_NAME.$JOB_ID.out
#$ -e logs/$JOB_NAME.$JOB_ID.err
#
# NO `#$ -M` / `#$ -m` DIRECTIVE ON PURPOSE -- pass it at submit time:
#     qsub -M you@nd.edu -m abe -v SCOTUS_STAGE=teacher cluster/job_scotus.sh
#     qsub -M you@nd.edu -m abe -v SCOTUS_STAGE=student,SP_Q=13 -hold_jid <teacher job> cluster/job_scotus.sh
#
# Hyunjung's SCOTUS experiment on Mamba-2 with Brandon's replacements (scotus/train.py).
# The teacher stage does not depend on any Brandon setting. The student stage takes
# SP_Q, the degree of Q in Delta = Q(x)^2 -- set it to his Phase A choice.
#
# PREFETCH FIRST, on crcfe01 (jobs run offline):
#     python -c "from scotus.data import load_split; [load_split(s) for s in ('train','validation','test')]"
: "${FHEMAMBA_CONDA_ENV:=$HOME/.conda/envs/pdpo}"
: "${HF_HOME:=/groups/tjung/$USER/FHEMAMBA/hf-cache}"
export FHEMAMBA_CONDA_ENV HF_HOME

source cluster/env.sh
source cluster/job_common.sh

: "${SCOTUS_STAGE:?set SCOTUS_STAGE=teacher or student via qsub -v}"
: "${SP_Q:=13}"
echo "[scotus] stage=$SCOTUS_STAGE  softplus Q degree=$SP_Q"
python -m scotus.train --stage "$SCOTUS_STAGE" --backend local --device cuda \
    --degree-softplus-q "$SP_Q" --outdir runs/scotus
echo "scotus $SCOTUS_STAGE finished at $(date)"
