#!/bin/bash
# Shared CRC/SGE settings for this project. Sourced by every job script here so
# the queue, GPU request and notification address live in exactly one place.
#
#   queue   gpu@@jung_gpu        the Jung group's GPU hostgroup (node rtx6k-019,
#                                RTX 6000 = 24 GB, 4 cards on the node)
#   gpu=1   this project is sized for ONE 24 GB card. Raise it only if you have a
#           reason; nothing in this repo uses more than one.
#   email   YOUR_NETID@nd.edu
#
# Do not put `#$` directives in here -- SGE only reads them from the script it was
# given. They are duplicated at the top of each job script on purpose.

set -euo pipefail

export FHEMAMBA_QUEUE="gpu@@jung_gpu"
export FHEMAMBA_EMAIL="YOUR_NETID@nd.edu"
export FHEMAMBA_ROOT="${FHEMAMBA_ROOT:-${FHEMAMBA_ROOT}}"

echo "=============================================================="
echo "host      : $(hostname)"
echo "job       : ${JOB_NAME:-interactive} (${JOB_ID:-n/a})"
echo "queue     : ${QUEUE:-n/a}"
echo "started   : $(date)"
echo "cwd       : $(pwd)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "=============================================================="
nvidia-smi || echo "WARNING: nvidia-smi failed -- did the job land on a GPU node?"
echo "=============================================================="
