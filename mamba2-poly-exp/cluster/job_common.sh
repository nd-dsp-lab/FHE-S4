#!/bin/bash
# Shared CRC/SGE settings for this project. Sourced by every job script here so
# the queue, GPU request and notification address live in exactly one place.
#
#   queue   gpu@@jung_gpu        the Jung group's GPU hostgroup (node rtx6k-019,
#                                RTX 6000 = 24 GB, 4 cards on the node)
#   gpu=1   this project is sized for ONE 24 GB card. Raise it only if you have a
#           reason; nothing in this repo uses more than one.
#   email   passed at submit time as `qsub -M you@nd.edu -m abe`, NOT a directive
#
# Do not put `#$` directives in here -- SGE only reads them from the script it was
# given. They are duplicated at the top of each job script on purpose.

set -euo pipefail

export FHEMAMBA_QUEUE="gpu@@jung_gpu"

# FHEMAMBA_ROOT is normally exported by env.sh, which every job script sources
# first. Derive it here too so this file also works on its own -- e.g. inside an
# interactive `qrsh` session, which is what the handoff instructions tell a new
# user to do.
#
# This line used to read `${FHEMAMBA_ROOT:-${FHEMAMBA_ROOT}}`: a default that
# refers to the variable it is defaulting, left behind when personal paths were
# scrubbed out of this repo. It expands to the empty string when unset, and
# under the `set -u` that env.sh turns on it aborts the job outright. The same
# accident in env.sh killed two submitted jobs; this was the second copy.
_JOB_COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
export FHEMAMBA_ROOT="${FHEMAMBA_ROOT:-$(dirname "$_JOB_COMMON_DIR")}"

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
