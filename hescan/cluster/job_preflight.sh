#!/bin/bash
#$ -q gpu@@jung_gpu
#$ -l gpu=1
#$ -pe smp 8
#$ -N hescan_preflight
#$ -M YOUR_NETID@nd.edu
#$ -m abe
#$ -cwd
#$ -o logs/$JOB_NAME.$JOB_ID.out
#$ -e logs/$JOB_NAME.$JOB_ID.err
#
# PRE-FLIGHT: run this before any benchmark. From $FHE_ROOT:  qsub hescan/cluster/job_preflight.sh
#
#   1. which GPU did we get?
#   2. (re)build HEScan with the GPU backend
#   3. the toy-N correctness sweep on the CPU kernel
#   4. the same sweep on the GPU, each case also run on the CPU and compared
#      (max |m_GPU - m_CPU| and how many output ciphertexts are bit-identical)

set -uo pipefail
FAIL=0
step() { printf '\n=== %s ===\n' "$*"; }
check() { if [ "$1" -ne 0 ]; then echo ">>> STEP FAILED (rc=$1)"; FAIL=1; fi; }

source hescan/cluster/env.sh || exit 1

step "1. the GPU we were given"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,compute_cap --format=csv
check $?

step "2. build"
bash hescan/cluster/build.sh
check $?

step "3. CPU sweep"
bash hescan/run_tests.sh "$HESCAN_BUILD"
check $?

step "4. GPU sweep, checked against the CPU kernel"
HESCAN_ARGS="--gpu 1 --compare 1" bash hescan/run_tests.sh "$HESCAN_BUILD"
check $?

echo
if [ "$FAIL" -eq 0 ]; then echo "PRE-FLIGHT PASSED at $(date)"; else echo "PRE-FLIGHT FAILED at $(date)"; fi
exit $FAIL
