#!/bin/bash
#$ -q gpu@@jung_gpu
#$ -l gpu=1
#$ -pe smp 8
#$ -N hescan_bench
#$ -M YOUR_NETID@nd.edu
#$ -m abe
#$ -cwd
#$ -o logs/$JOB_NAME.$JOB_ID.out
#$ -e logs/$JOB_NAME.$JOB_ID.err
#
# Benchmarks, after job_preflight.sh has passed. From $FHE_ROOT:  qsub hescan/cluster/job_bench.sh
# N=2^15 and 2^16 at the default shape, plus the larger README shape, CPU vs GPU on the same
# packet. `qsub -v BENCH_COMPARE=0 ...` skips the (slow) CPU runs.
set -uo pipefail
source hescan/cluster/env.sh || exit 1
nvidia-smi --query-gpu=index,name,memory.total --format=csv
# log GPU memory once a second for the peak
nvidia-smi --query-gpu=timestamp,memory.used --format=csv -l 1 > "logs/hescan_bench.${JOB_ID:-local}.gpumem.csv" &
SMI=$!
bash hescan/cluster/build.sh && bash hescan/gpu_setup.sh bench
rc=$?
kill $SMI 2> /dev/null
echo "peak GPU memory (MiB): $(cut -d, -f2 "logs/hescan_bench.${JOB_ID:-local}.gpumem.csv" | grep -oE '[0-9]+' | sort -n | tail -1)"
exit $rc
