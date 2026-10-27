#!/bin/bash
#$ -q gpu@@jung_gpu
#$ -l gpu=1
#$ -pe smp 1
#$ -N fhe_seglut
#$ -cwd
#
#$ -o logs/$JOB_NAME.$JOB_ID.out
#$ -e logs/$JOB_NAME.$JOB_ID.err
#
# No `#$ -M` / `#$ -m` on purpose (see job_norm_floor.sh); submit with
#
#     qsub -M you@nd.edu -m abe cluster/job_seglut.sh
#
# ROSETTA-style segmented LUTs (real_mamba/seglut.py) on every gate a polynomial
# could not handle, PAIRED against the exact model -- same shards and verdict
# rule as eval/paired_significance.py, so the rows sit next to GATES.md.
#
# Four configurations, one question each:
#   clamp            the idealised protocol: inputs clamped into the table.
#                    Does the TABLE work?
#   wrap             what ROSETTA's protocol literally computes off the table
#                    (negacyclic blind rotation flips the sign). Does the
#                    off-interval failure come back?
#   clamp + 12b abs  noise at the bottom of ROSETTA's reported 12-19 bits
#   clamp + 12b rng  the same noise scaled to each table's output range, which
#                    is how a fixed-point TFHE encoding actually behaves
: "${FHEMAMBA_CONDA_ENV:=/groups/tjung/$USER/envs/fhemamba}"
: "${HF_HOME:=/groups/tjung/$USER/FHEMAMBA/hf-cache}"
export FHEMAMBA_CONDA_ENV HF_HOME

source cluster/env.sh
source cluster/job_common.sh

# 24 MB, excluded by the root .gitignore's `*.json`; regenerate if absent
if [ ! -s runs/gate_stats/gate_stats.json ]; then
  echo "=== runs/gate_stats/gate_stats.json missing -- collecting it first ==="
  python collect_gate_stats.py --backend official --device cuda --dtype float16 \
      --blocks 16 --outdir runs/gate_stats
fi

mkdir -p eval/seglut
GATES="invsqrt_pre invsqrt_gated invsqrt_f invsqrt fused_dt softplus silu_conv silu_norm all"
COMMON="--backend official --device cuda --shards 16 --lengths 512 2048 --gate $GATES"

run() {
  local tag=$1; shift
  echo "=== seglut: $tag ($*) started $(date) ==="
  python eval/seglut_paired.py $COMMON "$@" --out "eval/seglut/${tag}.json"
}

run clamp          --oob clamp
run wrap           --oob wrap
run clamp_n12_abs  --oob clamp --noise-bits 12 --noise-scale abs
run clamp_n12_rng  --oob clamp --noise-bits 12 --noise-scale range
echo "seglut job finished at $(date)"
