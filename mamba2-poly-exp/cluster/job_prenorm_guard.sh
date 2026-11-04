#!/bin/bash
#$ -q gpu@@jung_gpu
#$ -l gpu=1
#$ -pe smp 1
#$ -N fhe_prenorm_guard
#$ -cwd
#
#$ -o logs/$JOB_NAME.$JOB_ID.out
#$ -e logs/$JOB_NAME.$JOB_ID.err
#
# No `#$ -M` / `#$ -m` on purpose (see job_norm_floor.sh); submit with
#
#     qsub -M you@nd.edu -m abe cluster/job_prenorm_guard.sh
#
# The range guard for the OpenFHE 1/sqrt (GATES.md 9c/9d): two more comparisons
# clamp v to the table's range before the lookup, so no input reads past the table.
# Job 1658582 without it: the 1024-token / 0.05 range went to inf at every length,
# and one Newton step blew up shards even with the 8192-token / 0.5 range.
#   cal1k_m05_g     as benchmarked + guard: is calibration still needed at all?
#   cal8k_m05_g     8192-token range, back to margin 0.05 (narrower segments)
#   cal8k_m05_g_n1  the same with one Newton step (fewer levels)
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

mkdir -p eval/prenorm_tfhe
GATES="invsqrt_pre invsqrt_gated invsqrt_f invsqrt"
COMMON="--backend official --device cuda --shards 16 --lengths 512 2048 4096 8192 \
--calib-split train --invsqrt-impl openfhe --range-guard --gate $GATES"

run() {
  local tag=$1; shift
  echo "=== prenorm_guard: $tag ($*) started $(date) ==="
  python eval/seglut_paired.py $COMMON "$@" --out "eval/prenorm_tfhe/${tag}.json" \
      || echo "=== prenorm_guard: $tag FAILED (exit $?) ==="
}

run cal1k_m05_g     --calib-seq-len 1024 --margin 0.05 --newton 2
run cal8k_m05_g     --calib-seq-len 8192 --margin 0.05 --newton 2
run cal8k_m05_g_n1  --calib-seq-len 8192 --margin 0.05 --newton 1
echo "prenorm_guard job finished at $(date)"
