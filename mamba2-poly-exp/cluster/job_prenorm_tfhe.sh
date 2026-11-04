#!/bin/bash
#$ -q gpu@@jung_gpu
#$ -l gpu=1
#$ -pe smp 1
#$ -N fhe_prenorm_tfhe
#$ -cwd
#
#$ -o logs/$JOB_NAME.$JOB_ID.out
#$ -e logs/$JOB_NAME.$JOB_ID.err
#
# No `#$ -M` / `#$ -m` on purpose (see job_norm_floor.sh); submit with
#
#     qsub -M you@nd.edu -m abe cluster/job_prenorm_tfhe.sh
#
# RMSNorm's 1/sqrt replaced by the pipeline fhe/seglut_invsqrt/bench.cpp runs in
# OpenFHE (real_mamba/ss_invsqrt.py: ~5.5-bit scheme-switched table, then Newton
# in CKKS), staged pre-norms -> gated -> norm_f -> all 49, out to L = 8192.
#
# A CPU smoke run (2 shards) found the long-context failure: gated-norm v at
# L = 8192 exceeds a range calibrated on 1024-token blocks, an input past the
# table wraps to a reflected entry, Newton diverges, and all 49 norms go to inf.
# Calibrating on 8192-token blocks with a 0.5 log-margin removed every wrap.
#   cal1k_m05     as benchmarked: 6 x 1024 calibration, margin 0.05
#   cal8k_m50     the fix: 6 x 8192 calibration, margin 0.5
#   cal8k_m50_n1  the fix with one Newton step (3 levels fewer)
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
--calib-split train --invsqrt-impl openfhe --gate $GATES"

run() {
  local tag=$1; shift
  echo "=== prenorm_tfhe: $tag ($*) started $(date) ==="
  # job 1655430 lost all three runs to one crash in the first; keep going
  python eval/seglut_paired.py $COMMON "$@" --out "eval/prenorm_tfhe/${tag}.json" \
      || echo "=== prenorm_tfhe: $tag FAILED (exit $?) ==="
}

run cal1k_m05     --calib-seq-len 1024 --margin 0.05 --newton 2
run cal8k_m50     --calib-seq-len 8192 --margin 0.5  --newton 2
run cal8k_m50_n1  --calib-seq-len 8192 --margin 0.5  --newton 1
echo "prenorm_tfhe job finished at $(date)"
