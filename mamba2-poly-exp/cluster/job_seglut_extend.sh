#!/bin/bash
#$ -q gpu@@jung_gpu
#$ -l gpu=1
#$ -pe smp 1
#$ -N fhe_seglut_ext
#$ -cwd
#
#$ -o logs/$JOB_NAME.$JOB_ID.out
#$ -e logs/$JOB_NAME.$JOB_ID.err
#
# No `#$ -M` / `#$ -m` on purpose (see job_norm_floor.sh); submit with
#
#     qsub -M you@nd.edu -m abe cluster/job_seglut_extend.sh
#
# Follow-up to job_seglut.sh. There, clamping cost both SiLUs +0.1 to +0.35 ppl
# with tables accurate to >= 21 bits, and a small CPU run had `extend` (linear
# tails past the table) cut that 20-90x. This confirms it at full scale.
#
# --calib-split train moves the 1/sqrt(v) range calibration off the evaluation
# split. The SiLU / softplus / fused-dt tables still take their ranges from
# runs/gate_stats/gate_stats.json (validation), like the polynomial fits did.
#   extend           does `extend` fix the SiLUs at 16 shards x both lengths?
#   extend + 12b abs the same, with noise at the bottom of ROSETTA's 12-19 bits
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
COMMON="--backend official --device cuda --shards 16 --lengths 512 2048 --calib-split train --gate $GATES"

run() {
  local tag=$1; shift
  echo "=== seglut: $tag ($*) started $(date) ==="
  python eval/seglut_paired.py $COMMON "$@" --out "eval/seglut/${tag}.json"
}

run extend_trcal          --oob extend
run extend_trcal_n12_abs  --oob extend --noise-bits 12 --noise-scale abs
echo "seglut extend job finished at $(date)"
