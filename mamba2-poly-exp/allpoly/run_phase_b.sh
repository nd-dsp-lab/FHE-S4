#!/bin/bash
#$ -q gpu@@jung_gpu
#$ -l gpu=1
#$ -pe smp 1
#$ -N fhe_phase_b
#$ -cwd
#$ -o logs/$JOB_NAME.$JOB_ID.out
#$ -e logs/$JOB_NAME.$JOB_ID.err
#
# Phase B: fine-tune with polynomials installed, against a control. See
# allpoly/finetune_phase_b.py for what one run does and why.
#
# Interactively (inside qrsh, inside tmux), from mamba2-poly-exp/:
#   bash allpoly/run_phase_b.sh smoke         # CPU-sized pipeline check, a few minutes
#   bash allpoly/run_phase_b.sh stage1        # softplus: control + one run per Q in QS
# Or as a batch job (the #$ lines above are read by qsub, ignored by bash):
#   qsub -V -M you@nd.edu -m abe -v STAGE=stage1 allpoly/run_phase_b.sh
#   (-V passes your environment; set FHEMAMBA_CONDA_ENV to your conda env first, see cluster/env.sh)
#
# Knobs (defaults in brackets):
#   QS ["5 13"]   softplus Q degrees. Q<=5 costs 5 levels, 6..13 cost 6; Phase A picked 13
#                 untrained. Q=13 here shows what fine-tuning buys at Phase A's cost.
#   LR [1e-4]  BUDGET [1000000]  DEV [cuda]  PY [python]
#   CALIB [allpoly/results/calib.pt]  Phase A's calibration, so stage 1 fits to the same ranges
#   OUT [runs/phase_b]  where runs go; ~0.5 GB each (model.pt). Group space if $HOME is full
#
# One control per recipe: it does not depend on the degree, so every Q compares to it.
# Results: runs/phase_b/<run>/{metrics.json, config.json, training_log.csv, model.pt, refit_calib.pt}
set -euo pipefail
if [ -n "${JOB_ID:-}" ] && [ -n "${SGE_O_WORKDIR:-}" ]; then
  # batch job: qsub ran a COPY of this script, so locate the repo from where it was submitted,
  # and take the lab's environment (FHEMAMBA_CONDA_ENV; submit with -V)
  cd "$SGE_O_WORKDIR"
  source cluster/env.sh
else
  cd "$(dirname "$0")/.."
fi
[ -d allpoly ] || { echo "run this from mamba2-poly-exp/"; exit 1; }
mkdir -p logs

PY=${PY:-python}
DEV=${DEV:-cuda}
LR=${LR:-1e-4}
BUDGET=${BUDGET:-1000000}
QS=${QS:-"5 13"}
CALIB=${CALIB:-allpoly/results/calib.pt}
STAGE=${1:-${STAGE:-stage1}}
OUT=${OUT:-runs/phase_b}
TAG="lr${LR}_${BUDGET}tok"

# Each run saves model.pt (~0.5 GB) so the next stage can start from it. $HOME on CRC
# has been ~96% full, and a full $HOME has killed jobs here before (cluster/env.sh).
# If space is short, put the runs in group space:  OUT=/groups/tjung/$USER/phase_b
mkdir -p "$OUT"
free_gb=$(df -Pk "$OUT" | awk 'NR==2 {printf "%d", $4/1048576}')
echo "[phase B] writing to $OUT ($free_gb GB free)"
if [ "$STAGE" != smoke ] && [ "$free_gb" -lt 5 ]; then
  echo "FATAL: under 5 GB free where runs go. Set OUT to group space, e.g. OUT=/groups/tjung/\$USER/phase_b"
  exit 1
fi

case "$STAGE" in
  smoke)
    $PY -m allpoly.finetune_phase_b --smoke --device cpu --ops softplus --no-save-model --outdir "$OUT/smoke_poly"
    $PY -m allpoly.finetune_phase_b --smoke --device cpu --control --no-save-model --outdir "$OUT/smoke_control"
    $PY -m allpoly.compare_phase_b "$OUT/smoke_poly" "$OUT/smoke_control"
    echo; echo "smoke done: the numbers mean nothing; it passed if both runs printed SMOKE TEST PASSED"
    ;;
  stage1)
    if [ ! -f "$CALIB" ]; then
      echo "WARNING: $CALIB not found -- it will be recomputed (64 x 2048 wikitext-2 train tokens)."
      echo "         Phase A's runs cached it there; if you ran Phase A in another folder, set CALIB."
    fi
    echo "######## CONTROL (exact model, same recipe)  lr=$LR budget=$BUDGET"
    $PY -m allpoly.finetune_phase_b --device "$DEV" --control --lr-backbone "$LR" \
        --token-budget "$BUDGET" --outdir "$OUT/s1_control_$TAG"
    for Q in $QS; do
      echo; echo "######## SOFTPLUS Q=$Q  lr=$LR budget=$BUDGET"
      $PY -m allpoly.finetune_phase_b --device "$DEV" --ops softplus --degree-softplus-q "$Q" \
          --calib-cache "$CALIB" --lr-backbone "$LR" --token-budget "$BUDGET" \
          --outdir "$OUT/s1_softplus_q${Q}_$TAG"
      echo; echo "######## VERDICT  softplus Q=$Q vs control"
      $PY -m allpoly.compare_phase_b "$OUT/s1_softplus_q${Q}_$TAG" "$OUT/s1_control_$TAG" \
          | tee "$OUT/s1_softplus_q${Q}_$TAG/verdict.txt"
    done
    echo; echo "stage 1 done $(date). Pick the softplus degree, then stage 2 starts from that run:"
    echo "  python -m allpoly.finetune_phase_b --ops softplus silu_norm --degree-softplus-q <Q> \\"
    echo "      --init-from $OUT/s1_softplus_q<Q>_$TAG --outdir $OUT/s2_..."
    echo "  python -m allpoly.finetune_phase_b --control --init-from $OUT/s1_control_$TAG --outdir $OUT/s2_control_$TAG"
    ;;
  *)
    echo "unknown stage '$STAGE' (smoke | stage1). Later stages are written out at the end of stage1."
    exit 1
    ;;
esac
