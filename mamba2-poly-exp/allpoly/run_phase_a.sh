#!/bin/bash
# Phase A of Step 9: MEASURE, no training. See allpoly/README.md.
#
#   bash allpoly/run_phase_a.sh                          # steps 0-2: per-operator degree sweep
#   SILU_D=27 SP_Q=13 bash allpoly/run_phase_a.sh combined   # step 3, with the degrees you picked
#
# Environment knobs (defaults in brackets):
#   PY [python]  DEV [cuda]  SHARDS [8]  LENGTHS ["512 2048"]
# Run from anywhere; results land in allpoly/results/. Calibration is computed
# once (wikitext-2 TRAIN) and cached, so every run is fitted to the same ranges.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
DEV=${DEV:-cuda}
SHARDS=${SHARDS:-8}
LENGTHS=${LENGTHS:-"512 2048"}
OUT=allpoly/results
CAL=$OUT/calib.pt
mkdir -p "$OUT"
run() { echo; echo "######## $*"; $PY -m allpoly.eval_allpoly --device "$DEV" --shards "$SHARDS" \
          --lengths $LENGTHS --calib-cache "$CAL" "$@"; }

if [ "${1:-sweep}" = "combined" ]; then
  : "${SILU_D:?set SILU_D to the SiLU degree you picked}" "${SP_Q:?set SP_Q to the softplus Q degree you picked}"
  # step 3: the chosen degrees together, then with bootstrap noise
  #   11.35 bits = mean-abs precision after one bootstrap, uniform secret, N=2^17 (fhe/EPOCH.md)
  #   18.23 bits = same, sparse secret
  run --degree-silu "$SILU_D" --degree-softplus-q "$SP_Q" --noise-bits 11.35 18.23 \
      --configs gates exp+noise@11.35 gates+noise@11.35 exp+noise@18.23 gates+noise@18.23 \
      --out "$OUT/A3_gates_silu${SILU_D}_q${SP_Q}.json"
  exit 0
fi

# step 0: the reference. Known answer: the exp gate is indistinguishable from exact.
#         If it is not, stop -- the setup differs and nothing below is trustworthy.
run --noise-bits --configs only:exp --out "$OUT/A0_exp.json"

# step 1: SiLU, both sites, degree sweep
for d in 7 13 27 59; do
  run --degree-silu "$d" --noise-bits --configs only:silu_conv only:silu_norm --out "$OUT/A1_silu_d$d.json"
done

# step 2: softplus as Delta = Q(x)^2, degree of Q
for q in 5 13 27; do
  run --degree-softplus-q "$q" --noise-bits --configs only:softplus --out "$OUT/A2_softplus_q$q.json"
done
echo; echo "sweep done -- pick degrees, then: SILU_D=.. SP_Q=.. bash allpoly/run_phase_a.sh combined"
