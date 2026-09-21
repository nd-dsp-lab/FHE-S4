#!/usr/bin/env bash
# Everything in this project, in the order Part 14 requires, as one script.
#
#   ./run_all.sh              # laptop-friendly: Stages 0-8 (~15 min on a CPU)
#   ./run_all.sh --with-training   # also Stage 9 + 10 (slow on CPU; use the GPU box)
#
# Each command is exactly what the README says to run, so you can also just copy
# them one at a time and watch what happens. That is the recommended way the
# first time.
set -euo pipefail

PY="${PY:-python}"
TRAIN=0
[[ "${1:-}" == "--with-training" ]] && TRAIN=1

step() { printf '\n\n\033[1m=== %s ===\033[0m\n\n' "$*"; }

step "tests (Stages 1-5 correctness; 63 pass, 6 need CUDA)"
$PY -m pytest baby_mamba/tests real_mamba/tests -q

step "Stage 1 -- see the transition (Part 1)"
$PY -m baby_mamba.demo_transition --poly 4

step "Stage 2/3 -- fit the polynomial and count its FHE depth (Parts 2-3)"
$PY fit_exp_polynomial.py --degree 4 --xmin -8 --xmax 0 --compare
$PY fit_exp_polynomial.py --degree 4 --xmin -4 --xmax 0 --method lobatto --pin-zero

step "Stage 4 -- error propagation through the recurrence (Part 4)"
$PY -m baby_mamba.error_propagation
$PY -m baby_mamba.error_propagation --method lobatto --pin-zero --outdir runs/part4_pinzero
$PY -m baby_mamba.error_propagation --delta-scale 6 --outdir runs/part4_deltascale6 --no-plot

step "Stage 6 -- MEASURE z on the real 130M model BEFORE choosing an interval (Part 6)"
$PY collect_transition_stats.py --blocks 32 --seq-len 1024

step "Stage 7 -- polynomial in, zero training (Part 7)"
$PY eval_poly_exp.py --sweep --seq-len 1024 --blocks 120

step "Stage 8 -- stability at 128/512/1024/2048 (Part 8)"
$PY stability_check.py --sweep --lengths 128 512 1024 2048 --blocks 2

if [[ $TRAIN -eq 1 ]]; then
  step "Stage 9 -- MODE A (Part 9)"
  $PY finetune_poly_exp.py --transition poly2 --interval-mode per-head --pin-zero \
      --mode A --token-budget 100000 --train-data wikitext2 --train-split train

  step "Stage 9 -- MODE B"
  $PY finetune_poly_exp.py --transition poly2 --interval-mode per-head --pin-zero \
      --mode B --token-budget 100000 --train-data wikitext2 --train-split train

  step "Stage 9 -- THE CONTROL: exact exp, trained identically. Do not skip this."
  $PY finetune_poly_exp.py --transition exact --mode A --token-budget 100000 \
      --train-data wikitext2 --train-split train \
      --outdir runs/part9_finetune/control_exact_modeA_100000tok

  step "Stage 10 -- distillation (Part 10)"
  $PY distill_poly_exp.py --transition poly2 --interval-mode per-head --pin-zero \
      --mode A --token-budget 100000 --lambda-kd 1.0 --temperature 2.0 \
      --lambda-transition 1.0 --train-data wikitext2 --train-split train
fi

step "Stage 11 -- collect everything into results_summary.csv (Part 11)"
$PY make_results_summary.py

printf '\n\033[1mDone.\033[0m Read results_summary.csv, then README.md Stage 12.\n'
