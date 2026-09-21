#!/bin/bash
#$ -q gpu@@jung_gpu
#$ -l gpu=1
#$ -pe smp 1
#$ -N fhemamba_preflight
#$ -M YOUR_NETID@nd.edu
#$ -m abe
#$ -cwd
#$ -o logs/$JOB_NAME.$JOB_ID.out
#$ -e logs/$JOB_NAME.$JOB_ID.err
#
# PRE-FLIGHT -- the cheapest job that can tell us whether anything else will work.
# Read-only: it trains nothing and writes nothing outside logs/ and runs/.
#
#   qsub cluster/job_preflight.sh
#
# It answers, in order:
#   1. did we land on the RTX 6000, and how much memory does it really have?
#   2. does the pre-built `pdpo` conda env import torch+CUDA and mamba_ssm?
#   3. does the whole test suite pass HERE, including the CUDA-only tests that
#      are skipped on the laptop? -> the parity gate is
#      TestOfficialSSDParity: our scan vs mamba_chunk_scan_combined
#   4. does the OFFICIAL backend load the 130M checkpoint (no silent fallback)?
#   5. does a short evaluation give the same perplexity the laptop got?

set -uo pipefail
FAIL=0
step() { printf '\n\033[1m=== %s ===\033[0m\n' "$*"; }
check() { if [ "$1" -ne 0 ]; then echo ">>> STEP FAILED (rc=$1)"; FAIL=1; fi; }

echo "=============================================================="
echo "job      : ${JOB_NAME:-?} (${JOB_ID:-?})   queue: ${QUEUE:-?}"
echo "host     : $(hostname)"
echo "started  : $(date)"
echo "cwd      : $(pwd)"
echo "=============================================================="

step "1. the GPU we were given"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi; check $?
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv

step "2. environment"
source cluster/env.sh
PY="$FHEMAMBA_CONDA_ENV/bin/python"
$PY - <<'PYEOF'
import torch, sys
print("python        ", sys.version.split()[0])
print("torch         ", torch.__version__)
print("cuda available", torch.cuda.is_available())
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print("device        ", p.name, f"{p.total_memory/1024**3:.1f} GB", f"sm_{p.major}{p.minor}")
import mamba_ssm; print("mamba_ssm     ", mamba_ssm.__version__)
try:
    import causal_conv1d; print("causal_conv1d ", causal_conv1d.__version__)
except ImportError:
    print("causal_conv1d  MISSING -- fine, real_mamba/nn_ref.py has a pure-torch conv")
PYEOF
check $?

step "3. full test suite, INCLUDING the CUDA-only parity tests"
# No -x: one early failure must not hide the parity result, which is the whole
# reason this job exists.
# -s so the parity tests' measured relative errors reach this log instead of
# being swallowed by pytest's capture on passing tests.
FHEMAMBA_TEST_CHECKPOINT=1 $PY -m pytest baby_mamba/tests real_mamba/tests \
    -v -s --no-header -rf --tb=short
check $?

step "4. official backend loads the real checkpoint (no silent fallback)"
$PY - <<'PYEOF'
import torch
from real_mamba.model import load_model
model, cfg, backend = load_model(backend="official", device="cuda", dtype=torch.float32)
assert backend == "official", f"fell back to {backend}"
n = sum(p.numel() for p in model.parameters())
print(f"OK: official backend, {n/1e6:.1f}M params, {cfg.n_layer} layers, d_model={cfg.d_model}")
print(f"peak GPU memory after load: {torch.cuda.max_memory_allocated()/1024**3:.3f} GB")
PYEOF
check $?

step "5. short evaluation -- does the GPU reproduce the laptop's number?"
$PY eval_poly_exp.py --transition poly4 --interval-mode per-head --pin-zero \
    --backend official --device cuda --blocks 12 --seq-len 1024 \
    --stats-json runs/part6_transition_stats/transition_stats.json \
    --outdir runs/preflight 2>&1 | tail -20
check $?

echo
echo "=============================================================="
if [ "$FAIL" -eq 0 ]; then
  echo "PRE-FLIGHT PASSED at $(date) -- safe to queue job_eval / job_finetune"
else
  echo "PRE-FLIGHT FAILED at $(date) -- do not queue anything else until this is fixed"
fi
echo "=============================================================="
exit $FAIL
