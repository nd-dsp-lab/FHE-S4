#!/bin/bash
#$ -q gpu@@jung_gpu
#$ -l gpu=1
#$ -pe smp 1
#$ -N fhemamba_parity
#$ -cwd
#
#$ -o logs/$JOB_NAME.$JOB_ID.out
#$ -e logs/$JOB_NAME.$JOB_ID.err
#
# NO `#$ -M` / `#$ -m` DIRECTIVE ON PURPOSE.
# CRC's qsub is a site wrapper that VALIDATES the address inside the script, so a
# placeholder like YOUR_NETID@nd.edu is rejected outright and a command-line -M
# does NOT override it. Editing the line in place works, but then every later
# `git pull` aborts with "local changes would be overwritten" -- which is exactly
# what happened, on the pull carrying a fix. With no directive here there is
# nothing to validate and nothing to edit, so pass it at submit time:
#
#     qsub -M you@nd.edu -m abe cluster/<this script>
#
# JOB 1 -- RUN THIS FIRST. Nothing else is worth trusting until it passes.
#
# It compares our pure-PyTorch reference forward against the OFFICIAL fused
# Triton kernel on the same weights. That test is skipped on the laptop because
# mamba-ssm imports Triton at import time and needs CUDA. See
# real_mamba/tests/test_parity.py::TestOfficialParity.
#
#   qsub cluster/job_parity.sh

source cluster/job_common.sh
source cluster/env.sh

echo "--- full test suite, including the CUDA-only parity tests ---"
FHEMAMBA_TEST_CHECKPOINT=1 python -m pytest baby_mamba/tests real_mamba/tests -v

echo
echo "--- confirm the OFFICIAL backend loads (no silent fallback) ---"
python - <<'PY'
import torch
from real_mamba.model import load_model
model, cfg, backend = load_model(backend="official", device="cuda", dtype=torch.float32)
assert backend == "official", backend
print(f"official backend OK: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")
PY

echo "parity job finished at $(date)"
