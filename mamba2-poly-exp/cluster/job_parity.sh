#!/bin/bash
#$ -q gpu@@jung_gpu
#$ -l gpu=1
#$ -pe smp 1
#$ -N fhemamba_parity
#$ -M YOUR_NETID@nd.edu
#$ -m abe
#$ -cwd
#$ -o logs/$JOB_NAME.$JOB_ID.out
#$ -e logs/$JOB_NAME.$JOB_ID.err
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
