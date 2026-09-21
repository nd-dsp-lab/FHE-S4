#!/bin/bash
# The environment for every FHEMAMBA job on CRC. Sourced by all job scripts.
#
# THIS SCRIPT NEVER CREATES A CONDA ENV OR INSTALLS ANYTHING.
#
# An earlier version of this file did `conda create` + `pip install torch`. On
# 2026-09-19 that ran on a $HOME with 5.5 GB free, built 3.6 GB of a ~7 GB
# environment, filled the filesystem to 0 bytes, and killed the job with
# `OSError: [Errno 28] No space left on device`. A full $HOME also endangers
# every other job of Jiachen's that is running at the time. So: no installs from
# inside a batch job, ever. If something is missing, this script says so and
# exits, and a human installs it deliberately.
#
# We reuse ${FHEMAMBA_CONDA_ENV}, which already has everything:
#   python 3.10.18, torch 2.3.1+cu121, triton 2.3.1, mamba_ssm 2.2.2,
#   transformers, einops, pyarrow, numpy.
# `causal_conv1d` is absent and that is fine -- real_mamba/nn_ref.py has a
# pure-torch conv, and the patched forward never calls the official conv path.

set -uo pipefail

FHEMAMBA_CONDA_ENV="${FHEMAMBA_CONDA_ENV:-${FHEMAMBA_CONDA_ENV}}"
PY="$FHEMAMBA_CONDA_ENV/bin/python"

if [ ! -x "$PY" ]; then
  echo "FATAL: no python at $PY"
  echo "       Set FHEMAMBA_CONDA_ENV to an env that has torch + mamba_ssm."
  echo "       Do NOT let a batch job build one -- see the note at the top of this file."
  return 1 2>/dev/null || exit 1
fi

export PATH="$FHEMAMBA_CONDA_ENV/bin:$PATH"

# HF cache lives in the GROUP space, not $HOME.
#   /groups  : 5.0 T, 4% used
#   $HOME    : 100 G, 96% used
# It is also a cache of our OWN, not the shared ~/.cache/huggingface: the
# DP-GRPO project sets HF_HOME itself in several of its job scripts, and sharing
# a cache across projects on a nearly-full filesystem is how you get a surprise.
FHEMAMBA_HF_HOME_DEFAULT=${FHEMAMBA_ROOT}/hf-cache
if [ -d "$FHEMAMBA_HF_HOME_DEFAULT" ]; then
  export HF_HOME="${HF_HOME:-$FHEMAMBA_HF_HOME_DEFAULT}"
else
  export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
  echo "[env] WARNING: $FHEMAMBA_HF_HOME_DEFAULT missing; falling back to \$HOME cache"
fi
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"   # everything is pre-fetched from crcfe01
export PYTHONNOUSERSITE=1                      # ignore ~/.local, which can shadow the env

# Fail fast and loudly rather than part-way through an experiment.
"$PY" - <<'PYEOF' || { echo "FATAL: environment is incomplete (see above)"; return 1 2>/dev/null || exit 1; }
import importlib.util as u, sys
missing = [m for m in ("torch", "transformers", "einops", "pyarrow", "numpy")
           if u.find_spec(m) is None]
if missing:
    print("FATAL: missing packages:", ", ".join(missing))
    sys.exit(1)
import torch
print(f"[env] python {sys.version.split()[0]}  torch {torch.__version__}  "
      f"cuda={torch.cuda.is_available()}")
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f"[env] {p.name} sm_{p.major}{p.minor} {p.total_memory/1024**3:.1f} GB")
print("[env] mamba_ssm", "present" if u.find_spec("mamba_ssm") else "MISSING (--backend official will not work)")
print("[env] causal_conv1d", "present" if u.find_spec("causal_conv1d")
      else "absent (fine; pure-torch conv is used)")
PYEOF

# Guard both filesystems before doing real work. Outputs land in $PWD (the group
# space); $HOME still matters because conda, pip and stray caches live there.
echo "[env] HF_HOME: $HF_HOME"
for FS in "$HOME" "$PWD"; do
  AVAIL_GB=$(df -BG --output=avail "$FS" 2>/dev/null | tail -1 | tr -dc '0-9')
  echo "[env] free on $FS: ${AVAIL_GB:-?} GB"
  if [ "${AVAIL_GB:-99}" -lt 1 ]; then
    echo "FATAL: $FS has under 1 GB free. Free space before running experiments."
    return 1 2>/dev/null || exit 1
  fi
done
