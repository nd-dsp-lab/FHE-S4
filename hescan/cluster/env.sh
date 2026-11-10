#!/bin/bash
# Environment for HEScan GPU builds and jobs on CRC. Sourced by every script in this directory.
# It only loads modules and checks; it never installs anything (see install_deps.sh).
#
# Set once in ~/.bashrc on the login node (paths in the GROUP space, not $HOME):
#   export FHE_ROOT=/groups/<group>/$NETID/FHE-S4      # this repo
#   export FHE_OPT=/groups/<group>/$NETID/opt          # installed OpenFHE + FIDESlib
#   export FHE_SRC=/groups/<group>/$NETID/src          # sources and build trees
#   export HESCAN_MODULES="gcc/13 cmake/3.28 cuda/12.4"  # exact names from `module avail`
#
# Requirements (FIDESlib 2.2): CUDA 12 or 13, GCC >= 11, CMake >= 3.25.2.

set -uo pipefail

: "${FHE_ROOT:?set FHE_ROOT to the repo checkout (see the top of hescan/cluster/env.sh)}"
: "${FHE_OPT:?set FHE_OPT to the install prefix in group space}"
: "${FHE_SRC:=${FHE_OPT%/opt}/src}"
export FHE_ROOT FHE_OPT FHE_SRC

export HESCAN_OPENFHE_PREFIX="${HESCAN_OPENFHE_PREFIX:-$FHE_OPT/openfhe-1.6.1-fides}"
export HESCAN_FIDESLIB_PREFIX="${HESCAN_FIDESLIB_PREFIX:-$FHE_OPT/fideslib}"
export HESCAN_BUILD="${HESCAN_BUILD:-$FHE_ROOT/hescan/build-gpu}"
# RTX 6000 on qa-rtx6k-019 is sm_75. Use "auto" to build for whatever GPU nvidia-smi sees.
export HESCAN_CUDA_ARCH="${HESCAN_CUDA_ARCH:-75-real}"

if command -v module > /dev/null 2>&1 && [ -n "${HESCAN_MODULES:-}" ]; then
    # shellcheck disable=SC2086
    module load $HESCAN_MODULES || { echo "FATAL: module load $HESCAN_MODULES failed"; return 1 2>/dev/null || exit 1; }
fi

# FIDESlib's CMake finds CUDA through CUDA_PATH (default /usr/local/cuda), not the usual
# CMake variables, so derive it from whatever the cuda module put on PATH.
if [ -z "${CUDA_PATH:-}" ]; then
    if [ -n "${CUDA_HOME:-}" ]; then
        CUDA_PATH=$CUDA_HOME
    elif command -v nvcc > /dev/null 2>&1; then
        CUDA_PATH=$(dirname "$(dirname "$(command -v nvcc)")")
    fi
fi
export CUDA_PATH="${CUDA_PATH:-/usr/local/cuda}"

echo "[env] host $(hostname)   FHE_ROOT=$FHE_ROOT   FHE_OPT=$FHE_OPT"
echo "[env] CUDA_PATH=$CUDA_PATH   arch=$HESCAN_CUDA_ARCH"
command -v nvcc > /dev/null 2>&1 && nvcc --version | tail -1 || echo "[env] WARNING: nvcc not on PATH (load the cuda module)"
echo "[env] $(gcc --version | head -1)   $(cmake --version | head -1)"

# Guard the filesystems before doing real work.
for FS in "$HOME" "$FHE_OPT"; do
    [ -d "$FS" ] || continue
    AVAIL_GB=$(df -BG --output=avail "$FS" 2>/dev/null | tail -1 | tr -dc '0-9')
    echo "[env] free on $FS: ${AVAIL_GB:-?} GB"
    if [ "${AVAIL_GB:-99}" -lt 2 ]; then
        echo "FATAL: $FS has under 2 GB free."
        return 1 2>/dev/null || exit 1
    fi
done
