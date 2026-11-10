#!/bin/bash
# One-time install of the patched OpenFHE 1.6.1 and FIDESlib into $FHE_OPT on CRC.
#
#   1. on the LOGIN node (has internet):    hescan/cluster/install_deps.sh fetch
#   2. in an INTERACTIVE GPU session:       qrsh -q gpu@@jung_gpu -l gpu=1 -pe smp 8
#                                           hescan/cluster/install_deps.sh build
#
# `build` refuses to run from a batch job: installs are done deliberately, by hand, after
# checking free space (a batch-job install once filled $HOME to 0 bytes; see
# mamba2-poly-exp/cluster/README.md). The steps themselves live in hescan/gpu_setup.sh.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "$HERE/env.sh"
export JOBS="${JOBS:-${NSLOTS:-8}}"

case "${1:-}" in
    fetch) bash "$HERE/../gpu_setup.sh" fetch ;;
    build)
        if [ "${ENVIRONMENT:-}" = "BATCH" ]; then
            echo "FATAL: do not install from a batch job; run this inside qrsh"
            exit 1
        fi
        bash "$HERE/../gpu_setup.sh" check deps
        ;;
    *) echo "usage: $0 fetch|build"; exit 2 ;;
esac
