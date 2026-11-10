#!/bin/bash
# (Re)build HEScan with the GPU backend into $HESCAN_BUILD. Run after every code change,
# in qrsh or as the first step of a job; it only writes into the repo's build directory.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "$HERE/env.sh"
export JOBS="${JOBS:-${NSLOTS:-8}}"
bash "$HERE/../gpu_setup.sh" hescan
