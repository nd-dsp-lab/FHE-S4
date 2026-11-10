#!/bin/bash
# (Re)build HEScan with the GPU backend into $HESCAN_BUILD. Run after every code change,
# in qrsh or as the first step of a job; it only writes into the repo's build directory.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "$HERE/env.sh"

cmake -S "$FHE_ROOT/hescan" -B "$HESCAN_BUILD" -DHESCAN_GPU=ON \
    -DOpenFHE_DIR="$HESCAN_OPENFHE_PREFIX/lib/OpenFHE" \
    -Dfideslib_DIR="$HESCAN_FIDESLIB_PREFIX/share/fideslib/cmake" \
    -DCUDAToolkit_ROOT="$CUDA_PATH"
cmake --build "$HESCAN_BUILD" -j"${NSLOTS:-8}"
