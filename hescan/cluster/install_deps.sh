#!/bin/bash
# One-time install of the patched OpenFHE 1.6.1 and FIDESlib into $FHE_OPT.
#
#   1. on the LOGIN node (has internet):    hescan/cluster/install_deps.sh fetch
#   2. in an INTERACTIVE GPU session:       qrsh -q gpu@@jung_gpu -l gpu=1 -pe smp 8
#                                           hescan/cluster/install_deps.sh build
#
# `build` refuses to run from a batch job: installs are done deliberately, by hand, after
# checking free space (a batch-job install once filled $HOME to 0 bytes; see
# mamba2-poly-exp/cluster/README.md). FIDESlib needs OpenFHE 1.6.1 with its compile patch,
# installed with static libraries, so the stock OpenFHE 1.4 install is not reused.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "$HERE/env.sh"

FIDESLIB_REF="${FIDESLIB_REF:-main}"
JOBS="${JOBS:-${NSLOTS:-8}}"

fetch() {
    mkdir -p "$FHE_SRC"
    if [ ! -d "$FHE_SRC/FIDESlib/.git" ]; then
        git clone https://github.com/CAPS-UMU/FIDESlib.git "$FHE_SRC/FIDESlib"
    fi
    git -C "$FHE_SRC/FIDESlib" fetch origin && git -C "$FHE_SRC/FIDESlib" checkout "$FIDESLIB_REF"
    if [ ! -d "$FHE_SRC/openfhe-1.6.1/.git" ]; then
        git clone --branch v1.6.1 --depth 1 https://github.com/openfheorg/openfhe-development.git \
            "$FHE_SRC/openfhe-1.6.1"
        # The same patch FIDESlib's deps/build.sh applies in its default "patch" mode.
        git -C "$FHE_SRC/openfhe-1.6.1" apply "$FHE_SRC/FIDESlib/deps/fideslib-1.6.1-compile.patch"
    fi
    echo "fetched into $FHE_SRC; now run 'install_deps.sh build' inside qrsh on a GPU node"
}

build() {
    if [ "${ENVIRONMENT:-}" = "BATCH" ]; then
        echo "FATAL: do not install from a batch job; run this inside qrsh"
        exit 1
    fi
    [ -d "$FHE_SRC/openfhe-1.6.1" ] && [ -d "$FHE_SRC/FIDESlib" ] || { echo "run 'fetch' on the login node first"; exit 1; }

    # 1. patched OpenFHE 1.6.1, shared + static, no native/FMA codegen (FIDESlib checks this)
    cmake -S "$FHE_SRC/openfhe-1.6.1" -B "$FHE_SRC/openfhe-1.6.1-build" \
        -DCMAKE_BUILD_TYPE=Release -DBUILD_STATIC=ON -DWITH_NATIVEOPT=OFF \
        -DBUILD_UNITTESTS=OFF -DBUILD_EXAMPLES=OFF -DBUILD_BENCHMARKS=OFF \
        -DCMAKE_INSTALL_PREFIX="$HESCAN_OPENFHE_PREFIX"
    cmake --build "$FHE_SRC/openfhe-1.6.1-build" -j"$JOBS"
    cmake --install "$FHE_SRC/openfhe-1.6.1-build"

    # 2. FIDESlib against it, for the target GPU architecture only (much faster than "all")
    cmake -S "$FHE_SRC/FIDESlib" -B "$FHE_SRC/FIDESlib-build" \
        -DCUDA_PATH="$CUDA_PATH" -DFIDESLIB_ARCH="$HESCAN_CUDA_ARCH" \
        -DOPENFHE_INSTALL_PREFIX="$HESCAN_OPENFHE_PREFIX" -DFIDESLIB_INSTALL_OPENFHE=OFF \
        -DFIDESLIB_COMPILE_TESTS=OFF -DFIDESLIB_COMPILE_BENCHMARKS=OFF \
        -DFIDESLIB_INSTALL_PREFIX="$HESCAN_FIDESLIB_PREFIX" -DCMAKE_INSTALL_PREFIX="$HESCAN_FIDESLIB_PREFIX"
    cmake --build "$FHE_SRC/FIDESlib-build" -j"$JOBS"
    cmake --install "$FHE_SRC/FIDESlib-build"

    # FIDESlib's device probe: lists the GPUs it can see.
    "$FHE_SRC/FIDESlib-build/gpu-test" || true
    du -sh "$HESCAN_OPENFHE_PREFIX" "$HESCAN_FIDESLIB_PREFIX"
    echo "installed. Build trees can be removed: rm -rf $FHE_SRC/openfhe-1.6.1-build $FHE_SRC/FIDESlib-build"
}

case "${1:-}" in
    fetch) fetch ;;
    build) build ;;
    *) echo "usage: $0 fetch|build"; exit 2 ;;
esac
