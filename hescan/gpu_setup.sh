#!/usr/bin/env bash
# Build and run the GPU (FIDESlib) HEScan on a machine with an NVIDIA GPU.
#
#   hescan/gpu_setup.sh check     # driver, GPU, CUDA toolkit, gcc, cmake
#   hescan/gpu_setup.sh fetch     # clone FIDESlib + OpenFHE 1.6.1 and apply FIDESlib's patch (internet)
#   hescan/gpu_setup.sh deps      # build + install patched OpenFHE 1.6.1 and FIDESlib (once, slow)
#   hescan/gpu_setup.sh hescan    # build HEScan with -DHESCAN_GPU=ON (after every code change)
#   hescan/gpu_setup.sh test      # toy-N sweep: CPU kernel, then GPU kernel checked against CPU
#   hescan/gpu_setup.sh bench     # N=2^15 / 2^16 benchmarks, CPU vs GPU on the same packet
#   hescan/gpu_setup.sh all       # check fetch deps hescan test
#
# Everything installs under $FHE_PREFIX (default ~/fhe); nothing needs root once the NVIDIA
# driver and a CUDA 12/13 toolkit are present. Variables (all optional):
#   FHE_PREFIX              root for sources, installs        (default: $HOME/fhe)
#   CUDA_PATH               CUDA toolkit root                  (default: from nvcc on PATH, else /usr/local/cuda)
#   HESCAN_CUDA_ARCH        GPU architectures to compile for   (default: auto = the GPUs nvidia-smi sees;
#                           e.g. 75-real for an RTX 6000/2080, 86-real for an RTX 3090/A6000, 89-real for a 4090)
#   HESCAN_SRC              sources + build trees              (default: $FHE_PREFIX/src)
#   HESCAN_OPENFHE_PREFIX   patched OpenFHE install            (default: $FHE_PREFIX/opt/openfhe-1.6.1-fides)
#   HESCAN_FIDESLIB_PREFIX  FIDESlib install                   (default: $FHE_PREFIX/opt/fideslib)
#   HESCAN_BUILD            HEScan GPU build directory         (default: <repo>/hescan/build-gpu)
#   FIDESLIB_REF            FIDESlib git ref                    (default: main)
#   JOBS                    parallel build jobs                 (default: nproc)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FHE_PREFIX="${FHE_PREFIX:-$HOME/fhe}"
HESCAN_SRC="${HESCAN_SRC:-$FHE_PREFIX/src}"
HESCAN_OPENFHE_PREFIX="${HESCAN_OPENFHE_PREFIX:-$FHE_PREFIX/opt/openfhe-1.6.1-fides}"
HESCAN_FIDESLIB_PREFIX="${HESCAN_FIDESLIB_PREFIX:-$FHE_PREFIX/opt/fideslib}"
HESCAN_BUILD="${HESCAN_BUILD:-$HERE/build-gpu}"
HESCAN_CUDA_ARCH="${HESCAN_CUDA_ARCH:-auto}"
FIDESLIB_REF="${FIDESLIB_REF:-main}"
JOBS="${JOBS:-$(nproc)}"
if [ -z "${CUDA_PATH:-}" ]; then
    if [ -n "${CUDA_HOME:-}" ]; then
        CUDA_PATH=$CUDA_HOME
    elif command -v nvcc > /dev/null 2>&1; then
        CUDA_PATH=$(dirname "$(dirname "$(readlink -f "$(command -v nvcc)")")")
    else
        CUDA_PATH=/usr/local/cuda
    fi
fi
export CUDA_PATH
export PATH="$CUDA_PATH/bin:$PATH"

say()  { printf '\n=== %s ===\n' "$*"; }
die()  { echo "ERROR: $*" >&2; exit 1; }
# version_ge A B: true if version A >= version B
version_ge() { [ "$(printf '%s\n%s\n' "$2" "$1" | sort -V | head -1)" = "$2" ]; }

check() {
    say "check"
    local ok=1
    if ! command -v nvidia-smi > /dev/null 2>&1 || ! nvidia-smi > /dev/null 2>&1; then
        echo "  [FAIL] nvidia-smi does not work: the NVIDIA driver is missing (needs an admin)"
        ok=0
    else
        nvidia-smi --query-gpu=index,name,compute_cap,memory.total,driver_version --format=csv,noheader |
            sed 's/^/  GPU /'
        local drvcuda
        drvcuda=$(nvidia-smi | grep -oE 'CUDA Version: [0-9.]+' | grep -oE '[0-9.]+$' || true)
        echo "  driver supports CUDA up to ${drvcuda:-?}"
        local cc
        for cc in $(nvidia-smi --query-gpu=compute_cap --format=csv,noheader); do
            version_ge "$cc" 7.0 || { echo "  [FAIL] compute capability $cc < 7.0: FIDESlib needs sm_70+"; ok=0; }
        done
    fi
    if [ -x "$CUDA_PATH/bin/nvcc" ]; then
        local tk
        tk=$("$CUDA_PATH/bin/nvcc" --version | grep -oE 'release [0-9.]+' | grep -oE '[0-9.]+')
        echo "  CUDA toolkit $tk at $CUDA_PATH"
        case "$tk" in 12.* | 13.*) ;; *) echo "  [FAIL] FIDESlib needs CUDA 12 or 13"; ok=0 ;; esac
        if [ -n "${drvcuda:-}" ] && ! version_ge "$drvcuda" "${tk%.*}"; then
            echo "  [WARN] toolkit $tk is newer than the driver supports ($drvcuda): install an older toolkit or update the driver"
        fi
        if [[ "$tk" == 13.* ]] && nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2> /dev/null | grep -q '^7\.0'; then
            echo "  [FAIL] CUDA 13 dropped sm_70 (V100): use a CUDA 12 toolkit"
            ok=0
        fi
    else
        echo "  [FAIL] no nvcc at $CUDA_PATH/bin: install the CUDA toolkit or set CUDA_PATH"
        ok=0
    fi
    local gccv cmv
    gccv=$(gcc -dumpversion 2> /dev/null || echo 0)
    cmv=$(cmake --version 2> /dev/null | head -1 | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' || echo 0)
    echo "  gcc $gccv, cmake $cmv"
    version_ge "$gccv" 11 || { echo "  [FAIL] gcc >= 11 needed"; ok=0; }
    version_ge "$cmv" 3.25.2 || { echo "  [FAIL] cmake >= 3.25.2 needed (pip install cmake)"; ok=0; }
    [ "$ok" = 1 ] && echo "  all prerequisites OK" || die "fix the [FAIL] items above"
}

fetch() {
    say "fetch sources into $HESCAN_SRC"
    mkdir -p "$HESCAN_SRC"
    if [ ! -d "$HESCAN_SRC/FIDESlib/.git" ]; then
        git clone https://github.com/CAPS-UMU/FIDESlib.git "$HESCAN_SRC/FIDESlib"
    fi
    git -C "$HESCAN_SRC/FIDESlib" fetch origin
    git -C "$HESCAN_SRC/FIDESlib" checkout "$FIDESLIB_REF"
    if [ ! -d "$HESCAN_SRC/openfhe-1.6.1/.git" ]; then
        git clone --branch v1.6.1 --depth 1 https://github.com/openfheorg/openfhe-development.git \
            "$HESCAN_SRC/openfhe-1.6.1"
        # What FIDESlib's deps/build.sh does in its default "patch" mode.
        git -C "$HESCAN_SRC/openfhe-1.6.1" apply "$HESCAN_SRC/FIDESlib/deps/fideslib-1.6.1-compile.patch"
    fi
}

deps() {
    [ -d "$HESCAN_SRC/openfhe-1.6.1" ] && [ -d "$HESCAN_SRC/FIDESlib" ] || die "run '$0 fetch' first"

    say "patched OpenFHE 1.6.1 -> $HESCAN_OPENFHE_PREFIX"
    # Shared + static; FIDESlib links the static archives. No -march=native: FIDESlib refuses
    # an OpenFHE built with FMA codegen.
    cmake -S "$HESCAN_SRC/openfhe-1.6.1" -B "$HESCAN_SRC/openfhe-1.6.1-build" \
        -DCMAKE_BUILD_TYPE=Release -DBUILD_STATIC=ON -DWITH_NATIVEOPT=OFF \
        -DBUILD_UNITTESTS=OFF -DBUILD_EXAMPLES=OFF -DBUILD_BENCHMARKS=OFF \
        -DCMAKE_INSTALL_PREFIX="$HESCAN_OPENFHE_PREFIX"
    cmake --build "$HESCAN_SRC/openfhe-1.6.1-build" -j"$JOBS"
    cmake --install "$HESCAN_SRC/openfhe-1.6.1-build"

    say "FIDESlib ($HESCAN_CUDA_ARCH) -> $HESCAN_FIDESLIB_PREFIX"
    # FIDESlib finds CUDA through CUDA_PATH, not CUDAToolkit_ROOT.
    cmake -S "$HESCAN_SRC/FIDESlib" -B "$HESCAN_SRC/FIDESlib-build" \
        -DCUDA_PATH="$CUDA_PATH" -DFIDESLIB_ARCH="$HESCAN_CUDA_ARCH" \
        -DOPENFHE_INSTALL_PREFIX="$HESCAN_OPENFHE_PREFIX" -DFIDESLIB_INSTALL_OPENFHE=OFF \
        -DFIDESLIB_COMPILE_TESTS=OFF -DFIDESLIB_COMPILE_BENCHMARKS=OFF \
        -DFIDESLIB_INSTALL_PREFIX="$HESCAN_FIDESLIB_PREFIX" -DCMAKE_INSTALL_PREFIX="$HESCAN_FIDESLIB_PREFIX"
    cmake --build "$HESCAN_SRC/FIDESlib-build" -j"$JOBS"
    cmake --install "$HESCAN_SRC/FIDESlib-build"
    "$HESCAN_SRC/FIDESlib-build/gpu-test" || echo "  (gpu-test failed: is the driver loaded?)"
    du -sh "$HESCAN_OPENFHE_PREFIX" "$HESCAN_FIDESLIB_PREFIX"
    echo "  build trees can now be removed: rm -rf $HESCAN_SRC/openfhe-1.6.1-build $HESCAN_SRC/FIDESlib-build"
}

hescan() {
    say "HEScan (GPU backend) -> $HESCAN_BUILD"
    [ -f "$HESCAN_FIDESLIB_PREFIX/share/fideslib/cmake/fideslibConfig.cmake" ] || die "FIDESlib not installed: run '$0 deps'"
    cmake -S "$HERE" -B "$HESCAN_BUILD" -DHESCAN_GPU=ON \
        -DOpenFHE_DIR="$HESCAN_OPENFHE_PREFIX/lib/OpenFHE" \
        -Dfideslib_DIR="$HESCAN_FIDESLIB_PREFIX/share/fideslib/cmake" \
        -DCUDAToolkit_ROOT="$CUDA_PATH"
    cmake --build "$HESCAN_BUILD" -j"$JOBS"
}

test_() {
    say "CPU sweep"
    bash "$HERE/run_tests.sh" "$HESCAN_BUILD"
    say "GPU sweep, each case checked against the CPU kernel"
    HESCAN_ARGS="--gpu 1 --compare 1 ${HESCAN_ARGS:-}" bash "$HERE/run_tests.sh" "$HESCAN_BUILD"
}

bench() {
    local demo="$HESCAN_BUILD/hescan_demo" cmp="${BENCH_COMPARE:-1}"
    say "benchmarks (CPU vs GPU on the same packet; BENCH_COMPARE=0 skips the CPU run)"
    "$demo" --logn 15 --gpu 1 --compare "$cmp"
    "$demo" --logn 16 --gpu 1 --compare "$cmp"
    # the larger shape from README.md: K_s = 4, D = 12
    "$demo" --logn 16 --H 8 --P 8 --G 2 --ds 16 --L 32 --sstate 256 --gpu 1 --compare "$cmp"
}

[ $# -ge 1 ] || { sed -n '2,25p' "$0"; exit 2; }
for step in "$@"; do
    case "$step" in
        check) check ;;
        fetch) fetch ;;
        deps) deps ;;
        hescan) hescan ;;
        test) test_ ;;
        bench) bench ;;
        all) check; fetch; deps; hescan; test_ ;;
        *) die "unknown step '$step' (check|fetch|deps|hescan|test|bench|all)" ;;
    esac
done
