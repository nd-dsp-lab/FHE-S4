#!/usr/bin/env bash
# Correctness sweep for HEScan at a toy ring dimension (N=2^13, security NOT set).
# Usage: hescan/run_tests.sh [build-dir]   (default: hescan/build)
#
# Extra demo flags for every case go in HESCAN_ARGS, e.g. the GPU backend checked against
# the CPU kernel on the same ciphertexts:
#   HESCAN_ARGS="--gpu 1 --compare 1" hescan/run_tests.sh build-gpu
set -u
BIN="${1:-$(dirname "$0")/build}"
EXTRA=(${HESCAN_ARGS:-})
fail=0
run() {
    out=$("$BIN/hescan_demo" --logn 13 --secure 0 "$@" "${EXTRA[@]}" 2>&1)
    status=$?
    summary=$(echo "$out" | grep -oE 'depth D=[0-9]+|levels consumed: [0-9]+ / [0-9]+|max\|m_HE - m_ref\| = [0-9.e+-]+|max\|m_GPU - m_CPU\| = [0-9.e+-]+|bit-identical ciphertexts: [0-9]+ / [0-9]+' | tr '\n' ' ')
    if [ $status -eq 0 ]; then echo "PASS  $*  :: $summary"; else echo "FAIL  $*  :: $summary"; echo "$out" | tail -3; fail=1; fi
}
"$BIN/hescan_conj_check" > /dev/null && echo "PASS  conj_check" || { echo "FAIL  conj_check"; fail=1; }
run --scan bk
run --scan hs
run --scan seq
run --scan bk --complex 0
run --scan bk --complex 0 --sstate 32 --L 13
run --scaling fixed
run --h0 1
run --h0 1 --complex 0 --scan hs
run --radix 4
run --radix 8 --L 7 --scan hs
run --G 1 --sstate 16
run --H 3 --G 3 --P 4 --sstate 32
run --P 16 --ds 8 --sstate 32 --H 2 --G 1
run --sstate 128
exit $fail
