#!/usr/bin/env bash
# Correctness sweep for HEScan at a toy ring dimension (N=2^13, security NOT set).
# Usage: hescan/run_tests.sh [build-dir]   (default: hescan/build)
set -u
BIN="${1:-$(dirname "$0")/build}"
fail=0
run() {
    out=$("$BIN/hescan_demo" --logn 13 --secure 0 "$@" 2>&1)
    status=$?
    summary=$(echo "$out" | grep -oE 'depth D=[0-9]+|levels consumed: [0-9]+ / [0-9]+|max\|m_HE - m_ref\| = [0-9.e+-]+' | tr '\n' ' ')
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
