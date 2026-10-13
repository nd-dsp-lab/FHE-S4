#!/usr/bin/env python
"""The verdict for absorption: absorbed run vs its control, paired per block.

    python norm/compare_absorb.py runs/norm_absorb/absorb_A1_... runs/norm_absorb/control_...

Both runs are fine-tuned with the same recipe on the same text, so any gain from
fine-tuning itself cancels. What is left is the cost of replacing the pre-norms
by constants. Both are also shown against the untouched teacher.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np


def paired(a, b):
    d = np.exp(np.asarray(a)) - np.exp(np.asarray(b))
    if not np.all(np.isfinite(d)):
        return "BROKEN (inf)"
    m, s = float(d.mean()), float(d.std(ddof=1) / math.sqrt(len(d)))
    v = "not distinguishable" if abs(m) <= 2 * s else "SIGNIFICANT"
    return f"{m:+9.4f} +/- {2 * s:7.4f}  {v}"


def main():
    ab, ct = (json.loads((Path(p) / "metrics.json").read_text()) for p in sys.argv[1:3])
    if ab.get("control") or not ct.get("control"):
        raise SystemExit("usage: compare_absorb.py <absorbed run> <control run>")
    A, C = ab["per_block_nll"], ct["per_block_nll"]
    for L in A["after"]:
        if A["teacher"][L] != C["teacher"][L]:
            raise SystemExit(f"L={L}: the two runs were not evaluated on the same blocks")
        print(f"L={L}")
        print(f"  absorbed  vs teacher   {paired(A['after'][L], A['teacher'][L])}")
        print(f"  control   vs teacher   {paired(C['after'][L], C['teacher'][L])}")
        print(f"  ABSORBED vs CONTROL    {paired(A['after'][L], C['after'][L])}   <- the verdict")
    fc = ab.get("fold_check") or {}
    print(f"\nfold check: {fc.get('folded')} constants folded, max relative logit change "
          f"{fc.get('max_rel_logit_change', float('nan')):.2e}")


if __name__ == "__main__":
    main()
