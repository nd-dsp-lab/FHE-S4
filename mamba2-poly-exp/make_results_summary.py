#!/usr/bin/env python
"""Part 11 -- collect every run under runs/ into one results_summary.csv.

    python make_results_summary.py
    python make_results_summary.py --runs-dir runs --out results_summary.csv

Walks the run directories, reads the `config.json` / `metrics.json` /
`polynomial_coefficients.json` that every script writes, and emits one row per
experiment with the columns the project asked for. Missing values are left
empty rather than guessed.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

COLUMNS = [
    "run",
    "part",
    "model",
    "backend",
    "transition",
    "polynomial_degree",
    "approximation_interval",
    "fit_method",
    "pin_zero",
    "polynomial_trainable",
    "finetuning_mode",
    "token_budget",
    "tokens_seen",
    "n_trainable_params",
    "eval_data",
    "eval_tokens",
    "validation_loss",
    "perplexity",
    "exact_perplexity",
    "delta_perplexity",
    "starting_perplexity",
    "recovered_fraction",
    "transition_min",
    "transition_max",
    "frac_transition_lt_0",
    "frac_transition_gt_1",
    "frac_transition_nan",
    "z_min_observed",
    "estimated_ct_ct_depth",
    "n_degenerate_heads",
    "training_hours",
    "peak_gpu_memory_gb",
    "seed",
]


def read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except Exception:                                    # noqa: BLE001
        return None


def blank_row(**kw):
    row = {c: "" for c in COLUMNS}
    row.update({k: v for k, v in kw.items() if k in COLUMNS})
    return row


def interval_str(cfg, coeffs):
    """Prefer what the coefficients actually say over what was requested."""
    if cfg and cfg.get("transition") == "exact":
        return ""            # exact exp has no approximation interval
    if coeffs:
        for v in coeffs.values():
            if not isinstance(v, dict):
                continue
            if v.get("kind", "").startswith("per_head"):
                return "per-head"
            iv = v.get("interval")
            if iv:
                return f"[{iv[0]:g},{iv[1]:g}]"
    if cfg and cfg.get("interval_mode") == "per-head":
        return "per-head"
    if cfg and cfg.get("xmin") is not None:
        return f"[{cfg['xmin']:g},{cfg['xmax']:g}]"
    return ""


def degree_of(cfg, coeffs):
    if coeffs:
        for v in coeffs.values():
            if isinstance(v, dict) and v.get("degree") is not None:
                return v["degree"]
    t = (cfg or {}).get("transition", "")
    return t[4:] if t.startswith("poly") else ""


def collect(runs_dir: Path):
    rows = []
    for d in sorted(p for p in runs_dir.rglob("*") if p.is_dir()):
        # directories starting with "_" are scratch (smoke tests, aborted runs)
        if any(part.startswith("_") for part in d.relative_to(runs_dir).parts):
            continue
        cfg = read_json(d / "config.json")
        met = read_json(d / "metrics.json")
        coeffs = read_json(d / "polynomial_coefficients.json")
        results_csv = d / "results.csv"

        # ---- Part 7 / Part 4: a results.csv with one row per configuration ----
        if results_csv.exists():
            with results_csv.open() as f:
                for r in csv.DictReader(f):
                    if "perplexity" not in r:
                        continue
                    rows.append(blank_row(
                        run=str(d), part="7 (no training)",
                        model=(cfg or {}).get("model", ""),
                        backend=(cfg or {}).get("backend", ""),
                        transition=r.get("transition", ""),
                        polynomial_degree=r.get("degree", ""),
                        approximation_interval=r.get("interval", ""),
                        fit_method=r.get("fit_method", ""),
                        pin_zero=r.get("pin_zero", ""),
                        polynomial_trainable=False,
                        finetuning_mode="none", token_budget=0, tokens_seen=0,
                        eval_data=(cfg or {}).get("data", ""),
                        validation_loss=r.get("validation_loss", ""),
                        perplexity=r.get("perplexity", ""),
                        delta_perplexity=r.get("delta_perplexity", ""),
                        transition_min=r.get("transition_min", ""),
                        transition_max=r.get("transition_max", ""),
                        frac_transition_lt_0=r.get("frac_transition_lt_0", ""),
                        frac_transition_gt_1=r.get("frac_transition_gt_1", ""),
                        frac_transition_nan=r.get("frac_transition_nan", ""),
                        z_min_observed=r.get("z_min_observed", ""),
                        estimated_ct_ct_depth=r.get("fhe_depth_estimate", ""),
                        n_degenerate_heads=r.get("n_degenerate_heads", ""),
                        peak_gpu_memory_gb=r.get("peak_gpu_memory_gb", ""),
                        seed=(cfg or {}).get("seed", ""),
                    ))
            continue

        if not isinstance(met, dict):
            continue

        is_distill = "teacher_perplexity" in met
        is_finetune = "ending_perplexity" in met and not is_distill
        if not (is_distill or is_finetune):
            continue
        exact = met.get("teacher_perplexity", met.get("exact_perplexity", ""))
        rows.append(blank_row(
            run=str(d),
            part="10 (distillation)" if is_distill else "9 (fine-tuning)",
            model=(cfg or {}).get("model", ""),
            backend=(cfg or {}).get("backend", ""),
            transition=(cfg or {}).get("transition", ""),
            polynomial_degree=degree_of(cfg, coeffs.get("after_training") if coeffs else None),
            approximation_interval=interval_str(cfg, (coeffs or {}).get("after_training")
                                                or (coeffs or {}).get("after")),
            fit_method=(cfg or {}).get("fit_method", ""),
            pin_zero=(cfg or {}).get("pin_zero", ""),
            polynomial_trainable=(cfg or {}).get("trainable_poly", ""),
            finetuning_mode=(cfg or {}).get("mode", ""),
            token_budget=(cfg or {}).get("token_budget", ""),
            tokens_seen=met.get("tokens_seen", ""),
            n_trainable_params=(met.get("trainable") or {}).get("n_trainable", ""),
            eval_data=(cfg or {}).get("eval_data", ""),
            eval_tokens=((cfg or {}).get("eval_blocks", "") or "") and
                        (cfg["eval_blocks"] * cfg.get("seq_len", 0)),
            validation_loss=met.get("ending_loss", ""),
            perplexity=met.get("ending_perplexity", ""),
            exact_perplexity=exact,
            delta_perplexity=met.get("delta_perplexity_vs_exact",
                                     met.get("delta_perplexity_vs_teacher", "")),
            starting_perplexity=met.get("starting_perplexity", ""),
            recovered_fraction=met.get("recovered_fraction", ""),
            transition_min=met.get("transition_min", ""),
            transition_max=met.get("transition_max", ""),
            frac_transition_lt_0=met.get("frac_transition_lt_0", ""),
            frac_transition_gt_1=met.get("frac_transition_gt_1", ""),
            frac_transition_nan=met.get("frac_transition_nan", ""),
            z_min_observed=met.get("z_min_observed", ""),
            estimated_ct_ct_depth=met.get("fhe_depth_estimate", ""),
            training_hours=met.get("training_hours", ""),
            peak_gpu_memory_gb=met.get("peak_gpu_memory_gb", ""),
            seed=(cfg or {}).get("seed", ""),
        ))
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-dir", type=Path, default=Path("runs"))
    ap.add_argument("--out", type=Path, default=Path("results_summary.csv"))
    args = ap.parse_args(argv)

    rows = collect(args.runs_dir)
    if not rows:
        print(f"no runs found under {args.runs_dir}/ -- run eval_poly_exp.py first")
        return 1
    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)

    def fmt(v, n=4):
        try:
            return f"{float(v):.{n}f}"
        except (TypeError, ValueError):
            return str(v)[:12]

    print(f"{len(rows)} rows -> {args.out}\n")
    hdr = (f"{'part':>18} {'transition':>11} {'interval':>10} {'mode':>5} {'tokens':>9} "
           f"{'ppl':>10} {'d ppl':>9} {'depth':>6} {'frac a<0':>9}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['part']:>18} {str(r['transition']):>11} {str(r['approximation_interval']):>10} "
              f"{str(r['finetuning_mode']):>5} {str(r['tokens_seen'] or r['token_budget']):>9} "
              f"{fmt(r['perplexity']):>10} {fmt(r['delta_perplexity']):>9} "
              f"{str(r['estimated_ct_ct_depth'])[:6]:>6} {fmt(r['frac_transition_lt_0'], 5):>9}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
