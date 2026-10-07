#!/usr/bin/env python3
"""Run the STEP 1 sweep: one measure_epoch process per configuration.

    python3 fhe/run_epoch_sweep.py --smoke            # 1 tiny insecure config, ~1 min: does it work?
    python3 fhe/run_epoch_sweep.py --search-only      # EPOCH for every config, no keys: minutes
    python3 fhe/run_epoch_sweep.py                    # the full sweep at N = 2^16
    python3 fhe/run_epoch_sweep.py --logn 16 17       # add N = 2^17 if the machine has the memory

Why one process per configuration: bootstrapping keygen at N = 2^17 can exhaust
memory, and an out-of-memory kill cannot be caught from inside the process. Here
it becomes a row with status "crashed" and the stage it died in -- the brief says
every failure is a result, so no configuration is ever silently dropped.

Rows are written to fhe/epoch_rows/<config>.json as each configuration
progresses, and the merged fhe/epoch_results.json is rewritten after every
configuration, so a sweep interrupted half-way still leaves everything it
measured. Re-running skips finished configurations unless --force.

Standard library only.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import platform
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def total_ram_gb() -> float | None:
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) / 1024 / 1024
    except OSError:
        pass
    try:  # macOS
        out = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True)
        return int(out.stdout.strip()) / 1024**3
    except Exception:
        return None


def openfhe_version(binary: Path) -> str:
    f = binary.parent / "openfhe_version.txt"
    if f.exists():
        return f.read_text().strip() or "unknown"
    for cand in ("/usr/local/lib/OpenFHE/OpenFHEConfigVersion.cmake",):
        try:
            m = re.search(r'PACKAGE_VERSION\s+"([^"]+)"', Path(cand).read_text())
            if m:
                return m.group(1)
        except OSError:
            pass
    return "unknown"


def git_commit() -> str:
    try:
        return subprocess.run(["git", "-C", str(HERE), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def config_name(logn, scale, skdist, budget, insecure=False, iters=1) -> str:
    b = budget.replace(",", "-")
    it = f"_it{iters}" if iters and iters > 1 else ""
    return f"{'SMOKE_' if insecure else ''}logN{logn}_s{scale}_{skdist}_b{b}{it}"


def external_time_cmd() -> list[str]:
    """GNU time reports the child's peak RSS even when the child is SIGKILLed."""
    gt = shutil.which("time") if platform.system() == "Linux" else None
    if gt and os.path.exists("/usr/bin/time"):
        probe = subprocess.run(["/usr/bin/time", "-v", "true"], capture_output=True, text=True)
        if probe.returncode == 0 and "Maximum resident set size" in probe.stderr:
            return ["/usr/bin/time", "-v"]
    return []


def run_one(args, binary, logn, scale, skdist, budget, rows_dir: Path) -> dict:
    name = config_name(logn, scale, skdist, budget, args.smoke, args.iterations)
    row_path = rows_dir / f"{name}.json"
    log_path = rows_dir / f"{name}.log"
    done_states = {"search_only_done"} if args.search_only else {"ok"}
    if row_path.exists() and not args.force:
        try:
            old = json.loads(row_path.read_text())
            if old.get("status") in done_states:
                print(f"  [skip] {name}: already {old['status']}")
                return old
        except json.JSONDecodeError:
            pass
    if row_path.exists():
        row_path.unlink()

    cmd = [str(binary), "--logn", str(logn), "--scale", str(scale), "--skdist", skdist,
           "--budget", budget, "--boots", str(args.boots), "--out", str(row_path),
           "--iterations", str(args.iterations)]
    if args.search_only:
        cmd.append("--search-only")
    if args.smoke:
        cmd += ["--insecure", "--levels-after", str(args.smoke_levels)]
    full = external_time_cmd() + cmd

    print(f"  [run ] {name}", flush=True)
    t0 = time.time()
    with open(log_path, "w") as log:
        log.write("$ " + " ".join(full) + "\n\n")
        log.flush()
        try:
            p = subprocess.run(full, stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout)
            rc, timed_out = p.returncode, False
        except subprocess.TimeoutExpired:
            rc, timed_out = None, True
    wall = time.time() - t0

    try:
        row = json.loads(row_path.read_text())
    except (OSError, json.JSONDecodeError):
        row = {"logN": logn, "scaling_mod_bits": scale, "secret_key_dist": skdist,
               "level_budget": [int(v) for v in budget.split(",")], "status": "no_output"}

    log_text = log_path.read_text(errors="replace")
    m = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", log_text)
    drv = {"exit_code": rc, "timed_out": timed_out, "wall_seconds_cpu_only": round(wall, 1),
           "external_peak_rss_mb": round(int(m.group(1)) / 1024, 1) if m else None,
           "log": str(log_path.relative_to(HERE.parent)) if log_path.is_relative_to(HERE.parent)
           else str(log_path)}
    # GNU time wraps the child: "Command terminated by signal 9" means SIGKILL.
    sig = None
    ms = re.search(r"Command terminated by signal (\d+)", log_text)
    if ms:
        sig = int(ms.group(1))
    elif rc is not None and rc < 0:
        sig = -rc
    drv["signal"] = sig
    row["driver"] = drv

    last_stage = row.get("status", "no_output")
    if timed_out:
        row["status"] = "crashed"
        row["crash_reason"] = f"timed out after {args.timeout}s during stage '{last_stage}'"
    elif sig is not None or (rc not in (0, 3) and last_stage not in done_states | {"error"}):
        why = {9: "SIGKILL -- almost always the kernel's out-of-memory killer",
               6: "SIGABRT", 11: "SIGSEGV"}.get(sig, f"exit code {rc}, signal {sig}")
        row["status"] = "crashed"
        row["crash_reason"] = f"{why}; last completed stage was '{last_stage}'"
    row_path.write_text(json.dumps(row, indent=2) + "\n")
    print(f"         -> {row.get('status')}  ({wall:.0f}s)"
          + (f"  EPOCH={row.get('epoch_declared')}" if "epoch_declared" in row else "")
          + (f"  [{row.get('crash_reason') or row.get('error')}]"
             if row.get("status") in ("crashed", "error") else ""), flush=True)
    return row


def summarize(rows: list[dict]) -> str:
    hdr = (f"{'config':34s} {'status':16s} {'EPOCH':>5s} {'boot':>4s} {'emp':>4s} "
           f"{'bits@1':>6s} {'bits@5':>6s} {'keysMB':>8s} {'margin':>6s}")
    out = [hdr, "-" * len(hdr)]
    for r in rows:
        name = config_name(r.get("logN"), r.get("scaling_mod_bits"), r.get("secret_key_dist"),
                           ",".join(str(v) for v in r.get("level_budget", [])),
                           "NotSet" in str(r.get("security_level", "")), r.get("bootstrap_iterations", 1))

        def f(k, fmt="{:.1f}"):
            v = r.get(k)
            return "-" if v is None else (fmt.format(v) if isinstance(v, float) else str(v))
        bits = r.get("precision_bits_max_per_bootstrap") or []
        out.append(f"{name:34s} {r.get('status', '?'):16s} {f('epoch_declared'):>5s} "
                   f"{f('bootstrap_depth'):>4s} {f('empirical_mults_correct'):>4s} "
                   f"{f('precision_bits_max_after_1_boot'):>6s} "
                   f"{(f'{bits[-1]:.1f}' if bits and bits[-1] is not None else '-'):>6s} "
                   f"{f('key_bytes_total_mb', '{:.0f}'):>8s} {f('modulus_margin_bits'):>6s}")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--binary", type=Path, default=HERE / "build" / "measure_epoch")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--rows-dir", type=Path, default=None)
    ap.add_argument("--logn", type=int, nargs="+", default=[16])
    ap.add_argument("--scales", type=int, nargs="+", default=[40, 50, 59])
    ap.add_argument("--skdists", nargs="+", default=["sparse", "uniform"])
    ap.add_argument("--budgets", nargs="+", default=["3,3", "4,4"])
    ap.add_argument("--boots", type=int, default=5)
    ap.add_argument("--iterations", type=int, default=1, help="bootstrap iterations (2 ~ doubles precision)")
    ap.add_argument("--timeout", type=int, default=3 * 3600, help="seconds per configuration")
    ap.add_argument("--search-only", action="store_true", help="EPOCH only: no keys, no bootstraps")
    ap.add_argument("--smoke", action="store_true",
                    help="one tiny INSECURE config (N=2^12) to check the pipeline; never an EPOCH")
    ap.add_argument("--smoke-levels", type=int, default=10)
    ap.add_argument("--force", action="store_true", help="re-run finished configurations")
    args = ap.parse_args()

    if not args.binary.exists():
        print(f"no binary at {args.binary}. Build first:\n"
              f"  cmake -S fhe -B fhe/build -DOpenFHE_DIR=/usr/local/lib/OpenFHE && cmake --build fhe/build -j",
              file=sys.stderr)
        return 1
    if args.smoke:
        args.logn, args.scales, args.skdists, args.budgets = [12], [50], ["uniform"], ["3,3"]
        args.boots = min(args.boots, 3)
    tag = "smoke" if args.smoke else ("search" if args.search_only else "results")
    out = args.out or HERE / f"epoch_{tag}.json"
    rows_dir = args.rows_dir or HERE / f"epoch_rows_{tag}"
    rows_dir.mkdir(parents=True, exist_ok=True)

    meta = {"step": "STEP 1 -- usable CKKS epoch after bootstrapping",
            "host": socket.gethostname(), "platform": platform.platform(),
            "cpu_count": os.cpu_count(), "ram_gb": round(total_ram_gb() or 0, 1),
            "openfhe_version": openfhe_version(args.binary), "git_commit": git_commit(),
            "started": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
            "mode": tag, "timing_label": "CPU-ONLY, NON-TRANSFERABLE",
            "sweep": {"logN": args.logn, "scaling_mod_bits": args.scales,
                      "secret_key_dist": args.skdists, "level_budget": args.budgets,
                      "boots": args.boots}}
    print(json.dumps(meta, indent=2))

    rows = []
    for logn, scale, skd, bud in itertools.product(args.logn, args.scales, args.skdists, args.budgets):
        rows.append(run_one(args, args.binary, logn, scale, skd, bud, rows_dir))
        meta["finished"] = time.strftime("%Y-%m-%d %H:%M:%S %Z")
        out.write_text(json.dumps({"meta": meta, "rows": rows}, indent=2) + "\n")

    print("\n" + summarize(rows))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal.default_int_handler)
    raise SystemExit(main())
