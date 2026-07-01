# exp1_j10.py
#
# Experiment 1 — j10, K=1,2,3,4 (10 scenarios each)
# Head-to-head: Baseline AO-RF vs Group-Based AO-RF
#
# Design: Two-phase execution per k for fair comparison:
#   Phase 1 — Baseline runs on ALL instances first (clean system state)
#   Phase 2 — Grouped runs on ALL instances next  (clean system state)
#   Results merged into one CSV at the end.
#
# Both methods use the same SSGS warm-start, same epsilon,
# same total time budget, and same per-subproblem time limit.
#
# Output: Ergebnisse/EXP1_10/exp1_j10_k{k}.csv  (one file per k)

from __future__ import annotations

import os
import csv
import subprocess
from decimal import Decimal
from typing import Any, Dict, List, Optional

from Instance_Reader import read_instance
from SSGS import ssgs_est_worst_case
from AO_FR_Gurobi import activity_oriented_fix_and_relax
from AO_FR_Gurobi_updated import group_oriented_fix_and_relax

# Keep Mac awake for the duration of the script
subprocess.Popen(["caffeinate", "-i", "-w", str(os.getpid())])


# =============================================================================
# CONFIG
# =============================================================================

PROJECT_DIR    = "/Users/souvikchakraborty/Downloads/AO_RF_Gurobi-5"
OUTPUT_DIR     = os.path.join(PROJECT_DIR, "Ergebnisse/EXP1")

K_VALUES       = [1, 2, 3, 4]
N_INSTANCES    = 50
NUMBER_SCEN    = 10
EPSILON        = Decimal("0.1")

# Shared time limits (identical for both methods — fair comparison)
TIME_LIMIT_SEC   = 3600
TIME_LIMIT_MODEL = 300

# Baseline-specific
BASELINE_MAX_NO_IMPROVE = 0
BASELINE_OMEGA          = 1
BASELINE_DELTA          = 0

# Grouped-specific
GROUPED_MAX_NO_IMPROVE  = 0
GROUPED_MAX_GROUP_SIZE  = 5
GROUPED_MERGE_THRESHOLD = 2.5
GROUPED_CORR_THRESHOLD  = 0.95


# =============================================================================
# HELPERS
# =============================================================================

def iter_txt_files(root: str) -> List[str]:
    paths: List[str] = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower().endswith(".txt"):
                paths.append(os.path.join(dirpath, fn))
    paths.sort()
    return paths


def instance_name(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def gap_vs_ssgs(obj: Optional[int], ssgs_obj: int) -> Optional[float]:
    if obj is None or ssgs_obj == 0:
        return None
    return round((obj - ssgs_obj) / ssgs_obj * 100, 2)


def _write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter=";")
        w.writeheader()
        w.writerows(rows)


def avg(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else float("nan")


# =============================================================================
# MAIN
# =============================================================================

def run_k(k: int) -> None:
    instances_root = os.path.join(PROJECT_DIR, f"Instanzen_j10_FSRCPSP/k_{k}")
    output_csv     = os.path.join(OUTPUT_DIR, f"exp1_j10_k{k}.csv")
    base_csv       = os.path.join(OUTPUT_DIR, f"exp1_j10_k{k}_base_phase.csv")
    grp_csv        = os.path.join(OUTPUT_DIR, f"exp1_j10_k{k}_grp_phase.csv")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_txt  = iter_txt_files(instances_root)
    selected = all_txt[:N_INSTANCES]

    print(f"\n{'#'*65}")
    print(f"EXP-1 | j10 K={k} | {len(selected)} instances | TWO-PHASE")
    print(f"epsilon={EPSILON}  scen={NUMBER_SCEN}  "
          f"TL={TIME_LIMIT_SEC}s  TL_sub={TIME_LIMIT_MODEL}s\n")

    # ── Pre-compute SSGS once (shared across both phases) ─────────
    print("Pre-computing SSGS warm-starts...", flush=True)
    ssgs_cache: Dict[str, Any] = {}
    for txt_path in selected:
        inst = instance_name(txt_path)
        try:
            data = read_instance(txt_path, number_scen=NUMBER_SCEN, k_override=k)
            n    = int(data.n_jobs_including_dummy)
            base = ssgs_est_worst_case(data, scenario_keep=list(range(1, NUMBER_SCEN + 1)))
            ssgs_cache[inst] = {
                "ssgs_obj": int(base.S.get(n - 1, 10**9)),
                "init_S":   dict(base.S),
                "n_jobs":   n,
                "data":     data,
            }
        except Exception as e:
            print(f"  ERROR loading {inst}: {e}")
    print(f"  {len(ssgs_cache)} instances cached.\n")

    # =========================================================
    # PHASE 1 — Baseline AO-RF (all instances, clean state)
    # =========================================================
    print(f"{'='*65}")
    print(f"  PHASE 1 — Baseline AO-RF")
    print(f"{'='*65}\n")

    base_results: Dict[str, Any] = {}

    for idx, txt_path in enumerate(selected, 1):
        inst = instance_name(txt_path)
        if inst not in ssgs_cache:
            continue
        cache   = ssgs_cache[inst]
        data    = cache["data"]
        init_S  = dict(cache["init_S"])
        ssgs_obj= cache["ssgs_obj"]
        n_jobs  = cache["n_jobs"]

        print(f"[{idx:3d}/{len(selected)}] {inst}", flush=True)

        try:
            fr_base, iter_rows_base = activity_oriented_fix_and_relax(
                data             = data,
                number_scen      = NUMBER_SCEN,
                epsilon          = EPSILON,
                omega            = BASELINE_OMEGA,
                delta            = BASELINE_DELTA,
                time_limit_sec   = TIME_LIMIT_SEC,
                time_limit_model = TIME_LIMIT_MODEL,
                init_S           = dict(init_S),
                verbose          = False,
                working_directory= PROJECT_DIR,
                instance_name    = inst,
                max_no_improve   = BASELINE_MAX_NO_IMPROVE,
                log_to_file      = False,
            )
            base_obj      = int(fr_base.obj)
            base_feasible = int(fr_base.feasible)
            base_runtime  = round(float(fr_base.runtime_total_sec), 3)
            base_iters    = len(iter_rows_base)
        except Exception as e:
            print(f"  ERROR: {e}")
            base_obj = base_feasible = base_runtime = base_iters = None

        base_results[inst] = {
            "instance":             inst,
            "n_jobs":               n_jobs,
            "ssgs_obj":             ssgs_obj,
            "base_obj":             base_obj,
            "base_feasible":        base_feasible,
            "base_runtime_sec":     base_runtime,
            "base_iters":           base_iters,
            "base_gap_vs_ssgs_pct": gap_vs_ssgs(base_obj, ssgs_obj),
        }
        print(f"  Base={base_obj}(feas={base_feasible}, {base_iters}iters, {base_runtime}s)", flush=True)

        # Flush phase 1 progress
        _write_csv(base_csv, list(base_results.values()))

    print(f"\n  Phase 1 complete. Results saved to: {base_csv}\n")

    # =========================================================
    # PHASE 2 — Grouped AO-RF (all instances, clean state)
    # =========================================================
    print(f"{'='*65}")
    print(f"  PHASE 2 — Grouped AO-RF")
    print(f"{'='*65}\n")

    grp_results: Dict[str, Any] = {}

    for idx, txt_path in enumerate(selected, 1):
        inst = instance_name(txt_path)
        if inst not in ssgs_cache:
            continue
        cache   = ssgs_cache[inst]
        data    = cache["data"]
        init_S  = dict(cache["init_S"])
        ssgs_obj= cache["ssgs_obj"]
        n_jobs  = cache["n_jobs"]

        print(f"[{idx:3d}/{len(selected)}] {inst}", flush=True)

        try:
            fr_grp, iter_rows_grp = group_oriented_fix_and_relax(
                data               = data,
                number_scen        = NUMBER_SCEN,
                epsilon            = EPSILON,
                time_limit_sec     = TIME_LIMIT_SEC,
                time_limit_model   = TIME_LIMIT_MODEL,
                init_S             = dict(init_S),
                verbose            = False,
                working_directory  = PROJECT_DIR,
                instance_name      = inst,
                max_no_improve     = GROUPED_MAX_NO_IMPROVE,
                corr_threshold     = GROUPED_CORR_THRESHOLD,
                max_group_size     = GROUPED_MAX_GROUP_SIZE,
                merge_threshold    = GROUPED_MERGE_THRESHOLD,
                stop_on_first_feasible = True,
            )
            grp_obj      = int(fr_grp.obj)
            grp_feasible = int(fr_grp.feasible)
            grp_runtime  = round(float(fr_grp.runtime_total_sec), 3)
            grp_iters    = len(iter_rows_grp)
            grp_ngroups  = len(set(r["group"] for r in iter_rows_grp)) if iter_rows_grp else 0
        except Exception as e:
            print(f"  ERROR: {e}")
            grp_obj = grp_feasible = grp_runtime = grp_iters = grp_ngroups = None

        grp_results[inst] = {
            "instance":            inst,
            "grp_obj":             grp_obj,
            "grp_feasible":        grp_feasible,
            "grp_runtime_sec":     grp_runtime,
            "grp_iters":           grp_iters,
            "grp_n_groups":        grp_ngroups,
            "grp_gap_vs_ssgs_pct": gap_vs_ssgs(grp_obj, ssgs_obj),
        }
        print(f"  Grp={grp_obj}(feas={grp_feasible}, {grp_ngroups}grps, {grp_runtime}s)", flush=True)

        # Flush phase 2 progress
        _write_csv(grp_csv, list(grp_results.values()))

    print(f"\n  Phase 2 complete. Results saved to: {grp_csv}\n")

    # =========================================================
    # MERGE & WRITE FINAL CSV
    # =========================================================
    rows: List[Dict[str, Any]] = []
    for txt_path in selected:
        inst = instance_name(txt_path)
        if inst not in base_results or inst not in grp_results:
            continue
        b = base_results[inst]
        g = grp_results[inst]

        base_obj = b["base_obj"]
        grp_obj  = g["grp_obj"]
        makespan_diff = (
            grp_obj - base_obj
            if grp_obj is not None and base_obj is not None else None
        )

        row = {**b, **g,
            "makespan_diff": makespan_diff,
            "grp_better":    int(makespan_diff < 0)  if makespan_diff is not None else "",
            "same":          int(makespan_diff == 0) if makespan_diff is not None else "",
            "base_better":   int(makespan_diff > 0)  if makespan_diff is not None else "",
        }
        rows.append(row)

    _write_csv(output_csv, rows)

    # ── Summary ───────────────────────────────────────────────────
    n = len(rows)
    feas_base   = [r for r in rows if r["base_feasible"]  == 1]
    feas_grp    = [r for r in rows if r["grp_feasible"]   == 1]
    both_feas   = [r for r in rows if r["base_feasible"]  == 1 and r["grp_feasible"] == 1]
    grp_better  = [r for r in rows if r["grp_better"]     == 1]
    base_better = [r for r in rows if r["base_better"]    == 1]
    same        = [r for r in rows if r["same"]            == 1]

    print(f"\n{'='*65}")
    print(f"  EXP-1 SUMMARY  —  j10 K={k}  ({n} instances)")
    print(f"{'='*65}")
    print(f"  {'Metric':<42} {'Baseline':>10} {'Grouped':>10}")
    print(f"  {'-'*62}")
    print(f"  {'Feasible instances':<42} {len(feas_base):>10} {len(feas_grp):>10}")
    print(f"  {'Feasibility rate':<42} "
          f"{len(feas_base)/n*100:>9.1f}% {len(feas_grp)/n*100:>9.1f}%")
    print(f"  {'Avg makespan (all)':<42} "
          f"{avg(r['base_obj'] for r in rows):>10.2f} "
          f"{avg(r['grp_obj']  for r in rows):>10.2f}")
    if both_feas:
        print(f"  {'Avg makespan (both feasible)':<42} "
              f"{avg(r['base_obj'] for r in both_feas):>10.2f} "
              f"{avg(r['grp_obj']  for r in both_feas):>10.2f}")
    print(f"  {'Avg gap vs SSGS (%)':<42} "
          f"{avg(r['base_gap_vs_ssgs_pct'] for r in rows):>9.2f}% "
          f"{avg(r['grp_gap_vs_ssgs_pct']  for r in rows):>9.2f}%")
    print(f"  {'Avg runtime (s)':<42} "
          f"{avg(r['base_runtime_sec'] for r in rows):>10.3f} "
          f"{avg(r['grp_runtime_sec']  for r in rows):>10.3f}")
    print(f"  {'Avg subproblems / groups solved':<42} "
          f"{avg(r['base_iters']   for r in rows):>10.2f} "
          f"{avg(r['grp_n_groups'] for r in rows):>10.2f}")
    print(f"\n  Head-to-head:")
    print(f"  {'  Grouped BETTER  (diff < 0)':<42} {len(grp_better):>22}")
    print(f"  {'  SAME quality    (diff = 0)':<42} {len(same):>22}")
    print(f"  {'  Baseline BETTER (diff > 0)':<42} {len(base_better):>22}")
    print(f"\n  Results saved to: {output_csv}")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    for k in K_VALUES:
        run_k(k)
