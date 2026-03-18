# Main_Group.py
#
# Runs the TRUE group-based A-R&F where each subproblem
# fixes an ENTIRE GROUP simultaneously (not one activity at a time).
#
# Number of subproblems = number of groups (e.g. 4 for n=10)
# instead of number of activities (e.g. 10 for n=10).

from __future__ import annotations

import os
import csv
from decimal import Decimal
from typing import Any, Dict, List

from Instance_Reader import read_instance
from SSGS import ssgs_est_worst_case, print_schedule_vector
from AO_FR_Gurobi_updated import group_oriented_fix_and_relax


def iter_instance_txt_files(root_dir: str) -> list[str]:
    paths: list[str] = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fn in filenames:
            if fn.lower().endswith(".txt"):
                paths.append(os.path.join(dirpath, fn))
    paths.sort()
    return paths


def instance_name_from_path(txt_path: str) -> str:
    return os.path.splitext(os.path.basename(txt_path))[0]


def ensure_dir(path: str) -> None:
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)


def _safe_obj(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    try:
        fv = float(v)
        return str(int(fv)) if fv.is_integer() else str(fv)
    except Exception:
        return ""


def main() -> None:

    # ── CONFIG — edit these 4 lines for each run ──────────────────
    N_JOBS       = 30
    K_RESOURCES  = None
    PROJECT_DIR  = "/Users/souvikchakraborty/Downloads/AO_RF_Gurobi-5"
    # ──────────────────────────────────────────────────────────────

    instances_root = os.path.join(
        PROJECT_DIR, f"Instanzen_j10_FSRCPSP/k_4"
        
    )
    results_csv = os.path.join(
        PROJECT_DIR,
        f"Ergebnisse/j10_k4_grouped.csv"           #ao_fr_results_group_based_{N_JOBS}_k_{K_RESOURCES}.csv"
    )
    ensure_dir(results_csv)

    # ── Solver parameters ─────────────────────────────────────────
    number_scen      = 3
    epsilon          = Decimal("0.1")
    CL               = Decimal("1.0") - epsilon
    time_limit_sec   = 7200
    time_limit_model = 30     # per GROUP subproblem
    # max_no_improve   = 0
    max_no_improve   = 2

    # ── Grouper parameters ────────────────────────────────────────
    CORR_THRESHOLD  = 0.95
    MIN_GROUP_SIZE  = 1
    MAX_GROUP_SIZE  = 4
    MERGE_THRESHOLD = 2.5

    PRINT_SSGS_VECTOR     = True
    PRINT_ITER_ROWS_DEBUG = True
    DEBUG_MAX_ROWS        = 10

    all_txt = iter_instance_txt_files(instances_root)
    print(f"Found {len(all_txt)} instances in {instances_root}")
    if not all_txt:
        print("ERROR: No instances found. Check instances_root path.")
        return

    per_instance: Dict[str, Dict[str, Any]] = {}
    global_max_iter = -1

    for txt_path in all_txt:
        inst_name = instance_name_from_path(txt_path)
        print("\n" + "=" * 60)
        print(f"Running instance : {inst_name}")
        print(f"Path             : {txt_path}")
        print(f"epsilon = {epsilon} | CL = {CL}")

        data = read_instance(txt_path, number_scen=number_scen,
                             k_override=K_RESOURCES)

        base   = ssgs_est_worst_case(
            data, scenario_keep=list(range(1, number_scen + 1))
        )
        init_S = dict(base.S)
        n_jobs = int(getattr(data, "n_jobs_including_dummy"))

        if PRINT_SSGS_VECTOR:
            print_schedule_vector(base.S, n_jobs=n_jobs)

        fr, iter_rows = group_oriented_fix_and_relax(
            data                 = data,
            number_scen          = number_scen,
            epsilon              = epsilon,
            time_limit_sec       = time_limit_sec,
            time_limit_model     = time_limit_model,
            init_S               = init_S,
            verbose              = True,
            working_directory    = PROJECT_DIR,
            instance_name        = inst_name,
            max_no_improve       = max_no_improve,
            max_retries_per_group= 2,
            corr_threshold       = CORR_THRESHOLD,
            min_group_size       = MIN_GROUP_SIZE,
            max_group_size       = MAX_GROUP_SIZE,
            merge_threshold      = MERGE_THRESHOLD,
            stop_on_first_feasible = False
        )

        if PRINT_ITER_ROWS_DEBUG:
            print(f"\nFirst iter_rows entries for {inst_name}")
            for row in iter_rows[:DEBUG_MAX_ROWS]:
                print(row)

        iter_obj: Dict[int, str] = {}
        for r in iter_rows:
            it = int(r["iter"])
            iter_obj[it] = _safe_obj(r.get("obj_best", ""))
            if it > global_max_iter:
                global_max_iter = it

        n_groups = len(set(r["group"] for r in iter_rows)) if iter_rows else 0

        per_instance[inst_name] = {
            "iter_obj"          : iter_obj,
            "runtime_total_sec" : round(float(fr.runtime_total_sec), 3),
            "best_obj"          : int(fr.obj),
            "iter_best"         : int(fr.iter_best),
            "feasible"          : int(fr.feasible),
            "n_groups"          : n_groups,
            "iters_seen"        : len(iter_rows),
        }

        print(
            f"Done {inst_name} | best_obj {fr.obj} | "
            f"feasible {int(fr.feasible)} | "
            f"iters_seen {len(iter_rows)} | "
            f"n_groups {n_groups} | "
            f"runtime {round(float(fr.runtime_total_sec), 3)}s"
        )

    # ── Write CSV ─────────────────────────────────────────────────
    header: List[str] = ["instance"]
    header += [f"z_{k}" for k in range(global_max_iter + 1)]
    header += ["best_obj", "iter_best", "feasible",
               "runtime_total_sec", "n_groups", "iters_seen"]

    with open(results_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(header)
        for inst_name in sorted(per_instance.keys()):
            rec      = per_instance[inst_name]
            iter_obj = rec["iter_obj"]
            row: List[Any] = [inst_name]
            for k in range(global_max_iter + 1):
                row.append(iter_obj.get(k, ""))
            row.append(rec["best_obj"])
            row.append(rec["iter_best"])
            row.append(rec["feasible"])
            row.append(rec["runtime_total_sec"])
            row.append(rec["n_groups"])
            row.append(rec["iters_seen"])
            w.writerow(row)

    print("\n" + "=" * 60)
    print(f"Results written to: {results_csv}")
    print("=" * 60)


if __name__ == "__main__":
    main()