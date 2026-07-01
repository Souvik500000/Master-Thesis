# exp_scen_red_j120_allk_n50.py
#
# Full 50-instance run of j120 all-K scenario reduction.
# Matches EXP1 (j10) and EXP2 (j30) methodology: 50 instances per K.
#
# For each K, compare:
#   baseline     : S=10, no reduction (group-based AO-RF, full scenarios)
#   pergroup_kmd : per-group PAM K-medoids, n_keep=3
#
# Settings per K:
#   K=1  alpha=10  gs=10  (easy)
#   K=2  alpha=15  gs=10  (medium)
#   K=3  alpha=15  gs=10  (medium-hard)
#   K=4  alpha=15  gs=10  (hard)
#
# Pool(4) × 2 Gurobi threads = 8 cores, no contention.
# 8 tasks total (4 K × 2 methods), processed in 2 batches of 4.
#
# Output: Ergebnisse/EXP_SCEN/scen_red_j120_allk_n50_{method}_k{k}.csv
#         Ergebnisse/EXP_SCEN/scen_red_j120_allk_n50_summary.csv

from __future__ import annotations
import os, csv, glob, time, subprocess
from decimal import Decimal
from multiprocessing import Pool
from typing import Any, Dict, List, Tuple

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

PROJECT_DIR = "/Users/souvikchakraborty/Downloads/AO_RF_Gurobi-5"
OUTPUT_DIR  = os.path.join(PROJECT_DIR, "Ergebnisse/EXP_SCEN")

N_INSTANCES    = 50
NUMBER_SCEN    = 10
N_KEEP         = 3
EPSILON        = Decimal("0.1")
TIME_LIMIT_SEC   = 600
TIME_LIMIT_MODEL = 60
MAX_GROUP_SIZE   = 10
MERGE_THRESHOLD  = 2.5
CORR_THRESHOLD   = 0.95
GUROBI_THREADS   = 2   # Pool(4) × 2 = 8 cores

# Per-K settings: (alpha, note)
K_SETTINGS = {
    1: (10,  "easy"),
    2: (15,  "medium"),
    3: (15,  "medium-hard"),
    4: (15,  "hard"),
}


def _avg(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else float("nan")


def run_k(args: Tuple[int, str]) -> List[Dict[str, Any]]:
    k, method = args

    from Instance_Reader import read_instance
    from SSGS import ssgs_est_worst_case
    from AO_FR_Gurobi_updated import group_oriented_fix_and_relax

    time_alpha = K_SETTINGS[k][0]
    use_pergroup = (method == "pergroup_kmd")
    output_csv = os.path.join(OUTPUT_DIR,
                              f"scen_red_j120_allk_n50_{method}_k{k}.csv")

    print(f"\n{'#'*65}")
    print(f"  j120  K={k}  alpha={time_alpha}  gs={MAX_GROUP_SIZE}  n={N_INSTANCES}")
    print(f"  method={method}  n_keep={N_KEEP if use_pergroup else NUMBER_SCEN}")
    print(f"{'#'*65}", flush=True)

    inst_dir = os.path.join(PROJECT_DIR, f"FSRCPSP_Instanzen/j120/k = {k}")
    paths    = sorted(glob.glob(os.path.join(inst_dir, "**/*.txt"),
                                recursive=True))[:N_INSTANCES]
    if not paths:
        print(f"  No instances found under {inst_dir}"); return []

    rows: List[Dict[str, Any]] = []

    for i, txt_path in enumerate(paths, 1):
        inst   = os.path.splitext(os.path.basename(txt_path))[0]
        data   = read_instance(txt_path, number_scen=NUMBER_SCEN, k_override=k)
        n_jobs = int(data.n_jobs_including_dummy)
        T_max  = int(max(data.ls.values())) if data.ls else 0
        T_agg  = (T_max + time_alpha - 1) // time_alpha

        base     = ssgs_est_worst_case(data,
                       scenario_keep=list(range(1, NUMBER_SCEN + 1)))
        ssgs_obj = int(base.S.get(n_jobs - 1, 10**9))

        print(f"[{i:2d}/{len(paths)}] {inst}  T_agg={T_agg}  ssgs={ssgs_obj}",
              flush=True)
        t0 = time.time()
        try:
            res, iters = group_oriented_fix_and_relax(
                data                   = data,
                number_scen            = NUMBER_SCEN,
                epsilon                = EPSILON,
                time_limit_sec         = TIME_LIMIT_SEC,
                time_limit_model       = TIME_LIMIT_MODEL,
                init_S                 = None,
                verbose                = False,
                working_directory      = PROJECT_DIR,
                instance_name          = inst,
                max_no_improve         = 0,
                corr_threshold         = CORR_THRESHOLD,
                max_group_size         = MAX_GROUP_SIZE,
                merge_threshold        = MERGE_THRESHOLD,
                stop_on_first_feasible = True,
                log_to_file            = False,
                time_alpha             = time_alpha,
                gurobi_threads         = GUROBI_THREADS,
                enable_per_group_scenario_reduction = use_pergroup,
                n_keep_per_group       = N_KEEP if use_pergroup else NUMBER_SCEN,
            )
            obj  = int(res.obj)
            feas = int(res.feasible)
            rt   = round(float(res.runtime_total_sec), 2)
            wall = round(time.time() - t0, 2)
        except Exception as e:
            print(f"  ERROR: {e}", flush=True)
            import traceback; traceback.print_exc()
            obj = feas = rt = wall = None

        gap = (None if obj is None or ssgs_obj == 0
               else round((obj - ssgs_obj) / ssgs_obj * 100, 2))
        print(f"  feas={feas}  gap={gap}%  rt={rt}s", flush=True)

        rows.append({
            "method":   method,
            "k":        k,
            "alpha":    time_alpha,
            "instance": inst,
            "n_jobs":   n_jobs,
            "T_max":    T_max,
            "T_agg":    T_agg,
            "ssgs_obj": ssgs_obj,
            "obj":      obj,
            "feas":     feas,
            "rt_sec":   rt,
            "wall_sec": wall,
            "gap_pct":  gap,
        })

        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()),
                               delimiter=";")
            w.writeheader(); w.writerows(rows)

    n_feas  = sum(1 for r in rows if r["feas"] == 1)
    avg_gap = _avg(r["gap_pct"] for r in rows if r["feas"] == 1)
    avg_rt  = _avg(r["rt_sec"]  for r in rows)
    print(f"\n  K={k} {method}: {n_feas}/{len(rows)} feas | "
          f"avg_gap={avg_gap:.2f}% | avg_rt={avg_rt:.1f}s", flush=True)
    return rows


if __name__ == "__main__":
    subprocess.Popen(["caffeinate", "-i", "-w", str(os.getpid())])
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 8 tasks: K=1..4 × baseline/pergroup_kmd
    # Pool(4): 4 workers × 2 Gurobi threads = 8 cores
    tasks = [(k, method)
             for k in [1, 2, 3, 4]
             for method in ["baseline", "pergroup_kmd"]]

    print(f"j120 all-K scenario reduction (N=50) | {len(tasks)} tasks | "
          f"Pool(4) × {GUROBI_THREADS} threads")
    print(f"n_keep={N_KEEP} for pergroup_kmd\n")
    for k, method in tasks:
        alpha = K_SETTINGS[k][0]
        print(f"  K={k}  alpha={alpha}  gs={MAX_GROUP_SIZE}  method={method}")

    with Pool(processes=4) as pool:
        results = pool.map(run_k, tasks)

    # Build summary
    print(f"\n{'='*65}")
    print(f"  j120 ALL-K SCENARIO REDUCTION SUMMARY (N=50)")
    print(f"  {'K':>3}  {'method':<16} {'feas':>8} {'avg_gap':>9} {'avg_rt':>8}")
    print(f"  {'-'*52}")

    summary = []
    for (k, method), task_rows in zip(tasks, results):
        if not task_rows:
            continue
        n_feas  = sum(1 for r in task_rows if r["feas"] == 1)
        avg_gap = _avg(r["gap_pct"] for r in task_rows if r["feas"] == 1)
        avg_rt  = _avg(r["rt_sec"]  for r in task_rows)
        print(f"  {k:>3}  {method:<16} {n_feas:>4}/{len(task_rows)}  "
              f"{avg_gap:>8.2f}%  {avg_rt:>7.1f}s")
        summary.append({
            "k":           k,
            "method":      method,
            "alpha":       K_SETTINGS[k][0],
            "n_instances": len(task_rows),
            "feas_count":  n_feas,
            "feas_pct":    round(n_feas / len(task_rows) * 100, 1),
            "avg_gap_pct": round(avg_gap, 2),
            "avg_rt_sec":  round(avg_rt, 2),
        })
    print(f"{'='*65}")

    summary_path = os.path.join(OUTPUT_DIR, "scen_red_j120_allk_n50_summary.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()),
                           delimiter=";")
        w.writeheader(); w.writerows(summary)
    print(f"  Summary saved: {summary_path}")
