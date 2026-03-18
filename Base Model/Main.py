# Main.py
from __future__ import annotations

import os
import csv
from decimal import Decimal
from typing import Any, Dict, List

from Instance_Reader import read_instance
from SSGS import ssgs_est_worst_case, print_schedule_vector
from AO_FR_Gurobi import activity_oriented_fix_and_relax


def iter_instance_txt_files(root_dir: str) -> list[str]:
    paths: list[str] = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fn in filenames:
            if fn.lower().endswith(".txt"):
                paths.append(os.path.join(dirpath, fn))
    paths.sort()
    return paths


def instance_name_from_path(txt_path: str) -> str:
    base = os.path.basename(txt_path)
    return os.path.splitext(base)[0]


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
        if fv.is_integer():
            return str(int(fv))
        return str(fv)
    except Exception:
        return ""


def main() -> None:
    PROJECT_DIR = "/Users/souvikchakraborty/Downloads/AO_RF_Gurobi-5"
    instances_root = os.path.join(PROJECT_DIR, "Instanzen_j10_FSRCPSP/k_4")
    results_csv = os.path.join(PROJECT_DIR, "Ergebnisse/ao_fr_results_j10_k4.csv")
    
    
    ensure_dir(results_csv)

    number_scen = 3

    # -------------------------------------------------
    # Eingabe jetzt über epsilon statt CL
    # WICHTIG: als String an Decimal übergeben, nicht als float
    # Beispiele:
    # epsilon = Decimal("0.1")   -> CL = 0.9
    # epsilon = Decimal("0.05")  -> CL = 0.95
    # epsilon = Decimal("0.01")  -> CL = 0.99
    # -------------------------------------------------
    epsilon = Decimal("0.1")
    CL = Decimal("1.0") - epsilon

    time_limit_sec = 7200
    time_limit_model = 30

    max_no_improve = 0

    PRINT_SSGS_VECTOR = True
    PRINT_ITER_ROWS_DEBUG = True
    DEBUG_MAX_ROWS = 10
    K_RESOURCES = None
    
    all_txt = iter_instance_txt_files(instances_root)

    per_instance: Dict[str, Dict[str, Any]] = {}
    global_max_iter = -1

    for txt_path in all_txt:
        inst_name = instance_name_from_path(txt_path)
        print("\nRunning instance", inst_name)
        print("Path", txt_path)
        print("epsilon =", str(epsilon), "| CL =", str(CL))
        

        data = read_instance(txt_path, number_scen=number_scen, k_override=K_RESOURCES)

        base = ssgs_est_worst_case(data, scenario_keep=list(range(1, number_scen + 1)))
        init_S = dict(base.S)

        n_jobs = int(getattr(data, "n_jobs_including_dummy"))

        if PRINT_SSGS_VECTOR:
            print_schedule_vector(base.S, n_jobs=n_jobs)

        fr, iter_rows = activity_oriented_fix_and_relax(
            data=data,
            number_scen=number_scen,
            epsilon=epsilon,
            omega=1,
            delta=0,
            time_limit_sec=time_limit_sec,
            time_limit_model=time_limit_model,
            init_S=init_S,
            verbose=True,
            working_directory=PROJECT_DIR,
            instance_name=inst_name,
            max_no_improve=max_no_improve,
        )

        if PRINT_ITER_ROWS_DEBUG:
            print("\nFirst iter_rows entries for", inst_name)
            for row in iter_rows[:DEBUG_MAX_ROWS]:
                print(row)

        iter_obj: Dict[int, str] = {}
        for r in iter_rows:
            it = int(r["iter"])
            iter_obj[it] = _safe_obj(r.get("obj_best", ""))
            if it > global_max_iter:
                global_max_iter = it

        per_instance[inst_name] = {
            "iter_obj": iter_obj,
            "runtime_total_sec": round(float(fr.runtime_total_sec), 3),
            "best_obj": int(fr.obj),
            "iter_best": int(fr.iter_best),
            "feasible": int(fr.feasible),
        }

        print(
            "Done",
            inst_name,
            "best_obj",
            fr.obj,
            "feasible",
            int(fr.feasible),
            "iters_seen",
            len(iter_rows),
            "runtime_total_sec",
            round(float(fr.runtime_total_sec), 3),
        )

    header: List[str] = ["instance"]
    header += [f"z_{k}" for k in range(global_max_iter + 1)]
    header += ["best_obj", "iter_best", "feasible", "runtime_total_sec"]

    with open(results_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(header)

        for inst_name in sorted(per_instance.keys()):
            rec = per_instance[inst_name]
            iter_obj = rec["iter_obj"]

            row: List[Any] = [inst_name]
            for k in range(global_max_iter + 1):
                row.append(iter_obj.get(k, ""))

            row.append(rec["best_obj"])
            row.append(rec["iter_best"])
            row.append(rec["feasible"])
            row.append(rec["runtime_total_sec"])
            w.writerow(row)


if __name__ == "__main__":
    main()