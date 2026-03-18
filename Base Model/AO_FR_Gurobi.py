# AO_FR_Gurobi.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from decimal import Decimal
import os
import time
import random

from SSGS import ssgs_est_worst_case

from AO_Model import (
    buildModel,
    reset_bounds_for_subproblem_Mz,
    set_bounds_for_subproblem_Mz,
    set_warmstart_for_subproblem_Mz,
)


@dataclass
class FRResult:
    S: Dict[int, int]
    obj: int
    feasible: bool
    violated: int
    allowed: int
    violated_list: List[int]
    runtime_total_sec: float
    iter_best: int


def makespan_from_S(S: Dict[int, int], n_jobs: int) -> int:
    sink = n_jobs - 1
    return int(S.get(sink, 10**9))


def horizon_length_from_data(data) -> int:
    h = getattr(data, "horizon", None)
    if h is not None:
        return int(h)
    ls: Dict[int, int] = getattr(data, "ls")
    return int(max(ls.values()))


def _fmt_time(sec: float) -> str:
    return f"{sec:.3f}s"


def _fmt_set(ints: List[int]) -> str:
    if not ints:
        return "{}"
    return "{" + ",".join(str(x) for x in ints) + "}"


def _log(msg: str, log_path: Optional[str]) -> None:
    print(msg)
    if log_path is not None:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")


def _clamp(v: int, lo: int, hi: int) -> int:
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v


def _window_for_activity(es_i: int, ls_i: int, s_i_center: int, w: int) -> Tuple[int, int]:
    left = max(es_i, s_i_center - w)
    right = min(ls_i, s_i_center + w)
    if right < left:
        right = left
    return int(left), int(right)


def _activity_status_sets(
    n: int, i_curr: int, fixed_start: Dict[int, int]
) -> Tuple[List[int], List[int], List[int]]:
    fixed = sorted(fixed_start.keys())
    binary = [i_curr] if i_curr not in fixed_start else []
    relaxed = [j for j in range(n) if (j not in fixed_start) and (j != i_curr)]
    return fixed, binary, relaxed


def _extract_schedule_from_solution(
    ao,
    data,
) -> Dict[int, int]:
    S: Dict[int, int] = {}

    for i in ao.activities:
        es_i = int(data.es[i])
        ls_i = int(data.ls[i])

        best_t: Optional[int] = None
        best_val = -1.0

        for t in range(es_i, ls_i + 1):
            val = ao.xEff[i, t].X
            if val > best_val:
                best_val = val
                best_t = t

        if best_t is None:
            raise RuntimeError(f"No valid start time found for activity {i}")

        S[i] = int(best_t)

    return S


def _check_integrality(
    ao,
    data,
    tol: float = 1e-6,
) -> bool:
    for i in ao.activities:
        es_i = int(data.es[i])
        ls_i = int(data.ls[i])

        for t in range(es_i, ls_i + 1):
            val = ao.xEff[i, t].X
            if abs(val - 0.0) > tol and abs(val - 1.0) > tol:
                return False

    return True


def _get_objective_value_if_available(
    ao,
) -> Optional[float]:
    if ao.model.SolCount > 0:
        return float(ao.model.ObjVal)
    return None


def activity_oriented_fix_and_relax(
    data,
    number_scen: int,
    epsilon,
    omega: int,
    delta: int,
    time_limit_sec: float,
    time_limit_model: float,
    init_S: Optional[Dict[int, int]] = None,
    verbose: bool = True,
    working_directory: str = ".",
    instance_name: str = "inst",
    max_no_improve: int = 0,
    log_to_file: bool = True,
    max_retries_per_activity: int = 2,
    max_subproblems: Optional[int] = None,
) -> Tuple[FRResult, List[Dict[str, Any]]]:
    # -------------------------------------------------
    # Epsilon exakt behandeln
    # -------------------------------------------------
    epsilon_dec = epsilon if isinstance(epsilon, Decimal) else Decimal(str(epsilon))
    CL_dec = Decimal("1.0") - epsilon_dec

    if not (Decimal("0.0") <= epsilon_dec <= Decimal("1.0")):
        raise ValueError("epsilon must be in [0,1]")
    if time_limit_sec <= 0:
        raise ValueError("time_limit_sec must be > 0")
    if time_limit_model <= 0:
        raise ValueError("time_limit_model must be > 0")
    if omega <= 0:
        raise ValueError("omega must be >= 1")
    if delta < 0:
        raise ValueError("delta must be >= 0")
    if delta >= omega:
        raise ValueError("delta must be < omega")
    if max_no_improve < 0:
        raise ValueError("max_no_improve must be >= 0")
    if max_retries_per_activity <= 0:
        raise ValueError("max_retries_per_activity must be >= 1")
    if max_subproblems is not None and max_subproblems <= 0:
        raise ValueError("max_subproblems must be >= 1 or None")

    start_wall = time.time()

    n = int(getattr(data, "n_jobs_including_dummy"))
    found_integral_solution = False

    if max_subproblems is None:
        max_subproblems = n

    results_dir = os.path.join(working_directory, "Results")
    os.makedirs(results_dir, exist_ok=True)

    log_path: Optional[str] = None
    if log_to_file:
        log_path = os.path.join(results_dir, "ao_fr_log.txt")
        try:
            if os.path.exists(log_path):
                os.remove(log_path)
        except OSError:
            pass

    es: Dict[int, int] = getattr(data, "es")
    ls: Dict[int, int] = getattr(data, "ls")

    if init_S is None:
        base = ssgs_est_worst_case(data, scenario_keep=list(range(1, number_scen + 1)))
        S_ssgs = dict(base.S)
        S_best = dict(base.S)
    else:
        S_ssgs = dict(init_S)
        S_best = dict(init_S)

    obj_best = makespan_from_S(S_best, n)
    iter_best = -1

    S_prime = dict(S_best)

    fixed_start: Dict[int, int] = {0: 0}
    S_prime[0] = 0
    S_best[0] = 0
    S_ssgs[0] = 0

    ao = buildModel(
        data=data,
        number_scen=number_scen,
        epsilon=epsilon_dec,
        threads=4,
        mip_gap=1e-5,
        time_limit_seconds=None,
    )

    es0 = int(es.get(0, 0))
    ls0 = int(ls.get(0, es0))
    if ls0 < es0:
        ls0 = es0
    fixed_start[0] = _clamp(0, es0, ls0)
    S_prime[0] = fixed_start[0]
    S_best[0] = fixed_start[0]
    S_ssgs[0] = fixed_start[0]

    model_time_sum_sec = 0.0
    iter_rows: List[Dict[str, Any]] = []

    if verbose:
        _log("", log_path)
        _log("[AO-FR] instance " + str(instance_name), log_path)
        _log("[AO-FR] epsilon = " + str(epsilon_dec) + ", CL = " + str(CL_dec), log_path)
        _log("[AO-FR] time_limit_sec = " + str(float(time_limit_sec)), log_path)
        _log("[AO-FR] time_limit_model = " + str(float(time_limit_model)), log_path)
        _log("[AO-FR] initial obj = " + str(obj_best), log_path)
        _log("[AO-FR] fixed: activity 0 starts at t=0", log_path)
        _log("[AO-FR] max_subproblems (activities tested) = " + str(int(max_subproblems)), log_path)

    sub_id = 0
    tested_activities = 0
    no_improve = 0
    stop_early = False

    while (time.time() - start_wall) < time_limit_sec and (not stop_early):
        improved_in_pass = False

        for i in range(n):
            if stop_early:
                break
            if (time.time() - start_wall) >= time_limit_sec:
                break

            if i == 0:
                continue

            z = 0
            got_feasible_for_i = False

            while (not got_feasible_for_i) and z < max_retries_per_activity and (time.time() - start_wall) < time_limit_sec:
                if stop_early:
                    break

                z += 1

                es_i = int(es.get(i, 0))
                ls_i = int(ls.get(i, es_i))
                if ls_i < es_i:
                    ls_i = es_i

                w = random.randint(1, 10)

                s_center = int(S_prime.get(i, es_i))
                s_center = _clamp(s_center, es_i, ls_i)

                t_left, t_right = _window_for_activity(es_i, ls_i, s_center, w)

                fixed_list, binary_list, relaxed_list = _activity_status_sets(
                    n=n, i_curr=i, fixed_start=fixed_start
                )

                if verbose:
                    _log("", log_path)
                    _log(
                        "[AO-FR] i = " + str(i) + " window = [" + str(es_i) + "," + str(ls_i) + "], with w = " + str(w),
                        log_path,
                    )
                    _log("[AO-FR] fixed_activities = " + _fmt_set(fixed_list), log_path)
                    _log("[AO-FR] binary_activities = " + _fmt_set(binary_list), log_path)
                    _log("[AO-FR] relaxed_activities = " + _fmt_set(relaxed_list), log_path)

                elapsed_so_far_wall = time.time() - start_wall
                remaining_time = max(0.0, time_limit_sec - elapsed_so_far_wall)
                effective_model_limit = max(1.0, min(float(time_limit_model), float(remaining_time)))

                reset_bounds_for_subproblem_Mz(
                    ao=ao,
                    data=data,
                )

                set_bounds_for_subproblem_Mz(
                    ao=ao,
                    data=data,
                    i_curr=i,
                    t_left=t_left,
                    t_right=t_right,
                    fixed_start_times=fixed_start,
                )

                set_warmstart_for_subproblem_Mz(
                    ao=ao,
                    data=data,
                    S_warm=S_prime,
                    i_curr=i,
                    t_left=t_left,
                    t_right=t_right,
                    fixed_start_times=fixed_start,
                )

                ao.model.Params.TimeLimit = float(effective_model_limit)
                ao.model.optimize()

                obj_z: Optional[float] = _get_objective_value_if_available(ao)
                res_usd: Optional[float] = float(ao.model.Runtime)

                if res_usd is not None and res_usd >= 0:
                    model_time_sum_sec += float(res_usd)

                has_solution = ao.model.SolCount > 0
                feasible = bool(has_solution)
                label = "feasible" if feasible else "infeasible"

                okIntegral: Optional[int] = None
                if has_solution:
                    okIntegral = 1 if _check_integrality(ao, data) else 0

                integral_label = "NA"
                if okIntegral == 1:
                    integral_label = "integral"
                elif okIntegral == 0:
                    integral_label = "non-integral"

                case = 5
                if has_solution:
                    case = 2 if okIntegral == 1 else 1

                exported = False
                accepted = False
                cand_obj: Optional[int] = None
                cand_S: Optional[Dict[int, int]] = None

                if has_solution:
                    cand_S = _extract_schedule_from_solution(ao, data)
                    exported = True

                    for j in range(n):
                        es_j = int(es.get(j, 0))
                        ls_j = int(ls.get(j, es_j))
                        if ls_j < es_j:
                            ls_j = es_j
                        val = int(cand_S.get(j, es_j))
                        cand_S[j] = _clamp(val, es_j, ls_j)

                    cand_S[0] = int(fixed_start[0])

                    cand_obj = makespan_from_S(cand_S, n)

                    got_feasible_for_i = True
                    fixed_start[i] = int(cand_S.get(i, es_i))
                    S_prime = dict(cand_S)

                    if cand_obj < obj_best:
                        S_best = dict(cand_S)
                        obj_best = int(cand_obj)
                        iter_best = int(sub_id)
                        accepted = True
                        improved_in_pass = True
                        no_improve = 0

                    if okIntegral == 1:
                        found_integral_solution = True
                        stop_early = True
                        got_feasible_for_i = True
                        if verbose:
                            _log(
                                "[AO-FR] stopping early because a feasible integral solution for the whole problem was found",
                                log_path,
                            )

                if case == 5:
                    if z < max_retries_per_activity:
                        _log(
                            "Infeasible for i = " + str(i)
                            + " on try " + str(z)
                            + " of " + str(max_retries_per_activity)
                            + " retry with new window",
                            log_path,
                        )
                    else:
                        s_i = int(S_ssgs.get(i, es_i))
                        s_i = _clamp(s_i, es_i, ls_i)
                        fixed_start[i] = s_i
                        S_prime[i] = s_i
                        got_feasible_for_i = True
                        _log(
                            "Infeasible for i = " + str(i)
                            + " on last try fix to SSGS start t = " + str(s_i),
                            log_path,
                        )

                elapsed = model_time_sum_sec

                if verbose:
                    if obj_z is not None:
                        zfw_for_print = float(obj_z)
                    elif cand_obj is not None:
                        zfw_for_print = float(cand_obj)
                    else:
                        zfw_for_print = float(obj_best)

                    zfw_str = f"{zfw_for_print:.6f}".rstrip("0").rstrip(".")
                    _log(
                        "[AO-FR-STATUS] i = " + str(i)
                        + " | " + label
                        + " | " + integral_label
                        + " | case=" + str(case)
                        + " | ZFW=" + zfw_str
                        + " | S*=" + str(int(obj_best))
                        + " | t=" + _fmt_time(elapsed),
                        log_path,
                    )

                iter_rows.append(
                    {
                        "iter": int(sub_id),
                        "activity": int(i),
                        "retry": int(z),
                        "w": int(w),
                        "t_left": int(t_left),
                        "t_right": int(t_right),
                        "okIntegral": "" if okIntegral is None else int(okIntegral),
                        "resUsd": "" if res_usd is None else float(res_usd),
                        "model_time_sum_sec": float(model_time_sum_sec),
                        "case": int(case),
                        "feasible": label,
                        "integrality": integral_label,
                        "exported": int(exported),
                        "accepted": int(accepted),
                        "obj_best": int(obj_best),
                        "obj_z": "" if obj_z is None else float(obj_z),
                        "exported_obj": "" if cand_obj is None else int(cand_obj),
                        "stop_early": int(stop_early),
                    }
                )

                sub_id += 1

                if stop_early:
                    break

            tested_activities += 1

            if max_subproblems is not None and tested_activities >= max_subproblems:
                stop_early = True
                if verbose:
                    _log(
                        "[AO-FR] stopping early because max_subproblems (activities) was reached: "
                        + str(max_subproblems),
                        log_path,
                    )
                break

            if stop_early:
                break

            if not got_feasible_for_i and verbose:
                _log("[AO-FR] activity " + str(i) + " no feasible solution within retries", log_path)

        if stop_early:
            break

        if not improved_in_pass:
            no_improve += 1
        else:
            no_improve = 0

        if max_no_improve > 0 and no_improve >= max_no_improve:
            if verbose:
                _log("[AO-FR] stopping because no improvement for " + str(no_improve) + " passes", log_path)
            break

    if verbose:
        _log("", log_path)
        _log("[AO-FR] FINAL obj = " + str(obj_best), log_path)
        _log("[AO-FR] total_time = " + str(round(float(model_time_sum_sec), 3)), log_path)

    fr = FRResult(
        S=dict(S_best),
        obj=int(obj_best),
        feasible=found_integral_solution,
        violated=0,
        allowed=0,
        violated_list=[],
        runtime_total_sec=float(model_time_sum_sec),
        iter_best=int(iter_best),
    )
    return fr, iter_rows