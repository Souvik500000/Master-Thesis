# AO_FR_Gurobi_group.py
#
# Group-based Activity-Oriented Relax-and-Fix for FSRCPSP.
#
# KEY DIFFERENCE from original AO_FR_Gurobi.py:
#   Original : fixes ONE activity per subproblem  -> n subproblems
#   This file: fixes ONE GROUP per subproblem     -> n_groups subproblems
#
# Example for n=10, 4 groups:
#   Subproblem 1: binary={1,2,3}        fixed={}           relaxed={4..10}
#   Subproblem 2: binary={4,5,9}        fixed={1,2,3}      relaxed={6,7,8,10}
#   Subproblem 3: binary={6,7}          fixed={1,2,3,4,5,9} relaxed={8,10}
#   Subproblem 4: binary={8,10}         fixed={1..9}       relaxed={}
#
# Result: 4 subproblems instead of 10. Each subproblem fixes a
# semantically coherent group (similar resource/stochastic profile)
# simultaneously, giving Gurobi richer context per iteration.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from decimal import Decimal
import os
import time
import random

from SSGS import ssgs_est_worst_case

from AO_Model import (
    AO_Model,
    buildModel,
    reset_bounds_for_subproblem_Mz,
    set_warmstart_for_subproblem_Mz,
)

from activity_grouping import ActivityGrouper


# =============================================================================
# RESULT DATACLASS  (identical to original)
# =============================================================================

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


# =============================================================================
# UTILITIES  (identical to original)
# =============================================================================

def makespan_from_S(S: Dict[int, int], n_jobs: int) -> int:
    return int(S.get(n_jobs - 1, 10**9))


def _fmt_time(sec: float) -> str:
    return f"{sec:.3f}s"


def _fmt_set(ints: List[int]) -> str:
    if not ints:
        return "{}"
    return "{" + ",".join(str(x) for x in sorted(ints)) + "}"


def _log(msg: str, log_path: Optional[str]) -> None:
    print(msg)
    if log_path is not None:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")


def _clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def _window_for_activity(es_i: int, ls_i: int, s_center: int, w: int) -> Tuple[int, int]:
    left  = max(es_i, s_center - w)
    right = min(ls_i, s_center + w)
    if right < left:
        right = left
    return int(left), int(right)


def _extract_schedule(
    ao: AO_Model,
    data,
    binary_and_fixed: Optional[set] = None,
) -> Dict[int, int]:
    """
    Extract start times from the solved model.
    - Fixed and binary activities: read from xB (integer variable)
    - Relaxed activities:          read from xC (continuous variable)
    Both contribute to xEff, so reading xEff also works — but for
    relaxed activities the xC value is fractional; argmax(xEff) still
    gives the correct "most likely" start time for warm-starting.
    """
    S: Dict[int, int] = {}
    for i in ao.activities:
        es_i, ls_i = int(data.es[i]), int(data.ls[i])
        best_t, best_val = es_i, -1.0
        for t in range(es_i, ls_i + 1):
            val = ao.xEff[i, t].X   # xB+xC — argmax works for both
            if val > best_val:
                best_val = val
                best_t   = t
        S[i] = int(best_t)
    return S


def _check_integrality(ao: AO_Model, data, tol: float = 1e-6) -> bool:
    """Only check xB variables — relaxed activities use xC which is always fractional."""
    for i in ao.activities:
        for t in range(int(data.es[i]), int(data.ls[i]) + 1):
            val = ao.xB[i, t].X
            if abs(val) > tol and abs(val - 1.0) > tol:
                return False
    return True


# =============================================================================
# GROUP-BASED BOUNDS SETTER
# replaces set_bounds_for_subproblem_Mz for a list of binary activities
# =============================================================================

def set_bounds_for_group_subproblem(
    ao:                AO_Model,
    data,
    group:             List[int],          # activities to set as BINARY this round
    windows:           Dict[int, Tuple[int, int]],  # {job: (t_left, t_right)}
    fixed_start_times: Dict[int, int],
) -> None:
    """
    Set variable bounds for a group-based subproblem:
      - fixed activities  : xB fixed at their start time, xC=0
      - group activities  : xB binary within their window, xC=0
      - relaxed activities: xB=0, xC continuous [0,1]
    """
    fixed_set = set(fixed_start_times.keys())
    group_set = set(group)

    for i in ao.activities:
        es_i = int(data.es[i])
        ls_i = int(data.ls[i])

        for t in range(es_i, ls_i + 1):

            if i in fixed_set:
                # ── FIXED: pin to known start time ───────────────
                s_fix = int(fixed_start_times[i])
                ao.xC[i, t].LB = 0.0
                ao.xC[i, t].UB = 0.0
                if t == s_fix:
                    ao.xB[i, t].LB = 1.0
                    ao.xB[i, t].UB = 1.0
                else:
                    ao.xB[i, t].LB = 0.0
                    ao.xB[i, t].UB = 0.0

            elif i in group_set:
                # ── BINARY: free within window ────────────────────
                t_left, t_right = windows.get(i, (es_i, ls_i))
                ao.xC[i, t].LB = 0.0
                ao.xC[i, t].UB = 0.0
                if t_left <= t <= t_right:
                    ao.xB[i, t].LB = 0.0
                    ao.xB[i, t].UB = 1.0
                else:
                    ao.xB[i, t].LB = 0.0
                    ao.xB[i, t].UB = 0.0

            else:
                # ── RELAXED: continuous only ──────────────────────
                ao.xB[i, t].LB = 0.0
                ao.xB[i, t].UB = 0.0
                ao.xC[i, t].LB = 0.0
                ao.xC[i, t].UB = 1.0

    ao.model.update()


def set_warmstart_for_group_subproblem(
    ao:                AO_Model,
    data,
    S_warm:            Dict[int, int],
    group:             List[int],
    windows:           Dict[int, Tuple[int, int]],
    fixed_start_times: Dict[int, int],
) -> None:
    """Warm-start hint for a group subproblem."""
    fixed_set = set(fixed_start_times.keys())
    group_set = set(group)

    for _, var in ao.xB.items():
        var.Start = 0.0
    for _, var in ao.xC.items():
        var.Start = 0.0
    for _, var in ao.xEff.items():
        var.Start = 0.0

    for i in ao.activities:
        es_i = int(data.es[i])
        ls_i = int(data.ls[i])

        if i in fixed_set:
            s_i = int(fixed_start_times[i])
            ao.xB[i, s_i].Start   = 1.0
            ao.xC[i, s_i].Start   = 0.0
            ao.xEff[i, s_i].Start = 1.0

        elif i in group_set:
            s_i = int(S_warm.get(i, es_i))
            t_left, t_right = windows.get(i, (es_i, ls_i))
            s_i = _clamp(s_i, t_left, t_right)
            ao.xB[i, s_i].Start   = 1.0
            ao.xC[i, s_i].Start   = 0.0
            ao.xEff[i, s_i].Start = 1.0

        else:
            s_i = int(S_warm.get(i, es_i))
            s_i = _clamp(s_i, es_i, ls_i)
            ao.xB[i, s_i].Start   = 0.0
            ao.xC[i, s_i].Start   = 1.0
            ao.xEff[i, s_i].Start = 1.0

    ao.model.update()


# =============================================================================
# MAIN SOLVER
# =============================================================================

def group_oriented_fix_and_relax(
    data,
    number_scen:          int,
    epsilon,
    time_limit_sec:       float,
    time_limit_model:     float,
    init_S:               Optional[Dict[int, int]] = None,
    verbose:              bool  = True,
    working_directory:    str   = ".",
    instance_name:        str   = "inst",
    max_no_improve:       int   = 0,
    log_to_file:          bool  = True,
    max_retries_per_group: int  = 2,
    # Grouper parameters
    corr_threshold:       float = 0.95,
    min_group_size:       int   = 1,
    max_group_size:       int   = 4,
    merge_threshold:      float = 2.5,
    stop_on_first_feasible: bool = True,
) -> Tuple[FRResult, List[Dict[str, Any]]]:

    # ── Validate inputs ───────────────────────────────────────────
    epsilon_dec = epsilon if isinstance(epsilon, Decimal) else Decimal(str(epsilon))
    CL_dec      = Decimal("1.0") - epsilon_dec

    if time_limit_sec   <= 0: raise ValueError("time_limit_sec must be > 0")
    if time_limit_model <= 0: raise ValueError("time_limit_model must be > 0")
    if max_retries_per_group <= 0: raise ValueError("max_retries_per_group must be >= 1")

    start_wall = time.time()
    n          = int(getattr(data, "n_jobs_including_dummy"))

    # ── Logging ───────────────────────────────────────────────────
    results_dir = os.path.join(working_directory, "Results")
    os.makedirs(results_dir, exist_ok=True)
    log_path: Optional[str] = None
    if log_to_file:
        log_path = os.path.join(results_dir, "ao_fr_group_log.txt")
        try:
            if os.path.exists(log_path):
                os.remove(log_path)
        except OSError:
            pass

    es: Dict[int, int] = getattr(data, "es")
    ls: Dict[int, int] = getattr(data, "ls")

    # ── SSGS warm-start ───────────────────────────────────────────
    if init_S is None:
        base   = ssgs_est_worst_case(data, scenario_keep=list(range(1, number_scen + 1)))
        S_ssgs = dict(base.S)
        S_best = dict(base.S)
    else:
        S_ssgs = dict(init_S)
        S_best = dict(init_S)

    obj_best  = makespan_from_S(S_best, n)
    iter_best = -1
    S_prime   = dict(S_best)

    fixed_start: Dict[int, int] = {0: 0}
    for d_ in (S_prime, S_best, S_ssgs):
        d_[0] = 0

    # ── Build Gurobi model ────────────────────────────────────────
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
    fixed_start[0] = _clamp(0, es0, max(es0, ls0))
    for d_ in (S_prime, S_best, S_ssgs):
        d_[0] = fixed_start[0]

    # ── Build activity groups ─────────────────────────────────────
    grouper = ActivityGrouper(
        data            = data,
        number_scen     = number_scen,
        corr_threshold  = corr_threshold,
        min_group_size  = min_group_size,
        max_group_size  = max_group_size,
        merge_threshold = merge_threshold,
        verbose         = verbose,
    )
    groups = grouper.group()   # List[List[int]]

    # ── Verbose header ────────────────────────────────────────────
    if verbose:
        _log("", log_path)
        _log(f"[GO-FR] instance       = {instance_name}", log_path)
        _log(f"[GO-FR] epsilon        = {epsilon_dec}, CL = {CL_dec}", log_path)
        _log(f"[GO-FR] time_limit_sec = {time_limit_sec}", log_path)
        _log(f"[GO-FR] time_limit_model = {time_limit_model}", log_path)
        _log(f"[GO-FR] initial obj    = {obj_best}", log_path)
        _log(f"[GO-FR] n_groups       = {len(groups)}", log_path)
        for gi, g in enumerate(groups):
            _log(f"[GO-FR] group {gi+1:2d}       = {g}", log_path)

    # ── Main loop ─────────────────────────────────────────────────
    sub_id               = 0
    no_improve           = 0
    stop_early           = False
    found_integral       = False
    iter_rows: List[Dict[str, Any]] = []
    model_time_sum_sec   = 0.0

    while (time.time() - start_wall) < time_limit_sec and not stop_early:
        improved_in_pass = False

        for group in groups:
            if stop_early:
                break
            if (time.time() - start_wall) >= time_limit_sec:
                break

            # Skip source/sink if accidentally included
            group = [i for i in group if i not in fixed_start or i == 0]
            group = [i for i in group if i != 0 and i != n - 1]
            if not group:
                continue

            z                    = 0
            got_feasible_for_grp = False

            while (not got_feasible_for_grp) and z < max_retries_per_group \
                    and (time.time() - start_wall) < time_limit_sec:

                if stop_early:
                    break

                z += 1

                # ── Build per-activity windows for the group ──────
                windows: Dict[int, Tuple[int, int]] = {}
                for i in group:
                    es_i     = int(es.get(i, 0))
                    ls_i     = int(ls.get(i, es_i))
                    ls_i     = max(es_i, ls_i)
                    # w        = random.randint(1, 10)
                    slack_i  = max(0,ls_i - es_i)
                    w = max(2, int(slack_i * 0.15))
                    s_center = _clamp(int(S_prime.get(i, es_i)), es_i, ls_i)
                    t_left   = max(es_i, s_center - w)
                    t_right  = min(ls_i, s_center + w)
                    windows[i] = (t_left, t_right)

                fixed_list  = sorted(fixed_start.keys())
                binary_list = sorted(group)
                relaxed_list= [j for j in range(n)
                               if j not in fixed_start and j not in group]

                if verbose:
                    _log("", log_path)
                    _log(f"[GO-FR] group={binary_list}  retry={z}", log_path)
                    _log(f"[GO-FR] fixed    = {_fmt_set(fixed_list)}", log_path)
                    _log(f"[GO-FR] binary   = {_fmt_set(binary_list)}", log_path)
                    _log(f"[GO-FR] relaxed  = {_fmt_set(relaxed_list)}", log_path)
                    for i in group:
                        _log(f"[GO-FR]   job {i:3d} window = {windows[i]}", log_path)

                elapsed_so_far = time.time() - start_wall
                remaining      = max(0.0, time_limit_sec - elapsed_so_far)
                eff_limit      = max(1.0, min(float(time_limit_model), remaining))

                # ── Set bounds and warm-start ─────────────────────
                reset_bounds_for_subproblem_Mz(ao=ao, data=data)

                set_bounds_for_group_subproblem(
                    ao                = ao,
                    data              = data,
                    group             = group,
                    windows           = windows,
                    fixed_start_times = fixed_start,
                )

                set_warmstart_for_group_subproblem(
                    ao                = ao,
                    data              = data,
                    S_warm            = S_prime,
                    group             = group,
                    windows           = windows,
                    fixed_start_times = fixed_start,
                )

                ao.model.Params.TimeLimit = float(eff_limit)
                ao.model.optimize()

                res_usd      = float(ao.model.Runtime)
                model_time_sum_sec += res_usd
                has_solution = ao.model.SolCount > 0
                obj_z        = float(ao.model.ObjVal) if has_solution else None
                label        = "feasible" if has_solution else "infeasible"

                ok_integral: Optional[int] = None
                if has_solution:
                    ok_integral = 1 if _check_integrality(ao, data) else 0

                integral_label = "NA"
                if ok_integral == 1:
                    integral_label = "integral"
                elif ok_integral == 0:
                    integral_label = "non-integral"

                case       = 5
                exported   = False
                accepted   = False
                cand_obj: Optional[int]        = None
                cand_S:   Optional[Dict[int, int]] = None

                if has_solution:
                    case   = 2 if ok_integral == 1 else 1
                    cand_S = _extract_schedule(
                        ao, data,
                        binary_and_fixed=set(fixed_start.keys()) | set(group)
                    )
                    exported = True

                    # Clamp all activities
                    for j in range(n):
                        es_j = int(es.get(j, 0))
                        ls_j = max(es_j, int(ls.get(j, es_j)))
                        cand_S[j] = _clamp(int(cand_S.get(j, es_j)), es_j, ls_j)
                    cand_S[0] = int(fixed_start[0])

                    cand_obj = makespan_from_S(cand_S, n)
                    got_feasible_for_grp = True

                    # Fix all group members at their found start times
                    for i in group:
                        fixed_start[i] = int(cand_S[i])

                    S_prime = dict(cand_S)

                    if cand_obj < obj_best:
                        S_best    = dict(cand_S)
                        obj_best  = int(cand_obj)
                        iter_best = int(sub_id)
                        accepted  = True
                        improved_in_pass = True
                        no_improve = 0

                    if ok_integral == 1:
                        found_integral = True
                        if stop_on_first_feasible:
                            stop_early = True
                        # Do NOT stop early — continue to next group
                        # to allow further improvement

                if case == 5:
                    if z < max_retries_per_group:
                        _log(
                            f"Infeasible for group {binary_list} "
                            f"on try {z} of {max_retries_per_group} — retry",
                            log_path,
                        )
                    else:
                        # Fallback: fix each group member at SSGS time
                        for i in group:
                            es_i = int(es.get(i, 0))
                            ls_i = max(es_i, int(ls.get(i, es_i)))
                            s_i  = _clamp(int(S_ssgs.get(i, es_i)), es_i, ls_i)
                            fixed_start[i] = s_i
                            S_prime[i]     = s_i
                        got_feasible_for_grp = True
                        _log(
                            f"Infeasible for group {binary_list} "
                            f"on last try — fixed to SSGS times",
                            log_path,
                        )

                if verbose:
                    zfw = obj_z if obj_z is not None else (cand_obj if cand_obj else obj_best)
                    _log(
                        f"[GO-FR-STATUS] group={binary_list} | {label} | "
                        f"{integral_label} | case={case} | "
                        f"ZFW={zfw} | S*={obj_best} | t={_fmt_time(model_time_sum_sec)}",
                        log_path,
                    )

                iter_rows.append({
                    "iter":              int(sub_id),
                    "group":             str(binary_list),
                    "group_size":        len(binary_list),
                    "retry":             int(z),
                    "ok_integral":       "" if ok_integral is None else int(ok_integral),
                    "res_usd":           float(res_usd),
                    "model_time_sum_sec": float(model_time_sum_sec),
                    "case":              int(case),
                    "feasible":          label,
                    "integrality":       integral_label,
                    "exported":          int(exported),
                    "accepted":          int(accepted),
                    "obj_best":          int(obj_best),
                    "obj_z":             "" if obj_z is None else float(obj_z),
                    "exported_obj":      "" if cand_obj is None else int(cand_obj),
                    "stop_early":        int(stop_early),
                })

                sub_id += 1

                if stop_early:
                    break

        if stop_early:
            break

        if not improved_in_pass:
            no_improve += 1
        else:
            no_improve = 0

        if max_no_improve > 0 and no_improve >= max_no_improve:
            if verbose:
                _log(f"[GO-FR] stopping — no improvement for {no_improve} passes", log_path)
            break

    if verbose:
        _log("", log_path)
        _log(f"[GO-FR] FINAL obj = {obj_best}", log_path)
        _log(f"[GO-FR] total_time = {round(model_time_sum_sec, 3)}", log_path)

    return FRResult(
        S                 = dict(S_best),
        obj               = int(obj_best),
        feasible          = found_integral,
        violated          = 0,
        allowed           = 0,
        violated_list     = [],
        runtime_total_sec = float(model_time_sum_sec),
        iter_best         = int(iter_best),
    ), iter_rows