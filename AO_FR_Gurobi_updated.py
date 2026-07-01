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

from SSGS import ssgs_est_worst_case

from AO_Model import (
    AO_Model,
    buildModel,
    reset_bounds_for_subproblem_Mz,
    set_warmstart_for_subproblem_Mz,
)

from activity_grouping import ActivityGrouper, DEFAULT_FEATURE_COLS


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


def _compute_d_bounds(data) -> Tuple[Dict[Tuple[int, int], int],
                                      Dict[Tuple[int, int], int]]:
    """
    Per-(activity, scenario) duration bounds:
       d_min(i, π) = max_k ceil(W(π,k,i) / uR(i,k))   for uR > 0
       d_max(i, π) = max_k ceil(W(π,k,i) / lR(i,k))   for lR > 0
    Falls back to 0 / T_max if no positive rate. Used to tighten y-bounds
    once xB is fixed: y must be 1 while activity is running, 0 after it
    must be done.
    """
    activities = list(range(int(data.n_jobs_including_dummy)))
    resources  = list(range(1, int(data.n_renewable_resources) + 1))
    scenarios  = sorted(int(s) for s in data.scenarios.keys())
    T_max = int(max(data.ls.values())) if getattr(data, "ls", None) else 0

    d_min: Dict[Tuple[int, int], int] = {}
    d_max: Dict[Tuple[int, int], int] = {}

    for i in activities:
        for pi in scenarios:
            dmin_v, dmax_v = 0, 0
            scen_pi = data.scenarios.get(pi, {})
            for k_idx, k in enumerate(resources):
                arr = scen_pi.get(k, [])
                w = int(arr[i]) if i < len(arr) else 0
                if w <= 0:
                    continue
                uR_v = int(data.uR[i][k_idx]) if data.uR[i][k_idx] > 0 else 0
                lR_v = int(data.lR[i][k_idx]) if data.lR[i][k_idx] > 0 else 0
                if uR_v > 0:
                    dmin_v = max(dmin_v, (w + uR_v - 1) // uR_v)
                if lR_v > 0:
                    dmax_v = max(dmax_v, (w + lR_v - 1) // lR_v)
            if dmax_v == 0:
                dmax_v = T_max          # no positive lR -> unbounded above
            d_min[(i, pi)] = int(dmin_v)
            d_max[(i, pi)] = int(dmax_v)
    return d_min, d_max


def _tighten_y_bounds_for_fixed_activities(
    ao,
    fixed_start_times: Dict[int, int],
    d_min: Dict[Tuple[int, int], int],
    d_max: Dict[Tuple[int, int], int],
) -> None:
    """
    For each FIXED activity i (xB[i, s_i] = 1), tighten y bounds:
       y = 1   for t < s_i + d_min[i,π]
       y free  for t in [s_i + d_min, s_i + d_max - 1]
       y = 0   for t >= s_i + d_max

    This drastically shrinks the unfixed-binary count at j60+ scale,
    where the y tensor (n_jobs × T × n_scen) dominates the per-subproblem
    MIP. Expected: ~200k unfixed binaries -> ~10-30k at j60 K=1.
    """
    n_t = len(ao.time_points)
    for i, s in fixed_start_times.items():
        s_i = int(s)
        for pi in ao.scenarios:
            pi_int = int(pi)
            dmin = int(d_min.get((i, pi_int), 0))
            dmax = int(d_max.get((i, pi_int), n_t))
            must_active_until = s_i + dmin    # exclusive
            done_from         = s_i + dmax    # inclusive
            for t in ao.time_points:
                if t < must_active_until:
                    ao.y[i, t, pi].LB = 1.0
                    ao.y[i, t, pi].UB = 1.0
                elif t >= done_from:
                    ao.y[i, t, pi].LB = 0.0
                    ao.y[i, t, pi].UB = 0.0
                else:
                    ao.y[i, t, pi].LB = 0.0
                    ao.y[i, t, pi].UB = 1.0
    ao.model.update()


def _tighten_y_bounds_for_unfixed_activities(
    ao,
    fixed_start_times: Dict[int, int],
    data,
    d_max: Dict[Tuple[int, int], int],
) -> None:
    """
    Note on y semantics: y[i, t, π] = 1 means "activity i has NOT yet
    finished by time t". Constraint Test_2 in AO_Model already forces
    y = 1 for all t <= ES_i, so we cannot pin y = 0 below ES_i.
    The remaining free win: pin y = 0 for t >= LS_i + d_max(i, π),
    since the activity must be finished by that point in any feasible
    schedule (s_i <= LS_i and duration <= d_max(i, π)).
    """
    fixed_set = set(fixed_start_times.keys())
    T_last = max(ao.time_points) if ao.time_points else 0
    for i in ao.activities:
        if i in fixed_set:
            continue
        ls_i = int(data.ls.get(i, T_last))
        for pi in ao.scenarios:
            dmax_i = int(d_max.get((i, int(pi)), T_last))
            t_done = ls_i + dmax_i
            for t in ao.time_points:
                if t >= t_done:
                    ao.y[i, t, pi].LB = 0.0
                    ao.y[i, t, pi].UB = 0.0
    ao.model.update()


def _compute_windows(
    group:    List[int],
    es:       Dict[int, int],
    ls:       Dict[int, int],
    S_prime:  Dict[int, int],
    slack_frac: float,
) -> Dict[int, Tuple[int, int]]:
    """
    Compute per-activity search windows for a group.
    slack_frac=0.15  -> narrow window (15% of slack around S_prime center)
    slack_frac=1.0   -> full window   [es_i, ls_i]
    S_prime[i] is always inside the returned window by construction.
    """
    windows: Dict[int, Tuple[int, int]] = {}
    for i in group:
        es_i     = int(es.get(i, 0))
        ls_i     = max(es_i, int(ls.get(i, es_i)))
        slack_i  = max(0, ls_i - es_i)
        w        = max(2, int(slack_i * slack_frac))
        s_center = _clamp(int(S_prime.get(i, es_i)), es_i, ls_i)
        t_left   = max(es_i, s_center - w)
        t_right  = min(ls_i, s_center + w)
        windows[i] = (t_left, t_right)
    return windows


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

def _group_features(group: List[int], feat_df) -> Dict[str, float]:
    """Aggregate per-activity features into group-level features for the NN."""
    import numpy as np
    rows = feat_df[feat_df.index.isin(group)]
    if rows.empty:
        return {}
    feats: Dict[str, float] = {"group_size": float(len(group))}
    for col in rows.columns:
        vals = rows[col].dropna().values.astype(float)
        if len(vals) == 0:
            continue
        feats[f"{col}_mean"] = float(np.mean(vals))
        feats[f"{col}_min"]  = float(np.min(vals))
        feats[f"{col}_max"]  = float(np.max(vals))
    return feats


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
    # Grouper parameters
    corr_threshold:       float = 0.95,
    min_group_size:       int   = 1,
    max_group_size:       int   = 4,
    merge_threshold:      float = 2.5,
    merge_percentile:     Optional[float] = None,
    stop_on_first_feasible: bool = True,
    feature_cols:         Optional[List[str]] = DEFAULT_FEATURE_COLS,
    # NN urgency model (optional)
    urgency_model_path:   Optional[str] = None,
    log_group_features:   bool = False,
    # Per-group scenario reduction (online, no training)
    enable_per_group_scenario_reduction: bool = False,
    n_keep_per_group:     int = 5,
    # Time aggregation (bucket physical time into windows of size alpha)
    time_alpha:           int = 1,
    # Gurobi threads per solve (reduce when running parallel K experiments)
    gurobi_threads:       int = 4,
) -> Tuple[FRResult, List[Dict[str, Any]]]:

    # ── Validate inputs ───────────────────────────────────────────
    epsilon_dec = epsilon if isinstance(epsilon, Decimal) else Decimal(str(epsilon))
    CL_dec      = Decimal("1.0") - epsilon_dec

    if time_limit_sec   <= 0: raise ValueError("time_limit_sec must be > 0")
    if time_limit_model <= 0: raise ValueError("time_limit_model must be > 0")

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

    # ── Time aggregation: switch to aggregated grid for the F&R run ──
    data_phys = data
    if time_alpha > 1:
        from time_aggregation import aggregate_instance
        data = aggregate_instance(data_phys, time_alpha)
        # Recompute SSGS on the aggregated grid — translating a physical
        # warmstart via floor() can break precedence on the aggregated grid
        # (tight pairs with d=1 can collapse into the same bucket).
        init_S = None

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

    # ── Stash originals (per-group reduction rebinds these in-loop) ──
    data_orig         = data
    number_scen_orig  = number_scen

    # ── Build Gurobi model ────────────────────────────────────────
    if enable_per_group_scenario_reduction:
        # Defer initial build — each iteration builds its own reduced model.
        ao = None
    else:
        ao = buildModel(
            data=data,
            number_scen=number_scen,
            epsilon=epsilon_dec,
            threads=gurobi_threads,
            mip_gap=1e-5,
            time_limit_seconds=None,
        )

    # ── Pre-compute duration bounds (used to tighten y-bounds per iter) ──
    d_min_bounds, d_max_bounds = _compute_d_bounds(data_orig)
    d_min_bounds_orig = d_min_bounds
    d_max_bounds_orig = d_max_bounds

    es0 = int(es.get(0, 0))
    ls0 = int(ls.get(0, es0))
    fixed_start[0] = _clamp(0, es0, max(es0, ls0))
    for d_ in (S_prime, S_best, S_ssgs):
        d_[0] = fixed_start[0]

    # ── Build activity groups ─────────────────────────────────────
    grouper = ActivityGrouper(
        data             = data,
        number_scen      = number_scen,
        corr_threshold   = corr_threshold,
        min_group_size   = min_group_size,
        max_group_size   = max_group_size,
        merge_threshold  = merge_threshold,
        merge_percentile = merge_percentile,
        verbose          = verbose,
        feature_cols     = feature_cols,
    )
    groups = grouper.group()   # List[List[int]]

    # ── Extract per-group features (for NN logging / reordering) ──
    feat_df = grouper._feat_df if hasattr(grouper, "_feat_df") else None
    group_feat_list: List[Optional[Dict[str, float]]] = [None] * len(groups)
    if feat_df is not None:
        for gi, g in enumerate(groups):
            group_feat_list[gi] = _group_features(g, feat_df)

    # ── NN urgency reordering (optional) ─────────────────────────
    if urgency_model_path is not None and feat_df is not None:
        try:
            import numpy as _np
            from urgency_nn import UrgencyNNNumpy, FEATURE_COLS
            nn_model = UrgencyNNNumpy(path=urgency_model_path)
            X_groups = _np.array(
                [[( group_feat_list[gi] or {}).get(c, 0.0) for c in FEATURE_COLS]
                 for gi in range(len(groups))],
                dtype=_np.float32,
            )
            scores = nn_model.predict(X_groups)
            order  = sorted(range(len(groups)), key=lambda i: -float(scores[i]))
            groups          = [groups[i] for i in order]
            group_feat_list = [group_feat_list[i] for i in order]
            if verbose:
                _log(f"[GO-FR] NN urgency reorder: {order} scores={scores.round(3).tolist()}", log_path)
        except Exception as e:
            _log(f"[GO-FR] NN reorder skipped: {e}", log_path)

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
    n_groups             = len(groups)

    for group_idx, group in enumerate(groups):
        if stop_early:
            break
        if (time.time() - start_wall) >= time_limit_sec:
            break

        # Skip source/sink if accidentally included
        group = [i for i in group if i not in fixed_start or i == 0]
        group = [i for i in group if i != 0 and i != n - 1]
        if not group:
            continue

        # ── Per-group scenario reduction: rebuild ao on K medoids ──
        if enable_per_group_scenario_reduction:
            from scenario_reduction import reduce_kmedoids_for_subset
            kmedoids_acts = [i for i in group if 0 < i < n - 1] or list(group)
            data_red, kept_ids = reduce_kmedoids_for_subset(
                data_orig,
                n_keep=n_keep_per_group,
                activities=kmedoids_acts,
            )
            n_scen_red = len(kept_ids)
            if ao is not None:
                try: ao.model.dispose()
                except Exception: pass
            ao = buildModel(
                data=data_red,
                number_scen=n_scen_red,
                epsilon=epsilon_dec,
                threads=gurobi_threads,
                mip_gap=1e-5,
                time_limit_seconds=None,
            )
            # Skip warm-start subMIP completion: at j60 scale this
            # eats the entire per-iteration TL with 0 nodes explored.
            # With all xB pinned via bounds, the warmstart contributes
            # little — going cold is faster than thrashing.
            ao.model.Params.StartNodeLimit = 0
            data         = data_red
            number_scen  = n_scen_red
            d_min_bounds, d_max_bounds = _compute_d_bounds(data_red)
            if verbose:
                _log(
                    f"[GO-FR] per-group scen-red: kept={kept_ids}",
                    log_path,
                )

        binary_list  = sorted(group)
        fixed_list   = sorted(fixed_start.keys())
        relaxed_list = [j for j in range(n)
                        if j not in fixed_start and j not in group]

        # ── Attempt 1: narrow window (15% of slack) ──────────
        windows = _compute_windows(group, es, ls, S_prime, slack_frac=0.15)

        if verbose:
            _log("", log_path)
            _log(f"[GO-FR] group={binary_list}  ({group_idx+1}/{n_groups})", log_path)
            _log(f"[GO-FR] fixed    = {_fmt_set(fixed_list)}", log_path)
            _log(f"[GO-FR] binary   = {_fmt_set(binary_list)}", log_path)
            _log(f"[GO-FR] relaxed  = {_fmt_set(relaxed_list)}", log_path)
            for i in group:
                _log(f"[GO-FR]   job {i:3d} window = {windows[i]}", log_path)

        elapsed_so_far   = time.time() - start_wall
        remaining        = max(0.0, time_limit_sec - elapsed_so_far)
        groups_remaining = max(1, n_groups - group_idx)
        eff_limit        = max(1.0, min(float(time_limit_model),
                                        remaining / groups_remaining))

        reset_bounds_for_subproblem_Mz(ao=ao, data=data)
        set_bounds_for_group_subproblem(
            ao=ao, data=data, group=group,
            windows=windows, fixed_start_times=fixed_start,
        )
        _tighten_y_bounds_for_fixed_activities(
            ao, fixed_start, d_min_bounds, d_max_bounds
        )
        _tighten_y_bounds_for_unfixed_activities(
            ao, fixed_start, data, d_max_bounds
        )
        if not enable_per_group_scenario_reduction:
            set_warmstart_for_group_subproblem(
                ao=ao, data=data, S_warm=S_prime,
                group=group, windows=windows, fixed_start_times=fixed_start,
            )
        else:
            # Skip warmstart: at j60 scale Gurobi's user-MIP-start LP
            # completion eats the entire TL with 0 nodes branched. xB
            # bounds already encode the schedule-so-far; let Gurobi
            # solve cold from there.
            ao.model.NumStart = 0
        ao.model.Params.TimeLimit = float(eff_limit)
        ao.model.optimize()

        res_usd            = float(ao.model.Runtime)
        model_time_sum_sec += res_usd
        has_solution       = ao.model.SolCount > 0

        # ── Fallback: full [es_i, ls_i] window if narrow solve failed ──
        if not has_solution and (time.time() - start_wall) < time_limit_sec:
            if verbose:
                _log(
                    f"[GO-FR] narrow window yielded no solution for group "
                    f"{binary_list} — retrying with full slack window",
                    log_path,
                )
            windows = _compute_windows(group, es, ls, S_prime, slack_frac=1.0)

            elapsed_so_far   = time.time() - start_wall
            remaining        = max(0.0, time_limit_sec - elapsed_so_far)
            groups_remaining = max(1, n_groups - group_idx)
            eff_limit        = max(1.0, min(float(time_limit_model),
                                            remaining / groups_remaining))

            reset_bounds_for_subproblem_Mz(ao=ao, data=data)
            set_bounds_for_group_subproblem(
                ao=ao, data=data, group=group,
                windows=windows, fixed_start_times=fixed_start,
            )
            _tighten_y_bounds_for_fixed_activities(
                ao, fixed_start, d_min_bounds, d_max_bounds
            )
            _tighten_y_bounds_for_unfixed_activities(
                ao, fixed_start, data, d_max_bounds
            )
            if not enable_per_group_scenario_reduction:
                set_warmstart_for_group_subproblem(
                    ao=ao, data=data, S_warm=S_prime,
                    group=group, windows=windows, fixed_start_times=fixed_start,
                )
            else:
                ao.model.NumStart = 0
            ao.model.Params.TimeLimit = float(eff_limit)
            ao.model.optimize()

            res_usd_fb          = float(ao.model.Runtime)
            model_time_sum_sec += res_usd_fb
            res_usd            += res_usd_fb
            has_solution        = ao.model.SolCount > 0

        obj_z  = float(ao.model.ObjVal) if has_solution else None
        label  = "feasible" if has_solution else "infeasible"

        ok_integral: Optional[int] = None
        if has_solution:
            ok_integral = 1 if _check_integrality(ao, data) else 0

        integral_label = "NA"
        if ok_integral == 1:
            integral_label = "integral"
        elif ok_integral == 0:
            integral_label = "non-integral"

        case     = 5
        exported = False
        accepted = False
        cand_obj: Optional[int]            = None
        cand_S:   Optional[Dict[int, int]] = None

        if has_solution:
            case   = 2 if ok_integral == 1 else 1
            cand_S = _extract_schedule(
                ao, data,
                binary_and_fixed=set(fixed_start.keys()) | set(group)
            )
            exported = True

            for j in range(n):
                es_j = int(es.get(j, 0))
                ls_j = max(es_j, int(ls.get(j, es_j)))
                cand_S[j] = _clamp(int(cand_S.get(j, es_j)), es_j, ls_j)
            cand_S[0] = int(fixed_start[0])

            cand_obj = makespan_from_S(cand_S, n)

            for i in group:
                fixed_start[i] = int(cand_S[i])
            S_prime = dict(cand_S)

            if cand_obj < obj_best:
                S_best    = dict(cand_S)
                obj_best  = int(cand_obj)
                iter_best = int(sub_id)
                accepted  = True
                no_improve = 0
            elif found_integral:
                # Only penalise non-improvement once we already have a
                # feasible solution — before that we are still searching
                # for feasibility, not improving quality.
                no_improve += 1

            if max_no_improve > 0 and no_improve >= max_no_improve:
                if verbose:
                    _log(
                        f"[GO-FR] stopping — no improvement "
                        f"for {no_improve} consecutive groups",
                        log_path,
                    )
                stop_early = True

            if ok_integral == 1:
                found_integral = True
                if stop_on_first_feasible:
                    stop_early = True

        if case == 5:
            # Both solves failed — fix to S_prime (always within [es_i, ls_i])
            for i in group:
                es_i = int(es.get(i, 0))
                ls_i = max(es_i, int(ls.get(i, es_i)))
                fixed_start[i] = _clamp(int(S_prime.get(i, es_i)), es_i, ls_i)
            _log(
                f"[GO-FR] both solves failed for group {binary_list} "
                f"— fixed to S_prime times",
                log_path,
            )
            if found_integral:
                no_improve += 1
            if max_no_improve > 0 and no_improve >= max_no_improve:
                _log(
                    f"[GO-FR] stopping — {no_improve} consecutive "
                    f"infeasible/non-improving groups",
                    log_path,
                )
                stop_early = True

        if verbose:
            zfw = obj_z if obj_z is not None else (cand_obj if cand_obj else obj_best)
            _log(
                f"[GO-FR-STATUS] group={binary_list} | {label} | "
                f"{integral_label} | case={case} | "
                f"ZFW={zfw} | S*={obj_best} | t={_fmt_time(model_time_sum_sec)}",
                log_path,
            )

        row_dict = {
            "iter":               int(sub_id),
            "group":              str(binary_list),
            "group_size":         len(binary_list),
            "ok_integral":        "" if ok_integral is None else int(ok_integral),
            "res_usd":            float(res_usd),
            "model_time_sum_sec": float(model_time_sum_sec),
            "case":               int(case),
            "feasible":           label,
            "integrality":        integral_label,
            "exported":           int(exported),
            "accepted":           int(accepted),
            "obj_best":           int(obj_best),
            "obj_z":              "" if obj_z is None else float(obj_z),
            "exported_obj":       "" if cand_obj is None else int(cand_obj),
            "stop_early":         int(stop_early),
        }
        if log_group_features and group_feat_list[group_idx] is not None:
            row_dict.update(group_feat_list[group_idx])
        iter_rows.append(row_dict)

        sub_id += 1

    # ── Final full-scenario feasibility/objective check ──
    # Per-group reduction solved on K representative scenarios; verify the
    # resulting schedule on all 10 scenarios and re-certify feasibility.
    if enable_per_group_scenario_reduction:
        data         = data_orig
        number_scen  = number_scen_orig
        d_min_bounds = d_min_bounds_orig
        d_max_bounds = d_max_bounds_orig

        S_check: Dict[int, int] = {}
        for i in range(n):
            es_i = int(es.get(i, 0))
            ls_i = max(es_i, int(ls.get(i, es_i)))
            src  = S_best.get(i, fixed_start.get(i, es_i))
            S_check[i] = _clamp(int(src), es_i, ls_i)

        if ao is not None:
            try: ao.model.dispose()
            except Exception: pass
        ao = buildModel(
            data=data, number_scen=number_scen, epsilon=epsilon_dec,
            threads=4, mip_gap=1e-5, time_limit_seconds=None,
        )
        ao.model.Params.StartNodeLimit = 0

        reset_bounds_for_subproblem_Mz(ao=ao, data=data)
        set_bounds_for_group_subproblem(
            ao=ao, data=data, group=[],
            windows={}, fixed_start_times=S_check,
        )
        _tighten_y_bounds_for_fixed_activities(
            ao, S_check, d_min_bounds, d_max_bounds
        )
        # Skip warmstart on the final pass too — all xB are pinned by
        # bounds, so the warmstart's only contribution would trigger
        # the user-MIP-start LP completion overhead at j60 scale.
        ao.model.NumStart = 0

        elapsed   = time.time() - start_wall
        final_tl  = max(30.0, min(float(time_limit_model) * 2.0,
                                  float(time_limit_sec) - elapsed))
        ao.model.Params.TimeLimit = float(final_tl)
        ao.model.optimize()

        model_time_sum_sec += float(ao.model.Runtime)
        has_solution = ao.model.SolCount > 0
        if has_solution and _check_integrality(ao, data):
            found_integral = True
            obj_full = makespan_from_S(S_check, n)
            S_best   = dict(S_check)
            obj_best = int(obj_full)
            if verbose:
                _log(f"[GO-FR] FINAL FULL-SCEN: feas obj={obj_full}", log_path)
        else:
            if verbose:
                _log(f"[GO-FR] FINAL FULL-SCEN: schedule not certified on full data",
                     log_path)

    # ── Expand schedule back to physical time ──
    if time_alpha > 1:
        from time_aggregation import expand_schedule
        S_best   = expand_schedule(S_best,   time_alpha)
        # Recompute makespan in physical units
        obj_best = makespan_from_S(S_best, n)

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