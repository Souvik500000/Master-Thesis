from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import math


# ----------------------------
# Data containers
# ----------------------------

@dataclass
class ScheduleResult:
    S: Dict[int, int]
    D: Dict[int, int]
    profiles: Dict[int, List[List[int]]]


# ----------------------------
# Printing helpers
# ----------------------------

def print_schedule_vector(S: Dict[int, int], n_jobs: int) -> None:
    """
    Prints S = (S_0,S_1,...,S_{n-1})
    """
    vec = [S.get(i, None) for i in range(n_jobs)]
    print("\nSSGS start solution")
    print("S = (" + ",".join("NA" if v is None else str(v) for v in vec) + ")")


def print_schedule_details(res: ScheduleResult, max_jobs: int = 200) -> None:
    n = min(max_jobs, len(res.S))
    print("\nSSGS detailed schedule")
    for i in range(n):
        print(f"job {i}: S={res.S.get(i)}, D={res.D.get(i)}")


# ----------------------------
# Capacity helpers
# ----------------------------

def _get_capacities(data) -> List[int]:
    K = int(getattr(data, "n_renewable_resources"))

    if hasattr(data, "resource_capacities"):
        rc = getattr(data, "resource_capacities")
        if isinstance(rc, dict):
            return [int(rc[k]) for k in range(1, K + 1)]
        if isinstance(rc, (list, tuple)):
            return [int(x) for x in rc[:K]]

    if hasattr(data, "capacities_R"):
        rc = getattr(data, "capacities_R")
        if isinstance(rc, (list, tuple)):
            return [int(x) for x in rc[:K]]

    raise ValueError("No renewable resource capacities found in data")


def _ensure_remaining_horizon(remaining: List[List[int]], T_needed: int, caps: List[int]) -> None:
    K = len(caps)
    while len(remaining) < T_needed:
        remaining.append([caps[k] for k in range(K)])


# ----------------------------
# Resource profile construction
# ----------------------------

def minimal_duration_feasible(wk: int, lk: int, uk: int) -> int:
    if wk == 0:
        return 0
    if uk <= 0:
        raise ValueError("uk must be > 0 when wk > 0")
    if lk < 0:
        raise ValueError("lk must be >= 0")
    if lk > uk:
        raise ValueError("lk must be <= uk")

    D_min = math.ceil(wk / uk)
    if lk == 0:
        return D_min

    D_max = wk // lk
    if D_min > D_max:
        raise ValueError(f"Infeasible workload/resource bounds: wk={wk}, lk={lk}, uk={uk}")

    return D_min


def build_profile_for_resource(wk: int, lk: int, uk: int, D: int) -> List[int]:
    if D == 0:
        return []
    if wk == 0:
        return [0] * D

    r = [lk] * D
    remaining = wk - D * lk

    for t in range(D):
        add = min(uk - lk, remaining)
        r[t] += add
        remaining -= add
        if remaining == 0:
            break

    return r


def build_activity_profile(
    w_vec: List[int],
    l_vec: List[int],
    u_vec: List[int],
) -> Tuple[int, List[List[int]]]:
    K = len(w_vec)
    D_low = 0
    D_high: Optional[int] = None

    for k in range(K):
        wk = w_vec[k]
        lk = l_vec[k]
        uk = u_vec[k]

        D_k_min = minimal_duration_feasible(wk, lk, uk)
        D_low = max(D_low, D_k_min)

        if wk > 0 and lk > 0:
            D_k_max = wk // lk
            D_high = D_k_max if D_high is None else min(D_high, D_k_max)

    if D_high is not None and D_low > D_high:
        raise ValueError(
            f"No common feasible duration D across resources: D_low={D_low}, D_high={D_high}"
        )

    D = D_low

    per_k = [build_profile_for_resource(w_vec[k], l_vec[k], u_vec[k], D) for k in range(K)]
    profile = [[per_k[k][dt] for k in range(K)] for dt in range(D)]

    return D, profile


# ----------------------------
# Scenario selection
# ----------------------------

def select_workload_worst_case_for_activity(
    job: int,
    scenarios: Dict[int, Dict[int, List[int]]],
    n_R: int,
    scenario_ids: List[int],
    uR: Optional[Dict[int, List[int]]] = None,
) -> List[int]:
    """Pick the scenario that results in the longest duration for job j.

    For each scenario, compute the implied duration per resource:
        D_k = ceil(w_k / u_k)
    and take the max across resources as the scenario's "hardness".
    The scenario with the highest implied duration is selected.
    Falls back to summing workloads if uR is not provided.
    """
    best_s = None
    best_val = -10**18

    for s in scenario_ids:
        sk = scenarios.get(s, {})
        w_vec = [
            sk.get(k, [0])[job] if job < len(sk.get(k, [])) else 0
            for k in range(1, n_R + 1)
        ]
        if uR is not None:
            uk_list = uR.get(job, [])
            val = 0
            for k_idx, wk in enumerate(w_vec):
                uk = uk_list[k_idx] if k_idx < len(uk_list) else 1
                if uk > 0 and wk > 0:
                    val = max(val, math.ceil(wk / uk))
        else:
            val = sum(w_vec)

        if val > best_val:
            best_val = val
            best_s = s

    if best_s is None:
        return [0] * n_R

    sk = scenarios.get(best_s, {})
    return [
        sk.get(k, [0])[job] if job < len(sk.get(k, [])) else 0
        for k in range(1, n_R + 1)
    ]


# ----------------------------
# EST SSGS
# ----------------------------

def ssgs_est_worst_case(
    data,
    scenario_keep: List[int],
    allow_infeasible_fallback: bool = True,
) -> ScheduleResult:
    n = int(getattr(data, "n_jobs_including_dummy"))
    K = int(getattr(data, "n_renewable_resources"))
    caps = _get_capacities(data)

    predecessors = data.predecessors
    lR = data.lR
    uR = data.uR
    es = data.es
    ls = data.ls

    remaining: List[List[int]] = []
    S: Dict[int, int] = {0: 0}
    D: Dict[int, int] = {0: 0}
    profiles: Dict[int, List[List[int]]] = {0: []}
    scheduled = {0}

    def finish(i: int) -> int:
        return S[i] + D[i]

    def est(j: int) -> int:
        preds = predecessors.get(j, [])
        est_pred = max((finish(p) for p in preds), default=0)
        return max(est_pred, es.get(j, 0))

    def eligible() -> List[int]:
        return [
            j for j in range(n)
            if j not in scheduled
            and all(p in scheduled for p in predecessors.get(j, []))
        ]

    def can_place(t0: int, prof: List[List[int]]) -> bool:
        _ensure_remaining_horizon(remaining, t0 + len(prof), caps)
        for dt, row in enumerate(prof):
            for k in range(K):
                if row[k] > remaining[t0 + dt][k]:
                    return False
        return True

    def place(j: int, t0: int, prof: List[List[int]]) -> None:
        _ensure_remaining_horizon(remaining, t0 + len(prof), caps)
        for dt, row in enumerate(prof):
            for k in range(K):
                remaining[t0 + dt][k] -= row[k]
        S[j] = t0
        D[j] = len(prof)
        profiles[j] = prof
        scheduled.add(j)

    while len(scheduled) < n:
        E = eligible()
        if not E:
            raise RuntimeError("No eligible job found")

        j = min(E, key=lambda x: (est(x), x))
        t0 = est(j)
        latest = ls.get(j, 10**9)

        w_vec = select_workload_worst_case_for_activity(
            j, data.scenarios, K, scenario_keep, uR=uR
        )

        try:
            D_j, prof = build_activity_profile(w_vec, lR[j], uR[j])
            t = t0
            while t <= latest:
                if can_place(t, prof):
                    place(j, t, prof)
                    break
                t += 1
            else:
                raise RuntimeError
        except Exception:
            if not allow_infeasible_fallback:
                raise
            S[j] = t0
            D[j] = 0
            profiles[j] = []
            scheduled.add(j)

    return ScheduleResult(S=S, D=D, profiles=profiles)
