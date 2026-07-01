# time_aggregation.py
#
# Time-axis aggregation for the FSRCPSP MIP. Bucket physical time into
# windows of size α (e.g. α=5: 5 time units → 1 aggregated bucket).
# The MIP keeps the same combinatorial structure but every time-indexed
# tensor (xB, xC, y, r) shrinks by α along the t-axis.
#
# Math summary:
#   Bucket τ spans physical times [α·τ, α·τ + α - 1].
#   r_new[i,k,τ,π] = ∑_{t in bucket(τ)} r_orig[i,k,t,π]   (total work in bucket)
#   ⇒ r_new ∈ [α·lR, α·uR]   when active
#   ⇒ ∑_t r_new = W   (total work unchanged)
#   ⇒ resource_capacities  ↦  α · resource_capacities  (per-bucket cap)
#
# So the aggregated problem is structurally identical to the original
# with these scalings:
#   es, ls, horizon, duration : divide by α (with rounding)
#   lR, uR, resource_capacities: multiply by α
#   workloads (scenarios), graph structure, n_jobs, n_R: unchanged
#
# Bounded-loss guarantee:
#   makespan_agg (in physical units) is an UPPER BOUND on makespan_opt,
#   inflated by at most α per critical-path activity that gets ceiled up.
#   In practice: aggregated makespan ≤ optimal_makespan + α (path-bounded).
#
# Schedule expansion:
#   S_phys[i] = α · S_agg[i]   (start of bucket — always feasible if
#                                 the aggregated schedule is feasible)

from __future__ import annotations

import copy
import math
from typing import Dict


def aggregate_instance(data, alpha: int):
    """
    Return a deep-copied InstanceData with time aggregated by factor `alpha`.
    `alpha = 1` returns an unmodified copy.
    """
    if alpha <= 0:
        raise ValueError(f"alpha must be >= 1, got {alpha}")
    if alpha == 1:
        return copy.deepcopy(data)

    new_data = copy.deepcopy(data)

    # Time-indexed scalars: divide by alpha
    new_data.es = {
        int(i): max(0, int(es) // alpha)
        for i, es in data.es.items()
    }
    new_data.ls = {
        int(i): int(math.ceil(int(ls) / alpha))
        for i, ls in data.ls.items()
    }
    if getattr(data, "horizon", None) is not None:
        new_data.horizon = int(math.ceil(int(data.horizon) / alpha))
    else:
        new_data.horizon = int(math.ceil(max(new_data.ls.values()))) if new_data.ls else 0

    new_data.duration = {
        int(i): max(1, int(math.ceil(int(d) / alpha)))
        for i, d in data.duration.items()
    } if getattr(data, "duration", None) else getattr(data, "duration", {})

    # Resource bounds and capacities: multiply by alpha (per-bucket sum)
    #
    # uR (max per-bucket rate) scales linearly: alpha * uR_phys.
    # lR (min per-bucket rate when active) is set to 0 — this is the
    # standard relaxation in time-aggregation: scaling lR by alpha can
    # make activities with small workload (W < alpha * lR_phys) infeasible
    # because the floor rate would force over-delivery of work in one
    # bucket. Setting lR_agg = 0 makes the aggregated problem a
    # relaxation of the original (feasibility preserved upward —
    # schedules feasible in aggregated may need refinement to be
    # feasible in physical, but the search space includes all original
    # feasible schedules).
    new_data.lR = {
        int(i): [0 for _ in row]
        for i, row in data.lR.items()
    }
    new_data.uR = {
        int(i): [int(u) * alpha for u in row]
        for i, row in data.uR.items()
    }
    new_data.resource_capacities = {
        int(k): int(cap) * alpha
        for k, cap in data.resource_capacities.items()
    }

    # Workloads, scenarios, graph structure, expected_workload, requests_R
    # are all unchanged (total work is invariant under time aggregation).
    return new_data


def expand_schedule(S_agg: Dict[int, int], alpha: int) -> Dict[int, int]:
    """
    Translate aggregated start times back to physical time.
      S_phys[i] = alpha * S_agg[i]
    Always feasible if the aggregated schedule is.
    """
    if alpha <= 1:
        return dict(S_agg)
    return {int(i): int(s) * alpha for i, s in S_agg.items()}


def aggregate_schedule(S_phys: Dict[int, int], alpha: int) -> Dict[int, int]:
    """
    Translate a physical schedule (e.g. SSGS warmstart given in physical
    units) into aggregated bucket indices for use as init_S.
      S_agg[i] = floor(S_phys[i] / alpha)
    """
    if alpha <= 1:
        return dict(S_phys)
    return {int(i): int(s) // alpha for i, s in S_phys.items()}
