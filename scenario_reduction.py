# scenario_reduction.py
#
# Reduce the number of stochastic scenarios in an InstanceData to a
# smaller representative set. The per-group MIP scales linearly with
# the number of scenarios, so going from 10 -> 3 roughly triples the
# speed and makes previously intractable j120 instances feasible.
#
# Methods provided:
#   reduce_naive             : keep the first n_keep scenarios (trivial baseline)
#   reduce_random            : sample n_keep at random (seeded baseline)
#   reduce_kmedoids          : true PAM K-medoids on flattened workload vectors
#   reduce_backward          : Heitsch-Römisch (2003) backward deletion —
#                              greedily removes scenarios in order of minimum
#                              redistribution cost, minimising the Kantorovich
#                              distance to the original distribution
#   reduce_kmedoids_for_subset: PAM K-medoids restricted to one activity group
#                              (used inside the F&R loop for per-group reduction)
#
# Why PAM over K-means:
#   K-means centroids are synthetic averages — not valid scenario data.
#   PAM guarantees cluster centres are REAL scenarios, which is required
#   because the downstream MIP uses raw workload values from scenarios.
#
# Why Heitsch-Römisch:
#   Directly minimises the Kantorovich (earth-mover) distance between the
#   original S-scenario distribution and the reduced n_keep-scenario
#   distribution. Provides theoretical solution-quality bounds.
#   Reference: Heitsch & Römisch, "Scenario Reduction Algorithms in
#              Stochastic Programming", Comput Optim Appl 24, 2003.
#
# All methods return a NEW InstanceData with scenarios renumbered 1..n_keep
# (the rest of the pipeline expects contiguous 1-indexed scenario IDs).

from __future__ import annotations

import copy
import math
from typing import List, Tuple

import numpy as np


# ============================================================================
# CORE: scenario-as-vector
# ============================================================================

def _scenario_vector(data, scen_id: int) -> np.ndarray:
    """
    Flatten data.scenarios[scen_id] into a 1-D vector of length n_R * n_jobs.
    Order: resource k=1..K, then activity i=0..n-1.
    """
    activities = list(range(int(data.n_jobs_including_dummy)))
    resources  = list(range(1, int(data.n_renewable_resources) + 1))
    scen = data.scenarios.get(scen_id, {})
    flat: List[int] = []
    for k in resources:
        arr = scen.get(k, [])
        for i in activities:
            flat.append(int(arr[i]) if i < len(arr) else 0)
    return np.array(flat, dtype=np.float64)


def _scenario_matrix(data) -> Tuple[np.ndarray, List[int]]:
    """
    Build (S × F) matrix of scenario vectors. Returns (matrix, scenario_ids).
    """
    scen_ids = sorted(int(s) for s in data.scenarios.keys())
    rows = [_scenario_vector(data, s) for s in scen_ids]
    return np.vstack(rows), scen_ids


def _renumber_scenarios(data, kept_ids: List[int]):
    """
    Return a deep copy of `data` whose .scenarios dict contains only the
    scenarios in `kept_ids`, renumbered 1..len(kept_ids).
    """
    new_data = copy.deepcopy(data)
    new_scenarios = {}
    for new_idx, orig_id in enumerate(kept_ids, start=1):
        new_scenarios[new_idx] = data.scenarios[orig_id]
    new_data.scenarios = new_scenarios
    return new_data


# ============================================================================
# REDUCERS
# ============================================================================

def reduce_naive(data, n_keep: int):
    """Keep the first n_keep scenarios (1, 2, ..., n_keep). Trivial baseline."""
    available = sorted(int(s) for s in data.scenarios.keys())
    kept      = available[:n_keep]
    return _renumber_scenarios(data, kept), kept


def reduce_random(data, n_keep: int, seed: int = 42):
    """Random subset of n_keep scenarios (seeded)."""
    available = sorted(int(s) for s in data.scenarios.keys())
    rng = np.random.RandomState(seed)
    if n_keep >= len(available):
        kept = available
    else:
        kept = sorted(rng.choice(available, size=n_keep, replace=False).tolist())
    return _renumber_scenarios(data, kept), kept


def _scenario_vector_subset(data, scen_id: int, activities: List[int]) -> np.ndarray:
    """
    Workload vector restricted to a subset of activities.
    Used for group-local scenario reduction: when picking representative
    scenarios for one group's subMIP, only the workload of *that group's*
    activities matters — activities outside the group are fixed and
    don't drive the per-subMIP MIP cost.
    """
    resources = list(range(1, int(data.n_renewable_resources) + 1))
    scen = data.scenarios.get(scen_id, {})
    flat: List[int] = []
    for k in resources:
        arr = scen.get(k, [])
        for i in activities:
            flat.append(int(arr[i]) if i < len(arr) else 0)
    return np.array(flat, dtype=np.float64)


# ============================================================================
# CORE: PAM K-medoids (true medoid clustering, no synthetic centroids)
# ============================================================================

def _pam_kmedoids(X: np.ndarray, n_clusters: int, seed: int = 42) -> np.ndarray:
    """
    Partitioning Around Medoids (PAM) — true K-medoids algorithm.

    Unlike K-means, PAM constrains cluster centres to be ACTUAL data points.
    This is required here because we must return real scenarios (not synthetic
    averages) for the downstream MIP.

    Algorithm (Kaufman & Rousseeuw, 1987):
      1. Initialise medoids by greedy farthest-point sampling.
      2. Repeat until convergence:
         a. Assign each point to its nearest medoid.
         b. For each cluster c and each non-medoid point o in c:
            compute cost of swapping medoid m_c with o.
            If total cost decreases, perform the swap.

    Complexity: O(S² · k · iter) — negligible for S ≤ 10.

    Returns
    -------
    medoid_indices : np.ndarray of shape (n_clusters,)
        Indices into X of the selected medoids.
    """
    rng = np.random.RandomState(seed)
    S   = X.shape[0]

    # Precompute full pairwise distance matrix (S×S)
    D = np.zeros((S, S))
    for i in range(S):
        for j in range(i + 1, S):
            d = float(np.linalg.norm(X[i] - X[j]))
            D[i, j] = D[j, i] = d

    # Initialise: pick first medoid as point minimising sum of distances,
    # then add each successive medoid to maximise coverage.
    medoids = [int(np.argmin(D.sum(axis=1)))]
    for _ in range(1, n_clusters):
        # Distance of each point to its nearest current medoid
        dist_to_nearest = np.min(D[:, medoids], axis=1)
        # Next medoid = farthest from current set
        medoids.append(int(np.argmax(dist_to_nearest)))

    # PAM swap phase
    for _ in range(100):  # max iterations
        labels  = np.argmin(D[:, medoids], axis=1)
        changed = False

        for c, m in enumerate(medoids):
            # Consider all non-medoid points in this cluster as replacements
            cluster_pts = np.where(labels == c)[0]
            for o in cluster_pts:
                if o == m:
                    continue
                # Compute total cost change if we swap m ↔ o
                new_medoids = medoids[:]
                new_medoids[c] = o
                new_cost = np.sum(np.min(D[:, new_medoids], axis=1))
                old_cost = np.sum(np.min(D[:, medoids],     axis=1))
                if new_cost < old_cost - 1e-9:
                    medoids[c] = o
                    changed    = True
                    # Recompute labels for next iteration
                    labels = np.argmin(D[:, medoids], axis=1)

        if not changed:
            break

    return np.array(medoids)


def reduce_kmedoids(data, n_keep: int, seed: int = 42):
    """
    True PAM K-medoids on flattened scenario workload vectors.

    Selects n_keep real scenarios that best represent the full S-scenario
    distribution. Cluster centres are guaranteed to be actual scenarios
    (not synthetic averages), so all downstream MIP constraints remain valid.
    """
    X, scen_ids = _scenario_matrix(data)
    n_scen      = X.shape[0]
    n_clusters  = max(1, min(n_keep, n_scen))

    if n_clusters >= n_scen:
        return _renumber_scenarios(data, scen_ids), scen_ids

    if np.allclose(X, X[0]):
        kept = sorted(scen_ids[:n_clusters])
        return _renumber_scenarios(data, kept), kept

    medoid_idxs = _pam_kmedoids(X, n_clusters, seed=seed)
    kept = sorted(scen_ids[i] for i in medoid_idxs)
    return _renumber_scenarios(data, kept), kept


# ============================================================================
# Heitsch-Römisch backward scenario reduction
# ============================================================================

def reduce_backward(data, n_keep: int, seed: int = 42):
    """
    Heitsch-Römisch (2003) fast backward scenario reduction.

    Greedily deletes scenarios one at a time in the order of minimum
    redistribution cost, which directly minimises the Kantorovich
    (earth-mover) distance between the original and reduced distributions.

    Algorithm:
      At each step, for every remaining scenario j, compute:
          cost(j) = p_j * min_{i ≠ j, i active} D(j, i)
      Delete the scenario j* with minimum cost(j*).
      Redistribute p_j* to the nearest remaining scenario.
      Repeat until n_keep scenarios remain.

    With uniform weights (p_j = 1/S), the retained set minimises:
        Σ_{j deleted} min_{i retained} D(j, i)

    This gives a theoretical bound: the Wasserstein distance between
    the original and reduced distributions is bounded by the total
    redistribution cost.

    Reference:
        Heitsch & Römisch, "Scenario Reduction Algorithms in Stochastic
        Programming", Comput Optim Appl 24(2–3), pp 187–206, 2003.
    """
    X, scen_ids = _scenario_matrix(data)
    n_scen      = X.shape[0]
    n_clusters  = max(1, min(n_keep, n_scen))

    if n_clusters >= n_scen:
        return _renumber_scenarios(data, scen_ids), scen_ids

    # Pairwise distances
    D = np.zeros((n_scen, n_scen))
    for i in range(n_scen):
        for j in range(i + 1, n_scen):
            d = float(np.linalg.norm(X[i] - X[j]))
            D[i, j] = D[j, i] = d

    weights = np.ones(n_scen) / n_scen   # uniform
    active  = list(range(n_scen))

    while len(active) > n_clusters:
        # Cost of deleting each active scenario
        best_cost = float("inf")
        best_j    = -1
        for j in active:
            others       = [i for i in active if i != j]
            nearest_dist = min(D[j, i] for i in others)
            cost         = weights[j] * nearest_dist
            if cost < best_cost:
                best_cost = cost
                best_j    = j

        # Redistribute weight to nearest remaining scenario
        others  = [i for i in active if i != best_j]
        nearest = min(others, key=lambda i: D[best_j, i])
        weights[nearest] += weights[best_j]
        active.remove(best_j)

    kept = sorted(scen_ids[i] for i in active)
    return _renumber_scenarios(data, kept), kept


# ============================================================================
# Per-group PAM K-medoids (used inside the F&R loop)
# ============================================================================

def reduce_kmedoids_for_subset(
    data,
    n_keep: int,
    activities: List[int],
    seed: int = 42,
):
    """
    True PAM K-medoids scenario reduction restricted to one activity group.

    When picking representative scenarios for one group's subMIP, only the
    workload of THAT GROUP's activities matters — activities outside the group
    are fixed and don't drive the per-subMIP cost.

    Used inside the F&R loop:
      - SSGS warm-start runs on the full S scenarios (correct initialisation)
      - per-group subMIPs run on n_keep representative scenarios (smaller MIP)
    """
    scen_ids   = sorted(int(s) for s in data.scenarios.keys())
    rows       = [_scenario_vector_subset(data, s, activities) for s in scen_ids]
    X          = np.vstack(rows)
    n_scen     = X.shape[0]
    n_clusters = max(1, min(n_keep, n_scen))

    if n_clusters >= n_scen:
        return _renumber_scenarios(data, scen_ids), scen_ids

    if np.allclose(X, X[0]):
        kept = sorted(scen_ids[:n_clusters])
        return _renumber_scenarios(data, kept), kept

    medoid_idxs = _pam_kmedoids(X, n_clusters, seed=seed)
    kept = sorted(scen_ids[i] for i in medoid_idxs)
    return _renumber_scenarios(data, kept), kept


# ============================================================================
# SMALL CLI / SELF-TEST
# ============================================================================

if __name__ == "__main__":
    import os, glob
    from Instance_Reader import read_instance

    candidates = sorted(glob.glob(
        "Instances_j30_test/K=4/**/*.txt", recursive=True
    ))
    p = candidates[0]
    print(f"Sanity test on: {os.path.basename(p)}")
    data = read_instance(p, number_scen=10, k_override=4)
    print(f"  Original: {len(data.scenarios)} scenarios")

    for fn, label in [(reduce_naive,    "naive    "),
                      (reduce_random,   "random   "),
                      (reduce_kmedoids, "kmedoids ")]:
        d2, kept = fn(data, n_keep=5)
        # quick sanity: every kept scenario produced same data per (k,i)
        ok = all(int(d2.scenarios[i + 1][k][a]) ==
                 int(data.scenarios[kept[i]][k][a])
                 for i in range(len(kept))
                 for k in d2.scenarios[i + 1]
                 for a in range(min(5, data.n_jobs_including_dummy)))
        print(f"  {label} kept={kept}  sanity={'OK' if ok else 'FAIL'}")
