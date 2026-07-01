"""
feature_extractor.py
---------------------
Builds a comprehensive candidate feature matrix for activity grouping
in FSRCPSP instances. All plausible features are computed here; the
downstream ablation experiment selects the optimal subset.

Feature groups and counts (K = number of renewable resources):
  ─────────────────────────────────────────────────────────────────────
  GROUP A — NETWORK BASIC       (4) : precedence_level, in_degree,
                                      out_degree, longest_path_to_sink
  GROUP B — NETWORK EXTENDED    (3) : n_transitive_succ, n_transitive_pred,
                                      betweenness_centrality
  GROUP C — DURATION            (3) : duration, rel_duration, duration_rank
  GROUP D — URGENCY             (3) : slack, pressure_index, n_concurrent
  GROUP E — TIMING              (4) : rel_es, rel_ls, critical_path,
                                      float_ratio
  GROUP F — RESOURCE JOINT      (1) : resource_pressure
  GROUP G — PER-RESOURCE STATS  (8) : mean_Wk, cv_Wk, iqr_Wk,
                                      tail_ratio_Wk, max_Wk, min_Wk,
                                      range_Wk, p_nonzero_Wk
  ─────────────────────────────────────────────────────────────────────
  Total features (K=1): 18 + 8×1  = 26
  Total features (K=2): 18 + 8×2  = 34
  Total features (K=4): 18 + 8×4  = 50
  (all-zero and constant columns dropped automatically)

Ablation-validated optimal subset (DEFAULT_FEATURE_COLS):
  ["precedence_level", "in_degree", "out_degree", "longest_path_to_sink"]
  — Adding any other feature group increases Ward clustering gap.
  — See Ergebnisse/Ablation/ for full ablation results.

Usage:
    from Instance_Reader import read_instance
    from feature_extractor import EnrichedFeatureExtractor

    data = read_instance("instance.txt", number_scen=10, k_override=4)
    df_raw, df_scaled = EnrichedFeatureExtractor(data).extract()
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from collections import deque
from typing import Dict, List, Tuple
from sklearn.preprocessing import StandardScaler

from Instance_Reader import InstanceData, read_instance


# =============================================================================
# GROUP A — NETWORK BASIC UTILITIES
# =============================================================================

def _precedence_levels(data: InstanceData) -> Tuple[Dict[int, int], List[int]]:
    """
    Longest-path DP from source in topological order.

    level[i] = length of longest chain from source to activity i.
    Activities at the same level share no precedence relationship —
    safe to group and fix simultaneously in the MIP.

    Returns (levels, topological_order).
    """
    n     = data.n_jobs_including_dummy
    level = {j: 0 for j in range(n)}

    in_deg = {j: 0 for j in range(n)}
    for j in range(n):
        for s in data.successors.get(j, []):
            in_deg[s] = in_deg.get(s, 0) + 1

    queue = deque([j for j in range(n) if in_deg[j] == 0])
    topo: List[int] = []

    while queue:
        node = queue.popleft()
        topo.append(node)
        for s in data.successors.get(node, []):
            in_deg[s] -= 1
            if in_deg[s] == 0:
                queue.append(s)

    for j in topo:
        for s in data.successors.get(j, []):
            if level[j] + 1 > level[s]:
                level[s] = level[j] + 1

    return level, topo


def _longest_paths_to_sink(data: InstanceData, topo: List[int]) -> Dict[int, int]:
    """
    Backward longest-path DP to sink.
    depth[i] = length of longest chain from i to sink.
    """
    depth = {j: 0 for j in range(data.n_jobs_including_dummy)}
    for j in reversed(topo):
        for s in data.successors.get(j, []):
            if depth[j] < depth[s] + 1:
                depth[j] = depth[s] + 1
    return depth


# =============================================================================
# GROUP B — NETWORK EXTENDED UTILITIES
# =============================================================================

def _transitive_counts(
    data: InstanceData, topo: List[int]
) -> Tuple[Dict[int, int], Dict[int, int]]:
    """
    Count transitive successors and predecessors for each activity.

    n_transitive_succ[i] = |{j : i can reach j via precedences}|
    n_transitive_pred[i] = |{j : j can reach i via precedences}|

    Computed with forward and backward reachability passes over the
    topological order. Uses integer counting, not bitsets — sufficient
    for n ≤ 120.
    """
    n = data.n_jobs_including_dummy

    # Forward pass: count all descendants
    # reach_fwd[i] = set of nodes reachable from i
    reach_fwd: Dict[int, set] = {j: set() for j in range(n)}
    for j in reversed(topo):
        for s in data.successors.get(j, []):
            reach_fwd[j].add(s)
            reach_fwd[j].update(reach_fwd[s])

    # Backward pass: count all ancestors
    # reach_bwd[i] = set of nodes that can reach i
    reach_bwd: Dict[int, set] = {j: set() for j in range(n)}
    for j in topo:
        for p in data.predecessors.get(j, []):
            reach_bwd[j].add(p)
            reach_bwd[j].update(reach_bwd[p])

    n_succ = {j: len(reach_fwd[j]) for j in range(n)}
    n_pred = {j: len(reach_bwd[j]) for j in range(n)}
    return n_succ, n_pred


def _betweenness_centrality(
    data: InstanceData, topo: List[int]
) -> Dict[int, float]:
    """
    DAG betweenness centrality for each activity.

    bc(v) = (paths from source through v to sink) / (total source-to-sink paths)

    Computed efficiently with two DP passes:
      1. Forward: count paths from source to each node.
      2. Backward: count paths from each node to sink.
      bc(v) = forward[v] * backward[v] / forward[sink]

    bc = 1.0 means ALL paths must pass through v (a mandatory bottleneck).
    bc = 0.0 means v is never on any source-to-sink path (isolated branch).

    This is a structural importance measure: high-betweenness activities
    are scheduling bottlenecks whose delays propagate through the project.
    """
    n    = data.n_jobs_including_dummy
    sink = n - 1

    # Forward: n_paths_from_source[v]
    fwd: Dict[int, int] = {j: 0 for j in range(n)}
    fwd[0] = 1  # source has exactly 1 path to itself
    for j in topo:
        for s in data.successors.get(j, []):
            fwd[s] += fwd[j]

    total_paths = fwd[sink]
    if total_paths == 0:
        return {j: 0.0 for j in range(n)}

    # Backward: n_paths_to_sink[v]
    bwd: Dict[int, int] = {j: 0 for j in range(n)}
    bwd[sink] = 1
    for j in reversed(topo):
        for s in data.successors.get(j, []):
            bwd[j] += bwd[s]

    return {j: float(fwd[j] * bwd[j]) / float(total_paths) for j in range(n)}


# =============================================================================
# GROUP G — PER-RESOURCE STATISTICS
# =============================================================================

def _scenario_arrays(data: InstanceData) -> Dict[int, Dict[int, np.ndarray]]:
    """Build {resource_k: {job_i: array of workloads across all scenarios}}."""
    n      = data.n_jobs_including_dummy
    n_R    = data.n_renewable_resources
    s_keys = sorted(data.scenarios.keys())

    out: Dict[int, Dict[int, np.ndarray]] = {}
    for k in range(1, n_R + 1):
        out[k] = {}
        for job in range(n):
            vals = []
            for s in s_keys:
                row = data.scenarios[s].get(k, [])
                vals.append(float(row[job]) if job < len(row) else 0.0)
            out[k][job] = np.array(vals, dtype=float)
    return out


def _resource_stats(arr: np.ndarray) -> Dict[str, float]:
    """
    Compute all 8 stochastic statistics for one activity on one resource.

    mean       = expected workload across scenarios
    cv         = std / mean  (relative variability; 0 if deterministic)
    iqr        = Q75 - Q25   (robust spread; correlated with mean for S=10)
    tail_ratio = p90 / mean  (upper-tail weight; ≈cv for S=10)
    max        = worst-case scenario demand
    min        = best-case scenario demand
    range      = max - min   (total demand spread)
    p_nonzero  = fraction of scenarios with positive demand
    """
    if len(arr) == 0 or np.all(arr == 0):
        return {
            "mean": 0.0, "cv": 0.0, "iqr": 0.0, "tail_ratio": 0.0,
            "max":  0.0, "min": 0.0, "range": 0.0, "p_nonzero": 0.0,
        }

    mean = float(np.mean(arr))
    std  = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    cv   = std / mean if mean > 0 else 0.0

    p25  = float(np.percentile(arr, 25))
    p75  = float(np.percentile(arr, 75))
    p90  = float(np.percentile(arr, 90))
    iqr  = p75 - p25
    tail_ratio = p90 / mean if mean > 0 else 0.0

    arr_max   = float(np.max(arr))
    arr_min   = float(np.min(arr))
    arr_range = arr_max - arr_min
    p_nonzero = float(np.mean(arr > 0))

    return {
        "mean":       mean,
        "cv":         cv,
        "iqr":        iqr,
        "tail_ratio": tail_ratio,
        "max":        arr_max,
        "min":        arr_min,
        "range":      arr_range,
        "p_nonzero":  p_nonzero,
    }


# =============================================================================
# FEATURE EXTRACTOR
# =============================================================================

class EnrichedFeatureExtractor:
    """
    Builds the comprehensive candidate feature matrix for FSRCPSP grouping.

    Computes 50 features for K=4 (18 structural + 8*K stochastic).
    All features are returned; selection is performed downstream by ablation.

    Parameters
    ----------
    data            : InstanceData from Instance_Reader.read_instance()
    include_dummies : if True, include source/sink dummy activities (default False)
    """

    def __init__(self, data: InstanceData, include_dummies: bool = False):
        self.data            = data
        self.include_dummies = include_dummies

    def extract(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Compute all candidate features for all real activities.

        Returns
        -------
        df_raw    : features in original units
        df_scaled : z-score normalised  →  use this for clustering
        """
        data = self.data
        n    = data.n_jobs_including_dummy
        sink = n - 1
        n_R  = data.n_renewable_resources

        # ── Precompute structural quantities ─────────────────────────
        levels, topo = _precedence_levels(data)
        depths       = _longest_paths_to_sink(data, topo)
        n_succ, n_pred = _transitive_counts(data, topo)
        bc           = _betweenness_centrality(data, topo)
        scen_w       = _scenario_arrays(data)

        # ── Group C: Duration scaling ─────────────────────────────────
        all_durs = [data.duration.get(j, 0) for j in range(n)
                    if j not in (0, sink)]
        max_dur = max(all_durs) if all_durs else 1
        dur_sorted = sorted(all_durs)
        dur_rank_map = {d: (dur_sorted.index(d) + 1) / len(dur_sorted)
                        for d in set(dur_sorted)}

        # ── Group E: Timing scaling ───────────────────────────────────
        T_max = max(data.ls.values()) if data.ls else 1

        # ── Group D: n_concurrent — tight earliest-start windows ─────
        windows: Dict[int, tuple] = {}
        for j in range(n):
            es_j  = data.es.get(j, 0)
            dur_j = data.duration.get(j, 0)
            windows[j] = (es_j, es_j + dur_j)

        rows: List[Dict] = []

        for job in range(n):
            if not self.include_dummies and job in (0, sink):
                continue

            dur  = int(data.duration.get(job, 0))
            es_i = int(data.es.get(job, 0))
            ls_i = int(data.ls.get(job, es_i))

            # ── GROUP A: Network Basic ────────────────────────────────
            precedence_level     = float(levels[job])
            in_degree            = float(len(data.predecessors.get(job, [])))
            out_degree           = float(len(data.successors.get(job, [])))
            longest_path_to_sink = float(depths[job])

            # ── GROUP B: Network Extended ─────────────────────────────
            n_transitive_succ    = float(n_succ[job])
            n_transitive_pred    = float(n_pred[job])
            betweenness          = float(bc[job])

            # ── GROUP C: Duration ─────────────────────────────────────
            duration      = float(dur)
            rel_duration  = float(dur) / float(max_dur) if max_dur > 0 else 0.0
            duration_rank = float(dur_rank_map.get(dur, 0.0))

            # ── GROUP D: Urgency ──────────────────────────────────────
            slack          = float(max(0, ls_i - es_i))
            window         = max(1, ls_i - es_i + 1)
            pressure_index = float(dur) / float(window)

            win_start = es_i
            win_end   = es_i + dur
            n_concurrent = float(sum(
                1 for j2, (s2, e2) in windows.items()
                if j2 != job and j2 not in (0, sink)
                and s2 < win_end and e2 > win_start
            ))

            # ── GROUP E: Timing ───────────────────────────────────────
            rel_es        = float(es_i) / float(T_max) if T_max > 0 else 0.0
            rel_ls        = float(ls_i) / float(T_max) if T_max > 0 else 0.0
            critical_path = 1.0 if slack == 0.0 else 0.0
            float_ratio   = slack / float(T_max) if T_max > 0 else 0.0

            # ── GROUP F: Resource Joint ───────────────────────────────
            caps = data.resource_capacities
            resource_pressure = float(sum(
                float(np.mean(scen_w.get(k, {}).get(job, np.array([0.0])))) /
                float(caps.get(k, 1))
                for k in range(1, n_R + 1)
            ))

            # ── Assemble row ──────────────────────────────────────────
            row: Dict = {
                "job": job,
                # Group A
                "precedence_level":     precedence_level,
                "in_degree":            in_degree,
                "out_degree":           out_degree,
                "longest_path_to_sink": longest_path_to_sink,
                # Group B
                "n_transitive_succ":    n_transitive_succ,
                "n_transitive_pred":    n_transitive_pred,
                "betweenness_centrality": betweenness,
                # Group C
                "duration":             duration,
                "rel_duration":         rel_duration,
                "duration_rank":        duration_rank,
                # Group D
                "slack":                slack,
                "pressure_index":       pressure_index,
                "n_concurrent":         n_concurrent,
                # Group E
                "rel_es":               rel_es,
                "rel_ls":               rel_ls,
                "critical_path":        critical_path,
                "float_ratio":          float_ratio,
                # Group F
                "resource_pressure":    resource_pressure,
            }

            # ── GROUP G: Per-resource statistics (8 per resource) ────
            for k in range(1, n_R + 1):
                arr   = scen_w.get(k, {}).get(job, np.array([]))
                stats = _resource_stats(arr)
                row[f"mean_W{k}"]       = stats["mean"]
                row[f"cv_W{k}"]         = stats["cv"]
                row[f"iqr_W{k}"]        = stats["iqr"]
                row[f"tail_ratio_W{k}"] = stats["tail_ratio"]
                row[f"max_W{k}"]        = stats["max"]
                row[f"min_W{k}"]        = stats["min"]
                row[f"range_W{k}"]      = stats["range"]
                row[f"p_nonzero_W{k}"]  = stats["p_nonzero"]

            rows.append(row)

        # ── Build DataFrame ───────────────────────────────────────────
        df_raw = pd.DataFrame(rows).set_index("job")

        # Drop all-zero columns (e.g. W2/W3/W4 stats for k=1 instances)
        df_raw = df_raw.loc[:, (df_raw != 0).any(axis=0)]

        # Drop constant columns (zero variance → no clustering information)
        df_raw = df_raw.loc[:, df_raw.nunique() > 1]

        # Z-score normalisation: x̃ = (x - μ) / σ
        scaler    = StandardScaler()
        scaled    = scaler.fit_transform(df_raw.values)
        df_scaled = pd.DataFrame(
            scaled, index=df_raw.index, columns=df_raw.columns
        )

        return df_raw, df_scaled

    def feature_summary(self) -> None:
        """Print a summary of all feature groups and counts."""
        n_R = self.data.n_renewable_resources
        df_raw, _ = self.extract()
        n_features = df_raw.shape[1]

        print("\n" + "=" * 70)
        print("  EnrichedFeatureExtractor — Full Candidate Feature Set")
        print("=" * 70)
        groups = [
            ("A", "Network Basic",      4,   "precedence_level, in_degree, out_degree, longest_path_to_sink"),
            ("B", "Network Extended",   3,   "n_transitive_succ, n_transitive_pred, betweenness_centrality"),
            ("C", "Duration",           3,   "duration, rel_duration, duration_rank"),
            ("D", "Urgency",            3,   "slack, pressure_index, n_concurrent"),
            ("E", "Timing",             4,   "rel_es, rel_ls, critical_path, float_ratio"),
            ("F", "Resource Joint",     1,   "resource_pressure"),
            ("G", f"Per-Resource x{n_R}", 8*n_R, f"mean/cv/iqr/tail_ratio/max/min/range/p_nonzero × {n_R} resources"),
        ]
        total = 0
        for grp, name, cnt, features in groups:
            print(f"  [{grp}] {name:<22} ({cnt:>2} features)  {features}")
            total += cnt
        print(f"\n  Total candidate features (K={n_R}): {total}")
        print(f"  After auto-drop (zeros/constants): {n_features}")
        print(f"\n  Ablation-validated clustering subset: net4 (Group A only)")
        print("=" * 70 + "\n")
        return df_raw.columns.tolist()


# =============================================================================
# FEATURE GROUP DEFINITIONS  (import these in ablation experiments)
# =============================================================================

def get_feature_groups(K: int) -> Dict[str, List[str]]:
    """
    Return all named feature groups for a given K.
    Use these to define ablation configs without hardcoding column names.
    """
    return {
        "net_basic":   ["precedence_level", "in_degree", "out_degree",
                        "longest_path_to_sink"],
        "net_ext":     ["n_transitive_succ", "n_transitive_pred",
                        "betweenness_centrality"],
        "duration":    ["duration", "rel_duration", "duration_rank"],
        "urgency":     ["slack", "pressure_index", "n_concurrent"],
        "timing":      ["rel_es", "rel_ls", "critical_path", "float_ratio"],
        "res_joint":   ["resource_pressure"],
        "res_mean":    [f"mean_W{k}"       for k in range(1, K + 1)],
        "res_cv":      [f"cv_W{k}"         for k in range(1, K + 1)],
        "res_iqr":     [f"iqr_W{k}"        for k in range(1, K + 1)],
        "res_tail":    [f"tail_ratio_W{k}" for k in range(1, K + 1)],
        "res_max":     [f"max_W{k}"        for k in range(1, K + 1)],
        "res_min":     [f"min_W{k}"        for k in range(1, K + 1)],
        "res_range":   [f"range_W{k}"      for k in range(1, K + 1)],
        "res_pnz":     [f"p_nonzero_W{k}"  for k in range(1, K + 1)],
    }


# =============================================================================
# MAIN — demo / sanity check
# =============================================================================

if __name__ == "__main__":
    import os, glob

    FILEPATH    = "/Users/souvikchakraborty/Downloads/AO_RF_Gurobi-5/Test_instances/j301_1/j301_1.txt"
    NUMBER_SCEN = 10
    K_OVERRIDE  = 4

    data = read_instance(FILEPATH, number_scen=NUMBER_SCEN,
                         k_override=K_OVERRIDE)

    extractor = EnrichedFeatureExtractor(data, include_dummies=False)
    extractor.feature_summary()

    df_raw, df_scaled = extractor.extract()

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", "{:.3f}".format)

    print(f"\n{'=' * 70}")
    print(f"  Full Candidate Feature Matrix: {len(df_raw)} activities × {df_raw.shape[1]} features")
    print(f"{'=' * 70}")
    print(df_raw.to_string())

    # Correlation check
    print(f"\n{'=' * 70}")
    print("  High-correlation pairs (|r| >= 0.85) — candidates for pruning")
    print(f"{'=' * 70}")
    corr = df_raw.corr().round(2)
    cols = df_raw.columns.tolist()
    high_corr = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = abs(float(corr.iloc[i, j]))
            if r >= 0.85:
                high_corr.append((cols[i], cols[j], round(r, 2)))
    if high_corr:
        print(f"  {len(high_corr)} pairs found:")
        for a, b, r in sorted(high_corr, key=lambda x: -x[2]):
            print(f"    {a:<30} vs {b:<30}  |r| = {r}")
    else:
        print("  No pairs with |r| >= 0.85 found.")

    groups = get_feature_groups(K_OVERRIDE)
    print(f"\n  Feature groups available for ablation:")
    for name, cols_list in groups.items():
        print(f"    {name:<14} ({len(cols_list):>2}): {cols_list}")
