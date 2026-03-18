"""
enriched_features.py
---------------------
Builds a lean, empirically validated feature matrix for activity
grouping in FSRCPSP instances.

Design philosophy:
  MINIMUM SUFFICIENT REPRESENTATION — exactly one feature per distinct
  aspect of activity behaviour, validated by correlation analysis.

Feature set:
  ─────────────────────────────────────────────────────────────────────
  NETWORK POSITION  (3) : precedence_level, in_degree, out_degree
  SCHEDULING URGENCY(2) : slack, pressure_index
  PER RESOURCE k    (2) : mean_Wk, cv_Wk
  ─────────────────────────────────────────────────────────────────────
  k=1 → 5 + 2×1 = 7  features
  k=2 → 5 + 2×2 = 9  features
  k=4 → 5 + 2×4 = 13 features
  (all-zero and constant columns dropped automatically)

Dropped features (based on correlation analysis on PSPLIB j30 instances):
  iqr_Wk       → |r| = 0.95–1.00 with mean_Wk  (IQR scales with mean)
  tail_ratio_Wk→ |r| = 0.98–1.00 with cv_Wk    (p90≈max with S=10)
  density_W1   → |r| = 0.86      with mean_W1  (density ∝ mean/d)

Each surviving feature answers a distinct question:
  precedence_level → WHERE in the network?
  in_degree        → HOW MANY predecessors?
  out_degree       → HOW MANY successors unlocked?
  slack            → HOW MUCH scheduling freedom?
  pressure_index   → HOW TIGHT is the window?
  mean_Wk          → HOW MUCH of resource k consumed?
  cv_Wk            → HOW VARIABLE is the demand?

Usage:
    from Instance_Reader import read_instance
    from enriched_features import EnrichedFeatureExtractor

    data = read_instance("instance.txt", number_scen=10, k_override=4)
    df_raw, df_scaled = EnrichedFeatureExtractor(data).extract()
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from collections import deque
from typing import Dict, List
from sklearn.preprocessing import StandardScaler

from Instance_Reader import InstanceData, read_instance


# =============================================================================
# GRAPH UTILITIES
# =============================================================================

def _precedence_levels(data: InstanceData) -> Dict[int, int]:
    """
    Longest-path DP in topological order.

    level[i] = length of longest path from source to activity i
             = number of edges on the longest chain ending at i

    Formula:  level[i] = max over all predecessors j of (level[j] + 1)
    Computed in topological order so level[j] is final when we process j.

    Guarantees: for every edge j -> i,  level[j] < level[i]
    This means activities at the same level have NO precedence
    relationship — safe to group and fix simultaneously in MIP.
    """
    n     = data.n_jobs_including_dummy
    level = {j: 0 for j in range(n)}

    # Step 1: Kahn's topological sort (compute in-degrees)
    in_deg = {j: 0 for j in range(n)}
    for j in range(n):
        for s in data.successors.get(j, []):
            in_deg[s] = in_deg.get(s, 0) + 1

    # Step 2: Initialize queue with zero in-degree nodes
    queue = deque([j for j in range(n) if in_deg[j] == 0])
    topo:  List[int] = []

    while queue:
        node = queue.popleft()
        topo.append(node)
        for s in data.successors.get(node, []):
            in_deg[s] -= 1
            if in_deg[s] == 0:
                queue.append(s)

    # Step 3: Longest-path DP over topological order
    # level[s] = max(level[s], level[j] + 1) for every edge j -> s
    for j in topo:
        for s in data.successors.get(j, []):
            if level[j] + 1 > level[s]:
                level[s] = level[j] + 1

    return level


# =============================================================================
# SCENARIO UTILITIES
# =============================================================================

def _scenario_arrays(data: InstanceData) -> Dict[int, Dict[int, np.ndarray]]:
    """
    Build {resource_k: {job_i: array of workloads across all scenarios}}.

    W_{i,k,pi} = workload of activity i on resource k in scenario pi
    Array for job i, resource k = [W_{i,k,1}, W_{i,k,2}, ..., W_{i,k,S}]
    """
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
    Compute the 4 stochastic features for one activity on one resource.

    mean       = (1/S) * sum(W_pi)
               = expected workload

    cv         = std(W_pi) / mean
               = coefficient of variation
               = 0   if deterministic
               = high if highly variable

    iqr        = Q75(W_pi) - Q25(W_pi)
               = interquartile range
               = robust measure of spread (ignores extreme scenarios)

    tail_ratio = p90(W_pi) / mean
               = how heavy is the upper tail relative to average
               = 1.0 if no surprise scenarios
               = high if worst scenarios demand much more than average
    """
    if len(arr) == 0 or np.all(arr == 0):
        return {"mean": 0.0, "cv": 0.0, "iqr": 0.0, "tail_ratio": 0.0}

    mean = float(np.mean(arr))
    std  = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    cv   = std / mean if mean > 0 else 0.0

    p25  = float(np.percentile(arr, 25))
    p75  = float(np.percentile(arr, 75))
    p90  = float(np.percentile(arr, 90))
    iqr  = p75 - p25

    tail_ratio = p90 / mean if mean > 0 else 0.0

    return {
        "mean":       mean,
        "cv":         cv,
        "iqr":        iqr,
        "tail_ratio": tail_ratio,
    }


# =============================================================================
# FEATURE EXTRACTOR
# =============================================================================

class EnrichedFeatureExtractor:
    """
    Builds the minimum sufficient feature matrix for FSRCPSP grouping.

    Parameters
    ----------
    data            : InstanceData from Instance_Reader.read_instance()
    include_dummies : if True, include source/sink (default False)
    """

    def __init__(self, data: InstanceData, include_dummies: bool = False):
        self.data            = data
        self.include_dummies = include_dummies

    def extract(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Compute features for all real activities.

        Returns
        -------
        df_raw    : features in original units
        df_scaled : z-score normalised  →  use this for clustering
        """
        data = self.data
        n    = data.n_jobs_including_dummy
        sink = n - 1
        n_R  = data.n_renewable_resources

        # Precompute
        levels = _precedence_levels(data)
        scen_w = _scenario_arrays(data)

        rows: List[Dict] = []

        for job in range(n):
            if not self.include_dummies and job in (0, sink):
                continue

            # ── NETWORK POSITION ──────────────────────────────────
            # precedence_level:
            #   level[i] = max(level[j] + 1) for all predecessors j
            #   = length of longest path from source to i
            precedence_level = float(levels[job])

            # in_degree: |{j : j -> i}|
            in_degree  = float(len(data.predecessors.get(job, [])))

            # out_degree: |{j : i -> j}|
            out_degree = float(len(data.successors.get(job, [])))

            # ── SCHEDULING URGENCY ────────────────────────────────
            es_i = int(data.es.get(job, 0))
            ls_i = int(data.ls.get(job, es_i))
            dur  = int(data.duration.get(job, 0))

            # slack = ls_i - es_i
            # = 0   → activity is on the critical path
            # large → activity has scheduling freedom
            slack = float(max(0, ls_i - es_i))

            # pressure_index = d_i / (ls_i - es_i + 1)
            # = 1.0 → duration fills entire window (very tight)
            # ≈ 0.0 → duration is short relative to window (loose)
            window         = max(1, ls_i - es_i + 1)
            pressure_index = float(dur) / float(window)

            # ── ASSEMBLE ROW ──────────────────────────────────────
            row: Dict = {
                "job": job,

                # Network position (3 features)
                "precedence_level": precedence_level,
                "in_degree":        in_degree,
                "out_degree":       out_degree,

                # Scheduling urgency (2 features)
                "slack":            slack,
                "pressure_index":   pressure_index,
            }

            # Per-resource features (2 per resource only)
            #
            # Empirical correlation analysis on j301_1 (n=30, k=4) showed:
            #   iqr_Wk     vs mean_Wk  → |r| = 0.95–1.00  DROP iqr
            #   tail_ratio vs cv_Wk    → |r| = 0.98–1.00  DROP tail_ratio
            #   density_W1 vs mean_W1  → |r| = 0.86       DROP density
            #
            # Root cause: with S=10 scenarios, p90 ≈ max, so tail_ratio
            # behaves like cv. IQR scales proportionally with mean because
            # all scenarios share the same relative spread pattern.
            #
            # Final selection — 2 orthogonal features per resource:
            #   mean_Wk → magnitude of demand  (HOW MUCH resource)
            #   cv_Wk   → relative variability (HOW VARIABLE)
            #
            # These two are genuinely orthogonal:
            #   high mean, low  cv → heavy but predictable
            #   low  mean, high cv → light but unpredictable
            #   high mean, high cv → heavy and unpredictable (hardest)
            for k in range(1, n_R + 1):
                arr   = scen_w.get(k, {}).get(job, np.array([]))
                stats = _resource_stats(arr)

                # mean_Wk = (1/S) * Σ W_{i,k,π}
                row[f"mean_W{k}"] = stats["mean"]

                # cv_Wk = σ_k / μ_k
                row[f"cv_W{k}"]   = stats["cv"]

            rows.append(row)

        # ── Build DataFrame ───────────────────────────────────────
        df_raw = pd.DataFrame(rows).set_index("job")

        # Drop all-zero columns
        # e.g. W2, W3, W4 features are all zero for k=1 instances
        df_raw = df_raw.loc[:, (df_raw != 0).any(axis=0)]

        # Drop constant columns
        # A feature with zero variance adds no clustering information
        df_raw = df_raw.loc[:, df_raw.nunique() > 1]

        # Z-score normalisation: x̃ = (x - μ) / σ
        # Without this, slack=80 dominates cv=0.3 purely due to scale
        scaler    = StandardScaler()
        scaled    = scaler.fit_transform(df_raw.values)
        df_scaled = pd.DataFrame(
            scaled, index=df_raw.index, columns=df_raw.columns
        )

        return df_raw, df_scaled

    def feature_summary(self) -> None:
        """Print a clean summary of all features and their justification."""
        n_R        = self.data.n_renewable_resources
        n_features = 5 + n_R * 2

        print("\n" + "=" * 70)
        print("  EnrichedFeatureExtractor — Minimum Sufficient Feature Set")
        print("  (Validated by correlation analysis)")
        print("=" * 70)
        print(f"\n  {'#':<3} {'Feature':<22} {'Formula':<28} {'Answers'}")
        print("  " + "-" * 65)

        base = [
            (1, "precedence_level",  "longest_path_DP(i)",   "Where in network?"),
            (2, "in_degree",         "|predecessors(i)|",    "How many must finish first?"),
            (3, "out_degree",        "|successors(i)|",      "How many does it unlock?"),
            (4, "slack",             "ls_i - es_i",          "How much scheduling freedom?"),
            (5, "pressure_index",    "d_i / (ls_i-es_i+1)",  "How tight is the window?"),
        ]
        for num, name, formula, question in base:
            print(f"  {num:<3} {name:<22} {formula:<28} {question}")

        idx = 6
        for k in range(1, n_R + 1):
            print(f"\n  --- Resource {k} ---")
            print(f"  {idx:<3} {'mean_W'+str(k):<22} {'(1/S) Σ W_{{i,'+str(k)+',π}}':<28} Expected R{k} demand")
            print(f"  {idx+1:<3} {'cv_W'+str(k):<22} {'σ_k / μ_k':<28} Relative variability R{k}")
            idx += 2

        print(f"\n  Total (before auto-drop): {n_features} features")
        print(f"\n  Dropped (high correlation):")
        print(f"    iqr_Wk       — |r|=0.95-1.00 with mean_Wk")
        print(f"    tail_ratio_Wk— |r|=0.98-1.00 with cv_Wk")
        print(f"    density_W1   — |r|=0.86      with mean_W1")
        print("=" * 70 + "\n")


# =============================================================================
# MAIN — demo
# =============================================================================

if __name__ == "__main__":

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
    print(f"  Raw Feature Matrix")
    print(f"  {len(df_raw)} activities  x  {df_raw.shape[1]} features")
    print(f"{'=' * 70}")
    print(df_raw.to_string())

    print(f"\n{'=' * 70}")
    print(f"  Scaled Feature Matrix (z-scores)")
    print(f"{'=' * 70}")
    print(df_scaled.to_string())

    # Check for high correlations — should be none above 0.85
    print(f"\n{'=' * 70}")
    print("  Correlation check (flag |r| >= 0.85)")
    print(f"{'=' * 70}")
    corr      = df_raw.corr().round(2)
    cols      = df_raw.columns.tolist()
    high_corr = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = abs(float(corr.iloc[i, j]))
            if r >= 0.85:
                high_corr.append((cols[i], cols[j], round(r, 2)))

    if high_corr:
        print(f"  WARNING: {len(high_corr)} high-correlation pairs found:")
        for a, b, r in sorted(high_corr, key=lambda x: -x[2]):
            print(f"    {a:<25} vs {b:<25}  |r| = {r}")
    else:
        print("  All clear — no |r| >= 0.85 pairs. Feature set is clean.")