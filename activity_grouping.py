"""
activity_grouping.py
---------------------
Groups activities for the group-based Activity-Oriented Relax-and-Fix.

Pipeline:
    1. Extract enriched features via EnrichedFeatureExtractor
    2. Prune redundant features (perfect/near-perfect correlations)
    3. Cluster activities within each precedence level using
       hierarchical clustering (Ward linkage) with dynamic cut
       via silhouette score
    4. Optionally merge adjacent-level groups that are sufficiently
       similar (inter-group distance below a threshold)
    5. Validate topological ordering — no group may contain a
       successor before its predecessor group
    6. Return ordered list of groups ready for AO_FR_Gurobi.py

Interface:
    from Instance_Reader import read_instance
    from activity_grouping import ActivityGrouper

    data   = read_instance("instance.txt", number_scen=200, k_override=2)
    grouper = ActivityGrouper(data, number_scen=200)
    groups  = grouper.group()   # List[List[int]]
    order   = grouper.flat_order()  # List[int]  -- drop-in for range(n)

    # Plug into AO_FR_Gurobi.py:
    # replace:  for i in range(n):
    # with:     for i in grouper.flat_order():
"""

from __future__ import annotations

import re
import warnings
import numpy as np
import pandas as pd
from collections import deque, defaultdict
from typing import Dict, List, Optional, Tuple

from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from Instance_Reader import InstanceData, read_instance
from feature_extractor import EnrichedFeatureExtractor

# Ablation-validated default: network topology features are both necessary
# and sufficient. Adding urgency/resource features increases gap (8.09% vs 6.00%)
# due to curse of dimensionality in Ward clustering.
DEFAULT_FEATURE_COLS: List[str] = [
    "precedence_level",
    "in_degree",
    "out_degree",
    "longest_path_to_sink",
]


# =============================================================================
# REDUNDANCY PRUNER
# =============================================================================

def _prune_redundant(df: pd.DataFrame, corr_threshold: float = 0.95) -> pd.DataFrame:
    """
    Remove features that are perfectly or near-perfectly correlated
    with an already-kept feature (greedy, keep first encountered).
    Also drops constant columns (zero variance).
    """
    # Drop constant columns
    df = df.loc[:, df.nunique() > 1]

    corr  = df.corr().abs()
    cols  = list(df.columns)
    drop  : set = set()

    for i in range(len(cols)):
        if cols[i] in drop:
            continue
        for j in range(i + 1, len(cols)):
            if cols[j] in drop:
                continue
            if corr.loc[cols[i], cols[j]] >= corr_threshold:
                drop.add(cols[j])   # drop the later one, keep the earlier

    kept = [c for c in cols if c not in drop]
    return df[kept]


# =============================================================================
# TOPOLOGY UTILITIES
# =============================================================================

def _precedence_levels(data: InstanceData) -> Dict[int, int]:
    """
    Assign each job its precedence level = length of the longest path
    from source to that job (in terms of number of edges).
    Uses iterative longest-path DP in topological order.
    This correctly handles skip edges (predecessor at level k pointing
    to successor at level k+2 or higher).
    """
    n     = data.n_jobs_including_dummy
    level = {j: 0 for j in range(n)}

    # Topological order via Kahn's algorithm
    in_deg  = {j: 0 for j in range(n)}
    for j in range(n):
        for s in data.successors.get(j, []):
            in_deg[s] = in_deg.get(s, 0) + 1

    queue = deque([j for j in range(n) if in_deg[j] == 0])
    topo  : List[int] = []
    while queue:
        node = queue.popleft()
        topo.append(node)
        for s in data.successors.get(node, []):
            in_deg[s] -= 1
            if in_deg[s] == 0:
                queue.append(s)

    # DP over topological order: level[s] = max(level[s], level[j]+1)
    for j in topo:
        for s in data.successors.get(j, []):
            if level[j] + 1 > level[s]:
                level[s] = level[j] + 1

    return level


def _all_predecessors(data: InstanceData) -> Dict[int, set]:
    """Full transitive predecessor set for each job (via BFS)."""
    n      = data.n_jobs_including_dummy
    result = {j: set() for j in range(n)}
    for j in range(n):
        visited: set = set()
        queue = deque(data.predecessors.get(j, []))
        while queue:
            p = queue.popleft()
            if p in visited:
                continue
            visited.add(p)
            result[j].add(p)
            queue.extend(data.predecessors.get(p, []))
    return result


def _validate_group_order(groups: List[List[int]], data: InstanceData) -> bool:
    """
    Check that for every direct precedence edge (job -> successor),
    job's group index is strictly <= successor's group index.

    Uses only DIRECT edges (not transitive) to avoid false positives
    from multi-level skip edges that are topologically valid.
    Returns True if valid, raises ValueError if not.
    """
    job_to_group: Dict[int, int] = {}
    for g_idx, group in enumerate(groups):
        for job in group:
            job_to_group[job] = g_idx

    n    = data.n_jobs_including_dummy
    sink = n - 1

    for job, succs in data.successors.items():
        # Skip source and sink — they are not in any group
        if job in (0, sink):
            continue
        g_job = job_to_group.get(job, -1)
        if g_job == -1:
            continue
        for s in succs:
            if s in (0, sink):
                continue
            g_s = job_to_group.get(s, -1)
            if g_s == -1:
                continue
            # Violation: predecessor group comes AFTER successor group
            if g_job > g_s:
                raise ValueError(
                    f"Topological violation: job {job} (group {g_job}) "
                    f"-> job {s} (group {g_s}), but {g_job} > {g_s}"
                )
    return True


# =============================================================================
# CLUSTERING WITHIN A LEVEL
# =============================================================================

def _cluster_level(
    jobs:            List[int],
    Z_features:      np.ndarray,
    min_group_size:  int = 1,
    max_group_size:  int = 4,
    min_clusters:    int = 1,
) -> List[List[int]]:
    """
    Cluster a set of jobs using Ward hierarchical clustering.
    Optimal number of clusters chosen by silhouette score.
    Falls back to a single group if clustering is not meaningful
    (fewer than 3 jobs or all identical features).

    Parameters
    ----------
    jobs           : job indices (in topological order within the level)
    Z_features     : z-scored feature rows for these jobs (shape: n_jobs x n_feat)
    min_group_size : minimum activities per cluster
    max_group_size : maximum activities per cluster
    min_clusters   : minimum number of clusters to consider (default 1)
    """
    n = len(jobs)

    # Trivial cases
    if n == 1:
        return [jobs]
    if n == 2:
        return [jobs]   # keep together if only 2

    # Maximum sensible k given size constraints
    max_k = min(n, max(2, n // min_group_size))
    max_k = min(max_k, n - 1)   # need at least 2 elements per cluster for silhouette

    if max_k < 2:
        return [jobs]

    # Check if all rows are identical (zero variance -> clustering meaningless)
    if np.allclose(Z_features, Z_features[0]):
        return [jobs]

    # Ward linkage on the feature matrix
    try:
        Z_link = linkage(Z_features, method='ward', optimal_ordering=True)
    except Exception:
        return [jobs]

    # Dynamic cut: sweep k from 2 to max_k, pick best silhouette
    best_k      = 1
    best_score  = -1.0
    dist_matrix = squareform(pdist(Z_features, metric='euclidean'))

    for k in range(2, max_k + 1):
        labels = fcluster(Z_link, k, criterion='maxclust')
        # silhouette requires at least 2 distinct labels
        if len(set(labels)) < 2:
            continue
        try:
            score = silhouette_score(dist_matrix, labels, metric='precomputed')
        except Exception:
            continue
        if score > best_score:
            best_score = score
            best_k     = k

    # If best silhouette is poor (< 0.3), keep as one group
    if best_score < 0.30 and min_clusters <= 1:
        return [jobs]

    labels = fcluster(Z_link, best_k, criterion='maxclust')

    # Assemble clusters, preserving topological order within each cluster
    clusters: Dict[int, List[int]] = defaultdict(list)
    for job, lbl in zip(jobs, labels):
        clusters[lbl].append(job)

    return [clusters[lbl] for lbl in sorted(clusters.keys())]


# =============================================================================
# INTER-GROUP MERGE
# =============================================================================

def _group_centroid(jobs: List[int], Z_df: pd.DataFrame) -> np.ndarray:
    return Z_df.loc[jobs].values.mean(axis=0)


def _adjacent_centroid_distances(
    groups: List[List[int]],
    Z_df:   pd.DataFrame,
) -> List[float]:
    """All Euclidean distances between consecutive group centroids (in z-score space)."""
    dists: List[float] = []
    for i in range(len(groups) - 1):
        c1 = _group_centroid(groups[i],     Z_df)
        c2 = _group_centroid(groups[i + 1], Z_df)
        dists.append(float(np.linalg.norm(c1 - c2)))
    return dists


def _try_merge_adjacent(
    groups:         List[List[int]],
    Z_df:           pd.DataFrame,
    data:           InstanceData,
    merge_threshold: float,
) -> List[List[int]]:
    """
    Merge consecutive groups whose centroids are closer than merge_threshold
    (Euclidean distance in z-score space), provided the merge does not
    introduce any topological violation.

    A merge of g1 + g2 is SAFE only if:
      - No job in g2 is a (direct or transitive) predecessor of any job in g1
        i.e. merging cannot place a predecessor AFTER its successor in group order

    Operates iteratively until no more merges are possible.
    """
    all_pred = _all_predecessors(data)
    changed  = True

    while changed:
        changed = False
        new_groups: List[List[int]] = []
        i = 0
        while i < len(groups):
            if i + 1 < len(groups):
                g1     = groups[i]
                g2     = groups[i + 1]
                g1_set = set(g1)
                g2_set = set(g2)

                c1   = _group_centroid(g1, Z_df)
                c2   = _group_centroid(g2, Z_df)
                dist = float(np.linalg.norm(c1 - c2))

                # Merge is safe if no job in g2 is a predecessor
                # (direct OR transitive) of any job in g1.
                # This prevents placing a g2-job after a g1-job that
                # it must precede.
                merge_safe = True
                for job in g2:
                    # all_pred[j] = full transitive predecessor set of j
                    # if any g1-job has 'job' as a predecessor it means
                    # job must come before that g1-job -> unsafe to merge
                    for g1_job in g1_set:
                        if job in all_pred.get(g1_job, set()):
                            merge_safe = False
                            break
                    if not merge_safe:
                        break

                if dist < merge_threshold and merge_safe:
                    new_groups.append(g1 + g2)
                    i += 2
                    changed = True
                    continue

            new_groups.append(groups[i])
            i += 1

        groups = new_groups

    return groups


# =============================================================================
# MAIN GROUPER
# =============================================================================

class ActivityGrouper:
    """
    Groups activities for group-based Activity-Oriented Relax-and-Fix.

    Parameters
    ----------
    data             : InstanceData from Instance_Reader.read_instance()
    number_scen      : number of scenarios (passed to EnrichedFeatureExtractor)
    corr_threshold   : correlation threshold for redundancy pruning (default 0.95)
    min_group_size   : minimum activities per cluster (default 1)
    max_group_size   : maximum activities per cluster (default 5)
    merge_threshold  : centroid distance threshold for merging adjacent groups.
                       Set to 0.0 to disable merging (default 2.5)
    merge_percentile : if set (0..100), overrides merge_threshold with the
                       given percentile of all adjacent-centroid distances
                       computed once at the start of merging. Self-calibrates
                       per instance, robust to feature-space dimensionality.
                       (default None — use fixed merge_threshold)
    silhouette_min   : minimum silhouette score to accept a split (default 0.30)
    verbose          : print grouping summary (default True)
    """

    def __init__(
        self,
        data:             InstanceData,
        number_scen:      int,
        corr_threshold:   float = 0.95,
        min_group_size:   int   = 1,
        max_group_size:   int   = 5,
        merge_threshold:  float = 2.5,
        merge_percentile: Optional[float] = None,
        verbose:          bool  = True,
        feature_cols:     Optional[List[str]] = None,
    ):
        self.data             = data
        self.number_scen      = number_scen
        self.corr_threshold   = corr_threshold
        self.min_group_size   = min_group_size
        self.max_group_size   = max_group_size
        self.merge_threshold  = merge_threshold
        self.merge_percentile = merge_percentile
        self.verbose          = verbose
        self.feature_cols     = feature_cols   # None = use all; list = restrict to these columns

        self._groups: Optional[List[List[int]]] = None

    # ------------------------------------------------------------------
    def group(self) -> List[List[int]]:
        """
        Run the full grouping pipeline and return ordered list of groups.
        Each group is a list of job indices.
        The outer list is in topological processing order.
        Source (job 0) and sink (job n-1) are excluded.
        """
        data = self.data
        n    = data.n_jobs_including_dummy
        sink = n - 1

        # ── Step 1: Extract enriched features ─────────────────────
        df_raw, df_scaled = EnrichedFeatureExtractor(
            data, include_dummies=False
        ).extract()

        # ── Step 1b: Restrict to requested feature subset ─────────
        if self.feature_cols is not None:
            keep = [c for c in self.feature_cols if c in df_scaled.columns]
            if not keep:
                raise ValueError(
                    f"feature_cols={self.feature_cols} matched no columns in "
                    f"{list(df_scaled.columns)}"
                )
            df_scaled = df_scaled[keep]
            df_raw    = df_raw[[c for c in keep if c in df_raw.columns]]

        # ── Step 2: Prune redundant features ──────────────────────
        df_pruned = _prune_redundant(df_scaled, corr_threshold=self.corr_threshold)

        if self.verbose:
            print(f"\n[Grouper] Features after pruning: "
                  f"{df_scaled.shape[1]} -> {df_pruned.shape[1]}")

        # ── Step 3: Compute precedence levels ─────────────────────
        levels   = _precedence_levels(data)
        real_jobs = [j for j in range(n) if j not in (0, sink)]

        # Group jobs by precedence level
        level_jobs: Dict[int, List[int]] = defaultdict(list)
        for job in real_jobs:
            level_jobs[levels[job]].append(job)

        # ── Step 4: Cluster within each precedence level ──────────
        all_groups: List[List[int]] = []

        for lvl in sorted(level_jobs.keys()):
            jobs_at_level = sorted(level_jobs[lvl])   # sorted for reproducibility

            # Feature rows for these jobs
            feat_rows = df_pruned.loc[jobs_at_level].values

            clusters = _cluster_level(
                jobs          = jobs_at_level,
                Z_features    = feat_rows,
                min_group_size = self.min_group_size,
                max_group_size = self.max_group_size,
            )

            all_groups.extend(clusters)

            if self.verbose:
                for c in clusters:
                    print(f"  Level {lvl:2d} | group {c}")

        # ── Step 5: Optionally merge adjacent similar groups ───────
        # Percentile mode self-calibrates the threshold to the instance:
        # we compute the percentile over the centroid-distance distribution
        # ONCE before merging, and use that as a fixed threshold throughout.
        # This neutralises curse-of-dimensionality effects across k values.
        effective_threshold = self.merge_threshold
        if self.merge_percentile is not None and len(all_groups) >= 2:
            dists = _adjacent_centroid_distances(all_groups, df_pruned)
            if dists:
                effective_threshold = float(
                    np.percentile(dists, self.merge_percentile)
                )
                if self.verbose:
                    print(f"[Grouper] merge_percentile={self.merge_percentile} "
                          f"-> threshold={effective_threshold:.3f} "
                          f"(min={min(dists):.2f}, max={max(dists):.2f})")

        if effective_threshold > 0.0:
            before = len(all_groups)
            all_groups = _try_merge_adjacent(
                groups          = all_groups,
                Z_df            = df_pruned,
                data            = data,
                merge_threshold = effective_threshold,
            )
            after = len(all_groups)
            if self.verbose and after < before:
                print(f"\n[Grouper] Merged {before} -> {after} groups "
                      f"(threshold={self.merge_threshold})")

        # ── Step 6: Validate topology ──────────────────────────────
        try:
            _validate_group_order(all_groups, data)
        except ValueError as e:
            warnings.warn(
                f"[Grouper] Topological violation after merge, "
                f"disabling merge and re-validating: {e}"
            )
            # First try: rebuild without merging
            all_groups = []
            for lvl in sorted(level_jobs.keys()):
                jobs_at_level = sorted(level_jobs[lvl])
                feat_rows = df_pruned.loc[jobs_at_level].values
                clusters  = _cluster_level(
                    jobs           = jobs_at_level,
                    Z_features     = feat_rows,
                    min_group_size = self.min_group_size,
                    max_group_size = self.max_group_size,
                )
                all_groups.extend(clusters)

            # Second try: if still invalid, fall back to one group per level
            try:
                _validate_group_order(all_groups, data)
            except ValueError:
                all_groups = [
                    sorted(level_jobs[lvl])
                    for lvl in sorted(level_jobs.keys())
                ]

        self._groups  = all_groups
        self._feat_df = df_raw   # expose raw features for NN

        if self.verbose:
            self._print_summary(df_raw)

        return self._groups

    # ------------------------------------------------------------------
    def flat_order(self) -> List[int]:
        """
        Flat ordered list of job indices.
        Direct drop-in replacement for range(n) in AO_FR_Gurobi.py.

        Usage in AO_FR_Gurobi.py:
            # replace:
            for i in range(n):
                if i == 0: continue
            # with:
            for i in grouper.flat_order():
        """
        if self._groups is None:
            self.group()
        return [job for group in self._groups for job in group]

    # ------------------------------------------------------------------
    def _print_summary(self, df_raw: pd.DataFrame) -> None:
        n_groups = len(self._groups)
        n_jobs   = sum(len(g) for g in self._groups)
        print(f"\n{'='*60}")
        print(f"  ActivityGrouper Summary")
        print(f"  {n_jobs} activities  ->  {n_groups} groups")
        print(f"{'='*60}")

        key_cols = [c for c in
                    ["mean_W1", "duration", "cv_W1", "slack",
                     "pressure_index", "chain_length", "density_W1"]
                    if c in df_raw.columns]

        for g_idx, group in enumerate(self._groups):
            sub = df_raw.loc[group]
            desc_parts = []
            for c in key_cols:
                vals = sub[c].values
                if len(vals) == 1:
                    desc_parts.append(f"{c}={vals[0]:.2f}")
                else:
                    desc_parts.append(f"{c}=[{vals.min():.1f},{vals.max():.1f}]")
            print(f"  Group {g_idx+1:2d}: jobs={group}  |  "
                  + "  ".join(desc_parts))

        print(f"\n  Processing order: {self.flat_order()}")
        print(f"{'='*60}\n")


# =============================================================================
# MAIN -- demo
# =============================================================================

if __name__ == "__main__":

    # CONFIG -- edit as needed ------------------------------------
    FILEPATH        = "/Users/souvikchakraborty/Downloads/AO_RF_Gurobi-5/Instanzen_j10_FSRCPSP/k_1_test/j102_2/j102_2.txt"
    NUMBER_SCEN     = 200
    K_OVERRIDE      = None   # None = auto-detect from folder name
    # Grouping parameters
    CORR_THRESHOLD  = 0.95   # features with |r| >= this are pruned
    MIN_GROUP_SIZE  = 1      # min activities per group
    MAX_GROUP_SIZE  = 4      # max activities per group
    MERGE_THRESHOLD = 2.5    # centroid distance to merge adjacent groups
                             # set 0.0 to disable merging
    # -------------------------------------------------------------

    if K_OVERRIDE is not None:
        k = K_OVERRIDE
    else:
        m = re.search(r"[/\\]k_(\d+)", FILEPATH)
        k = int(m.group(1)) if m else 2
        print(f"[info] Auto-detected k={k} from path")

    data = read_instance(FILEPATH, number_scen=NUMBER_SCEN, k_override=k)

    grouper = ActivityGrouper(
        data            = data,
        number_scen     = NUMBER_SCEN,
        corr_threshold  = CORR_THRESHOLD,
        min_group_size  = MIN_GROUP_SIZE,
        max_group_size  = MAX_GROUP_SIZE,
        merge_threshold = MERGE_THRESHOLD,
        verbose         = True,
    )

    groups = grouper.group()

    print("\nFinal groups (for AO_FR_Gurobi.py):")
    for i, g in enumerate(groups):
        print(f"  Group {i+1}: {g}")

    print(f"\nDrop-in flat order (replaces range(n)):")
    print(f"  {grouper.flat_order()}")

    print(f"\nUsage in AO_FR_Gurobi.py:")
    print(f"  from activity_grouping import ActivityGrouper")
    print(f"  grouper = ActivityGrouper(data, number_scen=NUMBER_SCEN)")
    print(f"  groups  = grouper.group()")
    print(f"  # then replace:  for i in range(n):")
    print(f"  #         with:  for i in grouper.flat_order():")