# Project: AO_RF_Gurobi — FSRCPSP Fix-and-Relax

## What this project is
Solving the **Flexible Stochastic Resource-Constrained Project Scheduling Problem (FSRCPSP)** using two methods:
- **Baseline AO-RF**: Activity-Oriented Fix-and-Relax — solves one activity per subproblem
- **Group-Based AO-RF**: clusters activities via Ward hierarchical clustering, solves one group per subproblem

Warm-start: **SSGS** (Serial Schedule Generation Scheme, worst-case scenario selection).

## Key files
| File | Purpose |
|------|---------|
| `AO_FR_Gurobi.py` | Baseline AO-RF |
| `AO_FR_Gurobi_updated.py` | Group-Based AO-RF (main method) |
| `AO_Model.py` | Gurobi MIP model builder |
| `Instance_Reader.py` | Parses `.txt` instance files, infers k from path |
| `SSGS.py` | SSGS warm-start heuristic |
| `activity_grouping.py` | Ward clustering + feature extraction |
| `exp1_j10.py` | Experiment 1 — j10, k=1..4, baseline vs grouped |
| `exp2_j30.py` | Experiment 2 — j30, k=4, baseline vs grouped |
| `exp_ablation_j30.py` | Ablation study — which feature groups matter |

## Instance folders
| Folder | Format | k detection |
|--------|--------|------------|
| `FSRCPSP_Instanzen/j10/k = {k}/` | j10, k=1..4 | `k = N` — works after regex fix |
| `FSRCPSP_Instanzen/j30/k = {k}/` | j30, k=1..4 | same |
| `Instances_j30_test/K={k}/` | j30 test set, k=1..4 | `K=N` — works |
| `Instanzen_j10_FSRCPSP/k_{k}/` | j10 alternative | `k_N` — works |

## k detection (`_infer_k_from_path` in Instance_Reader.py)
Detects number of renewable resources from folder path. Supports:
- `k_1`, `k_2`, ... (underscore)
- `K=4`, `k=4`, `k = 4`, `K = 4` (equals, case-insensitive, spaces allowed)

**Critical**: if k is not detected, `n_R` silently defaults to 1 — produces wrong SSGS and wrong MIP models. Always verify k is detected correctly when using a new folder.

## Feasibility note
`xEff = xB + xC` where `xEff` is **CONTINUOUS** (not binary). `feasible=1` only when the LP relaxation lands on an integer vertex. `feasible=0` does not mean truly infeasible — it means no integer solution was certified within the time limit.

## Known issues / decisions
- **SSGS bug**: `select_workload_worst_case_for_activity` selects worst-case scenario by comparing workloads across all resources (not per-resource) — dimensionally inconsistent but not yet fixed
- **Old results invalid**: CSVs in `Ergebnisse/Baseline/j30/` and `Ergebnisse/Grouped/j30/` used `FSRCPSP_Instanzen/j30/k = 4/` when k detection was broken → n_R=1 was used → SSGS=76 (wrong). Correct value is SSGS=81 with k=4
- **j30 k=4 is very slow**: 10 scenarios → 557k rows, 280k columns, 61M nonzeros. MIP start alone takes ~115s. Each instance can take 1-2 hours. Consider reducing scenarios or time limits

## Experiment config (current)
```
NUMBER_SCEN      = 10
EPSILON          = 0.1
TIME_LIMIT_SEC   = 3600   (exp2), 600 (ablation)
TIME_LIMIT_MODEL = 60
stop_on_first_feasible = True  (grouped method)
```

## Results location
```
Ergebnisse/EXP1/exp1_j10_k{k}.csv       — per-k results
Ergebnisse/EXP1/exp1_j10_summary.csv    — summary across k=1..4
Ergebnisse/EXP2/exp2_j30_k4_first_feasible.csv  — current run (in progress)
Ergebnisse/Ablation/ablation_j30_k4_{config}.csv
```

## EXP1 summary (j10, 10 scen, 50 instances per k)
| K | Base Feas% | Grp Feas% | Avg Base RT | Avg Grp RT |
|---|-----------|----------|------------|-----------|
| 1 | 62% | 66% | 58.2s | 43.4s |
| 2 | 88% | 90% | 46.5s | 23.5s |
| 3 | 100% | 100% | 14.6s | 14.4s |
| 4 | 88% | 98% | 103.6s | 70.0s |

Grouped method is faster and finds more feasible solutions, especially at k=4.

## Ablation feature groups (j30 k=4)
```python
NETWORK_COLS  = ["precedence_level", "in_degree", "out_degree"]
URGENCY_COLS  = ["slack", "pressure_index"]
RESOURCE_COLS = [f"mean_W{k}" for k in range(1, K+1)] + [f"cv_W{k}" for k in range(1, K+1)]
```
7 configs: all, no_network, no_urgency, no_resource, network_only, urgency_only, resource_only.
Ablation on j30 k=4 is too slow (model too large). Consider j10 k=4 or j30 k=1/k=2 instead.
