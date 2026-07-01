# Group-Based Activity-Oriented Fix-and-Relax for the FSRCPSP

Master's thesis implementation — solving the **Flexible Stochastic Resource-Constrained Project Scheduling Problem (FSRCPSP)** via a group-based Fix-and-Relax (F&R) decomposition using Gurobi.

## Problem

The FSRCPSP extends the classical RCPSP with:
- **Flexible resource allocations**: each activity can be executed at different resource intensity levels
- **Stochastic workloads**: resource consumption is scenario-dependent
- **Objective**: minimise project makespan subject to resource capacity constraints across all scenarios

## Methods

### Baseline — Activity-Oriented AO-RF (`AO_FR_Gurobi.py`)
Classic Fix-and-Relax: solves one activity at a time, fixing previously scheduled activities and relaxing the rest.

### Proposed — Group-Based AO-RF (`AO_FR_Gurobi_updated.py`)
Key contributions:
1. **Activity grouping** via Ward hierarchical clustering on 4 network topology features (`precedence_level`, `in_degree`, `out_degree`, `longest_path_to_sink`) — empirically validated as optimal out of 50 candidate features
2. **Group-level F&R**: one group of activities per subproblem instead of one activity
3. **Per-group scenario reduction**: K-medoids (PAM) restricted to each group's workload vectors reduces scenarios from S=10 to n=3 per subproblem, enabling tractability on large instances (j120)

Warm-start: **SSGS** (Serial Schedule Generation Scheme, worst-case scenario selection).

## Repository Structure

```
.
├── AO_FR_Gurobi.py              # Baseline AO-RF
├── AO_FR_Gurobi_updated.py      # Group-based AO-RF (main method)
├── AO_Model.py                  # Gurobi MIP model builder
├── Instance_Reader.py           # Parses .txt instance files
├── SSGS.py                      # SSGS warm-start heuristic
├── activity_grouping.py         # Ward clustering + group merging
├── feature_extractor.py         # 50-feature candidate pool extractor
├── scenario_reduction.py        # Global and per-group scenario reduction
├── time_aggregation.py          # Time-window aggregation utilities
│
├── exp1_j10.py                  # EXP1: j10 baseline vs grouped, K=1..4
├── exp2_j30.py                  # EXP2: j30 baseline vs grouped, K=1..4
├── exp_ablation_comprehensive.py # Ablation: 31 configs over 50 features
├── exp_scen_red_j120_allk_n50.py # EXP_SCEN: j120 scenario reduction, K=1..4
│
└── Ergebnisse/
    ├── EXP1/                    # j10 results (50 instances × K=1..4)
    ├── EXP2_sil030_fixed_merge/ # j30 results (50 instances × K=1..4)
    ├── Ablation/                # Feature ablation results (j60, K=4)
    └── EXP_SCEN/                # Scenario reduction results (j120, K=1..4)
```

## Experimental Results

### EXP1 — j10 (50 instances per K, S=10 scenarios)

| K | Base Feas% | Grp Feas% | Base Avg RT | Grp Avg RT | Base Gap | Grp Gap |
|---|:----------:|:---------:|:-----------:|:----------:|:--------:|:-------:|
| 1 | 100% | 100% | 112.5s | 33.4s | −1.97% | −1.97% |
| 2 | 100% | 100% | 176.4s | 28.8s | −3.25% | −3.66% |
| 3 | 100% | 100% | 137.6s | 298.5s | −3.54% | −2.88% |
| 4 | 86% | **100%** | 796.3s | **68.8s** | −7.37% | **−8.11%** |

Gap = (obj − SSGS) / SSGS × 100% — negative means improvement over warm-start.

### EXP2 — j30 (50 instances per K, S=10 scenarios)

| K | Base Feas% | Grp Feas% | Base Avg RT | Grp Avg RT |
|---|:----------:|:---------:|:-----------:|:----------:|
| 1 | 70% | **78%** | 541.3s | **100.1s** |
| 2 | 76% | **80%** | 656.5s | **160.2s** |
| 3 | 62% | **64%** | 1477.2s | **288.2s** |
| 4 | **56%** | 52% | 1746.8s | **368.1s** |

### Feature Ablation — j60 K=4 (31 configs, 50 candidate features, 10 instances)

The 4-feature network topology set (`net4`) achieves **10/10 feasibility and 8.09% avg gap** — matching or outperforming all 31 configurations including the full 50-feature set. Adding more features never improves results, confirming that network topology alone is sufficient for effective activity grouping.

### EXP_SCEN — j120 Scenario Reduction (50 instances per K)

| K | Baseline Feas% | Baseline Gap | Per-group KMD Feas% | Per-group KMD Gap |
|---|:--------------:|:------------:|:-------------------:|:-----------------:|
| 1 | 100% | +15.73% | 98% | **−12.94%** |
| 2 | 98% | +22.66% | 98% | **−13.18%** |
| 3 | 82% | +18.63% | **92%** | **−13.74%** |
| 4 | 92% | +16.24% | **96%** | **−16.70%** |

Per-group K-medoids (n\_keep=3 from S=10) consistently produces solutions better than SSGS while the baseline without reduction fails to improve on the warm-start.

## Requirements

- Python 3.9+
- [Gurobi](https://www.gurobi.com/) (academic license)
- `numpy`, `scipy`, `scikit-learn`, `pandas`

```bash
pip install numpy scipy scikit-learn pandas
```

## Instance Format

Instances follow the PSPLIB-style `.txt` format, organised by problem size and number of resources:

```
FSRCPSP_Instanzen/j{size}/k = {K}/{instance_name}/{instance_name}.txt
```

`k` is inferred automatically from the folder path by `Instance_Reader.py`.

## Running Experiments

```bash
# EXP1 — j10 comparison
python exp1_j10.py

# EXP2 — j30 comparison
python exp2_j30.py

# Comprehensive feature ablation (j60 K=4)
python exp_ablation_comprehensive.py

# Scenario reduction on j120
python exp_scen_red_j120_allk_n50.py
```

Results are written to `Ergebnisse/` as semicolon-delimited CSV files.
