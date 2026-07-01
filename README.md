# Group-Based Activity-Oriented Relax-and-Fix for the FSRCPSP

Master's thesis implementation — solving the **Flexible Stochastic Resource-Constrained Project Scheduling Problem (FSRCPSP)** via a group-based Relax-and-Fix decomposition with Gurobi.

> **Solving the FSRCPSP Using Hybrid Machine Learning Approaches**
> Souvik Chakraborty, Universitat Hildesheim, 2026

## Problem

The FSRCPSP extends the classical RCPSP with:
- **Flexible resource allocations**: each activity's resource usage can vary within bounds across its duration
- **Stochastic workloads**: resource consumption is scenario-dependent, drawn from a Beta(0.5, 2.25) distribution
- **Objective**: minimise expected project makespan via a chance-constrained SAA formulation

The SAA model grows as O(|V| . |K| . |T| . |S|), reaching over 1.3M variables for n=60 with |K|=4 and |S|=10 scenarios.

## Approach

### Baseline — Activity-Oriented R&F (`Base Model/AO_FR_Gurobi.py`)
Classic Relax-and-Fix: fixes one activity per subproblem in topological order, relaxing the rest.

### Proposed — Group-Based R&F (`AO_FR_Gurobi_updated.py`)
1. **Feature extraction** (`feature_extractor.py`) — 50 candidate activity features (18 + 8|K|) spanning network topology, duration, urgency, timing, and stochastic resource statistics
2. **Ward hierarchical clustering** (`activity_grouping.py`) — partitions activities into precedence-safe groups with silhouette-based cut and z-score merge
3. **Group-level R&F** — one group per subproblem instead of one activity, giving the solver richer combinatorial context
4. **Time aggregation** (`time_aggregation.py`) — compresses the time horizon from |T| to ceil(|T|/alpha) periods
5. **Per-group scenario reduction** (`scenario_reduction.py`) — PAM K-medoids selects the 3 most representative scenarios per group subproblem

Warm-start: **SSGS** (`SSGS.py`) — Serial Schedule Generation Scheme with worst-case scenario selection.

## Repository Structure

```
.
├── Base Model/                    # Baseline activity-oriented R&F
│   ├── AO_FR_Gurobi.py           #   Baseline R&F solver
│   ├── AO_Model.py               #   SAA-FSRCPSP MIP model builder
│   ├── SSGS.py                   #   SSGS warm-start heuristic
│   └── Main.py                   #   Baseline experiment runner
│
├── AO_FR_Gurobi_updated.py       # Group-based R&F solver (main method)
├── main_updated.py               # Group-based experiment runner
├── activity_grouping.py          # Ward clustering + group merging
├── feature_extractor.py          # EnrichedFeatureExtractor (50 features)
├── Instance_Reader.py            # PSPLIB instance parser
├── time_aggregation.py           # Time horizon compression
├── scenario_reduction.py         # Global and per-group scenario reduction
│
├── exp1_j10.py                   # EXP1: j10, baseline vs. grouped, |K|=1..4
├── exp2_j30.py                   # EXP2: j30, baseline vs. grouped, |K|=1..4
├── exp5_j60_time_agg.py          # EXP3: j60 with time aggregation
├── exp7_j30_time_agg.py          # j30 with time aggregation (alpha=2)
├── exp_alpha_sensitivity.py      # Alpha sensitivity across j30/j60/j120
├── exp_ablation_comprehensive.py # 32-config feature ablation (j60, |K|=4)
├── exp_scen_red_j120_allk_n50.py # j120 scenario reduction, |K|=1..4
│
├── FSRCPSP_Instanzen/            # PSPLIB benchmark instances
├── Ergebnisse/                   # Experiment results (CSV)
│   ├── EXP1/                     #   j10 results
│   ├── EXP2_sil030_fixed_merge/  #   j30 results
│   ├── Ablation/                 #   Feature ablation results
│   ├── EXP_SCEN/                 #   Scenario reduction results
│   └── Baseline/                 #   Baseline reference runs
│
├── chapter*.tex                  # LaTeX thesis chapters
├── figures/                      # Thesis figures (TikZ + generated)
└── README.md
```

## Key Results

### j10 — Baseline vs. Group-based (50 instances per |K|, |S|=10)

| |K| | Base Feas | Grp Feas | Base RT | Grp RT | Speedup | Base Gap | Grp Gap |
|:---:|:---------:|:--------:|:-------:|:------:|:-------:|:--------:|:-------:|
| 1 | 50/50 | 50/50 | 112.5s | 33.4s | 3.4x | -1.97% | -1.97% |
| 2 | 50/50 | 50/50 | 176.4s | 28.8s | 6.1x | -3.25% | -3.66% |
| 3 | 50/50 | 50/50 | 137.6s | 298.5s | 0.5x | -3.54% | -2.88% |
| 4 | 43/50 | **50/50** | 796.3s | **68.8s** | **11.6x** | -7.37% | **-8.11%** |

Gap = (makespan - SSGS) / SSGS x 100%. Negative = improvement over warm-start.

### j30 — Baseline vs. Group-based (50 instances per |K|, |S|=10)

| |K| | Base Feas | Grp Feas | Base RT | Grp RT | Speedup |
|:---:|:---------:|:--------:|:-------:|:------:|:-------:|
| 1 | 35/50 | **39/50** | 541.3s | **100.1s** | 5.4x |
| 2 | 38/50 | **40/50** | 656.5s | **160.2s** | 4.1x |
| 3 | 31/50 | **32/50** | 1477.2s | **288.2s** | 5.1x |
| 4 | **28/50** | 26/50 | 1746.8s | **368.1s** | 4.7x |

### j60 — Group-based with Time Aggregation (50 instances per |K|, |S|=10)

| |K| | alpha | Feas | Gap | Impr. | RT |
|:---:|:-----:|:----:|:---:|:-----:|:--:|
| 1 | 5 | **50/50** | +4.62% | 13 | 55.1s |
| 2 | 5 | **50/50** | +4.10% | 11 | 55.0s |
| 3 | 7 | **50/50** | +9.32% | 9 | 54.7s |
| 4 | 7 | **50/50** | +8.76% | 11 | 56.2s |

### j120 — Per-group Scenario Reduction (50 instances per |K|, alpha=15)

| |K| | Baseline Gap | Per-group K-med Gap | Baseline Feas | Per-group Feas |
|:---:|:------------:|:-------------------:|:-------------:|:--------------:|
| 1 | +15.73% | **-12.94%** | 50/50 | 49/50 |
| 2 | +22.66% | **-13.18%** | 49/50 | 49/50 |
| 3 | +18.63% | **-13.74%** | 41/50 | **46/50** |
| 4 | +16.24% | **-16.70%** | 46/50 | **48/50** |

### Feature Ablation (j60, |K|=4, 32 configurations, 10 instances)

The 4-feature network topology set (`net4`: precedence level, in-degree, out-degree, longest path to sink) achieves **10/10 feasibility and 8.09% gap** — Pareto-optimal across all 32 configurations. Adding resource statistics can lower the gap to 7.72% but at the cost of feasibility (9/10) and up to 2.9x higher runtime.

## Requirements

- **Python** 3.11+
- **Gurobi Optimizer** 11.0+ with a valid licence (academic or commercial)
- **NumPy**, **SciPy** 1.12+, **scikit-learn** 1.4+, **pandas**

```bash
pip install numpy scipy scikit-learn pandas
```

## Usage

### Running the group-based approach on a single instance

```python
from Instance_Reader import read_instance
from SSGS import ssgs_est_worst_case
from AO_FR_Gurobi_updated import group_oriented_fix_and_relax

data = read_instance("FSRCPSP_Instanzen/j30/k = 2/j3010_1/j3010_1.txt",
                     number_scen=10, k_override=2)
ssgs_result = ssgs_est_worst_case(data)
result = group_oriented_fix_and_relax(data, ssgs_result, time_limit=600,
                                      use_time_aggregation=True, time_alpha=5)
```

### Reproducing experiments

Each `exp_*.py` script is self-contained:

```bash
python exp1_j10.py                    # j10 baseline vs. group-based
python exp2_j30.py                    # j30 baseline vs. group-based
python exp5_j60_time_agg.py           # j60 with time aggregation
python exp_alpha_sensitivity.py       # Alpha sensitivity (j30/j60/j120)
python exp_ablation_comprehensive.py  # 32-config feature ablation
python exp_scen_red_j120_allk_n50.py  # j120 scenario reduction
```

Results are written to `Ergebnisse/` as semicolon-delimited CSV files.

## Instance Format

Instances follow PSPLIB-style `.txt` format (Kolisch & Sprecher, 1997), organised as:

```
FSRCPSP_Instanzen/j{size}/k = {K}/{instance_name}/{instance_name}.txt
```

The resource flexibility level `k` is inferred from the folder path by `Instance_Reader.py`. Stochastic workloads are generated at runtime by scaling deterministic requirements with Beta(0.5, 2.25) draws.

## Citation

```bibtex
@mastersthesis{chakraborty2026fsrcpsp,
  author  = {Chakraborty, Souvik},
  title   = {Solving the Flexible Stochastic Resource-Constrained Project
             Scheduling Problem Using Hybrid Machine Learning Approaches},
  school  = {Universit{\"a}t Hildesheim},
  year    = {2026}
}
```

## Licence

This project was developed as part of a Master's thesis at Universitat Hildesheim. Please contact the author for licensing details.
