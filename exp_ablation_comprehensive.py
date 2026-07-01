# exp_ablation_comprehensive.py
#
# Comprehensive ablation over the FULL 50-feature candidate pool (K=4).
#
# Motivation: the original feature set of 25 (later pruned to 16 then 4)
# was designed ad-hoc without a reference paper. For thesis rigour, we
# enumerate ALL plausible activity features (50 for K=4) and show that
# the 4-feature network subset remains optimal across this wider search.
#
# Design (3 phases):
#
#   Phase 1 — Single-group baselines (which group is best alone?):
#     net_basic, net_ext, duration, urgency, timing,
#     res_joint, res_mean, res_cv, res_iqr, res_tail,
#     res_max, res_min, res_range, res_pnz
#
#   Phase 2 — net4 + one group (does anything improve net4?):
#     net4+net_ext, net4+duration, net4+urgency, net4+timing,
#     net4+res_joint, net4+res_mean, net4+res_cv, net4+res_iqr,
#     net4+res_tail, net4+res_max, net4+res_min, net4+res_range, net4+res_pnz
#
#   Phase 3 — Aggregate controls:
#     all_net   (A+B), all_struct (A+B+C+D+E+F),
#     all_res   (F+G all stats), all_50 (everything)
#
# Reference: net4 = 6.00% (ablation-validated on j60 K=4, 10 instances)
#
# Settings: j60 K=4, alpha=7, 10 instances, Pool(6) × 2 Gurobi threads
# Output: Ergebnisse/Ablation/comp_ablation_{config}.csv
#         Ergebnisse/Ablation/comp_ablation_summary.csv

from __future__ import annotations
import os, csv, glob, time, subprocess
from decimal import Decimal
from multiprocessing import Pool
from typing import Any, Dict, List, Tuple

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

PROJECT_DIR = "/Users/souvikchakraborty/Downloads/AO_RF_Gurobi-5"
OUTPUT_DIR  = os.path.join(PROJECT_DIR, "Ergebnisse/Ablation")

K              = 4
TIME_ALPHA     = 7
N_INSTANCES    = 10
NUMBER_SCEN    = 10
EPSILON        = Decimal("0.1")
TIME_LIMIT_SEC   = 600
TIME_LIMIT_MODEL = 60
MAX_GROUP_SIZE   = 10
MERGE_THRESHOLD  = 2.5
CORR_THRESHOLD   = 0.95
GUROBI_THREADS   = 2   # Pool(6) × 2 = 12 cores (slight oversubscription, but each run is short)

ABLATION_REFERENCE = 6.00   # net4 result on j60 K=4

# Feature groups (K=4)
NET4      = ["precedence_level", "in_degree", "out_degree", "longest_path_to_sink"]
NET_EXT   = ["n_transitive_succ", "n_transitive_pred", "betweenness_centrality"]
DURATION  = ["duration", "rel_duration", "duration_rank"]
URGENCY   = ["slack", "pressure_index", "n_concurrent"]
TIMING    = ["rel_es", "rel_ls", "critical_path", "float_ratio"]
RES_JOINT = ["resource_pressure"]
RES_MEAN  = [f"mean_W{k}"       for k in range(1, K + 1)]
RES_CV    = [f"cv_W{k}"         for k in range(1, K + 1)]
RES_IQR   = [f"iqr_W{k}"        for k in range(1, K + 1)]
RES_TAIL  = [f"tail_ratio_W{k}" for k in range(1, K + 1)]
RES_MAX   = [f"max_W{k}"        for k in range(1, K + 1)]
RES_MIN   = [f"min_W{k}"        for k in range(1, K + 1)]
RES_RANGE = [f"range_W{k}"      for k in range(1, K + 1)]
RES_PNZ   = [f"p_nonzero_W{k}"  for k in range(1, K + 1)]

ALL_NET    = NET4 + NET_EXT
ALL_STRUCT = NET4 + NET_EXT + DURATION + URGENCY + TIMING + RES_JOINT
ALL_RES    = RES_JOINT + RES_MEAN + RES_CV + RES_IQR + RES_TAIL + RES_MAX + RES_MIN + RES_RANGE + RES_PNZ
ALL_50     = ALL_STRUCT + RES_MEAN + RES_CV + RES_IQR + RES_TAIL + RES_MAX + RES_MIN + RES_RANGE + RES_PNZ

CONFIGS: Dict[str, List[str]] = {
    # ── Reference ────────────────────────────────────────────────────
    "net4":              NET4,

    # ── Phase 1: Individual groups alone ─────────────────────────────
    "p1_net_basic":      NET4,         # same as net4, explicit label
    "p1_net_ext":        NET_EXT,
    "p1_duration":       DURATION,
    "p1_urgency":        URGENCY,
    "p1_timing":         TIMING,
    "p1_res_joint":      RES_JOINT,
    "p1_res_mean":       RES_MEAN,
    "p1_res_cv":         RES_CV,
    "p1_res_iqr":        RES_IQR,
    "p1_res_tail":       RES_TAIL,
    "p1_res_max":        RES_MAX,
    "p1_res_min":        RES_MIN,
    "p1_res_range":      RES_RANGE,
    "p1_res_pnz":        RES_PNZ,

    # ── Phase 2: net4 + one additional group ─────────────────────────
    "p2_net4+net_ext":   NET4 + NET_EXT,
    "p2_net4+duration":  NET4 + DURATION,
    "p2_net4+urgency":   NET4 + URGENCY,
    "p2_net4+timing":    NET4 + TIMING,
    "p2_net4+res_joint": NET4 + RES_JOINT,
    "p2_net4+res_mean":  NET4 + RES_MEAN,
    "p2_net4+res_cv":    NET4 + RES_CV,
    "p2_net4+res_iqr":   NET4 + RES_IQR,
    "p2_net4+res_tail":  NET4 + RES_TAIL,
    "p2_net4+res_max":   NET4 + RES_MAX,
    "p2_net4+res_min":   NET4 + RES_MIN,
    "p2_net4+res_range": NET4 + RES_RANGE,
    "p2_net4+res_pnz":   NET4 + RES_PNZ,

    # ── Phase 3: Aggregate controls ───────────────────────────────────
    "p3_all_net":        ALL_NET,
    "p3_all_struct":     ALL_STRUCT,
    "p3_all_res":        ALL_RES,
    "p3_all_50":         ALL_50,
}


def _avg(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else float("nan")


def run_config(args: Tuple[str, List[str], List[str]]) -> List[Dict[str, Any]]:
    config_name, feature_cols, paths = args

    from Instance_Reader import read_instance
    from SSGS import ssgs_est_worst_case
    from AO_FR_Gurobi_updated import group_oriented_fix_and_relax

    output_csv = os.path.join(OUTPUT_DIR, f"comp_ablation_{config_name}.csv")

    print(f"\n{'#'*65}")
    print(f"  CONFIG: {config_name}  ({len(feature_cols)} features)")
    print(f"{'#'*65}", flush=True)

    rows: List[Dict[str, Any]] = []

    for i, txt_path in enumerate(paths, 1):
        inst   = os.path.splitext(os.path.basename(txt_path))[0]
        data   = read_instance(txt_path, number_scen=NUMBER_SCEN, k_override=K)
        n_jobs = int(data.n_jobs_including_dummy)

        base     = ssgs_est_worst_case(data, scenario_keep=list(range(1, NUMBER_SCEN + 1)))
        ssgs_obj = int(base.S.get(n_jobs - 1, 10**9))

        print(f"[{i:2d}/{len(paths)}] {inst}  ssgs={ssgs_obj}", flush=True)
        t0 = time.time()
        try:
            res, iters = group_oriented_fix_and_relax(
                data                   = data,
                number_scen            = NUMBER_SCEN,
                epsilon                = EPSILON,
                time_limit_sec         = TIME_LIMIT_SEC,
                time_limit_model       = TIME_LIMIT_MODEL,
                init_S                 = None,
                verbose                = False,
                working_directory      = PROJECT_DIR,
                instance_name          = inst,
                max_no_improve         = 0,
                corr_threshold         = CORR_THRESHOLD,
                max_group_size         = MAX_GROUP_SIZE,
                merge_threshold        = MERGE_THRESHOLD,
                stop_on_first_feasible = True,
                log_to_file            = False,
                time_alpha             = TIME_ALPHA,
                gurobi_threads         = GUROBI_THREADS,
                feature_cols           = feature_cols,
            )
            obj  = int(res.obj)
            feas = int(res.feasible)
            rt   = round(float(res.runtime_total_sec), 2)
            wall = round(time.time() - t0, 2)
        except Exception as e:
            print(f"  ERROR: {e}", flush=True)
            import traceback; traceback.print_exc()
            obj = feas = rt = wall = None

        gap = (None if obj is None or ssgs_obj == 0
               else round((obj - ssgs_obj) / ssgs_obj * 100, 2))
        print(f"  feas={feas}  gap={gap}%  rt={rt}s", flush=True)

        rows.append({
            "config":   config_name,
            "n_feats":  len(feature_cols),
            "instance": inst,
            "ssgs_obj": ssgs_obj,
            "obj":      obj,
            "feas":     feas,
            "rt_sec":   rt,
            "wall_sec": wall,
            "gap_pct":  gap,
        })

        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter=";")
            w.writeheader(); w.writerows(rows)

    n_feas  = sum(1 for r in rows if r["feas"] == 1)
    avg_gap = _avg(r["gap_pct"] for r in rows if r["feas"] == 1)
    avg_rt  = _avg(r["rt_sec"]  for r in rows)
    print(f"\n  {config_name}: {n_feas}/{len(rows)} feas | "
          f"avg_gap={avg_gap:.2f}% | avg_rt={avg_rt:.1f}s", flush=True)
    return rows


if __name__ == "__main__":
    subprocess.Popen(["caffeinate", "-i", "-w", str(os.getpid())])
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    inst_dir = os.path.join(PROJECT_DIR, f"FSRCPSP_Instanzen/j60/k = {K}")
    paths    = sorted(glob.glob(os.path.join(inst_dir, "**/*.txt"),
                                recursive=True))[:N_INSTANCES]
    if not paths:
        print(f"No instances found under {inst_dir}"); exit(1)

    print(f"Comprehensive Ablation | j60 K={K} | alpha={TIME_ALPHA}")
    print(f"{len(paths)} instances | {len(CONFIGS)} configs | Pool(6) × {GUROBI_THREADS} threads")
    print(f"Reference: net4 = {ABLATION_REFERENCE}%\n")
    for name, cols in CONFIGS.items():
        print(f"  {name:<24} ({len(cols):>2} features)")

    tasks = [(name, cols, paths) for name, cols in CONFIGS.items()]
    with Pool(processes=6) as pool:
        results = pool.map(run_config, tasks)

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"  COMPREHENSIVE ABLATION SUMMARY — j60 K={K} alpha={TIME_ALPHA}")
    print(f"  Reference: net4 = {ABLATION_REFERENCE}%  (validated baseline)")
    print(f"  {'config':<24} {'feats':>6} {'feas':>6} {'avg_gap':>9} {'vs net4':>9}")
    print(f"  {'-'*62}")

    summary = []
    for name, config_rows in zip(CONFIGS.keys(), results):
        n_feas  = sum(1 for r in config_rows if r["feas"] == 1)
        avg_gap = _avg(r["gap_pct"] for r in config_rows if r["feas"] == 1)
        avg_rt  = _avg(r["rt_sec"]  for r in config_rows)
        n_feats = len(CONFIGS[name])
        delta   = avg_gap - ABLATION_REFERENCE
        marker  = " *** BETTER" if delta < -0.5 else (" WORSE" if delta > 0.5 else " (≈)")
        print(f"  {name:<24} {n_feats:>6} {n_feas:>3}/{len(config_rows)}  "
              f"{avg_gap:>8.2f}%  {delta:>+8.2f}%{marker}")
        summary.append({
            "config":       name,
            "n_feats":      n_feats,
            "feas_count":   n_feas,
            "avg_gap_pct":  round(avg_gap, 2),
            "avg_rt_sec":   round(avg_rt, 2),
            "delta_vs_net4": round(delta, 2),
        })
    print(f"{'='*72}")

    # Sort and show top configs
    ranked = sorted(summary, key=lambda x: x["avg_gap_pct"])
    print(f"\n  Top 5 configs by avg_gap:")
    for r in ranked[:5]:
        print(f"    {r['config']:<24} {r['avg_gap_pct']:>6.2f}%  "
              f"({r['delta_vs_net4']:>+.2f}% vs net4)")

    summary_path = os.path.join(OUTPUT_DIR, "comp_ablation_summary.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()), delimiter=";")
        w.writeheader(); w.writerows(summary)
    print(f"\n  Summary saved: {summary_path}")
