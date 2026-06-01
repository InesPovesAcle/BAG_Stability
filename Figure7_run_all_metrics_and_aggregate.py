#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run Figure 7 OOF cross-cohort replication for all Panel A metrics and aggregate outputs.

This wrapper expects the patched OOF Figure 7 script to be in the same code directory:

    Figure7_cross_cohort_final_reads_oof.py

That script reads subject-level OOF validation tables from:

    ablation_<feature_set>/validation_figures_oof/subject_level_validation_input.csv

This wrapper runs Figure 7 for:
    MAE, RMSE, R2, r

Each metric gets its own output directory:
    <base-dir>/Figure7_cross_cohort_replication_<METRIC>/

Then this wrapper aggregates the CSV outputs into:
    <base-dir>/Figure7_cross_cohort_replication_ALL_METRICS/

Example:
    python Figure7_run_all_metrics_and_aggregate.py

Optional:
    python Figure7_run_all_metrics_and_aggregate.py \
        --figure7-script Figure7_cross_cohort_final_reads_oof.py \
        --main-feature-set imaging_only \
        --formats png,pdf
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List

import pandas as pd


DEFAULT_BASE_DIR = Path(
    "/mnt/newStor/paros/paros_WORK/ines/results/"
    "BrainAgeValidation_AllCohorts_BAGBiasCorr_OOFGlobal_BiologicalValidation"
)

DEFAULT_CODE_DIR = Path("/mnt/newStor/paros/paros_WORK/ines/code/BAG_Stability052627")

DEFAULT_METRICS = ["MAE", "RMSE", "R2", "r"]

TABLES_TO_AGGREGATE = [
    "figure7_dataset_summary.csv",
    "figure7_model_metric_summary.csv",
    "figure7_endpoint_correlation_stats.csv",
    "figure7_replication_summary.csv",
    "figure7_manifest.csv",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run patched OOF Figure 7 script for all metrics and aggregate outputs."
    )
    p.add_argument(
        "--code-dir",
        default=str(DEFAULT_CODE_DIR),
        help="Directory containing Figure7_cross_cohort_final_reads_oof.py.",
    )
    p.add_argument(
        "--figure7-script",
        default="Figure7_cross_cohort_final_reads_oof.py",
        help="Patched Figure 7 script filename or absolute path.",
    )
    p.add_argument(
        "--base-dir",
        default=str(DEFAULT_BASE_DIR),
        help="Combined validation base directory.",
    )
    p.add_argument(
        "--results-root",
        default=None,
        help="Optional results root. If omitted, Figure 7 script uses <base-dir>/..",
    )
    p.add_argument(
        "--main-feature-set",
        default="imaging_only",
        help="Main feature set for biological endpoint panels.",
    )
    p.add_argument(
        "--feature-sets",
        default="imaging_only,imaging_demographics,imaging_biomarkers,full,full_no_cardiovascular",
        help="Comma-separated feature sets for performance panel.",
    )
    p.add_argument(
        "--cohorts",
        default="ADNI,ADRC,HABS,AD_DECODE",
        help="Comma-separated cohorts.",
    )
    p.add_argument(
        "--metrics",
        default=",".join(DEFAULT_METRICS),
        help="Comma-separated Panel A metrics to run. Choices: MAE,RMSE,R2,r,n",
    )
    p.add_argument(
        "--cv-evaluation",
        default="OOF_BIAS_CORRECTED",
        help="Preferred evaluation row in CV summaries.",
    )
    p.add_argument(
        "--formats",
        default="png,pdf",
        help="Comma-separated output formats.",
    )
    p.add_argument(
        "--dpi",
        type=int,
        default=450,
        help="Output DPI.",
    )
    p.add_argument(
        "--max-endpoints-main",
        type=int,
        default=10,
        help="Max endpoints in main Figure 7 forest/summary panels.",
    )
    p.add_argument(
        "--max-endpoints-supp",
        type=int,
        default=30,
        help="Max endpoints in supplementary Figure 7 panel.",
    )
    p.add_argument(
        "--endpoint-order",
        default="replication",
        choices=["replication", "effect", "original"],
        help="Endpoint ordering for plots.",
    )
    p.add_argument(
        "--endpoint-mapping-mode",
        default="strict",
        choices=["strict", "fallback"],
        help="Strict uses curated cohort-specific endpoint columns.",
    )
    p.add_argument(
        "--show-missing-cohort-markers",
        type=int,
        choices=[0, 1],
        default=1,
        help="Show gray x markers for unavailable/invalid cohort-endpoint pairs.",
    )
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip metric run if its manifest already exists.",
    )
    p.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Do not run Figure 7 scripts; only aggregate existing metric directories.",
    )
    return p.parse_args()


def set_thread_env() -> dict:
    env = os.environ.copy()
    for key in [
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    ]:
        env[key] = "1"
    return env


def resolve_script_path(code_dir: Path, script_arg: str) -> Path:
    p = Path(script_arg)
    if p.is_absolute():
        return p
    return code_dir / p


def run_one_metric(
    metric: str,
    args: argparse.Namespace,
    script_path: Path,
    code_dir: Path,
    base_dir: Path,
    env: dict,
) -> dict:
    outdir = base_dir / f"Figure7_cross_cohort_replication_{metric}"
    outdir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = code_dir / f"Figure7_cross_cohort_OOF_{metric}_{timestamp}.log"

    manifest = outdir / "figure7_manifest.csv"
    if args.skip_existing and manifest.exists():
        return {
            "metric": metric,
            "status": "skipped_existing",
            "returncode": 0,
            "outdir": str(outdir),
            "log": str(log_path),
        }

    cmd: List[str] = [
        sys.executable,
        str(script_path),
        "--base-dir",
        str(base_dir),
        "--outdir",
        str(outdir),
        "--main-feature-set",
        args.main_feature_set,
        "--feature-sets",
        args.feature_sets,
        "--cohorts",
        args.cohorts,
        "--metric",
        metric,
        "--cv-evaluation",
        args.cv_evaluation,
        "--formats",
        args.formats,
        "--dpi",
        str(args.dpi),
        "--max-endpoints-main",
        str(args.max_endpoints_main),
        "--max-endpoints-supp",
        str(args.max_endpoints_supp),
        "--endpoint-order",
        args.endpoint_order,
        "--endpoint-mapping-mode",
        args.endpoint_mapping_mode,
        "--show-missing-cohort-markers",
        str(args.show_missing_cohort_markers),
    ]

    if args.results_root:
        cmd.extend(["--results-root", args.results_root])

    print("\n" + "=" * 100)
    print(f"[RUN] Figure 7 metric={metric}")
    print(f"[OUTDIR] {outdir}")
    print(f"[LOG] {log_path}")
    print("=" * 100)

    with open(log_path, "w") as log:
        proc = subprocess.run(
            cmd,
            cwd=str(code_dir),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )

    status = "ok" if proc.returncode == 0 else "failed"
    print(f"[DONE] metric={metric} status={status} returncode={proc.returncode}")
    if proc.returncode != 0:
        print(f"[ERROR] See log: {log_path}")

    return {
        "metric": metric,
        "status": status,
        "returncode": proc.returncode,
        "outdir": str(outdir),
        "log": str(log_path),
    }


def aggregate_tables(base_dir: Path, metrics: List[str]) -> Path:
    outdir = base_dir / "Figure7_cross_cohort_replication_ALL_METRICS"
    outdir.mkdir(parents=True, exist_ok=True)

    for table in TABLES_TO_AGGREGATE:
        frames = []
        for metric in metrics:
            metric_dir = base_dir / f"Figure7_cross_cohort_replication_{metric}"
            path = metric_dir / table
            if not path.exists():
                print(f"[WARN] Missing table for aggregation: {path}")
                continue
            df = pd.read_csv(path)
            df.insert(0, "figure7_metric_run", metric)
            df.insert(1, "source_dir", str(metric_dir))
            frames.append(df)

        if frames:
            combined = pd.concat(frames, ignore_index=True, sort=False)
            outpath = outdir / table.replace(".csv", "_all_metrics.csv")
            combined.to_csv(outpath, index=False)
            print(f"[AGG] wrote {outpath} shape={combined.shape}")

    # Create a compact run inventory.
    rows = []
    for metric in metrics:
        metric_dir = base_dir / f"Figure7_cross_cohort_replication_{metric}"
        rows.append({
            "metric": metric,
            "outdir": str(metric_dir),
            "exists": metric_dir.exists(),
            "figure_png": str(metric_dir / "Figure7_cross_cohort_replication.png"),
            "figure_pdf": str(metric_dir / "Figure7_cross_cohort_replication.pdf"),
            "supp_png": str(metric_dir / "Supplementary_Figure7_cross_cohort_replication.png"),
            "supp_pdf": str(metric_dir / "Supplementary_Figure7_cross_cohort_replication.pdf"),
            "manifest": str(metric_dir / "figure7_manifest.csv"),
        })
    inventory = pd.DataFrame(rows)
    inventory_path = outdir / "figure7_all_metrics_output_inventory.csv"
    inventory.to_csv(inventory_path, index=False)
    print(f"[AGG] wrote {inventory_path} shape={inventory.shape}")

    return outdir


def main() -> None:
    args = parse_args()

    code_dir = Path(args.code_dir).expanduser().resolve()
    base_dir = Path(args.base_dir).expanduser().resolve()
    script_path = resolve_script_path(code_dir, args.figure7_script).expanduser().resolve()
    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]

    if not script_path.exists():
        raise SystemExit(f"Figure 7 script not found: {script_path}")

    print("Figure 7 all-metrics wrapper")
    print(f"CODE_DIR:       {code_dir}")
    print(f"BASE_DIR:       {base_dir}")
    print(f"FIGURE7_SCRIPT: {script_path}")
    print(f"METRICS:        {', '.join(metrics)}")
    print(f"AGGREGATE_ONLY: {args.aggregate_only}")

    env = set_thread_env()

    run_rows = []
    if not args.aggregate_only:
        for metric in metrics:
            run_rows.append(run_one_metric(metric, args, script_path, code_dir, base_dir, env))

        run_summary = pd.DataFrame(run_rows)
        run_summary_path = code_dir / f"Figure7_all_metrics_run_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        run_summary.to_csv(run_summary_path, index=False)
        print(f"\n[RUN SUMMARY] {run_summary_path}")
        print(run_summary.to_string(index=False))

        failed = run_summary[run_summary["returncode"] != 0]
        if not failed.empty:
            print("\n[WARN] One or more metric runs failed. Aggregating available outputs anyway.")
            print(failed.to_string(index=False))

    all_metrics_dir = aggregate_tables(base_dir, metrics)

    print("\n" + "=" * 100)
    print("[DONE] Figure 7 all-metrics run/aggregation complete.")
    print(f"ALL_METRICS_DIR: {all_metrics_dir}")
    print("=" * 100)


if __name__ == "__main__":
    main()
