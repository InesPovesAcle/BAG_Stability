#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure 6 OOF SHAP pipeline wrapper
==================================

Purpose
-------
Run the Figure 6 SHAP workflow safely from the command line.

Recommended default is aggregation/figure rebuild from existing SHAP CSVs:
    python Figure6_OOF_SHAP_pipeline.py \
        --build-script Figure6_build.py \
        --aggregate-script FIgure6_aggregateC.py \
        --mode aggregate-only

This wrapper keeps SHAP interpretation tied to the OOF-global model/checkpoint
outputs. It does not use validation_figures_full_cohort, because Figure 6 is an
interpretability analysis of the trained OOF-global models rather than a
full-cohort validation association figure.

Modes
-----
aggregate-only
    Do not recompute SHAP. Run Figure6_build.py with --skip-computation 1 to
    rebuild its local figures from existing CSVs, then run the cross-cohort
    aggregation script.

compute-and-aggregate
    Recompute SHAP first, then aggregate. Use only when raw SHAP CSVs are missing
    or stale. This can be slow.

aggregate-master-only
    Skip Figure6_build.py entirely and run only FIgure6_aggregateC.py. Use this
    when per-cohort/per-model SHAP CSVs already exist and you only want the final
    cross-cohort figures/tables.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path


DEFAULT_WORK = os.environ.get("WORK", "/mnt/newStor/paros/paros_WORK")
DEFAULT_OUT_BASE = (
    Path(DEFAULT_WORK)
    / "ines/results"
    / "BrainAgeValidation_AllCohorts_BAGBiasCorr_OOFGlobal_BiologicalValidation"
    / "Figure6_SHAP"
)

DEFAULT_COHORTS = "ADNI,ADRC,HABS,AD_DECODE"
DEFAULT_FEATURE_SETS = "imaging_only,imaging_demographics,imaging_biomarkers,full,full_no_cardiovascular"
DEFAULT_MAIN_MODEL = "imaging_only"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Figure 6 OOF SHAP build/aggregation workflow.")
    p.add_argument("--mode", choices=["aggregate-only", "compute-and-aggregate", "aggregate-master-only"], default="aggregate-only")
    p.add_argument("--build-script", default="Figure6_build.py", help="Path to Figure6_build.py")
    p.add_argument("--aggregate-script", default="FIgure6_aggregateC.py", help="Path to FIgure6_aggregateC.py master aggregation script")
    p.add_argument("--out-base", default=str(DEFAULT_OUT_BASE))
    p.add_argument("--cohorts", default=DEFAULT_COHORTS)
    p.add_argument("--feature-sets", "--models", dest="feature_sets", default=DEFAULT_FEATURE_SETS)
    p.add_argument("--main-model", default=DEFAULT_MAIN_MODEL)
    p.add_argument("--n-jobs", type=int, default=4, help="Only used when recomputing SHAP")
    p.add_argument("--max-subjects", default="None", help="Only used when recomputing SHAP; use None for all")
    p.add_argument("--run-global-shap", type=int, choices=[0, 1], default=1)
    p.add_argument("--run-node-shap", type=int, choices=[0, 1], default=1)
    p.add_argument("--run-edge-shap", type=int, choices=[0, 1], default=1)
    p.add_argument("--formats", default="png,pdf")
    p.add_argument("--dpi", type=int, default=450)
    p.add_argument("--skip-beeswarm", action="store_true")
    p.add_argument("--skip-final-assembly", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def resolve_script(path_like: str) -> Path:
    p = Path(path_like).expanduser()
    if p.exists():
        return p.resolve()
    cwd_candidate = Path.cwd() / path_like
    if cwd_candidate.exists():
        return cwd_candidate.resolve()
    raise FileNotFoundError(f"Script not found: {path_like}")


def run_cmd(cmd: list[str], dry_run: bool = False) -> None:
    print("\n" + "=" * 100)
    print("RUN:", " ".join(shlex.quote(x) for x in cmd))
    print("=" * 100)
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()

    out_base = str(Path(args.out_base).expanduser())
    build_script = resolve_script(args.build_script)
    aggregate_script = resolve_script(args.aggregate_script)

    # Keep hidden thread libraries capped. SHAP parallelism should be controlled by --n-jobs.
    thread_caps = {
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "BLIS_NUM_THREADS": "1",
    }
    for k, v in thread_caps.items():
        os.environ.setdefault(k, v)

    print("Figure 6 OOF SHAP pipeline")
    print("mode:", args.mode)
    print("out_base:", out_base)
    print("cohorts:", args.cohorts)
    print("feature_sets:", args.feature_sets)
    print("main_model:", args.main_model)
    print("thread caps:", {k: os.environ.get(k) for k in thread_caps})

    if args.mode in {"aggregate-only", "compute-and-aggregate"}:
        skip_computation = "0" if args.mode == "compute-and-aggregate" else "1"
        build_cmd = [
            sys.executable,
            str(build_script),
            "--cohorts", args.cohorts,
            "--feature-sets", args.feature_sets,
            "--main-feature-set", args.main_model,
            "--out-base", out_base,
            "--skip-computation", skip_computation,
            "--formats", args.formats,
            "--dpi", str(args.dpi),
            "--n-jobs", str(args.n_jobs),
            "--run-global-shap", str(args.run_global_shap),
            "--run-node-shap", str(args.run_node_shap),
            "--run-edge-shap", str(args.run_edge_shap),
        ]
        if args.mode == "compute-and-aggregate":
            build_cmd.extend(["--max-subjects", str(args.max_subjects)])
        run_cmd(build_cmd, dry_run=args.dry_run)

    agg_cmd = [
        sys.executable,
        str(aggregate_script),
        "--out-base", out_base,
        "--cohorts", args.cohorts,
        "--models", args.feature_sets,
        "--main-model", args.main_model,
        "--dpi", str(args.dpi),
    ]
    if args.skip_beeswarm:
        agg_cmd.append("--skip-beeswarm")
    if args.skip_final_assembly:
        agg_cmd.append("--skip-final-assembly")
    run_cmd(agg_cmd, dry_run=args.dry_run)

    print("\n[DONE] Figure 6 OOF SHAP pipeline finished.")
    print("Check outputs under:")
    print(Path(out_base) / "cross_cohort_aggregated")


if __name__ == "__main__":
    main()
