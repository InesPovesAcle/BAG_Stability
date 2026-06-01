#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Audit Figure 6 SHAP recomputation outputs.

Checks:
  - Figure6_SHAP output folder exists
  - cohort/model folders exist
  - counts of subject-level global/node/edge SHAP CSVs
  - presence of summary CSVs
  - shap_run_summary.csv status
  - traceback/error files
  - final main/supplementary figures
  - cross-cohort aggregated outputs
  - newest files

Usage:
  python audit_figure6_shap_results.py

Optional:
  python audit_figure6_shap_results.py --out-base /path/to/Figure6_SHAP
  python audit_figure6_shap_results.py --cohorts ADNI,ADRC,HABS,AD_DECODE
  python audit_figure6_shap_results.py --models imaging_only,full
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from datetime import datetime
import pandas as pd


DEFAULT_WORK = os.environ.get("WORK", "/mnt/newStor/paros/paros_WORK")
DEFAULT_OUT_BASE = (
    Path(DEFAULT_WORK)
    / "ines/results"
    / "BrainAgeValidation_AllCohorts_BAGBiasCorr_OOFGlobal_BiologicalValidation"
    / "Figure6_SHAP"
)

COHORT_SLUGS = {
    "ADNI": "adni",
    "ADRC": "adrc",
    "HABS": "habs",
    "AD_DECODE": "addecode",
}

DEFAULT_COHORTS = ["ADNI", "ADRC", "HABS", "AD_DECODE"]
DEFAULT_MODELS = [
    "imaging_only",
    "imaging_demographics",
    "imaging_biomarkers",
    "full",
    "full_no_cardiovascular",
]


def parse_args():
    p = argparse.ArgumentParser(description="Audit Figure 6 SHAP outputs.")
    p.add_argument("--out-base", default=str(DEFAULT_OUT_BASE))
    p.add_argument("--cohorts", default=",".join(DEFAULT_COHORTS))
    p.add_argument("--models", "--feature-sets", dest="models", default=",".join(DEFAULT_MODELS))
    p.add_argument("--recent-n", type=int, default=40)
    return p.parse_args()


def count_glob(path: Path, pattern: str) -> int:
    if not path.exists():
        return 0
    return sum(1 for _ in path.glob(pattern))


def exists(path: Path) -> bool:
    return path.exists()


def file_size(path: Path) -> int:
    return path.stat().st_size if path.exists() else 0


def mtime_str(path: Path) -> str:
    if not path.exists():
        return ""
    return datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")


def read_status_summary(path: Path) -> dict:
    out = {
        "run_summary_exists": path.exists(),
        "run_status": "",
        "n_graphs": "",
        "global_ok_exists": "",
        "node_ok_exists": "",
        "edge_ok_exists": "",
        "run_error": "",
    }
    if not path.exists():
        return out
    try:
        df = pd.read_csv(path)
        if df.empty:
            out["run_status"] = "EMPTY"
            return out
        row = df.iloc[0]
        out["run_status"] = str(row.get("status", ""))
        out["n_graphs"] = row.get("n_graphs", "")
        out["global_ok_exists"] = row.get("global_status_ok_or_exists", "")
        out["node_ok_exists"] = row.get("node_status_ok_or_exists", "")
        out["edge_ok_exists"] = row.get("edge_status_ok_or_exists", "")
        out["run_error"] = str(row.get("error", ""))
    except Exception as e:
        out["run_status"] = f"READ_FAILED: {e}"
    return out


def audit_one(out_base: Path, cohort: str, model: str) -> dict:
    slug = COHORT_SLUGS.get(cohort, cohort.lower())
    root = out_base / slug / model
    global_dir = root / "global_feature_shap"
    node_dir = root / "node_feature_shap"
    edge_dir = root / "edge_shap"
    region_dir = root / "region_from_edge_shap"
    plots_dir = root / "plots"
    cluster_dir = root / "hierarchical_clustering"

    row = {
        "cohort": cohort,
        "slug": slug,
        "model": model,
        "root_exists": root.exists(),
        "root": str(root),
        "global_subject_csvs": count_glob(global_dir, "global_feature_shap_subject_*.csv"),
        "node_subject_csvs": count_glob(node_dir, "node_feature_shap_subject_*.csv"),
        "edge_subject_csvs": count_glob(edge_dir, "edge_shap_subject_*.csv"),
        "global_summary": exists(global_dir / "global_feature_shap_summary_all_subjects.csv"),
        "node_summary": exists(node_dir / "node_feature_shap_summary_all_subjects.csv"),
        "edge_summary": exists(edge_dir / "edge_shap_summary_all_subjects.csv"),
        "region_summary": exists(region_dir / "region_from_edge_shap_summary.csv"),
        "input_subject_summary": exists(root / "input_shap_summary_all_subjects.csv"),
        "top_compact": exists(root / "top_shap_features_compact.csv"),
        "plot_log": exists(root / "shap_plot_log.csv"),
        "n_plot_pngs": count_glob(plots_dir, "*.png") + count_glob(plots_dir / "composite", "*.png"),
        "n_cluster_pngs": count_glob(cluster_dir, "*.png"),
        "traceback_exists": exists(root / "shap_error_traceback.txt"),
        "traceback_size": file_size(root / "shap_error_traceback.txt"),
        "last_modified": mtime_str(root) if root.exists() else "",
    }

    row.update(read_status_summary(root / "shap_run_summary.csv"))

    # Basic decision.
    problems = []
    if not row["root_exists"]:
        problems.append("missing_root")
    if row["traceback_exists"] and row["traceback_size"] > 0:
        problems.append("traceback")
    if str(row["run_status"]).lower() not in ["ok", ""]:
        problems.append(f"run_status={row['run_status']}")
    if row["global_subject_csvs"] == 0:
        problems.append("no_global_subject_csvs")
    if row["node_subject_csvs"] == 0:
        problems.append("no_node_subject_csvs")
    if row["edge_subject_csvs"] == 0:
        problems.append("no_edge_subject_csvs")
    if not row["global_summary"]:
        problems.append("missing_global_summary")
    if not row["node_summary"]:
        problems.append("missing_node_summary")
    if not row["edge_summary"]:
        problems.append("missing_edge_summary")

    row["audit_status"] = "PASS" if not problems else "CHECK"
    row["problems"] = ";".join(problems)
    return row


def list_recent_files(out_base: Path, n: int) -> pd.DataFrame:
    rows = []
    if not out_base.exists():
        return pd.DataFrame()
    for p in out_base.rglob("*"):
        if p.is_file():
            st = p.stat()
            rows.append({
                "modified": datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
                "size": st.st_size,
                "path": str(p),
            })
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).sort_values("modified").tail(n)
    return df


def audit_final_figures(out_base: Path) -> pd.DataFrame:
    candidates = [
        out_base / "Figure6_main_SHAP_interpretability.png",
        out_base / "Figure6_main_SHAP_interpretability.pdf",
        out_base / "Supplementary_Figure6_SHAP_ablation_summary.png",
        out_base / "Supplementary_Figure6_SHAP_ablation_summary.pdf",
        out_base / "Figure6_SHAP_figure_manifest.csv",
        out_base / "combined_shap_run_summary.csv",
        out_base / "combined_shap_run_summary.xlsx",
        out_base / "Figure6_global_SHAP_combined_summary.csv",
        out_base / "Figure6_node_SHAP_combined_summary.csv",
        out_base / "Supplementary_Figure6_global_SHAP_all_ablation_summary.csv",
        out_base / "Supplementary_Figure6_node_SHAP_all_ablation_summary.csv",
    ]
    rows = []
    for p in candidates:
        rows.append({
            "file": p.name,
            "exists": p.exists(),
            "size": file_size(p),
            "modified": mtime_str(p),
            "path": str(p),
        })
    return pd.DataFrame(rows)


def audit_cross_cohort(out_base: Path) -> pd.DataFrame:
    root = out_base / "cross_cohort_aggregated"
    rows = []
    if not root.exists():
        return pd.DataFrame([{
            "kind": "cross_cohort_aggregated",
            "exists": False,
            "count": 0,
            "path": str(root),
        }])

    patterns = {
        "png": "*.png",
        "pdf": "*.pdf",
        "csv": "*.csv",
        "manifest": "*manifest*.csv",
        "final_figures_png": "final_figures/*.png",
        "final_figures_pdf": "final_figures/*.pdf",
    }
    for kind, pattern in patterns.items():
        files = list(root.rglob(pattern)) if "/" not in pattern else list(root.glob(pattern))
        rows.append({
            "kind": kind,
            "exists": True,
            "count": len(files),
            "path": str(root),
        })
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    out_base = Path(args.out_base).expanduser().resolve()
    cohorts = [x.strip() for x in args.cohorts.split(",") if x.strip()]
    models = [x.strip() for x in args.models.split(",") if x.strip()]

    print("=" * 100)
    print("FIGURE 6 SHAP AUDIT")
    print("=" * 100)
    print(f"OUT_BASE: {out_base}")
    print(f"Exists:   {out_base.exists()}")
    print(f"Cohorts:  {', '.join(cohorts)}")
    print(f"Models:   {', '.join(models)}")
    print("=" * 100)

    audit_rows = []
    for cohort in cohorts:
        for model in models:
            audit_rows.append(audit_one(out_base, cohort, model))

    audit_df = pd.DataFrame(audit_rows)
    audit_out = out_base / "Figure6_AUDIT_cohort_model_status.csv"
    audit_df.to_csv(audit_out, index=False)

    print("\nCOHORT x MODEL STATUS")
    cols = [
        "cohort", "model", "audit_status", "run_status", "n_graphs",
        "global_subject_csvs", "node_subject_csvs", "edge_subject_csvs",
        "global_summary", "node_summary", "edge_summary", "traceback_exists", "problems"
    ]
    print(audit_df[cols].to_string(index=False))

    print("\nSTATUS COUNTS")
    print(audit_df["audit_status"].value_counts(dropna=False).to_string())

    check_df = audit_df[audit_df["audit_status"] != "PASS"].copy()
    if not check_df.empty:
        print("\nITEMS NEEDING CHECK")
        print(check_df[["cohort", "model", "problems", "root"]].to_string(index=False))

    final_df = audit_final_figures(out_base)
    final_out = out_base / "Figure6_AUDIT_final_figures.csv"
    final_df.to_csv(final_out, index=False)

    print("\nFINAL FIGURES / SUMMARY FILES")
    print(final_df[["file", "exists", "size", "modified"]].to_string(index=False))

    cross_df = audit_cross_cohort(out_base)
    cross_out = out_base / "Figure6_AUDIT_cross_cohort_aggregated.csv"
    cross_df.to_csv(cross_out, index=False)

    print("\nCROSS-COHORT AGGREGATED")
    print(cross_df.to_string(index=False))

    recent_df = list_recent_files(out_base, args.recent_n)
    recent_out = out_base / "Figure6_AUDIT_recent_files.csv"
    recent_df.to_csv(recent_out, index=False)

    print(f"\nRECENT FILES saved to: {recent_out}")
    if not recent_df.empty:
        print(recent_df.tail(20).to_string(index=False))

    print("\nAUDIT FILES WRITTEN")
    print(f"  {audit_out}")
    print(f"  {final_out}")
    print(f"  {cross_out}")
    print(f"  {recent_out}")

    print("\nDone.")


if __name__ == "__main__":
    main()
