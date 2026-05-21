#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aggregate Figure 6 SHAP outputs without rerunning SHAP computation
==================================================================

Purpose
-------
This standalone script reads already-saved Figure 6 SHAP CSV/matrix outputs and
creates cross-cohort aggregate figures.

It does NOT recompute SHAP.
It does NOT load the GNN model.
It does NOT rerun the slow Figure6_build_shap.py computation.

Outputs
-------
Main aggregate Figure 6 candidate:
    cross_cohort_aggregated/Figure6_cross_cohort_<feature_set>_graph_SHAP_heatmap.png/pdf

Supplementary aggregate Figure 6 candidate:
    cross_cohort_aggregated/Supplementary_Figure6_cross_cohort_<feature_set>_node_feature_SHAP_heatmap.png/pdf

Default input root
------------------
/mnt/newStor/paros/paros_WORK/ines/results/
BrainAgeValidation_AllCohorts_BAGBiasCorr_OOFGlobal_BiologicalValidation/Figure6_SHAP

Run
---
    python aggregate_figure6.py

Typical full aggregation:
    python aggregate_figure6.py \
        --feature-set imaging_only \
        --make-graph 1 \
        --make-node-supplement 1 \
        --formats png,pdf

Only graph-level main aggregate:
    python aggregate_figure6.py --make-node-supplement 0

Only supplementary node-feature aggregate:
    python aggregate_figure6.py --make-graph 0 --make-node-supplement 1
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# =============================================================================
# Defaults
# =============================================================================
WORK = os.environ.get("WORK", "/mnt/newStor/paros/paros_WORK")
RESULTS_ROOT = os.path.join(WORK, "ines/results")
VALIDATION_BASE = os.path.join(
    RESULTS_ROOT,
    "BrainAgeValidation_AllCohorts_BAGBiasCorr_OOFGlobal_BiologicalValidation",
)
OUT_BASE = os.path.join(VALIDATION_BASE, "Figure6_SHAP")

DEFAULT_COHORTS = ["ADNI", "ADRC", "HABS", "AD_DECODE"]
DEFAULT_FEATURE_SET = "imaging_only"
DEFAULT_FORMATS = ["png", "pdf"]

COHORT_CONFIG = {
    "ADNI": {
        "cohort_slug": "adni",
        "file_prefix": "adni",
        "display": "ADNI",
        "color": "#4C78A8",
    },
    "ADRC": {
        "cohort_slug": "adrc",
        "file_prefix": "adrc",
        "display": "ADRC",
        "color": "#F58518",
    },
    "HABS": {
        "cohort_slug": "habs",
        "file_prefix": "habs",
        "display": "HABS",
        "color": "#54A24B",
    },
    "AD_DECODE": {
        "cohort_slug": "addecode",
        "file_prefix": "ad_decode",
        "display": "AD-DECODE",
        "color": "#E45756",
    },
}

GRAPH_FEATURE_RENAME = {
    "Local_Efficiency": "Local efficiency",
    "Global_Efficiency": "Global efficiency",
    "Clustering_Coeff": "Clustering coefficient",
    "Path_Length": "Path length",
    "local_efficiency": "Local efficiency",
    "global_efficiency": "Global efficiency",
    "clustering_coeff": "Clustering coefficient",
    "path_length": "Path length",
}


# =============================================================================
# CLI
# =============================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate existing Figure 6 SHAP outputs into cross-cohort main and supplementary figures."
    )
    parser.add_argument(
        "--out-base",
        default=OUT_BASE,
        help="Figure6_SHAP output directory from Figure6_build_shap.py.",
    )
    parser.add_argument("--feature-set", default=DEFAULT_FEATURE_SET, help="Feature set to aggregate. Default: imaging_only")
    parser.add_argument("--cohorts", default=",".join(DEFAULT_COHORTS), help="Comma-separated cohorts.")
    parser.add_argument("--formats", default=",".join(DEFAULT_FORMATS), help="Comma-separated output formats.")
    parser.add_argument("--dpi", type=int, default=450)
    parser.add_argument("--make-graph", type=int, choices=[0, 1], default=1, help="Make cross-cohort graph-level SHAP heatmap.")
    parser.add_argument("--make-node-supplement", type=int, choices=[0, 1], default=1, help="Make supplementary cross-cohort node-feature SHAP heatmap.")
    parser.add_argument("--top-n-graph-features", type=int, default=None, help="Optional top-N graph/global features; usually leave None.")
    parser.add_argument("--top-n-node-features", type=int, default=50, help="Top node-feature labels for supplementary aggregate. Default: 50")
    parser.add_argument("--max-subjects-per-cohort-node", type=int, default=None, help="Optional cap for supplementary node matrix reading. Default: all available subjects.")
    parser.add_argument("--no-standardize", action="store_true", help="Plot raw SHAP values instead of z-scoring each column after concatenation.")
    parser.add_argument("--graph-fig-width", type=float, default=9.5)
    parser.add_argument("--graph-fig-height", type=float, default=13.5)
    parser.add_argument("--node-fig-width", type=float, default=18.0)
    parser.add_argument("--node-fig-height", type=float, default=14.0)
    parser.add_argument("--graph-title", default=None)
    parser.add_argument("--node-title", default=None)
    parser.add_argument("--also-aggregate-summary-csvs", type=int, choices=[0, 1], default=1)
    return parser.parse_args()


# =============================================================================
# Helpers
# =============================================================================
def ensure_dir(path: str | Path) -> str:
    path = str(path)
    os.makedirs(path, exist_ok=True)
    return path


def safe_name(x: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", str(x))


def zscore_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().astype(float)
    mu = out.mean(axis=0)
    sd = out.std(axis=0, ddof=0).replace(0, np.nan)
    out = (out - mu) / sd
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def clean_feature_label(x: str) -> str:
    raw = str(x).strip()
    raw = GRAPH_FEATURE_RENAME.get(raw, raw)
    raw = raw.replace("_", " ")
    raw = re.sub(r"\s+", " ", raw).strip()
    return raw


def robust_vmax(values: np.ndarray, percentile: float = 99.0) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.nan
    vmax = np.nanpercentile(np.abs(finite), percentile)
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = np.nanmax(np.abs(finite))
    return float(vmax) if np.isfinite(vmax) and vmax > 0 else np.nan


def find_existing_path(candidates: Sequence[Path]) -> Optional[Path]:
    for p in candidates:
        if p.exists():
            return p
    return None


def prefixes_for_cohort(cohort: str) -> List[str]:
    cfg = COHORT_CONFIG[cohort]
    return list(dict.fromkeys([
        cohort.lower(),
        cohort.lower().replace("_", ""),
        cfg["cohort_slug"],
        cfg["file_prefix"],
    ]))


def save_formats(fig_or_grid, outbase: Path, formats: Sequence[str], dpi: int) -> List[str]:
    saved: List[str] = []
    for fmt in formats:
        fmt = fmt.strip().lower().lstrip(".")
        if not fmt:
            continue
        path = outbase.with_suffix(f".{fmt}")
        fig_or_grid.savefig(path, dpi=dpi, bbox_inches="tight")
        saved.append(str(path))
    return saved


# =============================================================================
# Input readers: graph matrices and summary CSVs
# =============================================================================
def graph_matrix_candidates(out_base: Path, cohort: str, feature_set: str) -> List[Path]:
    cfg = COHORT_CONFIG[cohort]
    cluster_dir = out_base / cfg["cohort_slug"] / feature_set / "hierarchical_clustering"
    candidates: List[Path] = []
    for pref in prefixes_for_cohort(cohort):
        candidates.extend([
            cluster_dir / f"{pref}_{feature_set}_global_shap_graph_hierarchical_heatmap_matrix_raw.csv",
            cluster_dir / f"{pref}_{feature_set}_global_feature_shap_hierarchical_heatmap_matrix_raw.csv",
        ])
    return candidates


def read_graph_category_matrix(out_base: Path, cohort: str, feature_set: str) -> Optional[pd.DataFrame]:
    candidates = graph_matrix_candidates(out_base, cohort, feature_set)
    path = find_existing_path(candidates)

    if path is None:
        print(f"[WARN] No graph/global SHAP matrix found for {cohort} {feature_set}")
        for p in candidates:
            print(f"       tried: {p}")
        return None

    print(f"[INFO] Reading {cohort} graph matrix: {path}")
    df = pd.read_csv(path, index_col=0)
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna(axis=1, how="all")
    df.columns = [clean_feature_label(c) for c in df.columns]
    df.index = [f"{cohort}__{safe_name(i)}" for i in df.index]
    return df


def read_global_summary_csv(out_base: Path, cohort: str, feature_set: str) -> Optional[pd.DataFrame]:
    cfg = COHORT_CONFIG[cohort]
    path = out_base / cfg["cohort_slug"] / feature_set / "global_feature_shap" / "global_feature_shap_summary_all_subjects.csv"
    if not path.exists():
        print(f"[WARN] No global SHAP summary CSV found for {cohort} {feature_set}: {path}")
        return None
    df = pd.read_csv(path)
    if df.empty:
        return None
    df.insert(0, "cohort", cohort)
    df.insert(1, "feature_set", feature_set)
    if "feature_name" in df.columns:
        df["feature_name_clean"] = df["feature_name"].map(clean_feature_label)
    return df


def read_node_summary_csv(out_base: Path, cohort: str, feature_set: str) -> Optional[pd.DataFrame]:
    cfg = COHORT_CONFIG[cohort]
    path = out_base / cfg["cohort_slug"] / feature_set / "node_feature_shap" / "node_feature_shap_summary_all_subjects.csv"
    if not path.exists():
        print(f"[WARN] No node SHAP summary CSV found for {cohort} {feature_set}: {path}")
        return None
    df = pd.read_csv(path)
    if df.empty:
        return None
    df.insert(0, "cohort", cohort)
    df.insert(1, "feature_set", feature_set)
    if {"node_label", "feature_name"}.issubset(df.columns):
        df["node_feature_label"] = df["node_label"].astype(str) + " | " + df["feature_name"].astype(str)
    elif "node_feature_label" not in df.columns:
        return None
    return df


def node_subject_file_dir(out_base: Path, cohort: str, feature_set: str) -> Path:
    cfg = COHORT_CONFIG[cohort]
    return out_base / cfg["cohort_slug"] / feature_set / "node_feature_shap"


# =============================================================================
# Shared output helpers
# =============================================================================
def save_subject_order_outputs(
    ordered_index: Sequence[str],
    cohort_series: pd.Series,
    outdir: Path,
    feature_set: str,
    prefix: str,
) -> Tuple[str, str]:
    ordered = pd.DataFrame({
        "subject": list(ordered_index),
        "cohort": cohort_series.loc[list(ordered_index)].values,
        "cluster_order": np.arange(len(ordered_index)),
    })

    if len(ordered) >= 4:
        ordered["cluster_order_quartile"] = pd.qcut(
            ordered["cluster_order"],
            q=4,
            labels=["Q1", "Q2", "Q3", "Q4"],
            duplicates="drop",
        )
    else:
        ordered["cluster_order_quartile"] = "all"

    composition = (
        ordered.groupby(["cluster_order_quartile", "cohort"], observed=False)
        .size()
        .reset_index(name="n")
    )

    ordered_path = outdir / f"{prefix}_{feature_set}_subject_order.csv"
    composition_path = outdir / f"{prefix}_{feature_set}_cluster_cohort_composition.csv"
    ordered.to_csv(ordered_path, index=False)
    composition.to_csv(composition_path, index=False)
    return str(ordered_path), str(composition_path)


def add_cohort_legend(cg, cohorts: Sequence[str], cohort_series: pd.Series) -> None:
    palette = {c: COHORT_CONFIG[c]["color"] for c in cohorts if c in COHORT_CONFIG}
    for cohort, color in palette.items():
        if cohort in set(cohort_series):
            cg.ax_col_dendrogram.bar(0, 0, color=color, label=COHORT_CONFIG[cohort]["display"], linewidth=0)
    cg.ax_col_dendrogram.legend(
        title="Cohort",
        loc="center",
        ncol=max(1, min(4, len(palette))),
        frameon=False,
        fontsize=8,
        title_fontsize=9,
    )


# =============================================================================
# Main aggregate: graph-level SHAP
# =============================================================================
def make_cross_cohort_graph_shap_heatmap(
    out_base: Path,
    cohorts: Sequence[str],
    feature_set: str,
    formats: Sequence[str],
    dpi: int = 450,
    top_n_features: Optional[int] = None,
    standardize: bool = True,
    figsize: Tuple[float, float] = (9.5, 13.5),
    title: Optional[str] = None,
) -> List[str]:
    outdir = Path(ensure_dir(out_base / "cross_cohort_aggregated"))

    frames: List[pd.DataFrame] = []
    labels: List[str] = []
    load_rows: List[Dict[str, object]] = []

    for cohort in cohorts:
        if cohort not in COHORT_CONFIG:
            print(f"[WARN] Unsupported cohort skipped: {cohort}")
            continue
        mat = read_graph_category_matrix(out_base, cohort, feature_set)
        if mat is None or mat.empty:
            load_rows.append({"cohort": cohort, "status": "missing", "n_subjects": 0, "n_features": 0})
            continue
        frames.append(mat)
        labels.extend([cohort] * mat.shape[0])
        load_rows.append({
            "cohort": cohort,
            "status": "loaded",
            "n_subjects": int(mat.shape[0]),
            "n_features": int(mat.shape[1]),
            "features": ";".join(map(str, mat.columns)),
        })

    load_log_path = outdir / f"Figure6_cross_cohort_{feature_set}_graph_SHAP_input_load_log.csv"
    pd.DataFrame(load_rows).to_csv(load_log_path, index=False)

    if not frames:
        print("[ERROR] No graph matrices were loaded.")
        return [str(load_log_path)]

    combined_raw = pd.concat(frames, axis=0, join="inner")
    combined_raw = combined_raw.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")

    if combined_raw.empty or combined_raw.shape[1] < 2:
        print("[ERROR] Combined graph matrix has fewer than two shared features.")
        return [str(load_log_path)]

    if top_n_features is not None and combined_raw.shape[1] > top_n_features:
        keep = combined_raw.abs().mean(axis=0).sort_values(ascending=False).head(top_n_features).index
        combined_raw = combined_raw.loc[:, keep].copy()

    cohort_series = pd.Series(labels, index=combined_raw.index, name="cohort")
    combined_scaled = zscore_columns(combined_raw) if standardize else combined_raw.copy()

    prefix = "Figure6_cross_cohort"
    raw_path = outdir / f"{prefix}_{feature_set}_graph_SHAP_matrix_raw.csv"
    scaled_path = outdir / f"{prefix}_{feature_set}_graph_SHAP_matrix_scaled.csv"
    cohorts_path = outdir / f"{prefix}_{feature_set}_graph_SHAP_subject_cohorts.csv"
    combined_raw.to_csv(raw_path)
    combined_scaled.to_csv(scaled_path)
    cohort_series.to_csv(cohorts_path)

    vmax = robust_vmax(combined_scaled.to_numpy(dtype=float), percentile=99.0)
    if not np.isfinite(vmax):
        print("[ERROR] Combined graph matrix has no finite variation.")
        return [str(raw_path), str(scaled_path), str(cohorts_path), str(load_log_path)]

    palette = {c: COHORT_CONFIG[c]["color"] for c in cohorts if c in COHORT_CONFIG}
    row_colors = cohort_series.map(palette).fillna("#999999")

    cg = sns.clustermap(
        combined_scaled,
        row_cluster=True,
        col_cluster=True,
        method="ward",
        metric="euclidean",
        cmap="coolwarm",
        center=0,
        vmin=-vmax,
        vmax=vmax,
        row_colors=row_colors,
        xticklabels=True,
        yticklabels=False,
        figsize=figsize,
        dendrogram_ratio=(0.14, 0.12),
        cbar_kws={"label": "Scaled SHAP value" if standardize else "SHAP value"},
    )

    cg.fig.suptitle(title or f"Cross-cohort {feature_set}: subject × graph SHAP heatmap", y=1.02, fontsize=16)
    cg.ax_heatmap.set_xlabel("Graph-level SHAP feature", fontsize=12)
    cg.ax_heatmap.set_ylabel("Subjects", fontsize=12)
    cg.ax_heatmap.tick_params(axis="x", labelrotation=90, labelsize=9)
    add_cohort_legend(cg, cohorts, cohort_series)

    row_order = combined_scaled.index[cg.dendrogram_row.reordered_ind]
    col_order = combined_scaled.columns[cg.dendrogram_col.reordered_ind]
    clustered_raw = combined_raw.loc[row_order, col_order]
    clustered_scaled = combined_scaled.loc[row_order, col_order]
    clustered_cohorts = cohort_series.loc[row_order]

    clustered_raw_path = outdir / f"{prefix}_{feature_set}_graph_SHAP_clustered_matrix_raw.csv"
    clustered_scaled_path = outdir / f"{prefix}_{feature_set}_graph_SHAP_clustered_matrix_scaled.csv"
    clustered_cohorts_path = outdir / f"{prefix}_{feature_set}_graph_SHAP_clustered_subject_cohorts.csv"
    clustered_raw.to_csv(clustered_raw_path)
    clustered_scaled.to_csv(clustered_scaled_path)
    clustered_cohorts.to_csv(clustered_cohorts_path)

    saved = save_formats(cg, outdir / f"{prefix}_{feature_set}_graph_SHAP_heatmap", formats, dpi)
    plt.close(cg.fig)

    ordered_path, composition_path = save_subject_order_outputs(row_order, cohort_series, outdir, feature_set, f"{prefix}_graph_SHAP")

    feature_summary = pd.DataFrame({
        "feature": combined_raw.columns,
        "mean_abs_SHAP_raw": combined_raw.abs().mean(axis=0).values,
        "mean_SHAP_raw": combined_raw.mean(axis=0).values,
        "sd_SHAP_raw": combined_raw.std(axis=0, ddof=1).values,
        "mean_abs_scaled_SHAP": combined_scaled.abs().mean(axis=0).values,
    }).sort_values("mean_abs_SHAP_raw", ascending=False)
    feature_summary_path = outdir / f"{prefix}_{feature_set}_graph_SHAP_feature_summary.csv"
    feature_summary.to_csv(feature_summary_path, index=False)

    outputs = [
        *saved,
        str(raw_path),
        str(scaled_path),
        str(clustered_raw_path),
        str(clustered_scaled_path),
        str(cohorts_path),
        str(clustered_cohorts_path),
        ordered_path,
        composition_path,
        str(feature_summary_path),
        str(load_log_path),
    ]

    manifest_path = outdir / f"{prefix}_{feature_set}_graph_SHAP_manifest.csv"
    pd.DataFrame([{
        "figure": "Figure 6 cross-cohort graph SHAP aggregation",
        "feature_set": feature_set,
        "cohorts": ",".join(cohorts),
        "n_subjects": int(combined_raw.shape[0]),
        "n_features": int(combined_raw.shape[1]),
        "outputs": ";".join(outputs),
    }]).to_csv(manifest_path, index=False)
    outputs.append(str(manifest_path))

    print("\nSaved cross-cohort graph SHAP aggregation:")
    for p in outputs:
        print(f"  {p}")
    return outputs


# =============================================================================
# Supplementary aggregate: node-feature SHAP
# =============================================================================
def collect_top_node_features(out_base: Path, cohorts: Sequence[str], feature_set: str, top_n: int) -> Tuple[List[str], pd.DataFrame]:
    frames = []
    for cohort in cohorts:
        if cohort not in COHORT_CONFIG:
            continue
        df = read_node_summary_csv(out_base, cohort, feature_set)
        if df is not None and not df.empty:
            frames.append(df)

    if not frames:
        return [], pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined["mean_abs_SHAP"] = pd.to_numeric(combined.get("mean_abs_SHAP"), errors="coerce")
    ranked = (
        combined.dropna(subset=["mean_abs_SHAP"])
        .groupby("node_feature_label", as_index=False)
        .agg(
            mean_abs_SHAP_across_cohorts=("mean_abs_SHAP", "mean"),
            max_abs_SHAP_across_cohorts=("mean_abs_SHAP", "max"),
            n_cohorts=("cohort", "nunique"),
            cohorts=("cohort", lambda x: ",".join(sorted(set(map(str, x))))),
        )
        .sort_values(["n_cohorts", "mean_abs_SHAP_across_cohorts"], ascending=[False, False])
    )
    keep = ranked["node_feature_label"].head(top_n).tolist()
    return keep, ranked


def build_node_subject_matrix_for_cohort(
    out_base: Path,
    cohort: str,
    feature_set: str,
    keep_features: Sequence[str],
    max_subjects: Optional[int] = None,
) -> Optional[pd.DataFrame]:
    node_dir = node_subject_file_dir(out_base, cohort, feature_set)
    if not node_dir.exists():
        print(f"[WARN] Node subject directory missing: {node_dir}")
        return None

    files = sorted(node_dir.glob("node_feature_shap_subject_*.csv"))
    if max_subjects is not None:
        files = files[:max_subjects]
    if not files:
        print(f"[WARN] No node subject CSVs found in {node_dir}")
        return None

    keep_set = set(keep_features)
    rows = []
    subject_ids = []

    for fpath in files:
        try:
            df = pd.read_csv(fpath)
        except Exception:
            continue
        if df.empty or not {"node_label", "feature_name", "SHAP_val"}.issubset(df.columns):
            continue
        df = df.copy()
        df["node_feature_label"] = df["node_label"].astype(str) + " | " + df["feature_name"].astype(str)
        df = df[df["node_feature_label"].isin(keep_set)].copy()
        if df.empty:
            continue
        vec = df.pivot_table(index=None, columns="node_feature_label", values="SHAP_val", aggfunc="mean")
        if isinstance(vec, pd.DataFrame) and len(vec) > 0:
            series = vec.iloc[0]
        else:
            series = df.groupby("node_feature_label")["SHAP_val"].mean()
        sid = fpath.name.replace("node_feature_shap_subject_", "").replace(".csv", "")
        subject_ids.append(f"{cohort}__{safe_name(sid)}")
        rows.append(series.reindex(keep_features))

    if not rows:
        return None

    mat = pd.DataFrame(rows, index=subject_ids, columns=list(keep_features))
    mat = mat.apply(pd.to_numeric, errors="coerce")
    return mat


def make_supplementary_cross_cohort_node_heatmap(
    out_base: Path,
    cohorts: Sequence[str],
    feature_set: str,
    formats: Sequence[str],
    dpi: int = 450,
    top_n_features: int = 50,
    standardize: bool = True,
    max_subjects_per_cohort: Optional[int] = None,
    figsize: Tuple[float, float] = (18.0, 14.0),
    title: Optional[str] = None,
) -> List[str]:
    outdir = Path(ensure_dir(out_base / "cross_cohort_aggregated"))
    prefix = "Supplementary_Figure6_cross_cohort"

    keep_features, ranked = collect_top_node_features(out_base, cohorts, feature_set, top_n_features)
    ranked_path = outdir / f"{prefix}_{feature_set}_node_feature_SHAP_ranked_features.csv"
    ranked.to_csv(ranked_path, index=False)

    if not keep_features:
        print("[WARN] No node features available for supplementary aggregate.")
        return [str(ranked_path)]

    frames: List[pd.DataFrame] = []
    labels: List[str] = []
    load_rows: List[Dict[str, object]] = []

    for cohort in cohorts:
        mat = build_node_subject_matrix_for_cohort(
            out_base=out_base,
            cohort=cohort,
            feature_set=feature_set,
            keep_features=keep_features,
            max_subjects=max_subjects_per_cohort,
        )
        if mat is None or mat.empty:
            load_rows.append({"cohort": cohort, "status": "missing", "n_subjects": 0, "n_features": len(keep_features)})
            continue
        frames.append(mat)
        labels.extend([cohort] * mat.shape[0])
        load_rows.append({"cohort": cohort, "status": "loaded", "n_subjects": int(mat.shape[0]), "n_features": int(mat.shape[1])})

    load_log_path = outdir / f"{prefix}_{feature_set}_node_feature_SHAP_input_load_log.csv"
    pd.DataFrame(load_rows).to_csv(load_log_path, index=False)

    if not frames:
        print("[WARN] No node subject matrices were loaded.")
        return [str(ranked_path), str(load_log_path)]

    combined_raw = pd.concat(frames, axis=0, join="outer").reindex(columns=keep_features)
    combined_raw = combined_raw.apply(pd.to_numeric, errors="coerce")
    # Missing means feature not found for a subject/cohort among selected top labels; fill with 0 contribution for visualization.
    combined_raw = combined_raw.fillna(0.0)
    combined_raw = combined_raw.loc[:, combined_raw.std(axis=0, ddof=0).fillna(0) > 0]

    if combined_raw.empty or combined_raw.shape[1] < 2:
        print("[WARN] Combined node matrix has fewer than two variable features.")
        return [str(ranked_path), str(load_log_path)]

    cohort_series = pd.Series(labels, index=combined_raw.index, name="cohort")
    combined_scaled = zscore_columns(combined_raw) if standardize else combined_raw.copy()

    raw_path = outdir / f"{prefix}_{feature_set}_node_feature_SHAP_matrix_raw.csv"
    scaled_path = outdir / f"{prefix}_{feature_set}_node_feature_SHAP_matrix_scaled.csv"
    cohorts_path = outdir / f"{prefix}_{feature_set}_node_feature_SHAP_subject_cohorts.csv"
    combined_raw.to_csv(raw_path)
    combined_scaled.to_csv(scaled_path)
    cohort_series.to_csv(cohorts_path)

    vmax = robust_vmax(combined_scaled.to_numpy(dtype=float), percentile=99.0)
    if not np.isfinite(vmax):
        print("[WARN] Combined node matrix has no finite variation.")
        return [str(raw_path), str(scaled_path), str(cohorts_path), str(ranked_path), str(load_log_path)]

    palette = {c: COHORT_CONFIG[c]["color"] for c in cohorts if c in COHORT_CONFIG}
    row_colors = cohort_series.map(palette).fillna("#999999")

    cg = sns.clustermap(
        combined_scaled,
        row_cluster=True,
        col_cluster=True,
        method="ward",
        metric="euclidean",
        cmap="coolwarm",
        center=0,
        vmin=-vmax,
        vmax=vmax,
        row_colors=row_colors,
        xticklabels=True,
        yticklabels=False,
        figsize=figsize,
        dendrogram_ratio=(0.12, 0.12),
        cbar_kws={"label": "Scaled SHAP value" if standardize else "SHAP value"},
    )

    cg.fig.suptitle(title or f"Supplementary Figure 6. Cross-cohort {feature_set}: subject × node-feature SHAP heatmap", y=1.02, fontsize=16)
    cg.ax_heatmap.set_xlabel("Node-feature SHAP label", fontsize=12)
    cg.ax_heatmap.set_ylabel("Subjects", fontsize=12)
    cg.ax_heatmap.tick_params(axis="x", labelrotation=90, labelsize=7)
    add_cohort_legend(cg, cohorts, cohort_series)

    row_order = combined_scaled.index[cg.dendrogram_row.reordered_ind]
    col_order = combined_scaled.columns[cg.dendrogram_col.reordered_ind]
    clustered_raw = combined_raw.loc[row_order, col_order]
    clustered_scaled = combined_scaled.loc[row_order, col_order]
    clustered_cohorts = cohort_series.loc[row_order]

    clustered_raw_path = outdir / f"{prefix}_{feature_set}_node_feature_SHAP_clustered_matrix_raw.csv"
    clustered_scaled_path = outdir / f"{prefix}_{feature_set}_node_feature_SHAP_clustered_matrix_scaled.csv"
    clustered_cohorts_path = outdir / f"{prefix}_{feature_set}_node_feature_SHAP_clustered_subject_cohorts.csv"
    clustered_raw.to_csv(clustered_raw_path)
    clustered_scaled.to_csv(clustered_scaled_path)
    clustered_cohorts.to_csv(clustered_cohorts_path)

    saved = save_formats(cg, outdir / f"{prefix}_{feature_set}_node_feature_SHAP_heatmap", formats, dpi)
    plt.close(cg.fig)

    ordered_path, composition_path = save_subject_order_outputs(row_order, cohort_series, outdir, feature_set, f"{prefix}_node_feature_SHAP")

    feature_summary = pd.DataFrame({
        "node_feature_label": combined_raw.columns,
        "mean_abs_SHAP_raw": combined_raw.abs().mean(axis=0).values,
        "mean_SHAP_raw": combined_raw.mean(axis=0).values,
        "sd_SHAP_raw": combined_raw.std(axis=0, ddof=1).values,
        "mean_abs_scaled_SHAP": combined_scaled.abs().mean(axis=0).values,
    }).sort_values("mean_abs_SHAP_raw", ascending=False)
    feature_summary_path = outdir / f"{prefix}_{feature_set}_node_feature_SHAP_feature_summary.csv"
    feature_summary.to_csv(feature_summary_path, index=False)

    outputs = [
        *saved,
        str(raw_path),
        str(scaled_path),
        str(clustered_raw_path),
        str(clustered_scaled_path),
        str(cohorts_path),
        str(clustered_cohorts_path),
        ordered_path,
        composition_path,
        str(feature_summary_path),
        str(ranked_path),
        str(load_log_path),
    ]

    manifest_path = outdir / f"{prefix}_{feature_set}_node_feature_SHAP_manifest.csv"
    pd.DataFrame([{
        "figure": "Supplementary Figure 6 cross-cohort node-feature SHAP aggregation",
        "feature_set": feature_set,
        "cohorts": ",".join(cohorts),
        "n_subjects": int(combined_raw.shape[0]),
        "n_features": int(combined_raw.shape[1]),
        "outputs": ";".join(outputs),
    }]).to_csv(manifest_path, index=False)
    outputs.append(str(manifest_path))

    print("\nSaved supplementary cross-cohort node-feature SHAP aggregation:")
    for p in outputs:
        print(f"  {p}")
    return outputs


# =============================================================================
# Aggregate summary CSVs
# =============================================================================
def aggregate_global_summary_csvs(out_base: Path, cohorts: Sequence[str], feature_set: str) -> List[str]:
    outdir = Path(ensure_dir(out_base / "cross_cohort_aggregated"))
    frames: List[pd.DataFrame] = []

    for cohort in cohorts:
        if cohort not in COHORT_CONFIG:
            continue
        df = read_global_summary_csv(out_base, cohort, feature_set)
        if df is not None and not df.empty:
            frames.append(df)

    if not frames:
        print("[WARN] No global summary CSVs found to aggregate.")
        return []

    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined_path = outdir / f"Figure6_cross_cohort_{feature_set}_global_feature_SHAP_summary_by_cohort.csv"
    combined.to_csv(combined_path, index=False)

    outputs = [str(combined_path)]
    if {"feature_name_clean", "mean_abs_SHAP", "mean_SHAP"}.issubset(combined.columns):
        summary = (
            combined.groupby("feature_name_clean", as_index=False)
            .agg(
                mean_abs_SHAP_across_cohorts=("mean_abs_SHAP", "mean"),
                sd_abs_SHAP_across_cohorts=("mean_abs_SHAP", "std"),
                mean_signed_SHAP_across_cohorts=("mean_SHAP", "mean"),
                n_cohorts=("cohort", "nunique"),
                cohorts=("cohort", lambda x: ",".join(sorted(set(map(str, x))))),
            )
            .sort_values("mean_abs_SHAP_across_cohorts", ascending=False)
        )
        summary_path = outdir / f"Figure6_cross_cohort_{feature_set}_global_feature_SHAP_summary_collapsed.csv"
        summary.to_csv(summary_path, index=False)
        outputs.append(str(summary_path))
    return outputs


def aggregate_node_summary_csvs(out_base: Path, cohorts: Sequence[str], feature_set: str) -> List[str]:
    outdir = Path(ensure_dir(out_base / "cross_cohort_aggregated"))
    frames: List[pd.DataFrame] = []

    for cohort in cohorts:
        if cohort not in COHORT_CONFIG:
            continue
        df = read_node_summary_csv(out_base, cohort, feature_set)
        if df is not None and not df.empty:
            frames.append(df)

    if not frames:
        print("[WARN] No node summary CSVs found to aggregate.")
        return []

    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined_path = outdir / f"Supplementary_Figure6_cross_cohort_{feature_set}_node_feature_SHAP_summary_by_cohort.csv"
    combined.to_csv(combined_path, index=False)

    outputs = [str(combined_path)]
    if {"node_feature_label", "mean_abs_SHAP", "mean_SHAP"}.issubset(combined.columns):
        summary = (
            combined.groupby("node_feature_label", as_index=False)
            .agg(
                mean_abs_SHAP_across_cohorts=("mean_abs_SHAP", "mean"),
                sd_abs_SHAP_across_cohorts=("mean_abs_SHAP", "std"),
                mean_signed_SHAP_across_cohorts=("mean_SHAP", "mean"),
                n_cohorts=("cohort", "nunique"),
                cohorts=("cohort", lambda x: ",".join(sorted(set(map(str, x))))),
            )
            .sort_values(["n_cohorts", "mean_abs_SHAP_across_cohorts"], ascending=[False, False])
        )
        summary_path = outdir / f"Supplementary_Figure6_cross_cohort_{feature_set}_node_feature_SHAP_summary_collapsed.csv"
        summary.to_csv(summary_path, index=False)
        outputs.append(str(summary_path))
    return outputs


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    args = parse_args()

    out_base = Path(args.out_base).expanduser().resolve()
    cohorts = [c.strip() for c in args.cohorts.split(",") if c.strip()]
    formats = [f.strip().lower().lstrip(".") for f in args.formats.split(",") if f.strip()]
    standardize = not args.no_standardize

    print("\n" + "=" * 100)
    print("AGGREGATE FIGURE 6")
    print("=" * 100)
    print("OUT_BASE:       ", out_base)
    print("Feature set:    ", args.feature_set)
    print("Cohorts:        ", ", ".join(cohorts))
    print("Formats:        ", ", ".join(formats))
    print("Make graph:     ", bool(args.make_graph))
    print("Make node supp: ", bool(args.make_node_supplement))
    print("Standardize:    ", standardize)
    print("=" * 100)

    all_outputs: List[str] = []

    if bool(args.make_graph):
        all_outputs.extend(make_cross_cohort_graph_shap_heatmap(
            out_base=out_base,
            cohorts=cohorts,
            feature_set=args.feature_set,
            formats=formats,
            dpi=args.dpi,
            top_n_features=args.top_n_graph_features,
            standardize=standardize,
            figsize=(args.graph_fig_width, args.graph_fig_height),
            title=args.graph_title,
        ))

    if bool(args.make_node_supplement):
        all_outputs.extend(make_supplementary_cross_cohort_node_heatmap(
            out_base=out_base,
            cohorts=cohorts,
            feature_set=args.feature_set,
            formats=formats,
            dpi=args.dpi,
            top_n_features=args.top_n_node_features,
            standardize=standardize,
            max_subjects_per_cohort=args.max_subjects_per_cohort_node,
            figsize=(args.node_fig_width, args.node_fig_height),
            title=args.node_title,
        ))

    if bool(args.also_aggregate_summary_csvs):
        all_outputs.extend(aggregate_global_summary_csvs(out_base, cohorts, args.feature_set))
        all_outputs.extend(aggregate_node_summary_csvs(out_base, cohorts, args.feature_set))

    outdir = Path(ensure_dir(out_base / "cross_cohort_aggregated"))
    overall_manifest = outdir / f"Figure6_aggregate_{args.feature_set}_overall_manifest.csv"
    pd.DataFrame([{
        "script": "aggregate_figure6.py",
        "feature_set": args.feature_set,
        "cohorts": ",".join(cohorts),
        "make_graph": bool(args.make_graph),
        "make_node_supplement": bool(args.make_node_supplement),
        "outputs": ";".join(all_outputs),
    }]).to_csv(overall_manifest, index=False)
    all_outputs.append(str(overall_manifest))

    print("\nFinished aggregate Figure 6.")
    print("\nOutputs:")
    for p in all_outputs:
        print(f"  {p}")


if __name__ == "__main__":
    main()
