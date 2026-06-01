#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure 6 cross-cohort SHAP hierarchical analysis
================================================

Purpose
-------
Build cross-cohort subject-level SHAP analyses from the existing per-subject CSVs
created by Figure6_build.py. This script does NOT recompute SHAP. It reads:

  <out-base>/<cohort_slug>/<model>/global_feature_shap/global_feature_shap_subject_*.csv
  <out-base>/<cohort_slug>/<model>/node_feature_shap/node_feature_shap_subject_*.csv
  <out-base>/<cohort_slug>/<model>/edge_shap/edge_shap_subject_*.csv

Then it saves cross-cohort outputs under:

  <out-base>/cross_cohort_aggregated/<model>/hierarchical_clustering/

Recommended first run
---------------------
python Figure6_cross_cohort_SHAP_hierarchical_analysis.py \
  --models imaging_only \
  --make-edge 0

Full run, including edge heatmap if edge subject CSVs exist
-----------------------------------------------------------
python Figure6_cross_cohort_SHAP_hierarchical_analysis.py \
  --models imaging_only \
  --make-edge 1

Useful outputs
--------------
For each model and SHAP kind, the script writes:
  - cross-cohort hierarchical heatmap PNG/PDF
  - raw subject x feature matrix CSV
  - scaled subject x feature matrix CSV
  - clustered raw/scaled matrices CSV
  - row/subject order CSV
  - column/feature order CSV
  - cohort composition by cluster CSV
  - chi-square/permutation cohort-enrichment summary CSV
  - load log CSV
  - output manifest CSV

Dependencies
------------
numpy, pandas, matplotlib, seaborn, scipy
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.cluster.hierarchy import leaves_list, linkage, fcluster
from scipy.spatial.distance import pdist
from scipy.stats import chi2_contingency


# =============================================================================
# Defaults matching your Figure 6 scripts
# =============================================================================
DEFAULT_WORK = os.environ.get("WORK", "/mnt/newStor/paros/paros_WORK")
DEFAULT_OUT_BASE = (
    Path(DEFAULT_WORK)
    / "ines/results"
    / "BrainAgeValidation_AllCohorts_BAGBiasCorr_OOFGlobal_BiologicalValidation"
    / "Figure6_SHAP"
)

COHORT_CONFIG = {
    "ADNI": {"cohort_slug": "adni", "display": "ADNI"},
    "ADRC": {"cohort_slug": "adrc", "display": "ADRC"},
    "HABS": {"cohort_slug": "habs", "display": "HABS"},
    "AD_DECODE": {"cohort_slug": "addecode", "display": "AD-DECODE"},
}
DEFAULT_COHORTS = ["ADNI", "ADRC", "HABS", "AD_DECODE"]

PREFERRED_MODEL_ORDER = [
    "imaging_only",
    "full",
    "imaging_demographics",
    "imaging_biomarkers",
    "full_no_cardiovascular",
    "clinical_only",
    "non_imaging_only",
    "multimodal",
    "graph_only",
]

COHORT_PALETTE = {
    "ADNI": "#1f77b4",
    "ADRC": "#ff7f0e",
    "HABS": "#2ca02c",
    "AD_DECODE": "#d62728",
}


# =============================================================================
# General helpers
# =============================================================================
def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def parse_bool01(x: str) -> bool:
    return str(x).strip().lower() in {"1", "true", "yes", "y", "on"}


def zscore_columns(matrix: pd.DataFrame) -> pd.DataFrame:
    m = matrix.astype(float).copy()
    mu = m.mean(axis=0, skipna=True)
    sd = m.std(axis=0, skipna=True).replace(0, np.nan)
    z = (m - mu) / sd
    return z.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def safe_corr_distance(matrix: pd.DataFrame) -> np.ndarray:
    """Correlation distance with a Euclidean fallback for degenerate matrices."""
    arr = matrix.to_numpy(dtype=float)
    if arr.shape[0] < 2:
        return np.array([])
    try:
        d = pdist(arr, metric="correlation")
        if np.isfinite(d).all():
            return d
    except Exception:
        pass
    d = pdist(arr, metric="euclidean")
    d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
    return d


def discover_models(out_base: Path, cohorts: List[str]) -> List[str]:
    found = set()
    for cohort in cohorts:
        cfg = COHORT_CONFIG.get(cohort)
        if cfg is None:
            continue
        cohort_dir = out_base / cfg["cohort_slug"]
        if not cohort_dir.exists():
            continue
        for child in cohort_dir.iterdir():
            if not child.is_dir():
                continue
            if (
                (child / "global_feature_shap").exists()
                or (child / "node_feature_shap").exists()
                or (child / "edge_shap").exists()
            ):
                found.add(child.name)
    ordered = [m for m in PREFERRED_MODEL_ORDER if m in found]
    ordered.extend(sorted(m for m in found if m not in ordered))
    return ordered


def subject_id_from_file(path: Path, prefix: str) -> str:
    return path.stem.replace(prefix, "")


# =============================================================================
# Load subject-level SHAP CSVs
# =============================================================================
def make_node_feature_label(df: pd.DataFrame) -> Optional[pd.Series]:
    if "node_feature_label" in df.columns:
        return df["node_feature_label"].astype(str)
    if {"node_label", "feature_name"}.issubset(df.columns):
        return df["node_label"].astype(str) + " | " + df["feature_name"].astype(str)
    if {"Structure", "feature_name"}.issubset(df.columns):
        return df["Structure"].astype(str) + " | " + df["feature_name"].astype(str)
    if {"structure", "feature_name"}.issubset(df.columns):
        return df["structure"].astype(str) + " | " + df["feature_name"].astype(str)
    return None


def make_edge_label(df: pd.DataFrame) -> Optional[pd.Series]:
    if "edge_feature_label" in df.columns:
        return df["edge_feature_label"].astype(str)
    if "Edge" in df.columns:
        label = df["Edge"].astype(str)
    elif {"Structure_i", "Structure_j"}.issubset(df.columns):
        label = df["Structure_i"].astype(str) + " -- " + df["Structure_j"].astype(str)
    elif {"structure_i", "structure_j"}.issubset(df.columns):
        label = df["structure_i"].astype(str) + " -- " + df["structure_j"].astype(str)
    elif {"Node_i", "Node_j"}.issubset(df.columns):
        label = df["Node_i"].astype(str) + " -- " + df["Node_j"].astype(str)
    else:
        return None

    if "edge_feature_name" in df.columns and df["edge_feature_name"].astype(str).nunique() > 1:
        label = label + " | " + df["edge_feature_name"].astype(str)
    return label


def read_subject_table(path: Path, kind: str, cohort: str, model: str) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    if df.empty or "SHAP_val" not in df.columns:
        return None

    df = df.copy()
    df["SHAP_val"] = pd.to_numeric(df["SHAP_val"], errors="coerce")
    if "abs_SHAP" not in df.columns:
        df["abs_SHAP"] = df["SHAP_val"].abs()
    else:
        df["abs_SHAP"] = pd.to_numeric(df["abs_SHAP"], errors="coerce")

    if kind == "global":
        if "feature_name" not in df.columns:
            return None
        df["feature_label"] = df["feature_name"].astype(str)
    elif kind == "node":
        label = make_node_feature_label(df)
        if label is None:
            return None
        df["feature_label"] = label
    elif kind == "edge":
        label = make_edge_label(df)
        if label is None:
            return None
        df["feature_label"] = label
    else:
        raise ValueError(f"Unknown kind: {kind}")

    prefixes = {
        "global": "global_feature_shap_subject_",
        "node": "node_feature_shap_subject_",
        "edge": "edge_shap_subject_",
    }
    df["subject"] = subject_id_from_file(path, prefixes[kind])
    df["subject_uid"] = cohort + "__" + df["subject"].astype(str)
    df["cohort"] = cohort
    df["model"] = model
    df["source_csv"] = str(path)
    return df[["subject_uid", "subject", "cohort", "model", "feature_label", "SHAP_val", "abs_SHAP", "source_csv"]]


def load_cross_cohort_long(out_base: Path, cohorts: List[str], model: str, kind: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    subdirs = {
        "global": "global_feature_shap",
        "node": "node_feature_shap",
        "edge": "edge_shap",
    }
    prefixes = {
        "global": "global_feature_shap_subject_*.csv",
        "node": "node_feature_shap_subject_*.csv",
        "edge": "edge_shap_subject_*.csv",
    }

    frames: List[pd.DataFrame] = []
    log_rows = []

    for cohort in cohorts:
        cfg = COHORT_CONFIG[cohort]
        input_dir = out_base / cfg["cohort_slug"] / model / subdirs[kind]
        files = sorted(input_dir.glob(prefixes[kind])) if input_dir.exists() else []
        loaded = 0
        skipped = 0

        for path in files:
            df = read_subject_table(path, kind=kind, cohort=cohort, model=model)
            if df is None or df.empty:
                skipped += 1
                continue
            frames.append(df)
            loaded += 1

        log_rows.append({
            "model": model,
            "kind": kind,
            "cohort": cohort,
            "input_dir": str(input_dir),
            "n_files_found": len(files),
            "n_files_loaded": loaded,
            "n_files_skipped": skipped,
            "status": "loaded" if loaded > 0 else "missing_or_empty",
        })

    if not frames:
        return pd.DataFrame(), pd.DataFrame(log_rows)

    long_df = pd.concat(frames, ignore_index=True, sort=False)
    long_df = long_df.dropna(subset=["subject_uid", "feature_label", "SHAP_val"])
    return long_df, pd.DataFrame(log_rows)


# =============================================================================
# Matrices, summaries, and statistics
# =============================================================================
def rank_features(long_df: pd.DataFrame, min_cohorts: int = 2) -> pd.DataFrame:
    if long_df.empty:
        return pd.DataFrame()

    by_cohort = (
        long_df.groupby(["cohort", "feature_label"], as_index=False)
        .agg(
            mean_abs_SHAP=("abs_SHAP", "mean"),
            mean_SHAP=("SHAP_val", "mean"),
            n_subjects=("subject_uid", "nunique"),
        )
    )

    ranked = (
        by_cohort.groupby("feature_label", as_index=False)
        .agg(
            mean_abs_SHAP_across_cohorts=("mean_abs_SHAP", "mean"),
            sd_abs_SHAP_across_cohorts=("mean_abs_SHAP", "std"),
            mean_signed_SHAP_across_cohorts=("mean_SHAP", "mean"),
            n_cohorts=("cohort", "nunique"),
            total_n_subjects=("n_subjects", "sum"),
            cohorts=("cohort", lambda x: ",".join(sorted(set(map(str, x))))),
        )
        .sort_values(["n_cohorts", "mean_abs_SHAP_across_cohorts"], ascending=[False, False])
        .reset_index(drop=True)
    )

    if min_cohorts > 1:
        ranked = ranked[ranked["n_cohorts"] >= min_cohorts].reset_index(drop=True)
    return ranked


def build_subject_feature_matrix(
    long_df: pd.DataFrame,
    ranked: pd.DataFrame,
    top_n: int,
    value_col: str = "SHAP_val",
) -> pd.DataFrame:
    if long_df.empty or ranked.empty:
        return pd.DataFrame()
    keep = ranked.head(top_n)["feature_label"].astype(str).tolist()
    tmp = long_df[long_df["feature_label"].astype(str).isin(keep)].copy()
    if tmp.empty:
        return pd.DataFrame()
    matrix = tmp.pivot_table(
        index="subject_uid",
        columns="feature_label",
        values=value_col,
        aggfunc="mean",
    )
    matrix = matrix.reindex(columns=keep)
    matrix = matrix.fillna(0.0)
    return matrix


def cluster_matrix(matrix_scaled: pd.DataFrame, n_clusters: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    if matrix_scaled.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), np.array([]), np.array([])

    row_link = None
    col_link = None

    if matrix_scaled.shape[0] >= 2:
        row_link = linkage(safe_corr_distance(matrix_scaled), method="average")
        row_order = leaves_list(row_link)
    else:
        row_order = np.arange(matrix_scaled.shape[0])

    if matrix_scaled.shape[1] >= 2:
        col_link = linkage(safe_corr_distance(matrix_scaled.T), method="average")
        col_order = leaves_list(col_link)
    else:
        col_order = np.arange(matrix_scaled.shape[1])

    clustered_scaled = matrix_scaled.iloc[row_order, col_order]

    if row_link is not None and n_clusters and n_clusters > 1:
        cluster_labels = fcluster(row_link, t=n_clusters, criterion="maxclust")
        cluster_series = pd.Series(cluster_labels, index=matrix_scaled.index, name="cluster")
    else:
        cluster_series = pd.Series(1, index=matrix_scaled.index, name="cluster")

    row_order_df = pd.DataFrame({
        "subject_uid": matrix_scaled.index[row_order],
        "row_order": np.arange(len(row_order)),
        "cluster": cluster_series.reindex(matrix_scaled.index[row_order]).astype(int).to_numpy(),
    })
    col_order_df = pd.DataFrame({
        "feature_label": matrix_scaled.columns[col_order],
        "column_order": np.arange(len(col_order)),
    })

    return clustered_scaled, row_order_df, col_order_df, row_link, col_link


def cohort_cluster_enrichment(
    row_order_df: pd.DataFrame,
    subject_meta: pd.DataFrame,
    n_permutations: int,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if row_order_df.empty or subject_meta.empty:
        return pd.DataFrame(), pd.DataFrame()

    meta = row_order_df.merge(subject_meta, on="subject_uid", how="left")
    observed = pd.crosstab(meta["cluster"], meta["cohort"])
    if observed.empty or observed.shape[0] < 2 or observed.shape[1] < 2:
        summary = pd.DataFrame([{
            "test": "cohort_by_cluster",
            "chi2": np.nan,
            "dof": np.nan,
            "asymptotic_p": np.nan,
            "permutation_p": np.nan,
            "n_permutations": n_permutations,
            "note": "Not enough clusters/cohorts for test",
        }])
        return observed.reset_index(), summary

    chi2, p, dof, _ = chi2_contingency(observed)

    perm_p = np.nan
    if n_permutations and n_permutations > 0:
        rng = np.random.default_rng(seed)
        cohorts = meta["cohort"].to_numpy().copy()
        clusters = meta["cluster"].to_numpy().copy()
        perm_stats = []
        for _ in range(n_permutations):
            permuted = rng.permutation(cohorts)
            tab = pd.crosstab(clusters, permuted)
            tab = tab.reindex(index=observed.index, columns=observed.columns, fill_value=0)
            try:
                stat, _, _, _ = chi2_contingency(tab)
                perm_stats.append(stat)
            except Exception:
                continue
        if perm_stats:
            perm_stats = np.asarray(perm_stats, dtype=float)
            perm_p = float((np.sum(perm_stats >= chi2) + 1) / (len(perm_stats) + 1))

    summary = pd.DataFrame([{
        "test": "cohort_by_cluster",
        "chi2": float(chi2),
        "dof": int(dof),
        "asymptotic_p": float(p),
        "permutation_p": perm_p,
        "n_permutations": int(n_permutations),
        "interpretation": "Low p suggests subject clusters are enriched by cohort/site; high p suggests clustering is not strongly cohort-driven.",
    }])
    return observed.reset_index(), summary


# =============================================================================
# Plotting
# =============================================================================
def make_col_colors(subject_ids: Iterable[str], subject_meta: pd.DataFrame) -> pd.Series:
    meta = subject_meta.set_index("subject_uid")
    cohorts = meta.reindex(list(subject_ids))["cohort"].fillna("Unknown")
    return cohorts.map(COHORT_PALETTE).fillna("#808080")


def save_clustermap(
    matrix_scaled: pd.DataFrame,
    subject_meta: pd.DataFrame,
    out_png: Path,
    title: str,
    figsize: Tuple[float, float],
    dpi: int,
    row_linkage_obj=None,
    col_linkage_obj=None,
    hide_subject_labels_if_gt: int = 80,
) -> bool:
    if matrix_scaled.empty or matrix_scaled.shape[0] < 2 or matrix_scaled.shape[1] < 2:
        return False

    row_colors = make_col_colors(matrix_scaled.index, subject_meta)

    g = sns.clustermap(
        matrix_scaled,
        row_linkage=row_linkage_obj,
        col_linkage=col_linkage_obj,
        row_colors=row_colors,
        cmap="vlag",
        center=0,
        linewidths=0.0,
        xticklabels=True,
        yticklabels=(matrix_scaled.shape[0] <= hide_subject_labels_if_gt),
        figsize=figsize,
        cbar_kws={"label": "Scaled signed SHAP value"},
    )
    g.fig.suptitle(title, y=1.02, fontsize=16)
    g.ax_heatmap.set_xlabel("SHAP feature")
    g.ax_heatmap.set_ylabel("Subject")
    plt.setp(g.ax_heatmap.get_xticklabels(), rotation=90, ha="center", fontsize=7)
    if matrix_scaled.shape[0] <= hide_subject_labels_if_gt:
        plt.setp(g.ax_heatmap.get_yticklabels(), fontsize=5)

    # Cohort legend
    for cohort, color in COHORT_PALETTE.items():
        g.ax_col_dendrogram.bar(0, 0, color=color, label=cohort, linewidth=0)
    g.ax_col_dendrogram.legend(
        title="Cohort",
        loc="center",
        ncol=min(4, len(COHORT_PALETTE)),
        bbox_to_anchor=(0.5, 1.10),
        frameon=False,
    )

    g.fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    g.fig.savefig(out_png.with_suffix(".pdf"), dpi=dpi, bbox_inches="tight")
    plt.close(g.fig)
    return True


def save_beeswarm_like(
    long_df: pd.DataFrame,
    ranked: pd.DataFrame,
    out_png: Path,
    title: str,
    top_n: int,
    dpi: int,
) -> bool:
    if long_df.empty or ranked.empty:
        return False
    keep = ranked.head(top_n)["feature_label"].astype(str).tolist()
    tmp = long_df[long_df["feature_label"].astype(str).isin(keep)].copy()
    if tmp.empty:
        return False
    order = keep[::-1]
    ymap = {f: i for i, f in enumerate(order)}
    rng = np.random.default_rng(12345)
    tmp["y"] = tmp["feature_label"].astype(str).map(ymap) + rng.normal(0, 0.08, len(tmp))

    fig_h = max(7, 0.45 * len(order))
    fig, ax = plt.subplots(figsize=(12, fig_h))
    for cohort, cdf in tmp.groupby("cohort"):
        ax.scatter(cdf["SHAP_val"], cdf["y"], s=9, alpha=0.55, label=cohort, color=COHORT_PALETTE.get(cohort, None))
    ax.axvline(0, linewidth=0.8, color="black")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=8)
    ax.set_xlabel("SHAP value")
    ax.set_ylabel("Feature")
    ax.set_title(title)
    ax.legend(title="Cohort", loc="best", frameon=False)
    fig.tight_layout()
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    fig.savefig(out_png.with_suffix(".pdf"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return True


def save_rank_barplot(ranked: pd.DataFrame, out_png: Path, title: str, top_n: int, dpi: int) -> bool:
    if ranked.empty:
        return False
    top = ranked.head(top_n).iloc[::-1].copy()
    fig_h = max(7, 0.36 * len(top))
    fig, ax = plt.subplots(figsize=(11, fig_h))
    ax.barh(top["feature_label"], top["mean_abs_SHAP_across_cohorts"])
    ax.set_xlabel("Mean |SHAP| across cohorts")
    ax.set_ylabel("Feature")
    ax.set_title(title)
    ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    fig.savefig(out_png.with_suffix(".pdf"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return True


# =============================================================================
# Main per-kind workflow
# =============================================================================
def run_kind(
    out_base: Path,
    cohorts: List[str],
    model: str,
    kind: str,
    top_n: int,
    min_cohorts: int,
    value_col: str,
    n_clusters: int,
    n_permutations: int,
    dpi: int,
    seed: int,
    figsize: Tuple[float, float],
) -> List[str]:
    outdir = ensure_dir(out_base / "cross_cohort_aggregated" / model / "hierarchical_clustering")
    prefix = f"Supplementary_Figure6_cross_cohort_{model}_{kind}_SHAP"
    outputs: List[str] = []

    long_df, load_log = load_cross_cohort_long(out_base, cohorts, model, kind)
    load_log_path = outdir / f"{prefix}_input_load_log.csv"
    load_log.to_csv(load_log_path, index=False)
    outputs.append(str(load_log_path))

    if long_df.empty:
        return outputs

    long_path = outdir / f"{prefix}_subject_level_long.csv"
    long_df.to_csv(long_path, index=False)
    outputs.append(str(long_path))

    subject_meta = long_df[["subject_uid", "subject", "cohort", "model"]].drop_duplicates()
    subject_meta_path = outdir / f"{prefix}_subject_metadata.csv"
    subject_meta.to_csv(subject_meta_path, index=False)
    outputs.append(str(subject_meta_path))

    ranked = rank_features(long_df, min_cohorts=min_cohorts)
    ranked_path = outdir / f"{prefix}_ranked_features.csv"
    ranked.to_csv(ranked_path, index=False)
    outputs.append(str(ranked_path))

    by_cohort = (
        long_df.groupby(["cohort", "feature_label"], as_index=False)
        .agg(
            mean_abs_SHAP=("abs_SHAP", "mean"),
            mean_SHAP=("SHAP_val", "mean"),
            n_subjects=("subject_uid", "nunique"),
        )
        .sort_values(["cohort", "mean_abs_SHAP"], ascending=[True, False])
    )
    by_cohort_path = outdir / f"{prefix}_summary_by_cohort.csv"
    by_cohort.to_csv(by_cohort_path, index=False)
    outputs.append(str(by_cohort_path))

    matrix_raw = build_subject_feature_matrix(long_df, ranked, top_n=top_n, value_col=value_col)
    raw_path = outdir / f"{prefix}_top{top_n}_matrix_raw.csv"
    matrix_raw.to_csv(raw_path)
    outputs.append(str(raw_path))

    if matrix_raw.empty:
        return outputs

    matrix_scaled = zscore_columns(matrix_raw)
    scaled_path = outdir / f"{prefix}_top{top_n}_matrix_scaled.csv"
    matrix_scaled.to_csv(scaled_path)
    outputs.append(str(scaled_path))

    clustered_scaled, row_order_df, col_order_df, row_link, col_link = cluster_matrix(matrix_scaled, n_clusters=n_clusters)
    row_order_path = outdir / f"{prefix}_top{top_n}_row_order_clusters.csv"
    col_order_path = outdir / f"{prefix}_top{top_n}_column_order.csv"
    row_order_df.to_csv(row_order_path, index=False)
    col_order_df.to_csv(col_order_path, index=False)
    outputs.extend([str(row_order_path), str(col_order_path)])

    clustered_raw = matrix_raw.reindex(index=row_order_df["subject_uid"], columns=col_order_df["feature_label"])
    clustered_raw_path = outdir / f"{prefix}_top{top_n}_clustered_matrix_raw.csv"
    clustered_scaled_path = outdir / f"{prefix}_top{top_n}_clustered_matrix_scaled.csv"
    clustered_raw.to_csv(clustered_raw_path)
    clustered_scaled.to_csv(clustered_scaled_path)
    outputs.extend([str(clustered_raw_path), str(clustered_scaled_path)])

    composition, enrichment = cohort_cluster_enrichment(row_order_df, subject_meta, n_permutations, seed)
    comp_path = outdir / f"{prefix}_top{top_n}_cohort_by_cluster_counts.csv"
    enrich_path = outdir / f"{prefix}_top{top_n}_cohort_cluster_enrichment.csv"
    composition.to_csv(comp_path, index=False)
    enrichment.to_csv(enrich_path, index=False)
    outputs.extend([str(comp_path), str(enrich_path)])

    heatmap_png = outdir / f"{prefix}_top{top_n}_hierarchical_heatmap.png"
    ok = save_clustermap(
        matrix_scaled=matrix_scaled,
        subject_meta=subject_meta,
        out_png=heatmap_png,
        title=f"Cross-cohort {model}: subject × {kind} SHAP heatmap",
        figsize=figsize,
        dpi=dpi,
        row_linkage_obj=row_link,
        col_linkage_obj=col_link,
    )
    if ok:
        outputs.extend([str(heatmap_png), str(heatmap_png.with_suffix(".pdf"))])

    beeswarm_png = outdir / f"{prefix}_top{min(top_n, 25)}_beeswarm.png"
    ok = save_beeswarm_like(
        long_df=long_df,
        ranked=ranked,
        out_png=beeswarm_png,
        title=f"Cross-cohort {model}: {kind} SHAP distribution",
        top_n=min(top_n, 25),
        dpi=dpi,
    )
    if ok:
        outputs.extend([str(beeswarm_png), str(beeswarm_png.with_suffix(".pdf"))])

    bar_png = outdir / f"{prefix}_top{min(top_n, 30)}_rank_barplot.png"
    ok = save_rank_barplot(
        ranked=ranked,
        out_png=bar_png,
        title=f"Cross-cohort {model}: top {kind} SHAP contributors",
        top_n=min(top_n, 30),
        dpi=dpi,
    )
    if ok:
        outputs.extend([str(bar_png), str(bar_png.with_suffix(".pdf"))])

    manifest_path = outdir / f"{prefix}_manifest.csv"
    pd.DataFrame({"output": outputs}).to_csv(manifest_path, index=False)
    outputs.append(str(manifest_path))
    return outputs


# =============================================================================
# CLI
# =============================================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cross-cohort subject-level SHAP hierarchical heatmaps and analyses.")
    p.add_argument("--out-base", default=str(DEFAULT_OUT_BASE), help="Root Figure6_SHAP folder.")
    p.add_argument("--cohorts", default=",".join(DEFAULT_COHORTS), help="Comma-separated cohort names.")
    p.add_argument("--models", default=None, help="Comma-separated models. If omitted, discover models under out-base.")
    p.add_argument("--top-n-global", type=int, default=35)
    p.add_argument("--top-n-node", type=int, default=50)
    p.add_argument("--top-n-edge", type=int, default=50)
    p.add_argument("--min-cohorts", type=int, default=2, help="Require features to appear in at least this many cohorts.")
    p.add_argument("--value-col", default="SHAP_val", choices=["SHAP_val", "abs_SHAP"])
    p.add_argument("--n-clusters", type=int, default=4, help="Number of row clusters for cohort-enrichment analysis.")
    p.add_argument("--n-permutations", type=int, default=1000, help="Permutations for cohort-by-cluster enrichment.")
    p.add_argument("--dpi", type=int, default=400)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--make-global", default="1")
    p.add_argument("--make-node", default="1")
    p.add_argument("--make-edge", default="0", help="Set 1 only if edge_shap_subject_*.csv files exist and you want edge heatmaps.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_base = Path(args.out_base).expanduser().resolve()
    cohorts = [c.strip() for c in args.cohorts.split(",") if c.strip()]
    missing_cohorts = [c for c in cohorts if c not in COHORT_CONFIG]
    if missing_cohorts:
        raise SystemExit(f"Unknown cohorts: {missing_cohorts}. Known: {list(COHORT_CONFIG)}")

    if args.models:
        models = [m.strip() for m in args.models.split(",") if m.strip()]
    else:
        models = discover_models(out_base, cohorts)
    if not models:
        raise SystemExit(f"No models found under {out_base}")

    jobs = []
    if parse_bool01(args.make_global):
        jobs.append(("global", args.top_n_global, (13, 18)))
    if parse_bool01(args.make_node):
        jobs.append(("node", args.top_n_node, (18, 18)))
    if parse_bool01(args.make_edge):
        jobs.append(("edge", args.top_n_edge, (22, 18)))

    all_outputs = []
    print("=" * 100)
    print("Cross-cohort SHAP hierarchical analysis")
    print(f"OUT_BASE: {out_base}")
    print(f"COHORTS:  {', '.join(cohorts)}")
    print(f"MODELS:   {', '.join(models)}")
    print("=" * 100)

    for model in models:
        print(f"\n--- Model: {model} ---")
        for kind, top_n, figsize in jobs:
            print(f"  Building {kind} cross-cohort analysis, top_n={top_n}")
            outputs = run_kind(
                out_base=out_base,
                cohorts=cohorts,
                model=model,
                kind=kind,
                top_n=top_n,
                min_cohorts=args.min_cohorts,
                value_col=args.value_col,
                n_clusters=args.n_clusters,
                n_permutations=args.n_permutations,
                dpi=args.dpi,
                seed=args.seed,
                figsize=figsize,
            )
            all_outputs.extend(outputs)
            print(f"    wrote {len(outputs)} outputs")

    final_dir = ensure_dir(out_base / "cross_cohort_aggregated" / "final_figures")
    run_manifest = final_dir / "Figure6_cross_cohort_SHAP_hierarchical_analysis_manifest.csv"
    pd.DataFrame({"output": all_outputs}).to_csv(run_manifest, index=False)
    print("\nDone.")
    print(f"Run manifest: {run_manifest}")


if __name__ == "__main__":
    main()
