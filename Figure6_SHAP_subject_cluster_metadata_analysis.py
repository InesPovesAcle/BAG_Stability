#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cross-cohort SHAP subject clustering + post-hoc metadata discrimination
======================================================================

This script does NOT recompute SHAP.

It creates subject clusters from SHAP profiles only, then tests whether those
SHAP-defined clusters differ by APOE4 carriage, sex, cognitive status, and AD
status.

Inputs expected from Figure6_build.py:
  <out-base>/<cohort_slug>/<model>/global_feature_shap/global_feature_shap_subject_*.csv
  <out-base>/<cohort_slug>/<model>/node_feature_shap/node_feature_shap_subject_*.csv
  <out-base>/<cohort_slug>/<model>/edge_shap/edge_shap_subject_*.csv

Default output:
  <out-base>/cross_cohort_aggregated/<model>/subject_shap_clusters_metadata/

Example:
  python Figure6_SHAP_subject_cluster_metadata_analysis.py \
    --models imaging_only \
    --kinds node,global \
    --n-clusters 4

If needed, force metadata columns:
  python Figure6_SHAP_subject_cluster_metadata_analysis.py \
    --models imaging_only \
    --kinds node \
    --n-clusters 4 \
    --apoe4-col APOE4 \
    --sex-col Sex \
    --cognitive-status-col DX \
    --ad-col AD_status
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.cluster.hierarchy import linkage, leaves_list, fcluster
from scipy.spatial.distance import pdist, squareform, cdist
from scipy.stats import chi2_contingency, fisher_exact


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
    "ADNI": {
        "cohort_slug": "adni",
        "display": "ADNI",
        "cohort_dir": "ADNI",
        "file_prefix": "adni",
    },
    "ADRC": {
        "cohort_slug": "adrc",
        "display": "ADRC",
        "cohort_dir": "ADRC",
        "file_prefix": "adrc",
    },
    "HABS": {
        "cohort_slug": "habs",
        "display": "HABS",
        "cohort_dir": "HABS",
        "file_prefix": "habs",
    },
    "AD_DECODE": {
        "cohort_slug": "addecode",
        "display": "AD-DECODE",
        "cohort_dir": "AD_DECODE",
        "file_prefix": "ad_decode",
    },
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

DEFAULT_SUBJECT_ID_CANDIDATES = [
    "subject", "subject_id", "Subject_ID", "PTID", "ptid", "RID", "rid",
    "participant_id", "participant", "match_id", "regional_id", "runno",
    "MRI_Exam", "graph_id", "connectome_key", "connectome_full_key",
]

DEFAULT_APOE4_CANDIDATES = [
    "APOE4", "apoe4", "APOE4_carrier", "APOE4_carriage", "apoe4_carrier",
    "apoe4_carriage", "APOE_e4", "APOE_e4_carrier", "APOE", "apoe", "genotype",
    "APOE_genotype", "apoe_genotype",
]
DEFAULT_SEX_CANDIDATES = [
    "Sex", "sex", "SEX", "Gender", "gender", "GENDER", "PTGENDER",
]
DEFAULT_COG_CANDIDATES = [
    "cognitive_status", "Cognitive_Status", "DX", "dx", "diagnosis", "Diagnosis",
    "clinical_diagnosis", "Clinical_Diagnosis", "Group", "group", "Status", "status",
    "ResearchGroup", "research_group",
]
DEFAULT_AD_CANDIDATES = [
    "AD", "ad", "AD_status", "ad_status", "AD_diagnosis", "ad_diagnosis",
    "is_AD", "is_ad", "Diagnosis_AD", "diagnosis_ad", "dementia", "Dementia",
]


# =============================================================================
# CLI
# =============================================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Cluster subjects from SHAP profiles, then test APOE4/Sex/Cognitive/AD distributions."
    )
    p.add_argument("--out-base", default=str(DEFAULT_OUT_BASE))
    p.add_argument("--work", default=DEFAULT_WORK)
    p.add_argument("--cohorts", default=",".join(DEFAULT_COHORTS))
    p.add_argument("--models", "--feature-sets", dest="models", default="imaging_only")
    p.add_argument("--kinds", default="node,global", help="Comma list: global,node,edge")
    p.add_argument("--n-clusters", type=int, default=4)
    p.add_argument("--top-n-global", type=int, default=30)
    p.add_argument("--top-n-node", type=int, default=50)
    p.add_argument("--top-n-edge", type=int, default=50)
    p.add_argument("--value-col", default="SHAP_val", choices=["SHAP_val", "abs_SHAP"])
    p.add_argument("--standardize", type=int, default=1)
    p.add_argument("--linkage-method", default="ward", help="ward, average, complete, etc.")
    p.add_argument("--distance-metric", default="euclidean", help="euclidean for ward; correlation works with average/complete")
    p.add_argument("--n-permutations", type=int, default=1000)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--metadata-root", default=None, help="Default: $WORK/ines/results/harmonized")
    p.add_argument("--subject-id-col", default=None)
    p.add_argument("--apoe4-col", default=None)
    p.add_argument("--sex-col", default=None)
    p.add_argument("--cognitive-status-col", default=None)
    p.add_argument("--ad-col", default=None)
    p.add_argument("--output-subdir", default="subject_shap_clusters_metadata")
    p.add_argument("--dpi", type=int, default=350)
    return p.parse_args()


# =============================================================================
# Helpers
# =============================================================================
def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_filename(x) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", str(x))


def first_existing_col(df: pd.DataFrame, candidates: List[Optional[str]]) -> Optional[str]:
    lookup = {str(c).strip().lower(): c for c in df.columns}
    for c in candidates:
        if c is None:
            continue
        key = str(c).strip().lower()
        if key in lookup:
            return lookup[key]
    return None


def zscore_columns(matrix: pd.DataFrame) -> pd.DataFrame:
    x = matrix.astype(float).copy()
    mu = x.mean(axis=0, skipna=True)
    sd = x.std(axis=0, skipna=True).replace(0, np.nan)
    z = (x - mu) / sd
    return z.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def discover_models(out_base: Path, cohorts: List[str]) -> List[str]:
    found = set()
    for cohort in cohorts:
        cfg = COHORT_CONFIG[cohort]
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


def normalize_apoe4(x) -> str:
    if pd.isna(x):
        return "Missing"
    s = str(x).strip()
    sl = s.lower().replace(" ", "")
    if sl in {"1", "1.0", "true", "yes", "y", "carrier", "positive", "pos"}:
        return "Carrier"
    if sl in {"0", "0.0", "false", "no", "n", "non-carrier", "noncarrier", "negative", "neg"}:
        return "Non-carrier"
    if "e4" in sl or "ε4" in sl or re.search(r"(^|[^0-9])4([^0-9]|$)", sl) or sl in {"24", "34", "44"}:
        return "Carrier"
    if sl in {"22", "23", "33", "2/2", "2/3", "3/3", "e2/e2", "e2/e3", "e3/e3"}:
        return "Non-carrier"
    return s


def normalize_sex(x) -> str:
    if pd.isna(x):
        return "Missing"
    s = str(x).strip()
    sl = s.lower()
    if sl in {"m", "male", "1", "1.0"}:
        return "Male"
    if sl in {"f", "female", "0", "0.0", "2", "2.0"}:
        return "Female"
    return s


def normalize_cognitive_status(x) -> str:
    if pd.isna(x):
        return "Missing"
    s = str(x).strip()
    sl = s.lower().replace(" ", "")
    cn = {"cn", "control", "controls", "normal", "cognitivelynormal", "healthycontrol", "hc"}
    mci = {"mci", "emci", "lmci", "mildcognitiveimpairment"}
    ad = {"ad", "dementia", "alzheimers", "alzheimersdisease", "alzheimer'sdisease"}
    if sl in cn:
        return "CN"
    if sl in mci:
        return "MCI"
    if sl in ad:
        return "AD"
    return s


def normalize_ad_status(x) -> str:
    if pd.isna(x):
        return "Missing"
    s = str(x).strip()
    sl = s.lower().replace(" ", "")
    if sl in {"1", "1.0", "true", "yes", "y", "ad", "dementia", "alzheimers", "alzheimersdisease", "alzheimer'sdisease"}:
        return "AD"
    if sl in {"0", "0.0", "false", "no", "n", "cn", "control", "normal", "mci", "nonad", "non-ad"}:
        return "Non-AD"
    return s


def cramers_v(table: pd.DataFrame) -> float:
    if table.empty:
        return np.nan
    arr = table.to_numpy(dtype=float)
    if arr.sum() <= 0 or min(arr.shape) < 2:
        return np.nan
    try:
        chi2, _, _, _ = chi2_contingency(arr, correction=False)
    except Exception:
        return np.nan
    n = arr.sum()
    r, c = arr.shape
    return float(np.sqrt((chi2 / n) / max(min(r - 1, c - 1), 1)))


def permutation_pvalue(table: pd.DataFrame, n_perm: int, seed: int) -> float:
    """Permutation p-value for chi-square statistic by shuffling labels."""
    arr = table.to_numpy(dtype=int)
    if arr.sum() <= 0 or min(arr.shape) < 2:
        return np.nan
    try:
        obs, _, _, _ = chi2_contingency(arr, correction=False)
    except Exception:
        return np.nan
    row_labels = []
    col_labels = []
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            n = int(arr[i, j])
            row_labels.extend([i] * n)
            col_labels.extend([j] * n)
    row_labels = np.asarray(row_labels)
    col_labels = np.asarray(col_labels)
    if len(np.unique(row_labels)) < 2 or len(np.unique(col_labels)) < 2:
        return np.nan
    rng = np.random.default_rng(seed)
    ge = 0
    for _ in range(n_perm):
        shuffled = rng.permutation(col_labels)
        perm = pd.crosstab(row_labels, shuffled).reindex(index=range(arr.shape[0]), columns=range(arr.shape[1]), fill_value=0)
        try:
            stat, _, _, _ = chi2_contingency(perm.to_numpy(), correction=False)
        except Exception:
            stat = 0.0
        ge += int(stat >= obs)
    return float((ge + 1) / (n_perm + 1))


# =============================================================================
# Load SHAP subject-level data
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
    if "Edge" in df.columns:
        return df["Edge"].astype(str)
    if {"Structure_i", "Structure_j"}.issubset(df.columns):
        return df["Structure_i"].astype(str) + " -- " + df["Structure_j"].astype(str)
    if {"structure_i", "structure_j"}.issubset(df.columns):
        return df["structure_i"].astype(str) + " -- " + df["structure_j"].astype(str)
    if {"Node_i", "Node_j"}.issubset(df.columns):
        return df["Node_i"].astype(str) + " -- " + df["Node_j"].astype(str)
    return None


def load_subject_shap(out_base: Path, cohorts: List[str], model: str, kind: str, value_col: str) -> pd.DataFrame:
    frames = []
    logs = []
    for cohort in cohorts:
        cfg = COHORT_CONFIG[cohort]
        if kind == "global":
            subdir = "global_feature_shap"
            pattern = "global_feature_shap_subject_*.csv"
            prefix = "global_feature_shap_subject_"
            label_col = "feature_name"
        elif kind == "node":
            subdir = "node_feature_shap"
            pattern = "node_feature_shap_subject_*.csv"
            prefix = "node_feature_shap_subject_"
            label_col = "node_feature_label"
        elif kind == "edge":
            subdir = "edge_shap"
            pattern = "edge_shap_subject_*.csv"
            prefix = "edge_shap_subject_"
            label_col = "Edge"
        else:
            raise ValueError(f"Unknown SHAP kind: {kind}")

        shap_dir = out_base / cfg["cohort_slug"] / model / subdir
        files = sorted(shap_dir.glob(pattern))
        logs.append({"cohort": cohort, "kind": kind, "input_dir": str(shap_dir), "n_files": len(files)})
        for f in files:
            try:
                df = pd.read_csv(f)
            except Exception:
                continue
            if df.empty or "SHAP_val" not in df.columns:
                continue
            df = df.copy()
            if "abs_SHAP" not in df.columns:
                df["abs_SHAP"] = pd.to_numeric(df["SHAP_val"], errors="coerce").abs()
            if value_col not in df.columns:
                continue
            if kind == "node":
                lab = make_node_feature_label(df)
                if lab is None:
                    continue
                df[label_col] = lab
            elif kind == "edge":
                lab = make_edge_label(df)
                if lab is None:
                    continue
                df[label_col] = lab
            elif label_col not in df.columns:
                continue

            subject_safe = f.stem.replace(prefix, "")
            tmp = pd.DataFrame({
                "cohort": cohort,
                "subject_safe": subject_safe,
                "subject_key": cohort + "__" + subject_safe,
                "feature_label": df[label_col].astype(str),
                "shap_value": pd.to_numeric(df[value_col], errors="coerce"),
                "abs_shap": pd.to_numeric(df["abs_SHAP"], errors="coerce"),
            })
            tmp = tmp.dropna(subset=["feature_label", "shap_value"])
            frames.append(tmp)

    if not frames:
        return pd.DataFrame(), pd.DataFrame(logs)
    return pd.concat(frames, ignore_index=True), pd.DataFrame(logs)


def build_subject_feature_matrix(long_df: pd.DataFrame, top_n: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if long_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    # Rank features by mean absolute SHAP over all subjects/cohorts.
    ranked = (
        long_df.groupby("feature_label", as_index=False)
        .agg(
            mean_abs_shap=("abs_shap", "mean"),
            mean_signed_shap=("shap_value", "mean"),
            n_subjects=("subject_key", "nunique"),
            n_cohorts=("cohort", "nunique"),
        )
        .sort_values(["n_cohorts", "mean_abs_shap"], ascending=[False, False])
        .reset_index(drop=True)
    )
    keep = ranked.head(top_n)["feature_label"].tolist()
    use = long_df[long_df["feature_label"].isin(keep)].copy()
    matrix = use.pivot_table(index="subject_key", columns="feature_label", values="shap_value", aggfunc="mean")
    matrix = matrix.reindex(columns=keep).fillna(0.0)
    return matrix, ranked


# =============================================================================
# Metadata
# =============================================================================
def metadata_candidates(metadata_root: Path, cohort: str, model: str) -> List[Path]:
    cfg = COHORT_CONFIG[cohort]
    graph_dir = metadata_root / cfg["cohort_dir"] / "graphs" / model
    fp = cfg["file_prefix"]
    return [
        graph_dir / f"{fp}_metadata_all_aligned_raw.csv",
        graph_dir / f"{fp}_metadata_all_aligned.csv",
        graph_dir / f"{fp}_metadata_aligned_raw.csv",
        graph_dir / f"{fp}_metadata_aligned.csv",
    ]


def load_metadata(
    metadata_root: Path,
    cohorts: List[str],
    model: str,
    subject_id_col: Optional[str],
    apoe4_col: Optional[str],
    sex_col: Optional[str],
    cognitive_status_col: Optional[str],
    ad_col: Optional[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    frames = []
    logs = []
    for cohort in cohorts:
        paths = metadata_candidates(metadata_root, cohort, model)
        path = next((p for p in paths if p.exists()), None)
        if path is None:
            logs.append({"cohort": cohort, "metadata_path": "", "status": "missing"})
            continue
        try:
            df = pd.read_csv(path)
        except Exception as e:
            logs.append({"cohort": cohort, "metadata_path": str(path), "status": f"failed: {e}"})
            continue
        if df.empty:
            logs.append({"cohort": cohort, "metadata_path": str(path), "status": "empty"})
            continue

        sid_col = first_existing_col(df, [subject_id_col] + DEFAULT_SUBJECT_ID_CANDIDATES)
        ap_col = first_existing_col(df, [apoe4_col] + DEFAULT_APOE4_CANDIDATES)
        sx_col = first_existing_col(df, [sex_col] + DEFAULT_SEX_CANDIDATES)
        cg_col = first_existing_col(df, [cognitive_status_col] + DEFAULT_COG_CANDIDATES)
        ad_status_col = first_existing_col(df, [ad_col] + DEFAULT_AD_CANDIDATES)

        out = pd.DataFrame(index=df.index)
        out["cohort"] = cohort
        if sid_col is not None:
            out["subject_raw"] = df[sid_col].astype(str)
            out["subject_safe"] = out["subject_raw"].map(safe_filename)
            out["subject_key"] = cohort + "__" + out["subject_safe"]
        else:
            # Fallback: row order rarely matches SHAP filenames, but log clearly.
            out["subject_raw"] = np.arange(len(df)).astype(str)
            out["subject_safe"] = out["subject_raw"].map(safe_filename)
            out["subject_key"] = cohort + "__" + out["subject_safe"]

        out["apoe4_carriage"] = df[ap_col].map(normalize_apoe4) if ap_col else "Missing"
        out["sex"] = df[sx_col].map(normalize_sex) if sx_col else "Missing"
        out["cognitive_status"] = df[cg_col].map(normalize_cognitive_status) if cg_col else "Missing"
        if ad_status_col:
            out["ad_status"] = df[ad_status_col].map(normalize_ad_status)
        elif cg_col:
            out["ad_status"] = df[cg_col].map(normalize_ad_status)
        else:
            out["ad_status"] = "Missing"

        frames.append(out.drop_duplicates("subject_key"))
        logs.append({
            "cohort": cohort,
            "metadata_path": str(path),
            "status": "loaded",
            "n_rows": len(df),
            "subject_id_col": sid_col or "",
            "apoe4_col": ap_col or "",
            "sex_col": sx_col or "",
            "cognitive_status_col": cg_col or "",
            "ad_col": ad_status_col or "derived_from_cognitive_status" if cg_col and not ad_status_col else "",
        })

    if not frames:
        return pd.DataFrame(), pd.DataFrame(logs)
    return pd.concat(frames, ignore_index=True), pd.DataFrame(logs)


# =============================================================================
# Clustering and outputs
# =============================================================================
def cluster_subjects(matrix_scaled: pd.DataFrame, n_clusters: int, method: str, metric: str) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    if matrix_scaled.shape[0] < 2:
        raise ValueError("Need at least two subjects to cluster.")
    if method == "ward" and metric != "euclidean":
        print("WARNING: Ward linkage requires Euclidean distance. Switching metric to euclidean.")
        metric = "euclidean"
    condensed = pdist(matrix_scaled.to_numpy(dtype=float), metric=metric)
    if not np.isfinite(condensed).all():
        condensed = np.nan_to_num(condensed, nan=0.0, posinf=0.0, neginf=0.0)
    Z = linkage(condensed, method=method)
    order = leaves_list(Z)
    labels = fcluster(Z, t=n_clusters, criterion="maxclust")
    clusters = pd.DataFrame({"subject_key": matrix_scaled.index, "shap_cluster": labels.astype(int)})
    return clusters, Z, order


def save_heatmap(matrix_scaled: pd.DataFrame, subject_info: pd.DataFrame, order: np.ndarray, out_png: Path, out_pdf: Path, title: str, dpi: int):
    ordered = matrix_scaled.iloc[order, :]
    row_info = subject_info.set_index("subject_key").reindex(ordered.index)

    # Compact heatmap for many subjects/features.
    fig_h = max(8, min(28, 0.08 * ordered.shape[0] + 4))
    fig_w = max(10, min(30, 0.20 * ordered.shape[1] + 6))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    sns.heatmap(
        ordered,
        ax=ax,
        cmap="vlag",
        center=0,
        xticklabels=True if ordered.shape[1] <= 80 else False,
        yticklabels=False,
        cbar_kws={"label": "Column-standardized SHAP value"},
    )
    ax.set_title(title)
    ax.set_xlabel("SHAP feature")
    ax.set_ylabel("Subjects ordered by SHAP-profile clustering")
    fig.tight_layout()
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def cluster_distance_outputs(matrix_scaled: pd.DataFrame, clusters: pd.DataFrame, out_prefix: Path) -> List[str]:
    outputs = []
    labels = clusters.set_index("subject_key").loc[matrix_scaled.index, "shap_cluster"]
    X = matrix_scaled.to_numpy(dtype=float)
    labs = sorted(labels.unique())

    centroids = []
    for lab in labs:
        sub = matrix_scaled.loc[labels[labels == lab].index]
        centroids.append(sub.mean(axis=0))
    centroids = pd.DataFrame(centroids, index=[f"cluster_{x}" for x in labs], columns=matrix_scaled.columns)
    path = out_prefix.with_suffix("_cluster_centroids_scaled.csv")
    centroids.to_csv(path)
    outputs.append(str(path))

    eu = pd.DataFrame(
        squareform(pdist(centroids.to_numpy(dtype=float), metric="euclidean")),
        index=centroids.index,
        columns=centroids.index,
    )
    path = out_prefix.with_suffix("_cluster_centroid_distance_euclidean.csv")
    eu.to_csv(path)
    outputs.append(str(path))

    corr_dist = pd.DataFrame(
        squareform(pdist(centroids.to_numpy(dtype=float), metric="correlation")),
        index=centroids.index,
        columns=centroids.index,
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    path = out_prefix.with_suffix("_cluster_centroid_distance_correlation.csv")
    corr_dist.to_csv(path)
    outputs.append(str(path))

    compact = []
    for lab in labs:
        idx = labels[labels == lab].index
        sub = matrix_scaled.loc[idx]
        centroid = centroids.loc[f"cluster_{lab}"].to_numpy(dtype=float).reshape(1, -1)
        d = cdist(sub.to_numpy(dtype=float), centroid, metric="euclidean").ravel()
        compact.append({
            "shap_cluster": lab,
            "n_subjects": len(idx),
            "mean_distance_to_centroid": float(np.mean(d)) if len(d) else np.nan,
            "median_distance_to_centroid": float(np.median(d)) if len(d) else np.nan,
            "sd_distance_to_centroid": float(np.std(d, ddof=1)) if len(d) > 1 else np.nan,
        })
    path = out_prefix.with_suffix("_cluster_compactness.csv")
    pd.DataFrame(compact).to_csv(path, index=False)
    outputs.append(str(path))

    pair_rows = []
    for i, a in enumerate(labs):
        for b in labs[i + 1:]:
            Xa = matrix_scaled.loc[labels[labels == a].index].to_numpy(dtype=float)
            Xb = matrix_scaled.loc[labels[labels == b].index].to_numpy(dtype=float)
            d = cdist(Xa, Xb, metric="euclidean").ravel()
            pair_rows.append({
                "cluster_a": a,
                "cluster_b": b,
                "n_pairs": len(d),
                "mean_between_subject_distance": float(np.mean(d)) if len(d) else np.nan,
                "median_between_subject_distance": float(np.median(d)) if len(d) else np.nan,
                "min_between_subject_distance": float(np.min(d)) if len(d) else np.nan,
                "max_between_subject_distance": float(np.max(d)) if len(d) else np.nan,
            })
    path = out_prefix.with_suffix("_cluster_pair_distance_summary.csv")
    pd.DataFrame(pair_rows).to_csv(path, index=False)
    outputs.append(str(path))
    return outputs


def plot_distribution(props: pd.DataFrame, out_png: Path, title: str, dpi: int):
    ax = props.plot(kind="bar", stacked=True, figsize=(9, 5))
    ax.set_title(title)
    ax.set_xlabel("SHAP cluster")
    ax.set_ylabel("Proportion")
    ax.legend(title="Category", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close()


def distribution_stats(subject_meta: pd.DataFrame, variable: str, out_prefix: Path, n_perm: int, seed: int, dpi: int) -> List[str]:
    outputs = []
    df = subject_meta[["shap_cluster", variable]].copy()
    df[variable] = df[variable].fillna("Missing").astype(str)
    df = df[df[variable] != "Missing"].copy()
    if df.empty or df["shap_cluster"].nunique() < 2 or df[variable].nunique() < 2:
        counts = pd.crosstab(subject_meta["shap_cluster"], subject_meta[variable].fillna("Missing"))
        path = out_prefix.with_suffix(f"_cluster_distribution_{variable}_counts.csv")
        counts.to_csv(path)
        outputs.append(str(path))
        stats = pd.DataFrame([{
            "variable": variable,
            "status": "insufficient_nonmissing_categories",
            "n_subjects_nonmissing": len(df),
            "n_clusters": df["shap_cluster"].nunique() if not df.empty else 0,
            "n_categories": df[variable].nunique() if not df.empty else 0,
        }])
        path = out_prefix.with_suffix(f"_cluster_distribution_{variable}_stats.csv")
        stats.to_csv(path, index=False)
        outputs.append(str(path))
        return outputs

    counts = pd.crosstab(df["shap_cluster"], df[variable])
    props = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)

    path = out_prefix.with_suffix(f"_cluster_distribution_{variable}_counts.csv")
    counts.to_csv(path)
    outputs.append(str(path))
    path = out_prefix.with_suffix(f"_cluster_distribution_{variable}_proportions.csv")
    props.to_csv(path)
    outputs.append(str(path))

    try:
        chi2, p_chi, dof, expected = chi2_contingency(counts.to_numpy(), correction=False)
    except Exception:
        chi2, p_chi, dof = np.nan, np.nan, np.nan

    fisher_p = np.nan
    odds_ratio = np.nan
    if counts.shape == (2, 2):
        try:
            odds_ratio, fisher_p = fisher_exact(counts.to_numpy())
        except Exception:
            pass

    perm_p = permutation_pvalue(counts, n_perm=n_perm, seed=seed)
    stats = pd.DataFrame([{
        "variable": variable,
        "status": "ok",
        "n_subjects_nonmissing": int(counts.to_numpy().sum()),
        "n_clusters": int(counts.shape[0]),
        "n_categories": int(counts.shape[1]),
        "chi_square": chi2,
        "chi_square_p": p_chi,
        "degrees_of_freedom": dof,
        "fisher_exact_p_if_2x2": fisher_p,
        "odds_ratio_if_2x2": odds_ratio,
        "permutation_p": perm_p,
        "cramers_v": cramers_v(counts),
    }])
    path = out_prefix.with_suffix(f"_cluster_distribution_{variable}_stats.csv")
    stats.to_csv(path, index=False)
    outputs.append(str(path))

    path = out_prefix.with_suffix(f"_cluster_distribution_{variable}.png")
    plot_distribution(props, path, f"Distribution of {variable} by SHAP cluster", dpi=dpi)
    outputs.append(str(path))
    return outputs


def analyze_one_kind(
    out_base: Path,
    metadata_root: Path,
    cohorts: List[str],
    model: str,
    kind: str,
    args: argparse.Namespace,
) -> List[str]:
    outdir = ensure_dir(out_base / "cross_cohort_aggregated" / model / args.output_subdir)
    prefix = outdir / f"cross_cohort_{model}_{kind}_SHAP_subject_clusters_k{args.n_clusters}"
    outputs = []

    long_df, load_log = load_subject_shap(out_base, cohorts, model, kind, args.value_col)
    path = prefix.with_suffix("_input_load_log.csv")
    load_log.to_csv(path, index=False)
    outputs.append(str(path))
    if long_df.empty:
        print(f"No {kind} SHAP subject files found for model={model}")
        return outputs

    top_n = {"global": args.top_n_global, "node": args.top_n_node, "edge": args.top_n_edge}[kind]
    matrix_raw, ranked = build_subject_feature_matrix(long_df, top_n=top_n)
    if matrix_raw.shape[0] < args.n_clusters:
        print(f"Not enough subjects for {kind}: n_subjects={matrix_raw.shape[0]}, n_clusters={args.n_clusters}")
        return outputs

    path = prefix.with_suffix("_subject_level_long.csv")
    long_df.to_csv(path, index=False)
    outputs.append(str(path))
    path = prefix.with_suffix("_ranked_SHAP_features.csv")
    ranked.to_csv(path, index=False)
    outputs.append(str(path))
    path = prefix.with_suffix("_matrix_raw.csv")
    matrix_raw.to_csv(path)
    outputs.append(str(path))

    matrix_scaled = zscore_columns(matrix_raw) if int(args.standardize) else matrix_raw.astype(float).copy()
    path = prefix.with_suffix("_matrix_scaled.csv")
    matrix_scaled.to_csv(path)
    outputs.append(str(path))

    clusters, Z, order = cluster_subjects(
        matrix_scaled,
        n_clusters=args.n_clusters,
        method=args.linkage_method,
        metric=args.distance_metric,
    )
    meta, meta_log = load_metadata(
        metadata_root=metadata_root,
        cohorts=cohorts,
        model=model,
        subject_id_col=args.subject_id_col,
        apoe4_col=args.apoe4_col,
        sex_col=args.sex_col,
        cognitive_status_col=args.cognitive_status_col,
        ad_col=args.ad_col,
    )
    path = prefix.with_suffix("_metadata_load_log.csv")
    meta_log.to_csv(path, index=False)
    outputs.append(str(path))

    subject_info = clusters.copy()
    subject_info["cohort"] = subject_info["subject_key"].str.split("__", n=1).str[0]
    subject_info["subject_safe"] = subject_info["subject_key"].str.split("__", n=1).str[1]
    if not meta.empty:
        subject_info = subject_info.merge(
            meta[["subject_key", "subject_raw", "apoe4_carriage", "sex", "cognitive_status", "ad_status"]],
            on="subject_key",
            how="left",
        )
    for col in ["subject_raw", "apoe4_carriage", "sex", "cognitive_status", "ad_status"]:
        if col not in subject_info.columns:
            subject_info[col] = "Missing"
        subject_info[col] = subject_info[col].fillna("Missing")

    path = prefix.with_suffix("_subject_clusters_with_metadata.csv")
    subject_info.sort_values(["shap_cluster", "cohort", "subject_safe"]).to_csv(path, index=False)
    outputs.append(str(path))

    # Clustered matrices and order files.
    ordered_raw = matrix_raw.iloc[order, :]
    ordered_scaled = matrix_scaled.iloc[order, :]
    path = prefix.with_suffix("_clustered_matrix_raw.csv")
    ordered_raw.to_csv(path)
    outputs.append(str(path))
    path = prefix.with_suffix("_clustered_matrix_scaled.csv")
    ordered_scaled.to_csv(path)
    outputs.append(str(path))
    row_order = pd.DataFrame({
        "order_index": np.arange(len(order)),
        "subject_key": matrix_scaled.index[order],
    }).merge(subject_info, on="subject_key", how="left")
    path = prefix.with_suffix("_row_order_clusters_metadata.csv")
    row_order.to_csv(path, index=False)
    outputs.append(str(path))
    col_order = pd.DataFrame({"column_order": np.arange(matrix_scaled.shape[1]), "feature_label": matrix_scaled.columns})
    path = prefix.with_suffix("_column_order.csv")
    col_order.to_csv(path, index=False)
    outputs.append(str(path))

    # Heatmap.
    png = prefix.with_suffix("_hierarchical_heatmap.png")
    pdf = prefix.with_suffix("_hierarchical_heatmap.pdf")
    save_heatmap(
        matrix_scaled,
        subject_info=subject_info,
        order=order,
        out_png=png,
        out_pdf=pdf,
        title=f"Cross-cohort {model} {kind} SHAP subject clusters (k={args.n_clusters})",
        dpi=args.dpi,
    )
    outputs.extend([str(png), str(pdf)])

    # Cluster distance analyses.
    outputs.extend(cluster_distance_outputs(matrix_scaled, clusters, prefix))

    # Primary post-hoc clinical/genetic distributions. Cohort is deliberately not primary.
    all_stats = []
    for variable in ["apoe4_carriage", "sex", "cognitive_status", "ad_status"]:
        before = set(outputs)
        outputs.extend(distribution_stats(subject_info, variable, prefix, args.n_permutations, args.random_state, args.dpi))
        stats_file = prefix.with_suffix(f"_cluster_distribution_{variable}_stats.csv")
        if stats_file.exists():
            try:
                all_stats.append(pd.read_csv(stats_file))
            except Exception:
                pass

    if all_stats:
        path = prefix.with_suffix("_cluster_distribution_all_variables_stats.csv")
        pd.concat(all_stats, ignore_index=True).to_csv(path, index=False)
        outputs.append(str(path))

    # Cohort composition is diagnostic only, not the target result.
    cohort_counts = pd.crosstab(subject_info["shap_cluster"], subject_info["cohort"])
    path = prefix.with_suffix("_diagnostic_cohort_distribution_counts.csv")
    cohort_counts.to_csv(path)
    outputs.append(str(path))

    manifest = pd.DataFrame([{
        "model": model,
        "kind": kind,
        "n_clusters": args.n_clusters,
        "n_subjects": matrix_scaled.shape[0],
        "n_features": matrix_scaled.shape[1],
        "output_dir": str(outdir),
        "outputs": ";".join(outputs),
    }])
    path = prefix.with_suffix("_manifest.csv")
    manifest.to_csv(path, index=False)
    outputs.append(str(path))
    return outputs


# =============================================================================
# Main
# =============================================================================
def main():
    args = parse_args()
    out_base = Path(args.out_base).expanduser().resolve()
    work = Path(args.work).expanduser().resolve()
    metadata_root = Path(args.metadata_root).expanduser().resolve() if args.metadata_root else work / "ines/results/harmonized"
    cohorts = [x.strip() for x in args.cohorts.split(",") if x.strip()]
    models = [x.strip() for x in args.models.split(",") if x.strip()]
    kinds = [x.strip().lower() for x in args.kinds.split(",") if x.strip()]

    if models == ["auto"]:
        models = discover_models(out_base, cohorts)
    if not models:
        raise SystemExit(f"No models specified/found under {out_base}")

    print("\n" + "=" * 100)
    print("SHAP SUBJECT CLUSTERING + POST-HOC APOE4/SEX/COGNITIVE/AD DISTRIBUTIONS")
    print("=" * 100)
    print(f"OUT_BASE:      {out_base}")
    print(f"METADATA_ROOT: {metadata_root}")
    print(f"Cohorts:       {', '.join(cohorts)}")
    print(f"Models:        {', '.join(models)}")
    print(f"Kinds:         {', '.join(kinds)}")
    print(f"n_clusters:    {args.n_clusters}")
    print(f"Output subdir: {args.output_subdir}")
    print("=" * 100)

    all_outputs = []
    for model in models:
        for kind in kinds:
            print(f"\n--- model={model}; kind={kind} ---")
            outs = analyze_one_kind(out_base, metadata_root, cohorts, model, kind, args)
            all_outputs.extend(outs)
            print(f"Wrote {len(outs)} outputs")

    master_dir = ensure_dir(out_base / "cross_cohort_aggregated" / "cluster_metadata_manifests")
    master_path = master_dir / "SHAP_subject_cluster_metadata_analysis_master_manifest.csv"
    pd.DataFrame([{
        "out_base": str(out_base),
        "metadata_root": str(metadata_root),
        "models": ",".join(models),
        "kinds": ",".join(kinds),
        "n_clusters": args.n_clusters,
        "outputs": ";".join(all_outputs),
    }]).to_csv(master_path, index=False)

    print("\nDone.")
    print(f"Master manifest: {master_path}")
    print("Primary output folders:")
    for model in models:
        print(f"  {out_base / 'cross_cohort_aggregated' / model / args.output_subdir}")


if __name__ == "__main__":
    main()
