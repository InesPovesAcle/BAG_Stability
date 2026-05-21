#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate Figure 7: cross-cohort replication / synthesis analysis for cBAG.

Purpose
-------
This script brings together the cohort-specific biological-validation outputs
for ADNI, ADRC, HABS, and AD_DECODE and creates a manuscript-ready synthesis
figure showing whether cBAG effects replicate across cohorts.

It does NOT retrain models and does NOT modify validation outputs.

Expected input files
--------------------
For each cohort and feature set:

    <RESULTS_ROOT>/<BrainAgePrediction...>/ablation_<feature_set>/validation_figures/
        subject_level_validation_input.csv

These are the same inputs used by the existing Figure 4 / Figure 5 scripts.

Default paths
-------------
BASE_DIR:
    /mnt/newStor/paros/paros_WORK/ines/results/
    BrainAgeValidation_AllCohorts_BAGBiasCorr_OOFGlobal_BiologicalValidation

RESULTS_ROOT defaults to BASE_DIR.parent:
    /mnt/newStor/paros/paros_WORK/ines/results

Default output directory:
    <BASE_DIR>/Figure7_cross_cohort_replication/

Outputs
-------
Main figure:
    Figure7_cross_cohort_replication.png
    Figure7_cross_cohort_replication.pdf

Tables:
    figure7_dataset_summary.csv
    figure7_model_metric_summary.csv
    figure7_endpoint_correlation_stats.csv
    figure7_replication_summary.csv
    figure7_manifest.csv

Run
---
    python make_figure7_cross_cohort_analysis.py

Optional
--------
    python make_figure7_cross_cohort_analysis.py \
        --base-dir /mnt/newStor/paros/paros_WORK/ines/results/BrainAgeValidation_AllCohorts_BAGBiasCorr_OOFGlobal_BiologicalValidation \
        --results-root /mnt/newStor/paros/paros_WORK/ines/results \
        --main-feature-set imaging_only \
        --formats png,pdf
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import combine_pvalues, pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =============================================================================
# Defaults
# =============================================================================
BASE_DIR = (
    "/mnt/newStor/paros/paros_WORK/ines/results/"
    "BrainAgeValidation_AllCohorts_BAGBiasCorr_OOFGlobal_BiologicalValidation"
)

DEFAULT_OUTDIR_NAME = "Figure7_cross_cohort_replication"
DEFAULT_MAIN_FEATURE_SET = "imaging_only"
DEFAULT_FEATURE_SETS = [
    "imaging_only",
    "imaging_demographics",
    "imaging_biomarkers",
    "full",
    "full_no_cardiovascular",
]

RESULTS_DIR_MAP = {
    "ADNI": "BrainAgePredictionADNI_stratified_groupcv_targetnorm_bagbiascorr_oofglobal",
    "ADRC": "BrainAgePredictionADRC_stratified_groupcv_targetnorm_bagbiascorr_oofglobal",
    "HABS": "BrainAgePredictionHABS_stratified_groupcv_targetnorm_bagbiascorr_oofglobal",
    "AD_DECODE": "BrainAgePredictionADDECODE_stratified_groupcv_targetnorm_bagbiascorr_oofglobal",
}

DEFAULT_COHORTS = ["ADNI", "ADRC", "HABS", "AD_DECODE"]

SENTINEL_VALUES = {
    -999999, -888888, -777777,
    -99999, -88888, -77777,
    -9999, -8888, -7777,
    -999, -888, -777,
    999, 888, 777,
    9999, 8888, 7777,
    99999, 88888, 77777,
    999999, 888888, 777777,
}

CBAG_PRIORITY = [
    "cBAG_oof_global_raw_clean",
    "cBAG_oof_global",
    "cBAG_foldwise_raw_clean",
    "cBAG_foldwise",
    "cBAG_raw_clean",
    "cBAG",
    "cBAG_global_raw_clean",
    "cBAG_global",
    "BAG_raw_clean",
    "BAG",
    "BAG_raw",
]

AGE_PRIORITY = [
    "Real_Age_raw_clean",
    "Real_Age",
    "Chronological_Age_raw_clean",
    "Chronological_Age",
    "Age_raw_clean",
    "Age",
    "age",
    "VISIT_AGE",
    "AGE",
]

PRED_AGE_PRIORITY = [
    "Predicted_Age_raw_clean",
    "Predicted_Age",
    "predicted_age_raw_clean",
    "predicted_age",
    "Pred_Age_raw_clean",
    "Pred_Age",
    "y_pred_raw_clean",
    "y_pred",
    "prediction_raw_clean",
    "prediction",
]

# Endpoint groups for cross-cohort replication.
# These names are deliberately broad. The script chooses the first usable column
# present in each cohort table and falls back to conservative keyword matching.
ENDPOINTS: Dict[str, Dict[str, object]] = {
    "Cognition": {
        "domain": "Clinical",
        "candidates": [
            "Global_Cognition_Composite",
            "Global_Cognition_Composite_resid",
            "cognition_composite",
            "Cognition_Composite",
            "Memory_Composite",
            "Memory_Composite_resid",
            "Executive_Function_Composite",
            "Executive_Function_Composite_resid",
            "Processing_Speed_Composite",
            "Language_Composite",
            "Visuospatial_Composite",
            "MMSE_total",
            "MOCA_total_corrected",
            "MOCA_total",
            "ADAS_total",
            "CDRSB",
            "CDGLOBAL",
        ],
        "prefer_keywords": ["cognition", "memory", "executive", "mmse", "moca", "adas", "cdr"],
        "avoid_keywords": ["status", "group", "diagnosis", "dx", "apoe", "cluster"],
        "value_range": None,
    },
    "Hippocampal volume": {
        "domain": "Neuroimaging",
        "candidates": [
            "Hippocampus_Total_pct",
            "Hippocampus_Total",
            "Hippocampus_volume",
            "Hippocampal_volume",
            "hippocampal_volume",
            "HC_volume",
            "Left_Hippocampus_volume",
            "Right_Hippocampus_volume",
        ],
        "prefer_keywords": ["hipp", "hippocampus", "hippocampal"],
        "avoid_keywords": ["fa", "md", "rd", "ad", "diagnosis", "status", "cluster"],
        "value_range": None,
    },
    "Hippocampal FA": {
        "domain": "Neuroimaging",
        "candidates": [
            "Hippocampus_FA_Mean",
            "Hippocampus_FA_Total",
            "Hippocampal_FA",
            "hippocampal_FA",
            "HC_FA",
            "Left_Hippocampus_FA",
            "Right_Hippocampus_FA",
        ],
        "prefer_keywords": ["hipp", "hippocampus", "hippocampal", "fa"],
        "avoid_keywords": ["volume", "vol", "thickness", "area", "diagnosis", "status"],
        "value_range": (0.0, 1.0),
    },
    "Whole-brain volume": {
        "domain": "Neuroimaging",
        "candidates": [
            "Total_Brain_volume",
            "TotalBrainVolume",
            "Brain_Volume",
            "brain_volume",
            "Volume_mean",
            "Volume_median",
            "GM_volume",
            "WM_volume",
        ],
        "prefer_keywords": ["brain", "volume", "vol"],
        "avoid_keywords": ["hipp", "hippocampus", "fa", "md", "rd", "ad", "status", "cluster"],
        "value_range": None,
    },
    "Global FA": {
        "domain": "Neuroimaging",
        "candidates": [
            "FA_mean",
            "FA_median",
            "Global_FA",
            "global_FA",
            "Mean_FA",
            "Median_FA",
        ],
        "prefer_keywords": ["fa"],
        "avoid_keywords": ["hipp", "hippocampus", "volume", "vol", "status", "cluster"],
        "value_range": (0.0, 1.0),
    },
    "Graph/network metric": {
        "domain": "Network",
        "candidates": [
            "global_efficiency",
            "Global_Efficiency",
            "clustering_coefficient",
            "Clustering_Coefficient",
            "modularity",
            "Modularity",
            "characteristic_path_length",
            "Characteristic_Path_Length",
            "small_worldness",
            "Small_Worldness",
        ],
        "prefer_keywords": ["efficiency", "clustering", "modularity", "path", "network", "graph"],
        "avoid_keywords": ["cluster_id", "clusterlabel", "diagnosis", "status"],
        "value_range": None,
    },
    "Amyloid / Aβ": {
        "domain": "Biomarker",
        "candidates": [
            "ABETA42",
            "ABETA40",
            "ABETA42_40",
            "ABETA42_ABETA40",
            "Aβ42",
            "Aβ40",
            "Amyloid",
            "amyloid",
            "Centiloid",
            "centiloid",
        ],
        "prefer_keywords": ["abeta", "amyloid", "centiloid"],
        "avoid_keywords": ["status", "positive", "pos", "binary", "dx", "cluster"],
        "value_range": None,
    },
    "pTau": {
        "domain": "Biomarker",
        "candidates": [
            "PLASMA_PTAU217",
            "ptau217",
            "PTAU217",
            "pTau217",
            "PLASMA_PTAU181",
            "ptau181",
            "PTAU181",
            "pTau181",
            "PTAU",
            "pTau",
            "TAU",
        ],
        "prefer_keywords": ["ptau", "tau"],
        "avoid_keywords": ["status", "positive", "pos", "binary", "dx", "cluster"],
        "value_range": None,
    },
    "GFAP": {
        "domain": "Biomarker",
        "candidates": ["GFAP", "gfap", "PLASMA_GFAP", "plasma_GFAP"],
        "prefer_keywords": ["gfap"],
        "avoid_keywords": ["status", "positive", "pos", "binary", "dx", "cluster"],
        "value_range": None,
    },
    "NfL": {
        "domain": "Biomarker",
        "candidates": ["NfL", "NFL", "nfl", "PLASMA_NFL", "plasma_NfL", "NEUROFILAMENT_LIGHT"],
        "prefer_keywords": ["nfl", "nfl", "neurofilament"],
        "avoid_keywords": ["status", "positive", "pos", "binary", "dx", "cluster"],
        "value_range": None,
    },
    "AD_DECODE transcriptomic PC": {
        "domain": "Transcriptomic",
        "candidates": [
            "PC1",
            "PC2",
            "PC3",
            "transcriptomic_PC1",
            "transcriptomic_PC2",
            "transcriptomic_PC3",
            "RNA_PC1",
            "RNA_PC2",
            "RNA_PC3",
            "gene_PC1",
            "gene_PC2",
            "gene_PC3",
        ],
        "prefer_keywords": ["pc", "transcript", "rna", "gene"],
        "avoid_keywords": ["apoe", "diagnosis", "status", "cluster"],
        "value_range": None,
    },
}


# =============================================================================
# Utilities
# =============================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Figure 7 cross-cohort cBAG replication/synthesis analysis."
    )
    parser.add_argument(
        "--base-dir",
        default=BASE_DIR,
        help="Combined validation directory. Default is the project direct path.",
    )
    parser.add_argument(
        "--results-root",
        default=None,
        help="Root containing per-cohort BrainAgePrediction... folders. Default: <base-dir>/..",
    )
    parser.add_argument(
        "--outdir",
        default=None,
        help=f"Output directory. Default: <base-dir>/{DEFAULT_OUTDIR_NAME}",
    )
    parser.add_argument(
        "--main-feature-set",
        default=DEFAULT_MAIN_FEATURE_SET,
        help="Feature set used for endpoint replication panels. Default: imaging_only",
    )
    parser.add_argument(
        "--feature-sets",
        default=",".join(DEFAULT_FEATURE_SETS),
        help="Comma-separated feature sets for model-performance panel. Default: all final ablations.",
    )
    parser.add_argument(
        "--cohorts",
        default=",".join(DEFAULT_COHORTS),
        help="Comma-separated cohorts. Default: ADNI,ADRC,HABS,AD_DECODE",
    )
    parser.add_argument(
        "--formats",
        default="png,pdf",
        help="Comma-separated output formats. Default: png,pdf",
    )
    parser.add_argument("--dpi", type=int, default=450, help="Raster output DPI. Default: 450")
    parser.add_argument("--min-n", type=int, default=8, help="Minimum complete cases for correlations. Default: 8")
    parser.add_argument(
        "--metric",
        default="MAE",
        choices=["MAE", "RMSE", "R2", "r", "n"],
        help="Metric shown in Panel A. Uses predicted age if available. Default: MAE",
    )
    parser.add_argument(
        "--max-endpoints-forest",
        type=int,
        default=10,
        help="Maximum endpoints shown in forest panel. Default: 10",
    )
    parser.add_argument(
        "--endpoint-order",
        default="effect",
        choices=["effect", "original"],
        help="Order endpoints by absolute median effect or original endpoint list. Default: effect",
    )
    return parser.parse_args()


def clean_numeric(series: pd.Series) -> pd.Series:
    out = pd.to_numeric(series, errors="coerce").copy()
    for value in SENTINEL_VALUES:
        out = out.mask(out == value, np.nan)
    return out


def first_existing(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def usable_n(series: pd.Series, value_range: Optional[Tuple[float, float]] = None) -> int:
    x = clean_numeric(series)
    if value_range is not None:
        lo, hi = value_range
        x = x.where((x >= lo) & (x <= hi))
    return int(x.notna().sum())


def candidate_tokens(candidates: Sequence[str]) -> List[str]:
    tokens: List[str] = []
    for c in candidates:
        tokens.extend([t for t in re.split(r"[_\s/\-]+", str(c).lower()) if len(t) >= 3])
    return sorted(set(tokens), key=len, reverse=True)


def find_numeric_col(
    df: pd.DataFrame,
    candidates: Sequence[str],
    min_n: int = 8,
    avoid_keywords: Optional[Sequence[str]] = None,
    prefer_keywords: Optional[Sequence[str]] = None,
    value_range: Optional[Tuple[float, float]] = None,
) -> Optional[str]:
    """
    Robust but conservative numeric column finder.

    Order:
      1. Exact candidate names and *_raw_clean variants.
      2. Keyword fallback using tokens from candidates and prefer_keywords.
    """
    avoid_keywords = [x.lower() for x in (avoid_keywords or [])]
    prefer_keywords = [x.lower() for x in (prefer_keywords or [])]

    def eligible(col: str) -> bool:
        low = str(col).lower()
        if any(bad in low for bad in avoid_keywords):
            return False
        return usable_n(df[col], value_range=value_range) >= min_n

    expanded: List[str] = []
    for c in candidates:
        expanded.extend([f"{c}_raw_clean", c])

    for col in expanded:
        if col in df.columns and eligible(col):
            return col

    tokens = candidate_tokens(candidates)
    fallback: List[Tuple[int, str]] = []
    for col in df.columns:
        low = str(col).lower()
        if not eligible(col):
            continue
        token_hit = any(tok in low for tok in tokens)
        prefer_hit = any(tok in low for tok in prefer_keywords)
        if not token_hit and not prefer_hit:
            continue
        score = int(token_hit) + 2 * sum(tok in low for tok in prefer_keywords)
        fallback.append((score, col))

    if not fallback:
        return None

    fallback.sort(key=lambda x: (-x[0], str(x[1]).lower()))
    return fallback[0][1]


def validation_dir(results_root: Path, cohort: str, feature_set: str) -> Path:
    if cohort not in RESULTS_DIR_MAP:
        raise ValueError(f"Unknown cohort: {cohort}. Known cohorts: {sorted(RESULTS_DIR_MAP)}")
    return results_root / RESULTS_DIR_MAP[cohort] / f"ablation_{feature_set}" / "validation_figures"


def load_subject_level_tables(
    results_root: Path,
    cohorts: Sequence[str],
    feature_set: str,
    min_n: int = 8,
) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []

    for cohort in cohorts:
        path = validation_dir(results_root, cohort, feature_set) / "subject_level_validation_input.csv"
        if not path.exists():
            print(f"[WARN] Missing input table: {path}")
            continue

        df = pd.read_csv(path, low_memory=False)
        df["cohort"] = cohort
        df["feature_set"] = feature_set
        df["_input_path"] = str(path)

        cbag_col = first_existing(df, CBAG_PRIORITY)
        if cbag_col is None:
            print(f"[WARN] No usable cBAG/BAG column found in {path}")
            continue

        df["_cbag"] = clean_numeric(df[cbag_col])
        df["_cbag_source"] = cbag_col

        age_col = first_existing(df, AGE_PRIORITY)
        pred_col = first_existing(df, PRED_AGE_PRIORITY)
        if age_col is not None:
            df["_age"] = clean_numeric(df[age_col])
            df["_age_source"] = age_col
        else:
            df["_age"] = np.nan
            df["_age_source"] = ""

        if pred_col is not None:
            df["_pred_age"] = clean_numeric(df[pred_col])
            df["_pred_age_source"] = pred_col
        else:
            df["_pred_age"] = np.nan
            df["_pred_age_source"] = ""

        frames.append(df)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True, sort=False)
    out = out.dropna(subset=["_cbag"]).copy()
    return out


def nice_label(name: Optional[str]) -> str:
    if name is None:
        return ""
    mapping = {
        "_cbag": "cBAG",
        "_age": "Age",
        "_pred_age": "Predicted age",
        "imaging_only": "Imaging only",
        "imaging_demographics": "Imaging + demographics",
        "imaging_biomarkers": "Imaging + biomarkers",
        "full": "Full",
        "full_no_cardiovascular": "Full, no cardiovascular",
    }
    if name in mapping:
        return mapping[name]
    raw = str(name).replace("_raw_clean", "")
    raw = raw.replace("cBAG_oof_global", "cBAG")
    raw = raw.replace("PLASMA_", "")
    raw = raw.replace("ABETA", "Aβ")
    raw = raw.replace("PTAU", "pTau")
    raw = raw.replace("ptau", "pTau")
    raw = raw.replace("NFL", "NfL")
    raw = raw.replace("_", " ")
    raw = re.sub(r"\s+", " ", raw).strip()
    return raw


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.12,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=14,
        fontweight="bold",
        va="top",
        ha="left",
    )


def save_all(fig: plt.Figure, outbase: Path, formats: Sequence[str], dpi: int) -> List[str]:
    outbase.parent.mkdir(parents=True, exist_ok=True)
    saved: List[str] = []
    for fmt in formats:
        fmt = fmt.strip().lower().lstrip(".")
        if not fmt:
            continue
        path = outbase.with_suffix(f".{fmt}")
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        saved.append(str(path))
    plt.close(fig)
    return saved


# =============================================================================
# Stats
# =============================================================================
def summarize_dataset(df: pd.DataFrame, feature_set: str, cohorts: Sequence[str]) -> pd.DataFrame:
    rows = []
    for cohort in cohorts:
        sub = df[df["cohort"] == cohort]
        rows.append({
            "feature_set": feature_set,
            "cohort": cohort,
            "n_rows": int(len(sub)),
            "n_cbag": int(sub["_cbag"].notna().sum()) if "_cbag" in sub else 0,
            "cbag_source": ";".join(sorted(set(map(str, sub.get("_cbag_source", pd.Series(dtype=str)).dropna())))),
            "age_source": ";".join(sorted(set(map(str, sub.get("_age_source", pd.Series(dtype=str)).dropna())))),
            "pred_age_source": ";".join(sorted(set(map(str, sub.get("_pred_age_source", pd.Series(dtype=str)).dropna())))),
        })
    return pd.DataFrame(rows)


def compute_model_metrics(df: pd.DataFrame, cohorts: Sequence[str], feature_sets: Sequence[str]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    for feature_set in feature_sets:
        fs = df[df["feature_set"] == feature_set]
        for cohort in cohorts:
            sub = fs[fs["cohort"] == cohort].copy()
            if sub.empty:
                rows.append({
                    "cohort": cohort,
                    "feature_set": feature_set,
                    "n": 0,
                    "MAE": np.nan,
                    "RMSE": np.nan,
                    "R2": np.nan,
                    "r": np.nan,
                    "cbag_mean": np.nan,
                    "cbag_sd": np.nan,
                    "status": "missing",
                })
                continue

            age = clean_numeric(sub["_age"])
            pred = clean_numeric(sub["_pred_age"])
            cbag = clean_numeric(sub["_cbag"])

            ok_pred = age.notna() & pred.notna()
            if ok_pred.sum() >= 3:
                y = age[ok_pred]
                yhat = pred[ok_pred]
                mae = mean_absolute_error(y, yhat)
                rmse = math.sqrt(mean_squared_error(y, yhat))
                r2 = r2_score(y, yhat) if y.nunique() > 1 else np.nan
                r = pearsonr(y, yhat)[0] if y.nunique() > 1 and yhat.nunique() > 1 else np.nan
                status = "pred_age_metrics"
                n_metric = int(ok_pred.sum())
            else:
                mae = np.nan
                rmse = np.nan
                r2 = np.nan
                r = np.nan
                status = "no_pred_age_metrics"
                n_metric = int(cbag.notna().sum())

            rows.append({
                "cohort": cohort,
                "feature_set": feature_set,
                "n": n_metric,
                "MAE": mae,
                "RMSE": rmse,
                "R2": r2,
                "r": r,
                "cbag_mean": float(cbag.mean()) if cbag.notna().any() else np.nan,
                "cbag_sd": float(cbag.std()) if cbag.notna().sum() > 1 else np.nan,
                "status": status,
            })

    return pd.DataFrame(rows)


def compute_endpoint_correlations(
    df: pd.DataFrame,
    cohorts: Sequence[str],
    min_n: int = 8,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    for endpoint, spec in ENDPOINTS.items():
        candidates = spec["candidates"]  # type: ignore[index]
        domain = spec["domain"]  # type: ignore[index]
        prefer_keywords = spec.get("prefer_keywords", [])  # type: ignore[union-attr]
        avoid_keywords = spec.get("avoid_keywords", [])  # type: ignore[union-attr]
        value_range = spec.get("value_range", None)  # type: ignore[union-attr]

        for cohort in cohorts:
            sub = df[df["cohort"] == cohort].copy()
            if sub.empty:
                rows.append({
                    "endpoint": endpoint,
                    "domain": domain,
                    "cohort": cohort,
                    "column": None,
                    "n": 0,
                    "r": np.nan,
                    "p": np.nan,
                    "status": "missing_cohort",
                })
                continue

            col = find_numeric_col(
                sub,
                candidates=candidates,  # type: ignore[arg-type]
                min_n=min_n,
                avoid_keywords=avoid_keywords,  # type: ignore[arg-type]
                prefer_keywords=prefer_keywords,  # type: ignore[arg-type]
                value_range=value_range,  # type: ignore[arg-type]
            )

            if col is None:
                rows.append({
                    "endpoint": endpoint,
                    "domain": domain,
                    "cohort": cohort,
                    "column": None,
                    "n": 0,
                    "r": np.nan,
                    "p": np.nan,
                    "status": "missing_endpoint",
                })
                continue

            x = clean_numeric(sub["_cbag"])
            y = clean_numeric(sub[col])
            if value_range is not None:
                lo, hi = value_range  # type: ignore[misc]
                y = y.where((y >= lo) & (y <= hi))
            ok = x.notna() & y.notna()

            if ok.sum() < min_n or x[ok].nunique() < 2 or y[ok].nunique() < 2:
                rows.append({
                    "endpoint": endpoint,
                    "domain": domain,
                    "cohort": cohort,
                    "column": col,
                    "n": int(ok.sum()),
                    "r": np.nan,
                    "p": np.nan,
                    "status": "insufficient_n_or_variance",
                })
                continue

            r, p = pearsonr(x[ok], y[ok])
            rows.append({
                "endpoint": endpoint,
                "domain": domain,
                "cohort": cohort,
                "column": col,
                "n": int(ok.sum()),
                "r": float(r),
                "p": float(p),
                "status": "ok",
            })

    return pd.DataFrame(rows)


def fisher_z_summary(r_values: Sequence[float], n_values: Sequence[int]) -> Tuple[float, float, float]:
    """
    Fixed-effect Fisher z meta-summary for correlations.

    Returns:
      meta_r, z_mean, se_z
    """
    vals = []
    weights = []
    for r, n in zip(r_values, n_values):
        if pd.isna(r) or pd.isna(n) or n <= 3:
            continue
        rr = float(np.clip(r, -0.999999, 0.999999))
        vals.append(np.arctanh(rr))
        weights.append(max(float(n) - 3.0, 1.0))
    if not vals:
        return np.nan, np.nan, np.nan
    z_mean = float(np.average(vals, weights=weights))
    se_z = float(1.0 / math.sqrt(np.sum(weights)))
    meta_r = float(np.tanh(z_mean))
    return meta_r, z_mean, se_z


def summarize_replication(stats: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    for endpoint, sub in stats.groupby("endpoint", sort=False):
        valid = sub.dropna(subset=["r"]).copy()
        domain = str(sub["domain"].dropna().iloc[0]) if sub["domain"].notna().any() else ""

        if valid.empty:
            rows.append({
                "endpoint": endpoint,
                "domain": domain,
                "n_cohorts": 0,
                "total_n": 0,
                "median_r": np.nan,
                "mean_abs_r": np.nan,
                "meta_r": np.nan,
                "same_direction_n": 0,
                "direction_consistency": np.nan,
                "fisher_combined_p": np.nan,
                "available_cohorts": "",
                "columns_used": "",
            })
            continue

        median_r = float(valid["r"].median())
        dominant_sign = np.sign(median_r)
        if dominant_sign == 0:
            dominant_sign = np.sign(valid["r"].mean())
        if dominant_sign == 0:
            same_direction_n = int((np.sign(valid["r"]) == 0).sum())
        else:
            same_direction_n = int((np.sign(valid["r"]) == dominant_sign).sum())

        meta_r, _, _ = fisher_z_summary(valid["r"].tolist(), valid["n"].tolist())

        pvals = valid["p"].dropna()
        if len(pvals) > 0:
            try:
                fisher_p = float(combine_pvalues(np.clip(pvals.values, 1e-300, 1.0), method="fisher")[1])
            except Exception:
                fisher_p = np.nan
        else:
            fisher_p = np.nan

        rows.append({
            "endpoint": endpoint,
            "domain": domain,
            "n_cohorts": int(valid["cohort"].nunique()),
            "total_n": int(valid["n"].sum()),
            "median_r": median_r,
            "mean_abs_r": float(valid["r"].abs().mean()),
            "meta_r": meta_r,
            "same_direction_n": same_direction_n,
            "direction_consistency": float(same_direction_n / max(len(valid), 1)),
            "fisher_combined_p": fisher_p,
            "available_cohorts": ", ".join(valid["cohort"].astype(str).tolist()),
            "columns_used": "; ".join(
                f"{row.cohort}:{row.column}" for row in valid.itertuples() if pd.notna(row.column)
            ),
        })

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["n_cohorts", "mean_abs_r", "total_n"], ascending=[False, False, False])
    return out


# =============================================================================
# Plotting
# =============================================================================
def plot_heatmap_panel(
    ax: plt.Axes,
    metrics: pd.DataFrame,
    cohorts: Sequence[str],
    feature_sets: Sequence[str],
    metric: str,
) -> None:
    if metrics.empty:
        ax.text(0.5, 0.5, "No model metrics found", ha="center", va="center")
        ax.axis("off")
        return

    if metric not in metrics.columns:
        metric = "n"

    mat = metrics.pivot(index="cohort", columns="feature_set", values=metric)
    mat = mat.reindex(index=cohorts, columns=feature_sets)

    arr = mat.to_numpy(dtype=float)
    masked = np.ma.masked_invalid(arr)
    im = ax.imshow(masked, aspect="auto")

    ax.set_xticks(range(len(feature_sets)))
    ax.set_xticklabels([nice_label(x) for x in feature_sets], rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(len(cohorts)))
    ax.set_yticklabels(cohorts, fontsize=9)

    for i in range(len(cohorts)):
        for j in range(len(feature_sets)):
            val = arr[i, j]
            if np.isfinite(val):
                if metric in ["MAE", "RMSE"]:
                    txt = f"{val:.2f}"
                elif metric in ["R2", "r"]:
                    txt = f"{val:.2f}"
                else:
                    txt = f"{int(val)}"
            else:
                txt = "NA"
            ax.text(j, i, txt, ha="center", va="center", fontsize=7)

    title_metric = {"MAE": "MAE", "RMSE": "RMSE", "R2": "R²", "r": "r", "n": "N"}.get(metric, metric)
    ax.set_title(f"Model performance across cohorts ({title_metric})", fontsize=11)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=7)


def plot_cbag_distribution_panel(ax: plt.Axes, df: pd.DataFrame, cohorts: Sequence[str]) -> None:
    data = []
    labels = []
    for cohort in cohorts:
        vals = clean_numeric(df.loc[df["cohort"] == cohort, "_cbag"]).dropna().values
        if len(vals) > 0:
            data.append(vals)
            labels.append(cohort)

    if not data:
        ax.text(0.5, 0.5, "No cBAG values found", ha="center", va="center")
        ax.axis("off")
        return

    ax.boxplot(data, labels=labels, showfliers=False)
    for i, vals in enumerate(data, start=1):
        rng = np.random.default_rng(12345 + i)
        if len(vals) > 800:
            vals_plot = rng.choice(vals, size=800, replace=False)
        else:
            vals_plot = vals
        jitter = rng.normal(loc=i, scale=0.045, size=len(vals_plot))
        ax.scatter(jitter, vals_plot, s=5, alpha=0.20, linewidths=0)

    ax.axhline(0, linewidth=1)
    ax.set_ylabel("cBAG")
    ax.set_title("cBAG distribution by cohort", fontsize=11)
    ax.tick_params(axis="x", rotation=0, labelsize=9)


def endpoint_order_for_plot(summary: pd.DataFrame, max_endpoints: int, mode: str = "effect") -> List[str]:
    if summary.empty:
        return []
    tmp = summary.dropna(subset=["median_r"]).copy()
    if tmp.empty:
        return []
    if mode == "effect":
        tmp["_sort"] = tmp["median_r"].abs()
        tmp = tmp.sort_values(["n_cohorts", "_sort", "total_n"], ascending=[False, False, False])
        return tmp["endpoint"].head(max_endpoints).tolist()
    endpoint_names = list(ENDPOINTS.keys())
    endpoints = [e for e in endpoint_names if e in set(tmp["endpoint"])]
    return endpoints[:max_endpoints]


def plot_forest_panel(
    ax: plt.Axes,
    stats: pd.DataFrame,
    summary: pd.DataFrame,
    cohorts: Sequence[str],
    max_endpoints: int,
    endpoint_order: str,
) -> None:
    endpoints = endpoint_order_for_plot(summary, max_endpoints=max_endpoints, mode=endpoint_order)
    if not endpoints:
        ax.text(0.5, 0.5, "No endpoint correlations available", ha="center", va="center")
        ax.axis("off")
        return

    y_positions = np.arange(len(endpoints))
    cohort_offsets = np.linspace(-0.25, 0.25, max(len(cohorts), 1))

    for c_idx, cohort in enumerate(cohorts):
        xs = []
        ys = []
        sizes = []
        for e_idx, endpoint in enumerate(endpoints):
            row = stats[(stats["endpoint"] == endpoint) & (stats["cohort"] == cohort)]
            if row.empty or pd.isna(row.iloc[0]["r"]):
                continue
            r = float(row.iloc[0]["r"])
            n = int(row.iloc[0]["n"])
            xs.append(r)
            ys.append(e_idx + cohort_offsets[c_idx])
            sizes.append(max(20, min(120, n / 3)))
        if xs:
            ax.scatter(xs, ys, s=sizes, alpha=0.75, label=cohort)

    # Add meta/median line markers.
    for e_idx, endpoint in enumerate(endpoints):
        row = summary[summary["endpoint"] == endpoint]
        if row.empty:
            continue
        meta_r = row.iloc[0].get("meta_r", np.nan)
        median_r = row.iloc[0].get("median_r", np.nan)
        val = meta_r if pd.notna(meta_r) else median_r
        if pd.notna(val):
            ax.plot([float(val), float(val)], [e_idx - 0.34, e_idx + 0.34], linewidth=2)

    ax.axvline(0, linewidth=1)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(endpoints, fontsize=8)
    ax.set_xlabel("Pearson r with cBAG")
    ax.set_title("Cross-cohort endpoint associations", fontsize=11)
    ax.set_xlim(-1, 1)
    ax.invert_yaxis()
    ax.legend(fontsize=7, frameon=False, loc="lower right")


def plot_replication_summary_panel(
    ax: plt.Axes,
    summary: pd.DataFrame,
    max_endpoints: int,
    endpoint_order: str,
) -> None:
    endpoints = endpoint_order_for_plot(summary, max_endpoints=max_endpoints, mode=endpoint_order)
    if not endpoints:
        ax.text(0.5, 0.5, "No replication summary available", ha="center", va="center")
        ax.axis("off")
        return

    tmp = summary.set_index("endpoint").loc[endpoints].reset_index()
    y = np.arange(len(tmp))
    x = tmp["meta_r"].where(tmp["meta_r"].notna(), tmp["median_r"]).astype(float)
    sizes = 60 + 50 * tmp["n_cohorts"].fillna(0).astype(float)

    ax.scatter(x, y, s=sizes, alpha=0.8)
    for i, row in tmp.iterrows():
        label = f"{int(row['same_direction_n'])}/{int(row['n_cohorts'])} same dir"
        p = row.get("fisher_combined_p", np.nan)
        if pd.notna(p):
            label += f", p={p:.1e}" if p < 0.001 else f", p={p:.3f}"
        ax.text(0.98, i, label, transform=ax.get_yaxis_transform(), fontsize=7, va="center", ha="right")

    ax.axvline(0, linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(tmp["endpoint"].tolist(), fontsize=8)
    ax.set_xlabel("Meta / median r with cBAG")
    ax.set_title("Replication consistency summary", fontsize=11)
    ax.set_xlim(-1, 1)
    ax.invert_yaxis()


def make_figure7(
    main_df: pd.DataFrame,
    metrics: pd.DataFrame,
    endpoint_stats: pd.DataFrame,
    replication_summary: pd.DataFrame,
    cohorts: Sequence[str],
    feature_sets: Sequence[str],
    outdir: Path,
    formats: Sequence[str],
    dpi: int,
    metric: str,
    main_feature_set: str,
    max_endpoints_forest: int,
    endpoint_order: str,
) -> List[str]:
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    ax_a, ax_b, ax_c, ax_d = axes.ravel()

    plot_heatmap_panel(ax_a, metrics, cohorts, feature_sets, metric=metric)
    panel_label(ax_a, "A")

    plot_cbag_distribution_panel(ax_b, main_df, cohorts)
    ax_b.set_title(f"cBAG distribution by cohort ({nice_label(main_feature_set)})", fontsize=11)
    panel_label(ax_b, "B")

    plot_forest_panel(
        ax_c,
        endpoint_stats,
        replication_summary,
        cohorts,
        max_endpoints=max_endpoints_forest,
        endpoint_order=endpoint_order,
    )
    panel_label(ax_c, "C")

    plot_replication_summary_panel(
        ax_d,
        replication_summary,
        max_endpoints=max_endpoints_forest,
        endpoint_order=endpoint_order,
    )
    panel_label(ax_d, "D")

    fig.suptitle(
        "Figure 7. Cross-cohort replication of cBAG associations across independent cohorts",
        fontsize=15,
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    return save_all(fig, outdir / "Figure7_cross_cohort_replication", formats=formats, dpi=dpi)


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    args = parse_args()

    base_dir = Path(args.base_dir).expanduser().resolve()
    results_root = Path(args.results_root).expanduser().resolve() if args.results_root else base_dir.parent
    outdir = Path(args.outdir).expanduser().resolve() if args.outdir else base_dir / DEFAULT_OUTDIR_NAME
    outdir.mkdir(parents=True, exist_ok=True)

    cohorts = [c.strip() for c in args.cohorts.split(",") if c.strip()]
    feature_sets = [fs.strip() for fs in args.feature_sets.split(",") if fs.strip()]
    formats = [f.strip().lower().lstrip(".") for f in args.formats.split(",") if f.strip()]

    print(f"BASE_DIR:         {base_dir}")
    print(f"RESULTS_ROOT:     {results_root}")
    print(f"OUTDIR:           {outdir}")
    print(f"Main feature set: {args.main_feature_set}")
    print(f"Cohorts:          {', '.join(cohorts)}")
    print(f"Feature sets:     {', '.join(feature_sets)}")
    print(f"Formats:          {', '.join(formats)}")

    # Load all feature sets for Panel A.
    all_frames: List[pd.DataFrame] = []
    dataset_summaries: List[pd.DataFrame] = []
    for feature_set in feature_sets:
        df = load_subject_level_tables(results_root, cohorts, feature_set, min_n=args.min_n)
        if df.empty:
            print(f"[WARN] No usable subject-level data found for feature set: {feature_set}")
            continue
        all_frames.append(df)
        dataset_summaries.append(summarize_dataset(df, feature_set, cohorts))

    if not all_frames:
        raise SystemExit(
            "No usable subject-level validation tables found. "
            "Run brainage_validation_ALL_COHORTS_BAG_OOFGLOBAL_FINAL_directpaths.py first "
            "or check --results-root."
        )

    all_df = pd.concat(all_frames, ignore_index=True, sort=False)
    metrics = compute_model_metrics(all_df, cohorts=cohorts, feature_sets=feature_sets)

    # Main feature set for endpoint synthesis.
    main_df = all_df[all_df["feature_set"] == args.main_feature_set].copy()
    if main_df.empty:
        print(f"[WARN] Main feature set {args.main_feature_set} was not loaded from all-feature table. Trying direct load.")
        main_df = load_subject_level_tables(results_root, cohorts, args.main_feature_set, min_n=args.min_n)

    if main_df.empty:
        raise SystemExit(
            f"No usable data found for main feature set: {args.main_feature_set}. "
            "Check --main-feature-set or run the validation script first."
        )

    endpoint_stats = compute_endpoint_correlations(main_df, cohorts=cohorts, min_n=args.min_n)
    replication_summary = summarize_replication(endpoint_stats)

    outputs = make_figure7(
        main_df=main_df,
        metrics=metrics,
        endpoint_stats=endpoint_stats,
        replication_summary=replication_summary,
        cohorts=cohorts,
        feature_sets=feature_sets,
        outdir=outdir,
        formats=formats,
        dpi=args.dpi,
        metric=args.metric,
        main_feature_set=args.main_feature_set,
        max_endpoints_forest=args.max_endpoints_forest,
        endpoint_order=args.endpoint_order,
    )

    dataset_summary = pd.concat(dataset_summaries, ignore_index=True, sort=False) if dataset_summaries else pd.DataFrame()

    dataset_summary_path = outdir / "figure7_dataset_summary.csv"
    metrics_path = outdir / "figure7_model_metric_summary.csv"
    endpoint_stats_path = outdir / "figure7_endpoint_correlation_stats.csv"
    replication_summary_path = outdir / "figure7_replication_summary.csv"
    manifest_path = outdir / "figure7_manifest.csv"

    dataset_summary.to_csv(dataset_summary_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    endpoint_stats.to_csv(endpoint_stats_path, index=False)
    replication_summary.to_csv(replication_summary_path, index=False)

    manifest = pd.DataFrame([
        {
            "figure": "Figure7",
            "description": "Cross-cohort cBAG synthesis: performance, cBAG distribution, endpoint forest plot, replication summary",
            "main_feature_set": args.main_feature_set,
            "feature_sets": ",".join(feature_sets),
            "cohorts": ",".join(cohorts),
            "outputs": ";".join(outputs),
            "dataset_summary": str(dataset_summary_path),
            "model_metric_summary": str(metrics_path),
            "endpoint_correlation_stats": str(endpoint_stats_path),
            "replication_summary": str(replication_summary_path),
        }
    ])
    manifest.to_csv(manifest_path, index=False)

    print("\nSaved Figure 7 outputs:")
    for output in outputs:
        print(f"  {output}")

    print("\nSaved Figure 7 tables:")
    print(f"  Dataset summary:       {dataset_summary_path}")
    print(f"  Model metric summary:  {metrics_path}")
    print(f"  Endpoint stats:        {endpoint_stats_path}")
    print(f"  Replication summary:   {replication_summary_path}")
    print(f"  Manifest:              {manifest_path}")

    if not replication_summary.empty:
        print("\nTop replication endpoints:")
        cols = ["endpoint", "domain", "n_cohorts", "median_r", "meta_r", "same_direction_n", "fisher_combined_p"]
        print(replication_summary[cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
