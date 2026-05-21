#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate Main Figure 4 and Supplementary Figure S4A-E for cBAG neuroimaging associations.

Main Figure 4:
    Imaging-only model, cohort-specific panels.

Supplementary Figure S4A-E:
    Same cohort-specific layout for all ablation models.

Cohort panel mapping inside each figure:
    A = ADNI
    B = ADRC
    C = HABS
    D = AD_DECODE

Feature-set mapping for supplementary figures:
    S4A = imaging_only
    S4B = imaging_demographics
    S4C = imaging_biomarkers
    S4D = full
    S4E = full_no_cardiovascular

Within each cohort panel, the script creates a 2 x 4 mini-grid:
    First row, all hippocampal-specific predictors:
        1) cBAG vs hippocampal volume relative to brain
        2) cBAG vs hippocampal FA
        3) cBAG vs hippocampal clustering coefficient
        4) cBAG vs hippocampal path length
    Second row, all total/global brain predictors:
        5) cBAG vs total brain volume
        6) cBAG vs total brain FA
        7) cBAG vs total/global graph clustering coefficient
        8) cBAG vs total/global graph path length

Expected input files:
    <RESULTS_ROOT>/<BrainAgePrediction...>/ablation_<feature_set>/validation_figures/
        subject_level_validation_input.csv

Default BASE_DIR:
    /mnt/newStor/paros/paros_WORK/ines/results/
    BrainAgeValidation_AllCohorts_BAGBiasCorr_OOFGlobal_BiologicalValidation

Default output directory:
    <BASE_DIR>/Figure4_main_and_S4_supplement_neuroimaging_associations/

Run:
    python make_figure4_main_and_supp_neuroimaging_associations.py

Optional:
    python make_figure4_main_and_supp_neuroimaging_associations.py \
        --base-dir /path/to/BrainAgeValidation_AllCohorts_BAGBiasCorr_OOFGlobal_BiologicalValidation \
        --results-root /path/to/ines/results \
        --outdir /path/to/output
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import linregress, pearsonr


# =============================================================================
# Defaults
# =============================================================================
BASE_DIR = (
    "/mnt/newStor/paros/paros_WORK/ines/results/"
    "BrainAgeValidation_AllCohorts_BAGBiasCorr_OOFGlobal_BiologicalValidation"
)

DEFAULT_OUTDIR_NAME = "Figure4_main_and_S4_supplement_neuroimaging_associations_fixed"
DEFAULT_MAIN_FEATURE_SET = "imaging_only"
DEFAULT_SUPPLEMENT_FEATURE_SETS = [
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

COHORT_ORDER = ["ADNI", "ADRC", "HABS", "AD_DECODE"]
COHORT_LETTERS = {
    "ADNI": "A",
    "ADRC": "B",
    "HABS": "C",
    "AD_DECODE": "D",
}

MODEL_LETTERS = {
    "imaging_only": "A",
    "imaging_demographics": "B",
    "imaging_biomarkers": "C",
    "full": "D",
    "full_no_cardiovascular": "E",
}

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

# Row 1, Slot 1: hippocampal volume relative to brain.
# Prefer explicitly relative/percent/normalized hippocampal volume, not raw left/right volume.
HIPPO_VOLUME_REL_PRIORITY = [
    "Hc_volume_relative_to_brain",
    "HC_volume_relative_to_brain",
    "Hippocampus_volume_relative_to_brain",
    "hippocampus_volume_relative_to_brain",
    "Hippocampus_Total_relative_to_brain",
    "Hippocampus_Total_pct",
    "Hippocampus_Total_percent",
    "Hippocampus_Total_norm",
    "Hippocampus_Total_normalized",
    "Hippocampus_Total_ICV_norm",
    "Hippocampus_Total_ICV_normalized",
    "Hippocampus_pct",
    "hippocampus_pct",
    "HC_volume_pct",
    "hc_volume_pct",
    "HC_vol_pct",
    "hc_vol_pct",
    "Hippocampus_Total",
    "Hippocampus_volume",
    "hippocampus_volume",
    "Hippocampal_volume",
    "hippocampal_volume",
    "hippocampal_vol",
    "HC_volume",
    "hc_volume",
    "HC_vol",
    "hc_vol",
]

# Row 1, Slot 2: hippocampal FA. Values are required to be in [0, 1].
HIPPO_FA_PRIORITY = [
    "Hc_FA",
    "HC_FA",
    "Hc_Fa",
    "HC_Fa",
    "Hippocampus_FA_Mean",
    "Hippocampus_FA_Total",
    "Hippocampus_FA",
    "hippocampus_FA",
    "hippocampal_FA",
    "Hippocampal_FA",
    "Left_Hippocampus_FA",
    "Right_Hippocampus_FA",
    "hc_fa",
]

# Row 1, Slot 3: hippocampal graph clustering coefficient.
HIPPO_CLUSTERING_PRIORITY = [
    "Hc_clustering_coeff",
    "HC_clustering_coeff",
    "Hc_Clustering_Coeff",
    "HC_Clustering_Coeff",
    "Hippocampus_clustering_coeff",
    "Hippocampus_Clustering_Coeff",
    "hippocampal_clustering_coeff",
    "Hippocampal_Clustering_Coeff",
    "Hc_clustering_coefficient",
    "HC_clustering_coefficient",
    "Hippocampus_clustering_coefficient",
    "Hippocampal_clustering_coefficient",
]

# Row 1, Slot 4: hippocampal graph path length.
HIPPO_PATH_LENGTH_PRIORITY = [
    "Hc_path_length",
    "HC_path_length",
    "Hc_Path_Length",
    "HC_Path_Length",
    "Hippocampus_path_length",
    "Hippocampus_Path_Length",
    "hippocampal_path_length",
    "Hippocampal_Path_Length",
    "Hc_characteristic_path_length",
    "HC_characteristic_path_length",
    "Hippocampus_characteristic_path_length",
    "Hippocampal_characteristic_path_length",
]

# Row 2, Slot 1: total brain volume.
TOTAL_BRAIN_VOLUME_PRIORITY = [
    "Total_Brain_volume",
    "total_brain_volume",
    "Total_Brain_Volume",
    "Brain_volume_total",
    "brain_volume_total",
    "TBV",
    "TotalBrainVolume",
    "Total_Brain_volume_pct",
    "Total_Brain_volume_norm",
    "Total_Brain_volume_normalized",
    "Relative_Brain_Volume",
    "relative_brain_volume",
    "Normalized_Brain_Volume",
    "normalized_brain_volume",
    "ICV_normalized_volume",
    "Volume_mean",
    "Volume_median",
]

# Row 2, Slot 2: total/global brain FA. Values are required to be in [0, 1].
TOTAL_BRAIN_FA_PRIORITY = [
    "Total_Brain_FA",
    "total_brain_FA",
    "Brain_FA_total",
    "brain_FA_total",
    "Global_FA",
    "global_FA",
    "FA_mean",
    "Mean_FA",
    "mean_FA",
    "FA_median",
    "FA",
]

# Row 2, Slot 3: total/global graph clustering coefficient.
TOTAL_GRAPH_CLUSTERING_PRIORITY = [
    "Total_graph_clustering_coeff",
    "total_graph_clustering_coeff",
    "Global_graph_clustering_coeff",
    "global_graph_clustering_coeff",
    "Graph_clustering_coeff",
    "graph_clustering_coeff",
    "Clustering_Coeff",
    "clustering_coeff",
    "ClusteringCoefficient",
    "Global clustering coefficient",
    "global_clustering_coefficient",
    "clustering coefficient",
]

# Row 2, Slot 4: total/global graph path length.
TOTAL_GRAPH_PATH_LENGTH_PRIORITY = [
    "Total_graph_path_length",
    "total_graph_path_length",
    "Global_graph_path_length",
    "global_graph_path_length",
    "Graph_path_length",
    "graph_path_length",
    "Path_Length",
    "path_length",
    "Characteristic path length",
    "characteristic_path_length",
    "PathLength",
    "path length",
]

# FA slots should only use physically plausible FA values.
SLOT_X_RANGES = {
    "hippocampal_fa": (0.0, 1.0),
    "total_brain_fa": (0.0, 1.0),
}


# =============================================================================
# CLI
# =============================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate Main Figure 4 and Supplementary Figure S4A-E for cohort-specific "
            "cBAG neuroimaging associations."
        )
    )
    parser.add_argument(
        "--base-dir",
        default=BASE_DIR,
        help="Combined biological-validation output directory.",
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
        help="Feature set for main Figure 4. Default: imaging_only",
    )
    parser.add_argument(
        "--supplement-feature-sets",
        default=",".join(DEFAULT_SUPPLEMENT_FEATURE_SETS),
        help=(
            "Comma-separated feature sets for supplementary S4 figures. "
            "Default: imaging_only,imaging_demographics,imaging_biomarkers,full,full_no_cardiovascular"
        ),
    )
    parser.add_argument(
        "--cohorts",
        default=",".join(COHORT_ORDER),
        help="Comma-separated cohorts. Default: ADNI,ADRC,HABS,AD_DECODE",
    )
    parser.add_argument(
        "--formats",
        default="png,pdf",
        help="Comma-separated output formats. Default: png,pdf",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=450,
        help="Raster output DPI. Default: 450",
    )
    parser.add_argument(
        "--min-n",
        type=int,
        default=8,
        help="Minimum complete cases required for a scatter/correlation panel. Default: 8",
    )
    return parser.parse_args()


# =============================================================================
# Utility functions
# =============================================================================
def clean_numeric(series: pd.Series) -> pd.Series:
    out = pd.to_numeric(series, errors="coerce").copy()
    for val in SENTINEL_VALUES:
        out = out.mask(out == val, np.nan)
    return out


def apply_value_range(series: pd.Series, value_range: Optional[tuple[float, float]] = None) -> pd.Series:
    out = clean_numeric(series)
    if value_range is None:
        return out
    lo, hi = value_range
    return out.mask((out < lo) | (out > hi), np.nan)


def usable_n(series: pd.Series, value_range: Optional[tuple[float, float]] = None) -> int:
    return int(apply_value_range(series, value_range=value_range).notna().sum())


def first_existing(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _candidate_tokens(candidates: Sequence[str]) -> List[str]:
    tokens: List[str] = []
    for c in candidates:
        tokens.extend([t for t in re.split(r"[_\s]+", str(c).lower()) if len(t) >= 3])
    return sorted(set(tokens), key=len, reverse=True)


def find_numeric_col(
    df: pd.DataFrame,
    candidates: Sequence[str],
    min_n: int = 8,
    avoid_keywords: Optional[Sequence[str]] = None,
    require_keywords: Optional[Sequence[str]] = None,
    prefer_keywords: Optional[Sequence[str]] = None,
    value_range: Optional[tuple[float, float]] = None,
) -> Optional[str]:
    """
    First checks exact candidate names and *_raw_clean versions.
    Then performs a conservative keyword fallback.

    value_range is used for FA slots so columns with values outside [0, 1]
    are not silently accepted.
    """
    avoid_keywords = [x.lower() for x in (avoid_keywords or [])]
    require_keywords = [x.lower() for x in (require_keywords or [])]
    prefer_keywords = [x.lower() for x in (prefer_keywords or [])]

    def eligible(col: str) -> bool:
        low = str(col).lower()
        if any(bad in low for bad in avoid_keywords):
            return False
        if require_keywords and not any(req in low for req in require_keywords):
            return False
        return usable_n(df[col], value_range=value_range) >= min_n

    expanded: List[str] = []
    for c in candidates:
        expanded.extend([f"{c}_raw_clean", c])

    # Exact candidate match always wins.
    for col in expanded:
        if col in df.columns and eligible(col):
            return col

    tokens = _candidate_tokens(candidates)

    # Prefer columns that contain explicit anatomical/scope words requested for the slot.
    fallback_cols = []
    for col in df.columns:
        low = str(col).lower()
        if not any(tok in low for tok in tokens):
            continue
        if not eligible(col):
            continue
        score = sum(kw in low for kw in prefer_keywords)
        fallback_cols.append((score, col))

    if not fallback_cols:
        return None

    fallback_cols.sort(key=lambda x: (-x[0], str(x[1]).lower()))
    return fallback_cols[0][1]

def validation_dir(results_root: Path, cohort: str, feature_set: str) -> Path:
    return results_root / RESULTS_DIR_MAP[cohort] / f"ablation_{feature_set}" / "validation_figures"


def load_subject_level_tables(
    results_root: Path,
    cohorts: Sequence[str],
    feature_set: str,
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

        cbag_col = first_existing(df, CBAG_PRIORITY)
        if cbag_col is None:
            print(f"[WARN] No usable cBAG/BAG column found in {path}")
            continue

        df["_cbag"] = clean_numeric(df[cbag_col])
        df["_cbag_source"] = cbag_col
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True, sort=False)
    out = out.dropna(subset=["_cbag"]).copy()
    return out


def nice_label(name: Optional[str]) -> str:
    if name is None:
        return ""

    raw = str(name).replace("_raw_clean", "")
    mapping = {
        "_cbag": "cBAG",
        "Hc_volume_relative_to_brain": "Hc volume (relative to brain)",
        "HC_volume_relative_to_brain": "Hc volume (relative to brain)",
        "Hippocampus_volume_relative_to_brain": "Hc volume (relative to brain)",
        "Hippocampus_Total_relative_to_brain": "Hc volume (relative to brain)",
        "Hippocampus_Total_pct": "Hc volume (relative to brain)",
        "Hippocampus_Total_percent": "Hc volume (relative to brain)",
        "Hippocampus_Total_norm": "Hc volume (relative to brain)",
        "Hippocampus_Total_normalized": "Hc volume (relative to brain)",
        "Hc_FA": "Hc FA",
        "HC_FA": "Hc FA",
        "Hc_Fa": "Hc FA",
        "HC_Fa": "Hc FA",
        "Hippocampus_FA": "Hc FA",
        "Hippocampus_FA_Mean": "Hc FA",
        "Hippocampus_FA_Total": "Hc FA",
        "Hc_clustering_coeff": "Hc clustering coeff",
        "HC_clustering_coeff": "Hc clustering coeff",
        "Hippocampus_clustering_coeff": "Hc clustering coeff",
        "Hippocampal_Clustering_Coeff": "Hc clustering coeff",
        "Hc_path_length": "Hc path length",
        "HC_path_length": "Hc path length",
        "Hippocampus_path_length": "Hc path length",
        "Hippocampal_Path_Length": "Hc path length",
        "Total_Brain_FA": "Total brain FA",
        "total_brain_FA": "Total brain FA",
        "Total_graph_clustering_coeff": "Total graph clustering coeff",
        "Global_graph_clustering_coeff": "Total graph clustering coeff",
        "Total_graph_path_length": "Total graph path length",
        "Global_graph_path_length": "Total graph path length",

        "cBAG_oof_global": "cBAG",
        "Hippocampus_Total_pct": "Hippocampal volume",
        "Hippocampus_Total": "Hippocampal volume",
        "Hippocampus_volume": "Hippocampal volume",
        "hippocampus_volume": "Hippocampal volume",
        "Hippocampal_volume": "Hippocampal volume",
        "Left_Hippocampus_pct": "Left hippocampal volume",
        "Right_Hippocampus_pct": "Right hippocampal volume",
        "Hippocampus_FA_Mean": "Hippocampal FA",
        "Hippocampus_FA_Total": "Hippocampal FA",
        "Left_Hippocampus_FA": "Left hippocampal FA",
        "Right_Hippocampus_FA": "Right hippocampal FA",
        "Relative_Brain_Volume": "Relative brain volume",
        "relative_brain_volume": "Relative brain volume",
        "Normalized_Brain_Volume": "Normalized brain volume",
        "normalized_brain_volume": "Normalized brain volume",
        "Total_Brain_volume_pct": "Relative brain volume",
        "Total_Brain_volume_norm": "Normalized total brain volume",
        "Total_Brain_volume_normalized": "Normalized total brain volume",
        "Total_Brain_volume": "Total brain volume",
        "total_brain_volume": "Total brain volume",
        "TBV": "Total brain volume",
        "Volume_mean": "Mean volume",
        "Volume_median": "Median volume",
        "FA_mean": "Mean FA",
        "FA_median": "Median FA",
        "Mean_FA": "Mean FA",
        "Global_FA": "Global FA",
        "Clustering_Coeff": "Clustering coefficient",
        "clustering_coeff": "Clustering coefficient",
        "Global clustering coefficient": "Clustering coefficient",
        "Path_Length": "Path length",
        "path_length": "Path length",
        "Characteristic path length": "Path length",
    }
    return mapping.get(raw, raw.replace("_", " "))


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


def style_axis(ax: plt.Axes) -> None:
    ax.tick_params(axis="both", labelsize=7)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)


def empty_axis(ax: plt.Axes, title: str, message: str) -> None:
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=8, wrap=True)
    ax.set_title(title, fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def cohort_header(ax: plt.Axes, cohort: str) -> None:
    letter = COHORT_LETTERS.get(cohort, "?")
    ax.text(
        -0.20,
        1.23,
        f"{letter}. {cohort}",
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        ha="left",
        va="top",
    )


def scatter_with_fit(
    ax: plt.Axes,
    df: pd.DataFrame,
    x_col: Optional[str],
    title: str,
    min_n: int,
    x_range: Optional[tuple[float, float]] = None,
) -> Dict[str, object]:
    if x_col is None:
        empty_axis(ax, title, "Variable not found")
        return {
            "status": "missing",
            "x_col": "",
            "n": 0,
            "r": np.nan,
            "p": np.nan,
        }

    tmp = pd.DataFrame({
        "x": apply_value_range(df[x_col], value_range=x_range),
        "y": clean_numeric(df["_cbag"]),
    }).dropna()

    if len(tmp) < min_n:
        empty_axis(ax, title, f"Insufficient data\nn={len(tmp)}")
        return {
            "status": "insufficient_n",
            "x_col": x_col,
            "n": int(len(tmp)),
            "r": np.nan,
            "p": np.nan,
        }

    if tmp["x"].nunique() < 2 or tmp["y"].nunique() < 2:
        empty_axis(ax, title, "Constant variable")
        return {
            "status": "constant",
            "x_col": x_col,
            "n": int(len(tmp)),
            "r": np.nan,
            "p": np.nan,
        }

    ax.scatter(tmp["x"], tmp["y"], s=14, alpha=0.72, edgecolors="none")

    lr = linregress(tmp["x"], tmp["y"])
    xx = np.linspace(tmp["x"].min(), tmp["x"].max(), 100)
    ax.plot(xx, lr.intercept + lr.slope * xx, linestyle="--", linewidth=1.2)

    r, p = pearsonr(tmp["x"], tmp["y"])

    ax.text(
        0.03,
        0.97,
        f"n={len(tmp)}\nr={r:.2f}\np={p:.2g}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=7,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.82, linewidth=0.4),
    )
    ax.set_title(title, fontsize=8)
    ax.set_xlabel(nice_label(x_col), fontsize=7)
    ax.set_ylabel("cBAG", fontsize=7)
    ax.grid(True, alpha=0.22)
    style_axis(ax)

    return {
        "status": "plotted",
        "x_col": x_col,
        "n": int(len(tmp)),
        "r": float(r),
        "p": float(p),
        "slope": float(lr.slope),
        "intercept": float(lr.intercept),
    }


def variable_for_slot(
    df: pd.DataFrame,
    slot_name: str,
    min_n: int,
) -> Optional[str]:
    if slot_name == "hippocampal_volume_relative_to_brain":
        return find_numeric_col(
            df,
            HIPPO_VOLUME_REL_PRIORITY,
            min_n=min_n,
            avoid_keywords=["fa", "clustering", "cluster", "path", "graph"],
            require_keywords=["hippocampus", "hippocampal", "hc_"],
            prefer_keywords=["relative", "pct", "percent", "norm", "normalized", "icv"],
        )
    if slot_name == "hippocampal_fa":
        return find_numeric_col(
            df,
            HIPPO_FA_PRIORITY,
            min_n=min_n,
            avoid_keywords=["total_brain", "global", "whole", "graph", "clustering", "path"],
            require_keywords=["hippocampus", "hippocampal", "hc_"],
            prefer_keywords=["fa"],
            value_range=SLOT_X_RANGES[slot_name],
        )
    if slot_name == "hippocampal_clustering":
        return find_numeric_col(
            df,
            HIPPO_CLUSTERING_PRIORITY,
            min_n=min_n,
            avoid_keywords=["total", "global", "brain", "whole"],
            require_keywords=["hippocampus", "hippocampal", "hc_"],
            prefer_keywords=["clustering", "cluster", "coeff"],
        )
    if slot_name == "hippocampal_path_length":
        return find_numeric_col(
            df,
            HIPPO_PATH_LENGTH_PRIORITY,
            min_n=min_n,
            avoid_keywords=["total", "global", "brain", "whole"],
            require_keywords=["hippocampus", "hippocampal", "hc_"],
            prefer_keywords=["path", "length"],
        )
    if slot_name == "total_brain_volume":
        return find_numeric_col(
            df,
            TOTAL_BRAIN_VOLUME_PRIORITY,
            min_n=min_n,
            avoid_keywords=["hippocampus", "hippocampal", "hc_", "fa", "clustering", "cluster", "path", "graph"],
            prefer_keywords=["total", "brain", "volume", "tbv"],
        )
    if slot_name == "total_brain_fa":
        return find_numeric_col(
            df,
            TOTAL_BRAIN_FA_PRIORITY,
            min_n=min_n,
            avoid_keywords=["hippocampus", "hippocampal", "hc_", "clustering", "cluster", "path", "graph"],
            prefer_keywords=["total", "brain", "global", "fa"],
            value_range=SLOT_X_RANGES[slot_name],
        )
    if slot_name == "total_graph_clustering":
        return find_numeric_col(
            df,
            TOTAL_GRAPH_CLUSTERING_PRIORITY,
            min_n=min_n,
            avoid_keywords=["hippocampus", "hippocampal", "hc_"],
            prefer_keywords=["total", "global", "graph", "clustering", "cluster", "coeff"],
        )
    if slot_name == "total_graph_path_length":
        return find_numeric_col(
            df,
            TOTAL_GRAPH_PATH_LENGTH_PRIORITY,
            min_n=min_n,
            avoid_keywords=["hippocampus", "hippocampal", "hc_"],
            prefer_keywords=["total", "global", "graph", "path", "length"],
        )
    raise ValueError(f"Unknown slot name: {slot_name}")


# =============================================================================
# Figure generation
# =============================================================================
FIGURE4_SLOTS = [
    # Row 1: hippocampal-specific predictors.
    ("hippocampal_volume_relative_to_brain", "cBAG vs Hc volume\n(relative to brain)"),
    ("hippocampal_fa", "cBAG vs Hc FA"),
    ("hippocampal_clustering", "cBAG vs Hc clustering coeff"),
    ("hippocampal_path_length", "cBAG vs Hc path length"),
    # Row 2: total/global brain predictors.
    ("total_brain_volume", "cBAG vs total brain volume"),
    ("total_brain_fa", "cBAG vs total brain FA"),
    ("total_graph_clustering", "cBAG vs total graph clustering coeff"),
    ("total_graph_path_length", "cBAG vs total graph path length"),
]


def plot_cohort_block(
    fig: plt.Figure,
    outer_spec,
    df: pd.DataFrame,
    cohort: str,
    min_n: int,
) -> List[Dict[str, object]]:
    inner = outer_spec.subgridspec(2, 4, wspace=0.45, hspace=0.50)
    axes = [fig.add_subplot(inner[i, j]) for i in range(2) for j in range(4)]

    results: List[Dict[str, object]] = []

    for i, ((slot_name, title), ax) in enumerate(zip(FIGURE4_SLOTS, axes)):
        if i == 0:
            cohort_header(ax, cohort)

        if df.empty:
            empty_axis(ax, title, "No data for cohort")
            results.append({
                "cohort": cohort,
                "slot": slot_name,
                "title": title,
                "status": "empty_cohort",
                "x_col": "",
                "n": 0,
                "r": np.nan,
                "p": np.nan,
            })
            continue

        x_col = variable_for_slot(df, slot_name, min_n=min_n)
        stats = scatter_with_fit(ax, df, x_col, title, min_n=min_n, x_range=SLOT_X_RANGES.get(slot_name))
        results.append({
            "cohort": cohort,
            "slot": slot_name,
            "title": title,
            **stats,
        })

    return results


def make_cohort_panel_figure(
    df: pd.DataFrame,
    cohorts: Sequence[str],
    outdir: Path,
    figure_prefix: str,
    feature_set: str,
    figure_title: str,
    formats: Sequence[str],
    dpi: int,
    min_n: int,
) -> Dict[str, object]:
    fig = plt.figure(figsize=(22, 12))
    outer = fig.add_gridspec(2, 2, wspace=0.18, hspace=0.25)

    stats_rows: List[Dict[str, object]] = []

    for idx, cohort in enumerate(cohorts):
        r, c = divmod(idx, 2)
        df_cohort = df[df["cohort"] == cohort].copy()
        cohort_stats = plot_cohort_block(fig, outer[r, c], df_cohort, cohort, min_n=min_n)
        for row in cohort_stats:
            row["feature_set"] = feature_set
            row["figure"] = figure_prefix
        stats_rows.extend(cohort_stats)

    fig.suptitle(figure_title, fontsize=16, y=0.995)
    fig.text(
        0.5,
        0.005,
        (
            "Each cohort panel uses row 1 for hippocampal-specific metrics "
            "(Hc volume relative to brain, Hc FA, Hc clustering coefficient, Hc path length) "
            "and row 2 for total/global metrics "
            "(total brain volume, total brain FA, total graph clustering coefficient, total graph path length)."
        ),
        ha="center",
        fontsize=9,
    )

    outbase = outdir / f"{figure_prefix}_{feature_set}_cohort_panels_neuroimaging_associations"
    outputs = save_all(fig, outbase, formats=formats, dpi=dpi)

    return {
        "outputs": outputs,
        "stats": stats_rows,
    }


def summarize_dataset(df: pd.DataFrame, feature_set: str, cohorts: Sequence[str]) -> Dict[str, object]:
    row: Dict[str, object] = {
        "feature_set": feature_set,
        "n_total": int(len(df)),
        "cohorts_loaded": ",".join(sorted(df["cohort"].dropna().astype(str).unique())) if not df.empty else "",
        "cbag_sources": ",".join(sorted(df["_cbag_source"].dropna().astype(str).unique())) if "_cbag_source" in df.columns else "",
    }

    for cohort in cohorts:
        dfc = df[df["cohort"] == cohort].copy()
        row[f"n_{cohort}"] = int(len(dfc))
        for slot_name, _ in FIGURE4_SLOTS:
            row[f"{slot_name}_col_{cohort}"] = variable_for_slot(dfc, slot_name, min_n=3) or ""

    return row


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
    formats = [f.strip().lower().lstrip(".") for f in args.formats.split(",") if f.strip()]
    supplement_feature_sets = [fs.strip() for fs in args.supplement_feature_sets.split(",") if fs.strip()]

    print(f"BASE_DIR:     {base_dir}")
    print(f"RESULTS_ROOT: {results_root}")
    print(f"OUTDIR:       {outdir}")
    print(f"Cohorts:      {', '.join(cohorts)}")
    print(f"Formats:      {', '.join(formats)}")

    figure_manifest_rows: List[Dict[str, object]] = []
    association_stats_rows: List[Dict[str, object]] = []
    dataset_summary_rows: List[Dict[str, object]] = []

    # Main Figure 4: imaging_only by default.
    main_df = load_subject_level_tables(results_root, cohorts, args.main_feature_set)
    if main_df.empty:
        raise SystemExit(
            f"No usable subject-level validation data found for main feature set: {args.main_feature_set}. "
            "Run the biological-validation script first or check --results-root."
        )

    dataset_summary_rows.append(summarize_dataset(main_df, args.main_feature_set, cohorts))

    main_result = make_cohort_panel_figure(
        df=main_df,
        cohorts=cohorts,
        outdir=outdir,
        figure_prefix="Figure4",
        feature_set=args.main_feature_set,
        figure_title="Figure 4. Imaging-only model: cohort-specific cBAG neuroimaging associations",
        formats=formats,
        dpi=args.dpi,
        min_n=args.min_n,
    )

    figure_manifest_rows.append({
        "figure": "Figure4",
        "feature_set": args.main_feature_set,
        "description": "Main manuscript Figure 4: imaging-only cohort-specific neuroimaging associations",
        "outputs": ";".join(main_result["outputs"]),
    })
    association_stats_rows.extend(main_result["stats"])

    # Supplementary Figure S4A-E: all requested feature sets.
    for feature_set in supplement_feature_sets:
        letter = MODEL_LETTERS.get(feature_set, "")
        figure_prefix = f"FigureS4{letter}" if letter else f"FigureS4_{feature_set}"

        df = load_subject_level_tables(results_root, cohorts, feature_set)
        if df.empty:
            print(f"[WARN] No usable data found for supplementary feature set: {feature_set}")
            figure_manifest_rows.append({
                "figure": figure_prefix,
                "feature_set": feature_set,
                "description": "Supplementary Figure S4 feature-set panel",
                "outputs": "",
                "status": "missing_or_empty",
            })
            continue

        dataset_summary_rows.append(summarize_dataset(df, feature_set, cohorts))

        result = make_cohort_panel_figure(
            df=df,
            cohorts=cohorts,
            outdir=outdir,
            figure_prefix=figure_prefix,
            feature_set=feature_set,
            figure_title=(
                f"Figure S4{letter}. {feature_set.replace('_', ' ')} model: "
                "cohort-specific cBAG neuroimaging associations"
            ),
            formats=formats,
            dpi=args.dpi,
            min_n=args.min_n,
        )

        figure_manifest_rows.append({
            "figure": figure_prefix,
            "feature_set": feature_set,
            "description": "Supplementary Figure S4 feature-set panel",
            "outputs": ";".join(result["outputs"]),
            "status": "saved",
        })
        association_stats_rows.extend(result["stats"])

    figure_manifest = pd.DataFrame(figure_manifest_rows)
    association_stats = pd.DataFrame(association_stats_rows)
    dataset_summary = pd.DataFrame(dataset_summary_rows)

    manifest_path = outdir / "figure4_s4_generation_manifest.csv"
    stats_path = outdir / "figure4_s4_neuroimaging_association_stats.csv"
    summary_path = outdir / "figure4_s4_dataset_summary.csv"

    figure_manifest.to_csv(manifest_path, index=False)
    association_stats.to_csv(stats_path, index=False)
    dataset_summary.to_csv(summary_path, index=False)

    print("\nSaved figures:")
    for row in figure_manifest_rows:
        print(f"  {row.get('figure', '')} ({row.get('feature_set', '')}): {row.get('outputs', '')}")

    print("\nSaved tables:")
    print(f"  Manifest: {manifest_path}")
    print(f"  Association stats: {stats_path}")
    print(f"  Dataset summary: {summary_path}")


if __name__ == "__main__":
    main()
