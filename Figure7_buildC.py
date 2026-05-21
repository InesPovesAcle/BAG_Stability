#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure 7 + Supplementary Figure 7: cross-cohort cBAG replication synthesis
============================================================================

Purpose
-------
Build a high-impact cross-cohort synthesis figure for cBAG biological replication
across ADNI, ADRC, HABS, and AD_DECODE.

This version fixes the previous Panel A issue by reading model-performance metrics
from the CV-summary files saved by the OOF-global training/validation pipeline,
instead of trying to recompute MAE from subject_level_validation_input.csv.

Main Figure 7
-------------
A. Model performance heatmap across cohorts and feature sets, using CV summaries
B. cBAG distribution by cohort for the main feature set
C. Cohort-level cBAG endpoint forest plot with:
      - color = cohort
      - marker size = endpoint-specific n
      - horizontal bars = 95% Fisher-z CI for r
      - vertical tick = fixed-effect Fisher-z meta r
D. Replication summary with meta r, direction consistency, and combined p-values

Supplementary Figure 7
----------------------
A. MAE heatmap across cohorts and feature sets
B. R² heatmap across cohorts and feature sets
C. All available endpoint-level cohort associations
D. Endpoint availability matrix

Expected inputs
---------------
Per cohort + feature set biological validation table:

    <RESULTS_ROOT>/<BrainAgePrediction...>/ablation_<feature_set>/validation_figures/
        subject_level_validation_input.csv

Performance metrics, preferred:

    <BASE_DIR>/combined_cv_summaries.csv

Fallback performance metrics:

    <RESULTS_ROOT>/<BrainAgePrediction...>/ablation_<feature_set>/
        <prefix>_<feature_set>_cv_summary_metrics.csv

Default BASE_DIR:
    /mnt/newStor/paros/paros_WORK/ines/results/
    BrainAgeValidation_AllCohorts_BAGBiasCorr_OOFGlobal_BiologicalValidation

Run
---
    python Figure7_cross_cohort_final.py

Useful options
--------------
    python Figure7_cross_cohort_final.py \
        --main-feature-set imaging_only \
        --metric MAE \
        --cv-evaluation OOF_BIAS_CORRECTED \
        --formats png,pdf

    python Figure7_cross_cohort_final.py \
        --max-endpoints-main 8 \
        --endpoint-order replication
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import combine_pvalues, norm, pearsonr


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
DEFAULT_COHORTS = ["ADNI", "ADRC", "HABS", "AD_DECODE"]

RESULTS_DIR_MAP = {
    "ADNI": "BrainAgePredictionADNI_stratified_groupcv_targetnorm_bagbiascorr_oofglobal",
    "ADRC": "BrainAgePredictionADRC_stratified_groupcv_targetnorm_bagbiascorr_oofglobal",
    "HABS": "BrainAgePredictionHABS_stratified_groupcv_targetnorm_bagbiascorr_oofglobal",
    "AD_DECODE": "BrainAgePredictionADDECODE_stratified_groupcv_targetnorm_bagbiascorr_oofglobal",
}

PREFIX_MAP = {
    "ADNI": "adni",
    "ADRC": "adrc",
    "HABS": "habs",
    "AD_DECODE": "addecode",
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

# Endpoint groups. The script first tries exact names, then conservative keyword fallback.
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
            "Clustering_Coeff",
            "Path_Length",
            "Global_Efficiency",
            "Local_Efficiency",
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
        "prefer_keywords": ["nfl", "neurofilament"],
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
            "transcriptomic_pca_1",
            "transcriptomic_pca_2",
            "transcriptomic_pca_3",
        ],
        "prefer_keywords": ["pc", "transcript", "rna", "gene", "pca"],
        "avoid_keywords": ["apoe", "diagnosis", "status", "cluster"],
        "value_range": None,
    },
}



# =============================================================================
# Curated endpoint-column map
# =============================================================================
# IMPORTANT:
# Panel C should never use broad keyword fallback for manuscript endpoint labels.
# The earlier permissive fallback could map e.g. "Hippocampal volume" to MMSE or
# Animal_Total when true hippocampal columns were absent. This curated map avoids
# that. If a curated endpoint is absent in a cohort, the script records it as
# missing/insufficient and Panel C shows a gray x-marker rather than plotting a
# misleading dot.
ENDPOINT_COLUMN_MAP: Dict[str, Dict[str, List[str]]] = {
    "Cognition": {
        "ADNI": [
            "cognition_composite_raw_clean", "Global_Cognition_Composite_raw_clean",
            "Global_Cognition_Composite", "Memory_Composite_raw_clean", "Memory_Composite",
            "MMSE_total", "MOCA_total_corrected", "MOCA_total", "ADAS_total", "CDRSB",
        ],
        "ADRC": [
            "Global_Cognition_Composite_raw_clean", "Global_Cognition_Composite",
            "MOCATOTS", "MMSE_total", "MOCA_total_corrected", "MOCA_total", "CDRSB",
        ],
        "HABS": [
            "Global_Cognition_Composite_raw_clean", "Global_Cognition_Composite",
            "CDR_M_Memory", "Animal_Total", "MOCA_total_corrected", "MOCA_total", "MMSE_total",
        ],
        "AD_DECODE": [
            "Global_Cognition_Composite_raw_clean", "Global_Cognition_Composite",
            "Composite_Familiarity_z_clean", "bckwds_total_correct", "MOCA_total", "MMSE_total",
        ],
    },

    # Use true hippocampal volume columns only. Do not fall back to cognitive scores.
    "Hippocampal volume": {
        "ADNI": [
            "Hippocampus_Total_pct_raw_clean", "Hippocampus_Total_pct", "Hippocampus_Total_raw_clean",
            "Hippocampus_Total", "Hippocampus_volume_raw_clean", "Hippocampus_volume",
            "Hippocampal_volume_raw_clean", "Hippocampal_volume", "HC_volume_raw_clean", "HC_volume",
            "Left_Hippocampus_volume", "Right_Hippocampus_volume", "HIPPATR", "HIPPOTOT",
        ],
        "ADRC": [
            "Hippocampus_Total_pct_raw_clean", "Hippocampus_Total_pct", "Hippocampus_Total_raw_clean",
            "Hippocampus_Total", "HIPPATR", "HIPPOTOT", "HC_volume_raw_clean", "HC_volume",
            "Left_Hippocampus_volume", "Right_Hippocampus_volume",
        ],
        "HABS": [
            "Hippocampus_Total_pct_raw_clean", "Hippocampus_Total_pct", "Hippocampus_Total_raw_clean",
            "Hippocampus_Total", "HC_volume_raw_clean", "HC_volume", "HIPPATR", "HIPPOTOT",
            "Left_Hippocampus_volume", "Right_Hippocampus_volume",
        ],
        "AD_DECODE": [
            "Hippocampus_Total_pct_raw_clean", "Hippocampus_Total_pct", "Hippocampus_Total_raw_clean",
            "Hippocampus_Total", "HC_volume_raw_clean", "HC_volume", "HIPPATR", "HIPPOTOT",
            "Left_Hippocampus_volume", "Right_Hippocampus_volume",
        ],
    },

    "Hippocampal FA": {
        "ADNI": [
            "Hippocampus_FA_Mean_raw_clean", "Hippocampus_FA_Mean", "Hippocampus_FA_Total_raw_clean",
            "Hippocampus_FA_Total", "Hippocampal_FA_raw_clean", "Hippocampal_FA", "HC_FA_raw_clean", "HC_FA",
            "Left_Hippocampus_FA", "Right_Hippocampus_FA",
        ],
        "ADRC": [
            "Hippocampus_FA_Mean_raw_clean", "Hippocampus_FA_Mean", "Hippocampus_FA_Total_raw_clean",
            "Hippocampus_FA_Total", "Hippocampal_FA_raw_clean", "Hippocampal_FA", "HC_FA_raw_clean", "HC_FA",
            "Left_Hippocampus_FA", "Right_Hippocampus_FA",
        ],
        "HABS": [
            "Hippocampus_FA_Mean_raw_clean", "Hippocampus_FA_Mean", "Hippocampus_FA_Total_raw_clean",
            "Hippocampus_FA_Total", "Hippocampal_FA_raw_clean", "Hippocampal_FA", "HC_FA_raw_clean", "HC_FA",
            "Left_Hippocampus_FA", "Right_Hippocampus_FA",
        ],
        "AD_DECODE": [
            "Hippocampus_FA_Mean_raw_clean", "Hippocampus_FA_Mean", "Hippocampus_FA_Total_raw_clean",
            "Hippocampus_FA_Total", "Hippocampal_FA_raw_clean", "Hippocampal_FA", "HC_FA_raw_clean", "HC_FA",
            "Left_Hippocampus_FA", "Right_Hippocampus_FA",
        ],
    },

    "Whole-brain volume": {
        "ADNI": [
            "Total_Brain_volume_raw_clean", "Total_Brain_volume", "TotalBrainVolume_raw_clean", "TotalBrainVolume",
            "Brain_Volume_raw_clean", "Brain_Volume", "GM_volume_raw_clean", "GM_volume", "WM_volume_raw_clean", "WM_volume",
            "Volume_mean_raw_clean", "Volume_mean", "Volume_median_raw_clean", "Volume_median",
        ],
        "ADRC": [
            "Total_Brain_volume_raw_clean", "Total_Brain_volume", "TotalBrainVolume_raw_clean", "TotalBrainVolume",
            "Brain_Volume_raw_clean", "Brain_Volume", "GM_volume_raw_clean", "GM_volume", "WM_volume_raw_clean", "WM_volume",
            "Volume_mean_raw_clean", "Volume_mean", "Volume_median_raw_clean", "Volume_median",
        ],
        "HABS": [
            "Total_Brain_volume_raw_clean", "Total_Brain_volume", "TotalBrainVolume_raw_clean", "TotalBrainVolume",
            "Brain_Volume_raw_clean", "Brain_Volume", "GM_volume_raw_clean", "GM_volume", "WM_volume_raw_clean", "WM_volume",
            "Volume_mean_raw_clean", "Volume_mean", "Volume_median_raw_clean", "Volume_median",
        ],
        "AD_DECODE": [
            "Total_Brain_volume_raw_clean", "Total_Brain_volume", "TotalBrainVolume_raw_clean", "TotalBrainVolume",
            "Brain_Volume_raw_clean", "Brain_Volume", "GM_volume_raw_clean", "GM_volume", "WM_volume_raw_clean", "WM_volume",
            "Volume_mean_raw_clean", "Volume_mean", "Volume_median_raw_clean", "Volume_median",
        ],
    },

    "Global FA": {
        "ADNI": ["FA_mean_raw_clean", "FA_mean", "FA_median_raw_clean", "FA_median", "Global_FA_raw_clean", "Global_FA", "Mean_FA", "Median_FA"],
        "ADRC": ["FA_mean_raw_clean", "FA_mean", "FA_median_raw_clean", "FA_median", "Global_FA_raw_clean", "Global_FA", "Mean_FA", "Median_FA"],
        "HABS": ["FA_mean_raw_clean", "FA_mean", "FA_median_raw_clean", "FA_median", "Global_FA_raw_clean", "Global_FA", "Mean_FA", "Median_FA"],
        "AD_DECODE": ["FA_mean_raw_clean", "FA_mean", "FA_median_raw_clean", "FA_median", "Global_FA_raw_clean", "Global_FA", "Mean_FA", "Median_FA"],
    },

    "Graph/network metric": {
        "ADNI": ["Clustering_Coeff_raw_clean", "Global_Efficiency_raw_clean", "Local_Efficiency_raw_clean", "Path_Length_raw_clean", "Clustering_Coeff", "Global_Efficiency", "Local_Efficiency", "Path_Length"],
        "ADRC": ["Clustering_Coeff_raw_clean", "Global_Efficiency_raw_clean", "Local_Efficiency_raw_clean", "Path_Length_raw_clean", "Clustering_Coeff", "Global_Efficiency", "Local_Efficiency", "Path_Length"],
        "HABS": ["Clustering_Coeff_raw_clean", "Global_Efficiency_raw_clean", "Local_Efficiency_raw_clean", "Path_Length_raw_clean", "Clustering_Coeff", "Global_Efficiency", "Local_Efficiency", "Path_Length"],
        "AD_DECODE": ["Clustering_Coeff_raw_clean", "Global_Efficiency_raw_clean", "Local_Efficiency_raw_clean", "Path_Length_raw_clean", "Clustering_Coeff", "Global_Efficiency", "Local_Efficiency", "Path_Length"],
    },

    "Amyloid / Aβ": {
        "ADNI": ["amyloid_42_raw_clean", "amyloid_42", "amyloid_40_raw_clean", "amyloid_40", "ABETA42_raw_clean", "ABETA42", "ABETA40_raw_clean", "ABETA40", "Centiloid_raw_clean", "Centiloid"],
        "ADRC": ["amyloid_42_raw_clean", "amyloid_42", "amyloid_40_raw_clean", "amyloid_40", "ABETA42_raw_clean", "ABETA42", "ABETA40_raw_clean", "ABETA40", "Centiloid_raw_clean", "Centiloid"],
        "HABS": ["Abeta42_raw_clean", "Abeta42", "Abeta40_raw_clean", "Abeta40", "amyloid_42_raw_clean", "amyloid_42", "amyloid_40_raw_clean", "amyloid_40", "Centiloid_raw_clean", "Centiloid"],
        "AD_DECODE": [],
    },

    "pTau": {
        "ADNI": ["ptau217_raw_clean", "PTAU217_raw_clean", "pTau217_raw_clean", "ptau181_raw_clean", "PTAU181_raw_clean", "pTau181_raw_clean", "PTAU_raw_clean", "pTau_raw_clean"],
        "ADRC": ["ptau217_raw_clean", "PTAU217_raw_clean", "pTau217_raw_clean", "ptau181_raw_clean", "PTAU181_raw_clean", "pTau181_raw_clean", "PTAU_raw_clean", "pTau_raw_clean"],
        "HABS": ["ptau217_raw_clean", "PTAU217_raw_clean", "pTau217_raw_clean", "ptau181_raw_clean", "PTAU181_raw_clean", "pTau181_raw_clean", "PTAU_raw_clean", "pTau_raw_clean"],
        "AD_DECODE": [],
    },

    "GFAP": {
        "ADNI": ["gfap_raw_clean", "GFAP_raw_clean", "PLASMA_GFAP_raw_clean", "gfap", "GFAP", "PLASMA_GFAP"],
        "ADRC": ["GFAP_raw_clean", "gfap_raw_clean", "PLASMA_GFAP_raw_clean", "GFAP", "gfap", "PLASMA_GFAP"],
        "HABS": ["GFAP_raw_clean", "gfap_raw_clean", "PLASMA_GFAP_raw_clean", "GFAP", "gfap", "PLASMA_GFAP"],
        "AD_DECODE": [],
    },

    "NfL": {
        "ADNI": ["nfl_raw_clean", "NFL_raw_clean", "NfL_raw_clean", "PLASMA_NFL_raw_clean", "nfl", "NFL", "NfL"],
        "ADRC": ["NFL_raw_clean", "nfl_raw_clean", "NfL_raw_clean", "PLASMA_NFL_raw_clean", "NFL", "nfl", "NfL"],
        "HABS": ["NFL_raw_clean", "nfl_raw_clean", "NfL_raw_clean", "PLASMA_NFL_raw_clean", "NFL", "nfl", "NfL"],
        "AD_DECODE": [],
    },

    "AD_DECODE transcriptomic PC": {
        "ADNI": [],
        "ADRC": [],
        "HABS": [],
        "AD_DECODE": ["PC1_raw_clean", "PC1", "PC2_raw_clean", "PC2", "PC3_raw_clean", "PC3", "transcriptomic_PC1", "RNA_PC1", "gene_PC1"],
    },
}


# =============================================================================
# CLI
# =============================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Figure 7 and Supplementary Figure 7 cross-cohort cBAG synthesis.")
    parser.add_argument("--base-dir", default=BASE_DIR, help="Combined validation directory.")
    parser.add_argument("--results-root", default=None, help="Root containing per-cohort BrainAgePrediction folders. Default: <base-dir>/..")
    parser.add_argument("--outdir", default=None, help=f"Output directory. Default: <base-dir>/{DEFAULT_OUTDIR_NAME}")
    parser.add_argument("--main-feature-set", default=DEFAULT_MAIN_FEATURE_SET, help="Feature set for biological endpoint panels.")
    parser.add_argument("--feature-sets", default=",".join(DEFAULT_FEATURE_SETS), help="Comma-separated feature sets for performance panel.")
    parser.add_argument("--cohorts", default=",".join(DEFAULT_COHORTS), help="Comma-separated cohorts.")
    parser.add_argument("--formats", default="png,pdf", help="Comma-separated output formats.")
    parser.add_argument("--dpi", type=int, default=450)
    parser.add_argument("--min-n", type=int, default=8, help="Minimum complete cases for endpoint correlations.")
    parser.add_argument("--metric", default="MAE", choices=["MAE", "RMSE", "R2", "r", "n"], help="Metric shown in main Figure 7A.")
    parser.add_argument("--cv-evaluation", default="OOF_BIAS_CORRECTED", help="Preferred evaluation row in CV summaries.")
    parser.add_argument("--max-endpoints-main", type=int, default=10, help="Max endpoints in main figure forest/summary panels.")
    parser.add_argument("--max-endpoints-supp", type=int, default=30, help="Max endpoints in supplementary all-endpoint panel.")
    parser.add_argument("--endpoint-order", default="replication", choices=["replication", "effect", "original"], help="Endpoint ordering for plots.")
    parser.add_argument("--endpoint-mapping-mode", default="strict", choices=["strict", "fallback"], help="strict uses curated cohort-specific endpoint columns; fallback allows keyword matching only when no curated map exists.")
    parser.add_argument("--show-missing-cohort-markers", type=int, choices=[0, 1], default=1, help="Show gray x markers in Panel C for unavailable/invalid cohort-endpoint pairs.")
    parser.add_argument("--main-title", default="Figure 7. Cross-cohort replication of cBAG associations across independent cohorts")
    parser.add_argument("--supp-title", default="Supplementary Figure 7. Extended cross-cohort cBAG replication and model-performance summary")
    return parser.parse_args()


# =============================================================================
# Basic utilities
# =============================================================================
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
        prefer_score = sum(tok in low for tok in prefer_keywords)
        if not token_hit and prefer_score == 0:
            continue
        score = int(token_hit) + 2 * prefer_score
        fallback.append((score, col))

    if not fallback:
        return None

    fallback.sort(key=lambda x: (-x[0], str(x[1]).lower()))
    return fallback[0][1]


def nice_label(name: Optional[str]) -> str:
    if name is None:
        return ""
    mapping = {
        "imaging_only": "Imaging only",
        "imaging_demographics": "Imaging + demographics",
        "imaging_biomarkers": "Imaging + biomarkers",
        "full": "Full",
        "full_no_cardiovascular": "Full, no cardiovascular",
        "R2": "R²",
        "r": "r",
        "n": "N",
        "MAE": "MAE",
        "RMSE": "RMSE",
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
    return re.sub(r"\s+", " ", raw).strip()


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.12, 1.08, label, transform=ax.transAxes, fontsize=15, fontweight="bold", va="top", ha="left")


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


def validation_dir(results_root: Path, cohort: str, feature_set: str) -> Path:
    return results_root / RESULTS_DIR_MAP[cohort] / f"ablation_{feature_set}" / "validation_figures"


# =============================================================================
# Subject-level biological validation input loading
# =============================================================================
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
        df["_input_path"] = str(path)

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
        })
    return pd.DataFrame(rows)


# =============================================================================
# CV metric loading: fixes MAE Panel A
# =============================================================================
def _standardize_metric_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    rename = {}
    aliases = {
        "cohort": ["cohort", "Cohort"],
        "feature_set": ["feature_set", "feature", "Feature_Set", "ablation", "Ablation"],
        "evaluation": ["evaluation", "Evaluation", "metric_set", "Metric_Set", "prediction_type"],
        "MAE": ["MAE", "mae", "mean_absolute_error"],
        "RMSE": ["RMSE", "rmse", "root_mean_squared_error"],
        "R2": ["R2", "R²", "r2", "R_squared", "R2_score"],
        "r": ["r", "R", "pearson_r", "Pearson_r", "PearsonR"],
        "n": ["n", "N", "n_subjects", "n_samples", "sample_size"],
    }
    for target, names in aliases.items():
        if target in df.columns:
            continue
        for name in names:
            if name in df.columns:
                rename[name] = target
                break
    df = df.rename(columns=rename)
    return df


def _load_one_cv_summary(results_root: Path, cohort: str, feature_set: str) -> Optional[pd.DataFrame]:
    train_prefix = f"{PREFIX_MAP[cohort]}_{feature_set}"
    ablation_dir = results_root / RESULTS_DIR_MAP[cohort] / f"ablation_{feature_set}"
    candidates = [
        ablation_dir / f"{train_prefix}_cv_summary_metrics.csv",
        ablation_dir / f"{train_prefix}_cv_summary_metrics.xlsx",
    ]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        print(f"[WARN] Missing CV summary for {cohort} {feature_set}: {candidates[0]}")
        return None

    df = pd.read_csv(path) if path.suffix.lower() == ".csv" else pd.read_excel(path)
    df = _standardize_metric_columns(df)
    df["cohort"] = cohort
    df["feature_set"] = feature_set
    df["source_path"] = str(path)
    return df


def load_cv_metrics_for_figure7(
    base_dir: Path,
    results_root: Path,
    cohorts: Sequence[str],
    feature_sets: Sequence[str],
    evaluation_filter: str = "OOF_BIAS_CORRECTED",
) -> pd.DataFrame:
    """
    Load MAE/RMSE/R2/r from the training/validation CV summary files.

    Preferred input:
        <base_dir>/combined_cv_summaries.csv

    Fallback:
        per-cohort *_cv_summary_metrics.csv files.
    """
    frames: List[pd.DataFrame] = []
    combined_candidates = [
        base_dir / "combined_cv_summaries.csv",
        base_dir / "combined_cv_summary_table.csv",
        base_dir / "combined_cv_summary.csv",
    ]
    combined_path = next((p for p in combined_candidates if p.exists()), None)

    if combined_path is not None:
        print(f"[INFO] Loading CV metrics from combined summary: {combined_path}")
        df = pd.read_csv(combined_path) if combined_path.suffix.lower() == ".csv" else pd.read_excel(combined_path)
        df = _standardize_metric_columns(df)
        df["source_path"] = str(combined_path)
        frames.append(df)
    else:
        print("[INFO] No combined CV summary found; falling back to per-ablation CV summaries.")
        for cohort in cohorts:
            for feature_set in feature_sets:
                df = _load_one_cv_summary(results_root, cohort, feature_set)
                if df is not None:
                    frames.append(df)

    if not frames:
        return pd.DataFrame(columns=["cohort", "feature_set", "n", "MAE", "RMSE", "R2", "r", "status"])

    cv = pd.concat(frames, ignore_index=True, sort=False)
    cv = _standardize_metric_columns(cv)

    # Keep requested cohorts/features if available.
    if "cohort" in cv.columns:
        cv = cv[cv["cohort"].astype(str).isin(cohorts)].copy()
    if "feature_set" in cv.columns:
        cv = cv[cv["feature_set"].astype(str).isin(feature_sets)].copy()

    # Prefer the same evaluation row used by validation plots.
    if "evaluation" in cv.columns and evaluation_filter:
        exact = cv[cv["evaluation"].astype(str) == evaluation_filter].copy()
        if exact.empty:
            # Try case-insensitive matching.
            exact = cv[cv["evaluation"].astype(str).str.upper() == evaluation_filter.upper()].copy()
        if exact.empty:
            print(f"[WARN] No rows for evaluation={evaluation_filter}; using all CV rows.")
        else:
            cv = exact

    # Wide format: direct metric columns.
    metric_cols = [c for c in ["MAE", "RMSE", "R2", "r", "n"] if c in cv.columns]
    if any(c in cv.columns for c in ["MAE", "RMSE", "R2", "r"]):
        keep = ["cohort", "feature_set"] + metric_cols
        keep = [c for c in keep if c in cv.columns]
        out = cv[keep].copy()
        for c in ["MAE", "RMSE", "R2", "r", "n"]:
            if c in out.columns:
                out[c] = pd.to_numeric(out[c], errors="coerce")

        agg = {c: "mean" for c in ["MAE", "RMSE", "R2", "r", "n"] if c in out.columns}
        out = out.groupby(["cohort", "feature_set"], as_index=False).agg(agg)
        for c in ["MAE", "RMSE", "R2", "r", "n"]:
            if c not in out.columns:
                out[c] = np.nan
        out["status"] = "loaded_from_cv_summary"
        return out

    # Long format: metric / value style.
    metric_name_cols = [c for c in ["metric", "Metric", "measure", "Measure"] if c in cv.columns]
    value_cols = [c for c in ["point_estimate", "value", "Value", "mean", "Mean"] if c in cv.columns]
    if metric_name_cols and value_cols:
        mcol = metric_name_cols[0]
        vcol = value_cols[0]
        sub = cv[cv[mcol].astype(str).isin(["MAE", "RMSE", "R2", "r"])].copy()
        sub[vcol] = pd.to_numeric(sub[vcol], errors="coerce")
        out = sub.pivot_table(
            index=["cohort", "feature_set"],
            columns=mcol,
            values=vcol,
            aggfunc="mean",
        ).reset_index()
        for c in ["MAE", "RMSE", "R2", "r", "n"]:
            if c not in out.columns:
                out[c] = np.nan
        out["status"] = "loaded_from_long_cv_summary"
        return out

    print("[WARN] CV summary exists but metric columns were not recognized.")
    print("[WARN] Columns found:", list(cv.columns))
    return pd.DataFrame(columns=["cohort", "feature_set", "n", "MAE", "RMSE", "R2", "r", "status"])


# =============================================================================
# Endpoint statistics
# =============================================================================
def fisher_ci_for_r(r: float, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    if pd.isna(r) or pd.isna(n) or n <= 3:
        return np.nan, np.nan
    rr = float(np.clip(r, -0.999999, 0.999999))
    z = np.arctanh(rr)
    se = 1.0 / math.sqrt(n - 3)
    zcrit = norm.ppf(1 - alpha / 2)
    lo = np.tanh(z - zcrit * se)
    hi = np.tanh(z + zcrit * se)
    return float(lo), float(hi)


def fixed_effect_meta_r(r_values: Sequence[float], n_values: Sequence[int]) -> Tuple[float, float, float, float]:
    """Return meta_r, ci_low, ci_high, z_p_value using Fisher-z fixed effect."""
    zs = []
    weights = []
    for r, n in zip(r_values, n_values):
        if pd.isna(r) or pd.isna(n) or n <= 3:
            continue
        zs.append(np.arctanh(float(np.clip(r, -0.999999, 0.999999))))
        weights.append(max(float(n) - 3.0, 1.0))
    if not zs:
        return np.nan, np.nan, np.nan, np.nan
    weights = np.asarray(weights, dtype=float)
    zs = np.asarray(zs, dtype=float)
    zbar = float(np.average(zs, weights=weights))
    se = float(1.0 / math.sqrt(weights.sum()))
    ci_low = float(np.tanh(zbar - 1.96 * se))
    ci_high = float(np.tanh(zbar + 1.96 * se))
    meta_r = float(np.tanh(zbar))
    z_stat = zbar / se if se > 0 else np.nan
    p = float(2 * norm.sf(abs(z_stat))) if np.isfinite(z_stat) else np.nan
    return meta_r, ci_low, ci_high, p


def choose_endpoint_column(
    df: pd.DataFrame,
    endpoint: str,
    cohort: str,
    candidates: Sequence[str],
    min_n: int,
    value_range: Optional[Tuple[float, float]],
    avoid_keywords: Sequence[str],
    prefer_keywords: Sequence[str],
    mapping_mode: str = "strict",
) -> Tuple[Optional[str], str, int]:
    """
    Choose a biologically valid endpoint column.

    In strict mode, curated endpoint/cohort mappings are authoritative. This
    prevents broad keyword fallback from mapping endpoints to unrelated columns
    such as MMSE_total for hippocampal volume.

    Returns
    -------
    col, status, available_n
    """
    curated = ENDPOINT_COLUMN_MAP.get(endpoint, {}).get(cohort, None)

    if curated is not None:
        if len(curated) == 0:
            return None, "not_applicable_or_missing_by_design", 0

        best_low_n_col = None
        best_low_n = 0
        for col in curated:
            if col not in df.columns:
                continue
            n = usable_n(df[col], value_range=value_range)
            if n >= min_n:
                return col, "mapped", n
            if n > best_low_n:
                best_low_n_col = col
                best_low_n = n

        if best_low_n_col is not None:
            return best_low_n_col, "mapped_but_insufficient_n", best_low_n

        return None, "mapped_column_not_found", 0

    if mapping_mode == "fallback":
        col = find_numeric_col(
            df,
            candidates=candidates,
            min_n=min_n,
            avoid_keywords=avoid_keywords,
            prefer_keywords=prefer_keywords,
            value_range=value_range,
        )
        if col is not None:
            return col, "fallback_keyword_match", usable_n(df[col], value_range=value_range)

    return None, "missing_endpoint", 0


def compute_endpoint_correlations(
    df: pd.DataFrame,
    cohorts: Sequence[str],
    min_n: int = 8,
    mapping_mode: str = "strict",
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
                    "endpoint": endpoint, "domain": domain, "cohort": cohort,
                    "column": None, "column_selection": "missing_cohort",
                    "n": 0, "r": np.nan, "p": np.nan,
                    "ci_low": np.nan, "ci_high": np.nan,
                    "status": "missing_cohort",
                })
                continue

            col, col_status, available_n = choose_endpoint_column(
                df=sub,
                endpoint=endpoint,
                cohort=cohort,
                candidates=candidates,  # type: ignore[arg-type]
                min_n=min_n,
                value_range=value_range,  # type: ignore[arg-type]
                avoid_keywords=avoid_keywords,  # type: ignore[arg-type]
                prefer_keywords=prefer_keywords,  # type: ignore[arg-type]
                mapping_mode=mapping_mode,
            )

            if col is None:
                rows.append({
                    "endpoint": endpoint, "domain": domain, "cohort": cohort,
                    "column": None, "column_selection": col_status,
                    "n": int(available_n), "r": np.nan, "p": np.nan,
                    "ci_low": np.nan, "ci_high": np.nan,
                    "status": "missing_endpoint",
                })
                continue

            x = clean_numeric(sub["_cbag"])
            y = clean_numeric(sub[col])
            if value_range is not None:
                lo, hi = value_range  # type: ignore[misc]
                y = y.where((y >= lo) & (y <= hi))
            ok = x.notna() & y.notna()
            n = int(ok.sum())

            if n < min_n or x[ok].nunique() < 2 or y[ok].nunique() < 2:
                rows.append({
                    "endpoint": endpoint, "domain": domain, "cohort": cohort,
                    "column": col, "column_selection": col_status,
                    "n": n, "r": np.nan, "p": np.nan,
                    "ci_low": np.nan, "ci_high": np.nan,
                    "status": "insufficient_n_or_variance",
                })
                continue

            r, p = pearsonr(x[ok], y[ok])
            ci_low, ci_high = fisher_ci_for_r(float(r), n)
            rows.append({
                "endpoint": endpoint,
                "domain": domain,
                "cohort": cohort,
                "column": col,
                "column_selection": col_status,
                "n": n,
                "r": float(r),
                "p": float(p),
                "ci_low": ci_low,
                "ci_high": ci_high,
                "status": "ok",
            })

    return pd.DataFrame(rows)

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
                "meta_ci_low": np.nan,
                "meta_ci_high": np.nan,
                "meta_p": np.nan,
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
        same_direction_n = int((np.sign(valid["r"]) == dominant_sign).sum()) if dominant_sign != 0 else int((np.sign(valid["r"]) == 0).sum())

        meta_r, meta_ci_low, meta_ci_high, meta_p = fixed_effect_meta_r(valid["r"].tolist(), valid["n"].tolist())

        pvals = valid["p"].dropna()
        fisher_p = np.nan
        if len(pvals) > 0:
            try:
                fisher_p = float(combine_pvalues(np.clip(pvals.values, 1e-300, 1.0), method="fisher")[1])
            except Exception:
                pass

        rows.append({
            "endpoint": endpoint,
            "domain": domain,
            "n_cohorts": int(valid["cohort"].nunique()),
            "total_n": int(valid["n"].sum()),
            "median_r": median_r,
            "mean_abs_r": float(valid["r"].abs().mean()),
            "meta_r": meta_r,
            "meta_ci_low": meta_ci_low,
            "meta_ci_high": meta_ci_high,
            "meta_p": meta_p,
            "same_direction_n": same_direction_n,
            "direction_consistency": float(same_direction_n / max(len(valid), 1)),
            "fisher_combined_p": fisher_p,
            "available_cohorts": ", ".join(valid["cohort"].astype(str).tolist()),
            "columns_used": "; ".join(f"{row.cohort}:{row.column}" for row in valid.itertuples() if pd.notna(row.column)),
        })

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["n_cohorts", "direction_consistency", "mean_abs_r", "total_n"], ascending=[False, False, False, False])
    return out


def endpoint_order_for_plot(summary: pd.DataFrame, max_endpoints: int, mode: str = "replication") -> List[str]:
    if summary.empty:
        return []
    tmp = summary.dropna(subset=["median_r"]).copy()
    if tmp.empty:
        return []
    if mode == "effect":
        tmp["_sort"] = tmp["median_r"].abs()
        tmp = tmp.sort_values(["n_cohorts", "_sort", "total_n"], ascending=[False, False, False])
    elif mode == "replication":
        tmp["_abs_meta"] = tmp["meta_r"].abs().fillna(tmp["median_r"].abs())
        tmp = tmp.sort_values(["n_cohorts", "direction_consistency", "_abs_meta", "total_n"], ascending=[False, False, False, False])
    else:
        endpoint_names = list(ENDPOINTS.keys())
        endpoints = [e for e in endpoint_names if e in set(tmp["endpoint"])]
        return endpoints[:max_endpoints]
    return tmp["endpoint"].head(max_endpoints).tolist()


# =============================================================================
# Plotting
# =============================================================================
def plot_metric_heatmap(ax: plt.Axes, metrics: pd.DataFrame, cohorts: Sequence[str], feature_sets: Sequence[str], metric: str, title: Optional[str] = None) -> None:
    if metrics.empty or metric not in metrics.columns:
        ax.text(0.5, 0.5, f"No {metric} metrics found", ha="center", va="center")
        ax.axis("off")
        return

    mat = metrics.pivot_table(index="cohort", columns="feature_set", values=metric, aggfunc="mean")
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
                txt = f"{val:.2f}" if metric in ["MAE", "RMSE", "R2", "r"] else f"{int(val)}"
            else:
                txt = "NA"
            ax.text(j, i, txt, ha="center", va="center", fontsize=7)

    title_metric = nice_label(metric)
    ax.set_title(title or f"Model performance across cohorts ({title_metric})", fontsize=11)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(title_metric, fontsize=8)
    cbar.ax.tick_params(labelsize=7)


def plot_cbag_distribution_panel(ax: plt.Axes, df: pd.DataFrame, cohorts: Sequence[str], main_feature_set: str) -> None:
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
        vals_plot = rng.choice(vals, size=min(len(vals), 800), replace=False) if len(vals) > 800 else vals
        jitter = rng.normal(loc=i, scale=0.045, size=len(vals_plot))
        ax.scatter(jitter, vals_plot, s=5, alpha=0.22, linewidths=0)
        ax.text(i, ax.get_ylim()[0], f"n={len(vals)}", ha="center", va="bottom", fontsize=7)

    ax.axhline(0, linewidth=1)
    ax.set_ylabel("cBAG")
    ax.set_title(f"cBAG distribution by cohort ({nice_label(main_feature_set)})", fontsize=11)
    ax.tick_params(axis="x", rotation=0, labelsize=9)


def marker_size_from_n(n: int, n_min: int, n_max: int) -> float:
    if n <= 0 or not np.isfinite(n):
        return 20.0
    if n_max <= n_min:
        return 70.0
    # sqrt scaling avoids over-dominating large cohorts.
    val = (math.sqrt(n) - math.sqrt(n_min)) / max(math.sqrt(n_max) - math.sqrt(n_min), 1e-9)
    return 35.0 + 130.0 * float(np.clip(val, 0, 1))


def plot_forest_panel(
    ax: plt.Axes,
    stats: pd.DataFrame,
    summary: pd.DataFrame,
    cohorts: Sequence[str],
    max_endpoints: int,
    endpoint_order: str,
    show_legend: bool = True,
    show_missing_markers: bool = True,
) -> None:
    endpoints = endpoint_order_for_plot(summary, max_endpoints=max_endpoints, mode=endpoint_order)
    if not endpoints:
        ax.text(0.5, 0.5, "No endpoint correlations available", ha="center", va="center")
        ax.axis("off")
        return

    valid_stats = stats[stats["endpoint"].isin(endpoints)].dropna(subset=["r"]).copy()
    n_min = int(valid_stats["n"].min()) if not valid_stats.empty else 1
    n_max = int(valid_stats["n"].max()) if not valid_stats.empty else 1

    y_positions = np.arange(len(endpoints))
    cohort_offsets = np.linspace(-0.27, 0.27, max(len(cohorts), 1))

    missing_plotted = False
    for c_idx, cohort in enumerate(cohorts):
        first_for_legend = True
        for e_idx, endpoint in enumerate(endpoints):
            row = stats[(stats["endpoint"] == endpoint) & (stats["cohort"] == cohort)]
            y = e_idx + cohort_offsets[c_idx]

            if row.empty or pd.isna(row.iloc[0]["r"]):
                if show_missing_markers:
                    ax.scatter(
                        [0.0],
                        [y],
                        s=32,
                        marker="x",
                        color="lightgray",
                        alpha=0.85,
                        linewidth=0.9,
                        label="Unavailable / insufficient" if not missing_plotted else None,
                        zorder=1,
                    )
                    missing_plotted = True
                continue

            r = float(row.iloc[0]["r"])
            n = int(row.iloc[0]["n"])
            lo = row.iloc[0].get("ci_low", np.nan)
            hi = row.iloc[0].get("ci_high", np.nan)

            if pd.notna(lo) and pd.notna(hi):
                ax.plot([float(lo), float(hi)], [y, y], linewidth=0.8, alpha=0.55, zorder=2)
            ax.scatter(
                [r],
                [y],
                s=marker_size_from_n(n, n_min, n_max),
                alpha=0.78,
                label=cohort if first_for_legend else None,
                edgecolor="black",
                linewidth=0.3,
                zorder=3,
            )
            first_for_legend = False

    # Add meta-r vertical ticks for each endpoint.
    for e_idx, endpoint in enumerate(endpoints):
        row = summary[summary["endpoint"] == endpoint]
        if row.empty:
            continue
        meta_r = row.iloc[0].get("meta_r", np.nan)
        if pd.notna(meta_r):
            ax.plot([float(meta_r), float(meta_r)], [e_idx - 0.38, e_idx + 0.38], linewidth=2.2, zorder=4)

    ax.axvline(0, linewidth=1)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(endpoints, fontsize=8)
    ax.set_xlabel("Pearson r with cBAG")
    ax.set_title("Cross-cohort endpoint associations", fontsize=11)
    ax.set_xlim(-1, 1)
    ax.invert_yaxis()

    if show_legend:
        leg1 = ax.legend(fontsize=7, frameon=False, loc="lower right", title="Cohort / availability")
        if leg1 is not None:
            leg1.get_title().set_fontsize(7)

    # Dot-size guide.
    if not valid_stats.empty and n_max > n_min:
        guide_ns = sorted(set([n_min, int(np.median(valid_stats["n"])), n_max]))
        handles = [plt.scatter([], [], s=marker_size_from_n(n, n_min, n_max), edgecolor="black", facecolor="gray", alpha=0.6) for n in guide_ns]
        labels = [f"n={n}" for n in guide_ns]
        leg2 = ax.legend(handles, labels, fontsize=6, frameon=False, loc="upper right", title="Dot size")
        if leg2 is not None:
            leg2.get_title().set_fontsize(6)
        if show_legend and 'leg1' in locals() and leg1 is not None:
            ax.add_artist(leg1)

def plot_replication_summary_panel(ax: plt.Axes, summary: pd.DataFrame, max_endpoints: int, endpoint_order: str) -> None:
    endpoints = endpoint_order_for_plot(summary, max_endpoints=max_endpoints, mode=endpoint_order)
    if not endpoints:
        ax.text(0.5, 0.5, "No replication summary available", ha="center", va="center")
        ax.axis("off")
        return

    tmp = summary.set_index("endpoint").loc[endpoints].reset_index()
    y = np.arange(len(tmp))
    x = tmp["meta_r"].where(tmp["meta_r"].notna(), tmp["median_r"]).astype(float)
    sizes = 55 + 45 * tmp["n_cohorts"].fillna(0).astype(float)

    for i, row in tmp.iterrows():
        lo = row.get("meta_ci_low", np.nan)
        hi = row.get("meta_ci_high", np.nan)
        if pd.notna(lo) and pd.notna(hi):
            ax.plot([float(lo), float(hi)], [i, i], linewidth=1.2, alpha=0.65)

    ax.scatter(x, y, s=sizes, alpha=0.82, edgecolor="black", linewidth=0.3)

    for i, row in tmp.iterrows():
        p = row.get("meta_p", np.nan)
        label = f"{int(row['same_direction_n'])}/{int(row['n_cohorts'])} same dir"
        if pd.notna(p):
            label += f", p={p:.1e}" if p < 0.001 else f", p={p:.3f}"
        ax.text(0.98, i, label, transform=ax.get_yaxis_transform(), fontsize=7, va="center", ha="right")

    ax.axvline(0, linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(tmp["endpoint"].tolist(), fontsize=8)
    ax.set_xlabel("Fixed-effect meta r with cBAG")
    ax.set_title("Replication consistency summary", fontsize=11)
    ax.set_xlim(-1, 1)
    ax.invert_yaxis()


def plot_endpoint_availability(ax: plt.Axes, stats: pd.DataFrame, cohorts: Sequence[str], max_endpoints: int, endpoint_order: str, summary: pd.DataFrame) -> None:
    endpoints = endpoint_order_for_plot(summary, max_endpoints=max_endpoints, mode=endpoint_order)
    if not endpoints:
        ax.text(0.5, 0.5, "No endpoint availability", ha="center", va="center")
        ax.axis("off")
        return

    mat = pd.DataFrame(index=endpoints, columns=cohorts, dtype=float)
    text = pd.DataFrame(index=endpoints, columns=cohorts, dtype=object)
    for endpoint in endpoints:
        for cohort in cohorts:
            row = stats[(stats["endpoint"] == endpoint) & (stats["cohort"] == cohort)]
            if row.empty or pd.isna(row.iloc[0]["r"]):
                mat.loc[endpoint, cohort] = 0
                text.loc[endpoint, cohort] = ""
            else:
                mat.loc[endpoint, cohort] = row.iloc[0]["n"]
                text.loc[endpoint, cohort] = f"n={int(row.iloc[0]['n'])}"

    im = ax.imshow(mat.fillna(0).to_numpy(dtype=float), aspect="auto")
    ax.set_xticks(range(len(cohorts)))
    ax.set_xticklabels(cohorts, rotation=0, fontsize=8)
    ax.set_yticks(range(len(endpoints)))
    ax.set_yticklabels(endpoints, fontsize=7)
    for i in range(len(endpoints)):
        for j in range(len(cohorts)):
            ax.text(j, i, text.iloc[i, j], ha="center", va="center", fontsize=6)
    ax.set_title("Endpoint availability by cohort", fontsize=11)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Complete cases", fontsize=8)
    cbar.ax.tick_params(labelsize=7)


# =============================================================================
# Figure builders
# =============================================================================
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
    max_endpoints: int,
    endpoint_order: str,
    title: str,
    show_missing_markers: bool = True,
) -> List[str]:
    fig, axes = plt.subplots(2, 2, figsize=(15.8, 11.2))
    ax_a, ax_b, ax_c, ax_d = axes.ravel()

    plot_metric_heatmap(ax_a, metrics, cohorts, feature_sets, metric=metric)
    panel_label(ax_a, "A")

    plot_cbag_distribution_panel(ax_b, main_df, cohorts, main_feature_set)
    panel_label(ax_b, "B")

    plot_forest_panel(ax_c, endpoint_stats, replication_summary, cohorts, max_endpoints=max_endpoints, endpoint_order=endpoint_order, show_legend=True, show_missing_markers=show_missing_markers)
    panel_label(ax_c, "C")

    plot_replication_summary_panel(ax_d, replication_summary, max_endpoints=max_endpoints, endpoint_order=endpoint_order)
    panel_label(ax_d, "D")

    fig.suptitle(title, fontsize=15, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return save_all(fig, outdir / "Figure7_cross_cohort_replication", formats=formats, dpi=dpi)


def make_supplementary_figure7(
    metrics: pd.DataFrame,
    endpoint_stats: pd.DataFrame,
    replication_summary: pd.DataFrame,
    cohorts: Sequence[str],
    feature_sets: Sequence[str],
    outdir: Path,
    formats: Sequence[str],
    dpi: int,
    max_endpoints: int,
    endpoint_order: str,
    title: str,
    show_missing_markers: bool = True,
) -> List[str]:
    fig, axes = plt.subplots(2, 2, figsize=(17.5, 12.5))
    ax_a, ax_b, ax_c, ax_d = axes.ravel()

    plot_metric_heatmap(ax_a, metrics, cohorts, feature_sets, metric="MAE", title="OOF bias-corrected MAE across cohorts and ablations")
    panel_label(ax_a, "A")

    plot_metric_heatmap(ax_b, metrics, cohorts, feature_sets, metric="R2", title="OOF bias-corrected R² across cohorts and ablations")
    panel_label(ax_b, "B")

    plot_forest_panel(ax_c, endpoint_stats, replication_summary, cohorts, max_endpoints=max_endpoints, endpoint_order=endpoint_order, show_legend=True, show_missing_markers=show_missing_markers)
    ax_c.set_title("All available endpoint associations", fontsize=11)
    panel_label(ax_c, "C")

    plot_endpoint_availability(ax_d, endpoint_stats, cohorts, max_endpoints=max_endpoints, endpoint_order=endpoint_order, summary=replication_summary)
    panel_label(ax_d, "D")

    fig.suptitle(title, fontsize=15, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return save_all(fig, outdir / "Supplementary_Figure7_cross_cohort_replication", formats=formats, dpi=dpi)


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

    print(f"BASE_DIR:          {base_dir}")
    print(f"RESULTS_ROOT:      {results_root}")
    print(f"OUTDIR:            {outdir}")
    print(f"Main feature set:  {args.main_feature_set}")
    print(f"CV evaluation:     {args.cv_evaluation}")
    print(f"Cohorts:           {', '.join(cohorts)}")
    print(f"Feature sets:      {', '.join(feature_sets)}")
    print(f"Formats:           {', '.join(formats)}")

    # Panel A metrics: read actual saved CV metrics, not subject-level validation tables.
    metrics = load_cv_metrics_for_figure7(
        base_dir=base_dir,
        results_root=results_root,
        cohorts=cohorts,
        feature_sets=feature_sets,
        evaluation_filter=args.cv_evaluation,
    )

    # Load subject-level biological validation inputs for cBAG distributions and endpoint associations.
    all_frames: List[pd.DataFrame] = []
    dataset_summaries: List[pd.DataFrame] = []
    for feature_set in feature_sets:
        df = load_subject_level_tables(results_root, cohorts, feature_set)
        if df.empty:
            print(f"[WARN] No usable subject-level data found for feature set: {feature_set}")
            continue
        all_frames.append(df)
        dataset_summaries.append(summarize_dataset(df, feature_set, cohorts))

    if not all_frames:
        raise SystemExit(
            "No usable subject-level validation tables found. "
            "Run brainage_validation_ALL_COHORTS_BAG_OOFGLOBAL_FINAL_directpaths.py first or check --results-root."
        )

    all_df = pd.concat(all_frames, ignore_index=True, sort=False)
    main_df = all_df[all_df["feature_set"] == args.main_feature_set].copy()
    if main_df.empty:
        raise SystemExit(f"No usable data found for main feature set: {args.main_feature_set}.")

    endpoint_stats = compute_endpoint_correlations(
        main_df,
        cohorts=cohorts,
        min_n=args.min_n,
        mapping_mode=args.endpoint_mapping_mode,
    )
    replication_summary = summarize_replication(endpoint_stats)

    # Save tables before plotting, so problems are easy to inspect.
    dataset_summary = pd.concat(dataset_summaries, ignore_index=True, sort=False) if dataset_summaries else pd.DataFrame()
    dataset_summary_path = outdir / "figure7_dataset_summary.csv"
    metrics_path = outdir / "figure7_model_metric_summary.csv"
    endpoint_stats_path = outdir / "figure7_endpoint_correlation_stats.csv"
    replication_summary_path = outdir / "figure7_replication_summary.csv"

    dataset_summary.to_csv(dataset_summary_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    endpoint_stats.to_csv(endpoint_stats_path, index=False)
    replication_summary.to_csv(replication_summary_path, index=False)

    fig7_outputs = make_figure7(
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
        max_endpoints=args.max_endpoints_main,
        endpoint_order=args.endpoint_order,
        title=args.main_title,
        show_missing_markers=bool(args.show_missing_cohort_markers),
    )

    supp7_outputs = make_supplementary_figure7(
        metrics=metrics,
        endpoint_stats=endpoint_stats,
        replication_summary=replication_summary,
        cohorts=cohorts,
        feature_sets=feature_sets,
        outdir=outdir,
        formats=formats,
        dpi=args.dpi,
        max_endpoints=args.max_endpoints_supp,
        endpoint_order=args.endpoint_order,
        title=args.supp_title,
        show_missing_markers=bool(args.show_missing_cohort_markers),
    )

    manifest_path = outdir / "figure7_manifest.csv"
    manifest = pd.DataFrame([
        {
            "figure": "Figure 7",
            "description": "Cross-cohort cBAG synthesis: CV performance, cBAG distribution, n-scaled endpoint forest plot with 95% CI, replication summary",
            "main_feature_set": args.main_feature_set,
            "cv_evaluation": args.cv_evaluation,
            "feature_sets": ",".join(feature_sets),
            "cohorts": ",".join(cohorts),
            "endpoint_mapping_mode": args.endpoint_mapping_mode,
            "show_missing_cohort_markers": bool(args.show_missing_cohort_markers),
            "outputs": ";".join(fig7_outputs),
        },
        {
            "figure": "Supplementary Figure 7",
            "description": "Extended cross-cohort performance, endpoint association, and endpoint availability summary",
            "main_feature_set": args.main_feature_set,
            "cv_evaluation": args.cv_evaluation,
            "feature_sets": ",".join(feature_sets),
            "cohorts": ",".join(cohorts),
            "endpoint_mapping_mode": args.endpoint_mapping_mode,
            "show_missing_cohort_markers": bool(args.show_missing_cohort_markers),
            "outputs": ";".join(supp7_outputs),
        },
    ])
    manifest.to_csv(manifest_path, index=False)

    print("\nSaved Figure 7 outputs:")
    for output in fig7_outputs:
        print(f"  {output}")
    print("\nSaved Supplementary Figure 7 outputs:")
    for output in supp7_outputs:
        print(f"  {output}")
    print("\nSaved tables:")
    print(f"  Dataset summary:       {dataset_summary_path}")
    print(f"  Model metric summary:  {metrics_path}")
    print(f"  Endpoint stats:        {endpoint_stats_path}")
    print(f"  Replication summary:   {replication_summary_path}")
    print(f"  Manifest:              {manifest_path}")

    if not replication_summary.empty:
        cols = ["endpoint", "domain", "n_cohorts", "total_n", "meta_r", "meta_ci_low", "meta_ci_high", "same_direction_n", "meta_p"]
        cols = [c for c in cols if c in replication_summary.columns]
        print("\nTop replication endpoints:")
        print(replication_summary[cols].head(12).to_string(index=False))


if __name__ == "__main__":
    main()
