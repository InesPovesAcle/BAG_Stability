#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure 8 FINAL: harmonized-metadata longitudinal delta-cBAG validation
=====================================================================

Purpose
-------
Build Figure 8 using scan/session-level full-cohort cBAG and properly
harmonized metadata endpoints.

Key design
----------
1) Read full-cohort validation rows for cBAG:
   <results-root>/<BrainAgePrediction...>/ablation_<feature_set>/validation_figures_full_cohort/
       subject_level_validation_input_enriched_for_Figure4.csv
   with fallback to:
       subject_level_validation_input.csv

2) Merge each validation scan to final harmonized metadata:
   $WORK/ines/data/harmonization/harmonized_metadata/<COHORT>_harmonized_metadata.csv

3) Merge by normalized connectome/session key, using the same idea as Figure 5:
   choose the validation/metadata key pair with largest normalized overlap.

4) Compute within-person deltas:
   ΔcBAG = follow-up cBAG - baseline cBAG
   Δendpoint = follow-up endpoint - baseline endpoint

5) Test association:
   ΔcBAG vs Δendpoint

Default cohorts:
   ADNI, HABS

Outputs
-------
<base-dir>/Figure8_longitudinal_delta_cBAG_HARMONIZED_FINAL/
    Figure8_longitudinal_delta_cBAG_HARMONIZED_FINAL.png/pdf
    Supplementary_Figure8_longitudinal_delta_cBAG_HARMONIZED_FINAL.png/pdf
    longitudinal_delta_subject_table.csv
    longitudinal_delta_association_stats.csv
    longitudinal_delta_dataset_summary.csv
    longitudinal_endpoint_column_selection.csv
    longitudinal_merge_QA.csv
    figure8_manifest.csv

Run
---
python Figure8_longitudinal_delta_cBAG_HARMONIZED_FINAL.py \
  --feature-set imaging_only \
  --cohorts ADNI,HABS
"""

from __future__ import annotations

import argparse
import math
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


# =============================================================================
# Defaults
# =============================================================================

WORK = Path(os.environ.get("WORK", "/mnt/newStor/paros/paros_WORK"))
RESULTS_ROOT = WORK / "ines/results"
BASE_DIR = RESULTS_ROOT / "BrainAgeValidation_AllCohorts_BAGBiasCorr_OOFGlobal_BiologicalValidation"
HARMONIZED_METADATA_DIR = WORK / "ines/data/harmonization/harmonized_metadata"

DEFAULT_OUTDIR_NAME = "Figure8_longitudinal_delta_cBAG_HARMONIZED_FINAL"
DEFAULT_FEATURE_SET = "imaging_only"
DEFAULT_COHORTS = ["ADNI", "HABS"]

RESULTS_DIR_MAP = {
    "ADNI": "BrainAgePredictionADNI_stratified_groupcv_targetnorm_bagbiascorr_oofglobal",
    "ADRC": "BrainAgePredictionADRC_stratified_groupcv_targetnorm_bagbiascorr_oofglobal",
    "HABS": "BrainAgePredictionHABS_stratified_groupcv_targetnorm_bagbiascorr_oofglobal",
    "AD_DECODE": "BrainAgePredictionADDECODE_stratified_groupcv_targetnorm_bagbiascorr_oofglobal",
}

CBAG_PRIORITY = [
    "cBAG_global_raw_clean",
    "cBAG_global",
    "cBAG_raw_clean",
    "cBAG",
    "cBAG_oof_global_raw_clean",
    "cBAG_oof_global",
    "cBAG_foldwise_raw_clean",
    "cBAG_foldwise",
    "BAG_raw_clean",
    "BAG",
    "BAG_raw",
]

VAL_KEY_CANDIDATES = [
    "connectome_key", "connectome_full_key", "CONNECTOME_KEY_CLEAN",
    "CONNECTOME_KEY_USED_FOR_INTERSECTION", "subject_match", "subject_source",
    "Subject_ID", "subject_id", "Subject", "participant_id", "DWI", "DWI_key",
    "runno", "graph_id", "MRI_Exam", "MRI_Exam_fixed", "PTID", "RID", "ID",
    "id", "match_id",
]
META_KEY_CANDIDATES = [
    "CONNECTOME_KEY_USED_FOR_INTERSECTION", "CONNECTOME_KEY_CLEAN", "table1_session_key",
    "DWI", "DWI_key", "connectome_key", "connectome_full_key", "graph_id", "runno",
    "MRI_Exam", "MRI_Exam_fixed", "PTID", "RID", "subject_id", "Subject", "ID",
    "id", "match_id",
]

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


# Curated endpoint candidates.
# The script tries harmonized metadata columns first by default.
ENDPOINTS: Dict[str, Dict[str, object]] = {
    "Global cognition": {
        "domain": "Clinical",
        "candidates": [
            "Global_Cognition_Composite_raw_clean",
            "Global_Cognition_Composite",
            "cognition_composite_raw_clean",
            "cognition_composite",
            "Memory_Composite_raw_clean",
            "Memory_Composite",
            "MOCA_total_corrected_raw_clean",
            "MOCA_total_corrected",
            "MOCA_total_raw_clean",
            "MOCA_total",
            "MMSE_total_raw_clean",
            "MMSE_total",
            "ADAS_total_raw_clean",
            "ADAS_total",
            "CDRSB_raw_clean",
            "CDRSB",
        ],
        "prefer_keywords": ["global_cognition", "cognition", "memory", "moca", "mmse", "animal", "adas", "cdr"],
        "avoid_keywords": ["diagnosis", "status", "group", "dx", "apoe", "risk"],
        "value_range": None,
    },
    "Hippocampal volume": {
        "domain": "Neuroimaging",
        "candidates": [
            "Hc_volume_pct_brain_raw_clean",
            "Hc_volume_pct_brain",
            "Hc_volume_mm3_raw_clean",
            "Hc_volume_mm3",
            "Hippocampus_Total_pct_raw_clean",
            "Hippocampus_Total_pct",
            "Hippocampus_Total_raw_clean",
            "Hippocampus_Total",
            "HC_volume_raw_clean",
            "HC_volume",
            "Hippocampus_volume_raw_clean",
            "Hippocampus_volume",
            "Left_Hippocampus_volume",
            "Right_Hippocampus_volume",
        ],
        "prefer_keywords": ["hc_volume", "hippocampus", "hippocampal", "volume", "pct_brain"],
        "avoid_keywords": ["fa", "rd", "adc", "cluster", "path", "efficiency", "diagnosis", "status"],
        "value_range": None,
    },
    "Hippocampal FA": {
        "domain": "Neuroimaging",
        "candidates": [
            "Hc_FA_raw_clean",
            "Hc_FA",
            "Hippocampus_FA_Mean_raw_clean",
            "Hippocampus_FA_Mean",
            "Hippocampus_FA_Total_raw_clean",
            "Hippocampus_FA_Total",
            "Hippocampal_FA_raw_clean",
            "Hippocampal_FA",
            "Left_Hippocampus_FA",
            "Right_Hippocampus_FA",
        ],
        "prefer_keywords": ["hc_fa", "hippocampus", "hippocampal", "fa"],
        "avoid_keywords": ["volume", "vol", "rd", "adc", "diagnosis", "status"],
        "value_range": (0.0, 1.0),
    },
    "Hippocampal RD": {
        "domain": "Neuroimaging",
        "candidates": ["Hc_RD_raw_clean", "Hc_RD", "Hippocampus_RD_Mean_raw_clean", "Hippocampus_RD_Mean"],
        "prefer_keywords": ["hc_rd", "hippocampus", "hippocampal", "rd"],
        "avoid_keywords": ["volume", "fa", "diagnosis", "status"],
        "value_range": None,
    },
    "Hippocampal ADC": {
        "domain": "Neuroimaging",
        "candidates": ["Hc_ADC_raw_clean", "Hc_ADC", "Hippocampus_ADC_Mean_raw_clean", "Hippocampus_ADC_Mean"],
        "prefer_keywords": ["hc_adc", "hippocampus", "hippocampal", "adc"],
        "avoid_keywords": ["volume", "fa", "diagnosis", "status"],
        "value_range": None,
    },
    "Whole-brain volume": {
        "domain": "Neuroimaging",
        "candidates": [
            "Total_Brain_volume_raw_clean",
            "Total_Brain_volume",
            "TotalBrainVolume_raw_clean",
            "TotalBrainVolume",
            "Brain_Volume_raw_clean",
            "Brain_Volume",
            "Volume_mean_raw_clean",
            "Volume_mean",
            "GM_volume_raw_clean",
            "GM_volume",
            "WM_volume_raw_clean",
            "WM_volume",
        ],
        "prefer_keywords": ["total_brain", "brain_volume", "volume"],
        "avoid_keywords": ["hipp", "hc_", "fa", "rd", "adc", "diagnosis", "status"],
        "value_range": None,
    },
    "Global FA": {
        "domain": "Neuroimaging",
        "candidates": [
            "Total_Brain_FA_raw_clean",
            "Total_Brain_FA",
            "FA_mean_raw_clean",
            "FA_mean",
            "FA_median_raw_clean",
            "FA_median",
            "Global_FA_raw_clean",
            "Global_FA",
        ],
        "prefer_keywords": ["total_brain_fa", "global_fa", "fa_mean", "fa_median"],
        "avoid_keywords": ["hipp", "hc_", "volume", "diagnosis", "status"],
        "value_range": (0.0, 1.0),
    },
    "Graph clustering": {
        "domain": "Network",
        "candidates": [
            "Clustering_Coeff_raw_clean",
            "Clustering_Coeff",
            "Total_graph_clustering_coeff_raw_clean",
            "Total_graph_clustering_coeff",
            "clustering_coefficient",
        ],
        "prefer_keywords": ["clustering"],
        "avoid_keywords": ["cluster_id", "diagnosis", "status"],
        "value_range": None,
    },
    "Graph path length": {
        "domain": "Network",
        "candidates": [
            "Path_Length_raw_clean",
            "Path_Length",
            "Total_graph_path_length_raw_clean",
            "Total_graph_path_length",
            "Characteristic_Path_Length",
        ],
        "prefer_keywords": ["path_length", "path"],
        "avoid_keywords": ["diagnosis", "status"],
        "value_range": None,
    },
    "Global efficiency": {
        "domain": "Network",
        "candidates": ["Global_Efficiency_raw_clean", "Global_Efficiency", "global_efficiency"],
        "prefer_keywords": ["global_efficiency", "efficiency"],
        "avoid_keywords": ["local", "diagnosis", "status"],
        "value_range": None,
    },
    "Local efficiency": {
        "domain": "Network",
        "candidates": ["Local_Efficiency_raw_clean", "Local_Efficiency", "local_efficiency"],
        "prefer_keywords": ["local_efficiency"],
        "avoid_keywords": ["global", "diagnosis", "status"],
        "value_range": None,
    },
    "Amyloid / Aβ": {
        "domain": "Biomarker",
        "candidates": ["amyloid_42_raw_clean", "amyloid_42", "Abeta42_raw_clean", "Abeta42", "ABETA42_raw_clean", "ABETA42", "Centiloid_raw_clean", "Centiloid"],
        "prefer_keywords": ["amyloid", "abeta", "centiloid"],
        "avoid_keywords": ["status", "positive", "binary", "diagnosis", "dx"],
        "value_range": None,
    },
    "pTau": {
        "domain": "Biomarker",
        "candidates": ["ptau217_raw_clean", "ptau217", "pTau_raw_clean", "pTau", "PTAU_raw_clean", "PTAU"],
        "prefer_keywords": ["ptau", "p_tau"],
        "avoid_keywords": ["status", "positive", "binary", "diagnosis", "dx"],
        "value_range": None,
    },
    "BMI": {
        "domain": "Vascular/metabolic",
        "candidates": ["BMI_raw_clean", "BMI", "bmi_raw_clean", "bmi", "BMI_calculated_raw_clean", "BMI_calculated", "PHC_BMI_raw_clean", "PHC_BMI"],
        "prefer_keywords": ["bmi"],
        "avoid_keywords": ["zscore", "status", "binary", "diagnosis", "dx"],
        "value_range": (10.0, 80.0),
    },
    "Systolic BP": {
        "domain": "Vascular/metabolic",
        "candidates": ["bp_sys_raw_clean", "bp_sys", "systolic_raw_clean", "systolic", "SBP_raw_clean", "SBP"],
        "prefer_keywords": ["systolic", "sbp", "bp_sys"],
        "avoid_keywords": ["status", "binary", "diagnosis", "dx"],
        "value_range": (70.0, 250.0),
    },
}


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate Figure 8 using final harmonized metadata merge.")
    p.add_argument("--results-root", default=str(RESULTS_ROOT))
    p.add_argument("--base-dir", default=str(BASE_DIR))
    p.add_argument("--metadata-dir", default=str(HARMONIZED_METADATA_DIR))
    p.add_argument("--outdir", default=None)
    p.add_argument("--feature-set", default=DEFAULT_FEATURE_SET)
    p.add_argument("--cohorts", default=",".join(DEFAULT_COHORTS))
    p.add_argument("--formats", default="png,pdf")
    p.add_argument("--dpi", type=int, default=450)
    p.add_argument("--min-n", type=int, default=12)
    p.add_argument("--corr-method", default="pearson", choices=["pearson", "spearman"])
    p.add_argument("--input-kind", default="auto", choices=["auto", "enriched", "validation"])
    p.add_argument("--endpoint-source-priority", default="metadata", choices=["metadata", "validation", "any"])
    p.add_argument("--max-endpoints-main", type=int, default=8)
    p.add_argument("--max-endpoints-supp", type=int, default=30)
    p.add_argument("--main-title", default="Figure 8. Longitudinal cBAG and within-person biological change")
    p.add_argument("--supp-title", default="Supplementary Figure 8. Extended longitudinal cBAG associations")
    return p.parse_args()


# =============================================================================
# Utilities
# =============================================================================

def normalize_name(x: object) -> str:
    s = str(x).lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return re.sub(r"_+", "_", s).strip("_")


def clean_numeric(series: pd.Series) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce").copy()
    for val in SENTINEL_VALUES:
        x = x.mask(x == val, np.nan)
    return x.replace([np.inf, -np.inf], np.nan)


def first_existing(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    exact = {str(c): c for c in df.columns}
    lower = {str(c).lower(): c for c in df.columns}
    norm = {normalize_name(c): c for c in df.columns}
    for c in candidates:
        if c in exact:
            return exact[c]
    for c in candidates:
        if str(c).lower() in lower:
            return lower[str(c).lower()]
    for c in candidates:
        nc = normalize_name(c)
        if nc in norm:
            return norm[nc]
    return None


def normalize_connectome_key(x: object, cohort: Optional[str] = None) -> Optional[str]:
    if pd.isna(x):
        return None
    s = str(x).strip()
    if not s or s.lower() in {"nan", "none", "<na>"}:
        return None

    s = Path(s).stem
    s = re.sub(r"^\._", "", s)
    s = re.sub(r"\.0$", "", s)
    s = re.sub(r"_conn_plain$", "", s, flags=re.I)
    s = re.sub(r"conn_plain$", "", s, flags=re.I)
    s = re.sub(r"_connectomics$", "", s, flags=re.I)

    m = re.search(r"ADRC\s*0*(\d{4})", s, flags=re.I)
    if m:
        return f"d{int(m.group(1)):04d}"

    m = re.search(r"\b([RH]\d{4,5}_y\d+)\b", s, flags=re.I)
    if m:
        return m.group(1).lower()

    m = re.search(r"\b(S\d{5})\b", s, flags=re.I)
    if m:
        return m.group(1).lower()

    if cohort == "AD_DECODE" and re.fullmatch(r"\d+(?:\.0)?", s):
        return f"s{int(float(s)):05d}"

    m = re.search(r"\b(D\d{4})\b", s, flags=re.I)
    if m:
        return m.group(1).lower()

    m = re.search(r"\b([A-Za-z]\d{5})\b", s, flags=re.I)
    if m:
        return m.group(1).lower()
    m = re.search(r"\b([A-Za-z]\d{4})\b", s, flags=re.I)
    if m:
        return m.group(1).lower()

    s2 = re.sub(r"_master_T.*$", "", s, flags=re.I)
    s2 = re.sub(r"_temp_T.*$", "", s2, flags=re.I)
    s2 = re.sub(r"_T\d?.*$", "", s2, flags=re.I)
    if s2 != s:
        return normalize_connectome_key(s2, cohort=cohort)

    return normalize_name(s) or None


def parse_longitudinal_id(x) -> Tuple[str, Optional[float]]:
    key = normalize_connectome_key(x)
    if key is None:
        return str(x).upper(), None
    m = re.match(r"^(.+?)_y(\d+)$", key, flags=re.I)
    if m:
        return m.group(1).upper(), float(m.group(2))
    # HABS/ADNI without y visit cannot be used for longitudinal delta
    return key.upper(), None


def _first_existing_local(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    """Case-insensitive, normalization-tolerant column finder."""
    if df is None or df.empty:
        return None
    exact = {str(c): c for c in df.columns}
    lower = {str(c).lower(): c for c in df.columns}
    norm = {normalize_name(c): c for c in df.columns}
    for c in candidates:
        if c in exact:
            return exact[c]
    for c in candidates:
        if str(c).lower() in lower:
            return lower[str(c).lower()]
    for c in candidates:
        nc = normalize_name(c)
        if nc in norm:
            return norm[nc]
    return None


def parse_subject_visit_from_pair(subject_value, visit_value) -> Tuple[str, Optional[float]]:
    """Parse a base subject column plus a visit/year column."""
    if pd.isna(subject_value) or pd.isna(visit_value):
        return None, None

    subject = normalize_connectome_key(subject_value)
    if subject is None:
        subject = normalize_name(subject_value)
    if subject is None:
        return None, None

    # If subject itself contains _y#, strip it down to the base subject.
    m = re.search(r"^(.+?)_y(\d+)$", subject, flags=re.I)
    if m:
        subject = m.group(1)

    vraw = str(visit_value).strip()
    if not vraw or vraw.lower() in {"nan", "none", "<na>"}:
        return subject.upper(), None

    # Accept y0/y2/Y4, visit labels, or numeric year/visit.
    m = re.search(r"y\s*(\d+(?:\.\d+)?)", vraw, flags=re.I)
    if m:
        return subject.upper(), float(m.group(1))
    m = re.search(r"(\d+(?:\.\d+)?)", vraw)
    if m:
        return subject.upper(), float(m.group(1))
    return subject.upper(), None


def choose_longitudinal_subject_visit(merged: pd.DataFrame, cohort: str, default_col: str) -> Tuple[pd.Series, pd.Series, str]:
    """
    Choose the best source for longitudinal subject and visit.

    For ADNI, keys like R1195_y0/R1195_y4 usually work directly. For HABS,
    connectome_key values may be session-specific and not repeated by base key,
    while DWI_subject + DWI_visit/DWI_visit_year can preserve the true repeated
    subject identity. This function tries paired subject/visit columns first
    and selects the option with the largest number of subjects having >=2 visits.
    """
    candidates = []

    # Candidate 1: parse visit from a single scan/session key.
    if default_col and default_col in merged.columns:
        parsed = merged[default_col].apply(parse_longitudinal_id)
        subj = parsed.apply(lambda z: z[0])
        visit = parsed.apply(lambda z: z[1])
        visit_ok = visit.notna()
        n_repeated = int(
            merged.loc[visit_ok].assign(_s=subj[visit_ok], _v=visit[visit_ok])
            .groupby("_s")["_v"].nunique().ge(2).sum()
        ) if visit_ok.any() else 0
        candidates.append((n_repeated, subj, visit, f"single:{default_col}"))

    # Candidate 2+: parse from explicit base-subject and visit columns.
    subject_cols = [
        "DWI_subject", "subject_id", "Subject", "participant_id", "PTID", "RID",
        "runno_subject", "subject_base", "meta__DWI_subject", "meta__subject_id",
        "meta__Subject", "meta__PTID", "meta__RID",
    ]
    visit_cols = [
        "DWI_visit", "DWI_visit_year", "Visit_ID", "visit", "visit_year",
        "timepoint", "year", "meta__DWI_visit", "meta__DWI_visit_year",
        "meta__Visit_ID", "meta__visit", "meta__visit_year",
    ]
    available_subject_cols = [c for c in subject_cols if c in merged.columns]
    available_visit_cols = [c for c in visit_cols if c in merged.columns]

    for sc in available_subject_cols:
        for vc in available_visit_cols:
            parsed = [parse_subject_visit_from_pair(s, v) for s, v in zip(merged[sc], merged[vc])]
            subj = pd.Series([z[0] for z in parsed], index=merged.index, dtype="object")
            visit = pd.Series([z[1] for z in parsed], index=merged.index, dtype="float")
            visit_ok = visit.notna() & subj.notna()
            if not visit_ok.any():
                continue
            n_repeated = int(
                pd.DataFrame({"_s": subj[visit_ok], "_v": visit[visit_ok]})
                .groupby("_s")["_v"].nunique().ge(2).sum()
            )
            candidates.append((n_repeated, subj, visit, f"pair:{sc}+{vc}"))

    if not candidates:
        parsed = merged[default_col].apply(parse_longitudinal_id)
        return parsed.apply(lambda z: z[0]), parsed.apply(lambda z: z[1]), f"single:{default_col}"

    # Prefer the source that yields the most repeated longitudinal subjects.
    # Tie-breaker: prefer explicit subject+visit pairs over single scan key.
    candidates = sorted(candidates, key=lambda z: (z[0], z[3].startswith("pair:")), reverse=True)
    _, subj, visit, source = candidates[0]
    return subj, visit, source


def nice_label(x: str) -> str:
    mapping = {
        "ADNI": "ADNI",
        "HABS": "HABS",
        "cBAG": "cBAG",
        "delta_cbag": "ΔcBAG",
        "Global cognition": "Cognition",
        "Hippocampal volume": "Hippocampal volume",
        "Hippocampal FA": "Hippocampal FA",
        "Graph clustering": "Graph clustering",
        "Graph path length": "Graph path length",
        "Global efficiency": "Global efficiency",
        "Local efficiency": "Local efficiency",
        "Whole-brain volume": "Whole-brain volume",
        "Global FA": "Global FA",
        "Amyloid / Aβ": "Amyloid / Aβ",
        "pTau": "pTau",
    }
    if x in mapping:
        return mapping[x]
    out = str(x).replace("_raw_clean", "").replace("_", " ")
    return re.sub(r"\s+", " ", out).strip()


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


def fisher_ci_for_r(r: float, n: int) -> Tuple[float, float]:
    if pd.isna(r) or n <= 3:
        return np.nan, np.nan
    rr = float(np.clip(r, -0.999999, 0.999999))
    z = np.arctanh(rr)
    se = 1.0 / math.sqrt(n - 3)
    return float(np.tanh(z - 1.96 * se)), float(np.tanh(z + 1.96 * se))


def corr_xy(x: pd.Series, y: pd.Series, method: str = "pearson") -> Dict[str, float]:
    xx = clean_numeric(x)
    yy = clean_numeric(y)
    ok = xx.notna() & yy.notna()
    n = int(ok.sum())
    if n < 3 or xx[ok].nunique() < 2 or yy[ok].nunique() < 2:
        return {"n": n, "r": np.nan, "p": np.nan, "ci_low": np.nan, "ci_high": np.nan}
    if method == "spearman":
        r, p = spearmanr(xx[ok], yy[ok])
    else:
        r, p = pearsonr(xx[ok], yy[ok])
    lo, hi = fisher_ci_for_r(float(r), n)
    return {"n": n, "r": float(r), "p": float(p), "ci_low": lo, "ci_high": hi}


# =============================================================================
# Input + merge
# =============================================================================

def cohort_validation_dir(results_root: Path, cohort: str, feature_set: str) -> Path:
    return results_root / RESULTS_DIR_MAP[cohort] / f"ablation_{feature_set}" / "validation_figures_full_cohort"


def choose_input_path(results_root: Path, cohort: str, feature_set: str, input_kind: str) -> Tuple[Optional[Path], str]:
    vdir = cohort_validation_dir(results_root, cohort, feature_set)
    enriched = vdir / "subject_level_validation_input_enriched_for_Figure4.csv"
    validation = vdir / "subject_level_validation_input.csv"
    if input_kind == "enriched":
        return (enriched if enriched.exists() else None), "enriched"
    if input_kind == "validation":
        return (validation if validation.exists() else None), "validation"
    if enriched.exists():
        return enriched, "enriched"
    if validation.exists():
        return validation, "validation"
    return None, "missing"


def metadata_path(metadata_dir: Path, cohort: str) -> Path:
    return metadata_dir / f"{cohort}_harmonized_metadata.csv"


def deduplicate_by_scan(df: pd.DataFrame, key_col: str) -> pd.DataFrame:
    if df.empty:
        return df
    tmp = df.copy()
    tmp["_dedup_key"] = tmp[key_col].map(lambda x: normalize_connectome_key(x))
    tmp["_orig_order_for_dedup"] = np.arange(len(tmp))
    tmp = (
        tmp.sort_values(["_dedup_key", "_orig_order_for_dedup"])
        .drop_duplicates("_dedup_key", keep="first")
        .drop(columns=["_dedup_key", "_orig_order_for_dedup"], errors="ignore")
    )
    return tmp


def plausible_key_cols(cols: Sequence[str]) -> List[str]:
    toks = ["connectome", "dwi", "subject", "participant", "runno", "graph", "mri", "ptid", "rid", "id", "match"]
    return [c for c in cols if any(t in str(c).lower() for t in toks)]


def choose_best_key_pair(val: pd.DataFrame, meta: pd.DataFrame, cohort: str, val_id_col: str) -> Tuple[str, str, int]:
    val_cols = [val_id_col] + [c for c in VAL_KEY_CANDIDATES if c in val.columns and c != val_id_col]
    meta_cols = [c for c in META_KEY_CANDIDATES if c in meta.columns]
    val_cols += [c for c in plausible_key_cols(val.columns) if c not in val_cols]
    meta_cols += [c for c in plausible_key_cols(meta.columns) if c not in meta_cols]

    best = ("", "", 0)
    for vc in val_cols:
        vkeys = set(val[vc].map(lambda x: normalize_connectome_key(x, cohort)).dropna())
        if not vkeys:
            continue
        for mc in meta_cols:
            mkeys = set(meta[mc].map(lambda x: normalize_connectome_key(x, cohort)).dropna())
            if not mkeys:
                continue
            overlap = len(vkeys & mkeys)
            if overlap > best[2]:
                best = (vc, mc, overlap)
    return best


def prefix_metadata(meta: pd.DataFrame) -> pd.DataFrame:
    rename = {c: f"meta__{c}" for c in meta.columns if not str(c).startswith("meta__")}
    return meta.rename(columns=rename)


def load_and_merge_cohort(
    results_root: Path,
    metadata_dir: Path,
    cohort: str,
    feature_set: str,
    input_kind: str,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    path, source_kind = choose_input_path(results_root, cohort, feature_set, input_kind)
    qa: Dict[str, object] = {
        "cohort": cohort,
        "feature_set": feature_set,
        "validation_path": str(path) if path else "",
        "input_kind_used": source_kind,
        "metadata_path": str(metadata_path(metadata_dir, cohort)),
        "rows_before_dedup": 0,
        "rows_after_dedup": 0,
        "id_col": "",
        "cbag_col": "",
        "validation_key_col": "",
        "metadata_key_col": "",
        "key_overlap_unique": 0,
        "matched_metadata_rows": 0,
        "status": "missing_validation",
    }

    if path is None or not path.exists():
        return pd.DataFrame(), qa

    val = pd.read_csv(path, low_memory=False)
    qa["rows_before_dedup"] = len(val)

    id_col = first_existing(val, VAL_KEY_CANDIDATES)
    cbag_col = first_existing(val, CBAG_PRIORITY)
    qa["id_col"] = id_col or ""
    qa["cbag_col"] = cbag_col or ""

    if id_col is None:
        qa["status"] = "missing_id_col"
        return pd.DataFrame(), qa
    if cbag_col is None:
        qa["status"] = "missing_cbag_col"
        return pd.DataFrame(), qa

    val = deduplicate_by_scan(val, id_col)
    qa["rows_after_dedup"] = len(val)

    mpath = metadata_path(metadata_dir, cohort)
    if not mpath.exists():
        qa["status"] = "missing_metadata_file"
        # Still return validation-only rows.
        merged = val.copy()
        best_vc, best_mc, overlap = id_col, "", 0
    else:
        meta = pd.read_csv(mpath, low_memory=False)
        best_vc, best_mc, overlap = choose_best_key_pair(val, meta, cohort, id_col)
        qa["validation_key_col"] = best_vc
        qa["metadata_key_col"] = best_mc
        qa["key_overlap_unique"] = overlap

        if not best_vc or not best_mc or overlap == 0:
            qa["status"] = "no_metadata_key_overlap"
            merged = val.copy()
        else:
            val = val.copy()
            meta = meta.copy()
            val["_merge_key"] = val[best_vc].map(lambda x: normalize_connectome_key(x, cohort))
            meta["_merge_key"] = meta[best_mc].map(lambda x: normalize_connectome_key(x, cohort))

            # Prefer metadata rows with diagnosis/APOE/cognition if duplicates exist.
            meta["_has_dx"] = meta.get("DX_Label_harmonized", pd.Series(np.nan, index=meta.index)).notna()
            meta["_has_cog"] = meta.get("Global_Cognition_Composite", pd.Series(np.nan, index=meta.index)).notna()
            meta["_has_apoe"] = meta.get("APOE_genotype_harmonized", pd.Series(np.nan, index=meta.index)).notna()
            meta["_orig_order"] = np.arange(len(meta))

            meta_dedup = (
                meta.dropna(subset=["_merge_key"])
                .sort_values(["_merge_key", "_has_dx", "_has_cog", "_has_apoe", "_orig_order"],
                             ascending=[True, False, False, False, True])
                .drop_duplicates("_merge_key", keep="first")
                .drop(columns=["_has_dx", "_has_cog", "_has_apoe", "_orig_order"], errors="ignore")
            )
            meta_pref = prefix_metadata(meta_dedup).rename(columns={"meta___merge_key": "_merge_key"})
            merged = val.merge(meta_pref, on="_merge_key", how="left", validate="m:1")
            qa["matched_metadata_rows"] = int(merged.filter(like="meta__").notna().any(axis=1).sum())
            qa["status"] = "ok"

    # Longitudinal ID / visit should come from the chosen validation key if possible,
    # but for HABS we also test explicit subject/visit columns such as
    # DWI_subject + DWI_visit or DWI_visit_year.
    parse_source = best_vc if best_vc and best_vc in merged.columns else id_col
    longitudinal_subject, visit, longitudinal_parse_source = choose_longitudinal_subject_visit(
        merged, cohort=cohort, default_col=parse_source
    )
    merged["cohort"] = cohort
    merged["feature_set"] = feature_set
    merged["_source_path"] = str(path)
    merged["_input_kind_used"] = source_kind
    merged["_id_col"] = id_col
    merged["_cbag_col"] = cbag_col
    merged["_parse_source_col"] = longitudinal_parse_source
    merged["_longitudinal_subject"] = longitudinal_subject
    merged["_visit"] = visit
    merged["_cbag"] = clean_numeric(merged[cbag_col])
    qa["longitudinal_parse_source"] = longitudinal_parse_source

    visit_ok = merged["_visit"].notna()
    cbag_ok = merged["_cbag"].notna()
    visits = merged.loc[visit_ok, :].groupby("_longitudinal_subject")["_visit"].nunique()
    repeated = visits[visits >= 2].index
    qa["unique_subjects_with_visit"] = int(merged.loc[visit_ok, "_longitudinal_subject"].nunique())
    qa["subjects_with_ge2_visits"] = int(len(repeated))
    qa["longitudinal_rows"] = int(merged["_longitudinal_subject"].isin(repeated).sum())
    qa["n_cbag_nonnull"] = int(cbag_ok.sum())

    return merged.loc[visit_ok & cbag_ok].copy(), qa


def load_longitudinal_tables(
    results_root: Path,
    metadata_dir: Path,
    cohorts: Sequence[str],
    feature_set: str,
    input_kind: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    frames = []
    qa_rows = []
    for cohort in cohorts:
        df, qa = load_and_merge_cohort(results_root, metadata_dir, cohort, feature_set, input_kind)
        qa_rows.append(qa)
        if not df.empty:
            frames.append(df)
    out = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    return out, pd.DataFrame(qa_rows)


# =============================================================================
# Endpoint selection, deltas, stats
# =============================================================================

def candidate_tokens(candidates: Sequence[str]) -> List[str]:
    tokens: List[str] = []
    for c in candidates:
        tokens.extend([t for t in re.split(r"[_\s/\-]+", str(c).lower()) if len(t) >= 3])
    return sorted(set(tokens), key=len, reverse=True)


def ordered_candidate_names(candidates: Sequence[str], source_priority: str) -> List[str]:
    out: List[str] = []
    for c in candidates:
        base = [c, f"{c}_raw_clean"] if not str(c).endswith("_raw_clean") else [c]
        for b in base:
            if source_priority == "metadata":
                out.extend([f"meta__{b}", b])
            elif source_priority == "validation":
                out.extend([b, f"meta__{b}"])
            else:
                out.extend([f"meta__{b}", b])
    # preserve order, remove duplicates
    seen = set()
    unique = []
    for c in out:
        if c not in seen:
            seen.add(c)
            unique.append(c)
    return unique


def choose_endpoint_column(
    df: pd.DataFrame,
    endpoint: str,
    spec: Dict[str, object],
    min_n_visits: int,
    source_priority: str,
) -> Tuple[Optional[str], str, int]:
    candidates = list(spec.get("candidates", []))  # type: ignore[arg-type]
    prefer = [str(x).lower() for x in spec.get("prefer_keywords", [])]  # type: ignore[union-attr]
    avoid = [str(x).lower() for x in spec.get("avoid_keywords", [])]  # type: ignore[union-attr]
    value_range = spec.get("value_range", None)

    def usable(col: str) -> int:
        if col not in df.columns:
            return 0
        low = col.lower()
        if any(bad in low for bad in avoid):
            return 0
        x = clean_numeric(df[col])
        if value_range is not None:
            lo, hi = value_range  # type: ignore[misc]
            x = x.where((x >= lo) & (x <= hi))
        return int(x.notna().sum())

    for col in ordered_candidate_names(candidates, source_priority):
        n = usable(col)
        if n >= min_n_visits:
            return col, "exact", n

    tokens = candidate_tokens(candidates)
    hits: List[Tuple[int, int, str]] = []
    for col in df.columns:
        low = col.lower()
        if source_priority == "metadata" and not low.startswith("meta__"):
            # If metadata is requested, allow fallback to validation only after metadata-like hits fail.
            pass
        if any(bad in low for bad in avoid):
            continue
        n = usable(col)
        if n < min_n_visits:
            continue
        token_hit = any(tok in low for tok in tokens)
        prefer_score = sum(tok in low for tok in prefer)
        metadata_bonus = 2 if low.startswith("meta__") and source_priority == "metadata" else 0
        validation_bonus = 2 if (not low.startswith("meta__")) and source_priority == "validation" else 0
        if token_hit or prefer_score > 0:
            hits.append((2 * prefer_score + int(token_hit) + metadata_bonus + validation_bonus, n, col))

    if hits:
        hits.sort(key=lambda x: (-x[0], -x[1], x[2].lower()))
        return hits[0][2], "fallback_keyword", hits[0][1]

    best_col, best_n = None, 0
    for col in ordered_candidate_names(candidates, source_priority):
        n = usable(col)
        if n > best_n:
            best_col, best_n = col, n
    if best_col:
        return best_col, "insufficient_n", best_n
    return None, "missing", 0


def build_delta_table(long_df: pd.DataFrame, endpoint_cols: Dict[str, Optional[str]]) -> pd.DataFrame:
    rows = []
    num_cols = {"_cbag": clean_numeric(long_df["_cbag"])}
    for col in [c for c in endpoint_cols.values() if c is not None]:
        if col in long_df.columns and col not in num_cols:
            num_cols[col] = clean_numeric(long_df[col])

    num_df = pd.DataFrame(num_cols, index=long_df.index)
    work = pd.concat([long_df, num_df.add_prefix("_num__")], axis=1)

    for (cohort, subject), g in work.groupby(["cohort", "_longitudinal_subject"], sort=False):
        g = g.dropna(subset=["_visit"]).copy()
        if len(g) < 2:
            continue
        g = g.sort_values("_visit")
        first = g.iloc[0]
        last = g.iloc[-1]

        row = {
            "cohort": cohort,
            "longitudinal_subject": subject,
            "baseline_visit": first["_visit"],
            "followup_visit": last["_visit"],
            "delta_visit": last["_visit"] - first["_visit"],
            "n_visits": int(len(g)),
            "baseline_cbag": first["_num___cbag"],
            "followup_cbag": last["_num___cbag"],
            "delta_cbag": last["_num___cbag"] - first["_num___cbag"],
            "cbag_col": first["_cbag_col"],
            "input_kind_used": first["_input_kind_used"],
        }

        for endpoint, col in endpoint_cols.items():
            if col is None or col not in long_df.columns:
                row[f"{endpoint}__column"] = col
                row[f"baseline__{endpoint}"] = np.nan
                row[f"followup__{endpoint}"] = np.nan
                row[f"delta__{endpoint}"] = np.nan
                continue
            num_col = f"_num__{col}"
            row[f"{endpoint}__column"] = col
            row[f"baseline__{endpoint}"] = first.get(num_col, np.nan)
            row[f"followup__{endpoint}"] = last.get(num_col, np.nan)
            row[f"delta__{endpoint}"] = last.get(num_col, np.nan) - first.get(num_col, np.nan)

        rows.append(row)
    return pd.DataFrame(rows)


def compute_delta_associations(delta_df: pd.DataFrame, endpoint_cols: Dict[str, Optional[str]], min_n: int, method: str) -> pd.DataFrame:
    rows = []
    cohorts = sorted(delta_df["cohort"].dropna().unique()) if not delta_df.empty else []
    for endpoint, col in endpoint_cols.items():
        delta_col = f"delta__{endpoint}"
        domain = str(ENDPOINTS[endpoint].get("domain", ""))
        for cohort in cohorts:
            sub = delta_df[delta_df["cohort"] == cohort].copy()
            if delta_col not in sub.columns:
                stats = {"n": 0, "r": np.nan, "p": np.nan, "ci_low": np.nan, "ci_high": np.nan}
                status = "missing_delta_column"
            else:
                stats = corr_xy(sub["delta_cbag"], sub[delta_col], method=method)
                status = "ok" if stats["n"] >= min_n and np.isfinite(stats["r"]) else "insufficient_n_or_zero_variance"
            rows.append({
                "endpoint": endpoint,
                "domain": domain,
                "cohort": cohort,
                "column": col,
                "delta_column": delta_col,
                **stats,
                "status": status,
                "abs_r": abs(stats["r"]) if np.isfinite(stats["r"]) else np.nan,
            })

        if len(cohorts) > 1 and delta_col in delta_df.columns:
            stats = corr_xy(delta_df["delta_cbag"], delta_df[delta_col], method=method)
            status = "ok" if stats["n"] >= min_n and np.isfinite(stats["r"]) else "insufficient_n_or_zero_variance"
            rows.append({
                "endpoint": endpoint,
                "domain": domain,
                "cohort": "Pooled",
                "column": col,
                "delta_column": delta_col,
                **stats,
                "status": status,
                "abs_r": abs(stats["r"]) if np.isfinite(stats["r"]) else np.nan,
            })
    return pd.DataFrame(rows)


def endpoint_availability(delta_df: pd.DataFrame, endpoint_cols: Dict[str, Optional[str]]) -> pd.DataFrame:
    rows = []
    for endpoint, col in endpoint_cols.items():
        delta_col = f"delta__{endpoint}"
        domain = str(ENDPOINTS[endpoint].get("domain", ""))
        for cohort in sorted(delta_df["cohort"].dropna().unique()):
            sub = delta_df[delta_df["cohort"] == cohort]
            x = clean_numeric(sub[delta_col]) if delta_col in sub.columns else pd.Series(dtype=float)
            rows.append({
                "endpoint": endpoint,
                "domain": domain,
                "cohort": cohort,
                "selected_column": col or "",
                "n_delta": int(x.notna().sum()),
                "sd_delta": float(x.std(ddof=1)) if x.notna().sum() > 1 else np.nan,
                "n_unique_delta": int(x.dropna().nunique()),
                "usable": bool(x.notna().sum() > 0 and x.dropna().nunique() > 1),
            })
    return pd.DataFrame(rows)


# =============================================================================
# Plotting
# =============================================================================

def p_text(p: float) -> str:
    if not np.isfinite(p):
        return "p=NA"
    if p < 1e-4:
        return "p<1e-4"
    if p < 0.001:
        return "p<0.001"
    return f"p={p:.3g}"


def plot_delta_cbag_distribution(ax: plt.Axes, delta_df: pd.DataFrame):
    cohorts = sorted(delta_df["cohort"].dropna().unique())
    data = [delta_df.loc[delta_df["cohort"] == c, "delta_cbag"].dropna().to_numpy() for c in cohorts]
    ax.axhline(0, color="0.6", linestyle="--", linewidth=0.8)
    ax.boxplot(data, labels=cohorts, showfliers=False)
    for i, vals in enumerate(data, start=1):
        rng = np.random.default_rng(100 + i)
        x = rng.normal(i, 0.045, size=len(vals))
        ax.scatter(x, vals, s=12, alpha=0.55)
    ax.set_ylabel("ΔcBAG")
    ax.set_title("Within-person ΔcBAG")
    ax.grid(axis="y", alpha=0.25)


def scatter_delta(ax: plt.Axes, delta_df: pd.DataFrame, stats_df: pd.DataFrame, endpoint: str, cohort: str = "Pooled"):
    delta_col = f"delta__{endpoint}"
    if delta_col not in delta_df.columns:
        ax.text(0.5, 0.5, "Endpoint unavailable", ha="center", va="center")
        ax.axis("off")
        return

    sub = delta_df.copy() if cohort == "Pooled" else delta_df[delta_df["cohort"] == cohort].copy()
    tmp = sub[["delta_cbag", delta_col, "cohort"]].dropna()
    if tmp.empty:
        ax.text(0.5, 0.5, "No paired deltas", ha="center", va="center")
        ax.axis("off")
        return

    for c in sorted(tmp["cohort"].unique()):
        t = tmp[tmp["cohort"] == c]
        ax.scatter(t["delta_cbag"], t[delta_col], s=30, alpha=0.72, label=c, edgecolors="white", linewidths=0.35)

    row = stats_df[(stats_df["endpoint"] == endpoint) & (stats_df["cohort"] == cohort)]
    if not row.empty and row.iloc[0]["status"] == "ok":
        r = float(row.iloc[0]["r"])
        p = float(row.iloc[0]["p"])
        n = int(row.iloc[0]["n"])
        x = tmp["delta_cbag"].to_numpy(float)
        y = tmp[delta_col].to_numpy(float)
        if len(x) >= 2 and np.nanstd(x) > 0:
            coef = np.polyfit(x, y, 1)
            xs = np.linspace(np.nanmin(x), np.nanmax(x), 100)
            ax.plot(xs, coef[0] * xs + coef[1], color="black", linewidth=1.8)
        annot = f"n={n}\nr={r:.2f}\n{p_text(p)}"
        ax.text(0.03, 0.97, annot, transform=ax.transAxes, ha="left", va="top", fontsize=8.5,
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="0.65", alpha=0.85))
    else:
        annot = f"n={len(tmp)}"

    ax.axvline(0, color="0.75", linestyle="--", linewidth=0.7)
    ax.axhline(0, color="0.75", linestyle="--", linewidth=0.7)
    ax.set_xlabel("ΔcBAG")
    ax.set_ylabel(f"Δ{nice_label(endpoint)}")
    ax.set_title(f"ΔcBAG vs Δ{nice_label(endpoint)}", fontsize=10)
    ax.grid(alpha=0.25)
    if tmp["cohort"].nunique() > 1:
        ax.legend(frameon=False, fontsize=7, loc="best")


def plot_forest(ax: plt.Axes, stats_df: pd.DataFrame, max_n: int = 12):
    df = stats_df[(stats_df["cohort"] == "Pooled") & (stats_df["status"] == "ok")].copy()
    if df.empty:
        df = stats_df[stats_df["status"] == "ok"].copy()
    if df.empty:
        ax.text(0.5, 0.5, "No usable associations", ha="center", va="center")
        ax.axis("off")
        return
    df = df.sort_values("abs_r", ascending=False).head(max_n).iloc[::-1].copy()
    y = np.arange(len(df))
    ax.axvline(0, color="0.4", linewidth=0.8)
    ax.errorbar(
        df["r"], y,
        xerr=[df["r"] - df["ci_low"], df["ci_high"] - df["r"]],
        fmt="o",
        capsize=2,
    )
    labels = [f"{nice_label(e)} ({c})" if c != "Pooled" else nice_label(e) for e, c in zip(df["endpoint"], df["cohort"])]
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Correlation r")
    ax.set_title("Strongest Δ associations")
    ax.grid(axis="x", alpha=0.25)


def plot_availability(ax: plt.Axes, availability: pd.DataFrame):
    if availability.empty:
        ax.text(0.5, 0.5, "No availability data", ha="center", va="center")
        ax.axis("off")
        return
    piv = availability.pivot_table(index="endpoint", columns="cohort", values="n_delta", aggfunc="max").fillna(0)
    piv = piv.loc[piv.max(axis=1).sort_values(ascending=True).index]
    im = ax.imshow(piv.to_numpy(), aspect="auto", cmap="viridis")
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels([nice_label(x) for x in piv.index], fontsize=8)
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels(piv.columns, fontsize=9)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            ax.text(j, i, str(int(piv.iloc[i, j])), ha="center", va="center", color="white", fontsize=7)
    ax.set_title("Complete longitudinal delta pairs")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def plot_r_heatmap(ax: plt.Axes, stats_df: pd.DataFrame):
    df = stats_df[stats_df["status"] == "ok"].copy()
    if df.empty:
        ax.text(0.5, 0.5, "No usable associations", ha="center", va="center")
        ax.axis("off")
        return
    piv = df.pivot_table(index="endpoint", columns="cohort", values="r", aggfunc="first")
    piv = piv.loc[piv.abs().max(axis=1).sort_values(ascending=True).index]
    vmax = max(0.05, np.nanmax(np.abs(piv.to_numpy())))
    im = ax.imshow(piv.to_numpy(), aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels([nice_label(x) for x in piv.index], fontsize=8)
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels(piv.columns, fontsize=9)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.iloc[i, j]
            txt = "" if pd.isna(v) else f"{v:.2f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=7)
    ax.set_title("Correlation r for ΔcBAG vs Δendpoint")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def make_figure8(delta_df: pd.DataFrame, stats_df: pd.DataFrame, outdir: Path, formats: Sequence[str], dpi: int, title: str) -> List[str]:
    main_endpoints = ["Global cognition", "Hippocampal volume", "Hippocampal FA", "Whole-brain volume", "Global efficiency"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9.5))
    axes = axes.ravel()

    plot_delta_cbag_distribution(axes[0], delta_df)
    panel_label(axes[0], "A")

    for ax, label, endpoint in zip(axes[1:5], list("BCDE"), main_endpoints[:4]):
        scatter_delta(ax, delta_df, stats_df, endpoint, cohort="Pooled")
        panel_label(ax, label)

    plot_forest(axes[5], stats_df, max_n=10)
    panel_label(axes[5], "F")

    fig.suptitle(title, fontsize=15, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return save_all(fig, outdir / "Figure8_longitudinal_delta_cBAG_HARMONIZED_FINAL", formats, dpi)


def make_supp_figure8(delta_df: pd.DataFrame, stats_df: pd.DataFrame, availability: pd.DataFrame, outdir: Path, formats: Sequence[str], dpi: int, title: str) -> List[str]:
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    axes = axes.ravel()
    plot_availability(axes[0], availability)
    panel_label(axes[0], "A")
    plot_r_heatmap(axes[1], stats_df)
    panel_label(axes[1], "B")
    plot_forest(axes[2], stats_df, max_n=20)
    panel_label(axes[2], "C")

    # Delta endpoint distributions for top pooled endpoints
    ax = axes[3]
    top = stats_df[(stats_df["cohort"] == "Pooled") & (stats_df["status"] == "ok")].sort_values("abs_r", ascending=False).head(6)
    labels, data = [], []
    for endpoint in top["endpoint"]:
        col = f"delta__{endpoint}"
        if col in delta_df.columns:
            vals = delta_df[col].dropna().to_numpy()
            if len(vals):
                labels.append(nice_label(endpoint))
                data.append(vals)
    if data:
        ax.boxplot(data, labels=labels, showfliers=False)
        ax.tick_params(axis="x", rotation=45, labelsize=8)
        ax.axhline(0, color="0.6", linestyle="--", linewidth=0.8)
        ax.set_ylabel("Δendpoint")
        ax.set_title("Distribution of selected endpoint deltas")
        ax.grid(axis="y", alpha=0.25)
    else:
        ax.text(0.5, 0.5, "No endpoint delta distributions", ha="center", va="center")
        ax.axis("off")
    panel_label(ax, "D")

    fig.suptitle(title, fontsize=15, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return save_all(fig, outdir / "Supplementary_Figure8_longitudinal_delta_cBAG_HARMONIZED_FINAL", formats, dpi)


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()
    results_root = Path(args.results_root).expanduser().resolve()
    base_dir = Path(args.base_dir).expanduser().resolve()
    metadata_dir = Path(args.metadata_dir).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve() if args.outdir else base_dir / DEFAULT_OUTDIR_NAME
    outdir.mkdir(parents=True, exist_ok=True)

    cohorts = [x.strip() for x in args.cohorts.split(",") if x.strip()]
    formats = [x.strip() for x in args.formats.split(",") if x.strip()]

    print("\n" + "=" * 100)
    print("Figure 8 FINAL harmonized-metadata longitudinal delta-cBAG")
    print("=" * 100)
    print(f"RESULTS_ROOT: {results_root}")
    print(f"BASE_DIR: {base_dir}")
    print(f"METADATA_DIR: {metadata_dir}")
    print(f"OUTDIR: {outdir}")
    print(f"FEATURE_SET: {args.feature_set}")
    print(f"COHORTS: {', '.join(cohorts)}")
    print(f"INPUT_KIND: {args.input_kind}")
    print(f"ENDPOINT_SOURCE_PRIORITY: {args.endpoint_source_priority}")
    print(f"MIN_N: {args.min_n}")
    print(f"CORR_METHOD: {args.corr_method}")

    long_df, merge_qa = load_longitudinal_tables(
        results_root=results_root,
        metadata_dir=metadata_dir,
        cohorts=cohorts,
        feature_set=args.feature_set,
        input_kind=args.input_kind,
    )
    merge_qa.to_csv(outdir / "longitudinal_merge_QA.csv", index=False)

    if long_df.empty:
        raise SystemExit("No longitudinal validation rows loaded. See longitudinal_merge_QA.csv.")

    endpoint_rows = []
    endpoint_cols: Dict[str, Optional[str]] = {}
    min_n_visits = max(args.min_n, 3)
    for endpoint, spec in ENDPOINTS.items():
        col, method, n = choose_endpoint_column(
            long_df,
            endpoint,
            spec,
            min_n_visits=min_n_visits,
            source_priority=args.endpoint_source_priority,
        )
        endpoint_cols[endpoint] = col
        endpoint_rows.append({
            "endpoint": endpoint,
            "domain": spec.get("domain", ""),
            "selected_column": col or "",
            "selection_method": method,
            "n_nonmissing_visits": n,
            "source": "metadata" if col and col.startswith("meta__") else "validation" if col else "",
        })
    endpoint_selection = pd.DataFrame(endpoint_rows)
    endpoint_selection.to_csv(outdir / "longitudinal_endpoint_column_selection.csv", index=False)

    delta_df = build_delta_table(long_df, endpoint_cols)
    if delta_df.empty:
        raise SystemExit("No within-subject delta rows could be built.")

    stats_df = compute_delta_associations(delta_df, endpoint_cols, min_n=args.min_n, method=args.corr_method)
    availability = endpoint_availability(delta_df, endpoint_cols)

    dataset_summary = (
        delta_df.groupby("cohort", as_index=False)
        .agg(
            n_subjects=("longitudinal_subject", "nunique"),
            n_delta_rows=("longitudinal_subject", "count"),
            median_delta_visit=("delta_visit", "median"),
            mean_delta_cbag=("delta_cbag", "mean"),
            sd_delta_cbag=("delta_cbag", "std"),
        )
    )

    delta_df.to_csv(outdir / "longitudinal_delta_subject_table.csv", index=False)
    stats_df.to_csv(outdir / "longitudinal_delta_association_stats.csv", index=False)
    availability.to_csv(outdir / "longitudinal_endpoint_availability.csv", index=False)
    dataset_summary.to_csv(outdir / "longitudinal_delta_dataset_summary.csv", index=False)

    outputs = []
    outputs.extend(make_figure8(delta_df, stats_df, outdir, formats, args.dpi, args.main_title))
    outputs.extend(make_supp_figure8(delta_df, stats_df, availability, outdir, formats, args.dpi, args.supp_title))

    manifest = pd.DataFrame([{
        "results_root": str(results_root),
        "base_dir": str(base_dir),
        "metadata_dir": str(metadata_dir),
        "outdir": str(outdir),
        "feature_set": args.feature_set,
        "cohorts": ",".join(cohorts),
        "input_kind": args.input_kind,
        "endpoint_source_priority": args.endpoint_source_priority,
        "n_delta_subjects": int(len(delta_df)),
        "outputs": ";".join(outputs),
    }])
    manifest_path = outdir / "figure8_manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    print("\nMerge QA:")
    print(merge_qa.to_string(index=False))
    print("\nDataset summary:")
    print(dataset_summary.to_string(index=False))
    print("\nEndpoint selection:")
    print(endpoint_selection.to_string(index=False))
    print("\nTop associations:")
    ok = stats_df[stats_df["status"] == "ok"].sort_values("abs_r", ascending=False)
    print(ok.head(20).to_string(index=False))
    print("\nSaved:")
    for p in outputs:
        print(f"  {p}")
    print(f"  {manifest_path}")


if __name__ == "__main__":
    main()
