#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Final Table 1 builder + enriched metadata status writer.

This script does two things:

1) Adds harmonized cognitive-status columns to every model/cohort
   metadata_all_with_predictions table:
      NORMCOG_01
      DEMENTIA_01
      DX_Label_harmonized
      COG_STATUS_SOURCE

   and saves one enriched CSV per feature set/cohort.

2) Builds corrected Table 1 from the primary feature set.

ADNI diagnosis source
---------------------
Primary ADNI diagnosis source is DXSUM:

  $WORK/ines/data/harmonization/ADNI/metadata/ADNI_Metadata/MetaData/
      All_Subjects_DXSUM_26Mar2026.csv

Read:
  RID
  VISCODE
  VISCODE2
  DIAGNOSIS

Map:
  DIAGNOSIS == 1 -> CN / NORMCOG_01 = 1 / DEMENTIA_01 = 0
  DIAGNOSIS == 2 -> MCI / NORMCOG_01 = 0 / DEMENTIA_01 = 0
  DIAGNOSIS == 3 -> AD / NORMCOG_01 = 0 / DEMENTIA_01 = 1

Fallback ADNI diagnosis:
  GROUP from ADNI_covars_no_underscores.csv if DXSUM is unavailable or missing.

Other cohorts:
  ADRC: NORMCOG / IMPNOMCI / DEMENTED
  HABS: HABS_metadata_session_level_DWI_filled_cognitive_composites.csv plus CDX_Cog
  AD_DECODE: Risk, with Risk 0 and Risk 1 combined as NORMCOG/control

Outputs
-------
$WORK/ines/results/Table1_Corrected/

Enriched metadata:
  enriched_metadata_with_status/
    metadata_all_with_status_<feature_set>_<cohort>.csv

Table 1:
  Table1_corrected_formatted.xlsx
  Table1_corrected_wide.csv
  Table1_corrected_long.csv
  Table1_subject_level_corrected.csv
  Table1_scan_level_corrected.csv
  Table1_diagnostic_sources.csv
  Table1_feature_set_consistency_check.csv
  Table1_HABS_DWI_availability_summary.csv
  Table1_HABS_DWI_visit_patterns.csv
  Table1_corrected_README.md

Run
---
python final_build_table1_and_enriched_status_metadata.py
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd


# =============================================================================
# SETTINGS
# =============================================================================

WORK = Path(os.environ.get("WORK", "/mnt/newStor/paros/paros_WORK"))
BASE_DIR = WORK / "ines"
RESULTS_ROOT = BASE_DIR / "results"

OUTDIR = RESULTS_ROOT / "Table1_Corrected"
ENRICHED_OUTDIR = OUTDIR / "enriched_metadata_with_status"

PRIMARY_FEATURE_SET = "imaging_only"

FEATURE_SETS = [
    "imaging_only",
    "imaging_demographics",
    "imaging_biomarkers",
    "full",
    "full_no_cardiovascular",
]

COHORTS = ["ADNI", "ADRC", "HABS", "AD_DECODE"]

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

STUDY_DESIGN = {
    "ADNI": "Longitudinal",
    "ADRC": "Cross-sectional",
    "HABS": "Longitudinal",
    "AD_DECODE": "Cross-sectional",
}

COHORT_DESCRIPTION = {
    "ADNI": "Longitudinal aging/AD cohort",
    "ADRC": "Cross-sectional aging/AD cohort",
    "HABS": "Longitudinal aging cohort",
    "AD_DECODE": "Cross-sectional AD-risk/transcriptomics cohort",
}

FEATURE_DESCRIPTION = {
    "ADNI": "DTI connectome; FA; volume; graph metrics; APOE; vascular measures",
    "ADRC": "DTI connectome; FA; volume; graph metrics; APOE; vascular measures",
    "HABS": "DTI connectome; FA; volume; graph metrics; APOE; vascular measures",
    "AD_DECODE": "DTI connectome; FA; volume; graph metrics; APOE; vascular measures",
}

ADNI_DXSUM_CANDIDATES = [
    BASE_DIR / "data/harmonization/ADNI/metadata/ADNI_Metadata/MetaData/All_Subjects_DXSUM_26Mar2026.csv",
    BASE_DIR / "data/harmonization/ADNI/metadata/ADNI_Metadata/MetaData/DXSUM_25Mar2026.csv",
    BASE_DIR / "data/harmonization/ADNI/metadata/ADNI_Metadata/MetaData/DXSUM_26Mar2026.csv",
]

ADNI_COVARS_CANDIDATES = [
    BASE_DIR / "data/harmonization/ADNI/metadata/ADNI_covars_no_underscores.csv",
    BASE_DIR / "data/harmonization/ADNI/metadata/ADNI_covars_no_underscores.tsv",
    BASE_DIR / "data/harmonization/ADNI/metadata/ADNI_covars.csv",
]

# Corrected metadata sources used to fill ADNI/HABS model metadata before Table 1.
# ADNI should use the metadata file with DWI keys such as R0072_y0.
# HABS should use the new DWI-session-level metadata, preferably the filled version.
ADNI_CORRECTED_METADATA_CANDIDATES = [
    BASE_DIR / "data/harmonization/ADNI/metadata/ADNI_metadata_with_DWI.csv",
    BASE_DIR / "data/harmonization/ADNI/metadata/ADNI_metadata.csv",
    BASE_DIR / "data/harmonization/ADNI/metadata/ADNI_metadata.xlsx",
]

HABS_CORRECTED_METADATA_CANDIDATES = [
    BASE_DIR / "data/harmonization/HABS/metadata/HABS_metadata_session_level_DWI_filled_cognitive_composites.csv",
    BASE_DIR / "data/harmonization/HABS/metadata/HABS_metadata_session_level_DWI_filled.csv",
    BASE_DIR / "data/harmonization/HABS/metadata/HABS_metadata_session_level_DWI.csv",
    BASE_DIR / "data/harmonization/HABS/metadata/HABS_metadata_with_DWI.csv",
    BASE_DIR / "data/harmonization/HABS/metadata/HABS_metadata.xlsx",
]

CBAG_CANDIDATES = [
    "cBAG_oof_global_raw_clean",
    "cBAG_oof_global",
    "cBAG_global_raw_clean",
    "cBAG_global",
    "cBAG_bias_corrected",
    "cBAG_BiasCorrected",
    "cBAG",
]

BAG_CANDIDATES = [
    "BAG_raw",
    "BAG",
    "brain_age_gap",
    "brain_age_gap_raw",
    "BrainAgeGap",
    "age_gap",
    "age_gap_raw",
]

PREDICTION_COLUMNS_TO_COPY = [
    "age_true",
    "pred_raw",
    "pred_bias_corrected",
    "pred_bias_corrected_global",
    "BAG_raw",
    "expected_BAG_global_from_oof",
    "cBAG",
    "cBAG_global",
]


# =============================================================================
# PATHS
# =============================================================================

def ablation_dir(cohort: str, feature_set: str) -> Path:
    return RESULTS_ROOT / RESULTS_DIR_MAP[cohort] / f"ablation_{feature_set}"


def metadata_all_path(cohort: str, feature_set: str) -> Path:
    prefix = PREFIX_MAP[cohort]
    return ablation_dir(cohort, feature_set) / f"{prefix}_{feature_set}_metadata_all_with_predictions.csv"


def full_predictions_path(cohort: str, feature_set: str) -> Path:
    prefix = PREFIX_MAP[cohort]
    return ablation_dir(cohort, feature_set) / f"{prefix}_{feature_set}_full_cohort_predictions.csv"


def enriched_status_path(cohort: str, feature_set: str) -> Path:
    return ENRICHED_OUTDIR / f"metadata_all_with_status_{feature_set}_{cohort}.csv"


# =============================================================================
# GENERAL HELPERS
# =============================================================================
def make_feature_set_consistency(diag_all: pd.DataFrame) -> pd.DataFrame:
    """
    Save the per-cohort/per-feature-set enrichment diagnostics.
    """
    if diag_all is None or diag_all.empty:
        return pd.DataFrame()

    sort_cols = [c for c in ["cohort", "feature_set"] if c in diag_all.columns]
    if sort_cols:
        return diag_all.sort_values(sort_cols).reset_index(drop=True)

    return diag_all.reset_index(drop=True)

def normalize_name(x: object) -> str:
    s = str(x).lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return re.sub(r"_+", "_", s).strip("_")


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


def as_key(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
        .str.strip()
        .replace({"nan": np.nan, "None": np.nan, "<NA>": np.nan, "": np.nan})
    )


def get_cbag_col(df: pd.DataFrame) -> str:
    col = first_existing(df, CBAG_CANDIDATES)
    if col is not None:
        return col
    col = first_existing(df, BAG_CANDIDATES)
    if col is not None:
        return col
    return ""


def fmt_count_pct(n: int, denom: int) -> str:
    if denom <= 0:
        return "—"
    return f"{int(n)} ({100*n/denom:.1f}%)"


def fmt_mean_sd(x: pd.Series) -> str:
    y = pd.to_numeric(x, errors="coerce").dropna()
    if len(y) == 0:
        return "—"
    return f"{y.mean():.2f} ± {y.std(ddof=1):.2f}"


def fmt_range(x: pd.Series) -> str:
    y = pd.to_numeric(x, errors="coerce").dropna()
    if len(y) == 0:
        return "—"
    return f"{y.min():.1f}–{y.max():.1f}"


def dash_if_zero_count(n: int, denom: int, use_dash: bool = True) -> str:
    if n == 0 and use_dash:
        return "—"
    return fmt_count_pct(n, denom)


# =============================================================================
# PREDICTION MERGE
# =============================================================================

def merge_predictions_if_needed(meta_all: pd.DataFrame, cohort: str, feature_set: str) -> tuple[pd.DataFrame, str]:
    cbag_col = get_cbag_col(meta_all)
    if cbag_col:
        return meta_all.copy(), f"already_in_metadata_all:{cbag_col}"

    full_path = full_predictions_path(cohort, feature_set)
    if not full_path.exists():
        return meta_all.copy(), "missing_full_prediction_file"

    full = pd.read_csv(full_path, low_memory=False)
    graph_col = first_existing(full, ["graph_id"])
    key_col = first_existing(meta_all, ["connectome_key", "connectome_full_key", "graph_id", "subject_id"])
    pred_cols = [c for c in PREDICTION_COLUMNS_TO_COPY if c in full.columns]

    if not pred_cols:
        return meta_all.copy(), "full_prediction_has_no_prediction_columns"

    if graph_col is not None and key_col is not None:
        left = meta_all.copy()
        left["_merge_key_prediction"] = as_key(left[key_col])
        right = full[[graph_col] + pred_cols].copy()
        right["_merge_key_prediction"] = as_key(right[graph_col])
        right = right.drop_duplicates("_merge_key_prediction", keep="first")

        merged = left.merge(
            right.drop(columns=[graph_col]),
            on="_merge_key_prediction",
            how="left",
            suffixes=("", "_from_full"),
        )

        for c in pred_cols:
            cf = f"{c}_from_full"
            if cf in merged.columns:
                if c not in merged.columns:
                    merged[c] = merged[cf]
                else:
                    merged[c] = merged[c].combine_first(merged[cf])
                merged = merged.drop(columns=[cf])

        n_mapped = int(merged[pred_cols].notna().any(axis=1).sum())
        if n_mapped > 0:
            return merged, f"merged_from_full_by_{key_col}_to_graph_id:n_mapped={n_mapped}"

    if len(meta_all) == len(full):
        merged = meta_all.copy()
        for c in pred_cols:
            merged[c] = full[c].values
        return merged, f"merged_from_full_by_row_order:n_rows={len(merged)}"

    return meta_all.copy(), "could_not_merge_full_predictions"


# =============================================================================
# ADNI DXSUM / GROUP
# =============================================================================

def load_adni_dxsum() -> tuple[Optional[pd.DataFrame], str]:
    for path in ADNI_DXSUM_CANDIDATES:
        if path.exists():
            return pd.read_csv(path, low_memory=False), str(path)
    return None, ""


def normalize_adni_visit(x: object) -> object:
    if pd.isna(x):
        return np.nan
    s = str(x).strip().lower()
    mapping = {
        "sc": "sc",
        "bl": "bl",
        "init": "bl",
        "m00": "bl",
        "m0": "bl",
        "y0": "bl",
        "m06": "m06",
        "m6": "m06",
        "y0.5": "m06",
        "m12": "m12",
        "y1": "m12",
        "m24": "m24",
        "y2": "m24",
        "m36": "m36",
        "y3": "m36",
        "m48": "m48",
        "y4": "m48",
        "m60": "m60",
        "y5": "m60",
    }
    return mapping.get(s, s)


def adni_diag_label_from_code(x: object) -> object:
    if pd.isna(x):
        return np.nan
    try:
        v = int(float(x))
    except Exception:
        return np.nan
    return {1: "CN", 2: "MCI", 3: "AD"}.get(v, np.nan)


def merge_adni_dxsum_diagnosis(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    dxsum, dxsum_path = load_adni_dxsum()
    if dxsum is None:
        return df.copy(), "ADNI_DXSUM_not_found"

    if not {"RID", "DIAGNOSIS"}.issubset(set(dxsum.columns)):
        return df.copy(), f"ADNI_DXSUM_missing_required_columns:{dxsum_path}"

    rid_col = first_existing(df, ["RID", "rid"])
    visit_col = first_existing(df, ["VISCODE", "VISCODE2", "visit", "VISIT"])
    if rid_col is None or visit_col is None:
        return df.copy(), f"ADNI_DXSUM_no_RID_or_VISCODE_in_metadata_all:{dxsum_path}"

    out = df.copy()
    out["_adni_dxsum_RID"] = pd.to_numeric(out[rid_col], errors="coerce")
    out["_adni_dxsum_VIS"] = out[visit_col].map(normalize_adni_visit)

    dx = dxsum.copy()
    dx["_adni_dxsum_RID"] = pd.to_numeric(dx["RID"], errors="coerce")
    dx["DXSUM_Label"] = dx["DIAGNOSIS"].map(adni_diag_label_from_code)
    sort_cols = [c for c in ["EXAMDATE", "update_stamp"] if c in dx.columns]

    merged = out.copy()

    if "VISCODE" in dx.columns:
        dx1 = dx[["_adni_dxsum_RID", "VISCODE", "DXSUM_Label", "DIAGNOSIS"] + sort_cols].copy()
        dx1["_adni_dxsum_VIS"] = dx1["VISCODE"].map(normalize_adni_visit)
        dx1 = (
            dx1.dropna(subset=["_adni_dxsum_RID", "_adni_dxsum_VIS"])
               .sort_values(sort_cols if sort_cols else ["_adni_dxsum_RID", "_adni_dxsum_VIS"])
               .drop_duplicates(["_adni_dxsum_RID", "_adni_dxsum_VIS"], keep="first")
        )
        merged = merged.merge(
            dx1[["_adni_dxsum_RID", "_adni_dxsum_VIS", "DXSUM_Label", "DIAGNOSIS"]],
            on=["_adni_dxsum_RID", "_adni_dxsum_VIS"],
            how="left",
        )
    else:
        merged["DXSUM_Label"] = np.nan
        merged["DIAGNOSIS"] = np.nan

    if "VISCODE2" in dx.columns:
        dx2 = dx[["_adni_dxsum_RID", "VISCODE2", "DXSUM_Label", "DIAGNOSIS"] + sort_cols].copy()
        dx2["_adni_dxsum_VIS"] = dx2["VISCODE2"].map(normalize_adni_visit)
        dx2 = (
            dx2.dropna(subset=["_adni_dxsum_RID", "_adni_dxsum_VIS"])
               .sort_values(sort_cols if sort_cols else ["_adni_dxsum_RID", "_adni_dxsum_VIS"])
               .drop_duplicates(["_adni_dxsum_RID", "_adni_dxsum_VIS"], keep="first")
               .rename(columns={"DXSUM_Label": "DXSUM_Label_v2", "DIAGNOSIS": "DIAGNOSIS_v2"})
        )
        merged = merged.merge(
            dx2[["_adni_dxsum_RID", "_adni_dxsum_VIS", "DXSUM_Label_v2", "DIAGNOSIS_v2"]],
            on=["_adni_dxsum_RID", "_adni_dxsum_VIS"],
            how="left",
        )
        merged["DXSUM_Label"] = merged["DXSUM_Label"].combine_first(merged["DXSUM_Label_v2"])
        merged["DIAGNOSIS"] = merged["DIAGNOSIS"].combine_first(merged["DIAGNOSIS_v2"])

    n = int(merged["DXSUM_Label"].notna().sum())
    return merged, f"ADNI_DXSUM:{dxsum_path};mapped={n}"


def load_adni_covars() -> tuple[Optional[pd.DataFrame], str]:
    for path in ADNI_COVARS_CANDIDATES:
        if path.exists():
            if path.suffix.lower() == ".tsv":
                return pd.read_csv(path, sep="\t", low_memory=False), str(path)
            return pd.read_csv(path, low_memory=False), str(path)
    return None, ""


def adni_graph_key(x: object) -> object:
    if pd.isna(x):
        return np.nan
    return str(x).strip().replace("_", "")


def merge_adni_group_from_covars(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    cov, cov_path = load_adni_covars()
    if cov is None or "GROUP" not in cov.columns:
        return df.copy(), "ADNI_GROUP_covars_not_found"

    sid_col = first_existing(cov, ["subject_id", "SUBJECT_ID", "connectome_key", "graph_id"])
    if sid_col is None:
        return df.copy(), f"ADNI_GROUP_covars_no_subject_id:{cov_path}"

    key_col = first_existing(df, ["connectome_key", "connectome_full_key", "graph_id", "subject_id"])
    if key_col is None:
        return df.copy(), f"ADNI_GROUP_no_key_in_metadata_all:{cov_path}"

    cov2 = cov[[sid_col, "GROUP"]].copy()
    cov2["_adni_graph_key"] = cov2[sid_col].map(adni_graph_key)
    cov2 = cov2.dropna(subset=["_adni_graph_key"]).drop_duplicates("_adni_graph_key", keep="first")

    out = df.copy()
    out["_adni_graph_key"] = out[key_col].map(adni_graph_key)
    out = out.merge(
        cov2[["_adni_graph_key", "GROUP"]].rename(columns={"GROUP": "ADNI_GROUP"}),
        on="_adni_graph_key",
        how="left",
    )
    n = int(out["ADNI_GROUP"].notna().sum())
    return out, f"ADNI_GROUP_from_covars:{cov_path};mapped={n}"



# =============================================================================
# CORRECTED ADNI/HABS METADATA MERGE
# =============================================================================

def read_table_any(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, low_memory=False)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported file type: {path}")


def first_existing_path(paths: Sequence[Path]) -> Optional[Path]:
    for p in paths:
        if p.exists():
            return p
    return None


def rid_to_Rxxxx(x: object) -> object:
    """Convert ADNI RID-like value to R####."""
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    m = re.search(r"R(\d+)", s, flags=re.IGNORECASE)
    if m:
        return "R" + str(int(m.group(1))).zfill(4)
    if re.fullmatch(r"\d+(?:\.0)?", s):
        return "R" + str(int(float(s))).zfill(4)
    return np.nan


def adni_visit_to_y(vis: object) -> object:
    """Map ADNI visit labels to DWI y-labels."""
    if pd.isna(vis):
        return np.nan
    s = str(vis).strip().lower()
    mapping = {
        "sc": "y0", "bl": "y0", "baseline": "y0", "init": "y0",
        "m00": "y0", "m0": "y0", "0": "y0", "0.0": "y0",
        "m06": "y0.5", "m6": "y0.5", "6": "y0.5", "6.0": "y0.5",
        "m12": "y1", "12": "y1", "12.0": "y1", "y1": "y1",
        "m24": "y2", "24": "y2", "24.0": "y2", "y2": "y2",
        "m36": "y3", "36": "y3", "36.0": "y3", "y3": "y3",
        "m48": "y4", "48": "y4", "48.0": "y4", "y4": "y4",
        "m60": "y5", "60": "y5", "60.0": "y5", "y5": "y5",
    }
    if s in mapping:
        return mapping[s]
    m = re.search(r"y(\d+(?:\.\d+)?)", s)
    if m:
        return f"y{m.group(1)}"
    m = re.search(r"m(\d+)", s)
    if m:
        years = float(m.group(1)) / 12.0
        return f"y{int(years)}" if years.is_integer() else f"y{years:g}"
    return np.nan


def make_adni_dwi_key(df: pd.DataFrame) -> pd.Series:
    """Construct ADNI DWI key R####_y# from DWI/connectome key or RID+VISCODE."""
    out = pd.Series(np.nan, index=df.index, dtype=object)

    existing = first_existing(df, ["DWI", "connectome_key", "connectome_full_key", "graph_id", "table1_session_key"])
    if existing is not None:
        s = as_key(df[existing])
        valid = s.astype(str).str.match(r"^R\d+_y\d+(?:\.\d+)?$", na=False)
        out.loc[valid] = s.loc[valid]

    rid_col = first_existing(df, ["RID", "RID_num", "rid"])
    visit_col = first_existing(df, ["VISCODE", "VISCODE2", "visit", "VISIT"])
    if rid_col is not None and visit_col is not None:
        r = df[rid_col].map(rid_to_Rxxxx)
        y = df[visit_col].map(adni_visit_to_y)
        ok = r.notna() & y.notna() & out.isna()
        out.loc[ok] = r.loc[ok].astype(str) + "_" + y.loc[ok].astype(str)

    return out


def make_habs_dwi_key(df: pd.DataFrame) -> pd.Series:
    """Extract HABS DWI key H####_y# from DWI/runno/connectome key."""
    existing = first_existing(df, ["DWI", "runno", "connectome_key", "connectome_full_key", "graph_id", "table1_session_key"])
    if existing is None:
        return pd.Series(np.nan, index=df.index, dtype=object)
    s = as_key(df[existing])
    return s.where(s.astype(str).str.match(r"^H\d+_y\d+(?:\.\d+)?$", na=False), np.nan)


def load_corrected_metadata(cohort: str) -> tuple[Optional[pd.DataFrame], str]:
    if cohort == "ADNI":
        path = first_existing_path(ADNI_CORRECTED_METADATA_CANDIDATES)
    elif cohort == "HABS":
        path = first_existing_path(HABS_CORRECTED_METADATA_CANDIDATES)
    else:
        return None, ""

    if path is None:
        return None, ""

    return read_table_any(path), str(path)


def merge_corrected_metadata(df: pd.DataFrame, cohort: str) -> tuple[pd.DataFrame, str]:
    """
    Merge corrected cohort metadata onto model metadata rows before harmonization.

    ADNI:
      Prefer $WORK/ines/data/harmonization/ADNI/metadata/ADNI_metadata_with_DWI.csv
      and merge by DWI key R####_y#.

    HABS:
      Prefer $WORK/ines/data/harmonization/HABS/metadata/HABS_metadata_session_level_DWI_filled.csv
      and merge by DWI key H####_y#.
    """
    if cohort not in {"ADNI", "HABS"}:
        return df.copy(), "not_applicable"

    meta, meta_path = load_corrected_metadata(cohort)
    if meta is None or meta.empty:
        return df.copy(), f"{cohort}_corrected_metadata_not_found"

    left = df.copy()
    right = meta.copy()

    if cohort == "ADNI":
        left["_corrected_DWI"] = make_adni_dwi_key(left)
        right["_corrected_DWI"] = make_adni_dwi_key(right)
    else:
        left["_corrected_DWI"] = make_habs_dwi_key(left)
        right["_corrected_DWI"] = make_habs_dwi_key(right)

    right = right.dropna(subset=["_corrected_DWI"]).drop_duplicates("_corrected_DWI", keep="first")
    if right.empty:
        return left.drop(columns=["_corrected_DWI"], errors="ignore"), f"{cohort}_corrected_metadata_no_DWI_keys:{meta_path}"

    left_keys_nonmissing = int(left["_corrected_DWI"].notna().sum())

    before_cols = set(left.columns)
    merged = left.merge(
        right,
        on="_corrected_DWI",
        how="left",
        suffixes=("", "_correctedmeta"),
    )

    # Fill missing values in existing columns from corrected metadata. This keeps
    # model prediction columns intact while improving demographics/status/APOE.
    fill_candidates = [
        "DWI", "runno", "Subject", "Med_ID", "Visit_ID", "Year",
        "Age", "AGE", "age", "VISIT_AGE", "SUBJECT_AGE_SCREEN",
        "Sex", "sex", "PTGENDER", "gender", "ID_Gender",
        "CDX_Cog", "NORMCOG", "DEMENTED", "IMPNOMCI",
        "APOE4_Genotype", "APOE4_Positivity", "APOE", "genotype",
        "APOE_A1", "APOE_A2", "VISCODE", "VISCODE2", "RID", "PTID",
        "BMI", "OM_BMI", "SBP", "DBP", "Pulse",
        "OM_BP1_SYS", "OM_BP2_SYS", "OM_BP1_DIA", "OM_BP2_DIA",
    ]

    filled_cells = 0
    for c in fill_candidates:
        cc = f"{c}_correctedmeta"
        if c in merged.columns and cc in merged.columns:
            before_missing = int(merged[c].isna().sum())
            merged[c] = merged[c].combine_first(merged[cc])
            filled_cells += before_missing - int(merged[c].isna().sum())

    # Keep useful corrected metadata columns that were not already present.
    for c in list(right.columns):
        if c == "_corrected_DWI":
            continue
        cc = f"{c}_correctedmeta"
        if cc in merged.columns and c not in before_cols:
            merged[c] = merged[cc]

    if "DWI" not in merged.columns:
        merged["DWI"] = merged["_corrected_DWI"]
    else:
        merged["DWI"] = merged["DWI"].combine_first(merged["_corrected_DWI"])

    suffix_cols = [c for c in merged.columns if c.endswith("_correctedmeta")]
    rows_with_corrected_values = int(merged[suffix_cols].notna().any(axis=1).sum()) if suffix_cols else 0

    merged = merged.drop(columns=suffix_cols + ["_corrected_DWI"], errors="ignore")

    return (
        merged,
        f"{cohort}_corrected_metadata:{meta_path};"
        f"input_DWI_keys={left_keys_nonmissing};"
        f"rows_with_corrected_metadata={rows_with_corrected_values};"
        f"filled_cells={filled_cells}"
    )

# =============================================================================
# STATUS / DEMOGRAPHICS HARMONIZATION
# =============================================================================

def normalize_sex_value(x: object) -> str:
    if pd.isna(x):
        return "Unknown"
    s = str(x).strip().lower()
    if s in ["m", "male", "1", "1.0"]:
        return "Male"
    if s in ["f", "female", "2", "2.0"]:
        return "Female"
    if s.startswith("m"):
        return "Male"
    if s.startswith("f"):
        return "Female"
    return "Unknown"


def normalize_apoe_single(x: object) -> object:
    if pd.isna(x):
        return np.nan
    s = str(x).strip().upper().replace("/", "").replace("-", "").replace(" ", "").replace("_", "")
    if s in ["NAN", "NONE", ""]:
        return np.nan

    m = re.match(r"APOE?([234])([234])$", s)
    if m:
        a1, a2 = sorted([m.group(1), m.group(2)])
        return f"APOE{a1}{a2}"

    alleles = re.findall(r"[234]", s)
    if len(alleles) >= 2:
        a1, a2 = sorted([alleles[0], alleles[1]])
        return f"APOE{a1}{a2}"

    return s


def normalize_apoe_from_two_cols(a1: pd.Series, a2: pd.Series) -> pd.Series:
    x = pd.to_numeric(a1, errors="coerce")
    y = pd.to_numeric(a2, errors="coerce")
    out = []
    for aa, bb in zip(x, y):
        if pd.isna(aa) or pd.isna(bb):
            out.append(np.nan)
        else:
            aa, bb = int(aa), int(bb)
            if aa in [2, 3, 4] and bb in [2, 3, 4]:
                lo, hi = sorted([aa, bb])
                out.append(f"APOE{lo}{hi}")
            else:
                out.append(np.nan)
    return pd.Series(out, index=a1.index)


def add_harmonized_columns(df: pd.DataFrame, cohort: str) -> tuple[pd.DataFrame, str]:
    out = df.copy()
    notes = []

    if cohort in {"ADNI", "HABS"}:
        out, src_corrected = merge_corrected_metadata(out, cohort)
        notes.append(src_corrected)

    if cohort == "ADNI":
        out, src_dxsum = merge_adni_dxsum_diagnosis(out)
        notes.append(src_dxsum)
        out, src_group = merge_adni_group_from_covars(out)
        notes.append(src_group)

    # Age
    age_col = first_existing(out, ["age_true", "Age", "AGE", "age", "VISIT_AGE", "SUBJECT_AGE_SCREEN"])
    out["age_for_table1"] = pd.to_numeric(out[age_col], errors="coerce") if age_col else np.nan
    notes.append(f"age={age_col}")

    # Sex
    sex_col = first_existing(out, ["sex", "Sex", "PTGENDER", "SUBJECT_SEX", "gender", "sex_numeric"])
    out["sex_label"] = out[sex_col].map(normalize_sex_value) if sex_col else "Unknown"
    notes.append(f"sex={sex_col}")

    # APOE
    apoe_label = pd.Series(np.nan, index=out.index, dtype=object)
    if cohort == "ADNI":
        a1 = first_existing(out, ["APOE_A1", "APGEN1"])
        a2 = first_existing(out, ["APOE_A2", "APGEN2"])
        geno = first_existing(out, ["genotype", "APOE", "GENOTYPE", "APOE4_Genotype"])
        if a1 and a2:
            apoe_label = normalize_apoe_from_two_cols(out[a1], out[a2])
            notes.append(f"apoe={a1}+{a2}")
        elif geno:
            apoe_label = out[geno].map(normalize_apoe_single)
            notes.append(f"apoe={geno}")
        else:
            notes.append("apoe=missing")
    else:
        geno = first_existing(out, ["genotype", "APOE", "APOE4_Genotype", "APOE4_Genotype_x", "APOE4_Genotype_y"])
        if geno:
            apoe_label = out[geno].map(normalize_apoe_single)
            notes.append(f"apoe={geno}")
        else:
            notes.append("apoe=missing")

    out["APOE_Label"] = apoe_label
    out["APOE4_carrier_strict_34_44"] = np.nan
    out.loc[out["APOE_Label"].isin(["APOE22", "APOE23", "APOE33"]), "APOE4_carrier_strict_34_44"] = 0
    out.loc[out["APOE_Label"].isin(["APOE34", "APOE44"]), "APOE4_carrier_strict_34_44"] = 1

    # Diagnosis
    norm = pd.Series(pd.NA, index=out.index, dtype="Int64")
    dem = pd.Series(pd.NA, index=out.index, dtype="Int64")
    dx = pd.Series("Unknown", index=out.index, dtype=object)
    dx_source = ""

    if cohort == "ADNI":
        if "DXSUM_Label" in out.columns and out["DXSUM_Label"].notna().any():
            s = out["DXSUM_Label"].astype(str).str.strip()
            sl = s.str.lower()
            dx = s.where(~sl.isin(["nan", "none", ""]), "Unknown")
            norm.loc[sl.eq("cn")] = 1
            norm.loc[sl.isin(["mci", "ad"])] = 0
            dem.loc[sl.eq("ad")] = 1
            dem.loc[sl.isin(["cn", "mci"])] = 0
            dx_source = "ADNI_DXSUM_DIAGNOSIS"

            missing_dx = norm.isna() & dem.isna()
            if missing_dx.any() and "ADNI_GROUP" in out.columns:
                g = out.loc[missing_dx, "ADNI_GROUP"].astype(str).str.strip()
                gl = g.str.lower()
                dx.loc[missing_dx] = g.where(~gl.isin(["nan", "none", ""]), "Unknown")
                norm.loc[missing_dx & gl.eq("cn")] = 1
                norm.loc[missing_dx & gl.isin(["smc", "emci", "mci", "lmci", "ad"])] = 0
                dem.loc[missing_dx & gl.eq("ad")] = 1
                dem.loc[missing_dx & gl.isin(["cn", "smc", "emci", "mci", "lmci"])] = 0
                dx_source += "+ADNI_covars_GROUP_fallback"

        elif "ADNI_GROUP" in out.columns and out["ADNI_GROUP"].notna().any():
            s = out["ADNI_GROUP"].astype(str).str.strip()
            sl = s.str.lower()
            dx = s.where(~sl.isin(["nan", "none", ""]), "Unknown")
            norm.loc[sl.eq("cn")] = 1
            norm.loc[sl.isin(["smc", "emci", "mci", "lmci", "ad"])] = 0
            dem.loc[sl.eq("ad")] = 1
            dem.loc[sl.isin(["cn", "smc", "emci", "mci", "lmci"])] = 0
            dx_source = "ADNI_covars_GROUP"

        else:
            col = first_existing(out, ["Research Group", "DX", "DX_bl", "group_status", "NORMCOG"])
            dx_source = col or "missing"
            if col is not None:
                s = out[col].astype(str).str.strip()
                sl = s.str.lower()
                missing = sl.isin(["nan", "none", ""])
                is_cn = sl.isin(["cn", "normal", "control", "cognitively normal", "nc", "healthy_control"])
                is_dem = sl.str.contains("dement|ad", regex=True, na=False)
                dx.loc[~missing] = s.loc[~missing]
                dx.loc[is_cn] = "CN"
                norm.loc[is_cn] = 1
                norm.loc[(~is_cn) & (~missing)] = 0
                dem.loc[is_dem] = 1
                dem.loc[(~is_dem) & (~missing)] = 0

    elif cohort == "ADRC":
        norm_col = first_existing(out, ["NORMCOG", "normcog"])
        mci_col = first_existing(out, ["IMPNOMCI", "impnomci"])
        dem_col = first_existing(out, ["DEMENTED", "demented"])
        dx_source = "+".join([c for c in [norm_col, mci_col, dem_col] if c is not None]) or "missing"
        if norm_col:
            x = pd.to_numeric(out[norm_col], errors="coerce")
            norm.loc[x.eq(1)] = 1
            dem.loc[x.eq(1)] = 0
            dx.loc[x.eq(1)] = "Normal"
        if mci_col:
            x = pd.to_numeric(out[mci_col], errors="coerce")
            norm.loc[x.eq(1)] = 0
            dem.loc[x.eq(1)] = 0
            dx.loc[x.eq(1)] = "MCI"
        if dem_col:
            x = pd.to_numeric(out[dem_col], errors="coerce")
            norm.loc[x.eq(1)] = 0
            dem.loc[x.eq(1)] = 1
            dx.loc[x.eq(1)] = "Demented"

    elif cohort == "HABS":
        col = first_existing(out, ["CDX_Cog", "CDX_COG", "NORMCOG", "group_status"])
        dx_source = col or "missing"
        if col:
            if normalize_name(col) == "normcog":
                x = pd.to_numeric(out[col], errors="coerce")
                norm.loc[x.eq(1)] = 1
                norm.loc[x.eq(0)] = 0
                dem.loc[x.notna()] = 0
                dx.loc[x.eq(1)] = "CN"
                dx.loc[x.eq(0)] = "Not CN"
            else:
                x = pd.to_numeric(out[col], errors="coerce")
                norm.loc[x.eq(0)] = 1
                norm.loc[x.isin([1, 2, 9])] = 0
                dem.loc[x.eq(2)] = 1
                dem.loc[x.isin([0, 1, 9])] = 0
                dx.loc[x.eq(0)] = "CN"
                dx.loc[x.eq(1)] = "MCI"
                dx.loc[x.eq(2)] = "AD/Dementia"
                dx.loc[x.eq(9)] = "Other/Unknown"

    elif cohort == "AD_DECODE":
        col = first_existing(out, ["Risk", "risk", "group_status", "NORMCOG"])
        dx_source = col or "missing"
        if col:
            if normalize_name(col) == "normcog":
                x = pd.to_numeric(out[col], errors="coerce")
                norm.loc[x.eq(1)] = 1
                norm.loc[x.eq(0)] = 0
                dem.loc[x.notna()] = 0
                dx.loc[x.eq(1)] = "Risk0/1/NoRisk control"
                dx.loc[x.eq(0)] = "Risk/Impaired"
            else:
                s = out[col].astype(str).str.strip()
                sl = s.str.lower()

                # AD_DECODE update: Risk 0 and Risk 1 are controls.
                is_control = sl.isin([
                    "", "nan", "none",
                    "0", "0.0", "risk 0", "risk0",
                    "1", "1.0", "risk 1", "risk1",
                    "norisk", "no risk", "control", "cn", "normal", "unimpaired",
                ])
                is_dem = sl.str.contains(r"\bad\b|dement", regex=True, na=False)

                dx = s.replace(r"^\s*$", "NoRisk", regex=True)
                dx.loc[sl.isin(["nan", "none", ""])] = "NoRisk"
                dx.loc[sl.isin(["0", "0.0", "risk 0", "risk0"])] = "Risk0_control"
                dx.loc[sl.isin(["1", "1.0", "risk 1", "risk1"])] = "Risk1_control"

                norm.loc[is_control] = 1
                norm.loc[~is_control] = 0
                dem.loc[is_dem] = 1
                dem.loc[~is_dem] = 0

    out["DX_Label_harmonized"] = dx
    out["NORMCOG_01"] = norm
    out["DEMENTIA_01"] = dem
    out["COG_STATUS_SOURCE"] = dx_source
    notes.append(f"dx={dx_source}")

    # Keys
    if cohort == "ADNI":
        subj = first_existing(out, ["PTID", "subject_id", "RID"])
        sess = first_existing(out, ["DWI", "connectome_key", "connectome_full_key", "graph_id"])
    elif cohort == "ADRC":
        subj = first_existing(out, ["PTID", "subject_id", "match_id", "connectome_key"])
        sess = first_existing(out, ["connectome_key", "connectome_full_key", "graph_id"])
    elif cohort == "HABS":
        subj = first_existing(out, ["DWI_subject", "Subject", "subject_id"])
        sess = first_existing(out, ["DWI", "runno", "connectome_key", "connectome_full_key", "graph_id"])
    elif cohort == "AD_DECODE":
        subj = first_existing(out, ["MRI_Exam", "MRI_Exam_fixed", "subject_id"])
        sess = first_existing(out, ["connectome_key", "connectome_full_key", "graph_id"])
    else:
        subj, sess = None, None

    out["table1_subject_key"] = as_key(out[subj]) if subj else pd.Series(np.arange(len(out)).astype(str), index=out.index)
    out["table1_session_key"] = as_key(out[sess]) if sess else out["table1_subject_key"]
    notes.append(f"subject_key={subj};session_key={sess}")

    return out, "; ".join(notes)


# =============================================================================
# WRITE ENRICHED METADATA FOR ALL MODEL/COHORTS
# =============================================================================

def enrich_one(cohort: str, feature_set: str) -> tuple[Optional[pd.DataFrame], dict]:
    path = metadata_all_path(cohort, feature_set)
    diag = {
        "cohort": cohort,
        "feature_set": feature_set,
        "metadata_all_path": str(path),
        "metadata_all_exists": path.exists(),
        "n_rows_loaded": 0,
        "prediction_merge_source": "",
        "cbag_col": "",
        "n_rows_with_cbag_or_BAG": 0,
        "source_notes": "",
        "status": "",
        "enriched_output_path": str(enriched_status_path(cohort, feature_set)),
    }

    if not path.exists():
        diag["status"] = "missing_metadata_all_with_predictions"
        return None, diag

    df = pd.read_csv(path, low_memory=False)
    diag["n_rows_loaded"] = len(df)

    df, merge_source = merge_predictions_if_needed(df, cohort, feature_set)
    diag["prediction_merge_source"] = merge_source

    df, source_notes = add_harmonized_columns(df, cohort)
    diag["source_notes"] = source_notes

    cbag_col = get_cbag_col(df)
    diag["cbag_col"] = cbag_col
    if cbag_col:
        diag["n_rows_with_cbag_or_BAG"] = int(pd.to_numeric(df[cbag_col], errors="coerce").notna().sum())
        diag["status"] = "ok"
    else:
        diag["status"] = "missing_cbag_or_BAG_after_merge"

    df["cohort"] = cohort
    df["feature_set"] = feature_set

    out_path = enriched_status_path(cohort, feature_set)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    return df, diag


def enrich_all_metadata() -> pd.DataFrame:
    rows = []
    for fs in FEATURE_SETS:
        for cohort in COHORTS:
            _, diag = enrich_one(cohort, fs)
            rows.append(diag)
            print(
                f"[{diag['status']:<35}] {fs:<24} {cohort:<10} "
                f"rows={diag['n_rows_loaded']:<4} cBAG={diag['n_rows_with_cbag_or_BAG']:<4} "
                f"out={diag['enriched_output_path']}"
            )
    return pd.DataFrame(rows)


# =============================================================================
# TABLE 1 BUILDING
# =============================================================================

def load_primary_scan_level(cohort: str, feature_set: str = PRIMARY_FEATURE_SET) -> tuple[pd.DataFrame, dict]:
    out_path = enriched_status_path(cohort, feature_set)
    if not out_path.exists():
        _, diag = enrich_one(cohort, feature_set)
    else:
        diag = {"cohort": cohort, "feature_set": feature_set, "status": "ok", "enriched_output_path": str(out_path)}

    if not out_path.exists():
        return pd.DataFrame(), diag

    df = pd.read_csv(out_path, low_memory=False)
    cbag_col = get_cbag_col(df)
    diag["cbag_col"] = cbag_col
    diag["n_rows_loaded"] = len(df)

    if cbag_col:
        mask = pd.to_numeric(df[cbag_col], errors="coerce").notna()
        df = df.loc[mask].copy()
        diag["n_rows_with_cbag_or_BAG"] = len(df)
        diag["status"] = "ok" if len(df) else "no_cbag_or_BAG_rows"
    else:
        diag["n_rows_with_cbag_or_BAG"] = 0
        diag["status"] = "missing_cbag_or_BAG_after_merge"

    return df, diag


def make_scan_and_subject_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scans = []
    diags = []
    for cohort in COHORTS:
        df, diag = load_primary_scan_level(cohort, PRIMARY_FEATURE_SET)
        diags.append(diag)
        if not df.empty:
            scans.append(df)

    scan_df = pd.concat(scans, ignore_index=True, sort=False) if scans else pd.DataFrame()
    diag_df = pd.DataFrame(diags)

    if scan_df.empty:
        return scan_df, pd.DataFrame(), diag_df

    scan_df["_subject_sort_age"] = pd.to_numeric(scan_df["age_for_table1"], errors="coerce")
    subject_df = (
        scan_df
        .sort_values(["cohort", "table1_subject_key", "_subject_sort_age"], na_position="last")
        .drop_duplicates(["cohort", "table1_subject_key"], keep="first")
        .drop(columns=["_subject_sort_age"])
        .copy()
    )
    return scan_df, subject_df, diag_df


def diagnostic_counts_for_cohort(subject_df: pd.DataFrame, cohort: str) -> dict:
    sub = subject_df[subject_df["cohort"].eq(cohort)].copy()
    n = len(sub)
    dx = sub["DX_Label_harmonized"].fillna("Unknown").astype(str)
    vc = dx.value_counts(dropna=False).to_dict()

    if cohort == "ADNI":
        normal = int(dx.eq("CN").sum())
        mci = int(dx.eq("MCI").sum())
        dementia = int(dx.eq("AD").sum())
        familial = 0
        other = 0
        missing = int(dx.isin(["Unknown", "Missing", "nan", "None", ""]).sum())
    elif cohort == "ADRC":
        normal = int(dx.eq("Normal").sum())
        mci = int(dx.eq("MCI").sum())
        dementia = int(dx.eq("Demented").sum())
        familial = 0
        other = int(dx.eq("Unknown").sum())
        missing = 0
    elif cohort == "HABS":
        normal = int(dx.eq("CN").sum())
        mci = int(dx.eq("MCI").sum())
        dementia = int(dx.eq("AD/Dementia").sum())
        familial = 0
        other = int(dx.eq("Other/Unknown").sum())
        missing = 0
    elif cohort == "AD_DECODE":
        normal = int(dx.isin(["NoRisk", "Risk0_control", "Risk1_control"]).sum())
        mci = int(dx.eq("MCI").sum())
        dementia = int(dx.str.contains(r"\bAD\b|Dement", case=False, regex=True, na=False).sum())
        familial = int(dx.eq("Familial").sum())
        listed = normal + mci + dementia + familial
        other = 0
        missing = max(n - listed, 0)
    else:
        normal = mci = dementia = familial = other = missing = 0

    return {
        "normal": normal,
        "mci": mci,
        "dementia": dementia,
        "familial": familial,
        "other": other,
        "missing": missing,
        "raw_dx_counts": "; ".join([f"{k}:{v}" for k, v in vc.items()]),
    }


def cohort_col_values(subject_df: pd.DataFrame, scan_df: pd.DataFrame, cohort: str) -> dict:
    sub = subject_df[subject_df["cohort"].eq(cohort)].copy()
    scans = scan_df[scan_df["cohort"].eq(cohort)].copy()
    n = len(sub)

    sex = sub["sex_label"].fillna("Unknown")
    apoe = sub["APOE_Label"].fillna("Missing")
    apoe4 = pd.to_numeric(sub["APOE4_carrier_strict_34_44"], errors="coerce")
    dx_counts = diagnostic_counts_for_cohort(subject_df, cohort)
    apoe4_denom = int(apoe4.notna().sum())

    return {
        "Study design": STUDY_DESIGN.get(cohort, "—"),
        "Cohort description": COHORT_DESCRIPTION.get(cohort, "—"),
        "Modalities/features included": FEATURE_DESCRIPTION.get(cohort, "—"),
        "cBAG sessions included, n": int(scans["table1_session_key"].nunique()),
        "Unique cBAG subjects, n": int(sub["table1_subject_key"].nunique()),
        "Matched cBAG records/sessions, n": int(len(scans)),
        "Matched subjects used for characteristics, n": n,
        "Age, years, mean ± SD": fmt_mean_sd(sub["age_for_table1"]),
        "Age, years, range": fmt_range(sub["age_for_table1"]),
        "Female, n (%)": fmt_count_pct(int(sex.eq("Female").sum()), n),
        "Male, n (%)": fmt_count_pct(int(sex.eq("Male").sum()), n),
        "Sex missing/not reported, n (%)": fmt_count_pct(int(sex.eq("Unknown").sum()), n),
        "APOE ε4 carrier, n (%)": fmt_count_pct(int(apoe4.eq(1).sum()), apoe4_denom) if apoe4_denom > 0 else "—",
        "APOE22, n (%)": fmt_count_pct(int(apoe.eq("APOE22").sum()), n),
        "APOE23, n (%)": fmt_count_pct(int(apoe.eq("APOE23").sum()), n),
        "APOE24, n (%)": fmt_count_pct(int(apoe.eq("APOE24").sum()), n),
        "APOE33, n (%)": fmt_count_pct(int(apoe.eq("APOE33").sum()), n),
        "APOE34, n (%)": fmt_count_pct(int(apoe.eq("APOE34").sum()), n),
        "APOE44, n (%)": fmt_count_pct(int(apoe.eq("APOE44").sum()), n),
        "APOE genotype missing/not reported, n (%)": fmt_count_pct(int(apoe.eq("Missing").sum()), n),
        "Cognitively normal/normal, n (%)": dash_if_zero_count(dx_counts["normal"], n),
        "MCI, n (%)": dash_if_zero_count(dx_counts["mci"], n),
        "AD/dementia, n (%)": dash_if_zero_count(dx_counts["dementia"], n),
        "Familial AD-risk cohort, n (%)": dash_if_zero_count(dx_counts["familial"], n),
        "Other/unknown diagnostic label, n (%)": dash_if_zero_count(dx_counts["other"], n),
        "Diagnostic group missing/not otherwise classified, n (%)": dash_if_zero_count(dx_counts["missing"], n, use_dash=False),
        "Raw diagnostic counts": dx_counts["raw_dx_counts"],
    }


def make_table1_values(subject_df: pd.DataFrame, scan_df: pd.DataFrame) -> pd.DataFrame:
    row_order = [
        ("Cohort information", "Study design"),
        ("Cohort information", "Cohort description"),
        ("Cohort information", "Modalities/features included"),
        ("Sample size", "cBAG sessions included, n"),
        ("Sample size", "Unique cBAG subjects, n"),
        ("Sample size", "Matched cBAG records/sessions, n"),
        ("Sample size", "Matched subjects used for characteristics, n"),
        ("Demographic characteristics", "Age, years, mean ± SD"),
        ("Demographic characteristics", "Age, years, range"),
        ("Demographic characteristics", "Female, n (%)"),
        ("Demographic characteristics", "Male, n (%)"),
        ("Demographic characteristics", "Sex missing/not reported, n (%)"),
        ("APOE genotype", "APOE ε4 carrier, n (%)"),
        ("APOE genotype", "APOE22, n (%)"),
        ("APOE genotype", "APOE23, n (%)"),
        ("APOE genotype", "APOE24, n (%)"),
        ("APOE genotype", "APOE33, n (%)"),
        ("APOE genotype", "APOE34, n (%)"),
        ("APOE genotype", "APOE44, n (%)"),
        ("APOE genotype", "APOE genotype missing/not reported, n (%)"),
        ("Clinical/diagnostic group", "Cognitively normal/normal, n (%)"),
        ("Clinical/diagnostic group", "MCI, n (%)"),
        ("Clinical/diagnostic group", "AD/dementia, n (%)"),
        ("Clinical/diagnostic group", "Familial AD-risk cohort, n (%)"),
        ("Clinical/diagnostic group", "Other/unknown diagnostic label, n (%)"),
        ("Clinical/diagnostic group", "Diagnostic group missing/not otherwise classified, n (%)"),
    ]

    values = {cohort: cohort_col_values(subject_df, scan_df, cohort) for cohort in COHORTS}
    rows = []
    for section, characteristic in row_order:
        row = {"Section": section, "Characteristic": characteristic}
        for cohort in COHORTS:
            row[cohort] = values[cohort].get(characteristic, "—")
        rows.append(row)
    return pd.DataFrame(rows)


def make_table1_long(table1_values: pd.DataFrame) -> pd.DataFrame:
    return table1_values.melt(
        id_vars=["Section", "Characteristic"],
        value_vars=COHORTS,
        var_name="Cohort",
        value_name="Value",
    )


def make_subject_summary(subject_df: pd.DataFrame, scan_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cohort in COHORTS:
        sub = subject_df[subject_df["cohort"].eq(cohort)]
        scans = scan_df[scan_df["cohort"].eq(cohort)]
        rows.append({
            "cohort": cohort,
            "n_unique_subjects": sub["table1_subject_key"].nunique(),
            "n_sessions": scans["table1_session_key"].nunique(),
            "age_mean": pd.to_numeric(sub["age_for_table1"], errors="coerce").mean(),
            "age_sd": pd.to_numeric(sub["age_for_table1"], errors="coerce").std(ddof=1),
            "n_female": sub["sex_label"].eq("Female").sum(),
            "n_male": sub["sex_label"].eq("Male").sum(),
            "n_normcog": pd.to_numeric(sub["NORMCOG_01"], errors="coerce").eq(1).sum(),
            "n_dementia": pd.to_numeric(sub["DEMENTIA_01"], errors="coerce").eq(1).sum(),
            "dx_counts": "; ".join([f"{k}:{v}" for k, v in sub["DX_Label_harmonized"].fillna("Missing").astype(str).value_counts().items()]),
        })
    return pd.DataFrame(rows)


def make_habs_dwi_availability_summary() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Summarize the full corrected HABS DWI-session metadata, independent of cBAG/model
    availability. This documents why Table 1 analysis-sample counts may differ
    from full DWI-session availability.
    """
    path = first_existing_path(HABS_CORRECTED_METADATA_CANDIDATES)
    if path is None:
        return pd.DataFrame([{
            "metadata_path": "",
            "status": "missing_HABS_corrected_metadata",
            "n_sessions_total": 0,
            "n_unique_subjects_total": 0,
            "n_y0_sessions": 0,
            "n_y2_sessions": 0,
            "n_subjects_with_y0": 0,
            "n_subjects_with_y2": 0,
            "n_subjects_with_y0_y2": 0,
            "n_subjects_only_y0": 0,
            "n_subjects_only_y2": 0,
        }]), pd.DataFrame()

    df = read_table_any(path)
    if "DWI_subject" not in df.columns and "DWI" in df.columns:
        df["DWI_subject"] = df["DWI"].astype(str).str.extract(r"^(H\\d+)", expand=False)
    if "DWI_visit" not in df.columns and "DWI" in df.columns:
        df["DWI_visit"] = df["DWI"].astype(str).str.extract(r"_(y\\d+(?:\\.\\d+)?)", expand=False)

    if "DWI_subject" not in df.columns or "DWI_visit" not in df.columns:
        return pd.DataFrame([{
            "metadata_path": str(path),
            "status": "missing_DWI_subject_or_DWI_visit_columns",
            "n_sessions_total": len(df),
            "n_unique_subjects_total": np.nan,
            "n_y0_sessions": np.nan,
            "n_y2_sessions": np.nan,
            "n_subjects_with_y0": np.nan,
            "n_subjects_with_y2": np.nan,
            "n_subjects_with_y0_y2": np.nan,
            "n_subjects_only_y0": np.nan,
            "n_subjects_only_y2": np.nan,
        }]), pd.DataFrame()

    subject_visits = (
        df[["DWI_subject", "DWI_visit"]]
        .dropna()
        .drop_duplicates()
        .groupby("DWI_subject")["DWI_visit"]
        .apply(lambda x: tuple(sorted(set(x))))
        .reset_index(name="visits")
    )
    subject_visits["visit_pattern"] = subject_visits["visits"].apply(lambda v: ",".join(v))
    subject_visits["has_y0"] = subject_visits["visits"].apply(lambda v: "y0" in v)
    subject_visits["has_y2"] = subject_visits["visits"].apply(lambda v: "y2" in v)
    subject_visits["has_y0_y2"] = subject_visits["has_y0"] & subject_visits["has_y2"]
    subject_visits["only_y0"] = subject_visits["visits"].apply(lambda v: v == ("y0",))
    subject_visits["only_y2"] = subject_visits["visits"].apply(lambda v: v == ("y2",))

    summary = pd.DataFrame([{
        "metadata_path": str(path),
        "status": "ok",
        "n_sessions_total": int(len(df)),
        "n_unique_subjects_total": int(subject_visits["DWI_subject"].nunique()),
        "n_y0_sessions": int(df["DWI_visit"].eq("y0").sum()),
        "n_y2_sessions": int(df["DWI_visit"].eq("y2").sum()),
        "n_subjects_with_y0": int(subject_visits["has_y0"].sum()),
        "n_subjects_with_y2": int(subject_visits["has_y2"].sum()),
        "n_subjects_with_y0_y2": int(subject_visits["has_y0_y2"].sum()),
        "n_subjects_only_y0": int(subject_visits["only_y0"].sum()),
        "n_subjects_only_y2": int(subject_visits["only_y2"].sum()),
    }])

    return summary, subject_visits


# =============================================================================
# FORMATTED EXCEL
# =============================================================================

def save_formatted_excel(table1_values: pd.DataFrame, output_path: Path) -> None:
    try:
        from openpyxl import Workbook
        from openpyxl.styles import Font, PatternFill, Border, Side, Alignment
        from openpyxl.utils import get_column_letter
    except Exception as exc:
        print(f"[WARN] openpyxl not available, skipping formatted XLSX: {exc}")
        return

    wb = Workbook()
    ws = wb.active
    ws.title = "Table 1"

    n_cols = 1 + len(COHORTS)
    dark_blue = "1F4E79"
    medium_blue = "5B9BD5"
    light_blue = "BFE8F5"
    white = "FFFFFF"
    thin_blue = "4FA6D8"

    border = Border(
        left=Side(style="thin", color=thin_blue),
        right=Side(style="thin", color=thin_blue),
        top=Side(style="thin", color=thin_blue),
        bottom=Side(style="thin", color=thin_blue),
    )

    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=n_cols)
    ws["A1"] = "Table 1. Cohort characteristics and analysis sample sizes"
    ws["A1"].font = Font(bold=True, size=12)
    ws["A1"].alignment = Alignment(horizontal="left", vertical="center")
    ws.row_dimensions[1].height = 24

    ws.merge_cells(start_row=3, start_column=1, end_row=3, end_column=n_cols)
    ws["A3"] = "Values are n (%) unless otherwise noted. Percentages are calculated using matched subjects as the denominator for subject-level characteristics."
    ws["A3"].font = Font(italic=True, size=10)
    ws["A3"].alignment = Alignment(wrap_text=True, vertical="center")
    ws.row_dimensions[3].height = 30

    header_row = 5
    headers = ["Characteristic"] + COHORTS
    for j, h in enumerate(headers, start=1):
        cell = ws.cell(header_row, j, h.replace("_", "-"))
        cell.fill = PatternFill("solid", fgColor=dark_blue)
        cell.font = Font(bold=True, color=white)
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cell.border = border

    current_row = header_row + 1
    current_section = None
    for _, r in table1_values.iterrows():
        section = r["Section"]
        characteristic = r["Characteristic"]

        if section != current_section:
            current_section = section
            ws.merge_cells(start_row=current_row, start_column=1, end_row=current_row, end_column=n_cols)
            c = ws.cell(current_row, 1, section)
            c.fill = PatternFill("solid", fgColor=medium_blue)
            c.font = Font(bold=True, color="000000")
            c.alignment = Alignment(horizontal="left", vertical="center")
            for j in range(1, n_cols + 1):
                ws.cell(current_row, j).border = border
                ws.cell(current_row, j).fill = PatternFill("solid", fgColor=medium_blue)
            current_row += 1

        values = [characteristic] + [r[c] for c in COHORTS]
        for j, val in enumerate(values, start=1):
            cell = ws.cell(current_row, j, val)
            cell.fill = PatternFill("solid", fgColor=light_blue if current_row % 2 == 0 else white)
            cell.border = border
            cell.alignment = Alignment(horizontal="left" if j == 1 else "center", vertical="center", wrap_text=True)
        current_row += 1

    current_row += 1
    ws.merge_cells(start_row=current_row, start_column=1, end_row=current_row, end_column=n_cols)
    ws.cell(current_row, 1).value = (
        "Note. Abbreviations: AD, Alzheimer’s disease; ADNI, Alzheimer’s Disease Neuroimaging Initiative; "
        "ADRC, Alzheimer’s Disease Research Center; APOE, apolipoprotein E; DTI, diffusion tensor imaging; "
        "FA, fractional anisotropy; HABS, Harvard Aging Brain Study; MCI, mild cognitive impairment; "
        "SD, standard deviation. ADNI diagnostic labels use visit-level DXSUM DIAGNOSIS when available "
        "(1=CN, 2=MCI, 3=AD/dementia), with GROUP from ADNI_covars_no_underscores.csv as fallback. "
        "HABS uses the corrected filled DWI-session metadata when available. Main Table 1 values reflect the cBAG/model analysis sample; full HABS DWI y0/y2 availability is saved separately. AD_DECODE Risk 0 and Risk 1 were combined as NORMCOG/control. APOE ε4 strict carriage excludes APOE2/4."
    )
    ws.cell(current_row, 1).font = Font(size=9)
    ws.cell(current_row, 1).alignment = Alignment(wrap_text=True, vertical="top")
    ws.row_dimensions[current_row].height = 85

    ws.freeze_panes = "B6"
    widths = {"A": 40, "B": 28, "C": 28, "D": 28, "E": 28}
    for col, width in widths.items():
        ws.column_dimensions[col].width = width
    for row in range(5, current_row):
        ws.row_dimensions[row].height = 23

    ws2 = wb.create_sheet("README")
    ws2["A1"] = "Generated by final_build_table1_and_enriched_status_metadata.py"
    ws2["A2"] = f"Primary feature set: {PRIMARY_FEATURE_SET}"
    ws2["A3"] = "See CSV diagnostics for full source information."

    wb.save(output_path)


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    ENRICHED_OUTDIR.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("FINAL TABLE 1 + ENRICHED STATUS METADATA")
    print("=" * 100)
    print("WORK:", WORK)
    print("Output:", OUTDIR)
    print("Enriched metadata output:", ENRICHED_OUTDIR)

    print("\nWriting enriched metadata for all feature sets/cohorts...")
    diag_all = enrich_all_metadata()
    diag_all.to_csv(OUTDIR / "Table1_all_enriched_metadata_diagnostics.csv", index=False)

    print("\nBuilding Table 1 from primary feature set:", PRIMARY_FEATURE_SET)
    scan_df, subject_df, diag_primary = make_scan_and_subject_tables()
    if scan_df.empty or subject_df.empty:
        diag_primary.to_csv(OUTDIR / "Table1_diagnostic_sources.csv", index=False)
        raise SystemExit("[ERROR] No scan/subject rows found.")

    table1_wide = make_table1_values(subject_df, scan_df)
    table1_long = make_table1_long(table1_wide)
    subject_summary = make_subject_summary(subject_df, scan_df)
    consistency = make_feature_set_consistency(diag_all)
    habs_dwi_summary, habs_dwi_patterns = make_habs_dwi_availability_summary()

    table1_wide.to_csv(OUTDIR / "Table1_corrected_wide.csv", index=False)
    table1_long.to_csv(OUTDIR / "Table1_corrected_long.csv", index=False)
    subject_df.to_csv(OUTDIR / "Table1_subject_level_corrected.csv", index=False)
    scan_df.to_csv(OUTDIR / "Table1_scan_level_corrected.csv", index=False)
    diag_primary.to_csv(OUTDIR / "Table1_diagnostic_sources.csv", index=False)
    consistency.to_csv(OUTDIR / "Table1_feature_set_consistency_check.csv", index=False)
    subject_summary.to_csv(OUTDIR / "Table1_subject_summary_check.csv", index=False)
    habs_dwi_summary.to_csv(OUTDIR / "Table1_HABS_DWI_availability_summary.csv", index=False)
    habs_dwi_patterns.to_csv(OUTDIR / "Table1_HABS_DWI_visit_patterns.csv", index=False)

    save_formatted_excel(table1_wide, OUTDIR / "Table1_corrected_formatted.xlsx")

    readme = OUTDIR / "Table1_corrected_README.md"
    readme.write_text(
        "# Corrected unified Table 1 and enriched status metadata\n\n"
        f"Primary feature set used for Table 1: `{PRIMARY_FEATURE_SET}`.\n\n"
        "ADNI metadata/diagnosis sources:\n"
        "- Corrected metadata: `$WORK/ines/data/harmonization/ADNI/metadata/ADNI_metadata_with_DWI.csv` when available.\n"
        "- Diagnosis file: `$WORK/ines/data/harmonization/ADNI/metadata/ADNI_Metadata/MetaData/All_Subjects_DXSUM_26Mar2026.csv`\n"
        "- Columns: `RID`, `VISCODE`, `VISCODE2`, `DIAGNOSIS`\n"
        "- Mapping: `DIAGNOSIS=1` CN/control, `2` MCI, `3` AD/dementia.\n"
        "- Fallback: `GROUP` from `ADNI_covars_no_underscores.csv` if DXSUM is unavailable or missing.\n\n"
        "Other diagnosis/status rules:\n"
        "- ADRC: `NORMCOG`, `IMPNOMCI`, `DEMENTED`.\n"
        "- HABS: corrected DWI-session metadata `$WORK/ines/data/harmonization/HABS/metadata/HABS_metadata_session_level_DWI_filled_cognitive_composites.csv` when available, then `_filled.csv`; `CDX_Cog` where 0=CN, 1=MCI, 2=AD/Dementia.\n"
        "- AD_DECODE: `Risk`; Risk 0 and Risk 1 are combined as NORMCOG/control.\n\n"
        "Enriched metadata files are saved in:\n"
        f"`{ENRICHED_OUTDIR}`\n\n"
        "Each enriched metadata file includes `NORMCOG_01`, `DEMENTIA_01`, "
        "`DX_Label_harmonized`, and `COG_STATUS_SOURCE`.\n\n"
        "Main Table 1 files:\n"
        "- `Table1_corrected_formatted.xlsx`\n"
        "- `Table1_corrected_wide.csv`\n"
        "- `Table1_subject_level_corrected.csv`\n"
        "- `Table1_scan_level_corrected.csv`\n"
        "- `Table1_diagnostic_sources.csv`\n"
        "- `Table1_feature_set_consistency_check.csv`\n"
        "- `Table1_HABS_DWI_availability_summary.csv`\n"
        "- `Table1_HABS_DWI_visit_patterns.csv`\n"
    )

    print("\nCorrected Table 1 wide:")
    print(table1_wide.to_string(index=False))

    print("\nSubject summary check:")
    print(subject_summary.to_string(index=False))

    print("\nFull HABS DWI availability summary:")
    print(habs_dwi_summary.to_string(index=False))

    print("\nPrimary diagnostics:")
    print(diag_primary.to_string(index=False))

    print("\nFiles saved under:")
    print(" ", OUTDIR)
    print("\nEnriched metadata saved under:")
    print(" ", ENRICHED_OUTDIR)


if __name__ == "__main__":
    main()
