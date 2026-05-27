#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create final harmonized metadata files for ADNI, ADRC, HABS, and AD_DECODE.

This integrated version does, in one sweep:
  1) Read the graph/DWI-linked cohort metadata used for graph making/counting.
  2) For ADNI, verify/clean DWI keys against actual connectome filenames such as
     R6703_y4_conn_plain.csv in:
       $WORK/ines/data/harmonization/ADNI/connectomes/DWI/plain/*.csv
  3) Filter to metadata rows with connectome/DWI evidence where possible.
  4) Harmonize cognitive status:
       NORMCOG_01, MCI_01, AD_01, DEMENTIA_01, DX_Label_harmonized, COG_STATUS_SOURCE
  5) Merge/harmonize APOE genotype columns:
       APOE_genotype_harmonized, APOE_e2_count, APOE_e3_count, APOE_e4_count,
       APOE4_carriage, APOE4_dosage, APOE4_strict_34_44_carrier, APOE_STATUS_SOURCE
     ADNI APOE source:
       $WORK/ines/data/harmonization/ADNI/metadata/ADNI_Metadata/MetaData/All_Subjects_APOERES_26Mar2026.csv
     HABS APOE source:
       $WORK/ines/data/harmonization/HABS/metadata/RP_HD_7_Genomics.xlsx
  6) Add harmonized cognitive composites:
       Memory_Composite, Executive_Function_Composite, Processing_Speed_Composite,
       Language_Composite, Visuospatial_Composite, Global_Cognition_Composite,
       cognition_composite, cognitive_impairment_composite
  7) Deduplicate to one row per connectome key when possible, preferring rows with diagnosis,
     cognitive composites, APOE, and clearer visit labels.

Outputs:
  $WORK/ines/data/harmonization/harmonized_metadata/
    ADNI_harmonized_metadata.csv
    ADRC_harmonized_metadata.csv
    HABS_harmonized_metadata.csv
    AD_DECODE_harmonized_metadata.csv
    <COHORT>_cognitive_composite_mapping.csv
    harmonized_metadata_summary.csv

This script does not overwrite original metadata files.
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
BASE = WORK / "ines" / "data" / "harmonization"
OUTDIR = BASE / "harmonized_metadata"
OUTDIR.mkdir(parents=True, exist_ok=True)

FILTER_TO_CONNECTOME_ROWS = True
DEDUPLICATE_TO_ONE_ROW_PER_CONNECTOME = True
COUNT_ADNI_SMC_AS_NORMAL = False
FILTER_ADNI_TO_Y0_Y4_SUBJECTS = False
MIN_COMPONENTS_PER_DOMAIN = 1

METADATA_CANDIDATES: dict[str, list[Path]] = {
    "ADNI": [
        BASE / "ADNI" / "metadata" / "ADNI_metadata_with_DWI.csv",
        BASE / "ADNI" / "metadata" / "ADNI_metadata_with_DWI_key.csv",
        BASE / "ADNI" / "metadata" / "ADNI_metadata_figure5_enriched_with_NORMCOG_DEMENTIA.csv",
        BASE / "ADNI" / "metadata" / "ADNI_metadata.xlsx",
    ],
    "ADRC": [
        BASE / "ADRC" / "metadata" / "ADRC_metadata_with_DWI.csv",
        BASE / "ADRC" / "metadata" / "ADRC_metadata_with_DWI_key.csv",
        BASE / "ADRC" / "metadata" / "ADRC_metadata_with_NORMCOG_DEMENTIA.csv",
        BASE / "ADRC" / "metadata" / "ADRC_metadata.xlsx",
        BASE / "ADRC" / "metadata" / "ADRC_metadata.csv",
    ],
    "HABS": [
        BASE / "HABS" / "metadata" / "HABS_metadata_session_level_DWI_filled_cognitive_composites.csv",
        BASE / "HABS" / "metadata" / "HABS_metadata_session_level_DWI_filled.csv",
        BASE / "HABS" / "metadata" / "HABS_metadata_session_level_DWI.csv",
        BASE / "HABS" / "metadata" / "HABS_metadata_with_DWI.csv",
        BASE / "HABS" / "metadata" / "HABS_metadata.xlsx",
        BASE / "HABS" / "metadata" / "HABS_metadata.csv",
    ],
    "AD_DECODE": [
        BASE / "AD_DECODE" / "metadata" / "AD_DECODE_metadata_with_DWI.csv",
        BASE / "AD_DECODE" / "metadata" / "AD_DECODE_metadata_with_DWI_key.csv",
        BASE / "AD_DECODE" / "metadata" / "AD_DECODE_metadata.xlsx",
        BASE / "AD_DECODE" / "metadata" / "AD_DECODE_metadata.csv",
        BASE / "ADDECODE" / "metadata" / "AD_DECODE_metadata_with_DWI.csv",
        BASE / "ADDECODE" / "metadata" / "AD_DECODE_metadata.xlsx",
        BASE / "ADDECODE" / "metadata" / "AD_DECODE_metadata.csv",
    ],
}

ADNI_DXSUM_CANDIDATES = [
    BASE / "ADNI" / "metadata" / "ADNI_Metadata" / "MetaData" / "All_Subjects_DXSUM_26Mar2026.csv",
    BASE / "ADNI" / "metadata" / "All_Subjects_DXSUM_26Mar2026.csv",
    BASE / "ADNI" / "metadata" / "All_Subjects_DXSUM_26Mar2026(2).csv",
]

ADNI_COVARS_CANDIDATES = [
    BASE / "ADNI" / "metadata" / "ADNI_covars_no_underscores.csv",
    BASE / "ADNI" / "metadata" / "ADNI_covars.csv",
    BASE / "ADNI" / "metadata" / "ADNI_covars(1).csv",
    WORK / "ADNI" / "metadata" / "ADNI_covars.csv",
]

ADNI_APOE_CANDIDATES = [
    BASE / "ADNI" / "metadata" / "ADNI_Metadata" / "MetaData" / "All_Subjects_APOERES_26Mar2026.csv",
    BASE / "ADNI" / "metadata" / "All_Subjects_APOERES_26Mar2026.csv",
]

HABS_GENOMICS_CANDIDATES = [
    BASE / "HABS" / "metadata" / "RP_HD_7_Genomics.xlsx",
    BASE / "HABS" / "metadata" / "RP_HD_7_Genomics.csv",
]

ADNI_Y0_Y4_CANDIDATES = [
    BASE / "ADNI" / "metadata" / "ADNI_subjects_with_DWI_y0_y4.csv",
]

CONNECTOME_ROOT_CANDIDATES: dict[str, list[Path]] = {
    "ADNI": [
        BASE / "ADNI" / "connectomes" / "DWI" / "plain",
        BASE / "ADNI" / "connectomes" / "DWI",
        BASE / "ADNI" / "connectomes",
        BASE / "ADNI" / "graphs",
        BASE / "ADNI" / "dwi_connectomes",
    ],
    "ADRC": [
        BASE / "ADRC" / "connectomes" / "DWI" / "plain",
        BASE / "ADRC" / "connectomes" / "DWI",
        BASE / "ADRC" / "connectomes",
        BASE / "ADRC" / "graphs",
        BASE / "ADRC" / "dwi_connectomes",
    ],
    "HABS": [
        BASE / "HABS" / "connectomes" / "DWI" / "plain",
        BASE / "HABS" / "connectomes" / "DWI",
        BASE / "HABS" / "connectomes",
        BASE / "HABS" / "graphs",
        BASE / "HABS" / "dwi_connectomes",
    ],
    "AD_DECODE": [
        BASE / "AD_DECODE" / "connectomes" / "DWI" / "plain",
        BASE / "AD_DECODE" / "connectomes" / "DWI",
        BASE / "AD_DECODE" / "connectomes",
        BASE / "AD_DECODE" / "graphs",
        BASE / "ADDECODE" / "connectomes",
        BASE / "ADDECODE" / "graphs",
    ],
}

CONNECTOME_EXTENSIONS = [".csv", ".npy", ".txt", ".mat", ".pkl", ".pickle", ".parquet"]


# =============================================================================
# GENERIC HELPERS
# =============================================================================

def normalize_name(x: object) -> str:
    s = str(x).lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return re.sub(r"_+", "_", s).strip("_")


def first_col(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
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


def first_existing(paths: Sequence[Path]) -> Optional[Path]:
    for p in paths:
        if p.exists():
            return p
    return None


def read_table(path: Path) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, low_memory=False)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported input file type: {path}")


def clean_id(x: object) -> object:
    if pd.isna(x):
        return pd.NA
    s = str(x).strip()
    if not s or s.lower() in {"nan", "none", "<na>"}:
        return pd.NA
    try:
        f = float(s)
        if np.isfinite(f) and f.is_integer():
            return str(int(f))
    except Exception:
        pass
    return s


def coerce_bool_series(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s.fillna(False)
    return s.astype(str).str.strip().str.lower().isin(["true", "1", "yes", "y", "t"])


def zscore_series(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce").astype(float).replace([np.inf, -np.inf], np.nan)
    sd = x.std(skipna=True, ddof=0)
    if pd.isna(sd) or sd == 0:
        return pd.Series(np.nan, index=s.index, dtype=float)
    return (x - x.mean(skipna=True)) / sd


def init_status(df: pd.DataFrame):
    normcog = pd.Series(pd.NA, index=df.index, dtype="Int64")
    mci = pd.Series(pd.NA, index=df.index, dtype="Int64")
    ad = pd.Series(pd.NA, index=df.index, dtype="Int64")
    label = pd.Series("Unknown", index=df.index, dtype="object")
    return normcog, mci, ad, label


def finalize_status(df: pd.DataFrame, normcog: pd.Series, mci: pd.Series, ad: pd.Series, label: pd.Series, source: str) -> pd.DataFrame:
    label = label.copy()
    label.loc[normcog.eq(1)] = "Normal"
    label.loc[mci.eq(1)] = "MCI"
    label.loc[ad.eq(1)] = "AD"
    out = df.copy()
    out["NORMCOG_01"] = normcog.astype("Int64")
    out["MCI_01"] = mci.astype("Int64")
    out["AD_01"] = ad.astype("Int64")
    out["DEMENTIA_01"] = ad.astype("Int64")
    out["DX_Label_harmonized"] = label
    out["COG_STATUS_SOURCE"] = source
    return out


# =============================================================================
# CONNECTOME / DWI INTERSECTION AND DEDUPLICATION
# =============================================================================

def normalize_connectome_key(x: object) -> object:
    """Normalize metadata DWI keys and connectome filenames to e.g. R6703_y4."""
    if pd.isna(x):
        return pd.NA
    s = str(x).strip()
    if not s or s.lower() in {"nan", "none", "<na>"}:
        return pd.NA

    # If value is a path, operate on the basename stem.
    s = Path(s).stem

    # Pull out canonical ADNI/HABS keys wherever embedded in filenames.
    # Examples: R6703_y4_conn_plain -> R6703_y4; H4369_y0_connectome -> H4369_y0.
    m = re.search(r"(R\d+_y\d+(?:\.\d+)?)", s, flags=re.I)
    if m:
        rkey = m.group(1)
        rid, y = rkey.split("_", 1)
        rid_num = re.sub(r"^R", "", rid, flags=re.I)
        try:
            rid = "R" + str(int(rid_num)).zfill(4)
        except Exception:
            rid = rid.upper()
        return rid + "_" + y.lower()

    m = re.search(r"(H\d+_y\d+(?:\.\d+)?)", s, flags=re.I)
    if m:
        hkey = m.group(1)
        subj, y = hkey.split("_", 1)
        return subj.upper() + "_" + y.lower()

    # Generic cleanup for non-ADNI/HABS cohorts.
    low_suffixes = [
        "_conn_plain", "_plain", "_connectome", "_connectivity", "_matrix",
        "_adjacency", "_corr", "_fa", "_md", "_weighted", "_unweighted", "_conn",
    ]
    changed = True
    while changed:
        changed = False
        low = s.lower()
        for suf in low_suffixes:
            if low.endswith(suf):
                s = s[: -len(suf)]
                changed = True
                break
    return s


def collect_connectome_keys(cohort: str) -> tuple[set[str], dict[str, object], dict[str, str]]:
    roots = [p for p in CONNECTOME_ROOT_CANDIDATES.get(cohort, []) if p.exists()]
    if not roots:
        return set(), {
            "connectome_scan_status": "no_root_found",
            "connectome_roots_used": "",
            "n_connectome_files": 0,
            "n_connectome_keys": 0,
        }, {}

    files: list[Path] = []
    for root in roots:
        for ext in CONNECTOME_EXTENSIONS:
            files.extend(root.rglob(f"*{ext}"))

    key_to_file: dict[str, str] = {}
    for f in files:
        k = normalize_connectome_key(f.name)
        if pd.notna(k):
            key_to_file.setdefault(str(k), str(f))

    info = {
        "connectome_scan_status": "ok",
        "connectome_roots_used": ";".join(str(p) for p in roots),
        "n_connectome_files": len(files),
        "n_connectome_keys": len(key_to_file),
    }
    return set(key_to_file), info, key_to_file


def find_metadata_key_col(df: pd.DataFrame) -> Optional[str]:
    return first_col(df, [
        "DWI", "DWI_key", "DWI_connectome_key", "connectome_key", "connectome_full_key",
        "graph_key", "graph_id", "runno", "subject_visit_key", "dwi_key",
    ])


def add_connectome_match_columns(df: pd.DataFrame, cohort: str) -> tuple[pd.DataFrame, dict[str, object]]:
    """Add clean connectome key/path columns and optionally filter to connectome-backed rows."""
    out = df.copy()
    info: dict[str, object] = {
        "connectome_filter_applied": "no",
        "connectome_filter_method": "not_requested_or_not_available",
        "metadata_rows_before_connectome_filter": len(df),
        "metadata_rows_after_connectome_filter": len(df),
        "metadata_connectome_key_col": "",
        "n_matched_metadata_keys": 0,
    }

    key_col = find_metadata_key_col(out)
    if key_col is not None:
        out["CONNECTOME_KEY_CLEAN"] = out[key_col].map(normalize_connectome_key)
        info["metadata_connectome_key_col"] = key_col
    else:
        out["CONNECTOME_KEY_CLEAN"] = pd.NA

    keys, scan_info, key_to_file = collect_connectome_keys(cohort)
    info.update(scan_info)

    # For ADNI and any cohort with connectome roots, scan filenames first. This catches
    # R6703_y4_conn_plain.csv and cleans metadata DWI to R6703_y4.
    if keys and key_col is not None:
        mask = out["CONNECTOME_KEY_CLEAN"].astype(str).isin(keys)
        out["CONNECTOME_KEY_USED_FOR_INTERSECTION"] = out["CONNECTOME_KEY_CLEAN"].where(mask, pd.NA)
        out["MATCHED_CONNECTOME_FILE"] = out["CONNECTOME_KEY_USED_FOR_INTERSECTION"].map(key_to_file)
        if FILTER_TO_CONNECTOME_ROWS:
            out_f = out.loc[mask].copy()
            info.update({
                "connectome_filter_applied": "yes",
                "connectome_filter_method": "connectome_filename_scan",
                "metadata_rows_after_connectome_filter": len(out_f),
                "n_matched_metadata_keys": int(mask.sum()),
            })
            return out_f, info
        info.update({"n_matched_metadata_keys": int(mask.sum())})
        return out, info

    # Fallback 1: explicit has_DWI_connectome.
    has_col = first_col(out, ["has_DWI_connectome", "has_dwi_connectome", "has_connectome", "has_graph"])
    if FILTER_TO_CONNECTOME_ROWS and has_col is not None:
        mask = coerce_bool_series(out[has_col])
        out["CONNECTOME_KEY_USED_FOR_INTERSECTION"] = out["CONNECTOME_KEY_CLEAN"].where(mask, pd.NA)
        out_f = out.loc[mask].copy()
        info.update({
            "connectome_filter_applied": "yes",
            "connectome_filter_method": has_col,
            "metadata_rows_after_connectome_filter": len(out_f),
            "n_matched_metadata_keys": int(mask.sum()),
        })
        return out_f, info

    # Fallback 2: nonmissing path, checking absolute paths when possible.
    path_col = first_col(out, ["DWI_connectome_path", "connectome_path", "graph_path", "dwi_graph_path"])
    if FILTER_TO_CONNECTOME_ROWS and path_col is not None:
        def path_ok(p: object) -> bool:
            if pd.isna(p):
                return False
            ps = str(p).strip()
            if not ps:
                return False
            pp = Path(ps)
            return pp.exists() if pp.is_absolute() else True
        mask = out[path_col].map(path_ok)
        out["CONNECTOME_KEY_USED_FOR_INTERSECTION"] = out["CONNECTOME_KEY_CLEAN"].where(mask, pd.NA)
        out_f = out.loc[mask].copy()
        info.update({
            "connectome_filter_applied": "yes",
            "connectome_filter_method": path_col,
            "metadata_rows_after_connectome_filter": len(out_f),
            "n_matched_metadata_keys": int(mask.sum()),
        })
        return out_f, info

    out["CONNECTOME_KEY_USED_FOR_INTERSECTION"] = out["CONNECTOME_KEY_CLEAN"]
    out["MATCHED_CONNECTOME_FILE"] = pd.NA
    return out, info


def deduplicate_connectome_rows(df: pd.DataFrame, cohort: str) -> tuple[pd.DataFrame, dict[str, object]]:
    info = {
        "dedup_applied": "no",
        "dedup_key_col": "",
        "n_rows_before_dedup": len(df),
        "n_rows_after_dedup": len(df),
        "n_duplicate_connectome_rows_removed": 0,
    }
    if not DEDUPLICATE_TO_ONE_ROW_PER_CONNECTOME:
        return df.copy(), info

    key_col = first_col(df, ["CONNECTOME_KEY_USED_FOR_INTERSECTION", "CONNECTOME_KEY_CLEAN", "DWI", "DWI_key", "connectome_key", "graph_id"])
    if key_col is None:
        return df.copy(), info

    out = df.copy()
    out["_dedup_key"] = out[key_col].map(normalize_connectome_key)
    valid = out["_dedup_key"].notna()
    if not valid.any():
        return df.copy(), info

    dx_known = ~out.get("DX_Label_harmonized", pd.Series("Unknown", index=out.index)).astype(str).str.lower().isin(["unknown", "nan", "none", ""])
    cog_known = pd.to_numeric(out.get("Global_Cognition_Composite", pd.Series(np.nan, index=out.index)), errors="coerce").notna()
    apoe_known = out.get("APOE_genotype_harmonized", pd.Series(pd.NA, index=out.index)).notna()
    visit_col = first_col(out, ["VISCODE", "VISCODE2", "DWI_visit_label", "DWI_visit", "visit"])
    if visit_col is not None:
        visit = out[visit_col].astype(str).str.lower()
        visit_score = np.select(
            [visit.isin(["sc", "bl", "y0", "m00", "m0"]), visit.isin(["y4", "m48"]), visit.eq("init")],
            [3, 3, 1],
            default=2,
        )
    else:
        visit_score = np.zeros(len(out), dtype=int)

    out["_dedup_dx_known"] = dx_known.astype(int)
    out["_dedup_cog_known"] = cog_known.astype(int)
    out["_dedup_apoe_known"] = apoe_known.astype(int)
    out["_dedup_visit_score"] = visit_score
    out["_dedup_original_order"] = np.arange(len(out))

    keep_valid = (
        out.loc[valid]
        .sort_values(["_dedup_key", "_dedup_dx_known", "_dedup_cog_known", "_dedup_apoe_known", "_dedup_visit_score", "_dedup_original_order"],
                     ascending=[True, False, False, False, False, True])
        .drop_duplicates("_dedup_key", keep="first")
    )
    # Keep rows without keys as-is, then combine.
    no_key = out.loc[~valid]
    final = pd.concat([keep_valid, no_key], ignore_index=False, sort=False).sort_values("_dedup_original_order")
    helper = [c for c in final.columns if c.startswith("_dedup_")]
    final = final.drop(columns=helper, errors="ignore")

    info.update({
        "dedup_applied": "yes",
        "dedup_key_col": key_col,
        "n_rows_after_dedup": len(final),
        "n_duplicate_connectome_rows_removed": len(out) - len(final),
    })
    return final.reset_index(drop=True), info


# =============================================================================
# ADNI MERGES AND STATUS
# =============================================================================

def normalize_adni_visit(x: object) -> object:
    if pd.isna(x):
        return pd.NA
    s = str(x).strip().lower()
    mapping = {
        "sc": "bl", "screening": "bl", "baseline": "bl", "bl": "bl", "init": "bl",
        "m0": "bl", "m00": "bl", "0": "bl", "0.0": "bl", "y0": "bl", "y0.0": "bl",
        "m06": "m06", "m6": "m06", "y0.5": "m06", "6": "m06", "6.0": "m06",
        "m12": "m12", "y1": "m12", "1": "m12", "1.0": "m12", "12": "m12", "12.0": "m12",
        "m24": "m24", "y2": "m24", "2": "m24", "2.0": "m24", "24": "m24", "24.0": "m24",
        "m36": "m36", "y3": "m36", "3": "m36", "3.0": "m36", "36": "m36", "36.0": "m36",
        "m48": "m48", "y4": "m48", "4": "m48", "4.0": "m48", "48": "m48", "48.0": "m48",
        "m60": "m60", "y5": "m60", "5": "m60", "5.0": "m60", "60": "m60", "60.0": "m60",
        "m72": "m72", "y6": "m72",
    }
    return mapping.get(s, s)


def merge_adni_covars(meta: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    cov_path = first_existing(ADNI_COVARS_CANDIDATES)
    if cov_path is None:
        return meta.copy(), {"adni_covars_status": "missing", "adni_covars_file": "", "adni_covars_rows": 0, "adni_covars_matched_rows": 0}

    covars = read_table(cov_path)
    meta_subject_col = first_col(meta, ["subject_id", "PTID", "RID", "Subject", "subject"])
    cov_subject_col = first_col(covars, ["subject_id", "PTID", "RID", "Subject", "subject"])
    group_col = first_col(covars, ["GROUP", "Research Group", "DX_bl", "DX", "diagnosis"])
    if meta_subject_col is None or cov_subject_col is None or group_col is None:
        return meta.copy(), {"adni_covars_status": "missing_keys_or_group", "adni_covars_file": str(cov_path), "adni_covars_rows": len(covars), "adni_covars_matched_rows": 0}

    m = meta.copy()
    c = covars.copy()
    m["_cov_subj"] = m[meta_subject_col].map(clean_id)
    c["_cov_subj"] = c[cov_subject_col].map(clean_id)
    keep = ["_cov_subj", group_col]
    for opt in ["SITE", "AGE", "SEX"]:
        hit = first_col(c, [opt])
        if hit is not None and hit not in keep:
            keep.append(hit)
    small = c[keep].dropna(subset=["_cov_subj"]).drop_duplicates("_cov_subj", keep="first")
    rename = {group_col: "GROUP_from_covars"}
    for col in keep:
        if col not in ["_cov_subj", group_col]:
            rename[col] = f"{col}_from_covars"
    small = small.rename(columns=rename)
    merged = m.merge(small, on="_cov_subj", how="left", validate="m:1")
    existing_group = first_col(merged, ["GROUP"])
    if existing_group is not None and existing_group != "GROUP_from_covars":
        merged["GROUP_original"] = merged[existing_group]
        merged["GROUP"] = merged[existing_group].where(merged[existing_group].notna(), merged["GROUP_from_covars"])
    else:
        merged["GROUP"] = merged["GROUP_from_covars"]
    matched = int(merged["GROUP_from_covars"].notna().sum())
    merged = merged.drop(columns=["_cov_subj"])
    return merged, {
        "adni_covars_status": "ok",
        "adni_covars_file": str(cov_path),
        "adni_covars_rows": len(covars),
        "adni_covars_matched_rows": matched,
        "adni_covars_merge_key": f"{meta_subject_col}+{cov_subject_col}",
    }


def merge_adni_dxsum(meta: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    dx_path = first_existing(ADNI_DXSUM_CANDIDATES)
    if dx_path is None:
        return meta.copy(), {"adni_dxsum_status": "missing", "adni_dxsum_file": "", "adni_dxsum_rows": 0, "adni_dxsum_matched_rows": 0}
    dxsum = read_table(dx_path)
    meta_rid_col = first_col(meta, ["RID", "rid"])
    meta_visit_col = first_col(meta, ["VISCODE2", "VISCODE", "DWI_visit_label", "visit", "Visit"])
    dx_rid_col = first_col(dxsum, ["RID", "rid"])
    dx_diag_col = first_col(dxsum, ["DIAGNOSIS"])
    if meta_rid_col is None or meta_visit_col is None:
        return meta.copy(), {"adni_dxsum_status": "missing_metadata_keys", "adni_dxsum_file": str(dx_path), "adni_dxsum_rows": len(dxsum), "adni_dxsum_matched_rows": 0}
    if dx_rid_col is None or dx_diag_col is None:
        return meta.copy(), {"adni_dxsum_status": "missing_dxsum_keys", "adni_dxsum_file": str(dx_path), "adni_dxsum_rows": len(dxsum), "adni_dxsum_matched_rows": 0}

    m = meta.copy()
    m["_dx_rid"] = pd.to_numeric(m[meta_rid_col], errors="coerce").astype("Int64")
    m["_dx_visit"] = m[meta_visit_col].map(normalize_adni_visit)

    frames = []
    for visit_candidate in ["VISCODE2", "VISCODE"]:
        dx_visit_col = first_col(dxsum, [visit_candidate])
        if dx_visit_col is None:
            continue
        d = dxsum.copy()
        d["_dx_rid"] = pd.to_numeric(d[dx_rid_col], errors="coerce").astype("Int64")
        d["_dx_visit"] = d[dx_visit_col].map(normalize_adni_visit)
        keep = ["_dx_rid", "_dx_visit", dx_diag_col]
        for opt in ["EXAMDATE", "VISCODE", "VISCODE2", "DXNORM", "DXMCI", "DXAD"]:
            hit = first_col(d, [opt])
            if hit is not None and hit not in keep:
                keep.append(hit)
        small = d[keep].dropna(subset=["_dx_rid", "_dx_visit"]).copy()
        small["_dx_visit_source"] = visit_candidate
        frames.append(small)

    if not frames:
        return meta.copy(), {"adni_dxsum_status": "missing_dxsum_visit", "adni_dxsum_file": str(dx_path), "adni_dxsum_rows": len(dxsum), "adni_dxsum_matched_rows": 0}

    dx_all = pd.concat(frames, ignore_index=True, sort=False)
    dx_all = dx_all.drop_duplicates(["_dx_rid", "_dx_visit"], keep="last")
    dx_all = dx_all.rename(columns={dx_diag_col: "dxsum_DIAGNOSIS_merged"})
    merged = m.merge(dx_all, on=["_dx_rid", "_dx_visit"], how="left", validate="m:1")
    matched = int(merged["dxsum_DIAGNOSIS_merged"].notna().sum())
    merged = merged.drop(columns=["_dx_rid", "_dx_visit"])
    return merged, {
        "adni_dxsum_status": "ok",
        "adni_dxsum_file": str(dx_path),
        "adni_dxsum_rows": len(dxsum),
        "adni_dxsum_matched_rows": matched,
        "adni_dxsum_merge_key": f"{meta_rid_col}+{meta_visit_col}",
    }


def filter_adni_y0_y4(meta: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    if not FILTER_ADNI_TO_Y0_Y4_SUBJECTS:
        return meta.copy(), {"adni_y0_y4_filter": "not_applied", "adni_y0_y4_subjects": 0, "adni_rows_after_y0_y4_filter": len(meta)}
    y_path = first_existing(ADNI_Y0_Y4_CANDIDATES)
    if y_path is None:
        return meta.copy(), {"adni_y0_y4_filter": "missing_file", "adni_y0_y4_subjects": 0, "adni_rows_after_y0_y4_filter": len(meta)}
    y = read_table(y_path)
    subj_col = first_col(y, ["DWI_subject", "DWI_subject_key", "subject_id", "RID"])
    flag_col = first_col(y, ["has_y0_and_y4"])
    meta_subj_col = first_col(meta, ["DWI_subject_key", "DWI_subject", "subject_id", "RID", "PTID"])
    if subj_col is None or flag_col is None or meta_subj_col is None:
        return meta.copy(), {"adni_y0_y4_filter": "missing_columns", "adni_y0_y4_subjects": 0, "adni_rows_after_y0_y4_filter": len(meta)}
    keep_subjects = set(y.loc[coerce_bool_series(y[flag_col]), subj_col].astype(str).str.strip())
    out = meta.loc[meta[meta_subj_col].astype(str).str.strip().isin(keep_subjects)].copy()
    return out, {"adni_y0_y4_filter": "applied", "adni_y0_y4_subjects": len(keep_subjects), "adni_rows_after_y0_y4_filter": len(out)}


def harmonize_adni(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    df, covars_info = merge_adni_covars(df)
    df, dxsum_info = merge_adni_dxsum(df)
    df, y0y4_info = filter_adni_y0_y4(df)
    normcog, mci, ad, label = init_status(df)

    # Row-wise priority: merged DXSUM -> existing dxsum_DIAGNOSIS -> generic DIAGNOSIS -> GROUP.
    diag = pd.Series(np.nan, index=df.index, dtype="float")
    diag_source_parts = []
    for colname in ["dxsum_DIAGNOSIS_merged", "dxsum_DIAGNOSIS", "DXSUM_DIAGNOSIS", "DIAGNOSIS"]:
        col = first_col(df, [colname])
        if col is not None:
            x = pd.to_numeric(df[col], errors="coerce")
            before = diag.notna().sum()
            diag = diag.combine_first(x)
            if diag.notna().sum() > before:
                diag_source_parts.append(col)

    if diag.notna().any():
        is_normal = diag.eq(1)
        is_mci = diag.eq(2)
        is_ad = diag.eq(3)
        normcog.loc[is_normal] = 1
        normcog.loc[is_mci | is_ad] = 0
        mci.loc[is_mci] = 1
        mci.loc[is_normal | is_ad] = 0
        ad.loc[is_ad] = 1
        ad.loc[is_normal | is_mci] = 0

    # Fallback GROUP only where DXSUM-style diagnosis did not classify.
    missing = normcog.isna() & mci.isna() & ad.isna()
    group_col = first_col(df, ["GROUP", "GROUP_from_covars", "Research Group", "DX_bl", "DX", "diagnosis"])
    if group_col is not None and missing.any():
        s = df.loc[missing, group_col].astype(str).str.strip().str.upper()
        is_normal = s.isin(["CN", "NC", "NL", "NORMAL", "CONTROL", "CONTROLS"])
        if COUNT_ADNI_SMC_AS_NORMAL:
            is_normal = is_normal | s.eq("SMC")
        is_mci = s.str.contains(r"\bMCI\b|EMCI|LMCI", regex=True, na=False)
        is_ad = s.str.contains(r"\bAD\b|DEMENT|ALZHEIMER", regex=True, na=False)
        idx = s.index
        normcog.loc[idx[is_normal]] = 1
        normcog.loc[idx[is_mci | is_ad]] = 0
        mci.loc[idx[is_mci]] = 1
        mci.loc[idx[is_normal | is_ad]] = 0
        ad.loc[idx[is_ad]] = 1
        ad.loc[idx[is_normal | is_mci]] = 0
        diag_source_parts.append(f"GROUP fallback:{group_col}")

    source = "ADNI diagnosis priority: " + ";".join(diag_source_parts or ["missing DXSUM/GROUP"])
    out = finalize_status(df, normcog, mci, ad, label, source)
    out.attrs["extra_summary"] = {**covars_info, **dxsum_info, **y0y4_info}
    return out, source


def harmonize_ad_decode(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    df = df.copy()
    normcog, mci, ad, label = init_status(df)
    risk_col = first_col(df, ["risk_for_ad", "Risk_y", "risk", "Risk_num", "risk_for_AD"])
    if risk_col is not None:
        x = pd.to_numeric(df[risk_col], errors="coerce")
        is_normal = x.isin([0, 1])
        is_mci = x.eq(2)
        is_ad = x.eq(3)
        source = f"{risk_col}: 0/1 Normal, 2 MCI, 3 AD"
    else:
        risk_str = first_col(df, ["Risk", "risk_group", "group", "Group"])
        if risk_str is None:
            source = "missing AD_DECODE risk_for_ad/Risk"
            return finalize_status(df, normcog, mci, ad, label, source), source
        s = df[risk_str].astype(str).str.strip().str.upper()
        is_normal = s.isin(["NORMAL", "CONTROL", "CONTROLS", "CN", "NC", "FAMILIAL", "RISK", "AT RISK"])
        is_mci = s.str.contains(r"\bMCI\b", regex=True, na=False)
        is_ad = s.str.contains(r"\bAD\b|ALZHEIMER|DEMENT", regex=True, na=False)
        source = risk_str
    normcog.loc[is_normal] = 1
    normcog.loc[is_mci | is_ad] = 0
    mci.loc[is_mci] = 1
    mci.loc[is_normal | is_ad] = 0
    ad.loc[is_ad] = 1
    ad.loc[is_normal | is_mci] = 0
    return finalize_status(df, normcog, mci, ad, label, source), source


def harmonize_adrc(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    df = df.copy()
    normcog, mci, ad, label = init_status(df)
    sources = []
    norm_col = first_col(df, ["NORMCOG_01", "NORMCOG", "normcog"])
    mci_col = first_col(df, ["MCI_01", "IMPNOMCI", "impnomci", "MCI"])
    ad_col = first_col(df, ["AD_01", "DEMENTIA_01", "DEMENTED", "demented", "AD"])
    if norm_col is not None:
        x = pd.to_numeric(df[norm_col], errors="coerce")
        normcog.loc[x.eq(1)] = 1
        normcog.loc[x.eq(0)] = 0
        mci.loc[x.eq(1)] = 0
        ad.loc[x.eq(1)] = 0
        sources.append(norm_col)
    if mci_col is not None:
        x = pd.to_numeric(df[mci_col], errors="coerce")
        mci.loc[x.eq(1)] = 1
        mci.loc[x.eq(0) & mci.isna()] = 0
        normcog.loc[x.eq(1)] = 0
        ad.loc[x.eq(1)] = 0
        sources.append(mci_col)
    if ad_col is not None:
        x = pd.to_numeric(df[ad_col], errors="coerce")
        ad.loc[x.eq(1)] = 1
        ad.loc[x.eq(0) & ad.isna()] = 0
        normcog.loc[x.eq(1)] = 0
        mci.loc[x.eq(1)] = 0
        sources.append(ad_col)
    source = ";".join(sources) if sources else "missing ADRC NORMCOG/MCI/AD columns"
    return finalize_status(df, normcog, mci, ad, label, source), source


def harmonize_habs(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    df = df.copy()
    normcog, mci, ad, label = init_status(df)
    norm_col = first_col(df, ["NORMCOG_01", "NORMCOG", "normcog"])
    mci_col = first_col(df, ["MCI_01", "MCI", "mci"])
    ad_col = first_col(df, ["AD_01", "DEMENTIA_01", "DEMENTED", "demented", "AD"])
    if norm_col is not None or mci_col is not None or ad_col is not None:
        sources = []
        if norm_col is not None:
            x = pd.to_numeric(df[norm_col], errors="coerce")
            normcog.loc[x.eq(1)] = 1
            normcog.loc[x.eq(0)] = 0
            sources.append(norm_col)
        if mci_col is not None:
            x = pd.to_numeric(df[mci_col], errors="coerce")
            mci.loc[x.eq(1)] = 1
            mci.loc[x.eq(0)] = 0
            normcog.loc[x.eq(1)] = 0
            sources.append(mci_col)
        if ad_col is not None:
            x = pd.to_numeric(df[ad_col], errors="coerce")
            ad.loc[x.eq(1)] = 1
            ad.loc[x.eq(0)] = 0
            normcog.loc[x.eq(1)] = 0
            mci.loc[x.eq(1)] = 0
            sources.append(ad_col)
        source = ";".join(sources)
        return finalize_status(df, normcog, mci, ad, label, source), source
    cdx_col = first_col(df, ["CDX_Cog", "CDX_COG", "cdx_cog"])
    if cdx_col is not None:
        x = pd.to_numeric(df[cdx_col], errors="coerce")
        is_normal = x.eq(0)
        is_mci = x.eq(1)
        is_ad = x.eq(2)
        normcog.loc[is_normal] = 1
        normcog.loc[is_mci | is_ad] = 0
        mci.loc[is_mci] = 1
        mci.loc[is_normal | is_ad] = 0
        ad.loc[is_ad] = 1
        ad.loc[is_normal | is_mci] = 0
        source = f"{cdx_col}: 0 Normal, 1 MCI, 2 AD"
        return finalize_status(df, normcog, mci, ad, label, source), source
    source = "missing HABS CDX_Cog/NORMCOG/MCI/AD"
    return finalize_status(df, normcog, mci, ad, label, source), source


def harmonize_status(df: pd.DataFrame, cohort: str) -> tuple[pd.DataFrame, str]:
    if cohort == "ADNI":
        return harmonize_adni(df)
    if cohort == "ADRC":
        return harmonize_adrc(df)
    if cohort == "HABS":
        return harmonize_habs(df)
    if cohort == "AD_DECODE":
        return harmonize_ad_decode(df)
    raise ValueError(f"Unknown cohort: {cohort}")


# =============================================================================
# APOE HARMONIZATION
# =============================================================================

def extract_rid(x: object) -> object:
    if pd.isna(x):
        return pd.NA
    s = str(x).strip()
    if not s or s.lower() == "nan":
        return pd.NA
    try:
        f = float(s)
        if np.isfinite(f) and f.is_integer():
            return str(int(f))
    except Exception:
        pass
    m = re.search(r"R(\d+)", s, flags=re.I)
    if m:
        return str(int(m.group(1)))
    m = re.search(r"_S_(\d+)", s, flags=re.I)
    if m:
        return str(int(m.group(1)))
    nums = re.findall(r"\d+", s)
    if nums:
        return str(int(nums[-1]))
    return pd.NA


def clean_apoe_allele(x: object) -> object:
    if pd.isna(x):
        return np.nan
    s = str(x).strip().upper()
    if not s or s in {"NAN", "NONE", "<NA>", "-9999", "-777777", "999"}:
        return np.nan
    try:
        f = float(s)
        if f in {2.0, 3.0, 4.0}:
            return int(f)
    except Exception:
        pass
    m = re.search(r"([234])", s)
    if m:
        return int(m.group(1))
    return np.nan


def normalize_apoe_genotype(x: object) -> object:
    if pd.isna(x):
        return pd.NA
    s = str(x).strip().upper().replace(" ", "").replace("-", "_")
    if not s or s in {"NAN", "NONE", "<NA>", "-9999", "-777777", "999"}:
        return pd.NA
    mapping = {
        "E2E2": "2_2", "E2E3": "2_3", "E3E2": "2_3", "E2E4": "2_4", "E4E2": "2_4",
        "E3E3": "3_3", "E3E4": "3_4", "E4E3": "3_4", "E4E4": "4_4",
        "APOE22": "2_2", "APOE23": "2_3", "APOE32": "2_3", "APOE24": "2_4", "APOE42": "2_4",
        "APOE33": "3_3", "APOE34": "3_4", "APOE43": "3_4", "APOE44": "4_4",
        "22": "2_2", "23": "2_3", "32": "2_3", "24": "2_4", "42": "2_4",
        "33": "3_3", "34": "3_4", "43": "3_4", "44": "4_4",
        "2/2": "2_2", "2/3": "2_3", "3/2": "2_3", "2/4": "2_4", "4/2": "2_4",
        "3/3": "3_3", "3/4": "3_4", "4/3": "3_4", "4/4": "4_4",
        "2_2": "2_2", "2_3": "2_3", "3_2": "2_3", "2_4": "2_4", "4_2": "2_4",
        "3_3": "3_3", "3_4": "3_4", "4_3": "3_4", "4_4": "4_4",
    }
    if s in mapping:
        return mapping[s]
    alleles = re.findall(r"(?:E|APOE)?([234])", s)
    if len(alleles) >= 2:
        a, b = sorted([alleles[0], alleles[1]])
        return f"{a}_{b}"
    return pd.NA


def genotype_from_alleles(a1: object, a2: object) -> object:
    a = clean_apoe_allele(a1)
    b = clean_apoe_allele(a2)
    if pd.isna(a) or pd.isna(b):
        return pd.NA
    aa, bb = sorted([int(a), int(b)])
    return f"{aa}_{bb}"


def apoe_counts_from_genotype(g: object) -> tuple[object, object, object, object, object, object]:
    if pd.isna(g):
        return pd.NA, pd.NA, pd.NA, pd.NA, pd.NA, pd.NA
    parts = str(g).split("_")
    if len(parts) != 2:
        return pd.NA, pd.NA, pd.NA, pd.NA, pd.NA, pd.NA
    try:
        alleles = [int(parts[0]), int(parts[1])]
    except Exception:
        return pd.NA, pd.NA, pd.NA, pd.NA, pd.NA, pd.NA
    e2 = alleles.count(2)
    e3 = alleles.count(3)
    e4 = alleles.count(4)
    any4 = 1 if e4 > 0 else 0
    strict = 1 if str(g) in {"3_4", "4_4"} else (0 if str(g) in {"2_2", "2_3", "3_3"} else pd.NA)
    return e2, e3, e4, any4, e4, strict


def detect_apoe_columns(df: pd.DataFrame) -> tuple[Optional[str], Optional[str], Optional[str]]:
    a1 = first_col(df, ["APOE_A1", "APOEA1", "APOE_A_1", "APOE allele 1", "APOE_ALLELE1", "APGEN1", "APGENE1", "APOEGEN1", "APOE_GEN1", "APOE1", "A1", "ALLELE1"])
    a2 = first_col(df, ["APOE_A2", "APOEA2", "APOE_A_2", "APOE allele 2", "APOE_ALLELE2", "APGEN2", "APGENE2", "APOEGEN2", "APOE_GEN2", "APOE2", "A2", "ALLELE2"])
    geno = first_col(df, ["APOE_genotype_harmonized", "genotype", "GENOTYPE", "APOE", "APOERES", "APOE_RESULT", "APOE_GENOTYPE", "APOE Genotype", "APOE4_Genotype", "APOE_GTYPE"])
    return a1, a2, geno


def fill_apoe_output(out: pd.DataFrame, genotype: pd.Series, source: str, source_mask: Optional[pd.Series] = None) -> pd.DataFrame:
    g = genotype.map(normalize_apoe_genotype)
    if source_mask is not None:
        g = g.where(source_mask)
    if "APOE_genotype_harmonized" not in out.columns:
        out["APOE_genotype_harmonized"] = pd.NA
    out["APOE_genotype_harmonized"] = out["APOE_genotype_harmonized"].combine_first(g)

    counts = out["APOE_genotype_harmonized"].map(apoe_counts_from_genotype)
    vals = pd.DataFrame(counts.tolist(), index=out.index, columns=["APOE_e2_count", "APOE_e3_count", "APOE_e4_count", "APOE4_carriage", "APOE4_dosage", "APOE4_strict_34_44_carrier"])
    for c in vals.columns:
        if c not in out.columns:
            out[c] = vals[c]
        else:
            out[c] = out[c].combine_first(vals[c])
    if "APOE_STATUS_SOURCE" not in out.columns:
        out["APOE_STATUS_SOURCE"] = pd.NA
    mask = g.notna() if source_mask is None else g.notna() & source_mask
    out.loc[mask, "APOE_STATUS_SOURCE"] = source
    return out


def add_apoe_harmonized_from_existing(df: pd.DataFrame, source_prefix: str) -> tuple[pd.DataFrame, dict[str, object]]:
    out = df.copy()
    a1, a2, geno = detect_apoe_columns(out)
    if a1 is not None and a2 is not None:
        genotype = pd.Series([genotype_from_alleles(a, b) for a, b in zip(out[a1], out[a2])], index=out.index)
        source = f"{source_prefix}: {a1}+{a2}"
    elif geno is not None:
        genotype = out[geno]
        source = f"{source_prefix}: {geno}"
    else:
        genotype = pd.Series(pd.NA, index=out.index)
        source = f"{source_prefix}: missing"
    out = fill_apoe_output(out, genotype, source)
    return out, {
        "apoe_existing_source": source,
        "n_APOE_genotype_harmonized": int(out["APOE_genotype_harmonized"].notna().sum()),
        "n_APOE4_carriage": int(pd.to_numeric(out["APOE4_carriage"], errors="coerce").eq(1).sum()),
    }


def choose_adni_apoe_merge_key(meta: pd.DataFrame, apoe: pd.DataFrame) -> tuple[pd.Series, pd.Series, str]:
    meta_rid = first_col(meta, ["RID", "rid"])
    apoe_rid = first_col(apoe, ["RID", "rid"])
    meta_ptid = first_col(meta, ["PTID", "ptid", "subject_id", "Subject ID"])
    apoe_ptid = first_col(apoe, ["PTID", "ptid", "subject_id", "Subject ID"])
    if meta_rid is not None and apoe_rid is not None:
        return meta[meta_rid].map(extract_rid), apoe[apoe_rid].map(extract_rid), f"{meta_rid}<->{apoe_rid}"
    if meta_ptid is not None and apoe_ptid is not None:
        return meta[meta_ptid].map(extract_rid), apoe[apoe_ptid].map(extract_rid), f"{meta_ptid}<->{apoe_ptid}"
    if meta_ptid is not None and apoe_rid is not None:
        return meta[meta_ptid].map(extract_rid), apoe[apoe_rid].map(extract_rid), f"{meta_ptid}<->{apoe_rid}"
    if meta_rid is not None and apoe_ptid is not None:
        return meta[meta_rid].map(extract_rid), apoe[apoe_ptid].map(extract_rid), f"{meta_rid}<->{apoe_ptid}"
    raise ValueError("Could not find ADNI APOE merge key")


def merge_adni_apoe(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    out, info = add_apoe_harmonized_from_existing(df, "ADNI existing metadata")
    apoe_path = first_existing(ADNI_APOE_CANDIDATES)
    if apoe_path is None:
        info.update({"adni_apoe_merge_status": "missing_file", "adni_apoe_file": "", "adni_apoe_rows": 0, "adni_apoe_matched_rows": 0})
        return out, info
    apoe = read_table(apoe_path)
    a1, a2, geno = detect_apoe_columns(apoe)
    if (a1 is None or a2 is None) and geno is None:
        info.update({"adni_apoe_merge_status": "missing_apoe_columns", "adni_apoe_file": str(apoe_path), "adni_apoe_rows": len(apoe), "adni_apoe_matched_rows": 0})
        return out, info
    apoe = apoe.copy()
    if a1 is not None and a2 is not None:
        apoe["_APOE_genotype_norm"] = [genotype_from_alleles(a, b) for a, b in zip(apoe[a1], apoe[a2])]
    else:
        apoe["_APOE_genotype_norm"] = pd.NA
    if geno is not None:
        apoe["_APOE_genotype_norm"] = apoe["_APOE_genotype_norm"].combine_first(apoe[geno].map(normalize_apoe_genotype))
    try:
        meta_key, apoe_key, key_label = choose_adni_apoe_merge_key(out, apoe)
    except ValueError:
        info.update({"adni_apoe_merge_status": "missing_merge_key", "adni_apoe_file": str(apoe_path), "adni_apoe_rows": len(apoe), "adni_apoe_matched_rows": 0})
        return out, info
    out = out.copy()
    out["_APOE_merge_key"] = meta_key
    apoe["_APOE_merge_key"] = apoe_key
    small = apoe[["_APOE_merge_key", "_APOE_genotype_norm"]].dropna(subset=["_APOE_merge_key"]).drop_duplicates("_APOE_merge_key", keep="first")
    merged = out.merge(small, on="_APOE_merge_key", how="left")
    merged = fill_apoe_output(merged, merged["_APOE_genotype_norm"], f"ADNI APOERES: {apoe_path}", source_mask=merged["_APOE_genotype_norm"].notna())
    matched = int(merged["_APOE_genotype_norm"].notna().sum())
    merged["APOE_merge_key"] = merged["_APOE_merge_key"]
    merged = merged.drop(columns=["_APOE_merge_key", "_APOE_genotype_norm"])
    info.update({
        "adni_apoe_merge_status": "ok",
        "adni_apoe_file": str(apoe_path),
        "adni_apoe_rows": len(apoe),
        "adni_apoe_matched_rows": matched,
        "adni_apoe_merge_key": key_label,
        "n_APOE_genotype_harmonized": int(merged["APOE_genotype_harmonized"].notna().sum()),
        "n_APOE4_carriage": int(pd.to_numeric(merged["APOE4_carriage"], errors="coerce").eq(1).sum()),
    })
    return merged, info


def merge_habs_genomics_apoe(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    out, info = add_apoe_harmonized_from_existing(df, "HABS existing metadata")
    genomics_path = first_existing(HABS_GENOMICS_CANDIDATES)
    if genomics_path is None:
        info.update({"habs_apoe_status": "missing_genomics_file", "habs_apoe_file": "", "habs_apoe_rows": 0, "habs_apoe_matched_rows": 0})
        return out, info
    genomics = read_table(genomics_path)
    meta_med_col = first_col(out, ["Med_ID", "MED_ID", "med_id"])
    gen_med_col = first_col(genomics, ["Med_ID", "MED_ID", "med_id"])
    geno_col = first_col(genomics, ["APOE4_Genotype", "APOE_Genotype", "genotype", "GENOTYPE"])
    if meta_med_col is None or gen_med_col is None or geno_col is None:
        info.update({"habs_apoe_status": "missing_med_id_or_genotype", "habs_apoe_file": str(genomics_path), "habs_apoe_rows": len(genomics), "habs_apoe_matched_rows": 0})
        return out, info
    g = genomics.copy()
    g["_habs_med_id"] = g[gen_med_col].astype(str).str.strip()
    g["_HABS_genotype_norm"] = g[geno_col].map(normalize_apoe_genotype)
    keep_cols = ["_habs_med_id", "_HABS_genotype_norm"]
    for opt in ["APOE4_Positivity", "APOE4_rs429358", "APOE4_rs7412", "Visit_ID", "Age", "ID_Education", "ID_Gender"]:
        hit = first_col(g, [opt])
        if hit is not None and hit not in keep_cols:
            keep_cols.append(hit)
    small = g[keep_cols].dropna(subset=["_habs_med_id"]).drop_duplicates("_habs_med_id", keep="first")
    out = out.copy()
    out["_habs_med_id"] = out[meta_med_col].astype(str).str.strip()
    merged = out.merge(small, on="_habs_med_id", how="left", suffixes=("", "_from_habs_genomics"))
    matched = int(merged["_HABS_genotype_norm"].notna().sum())
    merged = fill_apoe_output(merged, merged["_HABS_genotype_norm"], f"HABS genomics: {genomics_path}", source_mask=merged["_HABS_genotype_norm"].notna())
    # Keep useful HABS genomics raw columns if present.
    for raw in ["APOE4_Positivity", "APOE4_rs429358", "APOE4_rs7412"]:
        src = f"{raw}_from_habs_genomics"
        if src in merged.columns:
            if raw not in merged.columns:
                merged[raw] = merged[src]
            else:
                merged[raw] = merged[raw].combine_first(merged[src])
    drop_cols = ["_habs_med_id", "_HABS_genotype_norm"] + [c for c in merged.columns if c.endswith("_from_habs_genomics")]
    merged = merged.drop(columns=[c for c in drop_cols if c in merged.columns])
    info.update({
        "habs_apoe_status": "ok",
        "habs_apoe_file": str(genomics_path),
        "habs_apoe_rows": len(genomics),
        "habs_apoe_matched_rows": matched,
        "habs_apoe_merge_key": meta_med_col,
        "n_APOE_genotype_harmonized": int(merged["APOE_genotype_harmonized"].notna().sum()),
        "n_APOE4_carriage": int(pd.to_numeric(merged["APOE4_carriage"], errors="coerce").eq(1).sum()),
    })
    return merged, info


def add_apoe_harmonized(df: pd.DataFrame, cohort: str) -> tuple[pd.DataFrame, dict[str, object]]:
    if cohort == "ADNI":
        return merge_adni_apoe(df)
    if cohort == "HABS":
        return merge_habs_genomics_apoe(df)
    return add_apoe_harmonized_from_existing(df, f"{cohort} existing metadata")


# =============================================================================
# COGNITIVE COMPOSITE HARMONIZATION
# =============================================================================

DOMAINS = [
    "Memory_Composite",
    "Executive_Function_Composite",
    "Processing_Speed_Composite",
    "Language_Composite",
    "Visuospatial_Composite",
]
GLOBAL_COG = "Global_Cognition_Composite"

DOMAIN_COMPONENTS: dict[str, dict[str, list[tuple[str, int]]]] = {
    "AD_DECODE": {
        "Memory_Composite": [("Memory_Composite", +1)],
        "Executive_Function_Composite": [("Executive_Function_Composite", +1)],
        "Processing_Speed_Composite": [("Processing_Speed_Composite", +1)],
        "Language_Composite": [("Language_Composite", +1)],
        "Visuospatial_Composite": [("Visuospatial_Composite", +1)],
    },
    "ADNI": {
        "Memory_Composite": [("Memory_Composite", +1), ("memory_composite", +1), ("MOCA_delayed_recall_z", +1)],
        "Executive_Function_Composite": [("Executive_Function_Composite", +1), ("executive_function_composite", +1), ("MOCA_attention_z", +1), ("MOCA_abstraction_z", +1)],
        "Processing_Speed_Composite": [("Processing_Speed_Composite", +1), ("processing_speed_composite", +1)],
        "Language_Composite": [("Language_Composite", +1), ("language_composite", +1), ("MOCA_language_z", +1), ("MOCA_naming_z", +1)],
        "Visuospatial_Composite": [("Visuospatial_Composite", +1), ("visuospatial_composite", +1), ("MOCA_visuospatial_z", +1)],
    },
    "ADRC": {
        "Memory_Composite": [("Memory_Composite", +1), ("UDSBENTD", +1)],
        "Executive_Function_Composite": [("Executive_Function_Composite", +1), ("TRAILB", -1)],
        "Processing_Speed_Composite": [("Processing_Speed_Composite", +1), ("TRAILA", -1)],
        "Language_Composite": [("Language_Composite", +1), ("ANIMALS", +1), ("UDSVERFC", +1), ("UDSVERLC", +1), ("UDSVERTN", +1)],
        "Visuospatial_Composite": [("Visuospatial_Composite", +1), ("UDSBENTC", +1)],
    },
    "HABS": {
        "Memory_Composite": [("Memory_Composite", +1), ("SEVLT_T1235_ZScore", +1), ("SEVLT_DR_ZScore", +1), ("LM1_AB_ZScore", +1), ("LM2_AB_ZScore", +1)],
        "Executive_Function_Composite": [("Executive_Function_Composite", +1), ("Trails_B_ZScore", -1), ("DS_ZScore", +1)],
        "Processing_Speed_Composite": [("Processing_Speed_Composite", +1), ("Trails_A_ZScore", -1), ("Digit_Symbol_Substitution_ZScore", +1)],
        "Language_Composite": [("Language_Composite", +1), ("FAS_ZScore", +1), ("Animal_ZScore", +1), ("WAT_Correct", +1)],
        "Visuospatial_Composite": [("Visuospatial_Composite", +1), ("visuospatial_composite", +1)],
    },
}


def clean_component(cohort: str, col: str, s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)
    x = x.mask(x.isin([-9999, -999, -99, 9999, 999, 998, 997, 996, 995]))
    cname = normalize_name(col).upper()
    if cohort == "ADRC":
        if cname in {"TRAILA", "TRAILB"}:
            x = x.mask((x <= 0) | (x >= 995))
        elif cname in {"ANIMALS", "UDSVERFC", "UDSVERLC", "UDSVERTN"}:
            x = x.mask((x < 0) | (x >= 88))
        elif cname in {"UDSBENTC", "UDSBENTD"}:
            x = x.mask((x < 0) | (x > 30))
    return x


def add_cognitive_composites(df: pd.DataFrame, cohort: str, min_components: int = MIN_COMPONENTS_PER_DOMAIN) -> tuple[pd.DataFrame, dict[str, object], pd.DataFrame]:
    out = df.copy()
    mapping_rows = []
    for domain in DOMAINS:
        components = DOMAIN_COMPONENTS.get(cohort, {}).get(domain, [])
        zcols: list[str] = []
        used_source_cols: list[str] = []
        for wanted_col, direction in components:
            hit = first_col(out, [wanted_col])
            found = hit is not None
            n_nonmiss = 0
            if found:
                raw = clean_component(cohort, hit, out[hit]) * float(direction)
                z = zscore_series(raw)
                zname = f"__z_{domain}_{normalize_name(hit)}"
                if z.notna().any():
                    out[zname] = z
                    zcols.append(zname)
                    used_source_cols.append(hit)
                    n_nonmiss = int(z.notna().sum())
            mapping_rows.append({
                "cohort": cohort,
                "domain": domain,
                "component_column_requested": wanted_col,
                "component_column_found": hit or "",
                "direction": direction,
                "found": found,
                "used": found and n_nonmiss > 0,
                "n_nonmissing_after_cleaning": n_nonmiss,
            })
        if zcols:
            n_available = out[zcols].notna().sum(axis=1)
            comp = out[zcols].mean(axis=1, skipna=True).where(n_available >= min_components)
            out[domain] = zscore_series(comp)
            out[f"{domain}_n_components"] = n_available
            out[f"{domain}_source_columns"] = ";".join(sorted(set(used_source_cols)))
        else:
            out[domain] = np.nan
            out[f"{domain}_n_components"] = 0
            out[f"{domain}_source_columns"] = ""

    domain_mat = out[DOMAINS].apply(pd.to_numeric, errors="coerce")
    n_domains = domain_mat.notna().sum(axis=1)
    out[GLOBAL_COG] = zscore_series(domain_mat.mean(axis=1, skipna=True).where(n_domains >= 1))
    out[f"{GLOBAL_COG}_n_domains"] = n_domains
    out["cognition_composite"] = out[GLOBAL_COG]
    out["cognitive_impairment_composite"] = -out[GLOBAL_COG]
    hidden = [c for c in out.columns if c.startswith("__z_")]
    out = out.drop(columns=hidden)
    info = {"n_Global_Cognition_Composite": int(out[GLOBAL_COG].notna().sum())}
    for domain in DOMAINS:
        info[f"n_{domain}"] = int(out[domain].notna().sum())
    return out, info, pd.DataFrame(mapping_rows)


# =============================================================================
# AUDIT HELPERS
# =============================================================================

def audit_key_columns(df: pd.DataFrame) -> dict[str, object]:
    expected = [
        "DWI", "DWI_key", "DWI_subject_key", "DWI_visit_label", "DWI_visit_year",
        "connectome_key", "connectome_full_key", "has_DWI_connectome", "DWI_connectome_path",
        "cognition_composite", "ATN_composite", "cardio_composite",
    ]
    present = [c for c in expected if c in df.columns]
    missing = [c for c in expected if c not in df.columns]
    return {
        "audit_expected_columns_present": ";".join(present),
        "audit_expected_columns_missing": ";".join(missing),
    }


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    print("=" * 100)
    print("CREATING FINAL HARMONIZED METADATA FILES")
    print("=" * 100)
    print("WORK:", WORK)
    print("Base:", BASE)
    print("Output:", OUTDIR)
    print("Filter to connectome rows:", FILTER_TO_CONNECTOME_ROWS)
    print("Deduplicate one row per connectome:", DEDUPLICATE_TO_ONE_ROW_PER_CONNECTOME)
    print("ADNI plain connectome candidate:", BASE / "ADNI" / "connectomes" / "DWI" / "plain")
    print("ADNI DXSUM candidates:")
    for p in ADNI_DXSUM_CANDIDATES:
        print("  ", p, "[exists]" if p.exists() else "[missing]")
    print("HABS genomics candidates:")
    for p in HABS_GENOMICS_CANDIDATES:
        print("  ", p, "[exists]" if p.exists() else "[missing]")
    print("ADNI SMC counted as normal:", COUNT_ADNI_SMC_AS_NORMAL)
    print("Filter ADNI to y0/y4 subjects:", FILTER_ADNI_TO_Y0_Y4_SUBJECTS)

    summary_rows = []
    for cohort, candidates in METADATA_CANDIDATES.items():
        path = first_existing(candidates)
        if path is None:
            print(f"[MISSING] {cohort}: no candidate metadata file found")
            summary_rows.append({
                "cohort": cohort, "input_file": "", "output_file": "", "status": "missing_input", "source": "",
                "n_rows_input": 0, "n_rows_after_connectome_filter": 0, "n_rows_output": 0,
                "n_normcog": 0, "n_mci": 0, "n_ad": 0, "n_unknown": 0,
            })
            continue

        print(f"\n[READ] {cohort}: {path}")
        df = read_table(path)
        n_input = len(df)
        audit_before = audit_key_columns(df)

        df_filtered, connectome_info = add_connectome_match_columns(df, cohort)
        if len(df_filtered) == 0 and len(df) > 0:
            print(f"[WARN] {cohort}: connectome filter produced 0 rows; keeping unfiltered metadata to avoid empty output.")
            connectome_info["connectome_filter_zero_row_fallback"] = "kept_unfiltered"
            df_filtered = df.copy()
        else:
            connectome_info["connectome_filter_zero_row_fallback"] = "not_needed"

        out, source = harmonize_status(df_filtered, cohort)
        out, apoe_info = add_apoe_harmonized(out, cohort)
        out, composite_info, composite_mapping = add_cognitive_composites(out, cohort)
        out, dedup_info = deduplicate_connectome_rows(out, cohort)

        outpath = OUTDIR / f"{cohort}_harmonized_metadata.csv"
        out.to_csv(outpath, index=False)
        mapping_path = OUTDIR / f"{cohort}_cognitive_composite_mapping.csv"
        composite_mapping.to_csv(mapping_path, index=False)

        n_norm = int(out["NORMCOG_01"].eq(1).sum())
        n_mci = int(out["MCI_01"].eq(1).sum())
        n_ad = int(out["AD_01"].eq(1).sum())
        n_unknown = int(out["DX_Label_harmonized"].eq("Unknown").sum())
        n_apoe = int(out.get("APOE_genotype_harmonized", pd.Series(pd.NA, index=out.index)).notna().sum())
        n_apoe4 = int(pd.to_numeric(out.get("APOE4_carriage", pd.Series(np.nan, index=out.index)), errors="coerce").eq(1).sum())
        n_global = int(pd.to_numeric(out.get(GLOBAL_COG, pd.Series(np.nan, index=out.index)), errors="coerce").notna().sum())

        row: dict[str, object] = {
            "cohort": cohort,
            "input_file": str(path),
            "output_file": str(outpath),
            "status": "ok",
            "source": source,
            "n_rows_input": n_input,
            "n_rows_after_connectome_filter": len(df_filtered),
            "n_rows_output": len(out),
            "n_normcog": n_norm,
            "n_mci": n_mci,
            "n_ad": n_ad,
            "n_unknown": n_unknown,
            "n_APOE_genotype_harmonized_final": n_apoe,
            "n_APOE4_carriage_final": n_apoe4,
            "n_Global_Cognition_Composite_final": n_global,
            "cognitive_composite_mapping_file": str(mapping_path),
        }
        row.update(audit_before)
        row.update(connectome_info)
        row.update(apoe_info)
        row.update(composite_info)
        row.update(dedup_info)
        row.update(out.attrs.get("extra_summary", {}))
        summary_rows.append(row)

        print(
            f"[DONE] {cohort:<9} input={n_input:<7} output={len(out):<7} "
            f"Normal={n_norm:<5} MCI={n_mci:<5} AD={n_ad:<5} Unknown={n_unknown:<5} "
            f"APOE={n_apoe:<5} APOE4={n_apoe4:<5} Cog={n_global:<5} source={source}"
        )
        print(f"       wrote: {outpath}")

    summary = pd.DataFrame(summary_rows)
    summary_path = OUTDIR / "harmonized_metadata_summary.csv"
    summary.to_csv(summary_path, index=False)

    print("\n" + "=" * 100)
    print("[DONE] Summary:")
    print(summary_path)
    print("=" * 100)
    if not summary.empty:
        cols = [
            "cohort", "status", "n_rows_input", "n_rows_after_connectome_filter", "n_rows_output",
            "n_normcog", "n_mci", "n_ad", "n_unknown",
            "n_APOE_genotype_harmonized_final", "n_APOE4_carriage_final", "n_Global_Cognition_Composite_final",
            "connectome_filter_method", "dedup_applied", "n_duplicate_connectome_rows_removed",
            "source",
        ]
        cols = [c for c in cols if c in summary.columns]
        print(summary[cols].to_string(index=False))


if __name__ == "__main__":
    main()
