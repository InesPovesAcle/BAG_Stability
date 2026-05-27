#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build Table 1 directly from final harmonized metadata files.

This script intentionally does NOT redo diagnosis harmonization, APOE merging,
connectome filtering, ADNI DXSUM merging, HABS genomics merging, or cognitive
composite construction. Those steps should be completed upstream by:

  0_create_harmonized_metadata_FINAL.py

Inputs:
  $WORK/ines/data/harmonization/harmonized_metadata/ADNI_harmonized_metadata.csv
  $WORK/ines/data/harmonization/harmonized_metadata/ADRC_harmonized_metadata.csv
  $WORK/ines/data/harmonization/harmonized_metadata/HABS_harmonized_metadata.csv
  $WORK/ines/data/harmonization/harmonized_metadata/AD_DECODE_harmonized_metadata.csv

Outputs:
  $WORK/ines/results/Table1_FromHarmonizedMetadata/
    Table1_from_harmonized_metadata_formatted.xlsx
    Table1_from_harmonized_metadata_wide.csv
    Table1_from_harmonized_metadata_long.csv
    Table1_subject_level_from_harmonized_metadata.csv
    Table1_scan_level_from_harmonized_metadata.csv
    Table1_harmonized_metadata_source_summary.csv
    Table1_harmonized_metadata_README.md

Run:
  python 3_Table1_from_harmonized_metadata.py
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
HARMONIZED_DIR = BASE_DIR / "data" / "harmonization" / "harmonized_metadata"
OUTDIR = BASE_DIR / "results" / "Table1_FromHarmonizedMetadata"

COHORTS = ["ADNI", "ADRC", "HABS", "AD_DECODE"]

HARMONIZED_METADATA_FILES = {
    "ADNI": HARMONIZED_DIR / "ADNI_harmonized_metadata.csv",
    "ADRC": HARMONIZED_DIR / "ADRC_harmonized_metadata.csv",
    "HABS": HARMONIZED_DIR / "HABS_harmonized_metadata.csv",
    "AD_DECODE": HARMONIZED_DIR / "AD_DECODE_harmonized_metadata.csv",
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

# If True, Table 1 uses the strict APOE4 definition that excludes APOE2/4.
# If False, it uses any APOE4 carriage, including APOE2/4, APOE3/4, APOE4/4.
USE_STRICT_APOE4_FOR_TABLE1 = True

# Whether to include cognitive-composite availability rows in the main table.
INCLUDE_COGNITIVE_COMPOSITE_ROWS = True


# =============================================================================
# HELPERS
# =============================================================================

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


def fmt_count_pct(n: int, denom: int) -> str:
    if denom <= 0:
        return "—"
    return f"{int(n)} ({100 * n / denom:.1f}%)"


def fmt_mean_sd(x: pd.Series) -> str:
    y = pd.to_numeric(x, errors="coerce").dropna()
    if len(y) == 0:
        return "—"
    return f"{y.mean():.2f} ± {y.std(ddof=1):.2f}"


def fmt_median_iqr(x: pd.Series) -> str:
    y = pd.to_numeric(x, errors="coerce").dropna()
    if len(y) == 0:
        return "—"
    q1, q3 = y.quantile([0.25, 0.75])
    return f"{y.median():.2f} [{q1:.2f}, {q3:.2f}]"


def fmt_range(x: pd.Series) -> str:
    y = pd.to_numeric(x, errors="coerce").dropna()
    if len(y) == 0:
        return "—"
    return f"{y.min():.1f}–{y.max():.1f}"


def dash_if_zero_count(n: int, denom: int, use_dash: bool = True) -> str:
    if n == 0 and use_dash:
        return "—"
    return fmt_count_pct(n, denom)


def normalize_sex_value(x: object) -> str:
    if pd.isna(x):
        return "Unknown"
    s = str(x).strip().lower()
    if s in ["m", "male", "1", "1.0"] or s.startswith("m"):
        return "Male"
    if s in ["f", "female", "2", "2.0"] or s.startswith("f"):
        return "Female"
    return "Unknown"


def normalize_apoe_label(x: object) -> object:
    """Return APOE22/APOE23/... from common harmonized genotype formats."""
    if pd.isna(x):
        return np.nan
    s = str(x).strip().upper().replace(" ", "").replace("/", "_").replace("-", "_")
    if s in ["", "NAN", "NONE", "<NA>"]:
        return np.nan

    # Already normalized 2_3 or APOE23/E2E3.
    if re.fullmatch(r"[234]_[234]", s):
        a, b = sorted(s.split("_"))
        return f"APOE{a}{b}"
    if re.fullmatch(r"[234][234]", s):
        a, b = sorted(list(s))
        return f"APOE{a}{b}"

    s2 = s.replace("APOE", "").replace("E", "").replace("_", "")
    if re.fullmatch(r"[234][234]", s2):
        a, b = sorted(list(s2))
        return f"APOE{a}{b}"

    alleles = re.findall(r"[234]", s)
    if len(alleles) >= 2:
        a, b = sorted([alleles[0], alleles[1]])
        return f"APOE{a}{b}"

    return np.nan


def choose_age_col(df: pd.DataFrame) -> Optional[str]:
    return first_existing(
        df,
        [
            "age_for_table1",
            "age_true",
            "Age",
            "AGE",
            "age",
            "VISIT_AGE",
            "SUBJECT_AGE_SCREEN",
            "DWI_visit_age",
        ],
    )


def choose_sex_col(df: pd.DataFrame) -> Optional[str]:
    return first_existing(
        df,
        [
            "sex_label",
            "sex",
            "Sex",
            "SEX",
            "PTGENDER",
            "SUBJECT_SEX",
            "gender",
            "ID_Gender",
            "sex_numeric",
        ],
    )


def choose_subject_key_col(df: pd.DataFrame, cohort: str) -> Optional[str]:
    if cohort == "ADNI":
        candidates = ["DWI_subject_key", "DWI_subject", "PTID", "subject_id", "RID"]
    elif cohort == "HABS":
        candidates = ["DWI_subject", "Subject", "Med_ID", "subject_id"]
    elif cohort == "ADRC":
        candidates = ["PTID", "subject_id", "Subject", "match_id", "connectome_subject"]
    elif cohort == "AD_DECODE":
        candidates = ["MRI_Exam", "MRI_Exam_fixed", "ID", "match_id", "subject_id", "IDRNA"]
    else:
        candidates = ["subject_id", "Subject", "ID"]
    return first_existing(df, candidates)


def choose_session_key_col(df: pd.DataFrame, cohort: str) -> Optional[str]:
    candidates = [
        "CONNECTOME_KEY_USED_FOR_INTERSECTION",
        "CONNECTOME_KEY_CLEAN",
        "DWI_key",
        "DWI",
        "connectome_key",
        "connectome_full_key",
        "graph_id",
        "runno",
        "table1_session_key",
    ]
    return first_existing(df, candidates)


def choose_apoe_genotype_col(df: pd.DataFrame) -> Optional[str]:
    return first_existing(
        df,
        [
            "APOE_genotype_harmonized",
            "APOE_Label",
            "genotype",
            "APOE4_Genotype",
            "APOE",
            "GENOTYPE",
        ],
    )


def choose_apoe4_col(df: pd.DataFrame) -> Optional[str]:
    if USE_STRICT_APOE4_FOR_TABLE1:
        return first_existing(
            df,
            [
                "APOE4_strict_34_44_carrier",
                "APOE4_carrier_strict_34_44",
                "APOE4_strict_carrier",
            ],
        )
    return first_existing(df, ["APOE4_carriage", "APOE4_carrier", "APOE4_Positivity"])


def choose_global_cog_col(df: pd.DataFrame) -> Optional[str]:
    return first_existing(
        df,
        [
            "Global_Cognition_Composite",
            "cognition_composite",
            "global_cognition_composite",
            "cognitive_impairment_composite",
        ],
    )


# =============================================================================
# LOAD AND STANDARDIZE FINAL HARMONIZED METADATA
# =============================================================================

def load_harmonized_metadata() -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = []
    summary_rows = []

    for cohort, path in HARMONIZED_METADATA_FILES.items():
        row = {
            "cohort": cohort,
            "input_file": str(path),
            "exists": path.exists(),
            "n_rows": 0,
            "subject_key_col": "",
            "session_key_col": "",
            "age_col": "",
            "sex_col": "",
            "apoe_genotype_col": "",
            "apoe4_col": "",
            "global_cognition_col": "",
            "n_subjects": 0,
            "n_sessions": 0,
            "n_normcog": 0,
            "n_mci": 0,
            "n_ad": 0,
            "n_unknown_dx": 0,
            "n_apoe_genotype": 0,
            "n_apoe4_nonmissing": 0,
            "n_global_cognition": 0,
            "status": "",
        }

        if not path.exists():
            row["status"] = "missing_input"
            summary_rows.append(row)
            print(f"[MISSING] {cohort}: {path}")
            continue

        df = pd.read_csv(path, low_memory=False)
        df = df.copy()
        df["cohort"] = cohort

        subj_col = choose_subject_key_col(df, cohort)
        sess_col = choose_session_key_col(df, cohort)
        age_col = choose_age_col(df)
        sex_col = choose_sex_col(df)
        apoe_geno_col = choose_apoe_genotype_col(df)
        apoe4_col = choose_apoe4_col(df)
        cog_col = choose_global_cog_col(df)

        if subj_col is not None:
            df["table1_subject_key"] = as_key(df[subj_col])
        else:
            df["table1_subject_key"] = pd.Series([f"{cohort}_{i}" for i in range(len(df))], index=df.index)

        if sess_col is not None:
            df["table1_session_key"] = as_key(df[sess_col])
        else:
            df["table1_session_key"] = df["table1_subject_key"]

        if age_col is not None:
            df["age_for_table1"] = pd.to_numeric(df[age_col], errors="coerce")
        else:
            df["age_for_table1"] = np.nan

        if sex_col is not None and sex_col == "sex_label":
            df["sex_label_for_table1"] = df[sex_col].astype(str).replace({"nan": "Unknown", "None": "Unknown"})
        elif sex_col is not None:
            df["sex_label_for_table1"] = df[sex_col].map(normalize_sex_value)
        else:
            df["sex_label_for_table1"] = "Unknown"

        if apoe_geno_col is not None:
            df["APOE_Label_for_table1"] = df[apoe_geno_col].map(normalize_apoe_label)
        else:
            df["APOE_Label_for_table1"] = np.nan

        if apoe4_col is not None:
            df["APOE4_for_table1"] = pd.to_numeric(df[apoe4_col], errors="coerce")
        else:
            # Derive strict or any-carrier from genotype if APOE4 column is missing.
            label = df["APOE_Label_for_table1"]
            if USE_STRICT_APOE4_FOR_TABLE1:
                df["APOE4_for_table1"] = np.nan
                df.loc[label.isin(["APOE22", "APOE23", "APOE33"]), "APOE4_for_table1"] = 0
                df.loc[label.isin(["APOE34", "APOE44"]), "APOE4_for_table1"] = 1
            else:
                df["APOE4_for_table1"] = np.where(label.astype(str).str.contains("4", na=False), 1, np.nan)
                df.loc[label.notna() & ~label.astype(str).str.contains("4", na=False), "APOE4_for_table1"] = 0

        if cog_col is not None:
            df["Global_Cognition_for_table1"] = pd.to_numeric(df[cog_col], errors="coerce")
        else:
            df["Global_Cognition_for_table1"] = np.nan

        # Ensure core columns exist.
        for col in ["NORMCOG_01", "MCI_01", "AD_01", "DEMENTIA_01"]:
            if col not in df.columns:
                df[col] = np.nan
            df[col] = pd.to_numeric(df[col], errors="coerce")

        if "DX_Label_harmonized" not in df.columns:
            df["DX_Label_harmonized"] = "Unknown"

        row.update(
            {
                "n_rows": len(df),
                "subject_key_col": subj_col or "",
                "session_key_col": sess_col or "",
                "age_col": age_col or "",
                "sex_col": sex_col or "",
                "apoe_genotype_col": apoe_geno_col or "",
                "apoe4_col": apoe4_col or "",
                "global_cognition_col": cog_col or "",
                "n_subjects": int(df["table1_subject_key"].nunique(dropna=True)),
                "n_sessions": int(df["table1_session_key"].nunique(dropna=True)),
                "n_normcog": int(df["NORMCOG_01"].eq(1).sum()),
                "n_mci": int(df["MCI_01"].eq(1).sum()),
                "n_ad": int(df["AD_01"].eq(1).sum()),
                "n_unknown_dx": int(df["DX_Label_harmonized"].fillna("Unknown").astype(str).eq("Unknown").sum()),
                "n_apoe_genotype": int(df["APOE_Label_for_table1"].notna().sum()),
                "n_apoe4_nonmissing": int(df["APOE4_for_table1"].notna().sum()),
                "n_global_cognition": int(df["Global_Cognition_for_table1"].notna().sum()),
                "status": "ok",
            }
        )

        summary_rows.append(row)
        frames.append(df)

        print(
            f"[OK] {cohort:<9} rows={len(df):<5} "
            f"subjects={row['n_subjects']:<5} sessions={row['n_sessions']:<5} "
            f"NORM={row['n_normcog']:<4} MCI={row['n_mci']:<4} AD={row['n_ad']:<4} "
            f"UnknownDX={row['n_unknown_dx']:<4} APOE={row['n_apoe_genotype']:<4} "
            f"Cog={row['n_global_cognition']:<4}"
        )

    scan_df = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    source_summary = pd.DataFrame(summary_rows)
    return scan_df, source_summary


def make_subject_level(scan_df: pd.DataFrame) -> pd.DataFrame:
    if scan_df.empty:
        return pd.DataFrame()

    # Select one representative row per subject for subject-level characteristics.
    # Prefer known diagnosis, APOE, cognition, and earliest age.
    tmp = scan_df.copy()
    tmp["_has_dx"] = ~tmp["DX_Label_harmonized"].fillna("Unknown").astype(str).eq("Unknown")
    tmp["_has_apoe"] = tmp["APOE_Label_for_table1"].notna()
    tmp["_has_cog"] = tmp["Global_Cognition_for_table1"].notna()
    tmp["_age_sort"] = pd.to_numeric(tmp["age_for_table1"], errors="coerce")

    subject_df = (
        tmp.sort_values(
            ["cohort", "table1_subject_key", "_has_dx", "_has_apoe", "_has_cog", "_age_sort"],
            ascending=[True, True, False, False, False, True],
            na_position="last",
        )
        .drop_duplicates(["cohort", "table1_subject_key"], keep="first")
        .drop(columns=["_has_dx", "_has_apoe", "_has_cog", "_age_sort"])
        .copy()
    )
    return subject_df


# =============================================================================
# TABLE VALUES
# =============================================================================

def diagnostic_counts_for_cohort(subject_df: pd.DataFrame, cohort: str) -> dict:
    sub = subject_df[subject_df["cohort"].eq(cohort)].copy()
    n = len(sub)

    norm = pd.to_numeric(sub["NORMCOG_01"], errors="coerce")
    mci = pd.to_numeric(sub["MCI_01"], errors="coerce")
    ad = pd.to_numeric(sub["AD_01"], errors="coerce")
    dem = pd.to_numeric(sub["DEMENTIA_01"], errors="coerce")
    dx = sub["DX_Label_harmonized"].fillna("Unknown").astype(str)

    normal_n = int(norm.eq(1).sum())
    mci_n = int(mci.eq(1).sum())
    ad_n = int((ad.eq(1) | dem.eq(1)).sum())
    unknown_n = int(dx.eq("Unknown").sum())

    known_categories = norm.eq(1) | mci.eq(1) | ad.eq(1) | dem.eq(1) | dx.eq("Unknown")
    other_n = int((~known_categories).sum())

    if cohort == "ADRC":
        other_n += int(dx.str.contains("Other|Impaired|non-demented", case=False, regex=True, na=False).sum())
    if cohort == "HABS":
        other_n += int(dx.str.contains("Other", case=False, regex=True, na=False).sum())

    vc = dx.value_counts(dropna=False).to_dict()

    return {
        "normal": normal_n,
        "mci": mci_n,
        "ad": ad_n,
        "other": other_n,
        "unknown": unknown_n,
        "raw_dx_counts": "; ".join([f"{k}:{v}" for k, v in vc.items()]),
        "n": n,
    }


def cohort_col_values(subject_df: pd.DataFrame, scan_df: pd.DataFrame, cohort: str) -> dict:
    sub = subject_df[subject_df["cohort"].eq(cohort)].copy()
    scans = scan_df[scan_df["cohort"].eq(cohort)].copy()
    n = len(sub)

    sex = sub["sex_label_for_table1"].fillna("Unknown").astype(str)
    apoe_label = sub["APOE_Label_for_table1"].fillna("Missing").astype(str)
    apoe4 = pd.to_numeric(sub["APOE4_for_table1"], errors="coerce")
    apoe4_denom = int(apoe4.notna().sum())
    cog = pd.to_numeric(sub["Global_Cognition_for_table1"], errors="coerce")

    dx_counts = diagnostic_counts_for_cohort(subject_df, cohort)

    vals = {
        "Study design": STUDY_DESIGN.get(cohort, "—"),
        "Cohort description": COHORT_DESCRIPTION.get(cohort, "—"),
        "Modalities/features included": FEATURE_DESCRIPTION.get(cohort, "—"),
        "Connectome/DWI sessions included, n": int(scans["table1_session_key"].nunique(dropna=True)),
        "Unique subjects, n": int(sub["table1_subject_key"].nunique(dropna=True)),
        "Matched records/sessions, n": int(len(scans)),
        "Subjects used for characteristics, n": n,
        "Age, years, mean ± SD": fmt_mean_sd(sub["age_for_table1"]),
        "Age, years, median [IQR]": fmt_median_iqr(sub["age_for_table1"]),
        "Age, years, range": fmt_range(sub["age_for_table1"]),
        "Female, n (%)": fmt_count_pct(int(sex.eq("Female").sum()), n),
        "Male, n (%)": fmt_count_pct(int(sex.eq("Male").sum()), n),
        "Sex missing/not reported, n (%)": fmt_count_pct(int(sex.eq("Unknown").sum()), n),
        "APOE ε4 carrier, n (%)": fmt_count_pct(int(apoe4.eq(1).sum()), apoe4_denom) if apoe4_denom > 0 else "—",
        "APOE22, n (%)": fmt_count_pct(int(apoe_label.eq("APOE22").sum()), n),
        "APOE23, n (%)": fmt_count_pct(int(apoe_label.eq("APOE23").sum()), n),
        "APOE24, n (%)": fmt_count_pct(int(apoe_label.eq("APOE24").sum()), n),
        "APOE33, n (%)": fmt_count_pct(int(apoe_label.eq("APOE33").sum()), n),
        "APOE34, n (%)": fmt_count_pct(int(apoe_label.eq("APOE34").sum()), n),
        "APOE44, n (%)": fmt_count_pct(int(apoe_label.eq("APOE44").sum()), n),
        "APOE genotype missing/not reported, n (%)": fmt_count_pct(int(apoe_label.eq("Missing").sum()), n),
        "Cognitively normal/normal, n (%)": dash_if_zero_count(dx_counts["normal"], n),
        "MCI, n (%)": dash_if_zero_count(dx_counts["mci"], n),
        "AD/dementia, n (%)": dash_if_zero_count(dx_counts["ad"], n),
        "Other/unknown diagnostic label, n (%)": dash_if_zero_count(dx_counts["other"], n),
        "Diagnostic group missing/not classified, n (%)": dash_if_zero_count(dx_counts["unknown"], n, use_dash=False),
        "Raw diagnostic counts": dx_counts["raw_dx_counts"],
    }

    if INCLUDE_COGNITIVE_COMPOSITE_ROWS:
        vals.update(
            {
                "Global cognition composite available, n (%)": fmt_count_pct(int(cog.notna().sum()), n),
                "Global cognition composite, mean ± SD": fmt_mean_sd(cog),
            }
        )

    return vals


def make_table1_values(subject_df: pd.DataFrame, scan_df: pd.DataFrame) -> pd.DataFrame:
    row_order = [
        ("Cohort information", "Study design"),
        ("Cohort information", "Cohort description"),
        ("Cohort information", "Modalities/features included"),
        ("Sample size", "Connectome/DWI sessions included, n"),
        ("Sample size", "Unique subjects, n"),
        ("Sample size", "Matched records/sessions, n"),
        ("Sample size", "Subjects used for characteristics, n"),
        ("Demographic characteristics", "Age, years, mean ± SD"),
        ("Demographic characteristics", "Age, years, median [IQR]"),
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
        ("Clinical/diagnostic group", "Other/unknown diagnostic label, n (%)"),
        ("Clinical/diagnostic group", "Diagnostic group missing/not classified, n (%)"),
    ]

    if INCLUDE_COGNITIVE_COMPOSITE_ROWS:
        row_order.extend(
            [
                ("Cognitive composites", "Global cognition composite available, n (%)"),
                ("Cognitive composites", "Global cognition composite, mean ± SD"),
            ]
        )

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
        rows.append(
            {
                "cohort": cohort,
                "n_unique_subjects": int(sub["table1_subject_key"].nunique(dropna=True)),
                "n_sessions": int(scans["table1_session_key"].nunique(dropna=True)),
                "n_rows": int(len(scans)),
                "age_mean": pd.to_numeric(sub["age_for_table1"], errors="coerce").mean(),
                "age_sd": pd.to_numeric(sub["age_for_table1"], errors="coerce").std(ddof=1),
                "n_female": int(sub["sex_label_for_table1"].eq("Female").sum()),
                "n_male": int(sub["sex_label_for_table1"].eq("Male").sum()),
                "n_normcog": int(pd.to_numeric(sub["NORMCOG_01"], errors="coerce").eq(1).sum()),
                "n_mci": int(pd.to_numeric(sub["MCI_01"], errors="coerce").eq(1).sum()),
                "n_ad": int(pd.to_numeric(sub["AD_01"], errors="coerce").eq(1).sum()),
                "n_dementia": int(pd.to_numeric(sub["DEMENTIA_01"], errors="coerce").eq(1).sum()),
                "n_apoe_genotype": int(sub["APOE_Label_for_table1"].notna().sum()),
                "n_apoe4": int(pd.to_numeric(sub["APOE4_for_table1"], errors="coerce").eq(1).sum()),
                "n_global_cognition": int(pd.to_numeric(sub["Global_Cognition_for_table1"], errors="coerce").notna().sum()),
                "dx_counts": "; ".join(
                    [
                        f"{k}:{v}"
                        for k, v in sub["DX_Label_harmonized"].fillna("Missing").astype(str).value_counts().items()
                    ]
                ),
            }
        )
    return pd.DataFrame(rows)


# =============================================================================
# FORMATTED EXCEL
# =============================================================================

def save_formatted_excel(table1_values: pd.DataFrame, source_summary: pd.DataFrame, subject_summary: pd.DataFrame, output_path: Path) -> None:
    try:
        from openpyxl import Workbook
        from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
    except Exception as exc:
        print(f"[WARN] openpyxl not available; skipping formatted XLSX: {exc}")
        return

    wb = Workbook()
    ws = wb.active
    ws.title = "Table 1"

    n_cols = 1 + len(COHORTS)
    dark_blue = "1F4E79"
    medium_blue = "5B9BD5"
    light_blue = "DDEBF7"
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
    ws["A3"] = (
        "Values are n (%) unless otherwise noted. Percentages are calculated using unique subjects as the "
        "denominator for subject-level characteristics. Input files are the final harmonized metadata files."
    )
    ws["A3"].font = Font(italic=True, size=10)
    ws["A3"].alignment = Alignment(wrap_text=True, vertical="center")
    ws.row_dimensions[3].height = 34

    header_row = 5
    headers = ["Characteristic"] + [c.replace("_", "-") for c in COHORTS]
    for j, h in enumerate(headers, start=1):
        cell = ws.cell(header_row, j, h)
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
            c.font = Font(bold=True)
            c.alignment = Alignment(horizontal="left", vertical="center")
            for j in range(1, n_cols + 1):
                ws.cell(current_row, j).border = border
                ws.cell(current_row, j).fill = PatternFill("solid", fgColor=medium_blue)
            current_row += 1

        vals = [characteristic] + [r[c] for c in COHORTS]
        for j, val in enumerate(vals, start=1):
            cell = ws.cell(current_row, j, val)
            cell.fill = PatternFill("solid", fgColor=light_blue if current_row % 2 == 0 else white)
            cell.border = border
            cell.alignment = Alignment(horizontal="left" if j == 1 else "center", vertical="center", wrap_text=True)
        current_row += 1

    current_row += 1
    ws.merge_cells(start_row=current_row, start_column=1, end_row=current_row, end_column=n_cols)
    apoe_note = "strict APOE ε4 carriage excludes APOE2/4" if USE_STRICT_APOE4_FOR_TABLE1 else "APOE ε4 carriage includes any ε4 allele"
    ws.cell(current_row, 1).value = (
        "Note. Abbreviations: AD, Alzheimer’s disease; ADNI, Alzheimer’s Disease Neuroimaging Initiative; "
        "ADRC, Alzheimer’s Disease Research Center; APOE, apolipoprotein E; DTI, diffusion tensor imaging; "
        "FA, fractional anisotropy; HABS, Harvard Aging Brain Study; MCI, mild cognitive impairment; "
        "SD, standard deviation. Diagnostic labels, APOE variables, cognitive composites, and connectome/DWI keys "
        f"come from the final harmonized metadata files. For this table, {apoe_note}."
    )
    ws.cell(current_row, 1).font = Font(size=9)
    ws.cell(current_row, 1).alignment = Alignment(wrap_text=True, vertical="top")
    ws.row_dimensions[current_row].height = 85

    ws.freeze_panes = "B6"
    widths = {"A": 44, "B": 30, "C": 30, "D": 30, "E": 30}
    for col, width in widths.items():
        ws.column_dimensions[col].width = width

    # Add source summary sheet.
    ws2 = wb.create_sheet("Source summary")
    for col_idx, col in enumerate(source_summary.columns, start=1):
        ws2.cell(1, col_idx, col).font = Font(bold=True)
        ws2.cell(1, col_idx).fill = PatternFill("solid", fgColor=light_blue)
    for row_idx, (_, row) in enumerate(source_summary.iterrows(), start=2):
        for col_idx, col in enumerate(source_summary.columns, start=1):
            ws2.cell(row_idx, col_idx, row[col])

    ws3 = wb.create_sheet("Subject summary")
    for col_idx, col in enumerate(subject_summary.columns, start=1):
        ws3.cell(1, col_idx, col).font = Font(bold=True)
        ws3.cell(1, col_idx).fill = PatternFill("solid", fgColor=light_blue)
    for row_idx, (_, row) in enumerate(subject_summary.iterrows(), start=2):
        for col_idx, col in enumerate(subject_summary.columns, start=1):
            ws3.cell(row_idx, col_idx, row[col])

    wb.save(output_path)


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("TABLE 1 FROM FINAL HARMONIZED METADATA")
    print("=" * 100)
    print("WORK:", WORK)
    print("Harmonized metadata dir:", HARMONIZED_DIR)
    print("Output:", OUTDIR)
    print("Use strict APOE4 for Table 1:", USE_STRICT_APOE4_FOR_TABLE1)

    scan_df, source_summary = load_harmonized_metadata()
    if scan_df.empty:
        source_summary.to_csv(OUTDIR / "Table1_harmonized_metadata_source_summary.csv", index=False)
        raise SystemExit("[ERROR] No harmonized metadata rows found.")

    subject_df = make_subject_level(scan_df)
    table1_wide = make_table1_values(subject_df, scan_df)
    table1_long = make_table1_long(table1_wide)
    subject_summary = make_subject_summary(subject_df, scan_df)

    # Write outputs.
    scan_df.to_csv(OUTDIR / "Table1_scan_level_from_harmonized_metadata.csv", index=False)
    subject_df.to_csv(OUTDIR / "Table1_subject_level_from_harmonized_metadata.csv", index=False)
    table1_wide.to_csv(OUTDIR / "Table1_from_harmonized_metadata_wide.csv", index=False)
    table1_long.to_csv(OUTDIR / "Table1_from_harmonized_metadata_long.csv", index=False)
    source_summary.to_csv(OUTDIR / "Table1_harmonized_metadata_source_summary.csv", index=False)
    subject_summary.to_csv(OUTDIR / "Table1_subject_summary_from_harmonized_metadata.csv", index=False)

    save_formatted_excel(
        table1_wide,
        source_summary,
        subject_summary,
        OUTDIR / "Table1_from_harmonized_metadata_formatted.xlsx",
    )

    readme = OUTDIR / "Table1_harmonized_metadata_README.md"
    readme.write_text(
        "# Table 1 from final harmonized metadata\n\n"
        "This table is generated directly from the final cohort-level harmonized metadata files in:\n\n"
        f"`{HARMONIZED_DIR}`\n\n"
        "It does not redo ADNI DXSUM merging, HABS APOE merging, connectome filtering, diagnosis harmonization, "
        "APOE harmonization, or cognitive composite construction. Those are performed upstream by the final "
        "harmonized metadata creation script.\n\n"
        "Main outputs:\n"
        "- `Table1_from_harmonized_metadata_formatted.xlsx`\n"
        "- `Table1_from_harmonized_metadata_wide.csv`\n"
        "- `Table1_from_harmonized_metadata_long.csv`\n"
        "- `Table1_subject_level_from_harmonized_metadata.csv`\n"
        "- `Table1_scan_level_from_harmonized_metadata.csv`\n"
        "- `Table1_harmonized_metadata_source_summary.csv`\n"
        "- `Table1_subject_summary_from_harmonized_metadata.csv`\n\n"
        f"APOE4 definition used in Table 1: {'strict 3/4 or 4/4 only; excludes 2/4' if USE_STRICT_APOE4_FOR_TABLE1 else 'any APOE4 allele; includes 2/4, 3/4, and 4/4'}.\n"
    )

    print("\n[DONE] Outputs written to:")
    print(OUTDIR)
    print("\nSource summary:")
    print(source_summary.to_string(index=False))
    print("\nSubject summary:")
    print(subject_summary.to_string(index=False))


if __name__ == "__main__":
    main()
