#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ALL-IN-ONE cBAG biological validation pipeline.

This single script replaces the three-script workflow:

1) rank_cbag_metadata_merged_variables_all_models.py
2) make_final_figure5_main_and_supp_cbag_biological_validation.py
3) run_final_cbag_biological_validation_pipeline.py

It does everything in one file:

- reads bias-corrected cBAG validation outputs
- reads corrected Table 1 enriched status metadata and cohort metadata/biomarker files
- reads harmonized cognitive composite CSVs
- merges validation + enriched status metadata + metadata/biomarkers + cognitive composites
- screens all usable numeric variables against bias-corrected cBAG
- computes Pearson r, p-values, FDR q-values
- ranks variables by |r|, p-value, and FDR q-value
- saves all CSV outputs
- saves merged tables for audit and plotting
- creates curated family heatmaps
- creates final Figure 5 scatterplot panel
- creates Supplementary Figure S5 heatmap/model-comparison panel
- runs APOE4 / sex / cognitive-impairment AUC using full-cohort inferred cBAG
  when available, so impaired subjects are included for AD_DECODE/HABS/ADRC if
  they exist in the full-cohort prediction files.

No command-line arguments are needed. Edit USER SETTINGS below.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


# ============================================================
# USER SETTINGS — EDIT ONLY THIS SECTION
# ============================================================

BASE_DIR = Path("/mnt/newStor/paros/paros_WORK/ines")
RESULTS_ROOT = BASE_DIR / "results"

RUN_SCREENING = True
RUN_FINAL_FIGURES = True
SAVE_MERGED_TABLES = True

MAIN_FEATURE_SET = "imaging_only"

# Save Figure 5 and Supplementary Figure S5 for all desired models.
FINAL_FIGURE_FEATURE_SETS = [
    "imaging_only",
    "imaging_demographics",
    "imaging_biomarkers",
    "full",
    "full_no_cardiovascular",
]

FEATURE_SETS_TO_SCREEN = [
    "imaging_only",
    "imaging_demographics",
    "imaging_biomarkers",
    "full",
    "full_no_cardiovascular",
]

MODEL_COMPARISON_FEATURE_SETS = [
    "imaging_only",
    "imaging_demographics",
    "imaging_biomarkers",
    "full",
    "full_no_cardiovascular",
]

COHORTS = ["ADNI", "ADRC", "HABS", "AD_DECODE"]

FDR_THRESHOLD = 0.05
MIN_N = 30
MIN_UNIQUE = 5
TOP_N = 50
INCLUDE_DEMOGRAPHICS = False
ALLOW_NONFDR_MAIN_FALLBACK = False
COMMON_MIN_COHORTS = 2
FIGURE_FORMATS = ["png", "pdf"]

SCREENING_OUTDIR = RESULTS_ROOT / "Figure5_AssociationsOnly"
FINAL_FIGURE_OUTDIR = SCREENING_OUTDIR / "final_figures"
MERGED_OUTDIR = SCREENING_OUTDIR / "merged_tables"

# Derived ROI/brain scalar outputs used by Figure 5.
# These are saved separately for audit in addition to being merged into the
# Figure 5 screening tables.
REGIONAL_STATS_OUTDIR = SCREENING_OUTDIR / "regional_stats"

# Enriched status metadata produced by final_build_table1_and_enriched_status_metadata.py.
# These files are the preferred source for NORMCOG_01 / DEMENTIA_01 and for
# full-cohort cBAG with cohort IDs. Figure 5 uses them for point colouring and
# APOE4/sex/cognitive-impairment AUC.
ENRICHED_STATUS_METADATA_DIR = RESULTS_ROOT / "Table1_Corrected" / "enriched_metadata_with_status"

# Binary discrimination analyses: APOE4, sex, cognitive impairment
RUN_BINARY_AUC_ANALYSIS = True
# Run binary AUC for all model feature sets, or only MAIN_FEATURE_SET.
RUN_BINARY_AUC_ALL_FEATURE_SETS = True
BINARY_AUC_FEATURE_SETS = FEATURE_SETS_TO_SCREEN if RUN_BINARY_AUC_ALL_FEATURE_SETS else [MAIN_FEATURE_SET]

# Save a copy of each AUC input table with the strict APOE4 target appended.
# Strict APOE4 rule for this figure:
#   positive = APOE 3/4 or 4/4
#   negative = APOE 2/2, 2/3, 3/3
#   excluded/missing = APOE 2/4 or any non-interpretable genotype
BINARY_AUC_SAVE_ANNOTATED_METADATA = True

# AUC should use final-model full-cohort cBAG when available.
# This is needed for ADRC / AD_DECODE / HABS cognitive-impairment AUC because the OOF
# validation subset may contain only cognitively normal subjects.
BINARY_AUC_USE_FULL_COHORT_PREDICTIONS = True

# Minimum group size for binary AUC. Use 5 so AD_DECODE MCI/AD cases can be estimated,
# but the output is labelled as small-sample where applicable.
MIN_AUC_GROUP_N = 5

BINARY_AUC_OUTDIR = SCREENING_OUTDIR / "APOE4_Sex_CognitiveImpairment_AUC"
BOOTSTRAP_N = 2000
BOOTSTRAP_SEED = 42

# Scatter colours for Figure 5 association panels.
# Colour-blind-friendly green/purple pair:
#   0 = cognitively normal / unimpaired
#   1 = cognitively impaired / MCI / dementia / AD
NORMAL_DOT_COLOR = "#009E73"      # green
IMPAIRED_DOT_COLOR = "#CC79A7"    # purple/magenta
UNKNOWN_DOT_COLOR = "#BDBDBD"     # grey
REGRESSION_LINE_COLOR = "#222222"
CI_BAND_COLOR = "#888888"


# ============================================================
# CONSTANTS
# ============================================================

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

# Only bias-corrected cBAG columns. Do not fall back to raw BAG.
CBAG_PRIORITY = [
    # OOF-global cBAG for cross-validated biological associations.
    "cBAG_oof_global_raw_clean",
    "cBAG_oof_global",

    # Full-cohort final-model inference columns.
    "cBAG_global_raw_clean",
    "cBAG_global",

    # Backward-compatible bias-corrected cBAG names.
    "cBAG_bias_corrected",
    "cBAG_BiasCorrected",
    "cBAG",
]

CURATED_FAMILIES = [
    "memory",
    "executive_function",
    "processing_speed",
    "language",
    "visuospatial",
    "global_cognition_screening",
    "tau_ptau",
    "amyloid_abeta",
    "nfl",
    "gfap",
    "apoe",
    "hippocampus",
    "brain_volume",
    "fa_diffusion",
    "graph_clustering",
    "graph_path_length",
    "global_efficiency",
    "local_efficiency",
    "cardiovascular",
    "depression_anxiety",
]

CURATED_LABELS = {
    "memory": "Memory",
    "executive_function": "Executive function",
    "processing_speed": "Processing speed",
    "language": "Language",
    "visuospatial": "Visuospatial",
    "global_cognition_screening": "MoCA / MMSE / CDR",
    "tau_ptau": "Tau / pTau",
    "amyloid_abeta": "Amyloid / Aβ",
    "nfl": "NfL",
    "gfap": "GFAP",
    "apoe": "APOE",
    "hippocampus": "Hippocampus",
    "brain_volume": "Brain volume",
    "fa_diffusion": "FA / diffusion",
    "graph_clustering": "Graph clustering",
    "graph_path_length": "Graph path length",
    "global_efficiency": "Global efficiency",
    "local_efficiency": "Local efficiency",
    "cardiovascular": "Cardiovascular",
    "depression_anxiety": "Depression/anxiety",
}

MAIN_COLUMNS = {
    "Cognition": [
        "memory",
        "executive_function",
        "processing_speed",
        "language",
        "visuospatial",
        "global_cognition_screening",
    ],
    "Fluid / genetic biomarker": [
        "tau_ptau",
        "amyloid_abeta",
        "nfl",
        "gfap",
        "apoe",
    ],
    "Brain network / structure": [
        "global_efficiency",
        "local_efficiency",
        "graph_clustering",
        "graph_path_length",
        "hippocampus",
        "brain_volume",
        "fa_diffusion",
    ],
    "Clinical / vascular": [
        "cardiovascular",
        "depression_anxiety",
    ],
}

CURATED_RULES: List[Tuple[str, str]] = [
    ("global_efficiency", r"global[_\s]*efficiency"),
    ("local_efficiency", r"local[_\s]*efficiency"),
    ("graph_clustering", r"cluster|clustering|coeff|transitivity|modular"),
    ("graph_path_length", r"path[_\s]*length|shortest[_\s]*path|characteristic[_\s]*path|pathlength"),

    # Hippocampus-specific ROI extractions must be classified before generic
    # volume/FA rules.
    ("hippocampus", r"hippocampus|hippocampal|(^|_)hippo($|_)|(^|_)lh_hippo|(^|_)rh_hippo"),

    # Put FA before generic volume so total_brain_FA_* is not captured by the
    # generic brain/volume pattern.
    ("fa_diffusion", r"(^|_)fa($|_)|fractional[_\s]*anisotropy|diffusion|dti|(^|_)md($|_)|mean[_\s]*diffusivity|total[_\s]*brain[_\s]*fa|whole[_\s]*brain[_\s]*fa"),
    ("brain_volume", r"brain[_\s]*volume|total[_\s]*brain[_\s]*volume|whole[_\s]*brain[_\s]*volume|\btbv\b|\bicv\b|volume|volumetric|cortical[_\s]*thickness|thickness"),
    ("tau_ptau", r"ptau|p[_\s\-]*tau|tau|total[_\s]*tau|totaltau"),
    ("amyloid_abeta", r"abeta|aβ|amyloid|a_beta|ab42|ab40"),
    ("nfl", r"\bnfl\b|nfl_|nefl|neurofilament"),
    ("gfap", r"gfap"),
    ("apoe", r"apoe|e4|genotype"),
    ("memory", r"memory|sevlt|ravlt|avlt|logical|lm1|lm2|delayed|recall|bentd|story|limmtotal|ldeltotal|avdel30min|avdeltot|avtot[1-6b]"),
    ("executive_function", r"executive|trail[_\s]*b|trailb|trabscor|bckwds|backward|dspanbac|dspanblth|abstraction|set[_\s]*shift"),
    ("processing_speed", r"processing[_\s]*speed|trail[_\s]*a|traila|traascor|digitscor|digit[_\s]*symbol|digitsymbol|symbol[_\s]*substitution|ufov"),
    ("language", r"language|fluency|fas|animals|animal|catanimsc|catvegesc|bnttotal|minttotal|mintuncued|naming|wat|word[_\s]*accent|verbal[_\s]*flu"),
    ("visuospatial", r"visuospatial|benson|figure|copy|copyscor|clockscor|construction"),
    ("global_cognition_screening", r"moca|mocatots|mmse|cdr|cdrsb|cdglobal|adas|cognition|cognitive|global[_\s]*cog"),
    ("cardiovascular", r"blood[_\s]*pressure|\bsbp\b|\bdbp\b|pulse|chol|hdl|ldl|triglycer|glucose|hba1c|diabetes|hypertension|bmi|body[_\s]*mass|insulin|homa|egfr|creatinine"),
    ("depression_anxiety", r"depress|anxiety|\bgds\b|pswq|worry"),
]


# Preferred variables for curated heatmap selection when these variables are present.
# This prevents generic nodewise summaries or assay QC columns from replacing the
# biologically clearer Figure 5 variables.
PREFERRED_CURATED_VARIABLE_PATTERNS = {
    "brain_volume": [
        r"(^|_)total_brain_volume_abs_mm3$",
    ],
    "fa_diffusion": [
        r"(^|_)total_brain_fa_mean$",
    ],
    "hippocampus": [
        r"(^|_)hippocampus_volume_abs_sum_mm3$",
        r"(^|_)hippocampus_volume_abs_mean_mm3$",
        r"(^|_)hippocampus_fa_mean$",
    ],
    "tau_ptau": [
        r"dilution_corrected_conc",
        r"plasma_ptau217",
        r"ptau217",
        r"ptau",
        r"tau",
    ],
    "amyloid_abeta": [
        r"abeta42_40_ratio",
        r"abeta42",
        r"abeta40",
    ],
    "apoe": [
        r"apoe_e4_dose_screening",
        r"genotype_encoded",
        r"apoe4_strict_34_44_carrier_screening",
    ],
}

def curated_preference_rank(fam: object, var: object) -> int:
    fam = str(fam)
    low = normalize_name(var)
    pats = PREFERRED_CURATED_VARIABLE_PATTERNS.get(fam, [])
    for i, pat in enumerate(pats):
        if re.search(pat, low, flags=re.IGNORECASE):
            return i
    return 999

EXCLUDE_CURATED_PATTERNS = [
    r"sars|covid|spike|nucleocapsid|rbd",
    r"\bpc\d+\b|pc\d+[_\s]*z|transcriptomic[_\s]*pc",
    r"height[_\s]*cm|\bheight\b|vsheight",
    r"dsq[_\s]*onset",
    r"runno|visit_id|med_id|subject|id_pca|idrna|id_rna|match_id",
]

EXCLUDE_TOKENS = [
    "cbag",
    "bag",
    "brainage",
    "brain_age",
    "predictedage",
    "predicted_age",
    "prediction",
    "biascorrect",
    "bias_correct",
    "correctedage",
    "chronological",
    "fold",
    "split",
    "train",
    "test",
    "val",
    "oof",
    "rmse",
    "mae",
    "r2",
    "auc",
    "target",
    "label",
    "index",
    "unnamed",
    "file",
    "filename",
    "metadata",
    "connectome_key",
    "connectome_full_key",
    "graph_path",
    "source",
    "sample_id",
    "siteid",
    "userdate",
    "update_stamp",
    "dd_crf",
    "language_code",
    "has_qc_error",
]

DEMOGRAPHIC_TOKENS = ["age", "sex", "gender", "educ", "education", "site", "scanner", "race", "ethnic"]


# ============================================================
# PATH HELPERS
# ============================================================

def metadata_path(cohort: str) -> Path:
    """
    Metadata file used by Figure 5.

    For ADNI, prefer the enriched CSV produced by:
        prepare_adni_metadata_and_composites_for_figure5.py

    This avoids slow/large Excel reads and guarantees Figure 5 sees the merged
    NEUROBAT, CDR, biomarker, Janssen p217 tau, diagnosis, and vascular fields.
    Other cohorts keep their original metadata workbooks.
    """
    if cohort == "ADNI":
        enriched_csv = BASE_DIR / "data" / "harmonization" / "ADNI" / "metadata" / "ADNI_metadata_figure5_enriched.csv"
        if enriched_csv.exists():
            return enriched_csv

    return BASE_DIR / "data" / "harmonization" / cohort / "metadata" / f"{cohort}_metadata.xlsx"


def read_metadata_table(path: Path) -> pd.DataFrame:
    """Read cohort metadata from CSV or Excel."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, low_memory=False)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported metadata file type: {path}")


def composite_path(cohort: str) -> Path:
    return BASE_DIR / "results" / "composite_cognition" / f"{cohort}_cognitive_composites.csv"


def validation_path(cohort: str, feature_set: str) -> Path:
    return RESULTS_ROOT / RESULTS_DIR_MAP[cohort] / f"ablation_{feature_set}" / "validation_figures" / "subject_level_validation_input.csv"


def metadata_all_with_predictions_path(cohort: str, feature_set: str) -> Path:
    prefix = PREFIX_MAP[cohort]
    return (
        RESULTS_ROOT
        / RESULTS_DIR_MAP[cohort]
        / f"ablation_{feature_set}"
        / f"{prefix}_{feature_set}_metadata_all_with_predictions.csv"
    )


def enriched_status_metadata_path(cohort: str, feature_set: str) -> Path:
    """
    Table 1 corrected/enriched metadata with:
      NORMCOG_01, DEMENTIA_01, DX_Label_harmonized, COG_STATUS_SOURCE.

    For ADNI, those columns are based on DXSUM DIAGNOSIS:
      1=CN, 2=MCI, 3=AD/dementia.
    """
    return ENRICHED_STATUS_METADATA_DIR / f"metadata_all_with_status_{feature_set}_{cohort}.csv"


def full_cohort_predictions_path(cohort: str, feature_set: str) -> Path:
    prefix = PREFIX_MAP[cohort]
    return (
        RESULTS_ROOT
        / RESULTS_DIR_MAP[cohort]
        / f"ablation_{feature_set}"
        / f"{prefix}_{feature_set}_full_cohort_predictions.csv"
    )


# ============================================================
# GRAPH-BUILDER-DERIVED FEATURES
# ============================================================

# These names must match graph_builder_with_normcog_feature_sets.py.
GRAPH_METRIC_COLS = [
    "Clustering_Coeff",
    "Path_Length",
    "Global_Efficiency",
    "Local_Efficiency",
]

# Hippocampus and total-brain extraction from the same ROI-wise FA/volume
# node-feature tables used by the graph builder.
#
# Hippocampus:
#   Preferred route: use structure/Structure labels containing hippocampus/hippo.
#   Fallback route: FreeSurfer-style aseg ROI IDs 17 and 53, commonly
#   left/right hippocampus.
#
# Total brain:
#   Volume = sum/mean/sd/min/max over all non-exterior volume ROIs.
#   FA     = mean/sd/min/max over all non-exterior FA ROIs because FA is not a
#            volumetric quantity and should not be summed.
#
# The graph-builder cleaning already excludes exterior/non-brain rows:
#   FA ROI 0 is removed; volume ROI -1 is removed.
HIPPOCAMPUS_STRUCTURE_PATTERN = r"hippocampus|hippocampal|\bhippo\b"
HIPPOCAMPUS_FALLBACK_ROI_IDS = {"17", "53"}

# Same cohort-specific node-feature keys and raw FA / volume files used by the
# graph builder. ADNI and HABS use connectome_key for FA/volume; ADRC and
# AD_DECODE use regional_id.
NODE_FEATURE_CONFIG = {
    "ADNI": {
        "fa_path": BASE_DIR / "data" / "Regional_stats" / "ADNI" / "ADNI_studywide_stats_for_fa.txt",
        "vol_path": BASE_DIR / "data" / "Regional_stats" / "ADNI" / "ADNI_studywide_stats_BrainPct.csv",
        "vol_abs_path": BASE_DIR / "data" / "Regional_stats" / "ADNI" / "ADNI_studywide_stats_BrainAbs.csv",
        "fa_sep": "\t",
        "node_feature_key": "connectome_key",
    },
    "ADRC": {
        "fa_path": BASE_DIR / "data" / "Regional_stats" / "ADRC" / "ADRC_studywide_stats_for_fa.txt",
        "vol_path": BASE_DIR / "data" / "Regional_stats" / "ADRC" / "ADRC_studywide_stats_BrainPct.csv",
        "vol_abs_path": BASE_DIR / "data" / "Regional_stats" / "ADRC" / "ADRC_studywide_stats_BrainAbs.csv",
        "fa_sep": "\t",
        "node_feature_key": "regional_id",
    },
    "HABS": {
        "fa_path": BASE_DIR / "data" / "Regional_stats" / "HABS" / "HABS_studywide_stats_for_fa.txt",
        "vol_path": BASE_DIR / "data" / "Regional_stats" / "HABS" / "HABS_studywide_stats_BrainPct.csv",
        "vol_abs_path": BASE_DIR / "data" / "Regional_stats" / "HABS" / "HABS_studywide_stats_BrainAbs.csv",
        "fa_sep": "\t",
        "node_feature_key": "connectome_key",
    },
    "AD_DECODE": {
        "fa_path": BASE_DIR / "data" / "Regional_stats" / "ADDecode" / "ADDecode_studywide_stats_for_fa.txt",
        "vol_path": BASE_DIR / "data" / "Regional_stats" / "ADDecode" / "ADDecode_studywide_stats_BrainPct.csv",
        "vol_abs_path": BASE_DIR / "data" / "Regional_stats" / "ADDecode" / "ADDecode_studywide_stats_BrainAbs.csv",
        "fa_sep": "\t",
        "node_feature_key": "regional_id",
    },
}


def graph_builder_dir(cohort: str, feature_set: str) -> Path:
    return RESULTS_ROOT / "harmonized" / cohort / "graphs" / feature_set


def graph_builder_metadata_path(cohort: str, feature_set: str, all_subjects: bool = True, model_ready: bool = True) -> Path:
    cohort_lc = cohort.lower()
    if all_subjects:
        suffix = "metadata_all_aligned" if model_ready else "metadata_all_aligned_raw"
    else:
        suffix = "metadata_aligned" if model_ready else "metadata_aligned_raw"
    return graph_builder_dir(cohort, feature_set) / f"{cohort_lc}_{suffix}.csv"


def load_graph_builder_metrics(cohort: str, feature_set: str) -> Tuple[Optional[pd.DataFrame], str]:
    """
    Load the graph-builder aligned metadata table that contains the graph
    properties actually used by the model.

    The graph builder saves graph metrics in the model-ready aligned tables under
    exactly these columns:
        Clustering_Coeff, Path_Length, Global_Efficiency, Local_Efficiency

    Those columns may be z-scored in the graph-builder output, but Pearson r and
    p-values are invariant to positive affine scaling, so screening against cBAG
    is unchanged while the column names and subject matching stay correct.
    """
    candidates = [
        graph_builder_metadata_path(cohort, feature_set, all_subjects=True, model_ready=True),
        graph_builder_metadata_path(cohort, feature_set, all_subjects=False, model_ready=True),
    ]
    for p in candidates:
        if not p.exists():
            continue
        df = pd.read_csv(p, low_memory=False)
        keep = []
        for c in possible_keys(df) + GRAPH_METRIC_COLS:
            if c in df.columns and c not in keep:
                keep.append(c)
        metric_present = [c for c in GRAPH_METRIC_COLS if c in df.columns]
        if metric_present:
            return df[keep].copy(), str(p)
    return None, "; ".join(str(p) for p in candidates)


def _clean_graph_builder_fa_table(df_fa: pd.DataFrame) -> pd.DataFrame:
    # Mirrors graph_builder_with_normcog_feature_sets.py: remove an optional
    # nonnumeric ROI header row and drop ROI == 0.
    if "ROI" in df_fa.columns:
        try:
            roi_numeric = pd.to_numeric(df_fa["ROI"], errors="coerce")
            if len(df_fa) > 0 and pd.isna(roi_numeric.iloc[0]):
                df_fa = df_fa.iloc[1:].copy()
            df_fa = df_fa[df_fa["ROI"].astype(str) != "0"].copy().reset_index(drop=True)
        except Exception:
            pass
    return df_fa


def _clean_graph_builder_volume_table(df_vol: pd.DataFrame) -> pd.DataFrame:
    # Mirrors graph_builder_with_normcog_feature_sets.py: drop ROI == -1.
    if "ROI" in df_vol.columns:
        df_vol = df_vol[df_vol["ROI"].astype(str) != "-1"].copy().reset_index(drop=True)
    return df_vol


def _transpose_subject_feature_table(df: pd.DataFrame, valid_subjects: Optional[set], key_name: str) -> pd.DataFrame:
    if valid_subjects:
        subject_cols = [c for c in df.columns if str(c).strip() in valid_subjects]
    else:
        non_subject = {"ROI", "roi", "ROI_numeric", "index", "Unnamed: 0"}
        subject_cols = [c for c in df.columns if str(c) not in non_subject]
    out = df[subject_cols].transpose()
    out.columns = [f"ROI_{i+1}" for i in range(out.shape[1])]
    out.index = out.index.astype(str).str.strip()
    out.index.name = key_name
    return out.apply(pd.to_numeric, errors="coerce")


def _find_structure_col(df: pd.DataFrame) -> Optional[str]:
    """Return the structure-name column in regional stats tables, if present."""
    for c in ["structure", "Structure", "STRUCTURE", "label", "Label", "name", "Name"]:
        if c in df.columns:
            return c
    for c in df.columns:
        if "struct" in str(c).lower():
            return c
    return None


def _clean_roi_value(x: object) -> Optional[str]:
    if pd.isna(x):
        return None
    s = str(x).strip()
    if not s or s.lower() == "nan":
        return None
    try:
        f = float(s)
        if np.isfinite(f) and f.is_integer():
            return str(int(f))
    except Exception:
        pass
    return re.sub(r"\.0$", "", s)


def _select_hippocampus_rows(df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    """
    Select hippocampus ROI rows from a regional stats table.

    First tries the structure/Structure text column. If that fails, falls back
    to FreeSurfer-style ROI IDs 17 and 53, commonly left/right hippocampus.
    """
    if df is None or df.empty:
        return df.iloc[0:0].copy(), "empty_table"

    structure_col = _find_structure_col(df)
    if structure_col is not None:
        labels = df[structure_col].astype(str)
        mask = labels.str.contains(HIPPOCAMPUS_STRUCTURE_PATTERN, case=False, regex=True, na=False)
        if mask.any():
            return df.loc[mask].copy(), f"structure_col={structure_col}"

    if "ROI" in df.columns:
        roi_clean = df["ROI"].map(_clean_roi_value)
        mask = roi_clean.isin(HIPPOCAMPUS_FALLBACK_ROI_IDS)
        if mask.any():
            return df.loc[mask].copy(), f"fallback_ROI_ids={','.join(sorted(HIPPOCAMPUS_FALLBACK_ROI_IDS))}"

    return df.iloc[0:0].copy(), "not_found"


def _subject_columns_from_regional_table(df: pd.DataFrame, valid_subjects: Optional[set]) -> List[str]:
    """Return subject columns from ROI x subject regional-stats tables."""
    if valid_subjects:
        cols = [c for c in df.columns if str(c).strip() in valid_subjects]
        if cols:
            return cols

    non_subject = {
        "ROI", "roi", "ROI_numeric", "index", "Index", "Index2",
        "structure", "Structure", "STRUCTURE", "label", "Label", "name", "Name",
        "Unnamed: 0",
    }
    return [c for c in df.columns if str(c) not in non_subject]


def _summarize_roi_rows_by_subject(
    df: pd.DataFrame,
    valid_subjects: Optional[set],
    key_name: str,
    prefix: str,
) -> pd.DataFrame:
    """
    Summarize selected ROI rows per subject from an ROI x subject table.

    Used for hippocampus-specific variables where rows are left/right hippocampus
    and columns are subjects.
    """
    subject_cols = _subject_columns_from_regional_table(df, valid_subjects)
    out = pd.DataFrame({key_name: [str(c).strip() for c in subject_cols]})
    if len(subject_cols) == 0 or df.empty:
        out[f"{prefix}_mean"] = np.nan
        out[f"{prefix}_sum"] = np.nan
        out[f"{prefix}_min"] = np.nan
        out[f"{prefix}_max"] = np.nan
        out[f"{prefix}_n_rois"] = 0
        return out

    vals = df[subject_cols].apply(pd.to_numeric, errors="coerce")
    out[f"{prefix}_mean"] = vals.mean(axis=0).to_numpy()
    out[f"{prefix}_sum"] = vals.sum(axis=0, min_count=1).to_numpy()
    out[f"{prefix}_min"] = vals.min(axis=0).to_numpy()
    out[f"{prefix}_max"] = vals.max(axis=0).to_numpy()
    out[f"{prefix}_n_rois"] = vals.notna().sum(axis=0).to_numpy()
    return out




def _extract_total_brain_pct_value(df_pct: pd.DataFrame, valid_subjects: Optional[set], key_name: str) -> Tuple[pd.DataFrame, str]:
    """
    Extract the BrainPct total Brain row if present.

    This is normally 100 for every subject, so it is saved for audit only and
    should not be interpreted as an association variable.
    """
    subject_cols = _subject_columns_from_regional_table(df_pct, valid_subjects)
    out = pd.DataFrame({key_name: [str(c).strip() for c in subject_cols]})
    if df_pct is None or df_pct.empty or not subject_cols:
        out["brainpct_total_brain_row_value"] = np.nan
        return out, "missing_or_no_subject_columns"

    brain_mask = pd.Series(False, index=df_pct.index)
    structure_col = _find_structure_col(df_pct)
    if structure_col is not None:
        brain_mask = brain_mask | df_pct[structure_col].astype(str).str.fullmatch("Brain", case=False, na=False)
    if "ROI" in df_pct.columns:
        brain_mask = brain_mask | df_pct["ROI"].map(_clean_roi_value).eq("-1")

    vals = df_pct[subject_cols].apply(pd.to_numeric, errors="coerce")
    if brain_mask.any():
        out["brainpct_total_brain_row_value"] = vals.loc[brain_mask].iloc[0].to_numpy()
        return out, "BrainPct_Brain_row"
    out["brainpct_total_brain_row_value"] = np.nan
    return out, "BrainPct_Brain_row_not_found"


def _extract_total_brain_abs_volume(df_abs: Optional[pd.DataFrame], valid_subjects: Optional[set], key_name: str) -> Tuple[pd.DataFrame, str]:
    """
    Extract total brain absolute volume per subject from a BrainAbs regional table.

    Preferred: use the row with structure == Brain or ROI == -1.
    Fallback: sum all non-exterior/non-Brain regional rows.
    """
    if df_abs is None or df_abs.empty:
        return pd.DataFrame(columns=[key_name, "total_brain_volume_abs_mm3"]), "missing_abs_volume_table"

    subject_cols = _subject_columns_from_regional_table(df_abs, valid_subjects)
    out = pd.DataFrame({key_name: [str(c).strip() for c in subject_cols]})
    if not subject_cols:
        out["total_brain_volume_abs_mm3"] = np.nan
        return out, "no_subject_columns"

    source = "fallback_sum_nonexterior_abs_rois"
    brain_mask = pd.Series(False, index=df_abs.index)

    structure_col = _find_structure_col(df_abs)
    if structure_col is not None:
        brain_mask = brain_mask | df_abs[structure_col].astype(str).str.fullmatch("Brain", case=False, na=False)

    if "ROI" in df_abs.columns:
        brain_mask = brain_mask | df_abs["ROI"].map(_clean_roi_value).eq("-1")

    vals = df_abs[subject_cols].apply(pd.to_numeric, errors="coerce")
    if brain_mask.any():
        out["total_brain_volume_abs_mm3"] = vals.loc[brain_mask].iloc[0].to_numpy()
        source = "Brain_row_abs_volume"
    else:
        keep = pd.Series(True, index=df_abs.index)
        if structure_col is not None:
            keep = keep & ~df_abs[structure_col].astype(str).str.contains("exterior", case=False, na=False)
        if "ROI" in df_abs.columns:
            roi_clean = df_abs["ROI"].map(_clean_roi_value)
            keep = keep & ~roi_clean.isin(["-1", "0"])
        out["total_brain_volume_abs_mm3"] = vals.loc[keep].sum(axis=0, min_count=1).to_numpy()

    return out, source


def load_graph_builder_node_feature_summary(
    cohort: str,
    graph_metrics_df: Optional[pd.DataFrame] = None,
    feature_set: Optional[str] = None,
) -> Tuple[Optional[pd.DataFrame], str]:
    """
    Read raw FA and volume exactly as the graph builder does, transpose by the
    cohort-specific node_feature_key, and add subject-level summaries.

    Adds:
      1) whole-node summaries:
         nodewise_FA_mean/sd/min/max
         nodewise_volume_mean/sd/min/max

      2) hippocampus-specific summaries:
         hippocampus_FA_mean/sum/min/max
         hippocampus_volume_mean/sum/min/max

      3) total-brain scalar summaries:
         total_brain_FA_mean/sd/min/max
         total_brain_volume_abs_mm3 from BrainAbs, when available
         brainpct_total_brain_row_value from BrainPct for audit only

    The hippocampus and total-brain summaries are extracted from the same
    ROI-wise FA/volume tables used as graph node features. They are not
    graph-builder global features; they are explicit scalar readouts for
    Figure 5 biological validation.
    """
    cfg = NODE_FEATURE_CONFIG[cohort]
    fa_path = Path(cfg["fa_path"])
    vol_path = Path(cfg["vol_path"])
    vol_abs_path = Path(cfg.get("vol_abs_path", ""))
    key = str(cfg["node_feature_key"])

    if not fa_path.exists() or not vol_path.exists():
        return None, f"missing FA or volume file: {fa_path}; {vol_path}"

    valid_subjects = None
    if graph_metrics_df is not None and key in graph_metrics_df.columns:
        valid_subjects = set(graph_metrics_df[key].dropna().astype(str).str.strip())

    df_fa_raw = pd.read_csv(fa_path, sep=cfg["fa_sep"], low_memory=False)
    df_vol_raw = pd.read_csv(vol_path, low_memory=False)

    # Absolute volume table is preferred for biological volume readouts.
    # BrainPct is retained for graph-builder-compatible nodewise summaries.
    df_vol_abs_raw = None
    if vol_abs_path.exists():
        df_vol_abs_raw = pd.read_csv(vol_abs_path, low_memory=False)

    # Select hippocampal rows before generic ROI cleaning, so structure labels
    # and original ROI IDs are still available.
    df_fa_hippo_raw, fa_hippo_source = _select_hippocampus_rows(df_fa_raw)
    df_vol_hippo_raw, vol_hippo_source = _select_hippocampus_rows(df_vol_abs_raw if df_vol_abs_raw is not None else df_vol_raw)
    vol_biological_source = str(vol_abs_path) if df_vol_abs_raw is not None else str(vol_path)

    df_fa = _clean_graph_builder_fa_table(df_fa_raw.copy())
    df_vol = _clean_graph_builder_volume_table(df_vol_raw.copy())

    df_fa_t = _transpose_subject_feature_table(df_fa, valid_subjects, key)
    df_vol_t = _transpose_subject_feature_table(df_vol, valid_subjects, key)

    common = sorted(set(df_fa_t.index).intersection(df_vol_t.index))
    if not common:
        return None, f"no common FA/volume subjects after transposition: {fa_path}; {vol_path}"

    fa = df_fa_t.loc[common]
    vol = df_vol_t.loc[common]

    out = pd.DataFrame({key: common})

    # Existing generic nodewise summaries.
    out["nodewise_FA_mean"] = fa.mean(axis=1).to_numpy()
    out["nodewise_FA_sd"] = fa.std(axis=1, ddof=0).to_numpy()
    out["nodewise_FA_min"] = fa.min(axis=1).to_numpy()
    out["nodewise_FA_max"] = fa.max(axis=1).to_numpy()
    out["nodewise_volume_mean"] = vol.mean(axis=1).to_numpy()
    out["nodewise_volume_sd"] = vol.std(axis=1, ddof=0).to_numpy()
    out["nodewise_volume_min"] = vol.min(axis=1).to_numpy()
    out["nodewise_volume_max"] = vol.max(axis=1).to_numpy()
    out["nodewise_FA_n_rois"] = fa.notna().sum(axis=1).to_numpy()
    out["nodewise_volume_n_rois"] = vol.notna().sum(axis=1).to_numpy()

    # Explicit total-brain scalar readouts over all non-exterior ROIs.
    # FA is averaged rather than summed because FA is a regional scalar, not a
    # volumetric quantity. Volume gets sum plus distribution summaries.
    out["total_brain_FA_mean"] = out["nodewise_FA_mean"]
    out["total_brain_FA_sd"] = out["nodewise_FA_sd"]
    out["total_brain_FA_min"] = out["nodewise_FA_min"]
    out["total_brain_FA_max"] = out["nodewise_FA_max"]
    out["total_brain_FA_n_rois"] = out["nodewise_FA_n_rois"]

    # BrainPct-derived nodewise summaries above are regional percentages, not a
    # meaningful "total brain percent". If the BrainPct file has a Brain row,
    # that row is normally 100 by definition and has no useful variance for
    # association testing. Keep it only as an audit column.
    brain_pct_total, brain_pct_source = _extract_total_brain_pct_value(df_vol_raw, valid_subjects, key)
    out = out.merge(brain_pct_total, on=key, how="left")

    # Absolute total brain volume from BrainAbs, when available. This is the
    # biologically interpretable total-brain volume variable for Figure 5.
    total_abs, total_abs_source = _extract_total_brain_abs_volume(df_vol_abs_raw, valid_subjects, key)
    out = out.merge(total_abs, on=key, how="left")

    # Explicit hippocampus scalars.
    hippo_fa = _summarize_roi_rows_by_subject(
        df_fa_hippo_raw,
        valid_subjects,
        key,
        "hippocampus_FA",
    )
    hippo_vol = _summarize_roi_rows_by_subject(
        df_vol_hippo_raw,
        valid_subjects,
        key,
        "hippocampus_volume",
    )

    out = out.merge(hippo_fa, on=key, how="left")
    out = out.merge(hippo_vol, on=key, how="left")

    # For clarity, alias hippocampus volume columns to absolute-volume names
    # when they came from BrainAbs. If BrainAbs is missing, these contain the
    # same fallback values as hippocampus_volume_* and the source column records it.
    out["hippocampus_volume_abs_mean_mm3"] = out.get("hippocampus_volume_mean", np.nan)
    out["hippocampus_volume_abs_sum_mm3"] = out.get("hippocampus_volume_sum", np.nan)
    out["hippocampus_volume_abs_min_mm3"] = out.get("hippocampus_volume_min", np.nan)
    out["hippocampus_volume_abs_max_mm3"] = out.get("hippocampus_volume_max", np.nan)

    # Constant diagnostic columns for audit/debugging; nonnumeric, so they are
    # automatically ignored by scan_numeric_vars().
    out["hippocampus_FA_extraction_source"] = fa_hippo_source
    out["hippocampus_volume_extraction_source"] = vol_hippo_source
    out["brainpct_total_brain_row_extraction_source"] = brain_pct_source
    out["total_brain_abs_volume_extraction_source"] = total_abs_source
    out["regional_stats_FA_source_file"] = str(fa_path)
    out["regional_stats_BrainPct_source_file"] = str(vol_path)
    out["regional_stats_BrainAbs_source_file"] = str(vol_biological_source)

    # Save the exact derived brain/hippocampus scalar table used by Figure 5.
    # This is intentionally separate from source Regional_stats files so the
    # raw graph-builder inputs are not modified.
    if feature_set is not None:
        REGIONAL_STATS_OUTDIR.mkdir(parents=True, exist_ok=True)
        safe_fs = str(feature_set)
        regional_out = REGIONAL_STATS_OUTDIR / f"derived_brain_region_scalars_{safe_fs}_{cohort}.csv"
        out.to_csv(regional_out, index=False)

    return out, (
        f"{fa_path}; BrainPct={vol_path}; BrainAbs={vol_biological_source}; "
        f"hippocampus_FA={fa_hippo_source}; "
        f"hippocampus_volume={vol_hippo_source}; "
        f"brainpct_total_brain_row={brain_pct_source}; "
        f"total_brain_abs_volume={total_abs_source}"
    )

# ============================================================
# GENERAL HELPERS
# ============================================================

def normalize_name(x: object) -> str:
    s = str(x).lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def safe_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def first_existing(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    lower = {str(c).lower(): c for c in df.columns}
    for c in candidates:
        if str(c).lower() in lower:
            return lower[str(c).lower()]
    return None


def p_text(p: float) -> str:
    if pd.isna(p):
        return "p=NA"
    if p < 1e-4:
        return "p<1e-4"
    if p < 0.001:
        return "p<0.001"
    return f"p={p:.3g}"


def q_text(q: float) -> str:
    if pd.isna(q):
        return "q=NA"
    if q < 1e-4:
        return "q<1e-4"
    if q < 0.001:
        return "q<0.001"
    return f"q={q:.3g}"


def shorten(x: object, n: int = 28) -> str:
    s = str(x)
    return s if len(s) <= n else s[: n - 1] + "…"


def family_label(fam: str) -> str:
    return CURATED_LABELS.get(str(fam), str(fam).replace("_", " "))


def title_feature_set(feature_set: str) -> str:
    return feature_set.replace("_", " ")


def fdr_bh(pvals: Sequence[float]) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    q = np.full_like(p, np.nan, dtype=float)
    ok = np.isfinite(p)
    if ok.sum() == 0:
        return q
    p_ok = p[ok]
    order = np.argsort(p_ok)
    ranked = p_ok[order]
    m = len(ranked)
    q_ranked = ranked * m / (np.arange(1, m + 1))
    q_ranked = np.minimum.accumulate(q_ranked[::-1])[::-1]
    q_ranked = np.clip(q_ranked, 0, 1)
    q_ok = np.empty_like(q_ranked)
    q_ok[order] = q_ranked
    q[ok] = q_ok
    return q


# ============================================================
# MERGING HELPERS
# ============================================================

def norm_key(x) -> Optional[str]:
    if pd.isna(x):
        return None
    s = str(x).strip()
    if not s or s.lower() == "nan":
        return None
    s = re.sub(r"\.csv$", "", s)
    s = re.sub(r"_conn_plain$", "", s)
    s = re.sub(r"_master_T_?\d*$", "", s)
    s = re.sub(r"_temp_T_?\d*$", "", s)
    s = re.sub(r"^(R\d+)(y\d+)$", r"\1_\2", s)
    return s


def rid_from_any(x) -> Optional[str]:
    if pd.isna(x):
        return None
    s = str(x).strip()
    m = re.search(r"R(\d+)", s)
    if m:
        return str(int(m.group(1)))
    m = re.search(r"_S_(\d+)", s)
    if m:
        return str(int(m.group(1)))
    if re.fullmatch(r"\d+(\.0)?", s):
        return str(int(float(s)))
    return None


def adni_y_from_key(x) -> Optional[str]:
    if pd.isna(x):
        return None
    m = re.search(r"_y(\d+)", str(x), flags=re.IGNORECASE)
    return f"y{m.group(1)}" if m else None


def possible_keys(df: pd.DataFrame) -> List[str]:
    tokens = [
        "connectome",
        "subject",
        "participant",
        "match",
        "regional",
        "id",
        "rid",
        "ptid",
        "visit",
        "viscode",
        "runno",
        "med_id",
        "exam",
    ]
    return [c for c in df.columns if any(t in str(c).lower() for t in tokens)]


def prefix_columns(df: pd.DataFrame, prefix: str, keep_cols: Sequence[str]) -> pd.DataFrame:
    out = df.copy()
    keep = set(keep_cols)
    rename = {}
    for c in out.columns:
        if c in keep:
            continue
        if str(c).startswith(f"{prefix}__"):
            continue
        rename[c] = f"{prefix}__{c}"
    return out.rename(columns=rename)


def choose_merge(val: pd.DataFrame, extra: pd.DataFrame, cohort: str, source_name: str) -> Tuple[pd.DataFrame, Dict[str, object]]:
    val = val.copy()
    extra = extra.copy()
    report = {"source": source_name, "strategy": "none", "val_key": "", "extra_key": "", "overlap": 0}

    val_keys = possible_keys(val)
    extra_keys = possible_keys(extra)

    # ADNI special: RID + visit, then RID-only.
    if cohort == "ADNI":
        vk_candidates = [k for k in ["connectome_key", "subject_id", "connectome_full_key"] if k in val.columns] or val_keys
        rid_col = first_existing(extra, ["RID", "rid"])
        visit_col = first_existing(extra, ["VISCODE", "VISCODE2", "visit", "Visit", "EXAMDATE"])
        if rid_col:
            for vk in vk_candidates:
                val["_merge_rid"] = val[vk].map(rid_from_any)
                val["_merge_visit"] = val[vk].map(adni_y_from_key)
                extra["_merge_rid"] = extra[rid_col].map(rid_from_any)

                if visit_col:
                    extra["_merge_visit"] = extra[visit_col].astype(str).str.strip().str.lower()
                    visit_map = {
                        "bl": "y0", "baseline": "y0", "sc": "y0", "m00": "y0", "m0": "y0", "0": "y0", "0.0": "y0",
                        "m12": "y1", "12": "y1", "12.0": "y1",
                        "m24": "y2", "24": "y2", "24.0": "y2",
                        "m36": "y3", "36": "y3", "36.0": "y3",
                        "m48": "y4", "48": "y4", "48.0": "y4",
                    }
                    extra["_merge_visit"] = extra["_merge_visit"].replace(visit_map)
                    overlap = len(set(zip(val["_merge_rid"], val["_merge_visit"])).intersection(set(zip(extra["_merge_rid"], extra["_merge_visit"]))))
                    if overlap > 0:
                        extra2 = extra.drop_duplicates(["_merge_rid", "_merge_visit"], keep="first")
                        merged = val.merge(extra2, on=["_merge_rid", "_merge_visit"], how="left", suffixes=("", f"_{source_name}"))
                        report.update({"strategy": "ADNI_RID_visit", "val_key": vk, "extra_key": f"{rid_col}+{visit_col}", "overlap": overlap})
                        return merged, report

                overlap = len(set(val["_merge_rid"].dropna()).intersection(set(extra["_merge_rid"].dropna())))
                if overlap > 0:
                    extra2 = extra.drop_duplicates("_merge_rid", keep="first")
                    merged = val.merge(extra2, on="_merge_rid", how="left", suffixes=("", f"_{source_name}"))
                    report.update({"strategy": "ADNI_RID_only", "val_key": vk, "extra_key": rid_col, "overlap": overlap})
                    return merged, report

    # General normalized-key merge.
    best = None
    for vk in val_keys:
        vset = set(val[vk].map(norm_key).dropna().astype(str))
        if not vset:
            continue
        for ek in extra_keys:
            eset = set(extra[ek].map(norm_key).dropna().astype(str))
            overlap = len(vset.intersection(eset))
            if best is None or overlap > best[0]:
                best = (overlap, vk, ek)

    if best and best[0] > 0:
        overlap, vk, ek = best
        val["_merge_key"] = val[vk].map(norm_key)
        extra["_merge_key"] = extra[ek].map(norm_key)
        extra2 = extra.drop_duplicates("_merge_key", keep="first")
        merged = val.merge(extra2, on="_merge_key", how="left", suffixes=("", f"_{source_name}"))
        report.update({"strategy": "normalized_exact", "val_key": vk, "extra_key": ek, "overlap": overlap})
        return merged, report

    # Row-order fallback only when row counts match.
    if len(val) == len(extra):
        merged = pd.concat([val.reset_index(drop=True), extra.reset_index(drop=True)], axis=1)
        report.update({"strategy": "row_order_fallback", "val_key": "row_index", "extra_key": "row_index", "overlap": len(val)})
        return merged, report

    return val, report



# ============================================================
# APOE ENCODING FOR METADATA SCREENING HEATMAP
# ============================================================

def _normalize_apoe_genotype_for_screening(x: object) -> Optional[str]:
    """Normalize APOE genotype strings to 2_2, 2_3, 2_4, 3_3, 3_4, 4_4."""
    if pd.isna(x):
        return None
    s = str(x).strip().upper().replace(" ", "")
    if not s or s == "NAN":
        return None
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
    return None


def _compose_apoe_genotype_from_two_cols(a1: pd.Series, a2: pd.Series) -> pd.Series:
    aa = pd.to_numeric(a1, errors="coerce")
    bb = pd.to_numeric(a2, errors="coerce")
    out = pd.Series(np.nan, index=a1.index, dtype=object)
    ok = aa.isin([2, 3, 4]) & bb.isin([2, 3, 4])
    lo = np.minimum(aa[ok].astype(int), bb[ok].astype(int)).astype(str)
    hi = np.maximum(aa[ok].astype(int), bb[ok].astype(int)).astype(str)
    out.loc[ok] = lo + "_" + hi
    return out


def _apoe4_strict_from_genotype_norm(g: object) -> float:
    if pd.isna(g):
        return np.nan
    g = str(g)
    if g in {"3_4", "4_4"}:
        return 1.0
    if g in {"2_2", "2_3", "3_3"}:
        return 0.0
    return np.nan


def _numeric_genotype_dose_from_genotype_norm(g: object) -> float:
    if pd.isna(g):
        return np.nan
    g = str(g)
    if g not in {"2_2", "2_3", "2_4", "3_3", "3_4", "4_4"}:
        return np.nan
    return float(g.count("4"))


def append_apoe_screening_columns(df: pd.DataFrame, cohort: str) -> pd.DataFrame:
    """
    Add robust numeric APOE columns for the metadata-wide Pearson screening.
    This is needed when APOE is stored as strings like 3/4 or E3E4.
    """
    out = df.copy()

    genotype_candidates = []
    for c in out.columns:
        low = normalize_name(c)
        if low in {"genotype", "apoe", "apoe_genotype", "apoe_status"} or "apoe_genotype" in low:
            genotype_candidates.append(c)

    best_geno = pd.Series(np.nan, index=out.index, dtype=object)
    best_source = ""

    for c in genotype_candidates:
        g = out[c].map(_normalize_apoe_genotype_for_screening)
        if g.notna().sum() > best_geno.notna().sum():
            best_geno = g
            best_source = c

    norm_to_col = {normalize_name(c): c for c in out.columns}
    allele_pairs = [
        ("apoe_a1", "apoe_a2"),
        ("meta_apoe_a1", "meta_apoe_a2"),
        ("apoe_a1_metadata", "apoe_a2_metadata"),
    ]

    for a_norm, b_norm in allele_pairs:
        if a_norm in norm_to_col and b_norm in norm_to_col:
            g = _compose_apoe_genotype_from_two_cols(out[norm_to_col[a_norm]], out[norm_to_col[b_norm]])
            best_geno = best_geno.where(best_geno.notna(), g)
            if not best_source:
                best_source = f"{norm_to_col[a_norm]}+{norm_to_col[b_norm]}"

    a_hits = [c for c in out.columns if re.search(r"apoe.*(?:a1|allele_?1)($|_)", normalize_name(c))]
    b_hits = [c for c in out.columns if re.search(r"apoe.*(?:a2|allele_?2)($|_)", normalize_name(c))]
    if a_hits and b_hits:
        a_col = sorted(a_hits, key=lambda c: len(str(c)))[0]
        b_col = sorted(b_hits, key=lambda c: len(str(c)))[0]
        g = _compose_apoe_genotype_from_two_cols(out[a_col], out[b_col])
        best_geno = best_geno.where(best_geno.notna(), g)
        if not best_source:
            best_source = f"{a_col}+{b_col}"

    out["apoe_genotype_norm_screening"] = best_geno
    out["apoe_e4_dose_screening"] = best_geno.map(_numeric_genotype_dose_from_genotype_norm)
    out["apoe4_strict_34_44_carrier_screening"] = best_geno.map(_apoe4_strict_from_genotype_norm)
    out["apoe_screening_source"] = best_source

    if int(out["apoe_genotype_norm_screening"].notna().sum()) > 0:
        print(
            f"[INFO] APOE screening columns | {cohort}: "
            f"source={best_source}; genotype_n={int(out['apoe_genotype_norm_screening'].notna().sum())}; "
            f"e4_dose_n={int(out['apoe_e4_dose_screening'].notna().sum())}; "
            f"strict_carrier_n={int(out['apoe4_strict_34_44_carrier_screening'].notna().sum())}"
        )
    else:
        print(f"[WARN] APOE screening columns | {cohort}: no genotype/allele source found")
    return out


# ============================================================
# SCREENING
# ============================================================

def is_excluded_col(col: str) -> bool:
    low = normalize_name(col)

    # Exclude assay/batch/QC metadata that can be numeric but is not a biological variable.
    if re.search(r"(^|_)(run|cv|sample_id|source|siteid|userdate|update_stamp|has_qc_error|language_code)($|_)", low):
        return True

    # Preserve true graph/network metrics. The previous generic "path" exclusion
    # incorrectly removed Path_Length from all cohorts.
    if re.search(r"(^|_)path_length($|_)|characteristic_path|shortest_path|pathlength", low):
        return False
    if re.search(r"global_efficiency|local_efficiency|clustering_coeff|clustering|path_length", low):
        return False

    if any(tok in low for tok in EXCLUDE_TOKENS):
        return True
    if not INCLUDE_DEMOGRAPHICS and any(tok in low for tok in DEMOGRAPHIC_TOKENS):
        return True
    return False


def is_excluded_curated(var: str) -> bool:
    low = normalize_name(var)
    return any(re.search(pat, low, flags=re.IGNORECASE) for pat in EXCLUDE_CURATED_PATTERNS)


def curated_family(var: str) -> Optional[str]:
    low = normalize_name(var)
    if is_excluded_curated(var):
        return None
    for fam, pat in CURATED_RULES:
        if re.search(pat, low, flags=re.IGNORECASE):
            return fam
    return None


def scan_numeric_vars(df: pd.DataFrame, cohort: str, feature_set: str, cbag_col: str) -> pd.DataFrame:
    rows = []
    y = safe_numeric(df[cbag_col])

    for col in df.columns:
        if col == cbag_col or is_excluded_col(col):
            continue
        x = safe_numeric(df[col])
        tmp = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
        if len(tmp) < MIN_N or tmp["x"].nunique() < MIN_UNIQUE or tmp["y"].nunique() < 2:
            continue
        try:
            r, p = stats.pearsonr(tmp["x"], tmp["y"])
            slope, intercept, *_ = stats.linregress(tmp["x"], tmp["y"])
        except Exception:
            continue

        fam = curated_family(col)
        rows.append({
            "feature_set": feature_set,
            "cohort": cohort,
            "variable": col,
            "variable_norm": normalize_name(col),
            "curated_family": fam,
            "curated_family_label": CURATED_LABELS.get(fam, "") if fam else "",
            "cbag_col": cbag_col,
            "n": int(len(tmp)),
            "n_unique": int(tmp["x"].nunique()),
            "pearson_r": float(r),
            "pearson_p": float(p),
            "abs_pearson_r": float(abs(r)),
            "slope": float(slope),
            "intercept": float(intercept),
            "x_mean": float(tmp["x"].mean()),
            "x_sd": float(tmp["x"].std(ddof=0)),
            "x_min": float(tmp["x"].min()),
            "x_max": float(tmp["x"].max()),
        })

    out = pd.DataFrame(rows)
    if not out.empty:
        out["fdr_q_within_cohort"] = fdr_bh(out["pearson_p"].values)
    return out


def best_curated(assoc: pd.DataFrame, fdr_only: bool = False) -> pd.DataFrame:
    if assoc.empty or "curated_family" not in assoc.columns:
        return pd.DataFrame()
    df = assoc[assoc["curated_family"].notna()].copy()
    if fdr_only:
        df = df[pd.to_numeric(df["fdr_q_within_cohort"], errors="coerce") < FDR_THRESHOLD]
    if df.empty:
        return pd.DataFrame()

    df["_curated_preference_rank"] = [
        curated_preference_rank(fam, var)
        for fam, var in zip(df["curated_family"], df["variable"])
    ]

    best = (
        df.sort_values(
            ["cohort", "curated_family", "_curated_preference_rank", "abs_pearson_r", "pearson_p"],
            ascending=[True, True, True, False, True],
        )
        .groupby(["cohort", "curated_family"], as_index=False)
        .head(1)
        .copy()
    )
    best["r2"] = best["pearson_r"] ** 2
    best = best.drop(columns=["_curated_preference_rank"], errors="ignore")
    return best


def matrix_from_best(best: pd.DataFrame, families: Sequence[str], cohorts: Sequence[str] = COHORTS) -> Tuple[pd.DataFrame, pd.DataFrame]:
    mat = pd.DataFrame(index=families, columns=cohorts, dtype=float)
    ann = pd.DataFrame("", index=families, columns=cohorts)
    if best.empty:
        return mat, ann

    for _, row in best.iterrows():
        fam = row.get("curated_family")
        cohort = row.get("cohort")
        if fam not in mat.index or cohort not in mat.columns:
            continue
        r = float(row["pearson_r"])
        q = pd.to_numeric(row.get("fdr_q_within_cohort", np.nan), errors="coerce")
        n = int(row.get("n", 0))
        var = str(row.get("variable", ""))
        star = "***" if pd.notna(q) and q < 0.001 else "**" if pd.notna(q) and q < 0.01 else "*" if pd.notna(q) and q < 0.05 else ""
        mat.loc[fam, cohort] = r
        ann.loc[fam, cohort] = f"{r:.2f}{star}\nR²={r*r:.2f}\nn={n}\n{shorten(var, 16)}"
    return mat, ann


def plot_heatmap(mat: pd.DataFrame, ann: pd.DataFrame, title: str, outstem: Path, formats: Sequence[str] = FIGURE_FORMATS) -> None:
    if mat.empty or len(mat.index) == 0:
        print(f"[WARN] Empty heatmap: {outstem.name}")
        return
    data = mat.to_numpy(dtype=float)
    fig_h = max(4.0, 0.55 * len(mat.index) + 1.6)
    fig, ax = plt.subplots(figsize=(8.2, fig_h))
    vmax = max(0.05, np.nanmax(np.abs(data)) if np.isfinite(data).any() else 1.0)
    im = ax.imshow(data, aspect="auto", vmin=-vmax, vmax=vmax, cmap="coolwarm")
    ax.set_title(title, fontsize=12, pad=12)
    ax.set_xticks(np.arange(len(mat.columns)))
    ax.set_xticklabels(mat.columns.tolist(), fontsize=9)
    ax.set_yticks(np.arange(len(mat.index)))
    ax.set_yticklabels([family_label(f) for f in mat.index.tolist()], fontsize=8)
    for i, fam in enumerate(mat.index):
        for j, cohort in enumerate(mat.columns):
            txt = ann.loc[fam, cohort]
            if txt:
                ax.text(j, i, txt, ha="center", va="center", fontsize=6.2)
    ax.set_xticks(np.arange(-0.5, len(mat.columns), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(mat.index), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.0)
    ax.tick_params(which="minor", bottom=False, left=False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    cbar.set_label("Signed Pearson r", fontsize=9)
    fig.tight_layout()
    for fmt in formats:
        path = outstem.with_suffix(f".{fmt}")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"[INFO] Saved {path}")
    plt.close(fig)


def screen_one_feature_set(feature_set: str) -> None:
    assoc_frames = []
    audit_rows = []
    merge_rows = []
    MERGED_OUTDIR.mkdir(parents=True, exist_ok=True)

    for cohort in COHORTS:
        print(f"\n[INFO] Screening {feature_set} | {cohort}")
        vpath = validation_path(cohort, feature_set)
        if not vpath.exists():
            print(f"[WARN] Missing validation table: {vpath}")
            continue

        val = pd.read_csv(vpath, low_memory=False)
        cbag_col = first_existing(val, CBAG_PRIORITY)
        if cbag_col is None:
            print(f"[WARN] No bias-corrected cBAG column found in {vpath}")
            audit_rows.append({"feature_set": feature_set, "cohort": cohort, "issue": "missing_cbag", "validation_path": str(vpath)})
            continue

        merged = val.copy()

        # Merge harmonized status metadata produced by the corrected Table 1 script.
        # This is the authoritative source for NORMCOG_01 / DEMENTIA_01:
        #   ADNI: DXSUM DIAGNOSIS by RID+VISCODE; 1=CN, 2=MCI, 3=AD
        #   ADRC: NORMCOG / IMPNOMCI / DEMENTED
        #   HABS: CDX_Cog
        #   AD_DECODE: Risk, with Risk 0 and Risk 1 combined as NORMCOG/control
        spath = enriched_status_metadata_path(cohort, feature_set)
        if spath.exists():
            status_meta = pd.read_csv(spath, low_memory=False)
            status_meta = prefix_columns(status_meta, "status", keep_cols=possible_keys(status_meta))
            merged, rep = choose_merge(merged, status_meta, cohort, "status_metadata")
            rep.update({"feature_set": feature_set, "cohort": cohort, "path": str(spath)})
            merge_rows.append(rep)
            print(f"[INFO] status metadata merge: {rep}")
        else:
            print(f"[WARN] Enriched status metadata not found: {spath}")
            print("[WARN] Run final_build_table1_and_enriched_status_metadata.py before Figure 5 for final labels.")

        # Merge broader cohort metadata/biomarkers/cognitive raw fields.
        mpath = metadata_path(cohort)
        if mpath.exists():
            meta = read_metadata_table(mpath)
            meta = prefix_columns(meta, "meta", keep_cols=possible_keys(meta))
            merged, rep = choose_merge(merged, meta, cohort, "metadata")
            rep.update({"feature_set": feature_set, "cohort": cohort, "path": str(mpath)})
            merge_rows.append(rep)
            print(f"[INFO] metadata merge: {rep}")
        else:
            print(f"[WARN] Metadata not found: {mpath}")

        # Merge harmonized cognitive composites.
        cpath = composite_path(cohort)
        if cpath.exists():
            cog = pd.read_csv(cpath, low_memory=False)
            cog = prefix_columns(cog, "cog", keep_cols=possible_keys(cog))
            merged, rep = choose_merge(merged, cog, cohort, "cognition")
            rep.update({"feature_set": feature_set, "cohort": cohort, "path": str(cpath)})
            merge_rows.append(rep)
            print(f"[INFO] cognition merge: {rep}")
        else:
            print(f"[WARN] Cognitive composite not found: {cpath}")

        # Merge graph properties from the graph-builder aligned metadata.
        graph_metrics, graph_path = load_graph_builder_metrics(cohort, feature_set)
        if graph_metrics is not None:
            merged, rep = choose_merge(merged, graph_metrics, cohort, "graph_builder_metrics")
            rep.update({"feature_set": feature_set, "cohort": cohort, "path": graph_path})
            merge_rows.append(rep)
            print(f"[INFO] graph-builder metric merge: {rep}")
        else:
            print(f"[WARN] Graph-builder metric table not found or lacks metrics: {graph_path}")

        # Merge scalar summaries of the same raw FA and volume node-feature
        # files used by the graph builder.
        node_summary, node_path = load_graph_builder_node_feature_summary(cohort, graph_metrics, feature_set)
        if node_summary is not None:
            merged, rep = choose_merge(merged, node_summary, cohort, "graph_builder_node_features")
            rep.update({"feature_set": feature_set, "cohort": cohort, "path": node_path})
            merge_rows.append(rep)
            print(f"[INFO] graph-builder FA/volume summary merge: {rep}")
        else:
            print(f"[WARN] Graph-builder FA/volume summary not available: {node_path}")

        # Add numeric APOE variables for the curated APOE heatmap row.
        merged = append_apoe_screening_columns(merged, cohort)

        if SAVE_MERGED_TABLES:
            merged_path = MERGED_OUTDIR / f"merged_metadata_screening_{feature_set}_{cohort}.csv"
            merged.to_csv(merged_path, index=False)
            print(f"[INFO] Saved merged table: {merged_path}")

        assoc = scan_numeric_vars(merged, cohort, feature_set, cbag_col)
        if not assoc.empty:
            assoc_frames.append(assoc)

        audit_rows.append({
            "feature_set": feature_set,
            "cohort": cohort,
            "validation_path": str(vpath),
            "n_rows": len(merged),
            "n_cols_after_merge": merged.shape[1],
            "cbag_col": cbag_col,
            "n_tested": len(assoc),
        })
        print(f"[INFO] Tested variables: {len(assoc)}")

    assoc_all = pd.concat(assoc_frames, ignore_index=True, sort=False) if assoc_frames else pd.DataFrame()
    if not assoc_all.empty:
        assoc_all["fdr_q_all_tests_feature_set"] = fdr_bh(assoc_all["pearson_p"].values)
        assoc_all = assoc_all.sort_values(["cohort", "abs_pearson_r", "pearson_p"], ascending=[True, False, True])

    top_abs = assoc_all.groupby("cohort", group_keys=False).head(TOP_N).copy() if not assoc_all.empty else pd.DataFrame()
    if not top_abs.empty:
        top_abs.insert(2, "rank_within_cohort_by_abs_r", top_abs.groupby("cohort").cumcount() + 1)

    top_p = assoc_all.sort_values(["cohort", "pearson_p"]).groupby("cohort", group_keys=False).head(TOP_N).copy() if not assoc_all.empty else pd.DataFrame()
    if not top_p.empty:
        top_p.insert(2, "rank_within_cohort_by_p", top_p.groupby("cohort").cumcount() + 1)

    top_fdr = assoc_all.sort_values(["cohort", "fdr_q_within_cohort", "pearson_p"]).groupby("cohort", group_keys=False).head(TOP_N).copy() if not assoc_all.empty else pd.DataFrame()
    if not top_fdr.empty:
        top_fdr.insert(2, "rank_within_cohort_by_fdr", top_fdr.groupby("cohort").cumcount() + 1)

    sig = assoc_all[pd.to_numeric(assoc_all.get("fdr_q_within_cohort", np.nan), errors="coerce") < FDR_THRESHOLD].copy() if not assoc_all.empty else pd.DataFrame()
    best = best_curated(assoc_all, fdr_only=False)
    best_fdr = best_curated(assoc_all, fdr_only=True)

    SCREENING_OUTDIR.mkdir(parents=True, exist_ok=True)
    assoc_all.to_csv(SCREENING_OUTDIR / f"metadata_variable_associations_{feature_set}.csv", index=False)
    top_abs.to_csv(SCREENING_OUTDIR / f"metadata_variable_top_by_abs_r_{feature_set}.csv", index=False)
    top_p.to_csv(SCREENING_OUTDIR / f"metadata_variable_top_by_p_{feature_set}.csv", index=False)
    top_fdr.to_csv(SCREENING_OUTDIR / f"metadata_variable_top_by_fdr_{feature_set}.csv", index=False)
    sig.to_csv(SCREENING_OUTDIR / f"metadata_variable_fdr_significant_{feature_set}.csv", index=False)
    best.to_csv(SCREENING_OUTDIR / f"curated_family_best_by_cohort_{feature_set}.csv", index=False)
    best_fdr.to_csv(SCREENING_OUTDIR / f"curated_family_fdr_significant_{feature_set}.csv", index=False)
    pd.DataFrame(audit_rows).to_csv(SCREENING_OUTDIR / f"metadata_variable_screening_audit_{feature_set}.csv", index=False)
    pd.DataFrame(merge_rows).to_csv(SCREENING_OUTDIR / f"metadata_variable_merge_report_{feature_set}.csv", index=False)

    # Basic curated heatmaps.
    mat, ann = matrix_from_best(best, CURATED_FAMILIES)
    plot_heatmap(mat, ann, f"Curated metadata cBAG associations ({title_feature_set(feature_set)})", SCREENING_OUTDIR / f"curated_metadata_heatmap_{feature_set}")

    mat_fdr, ann_fdr = matrix_from_best(best_fdr, CURATED_FAMILIES)
    plot_heatmap(mat_fdr, ann_fdr, f"FDR-significant curated metadata cBAG associations ({title_feature_set(feature_set)}, q<{FDR_THRESHOLD})", SCREENING_OUTDIR / f"curated_metadata_heatmap_FDRonly_{feature_set}")

    print(f"[DONE] {feature_set}: associations={len(assoc_all)}, FDR-significant={len(sig)}")


def run_screening() -> None:
    SCREENING_OUTDIR.mkdir(parents=True, exist_ok=True)
    if SAVE_MERGED_TABLES:
        MERGED_OUTDIR.mkdir(parents=True, exist_ok=True)
    REGIONAL_STATS_OUTDIR.mkdir(parents=True, exist_ok=True)
    for fs in FEATURE_SETS_TO_SCREEN:
        screen_one_feature_set(fs)


# ============================================================
# FINAL FIGURE HELPERS
# ============================================================

def load_assoc(feature_set: str) -> pd.DataFrame:
    path = SCREENING_OUTDIR / f"metadata_variable_associations_{feature_set}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing association file: {path}. Run screening first.")
    df = pd.read_csv(path, low_memory=False)
    for c in ["pearson_r", "pearson_p", "fdr_q_within_cohort", "fdr_q_all_tests_feature_set", "abs_pearson_r", "n"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "abs_pearson_r" not in df.columns and "pearson_r" in df.columns:
        df["abs_pearson_r"] = df["pearson_r"].abs()
    return df


def load_merged(feature_set: str, cohort: str) -> Optional[pd.DataFrame]:
    path = MERGED_OUTDIR / f"merged_metadata_screening_{feature_set}_{cohort}.csv"
    if not path.exists():
        return None
    return pd.read_csv(path, low_memory=False)


def _auc_target_clue_cols(df: pd.DataFrame) -> List[str]:
    """Columns that indicate the AUC table already carries phenotype targets."""
    pats = r"apoe|genotype|sex|gender|normcog|dement|impnomci|group_status|cognitive|diagnos|dx"
    return [c for c in df.columns if re.search(pats, normalize_name(c), flags=re.IGNORECASE)]


def _auc_normalize_match4(x: object) -> Optional[str]:
    """
    Normalize subject IDs used by full-cohort prediction files and graph-builder
    metadata to a shared four-digit key.

    Important ADRC case:
      graph_id may be bare numeric 7/29/204, while metadata uses match_id 0007
      or regional_id D0007. These must all become 0007/0029/0204.
    """
    if pd.isna(x):
        return None
    s = str(x).strip().upper()
    if not s or s == "NAN":
        return None

    # Numeric-like IDs, including 7, 7.0, 29, 29.0.
    try:
        f = float(s)
        if np.isfinite(f) and f.is_integer():
            return str(int(f))[-4:].zfill(4)
    except Exception:
        pass

    groups = re.findall(r"(\d+)", s)
    if not groups:
        return None
    digits = "".join(groups)
    return digits[-4:].zfill(4)


def _auc_regional_from_match4(match4: object, cohort: str) -> object:
    if pd.isna(match4) or match4 is None:
        return np.nan
    m = str(match4).zfill(4)
    if cohort == "ADRC":
        return f"D{m}"
    if cohort == "AD_DECODE":
        return f"S0{m}"
    if cohort == "HABS":
        return f"H{m}"
    if cohort == "ADNI":
        return f"R{m}"
    return m


def _add_auc_merge_keys(df: pd.DataFrame, cohort: str) -> pd.DataFrame:
    """
    Add robust merge keys for prediction tables and graph-builder/raw metadata.

    The ADRC full-cohort prediction table contains cBAG in a file whose only
    subject identifier can be graph_id. In those files graph_id is often a bare
    numeric match ID (e.g., 7, 29), while graph-builder metadata stores match_id
    or regional_id (e.g., 0007, D0007). Therefore graph_id must be included and
    numeric IDs must be zero-padded to four digits.
    """
    out = df.copy()
    candidate_cols = [
        "graph_id",
        "match_id", "regional_id", "connectome_key", "connectome_full_key",
        "graph_subject_key", "node_feature_key", "subject_id", "participant_id",
        "PTID", "ptid", "RID", "MRI_Exam", "runno", "Subject",
    ]
    existing = [c for c in candidate_cols if c in out.columns]

    # Also consider any column whose name looks like an identifier, but avoid broad numeric feature columns.
    for c in out.columns:
        low = normalize_name(c)
        if c not in existing and re.search(
            r"(^|_)(graph_id|id|ptid|rid|subject|participant|regional|connectome|match|mri_exam|runno)($|_)",
            low,
        ):
            existing.append(c)

    # Build one normalized key per candidate, then choose the first nonmissing key row-wise.
    match_key_parts = []
    exact_parts = []
    for c in existing:
        match_key_parts.append(out[c].map(_auc_normalize_match4).astype(object))
        exact_parts.append(out[c].map(norm_key).astype(object))

    if match_key_parts:
        out["_auc_key_match4"] = match_key_parts[0]
        for part in match_key_parts[1:]:
            out["_auc_key_match4"] = out["_auc_key_match4"].where(out["_auc_key_match4"].notna(), part)
    else:
        out["_auc_key_match4"] = np.nan

    # Regional key used by graph builder for ADRC node features: D####.
    out["_auc_key_regional"] = out["_auc_key_match4"].apply(lambda x: _auc_regional_from_match4(x, cohort))

    # Exact normalized keys from common columns.
    if exact_parts:
        out["_auc_key_exact_any"] = exact_parts[0]
        for part in exact_parts[1:]:
            out["_auc_key_exact_any"] = out["_auc_key_exact_any"].where(out["_auc_key_exact_any"].notna(), part)
    else:
        out["_auc_key_exact_any"] = np.nan

    return out

def _merge_on_auc_key(base: pd.DataFrame, meta: pd.DataFrame, cohort: str, source_name: str) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Try robust AUC metadata merges before falling back to choose_merge."""
    b = _add_auc_merge_keys(base, cohort)
    m = _add_auc_merge_keys(meta, cohort)

    key_pairs = [
        ("_auc_key_regional", "_auc_key_regional"),
        ("_auc_key_match4", "_auc_key_match4"),
        ("_auc_key_exact_any", "_auc_key_exact_any"),
    ]
    best = None
    for bk, mk in key_pairs:
        bset = set(b[bk].dropna().astype(str))
        mset = set(m[mk].dropna().astype(str))
        overlap = len(bset.intersection(mset))
        if overlap > 0 and (best is None or overlap > best[0]):
            best = (overlap, bk, mk)

    if best is not None:
        overlap, bk, mk = best
        meta_cols_before = set(b.columns)
        m2 = m.drop_duplicates(mk, keep="first")
        merged = b.merge(m2, left_on=bk, right_on=mk, how="left", suffixes=("", f"_{source_name}"))
        rep = {"source": source_name, "strategy": "auc_key_merge", "val_key": bk, "extra_key": mk, "overlap": overlap}
        return merged, rep

    merged, rep = choose_merge(base.copy(), meta.copy(), cohort, source_name)
    return merged, rep


def _merge_auc_predictions_with_graph_metadata(pred: pd.DataFrame, cohort: str, feature_set: str) -> Tuple[pd.DataFrame, str]:
    """
    Merge a prediction/AUC table with graph-builder or raw metadata so binary
    AUC targets are present. This is deliberately run even when a
    metadata_all_with_predictions file exists, because some ADRC outputs contain
    cBAG predictions but no APOE/sex/NORMCOG target columns.
    """
    candidates = [
        graph_builder_metadata_path(cohort, feature_set, all_subjects=True, model_ready=False),
        graph_builder_metadata_path(cohort, feature_set, all_subjects=True, model_ready=True),
        graph_builder_metadata_path(cohort, feature_set, all_subjects=False, model_ready=False),
        graph_builder_metadata_path(cohort, feature_set, all_subjects=False, model_ready=True),
        metadata_path(cohort),
    ]

    best_df = pred.copy()
    best_source = "prediction_only"
    best_score = -1

    for meta_path in candidates:
        if not meta_path.exists():
            continue
        try:
            if str(meta_path).lower().endswith((".xlsx", ".xls")):
                meta = pd.read_excel(meta_path)
            else:
                meta = pd.read_csv(meta_path, low_memory=False)
        except Exception as exc:
            print(f"[WARN] Could not read AUC metadata candidate {meta_path}: {exc}")
            continue

        merged, rep = _merge_on_auc_key(pred.copy(), meta.copy(), cohort, f"auc_meta_{meta_path.stem}")
        overlap = int(rep.get("overlap", 0) or 0)
        if rep.get("strategy") == "row_order_fallback":
            overlap = min(overlap, 1)

        useful_cols = _auc_target_clue_cols(merged)
        useful_nonmissing = int(sum(pd.Series(merged[c]).notna().sum() for c in useful_cols)) if useful_cols else 0

        # The first term demands actual row overlap. The second term rewards metadata targets.
        combined_score = overlap * 100000 + useful_nonmissing
        if combined_score > best_score:
            best_df = merged
            best_source = f"{meta_path} | merge={rep} | useful_cols={useful_cols[:20]}"
            best_score = combined_score

    return best_df, best_source

def load_auc_dataframe(feature_set: str, cohort: str) -> Tuple[Optional[pd.DataFrame], str]:
    """
    Load and, when needed, enrich the table used for binary AUC analyses.

    Critical fix:
    Some ADRC ablation folders contain a file named
    *_metadata_all_with_predictions.csv that has metadata targets but NO cBAG
    prediction column. If that table is used as the AUC base, APOE/sex/cognition
    targets appear present but every AUC cell is still blank because the score
    column is missing.

    Therefore, choose the AUC base table by cBAG availability first:
      1) enriched Table 1 status metadata, when available
      2) full_cohort_predictions, when enabled
      3) metadata_all_with_predictions, only if it actually contains cBAG
      4) OOF validation table
      5) merged screening table

    Then merge phenotype metadata onto that cBAG-bearing base.
    """
    candidate_paths: List[Tuple[str, Path]] = []

    if BINARY_AUC_USE_FULL_COHORT_PREDICTIONS:
        # Preferred final source: Table 1 enriched metadata, which contains
        # full-cohort cBAG/BAG plus NORMCOG_01 and DEMENTIA_01.
        candidate_paths.append(("enriched_status_metadata", enriched_status_metadata_path(cohort, feature_set)))

        # Fallbacks if enriched files have not been generated.
        candidate_paths.append(("full_cohort_predictions", full_cohort_predictions_path(cohort, feature_set)))
        candidate_paths.append(("metadata_all_with_predictions", metadata_all_with_predictions_path(cohort, feature_set)))

    candidate_paths.append(("validation_oof", validation_path(cohort, feature_set)))
    candidate_paths.append(("merged_screening", MERGED_OUTDIR / f"merged_metadata_screening_{feature_set}_{cohort}.csv"))

    loaded: List[Tuple[str, Path, pd.DataFrame, Optional[str], int]] = []
    for label, path in candidate_paths:
        if not path.exists():
            continue
        try:
            cand = pd.read_csv(path, low_memory=False)
        except Exception:
            continue
        cbag_col = first_existing(cand, CBAG_PRIORITY)
        cbag_n = int(pd.to_numeric(cand[cbag_col], errors="coerce").notna().sum()) if cbag_col else 0
        loaded.append((label, path, cand, cbag_col, cbag_n))

    # Pick the first candidate with usable cBAG, respecting the priority above.
    chosen = next((item for item in loaded if item[4] > 0), None)

    # Fallback only for debugging: use the first loaded table, but report missing cBAG.
    if chosen is None:
        if loaded:
            label, path, df, cbag_col, cbag_n = loaded[0]
            source_parts = [f"{label}={path}", "WARNING=no usable cBAG column in any candidate"]
        else:
            # Last fallback to previous in-memory merged loader.
            df = load_merged(feature_set, cohort)
            if df is None:
                return None, ""
            source_parts = [f"load_merged={MERGED_OUTDIR / f'merged_metadata_screening_{feature_set}_{cohort}.csv'}"]
    else:
        label, path, df, cbag_col, cbag_n = chosen
        source_parts = [f"{label}={path}", f"cbag_col={cbag_col}", f"cbag_nonnull={cbag_n}"]

    # Always try metadata enrichment if target clues are missing/sparse. This is safe:
    # existing cBAG columns are preserved, and metadata columns are suffix-appended on collisions.
    clue_cols = _auc_target_clue_cols(df)
    clue_nonmissing = int(sum(pd.Series(df[c]).notna().sum() for c in clue_cols)) if clue_cols else 0
    needs_enrichment = (len(clue_cols) == 0) or (clue_nonmissing < max(10, len(df) // 4))

    if needs_enrichment or cohort == "ADRC":
        enriched, meta_source = _merge_auc_predictions_with_graph_metadata(df, cohort, feature_set)
        source_parts.append(f"metadata_enrichment={meta_source}")
        df = enriched

    return df, " + ".join(source_parts)


# ============================================================
# ADNI-ONLY APOE ENRICHMENT FOR BINARY AUC
# ============================================================

def _adni_rid_for_apoe_merge(x: object) -> Optional[str]:
    """Extract an ADNI RID as a plain integer string.

    Handles values such as 2, 2.0, PTID strings like 002_S_0295,
    and graph/connectome keys like R0295_y0. APOE is subject-level, so
    RID-only matching is sufficient and avoids visit-code mismatches.
    """
    if pd.isna(x):
        return None
    s = str(x).strip()
    if not s or s.lower() == "nan":
        return None

    # Numeric-looking RID values must be handled before regex parsing;
    # otherwise "2.0" becomes groups ["2", "0"] and incorrectly maps to 0.
    try:
        f = float(s)
        if np.isfinite(f) and f.is_integer():
            return str(int(f))
    except Exception:
        pass

    m = re.search(r"R(\d+)", s, flags=re.IGNORECASE)
    if m:
        return str(int(m.group(1)))

    m = re.search(r"_S_(\d+)", s, flags=re.IGNORECASE)
    if m:
        return str(int(m.group(1)))

    groups = re.findall(r"\d+", s)
    if groups:
        return str(int(groups[-1]))
    return None


def _add_adni_rid_apoe_key(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    candidate_cols = [
        "RID", "rid", "PTID", "ptid", "subject_id", "Subject", "subject",
        "connectome_key", "connectome_full_key", "graph_id", "regional_id",
        "match_id", "graph_subject_key", "node_feature_key",
    ]
    parts = []
    for c in candidate_cols:
        if c in out.columns:
            parts.append(out[c].map(_adni_rid_for_apoe_merge).astype(object))
    if parts:
        key = parts[0]
        for part in parts[1:]:
            key = key.where(key.notna(), part)
        out["_adni_apoe_rid_key"] = key
    else:
        out["_adni_apoe_rid_key"] = np.nan
    return out


def _merge_adni_apoe_from_updated_metadata(df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    """ADNI-only rescue merge for APOE4 AUC.

    The updated ADNI_metadata.xlsx contains APOE_A1/APOE_A2/genotype after the
    APOERES merge. Graph-builder aligned metadata may be older and may not have
    APOE, while sex/cognition can still be present. This function enriches only
    ADNI AUC input tables with APOE fields from the updated workbook.
    """
    meta_file = metadata_path("ADNI")
    if not meta_file.exists():
        return df, f"ADNI_APOE_merge_skipped_missing_metadata={meta_file}"

    try:
        meta = pd.read_excel(meta_file)
    except Exception as exc:
        return df, f"ADNI_APOE_merge_skipped_read_error={meta_file}: {exc}"

    apoe_payload_cols = [
        "APOE_A1", "APOE_A2", "genotype", "APOE4_strict_34_44_carrier",
        "APOE_merge_source", "APOE_merge_key",
    ]
    available_payload = [c for c in apoe_payload_cols if c in meta.columns]
    if not available_payload:
        return df, f"ADNI_APOE_merge_skipped_no_apoe_columns={meta_file}"

    base = _add_adni_rid_apoe_key(df)
    meta2 = _add_adni_rid_apoe_key(meta)

    bset = set(base["_adni_apoe_rid_key"].dropna().astype(str))
    mset = set(meta2["_adni_apoe_rid_key"].dropna().astype(str))
    overlap = len(bset.intersection(mset))
    if overlap == 0:
        return base, f"ADNI_APOE_merge_no_overlap={meta_file}"

    payload = meta2[["_adni_apoe_rid_key"] + available_payload].drop_duplicates("_adni_apoe_rid_key", keep="first")
    payload = payload.rename(columns={c: f"_adni_meta_{c}" for c in available_payload})
    merged = base.merge(payload, on="_adni_apoe_rid_key", how="left")

    # Fill or create top-level APOE columns expected by append_strict_apoe4_columns().
    for c in ["APOE_A1", "APOE_A2", "genotype", "APOE4_strict_34_44_carrier", "APOE_merge_source", "APOE_merge_key"]:
        mc = f"_adni_meta_{c}"
        if mc not in merged.columns:
            continue
        if c in merged.columns:
            merged[c] = merged[c].where(merged[c].notna(), merged[mc])
        else:
            merged[c] = merged[mc]

    n_genotype = int(pd.Series(merged.get("genotype", np.nan)).notna().sum())
    n_a1 = int(pd.Series(merged.get("APOE_A1", np.nan)).notna().sum())
    return merged, f"ADNI_APOE_merge={meta_file} | key=RID_only | overlap={overlap} | genotype_nonnull={n_genotype} | APOE_A1_nonnull={n_a1}"


def enrich_adni_apoe_for_auc_if_needed(df: pd.DataFrame, cohort: str, source_parts: Optional[List[str]] = None) -> Tuple[pd.DataFrame, str]:
    """Only fix ADNI APOE. Other cohorts are left untouched."""
    if cohort != "ADNI":
        return df, ""

    # If APOE4 already works, do nothing.
    test_df, test_source = append_strict_apoe4_columns(df)
    y = pd.to_numeric(test_df.get("AUC_APOE4_strict_34_44_carrier", np.nan), errors="coerce")
    if y.dropna().nunique() == 2:
        return test_df, f"ADNI_APOE_already_present={test_source}"

    enriched, msg = _merge_adni_apoe_from_updated_metadata(df)
    enriched, source = append_strict_apoe4_columns(enriched)
    y2 = pd.to_numeric(enriched.get("AUC_APOE4_strict_34_44_carrier", np.nan), errors="coerce")
    msg = f"{msg} | strict_source={source} | strict_target_n={int(y2.notna().sum())}"
    return enriched, msg

def best_fdr_by_cohort_family(assoc: pd.DataFrame) -> pd.DataFrame:
    df = assoc.copy()
    df = df[df["curated_family"].notna()].copy()
    df = df[pd.to_numeric(df["fdr_q_within_cohort"], errors="coerce") < FDR_THRESHOLD].copy()
    if df.empty:
        return df
    best = (
        df.sort_values(["cohort", "curated_family", "abs_pearson_r", "pearson_p"], ascending=[True, True, False, True])
        .groupby(["cohort", "curated_family"], as_index=False)
        .head(1)
        .copy()
    )
    best["curated_family_label"] = best["curated_family"].map(CURATED_LABELS)
    best["r2"] = best["pearson_r"] ** 2
    return best


def select_main_associations(assoc: pd.DataFrame) -> pd.DataFrame:
    selected_rows = []
    fdr = assoc[
        assoc["curated_family"].notna()
        & (pd.to_numeric(assoc["fdr_q_within_cohort"], errors="coerce") < FDR_THRESHOLD)
    ].copy()

    for cohort in COHORTS:
        for category, families in MAIN_COLUMNS.items():
            sub = fdr[(fdr["cohort"] == cohort) & (fdr["curated_family"].isin(families))].copy()
            selected_from = "FDR"
            if sub.empty and ALLOW_NONFDR_MAIN_FALLBACK:
                sub = assoc[(assoc["cohort"] == cohort) & (assoc["curated_family"].isin(families))].copy()
                selected_from = "non-FDR fallback"
            if sub.empty:
                continue
            sub = sub.sort_values(["abs_pearson_r", "pearson_p"], ascending=[False, True])
            row = sub.iloc[0].copy()
            row["main_category"] = category
            row["selected_from"] = selected_from
            row["r2"] = row["pearson_r"] ** 2
            selected_rows.append(row)

    out = pd.DataFrame(selected_rows)
    if not out.empty:
        out["curated_family_label"] = out["curated_family"].map(CURATED_LABELS)
    return out


def regression_ci(x: np.ndarray, y: np.ndarray, x_grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(x)
    x_mean = np.mean(x)
    sxx = np.sum((x - x_mean) ** 2)
    slope, intercept, _, _, _ = stats.linregress(x, y)
    y_grid = intercept + slope * x_grid
    y_hat = intercept + slope * x
    resid = y - y_hat
    dof = max(n - 2, 1)
    mse = np.sum(resid ** 2) / dof
    if sxx <= 0:
        se = np.full_like(x_grid, np.nan, dtype=float)
    else:
        se = np.sqrt(mse * (1.0 / n + (x_grid - x_mean) ** 2 / sxx))
    tcrit = stats.t.ppf(0.975, dof)
    return y_grid, y_grid - tcrit * se, y_grid + tcrit * se


def find_column(df: pd.DataFrame, col: str) -> Optional[str]:
    if col in df.columns:
        return col
    lower = {str(c).lower(): c for c in df.columns}
    if str(col).lower() in lower:
        return lower[str(col).lower()]
    return None


def build_plot_dataframe_with_cognitive_status(df: pd.DataFrame, x_col: str, cbag_col: str) -> Tuple[pd.DataFrame, Optional[str]]:
    """
    Build plotting dataframe for the association panel.

    The regression/statistics use all available x/cBAG pairs. If a cognitive
    impairment target can be inferred, points are coloured as:
        0 = cognitively normal / unimpaired
        1 = cognitively impaired / MCI / dementia / AD
    """
    plot_df = pd.DataFrame({
        "x": safe_numeric(df[x_col]),
        "y": safe_numeric(df[cbag_col]),
    })

    cog_source = None
    try:
        cog_target, cog_col = infer_cognitive_impairment_target(df)
    except Exception:
        cog_target, cog_col = None, None

    if cog_target is not None:
        plot_df["cog_impairment"] = pd.to_numeric(cog_target, errors="coerce")
        cog_source = cog_col
    else:
        plot_df["cog_impairment"] = np.nan

    plot_df = plot_df.replace([np.inf, -np.inf], np.nan).dropna(subset=["x", "y"])
    return plot_df, cog_source


def plot_association_panel(ax: plt.Axes, row: Optional[pd.Series], merged_cache: Dict[str, Optional[pd.DataFrame]], feature_set: str) -> None:
    if row is None or len(row) == 0:
        ax.axis("off")
        ax.text(0.5, 0.5, "No FDR-significant\nassociation", ha="center", va="center", fontsize=9)
        return

    cohort = str(row["cohort"])
    variable = str(row["variable"])
    fam = str(row["curated_family"])
    category = str(row.get("main_category", family_label(fam)))

    if cohort not in merged_cache:
        merged_cache[cohort] = load_merged(feature_set, cohort)
    df = merged_cache[cohort]

    if df is None:
        ax.axis("off")
        ax.text(0.5, 0.5, "Merged table missing\nRUN_SCREENING=True and\nSAVE_MERGED_TABLES=True", ha="center", va="center", fontsize=8)
        return

    cbag_col = first_existing(df, CBAG_PRIORITY)
    x_col = find_column(df, variable)
    if cbag_col is None or x_col is None:
        ax.axis("off")
        ax.text(0.5, 0.5, "Variable/cBAG\nnot found", ha="center", va="center", fontsize=8)
        return

    tmp, cog_source = build_plot_dataframe_with_cognitive_status(df, x_col, cbag_col)
    if len(tmp) < 10 or tmp["x"].nunique() < 2 or tmp["y"].nunique() < 2:
        ax.axis("off")
        ax.text(0.5, 0.5, f"Insufficient data\nn={len(tmp)}", ha="center", va="center", fontsize=8)
        return

    x = tmp["x"].to_numpy(dtype=float)
    y = tmp["y"].to_numpy(dtype=float)
    r, p = stats.pearsonr(x, y)
    r2 = r * r
    x_grid = np.linspace(np.min(x), np.max(x), 150)
    y_fit, y_low, y_high = regression_ci(x, y, x_grid)

    normal = tmp["cog_impairment"] == 0
    impaired = tmp["cog_impairment"] == 1
    unknown = ~(normal | impaired)

    if unknown.any():
        ax.scatter(
            tmp.loc[unknown, "x"], tmp.loc[unknown, "y"],
            s=14, alpha=0.45, linewidths=0,
            color=UNKNOWN_DOT_COLOR, label="Unknown status",
        )
    if normal.any():
        ax.scatter(
            tmp.loc[normal, "x"], tmp.loc[normal, "y"],
            s=15, alpha=0.72, linewidths=0,
            color=NORMAL_DOT_COLOR, label="Cognitively normal",
        )
    if impaired.any():
        ax.scatter(
            tmp.loc[impaired, "x"], tmp.loc[impaired, "y"],
            s=22, alpha=0.9, linewidths=0.25, edgecolors="black",
            color=IMPAIRED_DOT_COLOR, label="Cognitively impaired",
        )

    ax.plot(x_grid, y_fit, lw=1.4, color=REGRESSION_LINE_COLOR)
    ax.fill_between(x_grid, y_low, y_high, alpha=0.18, color=CI_BAND_COLOR)
    ax.axhline(0, lw=0.7, ls="--", alpha=0.5, color=REGRESSION_LINE_COLOR)

    q = row.get("fdr_q_within_cohort", np.nan)
    source = row.get("selected_from", "FDR")
    title = f"{cohort}: {category}\n{family_label(fam)} — {shorten(variable, 26)}"
    stat_line = f"n={len(tmp)}, r={r:.2f}, R²={r2:.2f}, {p_text(p)}, {q_text(q)}"
    if source != "FDR":
        stat_line += "\nnon-FDR fallback"

    ax.set_title(f"{title}\n{stat_line}", fontsize=8)
    ax.set_xlabel(shorten(variable, 28), fontsize=7)
    ax.set_ylabel("bias-corrected cBAG", fontsize=7)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.2)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, fontsize=5.8, frameon=False, loc="best", handletextpad=0.3, borderpad=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def make_main_figure(selected: pd.DataFrame, feature_set: str) -> None:
    categories = list(MAIN_COLUMNS.keys())
    fig, axes = plt.subplots(len(COHORTS), len(categories), figsize=(4.0 * len(categories), 3.2 * len(COHORTS)), squeeze=False)
    merged_cache: Dict[str, Optional[pd.DataFrame]] = {}

    for i, cohort in enumerate(COHORTS):
        for j, category in enumerate(categories):
            ax = axes[i, j]
            sub = selected[(selected["cohort"] == cohort) & (selected["main_category"] == category)] if not selected.empty else pd.DataFrame()
            row = sub.iloc[0] if not sub.empty else None
            plot_association_panel(ax, row, merged_cache, feature_set)
            if i == 0:
                ax.text(0.5, 1.36, category, transform=ax.transAxes, ha="center", va="bottom", fontsize=11, fontweight="bold")
            if j == 0:
                ax.text(-0.32, 0.5, cohort, transform=ax.transAxes, ha="right", va="center", rotation=90, fontsize=11, fontweight="bold")

    fig.suptitle(f"Figure 5. Biological validation of bias-corrected cBAG ({title_feature_set(feature_set)})", fontsize=15, y=0.998)
    fig.text(0.5, 0.01, "Each panel shows the strongest FDR-significant association within the indicated biological category and cohort. Points are coloured by cognitive status where available (green = cognitively normal, purple = cognitively impaired). Lines show linear regression fits with 95% confidence intervals.", ha="center", va="bottom", fontsize=9)
    fig.tight_layout(rect=[0.02, 0.035, 1, 0.965])

    FINAL_FIGURE_OUTDIR.mkdir(parents=True, exist_ok=True)
    stem = FINAL_FIGURE_OUTDIR / f"Figure5_Main_BiologicalValidation_{feature_set}"
    for fmt in FIGURE_FORMATS:
        path = stem.with_suffix(f".{fmt}")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"[INFO] Saved {path}")
    plt.close(fig)


def draw_heatmap_on_axis(ax: plt.Axes, mat: pd.DataFrame, ann: pd.DataFrame, title: str):
    if mat.empty or len(mat.index) == 0:
        ax.axis("off")
        ax.text(0.5, 0.5, "No FDR-significant\nfamilies", ha="center", va="center", fontsize=9)
        ax.set_title(title, fontsize=11)
        return None
    data = mat.to_numpy(dtype=float)
    vmax = max(0.05, np.nanmax(np.abs(data)) if np.isfinite(data).any() else 1.0)
    im = ax.imshow(data, aspect="auto", vmin=-vmax, vmax=vmax, cmap="coolwarm")
    ax.set_title(title, fontsize=11, pad=10)
    ax.set_xticks(np.arange(len(mat.columns)))
    ax.set_xticklabels(mat.columns.tolist(), fontsize=8)
    ax.set_yticks(np.arange(len(mat.index)))
    ax.set_yticklabels([family_label(f) for f in mat.index.tolist()], fontsize=8)
    for i, fam in enumerate(mat.index):
        for j, cohort in enumerate(mat.columns):
            txt = ann.loc[fam, cohort]
            if txt:
                ax.text(j, i, txt, ha="center", va="center", fontsize=5.8)
    ax.set_xticks(np.arange(-0.5, len(mat.columns), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(mat.index), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.0)
    ax.tick_params(which="minor", bottom=False, left=False)
    return im


def make_model_comparison_matrix(feature_sets: Sequence[str]) -> pd.DataFrame:
    matrix = pd.DataFrame(index=CURATED_FAMILIES, columns=feature_sets, dtype=float)
    for fs in feature_sets:
        try:
            assoc = load_assoc(fs)
        except FileNotFoundError:
            matrix[fs] = np.nan
            continue
        best = best_fdr_by_cohort_family(assoc)
        if best.empty:
            matrix[fs] = 0
            continue
        counts = best.groupby("curated_family")["cohort"].nunique()
        matrix[fs] = [counts.get(fam, 0) for fam in CURATED_FAMILIES]
    return matrix


def draw_count_heatmap(ax: plt.Axes, count_mat: pd.DataFrame, title: str):
    keep = count_mat.fillna(0).sum(axis=1) > 0
    mat = count_mat.loc[keep].copy()
    if mat.empty:
        ax.axis("off")
        ax.text(0.5, 0.5, "No model-comparison\nFDR signals", ha="center", va="center", fontsize=9)
        return None
    data = mat.to_numpy(dtype=float)
    im = ax.imshow(data, aspect="auto", vmin=0, vmax=max(1, np.nanmax(data)), cmap="Greys")
    ax.set_title(title, fontsize=11, pad=10)
    ax.set_xticks(np.arange(len(mat.columns)))
    ax.set_xticklabels([c.replace("_", "\n") for c in mat.columns], fontsize=7)
    ax.set_yticks(np.arange(len(mat.index)))
    ax.set_yticklabels([family_label(f) for f in mat.index], fontsize=8)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            if np.isfinite(val) and val > 0:
                ax.text(j, i, str(int(val)), ha="center", va="center", fontsize=8)
    ax.set_xticks(np.arange(-0.5, len(mat.columns), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(mat.index), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.0)
    ax.tick_params(which="minor", bottom=False, left=False)
    return im


def make_supplementary_figure(assoc: pd.DataFrame, feature_set: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    best_fdr = best_fdr_by_cohort_family(assoc)

    if not best_fdr.empty:
        family_summary = (
            best_fdr.groupby("curated_family")
            .agg(
                curated_family_label=("curated_family_label", "first"),
                n_significant_cohorts=("cohort", lambda x: len(set(x))),
                cohorts=("cohort", lambda x: ",".join(sorted(set(x)))),
                max_abs_r=("abs_pearson_r", "max"),
                mean_abs_r=("abs_pearson_r", "mean"),
                min_q=("fdr_q_within_cohort", "min"),
            )
            .reset_index()
            .sort_values(["n_significant_cohorts", "mean_abs_r"], ascending=[False, False])
        )
    else:
        family_summary = pd.DataFrame()

    any_families = [f for f in CURATED_FAMILIES if f in set(best_fdr.get("curated_family", []))]
    common_counts = best_fdr.groupby("curated_family")["cohort"].nunique() if not best_fdr.empty else pd.Series(dtype=float)
    common_families = [f for f in CURATED_FAMILIES if common_counts.get(f, 0) >= COMMON_MIN_COHORTS]

    mat_any, ann_any = matrix_from_best(best_fdr, any_families)
    mat_common, ann_common = matrix_from_best(best_fdr[best_fdr["curated_family"].isin(common_families)].copy(), common_families)
    count_mat = make_model_comparison_matrix(MODEL_COMPARISON_FEATURE_SETS)

    fig = plt.figure(figsize=(16, 11))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.35, 1.0], height_ratios=[1.0, 1.0])
    ax_a = fig.add_subplot(gs[:, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 1])

    im_a = draw_heatmap_on_axis(ax_a, mat_any, ann_any, f"A. FDR-significant curated families in ≥1 cohort\n({title_feature_set(feature_set)})")
    draw_heatmap_on_axis(ax_b, mat_common, ann_common, f"B. Common FDR-significant families in ≥{COMMON_MIN_COHORTS} cohorts")
    im_c = draw_count_heatmap(ax_c, count_mat, "C. Number of cohorts with FDR signal across models")

    if im_a is not None:
        cbar = fig.colorbar(im_a, ax=[ax_a, ax_b], fraction=0.025, pad=0.02)
        cbar.set_label("Signed Pearson r", fontsize=9)
        cbar.ax.tick_params(labelsize=8)
    if im_c is not None:
        cbar2 = fig.colorbar(im_c, ax=ax_c, fraction=0.045, pad=0.02)
        cbar2.set_label("No. cohorts", fontsize=9)
        cbar2.ax.tick_params(labelsize=8)

    fig.suptitle("Supplementary Figure S5. Curated metadata-wide screen of bias-corrected cBAG associations", fontsize=15, y=0.995)
    fig.text(0.5, 0.01, "Cells show the strongest FDR-significant association within each cohort-family pair. Empty cells indicate no variable in that family survived within-cohort FDR correction.", ha="center", va="bottom", fontsize=9)
    fig.tight_layout(rect=[0, 0.03, 1, 0.965])

    FINAL_FIGURE_OUTDIR.mkdir(parents=True, exist_ok=True)
    stem = FINAL_FIGURE_OUTDIR / f"FigureS5_CuratedMetadataScreen_{feature_set}"
    for fmt in FIGURE_FORMATS:
        path = stem.with_suffix(f".{fmt}")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"[INFO] Saved {path}")
    plt.close(fig)

    return family_summary, count_mat


def run_final_figures() -> None:
    FINAL_FIGURE_OUTDIR.mkdir(parents=True, exist_ok=True)

    # Save one shared model-comparison count matrix once.
    shared_count_mat = make_model_comparison_matrix(MODEL_COMPARISON_FEATURE_SETS)
    shared_count_path = FINAL_FIGURE_OUTDIR / "FigureS5_ModelComparison_FDR_CohortCounts.csv"
    shared_count_mat.to_csv(shared_count_path)
    print(f"[INFO] Saved {shared_count_path}")

    for feature_set in FINAL_FIGURE_FEATURE_SETS:
        print(f"\n[INFO] Building Figure 5 / Figure S5 for: {feature_set}")
        assoc = load_assoc(feature_set)

        selected = select_main_associations(assoc)
        selected_path = FINAL_FIGURE_OUTDIR / f"Figure5_Main_SelectedAssociations_{feature_set}.csv"
        selected.to_csv(selected_path, index=False)
        print(f"[INFO] Saved {selected_path}")

        make_main_figure(selected, feature_set)

        family_summary, count_mat = make_supplementary_figure(assoc, feature_set)
        family_summary_path = FINAL_FIGURE_OUTDIR / f"FigureS5_FDR_FamilySummary_{feature_set}.csv"
        count_path = FINAL_FIGURE_OUTDIR / f"FigureS5_ModelComparison_FDR_CohortCounts_{feature_set}.csv"
        family_summary.to_csv(family_summary_path, index=False)
        count_mat.to_csv(count_path)
        print(f"[INFO] Saved {family_summary_path}")
        print(f"[INFO] Saved {count_path}")

        methods_path = FINAL_FIGURE_OUTDIR / f"Figure5_S5_methods_note_{feature_set}.md"
        methods_path.write_text(
            "# Figure 5 / Supplementary Figure S5 methods note\n\n"
            "Biological validation analyses used bias-corrected cBAG only. For each cohort and model, "
            "the metadata-wide screen tested usable numeric variables from the merged validation, metadata, "
            "and harmonized cognitive-composite tables against cBAG using Pearson correlation. P-values were "
            "FDR-corrected within cohort. Variables were grouped into curated biological families. The main figure "
            "shows one representative FDR-significant association per cohort and biological category where available. "
            "The supplementary figure summarizes the full curated FDR-significant metadata-wide screen and model stability. "
            "Binary APOE4, sex, and cognitive-impairment AUC analyses used full-cohort inferred cBAG when available, "
            "and the strict APOE4 rule coded 3/4 and 4/4 as carriers, 2/2, 2/3, and 3/3 as non-carriers, while excluding 2/4. "
            "For ADNI, the enriched Figure 5 metadata CSV was used when available. Hippocampus and total-brain FA/volume readouts were extracted from the same ROI-wise node-feature tables "
            "used by the graph builder. Total-brain FA is the mean across all non-exterior FA ROIs; total-brain volume "
            "includes the sum across all non-exterior volume ROIs.\n"
        )
        print(f"[INFO] Saved {methods_path}")

# ============================================================
# BINARY AUC ANALYSIS: APOE4, SEX, COGNITIVE IMPAIRMENT
# ============================================================

def find_candidate_cols(df: pd.DataFrame, patterns: Sequence[str]) -> List[str]:
    cols = []
    for c in df.columns:
        low = normalize_name(c)
        if any(re.search(pat, low, flags=re.IGNORECASE) for pat in patterns):
            cols.append(c)
    return cols


def _normalize_apoe_genotype_value(x: object) -> object:
    """Normalize APOE genotype strings to 2_2, 2_3, 2_4, 3_3, 3_4, or 4_4."""
    if pd.isna(x):
        return np.nan
    raw = str(x).strip().upper()
    if raw == "" or raw in {"NAN", "NONE", "NA", "N/A"}:
        return np.nan

    s = raw.replace(" ", "")
    mapping = {
        "E2E2": "2_2", "E2E3": "2_3", "E2E4": "2_4",
        "E3E2": "2_3", "E3E3": "3_3", "E3E4": "3_4",
        "E4E2": "2_4", "E4E3": "3_4", "E4E4": "4_4",
        "APOE22": "2_2", "APOE23": "2_3", "APOE24": "2_4",
        "APOE32": "2_3", "APOE33": "3_3", "APOE34": "3_4",
        "APOE42": "2_4", "APOE43": "3_4", "APOE44": "4_4",
        "22": "2_2", "23": "2_3", "24": "2_4",
        "32": "2_3", "33": "3_3", "34": "3_4",
        "42": "2_4", "43": "3_4", "44": "4_4",
        "2/2": "2_2", "2/3": "2_3", "2/4": "2_4",
        "3/2": "2_3", "3/3": "3_3", "3/4": "3_4",
        "4/2": "2_4", "4/3": "3_4", "4/4": "4_4",
        "2_2": "2_2", "2_3": "2_3", "2_4": "2_4",
        "3_2": "2_3", "3_3": "3_3", "3_4": "3_4",
        "4_2": "2_4", "4_3": "3_4", "4_4": "4_4",
        "2-2": "2_2", "2-3": "2_3", "2-4": "2_4",
        "3-2": "2_3", "3-3": "3_3", "3-4": "3_4",
        "4-2": "2_4", "4-3": "3_4", "4-4": "4_4",
    }
    if s in mapping:
        return mapping[s]

    # Generic fallback: extract two APOE allele calls from strings like E3/E4, APOE e3 e4, etc.
    alleles = re.findall(r"(?:E|APOE)?([234])", s)
    if len(alleles) >= 2:
        a, b = sorted([alleles[0], alleles[1]])
        return f"{a}_{b}"

    return np.nan


def _strict_apoe4_from_genotype_norm(g: object) -> float:
    """Strict Figure 5 APOE4 carriage target; deliberately excludes APOE 2/4."""
    if pd.isna(g):
        return np.nan
    g = str(g)
    if g in {"3_4", "4_4"}:
        return 1.0
    if g in {"2_2", "2_3", "3_3"}:
        return 0.0
    if g == "2_4":
        return np.nan
    return np.nan


def _find_apoe_allele_pair_cols(df: pd.DataFrame) -> Optional[Tuple[str, str]]:
    cols = list(df.columns)
    norm_to_col = {normalize_name(c): c for c in cols}

    exact_pairs = [
        ("apoe_a1", "apoe_a2"),
        ("meta_apoe_a1", "meta_apoe_a2"),
        ("meta__apoe_a1", "meta__apoe_a2"),
        ("apoe1", "apoe2"),
        ("allele1", "allele2"),
        ("apoe_allele1", "apoe_allele2"),
        ("apoe_allele_1", "apoe_allele_2"),
    ]
    for a, b in exact_pairs:
        if a in norm_to_col and b in norm_to_col:
            return norm_to_col[a], norm_to_col[b]

    a_hits = [c for c in cols if re.search(r"apoe.*(?:a1|allele_?1)($|_)", normalize_name(c))]
    b_hits = [c for c in cols if re.search(r"apoe.*(?:a2|allele_?2)($|_)", normalize_name(c))]
    if a_hits and b_hits:
        return sorted(a_hits, key=lambda c: len(str(c)))[0], sorted(b_hits, key=lambda c: len(str(c)))[0]
    return None


def _strict_apoe4_from_allele_pair(df: pd.DataFrame) -> Tuple[Optional[pd.Series], Optional[pd.Series], Optional[str]]:
    pair = _find_apoe_allele_pair_cols(df)
    if pair is None:
        return None, None, None
    a_col, b_col = pair
    a = pd.to_numeric(df[a_col], errors="coerce")
    b = pd.to_numeric(df[b_col], errors="coerce")
    genotype_norm = pd.Series(np.nan, index=df.index, dtype=object)
    ok = a.isin([2, 3, 4]) & b.isin([2, 3, 4])
    lo = np.minimum(a[ok].astype(int), b[ok].astype(int)).astype(str)
    hi = np.maximum(a[ok].astype(int), b[ok].astype(int)).astype(str)
    genotype_norm.loc[ok] = lo + "_" + hi
    y = genotype_norm.map(_strict_apoe4_from_genotype_norm).astype(float)
    if y.dropna().nunique() == 2:
        return y, genotype_norm, f"{a_col}+{b_col}"
    return None, genotype_norm, f"{a_col}+{b_col}"


def _candidate_apoe_genotype_cols(df: pd.DataFrame) -> List[str]:
    preferred_patterns = [
        r"(^|_)genotype($|_)",
        r"(^|_)apoe($|_)",
        r"apoe_genotype",
        r"apoe4_genotype",
        r"apoe_e",
    ]
    cols = find_candidate_cols(df, preferred_patterns)

    # Keep string-like genotype columns. Avoid label-encoded numeric genotype columns; their codes
    # are not biologically interpretable and caused missing/misclassified ADRC APOE targets.
    out = []
    for c in cols:
        low = normalize_name(c)
        if "encoded" in low or "label" in low or "carrier" in low or "count" in low:
            continue
        s = df[c]
        if s.dtype == object or s.astype(str).str.contains(r"[/_\-]|E|APOE", case=False, na=False).any():
            out.append(c)

    # Prefer harmonized graph-builder genotype / raw APOE over broad matches.
    out = sorted(
        dict.fromkeys(out),
        key=lambda c: (
            0 if normalize_name(c) in {"genotype", "meta_genotype", "meta__genotype", "apoe", "meta_apoe", "meta__apoe"} else 1,
            0 if "apoe" in normalize_name(c) or "genotype" in normalize_name(c) else 1,
            len(str(c)),
        ),
    )
    return out


def append_strict_apoe4_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[str]]:
    """Append strict APOE4 columns used by the AUC analysis and return the source column."""
    out = df.copy()

    # 1) Best case: allele pair columns, e.g. ADNI APOE_A1/APOE_A2.
    y, genotype_norm, source = _strict_apoe4_from_allele_pair(out)
    if genotype_norm is not None:
        out["AUC_APOE_genotype_norm"] = genotype_norm
        out["AUC_APOE4_strict_34_44_carrier"] = y
        out["AUC_APOE4_rule"] = "1=3/4 or 4/4; 0=2/2,2/3,3/3; 2/4 excluded"
        out["AUC_APOE_source_col"] = source
        if y is not None and y.dropna().nunique() == 2:
            return out, source

    # 2) Genotype string columns, e.g. ADRC APOE = 2/4, 3/3, 4/4, 3/4, 2/3
    for col in _candidate_apoe_genotype_cols(out):
        genotype_norm = out[col].map(_normalize_apoe_genotype_value)
        y = genotype_norm.map(_strict_apoe4_from_genotype_norm).astype(float)
        if y.dropna().nunique() == 2:
            out["AUC_APOE_genotype_norm"] = genotype_norm
            out["AUC_APOE4_strict_34_44_carrier"] = y
            out["AUC_APOE4_rule"] = "1=3/4 or 4/4; 0=2/2,2/3,3/3; 2/4 excluded"
            out["AUC_APOE_source_col"] = col
            return out, col

    # 3) Last-resort explicit carrier/count columns. These cannot identify APOE 2/4, so only
    # use them when no interpretable genotype field exists.
    explicit_patterns = [
        r"apoe4_carrier", r"apoe4_count", r"apoe_4_count", r"e4_count",
        r"e4_carrier", r"apoe_e4", r"e4_status",
    ]
    for col in find_candidate_cols(out, explicit_patterns):
        s = out[col]
        if s.dtype == object:
            low = s.astype(str).str.lower().str.strip()
            y = pd.Series(np.nan, index=out.index, dtype=float)
            y[low.isin(["1", "1.0", "yes", "y", "true", "carrier", "positive", "pos", "e4+", "apoe4+"])] = 1
            y[low.isin(["0", "0.0", "no", "n", "false", "non-carrier", "noncarrier", "negative", "neg", "e4-", "apoe4-"])] = 0
        else:
            x = pd.to_numeric(s, errors="coerce")
            vals = set(x.dropna().unique())
            if vals.issubset({0, 1, 2}) and len(vals) >= 2:
                y = (x > 0).astype(float)
                y[x.isna()] = np.nan
            else:
                continue

        if y.dropna().nunique() == 2:
            out["AUC_APOE_genotype_norm"] = np.nan
            out["AUC_APOE4_strict_34_44_carrier"] = y
            out["AUC_APOE4_rule"] = "fallback explicit APOE4 carrier/count; 2/4 cannot be excluded from this source"
            out["AUC_APOE_source_col"] = col
            return out, col

    out["AUC_APOE_genotype_norm"] = out.get("AUC_APOE_genotype_norm", np.nan)
    out["AUC_APOE4_strict_34_44_carrier"] = out.get("AUC_APOE4_strict_34_44_carrier", np.nan)
    out["AUC_APOE4_rule"] = "target not found"
    out["AUC_APOE_source_col"] = ""
    return out, None


def save_auc_input_with_strict_apoe4(df: pd.DataFrame, feature_set: str, cohort: str, source_path: str) -> Path:
    outdir = BINARY_AUC_OUTDIR / "annotated_auc_input"
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"AUC_input_with_strict_APOE4_{feature_set}_{cohort}.csv"
    df.to_csv(path, index=False)
    print(f"[INFO] Saved APOE4-annotated AUC input: {path}")
    return path


def infer_apoe4_target(df: pd.DataFrame) -> Tuple[Optional[pd.Series], Optional[str]]:
    """
    Strict APOE4 target for binary AUC.

    Positive: APOE 3/4 or 4/4.
    Negative: APOE 2/2, 2/3, or 3/3.
    Excluded/missing: APOE 2/4 and any non-interpretable genotype.
    """
    if "AUC_APOE4_strict_34_44_carrier" in df.columns:
        y = pd.to_numeric(df["AUC_APOE4_strict_34_44_carrier"], errors="coerce")
        source = "AUC_APOE4_strict_34_44_carrier"
        if "AUC_APOE_source_col" in df.columns and df["AUC_APOE_source_col"].notna().any():
            source_vals = df["AUC_APOE_source_col"].dropna().astype(str).unique()
            source = f"AUC_APOE4_strict_34_44_carrier from {source_vals[0]}" if len(source_vals) else source
        if y.dropna().nunique() == 2:
            return y, source

    annotated, source = append_strict_apoe4_columns(df)
    y = pd.to_numeric(annotated["AUC_APOE4_strict_34_44_carrier"], errors="coerce")
    if source is not None and y.dropna().nunique() == 2:
        return y, source
    return None, None

def infer_sex_target(df: pd.DataFrame) -> Tuple[Optional[pd.Series], Optional[str]]:
    """
    Returns sex target:
        1 = female
        0 = male
    """
    patterns = [r"^sex$", r"sex_", r"_sex", r"gender"]
    for col in find_candidate_cols(df, patterns):
        s = df[col]

        if s.dtype == object:
            low = s.astype(str).str.lower().str.strip()
            y = pd.Series(np.nan, index=df.index, dtype=float)
            y[low.isin(["f", "female", "woman", "women"])] = 1
            y[low.isin(["m", "male", "man", "men"])] = 0
            if y.dropna().nunique() == 2:
                return y, col

        x = pd.to_numeric(s, errors="coerce")
        vals = set(x.dropna().unique())
        if vals.issubset({0, 1}) and len(vals) == 2:
            return x.astype(float), col
        if vals.issubset({1, 2}) and len(vals) == 2:
            y = (x == 2).astype(float)
            y[x.isna()] = np.nan
            return y, col

    return None, None


def infer_cognitive_impairment_target(df: pd.DataFrame) -> Tuple[Optional[pd.Series], Optional[str]]:
    """
    Returns cognitive impairment target:
        1 = impaired / MCI / dementia / AD / CDR>0
        0 = cognitively normal / control / CDR=0
    """

    def find_one(patterns: Sequence[str]) -> Optional[str]:
        hits = find_candidate_cols(df, patterns)
        if not hits:
            return None
        hits = sorted(
            hits,
            key=lambda c: (
                0 if str(c).lower().startswith("meta__") else 1,
                len(str(c)),
            ),
        )
        return hits[0]

    # Prefer harmonized NORMCOG if present.
    norm_col = find_one([r"(^|_)normcog($|_)", r"normcog"])
    if norm_col is not None:
        norm = pd.to_numeric(df[norm_col], errors="coerce")
        y = pd.Series(np.nan, index=df.index, dtype=float)
        y[norm == 1] = 0
        y[norm == 0] = 1
        if y.dropna().nunique() == 2:
            return y, norm_col

    # ADRC-specific rule using DEMENTED / IMPNOMCI if NORMCOG alone is not enough.
    dem_col = find_one([r"(^|_)demented($|_)", r"demented"])
    imp_col = find_one([r"(^|_)impnomci($|_)", r"impnomci"])

    if norm_col is not None or dem_col is not None or imp_col is not None:
        y = pd.Series(np.nan, index=df.index, dtype=float)

        if norm_col is not None:
            norm = pd.to_numeric(df[norm_col], errors="coerce")
            y[norm == 1] = 0
            y[norm == 0] = 1

        if dem_col is not None:
            dem = pd.to_numeric(df[dem_col], errors="coerce")
            y[dem == 1] = 1

        if imp_col is not None:
            imp = pd.to_numeric(df[imp_col], errors="coerce")
            y[imp == 1] = 1

        if y.dropna().nunique() == 2:
            used = "+".join([c for c in [norm_col, dem_col, imp_col] if c is not None])
            return y, used

    # AD_DECODE-specific Risk string column.
    risk_cols = find_candidate_cols(df, [r"(^|_)risk($|_)", r"risk"])
    risk_cols = [
        c for c in risk_cols
        if "risk_for_ad" not in normalize_name(c)
        and "cardio" not in normalize_name(c)
        and "vascular" not in normalize_name(c)
    ]

    for risk_col in risk_cols:
        low = df[risk_col].astype(str).str.lower().str.strip()
        y = pd.Series(np.nan, index=df.index, dtype=float)

        y[low.str.contains(r"\bmci\b|alzheimer|dement|^ad$|\bad\b|alz", na=False)] = 1
        y[low.str.contains(r"familial|control|normal|healthy|unimpaired|risk", na=False)] = 0

        if y.dropna().nunique() == 2:
            return y, risk_col

    # AD_DECODE-specific numeric risk_for_ad.
    risk_num_col = find_one([r"risk_for_ad", r"riskforad"])
    if risk_num_col is not None:
        x = pd.to_numeric(df[risk_num_col], errors="coerce")
        y = pd.Series(np.nan, index=df.index, dtype=float)

        y[x.isin([0, 1])] = 0
        y[x.isin([2, 3])] = 1

        if y.dropna().nunique() == 2:
            return y, risk_num_col

    # General diagnosis/status fallback.
    patterns = [
        r"diagnosis",
        r"(^|_)dx($|_)",
        r"cognitive_status",
        r"cog_status",
        r"clinical_status",
        r"diagnostic",
        r"clinical_group",
        r"(^|_)group($|_)",
        r"cdr_global",
        r"cdrglobal",
        r"(^|_)cdr($|_)",
        r"cdrsb",
    ]

    for col in find_candidate_cols(df, patterns):
        low_col = normalize_name(col)

        if (
            "cdr" not in low_col
            and any(bad in low_col for bad in ["score", "composite", "memory", "language", "executive", "processing"])
        ):
            continue

        s = df[col]

        if s.dtype == object:
            low = s.astype(str).str.lower().str.strip()
            y = pd.Series(np.nan, index=df.index, dtype=float)

            impaired = low.str.contains(
                r"\bmci\b|impair|dement|alzheimer|alz|^ad$|\bad\b|disease|patient|case|cdr>0|cdr_0_5|cdr0_5",
                na=False,
            )
            normal = (
                low.str.contains(
                    r"\bcn\b|control|normal|cognitively normal|healthy|unimpaired|cdr0|cdr=0",
                    na=False,
                )
                & ~impaired
            )

            y[impaired] = 1
            y[normal] = 0

            if y.dropna().nunique() == 2:
                return y, col

        x = pd.to_numeric(s, errors="coerce")

        if "cdr" in low_col and x.dropna().nunique() >= 2:
            y = (x > 0).astype(float)
            y[x.isna()] = np.nan
            if y.dropna().nunique() == 2:
                return y, col

        vals = set(x.dropna().unique())
        if vals.issubset({0, 1}) and len(vals) == 2:
            y = x.astype(float)
            return y, col

    return None, None


def auc_rank_stat(y_true: np.ndarray, scores: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores).astype(float)
    pos = y_true == 1
    neg = y_true == 0
    n_pos = pos.sum()
    n_neg = neg.sum()
    if n_pos == 0 or n_neg == 0:
        return np.nan
    ranks = stats.rankdata(scores)
    auc = (ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def roc_points(y_true: np.ndarray, scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores).astype(float)
    order = np.argsort(-scores)
    y_sorted = y_true[order]
    n_pos = np.sum(y_sorted == 1)
    n_neg = np.sum(y_sorted == 0)
    if n_pos == 0 or n_neg == 0:
        return np.array([0, 1]), np.array([0, 1])
    tps = np.cumsum(y_sorted == 1)
    fps = np.cumsum(y_sorted == 0)
    tpr = np.r_[0, tps / n_pos, 1]
    fpr = np.r_[0, fps / n_neg, 1]
    return fpr, tpr


def bootstrap_auc_ci(y_true: np.ndarray, scores: np.ndarray, n_boot: int, seed: int) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores).astype(float)
    n = len(y_true)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yb = y_true[idx]
        sb = scores[idx]
        if len(np.unique(yb)) < 2:
            continue
        aucs.append(auc_rank_stat(yb, sb))
    if len(aucs) < 20:
        return np.nan, np.nan
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def cohen_d_two_groups(y: np.ndarray, group: np.ndarray) -> float:
    y0 = y[group == 0]
    y1 = y[group == 1]
    if len(y0) < 2 or len(y1) < 2:
        return np.nan
    s0 = np.var(y0, ddof=1)
    s1 = np.var(y1, ddof=1)
    pooled = np.sqrt(((len(y0) - 1) * s0 + (len(y1) - 1) * s1) / (len(y0) + len(y1) - 2))
    if pooled == 0:
        return np.nan
    return float((np.mean(y1) - np.mean(y0)) / pooled)


def prepare_binary_targets(df: pd.DataFrame) -> Dict[str, Tuple[Optional[pd.Series], Optional[str], str]]:
    apoe_y, apoe_col = infer_apoe4_target(df)
    sex_y, sex_col = infer_sex_target(df)
    cog_y, cog_col = infer_cognitive_impairment_target(df)
    return {
        "APOE4_carriage": (apoe_y, apoe_col, "APOE4 carrier vs non-carrier"),
        "Sex": (sex_y, sex_col, "Female vs male / coded 1 vs 0"),
        "Cognitive_impairment": (cog_y, cog_col, "Impaired vs unimpaired"),
    }


def run_binary_auc_for_feature_set(feature_set: str) -> pd.DataFrame:
    rows = []
    roc_data = {}
    for cohort in COHORTS:
        print(f"[AUC] {feature_set} | {cohort}")
        df, auc_source_path = load_auc_dataframe(feature_set, cohort)
        if df is None:
            print(f"[WARN] Missing AUC table for {feature_set} | {cohort}")
            continue

        # ADNI-only APOE rescue: other cohorts already work and are left untouched.
        # This merges APOE_A1/APOE_A2/genotype from the updated ADNI_metadata.xlsx
        # when the selected AUC table has cBAG/sex/cognition but lacks APOE.
        df, adni_apoe_msg = enrich_adni_apoe_for_auc_if_needed(df, cohort)
        if adni_apoe_msg:
            auc_source_path = f"{auc_source_path} + {adni_apoe_msg}"

        # Append the strict APOE4 target to a copy of the AUC input table before
        # target inference. Rule: 1=3/4 or 4/4; 0=2/2,2/3,3/3; 2/4 excluded.
        df, apoe_source_col = append_strict_apoe4_columns(df)
        if BINARY_AUC_SAVE_ANNOTATED_METADATA:
            save_auc_input_with_strict_apoe4(df, feature_set, cohort, auc_source_path)

        cbag_col = first_existing(df, CBAG_PRIORITY)
        if cbag_col is None:
            print(f"[WARN] Missing bias-corrected cBAG for {feature_set} | {cohort}")
            rows.append({
                "feature_set": feature_set,
                "cohort": cohort,
                "outcome": "ALL",
                "target_col": "",
                "cbag_col": "",
                "status": "missing_cbag",
                "auc_source_path": auc_source_path,
            })
            continue
        cbag = safe_numeric(df[cbag_col])
        targets = prepare_binary_targets(df)
        for outcome, (target, target_col, outcome_label) in targets.items():
            if target is None or target_col is None:
                rows.append({"feature_set": feature_set, "cohort": cohort, "outcome": outcome, "outcome_label": outcome_label, "target_col": "", "cbag_col": cbag_col, "status": "target_not_found", "auc_source_path": auc_source_path})
                continue
            tmp = pd.DataFrame({"cbag": cbag, "target": pd.to_numeric(target, errors="coerce")}).replace([np.inf, -np.inf], np.nan).dropna()
            if tmp["target"].nunique() != 2:
                rows.append({"feature_set": feature_set, "cohort": cohort, "outcome": outcome, "outcome_label": outcome_label, "target_col": target_col, "cbag_col": cbag_col, "status": "target_not_binary", "n": len(tmp), "auc_source_path": auc_source_path})
                continue
            y = tmp["target"].astype(int).to_numpy()
            score_raw = tmp["cbag"].to_numpy(dtype=float)
            n0 = int(np.sum(y == 0))
            n1 = int(np.sum(y == 1))
            if n0 < MIN_AUC_GROUP_N or n1 < MIN_AUC_GROUP_N:
                rows.append({"feature_set": feature_set, "cohort": cohort, "outcome": outcome, "outcome_label": outcome_label, "target_col": target_col, "cbag_col": cbag_col, "status": "insufficient_group_size", "n": len(tmp), "n_0": n0, "n_1": n1, "auc_source_path": auc_source_path})
                continue
            auc_raw = auc_rank_stat(y, score_raw)
            if auc_raw < 0.5:
                score_plot = -score_raw
                auc_oriented = 1.0 - auc_raw
                direction = "higher cBAG predicts class 0; ROC plotted with -cBAG"
            else:
                score_plot = score_raw
                auc_oriented = auc_raw
                direction = "higher cBAG predicts class 1"
            ci_low, ci_high = bootstrap_auc_ci(y, score_plot, BOOTSTRAP_N, BOOTSTRAP_SEED)
            fpr, tpr = roc_points(y, score_plot)
            try:
                mw = stats.mannwhitneyu(score_raw[y == 1], score_raw[y == 0], alternative="two-sided")
                auc_p = float(mw.pvalue)
            except Exception:
                auc_p = np.nan
            try:
                ttest = stats.ttest_ind(score_raw[y == 1], score_raw[y == 0], equal_var=False, nan_policy="omit")
                group_p = float(ttest.pvalue)
            except Exception:
                group_p = np.nan
            d = cohen_d_two_groups(score_raw, y)
            rows.append({
                "feature_set": feature_set, "cohort": cohort, "outcome": outcome, "outcome_label": outcome_label,
                "target_col": target_col, "cbag_col": cbag_col, "status": "ok", "n": len(tmp), "n_0": n0, "n_1": n1,
                "mean_cbag_0": float(np.mean(score_raw[y == 0])), "mean_cbag_1": float(np.mean(score_raw[y == 1])),
                "sd_cbag_0": float(np.std(score_raw[y == 0], ddof=1)), "sd_cbag_1": float(np.std(score_raw[y == 1], ddof=1)),
                "cohen_d_class1_minus_class0": d, "group_ttest_p": group_p,
                "auc_raw_class1_vs_class0": auc_raw, "auc_oriented": auc_oriented,
                "auc_ci_low": ci_low, "auc_ci_high": ci_high, "auc_mannwhitney_p": auc_p, "auc_direction": direction,
                "auc_source_path": auc_source_path,
                "small_sample_flag": bool(min(n0, n1) < 10),
            })
            roc_data[(cohort, outcome)] = {"fpr": fpr, "tpr": tpr, "auc": auc_oriented, "ci_low": ci_low, "ci_high": ci_high, "n": len(tmp), "n0": n0, "n1": n1, "target_col": target_col, "direction": direction}

    debug_rows = []
    for r in rows:
        debug_rows.append({
            "feature_set": r.get("feature_set"),
            "cohort": r.get("cohort"),
            "outcome": r.get("outcome"),
            "target_col": r.get("target_col"),
            "cbag_col": r.get("cbag_col"),
            "status": r.get("status"),
            "n": r.get("n", np.nan),
            "n_0": r.get("n_0", np.nan),
            "n_1": r.get("n_1", np.nan),
            "auc_source_path": r.get("auc_source_path", ""),
        })

    debug_df = pd.DataFrame(debug_rows)
    BINARY_AUC_OUTDIR.mkdir(parents=True, exist_ok=True)
    debug_path = BINARY_AUC_OUTDIR / f"BinaryAUC_target_debug_{feature_set}.csv"
    debug_df.to_csv(debug_path, index=False)
    print(f"[INFO] Saved {debug_path}")

    stats_df = pd.DataFrame(rows)
    stats_path = BINARY_AUC_OUTDIR / f"BinaryAUC_stats_{feature_set}.csv"
    stats_df.to_csv(stats_path, index=False)
    print(f"[INFO] Saved {stats_path}")
    plot_binary_auc_roc_grid(roc_data, feature_set)
    plot_binary_auc_summary_heatmap(stats_df, feature_set)
    return stats_df


def plot_binary_auc_roc_grid(roc_data: Dict[Tuple[str, str], Dict[str, object]], feature_set: str) -> None:
    outcomes = ["APOE4_carriage", "Sex", "Cognitive_impairment"]
    outcome_titles = {
        "APOE4_carriage": "APOE4 carriage",
        "Sex": "Sex",
        "Cognitive_impairment": "Cognitive impairment",
    }

    fig, axes = plt.subplots(
        len(COHORTS),
        len(outcomes),
        figsize=(4.0 * len(outcomes), 3.2 * len(COHORTS)),
        squeeze=False,
    )

    for i, cohort in enumerate(COHORTS):
        for j, outcome in enumerate(outcomes):
            ax = axes[i, j]
            key = (cohort, outcome)

            if key not in roc_data:
                ax.axis("off")
                ax.text(
                    0.5, 0.5,
                    "Insufficient data\nor target missing",
                    ha="center",
                    va="center",
                    fontsize=8,
                )
            else:
                d = roc_data[key]
                ax.plot(d["fpr"], d["tpr"], lw=1.5)
                ax.plot([0, 1], [0, 1], ls="--", lw=0.8, alpha=0.6)
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.set_xlabel("False positive rate", fontsize=8)
                ax.set_ylabel("True positive rate", fontsize=8)
                ax.grid(True, alpha=0.2)
                ax.set_title(
                    f"{cohort}: {outcome_titles[outcome]}\n"
                    f"AUC={d['auc']:.2f} [{d['ci_low']:.2f}, {d['ci_high']:.2f}], n={d['n']}",
                    fontsize=8,
                )

            if i == 0:
                ax.text(
                    0.5, 1.24,
                    outcome_titles[outcome],
                    transform=ax.transAxes,
                    ha="center",
                    va="bottom",
                    fontsize=11,
                    fontweight="bold",
                )

            if j == 0:
                ax.text(
                    -0.28, 0.5,
                    cohort,
                    transform=ax.transAxes,
                    ha="right",
                    va="center",
                    rotation=90,
                    fontsize=11,
                    fontweight="bold",
                )

    fig.suptitle(
        f"Binary discrimination by bias-corrected cBAG ({title_feature_set(feature_set)})",
        fontsize=14,
        y=0.995,
    )
    fig.tight_layout(rect=[0.02, 0.02, 1, 0.96])

    stem = BINARY_AUC_OUTDIR / f"BinaryAUC_ROC_{feature_set}"
    for fmt in FIGURE_FORMATS:
        path = stem.with_suffix(f".{fmt}")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"[INFO] Saved {path}")
    plt.close(fig)


def plot_binary_auc_summary_heatmap(stats_df: pd.DataFrame, feature_set: str) -> None:
    if stats_df.empty or "status" not in stats_df.columns:
        print(f"[WARN] No valid AUC results for {feature_set}")
        return

    ok = stats_df[stats_df["status"] == "ok"].copy()
    if ok.empty:
        print(f"[WARN] No valid AUC results for {feature_set}")
        return

    outcomes = ["APOE4_carriage", "Sex", "Cognitive_impairment"]
    mat = pd.DataFrame(index=outcomes, columns=COHORTS, dtype=float)
    ann = pd.DataFrame("", index=outcomes, columns=COHORTS)

    for _, r in ok.iterrows():
        outcome = r["outcome"]
        cohort = r["cohort"]
        if outcome not in mat.index or cohort not in mat.columns:
            continue
        mat.loc[outcome, cohort] = r["auc_oriented"]
        small = "*" if bool(r.get("small_sample_flag", False)) else ""
        ann.loc[outcome, cohort] = (
            f"AUC={r['auc_oriented']:.2f}{small}\n"
            f"[{r['auc_ci_low']:.2f}, {r['auc_ci_high']:.2f}]\n"
            f"n={int(r['n'])}"
        )

    labels = {
        "APOE4_carriage": "APOE4 carriage",
        "Sex": "Sex",
        "Cognitive_impairment": "Cognitive impairment",
    }

    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    data = mat.to_numpy(dtype=float)
    im = ax.imshow(data, aspect="auto", vmin=0.5, vmax=1.0, cmap="Greys")

    ax.set_title(f"cBAG binary discrimination summary ({title_feature_set(feature_set)})", fontsize=12, pad=12)
    ax.set_xticks(np.arange(len(COHORTS)))
    ax.set_xticklabels(COHORTS, fontsize=9)
    ax.set_yticks(np.arange(len(outcomes)))
    ax.set_yticklabels([labels[o] for o in outcomes], fontsize=9)

    for i, outcome in enumerate(outcomes):
        for j, cohort in enumerate(COHORTS):
            txt = ann.loc[outcome, cohort]
            if txt:
                ax.text(j, i, txt, ha="center", va="center", fontsize=7)

    ax.set_xticks(np.arange(-0.5, len(COHORTS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(outcomes), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.0)
    ax.tick_params(which="minor", bottom=False, left=False)

    cbar = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
    cbar.set_label("Oriented AUC", fontsize=9)

    fig.tight_layout()
    stem = BINARY_AUC_OUTDIR / f"BinaryAUC_SummaryHeatmap_{feature_set}"
    for fmt in FIGURE_FORMATS:
        path = stem.with_suffix(f".{fmt}")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"[INFO] Saved {path}")
    plt.close(fig)


def run_binary_auc_analysis() -> None:
    BINARY_AUC_OUTDIR.mkdir(parents=True, exist_ok=True)
    all_stats = []
    for fs in BINARY_AUC_FEATURE_SETS:
        stats_df = run_binary_auc_for_feature_set(fs)
        if not stats_df.empty:
            all_stats.append(stats_df)
    all_df = pd.concat(all_stats, ignore_index=True, sort=False) if all_stats else pd.DataFrame()
    all_path = BINARY_AUC_OUTDIR / "BinaryAUC_stats_all_feature_sets.csv"
    all_df.to_csv(all_path, index=False)
    print(f"[INFO] Saved {all_path}")


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    print("=" * 100)
    print("ALL-IN-ONE cBAG BIOLOGICAL VALIDATION PIPELINE")
    print("=" * 100)
    print("BASE_DIR:", BASE_DIR)
    print("SCREENING_OUTDIR:", SCREENING_OUTDIR)
    print("FINAL_FIGURE_OUTDIR:", FINAL_FIGURE_OUTDIR)
    print("REGIONAL_STATS_OUTDIR:", REGIONAL_STATS_OUTDIR)
    print("RUN_SCREENING:", RUN_SCREENING)
    print("RUN_FINAL_FIGURES:", RUN_FINAL_FIGURES)
    print("RUN_BINARY_AUC_ANALYSIS:", RUN_BINARY_AUC_ANALYSIS)
    print("BINARY_AUC_USE_FULL_COHORT_PREDICTIONS:", BINARY_AUC_USE_FULL_COHORT_PREDICTIONS)
    print("RUN_BINARY_AUC_ALL_FEATURE_SETS:", RUN_BINARY_AUC_ALL_FEATURE_SETS)
    print("BINARY_AUC_FEATURE_SETS:", ", ".join(BINARY_AUC_FEATURE_SETS))
    print("MIN_AUC_GROUP_N:", MIN_AUC_GROUP_N)
    print("MAIN_FEATURE_SET:", MAIN_FEATURE_SET)
    print("FEATURE_SETS_TO_SCREEN:", ", ".join(FEATURE_SETS_TO_SCREEN))
    print("FINAL_FIGURE_FEATURE_SETS:", ", ".join(FINAL_FIGURE_FEATURE_SETS))
    print("FDR_THRESHOLD:", FDR_THRESHOLD)
    print("SAVE_MERGED_TABLES:", SAVE_MERGED_TABLES)

    if RUN_SCREENING:
        print("\n[STEP 1] Metadata-wide screening")
        run_screening()
    else:
        print("\n[SKIP] RUN_SCREENING=False")

    if RUN_FINAL_FIGURES:
        print("\n[STEP 2] Final Figure 5 and Supplementary Figure S5")
        run_final_figures()
    else:
        print("\n[SKIP] RUN_FINAL_FIGURES=False")

    if RUN_BINARY_AUC_ANALYSIS:
        print("\n[STEP 3] Binary AUC analyses: APOE4, sex, cognitive impairment")
        run_binary_auc_analysis()
    else:
        print("\n[SKIP] RUN_BINARY_AUC_ANALYSIS=False")

    print("\n" + "=" * 100)
    print("PIPELINE COMPLETE")
    print("=" * 100)
    print("Screening outputs:")
    print(SCREENING_OUTDIR)
    print("Final figure outputs:")
    print(FINAL_FIGURE_OUTDIR)
    print("Derived regional stats outputs:")
    print(REGIONAL_STATS_OUTDIR)
    print("Binary AUC outputs:")
    print(BINARY_AUC_OUTDIR)


if __name__ == "__main__":
    main()
