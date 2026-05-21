#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate Main Figure 5 and Supplementary Figure S5A-E for cBAG clinical,
biomarker, AUC, transcriptomic, and subgroup/cluster validation.

Main Figure 5:
    Imaging-only model, cohort-specific panels.

Supplementary Figure S5A-E:
    Same cohort-specific layout for all ablation models.

Cohort panel mapping inside each figure:
    A = ADNI
    B = ADRC
    C = HABS
    D = AD_DECODE

Feature-set mapping for supplementary figures:
    S5A = imaging_only
    S5B = imaging_demographics
    S5C = imaging_biomarkers
    S5D = full
    S5E = full_no_cardiovascular

Within each cohort panel, the script creates a 2 x 3 mini-grid:
    1) cBAG by cognitive/diagnostic status
    2) cBAG vs cognitive score/composite
    3) cBAG vs best available AD/plasma biomarker
       AD_DECODE fallback/preference: transcriptomic PCA summary if available
    4) APOE4 carriage ROC/AUC
    5) Cognitive impairment/status ROC/AUC
    6) Cluster/subgroup composition
       priority: cluster by cognitive status, cluster by APOE4, cluster by sex,
                 cluster distribution, then sex/cognitive/APOE4 composition.

Expected input files:
    <RESULTS_ROOT>/<BrainAgePrediction...>/ablation_<feature_set>/validation_figures/
        subject_level_validation_input.csv

Default BASE_DIR:
    /mnt/newStor/paros/paros_WORK/ines/results/
    BrainAgeValidation_AllCohorts_BAGBiasCorr_OOFGlobal_BiologicalValidation

Default output directory:
    <BASE_DIR>/Figure5_main_and_S5_supplement_clinical_biomarker_auc_clusters/

Run:
    python make_figure5_main_and_supp_clinical_biomarker_auc_clusters.py

Optional:
    python make_figure5_main_and_supp_clinical_biomarker_auc_clusters.py \
        --base-dir /path/to/BrainAgeValidation_AllCohorts_BAGBiasCorr_OOFGlobal_BiologicalValidation \
        --results-root /path/to/ines/results \
        --outdir /path/to/output
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import f_oneway, linregress, pearsonr, ttest_ind
from sklearn.metrics import roc_auc_score, roc_curve


BASE_DIR = (
    "/mnt/newStor/paros/paros_WORK/ines/results/"
    "BrainAgeValidation_AllCohorts_BAGBiasCorr_OOFGlobal_BiologicalValidation"
)

DEFAULT_OUTDIR_NAME = "Figure5_main_and_S5_supplement_clinical_biomarker_auc_clusters"
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

COG_STATUS_COLS = [
    "group_status",
    "NORMCOG",
    "DEMENTED",
    "Research Group",
    "Diagnosis",
    "DX",
    "DX_bl",
    "Diagnostic_Group",
    "Group",
    "cognitive_status",
    "Cognitive_Status",
    "Risk",
    "Risk_y",
    "clinical_status",
    "Clinical_Status",
]

COGNITION_PRIORITY = [
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
]

BIOMARKER_PRIORITY = [
    "PLASMA_PTAU217",
    "plasma_ptau217",
    "ptau217",
    "P_TAU217",
    "PTAU217",
    "PTAU",
    "ptau",
    "TAU",
    "tau_total",
    "ABETA42",
    "amyloid_42",
    "Aβ42",
    "ABETA40",
    "amyloid_40",
    "Aβ40",
    "ABETA42_40_RATIO",
    "ABETA42_ABETA40",
    "amyloid_42_40_ratio",
    "GFAP",
    "gfap",
    "NfL",
    "NFL",
    "nfl",
    "NEFL",
]

APOE_COLS = [
    "APOE4_Positivity_y",
    "APOE4_Positivity",
    "APOE4_carrier",
    "APOE4",
    "apoe4_carrier",
    "apoe4",
    "genotype",
    "APOE_genotype",
    "APOE",
    "APOE_y",
]

SEX_COLS = [
    "sex",
    "Sex",
    "SEX",
    "gender",
    "Gender",
    "PTGENDER",
    "PTSEX",
]

CLUSTER_COLS = [
    "cluster",
    "Cluster",
    "cluster_id",
    "Cluster_ID",
    "cluster_label",
    "Cluster_Label",
    "subtype",
    "Subtype",
    "subtype_label",
    "community",
    "Community",
    "module",
    "Module",
    "phenotype_cluster",
    "Phenotype_Cluster",
    "biotype",
    "Biotype",
]

TRANSCRIPTOMIC_PAT = re.compile(
    r"(transcriptomic|transcriptome|rna|gene).*pca|^pc(?:[1-9]|10)$|^pca[_ ]?(?:[1-9]|10)$",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate Main Figure 5 and Supplementary Figure S5A-E for cBAG clinical, "
            "biomarker, AUC, transcriptomic, and cluster/subgroup validation."
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
        help="Feature set for main Figure 5. Default: imaging_only",
    )
    parser.add_argument(
        "--supplement-feature-sets",
        default=",".join(DEFAULT_SUPPLEMENT_FEATURE_SETS),
        help=(
            "Comma-separated feature sets for supplementary S5 figures. "
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
        help="Minimum complete cases required for scatter/ROC panels. Default: 8",
    )
    parser.add_argument(
        "--min-group-n",
        type=int,
        default=3,
        help="Minimum observations per group for boxplots/composition. Default: 3",
    )
    return parser.parse_args()


def clean_numeric(series: pd.Series) -> pd.Series:
    out = pd.to_numeric(series, errors="coerce").copy()
    for val in SENTINEL_VALUES:
        out = out.mask(out == val, np.nan)
    return out


def first_existing(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def find_numeric_col(
    df: pd.DataFrame,
    candidates: Sequence[str],
    min_n: int = 8,
    avoid_keywords: Optional[Sequence[str]] = None,
) -> Optional[str]:
    avoid_keywords = [x.lower() for x in (avoid_keywords or [])]

    expanded: List[str] = []
    for c in candidates:
        expanded.extend([f"{c}_raw_clean", c])

    for col in expanded:
        if col in df.columns:
            low = str(col).lower()
            if any(bad in low for bad in avoid_keywords):
                continue
            if clean_numeric(df[col]).notna().sum() >= min_n:
                return col

    tokens: List[str] = []
    for c in candidates:
        tokens.extend([t for t in re.split(r"[_\s]+", str(c).lower()) if len(t) >= 3])
    tokens = sorted(set(tokens), key=len, reverse=True)

    for col in df.columns:
        low = str(col).lower()
        if any(bad in low for bad in avoid_keywords):
            continue
        if any(tok in low for tok in tokens):
            if clean_numeric(df[col]).notna().sum() >= min_n:
                return col

    return None


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
        "cBAG_oof_global": "cBAG",
        "Global_Cognition_Composite": "Global cognition composite",
        "Global_Cognition_Composite_resid": "Global cognition composite",
        "cognition_composite": "Cognition composite",
        "Memory_Composite": "Memory composite",
        "Executive_Function_Composite": "Executive function composite",
        "Processing_Speed_Composite": "Processing speed composite",
        "Language_Composite": "Language composite",
        "Visuospatial_Composite": "Visuospatial composite",
        "MMSE_total": "MMSE",
        "MOCA_total_corrected": "MoCA",
        "MOCA_total": "MoCA",
        "ADAS_total": "ADAS",
        "CDRSB": "CDR-SB",
        "CDGLOBAL": "CDR global",
        "PLASMA_PTAU217": "Plasma pTau217",
        "ptau217": "pTau217",
        "PTAU": "pTau",
        "ptau": "pTau",
        "TAU": "Total tau",
        "tau_total": "Total tau",
        "ABETA42": "Aβ42",
        "amyloid_42": "Aβ42",
        "ABETA40": "Aβ40",
        "amyloid_40": "Aβ40",
        "ABETA42_40_RATIO": "Aβ42/Aβ40",
        "GFAP": "GFAP",
        "gfap": "GFAP",
        "NfL": "NfL",
        "NFL": "NfL",
        "nfl": "NfL",
        "NEFL": "NfL",
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


def derive_apoe4(series: pd.Series) -> pd.Series:
    out = pd.Series(np.nan, index=series.index, dtype=float)

    numeric = pd.to_numeric(series, errors="coerce")
    out[numeric == 0] = 0
    out[numeric == 1] = 1
    out[numeric == 2] = 1

    text = series.astype(str).str.strip().str.upper().str.replace(" ", "", regex=False)

    out[text.isin([
        "0", "NO", "FALSE", "NEGATIVE", "N",
        "NONCARRIER", "NON-CARRIER", "E4-", "APOE4-",
    ])] = 0

    out[text.isin([
        "1", "YES", "TRUE", "POSITIVE", "Y",
        "CARRIER", "E4+", "APOE4+",
    ])] = 1

    genotype_like = text.str.contains(r"E?[234][/_]E?[234]|^[234][234]$", regex=True, na=False)
    out[genotype_like & text.str.contains("4", na=False)] = 1
    out[genotype_like & ~text.str.contains("4", na=False)] = 0

    return out


def derive_binary_cognitive_status(df: pd.DataFrame) -> Tuple[Optional[pd.Series], Optional[str]]:
    col = first_existing(df, COG_STATUS_COLS)
    if col is None:
        return None, None

    s = df[col]
    y = pd.Series(np.nan, index=df.index, dtype=float)
    numeric = pd.to_numeric(s, errors="coerce")

    if col.upper() == "NORMCOG":
        y[numeric == 1] = 0
        y[numeric == 0] = 1
        return y, col

    if col.upper() == "DEMENTED":
        y[numeric == 0] = 0
        y[numeric == 1] = 1
        return y, col

    text = s.astype(str).str.upper().str.strip()

    control_pat = r"\b(CN|CU|CONTROL|CONTROLS|HEALTHY|NORMAL|NORMCOG|NONDEMENTED|NON-DEMENTED)\b"
    impaired_pat = r"MCI|DEMENT|ALZ|IMPAIRED|RISK|PATIENT|\bAD\b|DEMENTIA"

    y[text.str.contains(control_pat, regex=True, na=False)] = 0
    y[text.str.contains(impaired_pat, regex=True, na=False)] = 1

    return y, col


def normalized_category(series: pd.Series) -> pd.Series:
    out = series.copy()
    out = out.astype(str).str.strip()
    out = out.replace({"nan": np.nan, "None": np.nan, "": np.nan, "NA": np.nan, "N/A": np.nan})
    return out


def scatter_with_fit(
    ax: plt.Axes,
    df: pd.DataFrame,
    x_col: Optional[str],
    title: str,
    min_n: int,
) -> Dict[str, object]:
    if x_col is None:
        empty_axis(ax, title, "Variable not found")
        return {"status": "missing", "x_col": "", "n": 0, "r": np.nan, "p": np.nan}

    tmp = pd.DataFrame({
        "x": clean_numeric(df[x_col]),
        "y": clean_numeric(df["_cbag"]),
    }).dropna()

    if len(tmp) < min_n:
        empty_axis(ax, title, f"Insufficient data\nn={len(tmp)}")
        return {"status": "insufficient_n", "x_col": x_col, "n": int(len(tmp)), "r": np.nan, "p": np.nan}

    if tmp["x"].nunique() < 2 or tmp["y"].nunique() < 2:
        empty_axis(ax, title, "Constant variable")
        return {"status": "constant", "x_col": x_col, "n": int(len(tmp)), "r": np.nan, "p": np.nan}

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


def boxplot_panel(
    ax: plt.Axes,
    df: pd.DataFrame,
    group_col: Optional[str],
    title: str,
    min_group_n: int,
    max_groups: int = 6,
) -> Dict[str, object]:
    if group_col is None:
        empty_axis(ax, title, "Grouping column not found")
        return {"status": "missing", "group_col": "", "n": 0, "p": np.nan}

    tmp = pd.DataFrame({
        "group": normalized_category(df[group_col]),
        "y": clean_numeric(df["_cbag"]),
    }).dropna()

    if tmp.empty:
        empty_axis(ax, title, "No complete cases")
        return {"status": "empty", "group_col": group_col, "n": 0, "p": np.nan}

    counts = tmp["group"].value_counts()
    keep = counts[counts >= min_group_n].index[:max_groups]
    tmp = tmp[tmp["group"].isin(keep)].copy()

    groups: List[np.ndarray] = []
    labels: List[str] = []

    for group_name, sub in tmp.groupby("group"):
        vals = sub["y"].dropna().values
        if len(vals) >= min_group_n:
            groups.append(vals)
            labels.append(str(group_name))

    if len(groups) < 2:
        empty_axis(ax, title, "Need ≥2 groups")
        return {"status": "insufficient_groups", "group_col": group_col, "n": int(len(tmp)), "p": np.nan}

    ax.boxplot(groups, labels=labels, showfliers=False)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=7)
    ax.set_ylabel("cBAG", fontsize=7)

    try:
        if len(groups) == 2:
            p = ttest_ind(groups[0], groups[1], equal_var=False, nan_policy="omit").pvalue
            test_label = "Welch"
        else:
            p = f_oneway(*groups).pvalue
            test_label = "ANOVA"
        ax.set_title(f"{title}\n{test_label} p={p:.2g}", fontsize=8)
    except Exception:
        p = np.nan
        test_label = ""
        ax.set_title(title, fontsize=8)

    ax.grid(True, axis="y", alpha=0.22)
    style_axis(ax)

    return {
        "status": "plotted",
        "group_col": group_col,
        "n": int(len(tmp)),
        "n_groups": len(groups),
        "groups": "|".join(labels),
        "test": test_label,
        "p": float(p) if pd.notna(p) else np.nan,
    }


def roc_panel(
    ax: plt.Axes,
    y_true: Optional[pd.Series],
    score: pd.Series,
    title: str,
    min_n: int,
) -> Dict[str, object]:
    if y_true is None:
        empty_axis(ax, title, "Binary target not found")
        return {"status": "missing", "n": 0, "auc": np.nan}

    tmp = pd.DataFrame({
        "y": pd.to_numeric(y_true, errors="coerce"),
        "score": clean_numeric(score),
    }).dropna()

    if len(tmp) < min_n:
        empty_axis(ax, title, f"Insufficient data\nn={len(tmp)}")
        return {"status": "insufficient_n", "n": int(len(tmp)), "auc": np.nan}

    unique = sorted(tmp["y"].astype(int).unique().tolist())
    if unique != [0, 1]:
        empty_axis(ax, title, f"Target not binary\nclasses={unique}")
        return {"status": "not_binary", "n": int(len(tmp)), "auc": np.nan}

    y = tmp["y"].astype(int)
    x = tmp["score"].astype(float)

    auc_raw = roc_auc_score(y, x)
    flipped = False
    if auc_raw < 0.5:
        x = -x
        auc = 1.0 - auc_raw
        flipped = True
    else:
        auc = auc_raw

    fpr, tpr, _ = roc_curve(y, x)

    ax.plot(fpr, tpr, linewidth=1.6)
    ax.plot([0, 1], [0, 1], linestyle=":", linewidth=0.9)
    ax.set_title(f"{title}\nAUC={auc:.2f}", fontsize=8)
    ax.set_xlabel("False positive rate", fontsize=7)
    ax.set_ylabel("True positive rate", fontsize=7)
    ax.grid(True, alpha=0.22)
    style_axis(ax)

    return {
        "status": "plotted",
        "n": int(len(tmp)),
        "n_negative": int((y == 0).sum()),
        "n_positive": int((y == 1).sum()),
        "auc": float(auc),
        "auc_raw": float(auc_raw),
        "score_flipped": bool(flipped),
    }


def transcriptomic_summary_panel(
    ax: plt.Axes,
    df: pd.DataFrame,
    title: str,
    min_n: int,
) -> Dict[str, object]:
    pca_cols = [
        c for c in df.columns
        if TRANSCRIPTOMIC_PAT.search(str(c)) and clean_numeric(df[c]).notna().sum() >= min_n
    ]

    rows: List[Tuple[str, float, float, int]] = []
    for col in pca_cols:
        tmp = pd.DataFrame({
            "x": clean_numeric(df[col]),
            "y": clean_numeric(df["_cbag"]),
        }).dropna()

        if len(tmp) >= min_n and tmp["x"].nunique() > 1 and tmp["y"].nunique() > 1:
            r, p = pearsonr(tmp["x"], tmp["y"])
            rows.append((col, float(r), float(p), int(len(tmp))))

    rows = sorted(rows, key=lambda z: abs(z[1]), reverse=True)[:6]

    if not rows:
        empty_axis(ax, title, "No transcriptomic PCA columns")
        return {"status": "missing", "n_components": 0}

    labels = [nice_label(row[0]) for row in rows]
    values = [row[1] for row in rows]

    ax.barh(range(len(rows)), values)
    ax.set_yticks(range(len(rows)), labels, fontsize=7)
    ax.axvline(0, linewidth=0.9)
    ax.invert_yaxis()
    ax.set_xlabel("Pearson r with cBAG", fontsize=7)
    ax.set_title(title, fontsize=8)
    ax.grid(True, axis="x", alpha=0.22)
    style_axis(ax)

    return {
        "status": "plotted",
        "n_components": len(rows),
        "top_component": rows[0][0],
        "top_r": rows[0][1],
        "top_p": rows[0][2],
        "top_n": rows[0][3],
        "components": "|".join([r[0] for r in rows]),
        "r_values": "|".join([f"{r[1]:.4f}" for r in rows]),
        "p_values": "|".join([f"{r[2]:.4g}" for r in rows]),
    }


def composition_panel(
    ax: plt.Axes,
    df: pd.DataFrame,
    cluster_col: Optional[str],
    stratify_col: Optional[str],
    title: str,
    min_group_n: int,
    max_cluster_levels: int = 6,
    max_strata_levels: int = 6,
) -> Dict[str, object]:
    if cluster_col is None:
        empty_axis(ax, title, "Cluster/subgroup column not found")
        return {"status": "missing", "cluster_col": "", "stratify_col": "", "n": 0}

    cluster = normalized_category(df[cluster_col])
    if stratify_col is None:
        tmp = pd.DataFrame({"cluster": cluster}).dropna()
        if tmp.empty or tmp["cluster"].nunique() < 2:
            empty_axis(ax, title, "Insufficient composition data")
            return {"status": "insufficient", "cluster_col": cluster_col, "stratify_col": "", "n": int(len(tmp))}

        counts = tmp["cluster"].value_counts()
        keep = counts[counts >= min_group_n].index[:max_cluster_levels]
        tmp["cluster"] = tmp["cluster"].where(tmp["cluster"].isin(keep), "Other")
        pct = tmp["cluster"].value_counts(normalize=True).sort_index() * 100
        pct.plot(kind="bar", ax=ax, width=0.75)
        ax.set_ylabel("%", fontsize=7)
        ax.set_xlabel("")
        ax.set_title(title, fontsize=8)
        ax.tick_params(axis="x", labelrotation=25, labelsize=7)
        ax.grid(True, axis="y", alpha=0.22)
        style_axis(ax)
        return {
            "status": "plotted",
            "cluster_col": cluster_col,
            "stratify_col": "",
            "n": int(len(tmp)),
            "levels": "|".join(pct.index.astype(str)),
        }

    strata = normalized_category(df[stratify_col])
    tmp = pd.DataFrame({
        "cluster": cluster,
        "stratum": strata,
    }).dropna()

    if tmp.empty or tmp["cluster"].nunique() < 2 or tmp["stratum"].nunique() < 2:
        empty_axis(ax, title, "Insufficient composition data")
        return {
            "status": "insufficient",
            "cluster_col": cluster_col,
            "stratify_col": stratify_col,
            "n": int(len(tmp)),
        }

    cluster_counts = tmp["cluster"].value_counts()
    cluster_keep = cluster_counts[cluster_counts >= min_group_n].index[:max_cluster_levels]
    tmp["cluster"] = tmp["cluster"].where(tmp["cluster"].isin(cluster_keep), "Other")

    strata_counts = tmp["stratum"].value_counts()
    strata_keep = strata_counts[strata_counts >= min_group_n].index[:max_strata_levels]
    tmp = tmp[tmp["stratum"].isin(strata_keep)].copy()

    if tmp.empty or tmp["cluster"].nunique() < 2 or tmp["stratum"].nunique() < 2:
        empty_axis(ax, title, "Insufficient composition data")
        return {
            "status": "insufficient_after_filter",
            "cluster_col": cluster_col,
            "stratify_col": stratify_col,
            "n": int(len(tmp)),
        }

    tab = pd.crosstab(tmp["stratum"], tmp["cluster"], normalize="index") * 100
    tab.plot(kind="bar", stacked=True, ax=ax, width=0.75)

    ax.set_ylabel("% within group", fontsize=7)
    ax.set_xlabel("")
    ax.set_title(title, fontsize=8)
    ax.tick_params(axis="x", labelrotation=25, labelsize=7)
    ax.legend(fontsize=6, bbox_to_anchor=(1.02, 1.0), loc="upper left", frameon=False)
    ax.grid(True, axis="y", alpha=0.22)
    style_axis(ax)

    return {
        "status": "plotted",
        "cluster_col": cluster_col,
        "stratify_col": stratify_col,
        "n": int(len(tmp)),
        "cluster_levels": "|".join(tab.columns.astype(str)),
        "strata_levels": "|".join(tab.index.astype(str)),
    }


def plot_best_composition_panel(
    ax: plt.Axes,
    df: pd.DataFrame,
    title: str,
    min_group_n: int,
) -> Dict[str, object]:
    cluster_col = first_existing(df, CLUSTER_COLS)

    if cluster_col is not None:
        cognitive_col = first_existing(df, COG_STATUS_COLS)
        apoe_col = first_existing(df, APOE_COLS)
        sex_col = first_existing(df, SEX_COLS)

        if cognitive_col is not None:
            return composition_panel(
                ax, df, cluster_col, cognitive_col,
                "Cluster composition by cognitive status",
                min_group_n=min_group_n,
            )

        if apoe_col is not None:
            tmp = df.copy()
            tmp["_APOE4_carrier"] = derive_apoe4(tmp[apoe_col]).map({0.0: "non-carrier", 1.0: "carrier"})
            return composition_panel(
                ax, tmp, cluster_col, "_APOE4_carrier",
                "Cluster composition by APOE4",
                min_group_n=min_group_n,
            )

        if sex_col is not None:
            return composition_panel(
                ax, df, cluster_col, sex_col,
                "Cluster composition by sex",
                min_group_n=min_group_n,
            )

        return composition_panel(
            ax, df, cluster_col, None,
            "Cluster composition",
            min_group_n=min_group_n,
        )

    cognitive_col = first_existing(df, COG_STATUS_COLS)
    apoe_col = first_existing(df, APOE_COLS)
    sex_col = first_existing(df, SEX_COLS)

    if cognitive_col is not None and apoe_col is not None:
        tmp = df.copy()
        tmp["_APOE4_carrier"] = derive_apoe4(tmp[apoe_col]).map({0.0: "non-carrier", 1.0: "carrier"})
        return composition_panel(
            ax, tmp, cognitive_col, "_APOE4_carrier",
            "Cognitive-status composition by APOE4",
            min_group_n=min_group_n,
        )

    if cognitive_col is not None and sex_col is not None:
        return composition_panel(
            ax, df, cognitive_col, sex_col,
            "Cognitive-status composition by sex",
            min_group_n=min_group_n,
        )

    if apoe_col is not None and sex_col is not None:
        tmp = df.copy()
        tmp["_APOE4_carrier"] = derive_apoe4(tmp[apoe_col]).map({0.0: "non-carrier", 1.0: "carrier"})
        return composition_panel(
            ax, tmp, "_APOE4_carrier", sex_col,
            "APOE4 composition by sex",
            min_group_n=min_group_n,
        )

    if cognitive_col is not None:
        return composition_panel(
            ax, df, cognitive_col, None,
            "Cognitive-status composition",
            min_group_n=min_group_n,
        )

    if sex_col is not None:
        return composition_panel(
            ax, df, sex_col, None,
            "Sex composition",
            min_group_n=min_group_n,
        )

    empty_axis(ax, title, "No cluster/subgroup composition data")
    return {"status": "missing", "cluster_col": "", "stratify_col": "", "n": 0}


def plot_cohort_block(
    fig: plt.Figure,
    outer_spec,
    df: pd.DataFrame,
    cohort: str,
    min_n: int,
    min_group_n: int,
) -> List[Dict[str, object]]:
    inner = outer_spec.subgridspec(2, 3, wspace=0.45, hspace=0.50)
    axes = [fig.add_subplot(inner[i, j]) for i in range(2) for j in range(3)]
    ax1, ax2, ax3, ax4, ax5, ax6 = axes

    cohort_header(ax1, cohort)

    results: List[Dict[str, object]] = []

    if df.empty:
        for slot, ax in zip(
            ["status_box", "cognition_scatter", "biomarker_or_transcriptomic", "apoe4_auc", "cognitive_auc", "composition"],
            axes,
        ):
            empty_axis(ax, slot, "No data for cohort")
            results.append({"cohort": cohort, "slot": slot, "status": "empty_cohort", "n": 0})
        return results

    status_col = first_existing(df, COG_STATUS_COLS)
    stats = boxplot_panel(
        ax1,
        df,
        status_col,
        "cBAG by cognitive/diagnostic status",
        min_group_n=min_group_n,
    )
    results.append({"cohort": cohort, "slot": "status_box", **stats})

    cognition_col = find_numeric_col(df, COGNITION_PRIORITY, min_n=min_n)
    stats = scatter_with_fit(
        ax2,
        df,
        cognition_col,
        f"cBAG vs {nice_label(cognition_col) if cognition_col else 'cognition'}",
        min_n=min_n,
    )
    results.append({"cohort": cohort, "slot": "cognition_scatter", **stats})

    if cohort == "AD_DECODE":
        pca_stats = transcriptomic_summary_panel(ax3, df, "Transcriptomic PCA summary", min_n=min_n)
        if pca_stats.get("status") == "plotted":
            results.append({"cohort": cohort, "slot": "transcriptomic_pca", **pca_stats})
        else:
            biomarker_col = find_numeric_col(df, BIOMARKER_PRIORITY, min_n=min_n)
            stats = scatter_with_fit(
                ax3,
                df,
                biomarker_col,
                f"cBAG vs {nice_label(biomarker_col) if biomarker_col else 'biomarker'}",
                min_n=min_n,
            )
            results.append({"cohort": cohort, "slot": "biomarker_scatter", **stats})
    else:
        biomarker_col = find_numeric_col(df, BIOMARKER_PRIORITY, min_n=min_n)
        stats = scatter_with_fit(
            ax3,
            df,
            biomarker_col,
            f"cBAG vs {nice_label(biomarker_col) if biomarker_col else 'biomarker'}",
            min_n=min_n,
        )
        results.append({"cohort": cohort, "slot": "biomarker_scatter", **stats})

    apoe_col = first_existing(df, APOE_COLS)
    apoe_y = derive_apoe4(df[apoe_col]) if apoe_col is not None else None
    stats = roc_panel(
        ax4,
        apoe_y,
        df["_cbag"],
        "APOE4 carriage ROC",
        min_n=min_n,
    )
    stats["target_col"] = apoe_col or ""
    results.append({"cohort": cohort, "slot": "apoe4_auc", **stats})

    cog_y, cog_target_col = derive_binary_cognitive_status(df)
    stats = roc_panel(
        ax5,
        cog_y,
        df["_cbag"],
        "Cognitive impairment ROC",
        min_n=min_n,
    )
    stats["target_col"] = cog_target_col or ""
    results.append({"cohort": cohort, "slot": "cognitive_status_auc", **stats})

    stats = plot_best_composition_panel(
        ax6,
        df,
        "Cluster/subgroup composition",
        min_group_n=min_group_n,
    )
    results.append({"cohort": cohort, "slot": "composition", **stats})

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
    min_group_n: int,
) -> Dict[str, object]:
    fig = plt.figure(figsize=(16, 12))
    outer = fig.add_gridspec(2, 2, wspace=0.18, hspace=0.25)

    stats_rows: List[Dict[str, object]] = []

    for idx, cohort in enumerate(cohorts):
        r, c = divmod(idx, 2)
        df_cohort = df[df["cohort"] == cohort].copy()
        cohort_stats = plot_cohort_block(
            fig,
            outer[r, c],
            df_cohort,
            cohort,
            min_n=min_n,
            min_group_n=min_group_n,
        )
        for row in cohort_stats:
            row["feature_set"] = feature_set
            row["figure"] = figure_prefix
        stats_rows.extend(cohort_stats)

    fig.suptitle(figure_title, fontsize=16, y=0.995)
    fig.text(
        0.5,
        0.005,
        (
            "Each cohort panel shows cBAG associations with clinical/cognitive status, cognition, "
            "biomarkers or transcriptomic PCA, APOE4/cognitive AUC, and cluster/subgroup composition."
        ),
        ha="center",
        fontsize=9,
    )

    outbase = outdir / f"{figure_prefix}_{feature_set}_cohort_panels_clinical_biomarker_auc_clusters"
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
        row[f"status_col_{cohort}"] = first_existing(dfc, COG_STATUS_COLS) or ""
        row[f"cognition_col_{cohort}"] = find_numeric_col(dfc, COGNITION_PRIORITY, min_n=3) or ""
        row[f"biomarker_col_{cohort}"] = find_numeric_col(dfc, BIOMARKER_PRIORITY, min_n=3) or ""
        row[f"apoe_col_{cohort}"] = first_existing(dfc, APOE_COLS) or ""
        row[f"sex_col_{cohort}"] = first_existing(dfc, SEX_COLS) or ""
        row[f"cluster_col_{cohort}"] = first_existing(dfc, CLUSTER_COLS) or ""

        pca_cols = [
            c for c in dfc.columns
            if TRANSCRIPTOMIC_PAT.search(str(c)) and clean_numeric(dfc[c]).notna().sum() >= 3
        ]
        row[f"transcriptomic_pca_cols_{cohort}"] = "|".join(pca_cols[:20])

    return row


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
        figure_prefix="Figure5",
        feature_set=args.main_feature_set,
        figure_title=(
            "Figure 5. Imaging-only model: cohort-specific clinical, biomarker, "
            "AUC, transcriptomic, and subgroup validation"
        ),
        formats=formats,
        dpi=args.dpi,
        min_n=args.min_n,
        min_group_n=args.min_group_n,
    )

    figure_manifest_rows.append({
        "figure": "Figure5",
        "feature_set": args.main_feature_set,
        "description": "Main manuscript Figure 5: imaging-only clinical/biomarker/AUC/cluster validation",
        "outputs": ";".join(main_result["outputs"]),
        "status": "saved",
    })
    association_stats_rows.extend(main_result["stats"])

    for feature_set in supplement_feature_sets:
        letter = MODEL_LETTERS.get(feature_set, "")
        figure_prefix = f"FigureS5{letter}" if letter else f"FigureS5_{feature_set}"

        df = load_subject_level_tables(results_root, cohorts, feature_set)
        if df.empty:
            print(f"[WARN] No usable data found for supplementary feature set: {feature_set}")
            figure_manifest_rows.append({
                "figure": figure_prefix,
                "feature_set": feature_set,
                "description": "Supplementary Figure S5 feature-set panel",
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
                f"Figure S5{letter}. {feature_set.replace('_', ' ')} model: "
                "cohort-specific clinical, biomarker, AUC, transcriptomic, and subgroup validation"
            ),
            formats=formats,
            dpi=args.dpi,
            min_n=args.min_n,
            min_group_n=args.min_group_n,
        )

        figure_manifest_rows.append({
            "figure": figure_prefix,
            "feature_set": feature_set,
            "description": "Supplementary Figure S5 feature-set panel",
            "outputs": ";".join(result["outputs"]),
            "status": "saved",
        })
        association_stats_rows.extend(result["stats"])

    figure_manifest = pd.DataFrame(figure_manifest_rows)
    association_stats = pd.DataFrame(association_stats_rows)
    dataset_summary = pd.DataFrame(dataset_summary_rows)

    manifest_path = outdir / "figure5_s5_generation_manifest.csv"
    stats_path = outdir / "figure5_s5_clinical_biomarker_auc_cluster_stats.csv"
    summary_path = outdir / "figure5_s5_dataset_summary.csv"

    figure_manifest.to_csv(manifest_path, index=False)
    association_stats.to_csv(stats_path, index=False)
    dataset_summary.to_csv(summary_path, index=False)

    print("\nSaved figures:")
    for row in figure_manifest_rows:
        print(f"  {row.get('figure', '')} ({row.get('feature_set', '')}): {row.get('outputs', '')}")

    print("\nSaved tables:")
    print(f"  Manifest: {manifest_path}")
    print(f"  Figure-specific stats: {stats_path}")
    print(f"  Dataset summary: {summary_path}")


if __name__ == "__main__":
    main()
