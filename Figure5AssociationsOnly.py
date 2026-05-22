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
- reads cohort metadata Excel files
- reads harmonized cognitive composite CSVs
- merges validation + metadata + cognitive composites
- screens all usable numeric variables against bias-corrected cBAG
- computes Pearson r, p-values, FDR q-values
- ranks variables by |r|, p-value, and FDR q-value
- saves all CSV outputs
- saves merged tables for audit and plotting
- creates curated family heatmaps
- creates final Figure 5 scatterplot panel
- creates Supplementary Figure S5 heatmap/model-comparison panel

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

SCREENING_OUTDIR = RESULTS_ROOT / "cbag_metadata_variable_screening"
FINAL_FIGURE_OUTDIR = SCREENING_OUTDIR / "final_figures"
MERGED_OUTDIR = SCREENING_OUTDIR / "merged_tables"


# ============================================================
# CONSTANTS
# ============================================================

RESULTS_DIR_MAP = {
    "ADNI": "BrainAgePredictionADNI_stratified_groupcv_targetnorm_bagbiascorr_oofglobal",
    "ADRC": "BrainAgePredictionADRC_stratified_groupcv_targetnorm_bagbiascorr_oofglobal",
    "HABS": "BrainAgePredictionHABS_stratified_groupcv_targetnorm_bagbiascorr_oofglobal",
    "AD_DECODE": "BrainAgePredictionADDECODE_stratified_groupcv_targetnorm_bagbiascorr_oofglobal",
}

# Only bias-corrected cBAG columns. Do not fall back to raw BAG.
CBAG_PRIORITY = [
    "cBAG_oof_global_raw_clean",
    "cBAG_oof_global",
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
    ("hippocampus", r"hippo|hippocampus|\bhc[_\s]"),
    ("brain_volume", r"brain[_\s]*volume|total[_\s]*brain|\btbv\b|\bicv\b|volume|volumetric|cortical[_\s]*thickness|thickness"),
    ("fa_diffusion", r"\bfa\b|fractional[_\s]*anisotropy|diffusion|dti|mean[_\s]*diffus|\bmd\b"),
    ("tau_ptau", r"ptau|p[_\s\-]*tau|tau|total[_\s]*tau|totaltau"),
    ("amyloid_abeta", r"abeta|aβ|amyloid|a_beta|ab42|ab40"),
    ("nfl", r"\bnfl\b|nfl_|nefl|neurofilament"),
    ("gfap", r"gfap"),
    ("apoe", r"apoe|e4|genotype"),
    ("memory", r"memory|sevlt|ravlt|avlt|logical|lm1|lm2|delayed|recall|bentd|story"),
    ("executive_function", r"executive|trail[_\s]*b|trailb|bckwds|backward|abstraction|set[_\s]*shift"),
    ("processing_speed", r"processing[_\s]*speed|trail[_\s]*a|traila|digit[_\s]*symbol|digitsymbol|symbol[_\s]*substitution|ufov"),
    ("language", r"language|fluency|fas|animals|animal|naming|wat|word[_\s]*accent|verbal[_\s]*flu"),
    ("visuospatial", r"visuospatial|benson|figure|copy|construction"),
    ("global_cognition_screening", r"moca|mocatots|mmse|cdr|cdrsb|cdglobal|adas|cognition|cognitive|global[_\s]*cog"),
    ("cardiovascular", r"blood[_\s]*pressure|\bsbp\b|\bdbp\b|pulse|chol|hdl|ldl|triglycer|glucose|hba1c|diabetes|hypertension|bmi|body[_\s]*mass|insulin|homa|egfr|creatinine"),
    ("depression_anxiety", r"depress|anxiety|\bgds\b|pswq|worry"),
]

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
    "path",
    "file",
    "filename",
    "metadata",
    "connectome_key",
    "connectome_full_key",
    "graph_path",
]

DEMOGRAPHIC_TOKENS = ["age", "sex", "gender", "educ", "education", "site", "scanner", "race", "ethnic"]


# ============================================================
# PATH HELPERS
# ============================================================

def metadata_path(cohort: str) -> Path:
    return BASE_DIR / "data" / "harmonization" / cohort / "metadata" / f"{cohort}_metadata.xlsx"


def composite_path(cohort: str) -> Path:
    return BASE_DIR / "results" / "composite_cognition" / f"{cohort}_cognitive_composites.csv"


def validation_path(cohort: str, feature_set: str) -> Path:
    return RESULTS_ROOT / RESULTS_DIR_MAP[cohort] / f"ablation_{feature_set}" / "validation_figures" / "subject_level_validation_input.csv"


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
# SCREENING
# ============================================================

def is_excluded_col(col: str) -> bool:
    low = normalize_name(col)
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
    best = (
        df.sort_values(["cohort", "curated_family", "abs_pearson_r", "pearson_p"], ascending=[True, True, False, True])
        .groupby(["cohort", "curated_family"], as_index=False)
        .head(1)
        .copy()
    )
    best["r2"] = best["pearson_r"] ** 2
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

        # Merge metadata.
        mpath = metadata_path(cohort)
        if mpath.exists():
            meta = pd.read_excel(mpath)
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

    tmp = pd.DataFrame({"x": safe_numeric(df[x_col]), "y": safe_numeric(df[cbag_col])})
    tmp = tmp.replace([np.inf, -np.inf], np.nan).dropna()
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

    ax.scatter(x, y, s=14, alpha=0.65, linewidths=0)
    ax.plot(x_grid, y_fit, lw=1.4)
    ax.fill_between(x_grid, y_low, y_high, alpha=0.18)
    ax.axhline(0, lw=0.7, ls="--", alpha=0.5)

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
    fig.text(0.5, 0.01, "Each panel shows the strongest FDR-significant association within the indicated biological category and cohort. Lines show linear regression fits with 95% confidence intervals.", ha="center", va="bottom", fontsize=9)
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
    assoc = load_assoc(MAIN_FEATURE_SET)

    selected = select_main_associations(assoc)
    selected_path = FINAL_FIGURE_OUTDIR / f"Figure5_Main_SelectedAssociations_{MAIN_FEATURE_SET}.csv"
    selected.to_csv(selected_path, index=False)
    print(f"[INFO] Saved {selected_path}")

    make_main_figure(selected, MAIN_FEATURE_SET)

    family_summary, count_mat = make_supplementary_figure(assoc, MAIN_FEATURE_SET)
    family_summary_path = FINAL_FIGURE_OUTDIR / f"FigureS5_FDR_FamilySummary_{MAIN_FEATURE_SET}.csv"
    count_path = FINAL_FIGURE_OUTDIR / "FigureS5_ModelComparison_FDR_CohortCounts.csv"
    family_summary.to_csv(family_summary_path, index=False)
    count_mat.to_csv(count_path)
    print(f"[INFO] Saved {family_summary_path}")
    print(f"[INFO] Saved {count_path}")

    methods_path = FINAL_FIGURE_OUTDIR / f"Figure5_S5_methods_note_{MAIN_FEATURE_SET}.md"
    methods_path.write_text(
        "# Figure 5 / Supplementary Figure S5 methods note\n\n"
        "Biological validation analyses used bias-corrected cBAG only. For each cohort and model, "
        "the metadata-wide screen tested usable numeric variables from the merged validation, metadata, "
        "and harmonized cognitive-composite tables against cBAG using Pearson correlation. P-values were "
        "FDR-corrected within cohort. Variables were grouped into curated biological families. The main figure "
        "shows one representative FDR-significant association per cohort and biological category where available. "
        "The supplementary figure summarizes the full curated FDR-significant metadata-wide screen and model stability.\n",
        encoding="utf-8",
    )
    print(f"[INFO] Saved {methods_path}")


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
    print("RUN_SCREENING:", RUN_SCREENING)
    print("RUN_FINAL_FIGURES:", RUN_FINAL_FIGURES)
    print("MAIN_FEATURE_SET:", MAIN_FEATURE_SET)
    print("FEATURE_SETS_TO_SCREEN:", ", ".join(FEATURE_SETS_TO_SCREEN))
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

    print("\n" + "=" * 100)
    print("PIPELINE COMPLETE")
    print("=" * 100)
    print("Screening outputs:")
    print(SCREENING_OUTDIR)
    print("Final figure outputs:")
    print(FINAL_FIGURE_OUTDIR)


if __name__ == "__main__":
    main()
