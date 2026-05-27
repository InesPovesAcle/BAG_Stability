#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build updated composite supplementary training-QC figure.

Changes versus the first QC script
----------------------------------
1. Panel C is changed:
   Raw BAG-age slope is calculated directly from each
   <prefix>_cv_oof_predictions.csv file using:
       BAG_raw ~ age_true
   This is more informative for QC because cBAG_oof_global is explicitly
   residualized against age and should have slope approximately zero by design.
   The corrected cBAG-age slope is still saved in the summary CSV.

2. Runtime is reported in minutes instead of hours.

3. Training and validation learning curves are both plotted:
   - Training loss
   - Validation loss proxy = validation RMSE^2
   These are QC curves on the training target scale used during model fitting.
   The training script stores val_rmse per epoch, not explicit val_loss.

4. Longitudinal availability is summarized from full-cohort prediction metadata:
   ADNI/HABS graph IDs are parsed into subject and visit labels.

Outputs
-------
$WORK/ines/results/Supplementary_TrainingQC_v3/
  Supplementary_Figure_training_QC_v3.png
  Supplementary_Figure_training_QC_v3.pdf
  Supplementary_training_QC_v3_model_summary.csv
  Supplementary_training_QC_v3_learning_curves.csv
  Supplementary_training_QC_v3_cv_fold_balance.csv
  Supplementary_training_QC_v3_longitudinal_visit_patterns.csv
  Supplementary_training_QC_v3_missing_inputs.csv
  Supplementary_training_QC_v3_README.md

Run
---
python make_supplementary_training_QC_figure_v3.py
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# SETTINGS
# =============================================================================

WORK = Path(os.environ.get("WORK", "/mnt/newStor/paros/paros_WORK"))
RESULTS_ROOT = WORK / "ines/results"
EXPERIMENT_TAG = "stratified_groupcv_targetnorm_bagbiascorr_oofglobal"

COHORT_DIRS = {
    "ADNI": "BrainAgePredictionADNI",
    "ADRC": "BrainAgePredictionADRC",
    "HABS": "BrainAgePredictionHABS",
    "AD_DECODE": "BrainAgePredictionADDECODE",
}

COHORT_PREFIX = {
    "ADNI": "adni",
    "ADRC": "adrc",
    "HABS": "habs",
    "AD_DECODE": "addecode",
}

FEATURE_SETS = [
    "imaging_only",
    "imaging_demographics",
    "imaging_biomarkers",
    "full",
    "full_no_cardiovascular",
]

FEATURE_LABELS = {
    "imaging_only": "Imaging",
    "imaging_demographics": "Imaging + demographics",
    "imaging_biomarkers": "Imaging + biomarkers",
    "full": "Full",
    "full_no_cardiovascular": "Full - cardiovascular",
}

# Keep learning curves readable by plotting one feature set. Change to "full"
# if preferred, or set PLOT_ONE_LEARNING_FEATURE_SET=False to overlay all.
LEARNING_CURVE_FEATURE_SET = "imaging_only"
PLOT_ONE_LEARNING_FEATURE_SET = True

OUTDIR = RESULTS_ROOT / "Supplementary_TrainingQC_v3"
OUTDIR.mkdir(parents=True, exist_ok=True)

OUT_FIG_PNG = OUTDIR / "Supplementary_Figure_training_QC_v3.png"
OUT_FIG_PDF = OUTDIR / "Supplementary_Figure_training_QC_v3.pdf"
OUT_SUMMARY = OUTDIR / "Supplementary_training_QC_v3_model_summary.csv"
OUT_LEARNING = OUTDIR / "Supplementary_training_QC_v3_learning_curves.csv"
OUT_FOLDS = OUTDIR / "Supplementary_training_QC_v3_cv_fold_balance.csv"
OUT_LONGITUDINAL = OUTDIR / "Supplementary_training_QC_v3_longitudinal_visit_patterns.csv"
OUT_MISSING = OUTDIR / "Supplementary_training_QC_v3_missing_inputs.csv"
OUT_README = OUTDIR / "Supplementary_training_QC_v3_README.md"

DPI = 300


# =============================================================================
# PATH AND IO HELPERS
# =============================================================================

def ablation_dir(cohort: str, feature_set: str) -> Path:
    return RESULTS_ROOT / f"{COHORT_DIRS[cohort]}_{EXPERIMENT_TAG}" / f"ablation_{feature_set}"


def prefix_for(cohort: str, feature_set: str) -> str:
    return f"{COHORT_PREFIX[cohort]}_{feature_set}"


def read_csv_if_exists(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception as exc:
        print(f"[WARN] Could not read {path}: {exc}")
        return None


def first_existing_col(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    if df is None or df.empty:
        return None
    lower = {str(c).lower(): c for c in df.columns}
    for c in candidates:
        if c in df.columns:
            return c
        if c.lower() in lower:
            return lower[c.lower()]
    return None


def parse_hms_to_seconds(x) -> float:
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    m = re.match(r"^(\d+):(\d+):(\d+(?:\.\d+)?)$", s)
    if not m:
        return np.nan
    h, mi, sec = m.groups()
    return int(h) * 3600 + int(mi) * 60 + float(sec)


def get_scalar_from_df(df: Optional[pd.DataFrame], candidates: list[str]) -> float:
    if df is None or df.empty:
        return np.nan
    col = first_existing_col(df, candidates)
    if col is None:
        return np.nan
    vals = pd.to_numeric(df[col], errors="coerce").dropna()
    return float(vals.iloc[0]) if len(vals) else np.nan


# =============================================================================
# METRIC EXTRACTION
# =============================================================================

def get_metric_from_summary(df: Optional[pd.DataFrame], metric: str, prefer: str = "GLOBAL") -> float:
    """
    Extract metric from either long-format summary or wide fallback.
    """
    if df is None or df.empty:
        return np.nan

    metric_col = first_existing_col(df, ["metric"])
    eval_col = first_existing_col(df, ["evaluation"])
    value_col = first_existing_col(df, ["point_estimate", "value", "mean", metric, metric.upper(), metric.lower()])

    if metric_col and value_col:
        sub = df[df[metric_col].astype(str).str.upper().eq(metric.upper())].copy()
        if eval_col and prefer:
            preferred = sub[sub[eval_col].astype(str).str.upper().str.contains(prefer.upper(), na=False)]
            if not preferred.empty:
                sub = preferred
        vals = pd.to_numeric(sub[value_col], errors="coerce").dropna()
        if len(vals):
            return float(vals.iloc[0])

    for c in [metric, metric.upper(), metric.lower(), f"{metric}_oof", f"OOF_{metric}"]:
        if c in df.columns:
            vals = pd.to_numeric(df[c], errors="coerce").dropna()
            if len(vals):
                return float(vals.iloc[0])

    return np.nan


def get_ci_width_from_bootstrap(df: Optional[pd.DataFrame], metric: str = "MAE") -> float:
    if df is None or df.empty:
        return np.nan
    metric_col = first_existing_col(df, ["metric"])
    low_col = first_existing_col(df, ["ci_low"])
    high_col = first_existing_col(df, ["ci_high"])
    if metric_col and low_col and high_col:
        sub = df[df[metric_col].astype(str).str.upper().eq(metric.upper())].copy()
        if not sub.empty:
            low = pd.to_numeric(sub[low_col], errors="coerce").dropna()
            high = pd.to_numeric(sub[high_col], errors="coerce").dropna()
            if len(low) and len(high):
                return float(high.iloc[0] - low.iloc[0])
    return np.nan


def extract_total_runtime_seconds(compute_df: Optional[pd.DataFrame]) -> float:
    if compute_df is None or compute_df.empty:
        return np.nan

    for c in [
        "timing.total_feature_set_elapsed_secs",
        "total_feature_set_elapsed_secs",
        "feature_elapsed_secs",
        "elapsed_secs",
        "runtime_secs",
    ]:
        if c in compute_df.columns:
            vals = pd.to_numeric(compute_df[c], errors="coerce").dropna()
            if len(vals):
                return float(vals.iloc[0])

    for c in [
        "timing.total_feature_set_elapsed_hms",
        "total_feature_set_elapsed_hms",
        "feature_elapsed_hms",
        "elapsed_hms",
    ]:
        if c in compute_df.columns:
            vals = compute_df[c].dropna()
            if len(vals):
                return parse_hms_to_seconds(vals.iloc[0])

    return np.nan


def compute_oof_age_dependence(oof: Optional[pd.DataFrame]) -> dict:
    """
    Compute age-dependence for raw BAG and corrected cBAG.

    Raw BAG age-dependence is useful for visualization because it shows the
    uncorrected age bias.

    Corrected cBAG age-dependence should be approximately zero when using
    cBAG_oof_global, because the training script defines cBAG_oof_global by
    residualizing pooled OOF BAG_raw with respect to age_true.
    """
    out = {
        "raw_BAG_age_slope": np.nan,
        "raw_BAG_age_r": np.nan,
        "corrected_cBAG_age_slope": np.nan,
        "corrected_cBAG_age_r": np.nan,
        "corrected_cBAG_column_used": "",
        "age_dependence_n": 0,
    }

    if oof is None or oof.empty:
        return out

    age_col = first_existing_col(oof, ["age_true", "Age", "age"])
    if age_col is None:
        return out

    age = pd.to_numeric(oof[age_col], errors="coerce")

    def slope_and_r(y):
        tmp = pd.DataFrame({"age": age, "y": pd.to_numeric(y, errors="coerce")}).dropna()
        if len(tmp) < 3 or tmp["age"].nunique() < 2 or tmp["y"].nunique() < 2:
            return np.nan, np.nan, int(len(tmp))
        slope, intercept = np.polyfit(tmp["age"].values, tmp["y"].values, 1)
        r = np.corrcoef(tmp["age"].values, tmp["y"].values)[0, 1]
        return float(slope), float(r), int(len(tmp))

    # Raw BAG: prefer explicit BAG_raw; otherwise compute from pred_raw - age_true.
    if "BAG_raw" in oof.columns:
        raw_y = oof["BAG_raw"]
    elif "pred_raw" in oof.columns:
        raw_y = pd.to_numeric(oof["pred_raw"], errors="coerce") - age
    else:
        raw_y = None

    if raw_y is not None:
        slope, r, n = slope_and_r(raw_y)
        out["raw_BAG_age_slope"] = slope
        out["raw_BAG_age_r"] = r
        out["age_dependence_n"] = n

    # Corrected cBAG: use the strongest age-residualized output if present.
    cbag_col = first_existing_col(
        oof,
        ["cBAG_oof_global", "cBAG_global", "cBAG", "cBAG_foldwise"]
    )
    if cbag_col is not None:
        slope, r, n = slope_and_r(oof[cbag_col])
        out["corrected_cBAG_age_slope"] = slope
        out["corrected_cBAG_age_r"] = r
        out["corrected_cBAG_column_used"] = cbag_col
        out["age_dependence_n"] = max(out["age_dependence_n"], n)

    return out


# =============================================================================
# LEARNING CURVES
# =============================================================================

def read_learning_histories(d: Path, prefix: str) -> Optional[pd.DataFrame]:
    """
    Prefer all-fold histories because they contain train_loss, val_mae, val_rmse.
    """
    path = d / "learning_curves" / f"{prefix}_all_fold_learning_histories.csv"
    hist = read_csv_if_exists(path)
    if hist is not None and not hist.empty:
        return hist

    # Fallback to summaries if all-fold histories are unavailable.
    train = read_csv_if_exists(d / "learning_curves" / f"{prefix}_learning_curve_train_loss_summary.csv")
    val_mae = read_csv_if_exists(d / "learning_curves" / f"{prefix}_learning_curve_val_mae_summary.csv")
    val_rmse = read_csv_if_exists(d / "learning_curves" / f"{prefix}_learning_curve_val_rmse_summary.csv")

    frames = []
    for name, df in [("train_loss", train), ("val_mae", val_mae), ("val_rmse", val_rmse)]:
        if df is None or df.empty:
            continue
        tmp = df.copy()
        tmp["summary_metric"] = name
        frames.append(tmp)

    if not frames:
        return None

    # Not as good as all-fold histories, but enough for plotting.
    return pd.concat(frames, ignore_index=True)


def summarize_learning_for_plot(hist: Optional[pd.DataFrame], cohort: str, feature_set: str) -> pd.DataFrame:
    if hist is None or hist.empty:
        return pd.DataFrame()

    # Case 1: all-fold histories.
    if {"epoch", "train_loss", "val_rmse"}.issubset(hist.columns):
        x = hist.copy()
        x["val_loss_proxy"] = pd.to_numeric(x["val_rmse"], errors="coerce") ** 2
        rows = []
        for metric in ["train_loss", "val_loss_proxy", "val_mae"]:
            if metric not in x.columns:
                continue
            g = (
                x.groupby("epoch")[metric]
                .agg(["mean", "std", "count"])
                .reset_index()
                .rename(columns={"count": "n"})
            )
            g["sem"] = g["std"].fillna(0) / np.sqrt(g["n"].clip(lower=1))
            g["ci95"] = 1.96 * g["sem"]
            g["metric"] = metric
            g["cohort"] = cohort
            g["feature_set"] = feature_set
            rows.append(g)
        return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

    # Case 2: summary fallback.
    if {"epoch", "mean", "summary_metric"}.issubset(hist.columns):
        rows = []
        for metric, sub in hist.groupby("summary_metric"):
            tmp = sub.copy()
            tmp["metric"] = "val_loss_proxy" if metric == "val_rmse" else metric
            if metric == "val_rmse":
                tmp["mean"] = pd.to_numeric(tmp["mean"], errors="coerce") ** 2
                if "ci95" in tmp.columns:
                    # approximate propagation; adequate for QC visualization only
                    tmp["ci95"] = 2 * np.sqrt(tmp["mean"].clip(lower=0)) * pd.to_numeric(tmp["ci95"], errors="coerce")
            tmp["cohort"] = cohort
            tmp["feature_set"] = feature_set
            rows.append(tmp)
        return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

    return pd.DataFrame()


# =============================================================================
# LONGITUDINAL / VISIT PATTERN SUMMARIES
# =============================================================================

def parse_subject_visit_from_graph_id(s: object) -> tuple[object, object]:
    if pd.isna(s):
        return np.nan, np.nan
    t = str(s).strip().replace("_Y", "_y")
    m = re.match(r"^([A-Za-z]\d+)_y(\d+(?:\.\d+)?)$", t)
    if m:
        return m.group(1).upper(), "y" + m.group(2)
    if "_y" in t:
        parts = t.split("_y")
        return parts[0].upper(), "y" + parts[1]
    return np.nan, np.nan


def make_visit_pattern_summary(cohort: str, feature_set: str, df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame([{
            "cohort": cohort, "feature_set": feature_set, "source": "missing",
            "n_rows": 0, "n_unique_graphs": 0, "n_unique_subjects": 0,
            "visit_pattern": "missing", "n_subjects": 0,
        }])

    id_col = first_existing_col(df, ["graph_id", "connectome_key", "subject_id", "regional_id"])
    if id_col is None:
        return pd.DataFrame([{
            "cohort": cohort, "feature_set": feature_set, "source": "no_id_column",
            "n_rows": len(df), "n_unique_graphs": np.nan, "n_unique_subjects": np.nan,
            "visit_pattern": "no_id_column", "n_subjects": np.nan,
        }])

    tmp = df[[id_col]].copy()
    tmp["graph_id_for_visit"] = tmp[id_col].astype(str)
    tmp[["parsed_subject", "parsed_visit"]] = tmp["graph_id_for_visit"].apply(
        lambda x: pd.Series(parse_subject_visit_from_graph_id(x))
    )

    valid = tmp.dropna(subset=["parsed_subject", "parsed_visit"]).drop_duplicates(["parsed_subject", "parsed_visit"])

    if valid.empty:
        return pd.DataFrame([{
            "cohort": cohort, "feature_set": feature_set, "source": id_col,
            "n_rows": len(df), "n_unique_graphs": tmp["graph_id_for_visit"].nunique(),
            "n_unique_subjects": np.nan,
            "visit_pattern": "not_longitudinal_keyed", "n_subjects": np.nan,
        }])

    subject_patterns = (
        valid.groupby("parsed_subject")["parsed_visit"]
        .apply(lambda x: ",".join(sorted(set(x), key=lambda z: float(z.replace("y", "")))))
        .reset_index(name="visit_pattern")
    )

    counts = (
        subject_patterns["visit_pattern"]
        .value_counts()
        .rename_axis("visit_pattern")
        .reset_index(name="n_subjects")
    )

    counts["cohort"] = cohort
    counts["feature_set"] = feature_set
    counts["source"] = id_col
    counts["n_rows"] = len(df)
    counts["n_unique_graphs"] = tmp["graph_id_for_visit"].nunique()
    counts["n_unique_subjects"] = subject_patterns["parsed_subject"].nunique()

    return counts[[
        "cohort", "feature_set", "source", "n_rows", "n_unique_graphs",
        "n_unique_subjects", "visit_pattern", "n_subjects"
    ]]


# =============================================================================
# COLLECTION
# =============================================================================

def collect_one_run(cohort: str, feature_set: str) -> tuple[dict, list[dict], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    d = ablation_dir(cohort, feature_set)
    prefix = prefix_for(cohort, feature_set)

    paths = {
        "cv_summary": d / f"{prefix}_cv_summary_metrics.csv",
        "cv_fold_bc": d / f"{prefix}_cv_fold_metrics_bias_corrected.csv",
        "oof": d / f"{prefix}_cv_oof_predictions.csv",
        "bootstrap": d / f"{prefix}_bootstrap_metric_summary.csv",
        "cv_split": d / f"{prefix}_cv_split_summary.csv",
        "full_summary": d / f"{prefix}_full_cohort_prediction_summary.csv",
        "full_metadata": d / f"{prefix}_metadata_all_with_predictions.csv",
        "full_pred": d / f"{prefix}_full_cohort_predictions.csv",
        "compute": d / f"{prefix}_training_compute_report.csv",
        "learning_histories": d / "learning_curves" / f"{prefix}_all_fold_learning_histories.csv",
    }

    missing = []
    for name, p in paths.items():
        if not p.exists():
            missing.append({
                "cohort": cohort,
                "feature_set": feature_set,
                "input_name": name,
                "missing_path": str(p),
            })

    cv_summary = read_csv_if_exists(paths["cv_summary"])
    cv_fold_bc = read_csv_if_exists(paths["cv_fold_bc"])
    oof = read_csv_if_exists(paths["oof"])
    bootstrap = read_csv_if_exists(paths["bootstrap"])
    cv_split = read_csv_if_exists(paths["cv_split"])
    full_summary = read_csv_if_exists(paths["full_summary"])
    full_metadata = read_csv_if_exists(paths["full_metadata"])
    full_pred = read_csv_if_exists(paths["full_pred"])
    compute = read_csv_if_exists(paths["compute"])
    histories = read_learning_histories(d, prefix)

    mae = get_metric_from_summary(cv_summary, "MAE", prefer="GLOBAL")
    if pd.isna(mae):
        mae = get_metric_from_summary(cv_summary, "MAE", prefer="BIAS")
    if pd.isna(mae) and cv_fold_bc is not None and "MAE" in cv_fold_bc.columns:
        mae = float(pd.to_numeric(cv_fold_bc["MAE"], errors="coerce").mean())

    rmse = get_metric_from_summary(cv_summary, "RMSE", prefer="GLOBAL")
    if pd.isna(rmse):
        rmse = get_metric_from_summary(cv_summary, "RMSE", prefer="BIAS")
    if pd.isna(rmse) and cv_fold_bc is not None and "RMSE" in cv_fold_bc.columns:
        rmse = float(pd.to_numeric(cv_fold_bc["RMSE"], errors="coerce").mean())

    r2 = get_metric_from_summary(cv_summary, "R2", prefer="GLOBAL")
    if pd.isna(r2):
        r2 = get_metric_from_summary(cv_summary, "R2", prefer="BIAS")
    if pd.isna(r2) and cv_fold_bc is not None and "R2" in cv_fold_bc.columns:
        r2 = float(pd.to_numeric(cv_fold_bc["R2"], errors="coerce").mean())

    r = get_metric_from_summary(cv_summary, "r", prefer="GLOBAL")
    if pd.isna(r):
        r = get_metric_from_summary(cv_summary, "r", prefer="BIAS")
    if pd.isna(r) and cv_fold_bc is not None and "r" in cv_fold_bc.columns:
        r = float(pd.to_numeric(cv_fold_bc["r"], errors="coerce").mean())

    age_dep = compute_oof_age_dependence(oof)

    runtime_sec = extract_total_runtime_seconds(compute)
    n_train = get_scalar_from_df(compute, ["n_scans_training", "n_training_graphs", "n_graphs_training"])
    n_train_subjects = get_scalar_from_df(compute, ["n_unique_subjects_training", "n_training_subjects"])
    n_full = get_scalar_from_df(full_summary, ["n_subjects_full_cohort", "n_full_cohort_graphs_inference"])
    if pd.isna(n_full):
        n_full = get_scalar_from_df(compute, ["n_full_cohort_graphs_inference", "n_subjects_full_cohort"])
    if pd.isna(n_full) and full_pred is not None:
        n_full = len(full_pred)

    mae_ci_width = get_ci_width_from_bootstrap(bootstrap, "MAE")

    gpu = ""
    if compute is not None and not compute.empty:
        for c in ["hardware.cuda_device_name", "cuda_device_name", "hardware.nvidia_smi_gpu_summary"]:
            if c in compute.columns:
                vals = compute[c].dropna()
                if len(vals):
                    gpu = str(vals.iloc[0])
                    break

    summary_row = {
        "cohort": cohort,
        "feature_set": feature_set,
        "feature_set_label": FEATURE_LABELS.get(feature_set, feature_set),
        "ablation_dir": str(d),
        "cv_MAE": mae,
        "cv_RMSE": rmse,
        "cv_R2": r2,
        "cv_r": r,
        "raw_BAG_age_slope": age_dep["raw_BAG_age_slope"],
        "raw_BAG_age_r": age_dep["raw_BAG_age_r"],
        "corrected_cBAG_age_slope": age_dep["corrected_cBAG_age_slope"],
        "corrected_cBAG_age_r": age_dep["corrected_cBAG_age_r"],
        "corrected_cBAG_column_used": age_dep["corrected_cBAG_column_used"],
        "age_dependence_n": age_dep["age_dependence_n"],
        "bootstrap_MAE_CI_width": mae_ci_width,
        "runtime_seconds": runtime_sec,
        "runtime_minutes": runtime_sec / 60 if pd.notna(runtime_sec) else np.nan,
        "n_training_graphs": n_train,
        "n_training_subjects": n_train_subjects,
        "n_full_cohort_graphs": n_full,
        "gpu": gpu,
        "n_missing_expected_inputs": len(missing),
    }

    learning = summarize_learning_for_plot(histories, cohort, feature_set)

    if cv_split is not None and not cv_split.empty:
        folds = cv_split.copy()
        folds["cohort"] = cohort
        folds["feature_set"] = feature_set
    else:
        folds = pd.DataFrame()

    # Prefer full metadata for visit patterns; fallback to prediction file.
    visit_df = make_visit_pattern_summary(cohort, feature_set, full_metadata if full_metadata is not None else full_pred)

    return summary_row, missing, learning, folds, visit_df


def collect_all_outputs():
    rows = []
    missing_rows = []
    learning_frames = []
    fold_frames = []
    visit_frames = []

    for cohort in COHORT_DIRS:
        for fs in FEATURE_SETS:
            row, missing, learning, folds, visits = collect_one_run(cohort, fs)
            rows.append(row)
            missing_rows.extend(missing)
            if not learning.empty:
                learning_frames.append(learning)
            if not folds.empty:
                fold_frames.append(folds)
            if not visits.empty:
                visit_frames.append(visits)

    summary = pd.DataFrame(rows)
    missing = pd.DataFrame(missing_rows)
    learning = pd.concat(learning_frames, ignore_index=True) if learning_frames else pd.DataFrame()
    folds = pd.concat(fold_frames, ignore_index=True) if fold_frames else pd.DataFrame()
    visits = pd.concat(visit_frames, ignore_index=True) if visit_frames else pd.DataFrame()

    return summary, learning, folds, visits, missing


# =============================================================================
# PLOTTING
# =============================================================================

def pivot_for_heatmap(df: pd.DataFrame, value: str) -> pd.DataFrame:
    out = df.pivot(index="cohort", columns="feature_set_label", values=value)
    cols = [FEATURE_LABELS[x] for x in FEATURE_SETS if FEATURE_LABELS[x] in out.columns]
    rows = [c for c in COHORT_DIRS if c in out.index]
    return out.loc[rows, cols]


def draw_heatmap(ax, data: pd.DataFrame, title: str, cbar_label: str, fmt: str = ".2f"):
    if data.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.set_title(title)
        ax.axis("off")
        return

    arr = data.values.astype(float)
    finite = np.isfinite(arr)

    if finite.any():
        vmin = np.nanmin(arr)
        vmax = np.nanmax(arr)
        if np.isclose(vmin, vmax):
            vmin -= 1
            vmax += 1
        im = ax.imshow(arr, aspect="auto", vmin=vmin, vmax=vmax)
    else:
        im = ax.imshow(np.zeros_like(arr), aspect="auto")

    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    ax.set_xticklabels(data.columns, rotation=35, ha="right")
    ax.set_yticklabels(data.index)
    ax.set_title(title)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = arr[i, j]
            label = "NA" if not np.isfinite(val) else format(val, fmt)
            ax.text(j, i, label, ha="center", va="center", fontsize=8)

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label)


def plot_cv_mae(ax, summary):
    draw_heatmap(ax, pivot_for_heatmap(summary, "cv_MAE"), "A. Cross-validated MAE", "MAE years", ".2f")


def plot_cv_r2(ax, summary):
    draw_heatmap(ax, pivot_for_heatmap(summary, "cv_R2"), "B. Cross-validated R²", "R²", ".2f")


def plot_raw_bag_age_slope(ax, summary):
    draw_heatmap(ax, pivot_for_heatmap(summary, "raw_BAG_age_slope"), "C. Raw BAG-age slope before correction", "Slope/year", ".3f")


def plot_learning_metric(ax, learning, metric: str, title: str, ylabel: str):
    if learning.empty:
        ax.text(0.5, 0.5, "No learning curves found", ha="center", va="center")
        ax.set_title(title)
        ax.axis("off")
        return

    df = learning[learning["metric"].eq(metric)].copy()

    if PLOT_ONE_LEARNING_FEATURE_SET and LEARNING_CURVE_FEATURE_SET is not None:
        df = df[df["feature_set"].eq(LEARNING_CURVE_FEATURE_SET)].copy()
        title_suffix = FEATURE_LABELS.get(LEARNING_CURVE_FEATURE_SET, LEARNING_CURVE_FEATURE_SET)
    else:
        title_suffix = "all feature sets"

    if df.empty:
        ax.text(0.5, 0.5, f"No {metric} curves found", ha="center", va="center")
        ax.set_title(title)
        ax.axis("off")
        return

    for cohort in COHORT_DIRS:
        sub = df[df["cohort"].eq(cohort)].sort_values("epoch")
        if sub.empty:
            continue
        x = pd.to_numeric(sub["epoch"], errors="coerce")
        y = pd.to_numeric(sub["mean"], errors="coerce")
        ax.plot(x, y, label=cohort)
        if "ci95" in sub.columns:
            ci = pd.to_numeric(sub["ci95"], errors="coerce").fillna(0)
            ax.fill_between(x, y - ci, y + ci, alpha=0.12)

    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title} ({title_suffix})")
    ax.legend(fontsize=8, frameon=False)


def plot_age_balance(ax, folds):
    if folds.empty or "val_age_mean" not in folds.columns:
        ax.text(0.5, 0.5, "No CV split summaries found", ha="center", va="center")
        ax.set_title("F. CV fold age balance")
        ax.axis("off")
        return

    df = folds.copy()
    if PLOT_ONE_LEARNING_FEATURE_SET and LEARNING_CURVE_FEATURE_SET is not None:
        df = df[df["feature_set"].eq(LEARNING_CURVE_FEATURE_SET)].copy()

    values = []
    labels = []
    for cohort in COHORT_DIRS:
        vals = pd.to_numeric(df.loc[df["cohort"].eq(cohort), "val_age_mean"], errors="coerce").dropna().values
        if len(vals):
            values.append(vals)
            labels.append(cohort)

    if not values:
        ax.text(0.5, 0.5, "No age balance data", ha="center", va="center")
        ax.set_title("F. CV fold age balance")
        ax.axis("off")
        return

    ax.boxplot(values, labels=labels)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Validation fold mean age")
    ax.set_title("F. Age balance across CV folds")


def plot_runtime_minutes(ax, summary):
    draw_heatmap(ax, pivot_for_heatmap(summary, "runtime_minutes"), "G. Runtime by model", "Minutes", ".1f")


def plot_full_n(ax, summary):
    draw_heatmap(ax, pivot_for_heatmap(summary, "n_full_cohort_graphs"), "H. Full-cohort prediction N", "Graphs", ".0f")


def plot_bootstrap_width(ax, summary):
    draw_heatmap(ax, pivot_for_heatmap(summary, "bootstrap_MAE_CI_width"), "I. Bootstrap MAE CI width", "Years", ".2f")


def make_figure(summary, learning, folds):
    plt.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
    })

    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    axes = axes.ravel()

    plot_cv_mae(axes[0], summary)
    plot_cv_r2(axes[1], summary)
    plot_raw_bag_age_slope(axes[2], summary)
    plot_learning_metric(axes[3], learning, "train_loss", "D. Training loss", "Training loss")
    plot_learning_metric(axes[4], learning, "val_loss_proxy", "E. Validation loss proxy", "Validation RMSE²")
    plot_age_balance(axes[5], folds)
    plot_runtime_minutes(axes[6], summary)
    plot_full_n(axes[7], summary)
    plot_bootstrap_width(axes[8], summary)

    fig.suptitle("Supplementary Figure. Brain-age model training quality control", y=0.995, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.985])

    fig.savefig(OUT_FIG_PNG, dpi=DPI, bbox_inches="tight")
    fig.savefig(OUT_FIG_PDF, bbox_inches="tight")
    plt.close(fig)

    print("Saved:", OUT_FIG_PNG)
    print("Saved:", OUT_FIG_PDF)


def write_readme(summary, missing):
    n_runs = len(summary)
    n_complete = int(summary["n_missing_expected_inputs"].eq(0).sum()) if "n_missing_expected_inputs" in summary.columns else 0
    n_missing = len(missing) if missing is not None and not missing.empty else 0

    text = f"""# Supplementary Training QC v3

This folder contains an updated composite supplementary training QC figure and source tables.

## Key updates

- Panel C plots raw BAG-age slope before correction; corrected cBAG-age slope is saved in the summary table and should be approximately zero by design.
- Runtime is reported in minutes.
- Training loss and validation loss proxy curves are both shown.
- Longitudinal visit patterns are summarized for ADNI/HABS when graph IDs contain y-visit labels.

## Inputs searched

Root:
`{RESULTS_ROOT}`

Experiment tag:
`{EXPERIMENT_TAG}`

Expected cohort × feature-set runs:
{len(COHORT_DIRS) * len(FEATURE_SETS)}

Runs represented in summary table:
{n_runs}

Runs with all expected input files present:
{n_complete}

Missing expected files listed:
{n_missing}

## Notes

The validation loss proxy is computed as validation RMSE² because the training script stores per-epoch validation RMSE rather than explicit validation MSE loss.

Main model performance should still be reported in Table 2. This figure is a supplementary QC figure for convergence, fold balance, runtime, sample size, and uncertainty.
"""
    OUT_README.write_text(text)
    print("Saved:", OUT_README)


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 100)
    print("BUILD SUPPLEMENTARY TRAINING QC FIGURE V3")
    print("=" * 100)

    summary, learning, folds, visits, missing = collect_all_outputs()

    summary.to_csv(OUT_SUMMARY, index=False)
    learning.to_csv(OUT_LEARNING, index=False)
    folds.to_csv(OUT_FOLDS, index=False)
    visits.to_csv(OUT_LONGITUDINAL, index=False)
    missing.to_csv(OUT_MISSING, index=False)

    display_cols = [
        "cohort", "feature_set", "cv_MAE", "cv_R2",
        "raw_BAG_age_slope", "raw_BAG_age_r",
        "corrected_cBAG_age_slope", "corrected_cBAG_age_r", "corrected_cBAG_column_used",
        "runtime_minutes", "n_training_graphs", "n_full_cohort_graphs",
        "n_missing_expected_inputs",
    ]
    print("\nRun summary:")
    print(summary[[c for c in display_cols if c in summary.columns]].to_string(index=False))

    print("\nLongitudinal visit-pattern summary:")
    if visits.empty:
        print("No visit-pattern data.")
    else:
        print(visits.to_string(index=False))

    if not missing.empty:
        print("\nMissing expected inputs:")
        print(missing.to_string(index=False))
    else:
        print("\nNo missing expected input files.")

    make_figure(summary, learning, folds)
    write_readme(summary, missing)

    print("\nSaved:")
    for p in [
        OUT_FIG_PNG, OUT_FIG_PDF, OUT_SUMMARY, OUT_LEARNING, OUT_FOLDS,
        OUT_LONGITUDINAL, OUT_MISSING, OUT_README
    ]:
        print(" ", p)


if __name__ == "__main__":
    main()
