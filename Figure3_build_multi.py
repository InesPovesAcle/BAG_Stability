#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate Supplementary Figure 3A-E.

Supplementary Figure 3:
    BAG/cBAG age-dependence diagnostics across all feature sets.

Panels:
    Supplementary Figure 3A: imaging_only
    Supplementary Figure 3B: imaging_demographics
    Supplementary Figure 3C: imaging_biomarkers
    Supplementary Figure 3D: full
    Supplementary Figure 3E: full_no_cardiovascular

Each figure contains:
    rows = cohorts
    columns = BAG_raw, cBAG_foldwise, cBAG_oof_global

Input:
    *_cv_oof_predictions.csv files from the final oofglobal training run.

Output directory:
    /mnt/newStor/paros/paros_WORK/ines/results/
    BrainAgeValidation_AllCohorts_BAGBiasCorr_OOFGlobal_BiologicalValidation/
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import pearsonr, linregress


# =========================
# PATHS
# =========================
RESULTS_ROOT = "/mnt/newStor/paros/paros_WORK/ines/results"

COMBINED_DIR = os.path.join(
    RESULTS_ROOT,
    "BrainAgeValidation_AllCohorts_BAGBiasCorr_OOFGlobal_BiologicalValidation",
)

os.makedirs(COMBINED_DIR, exist_ok=True)


# =========================
# SETTINGS
# =========================
COHORT_ORDER = ["ADNI", "ADRC", "HABS", "AD_DECODE"]

FEATURE_FIGURE_MAP = {
    "A": "imaging_only",
    "B": "imaging_demographics",
    "C": "imaging_biomarkers",
    "D": "full",
    "E": "full_no_cardiovascular",
}

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

BAG_COLUMNS = [
    "BAG_raw",
    "cBAG_foldwise",
    "cBAG_oof_global",
]

BAG_LABELS = {
    "BAG_raw": "Raw BAG",
    "cBAG_foldwise": "Fold-wise cBAG",
    "cBAG_oof_global": "OOF-global cBAG",
}


# =========================
# HELPERS
# =========================
def get_oof_path(cohort, feature_set):
    folder = RESULTS_DIR_MAP[cohort]
    prefix = PREFIX_MAP[cohort]

    return os.path.join(
        RESULTS_ROOT,
        folder,
        f"ablation_{feature_set}",
        f"{prefix}_{feature_set}_cv_oof_predictions.csv",
    )


def clean_numeric(x):
    return pd.to_numeric(x, errors="coerce")


def safe_pearsonr(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)

    if mask.sum() < 3:
        return np.nan, np.nan

    if len(np.unique(x[mask])) < 2 or len(np.unique(y[mask])) < 2:
        return np.nan, np.nan

    return pearsonr(x[mask], y[mask])


def scatter_stats(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)

    x = x[mask]
    y = y[mask]

    if len(x) < 3:
        return {
            "n": int(len(x)),
            "r": np.nan,
            "p": np.nan,
            "r2": np.nan,
            "slope": np.nan,
            "intercept": np.nan,
        }

    r, p = safe_pearsonr(x, y)
    lr = linregress(x, y)

    return {
        "n": int(len(x)),
        "r": float(r),
        "p": float(p),
        "r2": float(r ** 2),
        "slope": float(lr.slope),
        "intercept": float(lr.intercept),
    }


def format_p(p):
    if pd.isna(p):
        return "nan"
    if p < 1e-4:
        return f"{p:.2e}"
    return f"{p:.4f}"


def add_bag_metrics_box(ax, stats):
    text = (
        f"n = {stats['n']}\n"
        f"r = {stats['r']:.3f}\n"
        f"R² = {stats['r2']:.3f}\n"
        f"p = {format_p(stats['p'])}\n"
        f"slope = {stats['slope']:.3f}"
    )

    ax.text(
        0.03,
        0.97,
        text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8.5,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )


def load_oof_data_for_feature_set(feature_set):
    rows = []
    dfs = {}

    for cohort in COHORT_ORDER:
        path = get_oof_path(cohort, feature_set)

        if not os.path.exists(path):
            print(f"Missing OOF file for {cohort} {feature_set}: {path}")
            rows.append({
                "cohort": cohort,
                "feature_set": feature_set,
                "source_path": path,
                "status": "missing_file",
            })
            continue

        df = pd.read_csv(path)

        required = ["age_true", "pred_raw"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            print(f"Skipping {cohort} {feature_set}; missing columns: {missing}")
            rows.append({
                "cohort": cohort,
                "feature_set": feature_set,
                "source_path": path,
                "status": "missing_columns",
                "missing_columns": ",".join(missing),
            })
            continue

        df = df.copy()
        df["cohort"] = cohort
        df["feature_set"] = feature_set
        df["source_path"] = path

        df["age_true"] = clean_numeric(df["age_true"])
        df["pred_raw"] = clean_numeric(df["pred_raw"])

        if "BAG_raw" not in df.columns:
            df["BAG_raw"] = df["pred_raw"] - df["age_true"]

        if "cBAG_foldwise" not in df.columns and "cBAG" in df.columns:
            df["cBAG_foldwise"] = df["cBAG"]

        dfs[cohort] = df

        rows.append({
            "cohort": cohort,
            "feature_set": feature_set,
            "n_rows": len(df),
            "source_path": path,
            "status": "loaded",
            "has_BAG_raw": "BAG_raw" in df.columns,
            "has_cBAG_foldwise": "cBAG_foldwise" in df.columns,
            "has_cBAG_oof_global": "cBAG_oof_global" in df.columns,
        })

    availability_df = pd.DataFrame(rows)

    return dfs, availability_df


# =========================
# SUPPLEMENTARY FIGURE 3A-E
# =========================
def make_supplementary_figure3_for_feature_set(panel_letter, feature_set):
    dfs, availability_df = load_oof_data_for_feature_set(feature_set)

    base = f"SupplementaryFigure3{panel_letter}_BAG_cBAG_age_dependence_{feature_set}"

    out_png = os.path.join(COMBINED_DIR, f"{base}.png")
    out_pdf = os.path.join(COMBINED_DIR, f"{base}.pdf")
    out_csv = os.path.join(COMBINED_DIR, f"{base}_source_data.csv")
    out_stats_csv = os.path.join(COMBINED_DIR, f"{base}_stats.csv")
    out_availability_csv = os.path.join(COMBINED_DIR, f"{base}_input_availability.csv")

    source_rows = []
    stats_rows = []

    n_rows = len(COHORT_ORDER)
    n_cols = len(BAG_COLUMNS)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5.2 * n_cols, 4.2 * n_rows),
        dpi=300,
        squeeze=False,
    )

    for row_idx, cohort in enumerate(COHORT_ORDER):
        for col_idx, bag_col in enumerate(BAG_COLUMNS):
            ax = axes[row_idx, col_idx]

            if cohort not in dfs:
                ax.set_title(f"{cohort}: missing")
                ax.axis("off")
                continue

            df = dfs[cohort].copy()

            if bag_col not in df.columns:
                ax.set_title(f"{cohort}: missing {bag_col}")
                ax.axis("off")
                continue

            x = clean_numeric(df["age_true"]).values
            y = clean_numeric(df[bag_col]).values

            stats = scatter_stats(x, y)

            plot_df = pd.DataFrame({
                "supplementary_figure": f"SupplementaryFigure3{panel_letter}",
                "feature_set": feature_set,
                "cohort": cohort,
                "brain_metric": bag_col,
                "age_true": x,
                "value": y,
                "source_path": df["source_path"].iloc[0],
            })
            source_rows.append(plot_df)

            stats_rows.append({
                "supplementary_figure": f"SupplementaryFigure3{panel_letter}",
                "feature_set": feature_set,
                "cohort": cohort,
                "x": "age_true",
                "y": bag_col,
                **stats,
            })

            mask = np.isfinite(x) & np.isfinite(y)

            ax.scatter(
                x[mask],
                y[mask],
                alpha=0.72,
                edgecolors="black",
                linewidth=0.3,
            )

            if mask.sum() >= 3:
                lr = linregress(x[mask], y[mask])
                xx = np.linspace(np.nanmin(x[mask]), np.nanmax(x[mask]), 100)
                ax.plot(xx, lr.slope * xx + lr.intercept, linestyle="--")
                ax.axhline(0, linestyle=":", linewidth=1)

            add_bag_metrics_box(ax, stats)

            if row_idx == 0:
                ax.set_title(BAG_LABELS.get(bag_col, bag_col))

            if col_idx == 0:
                ax.set_ylabel(f"{cohort}\nBrain-age gap")
            else:
                ax.set_ylabel("Brain-age gap")

            ax.set_xlabel("Chronological age")
            ax.grid(alpha=0.3)

    fig.suptitle(
        f"Supplementary Figure 3{panel_letter}. BAG/cBAG age-dependence diagnostics "
        f"({feature_set})",
        fontsize=16,
        y=1.01,
    )

    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()

    if source_rows:
        pd.concat(source_rows, ignore_index=True).to_csv(out_csv, index=False)
    else:
        pd.DataFrame().to_csv(out_csv, index=False)

    pd.DataFrame(stats_rows).to_csv(out_stats_csv, index=False)
    availability_df.to_csv(out_availability_csv, index=False)

    print(f"\nSaved Supplementary Figure 3{panel_letter}:")
    print(out_png)
    print(out_pdf)
    print(out_csv)
    print(out_stats_csv)
    print(out_availability_csv)

    return {
        "panel": f"SupplementaryFigure3{panel_letter}",
        "feature_set": feature_set,
        "png": out_png,
        "pdf": out_pdf,
        "source_data": out_csv,
        "stats": out_stats_csv,
        "availability": out_availability_csv,
    }


# =========================
# MAIN
# =========================
def main():
    print("Generating Supplementary Figure 3A-E")
    print("Output directory:", COMBINED_DIR)

    manifest_rows = []

    for panel_letter, feature_set in FEATURE_FIGURE_MAP.items():
        result = make_supplementary_figure3_for_feature_set(
            panel_letter=panel_letter,
            feature_set=feature_set,
        )
        manifest_rows.append(result)

    manifest_df = pd.DataFrame(manifest_rows)

    manifest_csv = os.path.join(
        COMBINED_DIR,
        "SupplementaryFigure3A_E_manifest.csv",
    )
    manifest_xlsx = os.path.join(
        COMBINED_DIR,
        "SupplementaryFigure3A_E_manifest.xlsx",
    )

    manifest_df.to_csv(manifest_csv, index=False)
    manifest_df.to_excel(manifest_xlsx, index=False)

    print("\nSaved manifest:")
    print(manifest_csv)
    print(manifest_xlsx)


if __name__ == "__main__":
    main()