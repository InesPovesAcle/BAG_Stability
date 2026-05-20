#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate Figure 3 and Supplementary Figure 3.

Figure 3:
    OOF raw predicted age vs chronological age for the primary model.

Supplementary Figure 3:
    BAG/cBAG age-dependence diagnostics for the primary model:
        BAG_raw vs chronological age
        cBAG_foldwise vs chronological age
        cBAG_oof_global vs chronological age

Recommended IEEE setting:
    MAIN_FEATURE_SET = "imaging_only"

Input:
    *_cv_oof_predictions.csv files from the final oofglobal training run.

Outputs:
    Figure3_PredictedAge_vs_RealAge_imaging_only.png/pdf/csv
    Figure3_PredictedAge_vs_RealAge_imaging_only_stats.csv

    SupplementaryFigure3_BAG_cBAG_age_dependence_imaging_only.png/pdf/csv
    SupplementaryFigure3_BAG_cBAG_age_dependence_imaging_only_stats.csv
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import pearsonr, linregress
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


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
# IEEE primary model recommendation:
# imaging_only = cleanest connectome/neuroimaging-only brain-age model.
MAIN_FEATURE_SET = "imaging_only"

COHORT_ORDER = ["ADNI", "ADRC", "HABS", "AD_DECODE"]

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

PANEL_LABELS_4 = ["A", "B", "C", "D"]


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


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def safe_pearsonr(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)

    if mask.sum() < 3:
        return np.nan, np.nan

    if len(np.unique(x[mask])) < 2 or len(np.unique(y[mask])) < 2:
        return np.nan, np.nan

    return pearsonr(x[mask], y[mask])


def scatter_stats(x, y, prediction_plot=False):
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
            "r2_corr": np.nan,
            "r2_identity": np.nan,
            "mae": np.nan,
            "rmse": np.nan,
            "slope": np.nan,
            "intercept": np.nan,
        }

    r, p = safe_pearsonr(x, y)
    lr = linregress(x, y)

    out = {
        "n": int(len(x)),
        "r": float(r),
        "p": float(p),
        "r2_corr": float(r ** 2),
        "slope": float(lr.slope),
        "intercept": float(lr.intercept),
    }

    if prediction_plot:
        out["r2_identity"] = float(r2_score(x, y))
        out["mae"] = float(mean_absolute_error(x, y))
        out["rmse"] = float(rmse(x, y))
    else:
        out["r2_identity"] = np.nan
        out["mae"] = np.nan
        out["rmse"] = np.nan

    return out


def format_p(p):
    if pd.isna(p):
        return "nan"
    if p < 1e-4:
        return f"{p:.2e}"
    return f"{p:.4f}"


def add_prediction_metrics_box(ax, stats):
    text = (
        f"n = {stats['n']}\n"
        f"r = {stats['r']:.3f}\n"
        f"R² = {stats['r2_identity']:.3f}\n"
        f"MAE = {stats['mae']:.2f}\n"
        f"RMSE = {stats['rmse']:.2f}"
    )

    ax.text(
        0.03,
        0.97,
        text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )


def add_bag_metrics_box(ax, stats):
    text = (
        f"n = {stats['n']}\n"
        f"r = {stats['r']:.3f}\n"
        f"R² = {stats['r2_corr']:.3f}\n"
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
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )


def set_equal_age_axes(ax, x, y):
    """Make predicted-age plots easier to compare by using shared x/y limits."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)

    if mask.sum() < 3:
        return

    lo = min(np.nanmin(x[mask]), np.nanmin(y[mask]))
    hi = max(np.nanmax(x[mask]), np.nanmax(y[mask]))
    pad = 0.05 * (hi - lo) if hi > lo else 1.0

    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)


def load_oof_data_for_feature_set(feature_set):
    rows = []
    dfs = {}

    for cohort in COHORT_ORDER:
        path = get_oof_path(cohort, feature_set)

        if not os.path.exists(path):
            print(f"Missing OOF file for {cohort} {feature_set}: {path}")
            continue

        df = pd.read_csv(path)

        required = ["age_true", "pred_raw"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            print(f"Skipping {cohort} {feature_set}; missing columns: {missing}")
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

        if "cBAG_oof_global" not in df.columns:
            print(f"Warning: {cohort} {feature_set} does not contain cBAG_oof_global.")

        dfs[cohort] = df

        rows.append({
            "cohort": cohort,
            "feature_set": feature_set,
            "n_rows": len(df),
            "source_path": path,
            "has_BAG_raw": "BAG_raw" in df.columns,
            "has_cBAG_foldwise": "cBAG_foldwise" in df.columns,
            "has_cBAG_oof_global": "cBAG_oof_global" in df.columns,
        })

    availability_df = pd.DataFrame(rows)

    return dfs, availability_df


# =========================
# FIGURE 3
# =========================
def make_figure3_predicted_age_vs_real_age(feature_set):
    dfs, availability_df = load_oof_data_for_feature_set(feature_set)

    out_png = os.path.join(
        COMBINED_DIR,
        f"Figure3_PredictedAge_vs_RealAge_{feature_set}.png",
    )
    out_pdf = os.path.join(
        COMBINED_DIR,
        f"Figure3_PredictedAge_vs_RealAge_{feature_set}.pdf",
    )
    out_csv = os.path.join(
        COMBINED_DIR,
        f"Figure3_PredictedAge_vs_RealAge_{feature_set}_source_data.csv",
    )
    out_stats_csv = os.path.join(
        COMBINED_DIR,
        f"Figure3_PredictedAge_vs_RealAge_{feature_set}_stats.csv",
    )
    out_availability_csv = os.path.join(
        COMBINED_DIR,
        f"Figure3_{feature_set}_input_availability.csv",
    )

    source_rows = []
    stats_rows = []

    fig, axes = plt.subplots(2, 2, figsize=(14, 12), dpi=300)
    axes = axes.flatten()

    for i, cohort in enumerate(COHORT_ORDER):
        ax = axes[i]

        if cohort not in dfs:
            ax.set_title(f"{PANEL_LABELS_4[i]}. {cohort}: missing")
            ax.axis("off")
            continue

        df = dfs[cohort].copy()

        x = clean_numeric(df["age_true"]).values
        y = clean_numeric(df["pred_raw"]).values
        stats = scatter_stats(x, y, prediction_plot=True)

        plot_df = pd.DataFrame({
            "cohort": cohort,
            "feature_set": feature_set,
            "age_true": x,
            "pred_raw": y,
            "source_path": df["source_path"].iloc[0],
        })
        source_rows.append(plot_df)

        stats_rows.append({
            "figure": "Figure3",
            "cohort": cohort,
            "feature_set": feature_set,
            "x": "age_true",
            "y": "pred_raw",
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
            ax.plot(xx, lr.slope * xx + lr.intercept, linestyle="--", label="Fit")

            lo = min(np.nanmin(x[mask]), np.nanmin(y[mask]))
            hi = max(np.nanmax(x[mask]), np.nanmax(y[mask]))
            ax.plot([lo, hi], [lo, hi], linestyle=":", label="Identity")

        add_prediction_metrics_box(ax, stats)
        set_equal_age_axes(ax, x, y)

        ax.set_title(f"{PANEL_LABELS_4[i]}. {cohort}")
        ax.set_xlabel("Chronological age")
        ax.set_ylabel("Predicted age")
        ax.grid(alpha=0.3)

    fig.suptitle(
        f"Figure 3. OOF raw predicted age vs chronological age ({feature_set})",
        fontsize=16,
        y=1.02,
    )

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=2,
            frameon=True,
            bbox_to_anchor=(0.5, 0.985),
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

    print("\nSaved Figure 3:")
    print(out_png)
    print(out_pdf)
    print(out_csv)
    print(out_stats_csv)
    print(out_availability_csv)


# =========================
# SUPPLEMENTARY FIGURE 3
# =========================
def make_supplementary_figure3_bag_cbag_age_dependence(feature_set):
    dfs, availability_df = load_oof_data_for_feature_set(feature_set)

    out_png = os.path.join(
        COMBINED_DIR,
        f"SupplementaryFigure3_BAG_cBAG_age_dependence_{feature_set}.png",
    )
    out_pdf = os.path.join(
        COMBINED_DIR,
        f"SupplementaryFigure3_BAG_cBAG_age_dependence_{feature_set}.pdf",
    )
    out_csv = os.path.join(
        COMBINED_DIR,
        f"SupplementaryFigure3_BAG_cBAG_age_dependence_{feature_set}_source_data.csv",
    )
    out_stats_csv = os.path.join(
        COMBINED_DIR,
        f"SupplementaryFigure3_BAG_cBAG_age_dependence_{feature_set}_stats.csv",
    )
    out_availability_csv = os.path.join(
        COMBINED_DIR,
        f"SupplementaryFigure3_{feature_set}_input_availability.csv",
    )

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
            stats = scatter_stats(x, y, prediction_plot=False)

            plot_df = pd.DataFrame({
                "cohort": cohort,
                "feature_set": feature_set,
                "brain_metric": bag_col,
                "age_true": x,
                "value": y,
                "source_path": df["source_path"].iloc[0],
            })
            source_rows.append(plot_df)

            stats_rows.append({
                "figure": "SupplementaryFigure3",
                "cohort": cohort,
                "feature_set": feature_set,
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
        f"Supplementary Figure 3. BAG/cBAG age-dependence diagnostics ({feature_set})",
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

    print("\nSaved Supplementary Figure 3:")
    print(out_png)
    print(out_pdf)
    print(out_csv)
    print(out_stats_csv)
    print(out_availability_csv)


# =========================
# MAIN
# =========================
def main():
    print("Generating Figure 3 and Supplementary Figure 3")
    print("Feature set:", MAIN_FEATURE_SET)
    print("Output directory:", COMBINED_DIR)

    make_figure3_predicted_age_vs_real_age(MAIN_FEATURE_SET)
    make_supplementary_figure3_bag_cbag_age_dependence(MAIN_FEATURE_SET)


if __name__ == "__main__":
    main()