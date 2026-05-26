#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create final Figure 4 and Supplementary Figure S4 neuroimaging associations.

Main outputs
------------
1. Figure 4A: per-cohort panel grid
   - imaging_only model
   - bias-corrected cBAG only
   - rows = six primary neuroimaging variables
   - columns = cohorts

2. Figure 4B: cross-cohort pooled figure
   - imaging_only model
   - bias-corrected cBAG only
   - six panels, one per primary neuroimaging variable
   - all cohorts overlaid in each panel
   - pooled regression line

Supplementary outputs
---------------------
- all five models
- all cohorts
- bias-corrected cBAG
- raw/unadjusted BAG or cBAG when available
- eight neuroimaging variables:
    hippocampus volume, hippocampus FA, total brain volume, total brain FA,
    clustering, path length, global efficiency, local efficiency

Inputs
------
$WORK/ines/results/Figure5_AssociationsOnly/merged_tables/
  merged_metadata_screening_<feature_set>_<cohort>.csv

Outputs
-------
$WORK/ines/results/Figure4_NeuroimagingAssociations/final_figure4/
$WORK/ines/results/Figure4_NeuroimagingAssociations/supplementary_figureS4/
$WORK/ines/results/Figure4_NeuroimagingAssociations/tables/

Run
---
python make_final_figure4_and_supplementaryS4_neuroimaging_v2.py
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


# =============================================================================
# SETTINGS
# =============================================================================

WORK = Path(os.environ.get("WORK", "/mnt/newStor/paros/paros_WORK"))
BASE_DIR = WORK / "ines"
RESULTS_ROOT = BASE_DIR / "results"

FIG5_MERGED_DIR = RESULTS_ROOT / "Figure5_AssociationsOnly" / "merged_tables"

FIG4_ROOT = RESULTS_ROOT / "Figure4_NeuroimagingAssociations"
FIG4_FINAL_DIR = FIG4_ROOT / "final_figure4"
FIG4_SUPP_DIR = FIG4_ROOT / "supplementary_figureS4"
FIG4_TABLE_DIR = FIG4_ROOT / "tables"

PRIMARY_FEATURE_SET = "imaging_only"
FEATURE_SETS = [
    "imaging_only",
    "imaging_demographics",
    "imaging_biomarkers",
    "full",
    "full_no_cardiovascular",
]
COHORTS = ["ADNI", "ADRC", "HABS", "AD_DECODE"]

MIN_N = 20
FIGURE_FORMATS = ["png", "pdf"]

BIAS_CORRECTED_CBAG_PRIORITY = [
    "cBAG_oof_global_raw_clean",
    "cBAG_oof_global",
    "cBAG_global_raw_clean",
    "cBAG_global",
    "cBAG_bias_corrected",
    "cBAG_BiasCorrected",
    "cBAG",
]

RAW_BAG_OR_CBAG_PRIORITY = [
    "BAG",
    "BAG_raw",
    "brain_age_gap",
    "brain_age_gap_raw",
    "BrainAgeGap",
    "age_gap",
    "age_gap_raw",
    "cBAG_raw",
    "cBAG_raw_clean",
    "cBAG_uncorrected",
    "cBAG_oof_global_uncorrected",
    "BAG_oof",
    "BAG_oof_global",
]

MAIN_FIGURE_VARIABLES = [
    {
        "key": "hippocampus_volume",
        "label": "Hippocampal volume",
        "short_label": "Hc volume",
        "candidates": [
            "hippocampus_volume_abs_sum_mm3",
            "hippocampus_volume_sum",
            "hippocampus_volume_abs_mean_mm3",
            "hippocampus_volume_mean",
        ],
    },
    {
        "key": "hippocampus_FA",
        "label": "Hippocampal FA",
        "short_label": "Hc FA",
        "candidates": ["hippocampus_FA_mean", "hippocampus_FA_sum"],
    },
    {
        "key": "total_brain_volume",
        "label": "Total brain volume",
        "short_label": "Brain volume",
        "candidates": ["total_brain_volume_abs_mm3"],
    },
    {
        "key": "total_brain_FA",
        "label": "Total brain FA",
        "short_label": "Brain FA",
        "candidates": ["total_brain_FA_mean"],
    },
    {
        "key": "graph_clustering",
        "label": "Graph clustering coefficient",
        "short_label": "Clustering",
        "candidates": ["Clustering_Coeff", "Clustering_Coeff_graph_builder_metrics"],
    },
    {
        "key": "graph_path_length",
        "label": "Graph path length",
        "short_label": "Path length",
        "candidates": ["Path_Length", "Path_Length_graph_builder_metrics"],
    },
]

SUPPLEMENTARY_VARIABLES = MAIN_FIGURE_VARIABLES + [
    {
        "key": "global_efficiency",
        "label": "Global efficiency",
        "short_label": "Global eff.",
        "candidates": ["Global_Efficiency", "Global_Efficiency_graph_builder_metrics"],
    },
    {
        "key": "local_efficiency",
        "label": "Local efficiency",
        "short_label": "Local eff.",
        "candidates": ["Local_Efficiency", "Local_Efficiency_graph_builder_metrics"],
    },
]


# =============================================================================
# HELPERS
# =============================================================================

def normalize_name(x: object) -> str:
    s = str(x).lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


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
    q_ranked = ranked * m / np.arange(1, m + 1)
    q_ranked = np.minimum.accumulate(q_ranked[::-1])[::-1]
    q_ranked = np.clip(q_ranked, 0, 1)

    q_ok = np.empty_like(q_ranked)
    q_ok[order] = q_ranked
    q[ok] = q_ok
    return q


def fisher_r_ci(r: float, n: int, alpha: float = 0.05) -> tuple[float, float]:
    if n <= 3 or not np.isfinite(r) or abs(r) >= 1:
        return np.nan, np.nan
    z = np.arctanh(r)
    se = 1.0 / np.sqrt(n - 3)
    zcrit = stats.norm.ppf(1 - alpha / 2)
    return float(np.tanh(z - zcrit * se)), float(np.tanh(z + zcrit * se))


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

    for c in candidates:
        nc = normalize_name(c)
        hits = [col for col in df.columns if normalize_name(col).endswith(nc)]
        if hits:
            return hits[0]

    return None


def p_text(p: float) -> str:
    if not np.isfinite(p):
        return "p=NA"
    if p < 1e-4:
        return "p<1e-4"
    if p < 0.001:
        return "p<0.001"
    return f"p={p:.3g}"


def q_text(q: float) -> str:
    if not np.isfinite(q):
        return "q=NA"
    if q < 1e-4:
        return "q<1e-4"
    if q < 0.001:
        return "q<0.001"
    return f"q={q:.3g}"


def merged_path(feature_set: str, cohort: str) -> Path:
    return FIG5_MERGED_DIR / f"merged_metadata_screening_{feature_set}_{cohort}.csv"


def load_merged(feature_set: str, cohort: str) -> Optional[pd.DataFrame]:
    path = merged_path(feature_set, cohort)
    if not path.exists():
        print(f"[WARN] Missing merged table: {path}")
        return None
    return pd.read_csv(path, low_memory=False)


def get_target_columns(df: pd.DataFrame) -> list[dict]:
    targets = []

    raw_col = first_existing(df, RAW_BAG_OR_CBAG_PRIORITY)
    if raw_col is not None:
        targets.append({
            "target_type": "raw_BAG_or_cBAG",
            "target_label": "Raw BAG / uncorrected cBAG",
            "target_col": raw_col,
        })

    bc_col = first_existing(df, BIAS_CORRECTED_CBAG_PRIORITY)
    if bc_col is not None:
        targets.append({
            "target_type": "bias_corrected_cBAG",
            "target_label": "Bias-corrected cBAG",
            "target_col": bc_col,
        })

    return targets


def compute_stats_for_table(
    df: pd.DataFrame,
    feature_set: str,
    cohort: str,
    variables: list[dict],
) -> list[dict]:
    rows = []

    targets = get_target_columns(df)
    if not targets:
        print(f"[WARN] No raw or bias-corrected BAG/cBAG target found for {feature_set} | {cohort}")
        return rows

    for target in targets:
        y = pd.to_numeric(df[target["target_col"]], errors="coerce")

        for spec in variables:
            x_col = first_existing(df, spec["candidates"])
            if x_col is None:
                rows.append({
                    "feature_set": feature_set,
                    "cohort": cohort,
                    "target_type": target["target_type"],
                    "target_label": target["target_label"],
                    "target_col": target["target_col"],
                    "variable_key": spec["key"],
                    "variable_label": spec["label"],
                    "variable_short_label": spec["short_label"],
                    "variable_col": "",
                    "status": "missing_variable",
                    "n": 0,
                })
                continue

            x = pd.to_numeric(df[x_col], errors="coerce")
            tmp = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()

            if len(tmp) < MIN_N or tmp["x"].nunique() < 3 or tmp["y"].nunique() < 3:
                rows.append({
                    "feature_set": feature_set,
                    "cohort": cohort,
                    "target_type": target["target_type"],
                    "target_label": target["target_label"],
                    "target_col": target["target_col"],
                    "variable_key": spec["key"],
                    "variable_label": spec["label"],
                    "variable_short_label": spec["short_label"],
                    "variable_col": x_col,
                    "status": "too_few_usable_values",
                    "n": int(len(tmp)),
                    "n_unique_x": int(tmp["x"].nunique()) if len(tmp) else 0,
                    "n_unique_y": int(tmp["y"].nunique()) if len(tmp) else 0,
                })
                continue

            pearson_r, pearson_p = stats.pearsonr(tmp["x"], tmp["y"])
            spearman_rho, spearman_p = stats.spearmanr(tmp["x"], tmp["y"])
            slope, intercept, _, _, slope_se = stats.linregress(tmp["x"], tmp["y"])
            ci_lo, ci_hi = fisher_r_ci(float(pearson_r), int(len(tmp)))

            rows.append({
                "feature_set": feature_set,
                "cohort": cohort,
                "target_type": target["target_type"],
                "target_label": target["target_label"],
                "target_col": target["target_col"],
                "variable_key": spec["key"],
                "variable_label": spec["label"],
                "variable_short_label": spec["short_label"],
                "variable_col": x_col,
                "status": "ok",
                "n": int(len(tmp)),
                "n_unique_x": int(tmp["x"].nunique()),
                "n_unique_y": int(tmp["y"].nunique()),
                "pearson_r": float(pearson_r),
                "pearson_r_ci95_low": ci_lo,
                "pearson_r_ci95_high": ci_hi,
                "pearson_p": float(pearson_p),
                "spearman_rho": float(spearman_rho),
                "spearman_p": float(spearman_p),
                "linear_slope": float(slope),
                "linear_intercept": float(intercept),
                "linear_slope_se": float(slope_se),
                "x_mean": float(tmp["x"].mean()),
                "x_sd": float(tmp["x"].std(ddof=0)),
                "x_min": float(tmp["x"].min()),
                "x_max": float(tmp["x"].max()),
                "y_mean": float(tmp["y"].mean()),
                "y_sd": float(tmp["y"].std(ddof=0)),
            })

    return rows


def add_fdr_columns(stats_df: pd.DataFrame) -> pd.DataFrame:
    out = stats_df.copy()
    out["pearson_q_fdr_within_feature_set_target"] = np.nan
    out["spearman_q_fdr_within_feature_set_target"] = np.nan

    ok = out["status"].eq("ok") if "status" in out.columns else pd.Series(False, index=out.index)
    for _, idx in out[ok].groupby(["feature_set", "target_type"]).groups.items():
        idx = list(idx)
        out.loc[idx, "pearson_q_fdr_within_feature_set_target"] = fdr_bh(out.loc[idx, "pearson_p"].values)
        out.loc[idx, "spearman_q_fdr_within_feature_set_target"] = fdr_bh(out.loc[idx, "spearman_p"].values)

    return out


def load_primary_scatter_data(
    feature_set: str,
    cohort: str,
    variable_spec: dict,
    target_type: str = "bias_corrected_cBAG",
) -> tuple[pd.DataFrame, str, str]:
    df = load_merged(feature_set, cohort)
    if df is None:
        return pd.DataFrame(), "", ""

    targets = [t for t in get_target_columns(df) if t["target_type"] == target_type]
    if not targets:
        return pd.DataFrame(), "", ""

    x_col = first_existing(df, variable_spec["candidates"])
    if x_col is None:
        return pd.DataFrame(), "", targets[0]["target_col"]

    y_col = targets[0]["target_col"]
    tmp = pd.DataFrame({
        "x": pd.to_numeric(df[x_col], errors="coerce"),
        "y": pd.to_numeric(df[y_col], errors="coerce"),
        "cohort": cohort,
    }).replace([np.inf, -np.inf], np.nan).dropna()

    return tmp, x_col, y_col


# =============================================================================
# MAIN FIGURE 4A: PER-COHORT GRID
# =============================================================================

def plot_final_figure4_per_cohort(stats_df: pd.DataFrame) -> None:
    feature_set = PRIMARY_FEATURE_SET
    target_type = "bias_corrected_cBAG"

    n_rows = len(MAIN_FIGURE_VARIABLES)
    n_cols = len(COHORTS)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14.2, 12.8), squeeze=False)

    for i, spec in enumerate(MAIN_FIGURE_VARIABLES):
        for j, cohort in enumerate(COHORTS):
            ax = axes[i, j]
            tmp, x_col, y_col = load_primary_scatter_data(feature_set, cohort, spec, target_type)

            if tmp.empty or len(tmp) < MIN_N or tmp["x"].nunique() < 3 or tmp["y"].nunique() < 3:
                ax.text(0.5, 0.5, "No usable data", ha="center", va="center", transform=ax.transAxes, fontsize=8)
                if i == 0:
                    ax.set_title(cohort, fontsize=10)
                if j == 0:
                    ax.set_ylabel("Bias-corrected cBAG", fontsize=8)
                ax.set_xlabel(spec["short_label"], fontsize=8)
                ax.tick_params(labelsize=7)
                continue

            ax.scatter(tmp["x"], tmp["y"], s=12, alpha=0.65)

            slope, intercept, *_ = stats.linregress(tmp["x"], tmp["y"])
            xx = np.linspace(tmp["x"].min(), tmp["x"].max(), 150)
            ax.plot(xx, intercept + slope * xx, linewidth=1.3)

            stat_row = stats_df[
                (stats_df["feature_set"] == feature_set)
                & (stats_df["cohort"] == cohort)
                & (stats_df["target_type"] == target_type)
                & (stats_df["variable_key"] == spec["key"])
                & (stats_df["status"] == "ok")
            ]
            if not stat_row.empty:
                row = stat_row.iloc[0]
                ax.text(
                    0.04, 0.96,
                    f"r={row['pearson_r']:.2f}\n{q_text(row['pearson_q_fdr_within_feature_set_target'])}\nn={int(row['n'])}",
                    ha="left", va="top", transform=ax.transAxes, fontsize=7,
                    bbox=dict(boxstyle="round,pad=0.22", facecolor="white", alpha=0.80, edgecolor="none"),
                )

            if i == 0:
                ax.set_title(cohort, fontsize=10)
            if j == 0:
                ax.set_ylabel(f"{spec['short_label']}\nBias-corrected cBAG", fontsize=8)
            else:
                ax.set_ylabel("")
            if i == n_rows - 1:
                ax.set_xlabel(spec["short_label"], fontsize=8)
            else:
                ax.set_xlabel("")
            ax.tick_params(labelsize=7)

    fig.suptitle(
        "Figure 4A. Per-cohort neuroimaging associations with bias-corrected cBAG\n"
        f"Primary model: {PRIMARY_FEATURE_SET.replace('_', ' ')}",
        fontsize=13,
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.975])

    outstem = FIG4_FINAL_DIR / "Figure4A_Main_per_cohort_bias_corrected_cBAG"
    for fmt in FIGURE_FORMATS:
        out = outstem.with_suffix(f".{fmt}")
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"[INFO] Saved {out}")
    plt.close(fig)


# =============================================================================
# MAIN FIGURE 4B: CROSS-COHORT POOLED
# =============================================================================

def plot_final_figure4_cross_cohort(stats_df: pd.DataFrame) -> None:
    feature_set = PRIMARY_FEATURE_SET
    target_type = "bias_corrected_cBAG"

    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.8), squeeze=False)
    cohort_markers = {"ADNI": "o", "ADRC": "s", "HABS": "^", "AD_DECODE": "D"}

    for ax, spec in zip(axes.ravel(), MAIN_FIGURE_VARIABLES):
        pooled = []

        for cohort in COHORTS:
            tmp, x_col, y_col = load_primary_scatter_data(feature_set, cohort, spec, target_type)
            if tmp.empty:
                continue

            pooled.append(tmp)
            ax.scatter(
                tmp["x"], tmp["y"],
                s=18,
                alpha=0.70,
                marker=cohort_markers.get(cohort, "o"),
                label=cohort,
            )

        if not pooled:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(spec["label"])
            continue

        data = pd.concat(pooled, ignore_index=True)

        if len(data) >= MIN_N and data["x"].nunique() >= 3 and data["y"].nunique() >= 3:
            r, p = stats.pearsonr(data["x"], data["y"])
            slope, intercept, *_ = stats.linregress(data["x"], data["y"])
            ci_lo, ci_hi = fisher_r_ci(float(r), int(len(data)))

            xx = np.linspace(data["x"].min(), data["x"].max(), 150)
            ax.plot(xx, intercept + slope * xx, color="black", linewidth=1.5)

            ax.text(
                0.04, 0.96,
                f"pooled r={r:.2f} [{ci_lo:.2f}, {ci_hi:.2f}]\n{p_text(p)}\nn={len(data)}",
                ha="left",
                va="top",
                transform=ax.transAxes,
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.82, edgecolor="none"),
            )

        ax.set_title(spec["label"], fontsize=11)
        ax.set_xlabel(spec["short_label"], fontsize=9)
        ax.set_ylabel("Bias-corrected cBAG", fontsize=9)
        ax.tick_params(labelsize=8)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        by_label = dict(zip(labels, handles))
        fig.legend(
            by_label.values(),
            by_label.keys(),
            loc="lower center",
            ncol=len(by_label),
            frameon=False,
            fontsize=9,
        )

    fig.suptitle(
        "Figure 4B. Cross-cohort neuroimaging associations with bias-corrected cBAG\n"
        f"Primary model: {PRIMARY_FEATURE_SET.replace('_', ' ')}",
        fontsize=13,
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0.055, 1, 0.94])

    outstem = FIG4_FINAL_DIR / "Figure4B_Main_cross_cohort_bias_corrected_cBAG"
    for fmt in FIGURE_FORMATS:
        out = outstem.with_suffix(f".{fmt}")
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"[INFO] Saved {out}")
    plt.close(fig)


# =============================================================================
# SUPPLEMENTARY PLOTS
# =============================================================================

def plot_supp_heatmap(
    stats_df: pd.DataFrame,
    feature_set: str,
    target_type: str,
    variables: list[dict],
) -> None:
    ok = stats_df[
        (stats_df["feature_set"] == feature_set)
        & (stats_df["target_type"] == target_type)
        & (stats_df["status"] == "ok")
    ].copy()

    if ok.empty:
        print(f"[WARN] No ok stats for supplementary heatmap: {feature_set} | {target_type}")
        return

    var_order = [v["key"] for v in variables]
    var_labels = {v["key"]: v["label"] for v in variables}

    mat = pd.DataFrame(index=var_order, columns=COHORTS, dtype=float)
    ann = pd.DataFrame("", index=var_order, columns=COHORTS)

    for _, row in ok.iterrows():
        vk = row["variable_key"]
        cohort = row["cohort"]
        if vk not in mat.index or cohort not in mat.columns:
            continue
        r = row["pearson_r"]
        q = row["pearson_q_fdr_within_feature_set_target"]
        n = int(row["n"])
        star = "***" if pd.notna(q) and q < 0.001 else "**" if pd.notna(q) and q < 0.01 else "*" if pd.notna(q) and q < 0.05 else ""
        mat.loc[vk, cohort] = r
        ann.loc[vk, cohort] = f"{r:.2f}{star}\nn={n}"

    data = mat.to_numpy(dtype=float)
    vmax = max(0.05, np.nanmax(np.abs(data)) if np.isfinite(data).any() else 1.0)

    fig, ax = plt.subplots(figsize=(7.8, 5.7))
    im = ax.imshow(data, aspect="auto", vmin=-vmax, vmax=vmax, cmap="coolwarm")
    ax.set_title(f"Supplementary Figure S4 heatmap\n{feature_set} | {target_type}", fontsize=11)
    ax.set_xticks(np.arange(len(COHORTS)))
    ax.set_xticklabels(COHORTS, fontsize=9)
    ax.set_yticks(np.arange(len(var_order)))
    ax.set_yticklabels([var_labels[v] for v in var_order], fontsize=8)

    for i, vk in enumerate(var_order):
        for j, cohort in enumerate(COHORTS):
            if ann.loc[vk, cohort]:
                ax.text(j, i, ann.loc[vk, cohort], ha="center", va="center", fontsize=7)

    ax.set_xticks(np.arange(-0.5, len(COHORTS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(var_order), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.0)
    ax.tick_params(which="minor", bottom=False, left=False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cbar.set_label("Pearson r", fontsize=9)
    fig.tight_layout()

    outstem = FIG4_SUPP_DIR / f"FigureS4_heatmap_{feature_set}_{target_type}"
    for fmt in FIGURE_FORMATS:
        out = outstem.with_suffix(f".{fmt}")
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"[INFO] Saved {out}")
    plt.close(fig)


def plot_supp_model_comparison(stats_df: pd.DataFrame, target_type: str = "bias_corrected_cBAG") -> None:
    ok = stats_df[
        (stats_df["target_type"] == target_type)
        & (stats_df["status"] == "ok")
    ].copy()

    if ok.empty:
        print(f"[WARN] No data for model-comparison heatmap: {target_type}")
        return

    ok["row_label"] = ok["variable_short_label"] + " | " + ok["cohort"]
    row_order = []
    for spec in SUPPLEMENTARY_VARIABLES:
        for cohort in COHORTS:
            row_order.append(f"{spec['short_label']} | {cohort}")

    mat = pd.DataFrame(index=row_order, columns=FEATURE_SETS, dtype=float)
    ann = pd.DataFrame("", index=row_order, columns=FEATURE_SETS)

    for _, row in ok.iterrows():
        rl = row["row_label"]
        fs = row["feature_set"]
        if rl not in mat.index or fs not in mat.columns:
            continue
        r = row["pearson_r"]
        q = row["pearson_q_fdr_within_feature_set_target"]
        star = "*" if pd.notna(q) and q < 0.05 else ""
        mat.loc[rl, fs] = r
        ann.loc[rl, fs] = f"{r:.2f}{star}"

    mat = mat.dropna(how="all")
    ann = ann.loc[mat.index]

    if mat.empty:
        return

    data = mat.to_numpy(dtype=float)
    vmax = max(0.05, np.nanmax(np.abs(data)) if np.isfinite(data).any() else 1.0)

    fig_h = max(8, 0.24 * len(mat.index) + 1.6)
    fig, ax = plt.subplots(figsize=(9.8, fig_h))
    im = ax.imshow(data, aspect="auto", vmin=-vmax, vmax=vmax, cmap="coolwarm")
    ax.set_title(f"Supplementary Figure S4 model comparison | {target_type}", fontsize=11)
    ax.set_xticks(np.arange(len(FEATURE_SETS)))
    ax.set_xticklabels([fs.replace("_", "\n") for fs in FEATURE_SETS], fontsize=8)
    ax.set_yticks(np.arange(len(mat.index)))
    ax.set_yticklabels(mat.index.tolist(), fontsize=6.5)

    for i, rlab in enumerate(mat.index):
        for j, fs in enumerate(FEATURE_SETS):
            if ann.loc[rlab, fs]:
                ax.text(j, i, ann.loc[rlab, fs], ha="center", va="center", fontsize=6)

    ax.set_xticks(np.arange(-0.5, len(FEATURE_SETS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(mat.index), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
    cbar.set_label("Pearson r", fontsize=9)
    fig.tight_layout()

    outstem = FIG4_SUPP_DIR / f"FigureS4_model_comparison_{target_type}"
    for fmt in FIGURE_FORMATS:
        out = outstem.with_suffix(f".{fmt}")
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"[INFO] Saved {out}")
    plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    FIG4_FINAL_DIR.mkdir(parents=True, exist_ok=True)
    FIG4_SUPP_DIR.mkdir(parents=True, exist_ok=True)
    FIG4_TABLE_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("FINAL FIGURE 4A / 4B AND SUPPLEMENTARY FIGURE S4 NEUROIMAGING ASSOCIATIONS")
    print("=" * 100)
    print("Input merged tables:", FIG5_MERGED_DIR)
    print("Final Figure 4 output:", FIG4_FINAL_DIR)
    print("Supplementary Figure S4 output:", FIG4_SUPP_DIR)
    print("Tables output:", FIG4_TABLE_DIR)

    rows = []
    for fs in FEATURE_SETS:
        for cohort in COHORTS:
            df = load_merged(fs, cohort)
            if df is None:
                continue
            rows.extend(compute_stats_for_table(df, fs, cohort, SUPPLEMENTARY_VARIABLES))

    stats_df = pd.DataFrame(rows)
    if stats_df.empty:
        raise RuntimeError("No Figure 4 association rows were produced. Check Figure 5 merged tables first.")

    stats_df = add_fdr_columns(stats_df)

    all_stats_path = FIG4_TABLE_DIR / "Figure4_S4_neuroimaging_association_stats_all_models.csv"
    stats_df.to_csv(all_stats_path, index=False)
    print(f"[INFO] Saved {all_stats_path}")

    for fs in FEATURE_SETS:
        fs_path = FIG4_TABLE_DIR / f"Figure4_S4_neuroimaging_association_stats_{fs}.csv"
        stats_df[stats_df["feature_set"] == fs].to_csv(fs_path, index=False)
        print(f"[INFO] Saved {fs_path}")

    # Final Figure 4A and 4B
    plot_final_figure4_per_cohort(stats_df)
    plot_final_figure4_cross_cohort(stats_df)

    # Supplementary heatmaps for all model/target combinations with valid results
    ok_targets = stats_df.loc[stats_df["status"].eq("ok"), ["feature_set", "target_type"]].drop_duplicates()
    for _, row in ok_targets.iterrows():
        plot_supp_heatmap(stats_df, row["feature_set"], row["target_type"], SUPPLEMENTARY_VARIABLES)

    # Supplementary model-comparison heatmaps
    for target_type in stats_df.loc[stats_df["status"].eq("ok"), "target_type"].dropna().unique().tolist():
        plot_supp_model_comparison(stats_df, target_type=target_type)

    readme = FIG4_ROOT / "Figure4_S4_neuroimaging_README.md"
    readme.write_text(
        "# Figure 4 and Supplementary Figure S4 neuroimaging associations\n\n"
        "Final Figure 4A uses the imaging_only model and bias-corrected cBAG only, with individual panels for each cohort.\n\n"
        "Final Figure 4B uses the imaging_only model and bias-corrected cBAG only, with cross-cohort pooled scatter panels.\n\n"
        "Supplementary Figure S4 includes all five models, all cohorts, bias-corrected cBAG, and raw/unadjusted BAG or cBAG when available. "
        "It also includes global and local efficiency.\n\n"
        "Statistics include Pearson r, Fisher-z 95% CI for r, Pearson p-value, Spearman rho/p-value, linear slope/intercept, "
        "and Benjamini-Hochberg FDR q-values within each feature set and target type.\n\n"
        "The script reads the merged screening tables produced by the Figure 5 pipeline, so run final_Figure5_pipeline_after_ADNI_enrichment_v2.py first.\n"
    )
    print(f"[INFO] Saved {readme}")

    print("\nDone.")
    print("Valid tests:", int(stats_df["status"].eq("ok").sum()))
    print("Final Figure 4:", FIG4_FINAL_DIR)
    print("Supplementary Figure S4:", FIG4_SUPP_DIR)
    print("Tables:", FIG4_TABLE_DIR)


if __name__ == "__main__":
    main()
