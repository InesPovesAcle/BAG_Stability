#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
discover_and_plot_longitudinal_metadata.py
==========================================

Find genuinely longitudinal columns in ADNI and HABS harmonized metadata/results files
and generate longitudinal plots.

What it does
------------
1. Searches candidate ADNI/HABS CSV files under:
   - /mnt/newStor/paros/paros_WORK/ines/data/harmonization/<COHORT>
   - /mnt/newStor/paros/paros_WORK/ines/results/BrainAgePrediction<COHORT>.../ablation_<feature_set>/validation_figures_full_cohort
   - /mnt/newStor/paros/paros_WORK/ines/results/BrainAgePrediction<COHORT>.../ablation_<feature_set>/validation_figures_oof

2. Detects subject and visit structure using either:
   - scan IDs like R1234_y0, R1234_y4, H4369_y0, H4369_y2
   - separate subject + visit/year columns such as DWI_subject + DWI_visit_year

3. Collapses duplicate rows per subject/visit.

4. For every numeric column, computes longitudinal delta summaries:
   n paired, sd(delta), unique deltas, missingness, etc.

5. Writes reports:
   longitudinal_metadata_file_summary.csv
   longitudinal_metadata_column_summary.csv
   longitudinal_metadata_delta_subject_table_<COHORT>_<TAG>.csv

6. Makes plots for selected or automatically chosen longitudinal variables:
   - spaghetti plots by visit
   - delta histograms
   - delta boxplots by cohort for shared variables
   - availability heatmap

Run
---
python discover_and_plot_longitudinal_metadata.py

Useful options
--------------
python discover_and_plot_longitudinal_metadata.py \
  --cohorts ADNI,HABS \
  --feature-set imaging_only \
  --min-pairs 20 \
  --top-n 24

python discover_and_plot_longitudinal_metadata.py \
  --preferred-columns Hc_volume_pct_brain,Hc_FA,Total_Brain_volume,Total_Brain_FA,BMI_raw_clean,OM_BMI_raw_clean,Global_Efficiency_raw_clean
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


WORK = Path("/mnt/newStor/paros/paros_WORK")
BASE = WORK / "ines"
DATA_ROOT = BASE / "data"
RESULTS_ROOT = BASE / "results"

OUT_BASE = (
    RESULTS_ROOT
    / "BrainAgeValidation_AllCohorts_BAGBiasCorr_OOFGlobal_BiologicalValidation"
    / "Longitudinal_metadata_discovery"
)

PREDICTION_DIRS = {
    "ADNI": "BrainAgePredictionADNI_stratified_groupcv_targetnorm_bagbiascorr_oofglobal",
    "HABS": "BrainAgePredictionHABS_stratified_groupcv_targetnorm_bagbiascorr_oofglobal",
}

DEFAULT_PREFERRED = [
    "Hc_volume_pct_brain",
    "Hc_volume_mm3",
    "Hc_FA",
    "Hc_RD",
    "Hc_AD",
    "Hc_ADC",
    "Total_Brain_volume",
    "Total_Brain_FA",
    "Clustering_Coeff_raw_clean",
    "Path_Length_raw_clean",
    "Global_Efficiency_raw_clean",
    "Local_Efficiency_raw_clean",
    "BMI_raw_clean",
    "BMI_calculated_raw_clean",
    "PHC_BMI",
    "vrf_bmi_PHC_BMI_raw_clean",
    "OM_BMI_raw_clean",
    "OM_BMI_clinical_raw_clean",
    "clinical__BW_CholTotal_raw_clean",
    "BW_CholTotal_y_raw_clean",
    "GFAP_raw_clean",
    "gfap_raw_clean",
    "tau_total_raw_clean",
    "TotalTau_raw_clean",
    "cognition_composite_raw_clean",
    "MMSE_total_raw_clean",
    "MOCA_total_corrected_raw_clean",
    "ADAS_total_raw_clean",
    "clinical__MMSE_Total",
    "Animal_Total_raw_clean",
    "FAS_Total_raw_clean",
]

SENTINELS = {
    -999999, -99999, -9999, -999,
    -888888, -88888, -8888, -888,
    -777777, -77777, -7777, -777,
    999, 9999, 99999, 999999,
    888, 8888, 88888, 888888,
    777, 7777, 77777, 777777,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Discover longitudinal metadata columns and plot ADNI/HABS trajectories.")
    p.add_argument("--data-root", default=str(DATA_ROOT))
    p.add_argument("--results-root", default=str(RESULTS_ROOT))
    p.add_argument("--outdir", default=str(OUT_BASE))
    p.add_argument("--cohorts", default="ADNI,HABS")
    p.add_argument("--feature-set", default="imaging_only")
    p.add_argument("--min-pairs", type=int, default=20)
    p.add_argument("--top-n", type=int, default=24)
    p.add_argument("--max-files-per-cohort", type=int, default=200)
    p.add_argument("--preferred-columns", default=",".join(DEFAULT_PREFERRED))
    p.add_argument("--dpi", type=int, default=350)
    p.add_argument("--formats", default="png,pdf")
    return p.parse_args()


def clean_numeric(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    for v in SENTINELS:
        x = x.mask(x == v, np.nan)
    return x.replace([np.inf, -np.inf], np.nan)


def norm_colname(c: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(c).lower())


def parse_scan_id(x) -> Tuple[Optional[str], Optional[int]]:
    s = str(x).strip()

    # R1234_y0, H4369_Y2, R12345-y4, etc.
    m = re.search(r"\b([RH]\d{4,5})[_-]?[Yy](\d+)\b", s)
    if m:
        return m.group(1).upper(), int(m.group(2))

    # Sometimes subject has no cohort letter in a separate column; keep None visit.
    m = re.search(r"\b([RH]\d{4,5})\b", s)
    if m:
        return m.group(1).upper(), None

    return None, None


def parse_visit_value(x) -> Optional[int]:
    if pd.isna(x):
        return None
    s = str(x).strip()

    # y0, Y2, visit 4, etc.
    m = re.search(r"[Yy]\s*([0-9]+)", s)
    if m:
        return int(m.group(1))

    # numeric year/visit
    try:
        f = float(s)
        if np.isfinite(f):
            # HABS often has 0/2; ADNI 0/4.
            return int(round(f))
    except Exception:
        pass

    return None


def possible_scan_id_columns(df: pd.DataFrame) -> List[str]:
    priority = [
        "Subject_ID", "graph_id", "connectome_key", "runno", "scan_id",
        "DWI_subject", "regional_id", "subject_id", "Subject", "ID",
    ]
    cols = [c for c in priority if c in df.columns]

    # Add other ID-like columns, but avoid obvious non-subject demographics.
    for c in df.columns:
        low = str(c).lower()
        if c in cols:
            continue
        if any(k in low for k in ["subject", "subj", "scan", "graph", "runno", "connectome", "regional"]):
            if not any(bad in low for bad in ["sex", "race", "education", "marital", "language"]):
                cols.append(c)
    return cols


def possible_visit_columns(df: pd.DataFrame) -> List[str]:
    priority = [
        "visit", "Visit", "DWI_visit", "DWI_visit_year", "Year", "Visit_ID",
        "session", "Session", "timepoint", "Timepoint",
    ]
    cols = [c for c in priority if c in df.columns]
    for c in df.columns:
        low = str(c).lower()
        if c in cols:
            continue
        if any(k in low for k in ["visit", "year", "wave", "session", "timepoint"]):
            cols.append(c)
    return cols


def infer_subject_visit(df: pd.DataFrame, cohort: str) -> Tuple[Optional[pd.DataFrame], dict]:
    """
    Try to infer subject_base and visit from the file.
    Returns a two-column frame plus diagnostic info.
    """
    diagnostics = {
        "id_col": "",
        "visit_col": "",
        "method": "",
        "parsed_rows": 0,
        "unique_scan_rows": 0,
        "subjects_ge2_visits": 0,
    }

    id_cols = possible_scan_id_columns(df)
    visit_cols = possible_visit_columns(df)

    # 1) ID column already contains R/H + y visit.
    best = None
    best_score = -1
    for id_col in id_cols:
        parsed = df[id_col].apply(parse_scan_id)
        subj = parsed.apply(lambda z: z[0])
        visit = parsed.apply(lambda z: z[1])
        ok = subj.notna() & visit.notna()
        if ok.sum() == 0:
            continue
        tmp = pd.DataFrame({"subject_base": subj[ok], "visit": visit[ok].astype(int)})
        scans = tmp.drop_duplicates()
        subj_vis = scans.groupby("subject_base")["visit"].nunique()
        paired = int((subj_vis >= 2).sum())
        score = paired * 100000 + len(scans)
        if score > best_score:
            best_score = score
            best = (id_col, "", "scan_id_contains_visit", subj, visit, ok, len(scans), paired)

    if best is not None:
        id_col, visit_col, method, subj, visit, ok, nscan, paired = best
        diagnostics.update({
            "id_col": id_col,
            "visit_col": visit_col,
            "method": method,
            "parsed_rows": int(ok.sum()),
            "unique_scan_rows": int(nscan),
            "subjects_ge2_visits": int(paired),
        })
        return pd.DataFrame({
            "subject_base": subj,
            "visit": visit,
        }), diagnostics

    # 2) ID column has subject only, separate visit/year column has visit.
    best = None
    best_score = -1
    for id_col in id_cols:
        parsed_id = df[id_col].apply(parse_scan_id)
        subj = parsed_id.apply(lambda z: z[0])

        # If no R/H in ID, try prefix based on cohort for numeric subject IDs.
        if subj.notna().sum() == 0:
            raw = df[id_col].astype(str).str.replace(r"\.0$", "", regex=True).str.strip()
            if cohort == "HABS":
                subj = raw.where(raw.str.match(r"^\d{4,5}$", na=False), None).map(lambda x: f"H{x}" if pd.notna(x) else None)
            elif cohort == "ADNI":
                subj = raw.where(raw.str.match(r"^\d{4,5}$", na=False), None).map(lambda x: f"R{x}" if pd.notna(x) else None)

        for visit_col in visit_cols:
            visit = df[visit_col].apply(parse_visit_value)
            ok = subj.notna() & visit.notna()
            if ok.sum() == 0:
                continue
            tmp = pd.DataFrame({"subject_base": subj[ok], "visit": visit[ok].astype(int)})
            scans = tmp.drop_duplicates()
            subj_vis = scans.groupby("subject_base")["visit"].nunique()
            paired = int((subj_vis >= 2).sum())
            score = paired * 100000 + len(scans)
            if score > best_score:
                best_score = score
                best = (id_col, visit_col, "separate_subject_and_visit", subj, visit, ok, len(scans), paired)

    if best is not None:
        id_col, visit_col, method, subj, visit, ok, nscan, paired = best
        diagnostics.update({
            "id_col": id_col,
            "visit_col": visit_col,
            "method": method,
            "parsed_rows": int(ok.sum()),
            "unique_scan_rows": int(nscan),
            "subjects_ge2_visits": int(paired),
        })
        return pd.DataFrame({
            "subject_base": subj,
            "visit": visit,
        }), diagnostics

    return None, diagnostics


def candidate_files_for_cohort(data_root: Path, results_root: Path, cohort: str, feature_set: str, max_files: int) -> List[Path]:
    files: List[Path] = []

    # Harmonization folders.
    for sub in [
        data_root / "harmonization" / cohort,
        data_root / "pre_harmonization" / cohort,
        data_root / "Regional_stats" / cohort,
    ]:
        if sub.exists():
            files.extend(sub.rglob("*.csv"))
            files.extend(sub.rglob("*.txt"))
            files.extend(sub.rglob("*.tsv"))

    # Results validation products.
    pred_dir = PREDICTION_DIRS.get(cohort)
    if pred_dir:
        base = results_root / pred_dir / f"ablation_{feature_set}"
        for sub in [
            base,
            base / "validation_figures_oof",
            base / "validation_figures_full_cohort",
        ]:
            if sub.exists():
                files.extend(sub.glob("*.csv"))

    # Prefer names likely to contain metadata or validation tables.
    def rank(p: Path) -> Tuple[int, str]:
        s = str(p).lower()
        score = 0
        for token in ["enriched", "subject_level", "metadata", "clinical", "harmon", "prediction", "oof", "full_cohort", "regional", "stats"]:
            if token in s:
                score -= 10
        for bad in ["manifest", "summary", "bootstrap", "figure", "png", "pdf"]:
            if bad in s:
                score += 10
        return score, str(p)

    unique = sorted(set(files), key=rank)
    return unique[:max_files]


def load_table(path: Path) -> Optional[pd.DataFrame]:
    try:
        if path.suffix.lower() in [".txt", ".tsv"]:
            return pd.read_csv(path, sep="\t", low_memory=False)
        return pd.read_csv(path, low_memory=False)
    except Exception:
        try:
            return pd.read_csv(path, sep=None, engine="python", low_memory=False)
        except Exception:
            return None


def summarize_longitudinal_columns(df: pd.DataFrame, sv: pd.DataFrame, cohort: str, file_tag: str, min_pairs: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    work = df.copy()
    work["_subject_base"] = sv["subject_base"].values
    work["_visit"] = sv["visit"].values
    work = work[work["_subject_base"].notna() & work["_visit"].notna()].copy()
    work["_visit"] = work["_visit"].astype(int)

    # Collapse duplicate rows per subject/visit by first non-null per column.
    work = (
        work.sort_values(["_subject_base", "_visit"])
            .groupby(["_subject_base", "_visit"], as_index=False)
            .first()
    )

    subj_vis = work.groupby("_subject_base")["_visit"].nunique()
    paired_subjects = subj_vis[subj_vis >= 2].index
    paired = work[work["_subject_base"].isin(paired_subjects)].copy()

    delta_rows = []
    subject_delta_rows = []

    # numeric columns only, excluding helper/id fields.
    exclude_patterns = [
        "_subject_base", "_visit",
        "subject", "subj", "scan", "graph", "runno", "connectome", "path", "file",
    ]

    for col in paired.columns:
        low = str(col).lower()
        if col in ["_subject_base", "_visit"]:
            continue
        if any(pat in low for pat in exclude_patterns):
            # Allow explicit biological columns containing subject? no.
            continue

        values = clean_numeric(paired[col])
        if values.notna().sum() < min_pairs:
            continue

        deltas = []
        for sid, g in paired.groupby("_subject_base"):
            if g["_visit"].nunique() < 2:
                continue
            g = g.sort_values("_visit")
            first = clean_numeric(pd.Series([g.iloc[0][col]])).iloc[0]
            last = clean_numeric(pd.Series([g.iloc[-1][col]])).iloc[0]
            if pd.notna(first) and pd.notna(last):
                deltas.append({
                    "cohort": cohort,
                    "file_tag": file_tag,
                    "subject_base": sid,
                    "baseline_visit": int(g.iloc[0]["_visit"]),
                    "followup_visit": int(g.iloc[-1]["_visit"]),
                    "column": col,
                    "baseline_value": first,
                    "followup_value": last,
                    "delta": last - first,
                })

        if not deltas:
            continue

        ddf = pd.DataFrame(deltas)
        x = ddf["delta"].astype(float)

        row = {
            "cohort": cohort,
            "file_tag": file_tag,
            "column": col,
            "n_pairs": int(x.notna().sum()),
            "sd_delta": float(x.std()) if x.notna().sum() > 1 else np.nan,
            "n_unique_delta": int(x.nunique(dropna=True)),
            "min_delta": float(x.min()) if x.notna().any() else np.nan,
            "median_delta": float(x.median()) if x.notna().any() else np.nan,
            "max_delta": float(x.max()) if x.notna().any() else np.nan,
            "all_zero": bool(len(x) and np.isclose(x, 0).all()),
            "usable_longitudinal": bool(x.notna().sum() >= min_pairs and x.nunique(dropna=True) >= 2 and x.std() > 0),
        }
        delta_rows.append(row)
        subject_delta_rows.extend(deltas)

    return pd.DataFrame(delta_rows), pd.DataFrame(subject_delta_rows)


def safe_tag(path: Path) -> str:
    s = path.name
    s = re.sub(r"\.(csv|txt|tsv)$", "", s, flags=re.I)
    s = re.sub(r"[^A-Za-z0-9_]+", "_", s)
    return s[:80]


def pick_columns_for_plots(summary: pd.DataFrame, preferred: List[str], top_n: int) -> pd.DataFrame:
    usable = summary[summary["usable_longitudinal"]].copy()
    if usable.empty:
        return usable

    pref_norm = {norm_colname(c): i for i, c in enumerate(preferred)}
    usable["preferred_rank"] = usable["column"].map(lambda c: pref_norm.get(norm_colname(c), 9999))

    # Prefer exact preferred columns, then large n and high sd.
    usable = usable.sort_values(
        ["preferred_rank", "n_pairs", "sd_delta"],
        ascending=[True, False, False],
    )
    return usable.head(top_n)


def plot_spaghetti(delta_subjects: pd.DataFrame, cohort: str, file_tag: str, columns: List[str], outdir: Path, formats: List[str], dpi: int):
    if not columns:
        return []

    ncols = 3
    nrows = int(np.ceil(len(columns) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 3.8 * nrows), squeeze=False)
    axes_flat = axes.ravel()

    for ax, col in zip(axes_flat, columns):
        sub = delta_subjects[(delta_subjects["cohort"] == cohort) & (delta_subjects["file_tag"] == file_tag) & (delta_subjects["column"] == col)]
        if sub.empty:
            ax.axis("off")
            continue

        for _, r in sub.iterrows():
            ax.plot([r["baseline_visit"], r["followup_visit"]], [r["baseline_value"], r["followup_value"]], alpha=0.25, linewidth=0.8)
        mean_by_visit = (
            pd.concat([
                sub[["baseline_visit", "baseline_value"]].rename(columns={"baseline_visit": "visit", "baseline_value": "value"}),
                sub[["followup_visit", "followup_value"]].rename(columns={"followup_visit": "visit", "followup_value": "value"}),
            ])
            .groupby("visit")["value"].mean()
        )
        ax.plot(mean_by_visit.index, mean_by_visit.values, marker="o", linewidth=2.2)
        ax.set_title(f"{col}\nn={len(sub)}", fontsize=9)
        ax.set_xlabel("Visit")
        ax.set_ylabel("Value")
        ax.grid(True, alpha=0.25)

    for ax in axes_flat[len(columns):]:
        ax.axis("off")

    fig.suptitle(f"{cohort}: longitudinal trajectories from {file_tag}", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    saved = []
    for fmt in formats:
        path = outdir / f"{cohort}_{file_tag}_spaghetti.{fmt}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        saved.append(str(path))
    plt.close(fig)
    return saved


def plot_delta_histograms(delta_subjects: pd.DataFrame, cohort: str, file_tag: str, columns: List[str], outdir: Path, formats: List[str], dpi: int):
    if not columns:
        return []

    ncols = 3
    nrows = int(np.ceil(len(columns) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 3.6 * nrows), squeeze=False)
    axes_flat = axes.ravel()

    for ax, col in zip(axes_flat, columns):
        sub = delta_subjects[(delta_subjects["cohort"] == cohort) & (delta_subjects["file_tag"] == file_tag) & (delta_subjects["column"] == col)]
        x = pd.to_numeric(sub["delta"], errors="coerce").dropna()
        if x.empty:
            ax.axis("off")
            continue
        ax.hist(x, bins=20, alpha=0.85)
        ax.axvline(0, linewidth=1)
        ax.set_title(f"Δ{col}\nn={len(x)}, sd={x.std():.3g}", fontsize=9)
        ax.set_xlabel("Follow-up - baseline")
        ax.set_ylabel("Subjects")
        ax.grid(True, axis="y", alpha=0.25)

    for ax in axes_flat[len(columns):]:
        ax.axis("off")

    fig.suptitle(f"{cohort}: delta distributions from {file_tag}", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    saved = []
    for fmt in formats:
        path = outdir / f"{cohort}_{file_tag}_delta_histograms.{fmt}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        saved.append(str(path))
    plt.close(fig)
    return saved


def plot_shared_boxplots(all_delta_subjects: pd.DataFrame, chosen: pd.DataFrame, outdir: Path, formats: List[str], dpi: int):
    # Use exact shared column names appearing in both cohorts.
    cohorts = sorted(chosen["cohort"].unique())
    if len(cohorts) < 2:
        return []

    counts = chosen.groupby("column")["cohort"].nunique()
    shared_cols = counts[counts >= 2].index.tolist()
    if not shared_cols:
        return []

    shared_cols = shared_cols[:16]

    fig, axes = plt.subplots(len(shared_cols), 1, figsize=(9, max(4, 1.25 * len(shared_cols))), squeeze=False)
    axes_flat = axes.ravel()

    for ax, col in zip(axes_flat, shared_cols):
        data = []
        labels = []
        for cohort in cohorts:
            sub = all_delta_subjects[(all_delta_subjects["cohort"] == cohort) & (all_delta_subjects["column"] == col)]
            x = pd.to_numeric(sub["delta"], errors="coerce").dropna()
            if not x.empty:
                data.append(x.values)
                labels.append(cohort)
        if data:
            ax.boxplot(data, labels=labels, vert=False, showfliers=False)
            ax.axvline(0, linewidth=1)
            ax.set_title(f"Δ{col}", fontsize=9)
            ax.grid(True, axis="x", alpha=0.25)
        else:
            ax.axis("off")

    fig.suptitle("Shared longitudinal metadata deltas by cohort", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    saved = []
    for fmt in formats:
        path = outdir / f"ADNI_HABS_shared_delta_boxplots.{fmt}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        saved.append(str(path))
    plt.close(fig)
    return saved


def plot_availability_heatmap(summary: pd.DataFrame, outdir: Path, formats: List[str], dpi: int):
    usable = summary[summary["usable_longitudinal"]].copy()
    if usable.empty:
        return []

    # top by cohort and n pairs
    usable = usable.sort_values(["cohort", "n_pairs", "sd_delta"], ascending=[True, False, False])
    top = usable.groupby("cohort").head(25)
    labels = (top["cohort"] + " | " + top["column"]).tolist()
    vals = top[["n_pairs", "n_unique_delta"]].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(9, max(6, 0.25 * len(labels))))
    im = ax.imshow(vals, aspect="auto")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["N pairs", "N unique Δ"])
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels, fontsize=7)

    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            ax.text(j, i, f"{int(vals[i,j])}", ha="center", va="center", fontsize=7)

    ax.set_title("Usable longitudinal metadata columns")
    cb = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
    cb.set_label("Count")
    fig.tight_layout()

    saved = []
    for fmt in formats:
        path = outdir / f"longitudinal_metadata_availability_heatmap.{fmt}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        saved.append(str(path))
    plt.close(fig)
    return saved


def main():
    args = parse_args()

    data_root = Path(args.data_root).expanduser().resolve()
    results_root = Path(args.results_root).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    cohorts = [c.strip() for c in args.cohorts.split(",") if c.strip()]
    preferred = [c.strip() for c in args.preferred_columns.split(",") if c.strip()]
    formats = [f.strip().lower().lstrip(".") for f in args.formats.split(",") if f.strip()]

    print("=" * 100)
    print("Longitudinal metadata discovery and plotting")
    print("=" * 100)
    print("DATA_ROOT:", data_root)
    print("RESULTS_ROOT:", results_root)
    print("OUTDIR:", outdir)
    print("COHORTS:", cohorts)
    print("MIN_PAIRS:", args.min_pairs)

    file_summary_rows = []
    column_summaries = []
    subject_delta_tables = []
    chosen_rows = []
    plot_paths = []

    for cohort in cohorts:
        print("\n" + "=" * 100)
        print("COHORT", cohort)
        print("=" * 100)

        files = candidate_files_for_cohort(data_root, results_root, cohort, args.feature_set, args.max_files_per_cohort)
        print("candidate files:", len(files))

        for path in files:
            df = load_table(path)
            if df is None or df.empty:
                continue

            sv, diag = infer_subject_visit(df, cohort)

            tag = safe_tag(path)
            row = {
                "cohort": cohort,
                "file": str(path),
                "file_tag": tag,
                "shape_rows": int(df.shape[0]),
                "shape_cols": int(df.shape[1]),
                **diag,
            }
            file_summary_rows.append(row)

            if sv is None or diag["subjects_ge2_visits"] <= 0:
                continue

            print(f"\n{cohort} {tag}")
            print(" ", path)
            print(" ", diag)

            summary, delta_subjects = summarize_longitudinal_columns(df, sv, cohort, tag, args.min_pairs)
            if summary.empty:
                continue

            summary["file"] = str(path)
            delta_subjects["file"] = str(path)

            column_summaries.append(summary)
            subject_delta_tables.append(delta_subjects)

            chosen = pick_columns_for_plots(summary, preferred, args.top_n)
            if not chosen.empty:
                chosen_rows.append(chosen)
                cols = chosen["column"].tolist()
                plot_paths.extend(plot_spaghetti(delta_subjects, cohort, tag, cols, outdir, formats, args.dpi))
                plot_paths.extend(plot_delta_histograms(delta_subjects, cohort, tag, cols, outdir, formats, args.dpi))

            # To avoid producing too many plots per cohort, plot the first/best few files only.
            if len([r for r in chosen_rows if not r.empty and r.iloc[0]["cohort"] == cohort]) >= 3:
                pass

    file_summary = pd.DataFrame(file_summary_rows)
    file_summary_path = outdir / "longitudinal_metadata_file_summary.csv"
    file_summary.to_csv(file_summary_path, index=False)

    if column_summaries:
        all_summary = pd.concat(column_summaries, ignore_index=True, sort=False)
    else:
        all_summary = pd.DataFrame()
    column_summary_path = outdir / "longitudinal_metadata_column_summary.csv"
    all_summary.to_csv(column_summary_path, index=False)

    if subject_delta_tables:
        all_delta = pd.concat(subject_delta_tables, ignore_index=True, sort=False)
    else:
        all_delta = pd.DataFrame()
    delta_path = outdir / "longitudinal_metadata_delta_subject_table_ALL.csv"
    all_delta.to_csv(delta_path, index=False)

    if chosen_rows:
        chosen_all = pd.concat(chosen_rows, ignore_index=True, sort=False)
    else:
        chosen_all = pd.DataFrame()
    chosen_path = outdir / "longitudinal_metadata_chosen_plot_columns.csv"
    chosen_all.to_csv(chosen_path, index=False)

    if not all_summary.empty:
        plot_paths.extend(plot_availability_heatmap(all_summary, outdir, formats, args.dpi))

    if not all_delta.empty and not chosen_all.empty:
        plot_paths.extend(plot_shared_boxplots(all_delta, chosen_all, outdir, formats, args.dpi))

    manifest = pd.DataFrame([{
        "file_summary": str(file_summary_path),
        "column_summary": str(column_summary_path),
        "delta_subject_table": str(delta_path),
        "chosen_plot_columns": str(chosen_path),
        "plots": ";".join(plot_paths),
    }])
    manifest_path = outdir / "longitudinal_metadata_discovery_manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    print("\nSaved:")
    print(file_summary_path)
    print(column_summary_path)
    print(delta_path)
    print(chosen_path)
    print(manifest_path)

    print("\nPlots:")
    for p in plot_paths:
        print(p)

    print("\nTop usable columns:")
    if not all_summary.empty:
        print(
            all_summary[all_summary["usable_longitudinal"]]
            .sort_values(["cohort", "n_pairs", "sd_delta"], ascending=[True, False, False])
            .head(100)
            .to_string(index=False)
        )

    print("\nDONE")


if __name__ == "__main__":
    main()
