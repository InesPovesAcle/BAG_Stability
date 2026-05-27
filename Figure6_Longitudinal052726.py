#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run Figure 6 longitudinal biological validation from standalone metadata files.

By default this script looks in the same folder as the script for:
  - ADNI_metadata.csv / .tsv / .xlsx / .xls
  - HABS_metadata.csv / .tsv / .xlsx / .xls
  - AD_DECODE_metadata.csv / .tsv / .xlsx / .xls
  - ADRC_metadata.csv / .tsv / .xlsx / .xls

It can be run from Spyder with runfile(...) and no command-line arguments.
You can also override paths with command-line options.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

COHORTS = ["ADNI", "HABS", "AD_DECODE", "ADRC"]
FIGURE_FORMATS = ["png", "pdf"]
MIN_N = 10
MIN_UNIQUE_X = 4

SCRIPT_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()

# Default metadata paths. Put these files in the same folder as this script,
# or edit these paths to your exact metadata locations.
DEFAULT_METADATA_PATHS = {
    "ADNI": SCRIPT_DIR / "ADNI_metadata.csv",
    "HABS": SCRIPT_DIR / "HABS_metadata.csv",
    "AD_DECODE": SCRIPT_DIR / "AD_DECODE_metadata.csv",
    "ADRC": SCRIPT_DIR / "ADRC_metadata.csv",
}

DEFAULT_OUTDIR = SCRIPT_DIR / "Figure6_Longitudinal_from_metadata"

CBAG_PRIORITY = [
    "cBAG_global_raw_clean", "cBAG_global",
    "cBAG_oof_global_raw_clean", "cBAG_oof_global",
    "cBAG_bias_corrected", "cBAG_BiasCorrected", "cBAG",
    "bag", "brain_age_gap", "BrainAgeGap",
]

FAMILY_PATTERNS = {
    "hippocampus_volume": [r"hippocampus.*volume.*abs.*sum", r"hippocampus.*volume.*abs.*mean", r"hippocampus.*volume", r"hippocampal.*volume"],
    "hippocampus_FA": [r"hippocampus.*FA.*mean", r"hippocampus.*fractional", r"hippocampal.*FA"],
    "brain_volume": [r"total.*brain.*volume.*abs", r"whole.*brain.*volume", r"\btbv\b", r"brain.*volume"],
    "brain_FA": [r"total.*brain.*FA.*mean", r"whole.*brain.*FA", r"nodewise_FA_mean", r"brain.*FA"],
    "clustering": [r"^Clustering_Coeff$", r"clustering.*coeff", r"cluster"],
    "path_length": [r"^Path_Length$", r"path.*length"],
    "global_efficiency": [r"^Global_Efficiency$", r"global.*efficiency"],
    "local_efficiency": [r"^Local_Efficiency$", r"local.*efficiency"],
    "memory": [r"memory.*composite", r"memory.*z", r"ravlt|avlt|delayed|recall|limmtotal|ldeltotal"],
    "executive_function": [r"executive.*composite", r"executive.*z", r"trail.*b|trabscor|backward|dspanbac"],
    "processing_speed": [r"processing.*speed", r"trail.*a|traascor|digit.*symbol|digitscor"],
    "global_cognition": [r"global.*cog", r"moca|mmse|cdrsb|cdr"],
    "tau_ptau": [r"ptau|p_tau|tau"],
    "amyloid_abeta": [r"abeta|a_beta|amyloid|ab42|ab40"],
    "nfl": [r"\bnfl\b|neurofilament"],
    "gfap": [r"gfap"],
}

PLOT_FAMILIES = [
    "hippocampus_volume", "hippocampus_FA", "brain_volume", "brain_FA",
    "clustering", "path_length", "memory", "global_cognition",
    "tau_ptau", "amyloid_abeta", "nfl", "gfap",
]


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
    return s.astype(str).str.strip().replace({"nan": np.nan, "None": np.nan, "<NA>": np.nan, "": np.nan})


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix in {".tsv", ".tab"}:
        return pd.read_csv(path, sep="\t", low_memory=False)
    return pd.read_csv(path, low_memory=False)


def get_cbag_col(df: pd.DataFrame) -> Optional[str]:
    return first_existing(df, CBAG_PRIORITY)


def subject_and_session_cols(df: pd.DataFrame, cohort: str) -> tuple[str, Optional[str]]:
    if cohort == "ADNI":
        subject_candidates = ["PTID", "subject_id", "Subject", "RID", "participant_id", "table1_subject_key"]
        session_candidates = ["connectome_key", "connectome_full_key", "graph_id", "session_id", "VISCODE", "VISCODE2", "EXAMDATE", "table1_session_key"]
    elif cohort == "HABS":
        subject_candidates = ["Subject", "subject_id", "participant_id", "PTID", "RID", "table1_subject_key"]
        session_candidates = ["runno", "connectome_key", "connectome_full_key", "graph_id", "session_id", "VISCODE", "EXAMDATE", "table1_session_key"]
    elif cohort == "AD_DECODE":
        subject_candidates = ["Subject", "subject_id", "participant_id", "PTID", "RID", "ID", "table1_subject_key"]
        session_candidates = ["session_id", "visit", "VISCODE", "VISCODE2", "EXAMDATE", "runno", "connectome_key", "connectome_full_key", "graph_id", "table1_session_key"]
    elif cohort == "ADRC":
        subject_candidates = ["Subject", "subject_id", "participant_id", "PTID", "RID", "ID", "table1_subject_key"]
        session_candidates = ["session_id", "visit", "VISCODE", "VISCODE2", "EXAMDATE", "runno", "connectome_key", "connectome_full_key", "graph_id", "table1_session_key"]
    else:
        subject_candidates = ["Subject", "subject_id", "participant_id", "PTID", "RID", "ID", "table1_subject_key"]
        session_candidates = ["session_id", "visit", "VISCODE", "VISCODE2", "EXAMDATE", "runno", "connectome_key", "connectome_full_key", "graph_id", "table1_session_key"]

    subject = first_existing(df, subject_candidates)
    session = first_existing(df, session_candidates)
    if subject is None:
        raise KeyError(
            f"Could not identify subject column for {cohort}. "
            f"Rename/add one of: {subject_candidates}"
        )
    return subject, session

def age_col(df: pd.DataFrame) -> Optional[str]:
    return first_existing(df, ["age_true", "age_for_table1", "Age", "AGE", "age", "VISIT_AGE", "SUBJECT_AGE_SCREEN"])


def sex_col(df: pd.DataFrame) -> Optional[str]:
    return first_existing(df, ["sex_label", "sex", "Sex", "PTGENDER", "gender"])


def normalize_sex(x: object) -> float:
    if pd.isna(x):
        return np.nan
    s = str(x).strip().lower()
    if s in {"f", "female", "2", "2.0"} or s.startswith("f"):
        return 1.0
    if s in {"m", "male", "1", "1.0"} or s.startswith("m"):
        return 0.0
    return np.nan


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


def family_label(fam: str) -> str:
    return fam.replace("_", " ").title().replace("Fa", "FA").replace("Nfl", "NfL").replace("Gfap", "GFAP")


def p_text(p: float) -> str:
    if pd.isna(p): return "p=NA"
    if p < 1e-4: return "p<1e-4"
    if p < 0.001: return "p<0.001"
    return f"p={p:.3g}"


def q_text(q: float) -> str:
    if pd.isna(q): return "q=NA"
    if q < 1e-4: return "q<1e-4"
    if q < 0.001: return "q<0.001"
    return f"q={q:.3g}"


def find_family_variables(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    blocked = ["cbag", "bag_raw", "pred", "age_true", "prediction", "fold"]
    for family, patterns in FAMILY_PATTERNS.items():
        for col in df.columns:
            low = normalize_name(col)
            if any(tok in low for tok in blocked):
                continue
            for rank, pat in enumerate(patterns):
                if re.search(pat, str(col), flags=re.IGNORECASE) or re.search(pat, low, flags=re.IGNORECASE):
                    x = pd.to_numeric(df[col], errors="coerce")
                    if x.notna().sum() >= MIN_N and x.nunique(dropna=True) >= MIN_UNIQUE_X:
                        rows.append({
                            "family": family,
                            "variable": col,
                            "preference_rank": rank,
                            "n_nonmissing": int(x.notna().sum()),
                            "n_unique": int(x.nunique(dropna=True)),
                        })
                    break
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["family", "preference_rank", "n_nonmissing"], ascending=[True, True, False])
        out = out.drop_duplicates(["family", "variable"], keep="first")
    return out


def build_delta_table(df: pd.DataFrame, cohort: str) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    logs = []
    subj_col, sess_col = subject_and_session_cols(df, cohort)
    a_col = age_col(df)
    c_col = get_cbag_col(df)
    s_col = sex_col(df)
    if a_col is None:
        raise KeyError(f"No age column found for {cohort}. Rename/add one of: age_true, Age, AGE, age, VISIT_AGE")
    if c_col is None:
        raise KeyError(f"No cBAG column found for {cohort}. Rename/add one of: {CBAG_PRIORITY}")

    logs.append(f"subject_col={subj_col}; session_col={sess_col or 'synthetic row index'}; age_col={a_col}; cbag_col={c_col}; sex_col={s_col}")
    fam_vars = find_family_variables(df)
    var_list = fam_vars["variable"].tolist() if not fam_vars.empty else []
    logs.append(f"candidate biological variables={len(var_list)}")

    work = df.copy()
    work["_row_index_session"] = np.arange(len(work)).astype(str)
    work["_subject"] = as_key(work[subj_col])
    work["_session"] = as_key(work[sess_col]) if sess_col else work["_row_index_session"]
    work["_age"] = pd.to_numeric(work[a_col], errors="coerce")
    work["_cbag"] = pd.to_numeric(work[c_col], errors="coerce")
    work["_sex_numeric"] = work[s_col].map(normalize_sex) if s_col else np.nan
    work = work.dropna(subset=["_subject", "_age", "_cbag"]).sort_values(["_subject", "_age", "_session"])
    work = work.drop_duplicates(["_subject", "_session"], keep="first")

    delta_rows = []
    for subject, sub in work.groupby("_subject", sort=False):
        sub = sub.sort_values("_age")
        if len(sub) < 2:
            continue
        baseline = sub.iloc[0]
        for _, fu in sub.iloc[1:].iterrows():
            dy = float(fu["_age"] - baseline["_age"])
            if not np.isfinite(dy) or dy <= 0:
                continue
            row = {
                "cohort": cohort,
                "subject_key": subject,
                "baseline_session": baseline["_session"],
                "followup_session": fu["_session"],
                "baseline_age": baseline["_age"],
                "followup_age": fu["_age"],
                "delta_years": dy,
                "baseline_cBAG": baseline["_cbag"],
                "followup_cBAG": fu["_cbag"],
                "delta_cBAG": fu["_cbag"] - baseline["_cbag"],
                "annualized_delta_cBAG": (fu["_cbag"] - baseline["_cbag"]) / dy,
                "sex_numeric": baseline["_sex_numeric"],
            }
            for var in var_list:
                b = pd.to_numeric(pd.Series([baseline.get(var, np.nan)]), errors="coerce").iloc[0]
                f = pd.to_numeric(pd.Series([fu.get(var, np.nan)]), errors="coerce").iloc[0]
                row[f"baseline__{var}"] = b
                row[f"followup__{var}"] = f
                row[f"delta__{var}"] = f - b if pd.notna(f) and pd.notna(b) else np.nan
                row[f"annualized_delta__{var}"] = (f - b) / dy if pd.notna(f) and pd.notna(b) else np.nan
            delta_rows.append(row)

    delta = pd.DataFrame(delta_rows)
    if delta.empty:
        logs.append("No valid longitudinal follow-up rows after filtering. Check that each subject has at least two visits with increasing age and nonmissing cBAG.")
    else:
        logs.append(f"delta rows={len(delta)}; subjects={delta['subject_key'].nunique()}; median_delta_years={delta['delta_years'].median():.2f}")
    return delta, fam_vars, logs


def ols_beta_p(y: pd.Series, x: pd.Series, covars: pd.DataFrame) -> dict:
    covars = covars.copy()
    # Keep only usable covariates; prevents all-NaN sex from deleting all rows.
    for c in list(covars.columns):
        v = pd.to_numeric(covars[c], errors="coerce")
        if v.notna().sum() < MIN_N or v.nunique(dropna=True) <= 1:
            covars = covars.drop(columns=[c])
        else:
            covars[c] = v

    dat = pd.concat([y.rename("y"), x.rename("x"), covars], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if len(dat) < MIN_N or dat["x"].nunique() < MIN_UNIQUE_X or dat["y"].nunique() < 2:
        return {"n_adj": len(dat), "beta": np.nan, "se": np.nan, "t": np.nan, "p": np.nan, "std_beta": np.nan}

    yv = dat["y"].to_numpy(dtype=float)
    X_parts = [np.ones(len(dat)), dat["x"].to_numpy(dtype=float)]
    for c in covars.columns:
        v = dat[c].to_numpy(dtype=float)
        if np.nanstd(v) > 0:
            X_parts.append(v)
    X = np.column_stack(X_parts)
    if X.shape[0] <= X.shape[1] + 1:
        return {"n_adj": len(dat), "beta": np.nan, "se": np.nan, "t": np.nan, "p": np.nan, "std_beta": np.nan}
    try:
        beta = np.linalg.lstsq(X, yv, rcond=None)[0]
        resid = yv - X @ beta
        dof = X.shape[0] - X.shape[1]
        mse = float(np.sum(resid ** 2) / dof)
        cov_beta = mse * np.linalg.pinv(X.T @ X)
        se = float(np.sqrt(cov_beta[1, 1]))
        tval = float(beta[1] / se) if se > 0 else np.nan
        pval = float(2 * stats.t.sf(abs(tval), dof)) if np.isfinite(tval) else np.nan
        sx, sy = float(dat["x"].std(ddof=0)), float(dat["y"].std(ddof=0))
        std_beta = float(beta[1] * sx / sy) if sy > 0 else np.nan
        return {"n_adj": len(dat), "beta": float(beta[1]), "se": se, "t": tval, "p": pval, "std_beta": std_beta}
    except Exception:
        return {"n_adj": len(dat), "beta": np.nan, "se": np.nan, "t": np.nan, "p": np.nan, "std_beta": np.nan}


def association_stats(delta: pd.DataFrame, fam_vars: pd.DataFrame, cohort: str) -> pd.DataFrame:
    rows = []
    if delta.empty or fam_vars.empty:
        return pd.DataFrame()
    for _, fv in fam_vars.iterrows():
        family, var = fv["family"], fv["variable"]
        dcol, acol, bcol = f"delta__{var}", f"annualized_delta__{var}", f"baseline__{var}"
        if dcol not in delta.columns:
            continue
        x = pd.to_numeric(delta[dcol], errors="coerce")
        y = pd.to_numeric(delta["delta_cBAG"], errors="coerce")
        tmp = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
        if len(tmp) < MIN_N or tmp["x"].nunique() < MIN_UNIQUE_X or tmp["y"].nunique() < 2:
            continue
        r, p = stats.pearsonr(tmp["x"], tmp["y"])

        xa = pd.to_numeric(delta[acol], errors="coerce")
        ya = pd.to_numeric(delta["annualized_delta_cBAG"], errors="coerce")
        tmpa = pd.DataFrame({"x": xa, "y": ya}).replace([np.inf, -np.inf], np.nan).dropna()
        if len(tmpa) >= MIN_N and tmpa["x"].nunique() >= MIN_UNIQUE_X and tmpa["y"].nunique() >= 2:
            r_ann, p_ann = stats.pearsonr(tmpa["x"], tmpa["y"])
        else:
            r_ann, p_ann = np.nan, np.nan

        covars = pd.DataFrame({
            "delta_years": pd.to_numeric(delta["delta_years"], errors="coerce"),
            "baseline_age": pd.to_numeric(delta["baseline_age"], errors="coerce"),
            "sex_numeric": pd.to_numeric(delta["sex_numeric"], errors="coerce"),
        })
        if bcol in delta.columns:
            covars["baseline_metric"] = pd.to_numeric(delta[bcol], errors="coerce")
        adj = ols_beta_p(y, x, covars)
        rows.append({
            "cohort": cohort,
            "family": family,
            "family_label": family_label(family),
            "variable": var,
            "delta_variable": dcol,
            "annualized_delta_variable": acol,
            "n": int(len(tmp)),
            "n_subjects": int(delta.loc[tmp.index, "subject_key"].nunique()),
            "pearson_r_delta": float(r),
            "pearson_p_delta": float(p),
            "abs_pearson_r_delta": float(abs(r)),
            "pearson_r_annualized": float(r_ann) if pd.notna(r_ann) else np.nan,
            "pearson_p_annualized": float(p_ann) if pd.notna(p_ann) else np.nan,
            "adjusted_beta_delta_metric": adj["beta"],
            "adjusted_se": adj["se"],
            "adjusted_t": adj["t"],
            "adjusted_p": adj["p"],
            "adjusted_std_beta": adj["std_beta"],
            "n_adjusted": adj["n_adj"],
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out["fdr_q_delta"] = fdr_bh(out["pearson_p_delta"].values)
        out["fdr_q_annualized"] = fdr_bh(out["pearson_p_annualized"].values)
        out["fdr_q_adjusted"] = fdr_bh(out["adjusted_p"].values)
        out = out.sort_values(["cohort", "family", "fdr_q_adjusted", "adjusted_p", "abs_pearson_r_delta"], na_position="last")
    return out


def select_best_for_plot(stats_df: pd.DataFrame) -> pd.DataFrame:
    if stats_df.empty:
        return pd.DataFrame()
    df = stats_df[stats_df["family"].isin(PLOT_FAMILIES)].copy()
    if df.empty:
        return df
    df = df.sort_values(["cohort", "family", "fdr_q_adjusted", "adjusted_p", "abs_pearson_r_delta"], ascending=[True, True, True, True, False], na_position="last")
    return df.groupby(["cohort", "family"], as_index=False).head(1).copy()


def plot_heatmap(all_stats: pd.DataFrame, fig_outdir: Path) -> None:
    df = select_best_for_plot(all_stats)
    if df.empty:
        print("[WARN] No heatmap data")
        return
    families = [f for f in PLOT_FAMILIES if f in set(df["family"])]
    cohorts = [c for c in COHORTS if c in set(df["cohort"])]
    mat = pd.DataFrame(index=families, columns=cohorts, dtype=float)
    ann = pd.DataFrame("", index=families, columns=cohorts)
    for _, r in df.iterrows():
        fam, cohort = r["family"], r["cohort"]
        val = r["adjusted_std_beta"] if pd.notna(r["adjusted_std_beta"]) else r["pearson_r_delta"]
        mat.loc[fam, cohort] = val
        q = r.get("fdr_q_adjusted", np.nan)
        star = "***" if pd.notna(q) and q < 0.001 else "**" if pd.notna(q) and q < 0.01 else "*" if pd.notna(q) and q < 0.05 else ""
        ann.loc[fam, cohort] = f"{val:.2f}{star}\nn={int(r['n'])}\n{str(r['variable'])[:18]}"
    data = mat.to_numpy(dtype=float)
    vmax = max(0.05, np.nanmax(np.abs(data)) if np.isfinite(data).any() else 1.0)
    fig_h = max(5, 0.55 * len(families) + 1.5)
    fig, ax = plt.subplots(figsize=(7.5, fig_h))
    im = ax.imshow(data, aspect="auto", vmin=-vmax, vmax=vmax, cmap="coolwarm")
    ax.set_title("Figure 6. Longitudinal ΔcBAG biological validation")
    ax.set_xticks(np.arange(len(cohorts)))
    ax.set_xticklabels(cohorts)
    ax.set_yticks(np.arange(len(families)))
    ax.set_yticklabels([family_label(f) for f in families])
    for i, fam in enumerate(families):
        for j, cohort in enumerate(cohorts):
            if ann.loc[fam, cohort]:
                ax.text(j, i, ann.loc[fam, cohort], ha="center", va="center", fontsize=7)
    ax.set_xticks(np.arange(-0.5, len(cohorts), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(families), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1)
    ax.tick_params(which="minor", bottom=False, left=False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Adjusted standardized β\nor Pearson r if adjusted unavailable")
    fig.tight_layout()
    for fmt in FIGURE_FORMATS:
        out = fig_outdir / f"Figure6_longitudinal_heatmap.{fmt}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"[DONE] {out}")
    plt.close(fig)


def plot_scatter_for_cohort(delta: pd.DataFrame, best: pd.DataFrame, cohort: str, fig_outdir: Path) -> None:
    rows = best[(best["cohort"].eq(cohort)) & (best["family"].isin(PLOT_FAMILIES))].copy()
    if delta.empty or rows.empty:
        return
    rows = rows.sort_values(["fdr_q_adjusted", "adjusted_p", "abs_pearson_r_delta"], ascending=[True, True, False], na_position="last").head(8)
    n = len(rows)
    ncols = 4 if n >= 4 else max(1, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4*ncols, 3.5*nrows), squeeze=False)
    for ax, (_, r) in zip(axes.ravel(), rows.iterrows()):
        x = pd.to_numeric(delta[r["delta_variable"]], errors="coerce")
        y = pd.to_numeric(delta["delta_cBAG"], errors="coerce")
        tmp = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
        ax.scatter(tmp["x"], tmp["y"], alpha=0.75, s=24)
        if len(tmp) >= 3 and tmp["x"].nunique() > 1:
            slope, intercept, *_ = stats.linregress(tmp["x"], tmp["y"])
            xx = np.linspace(tmp["x"].min(), tmp["x"].max(), 100)
            ax.plot(xx, intercept + slope * xx, linewidth=1.5)
        ax.axhline(0, linewidth=0.7, linestyle="--")
        ax.axvline(0, linewidth=0.7, linestyle="--")
        ax.set_title(f"{family_label(r['family'])}\nr={r['pearson_r_delta']:.2f}, {p_text(r['pearson_p_delta'])}, {q_text(r['fdr_q_delta'])}", fontsize=9)
        ax.set_xlabel("Δ " + str(r["variable"])[:32])
        ax.set_ylabel("Δ cBAG")
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle(f"Figure 6 longitudinal scatter panels — {cohort}", y=1.02)
    fig.tight_layout()
    for fmt in FIGURE_FORMATS:
        out = fig_outdir / f"Figure6_longitudinal_scatter_{cohort}.{fmt}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"[DONE] {out}")
    plt.close(fig)


def resolve_existing_path(path: Path) -> Optional[Path]:
    """Return an existing path, trying common table extensions when needed."""
    path = Path(path).expanduser()
    if path.exists():
        return path
    if path.suffix:
        return None
    for suffix in [".csv", ".tsv", ".tab", ".xlsx", ".xls"]:
        candidate = path.with_suffix(suffix)
        if candidate.exists():
            return candidate
    return None


def metadata_paths_from_args(args: argparse.Namespace) -> dict[str, Path]:
    requested = {
        "ADNI": args.adni_metadata,
        "HABS": args.habs_metadata,
        "AD_DECODE": args.ad_decode_metadata,
        "ADRC": args.adrc_metadata,
    }
    out = {}
    for cohort in COHORTS:
        raw = requested.get(cohort) or DEFAULT_METADATA_PATHS[cohort]
        found = resolve_existing_path(raw)
        if found is not None:
            out[cohort] = found
    return out


def run_one(path: Path, cohort: str, outdir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    print("\n" + "=" * 90)
    print(f"{cohort}: {path}")
    print("=" * 90)
    df = read_table(path)
    delta, fam_vars, logs = build_delta_table(df, cohort)
    stats_df = association_stats(delta, fam_vars, cohort)
    merged_outdir = outdir / "merged_longitudinal"
    stats_outdir = outdir / "stats"
    merged_outdir.mkdir(parents=True, exist_ok=True)
    stats_outdir.mkdir(parents=True, exist_ok=True)
    delta.to_csv(merged_outdir / f"longitudinal_delta_table_{cohort}.csv", index=False)
    fam_vars.to_csv(merged_outdir / f"longitudinal_candidate_variables_{cohort}.csv", index=False)
    stats_df.to_csv(stats_outdir / f"longitudinal_associations_{cohort}.csv", index=False)
    for msg in logs:
        print("[INFO]", msg)
    print(f"[DONE] delta rows={len(delta)}; stats rows={len(stats_df)}")
    return delta, stats_df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Figure 6 longitudinal validation from ADNI, HABS, AD_DECODE, and/or ADRC metadata files."
    )
    parser.add_argument("--adni_metadata", default=None, type=Path)
    parser.add_argument("--habs_metadata", default=None, type=Path)
    parser.add_argument("--ad_decode_metadata", default=None, type=Path)
    parser.add_argument("--adrc_metadata", default=None, type=Path)
    parser.add_argument("--outdir", default=DEFAULT_OUTDIR, type=Path)
    parser.add_argument("--require_all", action="store_true", help="Fail if any of the four cohort metadata files are missing.")

    # parse_known_args makes the script friendlier in Spyder/Jupyter/runfile contexts.
    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"[INFO] Ignoring unknown arguments from interactive environment: {unknown}")

    metadata_paths = metadata_paths_from_args(args)
    missing = [c for c in COHORTS if c not in metadata_paths]

    if args.require_all and missing:
        searched = "\n".join(f"  {c}: {DEFAULT_METADATA_PATHS[c]}" for c in missing)
        raise FileNotFoundError("Missing required metadata files:\n" + searched)

    if missing:
        for c in missing:
            print(f"[WARN] Skipping {c}; metadata file not found at default path {DEFAULT_METADATA_PATHS[c]}")

    if not metadata_paths:
        searched = "\n".join(f"  {c}: {DEFAULT_METADATA_PATHS[c]}" for c in COHORTS)
        raise FileNotFoundError(
            "No metadata files found. Either put metadata files next to this script or pass explicit paths.\n"
            "Default paths searched:\n" + searched +
            "\n\nExample:\n"
            "python Figure6_LongitudinalB.py --adni_metadata /path/ADNI_metadata.csv --habs_metadata /path/HABS_metadata.csv"
        )

    args.outdir.mkdir(parents=True, exist_ok=True)
    fig_outdir = args.outdir / "figures"
    stats_outdir = args.outdir / "stats"
    fig_outdir.mkdir(parents=True, exist_ok=True)
    stats_outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 90)
    print("FIGURE 6 LONGITUDINAL BIOLOGICAL VALIDATION")
    print("=" * 90)
    print("Output:", args.outdir)
    print("Cohorts to run:", ", ".join(metadata_paths.keys()))

    delta_cache = {}
    all_stats = []
    logs = []

    for cohort in COHORTS:
        path = metadata_paths.get(cohort)
        if path is None:
            continue
        try:
            delta, stats_df = run_one(path, cohort, args.outdir)
            delta_cache[cohort] = delta
            logs.append({"cohort": cohort, "metadata_path": str(path), "delta_rows": len(delta), "stats_rows": len(stats_df), "status": "ok"})
            if not stats_df.empty:
                all_stats.append(stats_df)
        except Exception as exc:
            print(f"[ERROR] {cohort}: {exc}")
            logs.append({"cohort": cohort, "metadata_path": str(path), "delta_rows": np.nan, "stats_rows": np.nan, "status": f"ERROR: {exc}"})

    all_stats_df = pd.concat(all_stats, ignore_index=True, sort=False) if all_stats else pd.DataFrame()
    if not all_stats_df.empty:
        all_stats_df["fdr_q_adjusted_all_tests"] = fdr_bh(all_stats_df["adjusted_p"].values)
        all_stats_df["fdr_q_delta_all_tests"] = fdr_bh(all_stats_df["pearson_p_delta"].values)
        all_stats_df = all_stats_df.sort_values(["cohort", "fdr_q_adjusted", "adjusted_p"], na_position="last")
    all_stats_path = stats_outdir / "longitudinal_associations_all.csv"
    all_stats_df.to_csv(all_stats_path, index=False)
    pd.DataFrame(logs).to_csv(args.outdir / "longitudinal_run_log.csv", index=False)
    print("\n[DONE]", all_stats_path)

    best = select_best_for_plot(all_stats_df)
    plot_heatmap(all_stats_df, fig_outdir)
    for cohort in COHORTS:
        plot_scatter_for_cohort(delta_cache.get(cohort, pd.DataFrame()), best, cohort, fig_outdir)

    readme = args.outdir / "README.md"
    readme.write_text(
        "# Figure 6 longitudinal biological validation from standalone metadata files\n\n"
        "Inputs are any available ADNI, HABS, AD_DECODE, and ADRC metadata files. Missing cohorts are skipped unless `--require_all` is used.\n\n"
        "Baseline is the earliest cBAG visit per subject. Each later visit is compared with baseline.\n\n"
        "Main adjusted model: `delta_cBAG ~ delta_metric + delta_years + baseline_age + sex + baseline_metric`, "
        "with unusable/all-missing covariates automatically dropped.\n\n"
        "Main outputs:\n"
        "- `stats/longitudinal_associations_all.csv`\n"
        "- `merged_longitudinal/longitudinal_delta_table_<COHORT>.csv`\n"
        "- `merged_longitudinal/longitudinal_candidate_variables_<COHORT>.csv`\n"
        "- `figures/Figure6_longitudinal_heatmap.png/pdf`\n"
        "- `figures/Figure6_longitudinal_scatter_<COHORT>.png/pdf`\n"
    )
    print("[DONE]", readme)


if __name__ == "__main__":
    main()
