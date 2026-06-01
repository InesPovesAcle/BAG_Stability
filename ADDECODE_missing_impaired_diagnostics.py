#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diagnose why AD_DECODE MCI/AD subjects with connectomes are absent from
subject_level_validation_input[_enriched_for_Figure4].csv.

This script does not modify validation files. It writes QA CSVs showing:
  1) which harmonized-metadata impaired subjects are absent from validation;
  2) whether they have connectomes;
  3) whether they appear in any AD_DECODE prediction/results CSV;
  4) whether those candidate rows contain cBAG/BAG prediction columns.

Run:
  python ADDECODE_missing_impaired_diagnostics.py
"""
from __future__ import annotations

import os
import re
from pathlib import Path
import pandas as pd

WORK = Path(os.environ.get("WORK", "/mnt/newStor/paros/paros_WORK"))
BASE = WORK / "ines"
RESULTS_ROOT = BASE / "results"
DATA_ROOT = BASE / "data"

PRED_ROOT = RESULTS_ROOT / "BrainAgePredictionADDECODE_stratified_groupcv_targetnorm_bagbiascorr_oofglobal"
HARM_META = DATA_ROOT / "harmonization" / "harmonized_metadata" / "AD_DECODE_harmonized_metadata.csv"
OUTDIR = RESULTS_ROOT / "Figure5_Final_FromHarmonizedMetadata" / "qa"
OUTDIR.mkdir(parents=True, exist_ok=True)

FEATURE_SETS = [
    "imaging_only",
    "imaging_demographics",
    "imaging_biomarkers",
    "full",
    "full_no_cardiovascular",
]

CBAG_PAT = re.compile(r"(^|_)(c?bag|cbag|bag)(_|$)|cBAG|BAG")
SID_PAT = re.compile(r"S\d{5}", re.I)

KEY_COL_HINTS = [
    "graph_id", "connectome_key", "connectome_full_key", "MRI_Exam", "MRI_Exam_key",
    "Subject_ID", "subject_id", "match_id", "ID", "CONNECTOME_KEY_CLEAN",
    "CONNECTOME_KEY_USED_FOR_INTERSECTION", "MATCHED_CONNECTOME_FILE",
]

CONNECTOME_SEARCH_ROOTS = [
    DATA_ROOT / "harmonization" / "AD_DECODE" / "connectomes",
    DATA_ROOT / "harmonization" / "ADDECODE" / "connectomes",
    DATA_ROOT / "pre_harmonization" / "AD_DECODE",
]


def root_s(x) -> str | None:
    m = SID_PAT.search(str(x))
    return m.group(0).upper() if m else None


def norm_mri_exam_to_s(x) -> str | None:
    if pd.isna(x):
        return None
    s = str(x).strip()
    r = root_s(s)
    if r:
        return r
    if re.fullmatch(r"\d+(?:\.0)?", s):
        return "S" + str(int(float(s))).zfill(5)
    return None


def row_roots(df: pd.DataFrame) -> pd.Series:
    roots = pd.Series([None] * len(df), index=df.index, dtype="object")
    for c in KEY_COL_HINTS:
        if c not in df.columns:
            continue
        if c in {"MRI_Exam", "MRI_Exam_key"}:
            r = df[c].map(norm_mri_exam_to_s)
        else:
            r = df[c].map(root_s)
        roots = roots.combine_first(r)
    return roots


def validation_path(feature_set: str) -> Path:
    base = PRED_ROOT / f"ablation_{feature_set}" / "validation_figures"
    enriched = base / "subject_level_validation_input_enriched_for_Figure4.csv"
    raw = base / "subject_level_validation_input.csv"
    return enriched if enriched.exists() else raw


def connectome_hits(sid: str) -> list[str]:
    hits = []
    for root in CONNECTOME_SEARCH_ROOTS:
        if root.exists():
            hits.extend(str(p) for p in root.rglob(f"{sid}*conn_plain*.csv") if p.is_file())
    return sorted(set(hits))


def find_hits_in_csv(path: Path, target_roots: set[str]) -> list[dict]:
    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception:
        return []
    if df.empty:
        return []

    roots = row_roots(df)
    # Slow fallback: if row_roots found nothing, scan all object columns for S roots.
    if roots.notna().sum() == 0:
        roots = pd.Series([None] * len(df), index=df.index, dtype="object")
        for c in df.columns:
            if df[c].dtype == object or "id" in str(c).lower() or "key" in str(c).lower():
                r = df[c].map(root_s)
                roots = roots.combine_first(r)
    mask = roots.isin(target_roots)
    if not mask.any():
        return []

    cbag_cols = [c for c in df.columns if CBAG_PAT.search(str(c))]
    out = []
    for sid, sub in df.loc[mask].assign(_root=roots[mask]).groupby("_root"):
        nonnull_cbag = {c: int(pd.to_numeric(sub[c], errors="coerce").notna().sum()) for c in cbag_cols}
        out.append({
            "sid": sid,
            "csv_path": str(path),
            "n_rows_found": len(sub),
            "columns": "|".join(map(str, df.columns[:80])),
            "cbag_columns": "|".join(cbag_cols),
            "nonnull_cbag_counts": str(nonnull_cbag),
        })
    return out


def main():
    if not HARM_META.exists():
        raise FileNotFoundError(HARM_META)
    meta = pd.read_csv(HARM_META, low_memory=False)
    meta["_root"] = row_roots(meta).combine_first(meta.get("MRI_Exam", pd.Series(index=meta.index)).map(norm_mri_exam_to_s))

    impaired = meta[meta["DX_Label_harmonized"].isin(["MCI", "AD"])].copy()
    target_roots = set(impaired["_root"].dropna().astype(str))

    print("AD_DECODE impaired subjects in harmonized metadata:", sorted(target_roots))

    val_rows = []
    all_validation_roots = set()
    for fs in FEATURE_SETS:
        vp = validation_path(fs)
        if not vp.exists():
            val_rows.append({"feature_set": fs, "validation_path": str(vp), "status": "missing"})
            continue
        v = pd.read_csv(vp, low_memory=False)
        v_roots = set(row_roots(v).dropna().astype(str))
        all_validation_roots |= v_roots
        val_rows.append({
            "feature_set": fs,
            "validation_path": str(vp),
            "n_validation_rows": len(v),
            "n_validation_roots": len(v_roots),
            "impaired_present": ",".join(sorted(target_roots & v_roots)),
            "impaired_missing": ",".join(sorted(target_roots - v_roots)),
            "status": "ok",
        })

    missing = target_roots - all_validation_roots

    missing_rows = []
    for _, row in impaired.iterrows():
        sid = row.get("_root")
        if not sid:
            continue
        hits = connectome_hits(str(sid))
        missing_rows.append({
            "sid": sid,
            "DX_Label_harmonized": row.get("DX_Label_harmonized"),
            "MRI_Exam": row.get("MRI_Exam"),
            "ID": row.get("ID"),
            "CONNECTOME_KEY_CLEAN": row.get("CONNECTOME_KEY_CLEAN"),
            "in_any_validation_file": sid in all_validation_roots,
            "n_connectome_hits": len(hits),
            "preferred_plain_connectome": next((h for h in hits if h.endswith(f"/{sid}_conn_plain.csv")), ""),
            "all_connectome_hits": "|".join(hits[:25]),
        })

    print("Missing impaired roots from all validation files:", sorted(missing))
    print("Searching AD_DECODE prediction/results CSVs for missing roots...")
    hit_rows = []
    for p in sorted(PRED_ROOT.rglob("*.csv")):
        hit_rows.extend(find_hits_in_csv(p, missing))

    pd.DataFrame(val_rows).to_csv(OUTDIR / "ADDECODE_validation_impaired_presence_by_model.csv", index=False)
    pd.DataFrame(missing_rows).to_csv(OUTDIR / "ADDECODE_impaired_metadata_connectome_status.csv", index=False)
    pd.DataFrame(hit_rows).to_csv(OUTDIR / "ADDECODE_missing_impaired_hits_in_prediction_csvs.csv", index=False)

    print("\nWrote:")
    print(OUTDIR / "ADDECODE_validation_impaired_presence_by_model.csv")
    print(OUTDIR / "ADDECODE_impaired_metadata_connectome_status.csv")
    print(OUTDIR / "ADDECODE_missing_impaired_hits_in_prediction_csvs.csv")
    print("\nIf the third file is empty, no existing prediction CSV under the AD_DECODE prediction folder contains cBAG rows for the missing impaired subjects.")


if __name__ == "__main__":
    main()
