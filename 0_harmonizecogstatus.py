#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Create final harmonized metadata files for ADNI, ADRC, HABS, and AD_DECODE.

Outputs are saved under:

  $WORK/ines/data/harmonization/harmonized_metadata/

This script does NOT overwrite original metadata files.

Final harmonized status columns:
  NORMCOG_01
  MCI_01
  AD_01
  DEMENTIA_01
  DX_Label_harmonized
  COG_STATUS_SOURCE

AD_DECODE rule:
  risk_for_ad 0/1 = Normal
  risk_for_ad 2   = MCI
  risk_for_ad 3   = AD

HABS rule:
  CDX_Cog 0 = Normal
  CDX_Cog 1 = MCI
  CDX_Cog 2 = AD

ADNI:
  Uses GROUP from ADNI_covars.csv when possible.
  CN = Normal
  EMCI/MCI/LMCI = MCI
  AD = AD
  SMC = Unknown by default

ADRC:
  Uses existing NORMCOG / IMPNOMCI / DEMENTED-style columns.
"""

from pathlib import Path
import os
import re
import pandas as pd


# =============================================================================
# PATHS
# =============================================================================

WORK = Path(os.environ.get("WORK", "/mnt/newStor/paros/paros_WORK"))
BASE = WORK / "ines" / "data" / "harmonization"

OUTDIR = BASE / "harmonized_metadata"
OUTDIR.mkdir(parents=True, exist_ok=True)

METADATA_FILES = {
    "ADNI": BASE / "ADNI" / "metadata" / "ADNI_metadata_with_DWI.csv",
    "ADRC": BASE / "ADRC" / "metadata" / "ADRC_metadata_with_DWI.csv",
    "HABS": BASE / "HABS" / "metadata" / "HABS_metadata_session_level_DWI_filled_cognitive_composites.csv",
    "AD_DECODE": BASE / "AD_DECODE" / "metadata" / "AD_DECODE_metadata_with_DWI.csv",
}

ADNI_COVARS_PATH = BASE / "ADNI" / "metadata" / "ADNI_covars.csv"
ADNI_Y0_Y4_PATH = BASE / "ADNI" / "metadata" / "ADNI_subjects_with_DWI_y0_y4.csv"

COUNT_ADNI_SMC_AS_NORMAL = False
FILTER_ADNI_TO_Y0_Y4 = False


# =============================================================================
# HELPERS
# =============================================================================

def normalize_name(x):
    s = str(x).lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return re.sub(r"_+", "_", s).strip("_")


def first_col(df, candidates):
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


def read_table(path):
    path = Path(path)

    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, low_memory=False)

    if path.suffix.lower() in [".xlsx", ".xls"]:
        return pd.read_excel(path)

    raise ValueError(f"Unsupported file type: {path}")


def clean_id(x):
    if pd.isna(x):
        return pd.NA
    return str(x).strip()


def init_status(df):
    normcog = pd.Series(pd.NA, index=df.index, dtype="Int64")
    mci = pd.Series(pd.NA, index=df.index, dtype="Int64")
    ad = pd.Series(pd.NA, index=df.index, dtype="Int64")
    label = pd.Series("Unknown", index=df.index, dtype="object")
    return normcog, mci, ad, label


def finalize(df, normcog, mci, ad, label, source):
    label.loc[normcog.eq(1)] = "Normal"
    label.loc[mci.eq(1)] = "MCI"
    label.loc[ad.eq(1)] = "AD"

    df["NORMCOG_01"] = normcog
    df["MCI_01"] = mci
    df["AD_01"] = ad
    df["DEMENTIA_01"] = ad
    df["DX_Label_harmonized"] = label
    df["COG_STATUS_SOURCE"] = source

    return df


# =============================================================================
# OPTIONAL ADNI MERGES/FILTERS
# =============================================================================

def merge_adni_covars(meta):
    if not ADNI_COVARS_PATH.exists():
        print(f"[WARN] ADNI covars file not found: {ADNI_COVARS_PATH}")
        return meta.copy(), {
            "adni_covars_status": "missing",
            "adni_covars_rows": 0,
            "adni_covars_matched_rows": 0,
        }

    covars = read_table(ADNI_COVARS_PATH)

    meta_subject_col = first_col(meta, ["subject_id", "PTID", "RID", "Subject", "subject"])
    covars_subject_col = first_col(covars, ["subject_id", "PTID", "RID", "Subject", "subject"])

    if meta_subject_col is None:
        print("[WARN] ADNI metadata has no subject_id/PTID/RID-like column")
        return meta.copy(), {
            "adni_covars_status": "missing_meta_subject_id",
            "adni_covars_rows": len(covars),
            "adni_covars_matched_rows": 0,
        }

    if covars_subject_col is None:
        print("[WARN] ADNI covars has no subject_id/PTID/RID-like column")
        return meta.copy(), {
            "adni_covars_status": "missing_covars_subject_id",
            "adni_covars_rows": len(covars),
            "adni_covars_matched_rows": 0,
        }

    group_col = first_col(covars, ["GROUP", "Research Group", "DX_bl", "DX", "diagnosis"])
    if group_col is None:
        print("[WARN] ADNI covars has no GROUP/diagnosis-like column")
        return meta.copy(), {
            "adni_covars_status": "missing_group",
            "adni_covars_rows": len(covars),
            "adni_covars_matched_rows": 0,
        }

    meta2 = meta.copy()
    covars2 = covars.copy()

    meta2["_merge_subject_id"] = meta2[meta_subject_col].map(clean_id)
    covars2["_merge_subject_id"] = covars2[covars_subject_col].map(clean_id)

    keep_cols = ["_merge_subject_id", group_col]

    for c in ["SITE", "AGE", "SEX"]:
        hit = first_col(covars2, [c])
        if hit is not None and hit not in keep_cols:
            keep_cols.append(hit)

    covars_keep = covars2[keep_cols].copy()
    rename_map = {group_col: "GROUP_from_covars"}

    for c in keep_cols:
        if c not in ["_merge_subject_id", group_col]:
            rename_map[c] = f"{c}_from_covars"

    covars_keep = covars_keep.rename(columns=rename_map)
    covars_keep = covars_keep.dropna(subset=["_merge_subject_id"])
    covars_keep = covars_keep.drop_duplicates("_merge_subject_id", keep="first")

    merged = meta2.merge(covars_keep, on="_merge_subject_id", how="left", validate="m:1")

    existing_group_col = first_col(merged, ["GROUP"])

    if existing_group_col is not None and existing_group_col != "GROUP_from_covars":
        merged["GROUP_original"] = merged[existing_group_col]
        merged["GROUP"] = merged[existing_group_col].where(
            merged[existing_group_col].notna(),
            merged["GROUP_from_covars"],
        )
    else:
        merged["GROUP"] = merged["GROUP_from_covars"]

    matched = int(merged["GROUP_from_covars"].notna().sum())

    merged = merged.drop(columns=["_merge_subject_id"])

    return merged, {
        "adni_covars_status": "ok",
        "adni_covars_rows": len(covars),
        "adni_covars_matched_rows": matched,
    }


def filter_adni_y0_y4(meta):
    if not FILTER_ADNI_TO_Y0_Y4:
        return meta, {
            "adni_y0_y4_filter": "not_applied",
            "adni_y0_y4_subjects": 0,
            "adni_rows_after_y0_y4_filter": len(meta),
        }

    if not ADNI_Y0_Y4_PATH.exists():
        print(f"[WARN] ADNI Y0/Y4 file not found: {ADNI_Y0_Y4_PATH}")
        return meta, {
            "adni_y0_y4_filter": "file_missing",
            "adni_y0_y4_subjects": 0,
            "adni_rows_after_y0_y4_filter": len(meta),
        }

    y0y4 = read_table(ADNI_Y0_Y4_PATH)

    subj_col = first_col(y0y4, ["DWI_subject", "DWI_subject_key", "subject_id", "RID"])
    flag_col = first_col(y0y4, ["has_y0_and_y4"])

    meta_subj_col = first_col(meta, ["DWI_subject_key", "DWI_subject", "subject_id", "RID", "PTID"])

    if subj_col is None or flag_col is None or meta_subj_col is None:
        print("[WARN] Could not apply ADNI Y0/Y4 filter due to missing columns")
        return meta, {
            "adni_y0_y4_filter": "missing_columns",
            "adni_y0_y4_subjects": 0,
            "adni_rows_after_y0_y4_filter": len(meta),
        }

    keep_subjects = set(
        y0y4.loc[y0y4[flag_col].astype(str).str.lower().isin(["true", "1", "yes"]), subj_col]
        .astype(str)
        .str.strip()
    )

    out = meta.loc[meta[meta_subj_col].astype(str).str.strip().isin(keep_subjects)].copy()

    return out, {
        "adni_y0_y4_filter": "applied",
        "adni_y0_y4_subjects": len(keep_subjects),
        "adni_rows_after_y0_y4_filter": len(out),
    }


# =============================================================================
# COHORT STATUS RULES
# =============================================================================

def harmonize_ad_decode(df):
    df = df.copy()
    normcog, mci, ad, label = init_status(df)

    risk_col = first_col(df, ["risk_for_ad", "Risk_y", "risk", "Risk_num"])

    if risk_col is None:
        source = "missing risk_for_ad"
        return finalize(df, normcog, mci, ad, label, source), source

    x = pd.to_numeric(df[risk_col], errors="coerce")

    is_normal = x.isin([0, 1])
    is_mci = x.eq(2)
    is_ad = x.eq(3)

    normcog.loc[is_normal] = 1
    normcog.loc[is_mci | is_ad] = 0

    mci.loc[is_mci] = 1
    mci.loc[is_normal | is_ad] = 0

    ad.loc[is_ad] = 1
    ad.loc[is_normal | is_mci] = 0

    source = f"{risk_col}: 0/1 Normal, 2 MCI, 3 AD"
    return finalize(df, normcog, mci, ad, label, source), source


def harmonize_adni(df):
    df, covars_info = merge_adni_covars(df)
    df, y0y4_info = filter_adni_y0_y4(df)

    normcog, mci, ad, label = init_status(df)

    group_col = first_col(df, ["GROUP", "Research Group", "DX_bl", "DX", "diagnosis"])

    if group_col is not None:
        s = df[group_col].astype(str).str.strip().str.upper()

        is_normal = s.isin(["CN", "NC", "NL", "NORMAL", "CONTROL", "CONTROLS"])

        if COUNT_ADNI_SMC_AS_NORMAL:
            is_normal = is_normal | s.eq("SMC")

        is_mci = s.str.contains(r"\bMCI\b|EMCI|LMCI", regex=True, na=False)
        is_ad = s.str.contains(r"\bAD\b|DEMENT|ALZHEIMER", regex=True, na=False)

        normcog.loc[is_normal] = 1
        normcog.loc[is_mci | is_ad] = 0

        mci.loc[is_mci] = 1
        mci.loc[is_normal | is_ad] = 0

        ad.loc[is_ad] = 1
        ad.loc[is_normal | is_mci] = 0

        source = f"ADNI covars merge + {group_col}"
        out = finalize(df, normcog, mci, ad, label, source)
        out.attrs["extra_summary"] = {**covars_info, **y0y4_info}
        return out, source

    dx_col = first_col(df, ["dxsum_DIAGNOSIS", "DXSUM_DIAGNOSIS", "DIAGNOSIS"])

    if dx_col is not None:
        x = pd.to_numeric(df[dx_col], errors="coerce")

        is_normal = x.eq(1)
        is_mci = x.eq(2)
        is_ad = x.eq(3)

        normcog.loc[is_normal] = 1
        normcog.loc[is_mci | is_ad] = 0

        mci.loc[is_mci] = 1
        mci.loc[is_normal | is_ad] = 0

        ad.loc[is_ad] = 1
        ad.loc[is_normal | is_mci] = 0

        source = f"ADNI fallback {dx_col}"
        out = finalize(df, normcog, mci, ad, label, source)
        out.attrs["extra_summary"] = {**covars_info, **y0y4_info}
        return out, source

    source = "missing ADNI GROUP/DXSUM"
    out = finalize(df, normcog, mci, ad, label, source)
    out.attrs["extra_summary"] = {**covars_info, **y0y4_info}
    return out, source


def harmonize_adrc(df):
    df = df.copy()
    normcog, mci, ad, label = init_status(df)
    sources = []

    norm_col = first_col(df, ["NORMCOG_01", "NORMCOG", "normcog"])
    mci_col = first_col(df, ["MCI_01", "IMPNOMCI", "impnomci", "MCI"])
    ad_col = first_col(df, ["AD_01", "DEMENTIA_01", "DEMENTED", "demented", "AD"])

    if norm_col is not None:
        x = pd.to_numeric(df[norm_col], errors="coerce")
        normcog.loc[x.eq(1)] = 1
        normcog.loc[x.eq(0)] = 0
        sources.append(norm_col)

    if mci_col is not None:
        x = pd.to_numeric(df[mci_col], errors="coerce")
        mci.loc[x.eq(1)] = 1
        mci.loc[x.eq(0)] = 0
        sources.append(mci_col)

    if ad_col is not None:
        x = pd.to_numeric(df[ad_col], errors="coerce")
        ad.loc[x.eq(1)] = 1
        ad.loc[x.eq(0)] = 0
        sources.append(ad_col)

    source = ";".join(sources) if sources else "missing ADRC NORMCOG/MCI/AD columns"
    return finalize(df, normcog, mci, ad, label, source), source


def harmonize_habs(df):
    df = df.copy()
    normcog, mci, ad, label = init_status(df)

    norm_col = first_col(df, ["NORMCOG_01", "NORMCOG", "normcog"])
    mci_col = first_col(df, ["MCI_01", "MCI", "mci"])
    ad_col = first_col(df, ["AD_01", "DEMENTIA_01", "DEMENTED", "demented", "AD"])

    if norm_col is not None or mci_col is not None or ad_col is not None:
        sources = []

        if norm_col is not None:
            x = pd.to_numeric(df[norm_col], errors="coerce")
            normcog.loc[x.eq(1)] = 1
            normcog.loc[x.eq(0)] = 0
            sources.append(norm_col)

        if mci_col is not None:
            x = pd.to_numeric(df[mci_col], errors="coerce")
            mci.loc[x.eq(1)] = 1
            mci.loc[x.eq(0)] = 0
            sources.append(mci_col)

        if ad_col is not None:
            x = pd.to_numeric(df[ad_col], errors="coerce")
            ad.loc[x.eq(1)] = 1
            ad.loc[x.eq(0)] = 0
            sources.append(ad_col)

        source = ";".join(sources)
        return finalize(df, normcog, mci, ad, label, source), source

    cdx_col = first_col(df, ["CDX_Cog", "CDX_COG", "cdx_cog"])

    if cdx_col is not None:
        x = pd.to_numeric(df[cdx_col], errors="coerce")

        is_normal = x.eq(0)
        is_mci = x.eq(1)
        is_ad = x.eq(2)

        normcog.loc[is_normal] = 1
        normcog.loc[is_mci | is_ad] = 0

        mci.loc[is_mci] = 1
        mci.loc[is_normal | is_ad] = 0

        ad.loc[is_ad] = 1
        ad.loc[is_normal | is_mci] = 0

        source = cdx_col
        return finalize(df, normcog, mci, ad, label, source), source

    source = "missing HABS CDX_Cog/NORMCOG/MCI/AD"
    return finalize(df, normcog, mci, ad, label, source), source


def harmonize(df, cohort):
    if cohort == "ADNI":
        return harmonize_adni(df)
    if cohort == "ADRC":
        return harmonize_adrc(df)
    if cohort == "HABS":
        return harmonize_habs(df)
    if cohort == "AD_DECODE":
        return harmonize_ad_decode(df)

    raise ValueError(f"Unknown cohort: {cohort}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 100)
    print("CREATING FINAL HARMONIZED METADATA FILES")
    print("=" * 100)
    print("Base:", BASE)
    print("Output:", OUTDIR)
    print("ADNI SMC counted as normal:", COUNT_ADNI_SMC_AS_NORMAL)
    print("Filter ADNI to Y0/Y4:", FILTER_ADNI_TO_Y0_Y4)

    summary_rows = []

    for cohort, path in METADATA_FILES.items():
        path = Path(path)

        if not path.exists():
            print(f"[MISSING] {cohort}: {path}")
            summary_rows.append({
                "cohort": cohort,
                "input_file": str(path),
                "output_file": "",
                "status": "missing_input",
                "source": "",
                "n_rows": 0,
                "n_normcog": 0,
                "n_mci": 0,
                "n_ad": 0,
                "n_unknown": 0,
            })
            continue

        df = read_table(path)
        out, source = harmonize(df, cohort)

        outpath = OUTDIR / f"{cohort}_harmonized_metadata.csv"
        out.to_csv(outpath, index=False)

        n_norm = int(out["NORMCOG_01"].eq(1).sum())
        n_mci = int(out["MCI_01"].eq(1).sum())
        n_ad = int(out["AD_01"].eq(1).sum())
        n_unknown = int(out["DX_Label_harmonized"].eq("Unknown").sum())

        summary_row = {
            "cohort": cohort,
            "input_file": str(path),
            "output_file": str(outpath),
            "status": "ok",
            "source": source,
            "n_rows": len(out),
            "n_normcog": n_norm,
            "n_mci": n_mci,
            "n_ad": n_ad,
            "n_unknown": n_unknown,
        }

        summary_row.update(out.attrs.get("extra_summary", {}))
        summary_rows.append(summary_row)

        print(
            f"[DONE] {cohort:<9} "
            f"N={len(out):<6} "
            f"Normal={n_norm:<5} "
            f"MCI={n_mci:<5} "
            f"AD={n_ad:<5} "
            f"Unknown={n_unknown:<5} "
            f"source={source}"
        )

    summary = pd.DataFrame(summary_rows)
    summary_path = OUTDIR / "harmonized_metadata_summary.csv"
    summary.to_csv(summary_path, index=False)

    print("\n[DONE] Summary:")
    print(summary_path)


if __name__ == "__main__":
    main()