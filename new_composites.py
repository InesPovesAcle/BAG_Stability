from __future__ import annotations#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 22 12:35:40 2026

@author: ines
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build harmonized five-domain cognitive composite scores across ADNI, ADRC, HABS,
and AD_DECODE.

Outputs are written to: $WORK/ines/results/composite_cognition by default.

Domains:
  1. Memory_Composite
  2. Executive_Function_Composite
  3. Processing_Speed_Composite
  4. Language_Composite
  5. Visuospatial_Composite

Residualized versions:
  *_resid

Design choices:
  - Each source test is oriented so higher values mean better cognition.
  - Each source test is z-scored within cohort before averaging into a domain.
  - Domain composites are the row-wise mean of available oriented z-scored component tests.
  - AD_DECODE composites are recalculated from their underlying z-scored neuropsychological component columns.
  - Residualized composites are recomputed within each cohort by regressing the domain score on available age, sex,
    education, and site variables. Residuals are then z-scored within cohort.
  - A mapping CSV and methods markdown are saved for documentation.
"""



import argparse
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

DOMAINS = [
    "Memory_Composite",
    "Executive_Function_Composite",
    "Processing_Speed_Composite",
    "Language_Composite",
    "Visuospatial_Composite",
]
GLOBAL = "Global_Cognition_Composite"

# Each component is (column_name, direction, rationale).
# direction = +1 means higher is better; -1 means higher is worse and is multiplied by -1 before z-scoring.
DOMAIN_COMPONENTS: Dict[str, Dict[str, List[Tuple[str, int, str]]]] = {
    "AD_DECODE": {
        "Memory_Composite": [
            ("AVLT_Trial6_z", +1, "AVLT Trial 6 z-score; delayed verbal list recall"),
            ("AVLT_Trial7_z", +1, "AVLT Trial 7 z-score; recognition/delayed verbal memory"),
            ("RAVLT_IMMEDIATE_z", +1, "RAVLT immediate recall z-score; verbal learning/immediate memory"),
            ("RAVLT_LEARNING_z", +1, "RAVLT learning z-score; verbal learning across trials"),
            ("RAVLT_FORGETTING_rev_z", +1, "Reversed RAVLT forgetting z-score; higher means less forgetting"),
            ("Story_Immediate_verbatim_z", +1, "Immediate story recall, verbatim z-score"),
            ("Story_Immediate_paraphrase_z", +1, "Immediate story recall, paraphrase z-score"),
            ("Delayed_verbatim_z", +1, "Delayed story recall, verbatim z-score"),
            ("Delayed_paraphrase_z", +1, "Delayed story recall, paraphrase z-score"),
        ],
        "Executive_Function_Composite": [
            ("trailB_rev_z", +1, "Reversed Trail Making Test B z-score; set shifting/executive function"),
            ("trailDiff_rev_z", +1, "Reversed Trails B-A difference z-score; executive set shifting adjusted for speed"),
            ("bckwds_total_correct_z", +1, "Backward digit span total correct z-score; working memory/executive control"),
            ("bckwds_max_length_z", +1, "Backward digit span maximum length z-score; working memory span"),
        ],
        "Processing_Speed_Composite": [
            ("trailA_rev_z", +1, "Reversed Trail Making Test A z-score; visual scanning/processing speed"),
            ("DigitSymbol_z", +1, "Digit Symbol z-score; processing speed and attention"),
            ("fwd_total_correct_z", +1, "Forward digit span total correct z-score; attention span"),
            ("fwd_max_length_z", +1, "Forward digit span maximum length z-score; attention span"),
            ("ufov1_rev_z", +1, "Reversed UFOV subtest 1 z-score; visual processing speed"),
            ("ufov2_rev_z", +1, "Reversed UFOV subtest 2 z-score; divided/selective attention speed"),
            ("ufov3_rev_z", +1, "Reversed UFOV subtest 3 z-score; divided/selective attention speed"),
        ],
        "Language_Composite": [
            ("fluency_4x_z", +1, "Category/semantic fluency composite z-score"),
            ("letter_fluency_z", +1, "Letter/phonemic fluency z-score"),
        ],
        "Visuospatial_Composite": [
            ("Im_BensonTotal_z", +1, "Benson figure immediate/copy total z-score; visuospatial construction"),
            ("Delay_BensonTotal_z", +1, "Benson figure delayed total z-score; visuospatial memory/construction"),
        ],
    },
    "ADNI": {
        "Memory_Composite": [
            ("MOCA_delayed_recall_z", +1, "MoCA delayed recall z-score; memory proxy"),
        ],
        "Executive_Function_Composite": [
            ("MOCA_attention_z", +1, "MoCA attention z-score; attention/executive proxy"),
            ("MOCA_abstraction_z", +1, "MoCA abstraction z-score; executive proxy"),
        ],
        "Processing_Speed_Composite": [],
        "Language_Composite": [
            ("MOCA_language_z", +1, "MoCA language z-score"),
            ("MOCA_naming_z", +1, "MoCA naming z-score"),
        ],
        "Visuospatial_Composite": [
            ("MOCA_visuospatial_z", +1, "MoCA visuospatial/executive z-score"),
        ],
    },
    "ADRC": {
        "Memory_Composite": [
            ("UDSBENTD", +1, "Benson delayed recall; memory"),
        ],
        "Executive_Function_Composite": [
            ("TRAILB", -1, "Trail Making Test B time; higher/slower is worse"),
        ],
        "Processing_Speed_Composite": [
            ("TRAILA", -1, "Trail Making Test A time; higher/slower is worse"),
        ],
        "Language_Composite": [
            ("ANIMALS", +1, "Animal/category fluency"),
            ("UDSVERFC", +1, "UDS verbal fluency/language component"),
            ("UDSVERLC", +1, "UDS verbal fluency/language component"),
            ("UDSVERTN", +1, "UDS verbal fluency/language total"),
        ],
        "Visuospatial_Composite": [
            ("UDSBENTC", +1, "Benson figure copy; visuospatial construction"),
        ],
    },
    "HABS": {
        "Memory_Composite": [
            ("SEVLT_T1235_ZScore", +1, "SEVLT total learning z-score"),
            ("SEVLT_DR_ZScore", +1, "SEVLT delayed recall z-score"),
            ("LM1_AB_ZScore", +1, "Logical Memory immediate story z-score"),
            ("LM2_AB_ZScore", +1, "Logical Memory delayed story z-score"),
        ],
        "Executive_Function_Composite": [
            ("Trails_B_ZScore", -1, "Trails B time z-score; inverted so higher is better"),
            ("DS_ZScore", +1, "Digit span total z-score; working memory/attention-executive"),
        ],
        "Processing_Speed_Composite": [
            ("Trails_A_ZScore", -1, "Trails A time z-score; inverted so higher is better"),
            ("Digit_Symbol_Substitution_ZScore", +1, "Digit symbol substitution z-score"),
        ],
        "Language_Composite": [
            ("FAS_ZScore", +1, "Letter fluency z-score"),
            ("Animal_ZScore", +1, "Animal/category fluency z-score"),
            ("WAT_Correct", +1, "Word accentuation/reading vocabulary score; language proxy"),
        ],
        "Visuospatial_Composite": [],
    },
}

ID_CANDIDATES: Dict[str, List[str]] = {
    "AD_DECODE": ["ID", "match_id", "subject_id", "IDRNA", "ID_PCA"],
    "ADNI": ["subject_id", "PTID", "RID", "VISCODE", "EXAMDATE"],
    "ADRC": ["PTID", "subject_id", "Subject", "match_id"],
    "HABS": ["Subject", "runno", "Visit_ID", "Med_ID"],
}
AGE_CANDIDATES: Dict[str, List[str]] = {
    "AD_DECODE": ["age", "Age", "AGE"],
    "ADNI": ["AGE", "Age", "age"],
    "ADRC": ["VISIT_AGE", "SUBJECT_AGE_SCREEN", "DECAGE", "Age", "age"],
    "HABS": ["Age", "age", "AGE"],
}
SEX_CANDIDATES: Dict[str, List[str]] = {
    "AD_DECODE": ["sex", "sex_numeric", "Sex", "SEX"],
    "ADNI": ["SEX", "PTGENDER", "Sex", "sex"],
    "ADRC": ["SUBJECT_SEX", "Sex", "sex", "SEX"],
    "HABS": ["Sex", "sex", "SEX", "ID_Gender"],
}
EDU_CANDIDATES: Dict[str, List[str]] = {
    "AD_DECODE": ["education", "Education", "educ", "EDUC", "years_education"],
    "ADNI": ["PTEDUCAT", "Education", "education", "educ"],
    "ADRC": ["EDUC", "Education", "education", "years_education"],
    "HABS": ["ID_Education", "ID_Education_Degree", "Education", "education"],
}
SITE_CANDIDATES: Dict[str, List[str]] = {
    "AD_DECODE": ["site", "Site", "scanner", "Scanner"],
    "ADNI": ["SITE", "Site", "site"],
    "ADRC": ["SITE", "Site", "site"],
    "HABS": ["Interview_Site", "site", "Site"],
}


def parse_args() -> argparse.Namespace:
    default_work = os.environ.get("WORK", "/mnt/newStor/paros/paros_WORK")
    default_base = Path(default_work) / "ines"
    parser = argparse.ArgumentParser(description="Build harmonized cognitive domain composites across cohorts.")
    parser.add_argument("--base-dir", default=str(default_base), help="Base project dir, default $WORK/ines.")
    parser.add_argument("--outdir", default=None, help="Output dir, default <base-dir>/results/composite_cognition.")
    parser.add_argument("--min-components", type=int, default=1, help="Minimum non-missing component tests required per domain.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs.")
    parser.add_argument(
        "--input-override-dir",
        default=None,
        help="Optional directory containing ADNI_metadata.xlsx, ADRC_metadata.xlsx, HABS_metadata.xlsx, AD_DECODE_metadata.xlsx for testing.",
    )
    return parser.parse_args()


def input_paths(base_dir: Path, override_dir: Optional[str] = None) -> Dict[str, Path]:
    if override_dir:
        d = Path(override_dir)
        return {
            "AD_DECODE": d / "AD_DECODE_metadata.xlsx",
            "ADRC": d / "ADRC_metadata.xlsx",
            "ADNI": d / "ADNI_metadata.xlsx",
            "HABS": d / "HABS_metadata.xlsx",
        }
    return {
        "AD_DECODE": base_dir / "data" / "harmonization" / "AD_DECODE" / "metadata" / "AD_DECODE_metadata.xlsx",
        "ADRC": base_dir / "data" / "harmonization" / "ADRC" / "metadata" / "ADRC_metadata.xlsx",
        "ADNI": base_dir / "data" / "harmonization" / "ADNI" / "metadata" / "ADNI_metadata.xlsx",
        "HABS": base_dir / "data" / "harmonization" / "HABS" / "metadata" / "HABS_metadata.xlsx",
    }


def read_first_sheet(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing metadata file: {path}")
    return pd.read_excel(path)


def safe_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def clean_numeric_series(cohort: str, col: str, s: pd.Series) -> pd.Series:
    """Convert to numeric and remove common non-response / special missing codes."""
    x = safe_numeric(s)
    cname = str(col).upper()

    x = x.mask(x.isin([-9999, -999, -99, 9999, 999, 998, 997, 996, 995]))

    if cohort == "ADRC":
        if cname == "MOCATOTS":
            x = x.mask((x < 0) | (x > 30))
        elif cname in {"UDSBENTC", "UDSBENTD"}:
            x = x.mask((x < 0) | (x > 30))
        elif cname in {"ANIMALS", "UDSVERFC", "UDSVERLC", "UDSVERTN"}:
            x = x.mask((x < 0) | (x >= 88))
        elif cname in {"TRAILA", "TRAILB"}:
            x = x.mask((x <= 0) | (x >= 995))
        else:
            x = x.mask(x.isin([88, 95, 96, 97, 98, 99]))

    if cohort == "ADNI":
        if cname.startswith("MOCA_") and not cname.endswith("_Z"):
            x = x.mask(x < 0)
        if cname == "MMSE_TOTAL":
            x = x.mask((x < 0) | (x > 30))
        if cname == "MOCA_TOTAL_CORRECTED":
            x = x.mask((x < 0) | (x > 30))

    return x


def zscore(s: pd.Series) -> pd.Series:
    x = safe_numeric(s).astype(float)
    mu = x.mean(skipna=True)
    sd = x.std(skipna=True, ddof=0)
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(np.nan, index=s.index, dtype=float)
    return (x - mu) / sd


def first_existing(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    lower_map = {str(c).lower(): c for c in df.columns}
    for c in candidates:
        if str(c).lower() in lower_map:
            return lower_map[str(c).lower()]
    return None


def normalize_sex(s: pd.Series) -> pd.Series:
    raw = s.astype(str).str.strip().str.upper()
    mapping = {
        "M": "M", "MALE": "M", "MAN": "M", "1": "M", "1.0": "M",
        "F": "F", "FEMALE": "F", "WOMAN": "F", "2": "F", "2.0": "F",
        "0": np.nan, "0.0": np.nan,
        "NAN": np.nan, "NONE": np.nan, "": np.nan, "NA": np.nan, "N/A": np.nan,
        "UNKNOWN": np.nan, "UNK": np.nan,
    }
    out = raw.replace(mapping)
    return out.where(out.isin(["F", "M"]), np.nan)


def build_design_matrix(df: pd.DataFrame, cohort: str) -> Tuple[pd.DataFrame, List[str]]:
    """Build residualization covariate matrix from available age, sex, education, and site."""
    cov = pd.DataFrame(index=df.index)
    used: List[str] = []

    age_col = first_existing(df, AGE_CANDIDATES.get(cohort, []))
    if age_col is not None:
        age = safe_numeric(df[age_col])
        if age.notna().sum() >= 20 and age.nunique(dropna=True) >= 2:
            cov["age"] = age
            used.append(age_col)

    sex_col = first_existing(df, SEX_CANDIDATES.get(cohort, []))
    if sex_col is not None:
        sex = normalize_sex(df[sex_col])
        if sex.notna().sum() >= 20 and sex.nunique(dropna=True) >= 2:
            d = pd.get_dummies(sex, prefix="sex", drop_first=True, dtype=float)
            cov = pd.concat([cov, d], axis=1)
            used.append(sex_col)

    edu_col = first_existing(df, EDU_CANDIDATES.get(cohort, []))
    if edu_col is not None:
        edu = safe_numeric(df[edu_col])
        if edu.notna().sum() >= 20 and edu.nunique(dropna=True) >= 2:
            cov["education"] = edu
            used.append(edu_col)

    site_col = first_existing(df, SITE_CANDIDATES.get(cohort, []))
    if site_col is not None:
        site = df[site_col].astype(str).str.strip().replace({"nan": np.nan, "": np.nan})
        if site.notna().sum() >= 20 and 1 < site.nunique(dropna=True) <= 20:
            d = pd.get_dummies(site, prefix="site", drop_first=True, dtype=float)
            cov = pd.concat([cov, d], axis=1)
            used.append(site_col)

    return cov, used


def residualize(y: pd.Series, cov: pd.DataFrame, min_n: int = 30) -> pd.Series:
    """Residualize y on covariates using OLS; z-score residuals. If covariates insufficient, return z-score(y)."""
    y = safe_numeric(y)
    tmp = pd.concat([y.rename("y"), cov], axis=1).dropna()
    if tmp["y"].notna().sum() < min_n or cov.shape[1] == 0:
        return zscore(y)

    Xcov = tmp.drop(columns=["y"])
    keep_cols = [c for c in Xcov.columns if Xcov[c].nunique(dropna=True) >= 2]
    if not keep_cols:
        return zscore(y)
    Xcov = Xcov[keep_cols].astype(float)

    X = np.column_stack([np.ones(len(Xcov)), Xcov.to_numpy(dtype=float)])
    yy = tmp["y"].to_numpy(dtype=float)

    try:
        beta, *_ = np.linalg.lstsq(X, yy, rcond=None)
        pred = X @ beta
        resid = yy - pred
    except Exception:
        return zscore(y)

    out = pd.Series(np.nan, index=y.index, dtype=float)
    out.loc[tmp.index] = resid
    return zscore(out)


def component_series(df: pd.DataFrame, cohort: str, col: str, direction: int) -> Optional[pd.Series]:
    if col not in df.columns:
        lower_map = {str(c).lower(): c for c in df.columns}
        if col.lower() not in lower_map:
            return None
        col = lower_map[col.lower()]
    x = clean_numeric_series(cohort, col, df[col])
    x = x * float(direction)
    return zscore(x)


def build_composites_for_cohort(
    df: pd.DataFrame,
    cohort: str,
    min_components: int = 1,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    """Return output dataframe, long mapping, and audit dict for one cohort."""
    out = pd.DataFrame(index=df.index)
    out["cohort"] = cohort

    for c in ID_CANDIDATES.get(cohort, []):
        if c in df.columns and c not in out.columns:
            out[c] = df[c]
    for c in AGE_CANDIDATES.get(cohort, []) + SEX_CANDIDATES.get(cohort, []) + EDU_CANDIDATES.get(cohort, []):
        if c in df.columns and c not in out.columns:
            out[c] = df[c]

    mapping_rows = []

    for domain in DOMAINS:
        comp_list = DOMAIN_COMPONENTS.get(cohort, {}).get(domain, [])
        comp_z_cols = []
        for col, direction, rationale in comp_list:
            z = component_series(df, cohort, col, direction)
            found = z is not None
            n_nonmissing = int(z.notna().sum()) if z is not None else 0
            z_col_name = f"__z_{domain}_{re.sub(r'[^A-Za-z0-9]+', '_', col).strip('_')}"
            if z is not None:
                out[z_col_name] = z
                comp_z_cols.append(z_col_name)
            mapping_rows.append({
                "cohort": cohort,
                "domain": domain,
                "component_column": col,
                "found": found,
                "direction": direction,
                "higher_after_orientation": "better",
                "n_nonmissing_after_cleaning": n_nonmissing,
                "rationale": rationale,
            })

        if comp_z_cols:
            n_available = out[comp_z_cols].notna().sum(axis=1)
            raw = out[comp_z_cols].mean(axis=1, skipna=True)
            raw = raw.where(n_available >= min_components)
        else:
            n_available = pd.Series(0, index=out.index)
            raw = pd.Series(np.nan, index=out.index, dtype=float)
        out[domain] = zscore(raw)
        out[f"{domain}_n_components"] = n_available

    domain_mat = out[DOMAINS]
    n_domains = domain_mat.notna().sum(axis=1)
    out[GLOBAL] = zscore(domain_mat.mean(axis=1, skipna=True).where(n_domains >= 1))
    out[f"{GLOBAL}_n_domains"] = n_domains

    cov, used_covariates = build_design_matrix(df, cohort)
    for domain in DOMAINS + [GLOBAL]:
        out[f"{domain}_resid"] = residualize(out[domain], cov)

    hidden_cols = [c for c in out.columns if c.startswith("__z_")]
    out = out.drop(columns=hidden_cols)

    audit = {
        "cohort": cohort,
        "n_rows": len(df),
        "used_covariates_for_residuals": ", ".join(used_covariates),
        "n_covariate_columns_after_encoding": cov.shape[1],
    }
    for domain in DOMAINS:
        audit[f"n_{domain}"] = int(out[domain].notna().sum())
        audit[f"components_{domain}"] = "; ".join([
            f"{col}({'+' if direction > 0 else '-'})"
            for col, direction, _ in DOMAIN_COMPONENTS.get(cohort, {}).get(domain, [])
        ])
    audit[f"n_{GLOBAL}"] = int(out[GLOBAL].notna().sum())

    return out, pd.DataFrame(mapping_rows), audit


def methods_text(audit: pd.DataFrame, mapping: pd.DataFrame) -> str:
    lines = []
    lines.append("# Harmonized cognitive composite construction")
    lines.append("")
    lines.append("## Overview")
    lines.append(
        "Five cognitive domain composites were generated for each cohort where suitable measures were available: "
        "Memory, Executive Function, Processing Speed, Language, and Visuospatial cognition. "
        "For each cohort, source measures were oriented so that higher values indicate better cognition, z-scored within cohort, "
        "and averaged within each domain. The global cognition composite was computed as the mean of available domain composites."
    )
    lines.append("")
    lines.append("## Residualization")
    lines.append(
        "Residualized versions of each domain and the global composite were computed by regressing the composite on available "
        "age, sex, education, and site covariates within each cohort. Categorical covariates were dummy-coded. Residuals were then "
        "z-scored within cohort. If fewer than 30 complete observations or no usable covariates were available, the residualized "
        "score falls back to the cohort z-scored composite."
    )
    lines.append("")
    lines.append("## Cohort-specific domain definitions")
    for cohort in ["AD_DECODE", "ADNI", "ADRC", "HABS"]:
        lines.append(f"\n### {cohort}")
        sub = mapping[mapping["cohort"] == cohort]
        for domain in DOMAINS:
            rows = sub[sub["domain"] == domain]
            found_rows = rows[rows["found"] == True]
            if found_rows.empty:
                lines.append(f"- **{domain}**: not directly available in supplied metadata; output is missing.")
            else:
                comps = []
                for _, r in found_rows.iterrows():
                    orient = "higher=better" if int(r["direction"]) > 0 else "inverted; higher=better after orientation"
                    comps.append(f"`{r['component_column']}` ({orient})")
                lines.append(f"- **{domain}**: " + ", ".join(comps) + ".")
    lines.append("")
    lines.append("## Important limitations")
    lines.append(
        "Some cohorts do not contain a direct measure for every domain in the supplied metadata. In particular, the provided ADNI "
        "metadata lacks a clear processing-speed test, and the provided HABS metadata lacks a direct visuospatial construction/copy "
        "test comparable to AD_DECODE or ADRC. These fields are retained as missing rather than filled with weak proxies."
    )
    lines.append("")
    lines.append("## Output columns")
    lines.append("- `Memory_Composite`, `Executive_Function_Composite`, `Processing_Speed_Composite`, `Language_Composite`, `Visuospatial_Composite`")
    lines.append("- Residualized versions with `_resid` suffix")
    lines.append("- `Global_Cognition_Composite` and `Global_Cognition_Composite_resid`")
    lines.append("- `*_n_components` and `Global_Cognition_Composite_n_domains` audit columns")
    lines.append("")
    lines.append("## Audit summary")
    lines.append(audit.to_markdown(index=False))
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    base_dir = Path(args.base_dir)
    outdir = Path(args.outdir) if args.outdir else base_dir / "results" / "composite_cognition"
    outdir.mkdir(parents=True, exist_ok=True)

    paths = input_paths(base_dir, args.input_override_dir)

    all_outputs = []
    all_mapping = []
    all_audit = []

    for cohort, path in paths.items():
        print(f"\n[INFO] Loading {cohort}: {path}")
        df = read_first_sheet(path)
        print(f"[INFO] {cohort} shape: {df.shape}")

        out, mapping, audit = build_composites_for_cohort(df, cohort, min_components=args.min_components)
        all_outputs.append(out)
        all_mapping.append(mapping)
        all_audit.append(audit)

        cohort_path = outdir / f"{cohort}_cognitive_composites.csv"
        out.to_csv(cohort_path, index=False)
        print(f"[INFO] Saved {cohort_path} rows={len(out)}")

    combined = pd.concat(all_outputs, ignore_index=True, sort=False)
    mapping_df = pd.concat(all_mapping, ignore_index=True, sort=False)
    audit_df = pd.DataFrame(all_audit)

    combined_path = outdir / "harmonized_cognitive_composites_all_cohorts.csv"
    mapping_path = outdir / "harmonized_cognitive_composite_mapping.csv"
    audit_path = outdir / "harmonized_cognitive_composite_audit.csv"
    methods_path = outdir / "harmonized_cognitive_composite_methods.md"

    combined.to_csv(combined_path, index=False)
    mapping_df.to_csv(mapping_path, index=False)
    audit_df.to_csv(audit_path, index=False)
    methods_path.write_text(methods_text(audit_df, mapping_df), encoding="utf-8")

    print("\n[DONE] Outputs saved:")
    print(" ", combined_path)
    print(" ", mapping_path)
    print(" ", audit_path)
    print(" ", methods_path)
    print("\nAudit summary:")
    print(audit_df.to_string(index=False))


if __name__ == "__main__":
    main()
