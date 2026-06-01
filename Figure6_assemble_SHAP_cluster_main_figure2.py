#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Assemble all SHAP subject-cluster multipanel figures.

Creates:
  1. Main figure: node-SHAP heatmap + AD + cognitive status + sensitivity table
  2. Supplementary figure: node-SHAP heatmap + APOE4 + sex + sensitivity table
  3. Overview figure: node-SHAP heatmap + AD + cognition + APOE4 + sex + compact table

Run:
python Figure6_assemble_SHAP_cluster_all_figures.py \
  --model imaging_only \
  --kind node \
  --main-k 4 \
  --k-values 3,4,5,6
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


DEFAULT_WORK = os.environ.get("WORK", "/mnt/newStor/paros/paros_WORK")
DEFAULT_OUT_BASE = (
    Path(DEFAULT_WORK)
    / "ines/results"
    / "BrainAgeValidation_AllCohorts_BAGBiasCorr_OOFGlobal_BiologicalValidation"
    / "Figure6_SHAP"
)

VARIABLE_LABELS = {
    "ad_status": "AD status",
    "cognitive_status": "Cognitive status",
    "apoe4_carriage": "APOE4 carriage",
    "sex": "Sex",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out-base", default=str(DEFAULT_OUT_BASE))
    p.add_argument("--model", default="imaging_only")
    p.add_argument("--kind", default="node", choices=["node", "global"])
    p.add_argument("--main-k", type=int, default=4)
    p.add_argument("--k-values", default="3,4,5,6")
    p.add_argument("--input-subdir", default="subject_shap_clusters_metadata")
    p.add_argument("--dpi", type=int, default=450)
    p.add_argument("--formats", default="png,pdf")
    return p.parse_args()


def add_suffix(prefix: Path, suffix: str) -> Path:
    return prefix.with_name(prefix.name + suffix)


def prefix_for(base_dir: Path, model: str, kind: str, k: int) -> Path:
    return base_dir / f"cross_cohort_{model}_{kind}_SHAP_subject_clusters_k{k}"


def require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return path


def image_path(base_dir: Path, model: str, kind: str, k: int, suffix: str) -> Path:
    return require_file(add_suffix(prefix_for(base_dir, model, kind, k), suffix))


def show_image(ax, path: Path, label: str, title: str):
    img = mpimg.imread(require_file(path))
    ax.imshow(img)
    ax.axis("off")
    ax.text(
        -0.02, 1.04, label,
        transform=ax.transAxes,
        fontsize=18,
        fontweight="bold",
        ha="left",
        va="bottom",
    )
    ax.set_title(title, fontsize=12, pad=8)


def fmt_p(x):
    if pd.isna(x):
        return "NA"
    x = float(x)
    if x < 1e-4:
        return f"{x:.1e}"
    return f"{x:.3g}"


def fmt_v(x):
    if pd.isna(x):
        return "NA"
    return f"{float(x):.3f}"


def load_stats_row(base_dir: Path, model: str, kind: str, k: int, variable: str) -> pd.Series:
    f = require_file(add_suffix(
        prefix_for(base_dir, model, kind, k),
        "_cluster_distribution_all_variables_stats.csv",
    ))
    df = pd.read_csv(f)
    row = df[df["variable"].astype(str).eq(variable)]
    if row.empty:
        raise ValueError(f"Variable '{variable}' not found in {f}")
    return row.iloc[0]


def cluster_sizes(base_dir: Path, model: str, kind: str, k: int) -> str:
    f = require_file(add_suffix(
        prefix_for(base_dir, model, kind, k),
        "_subject_clusters_with_metadata.csv",
    ))
    df = pd.read_csv(f)
    counts = df["shap_cluster"].value_counts().sort_index()
    return ", ".join(str(int(x)) for x in counts.values)


def build_sensitivity_table(
    ax,
    base_dir: Path,
    model: str,
    kind: str,
    k_values: list[int],
    variables: list[str],
    label: str,
    title: str,
    font_size: float = 8.0,
):
    rows = []
    for k in k_values:
        r = [str(k), cluster_sizes(base_dir, model, kind, k)]
        for var in variables:
            s = load_stats_row(base_dir, model, kind, k, var)
            r.extend([
                fmt_p(s.get("chi_square_p")),
                fmt_p(s.get("permutation_p")),
                fmt_v(s.get("cramers_v")),
            ])
        rows.append(r)

    cols = ["k", "Cluster sizes"]
    for var in variables:
        lab = VARIABLE_LABELS.get(var, var.replace("_", " "))
        cols.extend([f"{lab}\nχ² p", f"{lab}\nperm p", f"{lab}\nV"])

    ax.axis("off")
    ax.text(
        -0.02, 1.04, label,
        transform=ax.transAxes,
        fontsize=18,
        fontweight="bold",
        ha="left",
        va="bottom",
    )
    ax.set_title(title, fontsize=12, pad=8)

    table = ax.table(
        cellText=rows,
        colLabels=cols,
        cellLoc="center",
        colLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    table.scale(1.0, 1.55)

    for (rr, cc), cell in table.get_celld().items():
        if rr == 0:
            cell.set_text_props(weight="bold")
        if cc == 1:
            cell.set_width(0.28)
        elif cc == 0:
            cell.set_width(0.05)

    ax.text(
        0.0, -0.07,
        "χ² p = chi-square p-value; perm p = permutation p-value; V = Cramér’s V.",
        transform=ax.transAxes,
        fontsize=8,
        ha="left",
        va="top",
    )


def save_formats(fig, out_stem: Path, formats: list[str], dpi: int):
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    outputs = []
    for fmt in formats:
        out = out_stem.with_suffix(f".{fmt}")
        if fmt == "png":
            fig.savefig(out, dpi=dpi, bbox_inches="tight")
        else:
            fig.savefig(out, bbox_inches="tight")
        outputs.append(str(out))
    plt.close(fig)
    return outputs


def make_four_panel(
    base_dir: Path,
    final_dir: Path,
    model: str,
    kind: str,
    main_k: int,
    k_values: list[int],
    var_b: str,
    var_c: str,
    title_b: str,
    title_c: str,
    table_title: str,
    suptitle: str,
    caption: str,
    out_name: str,
    formats: list[str],
    dpi: int,
):
    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(
        2, 2,
        width_ratios=[1.35, 1.0],
        height_ratios=[1.15, 1.0],
        wspace=0.16,
        hspace=0.22,
    )

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    show_image(
        ax_a,
        image_path(base_dir, model, kind, main_k, "_hierarchical_heatmap.png"),
        "A",
        f"Cross-cohort {kind}-SHAP subject clustering, k={main_k}",
    )
    show_image(
        ax_b,
        image_path(base_dir, model, kind, main_k, f"_cluster_distribution_{var_b}.png"),
        "B",
        title_b,
    )
    show_image(
        ax_c,
        image_path(base_dir, model, kind, main_k, f"_cluster_distribution_{var_c}.png"),
        "C",
        title_c,
    )
    build_sensitivity_table(
        ax_d,
        base_dir,
        model,
        kind,
        k_values,
        [var_b, var_c],
        "D",
        table_title,
    )

    fig.suptitle(suptitle, fontsize=17, fontweight="bold", y=0.985)
    fig.text(0.02, 0.012, caption, fontsize=9, ha="left", va="bottom", wrap=True)

    return save_formats(fig, final_dir / out_name, formats, dpi)


def make_overview(
    base_dir: Path,
    final_dir: Path,
    model: str,
    kind: str,
    main_k: int,
    k_values: list[int],
    formats: list[str],
    dpi: int,
):
    fig = plt.figure(figsize=(22, 16))
    gs = fig.add_gridspec(
        2, 3,
        width_ratios=[1.45, 1.0, 1.0],
        height_ratios=[1.05, 1.0],
        wspace=0.14,
        hspace=0.22,
    )

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])
    ax_d = fig.add_subplot(gs[1, 0])
    ax_e = fig.add_subplot(gs[1, 1])
    ax_f = fig.add_subplot(gs[1, 2])

    show_image(
        ax_a,
        image_path(base_dir, model, kind, main_k, "_hierarchical_heatmap.png"),
        "A",
        f"Cross-cohort {kind}-SHAP subject clustering, k={main_k}",
    )
    show_image(
        ax_b,
        image_path(base_dir, model, kind, main_k, "_cluster_distribution_ad_status.png"),
        "B",
        "AD-status composition",
    )
    show_image(
        ax_c,
        image_path(base_dir, model, kind, main_k, "_cluster_distribution_cognitive_status.png"),
        "C",
        "Cognitive-status composition",
    )
    show_image(
        ax_d,
        image_path(base_dir, model, kind, main_k, "_cluster_distribution_apoe4_carriage.png"),
        "D",
        "APOE4-carriage composition",
    )
    show_image(
        ax_e,
        image_path(base_dir, model, kind, main_k, "_cluster_distribution_sex.png"),
        "E",
        "Sex composition",
    )
    build_sensitivity_table(
        ax_f,
        base_dir,
        model,
        kind,
        k_values,
        ["ad_status", "cognitive_status", "apoe4_carriage", "sex"],
        "F",
        "Sensitivity across k",
        font_size=6.1,
    )

    fig.suptitle(
        "Cross-cohort SHAP-defined subject clusters and post-hoc phenotype composition",
        fontsize=17,
        fontweight="bold",
        y=0.985,
    )
    fig.text(
        0.02,
        0.012,
        "Subjects were clustered using SHAP profiles only. Clinical/genetic variables were evaluated post hoc. "
        "AD and cognitive-status enrichment were the primary disease-related findings; APOE4 and sex are shown as secondary characterization.",
        fontsize=9,
        ha="left",
        va="bottom",
        wrap=True,
    )

    return save_formats(
        fig,
        final_dir / f"Supplementary_Figure6_SHAP_subject_clusters_{model}_{kind}_k{main_k}_AllPhenotypes",
        formats,
        dpi,
    )


def main():
    args = parse_args()

    out_base = Path(args.out_base).expanduser().resolve()
    base_dir = out_base / "cross_cohort_aggregated" / args.model / args.input_subdir
    final_dir = out_base / "cross_cohort_aggregated" / "final_figures"

    if not base_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {base_dir}")

    k_values = [int(x.strip()) for x in args.k_values.split(",") if x.strip()]
    formats = [x.strip().lower().lstrip(".") for x in args.formats.split(",") if x.strip()]

    outputs = []

    outputs.extend(make_four_panel(
        base_dir=base_dir,
        final_dir=final_dir,
        model=args.model,
        kind=args.kind,
        main_k=args.main_k,
        k_values=k_values,
        var_b="ad_status",
        var_c="cognitive_status",
        title_b="Post-hoc AD-status composition",
        title_c="Post-hoc cognitive-status composition",
        table_title="Sensitivity of disease/cognitive enrichment across k",
        suptitle="Node-feature SHAP-defined subject clusters are enriched for cognitive and AD status",
        caption=(
            "Subjects were clustered using cross-cohort node-feature SHAP profiles from the imaging-only model. "
            "Clinical/genetic variables were not used for clustering and were evaluated only post hoc. "
            "The k=4 solution is shown as the primary clustering result, with sensitivity across k=3–6."
        ),
        out_name=f"Figure6_SHAP_subject_clusters_{args.model}_{args.kind}_k{args.main_k}_AD_Cognition",
        formats=formats,
        dpi=args.dpi,
    ))

    outputs.extend(make_four_panel(
        base_dir=base_dir,
        final_dir=final_dir,
        model=args.model,
        kind=args.kind,
        main_k=args.main_k,
        k_values=k_values,
        var_b="apoe4_carriage",
        var_c="sex",
        title_b="Post-hoc APOE4-carriage composition",
        title_c="Post-hoc sex composition",
        table_title="Sensitivity of APOE4/sex enrichment across k",
        suptitle="Secondary post-hoc characterization of SHAP-defined subject clusters",
        caption=(
            "Subjects were clustered using node-feature SHAP profiles only. APOE4 carriage and sex were not used during clustering. "
            "These variables are shown as secondary characterization because their associations were weaker and less stable than AD/cognitive status."
        ),
        out_name=f"Supplementary_Figure6_SHAP_subject_clusters_{args.model}_{args.kind}_k{args.main_k}_APOE4_Sex",
        formats=formats,
        dpi=args.dpi,
    ))

    outputs.extend(make_overview(
        base_dir=base_dir,
        final_dir=final_dir,
        model=args.model,
        kind=args.kind,
        main_k=args.main_k,
        k_values=k_values,
        formats=formats,
        dpi=args.dpi,
    ))

    manifest = final_dir / f"Figure6_SHAP_subject_cluster_assembled_figures_{args.model}_{args.kind}_k{args.main_k}_manifest.csv"
    pd.DataFrame([{
        "out_base": str(out_base),
        "input_dir": str(base_dir),
        "model": args.model,
        "kind": args.kind,
        "main_k": args.main_k,
        "k_values": ",".join(map(str, k_values)),
        "outputs": ";".join(outputs),
    }]).to_csv(manifest, index=False)

    print("Saved assembled figures:")
    for out in outputs:
        print(f"  {out}")
    print(f"  {manifest}")


if __name__ == "__main__":
    main()