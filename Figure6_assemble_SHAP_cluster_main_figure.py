#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Assemble main multipanel figure for cross-cohort node-feature SHAP clusters.

Panels:
A. Node-feature SHAP hierarchical heatmap, k=4
B. AD-status composition by SHAP cluster, k=4
C. Cognitive-status composition by SHAP cluster, k=4
D. Sensitivity table across k=3–6

Expected input directory:
<OUT_BASE>/cross_cohort_aggregated/<model>/subject_shap_clusters_metadata/

Run:
python Figure6_assemble_SHAP_cluster_main_figure.py \
  --model imaging_only \
  --main-k 4
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out-base", default=str(DEFAULT_OUT_BASE))
    p.add_argument("--model", default="imaging_only")
    p.add_argument("--kind", default="node", choices=["node", "global"])
    p.add_argument("--main-k", type=int, default=4)
    p.add_argument("--k-values", default="3,4,5,6")
    p.add_argument("--dpi", type=int, default=450)
    p.add_argument("--output-name", default=None)
    return p.parse_args()


def prefix_for(base_dir: Path, model: str, kind: str, k: int) -> Path:
    return base_dir / f"cross_cohort_{model}_{kind}_SHAP_subject_clusters_k{k}"


def read_png(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing PNG: {path}")
    return mpimg.imread(path)


def show_image_panel(ax, image_path: Path, panel_label: str, title: str):
    img = read_png(image_path)
    ax.imshow(img)
    ax.axis("off")
    ax.text(
        -0.02, 1.04, panel_label,
        transform=ax.transAxes,
        fontsize=18,
        fontweight="bold",
        va="bottom",
        ha="left",
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


def load_stats_row(base_dir: Path, model: str, kind: str, k: int, variable: str):
    stats_path = prefix_for(base_dir, model, kind, k).with_name(
        prefix_for(base_dir, model, kind, k).name + "_cluster_distribution_all_variables_stats.csv"
    )
    if not stats_path.exists():
        raise FileNotFoundError(f"Missing stats file: {stats_path}")

    df = pd.read_csv(stats_path)
    row = df[df["variable"].astype(str).eq(variable)]
    if row.empty:
        raise ValueError(f"Variable {variable} not found in {stats_path}")
    return row.iloc[0]


def cluster_sizes(base_dir: Path, model: str, kind: str, k: int) -> str:
    f = prefix_for(base_dir, model, kind, k).with_name(
        prefix_for(base_dir, model, kind, k).name + "_subject_clusters_with_metadata.csv"
    )
    if not f.exists():
        return "NA"

    df = pd.read_csv(f)
    counts = df["shap_cluster"].value_counts().sort_index()
    return ", ".join(str(int(x)) for x in counts.values)


def build_sensitivity_table(ax, base_dir: Path, model: str, kind: str, k_values: list[int]):
    rows = []
    for k in k_values:
        cog = load_stats_row(base_dir, model, kind, k, "cognitive_status")
        ad = load_stats_row(base_dir, model, kind, k, "ad_status")

        rows.append([
            str(k),
            cluster_sizes(base_dir, model, kind, k),
            fmt_p(cog.get("chi_square_p")),
            fmt_p(cog.get("permutation_p")),
            fmt_v(cog.get("cramers_v")),
            fmt_p(ad.get("chi_square_p")),
            fmt_p(ad.get("permutation_p")),
            fmt_v(ad.get("cramers_v")),
        ])

    cols = [
        "k",
        "Cluster sizes",
        "Cog. χ² p",
        "Cog. perm p",
        "Cog. V",
        "AD χ² p",
        "AD perm p",
        "AD V",
    ]

    ax.axis("off")
    ax.text(
        -0.02, 1.04, "D",
        transform=ax.transAxes,
        fontsize=18,
        fontweight="bold",
        va="bottom",
        ha="left",
    )
    ax.set_title("Sensitivity of phenotype enrichment across k", fontsize=12, pad=8)

    table = ax.table(
        cellText=rows,
        colLabels=cols,
        cellLoc="center",
        colLoc="center",
        loc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.0, 1.6)

    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_text_props(weight="bold")
        if c == 1:
            cell.set_width(0.30)
        else:
            cell.set_width(0.095)

    ax.text(
        0.0, -0.05,
        "χ² p = chi-square p-value; perm p = permutation p-value; V = Cramér’s V.",
        transform=ax.transAxes,
        fontsize=8,
        ha="left",
        va="top",
    )


def main():
    args = parse_args()

    out_base = Path(args.out_base).expanduser().resolve()
    base_dir = (
        out_base
        / "cross_cohort_aggregated"
        / args.model
        / "subject_shap_clusters_metadata"
    )

    if not base_dir.exists():
        raise FileNotFoundError(f"Output directory does not exist: {base_dir}")

    k_values = [int(x.strip()) for x in args.k_values.split(",") if x.strip()]
    main_prefix = prefix_for(base_dir, args.model, args.kind, args.main_k)

    heatmap_png = main_prefix.with_name(main_prefix.name + "_hierarchical_heatmap.png")
    ad_png = main_prefix.with_name(main_prefix.name + "_cluster_distribution_ad_status.png")
    cog_png = main_prefix.with_name(main_prefix.name + "_cluster_distribution_cognitive_status.png")

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

    show_image_panel(
        ax_a,
        heatmap_png,
        "A",
        f"Cross-cohort {args.kind}-SHAP subject clustering, k={args.main_k}",
    )
    show_image_panel(
        ax_b,
        ad_png,
        "B",
        "Post-hoc AD-status composition",
    )
    show_image_panel(
        ax_c,
        cog_png,
        "C",
        "Post-hoc cognitive-status composition",
    )
    build_sensitivity_table(
        ax_d,
        base_dir=base_dir,
        model=args.model,
        kind=args.kind,
        k_values=k_values,
    )

    fig.suptitle(
        "Node-feature SHAP-defined subject clusters are enriched for cognitive and AD status",
        fontsize=17,
        fontweight="bold",
        y=0.985,
    )

    caption = (
        "Subjects were clustered using cross-cohort node-feature SHAP profiles from the imaging-only model. "
        "APOE4 carriage, sex, cognitive status, and AD status were not used for clustering and were evaluated only post hoc. "
        "The k=4 solution is shown as the primary clustering result, with sensitivity across k=3–6."
    )
    fig.text(0.02, 0.012, caption, fontsize=9, ha="left", va="bottom", wrap=True)

    final_dir = out_base / "cross_cohort_aggregated" / "final_figures"
    final_dir.mkdir(parents=True, exist_ok=True)

    if args.output_name:
        out_stem = final_dir / args.output_name
    else:
        out_stem = final_dir / f"Figure6_SHAP_subject_clusters_{args.model}_{args.kind}_k{args.main_k}_main"

    png_out = out_stem.with_suffix(".png")
    pdf_out = out_stem.with_suffix(".pdf")

    fig.savefig(png_out, dpi=args.dpi, bbox_inches="tight")
    fig.savefig(pdf_out, bbox_inches="tight")
    plt.close(fig)

    print("Saved:")
    print(f"  {png_out}")
    print(f"  {pdf_out}")


if __name__ == "__main__":
    main()