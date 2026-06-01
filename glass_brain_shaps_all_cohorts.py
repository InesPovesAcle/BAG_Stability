#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate glass-brain figures from the NEW Figure 6 edge-level SHAP outputs.

This script:
  1) reads Figure 6 SHAP edge CSVs from all cohorts / models
  2) creates individual glass brain images per cohort x model
  3) creates ONE combined panel image:
        rows    = cohorts
        columns = models / feature sets
        cells   = glass brains
  4) uses COLORED NODES by region in the combined panel by default
  5) keeps EDGES in blue-red using plt.cm.bwr

Default input:
  $WORK/ines/results/BrainAgeValidation_AllCohorts_BAGBiasCorr_OOFGlobal_BiologicalValidation/Figure6_SHAP

Default output:
  $WORK/ines/results/BrainAgeValidation_AllCohorts_BAGBiasCorr_OOFGlobal_BiologicalValidation/Figure6_SHAP/glass_brain_figure6_all_cohorts

Typical run:
  python glass_brain_figure6_all_cohorts_colored_panel.py

Debug:
  python glass_brain_figure6_all_cohorts_colored_panel.py --list-only

Optional:
  python glass_brain_figure6_all_cohorts_colored_panel.py --combine-all
"""

import argparse
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.image as mpimg

from nilearn import plotting


# =========================================================
# REGION NAMES
# =========================================================

REGION_NAMES = [
    "Left-Cerebellum-Cortex", "Left-Thalamus-Proper", "Left-Caudate", "Left-Putamen", "Left-Pallidum",
    "Left-Hippocampus", "Left-Amygdala", "Left-Accumbens-area", "Right-Cerebellum-Cortex", "Right-Thalamus-Proper",
    "Right-Caudate", "Right-Putamen", "Right-Pallidum", "Right-Hippocampus", "Right-Amygdala", "Right-Accumbens-area",
    "ctx-lh-bankssts", "ctx-lh-caudalanteriorcingulate", "ctx-lh-caudalmiddlefrontal", "ctx-lh-cuneus",
    "ctx-lh-entorhinal", "ctx-lh-fusiform", "ctx-lh-inferiorparietal", "ctx-lh-inferiortemporal",
    "ctx-lh-isthmuscingulate", "ctx-lh-lateraloccipital", "ctx-lh-lateralorbitofrontal", "ctx-lh-lingual",
    "ctx-lh-medialorbitofrontal", "ctx-lh-middletemporal", "ctx-lh-parahippocampal", "ctx-lh-paracentral",
    "ctx-lh-parsopercularis", "ctx-lh-parsorbitalis", "ctx-lh-parstriangularis", "ctx-lh-pericalcarine",
    "ctx-lh-postcentral", "ctx-lh-posteriorcingulate", "ctx-lh-precentral", "ctx-lh-precuneus",
    "ctx-lh-rostralanteriorcingulate", "ctx-lh-rostralmiddlefrontal", "ctx-lh-superiorfrontal",
    "ctx-lh-superiorparietal", "ctx-lh-superiortemporal", "ctx-lh-supramarginal", "ctx-lh-frontalpole",
    "ctx-lh-temporalpole", "ctx-lh-transversetemporal", "ctx-lh-insula", "ctx-rh-bankssts", "ctx-rh-caudalanteriorcingulate",
    "ctx-rh-caudalmiddlefrontal", "ctx-rh-cuneus", "ctx-rh-entorhinal", "ctx-rh-fusiform", "ctx-rh-inferiorparietal",
    "ctx-rh-inferiortemporal", "ctx-rh-isthmuscingulate", "ctx-rh-lateraloccipital", "ctx-rh-lateralorbitofrontal",
    "ctx-rh-lingual", "ctx-rh-medialorbitofrontal", "ctx-rh-middletemporal", "ctx-rh-parahippocampal",
    "ctx-rh-paracentral", "ctx-rh-parsopercularis", "ctx-rh-parsorbitalis", "ctx-rh-parstriangularis",
    "ctx-rh-pericalcarine", "ctx-rh-postcentral", "ctx-rh-posteriorcingulate", "ctx-rh-precentral", "ctx-rh-precuneus",
    "ctx-rh-rostralanteriorcingulate", "ctx-rh-rostralmiddlefrontal", "ctx-rh-superiorfrontal",
    "ctx-rh-superiorparietal", "ctx-rh-superiortemporal", "ctx-rh-supramarginal", "ctx-rh-frontalpole",
    "ctx-rh-temporalpole", "ctx-rh-transversetemporal", "ctx-rh-insula"
]

if len(REGION_NAMES) != 84:
    raise RuntimeError(f"Expected 84 region names, got {len(REGION_NAMES)}")


KNOWN_COHORTS = [
    "ADNI", "ADRC", "HABS", "AD_DECODE", "ADDECODE", "AD-DECODE",
    "NACC", "AIBL", "OASIS", "UKB"
]

KNOWN_FEATURE_SETS = [
    "imaging_only",
    "imaging_demographics",
    "imaging_biomarkers",
    "full",
    "full_no_cardiovascular",
]

PREFERRED_COHORT_ORDER = [
    "ADNI", "ADRC", "HABS", "AD_DECODE", "NACC", "AIBL", "OASIS", "UKB"
]

PREFERRED_FEATURE_ORDER = [
    "imaging_only",
    "imaging_demographics",
    "imaging_biomarkers",
    "full",
    "full_no_cardiovascular",
]


# =========================================================
# HELPERS
# =========================================================

def clean_region_name(x: object) -> str:
    return str(x).strip().replace('"', "")


def safe_name(x: str) -> str:
    x = str(x).strip()
    x = re.sub(r"[^A-Za-z0-9_.-]+", "_", x)
    return x.strip("_") or "unnamed"


def norm_key(x: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(x).strip().lower())


def pretty_feature_name(x: str) -> str:
    return str(x).replace("_", "\n")


def split_label(label: str) -> Tuple[str, str]:
    if "__" in label:
        cohort, feature_set = label.split("__", 1)
        return cohort, feature_set
    return label, "unknown_feature_set"


def ordered_unique(items: List[str], preferred_order: List[str]) -> List[str]:
    items_unique = list(dict.fromkeys(items))
    preferred_norm = [norm_key(x) for x in preferred_order]

    first = []
    rest = []

    for item in items_unique:
        if norm_key(item) in preferred_norm:
            first.append(item)
        else:
            rest.append(item)

    first = sorted(first, key=lambda x: preferred_norm.index(norm_key(x)))
    rest = sorted(rest, key=lambda x: norm_key(x))

    return first + rest


def get_hemi(region_name: str) -> str:
    r = clean_region_name(region_name)

    if r.startswith("ctx-lh-") or r.startswith("Left-"):
        return "Left"
    if r.startswith("ctx-rh-") or r.startswith("Right-"):
        return "Right"

    return "Unknown"


def classify_connection(region1: str, region2: str) -> str:
    h1 = get_hemi(region1)
    h2 = get_hemi(region2)

    if h1 == "Left" and h2 == "Left":
        return "Left"
    if h1 == "Right" and h2 == "Right":
        return "Right"
    if {h1, h2} == {"Left", "Right"}:
        return "Interhemispheric"

    return "Unknown"


def find_col(columns: List[str], aliases: List[str]) -> Optional[str]:
    lookup = {norm_key(c): c for c in columns}

    for alias in aliases:
        hit = lookup.get(norm_key(alias))
        if hit is not None:
            return hit

    return None


def split_connection(x: object) -> Tuple[Optional[str], Optional[str]]:
    s = clean_region_name(x)

    for sep in ["↔", "<->", "--", "__", "|", ","]:
        if sep in s:
            a, b = s.split(sep, 1)
            return clean_region_name(a), clean_region_name(b)

    return None, None


# =========================================================
# DISCOVER SHAP CSVs
# =========================================================

def inspect_edge_shap_csv(path: Path) -> Optional[Dict[str, str]]:
    """
    Return column mapping if CSV looks like an edge-level SHAP table.

    Supported inputs:
      1) Node_i, Node_j, SHAP_val
      2) Region_1, Region_2, mean_SHAP / SHAP_val / PlotWeight
      3) Connection plus SHAP column
    """
    try:
        header = pd.read_csv(path, nrows=0)
    except Exception:
        return None

    cols = list(header.columns)

    node_i = find_col(cols, [
        "Node_i", "node_i", "NodeI", "i", "edge_i", "node1", "source", "source_node"
    ])
    node_j = find_col(cols, [
        "Node_j", "node_j", "NodeJ", "j", "edge_j", "node2", "target", "target_node"
    ])

    region_1 = find_col(cols, [
        "Region_1", "region_1", "Region1", "region1", "ROI_1", "roi1"
    ])
    region_2 = find_col(cols, [
        "Region_2", "region_2", "Region2", "region2", "ROI_2", "roi2"
    ])

    connection = find_col(cols, [
        "Connection", "edge", "Edge", "edge_name", "connection_name"
    ])

    shap_col = find_col(cols, [
        "SHAP_val", "shap_val", "SHAP_value", "shap_value", "SHAP", "shap",
        "mean_SHAP", "mean_shap", "PlotWeight", "plot_weight",
        "edge_SHAP", "edge_shap"
    ])

    abs_col = find_col(cols, [
        "mean_abs_SHAP", "mean_abs_shap", "abs_SHAP", "abs_shap",
        "mean_absolute_SHAP"
    ])

    has_edges = (node_i and node_j) or (region_1 and region_2) or connection

    if has_edges and (shap_col or abs_col):
        return {
            "node_i": node_i or "",
            "node_j": node_j or "",
            "region_1": region_1 or "",
            "region_2": region_2 or "",
            "connection": connection or "",
            "shap_col": shap_col or "",
            "abs_col": abs_col or "",
        }

    return None


def infer_cohort_and_feature(path: Path, shap_root: Path) -> Tuple[str, str]:
    rel_parts = list(path.relative_to(shap_root).parts[:-1])
    parts_norm = {norm_key(p): p for p in rel_parts}

    cohort = None
    for known in KNOWN_COHORTS:
        nk = norm_key(known)
        for p in rel_parts:
            if nk == norm_key(p):
                cohort = known.replace("-", "_")
                break
        if cohort:
            break

    if cohort is None:
        cohort = rel_parts[0] if rel_parts else shap_root.name

    feature = None
    for fs in KNOWN_FEATURE_SETS:
        if norm_key(fs) in parts_norm:
            feature = fs
            break

        for p in rel_parts:
            if norm_key(fs) == norm_key(p):
                feature = fs
                break

        if feature:
            break

    if feature is None:
        feature = "unknown_feature_set"

    return safe_name(cohort), safe_name(feature)


def discover_figure6_shap_files(
    shap_root: Path,
    cohorts: Optional[List[str]],
    feature_sets: Optional[List[str]],
    collapse_feature_sets: bool,
) -> Dict[str, List[Path]]:

    shap_root = shap_root.resolve()

    if not shap_root.exists():
        raise FileNotFoundError(f"SHAP root does not exist: {shap_root}")

    cohort_filter = {norm_key(c) for c in cohorts} if cohorts else None
    fs_filter = {norm_key(f) for f in feature_sets} if feature_sets else None

    groups: Dict[str, List[Path]] = {}
    inspected = 0
    valid = 0

    for p in sorted(shap_root.rglob("*.csv")):
        name = p.name.lower()
        full = str(p).lower()

        # Avoid reading this script's own outputs
        if "glass_brain" in full:
            continue
        if "region_color_index" in name or "manifest" in name:
            continue

        # Broad search; actual validation happens through inspect_edge_shap_csv()
        if not (("shap" in name) or ("edge" in name) or ("connection" in name)):
            continue

        inspected += 1

        mapping = inspect_edge_shap_csv(p)
        if mapping is None:
            continue

        cohort, feature = infer_cohort_and_feature(p, shap_root)

        if cohort_filter and norm_key(cohort) not in cohort_filter:
            continue
        if fs_filter and norm_key(feature) not in fs_filter:
            continue

        label = cohort if collapse_feature_sets else f"{cohort}__{feature}"
        groups.setdefault(label, []).append(p)
        valid += 1

    if not groups:
        raise RuntimeError(
            f"No usable Figure 6 edge SHAP CSVs found under {shap_root}.\n"
            f"Inspected candidate CSVs: {inspected}.\n"
            "Expected either Node_i/Node_j/SHAP_val or Region_1/Region_2 plus mean_SHAP/PlotWeight columns.\n"
            "Check the real folder with:\n"
            f"  find {shap_root} -type f -iname '*shap*.csv' | head -50"
        )

    print(f"Inspected candidate CSVs: {inspected} | valid edge SHAP CSVs: {valid}")
    return groups


# =========================================================
# LOAD AND NORMALIZE SHAP TABLES
# =========================================================

def normalize_one_shap_csv(
    path: Path,
    label: str,
    node_index_base: str = "auto"
) -> Optional[pd.DataFrame]:

    mapping = inspect_edge_shap_csv(path)

    if mapping is None:
        return None

    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    out = pd.DataFrame()
    out["SourceFile"] = str(path)
    out["Label"] = label

    shap_col = mapping.get("shap_col") or mapping.get("abs_col")
    abs_col = mapping.get("abs_col")

    out["SHAP_val"] = pd.to_numeric(df[shap_col], errors="coerce") if shap_col else np.nan

    if abs_col:
        out["Abs_SHAP_input"] = pd.to_numeric(df[abs_col], errors="coerce")

    if mapping.get("region_1") and mapping.get("region_2"):
        out["Region_1"] = df[mapping["region_1"]].map(clean_region_name)
        out["Region_2"] = df[mapping["region_2"]].map(clean_region_name)

    elif mapping.get("connection"):
        split = df[mapping["connection"]].map(split_connection)
        out["Region_1"] = [x[0] for x in split]
        out["Region_2"] = [x[1] for x in split]

    else:
        ni = pd.to_numeric(df[mapping["node_i"]], errors="coerce")
        nj = pd.to_numeric(df[mapping["node_j"]], errors="coerce")

        tmp = pd.concat([ni, nj], ignore_index=True).dropna().astype(int)

        if node_index_base == "auto":
            base = 1 if len(tmp) and tmp.min() >= 1 and tmp.max() <= 84 and 0 not in set(tmp) else 0
        else:
            base = int(node_index_base)

        ni = ni - base
        nj = nj - base

        out["Node_i"] = ni
        out["Node_j"] = nj

        ok_nodes = (
            ni.notna() & nj.notna() &
            (ni >= 0) & (ni < 84) &
            (nj >= 0) & (nj < 84)
        )

        out = out.loc[ok_nodes].copy()
        out["Node_i"] = out["Node_i"].astype(int)
        out["Node_j"] = out["Node_j"].astype(int)

        out["Region_1"] = out["Node_i"].map(lambda x: REGION_NAMES[int(x)])
        out["Region_2"] = out["Node_j"].map(lambda x: REGION_NAMES[int(x)])

    out["Region_1"] = out["Region_1"].map(clean_region_name)
    out["Region_2"] = out["Region_2"].map(clean_region_name)

    out = out.dropna(subset=["Region_1", "Region_2", "SHAP_val"]).copy()
    out = out[(out["Region_1"] != "None") & (out["Region_2"] != "None")]

    return out


def load_shap_files(
    csv_paths: List[Path],
    label: str,
    node_index_base: str = "auto"
) -> pd.DataFrame:

    dfs = []

    for p in csv_paths:
        try:
            df = normalize_one_shap_csv(p, label, node_index_base=node_index_base)
            if df is not None and not df.empty:
                dfs.append(df)
        except Exception as e:
            print(f"[WARN] Could not normalize {p}: {e}")

    if not dfs:
        raise RuntimeError(f"No valid edge SHAP rows for {label}")

    return pd.concat(dfs, ignore_index=True)


# =========================================================
# ATLAS CENTROIDS
# =========================================================

def load_region_centroids(
    atlas_nii: Path,
    lookup_xlsx: Path
) -> Dict[str, List[float]]:

    if not atlas_nii.exists():
        raise FileNotFoundError(f"Atlas not found: {atlas_nii}")

    if not lookup_xlsx.exists():
        raise FileNotFoundError(f"Lookup file not found: {lookup_xlsx}")

    img = nib.load(str(atlas_nii))
    data = img.get_fdata()
    affine = img.affine

    region_labels = np.unique(data)
    region_labels = region_labels[region_labels != 0]

    centroids = []

    for label in region_labels:
        coords = np.argwhere(data == label)
        center_voxel = coords.mean(axis=0)
        center_mni = nib.affines.apply_affine(affine, center_voxel)
        centroids.append(center_mni)

    centroid_df = pd.DataFrame(centroids, columns=["X", "Y", "Z"])
    centroid_df["Label"] = region_labels.astype(int)

    lookup = pd.read_excel(lookup_xlsx)
    lookup.columns = lookup.columns.str.strip()

    required_cols = {"index", "Structure"}

    if not required_cols.issubset(set(lookup.columns)):
        raise ValueError(f"Lookup must contain {required_cols}. Found: {list(lookup.columns)}")

    lookup["Structure"] = lookup["Structure"].map(clean_region_name)
    lookup["index"] = pd.to_numeric(lookup["index"], errors="coerce")
    centroid_df["Label"] = pd.to_numeric(centroid_df["Label"], errors="coerce").astype(int)

    final_df = pd.merge(
        centroid_df,
        lookup,
        left_on="Label",
        right_on="index",
        how="inner"
    )

    region_name_to_coords = {
        clean_region_name(row["Structure"]): [row["X"], row["Y"], row["Z"]]
        for _, row in final_df.iterrows()
    }

    return region_name_to_coords


# =========================================================
# NODE COLORS
# =========================================================

def build_region_colors(all_regions_sorted: List[str]) -> Dict[str, str]:
    """
    Fixed color for each region.
    Edges remain blue-red through edge_cmap=plt.cm.bwr.
    """
    left_regions = [
        r for r in all_regions_sorted
        if r.startswith("Left-") or r.startswith("ctx-lh-")
    ]
    right_regions = [
        r for r in all_regions_sorted
        if r.startswith("Right-") or r.startswith("ctx-rh-")
    ]
    other_regions = [
        r for r in all_regions_sorted
        if r not in left_regions and r not in right_regions
    ]

    left_color_pool = []
    for cmap in [plt.cm.tab20, plt.cm.tab20b, plt.cm.tab20c]:
        left_color_pool.extend([mcolors.to_hex(cmap(i)) for i in range(cmap.N)])

    right_color_pool = [
        "#ff1744", "#00e5ff", "#ffc400", "#651fff", "#00e676", "#ff6d00",
        "#d500f9", "#00b0ff", "#c6ff00", "#ff4081", "#1de9b6", "#7c4dff",
        "#ffea00", "#18ffff", "#ff5252", "#69f0ae", "#e040fb", "#40c4ff",
        "#aeea00", "#ffab00", "#f50057", "#536dfe", "#64dd17", "#ff9100",
        "#aa00ff", "#00bfa5", "#ffd600", "#304ffe", "#ff3d00", "#76ff03",
        "#ec407a", "#26c6da", "#8e24aa", "#9ccc65", "#ffa000",
    ]

    other_color_pool = ["#8d6e63", "#607d8b", "#795548", "#9e9e9e", "#546e7a"]

    region_to_color = {}

    for i, region in enumerate(left_regions):
        region_to_color[region] = left_color_pool[i % len(left_color_pool)]

    for i, region in enumerate(right_regions):
        region_to_color[region] = right_color_pool[i % len(right_color_pool)]

    for i, region in enumerate(other_regions):
        region_to_color[region] = other_color_pool[i % len(other_color_pool)]

    return region_to_color


def save_region_color_index(
    out_root: Path,
    region_name_to_coords: Dict[str, List[float]],
    region_to_color: Dict[str, str]
) -> None:

    rows = []

    for region in sorted(region_name_to_coords):
        rows.append({
            "Region": region,
            "Hemisphere": get_hemi(region),
            "Color": region_to_color.get(region, "silver"),
        })

    df = pd.DataFrame(rows)
    df.to_csv(out_root / "region_color_index.csv", index=False)
    df.to_excel(out_root / "region_color_index.xlsx", index=False)

    fig_height = max(12, len(df) * 0.28)
    fig, ax = plt.subplots(figsize=(10, fig_height))
    y_positions = np.arange(len(df))[::-1]

    for y, (_, row) in zip(y_positions, df.iterrows()):
        ax.scatter(0, y, s=120, color=row["Color"])
        ax.text(0.15, y, row["Region"], va="center", fontsize=9)

    ax.set_xlim(-0.2, 3.5)
    ax.set_ylim(-1, len(df))
    ax.axis("off")
    ax.set_title("Region Color Index", fontsize=14)

    plt.tight_layout()
    plt.savefig(out_root / "region_color_index.png", dpi=300, bbox_inches="tight")
    plt.close()


# =========================================================
# GROUP EDGES
# =========================================================

def prepare_grouped_edges(shap_df: pd.DataFrame) -> pd.DataFrame:
    shap_df = shap_df.copy()

    shap_df["Region_1"] = shap_df["Region_1"].map(clean_region_name)
    shap_df["Region_2"] = shap_df["Region_2"].map(clean_region_name)

    grouped = (
        shap_df
        .groupby(["Region_1", "Region_2"], dropna=False)["SHAP_val"]
        .agg(
            mean_SHAP="mean",
            mean_abs_SHAP=lambda x: np.mean(np.abs(x)),
            n_values="count",
        )
        .reset_index()
    )

    grouped = grouped.sort_values("mean_abs_SHAP", ascending=False).reset_index(drop=True)

    # Signed value for edge color.
    grouped["PlotWeight"] = grouped["mean_SHAP"]

    grouped["Connection"] = grouped["Region_1"] + " ↔ " + grouped["Region_2"]
    grouped["Connection_Type"] = grouped.apply(
        lambda row: classify_connection(row["Region_1"], row["Region_2"]),
        axis=1
    )

    return grouped


# =========================================================
# CONNECTOME PLOTTING
# =========================================================

def build_connectome(
    df_edges: pd.DataFrame,
    region_name_to_coords: Dict[str, List[float]]
) -> Tuple[np.ndarray, List[List[float]], List[str]]:

    if df_edges.empty:
        raise ValueError("No edges to plot")

    regions = sorted(set(df_edges["Region_1"]) | set(df_edges["Region_2"]))

    missing = [r for r in regions if r not in region_name_to_coords]
    if missing:
        raise KeyError("Regions missing from atlas lookup:\n" + "\n".join(missing))

    region_to_index = {r: i for i, r in enumerate(regions)}
    coords = [region_name_to_coords[r] for r in regions]

    mat = np.zeros((len(regions), len(regions)))

    for _, row in df_edges.iterrows():
        i = region_to_index[row["Region_1"]]
        j = region_to_index[row["Region_2"]]

        mat[i, j] = row["PlotWeight"]
        mat[j, i] = row["PlotWeight"]

    return mat, coords, regions


def plot_connectome_file(
    df_edges: pd.DataFrame,
    region_name_to_coords: Dict[str, List[float]],
    out_png: Path,
    title: str,
    colored_nodes: bool,
    region_to_color: Optional[Dict[str, str]] = None,
    node_size: int = 70,
) -> None:

    if df_edges.empty:
        print(f"[WARN] Nothing to plot for {title}")
        return

    mat, coords, regions = build_connectome(df_edges, region_name_to_coords)

    edge_max = float(np.max(np.abs(mat)))
    if edge_max == 0 or not np.isfinite(edge_max):
        edge_max = 1.0

    if colored_nodes and region_to_color is not None:
        node_color = [region_to_color.get(r, "silver") for r in regions]
    else:
        node_color = "silver"

    display = plotting.plot_connectome(
        mat,
        coords,
        edge_threshold="0%",

        # Nodes
        node_color=node_color,
        node_size=node_size,

        # Edges stay blue-red.
        edge_cmap=plt.cm.bwr,
        edge_vmin=-edge_max,
        edge_vmax=edge_max,

        colorbar=True,
        title=title,
    )

    out_png.parent.mkdir(parents=True, exist_ok=True)
    display.savefig(str(out_png))
    display.close()
    plt.close("all")

    print(f"[OK] Saved glass brain: {out_png}")


def plot_barh(df: pd.DataFrame, out_png: Path, title: str) -> None:
    if df.empty:
        return

    plot_df = df.copy()

    plot_df["Connection_Type"] = pd.Categorical(
        plot_df["Connection_Type"],
        categories=["Left", "Right", "Interhemispheric"],
        ordered=True,
    )

    plot_df = plot_df.sort_values(
        ["Connection_Type", "mean_abs_SHAP"],
        ascending=[True, True]
    )

    color_map = {
        "Left": "steelblue",
        "Right": "darkorange",
        "Interhemispheric": "seagreen",
    }

    colors = plot_df["Connection_Type"].map(color_map)

    plt.figure(figsize=(12, max(8, 0.35 * len(plot_df))))
    plt.barh(plot_df["Connection"], plot_df["mean_abs_SHAP"], color=colors)
    plt.xlabel("Mean |SHAP|")
    plt.ylabel("Connection")
    plt.title(title)
    plt.tight_layout()

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[OK] Saved barplot: {out_png}")


# =========================================================
# PROCESS ONE COHORT x MODEL
# =========================================================

def process_one_label(
    label: str,
    shap_df: pd.DataFrame,
    out_dir: Path,
    region_name_to_coords: Dict[str, List[float]],
    region_to_color: Dict[str, str],
    top_n: int,
    hemi_top_n: int,
    colored_nodes: bool,
) -> Dict[str, Path]:

    out_dir.mkdir(parents=True, exist_ok=True)

    tables_dir = out_dir / "tables"
    glass_dir = out_dir / "glass_brains"

    tables_dir.mkdir(exist_ok=True)
    glass_dir.mkdir(exist_ok=True)

    print("\n" + "=" * 100)
    print(f"PROCESSING: {label}")
    print(f"Rows: {len(shap_df):,} | files: {shap_df['SourceFile'].nunique() if 'SourceFile' in shap_df.columns else 'NA'}")
    print("=" * 100)

    grouped = prepare_grouped_edges(shap_df)
    grouped.to_csv(tables_dir / f"{safe_name(label)}_all_edges_grouped_shap.csv", index=False)

    top_df = grouped.head(top_n).copy()
    top_df.to_csv(tables_dir / f"{safe_name(label)}_top{top_n}_edges_overall.csv", index=False)

    print("Top connections:")
    print(
        top_df[["Connection", "mean_SHAP", "mean_abs_SHAP", "n_values"]]
        .head(top_n)
        .to_string(index=False)
    )

    main_gray = glass_dir / f"{safe_name(label)}_glass_brain_top{top_n}_signed_gray.png"
    main_colored = glass_dir / f"{safe_name(label)}_glass_brain_top{top_n}_signed_colored.png"

    # Gray version
    plot_connectome_file(
        df_edges=top_df,
        region_name_to_coords=region_name_to_coords,
        out_png=main_gray,
        title=f"{label}: Top {top_n} SHAP connections",
        colored_nodes=False,
        region_to_color=region_to_color,
        node_size=55,
    )

    # Colored-node version.
    if colored_nodes:
        plot_connectome_file(
            df_edges=top_df,
            region_name_to_coords=region_name_to_coords,
            out_png=main_colored,
            title=f"{label}: Top {top_n} SHAP connections",
            colored_nodes=True,
            region_to_color=region_to_color,
            node_size=75,
        )

    hemi_df = grouped[grouped["Connection_Type"] != "Unknown"].copy()

    top_left = hemi_df[hemi_df["Connection_Type"] == "Left"].head(hemi_top_n).copy()
    top_right = hemi_df[hemi_df["Connection_Type"] == "Right"].head(hemi_top_n).copy()
    top_inter = hemi_df[hemi_df["Connection_Type"] == "Interhemispheric"].head(hemi_top_n).copy()
    top_hemi = pd.concat([top_left, top_right, top_inter], ignore_index=True)

    for name, df_part in [
        ("left", top_left),
        ("right", top_right),
        ("interhemispheric", top_inter),
        ("top_by_hemisphere", top_hemi),
    ]:
        df_part.to_csv(tables_dir / f"{safe_name(label)}_{name}_top{hemi_top_n}.csv", index=False)

    paper = top_hemi.copy()
    paper["Rank_within_group"] = paper.groupby("Connection_Type").cumcount() + 1
    paper = paper.rename(columns={"Connection_Type": "Hemisphere_Group"})

    paper = paper[[
        "Hemisphere_Group",
        "Rank_within_group",
        "Region_1",
        "Region_2",
        "Connection",
        "mean_SHAP",
        "mean_abs_SHAP",
        "PlotWeight",
        "n_values",
    ]].copy()

    paper["mean_SHAP"] = paper["mean_SHAP"].round(8)
    paper["mean_abs_SHAP"] = paper["mean_abs_SHAP"].round(8)
    paper["PlotWeight"] = paper["PlotWeight"].round(8)

    paper.to_csv(
        tables_dir / f"{safe_name(label)}_paper_table_top_edges_by_hemisphere.csv",
        index=False
    )
    paper.to_excel(
        tables_dir / f"{safe_name(label)}_paper_table_top_edges_by_hemisphere.xlsx",
        index=False
    )

    plot_barh(
        top_hemi,
        tables_dir / f"{safe_name(label)}_top_edges_by_hemisphere_barplot.png",
        f"{label}: Top SHAP connections by hemisphere",
    )

    hemi_sets = [
        (top_left, "Left", "left"),
        (top_right, "Right", "right"),
        (top_inter, "Interhemispheric", "interhemispheric"),
    ]

    for df_part, title_part, file_part in hemi_sets:
        plot_connectome_file(
            df_edges=df_part,
            region_name_to_coords=region_name_to_coords,
            out_png=glass_dir / f"{safe_name(label)}_glass_brain_{file_part}_top{hemi_top_n}_gray.png",
            title=f"{label}: {title_part} connections",
            colored_nodes=False,
            region_to_color=region_to_color,
            node_size=55,
        )

        if colored_nodes:
            plot_connectome_file(
                df_edges=df_part,
                region_name_to_coords=region_name_to_coords,
                out_png=glass_dir / f"{safe_name(label)}_glass_brain_{file_part}_top{hemi_top_n}_colored.png",
                title=f"{label}: {title_part} connections",
                colored_nodes=True,
                region_to_color=region_to_color,
                node_size=75,
            )

    out_paths = {
        "main_gray": main_gray,
    }

    if colored_nodes:
        out_paths["main_colored"] = main_colored

    return out_paths


# =========================================================
# COMBINED COHORT x MODEL PANEL
# =========================================================

def build_comparison_panel(
    generated_pngs: Dict[Tuple[str, str], Path],
    cohorts: List[str],
    feature_sets: List[str],
    out_file: Path,
    panel_title: str,
) -> None:

    n_rows = len(cohorts)
    n_cols = len(feature_sets)

    if n_rows == 0 or n_cols == 0:
        print("[WARN] Cannot build comparison panel: empty rows or columns.")
        return

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.2 * n_cols, 3.8 * n_rows),
        squeeze=False,
    )

    for i, cohort in enumerate(cohorts):
        for j, feat in enumerate(feature_sets):
            ax = axes[i, j]
            ax.axis("off")

            png_path = generated_pngs.get((cohort, feat))

            if png_path is not None and Path(png_path).exists():
                img = mpimg.imread(png_path)
                ax.imshow(img)
            else:
                ax.text(
                    0.5,
                    0.5,
                    "No image",
                    ha="center",
                    va="center",
                    fontsize=12,
                )

            if i == 0:
                ax.set_title(pretty_feature_name(feat), fontsize=12, pad=12)

            if j == 0:
                ax.text(
                    -0.10,
                    0.5,
                    cohort,
                    transform=ax.transAxes,
                    rotation=90,
                    va="center",
                    ha="center",
                    fontsize=12,
                    fontweight="bold",
                )

    fig.suptitle(panel_title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[OK] Saved comparison panel: {out_file}")


# =========================================================
# ARGUMENTS
# =========================================================

def parse_args() -> argparse.Namespace:
    default_work = os.environ.get("WORK", "/mnt/newStor/paros/paros_WORK")

    default_shap_root = str(
        Path(default_work) /
        "ines/results/BrainAgeValidation_AllCohorts_BAGBiasCorr_OOFGlobal_BiologicalValidation/Figure6_SHAP"
    )

    default_out_root = str(
        Path(default_work) /
        "ines/results/BrainAgeValidation_AllCohorts_BAGBiasCorr_OOFGlobal_BiologicalValidation/Figure6_SHAP/glass_brain_figure6_all_cohorts"
    )

    parser = argparse.ArgumentParser(
        description="Generate colored-node glass brains from Figure 6 edge SHAP CSVs."
    )

    parser.add_argument(
        "--work",
        default=default_work,
        help="Project WORK directory."
    )

    parser.add_argument(
        "--shap-root",
        default=default_shap_root,
        help="Root of Figure6_SHAP outputs."
    )

    parser.add_argument(
        "--out-root",
        default=default_out_root,
        help="Output root for figures/tables."
    )

    parser.add_argument(
        "--atlas-nii",
        default=None,
        help="Atlas NIfTI path."
    )

    parser.add_argument(
        "--lookup-xlsx",
        default=None,
        help="Atlas lookup xlsx path."
    )

    parser.add_argument(
        "--cohorts",
        nargs="*",
        default=None,
        help="Optional cohorts, e.g. ADNI ADRC HABS AD_DECODE."
    )

    parser.add_argument(
        "--feature-sets",
        nargs="*",
        default=None,
        help="Optional feature sets, e.g. imaging_only full."
    )

    parser.add_argument(
        "--collapse-feature-sets",
        action="store_true",
        help="Combine all feature sets within each cohort. This skips the cohort x model panel."
    )

    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Top overall edges per cohort/model."
    )

    parser.add_argument(
        "--hemi-top-n",
        type=int,
        default=10,
        help="Top edges per hemisphere category."
    )

    parser.add_argument(
        "--combine-all",
        action="store_true",
        help="Also create combined ALL_DISCOVERED analysis."
    )

    parser.add_argument(
        "--node-index-base",
        choices=["auto", "0", "1"],
        default="auto",
        help="Node index base for Node_i/Node_j tables."
    )

    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Only list discovered files/groups; do not plot."
    )

    parser.add_argument(
        "--panel-image-type",
        choices=["colored", "gray"],
        default="colored",
        help="Which image to insert into the comparison panel. Default: colored."
    )

    parser.add_argument(
        "--skip-colored-individuals",
        action="store_true",
        help="Do not save individual colored-node glass brains. Not recommended if panel-image-type=colored."
    )

    return parser.parse_args()


# =========================================================
# MAIN
# =========================================================

def main() -> None:
    args = parse_args()

    work = Path(args.work)
    shap_root = Path(args.shap_root)
    out_root = Path(args.out_root)

    atlas_nii = (
        Path(args.atlas_nii)
        if args.atlas_nii
        else work / "ines/data/atlas/IITmean_RPI_labels.nii.gz"
    )

    lookup_xlsx = (
        Path(args.lookup_xlsx)
        if args.lookup_xlsx
        else work / "ines/data/atlas/IITmean_RPI_index.xlsx"
    )

    out_root.mkdir(parents=True, exist_ok=True)

    print(f"WORK:       {work}")
    print(f"SHAP root:  {shap_root}")
    print(f"Output:     {out_root}")
    print(f"Atlas:      {atlas_nii}")
    print(f"Lookup:     {lookup_xlsx}")

    groups = discover_figure6_shap_files(
        shap_root=shap_root,
        cohorts=args.cohorts,
        feature_sets=args.feature_sets,
        collapse_feature_sets=args.collapse_feature_sets,
    )

    print("\nDiscovered Figure 6 SHAP groups:")
    for label, files in groups.items():
        print(f"  {label:45s} {len(files):5d} CSVs | first: {files[0]}")

    manifest = pd.DataFrame([
        {
            "Label": label,
            "Cohort": split_label(label)[0],
            "FeatureSet": split_label(label)[1],
            "n_files": len(files),
            "first_file": str(files[0]),
        }
        for label, files in groups.items()
    ])

    manifest.to_csv(out_root / "input_manifest_figure6_shap_files.csv", index=False)

    if args.list_only:
        print(f"\nList-only mode. Manifest saved: {out_root / 'input_manifest_figure6_shap_files.csv'}")
        return

    region_name_to_coords = load_region_centroids(atlas_nii, lookup_xlsx)
    print(f"Atlas regions matched: {len(region_name_to_coords)}")

    region_to_color = build_region_colors(sorted(region_name_to_coords.keys()))
    save_region_color_index(out_root, region_name_to_coords, region_to_color)

    all_dfs = []
    manifest_rows = []

    generated_pngs: Dict[Tuple[str, str], Path] = {}
    discovered_cohorts: List[str] = []
    discovered_features: List[str] = []

    colored_individuals = not args.skip_colored_individuals

    if args.panel_image_type == "colored" and args.skip_colored_individuals:
        print("[WARN] panel-image-type=colored but --skip-colored-individuals was used.")
        print("[WARN] The panel will fall back to gray images if colored images are not created.")

    for label, files in groups.items():
        cohort, feature_set = split_label(label)

        df = load_shap_files(
            csv_paths=files,
            label=label,
            node_index_base=args.node_index_base,
        )

        all_dfs.append(df)

        manifest_rows.append({
            "Label": label,
            "Cohort": cohort,
            "FeatureSet": feature_set,
            "n_files": len(files),
            "n_rows": len(df),
            "first_file": str(files[0]),
        })

        out_paths = process_one_label(
            label=label,
            shap_df=df,
            out_dir=out_root / safe_name(label),
            region_name_to_coords=region_name_to_coords,
            region_to_color=region_to_color,
            top_n=args.top_n,
            hemi_top_n=args.hemi_top_n,
            colored_nodes=colored_individuals,
        )

        discovered_cohorts.append(cohort)
        discovered_features.append(feature_set)

        if args.panel_image_type == "colored" and "main_colored" in out_paths:
            generated_pngs[(cohort, feature_set)] = out_paths["main_colored"]
        else:
            generated_pngs[(cohort, feature_set)] = out_paths["main_gray"]

    pd.DataFrame(manifest_rows).to_csv(
        out_root / "input_manifest_by_label_with_rows.csv",
        index=False
    )

    # One figure: rows = cohorts, columns = models.
    if not args.collapse_feature_sets:
        cohorts_order = ordered_unique(discovered_cohorts, PREFERRED_COHORT_ORDER)
        feature_sets_order = ordered_unique(discovered_features, PREFERRED_FEATURE_ORDER)

        panel_name = f"Figure6_glass_brain_comparison_panel_{args.panel_image_type}_nodes_bwr_edges.png"

        build_comparison_panel(
            generated_pngs=generated_pngs,
            cohorts=cohorts_order,
            feature_sets=feature_sets_order,
            out_file=out_root / panel_name,
            panel_title="Top SHAP connectome glass brains across cohorts and models",
        )
    else:
        print("[INFO] Comparison panel skipped because --collapse-feature-sets was used.")

    if args.combine_all and all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)

        process_one_label(
            label="ALL_DISCOVERED_FIGURE6_SHAPS",
            shap_df=combined,
            out_dir=out_root / "ALL_DISCOVERED_FIGURE6_SHAPS",
            region_name_to_coords=region_name_to_coords,
            region_to_color=region_to_color,
            top_n=args.top_n,
            hemi_top_n=args.hemi_top_n,
            colored_nodes=colored_individuals,
        )

    print("\nDONE")
    print(f"Main output folder: {out_root}")
    print(f"Manifest: {out_root / 'input_manifest_by_label_with_rows.csv'}")
    print(
        "Main comparison panel should be named like:\n"
        f"  {out_root / f'Figure6_glass_brain_comparison_panel_{args.panel_image_type}_nodes_bwr_edges.png'}"
    )


if __name__ == "__main__":
    main()
