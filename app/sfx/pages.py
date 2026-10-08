"""Page renderers. Each takes the shared context built in app.py."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pandas as pd
import streamlit as st

from . import config as C
from . import data as D
from . import plots as P
from .text import methods_markdown

MAX_SET_GENES = 300


@dataclass
class Ctx:
    atlas: D.Atlas
    genes: list
    missing: list
    species: list
    sample_types: list
    datasets: list
    min_cells: int
    min_detection: Optional[float]
    styles: dict
    ct: Callable[[tuple], pd.DataFrame]          # cell-type level table for genes
    tt: Callable[[tuple], pd.DataFrame]          # time-resolved table for genes
    background: Callable[[str, int], pd.DataFrame]


def show(fig, key: str | None = None):
    st.plotly_chart(P.finalize(fig), use_container_width=True, theme=None, config=C.PLOT_DOWNLOAD, key=key)


def _need_genes(ctx: Ctx, n: int = 1) -> bool:
    if len(ctx.genes) < n:
        st.info(f"Enter at least {n} gene symbol{'s' if n > 1 else ''} in the sidebar.")
        return False
    return True


VALUE_OPTIONS = {
    "log2 CPM": ("none", C.LOG2CPM_LABEL),
    "Δ log2 CPM": ("center", C.CENTERED_LABEL),
    "z-score": ("zscore", C.ZSCORE_LABEL),
}


def _scaled(df: pd.DataFrame, choice: str, within: list) -> tuple[pd.DataFrame, str, str, bool]:
    how, label = VALUE_OPTIONS[choice]
    if how == "none":
        return df.assign(scaled=df["log2cpm"]), "scaled", label, False
    return D.add_scaled(df, how, within), "scaled", label, True


# ---------------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------------

def overview(ctx: Ctx):
    a = ctx.atlas
    ds = a.datasets
    st.markdown("### Atlas overview")
    c = st.columns(5)
    c[0].metric("Datasets", len(ds))
    c[1].metric("Species", ds["Species"].nunique())
    c[2].metric("Cells", f"{ds['Cells'].sum() / 1e6:.2f} M")
    n_pb = a.summary.get("pseudobulk_samples")
    c[3].metric("Pseudobulks", f"{int(n_pb):,}" if n_pb else "n/a",
                help="Dataset × donor/sample × cell type × time point")
    c[4].metric("Genes", f"{len(a.genes):,}", help="Human symbols; non-human genes mapped to human orthologs")

    st.markdown("#### Datasets")
    st.dataframe(ds, hide_index=True, height=36 * (len(ds) + 1) + 4, column_config={
        "Cells": st.column_config.NumberColumn(format="localized"),
        "DOI": st.column_config.LinkColumn("DOI", display_text=r"https://doi\.org/(.*)"),
        "Scope": st.column_config.TextColumn(width="medium"),
        "Reference": st.column_config.TextColumn(width="medium"),
    })

    st.markdown("#### Developmental coverage")
    st.caption("Each row is one dataset; markers are sampled ages, area ∝ log10(cells). The relative scale is "
               "anchored at fertilisation (0), birth/hatching (0.45) and sexual maturity (1.0) per species. "
               "Organoids are shown open and dotted; they were placed on the fetal segment by culture age, which is "
               "an approximation (see Methods). Wang (2022) has no age metadata.")
    show(P.fig_timeline(a.tgroups, ds), "timeline")

    st.markdown("#### Cell-type composition")
    st.caption("Cells per dataset and harmonised cell type (log scale). Labels in non-vertebrate data come from "
               "label transfer and do not imply homology (see Methods).")
    show(P.fig_composition(a.groups), "composition")

    if a.stage_mapping is not None:
        with st.expander("Age-to-stage mapping table"):
            sm = a.stage_mapping.sort_values(["species", "relative_dev_time"])
            st.dataframe(sm, hide_index=True)


# ---------------------------------------------------------------------------
# Single gene
# ---------------------------------------------------------------------------

def _annotation_block(ctx: Ctx, gene: str):
    ann = D.gene_annotation(ctx.atlas, gene)
    left, right = st.columns([3, 2], vertical_alignment="top")
    with left:
        name = ann.get("gene_name") if ann else None
        st.markdown(f"### {gene}" + (f"  \n<span style='color:{C.INK_SECONDARY}'>{name}</span>" if name else ""),
                    unsafe_allow_html=True)
        if ann:
            score = ann.get("sfari_score")
            parts = [f"SFARI Gene score **{score:g}**" if pd.notna(score) else "SFARI Gene: listed, not scored"]
            if ann.get("syndromic") == 1:
                parts.append("syndromic")
            if pd.notna(ann.get("n_reports")):
                parts.append(f"{int(ann['n_reports'])} reports")
            if pd.notna(ann.get("chromosome")):
                parts.append(f"chr{ann['chromosome']}")
            st.markdown(" · ".join(parts))
            if ann.get("genetic_category"):
                st.caption(f"Evidence: {ann['genetic_category']}")
        else:
            st.caption("Not listed in SFARI Gene (release 07-08-2025).")
    with right:
        ens = ann.get("ensembl_id") if ann else None
        links = {
            "NCBI Gene": f"https://www.ncbi.nlm.nih.gov/gene/?term={gene}%5Bsym%5D+AND+human%5Borgn%5D",
            "GeneCards": f"https://www.genecards.org/cgi-bin/carddisp.pl?gene={gene}",
        }
        if ens:
            links["Ensembl"] = f"https://www.ensembl.org/id/{ens}"
            links["gnomAD"] = f"https://gnomad.broadinstitute.org/gene/{ens}?dataset=gnomad_r4"
        if ann:
            links["SFARI Gene"] = f"https://gene.sfari.org/database/human-gene/{gene}"
        st.markdown(" · ".join(f"[{k}]({v})" for k, v in links.items()))


def gene_profile(ctx: Ctx):
    a = ctx.atlas
    default = ctx.genes[0] if ctx.genes else "SCN2A"
    options = a.genes
    idx = options.index(default) if default in options else 0
    gene = st.selectbox("Gene", options, index=idx, key=f"profile_gene_{default}",
                        help="Any gene in the atlas; defaults to the first gene of the sidebar selection.")
    _annotation_block(ctx, gene)

    df = ctx.ct((gene,))
    if df.empty:
        st.warning("No groups pass the current filters.")
        return
    n_det = df.groupby("species")["detected"].agg(["sum", "size"])
    summary = " · ".join(f"{sp} {int(n_det.loc[sp, 'sum'])}/{int(n_det.loc[sp, 'size'])}"
                         for sp in C.SPECIES_ORDER if sp in n_det.index)
    st.caption(f"Detected (detection rate above threshold) in dataset × cell-type groups: {summary}.")
    absent = [ds for ds in a.absent_in(gene) if ds in set(df["dataset"])]
    if absent:
        st.warning(f"{gene} has zero counts in every cell type of {', '.join(absent)}. In this build the gene is "
                   "most likely missing from that dataset's count matrix (feature set or symbol mapping), so "
                   "these datasets carry no information on it. They are shown as not detected, not as low "
                   "expression.")

    st.markdown("#### Cell-type profile")
    vc = st.segmented_control("Values", ["log2 CPM", "z-score"], default="log2 CPM", key="strip_values",
                              help="z-score: standardised across cell types within each dataset; removes "
                                   "dataset and species offsets.")
    vc = vc or "log2 CPM"
    sdf, val, label, _ = _scaled(df, vc, ["gene", "dataset", "sample_type"])
    show(P.fig_celltype_strip(sdf, ctx.styles, val, label), "strip")
    st.caption("One point per dataset (colour and symbol = dataset; open symbols = organoids); the black tick is the "
               "mean over datasets. Grey × at the left margin marks groups where the gene is not detected. "
               "Semi-transparent points are labels without a homologous cell class in that species.")

    st.markdown("#### Developmental trajectory")
    tdf = ctx.tt((gene,))
    if tdf.empty:
        st.info("No time-resolved data for this gene under the current filters.")
    else:
        mode = st.segmented_control("Time axis", ["Relative time, all species", "Native age, per species"],
                                    default="Relative time, all species", key="traj_mode") or "Relative time, all species"
        if mode.startswith("Relative"):
            c1, c2 = st.columns([3, 1])
            avail = P._ct_order(tdf.loc[tdf["log2cpm"].notna(), "cell_type"])
            default_cts = [ct for ct in ["Neural Progenitors & Stem Cells", "Excitatory Neurons", "Inhibitory Neurons",
                                         "Astrocytes", "Oligodendrocyte Lineage", "Microglia & Macrophages"] if ct in avail]
            cts = c1.multiselect("Cell types", avail, default=default_cts or avail[:6], max_selections=9,
                                 format_func=P._short, key="traj_cts")
            vchoice = c2.selectbox("Values", ["Δ log2 CPM", "log2 CPM", "z-score"], key="traj_values",
                                   help="Δ log2 CPM: relative to the mean of the same dataset × cell type.")
            sdf, val, label, _ = _scaled(tdf, vchoice, ["gene", "dataset", "sample_type", "cell_type"])
            show(P.fig_trajectory_relative(sdf, val, label, cts), "traj_rel")
            st.caption("Points: dataset × age groups (marker area ∝ log10 cells; open = organoid). Lines: "
                       "Gaussian-kernel trend per species, weighted by √cells, with the bandwidth adapted to "
                       "sampling density and broken across unsampled periods. Vertical lines: birth/hatching (0.45) "
                       "and sexual maturity (1.0). Absolute log2 CPM is not comparable across species; use Δ log2 "
                       "CPM to compare trajectory shapes. Use the native-age view to inspect individual datasets.")
        else:
            c1, c2 = st.columns([3, 1])
            avail = P._ct_order(tdf.loc[tdf["log2cpm"].notna(), "cell_type"])
            ct = c1.selectbox("Cell type", avail, format_func=P._short, key="traj_native_ct",
                              index=avail.index("Excitatory Neurons") if "Excitatory Neurons" in avail else 0)
            vchoice = c2.selectbox("Values", ["log2 CPM", "Δ log2 CPM", "z-score"], key="traj_native_values")
            sdf, val, label, _ = _scaled(tdf, vchoice, ["gene", "dataset", "sample_type", "cell_type"])
            show(P.fig_trajectory_native(sdf, ctx.styles, val, label, ct), "traj_native")
            st.caption("Each panel uses the species' own age unit (log scale where sampling spans orders of "
                       "magnitude); the vertical line marks birth/hatching.")

    st.markdown("#### Conservation of the cell-type profile")
    in_vivo = df[df["sample_type"] == "in_vivo"]
    prof = D.species_profiles(in_vivo)
    present = [s for s in C.SPECIES_ORDER if s in set(prof.loc[prof["value"].notna(), "species"])]
    pairs = [(x, y) for i, x in enumerate(present) for y in present[i + 1:]]
    cons = D.conservation(prof, pairs)
    if cons.empty:
        st.info("Needs detected values in at least two species.")
        return
    c1, c2 = st.columns([2, 3])
    with c1:
        tbl = cons.rename(columns={"pair": "Species pair", "rho": "Spearman ρ", "p": "p", "n": "Shared cell types"})
        st.dataframe(tbl.drop(columns="gene"), hide_index=True, column_config={
            "Spearman ρ": st.column_config.NumberColumn(format="%.2f"),
            "p": st.column_config.NumberColumn(format="%.2g")})
        st.caption("In vivo datasets only; species-level profile = mean over datasets per cell type. "
                   "n < 4 shared cell types gives no estimate.")
    with c2:
        pair = st.selectbox("Species pair", [f"{x}–{y}" for x, y in pairs], key="gene_pair")
        sa, sb = pair.split("–")
        show(P.fig_pair_scatter(prof, gene, sa, sb), "gene_pair_fig")


# ---------------------------------------------------------------------------
# Gene set
# ---------------------------------------------------------------------------

def gene_set(ctx: Ctx):
    st.markdown("### Gene set across cell types")
    if not _need_genes(ctx, 2):
        return
    genes = ctx.genes[:MAX_SET_GENES]
    if len(ctx.genes) > MAX_SET_GENES:
        st.warning(f"Showing the first {MAX_SET_GENES} of {len(ctx.genes)} genes.")
    df = ctx.ct(tuple(genes))
    if df.empty:
        st.warning("No groups pass the current filters.")
        return
    view = st.segmented_control("View", ["Heatmap", "Dot plot"], default="Heatmap", key="set_view") or "Heatmap"
    if view == "Heatmap":
        c1, c2, c3 = st.columns([2, 2, 1])
        vchoice = c1.selectbox("Values", ["z-score within dataset", "z-score across all columns", "log2 CPM"],
                               key="hm_values",
                               help="Within dataset: each gene standardised across the cell types of one dataset, "
                                    "so species and dataset offsets do not dominate.")
        order = c2.radio("Group columns by", ["dataset", "cell type"], horizontal=True, key="hm_order")
        cluster = c3.toggle("Cluster genes", value=True, key="hm_cluster")
        if vchoice == "log2 CPM":
            d, val, label, div = df.assign(scaled=df["log2cpm"]), "scaled", C.LOG2CPM_LABEL, False
        else:
            within = ["gene", "dataset", "sample_type"] if vchoice.endswith("dataset") else ["gene"]
            d, val, label, div = D.add_scaled(df, "zscore", within), "scaled", C.ZSCORE_LABEL, True
        show(P.fig_heatmap(d, val, label, div, order="species" if order == "dataset" else "celltype",
                           cluster=cluster, annotations=ctx.atlas.annotations), "set_heatmap")
        st.caption("Columns are dataset × cell-type groups passing the filters; tracks show species and cell-type "
                   "lineage, and grey × marks groups where the gene is not detected. With more than 60 columns the "
                   "column labels are hidden; hover for details or restrict species/datasets in the sidebar. Genes "
                   "are clustered by average linkage on correlation distance; hover also shows SFARI Gene scores.")
    else:
        c1, _ = st.columns([2, 3])
        vchoice = c1.selectbox("Colour", ["z-score across cell types (within species)", "log2 CPM"], key="dp_values")
        prof = D.species_profiles(df)
        if vchoice.startswith("z"):
            prof = D.add_scaled(prof, "zscore", ["gene", "species", "sample_type"], value="value", out="value")
            label, div = C.ZSCORE_LABEL, True
        else:
            label, div = C.LOG2CPM_LABEL, False
        show(P.fig_dotplot(prof, label, div), "set_dotplot")
        st.caption("Datasets are averaged within species (equal weight per dataset, detected groups only). "
                   f"Dot area = mean detection rate ({C.DETECTION_LONG.lower()}), not the fraction of cells.")


# ---------------------------------------------------------------------------
# Development
# ---------------------------------------------------------------------------

def development(ctx: Ctx):
    st.markdown("### Gene set over development")
    if not _need_genes(ctx, 2):
        return
    genes = ctx.genes[:MAX_SET_GENES]
    tdf = ctx.tt(tuple(genes))
    if tdf.empty:
        st.warning("No time-resolved groups pass the current filters.")
        return
    tdf = tdf.assign(grp=[P.species_group(s, t) for s, t in zip(tdf["species"], tdf["sample_type"])])

    st.markdown("#### Expression over time in one species and cell type")
    c1, c2, c3, c4 = st.columns([2, 2, 2, 1])
    groups = P._group_order(tdf["grp"])
    grp = c1.selectbox("Species", groups, key="dev_grp")
    sub = tdf[tdf["grp"] == grp]
    cts = P._ct_order(sub["cell_type"])
    ct = c2.selectbox("Cell type", cts, format_func=P._short, key="dev_ct",
                      index=cts.index("Excitatory Neurons") if "Excitatory Neurons" in cts else 0)
    vchoice = c3.selectbox("Values", ["z-score", "Δ log2 CPM", "log2 CPM"], key="dev_values",
                           help="Scaled within gene × dataset × cell type, across time points.")
    cluster = c4.toggle("Cluster", value=True, key="dev_cluster")
    s = sub[sub["cell_type"] == ct]
    sdf, val, label, div = _scaled(s, vchoice, ["gene", "dataset", "sample_type", "cell_type"])
    show(P.fig_temporal_heatmap(sdf, ctx.styles, val, label, div, cluster), "dev_heatmap")
    st.caption("Columns are time points ordered by age; the top track shows the contributing dataset. Values are "
               "scaled within each dataset, so differences between datasets reflect dynamics, not offsets.")

    st.markdown("#### Gene-set module across species")
    avail = P._ct_order(tdf.loc[tdf["log2cpm"].notna(), "cell_type"])
    default_cts = [c for c in ["Neural Progenitors & Stem Cells", "Excitatory Neurons", "Inhibitory Neurons",
                               "Astrocytes", "Oligodendrocyte Lineage", "Microglia & Macrophages"] if c in avail]
    mcts = st.multiselect("Cell types", avail, default=default_cts or avail[:6], max_selections=9,
                          format_func=P._short, key="mod_cts")
    mdf = D.add_scaled(tdf, "center", ["gene", "dataset", "sample_type", "cell_type"])
    show(P.fig_module_trajectory(mdf, "scaled", C.CENTERED_LABEL, mcts), "dev_module")
    st.caption("For each gene, log2 CPM is centred within dataset × cell type (Δ log2 CPM). Points show the median "
               "across genes per dataset × age; lines are kernel-smoothed species trends (as on the Gene page). The "
               "median summarises shared dynamics of the set and can hide opposing sub-modules; check the heatmap "
               "above.")


# ---------------------------------------------------------------------------
# Conservation
# ---------------------------------------------------------------------------

def conservation_page(ctx: Ctx):
    st.markdown("### Conservation of cell-type profiles")
    st.caption("For each gene and species pair: Spearman correlation between the two species' cell-type profiles "
               "(in vivo datasets, mean over datasets per cell type). High ρ means the gene ranks cell types "
               "similarly in both species. Absolute levels are not compared.")
    if not _need_genes(ctx, 1):
        return
    genes = ctx.genes[:MAX_SET_GENES]
    df = ctx.ct(tuple(genes))
    df = df[df["sample_type"] == "in_vivo"] if not df.empty else df
    if df.empty:
        st.warning("No in vivo groups pass the current filters.")
        return
    prof = D.species_profiles(df)
    present = [s for s in C.SPECIES_ORDER if s in set(prof["species"])]
    if len(present) < 2:
        st.info("Select at least two species in the sidebar.")
        return
    c1, c2 = st.columns(2)
    ref = c1.selectbox("Reference species", present, key="cons_ref")
    min_shared = c2.slider("Minimum shared cell types", 4, 10, 4, key="cons_min")
    pairs = [(ref, s) for s in present if s != ref]
    cons = D.conservation(prof, pairs, min_shared)
    bg = ctx.background(ref, min_shared)
    bg = bg[bg["pair"].isin([f"{x}–{y}" for x, y in pairs])]

    fig, stats = P.fig_conservation_summary(cons, bg)
    show(fig, "cons_summary")
    if not stats.empty:
        st.dataframe(stats, hide_index=True, column_config={
            "Median ρ (selected)": st.column_config.NumberColumn(format="%.2f"),
            "Median ρ (background)": st.column_config.NumberColumn(format="%.2f"),
            "Mann–Whitney p": st.column_config.NumberColumn(format="%.2g")})
        st.caption("Background: up to 600 randomly drawn genes (fixed seed) that pass the same filters and "
                   "detection criteria. p-values are not corrected for the number of species pairs.")
    if len(genes) > 1:
        st.markdown("#### Per gene")
        show(P.fig_conservation_heatmap(cons), "cons_heatmap")

    st.markdown("#### Inspect one gene")
    c1, c2 = st.columns(2)
    gene = c1.selectbox("Gene", genes, key="cons_gene")
    other = c2.selectbox("Compare with", [s for s in present if s != ref], key="cons_other")
    show(P.fig_pair_scatter(prof, gene, ref, other), "cons_scatter")


# ---------------------------------------------------------------------------
# Cell atlas
# ---------------------------------------------------------------------------

def cell_atlas(ctx: Ctx):
    st.markdown("### Cell atlas")
    u = ctx.atlas.umap
    if u is None:
        st.info("This data build has no embedding.")
        return
    m = np.ones(len(u), bool)
    if ctx.species:
        m &= u["species"].isin(ctx.species).to_numpy()
    if ctx.sample_types and "sample_type" in u:
        m &= u["sample_type"].isin(ctx.sample_types).to_numpy()
    if ctx.datasets:
        m &= u["dataset"].isin(ctx.datasets).to_numpy()
    u = u[m]
    c1, c2, c3 = st.columns([3, 2, 2])
    color_by = c1.segmented_control("Colour by", ["Species", "Sample type", "Lineage", "Dataset", "Cell type"],
                                    default="Species", key="umap_color") or "Species"
    col = {"Species": "species", "Sample type": "sample_type", "Lineage": "lineage",
           "Dataset": "dataset", "Cell type": "cell_type"}[color_by]
    highlight = None
    if col in ("dataset", "cell_type", "lineage"):
        if col == "dataset":
            opts = sorted(u[col].unique())
        elif col == "lineage":
            opts = [l for l in C.LINEAGES if l in set(u[col])]
        else:
            opts = P._ct_order(u[col].unique())
        highlight = c2.selectbox(f"Highlight {color_by.lower()}", opts, key=f"umap_hl_{col}")
    n = c3.select_slider("Cells shown", [25_000, 50_000, 100_000, 200_000], value=100_000, key="umap_n")
    if len(u) > n:
        u = u.sample(n=n, random_state=0)
    show(P.fig_umap(u, col, highlight), "umap")
    bi = ctx.atlas.build_info.get("integration", {})
    prov = (f"Embedding: UMAP of the {bi.get('method')} latent space ({bi.get('version', '')})."
            if bi else "Embedding provenance is not recorded in this data build.")
    st.caption(f"{prov} Random subsample of {len(u):,} cells. For variables with more categories than can be told "
               "apart reliably in a scatter (lineage, dataset, cell type), one category is highlighted at a time. "
               "UMAP distances between distant clusters are not interpretable.")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

COLUMN_HELP = {
    "log2cpm": C.LOG2CPM_LONG + "; empty where not detected",
    "log2cpm_unmasked": "log2 CPM without detection masking (voom value for zero counts included)",
    "detection": C.DETECTION_LONG,
    "detected": "detection rate above the sidebar threshold",
    "measured": "False if the gene has zero counts in every group of the dataset (likely absent from its matrix)",
    "n_cells": "cells contributing to the group",
    "relative_dev_time": "relative developmental time (0 fertilisation, 0.45 birth/hatching, 1 maturity)",
    "numeric_time": "age in the species' native unit (see Methods)",
}


def data_page(ctx: Ctx):
    st.markdown("### Data for the current selection")
    if not _need_genes(ctx, 1):
        return
    genes = tuple(ctx.genes)
    ct = ctx.ct(genes)
    tt = ctx.tt(genes)
    keep_ct = ["gene", "species", "dataset", "sample_type", "cell_type", "lineage", "n_cells",
               "log2cpm", "log2cpm_unmasked", "detection", "detected", "measured"]
    keep_tt = ["gene", "species", "dataset", "sample_type", "cell_type", "timepoint", "age_label", "numeric_time",
               "relative_dev_time", "n_cells", "log2cpm", "log2cpm_unmasked", "detected", "measured"]
    cfg = {k: st.column_config.Column(help=v) for k, v in COLUMN_HELP.items()}
    st.markdown(f"#### Cell-type level ({len(ct):,} rows)")
    if not ct.empty:
        st.dataframe(ct[keep_ct], hide_index=True, height=320, column_config=cfg)
        st.download_button("Download CSV", ct[keep_ct].to_csv(index=False), "sfariexplorer_celltype.csv", "text/csv",
                           key="dl_ct")
    st.markdown(f"#### Time-resolved ({len(tt):,} rows)")
    if not tt.empty:
        st.dataframe(tt[keep_tt], hide_index=True, height=320, column_config=cfg)
        st.download_button("Download CSV", tt[keep_tt].to_csv(index=False), "sfariexplorer_temporal.csv", "text/csv",
                           key="dl_tt")
    st.caption("Rows respect the sidebar filters. Hover a column header for its definition.")


def methods(ctx: Ctx):
    st.markdown("### Methods and definitions")
    st.markdown(methods_markdown(ctx.atlas))
