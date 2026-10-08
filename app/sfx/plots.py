"""Plotly figure builders. Every function consumes tidy frames from ``sfx.data``.

Conventions
  * colour follows the entity: species and datasets keep their colour under
    any filter; datasets are double-encoded with a marker symbol;
  * organoids are drawn with open markers / dashed lines;
  * groups where a gene is not detected are never drawn as values: they are
    shown as grey "not detected" marks (heatmaps: ×, strips: left margin);
  * sequential scale for log2 CPM, diverging (grey midpoint) for Δ / z.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.stats import mannwhitneyu

from . import config as C

FONT = '"Source Sans", "Source Sans Pro", "Helvetica Neue", Helvetica, Arial, sans-serif'
SPECIES_ABBR = {"Human": "Hs", "Mouse": "Mm", "Zebrafish": "Dr", "Drosophila": "Dm"}


def _template() -> go.layout.Template:
    axis = dict(showline=True, linecolor=C.AXIS, linewidth=1, ticks="outside", tickcolor=C.AXIS,
                ticklen=4, showgrid=False, zeroline=False, automargin=True,
                tickfont=dict(size=11, color=C.INK_SECONDARY),
                title=dict(font=dict(size=12, color=C.INK)))
    return go.layout.Template(layout=dict(
        font=dict(family=FONT, size=12, color=C.INK),
        paper_bgcolor=C.SURFACE, plot_bgcolor=C.SURFACE, colorway=C.DATASET_SLOTS,
        xaxis=axis, yaxis={**axis, "showgrid": True, "gridcolor": C.GRID},
        legend=dict(font=dict(size=11), bgcolor="rgba(255,255,255,0)", itemsizing="constant",
                    title=dict(font=dict(size=11, color=C.INK_SECONDARY))),
        hoverlabel=dict(bgcolor="white", bordercolor=C.AXIS, font=dict(family=FONT, size=12, color=C.INK)),
        margin=dict(l=64, r=24, t=56, b=52),
        title=dict(font=dict(size=14, color=C.INK), x=0, xanchor="left"),
    ))


TEMPLATE = _template()


def finalize(fig: go.Figure) -> go.Figure:
    """Pin surface colours and font on the figure itself.

    Streamlit's frontend overrides template-level backgrounds/fonts with its
    own theme even when ``theme=None``; explicit layout values take precedence.
    """
    lay = fig.layout
    fig.update_layout(
        plot_bgcolor=lay.plot_bgcolor or C.SURFACE,
        paper_bgcolor=lay.paper_bgcolor or C.SURFACE,
        font=dict(family=FONT, size=lay.font.size or 12, color=lay.font.color or C.INK),
    )
    return fig


def empty(message: str, height: int = 260) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False,
                       font=dict(size=13, color=C.INK_MUTED))
    fig.update_layout(template=TEMPLATE, height=height, xaxis=dict(visible=False), yaxis=dict(visible=False))
    return fig


# ---------------------------------------------------------------------------
# Shared encodings
# ---------------------------------------------------------------------------

def dataset_styles(datasets: pd.DataFrame) -> dict:
    """dataset -> {color, symbol}; slots assigned per species in a fixed order."""
    styles = {}
    for sp in C.SPECIES_ORDER:
        sub = datasets[datasets["Species"] == sp]
        for i, (ds, stype) in enumerate(zip(sub["Dataset"], sub["Sample type"])):
            symbol = C.DATASET_SYMBOLS[i % len(C.DATASET_SYMBOLS)]
            styles[ds] = dict(color=C.DATASET_SLOTS[i % len(C.DATASET_SLOTS)],
                              symbol=symbol + ("-open" if stype == "organoid" else ""),
                              organoid=stype == "organoid")
    return styles


def ds_label(species: str, dataset: str, sample_type: str = "in_vivo") -> str:
    org = " org." if sample_type == "organoid" else ""
    return f"{SPECIES_ABBR.get(species, species)}{org} · {dataset}"


def species_group(species: str, sample_type: str) -> str:
    return f"{species} organoid" if sample_type == "organoid" else species


def _group_order(keys) -> list:
    def k(g):
        sp, _, org = g.partition(" organoid")
        return (C.SPECIES_ORDER.index(sp) if sp in C.SPECIES_ORDER else 9, g.endswith("organoid"))
    return sorted(set(keys), key=k)


def _ct_order(cts) -> list:
    present = set(cts)
    return [c for c in C.CELLTYPE_ORDER if c in present] + sorted(present - set(C.CELLTYPE_ORDER))


def _short(ct: str) -> str:
    return C.CELLTYPE_SHORT.get(ct, ct)


def _cells_to_size(n, lo=5.0, hi=15.0) -> np.ndarray:
    v = np.log10(np.clip(np.asarray(n, float), 1, None))
    return lo + (hi - lo) * np.clip(v / 5.0, 0, 1)  # 1 cell -> lo, 100k cells -> hi


def _diverging_range(values, cap=3.0) -> float:
    v = np.abs(np.asarray(values, float))
    v = v[np.isfinite(v)]
    return float(min(cap, np.nanpercentile(v, 98))) if len(v) else 1.0


def kernel_trend(x, y, w=None, n_grid=200):
    """Gaussian-kernel (Nadaraya–Watson) trend, weighted by ``w``.

    The bandwidth adapts to the sampling density (1.5 × median spacing of the
    distinct ages, clipped to [0.04, 0.08] relative-time units). The line is
    broken across sampling gaps wider than max(4 h, 0.08), so the trend never
    bridges unsampled periods. Returns (grid, yhat) with NaN at breaks.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    w = np.ones_like(x) if w is None else np.asarray(w, float)
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(w)
    x, y, w = x[ok], y[ok], w[ok]
    ux = np.unique(x)
    if len(ux) < 3:
        return np.array([]), np.array([])
    h = float(np.clip(1.5 * np.median(np.diff(ux)), 0.04, 0.08))
    breaks = np.flatnonzero(np.diff(ux) > max(4 * h, 0.08))
    segments = np.split(ux, breaks + 1)
    gx, gy = [], []
    for seg in segments:
        if len(seg) < 3:
            continue
        grid = np.linspace(seg[0], seg[-1], max(10, int(n_grid * (seg[-1] - seg[0]) / (ux[-1] - ux[0] + 1e-9))))
        k = np.exp(-0.5 * ((grid[:, None] - x[None, :]) / h) ** 2) * w[None, :]
        gx += list(grid) + [np.nan]
        gy += list((k * y).sum(1) / k.sum(1)) + [np.nan]
    return np.array(gx), np.array(gy)


def _add_time_anchors(fig, rows_cols, rel=True, birth=None):
    """Hairlines at the anchors of the relative time scale (or native birth)."""
    xs = [(0.45, "birth/hatching"), (1.0, "maturity")] if rel else ([(birth, "birth")] if birth else [])
    for (r, c) in rows_cols:
        for x, _ in xs:
            fig.add_vline(x=x, line=dict(color=C.AXIS, width=1), layer="below", row=r, col=c)
    return xs


# ---------------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------------

def fig_timeline(tgroups: pd.DataFrame, datasets: pd.DataFrame) -> go.Figure:
    t = tgroups.dropna(subset=["relative_dev_time"])
    per_tp = (t.groupby(["species", "sample_type", "dataset", "relative_dev_time", "age_label"], observed=True)
                ["n_cells"].sum().reset_index())
    if per_tp.empty:
        return empty("No time-annotated samples in this build.")
    order = (per_tp.groupby(["species", "sample_type", "dataset"])["relative_dev_time"].min().reset_index())
    order["s"] = order["species"].map({s: i for i, s in enumerate(C.SPECIES_ORDER)})
    order = order.sort_values(["s", "sample_type", "relative_dev_time"], ascending=[True, True, True])
    labels = [ds_label(r.species, r.dataset, r.sample_type) for r in order.itertuples()]
    fig = go.Figure()
    shown = set()
    for r, lab in zip(order.itertuples(), labels):
        d = per_tp[(per_tp["dataset"] == r.dataset) & (per_tp["sample_type"] == r.sample_type)]
        color = C.SPECIES_COLORS.get(r.species, C.INK_MUTED)
        org = r.sample_type == "organoid"
        fig.add_trace(go.Scatter(x=[d["relative_dev_time"].min(), d["relative_dev_time"].max()], y=[lab, lab],
                                 mode="lines", line=dict(color=color, width=2, dash="dot" if org else "solid"),
                                 hoverinfo="skip", showlegend=False))
        grp = species_group(r.species, r.sample_type)
        fig.add_trace(go.Scatter(
            x=d["relative_dev_time"], y=[lab] * len(d), mode="markers", name=grp, legendgroup=grp,
            showlegend=grp not in shown,
            marker=dict(size=_cells_to_size(d["n_cells"], 6, 14), color="white" if org else color,
                        line=dict(color=color if org else "white", width=1.5),
                        symbol="circle"),
            customdata=np.c_[d["age_label"], d["n_cells"]],
            hovertemplate=f"<b>{r.dataset}</b><br>%{{customdata[0]}} · rel. time %{{x:.2f}}"
                          "<br>%{customdata[1]:,} cells<extra></extra>"))
        shown.add(grp)
    for x, name in ((0.45, "birth / hatching"), (1.0, "sexual maturity")):
        fig.add_vline(x=x, line=dict(color=C.AXIS, width=1), layer="below")
        fig.add_annotation(x=x, y=1.0, yref="paper", yanchor="bottom", text=name, showarrow=False,
                           font=dict(size=10, color=C.INK_MUTED))
    fig.update_layout(template=TEMPLATE, height=90 + 26 * len(labels),
                      margin=dict(t=40, l=10),
                      xaxis=dict(title="Relative developmental time (0 = fertilization)", range=[-0.02, 1.5]),
                      yaxis=dict(autorange="reversed", showgrid=False, title=None),
                      legend=dict(orientation="h", y=-0.22, x=0, title=None))
    return fig


def fig_composition(groups: pd.DataFrame) -> go.Figure:
    g = groups.copy()
    g["row"] = [ds_label(s, d, st) for s, d, st in zip(g["species"], g["dataset"], g["sample_type"])]
    g["s"] = g["species"].map({s: i for i, s in enumerate(C.SPECIES_ORDER)})
    rows = g.sort_values(["s", "sample_type", "dataset"])["row"].drop_duplicates().tolist()
    cols = _ct_order(g["cell_type"])
    piv = g.pivot_table(index="row", columns="cell_type", values="n_cells", aggfunc="sum").reindex(index=rows, columns=cols)
    z = np.log10(piv.to_numpy(float))
    ticks = [1, 10, 100, 1_000, 10_000, 100_000]
    fig = go.Figure(go.Heatmap(
        z=z, x=[_short(c) for c in cols], y=rows, colorscale=C.SEQUENTIAL, zmin=0, zmax=5.5,
        xgap=1, ygap=1, customdata=piv.to_numpy(float),
        hovertemplate="%{y}<br>%{x}<br>%{customdata:,.0f} cells<extra></extra>",
        colorbar=dict(title=dict(text="cells", side="right"), tickvals=np.log10(ticks),
                      ticktext=["1", "10", "100", "1k", "10k", "100k"], thickness=10, len=0.6, outlinewidth=0)))
    fig.update_layout(template=TEMPLATE, height=120 + 24 * len(rows), margin=dict(t=24, l=10, b=10),
                      xaxis=dict(side="top", tickangle=-45, showline=False, ticks=""),
                      yaxis=dict(autorange="reversed", showgrid=False, showline=False, ticks=""),
                      plot_bgcolor="#f7f7f5")
    return fig


# ---------------------------------------------------------------------------
# Single gene
# ---------------------------------------------------------------------------

def fig_celltype_strip(df: pd.DataFrame, styles: dict, value: str, value_label: str,
                       title: str = "") -> go.Figure:
    """Cell-type profile of one gene: one point per dataset, faceted by species group."""
    if df.empty:
        return empty("No groups pass the current filters.")
    df = df.copy()
    df["grp"] = [species_group(s, t) for s, t in zip(df["species"], df["sample_type"])]
    groups = _group_order(df["grp"])
    cts = _ct_order(df["cell_type"])
    det = df[df[value].notna()]
    if det.empty:
        return empty("The gene is not detected in any group that passes the filters.")
    lo, hi = det[value].min(), det[value].max()
    pad = 0.08 * max(hi - lo, 1.0)
    nd_x = lo - pad
    fig = make_subplots(rows=1, cols=len(groups), shared_yaxes=True, horizontal_spacing=0.025,
                        subplot_titles=groups)
    legend_seen = set()
    for c, grp in enumerate(groups, start=1):
        sub = df[df["grp"] == grp]
        for ds, d in sub.groupby("dataset", sort=False):
            st = styles.get(ds, dict(color=C.INK_MUTED, symbol="circle"))
            dd = d[d[value].notna()]
            if len(dd):
                opac = np.where(dd["caveat"].astype(bool), 0.4, 1.0)
                first_in_group = grp not in legend_seen
                legend_seen.add(grp)
                fig.add_trace(go.Scatter(
                    x=dd[value], y=dd["cell_type"].map(_short), mode="markers", name=ds,
                    legendgroup=grp, legendgrouptitle=dict(text=grp) if first_in_group else None,
                    legendrank=100 * c, showlegend=ds not in legend_seen,
                    marker=dict(color=st["color"], symbol=st["symbol"], size=10, opacity=opac,
                                line=dict(color="white", width=1)),
                    customdata=np.c_[dd["n_cells"], dd["detection"], dd["caveat"]],
                    hovertemplate=(f"<b>{ds}</b> · %{{y}}<br>{value_label}: %{{x:.2f}}"
                                   "<br>%{customdata[0]:,} cells · detection %{customdata[1]:.0%}"
                                   "<br><i>%{customdata[2]}</i><extra></extra>")), row=1, col=c)
                legend_seen.add(ds)
            nd = d[d[value].isna()]
            if len(nd):
                fig.add_trace(go.Scatter(
                    x=[nd_x] * len(nd), y=nd["cell_type"].map(_short), mode="markers", name="not detected",
                    legendgroup="marks", legendrank=10_000, showlegend="nd" not in legend_seen,
                    marker=dict(symbol="x-thin-open", size=8, color=C.NOT_DETECTED, line=dict(width=1.5, color=C.NOT_DETECTED)),
                    customdata=np.c_[nd["n_cells"], nd["detection"],
                                     np.where(nd.get("measured", True), "not detected",
                                              "gene absent from this dataset (zero counts in every group)")],
                    hovertemplate=f"<b>{ds}</b> · %{{y}}<br>%{{customdata[2]}} (detection %{{customdata[1]:.0%}})"
                                  "<br>%{customdata[0]:,} cells<extra></extra>"), row=1, col=c)
                legend_seen.add("nd")
        means = sub.groupby("cell_type")[value].mean().dropna()
        if len(means):
            fig.add_trace(go.Scatter(
                x=means.values, y=[_short(c) for c in means.index], mode="markers", name="mean of datasets",
                legendgroup="marks", legendrank=9_999, showlegend="mean" not in legend_seen,
                marker=dict(symbol="line-ns", size=18, line=dict(width=2.5, color=C.INK)),
                hovertemplate="mean of datasets: %{x:.2f}<extra></extra>"), row=1, col=c)
            legend_seen.add("mean")
        fig.update_xaxes(range=[nd_x - pad * 0.6, hi + pad], row=1, col=c,
                         showgrid=True, gridcolor=C.GRID)
    fig.update_yaxes(categoryorder="array", categoryarray=[_short(c) for c in cts][::-1], showgrid=False)
    for ann in list(fig.layout.annotations)[:len(groups)]:
        sp = ann.text.replace(" organoid", "")
        ann.font = dict(size=12, color=C.SPECIES_COLORS.get(sp, C.INK))
    fig.update_layout(template=TEMPLATE, title=title or None, height=230 + 30 * len(cts),
                      legend=dict(orientation="h", x=0, y=0, yanchor="bottom", yref="container",
                                  groupclick="toggleitem", tracegroupgap=24),
                      margin=dict(t=64 if title else 40, b=150))
    fig.add_annotation(text=value_label, x=0.5, y=0, xref="paper", yref="paper", yshift=-38,
                       showarrow=False, font=dict(size=12, color=C.INK))
    return fig


def fig_trajectory_relative(tdf: pd.DataFrame, value: str, value_label: str, cell_types: list,
                            ncols: int = 3, show_trend: bool = True, show_points: bool = True) -> go.Figure:
    """One gene (or module) across species on the relative developmental time scale."""
    tdf = tdf[tdf["cell_type"].isin(cell_types)]
    if tdf.empty or tdf[value].notna().sum() == 0:
        return empty("No detected time-resolved values for this selection.")
    cts = [c for c in _ct_order(tdf["cell_type"]) if c in cell_types]
    nrows = int(np.ceil(len(cts) / ncols))
    ncols = min(ncols, len(cts))
    fig = make_subplots(rows=nrows, cols=ncols, shared_yaxes=True, shared_xaxes=True,
                        subplot_titles=[_short(c) for c in cts],
                        horizontal_spacing=0.04, vertical_spacing=0.12 if nrows > 1 else 0.05)
    _trajectory_panels(fig, tdf, value, value_label, cts, ncols, show_trend, show_points)
    return _trajectory_layout(fig, cts, ncols, nrows, value_label, zero_line=value != "log2cpm")


def _trajectory_panels(fig, tdf, value, value_label, cts, ncols, show_trend=True, show_points=True,
                       hover_extra: str = ""):
    """Dataset-level points (context) + kernel-smoothed species trend (summary)."""
    seen = set()
    for i, ct in enumerate(cts):
        r, c = i // ncols + 1, i % ncols + 1
        sub = tdf[(tdf["cell_type"] == ct) & tdf[value].notna()]
        for (sp, stype), g in sub.groupby(["species", "sample_type"], sort=False):
            grp = species_group(sp, stype)
            color = C.SPECIES_COLORS.get(sp, C.INK_MUTED)
            org = stype == "organoid"
            rank = 10 * (C.SPECIES_ORDER.index(sp) if sp in C.SPECIES_ORDER else 9) + int(org)
            if show_points or not show_trend:
                fig.add_trace(go.Scatter(
                    x=g["relative_dev_time"], y=g[value], mode="markers", name=f"{grp} (data)",
                    legendgroup=grp, showlegend=False,
                    marker=dict(size=_cells_to_size(g["n_cells"], 4, 10), color="white" if org else color,
                                opacity=0.55, line=dict(color=color if org else "white", width=1)),
                    customdata=np.c_[g["dataset"], g["age_label"], g["n_cells"]],
                    hovertemplate=(f"<b>{sp} · %{{customdata[0]}}</b><br>{_short(ct)} · %{{customdata[1]}}"
                                   f" (rel. %{{x:.2f}})<br>{value_label}: %{{y:.2f}}<br>%{{customdata[2]:,}} cells"
                                   f"{hover_extra}<extra></extra>")), row=r, col=c)
            if show_trend:
                gx, gy = kernel_trend(g["relative_dev_time"], g[value], np.sqrt(g["n_cells"].clip(lower=1)))
                if len(gx):
                    fig.add_trace(go.Scatter(
                        x=gx, y=gy, mode="lines", name=grp, legendgroup=grp, showlegend=grp not in seen,
                        legendrank=rank,
                        line=dict(color=color, width=3, dash="dash" if org else "solid"),
                        hovertemplate=f"<b>{grp}</b> trend<br>rel. %{{x:.2f}}: %{{y:.2f}}<extra></extra>"),
                        row=r, col=c)
                    seen.add(grp)
            if grp not in seen:  # too few ages for a trend: legend entry from the points
                fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=grp, legendgroup=grp,
                                         legendrank=rank, marker=dict(color="white" if org else color, size=8,
                                                     line=dict(color=color, width=1))), row=r, col=c)
                seen.add(grp)


def _trajectory_layout(fig, cts, ncols, nrows, value_label, zero_line=True):
    _add_time_anchors(fig, [(i // ncols + 1, i % ncols + 1) for i in range(len(cts))])
    if zero_line:
        for i in range(len(cts)):
            fig.add_hline(y=0, line=dict(color=C.AXIS, width=1), layer="below", row=i // ncols + 1, col=i % ncols + 1)
    for ann in list(fig.layout.annotations)[:len(cts)]:
        ann.font = dict(size=12, color=C.INK)
    fig.update_xaxes(range=[-0.02, 1.5], showgrid=False)
    fig.update_xaxes(title_text="Relative developmental time", row=nrows)
    fig.update_yaxes(title_text=value_label, col=1)
    fig.update_layout(template=TEMPLATE, height=110 + 260 * nrows,
                      legend=dict(orientation="h", x=0, y=1, yanchor="top", yref="container"),
                      margin=dict(t=90))
    return fig


def fig_trajectory_native(tdf: pd.DataFrame, styles: dict, value: str, value_label: str,
                          cell_type: str) -> go.Figure:
    """One gene in one cell type, each species on its own native age axis."""
    tdf = tdf[tdf["cell_type"] == cell_type]
    if tdf.empty or tdf[value].notna().sum() == 0:
        return empty(f"No detected time-resolved values in {_short(cell_type)}.")
    tdf = tdf.copy()
    tdf["grp"] = [species_group(s, t) for s, t in zip(tdf["species"], tdf["sample_type"])]
    has = set(tdf.loc[tdf[value].notna(), "grp"])
    groups = [g for g in _group_order(tdf["grp"]) if g in has]
    missing = [g for g in _group_order(tdf["grp"]) if g not in has]
    tdf = tdf[tdf["grp"].isin(has) & tdf[value].notna()]
    ncols = min(3, len(groups))
    nrows = int(np.ceil(len(groups) / ncols))
    fig = make_subplots(rows=nrows, cols=ncols, shared_yaxes=True, subplot_titles=groups,
                        horizontal_spacing=0.06, vertical_spacing=0.2 if nrows > 1 else 0.05)
    for i, grp in enumerate(groups):
        r, c = i // ncols + 1, i % ncols + 1
        sub = tdf[tdf["grp"] == grp]
        sp, stype = sub["species"].iloc[0], sub["sample_type"].iloc[0]
        for ds, d in sub.groupby("dataset", sort=False):
            d = d.sort_values("numeric_time")
            st = styles.get(ds, dict(color=C.INK_MUTED, symbol="circle"))
            fig.add_trace(go.Scatter(
                x=d["numeric_time"], y=d[value], mode="lines+markers", name=ds, legendgroup=grp,
                legendgrouptitle=dict(text=grp), legendrank=100 * i,
                line=dict(color=st["color"], width=1.5), connectgaps=False,
                marker=dict(symbol=st["symbol"], color=st["color"], size=_cells_to_size(d["n_cells"], 6, 13),
                            line=dict(color="white", width=1)),
                customdata=np.c_[d["age_label"], d["n_cells"]],
                hovertemplate=(f"<b>{ds}</b><br>%{{customdata[0]}}<br>{value_label}: %{{y:.2f}}"
                               "<br>%{customdata[1]:,} cells<extra></extra>")), row=r, col=c)
        key = (sp, stype)
        is_log = C.NATIVE_TIME_LOG.get(key, False)
        fig.update_xaxes(type="log" if is_log else "linear", title_text=C.NATIVE_TIME_AXIS.get(key, "Age"),
                         title_font=dict(size=11), row=r, col=c)
        if key in C.BIRTH_NATIVE and sub["numeric_time"].max() > C.BIRTH_NATIVE[key] > sub["numeric_time"].min():
            fig.add_vline(x=C.BIRTH_NATIVE[key], line=dict(color=C.AXIS, width=1), layer="below", row=r, col=c)
    for ann in list(fig.layout.annotations)[:len(groups)]:
        ann.font = dict(size=12, color=C.SPECIES_COLORS.get(ann.text.replace(" organoid", ""), C.INK))
    fig.update_yaxes(title_text=value_label, col=1)
    fig.update_layout(template=TEMPLATE, height=200 + 280 * nrows,
                      legend=dict(orientation="h", x=0, y=0, yanchor="bottom", yref="container",
                                  groupclick="toggleitem", tracegroupgap=24),
                      margin=dict(b=140))
    if missing:
        fig.update_layout(title=dict(text=f"Not detected in {_short(cell_type)}: {', '.join(missing)}",
                                     font=dict(size=11, color=C.INK_MUTED)), margin=dict(t=80))
    return fig


# ---------------------------------------------------------------------------
# Gene sets: heatmap and dot plot
# ---------------------------------------------------------------------------

def _cluster_rows(mat: np.ndarray) -> np.ndarray:
    if mat.shape[0] < 3:
        return np.arange(mat.shape[0])
    m = mat.copy()
    rmean = np.nanmean(m, axis=1, keepdims=True)
    m = np.where(np.isnan(m), np.where(np.isnan(rmean), 0, rmean), m)
    metric = "correlation" if np.all(np.nanstd(m, axis=1) > 0) else "euclidean"
    try:
        return leaves_list(linkage(m, method="average", metric=metric))
    except Exception:
        return np.arange(mat.shape[0])


def _nan_overlay(fig, mat, xs, ys, row=None, col=None):
    ii, jj = np.where(np.isnan(mat))
    if 0 < len(ii) <= 6000:
        tr = go.Scatter(x=[xs[j] for j in jj], y=[ys[i] for i in ii], mode="markers",
                        marker=dict(symbol="x-thin-open", size=5, color=C.NOT_DETECTED, line=dict(width=1, color=C.NOT_DETECTED)),
                        name="not detected", legendrank=1000, hoverinfo="skip", showlegend=True)
        if row:
            fig.add_trace(tr, row=row, col=col)
        else:
            fig.add_trace(tr)


def fig_heatmap(df: pd.DataFrame, value: str, value_label: str, diverging: bool,
                order: str = "species", cluster: bool = True, annotations: pd.DataFrame | None = None) -> go.Figure:
    """Genes x (dataset x cell type) heatmap with species track and group separators."""
    if df.empty:
        return empty("No data for this selection.")
    d = df.copy()
    d["s"] = d["species"].map({s: i for i, s in enumerate(C.SPECIES_ORDER)})
    d["c"] = d["cell_type"].map({c: i for i, c in enumerate(C.CELLTYPE_ORDER)}).fillna(99)
    d["col"] = list(zip(d["species"], d["sample_type"], d["dataset"], d["cell_type"]))
    keys = ["s", "sample_type", "dataset", "c"] if order == "species" else ["c", "s", "sample_type", "dataset"]
    cols = d.sort_values(keys)["col"].drop_duplicates().tolist()
    piv = d.pivot_table(index="gene", columns="col", values=value, aggfunc="mean", dropna=False)
    genes_in = d["gene"].drop_duplicates().tolist()
    piv = piv.reindex(index=genes_in, columns=cols)
    mat = piv.to_numpy(float)
    ridx = _cluster_rows(mat) if cluster else np.arange(len(genes_in))
    mat = mat[ridx]
    genes = [genes_in[i] for i in ridx]
    n_col = len(cols)
    xs = list(range(n_col))

    if diverging:
        lim = _diverging_range(mat)
        cs, zmin, zmax = C.DIVERGING, -lim, lim
    else:
        fin = mat[np.isfinite(mat)]
        cs = C.SEQUENTIAL
        zmin, zmax = (np.percentile(fin, 2), np.percentile(fin, 98)) if len(fin) else (0, 1)

    track_px, row_px = 30, (16 if len(genes) <= 60 else 11)
    heat_px = max(220, row_px * len(genes))
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.004,
                        row_heights=[track_px / (track_px + heat_px), heat_px / (track_px + heat_px)])
    lineages = [C.CELLTYPE_LINEAGE.get(c[3], "Other") for c in cols]
    _tracks(fig, [("species", [c[0] for c in cols], C.SPECIES_COLORS, C.SPECIES_ORDER),
                  ("lineage", lineages, C.LINEAGE_COLORS, list(C.LINEAGES))])
    _track_legend(fig, "Species", [s for s in C.SPECIES_ORDER if s in {c[0] for c in cols}], C.SPECIES_COLORS, 0)
    _track_legend(fig, "Lineage", [l for l in C.LINEAGES if l in set(lineages)], C.LINEAGE_COLORS, 100)
    hover =[[f"{c[0]} · {c[2]}{' (organoid)' if c[1] == 'organoid' else ''}<br>{c[3]}" for c in cols] for _ in genes]
    sfari = []
    for g in genes:
        s = ""
        if annotations is not None and g.upper() in annotations.index:
            a = annotations.loc[g.upper()]
            s = f"SFARI score {a.get('sfari_score'):g}" if pd.notna(a.get("sfari_score")) else "SFARI"
            if a.get("syndromic") == 1:
                s += " · syndromic"
        sfari.append(s)
    custom = [[f"{hover[i][j]}<br>{sfari[i]}" for j in range(n_col)] for i in range(len(genes))]
    fig.add_trace(go.Heatmap(
        z=mat, x=xs, y=genes, colorscale=cs, zmin=zmin, zmax=zmax, zmid=0 if diverging else None,
        customdata=custom, xgap=0, ygap=1,
        hovertemplate="<b>%{y}</b><br>%{customdata}<br>" + value_label + ": %{z:.2f}<extra></extra>",
        colorbar=dict(title=dict(text=value_label, side="right"), thickness=10, len=0.45, y=0.5,
                      outlinewidth=0, tickfont=dict(size=10))), row=2, col=1)
    _nan_overlay(fig, mat, xs, genes, row=2, col=1)

    # group separators and headers
    gkey = (lambda c: (c[0], c[1], c[2])) if order == "species" else (lambda c: c[3])
    bounds, labels, start = [], [], 0
    for j in range(1, n_col + 1):
        if j == n_col or gkey(cols[j]) != gkey(cols[start]):
            bounds.append((start, j - 1))
            c0 = cols[start]
            labels.append(ds_label(c0[0], c0[2], c0[1]) if order == "species" else _short(c0[3]))
            start = j
    for (a, b), lab in zip(bounds, labels):
        if b < n_col - 1:
            fig.add_vline(x=b + 0.5, line=dict(color="white", width=3), row=2, col=1)
            fig.add_vline(x=b + 0.5, line=dict(color="white", width=3), row=1, col=1)
        fig.add_annotation(x=(a + b) / 2, y=1.0, xref="x", yref="paper", yanchor="bottom",
                           text=lab, showarrow=False, textangle=-50 if n_col > 24 else 0,
                           font=dict(size=10, color=C.INK_SECONDARY), xanchor="left" if n_col > 24 else "center")
    tick = [_short(c[3]) for c in cols] if order == "species" else [ds_label(c[0], c[2], c[1]) for c in cols]
    readable = n_col <= 60
    fig.update_xaxes(tickvals=xs, ticktext=tick, tickangle=-60, tickfont=dict(size=9 if n_col > 40 else 10),
                     showticklabels=readable, showline=False, ticks="", row=2, col=1)
    fig.update_xaxes(showticklabels=False, showline=False, ticks="", row=1, col=1)
    fig.update_yaxes(showline=False, ticks="", showgrid=False, autorange="reversed",
                     tickfont=dict(size=9, color=C.INK_MUTED), row=1, col=1)
    # explicit ranges: the "not detected" overlay would otherwise pad the autorange
    fig.update_yaxes(range=[len(genes) - 0.5, -0.5], autorange=False, showgrid=False, showline=False, ticks="",
                     tickfont=dict(size=10 if len(genes) <= 60 else 8), row=2, col=1)
    fig.update_xaxes(range=[-0.5, n_col - 0.5], autorange=False, row=2, col=1)
    fig.update_layout(template=TEMPLATE, height=heat_px + track_px + (330 if readable else 230),
                      margin=dict(t=150 if n_col > 24 else 60, b=170 if readable else 70, l=80, r=40),
                      legend=dict(orientation="h", x=0, y=0, yanchor="bottom", yref="container",
                                  groupclick="toggleitem", tracegroupgap=20),
                      plot_bgcolor="white")
    return fig


def _tracks(fig, tracks: list, row: int = 1):
    """Categorical annotation rows drawn as ONE heatmap with a discrete colour scale.

    ``tracks``: list of (name, categories per column, palette, category order).
    (Several heatmaps sharing a categorical y axis break Plotly's calc step.)
    """
    bins = [(name, c, pal.get(c, C.INK_MUTED)) for name, _, pal, order in tracks for c in order]
    k = len(bins)
    code = {(name, c): i for i, (name, c, _) in enumerate(bins)}
    scale = []
    for i, (_, _, color) in enumerate(bins):
        scale += [[i / k, color], [(i + 1) / k, color]]
    z = [[(code.get((name, c), 0) + 0.5) / k for c in cats] for name, cats, _, _ in tracks]
    fig.add_trace(go.Heatmap(z=z, x=list(range(len(tracks[0][1]))), y=[t[0] for t in tracks],
                             colorscale=scale, zmin=0, zmax=1, showscale=False, xgap=0, ygap=1,
                             customdata=[list(cats) for _, cats, _, _ in tracks],
                             hovertemplate="%{y}: %{customdata}<extra></extra>"), row=row, col=1)


def _track_legend(fig, title: str, cats: list, palette: dict, rank: int):
    for i, c in enumerate(cats):
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=c, legendgroup=title,
                                 legendgrouptitle=dict(text=title) if i == 0 else None, legendrank=rank + i,
                                 marker=dict(symbol="square", size=10, color=palette.get(c, C.INK_MUTED))))


def fig_dotplot(prof: pd.DataFrame, value_label: str, diverging: bool) -> go.Figure:
    """Genes x cell types per species group: colour = value, area = detection rate."""
    if prof.empty:
        return empty("No data for this selection.")
    p = prof.copy()
    p["grp"] = [species_group(s, t) for s, t in zip(p["species"], p["sample_type"])]
    groups = _group_order(p["grp"])
    genes = p["gene"].drop_duplicates().tolist()
    widths = [max(1, p.loc[p["grp"] == g, "cell_type"].nunique()) for g in groups]
    fig = make_subplots(rows=1, cols=len(groups), shared_yaxes=True, horizontal_spacing=0.012,
                        column_widths=[w / sum(widths) for w in widths], subplot_titles=groups)
    vals = p["value"].to_numpy(float)
    if diverging:
        lim = _diverging_range(vals)
        cs, cmin, cmax = C.DIVERGING, -lim, lim
    else:
        fin = vals[np.isfinite(vals)]
        cs = C.SEQUENTIAL
        cmin, cmax = (np.percentile(fin, 2), np.percentile(fin, 98)) if len(fin) else (0, 1)
    for i, grp in enumerate(groups, start=1):
        sub = p[(p["grp"] == grp)]
        cts = _ct_order(sub["cell_type"])
        sub = sub.assign(x=sub["cell_type"].map(_short))
        has = sub[sub["value"].notna()]
        fig.add_trace(go.Scatter(
            x=has["x"], y=has["gene"], mode="markers", showlegend=False,
            marker=dict(size=3 + 15 * np.sqrt(has["detection"].clip(0, 1).fillna(0)), color=has["value"],
                        colorscale=cs, cmin=cmin, cmax=cmax, line=dict(width=0.5, color="white"),
                        showscale=(i == 1),
                        colorbar=dict(title=dict(text=value_label, side="right"), thickness=10, len=0.4,
                                      y=0.75, outlinewidth=0, tickfont=dict(size=10))),
            customdata=np.c_[has["cell_type"], has["detection"], has["n_datasets"], has["n_cells"]],
            hovertemplate=(f"<b>%{{y}}</b> · {grp}<br>%{{customdata[0]}}<br>{value_label}: %{{marker.color:.2f}}"
                           "<br>detection %{customdata[1]:.0%} · %{customdata[2]} dataset(s)"
                           "<br>%{customdata[3]:,} cells<extra></extra>")), row=1, col=i)
        nd = sub[sub["value"].isna()]
        if len(nd):
            fig.add_trace(go.Scatter(x=nd["x"], y=nd["gene"], mode="markers", showlegend=False,
                                     marker=dict(symbol="x-thin-open", size=5, color=C.NOT_DETECTED, line=dict(width=1, color=C.NOT_DETECTED)),
                                     hovertemplate="%{y} · %{x}<br>not detected<extra></extra>"), row=1, col=i)
        fig.update_xaxes(categoryorder="array", categoryarray=[_short(c) for c in cts], tickangle=-60,
                         tickfont=dict(size=10), row=1, col=i)
    for frac in (0.1, 0.5, 1.0):
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=f"{frac:.0%}",
                                 legendgroup="size", legendgrouptitle=dict(text="Detection rate"),
                                 marker=dict(size=3 + 15 * np.sqrt(frac), color=C.INK_MUTED)))
    for ann in list(fig.layout.annotations)[:len(groups)]:
        ann.font = dict(size=12, color=C.SPECIES_COLORS.get(ann.text.replace(" organoid", ""), C.INK))
    fig.update_yaxes(categoryorder="array", categoryarray=genes[::-1], showgrid=False,
                     tickfont=dict(size=10 if len(genes) <= 60 else 8))
    fig.update_xaxes(showgrid=False)
    fig.update_layout(template=TEMPLATE, height=240 + (18 if len(genes) <= 60 else 12) * len(genes),
                      legend=dict(orientation="h", x=0, y=0, yanchor="bottom", yref="container", itemsizing="trace"),
                      margin=dict(t=48, b=190))
    return fig


# ---------------------------------------------------------------------------
# Development (gene sets)
# ---------------------------------------------------------------------------

def fig_temporal_heatmap(tdf: pd.DataFrame, styles: dict, value: str, value_label: str,
                         diverging: bool, cluster: bool = True) -> go.Figure:
    """Genes x time points for one species group and cell type; dataset track on top."""
    if tdf.empty or tdf[value].notna().sum() == 0:
        return empty("No detected time-resolved values for this selection.")
    d = tdf.copy()
    d["col"] = list(zip(d["numeric_time"], d["dataset"]))
    cols = sorted(d["col"].drop_duplicates().tolist())
    genes_in = d["gene"].drop_duplicates().tolist()
    piv = d.pivot_table(index="gene", columns="col", values=value, aggfunc="mean", dropna=False).reindex(index=genes_in, columns=cols)
    mat = piv.to_numpy(float)
    ridx = _cluster_rows(mat) if cluster else np.arange(len(genes_in))
    mat, genes = mat[ridx], [genes_in[i] for i in ridx]
    xs = list(range(len(cols)))
    age = d.drop_duplicates("col").set_index("col")["age_label"]
    datasets = list(dict.fromkeys(c[1] for c in cols))
    ds_code = {ds: i for i, ds in enumerate(datasets)}
    ds_scale = []
    for i, ds in enumerate(datasets):
        col = styles.get(ds, {}).get("color", C.INK_MUTED)
        ds_scale += [[i / len(datasets), col], [(i + 1) / len(datasets), col]]
    if diverging:
        lim = _diverging_range(mat)
        cs, zmin, zmax = C.DIVERGING, -lim, lim
    else:
        fin = mat[np.isfinite(mat)]
        cs, (zmin, zmax) = C.SEQUENTIAL, ((np.percentile(fin, 2), np.percentile(fin, 98)) if len(fin) else (0, 1))
    heat_px = max(220, 16 * len(genes))
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.006,
                        row_heights=[14 / (14 + heat_px), heat_px / (14 + heat_px)])
    fig.add_trace(go.Heatmap(z=[[(ds_code[c[1]] + 0.5) / len(datasets) for c in cols]], x=xs, y=["dataset"],
                             colorscale=ds_scale, zmin=0, zmax=1, showscale=False,
                             customdata=[[f"{c[1]} · {age[c]}" for c in cols]],
                             hovertemplate="%{customdata}<extra></extra>"), row=1, col=1)
    custom = [[f"{c[1]} · {age[c]}" for c in cols] for _ in genes]
    fig.add_trace(go.Heatmap(z=mat, x=xs, y=genes, colorscale=cs, zmin=zmin, zmax=zmax,
                             zmid=0 if diverging else None, ygap=1, customdata=custom,
                             hovertemplate="<b>%{y}</b><br>%{customdata}<br>" + value_label + ": %{z:.2f}<extra></extra>",
                             colorbar=dict(title=dict(text=value_label, side="right"), thickness=10, len=0.45,
                                           outlinewidth=0)), row=2, col=1)
    _nan_overlay(fig, mat, xs, genes, row=2, col=1)
    for ds in datasets:
        st = styles.get(ds, {})
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=ds,
                                 marker=dict(symbol="square", size=10, color=st.get("color", C.INK_MUTED))))
    fig.update_xaxes(tickvals=xs, ticktext=[age[c] for c in cols], tickangle=-60, showline=False, ticks="",
                     tickfont=dict(size=9 if len(cols) > 30 else 10), row=2, col=1)
    fig.update_xaxes(showticklabels=False, showline=False, ticks="", row=1, col=1)
    fig.update_yaxes(showticklabels=False, showline=False, ticks="", showgrid=False, row=1, col=1)
    fig.update_yaxes(range=[len(genes) - 0.5, -0.5], autorange=False, showgrid=False, showline=False,
                     ticks="", row=2, col=1)
    fig.update_xaxes(range=[-0.5, len(cols) - 0.5], autorange=False, row=2, col=1)
    fig.update_layout(template=TEMPLATE, height=heat_px + 270, margin=dict(t=70, b=110),
                      legend=dict(orientation="h", x=0, y=1, yanchor="top", yref="container",
                                  title=dict(text="Dataset")))
    return fig


def fig_module_trajectory(tdf: pd.DataFrame, value: str, value_label: str, cell_types: list,
                          ncols: int = 3) -> go.Figure:
    """Gene-set module: median across genes per time point, IQR as error bars."""
    tdf = tdf[tdf["cell_type"].isin(cell_types)]
    if tdf.empty or tdf[value].notna().sum() == 0:
        return empty("No detected time-resolved values for this selection.")
    q = (tdf.groupby(["species", "sample_type", "dataset", "cell_type", "relative_dev_time", "age_label"], observed=True)
            [value].agg(med="median", q1=lambda s: s.quantile(0.25), q3=lambda s: s.quantile(0.75), n="count")
            .reset_index())
    n_cells = (tdf.drop_duplicates(["species", "sample_type", "dataset", "cell_type", "relative_dev_time"])
                  .set_index(["species", "sample_type", "dataset", "cell_type", "relative_dev_time"])["n_cells"])
    q["n_cells"] = n_cells.reindex(pd.MultiIndex.from_frame(q[["species", "sample_type", "dataset", "cell_type",
                                                               "relative_dev_time"]])).to_numpy()
    q = q[q["n"] > 0]
    cts = [c for c in _ct_order(q["cell_type"]) if c in cell_types]
    if not cts:
        return empty("No detected time-resolved values for the selected cell types.")
    ncols = min(ncols, len(cts))
    nrows = int(np.ceil(len(cts) / ncols))
    fig = make_subplots(rows=nrows, cols=ncols, shared_yaxes=True, shared_xaxes=True,
                        subplot_titles=[_short(c) for c in cts], horizontal_spacing=0.04,
                        vertical_spacing=0.12 if nrows > 1 else 0.05)
    _trajectory_panels(fig, q, "med", f"median {value_label}", cts, ncols)
    return _trajectory_layout(fig, cts, ncols, nrows, f"median {value_label}")


# ---------------------------------------------------------------------------
# Conservation
# ---------------------------------------------------------------------------

def fig_conservation_heatmap(cons: pd.DataFrame) -> go.Figure:
    if cons.empty or cons["rho"].notna().sum() == 0:
        return empty("Not enough shared cell types to compute correlations.")
    piv = cons.pivot_table(index="gene", columns="pair", values="rho", dropna=False)
    nmat = cons.pivot_table(index="gene", columns="pair", values="n", dropna=False).reindex_like(piv)
    order = piv.mean(axis=1).sort_values(ascending=False).index
    piv, nmat = piv.loc[order], nmat.loc[order]
    mat = piv.to_numpy(float)
    fig = go.Figure(go.Heatmap(
        z=mat, x=list(piv.columns), y=list(piv.index), colorscale=C.DIVERGING, zmin=-1, zmax=1, zmid=0,
        xgap=2, ygap=1, customdata=nmat.to_numpy(),
        hovertemplate="<b>%{y}</b> · %{x}<br>Spearman ρ = %{z:.2f}<br>n = %{customdata:.0f} shared cell types<extra></extra>",
        colorbar=dict(title=dict(text="Spearman ρ", side="right"), thickness=10, len=0.4, outlinewidth=0)))
    _nan_overlay(fig, mat, list(piv.columns), list(piv.index))
    fig.update_layout(template=TEMPLATE, height=160 + 16 * len(piv), margin=dict(t=30),
                      xaxis=dict(side="top", showline=False, ticks="", range=[-0.5, piv.shape[1] - 0.5]),
                      yaxis=dict(range=[len(piv) - 0.5, -0.5], showgrid=False, showline=False, ticks="",
                                 tickfont=dict(size=10 if len(piv) <= 60 else 8)),
                      legend=dict(orientation="h", x=0, y=-0.02, yanchor="top", yref="container"))
    return fig


def fig_conservation_summary(cons: pd.DataFrame, background: pd.DataFrame) -> tuple[go.Figure, pd.DataFrame]:
    """Gene-set ρ (points) against a background of all detected genes (box), per species pair."""
    pairs = [p for p in background["pair"].drop_duplicates() if p in set(cons["pair"])]
    fig = go.Figure()
    stats = []
    for p in pairs:
        bg = background.loc[background["pair"] == p, "rho"].dropna()
        gs = cons.loc[cons["pair"] == p, "rho"].dropna()
        fig.add_trace(go.Box(x=bg, y=[p] * len(bg), name="background (random detected genes)", orientation="h",
                             legendgroup="bg", showlegend=p == pairs[0], boxpoints=False,
                             marker_color=C.INK_MUTED, line=dict(width=1), fillcolor="rgba(137,135,129,0.15)"))
        fig.add_trace(go.Box(x=gs, y=[p] * len(gs), name="selected genes", orientation="h",
                             legendgroup="gs", showlegend=p == pairs[0], boxpoints="all", jitter=0.4, pointpos=0,
                             marker=dict(color=C.DATASET_SLOTS[0], size=5, opacity=0.8),
                             line=dict(width=0), fillcolor="rgba(0,0,0,0)",
                             hovertext=cons.loc[(cons["pair"] == p) & cons["rho"].notna(), "gene"],
                             hovertemplate="%{hovertext}: ρ = %{x:.2f}<extra></extra>"))
        if len(gs) >= 3 and len(bg) >= 10:
            u, pval = mannwhitneyu(gs, bg, alternative="two-sided")
            stats.append({"Species pair": p, "Selected genes (n)": len(gs), "Median ρ (selected)": gs.median(),
                          "Background genes (n)": len(bg), "Median ρ (background)": bg.median(),
                          "Mann–Whitney p": pval})
    fig.add_vline(x=0, line=dict(color=C.AXIS, width=1), layer="below")
    fig.update_layout(template=TEMPLATE, boxmode="group", height=120 + 70 * max(1, len(pairs)),
                      xaxis=dict(title="Spearman ρ between cell-type profiles", range=[-1.05, 1.05], showgrid=True,
                                 gridcolor=C.GRID),
                      yaxis=dict(autorange="reversed", showgrid=False),
                      legend=dict(orientation="h", y=1, yanchor="top", x=0, yref="container"),
                      margin=dict(t=64))
    return fig, pd.DataFrame(stats)


def fig_pair_scatter(prof: pd.DataFrame, gene: str, sp_a: str, sp_b: str) -> go.Figure:
    """Cell-type profile of one gene in two species (z-scored within species)."""
    p = prof[(prof["gene"] == gene) & prof["value"].notna()]
    wide = p.pivot_table(index="cell_type", columns="species", values="value")
    if sp_a not in wide or sp_b not in wide:
        return empty(f"{gene} is not detected in both species.")
    z = (wide - wide.mean()) / wide.std(ddof=1)
    xy = z[[sp_a, sp_b]].dropna()
    if len(xy) < 3:
        return empty(f"Fewer than three shared cell types with detected {gene}.")
    from scipy.stats import spearmanr
    rho, pval = spearmanr(xy[sp_a], xy[sp_b])
    lim = float(np.nanmax(np.abs(xy.to_numpy()))) * 1.15 + 0.2
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[-lim, lim], y=[-lim, lim], mode="lines", line=dict(color=C.AXIS, width=1),
                             hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(
        x=xy[sp_a], y=xy[sp_b], mode="markers+text",
        # label only cell types away from the centre; all are named on hover
        text=[_short(c) if max(abs(a), abs(b)) >= 0.8 else "" for c, a, b in zip(xy.index, xy[sp_a], xy[sp_b])],
        hovertext=[_short(c) for c in xy.index],
        textposition="top center", textfont=dict(size=10, color=C.INK_SECONDARY), showlegend=False,
        marker=dict(size=10, color=[C.LINEAGE_COLORS.get(C.CELLTYPE_LINEAGE.get(c, "Other")) for c in xy.index],
                    line=dict(color="white", width=1)),
        hovertemplate="%{hovertext}<br>" + sp_a + ": %{x:.2f}<br>" + sp_b + ": %{y:.2f}<extra></extra>"))
    for lin, col in C.LINEAGE_COLORS.items():
        if any(C.CELLTYPE_LINEAGE.get(c) == lin for c in xy.index):
            fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=lin,
                                     marker=dict(size=9, color=col)))
    fig.add_annotation(x=0.02, y=0.98, xref="paper", yref="paper", xanchor="left", yanchor="top", showarrow=False,
                       text=f"Spearman ρ = {rho:.2f} (p = {pval:.2g}, n = {len(xy)} cell types)",
                       font=dict(size=12, color=C.INK))
    fig.update_layout(template=TEMPLATE, height=480, title=f"{gene}: {sp_a} vs {sp_b}",
                      xaxis=dict(title=f"{sp_a} (z-score across cell types)", range=[-lim, lim], zeroline=False),
                      yaxis=dict(title=f"{sp_b} (z-score across cell types)", range=[-lim, lim],
                                 scaleanchor="x", scaleratio=1),
                      legend=dict(title=dict(text="Lineage"), x=1.01, y=1))
    return fig


# ---------------------------------------------------------------------------
# Cell atlas
# ---------------------------------------------------------------------------

def fig_umap(u: pd.DataFrame, color_by: str, highlight: str | None = None) -> go.Figure:
    if u is None or u.empty:
        return empty("No embedding in this data build.")
    fig = go.Figure()
    base = dict(mode="markers", hoverinfo="skip")
    if color_by in ("species", "sample_type"):
        # <= 4 categories: palettes validated for all pairs (scatter)
        palette = {"species": C.SPECIES_COLORS,
                   "sample_type": {"in_vivo": C.DATASET_SLOTS[0], "organoid": C.DATASET_SLOTS[1]}}[color_by]
        order = {"species": C.SPECIES_ORDER, "sample_type": ["in_vivo", "organoid"]}[color_by]
        cats = [c for c in order if c in set(u[color_by])]
        # draw the largest category first so rarer ones stay visible
        cats = sorted(cats, key=lambda c: -(u[color_by] == c).sum())
        for cat in cats:
            d = u[u[color_by] == cat]
            name = C.SAMPLE_TYPE_LABEL.get(cat, cat)
            fig.add_trace(go.Scattergl(x=d["umap_1"], y=d["umap_2"], name=f"{name} ({len(d):,})", **base,
                                       marker=dict(size=2.5, color=palette.get(cat, C.INK_MUTED), opacity=0.6)))
    else:
        sel = u[color_by] == highlight
        fig.add_trace(go.Scattergl(x=u.loc[~sel, "umap_1"], y=u.loc[~sel, "umap_2"], name="other cells", **base,
                                   marker=dict(size=2, color="#dcdbd5", opacity=0.6)))
        fig.add_trace(go.Scattergl(x=u.loc[sel, "umap_1"], y=u.loc[sel, "umap_2"],
                                   name=f"{highlight} ({int(sel.sum()):,})", **base,
                                   marker=dict(size=3, color=C.DATASET_SLOTS[0], opacity=0.8)))
    fig.update_layout(template=TEMPLATE, height=640,
                      xaxis=dict(title="UMAP 1", showticklabels=False, ticks="", showgrid=False),
                      yaxis=dict(title="UMAP 2", showticklabels=False, ticks="", showgrid=False,
                                 scaleanchor="x", scaleratio=1),
                      legend=dict(itemsizing="constant", x=1.01, y=1, font=dict(size=11)),
                      margin=dict(t=24))
    return fig
