"""Data access for the wide-format parquet build.

Expected files in ``data/``:
  expression_meta.parquet   one row per species x dataset x cell type x sample type
  expression_mean.parquet   same rows x genes, mean pseudobulk log2 CPM
  expression_pct.parquet    same rows x genes, fraction of pseudobulk replicates
                            with >= 1 count (detection rate)
  temporal_meta.parquet     one row per group x time point
  temporal_mean.parquet     same rows x genes, mean pseudobulk log2 CPM
  plus dataset_overview, risk_genes, stage_mapping, summary_statistics,
  umap_subsample (optional) and build_info.json (optional, written by the
  CONCORD pipeline).

All public functions return tidy (long) DataFrames so that every plot and the
table view are built from the same rows.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import spearmanr

from . import config as C


@dataclass
class Atlas:
    groups: pd.DataFrame
    mean: np.ndarray
    detection: np.ndarray
    tgroups: pd.DataFrame
    tmean: np.ndarray
    genes: list
    gene_index: dict
    annotations: pd.DataFrame
    datasets: pd.DataFrame
    umap: Optional[pd.DataFrame]
    stage_mapping: Optional[pd.DataFrame]
    summary: dict
    build_info: dict
    dataset_index: dict      # dataset -> row of ``absent``
    absent: np.ndarray       # datasets x genes; True = zero counts in every group of the dataset

    def absent_in(self, gene: str) -> list:
        """Datasets in which a gene has zero counts in every group (likely not in that dataset's matrix)."""
        j = self.gene_index.get(gene.upper())
        if j is None:
            return []
        return [ds for ds, i in self.dataset_index.items() if self.absent[i, j]]

    def resolve(self, genes: Iterable[str]) -> tuple[list, list]:
        """Split user input into (found canonical symbols, missing inputs), order kept."""
        found, missing, seen = [], [], set()
        for g in genes:
            j = self.gene_index.get(g.upper())
            if j is None:
                missing.append(g)
            elif j not in seen:
                seen.add(j)
                found.append(self.genes[j])
        return found, missing


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _matrix(table, genes: list) -> np.ndarray:
    """Columns of an Arrow table as a (rows x genes) float32 matrix; absent genes -> NaN."""
    names = set(table.column_names)
    n = table.num_rows
    cols = [table.column(g).to_numpy(zero_copy_only=False).astype(np.float32, copy=False)
            if g in names else np.full(n, np.nan, np.float32) for g in genes]
    return np.ascontiguousarray(np.vstack(cols).T)


def _align(table, order: list) -> np.ndarray:
    pos = {r: i for i, r in enumerate(table.column("row_id").to_pylist())}
    return np.array([pos[r] for r in order])


def _celltype_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={"tissue": "dataset"}).copy()
    for col in ("species", "dataset", "cell_type", "sample_type"):
        df[col] = df[col].astype(str)
    df["lineage"] = df["cell_type"].map(C.CELLTYPE_LINEAGE).fillna("Other")
    df["caveat"] = [C.LABEL_CAVEATS.get((s, c), "") for s, c in zip(df["species"], df["cell_type"])]
    return df


def load_atlas(data_dir: str | Path = "data") -> Atlas:
    d = Path(data_dir)

    groups = _celltype_frame(pd.read_parquet(d / "expression_meta.parquet"))
    mean_tbl = pq.read_table(d / "expression_mean.parquet")
    genes = [c for c in mean_tbl.column_names if c != "row_id"]
    mean = _matrix(mean_tbl, genes)[_align(mean_tbl, groups["row_id"].tolist())]
    del mean_tbl

    pct_cols = set(pq.read_schema(d / "expression_pct.parquet").names)
    pct_tbl = pq.read_table(d / "expression_pct.parquet",
                            columns=["row_id"] + [g for g in genes if g in pct_cols])
    detection = _matrix(pct_tbl, genes)[_align(pct_tbl, groups["row_id"].tolist())]
    del pct_tbl

    tgroups = _celltype_frame(pd.read_parquet(d / "temporal_meta.parquet"))
    t_tbl = pq.read_table(d / "temporal_mean.parquet")
    tmean = _matrix(t_tbl, genes)[_align(t_tbl, tgroups["row_id"].tolist())]
    del t_tbl
    key = ["species", "dataset", "cell_type", "sample_type"]
    gpos = {tuple(r): i for i, r in enumerate(groups[key].itertuples(index=False, name=None))}
    tgroups["group_idx"] = [gpos.get(tuple(r), -1) for r in tgroups[key].itertuples(index=False, name=None)]
    tgroups["age_label"] = [format_age(s, st, t) for s, st, t in
                            zip(tgroups["species"], tgroups["sample_type"], tgroups["numeric_time"])]

    gene_index = {}
    for j, g in enumerate(genes):
        gene_index.setdefault(g.upper(), j)

    # Genes with zero counts in every group of a dataset were most likely not part of that
    # dataset's count matrix (feature set / symbol mapping), rather than biologically silent.
    dataset_index = {ds: i for i, ds in enumerate(sorted(groups["dataset"].unique()))}
    absent = np.zeros((len(dataset_index), len(genes)), bool)
    for ds, i in dataset_index.items():
        absent[i] = (detection[(groups["dataset"] == ds).to_numpy()] == 0).all(axis=0)

    annotations = _load_annotations(d / "risk_genes.parquet")
    datasets = _dataset_table(d / "dataset_overview.parquet", tgroups, _references(d))
    umap = _optional(d / "umap_subsample.parquet")
    if umap is not None:
        umap = umap.rename(columns={"organism": "species", "predicted_labels": "cell_type"})
        for col in ("species", "dataset", "cell_type", "sample_type"):
            if col in umap.columns:
                umap[col] = umap[col].astype(str)
        if "cell_type" in umap.columns:
            umap["lineage"] = umap["cell_type"].map(C.CELLTYPE_LINEAGE).fillna("Other")

    summary = {}
    stats = _optional(d / "summary_statistics.parquet")
    if stats is not None and len(stats):
        summary = stats.iloc[0].to_dict()
    build_info = {}
    if (d / "build_info.json").exists():
        build_info = json.loads((d / "build_info.json").read_text())

    return Atlas(groups=groups, mean=mean, detection=detection, tgroups=tgroups, tmean=tmean,
                 genes=genes, gene_index=gene_index, annotations=annotations, datasets=datasets,
                 umap=umap, stage_mapping=_optional(d / "stage_mapping.parquet"),
                 summary=summary, build_info=build_info, dataset_index=dataset_index, absent=absent)


def _optional(path: Path) -> Optional[pd.DataFrame]:
    return pd.read_parquet(path) if path.exists() else None


def _load_annotations(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["gene", "sfari_score", "syndromic"]).set_index("gene")
    df = pd.read_parquet(path).rename(columns={
        "gene-symbol": "gene_symbol", "gene-score": "gene_score", "gene-name": "gene_name",
        "ensembl-id": "ensembl_id", "genetic-category": "genetic_category",
        "number-of-reports": "n_reports"})
    df["gene"] = df["gene_symbol"].astype(str).str.upper()
    df = df.rename(columns={"gene_score": "sfari_score"})
    return df.drop_duplicates("gene").set_index("gene")


def _references(d: Path) -> dict:
    """Built-in references plus data/dataset_references.json (written by the CELLxGENE fetcher)."""
    refs = dict(C.DATASET_REFS)
    extra = d / "dataset_references.json"
    if extra.exists():
        for name, r in json.loads(extra.read_text()).items():
            refs.setdefault(name, (r.get("reference", ""), r.get("doi", ""), r.get("scope", "")))
    return refs


def _dataset_table(path: Path, tgroups: pd.DataFrame, refs: dict) -> pd.DataFrame:
    ov = pd.read_parquet(path)
    rows = []
    for _, r in ov.iterrows():
        ds = r["Dataset"]
        ref, doi, scope = refs.get(ds, ("", "", ""))
        t = tgroups[tgroups["dataset"] == ds]
        age = ""
        if len(t) and t["numeric_time"].notna().any():
            st = t["sample_type"].iloc[0]
            lo, hi = t["numeric_time"].min(), t["numeric_time"].max()
            age = f"{format_age(r['Species'], st, lo)} – {format_age(r['Species'], st, hi)}"
        rows.append({
            "Dataset": ds, "Species": r["Species"],
            "Sample type": C.SAMPLE_TYPE_LABEL.get(r["Sample Type"], r["Sample Type"]),
            "Scope": scope, "Age range": age or "not annotated",
            "Cells": int(r["Total Cells"]), "Donors / samples": int(r["Samples"]),
            "Cell types": int(r["Cell Types"]), "Time points": int(r["Timepoints"]),
            "Reference": ref, "DOI": f"https://doi.org/{doi}" if doi else "",
        })
    df = pd.DataFrame(rows)
    df["_s"] = df["Species"].map({s: i for i, s in enumerate(C.SPECIES_ORDER)})
    return df.sort_values(["_s", "Sample type", "Dataset"]).drop(columns="_s").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Developmental age formatting
# ---------------------------------------------------------------------------

def format_age(species: str, sample_type: str, t: float) -> str:
    """Human-readable native age for a numeric time value of the data build."""
    if t is None or (isinstance(t, float) and np.isnan(t)):
        return "n/a"
    if sample_type == "organoid":
        return f"day {t:.0f}"
    if species == "Human":
        if t < 280:
            return f"{t / 7:.1f} pcw"
        y = (t - 280) / 365
        return f"{(t - 280) / 30.4:.0f} mo" if y < 2 else f"{y:.0f} y"
    if species == "Mouse":
        if t <= 20:
            return f"E{t:g}"
        p = t - 20
        return f"P{p:.0f}" if p < 60 else f"{p / 30.4:.0f} mo"
    if species == "Zebrafish":
        return f"{t:g} hpf" if t < 72 else f"{t / 24:g} dpf"
    if species == "Drosophila":
        return f"day {t:g}"
    return f"{t:g}"


# ---------------------------------------------------------------------------
# Tidy extraction
# ---------------------------------------------------------------------------

def _row_filter(df: pd.DataFrame, species, sample_types, datasets, min_cells) -> np.ndarray:
    m = np.ones(len(df), bool)
    if species:
        m &= df["species"].isin(species).to_numpy()
    if sample_types:
        m &= df["sample_type"].isin(sample_types).to_numpy()
    if datasets:
        m &= df["dataset"].isin(datasets).to_numpy()
    if min_cells:
        m &= (df["n_cells"] >= min_cells).to_numpy()
    return np.flatnonzero(m)


def _long(atlas: Atlas, meta: pd.DataFrame, rows: np.ndarray, gene_cols: list, genes: list,
          values: np.ndarray, detection: np.ndarray, min_detection: Optional[float]) -> pd.DataFrame:
    n_r, n_g = len(rows), len(gene_cols)
    base = meta.iloc[rows].reset_index(drop=True)
    out = base.loc[np.tile(np.arange(n_r), n_g)].reset_index(drop=True)
    out.insert(0, "gene", np.repeat(np.array(genes, dtype=object), n_r))
    v = values.T.ravel()
    det = detection.T.ravel()
    detected = np.isnan(det) | (det > (min_detection if min_detection is not None else -1))
    ds_rows = base["dataset"].map(atlas.dataset_index).to_numpy()
    out["log2cpm_unmasked"] = v
    out["detection"] = det
    out["detected"] = detected
    out["measured"] = ~atlas.absent[np.ix_(ds_rows, gene_cols)].T.ravel()
    out["log2cpm"] = np.where(detected, v, np.nan) if min_detection is not None else v
    return out


def celltype_long(atlas: Atlas, genes: list, species=None, sample_types=None, datasets=None,
                  min_cells: int = 0, min_detection: Optional[float] = 0.0) -> pd.DataFrame:
    """Gene x (species, dataset, cell type, sample type) rows.

    ``min_detection``: groups whose detection rate is <= this value are marked
    not detected and their log2 CPM is set to NaN (pass None to disable).
    """
    cols = [atlas.gene_index[g.upper()] for g in genes if g.upper() in atlas.gene_index]
    if not cols:
        return pd.DataFrame()
    names = [atlas.genes[j] for j in cols]
    rows = _row_filter(atlas.groups, species, sample_types, datasets, min_cells)
    if not len(rows):
        return pd.DataFrame()
    return _long(atlas, atlas.groups, rows, cols, names,
                 atlas.mean[np.ix_(rows, cols)], atlas.detection[np.ix_(rows, cols)], min_detection)


def temporal_long(atlas: Atlas, genes: list, species=None, sample_types=None, datasets=None,
                  min_cells: int = 0, min_detection: Optional[float] = 0.0) -> pd.DataFrame:
    """Gene x (group, time point) rows. Detection is taken from the parent
    dataset x cell-type group (time-resolved detection is not in the build)."""
    cols = [atlas.gene_index[g.upper()] for g in genes if g.upper() in atlas.gene_index]
    if not cols:
        return pd.DataFrame()
    names = [atlas.genes[j] for j in cols]
    tg = atlas.tgroups
    rows = _row_filter(tg, species, sample_types, datasets, min_cells)
    rows = rows[tg["numeric_time"].to_numpy()[rows] == tg["numeric_time"].to_numpy()[rows]]  # drop NaN time
    if not len(rows):
        return pd.DataFrame()
    gidx = tg["group_idx"].to_numpy()[rows]
    det = np.where(gidx[:, None] >= 0, atlas.detection[np.ix_(np.maximum(gidx, 0), cols)], np.nan)
    out = _long(atlas, tg, rows, cols, names, atlas.tmean[np.ix_(rows, cols)], det, min_detection)
    return out.sort_values(["gene", "species", "dataset", "cell_type", "numeric_time"]).reset_index(drop=True)


def add_scaled(df: pd.DataFrame, how: str, within: list, value: str = "log2cpm",
               out: str = "scaled") -> pd.DataFrame:
    """Center (Δ log2 CPM) or z-score ``value`` within groups of ``within`` columns."""
    df = df.copy()
    g = df.groupby(within, observed=True)[value]
    mu = g.transform("mean")
    if how == "center":
        df[out] = df[value] - mu
    elif how == "zscore":
        sd = g.transform("std")
        df[out] = (df[value] - mu) / sd.where(sd > 0)
    else:
        df[out] = df[value]
    return df


def species_profiles(df: pd.DataFrame, value: str = "log2cpm") -> pd.DataFrame:
    """Average detected values over datasets within species x sample type x cell type.

    Datasets are averaged with equal weight (each is an independent study);
    batch effects between datasets were removed within species upstream.
    """
    agg = (df.groupby(["gene", "species", "sample_type", "cell_type"], observed=True)
             .agg(value=(value, "mean"), detection=("detection", "mean"),
                  n_datasets=(value, "count"), n_cells=("n_cells", "sum"))
             .reset_index())
    return agg


def conservation(profiles: pd.DataFrame, pairs: list, min_shared: int = 4) -> pd.DataFrame:
    """Spearman correlation of each gene's cell-type profile between species pairs."""
    rows = []
    wide = profiles.pivot_table(index=["gene", "cell_type"], columns="species", values="value")
    for gene, sub in wide.groupby(level=0):
        for a, b in pairs:
            if a not in sub.columns or b not in sub.columns:
                continue
            xy = sub[[a, b]].dropna()
            n = len(xy)
            if n < min_shared or xy[a].nunique() < 2 or xy[b].nunique() < 2:
                rows.append({"gene": gene, "pair": f"{a}–{b}", "rho": np.nan, "p": np.nan, "n": n})
                continue
            rho, p = spearmanr(xy[a], xy[b])
            rows.append({"gene": gene, "pair": f"{a}–{b}", "rho": rho, "p": p, "n": n})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Gene input & gene sets
# ---------------------------------------------------------------------------

def parse_gene_text(text: str) -> list:
    if not text:
        return []
    return [t for t in re.split(r"[\s,;]+", text.strip()) if t]


def gene_sets(atlas: Atlas) -> dict:
    ann = atlas.annotations
    sets = {"Example: high-confidence ASD genes": C.DEFAULT_GENES}
    if "sfari_score" in ann.columns:
        for s in (1, 2, 3):
            sets[f"SFARI Gene score {s}"] = ann.index[ann["sfari_score"] == s].tolist()
    if "syndromic" in ann.columns:
        sets["SFARI Gene syndromic"] = ann.index[ann["syndromic"] == 1].tolist()
    sets["Canonical cell-type markers"] = [g for gs in C.MARKER_GENES.values() for g in gs]
    return {k: atlas.resolve(v)[0] for k, v in sets.items()}


def gene_annotation(atlas: Atlas, gene: str) -> dict:
    ann = atlas.annotations
    if gene.upper() not in ann.index:
        return {}
    r = ann.loc[gene.upper()]
    return {k: r.get(k) for k in ("gene_name", "sfari_score", "syndromic", "genetic_category",
                                   "n_reports", "eagle", "chromosome", "ensembl_id")}
