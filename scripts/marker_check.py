#!/usr/bin/env python3
"""Canonical marker genes per author label of one dataset, next to inhibitory and excitatory neurons of a
reference dataset: settles whether author labels or CONCORD predictions fit the expression.

Reads only the counts of these cells and four obs columns (pipeline_output/sfari_final.h5ad). For every author
label: cells, the share of cells with >= 1 count per marker, and the CONCORD prediction. Markers with zero
counts in every cell of the dataset are flagged as absent from its count matrix.

    python scripts/marker_check.py "Nascimento (2024) a" [--reference "Nascimento (2024) b"]
"""
import argparse, os, sys
import h5py, numpy as np, pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "pipeline"))
from h5io import read_elem, sparse_dataset  # noqa: E402

ROOT = os.environ.get("SFARI_ROOT", "/sc/arion/projects/ad-omics/raphael/SFARI")
FINAL = f"{ROOT}/pipeline_output/sfari_final.h5ad"
ANNOT = f"{ROOT}/data/concord/concord_cell_annotations.parquet"
GABA = ["GAD1", "GAD2", "SLC32A1", "DLX1", "DLX2", "DLX5", "DLX6", "LHX6", "ADARB2", "SST", "VIP", "PVALB"]
GLUT = ["SLC17A7", "SLC17A6", "NEUROD2", "NEUROD6", "TBR1", "SATB2", "EMX1"]
OTHER = ["DCX", "SOX2", "MKI67", "GFAP", "OLIG2"]
MAX_REF = 20_000


def rows_of(f, dataset):
    return np.flatnonzero(np.asarray(read_elem(f["obs/dataset"])).astype(str) == dataset)


def read_rows(X, rows):
    rows = np.sort(rows)
    return (X[rows[0]:rows[-1] + 1] if rows[-1] - rows[0] + 1 == len(rows) else X[rows]).tocsr()


def profile(X, rows, genes):
    """Per-gene share of cells with >= 1 count, median counts and genes per cell, share of non-integer values."""
    sub = read_rows(X, rows)
    det = (sub[:, genes] > 0).mean(axis=0).A1
    depth = np.median(np.asarray(sub.sum(axis=1)).ravel())
    ngenes = np.median(np.diff(sub.indptr))
    v = sub.data[:200_000]
    return det, {"median counts": depth, "median genes": ngenes,
                 "non-integer": float(np.mean(np.abs(v - np.round(v)) > 1e-6)) if len(v) else np.nan}


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("dataset")
    p.add_argument("--reference", default="Nascimento (2024) b")
    a = p.parse_args()

    f = h5py.File(FINAL, "r")
    var = read_elem(f["var"]).index.astype(str)
    markers = [g for g in GABA + GLUT + OTHER if g in var]
    gidx = var.get_indexer(markers)
    ids = np.asarray(read_elem(f["obs"][f["obs"].attrs["_index"]]), dtype=object)
    author = np.asarray(read_elem(f["obs/cell_type_merged"]), dtype=object)
    supcat = np.asarray(read_elem(f["obs/cell_type_supercategory"]), dtype=object)
    rows = rows_of(f, a.dataset)
    ref_rows = rows_of(f, a.reference)
    if not len(rows):
        sys.exit(f"No cells for {a.dataset!r}")
    pred = pd.Series(dtype=object)
    if os.path.exists(ANNOT):
        ann = pd.read_parquet(ANNOT, columns=["dataset", "concord_label"])
        pred = ann.loc[ann["dataset"].astype(str) == a.dataset, "concord_label"].astype(str)

    X = sparse_dataset(f["X"])
    tbl = []
    all_det, _ = profile(X, rows, gidx)
    for lab in pd.Series(author[rows]).value_counts().index[:12]:
        r = rows[author[rows] == lab]
        det, qc = profile(X, r, gidx)
        top_pred = pred.reindex(ids[r]).value_counts(normalize=True)
        tbl.append({"label": f"{str(lab)[:34]} ({supcat[r[0]]})", "cells": len(r), **dict(zip(markers, det)), **qc,
                    "CONCORD": ", ".join(f"{k} {v:.0%}" for k, v in top_pred.head(2).items())})
    rng = np.random.default_rng(0)
    for cls in ["Inhibitory Neurons", "Excitatory Neurons"]:
        r = ref_rows[supcat[ref_rows] == cls]
        if len(r):
            r = np.sort(rng.choice(r, min(len(r), MAX_REF), replace=False))
            det, qc = profile(X, r, gidx)
            tbl.append({"label": f"REFERENCE {a.reference}: {cls}", "cells": len(r),
                        **dict(zip(markers, det)), **qc, "CONCORD": ""})
    f.close()

    df = pd.DataFrame(tbl).set_index("label")
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
    print(f"{a.dataset}: {len(rows):,} cells. Share of cells with >= 1 count per marker\n")
    for group, genes in [("GABAergic", GABA), ("Glutamatergic", GLUT), ("Other", OTHER)]:
        cols = [g for g in genes if g in df.columns]
        print(f"--- {group} markers")
        print((100 * df[cols]).round(0).astype("Int64").astype(str).add(" %").to_string())
        print()
    print("--- Depth (raw counts expected: integer values) and CONCORD prediction per author label")
    out = df[["cells", "median counts", "median genes", "non-integer", "CONCORD"]].copy()
    out["non-integer"] = (100 * out["non-integer"]).round(1).astype(str) + " %"
    print(out.round(0).to_string())
    absent = [g for g, d in zip(markers, all_det) if d == 0]
    missing = [g for g in GABA + GLUT + OTHER if g not in var]
    if absent:
        print(f"\nZero counts in every cell of {a.dataset} (likely absent from its count matrix): {', '.join(absent)}")
    if missing:
        print(f"Not in the gene universe: {', '.join(missing)}")


if __name__ == "__main__":
    main()
