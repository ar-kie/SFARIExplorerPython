#!/usr/bin/env python3
"""Step 1: Inspect Datasets (V4)

Finds where each dataset keeps raw counts (X, .raw.X or a layer) from 1,000 random cells per matrix.
Only those cells are read from disk; files in the pre-0.7 h5ad format are loaded with anndata.
"""
import os, json, h5py, numpy as np, pandas as pd, scanpy as sc
from scipy import sparse
import warnings; warnings.filterwarnings('ignore')
from config import *
import h5io

STEP_NAME = "01_inspect_datasets"

def summarize(Xnz, dtype):
    """Raw-count test on the nonzero values of the sampled cells."""
    if len(Xnz) == 0: return {"exists": True, "is_raw": False}
    is_int = np.allclose(Xnz, np.round(Xnz))
    is_roundable = np.all(np.abs(Xnz - np.round(Xnz)) < 0.01)
    has_neg = (Xnz < 0).any()
    is_raw = is_int and not has_neg and Xnz.max() > 10 and (Xnz < 1).sum()/len(Xnz) < 0.1
    return {"exists": True, "is_raw": bool(is_raw), "is_roundable": bool(is_roundable), "min": float(Xnz.min()), "max": float(Xnz.max()), "dtype": dtype}

def check_raw(X, n=1000):
    """In-memory matrix (old-format files)."""
    if X is None: return {"exists": False, "is_raw": False}
    idx = np.random.choice(X.shape[0], min(n, X.shape[0]), replace=False)
    Xs = X[idx, :].toarray() if sparse.issparse(X) else X[idx, :]
    Xnz = Xs.flatten(); Xnz = Xnz[Xnz != 0]
    return summarize(Xnz, str(X.dtype))

def check_raw_h5(f, key, n=1000):
    """Matrix in an open h5ad; only the sampled cells are read."""
    if key not in f: return {"exists": False, "is_raw": False}
    return summarize(*h5io.sample_rows(f[key], n))

def checks_on_disk(f):
    layers = h5io.layer_names(f)
    has_raw = h5io.has_raw(f)
    return {"n_cells": h5io.n_obs(f), "n_genes": len(h5io.var_names(f)), "layers": layers, "has_raw": has_raw,
            "X_check": check_raw_h5(f, "X"),
            "raw_X_check": check_raw_h5(f, "raw/X") if has_raw else {"exists": False},
            "layer_checks": {ln: check_raw_h5(f, f"layers/{ln}") for ln in layers}}

def checks_in_memory(path):
    adata = sc.read_h5ad(path)
    r = {"n_cells": adata.n_obs, "n_genes": adata.n_vars, "layers": list(adata.layers.keys()), "has_raw": adata.raw is not None,
         "X_check": check_raw(adata.X),
         "raw_X_check": check_raw(adata.raw.X) if adata.raw else {"exists": False},
         "layer_checks": {ln: check_raw(adata.layers[ln]) for ln in adata.layers}}
    del adata; import gc; gc.collect()
    return r

def inspect_h5ad(name, path):
    print(f"\n{'='*60}\n{name}\n{'='*60}")
    if not os.path.exists(path): return {"name": name, "error": "file not found"}
    try:
        with h5py.File(path, "r") as f:
            legacy = h5io.is_legacy(f)
            if not legacy: c = checks_on_disk(f)
        if legacy:
            print("  (old h5ad format: loading with anndata)")
            c = checks_in_memory(path)
    except Exception as e: return {"name": name, "error": str(e)}
    needs_round = name in DATASETS_TO_ROUND
    layer_override = DATASET_LAYER_OVERRIDE.get(name)
    r = {"name": name, "n_cells": c["n_cells"], "n_genes": c["n_genes"], "layers": c["layers"], "has_raw": c["has_raw"], "needs_rounding": needs_round, "layer_override": layer_override}
    print(f"  Shape: ({r['n_cells']}, {r['n_genes']}), Layers: {r['layers']}")
    if layer_override: print(f"  ** LAYER OVERRIDE: {layer_override} **")
    r["X_check"] = c["X_check"]
    print(f"  .X: raw={r['X_check']['is_raw']}, roundable={r['X_check'].get('is_roundable')}")
    r["raw_X_check"] = c["raw_X_check"]
    if r["raw_X_check"].get("exists"): print(f"  .raw.X: raw={r['raw_X_check']['is_raw']}")
    r["layer_checks"] = c["layer_checks"]
    for ln, chk in r["layer_checks"].items():
        print(f"  layer[{ln}]: raw={chk['is_raw']}, roundable={chk.get('is_roundable')}, max={chk.get('max','?')}")
    # Determine source
    if layer_override and layer_override in r["layers"]:
        r["recommended_source"] = f"layers[{layer_override}]_rounded" if needs_round else f"layers[{layer_override}]"
    elif r["X_check"]["is_raw"]: r["recommended_source"] = "X"
    elif r["raw_X_check"].get("is_raw"): r["recommended_source"] = "raw.X"
    else:
        src = None
        for ln, chk in r["layer_checks"].items():
            if chk.get("is_raw"): src = f"layers[{ln}]"; break
        r["recommended_source"] = src
    print(f"  RECOMMENDED: {r['recommended_source']}")
    return r

def sample_mtx(path, n_blocks=64, block_bytes=1 << 20):
    """Header and a sample of values of a Matrix Market file: n_blocks reads spread over the file."""
    size = os.path.getsize(path)
    with open(path, "rb") as fh:
        line = fh.readline()
        field = line.split()[3].decode().lower() if line.startswith(b"%%MatrixMarket") else "real"
        while line.startswith(b"%"): line = fh.readline()
        n_rows, n_cols, nnz = map(int, line.split())
        body = fh.tell()
        vals = []
        for off in np.linspace(body, size, n_blocks, endpoint=False).astype(np.int64):
            fh.seek(off)
            lines = fh.read(block_bytes).split(b"\n")
            lines = lines[(0 if off == body else 1):-1]  # drop partial lines at both ends
            vals.extend(float(l.split()[2]) for l in lines if l.strip())
    vals = np.asarray(vals)
    return (n_rows, n_cols, nnz), vals[vals != 0], "int64" if field == "integer" else "float64"

def inspect_mtx():
    print(f"\n{'='*60}\nCombined MTX\n{'='*60}")
    for p in [COMBINED_MTX["mtx_path"], COMBINED_MTX["features_path"], COMBINED_MTX["metadata_path"]]:
        if not os.path.exists(p): return {"error": f"not found: {p}"}
    feats = pd.read_csv(COMBINED_MTX["features_path"])
    meta = pd.read_csv(COMBINED_MTX["metadata_path"])
    print(f"  Genes: {len(feats):,}, Cells: {len(meta):,}")
    ds_col = None
    for c in meta.columns:
        if 'dataset' in c.lower(): ds_col = c; break
    if ds_col: print(f"  Dataset col: {ds_col}, Values: {meta[ds_col].value_counts().to_dict()}")
    (n_genes, n_cells, nnz), vals, dtype = sample_mtx(COMBINED_MTX["mtx_path"])
    chk = summarize(vals, dtype)
    print(f"  Matrix: ({n_cells}, {n_genes}), {nnz:,} non-zeros, raw={chk['is_raw']} ({len(vals):,} sampled values)")
    return {"name": "Combined_MTX", "n_genes": len(feats), "n_cells": len(meta), "dataset_column": ds_col, "matrix_check": chk, "recommended_source": "MTX" if chk["is_raw"] else None}

def main():
    ensure_dirs()
    if checkpoint_exists(STEP_NAME): print("Step done."); return
    np.random.seed(42); results = {}
    for n, p in {**SYMBOL_DATASETS, **ENSEMBL_DATASETS}.items(): results[n] = inspect_h5ad(n, p)
    results["Combined_MTX"] = inspect_mtx()
    if results["Combined_MTX"].get("recommended_source") == "MTX":
        for n in MTX_DATASETS: results[n] = {"name": n, "source": "Combined_MTX", "recommended_source": "MTX"}
    with open(f"{OUTPUT_DIR}/dataset_inspection.json", 'w') as f: json.dump(results, f, indent=2, default=str)
    mark_checkpoint(STEP_NAME); print("✓ Done")

if __name__ == "__main__": main()
