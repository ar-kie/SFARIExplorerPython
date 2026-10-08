#!/usr/bin/env python3
"""Step 1: Inspect Datasets (V4)"""
import os, json, numpy as np, pandas as pd, scanpy as sc
from scipy import sparse, io as spio
import warnings; warnings.filterwarnings('ignore')
from config import *

STEP_NAME = "01_inspect_datasets"

def check_raw(X, n=1000):
    if X is None: return {"exists": False, "is_raw": False}
    idx = np.random.choice(X.shape[0], min(n, X.shape[0]), replace=False)
    Xs = X[idx, :].toarray() if sparse.issparse(X) else X[idx, :]
    Xnz = Xs.flatten(); Xnz = Xnz[Xnz != 0]
    if len(Xnz) == 0: return {"exists": True, "is_raw": False}
    is_int = np.allclose(Xnz, np.round(Xnz))
    is_roundable = np.all(np.abs(Xnz - np.round(Xnz)) < 0.01)
    has_neg = (Xnz < 0).any()
    is_raw = is_int and not has_neg and Xnz.max() > 10 and (Xnz < 1).sum()/len(Xnz) < 0.1
    return {"exists": True, "is_raw": bool(is_raw), "is_roundable": bool(is_roundable), "min": float(Xnz.min()), "max": float(Xnz.max()), "dtype": str(X.dtype)}

def inspect_h5ad(name, path):
    print(f"\n{'='*60}\n{name}\n{'='*60}")
    if not os.path.exists(path): return {"name": name, "error": "file not found"}
    try: adata = sc.read_h5ad(path)
    except Exception as e: return {"name": name, "error": str(e)}
    needs_round = name in DATASETS_TO_ROUND
    layer_override = DATASET_LAYER_OVERRIDE.get(name)
    r = {"name": name, "n_cells": adata.n_obs, "n_genes": adata.n_vars, "layers": list(adata.layers.keys()), "has_raw": adata.raw is not None, "needs_rounding": needs_round, "layer_override": layer_override}
    print(f"  Shape: {adata.shape}, Layers: {r['layers']}")
    if layer_override: print(f"  ** LAYER OVERRIDE: {layer_override} **")
    r["X_check"] = check_raw(adata.X)
    print(f"  .X: raw={r['X_check']['is_raw']}, roundable={r['X_check'].get('is_roundable')}")
    r["raw_X_check"] = check_raw(adata.raw.X) if adata.raw else {"exists": False}
    if r["raw_X_check"].get("exists"): print(f"  .raw.X: raw={r['raw_X_check']['is_raw']}")
    r["layer_checks"] = {}
    for ln in adata.layers:
        r["layer_checks"][ln] = check_raw(adata.layers[ln])
        print(f"  layer[{ln}]: raw={r['layer_checks'][ln]['is_raw']}, roundable={r['layer_checks'][ln].get('is_roundable')}, max={r['layer_checks'][ln].get('max','?')}")
    # Determine source
    if layer_override and layer_override in adata.layers:
        r["recommended_source"] = f"layers[{layer_override}]_rounded" if needs_round else f"layers[{layer_override}]"
    elif r["X_check"]["is_raw"]: r["recommended_source"] = "X"
    elif r["raw_X_check"].get("is_raw"): r["recommended_source"] = "raw.X"
    else:
        src = None
        for ln, chk in r["layer_checks"].items():
            if chk.get("is_raw"): src = f"layers[{ln}]"; break
        r["recommended_source"] = src
    print(f"  RECOMMENDED: {r['recommended_source']}")
    del adata; import gc; gc.collect()
    return r

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
    X = spio.mmread(COMBINED_MTX["mtx_path"]).T.tocsr()
    chk = check_raw(X, 5000)
    print(f"  Matrix: {X.shape}, raw={chk['is_raw']}")
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
