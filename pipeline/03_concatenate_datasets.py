#!/usr/bin/env python3
"""Step 3: Concatenate (V4) - TRUE OUTER JOIN"""
import os, gc, json, re, numpy as np, pandas as pd, anndata as ad
from scipy import sparse, io as spio
import warnings; warnings.filterwarnings('ignore')
from config import *

STEP_NAME = "03_concatenate"

def strip_ver(s): return re.sub(r"\.\d+$", "", str(s))

def load_gene_map(name):
    p = f"{GENE_MAP_DIR}/{name}_gene_map.csv"
    if not os.path.exists(p): return None
    return dict(pd.read_csv(p).drop_duplicates(subset=["ensembl_gene_id"]).set_index("ensembl_gene_id")["symbol"])

def process_symbol(name, path, src, g2i, ng):
    print(f"\n  {name}...")
    adata = ad.read_h5ad(path)
    layer_override = DATASET_LAYER_OVERRIDE.get(name)
    needs_round = name in DATASETS_TO_ROUND or "_rounded" in src
    
    if layer_override and layer_override in adata.layers:
        X, vn = adata.layers[layer_override], adata.var_names.astype(str)
        print(f"    Using layer '{layer_override}'")
    elif "raw" in src and adata.raw: 
        X, vn = adata.raw.X, adata.raw.var_names.astype(str)
        print(f"    Using .raw.X")
    elif src.startswith("layers["): 
        ln = src.split("[")[1].split("]")[0]
        X, vn = adata.layers[ln], adata.var_names.astype(str)
        print(f"    Using layer '{ln}'")
    else: 
        X, vn = adata.X, adata.var_names.astype(str)
        print(f"    Using .X")
    
    if needs_round:
        print(f"    Rounding...")
        X = X.copy() if sparse.issparse(X) else X.copy()
        if sparse.issparse(X): X.data = np.round(X.data).astype(np.float32)
        else: X = np.round(X).astype(np.float32)
    
    seen, oi, ni = set(), [], []
    for i, g in enumerate(vn.values):
        if g not in seen: 
            seen.add(g)
            if g in g2i: oi.append(i); ni.append(g2i[g])
    
    Xo = X[:, oi]
    Xo = sparse.csr_matrix(Xo) if not sparse.issparse(Xo) else Xo.tocsr()
    Xc = Xo.tocoo(); nc = np.array(ni)[Xc.col]
    Xn = sparse.csr_matrix((Xc.data, (Xc.row, nc)), shape=(adata.n_obs, ng))
    
    obs = adata.obs.copy()
    m = DATASET_META.get(name, {"dataset": name, "organism": "?"})
    obs["dataset"], obs["organism"] = m["dataset"], m["organism"]
    obs.index = obs.index.astype(str) + f"-{name}"
    
    print(f"    {adata.n_obs:,} cells, {len(oi):,} genes mapped")
    del adata; gc.collect()
    return Xn, obs

def process_ensembl(name, path, src, g2i, ng):
    print(f"\n  {name} (Ensembl)...")
    gm = load_gene_map(name)
    if not gm: raise FileNotFoundError(f"Gene map missing for {name}")
    
    adata = ad.read_h5ad(path)
    if "raw" in src and adata.raw: 
        X, vn = adata.raw.X, adata.raw.var_names.astype(str)
        print(f"    Using .raw.X ({len(vn)} genes)")
    elif src.startswith("layers["): 
        ln = src.split("[")[1].split("]")[0]
        X, vn = adata.layers[ln], adata.var_names.astype(str)
        print(f"    Using layer '{ln}'")
    else: 
        X, vn = adata.X, adata.var_names.astype(str)
        print(f"    Using .X ({len(vn)} genes)")
    
    seen, oi, ni = set(), [], []
    for i, eid in enumerate(vn.values):
        sym = gm.get(strip_ver(eid))
        if sym and pd.notna(sym) and sym != '' and sym != 'nan' and sym not in seen:
            seen.add(sym)
            if sym in g2i: oi.append(i); ni.append(g2i[sym])
    
    Xo = X[:, oi]
    Xo = sparse.csr_matrix(Xo) if not sparse.issparse(Xo) else Xo.tocsr()
    Xc = Xo.tocoo(); nc = np.array(ni)[Xc.col]
    Xn = sparse.csr_matrix((Xc.data, (Xc.row, nc)), shape=(adata.n_obs, ng))
    
    obs = adata.obs.copy()
    m = DATASET_META.get(name, {"dataset": name, "organism": "?"})
    obs["dataset"], obs["organism"] = m["dataset"], m["organism"]
    obs.index = obs.index.astype(str) + f"-{name}"
    
    print(f"    {adata.n_obs:,} cells, {len(oi):,} genes mapped")
    del adata; gc.collect()
    return Xn, obs

def load_mtx(g2i, ng):
    print(f"\n{'='*60}\nLoading Combined MTX\n{'='*60}")
    feats = pd.read_csv(COMBINED_MTX["features_path"])
    gc_col = feats.columns[0]
    for c in feats.columns:
        if 'gene' in c.lower(): gc_col = c; break
    gn = feats[gc_col].astype(str).values
    print(f"  {len(gn):,} genes from column '{gc_col}'")
    
    meta = pd.read_csv(COMBINED_MTX["metadata_path"])
    ds_col = "dataset"  # Known from inspection
    print(f"  {len(meta):,} cells, dataset column: '{ds_col}'")
    print(f"  Datasets: {meta[ds_col].value_counts().to_dict()}")
    
    X = spio.mmread(COMBINED_MTX["mtx_path"]).T.tocsr()
    print(f"  Matrix shape: {X.shape}")
    
    seen, oi, ni = set(), [], []
    for i, g in enumerate(gn):
        if g not in seen: 
            seen.add(g)
            if g in g2i: oi.append(i); ni.append(g2i[g])
    print(f"  Mapped {len(oi):,} genes to filtered list")
    
    Xr = X[:, oi].tocoo()
    nc = np.array(ni)[Xr.col]
    Xm = sparse.csr_matrix((Xr.data, (Xr.row, nc)), shape=(X.shape[0], ng))
    del X; gc.collect()
    
    results = {}
    for mtx_name in meta[ds_col].unique():
        iname = MTX_DATASET_MAPPING.get(mtx_name, MTX_DATASET_MAPPING.get(str(mtx_name), str(mtx_name)))
        if iname not in MTX_DATASETS: 
            print(f"  Skipping unknown dataset: {mtx_name}")
            continue
        
        mask = meta[ds_col] == mtx_name
        cidx = np.where(mask)[0]
        Xd, od = Xm[cidx, :], meta.loc[mask].copy()
        
        m = DATASET_META.get(iname, {"dataset": iname, "organism": "?"})
        od["dataset"], od["organism"] = m["dataset"], m["organism"]
        
        # Find barcode column
        bc = None
        for c in ['barcode', 'cell', 'cell_id', 'CellID']:
            if c in od.columns: bc = c; break
        if bc:
            od.index = od[bc].astype(str) + f"-{iname}"
        else:
            od.index = [f"cell_{i}-{iname}" for i in range(len(od))]
        
        results[iname] = (Xd, od)
        print(f"  {iname}: {Xd.shape[0]:,} cells")
    
    del Xm; gc.collect()
    return results

def main():
    ensure_dirs()
    os.makedirs(TEMP_DIR, exist_ok=True)
    
    if checkpoint_exists(STEP_NAME): 
        print("Step done.")
        return
    
    print("="*60 + "\nSTEP 3: CONCATENATE DATASETS (V4)\n" + "="*60)
    
    with open(f"{OUTPUT_DIR}/filtered_genes.txt") as f: 
        genes = [l.strip() for l in f if l.strip()]
    g2i = {g: i for i, g in enumerate(genes)}
    ng = len(genes)
    print(f"Using {ng:,} genes (outer join)")
    
    with open(f"{OUTPUT_DIR}/dataset_inspection.json") as f: 
        insp = json.load(f)
    
    all_ds = []
    
    # Symbol datasets
    print("\n" + "-"*60 + "\nSYMBOL DATASETS\n" + "-"*60)
    for n, p in SYMBOL_DATASETS.items():
        r = insp.get(n, {})
        src = r.get("recommended_source")
        if "error" in r or not src: 
            print(f"\n  Skipping {n}: {r.get('error', 'no source')}")
            continue
        if checkpoint_exists(f"dataset_{n}"): 
            print(f"\n  {n}: Already done (checkpoint)")
            all_ds.append(n)
            continue
        try:
            Xn, obs = process_symbol(n, p, src, g2i, ng)
            sparse.save_npz(f"{TEMP_DIR}/{n}_X.npz", Xn)
            obs.to_parquet(f"{TEMP_DIR}/{n}_obs.parquet")
            mark_checkpoint(f"dataset_{n}")
            all_ds.append(n)
            del Xn, obs; gc.collect()
        except Exception as e: 
            print(f"    ERROR: {e}")
            import traceback; traceback.print_exc()
    
    # Ensembl datasets
    print("\n" + "-"*60 + "\nENSEMBL DATASETS\n" + "-"*60)
    for n, p in ENSEMBL_DATASETS.items():
        r = insp.get(n, {})
        src = r.get("recommended_source")
        if "error" in r or not src: 
            print(f"\n  Skipping {n}: {r.get('error', 'no source')}")
            continue
        if checkpoint_exists(f"dataset_{n}"): 
            print(f"\n  {n}: Already done (checkpoint)")
            all_ds.append(n)
            continue
        try:
            Xn, obs = process_ensembl(n, p, src, g2i, ng)
            sparse.save_npz(f"{TEMP_DIR}/{n}_X.npz", Xn)
            obs.to_parquet(f"{TEMP_DIR}/{n}_obs.parquet")
            mark_checkpoint(f"dataset_{n}")
            all_ds.append(n)
            del Xn, obs; gc.collect()
        except Exception as e: 
            print(f"    ERROR: {e}")
            import traceback; traceback.print_exc()
    
    # MTX datasets
    print("\n" + "-"*60 + "\nCOMBINED MTX DATASETS\n" + "-"*60)
    for n in MTX_DATASETS:
        if checkpoint_exists(f"dataset_{n}"): 
            print(f"\n  {n}: Already done (checkpoint)")
            all_ds.append(n)
    
    mtx_todo = [n for n in MTX_DATASETS if not checkpoint_exists(f"dataset_{n}")]
    if mtx_todo:
        try:
            res = load_mtx(g2i, ng)
            for n, (Xd, od) in res.items():
                sparse.save_npz(f"{TEMP_DIR}/{n}_X.npz", Xd)
                od.to_parquet(f"{TEMP_DIR}/{n}_obs.parquet")
                mark_checkpoint(f"dataset_{n}")
                if n not in all_ds: all_ds.append(n)
            del res; gc.collect()
        except Exception as e: 
            print(f"  MTX ERROR: {e}")
            import traceback; traceback.print_exc()
    
    # Stack all
    print("\n" + "="*60 + "\nSTACKING ALL DATASETS\n" + "="*60)
    Xb, oL = [], []
    for n in all_ds:
        xp, op = f"{TEMP_DIR}/{n}_X.npz", f"{TEMP_DIR}/{n}_obs.parquet"
        if os.path.exists(xp) and os.path.exists(op): 
            print(f"  Loading {n}...")
            Xb.append(sparse.load_npz(xp))
            oL.append(pd.read_parquet(op))
        else:
            print(f"  WARNING: Missing files for {n}")
    
    print(f"\n  Stacking {len(Xb)} matrices...")
    Xc = sparse.vstack(Xb, format="csr")
    del Xb; gc.collect()
    
    print("  Concatenating obs...")
    oc = pd.concat(oL, axis=0)
    del oL; gc.collect()
    
    print("\n  Creating AnnData...")
    adata = ad.AnnData(
        X=Xc, 
        obs=oc, 
        var=pd.DataFrame(index=pd.Index(genes, name="gene"))
    )
    print(f"  Shape: {adata.shape}")
    print(f"    Cells: {adata.n_obs:,}")
    print(f"    Genes: {adata.n_vars:,}")
    
    # Clean obs columns for h5ad
    print("\n  Cleaning obs columns...")
    for c in adata.obs.columns:
        if adata.obs[c].dtype == object: 
            adata.obs[c] = adata.obs[c].fillna('').astype(str)
        elif adata.obs[c].dtype == bool:
            adata.obs[c] = adata.obs[c].astype(str)
    
    print(f"\n  Saving to {CONCATENATED_PATH}...")
    adata.write(CONCATENATED_PATH)
    
    # Cleanup temp
    print("\n  Cleaning temp files...")
    for n in all_ds:
        for ext in ["_X.npz", "_obs.parquet"]:
            p = f"{TEMP_DIR}/{n}{ext}"
            if os.path.exists(p): os.remove(p)
    
    mark_checkpoint(STEP_NAME)
    
    print("\n" + "="*60)
    print("CONCATENATION COMPLETE")
    print("="*60)
    print(f"Output: {CONCATENATED_PATH}")
    print(f"Shape: {adata.shape}")
    
    print("\nCells per dataset:")
    for ds, c in adata.obs['dataset'].value_counts().items(): 
        print(f"  {ds}: {c:,}")
    print(f"\nTOTAL: {adata.n_obs:,} cells, {adata.n_vars:,} genes")

if __name__ == "__main__": main()
