#!/usr/bin/env python3
"""Step 2: Build Gene Universe (V4) - TRUE OUTER JOIN with minimal filtering

Reads only gene names and cell counts (.var / .raw.var and the obs index), never the matrices.
"""
import os, re, json, h5py, numpy as np, pandas as pd, anndata as ad
from collections import defaultdict
import warnings; warnings.filterwarnings('ignore')
from config import *
import h5io

STEP_NAME = "02_build_gene_universe"

def load_gene_map(name):
    p = f"{GENE_MAP_DIR}/{name}_gene_map.csv"
    if not os.path.exists(p): return None
    df = pd.read_csv(p).drop_duplicates(subset=["ensembl_gene_id"])
    return dict(zip(df["ensembl_gene_id"], df["symbol"]))

def strip_ver(s): return re.sub(r"\.\d+$", "", str(s))

def read_names(path, src, layer_override=None):
    """(gene names, n_cells, used_raw). Picks .var or .raw.var as step 03 picks the matrix: an existing
    override layer uses .var, otherwise .raw when the source mentions raw and the file has it."""
    with h5py.File(path, "r") as f:
        if not h5io.is_legacy(f):
            use_raw = (not (layer_override and layer_override in h5io.layer_names(f))
                       and "raw" in src and h5io.has_raw(f))
            return h5io.var_names(f, raw=use_raw), h5io.n_obs(f), use_raw
    adata = ad.read_h5ad(path, backed='r')
    use_raw = not (layer_override and layer_override in adata.layers) and "raw" in src and adata.raw is not None
    vn = (adata.raw.var_names if use_raw else adata.var_names).astype(str)
    n = adata.n_obs; adata.file.close()
    return vn, n, use_raw

def get_genes_symbol(path, name, src):
    print(f"  {name} (Symbol)...")
    vn, n, _ = read_names(path, src, DATASET_LAYER_OVERRIDE.get(name))
    seen, out = set(), []
    for g in vn:
        if g not in seen: seen.add(g); out.append(g)
    print(f"    {n:,} cells, {len(out):,} genes")
    return out, n

def get_genes_ensembl(path, name, src):
    print(f"  {name} (Ensembl -> Symbol)...")
    gm = load_gene_map(name)
    if not gm: 
        print(f"    WARNING: No gene map found!")
        return [], 0
    vn, n, used_raw = read_names(path, src)
    if used_raw:
        print(f"    Using .raw.var_names ({len(vn)} genes)")
    else:
        print(f"    Using .var_names ({len(vn)} genes)")
    seen, out, mapped, unmapped = set(), [], 0, 0
    for eid in vn:
        sym = gm.get(strip_ver(eid))
        if sym and pd.notna(sym) and sym != '' and sym != 'nan':
            if sym not in seen: seen.add(sym); out.append(sym)
            mapped += 1
        else: 
            unmapped += 1
    print(f"    {n:,} cells, mapped={mapped:,}, unmapped={unmapped:,}, unique symbols={len(out):,}")
    return out, n

def get_genes_mtx():
    print(f"  Combined MTX...")
    if not os.path.exists(COMBINED_MTX["features_path"]): return [], {}
    df = pd.read_csv(COMBINED_MTX["features_path"])
    gc = df.columns[0]
    for c in df.columns:
        if 'gene' in c.lower() or 'symbol' in c.lower(): gc = c; break
    seen, out = set(), []
    for g in df[gc].astype(str):
        if g not in seen: seen.add(g); out.append(g)
    cc = {}
    if os.path.exists(COMBINED_MTX["metadata_path"]):
        m = pd.read_csv(COMBINED_MTX["metadata_path"])
        for c in m.columns:
            if 'dataset' in c.lower(): cc = m[c].value_counts().to_dict(); break
    print(f"    {len(out):,} genes from column '{gc}'")
    return out, cc

def is_ensembl(g): return bool(re.match(r'^ENS[A-Z]*G\d+', g))
def is_flybase(g): return bool(re.match(r'^FBgn\d+', g))

def should_exclude(g):
    """Minimal filtering: only unmapped IDs and ERCC spike-ins"""
    for p in GENE_EXCLUDE_PREFIXES:
        if g.startswith(p): return True
    for p in GENE_EXCLUDE_PATTERNS:
        if re.match(p, g): return True
    return False

def main():
    ensure_dirs()
    if checkpoint_exists(STEP_NAME): print("Step done."); return
    print("="*60 + "\nSTEP 2: BUILD GENE UNIVERSE (V4 - TRUE OUTER JOIN)\n" + "="*60)
    print(f"MIN_DATASETS_PER_GENE = {MIN_DATASETS_PER_GENE}")
    print(f"Filtering: Only unmapped Ensembl/FlyBase IDs + ERCC spike-ins\n")
    
    with open(f"{OUTPUT_DIR}/dataset_inspection.json") as f: insp = json.load(f)
    all_g, g2d, stats = set(), defaultdict(set), {}
    
    # Symbol datasets
    print("\n" + "-"*60 + "\nSYMBOL DATASETS\n" + "-"*60)
    for n, p in SYMBOL_DATASETS.items():
        r = insp.get(n, {})
        if "error" in r or not r.get("recommended_source"): 
            print(f"  Skipping {n}: {r.get('error', 'no source')}")
            continue
        gs, nc = get_genes_symbol(p, n, r["recommended_source"])
        stats[n] = {"n_cells": nc, "n_genes": len(gs)}
        for g in gs: all_g.add(g); g2d[g].add(n)
    
    # Ensembl datasets - convert to symbols
    print("\n" + "-"*60 + "\nENSEMBL DATASETS (converting to symbols)\n" + "-"*60)
    for n, p in ENSEMBL_DATASETS.items():
        r = insp.get(n, {})
        if "error" in r or not r.get("recommended_source"): 
            print(f"  Skipping {n}: {r.get('error', 'no source')}")
            continue
        gs, nc = get_genes_ensembl(p, n, r["recommended_source"])
        stats[n] = {"n_cells": nc, "n_genes": len(gs)}
        for g in gs: all_g.add(g); g2d[g].add(n)
    
    # MTX datasets
    print("\n" + "-"*60 + "\nCOMBINED MTX\n" + "-"*60)
    mtx_g, mtx_cc = get_genes_mtx()
    if mtx_g:
        for mn in MTX_DATASETS:
            for g in mtx_g: all_g.add(g); g2d[g].add(mn)
            # Get cell count from metadata - use display name mapping
            display_name = DATASET_META.get(mn, {}).get("dataset", mn)
            if display_name in mtx_cc:
                stats[mn] = {"n_cells": mtx_cc[display_name], "n_genes": len(mtx_g)}
            else:
                # Try direct match
                for mtx_name, count in mtx_cc.items():
                    mapped_name = MTX_DATASET_MAPPING.get(mtx_name, mtx_name)
                    if mapped_name == mn:
                        stats[mn] = {"n_cells": count, "n_genes": len(mtx_g)}
                        break
                else:
                    stats[mn] = {"n_cells": 0, "n_genes": len(mtx_g)}
    
    # Stats before filter
    n_total = len(all_g)
    n_ens = sum(1 for g in all_g if is_ensembl(g))
    n_fb = sum(1 for g in all_g if is_flybase(g))
    n_ercc = sum(1 for g in all_g if g.startswith("ERCC"))
    
    print("\n" + "="*60)
    print("BEFORE FILTERING")
    print("="*60)
    print(f"Total unique genes: {n_total:,}")
    print(f"  - Gene symbols: {n_total - n_ens - n_fb - n_ercc:,}")
    print(f"  - Unmapped Ensembl IDs: {n_ens:,}")
    print(f"  - Unmapped FlyBase IDs: {n_fb:,}")
    print(f"  - ERCC spike-ins: {n_ercc:,}")
    
    # Filter - minimal!
    excluded = set()
    kept = set()
    
    for g in all_g:
        if should_exclude(g):
            excluded.add(g)
        elif len(g2d[g]) >= MIN_DATASETS_PER_GENE:
            kept.add(g)
        else:
            excluded.add(g)  # Only if MIN_DATASETS_PER_GENE > 1
    
    kept = sorted(kept)
    
    print("\n" + "="*60)
    print("AFTER FILTERING")
    print("="*60)
    print(f"Excluded: {len(excluded):,}")
    print(f"  - Unmapped Ensembl: {sum(1 for g in excluded if is_ensembl(g)):,}")
    print(f"  - Unmapped FlyBase: {sum(1 for g in excluded if is_flybase(g)):,}")
    print(f"  - ERCC spike-ins: {sum(1 for g in excluded if g.startswith('ERCC')):,}")
    print(f"  - Below min datasets: {len(excluded) - sum(1 for g in excluded if is_ensembl(g) or is_flybase(g) or g.startswith('ERCC')):,}")
    print(f"\nKEPT GENES (OUTER JOIN): {len(kept):,}")
    
    # Save
    rows = [{"gene": g, "dataset": DATASET_META.get(d,{}).get("dataset",d), "dataset_internal": d} for g in kept for d in sorted(g2d[g])]
    pd.DataFrame(rows).to_parquet(GENE_MANIFEST_PATH, index=False)
    print(f"\nSaved: {GENE_MANIFEST_PATH}")
    
    with open(f"{OUTPUT_DIR}/filtered_genes.txt", 'w') as f:
        for g in kept: f.write(g + "\n")
    print(f"Saved: {OUTPUT_DIR}/filtered_genes.txt")
    
    with open(f"{OUTPUT_DIR}/dataset_statistics.json", 'w') as f: json.dump(stats, f, indent=2)
    
    # Gene coverage summary
    print("\n" + "-"*60)
    print("GENE COVERAGE BY DATASET COUNT")
    print("-"*60)
    for n in range(1, min(14, len(stats) + 1)):
        count = sum(1 for g in kept if len(g2d[g]) >= n)
        print(f"  In >= {n} datasets: {count:,}")
    
    # Dataset summary
    print("\n" + "-"*60)
    print("DATASET SUMMARY")
    print("-"*60)
    total_cells = 0
    for name, s in sorted(stats.items()):
        nc = s.get('n_cells', 0)
        ng = s.get('n_genes', 0)
        total_cells += nc
        print(f"  {name}: {nc:,} cells, {ng:,} genes")
    print(f"\nTOTAL CELLS: {total_cells:,}")
    
    # Sanity check
    if len(kept) < 55000:
        print(f"\n⚠️  WARNING: Only {len(kept):,} genes. Expected ~60k for outer join!")
    else:
        print(f"\n✓ Gene count looks good for outer join: {len(kept):,}")
    
    mark_checkpoint(STEP_NAME)
    print(f"\n✓ Checkpoint saved")

if __name__ == "__main__": main()
