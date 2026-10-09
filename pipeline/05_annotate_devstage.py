#!/usr/bin/env python3
"""Step 5: Annotate Developmental Stages (only obs is loaded; HDF5 copies the count matrix)"""
import os, re, numpy as np, pandas as pd, scanpy as sc
import warnings; warnings.filterwarnings('ignore')
from config import *
import h5io

STEP_NAME = "05_annotate_devstage"

DEVSTAGE_COLS = ["development_stage", "Age", "PseudoAge", "age_cat", "age_group", "organoid_age_days", "Estimated_postconceptional_age_in_days"]

PATTERNS = [
    (re.compile(r"pcw|gestational|fetal|fetus", re.I), "Human Prenatal"),
    (re.compile(r"\be\s*\d+", re.I), "Mouse Embryonic"),
    (re.compile(r"\bp\s*\d+", re.I), "Mouse Postnatal"),
    (re.compile(r"organoid", re.I), "Organoid"),
    (re.compile(r"\badult\b", re.I), "Adult"),
]

def assign_dev(raw):
    if raw is None or (isinstance(raw, float) and np.isnan(raw)): return "Unknown"
    s = str(raw).strip().lower()
    if not s or s in {"unknown", "nan", ""}: return "Unknown"
    for p, c in PATTERNS:
        if p.search(s): return c
    return "Other"

def main():
    ensure_dirs()
    if checkpoint_exists(STEP_NAME): print("Step done."); return
    print("="*60 + "\nSTEP 5: ANNOTATE DEVELOPMENTAL STAGES\n" + "="*60)
    
    print(f"\nLoading {ANNOTATED_PATH}...")
    adata = sc.read_h5ad(ANNOTATED_PATH, backed='r')  # obs in memory, X stays on disk
    print(f"Shape: {adata.shape}")
    
    cols = [c for c in DEVSTAGE_COLS if c in adata.obs.columns]
    print(f"\nDev stage columns found: {cols}")
    
    merged = pd.Series(index=adata.obs.index, dtype='object')
    for c in cols:
        m = merged.isna() & adata.obs[c].notna() & (adata.obs[c].astype(str) != 'nan')
        merged.loc[m] = adata.obs.loc[m, c].astype(str)
    
    adata.obs['dev_stage_merged'] = merged
    adata.obs['dev_stage_supercategory'] = merged.map(assign_dev)
    
    n_ann = merged.notna().sum()
    print(f"\nAnnotated: {n_ann:,} / {adata.n_obs:,} ({100*n_ann/adata.n_obs:.1f}%)")
    
    print("\nSupercategory distribution:")
    for cat, cnt in adata.obs['dev_stage_supercategory'].value_counts().items():
        print(f"  {cat}: {cnt:,}")
    
    print("\nCleaning obs columns...")
    for c in adata.obs.columns:
        if adata.obs[c].dtype == 'object': 
            adata.obs[c] = adata.obs[c].fillna('').astype(str).astype('category')
    
    print(f"\nSaving to {FINAL_PATH}...")
    adata.file.close()
    h5io.write_with_obs(ANNOTATED_PATH, FINAL_PATH, adata.obs)
    mark_checkpoint(STEP_NAME)
    print(f"\n✓ PIPELINE COMPLETE")
    print(f"Final output: {FINAL_PATH}")

if __name__ == "__main__": main()
