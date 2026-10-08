"""
Add missing metadata to concatenated h5ad file.

This script adds sample/donor and timepoint metadata for:
- Raj (2020) - Zebrafish: extract timepoint from sample name (e.g., zf10s, zf24hpf, zf2dpf)
- Velmeshev (2019) - Human: merge from meta.tsv (age, individual, sample)
- Sziraki (2023) - Mouse: merge from GSM6538356_RNA_cell_annotation.csv (Type=age, Replicate_ID)
- Davie (2018) - Drosophila: merge from annotation.tsv (age, line, bio_replicate)
"""

import os
import re
import numpy as np
import pandas as pd
import anndata as ad
import gc

# =============================================================================
# CONFIG
# =============================================================================

INPUT_H5AD = '/sc/arion/projects/ad-omics/raphael/SFARI/data/combined_concord_label_transfer.h5ad'
OUTPUT_H5AD = '/sc/arion/projects/ad-omics/raphael/SFARI/data/combined_concord_with_meta.h5ad'

# Metadata files
VELMESHEV_2019_META = '/sc/arion/projects/ad-omics/raphael/SFARI/data/Velmeshev/data/meta.tsv'
SZIRAKI_META = '/sc/arion/projects/ad-omics/raphael/SFARI/data/Cao/data/GSM6538356_RNA_cell_annotation.csv'
DAVIE_META = '/sc/arion/projects/ad-omics/raphael/SFARI/data/Aerts/data/57k/annotation.tsv'

# =============================================================================
# LOAD DATA
# =============================================================================

print("=" * 60)
print("Adding missing metadata to concatenated h5ad")
print("=" * 60)

print("\n1. Loading h5ad file...")
adata = ad.read_h5ad(INPUT_H5AD)
print(f"   Shape: {adata.shape}")

# Initialize new columns
if 'meta_sample_id' not in adata.obs.columns:
    adata.obs['meta_sample_id'] = ''
if 'meta_timepoint' not in adata.obs.columns:
    adata.obs['meta_timepoint'] = ''
if 'meta_age' not in adata.obs.columns:
    adata.obs['meta_age'] = ''
if 'meta_individual' not in adata.obs.columns:
    adata.obs['meta_individual'] = ''

# Convert to string to avoid categorical issues
for col in ['meta_sample_id', 'meta_timepoint', 'meta_age', 'meta_individual']:
    adata.obs[col] = adata.obs[col].astype(str)

# =============================================================================
# RAJ (2020) - ZEBRAFISH
# Extract timepoint from sample name
# =============================================================================

print("\n2. Processing Raj (2020) - Zebrafish...")

raj_mask = adata.obs['dataset'] == 'Raj (2020)'
n_raj = raj_mask.sum()
print(f"   Cells: {n_raj:,}")

if n_raj > 0:
    # The 'sample' column contains names like:
    # GSE158142_zf10s_cc_filt.cluster
    # GSE158142_zf24hpf_cc_filt.cluster
    # GSE158142_zf2dpf_cc_filt.cluster
    
    raj_samples = adata.obs.loc[raj_mask, 'sample'].astype(str)
    
    # Extract timepoint from sample name
    def extract_raj_timepoint(sample_name):
        """Extract timepoint like '10s', '24hpf', '2dpf' from sample name."""
        match = re.search(r'zf(\d+(?:s|hpf|dpf))', str(sample_name))
        if match:
            return match.group(1)
        return 'unknown'
    
    raj_timepoints = raj_samples.apply(extract_raj_timepoint)
    
    # Use sample name as sample_id
    adata.obs.loc[raj_mask, 'meta_sample_id'] = raj_samples.values
    adata.obs.loc[raj_mask, 'meta_timepoint'] = raj_timepoints.values
    adata.obs.loc[raj_mask, 'meta_age'] = raj_timepoints.values
    
    print(f"   Timepoints found: {raj_timepoints.unique().tolist()}")
    print(f"   Distribution:")
    print(raj_timepoints.value_counts())

# =============================================================================
# VELMESHEV (2019) - HUMAN
# Merge from meta.tsv
# =============================================================================

print("\n3. Processing Velmeshev (2019) - Human...")

velm_mask = adata.obs['dataset'] == 'Velmeshev (2019)'
n_velm = velm_mask.sum()
print(f"   Cells: {n_velm:,}")

if n_velm > 0 and os.path.exists(VELMESHEV_2019_META):
    # Load metadata
    velm_meta = pd.read_csv(VELMESHEV_2019_META, sep='\t')
    print(f"   Metadata rows: {len(velm_meta):,}")
    print(f"   Columns: {velm_meta.columns.tolist()}")
    
    # Show example cell IDs from both sources
    velm_cells = adata.obs_names[velm_mask]
    print(f"   Example h5ad cell IDs: {velm_cells[:3].tolist()}")
    print(f"   Example meta cell IDs: {velm_meta['cell'].head(3).tolist()}")
    
    # Create a mapping from barcode (without suffix) to metadata
    # Meta cell IDs look like: AAACCTGGTACGCACC-1_1823_BA24
    velm_meta_dict = {}
    for _, row in velm_meta.iterrows():
        cell_id = str(row['cell'])
        velm_meta_dict[cell_id] = row
        
        # Also store by barcode only (first part before -1_)
        barcode = cell_id.split('-')[0] if '-' in cell_id else cell_id
        velm_meta_dict[barcode] = row
    
    # Match cells
    matched = 0
    for idx in velm_cells:
        # h5ad ID might look like: BARCODE-Velmeshev-2019 or similar
        # Try to extract the original barcode
        
        # Remove dataset suffix
        base_id = idx.replace('-Velmeshev (2019)', '').replace('-Velmeshev-2019', '')
        
        # Try different matching strategies
        row = None
        
        # Strategy 1: Exact match
        if base_id in velm_meta_dict:
            row = velm_meta_dict[base_id]
        
        # Strategy 2: Just the barcode (first 16-18 chars)
        if row is None:
            barcode = base_id.split('-')[0].split('_')[0].split(':')[0]
            if barcode in velm_meta_dict:
                row = velm_meta_dict[barcode]
        
        # Strategy 3: Check if base_id contains any meta cell ID
        if row is None:
            for meta_cell in velm_meta_dict.keys():
                if meta_cell in base_id or base_id in meta_cell:
                    row = velm_meta_dict[meta_cell]
                    break
        
        if row is not None:
            adata.obs.loc[idx, 'meta_sample_id'] = str(row['sample'])
            adata.obs.loc[idx, 'meta_individual'] = str(row['individual'])
            adata.obs.loc[idx, 'meta_age'] = str(row['age'])
            adata.obs.loc[idx, 'meta_timepoint'] = str(row['age']) + 'yo'
            matched += 1
    
    print(f"   Matched: {matched:,} / {n_velm:,}")
    
    if matched > 0:
        velm_ages = adata.obs.loc[velm_mask, 'meta_age']
        print(f"   Ages found: {velm_ages[velm_ages != ''].unique().tolist()[:20]}")
    else:
        # Debug: show what we're trying to match
        print("   DEBUG - No matches. Showing ID patterns:")
        print(f"   h5ad pattern: {velm_cells[0]}")
        print(f"   meta pattern: {velm_meta['cell'].iloc[0]}")
else:
    print(f"   Warning: Metadata file not found: {VELMESHEV_2019_META}")

# =============================================================================
# SZIRAKI (2023) - MOUSE
# Merge from GSM6538356_RNA_cell_annotation.csv
# Type column has age info (3mo, 6mo, 21mo, 5xFAD, APOE4/TREM2)
# =============================================================================

print("\n4. Processing Sziraki (2023) - Mouse...")

sziraki_mask = adata.obs['dataset'] == 'Sziraki (2023)'
n_sziraki = sziraki_mask.sum()
print(f"   Cells: {n_sziraki:,}")

if n_sziraki > 0 and os.path.exists(SZIRAKI_META):
    # Load metadata
    sziraki_meta = pd.read_csv(SZIRAKI_META)
    print(f"   Metadata rows: {len(sziraki_meta):,}")
    print(f"   Columns: {sziraki_meta.columns.tolist()}")
    
    # Get cell IDs
    sziraki_cells = adata.obs_names[sziraki_mask]
    print(f"   Example h5ad cell IDs: {sziraki_cells[:3].tolist()}")
    print(f"   Example meta cell IDs: {sziraki_meta['sample'].head(3).tolist()}")
    
    # Create mapping dict
    # Meta sample looks like: EasySci_001.AACCGATTGCAATCGAACTC
    sziraki_meta_dict = {}
    for _, row in sziraki_meta.iterrows():
        cell_id = str(row['sample'])
        sziraki_meta_dict[cell_id] = row
        
        # Also store by the barcode part only (after the dot)
        if '.' in cell_id:
            barcode = cell_id.split('.')[1]
            sziraki_meta_dict[barcode] = row
    
    matched = 0
    for idx in sziraki_cells:
        # Remove dataset suffix
        base_id = idx.replace('-Sziraki (2023)', '').replace('-Sziraki-2023', '')
        
        row = None
        
        # Strategy 1: Exact match
        if base_id in sziraki_meta_dict:
            row = sziraki_meta_dict[base_id]
        
        # Strategy 2: Just the barcode part
        if row is None:
            # Try extracting barcode-like pattern
            for part in base_id.replace(':', '.').replace('_', '.').split('.'):
                if len(part) >= 16 and part in sziraki_meta_dict:
                    row = sziraki_meta_dict[part]
                    break
        
        # Strategy 3: Check if any meta cell ID matches
        if row is None:
            for meta_cell in list(sziraki_meta_dict.keys())[:1000]:  # Limit for speed
                if meta_cell in base_id or base_id in meta_cell:
                    row = sziraki_meta_dict[meta_cell]
                    break
        
        if row is not None:
            type_val = str(row['Type'])
            replicate = str(row['Replicate_ID'])
            
            adata.obs.loc[idx, 'meta_sample_id'] = f"{type_val}_rep{replicate}"
            adata.obs.loc[idx, 'meta_individual'] = replicate
            
            # Extract age from Type (3mo, 6mo, 21mo) or mark as disease model
            if 'mo' in type_val:
                adata.obs.loc[idx, 'meta_age'] = type_val
                adata.obs.loc[idx, 'meta_timepoint'] = type_val
            else:
                # Disease models (5xFAD, APOE4/TREM2) - no clear age
                adata.obs.loc[idx, 'meta_age'] = type_val
                adata.obs.loc[idx, 'meta_timepoint'] = type_val
            
            matched += 1
    
    print(f"   Matched: {matched:,} / {n_sziraki:,}")
    
    if matched > 0:
        sziraki_ages = adata.obs.loc[sziraki_mask, 'meta_age']
        print(f"   Ages/Types found: {sziraki_ages[sziraki_ages != ''].unique().tolist()}")
    else:
        print("   DEBUG - No matches. Showing ID patterns:")
        print(f"   h5ad pattern: {sziraki_cells[0]}")
        print(f"   meta pattern: {sziraki_meta['sample'].iloc[0]}")
else:
    print(f"   Warning: Metadata file not found: {SZIRAKI_META}")

# =============================================================================
# DAVIE (2018) - DROSOPHILA
# Merge from annotation.tsv
# =============================================================================

print("\n5. Processing Davie (2018) - Drosophila...")

davie_mask = adata.obs['dataset'] == 'Davie (2018)'
n_davie = davie_mask.sum()
print(f"   Cells: {n_davie:,}")

if n_davie > 0 and os.path.exists(DAVIE_META):
    # Load metadata
    davie_meta = pd.read_csv(DAVIE_META, sep='\t')
    print(f"   Metadata rows: {len(davie_meta):,}")
    print(f"   Columns: {davie_meta.columns.tolist()}")
    
    # Get cell IDs
    davie_cells = adata.obs_names[davie_mask]
    print(f"   Example h5ad cell IDs: {davie_cells[:3].tolist()}")
    print(f"   Example meta cell IDs: {davie_meta['sample_id'].head(3).tolist()}")
    
    # Create mapping dict
    # Meta sample_id looks like: AAACCTGAGGCCATAG-DGRP-551_0d_r1
    davie_meta_dict = {}
    for _, row in davie_meta.iterrows():
        cell_id = str(row['sample_id'])
        davie_meta_dict[cell_id] = row
        
        # Also store by barcode only (first part before -)
        barcode = cell_id.split('-')[0]
        davie_meta_dict[barcode] = row
    
    matched = 0
    for idx in davie_cells:
        # Remove dataset suffix
        base_id = idx.replace('-Davie (2018)', '').replace('-Davie-2018', '')
        
        row = None
        
        # Strategy 1: Exact match
        if base_id in davie_meta_dict:
            row = davie_meta_dict[base_id]
        
        # Strategy 2: Just the barcode (16 char)
        if row is None:
            barcode = base_id.split('-')[0].split('_')[0].split(':')[0]
            if len(barcode) >= 16:
                barcode = barcode[:16]
            if barcode in davie_meta_dict:
                row = davie_meta_dict[barcode]
        
        # Strategy 3: Check for partial matches
        if row is None:
            for meta_cell in list(davie_meta_dict.keys())[:1000]:
                if meta_cell in base_id or base_id in meta_cell:
                    row = davie_meta_dict[meta_cell]
                    break
        
        if row is not None:
            line = str(row['line'])
            age = str(row['age'])
            bio_rep = str(row['bio_replicate'])
            
            adata.obs.loc[idx, 'meta_sample_id'] = f"{line}_{age}d_r{bio_rep}"
            adata.obs.loc[idx, 'meta_individual'] = f"{line}_r{bio_rep}"
            adata.obs.loc[idx, 'meta_age'] = f"{age}d"
            adata.obs.loc[idx, 'meta_timepoint'] = f"{age}d"
            matched += 1
    
    print(f"   Matched: {matched:,} / {n_davie:,}")
    
    if matched > 0:
        davie_ages = adata.obs.loc[davie_mask, 'meta_age']
        print(f"   Ages found: {davie_ages[davie_ages != ''].unique().tolist()}")
    else:
        print("   DEBUG - No matches. Showing ID patterns:")
        print(f"   h5ad pattern: {davie_cells[0]}")
        print(f"   meta pattern: {davie_meta['sample_id'].iloc[0]}")
else:
    print(f"   Warning: Metadata file not found: {DAVIE_META}")

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 60)
print("Summary of added metadata")
print("=" * 60)

for dataset in ['Raj (2020)', 'Velmeshev (2019)', 'Sziraki (2023)', 'Davie (2018)']:
    mask = adata.obs['dataset'] == dataset
    n_total = mask.sum()
    
    if n_total > 0:
        n_with_sample = (adata.obs.loc[mask, 'meta_sample_id'] != '').sum()
        n_with_time = (adata.obs.loc[mask, 'meta_timepoint'] != '').sum()
        
        print(f"\n{dataset}:")
        print(f"  Total cells: {n_total:,}")
        print(f"  With sample_id: {n_with_sample:,} ({100*n_with_sample/n_total:.1f}%)")
        print(f"  With timepoint: {n_with_time:,} ({100*n_with_time/n_total:.1f}%)")
        
        if n_with_time > 0:
            timepoints = adata.obs.loc[mask & (adata.obs['meta_timepoint'] != ''), 'meta_timepoint']
            print(f"  Timepoints: {timepoints.unique().tolist()}")

# =============================================================================
# SAVE
# =============================================================================

print("\n" + "=" * 60)
print("Saving...")
print("=" * 60)

# Clean obs columns for h5ad compatibility
print("  Cleaning obs columns...")
for col in adata.obs.columns:
    dtype = adata.obs[col].dtype
    if dtype == bool or dtype == 'boolean':
        adata.obs[col] = adata.obs[col].astype(str)
    elif dtype == object:
        adata.obs[col] = adata.obs[col].astype(str)
    elif hasattr(dtype, 'name') and dtype.name == 'category':
        adata.obs[col] = adata.obs[col].astype(str)
    elif pd.api.types.is_extension_array_dtype(dtype):
        try:
            adata.obs[col] = adata.obs[col].astype(str)
        except:
            pass

print(f"  Saving to {OUTPUT_H5AD}...")
adata.write(OUTPUT_H5AD)

print("\nDone!")
print(f"Output: {OUTPUT_H5AD}")
print("\nNew columns added:")
print("  - meta_sample_id: unified sample/replicate ID")
print("  - meta_individual: donor/individual ID")
print("  - meta_age: age value")
print("  - meta_timepoint: timepoint for temporal analysis")
