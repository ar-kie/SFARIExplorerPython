"""
Create merged_sample and merged_time columns from existing metadata.
Selects the best available column per dataset for pseudobulk aggregation.
"""

import numpy as np
import pandas as pd
import anndata as ad
import gc

# =============================================================================
# CONFIG
# =============================================================================

INPUT_H5AD = '/sc/arion/projects/ad-omics/raphael/SFARI/data/combined_concord_with_meta.h5ad'
OUTPUT_H5AD = '/sc/arion/projects/ad-omics/raphael/SFARI/data/combined_concord_with_merged_meta.h5ad'

DATASET_COL = 'dataset'

# =============================================================================
# LOAD DATA
# =============================================================================

print("=" * 60)
print("Creating merged_sample and merged_time columns")
print("=" * 60)

print("\n1. Loading h5ad file...")
adata = ad.read_h5ad(INPUT_H5AD)
print(f"   Shape: {adata.shape}")
print(f"   Datasets: {adata.obs[DATASET_COL].unique().tolist()}")

# =============================================================================
# DEFINE BEST COLUMNS PER DATASET
# =============================================================================

# Best sample column per dataset (for pseudobulk biological replicates)
# Priority: donor/individual > sample > fallback
dataset_sample_col = {
    # Human datasets
    'He (2024)': 'bio_sample',           # 303 samples
    'Wang (2022)': 'sample_id',          # 4 samples
    'Bhaduri (2021)': 'donor_id',        # 13 donors
    'Braun (2023)': 'donor_id',          # 26 donors
    'Velmeshev (2023)': 'donor_id',      # 106 donors
    'Zhu (2023)': 'donor_id',            # 12 donors
    'Wang (2025)': 'donor_id',           # 27 donors
    'Velmeshev (2019)': 'meta_individual',  # From added metadata
    
    # Mouse datasets
    'La Manno (2021)': 'DonorID',        # 43 donors
    'Jin (2025)': 'library_prep',        # Library prep as sample
    'Sziraki (2023)': 'meta_sample_id',  # From added metadata
    
    # Zebrafish
    'Raj (2020)': 'sample',              # 12 timepoint-based samples
    
    # Drosophila
    'Davie (2018)': '_parse_barcode_sample',  # Special: parse from barcode
}

# Best timepoint column per dataset
dataset_time_col = {
    # Human datasets
    'He (2024)': 'development_stage',                    # 18 stages
    'Wang (2022)': None,                                 # No timepoint
    'Bhaduri (2021)': 'development_stage',               # 8 stages
    'Braun (2023)': 'development_stage_ontology_term_id', # 10 stages
    'Velmeshev (2023)': 'development_stage',             # 40 stages
    'Zhu (2023)': 'development_stage',                   # 10 stages
    'Wang (2025)': 'development_stage',                  # 20 stages
    'Velmeshev (2019)': 'meta_timepoint',                # From added metadata (age)
    
    # Mouse datasets
    'La Manno (2021)': 'Age',            # 20 ages
    'Jin (2025)': 'age_cat',             # 2 ages (2mo, 18mo)
    'Sziraki (2023)': 'meta_timepoint',  # From added metadata (3mo, 6mo, 21mo, disease)
    
    # Zebrafish
    'Raj (2020)': 'meta_timepoint',      # From added metadata (10s, 24hpf, 2dpf, etc.)
    
    # Drosophila
    'Davie (2018)': '_parse_barcode_time',  # Special: parse from barcode
}

# =============================================================================
# HELPER FUNCTIONS FOR BARCODE PARSING
# =============================================================================

import re

def parse_davie_barcode(barcode):
    """
    Parse Davie barcode to extract sample and timepoint.
    Format: Davie-(2018)_BARCODE-{LINE}_{AGE}_r{REPLICATE}-Davie-2018
    Example: Davie-(2018)_AAACCTGAGGCCATAG-DGRP-551_0d_r1-Davie-2018
    Returns: (sample_id, timepoint)
    """
    # Pattern: LINE_AGE_rREPLICATE
    match = re.search(r'-(DGRP-\d+|w\d+)_(\d+d)_r(\d+)-', str(barcode))
    if match:
        line = match.group(1)
        age = match.group(2)
        replicate = match.group(3)
        sample_id = f"{line}_{age}_r{replicate}"
        return sample_id, age
    return 'unknown', 'unknown'

# =============================================================================
# CREATE MERGED COLUMNS
# =============================================================================

print("\n2. Creating merged columns...")

# Initialize
adata.obs['merged_sample'] = ''
adata.obs['merged_time'] = 'unknown'

# Process each dataset
for dataset in adata.obs[DATASET_COL].unique():
    mask = adata.obs[DATASET_COL] == dataset
    n_cells = mask.sum()
    
    print(f"\n{dataset} ({n_cells:,} cells):")
    
    # --- SAMPLE ---
    sample_col = dataset_sample_col.get(dataset)
    
    if sample_col == '_parse_barcode_sample':
        # Special handling: parse from barcode (Davie)
        if dataset == 'Davie (2018)':
            barcodes = adata.obs_names[mask]
            parsed = [parse_davie_barcode(bc) for bc in barcodes]
            samples = [p[0] for p in parsed]
            adata.obs.loc[mask, 'merged_sample'] = dataset + '|' + pd.Series(samples, index=adata.obs_names[mask]).values
            n_unique = len(set(samples)) - (1 if 'unknown' in samples else 0)
            print(f"  Sample: parsed from barcode ({n_unique} unique)")
    elif sample_col and sample_col in adata.obs.columns:
        vals = adata.obs.loc[mask, sample_col].astype(str)
        n_unique = vals.nunique()
        
        # Create merged_sample: dataset_value
        adata.obs.loc[mask, 'merged_sample'] = dataset + '|' + vals
        print(f"  Sample: {sample_col} ({n_unique} unique)")
    else:
        # Fallback
        adata.obs.loc[mask, 'merged_sample'] = dataset + '|nosample'
        if sample_col:
            print(f"  Sample: {sample_col} NOT FOUND - using fallback")
        else:
            print(f"  Sample: (none defined) - using fallback")
    
    # --- TIMEPOINT ---
    time_col = dataset_time_col.get(dataset)
    
    if time_col == '_parse_barcode_time':
        # Special handling: parse from barcode (Davie)
        if dataset == 'Davie (2018)':
            barcodes = adata.obs_names[mask]
            parsed = [parse_davie_barcode(bc) for bc in barcodes]
            times = [p[1] for p in parsed]
            adata.obs.loc[mask, 'merged_time'] = pd.Series(times, index=adata.obs_names[mask]).values
            n_unique = len(set(times)) - (1 if 'unknown' in times else 0)
            print(f"  Time: parsed from barcode ({n_unique} unique)")
            examples = list(set(times))[:5]
            print(f"    Examples: {examples}")
    elif time_col and time_col in adata.obs.columns:
        vals = adata.obs.loc[mask, time_col].astype(str)
        # Clean up empty/nan values
        vals = vals.replace({'': 'unknown', 'nan': 'unknown', 'None': 'unknown'})
        n_unique = vals[vals != 'unknown'].nunique()
        
        adata.obs.loc[mask, 'merged_time'] = vals
        
        if n_unique > 0:
            print(f"  Time: {time_col} ({n_unique} unique)")
            # Show example values
            examples = vals[vals != 'unknown'].unique()[:5].tolist()
            print(f"    Examples: {examples}")
        else:
            print(f"  Time: {time_col} (no valid values)")
    else:
        if time_col:
            print(f"  Time: {time_col} NOT FOUND")
        else:
            print(f"  Time: (none defined)")

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 60)
print("Summary")
print("=" * 60)

print("\nmerged_sample:")
for dataset in sorted(adata.obs[DATASET_COL].unique()):
    mask = adata.obs[DATASET_COL] == dataset
    n_unique = adata.obs.loc[mask, 'merged_sample'].nunique()
    print(f"  {dataset}: {n_unique} unique samples")

print("\nmerged_time:")
for dataset in sorted(adata.obs[DATASET_COL].unique()):
    mask = adata.obs[DATASET_COL] == dataset
    times = adata.obs.loc[mask, 'merged_time']
    times_known = times[times != 'unknown']
    n_unique = times_known.nunique()
    pct = 100 * len(times_known) / len(times)
    if n_unique > 0:
        print(f"  {dataset}: {n_unique} timepoints ({pct:.0f}% coverage)")
    else:
        print(f"  {dataset}: no timepoint info")

# Total pseudobulk groups
n_groups = adata.obs.groupby(['dataset', 'merged_sample', 'predicted_labels']).ngroups
print(f"\nTotal pseudobulk groups (dataset × sample × cell_type): {n_groups:,}")

# =============================================================================
# SAVE
# =============================================================================

print("\n" + "=" * 60)
print("Saving...")
print("=" * 60)

# Clean columns for h5ad
for col in ['merged_sample', 'merged_time']:
    adata.obs[col] = adata.obs[col].astype(str)

print(f"  Saving to {OUTPUT_H5AD}...")
adata.write(OUTPUT_H5AD)

print("\nDone!")
print(f"Output: {OUTPUT_H5AD}")
print("\nNew columns:")
print("  - merged_sample: for pseudobulk aggregation")
print("  - merged_time: for temporal analysis & variance partition")
