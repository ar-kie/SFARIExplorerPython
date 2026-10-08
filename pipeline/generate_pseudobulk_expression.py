"""
Generate sample-level pseudobulk expression for temporal analysis.

This script creates pseudobulk_expression.csv with expression per (sample, cell_type, gene)
preserving temporal resolution across timepoints.

Run this BEFORE generate_temporal_parquets_v2.py
"""

import scanpy as sc
import pandas as pd
import numpy as np
import re
import os
from collections import defaultdict

# =============================================================================
# CONFIG
# =============================================================================

SFARI_ROOT = os.environ.get('SFARI_ROOT', '/sc/arion/projects/ad-omics/raphael/SFARI')  # data root; override with $SFARI_ROOT
H5AD_PATH = f'{SFARI_ROOT}/data/combined_concord_label_transfer.h5ad'
OUTPUT_DIR = os.environ.get('SFARI_EXCHANGE_DIR', f'{SFARI_ROOT}/data/r_exchange')  # pseudobulk <-> R exchange
PARQUET_DIR = f'{SFARI_ROOT}/data/parquet_v2'

# Cell type column in adata.obs
CELLTYPE_COL = 'predicted_labels'  # Adjust if different

# Minimum cells per pseudobulk sample
MIN_CELLS = 10

# SFARI genes to focus on (or None for all genes)
SFARI_GENES_PATH = f'{SFARI_ROOT}/data/sfari_genes.txt'

# =============================================================================
# TIMEPOINT EXTRACTION
# =============================================================================

def extract_timepoint_from_barcode(barcode, dataset):
    """Extract timepoint from barcode based on dataset-specific patterns."""
    
    if 'Davie' in dataset:
        # Drosophila: DGRP-551_0d_r1, DGRP-551_1d_r2, etc.
        match = re.search(r'_(\d+)d_', str(barcode))
        if match:
            return f"{match.group(1)}d", float(match.group(1))
    
    # Add other dataset-specific patterns here as needed
    # elif 'SomeOtherDataset' in dataset:
    #     ...
    
    return None, None


def get_sample_id(obs_row, dataset):
    """Generate a unique sample ID from observation metadata."""
    
    # Try to extract from barcode first
    barcode = obs_row.name if hasattr(obs_row, 'name') else str(obs_row.get('barcode', ''))
    
    # Look for sample-related columns
    sample_cols = ['sample', 'sample_id', 'batch', 'donor', 'subject']
    sample_val = None
    for col in sample_cols:
        if col in obs_row.index and pd.notna(obs_row[col]):
            sample_val = str(obs_row[col])
            break
    
    if sample_val:
        return f"{dataset}|{sample_val}"
    
    # Fall back to extracting from barcode
    # Pattern: dataset_sampleinfo-barcode or similar
    if '-' in barcode:
        parts = barcode.split('-')
        if len(parts) >= 2:
            return f"{dataset}|{parts[-2]}"
    
    return f"{dataset}|unknown"


# =============================================================================
# MAIN
# =============================================================================

print("=" * 60)
print("Generating sample-level pseudobulk expression")
print("=" * 60)

# Load data
print("\n1. Loading h5ad...")
adata = sc.read_h5ad(H5AD_PATH)
print(f"   Shape: {adata.shape}")
print(f"   Obs columns: {adata.obs.columns.tolist()}")

# Load SFARI genes if available
sfari_genes = None
if os.path.exists(SFARI_GENES_PATH):
    with open(SFARI_GENES_PATH) as f:
        sfari_genes = set(line.strip().upper() for line in f if line.strip())
    print(f"   SFARI genes: {len(sfari_genes)}")

# Check for required columns
if 'dataset' not in adata.obs.columns:
    print("   ERROR: 'dataset' column not found in adata.obs")
    print(f"   Available columns: {adata.obs.columns.tolist()}")
    exit(1)

if CELLTYPE_COL not in adata.obs.columns:
    print(f"   ERROR: '{CELLTYPE_COL}' column not found")
    exit(1)

# -----------------------------------------------------------------------------
# 2. Extract timepoints per dataset
# -----------------------------------------------------------------------------

print("\n2. Extracting timepoints...")

datasets = adata.obs['dataset'].unique()
print(f"   Datasets: {datasets}")

# Add timepoint info to obs
adata.obs['timepoint_extracted'] = None
adata.obs['numeric_time_extracted'] = None

for dataset in datasets:
    mask = adata.obs['dataset'] == dataset
    barcodes = adata.obs.index[mask]
    
    timepoints = []
    numeric_times = []
    
    for bc in barcodes:
        tp, nt = extract_timepoint_from_barcode(bc, dataset)
        timepoints.append(tp)
        numeric_times.append(nt)
    
    adata.obs.loc[mask, 'timepoint_extracted'] = timepoints
    adata.obs.loc[mask, 'numeric_time_extracted'] = numeric_times
    
    # Report
    unique_tps = pd.Series(timepoints).dropna().unique()
    if len(unique_tps) > 0:
        print(f"   {dataset}: {len(unique_tps)} timepoints - {sorted(unique_tps)}")
    else:
        print(f"   {dataset}: No timepoints extracted from barcodes")
        # Try to get from existing columns
        for col in ['timepoint', 'age', 'time', 'stage']:
            if col in adata.obs.columns:
                vals = adata.obs.loc[mask, col].dropna().unique()
                if len(vals) > 0:
                    print(f"     -> Found in '{col}': {vals[:5]}...")
                    break

# -----------------------------------------------------------------------------
# 3. Create sample groupings
# -----------------------------------------------------------------------------

print("\n3. Creating sample groupings...")

# Determine sample column
# For Drosophila, sample = dataset + barcode prefix (strain_day_rep)
# For others, use existing sample/batch column or create from barcode

def get_sample_group(row):
    """Get sample grouping for pseudobulk."""
    dataset = row['dataset']
    barcode = row.name
    
    if 'Davie' in dataset:
        # Drosophila: group by strain_day_replicate
        match = re.search(r'(DGRP-\d+_\d+d_r\d+)', str(barcode))
        if match:
            return f"{dataset}|{match.group(1)}"
    
    # For other datasets, try sample column
    for col in ['sample', 'sample_id', 'batch', 'donor']:
        if col in row.index and pd.notna(row[col]):
            return f"{dataset}|{row[col]}"
    
    # Fallback: use timepoint if available
    if pd.notna(row.get('timepoint_extracted')):
        return f"{dataset}|{row['timepoint_extracted']}"
    
    return f"{dataset}|all"

adata.obs['sample_group'] = adata.obs.apply(get_sample_group, axis=1)

print(f"   Unique sample groups: {adata.obs['sample_group'].nunique()}")

# -----------------------------------------------------------------------------
# 4. Generate pseudobulk expression
# -----------------------------------------------------------------------------

print("\n4. Generating pseudobulk expression...")

# Get gene info
gene_names = adata.var.index.tolist()
human_orthologs = adata.var.get('human_ortholog', pd.Series([None]*len(adata.var), index=adata.var.index))

# Filter genes to SFARI if available
if sfari_genes:
    gene_mask = [g.upper() in sfari_genes or 
                 (pd.notna(human_orthologs.get(g)) and str(human_orthologs.get(g)).upper() in sfari_genes)
                 for g in gene_names]
    print(f"   Filtering to SFARI genes: {sum(gene_mask)} genes")
else:
    gene_mask = [True] * len(gene_names)

gene_indices = [i for i, m in enumerate(gene_mask) if m]
filtered_genes = [gene_names[i] for i in gene_indices]

# Group cells
groups = adata.obs.groupby(['sample_group', CELLTYPE_COL])

print(f"   Processing {len(groups)} (sample, celltype) groups...")

results = []
processed = 0

for (sample_group, cell_type), group_idx in groups.groups.items():
    if len(group_idx) < MIN_CELLS:
        continue
    
    # Get expression matrix for this group
    subset = adata[group_idx, gene_indices]
    
    if hasattr(subset.X, 'toarray'):
        expr_matrix = subset.X.toarray()
    else:
        expr_matrix = subset.X
    
    # Calculate pseudobulk metrics
    mean_expr = np.mean(expr_matrix, axis=0)
    pct_expressing = np.mean(expr_matrix > 0, axis=0) * 100
    n_cells = len(group_idx)
    
    # Get metadata from first cell in group
    first_cell = adata.obs.loc[group_idx[0]]
    dataset = first_cell['dataset']
    organism = first_cell.get('organism', first_cell.get('species', 'Unknown'))
    timepoint = first_cell.get('timepoint_extracted') or first_cell.get('timepoint')
    numeric_time = first_cell.get('numeric_time_extracted') or first_cell.get('numeric_time')
    sample_type = first_cell.get('sample_type', 'in_vivo')
    
    # Create sample_id
    sample_id = f"{dataset}|{cell_type}|{sample_group}"
    
    # Add to results (one row per gene)
    for i, gene in enumerate(filtered_genes):
        if mean_expr[i] > 0 or pct_expressing[i] > 0:  # Skip unexpressed genes
            results.append({
                'sample_id': sample_id,
                'sample_group': sample_group,
                'dataset': dataset,
                'cell_type': cell_type,
                'organism': organism,
                'timepoint': timepoint,
                'numeric_time': numeric_time,
                'sample_type': sample_type,
                'gene_native': gene,
                'gene_human': human_orthologs.get(gene),
                'mean_expr': mean_expr[i],
                'pct_expressing': pct_expressing[i],
                'n_cells': n_cells
            })
    
    processed += 1
    if processed % 100 == 0:
        print(f"     Processed {processed} groups...")

print(f"   Total groups processed: {processed}")

# Create dataframe
pb_expr = pd.DataFrame(results)
print(f"   Pseudobulk expression shape: {pb_expr.shape}")

# -----------------------------------------------------------------------------
# 5. Save outputs
# -----------------------------------------------------------------------------

print("\n5. Saving outputs...")

# Save pseudobulk expression
expr_path = f'{OUTPUT_DIR}/pseudobulk_expression.csv'
pb_expr.to_csv(expr_path, index=False)
print(f"   Written: {expr_path} ({len(pb_expr):,} rows)")

# Also save as parquet for faster loading
expr_parquet_path = f'{PARQUET_DIR}/pseudobulk_expression.parquet'
pb_expr.to_parquet(expr_parquet_path, compression='snappy')
print(f"   Written: {expr_parquet_path}")

# Update pseudobulk metadata with extracted timepoints
meta_path = f'{OUTPUT_DIR}/pseudobulk_meta_numeric_time.csv'
if os.path.exists(meta_path):
    pb_meta = pd.read_csv(meta_path)
    
    # Add any new samples from Drosophila with correct timepoints
    new_meta = pb_expr[['sample_id', 'sample_group', 'dataset', 'cell_type', 
                         'organism', 'timepoint', 'numeric_time', 'sample_type', 'n_cells']].drop_duplicates()
    
    # Merge/update
    pb_meta_updated = pd.concat([pb_meta, new_meta], ignore_index=True).drop_duplicates(
        subset=['dataset', 'cell_type', 'sample_group'], keep='last'
    )
    
    pb_meta_updated.to_csv(meta_path, index=False)
    print(f"   Updated: {meta_path} ({len(pb_meta_updated):,} rows)")

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------

print("\n" + "=" * 60)
print("Summary")
print("=" * 60)

print("\nTimepoints per species:")
for org in pb_expr['organism'].unique():
    org_data = pb_expr[pb_expr['organism'] == org]
    timepoints = org_data['timepoint'].dropna().unique()
    print(f"  {org}: {sorted(timepoints)}")

print("\nSamples per species:")
for org in pb_expr['organism'].unique():
    org_data = pb_expr[pb_expr['organism'] == org]
    n_samples = org_data['sample_group'].nunique()
    n_celltypes = org_data['cell_type'].nunique()
    print(f"  {org}: {n_samples} samples, {n_celltypes} cell types")

print("\n" + "=" * 60)
print("Next steps:")
print("=" * 60)
print("""
1. Run generate_temporal_parquets_v2.py to create:
   - temporal_expression.parquet (with full timepoint resolution)
   - stage_mapping.parquet (cross-species stage alignment)
   
2. Copy parquets to your app data directory

3. The app will now show:
   - Drosophila: Day 0, 1, 3, 6, 9, 15, 30, 50
   - All other species with their full timepoints
   - Cross-species comparisons by unified developmental stage
""")
