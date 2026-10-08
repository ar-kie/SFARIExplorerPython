"""
SFARI Explorer Data Preparation Pipeline v8
============================================

Complete pipeline for the FILTERED h5ad file that properly:
1. Adds missing metadata (Raj, Velmeshev 2019, Sziraki, Davie)
2. Creates merged_sample and merged_time columns
3. Transfers to full genes h5ad
4. Generates parquets with correct sample_type

This combines the logic from:
- add_missing_metadata.py
- create_merged_columns.py  
- transfer_annotations.py
- prep_sfari_data_v5.py

Run stages:
  Stage 1: python prep_sfari_data_v8.py --stage 1  # Add metadata + export for R
  Stage 2: Rscript run_batch_correction.R           # Run in R (optional)
  Stage 3: python prep_sfari_data_v8.py --stage 3   # Make parquets

Or direct (no R correction):
  python prep_sfari_data_v8.py --direct
"""

import os
import sys
import re
import argparse
import numpy as np
import pandas as pd
import anndata as ad
from scipy import sparse
import pyarrow as pa
import pyarrow.parquet as pq
import gc
from collections import defaultdict

# =============================================================================
# CONFIG
# =============================================================================

# Input files
# CONCORD output (integrate_concord.py). Previously: 03182026_combined_pegasus_harmony_pred_filt.h5ad.
# If cells were QC-filtered after integration, point this to the filtered file.
SFARI_ROOT = os.environ.get('SFARI_ROOT', '/sc/arion/projects/ad-omics/raphael/SFARI')  # data root; override with $SFARI_ROOT
REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FILTERED_H5AD = f'{SFARI_ROOT}/data/combined_concord_label_transfer.h5ad'
FULL_GENES_H5AD = f'{SFARI_ROOT}/pipeline_output/concatenated_annotated.h5ad'

# External metadata files (for datasets that need them)
VELMESHEV_2019_META = f'{SFARI_ROOT}/data/Velmeshev/data/meta.tsv'
SZIRAKI_META = f'{SFARI_ROOT}/data/Cao/data/GSM6538356_RNA_cell_annotation.csv'
DAVIE_META = f'{SFARI_ROOT}/data/Aerts/data/57k/annotation.tsv'

# Output directories
OUTPUT_DIR = f'{SFARI_ROOT}/data'
PARQUET_DIR = f'{SFARI_ROOT}/data/parquet_v3'
R_EXCHANGE_DIR = f'{SFARI_ROOT}/data/r_exchange_v3'

# Risk genes
RISK_GENES_CSV = os.path.join(REPO_DIR, 'resources', 'SFARI-Gene_genes_07-08-2025release_10-08-2025export.csv')

# Column names
SPECIES_COL = 'organism'
DATASET_COL = 'dataset'
CELLTYPE_COL = 'predicted_labels'
SAMPLE_COL = 'merged_sample'
TIMEPOINT_COL = 'merged_time'

# UMAP settings
UMAP_SUBSAMPLE_N = 200_000
UMAP_KEY = 'X_umap'

# =============================================================================
# ORGANOID DATASETS - CRITICAL FIX
# =============================================================================

ORGANOID_DATASETS = ['He (2024)', 'Wang (2022)']

# =============================================================================
# SAMPLE AND TIMEPOINT COLUMN MAPPINGS PER DATASET
# =============================================================================

DATASET_SAMPLE_COL = {
    'He (2024)': 'bio_sample',
    'Wang (2022)': 'sample_id',
    'Bhaduri (2021)': 'donor_id',
    'Braun (2023)': 'donor_id',
    'Velmeshev (2023)': 'donor_id',
    'Zhu (2023)': 'donor_id',
    'Wang (2025)': 'donor_id',
    'Velmeshev (2019)': 'meta_individual',
    'La Manno (2021)': 'DonorID',
    'Jin (2025)': 'library_prep',
    'Sziraki (2023)': 'meta_sample_id',
    'Raj (2020)': 'sample',
    'Davie (2018)': 'meta_sample_id',
}

DATASET_TIME_COL = {
    'He (2024)': 'organoid_age_days',
    'Wang (2022)': None,
    'Bhaduri (2021)': 'development_stage',
    'Braun (2023)': 'development_stage_ontology_term_id',
    'Velmeshev (2023)': 'development_stage',
    'Zhu (2023)': 'development_stage',
    'Wang (2025)': 'Estimated_postconceptional_age_in_days',
    'Velmeshev (2019)': 'meta_timepoint',
    'La Manno (2021)': 'Age',
    'Jin (2025)': 'age_cat',
    'Sziraki (2023)': 'meta_timepoint',
    'Raj (2020)': 'meta_timepoint',
    'Davie (2018)': 'meta_timepoint',
}

# Datasets added with pipeline/cellxgene/fetch_cellxgene.py (data/cellxgene/registry.json)
from dataset_registry import extend_dataset_maps
extend_dataset_maps(DATASET_SAMPLE_COL, DATASET_TIME_COL, ORGANOID_DATASETS)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def write_parquet(df, path, optimize=True):
    if optimize:
        df = df.copy()
        for col in df.columns:
            if df[col].dtype == 'float64':
                df[col] = df[col].astype('float32')
            elif df[col].dtype == 'int64':
                df[col] = df[col].astype('int32')
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, path, compression='snappy')
    size_mb = os.path.getsize(path) / 1024 / 1024
    print(f"  Wrote: {path} ({len(df):,} rows, {size_mb:.2f} MB)")

def get_sample_type(dataset):
    return 'organoid' if dataset in ORGANOID_DATASETS else 'in_vivo'

def extract_raj_timepoint(sample_name):
    match = re.search(r'zf(\d+(?:s|hpf|dpf))', str(sample_name))
    return match.group(1) if match else 'unknown'

def parse_davie_barcode(barcode):
    match = re.search(r'-(DGRP-\d+|w\d+)_(\d+d)_r(\d+)-', str(barcode))
    if match:
        return f"{match.group(1)}_{match.group(2)}_r{match.group(3)}", match.group(2)
    return None, None

# =============================================================================
# STAGE 1A: ADD MISSING METADATA
# =============================================================================

def add_missing_metadata(adata):
    print("\n" + "=" * 60)
    print("Adding missing metadata")
    print("=" * 60)
    
    for col in ['meta_sample_id', 'meta_timepoint', 'meta_age', 'meta_individual']:
        if col not in adata.obs.columns:
            adata.obs[col] = ''
        adata.obs[col] = adata.obs[col].astype(str)
    
    # RAJ (2020)
    print("\n  Processing Raj (2020)...")
    raj_mask = adata.obs[DATASET_COL] == 'Raj (2020)'
    if raj_mask.sum() > 0:
        raj_samples = adata.obs.loc[raj_mask, 'sample'].astype(str)
        raj_timepoints = raj_samples.apply(extract_raj_timepoint)
        adata.obs.loc[raj_mask, 'meta_sample_id'] = raj_samples.values
        adata.obs.loc[raj_mask, 'meta_timepoint'] = raj_timepoints.values
        print(f"    {raj_mask.sum():,} cells, timepoints: {raj_timepoints.unique().tolist()}")
    
    # VELMESHEV (2019)
    # h5ad format: Velmeshev-(2019)_AAACCTGGTACGCACC-1_1823_BA24-Velmeshev-2019
    # meta format: AAACCTGGTACGCACC-1_1823_BA24
    # Need to extract the middle part
    print("\n  Processing Velmeshev (2019)...")
    velm_mask = adata.obs[DATASET_COL] == 'Velmeshev (2019)'
    if velm_mask.sum() > 0 and os.path.exists(VELMESHEV_2019_META):
        velm_meta = pd.read_csv(VELMESHEV_2019_META, sep='\t')
        velm_meta_dict = {str(row['cell']): row for _, row in velm_meta.iterrows()}
        
        def extract_velmeshev_cell_id(h5ad_id):
            """
            Extract meta cell ID from h5ad cell ID.
            h5ad: Velmeshev-(2019)_AAACCTGGTACGCACC-1_1823_BA24-Velmeshev-2019
            meta: AAACCTGGTACGCACC-1_1823_BA24
            """
            s = str(h5ad_id)
            # Remove prefix "Velmeshev-(2019)_"
            if s.startswith('Velmeshev-(2019)_'):
                s = s[len('Velmeshev-(2019)_'):]
            # Remove suffix "-Velmeshev-2019"
            if s.endswith('-Velmeshev-2019'):
                s = s[:-len('-Velmeshev-2019')]
            return s
        
        matched = 0
        for idx in adata.obs_names[velm_mask]:
            meta_cell_id = extract_velmeshev_cell_id(idx)
            row = velm_meta_dict.get(meta_cell_id)
            if row is not None:
                adata.obs.loc[idx, 'meta_sample_id'] = str(row['sample'])
                adata.obs.loc[idx, 'meta_individual'] = str(row['individual'])
                adata.obs.loc[idx, 'meta_timepoint'] = str(row['age']) + 'yo'
                matched += 1
        print(f"    {velm_mask.sum():,} cells, matched: {matched:,}")
        if matched > 0:
            ages = adata.obs.loc[velm_mask, 'meta_timepoint'].unique()
            print(f"    Timepoints: {[a for a in ages if a and a != ''][:10]}")
    
    # SZIRAKI (2023)
    print("\n  Processing Sziraki (2023)...")
    sziraki_mask = adata.obs[DATASET_COL] == 'Sziraki (2023)'
    if sziraki_mask.sum() > 0 and os.path.exists(SZIRAKI_META):
        sziraki_meta = pd.read_csv(SZIRAKI_META)
        sziraki_meta_dict = {}
        for _, row in sziraki_meta.iterrows():
            sziraki_meta_dict[str(row['sample'])] = row
            if '.' in str(row['sample']):
                sziraki_meta_dict[str(row['sample']).split('.')[1]] = row
        
        matched = 0
        for idx in adata.obs_names[sziraki_mask]:
            base_id = idx.replace('-Sziraki (2023)', '').replace('-Sziraki-2023', '')
            row = sziraki_meta_dict.get(base_id)
            if row is None:
                for part in base_id.replace(':', '.').replace('_', '.').split('.'):
                    if len(part) >= 16 and part in sziraki_meta_dict:
                        row = sziraki_meta_dict[part]
                        break
            if row is not None:
                adata.obs.loc[idx, 'meta_sample_id'] = f"{row['Type']}_rep{row['Replicate_ID']}"
                adata.obs.loc[idx, 'meta_timepoint'] = str(row['Type'])
                matched += 1
        print(f"    {sziraki_mask.sum():,} cells, matched: {matched:,}")
    
    # DAVIE (2018)
    print("\n  Processing Davie (2018)...")
    davie_mask = adata.obs[DATASET_COL] == 'Davie (2018)'
    if davie_mask.sum() > 0:
        matched = 0
        for idx in adata.obs_names[davie_mask]:
            sample_id, age = parse_davie_barcode(idx)
            if sample_id:
                adata.obs.loc[idx, 'meta_sample_id'] = sample_id
                adata.obs.loc[idx, 'meta_timepoint'] = age
                matched += 1
        print(f"    {davie_mask.sum():,} cells, parsed: {matched:,}")
    
    return adata

# =============================================================================
# STAGE 1B: CREATE MERGED COLUMNS
# =============================================================================

def create_merged_columns(adata):
    print("\n" + "=" * 60)
    print("Creating merged columns")
    print("=" * 60)
    
    adata.obs['merged_sample'] = ''
    adata.obs['merged_time'] = 'unknown'
    adata.obs['sample_type'] = 'in_vivo'
    
    for dataset in adata.obs[DATASET_COL].unique():
        mask = adata.obs[DATASET_COL] == dataset
        n_cells = mask.sum()
        print(f"\n  {dataset} ({n_cells:,} cells):")
        
        adata.obs.loc[mask, 'sample_type'] = get_sample_type(dataset)
        
        sample_col = DATASET_SAMPLE_COL.get(dataset)
        if sample_col and sample_col in adata.obs.columns:
            vals = adata.obs.loc[mask, sample_col].astype(str).replace({'': 'unknown', 'nan': 'unknown'})
            adata.obs.loc[mask, 'merged_sample'] = dataset + '|' + vals
            print(f"    Sample: {sample_col} ({vals[vals != 'unknown'].nunique()} unique)")
        else:
            adata.obs.loc[mask, 'merged_sample'] = dataset + '|nosample'
        
        time_col = DATASET_TIME_COL.get(dataset)
        if time_col and time_col in adata.obs.columns:
            vals = adata.obs.loc[mask, time_col].astype(str).replace({'': 'unknown', 'nan': 'unknown'})
            adata.obs.loc[mask, 'merged_time'] = vals
            n_unique = vals[vals != 'unknown'].nunique()
            if n_unique > 0:
                print(f"    Time: {time_col} ({n_unique} unique)")
    
    return adata

def fix_organism_labels(adata):
    if adata.obs[SPECIES_COL].dtype.name == 'category':
        adata.obs[SPECIES_COL] = adata.obs[SPECIES_COL].astype(str)
    raj_mask = adata.obs[DATASET_COL] == 'Raj (2020)'
    if raj_mask.sum() > 0:
        adata.obs.loc[raj_mask, SPECIES_COL] = 'Zebrafish'
        print(f"  Fixed Raj (2020): {raj_mask.sum():,} cells -> Zebrafish")
    return adata

# =============================================================================
# STAGE 1: FULL PREPARATION
# =============================================================================

def stage1_prepare_and_export():
    print("=" * 70)
    print("STAGE 1: Prepare metadata and export for R")
    print("=" * 70)
    
    os.makedirs(R_EXCHANGE_DIR, exist_ok=True)
    os.makedirs(PARQUET_DIR, exist_ok=True)
    
    print(f"\n1. Loading filtered h5ad: {FILTERED_H5AD}")
    adata = ad.read_h5ad(FILTERED_H5AD)
    print(f"   Shape: {adata.shape}")
    
    filtered_cells = set(adata.obs_names)
    filtered_cells_list = list(adata.obs_names)
    
    adata = fix_organism_labels(adata)
    adata = add_missing_metadata(adata)
    adata = create_merged_columns(adata)
    
    # Summary
    print("\n" + "=" * 60)
    print("Metadata summary")
    print("=" * 60)
    for ds in sorted(adata.obs[DATASET_COL].unique()):
        mask = adata.obs[DATASET_COL] == ds
        n_samples = adata.obs.loc[mask, 'merged_sample'].nunique()
        times = adata.obs.loc[mask, 'merged_time']
        n_times = times[times != 'unknown'].nunique()
        st = adata.obs.loc[mask, 'sample_type'].iloc[0]
        print(f"  {ds}: {n_samples} samples, {n_times} timepoints, {st}")
    
    # Create pseudobulk groups BEFORE loading full genes
    print(f"\n2. Creating pseudobulk groups...")
    adata.obs['_pb_group'] = (
        adata.obs[DATASET_COL].astype(str) + '|' +
        adata.obs[CELLTYPE_COL].astype(str) + '|' +
        adata.obs['merged_sample'].astype(str) + '|' +
        adata.obs['merged_time'].astype(str)
    )
    
    groups = adata.obs['_pb_group'].unique()
    n_groups = len(groups)
    print(f"   Groups: {n_groups:,}")
    
    # Build group -> cell indices mapping
    group_to_cells = {}
    for group in groups:
        mask = adata.obs['_pb_group'] == group
        group_to_cells[group] = list(adata.obs_names[mask])
    
    # Build group metadata
    pb_meta = []
    for group in groups:
        mask = adata.obs['_pb_group'] == group
        row = adata.obs.loc[mask].iloc[0]
        pb_meta.append({
            'sample_id': group, 
            'dataset': row[DATASET_COL], 
            'cell_type': row[CELLTYPE_COL],
            'sample': row['merged_sample'], 
            'organism': row[SPECIES_COL],
            'timepoint': row['merged_time'], 
            'sample_type': row['sample_type'], 
            'n_cells': mask.sum()
        })
    pb_meta = pd.DataFrame(pb_meta)
    
    # Save UMAP before loading full genes
    if UMAP_KEY in adata.obsm:
        print(f"\n   Saving UMAP subsample...")
        # CONCORD fits UMAP on a subsample; other cells have NaN coordinates
        has_umap = np.flatnonzero(np.isfinite(np.asarray(adata.obsm[UMAP_KEY])[:, 0]))
        n_sample = min(UMAP_SUBSAMPLE_N, len(has_umap))
        idx = np.sort(np.random.choice(has_umap, n_sample, replace=False))
        umap_df = pd.DataFrame({
            'cell_id': adata.obs_names[idx],
            'umap_1': adata.obsm[UMAP_KEY][idx, 0],
            'umap_2': adata.obsm[UMAP_KEY][idx, 1],
        })
        for col in [SPECIES_COL, DATASET_COL, CELLTYPE_COL, 'sample_type']:
            if col in adata.obs.columns:
                umap_df[col] = adata.obs[col].iloc[idx].values
        write_parquet(umap_df, f'{PARQUET_DIR}/umap_subsample.parquet')
    
    del adata
    gc.collect()
    
    # Load full genes in BACKED mode (doesn't load X into memory)
    print(f"\n3. Loading full genes (backed mode): {FULL_GENES_H5AD}")
    adata_full = ad.read_h5ad(FULL_GENES_H5AD, backed='r')
    print(f"   Shape: {adata_full.shape}")
    
    # Get gene names
    gene_names = list(adata_full.var_names)
    n_genes = len(gene_names)
    
    # Find common cells and their indices in full file
    full_cells = list(adata_full.obs_names)
    full_cell_to_idx = {c: i for i, c in enumerate(full_cells)}
    
    common_cells = [c for c in filtered_cells_list if c in full_cell_to_idx]
    print(f"   Common cells: {len(common_cells):,}")
    
    # Update group_to_cells to only include common cells with their indices
    group_to_indices = {}
    for group, cells in group_to_cells.items():
        indices = [full_cell_to_idx[c] for c in cells if c in full_cell_to_idx]
        if indices:
            group_to_indices[group] = np.array(indices, dtype=np.int64)
    
    # Filter pb_meta to only groups with cells
    valid_groups = set(group_to_indices.keys())
    pb_meta = pb_meta[pb_meta['sample_id'].isin(valid_groups)].reset_index(drop=True)
    
    # Pseudobulk aggregation - process in chunks to avoid memory issues
    print(f"\n4. Computing pseudobulk (chunked)...")
    
    CHUNK_SIZE = 5000  # genes per chunk
    n_chunks = (n_genes + CHUNK_SIZE - 1) // CHUNK_SIZE
    
    # Initialize output array
    pb_counts = np.zeros((len(pb_meta), n_genes), dtype=np.float32)
    
    for chunk_idx in range(n_chunks):
        start_gene = chunk_idx * CHUNK_SIZE
        end_gene = min((chunk_idx + 1) * CHUNK_SIZE, n_genes)
        
        print(f"   Chunk {chunk_idx+1}/{n_chunks}: genes {start_gene}-{end_gene}")
        
        # Load this chunk of genes for ALL cells (backed mode reads on demand)
        # We need to read slices carefully
        chunk_data = adata_full.X[:, start_gene:end_gene]
        if sparse.issparse(chunk_data):
            chunk_data = chunk_data.tocsr()
        
        # Aggregate for each group
        for i, row in pb_meta.iterrows():
            group = row['sample_id']
            if group in group_to_indices:
                indices = group_to_indices[group]
                if sparse.issparse(chunk_data):
                    group_sum = np.asarray(chunk_data[indices, :].sum(axis=0)).ravel()
                else:
                    group_sum = chunk_data[indices, :].sum(axis=0)
                pb_counts[i, start_gene:end_gene] = group_sum
        
        del chunk_data
        gc.collect()
    
    adata_full.file.close()
    del adata_full
    gc.collect()
    
    # Save
    print(f"\n5. Saving...")
    pb_counts_df = pd.DataFrame(pb_counts, index=pb_meta['sample_id'], columns=gene_names)
    pb_counts_df.to_csv(f'{R_EXCHANGE_DIR}/pseudobulk_counts.csv')
    print(f"   Saved: pseudobulk_counts.csv ({pb_counts_df.shape})")
    
    pb_meta.to_csv(f'{R_EXCHANGE_DIR}/pseudobulk_meta.csv', index=False)
    print(f"   Saved: pseudobulk_meta.csv ({len(pb_meta)} rows)")
    
    pd.DataFrame({'gene': gene_names}).to_csv(f'{R_EXCHANGE_DIR}/gene_info.csv', index=False)
    print(f"   Saved: gene_info.csv")
    
    create_r_script()
    print("\nStage 1 complete!")

R_SCRIPT = '''
library(limma); library(edgeR)
R_EXCHANGE_DIR <- "{r_exchange_dir}"
counts_raw <- read.csv(file.path(R_EXCHANGE_DIR, "pseudobulk_counts.csv"), row.names=1)
meta <- read.csv(file.path(R_EXCHANGE_DIR, "pseudobulk_meta.csv"))
counts <- t(counts_raw)
rownames(meta) <- meta$sample_id; meta <- meta[colnames(counts), ]
keep <- rowSums(counts > 10) >= 3; counts <- counts[keep, ]
dge <- DGEList(counts=counts); dge <- calcNormFactors(dge)
cpm_matrix <- cpm(dge, log=TRUE, prior.count=1)
meta$organism <- as.factor(meta$organism); meta$dataset <- as.factor(meta$dataset); meta$cell_type <- as.factor(meta$cell_type)
design <- model.matrix(~ 0 + organism + cell_type, data=meta)
corrected <- removeBatchEffect(cpm_matrix, batch=meta$dataset, design=design)
write.csv(t(corrected), file.path(R_EXCHANGE_DIR, "corrected_expression.csv"))
'''

def create_r_script():
    with open(f'{R_EXCHANGE_DIR}/run_batch_correction.R', 'w') as f:
        f.write(R_SCRIPT.format(r_exchange_dir=R_EXCHANGE_DIR))

# =============================================================================
# STAGE 3: GENERATE PARQUETS
# =============================================================================

def stage3_make_parquets():
    print("=" * 70)
    print("STAGE 3: Generate parquets")
    print("=" * 70)
    
    os.makedirs(PARQUET_DIR, exist_ok=True)
    
    corrected_path = f'{R_EXCHANGE_DIR}/corrected_expression.csv'
    if os.path.exists(corrected_path):
        expr_matrix = pd.read_csv(corrected_path, index_col=0)
    else:
        pb_counts = pd.read_csv(f'{R_EXCHANGE_DIR}/pseudobulk_counts.csv', index_col=0)
        row_sums = pb_counts.sum(axis=1).replace(0, 1)
        expr_matrix = np.log1p(pb_counts.div(row_sums, axis=0) * 1e6)
    
    pb_meta = pd.read_csv(f'{R_EXCHANGE_DIR}/pseudobulk_meta.csv')
    pb_counts = pd.read_csv(f'{R_EXCHANGE_DIR}/pseudobulk_counts.csv', index_col=0)
    pb_meta = pb_meta[pb_meta['sample_id'].isin(expr_matrix.index)]
    genes = list(expr_matrix.columns)
    
    print(f"  Expression: {expr_matrix.shape}, Meta: {len(pb_meta)}")
    
    # Expression summaries
    rows = []
    for (org, ds, ct, tp, st), group in pb_meta.groupby(['organism', 'dataset', 'cell_type', 'timepoint', 'sample_type'], dropna=False):
        valid_ids = [s for s in group['sample_id'] if s in expr_matrix.index]
        if not valid_ids:
            continue
        mean_expr = expr_matrix.loc[valid_ids].mean(axis=0)
        pct_expr = (pb_counts.loc[[s for s in valid_ids if s in pb_counts.index]] > 0).mean(axis=0)
        n_cells = group['n_cells'].sum()
        for gene in genes:
            rows.append({'species': org, 'tissue': ds, 'cell_type': ct, 'gene_native': gene, 'gene_human': gene,
                        'mean_expr': float(mean_expr[gene]), 'pct_expressing': float(pct_expr[gene]),
                        'n_cells': int(n_cells), 'timepoint': tp, 'sample_type': st})
    
    expr_df = pd.DataFrame(rows)
    write_parquet(expr_df, f'{PARQUET_DIR}/expression_summaries.parquet')
    
    # Verify
    print("\n  sample_type verification:")
    for ds in sorted(expr_df['tissue'].unique()):
        st = expr_df[expr_df['tissue'] == ds]['sample_type'].unique()
        expected = 'organoid' if ds in ORGANOID_DATASETS else 'in_vivo'
        print(f"    {'✓' if list(st) == [expected] else '✗'} {ds}: {list(st)}")
    
    # Other parquets
    gene_map = pd.DataFrame([{'species': org, 'gene_native': g, 'gene_human': g} for org in pb_meta['organism'].unique() for g in genes])
    write_parquet(gene_map, f'{PARQUET_DIR}/gene_map.parquet')
    
    celltype_meta = pb_meta.groupby(['organism', 'dataset', 'cell_type']).agg({'n_cells': 'sum'}).reset_index()
    celltype_meta.columns = ['species', 'tissue', 'cell_type', 'n_cells']
    write_parquet(celltype_meta, f'{PARQUET_DIR}/celltype_meta.parquet')
    
    if os.path.exists(RISK_GENES_CSV):
        risk_df = pd.read_csv(RISK_GENES_CSV).rename(columns={'gene-symbol': 'gene_symbol', 'gene-score': 'gene_score'})
    else:
        risk_df = pd.DataFrame(columns=['gene_symbol', 'gene_score'])
    write_parquet(risk_df, f'{PARQUET_DIR}/risk_genes.parquet')
    
    # Dataset overview
    overview = []
    for ds in sorted(expr_df['tissue'].unique()):
        ds_data = expr_df[expr_df['tissue'] == ds]
        ds_meta = pb_meta[pb_meta['dataset'] == ds]
        tps = [t for t in ds_data['timepoint'].dropna().unique() if str(t) not in ['unknown', 'nan']]
        overview.append({'Dataset': ds, 'Species': ds_data['species'].iloc[0], 'Samples': ds_meta['sample'].nunique(),
                        'Cell Types': ds_data['cell_type'].nunique(), 'Timepoints': len(tps),
                        'Sample Type': 'organoid' if ds in ORGANOID_DATASETS else 'in_vivo', 'Total Cells': ds_meta['n_cells'].sum()})
    write_parquet(pd.DataFrame(overview), f'{PARQUET_DIR}/dataset_overview.parquet')
    
    print("\nStage 3 complete!")

def direct_generation():
    stage1_prepare_and_export()
    stage3_make_parquets()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', type=int, choices=[1, 3])
    parser.add_argument('--direct', action='store_true')
    args = parser.parse_args()
    
    if args.direct:
        direct_generation()
    elif args.stage == 1:
        stage1_prepare_and_export()
    elif args.stage == 3:
        stage3_make_parquets()
    else:
        print("Usage: python prep_sfari_data_v8.py --direct")
