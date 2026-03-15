"""
SFARI Explorer Data Preparation Pipeline - FIXED VERSION
=========================================================
This version PRESERVES TIMEPOINTS in expression_summaries.parquet

Key fix: Groups by (organism, dataset, cell_type, timepoint) instead of
just (organism, dataset, cell_type)

Run as: python prep_sfari_data_v4.py --stage 2
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

# =============================================================================
# CONFIG
# =============================================================================

# Input files
ANNOTATED_SCVI_PATH = '/sc/arion/projects/ad-omics/raphael/SFARI/data/combined_scanvi_label_transfer.h5ad'
FULL_GENES_PATH = '/sc/arion/projects/ad-omics/raphael/SFARI/pipeline_output/concatenated_annotated_with_meta.h5ad'

# Output directories
OUTPUT_DIR = '/sc/arion/projects/ad-omics/raphael/SFARI/data'
PARQUET_DIR = '/sc/arion/projects/ad-omics/raphael/SFARI/data/parquet_v2'
R_EXCHANGE_DIR = '/sc/arion/projects/ad-omics/raphael/SFARI/data/r_exchange'

# Column names
SPECIES_COL = 'organism'
DATASET_COL = 'dataset'
CELLTYPE_COL = 'predicted_labels'
SAMPLE_COL = 'merged_sample'
TIMEPOINT_COL = 'merged_time'

# Risk genes
RISK_GENES_CSV = '/sc/arion/projects/ad-omics/raphael/SFARI/SFARI_genes/SFARI-Gene_genes_07-08-2025release_10-08-2025export.csv'

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def write_parquet(df, path):
    """Write dataframe to parquet."""
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, path, compression='snappy')
    size_mb = os.path.getsize(path) / 1024 / 1024
    print(f"  Wrote: {path} ({len(df):,} rows, {size_mb:.2f} MB)")


# =============================================================================
# STAGE 2: Import corrected data and make parquets (FIXED)
# =============================================================================

def stage2_make_parquets():
    """
    Import corrected expression values and generate final parquet files.
    FIXED: Preserves timepoints in expression_summaries.parquet
    """
    print("=" * 60)
    print("STAGE 2: Generate parquet files (FIXED - preserves timepoints)")
    print("=" * 60)
    
    os.makedirs(PARQUET_DIR, exist_ok=True)
    
    # Check if corrected data exists
    corrected_path = f'{R_EXCHANGE_DIR}/corrected_expression.csv'
    
    if os.path.exists(corrected_path):
        print("\n1. Loading DESeq2/dream corrected expression...")
        corrected_df = pd.read_csv(corrected_path, index_col=0)
        use_corrected = True
        print(f"  Corrected matrix: {corrected_df.shape}")
    else:
        print("\n1. No corrected data found, using raw pseudobulk...")
        print(f"  (Expected: {corrected_path})")
        use_corrected = False
    
    # Load pseudobulk metadata (this has timepoint info!)
    print("\n2. Loading pseudobulk metadata...")
    
    # Try the numeric time version first
    meta_path_numeric = f'{R_EXCHANGE_DIR}/pseudobulk_meta_numeric_time.csv'
    meta_path = f'{R_EXCHANGE_DIR}/pseudobulk_meta.csv'
    
    if os.path.exists(meta_path_numeric):
        pb_meta = pd.read_csv(meta_path_numeric)
        print(f"  Using: pseudobulk_meta_numeric_time.csv")
    else:
        pb_meta = pd.read_csv(meta_path)
        print(f"  Using: pseudobulk_meta.csv")
    
    print(f"  Samples: {len(pb_meta):,}")
    print(f"  Columns: {pb_meta.columns.tolist()}")
    
    # Check timepoint coverage
    print("\n  Timepoints per organism:")
    for org in pb_meta['organism'].unique():
        org_data = pb_meta[pb_meta['organism'] == org]
        timepoints = org_data['timepoint'].dropna().unique()
        sample_types = org_data['sample_type'].unique() if 'sample_type' in org_data.columns else ['unknown']
        print(f"    {org}: {len(timepoints)} timepoints, sample_types={list(sample_types)}")
    
    # Always load raw counts for pct_expressing calculation
    pb_counts_path = f'{R_EXCHANGE_DIR}/pseudobulk_counts.csv'
    if os.path.exists(pb_counts_path):
        print("\n  Loading raw pseudobulk counts (for pct_expressing)...")
        pb_counts = pd.read_csv(pb_counts_path, index_col=0)
        have_raw_counts = True
        print(f"  Raw counts shape: {pb_counts.shape}")
    else:
        have_raw_counts = False
        print("  No raw counts file found")
    
    # Load or compute expression matrix
    if use_corrected:
        expr_matrix = corrected_df
    else:
        if have_raw_counts:
            print("\n  CPM normalizing...")
            row_sums = pb_counts.sum(axis=1)
            row_sums[row_sums == 0] = 1
            expr_matrix = pb_counts.div(row_sums, axis=0) * 1e6
            expr_matrix = np.log1p(expr_matrix)
        else:
            raise FileNotFoundError("No corrected data or raw counts found")
    
    # Filter metadata to only include samples in expression matrix
    print("\n3. Filtering metadata to match expression matrix...")
    samples_in_expr = set(expr_matrix.index)
    samples_in_meta = set(pb_meta['sample_id'])
    
    missing_samples = samples_in_meta - samples_in_expr
    if missing_samples:
        print(f"  Note: {len(missing_samples)} samples excluded during correction")
        pb_meta = pb_meta[pb_meta['sample_id'].isin(samples_in_expr)]
    
    print(f"  Final metadata: {len(pb_meta)} samples")
    
    # ==========================================================================
    # FIX: Parse Drosophila numeric_time from timepoint string
    # ==========================================================================
    
    print("\n  Fixing Drosophila numeric_time...")
    dros_mask = pb_meta['organism'] == 'Drosophila'
    if dros_mask.any():
        def parse_dros_time(row):
            if row['organism'] != 'Drosophila':
                return row['numeric_time']
            if pd.notna(row['numeric_time']):
                return row['numeric_time']
            # Parse from timepoint like '0d', '1d', '3d', etc.
            tp = str(row['timepoint']) if pd.notna(row['timepoint']) else ''
            match = re.match(r'(\d+)d', tp)
            if match:
                return float(match.group(1))
            return row['numeric_time']
        
        pb_meta['numeric_time'] = pb_meta.apply(parse_dros_time, axis=1)
        print(f"    Drosophila numeric_time values: {sorted(pb_meta[dros_mask]['numeric_time'].dropna().unique())}")
    
    # ==========================================================================
    # KEY FIX: Group by (organism, dataset, cell_type, timepoint) 
    # NOT just (organism, dataset, cell_type)
    # ==========================================================================
    
    print("\n4. Creating expression summaries (WITH TIMEPOINTS)...")
    
    # Determine grouping columns
    group_cols = ['organism', 'dataset', 'cell_type', 'timepoint']
    
    # Add sample_type if available
    if 'sample_type' in pb_meta.columns:
        group_cols.append('sample_type')
    
    # Add numeric_time if available
    if 'numeric_time' in pb_meta.columns:
        has_numeric_time = True
    else:
        has_numeric_time = False
    
    print(f"  Grouping by: {group_cols}")
    
    rows = []
    groups = list(pb_meta.groupby(group_cols))
    n_groups = len(groups)
    
    print(f"  Processing {n_groups} groups...")
    
    for i, (group_key, group_meta) in enumerate(groups):
        if i % 100 == 0:
            print(f"    Group {i+1}/{n_groups}")
        
        # Unpack group key
        if 'sample_type' in group_cols:
            org, ds, ct, tp, st = group_key
        else:
            org, ds, ct, tp = group_key
            st = 'in_vivo'
        
        sample_ids = group_meta['sample_id'].tolist()
        n_cells = group_meta['n_cells'].sum()
        
        # Get numeric_time (use first value in group)
        if has_numeric_time:
            numeric_time = group_meta['numeric_time'].iloc[0]
        else:
            numeric_time = None
        
        # Get expression for these samples
        valid_sample_ids = [s for s in sample_ids if s in expr_matrix.index]
        if not valid_sample_ids:
            continue
        
        expr_subset = expr_matrix.loc[valid_sample_ids]
        
        # Mean expression across samples
        mean_expr = expr_subset.mean(axis=0)
        
        # Percent expressing
        if have_raw_counts:
            valid_raw_ids = [s for s in valid_sample_ids if s in pb_counts.index]
            if valid_raw_ids:
                raw_subset = pb_counts.loc[valid_raw_ids]
                pct_expr = (raw_subset > 0).mean(axis=0)
            else:
                pct_expr = (expr_subset > 0).mean(axis=0)
        else:
            pct_expr = (expr_subset > 0).mean(axis=0)
        
        # Create row for each gene
        for gene in expr_matrix.columns:
            row = {
                'species': org,
                'tissue': ds,
                'cell_type': ct,
                'gene_native': gene,
                'gene_human': gene,
                'mean_expr': float(mean_expr[gene]),
                'pct_expressing': float(pct_expr[gene]),
                'n_cells': int(n_cells),
                'timepoint': tp,
                'sample_type': st
            }
            if has_numeric_time:
                row['numeric_time'] = numeric_time
            rows.append(row)
    
    expr_summary_df = pd.DataFrame(rows)
    print(f"\n  Expression summaries: {len(expr_summary_df):,} rows")
    
    # Validate timepoint coverage
    print("\n  Timepoints in final data:")
    for org in expr_summary_df['species'].unique():
        org_data = expr_summary_df[expr_summary_df['species'] == org]
        timepoints = org_data['timepoint'].dropna().unique()
        genes = org_data['gene_human'].nunique()
        print(f"    {org}: {len(timepoints)} timepoints, {genes:,} genes")
        print(f"      Timepoints: {sorted(timepoints)[:10]}{'...' if len(timepoints) > 10 else ''}")
    
    # Write expression summaries
    write_parquet(expr_summary_df, f'{PARQUET_DIR}/expression_summaries.parquet')
    
    # ==========================================================================
    # Create other parquet files
    # ==========================================================================
    
    # Gene map
    print("\n5. Creating gene map...")
    genes = expr_matrix.columns.tolist()
    organisms = pb_meta['organism'].unique()
    
    gene_map_rows = []
    for org in organisms:
        for gene in genes:
            gene_map_rows.append({
                'species': org,
                'gene_native': gene,
                'gene_human': gene
            })
    
    gene_map_df = pd.DataFrame(gene_map_rows)
    write_parquet(gene_map_df, f'{PARQUET_DIR}/gene_map.parquet')
    
    # Cell type metadata (aggregated across timepoints)
    print("\n6. Creating cell type metadata...")
    celltype_meta = pb_meta.groupby(['organism', 'dataset', 'cell_type']).agg({
        'n_cells': 'sum'
    }).reset_index()
    celltype_meta.columns = ['species', 'tissue', 'cell_type', 'n_cells']
    write_parquet(celltype_meta, f'{PARQUET_DIR}/celltype_meta.parquet')
    
    # Risk genes
    print("\n7. Processing risk genes...")
    if os.path.exists(RISK_GENES_CSV):
        risk_df = pd.read_csv(RISK_GENES_CSV)
        if 'gene-symbol' in risk_df.columns:
            risk_df = risk_df.rename(columns={'gene-symbol': 'gene_symbol'})
        if 'gene-score' in risk_df.columns:
            risk_df = risk_df.rename(columns={'gene-score': 'gene_score'})
    else:
        risk_df = pd.DataFrame(columns=['gene_symbol', 'gene_score', 'gene_name'])
    write_parquet(risk_df, f'{PARQUET_DIR}/risk_genes.parquet')
    
    # ==========================================================================
    # Create temporal_expression.parquet
    # ==========================================================================
    
    print("\n8. Creating temporal_expression.parquet...")
    
    # Filter to rows with valid timepoint
    temporal_df = expr_summary_df[expr_summary_df['timepoint'].notna() & 
                                   (expr_summary_df['timepoint'] != 'unknown')].copy()
    
    # Create time bins
    def create_time_bin(row):
        org = row['species']
        tp = str(row['timepoint']) if pd.notna(row['timepoint']) else ''
        st = row.get('sample_type', 'in_vivo')
        nt = row.get('numeric_time')
        
        # For Drosophila, parse numeric time from timepoint string if needed
        if org == 'Drosophila' and (pd.isna(nt) or nt is None):
            # Parse from timepoint like '0d', '1d', '3d', etc.
            match = re.match(r'(\d+)d', tp)
            if match:
                nt = float(match.group(1))
        
        if pd.isna(nt) or nt is None:
            # Can't determine numeric time, use raw timepoint as bin
            return tp, 99
        
        if org == 'Human':
            if st == 'organoid':
                if nt <= 30: return '0-30 days', 1
                elif nt <= 60: return '31-60 days', 2
                elif nt <= 90: return '61-90 days', 3
                elif nt <= 120: return '91-120 days', 4
                else: return '>120 days', 5
            else:
                if nt <= 70: return 'Early fetal (GW<10)', 1
                elif nt <= 140: return 'Mid fetal (GW10-20)', 2
                elif nt <= 280: return 'Late fetal (GW20-40)', 3
                elif nt <= 280 + 365*2: return 'Infant (0-2y)', 4
                elif nt <= 280 + 365*12: return 'Child (2-12y)', 5
                elif nt <= 280 + 365*18: return 'Adolescent (12-18y)', 6
                else: return 'Adult (18+y)', 7
        
        elif org == 'Mouse':
            if nt <= 12: return 'Early embryo (E<12)', 1
            elif nt <= 16: return 'Mid embryo (E12-16)', 2
            elif nt <= 20: return 'Late embryo (E16-20)', 3
            elif nt <= 50: return 'Neonatal (P0-P30)', 4
            elif nt <= 110: return 'Juvenile (1-3mo)', 5
            elif nt <= 385: return 'Adult (3-12mo)', 6
            else: return 'Aged (>12mo)', 7
        
        elif org == 'Zebrafish':
            if nt <= 24: return '0-24 hpf', 1
            elif nt <= 48: return '24-48 hpf', 2
            elif nt <= 72: return '48-72 hpf', 3
            elif nt <= 120: return '72-120 hpf (5dpf)', 4
            else: return '>5 dpf', 5
        
        elif org == 'Drosophila':
            if nt == 0: return 'Day 0 (eclosion)', 1
            elif nt <= 1: return 'Day 1', 2
            elif nt <= 3: return 'Day 3', 3
            elif nt <= 6: return 'Day 6', 4
            elif nt <= 9: return 'Day 9', 5
            elif nt <= 15: return 'Day 15', 6
            elif nt <= 30: return 'Day 30', 7
            else: return 'Day 50', 8
        
        return tp, 99
    
    time_bins = temporal_df.apply(create_time_bin, axis=1)
    temporal_df['time_bin'] = [tb[0] for tb in time_bins]
    temporal_df['time_order'] = [tb[1] for tb in time_bins]
    
    # Add unified stage mapping
    UNIFIED_STAGE_MAP = {
        'Early fetal (GW<10)': ('Early Development', 1),
        'Mid fetal (GW10-20)': ('Mid Development', 2),
        'Late fetal (GW20-40)': ('Late Development', 3),
        'Infant (0-2y)': ('Postnatal', 4),
        'Child (2-12y)': ('Juvenile', 5),
        'Adolescent (12-18y)': ('Juvenile', 5),
        'Adult (18+y)': ('Adult', 7),
        '0-30 days': ('Early Development', 1),
        '31-60 days': ('Early Development', 1),
        '61-90 days': ('Mid Development', 2),
        '91-120 days': ('Mid Development', 2),
        '>120 days': ('Late Development', 3),
        'Early embryo (E<12)': ('Early Development', 1),
        'Mid embryo (E12-16)': ('Mid Development', 2),
        'Late embryo (E16-20)': ('Late Development', 3),
        'Neonatal (P0-P30)': ('Postnatal', 4),
        'Juvenile (1-3mo)': ('Juvenile', 5),
        'Adult (3-12mo)': ('Adult', 7),
        'Aged (>12mo)': ('Aged', 8),
        '0-24 hpf': ('Early Development', 1),
        '24-48 hpf': ('Mid Development', 2),
        '48-72 hpf': ('Late Development', 3),
        '72-120 hpf (5dpf)': ('Late Development', 3),
        '>5 dpf': ('Juvenile', 5),
        'Day 0 (eclosion)': ('Adult (young)', 6),
        'Day 1': ('Adult (young)', 6),
        'Day 3': ('Adult (young)', 6),
        'Day 6': ('Adult (young)', 6),
        'Day 9': ('Adult (young)', 6),
        'Day 15': ('Adult (mature)', 7),
        'Day 30': ('Adult (mature)', 7),
        'Day 50': ('Aged', 8),
        # Fallback mappings for raw timepoint strings
        '0d': ('Adult (young)', 6),
        '1d': ('Adult (young)', 6),
        '3d': ('Adult (young)', 6),
        '6d': ('Adult (young)', 6),
        '9d': ('Adult (young)', 6),
        '15d': ('Adult (mature)', 7),
        '30d': ('Adult (mature)', 7),
        '50d': ('Aged', 8),
    }
    
    def get_unified_stage(time_bin):
        if time_bin in UNIFIED_STAGE_MAP:
            return UNIFIED_STAGE_MAP[time_bin]
        return (None, 99)
    
    unified = temporal_df['time_bin'].apply(get_unified_stage)
    temporal_df['unified_stage'] = [u[0] for u in unified]
    temporal_df['unified_stage_order'] = [u[1] for u in unified]
    
    print(f"  Temporal expression: {len(temporal_df):,} rows")
    
    # Summary
    print("\n  Temporal data summary:")
    for org in temporal_df['species'].unique():
        org_data = temporal_df[temporal_df['species'] == org]
        genes = org_data['gene_human'].nunique()
        time_bins = sorted(org_data['time_bin'].unique(), 
                          key=lambda x: org_data[org_data['time_bin']==x]['time_order'].iloc[0])
        sample_types = org_data['sample_type'].unique()
        print(f"    {org}:")
        print(f"      Genes: {genes:,}")
        print(f"      Sample types: {list(sample_types)}")
        print(f"      Time bins: {time_bins}")
    
    write_parquet(temporal_df, f'{PARQUET_DIR}/temporal_expression.parquet')
    
    # ==========================================================================
    # Summary
    # ==========================================================================
    
    print("\n" + "=" * 60)
    print("Stage 2 complete!")
    print("=" * 60)
    print(f"\nParquet files written to: {PARQUET_DIR}")
    print("\nFiles:")
    for f in sorted(os.listdir(PARQUET_DIR)):
        if f.endswith('.parquet'):
            size = os.path.getsize(f'{PARQUET_DIR}/{f}') / 1e6
            print(f"  - {f} ({size:.1f} MB)")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SFARI Explorer data preparation (FIXED)')
    parser.add_argument('--stage', type=int, choices=[2], default=2,
                        help='Stage 2: make parquets with timepoints preserved')
    
    args = parser.parse_args()
    
    if args.stage == 2:
        stage2_make_parquets()
    else:
        print("Usage: python prep_sfari_data_v5.py --stage 2")
