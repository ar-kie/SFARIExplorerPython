"""
Normalize timepoints to numeric developmental time.

Converts various timepoint formats to days post-conception (or equivalent).
This allows age to be treated as a continuous covariate in batch correction.

Species-specific conversions:
- Human (in-vivo): GW/pcw to days, postnatal years to days
- Human (organoid): use organoid_age_days column directly
- Mouse: E-days to days, postnatal months to days (excludes disease models)
- Zebrafish: hpf/dpf/somite stages to hours post-fertilization
- Drosophila: days post-eclosion
"""

import re
import numpy as np
import pandas as pd
import anndata as ad

# =============================================================================
# CONFIG
# =============================================================================

import os
SFARI_ROOT = os.environ.get('SFARI_ROOT', '/sc/arion/projects/ad-omics/raphael/SFARI')  # data root; override with $SFARI_ROOT
INPUT_H5AD = f'{SFARI_ROOT}/data/combined_concord_with_merged_meta.h5ad'
OUTPUT_H5AD = f'{SFARI_ROOT}/data/combined_concord_with_numeric_time.h5ad'

# Also update the pseudobulk metadata
PSEUDOBULK_META = f'{SFARI_ROOT}/data/r_exchange/pseudobulk_meta.csv'
PSEUDOBULK_META_OUT = f'{SFARI_ROOT}/data/r_exchange/pseudobulk_meta_numeric_time.csv'

# Organoid datasets - use organoid_age_days column instead of timepoint
# Note: Wang (2022) has no organoid_age_days in metadata - will be excluded from time analysis
ORGANOID_DATASETS = ['He (2024)', 'Wang (2022)']
ORGANOID_DATASETS_WITH_TIME = ['He (2024)']  # Only He has organoid_age_days

# Datasets added with pipeline/cellxgene/fetch_cellxgene.py (data/cellxgene/registry.json)
from dataset_registry import extend_dataset_maps, parse_tagged_age
extend_dataset_maps(organoid_datasets=ORGANOID_DATASETS)

# Mouse disease models to exclude (unknown age, not relevant)
MOUSE_DISEASE_MODELS = ['APOE4/TREM2', '5xFAD']

# =============================================================================
# CONVERSION FUNCTIONS
# =============================================================================

def parse_human_time(timepoint, dataset=None):
    """
    Convert human in-vivo timepoint to days post-conception.
    
    NOTE: For organoid datasets, this function returns NaN.
    Organoids use organoid_age_days column directly (handled separately).
    
    Human gestation: ~280 days (40 weeks)
    """
    if pd.isna(timepoint) or timepoint == 'unknown' or timepoint == '':
        return np.nan
    
    # Organoid datasets - return NaN (use organoid_age_days instead)
    if dataset in ORGANOID_DATASETS:
        return np.nan
    
    tp = str(timepoint).lower().strip()
    
    # Gestational weeks (GW or "Nth week post-fertilization")
    match = re.search(r'(\d+)(?:th|st|nd|rd)?\s*week\s*post-fertilization', tp)
    if match:
        weeks = int(match.group(1))
        return weeks * 7
    
    match = re.search(r'gw\s*(\d+)', tp)
    if match:
        weeks = int(match.group(1))
        return weeks * 7
    
    # HsapDv ontology terms
    hsapdv_map = {
        'hsapdv:0000099': 9 * 7,
        'hsapdv:0000100': 10 * 7,
        'hsapdv:0000101': 11 * 7,
        'hsapdv:0000102': 12 * 7,
        'hsapdv:0000103': 13 * 7,
        'hsapdv:0000104': 14 * 7,
        'hsapdv:0000105': 15 * 7,
        'hsapdv:0000106': 16 * 7,
        'hsapdv:0000107': 17 * 7,
        'hsapdv:0000108': 18 * 7,
    }
    for key, days in hsapdv_map.items():
        if key in tp:
            return days
    
    # Carnegie stages
    match = re.search(r'carnegie\s*stage\s*(\d+)', tp)
    if match:
        cs = int(match.group(1))
        carnegie_days = {
            1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7,
            8: 18, 9: 20, 10: 22, 11: 24, 12: 26, 13: 28,
            14: 32, 15: 36, 16: 40, 17: 42, 18: 44, 19: 46,
            20: 50, 21: 52, 22: 54, 23: 56
        }
        return carnegie_days.get(cs, cs * 2.5)
    
    # Postnatal ages (years, months) - add 280 days for gestation
    match = re.search(r'(\d+)\s*-?\s*year\s*-?\s*old', tp)
    if match:
        years = int(match.group(1))
        return 280 + years * 365
    
    match = re.search(r'(\d+)\s*yo', tp)
    if match:
        years = int(match.group(1))
        return 280 + years * 365
    
    match = re.search(r'(\d+)\s*-?\s*month\s*-?\s*old', tp)
    if match:
        months = int(match.group(1))
        return 280 + months * 30
    
    # LMP month stages
    lmp_map = {
        'fourth lmp month': 4 * 30,
        'fifth lmp month': 5 * 30,
        'sixth lmp month': 6 * 30,
        'seventh lmp month': 7 * 30,
        'eighth lmp month': 8 * 30,
        'ninth lmp month': 9 * 30,
    }
    for key, days in lmp_map.items():
        if key in tp:
            return days
    
    # Decade stages
    decade_map = {
        'fourth decade': 280 + 35 * 365,
        'fifth decade': 280 + 45 * 365,
        'sixth decade': 280 + 55 * 365,
        'seventh decade': 280 + 65 * 365,
    }
    for key, days in decade_map.items():
        if key in tp:
            return days
    
    # Special stages
    if 'blastula' in tp:
        return 5
    if 'newborn' in tp or 'infant' in tp:
        return 280
    
    return np.nan


def parse_mouse_time(timepoint):
    """
    Convert mouse timepoint to days post-conception.
    
    Excludes disease models (APOE4/TREM2, 5xFAD) - returns NaN.
    
    Mouse gestation: ~20 days
    """
    if pd.isna(timepoint) or timepoint == 'unknown' or timepoint == '':
        return np.nan
    
    tp = str(timepoint).strip()
    
    # Exclude disease models
    if tp in MOUSE_DISEASE_MODELS:
        return np.nan
    
    tp_lower = tp.lower()
    
    # Embryonic days (E notation)
    match = re.search(r'e(\d+\.?\d*)', tp_lower)
    if match:
        e_day = float(match.group(1))
        return e_day
    
    # Postnatal months
    match = re.search(r'(\d+)\s*mo(?:nth)?s?', tp_lower)
    if match:
        months = int(match.group(1))
        return 20 + months * 30  # 20 days gestation + postnatal
    
    # Adult/aged keywords (Jin 2025)
    if tp_lower == 'adult':
        return 20 + 2 * 30  # ~2 months postnatal = 80 days
    if tp_lower == 'aged':
        return 20 + 18 * 30  # ~18 months postnatal = 560 days
    
    return np.nan


def parse_zebrafish_time(timepoint):
    """
    Convert zebrafish timepoint to hours post-fertilization.
    """
    if pd.isna(timepoint) or timepoint == 'unknown' or timepoint == '':
        return np.nan
    
    tp = str(timepoint).lower().strip()
    
    # Somite stages (e.g., "10s" = 10-somite stage)
    match = re.search(r'^(\d+)s$', tp)
    if match:
        somites = int(match.group(1))
        return 10 + somites * 0.5  # hpf
    
    # Hours post-fertilization
    match = re.search(r'(\d+)\s*hpf', tp)
    if match:
        return int(match.group(1))
    
    # Days post-fertilization
    match = re.search(r'(\d+)\s*dpf', tp)
    if match:
        days = int(match.group(1))
        return days * 24
    
    return np.nan


def parse_drosophila_time(timepoint):
    """
    Convert Drosophila timepoint to days post-eclosion.
    """
    if pd.isna(timepoint) or timepoint == 'unknown' or timepoint == '':
        return np.nan
    
    tp = str(timepoint).lower().strip()
    
    match = re.search(r'^(\d+)d$', tp)
    if match:
        return int(match.group(1))
    
    return np.nan


def normalize_time(timepoint, organism, dataset=None):
    """Dispatch to appropriate parser based on organism."""
    # Ages already computed from ontology terms (CELLxGENE datasets), e.g. "98 dpc", "16 hpf"
    tagged = parse_tagged_age(timepoint)
    if tagged is not None:
        return tagged
    if organism == 'Human':
        return parse_human_time(timepoint, dataset)
    elif organism == 'Mouse':
        return parse_mouse_time(timepoint)
    elif organism == 'Zebrafish':
        return parse_zebrafish_time(timepoint)
    elif organism == 'Drosophila':
        return parse_drosophila_time(timepoint)
    else:
        return np.nan


# =============================================================================
# MAIN
# =============================================================================

print("=" * 60)
print("Normalizing timepoints to numeric values")
print("=" * 60)

# Process pseudobulk metadata first (smaller, faster to test)
print("\n1. Processing pseudobulk metadata...")

meta = pd.read_csv(PSEUDOBULK_META)
print(f"   Rows: {len(meta)}")

# Apply normalization for non-organoid samples
meta['numeric_time'] = meta.apply(
    lambda row: normalize_time(row['timepoint'], row['organism'], row['dataset']), 
    axis=1
)

# Add sample_type column (in_vivo vs organoid)
meta['sample_type'] = meta['dataset'].apply(
    lambda x: 'organoid' if x in ORGANOID_DATASETS else 'in_vivo'
)

# For organoids, we need to get organoid_age_days from the h5ad
# (pseudobulk meta doesn't have it - need to aggregate from cells)
print("\n   Note: Organoid time requires organoid_age_days from h5ad")
print("   Will be filled in during h5ad processing...")

# Mark excluded samples
meta['excluded'] = False
meta.loc[meta['timepoint'].isin(MOUSE_DISEASE_MODELS), 'excluded'] = True
n_excluded = meta['excluded'].sum()
print(f"\n   Excluded samples (disease models): {n_excluded}")

# Summary
print("\n   Results by organism:")
for org in meta['organism'].unique():
    org_meta = meta[meta['organism'] == org]
    org_meta_incl = org_meta[~org_meta['excluded']]
    n_total = len(org_meta_incl)
    n_valid = org_meta_incl['numeric_time'].notna().sum()
    pct = 100 * n_valid / n_total if n_total > 0 else 0
    
    print(f"\n   {org}:")
    if org_meta['excluded'].any():
        print(f"     Excluded: {org_meta['excluded'].sum()} samples (disease models)")
    print(f"     Coverage: {n_valid}/{n_total} ({pct:.0f}%)")
    
    # Show organoid vs in_vivo breakdown for Human
    if org == 'Human':
        for st in ['in_vivo', 'organoid']:
            st_meta = org_meta_incl[org_meta_incl['sample_type'] == st]
            if len(st_meta) > 0:
                st_valid = st_meta['numeric_time'].notna().sum()
                st_pct = 100 * st_valid / len(st_meta) if len(st_meta) > 0 else 0
                print(f"       {st}: {st_valid}/{len(st_meta)} ({st_pct:.0f}%)")
                
                if st_valid > 0:
                    valid_times = st_meta[st_meta['numeric_time'].notna()]
                    if st == 'in_vivo':
                        print(f"         Range: {valid_times['numeric_time'].min():.0f} - {valid_times['numeric_time'].max():.0f} days post-conception")
                    else:
                        print(f"         Range: {valid_times['numeric_time'].min():.0f} - {valid_times['numeric_time'].max():.0f} days differentiation")
                elif st == 'organoid':
                    print(f"         (will use organoid_age_days from h5ad)")
    
    if n_valid > 0:
        valid_times = org_meta_incl[org_meta_incl['numeric_time'].notna()]
        time_range = valid_times.groupby('timepoint')['numeric_time'].first()
        print(f"     Range: {time_range.min():.1f} - {time_range.max():.1f}")
        print(f"     Examples: {dict(list(time_range.items())[:5])}")

# Save (without organoid time for now - will update after h5ad)
meta.to_csv(PSEUDOBULK_META_OUT, index=False)
print(f"\n   Saved: {PSEUDOBULK_META_OUT}")

# Process h5ad
print("\n2. Processing h5ad file...")
try:
    adata = ad.read_h5ad(INPUT_H5AD)
    print(f"   Shape: {adata.shape}")
    
    # Apply normalization for non-organoid samples
    adata.obs['numeric_time'] = adata.obs.apply(
        lambda row: normalize_time(row['merged_time'], row['organism'], row['dataset']), 
        axis=1
    )
    
    # For organoids, use organoid_age_days directly (only He 2024 has this)
    if 'organoid_age_days' in adata.obs.columns:
        for ds in ORGANOID_DATASETS_WITH_TIME:
            ds_mask = adata.obs['dataset'] == ds
            if ds_mask.any():
                # Use organoid_age_days
                org_ages = pd.to_numeric(adata.obs.loc[ds_mask, 'organoid_age_days'], errors='coerce')
                adata.obs.loc[ds_mask, 'numeric_time'] = org_ages
                
                n_valid = org_ages.notna().sum()
                print(f"   {ds}: using organoid_age_days ({n_valid:,}/{ds_mask.sum():,} valid)")
                if n_valid > 0:
                    print(f"     Range: {org_ages.min():.0f} - {org_ages.max():.0f} days differentiation")
    
    # Note about Wang 2022
    wang_mask = adata.obs['dataset'] == 'Wang (2022)'
    if wang_mask.any():
        print(f"   Wang (2022): no organoid_age_days available ({wang_mask.sum():,} cells)")
        print(f"     Will be excluded from developmental time analysis")
    
    # Add sample_type column
    adata.obs['sample_type'] = adata.obs['dataset'].apply(
        lambda x: 'organoid' if x in ORGANOID_DATASETS else 'in_vivo'
    )
    
    # Mark excluded (disease models)
    adata.obs['excluded'] = adata.obs['merged_time'].isin(MOUSE_DISEASE_MODELS)
    n_excluded = adata.obs['excluded'].sum()
    print(f"\n   Excluded cells (disease models): {n_excluded:,}")
    
    # Summary
    print("\n   Results:")
    for org in adata.obs['organism'].unique():
        org_mask = (adata.obs['organism'] == org) & (~adata.obs['excluded'])
        n_total = org_mask.sum()
        n_valid = adata.obs.loc[org_mask, 'numeric_time'].notna().sum()
        pct = 100 * n_valid / n_total if n_total > 0 else 0
        print(f"     {org}: {n_valid:,}/{n_total:,} ({pct:.0f}%)")
    
    # Save
    adata.write(OUTPUT_H5AD)
    print(f"\n   Saved: {OUTPUT_H5AD}")
    
    # Now update pseudobulk meta with aggregated organoid times
    print("\n3. Updating pseudobulk meta with organoid times...")
    
    # Reload pseudobulk meta
    meta = pd.read_csv(PSEUDOBULK_META_OUT)
    
    # For each organoid pseudobulk sample (only He 2024), get median organoid_age_days
    for ds in ORGANOID_DATASETS_WITH_TIME:
        ds_mask_meta = meta['dataset'] == ds
        if not ds_mask_meta.any():
            continue
            
        # Get cells for this dataset
        ds_mask_adata = adata.obs['dataset'] == ds
        ds_obs = adata.obs.loc[ds_mask_adata]
        
        # Aggregate by pseudobulk grouping
        if 'organoid_age_days' in ds_obs.columns:
            # Group by the same keys used in pseudobulk
            for idx, row in meta[ds_mask_meta].iterrows():
                # Find matching cells
                cell_mask = (
                    (ds_obs['dataset'] == row['dataset']) &
                    (ds_obs['predicted_labels'] == row['cell_type'])
                )
                if 'merged_sample' in ds_obs.columns and 'sample' in row:
                    # Add sample matching if available
                    pass
                
                if cell_mask.any():
                    org_ages = pd.to_numeric(ds_obs.loc[cell_mask, 'organoid_age_days'], errors='coerce')
                    median_age = org_ages.median()
                    if pd.notna(median_age):
                        meta.loc[idx, 'numeric_time'] = median_age
    
    # Report organoid coverage
    for ds in ORGANOID_DATASETS:
        ds_meta = meta[meta['dataset'] == ds]
        n_valid = ds_meta['numeric_time'].notna().sum()
        if ds in ORGANOID_DATASETS_WITH_TIME:
            print(f"   {ds}: {n_valid}/{len(ds_meta)} samples with organoid_age_days")
            if n_valid > 0:
                print(f"     Range: {ds_meta['numeric_time'].min():.0f} - {ds_meta['numeric_time'].max():.0f} days")
        else:
            print(f"   {ds}: no organoid_age_days available (excluded from time analysis)")
    
    # Save final version
    meta.to_csv(PSEUDOBULK_META_OUT, index=False)
    print(f"\n   Updated: {PSEUDOBULK_META_OUT}")
    
except Exception as e:
    import traceback
    print(f"   Error: {e}")
    traceback.print_exc()
    print("   Skipping h5ad processing")

print("\n" + "=" * 60)
print("Done!")
print("=" * 60)
