#!/usr/bin/env python3
"""
Build the app's data folder (wide-format parquets) from the post-processing outputs
===================================================================================

From sfari_postprocess.py (step 3, "Generate parquets"), the script that built the app data of
earlier releases. The tables and their definitions are unchanged:

  expression_meta / _mean / _pct   one row per species x dataset x cell type x sample type;
                                   mean corrected log2 CPM, fraction of pseudobulk replicates with counts
  temporal_meta / _mean            the same per time point, with numeric_time, relative_dev_time,
                                   unified_stage, time_bin, time_order
  stage_mapping, gene_map, celltype_meta, risk_genes, dataset_overview, dataset_info,
  summary_statistics, umap_subsample

Inputs ($SFARI_EXCHANGE_DIR): pseudobulk_meta_numeric_time.csv (normalize_age.py), corrected_expression.csv
(correct_within_org.R), pseudobulk_counts.csv (prep_sfari_data_v8.py --stage 1); data/concord/
umap_subsample.parquet and build_info.json (step 06); data/cellxgene/dataset_references.json.
Output: $SFARI_ROOT/data/app/ (--outdir) - copy its contents into the Hugging Face Space's data/.

Changes from sfari_postprocess.py: paths from SFARI_ROOT; relative_dev_time and unified_stage are
computed here from numeric_time (normalize_age.py does not add them); sample type comes from the
metadata (CELLxGENE organoids included); UMAP from CONCORD; gene columns in a fixed order; the
CONCORD provenance and dataset references are copied alongside.

    python pipeline/build_app_data.py [--outdir DIR]
"""

import argparse
import os
import shutil

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

SFARI_ROOT = os.environ.get('SFARI_ROOT', '/sc/arion/projects/ad-omics/raphael/SFARI')
EXCHANGE_DIR = os.environ.get('SFARI_EXCHANGE_DIR', f'{SFARI_ROOT}/data/r_exchange')
OUTPUT_DIR = f'{SFARI_ROOT}/data/app'
CONCORD_DIR = f'{SFARI_ROOT}/data/concord'
CELLXGENE_DIR = os.environ.get('SFARI_CELLXGENE_DIR', f'{SFARI_ROOT}/data/cellxgene')
REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RISK_GENES_CSV = os.path.join(REPO_DIR, 'resources', 'SFARI-Gene_genes_07-08-2025release_10-08-2025export.csv')
UMAP_SUBSAMPLE_N = 200_000

UNIFIED_STAGE_ORDER = {
    'Early Development': 1, 'Mid Development': 2, 'Late Development': 3,
    'Postnatal': 4, 'Juvenile': 5, 'Adult': 6, 'Aged': 7
}

# =============================================================================
# RELATIVE DEVELOPMENTAL TIME (0-1 SCALE)
# =============================================================================
# Based on Workman et al. (2013) and comparative neurodevelopment literature
# 0.0 = conception/fertilization, 0.45 = birth/hatching, 1.0 = adult maturity, >1.0 = aging
# numeric_time units (normalize_age.py, cellxgene/stages.py): human and mouse days post-conception
# (birth = 280 and 20), human organoids days in culture, zebrafish hpf, Drosophila days post-eclosion.
# Drosophila data (Davie 2018) is all post-eclosion adult, so it maps to 1.0+.


def get_relative_dev_time(species, numeric_time, sample_type='in_vivo'):
    """Species-specific numeric time -> relative developmental time; None if unknown."""
    if pd.isna(numeric_time):
        return None
    nt = float(numeric_time)
    if species == 'Human':
        if sample_type == 'organoid':
            # Organoid days -> approximate developmental equivalence (Velasco et al., Gordon et al.)
            # Day 30 ~ GW8-10, Day 60 ~ GW12-14, Day 100 ~ GW16-18, Day 180 ~ GW22+
            if nt <= 30:
                return 0.05 + (nt / 30) * 0.05
            elif nt <= 60:
                return 0.10 + ((nt - 30) / 30) * 0.08
            elif nt <= 100:
                return 0.18 + ((nt - 60) / 40) * 0.10
            elif nt <= 180:
                return 0.28 + ((nt - 100) / 80) * 0.12
            else:
                return 0.40 + ((nt - 180) / 180) * 0.10
        # In vivo: conception to birth = 280 days, maturity ~25 years
        if nt <= 280:
            return (nt / 280) * 0.45
        postnatal_years = (nt - 280) / 365
        if postnatal_years <= 2:
            return 0.45 + (postnatal_years / 2) * 0.10
        elif postnatal_years <= 12:
            return 0.55 + ((postnatal_years - 2) / 10) * 0.15
        elif postnatal_years <= 25:
            return 0.70 + ((postnatal_years - 12) / 13) * 0.30
        else:
            return 1.0 + ((postnatal_years - 25) / 50) * 0.50
    elif species == 'Mouse':
        # Gestation ~20 days, maturity ~8 weeks (P56)
        if nt <= 20:
            return (nt / 20) * 0.45
        postnatal = nt - 20
        if postnatal <= 7:
            return 0.45 + (postnatal / 7) * 0.10
        elif postnatal <= 21:
            return 0.55 + ((postnatal - 7) / 14) * 0.15
        elif postnatal <= 56:
            return 0.70 + ((postnatal - 21) / 35) * 0.30
        elif postnatal <= 365:
            return 1.0 + ((postnatal - 56) / 309) * 0.30
        else:
            return 1.30 + ((postnatal - 365) / 365) * 0.20
    elif species == 'Zebrafish':
        # Hatching ~72 hpf, juvenile ~30 dpf, adult ~90 dpf
        if nt <= 24:
            return (nt / 24) * 0.15
        elif nt <= 48:
            return 0.15 + ((nt - 24) / 24) * 0.15
        elif nt <= 72:
            return 0.30 + ((nt - 48) / 24) * 0.15
        elif nt <= 120:
            return 0.45 + ((nt - 72) / 48) * 0.10
        elif nt <= 720:
            return 0.55 + ((nt - 120) / 600) * 0.25
        elif nt <= 2160:
            return 0.80 + ((nt - 720) / 1440) * 0.20
        else:
            return 1.0 + ((nt - 2160) / 2160) * 0.30
    elif species == 'Drosophila':
        if nt <= 7:
            return 1.0 + (nt / 7) * 0.10
        elif nt <= 30:
            return 1.10 + ((nt - 7) / 23) * 0.20
        else:
            return 1.30 + ((nt - 30) / 30) * 0.20
    return None


def get_unified_stage(relative_time):
    """Relative developmental time -> (unified stage, order)."""
    if relative_time is None or pd.isna(relative_time):
        return None, 99
    rt = float(relative_time)
    if rt <= 0.15:
        return 'Early Development', 1
    elif rt <= 0.30:
        return 'Mid Development', 2
    elif rt <= 0.45:
        return 'Late Development', 3
    elif rt <= 0.60:
        return 'Postnatal', 4
    elif rt <= 0.80:
        return 'Juvenile', 5
    elif rt <= 1.15:
        return 'Adult', 6
    else:
        return 'Aged', 7


def create_time_bin(species, numeric_time, sample_type='in_vivo'):
    """Human-readable time bin and its order within the species."""
    if pd.isna(numeric_time):
        return None, 99
    nt = float(numeric_time)
    if species == 'Human':
        if sample_type == 'organoid':
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
    elif species == 'Mouse':
        if nt <= 12: return 'Early embryo (E<12)', 1
        elif nt <= 16: return 'Mid embryo (E12-16)', 2
        elif nt <= 20: return 'Late embryo (E16-20)', 3
        elif nt <= 50: return 'Neonatal (P0-P30)', 4
        elif nt <= 110: return 'Juvenile (1-3mo)', 5
        elif nt <= 385: return 'Adult (3-12mo)', 6
        else: return 'Aged (>12mo)', 7
    elif species == 'Zebrafish':
        if nt <= 24: return '0-24 hpf', 1
        elif nt <= 48: return '24-48 hpf', 2
        elif nt <= 72: return '48-72 hpf', 3
        elif nt <= 120: return '72-120 hpf (5dpf)', 4
        else: return '>5 dpf', 5
    elif species == 'Drosophila':
        if nt == 0: return 'Day 0 (eclosion)', 1
        elif nt <= 1: return 'Day 1', 2
        elif nt <= 3: return 'Day 3', 3
        elif nt <= 6: return 'Day 6', 4
        elif nt <= 9: return 'Day 9', 5
        elif nt <= 15: return 'Day 15', 6
        elif nt <= 30: return 'Day 30', 7
        else: return 'Day 50', 8
    return None, 99


def write_parquet(df, path):
    """Write dataframe to parquet with zstd compression."""
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, path, compression='zstd', compression_level=9)
    size_mb = os.path.getsize(path) / 1024 / 1024
    print(f"  Wrote: {os.path.basename(path)} ({len(df):,} rows, {size_mb:.2f} MB)")


def read_matrix(path, dtype=np.float64):
    """Samples x genes CSV (first column = sample id)."""
    genes = pd.read_csv(path, nrows=0).columns[1:]
    return pd.read_csv(path, index_col=0, dtype={g: dtype for g in genes})


def add_developmental_time(pb_meta):
    """relative_dev_time and unified_stage from numeric_time, where not already present."""
    if 'relative_dev_time' not in pb_meta.columns:
        pb_meta['relative_dev_time'] = [
            get_relative_dev_time(o, t, s) for o, t, s in
            zip(pb_meta['organism'], pb_meta['numeric_time'], pb_meta['sample_type'])]
    if 'unified_stage' not in pb_meta.columns:
        unified = pb_meta['relative_dev_time'].apply(get_unified_stage)
        pb_meta['unified_stage'] = [u[0] for u in unified]
        pb_meta['unified_stage_order'] = [u[1] for u in unified]
    return pb_meta


def build(exchange_dir, parquet_dir):
    print("=" * 60)
    print("Build app data (wide-format parquets)")
    print("=" * 60)
    os.makedirs(parquet_dir, exist_ok=True)

    # Load expression
    corrected_path = os.path.join(exchange_dir, 'corrected_expression.csv')
    if not os.path.exists(corrected_path):
        raise FileNotFoundError(f"Missing: {corrected_path}\nRun correct_within_org.R first.")
    expr_matrix = read_matrix(corrected_path)
    normalization = 'voom_combat'
    print(f"\nLoaded corrected expression: {expr_matrix.shape}")

    meta_path = os.path.join(exchange_dir, 'pseudobulk_meta_numeric_time.csv')
    if not os.path.exists(meta_path):
        meta_path = os.path.join(exchange_dir, 'pseudobulk_meta.csv')
    pb_meta = pd.read_csv(meta_path)

    # Raw counts for pct_expressing (only compared with 0, so float32 is exact enough and halves memory)
    pb_counts = read_matrix(os.path.join(exchange_dir, 'pseudobulk_counts.csv'), np.float32)

    pb_meta = pb_meta[pb_meta['sample_id'].isin(expr_matrix.index)].copy()
    genes = list(expr_matrix.columns)
    print(f"Samples: {len(pb_meta)}, Genes: {len(genes)}")
    if 'sample_type' not in pb_meta.columns:
        raise KeyError(f"{meta_path} has no sample_type column (normalize_age.py adds it)")
    pb_meta = add_developmental_time(pb_meta)
    organoid = set(pb_meta.loc[pb_meta['sample_type'] == 'organoid', 'dataset'])

    # ==========================================================================
    # Expression summaries - wide: rows = organism x dataset x cell_type x sample_type, columns = genes
    # ==========================================================================
    print("\nGenerating expression summaries...")
    group_cols = ['organism', 'dataset', 'cell_type', 'sample_type']
    mean_expr_rows, pct_expr_rows, meta_rows = [], [], []
    for group_key, group in pb_meta.groupby(group_cols, dropna=False):
        org, ds, ct, st = group_key
        valid_ids = [s for s in group['sample_id'] if s in expr_matrix.index]
        if not valid_ids:
            continue
        mean_expr = expr_matrix.loc[valid_ids].mean(axis=0)
        raw_ids = [s for s in valid_ids if s in pb_counts.index]
        if raw_ids:
            pct_expr = (pb_counts.loc[raw_ids] > 0).mean(axis=0)
        else:
            pct_expr = (expr_matrix.loc[valid_ids] > 0).mean(axis=0)
        row_id = f"{org}|{ds}|{ct}|{st}"
        meta_rows.append({'row_id': row_id, 'species': org, 'tissue': ds, 'cell_type': ct,
                          'sample_type': st, 'n_cells': int(group['n_cells'].sum())})
        mean_expr_rows.append({'row_id': row_id, **mean_expr.to_dict()})
        pct_expr_rows.append({'row_id': row_id, **pct_expr.to_dict()})

    meta_df = pd.DataFrame(meta_rows)
    mean_expr_wide = pd.DataFrame(mean_expr_rows).set_index('row_id').astype('float32')
    pct_expr_wide = pd.DataFrame(pct_expr_rows).set_index('row_id').astype('float32')
    print(f"  Groups: {len(meta_df)}, Genes: {len(mean_expr_wide.columns)}")
    write_parquet(meta_df, os.path.join(parquet_dir, 'expression_meta.parquet'))
    write_parquet(mean_expr_wide.reset_index(), os.path.join(parquet_dir, 'expression_mean.parquet'))
    write_parquet(pct_expr_wide.reset_index(), os.path.join(parquet_dir, 'expression_pct.parquet'))

    # A gene is "expressed" if mean > 0.1 OR pct > 1% in ANY group
    pct_same = pct_expr_wide.reindex(columns=mean_expr_wide.columns)
    expressed_mask = (mean_expr_wide > 0.1) | (pct_same > 0.01)
    expressed_genes = [g for g in genes if g in set(expressed_mask.columns[expressed_mask.any(axis=0)])]
    print(f"  Expressed genes: {len(expressed_genes):,}")

    # ==========================================================================
    # Temporal expression - wide, with the time point dimension
    # ==========================================================================
    print("\nGenerating temporal expression...")
    temporal_group_cols = ['organism', 'dataset', 'cell_type', 'timepoint', 'sample_type']
    temporal_meta_rows, temporal_mean_rows = [], []
    for group_key, group in pb_meta.groupby(temporal_group_cols, dropna=False):
        org, ds, ct, tp, st = group_key
        if pd.isna(tp) or str(tp).lower() in ['unknown', 'nan', '']:
            continue
        numeric_time = group['numeric_time'].iloc[0]
        relative_dev_time = group['relative_dev_time'].iloc[0]
        unified_stage = group['unified_stage'].iloc[0]
        if pd.isna(relative_dev_time):  # no cross-species comparison possible
            continue
        valid_ids = [s for s in group['sample_id'] if s in expr_matrix.index]
        if not valid_ids:
            continue
        mean_expr = expr_matrix.loc[valid_ids].mean(axis=0)
        time_bin, time_order = create_time_bin(org, numeric_time, st)
        row_id = f"{org}|{ds}|{ct}|{tp}|{st}"
        temporal_meta_rows.append({
            'row_id': row_id, 'species': org, 'tissue': ds, 'cell_type': ct, 'timepoint': tp,
            'sample_type': st, 'n_cells': int(group['n_cells'].sum()), 'numeric_time': numeric_time,
            'relative_dev_time': relative_dev_time, 'unified_stage': unified_stage,
            'time_bin': time_bin, 'time_order': time_order})
        temporal_mean_rows.append({'row_id': row_id, **mean_expr[expressed_genes].to_dict()})

    temporal_meta_df = pd.DataFrame(temporal_meta_rows)
    temporal_meta_df = temporal_meta_df[temporal_meta_df['unified_stage'].notna()]
    temporal_mean_wide = pd.DataFrame(temporal_mean_rows).set_index('row_id')
    temporal_mean_wide = temporal_mean_wide.loc[temporal_meta_df['row_id']].astype('float32')
    print(f"  Temporal groups: {len(temporal_meta_df)}, Genes: {len(temporal_mean_wide.columns)}")
    temporal_meta_df['relative_dev_time'] = temporal_meta_df['relative_dev_time'].astype('float32')
    temporal_meta_df['time_order'] = temporal_meta_df['time_order'].astype('int16')
    write_parquet(temporal_meta_df, os.path.join(parquet_dir, 'temporal_meta.parquet'))
    write_parquet(temporal_mean_wide.reset_index(), os.path.join(parquet_dir, 'temporal_mean.parquet'))

    print("\n  Temporal data summary:")
    for sp in sorted(temporal_meta_df['species'].unique()):
        sp_df = temporal_meta_df[temporal_meta_df['species'] == sp]
        print(f"    {sp}: {sp_df['time_bin'].nunique()} time bins, relative_dev_time="
              f"{sp_df['relative_dev_time'].min():.2f}-{sp_df['relative_dev_time'].max():.2f}; "
              f"stages {sorted(sp_df['unified_stage'].unique(), key=UNIFIED_STAGE_ORDER.get)}")

    # ==========================================================================
    # Other tables
    # ==========================================================================
    print("\nGenerating other tables...")
    gene_list = list(mean_expr_wide.columns)
    gene_map = pd.DataFrame([{'species': org, 'gene_native': g, 'gene_human': g}
                             for org in pb_meta['organism'].unique() for g in gene_list])
    write_parquet(gene_map, os.path.join(parquet_dir, 'gene_map.parquet'))

    celltype_meta = pb_meta.groupby(['organism', 'dataset', 'cell_type']).agg({'n_cells': 'sum'}).reset_index()
    celltype_meta.columns = ['species', 'tissue', 'cell_type', 'n_cells']
    write_parquet(celltype_meta, os.path.join(parquet_dir, 'celltype_meta.parquet'))

    if os.path.exists(RISK_GENES_CSV):
        risk_df = pd.read_csv(RISK_GENES_CSV).rename(columns={'gene-symbol': 'gene_symbol', 'gene-score': 'gene_score'})
    else:
        print(f"  Warning: {RISK_GENES_CSV} not found; risk_genes.parquet is empty")
        risk_df = pd.DataFrame(columns=['gene_symbol', 'gene_score'])
    write_parquet(risk_df, os.path.join(parquet_dir, 'risk_genes.parquet'))

    overview = []
    for ds in sorted(pb_meta['dataset'].unique()):
        ds_meta = pb_meta[pb_meta['dataset'] == ds]
        tps = [t for t in ds_meta['timepoint'].dropna().unique() if str(t).lower() not in ['unknown', 'nan', '']]
        overview.append({
            'Dataset': ds, 'Species': ds_meta['organism'].iloc[0],
            'Samples': ds_meta['sample'].nunique() if 'sample' in ds_meta.columns else len(ds_meta),
            'Cell Types': ds_meta['cell_type'].nunique(), 'Timepoints': len(tps),
            'Sample Type': 'organoid' if ds in organoid else 'in_vivo',
            'Total Cells': int(ds_meta['n_cells'].sum())})
    write_parquet(pd.DataFrame(overview), os.path.join(parquet_dir, 'dataset_overview.parquet'))
    dataset_info = pd.DataFrame(overview)[['Dataset', 'Species', 'Sample Type', 'Total Cells']] \
        .rename(columns={'Sample Type': 'sample_type', 'Total Cells': 'n_cells'})
    write_parquet(dataset_info, os.path.join(parquet_dir, 'dataset_info.parquet'))

    summary_stats = pd.DataFrame([{
        'pseudobulk_samples': len(pb_meta), 'genes': len(genes), 'datasets': pb_meta['dataset'].nunique(),
        'cell_types': pb_meta['cell_type'].nunique(), 'organisms': pb_meta['organism'].nunique(),
        'normalization': normalization}])
    write_parquet(summary_stats, os.path.join(parquet_dir, 'summary_statistics.parquet'))

    # Stage mapping (cross-species reference); unified_stage_order is the species' time_order
    stage_cols = ['species', 'time_bin', 'relative_dev_time', 'unified_stage', 'time_order']
    stage_mapping_df = (temporal_meta_df[stage_cols].drop_duplicates()
                        .rename(columns={'time_order': 'unified_stage_order'}))
    stage_mapping_df['unified_stage_order'] = stage_mapping_df['unified_stage_order'].astype(int)
    stage_mapping_df = stage_mapping_df.sort_values(['unified_stage_order', 'species', 'relative_dev_time'])
    write_parquet(stage_mapping_df.reset_index(drop=True), os.path.join(parquet_dir, 'stage_mapping.parquet'))

    # UMAP subsample (CONCORD, step 06)
    umap_path = os.path.join(CONCORD_DIR, 'umap_subsample.parquet')
    if os.path.exists(umap_path):
        umap_df = pd.read_parquet(umap_path)
        if len(umap_df) > UMAP_SUBSAMPLE_N:
            umap_df = umap_df.sample(n=UMAP_SUBSAMPLE_N, random_state=42).sort_index()
        write_parquet(umap_df.reset_index(drop=True), os.path.join(parquet_dir, 'umap_subsample.parquet'))
    else:
        print(f"  Skipping UMAP (not found: {umap_path})")

    for src in [os.path.join(CONCORD_DIR, 'build_info.json'), os.path.join(CELLXGENE_DIR, 'dataset_references.json')]:
        if os.path.exists(src):
            shutil.copy(src, parquet_dir)
            print(f"  Copied: {os.path.basename(src)}")
    print(f"\nApp data written to: {parquet_dir}")


if __name__ == '__main__':
    p = argparse.ArgumentParser(description="Build the app's wide-format parquets")
    p.add_argument('--exchange-dir', default=EXCHANGE_DIR)
    p.add_argument('--outdir', default=OUTPUT_DIR)
    a = p.parse_args()
    build(a.exchange_dir, a.outdir)
