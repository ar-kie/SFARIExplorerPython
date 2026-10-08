import os
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

# Source: CONCORD output of integrate_concord.py (predicted_labels, X_concord, X_umap; selected features only)
annotated_path = '/sc/arion/projects/ad-omics/raphael/SFARI/data/combined_concord_label_transfer.h5ad'

# Target: full gene file (has all genes but no annotations)
full_genes_path = '/sc/arion/projects/ad-omics/raphael/SFARI/pipeline_output/concatenated_annotated.h5ad'

# Output
output_h5ad = '/sc/arion/projects/ad-omics/raphael/SFARI/data/combined_annotated_full_concord_labels.h5ad'
output_parquet_dir = '/sc/arion/projects/ad-omics/raphael/SFARI/data/parquet/'

# Parquet config
SPECIES_COL = "organism"
DATASET_COL = "dataset"
CELLTYPE_COL = "predicted_labels"  # Use predicted labels, or 'supercategories' for original
LAYER = None  # Use .X

RISK_GENES_CSV = "/sc/arion/projects/ad-omics/raphael/SFARI/SFARI_genes/SFARI-Gene_genes_07-08-2025release_10-08-2025export.csv"

# =============================================================================
# STEP 1: Load annotated file (CONCORD processed)
# =============================================================================

print("=" * 60)
print("STEP 1: Loading annotated file")
print("=" * 60)

adata_annotated = ad.read_h5ad(annotated_path)
print(f"  Annotated shape: {adata_annotated.shape}")
print(f"  Obs columns: {adata_annotated.obs.columns.tolist()}")
print(f"  Obsm keys: {list(adata_annotated.obsm.keys())}")
print(f"  Uns keys: {list(adata_annotated.uns.keys())}")

# Store all annotations
obs_annotated = adata_annotated.obs.copy()
obsm_annotated = {k: v.copy() for k, v in adata_annotated.obsm.items()}
uns_annotated = dict(adata_annotated.uns)
obsp_annotated = {k: v.copy() for k, v in adata_annotated.obsp.items()} if adata_annotated.obsp else {}

# Get cell order
annotated_cells = adata_annotated.obs_names.tolist()

del adata_annotated
gc.collect()

# =============================================================================
# STEP 2: Load full gene file
# =============================================================================

print("\n" + "=" * 60)
print("STEP 2: Loading full gene file")
print("=" * 60)

adata_full = ad.read_h5ad(full_genes_path)
print(f"  Full shape: {adata_full.shape}")

# Check cell alignment
full_cells = adata_full.obs_names.tolist()

if annotated_cells == full_cells:
    print("  Cell order matches perfectly!")
else:
    # Check if same cells, different order
    if set(annotated_cells) == set(full_cells):
        print("  Same cells, reordering to match annotated file...")
        adata_full = adata_full[annotated_cells, :].copy()
    else:
        # Find common cells
        common = set(annotated_cells) & set(full_cells)
        print(f"  Warning: Cell mismatch!")
        print(f"    Annotated: {len(annotated_cells):,}")
        print(f"    Full: {len(full_cells):,}")
        print(f"    Common: {len(common):,}")
        
        if len(common) > 0:
            # Subset both to common cells
            common_list = [c for c in annotated_cells if c in common]
            adata_full = adata_full[common_list, :].copy()
            obs_annotated = obs_annotated.loc[common_list]
            obsm_annotated = {k: v[obs_annotated.index.get_indexer(common_list)] for k, v in obsm_annotated.items()}
        else:
            raise ValueError("No common cells found!")

# =============================================================================
# STEP 3: Fix Raj (2020) organism - should be Zebrafish
# =============================================================================

print("\n" + "=" * 60)
print("STEP 3: Fixing organism labels")
print("=" * 60)

# Check current organism distribution
print("  Current organism distribution:")
print(obs_annotated[SPECIES_COL].value_counts())

# Fix Raj (2020) - should be Zebrafish
raj_mask = obs_annotated[DATASET_COL] == 'Raj (2020)'
n_raj = raj_mask.sum()
print(f"\n  Raj (2020) cells: {n_raj:,}")

if n_raj > 0:
    current_organism = obs_annotated.loc[raj_mask, SPECIES_COL].unique()
    print(f"  Current organism for Raj (2020): {current_organism}")
    
    obs_annotated.loc[raj_mask, SPECIES_COL] = 'Zebrafish'
    print("  Changed to: Zebrafish")

# Also fix in full adata obs
if SPECIES_COL in adata_full.obs.columns:
    raj_mask_full = adata_full.obs[DATASET_COL] == 'Raj (2020)'
    adata_full.obs.loc[raj_mask_full, SPECIES_COL] = 'Zebrafish'

print("\n  Updated organism distribution:")
print(obs_annotated[SPECIES_COL].value_counts())

# =============================================================================
# STEP 4: Transfer all annotations to full gene file
# =============================================================================

print("\n" + "=" * 60)
print("STEP 4: Transferring annotations")
print("=" * 60)

# Transfer obs columns (add new ones, update existing)
print("  Transferring obs columns...")
for col in obs_annotated.columns:
    adata_full.obs[col] = obs_annotated[col].values

# Transfer obsm
print("  Transferring obsm...")
for key, val in obsm_annotated.items():
    print(f"    {key}: {val.shape}")
    adata_full.obsm[key] = val

# Transfer uns
print("  Transferring uns...")
for key, val in uns_annotated.items():
    print(f"    {key}")
    adata_full.uns[key] = val

# Transfer obsp
if obsp_annotated:
    print("  Transferring obsp...")
    for key, val in obsp_annotated.items():
        print(f"    {key}: {val.shape}")
        adata_full.obsp[key] = val

print(f"\n  Final shape: {adata_full.shape}")
print(f"  Final obs columns: {adata_full.obs.columns.tolist()}")
print(f"  Final obsm keys: {list(adata_full.obsm.keys())}")

# =============================================================================
# STEP 5: Clean and save annotated h5ad
# =============================================================================

print("\n" + "=" * 60)
print("STEP 5: Saving annotated h5ad")
print("=" * 60)

# Clean obs columns for h5ad compatibility
print("  Cleaning obs columns...")
for col in adata_full.obs.columns:
    dtype = adata_full.obs[col].dtype
    if dtype == bool or dtype == 'boolean':
        adata_full.obs[col] = adata_full.obs[col].astype(str)
    elif dtype == object:
        adata_full.obs[col] = adata_full.obs[col].astype(str)
    elif pd.api.types.is_extension_array_dtype(dtype):
        try:
            adata_full.obs[col] = adata_full.obs[col].astype(str)
        except:
            pass

print(f"  Saving to {output_h5ad}...")
adata_full.write(output_h5ad)
print("  Done!")

# =============================================================================
# STEP 6: Generate parquet files
# =============================================================================

print("\n" + "=" * 60)
print("STEP 6: Generating parquet files")
print("=" * 60)

os.makedirs(output_parquet_dir, exist_ok=True)

# Get matrix
def get_matrix(adata, layer=None):
    X = adata.layers[layer] if (layer is not None and layer in adata.layers) else adata.X
    if sparse.issparse(X):
        return X.tocsr()
    return np.asarray(X)

X = get_matrix(adata_full, LAYER)
gene_names = adata_full.var_names.values.astype(str)

# Compute expression summaries per group
print("  Computing expression summaries...")

gcols = [SPECIES_COL, DATASET_COL, CELLTYPE_COL]

# Check columns exist
for col in gcols:
    if col not in adata_full.obs.columns:
        print(f"  Warning: {col} not in obs, using 'Unknown'")
        adata_full.obs[col] = 'Unknown'

obs_small = adata_full.obs[gcols].astype(str).copy()
obs_small["_row"] = np.arange(adata_full.n_obs, dtype=int)

rows = []
meta_rows = []

groups = list(obs_small.groupby(gcols, sort=False))
n_groups = len(groups)
print(f"  Processing {n_groups} groups...")

for i, ((sp, ds, ctype), sub) in enumerate(groups):
    if i % 50 == 0:
        print(f"    Group {i+1}/{n_groups}: {sp} / {ds} / {ctype}")
    
    idx = sub["_row"].to_numpy()
    
    if sparse.issparse(X):
        sub_X = X[idx]
        mean_expr = np.asarray(sub_X.mean(axis=0)).ravel()
        pct_expr = np.asarray((sub_X > 0).mean(axis=0)).ravel()
        n_cells = sub_X.shape[0]
    else:
        sub_X = X[idx, :]
        mean_expr = sub_X.mean(axis=0)
        pct_expr = (sub_X > 0).mean(axis=0)
        n_cells = sub_X.shape[0]
    
    df = pd.DataFrame({
        "species": sp,
        "tissue": ds,  # dataset written as 'tissue'
        "cell_type": ctype,
        "gene_native": gene_names,
        "mean_expr": mean_expr.astype(np.float32),
        "pct_expressing": pct_expr.astype(np.float32),
        "n_cells": int(n_cells),
    })
    rows.append(df)
    meta_rows.append({"species": sp, "tissue": ds, "cell_type": ctype, "n_cells": n_cells})

expr_df = pd.concat(rows, ignore_index=True)
del rows
gc.collect()

# Gene map (identity since all are human symbols now)
print("  Creating gene map...")
gene_map_df = pd.DataFrame({
    "species": "Human",  # All genes are human symbols
    "gene_native": gene_names,
    "gene_human": gene_names
})
# Add mouse mappings if organism includes mouse
if 'Mouse' in adata_full.obs[SPECIES_COL].unique():
    mouse_genes = pd.DataFrame({
        "species": "Mouse",
        "gene_native": gene_names,
        "gene_human": gene_names  # Already humanized
    })
    gene_map_df = pd.concat([gene_map_df, mouse_genes], ignore_index=True)

# Add zebrafish if present
if 'Zebrafish' in adata_full.obs[SPECIES_COL].unique():
    zf_genes = pd.DataFrame({
        "species": "Zebrafish",
        "gene_native": gene_names,
        "gene_human": gene_names  # Already humanized
    })
    gene_map_df = pd.concat([gene_map_df, zf_genes], ignore_index=True)

gene_map_df = gene_map_df.drop_duplicates().sort_values(["species", "gene_native"])
expr_df["gene_human"] = expr_df["gene_native"]  # Identity mapping

# Cell type metadata
print("  Creating cell type metadata...")
cellmeta_df = pd.DataFrame(meta_rows).drop_duplicates().sort_values(["species", "tissue", "cell_type"])

# Risk genes
print("  Loading risk genes...")
if RISK_GENES_CSV and os.path.exists(RISK_GENES_CSV):
    risk_df = pd.read_csv(RISK_GENES_CSV)
else:
    risk_df = pd.DataFrame(columns=["gene_human", "panel_name", "source", "notes"])

# Write parquet files
print("\n  Writing parquet files...")

def write_parquet(df, path):
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, path)
    print(f"    {path}: {len(df):,} rows")

write_parquet(expr_df, os.path.join(output_parquet_dir, "expression_summaries.parquet"))
write_parquet(gene_map_df, os.path.join(output_parquet_dir, "gene_map.parquet"))
write_parquet(cellmeta_df, os.path.join(output_parquet_dir, "celltype_meta.parquet"))
write_parquet(risk_df, os.path.join(output_parquet_dir, "risk_genes.parquet"))

# =============================================================================
# STEP 7: Also save cell-level metadata as parquet (for browser filtering)
# =============================================================================

print("\n  Saving cell metadata...")
cell_meta_df = adata_full.obs.copy()
cell_meta_df.index.name = 'cell_id'
cell_meta_df = cell_meta_df.reset_index()

# Add UMAP coordinates if available
if 'X_umap' in adata_full.obsm:
    cell_meta_df['umap_1'] = adata_full.obsm['X_umap'][:, 0]
    cell_meta_df['umap_2'] = adata_full.obsm['X_umap'][:, 1]

write_parquet(cell_meta_df, os.path.join(output_parquet_dir, "cell_metadata.parquet"))

# =============================================================================
# DONE
# =============================================================================

print("\n" + "=" * 60)
print("DONE!")
print("=" * 60)
print(f"\nOutputs:")
print(f"  - {output_h5ad}")
print(f"  - {output_parquet_dir}expression_summaries.parquet")
print(f"  - {output_parquet_dir}gene_map.parquet")
print(f"  - {output_parquet_dir}celltype_meta.parquet")
print(f"  - {output_parquet_dir}risk_genes.parquet")
print(f"  - {output_parquet_dir}cell_metadata.parquet")
