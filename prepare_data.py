#!/usr/bin/env python3
"""
SFARI Explorer - Enhanced Data Preparation Pipeline

This script processes h5ad (AnnData) files into optimized parquet files
for the SFARI Gene Expression Explorer web application.

Features:
- Multi-species support with ortholog mapping
- Developmental timepoint normalization
- Expression summarization by cell type
- SFARI risk gene integration

Usage:
    python prepare_data.py --config config.yaml
    # or
    python prepare_data.py --input /path/to/data.h5ad --output ./data/
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import anndata as ad
from scipy import sparse
import pyarrow as pa
import pyarrow.parquet as pq


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class DatasetConfig:
    """Configuration for a single dataset."""
    path: str
    name: str
    species: str
    
    # Column mappings in .obs
    cell_type_col: str = "cell_type"
    timepoint_col: Optional[str] = None  # e.g., "age", "stage", "pseudotime"
    batch_col: Optional[str] = None
    
    # Gene symbol source
    gene_col: str = "var_names"  # or a column in .var
    
    # Expression layer
    layer: Optional[str] = None  # None = use .X
    
    # Developmental time mapping (species-specific)
    # Maps original timepoint values to normalized developmental time
    timepoint_mapping: Optional[Dict[str, float]] = None


@dataclass
class PipelineConfig:
    """Full pipeline configuration."""
    output_dir: str
    datasets: List[DatasetConfig]
    
    # Ortholog mapping file (optional)
    # Expected format: species, gene_native, gene_human
    ortholog_map_path: Optional[str] = None
    
    # SFARI risk genes file
    risk_genes_path: Optional[str] = None
    
    # Output options
    write_partitioned: bool = False
    min_cells_per_group: int = 10


# =============================================================================
# Developmental Time Mapping
# =============================================================================

# Standard developmental time scales (0-1 normalized)
# These map species-specific timepoints to a common developmental timeline

HUMAN_DEV_MAPPING = {
    # Carnegie stages / gestational weeks
    "CS12": 0.05, "CS13": 0.06, "CS14": 0.07, "CS15": 0.08,
    "CS16": 0.09, "CS17": 0.10, "CS18": 0.11, "CS19": 0.12,
    "CS20": 0.13, "CS21": 0.14, "CS22": 0.15, "CS23": 0.16,
    # Gestational weeks
    "GW5": 0.05, "GW6": 0.07, "GW7": 0.09, "GW8": 0.11,
    "GW9": 0.13, "GW10": 0.15, "GW11": 0.17, "GW12": 0.19,
    "GW13": 0.21, "GW14": 0.23, "GW15": 0.25, "GW16": 0.27,
    "GW17": 0.29, "GW18": 0.31, "GW19": 0.33, "GW20": 0.35,
    "GW21": 0.37, "GW22": 0.39, "GW23": 0.41, "GW24": 0.43,
    # Postnatal
    "Birth": 0.45, "P0": 0.45,
    "1mo": 0.50, "3mo": 0.55, "6mo": 0.60, "1yr": 0.65,
    "2yr": 0.70, "5yr": 0.75, "10yr": 0.80, "Adult": 1.0,
}

MOUSE_DEV_MAPPING = {
    # Embryonic days
    "E9.5": 0.10, "E10.5": 0.12, "E11.5": 0.15, "E12.5": 0.18,
    "E13.5": 0.22, "E14.5": 0.26, "E15.5": 0.30, "E16.5": 0.34,
    "E17.5": 0.38, "E18.5": 0.42,
    # Postnatal days
    "P0": 0.45, "P1": 0.46, "P2": 0.47, "P3": 0.48,
    "P4": 0.49, "P5": 0.50, "P7": 0.52, "P10": 0.55,
    "P14": 0.60, "P21": 0.70, "P28": 0.80, "P56": 0.90,
    "Adult": 1.0,
}

ZEBRAFISH_DEV_MAPPING = {
    # Hours post-fertilization
    "1hpf": 0.01, "2hpf": 0.02, "4hpf": 0.03, "6hpf": 0.04,
    "8hpf": 0.05, "10hpf": 0.07, "12hpf": 0.09, "14hpf": 0.11,
    "16hpf": 0.13, "18hpf": 0.15, "20hpf": 0.17, "24hpf": 0.20,
    "30hpf": 0.25, "36hpf": 0.30, "48hpf": 0.35, "72hpf": 0.45,
    # Days post-fertilization
    "5dpf": 0.55, "7dpf": 0.60, "14dpf": 0.70, "21dpf": 0.80,
    "Adult": 1.0,
}

DROSOPHILA_DEV_MAPPING = {
    # Embryonic stages
    "Stage1": 0.01, "Stage2": 0.02, "Stage3": 0.03, "Stage4": 0.04,
    "Stage5": 0.05, "Stage6": 0.06, "Stage7": 0.07, "Stage8": 0.08,
    "Stage9": 0.09, "Stage10": 0.10, "Stage11": 0.12, "Stage12": 0.14,
    "Stage13": 0.16, "Stage14": 0.18, "Stage15": 0.20, "Stage16": 0.22,
    "Stage17": 0.24,
    # Larval stages
    "L1": 0.30, "L2": 0.40, "L3": 0.50,
    # Pupal
    "Pupa": 0.70,
    # Adult
    "Adult": 1.0,
}

SPECIES_TIME_MAPPINGS = {
    "Human": HUMAN_DEV_MAPPING,
    "Mouse": MOUSE_DEV_MAPPING,
    "Zebrafish": ZEBRAFISH_DEV_MAPPING,
    "Drosophila": DROSOPHILA_DEV_MAPPING,
}


def normalize_timepoint(
    timepoint: str,
    species: str,
    custom_mapping: Optional[Dict[str, float]] = None
) -> Optional[float]:
    """
    Convert a species-specific timepoint to normalized developmental time (0-1).
    """
    if pd.isna(timepoint):
        return None
    
    tp_str = str(timepoint).strip()
    
    # Try custom mapping first
    if custom_mapping and tp_str in custom_mapping:
        return custom_mapping[tp_str]
    
    # Try species-specific mapping
    if species in SPECIES_TIME_MAPPINGS:
        mapping = SPECIES_TIME_MAPPINGS[species]
        if tp_str in mapping:
            return mapping[tp_str]
        # Try case-insensitive
        for k, v in mapping.items():
            if k.lower() == tp_str.lower():
                return v
    
    # Try to parse numeric value (assume already normalized or raw)
    try:
        val = float(tp_str)
        if 0 <= val <= 1:
            return val
        # Could be hours/days - leave as-is for now
        return None
    except ValueError:
        return None


# =============================================================================
# Data Processing Functions
# =============================================================================

def get_expression_matrix(adata: ad.AnnData, layer: Optional[str] = None):
    """Get expression matrix, handling sparse matrices efficiently."""
    X = adata.layers[layer] if (layer and layer in adata.layers) else adata.X
    if sparse.issparse(X):
        return X.tocsr()
    return np.asarray(X)


def get_gene_names(adata: ad.AnnData, gene_col: str = "var_names") -> pd.Series:
    """Extract gene names from AnnData."""
    if gene_col == "var_names":
        return pd.Series(adata.var_names, index=adata.var_names, name="gene_native")
    elif gene_col in adata.var.columns:
        s = adata.var[gene_col].astype(str)
        s.index = adata.var_names
        s.name = "gene_native"
        return s
    else:
        raise ValueError(f"Gene column '{gene_col}' not found in .var")


def compute_group_stats(
    X: Union[np.ndarray, sparse.spmatrix],
    idx_rows: np.ndarray
) -> tuple:
    """
    Compute mean expression and percent expressing for a subset of cells.
    Returns: (mean_expr, pct_expressing, n_cells)
    """
    if sparse.issparse(X):
        sub = X[idx_rows]
        mean_expr = np.asarray(sub.mean(axis=0)).ravel()
        pct_expr = np.asarray((sub > 0).mean(axis=0)).ravel()
        n_cells = sub.shape[0]
    else:
        sub = X[idx_rows, :]
        n_cells = sub.shape[0]
        mean_expr = sub.mean(axis=0)
        pct_expr = (sub > 0).mean(axis=0)
    
    return mean_expr.astype(np.float32), pct_expr.astype(np.float32), int(n_cells)


def process_single_dataset(config: DatasetConfig) -> tuple:
    """
    Process a single h5ad file into expression summaries.
    
    Returns:
        expr_df: Expression summary DataFrame
        cellmeta_df: Cell type metadata DataFrame
        gene_df: Gene name mapping DataFrame
    """
    print(f"  Processing: {config.name} ({config.species})")
    print(f"    Loading: {config.path}")
    
    adata = ad.read_h5ad(config.path, backed=None)
    adata.var_names_make_unique()
    
    # Validate required columns
    if config.cell_type_col not in adata.obs.columns:
        raise ValueError(f"Cell type column '{config.cell_type_col}' not found in {config.path}")
    
    X = get_expression_matrix(adata, config.layer)
    gene_names = get_gene_names(adata, config.gene_col)
    gene_arr = gene_names.reindex(adata.var_names).astype(str).values
    
    # Build grouping columns
    group_cols = [config.cell_type_col]
    if config.timepoint_col and config.timepoint_col in adata.obs.columns:
        group_cols.append(config.timepoint_col)
    if config.batch_col and config.batch_col in adata.obs.columns:
        group_cols.append(config.batch_col)
    
    # Create observation subset for grouping
    obs_small = adata.obs[group_cols].astype(str).copy()
    obs_small["_row"] = np.arange(adata.n_obs, dtype=int)
    
    rows = []
    meta_rows = []
    
    for group_key, sub in obs_small.groupby(group_cols, sort=False):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        
        idx = sub["_row"].to_numpy()
        
        if len(idx) < 10:  # Skip small groups
            continue
        
        mean_expr, pct_expr, n_cells = compute_group_stats(X, idx)
        
        # Build row data
        row_data = {
            "species": config.species,
            "dataset": config.name,
            "cell_type": group_key[0],
            "gene_native": gene_arr,
            "mean_expr": mean_expr,
            "pct_expressing": pct_expr,
            "n_cells": n_cells,
        }
        
        # Add timepoint if present
        if len(group_key) > 1 and config.timepoint_col:
            row_data["timepoint"] = group_key[1]
            # Normalize timepoint
            row_data["dev_time"] = normalize_timepoint(
                group_key[1], config.species, config.timepoint_mapping
            )
        
        # Add batch if present
        if len(group_key) > 2 and config.batch_col:
            row_data["batch"] = group_key[2]
        
        rows.append(pd.DataFrame(row_data))
        
        meta_row = {
            "species": config.species,
            "dataset": config.name,
            "cell_type": group_key[0],
        }
        if config.timepoint_col and len(group_key) > 1:
            meta_row["timepoint"] = group_key[1]
        meta_rows.append(meta_row)
    
    expr_df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    cellmeta_df = pd.DataFrame(meta_rows).drop_duplicates() if meta_rows else pd.DataFrame()
    
    # Gene mapping (identity for now - orthologs added later)
    gene_df = pd.DataFrame({
        "species": config.species,
        "gene_native": gene_names.unique()
    })
    gene_df["gene_human"] = gene_df["gene_native"]  # Default to same
    
    print(f"    Processed {len(meta_rows)} cell type groups, {len(gene_arr)} genes")
    
    return expr_df, cellmeta_df, gene_df


def add_ortholog_mapping(
    expr_df: pd.DataFrame,
    gene_df: pd.DataFrame,
    ortholog_path: str
) -> tuple:
    """Add human ortholog gene symbols from a mapping file."""
    print(f"  Loading ortholog map: {ortholog_path}")
    
    ortho_map = pd.read_csv(ortholog_path)
    
    # Expected columns: species, gene_native, gene_human
    required = {"species", "gene_native", "gene_human"}
    if not required.issubset(ortho_map.columns):
        print(f"    Warning: Ortholog map missing columns {required - set(ortho_map.columns)}")
        return expr_df, gene_df
    
    # Update expression DataFrame
    expr_df = expr_df.merge(
        ortho_map[["species", "gene_native", "gene_human"]].drop_duplicates(),
        on=["species", "gene_native"],
        how="left",
        suffixes=("_old", "")
    )
    
    # Fill missing with original
    if "gene_human_old" in expr_df.columns:
        expr_df["gene_human"] = expr_df["gene_human"].fillna(expr_df["gene_human_old"])
        expr_df = expr_df.drop(columns=["gene_human_old"])
    
    # For human data, gene_human = gene_native
    human_mask = expr_df["species"].str.lower() == "human"
    expr_df.loc[human_mask, "gene_human"] = expr_df.loc[human_mask, "gene_native"]
    
    # Update gene map
    gene_df = expr_df[["species", "gene_native", "gene_human"]].drop_duplicates()
    
    return expr_df, gene_df


def load_risk_genes(path: str) -> pd.DataFrame:
    """Load and standardize SFARI risk genes file."""
    df = pd.read_csv(path)
    
    # Standardize column names
    col_mapping = {
        "gene-symbol": "gene_symbol",
        "gene-score": "gene_score",
        "gene-name": "gene_name",
        "ensembl-id": "ensembl_id",
        "genetic-category": "genetic_category",
        "number-of-reports": "n_reports",
    }
    
    df = df.rename(columns={k: v for k, v in col_mapping.items() if k in df.columns})
    
    return df


# =============================================================================
# Main Pipeline
# =============================================================================

def run_pipeline(config: PipelineConfig):
    """Run the full data preparation pipeline."""
    print("=" * 60)
    print("SFARI Explorer - Data Preparation Pipeline")
    print("=" * 60)
    
    os.makedirs(config.output_dir, exist_ok=True)
    
    all_expr = []
    all_meta = []
    all_genes = []
    
    # Process each dataset
    print("\n[1/4] Processing datasets...")
    for ds_config in config.datasets:
        expr_df, meta_df, gene_df = process_single_dataset(ds_config)
        all_expr.append(expr_df)
        all_meta.append(meta_df)
        all_genes.append(gene_df)
    
    # Combine
    expr_all = pd.concat(all_expr, ignore_index=True) if all_expr else pd.DataFrame()
    meta_all = pd.concat(all_meta, ignore_index=True).drop_duplicates() if all_meta else pd.DataFrame()
    genes_all = pd.concat(all_genes, ignore_index=True).drop_duplicates() if all_genes else pd.DataFrame()
    
    # Add ortholog mapping
    print("\n[2/4] Adding ortholog mapping...")
    if config.ortholog_map_path and os.path.exists(config.ortholog_map_path):
        expr_all, genes_all = add_ortholog_mapping(expr_all, genes_all, config.ortholog_map_path)
    else:
        print("  No ortholog map provided - using native gene names")
    
    # Load risk genes
    print("\n[3/4] Loading SFARI risk genes...")
    if config.risk_genes_path and os.path.exists(config.risk_genes_path):
        risk_df = load_risk_genes(config.risk_genes_path)
        print(f"  Loaded {len(risk_df)} risk genes")
    else:
        risk_df = pd.DataFrame(columns=["gene_symbol", "gene_score", "gene_name"])
        print("  No risk genes file provided")
    
    # Write outputs
    print("\n[4/4] Writing output files...")
    
    # Expression summaries
    expr_path = os.path.join(config.output_dir, "expression_summaries.parquet")
    pq.write_table(pa.Table.from_pandas(expr_all, preserve_index=False), expr_path)
    print(f"  ✓ {expr_path} ({len(expr_all):,} rows)")
    
    # For backward compatibility, also write with 'tissue' column name
    expr_compat = expr_all.copy()
    if "dataset" in expr_compat.columns and "tissue" not in expr_compat.columns:
        expr_compat["tissue"] = expr_compat["dataset"]
    
    # Cell type metadata
    meta_path = os.path.join(config.output_dir, "celltype_meta.parquet")
    pq.write_table(pa.Table.from_pandas(meta_all, preserve_index=False), meta_path)
    print(f"  ✓ {meta_path} ({len(meta_all)} entries)")
    
    # Gene map
    gene_path = os.path.join(config.output_dir, "gene_map.parquet")
    pq.write_table(pa.Table.from_pandas(genes_all, preserve_index=False), gene_path)
    print(f"  ✓ {gene_path} ({len(genes_all):,} genes)")
    
    # Risk genes
    risk_path = os.path.join(config.output_dir, "risk_genes.parquet")
    pq.write_table(pa.Table.from_pandas(risk_df, preserve_index=False), risk_path)
    print(f"  ✓ {risk_path} ({len(risk_df)} genes)")
    
    print("\n" + "=" * 60)
    print("Pipeline complete!")
    print("=" * 60)
    
    # Summary
    print(f"\nSummary:")
    print(f"  Species: {expr_all['species'].nunique()}")
    print(f"  Datasets: {expr_all['dataset'].nunique()}")
    print(f"  Cell types: {expr_all['cell_type'].nunique()}")
    print(f"  Genes: {expr_all['gene_native'].nunique()}")
    if "timepoint" in expr_all.columns:
        print(f"  Timepoints: {expr_all['timepoint'].nunique()}")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Prepare data for SFARI Gene Expression Explorer"
    )
    parser.add_argument(
        "--config", type=str,
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--input", type=str, nargs="+",
        help="Input h5ad file(s) - alternative to config file"
    )
    parser.add_argument(
        "--output", type=str, default="./data",
        help="Output directory for parquet files"
    )
    parser.add_argument(
        "--species", type=str, default="Unknown",
        help="Species name (used with --input)"
    )
    parser.add_argument(
        "--cell-type-col", type=str, default="cell_type",
        help="Column name for cell type annotations"
    )
    parser.add_argument(
        "--timepoint-col", type=str,
        help="Column name for timepoint/age annotations"
    )
    parser.add_argument(
        "--ortholog-map", type=str,
        help="Path to ortholog mapping CSV"
    )
    parser.add_argument(
        "--risk-genes", type=str,
        help="Path to SFARI risk genes CSV"
    )
    
    args = parser.parse_args()
    
    if args.config:
        # Load from YAML config
        import yaml
        with open(args.config) as f:
            cfg_dict = yaml.safe_load(f)
        
        datasets = [DatasetConfig(**ds) for ds in cfg_dict.get("datasets", [])]
        config = PipelineConfig(
            output_dir=cfg_dict.get("output_dir", args.output),
            datasets=datasets,
            ortholog_map_path=cfg_dict.get("ortholog_map_path"),
            risk_genes_path=cfg_dict.get("risk_genes_path"),
        )
    elif args.input:
        # Build config from CLI args
        datasets = []
        for i, path in enumerate(args.input):
            name = Path(path).stem
            datasets.append(DatasetConfig(
                path=path,
                name=name,
                species=args.species,
                cell_type_col=args.cell_type_col,
                timepoint_col=args.timepoint_col,
            ))
        
        config = PipelineConfig(
            output_dir=args.output,
            datasets=datasets,
            ortholog_map_path=args.ortholog_map,
            risk_genes_path=args.risk_genes,
        )
    else:
        parser.error("Either --config or --input must be provided")
    
    run_pipeline(config)


if __name__ == "__main__":
    main()
