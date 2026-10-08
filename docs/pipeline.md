# SFARI Single-Cell RNA-seq Pipeline

A modular, resumable pipeline for concatenating, cleaning, and annotating multiple single-cell RNA-seq datasets.

## Overview

This pipeline:
1. **Inspects datasets** for raw counts (checks `.X`, `.raw.X`, and layers)
2. **Builds a clean gene universe** (removes MT genes, ribosomal, etc.)
3. **Concatenates all datasets** with proper gene alignment
4. **Annotates cell types** (merges columns + assigns supercategories)
5. **Annotates developmental stages** (merges columns + assigns supercategories)

Followed by integration, label transfer and pseudobulk summarisation for the explorer app:

6. **Integrates and transfers labels with CONCORD** (`integrate_concord.py`, replaces scVI/scANVI)
7. **Pseudobulk, voom, within-species ComBat and parquet export** (`prep_sfari_data_v4.py --stage 1`,
   `normalize_age.py`, `correct_within_org.R`, `prep_sfari_data_v5.py --stage 2`; or the v8 route)

## Step 6: CONCORD integration and label transfer

`integrate_concord.py` replaces the scVI → scANVI cells of `data/01052026_SFARIExplorer_data-prep.ipynb`.

| | Before (scVI/scANVI) | Now (CONCORD 1.0.13) |
|---|---|---|
| Batch | `batch_key='dataset'` | `domain_key='dataset'`, dataset-aware minibatches (`p_intra_domain=1`) |
| Features | HVGs | seurat_v3 HVGs (batch = dataset) among orthologs detected in every species |
| Labels | scANVI, unlabelled = `Unknown` | CONCORD classifier head, unlabelled = `Unknown`, 10 % held-out evaluation |
| Latent | `X_scANVI` | `X_concord` (100 dims) |
| Output | `combined_scanvi_label_transfer.h5ad` | `combined_concord_label_transfer.h5ad` |

```bash
conda env create -f environment_concord.yml
bsub < run_integrate_concord.lsf          # GPU node; CPU works but is slow
```

Outputs (`data/concord/`): `concord_label_transfer_qc.csv` (held-out precision/recall per cell type and
accuracy per dataset), `concord_class_probabilities.parquet`, `concord_cell_annotations.parquet`,
`gene_detection_by_species.parquet`, `concord_features.csv`, `umap_subsample.parquet` and
`build_info.json`. Copy the last two into the app's `data/` folder; the app then reports the integration
provenance in *Methods* and *Cell atlas*.

`obs['predicted_labels']` keeps the harmonised author label where one exists and uses the CONCORD prediction
otherwise (`obs['label_source']`). Pass `--overwrite-author-labels` to reproduce the scANVI behaviour of using
predictions for every cell. All downstream scripts now read the CONCORD output; the previous versions are
in `archive_scvi_20261008/`.

## Features

- **Resumable**: Each step has checkpoints; pipeline can resume after crashes
- **Per-dataset checkpoints**: Individual datasets are tracked; re-run only failed ones
- **Memory efficient**: Gene filtering happens early to reduce memory burden
- **Raw count detection**: Automatically finds raw counts in `.X`, `.raw.X`, or layers
- **Modular**: Run individual steps or the full pipeline

## Directory Structure

```
sfari_pipeline/
├── config.py                    # Central configuration
├── 01_inspect_datasets.py       # Step 1: Find raw counts
├── 02_build_gene_universe.py    # Step 2: Build filtered gene list
├── 03_concatenate_datasets.py   # Step 3: Concatenate all datasets
├── 04_annotate_celltypes.py     # Step 4: Cell type annotation
├── 05_annotate_devstage.py      # Step 5: Developmental stage annotation
├── run_pipeline.sh              # Master submission script
├── pipeline_status.py           # Status/management utility
├── jobs/                        # Individual bsub scripts
│   ├── step01_inspect.bsub
│   ├── step02_genes.bsub
│   ├── step03_concatenate.bsub
│   ├── step04_celltypes.bsub
│   └── step05_devstage.bsub
└── README.md                    # This file
```

## Installation

1. Copy the pipeline to your working directory:
   ```bash
   cp -r /path/to/sfari_pipeline /sc/arion/projects/ad-omics/raphael/SFARI/pipeline
   ```

2. Edit `config.py` to set your paths:
   - `DATA_DIR`: Where your h5ad files are
   - `GENE_MAP_DIR`: Where your Ensembl→symbol mapping CSVs are
   - `OUTPUT_DIR`: Where outputs will be saved
   - Update `ENSEMBL_DATASETS` and `SYMBOL_DATASETS` with your files

3. Make the shell scripts executable:
   ```bash
   chmod +x run_pipeline.sh
   ```

## Usage

### Run the full pipeline

```bash
./run_pipeline.sh
```

This submits all 5 steps as LSF jobs with dependencies.

### Run with options

```bash
# Clean all checkpoints and start fresh
./run_pipeline.sh --clean

# Run only step 3
./run_pipeline.sh --step 3

# Run from step 2 onwards
./run_pipeline.sh --from 2

# Dry run (show what would be submitted)
./run_pipeline.sh --dry-run
```

### Run individual steps manually

```bash
# Submit a single step
bsub < jobs/step01_inspect.bsub

# Submit with dependency
bsub -w "done(sfari_step01)" < jobs/step02_genes.bsub
```

### Check pipeline status

```bash
python pipeline_status.py              # Show overall status
python pipeline_status.py datasets     # Show per-dataset status
python pipeline_status.py logs         # Show recent log files
python pipeline_status.py clear        # Clear all checkpoints
python pipeline_status.py clear 3      # Clear checkpoint for step 3 only
```

## Configuration

### Adding new datasets

Edit `config.py`:

```python
# For datasets with gene symbols as var_names
SYMBOL_DATASETS = {
    "My-New-Dataset": "/path/to/my_dataset.h5ad",
    ...
}

# For datasets with Ensembl IDs as var_names
ENSEMBL_DATASETS = {
    "Another-Dataset": "/path/to/another.h5ad",
    ...
}

# Add metadata
DATASET_META = {
    "My-New-Dataset": {"dataset": "My Dataset (2024)", "organism": "Human"},
    ...
}
```

### Gene filtering

Edit `config.py` to customize gene filtering:

```python
# Prefixes to exclude
GENE_EXCLUDE_PREFIXES = ["ERCC", "MT-", "mt-", ...]

# Regex patterns to exclude
GENE_EXCLUDE_PATTERNS = [r"^RP[SL]\d", ...]

# Minimum datasets a gene must appear in
MIN_DATASETS_PER_GENE = 2
```

### Cell type columns

Edit `config.py` to set priority order for cell type columns:

```python
CELLTYPE_COLUMNS = [
    "cell_type",
    "Subclass",
    "CellClass",
    ...
]
```

### External metadata

For datasets that need external cell type annotations:

```python
EXTERNAL_METADATA = {
    "Velmeshev-2019": {
        "path": "/path/to/meta.tsv",
        "sep": "\t",
        "barcode_col": "cell",
        "celltype_col": "cluster",
        "prefix": "Velmeshev-2019",
    },
    ...
}
```

## Outputs

After successful completion:

| File | Description |
|------|-------------|
| `dataset_inspection.json` | Per-dataset raw count inspection results |
| `filtered_genes.txt` | List of genes passing filters |
| `gene_manifest.parquet` | Gene × dataset presence matrix |
| `concatenated_raw.h5ad` | All datasets concatenated (raw counts) |
| `concatenated_annotated.h5ad` | With cell type annotations |
| `sfari_final.h5ad` | Final output with all annotations |

## Troubleshooting

### Job fails with memory error

Increase memory in the bsub script:
```bash
#BSUB -R "rusage[mem=512G]"
```

### Dataset processing fails

1. Check the logs: `python pipeline_status.py logs`
2. Clear that dataset's checkpoint and re-run step 3:
   ```bash
   rm /path/to/checkpoints/dataset_MyDataset.done
   bsub < jobs/step03_concatenate.bsub
   ```

### No raw counts found for a dataset

Check `dataset_inspection.json` to see what was detected. You may need to:
- Use a different source file with raw counts
- Check if `.raw` attribute exists
- Check available layers

### Gene mapping fails

Ensure you have a gene map CSV in `GENE_MAP_DIR`:
```
MyDataset_gene_map.csv
  ensembl_gene_id,symbol
  ENSG00000123456,BRCA1
  ...
```

## Resource Requirements

| Step | Memory | Time | Notes |
|------|--------|------|-------|
| 1. Inspect | 32 GB | 2h | Quick scans |
| 2. Genes | 64 GB | 4h | Collects gene lists |
| 3. Concatenate | 256 GB | 24h | Most intensive |
| 4. Cell types | 128 GB | 8h | Loads full adata |
| 5. Dev stages | 128 GB | 4h | Loads full adata |

Adjust these in `run_pipeline.sh` or individual bsub scripts as needed.
