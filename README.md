# SFARIExplorer

Cell-type-resolved gene expression across species and development. Single-cell and single-nucleus
RNA-seq data from human (in vivo and organoid), mouse, zebrafish and *Drosophila* are integrated,
summarised as pseudobulk per harmonised cell type and developmental time point, and served in a
Streamlit app.

- App: https://huggingface.co/spaces/ar-kie/SFARIExplorer
- This repository: app code, processing pipeline, Minerva (LSF) jobs. Data are not versioned here.

## Layout

```
app/                    Streamlit app (deployed to the Hugging Face Space); data in app/data/ (not in git)
pipeline/
  cellxgene/            00  fetch developmental brain datasets from CELLxGENE Discover
  config.py             01-05 settings (datasets, paths); merges CELLxGENE datasets from the registry
  01..05_*.py           01-05 inspect, gene universe, concatenate, cell types, developmental stages
  06_integrate_concord.py   CONCORD integration + label transfer (replaces scVI/scANVI)
  prep_sfari_data_v8.py, add_missing_metadata.py, create_merged_columns.py, normalize_age.py,
  correct_within_org.R, prep_sfari_data_v5.py     pseudobulk, ages, batch correction, parquets
  dataset_registry.py   per-dataset settings shared by the scripts above
jobs/                   LSF job scripts (submit from the repository root)
envs/concord.yml        conda environment for every Python step
resources/              SFARI Gene export (07-08-2025 release), Wang (2022) metadata
notebooks/              original data-preparation notebook (scVI era; superseded by step 06)
docs/pipeline.md        step-by-step details, inputs and outputs
scripts/                fetch app data, sync the app to the Space, sync to Minerva
```

## Run the app locally

```bash
pip install streamlit==1.50.0 -r app/requirements.txt
bash scripts/fetch_app_data.sh            # downloads the parquet build from the Space into app/data/
cd app && streamlit run app.py
```

## On Minerva

```bash
cd /sc/arion/projects/ad-omics/raphael/SFARI
git clone https://github.com/ar-kie/SFARIExplorerPython SFARIExplorer && cd SFARIExplorer
conda env create -f envs/concord.yml       # once: Python (pipeline, CONCORD, fetcher)
conda env create -f envs/r_correction.yml  # once: R (batch correction)
export SFARI_ROOT=/sc/arion/projects/ad-omics/raphael/SFARI   # data root (this is the default)
```

If the environments live under a path rather than a name, export `CONDA_ENV=/path/to/concord` and
`R_ENV=/path/to/r_correction` before submitting; the jobs pass them on.

Update later with `git pull`. To push uncommitted local changes instead, use
`bash scripts/sync_to_minerva.sh <minerva_user>`.

| Step | Run | Output (under `$SFARI_ROOT`) |
|---|---|---|
| 00 add CELLxGENE data (optional) | `python pipeline/cellxgene/fetch_cellxgene.py search`, review `manifest.tsv`, then `bsub < jobs/run_fetch_cellxgene.lsf` | `data/cellxgene/prepared/*.h5ad`, `registry.json` |
| 01–05 build | `bash jobs/run_pipeline_01-05.sh` | `pipeline_output/concatenated_annotated.h5ad` |
| 06 integrate | `bsub < jobs/run_integrate_concord.lsf` | `data/combined_concord_label_transfer.h5ad`, `data/concord/` |
| 07 post-process | `bsub < jobs/run_postprocess.lsf` | pseudobulk, voom + ComBat (within species), long-format parquets |
| **01–07 in one go** | `bash jobs/run_all.sh` (needs an empty `pipeline_output/`) | all of the above, chained by job dependencies |

Adding datasets changes the gene list, and step 03 caches each dataset aligned to the previous list, so a
rebuild starts from an empty `pipeline_output/`: move the old one aside (`run_all.sh` refuses to start
while checkpoints exist).

Details, inputs and outputs per script: [docs/pipeline.md](docs/pipeline.md).

### Adding data from CELLxGENE

`pipeline/cellxgene/fetch_cellxgene.py` searches CELLxGENE Discover for developmental central-nervous-system
datasets in the atlas species, writes a reviewable manifest, then downloads, maps genes to human orthologs,
parses developmental stages into the pipeline's age units and registers the prepared files. Steps 01–07 then
include them without code changes. See [pipeline/cellxgene/README.md](pipeline/cellxgene/README.md).

### Environment variables

| Variable | Default | Used by |
|---|---|---|
| `SFARI_ROOT` | `/sc/arion/projects/ad-omics/raphael/SFARI` | every pipeline script and job |
| `SFARI_EXCHANGE_DIR` | `$SFARI_ROOT/data/r_exchange` | pseudobulk, age, R correction and parquet steps |
| `SFARI_CELLXGENE_REGISTRY` | `$SFARI_ROOT/data/cellxgene/registry.json` | `dataset_registry.py` |
| `CONDA_ENV`, `R_ENV` | `concord`, `r_correction` | job scripts (name or path of the conda environments) |

## Deploying the app

The Space is a separate git repository (with the parquet data in Git LFS). Copy the app code into a clone of
it and push from there:

```bash
git clone https://huggingface.co/spaces/ar-kie/SFARIExplorer ../hf_space   # once
bash scripts/sync_hf_space.sh ../hf_space
cd ../hf_space && git diff && git commit -am "Update app" && git push
```

A new data build goes into the Space's `data/`. Add `build_info.json` (from `data/concord/`) and
`dataset_references.json` (from `data/cellxgene/`) so the app reports the integration method and the
references of added datasets.

## Known gaps

- The script that converts the long-format parquets (step 07) into the app's wide format
  (`expression_mean/pct/meta`, `temporal_mean/meta`, `stage_mapping` with `relative_dev_time`) is not in this
  repository. It was run on Minerva for the current build and should be added under `pipeline/`.
- Steps 06–07 have not yet been run with CONCORD on the full data. Step 06 was tested end to end on
  synthetic data.

## Data and citation

Please cite the original datasets (listed in the app's Overview and Methods) and SFARI Gene
(gene.sfari.org) when using values from this resource.
