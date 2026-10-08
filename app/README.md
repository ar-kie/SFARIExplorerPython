---
title: SFARIExplorer
emoji: 🧬
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.50.0
app_file: app.py
pinned: false
license: mit
short_description: Cell-type gene expression across species and development
---

# SFARIExplorer

Cell-type-resolved gene expression across species and development: 13 single-cell / single-nucleus
RNA-seq datasets from human (in vivo and organoid), mouse, zebrafish and *Drosophila*, summarised as
pseudobulk per harmonised cell type and developmental time point. All 17,255 genes are included
(human symbols; non-human genes mapped to human orthologs). SFARI Gene scores are shown as one
annotation layer.

## Pages

| Page | Question it answers |
|---|---|
| Overview | Which datasets, ages and cell types are covered? |
| Gene | Where is a gene expressed, how does it change over development, and is its cell-type profile conserved? |
| Gene set | Heatmap / dot plot of a gene list across datasets and cell types |
| Development | A gene list over time in one species and cell type; module trajectories across species |
| Conservation | Spearman correlation of cell-type profiles between species, against a random-gene background |
| Cell atlas | 2D embedding of a 200k-cell subsample |
| Data | Tidy tables of the current selection (CSV download) |
| Methods | Processing, definitions and caveats |

## Conventions

- Values are pseudobulk log2 CPM (voom), ComBat-corrected **within** species; they are not compared
  across species directly (cross-species views use Δ log2 CPM, z-scores or rank correlations).
- Groups where a gene is not detected (no counts in any pseudobulk replicate) are hidden by default
  instead of being shown with voom's floor value; genes absent from a dataset's count matrix are flagged.
- Colours: species palette validated for colour-vision deficiency; diverging blue–grey–red for scaled
  values, single-hue blue for log2 CPM. Figures download as SVG from the chart toolbar.

## Layout

```
app.py              page setup, sidebar (genes, filters), navigation
sfx/config.py       names, palettes, cell-type ontology, dataset references
sfx/data.py         parquet loading and tidy extraction (detection masking, scaling, conservation)
sfx/plots.py        Plotly figures
sfx/pages.py        page renderers
sfx/text.py         methods text
data/               wide-format parquet build (+ optional build_info.json from integrate_concord.py)
```

## Run locally

```bash
pip install streamlit==1.50.0 -r requirements.txt
streamlit run app.py
```
