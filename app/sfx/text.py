"""Long-form text: methods, definitions and caveats."""

from __future__ import annotations

import numpy as np

from . import config as C


def undetected_fractions(atlas) -> dict:
    out = {}
    for sp in C.SPECIES_ORDER:
        rows = np.flatnonzero(atlas.groups["species"].to_numpy() == sp)
        if len(rows):
            out[sp] = float(np.mean(atlas.detection[rows] == 0))
    return out


def methods_markdown(atlas) -> str:
    s = atlas.summary
    nd = undetected_fractions(atlas)
    nd_txt = ", ".join(f"{sp} {v:.0%}" for sp, v in nd.items())
    bi = atlas.build_info
    integration = bi.get("integration", {})
    if integration:
        integ_txt = (f"Cells were embedded with **{integration.get('method', 'n/a')}** "
                     f"(version {integration.get('version', 'n/a')}; domain = `{integration.get('domain_key', 'n/a')}`, "
                     f"{integration.get('n_features', 'n/a')} input features, latent dimension "
                     f"{integration.get('latent_dim', 'n/a')}). Harmonised cell-type labels were transferred to "
                     f"unlabelled cells with {integration.get('label_transfer', 'n/a')}.")
        if integration.get("holdout_accuracy") is not None:
            integ_txt += (f" On author-labelled cells hidden during training, transferred labels agreed with the "
                          f"author labels for {integration['holdout_accuracy']:.1%} of cells.")
    else:
        integ_txt = ("Harmonised cell-type labels were transferred to cells without a usable author annotation by "
                     "semi-supervised integration (scANVI in this data build). The integration method used for the "
                     "2D embedding is not recorded in this build; builds produced by the CONCORD pipeline write it "
                     "to `data/build_info.json`, and it then appears here.")
    absent = {ds: int(atlas.absent[i].sum()) for ds, i in atlas.dataset_index.items()}
    sp_of = dict(zip(atlas.groups["dataset"], atlas.groups["species"]))
    absent_human = ", ".join(f"{ds} {n:,}" for ds, n in sorted(absent.items(), key=lambda t: -t[1])
                             if sp_of.get(ds) == "Human")
    n_pb = s.get("pseudobulk_samples")
    n_pb_txt = f"{int(n_pb):,}" if n_pb else "n/a"

    return f"""
#### Overview
{C.APP_NAME} summarises single-cell and single-nucleus RNA-seq data from {len(atlas.datasets)} published
datasets ({", ".join(C.SPECIES_ORDER)}) at the level of harmonised cell types and developmental time points.
It covers all {len(atlas.genes):,} genes that pass expression filtering. SFARI Gene annotations are shown as one
gene-level annotation layer; the atlas itself is not restricted to SFARI genes.

#### Processing
1. **Gene space.** Non-human genes were mapped to human orthologs upstream of this build ("humanised"), and all
   datasets were concatenated on human gene symbols. Values for a non-human species therefore refer to the
   ortholog(s) of the human gene shown. Genes without an ortholog are not detected in that species.
2. **Cell types.** Author annotations were harmonised into {len(C.CELLTYPE_ORDER)} supercategories.
   {integ_txt}
3. **Pseudobulk.** Raw counts were summed per dataset × donor/sample × cell type × time point
   ({n_pb_txt} pseudobulk samples).
4. **Normalisation.** TMM normalisation (edgeR) and voom (limma), giving log2 counts per million (log2 CPM).
   Genes were retained with ≥ 10 counts in at least max(3, 5 %) of pseudobulk samples ({len(atlas.genes):,} genes).
5. **Batch correction.** ComBat on voom values, with dataset as batch, run **separately within each species**
   (human in vivo and human organoid separately) and preserving cell type (and age where estimable). Species and
   dataset are confounded, so nothing is corrected across species.
6. **Summaries.** For each group, the value shown is the mean log2 CPM across its pseudobulk replicates. The
   **detection rate** is the fraction of those replicates with ≥ 1 raw count.

#### How to read the values
- **log2 CPM is comparable within a species, not across species.** Chemistry, depth, ortholog mapping and library
  composition differ between species. Cross-species views therefore use within-dataset scaling (Δ log2 CPM,
  z-scores) or rank correlations of cell-type profiles.
- **Undetected groups are hidden by default.** voom assigns a finite value to zero counts (log of a 0.5
  pseudo-count over library size), which in small pseudobulks can look like moderate expression. In this build the
  share of gene × group values with zero detection is {nd_txt}. These are shown as grey × marks or as gaps,
  never as values. You can change the threshold in the sidebar.
- **Genes absent from a dataset.** Some genes have zero counts in *every* group of a dataset. For non-human
  species this is mostly a lack of orthologs. For human datasets it usually means the gene was not in that
  dataset's count matrix (feature set or symbol mapping); for example several high-confidence SFARI genes have no
  counts anywhere in Velmeshev (2023). Such datasets carry no information on the gene. They are labelled "gene
  absent from this dataset" (column `measured` in the Data view) and never shown as low expression. Genes absent
  per human dataset in this build: {absent_human}.
- **Group size matters.** Means over few cells are noisy; use *Minimum cells per group* in the sidebar.
  Marker size in trajectory plots scales with log10(cells).
- **Δ log2 CPM** subtracts the mean of the same dataset × cell type (i.e. a log2 fold change relative to that
  mean). It removes species/dataset offsets while keeping effect sizes in log2 units. **z-scores** additionally
  divide by the standard deviation, which can exaggerate flat profiles.
- **Conservation** is the Spearman correlation between two species' cell-type profiles of a gene (mean over in vivo
  datasets per cell type, ≥ 4 shared cell types). It asks whether the gene is relatively high and low in the same
  cell types. It does not compare absolute levels. Gene-set correlations are compared with a background of
  randomly drawn detected genes (two-sided Mann–Whitney U).

#### Developmental time
- Native ages were harmonised to days post-conception (human, mouse), hours post-fertilisation (zebrafish),
  days post-eclosion (*Drosophila*) and days in culture (organoids).
- The **relative developmental time** is a piecewise-linear life-history scale anchored at fertilisation (0),
  birth or hatching (0.45) and sexual maturity (1.0); values > 1 are adult ageing. Organoid culture days were placed
  on the fetal segment of this scale. This is a convenience alignment, not an equivalence of developmental events.
  For comparing neurodevelopmental milestones, event-based models such as *Translating Time*
  (Workman et al., J. Neurosci. 2013) are more appropriate.
- Known limitations of the age harmonisation in this build: last-menstrual-period ages and post-conception ages
  were not offset-corrected (≈ 2 weeks), birth was placed at 280 days for humans, the Jin (2025) "adult"/"aged"
  categories map to nominal ages, and Wang (2022) has no age metadata.
- Time-resolved detection is not stored, so masking in time-course views uses the detection rate of the parent
  dataset × cell-type group.

#### Cell-type labels across phyla
Labels were transferred from vertebrate references. For *Drosophila*, some labels lack a homologous cell class
(oligodendrocyte lineage, endothelial/vascular, microglia, fibroblast, erythrocytes). They indicate transcriptional
similarity only. These groups are drawn semi-transparent and annotated in hover text.

#### Citation
Please cite the original datasets (see *Overview*) and SFARI Gene (gene.sfari.org; release 07-08-2025) when using
values from this resource.
"""
