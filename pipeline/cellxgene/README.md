# CELLxGENE fetcher

Adds developmental brain datasets from [CZ CELLxGENE Discover](https://cellxgene.cziscience.com/)
to the pipeline for the atlas species (human, mouse, zebrafish; Discover has no *Drosophila* data yet).

```bash
python pipeline/cellxgene/fetch_cellxgene.py search      # writes $SFARI_ROOT/data/cellxgene/manifest.tsv
# review manifest.tsv: include = True/False; check the `reason` and `review_flags` columns
python pipeline/cellxgene/fetch_cellxgene.py run         # download, orthologs, prepare, register
```

After `register`, `pipeline/config.py` (steps 01–05) and the downstream scripts pick the datasets up from
`registry.json` (see `pipeline/dataset_registry.py`); no script needs editing.

| File | Content |
|---|---|
| `manifest.tsv` | every CNS candidate with metadata, parsed age range, `include`, exclusion `reason`, `review_flags`, `disease_excluded` |
| `raw/<dataset_id>.h5ad` | downloaded CELLxGENE files |
| `orthologs/<species>_to_human.tsv` | Ensembl Compara orthologs (+ `.json` with the Ensembl release host) |
| `prepared/<key>.h5ad` | raw counts × human gene symbols, primary CNS cells of eligible donors, harmonised `obs` |
| `registry.json` | prepared datasets for the pipeline |
| `dataset_references.json` | references for the app (copy into the app's `data/`) |
| `ontology_cache.json`, `disease_cache.json` | tissue term → CNS (UBERON); donor condition → compatible with healthy brain (MONDO) |

**Selection.** CNS tissue by ontology ancestry (so "middle temporal gyrus" counts and "cortex of kidney"
does not); single-cell/nucleus RNA assays, no spatial data, no cell cultures; donors without brain-relevant
conditions present (see below); at least one developmental stage (human < 18 y, mouse < 6 weeks, zebrafish < 90 days); primary data only
(re-deposited cells would be counted twice); ≥ 1,000 cells; publications already in the atlas are
excluded by DOI. Flags mark tumour, treatment/disease-model, meninges-only and multi-tissue datasets for
review. `--include-adult`, `--include-retina`, `--include-whole-embryo` and `--disease-policy all` widen the
search.

**Donor conditions** (`--disease-policy`, search and prepare). CELLxGENE records each donor's conditions,
often unrelated to the brain (control donors who died of heart failure). Default `brain-healthy` keeps cells
unless any of the donor's conditions falls under a MONDO nervous-system, psychiatric, neoplastic or
chromosomal disorder (e.g. trisomy 21 and 18, bipolar disorder, pilocytic astrocytoma are removed; heart
failure and myocardial infarction are kept). `normal-only` keeps only donors labelled normal; `all` keeps
everyone. Treatments that are not recorded as a disease (e.g. drug-exposed organoids) are not filtered;
the `review_flags` column marks them.

**Ages** are parsed from the HsapDv / MmusDv / ZFS stage terms into the pipeline's units (days
post-conception for human and mouse, hours post-fertilisation for zebrafish) with the conventions of
`normalize_age.py`, and written as tagged strings (`"98 dpc"`, `"16 hpf"`) that `normalize_age.py`
reads directly. Broad terms ("adult stage") have no age. Organoid datasets carry the donor's stage, not
the culture age, so their age is unknown unless `--organoid-age-col` names an obs column with days in
culture.

**Genes.** Human data use CELLxGENE `feature_name`; mouse and zebrafish Ensembl IDs are mapped to human
orthologs. Default policy `unique_human`: species genes with exactly one human ortholog are kept and
paralogs of the same human gene are summed (zebrafish *shank3a* + *shank3b* → SHANK3). `--orthology
one2one` is stricter but drops much of the zebrafish genome (9,779 vs 12,734 human genes in Ensembl 2026).
