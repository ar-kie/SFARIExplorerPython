"""
Cross-species integration and cell-type label transfer with CONCORD
===================================================================

Replaces the scVI/scANVI step of 01052026_SFARIExplorer_data-prep.ipynb (cells 26-34):

    scVI(batch_key='dataset') -> scANVI(unlabeled='Unknown') -> X_scANVI, C_scANVI

with CONCORD (Zhu et al., Gartner lab; https://github.com/Gartner-Lab/Concord), a
contrastive model whose dataset-aware minibatch sampling removes batch effects and
whose optional classifier head performs semi-supervised label transfer:

    CONCORD(domain_key='dataset', class_key=<supercategory>, unlabeled='Unknown')
        -> obsm['X_concord'], obs['predicted_labels'], obs['predicted_labels_confidence']

Input : pipeline_output/concatenated_annotated.h5ad  (raw counts, all genes, from step 04)
Output: data/combined_concord_label_transfer.h5ad    (raw counts of the selected features,
                                                      annotations, latent space, UMAP)
        data/concord/...                             (model, QC tables, provenance)

The output h5ad replaces combined_scanvi_label_transfer.h5ad for all downstream scripts
(transfer_annotations.py, add_missing_metadata.py, generate_pseudobulk_expression.py,
prep_sfari_data_v5.py / v8.py).

Design choices
  * Features: highly variable genes (seurat_v3, batch = dataset) chosen among genes that
    are detected in every species. Gene symbols are human orthologs, so this restricts the
    embedding to conserved orthologs and keeps "gene absent in species X" from driving
    the species separation.
  * Normalisation: library size over ALL genes (not only the selected features), scaled
    to 1e4, log1p. Raw counts of the selected features are kept for downstream pseudobulk.
  * Domain: dataset (as for scVI). Species are confounded with dataset, so species are
    integrated through the dataset-aware sampler; there is no separate species covariate.
  * Label transfer: CONCORD classifier head trained on harmonised author labels, with
    'Unknown' as the unlabelled class (as for scANVI). A stratified fraction of labelled
    cells is hidden during training and used to report held-out accuracy.
  * Author labels are kept where they exist (--keep-author-labels, default); transferred
    labels fill the unlabelled cells. Provenance is stored in obs['label_source'].
  * UMAP is fitted on a stratified subsample (the app only displays a subsample);
    obsm['X_umap'] is NaN for the remaining cells.

Usage (Minerva, GPU node; see run_integrate_concord.lsf):
    python integrate_concord.py
    python integrate_concord.py --n-features 4000 --n-epochs 15 --batch-size 512
"""

import argparse
import datetime as dt
import gc
import json
import os
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse

# =============================================================================
# CONFIG
# =============================================================================

INPUT_H5AD = '/sc/arion/projects/ad-omics/raphael/SFARI/pipeline_output/concatenated_annotated.h5ad'
OUTPUT_H5AD = '/sc/arion/projects/ad-omics/raphael/SFARI/data/combined_concord_label_transfer.h5ad'
OUTPUT_DIR = '/sc/arion/projects/ad-omics/raphael/SFARI/data/concord'

SPECIES_COL = 'organism'
DATASET_COL = 'dataset'
LABEL_COL = 'cell_type_supercategory'      # harmonised author labels from step 04
UNLABELED = 'Unknown'

ORGANOID_DATASETS = ['He (2024)', 'Wang (2022)']

# =============================================================================
# HELPERS
# =============================================================================


def log(msg):
    print(f"[{dt.datetime.now():%H:%M:%S}] {msg}", flush=True)


def fix_organism_labels(obs):
    """Raj (2020) is zebrafish (mislabelled upstream; same fix as in transfer_annotations.py)."""
    obs[SPECIES_COL] = obs[SPECIES_COL].astype(str)
    obs.loc[obs[DATASET_COL].astype(str) == 'Raj (2020)', SPECIES_COL] = 'Zebrafish'
    return obs


def stratified_sample(groups: pd.Series, n: int, min_per_group: int, rng) -> np.ndarray:
    """Proportional sample of ~n positions with at least min_per_group per group (or all)."""
    pos = np.arange(len(groups))
    out = []
    frac = min(1.0, n / len(groups))
    for _, idx in pd.Series(pos).groupby(groups.to_numpy()):
        k = min(len(idx), max(min_per_group, int(round(len(idx) * frac))))
        out.append(rng.choice(idx.to_numpy(), size=k, replace=False))
    return np.sort(np.concatenate(out))


def iter_row_chunks(adata_b, chunk):
    for start in range(0, adata_b.n_obs, chunk):
        end = min(start + chunk, adata_b.n_obs)
        X = adata_b.X[start:end]
        yield start, end, (X.tocsr() if sparse.issparse(X) else sparse.csr_matrix(X))


# =============================================================================
# STEPS
# =============================================================================


def scan_counts(adata_b, species, chunk):
    """One pass over the raw counts: library sizes and per-species detection per gene."""
    sp_codes, sp_names = pd.factorize(species)
    det = np.zeros((len(sp_names), adata_b.n_vars), np.int64)
    libsize = np.zeros(adata_b.n_obs, np.float64)
    for start, end, X in iter_row_chunks(adata_b, chunk):
        libsize[start:end] = np.asarray(X.sum(axis=1)).ravel()
        codes = sp_codes[start:end]
        Xb = (X > 0).astype(np.int32)
        for k in np.unique(codes):
            det[k] += np.asarray(Xb[codes == k].sum(axis=0)).ravel()
        log(f"  scanned {end:,}/{adata_b.n_obs:,} cells")
    n_per_sp = np.bincount(sp_codes, minlength=len(sp_names))
    frac = pd.DataFrame(det / n_per_sp[:, None], index=sp_names, columns=adata_b.var_names)
    return libsize, frac


def select_features(adata_b, obs, candidates, args, rng):
    """seurat_v3 HVGs (batch = dataset) among candidate genes, on a stratified subsample.

    Uses scanpy directly: the released concord-sc (1.0.13) select_features() has no
    batch_key, which would let the largest datasets dominate the selection.
    """
    rows = stratified_sample(obs[DATASET_COL].astype(str), args.hvg_cells, 5_000, rng)
    log(f"  loading {len(rows):,}-cell subsample for feature selection")
    sub = adata_b[rows].to_memory()
    sub = sub[:, candidates].copy()
    sub.obs[DATASET_COL] = obs[DATASET_COL].astype(str).iloc[rows].values
    sizes = sub.obs[DATASET_COL].value_counts()
    sub = sub[sub.obs[DATASET_COL].isin(sizes.index[sizes >= 50])].copy()   # per-batch loess needs cells
    sc.pp.filter_genes(sub, min_cells=10)
    sc.pp.highly_variable_genes(sub, flavor='seurat_v3', n_top_genes=min(args.n_features, sub.n_vars),
                                batch_key=DATASET_COL, subset=False)
    feats = sub.var_names[sub.var['highly_variable']].tolist()
    del sub
    gc.collect()
    return feats


def load_features(adata_b, feat_idx, libsize, chunk):
    """Raw counts and log-normalised values (library = all genes) for the selected features."""
    counts, lognorm = [], []
    scale = 1e4 / np.maximum(libsize, 1.0)
    for start, end, X in iter_row_chunks(adata_b, chunk):
        c = X[:, feat_idx].astype(np.float32)
        counts.append(c)
        n = sparse.diags(scale[start:end].astype(np.float32)) @ c
        n.data = np.log1p(n.data)
        lognorm.append(n.tocsr())
        log(f"  loaded features for {end:,}/{adata_b.n_obs:,} cells")
    return sparse.vstack(counts, format='csr'), sparse.vstack(lognorm, format='csr')


def run_concord(adata, args, device):
    import concord as ccd
    has_unlabeled = UNLABELED in set(adata.obs['_train_label'].astype(str))
    params = dict(
        domain_key=DATASET_COL,
        latent_dim=args.latent_dim,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        normalize_total=False,   # adata.X is already library-size normalised + log1p
        log1p=False,
        p_intra_domain=1.0,      # dataset-aware sampling: each minibatch from one dataset
        p_intra_knn=0.0,         # HCL mode, recommended by CONCORD for > 100k cells
        clr_beta=1.0,
        use_faiss=False,
        preload_dense=False,
        device=device,
    )
    if args.label_transfer:
        params.update(use_classifier=True, class_key='_train_label', classifier_weight=args.classifier_weight,
                      unlabeled_class=UNLABELED if has_unlabeled else None)
    log(f"  CONCORD {ccd.__version__} params: { {k: v for k, v in params.items() if k != 'device'} }")
    model = ccd.Concord(adata, save_dir=str(Path(args.outdir) / 'model'), verbose=True, **params)
    model.fit_transform(output_key='Concord', return_class=args.label_transfer,
                        return_class_prob=args.label_transfer)
    return ccd.__version__, params


def label_transfer_qc(obs, holdout):
    """Held-out accuracy of the classifier against author labels (hidden during training)."""
    from sklearn.metrics import classification_report
    h = obs[holdout]
    if h.empty:
        return pd.DataFrame(), float('nan')
    y_true, y_pred = h[LABEL_COL].astype(str), h['concord_label'].astype(str)
    rep = pd.DataFrame(classification_report(y_true, y_pred, output_dict=True, zero_division=0)).T
    per_ds = (h.assign(correct=(y_true == y_pred).values)
               .groupby(DATASET_COL, observed=True)['correct'].agg(['mean', 'size'])
               .rename(columns={'mean': 'accuracy', 'size': 'n_holdout'}))
    per_ds.index = [f"dataset: {i}" for i in per_ds.index]
    return pd.concat([rep, per_ds]), float((y_true == y_pred).mean())


def fit_umap(adata, args, rng, use_cuml):
    import concord as ccd
    groups = adata.obs[DATASET_COL].astype(str)
    rows = stratified_sample(groups, args.umap_cells, 2_000, rng)
    sub = ad.AnnData(obs=adata.obs.iloc[rows][[]].copy(), obsm={'X_concord': adata.obsm['X_concord'][rows]})
    ccd.ul.run_umap(sub, source_key='X_concord', result_key='X_umap', n_components=2,
                    n_neighbors=30, min_dist=0.1, metric='cosine', random_state=args.seed, use_cuml=use_cuml)
    full = np.full((adata.n_obs, 2), np.nan, np.float32)
    full[rows] = sub.obsm['X_umap']
    return full, rows


# =============================================================================
# MAIN
# =============================================================================


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    p.add_argument('--input', default=INPUT_H5AD)
    p.add_argument('--output', default=OUTPUT_H5AD)
    p.add_argument('--outdir', default=OUTPUT_DIR, help='model, QC tables and provenance')
    p.add_argument('--n-features', type=int, default=4000)
    p.add_argument('--min-species-frac', type=float, default=0.005,
                   help='candidate genes must be detected in at least this fraction of cells of every species')
    p.add_argument('--hvg-cells', type=int, default=500_000, help='subsample size for feature selection')
    p.add_argument('--latent-dim', type=int, default=100)
    p.add_argument('--n-epochs', type=int, default=15)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--lr', type=float, default=1e-2)
    p.add_argument('--classifier-weight', type=float, default=1.0)
    p.add_argument('--no-label-transfer', dest='label_transfer', action='store_false')
    p.add_argument('--holdout-frac', type=float, default=0.1,
                   help='fraction of labelled cells hidden during training to estimate accuracy')
    p.add_argument('--overwrite-author-labels', dest='keep_author_labels', action='store_false',
                   help='use CONCORD predictions for all cells (scANVI-style) instead of keeping author labels')
    p.add_argument('--umap-cells', type=int, default=300_000)
    p.add_argument('--cuml', action='store_true', help='use RAPIDS cuML for UMAP if available')
    p.add_argument('--chunk', type=int, default=200_000, help='rows per chunk when streaming counts')
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    import torch
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    rng = np.random.default_rng(args.seed)
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    log(f"Device: {device}")

    # 1. Metadata and one streaming pass over the counts ------------------------
    log(f"1. Opening {args.input} (backed)")
    adata_b = ad.read_h5ad(args.input, backed='r')
    obs = fix_organism_labels(adata_b.obs.copy())
    log(f"   {adata_b.n_obs:,} cells x {adata_b.n_vars:,} genes")
    log("   Library sizes and per-species detection")
    libsize, det_frac = scan_counts(adata_b, obs[SPECIES_COL].astype(str), args.chunk)
    det_frac.T.to_parquet(Path(args.outdir) / 'gene_detection_by_species.parquet')

    # 2. Features: HVGs among orthologs detected in every species ----------------
    shared = det_frac.columns[(det_frac >= args.min_species_frac).all(axis=0)]
    log(f"2. Candidate genes detected in >= {args.min_species_frac:.1%} of cells of every species: "
        f"{len(shared):,} / {adata_b.n_vars:,}")
    features = select_features(adata_b, obs, list(shared), args, rng)
    log(f"   Selected {len(features):,} features")
    pd.Series(features, name='gene').to_csv(Path(args.outdir) / 'concord_features.csv', index=False)
    feat_idx = adata_b.var_names.get_indexer(features)

    # 3. Load selected features for all cells ------------------------------------
    log("3. Loading counts for the selected features")
    counts, lognorm = load_features(adata_b, feat_idx, libsize, args.chunk)
    var = adata_b.var.iloc[feat_idx].copy()
    adata_b.file.close()
    del adata_b
    gc.collect()
    adata = ad.AnnData(X=lognorm, obs=obs, var=var)
    adata.obs['library_size'] = libsize

    # 4. Training labels with a stratified hold-out ----------------------------------
    labels = adata.obs[LABEL_COL].astype(str).replace({'': UNLABELED, 'nan': UNLABELED})
    adata.obs[LABEL_COL] = labels
    labelled = np.flatnonzero((labels != UNLABELED).to_numpy())
    holdout = np.zeros(adata.n_obs, bool)
    if args.label_transfer and args.holdout_frac > 0 and len(labelled):
        strata = (adata.obs[DATASET_COL].astype(str) + '|' + labels).iloc[labelled]
        for _, idx in pd.Series(labelled).groupby(strata.to_numpy()):
            k = int(np.floor(len(idx) * args.holdout_frac))
            if k:
                holdout[rng.choice(idx.to_numpy(), size=k, replace=False)] = True
    train = labels.copy()
    train[holdout] = UNLABELED
    adata.obs['_train_label'] = pd.Categorical(train)
    log(f"4. Labelled cells: {len(labelled):,} ({holdout.sum():,} held out); "
        f"unlabelled: {adata.n_obs - len(labelled):,}")

    # 5. CONCORD -----------------------------------------------------------------
    log("5. Training CONCORD")
    version, params = run_concord(adata, args, device)
    adata.obsm['X_concord'] = adata.obsm.pop('Concord').astype(np.float32)

    if args.label_transfer:
        prob_cols = [c for c in adata.obs.columns if c.startswith('Concord_class_prob_')]
        probs = adata.obs[prob_cols].astype(np.float32)
        probs.columns = [c.replace('Concord_class_prob_', '') for c in prob_cols]
        probs.index.name = 'cell_id'
        probs.reset_index().to_parquet(Path(args.outdir) / 'concord_class_probabilities.parquet')
        adata.obs['concord_label'] = adata.obs['Concord_class_pred'].astype(str)
        adata.obs['predicted_labels_confidence'] = probs.max(axis=1).to_numpy()
        adata.obs = adata.obs.drop(columns=prob_cols + [c for c in ('Concord_class_pred', 'Concord_class_true')
                                                        if c in adata.obs.columns])
        is_author = (labels != UNLABELED).to_numpy() & args.keep_author_labels
        adata.obs['predicted_labels'] = np.where(is_author, labels, adata.obs['concord_label'])
        adata.obs['label_source'] = np.where(is_author, 'author', 'CONCORD transfer')
        adata.obs['holdout'] = holdout
        qc, acc = label_transfer_qc(adata.obs, holdout)
        qc.to_csv(Path(args.outdir) / 'concord_label_transfer_qc.csv')
        log(f"   Held-out accuracy (author vs CONCORD): {acc:.3f}")
    else:
        adata.obs['predicted_labels'] = labels
        adata.obs['label_source'] = 'author'
        acc = float('nan')
    adata.obs = adata.obs.drop(columns=['_train_label'])

    # 6. UMAP on a stratified subsample -------------------------------------------
    log("6. UMAP")
    adata.obsm['X_umap'], umap_rows = fit_umap(adata, args, rng, args.cuml)
    adata.obs['umap_subsample'] = False
    adata.obs.iloc[umap_rows, adata.obs.columns.get_loc('umap_subsample')] = True
    u = adata.obs.iloc[umap_rows]
    umap_df = pd.DataFrame({
        'cell_id': u.index.astype(str),
        'umap_1': adata.obsm['X_umap'][umap_rows, 0],
        'umap_2': adata.obsm['X_umap'][umap_rows, 1],
        'organism': pd.Categorical(u[SPECIES_COL].astype(str)),
        'dataset': pd.Categorical(u[DATASET_COL].astype(str)),
        'predicted_labels': pd.Categorical(u['predicted_labels'].astype(str)),
        'sample_type': pd.Categorical(np.where(u[DATASET_COL].astype(str).isin(ORGANOID_DATASETS),
                                               'organoid', 'in_vivo')),
    })
    umap_df.to_parquet(Path(args.outdir) / 'umap_subsample.parquet', index=False)

    # 7. Save ----------------------------------------------------------------------
    log("7. Saving")
    adata.X = counts            # downstream steps sum raw counts
    for c in adata.obs.columns:
        if adata.obs[c].dtype == object:
            adata.obs[c] = adata.obs[c].astype(str)
    adata.uns['concord'] = {'version': version, 'features': len(features),
                            'params': {k: str(v) for k, v in params.items() if k != 'device'}}
    adata.write_h5ad(args.output)
    ann_cols = [SPECIES_COL, DATASET_COL, LABEL_COL, 'predicted_labels', 'label_source']
    if args.label_transfer:
        ann_cols += ['concord_label', 'predicted_labels_confidence', 'holdout']
    adata.obs[ann_cols].to_parquet(Path(args.outdir) / 'concord_cell_annotations.parquet')

    build_info = {
        'build_date': dt.date.today().isoformat(),
        'n_cells': int(adata.n_obs),
        'integration': {
            'method': 'CONCORD',
            'version': version,
            'domain_key': DATASET_COL,
            'n_features': len(features),
            'feature_selection': (f"seurat_v3 HVGs (batch = {DATASET_COL}) among genes detected in "
                                  f">= {args.min_species_frac:.1%} of cells of every species"),
            'latent_dim': args.latent_dim,
            'n_epochs': args.n_epochs,
            'batch_size': args.batch_size,
            'label_transfer': ("the CONCORD semi-supervised classifier head (unlabelled class 'Unknown'); "
                               + ("author labels kept where available" if args.keep_author_labels
                                  else "predictions used for all cells")) if args.label_transfer else 'none',
            'holdout_accuracy': None if np.isnan(acc) else round(acc, 4),
            'umap': (f"UMAP (cosine, n_neighbors = 30, min_dist = 0.1) of the CONCORD latent space on a "
                     f"dataset-stratified subsample of {len(umap_rows):,} cells"),
        },
    }
    (Path(args.outdir) / 'build_info.json').write_text(json.dumps(build_info, indent=2))
    log(f"Done. Wrote {args.output} and {args.outdir}/ (copy build_info.json and umap_subsample.parquet "
        "into the app's data/ folder).")


if __name__ == '__main__':
    main()
