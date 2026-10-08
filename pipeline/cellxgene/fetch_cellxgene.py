#!/usr/bin/env python3
"""
Find, download and prepare developmental brain single-cell / single-nucleus datasets
from CZ CELLxGENE Discover for the SFARIExplorer pipeline.

Species: Human, Mouse, Zebrafish, Drosophila (Discover currently hosts no Drosophila data;
the option is kept so new submissions are picked up automatically).

Workflow (default output directory: $SFARI_ROOT/data/cellxgene):

  1. search     query the public Curation API and write manifest.tsv: one row per candidate
                dataset with metadata, parsed age range and an `include` column + `reason`
  2. (review)   edit manifest.tsv: set include to True/False as needed
  3. download   fetch the H5AD of every included dataset (resumable, size-checked)
  4. orthologs  fetch species -> human ortholog tables from Ensembl Compara (BioMart)
  5. prepare    write pipeline-ready h5ad files: raw counts, human gene symbols, primary
                normal CNS cells, harmonised obs (dataset, organism, donor_id, age_label,
                numeric_time, cell_type, ...)
  6. register   write registry.json, which pipeline/config.py and the downstream scripts read
                (see pipeline/dataset_registry.py), and dataset_references.json for the app

  python pipeline/cellxgene/fetch_cellxgene.py search
  python pipeline/cellxgene/fetch_cellxgene.py run          # steps 3-6 for included rows

Selection rules (search; all recorded in the manifest):
  * species in --species; not tombstoned; public
  * at least one central-nervous-system tissue (retina and whole-embryo datasets only with
    --include-retina / --include-whole-embryo; prepare then keeps CNS cells only)
  * a single-cell / single-nucleus RNA assay; spatial datasets excluded
  * contains normal (healthy) cells unless --include-disease
  * at least one developmental stage (human < 18 y, mouse < 8 weeks, zebrafish < 90 days, or a
    prenatal/juvenile term) unless --include-adult
  * contains primary data (datasets that only re-deposit cells published elsewhere are excluded
    to avoid counting cells twice); at least --min-cells cells
  * not already in the atlas (matched by publication DOI)
"""

import argparse
import datetime as dt
import json
import re
import shutil
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from dataset_registry import SFARI_ROOT  # noqa: E402
from ontology import classify, classify_diseases, disease_allowed, split_disease  # noqa: E402
from orthologs import _http_get, fetch_orthologs, ortholog_map  # noqa: E402
from stages import format_age, parse_stage  # noqa: E402

API = 'https://api.cellxgene.cziscience.com/curation/v1'
DEFAULT_OUT = Path(SFARI_ROOT) / 'data' / 'cellxgene'

SPECIES = {'human': ('Homo sapiens', 'Human'), 'mouse': ('Mus musculus', 'Mouse'),
           'zebrafish': ('Danio rerio', 'Zebrafish'), 'drosophila': ('Drosophila melanogaster', 'Drosophila')}
LATIN_TO_SPECIES = {latin: name for latin, name in SPECIES.values()}
SPECIES_ORDER = ['Human', 'Mouse', 'Zebrafish', 'Drosophila']

CNS = re.compile(
    r'brain|cortex|cortical|cerebr|hippocamp|thalam|striat|cerebell|midbrain|hindbrain|forebrain|telenceph|'
    r'dienceph|mesenceph|rhombenceph|metenceph|myelenceph|spinal cord|neural tube|neural crest|nervous system|'
    r'ganglionic eminence|pons|medulla oblongata|hypothalam|amygdal|caudate|putamen|nucleus accumbens|'
    r'globus pallidus|substantia nigra|olfactory bulb|neocortex|prefrontal|entorhinal|claustrum|choroid plexus|'
    r'white matter|gr[ae]y matter|ventricular zone|subventricular|dentate gyrus|septum|habenula|pallium|'
    r'neuroepithel|brainstem|brain stem', re.I)
RETINA = re.compile(r'retina|optic|eye\b', re.I)
CULTURE = re.compile(r'in ?vitro|cultures?\b', re.I)
FLAGS = {'tumour': re.compile(r'tumou?r|cancer|glioma|astrocytoma|blastoma|carcinoma', re.I),
         'treatment/disease model': re.compile(r'treated|knock-?out|mutant|perturb|crispr|syndrome|disorder|disease|'
                                               r'epilep|autism|schizophren|alzheimer|parkinson', re.I)}
MENINGES = re.compile(r'meninx|mening|dura mater|pia mater|arachnoid', re.I)
WHOLE = re.compile(r'whole organism|^embryo$|^head$', re.I)
RNA_ASSAY = re.compile(
    r"10x (3'|5'|multiome)|smart-?seq|sci-rna-seq|drop-?seq|dronc-seq|split-seq|parse evercode|bd rhapsody|"
    r"indrop|cel-seq|strt-seq|quartz-seq|scrb-seq|microwell-seq|gexscope|smarter|seq-well|mars-seq|"
    r"single cell library|snrna|scrna", re.I)

# Publications already in the atlas (collection DOI -> display name).
ATLAS_DOIS = {
    '10.1038/s41586-021-03910-8': 'Bhaduri (2021)', '10.1126/science.adf1226': 'Braun (2023)',
    '10.1038/s41586-024-08172-8': 'He (2024)', '10.1126/science.aav8130': 'Velmeshev (2019)',
    '10.1126/science.adf0834': 'Velmeshev (2023)', '10.1038/s41467-022-33364-z': 'Wang (2022)',
    '10.1038/s41586-024-08351-7': 'Wang (2025)', '10.1126/sciadv.adg3754': 'Zhu (2023)',
    '10.1038/s41586-021-03775-x': 'La Manno (2021)', '10.1038/s41586-024-08350-8': 'Jin (2025)',
    '10.1038/s41588-023-01572-y': 'Sziraki (2023)', '10.1016/j.neuron.2020.09.023': 'Raj (2020)',
    '10.1016/j.cell.2018.05.057': 'Davie (2018)',
}


def log(msg):
    print(f"[{dt.datetime.now():%H:%M:%S}] {msg}", flush=True)


def get_json(url):
    body, _ = _http_get(url, timeout=300)
    return json.loads(body)


def norm_doi(doi):
    if not doi:
        return ''
    return re.sub(r'^https?://(dx\.)?doi\.org/', '', str(doi).strip().lower())


def slug(text):
    return re.sub(r'[^A-Za-z0-9]+', '', str(text)) or 'NA'


# =============================================================================
# search
# =============================================================================

def search(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    wanted = {SPECIES[s][0] for s in args.species}
    log('Querying CELLxGENE Discover (datasets, collections)')
    datasets = get_json(f'{API}/datasets')
    collections = {c['collection_id']: c for c in get_json(f'{API}/collections')}
    log(f'  {len(datasets):,} public datasets, {len(collections):,} collections')
    cand = [ds for ds in datasets if any(o['label'] in wanted for o in ds.get('organism', []))]
    terms = {t['ontology_term_id']: t['label'] for ds in cand for t in ds.get('tissue', [])}
    log(f'  classifying {len(terms)} tissue terms by ontology ancestry (cached in ontology_cache.json)')
    is_cns = classify(terms, out / 'ontology_cache.json', CNS)
    dterms = {p for ds in cand for d in ds.get('disease', []) for p in split_disease(d['ontology_term_id'])}
    log(f'  classifying {len(dterms)} donor-condition terms (MONDO ancestry, cached in disease_cache.json)')
    healthy = classify_diseases(dterms, out / 'disease_cache.json')

    rows = []
    for ds in datasets:
        if ds.get('tombstone') or ds.get('visibility', 'PUBLIC') != 'PUBLIC':
            continue
        latin = [o['label'] for o in ds.get('organism', []) if o['label'] in wanted]
        if not latin:
            continue
        species = LATIN_TO_SPECIES[latin[0]]
        tissues = ds.get('tissue', [])
        cns = [t['label'] for t in tissues if is_cns.get(t['ontology_term_id'])
               and (args.include_retina or not RETINA.search(t['label']))]
        if args.include_retina:
            cns += [t['label'] for t in tissues if RETINA.search(t['label']) and t['label'] not in cns]
        whole = [t['label'] for t in tissues if WHOLE.search(t['label'])]
        if not cns and not (args.include_whole_embryo and whole):
            continue

        assays = [a['label'] for a in ds.get('assay', [])]
        diseases = [d['label'] for d in ds.get('disease', [])]
        dis_out = [d['label'] for d in ds.get('disease', [])
                   if not disease_allowed(d['ontology_term_id'], healthy, args.disease_policy)]
        stages = ds.get('development_stage', [])
        parsed = [(s['label'], parse_stage(species, s['label'])) for s in stages]
        ages = [p['age'] for _, p in parsed if p['age'] is not None]
        n_dev = sum(bool(p['developmental']) for _, p in parsed)
        primary = ds.get('is_primary_data') or []
        tissue_types = {t.get('tissue_type', 'tissue') for t in tissues}
        n_cells = int(ds.get('cell_count') or 0)
        n_primary = int(ds.get('primary_cell_count') or 0)
        h5ad = next((a for a in ds.get('assets', []) if a.get('filetype') == 'H5AD'), {})
        doi = norm_doi(ds.get('collection_doi'))
        coll = collections.get(ds.get('collection_id'), {})
        pub = coll.get('publisher_metadata') or {}
        authors = pub.get('authors') or []
        first = authors[0].get('family', authors[0].get('name', '')) if authors else ''
        if not first:
            contact = (coll.get('contact_name') or '').replace(',', ' ').split()
            first = contact[-1] if contact else ((coll.get('consortia') or [''])[0])
        year = pub.get('published_year') or (str(ds.get('published_at', ''))[:4] or '')

        reasons = []
        if not any(RNA_ASSAY.search(a) for a in assays):
            reasons.append('no single-cell RNA assay')
        if ds.get('spatial'):
            reasons.append('spatial')
        if len(dis_out) == len(diseases):
            reasons.append(f'no donors allowed by disease policy ({args.disease_policy})')
        if not n_dev and not args.include_adult:
            reasons.append('no developmental stage')
        if True not in primary:
            reasons.append('secondary data only')
        if tissue_types & {'cell culture', 'primary cell culture'} or \
                (CULTURE.search(ds.get('title', '')) and 'organoid' not in tissue_types):
            reasons.append('cell culture')
        if n_cells < args.min_cells:
            reasons.append(f'< {args.min_cells} cells')
        if doi in ATLAS_DOIS:
            reasons.append(f'in atlas: {ATLAS_DOIS[doi]}')

        flags = [f for f, rx in FLAGS.items() if rx.search(f"{ds.get('title', '')} {coll.get('name', '')}")]
        if cns and all(MENINGES.search(t) for t in cns):
            flags.append('meninges only')
        if len(tissues) > len(cns) + 2:
            flags.append('multi-tissue (CNS cells kept)')
        rows.append({
            'include': not reasons, 'reason': '; '.join(reasons), 'review_flags': '; '.join(flags),
            'species': species, 'dataset_id': ds['dataset_id'], 'dataset_version_id': ds.get('dataset_version_id'),
            'title': ds.get('title', ''), 'collection_id': ds.get('collection_id'),
            'collection_name': ds.get('collection_name', coll.get('name', '')),
            'collection_doi': doi, 'first_author': first, 'year': year, 'journal': pub.get('journal', ''),
            'tissues_cns': '; '.join(sorted(set(cns + (whole if args.include_whole_embryo else [])))),
            'n_tissues': len(tissues),
            'sample_type': 'organoid' if tissue_types == {'organoid'} else
                           ('mixed' if 'organoid' in tissue_types else 'in_vivo'),
            'assays': '; '.join(assays), 'suspension_type': '; '.join(ds.get('suspension_type', [])),
            'disease': '; '.join(diseases), 'disease_excluded': '; '.join(dis_out), 'n_cells': n_cells, 'n_primary_cells': n_primary,
            'n_stages': len(stages), 'n_dev_stages': n_dev,
            'age_min': min(ages) if ages else None, 'age_max': max(ages) if ages else None,
            'age_range': ('n/a (organoid: donor stage only)' if 'organoid' in tissue_types else
                          f"{format_age(species, min(ages))} - {format_age(species, max(ages))}" if ages else 'n/a'),
            'stages': '; '.join(sorted({l for l, _ in parsed}))[:400],
            'schema_version': ds.get('schema_version'), 'published_at': str(ds.get('published_at', ''))[:10],
            'revised_at': str(ds.get('revised_at') or '')[:10], 'explorer_url': ds.get('explorer_url'),
            'h5ad_url': h5ad.get('url'), 'h5ad_gb': round((h5ad.get('filesize') or 0) / 1e9, 2),
            'h5ad_bytes': int(h5ad.get('filesize') or 0),
        })

    m = pd.DataFrame(rows)
    if m.empty:
        log('No candidate datasets.')
        return
    m = _assign_names(m)
    m['_s'] = m['species'].map({s: i for i, s in enumerate(SPECIES_ORDER)})
    m = m.sort_values(['include', '_s', 'n_cells'], ascending=[False, True, False]).drop(columns='_s')
    path = out / 'manifest.tsv'
    m.to_csv(path, sep='\t', index=False)
    inc = m[m['include']]
    log(f'Wrote {path}: {len(m)} CNS candidates, {len(inc)} included')
    for sp, g in inc.groupby('species', sort=False):
        log(f'  {sp:10s} {len(g):3d} datasets  {g["n_cells"].sum():>11,} cells  {g["h5ad_gb"].sum():7.1f} GB  '
            f'{g["display"].nunique()} publications')
    excluded = m.loc[~m['include'], 'reason'].str.split('; ').explode().value_counts()
    log('  Exclusion reasons (a dataset can have several):')
    for r, n in excluded.items():
        log(f'    {n:4d}  {r}')


def _assign_names(m):
    """'Author (Year)' per publication; suffix b, c, ... when names collide (incl. atlas names)."""
    taken = set(ATLAS_DOIS.values())
    names = {}
    for cid, g in m.groupby('collection_id', sort=False):
        r = g.iloc[0]
        doi = r['collection_doi']
        if doi in ATLAS_DOIS:
            names[cid] = ATLAS_DOIS[doi]
            continue
        base = f"{r['first_author'] or slug(r['collection_name'])[:20]} ({r['year'] or 'n.d.'})"
        name, k = base, 1
        while name in taken:
            name = base[:-1] + f"{'bcdefghij'[k - 1]})"
            k += 1
        taken.add(name)
        names[cid] = name
    m['display'] = m['collection_id'].map(names)
    m['key'] = [f"CXG-{slug(a) or 'NA'}-{y or 'NA'}-{d[:8]}" for a, y, d in zip(m['first_author'], m['year'], m['dataset_id'])]
    return m


# =============================================================================
# download
# =============================================================================

def _selected(args):
    m = pd.read_csv(Path(args.out) / 'manifest.tsv', sep='\t', dtype={'dataset_id': str})
    m['include'] = m['include'].astype(str).str.lower().isin(['true', '1', 'yes'])
    if getattr(args, 'ids', None):
        return m[m['dataset_id'].isin(args.ids)]
    return m[m['include']]


def _download(url, dest: Path, expected: int):
    part = dest.with_suffix(dest.suffix + '.part')
    start = part.stat().st_size if part.exists() else 0
    req = urllib.request.Request(url, headers={'User-Agent': 'SFARIExplorer-pipeline',
                                               **({'Range': f'bytes={start}-'} if start else {})})
    with urllib.request.urlopen(req, timeout=600) as r:
        mode = 'ab' if start and r.status == 206 else 'wb'
        done = start if mode == 'ab' else 0
        step, nxt = max(expected // 20, 1), done
        with open(part, mode) as fh:
            while chunk := r.read(8 << 20):
                fh.write(chunk)
                done += len(chunk)
                if expected and done >= nxt:
                    log(f'    {dest.name}: {done / 1e9:.2f} / {expected / 1e9:.2f} GB')
                    nxt += step
    if expected and part.stat().st_size != expected:
        raise IOError(f'size mismatch for {dest.name}: {part.stat().st_size} != {expected}')
    part.rename(dest)


def download(args):
    sel = _selected(args)
    raw = Path(args.out) / 'raw'
    raw.mkdir(parents=True, exist_ok=True)
    need = int(sel['h5ad_bytes'].sum())
    free = shutil.disk_usage(raw).free
    log(f'{len(sel)} datasets, {need / 1e9:.1f} GB to {raw} ({free / 1e9:.0f} GB free)')
    if need > free and not args.force:
        sys.exit('Not enough free space (use --force to try anyway).')
    for _, r in sel.iterrows():
        dest = raw / f"{r['dataset_id']}.h5ad"
        expected = int(r['h5ad_bytes'])
        if dest.exists() and (not expected or dest.stat().st_size == expected):
            log(f'  exists: {dest.name}')
            continue
        log(f"  downloading {r['display']} | {r['title'][:60]} ({r['h5ad_gb']} GB)")
        for attempt in range(3):
            try:
                _download(r['h5ad_url'], dest, expected)
                break
            except Exception as e:
                log(f'    attempt {attempt + 1} failed: {e}')
                time.sleep(30 * (attempt + 1))
        else:
            log(f"  FAILED: {r['dataset_id']}")


# =============================================================================
# orthologs
# =============================================================================

def orthologs(args):
    sel = _selected(args)
    for sp in sorted(set(sel['species']) - {'Human'}):
        path = fetch_orthologs(sp, Path(args.out) / 'orthologs', host=args.ensembl_host)
        log(f'  {sp}: {path} ({json.loads(path.with_suffix(".json").read_text())["ensembl_host"]})')


# =============================================================================
# prepare
# =============================================================================

def prepare(args):
    import anndata as ad
    from scipy import sparse

    out = Path(args.out)
    prep_dir = out / 'prepared'
    prep_dir.mkdir(parents=True, exist_ok=True)
    maps = {}
    for _, r in _selected(args).iterrows():
        src = out / 'raw' / f"{r['dataset_id']}.h5ad"
        dst = prep_dir / f"{r['key']}.h5ad"
        if not src.exists():
            log(f"  not downloaded: {r['dataset_id']}")
            continue
        if dst.exists() and not args.overwrite:
            log(f'  prepared: {dst.name}')
            continue
        species = r['species']
        if species != 'Human' and species not in maps:
            table = out / 'orthologs' / f'{species.lower()}_to_human.tsv'
            if not table.exists():
                sys.exit(f'Missing {table}; run the orthologs step first.')
            maps[species] = ortholog_map(table, policy=args.orthology, min_confidence=args.min_confidence)
        log(f"  preparing {r['display']} | {r['title'][:60]}")
        info = prepare_one(src, dst, r, maps.get(species), args, ad, sparse)
        (prep_dir / f"{r['key']}.json").write_text(json.dumps(info, indent=2))
        log(f"    {info['n_cells']:,} cells x {info['n_genes']:,} human genes -> {dst.name}")


def prepare_one(src, dst, r, omap, args, ad, sparse):
    species, latin = r['species'], SPECIES[r['species'].lower()][0]
    a = ad.read_h5ad(src, backed='r')            # stream counts; obs/var are in memory
    use_raw = a.raw is not None
    Xsrc = a.raw.X if use_raw else a.X
    var = (a.raw.var if use_raw else a.var).copy()
    if 'feature_name' not in var and 'feature_name' in a.var:
        var['feature_name'] = a.var['feature_name'].reindex(var.index)
    obs = a.obs
    keep = np.ones(a.n_obs, bool)
    if 'is_primary_data' in obs:
        keep &= obs['is_primary_data'].astype(bool).to_numpy()
    if 'organism' in obs:
        keep &= (obs['organism'].astype(str) == latin).to_numpy()
    if args.disease_policy != 'all':
        if 'disease_ontology_term_id' in obs:
            ids_d = obs['disease_ontology_term_id'].astype(str)
            healthy = classify_diseases({p for t in ids_d.unique() for p in split_disease(t)},
                                        Path(args.out) / 'disease_cache.json')
            allowed = {t: disease_allowed(t, healthy, args.disease_policy) for t in ids_d.unique()}
            keep &= ids_d.map(allowed).to_numpy(bool)
        elif 'disease' in obs:
            keep &= (obs['disease'].astype(str) == 'normal').to_numpy()
    if 'tissue' in obs:   # keep CNS cells only (multi-tissue datasets), same rule as search
        labels = obs['tissue'].astype(str)
        ids = obs['tissue_ontology_term_id'].astype(str) if 'tissue_ontology_term_id' in obs else labels
        terms = dict(zip(ids, labels))
        cns = classify(terms, Path(args.out) / 'ontology_cache.json', CNS)
        retina = labels.str.contains(RETINA)
        ok = ids.map(cns).fillna(False).astype(bool) & (~retina if not args.include_retina else True)
        if args.include_retina:
            ok |= retina
        keep &= ok.to_numpy()

    ids = pd.Index(var.index.astype(str))
    if species == 'Human':
        target = pd.Series(var['feature_name'].astype(str).values if 'feature_name' in var else ids, index=ids)
        target = target.where(~target.str.match(r'^ENSG\d+') & (target.str.len() > 0))
    else:
        target = pd.Series(ids.map(omap), index=ids)
    valid = np.flatnonzero(target.notna().to_numpy())
    codes, genes = pd.factorize(target.iloc[valid])
    M = sparse.csr_matrix((np.ones(len(valid), np.float32), (np.arange(len(valid)), codes)),
                          shape=(len(valid), len(genes)))   # sums paralogs / duplicate symbols

    rows = np.flatnonzero(keep)
    blocks, checked = [], False
    for start in range(0, a.n_obs, args.chunk):
        stop = min(start + args.chunk, a.n_obs)
        sel = rows[(rows >= start) & (rows < stop)] - start
        if not len(sel):
            continue
        block = sparse.csr_matrix(Xsrc[start:stop])[sel]
        if not checked and block.nnz:
            probe = block.data[:100_000]
            if not np.allclose(probe, np.round(probe)):
                raise ValueError(f"{r['dataset_id']}: counts are not integers (no raw counts found)")
            checked = True
        blocks.append((block[:, valid].astype(np.float32) @ M).tocsr())
    Xh = sparse.vstack(blocks, format='csr') if blocks else sparse.csr_matrix((0, len(genes)), dtype=np.float32)
    obs = obs.iloc[rows]
    a.file.close()

    def col(name, default=''):
        return obs[name].astype(str).to_numpy() if name in obs else np.full(len(obs), default, object)

    stage = col('development_stage', 'unknown')
    parsed = {s: parse_stage(species, s) for s in set(stage)}
    tissue_type = col('tissue_type', 'tissue')
    sample_type = np.where(tissue_type == 'organoid', 'organoid', 'in_vivo')
    age_label = np.array([parsed[s]['tag'] for s in stage], object)
    numeric = np.array([np.nan if parsed[s]['age'] is None else parsed[s]['age'] for s in stage], float)
    organoid = sample_type == 'organoid'
    if organoid.any():   # donor stage of an iPSC line is not the organoid age
        if args.organoid_age_col and args.organoid_age_col in obs:
            days = pd.to_numeric(obs[args.organoid_age_col], errors='coerce').to_numpy()
            age_label[organoid] = [f'{d:g} dic' if np.isfinite(d) else 'unknown' for d in days[organoid]]
            numeric[organoid] = days[organoid]
        else:
            age_label[organoid], numeric[organoid] = 'unknown', np.nan

    new_obs = pd.DataFrame({
        'dataset': r['display'], 'organism': species, 'sample_type': sample_type,
        'donor_id': col('donor_id', 'unknown'), 'development_stage': stage,
        'development_stage_ontology_term_id': col('development_stage_ontology_term_id'),
        'age_label': age_label, 'numeric_time': numeric,
        'cell_type': col('cell_type', 'unknown'), 'cell_type_ontology_term_id': col('cell_type_ontology_term_id'),
        'tissue': col('tissue'), 'assay': col('assay'), 'suspension_type': col('suspension_type'),
        'disease': col('disease'), 'sex': col('sex'), 'cellxgene_dataset_id': r['dataset_id'],
    }, index=pd.Index([f"{r['key']}:{c}" for c in obs.index.astype(str)], name='cell_id'))
    for c in new_obs.columns:
        if new_obs[c].dtype == object:
            new_obs[c] = new_obs[c].astype('category')
    out = ad.AnnData(X=Xh, obs=new_obs, var=pd.DataFrame(index=pd.Index(genes.astype(str), name='gene')))
    provenance = {
        'source': 'CZ CELLxGENE Discover', 'dataset_id': r['dataset_id'],
        'dataset_version_id': str(r.get('dataset_version_id', '')), 'collection_doi': r['collection_doi'],
        'gene_mapping': 'feature_name' if species == 'Human' else f'Ensembl Compara orthologs ({args.orthology})',
        'cell_filters': f'primary cells, CNS tissues, donor conditions: {args.disease_policy}',
        'prepared': dt.date.today().isoformat(),
    }
    if species != 'Human':
        meta = Path(args.out) / 'orthologs' / f'{species.lower()}_to_human.json'
        if meta.exists():
            provenance['ensembl_host'] = json.loads(meta.read_text()).get('ensembl_host')
    out.uns['cellxgene'] = provenance
    out.write_h5ad(dst, compression='lzf')
    return {
        'key': r['key'], 'display': r['display'], 'organism': species,
        'sample_type': 'organoid' if organoid.all() else 'in_vivo', 'h5ad': str(dst),
        'sample_col': 'donor_id', 'time_col': 'age_label', 'include': True,
        'dataset_id': r['dataset_id'], 'collection_doi': r['collection_doi'], 'title': r['title'],
        'reference': f"{r['first_author']} et al., {r['journal'] or 'unpublished'} ({r['year']})",
        'n_cells': int(out.n_obs), 'n_genes': int(out.n_vars),
        'age_range': r.get('age_range', ''), **{k: v for k, v in provenance.items() if k != 'source'},
    }


# =============================================================================
# register
# =============================================================================

def register(args):
    out = Path(args.out)
    entries = [json.loads(p.read_text()) for p in sorted((out / 'prepared').glob('*.json'))]
    entries = [e for e in entries if Path(e['h5ad']).exists()]
    registry = {'generated': dt.datetime.now().isoformat(timespec='seconds'), 'source': 'CZ CELLxGENE Discover',
                'datasets': entries}
    (out / 'registry.json').write_text(json.dumps(registry, indent=2))
    refs = {}
    for e in entries:
        refs.setdefault(e['display'], {'reference': e['reference'], 'doi': e['collection_doi'],
                                       'scope': e['title'][:80]})
    (out / 'dataset_references.json').write_text(json.dumps(refs, indent=2))
    log(f"Registered {len(entries)} prepared datasets ({len(refs)} publications) in {out / 'registry.json'}")
    log('Pipeline steps 01-05 now include them (pipeline/config.py). Copy dataset_references.json into the '
        "app's data/ folder to show their references.")


def run(args):
    download(args)
    orthologs(args)
    prepare(args)
    register(args)


def main():
    p = argparse.ArgumentParser(description='CELLxGENE Discover -> SFARIExplorer pipeline',
                                formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument('--out', default=str(DEFAULT_OUT), help=f'output directory (default {DEFAULT_OUT})')
    sub = p.add_subparsers(dest='cmd', required=True)

    s = sub.add_parser('search', help='write manifest.tsv of candidate datasets')
    s.add_argument('--species', nargs='+', default=list(SPECIES), choices=list(SPECIES))
    s.add_argument('--min-cells', type=int, default=1000)
    s.add_argument('--include-adult', action='store_true', help='also datasets without developmental stages')
    s.add_argument('--disease-policy', default='brain-healthy', choices=['brain-healthy', 'normal-only', 'all'],
                   help='donor conditions allowed (default: none affecting the brain; see ontology.py)')
    s.add_argument('--include-retina', action='store_true')
    s.add_argument('--include-whole-embryo', action='store_true')
    s.set_defaults(func=search)

    def common(sp):
        sp.add_argument('--ids', nargs='+', help='dataset_ids (default: rows with include=True)')
        sp.add_argument('--disease-policy', default='brain-healthy', choices=['brain-healthy', 'normal-only', 'all'],
                        help='cells kept by donor condition in prepare (default: none affecting the brain)')
        sp.add_argument('--include-retina', action='store_true')
        sp.add_argument('--force', action='store_true', help='download even if disk space looks insufficient')
        sp.add_argument('--ensembl-host', default='https://www.ensembl.org',
                        help='pin a release, e.g. https://jun2026.archive.ensembl.org')
        sp.add_argument('--orthology', default='unique_human', choices=['unique_human', 'one2one'])
        sp.add_argument('--min-confidence', type=int, default=0, choices=[0, 1],
                        help='1 = high-confidence Ensembl orthologs only')
        sp.add_argument('--organoid-age-col', help='obs column with organoid age in days, if present')
        sp.add_argument('--overwrite', action='store_true', help='re-prepare existing outputs')
        sp.add_argument('--chunk', type=int, default=100_000, help='cells per block when streaming counts')

    for name, func, hlp in [('download', download, 'download H5ADs of included datasets'),
                            ('orthologs', orthologs, 'fetch ortholog tables (non-human species)'),
                            ('prepare', prepare, 'write pipeline-ready h5ad files'),
                            ('register', register, 'write registry.json for the pipeline'),
                            ('run', run, 'download + orthologs + prepare + register')]:
        sp = sub.add_parser(name, help=hlp)
        common(sp)
        sp.set_defaults(func=func)

    args = p.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
