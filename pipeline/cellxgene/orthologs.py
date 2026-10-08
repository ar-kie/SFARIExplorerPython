"""Species -> human ortholog tables from Ensembl Compara (BioMart) and gene-ID mapping.

Mapping policy (``policy``):
  unique_human (default)  keep species genes that have exactly one human ortholog; several
                          species genes mapping to the same human gene (e.g. zebrafish
                          ohnologs shank3a/shank3b -> SHANK3) are summed
  one2one                 keep only ortholog_one2one pairs
Species genes with several human orthologs are dropped in both policies (ambiguous).
"""

import io
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

BIOMART_DATASET = {'Mouse': 'mmusculus_gene_ensembl', 'Zebrafish': 'drerio_gene_ensembl',
                   'Drosophila': 'dmelanogaster_gene_ensembl'}
ATTRIBUTES = ['ensembl_gene_id', 'external_gene_name', 'hsapiens_homolog_ensembl_gene',
              'hsapiens_homolog_associated_gene_name', 'hsapiens_homolog_orthology_type',
              'hsapiens_homolog_orthology_confidence', 'hsapiens_homolog_perc_id']
COLUMNS = ['gene_id', 'gene_name', 'human_gene_id', 'human_gene_name', 'orthology_type', 'confidence', 'perc_id']


def _http_get(url: str, timeout: int = 600, max_redirects: int = 5):
    """GET following redirects (urllib in Python < 3.11 does not follow HTTP 308)."""
    for _ in range(max_redirects + 1):
        req = urllib.request.Request(url, headers={'User-Agent': 'SFARIExplorer-pipeline'})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read(), r.geturl()
        except urllib.error.HTTPError as e:
            if e.code in (301, 302, 303, 307, 308) and e.headers.get('Location'):
                url = urllib.parse.urljoin(url, e.headers['Location'])
                continue
            raise
    raise RuntimeError(f'too many redirects: {url}')


def fetch_orthologs(species: str, out_dir: Path, host: str = 'https://www.ensembl.org', retries: int = 3) -> Path:
    """Download the species -> human ortholog table; returns the TSV path (provenance in .json)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    attrs = ''.join(f'<Attribute name="{a}"/>' for a in ATTRIBUTES)
    query = ('<?xml version="1.0" encoding="UTF-8"?><!DOCTYPE Query><Query virtualSchemaName="default" '
             'formatter="TSV" header="0" uniqueRows="1" datasetConfigVersion="0.6">'
             f'<Dataset name="{BIOMART_DATASET[species]}" interface="default">{attrs}</Dataset></Query>')
    url = f"{host.rstrip('/')}/biomart/martservice?" + urllib.parse.urlencode({'query': query})
    for attempt in range(retries):
        try:
            body, final_url = _http_get(url)
            break
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(10 * (attempt + 1))
    text = body.decode()
    if text.lstrip().startswith(('Query ERROR', '<html', '<!DOCTYPE')):
        raise RuntimeError(f'BioMart error for {species}: {text[:300]}')
    df = pd.read_csv(io.StringIO(text), sep='\t', header=None, names=COLUMNS, dtype=str)
    path = out_dir / f'{species.lower()}_to_human.tsv'
    df.to_csv(path, sep='\t', index=False)
    (out_dir / f'{species.lower()}_to_human.json').write_text(json.dumps({
        'species': species, 'biomart_dataset': BIOMART_DATASET[species],
        'ensembl_host': urllib.parse.urlparse(final_url).netloc, 'retrieved': time.strftime('%Y-%m-%d'),
        'n_rows': int(len(df)), 'n_with_human_ortholog': int(df['human_gene_id'].notna().sum())}, indent=2))
    return path


def ortholog_map(table: Path, policy: str = 'unique_human', min_confidence: int = 0) -> pd.Series:
    """Species Ensembl gene ID -> human gene symbol."""
    df = pd.read_csv(table, sep='\t', dtype=str).dropna(subset=['human_gene_id', 'human_gene_name'])
    df = df[df['human_gene_name'].str.len() > 0]
    if min_confidence:
        df = df[pd.to_numeric(df['confidence'], errors='coerce').fillna(0) >= min_confidence]
    if policy == 'one2one':
        df = df[df['orthology_type'] == 'ortholog_one2one']
    elif policy != 'unique_human':
        raise ValueError(f'unknown policy: {policy}')
    n_human = df.groupby('gene_id')['human_gene_id'].nunique()
    df = df[df['gene_id'].isin(n_human.index[n_human == 1])]
    return df.drop_duplicates('gene_id').set_index('gene_id')['human_gene_name']
