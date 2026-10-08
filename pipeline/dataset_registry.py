"""Per-dataset settings for datasets added after the original 13.

The original datasets keep their explicit settings in the scripts that use them. Datasets
fetched from CELLxGENE with ``pipeline/cellxgene/fetch_cellxgene.py`` are recorded in
``{SFARI_ROOT}/data/cellxgene/registry.json`` (one entry per prepared dataset) and merged
into those settings here, so a new dataset only has to be registered once:

    config.py                  SYMBOL_DATASETS, DATASET_META        (steps 01-05)
    prep_sfari_data_v8.py      DATASET_SAMPLE_COL, DATASET_TIME_COL, ORGANOID_DATASETS
    create_merged_columns.py   dataset_sample_col, dataset_time_col
    normalize_age.py           ORGANOID_DATASETS, plus "<n> dpc|hpf|dpe|dic" ages
    06_integrate_concord.py    ORGANOID_DATASETS
"""

import json
import os
import re

SFARI_ROOT = os.environ.get('SFARI_ROOT', '/sc/arion/projects/ad-omics/raphael/SFARI')
REGISTRY_PATH = os.environ.get('SFARI_CELLXGENE_REGISTRY', f'{SFARI_ROOT}/data/cellxgene/registry.json')

# Ages computed upstream from ontology terms, in each species' native unit:
#   dpc = days post-conception (human, mouse), hpf = hours post-fertilisation (zebrafish),
#   dpe = days post-eclosion (Drosophila), dic = days in culture (organoids)
TAGGED_AGE = re.compile(r'^\s*(\d+(?:\.\d+)?)\s*(dpc|hpf|dpe|dic)\s*$', re.I)


def parse_tagged_age(value):
    """Return the numeric age of a tagged age string, or None."""
    if value is None:
        return None
    m = TAGGED_AGE.match(str(value))
    return float(m.group(1)) if m else None


def load_registry(path: str = REGISTRY_PATH) -> list:
    if not os.path.exists(path):
        return []
    with open(path) as fh:
        return [e for e in json.load(fh).get('datasets', []) if e.get('include', True)]


def extend_pipeline_config(symbol_datasets: dict, dataset_meta: dict, path: str = REGISTRY_PATH) -> int:
    """Add prepared datasets (human-symbol h5ad, raw counts in X) to steps 01-05."""
    entries = load_registry(path)
    for e in entries:
        symbol_datasets[e['key']] = e['h5ad']
        dataset_meta[e['key']] = {'dataset': e['display'], 'organism': e['organism']}
    return len(entries)


def extend_dataset_maps(sample_col: dict = None, time_col: dict = None, organoid_datasets: list = None,
                        path: str = REGISTRY_PATH) -> int:
    """Add sample/time columns and organoid status keyed by display name."""
    entries = load_registry(path)
    for e in entries:
        if sample_col is not None:
            sample_col.setdefault(e['display'], e.get('sample_col', 'donor_id'))
        if time_col is not None:
            time_col.setdefault(e['display'], e.get('time_col', 'age_label'))
        if organoid_datasets is not None and e.get('sample_type') == 'organoid' \
                and e['display'] not in organoid_datasets:
            organoid_datasets.append(e['display'])
    return len(entries)
