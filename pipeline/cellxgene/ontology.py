"""Is a tissue term part of the central nervous system? (UBERON / ZFA ancestry via EBI OLS4)

Region terms such as "middle temporal gyrus", "Brodmann (1909) area 10" or "pallidum" do not
contain a generic keyword, so tissues are classified by their ontology ancestors (is_a and
part_of). Results are cached in a JSON file. Keyword matching is only a fallback when OLS cannot be
reached (keywords alone misfire: "cortex of kidney", "interventricular septum").
"""

import json
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

OLS = 'https://www.ebi.ac.uk/ols4/api/ontologies'
CNS_TERMS = {'central nervous system', 'brain', 'spinal cord', 'neural tube', 'neural rod', 'neural keel'}


def _ancestor_labels(curie: str) -> set:
    prefix, num = curie.split(':', 1)
    iri = f'http://purl.obolibrary.org/obo/{prefix}_{num}'
    enc = urllib.parse.quote(urllib.parse.quote(iri, safe=''), safe='')
    url = f'{OLS}/{prefix.lower()}/terms/{enc}/hierarchicalAncestors?size=1000'
    req = urllib.request.Request(url, headers={'Accept': 'application/json', 'User-Agent': 'SFARIExplorer-pipeline'})
    with urllib.request.urlopen(req, timeout=60) as r:
        data = json.load(r)
    return {t.get('label', '') for t in data.get('_embedded', {}).get('terms', [])}


def classify(terms: dict, cache_path: Path, label_regex, workers: int = 8) -> dict:
    """{curie: label} -> {curie: True/False}; cached by curie."""
    cache_path = Path(cache_path)
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    todo = {}
    for curie, label in terms.items():
        if curie in cache and cache[curie]['source'] != 'label (OLS unreachable)':
            continue
        if label.lower() in CNS_TERMS:
            cache[curie] = {'label': label, 'cns': True, 'source': 'label'}
        elif curie.split(':')[0] in ('UBERON', 'ZFA'):
            # keywords alone are not enough ("cortex of kidney", "interventricular septum")
            todo[curie] = label
        else:
            cache[curie] = {'label': label, 'cns': False, 'source': 'not an anatomy term'}

    def work(item):
        curie, label = item
        try:
            return curie, label, bool(_ancestor_labels(curie) & CNS_TERMS), 'ontology'
        except Exception:
            return curie, label, bool(label_regex.search(label)), 'label (OLS unreachable)'

    if todo:
        with ThreadPoolExecutor(workers) as ex:
            for curie, label, cns, src in ex.map(work, todo.items()):
                cache[curie] = {'label': label, 'cns': cns, 'source': src}
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache, indent=1, sort_keys=True))
    return {c: cache[c]['cns'] for c in terms}


# ---------------------------------------------------------------------------
# Donor diseases
# ---------------------------------------------------------------------------

NORMAL = 'PATO:0000461'
# A donor condition disqualifies brain tissue when it is (or descends from) one of these MONDO classes.
BRAIN_RELEVANT = {'nervous system disorder', 'psychiatric disorder', 'mental disorder', 'neurodevelopmental disorder',
                  'neurodegenerative disease', 'cancer', 'neoplasm', 'chromosomal disorder'}


def classify_diseases(term_ids, cache_path: Path, workers: int = 8) -> dict:
    """{MONDO curie: True if compatible with healthy brain tissue}.

    CELLxGENE records donor conditions, which are often unrelated to the brain (e.g. heart failure as
    cause of death in control donors). Conditions under nervous-system, psychiatric, neoplastic or
    chromosomal disorders are not compatible; anything else is. Unknown when OLS is unreachable -> False.
    """
    cache_path = Path(cache_path)
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    todo = [c for c in set(term_ids) if c not in cache or cache[c]['source'] == 'OLS unreachable']

    def work(curie):
        if curie == NORMAL:
            return curie, True, 'normal'
        if not curie.startswith('MONDO:'):
            return curie, False, 'not a MONDO term'
        try:
            labels = _ancestor_labels(curie) | {_term_label(curie)}
            return curie, not (labels & BRAIN_RELEVANT), 'ontology'
        except Exception:
            return curie, False, 'OLS unreachable'

    if todo:
        with ThreadPoolExecutor(workers) as ex:
            for curie, ok, src in ex.map(work, todo):
                cache[curie] = {'brain_healthy': ok, 'source': src}
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache, indent=1, sort_keys=True))
    return {c: cache[c]['brain_healthy'] for c in set(term_ids)}


def _term_label(curie: str) -> str:
    prefix, num = curie.split(':', 1)
    iri = f'http://purl.obolibrary.org/obo/{prefix}_{num}'
    enc = urllib.parse.quote(urllib.parse.quote(iri, safe=''), safe='')
    req = urllib.request.Request(f'{OLS}/{prefix.lower()}/terms/{enc}',
                                 headers={'Accept': 'application/json', 'User-Agent': 'SFARIExplorer-pipeline'})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r).get('label', '')


def split_disease(term_id: str) -> list:
    """Donors with several conditions carry 'MONDO:a || MONDO:b'."""
    return [t.strip() for t in str(term_id).split('||') if t.strip()]


def disease_allowed(term_id: str, healthy: dict, policy: str) -> bool:
    """policy: brain-healthy (every condition compatible), normal-only, all."""
    if policy == 'all':
        return True
    parts = split_disease(term_id)
    if policy == 'normal-only':
        return parts == [NORMAL]
    return bool(parts) and all(healthy.get(p, False) for p in parts)
