"""CELLxGENE development-stage terms -> numeric age in each species' native unit.

Units match the rest of the pipeline (normalize_age.py):
  Human, Mouse  days post-conception (dpc); birth = 280 dpc (human), 20 dpc (mouse)
  Zebrafish     hours post-fertilisation (hpf)
  Drosophila    days post-eclosion (dpe)
Conventions copied from normalize_age.py so that new and existing datasets align:
"Nth week post-fertilization" = 7N dpc, LMP months = 30-day months without the ~2-week
LMP offset, postnatal months = 30 days, years = 365 days. Terms that only name a broad
period ("adult stage", "fetal stage") have no age (None) but still count as
developmental or not.
"""

import re

HUMAN_BIRTH_DPC = 280.0
MOUSE_BIRTH_DPC = 20.0

# Carnegie stages: approximate post-ovulatory days (O'Rahilly & Mueller).
CARNEGIE_DPC = {1: 1, 2: 2.5, 3: 4.5, 4: 5.5, 5: 9.5, 6: 13, 7: 16, 8: 18, 9: 20, 10: 22, 11: 24, 12: 26,
                13: 28, 14: 32, 15: 33, 16: 37, 17: 41, 18: 44, 19: 47.5, 20: 50.5, 21: 52, 22: 54, 23: 56.5}

# Theiler stages: embryonic day (dpc), eMouseAtlas; TS27 = newborn.
THEILER_DPC = {1: 0.5, 2: 1, 3: 2, 4: 3, 5: 4, 6: 4.5, 7: 5, 8: 6, 9: 6.5, 10: 7, 11: 7.5, 12: 8, 13: 8.5,
               14: 9, 15: 9.5, 16: 10, 17: 10.5, 18: 11, 19: 11.5, 20: 12, 21: 13, 22: 14, 23: 15, 24: 16,
               25: 17, 26: 18, 27: MOUSE_BIRTH_DPC}

# Zebrafish stages (Kimmel et al., 1995): hpf at stage onset; multi-day larval periods use midpoints.
ZFS_HPF = {
    '1-cell': 0, '2-cell': 0.75, '4-cell': 1, '8-cell': 1.25, '16-cell': 1.5, '32-cell': 1.75, '64-cell': 2,
    '128-cell': 2.25, '256-cell': 2.5, '512-cell': 2.75, '1k-cell': 3, 'high': 3.3, 'oblong': 3.7, 'sphere': 4,
    'dome': 4.3, '30%-epiboly': 4.7, '50%-epiboly': 5.25, 'germ-ring': 5.7, 'shield': 6, '75%-epiboly': 8,
    '90%-epiboly': 9, 'bud': 10, '1-4 somites': 10.33, '5-9 somites': 11.66, '10-13 somites': 14,
    '14-19 somites': 16, '20-25 somites': 19, '26+ somites': 22, 'prim-5': 24, 'prim-15': 30, 'prim-25': 36,
    'high-pec': 42, 'long-pec': 48, 'pec-fin': 60, 'protruding-mouth': 72, 'day 4': 96, 'day 5': 120,
    'day 6': 144, 'days 7-13': 240, 'days 14-20': 408, 'days 21-29': 600, 'days 30-44': 888,
    'days 45-89': 1608,
}

ORDINALS = {w: i for i, w in enumerate(['first', 'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh',
                                         'eighth', 'ninth', 'tenth'], start=1)}

UNIT = {'Human': 'dpc', 'Mouse': 'dpc', 'Zebrafish': 'hpf', 'Drosophila': 'dpe'}

# End of development (exclusive), in native units: human 18 y, mouse 6 weeks (sexual maturity),
# zebrafish 90 days.
ADULT_FROM = {'Human': HUMAN_BIRTH_DPC + 18 * 365, 'Mouse': MOUSE_BIRTH_DPC + 42, 'Zebrafish': 90 * 24,
              'Drosophila': 0.0}

_DEV_WORDS = re.compile(r'embryo|fetal|foetal|prenatal|carnegie|theiler|post-fertilization|lmp month|newborn|'
                        r'infant|nursing|child|juvenile|pediatric|adolescen|immature|blastula|gastrula|'
                        r'organogenesis|segmentation|pharyngula|hatching|larva', re.I)


def _human(label: str):
    s = label.lower()
    if m := re.search(r'carnegie stage (\d+)', s):
        return CARNEGIE_DPC.get(int(m.group(1)))
    if m := re.search(r'(\d+)(?:st|nd|rd|th)? week post-fertilization', s):
        return 7.0 * int(m.group(1))
    if m := re.search(r'(\w+) lmp month', s):
        n = ORDINALS.get(m.group(1))
        return 30.0 * n if n else None
    if m := re.search(r'(\d+)-(\d+) year-old', s):
        return HUMAN_BIRTH_DPC + 365 * (int(m.group(1)) + int(m.group(2))) / 2
    if m := re.search(r'(\d+)-year-old', s):
        return HUMAN_BIRTH_DPC + 365 * int(m.group(1))
    if m := re.search(r'(\d+)-month-old', s):
        return HUMAN_BIRTH_DPC + 30 * int(m.group(1))
    if m := re.search(r'(\w+) decade stage', s):
        n = ORDINALS.get(m.group(1))
        return HUMAN_BIRTH_DPC + 365 * ((n - 1) * 10 + 5) if n else None
    if 'newborn' in s:
        return HUMAN_BIRTH_DPC + 14
    if 'nursing stage' in s:
        return HUMAN_BIRTH_DPC + 165
    if 'child stage (1-4' in s:
        return HUMAN_BIRTH_DPC + 2.5 * 365
    if 'juvenile stage (5-14' in s:
        return HUMAN_BIRTH_DPC + 9.5 * 365
    return None


def _mouse(label: str):
    s = label.lower()
    if m := re.search(r'theiler stage (\d+)', s):
        return THEILER_DPC.get(int(m.group(1)))
    if m := re.search(r'\be(\d+(?:\.\d+)?)\b', s):
        return float(m.group(1))
    if m := re.search(r'(\d+)-day-old', s):
        return MOUSE_BIRTH_DPC + int(m.group(1))
    if m := re.search(r'(\d+)-week-old', s):
        return MOUSE_BIRTH_DPC + 7 * int(m.group(1))
    if m := re.search(r'(\d+)-(\d+) month-old', s):
        return MOUSE_BIRTH_DPC + 30 * (int(m.group(1)) + int(m.group(2))) / 2
    if 'and over' in s:
        return None
    if m := re.search(r'(\d+)-month-old', s):
        return MOUSE_BIRTH_DPC + 30 * int(m.group(1))
    return None


def _zebrafish(label: str):
    s = label.lower()
    sub = s.split(':', 1)[1].strip() if ':' in s else s.strip()
    return ZFS_HPF.get(sub)


def _drosophila(label: str):
    s = label.lower()
    if m := re.search(r'day (\d+) of adulthood|(\d+)[- ]day[- ]old adult', s):
        return float(m.group(1) or m.group(2))
    return None


_PARSERS = {'Human': _human, 'Mouse': _mouse, 'Zebrafish': _zebrafish, 'Drosophila': _drosophila}


def parse_stage(species: str, label: str) -> dict:
    """Numeric age, tagged string for the pipeline and a developmental flag for one stage term."""
    if not label or str(label).lower() in {'unknown', 'na', 'nan'}:
        return {'age': None, 'tag': 'unknown', 'developmental': None}
    age = _PARSERS.get(species, lambda _: None)(str(label))
    if age is not None:
        dev = age < ADULT_FROM.get(species, float('inf'))
        unit = UNIT.get(species, 'dpc')
        return {'age': float(age), 'tag': f"{age:g} {unit}", 'developmental': dev}
    dev = bool(_DEV_WORDS.search(str(label)))
    if species == 'Zebrafish' and str(label).lower().startswith('adult'):
        dev = False
    return {'age': None, 'tag': 'unknown', 'developmental': dev}


def format_age(species: str, age) -> str:
    """Readable native age for manifests."""
    if age is None:
        return 'n/a'
    if species == 'Human':
        return f"{age / 7:.0f} pcw" if age < HUMAN_BIRTH_DPC else f"{(age - HUMAN_BIRTH_DPC) / 365:.1f} y"
    if species == 'Mouse':
        return f"E{age:g}" if age < MOUSE_BIRTH_DPC else f"P{age - MOUSE_BIRTH_DPC:.0f}"
    if species == 'Zebrafish':
        return f"{age:g} hpf" if age < 72 else f"{age / 24:g} dpf"
    return f"{age:g} {UNIT.get(species, '')}"
