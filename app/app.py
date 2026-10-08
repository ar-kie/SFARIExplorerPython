"""SFARIExplorer: cell-type-resolved gene expression across species and development.

Run locally:  streamlit run app.py
"""

from pathlib import Path

import numpy as np
import streamlit as st

from sfx import __version__
from sfx import config as C
from sfx import data as D
from sfx import pages
from sfx import plots as P

DATA_DIR = Path(__file__).parent / "data"

st.set_page_config(page_title=C.APP_NAME, page_icon=":material/hub:", layout="wide",
                   initial_sidebar_state="expanded",
                   menu_items={"About": f"{C.APP_NAME} v{__version__}. {C.APP_TAGLINE}."})

st.markdown("""
<style>
  .block-container {padding-top: 2.6rem; max-width: 1500px;}
  h3 {font-weight: 600; letter-spacing: -0.01em; margin-top: 0.4rem;}
  h4 {font-weight: 600; margin-top: 1.4rem; margin-bottom: 0.2rem;}
  [data-testid="stMetricValue"] {font-size: 1.6rem; font-weight: 500;}
  [data-testid="stCaptionContainer"] {color: #52514e;}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Data access (cached)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading expression atlas…")
def get_atlas() -> D.Atlas:
    return D.load_atlas(DATA_DIR)


ATLAS = get_atlas()


@st.cache_data(max_entries=64, show_spinner=False)
def celltype_table(genes, species, sample_types, datasets, min_cells, min_detection):
    return D.celltype_long(ATLAS, list(genes), list(species), list(sample_types), list(datasets),
                           min_cells, min_detection)


@st.cache_data(max_entries=64, show_spinner=False)
def temporal_table(genes, species, sample_types, datasets, min_cells, min_detection):
    return D.temporal_long(ATLAS, list(genes), list(species), list(sample_types), list(datasets),
                           min_cells, min_detection)


@st.cache_data(max_entries=16, show_spinner="Computing background distribution…")
def conservation_background(ref, min_shared, species, datasets, min_cells, min_detection, n_genes=600):
    rng = np.random.default_rng(0)
    genes = list(rng.choice(ATLAS.genes, size=min(n_genes, len(ATLAS.genes)), replace=False))
    df = D.celltype_long(ATLAS, genes, list(species), ["in_vivo"], list(datasets), min_cells, min_detection)
    prof = D.species_profiles(df)
    present = [s for s in C.SPECIES_ORDER if s in set(prof["species"]) and s != ref]
    return D.conservation(prof, [(ref, s) for s in present], min_shared)


@st.cache_data(show_spinner=False)
def preset_sets():
    return D.gene_sets(ATLAS)


# ---------------------------------------------------------------------------
# Sidebar: genes and global filters (scope every page)
# ---------------------------------------------------------------------------

SETS = preset_sets()
DEFAULT_SET = "Example: high-confidence ASD genes"
if "gene_text" not in st.session_state:
    st.session_state.gene_text = ", ".join(SETS[DEFAULT_SET])


def _apply_preset():
    name = st.session_state.preset
    if name in SETS:
        st.session_state.gene_text = ", ".join(SETS[name])


with st.sidebar:
    st.markdown(f"**{C.APP_NAME}**  \n<span style='color:{C.INK_SECONDARY};font-size:0.9rem'>{C.APP_TAGLINE}</span>",
                unsafe_allow_html=True)

    st.markdown("##### Genes")
    st.selectbox("Load a gene set", ["Custom"] + list(SETS), index=1 + list(SETS).index(DEFAULT_SET),
                 key="preset", on_change=_apply_preset,
                 format_func=lambda k: k if k == "Custom" else f"{k} ({len(SETS[k])})")
    st.text_area("Gene symbols", key="gene_text", height=110,
                 help="Human gene symbols, separated by commas, spaces or new lines. Non-human data are shown for "
                      "the corresponding orthologs.")
    found, missing = ATLAS.resolve(D.parse_gene_text(st.session_state.gene_text))
    msg = f"{len(found)} gene{'s' if len(found) != 1 else ''} matched"
    if missing:
        msg += f"; not in atlas: {', '.join(missing[:8])}{' …' if len(missing) > 8 else ''}"
    st.caption(msg)

    st.markdown("##### Filters")
    species = st.multiselect("Species", C.SPECIES_ORDER, default=C.SPECIES_ORDER, key="f_species")
    sample_types = st.multiselect("Sample type", ["in_vivo", "organoid"], default=["in_vivo", "organoid"],
                                  format_func=C.SAMPLE_TYPE_LABEL.get, key="f_sample")
    ds_all = ATLAS.datasets
    ds_opts = ds_all[ds_all["Species"].isin(species)]["Dataset"].tolist()
    with st.expander("Datasets"):
        datasets = st.multiselect("Restrict to datasets", ds_opts, default=[], key="f_datasets",
                                  placeholder="All datasets", label_visibility="collapsed")
    min_cells = st.number_input("Minimum cells per group", min_value=1, max_value=5000, value=30, step=10,
                                key="f_min_cells",
                                help="Groups (dataset × cell type, or × time point) with fewer cells are excluded.")
    mask = st.toggle("Hide undetected groups", value=True, key="f_mask",
                     help="voom assigns finite values to zero counts. When on, groups where the gene is detected in "
                          "too few pseudobulk replicates are shown as 'not detected' instead of as a value.")
    min_det = st.slider("Detection threshold", 0.0, 0.9, 0.0, 0.05, key="f_min_det", disabled=not mask,
                        help="A group counts as detected if the fraction of its pseudobulk replicates with ≥ 1 "
                             "count is greater than this value (0 = detected in at least one replicate).")

    st.divider()
    st.caption(f"v{__version__} · {len(ATLAS.datasets)} datasets · {len(ATLAS.genes):,} genes · "
               f"normalisation: {ATLAS.summary.get('normalization', 'n/a')}")

filters = (tuple(species), tuple(sample_types), tuple(datasets), int(min_cells), float(min_det) if mask else None)

CTX = pages.Ctx(
    atlas=ATLAS, genes=found, missing=missing, species=species, sample_types=sample_types, datasets=datasets,
    min_cells=int(min_cells), min_detection=filters[-1], styles=P.dataset_styles(ATLAS.datasets),
    ct=lambda genes: celltype_table(tuple(genes), *filters),
    tt=lambda genes: temporal_table(tuple(genes), *filters),
    background=lambda ref, k: conservation_background(ref, k, filters[0], filters[2], filters[3], filters[4]),
)


# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------

def overview(): pages.overview(CTX)
def gene(): pages.gene_profile(CTX)
def gene_set(): pages.gene_set(CTX)
def development(): pages.development(CTX)
def conservation(): pages.conservation_page(CTX)
def atlas(): pages.cell_atlas(CTX)
def table(): pages.data_page(CTX)
def methods(): pages.methods(CTX)


nav = st.navigation([
    st.Page(overview, title="Overview", url_path="overview", default=True),
    st.Page(gene, title="Gene", url_path="gene"),
    st.Page(gene_set, title="Gene set", url_path="gene-set"),
    st.Page(development, title="Development", url_path="development"),
    st.Page(conservation, title="Conservation", url_path="conservation"),
    st.Page(atlas, title="Cell atlas", url_path="cell-atlas"),
    st.Page(table, title="Data", url_path="data"),
    st.Page(methods, title="Methods", url_path="methods"),
], position="top")
nav.run()
