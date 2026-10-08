"""Constants: naming, palettes, units, cell-type ontology, dataset references.

Palettes were checked with the data-viz validator (OKLab ΔE under simulated
protan/deutan vision, normal-vision floor, contrast vs. white):
  * SPECIES_COLORS passes all-pairs (used in scatter/strip plots).
  * DATASET_SLOTS passes adjacent-pairs only, so dataset identity is always
    double-encoded with a marker symbol.
"""

APP_NAME = "SFARIExplorer"
APP_TAGLINE = "Cell-type-resolved gene expression across species and development"

# ---------------------------------------------------------------------------
# Value definitions (what the numbers in the parquet files mean)
# ---------------------------------------------------------------------------

LOG2CPM_LABEL = "log2 CPM"
LOG2CPM_LONG = "log2 CPM (pseudobulk, voom; ComBat-corrected within species)"
CENTERED_LABEL = "Δ log2 CPM"
CENTERED_LONG = "Δ log2 CPM relative to the dataset × cell-type mean"
ZSCORE_LABEL = "z-score"
DETECTION_LABEL = "Detection rate"
DETECTION_LONG = "Fraction of pseudobulk replicates with ≥ 1 count"

# ---------------------------------------------------------------------------
# Colour
# ---------------------------------------------------------------------------

INK = "#1f2328"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#ecebe6"
AXIS = "#c3c2b7"
NOT_DETECTED = "#b4b2aa"
SURFACE = "#ffffff"

SPECIES_ORDER = ["Human", "Mouse", "Zebrafish", "Drosophila"]
SPECIES_COLORS = {
    "Human": "#eb6834",
    "Mouse": "#2a78d6",
    "Zebrafish": "#16a070",
    "Drosophila": "#4a3aa7",
}
SPECIES_LATIN = {
    "Human": "Homo sapiens",
    "Mouse": "Mus musculus",
    "Zebrafish": "Danio rerio",
    "Drosophila": "Drosophila melanogaster",
}

# Reference categorical order (adjacent-pair validated). Never cycled: the
# largest species (Human) has exactly eight datasets.
DATASET_SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                 "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
DATASET_SYMBOLS = ["circle", "square", "diamond", "triangle-up",
                   "triangle-down", "pentagon", "hexagon", "star"]

# Diverging (blue <-> red, neutral grey midpoint) and sequential (one hue).
DIVERGING = [
    [0.00, "#104281"], [0.15, "#256abf"], [0.30, "#5598e7"], [0.42, "#9ec5f4"],
    [0.50, "#f0efec"],
    [0.58, "#f4a49b"], [0.70, "#e5675f"], [0.85, "#c0342f"], [1.00, "#8a1c1a"],
]
SEQUENTIAL = [
    [0.00, "#f3f7fd"], [0.15, "#cde2fb"], [0.30, "#9ec5f4"], [0.45, "#6da7ec"],
    [0.60, "#3987e5"], [0.75, "#256abf"], [0.88, "#184f95"], [1.00, "#0d366b"],
]

# ---------------------------------------------------------------------------
# Cell-type ontology (harmonised supercategories) grouped by lineage
# ---------------------------------------------------------------------------

LINEAGES = {
    "Progenitor": ["Early Developmental", "Neural Progenitors & Stem Cells"],
    "Neuronal": ["Excitatory Neurons", "Inhibitory Neurons",
                 "Dopaminergic & Monoaminergic", "Neurons (Regional)", "Neurons (General)"],
    "Glial": ["Astrocytes", "Oligodendrocyte Lineage",
              "Ependymal & Choroid Plexus", "Schwann / PNS Glia", "Glia (General)"],
    "Immune": ["Microglia & Macrophages", "Immune Cells"],
    "Vascular & stromal": ["Endothelial & Vascular", "Fibroblast / Mesenchymal",
                           "Erythrocytes"],
    "Other": ["Other"],
}
CELLTYPE_ORDER = [ct for cts in LINEAGES.values() for ct in cts]
CELLTYPE_LINEAGE = {ct: lin for lin, cts in LINEAGES.items() for ct in cts}
# Hues placed between the species hues (OKLCH h ≈ 41/163/256/284) so that a
# lineage track never reads as a species: ≥ 8.8 ΔE from every species colour
# under normal, protan and deutan vision; passes the adjacent-pair checks.
LINEAGE_COLORS = {
    "Progenitor": "#626a09", "Neuronal": "#985b93", "Glial": "#3ebfc6",
    "Immune": "#8d5406", "Vascular & stromal": "#c097df", "Other": "#898781",
}
CELLTYPE_SHORT = {
    "Early Developmental": "Early dev.",
    "Neural Progenitors & Stem Cells": "Neural progenitors",
    "Excitatory Neurons": "Excitatory neurons",
    "Inhibitory Neurons": "Inhibitory neurons",
    "Dopaminergic & Monoaminergic": "Monoaminergic neurons",
    "Neurons (Regional)": "Neurons (regional)",
    "Neurons (General)": "Neurons (unassigned)",
    "Astrocytes": "Astrocytes",
    "Oligodendrocyte Lineage": "Oligodendrocyte lin.",
    "Ependymal & Choroid Plexus": "Ependymal / ChP",
    "Schwann / PNS Glia": "Schwann / PNS glia",
    "Glia (General)": "Glia (unassigned)",
    "Microglia & Macrophages": "Microglia / macrophages",
    "Immune Cells": "Other immune cells",
    "Endothelial & Vascular": "Endothelial / vascular",
    "Fibroblast / Mesenchymal": "Fibroblast / mesench.",
    "Erythrocytes": "Erythrocytes",
    "Other": "Other",
}

# Labels were transferred from vertebrate reference annotations. For these
# species/label pairs there is no directly homologous cell class, so the label
# indicates transcriptional similarity only.
LABEL_CAVEATS = {
    ("Drosophila", "Oligodendrocyte Lineage"): "Drosophila has no oligodendrocytes (ensheathing/wrapping glia are the closest analogues).",
    ("Drosophila", "Endothelial & Vascular"): "Drosophila has an open circulatory system without endothelium.",
    ("Drosophila", "Microglia & Macrophages"): "Drosophila has no microglia; phagocytic roles are carried out by glia and hemocytes.",
    ("Drosophila", "Fibroblast / Mesenchymal"): "No direct fibroblast homologue in the Drosophila brain.",
    ("Drosophila", "Erythrocytes"): "Drosophila has no erythrocytes.",
}

# ---------------------------------------------------------------------------
# Developmental time
# ---------------------------------------------------------------------------

# Anchors of the relative developmental time scale used in the data build
# (piecewise linear between anchors, per species).
REL_TIME_ANCHORS = [(0.0, "fertilization"), (0.45, "birth / hatching"), (1.0, "sexual maturity")]

NATIVE_TIME_AXIS = {
    ("Human", "in_vivo"): "Age (days post-conception)",
    ("Human", "organoid"): "Organoid age (days in culture)",
    ("Mouse", "in_vivo"): "Age (days post-conception)",
    ("Zebrafish", "in_vivo"): "Age (hours post-fertilization)",
    ("Drosophila", "in_vivo"): "Age (days post-eclosion)",
}
# Log-scale native axes where sampling spans orders of magnitude.
NATIVE_TIME_LOG = {("Human", "in_vivo"): True, ("Mouse", "in_vivo"): True,
                   ("Zebrafish", "in_vivo"): True}
BIRTH_NATIVE = {("Human", "in_vivo"): 280.0, ("Mouse", "in_vivo"): 20.0,
                ("Zebrafish", "in_vivo"): 72.0}

SAMPLE_TYPE_LABEL = {"in_vivo": "in vivo", "organoid": "organoid"}

# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

DATASET_REFS = {
    "Bhaduri (2021)": ("Bhaduri et al., Nature 598, 200–204 (2021)", "10.1038/s41586-021-03910-8",
                       "Cortical arealization, 2nd trimester"),
    "Braun (2023)": ("Braun et al., Science 382, eadf1226 (2023)", "10.1126/science.adf1226",
                     "First-trimester whole brain"),
    "He (2024)": ("He et al., Nature 635, 690–698 (2024)", "10.1038/s41586-024-08172-8",
                  "Human Neural Organoid Cell Atlas (HNOCA)"),
    "Velmeshev (2019)": ("Velmeshev et al., Science 364, 685–689 (2019)", "10.1126/science.aav8130",
                         "Postnatal cortex, ASD and controls"),
    "Velmeshev (2023)": ("Velmeshev et al., Science 382, eadf0834 (2023)", "10.1126/science.adf0834",
                         "Prenatal to adult cortex"),
    "Wang (2022)": ("Wang et al., Nat. Commun. 13, 5688 (2022)", "10.1038/s41467-022-33364-z",
                    "Single-rosette telencephalic organoids"),
    "Wang (2025)": ("Wang et al., Nature (2025)", "10.1038/s41586-024-08351-7",
                    "Neocortex, first trimester to adolescence (multiome)"),
    "Zhu (2023)": ("Zhu et al., Sci. Adv. 9, eadg3754 (2023)", "10.1126/sciadv.adg3754",
                   "Cerebral cortex, fetal to adult (multiome)"),
    "La Manno (2021)": ("La Manno et al., Nature 596, 92–96 (2021)", "10.1038/s41586-021-03775-x",
                        "Embryonic mouse brain, E7–E18"),
    "Jin (2025)": ("Jin et al., Nature 638, 182–196 (2025)", "10.1038/s41586-024-08350-8",
                   "Brain-wide ageing atlas"),
    "Sziraki (2023)": ("Sziraki et al., Nat. Genet. 55, 2104–2116 (2023)", "10.1038/s41588-023-01572-y",
                       "Whole-brain ageing (EasySci)"),
    "Raj (2020)": ("Raj et al., Neuron 108, 1058–1074 (2020)", "10.1016/j.neuron.2020.09.023",
                   "Embryo-to-larva brain"),
    "Davie (2018)": ("Davie et al., Cell 174, 982–998 (2018)", "10.1016/j.cell.2018.05.057",
                     "Ageing adult brain"),
}

# ---------------------------------------------------------------------------
# Gene sets
# ---------------------------------------------------------------------------

DEFAULT_GENES = ["SHANK3", "CHD8", "SCN2A", "SYNGAP1", "ARID1B",
                 "MECP2", "FOXP1", "GRIN2B", "DYRK1A", "PTEN"]

# Canonical, largely species-agnostic markers (human symbols). Useful as a
# positive control for the harmonised cell-type labels in each species.
MARKER_GENES = {
    "Progenitor": ["SOX2", "PAX6", "NES", "VIM", "HES1", "MKI67", "TOP2A", "EOMES"],
    "Excitatory": ["SLC17A7", "SLC17A6", "NEUROD2", "NEUROD6", "TBR1", "SATB2", "BCL11B"],
    "Inhibitory": ["GAD1", "GAD2", "SLC32A1", "DLX2", "DLX5", "LHX6", "SST", "PVALB"],
    "Monoaminergic": ["TH", "DDC", "SLC6A3", "SLC18A2"],
    "Astrocyte": ["GFAP", "AQP4", "SLC1A3", "SLC1A2", "ALDH1L1"],
    "Oligodendrocyte": ["OLIG1", "OLIG2", "PDGFRA", "SOX10", "MBP", "PLP1", "MOG"],
    "Microglia": ["CX3CR1", "P2RY12", "C1QA", "PTPRC", "AIF1"],
    "Vascular": ["CLDN5", "PECAM1", "FLT1", "PDGFRB", "COL1A1", "COL1A2", "DCN"],
    "Ependymal / ChP": ["FOXJ1", "TTR"],
    "Erythroid": ["HBB", "HBA1"],
}

PLOT_DOWNLOAD = {
    "displaylogo": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"],
    "toImageButtonOptions": {"format": "svg", "filename": "sfariexplorer_figure", "scale": 1},
}
