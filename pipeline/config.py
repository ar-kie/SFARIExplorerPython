"""SFARI Pipeline Configuration (V4) - TRUE OUTER JOIN"""
import os

DATA_DIR = "/sc/arion/projects/ad-omics/raphael/SFARI/data"
GENE_MAP_DIR = "/sc/arion/projects/ad-omics/raphael/SFARI/gene_maps"
OUTPUT_DIR = "/sc/arion/projects/ad-omics/raphael/SFARI/pipeline_output"
CHECKPOINT_DIR = f"{OUTPUT_DIR}/checkpoints"
TEMP_DIR = f"{OUTPUT_DIR}/temp"
LOG_DIR = f"{OUTPUT_DIR}/logs"

GENE_MANIFEST_PATH = f"{OUTPUT_DIR}/gene_manifest.parquet"
CONCATENATED_PATH = f"{OUTPUT_DIR}/concatenated_raw.h5ad"
ANNOTATED_PATH = f"{OUTPUT_DIR}/concatenated_annotated.h5ad"
FINAL_PATH = f"{OUTPUT_DIR}/sfari_final.h5ad"

COMBINED_MTX = {
    "mtx_path": f"{DATA_DIR}/10242025_raj_aerts_cao_humanized_velmeshev_merged.mtx",
    "features_path": f"{DATA_DIR}/10242025_raj_aerts_cao_humanized_velmeshev_merged_features.csv",
    "metadata_path": f"{DATA_DIR}/10242025_raj_aerts_cao_humanized_velmeshev_merged_metadata.csv",
}

MTX_DATASET_MAPPING = {
    # Raw names
    "Raj": "Raj-2020", "raj": "Raj-2020", "Raj-2020": "Raj-2020",
    "Aerts": "Davie-2018", "aerts": "Davie-2018", "Davie": "Davie-2018", "Davie-2018": "Davie-2018",
    "Cao": "Sziraki-2023", "cao": "Sziraki-2023", "Sziraki": "Sziraki-2023", "Sziraki-2023": "Sziraki-2023",
    "Velmeshev": "Velmeshev-2019", "velmeshev": "Velmeshev-2019", "Velmeshev-2019": "Velmeshev-2019",
    # Display names from MTX metadata (CRITICAL!)
    "Raj (2020)": "Raj-2020",
    "Davie (2018)": "Davie-2018",
    "Sziraki (2023)": "Sziraki-2023",
    "Velmeshev (2019)": "Velmeshev-2019",
}

# He-2024: use counts_lengthnorm layer + rounding
DATASET_LAYER_OVERRIDE = {"He-2024": "counts_lengthnorm"}
DATASETS_TO_ROUND = ["He-2024"]

ENSEMBL_DATASETS = {
    "Bhaduri-2021": f"{DATA_DIR}/Bhaduri_2021/Bhaduri_2021.h5ad",
    "Braun-2023": f"{DATA_DIR}/Linnarsson/data/2023/Linnarsson_2023.h5ad",
    "Velmeshev-2023": f"{DATA_DIR}/Velmeshev/data/2023/Velmeshev_2023.h5ad",
    "Zhu-2023": f"{DATA_DIR}/Zhu_2023/Zhu_2023.h5ad",
    "Wang-2025": f"{DATA_DIR}/Wang_2025/Wang_2025.h5ad",
}

SYMBOL_DATASETS = {
    "La-Manno-2021": f"{DATA_DIR}/Linnarsson/data/07092025_linnarsson_humanized.h5ad",
    "He-2024": f"{DATA_DIR}/HNOCA/hnoca_cleanedmeta.h5ad",
    "Jin-2025": f"{DATA_DIR}/Zeng/data/07092025_zeng_humanized.h5ad",
    "Wang-2022": f"{DATA_DIR}/Wang_2022/wang2022_annotated.h5ad",
}

MTX_DATASETS = {
    "Raj-2020": {"organism": "Human"},
    "Davie-2018": {"organism": "Drosophila"},
    "Sziraki-2023": {"organism": "Mouse"},
    "Velmeshev-2019": {"organism": "Human"},
}

DATASET_META = {
    "La-Manno-2021": {"dataset": "La Manno (2021)", "organism": "Mouse"},
    "He-2024": {"dataset": "He (2024)", "organism": "Human"},
    "Jin-2025": {"dataset": "Jin (2025)", "organism": "Mouse"},
    "Bhaduri-2021": {"dataset": "Bhaduri (2021)", "organism": "Human"},
    "Braun-2023": {"dataset": "Braun (2023)", "organism": "Human"},
    "Velmeshev-2023": {"dataset": "Velmeshev (2023)", "organism": "Human"},
    "Zhu-2023": {"dataset": "Zhu (2023)", "organism": "Human"},
    "Wang-2025": {"dataset": "Wang (2025)", "organism": "Human"},
    "Wang-2022": {"dataset": "Wang (2022)", "organism": "Human"},
    "Raj-2020": {"dataset": "Raj (2020)", "organism": "Human"},
    "Davie-2018": {"dataset": "Davie (2018)", "organism": "Drosophila"},
    "Velmeshev-2019": {"dataset": "Velmeshev (2019)", "organism": "Human"},
    "Sziraki-2023": {"dataset": "Sziraki (2023)", "organism": "Mouse"},
}

EXTERNAL_METADATA = {
    "Velmeshev-2019": {"path": f"{DATA_DIR}/Velmeshev/data/meta.tsv", "sep": "\t", "barcode_col": "cell", "celltype_col": "cluster", "prefix": "Velmeshev-2019"},
    "Sziraki-2023": {"path": f"{DATA_DIR}/Cao/data/GSM6538356_RNA_cell_annotation.csv", "sep": ",", "barcode_col": "sample", "celltype_col": "Main_cluster_name", "prefix": "Sziraki-2023"},
}

CELLTYPE_COLUMNS = ["cell_type_velmeshev_2019", "cell_type_sziraki_2023", "Subclass", "cell_type", "CellClass", "cluster_names", "subclass_label"]

# =============================================================================
# GENE FILTERING - MINIMAL for TRUE OUTER JOIN
# =============================================================================

# Only remove spike-ins
GENE_EXCLUDE_PREFIXES = ["ERCC"]

# Only remove unmapped Ensembl/FlyBase IDs (these weren't converted to symbols)
GENE_EXCLUDE_PATTERNS = [
    r"^ENS[A-Z]*G\d+",    # Any Ensembl gene ID (human, mouse, etc.)
    r"^ENSG\d+",          # Human Ensembl
    r"^ENSMUSG\d+",       # Mouse Ensembl
    r"^FBgn\d+",          # FlyBase IDs (Drosophila)
]

# TRUE OUTER JOIN: keep genes even if only in 1 dataset
MIN_DATASETS_PER_GENE = 1

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def ensure_dirs():
    for d in [OUTPUT_DIR, CHECKPOINT_DIR, TEMP_DIR, LOG_DIR]: os.makedirs(d, exist_ok=True)
def get_checkpoint_path(s): return f"{CHECKPOINT_DIR}/{s}.done"
def checkpoint_exists(s): return os.path.exists(get_checkpoint_path(s))
def mark_checkpoint(s):
    ensure_dirs()
    with open(get_checkpoint_path(s), 'w') as f:
        import datetime; f.write(f"Completed: {datetime.datetime.now()}\n")
