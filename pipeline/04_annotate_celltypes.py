#!/usr/bin/env python3
"""
Step 4: Annotate Cell Types (v4)
- Fixed barcode extraction: Velmeshev-(2019)_BARCODE-Velmeshev-2019 → BARCODE
- Final pattern tweaks
"""
import os, re, numpy as np, pandas as pd, scanpy as sc
import warnings; warnings.filterwarnings('ignore')
from config import *

STEP_NAME = "04_annotate_celltypes"

# =============================================================================
# CELL TYPE SUPERCATEGORY PATTERNS
# =============================================================================

PATTERNS = [
    # Excitatory neurons
    (re.compile(r"excitatory|glutamatergic|glut[0-9]?|pyramidal|projection|^ex[_\-]|^ex[0-9]|cortic.*neuron|ctx.*ex|^l[2-6].*it|^l[2-6].*et|^l[2-6].*ct|^l[2-6].*np|intratelencephalic|extratelencephalic|corticofugal|corticothalamic|subcerebral|cajal.retzius|^cr\b|reelin|granule.neuron", re.I), "Excitatory Neurons"),
    
    # Inhibitory neurons
    (re.compile(r"inhibitory|gabaergic|gaba[_\-]?[0-9]?|interneuron|inter[_\-\s]neuron|\bin[_\-\s]?[0-9]|\bpv\b|\bsst\b|\bvip\b|lamp5|pvalb|somatostatin|parvalbumin|chandelier|basket|martinotti|neurogliaform|caudal.ganglionic|medial.ganglionic|cge|mge|lge|sncg|id2|htr3a|glycinergic|amacrine", re.I), "Inhibitory Neurons"),
    
    # Dopaminergic & monoaminergic
    (re.compile(r"dopamin|seroton|noradren|adrenergic|catecholamin|tyrosine.hydroxylase|\bth\+|\bda\b.*neuron|5-?ht|monoamin|\bdopa\b|snc.*vta|foxa1.*dopa", re.I), "Dopaminergic & Monoaminergic"),
    
    # Generic neurons
    (re.compile(r"\bneuron|\bnrn\b|neuronal|granule.cell|purkinje|motor.neuron|sensory.neuron|spiny.neuron|medium.spiny|cholinergic|^msn|striatal|^d1[_\-\s]|^d2[_\-\s]|spn\b|dspn|ispn|neural.cell|^en[0-9]|photoreceptor|mueller.cell|retinal.ganglion", re.I), "Neurons (General)"),
    
    # Neural progenitors & stem cells
    (re.compile(r"progenitor|radial.glia|\brg\b|rgc|neural.stem|neuroblast|\bnpc\b|\bnsc\b|intermediate.progenitor|\bipc\b|\bip\b|outer.radial|inner.radial|org\b|irg\b|ventricular.zone|\bvz\b|subventricular|\bsvz\b|dividing|proliferat|cycling|mitotic|^div|glioblast|neuroepithel|^np[0-9]|neuroplacodal|placode|optic.cup", re.I), "Neural Progenitors & Stem Cells"),
    
    # Astrocytes
    (re.compile(r"astrocyte|astro[_\-\s]?[0-9]?|\basc\b|bergmann|fibrous.astro|protoplasmic|gfap\+|^astr|astro.*(te|nt).*nn", re.I), "Astrocytes"),
    
    # Oligodendrocyte lineage
    (re.compile(r"oligodendro|\bolig[_\-]?[0-9]?|\bopc\b|\bol[_\-]?[0-9]|myelin|mature.oligo|committed.oligo|newly.formed|nfol|mfol|\bmol\b|cop\b|polydendrocyte|^oligo|opc.*nn|oligo.*nn|preopc", re.I), "Oligodendrocyte Lineage"),
    
    # Microglia & macrophages
    (re.compile(r"microglia|\bmg[_\-]?[0-9]?|macrophage|immune.*brain|brain.*immune|cx3cr1|iba1|aif1|myeloid|monocyte.*cns|^micro|^mac\b|microglia.*nn|bam.*nn|border.associated", re.I), "Microglia & Macrophages"),
    
    # Endothelial & vascular
    (re.compile(r"endotheli|\bec[_\-]?[0-9]?|\bend\b|vascular|pericyte|\bpc[_\-]?[0-9]?|blood.vessel|capillary|arterial|venous|angiogen|smooth.muscle.*vasc|vsmc|mural|vlmc|abc\b|smc\b|endo.*nn|peri.*nn|angioblast", re.I), "Endothelial & Vascular"),
    
    # Ependymal & choroid plexus
    (re.compile(r"ependym|choroid|plexus|ciliated|ventricle.*epithe|csf|^epend|^epen|^chor|tanycyte|roof.plate", re.I), "Ependymal & Choroid Plexus"),
    
    # Fibroblast / mesenchymal
    (re.compile(r"fibroblast|\bfb[_\-]?[0-9]?|mesenchym|meninges|meningeal|leptomening|dura|arachnoid|stromal|connective|perivascular.fibro|pvfb|extracellular.matrix|^pia\b", re.I), "Fibroblast / Mesenchymal"),
    
    # Immune cells
    (re.compile(r"\bt[_\-\s]?cell|\btc\b|cd4|cd8|lymphocyte|t.lymph|nk.cell|natural.killer|\bb[_\-\s]?cell|plasma.cell|b.lymph|immune|leukocyte", re.I), "Immune Cells"),
    
    # Red blood cells / blood
    (re.compile(r"erythro|red.blood|\brbc\b|hemoglobin|erythroid|\bblood\b", re.I), "Erythrocytes"),
    
    # Schwann / PNS glia
    (re.compile(r"schwann|peripheral.*glia|satellite.glia|enteric.glia", re.I), "Schwann / PNS Glia"),
    
    # Early developmental
    (re.compile(r"pluripotent|blastocyst|morula|epiblast|primitive|mesoderm|endoderm|ectoderm|neural.crest|^nc[_\-]|cranial.neural.crest|gastrulat|paraxial|\bgut\b|cortical.hem", re.I), "Early Developmental"),
    
    # Regional neurons
    (re.compile(r"^forebrain$|^hindbrain$|^midbrain$|^hypothalamus$|dorsal.forebrain|dorsal.hindbrain|dorsal.midbrain|ventral.forebrain|mixed.region|ventral.midbrain|ventral.hindbrain|midbrain.hindbrain|basal.plate|diencephalon|^caudal$|^anterior$|hypothalamus.cell", re.I), "Neurons (Regional)"),
    
    # Generic glia (catch-all)
    (re.compile(r"glial.cell|\bglia\b", re.I), "Glia (General)"),
]


def clean_cell_type(x):
    """Clean cell type string."""
    if x is None or (isinstance(x, float) and np.isnan(x)): 
        return ""
    
    s = str(x).strip()
    
    # Handle b'...' byte strings
    if s.startswith("b'") and s.endswith("'"):
        s = s[2:-1]
    elif s.startswith('b"') and s.endswith('"'):
        s = s[2:-1]
    
    # Handle null values
    if s.lower() in {"nan", "none", "", "unknown", "unannotated", "unassigned", 
                     "undefined", "unk", "unknown_unknown", "bad cells", "bad_cells"}: 
        return ""
    
    return s


def assign_supercategory(raw):
    """Assign supercategory based on cell type string."""
    s = clean_cell_type(raw)
    if not s:
        return "Unknown"
    
    for pattern, category in PATTERNS:
        if pattern.search(s):
            return category
    
    return "Other"


def extract_barcode(idx_str, dataset_name):
    """
    Extract barcode from adata index.
    
    Examples:
    - 'Velmeshev-(2019)_AAACCTGGTACGCACC-1_1823_BA24-Velmeshev-2019' → 'AAACCTGGTACGCACC-1_1823_BA24'
    - 'Sziraki-(2023)_EasySci_001.AACCGATTGCAATCGAACTC-Sziraki-2023' → 'EasySci_001.AACCGATTGCAATCGAACTC'
    """
    # Remove suffix first: -Velmeshev-2019 or -Sziraki-2023
    suffix = f"-{dataset_name}"
    if idx_str.endswith(suffix):
        idx_str = idx_str[:-len(suffix)]
    
    # Remove prefix: Velmeshev-(2019)_ or Sziraki-(2023)_
    # Pattern: DatasetName-(Year)_
    prefix_pattern = re.compile(r'^[A-Za-z]+\-?\(?[0-9]{4}\)?[_\-]')
    match = prefix_pattern.match(idx_str)
    if match:
        barcode = idx_str[match.end():]
    else:
        # Fallback: try splitting on first underscore after dataset-like prefix
        parts = idx_str.split('_', 1)
        if len(parts) > 1:
            barcode = parts[1]
        else:
            barcode = idx_str
    
    return barcode


def load_external_metadata(adata):
    """Load external cell type annotations."""
    
    for dataset_name, meta_info in EXTERNAL_METADATA.items():
        if not os.path.exists(meta_info["path"]): 
            print(f"  WARNING: Not found: {meta_info['path']}")
            continue
        
        print(f"\n  Loading external metadata for {dataset_name}...")
        
        ext_df = pd.read_csv(meta_info["path"], sep=meta_info["sep"])
        print(f"    External rows: {len(ext_df):,}")
        
        # Create lookup
        lookup = dict(zip(
            ext_df[meta_info["barcode_col"]].astype(str),
            ext_df[meta_info["celltype_col"]]
        ))
        print(f"    Sample lookup keys: {list(lookup.keys())[:2]}")
        
        # Find cells
        display_name = DATASET_META.get(dataset_name, {}).get("dataset", dataset_name)
        mask = adata.obs['dataset'] == display_name
        n_cells = mask.sum()
        
        if n_cells == 0:
            print(f"    No cells for '{display_name}'")
            continue
        
        print(f"    Cells in adata: {n_cells:,}")
        
        # Debug
        sample_idx = list(adata.obs[mask].index[:2])
        print(f"    Sample adata idx: {sample_idx}")
        sample_barcodes = [extract_barcode(str(idx), dataset_name) for idx in sample_idx]
        print(f"    Extracted barcodes: {sample_barcodes}")
        
        col_name = f"cell_type_{dataset_name.lower().replace('-','_')}"
        adata.obs[col_name] = None
        
        matched = 0
        for idx in adata.obs[mask].index:
            barcode = extract_barcode(str(idx), dataset_name)
            cell_type = lookup.get(barcode)
            
            if cell_type is not None:
                adata.obs.loc[idx, col_name] = cell_type
                matched += 1
        
        print(f"    Matched: {matched:,} / {n_cells:,} ({100*matched/n_cells:.1f}%)")
    
    return adata


def merge_cell_types(adata):
    """Merge cell type columns with priority ordering."""
    
    priority_cols = [
        "cell_type_velmeshev_2019",
        "cell_type_sziraki_2023",
        "Subclass",
        "cell_type",
        "CellClass", 
        "cluster_names",
        "subclass_label",
        "cell_type_original",
        "cell_type_ontology_term_id",
        "Class",
        "author_cell_type",
        "Cluster",
    ]
    
    available_cols = [c for c in priority_cols if c in adata.obs.columns]
    
    for c in adata.obs.columns:
        if c not in available_cols:
            c_lower = c.lower()
            if 'cell' in c_lower and 'type' in c_lower:
                available_cols.append(c)
    
    print(f"\n  Cell type columns:")
    for c in available_cols:
        n_valid = (adata.obs[c].notna() & (adata.obs[c].astype(str) != '')).sum()
        print(f"    {c}: {n_valid:,}")
    
    merged = pd.Series(index=adata.obs.index, dtype='object')
    
    for col in available_cols:
        cleaned = adata.obs[col].apply(clean_cell_type)
        has_value = cleaned != ''
        needs_fill = merged.isna() | (merged == '')
        to_fill = has_value & needs_fill
        
        if to_fill.sum() > 0:
            merged.loc[to_fill] = cleaned.loc[to_fill]
            print(f"    Filled {to_fill.sum():,} from '{col}'")
    
    adata.obs['cell_type_merged'] = merged
    return adata


def main():
    ensure_dirs()
    
    if checkpoint_exists(STEP_NAME): 
        print(f"Delete checkpoint: rm {CHECKPOINT_DIR}/{STEP_NAME}.done")
        return
    
    print("="*60)
    print("STEP 4: ANNOTATE CELL TYPES (v4)")
    print("="*60)
    
    print(f"\nLoading {CONCATENATED_PATH}...")
    adata = sc.read_h5ad(CONCATENATED_PATH)
    print(f"Shape: {adata.shape}")
    
    print(f"\nDatasets:")
    for ds, cnt in adata.obs['dataset'].value_counts().items():
        print(f"  {ds}: {cnt:,}")
    
    # External metadata
    print("\n" + "-"*60)
    print("EXTERNAL METADATA")
    print("-"*60)
    adata = load_external_metadata(adata)
    
    # Merge
    print("\n" + "-"*60)
    print("MERGING CELL TYPES")
    print("-"*60)
    adata = merge_cell_types(adata)
    
    n_merged = (adata.obs['cell_type_merged'] != '').sum()
    print(f"\n  Cells with cell type: {n_merged:,} / {adata.n_obs:,}")
    
    print("\n  Top 30 merged cell types:")
    for ct, cnt in adata.obs['cell_type_merged'].value_counts().head(30).items():
        if ct:
            print(f"    '{ct}': {cnt:,}")
    
    # Supercategories
    print("\n" + "-"*60)
    print("SUPERCATEGORIES")
    print("-"*60)
    
    adata.obs['cell_type_supercategory'] = adata.obs['cell_type_merged'].apply(assign_supercategory)
    
    print(f"\nDistribution:")
    for cat, cnt in adata.obs['cell_type_supercategory'].value_counts().items():
        pct = 100 * cnt / adata.n_obs
        print(f"  {cat}: {cnt:,} ({pct:.1f}%)")
    
    other_mask = adata.obs['cell_type_supercategory'] == 'Other'
    if other_mask.sum() > 0:
        print(f"\n  'Other' (top 20):")
        for ct, cnt in adata.obs.loc[other_mask, 'cell_type_merged'].value_counts().head(20).items():
            print(f"    '{ct}': {cnt:,}")
    
    # Save
    print("\n  Cleaning obs columns...")
    for c in adata.obs.columns:
        if adata.obs[c].dtype == 'object': 
            adata.obs[c] = adata.obs[c].fillna('').astype(str).astype('category')
    
    print(f"\n  Saving to {ANNOTATED_PATH}...")
    adata.write(ANNOTATED_PATH)
    
    mark_checkpoint(STEP_NAME)
    print(f"\n✓ Done: {ANNOTATED_PATH}")


if __name__ == "__main__":
    main()
