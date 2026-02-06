"""
SFARI Gene Expression Explorer
A sophisticated cross-species gene expression browser for single-cell RNA-seq data.

Features:
1. Data overview with batch correction & variance partition visualization
2. UMAP visualization of integrated data
3. Interactive heatmaps with clustering and faceting options
4. Dot plots for expression visualization
5. Temporal dynamics with developmental time trajectories
6. Cross-species comparison with ortholog expression
7. SFARI risk gene annotations
8. Dark mode support
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
from typing import Optional, List, Dict, Tuple
import re

# =============================================================================
# Configuration
# =============================================================================

st.set_page_config(
    page_title="SFARI Gene Explorer",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Check for dark mode preference
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False

def get_theme_colors():
    """Return colors based on current theme."""
    if st.session_state.dark_mode:
        return {
            'bg': '#0e1117',
            'text': '#fafafa',
            'card_bg': '#262730',
            'accent': '#4da6ff',
            'plot_bg': '#0e1117',
            'grid': '#333333'
        }
    else:
        return {
            'bg': '#ffffff',
            'text': '#1f4e79',
            'card_bg': '#f0f4f8',
            'accent': '#2c5282',
            'plot_bg': '#ffffff',
            'grid': '#e0e0e0'
        }

theme = get_theme_colors()

# Custom CSS with dark mode support
st.markdown(f"""
<style>
    .stApp {{
        max-width: 100%;
    }}
    .main .block-container {{
        padding-top: 2rem;
        padding-bottom: 2rem;
    }}
    h1 {{
        color: {theme['text']};
        font-weight: 600;
    }}
    h2, h3 {{
        color: {theme['accent']};
    }}
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
    }}
    .stTabs [data-baseweb="tab"] {{
        background-color: {theme['card_bg']};
        border-radius: 4px 4px 0 0;
        padding: 8px 16px;
    }}
    .stTabs [aria-selected="true"] {{
        background-color: {theme['accent']};
        color: white;
    }}
    div[data-testid="stMetricValue"] {{
        font-size: 1.8rem;
        font-weight: 600;
    }}
    .info-box {{
        background-color: {'#1e3a5f' if st.session_state.dark_mode else '#e8f4f8'};
        border-left: 4px solid {theme['accent']};
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 8px 8px 0;
    }}
    .success-box {{
        background-color: {'#1e3f1e' if st.session_state.dark_mode else '#e8f8e8'};
        border-left: 4px solid #28a745;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 8px 8px 0;
    }}
    .warning-box {{
        background-color: {'#3f3f1e' if st.session_state.dark_mode else '#fff8e8'};
        border-left: 4px solid #ffc107;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 8px 8px 0;
    }}
</style>
""", unsafe_allow_html=True)

# =============================================================================
# Color Palettes
# =============================================================================

SPECIES_COLORS = {
    'Human': '#e41a1c',
    'Mouse': '#377eb8', 
    'Zebrafish': '#4daf4a',
    'Drosophila': '#984ea3'
}

CELLTYPE_COLORS = {
    'Excitatory Neurons': '#e41a1c',
    'Inhibitory Neurons': '#377eb8',
    'Neural Progenitors & Stem Cells': '#4daf4a',
    'Astrocytes': '#984ea3',
    'Oligodendrocyte Lineage': '#ff7f00',
    'Microglia & Macrophages': '#ffff33',
    'Endothelial & Vascular Cells': '#a65628',
    'Endothelial & Vascular': '#a65628',
    'Other Glia & Support': '#f781bf',
    'Neurons (unspecified)': '#999999',
    'Fibroblast / Mesenchymal': '#66c2a5',
    'Early Embryonic / Germ Layers': '#fc8d62'
}

DATASET_COLORS = {
    'He (2024)': '#e41a1c', 'Bhaduri (2021)': '#377eb8', 'Braun (2023)': '#4daf4a',
    'Velmeshev (2023)': '#984ea3', 'Velmeshev (2019)': '#ff7f00', 'Zhu (2023)': '#ffff33',
    'Wang (2025)': '#a65628', 'Wang (2022)': '#f781bf', 'La Manno (2021)': '#66c2a5',
    'Jin (2025)': '#fc8d62', 'Sziraki (2023)': '#8da0cb', 'Raj (2020)': '#e78ac3',
    'Davie (2018)': '#a6d854',
}

TIME_BIN_COLORS = {
    # Human in-vivo (warm orange-red gradient)
    'Early fetal (GW<10)': '#fef0d9', 'Mid fetal (GW10-20)': '#fdcc8a',
    'Late fetal (GW20-40)': '#fc8d59', 'Infant (0-2y)': '#e34a33',
    'Child (2-12y)': '#b30000', 'Adolescent (12-18y)': '#7f0000', 'Adult (18+y)': '#4d0000',
    # Human organoid (green gradient)
    '0-30 days': '#edf8e9', '31-60 days': '#bae4b3', '61-90 days': '#74c476',
    '91-120 days': '#31a354', '>120 days': '#006d2c',
    # Mouse (blue gradient)
    'Early embryo (E<12)': '#eff3ff', 'Mid embryo (E12-16)': '#bdd7e7',
    'Late embryo (E16-20)': '#6baed6', 'Neonatal (P0-P30)': '#3182bd',
    'Juvenile (1-3mo)': '#08519c', 'Adult (3-12mo)': '#08306b', 'Aged (>12mo)': '#041f4a',
    # Zebrafish (teal gradient)
    '0-24 hpf': '#f7fcf5', '24-48 hpf': '#c7e9c0', '48-72 hpf': '#74c476',
    '72-120 hpf (5dpf)': '#238b45', '>5 dpf': '#00441b',
    # Drosophila (purple gradient)
    '0-1 day': '#f2f0f7', '1-7 days': '#cbc9e2', '7-30 days': '#9e9ac8', '>30 days': '#6a51a3',
}

# Gene symbols for multi-gene plots
GENE_SYMBOLS = ['circle', 'square', 'diamond', 'cross', 'x', 'triangle-up', 'triangle-down', 
                'pentagon', 'hexagon', 'star', 'hourglass', 'bowtie']

def get_color_palette(values: List[str], palette_type: str = 'auto') -> Dict[str, str]:
    """Generate a color palette for categorical values."""
    if palette_type == 'species':
        return {v: SPECIES_COLORS.get(v, '#999999') for v in values}
    elif palette_type == 'cell_type':
        return {v: CELLTYPE_COLORS.get(v, '#999999') for v in values}
    elif palette_type == 'dataset':
        return {v: DATASET_COLORS.get(v, '#999999') for v in values}
    elif palette_type == 'time_bin':
        return {v: TIME_BIN_COLORS.get(v, '#999999') for v in values}
    else:
        n = len(values)
        if n <= 12:
            preset_colors = [
                '#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33',
                '#a65628', '#f781bf', '#999999', '#66c2a5', '#fc8d62', '#8da0cb'
            ]
            return {v: preset_colors[i % len(preset_colors)] for i, v in enumerate(values)}
        else:
            import colorsys
            colors = {}
            for i, v in enumerate(values):
                hue = i / n
                rgb = colorsys.hsv_to_rgb(hue, 0.7, 0.9)
                colors[v] = f'rgb({int(rgb[0]*255)},{int(rgb[1]*255)},{int(rgb[2]*255)})'
            return colors

def get_plotly_template():
    """Return plotly template based on dark mode setting."""
    return 'plotly_dark' if st.session_state.dark_mode else 'plotly_white'

# =============================================================================
# Data Loading
# =============================================================================

@st.cache_data(ttl=3600)
def load_data(data_dir: str = "data") -> Dict[str, pd.DataFrame]:
    """Load all parquet data files with caching."""
    data = {}
    
    try:
        # Core expression data (required)
        data['expression'] = pd.read_parquet(f"{data_dir}/expression_summaries.parquet")
        data['cellmeta'] = pd.read_parquet(f"{data_dir}/celltype_meta.parquet")
        data['gene_map'] = pd.read_parquet(f"{data_dir}/gene_map.parquet")
        data['risk_genes'] = pd.read_parquet(f"{data_dir}/risk_genes.parquet")
        
        # Ensure consistent column names
        if 'gene-symbol' in data['risk_genes'].columns:
            data['risk_genes'] = data['risk_genes'].rename(columns={'gene-symbol': 'gene_symbol', 'gene-score': 'gene_score'})
    except Exception as e:
        st.error(f"Error loading core data: {e}")
        return None
    
    # Optional data files
    optional_files = {
        'umap': 'umap_subsample.parquet',
        'vp_summary': 'variance_partition_summary.parquet',
        'vp_by_gene': 'variance_partition_by_gene.parquet',
        'vp_by_gene_wide': 'variance_partition_by_gene_wide.parquet',
        'dataset_info': 'dataset_info.parquet',
        'batch_correction': 'batch_correction_info.parquet',
        'summary_stats': 'summary_statistics.parquet',
        'temporal': 'temporal_expression.parquet',
        'ortholog': 'ortholog_expression.parquet',
        'species_comparison': 'species_comparison.parquet',
    }
    
    for key, filename in optional_files.items():
        try:
            data[key] = pd.read_parquet(f"{data_dir}/{filename}")
        except:
            data[key] = None
    
    return data

def get_unique_values(df: pd.DataFrame, column: str) -> List[str]:
    """Get sorted unique values from a column."""
    if df is None or df.empty or column not in df.columns:
        return []
    return sorted(df[column].dropna().unique().tolist())
# =============================================================================
# Data Processing Functions
# =============================================================================

def filter_expression_data(
    expr_df: pd.DataFrame,
    species: Optional[List[str]] = None,
    datasets: Optional[List[str]] = None,
    cell_types: Optional[List[str]] = None,
    genes: Optional[List[str]] = None
) -> pd.DataFrame:
    """Filter expression data based on user selections."""
    if expr_df is None or expr_df.empty:
        return pd.DataFrame()
    
    df = expr_df.copy()
    
    if species and len(species) > 0:
        df = df[df['species'].isin(species)]
    if datasets and len(datasets) > 0:
        df = df[df['tissue'].isin(datasets)]
    if cell_types and len(cell_types) > 0:
        df = df[df['cell_type'].isin(cell_types)]
    if genes and len(genes) > 0:
        genes_upper = [g.upper() for g in genes]
        # Handle NaN values safely
        native_mask = df['gene_native'].notna() & df['gene_native'].str.upper().isin(genes_upper)
        human_mask = df['gene_human'].notna() & df['gene_human'].str.upper().isin(genes_upper)
        df = df[native_mask | human_mask]
    
    return df

def parse_gene_input(gene_text: str) -> List[str]:
    """Parse comma/space/newline separated gene names."""
    if not gene_text or not gene_text.strip():
        return []
    genes = re.split(r'[,\s;]+', gene_text.strip())
    return [g.strip() for g in genes if g.strip()]

def create_heatmap_matrix(
    df: pd.DataFrame,
    value_col: str = 'mean_expr',
    scale_rows: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create a matrix suitable for heatmap visualization."""
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()
    
    df = df.copy()
    df['gene_display'] = df['gene_human'].fillna(df['gene_native'])
    df['col_key'] = df.apply(lambda x: f"{x['species']}|{x['tissue']}|{x['cell_type']}", axis=1)
    
    pivot = df.pivot_table(
        index='gene_display', columns='col_key', values=value_col, aggfunc='mean'
    )
    
    if scale_rows and pivot.shape[0] > 0:
        row_means = pivot.mean(axis=1)
        row_stds = pivot.std(axis=1).replace(0, 1)
        pivot = pivot.sub(row_means, axis=0).div(row_stds, axis=0).clip(-3, 3)
    
    col_meta_records = []
    for col_key in pivot.columns:
        parts = col_key.split('|')
        col_meta_records.append({
            'col_key': col_key,
            'species': parts[0] if len(parts) > 0 else '',
            'dataset': parts[1] if len(parts) > 1 else '',
            'cell_type': parts[2] if len(parts) > 2 else ''
        })
    col_meta = pd.DataFrame(col_meta_records).set_index('col_key').reindex(pivot.columns)
    
    return pivot, col_meta

# =============================================================================
# Visualization Functions - Heatmap
# =============================================================================

def create_complexheatmap(
    matrix: pd.DataFrame,
    col_meta: pd.DataFrame,
    title: str = "Gene Expression Heatmap",
    color_scale: str = "RdBu_r",
    split_by: Optional[str] = None,
    annotation_col: Optional[str] = None,
    cluster_rows: bool = True,
    cluster_cols: bool = True,
    row_label_size: int = 9,
    col_label_size: int = 9,
    gap_between_splits: float = 0.02,
    legend_position: str = "bottom"
) -> go.Figure:
    """Create a ComplexHeatmap-like visualization using Plotly subplots."""
    
    template = get_plotly_template()
    
    if matrix.empty:
        fig = go.Figure()
        fig.add_annotation(text="No data to display", xref="paper", yref="paper",
                          x=0.5, y=0.5, showarrow=False, font=dict(size=20))
        fig.update_layout(template=template)
        return fig
    
    # Cluster rows if requested
    if cluster_rows and matrix.shape[0] > 1:
        try:
            mat_filled = matrix.fillna(0).values
            dist = pdist(mat_filled)
            link = linkage(dist, method='average')
            row_order = leaves_list(link)
            matrix = matrix.iloc[row_order]
        except:
            pass
    
    # Determine splits
    if split_by and split_by in col_meta.columns:
        split_values = [s for s in col_meta[split_by].unique() if pd.notna(s)]
    else:
        split_values = [None]
    
    n_splits = len(split_values)
    split_matrices, split_col_metas, split_widths = [], [], []
    
    for split_val in split_values:
        if split_val is not None:
            cols = col_meta[col_meta[split_by] == split_val].index.tolist()
        else:
            cols = col_meta.index.tolist()
        
        sub_matrix = matrix[[c for c in cols if c in matrix.columns]]
        if sub_matrix.empty:
            continue
        sub_col_meta = col_meta.loc[sub_matrix.columns]
        
        if cluster_cols and sub_matrix.shape[1] > 1:
            try:
                mat_filled = sub_matrix.fillna(0).values.T
                dist = pdist(mat_filled)
                link = linkage(dist, method='average')
                col_order = leaves_list(link)
                sub_matrix = sub_matrix.iloc[:, col_order]
                sub_col_meta = sub_col_meta.iloc[col_order]
            except:
                pass
        
        split_matrices.append(sub_matrix)
        split_col_metas.append(sub_col_meta)
        split_widths.append(max(sub_matrix.shape[1], 1))
    
    if not split_matrices:
        fig = go.Figure()
        fig.add_annotation(text="No data after filtering", xref="paper", yref="paper",
                          x=0.5, y=0.5, showarrow=False)
        fig.update_layout(template=template)
        return fig
    
    total_width = sum(split_widths)
    col_widths = [w / total_width for w in split_widths]
    n_splits = len(split_matrices)
    
    has_annotation = annotation_col and annotation_col in col_meta.columns
    n_rows = 2 if has_annotation else 1
    row_heights = [0.03, 0.97] if has_annotation else [1.0]
    
    fig = make_subplots(
        rows=n_rows, cols=n_splits, column_widths=col_widths, row_heights=row_heights,
        horizontal_spacing=gap_between_splits, vertical_spacing=0.01,
        subplot_titles=[str(s) if s else "" for s in split_values[:n_splits]] if n_splits > 1 else None
    )
    
    if has_annotation:
        all_annotation_values = col_meta[annotation_col].unique().tolist()
        if annotation_col == 'species':
            anno_colors = get_color_palette(all_annotation_values, 'species')
        elif annotation_col == 'cell_type':
            anno_colors = get_color_palette(all_annotation_values, 'cell_type')
        elif annotation_col == 'dataset':
            anno_colors = get_color_palette(all_annotation_values, 'dataset')
        else:
            anno_colors = get_color_palette(all_annotation_values, 'auto')
    
    colorbar_added = False
    
    for split_idx, (sub_matrix, sub_col_meta) in enumerate(zip(split_matrices, split_col_metas)):
        col_idx = split_idx + 1
        if sub_matrix.empty:
            continue
        
        # Create column labels
        if split_by == 'species':
            col_labels = [f"{d}\n{c}" for d, c in zip(sub_col_meta['dataset'], sub_col_meta['cell_type'])]
        elif split_by == 'dataset':
            col_labels = [f"{s}\n{c}" for s, c in zip(sub_col_meta['species'], sub_col_meta['cell_type'])]
        elif split_by == 'cell_type':
            col_labels = [f"{s}\n{d}" for s, d in zip(sub_col_meta['species'], sub_col_meta['dataset'])]
        else:
            col_labels = [f"{d}\n{c}" for d, c in zip(sub_col_meta['dataset'], sub_col_meta['cell_type'])]
        
        hover_data = [{'species': sub_col_meta['species'].iloc[i], 'dataset': sub_col_meta['dataset'].iloc[i],
                       'cell_type': sub_col_meta['cell_type'].iloc[i]} for i in range(len(sub_col_meta))]
        
        if has_annotation:
            anno_values = sub_col_meta[annotation_col].tolist()
            anno_colors_list = [anno_colors.get(v, '#999999') for v in anno_values]
            anno_z = [[i for i in range(len(anno_values))]]
            anno_hover = [[f"{annotation_col.replace('_', ' ').title()}: {v}" for v in anno_values]]
            
            fig.add_trace(
                go.Heatmap(z=anno_z, x=col_labels,
                          colorscale=[[i/max(len(anno_values)-1, 1), anno_colors_list[i]] for i in range(len(anno_values))],
                          showscale=False, hoverinfo='text', text=anno_hover),
                row=1, col=col_idx
            )
        
        heatmap_row = 2 if has_annotation else 1
        
        hover_text = []
        for gene in sub_matrix.index:
            row_hover = []
            for i, col in enumerate(sub_matrix.columns):
                val = sub_matrix.loc[gene, col]
                h = hover_data[i]
                text = f"Gene: {gene}<br>Species: {h['species']}<br>Dataset: {h['dataset']}<br>Cell type: {h['cell_type']}<br>Z-score: {val:.2f}" if pd.notna(val) else "Z-score: N/A"
                row_hover.append(text)
            hover_text.append(row_hover)
        
        fig.add_trace(
            go.Heatmap(z=sub_matrix.values, x=col_labels, y=sub_matrix.index.tolist(),
                      colorscale=color_scale, zmid=0, zmin=-3, zmax=3,
                      showscale=not colorbar_added,
                      colorbar=dict(title=dict(text="Z-score", side="right"), thickness=15, len=0.7, x=1.02) if not colorbar_added else None,
                      hoverinfo='text', text=hover_text),
            row=heatmap_row, col=col_idx
        )
        colorbar_added = True
    
    height = max(400, 80 + len(matrix) * 16)
    if has_annotation:
        height += 30
    
    fig.update_layout(title=dict(text=title, x=0.5, font=dict(size=18)), height=height,
                     margin=dict(l=150, r=80, t=100, b=120), showlegend=False, template=template)
    
    for i in range(1, n_splits + 1):
        heatmap_row = 2 if has_annotation else 1
        if has_annotation:
            fig.update_xaxes(showticklabels=False, row=1, col=i)
            fig.update_yaxes(showticklabels=False, row=1, col=i)
        fig.update_xaxes(tickangle=45, tickfont=dict(size=col_label_size), row=heatmap_row, col=i)
        fig.update_yaxes(tickfont=dict(size=row_label_size), autorange='reversed', showticklabels=(i == 1), row=heatmap_row, col=i)
    
    if has_annotation:
        for val, color in anno_colors.items():
            fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(size=10, color=color), name=str(val), showlegend=True))
        
        if legend_position == "bottom":
            legend_config = dict(orientation='h', yanchor='top', y=-0.15, xanchor='center', x=0.5)
            fig.update_layout(margin=dict(l=150, r=80, t=100, b=180))
        elif legend_position == "right":
            legend_config = dict(orientation='v', yanchor='middle', y=0.5, xanchor='left', x=1.02)
            fig.update_layout(margin=dict(l=150, r=200, t=100, b=120))
        else:
            legend_config = dict(orientation='h', yanchor='top', y=-0.15, xanchor='center', x=0.5)
        
        legend_config['title'] = dict(text=annotation_col.replace('_', ' ').title())
        fig.update_layout(legend=legend_config, showlegend=True)
    
    return fig

def create_dotplot(df: pd.DataFrame, genes: List[str], group_by: str = 'cell_type',
                   size_col: str = 'pct_expressing', color_col: str = 'mean_expr', title: str = "Dot Plot") -> go.Figure:
    """Create a dot plot (size = % expressing, color = mean expression)."""
    template = get_plotly_template()
    
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No data", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(template=template)
        return fig
    
    df = df.copy()
    df['gene_display'] = df['gene_human'].fillna(df['gene_native'])
    
    agg_df = df.groupby(['gene_display', group_by]).agg({
        size_col: 'mean', color_col: 'mean'
    }).reset_index()
    
    agg_df['size_scaled'] = agg_df[size_col] * 30 + 5
    
    fig = px.scatter(
        agg_df, x=group_by, y='gene_display', size='size_scaled', color=color_col,
        color_continuous_scale='Viridis', title=title,
        labels={color_col: 'Mean Expression', 'gene_display': 'Gene', group_by: group_by.replace('_', ' ').title()}
    )
    
    fig.update_traces(
        hovertemplate=f"Gene: %{{y}}<br>{group_by}: %{{x}}<br>Mean expr: %{{marker.color:.3f}}<br>% expressing: %{{customdata:.1%}}<extra></extra>",
        customdata=agg_df[size_col]
    )
    
    fig.update_layout(height=max(400, 50 + len(genes) * 25), xaxis_tickangle=45, 
                     yaxis=dict(autorange='reversed'), template=template)
    
    return fig
# =============================================================================
# Visualization Functions - Temporal Dynamics (IMPROVED)
# =============================================================================

def create_temporal_trajectory_plot(
    temporal_df: pd.DataFrame,
    genes: List[str],
    species: str = 'Human',
    sample_type: str = 'in_vivo',
    cell_types: Optional[List[str]] = None,
    value_col: str = 'mean_expr',
    color_by: str = 'cell_type'  # NEW: color by cell_type, not gene
) -> go.Figure:
    """
    Create temporal trajectory plot showing expression over developmental time.
    Colors by cell type, uses different symbols for genes.
    """
    template = get_plotly_template()
    
    try:
        if temporal_df is None or temporal_df.empty:
            fig = go.Figure()
            fig.add_annotation(text="No temporal data available", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            fig.update_layout(template=template)
            return fig
        
        df = temporal_df[temporal_df['species'] == species].copy()
        
        if 'sample_type' in df.columns and sample_type:
            df = df[df['sample_type'] == sample_type]
        
        # Filter genes safely
        genes_upper = [g.upper() for g in genes]
        df = df[df['gene_human'].notna()]
        df = df[df['gene_human'].str.upper().isin(genes_upper)]
        
        if cell_types and len(cell_types) > 0:
            df = df[df['cell_type'].isin(cell_types)]
        
        if df.empty:
            fig = go.Figure()
            fig.add_annotation(text="No temporal data for selected filters", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            fig.update_layout(template=template)
            return fig
        
        # Sort by time
        df = df.sort_values('time_order')
        
        fig = go.Figure()
        
        # Get unique values
        unique_genes = df['gene_human'].unique().tolist()
        unique_celltypes = df['cell_type'].unique().tolist()
        
        # Color by cell type, symbol by gene
        celltype_colors = get_color_palette(unique_celltypes, 'cell_type')
        gene_symbols = {g: GENE_SYMBOLS[i % len(GENE_SYMBOLS)] for i, g in enumerate(unique_genes)}
        
        for ct in unique_celltypes:
            ct_data = df[df['cell_type'] == ct]
            
            for gene in unique_genes:
                gene_data = ct_data[ct_data['gene_human'] == gene]
                
                if gene_data.empty:
                    continue
                
                # Aggregate by time bin
                agg = gene_data.groupby(['time_bin', 'time_order']).agg({
                    value_col: 'mean',
                    'pct_expressing': 'mean'
                }).reset_index().sort_values('time_order')
                
                if agg.empty or len(agg) < 1:
                    continue
                
                # Only show in legend once per cell type
                show_legend = (gene == unique_genes[0])
                
                fig.add_trace(go.Scatter(
                    x=agg['time_bin'],
                    y=agg[value_col],
                    mode='lines+markers',
                    name=ct if show_legend else None,
                    legendgroup=ct,
                    showlegend=show_legend,
                    line=dict(color=celltype_colors.get(ct, '#999999'), width=2),
                    marker=dict(
                        size=10,
                        symbol=gene_symbols.get(gene, 'circle'),
                        color=celltype_colors.get(ct, '#999999')
                    ),
                    hovertemplate=(
                        f"Gene: {gene}<br>"
                        f"Cell type: {ct}<br>"
                        f"Time: %{{x}}<br>"
                        f"Expression: %{{y:.3f}}<br>"
                        f"<extra></extra>"
                    )
                ))
        
        # Add gene symbol legend as annotation
        if len(unique_genes) > 1:
            symbol_text = "Symbols: " + ", ".join([f"{g} ({gene_symbols[g]})" for g in unique_genes])
            fig.add_annotation(
                text=symbol_text,
                xref="paper", yref="paper",
                x=0.5, y=-0.25,
                showarrow=False,
                font=dict(size=10)
            )
        
        # Title based on species/sample type
        title_map = {
            ('Human', 'in_vivo'): 'Human In Vivo Development',
            ('Human', 'organoid'): 'Human Organoid Differentiation',
            ('Mouse', 'in_vivo'): 'Mouse Development',
            ('Zebrafish', 'in_vivo'): 'Zebrafish Development',
            ('Drosophila', 'in_vivo'): 'Drosophila Aging'
        }
        title = title_map.get((species, sample_type), f'{species} Temporal Expression')
        
        fig.update_layout(
            title=dict(text=title, x=0.5, font=dict(size=18)),
            xaxis_title="Developmental Stage",
            yaxis_title="Mean Expression",
            height=550,
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
            template=template,
            margin=dict(b=100)
        )
        
        return fig
        
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(text=f"Error: {str(e)}", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(template=template)
        return fig

def create_temporal_heatmap(
    temporal_df: pd.DataFrame,
    genes: List[str],
    species: str = 'Human',
    sample_type: str = 'in_vivo',
    cell_type: Optional[str] = None,
    value_col: str = 'mean_expr'
) -> go.Figure:
    """Create a heatmap showing gene expression across developmental time."""
    template = get_plotly_template()
    
    try:
        if temporal_df is None or temporal_df.empty:
            fig = go.Figure()
            fig.add_annotation(text="No temporal data available", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            fig.update_layout(template=template)
            return fig
        
        df = temporal_df[temporal_df['species'] == species].copy()
        
        if 'sample_type' in df.columns and sample_type:
            df = df[df['sample_type'] == sample_type]
        
        genes_upper = [g.upper() for g in genes]
        df = df[df['gene_human'].notna()]
        df = df[df['gene_human'].str.upper().isin(genes_upper)]
        
        if cell_type and cell_type != 'All':
            df = df[df['cell_type'] == cell_type]
        
        if df.empty:
            fig = go.Figure()
            fig.add_annotation(text="No data for selected filters", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            fig.update_layout(template=template)
            return fig
        
        # Aggregate by gene and time_bin
        agg = df.groupby(['gene_human', 'time_bin', 'time_order']).agg({
            value_col: 'mean'
        }).reset_index()
        
        pivot = agg.pivot_table(index='gene_human', columns='time_bin', values=value_col, aggfunc='mean')
        
        if pivot.empty:
            fig = go.Figure()
            fig.add_annotation(text="No data to display", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            fig.update_layout(template=template)
            return fig
        
        # Sort columns by time order
        time_order = agg.groupby('time_bin')['time_order'].first().to_dict()
        sorted_cols = sorted([c for c in pivot.columns if c in time_order], key=lambda x: time_order.get(x, 99))
        pivot = pivot[[c for c in sorted_cols if c in pivot.columns]]
        
        # Z-score scale rows
        row_means = pivot.mean(axis=1)
        row_stds = pivot.std(axis=1).replace(0, 1)
        pivot_scaled = pivot.sub(row_means, axis=0).div(row_stds, axis=0).clip(-3, 3)
        
        fig = go.Figure(go.Heatmap(
            z=pivot_scaled.values,
            x=pivot_scaled.columns.tolist(),
            y=pivot_scaled.index.tolist(),
            colorscale='RdBu_r',
            zmid=0, zmin=-3, zmax=3,
            colorbar=dict(title='Z-score'),
            hovertemplate="Gene: %{y}<br>Time: %{x}<br>Z-score: %{z:.2f}<extra></extra>"
        ))
        
        cell_type_str = f" ({cell_type})" if cell_type and cell_type != 'All' else " (all cell types)"
        fig.update_layout(
            title=dict(text=f"Temporal Expression - {species}{cell_type_str}", x=0.5),
            xaxis_title="Developmental Time",
            yaxis_title="Gene",
            height=max(400, 50 + len(genes) * 20),
            xaxis=dict(tickangle=45),
            yaxis=dict(autorange='reversed'),
            template=template
        )
        
        return fig
        
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(text=f"Error: {str(e)}", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(template=template)
        return fig

def create_multi_species_temporal_comparison(
    temporal_df: pd.DataFrame,
    gene: str,
    cell_types: List[str],
    value_col: str = 'mean_expr'
) -> go.Figure:
    """
    Create a faceted plot comparing one gene's temporal dynamics across all species.
    Facet by species, color by cell type.
    """
    template = get_plotly_template()
    
    try:
        if temporal_df is None or temporal_df.empty:
            fig = go.Figure()
            fig.add_annotation(text="No temporal data", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            fig.update_layout(template=template)
            return fig
        
        df = temporal_df.copy()
        df = df[df['gene_human'].notna()]
        df = df[df['gene_human'].str.upper() == gene.upper()]
        
        if cell_types and len(cell_types) > 0:
            df = df[df['cell_type'].isin(cell_types)]
        
        if df.empty:
            fig = go.Figure()
            fig.add_annotation(text=f"No temporal data for {gene}", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            fig.update_layout(template=template)
            return fig
        
        df = df.sort_values(['species', 'time_order'])
        
        species_list = df['species'].unique().tolist()
        n_species = len(species_list)
        
        fig = make_subplots(
            rows=1, cols=n_species,
            subplot_titles=species_list,
            shared_yaxes=True,
            horizontal_spacing=0.08
        )
        
        celltype_colors = get_color_palette(df['cell_type'].unique().tolist(), 'cell_type')
        
        for i, species in enumerate(species_list):
            sp_data = df[df['species'] == species]
            
            for ct in sp_data['cell_type'].unique():
                ct_data = sp_data[sp_data['cell_type'] == ct]
                
                agg = ct_data.groupby(['time_bin', 'time_order']).agg({
                    value_col: 'mean'
                }).reset_index().sort_values('time_order')
                
                if agg.empty:
                    continue
                
                show_legend = (i == 0)
                
                fig.add_trace(
                    go.Scatter(
                        x=agg['time_bin'],
                        y=agg[value_col],
                        mode='lines+markers',
                        name=ct if show_legend else None,
                        legendgroup=ct,
                        showlegend=show_legend,
                        marker=dict(size=8, color=celltype_colors.get(ct, '#999999')),
                        line=dict(color=celltype_colors.get(ct, '#999999'), width=2),
                        hovertemplate=f"Cell type: {ct}<br>Time: %{{x}}<br>Expression: %{{y:.3f}}<extra></extra>"
                    ),
                    row=1, col=i+1
                )
            
            fig.update_xaxes(tickangle=45, row=1, col=i+1)
        
        fig.update_layout(
            title=dict(text=f"Temporal Expression of {gene.upper()} Across Species", x=0.5, font=dict(size=18)),
            height=450,
            legend=dict(orientation='h', yanchor='bottom', y=-0.35, xanchor='center', x=0.5),
            template=template,
            margin=dict(b=120)
        )
        
        fig.update_yaxes(title_text="Mean Expression", row=1, col=1)
        
        return fig
        
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(text=f"Error: {str(e)}", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(template=template)
        return fig

def create_celltype_by_species_temporal(
    temporal_df: pd.DataFrame,
    gene: str,
    time_bin: str,
    value_col: str = 'mean_expr'
) -> go.Figure:
    """
    Create a bar chart showing expression of a gene at a specific timepoint,
    grouped by cell type, faceted by species.
    """
    template = get_plotly_template()
    
    try:
        if temporal_df is None or temporal_df.empty:
            fig = go.Figure()
            fig.add_annotation(text="No temporal data", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            fig.update_layout(template=template)
            return fig
        
        df = temporal_df.copy()
        df = df[df['gene_human'].notna()]
        df = df[df['gene_human'].str.upper() == gene.upper()]
        df = df[df['time_bin'] == time_bin]
        
        if df.empty:
            fig = go.Figure()
            fig.add_annotation(text=f"No data for {gene} at {time_bin}", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            fig.update_layout(template=template)
            return fig
        
        agg = df.groupby(['species', 'cell_type']).agg({
            value_col: 'mean',
            'pct_expressing': 'mean'
        }).reset_index()
        
        fig = px.bar(
            agg,
            x='cell_type',
            y=value_col,
            color='species',
            barmode='group',
            color_discrete_map=SPECIES_COLORS,
            title=f"{gene.upper()} Expression at {time_bin}",
            labels={value_col: 'Mean Expression', 'cell_type': 'Cell Type'}
        )
        
        fig.update_layout(
            height=450,
            xaxis_tickangle=45,
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
            template=template
        )
        
        return fig
        
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(text=f"Error: {str(e)}", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(template=template)
        return fig
# =============================================================================
# Visualization Functions - Cross-Species Comparison
# =============================================================================

def create_species_expression_comparison(
    ortholog_df: pd.DataFrame,
    genes: List[str],
    cell_types: Optional[List[str]] = None
) -> go.Figure:
    """Create a grouped bar chart comparing expression across species."""
    template = get_plotly_template()
    
    try:
        if ortholog_df is None or ortholog_df.empty:
            fig = go.Figure()
            fig.add_annotation(text="No ortholog data", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            fig.update_layout(template=template)
            return fig
        
        df = ortholog_df.copy()
        genes_upper = [g.upper() for g in genes]
        df = df[df['gene_human'].notna()]
        df = df[df['gene_human'].str.upper().isin(genes_upper)]
        
        if cell_types and len(cell_types) > 0:
            df = df[df['cell_type'].isin(cell_types)]
        
        if df.empty:
            fig = go.Figure()
            fig.add_annotation(text="No data for selected genes", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            fig.update_layout(template=template)
            return fig
        
        agg = df.groupby(['gene_human', 'species']).agg({'mean_expr': 'mean'}).reset_index()
        
        fig = px.bar(
            agg, x='gene_human', y='mean_expr', color='species', barmode='group',
            color_discrete_map=SPECIES_COLORS, title="Cross-Species Expression Comparison",
            labels={'mean_expr': 'Mean Expression', 'gene_human': 'Gene', 'species': 'Species'}
        )
        
        fig.update_layout(
            height=450, xaxis_tickangle=45,
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
            template=template
        )
        
        return fig
        
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(text=f"Error: {str(e)}", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(template=template)
        return fig

def create_ortholog_scatter(
    ortholog_df: pd.DataFrame,
    gene: str,
    species_x: str,
    species_y: str
) -> go.Figure:
    """Create a scatter plot comparing expression of a gene between two species."""
    template = get_plotly_template()
    
    try:
        if ortholog_df is None or ortholog_df.empty:
            fig = go.Figure()
            fig.add_annotation(text="No ortholog data", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            fig.update_layout(template=template)
            return fig
        
        df = ortholog_df[ortholog_df['gene_human'].notna()].copy()
        df = df[df['gene_human'].str.upper() == gene.upper()]
        
        df_x = df[df['species'] == species_x][['cell_type', 'mean_expr']].rename(columns={'mean_expr': 'expr_x'})
        df_y = df[df['species'] == species_y][['cell_type', 'mean_expr']].rename(columns={'mean_expr': 'expr_y'})
        
        merged = df_x.merge(df_y, on='cell_type', how='inner')
        
        if merged.empty:
            fig = go.Figure()
            fig.add_annotation(text=f"No common cell types for {gene}", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            fig.update_layout(template=template)
            return fig
        
        corr_text = ""
        if len(merged) >= 3:
            corr, pval = spearmanr(merged['expr_x'], merged['expr_y'])
            corr_text = f"ρ = {corr:.3f}"
        
        fig = px.scatter(
            merged, x='expr_x', y='expr_y', color='cell_type',
            color_discrete_map=CELLTYPE_COLORS,
            title=f"{gene.upper()}: {species_x} vs {species_y}",
            labels={'expr_x': f'{species_x} Expression', 'expr_y': f'{species_y} Expression', 'cell_type': 'Cell Type'}
        )
        
        # Add diagonal line
        max_val = max(merged['expr_x'].max(), merged['expr_y'].max())
        fig.add_trace(go.Scatter(
            x=[0, max_val], y=[0, max_val],
            mode='lines', line=dict(dash='dash', color='gray'),
            showlegend=False, hoverinfo='skip'
        ))
        
        if corr_text:
            fig.add_annotation(text=corr_text, xref="paper", yref="paper", x=0.95, y=0.05, showarrow=False, font=dict(size=14))
        
        fig.update_traces(marker=dict(size=12))
        fig.update_layout(height=500, template=template)
        
        return fig
        
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(text=f"Error: {str(e)}", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(template=template)
        return fig

def create_species_correlation_heatmap(
    species_comparison_df: pd.DataFrame,
    genes: Optional[List[str]] = None
) -> go.Figure:
    """Create a heatmap showing cross-species expression correlations."""
    template = get_plotly_template()
    
    try:
        if species_comparison_df is None or species_comparison_df.empty:
            fig = go.Figure()
            fig.add_annotation(text="No species comparison data", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            fig.update_layout(template=template)
            return fig
        
        df = species_comparison_df.copy()
        
        if genes and len(genes) > 0:
            genes_upper = [g.upper() for g in genes]
            df = df[df['gene_human'].str.upper().isin(genes_upper)]
        
        if df.empty:
            fig = go.Figure()
            fig.add_annotation(text="No data for selected genes", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            fig.update_layout(template=template)
            return fig
        
        df['species_pair'] = df['species_1'] + ' vs ' + df['species_2']
        
        pivot = df.pivot_table(
            index='gene_human', columns='species_pair',
            values='expression_correlation', aggfunc='mean'
        )
        
        fig = go.Figure(go.Heatmap(
            z=pivot.values,
            x=pivot.columns.tolist(),
            y=pivot.index.tolist(),
            colorscale='RdBu',
            zmid=0, zmin=-1, zmax=1,
            colorbar=dict(title='Correlation'),
            hovertemplate="Gene: %{y}<br>Comparison: %{x}<br>Correlation: %{z:.3f}<extra></extra>"
        ))
        
        fig.update_layout(
            title=dict(text="Cross-Species Expression Correlation", x=0.5),
            xaxis_title="Species Comparison",
            yaxis_title="Gene",
            height=max(400, 50 + len(pivot) * 18),
            xaxis=dict(tickangle=45),
            yaxis=dict(autorange='reversed'),
            template=template
        )
        
        return fig
        
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(text=f"Error: {str(e)}", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(template=template)
        return fig

# =============================================================================
# Visualization Functions - Data Overview
# =============================================================================

def create_variance_partition_barplot(vp_summary: pd.DataFrame) -> go.Figure:
    """Create variance partition before/after bar plot."""
    template = get_plotly_template()
    
    try:
        if vp_summary is None or vp_summary.empty:
            fig = go.Figure()
            fig.add_annotation(text="No variance partition data", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            fig.update_layout(template=template)
            return fig
        
        colors = {'Before Correction': '#ff7f0e', 'After Correction': '#1f77b4'}
        
        fig = go.Figure()
        for stage in ['Before Correction', 'After Correction']:
            stage_data = vp_summary[vp_summary['stage'] == stage]
            fig.add_trace(go.Bar(
                name=stage, x=stage_data['variable'], y=stage_data['variance_explained'],
                marker_color=colors[stage],
                text=[f"{v:.1f}%" for v in stage_data['variance_explained']],
                textposition='outside'
            ))
        
        fig.update_layout(
            title=dict(text="Variance Explained by Each Factor", x=0.5, font=dict(size=18)),
            xaxis_title="Variable", yaxis_title="Variance Explained (%)",
            barmode='group', height=450,
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
            yaxis=dict(range=[0, max(vp_summary['variance_explained']) * 1.15]),
            template=template
        )
        
        fig.add_annotation(text="Goal: ↓ dataset (batch), ↑ or = biological factors",
                          xref="paper", yref="paper", x=0.5, y=-0.15, showarrow=False, font=dict(size=11, color='gray'))
        
        return fig
        
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(text=f"Error: {str(e)}", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(template=template)
        return fig

def create_variance_change_plot(vp_summary: pd.DataFrame) -> go.Figure:
    """Create variance change waterfall plot."""
    template = get_plotly_template()
    
    try:
        if vp_summary is None or vp_summary.empty:
            fig = go.Figure()
            fig.add_annotation(text="No variance partition data", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            fig.update_layout(template=template)
            return fig
        
        before = vp_summary[vp_summary['stage'] == 'Before Correction'].set_index('variable')['variance_explained']
        after = vp_summary[vp_summary['stage'] == 'After Correction'].set_index('variable')['variance_explained']
        
        common_vars = before.index.intersection(after.index)
        vars_df = pd.DataFrame({
            'variable': common_vars,
            'change': [after[v] - before[v] for v in common_vars]
        })
        
        colors = []
        for _, row in vars_df.iterrows():
            var, change = row['variable'], row['change']
            if var == 'dataset':
                colors.append('#28a745' if change < 0 else '#dc3545')
            elif var == 'Residuals':
                colors.append('#6c757d')
            else:
                colors.append('#28a745' if change >= 0 else '#dc3545')
        
        fig = go.Figure(go.Bar(
            x=vars_df['variable'], y=vars_df['change'], marker_color=colors,
            text=[f"{c:+.1f}%" for c in vars_df['change']], textposition='outside'
        ))
        
        fig.update_layout(
            title=dict(text="Change in Variance After Batch Correction", x=0.5, font=dict(size=18)),
            xaxis_title="Variable", yaxis_title="Change in Variance (%)", height=400,
            yaxis=dict(zeroline=True, zerolinewidth=2, zerolinecolor='black'),
            template=template
        )
        
        return fig
        
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(text=f"Error: {str(e)}", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(template=template)
        return fig

def create_dataset_summary_table(dataset_info: pd.DataFrame) -> pd.DataFrame:
    """Format dataset info for display."""
    if dataset_info is None or dataset_info.empty:
        return pd.DataFrame()
    
    display_cols = ['dataset', 'organism', 'sample_type', 'n_pseudobulk_samples', 'n_cell_types', 'n_timepoints']
    available_cols = [c for c in display_cols if c in dataset_info.columns]
    
    df = dataset_info[available_cols].copy()
    
    rename_map = {
        'dataset': 'Dataset', 'organism': 'Species', 'sample_type': 'Sample Type',
        'n_pseudobulk_samples': 'Pseudobulk Samples', 'n_cell_types': 'Cell Types', 'n_timepoints': 'Timepoints'
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    
    return df
# =============================================================================
# Main App
# =============================================================================

def main():
    # Header with dark mode toggle
    col_title, col_toggle = st.columns([9, 1])
    with col_title:
        st.title("🧬 SFARI Gene Expression Explorer")
        st.markdown("*Cross-species single-cell RNA-seq browser for neurodevelopmental gene expression*")
    with col_toggle:
        if st.button("🌙" if not st.session_state.dark_mode else "☀️", help="Toggle dark mode"):
            st.session_state.dark_mode = not st.session_state.dark_mode
            st.rerun()
    
    # Load data
    data = load_data()
    
    if data is None:
        st.error("Failed to load data. Please check that the data files exist in the 'data' directory.")
        st.stop()
    
    expr_df = data['expression']
    risk_genes = data['risk_genes']
    
    # ==========================================================================
    # Sidebar - Filters
    # ==========================================================================
    
    with st.sidebar:
        st.header("🔍 Query Filters")
        
        # Species selection
        all_species = get_unique_values(expr_df, 'species')
        selected_species = st.multiselect(
            "Species",
            options=all_species,
            default=all_species[:1] if all_species else [],
            help="Select one or more species"
        )
        
        # Dataset/Tissue selection (filtered by species)
        if selected_species:
            available_datasets = get_unique_values(
                expr_df[expr_df['species'].isin(selected_species)], 'tissue'
            )
        else:
            available_datasets = get_unique_values(expr_df, 'tissue')
        
        selected_datasets = st.multiselect(
            "Dataset",
            options=available_datasets,
            default=[],
            help="Filter by dataset (leave empty for all)"
        )
        
        # Cell type selection - filter based on selected species AND datasets
        if selected_species:
            subset = expr_df[expr_df['species'].isin(selected_species)]
            if selected_datasets:
                subset = subset[subset['tissue'].isin(selected_datasets)]
            available_celltypes = get_unique_values(subset, 'cell_type')
        else:
            available_celltypes = get_unique_values(expr_df, 'cell_type')
        
        selected_celltypes = st.multiselect(
            "Cell Types",
            options=available_celltypes,
            default=[],
            help="Filter by cell type (leave empty for all)"
        )
        
        st.divider()
        
        # Gene input
        st.subheader("🧬 Gene Selection")
        
        gene_preset = st.selectbox(
            "Quick Gene Sets",
            options=[
                "Custom",
                "SFARI Score 1 (High Confidence)",
                "SFARI Score 2",
                "SFARI Syndromic",
                "Top Variable Genes"
            ],
            index=0
        )
        
        preset_genes = ""
        if gene_preset == "SFARI Score 1 (High Confidence)":
            score1_genes = risk_genes[risk_genes['gene_score'] == 1]['gene_symbol'].dropna().tolist()[:50]
            preset_genes = ", ".join(score1_genes)
        elif gene_preset == "SFARI Score 2":
            score2_genes = risk_genes[risk_genes['gene_score'] == 2]['gene_symbol'].dropna().tolist()[:50]
            preset_genes = ", ".join(score2_genes)
        elif gene_preset == "SFARI Syndromic":
            if 'syndromic' in risk_genes.columns:
                synd_genes = risk_genes[risk_genes['syndromic'] == 1]['gene_symbol'].dropna().tolist()[:50]
                preset_genes = ", ".join(synd_genes)
        elif gene_preset == "Top Variable Genes":
            var_df = expr_df.groupby('gene_human')['mean_expr'].var().sort_values(ascending=False)
            preset_genes = ", ".join(var_df.head(30).index.tolist())
        
        gene_input = st.text_area(
            "Enter genes (comma/space separated)",
            value=preset_genes,
            height=100,
            placeholder="e.g., SHANK3, MECP2, CHD8, SCN2A",
            help="Enter gene symbols. Human gene names are preferred but native species names are also supported."
        )
        
        selected_genes = parse_gene_input(gene_input)
        
        st.divider()
        
        # Display options
        st.subheader("⚙️ Display Options")
        
        value_metric = st.radio(
            "Value to display",
            options=['mean_expr', 'pct_expressing'],
            format_func=lambda x: "Mean Expression" if x == 'mean_expr' else "% Expressing",
            horizontal=True
        )
        
        scale_rows = st.checkbox("Row Z-score scaling", value=True)
        cluster_rows = st.checkbox("Cluster rows (genes)", value=False)
        cluster_cols = st.checkbox("Cluster columns", value=False)
    
    # ==========================================================================
    # Main Content
    # ==========================================================================
    
    # Filter data
    filtered_df = filter_expression_data(
        expr_df,
        species=selected_species if selected_species else None,
        datasets=selected_datasets if selected_datasets else None,
        cell_types=selected_celltypes if selected_celltypes else None,
        genes=selected_genes if selected_genes else None
    )
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Genes", filtered_df['gene_human'].nunique() if not filtered_df.empty else 0)
    with col2:
        st.metric("Species", filtered_df['species'].nunique() if not filtered_df.empty else 0)
    with col3:
        st.metric("Datasets", filtered_df['tissue'].nunique() if not filtered_df.empty else 0)
    with col4:
        st.metric("Cell Types", filtered_df['cell_type'].nunique() if not filtered_df.empty else 0)
    
    # Tabs
    tab_overview, tab_umap, tab_heatmap, tab_dotplot, tab_temporal, tab_species, tab_table, tab_about = st.tabs([
        "📊 Data Overview", "🗺️ UMAP", "🔥 Heatmap", "🔵 Dot Plot",
        "📈 Temporal", "🔬 Cross-Species", "📋 Data Table", "📚 About"
    ])
    
    # --------------------------------------------------------------------------
    # Tab: Data Overview
    # --------------------------------------------------------------------------
    with tab_overview:
        st.header("Data Overview & Batch Correction")
        
        if data['summary_stats'] is not None:
            st.subheader("📈 Dataset Summary")
            summary = data['summary_stats']
            totals = summary[summary['category'] == 'totals'].set_index('label')['value']
            
            cols = st.columns(4)
            metrics = [('pseudobulk_samples', 'Pseudobulk Samples'), ('datasets', 'Datasets'),
                      ('cell_types', 'Cell Types'), ('organisms', 'Species')]
            for i, (key, label) in enumerate(metrics):
                if key in totals.index:
                    with cols[i]:
                        st.metric(label, f"{int(totals[key]):,}")
        
        if data['dataset_info'] is not None:
            st.subheader("📋 Datasets Included")
            dataset_table = create_dataset_summary_table(data['dataset_info'])
            st.dataframe(dataset_table, hide_index=True, use_container_width=True)
        
        st.divider()
        
        st.subheader("🔧 Batch Correction Methodology")
        
        if data['batch_correction'] is not None:
            st.markdown("""
            <div class="info-box">
            <strong>Approach:</strong> Within-organism batch correction using ComBat,
            preserving biological covariates (cell type, developmental time) while
            removing technical batch effects (dataset).
            </div>
            """, unsafe_allow_html=True)
            
            for _, row in data['batch_correction'].iterrows():
                if row.get('correction_applied', False):
                    st.markdown(f"- **{row['correction_group']}**: {row['n_datasets']} datasets ({row['method']})")
        else:
            st.info("Batch correction information not available.")
        
        st.divider()
        
        st.subheader("📊 Variance Partition Analysis")
        
        if data['vp_summary'] is not None:
            st.markdown("""
            Variance partition quantifies how much expression variance is explained by each factor.
            **Successful batch correction** should decrease variance from `dataset` while preserving biological factors.
            """)
            
            vp_col1, vp_col2 = st.columns(2)
            with vp_col1:
                st.plotly_chart(create_variance_partition_barplot(data['vp_summary']), use_container_width=True)
            with vp_col2:
                st.plotly_chart(create_variance_change_plot(data['vp_summary']), use_container_width=True)
            
            # Check for success
            vp_df = data['vp_summary']
            before_dataset = vp_df[(vp_df['variable'] == 'dataset') & (vp_df['stage'] == 'Before Correction')]['variance_explained'].values
            after_dataset = vp_df[(vp_df['variable'] == 'dataset') & (vp_df['stage'] == 'After Correction')]['variance_explained'].values
            
            if len(before_dataset) > 0 and len(after_dataset) > 0:
                reduction = before_dataset[0] - after_dataset[0]
                if reduction > 0:
                    st.markdown(f"""
                    <div class="success-box">
                    ✅ <strong>Batch correction successful!</strong> Dataset variance reduced by {reduction:.1f}%
                    ({before_dataset[0]:.1f}% → {after_dataset[0]:.1f}%)
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("Variance partition data not available.")
    # --------------------------------------------------------------------------
    # Tab: UMAP
    # --------------------------------------------------------------------------
    with tab_umap:
        st.header("UMAP Visualization")
        
        if data['umap'] is not None:
            umap_df = data['umap']
            st.markdown(f"Interactive UMAP of **{len(umap_df):,} cells** subsampled from integrated dataset.")
            
            available_color_cols = [c for c in umap_df.columns if c not in ['umap_1', 'umap_2', 'cell_id']]
            
            col1, col2 = st.columns([1, 3])
            with col1:
                color_by = st.selectbox(
                    "Color by:", options=available_color_cols,
                    index=available_color_cols.index('predicted_labels') if 'predicted_labels' in available_color_cols else 0
                )
                show_legend = st.checkbox("Show legend", value=True)
                point_size = st.slider("Point size", 1, 10, 3)
                opacity = st.slider("Opacity", 0.1, 1.0, 0.6)
            
            with col2:
                unique_vals = umap_df[color_by].unique().tolist()
                
                if color_by == 'organism':
                    color_map = SPECIES_COLORS
                elif color_by in ['predicted_labels', 'cell_type']:
                    color_map = CELLTYPE_COLORS
                elif color_by == 'dataset':
                    color_map = DATASET_COLORS
                else:
                    color_map = get_color_palette(unique_vals, 'auto')
                
                fig_umap = px.scatter(
                    umap_df,
                    x='umap_1' if 'umap_1' in umap_df.columns else umap_df.columns[0],
                    y='umap_2' if 'umap_2' in umap_df.columns else umap_df.columns[1],
                    color=color_by,
                    color_discrete_map=color_map,
                    title=f"UMAP colored by {color_by.replace('_', ' ').title()}",
                    labels={'umap_1': 'UMAP 1', 'umap_2': 'UMAP 2'},
                    hover_data=[c for c in ['organism', 'dataset', 'predicted_labels'] if c in umap_df.columns]
                )
                
                fig_umap.update_traces(marker=dict(size=point_size, opacity=opacity))
                fig_umap.update_layout(
                    height=650,
                    showlegend=show_legend,
                    legend=dict(orientation='v', yanchor='top', y=1, xanchor='left', x=1.02, font=dict(size=9)),
                    template=get_plotly_template()
                )
                
                st.plotly_chart(fig_umap, use_container_width=True)
            
            with st.expander("📥 Download UMAP Data"):
                csv = umap_df.to_csv(index=False)
                st.download_button("Download UMAP coordinates as CSV", csv, "umap_coordinates.csv", "text/csv")
        else:
            st.info("UMAP data not available. Generate `umap_subsample.parquet` to enable this feature.")
    
    # --------------------------------------------------------------------------
    # Tab: Heatmap
    # --------------------------------------------------------------------------
    with tab_heatmap:
        if filtered_df.empty:
            st.warning("No data matches your filters. Try broadening your selection.")
        elif not selected_genes:
            st.info("Enter gene names in the sidebar to generate a heatmap.")
        else:
            st.markdown("**Heatmap Settings**")
            hm_col1, hm_col2, hm_col3, hm_col4 = st.columns(4)
            
            with hm_col1:
                split_option = st.selectbox("Split heatmap by", ["None", "Species", "Dataset", "Cell Type"], index=0)
                split_by = {"None": None, "Species": "species", "Dataset": "dataset", "Cell Type": "cell_type"}[split_option]
            
            with hm_col2:
                annotation_option = st.selectbox("Top annotation", ["None", "Species", "Dataset", "Cell Type"], index=0)
                annotation_col = {"None": None, "Species": "species", "Dataset": "dataset", "Cell Type": "cell_type"}[annotation_option]
            
            with hm_col3:
                color_scale = st.selectbox("Color scale", ["RdBu_r", "Viridis", "Plasma", "Inferno", "Blues", "RdYlBu_r", "PiYG"])
            
            with hm_col4:
                legend_pos = st.selectbox("Legend position", ["Bottom", "Right", "Left", "Top"])
            
            hm_col5, hm_col6, hm_col7, hm_col8 = st.columns(4)
            
            with hm_col5:
                do_cluster_rows = st.checkbox("Cluster rows (genes)", value=cluster_rows, key='hm_cluster_rows')
            with hm_col6:
                do_cluster_cols = st.checkbox("Cluster columns", value=cluster_cols, key='hm_cluster_cols')
            with hm_col7:
                row_font = st.slider("Gene label size", 6, 14, 9)
            with hm_col8:
                col_font = st.slider("Column label size", 6, 14, 9)
            
            matrix, col_meta = create_heatmap_matrix(filtered_df, value_col=value_metric, scale_rows=scale_rows)
            
            fig = create_complexheatmap(
                matrix=matrix, col_meta=col_meta,
                title=f"Expression Heatmap ({len(selected_genes)} genes)",
                color_scale=color_scale, split_by=split_by, annotation_col=annotation_col,
                cluster_rows=do_cluster_rows, cluster_cols=do_cluster_cols,
                row_label_size=row_font, col_label_size=col_font,
                legend_position=legend_pos.lower()
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            with st.expander("📥 Download Matrix"):
                csv = matrix.to_csv()
                st.download_button("Download as CSV", csv, "expression_matrix.csv", "text/csv")
    
    # --------------------------------------------------------------------------
    # Tab: Dot Plot
    # --------------------------------------------------------------------------
    with tab_dotplot:
        if filtered_df.empty or not selected_genes:
            st.info("Select filters and genes to generate a dot plot.")
        else:
            dot_group = st.selectbox(
                "Group by", ['cell_type', 'tissue', 'species'],
                format_func=lambda x: x.replace('_', ' ').title()
            )
            
            fig = create_dotplot(filtered_df, selected_genes, group_by=dot_group,
                               title="Dot Plot: Size = % Expressing, Color = Mean Expression")
            st.plotly_chart(fig, use_container_width=True)
    
    # --------------------------------------------------------------------------
    # Tab: Temporal Dynamics
    # --------------------------------------------------------------------------
    with tab_temporal:
        st.header("Temporal Expression Dynamics")
        
        if data['temporal'] is not None:
            temporal_df = data['temporal']
            
            st.markdown("""
            Explore how gene expression changes across developmental time.
            **Colors = cell types, Symbols = genes** (in trajectory plots).
            """)
            
            # Visualization type selection
            viz_type = st.selectbox(
                "Visualization Type",
                ["Trajectory Plot", "Temporal Heatmap", "Multi-Species Comparison", "Timepoint Snapshot"],
                key='temporal_viz_type'
            )
            
            temp_col1, temp_col2, temp_col3 = st.columns(3)
            
            with temp_col1:
                available_species = temporal_df['species'].unique().tolist()
                temp_species = st.selectbox("Species", options=available_species, index=0, key='temp_species')
            
            with temp_col2:
                # Sample type (for Human)
                if temp_species == 'Human' and 'sample_type' in temporal_df.columns:
                    sample_types = temporal_df[temporal_df['species'] == 'Human']['sample_type'].dropna().unique().tolist()
                    if sample_types:
                        temp_sample_type = st.selectbox("Sample Type", options=sample_types, index=0, key='temp_sample_type')
                    else:
                        temp_sample_type = 'in_vivo'
                else:
                    temp_sample_type = 'in_vivo'
                    st.text("Sample Type: in_vivo")
            
            with temp_col3:
                # Get available cell types for this species
                species_temporal = temporal_df[temporal_df['species'] == temp_species]
                if 'sample_type' in species_temporal.columns:
                    species_temporal = species_temporal[species_temporal['sample_type'] == temp_sample_type]
                available_temp_cts = species_temporal['cell_type'].unique().tolist()
            
            # Cell type selection
            selected_temp_cts = st.multiselect(
                "Cell Types",
                options=available_temp_cts,
                default=available_temp_cts[:3] if len(available_temp_cts) > 3 else available_temp_cts,
                key='temp_celltypes'
            )
            
            if not selected_genes:
                st.info("Enter gene names in the sidebar to visualize temporal dynamics.")
            else:
                if viz_type == "Trajectory Plot":
                    fig = create_temporal_trajectory_plot(
                        temporal_df, selected_genes, temp_species, temp_sample_type,
                        selected_temp_cts, value_metric
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                elif viz_type == "Temporal Heatmap":
                    hm_celltype = st.selectbox("Cell Type for Heatmap", options=['All'] + available_temp_cts, key='temp_hm_ct')
                    fig = create_temporal_heatmap(
                        temporal_df, selected_genes, temp_species, temp_sample_type,
                        None if hm_celltype == 'All' else hm_celltype, value_metric
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                elif viz_type == "Multi-Species Comparison":
                    if len(selected_genes) > 0:
                        comp_gene = st.selectbox("Gene for comparison", options=selected_genes, key='temp_comp_gene')
                        fig = create_multi_species_temporal_comparison(
                            temporal_df, comp_gene, selected_temp_cts, value_metric
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
                elif viz_type == "Timepoint Snapshot":
                    # Get available time bins for selected species
                    available_time_bins = species_temporal['time_bin'].unique().tolist()
                    if available_time_bins:
                        selected_time_bin = st.selectbox("Select Timepoint", options=available_time_bins, key='temp_time_bin')
                        if len(selected_genes) > 0:
                            comp_gene = st.selectbox("Gene for snapshot", options=selected_genes, key='temp_snap_gene')
                            fig = create_celltype_by_species_temporal(temporal_df, comp_gene, selected_time_bin, value_metric)
                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("No time bins available for selected species.")
        else:
            st.info("""
            **Temporal data not available.**
            
            To enable temporal dynamics visualization, generate `temporal_expression.parquet`
            using the `generate_temporal_parquets.py` script.
            """)
    # --------------------------------------------------------------------------
    # Tab: Cross-Species Comparison
    # --------------------------------------------------------------------------
    with tab_species:
        st.header("Cross-Species Comparison")
        
        if data['ortholog'] is not None:
            ortholog_df = data['ortholog']
            
            st.markdown("""
            Compare gene expression across species using ortholog mappings.
            All genes are mapped to human gene symbols for cross-species comparison.
            """)
            
            species_viz = st.selectbox(
                "Visualization Type",
                ["Expression Bar Chart", "Species Correlation Heatmap", "Ortholog Scatter Plot"],
                key='species_viz'
            )
            
            if not selected_genes:
                st.info("Enter gene names in the sidebar to compare across species.")
            else:
                if species_viz == "Expression Bar Chart":
                    fig = create_species_expression_comparison(ortholog_df, selected_genes, selected_celltypes or None)
                    st.plotly_chart(fig, use_container_width=True)
                
                elif species_viz == "Species Correlation Heatmap":
                    if data['species_comparison'] is not None:
                        fig = create_species_correlation_heatmap(data['species_comparison'], selected_genes)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Species comparison data not available. Generate `species_comparison.parquet`.")
                
                elif species_viz == "Ortholog Scatter Plot":
                    scatter_col1, scatter_col2, scatter_col3 = st.columns(3)
                    
                    available_species = ortholog_df['species'].unique().tolist()
                    
                    with scatter_col1:
                        scatter_gene = st.selectbox("Gene", options=selected_genes, key='scatter_gene')
                    with scatter_col2:
                        species_x = st.selectbox("Species X", options=available_species, index=0, key='sp_x')
                    with scatter_col3:
                        other_species = [s for s in available_species if s != species_x]
                        species_y = st.selectbox("Species Y", options=other_species, index=0 if other_species else 0, key='sp_y')
                    
                    fig = create_ortholog_scatter(ortholog_df, scatter_gene, species_x, species_y)
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("""
            **Ortholog expression data not available.**
            
            To enable cross-species comparison, generate `ortholog_expression.parquet`
            using the `generate_temporal_parquets.py` script.
            """)
    
    # --------------------------------------------------------------------------
    # Tab: Data Table
    # --------------------------------------------------------------------------
    with tab_table:
        if filtered_df.empty:
            st.info("No data matches your current filters.")
        else:
            display_df = filtered_df.copy()
            display_df['gene_display'] = display_df['gene_human'].fillna(display_df['gene_native'])
            
            # Add SFARI score
            if 'gene_symbol' in risk_genes.columns:
                risk_lookup = risk_genes.set_index('gene_symbol')['gene_score'].to_dict()
                display_df['SFARI_score'] = display_df['gene_display'].map(risk_lookup)
            
            st.markdown("**Select columns to display:**")
            available_cols = display_df.columns.tolist()
            default_cols = ['gene_display', 'species', 'tissue', 'cell_type', 'mean_expr', 'pct_expressing', 'n_cells']
            default_cols = [c for c in default_cols if c in available_cols]
            
            selected_cols = st.multiselect(
                "Columns", options=available_cols, default=default_cols, label_visibility="collapsed"
            )
            
            if selected_cols:
                st.dataframe(
                    display_df[selected_cols].sort_values(['gene_display', 'species', 'tissue']),
                    height=500, use_container_width=True
                )
                
                csv = display_df[selected_cols].to_csv(index=False)
                st.download_button("📥 Download Table as CSV", csv, "filtered_expression_data.csv", "text/csv")
    
    # --------------------------------------------------------------------------
    # Tab: About
    # --------------------------------------------------------------------------
    with tab_about:
        st.markdown("""
        ## About SFARI Gene Expression Explorer
        
        This interactive web application enables exploration of gene expression patterns
        across multiple single-cell RNA-seq datasets from developing brain tissue, spanning
        multiple species (Human, Mouse, Zebrafish, Drosophila).
        
        ### Purpose
        
        The explorer is designed to help researchers:
        - **Search** for genes of interest across species and developmental stages
        - **Visualize** expression patterns using interactive heatmaps and dot plots
        - **Compare** expression across species, datasets, and cell types
        - **Track** temporal dynamics across development
        - **Identify** cell-type-specific expression of neurodevelopmental risk genes
        
        ---
        
        ## Datasets
        
        ### Human Datasets
        
        | Dataset | Sample Type | Description |
        |---------|-------------|-------------|
        | **He (2024)** | Organoid | Human Neural Organoid Cell Atlas (HNOCA) |
        | **Bhaduri (2021)** | Primary | Primary human cortical development |
        | **Braun (2023)** | Primary | Human brain cell atlas |
        | **Velmeshev (2023)** | Primary | Developing human brain cell types |
        | **Velmeshev (2019)** | Primary | Single-cell genomics of ASD brain |
        | **Zhu (2023)** | Primary | Human fetal brain development |
        | **Wang (2025)** | Primary | Human brain development atlas |
        | **Wang (2022)** | Organoid | Human cerebral organoids |
        
        ### Mouse Datasets
        
        | Dataset | Description |
        |---------|-------------|
        | **La Manno (2021)** | Mouse brain development atlas |
        | **Jin (2025)** | Mouse brain cell atlas |
        | **Sziraki (2023)** | Mouse brain cell types |
        
        ### Other Species
        
        | Dataset | Species | Description |
        |---------|---------|-------------|
        | **Raj (2020)** | Zebrafish | Zebrafish brain development |
        | **Davie (2018)** | Drosophila | *Drosophila* brain cell atlas |
        
        ---
        
        ## Data Processing
        
        Expression data was processed as follows:
        1. **Integration** via scVI/scANVI for batch correction
        2. **Within-organism ComBat** batch correction preserving cell type and developmental time
        3. **Timepoint normalization** to continuous developmental scale
        4. **Pseudobulk aggregation** per cell type per dataset
        5. **Ortholog mapping** to human gene symbols for cross-species comparison
        
        ### Cell Type Categories
        
        Cell types were harmonized into the following supercategories:
        """)
        
        celltype_df = pd.DataFrame([
            {"Category": "Excitatory Neurons", "Description": "Glutamatergic projection neurons", "Color": "🔴"},
            {"Category": "Inhibitory Neurons", "Description": "GABAergic interneurons", "Color": "🔵"},
            {"Category": "Neural Progenitors & Stem Cells", "Description": "NPCs, radial glia, neural stem cells", "Color": "🟢"},
            {"Category": "Astrocytes", "Description": "Astrocytes and astrocyte precursors", "Color": "🟣"},
            {"Category": "Oligodendrocyte Lineage", "Description": "OPCs, oligodendrocytes", "Color": "🟠"},
            {"Category": "Microglia & Macrophages", "Description": "Brain-resident immune cells", "Color": "🟡"},
            {"Category": "Endothelial & Vascular Cells", "Description": "Blood vessel cells", "Color": "🟤"},
            {"Category": "Other Glia & Support", "Description": "Other glial cell types", "Color": "🩷"},
            {"Category": "Neurons (unspecified)", "Description": "Neurons without E/I classification", "Color": "⚪"},
            {"Category": "Early Embryonic / Germ Layers", "Description": "Early developmental cell types", "Color": "🩵"},
        ])
        st.dataframe(celltype_df, hide_index=True, use_container_width=True)
        
        st.markdown("""
        ---
        
        ## SFARI Gene Integration
        
        This explorer integrates the [SFARI Gene database](https://gene.sfari.org/), which catalogs
        genes implicated in autism spectrum disorder (ASD). Genes are scored based on strength of evidence:
        
        - **Score 1 (High Confidence)**: Strong evidence from multiple studies
        - **Score 2**: Moderate evidence
        - **Score 3**: Suggestive evidence
        
        Use the "Quick Gene Sets" dropdown in the sidebar to quickly load SFARI risk genes.
        
        ---
        
        ## Citation
        
        If you use this resource in your research, please cite the original data publications
        listed above and this tool:
        
        ```
        SFARI Gene Expression Explorer (2025)
        https://github.com/ar-kie/SFARIExplorer
        ```
        
        ---
        
        ## Contact & Feedback
        
        For questions, bug reports, or feature requests, please open an issue on
        [GitHub](https://github.com/ar-kie/SFARIExplorer/issues).
        """)
    
    # ==========================================================================
    # Footer
    # ==========================================================================
    st.divider()
    st.markdown(f"""
    <div style="text-align: center; color: {'#aaa' if st.session_state.dark_mode else '#666'}; font-size: 0.9rem;">
        SFARI Gene Expression Explorer | Built with Streamlit & Plotly<br>
        Data: Cross-species single-cell RNA-seq atlas | 
        <a href="#" onclick="return false;">Toggle Dark Mode: {'🌙' if not st.session_state.dark_mode else '☀️'}</a>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()