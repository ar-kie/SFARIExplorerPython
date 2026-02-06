"""
SFARI Gene Expression Explorer
A sophisticated cross-species gene expression browser for single-cell RNA-seq data.

Features:
1. Flexible gene search across species, datasets, and cell-types
2. Interactive heatmaps with clustering and faceting options
3. Dynamic temporal expression plots (pseudotime, developmental stages)
4. SFARI risk gene annotations
5. Data overview with batch correction & variance partition visualization
6. UMAP visualization of integrated data
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list
from scipy.spatial.distance import pdist
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

# Custom CSS for a cleaner, more scientific look
st.markdown("""
<style>
    .stApp {
        max-width: 100%;
    }
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    h1 {
        color: #1f4e79;
        font-weight: 600;
    }
    h2, h3 {
        color: #2c5282;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #f0f4f8;
        border-radius: 4px 4px 0 0;
        padding: 8px 16px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2c5282;
        color: white;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 8px;
        color: white;
        text-align: center;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-weight: 600;
    }
    .info-box {
        background-color: #e8f4f8;
        border-left: 4px solid #2c5282;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 8px 8px 0;
    }
    .success-box {
        background-color: #e8f8e8;
        border-left: 4px solid #28a745;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 8px 8px 0;
    }
    .warning-box {
        background-color: #fff8e8;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 8px 8px 0;
    }
</style>
""", unsafe_allow_html=True)


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
        
        # Ensure consistent column names for risk_genes
        if 'gene-symbol' in data['risk_genes'].columns:
            data['risk_genes'] = data['risk_genes'].rename(columns={'gene-symbol': 'gene_symbol'})
        if 'gene-score' in data['risk_genes'].columns:
            data['risk_genes'] = data['risk_genes'].rename(columns={'gene-score': 'gene_score'})
    except Exception as e:
        st.error(f"Error loading core data: {e}")
        return None
    
    # Optional data files (new parquets)
    optional_files = {
        'umap': 'umap_subsample.parquet',
        'vp_summary': 'variance_partition_summary.parquet',
        'vp_by_gene': 'variance_partition_by_gene.parquet',
        'vp_by_gene_wide': 'variance_partition_by_gene_wide.parquet',
        'dataset_info': 'dataset_info.parquet',
        'batch_correction': 'batch_correction_info.parquet',
        'summary_stats': 'summary_statistics.parquet',
    }
    
    for key, filename in optional_files.items():
        try:
            data[key] = pd.read_parquet(f"{data_dir}/{filename}")
        except FileNotFoundError:
            data[key] = None
        except Exception as e:
            st.warning(f"Could not load {filename}: {e}")
            data[key] = None
    
    return data


def get_unique_values(df: pd.DataFrame, column: str) -> List[str]:
    """Get sorted unique values from a column."""
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
    df = expr_df.copy()
    
    if species:
        df = df[df['species'].isin(species)]
    if datasets:
        df = df[df['tissue'].isin(datasets)]
    if cell_types:
        df = df[df['cell_type'].isin(cell_types)]
    if genes:
        # Case-insensitive gene matching
        genes_upper = [g.upper() for g in genes]
        df = df[
            df['gene_native'].str.upper().isin(genes_upper) |
            df['gene_human'].str.upper().isin(genes_upper)
        ]
    
    return df


def parse_gene_input(gene_text: str) -> List[str]:
    """Parse comma/space/newline separated gene names."""
    if not gene_text or not gene_text.strip():
        return []
    # Split on comma, space, newline, semicolon
    genes = re.split(r'[,\s;]+', gene_text.strip())
    return [g.strip() for g in genes if g.strip()]


def create_heatmap_matrix(
    df: pd.DataFrame,
    value_col: str = 'mean_expr',
    scale_rows: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create a matrix suitable for heatmap visualization.
    
    Returns:
        matrix: The (optionally scaled) expression matrix
        col_meta: Metadata for each column (species, tissue/dataset, cell_type)
    """
    df = df.copy()
    
    # Create gene display name (prefer human symbol)
    df['gene_display'] = df['gene_human'].fillna(df['gene_native'])
    
    # Create unique column key from all metadata
    df['col_key'] = df.apply(
        lambda x: f"{x['species']}|{x['tissue']}|{x['cell_type']}", axis=1
    )
    
    # Pivot to wide format
    pivot = df.pivot_table(
        index='gene_display',
        columns='col_key',
        values=value_col,
        aggfunc='mean'
    )
    
    # Row-wise z-score scaling
    if scale_rows and pivot.shape[0] > 0:
        row_means = pivot.mean(axis=1)
        row_stds = pivot.std(axis=1)
        row_stds = row_stds.replace(0, 1)  # Avoid division by zero
        pivot = pivot.sub(row_means, axis=0).div(row_stds, axis=0)
        # Clip to [-3, 3]
        pivot = pivot.clip(-3, 3)
    
    # Build column metadata - parse from col_key
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


def cluster_matrix(
    matrix: pd.DataFrame,
    cluster_rows: bool = True,
    cluster_cols: bool = True
) -> Tuple[pd.DataFrame, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Hierarchically cluster rows and/or columns.
    
    Returns:
        reordered_matrix, row_order, col_order
    """
    row_order = None
    col_order = None
    
    if cluster_rows and matrix.shape[0] > 1:
        # Fill NaN with 0 for clustering
        mat_filled = matrix.fillna(0).values
        if mat_filled.shape[0] > 1:
            try:
                dist = pdist(mat_filled)
                link = linkage(dist, method='average')
                row_order = leaves_list(link)
            except:
                row_order = np.arange(matrix.shape[0])
    
    if cluster_cols and matrix.shape[1] > 1:
        mat_filled = matrix.fillna(0).values.T
        if mat_filled.shape[0] > 1:
            try:
                dist = pdist(mat_filled)
                link = linkage(dist, method='average')
                col_order = leaves_list(link)
            except:
                col_order = np.arange(matrix.shape[1])
    
    # Reorder matrix
    if row_order is not None:
        matrix = matrix.iloc[row_order]
    if col_order is not None:
        matrix = matrix.iloc[:, col_order]
    
    return matrix, row_order, col_order


# =============================================================================
# Visualization Functions
# =============================================================================

# Color palettes for annotations
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
    # Human datasets
    'He (2024)': '#e41a1c',
    'Bhaduri (2021)': '#377eb8',
    'Braun (2023)': '#4daf4a',
    'Velmeshev (2023)': '#984ea3',
    'Velmeshev (2019)': '#ff7f00',
    'Zhu (2023)': '#ffff33',
    'Wang (2025)': '#a65628',
    'Wang (2022)': '#f781bf',
    # Mouse datasets
    'La Manno (2021)': '#66c2a5',
    'Jin (2025)': '#fc8d62',
    'Sziraki (2023)': '#8da0cb',
    # Zebrafish
    'Raj (2020)': '#e78ac3',
    # Drosophila
    'Davie (2018)': '#a6d854',
}

def get_color_palette(values: List[str], palette_type: str = 'auto') -> Dict[str, str]:
    """Generate a color palette for categorical values."""
    if palette_type == 'species':
        return {v: SPECIES_COLORS.get(v, '#999999') for v in values}
    elif palette_type == 'cell_type':
        return {v: CELLTYPE_COLORS.get(v, '#999999') for v in values}
    elif palette_type == 'dataset':
        return {v: DATASET_COLORS.get(v, '#999999') for v in values}
    else:
        # Auto-generate colors using a good categorical palette
        n = len(values)
        if n <= 12:
            preset_colors = [
                '#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33',
                '#a65628', '#f781bf', '#999999', '#66c2a5', '#fc8d62', '#8da0cb'
            ]
            colors = {v: preset_colors[i % len(preset_colors)] for i, v in enumerate(values)}
        else:
            import colorsys
            colors = {}
            for i, v in enumerate(values):
                hue = i / n
                rgb = colorsys.hsv_to_rgb(hue, 0.7, 0.9)
                colors[v] = f'rgb({int(rgb[0]*255)},{int(rgb[1]*255)},{int(rgb[2]*255)})'
        return colors


def create_complexheatmap(
    matrix: pd.DataFrame,
    col_meta: pd.DataFrame,
    title: str = "Gene Expression Heatmap",
    color_scale: str = "RdBu_r",
    split_by: Optional[str] = None,
    annotation_col: Optional[str] = None,
    show_row_dendrogram: bool = False,
    show_col_dendrogram: bool = False,
    cluster_rows: bool = True,
    cluster_cols: bool = True,
    row_label_size: int = 9,
    col_label_size: int = 9,
    gap_between_splits: float = 0.02,
    legend_position: str = "bottom"
) -> go.Figure:
    """
    Create a ComplexHeatmap-like visualization using Plotly subplots.
    """
    from plotly.subplots import make_subplots
    
    if matrix.empty:
        fig = go.Figure()
        fig.add_annotation(text="No data to display", xref="paper", yref="paper",
                          x=0.5, y=0.5, showarrow=False, font=dict(size=20))
        return fig
    
    # Cluster rows if requested (applies to whole matrix)
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
        split_values = col_meta[split_by].unique().tolist()
        split_values = [s for s in split_values if pd.notna(s)]
    else:
        split_values = [None]
    
    n_splits = len(split_values)
    
    # Calculate subplot widths based on number of columns in each split
    split_widths = []
    split_matrices = []
    split_col_metas = []
    
    for split_val in split_values:
        if split_val is not None:
            mask = col_meta[split_by] == split_val
            cols = col_meta[mask].index.tolist()
        else:
            cols = col_meta.index.tolist()
        
        sub_matrix = matrix[[c for c in cols if c in matrix.columns]]
        sub_col_meta = col_meta.loc[sub_matrix.columns]
        
        # Cluster columns within this split
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
    
    # Normalize widths
    total_width = sum(split_widths)
    col_widths = [w / total_width for w in split_widths]
    
    # Determine if we need annotation row
    has_annotation = annotation_col and annotation_col in col_meta.columns
    
    # Create subplot structure
    n_rows = 2 if has_annotation else 1
    row_heights = [0.03, 0.97] if has_annotation else [1.0]
    
    fig = make_subplots(
        rows=n_rows,
        cols=n_splits,
        column_widths=col_widths,
        row_heights=row_heights,
        horizontal_spacing=gap_between_splits,
        vertical_spacing=0.01,
        subplot_titles=[str(s) if s else "" for s in split_values] if n_splits > 1 else None
    )
    
    # Get annotation colors
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
    
    # Track if we've added colorbar
    colorbar_added = False
    
    # Add heatmaps and annotations for each split
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
        
        # Build hover data
        hover_data = []
        for i in range(len(sub_col_meta)):
            hover_data.append({
                'species': sub_col_meta['species'].iloc[i],
                'dataset': sub_col_meta['dataset'].iloc[i],
                'cell_type': sub_col_meta['cell_type'].iloc[i]
            })
        
        # Add annotation bar if requested
        if has_annotation:
            anno_values = sub_col_meta[annotation_col].tolist()
            anno_colors_list = [anno_colors.get(v, '#999999') for v in anno_values]
            
            anno_z = [[i for i in range(len(anno_values))]]
            anno_hover = [[f"{annotation_col.replace('_', ' ').title()}: {v}" for v in anno_values]]
            
            fig.add_trace(
                go.Heatmap(
                    z=anno_z,
                    x=col_labels,
                    colorscale=[[i/max(len(anno_values)-1, 1), anno_colors_list[i]] 
                               for i in range(len(anno_values))],
                    showscale=False,
                    hoverinfo='text',
                    text=anno_hover
                ),
                row=1, col=col_idx
            )
        
        # Add main heatmap
        heatmap_row = 2 if has_annotation else 1
        
        # Build hover text
        hover_text = []
        for gene in sub_matrix.index:
            row_hover = []
            for i, col in enumerate(sub_matrix.columns):
                val = sub_matrix.loc[gene, col]
                h = hover_data[i]
                text = (f"Gene: {gene}<br>"
                       f"Species: {h['species']}<br>"
                       f"Dataset: {h['dataset']}<br>"
                       f"Cell type: {h['cell_type']}<br>"
                       f"Z-score: {val:.2f}" if pd.notna(val) else f"Z-score: N/A")
                row_hover.append(text)
            hover_text.append(row_hover)
        
        fig.add_trace(
            go.Heatmap(
                z=sub_matrix.values,
                x=col_labels,
                y=sub_matrix.index.tolist(),
                colorscale=color_scale,
                zmid=0,
                zmin=-3,
                zmax=3,
                showscale=not colorbar_added,
                colorbar=dict(
                    title=dict(text="Z-score", side="right"),
                    thickness=15,
                    len=0.7,
                    x=1.02
                ) if not colorbar_added else None,
                hoverinfo='text',
                text=hover_text
            ),
            row=heatmap_row, col=col_idx
        )
        colorbar_added = True
    
    # Update layout
    height = max(400, 80 + len(matrix) * 16)
    if has_annotation:
        height += 30
    
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=18)),
        height=height,
        margin=dict(l=150, r=80, t=100, b=120),
        showlegend=False
    )
    
    # Update axes
    for i in range(1, n_splits + 1):
        heatmap_row = 2 if has_annotation else 1
        
        if has_annotation:
            fig.update_xaxes(showticklabels=False, row=1, col=i)
            fig.update_yaxes(showticklabels=False, row=1, col=i)
        
        fig.update_xaxes(
            tickangle=45,
            tickfont=dict(size=col_label_size),
            row=heatmap_row, col=i
        )
        fig.update_yaxes(
            tickfont=dict(size=row_label_size),
            autorange='reversed',
            showticklabels=(i == 1),
            row=heatmap_row, col=i
        )
    
    # Add legend for annotation colors
    if has_annotation:
        for i, (val, color) in enumerate(anno_colors.items()):
            fig.add_trace(
                go.Scatter(
                    x=[None], y=[None],
                    mode='markers',
                    marker=dict(size=10, color=color),
                    name=str(val),
                    showlegend=True
                )
            )
        
        if legend_position == "bottom":
            legend_config = dict(orientation='h', yanchor='top', y=-0.15, xanchor='center', x=0.5)
            fig.update_layout(margin=dict(l=150, r=80, t=100, b=180))
        elif legend_position == "right":
            legend_config = dict(orientation='v', yanchor='middle', y=0.5, xanchor='left', x=1.02)
            fig.update_layout(margin=dict(l=150, r=200, t=100, b=120))
        elif legend_position == "left":
            legend_config = dict(orientation='v', yanchor='middle', y=0.5, xanchor='right', x=-0.15)
            fig.update_layout(margin=dict(l=250, r=80, t=100, b=120))
        else:
            legend_config = dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5)
            fig.update_layout(margin=dict(l=150, r=80, t=160, b=120))
        
        legend_config['title'] = dict(text=annotation_col.replace('_', ' ').title())
        fig.update_layout(legend=legend_config, showlegend=True)
    
    return fig


def create_dotplot(
    df: pd.DataFrame,
    genes: List[str],
    group_by: str = 'cell_type',
    size_col: str = 'pct_expressing',
    color_col: str = 'mean_expr',
    title: str = "Dot Plot"
) -> go.Figure:
    """Create a dot plot (size = % expressing, color = mean expression)."""
    
    df = df.copy()
    df['gene_display'] = df['gene_human'].fillna(df['gene_native'])
    
    agg_df = df.groupby(['gene_display', group_by]).agg({
        size_col: 'mean',
        color_col: 'mean'
    }).reset_index()
    
    agg_df['size_scaled'] = agg_df[size_col] * 30 + 5
    
    fig = px.scatter(
        agg_df,
        x=group_by,
        y='gene_display',
        size='size_scaled',
        color=color_col,
        color_continuous_scale='Viridis',
        title=title,
        labels={
            color_col: 'Mean Expression',
            'gene_display': 'Gene',
            group_by: group_by.replace('_', ' ').title()
        }
    )
    
    fig.update_traces(
        hovertemplate=(
            f"Gene: %{{y}}<br>"
            f"{group_by}: %{{x}}<br>"
            f"Mean expr: %{{marker.color:.3f}}<br>"
            f"% expressing: %{{customdata:.1%}}<extra></extra>"
        ),
        customdata=agg_df[size_col]
    )
    
    fig.update_layout(
        height=max(400, 50 + len(genes) * 25),
        xaxis_tickangle=45,
        yaxis=dict(autorange='reversed')
    )
    
    return fig


def create_temporal_plot(
    df: pd.DataFrame,
    genes: List[str],
    time_col: str = 'tissue',
    group_by: str = 'cell_type',
    value_col: str = 'mean_expr'
) -> go.Figure:
    """Create temporal expression dynamics plot."""
    df = df.copy()
    df['gene_display'] = df['gene_human'].fillna(df['gene_native'])
    
    agg_df = df.groupby(['gene_display', time_col, 'species']).agg({
        value_col: 'mean',
        'pct_expressing': 'mean',
        'n_cells': 'sum'
    }).reset_index()
    
    fig = px.line(
        agg_df,
        x=time_col,
        y=value_col,
        color='gene_display',
        facet_row='species',
        markers=True,
        title="Expression Across Datasets/Timepoints",
        labels={
            value_col: 'Mean Expression',
            time_col: 'Dataset',
            'gene_display': 'Gene'
        }
    )
    
    fig.update_layout(
        height=300 * agg_df['species'].nunique(),
        legend=dict(orientation='h', y=-0.15)
    )
    
    return fig


def create_species_comparison_plot(
    df: pd.DataFrame,
    genes: List[str],
    value_col: str = 'mean_expr'
) -> go.Figure:
    """Create side-by-side species comparison for selected genes."""
    
    df = df.copy()
    df['gene_display'] = df['gene_human'].fillna(df['gene_native'])
    
    agg_df = df.groupby(['gene_display', 'species', 'cell_type']).agg({
        value_col: 'mean',
        'pct_expressing': 'mean'
    }).reset_index()
    
    fig = px.bar(
        agg_df,
        x='cell_type',
        y=value_col,
        color='species',
        facet_row='gene_display',
        barmode='group',
        title="Cross-Species Expression Comparison",
        labels={
            value_col: 'Mean Expression',
            'cell_type': 'Cell Type'
        }
    )
    
    fig.update_layout(
        height=250 * min(len(genes), 6),
        xaxis_tickangle=45,
        legend=dict(orientation='h', y=-0.1)
    )
    
    return fig


# =============================================================================
# New Visualization Functions for Data Overview
# =============================================================================

def create_variance_partition_barplot(vp_summary: pd.DataFrame) -> go.Figure:
    """Create a grouped bar plot showing variance explained before/after correction."""
    
    # Define colors
    colors = {
        'Before Correction': '#ff7f0e',
        'After Correction': '#1f77b4'
    }
    
    fig = go.Figure()
    
    for stage in ['Before Correction', 'After Correction']:
        stage_data = vp_summary[vp_summary['stage'] == stage]
        fig.add_trace(go.Bar(
            name=stage,
            x=stage_data['variable'],
            y=stage_data['variance_explained'],
            marker_color=colors[stage],
            text=[f"{v:.1f}%" for v in stage_data['variance_explained']],
            textposition='outside'
        ))
    
    fig.update_layout(
        title=dict(
            text="Variance Explained by Each Factor",
            x=0.5,
            font=dict(size=18)
        ),
        xaxis_title="Variable",
        yaxis_title="Variance Explained (%)",
        barmode='group',
        height=450,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='center',
            x=0.5
        ),
        yaxis=dict(range=[0, max(vp_summary['variance_explained']) * 1.15])
    )
    
    # Add annotation for goal
    fig.add_annotation(
        text="Goal: ↓ dataset (batch), ↑ or = biological factors",
        xref="paper", yref="paper",
        x=0.5, y=-0.15,
        showarrow=False,
        font=dict(size=11, color='gray')
    )
    
    return fig


def create_variance_change_plot(vp_summary: pd.DataFrame) -> go.Figure:
    """Create a waterfall-style plot showing the change in variance."""
    
    # Get unique variables and their changes
    vars_df = vp_summary.groupby('variable').first().reset_index()
    
    if 'change' not in vars_df.columns:
        # Calculate change
        before = vp_summary[vp_summary['stage'] == 'Before Correction'].set_index('variable')['variance_explained']
        after = vp_summary[vp_summary['stage'] == 'After Correction'].set_index('variable')['variance_explained']
        vars_df = pd.DataFrame({
            'variable': before.index,
            'change': after.values - before.values
        })
    
    # Color by whether change is good or bad
    colors = []
    for _, row in vars_df.iterrows():
        var = row['variable']
        change = row['change']
        if var == 'dataset':
            # For dataset (batch), decrease is good
            colors.append('#28a745' if change < 0 else '#dc3545')
        elif var == 'Residuals':
            # Residuals increase is neutral
            colors.append('#6c757d')
        else:
            # For biological factors, increase is good
            colors.append('#28a745' if change >= 0 else '#dc3545')
    
    fig = go.Figure(go.Bar(
        x=vars_df['variable'],
        y=vars_df['change'],
        marker_color=colors,
        text=[f"{c:+.1f}%" for c in vars_df['change']],
        textposition='outside'
    ))
    
    fig.update_layout(
        title=dict(
            text="Change in Variance Explained After Batch Correction",
            x=0.5,
            font=dict(size=18)
        ),
        xaxis_title="Variable",
        yaxis_title="Change in Variance (%)",
        height=400,
        yaxis=dict(zeroline=True, zerolinewidth=2, zerolinecolor='black')
    )
    
    return fig


def create_umap_plot(
    umap_df: pd.DataFrame,
    color_by: str = 'predicted_labels',
    title: str = "UMAP Visualization"
) -> go.Figure:
    """Create an interactive UMAP scatter plot."""
    
    # Determine color column
    if color_by not in umap_df.columns:
        color_by = umap_df.columns[2] if len(umap_df.columns) > 2 else None
    
    # Get color palette
    if color_by:
        unique_vals = umap_df[color_by].unique().tolist()
        if color_by == 'organism':
            color_map = SPECIES_COLORS
        elif color_by in ['predicted_labels', 'cell_type']:
            color_map = CELLTYPE_COLORS
        elif color_by == 'dataset':
            color_map = DATASET_COLORS
        else:
            color_map = get_color_palette(unique_vals, 'auto')
    
    fig = px.scatter(
        umap_df,
        x='umap_1' if 'umap_1' in umap_df.columns else umap_df.columns[0],
        y='umap_2' if 'umap_2' in umap_df.columns else umap_df.columns[1],
        color=color_by,
        color_discrete_map=color_map if color_by else None,
        title=title,
        labels={
            'umap_1': 'UMAP 1',
            'umap_2': 'UMAP 2'
        },
        hover_data=[c for c in ['organism', 'dataset', 'predicted_labels'] if c in umap_df.columns]
    )
    
    fig.update_traces(marker=dict(size=3, opacity=0.6))
    
    fig.update_layout(
        height=600,
        legend=dict(
            orientation='v',
            yanchor='top',
            y=1,
            xanchor='left',
            x=1.02,
            font=dict(size=10)
        )
    )
    
    return fig


def create_dataset_summary_table(dataset_info: pd.DataFrame) -> pd.DataFrame:
    """Format dataset info for display."""
    display_cols = ['dataset', 'organism', 'sample_type', 'n_pseudobulk_samples', 
                    'n_cell_types', 'n_timepoints']
    available_cols = [c for c in display_cols if c in dataset_info.columns]
    
    df = dataset_info[available_cols].copy()
    
    # Rename for display
    rename_map = {
        'dataset': 'Dataset',
        'organism': 'Species',
        'sample_type': 'Sample Type',
        'n_pseudobulk_samples': 'Pseudobulk Samples',
        'n_cell_types': 'Cell Types',
        'n_timepoints': 'Timepoints'
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    
    return df


# =============================================================================
# Main App
# =============================================================================

def main():
    # Header
    st.title("🧬 SFARI Gene Expression Explorer")
    st.markdown("""
    *A cross-species single-cell RNA-seq browser for neurodevelopmental gene expression*
    """)
    
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
        
        # Cell type selection
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
        
        # Quick gene sets
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
        
        # Get preset genes
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
            else:
                synd_genes = []
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
        st.metric("Genes", filtered_df['gene_human'].nunique())
    with col2:
        st.metric("Species", filtered_df['species'].nunique())
    with col3:
        st.metric("Datasets", filtered_df['tissue'].nunique())
    with col4:
        st.metric("Cell Types", filtered_df['cell_type'].nunique())
    
    # Tabs for different views - UPDATED with new tabs
    tab_overview, tab_umap, tab_heatmap, tab_dotplot, tab_temporal, tab_species, tab_table, tab_about = st.tabs([
        "📊 Data Overview",
        "🗺️ UMAP",
        "🔥 Heatmap", 
        "🔵 Dot Plot", 
        "📈 Temporal",
        "🔬 Species Comparison",
        "📋 Data Table",
        "📚 About"
    ])
    
    # --------------------------------------------------------------------------
    # Tab 0: Data Overview (NEW)
    # --------------------------------------------------------------------------
    with tab_overview:
        st.header("Data Overview & Batch Correction")
        
        st.markdown("""
        This section provides an overview of the integrated single-cell dataset, 
        including batch correction methodology and quality metrics.
        """)
        
        # Summary statistics
        if data['summary_stats'] is not None:
            st.subheader("📈 Dataset Summary")
            
            summary = data['summary_stats']
            totals = summary[summary['category'] == 'totals'].set_index('label')['value']
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                if 'pseudobulk_samples' in totals.index:
                    st.metric("Pseudobulk Samples", f"{int(totals['pseudobulk_samples']):,}")
            with col2:
                if 'datasets' in totals.index:
                    st.metric("Datasets", int(totals['datasets']))
            with col3:
                if 'cell_types' in totals.index:
                    st.metric("Cell Types", int(totals['cell_types']))
            with col4:
                if 'organisms' in totals.index:
                    st.metric("Species", int(totals['organisms']))
        
        # Dataset information table
        if data['dataset_info'] is not None:
            st.subheader("📋 Datasets Included")
            
            dataset_table = create_dataset_summary_table(data['dataset_info'])
            st.dataframe(dataset_table, hide_index=True, use_container_width=True)
        
        st.divider()
        
        # Batch correction methodology
        st.subheader("🔧 Batch Correction Methodology")
        
        if data['batch_correction'] is not None:
            bc_info = data['batch_correction']
            
            st.markdown("""
            <div class="info-box">
            <strong>Approach:</strong> Within-organism batch correction using ComBat, 
            preserving biological covariates (cell type, developmental time) while 
            removing technical batch effects (dataset).
            </div>
            """, unsafe_allow_html=True)
            
            # Show correction groups
            st.markdown("**Correction Groups:**")
            
            for _, row in bc_info.iterrows():
                if row.get('correction_applied', False):
                    st.markdown(f"""
                    - **{row['correction_group']}**: {row['n_datasets']} datasets corrected using {row['method']}
                      - Covariates preserved: {row.get('covariates_preserved', 'cell_type')}
                      - Datasets: {row['datasets']}
                    """)
                else:
                    st.markdown(f"""
                    - **{row['correction_group']}**: Single dataset (no correction needed)
                    """)
        else:
            st.info("Batch correction information not available.")
        
        st.divider()
        
        # Variance partition results
        st.subheader("📊 Variance Partition Analysis")
        
        if data['vp_summary'] is not None:
            st.markdown("""
            Variance partition analysis quantifies how much of the gene expression 
            variance is explained by each factor. **Successful batch correction** should:
            - **Decrease** variance explained by `dataset` (technical/batch effect)
            - **Preserve or increase** variance explained by biological factors (cell_type, organism, timepoint)
            """)
            
            vp_col1, vp_col2 = st.columns(2)
            
            with vp_col1:
                fig_vp = create_variance_partition_barplot(data['vp_summary'])
                st.plotly_chart(fig_vp, use_container_width=True)
            
            with vp_col2:
                fig_change = create_variance_change_plot(data['vp_summary'])
                st.plotly_chart(fig_change, use_container_width=True)
            
            # Interpretation
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
                    st.markdown(f"""
                    <div class="warning-box">
                    ⚠️ <strong>Note:</strong> Dataset variance increased slightly. This may indicate 
                    confounding between batch and biological factors.
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("Variance partition data not available.")
        
        # Per-gene variance partition
        if data['vp_by_gene'] is not None:
            with st.expander("🔬 Per-Gene Variance Breakdown"):
                st.markdown("""
                Explore variance partition results for individual genes. 
                This shows how much of each gene's expression variance is explained by different factors.
                """)
                
                vp_genes = data['vp_by_gene']
                
                # Gene search
                gene_search = st.text_input("Search for a gene:", placeholder="e.g., SHANK3")
                
                if gene_search:
                    gene_data = vp_genes[vp_genes['gene'].str.upper() == gene_search.upper()]
                    
                    if not gene_data.empty:
                        fig_gene = px.bar(
                            gene_data,
                            x='variable',
                            y='variance_explained',
                            color='variable',
                            title=f"Variance Partition for {gene_search.upper()}",
                            labels={'variance_explained': 'Variance Explained (%)', 'variable': 'Factor'}
                        )
                        fig_gene.update_layout(showlegend=False, height=350)
                        st.plotly_chart(fig_gene, use_container_width=True)
                    else:
                        st.warning(f"Gene '{gene_search}' not found in variance partition results.")
    
    # --------------------------------------------------------------------------
    # Tab 1: UMAP (NEW)
    # --------------------------------------------------------------------------
    with tab_umap:
        st.header("UMAP Visualization")
        
        if data['umap'] is not None:
            umap_df = data['umap']
            
            st.markdown(f"""
            Interactive UMAP visualization of **{len(umap_df):,} cells** subsampled from the 
            integrated dataset. Cells are colored by various metadata attributes.
            """)
            
            # Color options
            available_color_cols = [c for c in umap_df.columns if c not in ['umap_1', 'umap_2', 'cell_id']]
            
            col1, col2 = st.columns([1, 3])
            
            with col1:
                color_by = st.selectbox(
                    "Color by:",
                    options=available_color_cols,
                    index=available_color_cols.index('predicted_labels') if 'predicted_labels' in available_color_cols else 0
                )
                
                # Show legend toggle
                show_legend = st.checkbox("Show legend", value=True)
                
                # Point size
                point_size = st.slider("Point size", 1, 10, 3)
                
                # Opacity
                opacity = st.slider("Opacity", 0.1, 1.0, 0.6)
            
            with col2:
                # Create UMAP plot
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
                    legend=dict(
                        orientation='v',
                        yanchor='top',
                        y=1,
                        xanchor='left',
                        x=1.02,
                        font=dict(size=9)
                    )
                )
                
                st.plotly_chart(fig_umap, use_container_width=True)
            
            # Download UMAP coordinates
            with st.expander("📥 Download UMAP Data"):
                csv = umap_df.to_csv(index=False)
                st.download_button(
                    "Download UMAP coordinates as CSV",
                    csv,
                    "umap_coordinates.csv",
                    "text/csv"
                )
        else:
            st.info("""
            UMAP visualization data not available. 
            
            To enable this feature, generate a `umap_subsample.parquet` file containing:
            - `umap_1`, `umap_2`: UMAP coordinates
            - `organism`, `dataset`, `predicted_labels`: Metadata columns
            """)
    
    # --------------------------------------------------------------------------
    # Tab 2: Heatmap
    # --------------------------------------------------------------------------
    with tab_heatmap:
        if filtered_df.empty:
            st.warning("No data matches your filters. Try broadening your selection.")
        elif not selected_genes:
            st.info("Enter gene names in the sidebar to generate a heatmap.")
        else:
            # Heatmap options
            st.markdown("**Heatmap Settings**")
            hm_col1, hm_col2, hm_col3, hm_col4 = st.columns(4)
            
            with hm_col1:
                split_option = st.selectbox(
                    "Split heatmap by",
                    options=["None", "Species", "Dataset", "Cell Type"],
                    index=0
                )
                split_by = {"None": None, "Species": "species", "Dataset": "dataset", "Cell Type": "cell_type"}[split_option]
            
            with hm_col2:
                annotation_option = st.selectbox(
                    "Top annotation",
                    options=["None", "Species", "Dataset", "Cell Type"],
                    index=0
                )
                annotation_col = {"None": None, "Species": "species", "Dataset": "dataset", "Cell Type": "cell_type"}[annotation_option]
            
            with hm_col3:
                color_scale = st.selectbox(
                    "Color scale",
                    options=["RdBu_r", "Viridis", "Plasma", "Inferno", "Blues", "RdYlBu_r", "PiYG"],
                    index=0
                )
            
            with hm_col4:
                legend_pos = st.selectbox(
                    "Legend position",
                    options=["Bottom", "Right", "Left", "Top"],
                    index=0
                )
            
            hm_col5, hm_col6, hm_col7, hm_col8 = st.columns(4)
            
            with hm_col5:
                do_cluster_rows = st.checkbox("Cluster rows", value=cluster_rows)
            with hm_col6:
                do_cluster_cols = st.checkbox("Cluster columns", value=cluster_cols)
            with hm_col7:
                row_font = st.slider("Gene label size", 6, 14, 9)
            with hm_col8:
                col_font = st.slider("Column label size", 6, 14, 9)
            
            # Create and display heatmap
            matrix, col_meta = create_heatmap_matrix(filtered_df, value_col=value_metric, scale_rows=scale_rows)
            
            fig = create_complexheatmap(
                matrix=matrix,
                col_meta=col_meta,
                title=f"Expression Heatmap ({len(selected_genes)} genes)",
                color_scale=color_scale,
                split_by=split_by,
                annotation_col=annotation_col,
                cluster_rows=do_cluster_rows,
                cluster_cols=do_cluster_cols,
                row_label_size=row_font,
                col_label_size=col_font,
                legend_position=legend_pos.lower()
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            with st.expander("📥 Download Matrix"):
                csv = matrix.to_csv()
                st.download_button("Download as CSV", csv, "expression_matrix.csv", "text/csv")
    
    # --------------------------------------------------------------------------
    # Tab 3: Dot Plot
    # --------------------------------------------------------------------------
    with tab_dotplot:
        if filtered_df.empty or not selected_genes:
            st.info("Select filters and genes to generate a dot plot.")
        else:
            dot_group = st.selectbox(
                "Group by",
                options=['cell_type', 'tissue', 'species'],
                format_func=lambda x: x.replace('_', ' ').title()
            )
            
            fig = create_dotplot(
                filtered_df,
                selected_genes,
                group_by=dot_group,
                title="Dot Plot: Size = % Expressing, Color = Mean Expression"
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # --------------------------------------------------------------------------
    # Tab 4: Temporal Dynamics
    # --------------------------------------------------------------------------
    with tab_temporal:
        st.markdown("""
        **Note:** Temporal dynamics visualization requires developmental timepoint annotations.
        Currently showing expression across datasets as a proxy.
        """)
        
        if filtered_df.empty or not selected_genes:
            st.info("Select filters and genes to visualize temporal dynamics.")
        else:
            fig = create_temporal_plot(filtered_df, selected_genes, time_col='tissue', value_col=value_metric)
            st.plotly_chart(fig, use_container_width=True)
    
    # --------------------------------------------------------------------------
    # Tab 5: Species Comparison
    # --------------------------------------------------------------------------
    with tab_species:
        if filtered_df.empty or not selected_genes:
            st.info("Select filters and genes to compare across species.")
        elif filtered_df['species'].nunique() < 2:
            st.warning("Select multiple species to enable cross-species comparison.")
        else:
            fig = create_species_comparison_plot(filtered_df, selected_genes[:6], value_col=value_metric)
            st.plotly_chart(fig, use_container_width=True)
    
    # --------------------------------------------------------------------------
    # Tab 6: Data Table
    # --------------------------------------------------------------------------
    with tab_table:
        if filtered_df.empty:
            st.info("No data matches your current filters.")
        else:
            display_df = filtered_df.copy()
            display_df['gene_display'] = display_df['gene_human'].fillna(display_df['gene_native'])
            
            if 'gene_symbol' in risk_genes.columns:
                risk_lookup = risk_genes.set_index('gene_symbol')['gene_score'].to_dict()
                display_df['SFARI_score'] = display_df['gene_display'].map(risk_lookup)
            
            available_cols = display_df.columns.tolist()
            default_cols = ['gene_display', 'species', 'tissue', 'cell_type', 'mean_expr', 'pct_expressing', 'n_cells']
            default_cols = [c for c in default_cols if c in available_cols]
            
            selected_cols = st.multiselect("Columns", options=available_cols, default=default_cols, label_visibility="collapsed")
            
            if selected_cols:
                st.dataframe(display_df[selected_cols].sort_values(['gene_display', 'species', 'tissue']), height=500, use_container_width=True)
                csv = display_df[selected_cols].to_csv(index=False)
                st.download_button("📥 Download Table as CSV", csv, "filtered_expression_data.csv", "text/csv")
    
    # --------------------------------------------------------------------------
    # Tab 7: About & Datasets
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
        - **Identify** cell-type-specific expression of neurodevelopmental risk genes
        
        ---
        
        ## Datasets
        
        The following single-cell RNA-seq datasets are included:
        
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
        | **Sziraki (2023)** | Mouse brain cell types (aging) |
        
        ### Other Species
        
        | Dataset | Species | Description |
        |---------|---------|-------------|
        | **Raj (2020)** | Zebrafish | Zebrafish brain development |
        | **Davie (2018)** | Drosophila | *Drosophila* brain cell atlas |
        
        ---
        
        ## Data Processing
        
        1. **Integration**: Datasets were integrated using scVI/scANVI
        2. **Cell type harmonization**: Unified cell type labels across datasets
        3. **Batch correction**: Within-organism ComBat correction preserving biological covariates
        4. **Pseudobulk aggregation**: Expression summarized per cell type per sample
        
        ---
        
        ## SFARI Gene Integration
        
        This explorer integrates the [SFARI Gene database](https://gene.sfari.org/), which catalogs 
        genes implicated in autism spectrum disorder (ASD).
        
        ---
        
        ## Citation
        
        If you use this resource in your research, please cite the original data publications 
        and this tool.
        """)
    
    # ==========================================================================
    # Footer
    # ==========================================================================
    st.divider()
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9rem;">
        SFARI Gene Expression Explorer | Built with Streamlit & Plotly<br>
        Data: Cross-species single-cell RNA-seq atlas
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()