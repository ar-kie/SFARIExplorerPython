"""
SFARI Gene Expression Explorer
A sophisticated cross-species gene expression browser for single-cell RNA-seq data.

Features:
1. Flexible gene search across species, datasets, and cell-types
2. Interactive heatmaps with clustering and faceting options
3. Dynamic temporal expression plots (pseudotime, developmental stages)
4. SFARI risk gene annotations
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
</style>
""", unsafe_allow_html=True)


# =============================================================================
# Data Loading
# =============================================================================

@st.cache_data(ttl=3600)
def load_data(data_dir: str = "data") -> Dict[str, pd.DataFrame]:
    """Load all parquet data files with caching."""
    try:
        expr_df = pd.read_parquet(f"{data_dir}/expression_summaries.parquet")
        cellmeta = pd.read_parquet(f"{data_dir}/celltype_meta.parquet")
        gene_map = pd.read_parquet(f"{data_dir}/gene_map.parquet")
        risk_genes = pd.read_parquet(f"{data_dir}/risk_genes.parquet")
        
        # Ensure consistent column names
        if 'gene-symbol' in risk_genes.columns:
            risk_genes = risk_genes.rename(columns={'gene-symbol': 'gene_symbol'})
        if 'gene-score' in risk_genes.columns:
            risk_genes = risk_genes.rename(columns={'gene-score': 'gene_score'})
            
        return {
            'expression': expr_df,
            'cellmeta': cellmeta,
            'gene_map': gene_map,
            'risk_genes': risk_genes
        }
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None


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
    # Alternative names (in case data uses different naming)
    'Linnarsson': '#66c2a5',
    'Zeng': '#fc8d62',
    'Raj': '#e78ac3',
    'Aerts': '#a6d854',
    'Cao': '#ffd92f',
    'Velmeshev': '#984ea3',
    'Bhaduri': '#377eb8',
    'Linnarsson_2023': '#8da0cb',
    'Velmeshev_2023': '#984ea3',
    'Zhu': '#ffff33',
    'Wang_2025': '#a65628',
    'Wang_2022': '#f781bf',
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
        import colorsys
        n = len(values)
        # Use a predefined palette for small n, generate for large n
        if n <= 12:
            preset_colors = [
                '#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33',
                '#a65628', '#f781bf', '#999999', '#66c2a5', '#fc8d62', '#8da0cb'
            ]
            colors = {v: preset_colors[i % len(preset_colors)] for i, v in enumerate(values)}
        else:
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
    legend_position: str = "bottom"  # "bottom", "right", "left", "top"
) -> go.Figure:
    """
    Create a ComplexHeatmap-like visualization using Plotly subplots.
    
    Args:
        matrix: Expression matrix (genes x samples)
        col_meta: Column metadata with 'species', 'dataset', 'cell_type'
        title: Plot title
        color_scale: Plotly colorscale name
        split_by: Column in col_meta to split heatmap by ('species', 'dataset', 'cell_type')
        annotation_col: Column in col_meta for top color annotation
        show_row_dendrogram: Show row dendrogram
        show_col_dendrogram: Show column dendrogram
        cluster_rows: Cluster rows hierarchically
        cluster_cols: Cluster columns (within splits if split_by is set)
        row_label_size: Font size for gene labels
        col_label_size: Font size for sample labels
        gap_between_splits: Gap between split panels (0-0.1)
        legend_position: Position of annotation legend ("bottom", "right", "left", "top")
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
            from scipy.cluster.hierarchy import linkage, leaves_list
            from scipy.spatial.distance import pdist
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
        
        # Create column labels based on what's NOT being split
        # If split by species: show "Dataset | Cell Type"
        # If split by dataset: show "Species | Cell Type" 
        # If split by cell_type: show "Species | Dataset"
        # If no split: show "Dataset | Cell Type"
        if split_by == 'species':
            col_labels = [f"{d}\n{c}" for d, c in zip(sub_col_meta['dataset'], sub_col_meta['cell_type'])]
        elif split_by == 'dataset':
            col_labels = [f"{s}\n{c}" for s, c in zip(sub_col_meta['species'], sub_col_meta['cell_type'])]
        elif split_by == 'cell_type':
            col_labels = [f"{s}\n{d}" for s, d in zip(sub_col_meta['species'], sub_col_meta['dataset'])]
        else:
            col_labels = [f"{d}\n{c}" for d, c in zip(sub_col_meta['dataset'], sub_col_meta['cell_type'])]
        
        # Build custom hover data with all metadata
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
            
            # Create annotation heatmap (just colored bars)
            anno_z = [[i for i in range(len(anno_values))]]
            
            # Build hover text for annotation
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
        
        # Build full hover text matrix
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
        
        # Hide annotation row axes
        if has_annotation:
            fig.update_xaxes(showticklabels=False, row=1, col=i)
            fig.update_yaxes(showticklabels=False, row=1, col=i)
        
        # Style heatmap axes
        fig.update_xaxes(
            tickangle=45,
            tickfont=dict(size=col_label_size),
            row=heatmap_row, col=i
        )
        fig.update_yaxes(
            tickfont=dict(size=row_label_size),
            autorange='reversed',
            showticklabels=(i == 1),  # Only show gene labels on first split
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
        
        # Configure legend position
        if legend_position == "bottom":
            legend_config = dict(
                orientation='h',
                yanchor='top',
                y=-0.15,
                xanchor='center',
                x=0.5
            )
            # Adjust bottom margin
            fig.update_layout(margin=dict(l=150, r=80, t=100, b=180))
        elif legend_position == "right":
            legend_config = dict(
                orientation='v',
                yanchor='middle',
                y=0.5,
                xanchor='left',
                x=1.02
            )
            # Adjust right margin
            fig.update_layout(margin=dict(l=150, r=200, t=100, b=120))
        elif legend_position == "left":
            legend_config = dict(
                orientation='v',
                yanchor='middle',
                y=0.5,
                xanchor='right',
                x=-0.15
            )
            # Adjust left margin
            fig.update_layout(margin=dict(l=250, r=80, t=100, b=120))
        elif legend_position == "top":
            legend_config = dict(
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='center',
                x=0.5
            )
            # Adjust top margin
            fig.update_layout(margin=dict(l=150, r=80, t=160, b=120))
        else:  # default to bottom
            legend_config = dict(
                orientation='h',
                yanchor='top',
                y=-0.15,
                xanchor='center',
                x=0.5
            )
        
        legend_config['title'] = dict(text=annotation_col.replace('_', ' ').title())
        
        fig.update_layout(
            legend=legend_config,
            showlegend=True
        )
    
    return fig


def create_heatmap(
    matrix: pd.DataFrame,
    col_meta: pd.DataFrame,
    title: str = "Gene Expression Heatmap",
    color_scale: str = "RdBu_r",
    facet_by: Optional[str] = None,
    height: int = 600,
    show_dendrograms: bool = False
) -> go.Figure:
    """Legacy wrapper - redirects to create_complexheatmap."""
    return create_complexheatmap(
        matrix=matrix,
        col_meta=col_meta,
        title=title,
        color_scale=color_scale,
        split_by=facet_by,
        annotation_col=facet_by,
        cluster_rows=True,
        cluster_cols=True
    )


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
    
    # Aggregate by gene and group
    agg_df = df.groupby(['gene_display', group_by]).agg({
        size_col: 'mean',
        color_col: 'mean'
    }).reset_index()
    
    # Scale size
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
    time_col: str = 'tissue',  # Placeholder - will use dataset as proxy
    group_by: str = 'cell_type',
    value_col: str = 'mean_expr'
) -> go.Figure:
    """
    Create temporal expression dynamics plot.
    
    Note: This is a placeholder that uses datasets as a proxy for time.
    For real temporal analysis, you'll need actual timepoint data.
    """
    df = df.copy()
    df['gene_display'] = df['gene_human'].fillna(df['gene_native'])
    
    # Aggregate
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
    
    # Aggregate by gene, species, cell_type
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
            synd_genes = risk_genes[risk_genes['syndromic'] == 1]['gene_symbol'].dropna().tolist()[:50]
            preset_genes = ", ".join(synd_genes)
        elif gene_preset == "Top Variable Genes":
            # Calculate variance across all data
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
    
    # Tabs for different views
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Heatmap", 
        "🔵 Dot Plot", 
        "📈 Temporal Dynamics",
        "🔬 Species Comparison",
        "📋 Data Table",
        "📚 About & Datasets"
    ])
    
    # --------------------------------------------------------------------------
    # Tab 1: Heatmap
    # --------------------------------------------------------------------------
    with tab1:
        if filtered_df.empty:
            st.warning("No data matches your filters. Try broadening your selection.")
        elif not selected_genes:
            st.info("Enter gene names in the sidebar to generate a heatmap.")
        else:
            # Heatmap options - Row 1
            st.markdown("**Heatmap Settings**")
            hm_col1, hm_col2, hm_col3, hm_col4 = st.columns(4)
            
            with hm_col1:
                split_option = st.selectbox(
                    "Split heatmap by",
                    options=["None", "Species", "Dataset", "Cell Type"],
                    index=0,
                    help="Divide heatmap into panels by this variable"
                )
                split_by = {
                    "None": None,
                    "Species": "species",
                    "Dataset": "dataset",
                    "Cell Type": "cell_type"
                }[split_option]
            
            with hm_col2:
                annotation_option = st.selectbox(
                    "Top annotation",
                    options=["None", "Species", "Dataset", "Cell Type"],
                    index=0,
                    help="Add colored annotation bar at top"
                )
                annotation_col = {
                    "None": None,
                    "Species": "species",
                    "Dataset": "dataset",
                    "Cell Type": "cell_type"
                }[annotation_option]
            
            with hm_col3:
                color_scale = st.selectbox(
                    "Color scale",
                    options=["RdBu_r", "Viridis", "Plasma", "Inferno", "Blues", "RdYlBu_r", "PiYG"],
                    index=0
                )
            
            with hm_col4:
                show_row_dend = st.checkbox("Show row dendrogram", value=False, disabled=True, 
                                           help="Coming soon")
                show_col_dend = st.checkbox("Show column dendrogram", value=False, disabled=True,
                                           help="Coming soon")
            
            # Row 2: Clustering options
            hm_col5, hm_col6, hm_col7, hm_col8 = st.columns(4)
            
            with hm_col5:
                do_cluster_rows = st.checkbox("Cluster rows (genes)", value=cluster_rows)
            
            with hm_col6:
                do_cluster_cols = st.checkbox("Cluster columns", value=cluster_cols)
            
            with hm_col7:
                row_font = st.slider("Gene label size", 6, 14, 9)
            
            with hm_col8:
                col_font = st.slider("Column label size", 6, 14, 9)
            
            # Row 3: Legend position
            hm_col9, hm_col10, hm_col11, hm_col12 = st.columns(4)
            
            with hm_col9:
                legend_pos = st.selectbox(
                    "Legend position",
                    options=["Bottom", "Right", "Left", "Top"],
                    index=0,
                    help="Position of the annotation legend"
                )
                legend_position = legend_pos.lower()
            
            # Create heatmap matrix
            matrix, col_meta = create_heatmap_matrix(
                filtered_df,
                value_col=value_metric,
                scale_rows=scale_rows
            )
            
            # Create and display heatmap using new ComplexHeatmap-like function
            fig = create_complexheatmap(
                matrix=matrix,
                col_meta=col_meta,
                title=f"Expression Heatmap ({len(selected_genes)} genes)",
                color_scale=color_scale,
                split_by=split_by,
                annotation_col=annotation_col,
                show_row_dendrogram=show_row_dend,
                show_col_dendrogram=show_col_dend,
                cluster_rows=do_cluster_rows,
                cluster_cols=do_cluster_cols,
                row_label_size=row_font,
                col_label_size=col_font,
                legend_position=legend_position
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Download matrix
            with st.expander("📥 Download Matrix"):
                csv = matrix.to_csv()
                st.download_button(
                    "Download as CSV",
                    csv,
                    "expression_matrix.csv",
                    "text/csv"
                )
    
    # --------------------------------------------------------------------------
    # Tab 2: Dot Plot
    # --------------------------------------------------------------------------
    with tab2:
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
    # Tab 3: Temporal Dynamics
    # --------------------------------------------------------------------------
    with tab3:
        st.markdown("""
        **Note:** Temporal dynamics visualization requires developmental timepoint annotations.
        Currently showing expression across datasets as a proxy.
        
        *For full temporal analysis, add `timepoint`, `pseudotime`, or `developmental_stage` 
        columns to your data.*
        """)
        
        if filtered_df.empty or not selected_genes:
            st.info("Select filters and genes to visualize temporal dynamics.")
        else:
            fig = create_temporal_plot(
                filtered_df,
                selected_genes,
                time_col='tissue',
                value_col=value_metric
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # --------------------------------------------------------------------------
    # Tab 4: Species Comparison
    # --------------------------------------------------------------------------
    with tab4:
        if filtered_df.empty or not selected_genes:
            st.info("Select filters and genes to compare across species.")
        elif filtered_df['species'].nunique() < 2:
            st.warning("Select multiple species to enable cross-species comparison.")
        else:
            fig = create_species_comparison_plot(
                filtered_df,
                selected_genes[:6],  # Limit to 6 genes for readability
                value_col=value_metric
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # --------------------------------------------------------------------------
    # Tab 5: Data Table
    # --------------------------------------------------------------------------
    with tab5:
        if filtered_df.empty:
            st.info("No data matches your current filters.")
        else:
            # Merge with risk gene info
            display_df = filtered_df.copy()
            display_df['gene_display'] = display_df['gene_human'].fillna(display_df['gene_native'])
            
            # Add SFARI score if available
            if 'gene_symbol' in risk_genes.columns:
                risk_lookup = risk_genes.set_index('gene_symbol')['gene_score'].to_dict()
                display_df['SFARI_score'] = display_df['gene_display'].map(risk_lookup)
            
            # Column selection
            st.markdown("**Select columns to display:**")
            available_cols = display_df.columns.tolist()
            default_cols = ['gene_display', 'species', 'tissue', 'cell_type', 'mean_expr', 'pct_expressing', 'n_cells']
            default_cols = [c for c in default_cols if c in available_cols]
            
            selected_cols = st.multiselect(
                "Columns",
                options=available_cols,
                default=default_cols,
                label_visibility="collapsed"
            )
            
            if selected_cols:
                st.dataframe(
                    display_df[selected_cols].sort_values(['gene_display', 'species', 'tissue']),
                    height=500,
                    use_container_width=True
                )
                
                # Download
                csv = display_df[selected_cols].to_csv(index=False)
                st.download_button(
                    "📥 Download Table as CSV",
                    csv,
                    "filtered_expression_data.csv",
                    "text/csv"
                )
    
    # --------------------------------------------------------------------------
    # Tab 6: About & Datasets
    # --------------------------------------------------------------------------
    with tab6:
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
        
        The following single-cell RNA-seq datasets are included in this resource:
        
        ### Human Datasets
        
        | Dataset | Publication | Journal | Species | Description |
        |---------|-------------|---------|---------|-------------|
        | **He (2024)** | He et al., 2024 | *Nature* | Human | Human Neural Organoid Cell Atlas (HNOCA) |
        | **Bhaduri (2021)** | Bhaduri et al., 2021 | *Nature* | Human | Primary human cortical development |
        | **Braun (2023)** | Braun et al., 2023 | *Science* | Human | Human brain cell atlas |
        | **Velmeshev (2023)** | Velmeshev et al., 2023 | *Science* | Human | Developing human brain cell types |
        | **Velmeshev (2019)** | Velmeshev et al., 2019 | *Science* | Human | Single-cell genomics of ASD brain |
        | **Zhu (2023)** | Zhu et al., 2023 | *Science Advances* | Human | Human fetal brain development |
        | **Wang (2025)** | Wang et al., 2025 | *Nature* | Human | Human brain development atlas |
        | **Wang (2022)** | Wang et al., 2022 | *Nature Communications* | Human | Human cerebral organoids |
        
        ### Mouse Datasets
        
        | Dataset | Publication | Journal | Species | Description |
        |---------|-------------|---------|---------|-------------|
        | **La Manno (2021)** | La Manno et al., 2021 | *Nature* | Mouse | Mouse brain development atlas |
        | **Jin (2025)** | Jin et al., 2025 | *Nature* | Mouse | Mouse brain cell atlas |
        | **Sziraki (2023)** | Sziraki et al., 2023 | *Nature Genetics* | Mouse | Mouse brain cell types |
        
        ### Zebrafish Datasets
        
        | Dataset | Publication | Journal | Species | Description |
        |---------|-------------|---------|---------|-------------|
        | **Raj (2020)** | Raj et al., 2020 | *Neuroscience* | Zebrafish | Zebrafish brain development |
        
        ### Drosophila Datasets
        
        | Dataset | Publication | Journal | Species | Description |
        |---------|-------------|---------|---------|-------------|
        | **Davie (2018)** | Davie et al., 2018 | *Cell* | Drosophila | *Drosophila* brain cell atlas |
        
        ---
        
        ## Data Processing
        
        Expression data was processed as follows:
        1. **Raw counts** were obtained from original publications
        2. **Cell type annotations** were harmonized across datasets into common supercategories
        3. **Gene symbols** were mapped to human orthologs for cross-species comparison
        4. **Expression summaries** (mean expression, % expressing) were computed per cell type per dataset
        
        ### Cell Type Categories
        
        Cell types were harmonized into the following supercategories:
        """)
        
        # Display cell type legend
        celltype_df = pd.DataFrame([
            {"Category": "Excitatory Neurons", "Description": "Glutamatergic projection neurons"},
            {"Category": "Inhibitory Neurons", "Description": "GABAergic interneurons"},
            {"Category": "Neural Progenitors & Stem Cells", "Description": "NPCs, radial glia, neural stem cells"},
            {"Category": "Astrocytes", "Description": "Astrocytes and astrocyte precursors"},
            {"Category": "Oligodendrocyte Lineage", "Description": "OPCs, oligodendrocytes"},
            {"Category": "Microglia & Macrophages", "Description": "Brain-resident immune cells"},
            {"Category": "Endothelial & Vascular Cells", "Description": "Blood vessel cells"},
            {"Category": "Other Glia & Support", "Description": "Other glial cell types"},
            {"Category": "Neurons (unspecified)", "Description": "Neurons without E/I classification"},
            {"Category": "Early Embryonic / Germ Layers", "Description": "Early developmental cell types"},
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
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9rem;">
        SFARI Gene Expression Explorer | Built with Streamlit & Plotly<br>
        Data: Cross-species single-cell RNA-seq atlas
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
