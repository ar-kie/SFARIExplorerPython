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
    row_var: str = 'gene_display',
    col_vars: List[str] = ['tissue', 'cell_type'],
    scale_rows: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create a matrix suitable for heatmap visualization.
    
    Returns:
        matrix: The (optionally scaled) expression matrix
        col_meta: Metadata for each column
    """
    df = df.copy()
    
    # Create gene display name (prefer human symbol)
    df['gene_display'] = df['gene_human'].fillna(df['gene_native'])
    
    # Create column key
    df['col_key'] = df[col_vars].apply(lambda x: ' | '.join(x.astype(str)), axis=1)
    
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
    
    # Build column metadata
    col_meta = df[['col_key'] + col_vars + ['species']].drop_duplicates()
    col_meta = col_meta.set_index('col_key').reindex(pivot.columns)
    
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

def create_heatmap(
    matrix: pd.DataFrame,
    col_meta: pd.DataFrame,
    title: str = "Gene Expression Heatmap",
    color_scale: str = "RdBu_r",
    facet_by: Optional[str] = None,
    height: int = 600,
    show_dendrograms: bool = False
) -> go.Figure:
    """Create an interactive heatmap using Plotly."""
    
    if matrix.empty:
        fig = go.Figure()
        fig.add_annotation(text="No data to display", xref="paper", yref="paper",
                          x=0.5, y=0.5, showarrow=False, font=dict(size=20))
        return fig
    
    # Format column labels
    col_labels = [c.replace(' | ', '\n') for c in matrix.columns]
    
    fig = go.Figure(data=go.Heatmap(
        z=matrix.values,
        x=col_labels,
        y=matrix.index.tolist(),
        colorscale=color_scale,
        zmid=0,
        zmin=-3,
        zmax=3,
        colorbar=dict(
            title=dict(text="Z-score", side="right"),
            thickness=15,
            len=0.7
        ),
        hovertemplate=(
            "Gene: %{y}<br>"
            "Sample: %{x}<br>"
            "Z-score: %{z:.2f}<extra></extra>"
        )
    ))
    
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=18)),
        xaxis=dict(
            tickangle=45,
            tickfont=dict(size=10),
            side='bottom'
        ),
        yaxis=dict(
            tickfont=dict(size=9),
            autorange='reversed'
        ),
        height=max(400, 50 + len(matrix) * 18),
        margin=dict(l=120, r=50, t=80, b=150)
    )
    
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
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Heatmap", 
        "🔵 Dot Plot", 
        "📈 Temporal Dynamics",
        "🔬 Species Comparison",
        "📋 Data Table"
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
            # Heatmap options
            hm_col1, hm_col2 = st.columns([1, 3])
            with hm_col1:
                facet_option = st.selectbox(
                    "Split by",
                    options=["None", "Species", "Dataset", "Cell Type"],
                    index=0
                )
                color_scale = st.selectbox(
                    "Color scale",
                    options=["RdBu_r", "Viridis", "Plasma", "Inferno", "Blues"],
                    index=0
                )
            
            # Create heatmap matrix
            matrix, col_meta = create_heatmap_matrix(
                filtered_df,
                value_col=value_metric,
                scale_rows=scale_rows
            )
            
            # Cluster if requested
            if cluster_rows or cluster_cols:
                matrix, _, _ = cluster_matrix(matrix, cluster_rows, cluster_cols)
            
            # Create and display heatmap
            fig = create_heatmap(
                matrix,
                col_meta,
                title=f"Expression Heatmap ({len(selected_genes)} genes)",
                color_scale=color_scale
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
