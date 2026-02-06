"""
SFARI Gene Expression Explorer v2
Fixed version addressing all reported issues.
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

st.set_page_config(page_title="SFARI Gene Explorer", page_icon="🧬", layout="wide", initial_sidebar_state="expanded")

if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False

def get_theme_colors():
    if st.session_state.dark_mode:
        return {'bg': '#0e1117', 'text': '#fafafa', 'card_bg': '#262730', 'accent': '#4da6ff', 'plot_bg': '#0e1117', 'grid': '#333333'}
    else:
        return {'bg': '#ffffff', 'text': '#1f4e79', 'card_bg': '#f0f4f8', 'accent': '#2c5282', 'plot_bg': '#ffffff', 'grid': '#e0e0e0'}

theme = get_theme_colors()

st.markdown(f"""
<style>
    .stApp {{ max-width: 100%; }}
    .main .block-container {{ padding-top: 2rem; padding-bottom: 2rem; }}
    h1 {{ color: {theme['text']}; font-weight: 600; }}
    h2, h3 {{ color: {theme['accent']}; }}
    .stTabs [data-baseweb="tab-list"] {{ gap: 8px; }}
    .stTabs [data-baseweb="tab"] {{ background-color: {theme['card_bg']}; border-radius: 4px 4px 0 0; padding: 8px 16px; }}
    .stTabs [aria-selected="true"] {{ background-color: {theme['accent']}; color: white; }}
    div[data-testid="stMetricValue"] {{ font-size: 1.8rem; font-weight: 600; }}
    .info-box {{ background-color: {'#1e3a5f' if st.session_state.dark_mode else '#e8f4f8'}; border-left: 4px solid {theme['accent']}; padding: 1rem; margin: 1rem 0; border-radius: 0 8px 8px 0; }}
    .success-box {{ background-color: {'#1e3f1e' if st.session_state.dark_mode else '#e8f8e8'}; border-left: 4px solid #28a745; padding: 1rem; margin: 1rem 0; border-radius: 0 8px 8px 0; }}
</style>
""", unsafe_allow_html=True)

# =============================================================================
# Color Palettes - FIXED: Better microglia color
# =============================================================================

SPECIES_COLORS = {'Human': '#e41a1c', 'Mouse': '#377eb8', 'Zebrafish': '#4daf4a', 'Drosophila': '#984ea3'}

CELLTYPE_COLORS = {
    'Excitatory Neurons': '#e41a1c',
    'Inhibitory Neurons': '#377eb8',
    'Neural Progenitors & Stem Cells': '#4daf4a',
    'Astrocytes': '#984ea3',
    'Oligodendrocyte Lineage': '#ff7f00',
    'Microglia & Macrophages': '#8B4513',  # FIXED: Changed from yellow to brown
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

# FIXED: Proper time bin ordering with numeric keys for sorting
TIME_BIN_ORDER = {
    # Human in-vivo (days post-conception based)
    'Early fetal (GW<10)': 1, 'Mid fetal (GW10-20)': 2, 'Late fetal (GW20-40)': 3,
    'Infant (0-2y)': 4, 'Child (2-12y)': 5, 'Adolescent (12-18y)': 6, 'Adult (18+y)': 7,
    # Human organoid (days of differentiation)
    '0-30 days': 10, '31-60 days': 11, '61-90 days': 12, '91-120 days': 13, '>120 days': 14,
    # Mouse (embryonic days + postnatal)
    'Early embryo (E<12)': 20, 'Mid embryo (E12-16)': 21, 'Late embryo (E16-20)': 22,
    'Neonatal (P0-P30)': 23, 'Juvenile (1-3mo)': 24, 'Adult (3-12mo)': 25, 'Aged (>12mo)': 26,
    # Zebrafish (hours/days post-fertilization)
    '0-24 hpf': 30, '24-48 hpf': 31, '48-72 hpf': 32, '72-120 hpf (5dpf)': 33, '>5 dpf': 34,
    # Drosophila (days)
    '0-1 day': 40, '1-7 days': 41, '7-30 days': 42, '>30 days': 43,
    'Unknown': 99
}

TIME_BIN_COLORS = {
    'Early fetal (GW<10)': '#fef0d9', 'Mid fetal (GW10-20)': '#fdcc8a', 'Late fetal (GW20-40)': '#fc8d59',
    'Infant (0-2y)': '#e34a33', 'Child (2-12y)': '#b30000', 'Adolescent (12-18y)': '#7f0000', 'Adult (18+y)': '#4d0000',
    '0-30 days': '#edf8e9', '31-60 days': '#bae4b3', '61-90 days': '#74c476', '91-120 days': '#31a354', '>120 days': '#006d2c',
    'Early embryo (E<12)': '#eff3ff', 'Mid embryo (E12-16)': '#bdd7e7', 'Late embryo (E16-20)': '#6baed6',
    'Neonatal (P0-P30)': '#3182bd', 'Juvenile (1-3mo)': '#08519c', 'Adult (3-12mo)': '#08306b', 'Aged (>12mo)': '#041f4a',
    '0-24 hpf': '#f7fcf5', '24-48 hpf': '#c7e9c0', '48-72 hpf': '#74c476', '72-120 hpf (5dpf)': '#238b45', '>5 dpf': '#00441b',
    '0-1 day': '#f2f0f7', '1-7 days': '#cbc9e2', '7-30 days': '#9e9ac8', '>30 days': '#6a51a3',
}

# FIXED: Sample type display names
SAMPLE_TYPE_DISPLAY = {
    'in_vivo': 'Brain (ex-vivo)',
    'organoid': 'Organoid (in-vitro)',
    'Brain (ex-vivo)': 'Brain (ex-vivo)',
    'Organoid (in-vitro)': 'Organoid (in-vitro)'
}

def get_sample_type_display(sample_type):
    """Convert internal sample_type to display name."""
    return SAMPLE_TYPE_DISPLAY.get(sample_type, sample_type)

def get_sample_type_internal(display_name):
    """Convert display name back to internal sample_type."""
    reverse_map = {v: k for k, v in SAMPLE_TYPE_DISPLAY.items()}
    return reverse_map.get(display_name, display_name)

GENE_SYMBOLS = ['circle', 'square', 'diamond', 'cross', 'x', 'triangle-up', 'triangle-down', 
                'pentagon', 'hexagon', 'star', 'hourglass', 'bowtie']

def get_color_palette(values: List[str], palette_type: str = 'auto') -> Dict[str, str]:
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
            preset_colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#8B4513',
                           '#a65628', '#f781bf', '#999999', '#66c2a5', '#fc8d62', '#8da0cb']
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
    return 'plotly_dark' if st.session_state.dark_mode else 'plotly_white'

def sort_time_bins(time_bins, species=None, sample_type=None):
    """Sort time bins in proper developmental order."""
    return sorted(time_bins, key=lambda x: TIME_BIN_ORDER.get(x, 99))

# =============================================================================
# Data Loading
# =============================================================================

@st.cache_data(ttl=3600)
def load_data(data_dir: str = "data") -> Dict[str, pd.DataFrame]:
    data = {}
    try:
        data['expression'] = pd.read_parquet(f"{data_dir}/expression_summaries.parquet")
        data['cellmeta'] = pd.read_parquet(f"{data_dir}/celltype_meta.parquet")
        data['gene_map'] = pd.read_parquet(f"{data_dir}/gene_map.parquet")
        data['risk_genes'] = pd.read_parquet(f"{data_dir}/risk_genes.parquet")
        if 'gene-symbol' in data['risk_genes'].columns:
            data['risk_genes'] = data['risk_genes'].rename(columns={'gene-symbol': 'gene_symbol', 'gene-score': 'gene_score'})
    except Exception as e:
        st.error(f"Error loading core data: {e}")
        return None
    
    optional_files = {
        'umap': 'umap_subsample.parquet', 'vp_summary': 'variance_partition_summary.parquet',
        'vp_by_gene': 'variance_partition_by_gene.parquet', 'dataset_info': 'dataset_info.parquet',
        'batch_correction': 'batch_correction_info.parquet', 'summary_stats': 'summary_statistics.parquet',
        'temporal': 'temporal_expression.parquet', 'ortholog': 'ortholog_expression.parquet',
        'species_comparison': 'species_comparison.parquet',
    }
    for key, filename in optional_files.items():
        try:
            data[key] = pd.read_parquet(f"{data_dir}/{filename}")
        except:
            data[key] = None
    return data

def get_unique_values(df: pd.DataFrame, column: str) -> List[str]:
    if df is None or df.empty or column not in df.columns:
        return []
    return sorted(df[column].dropna().unique().tolist())
# =============================================================================
# Data Processing Functions
# =============================================================================

def filter_expression_data(expr_df, species=None, datasets=None, cell_types=None, genes=None):
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
        native_mask = df['gene_native'].notna() & df['gene_native'].str.upper().isin(genes_upper)
        human_mask = df['gene_human'].notna() & df['gene_human'].str.upper().isin(genes_upper)
        df = df[native_mask | human_mask]
    return df

def parse_gene_input(gene_text):
    if not gene_text or not gene_text.strip():
        return []
    genes = re.split(r'[,\s;]+', gene_text.strip())
    return [g.strip() for g in genes if g.strip()]

def create_heatmap_matrix(df, value_col='mean_expr', scale_rows=True):
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()
    df = df.copy()
    df['gene_display'] = df['gene_human'].fillna(df['gene_native'])
    df['col_key'] = df.apply(lambda x: f"{x['species']}|{x['tissue']}|{x['cell_type']}", axis=1)
    pivot = df.pivot_table(index='gene_display', columns='col_key', values=value_col, aggfunc='mean')
    if scale_rows and pivot.shape[0] > 0:
        row_means = pivot.mean(axis=1)
        row_stds = pivot.std(axis=1).replace(0, 1)
        pivot = pivot.sub(row_means, axis=0).div(row_stds, axis=0).clip(-3, 3)
    col_meta_records = []
    for col_key in pivot.columns:
        parts = col_key.split('|')
        col_meta_records.append({
            'col_key': col_key, 'species': parts[0] if len(parts) > 0 else '',
            'dataset': parts[1] if len(parts) > 1 else '', 'cell_type': parts[2] if len(parts) > 2 else ''
        })
    col_meta = pd.DataFrame(col_meta_records).set_index('col_key').reindex(pivot.columns)
    return pivot, col_meta

# =============================================================================
# Visualization - Heatmap (unchanged, works well)
# =============================================================================

def create_complexheatmap(matrix, col_meta, title="Heatmap", color_scale="RdBu_r", split_by=None,
                          annotation_col=None, cluster_rows=True, cluster_cols=True,
                          row_label_size=9, col_label_size=9, legend_position="bottom"):
    template = get_plotly_template()
    if matrix.empty:
        fig = go.Figure()
        fig.add_annotation(text="No data to display", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(template=template)
        return fig
    
    if cluster_rows and matrix.shape[0] > 1:
        try:
            mat_filled = matrix.fillna(0).values
            link = linkage(pdist(mat_filled), method='average')
            matrix = matrix.iloc[leaves_list(link)]
        except: pass
    
    split_values = [s for s in col_meta[split_by].unique() if pd.notna(s)] if split_by and split_by in col_meta.columns else [None]
    n_splits = len(split_values)
    split_matrices, split_col_metas, split_widths = [], [], []
    
    for sv in split_values:
        cols = col_meta[col_meta[split_by] == sv].index.tolist() if sv else col_meta.index.tolist()
        sub_matrix = matrix[[c for c in cols if c in matrix.columns]]
        if sub_matrix.empty: continue
        sub_col_meta = col_meta.loc[sub_matrix.columns]
        if cluster_cols and sub_matrix.shape[1] > 1:
            try:
                link = linkage(pdist(sub_matrix.fillna(0).values.T), method='average')
                order = leaves_list(link)
                sub_matrix, sub_col_meta = sub_matrix.iloc[:, order], sub_col_meta.iloc[order]
            except: pass
        split_matrices.append(sub_matrix)
        split_col_metas.append(sub_col_meta)
        split_widths.append(max(sub_matrix.shape[1], 1))
    
    if not split_matrices:
        fig = go.Figure()
        fig.add_annotation(text="No data after filtering", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(template=template)
        return fig
    
    col_widths = [w/sum(split_widths) for w in split_widths]
    n_splits = len(split_matrices)
    has_anno = annotation_col and annotation_col in col_meta.columns
    
    fig = make_subplots(rows=2 if has_anno else 1, cols=n_splits, column_widths=col_widths,
                       row_heights=[0.03, 0.97] if has_anno else [1.0],
                       horizontal_spacing=0.02, vertical_spacing=0.01,
                       subplot_titles=[str(s) if s else "" for s in split_values[:n_splits]] if n_splits > 1 else None)
    
    if has_anno:
        all_vals = col_meta[annotation_col].unique().tolist()
        anno_colors = get_color_palette(all_vals, 'species' if annotation_col == 'species' else 'cell_type' if annotation_col == 'cell_type' else 'dataset' if annotation_col == 'dataset' else 'auto')
    
    colorbar_added = False
    for idx, (sub_m, sub_cm) in enumerate(zip(split_matrices, split_col_metas)):
        col_idx = idx + 1
        if sub_m.empty: continue
        
        if split_by == 'species':
            col_labels = [f"{d}\n{c}" for d, c in zip(sub_cm['dataset'], sub_cm['cell_type'])]
        elif split_by == 'dataset':
            col_labels = [f"{s}\n{c}" for s, c in zip(sub_cm['species'], sub_cm['cell_type'])]
        else:
            col_labels = [f"{d}\n{c}" for d, c in zip(sub_cm['dataset'], sub_cm['cell_type'])]
        
        if has_anno:
            av = sub_cm[annotation_col].tolist()
            fig.add_trace(go.Heatmap(z=[[i for i in range(len(av))]], x=col_labels,
                         colorscale=[[i/max(len(av)-1,1), anno_colors.get(av[i],'#999')] for i in range(len(av))],
                         showscale=False, hoverinfo='skip'), row=1, col=col_idx)
        
        hm_row = 2 if has_anno else 1
        fig.add_trace(go.Heatmap(z=sub_m.values, x=col_labels, y=sub_m.index.tolist(),
                     colorscale=color_scale, zmid=0, zmin=-3, zmax=3,
                     showscale=not colorbar_added,
                     colorbar=dict(title='Z-score', thickness=15, len=0.7) if not colorbar_added else None),
                     row=hm_row, col=col_idx)
        colorbar_added = True
    
    height = max(400, 80 + len(matrix) * 16)
    fig.update_layout(title=dict(text=title, x=0.5), height=height, margin=dict(l=150, r=80, t=100, b=120),
                     showlegend=False, template=template)
    
    for i in range(1, n_splits + 1):
        hm_row = 2 if has_anno else 1
        if has_anno:
            fig.update_xaxes(showticklabels=False, row=1, col=i)
            fig.update_yaxes(showticklabels=False, row=1, col=i)
        fig.update_xaxes(tickangle=45, tickfont=dict(size=col_label_size), row=hm_row, col=i)
        fig.update_yaxes(tickfont=dict(size=row_label_size), autorange='reversed', showticklabels=(i==1), row=hm_row, col=i)
    
    if has_anno:
        for v, c in anno_colors.items():
            fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(size=10, color=c), name=str(v), showlegend=True))
        legend_config = dict(orientation='h', y=-0.15, x=0.5, xanchor='center') if legend_position == 'bottom' else dict(orientation='v', y=0.5, x=1.02, xanchor='left')
        legend_config['title'] = dict(text=annotation_col.replace('_', ' ').title())
        fig.update_layout(legend=legend_config, showlegend=True, margin=dict(l=150, r=80 if legend_position=='bottom' else 200, t=100, b=180 if legend_position=='bottom' else 120))
    
    return fig

# =============================================================================
# FIXED: Dot Plot with dataset rename and timepoint option
# =============================================================================

def create_dotplot(df, genes, group_by='cell_type', size_col='pct_expressing', color_col='mean_expr', title="Dot Plot"):
    template = get_plotly_template()
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No data", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(template=template)
        return fig
    
    df = df.copy()
    df['gene_display'] = df['gene_human'].fillna(df['gene_native'])
    
    # FIXED: Rename tissue to dataset for display
    if group_by == 'tissue':
        group_by_display = 'Dataset'
        df['Dataset'] = df['tissue']
        group_col = 'Dataset'
    elif group_by == 'timepoint' and 'time_bin' in df.columns:
        group_by_display = 'Time'
        # Sort time bins properly
        df['Time'] = pd.Categorical(df['time_bin'], categories=sort_time_bins(df['time_bin'].unique()), ordered=True)
        group_col = 'Time'
    else:
        group_by_display = group_by.replace('_', ' ').title()
        group_col = group_by
    
    agg_df = df.groupby(['gene_display', group_col]).agg({size_col: 'mean', color_col: 'mean'}).reset_index()
    agg_df['size_scaled'] = agg_df[size_col] * 30 + 5
    
    fig = px.scatter(agg_df, x=group_col, y='gene_display', size='size_scaled', color=color_col,
                    color_continuous_scale='Viridis', title=title,
                    labels={color_col: 'Mean Expression', 'gene_display': 'Gene', group_col: group_by_display})
    
    fig.update_layout(height=max(400, 50 + len(genes) * 25), xaxis_tickangle=45,
                     yaxis=dict(autorange='reversed'), template=template)
    return fig
# =============================================================================
# FIXED: Temporal Dynamics - Trajectory Plot with proper time ordering
# =============================================================================

def create_temporal_trajectory_plot(temporal_df, genes, species='Human', sample_type='in_vivo',
                                    cell_types=None, value_col='mean_expr'):
    """Trajectory plot: colors = cell types, symbols = genes. Fixed time ordering."""
    template = get_plotly_template()
    
    try:
        if temporal_df is None or temporal_df.empty:
            fig = go.Figure()
            fig.add_annotation(text="No temporal data available", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            fig.update_layout(template=template)
            return fig
        
        df = temporal_df[temporal_df['species'] == species].copy()
        
        # Filter by sample type
        if 'sample_type' in df.columns and sample_type:
            df = df[df['sample_type'] == sample_type]
        
        # Filter genes
        genes_upper = [g.upper() for g in genes]
        df = df[df['gene_human'].notna()]
        df = df[df['gene_human'].str.upper().isin(genes_upper)]
        
        # Filter cell types
        if cell_types and len(cell_types) > 0:
            df = df[df['cell_type'].isin(cell_types)]
        
        if df.empty:
            fig = go.Figure()
            fig.add_annotation(text="No temporal data for selected filters", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            fig.update_layout(template=template)
            return fig
        
        # FIXED: Proper time bin sorting
        available_bins = df['time_bin'].unique().tolist()
        sorted_bins = sort_time_bins(available_bins)
        
        # Create categorical for proper ordering
        df['time_bin_cat'] = pd.Categorical(df['time_bin'], categories=sorted_bins, ordered=True)
        df = df.sort_values('time_bin_cat')
        
        fig = go.Figure()
        unique_genes = df['gene_human'].unique().tolist()
        unique_celltypes = df['cell_type'].unique().tolist()
        
        celltype_colors = get_color_palette(unique_celltypes, 'cell_type')
        gene_symbols = {g: GENE_SYMBOLS[i % len(GENE_SYMBOLS)] for i, g in enumerate(unique_genes)}
        
        for ct in unique_celltypes:
            ct_data = df[df['cell_type'] == ct]
            for gene in unique_genes:
                gene_data = ct_data[ct_data['gene_human'] == gene]
                if gene_data.empty:
                    continue
                
                # Aggregate by time bin, preserving order
                agg = gene_data.groupby('time_bin_cat', observed=True).agg({
                    value_col: 'mean', 'pct_expressing': 'mean'
                }).reset_index()
                agg = agg.sort_values('time_bin_cat')
                
                if agg.empty or len(agg) < 1:
                    continue
                
                show_legend = (gene == unique_genes[0])
                
                fig.add_trace(go.Scatter(
                    x=[str(t) for t in agg['time_bin_cat']],  # Convert to string for display
                    y=agg[value_col],
                    mode='lines+markers',
                    name=ct if show_legend else None,
                    legendgroup=ct,
                    showlegend=show_legend,
                    line=dict(color=celltype_colors.get(ct, '#999999'), width=2),
                    marker=dict(size=10, symbol=gene_symbols.get(gene, 'circle'), color=celltype_colors.get(ct, '#999999')),
                    hovertemplate=f"Gene: {gene}<br>Cell type: {ct}<br>Time: %{{x}}<br>Expression: %{{y:.3f}}<extra></extra>"
                ))
        
        # Gene symbol legend annotation
        if len(unique_genes) > 1:
            symbol_text = "Symbols: " + ", ".join([f"{g} ({gene_symbols[g]})" for g in unique_genes])
            fig.add_annotation(text=symbol_text, xref="paper", yref="paper", x=0.5, y=-0.22, showarrow=False, font=dict(size=10))
        
        # Title
        sample_display = get_sample_type_display(sample_type)
        title = f"{species} {sample_display} - Temporal Expression"
        
        fig.update_layout(
            title=dict(text=title, x=0.5, font=dict(size=18)),
            xaxis_title="Developmental Stage",
            yaxis_title="Mean Expression",
            height=550,
            legend=dict(orientation='v', yanchor='top', y=1, xanchor='left', x=1.02),
            template=template,
            margin=dict(b=100, r=150)
        )
        
        return fig
        
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(text=f"Error: {str(e)}", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(template=template)
        return fig

# =============================================================================
# Temporal Heatmap (works well, keeping as is)
# =============================================================================

def create_temporal_heatmap(temporal_df, genes, species='Human', sample_type='in_vivo',
                            cell_type=None, value_col='mean_expr'):
    template = get_plotly_template()
    
    try:
        if temporal_df is None or temporal_df.empty:
            fig = go.Figure()
            fig.add_annotation(text="No temporal data", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
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
        
        agg = df.groupby(['gene_human', 'time_bin']).agg({value_col: 'mean'}).reset_index()
        pivot = agg.pivot_table(index='gene_human', columns='time_bin', values=value_col, aggfunc='mean')
        
        if pivot.empty:
            fig = go.Figure()
            fig.add_annotation(text="No data to display", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            fig.update_layout(template=template)
            return fig
        
        # FIXED: Sort columns by proper time order
        sorted_cols = sort_time_bins([c for c in pivot.columns])
        pivot = pivot[[c for c in sorted_cols if c in pivot.columns]]
        
        row_means, row_stds = pivot.mean(axis=1), pivot.std(axis=1).replace(0, 1)
        pivot_scaled = pivot.sub(row_means, axis=0).div(row_stds, axis=0).clip(-3, 3)
        
        fig = go.Figure(go.Heatmap(
            z=pivot_scaled.values, x=pivot_scaled.columns.tolist(), y=pivot_scaled.index.tolist(),
            colorscale='RdBu_r', zmid=0, zmin=-3, zmax=3, colorbar=dict(title='Z-score'),
            hovertemplate="Gene: %{y}<br>Time: %{x}<br>Z-score: %{z:.2f}<extra></extra>"
        ))
        
        sample_display = get_sample_type_display(sample_type)
        cell_type_str = f" ({cell_type})" if cell_type and cell_type != 'All' else " (all cell types)"
        fig.update_layout(
            title=dict(text=f"{species} {sample_display} - Temporal Heatmap{cell_type_str}", x=0.5),
            xaxis_title="Developmental Time", yaxis_title="Gene",
            height=max(400, 50 + len(genes) * 20), xaxis=dict(tickangle=45),
            yaxis=dict(autorange='reversed'), template=template
        )
        return fig
        
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(text=f"Error: {str(e)}", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(template=template)
        return fig
# =============================================================================
# FIXED: Multi-Species Comparison - removed useless selectors, fixed legend
# =============================================================================

def create_multi_species_temporal_comparison(temporal_df, gene, cell_types=None, value_col='mean_expr'):
    """
    Compare one gene's temporal dynamics across ALL species.
    FIXED: Removed species/sample_type selectors (uses all), legend outside plot.
    Handles Human organoid vs brain by showing them separately.
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
        
        # Create species + sample_type combination for Human
        df['plot_group'] = df.apply(
            lambda x: f"Human ({get_sample_type_display(x.get('sample_type', 'in_vivo')).split('(')[0].strip()})" 
                      if x['species'] == 'Human' and 'sample_type' in df.columns 
                      else x['species'], 
            axis=1
        )
        
        plot_groups = sorted(df['plot_group'].unique().tolist())
        n_groups = len(plot_groups)
        
        if n_groups == 0:
            fig = go.Figure()
            fig.add_annotation(text="No data to plot", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            fig.update_layout(template=template)
            return fig
        
        fig = make_subplots(rows=1, cols=n_groups, subplot_titles=plot_groups, shared_yaxes=True, horizontal_spacing=0.06)
        
        celltype_colors = get_color_palette(df['cell_type'].unique().tolist(), 'cell_type')
        legend_added = set()
        
        for i, group in enumerate(plot_groups):
            group_data = df[df['plot_group'] == group]
            
            # Sort time bins for this group
            available_bins = group_data['time_bin'].unique().tolist()
            sorted_bins = sort_time_bins(available_bins)
            group_data['time_bin_cat'] = pd.Categorical(group_data['time_bin'], categories=sorted_bins, ordered=True)
            
            for ct in group_data['cell_type'].unique():
                ct_data = group_data[group_data['cell_type'] == ct]
                
                agg = ct_data.groupby('time_bin_cat', observed=True).agg({value_col: 'mean'}).reset_index()
                agg = agg.sort_values('time_bin_cat')
                
                if agg.empty:
                    continue
                
                show_legend = ct not in legend_added
                if show_legend:
                    legend_added.add(ct)
                
                fig.add_trace(
                    go.Scatter(
                        x=[str(t) for t in agg['time_bin_cat']],
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
            height=500,
            # FIXED: Legend on right side, outside plot
            legend=dict(orientation='v', yanchor='top', y=1, xanchor='left', x=1.02, title=dict(text='Cell Type')),
            template=template,
            margin=dict(b=100, r=180)
        )
        
        fig.update_yaxes(title_text="Mean Expression", row=1, col=1)
        
        return fig
        
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(text=f"Error: {str(e)}", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(template=template)
        return fig

# =============================================================================
# FIXED: Timepoint Snapshot - proper cell type filtering and time ordering
# =============================================================================

def create_celltype_by_species_temporal(temporal_df, gene, time_bin, selected_cell_types=None, value_col='mean_expr'):
    """
    Bar chart showing expression at a specific timepoint.
    FIXED: Proper cell type filtering, grouped by cell type with species colors.
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
        
        # FIXED: Actually filter by selected cell types!
        if selected_cell_types and len(selected_cell_types) > 0:
            df = df[df['cell_type'].isin(selected_cell_types)]
        
        if df.empty:
            fig = go.Figure()
            fig.add_annotation(text=f"No data for {gene} at {time_bin}", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            fig.update_layout(template=template)
            return fig
        
        # Aggregate - include sample_type for Human distinction
        if 'sample_type' in df.columns:
            df['species_sample'] = df.apply(
                lambda x: f"Human ({get_sample_type_display(x['sample_type']).split('(')[0].strip()})" 
                          if x['species'] == 'Human' else x['species'], axis=1
            )
            group_col = 'species_sample'
        else:
            group_col = 'species'
        
        agg = df.groupby([group_col, 'cell_type']).agg({value_col: 'mean', 'pct_expressing': 'mean'}).reset_index()
        
        fig = px.bar(
            agg, x='cell_type', y=value_col, color=group_col, barmode='group',
            color_discrete_map={**SPECIES_COLORS, 'Human (Brain)': '#e41a1c', 'Human (Organoid)': '#fbb4ae'},
            title=f"{gene.upper()} Expression at {time_bin}",
            labels={value_col: 'Mean Expression', 'cell_type': 'Cell Type', group_col: 'Species/Sample'}
        )
        
        fig.update_layout(
            height=450, xaxis_tickangle=45,
            legend=dict(orientation='v', yanchor='top', y=1, xanchor='left', x=1.02),
            template=template,
            margin=dict(r=150)
        )
        
        return fig
        
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(text=f"Error: {str(e)}", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(template=template)
        return fig
# =============================================================================
# FIXED: Cross-Species Comparison - Species Correlation Heatmap
# =============================================================================

def create_species_expression_comparison(ortholog_df, genes, cell_types=None):
    """Bar chart comparing expression across species."""
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
        
        fig = px.bar(agg, x='gene_human', y='mean_expr', color='species', barmode='group',
                    color_discrete_map=SPECIES_COLORS, title="Cross-Species Expression Comparison",
                    labels={'mean_expr': 'Mean Expression', 'gene_human': 'Gene', 'species': 'Species'})
        
        fig.update_layout(height=450, xaxis_tickangle=45,
                         legend=dict(orientation='h', y=1.02, x=0.5, xanchor='center'), template=template)
        return fig
        
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(text=f"Error: {str(e)}", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(template=template)
        return fig

def create_species_correlation_heatmap(species_comparison_df, genes=None):
    """
    FIXED: Species correlation heatmap - properly match genes.
    """
    template = get_plotly_template()
    
    try:
        if species_comparison_df is None or species_comparison_df.empty:
            fig = go.Figure()
            fig.add_annotation(text="No species comparison data available.\nGenerate species_comparison.parquet first.", 
                             xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False, font=dict(size=14))
            fig.update_layout(template=template)
            return fig
        
        df = species_comparison_df.copy()
        
        # Debug: show what's in the data
        available_genes = df['gene_human'].unique().tolist()[:20]  # First 20
        
        if genes and len(genes) > 0:
            genes_upper = [g.upper() for g in genes]
            # FIXED: Case-insensitive matching
            df['gene_upper'] = df['gene_human'].str.upper()
            df = df[df['gene_upper'].isin(genes_upper)]
            df = df.drop(columns=['gene_upper'])
        
        if df.empty:
            fig = go.Figure()
            msg = f"No correlation data for selected genes.\nAvailable genes include: {', '.join(available_genes[:10])}..."
            fig.add_annotation(text=msg, xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False, font=dict(size=12))
            fig.update_layout(template=template)
            return fig
        
        df['species_pair'] = df['species_1'] + ' vs ' + df['species_2']
        
        pivot = df.pivot_table(index='gene_human', columns='species_pair', values='expression_correlation', aggfunc='mean')
        
        if pivot.empty:
            fig = go.Figure()
            fig.add_annotation(text="No correlation data after aggregation", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            fig.update_layout(template=template)
            return fig
        
        fig = go.Figure(go.Heatmap(
            z=pivot.values, x=pivot.columns.tolist(), y=pivot.index.tolist(),
            colorscale='RdBu', zmid=0, zmin=-1, zmax=1, colorbar=dict(title='Correlation'),
            hovertemplate="Gene: %{y}<br>Comparison: %{x}<br>Correlation: %{z:.3f}<extra></extra>"
        ))
        
        fig.update_layout(
            title=dict(text="Cross-Species Expression Correlation", x=0.5),
            xaxis_title="Species Comparison", yaxis_title="Gene",
            height=max(400, 50 + len(pivot) * 20), xaxis=dict(tickangle=45),
            yaxis=dict(autorange='reversed'), template=template
        )
        return fig
        
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(text=f"Error: {str(e)}", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(template=template)
        return fig

def create_ortholog_scatter(ortholog_df, gene, species_x, species_y):
    """Scatter plot comparing expression between two species."""
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
            corr, _ = spearmanr(merged['expr_x'], merged['expr_y'])
            corr_text = f"ρ = {corr:.3f}"
        
        fig = px.scatter(merged, x='expr_x', y='expr_y', color='cell_type', color_discrete_map=CELLTYPE_COLORS,
                        title=f"{gene.upper()}: {species_x} vs {species_y}",
                        labels={'expr_x': f'{species_x} Expression', 'expr_y': f'{species_y} Expression', 'cell_type': 'Cell Type'})
        
        max_val = max(merged['expr_x'].max(), merged['expr_y'].max())
        fig.add_trace(go.Scatter(x=[0, max_val], y=[0, max_val], mode='lines', line=dict(dash='dash', color='gray'), showlegend=False))
        
        if corr_text:
            fig.add_annotation(text=corr_text, xref="paper", yref="paper", x=0.95, y=0.05, showarrow=False, font=dict(size=14))
        
        fig.update_traces(marker=dict(size=12))
        fig.update_layout(height=500, template=template,
                         legend=dict(orientation='v', yanchor='top', y=1, xanchor='left', x=1.02))
        return fig
        
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(text=f"Error: {str(e)}", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(template=template)
        return fig

# =============================================================================
# Cross-Species Trajectory Plot (NEW - like trajectory but across species)
# =============================================================================

def create_cross_species_trajectory(temporal_df, genes, cell_types=None, value_col='mean_expr'):
    """
    NEW: Trajectory-style plot for cross-species comparison.
    Shows all species in one plot, colored by cell type, faceted by species.
    """
    template = get_plotly_template()
    
    try:
        if temporal_df is None or temporal_df.empty:
            fig = go.Figure()
            fig.add_annotation(text="No temporal data", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            fig.update_layout(template=template)
            return fig
        
        df = temporal_df.copy()
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
        
        # Create species + sample_type combination
        if 'sample_type' in df.columns:
            df['plot_group'] = df.apply(
                lambda x: f"Human ({get_sample_type_display(x.get('sample_type', 'in_vivo')).split('(')[0].strip()})" 
                          if x['species'] == 'Human' else x['species'], axis=1
            )
        else:
            df['plot_group'] = df['species']
        
        plot_groups = sorted(df['plot_group'].unique().tolist())
        n_groups = len(plot_groups)
        
        fig = make_subplots(rows=1, cols=n_groups, subplot_titles=plot_groups, shared_yaxes=True, horizontal_spacing=0.06)
        
        unique_genes = df['gene_human'].unique().tolist()
        unique_cts = df['cell_type'].unique().tolist()
        
        celltype_colors = get_color_palette(unique_cts, 'cell_type')
        gene_symbols = {g: GENE_SYMBOLS[i % len(GENE_SYMBOLS)] for i, g in enumerate(unique_genes)}
        legend_added = set()
        
        for i, group in enumerate(plot_groups):
            group_data = df[df['plot_group'] == group]
            
            sorted_bins = sort_time_bins(group_data['time_bin'].unique().tolist())
            group_data['time_bin_cat'] = pd.Categorical(group_data['time_bin'], categories=sorted_bins, ordered=True)
            
            for ct in group_data['cell_type'].unique():
                ct_data = group_data[group_data['cell_type'] == ct]
                
                for gene in unique_genes:
                    gene_data = ct_data[ct_data['gene_human'] == gene]
                    if gene_data.empty:
                        continue
                    
                    agg = gene_data.groupby('time_bin_cat', observed=True).agg({value_col: 'mean'}).reset_index()
                    agg = agg.sort_values('time_bin_cat')
                    
                    if agg.empty:
                        continue
                    
                    show_legend = ct not in legend_added
                    if show_legend:
                        legend_added.add(ct)
                    
                    fig.add_trace(
                        go.Scatter(
                            x=[str(t) for t in agg['time_bin_cat']],
                            y=agg[value_col],
                            mode='lines+markers',
                            name=ct if show_legend else None,
                            legendgroup=ct,
                            showlegend=show_legend,
                            marker=dict(size=8, symbol=gene_symbols.get(gene, 'circle'), color=celltype_colors.get(ct, '#999999')),
                            line=dict(color=celltype_colors.get(ct, '#999999'), width=2),
                            hovertemplate=f"Gene: {gene}<br>Cell type: {ct}<br>Time: %{{x}}<br>Expression: %{{y:.3f}}<extra></extra>"
                        ),
                        row=1, col=i+1
                    )
            
            fig.update_xaxes(tickangle=45, row=1, col=i+1)
        
        # Gene symbol annotation
        if len(unique_genes) > 1:
            symbol_text = "Symbols: " + ", ".join([f"{g} ({gene_symbols[g]})" for g in unique_genes])
            fig.add_annotation(text=symbol_text, xref="paper", yref="paper", x=0.5, y=-0.2, showarrow=False, font=dict(size=10))
        
        gene_str = ", ".join(unique_genes[:3]) + ("..." if len(unique_genes) > 3 else "")
        fig.update_layout(
            title=dict(text=f"Cross-Species Temporal Comparison: {gene_str}", x=0.5, font=dict(size=18)),
            height=500,
            legend=dict(orientation='v', yanchor='top', y=1, xanchor='left', x=1.02, title=dict(text='Cell Type')),
            template=template,
            margin=dict(b=120, r=180)
        )
        
        fig.update_yaxes(title_text="Mean Expression", row=1, col=1)
        
        return fig
        
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(text=f"Error: {str(e)}", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(template=template)
        return fig
# =============================================================================
# Data Overview Functions
# =============================================================================

def create_variance_partition_barplot(vp_summary):
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
            sd = vp_summary[vp_summary['stage'] == stage]
            fig.add_trace(go.Bar(name=stage, x=sd['variable'], y=sd['variance_explained'], marker_color=colors[stage],
                                text=[f"{v:.1f}%" for v in sd['variance_explained']], textposition='outside'))
        
        fig.update_layout(title=dict(text="Variance Explained by Each Factor", x=0.5), xaxis_title="Variable",
                         yaxis_title="Variance (%)", barmode='group', height=450,
                         legend=dict(orientation='h', y=1.02, x=0.5, xanchor='center'),
                         yaxis=dict(range=[0, max(vp_summary['variance_explained']) * 1.15]), template=template)
        return fig
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(text=f"Error: {str(e)}", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(template=template)
        return fig

def create_variance_change_plot(vp_summary):
    template = get_plotly_template()
    try:
        if vp_summary is None or vp_summary.empty:
            fig = go.Figure()
            fig.add_annotation(text="No data", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            fig.update_layout(template=template)
            return fig
        
        before = vp_summary[vp_summary['stage'] == 'Before Correction'].set_index('variable')['variance_explained']
        after = vp_summary[vp_summary['stage'] == 'After Correction'].set_index('variable')['variance_explained']
        common_vars = before.index.intersection(after.index)
        vars_df = pd.DataFrame({'variable': common_vars, 'change': [after[v] - before[v] for v in common_vars]})
        
        colors = []
        for _, row in vars_df.iterrows():
            v, c = row['variable'], row['change']
            if v == 'dataset': colors.append('#28a745' if c < 0 else '#dc3545')
            elif v == 'Residuals': colors.append('#6c757d')
            else: colors.append('#28a745' if c >= 0 else '#dc3545')
        
        fig = go.Figure(go.Bar(x=vars_df['variable'], y=vars_df['change'], marker_color=colors,
                              text=[f"{c:+.1f}%" for c in vars_df['change']], textposition='outside'))
        fig.update_layout(title=dict(text="Change in Variance After Correction", x=0.5),
                         xaxis_title="Variable", yaxis_title="Change (%)", height=400,
                         yaxis=dict(zeroline=True, zerolinewidth=2, zerolinecolor='black'), template=template)
        return fig
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(text=f"Error: {str(e)}", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(template=template)
        return fig

def create_dataset_summary_table(dataset_info):
    if dataset_info is None or dataset_info.empty:
        return pd.DataFrame()
    display_cols = ['dataset', 'organism', 'sample_type', 'n_pseudobulk_samples', 'n_cell_types', 'n_timepoints']
    available_cols = [c for c in display_cols if c in dataset_info.columns]
    df = dataset_info[available_cols].copy()
    
    # FIXED: Rename sample_type values for display
    if 'sample_type' in df.columns:
        df['sample_type'] = df['sample_type'].apply(get_sample_type_display)
    
    rename_map = {'dataset': 'Dataset', 'organism': 'Species', 'sample_type': 'Sample Type',
                  'n_pseudobulk_samples': 'Samples', 'n_cell_types': 'Cell Types', 'n_timepoints': 'Timepoints'}
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    return df
# =============================================================================
# Main App
# =============================================================================

def main():
    col_title, col_toggle = st.columns([9, 1])
    with col_title:
        st.title("🧬 SFARI Gene Expression Explorer")
        st.markdown("*Cross-species single-cell RNA-seq browser for neurodevelopmental gene expression*")
    with col_toggle:
        if st.button("🌙" if not st.session_state.dark_mode else "☀️", help="Toggle dark mode"):
            st.session_state.dark_mode = not st.session_state.dark_mode
            st.rerun()
    
    data = load_data()
    if data is None:
        st.error("Failed to load data.")
        st.stop()
    
    expr_df, risk_genes = data['expression'], data['risk_genes']
    
    # ==========================================================================
    # Sidebar
    # ==========================================================================
    with st.sidebar:
        st.header("🔍 Query Filters")
        
        all_species = get_unique_values(expr_df, 'species')
        selected_species = st.multiselect("Species", all_species, default=all_species[:1] if all_species else [])
        
        try:
            if selected_species:
                available_datasets = get_unique_values(expr_df[expr_df['species'].isin(selected_species)], 'tissue')
            else:
                available_datasets = get_unique_values(expr_df, 'tissue')
        except:
            available_datasets = get_unique_values(expr_df, 'tissue')
        
        selected_datasets = st.multiselect("Dataset", available_datasets, default=[])
        
        try:
            subset = expr_df.copy()
            if selected_species: subset = subset[subset['species'].isin(selected_species)]
            if selected_datasets: subset = subset[subset['tissue'].isin(selected_datasets)]
            available_celltypes = get_unique_values(subset, 'cell_type')
        except:
            available_celltypes = get_unique_values(expr_df, 'cell_type')
        
        selected_celltypes = st.multiselect("Cell Types", available_celltypes, default=[])
        
        st.divider()
        st.subheader("🧬 Gene Selection")
        
        gene_preset = st.selectbox("Quick Gene Sets", ["Custom", "SFARI Score 1", "SFARI Score 2", "Top Variable"])
        preset_genes = ""
        if gene_preset == "SFARI Score 1":
            preset_genes = ", ".join(risk_genes[risk_genes['gene_score'] == 1]['gene_symbol'].dropna().tolist()[:50])
        elif gene_preset == "SFARI Score 2":
            preset_genes = ", ".join(risk_genes[risk_genes['gene_score'] == 2]['gene_symbol'].dropna().tolist()[:50])
        elif gene_preset == "Top Variable":
            preset_genes = ", ".join(expr_df.groupby('gene_human')['mean_expr'].var().sort_values(ascending=False).head(30).index.tolist())
        
        gene_input = st.text_area("Enter genes", value=preset_genes, height=100, placeholder="SHANK3, MECP2, CHD8")
        selected_genes = parse_gene_input(gene_input)
        
        st.divider()
        st.subheader("⚙️ Options")
        value_metric = st.radio("Value", ['mean_expr', 'pct_expressing'],
                               format_func=lambda x: "Mean Expression" if x == 'mean_expr' else "% Expressing", horizontal=True)
        scale_rows = st.checkbox("Z-score scaling", value=True)
        cluster_rows = st.checkbox("Cluster rows", value=False)
        cluster_cols = st.checkbox("Cluster columns", value=False)
    
    # Filter data
    try:
        filtered_df = filter_expression_data(expr_df,
            species=selected_species if selected_species else None,
            datasets=selected_datasets if selected_datasets else None,
            cell_types=selected_celltypes if selected_celltypes else None,
            genes=selected_genes if selected_genes else None)
    except:
        filtered_df = pd.DataFrame()
    
    # Metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Genes", filtered_df['gene_human'].nunique() if not filtered_df.empty else 0)
    c2.metric("Species", filtered_df['species'].nunique() if not filtered_df.empty else 0)
    c3.metric("Datasets", filtered_df['tissue'].nunique() if not filtered_df.empty else 0)
    c4.metric("Cell Types", filtered_df['cell_type'].nunique() if not filtered_df.empty else 0)
    
    # Tabs
    tabs = st.tabs(["📊 Overview", "🗺️ UMAP", "🔥 Heatmap", "🔵 Dot Plot", "📈 Temporal", "🔬 Cross-Species", "📋 Data", "📚 About"])
    
    # --------------------------------------------------------------------------
    # Tab: Overview
    # --------------------------------------------------------------------------
    with tabs[0]:
        st.header("Data Overview & Batch Correction")
        if data['summary_stats'] is not None:
            totals = data['summary_stats'][data['summary_stats']['category'] == 'totals'].set_index('label')['value']
            cols = st.columns(4)
            for i, (k, l) in enumerate([('pseudobulk_samples', 'Samples'), ('datasets', 'Datasets'), ('cell_types', 'Cell Types'), ('organisms', 'Species')]):
                if k in totals.index: cols[i].metric(l, f"{int(totals[k]):,}")
        
        if data['dataset_info'] is not None:
            st.subheader("📋 Datasets")
            st.dataframe(create_dataset_summary_table(data['dataset_info']), hide_index=True, use_container_width=True)
        
        st.divider()
        if data['batch_correction'] is not None:
            st.subheader("🔧 Batch Correction")
            st.markdown('<div class="info-box"><strong>Approach:</strong> Within-organism ComBat correction preserving cell type and developmental time.</div>', unsafe_allow_html=True)
        
        if data['vp_summary'] is not None:
            st.subheader("📊 Variance Partition")
            c1, c2 = st.columns(2)
            c1.plotly_chart(create_variance_partition_barplot(data['vp_summary']), use_container_width=True)
            c2.plotly_chart(create_variance_change_plot(data['vp_summary']), use_container_width=True)
    
    # --------------------------------------------------------------------------
    # Tab: UMAP
    # --------------------------------------------------------------------------
    with tabs[1]:
        st.header("UMAP Visualization")
        if data['umap'] is not None:
            umap_df = data['umap']
            st.markdown(f"**{len(umap_df):,} cells** subsampled.")
            color_cols = [c for c in umap_df.columns if c not in ['umap_1', 'umap_2', 'cell_id']]
            c1, c2 = st.columns([1, 3])
            with c1:
                color_by = st.selectbox("Color by", color_cols, index=color_cols.index('predicted_labels') if 'predicted_labels' in color_cols else 0)
                pt_size = st.slider("Point size", 1, 10, 3)
                opacity = st.slider("Opacity", 0.1, 1.0, 0.6)
            with c2:
                cmap = SPECIES_COLORS if color_by == 'organism' else CELLTYPE_COLORS if color_by in ['predicted_labels', 'cell_type'] else DATASET_COLORS if color_by == 'dataset' else get_color_palette(umap_df[color_by].unique().tolist())
                fig = px.scatter(umap_df, x='umap_1', y='umap_2', color=color_by, color_discrete_map=cmap)
                fig.update_traces(marker=dict(size=pt_size, opacity=opacity))
                fig.update_layout(height=650, legend=dict(font=dict(size=9)), template=get_plotly_template())
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("UMAP data not available.")
    
    # --------------------------------------------------------------------------
    # Tab: Heatmap
    # --------------------------------------------------------------------------
    with tabs[2]:
        if filtered_df.empty: st.warning("No data matches filters.")
        elif not selected_genes: st.info("Enter genes in sidebar.")
        else:
            c1, c2, c3, c4 = st.columns(4)
            split_by = {"None": None, "Species": "species", "Dataset": "dataset", "Cell Type": "cell_type"}[c1.selectbox("Split by", ["None", "Species", "Dataset", "Cell Type"])]
            anno_col = {"None": None, "Species": "species", "Dataset": "dataset", "Cell Type": "cell_type"}[c2.selectbox("Annotation", ["None", "Species", "Dataset", "Cell Type"])]
            color_scale = c3.selectbox("Colors", ["RdBu_r", "Viridis", "Plasma", "Blues"])
            legend_pos = c4.selectbox("Legend", ["Bottom", "Right"]).lower()
            matrix, col_meta = create_heatmap_matrix(filtered_df, value_metric, scale_rows)
            fig = create_complexheatmap(matrix, col_meta, f"Heatmap ({len(selected_genes)} genes)", color_scale, split_by, anno_col, cluster_rows, cluster_cols, legend_position=legend_pos)
            st.plotly_chart(fig, use_container_width=True)
    
    # --------------------------------------------------------------------------
    # Tab: Dot Plot - FIXED: tissue->dataset, added timepoint
    # --------------------------------------------------------------------------
    with tabs[3]:
        if filtered_df.empty or not selected_genes:
            st.info("Select filters and genes.")
        else:
            # FIXED: Show 'Dataset' instead of 'tissue', add 'Timepoint' option
            group_options = ['cell_type', 'tissue', 'species']
            group_labels = {'cell_type': 'Cell Type', 'tissue': 'Dataset', 'species': 'Species'}
            if 'time_bin' in filtered_df.columns:
                group_options.append('timepoint')
                group_labels['timepoint'] = 'Timepoint'
            
            group_by = st.selectbox("Group by", group_options, format_func=lambda x: group_labels.get(x, x))
            fig = create_dotplot(filtered_df, selected_genes, group_by=group_by)
            st.plotly_chart(fig, use_container_width=True)
    # --------------------------------------------------------------------------
    # Tab: Temporal - FIXED: removed redundant cell type selector, proper ordering
    # --------------------------------------------------------------------------
    with tabs[4]:
        st.header("Temporal Expression Dynamics")
        
        try:
            if data['temporal'] is not None and not data['temporal'].empty:
                temporal_df = data['temporal']
                
                st.markdown("**Colors = cell types, Symbols = genes** (in trajectory plots).")
                
                viz_type = st.selectbox("Visualization Type",
                    ["Trajectory Plot", "Temporal Heatmap", "Multi-Species Comparison", "Timepoint Snapshot"], key='temp_viz')
                
                # Only show species/sample_type for non-multi-species views
                if viz_type != "Multi-Species Comparison":
                    tc1, tc2 = st.columns(2)
                    with tc1:
                        available_species = temporal_df['species'].unique().tolist()
                        temp_species = st.selectbox("Species", available_species, key='temp_sp')
                    with tc2:
                        temp_sample_type = 'in_vivo'
                        if temp_species == 'Human' and 'sample_type' in temporal_df.columns:
                            sample_types = temporal_df[temporal_df['species'] == 'Human']['sample_type'].dropna().unique().tolist()
                            sample_display = [get_sample_type_display(s) for s in sample_types]
                            if sample_display:
                                selected_display = st.selectbox("Sample Type", sample_display, key='temp_st')
                                temp_sample_type = get_sample_type_internal(selected_display)
                        else:
                            st.text(f"Sample Type: {get_sample_type_display('in_vivo')}")
                    
                    # Cell types for this species/sample_type
                    species_temp = temporal_df[temporal_df['species'] == temp_species]
                    if 'sample_type' in species_temp.columns:
                        species_temp = species_temp[species_temp['sample_type'] == temp_sample_type]
                    available_cts = species_temp['cell_type'].unique().tolist()
                else:
                    temp_species = None
                    temp_sample_type = None
                    available_cts = temporal_df['cell_type'].unique().tolist()
                
                # Cell type selection (used by all views)
                selected_temp_cts = st.multiselect("Cell Types", available_cts,
                    default=available_cts[:3] if len(available_cts) > 3 else available_cts, key='temp_cts')
                
                if not selected_genes:
                    st.info("Enter genes in sidebar.")
                else:
                    if viz_type == "Trajectory Plot":
                        fig = create_temporal_trajectory_plot(temporal_df, selected_genes, temp_species, temp_sample_type,
                                                             selected_temp_cts if selected_temp_cts else None, value_metric)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    elif viz_type == "Temporal Heatmap":
                        hm_ct = st.selectbox("Cell Type for Heatmap", ['All'] + available_cts, key='temp_hm_ct')
                        fig = create_temporal_heatmap(temporal_df, selected_genes, temp_species, temp_sample_type,
                                                     None if hm_ct == 'All' else hm_ct, value_metric)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    elif viz_type == "Multi-Species Comparison":
                        comp_gene = st.selectbox("Gene for comparison", selected_genes, key='temp_comp_gene')
                        fig = create_multi_species_temporal_comparison(temporal_df, comp_gene,
                                                                       selected_temp_cts if selected_temp_cts else None, value_metric)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    elif viz_type == "Timepoint Snapshot":
                        if temp_species:
                            species_temp = temporal_df[temporal_df['species'] == temp_species]
                            if 'sample_type' in species_temp.columns:
                                species_temp = species_temp[species_temp['sample_type'] == temp_sample_type]
                            available_bins = sort_time_bins(species_temp['time_bin'].unique().tolist())
                        else:
                            available_bins = sort_time_bins(temporal_df['time_bin'].unique().tolist())
                        
                        if available_bins:
                            selected_bin = st.selectbox("Select Timepoint", available_bins, key='temp_bin')
                            comp_gene = st.selectbox("Gene for snapshot", selected_genes, key='temp_snap_gene')
                            fig = create_celltype_by_species_temporal(temporal_df, comp_gene, selected_bin,
                                                                      selected_temp_cts if selected_temp_cts else None, value_metric)
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning("No timepoints available.")
            else:
                st.info("Temporal data not available. Generate temporal_expression.parquet.")
        except Exception as e:
            st.error(f"Error in temporal visualization: {str(e)}")
    
    # --------------------------------------------------------------------------
    # Tab: Cross-Species
    # --------------------------------------------------------------------------
    with tabs[5]:
        st.header("Cross-Species Comparison")
        
        if data['ortholog'] is not None:
            ortholog_df = data['ortholog']
            
            species_viz = st.selectbox("Visualization Type",
                ["Expression Bar Chart", "Cross-Species Trajectory", "Species Correlation Heatmap", "Ortholog Scatter Plot"], key='sp_viz')
            
            if not selected_genes:
                st.info("Enter genes in sidebar.")
            else:
                if species_viz == "Expression Bar Chart":
                    fig = create_species_expression_comparison(ortholog_df, selected_genes, selected_celltypes or None)
                    st.plotly_chart(fig, use_container_width=True)
                
                elif species_viz == "Cross-Species Trajectory":
                    if data['temporal'] is not None:
                        temp_cts = data['temporal']['cell_type'].unique().tolist()
                        sel_cts = st.multiselect("Cell Types", temp_cts, default=temp_cts[:3] if len(temp_cts) > 3 else temp_cts, key='xsp_cts')
                        fig = create_cross_species_trajectory(data['temporal'], selected_genes, sel_cts if sel_cts else None, value_metric)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Temporal data required for trajectory comparison.")
                
                elif species_viz == "Species Correlation Heatmap":
                    if data['species_comparison'] is not None:
                        fig = create_species_correlation_heatmap(data['species_comparison'], selected_genes)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Species comparison data not available.")
                
                elif species_viz == "Ortholog Scatter Plot":
                    sc1, sc2, sc3 = st.columns(3)
                    available_sp = ortholog_df['species'].unique().tolist()
                    with sc1:
                        scatter_gene = st.selectbox("Gene", selected_genes, key='scat_gene')
                    with sc2:
                        sp_x = st.selectbox("Species X", available_sp, index=0, key='sp_x')
                    with sc3:
                        other_sp = [s for s in available_sp if s != sp_x]
                        sp_y = st.selectbox("Species Y", other_sp, index=0 if other_sp else 0, key='sp_y')
                    fig = create_ortholog_scatter(ortholog_df, scatter_gene, sp_x, sp_y)
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Ortholog data not available.")
    
    # --------------------------------------------------------------------------
    # Tab: Data Table
    # --------------------------------------------------------------------------
    with tabs[6]:
        if filtered_df.empty:
            st.info("No data.")
        else:
            df_disp = filtered_df.copy()
            df_disp['gene_display'] = df_disp['gene_human'].fillna(df_disp['gene_native'])
            if 'tissue' in df_disp.columns:
                df_disp = df_disp.rename(columns={'tissue': 'Dataset'})
            if 'gene_symbol' in risk_genes.columns:
                df_disp['SFARI_score'] = df_disp['gene_display'].map(risk_genes.set_index('gene_symbol')['gene_score'].to_dict())
            
            default_cols = [c for c in ['gene_display', 'species', 'Dataset', 'cell_type', 'mean_expr', 'pct_expressing', 'n_cells'] if c in df_disp.columns]
            cols = st.multiselect("Columns", df_disp.columns.tolist(), default=default_cols)
            if cols:
                st.dataframe(df_disp[cols].sort_values(['gene_display', 'species']), height=500, use_container_width=True)
                st.download_button("Download CSV", df_disp[cols].to_csv(index=False), "data.csv", "text/csv")
    
    # --------------------------------------------------------------------------
    # Tab: About
    # --------------------------------------------------------------------------
    with tabs[7]:
        st.markdown("""
## About SFARI Gene Expression Explorer

Cross-species single-cell RNA-seq browser for neurodevelopmental gene expression.

### Features
- **Data Overview**: Batch correction methodology & variance partition
- **UMAP**: Integrated cell visualization  
- **Heatmaps**: Clustered expression with faceting
- **Dot Plots**: Size=% expressing, Color=mean expression
- **Temporal Dynamics**: Expression across developmental time
- **Cross-Species**: Ortholog-based comparisons

### Datasets
**Human (Brain)**: Bhaduri (2021), Braun (2023), Velmeshev (2019/2023), Zhu (2023), Wang (2025)  
**Human (Organoid)**: He (2024), Wang (2022)  
**Mouse**: La Manno (2021), Jin (2025), Sziraki (2023)  
**Zebrafish**: Raj (2020) | **Drosophila**: Davie (2018)
        """)
    
    st.divider()
    st.markdown('<div style="text-align:center;color:#666;font-size:0.9rem;">SFARI Gene Explorer v2</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()