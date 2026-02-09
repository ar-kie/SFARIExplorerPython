"""
SFARI Gene Expression Explorer v4
Robust version: Genes first, per-tab filters.
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
from typing import Optional, List, Dict
import re

# =============================================================================
# Configuration
# =============================================================================

st.set_page_config(page_title="SFARI Gene Explorer", page_icon="🧬", layout="wide", initial_sidebar_state="expanded")

# Plotly config for mobile-friendly controls
PLOTLY_CONFIG = {
    'displayModeBar': True,
    'displaylogo': False,
    'modeBarButtonsToAdd': ['zoom2d', 'pan2d', 'zoomIn2d', 'zoomOut2d', 'autoScale2d', 'resetScale2d'],
    'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
    'scrollZoom': True,
    'responsive': True,
}

st.markdown("""
<style>
    .main .block-container { padding-top: 1rem; }
    h1 { color: #1f4e79; }
    div[data-testid="stMetricValue"] { font-size: 1.5rem; }
    .gene-box { background: #f0f8ff; padding: 1rem; border-radius: 8px; border-left: 4px solid #1f4e79; margin: 0.5rem 0; }
    /* Make plotly modebar always visible on mobile */
    .modebar { display: flex !important; }
    .modebar-group { display: flex !important; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# Color Palettes & Constants
# =============================================================================

SPECIES_COLORS = {'Human': '#e41a1c', 'Mouse': '#377eb8', 'Zebrafish': '#4daf4a', 'Drosophila': '#984ea3'}

CELLTYPE_COLORS = {
    'Excitatory Neurons': '#e41a1c', 'Inhibitory Neurons': '#377eb8',
    'Neural Progenitors & Stem Cells': '#4daf4a', 'Astrocytes': '#984ea3',
    'Oligodendrocyte Lineage': '#ff7f00', 'Microglia & Macrophages': '#8B4513',
    'Endothelial & Vascular Cells': '#a65628', 'Endothelial & Vascular': '#a65628',
    'Other Glia & Support': '#f781bf', 'Neurons (unspecified)': '#999999',
    'Fibroblast / Mesenchymal': '#66c2a5', 'Early Embryonic / Germ Layers': '#fc8d62'
}

DATASET_COLORS = {
    'He (2024)': '#e41a1c', 'Bhaduri (2021)': '#377eb8', 'Braun (2023)': '#4daf4a',
    'Velmeshev (2023)': '#984ea3', 'Velmeshev (2019)': '#ff7f00', 'Zhu (2023)': '#c9b400',
    'Wang (2025)': '#a65628', 'Wang (2022)': '#f781bf', 'La Manno (2021)': '#66c2a5',
    'Jin (2025)': '#fc8d62', 'Sziraki (2023)': '#8da0cb', 'Raj (2020)': '#e78ac3',
    'Davie (2018)': '#a6d854',
}

TIME_BIN_ORDER = {
    'Early fetal (GW<10)': 1, 'Mid fetal (GW10-20)': 2, 'Late fetal (GW20-40)': 3,
    'Infant (0-2y)': 4, 'Child (2-12y)': 5, 'Adolescent (12-18y)': 6, 'Adult (18+y)': 7,
    '0-30 days': 11, '31-60 days': 12, '61-90 days': 13, '91-120 days': 14, '>120 days': 15,
    'Early embryo (E<12)': 21, 'Mid embryo (E12-16)': 22, 'Late embryo (E16-20)': 23,
    'Neonatal (P0-P30)': 24, 'Juvenile (1-3mo)': 25, 'Adult (3-12mo)': 26, 'Aged (>12mo)': 27,
    '0-24 hpf': 31, '24-48 hpf': 32, '48-72 hpf': 33, '72-120 hpf (5dpf)': 34, '>5 dpf': 35,
    '0-1 day': 41, '1-7 days': 42, '7-30 days': 43, '>30 days': 44,
}

SAMPLE_TYPE_DISPLAY = {'in_vivo': 'Brain (ex-vivo)', 'organoid': 'Organoid (in-vitro)'}
GENE_SYMBOLS = ['circle', 'square', 'diamond', 'cross', 'x', 'triangle-up', 'triangle-down', 'star']

def sort_time_bins(bins):
    return sorted([b for b in bins if b], key=lambda x: TIME_BIN_ORDER.get(x, 99))

def get_color_palette(values, palette_type='auto'):
    if not values: return {}
    if palette_type == 'species': return {v: SPECIES_COLORS.get(v, '#999') for v in values}
    if palette_type == 'cell_type': return {v: CELLTYPE_COLORS.get(v, '#999') for v in values}
    if palette_type == 'dataset': return {v: DATASET_COLORS.get(v, '#999') for v in values}
    preset = ['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00','#8B4513','#a65628','#f781bf','#999999','#66c2a5','#fc8d62','#8da0cb']
    return {v: preset[i % len(preset)] for i, v in enumerate(values)}

# =============================================================================
# Data Loading
# =============================================================================

@st.cache_data(ttl=3600)
def load_data(data_dir="data"):
    data = {}
    try:
        data['expression'] = pd.read_parquet(f"{data_dir}/expression_summaries.parquet")
        data['cellmeta'] = pd.read_parquet(f"{data_dir}/celltype_meta.parquet")
        data['gene_map'] = pd.read_parquet(f"{data_dir}/gene_map.parquet")
        data['risk_genes'] = pd.read_parquet(f"{data_dir}/risk_genes.parquet")
        if 'gene-symbol' in data['risk_genes'].columns:
            data['risk_genes'] = data['risk_genes'].rename(columns={'gene-symbol': 'gene_symbol', 'gene-score': 'gene_score'})
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None
    
    for key, fn in [('umap', 'umap_subsample.parquet'), ('vp_summary', 'variance_partition_summary.parquet'),
                    ('dataset_info', 'dataset_info.parquet'), ('batch_correction', 'batch_correction_info.parquet'),
                    ('summary_stats', 'summary_statistics.parquet'), ('temporal', 'temporal_expression.parquet'),
                    ('ortholog', 'ortholog_expression.parquet'), ('species_comparison', 'species_comparison.parquet')]:
        try: data[key] = pd.read_parquet(f"{data_dir}/{fn}")
        except: data[key] = None
    return data

def get_unique(df, col):
    if df is None or df.empty or col not in df.columns: return []
    return sorted(df[col].dropna().unique().tolist())

def parse_genes(text):
    if not text or not text.strip(): return []
    return [g.strip().upper() for g in re.split(r'[,\s;]+', text.strip()) if g.strip()]

def filter_by_genes(df, genes):
    """Filter dataframe by genes only. Safe, returns empty on error."""
    if df is None or df.empty or not genes:
        return pd.DataFrame()
    try:
        mask = (df['gene_human'].fillna('').str.upper().isin(genes) | 
               df['gene_native'].fillna('').str.upper().isin(genes))
        return df[mask].copy()
    except:
        return pd.DataFrame()

def filter_df(df, species=None, datasets=None, cell_types=None):
    """Apply additional filters. Safe, returns input on error."""
    if df is None or df.empty:
        return df
    try:
        result = df.copy()
        if species: result = result[result['species'].isin(species)]
        if datasets: result = result[result['tissue'].isin(datasets)]
        if cell_types: result = result[result['cell_type'].isin(cell_types)]
        return result
    except:
        return df

# =============================================================================
# Visualization Functions
# =============================================================================

def create_heatmap(df, value_col='mean_expr', scale_rows=True, split_by=None, annotation_col=None, 
                   cluster_rows=True, cluster_cols=True):
    """Create heatmap. Expects pre-filtered data with genes."""
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No data to display", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False, font=dict(size=16))
        return fig
    
    df = df.copy()
    df['gene'] = df['gene_human'].fillna(df['gene_native'])
    df['col_key'] = df['species'] + '|' + df['tissue'] + '|' + df['cell_type']
    
    pivot = df.pivot_table(index='gene', columns='col_key', values=value_col, aggfunc='mean')
    if pivot.empty: 
        fig = go.Figure()
        fig.add_annotation(text="No data to display", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
        return fig
    
    # Apply Z-score scaling if requested
    if scale_rows and pivot.shape[0] > 0:
        means, stds = pivot.mean(axis=1), pivot.std(axis=1).replace(0, 1)
        pivot = pivot.sub(means, axis=0).div(stds, axis=0).clip(-3, 3)
        zmin, zmax, zmid = -3, 3, 0
        cbar_title = 'Z-score'
        colorscale = 'RdBu_r'
    else:
        # Use actual data range for non-scaled
        zmin, zmax = pivot.min().min(), pivot.max().max()
        zmid = (zmin + zmax) / 2
        cbar_title = 'Mean Expr' if value_col == 'mean_expr' else '% Expressing'
        colorscale = 'Viridis'
    
    col_meta = pd.DataFrame([{'col_key': c, 'species': c.split('|')[0], 'dataset': c.split('|')[1], 
                              'cell_type': c.split('|')[2]} for c in pivot.columns]).set_index('col_key')
    
    if cluster_rows and pivot.shape[0] > 1:
        try:
            link = linkage(pdist(pivot.fillna(0).values), method='average')
            pivot = pivot.iloc[leaves_list(link)]
        except: pass
    
    splits = [None]
    if split_by and split_by in col_meta.columns:
        splits = [s for s in col_meta[split_by].unique() if pd.notna(s)]
    
    matrices, metas = [], []
    for sv in splits:
        cols = col_meta[col_meta[split_by] == sv].index.tolist() if sv else col_meta.index.tolist()
        sub = pivot[[c for c in cols if c in pivot.columns]]
        if sub.empty: continue
        sub_meta = col_meta.loc[sub.columns]
        if cluster_cols and sub.shape[1] > 1:
            try:
                link = linkage(pdist(sub.fillna(0).values.T), method='average')
                sub, sub_meta = sub.iloc[:, leaves_list(link)], sub_meta.iloc[leaves_list(link)]
            except: pass
        matrices.append(sub)
        metas.append(sub_meta)
    
    if not matrices:
        fig = go.Figure()
        fig.add_annotation(text="No data after filtering", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
        return fig
    
    n_splits = len(matrices)
    widths = [m.shape[1] for m in matrices]
    col_widths = [w/sum(widths) for w in widths]
    
    has_anno = annotation_col and annotation_col in col_meta.columns
    fig = make_subplots(rows=2 if has_anno else 1, cols=n_splits, column_widths=col_widths,
                       row_heights=[0.03, 0.97] if has_anno else [1.0],
                       horizontal_spacing=0.02, vertical_spacing=0.01,
                       subplot_titles=[str(s) for s in splits[:n_splits]] if n_splits > 1 else None)
    
    if has_anno:
        ptype = 'species' if annotation_col == 'species' else 'cell_type' if annotation_col == 'cell_type' else 'dataset'
        anno_colors = get_color_palette(col_meta[annotation_col].unique().tolist(), ptype)
    
    cbar_added = False
    for idx, (mat, meta) in enumerate(zip(matrices, metas)):
        col_idx = idx + 1
        labels = [f"{meta['dataset'].iloc[i]}\n{meta['cell_type'].iloc[i]}" for i in range(len(meta))]
        
        if has_anno:
            vals = meta[annotation_col].tolist()
            colors = [anno_colors.get(v, '#999') for v in vals]
            fig.add_trace(go.Heatmap(z=[[i for i in range(len(vals))]], x=labels,
                         colorscale=[[i/max(len(vals)-1,1), colors[i]] for i in range(len(vals))],
                         showscale=False, hoverinfo='skip'), row=1, col=col_idx)
        
        hm_row = 2 if has_anno else 1
        fig.add_trace(go.Heatmap(z=mat.values, x=labels, y=mat.index.tolist(),
                     colorscale=colorscale, zmid=zmid if scale_rows else None, zmin=zmin, zmax=zmax, showscale=not cbar_added,
                     colorbar=dict(title=cbar_title, thickness=12, len=0.6) if not cbar_added else None),
                     row=hm_row, col=col_idx)
        cbar_added = True
    
    height = max(400, 60 + len(pivot) * 14)
    fig.update_layout(height=height, margin=dict(l=200, r=80, t=80, b=100), showlegend=False)
    
    for i in range(1, n_splits + 1):
        hm_row = 2 if has_anno else 1
        if has_anno:
            fig.update_xaxes(showticklabels=False, row=1, col=i)
            fig.update_yaxes(showticklabels=False, row=1, col=i)
        fig.update_xaxes(tickangle=45, tickfont=dict(size=8), row=hm_row, col=i)
        fig.update_yaxes(autorange='reversed', showticklabels=(i==1), tickfont=dict(size=9), row=hm_row, col=i)
    
    if has_anno:
        for v, c in anno_colors.items():
            fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(size=10, color=c), name=str(v), showlegend=True))
        fig.update_layout(legend=dict(orientation='v', y=0.5, x=-0.12, xanchor='right', title=dict(text=annotation_col.replace('_',' ').title())), showlegend=True)
    
    return fig

def create_dotplot(df, group_by='cell_type'):
    """Create dot plot. Expects pre-filtered data with genes."""
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No data to display", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
        return fig
    
    df = df.copy()
    df['gene'] = df['gene_human'].fillna(df['gene_native'])
    
    display_col = 'Dataset' if group_by == 'tissue' else group_by.replace('_', ' ').title()
    if group_by == 'tissue':
        df['Dataset'] = df['tissue']
        group_by = 'Dataset'
    
    agg = df.groupby(['gene', group_by]).agg({'pct_expressing': 'mean', 'mean_expr': 'mean'}).reset_index()
    agg['size'] = agg['pct_expressing'] * 25 + 5
    
    n_genes = df['gene'].nunique()
    fig = px.scatter(agg, x=group_by, y='gene', size='size', color='mean_expr',
                    color_continuous_scale='Viridis', labels={'mean_expr': 'Mean Expr', 'gene': 'Gene'})
    fig.update_layout(height=max(350, 40 + n_genes * 22), xaxis_tickangle=45, yaxis=dict(autorange='reversed'))
    return fig

# =============================================================================
# Temporal Functions
# =============================================================================

def create_temporal_trajectory(temporal_df, genes, species, sample_type, cell_types, value_col='mean_expr'):
    """Trajectory plot: colors = cell types, symbols = genes."""
    if temporal_df is None or temporal_df.empty or not genes:
        fig = go.Figure()
        fig.add_annotation(text="No data available", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
        return fig
    
    try:
        df = temporal_df[temporal_df['species'] == species].copy()
        if 'sample_type' in df.columns and sample_type:
            df = df[df['sample_type'] == sample_type]
        
        df = df[df['gene_human'].fillna('').str.upper().isin(genes)]
        if cell_types:
            df = df[df['cell_type'].isin(cell_types)]
        
        if df.empty:
            fig = go.Figure()
            fig.add_annotation(text=f"No data for selected genes in {species}", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
            return fig
        
        bins = sort_time_bins(df['time_bin'].dropna().unique().tolist())
        if not bins:
            fig = go.Figure()
            fig.add_annotation(text="No timepoints available", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
            return fig
        
        fig = go.Figure()
        ct_colors = get_color_palette(df['cell_type'].unique().tolist(), 'cell_type')
        gene_list = df['gene_human'].unique().tolist()
        gene_syms = {g: GENE_SYMBOLS[i % len(GENE_SYMBOLS)] for i, g in enumerate(gene_list)}
        
        for ct in df['cell_type'].unique():
            ct_data = df[df['cell_type'] == ct]
            for gene in gene_list:
                g_data = ct_data[ct_data['gene_human'] == gene]
                if g_data.empty: continue
                
                agg = g_data.groupby('time_bin')[value_col].mean().reset_index()
                agg['order'] = agg['time_bin'].map(lambda x: TIME_BIN_ORDER.get(x, 99))
                agg = agg.sort_values('order')
                
                if len(agg) == 0: continue
                
                show_leg = (gene == gene_list[0])
                fig.add_trace(go.Scatter(
                    x=agg['time_bin'], y=agg[value_col], mode='lines+markers',
                    name=ct if show_leg else None, legendgroup=ct, showlegend=show_leg,
                    line=dict(color=ct_colors.get(ct, '#999'), width=2),
                    marker=dict(size=9, symbol=gene_syms.get(gene, 'circle'), color=ct_colors.get(ct, '#999')),
                    hovertemplate=f"Gene: {gene}<br>Cell: {ct}<br>Time: %{{x}}<br>Expr: %{{y:.2f}}<extra></extra>"
                ))
        
        if len(gene_list) > 1:
            sym_text = "Symbols: " + ", ".join([f"{g}({gene_syms[g]})" for g in gene_list[:5]])
            fig.add_annotation(text=sym_text, x=0.5, y=-0.18, xref="paper", yref="paper", showarrow=False, font=dict(size=9))
        
        sample_disp = SAMPLE_TYPE_DISPLAY.get(sample_type, sample_type)
        fig.update_layout(
            title=f"{species} {sample_disp} - Temporal Expression",
            xaxis_title="Developmental Stage", yaxis_title="Mean Expression",
            height=480, legend=dict(orientation='v', y=1, x=1.02, xanchor='left'),
            margin=dict(b=80, r=140)
        )
        return fig
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(text=f"Error: {str(e)}", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
        return fig

def create_temporal_heatmap(temporal_df, genes, species, sample_type, cell_type, value_col='mean_expr'):
    """Temporal heatmap."""
    if temporal_df is None or temporal_df.empty or not genes:
        fig = go.Figure()
        fig.add_annotation(text="No data available", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
        return fig
    
    try:
        df = temporal_df[temporal_df['species'] == species].copy()
        if 'sample_type' in df.columns and sample_type:
            df = df[df['sample_type'] == sample_type]
        
        df = df[df['gene_human'].fillna('').str.upper().isin(genes)]
        if cell_type and cell_type != 'All':
            df = df[df['cell_type'] == cell_type]
        
        if df.empty:
            fig = go.Figure()
            fig.add_annotation(text="No data for selected genes", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
            return fig
        
        agg = df.groupby(['gene_human', 'time_bin'])[value_col].mean().reset_index()
        pivot = agg.pivot(index='gene_human', columns='time_bin', values=value_col)
        
        if pivot.empty:
            fig = go.Figure()
            fig.add_annotation(text="No data to display", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
            return fig
        
        sorted_cols = sort_time_bins(pivot.columns.tolist())
        pivot = pivot[[c for c in sorted_cols if c in pivot.columns]]
        
        means, stds = pivot.mean(axis=1), pivot.std(axis=1).replace(0, 1)
        pivot = pivot.sub(means, axis=0).div(stds, axis=0).clip(-3, 3)
        
        fig = go.Figure(go.Heatmap(
            z=pivot.values, x=pivot.columns.tolist(), y=pivot.index.tolist(),
            colorscale='RdBu_r', zmid=0, zmin=-3, zmax=3, colorbar=dict(title='Z-score')
        ))
        
        sample_disp = SAMPLE_TYPE_DISPLAY.get(sample_type, sample_type)
        ct_str = f" ({cell_type})" if cell_type and cell_type != 'All' else ""
        fig.update_layout(
            title=f"{species} {sample_disp} - Temporal Heatmap{ct_str}",
            xaxis_title="Time", yaxis_title="Gene",
            height=max(350, 40 + len(pivot) * 18), xaxis_tickangle=45, yaxis=dict(autorange='reversed')
        )
        return fig
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(text=f"Error: {str(e)}", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
        return fig

def create_multi_species_comparison(temporal_df, gene, cell_types, value_col='mean_expr'):
    """Multi-species temporal comparison."""
    if temporal_df is None or temporal_df.empty or not gene:
        fig = go.Figure()
        fig.add_annotation(text="Select a gene", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
        return fig
    
    try:
        df = temporal_df[temporal_df['gene_human'].fillna('').str.upper() == gene.upper()].copy()
        if cell_types:
            df = df[df['cell_type'].isin(cell_types)]
        
        if df.empty:
            fig = go.Figure()
            fig.add_annotation(text=f"No data for {gene}", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
            return fig
        
        if 'sample_type' in df.columns:
            df['group'] = df.apply(lambda x: f"Human ({SAMPLE_TYPE_DISPLAY.get(x['sample_type'], x['sample_type']).split('(')[0].strip()})" 
                                   if x['species'] == 'Human' else x['species'], axis=1)
        else:
            df['group'] = df['species']
        
        groups = sorted(df['group'].unique().tolist())
        n = len(groups)
        
        fig = make_subplots(rows=1, cols=n, subplot_titles=groups, shared_yaxes=True, horizontal_spacing=0.05)
        ct_colors = get_color_palette(df['cell_type'].unique().tolist(), 'cell_type')
        legend_added = set()
        
        for i, grp in enumerate(groups):
            g_data = df[df['group'] == grp]
            
            for ct in g_data['cell_type'].unique():
                ct_data = g_data[g_data['cell_type'] == ct]
                agg = ct_data.groupby('time_bin')[value_col].mean().reset_index()
                agg['order'] = agg['time_bin'].map(lambda x: TIME_BIN_ORDER.get(x, 99))
                agg = agg.sort_values('order')
                
                show_leg = ct not in legend_added
                if show_leg: legend_added.add(ct)
                
                fig.add_trace(go.Scatter(
                    x=agg['time_bin'], y=agg[value_col], mode='lines+markers',
                    name=ct if show_leg else None, legendgroup=ct, showlegend=show_leg,
                    marker=dict(size=7, color=ct_colors.get(ct, '#999')),
                    line=dict(color=ct_colors.get(ct, '#999'), width=2)
                ), row=1, col=i+1)
            
            fig.update_xaxes(tickangle=45, row=1, col=i+1)
        
        fig.update_layout(
            title=f"Temporal Expression of {gene.upper()} Across Species",
            height=450, legend=dict(orientation='v', y=1, x=1.02, xanchor='left', title='Cell Type'),
            margin=dict(b=100, r=160)
        )
        fig.update_yaxes(title_text="Mean Expression", row=1, col=1)
        return fig
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(text=f"Error: {str(e)}", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
        return fig

def create_timepoint_snapshot(temporal_df, gene, time_bin, cell_types, value_col='mean_expr'):
    """Bar chart at one timepoint."""
    if temporal_df is None or temporal_df.empty or not gene or not time_bin:
        fig = go.Figure()
        fig.add_annotation(text="Select gene and timepoint", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
        return fig
    
    try:
        df = temporal_df.copy()
        df = df[df['gene_human'].fillna('').str.upper() == gene.upper()]
        df = df[df['time_bin'] == time_bin]
        if cell_types:
            df = df[df['cell_type'].isin(cell_types)]
        
        if df.empty:
            fig = go.Figure()
            fig.add_annotation(text=f"No data for {gene} at {time_bin}", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
            return fig
        
        if 'sample_type' in df.columns:
            df['group'] = df.apply(lambda x: f"Human ({SAMPLE_TYPE_DISPLAY.get(x['sample_type'], x['sample_type']).split('(')[0].strip()})" 
                                   if x['species'] == 'Human' else x['species'], axis=1)
        else:
            df['group'] = df['species']
        
        agg = df.groupby(['group', 'cell_type'])[value_col].mean().reset_index()
        
        fig = px.bar(agg, x='cell_type', y=value_col, color='group', barmode='group',
                    color_discrete_map={**SPECIES_COLORS, 'Human (Brain)': '#e41a1c', 'Human (Organoid)': '#fbb4ae'},
                    title=f"{gene.upper()} at {time_bin}")
        fig.update_layout(height=400, xaxis_tickangle=45, legend=dict(orientation='v', y=1, x=1.02, xanchor='left'))
        return fig
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(text=f"Error: {str(e)}", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
        return fig

# =============================================================================
# Cross-Species & Overview Functions
# =============================================================================

def create_species_bar(ortholog_df, genes, cell_types):
    if ortholog_df is None or ortholog_df.empty or not genes:
        fig = go.Figure()
        fig.add_annotation(text="No data", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
        return fig
    
    try:
        df = ortholog_df[ortholog_df['gene_human'].fillna('').str.upper().isin(genes)].copy()
        if cell_types:
            df = df[df['cell_type'].isin(cell_types)]
        
        if df.empty:
            fig = go.Figure()
            fig.add_annotation(text="No data for selected genes", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
            return fig
        
        agg = df.groupby(['gene_human', 'species'])['mean_expr'].mean().reset_index()
        fig = px.bar(agg, x='gene_human', y='mean_expr', color='species', barmode='group',
                    color_discrete_map=SPECIES_COLORS, title="Cross-Species Expression")
        fig.update_layout(height=400, xaxis_tickangle=45, legend=dict(orientation='h', y=1.02, x=0.5, xanchor='center'))
        return fig
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(text=f"Error: {str(e)}", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
        return fig

def create_correlation_heatmap(species_comparison_df, genes):
    if species_comparison_df is None or species_comparison_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="Species comparison data not available", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
        return fig
    
    if not genes:
        fig = go.Figure()
        fig.add_annotation(text="Select genes", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
        return fig
    
    try:
        df = species_comparison_df[species_comparison_df['gene_human'].fillna('').str.upper().isin(genes)].copy()
        
        if df.empty:
            avail = species_comparison_df['gene_human'].unique()[:10].tolist()
            fig = go.Figure()
            fig.add_annotation(text=f"No data. Try: {', '.join(avail)}", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
            return fig
        
        df['pair'] = df['species_1'] + ' vs ' + df['species_2']
        pivot = df.pivot_table(index='gene_human', columns='pair', values='expression_correlation', aggfunc='mean')
        
        fig = go.Figure(go.Heatmap(z=pivot.values, x=pivot.columns.tolist(), y=pivot.index.tolist(),
                                   colorscale='RdBu', zmid=0, zmin=-1, zmax=1, colorbar=dict(title='ρ')))
        fig.update_layout(title="Cross-Species Correlation", height=max(300, 40 + len(pivot) * 18),
                         xaxis_tickangle=45, yaxis=dict(autorange='reversed'))
        return fig
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(text=f"Error: {str(e)}", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
        return fig

def create_ortholog_scatter(ortholog_df, gene, sp_x, sp_y):
    if ortholog_df is None or ortholog_df.empty or not gene:
        fig = go.Figure()
        fig.add_annotation(text="Select a gene", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
        return fig
    
    try:
        df = ortholog_df[ortholog_df['gene_human'].fillna('').str.upper() == gene.upper()].copy()
        
        dx = df[df['species'] == sp_x][['cell_type', 'mean_expr']].rename(columns={'mean_expr': 'x'})
        dy = df[df['species'] == sp_y][['cell_type', 'mean_expr']].rename(columns={'mean_expr': 'y'})
        merged = dx.merge(dy, on='cell_type')
        
        if merged.empty:
            fig = go.Figure()
            fig.add_annotation(text=f"No common cell types", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
            return fig
        
        corr = spearmanr(merged['x'], merged['y'])[0] if len(merged) >= 3 else float('nan')
        
        fig = px.scatter(merged, x='x', y='y', color='cell_type', color_discrete_map=CELLTYPE_COLORS,
                        title=f"{gene.upper()}: {sp_x} vs {sp_y}")
        
        max_val = max(merged['x'].max(), merged['y'].max())
        fig.add_trace(go.Scatter(x=[0, max_val], y=[0, max_val], mode='lines', 
                                line=dict(dash='dash', color='gray'), showlegend=False))
        
        if not np.isnan(corr):
            fig.add_annotation(text=f"ρ = {corr:.3f}", x=0.95, y=0.05, xref="paper", yref="paper", showarrow=False)
        
        fig.update_traces(marker=dict(size=11))
        fig.update_layout(height=450, xaxis_title=f'{sp_x} Expression', yaxis_title=f'{sp_y} Expression',
                         legend=dict(orientation='v', y=1, x=1.02, xanchor='left'))
        return fig
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(text=f"Error: {str(e)}", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
        return fig

def create_variance_barplot(vp_summary):
    if vp_summary is None or vp_summary.empty:
        fig = go.Figure()
        fig.add_annotation(text="No data", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
        return fig
    
    colors = {'Before Correction': '#ff7f0e', 'After Correction': '#1f77b4'}
    fig = go.Figure()
    for stage in ['Before Correction', 'After Correction']:
        sd = vp_summary[vp_summary['stage'] == stage]
        fig.add_trace(go.Bar(name=stage, x=sd['variable'], y=sd['variance_explained'], marker_color=colors[stage],
                            text=[f"{v:.1f}%" for v in sd['variance_explained']], textposition='outside'))
    
    fig.update_layout(title="Variance Explained", barmode='group', height=380,
                     legend=dict(orientation='h', y=1.02, x=0.5, xanchor='center'),
                     yaxis=dict(range=[0, vp_summary['variance_explained'].max() * 1.15]))
    return fig

def create_variance_change(vp_summary):
    if vp_summary is None or vp_summary.empty:
        fig = go.Figure()
        fig.add_annotation(text="No data", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
        return fig
    
    before = vp_summary[vp_summary['stage'] == 'Before Correction'].set_index('variable')['variance_explained']
    after = vp_summary[vp_summary['stage'] == 'After Correction'].set_index('variable')['variance_explained']
    common = before.index.intersection(after.index)
    changes = [(v, after[v] - before[v]) for v in common]
    
    colors = ['#28a745' if (v == 'dataset' and c < 0) or (v != 'dataset' and v != 'Residuals' and c >= 0) else '#dc3545' if v != 'Residuals' else '#6c757d' 
              for v, c in changes]
    
    fig = go.Figure(go.Bar(x=[v for v, _ in changes], y=[c for _, c in changes], marker_color=colors,
                          text=[f"{c:+.1f}%" for _, c in changes], textposition='outside'))
    fig.update_layout(title="Change After Correction", height=350, yaxis=dict(zeroline=True, zerolinewidth=2, zerolinecolor='black'))
    return fig

# =============================================================================
# Main App - GENES FIRST, PER-TAB FILTERS
# =============================================================================

def main():
    st.title("🧬 SFARI Gene Expression Explorer")
    st.caption("Cross-species single-cell RNA-seq browser for neurodevelopmental gene expression")
    
    data = load_data()
    if data is None:
        st.error("Failed to load data.")
        st.stop()
    
    expr_df, risk_genes = data['expression'], data['risk_genes']
    
    # ==========================================================================
    # Sidebar - ONLY GENE SELECTION
    # ==========================================================================
    with st.sidebar:
        st.header("🧬 Step 1: Select Genes")
        
        gene_preset = st.selectbox("Quick Sets", ["Custom", "SFARI Score 1", "SFARI Score 2", "SFARI Syndromic", "Top Variable"])
        
        # Build preset gene list based on selection
        if gene_preset == "SFARI Score 1":
            default_genes = ", ".join(risk_genes[risk_genes['gene_score'] == 1]['gene_symbol'].dropna().tolist())
        elif gene_preset == "SFARI Score 2":
            default_genes = ", ".join(risk_genes[risk_genes['gene_score'] == 2]['gene_symbol'].dropna().tolist())
        elif gene_preset == "SFARI Syndromic":
            default_genes = ", ".join(risk_genes[risk_genes.get('syndromic', pd.Series([0])) == 1]['gene_symbol'].dropna().tolist()) if 'syndromic' in risk_genes.columns else ""
        elif gene_preset == "Top Variable":
            default_genes = ", ".join(expr_df.groupby('gene_human')['mean_expr'].var().sort_values(ascending=False).head(50).index.tolist())
        else:
            default_genes = ""
        
        # Use a form so Enter key or button submits on mobile
        with st.form(key="gene_form"):
            gene_input = st.text_area(
                "Enter genes (comma or space separated)", 
                value=default_genes,
                height=120,
                placeholder="SHANK3, MECP2, CHD8, SCN2A"
            )
            submitted = st.form_submit_button("🔍 Search Genes", use_container_width=True, type="primary")
        
        input_genes = parse_genes(gene_input)
        
        # Check which genes are in the database
        all_genes_in_db = set(expr_df['gene_human'].dropna().str.upper().unique()) | set(expr_df['gene_native'].dropna().str.upper().unique())
        found_genes = [g for g in input_genes if g in all_genes_in_db]
        not_found_genes = [g for g in input_genes if g not in all_genes_in_db]
        
        selected_genes = found_genes
        
        # Show feedback
        if len(input_genes) > 0:
            if len(found_genes) >= 2:
                st.success(f"✓ {len(found_genes)} genes found")
            elif len(found_genes) == 1:
                st.warning(f"1 gene found (need 2+ for heatmaps)")
            else:
                st.error("No matching genes found")
            
            if not_found_genes:
                st.caption(f"⚠️ Not found: {', '.join(not_found_genes[:5])}{'...' if len(not_found_genes) > 5 else ''}")
        else:
            st.info("Select a preset or enter genes, then tap Search")
        
        st.divider()
        st.subheader("⚙️ Display Options")
        value_metric = st.radio("Value", ['mean_expr', 'pct_expressing'], 
                               format_func=lambda x: "Mean Expression" if x == 'mean_expr' else "% Expressing", horizontal=True)
        scale_rows = st.checkbox("Z-score scaling", value=True)
        
        st.divider()
        st.caption("📌 Filters are in each tab")
    
    # Pre-filter expression data by genes (safe operation)
    gene_filtered_df = filter_by_genes(expr_df, selected_genes)
    
    # Quick stats
    if not gene_filtered_df.empty:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Genes", gene_filtered_df['gene_human'].nunique())
        c2.metric("Species", gene_filtered_df['species'].nunique())
        c3.metric("Datasets", gene_filtered_df['tissue'].nunique())
        c4.metric("Cell Types", gene_filtered_df['cell_type'].nunique())
    
    # ==========================================================================
    # Tabs
    # ==========================================================================
    tabs = st.tabs(["📊 Overview", "🗺️ UMAP", "🔥 Heatmap", "🔵 Dot Plot", "📈 Temporal", "🔬 Cross-Species", "📋 Data", "📚 About"])
    
    # --------------------------------------------------------------------------
    # Overview Tab
    # --------------------------------------------------------------------------
    with tabs[0]:
        st.header("Data Overview")
        if data['summary_stats'] is not None:
            totals = data['summary_stats'][data['summary_stats']['category'] == 'totals'].set_index('label')['value']
            cols = st.columns(4)
            for i, (k, l) in enumerate([('pseudobulk_samples', 'Samples'), ('datasets', 'Datasets'), ('cell_types', 'Cell Types'), ('organisms', 'Species')]):
                if k in totals.index: cols[i].metric(l, f"{int(totals[k]):,}")
        
        if data['dataset_info'] is not None:
            st.subheader("Datasets")
            df_info = data['dataset_info'].copy()
            if 'sample_type' in df_info.columns:
                df_info['sample_type'] = df_info['sample_type'].map(lambda x: SAMPLE_TYPE_DISPLAY.get(x, x))
            st.dataframe(df_info, hide_index=True, use_container_width=True)
        
        if data['vp_summary'] is not None:
            st.subheader("Batch Correction")
            c1, c2 = st.columns(2)
            c1.plotly_chart(create_variance_barplot(data['vp_summary']), use_container_width=True)
            c2.plotly_chart(create_variance_change(data['vp_summary']), use_container_width=True)
    
    # --------------------------------------------------------------------------
    # UMAP Tab
    # --------------------------------------------------------------------------
    with tabs[1]:
        st.header("UMAP Visualization")
        if data['umap'] is not None:
            umap_df = data['umap']
            color_cols = [c for c in umap_df.columns if c not in ['umap_1', 'umap_2', 'cell_id']]
            c1, c2 = st.columns([1, 4])
            with c1:
                color_by = st.selectbox("Color by", color_cols, index=color_cols.index('predicted_labels') if 'predicted_labels' in color_cols else 0)
                pt_size = st.slider("Size", 1, 8, 2)
            with c2:
                cmap = SPECIES_COLORS if color_by == 'organism' else CELLTYPE_COLORS if color_by in ['predicted_labels','cell_type'] else DATASET_COLORS if color_by == 'dataset' else get_color_palette(umap_df[color_by].unique().tolist())
                fig = px.scatter(umap_df, x='umap_1', y='umap_2', color=color_by, color_discrete_map=cmap)
                fig.update_traces(marker=dict(size=pt_size, opacity=0.6))
                fig.update_layout(height=550, legend=dict(font=dict(size=8)))
                st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
        else:
            st.info("UMAP data not available")
    
    # --------------------------------------------------------------------------
    # Heatmap Tab - WITH ITS OWN FILTERS
    # --------------------------------------------------------------------------
    with tabs[2]:
        if len(selected_genes) < 2:
            st.warning("⬅️ Enter at least 2 genes in the sidebar to create a heatmap")
        else:
            st.subheader("Filters")
            hm_c1, hm_c2, hm_c3 = st.columns(3)
            
            with hm_c1:
                avail_species = get_unique(gene_filtered_df, 'species')
                hm_species = st.multiselect("Species", avail_species, default=avail_species, key='hm_sp')
            
            with hm_c2:
                hm_df_sp = filter_df(gene_filtered_df, species=hm_species or None)
                avail_datasets = get_unique(hm_df_sp, 'tissue')
                hm_datasets = st.multiselect("Dataset", avail_datasets, default=[], key='hm_ds')
            
            with hm_c3:
                hm_df_ds = filter_df(hm_df_sp, datasets=hm_datasets or None)
                avail_cts = get_unique(hm_df_ds, 'cell_type')
                hm_celltypes = st.multiselect("Cell Types", avail_cts, default=[], key='hm_ct')
            
            # Apply filters
            hm_df = filter_df(gene_filtered_df, species=hm_species or None, datasets=hm_datasets or None, cell_types=hm_celltypes or None)
            
            st.subheader("Display Options")
            hm_o1, hm_o2, hm_o3, hm_o4 = st.columns(4)
            split_by = {"None": None, "Species": "species", "Dataset": "dataset", "Cell Type": "cell_type"}[hm_o1.selectbox("Split by", ["None", "Species", "Dataset", "Cell Type"])]
            anno_col = {"None": None, "Species": "species", "Dataset": "dataset", "Cell Type": "cell_type"}[hm_o2.selectbox("Annotation", ["None", "Species", "Dataset", "Cell Type"])]
            cluster_rows = hm_o3.checkbox("Cluster rows", value=False)
            cluster_cols = hm_o4.checkbox("Cluster cols", value=False)
            
            fig = create_heatmap(hm_df, value_metric, scale_rows, split_by, anno_col, cluster_rows, cluster_cols)
            st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
    
    # --------------------------------------------------------------------------
    # Dot Plot Tab - WITH ITS OWN FILTERS
    # --------------------------------------------------------------------------
    with tabs[3]:
        if not selected_genes:
            st.warning("⬅️ Enter genes in the sidebar first")
        else:
            st.subheader("Filters")
            dp_c1, dp_c2, dp_c3 = st.columns(3)
            
            with dp_c1:
                avail_species = get_unique(gene_filtered_df, 'species')
                dp_species = st.multiselect("Species", avail_species, default=avail_species, key='dp_sp')
            
            with dp_c2:
                dp_df_sp = filter_df(gene_filtered_df, species=dp_species or None)
                avail_datasets = get_unique(dp_df_sp, 'tissue')
                dp_datasets = st.multiselect("Dataset", avail_datasets, default=[], key='dp_ds')
            
            with dp_c3:
                dp_df_ds = filter_df(dp_df_sp, datasets=dp_datasets or None)
                avail_cts = get_unique(dp_df_ds, 'cell_type')
                dp_celltypes = st.multiselect("Cell Types", avail_cts, default=[], key='dp_ct')
            
            dp_df = filter_df(gene_filtered_df, species=dp_species or None, datasets=dp_datasets or None, cell_types=dp_celltypes or None)
            
            group_by = st.selectbox("Group by", ['cell_type', 'tissue', 'species'], 
                                   format_func=lambda x: 'Dataset' if x == 'tissue' else x.replace('_',' ').title())
            
            fig = create_dotplot(dp_df, group_by)
            st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
    
    # --------------------------------------------------------------------------
    # Temporal Tab - WITH ITS OWN FILTERS
    # --------------------------------------------------------------------------
    with tabs[4]:
        st.header("Temporal Expression Dynamics")
        
        if data['temporal'] is None or data['temporal'].empty:
            st.warning("Temporal data not available. Generate temporal_expression.parquet.")
        elif not selected_genes:
            st.warning("⬅️ Enter genes in the sidebar first")
        else:
            temporal_df = data['temporal']
            
            viz_type = st.selectbox("View", ["Trajectory", "Heatmap", "Multi-Species", "Timepoint Snapshot"], key='temp_viz')
            
            if viz_type != "Multi-Species":
                tc1, tc2 = st.columns(2)
                with tc1:
                    temp_species = st.selectbox("Species", temporal_df['species'].unique().tolist(), key='temp_sp')
                with tc2:
                    temp_sample_type = 'in_vivo'
                    if temp_species == 'Human' and 'sample_type' in temporal_df.columns:
                        st_opts = temporal_df[temporal_df['species'] == 'Human']['sample_type'].dropna().unique().tolist()
                        if st_opts:
                            st_display = [SAMPLE_TYPE_DISPLAY.get(s, s) for s in st_opts]
                            sel = st.selectbox("Sample Type", st_display, key='temp_st')
                            temp_sample_type = st_opts[st_display.index(sel)]
                
                sp_df = temporal_df[temporal_df['species'] == temp_species]
                if 'sample_type' in sp_df.columns:
                    sp_df = sp_df[sp_df['sample_type'] == temp_sample_type]
                avail_temp_cts = sorted(sp_df['cell_type'].unique().tolist())
            else:
                temp_species = None
                temp_sample_type = None
                avail_temp_cts = sorted(temporal_df['cell_type'].unique().tolist())
            
            # Cell type selection - default to first 3
            default_cts = avail_temp_cts[:3] if len(avail_temp_cts) > 3 else avail_temp_cts
            sel_temp_cts = st.multiselect("Cell Types", avail_temp_cts, default=default_cts)
            
            # Handle empty selection
            if not sel_temp_cts:
                st.info("Select at least one cell type to view the plot")
            else:
                # Create plots based on view type
                if viz_type == "Trajectory":
                    try:
                        fig = create_temporal_trajectory(temporal_df, selected_genes, temp_species, temp_sample_type, sel_temp_cts, value_metric)
                        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
                    except Exception as e:
                        st.error(f"Error creating trajectory: {str(e)}")
                
                elif viz_type == "Heatmap":
                    hm_ct = st.selectbox("Cell Type for Heatmap", ['All'] + avail_temp_cts, key='temp_hm_ct')
                    try:
                        fig = create_temporal_heatmap(temporal_df, selected_genes, temp_species, temp_sample_type, 
                                                     hm_ct if hm_ct != 'All' else None, value_metric)
                        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
                    except Exception as e:
                        st.error(f"Error creating heatmap: {str(e)}")
                
                elif viz_type == "Multi-Species":
                    comp_gene = st.selectbox("Gene", selected_genes, key='temp_comp')
                    try:
                        fig = create_multi_species_comparison(temporal_df, comp_gene, sel_temp_cts, value_metric)
                        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
                    except Exception as e:
                        st.error(f"Error creating comparison: {str(e)}")
                
                elif viz_type == "Timepoint Snapshot":
                    sp_df = temporal_df[temporal_df['species'] == temp_species]
                    if 'sample_type' in sp_df.columns:
                        sp_df = sp_df[sp_df['sample_type'] == temp_sample_type]
                    avail_bins = sort_time_bins(sp_df['time_bin'].dropna().unique().tolist())
                
                    if avail_bins:
                        sel_bin = st.selectbox("Timepoint", avail_bins, key='temp_bin')
                        comp_gene = st.selectbox("Gene", selected_genes, key='temp_snap_gene')
                        try:
                            fig = create_timepoint_snapshot(temporal_df, comp_gene, sel_bin, sel_temp_cts, value_metric)
                            st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
                        except Exception as e:
                            st.error(f"Error creating snapshot: {str(e)}")
                    else:
                        st.warning("No timepoints available for this species/sample type")
    
    # --------------------------------------------------------------------------
    # Cross-Species Tab - WITH ITS OWN FILTERS
    # --------------------------------------------------------------------------
    with tabs[5]:
        st.header("Cross-Species Comparison")
        
        if data['ortholog'] is None:
            st.warning("Ortholog data not available.")
        elif not selected_genes:
            st.warning("⬅️ Enter genes in the sidebar first")
        else:
            ortholog_df = data['ortholog']
            
            sp_viz = st.selectbox("View", ["Bar Chart", "Correlation Heatmap", "Scatter Plot"], key='sp_viz')
            
            # Cell type filter for this tab
            avail_cts = get_unique(ortholog_df, 'cell_type')
            sp_celltypes = st.multiselect("Cell Types (optional)", avail_cts, default=[], key='sp_ct')
            
            if sp_viz == "Bar Chart":
                fig = create_species_bar(ortholog_df, selected_genes, sp_celltypes or None)
                st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
            
            elif sp_viz == "Correlation Heatmap":
                fig = create_correlation_heatmap(data['species_comparison'], selected_genes)
                st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
            
            elif sp_viz == "Scatter Plot":
                sc1, sc2, sc3 = st.columns(3)
                avail_sp = ortholog_df['species'].unique().tolist()
                with sc1:
                    scatter_gene = st.selectbox("Gene", selected_genes, key='scat_gene')
                with sc2:
                    sp_x = st.selectbox("Species X", avail_sp, index=0, key='sp_x')
                with sc3:
                    other_sp = [s for s in avail_sp if s != sp_x]
                    sp_y = st.selectbox("Species Y", other_sp, index=0 if other_sp else 0, key='sp_y')
                fig = create_ortholog_scatter(ortholog_df, scatter_gene, sp_x, sp_y)
                st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
    
    # --------------------------------------------------------------------------
    # Data Table Tab
    # --------------------------------------------------------------------------
    with tabs[6]:
        if not selected_genes:
            st.warning("⬅️ Enter genes in the sidebar first")
        elif gene_filtered_df.empty:
            st.info("No data for selected genes")
        else:
            st.subheader("Filters")
            dt_c1, dt_c2, dt_c3 = st.columns(3)
            
            with dt_c1:
                avail_species = get_unique(gene_filtered_df, 'species')
                dt_species = st.multiselect("Species", avail_species, default=avail_species, key='dt_sp')
            
            with dt_c2:
                dt_df_sp = filter_df(gene_filtered_df, species=dt_species or None)
                avail_datasets = get_unique(dt_df_sp, 'tissue')
                dt_datasets = st.multiselect("Dataset", avail_datasets, default=[], key='dt_ds')
            
            with dt_c3:
                dt_df_ds = filter_df(dt_df_sp, datasets=dt_datasets or None)
                avail_cts = get_unique(dt_df_ds, 'cell_type')
                dt_celltypes = st.multiselect("Cell Types", avail_cts, default=[], key='dt_ct')
            
            dt_df = filter_df(gene_filtered_df, species=dt_species or None, datasets=dt_datasets or None, cell_types=dt_celltypes or None)
            
            if not dt_df.empty:
                dt_df = dt_df.copy()
                dt_df['gene'] = dt_df['gene_human'].fillna(dt_df['gene_native'])
                dt_df = dt_df.rename(columns={'tissue': 'Dataset'})
                
                if 'gene_symbol' in risk_genes.columns:
                    dt_df['SFARI'] = dt_df['gene'].map(risk_genes.set_index('gene_symbol')['gene_score'].to_dict())
                
                default_cols = [c for c in ['gene', 'species', 'Dataset', 'cell_type', 'mean_expr', 'pct_expressing', 'n_cells'] if c in dt_df.columns]
                cols = st.multiselect("Columns", dt_df.columns.tolist(), default=default_cols)
                if cols:
                    st.dataframe(dt_df[cols].sort_values(['gene', 'species']), height=450, use_container_width=True)
                    st.download_button("Download CSV", dt_df[cols].to_csv(index=False), "data.csv", "text/csv")
            else:
                st.info("No data after filtering")
    
    # --------------------------------------------------------------------------
    # About Tab - FULL VERSION
    # --------------------------------------------------------------------------
    with tabs[7]:
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

| Dataset | Publication | Journal | Description |
|---------|-------------|---------|-------------|
| **He (2024)** | He et al., 2024 | *Nature* | Human Neural Organoid Cell Atlas (HNOCA) |
| **Bhaduri (2021)** | Bhaduri et al., 2021 | *Nature* | Primary human cortical development |
| **Braun (2023)** | Braun et al., 2023 | *Science* | Human brain cell atlas |
| **Velmeshev (2023)** | Velmeshev et al., 2023 | *Science* | Developing human brain cell types |
| **Velmeshev (2019)** | Velmeshev et al., 2019 | *Science* | Single-cell genomics of ASD brain |
| **Zhu (2023)** | Zhu et al., 2023 | *Science Advances* | Human fetal brain development |
| **Wang (2025)** | Wang et al., 2025 | *Nature* | Human brain development atlas |
| **Wang (2022)** | Wang et al., 2022 | *Nature Communications* | Human cerebral organoids |

### Mouse Datasets

| Dataset | Publication | Journal | Description |
|---------|-------------|---------|-------------|
| **La Manno (2021)** | La Manno et al., 2021 | *Nature* | Mouse brain development atlas |
| **Jin (2025)** | Jin et al., 2025 | *Nature* | Mouse brain cell atlas |
| **Sziraki (2023)** | Sziraki et al., 2023 | *Nature Genetics* | Mouse brain cell types |

### Zebrafish Datasets

| Dataset | Publication | Journal | Description |
|---------|-------------|---------|-------------|
| **Raj (2020)** | Raj et al., 2020 | *Neuroscience* | Zebrafish brain development |

### Drosophila Datasets

| Dataset | Publication | Journal | Description |
|---------|-------------|---------|-------------|
| **Davie (2018)** | Davie et al., 2018 | *Cell* | *Drosophila* brain cell atlas |

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
    
    st.divider()
    st.caption("SFARI Gene Explorer v4 | Built with Streamlit & Plotly")

if __name__ == "__main__":
    main()