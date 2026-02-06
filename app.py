"""
SFARIExplorer
A cross-species gene expression browser for single-cell RNA-seq data.
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

# Configuration
st.set_page_config(page_title="SFARI Gene Explorer", page_icon="🧬", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .stApp { max-width: 100%; }
    .main .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    h1 { color: #1f4e79; font-weight: 600; }
    h2, h3 { color: #2c5282; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { background-color: #f0f4f8; border-radius: 4px 4px 0 0; padding: 8px 16px; }
    .stTabs [aria-selected="true"] { background-color: #2c5282; color: white; }
    div[data-testid="stMetricValue"] { font-size: 1.8rem; font-weight: 600; }
    .info-box { background-color: #e8f4f8; border-left: 4px solid #2c5282; padding: 1rem; margin: 1rem 0; border-radius: 0 8px 8px 0; }
    .success-box { background-color: #e8f8e8; border-left: 4px solid #28a745; padding: 1rem; margin: 1rem 0; border-radius: 0 8px 8px 0; }
</style>
""", unsafe_allow_html=True)

# Color Palettes
SPECIES_COLORS = {'Human': '#e41a1c', 'Mouse': '#377eb8', 'Zebrafish': '#4daf4a', 'Drosophila': '#984ea3'}
CELLTYPE_COLORS = {
    'Excitatory Neurons': '#e41a1c', 'Inhibitory Neurons': '#377eb8', 'Neural Progenitors & Stem Cells': '#4daf4a',
    'Astrocytes': '#984ea3', 'Oligodendrocyte Lineage': '#ff7f00', 'Microglia & Macrophages': '#ffff33',
    'Endothelial & Vascular Cells': '#a65628', 'Endothelial & Vascular': '#a65628', 'Other Glia & Support': '#f781bf',
    'Neurons (unspecified)': '#999999', 'Fibroblast / Mesenchymal': '#66c2a5', 'Early Embryonic / Germ Layers': '#fc8d62'
}
DATASET_COLORS = {
    'He (2024)': '#e41a1c', 'Bhaduri (2021)': '#377eb8', 'Braun (2023)': '#4daf4a', 'Velmeshev (2023)': '#984ea3',
    'Velmeshev (2019)': '#ff7f00', 'Zhu (2023)': '#ffff33', 'Wang (2025)': '#a65628', 'Wang (2022)': '#f781bf',
    'La Manno (2021)': '#66c2a5', 'Jin (2025)': '#fc8d62', 'Sziraki (2023)': '#8da0cb', 'Raj (2020)': '#e78ac3', 'Davie (2018)': '#a6d854',
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

def get_color_palette(values, palette_type='auto'):
    if palette_type == 'species': return {v: SPECIES_COLORS.get(v, '#999999') for v in values}
    elif palette_type == 'cell_type': return {v: CELLTYPE_COLORS.get(v, '#999999') for v in values}
    elif palette_type == 'dataset': return {v: DATASET_COLORS.get(v, '#999999') for v in values}
    elif palette_type == 'time_bin': return {v: TIME_BIN_COLORS.get(v, '#999999') for v in values}
    else:
        preset = ['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00','#ffff33','#a65628','#f781bf','#999999','#66c2a5','#fc8d62','#8da0cb']
        return {v: preset[i % len(preset)] for i, v in enumerate(values)}

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
        st.error(f"Error loading core data: {e}")
        return None
    
    optional = {'umap': 'umap_subsample.parquet', 'vp_summary': 'variance_partition_summary.parquet',
                'vp_by_gene': 'variance_partition_by_gene.parquet', 'dataset_info': 'dataset_info.parquet',
                'batch_correction': 'batch_correction_info.parquet', 'summary_stats': 'summary_statistics.parquet',
                'temporal': 'temporal_expression.parquet', 'ortholog': 'ortholog_expression.parquet',
                'species_comparison': 'species_comparison.parquet'}
    for key, fn in optional.items():
        try: data[key] = pd.read_parquet(f"{data_dir}/{fn}")
        except: data[key] = None
    return data

def get_unique_values(df, column): return sorted(df[column].dropna().unique().tolist())

def filter_expression_data(expr_df, species=None, datasets=None, cell_types=None, genes=None):
    df = expr_df.copy()
    if species: df = df[df['species'].isin(species)]
    if datasets: df = df[df['tissue'].isin(datasets)]
    if cell_types: df = df[df['cell_type'].isin(cell_types)]
    if genes:
        genes_upper = [g.upper() for g in genes]
        df = df[df['gene_native'].str.upper().isin(genes_upper) | df['gene_human'].str.upper().isin(genes_upper)]
    return df

def parse_gene_input(gene_text):
    if not gene_text or not gene_text.strip(): return []
    return [g.strip() for g in re.split(r'[,\s;]+', gene_text.strip()) if g.strip()]

def create_heatmap_matrix(df, value_col='mean_expr', scale_rows=True):
    df = df.copy()
    df['gene_display'] = df['gene_human'].fillna(df['gene_native'])
    df['col_key'] = df.apply(lambda x: f"{x['species']}|{x['tissue']}|{x['cell_type']}", axis=1)
    pivot = df.pivot_table(index='gene_display', columns='col_key', values=value_col, aggfunc='mean')
    if scale_rows and pivot.shape[0] > 0:
        row_means, row_stds = pivot.mean(axis=1), pivot.std(axis=1).replace(0, 1)
        pivot = pivot.sub(row_means, axis=0).div(row_stds, axis=0).clip(-3, 3)
    col_meta = pd.DataFrame([{'col_key': c, 'species': c.split('|')[0], 'dataset': c.split('|')[1], 'cell_type': c.split('|')[2]} for c in pivot.columns]).set_index('col_key').reindex(pivot.columns)
    return pivot, col_meta

# Heatmap visualization
def create_complexheatmap(matrix, col_meta, title="Heatmap", color_scale="RdBu_r", split_by=None, annotation_col=None, cluster_rows=True, cluster_cols=True, row_label_size=9, col_label_size=9, legend_position="bottom"):
    if matrix.empty:
        fig = go.Figure()
        fig.add_annotation(text="No data", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig
    
    if cluster_rows and matrix.shape[0] > 1:
        try:
            mat = matrix.fillna(0).values
            link = linkage(pdist(mat), method='average')
            matrix = matrix.iloc[leaves_list(link)]
        except: pass
    
    split_values = [s for s in col_meta[split_by].unique() if pd.notna(s)] if split_by and split_by in col_meta.columns else [None]
    n_splits = len(split_values)
    split_matrices, split_col_metas, split_widths = [], [], []
    
    for sv in split_values:
        cols = col_meta[col_meta[split_by] == sv].index.tolist() if sv else col_meta.index.tolist()
        sub_matrix = matrix[[c for c in cols if c in matrix.columns]]
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
    
    col_widths = [w/sum(split_widths) for w in split_widths]
    has_anno = annotation_col and annotation_col in col_meta.columns
    fig = make_subplots(rows=2 if has_anno else 1, cols=n_splits, column_widths=col_widths, row_heights=[0.03, 0.97] if has_anno else [1.0], horizontal_spacing=0.02, vertical_spacing=0.01, subplot_titles=[str(s) if s else "" for s in split_values] if n_splits > 1 else None)
    
    if has_anno:
        anno_colors = get_color_palette(col_meta[annotation_col].unique().tolist(), 'species' if annotation_col == 'species' else 'cell_type' if annotation_col == 'cell_type' else 'dataset' if annotation_col == 'dataset' else 'auto')
    
    colorbar_added = False
    for idx, (sub_m, sub_cm) in enumerate(zip(split_matrices, split_col_metas)):
        if sub_m.empty: continue
        col_idx = idx + 1
        if split_by == 'species': col_labels = [f"{d}\n{c}" for d, c in zip(sub_cm['dataset'], sub_cm['cell_type'])]
        elif split_by == 'dataset': col_labels = [f"{s}\n{c}" for s, c in zip(sub_cm['species'], sub_cm['cell_type'])]
        else: col_labels = [f"{d}\n{c}" for d, c in zip(sub_cm['dataset'], sub_cm['cell_type'])]
        
        if has_anno:
            av = sub_cm[annotation_col].tolist()
            fig.add_trace(go.Heatmap(z=[[i for i in range(len(av))]], x=col_labels, colorscale=[[i/max(len(av)-1,1), anno_colors.get(av[i],'#999')] for i in range(len(av))], showscale=False, hoverinfo='skip'), row=1, col=col_idx)
        
        hm_row = 2 if has_anno else 1
        fig.add_trace(go.Heatmap(z=sub_m.values, x=col_labels, y=sub_m.index.tolist(), colorscale=color_scale, zmid=0, zmin=-3, zmax=3, showscale=not colorbar_added, colorbar=dict(title='Z-score', thickness=15, len=0.7) if not colorbar_added else None), row=hm_row, col=col_idx)
        colorbar_added = True
    
    fig.update_layout(title=dict(text=title, x=0.5), height=max(400, 80+len(matrix)*16), margin=dict(l=150, r=80, t=100, b=120), showlegend=False)
    for i in range(1, n_splits+1):
        hm_row = 2 if has_anno else 1
        if has_anno:
            fig.update_xaxes(showticklabels=False, row=1, col=i)
            fig.update_yaxes(showticklabels=False, row=1, col=i)
        fig.update_xaxes(tickangle=45, tickfont=dict(size=col_label_size), row=hm_row, col=i)
        fig.update_yaxes(tickfont=dict(size=row_label_size), autorange='reversed', showticklabels=(i==1), row=hm_row, col=i)
    
    if has_anno:
        for v, c in anno_colors.items():
            fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(size=10, color=c), name=str(v), showlegend=True))
        fig.update_layout(legend=dict(orientation='h', y=-0.15, x=0.5, xanchor='center'), showlegend=True)
    return fig

def create_dotplot(df, genes, group_by='cell_type'):
    df = df.copy()
    df['gene_display'] = df['gene_human'].fillna(df['gene_native'])
    agg = df.groupby(['gene_display', group_by]).agg({'pct_expressing': 'mean', 'mean_expr': 'mean'}).reset_index()
    agg['size_scaled'] = agg['pct_expressing'] * 30 + 5
    fig = px.scatter(agg, x=group_by, y='gene_display', size='size_scaled', color='mean_expr', color_continuous_scale='Viridis', title="Dot Plot")
    fig.update_layout(height=max(400, 50+len(genes)*25), xaxis_tickangle=45, yaxis=dict(autorange='reversed'))
    return fig

def create_temporal_trajectory_plot(temporal_df, genes, species='Human', sample_type='in_vivo', cell_types=None, value_col='mean_expr'):
    df = temporal_df[temporal_df['species'] == species].copy()
    if 'sample_type' in df.columns and sample_type: df = df[df['sample_type'] == sample_type]
    genes_upper = [g.upper() for g in genes]
    df = df[df['gene_human'].str.upper().isin(genes_upper)]
    if cell_types: df = df[df['cell_type'].isin(cell_types)]
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No temporal data", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig
    
    df = df.sort_values('time_order')
    fig = go.Figure()
    gene_colors = get_color_palette(list(df['gene_human'].unique()), 'auto')
    
    for gene in df['gene_human'].unique():
        gd = df[df['gene_human'] == gene]
        for ct in gd['cell_type'].unique():
            cd = gd[gd['cell_type'] == ct]
            agg = cd.groupby(['time_bin', 'time_order']).agg({value_col: 'mean'}).reset_index().sort_values('time_order')
            fig.add_trace(go.Scatter(x=agg['time_bin'], y=agg[value_col], mode='lines+markers', name=f"{gene} - {ct}", line=dict(color=gene_colors[gene]), marker=dict(size=8)))
    
    title_map = {'Human': 'Human Development' if sample_type != 'organoid' else 'Organoid Differentiation', 'Mouse': 'Mouse Development', 'Zebrafish': 'Zebrafish Development', 'Drosophila': 'Drosophila Aging'}
    fig.update_layout(title=f"Temporal Expression - {title_map.get(species, species)}", xaxis_title="Developmental Stage", yaxis_title="Mean Expression", height=500, legend=dict(orientation='h', y=-0.3, x=0.5, xanchor='center'))
    return fig

def create_temporal_heatmap(temporal_df, genes, species='Human', sample_type='in_vivo', cell_type=None, value_col='mean_expr'):
    df = temporal_df[temporal_df['species'] == species].copy()
    if 'sample_type' in df.columns and sample_type: df = df[df['sample_type'] == sample_type]
    genes_upper = [g.upper() for g in genes]
    df = df[df['gene_human'].str.upper().isin(genes_upper)]
    if cell_type: df = df[df['cell_type'] == cell_type]
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No data", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig
    
    agg = df.groupby(['gene_human', 'time_bin', 'time_order']).agg({value_col: 'mean'}).reset_index()
    pivot = agg.pivot_table(index='gene_human', columns='time_bin', values=value_col, aggfunc='mean')
    time_order = agg.groupby('time_bin')['time_order'].first().to_dict()
    pivot = pivot[[c for c in sorted(pivot.columns, key=lambda x: time_order.get(x, 99))]]
    
    row_means, row_stds = pivot.mean(axis=1), pivot.std(axis=1).replace(0, 1)
    pivot_scaled = pivot.sub(row_means, axis=0).div(row_stds, axis=0).clip(-3, 3)
    
    fig = go.Figure(go.Heatmap(z=pivot_scaled.values, x=pivot_scaled.columns.tolist(), y=pivot_scaled.index.tolist(), colorscale='RdBu_r', zmid=0, zmin=-3, zmax=3, colorbar=dict(title='Z-score')))
    fig.update_layout(title=f"Temporal Heatmap - {species}" + (f" ({cell_type})" if cell_type else ""), xaxis_title="Time", yaxis_title="Gene", height=max(400, 50+len(genes)*20), xaxis=dict(tickangle=45), yaxis=dict(autorange='reversed'))
    return fig

def create_species_expression_comparison(ortholog_df, genes, cell_types=None):
    df = ortholog_df.copy()
    genes_upper = [g.upper() for g in genes]
    df = df[df['gene_human'].str.upper().isin(genes_upper)]
    if cell_types: df = df[df['cell_type'].isin(cell_types)]
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No data", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig
    agg = df.groupby(['gene_human', 'species']).agg({'mean_expr': 'mean'}).reset_index()
    fig = px.bar(agg, x='gene_human', y='mean_expr', color='species', barmode='group', color_discrete_map=SPECIES_COLORS, title="Cross-Species Expression")
    fig.update_layout(height=450, xaxis_tickangle=45, legend=dict(orientation='h', y=1.02, x=0.5, xanchor='center'))
    return fig

def create_ortholog_scatter(ortholog_df, gene, species_x, species_y):
    df = ortholog_df[ortholog_df['gene_human'].str.upper() == gene.upper()]
    df_x = df[df['species'] == species_x][['cell_type', 'mean_expr']].rename(columns={'mean_expr': 'expr_x'})
    df_y = df[df['species'] == species_y][['cell_type', 'mean_expr']].rename(columns={'mean_expr': 'expr_y'})
    merged = df_x.merge(df_y, on='cell_type', how='inner')
    if merged.empty:
        fig = go.Figure()
        fig.add_annotation(text=f"No common cell types", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig
    
    corr_text = f"ρ = {spearmanr(merged['expr_x'], merged['expr_y'])[0]:.3f}" if len(merged) >= 3 else ""
    fig = px.scatter(merged, x='expr_x', y='expr_y', color='cell_type', color_discrete_map=CELLTYPE_COLORS, title=f"{gene.upper()}: {species_x} vs {species_y}")
    max_val = max(merged['expr_x'].max(), merged['expr_y'].max())
    fig.add_trace(go.Scatter(x=[0, max_val], y=[0, max_val], mode='lines', line=dict(dash='dash', color='gray'), showlegend=False))
    if corr_text: fig.add_annotation(text=corr_text, xref="paper", yref="paper", x=0.95, y=0.05, showarrow=False)
    fig.update_traces(marker=dict(size=12))
    fig.update_layout(height=500, xaxis_title=f'{species_x} Expression', yaxis_title=f'{species_y} Expression')
    return fig

def create_variance_partition_barplot(vp_summary):
    colors = {'Before Correction': '#ff7f0e', 'After Correction': '#1f77b4'}
    fig = go.Figure()
    for stage in ['Before Correction', 'After Correction']:
        sd = vp_summary[vp_summary['stage'] == stage]
        fig.add_trace(go.Bar(name=stage, x=sd['variable'], y=sd['variance_explained'], marker_color=colors[stage], text=[f"{v:.1f}%" for v in sd['variance_explained']], textposition='outside'))
    fig.update_layout(title=dict(text="Variance Explained", x=0.5), xaxis_title="Variable", yaxis_title="Variance (%)", barmode='group', height=450, legend=dict(orientation='h', y=1.02, x=0.5, xanchor='center'), yaxis=dict(range=[0, max(vp_summary['variance_explained'])*1.15]))
    return fig

def create_variance_change_plot(vp_summary):
    before = vp_summary[vp_summary['stage'] == 'Before Correction'].set_index('variable')['variance_explained']
    after = vp_summary[vp_summary['stage'] == 'After Correction'].set_index('variable')['variance_explained']
    vars_df = pd.DataFrame({'variable': before.index, 'change': after.values - before.values})
    colors = ['#28a745' if (r['variable'] == 'dataset' and r['change'] < 0) or (r['variable'] != 'dataset' and r['variable'] != 'Residuals' and r['change'] >= 0) else '#dc3545' if r['variable'] != 'Residuals' else '#6c757d' for _, r in vars_df.iterrows()]
    fig = go.Figure(go.Bar(x=vars_df['variable'], y=vars_df['change'], marker_color=colors, text=[f"{c:+.1f}%" for c in vars_df['change']], textposition='outside'))
    fig.update_layout(title=dict(text="Variance Change After Correction", x=0.5), xaxis_title="Variable", yaxis_title="Change (%)", height=400, yaxis=dict(zeroline=True, zerolinewidth=2, zerolinecolor='black'))
    return fig

# Main App
def main():
    st.title("🧬 SFARI Gene Expression Explorer")
    st.markdown("*Cross-species single-cell RNA-seq browser for neurodevelopmental gene expression*")
    
    data = load_data()
    if data is None:
        st.error("Failed to load data.")
        st.stop()
    
    expr_df, risk_genes = data['expression'], data['risk_genes']
    
    # Sidebar
    with st.sidebar:
        st.header("🔍 Query Filters")
        all_species = get_unique_values(expr_df, 'species')
        selected_species = st.multiselect("Species", all_species, default=all_species[:1] if all_species else [])
        
        available_datasets = get_unique_values(expr_df[expr_df['species'].isin(selected_species)], 'tissue') if selected_species else get_unique_values(expr_df, 'tissue')
        selected_datasets = st.multiselect("Dataset", available_datasets, default=[])
        
        subset = expr_df[expr_df['species'].isin(selected_species)] if selected_species else expr_df
        if selected_datasets: subset = subset[subset['tissue'].isin(selected_datasets)]
        available_celltypes = get_unique_values(subset, 'cell_type')
        selected_celltypes = st.multiselect("Cell Types", available_celltypes, default=[])
        
        st.divider()
        st.subheader("🧬 Gene Selection")
        gene_preset = st.selectbox("Quick Gene Sets", ["Custom", "SFARI Score 1", "SFARI Score 2", "Top Variable"])
        preset_genes = ""
        if gene_preset == "SFARI Score 1": preset_genes = ", ".join(risk_genes[risk_genes['gene_score'] == 1]['gene_symbol'].dropna().tolist()[:50])
        elif gene_preset == "SFARI Score 2": preset_genes = ", ".join(risk_genes[risk_genes['gene_score'] == 2]['gene_symbol'].dropna().tolist()[:50])
        elif gene_preset == "Top Variable": preset_genes = ", ".join(expr_df.groupby('gene_human')['mean_expr'].var().sort_values(ascending=False).head(30).index.tolist())
        
        gene_input = st.text_area("Genes (comma separated)", value=preset_genes, height=100, placeholder="SHANK3, MECP2, CHD8")
        selected_genes = parse_gene_input(gene_input)
        
        st.divider()
        st.subheader("⚙️ Options")
        value_metric = st.radio("Value", ['mean_expr', 'pct_expressing'], format_func=lambda x: "Mean Expression" if x == 'mean_expr' else "% Expressing", horizontal=True)
        scale_rows = st.checkbox("Z-score scaling", value=True)
        cluster_rows = st.checkbox("Cluster rows", value=False)
        cluster_cols = st.checkbox("Cluster columns", value=False)
    
    # Filter data
    filtered_df = filter_expression_data(expr_df, selected_species or None, selected_datasets or None, selected_celltypes or None, selected_genes or None)
    
    # Metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Genes", filtered_df['gene_human'].nunique())
    c2.metric("Species", filtered_df['species'].nunique())
    c3.metric("Datasets", filtered_df['tissue'].nunique())
    c4.metric("Cell Types", filtered_df['cell_type'].nunique())
    
    # Tabs
    tabs = st.tabs(["📊 Overview", "🗺️ UMAP", "🔥 Heatmap", "🔵 Dot Plot", "📈 Temporal", "🔬 Cross-Species", "📋 Data", "📚 About"])
    
    # Overview Tab
    with tabs[0]:
        st.header("Data Overview & Batch Correction")
        if data['summary_stats'] is not None:
            totals = data['summary_stats'][data['summary_stats']['category'] == 'totals'].set_index('label')['value']
            cols = st.columns(4)
            for i, (k, lbl) in enumerate([('pseudobulk_samples', 'Samples'), ('datasets', 'Datasets'), ('cell_types', 'Cell Types'), ('organisms', 'Species')]):
                if k in totals.index: cols[i].metric(lbl, f"{int(totals[k]):,}")
        
        if data['dataset_info'] is not None:
            st.subheader("📋 Datasets")
            st.dataframe(data['dataset_info'], hide_index=True, use_container_width=True)
        
        st.divider()
        if data['batch_correction'] is not None:
            st.subheader("🔧 Batch Correction")
            st.markdown('<div class="info-box"><strong>Approach:</strong> Within-organism ComBat correction preserving cell type and developmental time.</div>', unsafe_allow_html=True)
            for _, r in data['batch_correction'].iterrows():
                if r.get('correction_applied', False): st.markdown(f"- **{r['correction_group']}**: {r['n_datasets']} datasets ({r['method']})")
        
        if data['vp_summary'] is not None:
            st.subheader("📊 Variance Partition")
            c1, c2 = st.columns(2)
            c1.plotly_chart(create_variance_partition_barplot(data['vp_summary']), use_container_width=True)
            c2.plotly_chart(create_variance_change_plot(data['vp_summary']), use_container_width=True)
    
    # UMAP Tab
    with tabs[1]:
        st.header("UMAP Visualization")
        if data['umap'] is not None:
            umap_df = data['umap']
            st.markdown(f"**{len(umap_df):,} cells** subsampled from integrated dataset.")
            color_cols = [c for c in umap_df.columns if c not in ['umap_1', 'umap_2', 'cell_id']]
            c1, c2 = st.columns([1, 3])
            with c1:
                color_by = st.selectbox("Color by", color_cols, index=color_cols.index('predicted_labels') if 'predicted_labels' in color_cols else 0)
                pt_size = st.slider("Point size", 1, 10, 3)
                opacity = st.slider("Opacity", 0.1, 1.0, 0.6)
            with c2:
                cmap = SPECIES_COLORS if color_by == 'organism' else CELLTYPE_COLORS if color_by in ['predicted_labels', 'cell_type'] else DATASET_COLORS if color_by == 'dataset' else get_color_palette(umap_df[color_by].unique().tolist())
                fig = px.scatter(umap_df, x='umap_1', y='umap_2', color=color_by, color_discrete_map=cmap, title=f"UMAP - {color_by}")
                fig.update_traces(marker=dict(size=pt_size, opacity=opacity))
                fig.update_layout(height=650, legend=dict(font=dict(size=9)))
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("UMAP data not available.")
    
    # Heatmap Tab
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
    
    # Dot Plot Tab
    with tabs[3]:
        if filtered_df.empty or not selected_genes: st.info("Select filters and genes.")
        else:
            group_by = st.selectbox("Group by", ['cell_type', 'tissue', 'species'], format_func=lambda x: x.replace('_', ' ').title())
            st.plotly_chart(create_dotplot(filtered_df, selected_genes, group_by), use_container_width=True)
    
    # Temporal Tab
    with tabs[4]:
        st.header("Temporal Expression Dynamics")
        if data['temporal'] is not None:
            temporal_df = data['temporal']
            c1, c2, c3 = st.columns(3)
            temp_species = c1.selectbox("Species", temporal_df['species'].unique().tolist(), key='ts')
            sample_types = temporal_df[temporal_df['species'] == temp_species]['sample_type'].unique().tolist() if 'sample_type' in temporal_df.columns and temp_species == 'Human' else ['in_vivo']
            temp_st = c2.selectbox("Sample Type", sample_types, key='tst') if len(sample_types) > 1 else 'in_vivo'
            viz_type = c3.selectbox("View", ["Trajectory", "Heatmap"])
            
            temp_cts = temporal_df[temporal_df['species'] == temp_species]['cell_type'].unique().tolist()
            sel_cts = st.multiselect("Cell Types", temp_cts, default=temp_cts[:3] if len(temp_cts) > 3 else temp_cts)
            
            if not selected_genes: st.info("Enter genes in sidebar.")
            elif viz_type == "Trajectory":
                st.plotly_chart(create_temporal_trajectory_plot(temporal_df, selected_genes, temp_species, temp_st, sel_cts, value_metric), use_container_width=True)
            else:
                hm_ct = st.selectbox("Cell Type (heatmap)", ['All'] + temp_cts)
                st.plotly_chart(create_temporal_heatmap(temporal_df, selected_genes, temp_species, temp_st, None if hm_ct == 'All' else hm_ct, value_metric), use_container_width=True)
        else:
            st.info("Temporal data not available. Run generate_temporal_parquets.py.")
    
    # Cross-Species Tab
    with tabs[5]:
        st.header("Cross-Species Comparison")
        if data['ortholog'] is not None:
            ortholog_df = data['ortholog']
            viz = st.selectbox("View", ["Bar Chart", "Scatter Plot"])
            if not selected_genes: st.info("Enter genes in sidebar.")
            elif viz == "Bar Chart":
                st.plotly_chart(create_species_expression_comparison(ortholog_df, selected_genes, selected_celltypes or None), use_container_width=True)
            else:
                c1, c2, c3 = st.columns(3)
                gene = c1.selectbox("Gene", selected_genes)
                sp_list = ortholog_df['species'].unique().tolist()
                sp_x = c2.selectbox("Species X", sp_list, index=0)
                sp_y = c3.selectbox("Species Y", [s for s in sp_list if s != sp_x], index=0)
                st.plotly_chart(create_ortholog_scatter(ortholog_df, gene, sp_x, sp_y), use_container_width=True)
        else:
            st.info("Ortholog data not available. Run generate_temporal_parquets.py.")
    
    # Data Tab
    with tabs[6]:
        if filtered_df.empty: st.info("No data.")
        else:
            df_disp = filtered_df.copy()
            df_disp['gene_display'] = df_disp['gene_human'].fillna(df_disp['gene_native'])
            if 'gene_symbol' in risk_genes.columns:
                df_disp['SFARI_score'] = df_disp['gene_display'].map(risk_genes.set_index('gene_symbol')['gene_score'].to_dict())
            default_cols = [c for c in ['gene_display', 'species', 'tissue', 'cell_type', 'mean_expr', 'pct_expressing', 'n_cells'] if c in df_disp.columns]
            cols = st.multiselect("Columns", df_disp.columns.tolist(), default=default_cols)
            if cols:
                st.dataframe(df_disp[cols].sort_values(['gene_display', 'species']), height=500, use_container_width=True)
                st.download_button("📥 Download CSV", df_disp[cols].to_csv(index=False), "data.csv", "text/csv")
    
    # About Tab
    with tabs[7]:
        st.markdown("""
## About SFARI Gene Expression Explorer

Explore gene expression across single-cell RNA-seq datasets from developing brain (Human, Mouse, Zebrafish, Drosophila).

### Features
- **Overview**: Batch correction methodology & variance partition
- **UMAP**: Integrated cell visualization
- **Heatmaps**: Clustered expression with faceting
- **Temporal**: Expression across developmental time
- **Cross-Species**: Ortholog-based comparisons

### Data Processing
1. Integration via scVI/scANVI
2. Within-organism limma batch correction
3. Timepoint normalization to continuous scale
4. Pseudobulk aggregation per cell type

### Datasets
Human: He (2024), Bhaduri (2021), Braun (2023), Velmeshev (2019/2023), Zhu (2023), Wang (2022/2025)
Mouse: La Manno (2021), Jin (2025), Sziraki (2023)
Zebrafish: Raj (2020) | Drosophila: Davie (2018)
        """)
    
    st.divider()
    st.markdown('<div style="text-align:center;color:#666;font-size:0.9rem;">SFARI Gene Explorer | Streamlit & Plotly</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()