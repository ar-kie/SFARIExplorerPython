"""
Temporal Dynamics Module for SFARI Gene Expression Explorer

This module provides sophisticated visualization of gene expression
dynamics across developmental timepoints, with cross-species alignment.

Features:
- Pseudotime trajectory visualization
- Developmental stage alignment across species
- Spline-smoothed expression curves with confidence intervals
- Scaled/normalized time comparisons
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.interpolate import UnivariateSpline
from scipy.ndimage import gaussian_filter1d
from typing import Optional, List, Dict, Tuple, Union


# =============================================================================
# Developmental Time Alignment
# =============================================================================

# Cross-species developmental equivalence mapping
# Based on Workman et al. (2013) and other comparative developmental studies
SPECIES_DEV_LANDMARKS = {
    # Key developmental events mapped to normalized time (0-1)
    "neural_tube_closure": {"Human": 0.08, "Mouse": 0.12, "Zebrafish": 0.20, "Drosophila": 0.15},
    "neurogenesis_onset": {"Human": 0.10, "Mouse": 0.14, "Zebrafish": 0.25, "Drosophila": 0.20},
    "peak_neurogenesis": {"Human": 0.20, "Mouse": 0.22, "Zebrafish": 0.35, "Drosophila": 0.30},
    "gliogenesis_onset": {"Human": 0.25, "Mouse": 0.28, "Zebrafish": 0.45, "Drosophila": 0.40},
    "birth_hatching": {"Human": 0.45, "Mouse": 0.45, "Zebrafish": 0.50, "Drosophila": 0.70},
    "maturation": {"Human": 0.80, "Mouse": 0.75, "Zebrafish": 0.85, "Drosophila": 0.90},
}


def align_developmental_time(
    time_values: np.ndarray,
    source_species: str,
    target_species: str = "Human"
) -> np.ndarray:
    """
    Align developmental timepoints from one species to another using
    piecewise linear interpolation between conserved developmental landmarks.
    
    Args:
        time_values: Normalized developmental times (0-1) from source species
        source_species: The source species
        target_species: The target species for alignment (default: Human)
    
    Returns:
        Aligned time values in the target species' developmental timeline
    """
    if source_species == target_species:
        return time_values
    
    # Get landmark times for both species
    source_landmarks = []
    target_landmarks = []
    
    for event, species_times in SPECIES_DEV_LANDMARKS.items():
        if source_species in species_times and target_species in species_times:
            source_landmarks.append(species_times[source_species])
            target_landmarks.append(species_times[target_species])
    
    if len(source_landmarks) < 2:
        # Not enough landmarks - return original
        return time_values
    
    # Sort by source time
    sorted_idx = np.argsort(source_landmarks)
    source_landmarks = np.array(source_landmarks)[sorted_idx]
    target_landmarks = np.array(target_landmarks)[sorted_idx]
    
    # Add boundaries
    source_landmarks = np.concatenate([[0], source_landmarks, [1]])
    target_landmarks = np.concatenate([[0], target_landmarks, [1]])
    
    # Piecewise linear interpolation
    aligned = np.interp(time_values, source_landmarks, target_landmarks)
    
    return aligned


# =============================================================================
# Expression Smoothing
# =============================================================================

def smooth_expression_trajectory(
    times: np.ndarray,
    values: np.ndarray,
    n_cells: Optional[np.ndarray] = None,
    method: str = "spline",
    smoothing_factor: float = 0.5,
    n_points: int = 100
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Smooth expression values along a time trajectory.
    
    Args:
        times: Time points (pseudotime or developmental time)
        values: Expression values at each time point
        n_cells: Number of cells at each time point (for weighting)
        method: Smoothing method ("spline", "gaussian", "lowess")
        smoothing_factor: Degree of smoothing (0-1)
        n_points: Number of output points
    
    Returns:
        times_smooth: Smoothed time points
        values_smooth: Smoothed expression values
        ci_lower: Lower confidence interval
        ci_upper: Upper confidence interval
    """
    # Remove NaN
    valid = ~(np.isnan(times) | np.isnan(values))
    times = times[valid]
    values = values[valid]
    
    if len(times) < 3:
        return np.array([]), np.array([]), np.array([]), np.array([])
    
    # Sort by time
    sort_idx = np.argsort(times)
    times = times[sort_idx]
    values = values[sort_idx]
    if n_cells is not None:
        n_cells = n_cells[valid][sort_idx]
    
    # Create output time grid
    times_smooth = np.linspace(times.min(), times.max(), n_points)
    
    if method == "spline":
        # Weighted spline smoothing
        weights = np.sqrt(n_cells) if n_cells is not None else None
        s = smoothing_factor * len(times)  # Smoothing parameter
        
        try:
            spline = UnivariateSpline(times, values, w=weights, s=s)
            values_smooth = spline(times_smooth)
            
            # Estimate confidence interval using residuals
            residuals = values - spline(times)
            std_resid = np.std(residuals)
            ci_lower = values_smooth - 1.96 * std_resid
            ci_upper = values_smooth + 1.96 * std_resid
        except:
            # Fallback to linear interpolation
            values_smooth = np.interp(times_smooth, times, values)
            ci_lower = values_smooth
            ci_upper = values_smooth
    
    elif method == "gaussian":
        # Gaussian smoothing
        sigma = smoothing_factor * len(times) / 10
        
        # Interpolate to regular grid first
        values_interp = np.interp(times_smooth, times, values)
        values_smooth = gaussian_filter1d(values_interp, sigma=sigma)
        
        # Simple CI based on local variance
        window = max(3, int(len(times) * 0.1))
        local_std = pd.Series(values).rolling(window, center=True).std().fillna(method='bfill').fillna(method='ffill')
        std_interp = np.interp(times_smooth, times, local_std.values)
        ci_lower = values_smooth - 1.96 * std_interp
        ci_upper = values_smooth + 1.96 * std_interp
    
    else:
        # No smoothing
        values_smooth = np.interp(times_smooth, times, values)
        ci_lower = values_smooth
        ci_upper = values_smooth
    
    return times_smooth, values_smooth, ci_lower, ci_upper


# =============================================================================
# Visualization Functions
# =============================================================================

def create_temporal_trajectory_plot(
    df: pd.DataFrame,
    genes: List[str],
    time_col: str = "dev_time",
    value_col: str = "mean_expr",
    color_by: str = "species",
    facet_by: Optional[str] = None,
    smooth: bool = True,
    show_ci: bool = True,
    align_species: bool = False,
    reference_species: str = "Human",
    title: str = "Expression Dynamics"
) -> go.Figure:
    """
    Create publication-quality temporal trajectory plot.
    
    Args:
        df: Expression data with time information
        genes: List of gene symbols to plot
        time_col: Column containing time values
        value_col: Column containing expression values
        color_by: Variable for color encoding
        facet_by: Variable for faceting (creates subplots)
        smooth: Whether to smooth trajectories
        show_ci: Whether to show confidence intervals
        align_species: Whether to align developmental times across species
        reference_species: Reference species for time alignment
        title: Plot title
    
    Returns:
        Plotly Figure object
    """
    df = df.copy()
    df['gene_display'] = df['gene_human'].fillna(df['gene_native'])
    
    # Filter to selected genes
    df = df[df['gene_display'].isin(genes)]
    
    if df.empty or time_col not in df.columns:
        fig = go.Figure()
        fig.add_annotation(
            text="No temporal data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig
    
    # Align developmental times if requested
    if align_species and 'species' in df.columns:
        df['aligned_time'] = df.apply(
            lambda row: align_developmental_time(
                np.array([row[time_col]]),
                row['species'],
                reference_species
            )[0] if pd.notna(row[time_col]) else np.nan,
            axis=1
        )
        time_col = 'aligned_time'
    
    # Determine subplot structure
    if facet_by and facet_by in df.columns:
        facet_values = sorted(df[facet_by].unique())
        n_facets = len(facet_values)
    else:
        facet_values = [None]
        n_facets = 1
    
    n_genes = len(genes)
    
    # Create subplots
    fig = make_subplots(
        rows=n_genes,
        cols=n_facets,
        subplot_titles=[f"{g}" for g in genes] if n_facets == 1 
                       else [f"{g} - {f}" for g in genes for f in facet_values],
        vertical_spacing=0.08,
        horizontal_spacing=0.05
    )
    
    # Color palette
    color_values = df[color_by].unique() if color_by in df.columns else ['default']
    colors = {
        v: c for v, c in zip(
            color_values,
            ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
             '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        )
    }
    
    # Plot each gene × facet combination
    for gene_idx, gene in enumerate(genes):
        gene_df = df[df['gene_display'] == gene]
        
        for facet_idx, facet_val in enumerate(facet_values):
            if facet_val is not None:
                plot_df = gene_df[gene_df[facet_by] == facet_val]
            else:
                plot_df = gene_df
            
            row = gene_idx + 1
            col = facet_idx + 1
            
            for color_val in color_values:
                if color_by in plot_df.columns:
                    subset = plot_df[plot_df[color_by] == color_val]
                else:
                    subset = plot_df
                
                if subset.empty:
                    continue
                
                times = subset[time_col].values
                values = subset[value_col].values
                n_cells = subset['n_cells'].values if 'n_cells' in subset.columns else None
                
                color = colors.get(color_val, '#1f77b4')
                
                if smooth and len(times) >= 3:
                    t_smooth, v_smooth, ci_lo, ci_hi = smooth_expression_trajectory(
                        times, values, n_cells, method="spline"
                    )
                    
                    # Add confidence interval
                    if show_ci and len(t_smooth) > 0:
                        fig.add_trace(
                            go.Scatter(
                                x=np.concatenate([t_smooth, t_smooth[::-1]]),
                                y=np.concatenate([ci_hi, ci_lo[::-1]]),
                                fill='toself',
                                fillcolor=f'rgba{tuple(list(int(color[i:i+2], 16) for i in (1, 3, 5)) + [0.2])}',
                                line=dict(width=0),
                                showlegend=False,
                                hoverinfo='skip'
                            ),
                            row=row, col=col
                        )
                    
                    # Add smoothed line
                    if len(t_smooth) > 0:
                        fig.add_trace(
                            go.Scatter(
                                x=t_smooth,
                                y=v_smooth,
                                mode='lines',
                                name=str(color_val),
                                line=dict(color=color, width=2),
                                showlegend=(gene_idx == 0 and facet_idx == 0)
                            ),
                            row=row, col=col
                        )
                
                # Add data points
                fig.add_trace(
                    go.Scatter(
                        x=times,
                        y=values,
                        mode='markers',
                        name=str(color_val) if not smooth else None,
                        marker=dict(color=color, size=6, opacity=0.6),
                        showlegend=(gene_idx == 0 and facet_idx == 0 and not smooth),
                        hovertemplate=(
                            f"Gene: {gene}<br>"
                            f"Time: %{{x:.3f}}<br>"
                            f"Expression: %{{y:.3f}}<extra></extra>"
                        )
                    ),
                    row=row, col=col
                )
    
    # Update layout
    fig.update_layout(
        title=dict(text=title, x=0.5),
        height=250 * n_genes,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=-0.15 / n_genes,
            xanchor='center',
            x=0.5
        ),
        hovermode='closest'
    )
    
    # Update axes labels
    fig.update_xaxes(title_text="Developmental Time" if align_species else time_col)
    fig.update_yaxes(title_text=value_col.replace('_', ' ').title())
    
    return fig


def create_pseudotime_heatmap(
    df: pd.DataFrame,
    genes: List[str],
    time_col: str = "pseudotime",
    value_col: str = "mean_expr",
    n_bins: int = 50,
    cluster_genes: bool = True,
    title: str = "Pseudotime Expression Heatmap"
) -> go.Figure:
    """
    Create a heatmap showing gene expression along pseudotime.
    
    Args:
        df: Expression data with pseudotime
        genes: List of genes to include
        time_col: Column containing pseudotime values
        value_col: Expression value column
        n_bins: Number of time bins
        cluster_genes: Whether to cluster genes by expression pattern
        title: Plot title
    
    Returns:
        Plotly Figure object
    """
    df = df.copy()
    df['gene_display'] = df['gene_human'].fillna(df['gene_native'])
    df = df[df['gene_display'].isin(genes)]
    
    if df.empty or time_col not in df.columns:
        fig = go.Figure()
        fig.add_annotation(text="No data available", x=0.5, y=0.5, showarrow=False)
        return fig
    
    # Bin pseudotime
    df['time_bin'] = pd.cut(df[time_col], bins=n_bins, labels=False)
    
    # Aggregate by gene and time bin
    agg = df.groupby(['gene_display', 'time_bin'])[value_col].mean().unstack(fill_value=0)
    
    # Z-score normalize rows
    row_means = agg.mean(axis=1)
    row_stds = agg.std(axis=1).replace(0, 1)
    agg_scaled = agg.sub(row_means, axis=0).div(row_stds, axis=0).clip(-3, 3)
    
    # Cluster genes if requested
    if cluster_genes and len(genes) > 1:
        from scipy.cluster.hierarchy import linkage, leaves_list
        from scipy.spatial.distance import pdist
        
        try:
            dist = pdist(agg_scaled.fillna(0).values)
            link = linkage(dist, method='average')
            order = leaves_list(link)
            agg_scaled = agg_scaled.iloc[order]
        except:
            pass
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=agg_scaled.values,
        x=[f"{i/n_bins:.2f}" for i in range(n_bins)],
        y=agg_scaled.index.tolist(),
        colorscale='RdBu_r',
        zmid=0,
        zmin=-3,
        zmax=3,
        colorbar=dict(title="Z-score")
    ))
    
    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis_title="Pseudotime",
        yaxis_title="Gene",
        height=max(400, 20 * len(genes)),
        yaxis=dict(autorange='reversed')
    )
    
    return fig


def create_velocity_stream_plot(
    df: pd.DataFrame,
    genes: List[str],
    velocity_col: str = "velocity",
    time_col: str = "pseudotime",
    title: str = "RNA Velocity Stream"
) -> go.Figure:
    """
    Create RNA velocity stream visualization.
    
    Note: This requires velocity data computed externally (e.g., scVelo).
    
    Args:
        df: Data with velocity information
        genes: Genes to visualize
        velocity_col: Column containing velocity values
        time_col: Pseudotime column
        title: Plot title
    
    Returns:
        Plotly Figure object
    """
    # Placeholder - requires integration with scVelo output
    fig = go.Figure()
    fig.add_annotation(
        text="RNA velocity visualization requires scVelo output data.<br>"
             "Run scVelo on your h5ad file and export velocity results.",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=14)
    )
    fig.update_layout(
        title=title,
        height=400
    )
    return fig


# =============================================================================
# Streamlit Integration
# =============================================================================

def render_temporal_tab(
    df: pd.DataFrame,
    genes: List[str],
    st_module  # Streamlit module passed in
):
    """
    Render the temporal dynamics tab in Streamlit.
    
    Args:
        df: Expression DataFrame
        genes: Selected genes
        st_module: Streamlit module (import streamlit as st)
    """
    st = st_module
    
    st.markdown("""
    ### 📈 Temporal Expression Dynamics
    
    Visualize gene expression changes across developmental time or pseudotime.
    """)
    
    # Check for time columns
    time_columns = []
    for col in ['dev_time', 'pseudotime', 'timepoint', 'age', 'stage']:
        if col in df.columns:
            time_columns.append(col)
    
    if not time_columns:
        st.warning("""
        **No temporal data found.**
        
        To enable temporal visualization, your data needs one of these columns:
        - `dev_time`: Normalized developmental time (0-1)
        - `pseudotime`: Pseudotime from trajectory inference
        - `timepoint`: Discrete developmental stages
        - `age`: Sample age
        
        You can add these columns during data preparation.
        """)
        return
    
    # Settings
    col1, col2, col3 = st.columns(3)
    
    with col1:
        time_col = st.selectbox("Time variable", time_columns)
    
    with col2:
        color_by = st.selectbox(
            "Color by",
            options=['species', 'cell_type', 'dataset'],
            format_func=lambda x: x.replace('_', ' ').title()
        )
    
    with col3:
        smooth = st.checkbox("Smooth trajectories", value=True)
    
    col4, col5 = st.columns(2)
    
    with col4:
        show_ci = st.checkbox("Show confidence intervals", value=True)
    
    with col5:
        align = st.checkbox("Align developmental time across species", value=False)
    
    if not genes:
        st.info("Select genes to visualize their temporal dynamics.")
        return
    
    # Create plot
    fig = create_temporal_trajectory_plot(
        df=df,
        genes=genes[:10],  # Limit for performance
        time_col=time_col,
        color_by=color_by,
        smooth=smooth,
        show_ci=show_ci,
        align_species=align,
        title="Gene Expression Trajectories"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Additional view: Pseudotime heatmap
    if st.checkbox("Show pseudotime heatmap view"):
        fig_heatmap = create_pseudotime_heatmap(
            df=df,
            genes=genes[:30],
            time_col=time_col,
            title="Expression Pattern Across Time"
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)
