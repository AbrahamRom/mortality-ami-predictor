"""Visualization utilities for EDA."""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .multivariate import PCAResults


def plot_distribution(
    df: pd.DataFrame,
    col: str,
    is_numeric: bool,
    plot_type: str = 'auto'
) -> go.Figure:
    """Generate distribution plot for a variable.
    
    Args:
        df: DataFrame
        col: Column name
        is_numeric: Whether the column is numerical
        plot_type: 'histogram', 'box', 'violin', 'bar', 'pie', 'auto'
        
    Returns:
        Plotly Figure
    """
    if plot_type == 'auto':
        plot_type = 'histogram' if is_numeric else 'bar'
    
    if is_numeric:
        if plot_type == 'histogram':
            fig = px.histogram(
                df, x=col, 
                marginal='box',
                title=f'Distribution of {col}',
                labels={col: col, 'count': 'Frequency'}
            )
            
            # Add mean and median lines
            mean_val = df[col].mean()
            median_val = df[col].median()
            
            fig.add_vline(x=mean_val, line_dash="dash", line_color="red", 
                         annotation_text=f"Mean: {mean_val:.2f}")
            fig.add_vline(x=median_val, line_dash="dot", line_color="green",
                         annotation_text=f"Median: {median_val:.2f}")
        
        elif plot_type == 'box':
            fig = px.box(df, y=col, title=f'Boxplot of {col}')
        
        elif plot_type == 'violin':
            fig = px.violin(df, y=col, box=True, title=f'Violin Plot of {col}')
        
        else:
            fig = go.Figure()
            fig.add_annotation(text="Plot type not supported for numerical variable")
    
    else:  # Categorical
        value_counts = df[col].value_counts().head(20)  # Top 20
        
        if plot_type == 'bar':
            fig = px.bar(
                x=value_counts.index, 
                y=value_counts.values,
                title=f'Frequencies of {col}',
                labels={'x': col, 'y': 'Frequency'}
            )
        
        elif plot_type == 'pie':
            fig = px.pie(
                values=value_counts.values,
                names=value_counts.index,
                title=f'Distribution of {col}'
            )
        
        else:
            fig = go.Figure()
            fig.add_annotation(text="Plot type not supported for categorical variable")
    
    fig.update_layout(template='plotly_white')
    return fig


def plot_correlation_matrix(
    df: pd.DataFrame, 
    numeric_cols: List[str],
    method: str = 'pearson'
) -> go.Figure:
    """Generate correlation matrix heatmap.
    
    Args:
        df: DataFrame
        numeric_cols: List of numerical column names
        method: 'pearson' or 'spearman'
        
    Returns:
        Plotly Figure
    """
    if len(numeric_cols) == 0:
        fig = go.Figure()
        fig.add_annotation(text="No numerical variables for correlation")
        return fig
    
    # No eliminar filas con NaN, dejar que corr() maneje los valores faltantes
    df_numeric = df[numeric_cols]
    
    if method == 'pearson':
        corr_matrix = df_numeric.corr(method='pearson')
    elif method == 'spearman':
        corr_matrix = df_numeric.corr(method='spearman')
    else:  # kendall
        corr_matrix = df_numeric.corr(method='kendall')
    
    fig = px.imshow(
        corr_matrix,
        text_auto='.2f',
        aspect='auto',
        color_continuous_scale='RdBu_r',
        zmin=-1, zmax=1,
        title=f'Correlation Matrix ({method.capitalize()})',
        labels=dict(color="Correlation")
    )
    
    fig.update_layout(
        template='plotly_white',
        width=800,
        height=800
    )
    return fig


def plot_scatter(
    df: pd.DataFrame,
    var1: str,
    var2: str,
    color_by: Optional[str] = None,
    add_trendline: bool = True
) -> go.Figure:
    """Generate scatter plot between two variables.
    
    Args:
        df: DataFrame
        var1: X variable
        var2: Y variable
        color_by: Variable for coloring points
        add_trendline: Whether to add trendline
        
    Returns:
        Plotly Figure
    """
    trendline = 'ols' if add_trendline else None
    
    fig = px.scatter(
        df, x=var1, y=var2, 
        color=color_by,
        trendline=trendline,
        title=f'{var1} vs {var2}',
        opacity=0.6
    )
    
    fig.update_layout(template='plotly_white')
    return fig


def plot_pairwise_scatter(
    df: pd.DataFrame,
    variables: List[str],
    max_vars: int = 10
) -> go.Figure:
    """Generate pairwise scatter plot matrix.
    
    Args:
        df: DataFrame
        variables: List of variables to include
        max_vars: Maximum number of variables
        
    Returns:
        Plotly Figure
    """
    variables = variables[:max_vars]
    
    if len(variables) < 2:
        fig = go.Figure()
        fig.add_annotation(text="At least 2 numerical variables required")
        return fig
    
    fig = px.scatter_matrix(
        df[variables],
        dimensions=variables,
        title='Pairwise Scatter Plot Matrix'
    )
    
    fig.update_traces(diagonal_visible=False, showupperhalf=False)
    fig.update_layout(template='plotly_white')
    return fig


def plot_pca_scree(pca_results: PCAResults) -> go.Figure:
    """Generate scree plot (explained variance by component).
    
    Args:
        pca_results: PCAResults object
        
    Returns:
        Plotly Figure
    """
    pca = pca_results
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Explained Variance by Component', 
                       'Cumulative Variance')
    )
    
    # Bar chart - individual variance
    fig.add_trace(
        go.Bar(
            x=[f'PC{i+1}' for i in range(pca.n_components)],
            y=pca.explained_variance_ratio,
            name='Individual',
            marker_color='steelblue'
        ),
        row=1, col=1
    )
    
    # Line - cumulative variance
    fig.add_trace(
        go.Scatter(
            x=[f'PC{i+1}' for i in range(pca.n_components)],
            y=pca.cumulative_variance,
            mode='lines+markers',
            name='Cumulative',
            marker_color='orange'
        ),
        row=1, col=2
    )
    
    fig.update_yaxes(title_text="Explained Variance", row=1, col=1)
    fig.update_yaxes(title_text="Cumulative Variance", row=1, col=2)
    fig.update_xaxes(title_text="Principal Component", row=1, col=1)
    fig.update_xaxes(title_text="Principal Component", row=1, col=2)
    
    fig.update_layout(
        title='Variance Analysis - PCA',
        template='plotly_white',
        showlegend=False
    )
    
    return fig


def plot_pca_biplot(
    pca_results: PCAResults,
    pc_x: int = 1,
    pc_y: int = 2,
    n_features: int = 10
) -> go.Figure:
    """Generate PCA biplot.
    
    Args:
        pca_results: PCAResults object
        pc_x: Principal component for X axis (1-indexed)
        pc_y: Principal component for Y axis (1-indexed)
        n_features: Number of most important features to show
        
    Returns:
        Plotly Figure
    """
    pca = pca_results
    pc_x_idx = pc_x - 1
    pc_y_idx = pc_y - 1
    
    if pc_x_idx >= pca.n_components or pc_y_idx >= pca.n_components:
        raise ValueError(f"PC indices must be between 1 and {pca.n_components}")
    
    # Scatter of observations
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=pca.transformed_data.iloc[:, pc_x_idx],
        y=pca.transformed_data.iloc[:, pc_y_idx],
        mode='markers',
        marker=dict(size=5, opacity=0.5, color='steelblue'),
        name='Observations',
        text=[f'Obs {i}' for i in range(len(pca.transformed_data))]
    ))
    
    # Feature vectors
    loadings = pca.components[[pc_x_idx, pc_y_idx], :].T
    
    # Select top N features by importance
    importance = np.sqrt(loadings[:, 0]**2 + loadings[:, 1]**2)
    top_indices = np.argsort(importance)[-n_features:]
    
    for i in top_indices:
        fig.add_trace(go.Scatter(
            x=[0, loadings[i, 0]],
            y=[0, loadings[i, 1]],
            mode='lines+text',
            line=dict(color='red', width=2),
            text=['', pca.feature_names[i]],
            textposition='top center',
            showlegend=False,
            hoverinfo='text',
            hovertext=f'{pca.feature_names[i]}'
        ))
    
    fig.update_layout(
        title=f'PCA Biplot (PC{pc_x} vs PC{pc_y})',
        xaxis_title=f'PC{pc_x} ({pca.explained_variance_ratio[pc_x_idx]:.1%})',
        yaxis_title=f'PC{pc_y} ({pca.explained_variance_ratio[pc_y_idx]:.1%})',
        template='plotly_white'
    )
    
    return fig
