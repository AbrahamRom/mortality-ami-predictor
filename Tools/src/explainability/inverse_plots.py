"""Visualization functions for Inverse Optimization results.

This module provides interactive Plotly visualizations for inverse optimization,
including sensitivity plots, confidence intervals, feature importance for optimization,
and comparative analysis between current and optimal values.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


def plot_optimal_values_comparison(
    original_values: Dict[str, float],
    optimal_values: Dict[str, float],
    feature_names: Optional[List[str]] = None,
    title: str = "Original vs Optimal Feature Values",
) -> go.Figure:
    """Create comparison plot between original and optimal values.
    
    Args:
        original_values: Dictionary of original feature values
        optimal_values: Dictionary of optimal feature values
        feature_names: Optional list to order features
        title: Plot title
    
    Returns:
        Plotly figure
    """
    if feature_names is None:
        feature_names = sorted(optimal_values.keys())
    else:
        feature_names = [f for f in feature_names if f in optimal_values]
    
    original = [original_values.get(f, np.nan) for f in feature_names]
    optimal = [optimal_values.get(f, np.nan) for f in feature_names]
    changes = [opt - orig for orig, opt in zip(original, optimal)]
    
    # Create subplot with shared x-axis
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Feature Values Comparison", "Change Required"),
        vertical_spacing=0.15,
        row_heights=[0.6, 0.4],
    )
    
    # Top plot: Comparison
    fig.add_trace(
        go.Bar(
            x=feature_names,
            y=original,
            name="Original",
            marker_color='lightcoral',
            opacity=0.7,
            hovertemplate='<b>%{x}</b><br>Original: %{y:.3f}<extra></extra>',
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Bar(
            x=feature_names,
            y=optimal,
            name="Optimal",
            marker_color='lightseagreen',
            opacity=0.7,
            hovertemplate='<b>%{x}</b><br>Optimal: %{y:.3f}<extra></extra>',
        ),
        row=1, col=1
    )
    
    # Bottom plot: Changes
    colors = ['red' if c < 0 else 'green' for c in changes]
    
    fig.add_trace(
        go.Bar(
            x=feature_names,
            y=changes,
            name="Change",
            marker_color=colors,
            opacity=0.7,
            showlegend=False,
            hovertemplate='<b>%{x}</b><br>Change: %{y:.3f}<extra></extra>',
        ),
        row=2, col=1
    )
    
    # Add zero line to changes plot
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)
    
    # Update layout
    fig.update_xaxes(title_text="Features", row=2, col=1, tickangle=-45)
    fig.update_yaxes(title_text="Value", row=1, col=1)
    fig.update_yaxes(title_text="Δ Value", row=2, col=1)
    
    fig.update_layout(
        title=title,
        height=700,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode='x unified',
        template='plotly_white',
    )
    
    return fig


def plot_confidence_intervals(
    ci_results: Dict[str, Dict[str, Any]],
    optimal_values: Optional[Dict[str, float]] = None,
    title: str = "Optimal Values with Confidence Intervals",
) -> go.Figure:
    """Plot confidence intervals for optimal values.
    
    Args:
        ci_results: Dictionary from compute_confidence_intervals
        optimal_values: Optional point estimates
        title: Plot title
    
    Returns:
        Plotly figure
    """
    features = sorted(ci_results.keys())
    
    means = [ci_results[f]['mean'] for f in features]
    medians = [ci_results[f]['median'] for f in features]
    lower = [ci_results[f]['lower_ci'] for f in features]
    upper = [ci_results[f]['upper_ci'] for f in features]
    
    fig = go.Figure()
    
    # Add confidence intervals as error bars
    fig.add_trace(
        go.Scatter(
            x=features,
            y=medians,
            mode='markers',
            name='Median',
            marker=dict(size=10, color='darkblue'),
            error_y=dict(
                type='data',
                symmetric=False,
                array=[u - m for u, m in zip(upper, medians)],
                arrayminus=[m - l for m, l in zip(medians, lower)],
                color='lightblue',
                thickness=2,
                width=5,
            ),
            hovertemplate='<b>%{x}</b><br>Median: %{y:.3f}<br>CI: [%{customdata[0]:.3f}, %{customdata[1]:.3f}]<extra></extra>',
            customdata=list(zip(lower, upper)),
        )
    )
    
    # Add mean as separate trace
    fig.add_trace(
        go.Scatter(
            x=features,
            y=means,
            mode='markers',
            name='Mean',
            marker=dict(size=8, color='orange', symbol='diamond'),
            hovertemplate='<b>%{x}</b><br>Mean: %{y:.3f}<extra></extra>',
        )
    )
    
    # Add optimal values if provided
    if optimal_values:
        optimal = [optimal_values.get(f, np.nan) for f in features]
        fig.add_trace(
            go.Scatter(
                x=features,
                y=optimal,
                mode='markers',
                name='Point Estimate',
                marker=dict(size=8, color='red', symbol='x'),
                hovertemplate='<b>%{x}</b><br>Optimal: %{y:.3f}<extra></extra>',
            )
        )
    
    fig.update_layout(
        title=title,
        xaxis_title="Features",
        yaxis_title="Value",
        height=500,
        hovermode='x unified',
        template='plotly_white',
        showlegend=True,
        xaxis=dict(tickangle=-45),
    )
    
    return fig


def plot_sensitivity_analysis(
    sensitivity_df: pd.DataFrame,
    target_value: Optional[float] = None,
    title: str = "Sensitivity Analysis: Prediction vs Feature Perturbations",
) -> go.Figure:
    """Plot sensitivity analysis results.
    
    Args:
        sensitivity_df: DataFrame from sensitivity_analysis method
        target_value: Optional target value to show as reference
        title: Plot title
    
    Returns:
        Plotly figure
    """
    features = sensitivity_df['feature'].unique()
    
    fig = go.Figure()
    
    # Color palette
    colors = px.colors.qualitative.Plotly
    
    for i, feat in enumerate(features):
        feat_data = sensitivity_df[sensitivity_df['feature'] == feat]
        
        fig.add_trace(
            go.Scatter(
                x=feat_data['value'],
                y=feat_data['prediction'],
                mode='lines+markers',
                name=feat,
                line=dict(width=2, color=colors[i % len(colors)]),
                marker=dict(size=6),
                hovertemplate=f'<b>{feat}</b><br>Value: %{{x:.3f}}<br>Prediction: %{{y:.3f}}<extra></extra>',
            )
        )
    
    # Add target line if provided
    if target_value is not None:
        fig.add_hline(
            y=target_value,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Target: {target_value:.3f}",
            annotation_position="right",
        )
    
    fig.update_layout(
        title=title,
        xaxis_title="Feature Value",
        yaxis_title="Model Prediction",
        height=600,
        hovermode='closest',
        template='plotly_white',
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor='rgba(255, 255, 255, 0.8)',
        ),
    )
    
    return fig


def plot_sensitivity_heatmap(
    sensitivity_df: pd.DataFrame,
    title: str = "Feature Sensitivity Heatmap",
) -> go.Figure:
    """Create heatmap showing prediction sensitivity to each feature.
    
    Args:
        sensitivity_df: DataFrame from sensitivity_analysis
        title: Plot title
    
    Returns:
        Plotly figure
    """
    # Pivot data for heatmap
    features = sensitivity_df['feature'].unique()
    
    # Create matrix: rows = features, columns = perturbation levels
    pivot_data = []
    feature_labels = []
    
    for feat in features:
        feat_data = sensitivity_df[sensitivity_df['feature'] == feat].sort_values('delta_from_optimal')
        predictions = feat_data['prediction'].values
        pivot_data.append(predictions)
        feature_labels.append(feat)
    
    # Get perturbation levels (should be same for all features)
    first_feat = features[0]
    feat_data = sensitivity_df[sensitivity_df['feature'] == first_feat].sort_values('delta_from_optimal')
    perturbation_labels = [f"{d:+.2f}" for d in feat_data['delta_from_optimal']]
    
    fig = go.Figure(
        data=go.Heatmap(
            z=pivot_data,
            x=perturbation_labels,
            y=feature_labels,
            colorscale='RdYlGn_r',
            colorbar=dict(title="Prediction"),
            hovertemplate='Feature: %{y}<br>Δ: %{x}<br>Prediction: %{z:.3f}<extra></extra>',
        )
    )
    
    fig.update_layout(
        title=title,
        xaxis_title="Perturbation from Optimal",
        yaxis_title="Feature",
        height=max(400, len(features) * 30),
        template='plotly_white',
    )
    
    return fig


def plot_optimization_convergence(
    result_history: List[Dict[str, Any]],
    title: str = "Optimization Convergence",
) -> go.Figure:
    """Plot optimization convergence over iterations.
    
    Args:
        result_history: List of optimization results from multiple restarts
        title: Plot title
    
    Returns:
        Plotly figure
    """
    fig = go.Figure()
    
    for i, result in enumerate(result_history):
        if 'objective_value' in result:
            fig.add_trace(
                go.Scatter(
                    x=[i],
                    y=[result['objective_value']],
                    mode='markers',
                    name=f"Restart {i}",
                    marker=dict(
                        size=10,
                        color='green' if result.get('success', False) else 'red',
                        symbol='circle' if result.get('success', False) else 'x',
                    ),
                    hovertemplate=f'Restart {i}<br>Objective: %{{y:.6f}}<br>Success: {result.get("success", False)}<extra></extra>',
                )
            )
    
    fig.update_layout(
        title=title,
        xaxis_title="Restart Number",
        yaxis_title="Objective Value",
        height=400,
        showlegend=False,
        template='plotly_white',
    )
    
    return fig


def plot_feature_importance_for_optimization(
    sensitivity_df: pd.DataFrame,
    title: str = "Feature Importance for Target Achievement",
) -> go.Figure:
    """Plot how much each feature affects reaching the target.
    
    Computed as the range of predictions when perturbing each feature.
    
    Args:
        sensitivity_df: DataFrame from sensitivity_analysis
        title: Plot title
    
    Returns:
        Plotly figure
    """
    # Compute range of predictions for each feature
    importance_data = []
    
    for feat in sensitivity_df['feature'].unique():
        feat_data = sensitivity_df[sensitivity_df['feature'] == feat]
        pred_range = feat_data['prediction'].max() - feat_data['prediction'].min()
        pred_std = feat_data['prediction'].std()
        
        importance_data.append({
            'feature': feat,
            'prediction_range': pred_range,
            'prediction_std': pred_std,
        })
    
    importance_df = pd.DataFrame(importance_data).sort_values('prediction_range', ascending=True)
    
    fig = go.Figure()
    
    fig.add_trace(
        go.Bar(
            x=importance_df['prediction_range'],
            y=importance_df['feature'],
            orientation='h',
            marker=dict(
                color=importance_df['prediction_range'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Range"),
            ),
            hovertemplate='<b>%{y}</b><br>Prediction Range: %{x:.4f}<extra></extra>',
        )
    )
    
    fig.update_layout(
        title=title,
        xaxis_title="Prediction Range (Max - Min)",
        yaxis_title="Feature",
        height=max(400, len(importance_df) * 30),
        template='plotly_white',
    )
    
    return fig


def plot_bootstrap_distributions(
    ci_results: Dict[str, Dict[str, Any]],
    feature: str,
    bins: int = 30,
    title: Optional[str] = None,
) -> go.Figure:
    """Plot bootstrap distribution for a single feature.
    
    Args:
        ci_results: Results from compute_confidence_intervals
        feature: Feature name to plot
        bins: Number of histogram bins
        title: Plot title
    
    Returns:
        Plotly figure
    """
    if feature not in ci_results:
        raise ValueError(f"Feature '{feature}' not in results")
    
    feat_data = ci_results[feature]
    values = feat_data['all_values']
    
    if title is None:
        title = f"Bootstrap Distribution: {feature}"
    
    fig = go.Figure()
    
    # Histogram
    fig.add_trace(
        go.Histogram(
            x=values,
            nbinsx=bins,
            name='Distribution',
            marker_color='lightblue',
            opacity=0.7,
            hovertemplate='Value: %{x:.3f}<br>Count: %{y}<extra></extra>',
        )
    )
    
    # Add vertical lines for statistics
    fig.add_vline(
        x=feat_data['mean'],
        line_dash="dash",
        line_color="red",
        annotation_text=f"Mean: {feat_data['mean']:.3f}",
        annotation_position="top",
    )
    
    fig.add_vline(
        x=feat_data['median'],
        line_dash="dash",
        line_color="green",
        annotation_text=f"Median: {feat_data['median']:.3f}",
        annotation_position="top",
    )
    
    fig.add_vline(
        x=feat_data['lower_ci'],
        line_dash="dot",
        line_color="gray",
        annotation_text=f"Lower CI",
        annotation_position="bottom left",
    )
    
    fig.add_vline(
        x=feat_data['upper_ci'],
        line_dash="dot",
        line_color="gray",
        annotation_text=f"Upper CI",
        annotation_position="bottom right",
    )
    
    fig.update_layout(
        title=title,
        xaxis_title="Optimal Value",
        yaxis_title="Frequency",
        height=400,
        showlegend=False,
        template='plotly_white',
    )
    
    return fig


def plot_parallel_coordinates(
    scenarios: List[Dict[str, Any]],
    feature_names: List[str],
    color_by: str = 'achieved_prediction',
    title: str = "Optimization Scenarios Comparison",
) -> go.Figure:
    """Create parallel coordinates plot comparing multiple optimization scenarios.
    
    Args:
        scenarios: List of optimization results
        feature_names: Features to display
        color_by: Column to use for coloring
        title: Plot title
    
    Returns:
        Plotly figure
    """
    # Extract data
    data_rows = []
    for scenario in scenarios:
        row = scenario['optimal_values'].copy()
        row['achieved_prediction'] = scenario['achieved_prediction']
        row['success'] = scenario['success']
        data_rows.append(row)
    
    df = pd.DataFrame(data_rows)
    
    # Ensure all features are present
    for feat in feature_names:
        if feat not in df.columns:
            df[feat] = np.nan
    
    # Create dimensions for parallel coordinates
    dimensions = []
    
    for feat in feature_names:
        if feat in df.columns:
            dimensions.append(
                dict(
                    label=feat,
                    values=df[feat],
                )
            )
    
    # Add prediction dimension
    dimensions.append(
        dict(
            label='Prediction',
            values=df['achieved_prediction'],
        )
    )
    
    fig = go.Figure(
        data=go.Parcoords(
            line=dict(
                color=df[color_by] if color_by in df.columns else df['achieved_prediction'],
                colorscale='Viridis',
                showscale=True,
                cmin=df[color_by].min() if color_by in df.columns else df['achieved_prediction'].min(),
                cmax=df[color_by].max() if color_by in df.columns else df['achieved_prediction'].max(),
            ),
            dimensions=dimensions,
        )
    )
    
    fig.update_layout(
        title=title,
        height=600,
        template='plotly_white',
    )
    
    return fig


def create_optimization_summary_figure(
    result: Dict[str, Any],
    original_values: Optional[Dict[str, float]] = None,
) -> go.Figure:
    """Create comprehensive summary figure for optimization result.
    
    Args:
        result: Optimization result dictionary
        original_values: Optional original values for comparison
    
    Returns:
        Plotly figure with multiple subplots
    """
    optimal_vals = result['optimal_values']
    features = sorted(optimal_vals.keys())
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Optimal Values",
            "Optimization Success",
            "Feature Changes",
            "Summary Metrics"
        ),
        specs=[
            [{"type": "bar"}, {"type": "indicator"}],
            [{"type": "bar"}, {"type": "table"}],
        ],
    )
    
    # 1. Optimal values bar chart
    fig.add_trace(
        go.Bar(
            x=features,
            y=[optimal_vals[f] for f in features],
            marker_color='lightseagreen',
            name='Optimal Values',
            hovertemplate='<b>%{x}</b><br>Value: %{y:.3f}<extra></extra>',
        ),
        row=1, col=1
    )
    
    # 2. Success indicator
    fig.add_trace(
        go.Indicator(
            mode="gauge+number+delta",
            value=result['achieved_prediction'],
            title={'text': "Achieved Prediction"},
            delta={'reference': result.get('target_value', 0)},
            gauge={
                'axis': {'range': [0, 1]},
                'bar': {'color': "darkgreen" if result['success'] else "darkred"},
                'steps': [
                    {'range': [0, 0.5], 'color': "lightgray"},
                    {'range': [0.5, 1], 'color': "gray"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': result.get('target_value', 0.5)
                }
            }
        ),
        row=1, col=2
    )
    
    # 3. Feature changes (if original values provided)
    if original_values:
        changes = [optimal_vals.get(f, 0) - original_values.get(f, 0) for f in features]
        colors = ['red' if c < 0 else 'green' for c in changes]
        
        fig.add_trace(
            go.Bar(
                x=features,
                y=changes,
                marker_color=colors,
                name='Changes',
                hovertemplate='<b>%{x}</b><br>Δ: %{y:.3f}<extra></extra>',
            ),
            row=2, col=1
        )
    
    # 4. Summary table
    summary_data = {
        'Metric': [
            'Success',
            'Achieved Prediction',
            'Distance to Target',
            'Method',
            'Function Evals',
        ],
        'Value': [
            '✓' if result['success'] else '✗',
            f"{result['achieved_prediction']:.4f}",
            f"{result.get('distance_to_target', np.nan):.6f}",
            result.get('method', 'N/A'),
            str(result.get('n_function_evaluations', 'N/A')),
        ]
    }
    
    fig.add_trace(
        go.Table(
            header=dict(
                values=list(summary_data.keys()),
                fill_color='paleturquoise',
                align='left',
                font=dict(size=12, color='black')
            ),
            cells=dict(
                values=list(summary_data.values()),
                fill_color='lavender',
                align='left',
                font=dict(size=11)
            )
        ),
        row=2, col=2
    )
    
    fig.update_xaxes(tickangle=-45, row=1, col=1)
    fig.update_xaxes(tickangle=-45, row=2, col=1)
    
    fig.update_layout(
        height=800,
        showlegend=False,
        title_text="Optimization Summary",
        template='plotly_white',
    )
    
    return fig
