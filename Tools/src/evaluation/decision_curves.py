"""Decision curve analysis for clinical utility assessment."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Union, Optional

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go


# Use new processed/plots structure
ROOT_DIR = Path(__file__).parents[2]
FIG_DIR = ROOT_DIR / "processed" / "plots" / "evaluation"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def decision_curve_analysis(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    name: str = "model",
    n_thresholds: int = 50,
    save_path: Optional[str] = None,
) -> Union[go.Figure, str]:
    """Perform decision curve analysis using Plotly for interactivity.
    
    Decision curve analysis assesses the clinical utility of a prediction model
    by calculating the net benefit across different probability thresholds.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        name: Model name for filename
        n_thresholds: Number of threshold points to evaluate
        save_path: Optional path to save as PNG (for backward compatibility)
        
    Returns:
        Plotly Figure object (or path if save_path provided)
    """
    thresholds = np.linspace(0.01, 0.99, n_thresholds)
    net_benefits = []
    
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        
        tp = ((y_true == 1) & (y_pred == 1)).sum()
        fp = ((y_true == 0) & (y_pred == 1)).sum()
        n = len(y_true)
        
        # Net benefit = (TP/N) - (FP/N) * (threshold / (1 - threshold))
        pt = threshold / (1 - threshold)
        nb = (tp / n) - (fp / n) * pt
        net_benefits.append(nb)
    
    # Treat all strategy
    prevalence = y_true.mean()
    treat_all_nb = prevalence - (1 - prevalence) * (thresholds / (1 - thresholds))
    
    # Treat none strategy
    treat_none_nb = np.zeros_like(thresholds)
    
    # Create Plotly figure
    fig = go.Figure()
    
    # Model curve
    fig.add_trace(go.Scatter(
        x=thresholds,
        y=net_benefits,
        mode='lines',
        name='Model',
        line=dict(width=2, color='blue'),
        hovertemplate=(
            'Threshold: %{x:.3f}<br>'
            'Net Benefit: %{y:.4f}<br>'
            '<extra></extra>'
        )
    ))
    
    # Treat all
    fig.add_trace(go.Scatter(
        x=thresholds,
        y=treat_all_nb,
        mode='lines',
        name='Treat All',
        line=dict(width=2, color='green', dash='dash'),
        hovertemplate=(
            'Threshold: %{x:.3f}<br>'
            'Net Benefit: %{y:.4f}<br>'
            '<extra></extra>'
        )
    ))
    
    # Treat none
    fig.add_trace(go.Scatter(
        x=thresholds,
        y=treat_none_nb,
        mode='lines',
        name='Treat None',
        line=dict(width=2, color='red', dash='dot'),
        hovertemplate=(
            'Threshold: %{x:.3f}<br>'
            'Net Benefit: %{y:.4f}<br>'
            '<extra></extra>'
        )
    ))
    
    fig.update_layout(
        title=f'Decision Curve Analysis: {name}',
        xaxis=dict(
            title='Threshold Probability',
            gridcolor='lightgray'
        ),
        yaxis=dict(
            title='Net Benefit',
            gridcolor='lightgray'
        ),
        width=800,
        height=600,
        template='plotly_white',
        legend=dict(x=0.7, y=0.95),
        hovermode='x unified'
    )
    
    if save_path:
        fig.write_image(save_path, width=800, height=600)
        return save_path
    
    return fig
