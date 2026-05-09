"""Calibration assessment utilities."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple, Union, Optional

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.calibration import calibration_curve


# Use new processed/plots structure
ROOT_DIR = Path(__file__).parents[2]
FIG_DIR = ROOT_DIR / "processed" / "plots" / "evaluation"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def compute_calibration(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    strategy: str = "quantile",
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute calibration curve.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        n_bins: Number of bins
        strategy: Binning strategy ('uniform' or 'quantile')
        
    Returns:
        Tuple of (fraction_positives, mean_predicted_value)
    """
    return calibration_curve(y_true, y_prob, n_bins=n_bins, strategy=strategy)


def plot_calibration_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    name: str = "model",
    n_bins: int = 10,
    save_path: Optional[str] = None,
) -> Union[go.Figure, str]:
    """Plot calibration curve using Plotly for interactivity.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        name: Model name for filename
        n_bins: Number of bins
        save_path: Optional path to save as PNG (for backward compatibility)
        
    Returns:
        Plotly Figure object (or path if save_path provided)
    """
    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="quantile")
    
    fig = go.Figure()
    
    # Model calibration curve
    fig.add_trace(go.Scatter(
        x=mean_pred,
        y=frac_pos,
        mode='lines+markers',
        name='Model',
        line=dict(width=2, color='steelblue'),
        marker=dict(size=8, symbol='square'),
        hovertemplate=(
            'Mean Predicted: %{x:.3f}<br>'
            'Fraction Positive: %{y:.3f}<br>'
            '<extra></extra>'
        )
    ))
    
    # Perfect calibration line
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        name='Perfectly calibrated',
        line=dict(color='black', width=2, dash='dash'),
        opacity=0.5,
        hoverinfo='skip'
    ))
    
    fig.update_layout(
        title=f'Calibration Curve: {name}',
        xaxis=dict(
            title='Mean predicted probability',
            range=[0, 1],
            gridcolor='lightgray'
        ),
        yaxis=dict(
            title='Fraction of positives',
            range=[0, 1],
            gridcolor='lightgray'
        ),
        width=600,
        height=600,
        template='plotly_white',
        legend=dict(x=0.05, y=0.95),
        hovermode='closest'
    )
    
    if save_path:
        fig.write_image(save_path, width=600, height=600)
        return save_path
    
    return fig
