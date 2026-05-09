"""Partial dependence plot utilities."""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.inspection import partial_dependence


# Use new processed/plots structure
ROOT_DIR = Path(__file__).parents[2]
FIG_DIR = ROOT_DIR / "processed" / "plots" / "explainability"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def plot_partial_dependence(
    model,
    X: np.ndarray,
    features: List[int],
    feature_names: Optional[List[str]] = None,
    name: str = "model",
    grid_resolution: int = 20,
    save_path: Optional[str] = None,
) -> Union[go.Figure, str]:
    """Plot partial dependence plots as interactive Plotly figure.
    
    Args:
        model: Trained model
        X: Feature matrix
        features: List of feature indices to plot
        feature_names: List of feature names
        name: Model name for filename
        grid_resolution: Number of grid points for PDP
        save_path: Optional path to save static image. If None, returns interactive figure.
        
    Returns:
        Interactive Plotly figure, or path to saved image if save_path provided
    """
    # Calculate partial dependence
    pd_result = partial_dependence(
        model, X, features=features, grid_resolution=grid_resolution
    )
    
    # Determine grid layout
    n_features = len(features)
    n_cols = min(3, n_features)
    n_rows = (n_features + n_cols - 1) // n_cols
    
    # Create subplots
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=[
            feature_names[i] if feature_names else f"Feature {i}"
            for i in features
        ],
        vertical_spacing=0.15,
        horizontal_spacing=0.10,
    )
    
    # Add traces for each feature
    for idx, feature_idx in enumerate(features):
        row = idx // n_cols + 1
        col = idx % n_cols + 1
        
        grid_values = pd_result["grid_values"][idx]
        pd_values = pd_result["average"][idx]
        
        feature_name = feature_names[feature_idx] if feature_names else f"Feature {feature_idx}"
        
        fig.add_trace(
            go.Scatter(
                x=grid_values,
                y=pd_values,
                mode="lines",
                line=dict(color="#1f77b4", width=2),
                name=feature_name,
                showlegend=False,
                hovertemplate=f"<b>{feature_name}</b><br>" +
                              "Value: %{x:.3f}<br>" +
                              "PD: %{y:.3f}<extra></extra>",
            ),
            row=row,
            col=col,
        )
        
        # Update axes
        fig.update_xaxes(title_text=feature_name, row=row, col=col)
        fig.update_yaxes(title_text="Partial Dependence", row=row, col=col)
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f"Partial Dependence Plots: {name}",
            x=0.5,
            xanchor="center",
            font=dict(size=16),
        ),
        height=400 * n_rows,
        showlegend=False,
        hovermode="closest",
        template="plotly_white",
    )
    
    # Save or return
    if save_path:
        fig.write_image(save_path, width=1200, height=400 * n_rows)
        return save_path
    
    return fig
