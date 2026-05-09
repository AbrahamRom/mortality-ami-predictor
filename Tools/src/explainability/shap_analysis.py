"""SHAP (SHapley Additive exPlanations) analysis utilities."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


# Use new processed/plots structure
ROOT_DIR = Path(__file__).parents[2]
FIG_DIR = ROOT_DIR / "processed" / "plots" / "explainability"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def compute_shap_values(
    model,
    X: Union[np.ndarray, pd.DataFrame],
    feature_names: Optional[list] = None,
    max_samples: int = 100,
) -> Tuple[shap.Explainer, shap.Explanation]:
    """Compute SHAP values for a model with proper handling of multi-output models.
    
    Args:
        model: Trained model with predict method
        X: Feature matrix
        feature_names: List of feature names
        max_samples: Maximum samples for background data
        
    Returns:
        Tuple of (explainer, shap_explanation)
        
    Raises:
        ImportError: If SHAP is not installed
    """
    if not SHAP_AVAILABLE:
        raise ImportError(
            "SHAP is not installed. Install with: pip install shap"
        )
    
    # Convert to DataFrame if needed
    if isinstance(X, np.ndarray):
        if feature_names is not None:
            X_df = pd.DataFrame(X, columns=feature_names)
        else:
            X_df = pd.DataFrame(X)
    else:
        X_df = X.copy()
        if feature_names is None:
            feature_names = list(X_df.columns)
    
    # Ensure all columns are numeric - SHAP requires numeric data
    # This fixes: "ufunc 'isfinite' not supported for the input types"
    for col in X_df.columns:
        if X_df[col].dtype == 'object' or X_df[col].dtype.name == 'category':
            # Try to convert to numeric, coerce errors to NaN
            X_df[col] = pd.to_numeric(X_df[col], errors='coerce')
        elif not np.issubdtype(X_df[col].dtype, np.number):
            # Force conversion for any other non-numeric type
            X_df[col] = pd.to_numeric(X_df[col], errors='coerce')
    
    # Convert entire DataFrame to float64 to ensure compatibility
    X_df = X_df.astype(np.float64)
    
    # Handle any remaining NaN/inf values
    X_df = X_df.replace([np.inf, -np.inf], np.nan)
    X_df = X_df.fillna(X_df.median())
    
    # If still has NaN (e.g., all-NaN columns), fill with 0
    X_df = X_df.fillna(0)
    
    # Create explainer based on model type
    try:
        # Try TreeExplainer first (faster for tree-based models)
        explainer = shap.TreeExplainer(model)
        shap_values_raw = explainer.shap_values(X_df)
        base_value = explainer.expected_value
    except Exception:
        # Fall back to general Explainer
        if hasattr(model, "predict_proba"):
            explainer = shap.Explainer(model.predict_proba, X_df[:max_samples])
        else:
            explainer = shap.Explainer(model.predict, X_df[:max_samples])
        
        shap_values_raw = explainer(X_df)
        base_value = explainer.expected_value if hasattr(explainer, 'expected_value') else None
    
    # Handle multi-output models (binary classification returns 2 outputs)
    if isinstance(shap_values_raw, list) and len(shap_values_raw) == 2:
        # For binary classification, use positive class (index 1)
        shap_values_raw = shap_values_raw[1]
        if isinstance(base_value, (list, np.ndarray)):
            base_value = base_value[1]
    
    # If shap_values_raw is an Explanation object, extract and reshape
    if hasattr(shap_values_raw, 'values'):
        # It's already an Explanation object
        if shap_values_raw.values.ndim > 2:
            # Multi-output: take last dimension (positive class)
            shap_values_array = shap_values_raw.values[..., -1]
            # Handle base_values for multi-output
            if hasattr(shap_values_raw, 'base_values'):
                bv = shap_values_raw.base_values
                if isinstance(bv, np.ndarray) and bv.ndim > 1:
                    base_values_array = bv[..., -1]
                else:
                    base_values_array = bv
            else:
                base_values_array = base_value
        else:
            shap_values_array = shap_values_raw.values
            base_values_array = shap_values_raw.base_values if hasattr(shap_values_raw, 'base_values') else base_value
        
        # Create new Explanation object with correct shape
        shap_explanation = shap.Explanation(
            values=shap_values_array,
            base_values=base_values_array,
            data=shap_values_raw.data if hasattr(shap_values_raw, 'data') else X_df.values,
            feature_names=feature_names
        )
    else:
        # It's a numpy array
        shap_explanation = shap.Explanation(
            values=shap_values_raw,
            base_values=base_value,
            data=X_df.values,
            feature_names=feature_names
        )
    
    return explainer, shap_explanation


def plot_shap_beeswarm(
    shap_explanation: shap.Explanation,
    max_display: int = 20,
    save_path: Optional[str] = None,
) -> go.Figure:
    """Create SHAP beeswarm plot using Plotly for interactivity.
    
    Args:
        shap_explanation: SHAP Explanation object
        max_display: Maximum features to display
        save_path: Optional path to save as static image
        
    Returns:
        Plotly Figure object
    """
    if not SHAP_AVAILABLE:
        raise ImportError("SHAP is not installed. Install with: pip install shap")
    
    # Get SHAP values
    shap_vals = shap_explanation.values
    if shap_vals.ndim > 2:
        shap_vals = shap_vals[..., -1]
    
    # Get feature names
    if hasattr(shap_explanation, 'feature_names') and shap_explanation.feature_names is not None:
        feature_names = shap_explanation.feature_names
    else:
        feature_names = [f"Feature {i}" for i in range(shap_vals.shape[1])]
    
    # Get feature values
    if hasattr(shap_explanation, 'data') and shap_explanation.data is not None:
        feature_values = shap_explanation.data
    else:
        feature_values = np.zeros_like(shap_vals)
    
    # Calculate mean absolute SHAP values for sorting
    mean_abs_shap = np.abs(shap_vals).mean(axis=0)
    top_indices = np.argsort(mean_abs_shap)[-max_display:][::-1]
    
    # Create figure
    fig = go.Figure()
    
    # For each feature, create a scatter plot
    for idx in top_indices:
        feature_name = feature_names[idx]
        shap_values_feat = shap_vals[:, idx]
        feature_vals = feature_values[:, idx]
        
        # Add jitter to x-axis for better visualization
        jitter = np.random.uniform(-0.2, 0.2, size=len(shap_values_feat))
        y_positions = np.ones(len(shap_values_feat)) * np.where(top_indices == idx)[0][0]
        
        fig.add_trace(go.Scatter(
            x=shap_values_feat,
            y=y_positions + jitter,
            mode='markers',
            marker=dict(
                size=6,
                color=feature_vals,
                colorscale='RdBu_r',
                showscale=True if idx == top_indices[0] else False,
                colorbar=dict(title="Feature<br>Value", x=1.15),
                line=dict(width=0.5, color='white')
            ),
            name=feature_name,
            showlegend=False,
            hovertemplate=(
                f"<b>{feature_name}</b><br>"
                "SHAP value: %{x:.4f}<br>"
                "Feature value: %{marker.color:.4f}<br>"
                "<extra></extra>"
            )
        ))
    
    # Update layout
    fig.update_layout(
        title=f'SHAP Beeswarm Plot (Top {max_display} Features)',
        xaxis_title='SHAP Value (impact on model output)',
        yaxis=dict(
            tickmode='array',
            tickvals=list(range(len(top_indices))),
            ticktext=[feature_names[i] for i in top_indices],
            title=''
        ),
        height=max(400, len(top_indices) * 25),
        template='plotly_white',
        hovermode='closest'
    )
    
    # Add vertical line at x=0
    fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)
    
    if save_path:
        fig.write_image(save_path, width=1000, height=max(400, len(top_indices) * 25))
    
    return fig


def plot_shap_bar(
    shap_explanation: shap.Explanation,
    max_display: int = 20,
    save_path: Optional[str] = None,
) -> go.Figure:
    """Create SHAP bar plot (feature importance) using Plotly.
    
    Args:
        shap_explanation: SHAP Explanation object
        max_display: Maximum features to display
        save_path: Optional path to save as static image
        
    Returns:
        Plotly Figure object
    """
    if not SHAP_AVAILABLE:
        raise ImportError("SHAP is not installed. Install with: pip install shap")
    
    # Get SHAP values
    shap_vals = shap_explanation.values
    if shap_vals.ndim > 2:
        shap_vals = shap_vals[..., -1]
    
    # Get feature names
    if hasattr(shap_explanation, 'feature_names') and shap_explanation.feature_names is not None:
        feature_names = shap_explanation.feature_names
    else:
        feature_names = [f"Feature {i}" for i in range(shap_vals.shape[1])]
    
    # Calculate mean absolute SHAP values
    mean_abs_shap = np.abs(shap_vals).mean(axis=0)
    
    # Sort and get top features
    sorted_indices = np.argsort(mean_abs_shap)[-max_display:]
    sorted_features = [feature_names[i] for i in sorted_indices]
    sorted_values = mean_abs_shap[sorted_indices]
    
    # Create horizontal bar chart
    fig = go.Figure(go.Bar(
        x=sorted_values,
        y=sorted_features,
        orientation='h',
        marker=dict(
            color=sorted_values,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Mean |SHAP|")
        ),
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Mean |SHAP|: %{x:.4f}<br>"
            "<extra></extra>"
        )
    ))
    
    fig.update_layout(
        title=f'Feature Importance (Top {max_display} Features)',
        xaxis_title='Mean |SHAP value|',
        yaxis_title='',
        height=max(400, max_display * 25),
        template='plotly_white',
        showlegend=False
    )
    
    if save_path:
        fig.write_image(save_path, width=1000, height=max(400, max_display * 25))
    
    return fig


def plot_shap_waterfall(
    shap_explanation: shap.Explanation,
    sample_idx: int = 0,
    max_display: int = 20,
    save_path: Optional[str] = None,
) -> go.Figure:
    """Create SHAP waterfall plot for a single sample using Plotly.
    
    Args:
        shap_explanation: SHAP Explanation object
        sample_idx: Index of sample to explain
        max_display: Maximum features to display
        save_path: Optional path to save as static image
        
    Returns:
        Plotly Figure object
    """
    if not SHAP_AVAILABLE:
        raise ImportError("SHAP is not installed. Install with: pip install shap")
    
    # Get the explanation for this sample
    sample_explanation = shap_explanation[sample_idx]
    
    # Get base value
    base_value = sample_explanation.base_values
    if isinstance(base_value, np.ndarray):
        base_value = float(base_value[0]) if len(base_value) > 0 else 0.0
    else:
        base_value = float(base_value) if base_value is not None else 0.0
    
    # Get SHAP values and feature values
    shap_vals = sample_explanation.values
    feature_vals = sample_explanation.data
    feature_names = sample_explanation.feature_names
    
    # Handle multi-dimensional SHAP values (e.g., multiclass)
    # For binary classification, SHAP sometimes returns values for class 1 only
    if shap_vals.ndim > 1:
        # Take the last class (typically the positive class in binary classification)
        shap_vals = shap_vals[:, -1] if shap_vals.shape[1] > 1 else shap_vals[:, 0]
    
    # Sort by absolute SHAP value and get top features
    abs_shap = np.abs(shap_vals)
    sorted_indices = np.argsort(abs_shap)[-max_display:][::-1]
    
    # Build waterfall data
    features_display = []
    values_display = []
    cumulative = base_value
    cumulative_values = [base_value]
    
    for idx in sorted_indices:
        idx = idx.item() if hasattr(idx, 'item') else int(idx)  # Convert numpy integer to Python int
        feat_name = feature_names[idx]
        shap_val = float(shap_vals[idx])  # Also ensure scalar
        feat_val = float(feature_vals[idx])  # Also ensure scalar
        
        features_display.append(f"{feat_name} = {feat_val:.2f}")
        values_display.append(shap_val)
        cumulative += shap_val
        cumulative_values.append(cumulative)
    
    # Add final prediction
    features_display.insert(0, "Base Value")
    values_display.insert(0, 0)
    features_display.append("f(x)")
    values_display.append(0)
    cumulative_values.append(cumulative)
    
    # Create waterfall chart
    fig = go.Figure()
    
    # Add bars for each feature
    colors = ['lightgray'] + ['#FF0D57' if v > 0 else '#1E88E5' for v in values_display[1:-1]] + ['lightgray']
    
    for i in range(len(features_display)):
        if i == 0:
            # Base value
            fig.add_trace(go.Bar(
                x=[features_display[i]],
                y=[cumulative_values[i]],
                marker_color=colors[i],
                name=features_display[i],
                showlegend=False,
                hovertemplate=f"<b>{features_display[i]}</b><br>Value: {cumulative_values[i]:.4f}<extra></extra>"
            ))
        elif i == len(features_display) - 1:
            # Final prediction
            fig.add_trace(go.Bar(
                x=[features_display[i]],
                y=[cumulative_values[i]],
                marker_color=colors[i],
                name=features_display[i],
                showlegend=False,
                hovertemplate=f"<b>{features_display[i]}</b><br>Value: {cumulative_values[i]:.4f}<extra></extra>"
            ))
        else:
            # Feature contributions
            y_start = cumulative_values[i]
            y_end = cumulative_values[i+1]
            fig.add_trace(go.Bar(
                x=[features_display[i]],
                y=[y_end],
                base=y_start if values_display[i] < 0 else y_start,
                marker_color=colors[i],
                name=features_display[i],
                showlegend=False,
                hovertemplate=f"<b>{features_display[i]}</b><br>SHAP: {values_display[i]:.4f}<br>Cumulative: {y_end:.4f}<extra></extra>"
            ))
    
    fig.update_layout(
        title=f'Waterfall Plot - Sample {sample_idx}<br>Base: {base_value:.4f} → Prediction: {cumulative:.4f}',
        xaxis_title='',
        yaxis_title='Model Output',
        height=max(500, max_display * 30),
        template='plotly_white',
        showlegend=False,
        barmode='overlay'
    )
    
    if save_path:
        fig.write_image(save_path, width=1200, height=max(500, max_display * 30))
    
    return fig


def plot_shap_force(
    shap_explanation: shap.Explanation,
    sample_idx: int = 0,
    max_display: int = 20,
    save_path: Optional[str] = None,
) -> go.Figure:
    """Create SHAP force plot for a single sample using Plotly.
    
    This creates an interactive horizontal bar chart showing how each feature
    pushes the prediction from the base value. Optimized for many features.
    
    Args:
        shap_explanation: SHAP Explanation object
        sample_idx: Index of sample to explain
        max_display: Maximum features to display (default 20 to avoid overcrowding)
        save_path: Optional path to save as static image
        
    Returns:
        Plotly Figure object (interactive and readable even with many features)
    """
    if not SHAP_AVAILABLE:
        raise ImportError("SHAP is not installed. Install with: pip install shap")
    
    # Get the explanation for this sample
    sample_explanation = shap_explanation[sample_idx]
    
    # Get base value
    base_value = sample_explanation.base_values
    if isinstance(base_value, np.ndarray):
        base_value = float(base_value[0]) if len(base_value) > 0 else 0.0
    else:
        base_value = float(base_value) if base_value is not None else 0.0
    
    # Get SHAP values and feature values
    shap_vals = sample_explanation.values
    feature_vals = sample_explanation.data
    feature_names = sample_explanation.feature_names
    
    # Handle multi-dimensional SHAP values (e.g., multiclass)
    if shap_vals.ndim > 1:
        shap_vals = shap_vals[:, -1] if shap_vals.shape[1] > 1 else shap_vals[:, 0]
    
    # Sort by absolute SHAP value and get top features
    idx_sorted = np.argsort(np.abs(shap_vals))[-max_display:][::-1]
    
    # Prepare data for plotting
    feature_labels = []
    shap_values_sorted = []
    colors = []
    
    for i in idx_sorted:
        i = i.item() if hasattr(i, 'item') else int(i)  # Convert numpy integer to Python int
        # Format feature label with value
        feat_val_str = f"{feature_vals[i]:.2f}" if isinstance(feature_vals[i], (int, float, np.number)) else str(feature_vals[i])
        feature_labels.append(f"{feature_names[i]}")
        shap_values_sorted.append(float(shap_vals[i]))  # Ensure scalar
        colors.append('#FF0D57' if shap_vals[i] > 0 else '#1E88E5')
    
    # Create horizontal bar plot
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=feature_labels,
        x=shap_values_sorted,
        orientation='h',
        marker=dict(
            color=colors,
            line=dict(width=1, color='white')
        ),
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Feature Value: %{customdata[0]}<br>"
            "SHAP Value: %{x:.4f}<br>"
            "Impact: %{customdata[1]}<br>"
            "<extra></extra>"
        ),
        customdata=[[
            f"{feature_vals[i.item() if hasattr(i, 'item') else int(i)]:.2f}" if isinstance(feature_vals[i.item() if hasattr(i, 'item') else int(i)], (int, float, np.number)) else str(feature_vals[i.item() if hasattr(i, 'item') else int(i)]),
            "Increases prediction" if shap_vals[i.item() if hasattr(i, 'item') else int(i)] > 0 else "Decreases prediction"
        ] for i in idx_sorted]
    ))
    
    # Calculate final prediction
    prediction = base_value + np.sum(shap_vals)
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f'Force Plot - Sample {sample_idx}<br>'
                 f'<sub>Base Value: {base_value:.4f} → Prediction: {prediction:.4f} '
                 f'(Showing top {max_display} of {len(shap_vals)} features)</sub>',
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(
            title='SHAP Value (impact on model output)',
            zeroline=True,
            zerolinewidth=2,
            zerolinecolor='black',
            gridcolor='lightgray'
        ),
        yaxis=dict(
            title='',
            autorange='reversed',  # Most important at top
        ),
        height=max(600, max_display * 30),  # Dynamic height based on features
        width=1200,  # Wide plot for readability
        template='plotly_white',
        showlegend=False,
        hovermode='closest',
        margin=dict(l=200, r=50, t=100, b=50)  # Extra left margin for feature names
    )
    
    # Add annotations for positive/negative impact
    fig.add_annotation(
        x=0.98, y=0.98,
        xref='paper', yref='paper',
        text='<b style="color:#FF0D57">█</b> Pushes higher<br><b style="color:#1E88E5">█</b> Pushes lower',
        showarrow=False,
        font=dict(size=10),
        align='right',
        bgcolor='rgba(255,255,255,0.8)',
        bordercolor='gray',
        borderwidth=1
    )
    
    if save_path:
        fig.write_image(save_path, width=1200, height=max(600, max_display * 30))
    
    return fig


def get_feature_importance(
    shap_explanation: shap.Explanation,
) -> pd.DataFrame:
    """Calculate feature importance from SHAP values.
    
    Args:
        shap_explanation: SHAP Explanation object
        
    Returns:
        DataFrame with feature names and importance scores
    """
    # Get feature names
    if hasattr(shap_explanation, 'feature_names') and shap_explanation.feature_names is not None:
        feature_names = shap_explanation.feature_names
    else:
        feature_names = [f"Feature {i}" for i in range(shap_explanation.values.shape[1])]
    
    # Calculate mean absolute SHAP values
    shap_vals = shap_explanation.values
    if shap_vals.ndim > 2:
        # Multi-output: take last dimension
        shap_vals = shap_vals[..., -1]
    
    mean_shap = np.abs(shap_vals).mean(axis=0)
    
    # Create DataFrame
    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Mean |SHAP|": mean_shap
    }).sort_values("Mean |SHAP|", ascending=False)
    
    return importance_df


def get_sample_shap_values(
    shap_explanation: shap.Explanation,
    sample_idx: int = 0,
) -> pd.DataFrame:
    """Get SHAP values for a single sample as a DataFrame.
    
    Args:
        shap_explanation: SHAP Explanation object
        sample_idx: Index of sample
        
    Returns:
        DataFrame with feature names and SHAP values
    """
    # Get feature names
    if hasattr(shap_explanation, 'feature_names') and shap_explanation.feature_names is not None:
        feature_names = shap_explanation.feature_names
    else:
        feature_names = [f"Feature {i}" for i in range(shap_explanation.values.shape[1])]
    
    # Get SHAP values for this sample
    shap_vals = shap_explanation.values[sample_idx]
    
    # Create DataFrame
    shap_df = pd.DataFrame({
        "Feature": feature_names,
        "SHAP Value": shap_vals
    }).sort_values("SHAP Value", key=abs, ascending=False)
    
    return shap_df
