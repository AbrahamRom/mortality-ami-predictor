"""Rigorous statistical comparison between ML models and GRACE score.

This module provides comprehensive comparison tools including:
- AUC comparison with DeLong test
- Net Reclassification Improvement (NRI)
- Integrated Discrimination Improvement (IDI)
- Calibration comparison
- Decision curve analysis comparison
- Statistical tests and visualizations
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    brier_score_loss,
    confusion_matrix,
)
from sklearn.calibration import calibration_curve


@dataclass
class ComparisonResult:
    """Results from comparing ML model with GRACE score."""
    
    # Model names (required fields first)
    model_name: str
    
    # AUC values
    model_auc: float
    grace_auc: float
    auc_difference: float
    auc_p_value: float
    auc_ci_lower: float
    auc_ci_upper: float
    
    # NRI (Net Reclassification Improvement)
    nri: float
    nri_p_value: float
    nri_events: float  # NRI for events (cases)
    nri_nonevents: float  # NRI for non-events (controls)
    
    # IDI (Integrated Discrimination Improvement)
    idi: float
    idi_p_value: float
    
    # Calibration metrics
    model_brier: float
    grace_brier: float
    brier_difference: float
    
    # Additional metrics
    model_accuracy: float
    grace_accuracy: float
    model_sensitivity: float
    grace_sensitivity: float
    model_specificity: float
    grace_specificity: float
    
    # Statistical conclusion
    is_model_superior: bool
    superiority_level: str  # "significant", "marginal", "none", "inferior"
    
    # Fields with defaults come last
    grace_name: str = "GRACE Score"
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'model_name': self.model_name,
            'grace_name': self.grace_name,
            'model_auc': self.model_auc,
            'grace_auc': self.grace_auc,
            'auc_difference': self.auc_difference,
            'auc_p_value': self.auc_p_value,
            'auc_ci_lower': self.auc_ci_lower,
            'auc_ci_upper': self.auc_ci_upper,
            'nri': self.nri,
            'nri_p_value': self.nri_p_value,
            'nri_events': self.nri_events,
            'nri_nonevents': self.nri_nonevents,
            'idi': self.idi,
            'idi_p_value': self.idi_p_value,
            'model_brier': self.model_brier,
            'grace_brier': self.grace_brier,
            'brier_difference': self.brier_difference,
            'model_accuracy': self.model_accuracy,
            'grace_accuracy': self.grace_accuracy,
            'model_sensitivity': self.model_sensitivity,
            'grace_sensitivity': self.grace_sensitivity,
            'model_specificity': self.model_specificity,
            'grace_specificity': self.grace_specificity,
            'is_model_superior': self.is_model_superior,
            'superiority_level': self.superiority_level,
            'p_value': self.auc_p_value,
        }

    @property
    def p_value(self) -> float:
        """Backward-compatible alias for the primary p-value (AUC p-value).

        Some parts of the codebase expect `ComparisonResult` to expose a
        `p_value` attribute. The primary test used in this comparison is the
        DeLong test for AUC differences, so expose `auc_p_value` via this
        property to avoid attribute errors and preserve semantics.
        """
        return float(self.auc_p_value)


def delong_test(y_true: np.ndarray, pred1: np.ndarray, pred2: np.ndarray) -> Tuple[float, float]:
    """DeLong test for comparing two correlated ROC curves.
    
    Args:
        y_true: True labels
        pred1: Predictions from model 1
        pred2: Predictions from model 2
        
    Returns:
        Tuple of (z_statistic, p_value)
        
    References:
        DeLong, E. R., DeLong, D. M., & Clarke-Pearson, D. L. (1988).
        Comparing the areas under two or more correlated receiver operating
        characteristic curves: a nonparametric approach. Biometrics, 837-845.
    """
    # CRITICAL: Validate paired samples assumption for DeLong test
    if len(pred1) != len(pred2):
        raise ValueError(
            f"DeLong test requires paired samples (same n). "
            f"Got pred1 n={len(pred1)}, pred2 n={len(pred2)}."
        )
    if len(pred1) != len(y_true):
        raise ValueError(
            f"Predictions must match y_true length. "
            f"Got pred1 n={len(pred1)}, y_true n={len(y_true)}."
        )
    
    auc1 = roc_auc_score(y_true, pred1)
    auc2 = roc_auc_score(y_true, pred2)
    
    # Get indices for positive and negative samples
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]
    
    n_pos = len(pos_idx)
    n_neg = len(neg_idx)
    
    # Compute structural components for each model
    def structural_components(y, pred, pos_idx, neg_idx):
        """Compute structural components V_10 for DeLong test."""
        v10 = np.zeros(len(pos_idx))
        for i, pos_i in enumerate(pos_idx):
            v10[i] = np.mean(pred[pos_i] > pred[neg_idx]) + 0.5 * np.mean(pred[pos_i] == pred[neg_idx])
        return v10
    
    v10_1 = structural_components(y_true, pred1, pos_idx, neg_idx)
    v10_2 = structural_components(y_true, pred2, pos_idx, neg_idx)
    
    v01_1 = 1 - structural_components(y_true, -pred1, neg_idx, pos_idx)  # For negatives
    v01_2 = 1 - structural_components(y_true, -pred2, neg_idx, pos_idx)
    
    # Covariance estimation
    s10_1 = np.var(v10_1, ddof=1)
    s10_2 = np.var(v10_2, ddof=1)
    s01_1 = np.var(v01_1, ddof=1)
    s01_2 = np.var(v01_2, ddof=1)
    
    # Covariance between the two models
    cov10 = np.cov(v10_1, v10_2, ddof=1)[0, 1]
    cov01 = np.cov(v01_1, v01_2, ddof=1)[0, 1]
    
    # Variance of the difference
    var_diff = (s10_1 / n_pos) + (s10_2 / n_pos) + (s01_1 / n_neg) + (s01_2 / n_neg) - 2 * (cov10 / n_pos + cov01 / n_neg)
    
    # Avoid division by zero
    if var_diff <= 0:
        return 0.0, 1.0
    
    # Z-statistic
    z = (auc1 - auc2) / np.sqrt(var_diff)
    
    # Two-tailed p-value
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    
    return z, p_value


def compute_nri(
    y_true: np.ndarray,
    pred_model: np.ndarray,
    pred_grace: np.ndarray,
    threshold: float = 0.5,
    categories: Optional[List[float]] = None
) -> Tuple[float, float, float, float]:
    """Compute Net Reclassification Improvement (NRI).
    
    Args:
        y_true: True labels
        pred_model: Predictions from ML model
        pred_grace: Predictions from GRACE score
        threshold: Classification threshold (for continuous NRI)
        categories: Risk categories boundaries (e.g., [0.0, 0.1, 0.3, 1.0])
        
    Returns:
        Tuple of (nri_total, nri_events, nri_nonevents, p_value)
    """
    if categories is None:
        # Continuous NRI - based on movement across threshold
        # Events (y=1)
        events_mask = y_true == 1
        model_events = pred_model[events_mask]
        grace_events = pred_grace[events_mask]
        
        # NRI for events: proportion moving up minus proportion moving down
        moved_up_events = np.sum(model_events > grace_events)
        moved_down_events = np.sum(model_events < grace_events)
        nri_events = (moved_up_events - moved_down_events) / np.sum(events_mask)
        
        # Non-events (y=0)
        nonevents_mask = y_true == 0
        model_nonevents = pred_model[nonevents_mask]
        grace_nonevents = pred_grace[nonevents_mask]
        
        # NRI for non-events: proportion moving down minus proportion moving up
        moved_down_nonevents = np.sum(model_nonevents < grace_nonevents)
        moved_up_nonevents = np.sum(model_nonevents > grace_nonevents)
        nri_nonevents = (moved_down_nonevents - moved_up_nonevents) / np.sum(nonevents_mask)
        
    else:
        # Categorical NRI - based on risk categories
        def categorize(probs, cats):
            return np.digitize(probs, cats) - 1
        
        model_cats = categorize(pred_model, categories)
        grace_cats = categorize(pred_grace, categories)
        
        # Events
        events_mask = y_true == 1
        events_improved = np.sum(model_cats[events_mask] > grace_cats[events_mask])
        events_worsened = np.sum(model_cats[events_mask] < grace_cats[events_mask])
        nri_events = (events_improved - events_worsened) / np.sum(events_mask)
        
        # Non-events
        nonevents_mask = y_true == 0
        nonevents_improved = np.sum(model_cats[nonevents_mask] < grace_cats[nonevents_mask])
        nonevents_worsened = np.sum(model_cats[nonevents_mask] > grace_cats[nonevents_mask])
        nri_nonevents = (nonevents_improved - nonevents_worsened) / np.sum(nonevents_mask)
    
    # Total NRI
    nri_total = nri_events + nri_nonevents
    
    # Approximate p-value using normal approximation
    # Standard error estimation
    n_events = np.sum(y_true == 1)
    n_nonevents = np.sum(y_true == 0)
    se_nri = np.sqrt((nri_events ** 2 / n_events) + (nri_nonevents ** 2 / n_nonevents))
    
    if se_nri > 0:
        z_stat = nri_total / se_nri
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    else:
        p_value = 1.0
    
    return nri_total, nri_events, nri_nonevents, p_value


def compute_idi(
    y_true: np.ndarray,
    pred_model: np.ndarray,
    pred_grace: np.ndarray
) -> Tuple[float, float]:
    """Compute Integrated Discrimination Improvement (IDI).
    
    Args:
        y_true: True labels
        pred_model: Predictions from ML model
        pred_grace: Predictions from GRACE score
        
    Returns:
        Tuple of (idi, p_value)
    """
    # Separate events and non-events
    events_mask = y_true == 1
    nonevents_mask = y_true == 0
    
    # Mean predictions for events
    model_events_mean = np.mean(pred_model[events_mask])
    grace_events_mean = np.mean(pred_grace[events_mask])
    
    # Mean predictions for non-events
    model_nonevents_mean = np.mean(pred_model[nonevents_mask])
    grace_nonevents_mean = np.mean(pred_grace[nonevents_mask])
    
    # IDI = (IS_new - IS_old)
    # where IS (Integrated Sensitivity) = mean(pred|event) - mean(pred|nonevent)
    is_model = model_events_mean - model_nonevents_mean
    is_grace = grace_events_mean - grace_nonevents_mean
    
    idi = is_model - is_grace
    
    # P-value using bootstrap or normal approximation
    # Here we use a simple t-test approach
    n_events = np.sum(events_mask)
    n_nonevents = np.sum(nonevents_mask)
    
    # Standard error estimation
    se_events = np.sqrt(np.var(pred_model[events_mask]) / n_events + np.var(pred_grace[events_mask]) / n_events)
    se_nonevents = np.sqrt(np.var(pred_model[nonevents_mask]) / n_nonevents + np.var(pred_grace[nonevents_mask]) / n_nonevents)
    se_idi = np.sqrt(se_events**2 + se_nonevents**2)
    
    if se_idi > 0:
        z_stat = idi / se_idi
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    else:
        p_value = 1.0
    
    return idi, p_value


def compare_with_grace(
    y_true: np.ndarray,
    y_pred_model: np.ndarray,
    y_pred_grace: np.ndarray,
    model_name: str = "ML Model",
    threshold: float = 0.5,
    alpha: float = 0.05
) -> ComparisonResult:
    """Comprehensive comparison of ML model with GRACE score.
    
    Args:
        y_true: True labels
        y_pred_model: Predictions from ML model (probabilities)
        y_pred_grace: Predictions from GRACE score (probabilities or risk scores normalized)
        model_name: Name of the ML model
        threshold: Classification threshold
        alpha: Significance level for statistical tests
        
    Returns:
        ComparisonResult object with all metrics and test results
    """
    # AUC comparison with DeLong test
    auc_model = roc_auc_score(y_true, y_pred_model)
    auc_grace = roc_auc_score(y_true, y_pred_grace)
    auc_diff = auc_model - auc_grace
    
    z_stat, p_value_auc = delong_test(y_true, y_pred_model, y_pred_grace)
    
    # Confidence interval for AUC difference (using bootstrap would be better, but approximate here)
    se_diff = abs(auc_diff / z_stat) if z_stat != 0 else 0
    ci_lower = auc_diff - 1.96 * se_diff
    ci_upper = auc_diff + 1.96 * se_diff
    
    # NRI
    nri_total, nri_events, nri_nonevents, p_value_nri = compute_nri(
        y_true, y_pred_model, y_pred_grace, threshold
    )
    
    # IDI
    idi, p_value_idi = compute_idi(y_true, y_pred_model, y_pred_grace)
    
    # Calibration - Brier score
    brier_model = brier_score_loss(y_true, y_pred_model)
    brier_grace = brier_score_loss(y_true, y_pred_grace)
    brier_diff = brier_model - brier_grace  # Negative is better for model
    
    # Confusion matrix metrics
    y_pred_model_binary = (y_pred_model >= threshold).astype(int)
    y_pred_grace_binary = (y_pred_grace >= threshold).astype(int)
    
    cm_model = confusion_matrix(y_true, y_pred_model_binary)
    cm_grace = confusion_matrix(y_true, y_pred_grace_binary)
    
    # Metrics
    tn_m, fp_m, fn_m, tp_m = cm_model.ravel() if cm_model.size == 4 else (0, 0, 0, 0)
    tn_g, fp_g, fn_g, tp_g = cm_grace.ravel() if cm_grace.size == 4 else (0, 0, 0, 0)
    
    acc_model = (tp_m + tn_m) / (tp_m + tn_m + fp_m + fn_m) if (tp_m + tn_m + fp_m + fn_m) > 0 else 0
    acc_grace = (tp_g + tn_g) / (tp_g + tn_g + fp_g + fn_g) if (tp_g + tn_g + fp_g + fn_g) > 0 else 0
    
    sens_model = tp_m / (tp_m + fn_m) if (tp_m + fn_m) > 0 else 0
    sens_grace = tp_g / (tp_g + fn_g) if (tp_g + fn_g) > 0 else 0
    
    spec_model = tn_m / (tn_m + fp_m) if (tn_m + fp_m) > 0 else 0
    spec_grace = tn_g / (tn_g + fp_g) if (tn_g + fp_g) > 0 else 0
    
    # Determine superiority
    is_superior = auc_diff > 0 and p_value_auc < alpha
    
    if p_value_auc < 0.001:
        superiority_level = "highly_significant"
    elif p_value_auc < 0.01:
        superiority_level = "significant"
    elif p_value_auc < 0.05:
        superiority_level = "marginal"
    elif auc_diff > 0:
        superiority_level = "favorable_trend"
    elif auc_diff < 0:
        superiority_level = "inferior"
    else:
        superiority_level = "equivalent"
    
    return ComparisonResult(
        model_name=model_name,
        grace_name="GRACE Score",
        model_auc=auc_model,
        grace_auc=auc_grace,
        auc_difference=auc_diff,
        auc_p_value=p_value_auc,
        auc_ci_lower=ci_lower,
        auc_ci_upper=ci_upper,
        nri=nri_total,
        nri_p_value=p_value_nri,
        nri_events=nri_events,
        nri_nonevents=nri_nonevents,
        idi=idi,
        idi_p_value=p_value_idi,
        model_brier=brier_model,
        grace_brier=brier_grace,
        brier_difference=brier_diff,
        model_accuracy=acc_model,
        grace_accuracy=acc_grace,
        model_sensitivity=sens_model,
        grace_sensitivity=sens_grace,
        model_specificity=spec_model,
        grace_specificity=spec_grace,
        is_model_superior=is_superior,
        superiority_level=superiority_level,
    )


def plot_roc_comparison(
    y_true: np.ndarray,
    y_pred_model: np.ndarray,
    y_pred_grace: np.ndarray,
    comparison_result: ComparisonResult
) -> go.Figure:
    """Plot ROC curves comparison between model and GRACE.
    
    Args:
        y_true: True labels
        y_pred_model: Model predictions
        y_pred_grace: GRACE predictions
        comparison_result: ComparisonResult object
        
    Returns:
        Plotly Figure
    """
    # Compute ROC curves
    fpr_model, tpr_model, _ = roc_curve(y_true, y_pred_model)
    fpr_grace, tpr_grace, _ = roc_curve(y_true, y_pred_grace)
    
    fig = go.Figure()
    
    # Model ROC
    fig.add_trace(go.Scatter(
        x=fpr_model,
        y=tpr_model,
        mode='lines',
        name=f'{comparison_result.model_name} (AUC={comparison_result.model_auc:.3f})',
        line=dict(color='blue', width=3),
        hovertemplate='FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra></extra>'
    ))
    
    # GRACE ROC
    fig.add_trace(go.Scatter(
        x=fpr_grace,
        y=tpr_grace,
        mode='lines',
        name=f'GRACE Score (AUC={comparison_result.grace_auc:.3f})',
        line=dict(color='red', width=3, dash='dash'),
        hovertemplate='FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra></extra>'
    ))
    
    # Diagonal reference
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        name='Random',
        line=dict(color='gray', width=1, dash='dot'),
        hoverinfo='skip'
    ))
    
    # Add annotations for statistical significance
    annotation_text = f"ΔAUC = {comparison_result.auc_difference:+.3f}<br>"
    annotation_text += f"p-value = {comparison_result.p_value:.4f}<br>"
    
    if comparison_result.is_model_superior:
        annotation_text += "✅ <b>Model is statistically superior</b>"
        annotation_color = "green"
    elif comparison_result.superiority_level == "inferior":
        annotation_text += "⚠️ <b>Model is inferior to GRACE</b>"
        annotation_color = "red"
    else:
        annotation_text += "ℹ️ No significant difference"
        annotation_color = "orange"
    
    fig.add_annotation(
        x=0.6, y=0.2,
        text=annotation_text,
        showarrow=False,
        bgcolor=annotation_color,
        opacity=0.8,
        font=dict(color="white", size=12),
        bordercolor="white",
        borderwidth=2
    )
    
    fig.update_layout(
        title=f'ROC Curve Comparison: {comparison_result.model_name} vs GRACE Score',
        xaxis=dict(title='False Positive Rate', range=[0, 1]),
        yaxis=dict(title='True Positive Rate', range=[0, 1]),
        width=800,
        height=600,
        template='plotly_white',
        hovermode='closest',
        legend=dict(x=0.7, y=0.1)
    )
    
    return fig


def plot_calibration_comparison(
    y_true: np.ndarray,
    y_pred_model: np.ndarray,
    y_pred_grace: np.ndarray,
    comparison_result: ComparisonResult,
    n_bins: int = 10
) -> go.Figure:
    """Plot calibration curves comparison.
    
    Args:
        y_true: True labels
        y_pred_model: Model predictions
        y_pred_grace: GRACE predictions
        comparison_result: ComparisonResult object
        n_bins: Number of bins for calibration
        
    Returns:
        Plotly Figure
    """
    # Compute calibration curves
    try:
        prob_true_model, prob_pred_model = calibration_curve(
            y_true, y_pred_model, n_bins=n_bins, strategy='uniform'
        )
        prob_true_grace, prob_pred_grace = calibration_curve(
            y_true, y_pred_grace, n_bins=n_bins, strategy='uniform'
        )
    except ValueError:
        # Handle edge cases
        return go.Figure().add_annotation(
            text="Insufficient data for calibration analysis",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
    
    fig = go.Figure()
    
    # Perfect calibration line
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        name='Perfect Calibration',
        line=dict(color='gray', width=2, dash='dash'),
        hoverinfo='skip'
    ))
    
    # Model calibration
    fig.add_trace(go.Scatter(
        x=prob_pred_model,
        y=prob_true_model,
        mode='lines+markers',
        name=f'{comparison_result.model_name} (Brier={comparison_result.model_brier:.3f})',
        line=dict(color='blue', width=3),
        marker=dict(size=10),
        hovertemplate='Predicted: %{x:.3f}<br>Observed: %{y:.3f}<extra></extra>'
    ))
    
    # GRACE calibration
    fig.add_trace(go.Scatter(
        x=prob_pred_grace,
        y=prob_true_grace,
        mode='lines+markers',
        name=f'GRACE Score (Brier={comparison_result.grace_brier:.3f})',
        line=dict(color='red', width=3, dash='dot'),
        marker=dict(size=10, symbol='square'),
        hovertemplate='Predicted: %{x:.3f}<br>Observed: %{y:.3f}<extra></extra>'
    ))
    
    fig.update_layout(
        title='Calibration Comparison',
        xaxis=dict(title='Predicted Probability', range=[0, 1]),
        yaxis=dict(title='Observed Frequency', range=[0, 1]),
        width=800,
        height=600,
        template='plotly_white',
        hovermode='closest',
        legend=dict(x=0.05, y=0.95)
    )
    
    return fig


def plot_metrics_comparison(comparison_result: ComparisonResult) -> go.Figure:
    """Plot bar chart comparing all metrics.
    
    Args:
        comparison_result: ComparisonResult object
        
    Returns:
        Plotly Figure
    """
    metrics = ['AUC', 'Accuracy', 'Sensitivity', 'Specificity']
    model_values = [
        comparison_result.model_auc,
        comparison_result.model_accuracy,
        comparison_result.model_sensitivity,
        comparison_result.model_specificity
    ]
    grace_values = [
        comparison_result.grace_auc,
        comparison_result.grace_accuracy,
        comparison_result.grace_sensitivity,
        comparison_result.grace_specificity
    ]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name=comparison_result.model_name,
        x=metrics,
        y=model_values,
        marker_color='blue',
        text=[f'{v:.3f}' for v in model_values],
        textposition='auto',
        hovertemplate='%{x}: %{y:.3f}<extra></extra>'
    ))
    
    fig.add_trace(go.Bar(
        name='GRACE Score',
        x=metrics,
        y=grace_values,
        marker_color='red',
        text=[f'{v:.3f}' for v in grace_values],
        textposition='auto',
        hovertemplate='%{x}: %{y:.3f}<extra></extra>'
    ))
    
    fig.update_layout(
        title='Performance Metrics Comparison',
        yaxis=dict(title='Score', range=[0, 1.1]),
        barmode='group',
        width=800,
        height=500,
        template='plotly_white',
        showlegend=True
    )
    
    return fig


def plot_nri_idi(comparison_result: ComparisonResult) -> go.Figure:
    """Plot NRI and IDI results.
    
    Args:
        comparison_result: ComparisonResult object
        
    Returns:
        Plotly Figure
    """
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Net Reclassification Improvement', 'Integrated Discrimination Improvement'),
        specs=[[{"type": "bar"}, {"type": "indicator"}]]
    )
    
    # NRI bar chart
    fig.add_trace(
        go.Bar(
            x=['Events', 'Non-Events', 'Total'],
            y=[comparison_result.nri_events, comparison_result.nri_nonevents, comparison_result.nri],
            marker_color=['green', 'orange', 'blue'],
            text=[f'{comparison_result.nri_events:.3f}', 
                  f'{comparison_result.nri_nonevents:.3f}',
                  f'{comparison_result.nri:.3f}'],
            textposition='auto',
            hovertemplate='%{x}: %{y:.3f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # IDI indicator
    fig.add_trace(
        go.Indicator(
            mode="number+delta",
            value=comparison_result.idi,
            delta={'reference': 0, 'relative': False},
            title={'text': f"IDI<br><span style='font-size:0.8em'>p={comparison_result.idi_p_value:.4f}</span>"},
            domain={'x': [0, 1], 'y': [0, 1]}
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        height=400,
        showlegend=False,
        template='plotly_white'
    )
    
    fig.update_yaxes(title_text="NRI", row=1, col=1)
    
    return fig


def generate_comparison_report(comparison_result: ComparisonResult) -> pd.DataFrame:
    """Generate comprehensive comparison report as DataFrame.
    
    Args:
        comparison_result: ComparisonResult object
        
    Returns:
        DataFrame with all comparison metrics
    """
    report_data = [
        {
            'Metric': 'AUC',
            'Model': f'{comparison_result.model_auc:.4f}',
            'GRACE': f'{comparison_result.grace_auc:.4f}',
            'Difference': f'{comparison_result.auc_difference:+.4f}',
            'P-value': f'{comparison_result.auc_p_value:.4f}',
            'CI 95%': f'[{comparison_result.auc_ci_lower:.4f}, {comparison_result.auc_ci_upper:.4f}]'
        },
        {
            'Metric': 'Accuracy',
            'Model': f'{comparison_result.model_accuracy:.4f}',
            'GRACE': f'{comparison_result.grace_accuracy:.4f}',
            'Difference': f'{comparison_result.model_accuracy - comparison_result.grace_accuracy:+.4f}',
            'P-value': '-',
            'CI 95%': '-'
        },
        {
            'Metric': 'Sensitivity',
            'Model': f'{comparison_result.model_sensitivity:.4f}',
            'GRACE': f'{comparison_result.grace_sensitivity:.4f}',
            'Difference': f'{comparison_result.model_sensitivity - comparison_result.grace_sensitivity:+.4f}',
            'P-value': '-',
            'CI 95%': '-'
        },
        {
            'Metric': 'Specificity',
            'Model': f'{comparison_result.model_specificity:.4f}',
            'GRACE': f'{comparison_result.grace_specificity:.4f}',
            'Difference': f'{comparison_result.model_specificity - comparison_result.grace_specificity:+.4f}',
            'P-value': '-',
            'CI 95%': '-'
        },
        {
            'Metric': 'Brier Score',
            'Model': f'{comparison_result.model_brier:.4f}',
            'GRACE': f'{comparison_result.grace_brier:.4f}',
            'Difference': f'{comparison_result.brier_difference:+.4f}',
            'P-value': '-',
            'CI 95%': '-'
        },
        {
            'Metric': 'NRI (Total)',
            'Model': f'{comparison_result.nri:.4f}',
            'GRACE': '-',
            'Difference': f'{comparison_result.nri:+.4f}',
            'P-value': f'{comparison_result.nri_p_value:.4f}',
            'CI 95%': '-'
        },
        {
            'Metric': 'IDI',
            'Model': f'{comparison_result.idi:.4f}',
            'GRACE': '-',
            'Difference': f'{comparison_result.idi:+.4f}',
            'P-value': f'{comparison_result.idi_p_value:.4f}',
            'CI 95%': '-'
        },
    ]
    
    df = pd.DataFrame(report_data)
    
    return df
