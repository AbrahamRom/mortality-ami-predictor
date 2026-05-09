"""
Rigorous statistical comparison between multiple prediction methods.

This module provides comprehensive tools for comparing 3+ prediction methods
simultaneously, including:
- Global comparison test (Friedman for non-parametric data)
- Pairwise comparisons with DeLong test for correlated ROC curves
- Multiple comparison corrections (Bonferroni, Holm, Benjamini-Hochberg)
- Summary tables and visualization
- Overlaid ROC curves for all methods

References:
- DeLong, E. R., DeLong, D. M., & Clarke-Pearson, D. L. (1988).
  Comparing the areas under two or more correlated receiver operating
  characteristic curves: a nonparametric approach. Biometrics, 837-845.
- Friedman, M. (1937). The use of ranks to avoid the assumption of
  normality implicit in the analysis of variance. JRSS, 99(1), 30-35.
- Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery
  rate: a practical and powerful approach to multiple testing. JRSS, 57(1).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    accuracy_score,
    recall_score,
    precision_score,
    brier_score_loss,
)


@dataclass
class PairwiseComparisonResult:
    """Result from comparing two methods."""
    method1_name: str
    method2_name: str
    auc1: float
    auc2: float
    auc_difference: float
    z_statistic: float
    p_value: float
    p_value_holm: float = 1.0
    p_value_fdr: float = 1.0
    is_significant: bool = False
    effect_size: float = 0.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'method1': self.method1_name,
            'method2': self.method2_name,
            'AUC_method1': self.auc1,
            'AUC_method2': self.auc2,
            'ΔAUC': self.auc_difference,
            'Z': self.z_statistic,
            'p_value': self.p_value,
            'p_value_Holm': self.p_value_holm,
            'p_value_FDR': self.p_value_fdr,
            'Significant': 'Yes' if self.is_significant else 'No',
            'Effect_size': self.effect_size,
        }


@dataclass
class MultiMethodComparisonResult:
    """Results from comparing multiple methods simultaneously."""
    
    method_names: List[str]
    y_true: np.ndarray
    predictions: Dict[str, np.ndarray]  # method_name -> predicted probabilities
    
    # Global test results
    global_test_name: str
    global_test_statistic: float
    global_p_value: float
    global_significant: bool
    
    # AUC values
    aucs: Dict[str, float] = field(default_factory=dict)
    
    # Pairwise comparisons
    pairwise_results: List[PairwiseComparisonResult] = field(default_factory=list)
    
    # Multiple comparison corrections applied
    correction_method: str = "Holm-Bonferroni"
    
    # Diagnostic metrics
    sensitivities: Dict[str, float] = field(default_factory=dict)
    specificities: Dict[str, float] = field(default_factory=dict)
    brier_scores: Dict[str, float] = field(default_factory=dict)
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame for easy viewing."""
        rows = [pwr.to_dict() for pwr in self.pairwise_results]
        return pd.DataFrame(rows)
    
    def summary(self) -> str:
        """Generate text summary."""
        lines = [
            "=" * 100,
            "MULTI-METHOD COMPARISON SUMMARY",
            "=" * 100,
            "",
            f"Number of methods: {len(self.method_names)}",
            f"Methods: {', '.join(self.method_names)}",
            "",
            "GLOBAL TEST (Friedman):",
            f"  Test Statistic: {self.global_test_statistic:.4f}",
            f"  P-value: {self.global_p_value:.6f}",
            f"  Significant (α=0.05): {'YES' if self.global_significant else 'NO'}",
            "",
            "AUC VALUES:",
        ]
        
        # Sort by AUC descending
        sorted_methods = sorted(self.aucs.items(), key=lambda x: x[1], reverse=True)
        for rank, (method, auc) in enumerate(sorted_methods, 1):
            sens = self.sensitivities.get(method, np.nan)
            spec = self.specificities.get(method, np.nan)
            brier = self.brier_scores.get(method, np.nan)
            lines.append(
                f"  {rank}. {method:20s}: AUC={auc:.4f} | Sens={sens:.3f} | Spec={spec:.3f} | Brier={brier:.4f}"
            )
        
        lines.extend([
            "",
            f"PAIRWISE COMPARISONS: {len(self.pairwise_results)} pairs",
            f"Multiple Comparison Correction: {self.correction_method}",
            "",
            "=" * 100,
        ])
        
        return "\n".join(lines)


# =============================================================================
# STATISTICAL TESTS
# =============================================================================

def delong_test(
    y_true: np.ndarray,
    pred1: np.ndarray,
    pred2: np.ndarray
) -> Tuple[float, float, float, float]:
    """
    DeLong test for comparing two correlated ROC curves.
    
    Args:
        y_true: True labels
        pred1: Predictions from method 1
        pred2: Predictions from method 2
        
    Returns:
        Tuple of (auc1, auc2, z_statistic, p_value)
    """
    auc1 = roc_auc_score(y_true, pred1)
    auc2 = roc_auc_score(y_true, pred2)
    
    # Get indices for positive and negative samples
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]
    
    n_pos = len(pos_idx)
    n_neg = len(neg_idx)
    
    if n_pos < 2 or n_neg < 2:
        return auc1, auc2, 0.0, 1.0
    
    # Compute structural components
    def structural_components(pred, pos_idx, neg_idx):
        """Compute V_10 for DeLong test."""
        v10 = np.zeros(len(pos_idx))
        for i, pos_i in enumerate(pos_idx):
            v10[i] = np.mean(pred[pos_i] > pred[neg_idx]) + 0.5 * np.mean(pred[pos_i] == pred[neg_idx])
        return v10
    
    v10_1 = structural_components(pred1, pos_idx, neg_idx)
    v10_2 = structural_components(pred2, pos_idx, neg_idx)
    
    v01_1 = 1 - structural_components(-pred1, neg_idx, pos_idx)
    v01_2 = 1 - structural_components(-pred2, neg_idx, pos_idx)
    
    # Variance computation
    s10_1 = np.var(v10_1, ddof=1) if len(v10_1) > 1 else 0
    s10_2 = np.var(v10_2, ddof=1) if len(v10_2) > 1 else 0
    s01_1 = np.var(v01_1, ddof=1) if len(v01_1) > 1 else 0
    s01_2 = np.var(v01_2, ddof=1) if len(v01_2) > 1 else 0
    
    # Covariances
    cov10 = np.cov(v10_1, v10_2, ddof=1)[0, 1] if len(v10_1) > 1 else 0
    cov01 = np.cov(v01_1, v01_2, ddof=1)[0, 1] if len(v01_1) > 1 else 0
    
    # Variance of difference
    var_diff = (
        (s10_1 / n_pos) + (s10_2 / n_pos) + (s01_1 / n_neg) + (s01_2 / n_neg) -
        2 * (cov10 / n_pos + cov01 / n_neg)
    )
    
    if var_diff <= 0:
        return auc1, auc2, 0.0, 1.0
    
    z = (auc1 - auc2) / np.sqrt(var_diff)
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    
    return auc1, auc2, z, p_value


def friedman_test(
    y_true: np.ndarray,
    predictions: Dict[str, np.ndarray],
) -> Tuple[float, float]:
    """
    Friedman non-parametric test for multiple related samples.
    
    Tests if there are significant differences in AUC across multiple methods.
    
    Args:
        y_true: True labels
        predictions: Dict mapping method names to predicted probabilities
        
    Returns:
        Tuple of (test_statistic, p_value)
    """
    # Compute AUCs for all methods
    aucs = [roc_auc_score(y_true, pred) for pred in predictions.values()]
    
    if len(aucs) < 3:
        warnings.warn(f"Friedman test requires ≥3 methods, got {len(aucs)}")
        return np.nan, np.nan
    
    k = len(aucs)
    
    # Simplified Friedman-like test using ranks of AUCs
    # NOTE: This is a simplified implementation for comparing k independent methods.
    # For true repeated measures (blocked data), use scipy.stats.friedmanchisquare.
    # Current approach: rank the AUC values and use chi-square approximation.
    
    # Rank the AUC values (1 to k)
    sorted_indices = np.argsort(aucs)
    rank_values = np.zeros(k)
    for rank, idx in enumerate(sorted_indices):
        rank_values[idx] = rank + 1
    
    # Friedman-like statistic (simplified for single observation per method)
    R_bar = np.mean(rank_values)  # Expected value: (k+1)/2
    ss_ranks = np.sum((rank_values - R_bar) ** 2)
    
    # Chi-square approximation statistic
    # For k>2 methods, use chi-square distribution with k-1 degrees of freedom
    statistic = (12 * ss_ranks) / (k * (k + 1))
    p_value = 1 - stats.chi2.cdf(statistic, df=k - 1)
    
    return statistic, p_value


def apply_multiple_comparison_correction(
    p_values: np.ndarray,
    method: str = "holm"
) -> np.ndarray:
    """
    Apply multiple comparison correction to p-values.
    
    Args:
        p_values: Array of p-values
        method: "bonferroni", "holm" (default), or "fdr" (Benjamini-Hochberg)
        
    Returns:
        Corrected p-values
    """
    method = method.lower()
    n = len(p_values)
    
    if method == "bonferroni":
        return np.minimum(p_values * n, 1.0)
    
    elif method == "holm":
        # Holm-Bonferroni
        sorted_idx = np.argsort(p_values)
        corrected = np.ones(n)
        for i, idx in enumerate(sorted_idx):
            corrected[idx] = min(p_values[idx] * (n - i), 1.0)
        return corrected
    
    elif method == "fdr":
        # Benjamini-Hochberg
        sorted_idx = np.argsort(p_values)
        corrected = np.ones(n)
        for rank, idx in enumerate(sorted_idx):
            corrected[idx] = min(p_values[idx] * n / (rank + 1), 1.0)
        return corrected
    
    else:
        raise ValueError(f"Unknown correction method: {method}")


# =============================================================================
# MAIN COMPARISON FUNCTION
# =============================================================================

def compare_multiple_methods(
    y_true: np.ndarray,
    predictions: Dict[str, np.ndarray],
    method_names: Optional[List[str]] = None,
    alpha: float = 0.05,
    correction: str = "holm",
) -> MultiMethodComparisonResult:
    """
    Compare multiple prediction methods comprehensively.
    
    Performs:
    1. Friedman global test for differences
    2. Pairwise DeLong tests with multiple comparison correction
    3. Diagnostic metrics (sensitivity, specificity, Brier score)
    4. Summary table and visualization
    
    Args:
        y_true: True binary labels (0/1)
        predictions: Dict mapping method names to predicted probabilities (N,)
        method_names: List of method names (default: keys of predictions)
        alpha: Significance level (default 0.05)
        correction: Method for correcting p-values ("bonferroni", "holm", "fdr")
        
    Returns:
        MultiMethodComparisonResult object with all comparison results
    """
    if method_names is None:
        method_names = list(predictions.keys())
    
    if len(method_names) < 3:
        raise ValueError("Compare multiple methods requires at least 3 methods")
    
    # Validate inputs
    y_true = np.asarray(y_true, dtype=int)
    for method in method_names:
        if method not in predictions:
            raise ValueError(f"Method {method} not in predictions dict")
        predictions[method] = np.asarray(predictions[method], dtype=float)
    
    # =========================================================================
    # 1. GLOBAL TEST (Friedman)
    # =========================================================================
    pred_array = {m: predictions[m] for m in method_names}
    friedman_stat, friedman_p = friedman_test(y_true, pred_array)
    
    # =========================================================================
    # 2. COMPUTE AUCs AND DIAGNOSTICS
    # =========================================================================
    aucs = {}
    sensitivities = {}
    specificities = {}
    brier_scores = {}
    
    for method in method_names:
        pred = predictions[method]
        auc = roc_auc_score(y_true, pred)
        aucs[method] = auc
        
        # Optimal threshold (Youden index)
        fpr, tpr, thresholds = roc_curve(y_true, pred)
        j_index = tpr - fpr
        optimal_idx = np.argmax(j_index)
        threshold = thresholds[optimal_idx]
        
        # Metrics at optimal threshold
        y_pred = (pred >= threshold).astype(int)
        sensitivities[method] = recall_score(y_true, y_pred, zero_division=0)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        specificities[method] = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # Brier score
        brier_scores[method] = brier_score_loss(y_true, pred)
    
    # =========================================================================
    # 3. PAIRWISE DELONG TESTS
    # =========================================================================
    pairwise_results = []
    p_values_list = []
    
    method_pairs = []
    for i, m1 in enumerate(method_names):
        for m2 in method_names[i+1:]:
            method_pairs.append((m1, m2))
            
            auc1, auc2, z_stat, p_val = delong_test(y_true, predictions[m1], predictions[m2])
            p_values_list.append(p_val)
            
            result = PairwiseComparisonResult(
                method1_name=m1,
                method2_name=m2,
                auc1=auc1,
                auc2=auc2,
                auc_difference=auc1 - auc2,
                z_statistic=z_stat,
                p_value=p_val,
            )
            pairwise_results.append(result)
    
    # Apply multiple comparison corrections
    p_values_array = np.array(p_values_list)
    if correction.lower() == "holm":
        p_values_holm = apply_multiple_comparison_correction(p_values_array, "holm")
        p_values_fdr = apply_multiple_comparison_correction(p_values_array, "fdr")
    elif correction.lower() == "bonferroni":
        p_values_holm = apply_multiple_comparison_correction(p_values_array, "bonferroni")
        p_values_fdr = apply_multiple_comparison_correction(p_values_array, "fdr")
    else:
        p_values_holm = p_values_array
        p_values_fdr = apply_multiple_comparison_correction(p_values_array, "fdr")
    
    # Update pairwise results with corrected p-values
    for idx, result in enumerate(pairwise_results):
        result.p_value_holm = p_values_holm[idx]
        result.p_value_fdr = p_values_fdr[idx]
        result.is_significant = p_values_holm[idx] < alpha
        
        # Effect size: Cohen's h for proportions (AUC difference)
        # Simplified version
        result.effect_size = abs(result.auc_difference)
    
    # Create final result object
    result = MultiMethodComparisonResult(
        method_names=method_names,
        y_true=y_true,
        predictions=predictions,
        global_test_name="Friedman",
        global_test_statistic=friedman_stat,
        global_p_value=friedman_p,
        global_significant=friedman_p < alpha,
        aucs=aucs,
        pairwise_results=pairwise_results,
        correction_method=correction.capitalize(),
        sensitivities=sensitivities,
        specificities=specificities,
        brier_scores=brier_scores,
    )
    
    return result


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_roc_multiple_methods(
    y_true: np.ndarray,
    predictions: Dict[str, np.ndarray],
    method_names: Optional[List[str]] = None,
    title: str = "ROC Curve Comparison: Multiple Methods",
    width: int = 900,
    height: int = 700,
) -> go.Figure:
    """
    Generate overlaid ROC curves for multiple methods.
    
    Args:
        y_true: True labels
        predictions: Dict mapping method names to predicted probabilities
        method_names: List of method names to plot (default: all)
        title: Figure title
        width: Figure width in pixels
        height: Figure height in pixels
        
    Returns:
        Plotly Figure object
    """
    if method_names is None:
        method_names = list(predictions.keys())
    
    # Color palette
    colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
    ]
    
    fig = go.Figure()
    
    # Add ROC curves for each method
    for idx, method in enumerate(method_names):
        pred = np.asarray(predictions[method], dtype=float)
        y_true_array = np.asarray(y_true, dtype=int)
        
        fpr, tpr, thresholds = roc_curve(y_true_array, pred)
        auc = roc_auc_score(y_true_array, pred)
        
        color = colors[idx % len(colors)]
        
        fig.add_trace(go.Scatter(
            x=fpr,
            y=tpr,
            mode='lines',
            name=f'{method} (AUC = {auc:.4f})',
            line=dict(color=color, width=2.5),
            hovertemplate=(
                f'<b>{method}</b><br>'
                'FPR: %{x:.3f}<br>'
                'TPR: %{y:.3f}<br>'
                'Threshold: %{customdata:.3f}<br>'
                '<extra></extra>'
            ),
            customdata=thresholds,
        ))
    
    # Add diagonal reference line
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        name='Random (AUC = 0.5)',
        line=dict(color='gray', width=2, dash='dash'),
        hoverinfo='skip',
    ))
    
    fig.update_layout(
        title=title,
        xaxis=dict(
            title='False Positive Rate (1 - Specificity)',
            range=[0, 1],
            gridcolor='lightgray',
        ),
        yaxis=dict(
            title='True Positive Rate (Sensitivity)',
            range=[0, 1.05],
            gridcolor='lightgray',
        ),
        width=width,
        height=height,
        template='plotly_white',
        legend=dict(
            x=0.6,
            y=0.1,
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='gray',
            borderwidth=1,
        ),
        hovermode='closest',
        font=dict(size=11),
    )
    
    return fig


def plot_auroc_comparison(
    result: MultiMethodComparisonResult,
    width: int = 800,
    height: int = 600,
) -> go.Figure:
    """
    Generate bar chart comparing AUCs across methods.
    
    Args:
        result: MultiMethodComparisonResult object
        width: Figure width
        height: Figure height
        
    Returns:
        Plotly Figure object
    """
    # Sort by AUC descending
    methods = sorted(result.aucs.keys(), key=lambda m: result.aucs[m], reverse=True)
    aucs = [result.aucs[m] for m in methods]
    
    fig = go.Figure(data=[
        go.Bar(
            x=methods,
            y=aucs,
            text=[f'{auc:.4f}' for auc in aucs],
            textposition='auto',
            marker=dict(color=aucs, colorscale='Viridis', showscale=True),
        )
    ])
    
    fig.update_layout(
        title='AUROC Comparison Across Methods',
        xaxis_title='Method',
        yaxis_title='AUROC',
        yaxis=dict(range=[0, 1]),
        width=width,
        height=height,
        template='plotly_white',
    )
    
    return fig


# =============================================================================
# TABLE GENERATION
# =============================================================================

def create_summary_table(result: MultiMethodComparisonResult) -> pd.DataFrame:
    """
    Create comprehensive summary table.
    
    Args:
        result: MultiMethodComparisonResult object
        
    Returns:
        Pandas DataFrame with summary metrics
    """
    data = []
    
    # Sort by AUROC descending
    sorted_methods = sorted(result.aucs.items(), key=lambda x: x[1], reverse=True)
    
    for rank, (method, auc) in enumerate(sorted_methods, 1):
        data.append({
            'Rank': rank,
            'Method': method,
            'AUROC': f"{auc:.4f}",
            'Sensitivity': f"{result.sensitivities.get(method, np.nan):.4f}",
            'Specificity': f"{result.specificities.get(method, np.nan):.4f}",
            'Brier Score': f"{result.brier_scores.get(method, np.nan):.4f}",
        })
    
    return pd.DataFrame(data)


def create_pairwise_table(result: MultiMethodComparisonResult) -> pd.DataFrame:
    """
    Create pairwise comparison table with all tests.
    
    Args:
        result: MultiMethodComparisonResult object
        
    Returns:
        Pandas DataFrame with pairwise results
    """
    rows = []
    
    for pwr in result.pairwise_results:
        rows.append({
            'Method 1': pwr.method1_name,
            'Method 2': pwr.method2_name,
            'AUC_1': f"{pwr.auc1:.4f}",
            'AUC_2': f"{pwr.auc2:.4f}",
            'ΔAUC': f"{pwr.auc_difference:.4f}",
            'Z': f"{pwr.z_statistic:.4f}",
            'p-value': f"{pwr.p_value:.6f}",
            'p-Holm': f"{pwr.p_value_holm:.6f}",
            'p-FDR': f"{pwr.p_value_fdr:.6f}",
            'Significant': 'Yes' if pwr.is_significant else 'No',
            'Effect Size': f"{pwr.effect_size:.4f}",
        })
    
    return pd.DataFrame(rows)
