"""Rigorous statistical comparison between ML models and TIMI scores.

Supports both TIMI-STEMI (Morrow 2000) and TIMI-NSTEMI (Antman 2000) variants.
Provides comprehensive comparison tools including:
- AUC comparison with DeLong test
- Net Reclassification Improvement (NRI)
- Integrated Discrimination Improvement (IDI)
- Calibration comparison
- Statistical tests and visualizations
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    brier_score_loss,
    confusion_matrix,
)
from sklearn.calibration import calibration_curve


@dataclass
class TIMIComparisonResult:
    """Results from comparing ML model with TIMI score (STEMI or NSTEMI)."""
    
    # Model and scale names
    model_name: str
    timi_variant: str  # "STEMI" or "NSTEMI"
    
    # AUC values
    model_auc: float
    timi_auc: float
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
    timi_brier: float
    brier_difference: float
    
    # Additional metrics
    model_accuracy: float
    timi_accuracy: float
    model_sensitivity: float
    timi_sensitivity: float
    model_specificity: float
    timi_specificity: float
    
    # Statistical conclusion
    is_model_superior: bool
    superiority_level: str  # "significant", "marginal", "none", "inferior"
    
    # Fields with defaults come last
    timi_name: str = "TIMI Score"
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'model_name': self.model_name,
            'timi_variant': self.timi_variant,
            'timi_name': self.timi_name,
            'model_auc': self.model_auc,
            'timi_auc': self.timi_auc,
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
            'timi_brier': self.timi_brier,
            'brier_difference': self.brier_difference,
            'model_accuracy': self.model_accuracy,
            'timi_accuracy': self.timi_accuracy,
            'model_sensitivity': self.model_sensitivity,
            'timi_sensitivity': self.timi_sensitivity,
            'model_specificity': self.model_specificity,
            'timi_specificity': self.timi_specificity,
            'is_model_superior': self.is_model_superior,
            'superiority_level': self.superiority_level,
            'p_value': self.auc_p_value,
        }
    
    @property
    def p_value(self) -> float:
        """Backward-compatible alias for AUC p-value."""
        return float(self.auc_p_value)


def delong_test(y_true: np.ndarray, pred1: np.ndarray, pred2: np.ndarray) -> Tuple[float, float]:
    """DeLong test for comparing two correlated ROC curves.
    
    Args:
        y_true: True labels
        pred1: Predictions from model 1
        pred2: Predictions from model 2
        
    Returns:
        Tuple of (z_statistic, p_value)
    """
    # CRITICAL: Validate paired samples assumption for DeLong test
    assert len(pred1) == len(pred2), (
        f"DeLong test requires paired samples (same n). "
        f"pred1 has length {len(pred1)}, pred2 has length {len(pred2)}. "
        f"Lengths must match for valid ROC comparison."
    )
    
    auc1 = roc_auc_score(y_true, pred1)
    auc2 = roc_auc_score(y_true, pred2)
    
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]
    
    n_pos = len(pos_idx)
    n_neg = len(neg_idx)
    
    def structural_components(y, pred, pos_idx, neg_idx):
        v10 = np.zeros(len(pos_idx))
        for i, pos_i in enumerate(pos_idx):
            v10[i] = np.mean(pred[pos_i] > pred[neg_idx]) + 0.5 * np.mean(pred[pos_i] == pred[neg_idx])
        return v10
    
    v10_1 = structural_components(y_true, pred1, pos_idx, neg_idx)
    v10_2 = structural_components(y_true, pred2, pos_idx, neg_idx)
    
    v01_1 = 1 - structural_components(y_true, -pred1, neg_idx, pos_idx)
    v01_2 = 1 - structural_components(y_true, -pred2, neg_idx, pos_idx)
    
    s10_1 = np.var(v10_1, ddof=1) / n_pos
    s10_2 = np.var(v10_2, ddof=1) / n_pos
    s01_1 = np.var(v01_1, ddof=1) / n_neg
    s01_2 = np.var(v01_2, ddof=1) / n_neg
    
    cov10 = np.cov(v10_1, v10_2, ddof=1)[0, 1] / n_pos if n_pos > 1 else 0
    cov01 = np.cov(v01_1, v01_2, ddof=1)[0, 1] / n_neg if n_neg > 1 else 0
    
    se_auc_diff = np.sqrt(s10_1 + s10_2 - 2 * cov10 + s01_1 + s01_2 - 2 * cov01)
    
    z_stat = (auc1 - auc2) / (se_auc_diff + 1e-10)
    p_value = 2 * (1 - stats.norm.cdf(np.abs(z_stat)))
    
    return z_stat, p_value


def compute_nri(
    y_true: np.ndarray,
    pred_new: np.ndarray,
    pred_old: np.ndarray,
    threshold: float = 0.5,
) -> Tuple[float, float, float, float]:
    """Compute Net Reclassification Improvement.
    
    Args:
        y_true: True labels
        pred_new: Predictions from new model
        pred_old: Predictions from old/comparator model
        threshold: Classification threshold
        
    Returns:
        Tuple of (nri_total, nri_events, nri_nonevents, p_value)
    """
    # Reclassify events
    event_idx = y_true == 1
    pred_new_events = pred_new[event_idx] > threshold
    pred_old_events = pred_old[event_idx] > threshold
    
    # Proportion moving up in high-risk for events
    reclassified_events = (pred_new_events.astype(int) - pred_old_events.astype(int)) > 0
    nri_events = np.mean(reclassified_events) - np.mean((pred_old_events.astype(int) - pred_new_events.astype(int)) > 0)
    
    # Reclassify non-events
    nonevent_idx = y_true == 0
    pred_new_nonevents = pred_new[nonevent_idx] > threshold
    pred_old_nonevents = pred_old[nonevent_idx] > threshold
    
    # Proportion moving down in risk for non-events
    reclassified_nonevents = (pred_new_nonevents.astype(int) - pred_old_nonevents.astype(int)) < 0
    nri_nonevents = np.mean(reclassified_nonevents) - np.mean((pred_old_nonevents.astype(int) - pred_new_nonevents.astype(int)) < 0)
    
    nri_total = nri_events + nri_nonevents
    
    # Approximate p-value (Z-test)
    se = np.sqrt(
        np.var(reclassified_events) / np.sum(event_idx) +
        np.var(reclassified_nonevents) / np.sum(nonevent_idx)
    )
    z_stat = nri_total / (se + 1e-10)
    p_value = 2 * (1 - stats.norm.cdf(np.abs(z_stat)))
    
    return nri_total, nri_events, nri_nonevents, p_value


def compute_idi(
    y_true: np.ndarray,
    pred_new: np.ndarray,
    pred_old: np.ndarray,
) -> Tuple[float, float]:
    """Compute Integrated Discrimination Improvement.
    
    Args:
        y_true: True labels
        pred_new: Predictions from new model
        pred_old: Predictions from old/comparator model
        
    Returns:
        Tuple of (idi, p_value)
    """
    event_idx = y_true == 1
    nonevent_idx = y_true == 0
    
    idi = (np.mean(pred_new[event_idx]) - np.mean(pred_old[event_idx])) - \
          (np.mean(pred_new[nonevent_idx]) - np.mean(pred_old[nonevent_idx]))
    
    # Approximate p-value
    se = np.sqrt(
        np.var(pred_new[event_idx] - pred_old[event_idx]) / np.sum(event_idx) +
        np.var(pred_new[nonevent_idx] - pred_old[nonevent_idx]) / np.sum(nonevent_idx)
    )
    z_stat = idi / (se + 1e-10)
    p_value = 2 * (1 - stats.norm.cdf(np.abs(z_stat)))
    
    return idi, p_value


def compute_diagnostics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute diagnostic metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted probabilities
        threshold: Classification threshold
        
    Returns:
        Dictionary with accuracy, sensitivity, specificity, etc.
    """
    y_pred_binary = (y_pred > threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()
    
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    return {
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def compare_with_timi(
    y_true: np.ndarray,
    y_pred_model: np.ndarray,
    y_pred_timi: np.ndarray,
    model_name: str = "ML Model",
    timi_variant: str = "STEMI",
    threshold: float = 0.5,
    alpha: float = 0.05,
) -> TIMIComparisonResult:
    """Comprehensive comparison of ML model with TIMI score.
    
    Args:
        y_true: True labels
        y_pred_model: Predictions from ML model (probabilities)
        y_pred_timi: TIMI predictions (probabilities or risk scores normalized)
        model_name: Name of the ML model
        timi_variant: "STEMI" or "NSTEMI"
        threshold: Classification threshold
        alpha: Significance level for statistical tests
        
    Returns:
        TIMIComparisonResult object with all metrics and test results
    """
    # AUC comparison with DeLong test
    auc_model = roc_auc_score(y_true, y_pred_model)
    auc_timi = roc_auc_score(y_true, y_pred_timi)
    auc_diff = auc_model - auc_timi
    
    z_stat, p_value_auc = delong_test(y_true, y_pred_model, y_pred_timi)
    
    # Confidence interval for AUC difference
    se_diff = abs(auc_diff / z_stat) if z_stat != 0 else np.sqrt(0.0001)
    ci_lower = auc_diff - 1.96 * se_diff
    ci_upper = auc_diff + 1.96 * se_diff
    
    # NRI
    nri_total, nri_events, nri_nonevents, p_value_nri = compute_nri(
        y_true, y_pred_model, y_pred_timi, threshold
    )
    
    # IDI
    idi, p_value_idi = compute_idi(y_true, y_pred_model, y_pred_timi)
    
    # Diagnostics
    model_diags = compute_diagnostics(y_true, y_pred_model, threshold)
    timi_diags = compute_diagnostics(y_true, y_pred_timi, threshold)
    
    # Calibration (Brier score)
    model_brier = brier_score_loss(y_true, y_pred_model)
    timi_brier = brier_score_loss(y_true, y_pred_timi)
    
    # Determine superiority
    is_model_superior = auc_model > auc_timi and p_value_auc < alpha
    
    if is_model_superior:
        if p_value_auc < alpha / 10:
            superiority_level = "significant"
        else:
            superiority_level = "marginal"
    elif auc_model < auc_timi and p_value_auc < alpha:
        superiority_level = "inferior"
    else:
        superiority_level = "none"
    
    return TIMIComparisonResult(
        model_name=model_name,
        timi_variant=timi_variant,
        model_auc=auc_model,
        timi_auc=auc_timi,
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
        model_brier=model_brier,
        timi_brier=timi_brier,
        brier_difference=model_brier - timi_brier,
        model_accuracy=model_diags["accuracy"],
        timi_accuracy=timi_diags["accuracy"],
        model_sensitivity=model_diags["sensitivity"],
        timi_sensitivity=timi_diags["sensitivity"],
        model_specificity=model_diags["specificity"],
        timi_specificity=timi_diags["specificity"],
        is_model_superior=is_model_superior,
        superiority_level=superiority_level,
        timi_name=f"TIMI-{timi_variant}",
    )
