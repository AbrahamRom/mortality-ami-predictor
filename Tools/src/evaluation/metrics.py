"""Metrics computation for classification and regression."""
from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    brier_score_loss,
    confusion_matrix,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute comprehensive classification metrics.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        threshold: Classification threshold
        
    Returns:
        Dictionary of metric_name -> value
    """
    y_pred = (y_prob >= threshold).astype(int)
    
    metrics = {
        "auroc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else np.nan,
        "auprc": float(average_precision_score(y_true, y_prob)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "specificity": float(_compute_specificity(y_true, y_pred)),
        "npv": float(_compute_npv(y_true, y_pred)),
    }
    
    return metrics


def _compute_specificity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute specificity (true negative rate)."""
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        return tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return 0.0


def _compute_npv(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute negative predictive value."""
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        return tn / (tn + fn) if (tn + fn) > 0 else 0.0
    return 0.0


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """Compute regression metrics.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Dictionary of metric_name -> value
    """
    return {
        "mse": float(mean_squared_error(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred, squared=False)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }
