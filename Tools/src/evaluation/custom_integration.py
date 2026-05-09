"""Custom Models Evaluation Integration.

This module adapts the evaluation infrastructure to support custom models,
ensuring uniform metric calculation across sklearn, xgboost, and custom models.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Any, Tuple
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, brier_score_loss,
    confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score
)

from src.models.custom_base import (
    BaseCustomModel,
    BaseCustomClassifier,
    BaseCustomRegressor,
    CustomModelWrapper
)
from src.training.custom_integration import is_custom_model


def evaluate_custom_classifier(
    model: BaseCustomClassifier | CustomModelWrapper,
    X_test: pd.DataFrame | np.ndarray,
    y_test: pd.Series | np.ndarray,
    threshold: float = 0.5,
    model_name: str = "custom_model"
) -> Dict[str, Any]:
    """
    Evaluate a custom classifier with standard metrics.
    
    Args:
        model: Custom classifier to evaluate.
        X_test: Test features.
        y_test: Test targets.
        threshold: Classification threshold for binary classification.
        model_name: Name of the model (for reporting).
    
    Returns:
        Dictionary with evaluation metrics.
    """
    # Get predictions
    y_pred = model.predict(X_test)
    
    # Get probabilities if available
    try:
        if isinstance(model, CustomModelWrapper):
            y_proba = model.predict_proba(X_test)
        else:
            y_proba = model.predict_proba(X_test)
        
        has_proba = True
        
        # For binary classification, use positive class probabilities
        if y_proba.shape[1] == 2:
            y_proba_pos = y_proba[:, 1]
        else:
            y_proba_pos = None
    
    except (AttributeError, NotImplementedError):
        has_proba = False
        y_proba = None
        y_proba_pos = None
    
    # Convert to numpy if needed
    if isinstance(y_test, pd.Series):
        y_test = y_test.values
    
    # Basic metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
        'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0),
    }
    
    # Probability-based metrics
    if has_proba and y_proba_pos is not None:
        try:
            metrics['auroc'] = roc_auc_score(y_test, y_proba_pos)
        except ValueError as e:
            warnings.warn(f"Could not compute AUROC: {e}")
            metrics['auroc'] = None
        
        try:
            metrics['auprc'] = average_precision_score(y_test, y_proba_pos)
        except ValueError as e:
            warnings.warn(f"Could not compute AUPRC: {e}")
            metrics['auprc'] = None
        
        try:
            metrics['brier'] = brier_score_loss(y_test, y_proba_pos)
        except ValueError as e:
            warnings.warn(f"Could not compute Brier score: {e}")
            metrics['brier'] = None
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    metrics['confusion_matrix'] = cm
    
    # For binary classification, add specificity and sensitivity
    if len(np.unique(y_test)) == 2:
        tn, fp, fn, tp = cm.ravel()
        
        metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        metrics['ppv'] = tp / (tp + fp) if (tp + fp) > 0 else 0.0  # Positive Predictive Value
        metrics['npv'] = tn / (tn + fn) if (tn + fn) > 0 else 0.0  # Negative Predictive Value
    
    # Classification report
    try:
        metrics['classification_report'] = classification_report(y_test, y_pred, output_dict=True)
    except Exception as e:
        warnings.warn(f"Could not generate classification report: {e}")
        metrics['classification_report'] = None
    
    # Store predictions for further analysis
    metrics['y_pred'] = y_pred
    metrics['y_proba'] = y_proba
    metrics['y_true'] = y_test
    
    return metrics


def evaluate_custom_regressor(
    model: BaseCustomRegressor | CustomModelWrapper,
    X_test: pd.DataFrame | np.ndarray,
    y_test: pd.Series | np.ndarray
) -> Dict[str, Any]:
    """
    Evaluate a custom regressor with standard metrics.
    
    Args:
        model: Custom regressor to evaluate.
        X_test: Test features.
        y_test: Test targets.
    
    Returns:
        Dictionary with evaluation metrics.
    """
    # Get predictions
    y_pred = model.predict(X_test)
    
    # Convert to numpy if needed
    if isinstance(y_test, pd.Series):
        y_test = y_test.values
    
    # Calculate metrics
    metrics = {
        'mse': mean_squared_error(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'mae': mean_absolute_error(y_test, y_pred),
        'r2': r2_score(y_test, y_pred),
    }
    
    # Additional metrics
    try:
        from sklearn.metrics import explained_variance_score, median_absolute_error
        
        metrics['explained_variance'] = explained_variance_score(y_test, y_pred)
        metrics['median_absolute_error'] = median_absolute_error(y_test, y_pred)
    except ImportError:
        pass
    
    # Store predictions
    metrics['y_pred'] = y_pred
    metrics['y_true'] = y_test
    
    return metrics


def evaluate_model_universal(
    model: Any,
    X_test: pd.DataFrame | np.ndarray,
    y_test: pd.Series | np.ndarray,
    **kwargs
) -> Dict[str, Any]:
    """
    Universal evaluation function that works with both custom and standard models.
    
    Args:
        model: Model to evaluate (custom or standard).
        X_test: Test features.
        y_test: Test targets.
        **kwargs: Additional arguments (e.g., threshold).
    
    Returns:
        Dictionary with evaluation metrics.
    
    Example:
        ```python
        from src.evaluation.custom_integration import evaluate_model_universal
        
        # Works with any model
        metrics_xgb = evaluate_model_universal(xgb_model, X_test, y_test)
        metrics_custom = evaluate_model_universal(custom_model, X_test, y_test)
        ```
    """
    if is_custom_model(model):
        # Custom model path
        if isinstance(model, (BaseCustomClassifier,)) or (
            isinstance(model, CustomModelWrapper) and 
            isinstance(model.model, BaseCustomClassifier)
        ):
            return evaluate_custom_classifier(model, X_test, y_test, **kwargs)
        
        elif isinstance(model, (BaseCustomRegressor,)) or (
            isinstance(model, CustomModelWrapper) and 
            isinstance(model.model, BaseCustomRegressor)
        ):
            return evaluate_custom_regressor(model, X_test, y_test)
        
        else:
            raise ValueError(f"Unknown custom model type: {type(model)}")
    
    else:
        # Standard sklearn/xgboost model
        # Use existing evaluation functions
        from src.evaluation.metrics import compute_classification_metrics, compute_regression_metrics
        
        # Determine if classifier or regressor
        if hasattr(model, 'predict_proba'):
            # Classifier
            return compute_classification_metrics(model, X_test, y_test, **kwargs)
        else:
            # Try to evaluate as classifier first, fallback to regressor
            try:
                return compute_classification_metrics(model, X_test, y_test, **kwargs)
            except:
                return compute_regression_metrics(model, X_test, y_test)


def batch_evaluate_mixed_models(
    models: Dict[str, Any],
    X_test: pd.DataFrame | np.ndarray,
    y_test: pd.Series | np.ndarray,
    **kwargs
) -> Dict[str, Dict[str, Any]]:
    """
    Evaluate multiple models (mix of custom and standard) on test set.
    
    Args:
        models: Dictionary of {name: model}.
        X_test: Test features.
        y_test: Test targets.
        **kwargs: Additional evaluation arguments.
    
    Returns:
        Dictionary of {name: metrics_dict}.
    
    Example:
        ```python
        models = {
            'xgb': xgb_model,
            'custom_mlp': custom_mlp_model,
            'rf': rf_model,
        }
        
        all_metrics = batch_evaluate_mixed_models(models, X_test, y_test)
        ```
    """
    results = {}
    
    for name, model in models.items():
        try:
            metrics = evaluate_model_universal(model, X_test, y_test, **kwargs)
            results[name] = metrics
        
        except Exception as e:
            warnings.warn(f"Error evaluating {name}: {e}")
            results[name] = {'error': str(e)}
    
    return results


def compare_model_performance(
    results: Dict[str, Dict[str, Any]],
    primary_metric: str = 'auroc',
    sort_descending: bool = True
) -> pd.DataFrame:
    """
    Compare performance of multiple models.
    
    Args:
        results: Dictionary of {model_name: metrics_dict}.
        primary_metric: Metric to use for sorting.
        sort_descending: Whether to sort in descending order.
    
    Returns:
        DataFrame with model comparison.
    """
    comparison_data = []
    
    for name, metrics in results.items():
        if 'error' in metrics:
            continue
        
        row = {'model': name}
        
        # Extract key metrics
        for metric in ['accuracy', 'precision', 'recall', 'f1', 'auroc', 'auprc', 
                      'mse', 'rmse', 'mae', 'r2']:
            if metric in metrics:
                row[metric] = metrics[metric]
        
        comparison_data.append(row)
    
    df = pd.DataFrame(comparison_data)
    
    # Sort by primary metric if available
    if primary_metric in df.columns:
        df = df.sort_values(primary_metric, ascending=not sort_descending)
    
    return df


def create_evaluation_summary(
    results: Dict[str, Dict[str, Any]],
    include_predictions: bool = False
) -> Dict[str, Any]:
    """
    Create a comprehensive evaluation summary.
    
    Args:
        results: Dictionary of {model_name: metrics_dict}.
        include_predictions: Whether to include predictions in summary.
    
    Returns:
        Dictionary with evaluation summary.
    """
    summary = {
        'n_models': len(results),
        'models': list(results.keys()),
        'comparison': compare_model_performance(results).to_dict('records'),
    }
    
    # Find best model by AUROC (if available)
    auroc_scores = {name: m.get('auroc') for name, m in results.items() 
                   if 'auroc' in m and m['auroc'] is not None}
    
    if auroc_scores:
        best_model = max(auroc_scores, key=auroc_scores.get)
        summary['best_model'] = {
            'name': best_model,
            'auroc': auroc_scores[best_model],
            'metrics': {k: v for k, v in results[best_model].items() 
                       if not k.startswith('y_') and k != 'confusion_matrix'}
        }
    
    # Model-specific details
    summary['details'] = {}
    for name, metrics in results.items():
        detail = {k: v for k, v in metrics.items() 
                 if not k.startswith('y_') and k != 'confusion_matrix' and k != 'classification_report'}
        
        if include_predictions:
            detail['predictions'] = {
                'y_true': metrics.get('y_true'),
                'y_pred': metrics.get('y_pred'),
                'y_proba': metrics.get('y_proba'),
            }
        
        summary['details'][name] = detail
    
    return summary
