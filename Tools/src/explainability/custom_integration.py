"""Custom Models Explainability Integration.

This module provides explainability methods (SHAP, permutation importance)
that work uniformly with both custom and standard models.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Any, Tuple
import warnings

import numpy as np
import pandas as pd

from src.models.custom_base import (
    BaseCustomModel,
    BaseCustomClassifier,
    BaseCustomRegressor,
    CustomModelWrapper
)
from src.training.custom_integration import is_custom_model


def compute_shap_for_custom_model(
    model: Any,
    X: pd.DataFrame | np.ndarray,
    feature_names: Optional[List[str]] = None,
    max_samples: int = 100,
    **kwargs
) -> Tuple[Any, Any]:
    """
    Compute SHAP values for custom or standard models.
    
    Args:
        model: Model to explain (custom or standard).
        X: Data to explain.
        feature_names: Feature names.
        max_samples: Maximum samples for SHAP computation.
        **kwargs: Additional arguments for SHAP explainer.
    
    Returns:
        Tuple of (explainer, shap_values).
    
    Example:
        ```python
        from src.explainability.custom_integration import compute_shap_for_custom_model
        
        # Works with any model
        explainer, shap_values = compute_shap_for_custom_model(
            model, X_test, feature_names=feature_cols
        )
        ```
    """
    try:
        import shap
    except ImportError:
        raise ImportError("SHAP not installed. Install with: pip install shap")
    
    # Convert to numpy if needed
    if isinstance(X, pd.DataFrame):
        X_array = X.values
        if feature_names is None:
            feature_names = list(X.columns)
    else:
        X_array = X
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    
    # Limit samples
    if len(X_array) > max_samples:
        indices = np.random.choice(len(X_array), max_samples, replace=False)
        X_sample = X_array[indices]
    else:
        X_sample = X_array
    
    # Create prediction function
    if is_custom_model(model):
        # Custom model
        if isinstance(model, CustomModelWrapper):
            # Unwrap for prediction
            if isinstance(model.model, BaseCustomClassifier):
                def predict_fn(X):
                    return model.predict_proba(X)
            else:
                def predict_fn(X):
                    return model.predict(X)
        else:
            if isinstance(model, BaseCustomClassifier):
                def predict_fn(X):
                    return model.predict_proba(X)
            else:
                def predict_fn(X):
                    return model.predict(X)
    else:
        # Standard model
        if hasattr(model, 'predict_proba'):
            def predict_fn(X):
                return model.predict_proba(X)
        else:
            def predict_fn(X):
                return model.predict(X)
    
    # Create SHAP explainer
    try:
        # Try TreeExplainer first (for tree-based models)
        explainer = shap.TreeExplainer(model, **kwargs)
    except:
        try:
            # Try KernelExplainer as fallback
            explainer = shap.KernelExplainer(predict_fn, X_sample[:100], **kwargs)
        except:
            # Use PermutationExplainer as last resort
            explainer = shap.PermutationExplainer(predict_fn, X_sample[:100], **kwargs)
    
    # Compute SHAP values
    shap_values = explainer(X_sample)
    
    return explainer, shap_values


def compute_permutation_importance_custom(
    model: Any,
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
    n_repeats: int = 10,
    random_state: Optional[int] = None,
    scoring: Optional[str] = None,
    feature_names: Optional[list] = None
) -> pd.DataFrame:
    """
    Compute permutation importance for custom or standard models.
    
    Args:
        model: Model to explain.
        X: Features.
        y: Targets.
        n_repeats: Number of permutation repeats.
        random_state: Random seed.
        scoring: Scoring metric (auto-detected if None).
        feature_names: Optional feature names for DataFrame output.
    
    Returns:
        DataFrame with feature importance.
    """
    from sklearn.inspection import permutation_importance
    
    # Auto-detect scoring if not provided
    if scoring is None:
        if is_custom_model(model):
            if isinstance(model, (BaseCustomClassifier,)) or (
                isinstance(model, CustomModelWrapper) and 
                isinstance(model.model, BaseCustomClassifier)
            ):
                scoring = 'roc_auc'
            else:
                scoring = 'r2'
        else:
            if hasattr(model, 'predict_proba'):
                scoring = 'roc_auc'
            else:
                scoring = 'r2'
    
    # Compute permutation importance
    result = permutation_importance(
        model, X, y,
        n_repeats=n_repeats,
        random_state=random_state,
        scoring=scoring
    )
    
    # Create DataFrame with feature names
    if feature_names is None:
        if isinstance(X, pd.DataFrame):
            feature_names = list(X.columns)
        else:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance_mean': result.importances_mean,
        'importance_std': result.importances_std,
    })
    
    importance_df = importance_df.sort_values('importance_mean', ascending=False)
    
    return importance_df


def get_feature_importance_universal(
    model: Any,
    feature_names: Optional[List[str]] = None
) -> Optional[pd.DataFrame]:
    """
    Get feature importance from any model (if available).
    
    Args:
        model: Model to extract importance from.
        feature_names: Feature names.
    
    Returns:
        DataFrame with feature importance, or None if not available.
    """
    importance = None
    
    # Try to get built-in feature importance
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    elif hasattr(model, 'coef_'):
        # For linear models, use absolute coefficient values
        importance = np.abs(model.coef_[0] if model.coef_.ndim > 1 else model.coef_)
    elif is_custom_model(model):
        # For custom models wrapped
        if isinstance(model, CustomModelWrapper):
            return get_feature_importance_universal(model.model, feature_names)
    
    if importance is None:
        return None
    
    # Create DataFrame
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(len(importance))]
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance,
    })
    
    importance_df = importance_df.sort_values('importance', ascending=False)
    
    return importance_df


def explain_prediction_custom(
    model: Any,
    X_instance: np.ndarray | pd.Series,
    feature_names: Optional[List[str]] = None,
    background_data: Optional[np.ndarray | pd.DataFrame] = None
) -> Dict[str, Any]:
    """
    Explain a single prediction using SHAP.
    
    Args:
        model: Model to explain.
        X_instance: Single instance to explain.
        feature_names: Feature names.
        background_data: Background data for SHAP.
    
    Returns:
        Dictionary with explanation details.
    """
    try:
        import shap
    except ImportError:
        raise ImportError("SHAP not installed. Install with: pip install shap")
    
    # Convert to 2D array if needed
    if isinstance(X_instance, pd.Series):
        X_instance = X_instance.values
    if X_instance.ndim == 1:
        X_instance = X_instance.reshape(1, -1)
    
    # Get feature names
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(X_instance.shape[1])]
    
    # Compute SHAP values
    if background_data is not None:
        explainer, shap_values = compute_shap_for_custom_model(
            model, background_data, feature_names, max_samples=100
        )
    else:
        explainer, shap_values = compute_shap_for_custom_model(
            model, X_instance, feature_names, max_samples=1
        )
    
    # Extract SHAP values for the instance
    if hasattr(shap_values, 'values'):
        instance_shap = shap_values.values[0]
    else:
        instance_shap = shap_values[0]
    
    # Get prediction
    if hasattr(model, 'predict_proba'):
        if isinstance(model, CustomModelWrapper):
            prediction = model.predict_proba(X_instance)[0]
        else:
            prediction = model.predict_proba(X_instance)[0]
    else:
        prediction = model.predict(X_instance)[0]
    
    # Create explanation
    explanation = {
        'prediction': prediction,
        'shap_values': instance_shap,
        'feature_names': feature_names,
        'feature_values': X_instance[0],
        'top_features': []
    }
    
    # Find top contributing features
    abs_shap = np.abs(instance_shap)
    top_indices = np.argsort(abs_shap)[::-1][:10]
    
    for idx in top_indices:
        explanation['top_features'].append({
            'feature': feature_names[idx],
            'value': X_instance[0, idx],
            'shap_value': instance_shap[idx],
            'impact': 'positive' if instance_shap[idx] > 0 else 'negative'
        })
    
    return explanation


def batch_explain_models(
    models: Dict[str, Any],
    X: pd.DataFrame | np.ndarray,
    feature_names: Optional[List[str]] = None,
    max_samples: int = 100,
    compute_permutation: bool = True,
    y: Optional[pd.Series | np.ndarray] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Compute explanations for multiple models.
    
    Args:
        models: Dictionary of {name: model}.
        X: Data to explain.
        feature_names: Feature names.
        max_samples: Max samples for SHAP.
        compute_permutation: Whether to compute permutation importance.
        y: Targets (required if compute_permutation=True).
    
    Returns:
        Dictionary of {name: explanations_dict}.
    """
    results = {}
    
    for name, model in models.items():
        try:
            explanations = {}
            
            # Get built-in feature importance
            fi = get_feature_importance_universal(model, feature_names)
            if fi is not None:
                explanations['feature_importance'] = fi
            
            # Compute SHAP
            try:
                explainer, shap_values = compute_shap_for_custom_model(
                    model, X, feature_names, max_samples
                )
                explanations['shap_explainer'] = explainer
                explanations['shap_values'] = shap_values
            except Exception as e:
                warnings.warn(f"Could not compute SHAP for {name}: {e}")
            
            # Compute permutation importance
            if compute_permutation and y is not None:
                try:
                    perm_imp = compute_permutation_importance_custom(model, X, y)
                    explanations['permutation_importance'] = perm_imp
                except Exception as e:
                    warnings.warn(f"Could not compute permutation importance for {name}: {e}")
            
            results[name] = explanations
        
        except Exception as e:
            warnings.warn(f"Error explaining {name}: {e}")
            results[name] = {'error': str(e)}
    
    return results
