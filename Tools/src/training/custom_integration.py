"""Custom Models Training Integration.

This module extends the training pipeline to support custom models
alongside sklearn/xgboost models, with metadata tracking and
seamless integration with existing infrastructure.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.base import clone

from src.models.custom_base import (
    BaseCustomModel,
    BaseCustomClassifier,
    BaseCustomRegressor,
    CustomModelWrapper
)
from src.models.metadata import ModelMetadata, create_metadata_from_training
from src.preprocessing import PreprocessingConfig, build_preprocessing_pipeline
from src.config import RANDOM_SEED


def is_custom_model(model: Any) -> bool:
    """
    Check if a model is a custom model.
    
    Args:
        model: Model instance to check.
    
    Returns:
        True if model is custom, False otherwise.
    """
    return isinstance(model, (BaseCustomModel, CustomModelWrapper))


def prepare_custom_model_for_cv(
    model: BaseCustomModel,
    preprocessing_config: Optional[PreprocessingConfig] = None
) -> CustomModelWrapper:
    """
    Prepare a custom model for cross-validation by wrapping it with preprocessing.
    
    Args:
        model: Custom model instance.
        preprocessing_config: Preprocessing configuration.
    
    Returns:
        CustomModelWrapper with preprocessing pipeline.
    """
    if isinstance(model, CustomModelWrapper):
        return model
    
    # Build preprocessing pipeline if config provided
    preprocessing = None
    if preprocessing_config:
        preprocessing = build_preprocessing_pipeline(preprocessing_config)
    
    # Wrap the custom model
    wrapper = CustomModelWrapper(
        model=model,
        preprocessing=preprocessing,
        hyperparameters=model.get_params()
    )
    
    return wrapper


def train_custom_model(
    model: BaseCustomModel,
    X_train: pd.DataFrame | np.ndarray,
    y_train: pd.Series | np.ndarray,
    X_val: Optional[pd.DataFrame | np.ndarray] = None,
    y_val: Optional[pd.Series | np.ndarray] = None,
    preprocessing_config: Optional[PreprocessingConfig] = None,
    **fit_kwargs
) -> Tuple[CustomModelWrapper, Dict[str, Any]]:
    """
    Train a custom model with optional validation.
    
    Args:
        model: Custom model to train.
        X_train: Training features.
        y_train: Training targets.
        X_val: Optional validation features.
        y_val: Optional validation targets.
        preprocessing_config: Preprocessing configuration.
        **fit_kwargs: Additional arguments for model.fit().
    
    Returns:
        Tuple of (trained wrapper, training info dict).
    """
    # Prepare model with preprocessing
    wrapper = prepare_custom_model_for_cv(model, preprocessing_config)
    
    # Train
    wrapper.fit(X_train, y_train, **fit_kwargs)
    
    # Collect training info
    training_info = {
        'model_name': model.name,
        'model_class': model.__class__.__name__,
        'n_samples_train': len(X_train),
        'hyperparameters': model.get_params(),
    }
    
    # Validation metrics if provided
    if X_val is not None and y_val is not None:
        y_pred = wrapper.predict(X_val)
        
        # Compute metrics based on model type
        if isinstance(model, BaseCustomClassifier):
            from sklearn.metrics import accuracy_score, roc_auc_score
            
            training_info['val_accuracy'] = accuracy_score(y_val, y_pred)
            
            if hasattr(wrapper, 'predict_proba'):
                y_proba = wrapper.predict_proba(X_val)[:, 1]
                training_info['val_roc_auc'] = roc_auc_score(y_val, y_proba)
        else:
            from sklearn.metrics import mean_squared_error, r2_score
            
            training_info['val_mse'] = mean_squared_error(y_val, y_pred)
            training_info['val_r2'] = r2_score(y_val, y_pred)
    
    return wrapper, training_info


def cross_validate_custom_model(
    model: BaseCustomModel,
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
    cv: int = 5,
    scoring: str = 'roc_auc',
    preprocessing_config: Optional[PreprocessingConfig] = None,
    n_jobs: int = -1,
    return_train_score: bool = True
) -> Dict[str, Any]:
    """
    Perform cross-validation on a custom model.
    
    Args:
        model: Custom model to validate.
        X: Features.
        y: Targets.
        cv: Number of folds.
        scoring: Scoring metric.
        preprocessing_config: Preprocessing configuration.
        n_jobs: Number of parallel jobs.
        return_train_score: Whether to return train scores.
    
    Returns:
        Dictionary with CV results.
    """
    # Prepare model
    wrapper = prepare_custom_model_for_cv(model, preprocessing_config)
    
    # Create CV splitter
    cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_SEED)
    
    # Perform cross-validation
    cv_results = cross_validate(
        wrapper,
        X, y,
        cv=cv_splitter,
        scoring=scoring,
        n_jobs=n_jobs,
        return_train_score=return_train_score,
        error_score='raise'
    )
    
    # Summarize results
    results = {
        'test_scores': cv_results['test_score'],
        'mean_test_score': np.mean(cv_results['test_score']),
        'std_test_score': np.std(cv_results['test_score']),
    }
    
    if return_train_score:
        results['train_scores'] = cv_results['train_score']
        results['mean_train_score'] = np.mean(cv_results['train_score'])
        results['std_train_score'] = np.std(cv_results['train_score'])
    
    return results


def create_custom_model_metadata(
    model: BaseCustomModel,
    training_info: Dict[str, Any],
    X_train: pd.DataFrame | np.ndarray,
    y_train: pd.Series | np.ndarray,
    cv_results: Optional[Dict[str, Any]] = None
) -> ModelMetadata:
    """
    Create metadata for a trained custom model.
    
    Args:
        model: Trained custom model.
        training_info: Training information dictionary.
        X_train: Training features.
        y_train: Training targets.
        cv_results: Optional CV results.
    
    Returns:
        ModelMetadata instance.
    """
    from src.models.metadata import DatasetMetadata, TrainingMetadata, PerformanceMetrics
    from datetime import datetime
    
    # Extract feature info
    if isinstance(X_train, pd.DataFrame):
        feature_names = list(X_train.columns)
        n_features = len(feature_names)
    else:
        n_features = X_train.shape[1]
        feature_names = [f"feature_{i}" for i in range(n_features)]
    
    # Create dataset metadata
    dataset_meta = DatasetMetadata(
        n_samples=len(X_train),
        n_features=n_features,
        feature_names=feature_names,
        target_name='target',
        class_distribution=pd.Series(y_train).value_counts().to_dict() if isinstance(model, BaseCustomClassifier) else None
    )
    
    # Create training metadata
    training_meta = TrainingMetadata(
        algorithm=model.__class__.__name__,
        hyperparameters=training_info.get('hyperparameters', {}),
        training_time=training_info.get('training_time', 0.0),
        cv_folds=cv_results.get('n_splits', 0) if cv_results else 0,
        cv_repeats=1,
        random_state=RANDOM_SEED
    )
    
    # Create performance metrics
    if cv_results:
        perf_metrics = PerformanceMetrics(
            cv_scores=cv_results.get('test_scores', []).tolist(),
            mean_cv_score=cv_results.get('mean_test_score', 0.0),
            std_cv_score=cv_results.get('std_test_score', 0.0)
        )
    else:
        perf_metrics = PerformanceMetrics(
            cv_scores=[],
            mean_cv_score=0.0,
            std_cv_score=0.0
        )
    
    # Create full metadata
    metadata = ModelMetadata(
        model_name=model.name,
        model_type=model.__class__.__name__,
        task='classification' if isinstance(model, BaseCustomClassifier) else 'regression',
        framework='custom',
        created_at=datetime.now(),
        dataset=dataset_meta,
        training=training_meta,
        performance=perf_metrics
    )
    
    return metadata


def save_custom_model_with_metadata(
    model: BaseCustomModel | CustomModelWrapper,
    metadata: ModelMetadata,
    output_path: str | Path,
    overwrite: bool = False
) -> Path:
    """
    Save custom model with metadata.
    
    Args:
        model: Model to save.
        metadata: Model metadata.
        output_path: Path to save model.
        overwrite: Whether to overwrite existing files.
    
    Returns:
        Path to saved model.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Check if file exists
    if output_path.exists() and not overwrite:
        # Add timestamp to avoid overwriting
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem = output_path.stem
        output_path = output_path.parent / f"{stem}_{timestamp}{output_path.suffix}"
    
    # Save model
    if isinstance(model, CustomModelWrapper):
        model.save(output_path)
    else:
        model.save(output_path)
    
    # Save metadata
    metadata_path = output_path.parent / f"{output_path.stem}_metadata.json"
    metadata.save(metadata_path)
    
    return output_path


def integrate_custom_models_in_pipeline(
    models_dict: Dict[str, Tuple[Any, Dict]],
    preprocessing_config: Optional[PreprocessingConfig] = None
) -> Dict[str, Tuple[Any, Dict]]:
    """
    Integrate custom models into standard training pipeline.
    
    This function wraps custom models so they can be used alongside
    sklearn/xgboost models in the existing training infrastructure.
    
    Args:
        models_dict: Dictionary of {name: (model, param_grid)}.
        preprocessing_config: Preprocessing configuration.
    
    Returns:
        Updated models dictionary with wrapped custom models.
    """
    updated_dict = {}
    
    for name, (model, param_grid) in models_dict.items():
        if is_custom_model(model):
            # Wrap custom model
            wrapper = prepare_custom_model_for_cv(model, preprocessing_config)
            updated_dict[name] = (wrapper, param_grid)
        else:
            # Keep standard models as-is
            updated_dict[name] = (model, param_grid)
    
    return updated_dict


def train_mixed_models_with_cv(
    X: pd.DataFrame,
    y: pd.Series,
    models: Dict[str, Tuple[Any, Dict]],
    preprocessing_config: Optional[PreprocessingConfig] = None,
    cv: int = 5,
    scoring: str = 'roc_auc',
    n_jobs: int = -1,
    progress_callback: Optional[Callable[[str, float], None]] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Train both custom and standard models with cross-validation.
    
    This is the main entry point for training a mix of custom and
    standard models in a unified pipeline.
    
    Args:
        X: Training features.
        y: Training targets.
        models: Dictionary of {name: (model, param_grid)}.
        preprocessing_config: Preprocessing configuration.
        cv: Number of CV folds.
        scoring: Scoring metric.
        n_jobs: Number of parallel jobs.
        progress_callback: Optional progress callback.
    
    Returns:
        Dictionary with results for each model.
    
    Example:
        ```python
        from src.models.custom_base import SimpleMLPClassifier
        from src.training.custom_integration import train_mixed_models_with_cv
        
        # Define models (mix of sklearn and custom)
        models = {
            'xgb': (XGBClassifier(), {'max_depth': [3, 5]}),
            'custom_mlp': (SimpleMLPClassifier(), {'hidden_layer_sizes': [(100,), (50, 50)]}),
        }
        
        # Train
        results = train_mixed_models_with_cv(X_train, y_train, models)
        ```
    """
    results = {}
    n_models = len(models)
    
    for idx, (name, (model, param_grid)) in enumerate(models.items(), 1):
        if progress_callback:
            progress_callback(f"Training {name} ({idx}/{n_models})...", idx / n_models)
        
        try:
            if is_custom_model(model):
                # Custom model path
                cv_results = cross_validate_custom_model(
                    model, X, y,
                    cv=cv,
                    scoring=scoring,
                    preprocessing_config=preprocessing_config,
                    n_jobs=1  # Custom models may not support parallel
                )
                
                # Train full model
                wrapper, training_info = train_custom_model(
                    model, X, y,
                    preprocessing_config=preprocessing_config
                )
                
                # Create metadata
                metadata = create_custom_model_metadata(
                    model, training_info, X, y, cv_results
                )
                
                results[name] = {
                    'model': wrapper,
                    'cv_results': cv_results,
                    'metadata': metadata,
                    'is_custom': True
                }
            
            else:
                # Standard sklearn/xgboost model path
                # Use existing cross_validate
                from sklearn.model_selection import cross_validate
                
                cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_SEED)
                
                cv_results = cross_validate(
                    model, X, y,
                    cv=cv_splitter,
                    scoring=scoring,
                    n_jobs=n_jobs,
                    return_train_score=True
                )
                
                # Train full model
                model_clone = clone(model)
                model_clone.fit(X, y)
                
                results[name] = {
                    'model': model_clone,
                    'cv_results': {
                        'test_scores': cv_results['test_score'],
                        'mean_test_score': np.mean(cv_results['test_score']),
                        'std_test_score': np.std(cv_results['test_score']),
                    },
                    'is_custom': False
                }
        
        except Exception as e:
            warnings.warn(f"Error training {name}: {e}")
            results[name] = {
                'error': str(e),
                'is_custom': is_custom_model(model)
            }
    
    return results
