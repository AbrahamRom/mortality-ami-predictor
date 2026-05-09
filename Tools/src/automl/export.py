"""AutoML Export Module.

Provides functions to export AutoML results to standalone models
that integrate with the existing persistence infrastructure.
"""
from __future__ import annotations

import json
import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from ..config import CONFIG
from ..models.metadata import ModelMetadata, PerformanceMetrics, TrainingMetadata, DatasetMetadata

logger = logging.getLogger(__name__)


def export_best_model(
    automl_model: Any,
    output_dir: Union[str, Path],
    model_name: str = "automl_best",
    include_metadata: bool = True,
    training_data: Optional[pd.DataFrame] = None,
    target_column: Optional[str] = None,
) -> Path:
    """
    Export the best model from AutoML to a standalone file.
    
    This extracts the best single model from the AutoML ensemble
    and saves it in a format compatible with the existing pipeline.
    
    Args:
        automl_model: Fitted AutoML model (AutoMLClassifier or FLAMLClassifier)
        output_dir: Directory to save the model
        model_name: Name for the saved model
        include_metadata: Whether to save metadata JSON
        training_data: Optional training data for metadata
        target_column: Target column name for metadata
        
    Returns:
        Path to saved model file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract best model
    if hasattr(automl_model, 'get_best_model'):
        best_model = automl_model.get_best_model()
    elif hasattr(automl_model, 'best_model_'):
        best_model = automl_model.best_model_
    elif hasattr(automl_model, 'model'):
        best_model = automl_model.model
    else:
        raise ValueError("Could not extract best model from AutoML object")
    
    if best_model is None:
        raise ValueError("No best model available. Ensure AutoML has been fitted.")
    
    # Save model
    model_path = output_dir / f"{model_name}.joblib"
    
    import joblib
    joblib.dump(best_model, model_path)
    
    logger.info(f"Best model saved to: {model_path}")
    
    # Save metadata
    if include_metadata:
        metadata = _build_export_metadata(
            automl_model=automl_model,
            model_path=model_path,
            training_data=training_data,
            target_column=target_column,
        )
        
        metadata_path = output_dir / f"{model_name}_metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"Metadata saved to: {metadata_path}")
    
    return model_path


def export_ensemble(
    automl_model: Any,
    output_dir: Union[str, Path],
    model_name: str = "automl_ensemble",
    max_models: int = 10,
    include_weights: bool = True,
) -> Tuple[Path, List[Path]]:
    """
    Export the full ensemble from AutoML.
    
    This exports all models in the ensemble along with their weights,
    allowing reconstruction of the full ensemble predictions.
    
    Args:
        automl_model: Fitted AutoML model
        output_dir: Directory to save models
        model_name: Base name for saved models
        max_models: Maximum number of models to export
        include_weights: Whether to save model weights
        
    Returns:
        Tuple of (ensemble_info_path, list of model paths)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    import joblib
    
    model_paths = []
    ensemble_info = {
        'model_name': model_name,
        'exported_at': datetime.now().isoformat(),
        'models': [],
    }
    
    # Get ensemble models with weights
    if hasattr(automl_model, 'get_ensemble_weights'):
        ensemble = automl_model.get_ensemble_weights()
    elif hasattr(automl_model, 'automl_') and hasattr(automl_model.automl_, 'get_models_with_weights'):
        ensemble = automl_model.automl_.get_models_with_weights()
    else:
        # Fallback: just export best model
        logger.warning("Could not get ensemble, exporting best model only")
        path = export_best_model(automl_model, output_dir, model_name)
        return path, [path]
    
    # Export each model in ensemble
    for i, (weight, model) in enumerate(ensemble[:max_models]):
        model_file = output_dir / f"{model_name}_model_{i}.joblib"
        joblib.dump(model, model_file)
        model_paths.append(model_file)
        
        ensemble_info['models'].append({
            'index': i,
            'weight': weight,
            'path': str(model_file),
            'type': type(model).__name__,
        })
    
    # Save ensemble info
    ensemble_path = output_dir / f"{model_name}_ensemble.json"
    with open(ensemble_path, 'w', encoding='utf-8') as f:
        json.dump(ensemble_info, f, indent=2)
    
    logger.info(f"Ensemble with {len(model_paths)} models saved to: {output_dir}")
    
    return ensemble_path, model_paths


def convert_to_standalone(
    automl_model: Any,
    feature_names: List[str],
    classes: Optional[np.ndarray] = None,
) -> Any:
    """
    Convert AutoML model to a standalone sklearn-compatible model.
    
    This wraps the AutoML model in a simple wrapper class that
    provides only the essential sklearn interface.
    
    Args:
        automl_model: Fitted AutoML model
        feature_names: List of feature names
        classes: Array of class labels (for classifiers)
        
    Returns:
        Standalone model object
    """
    from sklearn.base import BaseEstimator, ClassifierMixin
    
    class StandaloneAutoMLModel(BaseEstimator, ClassifierMixin):
        """Standalone wrapper for AutoML model."""
        
        def __init__(self, automl_model, feature_names, classes=None):
            self.automl_model = automl_model
            self.feature_names_in_ = np.array(feature_names)
            self.n_features_in_ = len(feature_names)
            self.classes_ = classes if classes is not None else np.array([0, 1])
            
            # Store metadata
            if hasattr(automl_model, 'get_metadata'):
                self._automl_metadata = automl_model.get_metadata()
            else:
                self._automl_metadata = {}
        
        def fit(self, X, y):
            """This model is already fitted."""
            return self
        
        def predict(self, X):
            """Make predictions."""
            if hasattr(self.automl_model, 'predict'):
                return self.automl_model.predict(X)
            raise NotImplementedError()
        
        def predict_proba(self, X):
            """Predict probabilities."""
            if hasattr(self.automl_model, 'predict_proba'):
                return self.automl_model.predict_proba(X)
            raise NotImplementedError()
        
        def get_params(self, deep=True):
            """Get parameters."""
            return {
                'automl_model': self.automl_model,
                'feature_names': list(self.feature_names_in_),
                'classes': self.classes_,
            }
    
    return StandaloneAutoMLModel(
        automl_model=automl_model,
        feature_names=feature_names,
        classes=classes,
    )


def load_ensemble(
    ensemble_path: Union[str, Path],
) -> Tuple[Dict[str, Any], List[Tuple[float, Any]]]:
    """
    Load an exported ensemble.
    
    Args:
        ensemble_path: Path to ensemble JSON file
        
    Returns:
        Tuple of (ensemble_info, list of (weight, model) tuples)
    """
    import joblib
    
    ensemble_path = Path(ensemble_path)
    
    with open(ensemble_path, 'r', encoding='utf-8') as f:
        ensemble_info = json.load(f)
    
    models_with_weights = []
    for model_entry in ensemble_info['models']:
        model = joblib.load(model_entry['path'])
        weight = model_entry['weight']
        models_with_weights.append((weight, model))
    
    return ensemble_info, models_with_weights


def _build_export_metadata(
    automl_model: Any,
    model_path: Path,
    training_data: Optional[pd.DataFrame] = None,
    target_column: Optional[str] = None,
) -> Dict[str, Any]:
    """Build metadata dictionary for exported model."""
    
    metadata = {
        'model_name': model_path.stem,
        'model_type': 'automl_export',
        'exported_at': datetime.now().isoformat(),
        'model_file_path': str(model_path),
        'source': 'automl',
    }
    
    # Add AutoML-specific info
    if hasattr(automl_model, 'get_metadata'):
        automl_meta = automl_model.get_metadata()
        metadata['automl'] = automl_meta
    
    if hasattr(automl_model, 'backend_used_'):
        metadata['backend'] = automl_model.backend_used_
    
    if hasattr(automl_model, 'fit_time_'):
        metadata['fit_time_seconds'] = automl_model.fit_time_
    
    if hasattr(automl_model, 'config'):
        config = automl_model.config
        metadata['config'] = {
            'preset': config.preset.value if hasattr(config.preset, 'value') else str(config.preset),
            'time_budget': config.time_left_for_this_task,
            'ensemble_size': config.ensemble_size,
            'metric': config.metric,
        }
    
    # Add leaderboard summary
    if hasattr(automl_model, 'get_leaderboard'):
        leaderboard = automl_model.get_leaderboard()
        if leaderboard is not None and len(leaderboard) > 0:
            metadata['leaderboard_summary'] = {
                'n_models_evaluated': len(leaderboard),
                'top_models': leaderboard.head(5).to_dict('records') if len(leaderboard) > 0 else [],
            }
    
    # Add training data info
    if training_data is not None:
        feature_cols = [c for c in training_data.columns if c != target_column]
        metadata['dataset'] = {
            'n_samples': len(training_data),
            'n_features': len(feature_cols),
            'feature_names': feature_cols,
            'target_column': target_column,
        }
        
        if target_column and target_column in training_data.columns:
            class_dist = training_data[target_column].value_counts().to_dict()
            metadata['dataset']['class_distribution'] = {str(k): v for k, v in class_dist.items()}
    
    return metadata


def create_automl_report(
    automl_model: Any,
    output_path: Optional[Union[str, Path]] = None,
) -> str:
    """
    Create a text report of AutoML results.
    
    Args:
        automl_model: Fitted AutoML model
        output_path: Optional path to save report
        
    Returns:
        Report text
    """
    lines = [
        "=" * 60,
        "AutoML Training Report",
        "=" * 60,
        "",
    ]
    
    # Basic info
    if hasattr(automl_model, 'backend_used_'):
        lines.append(f"Backend: {automl_model.backend_used_}")
    
    if hasattr(automl_model, 'fit_time_'):
        lines.append(f"Training time: {automl_model.fit_time_:.1f} seconds")
    
    # Configuration
    if hasattr(automl_model, 'config'):
        config = automl_model.config
        lines.extend([
            "",
            "Configuration:",
            f"  - Preset: {config.preset.value if hasattr(config.preset, 'value') else config.preset}",
            f"  - Time budget: {config.time_left_for_this_task}s",
            f"  - Ensemble size: {config.ensemble_size}",
            f"  - Metric: {config.metric}",
        ])
    
    # Best model
    lines.extend(["", "Best Model:"])
    
    if hasattr(automl_model, 'best_estimator_'):
        lines.append(f"  - Estimator: {automl_model.best_estimator_}")
    
    if hasattr(automl_model, 'best_loss_'):
        lines.append(f"  - Loss: {automl_model.best_loss_:.4f}")
        lines.append(f"  - Score: {-automl_model.best_loss_:.4f}")
    
    # Leaderboard
    if hasattr(automl_model, 'get_leaderboard'):
        leaderboard = automl_model.get_leaderboard()
        if leaderboard is not None and len(leaderboard) > 0:
            lines.extend(["", "Top Models:"])
            for i, row in leaderboard.head(5).iterrows():
                lines.append(f"  {i+1}. {row.to_dict()}")
    
    lines.extend(["", "=" * 60])
    
    report = "\n".join(lines)
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
    
    return report
