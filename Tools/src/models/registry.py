"""Model registry for easy access to all models."""
from __future__ import annotations

from typing import Dict, Optional, Tuple, List

from .classifiers import make_classifiers, make_automl_classifiers
from .regressors import make_regressors


class ModelRegistry:
    """Central registry for all available models.
    
    Provides a single interface to access classification, regression,
    and AutoML models.
    """
    
    def __init__(self):
        """Initialize registry with available models."""
        self._classifiers = make_classifiers()
        self._regressors = make_regressors()
        self._automl_classifiers = make_automl_classifiers()
    
    def get_classifier(self, name: str) -> Tuple[object, Dict]:
        """Get classifier and its parameter grid.
        
        Args:
            name: Model name ('knn', 'logreg', 'dtree', 'xgb', 'lgbm')
            
        Returns:
            Tuple of (model, param_grid)
            
        Raises:
            KeyError: If model name not found
        """
        if name not in self._classifiers:
            available = list(self._classifiers.keys())
            raise KeyError(
                f"Classifier '{name}' not found. "
                f"Available: {available}"
            )
        return self._classifiers[name]
    
    def get_automl_classifier(self, name: str) -> Tuple[object, Dict]:
        """Get AutoML classifier and its configuration.
        
        Args:
            name: AutoML preset name ('automl_quick', 'automl_balanced', etc.)
            
        Returns:
            Tuple of (model, config_dict)
            
        Raises:
            KeyError: If preset name not found
        """
        if name not in self._automl_classifiers:
            available = list(self._automl_classifiers.keys())
            raise KeyError(
                f"AutoML preset '{name}' not found. "
                f"Available: {available}"
            )
        return self._automl_classifiers[name]
    
    def get_regressor(self, name: str) -> Tuple[object, Dict]:
        """Get regressor and its parameter grid.
        
        Args:
            name: Model name
            
        Returns:
            Tuple of (model, param_grid)
            
        Raises:
            KeyError: If model name not found
        """
        if name not in self._regressors:
            available = list(self._regressors.keys())
            raise KeyError(
                f"Regressor '{name}' not found. "
                f"Available: {available}"
            )
        return self._regressors[name]
    
    def list_classifiers(self) -> list[str]:
        """List all available classifier names."""
        return list(self._classifiers.keys())
    
    def list_automl_classifiers(self) -> list[str]:
        """List all available AutoML classifier presets."""
        return list(self._automl_classifiers.keys())
    
    def list_regressors(self) -> list[str]:
        """List all available regressor names."""
        return list(self._regressors.keys())
    
    def is_automl_model(self, name: str) -> bool:
        """Check if a model name is an AutoML model."""
        return name.startswith('automl_') or name in self._automl_classifiers


# Global registry instance
_registry = ModelRegistry()


def get_model(
    name: str,
    task: str = "classification",
) -> Tuple[object, Dict]:
    """Get model and parameter grid from global registry.
    
    Args:
        name: Model name
        task: Task type ('classification' or 'regression')
        
    Returns:
        Tuple of (model, param_grid)
    """
    if task == "classification":
        return _registry.get_classifier(name)
    elif task == "regression":
        return _registry.get_regressor(name)
    else:
        raise ValueError(f"Unknown task: {task}")


def get_automl_model(name: str) -> Tuple[object, Dict]:
    """Get AutoML model from global registry.
    
    Args:
        name: AutoML preset name
        
    Returns:
        Tuple of (model, config)
    """
    return _registry.get_automl_classifier(name)


def is_automl_model(name: str) -> bool:
    """Check if a model name is an AutoML model."""
    return _registry.is_automl_model(name)
