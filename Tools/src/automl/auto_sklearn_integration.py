"""Auto-sklearn Integration Module.

Provides wrapper classes for auto-sklearn that integrate with
the existing custom model infrastructure.

Note:
    auto-sklearn only works on Linux. For Windows, use WSL or
    the FLAML alternative (flaml_integration.py).
"""
from __future__ import annotations

import logging
import os
import pickle
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from ..models.custom_base import BaseCustomClassifier, BaseCustomRegressor
from ..config import RANDOM_SEED
from .config import AutoMLConfig, AutoMLPreset

logger = logging.getLogger(__name__)


def is_autosklearn_available() -> bool:
    """Check if auto-sklearn is available.
    
    Returns:
        True if auto-sklearn can be imported
    """
    try:
        import autosklearn
        return True
    except ImportError:
        return False


def is_linux() -> bool:
    """Check if running on Linux."""
    return sys.platform.startswith('linux')


class AutoMLClassifier(BaseCustomClassifier):
    """AutoML Classifier using auto-sklearn or FLAML fallback.
    
    This classifier automatically searches for the best model architecture
    and hyperparameters within a time budget. It integrates seamlessly
    with the existing sklearn-based pipeline.
    
    Attributes:
        config: AutoML configuration
        automl_: Fitted AutoML object
        leaderboard_: DataFrame with model rankings
        training_history_: List of evaluated configurations
        best_model_: Best single model (extracted from ensemble)
    
    Example:
        >>> config = AutoMLConfig.from_preset("quick")
        >>> clf = AutoMLClassifier(config=config)
        >>> clf.fit(X_train, y_train)
        >>> predictions = clf.predict_proba(X_test)
    """
    
    def __init__(
        self,
        config: Optional[AutoMLConfig] = None,
        preset: str | AutoMLPreset = AutoMLPreset.BALANCED,
        time_left_for_this_task: Optional[int] = None,
        per_run_time_limit: Optional[int] = None,
        ensemble_size: Optional[int] = None,
        metric: Optional[str] = None,
        include_estimators: Optional[List[str]] = None,
        exclude_estimators: Optional[List[str]] = None,
        n_jobs: int = -1,
        random_state: int = RANDOM_SEED,
        use_flaml_fallback: bool = True,
        name: str = "AutoMLClassifier",
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ):
        """
        Initialize AutoML Classifier.
        
        Args:
            config: Pre-built AutoMLConfig (overrides other params)
            preset: Configuration preset if config not provided
            time_left_for_this_task: Total time budget in seconds
            per_run_time_limit: Max time per model evaluation
            ensemble_size: Number of models in final ensemble
            metric: Optimization metric (roc_auc, f1, etc.)
            include_estimators: Estimators to include (None = all)
            exclude_estimators: Estimators to exclude
            n_jobs: Parallel jobs (-1 = all cores)
            random_state: Random seed
            use_flaml_fallback: Use FLAML if auto-sklearn unavailable
            name: Model name
            progress_callback: Optional callback(message, progress)
        """
        super().__init__(name=name)
        
        # Build or use provided config
        if config is not None:
            self.config = config
        else:
            self.config = AutoMLConfig.from_preset(preset)
            
            # Apply overrides
            if time_left_for_this_task is not None:
                self.config.time_left_for_this_task = time_left_for_this_task
            if per_run_time_limit is not None:
                self.config.per_run_time_limit = per_run_time_limit
            if ensemble_size is not None:
                self.config.ensemble_size = ensemble_size
            if metric is not None:
                self.config.metric = metric
            if include_estimators is not None:
                self.config.include_estimators = include_estimators
            if exclude_estimators is not None:
                self.config.exclude_estimators = exclude_estimators
            self.config.n_jobs = n_jobs
            self.config.random_state = random_state
        
        self.use_flaml_fallback = use_flaml_fallback
        self.progress_callback = progress_callback
        
        # Runtime attributes
        self.automl_: Optional[Any] = None
        self.leaderboard_: Optional[pd.DataFrame] = None
        self.training_history_: List[Dict[str, Any]] = []
        self.best_model_: Optional[Any] = None
        self.backend_used_: Optional[str] = None
        self.fit_time_: float = 0.0
        
        # Store original params for get_params/set_params
        self._init_params = {
            "config": config,
            "preset": preset,
            "time_left_for_this_task": time_left_for_this_task,
            "per_run_time_limit": per_run_time_limit,
            "ensemble_size": ensemble_size,
            "metric": metric,
            "include_estimators": include_estimators,
            "exclude_estimators": exclude_estimators,
            "n_jobs": n_jobs,
            "random_state": random_state,
            "use_flaml_fallback": use_flaml_fallback,
            "name": name,
        }
    
    def fit(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series,
        **kwargs
    ) -> "AutoMLClassifier":
        """
        Fit the AutoML classifier.
        
        This method runs the full AutoML search to find the best
        model architecture and hyperparameters.
        
        Args:
            X: Training features
            y: Training labels
            **kwargs: Additional arguments passed to fit
            
        Returns:
            Self (fitted classifier)
        """
        self._validate_input(X, training=True)
        y = self._validate_targets(y, training=True)
        
        X_array = self._convert_to_array(X)
        y_array = y if isinstance(y, np.ndarray) else y.values
        
        start_time = time.time()
        
        if self.progress_callback:
            self.progress_callback("🔍 Iniciando búsqueda AutoML...", 0.0)
        
        # Try auto-sklearn first
        if is_autosklearn_available() and is_linux():
            self._fit_autosklearn(X_array, y_array, **kwargs)
            self.backend_used_ = "auto-sklearn"
        elif self.use_flaml_fallback:
            if self.progress_callback:
                self.progress_callback("⚠️ auto-sklearn no disponible, usando FLAML...", 0.05)
            self._fit_flaml(X_array, y_array, **kwargs)
            self.backend_used_ = "flaml"
        else:
            raise ImportError(
                "auto-sklearn is not available and FLAML fallback is disabled. "
                "auto-sklearn requires Linux (use WSL on Windows)."
            )
        
        self.fit_time_ = time.time() - start_time
        self.is_fitted_ = True
        
        if self.progress_callback:
            self.progress_callback(
                f"✅ AutoML completado en {self.fit_time_:.1f}s usando {self.backend_used_}",
                1.0
            )
        
        return self
    
    def _fit_autosklearn(
        self,
        X: np.ndarray,
        y: np.ndarray,
        **kwargs
    ) -> None:
        """Fit using auto-sklearn."""
        from autosklearn.classification import AutoSklearnClassifier
        
        # Create temp/output folders
        os.makedirs(self.config.tmp_folder, exist_ok=True)
        os.makedirs(self.config.output_folder, exist_ok=True)
        
        # Get auto-sklearn kwargs
        ask_kwargs = self.config.to_autosklearn_kwargs()
        
        if self.progress_callback:
            self.progress_callback("🚀 Ejecutando auto-sklearn...", 0.1)
        
        # Initialize and fit
        self.automl_ = AutoSklearnClassifier(**ask_kwargs)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.automl_.fit(X, y, **kwargs)
        
        # Build leaderboard
        self._build_leaderboard_autosklearn()
        
        # Extract best single model
        self._extract_best_model_autosklearn()
    
    def _fit_flaml(
        self,
        X: np.ndarray,
        y: np.ndarray,
        **kwargs
    ) -> None:
        """Fit using FLAML."""
        try:
            from flaml import AutoML
        except ImportError:
            raise ImportError(
                "FLAML is not installed. Install it with: pip install flaml[automl]"
            )
        
        # Get FLAML kwargs
        flaml_kwargs = self.config.to_flaml_kwargs()
        flaml_kwargs["task"] = "classification"
        
        if self.progress_callback:
            self.progress_callback("🚀 Ejecutando FLAML...", 0.1)
        
        # Initialize and fit
        self.automl_ = AutoML()
        self.automl_.fit(X, y, **flaml_kwargs, **kwargs)
        
        # Build leaderboard
        self._build_leaderboard_flaml()
        
        # Best model is directly available
        self.best_model_ = self.automl_.model
    
    def _build_leaderboard_autosklearn(self) -> None:
        """Build leaderboard from auto-sklearn results."""
        try:
            # Get run history
            results = self.automl_.cv_results_
            
            rows = []
            for i in range(len(results.get('mean_test_score', []))):
                rows.append({
                    'rank': results.get('rank_test_score', [i+1])[i],
                    'mean_score': results.get('mean_test_score', [0])[i],
                    'std_score': results.get('std_test_score', [0])[i],
                    'mean_fit_time': results.get('mean_fit_time', [0])[i],
                    'params': str(results.get('params', [{}])[i]),
                })
            
            self.leaderboard_ = pd.DataFrame(rows).sort_values('rank')
            
            # Also get model types from sprint statistics
            try:
                stats = self.automl_.sprint_statistics()
                self.training_history_.append({
                    'sprint_stats': stats,
                    'n_models': len(rows),
                })
            except:
                pass
                
        except Exception as e:
            logger.warning(f"Could not build leaderboard: {e}")
            self.leaderboard_ = pd.DataFrame()
    
    def _build_leaderboard_flaml(self) -> None:
        """Build leaderboard from FLAML results."""
        try:
            # FLAML stores best configs
            rows = []
            
            if hasattr(self.automl_, 'best_config_per_estimator'):
                for estimator, config in self.automl_.best_config_per_estimator.items():
                    if config is not None:
                        rows.append({
                            'rank': len(rows) + 1,
                            'estimator': estimator,
                            'mean_score': config.get('val_loss', 0) * -1,  # FLAML uses loss
                            'config': str(config),
                        })
            
            # Add best overall
            if hasattr(self.automl_, 'best_estimator'):
                rows.insert(0, {
                    'rank': 0,
                    'estimator': self.automl_.best_estimator,
                    'mean_score': -self.automl_.best_loss if hasattr(self.automl_, 'best_loss') else 0,
                    'config': str(self.automl_.best_config) if hasattr(self.automl_, 'best_config') else '',
                })
            
            self.leaderboard_ = pd.DataFrame(rows)
            
            # Training history
            self.training_history_.append({
                'best_estimator': getattr(self.automl_, 'best_estimator', None),
                'best_loss': getattr(self.automl_, 'best_loss', None),
                'best_config': getattr(self.automl_, 'best_config', None),
                'time_total': getattr(self.automl_, 'time_total_s', 0),
            })
            
        except Exception as e:
            logger.warning(f"Could not build leaderboard: {e}")
            self.leaderboard_ = pd.DataFrame()
    
    def _extract_best_model_autosklearn(self) -> None:
        """Extract best single model from auto-sklearn ensemble."""
        try:
            # Get ensemble models
            ensemble = self.automl_.get_models_with_weights()
            
            if ensemble:
                # Get model with highest weight
                best_weight = 0
                for weight, model in ensemble:
                    if weight > best_weight:
                        best_weight = weight
                        self.best_model_ = model
        except Exception as e:
            logger.warning(f"Could not extract best model: {e}")
    
    def predict_proba(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Features to predict on
            
        Returns:
            Array of shape (n_samples, n_classes) with probabilities
        """
        self._validate_input(X, training=False)
        X_array = self._convert_to_array(X)
        
        return self.automl_.predict_proba(X_array)
    
    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """
        Predict class labels.
        
        Args:
            X: Features to predict on
            
        Returns:
            Predicted class labels
        """
        self._validate_input(X, training=False)
        X_array = self._convert_to_array(X)
        
        return self.automl_.predict(X_array)
    
    def get_leaderboard(self) -> pd.DataFrame:
        """Get the model leaderboard.
        
        Returns:
            DataFrame with model rankings and scores
        """
        if self.leaderboard_ is None:
            return pd.DataFrame()
        return self.leaderboard_.copy()
    
    def get_best_model(self) -> Any:
        """Get the best single model.
        
        Returns:
            Best model object (can be used independently)
        """
        return self.best_model_
    
    def get_ensemble_weights(self) -> List[Tuple[float, Any]]:
        """Get ensemble models with their weights.
        
        Returns:
            List of (weight, model) tuples
        """
        if self.backend_used_ == "auto-sklearn" and self.automl_ is not None:
            try:
                return self.automl_.get_models_with_weights()
            except:
                pass
        return []
    
    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get parameters for this estimator."""
        params = self._init_params.copy()
        params["name"] = self.name
        return params
    
    def set_params(self, **params) -> "AutoMLClassifier":
        """Set parameters for this estimator."""
        for key, value in params.items():
            if key in self._init_params:
                self._init_params[key] = value
            if hasattr(self, key):
                setattr(self, key, value)
        return self
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get extended metadata including AutoML info."""
        metadata = super().get_metadata()
        metadata.update({
            'backend': self.backend_used_,
            'fit_time_seconds': self.fit_time_,
            'config_preset': self.config.preset.value,
            'time_budget': self.config.time_left_for_this_task,
            'ensemble_size': self.config.ensemble_size,
            'metric': self.config.metric,
            'n_models_evaluated': len(self.leaderboard_) if self.leaderboard_ is not None else 0,
        })
        return metadata
    
    def save(self, path: Union[str, Path]) -> Path:
        """Save AutoML model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save the full automl object
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        
        return path
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> "AutoMLClassifier":
        """Load AutoML model from disk."""
        with open(path, 'rb') as f:
            model = pickle.load(f)
        
        if not isinstance(model, cls):
            raise TypeError(f"Loaded object is not a {cls.__name__}")
        
        return model


class AutoMLRegressor(BaseCustomRegressor):
    """AutoML Regressor using auto-sklearn or FLAML fallback.
    
    Similar to AutoMLClassifier but for regression tasks.
    """
    
    def __init__(
        self,
        config: Optional[AutoMLConfig] = None,
        preset: str | AutoMLPreset = AutoMLPreset.BALANCED,
        time_left_for_this_task: Optional[int] = None,
        metric: str = "r2",
        use_flaml_fallback: bool = True,
        name: str = "AutoMLRegressor",
        **kwargs
    ):
        """Initialize AutoML Regressor."""
        super().__init__(name=name)
        
        if config is not None:
            self.config = config
        else:
            self.config = AutoMLConfig.from_preset(preset)
            if time_left_for_this_task is not None:
                self.config.time_left_for_this_task = time_left_for_this_task
            self.config.metric = metric
        
        self.use_flaml_fallback = use_flaml_fallback
        self.automl_: Optional[Any] = None
        self.backend_used_: Optional[str] = None
        self.fit_time_: float = 0.0
    
    def fit(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series,
        **kwargs
    ) -> "AutoMLRegressor":
        """Fit the AutoML regressor."""
        self._validate_input(X, training=True)
        
        X_array = self._convert_to_array(X)
        y_array = y.values if isinstance(y, pd.Series) else y
        
        start_time = time.time()
        
        if is_autosklearn_available() and is_linux():
            self._fit_autosklearn(X_array, y_array, **kwargs)
            self.backend_used_ = "auto-sklearn"
        elif self.use_flaml_fallback:
            self._fit_flaml(X_array, y_array, **kwargs)
            self.backend_used_ = "flaml"
        else:
            raise ImportError("auto-sklearn not available and FLAML fallback disabled")
        
        self.fit_time_ = time.time() - start_time
        self.is_fitted_ = True
        
        return self
    
    def _fit_autosklearn(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
        """Fit using auto-sklearn regressor."""
        from autosklearn.regression import AutoSklearnRegressor
        
        os.makedirs(self.config.tmp_folder, exist_ok=True)
        os.makedirs(self.config.output_folder, exist_ok=True)
        
        ask_kwargs = self.config.to_autosklearn_kwargs()
        
        self.automl_ = AutoSklearnRegressor(**ask_kwargs)
        self.automl_.fit(X, y, **kwargs)
    
    def _fit_flaml(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
        """Fit using FLAML regressor."""
        from flaml import AutoML
        
        flaml_kwargs = self.config.to_flaml_kwargs()
        flaml_kwargs["task"] = "regression"
        
        self.automl_ = AutoML()
        self.automl_.fit(X, y, **flaml_kwargs, **kwargs)
    
    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        self._validate_input(X, training=False)
        X_array = self._convert_to_array(X)
        
        return self.automl_.predict(X_array)
    
    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get parameters."""
        return {
            "config": self.config,
            "use_flaml_fallback": self.use_flaml_fallback,
            "name": self.name,
        }
    
    def set_params(self, **params) -> "AutoMLRegressor":
        """Set parameters."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self
