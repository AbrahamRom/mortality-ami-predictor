"""Neural Architecture Search (NAS) Integration Module.

This module provides Neural Architecture Search capabilities using AutoKeras
for automatic neural network architecture optimization. AutoKeras automatically
searches for the best neural network architecture for tabular data classification
and regression tasks.

NAS benefits:
- Automatic architecture search (layers, units, activations)
- Hyperparameter optimization for neural networks
- Transfer learning capabilities
- Ensemble of neural networks
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from ..models.custom_base import BaseCustomClassifier, BaseCustomRegressor
from ..config import RANDOM_SEED

logger = logging.getLogger(__name__)


def is_autokeras_available() -> bool:
    """Check if AutoKeras is available.
    
    Returns:
        True if AutoKeras can be imported
    """
    try:
        import autokeras
        return True
    except ImportError:
        return False


def is_tensorflow_available() -> bool:
    """Check if TensorFlow is available.
    
    Returns:
        True if TensorFlow can be imported
    """
    try:
        import tensorflow
        return True
    except ImportError:
        return False


def get_autokeras_version() -> Optional[str]:
    """Get AutoKeras version if available."""
    try:
        import autokeras
        return autokeras.__version__
    except ImportError:
        return None


class NASClassifier(BaseCustomClassifier):
    """Neural Architecture Search Classifier using AutoKeras.
    
    AutoKeras automatically searches for the best neural network architecture
    for classification tasks. It uses Bayesian optimization and network morphism
    to efficiently explore the architecture space.
    
    Attributes:
        max_trials: Maximum number of neural network architectures to try
        epochs: Number of epochs for training each architecture
        tuner: Tuning algorithm ('greedy', 'bayesian', 'hyperband', 'random')
        
    Example:
        >>> clf = NASClassifier(max_trials=10, epochs=50)
        >>> clf.fit(X_train, y_train)
        >>> proba = clf.predict_proba(X_test)
    """
    
    def __init__(
        self,
        max_trials: int = 10,
        epochs: int = 100,
        tuner: str = "greedy",
        max_model_size: Optional[int] = None,
        objective: str = "val_accuracy",
        directory: Optional[str] = None,
        project_name: str = "nas_classifier",
        overwrite: bool = True,
        random_state: int = RANDOM_SEED,
        verbose: int = 1,
        name: str = "NASClassifier",
        progress_callback: Optional[Callable[[str, float], None]] = None,
        **kwargs
    ):
        """
        Initialize NAS Classifier.
        
        Args:
            max_trials: Maximum number of different neural network architectures to try.
                Higher values may find better architectures but take longer.
            epochs: Number of training epochs for each architecture trial.
            tuner: Tuning algorithm to use:
                - 'greedy': Greedy search, fast but may miss optimal
                - 'bayesian': Bayesian optimization, good balance
                - 'hyperband': Hyperband algorithm, efficient for large searches
                - 'random': Random search, baseline
            max_model_size: Maximum size of the model in bytes (optional).
            objective: Metric to optimize ('val_accuracy', 'val_loss', 'val_auc').
            directory: Directory to store the search results.
            project_name: Name of the project for organizing results.
            overwrite: Whether to overwrite previous search results.
            random_state: Random seed for reproducibility.
            verbose: Verbosity level (0=silent, 1=progress, 2=detailed).
            name: Model name.
            progress_callback: Optional callback for progress updates.
            **kwargs: Additional AutoKeras arguments.
        """
        super().__init__(name=name)
        
        self.max_trials = max_trials
        self.epochs = epochs
        self.tuner = tuner
        self.max_model_size = max_model_size
        self.objective = objective
        self.directory = directory or "./autokeras_models"
        self.project_name = project_name
        self.overwrite = overwrite
        self.random_state = random_state
        self.verbose = verbose
        self.extra_kwargs = kwargs
        self.progress_callback = progress_callback
        
        # Runtime attributes
        self.model_: Optional[Any] = None
        self.best_model_: Optional[Any] = None
        self.history_: Optional[Dict] = None
        self.fit_time_: float = 0.0
        self.n_classes_: int = 0
        self.architecture_summary_: Optional[str] = None
        self.training_logs_: List[str] = []
    
    def fit(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series,
        validation_split: float = 0.2,
        **kwargs
    ) -> "NASClassifier":
        """
        Fit the NAS classifier.
        
        Performs neural architecture search to find the best network
        architecture and trains it on the provided data.
        
        Args:
            X: Training features
            y: Training labels
            validation_split: Fraction of data to use for validation
            **kwargs: Additional fit arguments
            
        Returns:
            Self (fitted classifier)
        """
        if not is_autokeras_available():
            raise ImportError(
                "AutoKeras is not installed. Install with: pip install autokeras"
            )
        
        if not is_tensorflow_available():
            raise ImportError(
                "TensorFlow is not installed. Install with: pip install tensorflow"
            )
        
        import autokeras as ak
        import tensorflow as tf
        
        # Set random seed
        tf.random.set_seed(self.random_state)
        np.random.seed(self.random_state)
        
        # Fix for Keras 3.x threading issue with name_scope_stack
        # This prevents the 'NoneType' object has no attribute 'pop' error
        try:
            from keras.src.backend.common import global_state
            # Ensure name_scope_stack is properly initialized for this thread
            if not hasattr(global_state, '_name_scope_stack') or global_state._name_scope_stack is None:
                global_state._name_scope_stack = []
        except (ImportError, AttributeError):
            pass  # Older Keras version, doesn't need this fix
        
        self._validate_input(X, training=True)
        y = self._validate_targets(y, training=True)
        
        # Convert to numpy arrays
        if isinstance(X, pd.DataFrame):
            X_train = X.values.astype(np.float32)
        else:
            X_train = np.asarray(X, dtype=np.float32)
        
        if isinstance(y, pd.Series):
            y_train = y.values
        else:
            y_train = np.asarray(y)
        
        # Ensure y is integer type for classification
        y_train = y_train.astype(np.int32)
        
        # Get number of classes
        self.n_classes_ = len(np.unique(y_train))
        
        start_time = time.time()
        
        if self.progress_callback:
            self.progress_callback("🧠 Iniciando Neural Architecture Search...", 0.0)
        
        self._log(f"🚀 Iniciando AutoKeras NAS con {self.max_trials} trials")
        self._log(f"📊 Dataset: {X_train.shape[0]} muestras, {X_train.shape[1]} features")
        self._log(f"🎯 Clases: {self.n_classes_}")
        
        # Create the classifier with increased failure tolerance
        self.model_ = ak.StructuredDataClassifier(
            max_trials=self.max_trials,
            tuner=self.tuner,
            max_model_size=self.max_model_size,
            objective=self.objective,
            directory=self.directory,
            project_name=self.project_name,
            overwrite=self.overwrite,
            seed=self.random_state,
            **self.extra_kwargs
        )
        
        if self.progress_callback:
            self.progress_callback("🔍 Iniciando búsqueda de arquitecturas...", 0.2)
        
        # Fit the model with threading-safe approach
        # Use a try-except to handle Keras threading issues
        max_retries = 3
        last_error = None
        
        for retry in range(max_retries):
            try:
                # Re-initialize global state for each retry to fix threading issues
                try:
                    from keras.src.backend.common import global_state
                    if not hasattr(global_state, '_name_scope_stack') or global_state._name_scope_stack is None:
                        global_state._name_scope_stack = []
                except (ImportError, AttributeError):
                    pass
                
                self.model_.fit(
                    X_train, 
                    y_train,
                    epochs=self.epochs,
                    validation_split=validation_split,
                    verbose=self.verbose,
                )
                break  # Success, exit retry loop
            except AttributeError as e:
                if "'NoneType' object has no attribute 'pop'" in str(e):
                    self._log(f"⚠️ Error de threading en Keras (intento {retry + 1}/{max_retries}), reintentando...")
                    last_error = e
                    # Try to fix the global state
                    try:
                        from keras.src.backend.common import global_state
                        global_state._name_scope_stack = []
                    except (ImportError, AttributeError):
                        pass
                    if retry == max_retries - 1:
                        raise RuntimeError(
                            f"Error de compatibilidad con Keras 3.x y Python 3.13. "
                            f"Esto es un bug conocido. Posibles soluciones:\n"
                            f"1. Usar Python 3.11 o 3.12\n"
                            f"2. Actualizar keras-tuner y autokeras a la última versión\n"
                            f"3. Ejecutar en el hilo principal (no en Streamlit background)\n"
                            f"Error original: {e}"
                        ) from e
                else:
                    raise
            except RuntimeError as e:
                if "consecutive failures" in str(e).lower():
                    self._log(f"⚠️ Múltiples fallos en trials (intento {retry + 1}/{max_retries})")
                    last_error = e
                    if retry == max_retries - 1:
                        raise RuntimeError(
                            f"AutoKeras encontró demasiados errores consecutivos. "
                            f"Esto puede deberse a:\n"
                            f"1. Incompatibilidad entre Python 3.13, Keras 3.x y AutoKeras\n"
                            f"2. Datos con valores inválidos (NaN, Inf)\n"
                            f"3. Problema de memoria\n\n"
                            f"Recomendaciones:\n"
                            f"- Usa Python 3.11 o 3.12 para mejor compatibilidad\n"
                            f"- Verifica que los datos no contengan NaN o valores infinitos\n"
                            f"- Reduce el número de trials o epochs\n"
                            f"Error original: {e}"
                        ) from e
                else:
                    raise
            except Exception as e:
                self._log(f"❌ Error durante NAS: {str(e)}")
                raise
        
        # Get the best model
        self.best_model_ = self.model_.export_model()
        
        # Get architecture summary
        try:
            import io
            stream = io.StringIO()
            self.best_model_.summary(print_fn=lambda x: stream.write(x + '\n'))
            self.architecture_summary_ = stream.getvalue()
        except Exception:
            self.architecture_summary_ = "Architecture summary not available"
        
        self.fit_time_ = time.time() - start_time
        
        self._log(f"✅ NAS completado en {self.fit_time_:.1f}s")
        self._log(f"🏆 Mejor arquitectura encontrada")
        
        self.is_fitted_ = True
        
        if self.progress_callback:
            self.progress_callback(
                f"✅ NAS completado: {self.max_trials} arquitecturas evaluadas",
                1.0
            )
        
        logger.info(
            f"NAS training completed: time={self.fit_time_:.1f}s, "
            f"trials={self.max_trials}"
        )
        
        return self
    
    def _log(self, message: str) -> None:
        """Add a log entry."""
        timestamp = time.strftime("%H:%M:%S")
        entry = f"[{timestamp}] {message}"
        self.training_logs_.append(entry)
    
    def get_training_logs(self) -> List[str]:
        """Get the training logs."""
        return self.training_logs_.copy()
    
    def predict_proba(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Predict class probabilities."""
        self._validate_input(X, training=False)
        
        if isinstance(X, pd.DataFrame):
            X_pred = X.values
        else:
            X_pred = X
        
        # Get raw predictions
        predictions = self.best_model_.predict(X_pred, verbose=0)
        
        # Handle binary classification
        if self.n_classes_ == 2 and predictions.shape[1] == 1:
            proba = np.column_stack([1 - predictions.flatten(), predictions.flatten()])
        else:
            proba = predictions
        
        return proba
    
    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Predict class labels."""
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)
    
    def get_best_model(self) -> Any:
        """Get the best Keras model found by NAS."""
        return self.best_model_
    
    def get_architecture_summary(self) -> str:
        """Get the summary of the best architecture found."""
        return self.architecture_summary_ or "Model not fitted yet"
    
    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get parameters for this estimator."""
        return {
            "max_trials": self.max_trials,
            "epochs": self.epochs,
            "tuner": self.tuner,
            "objective": self.objective,
            "random_state": self.random_state,
            "verbose": self.verbose,
            "name": self.name,
        }
    
    def set_params(self, **params) -> "NASClassifier":
        """Set parameters for this estimator."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get extended metadata."""
        metadata = super().get_metadata()
        metadata.update({
            'backend': 'autokeras',
            'max_trials': self.max_trials,
            'epochs': self.epochs,
            'tuner': self.tuner,
            'fit_time_seconds': self.fit_time_,
            'n_classes': self.n_classes_,
        })
        return metadata
    
    def save_model(self, path: str) -> None:
        """Save the best model to disk."""
        if self.best_model_ is not None:
            self.best_model_.save(path)
            logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str) -> None:
        """Load a saved model from disk."""
        import tensorflow as tf
        self.best_model_ = tf.keras.models.load_model(path)
        self.is_fitted_ = True
        logger.info(f"Model loaded from {path}")


class NASRegressor(BaseCustomRegressor):
    """Neural Architecture Search Regressor using AutoKeras.
    
    AutoKeras automatically searches for the best neural network architecture
    for regression tasks.
    """
    
    def __init__(
        self,
        max_trials: int = 10,
        epochs: int = 100,
        tuner: str = "greedy",
        max_model_size: Optional[int] = None,
        objective: str = "val_loss",
        directory: Optional[str] = None,
        project_name: str = "nas_regressor",
        overwrite: bool = True,
        random_state: int = RANDOM_SEED,
        verbose: int = 1,
        name: str = "NASRegressor",
        progress_callback: Optional[Callable[[str, float], None]] = None,
        **kwargs
    ):
        """Initialize NAS Regressor."""
        super().__init__(name=name)
        
        self.max_trials = max_trials
        self.epochs = epochs
        self.tuner = tuner
        self.max_model_size = max_model_size
        self.objective = objective
        self.directory = directory or "./autokeras_models"
        self.project_name = project_name
        self.overwrite = overwrite
        self.random_state = random_state
        self.verbose = verbose
        self.extra_kwargs = kwargs
        self.progress_callback = progress_callback
        
        self.model_: Optional[Any] = None
        self.best_model_: Optional[Any] = None
        self.fit_time_: float = 0.0
        self.architecture_summary_: Optional[str] = None
        self.training_logs_: List[str] = []
    
    def fit(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series,
        validation_split: float = 0.2,
        **kwargs
    ) -> "NASRegressor":
        """Fit the NAS regressor."""
        if not is_autokeras_available():
            raise ImportError("AutoKeras is not installed")
        
        import autokeras as ak
        import tensorflow as tf
        
        tf.random.set_seed(self.random_state)
        np.random.seed(self.random_state)
        
        self._validate_input(X, training=True)
        
        if isinstance(X, pd.DataFrame):
            X_train = X.values
        else:
            X_train = X
        
        if isinstance(y, pd.Series):
            y_train = y.values
        else:
            y_train = y
        
        start_time = time.time()
        
        if self.progress_callback:
            self.progress_callback("🧠 Iniciando Neural Architecture Search...", 0.0)
        
        self._log(f"🚀 Iniciando AutoKeras NAS Regressor con {self.max_trials} trials")
        
        self.model_ = ak.StructuredDataRegressor(
            max_trials=self.max_trials,
            tuner=self.tuner,
            max_model_size=self.max_model_size,
            objective=self.objective,
            directory=self.directory,
            project_name=self.project_name,
            overwrite=self.overwrite,
            seed=self.random_state,
            **self.extra_kwargs
        )
        
        self.model_.fit(
            X_train,
            y_train,
            epochs=self.epochs,
            validation_split=validation_split,
            verbose=self.verbose,
            **kwargs
        )
        
        self.best_model_ = self.model_.export_model()
        
        try:
            import io
            stream = io.StringIO()
            self.best_model_.summary(print_fn=lambda x: stream.write(x + '\n'))
            self.architecture_summary_ = stream.getvalue()
        except Exception:
            self.architecture_summary_ = "Architecture summary not available"
        
        self.fit_time_ = time.time() - start_time
        
        self._log(f"✅ NAS completado en {self.fit_time_:.1f}s")
        
        self.is_fitted_ = True
        
        if self.progress_callback:
            self.progress_callback(
                f"✅ NAS Regressor completado",
                1.0
            )
        
        return self
    
    def _log(self, message: str) -> None:
        """Add a log entry."""
        timestamp = time.strftime("%H:%M:%S")
        entry = f"[{timestamp}] {message}"
        self.training_logs_.append(entry)
    
    def get_training_logs(self) -> List[str]:
        """Get the training logs."""
        return self.training_logs_.copy()
    
    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        self._validate_input(X, training=False)
        
        if isinstance(X, pd.DataFrame):
            X_pred = X.values
        else:
            X_pred = X
        
        return self.best_model_.predict(X_pred, verbose=0).flatten()
    
    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get parameters."""
        return {
            "max_trials": self.max_trials,
            "epochs": self.epochs,
            "tuner": self.tuner,
            "objective": self.objective,
            "random_state": self.random_state,
            "verbose": self.verbose,
            "name": self.name,
        }
    
    def set_params(self, **params) -> "NASRegressor":
        """Set parameters."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get extended metadata."""
        metadata = super().get_metadata()
        metadata.update({
            'backend': 'autokeras',
            'max_trials': self.max_trials,
            'epochs': self.epochs,
            'tuner': self.tuner,
            'fit_time_seconds': self.fit_time_,
        })
        return metadata


# ============================================================================
# Configuration classes for NAS
# ============================================================================

class NASConfig:
    """Configuration for Neural Architecture Search."""
    
    # Available tuning algorithms
    TUNERS = {
        "greedy": "Greedy search - fast but may miss optimal",
        "bayesian": "Bayesian optimization - good balance",
        "hyperband": "Hyperband - efficient for large searches", 
        "random": "Random search - baseline comparison",
    }
    
    # Preset configurations
    PRESETS = {
        "quick": {
            "max_trials": 5,
            "epochs": 50,
            "tuner": "greedy",
            "description": "Quick exploration (5-10 min)",
        },
        "balanced": {
            "max_trials": 15,
            "epochs": 100,
            "tuner": "bayesian",
            "description": "Balanced search (30-60 min)",
        },
        "thorough": {
            "max_trials": 30,
            "epochs": 150,
            "tuner": "hyperband",
            "description": "Thorough search (2-4 hours)",
        },
        "exhaustive": {
            "max_trials": 50,
            "epochs": 200,
            "tuner": "bayesian",
            "description": "Exhaustive search (4+ hours)",
        },
    }
    
    @classmethod
    def get_preset(cls, preset_name: str) -> Dict[str, Any]:
        """Get configuration for a preset."""
        if preset_name not in cls.PRESETS:
            raise ValueError(f"Unknown preset: {preset_name}. "
                           f"Available: {list(cls.PRESETS.keys())}")
        return cls.PRESETS[preset_name].copy()
    
    @classmethod
    def list_presets(cls) -> Dict[str, str]:
        """List available presets with descriptions."""
        return {k: v["description"] for k, v in cls.PRESETS.items()}
