"""Custom Models Base Infrastructure.

This module provides base classes and utilities for defining and managing
custom model architectures that can integrate seamlessly with the existing
sklearn-based pipeline.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List, Union, TYPE_CHECKING
import json
import pickle

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin


class BaseCustomModel(BaseEstimator, ABC):
    """Abstract base class for custom models.
    
    Custom models should inherit from this class and implement the required methods.
    This ensures compatibility with sklearn's ecosystem (cross-validation, pipelines, etc.).
    
    Attributes:
        name: Model name.
        n_features_in_: Number of input features (set during fit).
        feature_names_in_: Feature names (set during fit).
        is_fitted_: Whether model has been fitted.
    """
    
    def __init__(self, name: str = "CustomModel"):
        """
        Initialize custom model.
        
        Args:
            name: Model name for identification.
        """
        self.name = name
        self.n_features_in_: Optional[int] = None
        self.feature_names_in_: Optional[List[str]] = None
        self.is_fitted_: bool = False
    
    @abstractmethod
    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series], **kwargs) -> 'BaseCustomModel':
        """
        Fit the model to training data.
        
        Args:
            X: Training features.
            y: Training targets.
            **kwargs: Additional fitting parameters.
        
        Returns:
            Self (fitted model).
        """
        pass
    
    @abstractmethod
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Features to predict on.
        
        Returns:
            Predictions array.
        """
        pass
    
    def _validate_input(self, X: Union[np.ndarray, pd.DataFrame], training: bool = False):
        """
        Validate input data.
        
        Args:
            X: Input features.
            training: Whether this is during training (fit).
        
        Raises:
            ValueError: If input is invalid.
        """
        if isinstance(X, pd.DataFrame):
            n_features = X.shape[1]
            feature_names = list(X.columns)
        elif isinstance(X, np.ndarray):
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            n_features = X.shape[1]
            feature_names = [f"feature_{i}" for i in range(n_features)]
        else:
            raise ValueError(f"X must be numpy array or pandas DataFrame, got {type(X)}")
        
        if training:
            # During training, store feature info
            self.n_features_in_ = n_features
            self.feature_names_in_ = feature_names
        else:
            # During prediction, validate against training info
            if not self.is_fitted_:
                raise ValueError("Model must be fitted before prediction")
            
            if n_features != self.n_features_in_:
                raise ValueError(
                    f"X has {n_features} features, but model was fitted with "
                    f"{self.n_features_in_} features"
                )
    
    def _convert_to_array(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Convert input to numpy array."""
        if isinstance(X, pd.DataFrame):
            return X.values
        return X
    
    def __getstate__(self):
        """Support for pickling."""
        state = self.__dict__.copy()
        return state
    
    def __setstate__(self, state):
        """Support for unpickling."""
        self.__dict__.update(state)
    
    def save(self, path: Union[str, Path]) -> Path:
        """
        Save model to disk.
        
        Args:
            path: Path to save model (should end with .pkl).
        
        Returns:
            Path to saved model.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        
        return path
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'BaseCustomModel':
        """
        Load model from disk.
        
        Args:
            path: Path to saved model.
        
        Returns:
            Loaded model instance.
        """
        with open(path, 'rb') as f:
            model = pickle.load(f)
        
        if not isinstance(model, cls):
            raise TypeError(f"Loaded object is not an instance of {cls.__name__}")
        
        return model
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get model metadata.
        
        Returns:
            Dictionary with model metadata.
        """
        return {
            'name': self.name,
            'class': self.__class__.__name__,
            'n_features': self.n_features_in_,
            'feature_names': self.feature_names_in_,
            'is_fitted': self.is_fitted_,
        }


class BaseCustomClassifier(BaseCustomModel, ClassifierMixin):
    """Base class for custom classifiers.
    
    Extends BaseCustomModel with classification-specific functionality.
    """
    _estimator_type = "classifier"
    
    def __init__(self, name: str = "CustomClassifier"):
        super().__init__(name=name)
        self.classes_: Optional[np.ndarray] = None
        self.n_classes_: Optional[int] = None
    
    @abstractmethod
    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Features to predict on.
        
        Returns:
            Array of shape (n_samples, n_classes) with probabilities.
        """
        pass
    
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Predict class labels.
        
        Args:
            X: Features to predict on.
        
        Returns:
            Predicted class labels.
        """
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]
    
    def _validate_targets(self, y: Union[np.ndarray, pd.Series], training: bool = False):
        """
        Validate target labels.
        
        Args:
            y: Target labels.
            training: Whether this is during training.
        """
        if isinstance(y, pd.Series):
            y = y.values
        
        if training:
            self.classes_ = np.unique(y)
            self.n_classes_ = len(self.classes_)
        
        return y
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get classifier metadata."""
        metadata = super().get_metadata()
        metadata.update({
            'n_classes': self.n_classes_,
            'classes': self.classes_.tolist() if self.classes_ is not None else None,
        })
        return metadata


class BaseCustomRegressor(BaseCustomModel, RegressorMixin):
    """Base class for custom regressors.
    
    Extends BaseCustomModel with regression-specific functionality.
    """
    _estimator_type = "regressor"
    
    def __init__(self, name: str = "CustomRegressor"):
        super().__init__(name=name)


class CustomModelWrapper:
    """Wrapper for integrating custom models with the training pipeline.
    
    This wrapper provides a consistent interface for custom models,
    allowing them to work seamlessly with the existing infrastructure.
    """
    
    def __init__(
        self,
        model: BaseCustomModel,
        preprocessing: Optional[Any] = None,
        hyperparameters: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize wrapper.
        
        Args:
            model: Custom model instance.
            preprocessing: Optional preprocessing pipeline.
            hyperparameters: Model hyperparameters.
        """
        self.model = model
        self.preprocessing = preprocessing
        self.hyperparameters = hyperparameters or {}
    
    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series], **kwargs):
        """Fit the model."""
        if self.preprocessing:
            X = self.preprocessing.fit_transform(X)
        
        self.model.fit(X, y, **kwargs)
        return self
    
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Make predictions."""
        if self.preprocessing:
            X = self.preprocessing.transform(X)
        
        return self.model.predict(X)
    
    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Predict probabilities (for classifiers)."""
        if not isinstance(self.model, BaseCustomClassifier):
            raise AttributeError("Model does not support probability predictions")
        
        if self.preprocessing:
            X = self.preprocessing.transform(X)
        
        return self.model.predict_proba(X)
    
    def save(self, path: Union[str, Path]) -> Path:
        """Save wrapped model."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = path.parent / f"{path.stem}_model.pkl"
        self.model.save(model_path)
        
        # Save preprocessing if exists
        preprocessing_path = None
        if self.preprocessing:
            preprocessing_path = path.parent / f"{path.stem}_preprocessing.pkl"
            with open(preprocessing_path, 'wb') as f:
                pickle.dump(self.preprocessing, f)
        
        # Save wrapper metadata
        metadata = {
            'model_path': str(model_path),
            'preprocessing_path': str(preprocessing_path) if preprocessing_path else None,
            'hyperparameters': self.hyperparameters,
        }
        
        with open(path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return path
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'CustomModelWrapper':
        """Load wrapped model."""
        path = Path(path)
        
        with open(path, 'r') as f:
            metadata = json.load(f)
        
        # Load model
        model = BaseCustomModel.load(metadata['model_path'])
        
        # Load preprocessing if exists
        preprocessing = None
        if metadata['preprocessing_path']:
            with open(metadata['preprocessing_path'], 'rb') as f:
                preprocessing = pickle.load(f)
        
        return cls(
            model=model,
            preprocessing=preprocessing,
            hyperparameters=metadata.get('hyperparameters')
        )


# Example custom model implementations

class SimpleMLPClassifier(BaseCustomClassifier):
    """Simple MLP classifier as an example custom model.
    
    This is a basic implementation using sklearn's MLPClassifier
    to demonstrate the custom model interface.
    """
    
    def __init__(
        self,
        hidden_layer_sizes: Tuple[int, ...] = (100,),
        activation: str = 'relu',
        learning_rate: float = 0.001,
        epochs: int = 200,
        random_state: Optional[int] = None,
        name: str = "SimpleMLP"
    ):
        """
        Initialize MLP.
        
        Args:
            hidden_layer_sizes: Number of neurons in each hidden layer.
            activation: Activation function.
            learning_rate: Learning rate.
            epochs: Number of training epochs.
            random_state: Random seed.
            name: Model name.
        """
        super().__init__(name=name)
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.learning_rate = learning_rate
        self.max_iter = epochs
        self.random_state = random_state
        
        # Internal model (sklearn MLPClassifier)
        from sklearn.neural_network import MLPClassifier
        self._model = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            learning_rate_init=learning_rate,
            max_iter=epochs,
            random_state=random_state
        )
    
    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series], **kwargs):
        """Fit the MLP."""
        self._validate_input(X, training=True)
        y = self._validate_targets(y, training=True)
        
        X = self._convert_to_array(X)
        
        self._model.fit(X, y)
        self.is_fitted_ = True
        
        return self
    
    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Predict probabilities."""
        self._validate_input(X, training=False)
        X = self._convert_to_array(X)
        
        return self._model.predict_proba(X)
    
    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get parameters."""
        return {
            'hidden_layer_sizes': self.hidden_layer_sizes,
            'activation': self.activation,
            'learning_rate': self.learning_rate,
            'epochs': self.max_iter,
            'random_state': self.random_state,
            'name': self.name,
        }
    
    def set_params(self, **params) -> 'SimpleMLPClassifier':
        """Set parameters."""
        for key, value in params.items():
            if key == 'epochs':
                self.max_iter = value
                # Update internal model's max_iter
                if hasattr(self, '_model'):
                    self._model.max_iter = value
            else:
                setattr(self, key, value)
                # Update internal model if it exists
                if hasattr(self, '_model') and hasattr(self._model, key):
                    setattr(self._model, key, value)
        return self
