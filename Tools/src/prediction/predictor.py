"""Predictor class for making predictions."""
from __future__ import annotations

import os
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd
import joblib


MODELS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "models"
)


class Predictor:
    """Predictor wrapper for loaded models."""
    
    def __init__(self, model, model_name: str = "model"):
        """Initialize predictor.
        
        Args:
            model: Trained model with predict/predict_proba methods
            model_name: Name of the model
        """
        self.model = model
        self.model_name = model_name
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Make binary predictions.
        
        Args:
            X: Feature matrix
            
        Returns:
            Binary predictions
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        return self.model.predict(X)
    
    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predict probabilities.
        
        Args:
            X: Feature matrix
            
        Returns:
            Probability predictions
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)[:, 1]
        else:
            # Fallback for models without predict_proba
            return self.model.predict(X)
    
    def predict_with_metadata(
        self,
        X: Union[pd.DataFrame, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """Make predictions with additional metadata.
        
        Args:
            X: Feature matrix
            
        Returns:
            Dictionary with predictions, probabilities, and metadata
        """
        predictions = self.predict(X)
        probabilities = self.predict_proba(X)
        
        return {
            "predictions": predictions,
            "probabilities": probabilities,
            "model_name": self.model_name,
            "n_samples": len(X),
        }


def load_predictor(model_path: Optional[str] = None, model_name: str = "best_classifier_mortality") -> Predictor:
    """Load a trained model as a predictor.
    
    Args:
        model_path: Path to model file (if None, looks in MODELS_DIR)
        model_name: Name of model file (without .joblib extension)
        
    Returns:
        Predictor instance
        
    Raises:
        FileNotFoundError: If model file not found
    """
    if model_path is None:
        model_path = os.path.join(MODELS_DIR, f"{model_name}.joblib")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    model = joblib.load(model_path)
    return Predictor(model, model_name=model_name)


def predict_mortality(
    X: Union[pd.DataFrame, np.ndarray],
    model_path: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    """Quick function to predict mortality using the best model.
    
    Args:
        X: Feature matrix
        model_path: Optional path to model (uses best_classifier_mortality if None)
        
    Returns:
        Dictionary with predictions and probabilities
    """
    predictor = load_predictor(model_path=model_path)
    return predictor.predict_with_metadata(X)
