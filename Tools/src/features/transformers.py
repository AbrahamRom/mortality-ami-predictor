"""Custom feature transformers for sklearn pipelines."""
from __future__ import annotations

from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd


class FeatureTransformer(BaseEstimator, TransformerMixin):
    """Custom feature transformer for sklearn pipelines.
    
    This is a template for creating custom transformers.
    """
    
    def __init__(self):
        """Initialize transformer."""
        pass
    
    def fit(self, X, y=None):
        """Fit transformer to data.
        
        Args:
            X: Features
            y: Target (optional)
            
        Returns:
            self
        """
        return self
    
    def transform(self, X):
        """Transform features.
        
        Args:
            X: Features to transform
            
        Returns:
            Transformed features
        """
        return X
