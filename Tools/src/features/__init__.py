"""Feature engineering and selection module.

This module provides utilities for feature selection, custom transformers,
and dimensionality reduction (PCA, ICA).
"""

from .selectors import safe_feature_columns, select_important_features
from .transformers import FeatureTransformer
from .ica import ICATransformer, ICAResult, compare_pca_vs_ica

__all__ = [
    "safe_feature_columns",
    "select_important_features",
    "FeatureTransformer",
    "ICATransformer",
    "ICAResult",
    "compare_pca_vs_ica",
]
