"""Sklearn preprocessing pipelines for ML models.

This module provides ColumnTransformer-based preprocessing that can
work with both raw and cleaned data.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingRegressor


@dataclass
class PreprocessingConfig:
    """Configuration for preprocessing pipeline.
    
    Attributes:
        imputer_mode: Imputation strategy ('iterative', 'knn', 'simple')
        scale_numeric: Whether to standardize numeric features
        categorical_encoding: Encoding strategy ('onehot')
        feature_selection: Whether to apply feature selection
        selection_estimator: Estimator for feature selection
        drop_fully_missing: Drop fully missing columns
        drop_constant: Drop constant columns
        
        # Imbalance handling (added for mortality prediction)
        imbalance_strategy: Strategy for handling class imbalance
            Options: 'none', 'smote', 'smote_nc', 'adasyn', 'borderline_smote',
                    'svm_smote', 'smote_tomek', 'smote_enn', 'class_weight'
        imbalance_sampling_strategy: Target ratio for resampling ('auto' or float)
        imbalance_k_neighbors: Number of neighbors for SMOTE variants (default: 5)
    """
    imputer_mode: str = "iterative"
    scale_numeric: bool = True
    categorical_encoding: str = "onehot"
    feature_selection: bool = False
    selection_estimator: Optional[object] = None
    drop_fully_missing: bool = True
    drop_constant: bool = True
    
    # Imbalance handling options
    imbalance_strategy: str = "smote"  # Default to SMOTE for medical data
    imbalance_sampling_strategy: str = "auto"
    imbalance_k_neighbors: int = 5


def build_preprocessing_pipeline(
    X: pd.DataFrame,
    config: Optional[PreprocessingConfig] = None,
) -> Tuple[Pipeline, List[str]]:
    """Build a ColumnTransformer-based preprocessing pipeline.
    
    This pipeline is designed to work with sklearn models and includes:
    - Missing value imputation
    - Feature scaling
    - Categorical encoding
    - Optional feature selection
    
    Args:
        X: Input feature DataFrame (used to infer column types)
        config: Preprocessing configuration
        
    Returns:
        Tuple of (pipeline, approximate_feature_names)
    """
    if config is None:
        config = PreprocessingConfig()
    
    # Identify numeric and categorical columns
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
    
    # Drop problematic columns
    drop_cols: List[str] = []
    
    if config.drop_fully_missing:
        drop_cols.extend([c for c in X.columns if X[c].isna().all()])
    
    if config.drop_constant:
        drop_cols.extend([c for c in X.columns if X[c].nunique(dropna=True) <= 1])
    
    if drop_cols:
        drop_cols = sorted(set(drop_cols))
        numeric_cols = [c for c in numeric_cols if c not in drop_cols]
        categorical_cols = [c for c in categorical_cols if c not in drop_cols]
    
    # Build numeric pipeline
    if config.imputer_mode == "iterative":
        # Use HistGradientBoostingRegressor (handles NaN natively)
        num_imputer = IterativeImputer(
            estimator=HistGradientBoostingRegressor(random_state=42),
            random_state=42,
            sample_posterior=False,
            max_iter=10,
            initial_strategy="median",
        )
    elif config.imputer_mode == "knn":
        num_imputer = KNNImputer(n_neighbors=5)
    elif config.imputer_mode == "simple":
        num_imputer = SimpleImputer(strategy="median")
    else:
        raise ValueError(f"Invalid imputer_mode: {config.imputer_mode}")
    
    numeric_steps = [("imputer", num_imputer)]
    
    if config.scale_numeric and numeric_cols:
        numeric_steps.append(("scaler", StandardScaler(with_mean=True, with_std=True)))
    
    numeric_pipeline = Pipeline(steps=numeric_steps)
    
    # Build categorical pipeline
    if config.categorical_encoding == "onehot":
        categorical_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="constant",
                        fill_value="missing",
                        keep_empty_features=True
                    ),
                ),
                (
                    "encoder",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False)
                ),
            ]
        )
    else:
        raise ValueError(f"Unsupported encoding: {config.categorical_encoding}")
    
    # Combine with ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_cols),
            ("categorical", categorical_pipeline, categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    
    # Build full pipeline
    pipeline_steps = [("preprocessor", preprocessor)]
    
    # Optional feature selection
    if config.feature_selection:
        estimator = config.selection_estimator or RandomForestClassifier(
            n_estimators=200,
            random_state=42
        )
        pipeline_steps.append(
            ("feature_selection", SelectFromModel(estimator=estimator, threshold="median"))
        )
    
    pipeline = Pipeline(steps=pipeline_steps)
    
    # Approximate feature names (actual names determined after fitting)
    feature_names = numeric_cols + categorical_cols
    
    return pipeline, feature_names
