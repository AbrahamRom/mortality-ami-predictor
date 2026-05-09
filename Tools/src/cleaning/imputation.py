"""Imputation strategies for missing values.

This module provides various strategies for imputing missing values
in numerical and categorical variables.
"""
from __future__ import annotations

from enum import Enum
from typing import List, Optional
import warnings

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer, SimpleImputer


class ImputationStrategy(str, Enum):
    """Available imputation strategies."""
    
    # No imputation (preserve original data)
    NONE = "none"
    
    # Numerical strategies
    MEAN = "mean"
    MEDIAN = "median"
    KNN = "knn"
    FORWARD = "forward"
    BACKWARD = "backward"
    CONSTANT_NUMERIC = "constant_numeric"
    
    # Categorical strategies
    MODE = "mode"
    CONSTANT_CATEGORICAL = "constant_categorical"


def impute_numeric_column(
    series: pd.Series,
    strategy: ImputationStrategy = ImputationStrategy.MEDIAN,
    fill_value: float = 0.0,
    knn_neighbors: int = 5,
) -> pd.Series:
    """Impute missing values in a numeric column.
    
    Args:
        series: Input series with missing values
        strategy: Imputation strategy to use
        fill_value: Value for constant imputation
        knn_neighbors: Number of neighbors for KNN imputation
        
    Returns:
        Series with imputed values
    """
    if series.isna().sum() == 0:
        return series
    
    # NONE strategy: return original series without any imputation
    if strategy == ImputationStrategy.NONE:
        return series
    
    result = series.copy()
    
    # Suppress numpy warnings for operations on empty slices
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning, 
                              message='.*empty slice.*')
        warnings.filterwarnings('ignore', category=RuntimeWarning,
                              message='.*Mean of empty slice.*')
        
        if strategy == ImputationStrategy.MEAN:
            result.fillna(series.mean(), inplace=True)
        
        elif strategy == ImputationStrategy.MEDIAN:
            result.fillna(series.median(), inplace=True)
        
        elif strategy == ImputationStrategy.FORWARD:
            result.fillna(method='ffill', inplace=True)
            # Fill remaining NaN at the start with median
            if result.isna().any():
                result.fillna(series.median(), inplace=True)
        
        elif strategy == ImputationStrategy.BACKWARD:
            result.fillna(method='bfill', inplace=True)
            # Fill remaining NaN at the end with median
            if result.isna().any():
                result.fillna(series.median(), inplace=True)
        
        elif strategy == ImputationStrategy.CONSTANT_NUMERIC:
            result.fillna(fill_value, inplace=True)
        
        else:
            # Default to median
            result.fillna(series.median(), inplace=True)
    
    return result


def impute_categorical_column(
    series: pd.Series,
    strategy: ImputationStrategy = ImputationStrategy.MODE,
    fill_value: str = "missing",
) -> pd.Series:
    """Impute missing values in a categorical column.
    
    Args:
        series: Input series with missing values
        strategy: Imputation strategy to use
        fill_value: Value for constant imputation
        
    Returns:
        Series with imputed values
    """
    if series.isna().sum() == 0:
        return series
    
    # NONE strategy: return original series without any imputation
    if strategy == ImputationStrategy.NONE:
        return series
    
    result = series.copy()
    
    if strategy == ImputationStrategy.MODE:
        mode_val = series.mode()
        fill = mode_val[0] if len(mode_val) > 0 else fill_value
        result.fillna(fill, inplace=True)
    
    elif strategy == ImputationStrategy.CONSTANT_CATEGORICAL:
        result.fillna(fill_value, inplace=True)
    
    elif strategy == ImputationStrategy.FORWARD:
        result.fillna(method='ffill', inplace=True)
        if result.isna().any():
            mode_val = series.mode()
            fill = mode_val[0] if len(mode_val) > 0 else fill_value
            result.fillna(fill, inplace=True)
    
    elif strategy == ImputationStrategy.BACKWARD:
        result.fillna(method='bfill', inplace=True)
        if result.isna().any():
            mode_val = series.mode()
            fill = mode_val[0] if len(mode_val) > 0 else fill_value
            result.fillna(fill, inplace=True)
    
    else:
        # Default to mode
        mode_val = series.mode()
        fill = mode_val[0] if len(mode_val) > 0 else fill_value
        result.fillna(fill, inplace=True)
    
    return result


def impute_knn(
    df: pd.DataFrame,
    columns: List[str],
    n_neighbors: int = 5,
) -> pd.DataFrame:
    """Impute missing values using K-Nearest Neighbors.
    
    Args:
        df: Input DataFrame
        columns: Columns to impute (must be numeric)
        n_neighbors: Number of neighbors for KNN
        
    Returns:
        DataFrame with imputed values
    """
    if not columns:
        return df
    
    df_result = df.copy()
    subset = df_result[columns]
    
    if subset.isna().any().any():
        imputer = KNNImputer(n_neighbors=n_neighbors)
        imputed_data = imputer.fit_transform(subset)
        df_result[columns] = imputed_data
    
    return df_result
