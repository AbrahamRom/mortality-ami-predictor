"""Train/test splitting strategies.

This module provides various strategies for splitting datasets into
training and testing sets, including temporal and stratified splits.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split as sk_split

from ..config import RANDOM_SEED


def train_test_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    time_column: Optional[str] = None,
    stratify_column: Optional[str] = None,
    random_state: int = RANDOM_SEED,
    return_indices: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame] | Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """Split dataset into train and test sets with multiple strategies.
    
    Supports:
    - Random split (default)
    - Temporal split (when time_column is provided)
    - Stratified split (when stratify_column is provided)
    
    Args:
        df: Input DataFrame
        test_size: Fraction of data to use for testing (0.0 to 1.0)
        time_column: Column name for temporal ordering. If provided,
                    performs temporal split (most recent data becomes test)
        stratify_column: Column name for stratification. Ensures balanced
                        class distribution in train/test splits
        random_state: Random seed for reproducibility
        return_indices: If True, also return original DataFrame indices.
                       Useful for preserving original data for clinical scores.
        
    Returns:
        If return_indices=False: Tuple of (train_df, test_df)
        If return_indices=True: Tuple of (train_df, test_df, train_indices, test_indices)
    """
    # Temporal split takes precedence
    if time_column and time_column in df.columns:
        train_df, test_df = create_temporal_split(df, time_column, test_size)
        if return_indices:
            return train_df, test_df, train_df.index.values, test_df.index.values
        return train_df, test_df
    
    # Stratified split
    if stratify_column and stratify_column in df.columns:
        return create_stratified_split(
            df, stratify_column, test_size, random_state, return_indices=return_indices
        )
    
    # Default random split
    train_df, test_df = sk_split(
        df,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
    )
    
    if return_indices:
        return train_df, test_df, train_df.index.values, test_df.index.values
    
    return train_df, test_df


def create_temporal_split(
    df: pd.DataFrame,
    time_column: str,
    test_size: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split dataset temporally based on time column.
    
    Most recent data (by time_column) becomes the test set.
    Useful for time series or temporal data to avoid data leakage.
    
    Args:
        df: Input DataFrame
        time_column: Column name containing temporal information
        test_size: Fraction of data to use for testing
        
    Returns:
        Tuple of (train_df, test_df)
    """
    if time_column not in df.columns:
        raise KeyError(f"Time column '{time_column}' not found in DataFrame")
    
    # Sort by time
    df_sorted = df.sort_values(time_column)
    
    # Calculate split point
    n_test = int(np.floor(len(df_sorted) * test_size))
    
    # Split: most recent data is test
    test_df = df_sorted.iloc[-n_test:].copy()
    train_df = df_sorted.iloc[:-n_test].copy()
    
    return train_df, test_df


def create_stratified_split(
    df: pd.DataFrame,
    stratify_column: str,
    test_size: float = 0.2,
    random_state: int = RANDOM_SEED,
    return_indices: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame] | Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """Split dataset with stratification on specified column.
    
    Ensures balanced distribution of stratify_column values in both
    train and test sets. Useful for imbalanced classification problems.
    
    Args:
        df: Input DataFrame
        stratify_column: Column name to stratify on
        test_size: Fraction of data to use for testing
        random_state: Random seed for reproducibility
        return_indices: If True, also return original DataFrame indices
        
    Returns:
        If return_indices=False: Tuple of (train_df, test_df)
        If return_indices=True: Tuple of (train_df, test_df, train_indices, test_indices)
    """
    if stratify_column not in df.columns:
        raise KeyError(f"Stratify column '{stratify_column}' not found in DataFrame")
    
    # Store original indices before split
    original_indices = df.index.values
    
    train_df, test_df = sk_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[stratify_column],
        shuffle=True,
    )
    
    if return_indices:
        return train_df, test_df, train_df.index.values, test_df.index.values
    
    return train_df, test_df
