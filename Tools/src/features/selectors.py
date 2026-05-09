"""Feature selection utilities.

This module provides functions for selecting and filtering features.
"""
from __future__ import annotations

from typing import List, Set

import pandas as pd


# Columns to exclude from features
EXCLUDE_COLS: Set[str] = {"patient_id", "mrn", "admission_id"}


def safe_feature_columns(
    df: pd.DataFrame,
    target_cols: List[str],
) -> List[str]:
    """Return feature columns excluding targets and identifiers.
    
    Args:
        df: Input DataFrame
        target_cols: List of target column names to exclude
        
    Returns:
        List of feature column names
    """
    exclude_set = set(target_cols) | EXCLUDE_COLS
    feature_cols = [c for c in df.columns if c not in exclude_set]
    return feature_cols


def select_important_features(
    df: pd.DataFrame,
    importance_threshold: float = 0.01,
) -> List[str]:
    """Select features above importance threshold.
    
    Args:
        df: Input DataFrame
        importance_threshold: Minimum importance score
        
    Returns:
        List of selected feature names
    """
    # Placeholder - would use actual feature importance from model
    return list(df.columns)
