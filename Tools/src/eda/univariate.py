"""Univariate analysis utilities."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import pandas as pd


@dataclass
class UnivariateStats:
    """Univariate statistics for a variable."""
    
    variable_name: str
    variable_type: str  # 'numerical' or 'categorical'
    
    # For numerical variables
    count: Optional[int] = None
    mean: Optional[float] = None
    median: Optional[float] = None
    std: Optional[float] = None
    min: Optional[float] = None
    max: Optional[float] = None
    q25: Optional[float] = None
    q75: Optional[float] = None
    skewness: Optional[float] = None
    kurtosis: Optional[float] = None
    
    # For categorical variables
    n_categories: Optional[int] = None
    mode: Optional[str] = None
    mode_frequency: Optional[int] = None
    category_counts: Dict[str, int] = field(default_factory=dict)
    
    # General
    missing_count: int = 0
    missing_percent: float = 0.0


def compute_numeric_stats(df: pd.DataFrame, col: str) -> UnivariateStats:
    """Compute statistics for numerical variable.
    
    Args:
        df: DataFrame
        col: Column name
        
    Returns:
        UnivariateStats object
    """
    series = df[col].dropna()
    
    stats_obj = UnivariateStats(
        variable_name=col,
        variable_type='numerical',
        count=len(series),
        mean=float(series.mean()) if len(series) > 0 else None,
        median=float(series.median()) if len(series) > 0 else None,
        std=float(series.std()) if len(series) > 0 else None,
        min=float(series.min()) if len(series) > 0 else None,
        max=float(series.max()) if len(series) > 0 else None,
        q25=float(series.quantile(0.25)) if len(series) > 0 else None,
        q75=float(series.quantile(0.75)) if len(series) > 0 else None,
        missing_count=int(df[col].isna().sum()),
        missing_percent=float(df[col].isna().mean() * 100),
    )
    
    # Skewness and kurtosis
    try:
        stats_obj.skewness = float(series.skew())
        stats_obj.kurtosis = float(series.kurtosis())
    except Exception:
        pass
    
    return stats_obj


def compute_categorical_stats(df: pd.DataFrame, col: str) -> UnivariateStats:
    """Compute statistics for categorical variable.
    
    Args:
        df: DataFrame
        col: Column name
        
    Returns:
        UnivariateStats object
    """
    series = df[col].dropna()
    value_counts = series.value_counts()
    
    stats_obj = UnivariateStats(
        variable_name=col,
        variable_type='categorical',
        count=len(series),
        n_categories=len(value_counts),
        mode=str(value_counts.index[0]) if len(value_counts) > 0 else None,
        mode_frequency=int(value_counts.iloc[0]) if len(value_counts) > 0 else None,
        category_counts={str(k): int(v) for k, v in value_counts.items()},
        missing_count=int(df[col].isna().sum()),
        missing_percent=float(df[col].isna().mean() * 100),
    )
    
    return stats_obj
