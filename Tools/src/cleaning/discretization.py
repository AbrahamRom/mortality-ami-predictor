"""Discretization strategies for numeric variables.

This module provides various strategies for discretizing continuous
numeric variables into categorical bins.
"""
from __future__ import annotations

from enum import Enum
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer


class DiscretizationStrategy(str, Enum):
    """Available discretization strategies."""
    
    NONE = "none"
    UNIFORM = "uniform"  # Equal width bins
    QUANTILE = "quantile"  # Equal frequency bins
    KMEANS = "kmeans"  # K-means clustering
    CUSTOM = "custom"  # Custom bin edges


def discretize_column(
    series: pd.Series,
    strategy: DiscretizationStrategy = DiscretizationStrategy.UNIFORM,
    n_bins: int = 5,
    custom_bins: Optional[List[float]] = None,
    custom_labels: Optional[List[str]] = None,
    return_bins: bool = False,
) -> Union[pd.Series, Tuple[pd.Series, List[float], List[str]]]:
    """Discretize a numeric column into categorical bins.
    
    Args:
        series: Input numeric series
        strategy: Discretization strategy to use
        n_bins: Number of bins (for uniform, quantile, kmeans)
        custom_bins: Custom bin edges (for custom strategy)
        custom_labels: Custom labels for bins
        return_bins: If True, return tuple with (series, bins, labels)
        
    Returns:
        Discretized series, or tuple with (series, bins, labels) if return_bins=True
    """
    if strategy == DiscretizationStrategy.NONE:
        if return_bins:
            return series, None, None
        return series
    
    # Remove missing values for discretization
    valid_mask = series.notna()
    if valid_mask.sum() == 0:
        if return_bins:
            return series, None, None
        return series
    
    result = series.copy()
    
    if strategy == DiscretizationStrategy.CUSTOM:
        if custom_bins is None:
            raise ValueError("custom_bins must be provided for custom strategy")
        
        bins = custom_bins
        if custom_labels is not None:
            labels = custom_labels
        else:
            labels = [f"bin_{i}" for i in range(len(bins) - 1)]
        
        # Use pd.cut for custom bins
        result[valid_mask] = pd.cut(
            series[valid_mask],
            bins=bins,
            labels=labels,
            include_lowest=True,
            duplicates='drop'
        )
        
    elif strategy in [DiscretizationStrategy.UNIFORM, 
                      DiscretizationStrategy.QUANTILE, 
                      DiscretizationStrategy.KMEANS]:
        
        # Map strategy to sklearn strategy
        sklearn_strategy_map = {
            DiscretizationStrategy.UNIFORM: 'uniform',
            DiscretizationStrategy.QUANTILE: 'quantile',
            DiscretizationStrategy.KMEANS: 'kmeans',
        }
        sklearn_strategy = sklearn_strategy_map[strategy]
        
        # Use sklearn KBinsDiscretizer
        discretizer = KBinsDiscretizer(
            n_bins=n_bins,
            encode='ordinal',
            strategy=sklearn_strategy,
            subsample=None
        )
        
        # Fit and transform
        valid_data = series[valid_mask].values.reshape(-1, 1)
        discretized = discretizer.fit_transform(valid_data).flatten()
        
        # Get bin edges
        bins = discretizer.bin_edges_[0].tolist()
        
        # Create labels
        if custom_labels is not None and len(custom_labels) == n_bins:
            labels = custom_labels
        else:
            labels = []
            for i in range(n_bins):
                if i == 0:
                    labels.append(f"[{bins[i]:.2f}, {bins[i+1]:.2f}]")
                else:
                    labels.append(f"({bins[i]:.2f}, {bins[i+1]:.2f}]")
        
        # Map ordinal values to labels
        label_map = {i: labels[i] for i in range(n_bins)}
        result[valid_mask] = pd.Series(discretized, index=series[valid_mask].index).map(label_map)
        
    else:
        raise ValueError(f"Unknown discretization strategy: {strategy}")
    
    if return_bins:
        return result, bins, labels
    
    return result


def get_discretization_info(
    series: pd.Series,
    bins: Optional[List[float]] = None,
    labels: Optional[List[str]] = None,
) -> dict:
    """Get information about discretization results.
    
    Args:
        series: Discretized series
        bins: Bin edges used
        labels: Labels used
        
    Returns:
        Dictionary with discretization information
    """
    value_counts = series.value_counts().to_dict()
    
    return {
        'n_bins': len(value_counts),
        'bin_edges': bins,
        'bin_labels': labels,
        'value_counts': value_counts,
        'missing_count': series.isna().sum(),
    }
