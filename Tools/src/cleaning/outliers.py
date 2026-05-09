"""Outlier detection and treatment strategies.

This module provides methods for detecting and treating outliers
in numerical data using statistical and machine learning methods.
"""
from __future__ import annotations

from enum import Enum
from typing import Tuple, Optional, List

import numpy as np
import pandas as pd
from scipy import stats


class OutlierMethod(str, Enum):
    """Methods for outlier detection."""
    
    IQR = "iqr"  # Interquartile Range
    ZSCORE = "zscore"  # Z-score
    MODIFIED_ZSCORE = "modified_zscore"  # Modified Z-score using MAD
    ISOLATION_FOREST = "isolation_forest"  # Isolation Forest
    LOF = "lof"  # Local Outlier Factor
    PERCENTILE = "percentile"  # Percentile-based
    NONE = "none"


class OutlierTreatment(str, Enum):
    """Strategies for treating detected outliers."""
    
    CAP = "cap"  # Cap at threshold (Winsorization)
    REMOVE = "remove"  # Remove outliers (mark as NaN)
    TRANSFORM = "transform"  # Apply log/sqrt transformation
    NONE = "none"  # Detect but don't treat


def detect_outliers_iqr(
    series: pd.Series,
    multiplier: float = 1.5,
) -> Tuple[pd.Series, float, float]:
    """Detect outliers using Interquartile Range method.
    
    Args:
        series: Input numeric series
        multiplier: IQR multiplier (typically 1.5 or 3.0)
        
    Returns:
        Tuple of (outlier_mask, lower_bound, upper_bound)
    """
    clean_data = series.dropna()
    
    if len(clean_data) == 0:
        return pd.Series(False, index=series.index), 0.0, 0.0
    
    Q1 = clean_data.quantile(0.25)
    Q3 = clean_data.quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    outlier_mask = (series < lower_bound) | (series > upper_bound)
    
    return outlier_mask, float(lower_bound), float(upper_bound)


def detect_outliers_zscore(
    series: pd.Series,
    threshold: float = 3.0,
) -> pd.Series:
    """Detect outliers using Z-score method.
    
    Args:
        series: Input numeric series
        threshold: Z-score threshold (typically 3.0)
        
    Returns:
        Boolean series indicating outliers
    """
    clean_data = series.dropna()
    
    if len(clean_data) == 0:
        return pd.Series(False, index=series.index)
    
    z_scores = np.abs(stats.zscore(clean_data, nan_policy='omit'))
    
    outlier_mask = pd.Series(False, index=series.index)
    outlier_mask.loc[clean_data.index] = z_scores > threshold
    
    return outlier_mask


def detect_outliers_modified_zscore(
    series: pd.Series,
    threshold: float = 3.5,
) -> Tuple[pd.Series, float, float]:
    """Detect outliers using Modified Z-score (MAD-based) method.
    
    More robust than standard Z-score as it uses Median Absolute Deviation
    instead of standard deviation, making it resistant to extreme values.
    
    Args:
        series: Input numeric series
        threshold: Modified Z-score threshold (typically 3.5)
        
    Returns:
        Tuple of (outlier_mask, lower_bound, upper_bound)
    """
    clean_data = series.dropna()
    
    if len(clean_data) == 0:
        return pd.Series(False, index=series.index), 0.0, 0.0
    
    median = clean_data.median()
    mad = np.median(np.abs(clean_data - median))
    
    # Avoid division by zero
    if mad == 0:
        return pd.Series(False, index=series.index), float(median), float(median)
    
    # Modified Z-score constant (0.6745 is the 0.75th quantile of standard normal)
    modified_z_scores = 0.6745 * (clean_data - median) / mad
    
    outlier_mask = pd.Series(False, index=series.index)
    outlier_mask.loc[clean_data.index] = np.abs(modified_z_scores) > threshold
    
    # Calculate bounds based on MAD
    lower_bound = float(median - threshold * mad / 0.6745)
    upper_bound = float(median + threshold * mad / 0.6745)
    
    return outlier_mask, lower_bound, upper_bound


def detect_outliers_isolation_forest(
    series: pd.Series,
    contamination: float = 0.1,
    random_state: int = 42,
) -> pd.Series:
    """Detect outliers using Isolation Forest algorithm.
    
    Isolation Forest is an unsupervised learning algorithm that isolates
    anomalies by randomly selecting features and split values. Outliers
    are isolated faster (shorter path) than normal points.
    
    Args:
        series: Input numeric series
        contamination: Expected proportion of outliers (0.0 to 0.5)
        random_state: Random seed for reproducibility
        
    Returns:
        Boolean series indicating outliers
    """
    try:
        from sklearn.ensemble import IsolationForest
    except ImportError:
        # Fallback to IQR if sklearn not available
        mask, _, _ = detect_outliers_iqr(series, 1.5)
        return mask
    
    clean_data = series.dropna()
    
    if len(clean_data) < 10:
        return pd.Series(False, index=series.index)
    
    # Reshape for sklearn
    X = clean_data.values.reshape(-1, 1)
    
    # Fit Isolation Forest
    iso_forest = IsolationForest(
        contamination=contamination,
        random_state=random_state,
        n_estimators=100,
    )
    predictions = iso_forest.fit_predict(X)
    
    outlier_mask = pd.Series(False, index=series.index)
    outlier_mask.loc[clean_data.index] = predictions == -1
    
    return outlier_mask


def detect_outliers_lof(
    series: pd.Series,
    n_neighbors: int = 20,
    contamination: float = 0.1,
) -> pd.Series:
    """Detect outliers using Local Outlier Factor (LOF) algorithm.
    
    LOF measures the local deviation of density of a given sample with
    respect to its neighbors. It is based on the concept of local density.
    
    Args:
        series: Input numeric series
        n_neighbors: Number of neighbors for LOF calculation
        contamination: Expected proportion of outliers
        
    Returns:
        Boolean series indicating outliers
    """
    try:
        from sklearn.neighbors import LocalOutlierFactor
    except ImportError:
        # Fallback to IQR if sklearn not available
        mask, _, _ = detect_outliers_iqr(series, 1.5)
        return mask
    
    clean_data = series.dropna()
    
    if len(clean_data) < n_neighbors + 1:
        return pd.Series(False, index=series.index)
    
    # Reshape for sklearn
    X = clean_data.values.reshape(-1, 1)
    
    # Adjust n_neighbors if necessary
    effective_neighbors = min(n_neighbors, len(clean_data) - 1)
    
    # Fit LOF
    lof = LocalOutlierFactor(
        n_neighbors=effective_neighbors,
        contamination=contamination,
    )
    predictions = lof.fit_predict(X)
    
    outlier_mask = pd.Series(False, index=series.index)
    outlier_mask.loc[clean_data.index] = predictions == -1
    
    return outlier_mask


def detect_outliers_percentile(
    series: pd.Series,
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.0,
) -> Tuple[pd.Series, float, float]:
    """Detect outliers using percentile-based method.
    
    Simple but effective method that marks values outside specified
    percentile range as outliers. Common choices are 1-99 or 5-95.
    
    Args:
        series: Input numeric series
        lower_percentile: Lower percentile threshold (0-100)
        upper_percentile: Upper percentile threshold (0-100)
        
    Returns:
        Tuple of (outlier_mask, lower_bound, upper_bound)
    """
    clean_data = series.dropna()
    
    if len(clean_data) == 0:
        return pd.Series(False, index=series.index), 0.0, 0.0
    
    lower_bound = float(np.percentile(clean_data, lower_percentile))
    upper_bound = float(np.percentile(clean_data, upper_percentile))
    
    outlier_mask = (series < lower_bound) | (series > upper_bound)
    
    return outlier_mask, lower_bound, upper_bound


def treat_outliers(
    series: pd.Series,
    outlier_mask: pd.Series,
    treatment: OutlierTreatment,
    lower_bound: Optional[float] = None,
    upper_bound: Optional[float] = None,
) -> pd.Series:
    """Treat detected outliers using specified strategy.
    
    Args:
        series: Input series
        outlier_mask: Boolean series indicating outliers
        treatment: Treatment strategy
        lower_bound: Lower bound for capping
        upper_bound: Upper bound for capping
        
    Returns:
        Series with treated outliers
    """
    if treatment == OutlierTreatment.NONE:
        return series
    
    result = series.copy()
    
    if treatment == OutlierTreatment.CAP:
        if lower_bound is not None:
            result.loc[outlier_mask & (result < lower_bound)] = lower_bound
        if upper_bound is not None:
            result.loc[outlier_mask & (result > upper_bound)] = upper_bound
    
    elif treatment == OutlierTreatment.REMOVE:
        result.loc[outlier_mask] = np.nan
    
    elif treatment == OutlierTreatment.TRANSFORM:
        # Apply log1p transformation to reduce outlier impact
        # Only for positive values; use sqrt for data with zeros
        if (result > 0).all():
            result = np.log1p(result)
        else:
            # For data with zeros or negatives, shift and apply sqrt
            min_val = result.min()
            if min_val <= 0:
                result = np.sqrt(result - min_val + 1)
            else:
                result = np.sqrt(result)
    
    return result


def handle_column_outliers(
    series: pd.Series,
    method: OutlierMethod = OutlierMethod.IQR,
    treatment: OutlierTreatment = OutlierTreatment.CAP,
    iqr_multiplier: float = 1.5,
    zscore_threshold: float = 3.0,
    modified_zscore_threshold: float = 3.5,
    contamination: float = 0.1,
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.0,
    lof_neighbors: int = 20,
) -> Tuple[pd.Series, int]:
    """Detect and treat outliers in a numeric column.
    
    Args:
        series: Input numeric series
        method: Detection method
        treatment: Treatment strategy
        iqr_multiplier: Multiplier for IQR method
        zscore_threshold: Threshold for Z-score method
        modified_zscore_threshold: Threshold for Modified Z-score method
        contamination: Contamination rate for ML methods (Isolation Forest, LOF)
        lower_percentile: Lower percentile for percentile method
        upper_percentile: Upper percentile for percentile method
        lof_neighbors: Number of neighbors for LOF method
        
    Returns:
        Tuple of (treated_series, outlier_count)
    """
    if method == OutlierMethod.NONE:
        return series, 0
    
    lower, upper = None, None
    
    # Detect outliers based on method
    if method == OutlierMethod.IQR:
        outlier_mask, lower, upper = detect_outliers_iqr(series, iqr_multiplier)
    
    elif method == OutlierMethod.ZSCORE:
        outlier_mask = detect_outliers_zscore(series, zscore_threshold)
        # For z-score, use percentiles as bounds
        lower = series.quantile(0.01) if treatment == OutlierTreatment.CAP else None
        upper = series.quantile(0.99) if treatment == OutlierTreatment.CAP else None
    
    elif method == OutlierMethod.MODIFIED_ZSCORE:
        outlier_mask, lower, upper = detect_outliers_modified_zscore(
            series, modified_zscore_threshold
        )
    
    elif method == OutlierMethod.ISOLATION_FOREST:
        outlier_mask = detect_outliers_isolation_forest(series, contamination)
        # Use percentile bounds for capping
        if treatment == OutlierTreatment.CAP:
            lower = float(series.quantile(0.01))
            upper = float(series.quantile(0.99))
    
    elif method == OutlierMethod.LOF:
        outlier_mask = detect_outliers_lof(series, lof_neighbors, contamination)
        # Use percentile bounds for capping
        if treatment == OutlierTreatment.CAP:
            lower = float(series.quantile(0.01))
            upper = float(series.quantile(0.99))
    
    elif method == OutlierMethod.PERCENTILE:
        outlier_mask, lower, upper = detect_outliers_percentile(
            series, lower_percentile, upper_percentile
        )
    
    else:
        return series, 0
    
    # Apply treatment
    result = treat_outliers(series, outlier_mask, treatment, lower, upper)
    outlier_count = int(outlier_mask.sum())
    
    return result, outlier_count
