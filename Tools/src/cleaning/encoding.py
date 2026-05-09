"""Categorical encoding strategies.

This module provides various encoding strategies for categorical variables
including one-hot, label, and ordinal encoding.
"""
from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
import warnings

import pandas as pd
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder


class EncodingStrategy(str, Enum):
    """Available encoding strategies for categorical variables."""
    
    ONEHOT = "onehot"
    LABEL = "label"
    ORDINAL = "ordinal"
    NONE = "none"


def encode_label(
    series: pd.Series,
) -> Tuple[pd.Series, Dict[str, int]]:
    """Encode categorical variable using label encoding.
    
    Args:
        series: Input categorical series
        
    Returns:
        Tuple of (encoded_series, encoding_mapping)
    """
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        
        series_str = series.astype(str)
        
        encoder = LabelEncoder()
        encoded = encoder.fit_transform(series_str)
        
        mapping = dict(zip(
            encoder.classes_,
            encoder.transform(encoder.classes_)
        ))
        
        return pd.Series(encoded, index=series.index, name=series.name), mapping


def encode_ordinal(
    series: pd.Series,
    categories: Optional[List[str]] = None,
) -> Tuple[pd.Series, Dict[str, int]]:
    """Encode categorical variable using ordinal encoding.
    
    Args:
        series: Input categorical series
        categories: Ordered list of categories. If None, uses natural order
        
    Returns:
        Tuple of (encoded_series, encoding_mapping)
    """
    series_str = series.astype(str)
    
    if categories is None:
        # Use alphabetical order
        categories = sorted(series_str.dropna().unique())
    
    encoder = OrdinalEncoder(
        categories=[categories],
        handle_unknown='use_encoded_value',
        unknown_value=-1
    )
    
    encoded = encoder.fit_transform(series_str.values.reshape(-1, 1)).flatten()
    
    mapping = dict(zip(categories, range(len(categories))))
    
    return pd.Series(encoded, index=series.index, name=series.name), mapping


def encode_onehot(
    df: pd.DataFrame,
    column: str,
    drop_first: bool = False,
) -> Tuple[pd.DataFrame, List[str]]:
    """Encode categorical variable using one-hot encoding.
    
    Args:
        df: Input DataFrame
        column: Column name to encode
        drop_first: Whether to drop first dummy to avoid collinearity
        
    Returns:
        Tuple of (DataFrame with encoded columns, list of new column names)
    """
    dummies = pd.get_dummies(
        df[column],
        prefix=column,
        drop_first=drop_first,
        dtype=int
    )
    
    new_columns = dummies.columns.tolist()
    
    result_df = pd.concat(
        [df.drop(columns=[column]), dummies],
        axis=1
    )
    
    return result_df, new_columns


def encode_categorical_column(
    df: pd.DataFrame,
    column: str,
    strategy: EncodingStrategy,
    ordinal_categories: Optional[List[str]] = None,
    drop_first: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Encode a categorical column using specified strategy.
    
    Args:
        df: Input DataFrame
        column: Column name to encode
        strategy: Encoding strategy to use
        ordinal_categories: Ordered categories for ordinal encoding
        drop_first: Whether to drop first dummy in one-hot encoding
        
    Returns:
        Tuple of (DataFrame with encoded column, encoding_info)
        encoding_info contains: {'type': strategy, 'mapping': ..., 'columns': ...}
    """
    if strategy == EncodingStrategy.NONE:
        return df, {'type': 'none'}
    
    result_df = df.copy()
    info: Dict[str, Any] = {'type': strategy.value}
    
    if strategy == EncodingStrategy.LABEL:
        encoded, mapping = encode_label(df[column])
        result_df[column] = encoded
        info['mapping'] = mapping
    
    elif strategy == EncodingStrategy.ORDINAL:
        encoded, mapping = encode_ordinal(df[column], ordinal_categories)
        result_df[column] = encoded
        info['mapping'] = mapping
    
    elif strategy == EncodingStrategy.ONEHOT:
        result_df, new_cols = encode_onehot(df, column, drop_first)
        info['columns'] = new_cols
        info['mapping'] = {col: 1 for col in new_cols}
    
    return result_df, info
