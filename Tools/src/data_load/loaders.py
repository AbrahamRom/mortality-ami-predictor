"""Dataset loading utilities.

This module handles loading datasets from various file formats with
robust error handling and format detection.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional

import pandas as pd

from ..config import CONFIG, ProjectConfig


@dataclass
class DatasetInfo:
    """Container with dataset metadata and information.
    
    Attributes:
        columns: List of column names
        n_rows: Number of rows
        target_col: Target column name if available
        arrhythmia_col: Arrhythmia column name if available
        regression_target: Regression target column name if available
    """
    columns: List[str]
    n_rows: int
    target_col: Optional[str]
    arrhythmia_col: Optional[str]
    regression_target: Optional[str]


SUPPORTED_FORMATS = {".csv", ".parquet", ".feather", ".xlsx", ".xls"}


def _detect_excel_format(path: str) -> str:
    """Detect Excel format by examining file magic bytes.
    
    Args:
        path: Path to Excel file
        
    Returns:
        Format string: 'xlsx', 'xls', or 'unknown'
    """
    try:
        with open(path, "rb") as f:
            header = f.read(4)
        
        # XLSX (ZIP-based format)
        if header.startswith(b"PK\x03\x04"):
            return "xlsx"
        
        # XLS (Compound File Binary Format)
        if header.startswith(b"\xD0\xCF\x11\xE0"):
            return "xls"
    except Exception:
        pass
    
    return "unknown"


def load_dataset(path: Optional[str] = None) -> pd.DataFrame:
    """Load dataset from file path with format auto-detection.
    
    Supports CSV, Parquet, Feather, and Excel formats (.xlsx, .xls).
    Provides helpful error messages for missing dependencies.
    
    Args:
        path: Path to dataset file. If None, uses CONFIG.dataset_path
        
    Returns:
        pandas DataFrame with loaded data
        
    Raises:
        FileNotFoundError: If file does not exist
        ValueError: If file format is not supported
        ImportError: If required library for format is missing
        RuntimeError: If file is corrupted or cannot be read
    """
    # Use config path if not provided
    if path is None:
        from ..config import validate_config
        validate_config(CONFIG)
        path = CONFIG.dataset_path
    
    # Validate file exists
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")
    
    # Detect file extension
    ext = os.path.splitext(path)[1].lower()
    
    # Validate supported format
    if ext not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported file format '{ext}'. "
            f"Supported formats: {sorted(SUPPORTED_FORMATS)}"
        )
    
    # Load based on format
    try:
        if ext == ".csv":
            # Try multiple encodings for CSV files (common in Spanish datasets)
            encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'windows-1252']
            df = None
            last_error = None
            
            for encoding in encodings:
                try:
                    df = pd.read_csv(path, encoding=encoding)
                    break
                except (UnicodeDecodeError, LookupError) as e:
                    last_error = e
                    continue
            
            if df is None:
                raise RuntimeError(
                    f"Failed to decode CSV with any encoding. Last error: {last_error}"
                )
        
        elif ext == ".parquet":
            df = pd.read_parquet(path)
        
        elif ext == ".feather":
            df = pd.read_feather(path)
        
        elif ext in {".xlsx", ".xls"}:
            # Detect actual format
            excel_format = _detect_excel_format(path)
            
            try:
                if ext == ".xlsx" or excel_format == "xlsx":
                    df = pd.read_excel(path, engine="openpyxl")
                elif ext == ".xls" or excel_format == "xls":
                    df = pd.read_excel(path, engine="xlrd")
                else:
                    # Unknown format - try openpyxl first, then xlrd
                    try:
                        df = pd.read_excel(path, engine="openpyxl")
                    except Exception:
                        df = pd.read_excel(path, engine="xlrd")
            
            except ImportError as e:
                if ext == ".xlsx" or excel_format == "xlsx":
                    raise ImportError(
                        "To read .xlsx files, install openpyxl: "
                        "pip install openpyxl"
                    ) from e
                else:
                    raise ImportError(
                        "To read .xls files, install xlrd: "
                        "pip install xlrd==1.2.0"
                    ) from e
            
            except Exception as e:
                raise RuntimeError(
                    f"Failed to read Excel file. Possible causes: "
                    f"corrupted file or incorrect extension. "
                    f"Try re-saving as .xlsx or exporting to CSV. "
                    f"Error: {type(e).__name__}: {e}"
                ) from e
        
        else:
            raise ValueError(f"Unhandled extension: {ext}")
    
    except (pd.errors.ParserError, pd.errors.EmptyDataError) as e:
        raise RuntimeError(
            f"Failed to parse {ext} file. "
            f"Check file format and encoding. Error: {e}"
        ) from e
    
    return df


def get_dataset_info(df: pd.DataFrame, cfg: ProjectConfig = CONFIG) -> DatasetInfo:
    """Extract metadata information from DataFrame.
    
    Args:
        df: Input DataFrame
        cfg: Project configuration with target column names
        
    Returns:
        DatasetInfo object with metadata
    """
    return DatasetInfo(
        columns=list(df.columns),
        n_rows=len(df),
        target_col=cfg.target_column if cfg.target_column in df.columns else None,
        arrhythmia_col=(
            cfg.arrhythmia_column 
            if cfg.arrhythmia_column and cfg.arrhythmia_column in df.columns 
            else None
        ),
        regression_target=(
            cfg.regression_target 
            if cfg.regression_target and cfg.regression_target in df.columns 
            else None
        ),
    )


def summarize_dataframe(df: pd.DataFrame, cfg: ProjectConfig = CONFIG) -> dict:
    """Compute summary tables for EDA with compatibility across pandas versions.
    
    Args:
        df: Input DataFrame
        cfg: Project configuration
        
    Returns:
        Dictionary with summary DataFrames:
        - describe: Descriptive statistics
        - missing: Missing value rates
        - dtypes: Data types
        - class_balance_*: Class distributions for target columns
    """
    import numpy as np
    
    # Descriptive statistics with pandas version compatibility
    try:
        desc = df.describe(include="all", datetime_is_numeric=True).T
    except TypeError:
        # Fallback for older pandas without datetime_is_numeric
        desc = df.describe(include="all").T
    
    summary = {
        "describe": desc,
        "missing": df.isna().mean().sort_values(ascending=False).to_frame("missing_rate"),
        "dtypes": df.dtypes.astype(str).to_frame("dtype"),
    }
    
    # Class balance for target columns if present
    for col in [cfg.target_column, cfg.arrhythmia_column]:
        if col and col in df.columns:
            vc = df[col].value_counts(dropna=False)
            frac = vc / vc.sum()
            summary[f"class_balance_{col}"] = pd.DataFrame({"count": vc, "fraction": frac})
    
    return summary


def select_feature_target(df: pd.DataFrame, target_col: str) -> tuple[pd.DataFrame, pd.Series]:
    """Separate features and target, dropping target from features.
    
    Args:
        df: Input DataFrame
        target_col: Name of target column
        
    Returns:
        Tuple of (X, y) where X is features and y is target
        
    Raises:
        KeyError: If target column not found
    """
    if target_col not in df.columns:
        raise KeyError(f"Required target column '{target_col}' not found in dataset.")
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


def data_audit(df: pd.DataFrame, feature_cols: Optional[List[str]] = None, top_n: int = 30) -> dict:
    """Early audit to inspect NaNs and feature quality.
    
    Args:
        df: Input DataFrame
        feature_cols: List of feature columns to audit. If None, uses all columns
        top_n: Number of columns to show in NaN summary
        
    Returns:
        Dictionary with:
        - head: First rows of features
        - nan_summary: DataFrame with count and fraction of NaNs per column (sorted desc)
        - full_missing: List of columns fully NaN
        - mostly_missing: List of columns with >80% NaNs
        - constant: List of columns with <=1 unique non-NaN value
    """
    if feature_cols is None:
        feature_cols = list(df.columns)
    
    X = df[feature_cols]
    
    # NaN analysis
    nan_count = X.isna().sum()
    nan_frac = (nan_count / len(X)).fillna(0.0)
    nan_summary = (
        pd.DataFrame({"nan_count": nan_count, "nan_fraction": nan_frac})
        .sort_values("nan_fraction", ascending=False)
        .head(top_n)
    )
    
    # Quality issues
    full_missing = [c for c in feature_cols if X[c].isna().all()]
    mostly_missing = [c for c in feature_cols if X[c].isna().mean() > 0.8]
    constant = [c for c in feature_cols if X[c].nunique(dropna=True) <= 1]
    
    return {
        "head": X.head(10),
        "nan_summary": nan_summary,
        "full_missing": full_missing,
        "mostly_missing": mostly_missing,
        "constant": constant,
    }
