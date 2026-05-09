"""I/O utilities for data operations.

This module provides utilities for file format detection,
dataset saving, and other I/O operations.
"""
from __future__ import annotations

import os
from typing import Optional

import pandas as pd


def read_csv_with_encoding(path: str, **kwargs) -> pd.DataFrame:
    """Read CSV file with automatic encoding detection.
    
    Tries multiple common encodings used in Spanish datasets.
    Also handles parsing errors with robust settings.
    
    Args:
        path: Path to CSV file
        **kwargs: Additional arguments passed to pd.read_csv
        
    Returns:
        Loaded DataFrame
        
    Raises:
        RuntimeError: If file cannot be decoded with any encoding
    """
    encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'windows-1252']
    last_error = None
    
    # Add robust parsing parameters if not specified
    if 'on_bad_lines' not in kwargs:
        kwargs['on_bad_lines'] = 'warn'
    if 'low_memory' not in kwargs:
        kwargs['low_memory'] = False
    
    for encoding in encodings:
        try:
            return pd.read_csv(path, encoding=encoding, **kwargs)
        except (UnicodeDecodeError, LookupError) as e:
            last_error = e
            continue
        except pd.errors.ParserError:
            # If parser error, try with error_bad_lines=False (skip bad lines)
            try:
                return pd.read_csv(path, encoding=encoding, on_bad_lines='skip', **kwargs)
            except Exception as e:
                last_error = e
                continue
    
    raise RuntimeError(
        f"Failed to decode CSV with any encoding. Last error: {last_error}"
    )


def detect_file_format(path: str) -> str:
    """Detect file format from extension.
    
    Args:
        path: File path
        
    Returns:
        Format string: 'csv', 'parquet', 'feather', 'xlsx', 'xls', or 'unknown'
    """
    ext = os.path.splitext(path)[1].lower()
    
    format_map = {
        ".csv": "csv",
        ".parquet": "parquet",
        ".feather": "feather",
        ".xlsx": "xlsx",
        ".xls": "xls",
    }
    
    return format_map.get(ext, "unknown")


def save_dataset(
    df: pd.DataFrame,
    path: str,
    format: Optional[str] = None,
    **kwargs
) -> None:
    """Save DataFrame to file with format auto-detection.
    
    Args:
        df: DataFrame to save
        path: Output file path
        format: Format to use ('csv', 'parquet', 'feather', 'xlsx').
               If None, infers from file extension
        **kwargs: Additional arguments passed to pandas save method
    """
    if format is None:
        format = detect_file_format(path)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    if format == "csv":
        df.to_csv(path, index=False, **kwargs)
    elif format == "parquet":
        df.to_parquet(path, index=False, **kwargs)
    elif format == "feather":
        df.to_feather(path, **kwargs)
    elif format == "xlsx":
        df.to_excel(path, index=False, engine="openpyxl", **kwargs)
    else:
        raise ValueError(f"Unsupported format: {format}")
