"""Integration utilities for preprocessing with cleaned data.

This module provides utilities to integrate preprocessing pipelines
with the data cleaning module.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

from ..config import CONFIG


def get_latest_cleaned_dataset(task_name: Optional[str] = None) -> Optional[Path]:
    """Get path to the most recent cleaned dataset.
    
    Args:
        task_name: Optional task name to filter datasets
        
    Returns:
        Path to latest cleaned dataset, or None if not found
    """
    cleaned_dir = Path(CONFIG.cleaned_data_dir)
    
    if not cleaned_dir.exists():
        return None
    
    # Look for cleaned datasets
    pattern = "cleaned_dataset_*.csv"
    cleaned_files = sorted(
        cleaned_dir.glob(pattern),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )
    
    if cleaned_files:
        return cleaned_files[0]
    
    return None


def load_data_with_fallback(
    raw_data_path: str,
    use_cleaned: bool = True,
    task_name: Optional[str] = None,
) -> Tuple[pd.DataFrame, bool]:
    """Load dataset with fallback to raw data if cleaned not available.
    
    This function attempts to load cleaned data first, then falls back
    to raw data if cleaned data is not available.
    
    Args:
        raw_data_path: Path to raw dataset
        use_cleaned: Whether to try loading cleaned data first
        task_name: Optional task name for filtering cleaned datasets
        
    Returns:
        Tuple of (DataFrame, is_cleaned_data)
    """
    if use_cleaned:
        cleaned_path = get_latest_cleaned_dataset(task_name)
        if cleaned_path and cleaned_path.exists():
            try:
                # Try multiple encodings for CSV files
                encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'windows-1252']
                df = None
                for encoding in encodings:
                    try:
                        df = pd.read_csv(cleaned_path, encoding=encoding)
                        break
                    except (UnicodeDecodeError, LookupError):
                        continue
                
                if df is None:
                    raise RuntimeError("Failed to decode CSV with any encoding")
                
                print(f"✅ Loaded cleaned dataset from: {cleaned_path}")
                return df, True
            except Exception as e:
                print(f"⚠️  Failed to load cleaned data: {e}")
                print("Falling back to raw data...")
    
    # Fallback to raw data
    try:
        if raw_data_path.endswith('.csv'):
            # Try multiple encodings for CSV files
            encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'windows-1252']
            df = None
            last_error = None
            
            for encoding in encodings:
                try:
                    df = pd.read_csv(raw_data_path, encoding=encoding)
                    break
                except (UnicodeDecodeError, LookupError) as e:
                    last_error = e
                    continue
            
            if df is None:
                raise RuntimeError(
                    f"Failed to decode CSV with any encoding. Last error: {last_error}"
                )
        elif raw_data_path.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(raw_data_path)
        elif raw_data_path.endswith('.parquet'):
            df = pd.read_parquet(raw_data_path)
        else:
            raise ValueError(f"Unsupported file format: {raw_data_path}")
        
        print(f"📊 Loaded raw dataset from: {raw_data_path}")
        return df, False
    except Exception as e:
        raise RuntimeError(
            f"Failed to load dataset from {raw_data_path}: {e}"
        ) from e
