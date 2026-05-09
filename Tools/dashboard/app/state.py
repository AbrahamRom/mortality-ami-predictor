"""Session state management for dashboard."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from src.data_load import read_csv_with_encoding
from .config import MODELS_DIR, CLEANED_DATASETS_DIR, TESTSETS_DIR, PROCESSED_DIR


def initialize_state():
    """Initialize session state variables if not already set."""
    
    # Data paths - using new structure
    if "data_dir" not in st.session_state:
        st.session_state.data_dir = PROCESSED_DIR
    
    if "models_dir" not in st.session_state:
        st.session_state.models_dir = MODELS_DIR
    
    if "testsets_dir" not in st.session_state:
        st.session_state.testsets_dir = TESTSETS_DIR
    
    if "cleaned_datasets_dir" not in st.session_state:
        st.session_state.cleaned_datasets_dir = CLEANED_DATASETS_DIR
    
    # Raw and cleaned data
    if "raw_data" not in st.session_state:
        st.session_state.raw_data = None
    
    if "cleaned_data" not in st.session_state:
        st.session_state.cleaned_data = None
    
    # Dataset
    if "df" not in st.session_state:
        st.session_state.df = None
    
    if "df_cleaned" not in st.session_state:
        st.session_state.df_cleaned = None
    
    # Train/test split
    if "X_train" not in st.session_state:
        st.session_state.X_train = None
    
    if "X_test" not in st.session_state:
        st.session_state.X_test = None
    
    if "y_train" not in st.session_state:
        st.session_state.y_train = None
    
    if "y_test" not in st.session_state:
        st.session_state.y_test = None
    
    # Trained models
    if "trained_models" not in st.session_state:
        st.session_state.trained_models = {}
    
    # Model evaluation results
    if "evaluation_results" not in st.session_state:
        st.session_state.evaluation_results = {}
    
    # SHAP explainer
    if "shap_explainer" not in st.session_state:
        st.session_state.shap_explainer = None
    
    # SHAP values
    if "shap_values" not in st.session_state:
        st.session_state.shap_values = None
    
    # Predictions
    if "predictions" not in st.session_state:
        st.session_state.predictions = {}
    
    # Training flags
    if "is_trained" not in st.session_state:
        st.session_state.is_trained = False
    
    if "is_evaluated" not in st.session_state:
        st.session_state.is_evaluated = False


def get_state(key: str, default: Any = None) -> Any:
    """Get a value from session state.
    
    Args:
        key: Session state key
        default: Default value if key not found
        
    Returns:
        Value from session state or default
    """
    return st.session_state.get(key, default)


def set_state(key: str, value: Any):
    """Set a value in session state.
    
    Args:
        key: Session state key
        value: Value to set
    """
    st.session_state[key] = value


def clear_state():
    """Clear all session state variables."""
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    initialize_state()


def load_data(file_path: Path | str, use_cache: bool = True) -> pd.DataFrame:
    """Load data from CSV file.
    
    Args:
        file_path: Path to CSV file
        use_cache: Whether to use cached data if available
        
    Returns:
        Loaded DataFrame
    """
    file_path = Path(file_path)
    
    # Check if already loaded
    if use_cache and st.session_state.df is not None:
        return st.session_state.df
    
    # Use the robust CSV reader with encoding detection and error handling
    df = read_csv_with_encoding(str(file_path))
    
    st.session_state.df = df
    
    return df


def get_available_models() -> dict[str, Path]:
    """Get available saved models.
    
    Returns:
        Dictionary mapping model names to file paths
    """
    models_dir = get_state("models_dir")
    if not models_dir.exists():
        return {}
    
    models = {}
    for model_file in models_dir.glob("*.joblib"):
        model_name = model_file.stem
        models[model_name] = model_file
    
    return models


def is_data_loaded() -> bool:
    """Check if data is loaded.
    
    Returns:
        True if data is loaded
    """
    return st.session_state.df is not None


def is_trained() -> bool:
    """Check if models are trained.
    
    Returns:
        True if models are trained
    """
    return st.session_state.is_trained and len(st.session_state.trained_models) > 0


def is_evaluated() -> bool:
    """Check if models are evaluated.
    
    Returns:
        True if models are evaluated
    """
    return st.session_state.is_evaluated and len(st.session_state.evaluation_results) > 0
