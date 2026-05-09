"""Regression models for ML experiments."""
from __future__ import annotations

from typing import Dict, Tuple

from sklearn.linear_model import LinearRegression


def make_regressors() -> Dict[str, Tuple[object, Dict]]:
    """Create dictionary of regression models and parameter grids.
    
    Returns:
        Dictionary mapping model_name -> (model, param_grid)
    """
    return {
        "linreg": (LinearRegression(), {}),
    }
