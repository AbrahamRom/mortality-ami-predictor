"""Permutation importance utilities."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict

import numpy as np
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance


# Use new processed/plots structure
ROOT_DIR = Path(__file__).parents[2]
FIG_DIR = ROOT_DIR / "processed" / "plots" / "explainability"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def compute_permutation_importance(
    model,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list,
    n_repeats: int = 10,
    random_state: int = 42,
    scoring: str = "roc_auc",
) -> Dict:
    """Compute permutation importance.
    
    Args:
        model: Trained model
        X: Feature matrix
        y: Target labels
        feature_names: List of feature names
        n_repeats: Number of permutation repeats
        random_state: Random seed
        scoring: Scoring metric
        
    Returns:
        Dictionary with importances_mean, importances_std, feature_names
    """
    result = permutation_importance(
        model,
        X,
        y,
        n_repeats=n_repeats,
        random_state=random_state,
        scoring=scoring,
    )
    
    return {
        "importances_mean": result.importances_mean,
        "importances_std": result.importances_std,
        "feature_names": feature_names,
    }


def plot_permutation_importance(
    importances: Dict,
    name: str = "model",
    max_features: int = 20,
) -> str:
    """Plot permutation importance.
    
    Args:
        importances: Dictionary from compute_permutation_importance
        name: Model name for filename
        max_features: Maximum features to display
        
    Returns:
        Path to saved figure
    """
    mean_importance = importances["importances_mean"]
    std_importance = importances["importances_std"]
    feature_names = importances["feature_names"]
    
    # Sort by importance
    indices = np.argsort(mean_importance)[::-1][:max_features]
    
    plt.figure(figsize=(10, 8))
    plt.barh(
        range(len(indices)),
        mean_importance[indices],
        xerr=std_importance[indices],
        alpha=0.7,
    )
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel("Permutation Importance", fontsize=12)
    plt.ylabel("Feature", fontsize=12)
    plt.title(f"Permutation Importance: {name}", fontsize=14)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    path = os.path.join(FIG_DIR, f"permutation_importance_{name}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    
    return path
