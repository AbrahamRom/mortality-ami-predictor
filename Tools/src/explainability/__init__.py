"""Model explainability module.

This module provides SHAP analysis, permutation importance,
partial dependence plots, inverse optimization, and custom models explainability 
for comprehensive model interpretation.
"""

from .shap_analysis import (
    compute_shap_values,
    plot_shap_beeswarm,
    plot_shap_bar,
    plot_shap_waterfall,
    plot_shap_force,
    get_feature_importance,
    get_sample_shap_values,
)
from .permutation import compute_permutation_importance, plot_permutation_importance
from .partial_dependence import plot_partial_dependence
from .pdf_reports import generate_explainability_pdf
from .custom_integration import (
    compute_shap_for_custom_model,
    compute_permutation_importance_custom,
    get_feature_importance_universal,
    explain_prediction_custom,
    batch_explain_models,
)
from .inverse_optimization import InverseOptimizer, find_counterfactuals
from .inverse_plots import (
    plot_optimal_values_comparison,
    plot_confidence_intervals,
    plot_sensitivity_analysis,
    plot_sensitivity_heatmap,
    plot_optimization_convergence,
    plot_feature_importance_for_optimization,
    plot_bootstrap_distributions,
    plot_parallel_coordinates,
    create_optimization_summary_figure,
)

__all__ = [
    "compute_shap_values",
    "plot_shap_beeswarm",
    "plot_shap_bar",
    "plot_shap_waterfall",
    "plot_shap_force",
    "get_feature_importance",
    "get_sample_shap_values",
    "compute_permutation_importance",
    "plot_permutation_importance",
    "plot_partial_dependence",
    "generate_explainability_pdf",
    # Custom models explainability
    "compute_shap_for_custom_model",
    "compute_permutation_importance_custom",
    "get_feature_importance_universal",
    "explain_prediction_custom",
    "batch_explain_models",
    # Inverse optimization
    "InverseOptimizer",
    "find_counterfactuals",
    "plot_optimal_values_comparison",
    "plot_confidence_intervals",
    "plot_sensitivity_analysis",
    "plot_sensitivity_heatmap",
    "plot_optimization_convergence",
    "plot_feature_importance_for_optimization",
    "plot_bootstrap_distributions",
    "plot_parallel_coordinates",
    "create_optimization_summary_figure",
]
