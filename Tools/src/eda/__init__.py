"""EDA (Exploratory Data Analysis) module.

This module provides comprehensive tools for exploring datasets:
- Univariate statistics and distributions
- Bivariate relationships and correlations
- Multivariate analysis (PCA)
- Interactive visualizations
- PDF report generation
"""

from .analyzer import EDAAnalyzer, quick_eda
from .univariate import UnivariateStats, compute_numeric_stats, compute_categorical_stats
from .bivariate import (
    BivariateStats,
    analyze_numeric_numeric,
    analyze_numeric_categorical,
    analyze_categorical_categorical,
)
from .multivariate import PCAResults, perform_pca, get_feature_importance_pca
from .visualizations import (
    plot_distribution,
    plot_correlation_matrix,
    plot_scatter,
    plot_pairwise_scatter,
    plot_pca_scree,
    plot_pca_biplot,
)
from .pdf_reports import (
    generate_univariate_pdf,
    generate_bivariate_pdf,
    generate_multivariate_pdf,
)

__all__ = [
    # Main classes
    "EDAAnalyzer",
    "quick_eda",
    # Statistics
    "UnivariateStats",
    "BivariateStats",
    "PCAResults",
    # Univariate functions
    "compute_numeric_stats",
    "compute_categorical_stats",
    # Bivariate functions
    "analyze_numeric_numeric",
    "analyze_numeric_categorical",
    "analyze_categorical_categorical",
    # Multivariate functions
    "perform_pca",
    "get_feature_importance_pca",
    # Visualization functions
    "plot_distribution",
    "plot_correlation_matrix",
    "plot_scatter",
    "plot_pairwise_scatter",
    "plot_pca_scree",
    "plot_pca_biplot",
    # PDF report generation
    "generate_univariate_pdf",
    "generate_bivariate_pdf",
    "generate_multivariate_pdf",
]
