"""Evaluation module for model assessment.

This module provides utilities for computing metrics, calibration curves,
decision curves, resampling methods, custom models evaluation, GRACE comparison, and generating reports.
"""

from .metrics import compute_classification_metrics, compute_regression_metrics
from .calibration import plot_calibration_curve, compute_calibration
from .decision_curves import decision_curve_analysis
from .reporters import generate_evaluation_report, evaluate_main
from .resampling import (
    bootstrap_evaluation,
    jackknife_evaluation,
    combined_resampling_evaluation,
    plot_resampling_results,
    plot_resampling_results_plotly,
    ResamplingResult,
)
from .pdf_reports import generate_evaluation_pdf
from .custom_integration import (
    evaluate_custom_classifier,
    evaluate_custom_regressor,
    evaluate_model_universal,
    batch_evaluate_mixed_models,
    compare_model_performance,
    create_evaluation_summary,
)
from .grace_comparison import (
    compare_with_grace,
    plot_roc_comparison,
    plot_calibration_comparison,
    plot_metrics_comparison,
    plot_nri_idi,
    generate_comparison_report,
    ComparisonResult,
)
from .recuima_comparison import (
    compare_with_recuima,
    plot_roc_comparison_recuima,
    plot_calibration_comparison_recuima,
    plot_metrics_comparison_recuima,
    plot_nri_idi_recuima,
    generate_comparison_report_recuima,
    RECUIMAComparisonResult,
    check_recuima_requirements,
    compute_recuima_scores,
    get_recuima_info,
)

__all__ = [
    "compute_classification_metrics",
    "compute_regression_metrics",
    "plot_calibration_curve",
    "compute_calibration",
    "decision_curve_analysis",
    "generate_evaluation_report",
    "evaluate_main",
    "bootstrap_evaluation",
    "jackknife_evaluation",
    "combined_resampling_evaluation",
    "plot_resampling_results",
    "plot_resampling_results_plotly",
    "ResamplingResult",
    "generate_evaluation_pdf",
    # Custom models evaluation
    "evaluate_custom_classifier",
    "evaluate_custom_regressor",
    "evaluate_model_universal",
    "batch_evaluate_mixed_models",
    "compare_model_performance",
    "create_evaluation_summary",
    # GRACE comparison
    "compare_with_grace",
    "plot_roc_comparison",
    "plot_calibration_comparison",
    "plot_metrics_comparison",
    "plot_nri_idi",
    "generate_comparison_report",
    "ComparisonResult",
    # RECUIMA comparison
    "compare_with_recuima",
    "plot_roc_comparison_recuima",
    "plot_calibration_comparison_recuima",
    "plot_metrics_comparison_recuima",
    "plot_nri_idi_recuima",
    "generate_comparison_report_recuima",
    "RECUIMAComparisonResult",
    "check_recuima_requirements",
    "compute_recuima_scores",
    "get_recuima_info",
]
