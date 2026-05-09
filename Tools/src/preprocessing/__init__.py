"""Preprocessing module for ML pipelines.

This module provides sklearn-based preprocessing pipelines that integrate
with the data cleaning pipeline, including class imbalance handling.
"""

from .pipelines import build_preprocessing_pipeline, PreprocessingConfig
from .integrations import load_data_with_fallback, get_latest_cleaned_dataset

# Import imbalance handling module
from .imbalance import (
    ImbalanceStrategy,
    ImbalanceConfig,
    STRATEGY_DESCRIPTIONS,
    detect_imbalance,
    get_recommended_strategy,
    create_sampler,
    compute_class_weights,
    apply_class_weight_to_model,
    resample_data,
    get_imbalance_report,
    ImbalanceHandler,
    get_smote_sampler,
    get_adasyn_sampler,
    get_combined_sampler,
    is_imblearn_available,
)

# Backward compatibility alias
build_preprocess_pipelines = build_preprocessing_pipeline

__all__ = [
    # Pipeline building
    "build_preprocessing_pipeline",
    "PreprocessingConfig",
    "load_data_with_fallback",
    "get_latest_cleaned_dataset",
    # Imbalance handling
    "ImbalanceStrategy",
    "ImbalanceConfig",
    "STRATEGY_DESCRIPTIONS",
    "detect_imbalance",
    "get_recommended_strategy",
    "create_sampler",
    "compute_class_weights",
    "apply_class_weight_to_model",
    "resample_data",
    "get_imbalance_report",
    "ImbalanceHandler",
    "get_smote_sampler",
    "get_adasyn_sampler",
    "get_combined_sampler",
    "is_imblearn_available",
    # Backward compatibility
    "build_preprocess_pipelines",
]
