"""AutoML Module - Automated Machine Learning Integration.

This module provides AutoML capabilities using auto-sklearn, FLAML, and AutoKeras (NAS),
integrated seamlessly with the existing ML pipeline for AMI mortality prediction.

Main components:
- AutoMLClassifier: Wrapper for auto-sklearn classifier
- AutoMLRegressor: Wrapper for auto-sklearn regressor  
- FLAMLClassifier: Cross-platform alternative using FLAML
- NASClassifier: Neural Architecture Search using AutoKeras
- AutoMLConfig: Configuration management
- AutoMLSuggestions: Intelligent technique recommendations

Available Estimators in FLAML:
    - lgbm: LightGBM (gradient boosting)
    - xgboost: XGBoost 
    - xgb_limitdepth: XGBoost with limited depth
    - catboost: CatBoost
    - rf: Random Forest
    - extra_tree: Extra Trees
    - histgb: Histogram-based Gradient Boosting
    - kneighbor: K-Nearest Neighbors
    - lrl1: L1 Regularized Logistic Regression
    - lrl2: L2 Regularized Logistic Regression
    - svc: Support Vector Classifier
    - sgd: Stochastic Gradient Descent
    - nb: Naive Bayes (Gaussian)
    - mlp: Multi-Layer Perceptron (Neural Network)

Note:
    auto-sklearn requires Linux or WSL on Windows.
    FLAML works on all platforms and is used as fallback.
    AutoKeras requires TensorFlow for NAS.
"""

from .config import AutoMLConfig, AutoMLPreset, get_automl_config
from .auto_sklearn_integration import (
    AutoMLClassifier,
    AutoMLRegressor,
    is_autosklearn_available,
)
from .flaml_integration import (
    FLAMLClassifier,
    FLAMLRegressor,
    is_flaml_available,
    FLAML_ESTIMATORS,
    FLAML_ESTIMATOR_DESCRIPTIONS,
    FLAML_CLASSIFICATION_ESTIMATORS,
    FLAML_REGRESSION_ESTIMATORS,
    FLAML_PRESETS,
)
from .nas_integration import (
    NASClassifier,
    NASRegressor,
    NASConfig,
    is_autokeras_available,
    is_tensorflow_available,
)
from .suggestions import (
    AutoMLSuggestions,
    DatasetAnalysis,
    TechniqueSuggestion,
    analyze_dataset,
    get_suggestions,
)
from .export import (
    export_best_model,
    export_ensemble,
    convert_to_standalone,
    create_automl_report,
    load_ensemble,
)

__all__ = [
    # Configuration
    "AutoMLConfig",
    "AutoMLPreset",
    "get_automl_config",
    # auto-sklearn
    "AutoMLClassifier",
    "AutoMLRegressor",
    "is_autosklearn_available",
    # FLAML (cross-platform)
    "FLAMLClassifier",
    "FLAMLRegressor",
    "is_flaml_available",
    "FLAML_ESTIMATORS",
    "FLAML_ESTIMATOR_DESCRIPTIONS",
    "FLAML_CLASSIFICATION_ESTIMATORS",
    "FLAML_REGRESSION_ESTIMATORS",
    "FLAML_PRESETS",
    # NAS (Neural Architecture Search)
    "NASClassifier",
    "NASRegressor",
    "NASConfig",
    "is_autokeras_available",
    "is_tensorflow_available",
    # Suggestions
    "AutoMLSuggestions",
    "DatasetAnalysis",
    "TechniqueSuggestion",
    "analyze_dataset",
    "get_suggestions",
    # Export
    "export_best_model",
    "export_ensemble",
    "convert_to_standalone",
    "create_automl_report",
    "load_ensemble",
]
