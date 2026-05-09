"""Models module - Machine Learning model definitions.

This module provides:
- Classification models (KNN, Logistic Regression, Tree-based, XGBoost, LightGBM)
- Regression models
- Neural network models (PyTorch)
- Model registry and factory patterns
- Model metadata and selection
- Custom models infrastructure
"""

from .classifiers import make_classifiers
from .regressors import make_regressors
from .neural_networks import TorchTabularClassifier
from .registry import ModelRegistry, get_model
from .metadata import (
    ModelMetadata,
    DatasetMetadata,
    TrainingMetadata,
    PerformanceMetrics,
    create_metadata_from_training,
)
from .selection import (
    BestModelSelector,
    SelectionCriteria,
    ModelScore,
    select_best_model_simple,
)
from .custom_base import (
    BaseCustomModel,
    BaseCustomClassifier,
    BaseCustomRegressor,
    CustomModelWrapper,
    SimpleMLPClassifier,
)
from .persistence import (
    save_custom_model,
    load_custom_model,
    validate_loaded_model,
    migrate_model,
    create_model_bundle,
    load_model_bundle,
    list_saved_models,
    ModelPersistenceError,
    ModelValidationError,
)

__all__ = [
    # Factories
    "make_classifiers",
    "make_regressors",
    # Neural networks
    "TorchTabularClassifier",
    # Registry
    "ModelRegistry",
    "get_model",
    # Metadata
    "ModelMetadata",
    "DatasetMetadata",
    "TrainingMetadata",
    "PerformanceMetrics",
    "create_metadata_from_training",
    # Selection
    "BestModelSelector",
    "SelectionCriteria",
    "ModelScore",
    "select_best_model_simple",
    # Custom models
    "BaseCustomModel",
    "BaseCustomClassifier",
    "BaseCustomRegressor",
    "CustomModelWrapper",
    "SimpleMLPClassifier",
    # Persistence
    "save_custom_model",
    "load_custom_model",
    "validate_loaded_model",
    "migrate_model",
    "create_model_bundle",
    "load_model_bundle",
    "list_saved_models",
    "ModelPersistenceError",
    "ModelValidationError",
]
