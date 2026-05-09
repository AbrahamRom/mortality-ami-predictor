"""Mortality AMI Predictor - Source Code Package

This package provides a modular, scalable framework for:
- Data loading and preprocessing
- Data cleaning and quality assurance
- Exploratory data analysis
- Feature engineering
- Model training and evaluation
- Clinical risk scoring
- Model explainability

## Module Structure

- **data**: Dataset loading, I/O, and splitting
- **cleaning**: Data cleaning, imputation, outlier handling
- **eda**: Exploratory data analysis and visualization
- **features**: Feature selection and engineering
- **preprocessing**: ML preprocessing pipelines
- **models**: Model definitions and registry
- **training**: Training workflows and hyperparameter tuning
- **evaluation**: Model evaluation and metrics
- **scoring**: Clinical risk scores (GRACE, TIMI)
- **explainability**: Model interpretability (SHAP, PDP)

## Quick Start

```python
# Load and clean data
from src.data import load_dataset
from src.cleaning import quick_clean

df = load_dataset("path/to/data.csv")
df_clean, cleaner = quick_clean(df, target_column="mortality_inhospital")

# Explore data
from src.eda import quick_eda
analyzer = quick_eda(df_clean)

# Train model
from src.training import train_best_classifier
from src.features import safe_feature_columns

X = df_clean[safe_feature_columns(df_clean, ["mortality_inhospital"])]
y = df_clean["mortality_inhospital"]

model_path, model = train_best_classifier(X, y)
```

## Backward Compatibility

For backward compatibility with older code, you can still import from root-level modules:

```python
# Old style (still works)
from src.data import load_dataset, train_test_split, summarize_dataframe
from src.features import safe_feature_columns
from src.models import make_classifiers

# New style (recommended)
from src.data.loaders import load_dataset
from src.data.splitters import train_test_split
from src.features.selectors import safe_feature_columns
from src.models.classifiers import make_classifiers
```
"""

# Import configuration
from .config import CONFIG, ProjectConfig, RANDOM_SEED

# Re-export key functions for backward compatibility
# This allows: from src.data import load_dataset
# Instead of: from src.data.loaders import load_dataset

# Data module exports
from .data_load import (
    load_dataset,
    get_dataset_info,
    DatasetInfo,
    train_test_split,
    summarize_dataframe,
    select_feature_target,
    data_audit,
)

# Features module exports
from .features import safe_feature_columns

# Models module exports
from .models import make_classifiers, make_regressors

# Scoring module exports
from .scoring import get_score, list_scores

# Training module exports (backward compatibility with train.py)
from .training import (
    train_best_classifier,
    train_selected_classifiers,
    fit_and_save_best_classifier,
    fit_and_save_selected_classifiers,
)

# Evaluation module exports (backward compatibility with evaluate.py)
from .evaluation import (
    compute_classification_metrics,
    plot_calibration_curve as plot_calibration,
    decision_curve_analysis,
    evaluate_main,
)

# Preprocessing module exports (backward compatibility with preprocess.py)
from .preprocessing import (
    build_preprocessing_pipeline,
    build_preprocess_pipelines,
    PreprocessingConfig,
    load_data_with_fallback,
)

__all__ = [
    # Config
    "CONFIG",
    "ProjectConfig",
    "RANDOM_SEED",
    # Data functions (backward compatibility)
    "load_dataset",
    "get_dataset_info",
    "DatasetInfo",
    "train_test_split",
    "summarize_dataframe",
    "select_feature_target",
    "data_audit",
    # Features
    "safe_feature_columns",
    # Models
    "make_classifiers",
    "make_regressors",
    # Scoring
    "get_score",
    "list_scores",
    # Training (backward compatibility)
    "train_best_classifier",
    "train_selected_classifiers",
    "fit_and_save_best_classifier",
    "fit_and_save_selected_classifiers",
    # Evaluation (backward compatibility)
    "compute_classification_metrics",
    "plot_calibration",
    "decision_curve_analysis",
    "evaluate_main",
    # Preprocessing (backward compatibility)
    "build_preprocessing_pipeline",
    "build_preprocess_pipelines",
    "PreprocessingConfig",
    "load_data_with_fallback",
    # Modules
    "data_load",
    "cleaning",
    "eda",
    "features",
    "preprocessing",
    "models",
    "training",
    "evaluation",
    "scoring",
    "explainability",
]

# Version
__version__ = "2.0.0"
