"""Data cleaning module.

This module provides comprehensive data cleaning utilities:
- Missing value imputation (multiple strategies)
- Outlier detection and treatment
- Categorical encoding (one-hot, label, ordinal)
- Data type validation
- Duplicate removal
- Variable discretization
- Metadata generation and tracking
"""

from .cleaner import DataCleaner, CleaningConfig, quick_clean
from .metadata import VariableMetadata
from .imputation import ImputationStrategy
from .outliers import OutlierMethod, OutlierTreatment
from .encoding import EncodingStrategy
from .discretization import DiscretizationStrategy, discretize_column

__all__ = [
    # Main classes
    "DataCleaner",
    "CleaningConfig",
    "VariableMetadata",
    # Strategies
    "ImputationStrategy",
    "OutlierMethod",
    "OutlierTreatment",
    "EncodingStrategy",
    "DiscretizationStrategy",
    # Functions
    "discretize_column",
    "quick_clean",
]
