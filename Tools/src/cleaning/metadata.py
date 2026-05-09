"""Variable metadata classes for tracking data cleaning operations.

This module defines data structures for storing metadata about variables
during the cleaning process, including transformations applied and quality metrics.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
import numpy as np


@dataclass
class VariableMetadata:
    """Metadata for a variable after cleaning operations.
    
    Tracks all transformations, quality issues, and statistical properties
    of a variable throughout the cleaning pipeline.
    
    Attributes:
        name: Variable name
        description: Human-readable description
        original_type: Original pandas dtype
        cleaned_type: Dtype after cleaning
        is_numerical: Whether variable is numeric
        is_categorical: Whether variable is categorical
        original_min: Minimum value before cleaning
        original_max: Maximum value before cleaning
        cleaned_min: Minimum value after cleaning
        cleaned_max: Maximum value after cleaning
        unique_values: List of unique values (for categoricals)
        encoding_type: Type of encoding applied
        encoding_mapping: Mapping used for encoding
        discretization_bins: Bin edges if discretized
        discretization_labels: Labels for bins if discretized
        missing_count_original: Number of missing values before cleaning
        missing_percent_original: Percentage of missing values before cleaning
        imputation_method: Method used for imputation
        outliers_detected: Number of outliers detected
        outliers_treated: Number of outliers treated
        quality_flags: List of quality issues or notes
    """
    name: str
    description: str = ""
    original_type: str = ""
    cleaned_type: str = ""
    is_numerical: bool = True
    is_categorical: bool = False
    
    # Value ranges
    original_min: Optional[float] = None
    original_max: Optional[float] = None
    cleaned_min: Optional[float] = None
    cleaned_max: Optional[float] = None
    
    # Categorical information
    unique_values: List[str] = field(default_factory=list)
    encoding_type: Optional[str] = None
    encoding_mapping: Dict[str, Any] = field(default_factory=dict)
    
    # Discretization
    discretization_bins: Optional[List[float]] = None
    discretization_labels: Optional[List[str]] = None
    
    # Missing values
    missing_count_original: int = 0
    missing_percent_original: float = 0.0
    imputation_method: Optional[str] = None
    
    # Quality metrics
    outliers_detected: int = 0
    outliers_treated: int = 0
    quality_flags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        """Convert metadata to dictionary.
        
        Returns:
            Dictionary representation of metadata with numpy types converted to Python types
        """
        def convert_value(val):
            """Convert numpy types to Python types for JSON serialization."""
            if isinstance(val, (np.integer, np.int64, np.int32)):
                return int(val)
            elif isinstance(val, (np.floating, np.float64, np.float32)):
                return float(val)
            elif isinstance(val, np.ndarray):
                return val.tolist()
            elif isinstance(val, dict):
                return {k: convert_value(v) for k, v in val.items()}
            elif isinstance(val, list):
                return [convert_value(item) for item in val]
            return val
        
        data = asdict(self)
        # Convert all numpy types in the dictionary
        for key, value in data.items():
            data[key] = convert_value(value)
        return data
