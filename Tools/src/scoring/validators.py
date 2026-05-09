"""
Validation utilities for scoring modules.

Provides reusable validation functions for ensure data quality before
computing clinical risk scores.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple, Optional
import warnings
import math

import numpy as np
import pandas as pd


# =============================================================================
# VALIDATION RESULTS
# =============================================================================

class ValidationResult:
    """Result from validation operation."""
    
    def __init__(self):
        """Initialize validation result."""
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.is_valid = True
    
    def add_error(self, error: str) -> None:
        """Add validation error."""
        self.errors.append(error)
        self.is_valid = False
    
    def add_warning(self, warning: str) -> None:
        """Add validation warning."""
        self.warnings.append(warning)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'is_valid': self.is_valid,
            'errors': self.errors,
            'warnings': self.warnings,
            'error_count': len(self.errors),
            'warning_count': len(self.warnings),
        }


class ComputeResult:
    """Standardized result from compute_safe() methods."""
    
    def __init__(self, score: Optional[float] = None, risk_category: Optional[str] = None):
        """Initialize compute result."""
        self.success = True
        self.score = score
        self.risk_category = risk_category
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.data: Dict[str, Any] = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'success': self.success,
            'score': self.score,
            'risk_category': self.risk_category,
            'errors': self.errors,
            'warnings': self.warnings,
            'data': self.data,
        }


# =============================================================================
# TYPE VALIDATION
# =============================================================================

def is_missing(value: Any) -> bool:
    """Check if value is missing (None, NaN, or pd.NA)."""
    if value is None:
        return True
    if isinstance(value, (float, np.floating)):
        return math.isnan(value)
    if pd.isna(value):
        return True
    return False


def validate_numeric(value: Any, name: str, result: ValidationResult) -> Tuple[bool, float]:
    """
    Validate that value is numeric and not missing.
    
    Args:
        value: Value to validate
        name: Variable name (for error messages)
        result: ValidationResult to append errors/warnings
        
    Returns:
        Tuple of (is_valid, numeric_value)
    """
    # Check for missing
    if is_missing(value):
        result.add_error(f"{name}: Missing value (None, NaN, or NA)")
        return False, 0.0
    
    # Try to convert to float
    try:
        numeric_value = float(value)
        return True, numeric_value
    except (TypeError, ValueError):
        result.add_error(f"{name}: Invalid type (cannot convert to numeric). Got: {type(value).__name__}")
        return False, 0.0


def validate_bool(value: Any, name: str, result: ValidationResult) -> Tuple[bool, bool]:
    """
    Validate that value is boolean.
    
    Args:
        value: Value to validate
        name: Variable name (for error messages)
        result: ValidationResult to append errors/warnings
        
    Returns:
        Tuple of (is_valid, bool_value)
    """
    if is_missing(value):
        result.add_warning(f"{name}: Missing value, defaulting to False")
        return True, False
    
    if isinstance(value, bool):
        return True, value
    
    if isinstance(value, (int, np.integer)):
        return True, bool(value)
    
    result.add_error(f"{name}: Invalid type for boolean. Got: {type(value).__name__}")
    return False, False


def validate_int(value: Any, name: str, result: ValidationResult) -> Tuple[bool, int]:
    """
    Validate that value is integer.
    
    Args:
        value: Value to validate
        name: Variable name (for error messages)
        result: ValidationResult to append errors/warnings
        
    Returns:
        Tuple of (is_valid, int_value)
    """
    if is_missing(value):
        result.add_error(f"{name}: Missing value (None, NaN, or NA)")
        return False, 0
    
    try:
        int_value = int(value)
        return True, int_value
    except (TypeError, ValueError):
        result.add_error(f"{name}: Invalid type (cannot convert to int). Got: {type(value).__name__}")
        return False, 0


# =============================================================================
# RANGE VALIDATION
# =============================================================================

def validate_range(
    value: float,
    name: str,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
    result: Optional[ValidationResult] = None,
) -> bool:
    """
    Validate that value is within range.
    
    Args:
        value: Numeric value to validate
        name: Variable name (for messages)
        min_val: Minimum valid value (inclusive)
        max_val: Maximum valid value (inclusive)
        result: ValidationResult to append warnings
        
    Returns:
        True if within expected range, False otherwise
    """
    if result is None:
        result = ValidationResult()
    
    if min_val is not None and value < min_val:
        msg = f"{name}: Value {value} is below minimum {min_val}"
        result.add_warning(msg)
        return False
    
    if max_val is not None and value > max_val:
        msg = f"{name}: Value {value} is above maximum {max_val}"
        result.add_warning(msg)
        return False
    
    return True


# =============================================================================
# REQUIRED VARIABLES VALIDATION
# =============================================================================

def validate_required_vars(
    kwargs: Dict[str, Any],
    required_vars: List[str],
    result: ValidationResult,
) -> bool:
    """
    Check that all required variables are present and not missing.
    
    Args:
        kwargs: Dictionary of input variables
        required_vars: List of required variable names
        result: ValidationResult to append errors
        
    Returns:
        True if all required variables are present
    """
    missing_vars = []
    
    for var in required_vars:
        if var not in kwargs:
            missing_vars.append(f"{var} (not provided)")
        elif is_missing(kwargs[var]):
            missing_vars.append(f"{var} (None/NaN/NA)")
    
    if missing_vars:
        msg = f"Missing required variables: {', '.join(missing_vars)}"
        result.add_error(msg)
        return False
    
    return True


# =============================================================================
# CLINICAL RANGES (for warnings)
# =============================================================================

CLINICAL_RANGES = {
    'age': (0, 120),
    'heart_rate': (0, 250),
    'systolic_bp': (0, 300),
    'diastolic_bp': (0, 200),
    'creatinine': (0, 20),
    'gfr': (0, 120),
    'weight_kg': (20, 250),
    'time_to_treatment_hours': (0, 48),
    'ecg_leads_affected': (0, 12),
    'killip_class': (1, 4),
}


def validate_clinical_ranges(
    kwargs: Dict[str, Any],
    result: ValidationResult,
) -> None:
    """
    Validate that clinical values are within typical ranges.
    
    Adds warnings (not errors) for out-of-range values.
    
    Args:
        kwargs: Dictionary of input variables
        result: ValidationResult to append warnings
    """
    for var_name, (min_val, max_val) in CLINICAL_RANGES.items():
        if var_name in kwargs and not is_missing(kwargs[var_name]):
            try:
                value = float(kwargs[var_name])
                validate_range(value, var_name, min_val, max_val, result)
            except (TypeError, ValueError):
                pass  # Already handled in type validation


def validate_score_safe(
    score: Optional[float],
    method_name: str,
) -> ComputeResult:
    """
    Create a ComputeResult for a failed validation.
    
    Args:
        score: The computed score (or None if invalid)
        method_name: Name of the scoring method
        
    Returns:
        ComputeResult with success=False
    """
    result = ComputeResult()
    result.success = False
    result.score = score
    return result


# =============================================================================
# GRACE-SPECIFIC VALIDATION
# =============================================================================

def validate_grace_inputs(
    age: Any,
    heart_rate: Any,
    systolic_bp: Any,
    creatinine: Any,
) -> ValidationResult:
    """Validate GRACE score inputs."""
    result = ValidationResult()
    
    # Validate required variables
    required = ['age', 'heart_rate', 'systolic_bp', 'creatinine']
    kwargs = {
        'age': age,
        'heart_rate': heart_rate,
        'systolic_bp': systolic_bp,
        'creatinine': creatinine,
    }
    
    if not validate_required_vars(kwargs, required, result):
        return result
    
    # Validate numeric types
    is_valid, age_val = validate_numeric(age, 'age', result)
    is_valid, hr_val = validate_numeric(heart_rate, 'heart_rate', result)
    is_valid, sbp_val = validate_numeric(systolic_bp, 'systolic_bp', result)
    is_valid, cre_val = validate_numeric(creatinine, 'creatinine', result)
    
    if not result.is_valid:
        return result
    
    # Validate clinical ranges
    validate_clinical_ranges(kwargs, result)
    
    return result


# =============================================================================
# TIMI-SPECIFIC VALIDATION
# =============================================================================

def validate_timi_stemi_inputs(
    age: Any,
    systolic_bp: Any,
    heart_rate: Any,
    killip_class: Any,
) -> ValidationResult:
    """Validate TIMI STEMI score inputs."""
    result = ValidationResult()
    
    required = ['age', 'systolic_bp', 'heart_rate', 'killip_class']
    kwargs = {
        'age': age,
        'systolic_bp': systolic_bp,
        'heart_rate': heart_rate,
        'killip_class': killip_class,
    }
    
    if not validate_required_vars(kwargs, required, result):
        return result
    
    # Validate numeric types
    is_valid, age_val = validate_numeric(age, 'age', result)
    is_valid, sbp_val = validate_numeric(systolic_bp, 'systolic_bp', result)
    is_valid, hr_val = validate_numeric(heart_rate, 'heart_rate', result)
    is_valid, killip_val = validate_int(killip_class, 'killip_class', result)
    
    if not result.is_valid:
        return result
    
    # Validate killip class range
    if not (1 <= killip_val <= 4):
        result.add_error(f"killip_class: Must be 1-4, got {killip_val}")
    
    # Validate clinical ranges
    validate_clinical_ranges(kwargs, result)
    
    return result


def validate_timi_nstemi_inputs(age: Any) -> ValidationResult:
    """Validate TIMI NSTEMI score inputs (minimal validation for age)."""
    result = ValidationResult()
    
    if age is None or is_missing(age):
        result.add_error("age: Missing value (None, NaN, or NA)")
        return result
    
    is_valid, age_val = validate_numeric(age, 'age', result)
    if not result.is_valid:
        return result
    
    validate_clinical_ranges({'age': age_val}, result)
    
    return result


# =============================================================================
# RECUIMA-SPECIFIC VALIDATION
# =============================================================================

def validate_recuima_inputs(
    age: Any,
    systolic_bp: Any,
    gfr: Any,
    ecg_leads_affected: Any,
    killip_class: Any,
) -> ValidationResult:
    """Validate RECUIMA score inputs."""
    result = ValidationResult()
    
    required = ['age', 'systolic_bp', 'gfr', 'ecg_leads_affected', 'killip_class']
    kwargs = {
        'age': age,
        'systolic_bp': systolic_bp,
        'gfr': gfr,
        'ecg_leads_affected': ecg_leads_affected,
        'killip_class': killip_class,
    }
    
    if not validate_required_vars(kwargs, required, result):
        return result
    
    # Validate numeric types
    is_valid, age_val = validate_numeric(age, 'age', result)
    is_valid, sbp_val = validate_numeric(systolic_bp, 'systolic_bp', result)
    is_valid, gfr_val = validate_numeric(gfr, 'gfr', result)
    is_valid, ecg_val = validate_int(ecg_leads_affected, 'ecg_leads_affected', result)
    is_valid, killip_val = validate_int(killip_class, 'killip_class', result)
    
    if not result.is_valid:
        return result
    
    # Validate killip class range
    if not (1 <= killip_val <= 4):
        result.add_error(f"killip_class: Must be 1-4, got {killip_val}")
    
    # Validate ECG leads range
    if not (0 <= ecg_val <= 12):
        result.add_warning(f"ecg_leads_affected: Expected 0-12, got {ecg_val}")
    
    # Validate clinical ranges
    validate_clinical_ranges(kwargs, result)
    
    return result
