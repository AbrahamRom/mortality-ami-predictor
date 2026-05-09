"""Utilities for synchronizing and validating datasets across multiple comparison methods.

Ensures fair comparisons by:
- Enforcing same test set samples across all methods
- Validating paired sample assumptions
- Reporting and handling missing variable issues
- Creating synchronized masks for joint analysis
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import warnings


@dataclass
class SyncValidationReport:
    """Report from test set synchronization validation."""
    
    initial_size: int
    synchronized_size: int
    n_removed: int
    removal_rate: float
    
    # Per-method missing counts
    missing_by_method: Dict[str, int]
    
    # Variables causing issues
    problematic_variables: Dict[str, int]
    
    # Summary message
    is_valid: bool
    message: str


def validate_paired_samples(
    predictions: Dict[str, np.ndarray],
    method_names: Optional[List[str]] = None,
    raise_error: bool = True,
) -> Tuple[bool, str]:
    """Validate that all prediction arrays have the same length (paired samples).
    
    Args:
        predictions: Dictionary mapping method names to prediction arrays
        method_names: Subset of methods to validate (default: all)
        raise_error: If True, raise ValueError on mismatch; if False, warn
        
    Returns:
        Tuple of (is_valid, message)
        
    Raises:
        ValueError: If raise_error=True and arrays have different lengths
    """
    if method_names is None:
        method_names = list(predictions.keys())
    
    lengths = {name: len(predictions[name]) for name in method_names}
    unique_lengths = set(lengths.values())
    
    if len(unique_lengths) == 1:
        return True, f"All {len(method_names)} methods have matching length: {list(unique_lengths)[0]}"
    
    msg = "Paired sample length mismatch: " + ", ".join(
        f"{name}={length}" for name, length in lengths.items()
    )
    
    if raise_error:
        raise ValueError(msg)
    else:
        warnings.warn(msg)
        return False, msg


def synchronize_test_sets(
    predictions: Dict[str, np.ndarray],
    y_true: Optional[np.ndarray] = None,
) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Synchronize predictions across methods to shared valid samples.
    
    Creates intersection mask: samples with valid (non-NaN) predictions from ALL methods.
    This is critical for fair comparison (paired samples in DeLong test, etc).
    
    Args:
        predictions: Dictionary mapping method names to prediction arrays
        y_true: Optional true labels array (must match prediction length if provided)
        
    Returns:
        Tuple of (synchronized_predictions, valid_mask, y_true_synchronized)
        
    Raises:
        ValueError: If prediction arrays have mismatched lengths
    """
    if not predictions:
        raise ValueError("No predictions provided")
    
    # Validate array lengths
    lengths = {name: len(arr) for name, arr in predictions.items()}
    if len(set(lengths.values())) > 1:
        raise ValueError(
            f"Prediction arrays have mismatched lengths: {lengths}. "
            "All arrays must have same length for synchronization."
        )
    
    n_samples = len(list(predictions.values())[0])
    
    # Create valid mask: samples without NaN in ANY method
    valid_mask = np.ones(n_samples, dtype=bool)
    for method_name, preds in predictions.items():
        valid_mask &= ~np.isnan(preds)
    
    # Synchronize predictions
    synchronized = {}
    for method_name, preds in predictions.items():
        synchronized[method_name] = preds[valid_mask]
    
    # Synchronize y_true if provided
    y_true_sync = None
    if y_true is not None:
        if len(y_true) != n_samples:
            raise ValueError(
                f"y_true has length {len(y_true)} but predictions have length {n_samples}"
            )
        y_true_sync = y_true[valid_mask]
    
    return synchronized, valid_mask, y_true_sync


def report_missing_variables(
    dataframe: pd.DataFrame,
    required_variables: Dict[str, List[str]],
) -> Dict[str, pd.Series]:
    """Report missing values for required variables by method.
    
    Args:
        dataframe: Dataframe with patient data
        required_variables: Dict mapping method names to lists of required columns
        
    Returns:
        Dictionary mapping method names to missing value counts per variable
    """
    reports = {}
    
    for method, vars_list in required_variables.items():
        missing_counts = {}
        for var in vars_list:
            if var in dataframe.columns:
                n_missing = dataframe[var].isna().sum()
                if n_missing > 0:
                    missing_counts[var] = n_missing
        
        if missing_counts:
            reports[method] = pd.Series(missing_counts)
        else:
            reports[method] = pd.Series([], dtype=int)
    
    return reports


@dataclass
class MethodRequirements:
    """Requirements specification for a prediction method."""
    name: str
    required_variables: List[str]
    optional_variables: List[str] = None
    
    def __post_init__(self):
        if self.optional_variables is None:
            self.optional_variables = []


class TestSetValidator:
    """Validator for test set compatibility across multiple methods."""
    
    def __init__(self, 
                 dataframe: pd.DataFrame,
                 target_column: str = "muerte_inhospitalaria"):
        """Initialize validator.
        
        Args:
            dataframe: Test set dataframe
            target_column: Name of target/outcome column
        """
        self.dataframe = dataframe
        self.target_column = target_column
        self.n_rows = len(dataframe)
        self.n_events = (dataframe[target_column] == 1).sum() if target_column in dataframe.columns else 0
        
        # Track validation results
        self.method_compatibility: Dict[str, bool] = {}
        self.missing_summary: Dict[str, Dict] = {}
    
    def check_method_compatibility(
        self,
        method_names: List[str],
        requirements_dict: Dict[str, MethodRequirements],
    ) -> Dict[str, bool]:
        """Check if test set has required variables for each method.
        
        Args:
            method_names: List of method names to check
            requirements_dict: Dict mapping method names to MethodRequirements
            
        Returns:
            Dictionary mapping method names to compatibility boolean
        """
        for method_name in method_names:
            if method_name not in requirements_dict:
                self.method_compatibility[method_name] = False
                continue
            
            req = requirements_dict[method_name]
            
            # Check required variables
            has_all_required = all(
                var in self.dataframe.columns 
                for var in req.required_variables
            )
            
            # Check at least some optional variables if specified
            has_optional = True
            if req.optional_variables:
                has_optional = any(
                    var in self.dataframe.columns 
                    for var in req.optional_variables
                )
            
            self.method_compatibility[method_name] = has_all_required and has_optional
        
        return self.method_compatibility
    
    def summarize_compatibility(self) -> str:
        """Generate summary of method compatibility."""
        if not self.method_compatibility:
            return "No compatibility checks performed"
        
        compatible = [m for m, c in self.method_compatibility.items() if c]
        incompatible = [m for m, c in self.method_compatibility.items() if not c]
        
        msg = f"Test set compatibility ({self.n_rows} samples, {self.n_events} events):\n"
        msg += f"  Compatible: {', '.join(compatible) if compatible else 'None'}\n"
        msg += f"  Incompatible: {', '.join(incompatible) if incompatible else 'None'}"
        
        return msg


def create_intersection_mask(
    predictions_dict: Dict[str, np.ndarray],
    min_samples: int = 10,
    return_indices: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, List[int]]]:
    """Create mask for samples with valid predictions across all methods.
    
    Args:
        predictions_dict: Dictionary mapping method names to prediction arrays
        min_samples: Minimum number of samples required after filtering
        return_indices: If True, also return list of valid sample indices
        
    Returns:
        Valid mask (and indices if return_indices=True)
        
    Raises:
        ValueError: If result would have fewer than min_samples
    """
    n_samples = len(list(predictions_dict.values())[0])
    
    # Valid = non-NaN in ALL methods
    mask = np.ones(n_samples, dtype=bool)
    for preds in predictions_dict.values():
        mask &= ~np.isnan(preds)
    
    n_valid = mask.sum()
    if n_valid < min_samples:
        raise ValueError(
            f"Intersection of valid predictions would yield only {n_valid} samples, "
            f"but minimum {min_samples} required."
        )
    
    if return_indices:
        valid_indices = np.where(mask)[0].tolist()
        return mask, valid_indices
    
    return mask


def compare_masks(
    mask1: np.ndarray,
    mask2: np.ndarray,
    method1_name: str = "Mask 1",
    method2_name: str = "Mask 2",
) -> Dict[str, any]:
    """Compare two boolean masks (e.g., from different scale computations).
    
    Useful for detecting when different scales create different valid sample sets.
    
    Args:
        mask1, mask2: Boolean masks of same length
        method1_name, method2_name: Names for reporting
        
    Returns:
        Dictionary with comparison statistics
    """
    if len(mask1) != len(mask2):
        raise ValueError(f"Masks have different lengths: {len(mask1)} vs {len(mask2)}")
    
    # Overlap analysis
    both_true = (mask1 & mask2).sum()
    only_mask1 = (mask1 & ~mask2).sum()
    only_mask2 = (~mask1 & mask2).sum()
    both_false = (~mask1 & ~mask2).sum()
    
    return {
        "total_samples": len(mask1),
        "intersection": both_true,
        "only_" + method1_name.lower(): only_mask1,
        "only_" + method2_name.lower(): only_mask2,
        "both_invalid": both_false,
        "jaccard_similarity": both_true / (both_true + only_mask1 + only_mask2) if (both_true + only_mask1 + only_mask2) > 0 else 0,
    }


def validate_synchronization(
    original_predictions: Dict[str, np.ndarray],
    synchronized_predictions: Dict[str, np.ndarray],
    valid_mask: np.ndarray,
) -> SyncValidationReport:
    """Validate that synchronization was performed correctly.
    
    Args:
        original_predictions: Original prediction dictionaries
        synchronized_predictions: Synchronized predictions
        valid_mask: Mask used for synchronization
        
    Returns:
        SyncValidationReport with validation results
    """
    initial_size = len(list(original_predictions.values())[0])
    synchronized_size = len(list(synchronized_predictions.values())[0])
    n_removed = initial_size - synchronized_size
    removal_rate = n_removed / initial_size if initial_size > 0 else 0
    
    # Count missing per method
    missing_by_method = {}
    for method, orig_preds in original_predictions.items():
        n_nan = np.isnan(orig_preds).sum()
        missing_by_method[method] = int(n_nan)
    
    # Find max missing count (samples removed due to this method)
    max_missing = max(missing_by_method.values()) if missing_by_method else 0
    problematic_variables = {}
    for method, n_missing in missing_by_method.items():
        if n_missing > 0:
            problematic_variables[method] = int(n_missing)
    
    # Validation checks
    checks = [
        len(list(synchronized_predictions.values())[0]) == valid_mask.sum(),
        all(method in synchronized_predictions for method in original_predictions),
    ]
    
    is_valid = all(checks)
    
    if is_valid:
        message = (
            f"Synchronization successful: {initial_size} → {synchronized_size} samples "
            f"({removal_rate:.1%} removed). All {len(synchronized_predictions)} methods synchronized."
        )
    else:
        message = f"Synchronization validation FAILED. See details above."
    
    return SyncValidationReport(
        initial_size=initial_size,
        synchronized_size=synchronized_size,
        n_removed=n_removed,
        removal_rate=removal_rate,
        missing_by_method=missing_by_method,
        problematic_variables=problematic_variables,
        is_valid=is_valid,
        message=message,
    )
