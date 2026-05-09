"""Orchestrator for comprehensive ML vs Clinical Scales comparison.

Central hub coordinating:
- ML model prediction loading/computation
- Clinical scale (GRACE, TIMI, RECUIMA) batch computation
- Test set synchronization and validation
- Multi-method statistical comparison
- Unified report generation
"""
from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from ..prediction.predictor import Predictor, load_predictor
from ..scoring.grace import GRACEScore
from ..scoring.recuima import RECUIMAScorer
from ..scoring.timi import TIMISTEMIScore, TIMINSTEMIScore
from ..scoring.variable_mapper import VariableMapper
from ..scoring.score_converters import ScoreConverter
from .grace_comparison import compare_with_grace
from .recuima_comparison import compare_with_recuima
from .multi_method_comparison import compare_multiple_methods


@dataclass
class ScaleComputationStatus:
    """Status tracking for clinical scale computation."""
    scale_name: str
    n_valid: int
    n_missing: int
    missing_variables: List[str] = field(default_factory=list)
    missing_by_sample: Dict[int, List[str]] = field(default_factory=dict)  # Sample idx -> [missing vars]
    computation_time_ms: float = 0.0
    success: bool = True
    error_message: str = ""


@dataclass
class OrchestratorSummary:
    """Summary of orchestrator execution."""
    timestamp: str
    ml_model_name: str
    test_set_size: int
    synchronized_size: int
    n_events: int
    event_rate: float
    
    # Per-scale metrics
    scale_statuses: Dict[str, ScaleComputationStatus] = field(default_factory=dict)
    
    # Global comparison results
    ml_auc: float = 0.0
    grace_auc: float = 0.0
    recuima_auc: float = 0.0
    timi_stemi_auc: float = 0.0
    timi_nstemi_auc: float = 0.0
    
    # P-values from statistical tests
    grace_p_value: float = 1.0
    recuima_p_value: float = 1.0
    timi_stemi_p_value: float = 1.0
    timi_nstemi_p_value: float = 1.0
    friedman_p_value: float = 1.0
    
    # Messages
    warnings: List[str] = field(default_factory=list)
    notes: str = ""


@dataclass
class SynchronizationReport:
    """Report on test set synchronization across all methods."""
    original_size: int
    synchronized_size: int
    samples_removed: int
    removal_rate: float
    
    # Per-method validity
    method_validity: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # Format: {method_name: {valid_count, missing_count, missing_variables: [...], ...}}
    
    # Overall checks
    all_methods_valid: bool = True
    validation_warnings: List[str] = field(default_factory=list)
    
    def report(self) -> str:
        """Generate human-readable synchronization report."""
        lines = [
            "=" * 70,
            "SYNCHRONIZATION REPORT",
            "=" * 70,
            f"Original test set size: {self.original_size}",
            f"Synchronized size: {self.synchronized_size}",
            f"Samples removed: {self.samples_removed}",
            f"Removal rate: {self.removal_rate:.2%}",
            "",
            "Per-Method Validity:",
            "-" * 70,
        ]
        
        for method_name, validity_info in self.method_validity.items():
            valid_count = validity_info.get('valid_count', 0)
            missing_count = validity_info.get('missing_count', 0)
            missing_vars = validity_info.get('missing_variables', [])
            
            lines.append(f"\n  {method_name}:")
            lines.append(f"    Valid samples: {valid_count}")
            if missing_count > 0:
                lines.append(f"    Samples with missing variables: {missing_count}")
                if missing_vars:
                    lines.append(f"    Missing variables: {', '.join(missing_vars)}")
        
        if self.validation_warnings:
            lines.extend(["", "Warnings:", "-" * 70])
            for warning in self.validation_warnings:
                lines.append(f"  ⚠ {warning}")
        
        lines.append("=" * 70)
        return "\n".join(lines)


class ClinicalScaleMLOrchestrator:
    """Orchestrator for ML vs Clinical Scales comparison workflow.
    
    Responsibilities:
    1. Load/compute ML predictions
    2. Batch compute clinical scales
    3. Synchronize test sets (ensure same samples)
    4. Validate data integrity and paired samples
    5. Delegate to specialized comparison modules
    6. Generate unified summary report
    """
    
    def __init__(
        self,
        model: Optional[Union[Predictor, str]] = None,
        model_name: str = "XGBoost Mortality Predictor",
        test_df: Optional[pd.DataFrame] = None,
        target_col: str = "muerte_inhospitalaria",
        variable_mapper: Optional[VariableMapper] = None,
    ):
        """Initialize orchestrator.
        
        Args:
            model: Predictor instance or path to model file. If string, loads from path.
            model_name: Name for the ML model in reports
            test_df: Test dataframe. If None, must be provided to run_comparison()
            target_col: Name of target column (death outcome)
            variable_mapper: VariableMapper instance for column name normalization.
                           If None, creates a default mapper.
            
        Raises:
            ValueError: If model is invalid or test data missing when required
        """
        self.model_name = model_name
        self.target_col = target_col
        self.test_df = test_df
        self.variable_mapper = variable_mapper or VariableMapper()
        
        # Load model if path provided
        if isinstance(model, str):
            self.model = load_predictor(model_path=model, model_name=model_name)
        else:
            self.model = model
        
        # Initialize scorers
        self.grace = GRACEScore()
        self.recuima = RECUIMAScorer()
        self.timi_stemi = TIMISTEMIScore()
        self.timi_nstemi = TIMINSTEMIScore()
        
        # Results storage
        self._predictions: Dict[str, np.ndarray] = {}
        self._valid_mask: Optional[np.ndarray] = None
        self._comparison_results: Dict = {}
        self._sync_report: Optional[SynchronizationReport] = None
    
    # =========================================================================
    # PHASE 1: ML MODEL PREDICTIONS
    # =========================================================================
    
    def get_ml_predictions(
        self,
        X: Optional[pd.DataFrame] = None,
        use_probabilities: bool = True,
    ) -> np.ndarray:
        """Get ML model predictions.
        
        Args:
            X: Features (if None, uses test_df)
            use_probabilities: Return probabilities (True) or binary predictions (False)
            
        Returns:
            Prediction array
            
        Raises:
            ValueError: If no model or data available
        """
        if self.model is None:
            raise ValueError("No model loaded. Provide model in __init__ or load separately.")
        
        if X is None:
            if self.test_df is None:
                raise ValueError("No data provided. Pass X or set test_df in __init__")
            X = self.test_df
        
        if use_probabilities:
            preds = self.model.predict_proba(X)
        else:
            preds = self.model.predict(X)
        
        self._predictions["ML Model"] = preds
        return preds
    
    # =========================================================================
    # PHASE 2: CLINICAL SCALE COMPUTATION
    # =========================================================================
    
    def compute_all_scales(
        self,
        df: Optional[pd.DataFrame] = None,
        skip_timi: bool = False,
    ) -> Dict[str, ScaleComputationStatus]:
        """Batch compute all clinical scales.
        
        Args:
            df: Test dataframe (if None, uses self.test_df)
            skip_timi: Skip TIMI computation (for speed)
            
        Returns:
            Dictionary mapping scale names to computation status
            
        Raises:
            ValueError: If no dataframe available
        """
        if df is None:
            if self.test_df is None:
                raise ValueError("No dataframe provided")
            df = self.test_df.copy()
        else:
            df = df.copy()
        
        # Apply variable mapping to normalize column names across all scales
        # Each scale gets the mapped dataframe with standardized variable names
        mapped_dfs = {}
        mapping_warnings = []
        
        for score_type in ["grace", "timi_stemi", "timi_nstemi", "recuima"]:
            mapped_result = self.variable_mapper.map_dataframe(df, score_type)
            mapped_dfs[score_type] = mapped_result.mapped_df
            
            # Track any unmapped required variables
            if mapped_result.unmapped_required:
                mapping_warnings.append(
                    f"{score_type}: Missing required variables: {mapped_result.unmapped_required}"
                )
        
        statuses = {}
        
        # Dynamically get required variables from variable_mapper.SCORE_REQUIREMENTS
        # This ensures we use the same definitions as the score calculators
        
        # GRACE: required + optional variables
        grace_required = self.variable_mapper.SCORE_REQUIREMENTS.get("grace", [])
        grace_optional = self.variable_mapper.SCORE_REQUIREMENTS.get("grace_optional", [])
        statuses["GRACE"] = self._compute_scale(
            mapped_dfs["grace"], self.grace, "GRACE",
            required_vars=grace_required,
            optional_vars=grace_optional
        )
        
        # RECUIMA: required + optional variables
        recuima_required = self.variable_mapper.SCORE_REQUIREMENTS.get("recuima", [])
        recuima_optional = self.variable_mapper.SCORE_REQUIREMENTS.get("recuima_optional", [])
        statuses["RECUIMA"] = self._compute_scale(
            mapped_dfs["recuima"], self.recuima, "RECUIMA",
            required_vars=recuima_required,
            optional_vars=recuima_optional
        )
        
        # TIMI (if not skipped)
        if not skip_timi:
            # TIMI-STEMI: required + optional variables
            timi_stemi_required = self.variable_mapper.SCORE_REQUIREMENTS.get("timi_stemi", [])
            timi_stemi_optional = self.variable_mapper.SCORE_REQUIREMENTS.get("timi_stemi_optional", [])
            statuses["TIMI-STEMI"] = self._compute_scale(
                mapped_dfs["timi_stemi"], self.timi_stemi, "TIMI-STEMI",
                required_vars=timi_stemi_required,
                optional_vars=timi_stemi_optional
            )
            
            # TIMI-NSTEMI: required + optional variables
            timi_nstemi_required = self.variable_mapper.SCORE_REQUIREMENTS.get("timi_nstemi", [])
            timi_nstemi_optional = self.variable_mapper.SCORE_REQUIREMENTS.get("timi_nstemi_optional", [])
            statuses["TIMI-NSTEMI"] = self._compute_scale(
                mapped_dfs["timi_nstemi"], self.timi_nstemi, "TIMI-NSTEMI",
                required_vars=timi_nstemi_required,
                optional_vars=timi_nstemi_optional
            )
        
        # Store mapping warnings if any
        if mapping_warnings:
            statuses["mapping_warnings"] = mapping_warnings
        
        return statuses
    
    def _compute_scale(
        self,
        df: pd.DataFrame,
        scorer,
        scale_name: str,
        required_vars: List[str],
        optional_vars: List[str] = None,
    ) -> ScaleComputationStatus:
        """Compute a single scale for all samples.
        
        Args:
            df: Dataframe with patient data
            scorer: Scorer object with compute() method
            scale_name: Name for reporting
            required_vars: Variables required for scale computation
            optional_vars: Variables that are optional for computation (default: [])
            
        Returns:
            ScaleComputationStatus with computation results
        """
        if optional_vars is None:
            optional_vars = []
        import time
        start_time = time.time()
        
        status = ScaleComputationStatus(scale_name=scale_name, n_valid=0, n_missing=0)
        predictions = []
        valid_indices = []
        
        for idx, row in df.iterrows():
            try:
                # Check for missing required variables (Phase 3 - Missing Variable Tracking)
                missing_vars = []
                for req_var in required_vars:
                    if req_var not in row.index or pd.isna(row[req_var]):
                        missing_vars.append(req_var)
                
                if missing_vars:
                    status.n_missing += 1
                    status.missing_by_sample[idx] = missing_vars
                    # Add to summary if not already there
                    for var in missing_vars:
                        if var not in status.missing_variables:
                            status.missing_variables.append(var)
                    continue  # Skip this sample
                
                # STEP 4: Filter row to contain ONLY designated variables (variable isolation)
                # This ensures each score only receives its required and optional variables
                allowed_vars = set(required_vars + optional_vars)
                row_dict = row.to_dict()
                row_filtered = {k: v for k, v in row_dict.items() if k in allowed_vars}
                
                # Try to compute score (scorer returns dict or result object)
                result = scorer.compute_safe(**row_filtered)
                
                # Extract raw score from result
                if isinstance(result, dict):
                    if not result.get("success", False):
                        # Computation failed - track the error
                        status.n_missing += 1
                        error_msg = "; ".join(result.get("errors", ["Unknown error"]))[:100]
                        status.missing_by_sample[idx] = [f"Computation error: {error_msg}"]
                        continue
                    raw_score = result.get("score", np.nan)
                else:
                    # Handle result objects with score attribute
                    if not getattr(result, "success", False):
                        status.n_missing += 1
                        status.missing_by_sample[idx] = ["Computation failed"]
                        continue
                    raw_score = getattr(result, "score", np.nan)
                
                if np.isnan(raw_score):
                    status.n_missing += 1
                    status.missing_by_sample[idx] = ["Score computation returned NaN"]
                    continue
                
                # PHASE 1: Convert score to probability using ScoreConverter
                if scale_name == "GRACE":
                    probability = ScoreConverter.grace_score_to_probability(raw_score)
                elif scale_name == "TIMI-STEMI":
                    probability = ScoreConverter.timi_stemi_score_to_probability(raw_score)
                elif scale_name == "TIMI-NSTEMI":
                    probability = ScoreConverter.timi_nstemi_score_to_probability(raw_score)
                elif scale_name == "RECUIMA":
                    probability = ScoreConverter.recuima_score_to_probability(raw_score)
                else:
                    probability = raw_score  # Fallback (shouldn't happen)
                
                if not np.isnan(probability):
                    predictions.append(probability)
                    valid_indices.append(idx)
                    status.n_valid += 1
                else:
                    status.n_missing += 1
                    status.missing_by_sample[idx] = ["Score to probability conversion returned NaN"]
                    
            except (KeyError, AttributeError, ValueError, TypeError) as e:
                # Track missing variables or other errors
                status.n_missing += 1
                error_str = str(e)[:100]
                status.missing_by_sample[idx] = [f"Exception: {error_str}"]
                if error_str not in status.missing_variables:
                    status.missing_variables.append(error_str)
        
        if status.n_valid == 0:
            status.success = False
            status.error_message = f"No valid predictions computed for {scale_name}"
        
        status.computation_time_ms = (time.time() - start_time) * 1000
        
        # Store as numpy array (now containing PROBABILITIES, not raw scores)
        preds_array = np.full(len(df), np.nan)
        preds_array[valid_indices] = predictions
        self._predictions[scale_name] = preds_array
        
        return status
    
    # =========================================================================
    # PHASE 3: SYNCHRONIZATION & VALIDATION
    # =========================================================================
    
    def synchronize_predictions(self) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Synchronize all predictions to use only valid samples.
        
        Creates a mask of samples with valid predictions from ALL methods
        (ML + GRACE + RECUIMA + TIMI). Ensures fair comparison with paired samples.
        Also generates a detailed SynchronizationReport for transparency.
        
        Returns:
            Tuple of (valid_mask, synchronized_predictions_dict)
            
        Raises:
            ValueError: If predictions not computed yet
        """
        if not self._predictions:
            raise ValueError("No predictions computed yet. Call get_ml_predictions() and compute_all_scales() first.")
        
        original_size = len(self._predictions[list(self._predictions.keys())[0]])
        
        # Create mask: sample is valid if it has valid prediction from ALL methods
        valid_mask = np.ones(original_size, dtype=bool)
        
        for method_name, preds in self._predictions.items():
            valid_mask &= ~np.isnan(preds)
        
        self._valid_mask = valid_mask
        
        # Synchronize: keep only valid samples
        synchronized = {}
        for method_name, preds in self._predictions.items():
            synchronized[method_name] = preds[valid_mask]
        
        n_removed = (~valid_mask).sum()
        synchronized_size = valid_mask.sum()
        
        # Generate SynchronizationReport for detailed visibility
        sync_report = SynchronizationReport(
            original_size=original_size,
            synchronized_size=synchronized_size,
            samples_removed=n_removed,
            removal_rate=n_removed / original_size if original_size > 0 else 0.0,
        )
        
        # Populate per-method validity information
        for method_name, preds in self._predictions.items():
            valid_count = (~np.isnan(preds[valid_mask])).sum()
            missing_count = (np.isnan(preds)).sum()
            
            # Find which samples have missing values for this method
            missing_mask = np.isnan(preds)
            
            sync_report.method_validity[method_name] = {
                'valid_count': int(valid_count),
                'missing_count': int(missing_count),
                'missing_samples': int((~valid_mask & ~missing_mask).sum()),  # Samples failing sync
            }
        
        # Add warnings if removal rate is high
        if sync_report.removal_rate > 0.1:  # More than 10% removed
            sync_report.validation_warnings.append(
                f"High removal rate ({sync_report.removal_rate:.1%}): consider data quality review"
            )
        
        # Check if all methods have sufficient valid samples
        for method_name, validity_info in sync_report.method_validity.items():
            if validity_info['missing_count'] > 0:
                sync_report.all_methods_valid = False
                sync_report.validation_warnings.append(
                    f"{method_name}: {validity_info['missing_count']} samples with missing values"
                )
        
        self._sync_report = sync_report
        
        if n_removed > 0:
            warnings.warn(
                f"Removed {n_removed} samples with missing predictions. "
                f"Synchronized test set size: {synchronized_size}. "
                f"Call get_sync_report() for detailed breakdown."
            )
        
        return valid_mask, synchronized
    
    def get_sync_report(self) -> Optional[SynchronizationReport]:
        """Get the most recent synchronization report.
        
        Returns:
            SynchronizationReport if synchronize_predictions() was called, else None
        """
        return self._sync_report
    
    # =========================================================================
    # PHASE 3: MISSING VARIABLE REPORTING
    # =========================================================================
    
    def get_missing_variable_report(self, scale_status: ScaleComputationStatus) -> str:
        """Generate detailed report of missing variables per sample.
        
        Args:
            scale_status: ScaleComputationStatus from compute_all_scales()
            
        Returns:
            Human-readable report of missing variables
        """
        report = f"\n{'='*80}\n"
        report += f"MISSING VARIABLE REPORT: {scale_status.scale_name}\n"
        report += f"{'='*80}\n"
        report += f"Valid samples: {scale_status.n_valid} / {scale_status.n_valid + scale_status.n_missing}\n"
        report += f"Failed samples: {scale_status.n_missing}\n"
        
        if not scale_status.missing_by_sample:
            report += "\n✓ All samples computed successfully!\n"
            report += f"{'='*80}\n"
            return report
        
        # Group failures by variable/reason
        failure_count = {}
        for idx, missing_vars in scale_status.missing_by_sample.items():
            for var in missing_vars:
                failure_count[var] = failure_count.get(var, 0) + 1
        
        report += f"\nTop reasons for failure ({len(failure_count)} types):\n"
        for reason, count in sorted(failure_count.items(), key=lambda x: -x[1])[:10]:
            pct = (count / scale_status.n_missing) * 100 if scale_status.n_missing > 0 else 0
            report += f"  • {reason}: {count} samples ({pct:.1f}%)\n"
        
        if len(scale_status.missing_by_sample) <= 20:
            report += f"\nDetailed breakdown (all {len(scale_status.missing_by_sample)} failed samples):\n"
            for idx, missing_vars in sorted(scale_status.missing_by_sample.items()):
                report += f"  Sample {idx}: {', '.join(missing_vars)}\n"
        else:
            report += f"\nDetailed breakdown (first 20 of {len(scale_status.missing_by_sample)} failed samples):\n"
            for idx, missing_vars in list(sorted(scale_status.missing_by_sample.items()))[:20]:
                report += f"  Sample {idx}: {', '.join(missing_vars)}\n"
            report += f"  ... ({len(scale_status.missing_by_sample) - 20} more)\n"
        
        report += f"{'='*80}\n"
        return report
    
    # =========================================================================
    # PHASE 4: COMPARISON & STATISTICAL TESTS
    # =========================================================================
    
    def _find_stemi_column(self, df: pd.DataFrame) -> Optional[str]:
        """Find the STEMI indicator column in the dataframe.
        
        Searches for common STEMI column names:
            - "anterior" (1=anterior STEMI, 0=non-anterior)
            - "stemi_type" (categorical or binary)
            - "type_of_stemi" (categorical)
            - "type_of_mi" (categorical)
            - "nstemi_stemi_classification" (categorical)
            - "st_elevation" (boolean)
            
        Returns:
            Column name if found, None otherwise
        """
        candidate_names = [
            "anterior", "stemi_type", "type_of_stemi", "type_of_mi",
            "nstemi_stemi_classification", "st_elevation"
        ]
        
        for col in candidate_names:
            if col in df.columns:
                return col
        return None
    
    def _run_comparison_for_subgroup(
        self,
        subgroup_df: pd.DataFrame,
        valid_mask: np.ndarray,
        sync_preds: Dict[str, np.ndarray],
        subgroup_name: str,
        alpha: float = 0.05,
        skip_timi: bool = False,
    ) -> Dict[str, Any]:
        """Run comparison for a specific patient subgroup.
        
        Args:
            subgroup_df: Dataframe filtered to subgroup
            valid_mask: Boolean mask for synchronized samples
            sync_preds: Synchronized predictions (full dataset)
            subgroup_name: Human-readable subgroup name (\"STEMI\" or \"NSTEMI\")
            alpha: Significance level
            skip_timi: Skip TIMI comparisons
            
        Returns:
            Dictionary with comparison results for subgroup
        """
        # Filter to subgroup within valid mask
        subgroup_valid = valid_mask.copy()
        all_indices = np.arange(len(self.test_df))
        subgroup_indices = subgroup_df.index.values
        
        # Mark indices outside subgroup as invalid
        mask_in_subgroup = np.isin(all_indices, subgroup_indices)
        subgroup_valid = subgroup_valid & mask_in_subgroup
        
        if not np.any(subgroup_valid):
            return {}  # No valid samples in subgroup
        
        # Extract subgroup predictions
        subgroup_preds = {}
        for method_name, preds in sync_preds.items():
            subgroup_preds[method_name] = preds[subgroup_valid]
        
        y_true = self.test_df[self.target_col].values[subgroup_valid].astype(int)
        
        # Run comparisons for subgroup
        results = {}
        
        # GRACE comparison
        if "GRACE" in subgroup_preds:
            results["grace_vs_ml"] = compare_with_grace(
                y_true=y_true,
                y_pred_model=subgroup_preds["ML Model"],
                y_pred_grace=subgroup_preds["GRACE"],
                model_name=self.model_name,
                alpha=alpha,
            )
        
        # RECUIMA comparison
        if "RECUIMA" in subgroup_preds:
            results["recuima_vs_ml"] = compare_with_recuima(
                y_true=y_true,
                y_pred_model=subgroup_preds["ML Model"],
                y_pred_recuima=subgroup_preds["RECUIMA"],
                model_name=self.model_name,
                alpha=alpha,
            )
        
        # TIMI comparison (subgroup-specific)
        if not skip_timi:
            if subgroup_name == "STEMI" and "TIMI-STEMI" in subgroup_preds:
                from .timi_comparison import compare_with_timi
                results["timi_vs_ml"] = compare_with_timi(
                    y_true=y_true,
                    y_pred_model=subgroup_preds["ML Model"],
                    y_pred_timi=subgroup_preds["TIMI-STEMI"],
                    model_name=self.model_name,
                    timi_variant="STEMI",
                    alpha=alpha,
                )
            elif subgroup_name == "NSTEMI" and "TIMI-NSTEMI" in subgroup_preds:
                from .timi_comparison import compare_with_timi
                results["timi_vs_ml"] = compare_with_timi(
                    y_true=y_true,
                    y_pred_model=subgroup_preds["ML Model"],
                    y_pred_timi=subgroup_preds["TIMI-NSTEMI"],
                    model_name=self.model_name,
                    timi_variant="NSTEMI",
                    alpha=alpha,
                )
        
        # Multi-method comparison for subgroup
        methods_to_compare = {
            "ML Model": subgroup_preds["ML Model"],
            "GRACE": subgroup_preds["GRACE"],
            "RECUIMA": subgroup_preds["RECUIMA"],
        }
        if subgroup_name == "STEMI" and "TIMI-STEMI" in subgroup_preds:
            methods_to_compare["TIMI"] = subgroup_preds["TIMI-STEMI"]
        elif subgroup_name == "NSTEMI" and "TIMI-NSTEMI" in subgroup_preds:
            methods_to_compare["TIMI"] = subgroup_preds["TIMI-NSTEMI"]
        
        results["multiple_methods"] = compare_multiple_methods(
            y_true=y_true,
            predictions=methods_to_compare,
            alpha=alpha,
            correction="holm",
        )
        
        return results
    
    def compare_all(
        self,
        alpha: float = 0.05,
        skip_timi: bool = False,
        subgroup_analysis: bool = True,
    ) -> Dict[str, Any]:
        """Run all pairwise and global comparisons.
        
        If subgroup_analysis is True, separates STEMI/NSTEMI patients before
        applying TIMI-specific scores. Otherwise, runs as before (global only).
        
        Args:
            alpha: Significance level for statistical tests
            skip_timi: Skip TIMI comparisons
            subgroup_analysis: Separate STEMI/NSTEMI patients for TIMI analysis
            
        Returns:
            Dictionary with comparison results:
                - If subgroup_analysis: {global: {...}, STEMI: {...}, NSTEMI: {...}}
                - Otherwise: {grace_vs_ml, recuima_vs_ml, multiple_methods, ...}
        """
        if self.test_df is None:
            raise ValueError("Test dataframe required")
        
        # Get synchronized predictions and target
        valid_mask, sync_preds = self.synchronize_predictions()
        
        results = {}
        results["global"] = {}
        
        # Global comparisons (all patients, regardless of STEMI status)
        y_true = self.test_df[self.target_col].values[valid_mask].astype(int)
        
        results["global"]["grace_vs_ml"] = compare_with_grace(
            y_true=y_true,
            y_pred_model=sync_preds["ML Model"],
            y_pred_grace=sync_preds["GRACE"],
            model_name=self.model_name,
            alpha=alpha,
        )
        
        results["global"]["recuima_vs_ml"] = compare_with_recuima(
            y_true=y_true,
            y_pred_model=sync_preds["ML Model"],
            y_pred_recuima=sync_preds["RECUIMA"],
            model_name=self.model_name,
            alpha=alpha,
        )
        
        methods_to_compare = {
            "ML Model": sync_preds["ML Model"],
            "GRACE": sync_preds["GRACE"],
            "RECUIMA": sync_preds["RECUIMA"],
        }
        
        if not skip_timi and "TIMI-STEMI" in sync_preds:
            methods_to_compare["TIMI-STEMI"] = sync_preds["TIMI-STEMI"]
        if not skip_timi and "TIMI-NSTEMI" in sync_preds:
            methods_to_compare["TIMI-NSTEMI"] = sync_preds["TIMI-NSTEMI"]
        
        results["global"]["multiple_methods"] = compare_multiple_methods(
            y_true=y_true,
            predictions=methods_to_compare,
            alpha=alpha,
            correction="holm",
        )
        
        # Subgroup-specific comparisons (STEMI/NSTEMI separation)
        if subgroup_analysis and not skip_timi:
            stemi_col = self._find_stemi_column(self.test_df)
            
            if stemi_col:
                # Find STEMI and NSTEMI patients
                stemi_df = self.test_df[self.test_df[stemi_col] == 1]
                nstemi_df = self.test_df[self.test_df[stemi_col] == 0]
                
                if len(stemi_df) > 0:
                    results["STEMI"] = self._run_comparison_for_subgroup(
                        stemi_df, valid_mask, sync_preds, "STEMI", alpha, skip_timi
                    )
                
                if len(nstemi_df) > 0:
                    results["NSTEMI"] = self._run_comparison_for_subgroup(
                        nstemi_df, valid_mask, sync_preds, "NSTEMI", alpha, skip_timi
                    )
        
        self._comparison_results = results
        return results
    
    # =========================================================================
    # PHASE 5: REPORT GENERATION
    # =========================================================================
    
    def generate_summary(self) -> OrchestratorSummary:
        """Generate summary of orchestration run.
        
        Returns:
            OrchestratorSummary object with key metrics and statistics
        """
        if not self._predictions:
            raise ValueError("No predictions computed yet")
        
        valid_mask = self._valid_mask if self._valid_mask is not None else np.ones(len(self.test_df), dtype=bool)
        y_true = self.test_df[self.target_col].values[valid_mask].astype(int)
        
        summary = OrchestratorSummary(
            timestamp=datetime.now().isoformat(),
            ml_model_name=self.model_name,
            test_set_size=len(self.test_df),
            synchronized_size=valid_mask.sum(),
            n_events=int(y_true.sum()),
            event_rate=float(y_true.mean()),
        )
        
        # Add comparison results if available
        if self._comparison_results:
            if "grace_vs_ml" in self._comparison_results:
                result = self._comparison_results["grace_vs_ml"]
                summary.grace_auc = result.grace_auc
                summary.grace_p_value = result.auc_p_value
            
            if "recuima_vs_ml" in self._comparison_results:
                result = self._comparison_results["recuima_vs_ml"]
                summary.recuima_auc = result.recuima_auc
                summary.recuima_p_value = result.auc_p_value
            
            if "multiple_methods" in self._comparison_results:
                result = self._comparison_results["multiple_methods"]
                summary.friedman_p_value = result.friedman_p_value
                # Extract AUCs from result
                if hasattr(result, 'aucs') and isinstance(result.aucs, dict):
                    summary.ml_auc = result.aucs.get("ML Model", 0.0)
            
            # TIMI results if available
            if "timi_stemi_vs_ml" in self._comparison_results:
                result = self._comparison_results["timi_stemi_vs_ml"]
                summary.timi_stemi_auc = result.timi_auc if hasattr(result, 'timi_auc') else 0.0
                summary.timi_stemi_p_value = result.auc_p_value
            
            if "timi_nstemi_vs_ml" in self._comparison_results:
                result = self._comparison_results["timi_nstemi_vs_ml"]
                summary.timi_nstemi_auc = result.timi_auc if hasattr(result, 'timi_auc') else 0.0
                summary.timi_nstemi_p_value = result.auc_p_value
        
        return summary
    
    def export_results(
        self,
        output_dir: Union[str, Path] = ".",
        format: str = "csv",
    ) -> Dict[str, Path]:
        """Export all predictions and comparison results.
        
        Args:
            output_dir: Directory to save files
            format: Output format ("csv", "json", "both")
            
        Returns:
            Dictionary mapping result names to output file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        exported = {}
        
        # Export predictions
        if self._predictions:
            pred_df = pd.DataFrame(self._predictions)
            
            if format in ("csv", "both"):
                csv_path = output_dir / "predictions.csv"
                pred_df.to_csv(csv_path, index=False)
                exported["predictions_csv"] = csv_path
            
            if format in ("json", "both"):
                json_path = output_dir / "predictions.json"
                pred_df.to_json(json_path, orient="records")
                exported["predictions_json"] = json_path
        
        # Export comparison results summary
        if self._comparison_results:
            summary = self.generate_summary()
            summary_dict = {
                "timestamp": summary.timestamp,
                "ml_model_name": summary.ml_model_name,
                "test_set_size": summary.test_set_size,
                "synchronized_size": summary.synchronized_size,
                "n_events": summary.n_events,
                "event_rate": summary.event_rate,
                "grace_auc": summary.grace_auc,
                "grace_p_value": summary.grace_p_value,
                "recuima_auc": summary.recuima_auc,
                "recuima_p_value": summary.recuima_p_value,
                "friedman_p_value": summary.friedman_p_value,
            }
            
            if format in ("csv", "both"):
                import json
                summary_csv = output_dir / "comparison_summary.json"
                summary_csv.write_text(json.dumps(summary_dict, indent=2))
                exported["summary"] = summary_csv
        
        return exported
    
    # =========================================================================
    # CONVENIENCE: END-TO-END PIPELINE
    # =========================================================================
    
    def run_complete_comparison(
        self,
        X_test: Optional[pd.DataFrame] = None,
        y_test: Optional[np.ndarray] = None,
        skip_timi: bool = False,
        alpha: float = 0.05,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> OrchestratorSummary:
        """Run complete pipeline in one call: predict → compute scales → compare → report.
        
        Args:
            X_test: Test features (optional, uses self.test_df if None)
            y_test: Test labels (optional, uses self.test_df target if None)
            skip_timi: Skip TIMI computation for speed
            alpha: Significance level
            output_dir: If provided, export results to this directory
            
        Returns:
            OrchestratorSummary with results
        """
        # Set test data if provided
        if X_test is not None and y_test is not None:
            self.test_df = X_test.copy()
            self.test_df[self.target_col] = y_test
        elif X_test is not None:
            self.test_df = X_test.copy()
        
        # Execute pipeline
        self.get_ml_predictions()
        self.compute_all_scales(skip_timi=skip_timi)
        self.compare_all(skip_timi=skip_timi, alpha=alpha)
        summary = self.generate_summary()
        
        # Export if requested
        if output_dir:
            self.export_results(output_dir=output_dir)
        
        return summary
