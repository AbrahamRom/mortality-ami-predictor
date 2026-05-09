"""Main data cleaning orchestration class.

This module provides the DataCleaner class that orchestrates all cleaning
operations and tracks metadata throughout the process.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Literal, Optional, Any, Tuple

import pandas as pd

from .metadata import VariableMetadata
from .imputation import (
    ImputationStrategy,
    impute_numeric_column,
    impute_categorical_column,
    impute_knn,
)
from .outliers import (
    OutlierMethod,
    OutlierTreatment,
    handle_column_outliers,
)
from .encoding import (
    EncodingStrategy,
    encode_categorical_column,
)
from .discretization import (
    DiscretizationStrategy,
    discretize_column,
)


@dataclass
class CleaningConfig:
    """Configuration for data cleaning pipeline.
    
    Attributes:
        numeric_imputation: Strategy for numeric imputation
        categorical_imputation: Strategy for categorical imputation
        knn_neighbors: Number of neighbors for KNN imputation
        constant_fill_numeric: Fill value for constant numeric imputation
        constant_fill_categorical: Fill value for constant categorical imputation
        custom_imputation_strategies: Dict mapping column names to specific imputation strategies
        custom_constant_values: Dict mapping column names to specific constant values for imputation
        columns_to_drop: List of column names to drop (e.g., due to high missing rate)
        outlier_method: Method for outlier detection (iqr, zscore, modified_zscore, isolation_forest, lof, percentile)
        iqr_multiplier: Multiplier for IQR method
        zscore_threshold: Threshold for Z-score method
        modified_zscore_threshold: Threshold for Modified Z-score method (MAD-based)
        outlier_contamination: Contamination rate for ML methods (Isolation Forest, LOF)
        lower_percentile: Lower percentile for percentile method
        upper_percentile: Upper percentile for percentile method
        lof_neighbors: Number of neighbors for LOF method
        outlier_treatment: How to treat detected outliers (cap, remove, transform, none)
        categorical_encoding: Encoding strategy for categoricals
        ordinal_categories: Category orders for ordinal encoding
        discretization_strategy: Strategy for discretizing numeric variables
        discretization_bins: Number of bins for discretization
        custom_bins: Custom bin edges per variable
        custom_labels: Custom labels for bins per variable
        custom_discretization_strategies: Dict mapping column names to specific discretization strategies
        custom_discretization_bins: Dict mapping column names to specific number of bins
        drop_duplicates: Whether to drop duplicate rows
        drop_fully_missing: Whether to drop fully missing columns
        drop_constant: Whether to drop constant columns
        constant_threshold: Threshold for considering column constant
    """
    # Imputation - Default to 'none' to preserve original data relationships
    numeric_imputation: str = "none"
    categorical_imputation: str = "none"
    knn_neighbors: int = 5
    constant_fill_numeric: float = 0.0
    constant_fill_categorical: str = "missing"
    custom_imputation_strategies: Dict[str, str] = field(default_factory=dict)
    custom_constant_values: Dict[str, Any] = field(default_factory=dict)
    columns_to_drop: List[str] = field(default_factory=list)  # Columns to drop due to high missing rate
    
    # Outliers - Default to 'none' to preserve original data relationships
    outlier_method: str = "none"
    iqr_multiplier: float = 1.5
    zscore_threshold: float = 3.0
    modified_zscore_threshold: float = 3.5  # For Modified Z-score (MAD)
    outlier_contamination: float = 0.1  # For Isolation Forest and LOF
    lower_percentile: float = 1.0  # For percentile method
    upper_percentile: float = 99.0  # For percentile method
    lof_neighbors: int = 20  # For LOF method
    outlier_treatment: str = "none"
    # Per-variable outlier configuration
    custom_outlier_methods: Dict[str, str] = field(default_factory=dict)  # Per-variable detection method
    custom_outlier_treatments: Dict[str, str] = field(default_factory=dict)  # Per-variable treatment
    custom_outlier_params: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # Per-variable parameters
    columns_skip_outliers: List[str] = field(default_factory=list)  # Columns to skip outlier detection
    
    # Encoding
    categorical_encoding: str = "label"
    ordinal_categories: Dict[str, List[str]] = field(default_factory=dict)
    custom_encoding_strategies: Dict[str, str] = field(default_factory=dict)  # Per-variable encoding type
    
    # Discretization
    discretization_strategy: str = "none"
    discretization_bins: int = 5
    custom_bins: Dict[str, List[float]] = field(default_factory=dict)
    custom_labels: Dict[str, List[str]] = field(default_factory=dict)
    custom_discretization_strategies: Dict[str, str] = field(default_factory=dict)
    custom_discretization_bins: Dict[str, int] = field(default_factory=dict)
    
    # General
    drop_duplicates: bool = True
    drop_fully_missing: bool = True
    drop_constant: bool = True
    constant_threshold: float = 0.95
    
    def to_dict(self) -> Dict:
        """Convert configuration to dictionary."""
        return asdict(self)


class DataCleaner:
    """Main data cleaning pipeline with metadata tracking.
    
    This class orchestrates all cleaning operations and maintains
    detailed metadata about transformations applied to each variable.
    """
    
    def __init__(self, config: Optional[CleaningConfig] = None):
        """Initialize cleaner with configuration.
        
        Args:
            config: Cleaning configuration. If None, uses defaults
        """
        self.config = config or CleaningConfig()
        self.metadata: Dict[str, VariableMetadata] = {}
        self.encoders: Dict[str, Any] = {}
    
    def fit_transform(
        self,
        df: pd.DataFrame,
        target_column: Optional[str] = None,
    ) -> pd.DataFrame:
        """Apply complete cleaning pipeline to DataFrame.
        
        Args:
            df: Input DataFrame
            target_column: Target column to preserve (not cleaned)
            
        Returns:
            Cleaned DataFrame
            
        Raises:
            ValueError: If DataFrame is empty
        """
        # Validate input
        if df.empty:
            raise ValueError("Cannot clean an empty DataFrame")
        
        df_clean = df.copy()
        
        # Store target column name for later (but DON'T separate it yet)
        # The target must be present during duplicate removal to maintain alignment
        self._target_column = target_column
        
        # 1. Identify column types (excluding target from cleaning)
        numeric_cols, categorical_cols = self._identify_column_types(df_clean)
        
        # Remove target from columns to clean (if present)
        if target_column and target_column in numeric_cols:
            numeric_cols.remove(target_column)
        if target_column and target_column in categorical_cols:
            categorical_cols.remove(target_column)
        
        # 2. Initialize metadata
        self._initialize_metadata(df_clean, numeric_cols, categorical_cols)
        
        # 3. Drop problematic columns
        df_clean = self._drop_problematic_columns(df_clean, numeric_cols, categorical_cols)
        
        # 4. Remove duplicates
        if self.config.drop_duplicates:
            df_clean = self._remove_duplicates(df_clean)
        
        # 5. Handle outliers (before imputation)
        df_clean = self._handle_outliers(df_clean, numeric_cols)
        
        # 6. Impute missing values
        df_clean = self._impute_missing(df_clean, numeric_cols, categorical_cols)
        
        # 7. Apply discretization (after imputation, before encoding)
        df_clean = self._apply_discretization(df_clean, numeric_cols)
        
        # 8. Encode categoricals
        df_clean = self._encode_categorical(df_clean, categorical_cols)
        
        # 9. Validate dtypes
        df_clean = self._validate_dtypes(df_clean)
        
        # 10. Update final metadata
        self._update_final_metadata(df_clean)
        
        # Target is already in df_clean (was never removed), so just return
        return df_clean
    
    def _identify_column_types(
        self,
        df: pd.DataFrame,
    ) -> Tuple[List[str], List[str]]:
        """Identify numeric and categorical columns."""
        import numpy as np
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
        
        # Detect numeric columns that are actually categorical
        for col in numeric_cols[:]:
            if df[col].nunique() <= 10 and df[col].notna().sum() > 0:
                categorical_cols.append(col)
                numeric_cols.remove(col)
        
        return numeric_cols, categorical_cols
    
    def _initialize_metadata(
        self,
        df: pd.DataFrame,
        numeric_cols: List[str],
        categorical_cols: List[str],
    ) -> None:
        """Initialize metadata for each variable."""
        for col in numeric_cols:
            series = df[col]
            self.metadata[col] = VariableMetadata(
                name=col,
                original_type=str(series.dtype),
                is_numerical=True,
                is_categorical=False,
                original_min=float(series.min()) if series.notna().any() else None,
                original_max=float(series.max()) if series.notna().any() else None,
                missing_count_original=int(series.isna().sum()),
                missing_percent_original=float(series.isna().mean() * 100),
            )
        
        for col in categorical_cols:
            series = df[col]
            unique_vals = series.dropna().unique().tolist()
            self.metadata[col] = VariableMetadata(
                name=col,
                original_type=str(series.dtype),
                is_numerical=False,
                is_categorical=True,
                unique_values=[str(v) for v in unique_vals[:50]],
                missing_count_original=int(series.isna().sum()),
                missing_percent_original=float(series.isna().mean() * 100),
            )
    
    def _drop_problematic_columns(
        self,
        df: pd.DataFrame,
        numeric_cols: List[str],
        categorical_cols: List[str],
    ) -> pd.DataFrame:
        """Drop fully missing, constant, or user-specified columns."""
        df_clean = df.copy()
        cols_to_drop = []
        
        # First, add columns explicitly marked for dropping (e.g., due to high missing rate)
        if self.config.columns_to_drop:
            for col in self.config.columns_to_drop:
                if col in df_clean.columns and col not in cols_to_drop:
                    cols_to_drop.append(col)
                    if col in self.metadata:
                        self.metadata[col].quality_flags.append("dropped_by_user")
        
        for col in df_clean.columns:
            if col in cols_to_drop:
                continue
                
            # Fully missing
            if self.config.drop_fully_missing and df_clean[col].isna().all():
                cols_to_drop.append(col)
                if col in self.metadata:
                    self.metadata[col].quality_flags.append("fully_missing")
                continue
            
            # Nearly constant
            if self.config.drop_constant:
                value_counts = df_clean[col].value_counts(dropna=True)
                if len(value_counts) > 0:
                    most_common_ratio = value_counts.iloc[0] / df_clean[col].notna().sum()
                    if most_common_ratio >= self.config.constant_threshold:
                        cols_to_drop.append(col)
                        if col in self.metadata:
                            self.metadata[col].quality_flags.append("nearly_constant")
        
        if cols_to_drop:
            df_clean = df_clean.drop(columns=cols_to_drop)
            for col in cols_to_drop:
                if col in numeric_cols:
                    numeric_cols.remove(col)
                if col in categorical_cols:
                    categorical_cols.remove(col)
        
        return df_clean
    
    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate rows."""
        n_before = len(df)
        df_clean = df.drop_duplicates().reset_index(drop=True)
        n_after = len(df_clean)
        
        if n_after < n_before:
            print(f"⚠️  Removed {n_before - n_after} duplicate rows")
        
        return df_clean
    
    def _handle_outliers(
        self,
        df: pd.DataFrame,
        numeric_cols: List[str],
    ) -> pd.DataFrame:
        """Detect and treat outliers in numeric columns.
        
        Supports both global settings and per-variable custom configuration
        through custom_outlier_methods, custom_outlier_treatments, and custom_outlier_params.
        """
        global_method = OutlierMethod(self.config.outlier_method)
        global_treatment = OutlierTreatment(self.config.outlier_treatment)
        
        df_clean = df.copy()
        
        for col in numeric_cols:
            if col not in df_clean.columns:
                continue
            
            # Check if column should be skipped
            if col in self.config.columns_skip_outliers:
                continue
            
            # Determine method and treatment for this column
            if col in self.config.custom_outlier_methods:
                col_method_str = self.config.custom_outlier_methods[col]
                if col_method_str == "none":
                    continue  # Skip this column
                method = OutlierMethod(col_method_str)
            else:
                method = global_method
                if method == OutlierMethod.NONE:
                    continue
            
            if col in self.config.custom_outlier_treatments:
                treatment = OutlierTreatment(self.config.custom_outlier_treatments[col])
            else:
                treatment = global_treatment
            
            # Get parameters - use custom if available, else global
            col_params = self.config.custom_outlier_params.get(col, {})
            
            iqr_mult = col_params.get('iqr_multiplier', self.config.iqr_multiplier)
            zscore_thresh = col_params.get('zscore_threshold', self.config.zscore_threshold)
            mod_zscore_thresh = col_params.get('modified_zscore_threshold', self.config.modified_zscore_threshold)
            contamination = col_params.get('contamination', self.config.outlier_contamination)
            lower_pct = col_params.get('lower_percentile', self.config.lower_percentile)
            upper_pct = col_params.get('upper_percentile', self.config.upper_percentile)
            lof_neighbors = col_params.get('lof_neighbors', self.config.lof_neighbors)
            
            result, n_outliers = handle_column_outliers(
                df_clean[col],
                method=method,
                treatment=treatment,
                iqr_multiplier=iqr_mult,
                zscore_threshold=zscore_thresh,
                modified_zscore_threshold=mod_zscore_thresh,
                contamination=contamination,
                lower_percentile=lower_pct,
                upper_percentile=upper_pct,
                lof_neighbors=lof_neighbors,
            )
            
            df_clean[col] = result
            
            if col in self.metadata:
                self.metadata[col].outliers_detected = n_outliers
                self.metadata[col].outliers_treated = n_outliers
                if n_outliers > 0:
                    self.metadata[col].quality_flags.append(
                        f"outliers_{method.value}_{treatment.value}_{n_outliers}"
                    )
        
        return df_clean
    
    def _impute_missing(
        self,
        df: pd.DataFrame,
        numeric_cols: List[str],
        categorical_cols: List[str],
    ) -> pd.DataFrame:
        """Impute missing values.
        
        This method supports both global imputation strategies and custom
        per-variable strategies via custom_imputation_strategies config.
        """
        df_clean = df.copy()
        
        # Handle KNN imputation separately (needs all columns at once)
        # Only for columns without custom strategies
        knn_cols = []
        if self.config.numeric_imputation == "knn":
            knn_cols = [
                col for col in numeric_cols 
                if col not in self.config.custom_imputation_strategies
            ]
            if knn_cols:
                df_clean = impute_knn(df_clean, knn_cols, self.config.knn_neighbors)
                for col in knn_cols:
                    if col in self.metadata:
                        self.metadata[col].imputation_method = "knn"
        
        # Impute numeric columns individually
        for col in numeric_cols:
            if col not in df_clean.columns or df_clean[col].isna().sum() == 0:
                continue
            
            # Skip if already handled by KNN
            if col in knn_cols:
                continue
            
            # Check for custom strategy for this column
            if col in self.config.custom_imputation_strategies:
                strategy_str = self.config.custom_imputation_strategies[col]
                strategy = ImputationStrategy(strategy_str)
                
                # Check for custom constant value
                fill_value = self.config.custom_constant_values.get(
                    col, 
                    self.config.constant_fill_numeric
                )
            else:
                # Use global strategy
                strategy = ImputationStrategy(self.config.numeric_imputation)
                fill_value = self.config.constant_fill_numeric
            
            df_clean[col] = impute_numeric_column(
                df_clean[col],
                strategy=strategy,
                fill_value=fill_value,
                knn_neighbors=self.config.knn_neighbors,
            )
            
            if col in self.metadata:
                self.metadata[col].imputation_method = strategy.value
        
        # Impute categorical columns
        for col in categorical_cols:
            if col not in df_clean.columns or df_clean[col].isna().sum() == 0:
                continue
            
            # Check for custom strategy for this column
            if col in self.config.custom_imputation_strategies:
                strategy_str = self.config.custom_imputation_strategies[col]
                strategy = ImputationStrategy(strategy_str)
                
                # Check for custom constant value
                fill_value = self.config.custom_constant_values.get(
                    col, 
                    self.config.constant_fill_categorical
                )
            else:
                # Use global strategy
                strategy = ImputationStrategy(self.config.categorical_imputation)
                fill_value = self.config.constant_fill_categorical
            
            df_clean[col] = impute_categorical_column(
                df_clean[col],
                strategy=strategy,
                fill_value=fill_value,
            )
            
            if col in self.metadata:
                self.metadata[col].imputation_method = strategy.value
        
        return df_clean
    
    def _apply_discretization(
        self,
        df: pd.DataFrame,
        numeric_cols: List[str],
    ) -> pd.DataFrame:
        """Apply discretization to numeric columns.
        
        This method supports both global discretization strategy and custom
        per-variable strategies via custom_discretization_strategies config.
        """
        # Check if any discretization is needed
        global_strategy = DiscretizationStrategy(self.config.discretization_strategy)
        if global_strategy == DiscretizationStrategy.NONE and not self.config.custom_discretization_strategies:
            return df
        
        df_clean = df.copy()
        
        for col in numeric_cols:
            if col not in df_clean.columns:
                continue
            
            # Check for custom strategy for this column
            if col in self.config.custom_discretization_strategies:
                strategy_str = self.config.custom_discretization_strategies[col]
                strategy = DiscretizationStrategy(strategy_str)
                
                # Check for custom bins and labels
                custom_bins = self.config.custom_bins.get(col, None)
                custom_labels = self.config.custom_labels.get(col, None)
                n_bins = self.config.custom_discretization_bins.get(col, self.config.discretization_bins)
                
            elif global_strategy != DiscretizationStrategy.NONE:
                # Use global strategy
                strategy = global_strategy
                custom_bins = self.config.custom_bins.get(col, None)
                custom_labels = self.config.custom_labels.get(col, None)
                n_bins = self.config.discretization_bins
            else:
                # No discretization for this column
                continue
            
            # Apply discretization
            try:
                discretized, bins, labels = discretize_column(
                    df_clean[col],
                    strategy=strategy,
                    n_bins=n_bins,
                    custom_bins=custom_bins,
                    custom_labels=custom_labels,
                    return_bins=True,
                )
                
                df_clean[col] = discretized
                
                # Update metadata
                if col in self.metadata:
                    self.metadata[col].discretization_bins = bins
                    self.metadata[col].discretization_labels = labels
                    
            except Exception as e:
                print(f"⚠️  Could not discretize column {col}: {e}")
                continue
        
        return df_clean
    
    def _encode_categorical(
        self,
        df: pd.DataFrame,
        categorical_cols: List[str],
    ) -> pd.DataFrame:
        """Encode categorical variables.
        
        Uses custom_encoding_strategies for per-variable encoding if specified,
        otherwise falls back to the global categorical_encoding strategy.
        """
        global_strategy = EncodingStrategy(self.config.categorical_encoding)
        
        df_clean = df.copy()
        
        for col in categorical_cols:
            if col not in df_clean.columns:
                continue
            
            # Determine strategy for this column (custom or global)
            if col in self.config.custom_encoding_strategies:
                col_strategy_str = self.config.custom_encoding_strategies[col]
                col_strategy = EncodingStrategy(col_strategy_str)
            else:
                col_strategy = global_strategy
            
            # Skip if no encoding
            if col_strategy == EncodingStrategy.NONE:
                continue
            
            # Convert to string
            df_clean[col] = df_clean[col].astype(str)
            
            ordinal_cats = self.config.ordinal_categories.get(col, None)
            
            df_clean, encoding_info = encode_categorical_column(
                df_clean,
                col,
                col_strategy,
                ordinal_categories=ordinal_cats,
            )
            
            if col in self.metadata:
                self.metadata[col].encoding_type = encoding_info['type']
                self.metadata[col].encoding_mapping = encoding_info.get('mapping', {})
        
        return df_clean
    
    def _validate_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and correct data types."""
        df_clean = df.copy()
        
        for col in df_clean.columns:
            if df_clean[col].dtype == 'object':
                try:
                    df_clean[col] = pd.to_numeric(df_clean[col], errors='ignore')
                except Exception:
                    pass
        
        return df_clean
    
    def _update_final_metadata(self, df: pd.DataFrame) -> None:
        """Update metadata with final information after cleaning."""
        import pandas as pd
        
        for col in df.columns:
            if col not in self.metadata:
                continue
            
            meta = self.metadata[col]
            meta.cleaned_type = str(df[col].dtype)
            
            if pd.api.types.is_numeric_dtype(df[col]):
                meta.cleaned_min = float(df[col].min()) if df[col].notna().any() else None
                meta.cleaned_max = float(df[col].max()) if df[col].notna().any() else None
            
            # Quality flags
            missing_final = df[col].isna().sum()
            if missing_final > 0:
                meta.quality_flags.append(f"missing_values_remain_{missing_final}")
            else:
                meta.quality_flags.append("no_missing_values")
    
    def save_metadata(self, filepath: Path) -> None:
        """Save metadata to JSON file, preserving user-edited fields.
        
        If the metadata file already exists, this method will:
        1. Load the existing metadata
        2. Preserve user-editable fields (description)
        3. Update computed fields (stats, types, etc.)
        4. Save the merged metadata
        
        Args:
            filepath: Output path for JSON file
        """
        # Load existing metadata if file exists
        existing_metadata = {}
        if Path(filepath).exists():
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    existing_metadata = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                # If file is corrupted or doesn't exist, start fresh
                pass
        
        # Convert current metadata to dict
        metadata_dict = {
            name: meta.to_dict() 
            for name, meta in self.metadata.items()
        }
        
        # Merge with existing metadata, preserving user-editable fields
        user_editable_fields = ['description']  # Add more fields here if needed
        
        for var_name, new_meta in metadata_dict.items():
            if var_name in existing_metadata:
                old_meta = existing_metadata[var_name]
                # Preserve user-editable fields
                for field in user_editable_fields:
                    if field in old_meta and old_meta[field]:
                        # Only preserve if user has added content
                        new_meta[field] = old_meta[field]
        
        # Add informational fields
        metadata_dict['_readme'] = "Este archivo contiene metadatos de las variables del dataset después de la limpieza."
        metadata_dict['_instructions'] = "Puedes editar manualmente el campo 'description' para agregar contexto clínico a cada variable."
        metadata_dict['_auto_generated'] = True
        metadata_dict['_timestamp'] = datetime.now().isoformat()
        metadata_dict['_note'] = "Los campos 'description' se preservan al regenerar. Los demás campos se actualizan automáticamente."
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(metadata_dict, f, indent=2, ensure_ascii=False)
    
    def save_config(self, filepath: Path) -> None:
        """Save configuration to JSON file.
        
        Args:
            filepath: Output path for JSON file
        """
        config_dict = self.config.to_dict()
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
    
    def load_metadata(self, filepath: Path) -> None:
        """Load metadata from JSON file.
        
        Args:
            filepath: Path to JSON file with metadata
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            metadata_dict = json.load(f)
        
        # Filter out special fields (those starting with '_')
        self.metadata = {
            name: VariableMetadata(**meta_data)
            for name, meta_data in metadata_dict.items()
            if not name.startswith('_')
        }
    
    def get_cleaning_report(self) -> Dict[str, Any]:
        """Generate cleaning summary report.
        
        Returns:
            Dictionary with cleaning statistics
        """
        return {
            "timestamp": datetime.now().isoformat(),
            "config": self.config.to_dict(),
            "variables_cleaned": len(self.metadata),
            "variables_with_outliers": sum(
                1 for m in self.metadata.values() if m.outliers_detected > 0
            ),
            "variables_imputed": sum(
                1 for m in self.metadata.values() if m.imputation_method is not None
            ),
            "variables_encoded": sum(
                1 for m in self.metadata.values() if m.encoding_type is not None
            ),
            "variables_discretized": sum(
                1 for m in self.metadata.values() if m.discretization_bins is not None
            ),
            "quality_issues": {
                name: meta.quality_flags
                for name, meta in self.metadata.items()
                if meta.quality_flags
            }
        }


def quick_clean(
    df: pd.DataFrame,
    target_column: Optional[str] = None,
    **kwargs
) -> Tuple[pd.DataFrame, DataCleaner]:
    """Quick convenience function for data cleaning.
    
    Args:
        df: DataFrame to clean
        target_column: Target column to preserve
        **kwargs: Arguments for CleaningConfig
        
    Returns:
        Tuple of (cleaned_df, cleaner_with_metadata)
    """
    config = CleaningConfig(**kwargs)
    cleaner = DataCleaner(config)
    df_clean = cleaner.fit_transform(df, target_column=target_column)
    
    return df_clean, cleaner
