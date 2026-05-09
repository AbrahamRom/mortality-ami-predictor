"""Model metadata management system.

This module provides classes and functions to manage metadata associated with
trained models, including training data, hyperparameters, performance metrics,
and provenance information.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import yaml


@dataclass
class DatasetMetadata:
    """Metadata about datasets used in model training."""
    
    train_set_path: str
    test_set_path: str
    train_samples: int
    test_samples: int
    n_features: int
    target_column: str
    class_distribution_train: Dict[str, float]
    class_distribution_test: Dict[str, float]
    feature_names: List[str]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> DatasetMetadata:
        """Create from dictionary."""
        return cls(**data)


@dataclass
class TrainingMetadata:
    """Metadata about the training process."""
    
    training_date: str
    training_duration_seconds: float
    cv_strategy: str  # e.g., "RepeatedStratifiedKFold(n_splits=10, n_repeats=10)"
    n_cv_folds: int
    n_cv_repeats: int
    total_cv_runs: int
    scoring_metric: str
    preprocessing_config: Dict[str, Any]
    random_seed: int
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> TrainingMetadata:
        """Create from dictionary."""
        return cls(**data)


@dataclass
class PerformanceMetrics:
    """Performance metrics from cross-validation."""
    
    mean_score: float
    std_score: float
    min_score: float
    max_score: float
    all_scores: List[float]
    cv_scores_per_fold: Optional[List[float]] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> PerformanceMetrics:
        """Create from dictionary."""
        return cls(**data)


@dataclass
class ModelMetadata:
    """Complete metadata for a trained model.
    
    This class stores all relevant information about a model including:
    - Model identification and versioning
    - Training data and configuration
    - Hyperparameters
    - Performance metrics
    - Provenance information
    
    Example:
        >>> metadata = ModelMetadata(
        ...     model_name="dtree",
        ...     model_type="DecisionTreeClassifier",
        ...     task="mortality",
        ...     version="1.0.0",
        ...     dataset=dataset_meta,
        ...     training=training_meta,
        ...     hyperparameters={"max_depth": 5},
        ...     performance=perf_metrics
        ... )
        >>> metadata.save("model_dtree_20241031.metadata.json")
    """
    
    # Model identification
    model_name: str  # Short name: "dtree", "knn", "xgb"
    model_type: str  # Full class name: "DecisionTreeClassifier"
    task: str  # "mortality", "arrhythmia"
    version: str = "1.0.0"
    
    # Timestamps
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    model_file_path: Optional[str] = None
    
    # Associated data
    dataset: Optional[DatasetMetadata] = None
    training: Optional[TrainingMetadata] = None
    
    # Model configuration
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    
    # Performance
    performance: Optional[PerformanceMetrics] = None
    
    # Statistical comparison results (if applicable)
    statistical_comparison: Optional[Dict[str, Any]] = None
    
    # Learning curve results (if available)
    learning_curve_path: Optional[str] = None
    
    # Additional metadata
    tags: List[str] = field(default_factory=list)
    notes: str = ""
    author: str = "AutoML System"
    
    def to_dict(self) -> Dict:
        """Convert metadata to dictionary.
        
        Returns:
            Dictionary representation of metadata
        """
        data = asdict(self)
        
        # Convert nested dataclasses
        if self.dataset:
            data['dataset'] = self.dataset.to_dict()
        if self.training:
            data['training'] = self.training.to_dict()
        if self.performance:
            data['performance'] = self.performance.to_dict()
        
        # Make hyperparameters JSON-serializable
        data['hyperparameters'] = self._make_serializable(data['hyperparameters'])
        
        # Make statistical_comparison JSON-serializable if present
        if data.get('statistical_comparison'):
            data['statistical_comparison'] = self._make_serializable(data['statistical_comparison'])
        
        return data
    
    @staticmethod
    def _make_serializable(obj: Any) -> Any:
        """Convert object to JSON-serializable format.
        
        Args:
            obj: Object to convert
            
        Returns:
            JSON-serializable version of object
        """
        import numpy as np
        
        # Handle None
        if obj is None:
            return None
        
        # Handle basic types
        if isinstance(obj, (str, int, float, bool)):
            return obj
        
        # Handle numpy types
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        
        # Handle dictionaries
        if isinstance(obj, dict):
            return {str(k): ModelMetadata._make_serializable(v) for k, v in obj.items()}
        
        # Handle lists and tuples
        if isinstance(obj, (list, tuple)):
            return [ModelMetadata._make_serializable(item) for item in obj]
        
        # Handle objects with __dict__ (like sklearn estimators)
        if hasattr(obj, '__class__'):
            # For sklearn objects and other classes, return string representation
            class_name = obj.__class__.__name__
            module_name = obj.__class__.__module__
            
            # Try to get a simple string representation
            if hasattr(obj, 'get_params'):
                # For sklearn-like objects
                try:
                    params = obj.get_params()
                    # Recursively make params serializable
                    serialized_params = ModelMetadata._make_serializable(params)
                    return {
                        '__class__': class_name,
                        '__module__': module_name,
                        'params': serialized_params
                    }
                except Exception:
                    pass
            
            # Fallback to string representation
            return f"{module_name}.{class_name}"
        
        # Fallback: convert to string
        return str(obj)
    
    @classmethod
    def from_dict(cls, data: Dict) -> ModelMetadata:
        """Create metadata from dictionary.
        
        Args:
            data: Dictionary with metadata
            
        Returns:
            ModelMetadata instance
        """
        # Extract nested objects
        dataset_data = data.pop('dataset', None)
        training_data = data.pop('training', None)
        performance_data = data.pop('performance', None)
        
        # Create nested objects
        dataset = DatasetMetadata.from_dict(dataset_data) if dataset_data else None
        training = TrainingMetadata.from_dict(training_data) if training_data else None
        performance = PerformanceMetrics.from_dict(performance_data) if performance_data else None
        
        return cls(
            dataset=dataset,
            training=training,
            performance=performance,
            **data
        )
    
    def save(self, path: Union[str, Path], format: str = "json") -> Path:
        """Save metadata to file.
        
        Args:
            path: Path where to save metadata
            format: File format ("json" or "yaml")
            
        Returns:
            Path where metadata was saved
        """
        path = Path(path)
        data = self.to_dict()
        
        if format == "json":
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        elif format == "yaml":
            with open(path, 'w', encoding='utf-8') as f:
                yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
        else:
            raise ValueError(f"Unsupported format: {format}. Use 'json' or 'yaml'")
        
        return path
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> ModelMetadata:
        """Load metadata from file.
        
        Args:
            path: Path to metadata file
            
        Returns:
            ModelMetadata instance
        """
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Metadata file not found: {path}")
        
        # Detect format from extension
        if path.suffix in ['.json', '.metadata']:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        elif path.suffix in ['.yaml', '.yml']:
            with open(path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
        
        return cls.from_dict(data)
    
    def get_summary(self) -> str:
        """Get a human-readable summary of the metadata.
        
        Returns:
            Formatted string with metadata summary
        """
        lines = [
            "=" * 70,
            f"Model Metadata: {self.model_name} ({self.model_type})",
            "=" * 70,
            f"Task: {self.task}",
            f"Version: {self.version}",
            f"Created: {self.created_at}",
            f"Author: {self.author}",
            ""
        ]
        
        if self.dataset:
            lines.extend([
                "Dataset Information:",
                f"  Train samples: {self.dataset.train_samples}",
                f"  Test samples: {self.dataset.test_samples}",
                f"  Features: {self.dataset.n_features}",
                f"  Target: {self.dataset.target_column}",
                ""
            ])
        
        if self.training:
            lines.extend([
                "Training Information:",
                f"  Date: {self.training.training_date}",
                f"  Duration: {self.training.training_duration_seconds:.2f}s",
                f"  CV Strategy: {self.training.cv_strategy}",
                f"  Total runs: {self.training.total_cv_runs}",
                ""
            ])
        
        if self.hyperparameters:
            lines.extend(["Hyperparameters:"])
            for key, value in self.hyperparameters.items():
                lines.append(f"  {key}: {value}")
            lines.append("")
        
        if self.performance:
            lines.extend([
                "Performance Metrics:",
                f"  Mean {self.training.scoring_metric if self.training else 'score'}: "
                f"{self.performance.mean_score:.4f} ± {self.performance.std_score:.4f}",
                f"  Range: [{self.performance.min_score:.4f}, {self.performance.max_score:.4f}]",
                ""
            ])
        
        if self.notes:
            lines.extend([
                "Notes:",
                f"  {self.notes}",
                ""
            ])
        
        lines.append("=" * 70)
        
        return "\n".join(lines)
    
    def __str__(self) -> str:
        """String representation."""
        return f"ModelMetadata(name={self.model_name}, task={self.task}, version={self.version})"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return self.get_summary()


def create_metadata_from_training(
    model_name: str,
    model_type: str,
    task: str,
    model_file_path: str,
    train_set_path: str,
    test_set_path: str,
    train_df,
    test_df,
    target_column: str,
    hyperparameters: Dict[str, Any],
    cv_results: Dict[str, Any],
    training_config: Dict[str, Any],
    learning_curve_path: Optional[str] = None,
    statistical_comparison: Optional[Dict] = None,
    notes: str = "",
    actual_feature_names: Optional[List[str]] = None,
) -> ModelMetadata:
    """Create ModelMetadata from training results.
    
    Convenience function to create metadata after model training.
    
    Args:
        model_name: Short model name (e.g., "dtree")
        model_type: Full class name (e.g., "DecisionTreeClassifier")
        task: Task name (e.g., "mortality")
        model_file_path: Path to saved model file
        train_set_path: Path to training set
        test_set_path: Path to test set
        train_df: Training DataFrame
        test_df: Test DataFrame
        target_column: Name of target column
        hyperparameters: Model hyperparameters
        cv_results: Cross-validation results
        training_config: Training configuration
        learning_curve_path: Optional path to learning curve plot
        statistical_comparison: Optional statistical comparison results
        notes: Additional notes
        actual_feature_names: ACTUAL features from trained model (from feature_names_in_)
        
    Returns:
        ModelMetadata instance
    """
    import numpy as np
    
    # Dataset metadata
    y_train = train_df[target_column]
    y_test = test_df[target_column]
    
    train_dist = y_train.value_counts(normalize=True).to_dict()
    test_dist = y_test.value_counts(normalize=True).to_dict()
    
    # Convert numpy int64 keys to regular ints for JSON serialization
    train_dist = {str(k): float(v) for k, v in train_dist.items()}
    test_dist = {str(k): float(v) for k, v in test_dist.items()}
    
    # Use ACTUAL feature names from trained model if provided, otherwise fallback to DataFrame columns
    if actual_feature_names is not None:
        feature_names_to_store = actual_feature_names
        n_features = len(actual_feature_names)
    else:
        feature_names_to_store = train_df.drop(columns=[target_column]).columns.tolist()
        n_features = len(train_df.columns) - 1  # Exclude target
    
    dataset_meta = DatasetMetadata(
        train_set_path=str(train_set_path),
        test_set_path=str(test_set_path),
        train_samples=len(train_df),
        test_samples=len(test_df),
        n_features=n_features,
        target_column=target_column,
        class_distribution_train=train_dist,
        class_distribution_test=test_dist,
        feature_names=feature_names_to_store
    )
    
    # Training metadata
    training_meta = TrainingMetadata(
        training_date=datetime.now().isoformat(),
        training_duration_seconds=training_config.get('duration', 0.0),
        cv_strategy=training_config.get('cv_strategy', 'RepeatedStratifiedKFold'),
        n_cv_folds=training_config.get('n_splits', 10),
        n_cv_repeats=training_config.get('n_repeats', 10),
        total_cv_runs=training_config.get('total_runs', 100),
        scoring_metric=training_config.get('scoring', 'roc_auc'),
        preprocessing_config=training_config.get('preprocessing', {}),
        random_seed=training_config.get('random_seed', 42)
    )
    
    # Performance metrics
    all_scores = cv_results.get('all_scores', [])
    performance_meta = PerformanceMetrics(
        mean_score=float(cv_results.get('mean_score', 0.0)),
        std_score=float(cv_results.get('std_score', 0.0)),
        min_score=float(np.min(all_scores)) if all_scores else 0.0,
        max_score=float(np.max(all_scores)) if all_scores else 0.0,
        all_scores=[float(s) for s in all_scores]
    )
    
    # Create complete metadata
    metadata = ModelMetadata(
        model_name=model_name,
        model_type=model_type,
        task=task,
        model_file_path=str(model_file_path),
        dataset=dataset_meta,
        training=training_meta,
        hyperparameters=hyperparameters,
        performance=performance_meta,
        learning_curve_path=str(learning_curve_path) if learning_curve_path else None,
        statistical_comparison=statistical_comparison,
        notes=notes
    )
    
    return metadata
