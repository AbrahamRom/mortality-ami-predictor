"""AutoML Configuration Module.

Provides configuration classes and presets for AutoML experiments.
Supports both auto-sklearn and FLAML configurations.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from ..config import CONFIG, RANDOM_SEED


class AutoMLPreset(Enum):
    """Predefined AutoML configuration presets."""
    
    QUICK = "quick"           # Fast exploration (5-10 min)
    BALANCED = "balanced"     # Good balance (30-60 min)
    HIGH_PERFORMANCE = "high_performance"  # Best results (2-4 hours)
    OVERNIGHT = "overnight"   # Exhaustive search (8+ hours)
    CUSTOM = "custom"         # User-defined settings


@dataclass
class AutoMLConfig:
    """Configuration for AutoML experiments.
    
    Attributes:
        preset: Predefined configuration preset
        time_left_for_this_task: Total time budget in seconds
        per_run_time_limit: Max time per model evaluation
        memory_limit: Memory limit in MB (None = auto)
        ensemble_size: Number of models in ensemble
        ensemble_nbest: Number of best models to consider for ensemble
        metric: Optimization metric
        include_estimators: List of estimators to include (None = all)
        exclude_estimators: List of estimators to exclude
        include_preprocessors: List of preprocessors to include (None = all)
        exclude_preprocessors: List of preprocessors to exclude
        resampling_strategy: CV strategy for evaluation
        n_jobs: Number of parallel jobs
        random_state: Random seed
        tmp_folder: Temporary folder for auto-sklearn
        output_folder: Output folder for results
        delete_tmp_folder_after_terminate: Whether to cleanup
        initial_configurations_via_metalearning: Number of initial configs
        smac_scenario_args: Additional SMAC arguments
    """
    
    # Basic settings
    preset: AutoMLPreset = AutoMLPreset.BALANCED
    time_left_for_this_task: int = 3600  # 1 hour default
    per_run_time_limit: int = 360  # 6 minutes per model
    memory_limit: Optional[int] = 8192  # 8GB default
    
    # Ensemble settings
    ensemble_size: int = 50
    ensemble_nbest: int = 50
    
    # Optimization
    metric: str = "roc_auc"
    resampling_strategy: str = "cv"
    resampling_strategy_arguments: Dict[str, Any] = field(default_factory=lambda: {"folds": 5})
    
    # Model selection
    include_estimators: Optional[List[str]] = None
    exclude_estimators: Optional[List[str]] = None
    include_preprocessors: Optional[List[str]] = None
    exclude_preprocessors: Optional[List[str]] = None
    
    # Execution
    n_jobs: int = -1
    random_state: int = RANDOM_SEED
    
    # Paths
    tmp_folder: Optional[str] = None
    output_folder: Optional[str] = None
    delete_tmp_folder_after_terminate: bool = True
    
    # Meta-learning
    initial_configurations_via_metalearning: int = 25
    
    # Advanced SMAC settings
    smac_scenario_args: Optional[Dict[str, Any]] = None
    
    # FLAML-specific settings
    flaml_time_budget: Optional[int] = None  # Overrides time_left_for_this_task for FLAML
    flaml_estimator_list: Optional[List[str]] = None
    flaml_custom_hp: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Initialize paths if not set."""
        if self.tmp_folder is None:
            self.tmp_folder = str(Path(CONFIG.processed_dir) / "automl_tmp")
        if self.output_folder is None:
            self.output_folder = str(Path(CONFIG.processed_dir) / "automl_output")
        if self.flaml_time_budget is None:
            self.flaml_time_budget = self.time_left_for_this_task
    
    @classmethod
    def from_preset(cls, preset: AutoMLPreset | str) -> "AutoMLConfig":
        """Create configuration from a preset.
        
        Args:
            preset: Preset name or enum value
            
        Returns:
            AutoMLConfig instance
        """
        if isinstance(preset, str):
            preset = AutoMLPreset(preset)
        
        presets = {
            AutoMLPreset.QUICK: {
                "preset": AutoMLPreset.QUICK,
                "time_left_for_this_task": 300,  # 5 minutes
                "per_run_time_limit": 60,
                "ensemble_size": 10,
                "ensemble_nbest": 10,
                "initial_configurations_via_metalearning": 10,
                "resampling_strategy_arguments": {"folds": 3},
            },
            AutoMLPreset.BALANCED: {
                "preset": AutoMLPreset.BALANCED,
                "time_left_for_this_task": 3600,  # 1 hour
                "per_run_time_limit": 360,
                "ensemble_size": 50,
                "ensemble_nbest": 50,
                "initial_configurations_via_metalearning": 25,
                "resampling_strategy_arguments": {"folds": 5},
            },
            AutoMLPreset.HIGH_PERFORMANCE: {
                "preset": AutoMLPreset.HIGH_PERFORMANCE,
                "time_left_for_this_task": 14400,  # 4 hours
                "per_run_time_limit": 600,
                "ensemble_size": 100,
                "ensemble_nbest": 100,
                "initial_configurations_via_metalearning": 50,
                "resampling_strategy_arguments": {"folds": 10},
            },
            AutoMLPreset.OVERNIGHT: {
                "preset": AutoMLPreset.OVERNIGHT,
                "time_left_for_this_task": 28800,  # 8 hours
                "per_run_time_limit": 900,
                "ensemble_size": 200,
                "ensemble_nbest": 200,
                "initial_configurations_via_metalearning": 100,
                "resampling_strategy_arguments": {"folds": 10},
            },
        }
        
        if preset == AutoMLPreset.CUSTOM:
            return cls()
        
        return cls(**presets[preset])
    
    def to_autosklearn_kwargs(self) -> Dict[str, Any]:
        """Convert to auto-sklearn AutoSklearnClassifier/Regressor kwargs.
        
        Returns:
            Dictionary of keyword arguments
        """
        kwargs = {
            "time_left_for_this_task": self.time_left_for_this_task,
            "per_run_time_limit": self.per_run_time_limit,
            "ensemble_size": self.ensemble_size,
            "ensemble_nbest": self.ensemble_nbest,
            "metric": self._get_autosklearn_metric(),
            "resampling_strategy": self.resampling_strategy,
            "resampling_strategy_arguments": self.resampling_strategy_arguments,
            "n_jobs": self.n_jobs,
            "seed": self.random_state,
            "tmp_folder": self.tmp_folder,
            "output_folder": self.output_folder,
            "delete_tmp_folder_after_terminate": self.delete_tmp_folder_after_terminate,
            "initial_configurations_via_metalearning": self.initial_configurations_via_metalearning,
        }
        
        if self.memory_limit is not None:
            kwargs["memory_limit"] = self.memory_limit
        
        if self.include_estimators is not None:
            kwargs["include"] = {"classifier": self.include_estimators}
        
        if self.exclude_estimators is not None:
            kwargs["exclude"] = {"classifier": self.exclude_estimators}
        
        if self.smac_scenario_args is not None:
            kwargs["smac_scenario_args"] = self.smac_scenario_args
        
        return kwargs
    
    def to_flaml_kwargs(self) -> Dict[str, Any]:
        """Convert to FLAML AutoML kwargs.
        
        Returns:
            Dictionary of keyword arguments
        """
        kwargs = {
            "time_budget": self.flaml_time_budget or self.time_left_for_this_task,
            "metric": self._get_flaml_metric(),
            "task": "classification",  # Will be overridden for regressors
            "n_jobs": self.n_jobs,
            "seed": self.random_state,
            "ensemble": self.ensemble_size > 1,
            "verbose": 1,
        }
        
        if self.flaml_estimator_list is not None:
            kwargs["estimator_list"] = self.flaml_estimator_list
        elif self.include_estimators is not None:
            # Map auto-sklearn estimator names to FLAML
            kwargs["estimator_list"] = self._map_estimators_to_flaml(self.include_estimators)
        
        if self.flaml_custom_hp is not None:
            kwargs["custom_hp"] = self.flaml_custom_hp
        
        return kwargs
    
    def _get_autosklearn_metric(self):
        """Get auto-sklearn metric object."""
        try:
            import autosklearn.metrics as metrics
            
            metric_map = {
                "roc_auc": metrics.roc_auc,
                "accuracy": metrics.accuracy,
                "balanced_accuracy": metrics.balanced_accuracy,
                "f1": metrics.f1,
                "precision": metrics.precision,
                "recall": metrics.recall,
                "log_loss": metrics.log_loss,
            }
            
            return metric_map.get(self.metric, metrics.roc_auc)
        except ImportError:
            return self.metric
    
    def _get_flaml_metric(self) -> str:
        """Get FLAML metric name."""
        # FLAML uses same metric names mostly
        return self.metric
    
    def _map_estimators_to_flaml(self, estimators: List[str]) -> List[str]:
        """Map auto-sklearn estimator names to FLAML names."""
        mapping = {
            "random_forest": "rf",
            "extra_trees": "extra_tree",
            "gradient_boosting": "xgboost",
            "adaboost": "catboost",
            "mlp": "lgbm",
            "sgd": "lgbm",
            "libsvm_svc": "lgbm",
            "liblinear_svc": "lgbm",
            "k_nearest_neighbors": "kneighbor",
            "decision_tree": "extra_tree",
            "xgradient_boosting": "xgboost",
        }
        
        flaml_estimators = []
        for est in estimators:
            if est.lower() in mapping:
                flaml_estimators.append(mapping[est.lower()])
            else:
                flaml_estimators.append(est.lower())
        
        return list(set(flaml_estimators))  # Remove duplicates


# Available estimators for reference
AUTOSKLEARN_CLASSIFIERS: Set[str] = {
    "adaboost",
    "bernoulli_nb",
    "decision_tree",
    "extra_trees",
    "gaussian_nb",
    "gradient_boosting",
    "k_nearest_neighbors",
    "lda",
    "liblinear_svc",
    "libsvm_svc",
    "mlp",
    "multinomial_nb",
    "passive_aggressive",
    "qda",
    "random_forest",
    "sgd",
}

AUTOSKLEARN_REGRESSORS: Set[str] = {
    "adaboost",
    "ard_regression",
    "decision_tree",
    "extra_trees",
    "gaussian_process",
    "gradient_boosting",
    "k_nearest_neighbors",
    "liblinear_svr",
    "libsvm_svr",
    "mlp",
    "random_forest",
    "sgd",
}

FLAML_ESTIMATORS: Set[str] = {
    "lgbm",
    "xgboost",
    "xgb_limitdepth",
    "catboost",
    "rf",
    "extra_tree",
    "kneighbor",
    "lrl1",
    "lrl2",
}


def get_automl_config(
    preset: str | AutoMLPreset = AutoMLPreset.BALANCED,
    **overrides
) -> AutoMLConfig:
    """Get AutoML configuration with optional overrides.
    
    Args:
        preset: Configuration preset name
        **overrides: Parameter overrides
        
    Returns:
        AutoMLConfig instance
    """
    config = AutoMLConfig.from_preset(preset)
    
    for key, value in overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    return config
