"""Global configuration for the ML project.

Only one external input is required: DATASET_PATH. This can be set via
an environment variable, CLI argument, or directly in this variable.

Never hardcode other data paths in the codebase.
"""
from __future__ import annotations
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional


RANDOM_SEED: int = 42
ROOT_DIR = Path(__file__).parent.parent


@dataclass
class ProjectConfig:
    """Project-wide configuration parameters.

    Attributes:
        dataset_path: Path to the dataset file (.csv, .parquet, etc.).
        experiment_tracker: Name of experiment tracker ("mlflow" or "wandb").
        tracking_uri: Tracking URI for MLflow or W&B entity/project string.
        target_column: Binary target for in-hospital mortality (0/1).
        arrhythmia_column: Optional binary target for ventricular arrhythmia (0/1).
        regression_target: Optional continuous target (e.g., length_of_stay).
        
        # Data cleaning and EDA paths
        data_dir: Path to DATA directory
        cleaned_data_dir: Path to store cleaned datasets
        metadata_path: Path to variable metadata JSON
        preprocessing_config_path: Path to preprocessing configuration JSON
        eda_cache_path: Path to EDA analysis cache
    """

    dataset_path: str = os.environ.get("DATASET_PATH", "")
    experiment_tracker: str = os.environ.get("EXPERIMENT_TRACKER", "mlflow")
    tracking_uri: Optional[str] = os.environ.get("TRACKING_URI")
    target_column: str = os.environ.get("TARGET_COLUMN", "mortality_inhospital")
    arrhythmia_column: Optional[str] = os.environ.get("ARRHYTHMIA_COLUMN", "ventricular_arrhythmia")
    regression_target: Optional[str] = os.environ.get("REGRESSION_TARGET", None)
    
    # New processed directory structure
    processed_dir: str = os.environ.get("PROCESSED_DIR", str(ROOT_DIR / "processed"))
    cleaned_datasets_dir: str = os.environ.get("CLEANED_DATASETS_DIR", str(ROOT_DIR / "processed/cleaned_datasets"))
    plots_dir: str = os.environ.get("PLOTS_DIR", str(ROOT_DIR / "processed/plots"))
    models_dir: str = os.environ.get("MODELS_DIR", str(ROOT_DIR / "processed/models"))
    testsets_dir: str = os.environ.get("TESTSETS_DIR", str(ROOT_DIR / "processed/models/testsets"))
    metadata_path: str = os.environ.get("METADATA_PATH", str(ROOT_DIR / "processed/variable_metadata.json"))
    preprocessing_config_path: str = os.environ.get("PREPROCESSING_CONFIG_PATH", str(ROOT_DIR / "processed/preprocessing_config.json"))
    eda_cache_path: str = os.environ.get("EDA_CACHE_PATH", str(ROOT_DIR / "processed/eda_cache.pkl"))
    
    # Legacy paths (kept for backwards compatibility during migration)
    data_dir: str = os.environ.get("DATA_DIR", str(ROOT_DIR.parent / "DATA"))
    cleaned_data_dir: str = os.environ.get("CLEANED_DATA_DIR", str(ROOT_DIR / "processed/cleaned_datasets"))


CONFIG = ProjectConfig()


def validate_config(cfg: ProjectConfig) -> None:
    """Validate minimal config is provided and raise helpful errors.

    Args:
        cfg: ProjectConfig
    """
    if not cfg.dataset_path:
        raise ValueError(
            "DATASET_PATH is not set. Set env var DATASET_PATH or pass --data to scripts."
        )
