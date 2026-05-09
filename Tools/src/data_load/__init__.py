"""Data loading and I/O module.

This module provides utilities for:
- Loading datasets from multiple formats (CSV, Excel, Parquet, Feather)
- Train/test splitting with various strategies
- Dataset metadata and information
- Data auditing and summarization
- Path utilities for managing processed data files
"""

from .loaders import (
    load_dataset,
    get_dataset_info,
    DatasetInfo,
    summarize_dataframe,
    select_feature_target,
    data_audit,
)
from .splitters import train_test_split, create_temporal_split, create_stratified_split
from .io_utils import detect_file_format, save_dataset, read_csv_with_encoding
from .path_utils import (
    get_timestamp,
    normalize_resampling_strategy,
    get_model_combination_key,
    split_model_combination_key,
    model_combination_exists,
    get_latest_file,
    get_latest_model,
    get_latest_model_by_task,
    get_latest_testset,
    get_latest_trainset,
    get_latest_cleaned_dataset,
    get_latest_plot,
    save_plot_with_overwrite,
    save_model_with_cleanup,
    save_dataset_with_timestamp,
    cleanup_old_testsets,
    get_all_model_types,
    extract_timestamp_from_filename,
    get_model_metadata,
    load_model_with_metadata,
)

__all__ = [
    # Loaders
    "load_dataset",
    "get_dataset_info",
    "DatasetInfo",
    "summarize_dataframe",
    "select_feature_target",
    "data_audit",
    # Splitters
    "train_test_split",
    "create_temporal_split",
    "create_stratified_split",
    # IO Utils
    "detect_file_format",
    "save_dataset",
    "read_csv_with_encoding",
    # Path utilities
    "get_timestamp",
    "normalize_resampling_strategy",
    "get_model_combination_key",
    "split_model_combination_key",
    "model_combination_exists",
    "get_latest_file",
    "get_latest_model",
    "get_latest_model_by_task",
    "get_latest_testset",
    "get_latest_trainset",
    "get_latest_cleaned_dataset",
    "get_latest_plot",
    "save_plot_with_overwrite",
    "save_model_with_cleanup",
    "save_dataset_with_timestamp",
    "cleanup_old_testsets",
    "get_all_model_types",
    "extract_timestamp_from_filename",
    "get_model_metadata",
    "load_model_with_metadata",
]
