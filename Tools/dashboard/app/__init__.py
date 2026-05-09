"""Dashboard app utilities package."""
from __future__ import annotations

from .config import apply_custom_css, configure_page
from .state import (
    clear_state,
    get_available_models,
    get_state,
    initialize_state,
    is_data_loaded,
    is_evaluated,
    is_trained,
    load_data,
    set_state,
)
from .ui_utils import (
    display_data_audit,
    display_dataframe_info,
    display_dataset_preview,
    display_metrics_table,
    display_model_list,
    format_metric_value,
    get_default_data_path,
    list_saved_models,
    list_saved_model_versions,
    sidebar_data_controls,
    sidebar_training_controls,
    train_models_with_progress,
)

__all__ = [
    # config
    "configure_page",
    "apply_custom_css",
    # state
    "initialize_state",
    "get_state",
    "set_state",
    "clear_state",
    "load_data",
    "get_available_models",
    "is_data_loaded",
    "is_trained",
    "is_evaluated",
    # ui_utils
    "display_dataframe_info",
    "display_dataset_preview",
    "display_data_audit",
    "get_default_data_path",
    "sidebar_data_controls",
    "sidebar_training_controls",
    "train_models_with_progress",
    "list_saved_models",
    "list_saved_model_versions",
    "display_model_list",
    "format_metric_value",
    "display_metrics_table",
]
