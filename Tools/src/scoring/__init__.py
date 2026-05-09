"""Clinical scoring module."""

from .grace import GRACEScore
from .timi import TIMIScore
from .recuima import RECUIMAScorer, parse_complicaciones, compute_recuima_from_dataframe
from .registry import get_score, list_scores, register_score
from .score_data_manager import (
    ScoreDataConfig,
    load_original_dataset,
    extract_score_variables,
    save_testset_score_data,
    load_testset_score_data,
    check_score_data_availability,
    GRACE_REQUIRED_VARIABLES,
    RECUIMA_REQUIRED_VARIABLES,
    RECUIMA_ECG_LEADS,
    ALL_SCORE_VARIABLES,
    DEFAULT_ORIGINAL_DATASET_PATH,
)
from .variable_mapper import (
    VariableMapper,
    MappingResult,
    validate_mapped_dataframe,
)

__all__ = [
    "GRACEScore",
    "TIMIScore",
    "RECUIMAScorer",
    "parse_complicaciones",
    "compute_recuima_from_dataframe",
    "get_score",
    "list_scores",
    "register_score",
    # Score data management
    "ScoreDataConfig",
    "load_original_dataset",
    "extract_score_variables",
    "save_testset_score_data",
    "load_testset_score_data",
    "check_score_data_availability",
    "GRACE_REQUIRED_VARIABLES",
    "RECUIMA_REQUIRED_VARIABLES",
    "RECUIMA_ECG_LEADS",
    "ALL_SCORE_VARIABLES",
    "DEFAULT_ORIGINAL_DATASET_PATH",
    # Variable mapping
    "VariableMapper",
    "MappingResult",
    "validate_mapped_dataframe",
]
