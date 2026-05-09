"""Classification models for ML experiments.

This module provides factory functions for creating classification models
with predefined hyperparameter grids, including AutoML configurations.
"""
from __future__ import annotations

from typing import Dict, Tuple, Any

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
import inspect

try:
    from xgboost import XGBClassifier  # type: ignore
except ImportError:
    XGBClassifier = None  # type: ignore

try:
    from lightgbm import LGBMClassifier  # type: ignore
except ImportError:
    LGBMClassifier = None  # type: ignore


def make_classifiers() -> Dict[str, Tuple[object, Dict]]:
    """Create dictionary of classifiers and their parameter grids.
    
    Returns:
        Dictionary mapping model_name -> (model, param_grid)
        where param_grid is for RandomizedSearchCV
    """
    models: Dict[str, Tuple[object, Dict]] = {}
    
    # K-Nearest Neighbors
    models["knn"] = (
        KNeighborsClassifier(),
        {
            "n_neighbors": [3, 5, 7, 11],
            "weights": ["uniform", "distance"],
        },
    )
    
    # Calibrated Logistic Regression
    logreg = LogisticRegression(
        max_iter=500,
        tol=1e-3,
        solver="lbfgs",
        class_weight="balanced"
    )
    
    # Handle sklearn API changes (estimator vs base_estimator)
    sig = inspect.signature(CalibratedClassifierCV)
    if "estimator" in sig.parameters:
        calib = CalibratedClassifierCV(estimator=logreg, method="sigmoid", cv=2)
        grid = {"estimator__C": [0.01, 0.1, 1.0, 10.0]}
    else:
        calib = CalibratedClassifierCV(base_estimator=logreg, method="sigmoid", cv=2)
        grid = {"base_estimator__C": [0.01, 0.1, 1.0, 10.0]}
    
    models["logreg"] = (calib, grid)
    
    # Decision Tree
    models["dtree"] = (
        DecisionTreeClassifier(random_state=42, class_weight="balanced"),
        {
            "max_depth": [3, 5, 7, None],
            "min_samples_split": [2, 10, 20],
            "min_samples_leaf": [1, 5, 10],
        },
    )

    # Random Forest (Robust to overfitting and imbalance)
    models["rf"] = (
        RandomForestClassifier(
            random_state=42,
            class_weight="balanced",  # Handle 1:10 imbalance
            n_jobs=-1
        ),
        {
            "n_estimators": [100, 200, 300],
            "max_depth": [5, 10, 15, None],  # Limit depth to prevent overfitting
            "min_samples_split": [5, 10, 20],
            "min_samples_leaf": [2, 5, 10],  # Higher leaf size for regularization
            "max_features": ["sqrt", "log2"],
        },
    )
    
    # XGBoost (if available)
    if XGBClassifier is not None:
        models["xgb"] = (
            XGBClassifier(
                n_estimators=200,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                eval_metric="logloss",
                random_state=42,
                tree_method="hist",
                scale_pos_weight=1.0,
                n_jobs=-1,
            ),
            {
                "max_depth": [3, 5, 7],
                "min_child_weight": [1, 3, 5],
                "gamma": [0, 0.1, 0.5],
            },
        )

        # Balanced XGBoost for imbalanced datasets (High Recall)
        # Ratio calculation: 2819 (negative) / 274 (positive) ~= 10.28
        models["xgb_balanced"] = (
        XGBClassifier(
            n_estimators=309,
            learning_rate=0.29609308954455615,
            subsample=0.8625805397403512,
            colsample_bylevel=0.2971842185320908,
            colsample_bytree=0.8622641959799742,
            min_child_weight=1.787260636319972,
            reg_alpha=0.38257752773041476,
            reg_lambda=385.81547152378926,
            max_leaves=1658,
            grow_policy='lossguide',          # necesario para max_leaves
            eval_metric="logloss",
            random_state=42,
            tree_method="hist",
            n_jobs=-1,
        ),
        # Si ya no necesitas el grid, puedes dejar {} o eliminarlo
        {}
    )
    
    # LightGBM (if available)
    if LGBMClassifier is not None:
        models["lgbm"] = (
            LGBMClassifier(
                n_estimators=400,
                learning_rate=0.03,
                subsample=0.9,
                colsample_bytree=0.9,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
                verbosity=-1,
                min_data_in_leaf=20,
                min_split_gain=0.0,
            ),
            {
                "num_leaves": [31, 63, 127],
                "max_depth": [-1, 6, 10],
            },
        )
    
    return models


def make_automl_classifiers() -> Dict[str, Tuple[Any, Dict]]:
    """Create dictionary of AutoML classifiers with their configurations.
    
    Returns preconfigured AutoML models using either auto-sklearn (Linux)
    or FLAML (cross-platform) depending on availability.
    
    Returns:
        Dictionary mapping preset_name -> (model, config_dict)
        
    Note:
        These models have different behavior than standard classifiers:
        - They don't use RandomizedSearchCV (AutoML does its own search)
        - The config_dict contains AutoML settings, not param_grid
        - Training takes significantly longer due to model search
    """
    models: Dict[str, Tuple[Any, Dict]] = {}
    
    # Try to import AutoML components
    try:
        from ..automl import AutoMLClassifier, FLAMLClassifier, AutoMLConfig, AutoMLPreset
        from ..automl import is_autosklearn_available, is_flaml_available
        
        # Determine which backend to use
        if is_autosklearn_available():
            ClassifierClass = AutoMLClassifier
            backend = "auto-sklearn"
        elif is_flaml_available():
            ClassifierClass = FLAMLClassifier
            backend = "flaml"
        else:
            # No AutoML backend available
            return models
        
        # Quick preset (5 minutes)
        models["automl_quick"] = (
            ClassifierClass(
                time_budget=300 if backend == "flaml" else None,
                preset=AutoMLPreset.QUICK if backend != "flaml" else None,
                metric="roc_auc",
                name="AutoML-Quick",
            ) if backend == "flaml" else AutoMLClassifier(
                preset=AutoMLPreset.QUICK,
                name="AutoML-Quick",
            ),
            {
                "preset": "quick",
                "time_budget": 300,
                "description": "Búsqueda rápida (5 min) para exploración inicial",
                "backend": backend,
            },
        )
        
        # Balanced preset (1 hour)
        models["automl_balanced"] = (
            ClassifierClass(
                time_budget=3600 if backend == "flaml" else None,
                preset=AutoMLPreset.BALANCED if backend != "flaml" else None,
                metric="roc_auc",
                name="AutoML-Balanced",
            ) if backend == "flaml" else AutoMLClassifier(
                preset=AutoMLPreset.BALANCED,
                name="AutoML-Balanced",
            ),
            {
                "preset": "balanced",
                "time_budget": 3600,
                "description": "Balance entre tiempo y rendimiento (1 hora)",
                "backend": backend,
            },
        )
        
        # High performance preset (4 hours)
        models["automl_high_performance"] = (
            ClassifierClass(
                time_budget=14400 if backend == "flaml" else None,
                preset=AutoMLPreset.HIGH_PERFORMANCE if backend != "flaml" else None,
                metric="roc_auc",
                name="AutoML-HighPerformance",
            ) if backend == "flaml" else AutoMLClassifier(
                preset=AutoMLPreset.HIGH_PERFORMANCE,
                name="AutoML-HighPerformance",
            ),
            {
                "preset": "high_performance",
                "time_budget": 14400,
                "description": "Máximo rendimiento (4 horas)",
                "backend": backend,
            },
        )
        
        # Medical-focused preset (optimized for clinical prediction)
        # Uses AUPRC metric and includes only interpretable models
        if backend == "flaml":
            models["automl_clinical"] = (
                FLAMLClassifier(
                    time_budget=1800,
                    metric="roc_auc",
                    estimator_list=["lgbm", "xgboost", "rf", "extra_tree"],
                    name="AutoML-Clinical",
                ),
                {
                    "preset": "clinical",
                    "time_budget": 1800,
                    "description": "Optimizado para predicción clínica (30 min)",
                    "backend": backend,
                    "estimators": ["lgbm", "xgboost", "rf", "extra_tree"],
                },
            )
        else:
            models["automl_clinical"] = (
                AutoMLClassifier(
                    preset=AutoMLPreset.BALANCED,
                    include_estimators=["random_forest", "extra_trees", "gradient_boosting"],
                    metric="roc_auc",
                    name="AutoML-Clinical",
                ),
                {
                    "preset": "clinical",
                    "time_budget": 1800,
                    "description": "Optimizado para predicción clínica (30 min)",
                    "backend": backend,
                    "estimators": ["random_forest", "extra_trees", "gradient_boosting"],
                },
            )
        
    except ImportError as e:
        # AutoML module not available
        import logging
        logging.getLogger(__name__).debug(f"AutoML not available: {e}")
    
    return models
