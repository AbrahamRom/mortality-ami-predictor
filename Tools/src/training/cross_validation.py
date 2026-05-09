"""Cross-validation strategies for model evaluation."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Callable

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.base import clone

from ..config import RANDOM_SEED
from ..preprocessing import build_preprocessing_pipeline, PreprocessingConfig
from ..preprocessing.imbalance import (
    ImbalanceStrategy,
    ImbalanceConfig,
    create_sampler,
    detect_imbalance,
    compute_class_weights,
    apply_class_weight_to_model,
)
from ..models.custom_base import BaseCustomModel

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    IMBLEARN_AVAILABLE = True
except ImportError:
    SMOTE = None
    ImbPipeline = None
    IMBLEARN_AVAILABLE = False


def _get_sampler_from_config(preprocessing_config: Optional[PreprocessingConfig]) -> Optional[object]:
    """Get the appropriate sampler based on preprocessing config."""
    if not IMBLEARN_AVAILABLE:
        return None
    
    if preprocessing_config is None:
        # Default to SMOTE
        return SMOTE(random_state=RANDOM_SEED)
    
    strategy_str = getattr(preprocessing_config, 'imbalance_strategy', 'smote')
    
    # Handle 'none' and 'class_weight' - no sampler needed
    if strategy_str in ('none', 'class_weight'):
        return None
    
    try:
        strategy = ImbalanceStrategy(strategy_str)
        config = ImbalanceConfig(
            strategy=strategy,
            sampling_strategy=getattr(preprocessing_config, 'imbalance_sampling_strategy', 'auto'),
            k_neighbors=getattr(preprocessing_config, 'imbalance_k_neighbors', 5),
            random_state=RANDOM_SEED,
        )
        return create_sampler(config)
    except (ValueError, ImportError):
        # Fallback to basic SMOTE
        return SMOTE(random_state=RANDOM_SEED)


def _contains_custom_model(estimator) -> bool:
    """Check if estimator or any step in pipeline contains a custom model."""
    if isinstance(estimator, BaseCustomModel):
        return True
    if isinstance(estimator, (Pipeline, ImbPipeline)):
        for step in estimator.steps:
            if isinstance(step[1], BaseCustomModel):
                return True
    return False


def nested_cross_validation(
    X: pd.DataFrame,
    y: pd.Series,
    models: Dict[str, Tuple[object, Dict]],
    preprocessing_config: Optional[PreprocessingConfig] = None,
    quick: bool = False,
    progress_callback: Optional[Callable[[str, float], None]] = None,
) -> Dict[str, Dict]:
    """Perform nested cross-validation to evaluate models.
    
    Uses outer CV for evaluation and inner CV for hyperparameter tuning.
    
    Args:
        X: Training features
        y: Training labels
        models: Dictionary of {name: (model, param_grid)}
        preprocessing_config: Preprocessing configuration
        quick: If True, use reduced splits
        progress_callback: Optional callback for progress
        
    Returns:
        Dictionary mapping model_name -> {'mean_score': float, 'std_score': float, 'fold_scores': List[float]}
    """
    outer_splits = 3 if not quick else 2
    inner_splits = 3 if not quick else 2
    inner_iter = 10 if not quick else 4
    
    outer_cv = StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=RANDOM_SEED)
    
    results = {}
    total_models = len(models)
    
    for model_idx, (name, (model, grid)) in enumerate(models.items(), 1):
        fold_scores = []
        
        if progress_callback:
            progress_callback(f"Evaluating {name}...", (model_idx - 1) / total_models)
        
        for fold_idx, (train_idx, val_idx) in enumerate(outer_cv.split(X, y)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Build preprocessing pipeline
            preprocess_pipeline, _ = build_preprocessing_pipeline(X_train, preprocessing_config)
            
            # Extract the preprocessor from the pipeline (to avoid nested pipelines)
            # build_preprocessing_pipeline returns Pipeline([("preprocessor", ColumnTransformer)])
            if hasattr(preprocess_pipeline, 'steps') and len(preprocess_pipeline.steps) > 0:
                preprocess = preprocess_pipeline.steps[0][1]  # Get the ColumnTransformer
            else:
                preprocess = preprocess_pipeline
            
            # Get sampler based on preprocessing config (supports SMOTE, ADASYN, etc.)
            sampler = _get_sampler_from_config(preprocessing_config)
            
            # Check if using class_weight strategy instead of resampling
            use_class_weight = (
                preprocessing_config is not None and 
                getattr(preprocessing_config, 'imbalance_strategy', 'smote') == 'class_weight'
            )
            
            # Apply class weight to model if needed
            model_instance = clone(model)
            if use_class_weight:
                class_weights = compute_class_weights(y_train)
                model_instance = apply_class_weight_to_model(model_instance, class_weights)
            
            # Build full pipeline with resampling if available
            if sampler is not None and IMBLEARN_AVAILABLE:
                pipeline = ImbPipeline([
                    ("preprocess", preprocess),
                    ("sampler", sampler),
                    ("classifier", model_instance),
                ])
            else:
                pipeline = Pipeline([
                    ("preprocess", preprocess),
                    ("classifier", model_instance),
                ])
            
            # Inner CV for hyperparameter tuning
            param_grid = {f"classifier__{k}": v for k, v in grid.items()}
            
            # Custom models may not be picklable, so disable parallel processing
            n_jobs = 1 if _contains_custom_model(pipeline) else -1
            
            search = RandomizedSearchCV(
                pipeline,
                param_distributions=param_grid,
                n_iter=inner_iter,
                cv=StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=RANDOM_SEED),
                scoring="roc_auc",
                n_jobs=n_jobs,
                random_state=RANDOM_SEED + fold_idx,
                refit=True,
            )
            
            search.fit(X_train, y_train)
            
            # Evaluate on validation fold
            y_pred_proba = search.best_estimator_.predict_proba(X_val)[:, 1]
            auroc = roc_auc_score(y_val, y_pred_proba)
            fold_scores.append(auroc)
            
            if progress_callback:
                frac = (model_idx - 1 + (fold_idx + 1) / outer_splits) / total_models
                progress_callback(f"{name}: fold {fold_idx+1}/{outer_splits} AUROC={auroc:.3f}", frac)
        
        results[name] = {
            "mean_score": float(np.mean(fold_scores)),
            "std_score": float(np.std(fold_scores)),
            "fold_scores": fold_scores,
        }
    
    return results


def rigorous_repeated_cv(
    X: pd.DataFrame,
    y: pd.Series,
    models: Dict[str, Tuple[object, Dict]],
    preprocessing_config: Optional[PreprocessingConfig] = None,
    n_splits: int = 5,
    n_repeats: int = 20,
    scoring: str = "roc_auc",
    progress_callback: Optional[Callable[[str, float], None]] = None,
) -> Dict[str, Dict]:
    """Perform rigorous repeated stratified k-fold cross-validation.
    
    This implements the rigorous experimental protocol (context.txt):
    - Repeated Stratified K-Fold: k=5 folds × r=20 repeats = 100 estimations
    - For each model, estimate μ (mean) and σ (std) across all runs
    - No hyperparameter tuning (use default or pre-tuned parameters)
    - Sufficient sample size for Central Limit Theorem application
    - Allows detection of ΔAUROC ≥ 0.02 with power >80%
    
    This is the TRAIN & VALIDATION phase from the experimental pipeline.
    
    Args:
        X: Training features
        y: Training labels
        models: Dictionary of {name: (model, param_dict)} where param_dict contains
                fixed hyperparameters (not grid search)
        preprocessing_config: Preprocessing configuration
        n_splits: Number of folds in each repetition
        n_repeats: Number of repetitions
        scoring: Scoring metric
        progress_callback: Optional callback for progress
        
    Returns:
        Dictionary mapping model_name -> {
            'mean_score': float,  # μ
            'std_score': float,   # σ
            'all_scores': List[float],  # All 90+ scores
            'best_params': dict  # Best hyperparameters found
        }
    """
    # Create repeated stratified k-fold
    rskf = RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=RANDOM_SEED
    )
    
    total_iterations = n_splits * n_repeats
    results = {}
    total_models = len(models)
    
    for model_idx, (name, (base_model, param_grid)) in enumerate(models.items(), 1):
        all_scores = []
        
        if progress_callback:
            progress_callback(
                f"Modelo '{name}': Iniciando {total_iterations} corridas (RepeatedStratifiedKFold: {n_splits}×{n_repeats})",
                (model_idx - 1) / total_models
            )
        
        for run_idx, (train_idx, val_idx) in enumerate(rskf.split(X, y), 1):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Show class distribution only for first run
            if run_idx == 1 and progress_callback:
                train_dist = y_train.value_counts(normalize=True).sort_index()
                val_dist = y_val.value_counts(normalize=True).sort_index()
                dist_info = " | ".join([f"Clase {label}: Train={train_dist[label]*100:.1f}% Val={val_dist[label]*100:.1f}%" 
                                       for label in train_dist.index])
                progress_callback(f"Distribución: {dist_info}", (model_idx - 1) / total_models)
            
            # Show progress for each run
            if progress_callback:
                progress_callback(
                    f"⚙️ Modelo '{name}': Entrenando corrida {run_idx}/{total_iterations}...",
                    (model_idx - 1 + (run_idx - 0.5) / total_iterations) / total_models
                )
            
            # Build preprocessing pipeline
            preprocess_pipeline, _ = build_preprocessing_pipeline(X_train, preprocessing_config)
            
            # Extract the preprocessor
            if hasattr(preprocess_pipeline, 'steps') and len(preprocess_pipeline.steps) > 0:
                preprocess = preprocess_pipeline.steps[0][1]
            else:
                preprocess = preprocess_pipeline
            
            # Get sampler based on preprocessing config (supports SMOTE, ADASYN, etc.)
            sampler = _get_sampler_from_config(preprocessing_config)
            
            # Check if using class_weight strategy instead of resampling
            use_class_weight = (
                preprocessing_config is not None and 
                getattr(preprocessing_config, 'imbalance_strategy', 'smote') == 'class_weight'
            )
            
            # Clone and configure model
            model_instance = clone(base_model)
            if use_class_weight:
                class_weights = compute_class_weights(y_train)
                model_instance = apply_class_weight_to_model(model_instance, class_weights)
            
            # Build full pipeline with resampling if available
            if sampler is not None and IMBLEARN_AVAILABLE:
                pipeline = ImbPipeline([
                    ("preprocess", preprocess),
                    ("sampler", sampler),
                    ("classifier", model_instance),
                ])
            else:
                pipeline = Pipeline([
                    ("preprocess", preprocess),
                    ("classifier", model_instance),
                ])
            
            # Note: param_grid contains lists of values for search, not single values
            # In rigorous_repeated_cv, we use the model's default parameters
            # Hyperparameter tuning should be done separately with RandomizedSearchCV
            
            # Train and evaluate
            try:
                pipeline.fit(X_train, y_train)
                if hasattr(pipeline, "predict_proba"):
                    y_pred_proba = pipeline.predict_proba(X_val)[:, 1]
                else:
                    y_pred_proba = pipeline.decision_function(X_val)
                score = roc_auc_score(y_val, y_pred_proba)
                all_scores.append(score)
                
                # Update progress: every run if ≤20 total runs, else every 5 runs
                update_frequency = 1 if total_iterations <= 20 else 5
                if progress_callback and (run_idx % update_frequency == 0 or run_idx == total_iterations):
                    frac = (model_idx - 1 + run_idx / total_iterations) / total_models
                    current_mean = np.mean(all_scores)
                    current_std = np.std(all_scores)
                    progress_callback(
                        f"Modelo '{name}': Corrida {run_idx}/{total_iterations} | AUROC={score:.3f} | μ={current_mean:.3f}, σ={current_std:.3f}",
                        frac
                    )
            except Exception as e:
                print(f"Warning: Error in {name} run {run_idx}: {e}")
                continue
        
        # Calculate final statistics
        results[name] = {
            "mean_score": float(np.mean(all_scores)),
            "std_score": float(np.std(all_scores, ddof=1)),  # Sample std
            "all_scores": all_scores,
            "n_runs": len(all_scores),
            "best_params": param_grid if param_grid else {},
        }
        
        if progress_callback:
            progress_callback(
                f"✅ Modelo '{name}' COMPLETADO | μ={results[name]['mean_score']:.4f}, σ={results[name]['std_score']:.4f} ({len(all_scores)} corridas exitosas)",
                model_idx / total_models
            )
    
    return results

