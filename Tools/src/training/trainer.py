"""Main training orchestration functions.

This module provides high-level functions for training ML models
with nested cross-validation and SMOTE for class imbalance.
Includes AutoML training pipeline integration.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable, Any, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from ..config import RANDOM_SEED
from ..models import make_classifiers
from ..preprocessing import build_preprocessing_pipeline, PreprocessingConfig
from ..data_load import (
    save_model_with_cleanup,
    save_dataset_with_timestamp,
    cleanup_old_testsets,
    get_timestamp,
    get_model_combination_key,
)
from .cross_validation import nested_cross_validation, rigorous_repeated_cv
from .statistical_tests import compare_all_models, plot_model_comparison
from .learning_curves import generate_learning_curve, plot_learning_curve

# AutoML imports (optional)
try:
    from ..automl import (
        AutoMLClassifier,
        FLAMLClassifier,
        AutoMLConfig,
        AutoMLPreset,
        is_flaml_available,
        is_autosklearn_available,
        export_best_model,
        analyze_dataset,
        get_suggestions,
    )
    AUTOML_AVAILABLE = is_flaml_available() or is_autosklearn_available()
except ImportError:
    AUTOML_AVAILABLE = False
    AutoMLClassifier = None
    FLAMLClassifier = None
    AutoMLConfig = None
    AutoMLPreset = None

# Import imbalanced-learn support from preprocessing module
try:
    from imblearn.over_sampling import SMOTE  # type: ignore
    from imblearn.pipeline import Pipeline as ImbPipeline  # type: ignore
    IMBLEARN_AVAILABLE = True
except ImportError:
    SMOTE = None
    ImbPipeline = None
    IMBLEARN_AVAILABLE = False

# Import imbalance handling utilities
from ..preprocessing.imbalance import (
    ImbalanceStrategy,
    ImbalanceConfig,
    create_sampler,
    detect_imbalance,
    compute_class_weights,
    apply_class_weight_to_model,
    get_recommended_strategy,
    is_imblearn_available,
)


def run_rigorous_experiment_pipeline(
    X: pd.DataFrame,
    y: pd.Series,
    models: Dict[str, Tuple[object, Dict]],
    preprocessing_config: Optional[PreprocessingConfig] = None,
    n_splits: int = 10,
    n_repeats: int = 10,
    scoring: str = "roc_auc",
    test_set: Optional[Tuple[pd.DataFrame, pd.Series]] = None,
    output_dir: Optional[str] = None,
    progress_callback: Optional[Callable[[str, float], None]] = None,
) -> Dict:
    """
    Orchestrate rigorous experiment pipeline following academic standards.
    
    THIS FUNCTION HANDLES TRAINING PHASES (1 & 3):
    
    FASE 1: TRAIN + VALIDATION (done here)
    1. Repeated stratified k-fold CV for all models (≥30 runs)
       → Estimate μ (mean) and σ (std) for each model
    2. Learning curves for diagnostics (detect underfitting/overfitting)
    3. Select best model based on validation performance
    
    FASE 3: STATISTICAL COMPARISON (done here)
    4. Compare all models pairwise:
       - Shapiro-Wilk normality test
       - If normal: Paired t-test (parametric)
       - If not normal: Mann-Whitney U test (non-parametric)
       → Determine if differences are statistically significant
    
    FASE 2: TEST - EVALUATION (done in evaluation module)
    This phase is executed in the Model Evaluation module (04_📈_Model_Evaluation.py):
    - Bootstrap resampling (1000 iterations with replacement)
    - Jackknife resampling (leave-one-out)
    → Obtain final performance estimates with 95% confidence intervals
    
    Args:
        X: Training features
        y: Training labels
        models: Dictionary of {name: (model, param_dict)}
        preprocessing_config: Preprocessing configuration
        n_splits: Number of CV folds (default: 10)
        n_repeats: Number of CV repetitions (default: 10, gives 100 runs)
        scoring: Scoring metric
        test_set: Optional (X_test, y_test) for final evaluation
        output_dir: Directory to save results
        progress_callback: Optional callback for progress updates
        
    Returns:
        Dictionary with complete experiment results
    """
    import os
    results = {}
    if output_dir is None:
        output_dir = MODELS_DIR
    os.makedirs(output_dir, exist_ok=True)
    
    if progress_callback:
        progress_callback("🚀 Iniciando pipeline de experimentación riguroso...", 0.0)

    # =========================================================================
    # FASE 1: TRAIN + VALIDATION (30+ corridas con k-fold estratificado)
    # =========================================================================
    if progress_callback:
        progress_callback("📊 FASE 1: Entrenamiento y Validación con validación cruzada estratificada repetida", 0.1)
    
    # 1. Rigorous repeated CV (≥30 runs to estimate μ and σ)
    cv_results = rigorous_repeated_cv(
        X, y, models,
        preprocessing_config=preprocessing_config,
        n_splits=n_splits,
        n_repeats=n_repeats,
        scoring=scoring,
        progress_callback=progress_callback,
    )
    results['cv_results'] = cv_results
    
    # Log validation results
    if progress_callback:
        msg = "✅ Validación Cruzada Completada:\n"
        for name, res in cv_results.items():
            msg += f"   • {name}: μ={res['mean_score']:.4f}, σ={res['std_score']:.4f} ({res['n_runs']} corridas)\n"
        progress_callback(msg, 0.3)

    # 2. Learning curves for model selection
    if progress_callback:
        progress_callback("📈 Generando curvas de aprendizaje para cada modelo...", 0.35)
    
    # Crear timestamp único para esta sesión de entrenamiento
    from datetime import datetime
    from ..data_load.path_utils import get_plots_dir
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plots_training_dir = get_plots_dir('training')
    os.makedirs(plots_training_dir, exist_ok=True)
    
    # Guardar el timestamp en los resultados para referencia en el dashboard
    results['training_timestamp'] = timestamp
    
    lc_results = {}
    for idx, (name, (model, param_grid)) in enumerate(models.items(), 1):
        # Use the base model (without hyperparameter tuning) for learning curves
        # Learning curves show how model performs with different training sizes,
        # not different hyperparameters
        from sklearn.base import clone
        model_for_lc = clone(model)
        
        if progress_callback:
            progress_callback(f"🔄 Generando curva de aprendizaje: {name} (evaluando 10 tamaños × {n_splits} folds)...", 0.35 + 0.15 * idx / len(models))
        
        try:
            # IMPORTANT: Use n_jobs=1 for learning curve to avoid nested parallelism deadlocks
            # Models like XGBoost/LightGBM have their own internal parallelism
            lc_result = generate_learning_curve(
                model_for_lc,
                X.values,
                y.values,
                cv=n_splits,
                scoring=scoring,
                n_jobs=1,  # Avoid nested parallelism deadlocks with XGBoost/LightGBM
            )
            
            # Check if results are valid
            if np.all(np.isnan(lc_result.train_scores_mean)):
                if progress_callback:
                    progress_callback(f"⚠️ No se pudo generar curva para {name} (Scores NaN)", 0.35 + 0.15 * (idx + 0.5) / len(models))
            else:
                if progress_callback:
                    progress_callback(f"✓ Curva generada: {name} (Train={lc_result.train_scores_mean[-1]:.3f}, Val={lc_result.val_scores_mean[-1]:.3f})", 0.35 + 0.15 * (idx + 0.5) / len(models))
            
            lc_results[name] = lc_result
            
            if progress_callback:
                progress_callback(f"💾 Guardando gráfica de curva de aprendizaje: {name}", 0.35 + 0.15 * (idx + 0.7) / len(models))
            
            fig = plot_learning_curve(lc_result, title=f"Learning Curve: {name}")
            # Save as PNG using plotly's write_image (requires kaleido) with timestamp
            save_path = os.path.join(plots_training_dir, f"learning_curve_{name}_{timestamp}.png")
            try:
                fig.write_image(save_path, width=1000, height=600)
            except Exception as e:
                # If write_image fails (missing kaleido), save as HTML instead
                html_path = os.path.join(plots_training_dir, f"learning_curve_{name}_{timestamp}.html")
                fig.write_html(html_path)
                if progress_callback:
                    progress_callback(f"⚠️ Guardado como HTML (instalar kaleido para PNG): {html_path}", 0.35 + 0.15 * idx / len(models))
        except Exception as e:
            print(f"Error generating learning curve for {name}: {e}")
            if progress_callback:
                progress_callback(f"❌ Error generando curva para {name}: {e}", 0.35 + 0.15 * idx / len(models))
    results['learning_curves'] = lc_results
    
    # 3. Select best model
    best_name = max(cv_results.items(), key=lambda x: x[1]['mean_score'])[0]
    results['best_model'] = best_name
    
    if progress_callback:
        best_score = cv_results[best_name]['mean_score']
        progress_callback(f"🏆 Mejor modelo seleccionado: {best_name} (μ={best_score:.4f})", 0.5)

    # =========================================================================
    # FASE 3: COMPARACIÓN ESTADÍSTICA RIGUROSA (Protocolo context.txt)
    # =========================================================================
    if progress_callback:
        progress_callback("📊 FASE 3: Comparación estadística rigurosa entre modelos (protocolo completo)", 0.55)
    
    # Extract scores for statistical comparison
    model_scores = {name: res['all_scores'] for name, res in cv_results.items()}
    
    # 3.1 Global test (Friedman or ANOVA) + pairwise with corrections
    from .statistical_tests import compare_multiple_models
    multiple_comparison_result = compare_multiple_models(model_scores)
    results['multiple_comparison'] = multiple_comparison_result
    
    if progress_callback:
        if np.isnan(multiple_comparison_result.global_test_statistic):
            global_msg = (f"   Test Global: {multiple_comparison_result.global_test_name}\n"
                         f"   Nota: Se requieren ≥3 modelos para test Friedman/ANOVA\n"
                         f"   Procediendo con comparación pareada directa.")
        else:
            global_msg = (f"   Test Global: {multiple_comparison_result.global_test_name}\n"
                         f"   Estadístico: {multiple_comparison_result.global_test_statistic:.4f}\n"
                         f"   P-value global: {multiple_comparison_result.global_p_value:.6f}\n"
                         f"   Significativo (α=0.05): {'SÍ' if multiple_comparison_result.global_significant else 'NO'}\n"
                         f"   α Bonferroni: {multiple_comparison_result.bonferroni_alpha:.6f}")
        progress_callback(global_msg, 0.57)
    
    # 3.2 Pairwise comparisons (already computed in compare_multiple_models)
    stat_results = multiple_comparison_result.pairwise_results
    results['statistical_comparison'] = stat_results

    # Save pairwise comparison plots (only when there are at least 2 models)
    if stat_results:
        if progress_callback:
            progress_callback("Generando gráficos de comparación estadística...", 0.6)

        for (m1, m2), stat_res in stat_results.items():
            fig = plot_model_comparison(stat_res)
            save_path = os.path.join(plots_training_dir, f"comparison_{m1}_vs_{m2}_{timestamp}.png")
            try:
                # Check if it's a Plotly figure
                if hasattr(fig, 'write_image'):
                    fig.write_image(save_path, width=1000, height=600)
                else:
                    # Matplotlib figure
                    fig.savefig(save_path, dpi=300, bbox_inches='tight')
            except Exception:
                # If save fails, try HTML for Plotly
                if hasattr(fig, 'write_html'):
                    html_path = os.path.join(plots_training_dir, f"comparison_{m1}_vs_{m2}_{timestamp}.html")
                    fig.write_html(html_path)

            # Log statistical results with correction info
            if progress_callback:
                test_type = stat_res.test_used
                p_val = stat_res.p_value
                holm_p = multiple_comparison_result.holm_corrected_pvalues.get((m1, m2), p_val)
                significant = "SÍ" if stat_res.significant else "NO"
                sig_bonferroni = "SÍ" if p_val < multiple_comparison_result.bonferroni_alpha else "NO"
                msg = (f"   • {m1} vs {m2}:\n"
                    f"     Test: {test_type} | ΔAUROC={stat_res.delta_auroc:.4f}\n"
                    f"     P-value: {p_val:.4f} (Holm: {holm_p:.4f})\n"
                    f"     Sig. sin corrección: {significant} | Con Bonferroni: {sig_bonferroni}\n"
                    f"     Tamaño efecto: {stat_res.effect_size:.3f} ({stat_res.effect_size_interpretation})")
                progress_callback(msg, 0.6)

        # Create comparison matrix
        from .statistical_tests import create_comparison_matrix
        fig_matrix = create_comparison_matrix(stat_results)
        matrix_path = os.path.join(plots_training_dir, f"comparison_matrix_{timestamp}.png")
        try:
            if hasattr(fig_matrix, 'write_image'):
                fig_matrix.write_image(matrix_path, width=1000, height=600)
            else:
                fig_matrix.savefig(matrix_path, dpi=300, bbox_inches='tight')
        except Exception:
            if hasattr(fig_matrix, 'write_html'):
                html_path = os.path.join(plots_training_dir, f"comparison_matrix_{timestamp}.html")
                fig_matrix.write_html(html_path)
    elif progress_callback:
        progress_callback("ℹ️ Comparación estadística no aplicable: un solo modelo.", 0.6)
    
    # =========================================================================
    # SAVE TRAINED MODELS (all selected manual models)
    # =========================================================================
    # NOTE: Bootstrap and Jackknife evaluation (FASE 2) is done in the
    # EVALUATION module (04_📈_Model_Evaluation.py), not here.

    if progress_callback:
        progress_callback("💾 Entrenando y guardando todos los modelos seleccionados...", 0.7)

    from sklearn.base import clone
    from sklearn.pipeline import Pipeline
    from ..models import create_metadata_from_training

    # Save test/train sets once (shared by all models in this run)
    models_dir = Path(output_dir)
    testsets_dir = models_dir / "testsets"
    task_name = y.name if hasattr(y, 'name') and y.name else 'unknown'

    if test_set is not None:
        X_test, y_test = test_set
        test_df = pd.concat([X_test, y_test], axis=1)

        test_path = save_dataset_with_timestamp(
            test_df,
            testsets_dir,
            prefix=f"testset_{task_name}",
            format="parquet"
        )
        results['test_set_path'] = str(test_path)
        
        # =====================================================================
        # SAVE TESTSET METADATA (for ClinicalScaleMLOrchestrator)
        # =====================================================================
        # Create metadata for clinical scale computation and orchestration
        import json
        from pathlib import Path as PathlibPath
        
        testset_metadata = {
            'test_set_path': str(test_path),
            'test_set_creation_timestamp': timestamp,
            'dataset_info': {
                'n_samples': len(test_df),
                'n_features': len(X_test.columns),
                'n_events': int((y_test == 1).sum()),
                'event_rate': float((y_test == 1).mean()),
            },
            'target_column': y_test.name if hasattr(y_test, 'name') else 'muerte_inhospitalaria',
            'feature_columns': X_test.columns.tolist(),
            'column_dtypes': {col: str(dtype) for col, dtype in X_test.dtypes.items()},
            
            # Variable mappings for clinical scale computation
            'variable_mappings': {
                'grace': {
                    'age': 'age',
                    'heart_rate': 'heart_rate',
                    'systolic_bp': 'systolic_bp',
                    'creatinine': 'creatinine',
                    'cardiac_arrest': 'cardiac_arrest',
                    'st_deviation': 'st_deviation',
                    'elevated_enzymes': 'elevated_enzymes',
                    'killip_class': 'killip_class',
                },
                'recuima': {
                    'age': 'age',
                    'sex': 'sex',
                    'diabetes': 'diabetes',
                    'hypertension': 'hypertension',
                    'ckd': 'ckd',
                    'gfr': 'gfr',
                    'complications': 'complications',
                },
                'timi_stemi': {
                    'age': 'age',
                    'anterior': 'anterior',
                    'systolic_bp': 'systolic_bp',
                    'heart_rate': 'heart_rate',
                    'killip': 'killip',
                    'weight': 'weight',
                    'time_to_treatment': 'time_to_treatment',
                },
                'timi_nstemi': {
                    'age': 'age',
                    'weight': 'weight',
                    'sx_angina': 'sx_angina',
                    'st_deviation': 'st_deviation',
                    'elevated_enzymes': 'elevated_enzymes',
                },
            },
            'notes': 'Variable mappings specify required columns for each clinical scale. '
                     'Orchestrator will map these to actual dataframe columns during computation.',
        }
        
        # Save metadata as JSON
        testset_metadata_path = PathlibPath(test_path).parent / f"{PathlibPath(test_path).stem}_metadata.json"
        with open(testset_metadata_path, 'w') as f:
            json.dump(testset_metadata, f, indent=2)
        
        results['test_set_metadata_path'] = str(testset_metadata_path)

        train_df = pd.concat([X, y], axis=1)
        train_path = save_dataset_with_timestamp(
            train_df,
            testsets_dir,
            prefix=f"trainset_{task_name}",
            format="parquet"
        )
        results['train_set_path'] = str(train_path)

        cleanup_old_testsets(task_name, testsets_dir, keep_n_latest=1)
    else:
        test_path = None
        train_path = None
        test_df = None
        train_df = pd.concat([X, y], axis=1)

    if progress_callback:
        progress_callback(f"Test/Train sets guardados en: {testsets_dir}", 0.74)

    # Shared training configuration
    training_config = {
        'duration': results.get('training_duration', 0.0),
        'cv_strategy': f"RepeatedStratifiedKFold(n_splits={n_splits}, n_repeats={n_repeats})",
        'n_splits': n_splits,
        'n_repeats': n_repeats,
        'total_runs': n_splits * n_repeats,
        'scoring': scoring,
        'preprocessing': preprocessing_config.__dict__ if preprocessing_config else {},
        'random_seed': RANDOM_SEED,
    }

    imbalance_strategy_str = getattr(preprocessing_config, 'imbalance_strategy', 'smote') if preprocessing_config else 'smote'
    use_class_weight = imbalance_strategy_str == 'class_weight'

    full_training_df = pd.concat([X, y], axis=1)
    saved_models: Dict[str, str] = {}
    saved_metadata: Dict[str, str] = {}
    saved_combinations: Dict[str, str] = {}

    total_models = len(models)
    for idx, (model_name, (base_model, _)) in enumerate(models.items(), 1):
        progress_start = 0.74 + (idx - 1) * (0.22 / max(total_models, 1))
        progress_end = 0.74 + idx * (0.22 / max(total_models, 1))

        if progress_callback:
            progress_callback(
                f"🔧 Entrenando modelo final '{model_name}' ({idx}/{total_models})...",
                progress_start,
            )

        model_instance = clone(base_model)

        if use_class_weight:
            class_weights = compute_class_weights(y)
            model_instance = apply_class_weight_to_model(model_instance, class_weights)

        preprocess_pipeline, _ = build_preprocessing_pipeline(X, preprocessing_config)
        if hasattr(preprocess_pipeline, 'steps') and len(preprocess_pipeline.steps) > 0:
            preprocess = preprocess_pipeline.steps[0][1]
        else:
            preprocess = preprocess_pipeline

        sampler = None
        if not use_class_weight and imbalance_strategy_str != 'none' and is_imblearn_available():
            try:
                strategy = ImbalanceStrategy(imbalance_strategy_str)
                imb_config = ImbalanceConfig(
                    strategy=strategy,
                    sampling_strategy=getattr(preprocessing_config, 'imbalance_sampling_strategy', 'auto') if preprocessing_config else 'auto',
                    k_neighbors=getattr(preprocessing_config, 'imbalance_k_neighbors', 5) if preprocessing_config else 5,
                    random_state=RANDOM_SEED,
                )
                sampler = create_sampler(imb_config)
            except (ValueError, ImportError):
                sampler = SMOTE(random_state=RANDOM_SEED) if IMBLEARN_AVAILABLE else None

        if sampler is not None and IMBLEARN_AVAILABLE:
            final_pipeline = ImbPipeline([
                ("preprocess", preprocess),
                ("sampler", sampler),
                ("classifier", model_instance),
            ])
        else:
            final_pipeline = Pipeline([
                ("preprocess", preprocess),
                ("classifier", model_instance),
            ])

        final_pipeline.fit(X, y)

        try:
            if hasattr(final_pipeline, 'feature_names_in_'):
                actual_feature_names = final_pipeline.feature_names_in_.tolist()
            elif hasattr(final_pipeline.steps[-1][1], 'feature_names_in_'):
                actual_feature_names = final_pipeline.steps[-1][1].feature_names_in_.tolist()
            elif hasattr(final_pipeline.steps[0][1], 'feature_names_in_'):
                actual_feature_names = final_pipeline.steps[0][1].feature_names_in_.tolist()
            else:
                actual_feature_names = X.columns.tolist()
        except Exception:
            actual_feature_names = X.columns.tolist()

        model_type_str = str(type(model_instance).__name__)
        model_params = model_instance.get_params() if hasattr(model_instance, 'get_params') else {}

        combination_key = get_model_combination_key(model_name, imbalance_strategy_str)
        saved_combinations[model_name] = combination_key

        metadata = create_metadata_from_training(
            model_name=model_name,
            model_type=model_type_str,
            task=y.name if hasattr(y, 'name') else 'unknown',
            model_file_path='',
            train_set_path=str(train_path) if train_path else '',
            test_set_path=str(test_path) if test_path else '',
            train_df=train_df,
            test_df=test_df if test_df is not None else train_df.head(0),
            target_column=y.name if hasattr(y, 'name') else 'target',
            hyperparameters=model_params,
            cv_results=cv_results.get(model_name, {}),
            training_config=training_config,
            learning_curve_path=results.get('learning_curves_paths', {}).get(model_name),
            statistical_comparison=results.get('statistical_comparison'),
            notes=(
                f"Model trained with resampling strategy '{imbalance_strategy_str}'. "
                f"Combination key: {combination_key}"
            ),
            actual_feature_names=actual_feature_names,
        )

        final_model_path = save_model_with_cleanup(
            final_pipeline,
            model_name,
            models_dir,
            keep_n_latest=1,
            metadata=metadata,
            training_data=full_training_df,
            combination_key=combination_key,
        )

        saved_models[model_name] = str(final_model_path)
        saved_metadata[model_name] = str(final_model_path.with_suffix('.metadata.json'))

        if progress_callback:
            progress_callback(
                f"✅ Modelo guardado: {model_name} ({combination_key})",
                progress_end,
            )

    # Keep legacy fields for compatibility with dashboard and downstream modules
    results['saved_models'] = saved_models
    results['saved_metadata'] = saved_metadata
    results['saved_combinations'] = saved_combinations
    results['final_model_path'] = saved_models.get(best_name)
    results['metadata_path'] = saved_metadata.get(best_name)

    if progress_callback:
        progress_callback("✅ Pipeline de entrenamiento completado exitosamente!", 1.0)

    return results


MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models")
os.makedirs(MODELS_DIR, exist_ok=True)


def train_best_classifier(
    X: pd.DataFrame,
    y: pd.Series,
    quick: bool = False,
    task_name: str = "mortality",
    preprocessing_config: Optional[PreprocessingConfig] = None,
    progress_callback: Optional[Callable[[str, float], None]] = None,
) -> Tuple[str, object]:
    """Train multiple classifiers and select best based on nested CV.
    
    Performs nested cross-validation to evaluate all classifiers,
    selects the best, refits on full data, and saves.
    
    Args:
        X: Training features
        y: Training labels
        quick: If True, use reduced CV splits and iterations
        task_name: Task name for saving model
        preprocessing_config: Preprocessing configuration
        progress_callback: Optional callback(message, progress)
        
    Returns:
        Tuple of (save_path, fitted_model)
    """
    # Check if model already exists
    save_path = os.path.join(MODELS_DIR, f"best_classifier_{task_name}.joblib")
    if os.path.exists(save_path):
        if progress_callback:
            progress_callback("Existing model found. Loading...", 1.0)
        model = joblib.load(save_path)
        return save_path, model
    
    # Get all available models
    models = make_classifiers()
    
    # Remove neural network unless explicitly enabled
    if "nn" in models and os.environ.get("ENABLE_TORCH_MODEL", "0") != "1":
        models.pop("nn", None)
    
    if progress_callback:
        progress_callback(f"Evaluating {len(models)} models...", 0.1)
    
    # Evaluate all models with nested CV
    results = nested_cross_validation(
        X=X,
        y=y,
        models=models,
        preprocessing_config=preprocessing_config,
        quick=quick,
        progress_callback=progress_callback,
    )
    
    # Find best model
    best_name = max(results.items(), key=lambda x: x[1]["mean_score"])[0]
    best_model, best_grid = models[best_name]
    
    if progress_callback:
        best_score = results[best_name]["mean_score"]
        progress_callback(f"Best model: {best_name} (AUROC: {best_score:.3f})", 0.8)
    
    # Refit best model on all data
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import RandomizedSearchCV
    
    # Build preprocessing pipeline
    preprocess_pipeline, _ = build_preprocessing_pipeline(X, preprocessing_config)
    
    # Extract the preprocessor from the pipeline (to avoid nested pipelines)
    if hasattr(preprocess_pipeline, 'steps') and len(preprocess_pipeline.steps) > 0:
        preprocess = preprocess_pipeline.steps[0][1]  # Get the ColumnTransformer
    else:
        preprocess = preprocess_pipeline
    
    # Get imbalance handling strategy from config
    imbalance_strategy_str = getattr(preprocessing_config, 'imbalance_strategy', 'smote') if preprocessing_config else 'smote'
    use_class_weight = imbalance_strategy_str == 'class_weight'
    
    # Apply class weight if that strategy is selected
    if use_class_weight:
        class_weights = compute_class_weights(y)
        best_model = apply_class_weight_to_model(best_model, class_weights)
    
    # Get sampler based on preprocessing config
    sampler = None
    if not use_class_weight and imbalance_strategy_str != 'none' and is_imblearn_available():
        try:
            strategy = ImbalanceStrategy(imbalance_strategy_str)
            imb_config = ImbalanceConfig(
                strategy=strategy,
                sampling_strategy=getattr(preprocessing_config, 'imbalance_sampling_strategy', 'auto') if preprocessing_config else 'auto',
                k_neighbors=getattr(preprocessing_config, 'imbalance_k_neighbors', 5) if preprocessing_config else 5,
                random_state=RANDOM_SEED,
            )
            sampler = create_sampler(imb_config)
        except (ValueError, ImportError):
            sampler = SMOTE(random_state=RANDOM_SEED) if IMBLEARN_AVAILABLE else None
    
    # Build full pipeline with resampling if available
    if sampler is not None and IMBLEARN_AVAILABLE:
        pipeline = ImbPipeline([
            ("preprocess", preprocess),
            ("sampler", sampler),
            ("classifier", best_model),
        ])
    else:
        from sklearn.pipeline import Pipeline
        pipeline = Pipeline([
            ("preprocess", preprocess),
            ("classifier", best_model),
        ])
    
    # Hyperparameter search
    param_grid = {f"classifier__{k}": v for k, v in best_grid.items()}
    
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_grid,
        n_iter=15 if not quick else 6,
        cv=StratifiedKFold(n_splits=3 if not quick else 2, shuffle=True, random_state=RANDOM_SEED),
        scoring="roc_auc",
        n_jobs=-1,
        random_state=RANDOM_SEED,
        refit=True,
    )
    
    search.fit(X, y)
    best_model_fitted = search.best_estimator_
    
    # Save
    joblib.dump(best_model_fitted, save_path)
    
    if progress_callback:
        progress_callback(f"Model saved to {save_path}", 1.0)
    
    return save_path, best_model_fitted


def train_selected_classifiers(
    X: pd.DataFrame,
    y: pd.Series,
    selected_models: List[str],
    quick: bool = False,
    task_name: str = "mortality",
    preprocessing_config: Optional[PreprocessingConfig] = None,
    progress_callback: Optional[Callable[[str, float], None]] = None,
) -> Dict[str, str]:
    """Train specific classifiers and save all.
    
    Args:
        X: Training features
        y: Training labels
        selected_models: List of model names to train
        quick: If True, use reduced CV
        task_name: Task name for saving
        preprocessing_config: Preprocessing configuration
        progress_callback: Optional callback(message, progress)
        
    Returns:
        Dictionary mapping model_name -> save_path
    """
    all_models = make_classifiers()
    models = {k: v for k, v in all_models.items() if k in selected_models}
    
    if not models:
        raise ValueError("No valid models selected")
    
    # Evaluate with nested CV
    results = nested_cross_validation(
        X=X,
        y=y,
        models=models,
        preprocessing_config=preprocessing_config,
        quick=quick,
        progress_callback=progress_callback,
    )
    
    # Refit and save each model
    save_paths = {}
    
    for idx, (name, (model, grid)) in enumerate(models.items(), 1):
        if progress_callback:
            progress_callback(f"Refitting {name}...", 0.5 + (idx / len(models)) * 0.5)
        
        from sklearn.pipeline import Pipeline
        from sklearn.model_selection import RandomizedSearchCV
        
        # Build pipeline
        preprocess_pipeline, _ = build_preprocessing_pipeline(X, preprocessing_config)
        
        # Extract the preprocessor from the pipeline (to avoid nested pipelines)
        if hasattr(preprocess_pipeline, 'steps') and len(preprocess_pipeline.steps) > 0:
            preprocess = preprocess_pipeline.steps[0][1]  # Get the ColumnTransformer
        else:
            preprocess = preprocess_pipeline
        
        if ImbPipeline and SMOTE:
            try:
                from imblearn.pipeline import Pipeline as ImbPipeline
                from imblearn.over_sampling import SMOTE
                pipeline = ImbPipeline([
                    ("preprocess", preprocess),
                    ("smote", SMOTE(random_state=RANDOM_SEED)),
                    ("classifier", model),
                ])
            except ImportError:
                pipeline = Pipeline([
                    ("preprocess", preprocess),
                    ("classifier", model),
                ])
        else:
            pipeline = Pipeline([
                ("preprocess", preprocess),
                ("classifier", model),
            ])
        
        # Search and refit
        param_grid = {f"classifier__{k}": v for k, v in grid.items()}
        
        search = RandomizedSearchCV(
            pipeline,
            param_distributions=param_grid,
            n_iter=15 if not quick else 6,
            cv=StratifiedKFold(n_splits=3 if not quick else 2, shuffle=True, random_state=RANDOM_SEED),
            scoring="roc_auc",
            n_jobs=-1,
            random_state=RANDOM_SEED,
            refit=True,
        )
        
        search.fit(X, y)
        
        # Save
        path = os.path.join(MODELS_DIR, f"model_{task_name}_{name}.joblib")
        joblib.dump(search.best_estimator_, path)
        save_paths[name] = path
    
    # Also save best overall
    best_name = max(results.items(), key=lambda x: x[1]["mean_score"])[0]
    best_path = os.path.join(MODELS_DIR, f"best_classifier_{task_name}.joblib")
    joblib.dump(joblib.load(save_paths[best_name]), best_path)
    
    if progress_callback:
        progress_callback("Training complete!", 1.0)
    
    return save_paths


# =============================================================================
# AUTOML TRAINING PIPELINE
# =============================================================================

def run_automl_experiment_pipeline(
    X: pd.DataFrame,
    y: pd.Series,
    preset: str = "balanced",
    metric: str = "roc_auc",
    time_budget: Optional[int] = None,
    test_set: Optional[Tuple[pd.DataFrame, pd.Series]] = None,
    output_dir: Optional[str] = None,
    estimator_list: Optional[List[str]] = None,
    progress_callback: Optional[Callable[[str, float], None]] = None,
    include_suggestions: bool = True,
    compare_with_manual: bool = False,
    manual_models: Optional[Dict[str, Tuple[object, Dict]]] = None,
) -> Dict[str, Any]:
    """
    Run AutoML experiment pipeline with optional comparison to manual models.
    
    This pipeline automates model selection and hyperparameter tuning using
    auto-sklearn or FLAML, with intelligent suggestions linked to other modules.
    
    WORKFLOW:
    1. ANALYSIS: Analyze dataset characteristics
    2. SUGGESTIONS: Generate technique suggestions (linked to app modules)
    3. AUTOML: Run AutoML search (auto-sklearn or FLAML)
    4. COMPARISON: Optionally compare with manual models
    5. EXPORT: Save best model with metadata
    
    Args:
        X: Training features DataFrame
        y: Training labels Series
        preset: AutoML preset ('quick', 'balanced', 'high_performance', 'overnight')
        metric: Optimization metric ('roc_auc', 'f1', 'balanced_accuracy', etc.)
        time_budget: Override time budget in seconds (None = use preset default)
        test_set: Optional (X_test, y_test) tuple for final evaluation
        output_dir: Directory to save results (default: models/automl)
        estimator_list: Optional list of estimators to search (FLAML)
        progress_callback: Optional callback(message, progress) for updates
        include_suggestions: Whether to analyze dataset and provide suggestions
        compare_with_manual: Whether to also train manual models for comparison
        manual_models: Dictionary of {name: (model, param_dict)} for comparison
        
    Returns:
        Dictionary with complete AutoML experiment results:
        - 'automl_model': Fitted AutoML classifier
        - 'best_estimator': Name of best estimator found
        - 'best_score': Best validation score
        - 'leaderboard': DataFrame with model rankings
        - 'suggestions': List of technique suggestions (if include_suggestions)
        - 'dataset_analysis': Dataset analysis results
        - 'comparison': Statistical comparison with manual models (if compare_with_manual)
        - 'final_model_path': Path to saved model
        - 'metadata_path': Path to metadata JSON
        
    Raises:
        ImportError: If neither FLAML nor auto-sklearn is available
        ValueError: If invalid preset or metric
        
    Example:
        >>> from src.training import run_automl_experiment_pipeline
        >>> results = run_automl_experiment_pipeline(
        ...     X_train, y_train,
        ...     preset="balanced",
        ...     metric="roc_auc",
        ...     progress_callback=lambda msg, prog: print(f"{prog*100:.0f}% - {msg}")
        ... )
        >>> print(f"Best model: {results['best_estimator']}")
        >>> print(f"Best AUC: {results['best_score']:.4f}")
    """
    if not AUTOML_AVAILABLE:
        raise ImportError(
            "AutoML is not available. Please install FLAML: pip install flaml[automl] "
            "or auto-sklearn (Linux only): pip install auto-sklearn"
        )
    
    results = {}
    
    # Setup output directory
    if output_dir is None:
        output_dir = os.path.join(MODELS_DIR, "automl")
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert preset string to enum if needed
    preset_enum = AutoMLPreset[preset.upper()] if isinstance(preset, str) else preset
    
    if progress_callback:
        backend = "auto-sklearn" if is_autosklearn_available() else "FLAML"
        progress_callback(f"🚀 Iniciando pipeline AutoML ({backend})...", 0.0)
    
    # =========================================================================
    # PHASE 1: DATASET ANALYSIS
    # =========================================================================
    if include_suggestions:
        if progress_callback:
            progress_callback("📊 FASE 1: Analizando características del dataset...", 0.05)
        
        target_column = y.name if hasattr(y, 'name') and y.name else 'target'
        
        # Create combined dataframe for analysis
        df_combined = pd.concat([X, y.rename(target_column)], axis=1)
        
        analysis = analyze_dataset(df_combined, target_column=target_column)
        results['dataset_analysis'] = {
            'n_samples': analysis.n_samples,
            'n_features': analysis.n_features,
            'imbalance_ratio': analysis.imbalance_ratio,
            'is_imbalanced': analysis.is_imbalanced,
            'missing_percentage': analysis.missing_percentage,
            'high_cardinality_features': analysis.high_cardinality_features,
            'highly_correlated_pairs': len(analysis.highly_correlated_pairs),
            'skewed_features': len(analysis.skewed_features),
        }
        
        if progress_callback:
            msg = (f"   • Muestras: {analysis.n_samples}, Features: {analysis.n_features}\n"
                   f"   • Desbalanceado: {'Sí' if analysis.is_imbalanced else 'No'} "
                   f"(ratio: {analysis.imbalance_ratio:.2f})\n"
                   f"   • Datos faltantes: {analysis.missing_percentage:.1f}%")
            progress_callback(msg, 0.1)
        
        # Get suggestions
        if progress_callback:
            progress_callback("💡 Generando sugerencias de técnicas...", 0.12)
        
        suggestions = get_suggestions(df_combined, target_column=target_column)
        results['suggestions'] = [
            {
                'title': s.title,
                'description': s.description,
                'reason': s.reason,
                'module_link': s.module_link,
                'priority': s.priority.value if hasattr(s.priority, 'value') else str(s.priority),
            }
            for s in suggestions
        ]
        
        if progress_callback:
            n_high = sum(1 for s in suggestions if s.priority.value == 'high')
            progress_callback(f"   • {len(suggestions)} sugerencias generadas ({n_high} alta prioridad)", 0.15)
    
    # =========================================================================
    # PHASE 2: AUTOML TRAINING
    # =========================================================================
    if progress_callback:
        progress_callback("🤖 FASE 2: Ejecutando búsqueda AutoML...", 0.2)
    
    # Determine time budget
    preset_times = {
        AutoMLPreset.QUICK: 300,
        AutoMLPreset.BALANCED: 3600,
        AutoMLPreset.HIGH_PERFORMANCE: 14400,
        AutoMLPreset.OVERNIGHT: 28800,
    }
    actual_time_budget = time_budget or preset_times.get(preset_enum, 3600)
    
    # Create AutoML model
    if is_flaml_available():
        # Use FLAML (cross-platform)
        clf = FLAMLClassifier(
            time_budget=actual_time_budget,
            metric=metric,
            estimator_list=estimator_list,
            ensemble=True,
            name=f"AutoML_{preset}"
        )
        backend_used = "FLAML"
    else:
        # Use auto-sklearn (Linux only)
        config = AutoMLConfig.from_preset(preset_enum)
        config.metric = metric
        if time_budget:
            config.time_left_for_this_task = time_budget
        
        clf = AutoMLClassifier(
            config=config,
            name=f"AutoML_{preset}"
        )
        backend_used = "auto-sklearn"
    
    results['backend'] = backend_used
    results['time_budget'] = actual_time_budget
    
    # Training progress callback wrapper
    def automl_progress(msg: str, prog: float):
        if progress_callback:
            # Scale progress to 0.2-0.7 range (AutoML phase)
            scaled_prog = 0.2 + prog * 0.5
            progress_callback(msg, scaled_prog)
    
    # Fit AutoML
    import time
    start_time = time.time()
    
    if hasattr(clf, 'progress_callback'):
        clf.progress_callback = automl_progress
    
    clf.fit(X, y)
    
    training_duration = time.time() - start_time
    results['training_duration'] = training_duration
    results['automl_model'] = clf
    
    # Extract results
    if hasattr(clf, 'best_estimator_'):
        results['best_estimator'] = str(clf.best_estimator_)
    elif hasattr(clf, 'automl') and hasattr(clf.automl, 'best_estimator_'):
        results['best_estimator'] = str(clf.automl.best_estimator_)
    else:
        results['best_estimator'] = "Unknown"
    
    if hasattr(clf, 'best_loss_'):
        results['best_score'] = -clf.best_loss_  # FLAML returns negative loss
    elif hasattr(clf, 'automl') and hasattr(clf.automl, 'leaderboard'):
        try:
            lb = clf.automl.leaderboard()
            results['best_score'] = lb.iloc[0]['score'] if len(lb) > 0 else 0.0
        except Exception:
            results['best_score'] = 0.0
    else:
        results['best_score'] = 0.0
    
    if progress_callback:
        progress_callback(
            f"✅ AutoML completado en {training_duration/60:.1f} min\n"
            f"   • Mejor modelo: {results['best_estimator']}\n"
            f"   • Mejor {metric}: {results['best_score']:.4f}",
            0.7
        )
    
    # Get leaderboard
    try:
        if hasattr(clf, 'get_leaderboard'):
            leaderboard = clf.get_leaderboard()
        elif hasattr(clf, 'automl') and hasattr(clf.automl, 'leaderboard'):
            leaderboard = clf.automl.leaderboard()
        else:
            leaderboard = pd.DataFrame({
                'model': [results['best_estimator']],
                'score': [results['best_score']]
            })
        results['leaderboard'] = leaderboard
    except Exception as e:
        results['leaderboard'] = pd.DataFrame()
        if progress_callback:
            progress_callback(f"⚠️ No se pudo obtener leaderboard: {e}", 0.71)
    
    # =========================================================================
    # PHASE 3: COMPARISON WITH MANUAL MODELS (Optional)
    # =========================================================================
    if compare_with_manual and manual_models:
        if progress_callback:
            progress_callback("📊 FASE 3: Comparando con modelos manuales...", 0.75)
        
        # Run CV on manual models
        cv_results = rigorous_repeated_cv(
            X, y, manual_models,
            n_splits=5,
            n_repeats=2,  # Reduced for comparison
            scoring=metric,
        )
        
        # Add AutoML to comparison
        automl_scores = []
        from sklearn.model_selection import cross_val_score
        try:
            # Use cross-validation to get comparable scores
            automl_cv_scores = cross_val_score(
                clf, X, y, cv=5, scoring=metric, n_jobs=-1
            )
            automl_scores = automl_cv_scores.tolist()
        except Exception:
            # Use single best score repeated
            automl_scores = [results['best_score']] * 10
        
        cv_results['AutoML'] = {
            'mean_score': np.mean(automl_scores),
            'std_score': np.std(automl_scores),
            'all_scores': automl_scores,
            'n_runs': len(automl_scores),
        }
        
        # Statistical comparison
        model_scores = {name: res['all_scores'] for name, res in cv_results.items()}
        stat_results = compare_all_models(model_scores)
        
        results['comparison'] = {
            'cv_results': cv_results,
            'statistical_tests': {
                f"{m1}_vs_{m2}": {
                    'test_used': sr.test_used,
                    'p_value': sr.p_value,
                    'significant': sr.significant,
                    'effect_size': sr.effect_size,
                }
                for (m1, m2), sr in stat_results.items()
            }
        }
        
        if progress_callback:
            automl_rank = sorted(
                cv_results.items(),
                key=lambda x: x[1]['mean_score'],
                reverse=True
            )
            rank_pos = [name for name, _ in automl_rank].index('AutoML') + 1
            progress_callback(
                f"   • AutoML ranking: #{rank_pos} de {len(cv_results)} modelos\n"
                f"   • AutoML μ={cv_results['AutoML']['mean_score']:.4f}",
                0.85
            )
    
    # =========================================================================
    # PHASE 4: EXPORT AND SAVE
    # =========================================================================
    if progress_callback:
        progress_callback("💾 FASE 4: Exportando modelo y metadatos...", 0.9)
    
    # Create timestamp
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results['training_timestamp'] = timestamp
    
    # Export the model
    model_path = export_best_model(
        automl_model=clf,
        output_dir=output_dir,
        model_name=f"automl_{preset}_{timestamp}",
        include_metadata=True,
        training_data=pd.concat([X, y], axis=1),
        target_column=y.name if hasattr(y, 'name') else 'target',
    )
    
    results['final_model_path'] = str(model_path)
    results['metadata_path'] = str(model_path).replace('.joblib', '_metadata.json')
    
    # Save test set if provided
    if test_set is not None:
        X_test, y_test = test_set
        testsets_dir = Path(output_dir) / "testsets"
        testsets_dir.mkdir(exist_ok=True)
        
        test_df = pd.concat([X_test, y_test], axis=1)
        test_path = testsets_dir / f"testset_automl_{timestamp}.parquet"
        test_df.to_parquet(test_path)
        results['test_set_path'] = str(test_path)
        
        if progress_callback:
            progress_callback(f"   • Test set guardado: {test_path}", 0.95)
    
    if progress_callback:
        progress_callback(
            f"✅ Pipeline AutoML completado exitosamente!\n"
            f"   • Modelo guardado: {model_path}\n"
            f"   • Duración total: {training_duration/60:.1f} min",
            1.0
        )
    
    return results


def is_automl_available() -> bool:
    """Check if AutoML functionality is available."""
    return AUTOML_AVAILABLE
