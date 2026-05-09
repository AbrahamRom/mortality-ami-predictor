"""UI utilities and helper functions for dashboard pages."""
from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Callable

import pandas as pd
import streamlit as st

from src.config import CONFIG
from src.data_load import (
    data_audit,
    load_dataset,
    summarize_dataframe,
    train_test_split,
    save_dataset_with_timestamp,
    cleanup_old_testsets,
    get_latest_model,
    get_model_combination_key,
    get_latest_testset,
    get_timestamp,
)
from src.features import safe_feature_columns
from src.preprocessing import PreprocessingConfig
from src.training import fit_and_save_best_classifier, fit_and_save_selected_classifiers
from .config import MODELS_DIR, TESTSETS_DIR, CLEANED_DATASETS_DIR, PLOTS_TRAINING_DIR


def display_dataframe_info(df: pd.DataFrame, title: str = "Dataset Info"):
    """Display comprehensive dataframe information.
    
    Args:
        df: DataFrame to display
        title: Title for the section
    """
    st.subheader(title)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Rows", f"{len(df):,}")
    with col2:
        st.metric("Columns", len(df.columns))
    with col3:
        missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        st.metric("Missing %", f"{missing_pct:.2f}%")


def display_dataset_preview(df: pd.DataFrame, n_rows: int = 10):
    """Display dataset preview with expandable sections.
    
    Args:
        df: DataFrame to preview
        n_rows: Number of rows to show
    """
    st.subheader("Data Preview")
    st.dataframe(df.head(n_rows), width='stretch')
    
    summaries = summarize_dataframe(df)
    
    with st.expander("📊 Missing Values Summary"):
        st.dataframe(summaries["missing"].head(50), width='stretch')
    
    with st.expander("📈 Statistical Description"):
        st.dataframe(summaries["describe"].head(50), width='stretch')
    
    with st.expander("🔤 Data Types"):
        st.dataframe(summaries["dtypes"], width='stretch')


def display_data_audit(df: pd.DataFrame, feature_cols: list[str]):
    """Display data quality audit results.
    
    Args:
        df: DataFrame to audit
        feature_cols: List of feature column names
    """
    audit = data_audit(df, feature_cols)
    
    st.subheader("🔍 Data Quality Audit")
    
    with st.expander("Preview"):
        st.dataframe(audit["head"], width='stretch')
    
    with st.expander("Missing Values by Column"):
        st.dataframe(audit["nan_summary"], width='stretch')
    
    if audit["full_missing"]:
        st.warning(f"⚠️ Completely empty columns: {', '.join(audit['full_missing'][:20])}")
    
    if audit["mostly_missing"]:
        st.info(f"ℹ️ Columns with >80% missing: {', '.join(audit['mostly_missing'][:20])}")
    
    if audit["constant"]:
        st.info(f"ℹ️ Constant/near-constant columns: {', '.join(audit['constant'][:20])}")


def get_default_data_path() -> str:
    """Get default dataset path.
    
    Returns:
        Default dataset path as string
    """
    try:
        default_abs = r"C:\Users\HP\Desktop\ML\Proyecto\mortality-ami-predictor\DATA\recuima-020425.xlsx"
        default_rel = os.path.relpath(default_abs, start=os.getcwd()) if os.path.isabs(default_abs) else default_abs
    except Exception:
        default_rel = "DATA\\recuima-020425.xlsx"
    
    return os.environ.get("DATASET_PATH", default_rel)


def sidebar_data_controls():
    """Render sidebar controls for data loading and task selection.
    
    Returns:
        Tuple of (data_path, task)
    """
    st.sidebar.header("⚙️ Configuration")
    
    # Display logo if available
    try:
        assets_dir = Path(__file__).parent / "assets"
        for pat in ("logo.png", "logo.jpg", "logo.jpeg", "logo.ico"):
            cand = assets_dir / pat
            if cand.exists():
                st.sidebar.image(str(cand), width='stretch')
                break
    except Exception:
        pass
    
    data_path = st.sidebar.text_input(
        "Dataset Path",
        value=get_default_data_path(),
        help="Path to the dataset file (CSV or Excel)"
    )
    
    task = st.sidebar.selectbox(
        "Prediction Task",
        ["mortality", "arrhythmia"],
        index=0,
        help="Select the prediction target"
    )
    
    return data_path, task


def sidebar_training_controls():
    """Render sidebar controls for model training.
    
    Returns:
        Tuple of (quick_mode, imputer_mode, selected_models)
    """
    st.sidebar.header("🎯 Training Settings")
    
    quick = st.sidebar.checkbox(
        "Quick Mode (Debug)",
        value=True,
        help="Use faster settings for debugging"
    )
    
    imputer_mode = st.sidebar.selectbox(
        "Imputation Strategy",
        ["iterative", "knn", "simple"],
        index=0,
        help="Method for handling missing values"
    )
    
    # Model selection
    from src.models import make_classifiers
    model_keys = list(make_classifiers().keys())
    
    selected_models = st.sidebar.multiselect(
        "Models to Train",
        model_keys,
        default=model_keys,
        help="Select which models to train and evaluate"
    )
    
    return quick, imputer_mode, selected_models


def train_models_with_progress(
    data_path: str,
    task: str,
    quick: bool,
    imputer_mode: str,
    selected_models: list[str],
    custom_model_classes: dict = None,
    target_column: str = None,
    imbalance_strategy: str = "smote",
    imbalance_k_neighbors: int = 5,
    imbalance_sampling_strategy: str = "auto",
    existing_model_actions: dict[str, str] | None = None,
) -> tuple[dict[str, str], dict[str, object]]:
    """Train selected models using the rigorous experiment pipeline.
    
    This function ALWAYS uses the rigorous academic pipeline with:
    - 30+ repeated CV runs for model selection (FASE 1)
    - Learning curves for diagnostics
    - Statistical comparison (Shapiro-Wilk, t-test/Mann-Whitney) (FASE 3)
    - Configurable class imbalance handling (SMOTE, ADASYN, etc.)
    
    NOTE: Bootstrap and Jackknife evaluation (FASE 2) is done in the 
    EVALUATION module, not here.
    
    Args:
        data_path: Path to dataset
        task: Task name (for organizing saved models, e.g., 'mortality', 'arrhythmia', 'custom')
        quick: Whether to use quick mode (fewer CV splits)
        imputer_mode: Imputation strategy
        selected_models: List of model names to train (includes custom models)
        custom_model_classes: Dictionary mapping custom model names to their classes
        target_column: Explicit target column name. If None, uses CONFIG defaults based on task.
        imbalance_strategy: Strategy for handling class imbalance ('smote', 'adasyn', 'class_weight', etc.)
        imbalance_k_neighbors: Number of neighbors for SMOTE variants
        imbalance_sampling_strategy: Sampling strategy for resampling ('auto', 'minority', etc.)
        existing_model_actions: Optional mapping model_name -> "reuse" | "retrain"
        
    Returns:
        Tuple of (save_paths, experiment_results)
    """
    if custom_model_classes is None:
        custom_model_classes = {}
    if existing_model_actions is None:
        existing_model_actions = {}
    
    # Load data
    df = load_dataset(data_path)
    
    # Define clinical score columns to preserve (these are needed for GRACE/RECUIMA comparison)
    # They will be saved in testset but NOT used as features for training
    # Important: These are preserved in their ORIGINAL form (before encoding/transformation)
    CLINICAL_SCORE_COLUMNS = [
        # GRACE score columns
        'escala_grace', 'GRACE', 'grace_score',
        # RECUIMA required variables (need original values, not encoded)
        'complicaciones',      # Text field with FV/TV, BAV info
        'indice_killip',       # Original 1-4 or I-IV (before 0-3 encoding)
        'edad',                # Age in years
        'presion_arterial_sistolica',  # SBP in mmHg
        'filtrado_glomerular', # GFR in ml/min/1.73m²
        # Pre-computed scores
        'recuima', 'score_recuima',
    ]
    
    # Find which clinical score columns exist in the data
    preserved_score_columns = [col for col in CLINICAL_SCORE_COLUMNS if col in df.columns]
    if preserved_score_columns:
        st.info(f"ℹ️ Columnas de scores clínicos detectadas: {preserved_score_columns}")
    
    # Determine target column
    # Priority: 1) explicit target_column, 2) CONFIG based on task, 3) fallback search
    if target_column and target_column in df.columns:
        target = target_column
    elif task == "mortality":
        target = CONFIG.target_column
    elif task == "arrhythmia":
        target = CONFIG.arrhythmia_column
    else:
        # For custom tasks, use target_column or task name directly
        if target_column:
            target = target_column
        else:
            target = task
    
    # Verify target exists in dataframe
    if target not in df.columns:
        # Try case-insensitive match
        matches = [c for c in df.columns if c.lower() == target.lower()]
        if matches:
            target = matches[0]
        else:
            raise ValueError(f"Target column '{target}' not found in dataset. Available: {list(df.columns[:20])}")
    
    # Split data (80% train, 20% test) with STRATIFICATION
    # Also get indices to preserve original data for clinical scores
    train_df, test_df, train_indices, test_indices = train_test_split(
        df, stratify_column=target, test_size=0.2, random_state=42, return_indices=True
    )
    
    # Display split information with class distribution
    st.info(f"""
    📊 **División de datos (estratificada por {target})**
    - Train: {len(train_df)} muestras ({len(train_df)/len(df)*100:.1f}%)
    - Test: {len(test_df)} muestras ({len(test_df)/len(df)*100:.1f}%)
    """)
    
    # Show class distribution to verify stratification
    col1, col2 = st.columns(2)
    with col1:
        train_dist = train_df[target].value_counts(normalize=True).sort_index()
        st.write("**Distribución Train:**")
        for label, prop in train_dist.items():
            st.write(f"  Clase {label}: {prop*100:.2f}%")
    
    with col2:
        test_dist = test_df[target].value_counts(normalize=True).sort_index()
        st.write("**Distribución Test:**")
        for label, prop in test_dist.items():
            st.write(f"  Clase {label}: {prop*100:.2f}%")
    
    # Verify stratification
    max_diff = abs(train_dist - test_dist).max()
    if max_diff < 0.05:  # Less than 5% difference
        st.success(f"✅ Estratificación correcta (diferencia máxima: {max_diff*100:.2f}%)")
    else:
        st.warning(f"⚠️ Posible desbalance en estratificación (diferencia: {max_diff*100:.2f}%)")
    
    # Save test set and train set immediately after split to new location
    # Model type will be determined later, for now save with task name
    # This will be updated when individual models are trained
    timestamp = get_timestamp()
    
    try:
        # Add original indices to test_df and train_df for later matching with clinical scores
        # This allows proper alignment with original data for GRACE/RECUIMA comparisons
        test_df_with_metadata = test_df.copy()
        train_df_with_metadata = train_df.copy()
        
        # Add original indices
        test_df_with_metadata['_original_index'] = test_indices
        train_df_with_metadata['_original_index'] = train_indices
        
        # Preserve clinical score columns with _ prefix (so they're not used as features)
        # These are essential for GRACE/RECUIMA comparison in evaluation
        if preserved_score_columns:
            for col in preserved_score_columns:
                if col in df.columns:
                    # Get values using the split indices
                    test_df_with_metadata[f'_score_{col}'] = df.iloc[test_indices][col].values
                    train_df_with_metadata[f'_score_{col}'] = df.iloc[train_indices][col].values
            st.success(f"✅ Columnas de scores clínicos preservadas: {preserved_score_columns}")
        
        # Save both train and test sets with timestamp (including indices and scores)
        test_path = save_dataset_with_timestamp(
            test_df_with_metadata, 
            TESTSETS_DIR, 
            prefix=f"testset_{task}",
            format="parquet"
        )
        train_path = save_dataset_with_timestamp(
            train_df_with_metadata,
            TESTSETS_DIR,
            prefix=f"trainset_{task}",
            format="parquet"
        )
        st.success(f"✅ Test set guardado: {len(test_df)} muestras en {test_path.name}")
        st.info(f"ℹ️ Train set guardado: {len(train_df)} muestras en {train_path.name}")
        if preserved_score_columns:
            st.info(f"ℹ️ Scores clínicos incluidos para comparación: {preserved_score_columns}")
        
    except Exception as e:
        st.error(f"❌ Error guardando train/test sets: {e}")
        raise
    
    # Prepare features
    X_train = train_df[safe_feature_columns(train_df, [target])]
    y_train = train_df[target]
    X_test = test_df[safe_feature_columns(test_df, [target])]
    y_test = test_df[target]
    
    # Progress tracking with real-time updates
    progress_bar = st.progress(0.0)
    status_text = st.empty()  # For main status message
    details_text = st.empty()  # For detailed progress info
    metrics_container = st.empty()  # For metrics display
    
    def progress_callback(msg: str, frac: float):
        """Callback with real-time progress updates."""
        # Update progress bar
        progress_bar.progress(min(max(frac, 0.0), 1.0))
        
        # Update status text
        status_text.info(f"🔄 {msg}")
        
        # Parse message to extract metrics if available
        if "μ=" in msg and "σ=" in msg:
            try:
                # Extract mean and std
                mu_part = msg.split("μ=")[1].split(",")[0].strip()
                sigma_part = msg.split("σ=")[1].split()[0].strip()
                
                # Display metrics only if valid numbers
                try:
                    mu_val = float(mu_part)
                    sigma_val = float(sigma_part)
                    
                    with metrics_container.container():
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Progreso", f"{frac*100:.1f}%")
                        with col2:
                            st.metric("AUROC Promedio (μ)", f"{mu_val:.4f}")
                        with col3:
                            st.metric("Desviación (σ)", f"{sigma_val:.4f}")
                except ValueError:
                    pass
            except:
                pass
        else:
            # If no metrics, just show progress
            with metrics_container.container():
                st.metric("Progreso", f"{frac*100:.1f}%")
    
    # Create preprocessing config with imbalance handling
    preprocessing_config = PreprocessingConfig()
    preprocessing_config.imputer_mode = imputer_mode
    preprocessing_config.imbalance_strategy = imbalance_strategy
    preprocessing_config.imbalance_k_neighbors = imbalance_k_neighbors
    preprocessing_config.imbalance_sampling_strategy = imbalance_sampling_strategy
    
    # Log imbalance configuration
    st.info(f"""
    ⚖️ **Configuración de Balanceo de Clases:**
    - Estrategia: **{imbalance_strategy.upper()}**
    - K-Neighbors: {imbalance_k_neighbors}
    - Sampling: {imbalance_sampling_strategy}
    """)
    
    # Get selected models
    from src.models import make_classifiers
    all_models = make_classifiers()
    
    # Add custom models if provided
    if custom_model_classes:
        for model_name, model_class in custom_model_classes.items():
            # Instantiate custom model with default parameters
            # Models need to be in format (base_model, param_grid)
            model_instance = model_class()
            
            # Get parameters from the model to create param_grid
            params = model_instance.get_params()
            # Create empty param_grid (no hyperparameter tuning for custom models by default)
            param_grid = {}
            
            all_models[model_name] = (model_instance, param_grid)
    
    models_to_train = {k: v for k, v in all_models.items() if k in selected_models}
    
    if not models_to_train:
        raise ValueError("No models selected for training")

    # Check existing models by model+resampling combination.
    # This allows reusing previous runs to avoid unnecessary retraining.
    models_for_retraining = {}
    reused_paths = {}

    for model_name, model_def in models_to_train.items():
        combination_key = get_model_combination_key(model_name, imbalance_strategy)
        existing_path = get_latest_model(combination_key, MODELS_DIR)

        if existing_path is None:
            models_for_retraining[model_name] = model_def
            continue

        action = existing_model_actions.get(model_name, "reuse")
        if action == "reuse":
            reused_paths[model_name] = str(existing_path)
            st.info(f"♻️ Reutilizando modelo existente: {model_name} ({combination_key})")
        else:
            models_for_retraining[model_name] = model_def
            st.warning(f"🔄 Reentrenando modelo: {model_name} ({combination_key})")
    
    # ALWAYS use rigorous pipeline
    status_text.info("🚀 Ejecutando pipeline de experimentación riguroso...")
    
    # Import rigorous pipeline
    from src.training import run_rigorous_experiment_pipeline
    
    # Set CV parameters based on quick mode
    n_splits = 3 if quick else 10
    n_repeats = 3 if quick else 10  # 9 or 100 runs total
    
    # Run pipeline (FASE 1 + FASE 3 - train/validation + statistical comparison)
    if models_for_retraining:
        experiment_results = run_rigorous_experiment_pipeline(
            X=X_train,
            y=y_train,
            models=models_for_retraining,
            preprocessing_config=preprocessing_config,
            n_splits=n_splits,
            n_repeats=n_repeats,
            scoring="roc_auc",
            test_set=(X_test, y_test),  # Saved for later evaluation
            output_dir=str(MODELS_DIR),
            progress_callback=progress_callback,
        )
    else:
        experiment_results = {
            "best_model": next(iter(reused_paths)) if reused_paths else None,
            "final_model_path": reused_paths[next(iter(reused_paths))] if reused_paths else None,
            "cv_results": {},
            "learning_curves": {},
            "saved_models": {},
        }
        progress_callback("✅ Se reutilizaron todos los modelos seleccionados", 1.0)
    
    # Extract save paths
    best_model_name = experiment_results.get('best_model')
    final_model_path = experiment_results.get('final_model_path')
    learning_curves = experiment_results.get('learning_curves', {})
    training_timestamp = experiment_results.get('training_timestamp', '')
    
    save_paths = dict(reused_paths)

    saved_models = experiment_results.get("saved_models", {})
    if saved_models:
        save_paths.update(saved_models)
    elif final_model_path and best_model_name:
        save_paths[best_model_name] = final_model_path
    
    # Store learning curve results and figure paths in session state
    # Use the training timestamp to locate the correct plots
    if learning_curves:
        lc_figure_paths = {}
        for model_name in learning_curves.keys():
            # Try with timestamp first (new format)
            if training_timestamp:
                lc_path = PLOTS_TRAINING_DIR / f"learning_curve_{model_name}_{training_timestamp}.png"
                if not lc_path.exists():
                    # Try HTML version
                    lc_path = PLOTS_TRAINING_DIR / f"learning_curve_{model_name}_{training_timestamp}.html"
            else:
                # Fallback to old format (without timestamp) for backward compatibility
                lc_path = MODELS_DIR / f"learning_curve_{model_name}.png"
                if not lc_path.exists():
                    lc_path = MODELS_DIR / f"learning_curve_{model_name}.html"
            
            if lc_path.exists():
                lc_figure_paths[model_name] = str(lc_path)
        
        st.session_state.learning_curve_paths = lc_figure_paths
        st.session_state.learning_curve_results = learning_curves
    
    # Store experiment results for statistical comparison display
    st.session_state.experiment_results = experiment_results
    
    retrained_count = len(saved_models)
    reused_count = len(reused_paths)
    if retrained_count > 0:
        status_text.success(
            f"✅ Pipeline de entrenamiento completado!\n"
            f"Modelos reentrenados: {retrained_count}\n"
            f"Modelos reutilizados: {reused_count}\n"
            f"Mejor modelo: {best_model_name}\n"
            f"Modelo destacado en: {final_model_path}\n"
            f"Test set guardado en: {test_path}\n"
            f"⚠️ La evaluación final (Bootstrap/Jackknife) se hará en el módulo de EVALUACIÓN"
        )
    else:
        status_text.success(
            f"✅ No se reentrenaron modelos (todos reutilizados).\n"
            f"Modelos reutilizados: {reused_count}\n"
            f"⚠️ La evaluación final (Bootstrap/Jackknife) se hará en el módulo de EVALUACIÓN"
        )
    
    progress_bar.progress(1.0)
    
    return save_paths, experiment_results


def list_saved_models(task: str) -> dict[str, str]:
    """List all saved models for a given task.
    
    Args:
        task: Task name (mortality/arrhythmia)
        
    Returns:
        Dictionary mapping model names to file paths
    """
    mapping = {}
    
    if MODELS_DIR.exists():
        # Look in model directories (model type or model+resampling combination)
        from src.data_load import get_all_model_types, get_latest_model
        
        for model_type in get_all_model_types(MODELS_DIR):
            latest_model = get_latest_model(model_type, MODELS_DIR)
            if latest_model:
                mapping[model_type] = str(latest_model)
        
        # Also check for best classifier (legacy format)
        best = MODELS_DIR / f"best_classifier_{task}.joblib"
        if best.exists():
            mapping["best"] = str(best)
    
    return mapping


def list_saved_model_versions(task: str, max_per_combination: int | None = None) -> dict[str, str]:
    """List historical saved model artifacts across runs and resampling strategies.

    Args:
        task: Task name used to filter metadata when available.
        max_per_combination: Optional max number of versions per model directory.

    Returns:
        Dictionary mapping UI label -> absolute model path.
    """
    mapping: dict[str, str] = {}
    standard_task_aliases: dict[str, set[str]] = {
        "mortality": {"mortality", "mortality_inhospital", "exitus"},
        "arrhythmia": {"arrhythmia", "ventricular_arrhythmia"},
    }

    if not MODELS_DIR.exists():
        return mapping

    for subdir in sorted(MODELS_DIR.iterdir(), key=lambda p: p.name):
        if not subdir.is_dir() or subdir.name in {"testsets", "__pycache__"}:
            continue

        model_files = sorted(
            subdir.glob("model_*.joblib"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        if max_per_combination is not None:
            model_files = model_files[:max(0, max_per_combination)]

        for model_file in model_files:
            include_model = True
            metadata_file = model_file.with_suffix(".metadata.json")

            if metadata_file.exists():
                try:
                    with open(metadata_file, "r", encoding="utf-8") as f:
                        metadata = json.load(f)
                    metadata_task = str(metadata.get("task", "")).strip().lower()

                    if metadata_task:
                        if task in standard_task_aliases:
                            include_model = metadata_task in standard_task_aliases[task]
                        else:
                            include_model = metadata_task == task
                except Exception:
                    include_model = True

            if include_model:
                label = f"{subdir.name} | {model_file.name}"
                mapping[label] = str(model_file)

    return mapping


def display_model_list(task: str):
    """Display available saved models.
    
    Args:
        task: Task name (mortality/arrhythmia)
    """
    models = list_saved_models(task)
    
    if not models:
        st.warning(f"⚠️ No saved models found for task '{task}'")
        return
    
    st.success(f"✅ Found {len(models)} saved model(s)")
    
    for name, path in models.items():
        with st.expander(f"📦 {name}"):
            st.code(path, language="text")


def format_metric_value(value: float, metric_name: str) -> str:
    """Format metric value for display.
    
    Args:
        value: Metric value
        metric_name: Name of metric
        
    Returns:
        Formatted string
    """
    if metric_name.lower() in ("accuracy", "precision", "recall", "f1", "auc", "roc_auc"):
        return f"{value:.4f}"
    return f"{value:.2f}"


def display_metrics_table(metrics: dict[str, float], title: str = "Metrics"):
    """Display metrics in a formatted table.
    
    Args:
        metrics: Dictionary of metric names and values
        title: Title for the metrics section
    """
    st.subheader(title)
    
    # Create DataFrame
    df = pd.DataFrame([
        {"Metric": k, "Value": format_metric_value(v, k)}
        for k, v in metrics.items()
    ])
    
    st.dataframe(df, width='stretch', hide_index=True)

