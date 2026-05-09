"""Model Training page."""
from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime

# Add parent directories to path
root_dir = Path(__file__).parents[2]
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import joblib
import tempfile

from app import (
    display_model_list,
    get_state,
    initialize_state,
    list_saved_model_versions,
    set_state,
    sidebar_training_controls,
    train_models_with_progress,
)
from app.config import PLOTS_TRAINING_DIR, MODELS_DIR, TESTSETS_DIR
from src.data_load import get_latest_plot, get_latest_model, get_model_combination_key, get_latest_testset
from src.training import generate_training_pdf
from src.training import compare_multiple_models
from src.evaluation import compute_classification_metrics, bootstrap_evaluation
from src.reporting import pdf_export_section
from src.features import ICATransformer
from src.config import CONFIG, RANDOM_SEED

# Initialize
initialize_state()

# Page config
st.title("🤖 Model Training")
st.markdown("---")

# Check if data has been loaded
cleaned_data = st.session_state.get('cleaned_data')
raw_data = st.session_state.get('raw_data')

if cleaned_data is not None:
    df = cleaned_data
    data_path = st.session_state.get('data_path')
    st.success("✅ Usando datos limpios del proceso de limpieza")
elif raw_data is not None:
    df = raw_data
    data_path = st.session_state.get('data_path')
    st.warning("⚠️ Usando datos crudos (se recomienda limpiar primero)")
else:
    st.warning("⚠️ No hay datos cargados. Por favor, carga un dataset en la página **🧹 Data Cleaning and EDA** primero.")
    st.stop()

# Si no hay data_path o el path no existe, crear un archivo temporal
import tempfile
if not data_path or not Path(data_path).exists():
    st.info("ℹ️ Guardando datos en archivo temporal para el entrenamiento...")
    temp_dir = Path(tempfile.gettempdir())
    data_path = temp_dir / "streamlit_training_dataset.csv"
    df.to_csv(data_path, index=False)
    st.session_state.data_path = str(data_path)
    st.success(f"✅ Dataset guardado en: {data_path}")

# ==================== TARGET VARIABLE SELECTION ====================
st.sidebar.markdown("---")
st.sidebar.header("🎯 Variable Objetivo")

# Get available columns for target selection (binary/numeric columns)
potential_targets = []
for col in df.columns:
    # Check if column could be a valid target (binary or numeric with few unique values)
    if df[col].nunique() <= 10:  # Categorical or binary
        potential_targets.append(col)
    elif pd.api.types.is_numeric_dtype(df[col]):
        potential_targets.append(col)

# Determine default target from session_state or CONFIG
saved_target = st.session_state.get('target_column_name', None)
default_target = None

# Priority: 1) saved from Data Cleaning, 2) CONFIG.target_column, 3) first potential target
if saved_target and saved_target in potential_targets:
    default_target = saved_target
elif CONFIG.target_column in potential_targets:
    default_target = CONFIG.target_column
elif potential_targets:
    default_target = potential_targets[0]

# Create target selector
default_idx = potential_targets.index(default_target) if default_target in potential_targets else 0

target_col = st.sidebar.selectbox(
    "Variable a Predecir",
    potential_targets,
    index=default_idx,
    help="Selecciona la variable objetivo para el entrenamiento. Se recomienda seleccionarla primero en Data Cleaning."
)

# Save selection to session_state
st.session_state.target_column_name = target_col
st.session_state.target_column = target_col

# Determine task name for model organization (used for saving models in folders)
# If it's a known target, use the standard name; otherwise use 'custom'
if target_col == CONFIG.target_column or target_col in ['mortality', 'mortality_inhospital', 'exitus']:
    task = 'mortality'
elif target_col == CONFIG.arrhythmia_column or target_col in ['arrhythmia', 'ventricular_arrhythmia']:
    task = 'arrhythmia'
else:
    # Custom target - use a sanitized version of the column name
    task = target_col.lower().replace(' ', '_')[:20]  # Limit length for folder names

# Show target info
target_info = df[target_col].value_counts()
st.sidebar.markdown(f"**Distribución:**")
for val, count in target_info.head(5).items():
    pct = count / len(df) * 100
    st.sidebar.markdown(f"- `{val}`: {count} ({pct:.1f}%)")

if df[target_col].nunique() > 5:
    st.sidebar.caption(f"... y {df[target_col].nunique() - 5} valores más")

# Custom models section
st.sidebar.markdown("---")
st.sidebar.header("🔧 Custom Models")

use_custom_models = st.sidebar.checkbox(
    "Include Custom Models",
    value=False,
    help="Include custom models defined in Custom Models page"
)

custom_models_list = []
custom_model_classes = {}

if use_custom_models:
    import importlib.util
    import inspect
    import sys
    from src.models.custom_base import BaseCustomModel, BaseCustomClassifier, BaseCustomRegressor
    
    # Buscar archivos .py con definiciones de modelos custom
    code_templates_dir = root_dir / "src" / "models" / "custom"
    code_templates_dir.mkdir(parents=True, exist_ok=True)
    
    available_files = sorted([f for f in code_templates_dir.glob("*.py") if f.name != "__init__.py"])
    
    if available_files:
        st.sidebar.markdown(f"**Available: {len(available_files)} file(s)**")
        
        # Extraer clases de cada archivo
        available_classes = []
        for filepath in available_files:
            try:
                # Use unique module name for each file to avoid conflicts
                module_name = f"custom_models.{filepath.stem}"
                
                # Check if module is already loaded
                if module_name in sys.modules:
                    module = sys.modules[module_name]
                else:
                    spec = importlib.util.spec_from_file_location(module_name, filepath)
                    module = importlib.util.module_from_spec(spec)
                    sys.modules[module_name] = module  # Register in sys.modules for pickle
                    spec.loader.exec_module(module)
                
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if issubclass(obj, (BaseCustomClassifier, BaseCustomRegressor)):
                        if obj not in [BaseCustomModel, BaseCustomClassifier, BaseCustomRegressor]:
                            display_name = f"{name} ({filepath.stem})"
                            available_classes.append(display_name)
                            custom_model_classes[display_name] = obj
            except Exception as e:
                st.sidebar.warning(f"⚠️ Error loading {filepath.name}: {e}")
        
        if available_classes:
            custom_models_list = st.sidebar.multiselect(
                "Select Custom Models",
                available_classes,
                help="Select which custom models to include in training"
            )
            
            if custom_models_list:
                st.sidebar.success(f"✅ {len(custom_models_list)} custom model(s) selected")
        else:
            st.sidebar.warning("⚠️ No valid model classes found in files")
    else:
        st.sidebar.info("📭 No custom models available. Create one in Custom Models page (🔧).")

# ==================== AUTOML SECTION ====================
st.sidebar.markdown("---")
st.sidebar.header("🤖 AutoML")

# Check AutoML availability
try:
    from src.training import is_automl_available
    automl_available = is_automl_available()
except ImportError:
    automl_available = False

if automl_available:
    use_automl = st.sidebar.checkbox(
        "Enable AutoML",
        value=False,
        help="Use automated machine learning to find the best model"
    )
    
    if use_automl:
        # AutoML preset selection
        automl_preset = st.sidebar.selectbox(
            "AutoML Preset",
            ["quick", "balanced", "high_performance"],
            index=0,
            help="quick: 5 min | balanced: 1 hour | high_performance: 4 hours"
        )
        
        # Time budget override (optional)
        automl_time = st.sidebar.number_input(
            "Custom Time Budget (seconds)",
            min_value=60,
            max_value=28800,
            value={"quick": 300, "balanced": 3600, "high_performance": 14400}[automl_preset],
            help="Override the preset time budget"
        )
        
        # Backend info
        try:
            from src.automl import is_flaml_available, is_autosklearn_available
            if is_autosklearn_available():
                backend = "auto-sklearn"
            elif is_flaml_available():
                backend = "FLAML"
            else:
                backend = "Not available"
            st.sidebar.info(f"📦 Backend: **{backend}**")
        except ImportError:
            st.sidebar.info("📦 Backend: FLAML (default)")
        
        # Store in session state
        st.session_state.automl_enabled = True
        st.session_state.automl_preset = automl_preset
        st.session_state.automl_time = automl_time
        
        st.sidebar.success("✅ AutoML enabled")
        st.sidebar.markdown(f"⏱️ Time: {automl_time//60} min")
        
        # Link to full AutoML page
        st.sidebar.markdown("---")
        st.sidebar.info("💡 For advanced AutoML options, visit **🤖 AutoML** page")
    else:
        st.session_state.automl_enabled = False
else:
    st.sidebar.warning("⚠️ AutoML not available")
    st.sidebar.markdown("""
    Install FLAML to enable:
    ```
    pip install flaml[automl]
    ```
    """)
    st.session_state.automl_enabled = False

# ==================== CLASS IMBALANCE HANDLING ====================
st.sidebar.markdown("---")
st.sidebar.header("⚖️ Balanceo de Clases")

# Check imbalanced-learn availability
try:
    from src.preprocessing.imbalance import (
        detect_imbalance,
        get_recommended_strategy,
        ImbalanceStrategy,
        STRATEGY_DESCRIPTIONS,
        is_imblearn_available,
    )
    IMBALANCE_AVAILABLE = is_imblearn_available()
except ImportError:
    IMBALANCE_AVAILABLE = False

# Detect class imbalance
if df is not None and target_col in df.columns:
    y_target = df[target_col]
    try:
        is_imbalanced, imbalance_ratio, class_counts = detect_imbalance(y_target)
        
        # Show imbalance info
        if is_imbalanced:
            st.sidebar.warning(f"⚠️ **Dataset Desbalanceado**")
            st.sidebar.markdown(f"Ratio: **{imbalance_ratio:.1f}:1**")
        else:
            st.sidebar.success(f"✅ Dataset Balanceado (ratio {imbalance_ratio:.1f}:1)")
        
        # Show class distribution bar
        total = sum(class_counts.values())
        for cls, count in sorted(class_counts.items()):
            pct = count / total * 100
            st.sidebar.progress(pct / 100, text=f"Clase {cls}: {count} ({pct:.1f}%)")
        
    except Exception as e:
        is_imbalanced = True  # Assume imbalanced for safety
        imbalance_ratio = 1.0
        st.sidebar.info("ℹ️ No se pudo analizar el desbalance")
else:
    is_imbalanced = True
    imbalance_ratio = 1.0

# Imbalance strategy selector
if IMBALANCE_AVAILABLE:
    # Define available strategies with display names
    strategy_options = {
        "smote": "🔄 SMOTE (Recomendado)",
        "adasyn": "📈 ADASYN (Adaptativo)",
        "borderline_smote": "🎯 Borderline-SMOTE",
        "smote_tomek": "🔄+🧹 SMOTE + Tomek Links",
        "smote_enn": "🔄+✂️ SMOTE + ENN",
        "class_weight": "⚖️ Class Weight (sin resampleo)",
        "random_oversample": "📊 Random Oversampling",
        "none": "❌ Sin Balanceo",
    }
    
    # Get recommended strategy
    try:
        X_features = df.drop(columns=[target_col]) if target_col in df.columns else df
        recommended = get_recommended_strategy(y_target, X_features)
        recommended_key = recommended.value
    except:
        recommended_key = "smote"
    
    # Default to recommended strategy
    default_idx = list(strategy_options.keys()).index(recommended_key) if recommended_key in strategy_options else 0
    
    imbalance_strategy = st.sidebar.selectbox(
        "Estrategia de Balanceo",
        options=list(strategy_options.keys()),
        format_func=lambda x: strategy_options[x],
        index=default_idx,
        help="Técnica para manejar el desbalance de clases"
    )
    
    # Show strategy description
    try:
        strategy_enum = ImbalanceStrategy(imbalance_strategy)
        desc = STRATEGY_DESCRIPTIONS.get(strategy_enum, {})
        if desc:
            with st.sidebar.expander("ℹ️ Acerca de esta estrategia"):
                st.markdown(f"**{desc.get('name', '')}**")
                st.markdown(desc.get('description', ''))
                st.markdown(f"✅ **Ventajas:** {desc.get('pros', '')}")
                st.markdown(f"❌ **Desventajas:** {desc.get('cons', '')}")
                st.markdown(f"💡 **Recomendado para:** {desc.get('recommended_for', '')}")
    except:
        pass
    
    # Advanced options for SMOTE variants
    if imbalance_strategy in ['smote', 'adasyn', 'borderline_smote', 'smote_tomek', 'smote_enn']:
        with st.sidebar.expander("⚙️ Opciones Avanzadas"):
            imbalance_k_neighbors = st.slider(
                "K-Neighbors (SMOTE)",
                min_value=1,
                max_value=15,
                value=5,
                help="Número de vecinos para generación de datos sintéticos"
            )
            
            imbalance_sampling_strategy = st.selectbox(
                "Sampling Strategy",
                ["auto", "minority", "not majority", "all"],
                index=0,
                help="'auto' balancea las clases automáticamente"
            )
    else:
        imbalance_k_neighbors = 5
        imbalance_sampling_strategy = "auto"
    
    # Store in session state
    st.session_state.imbalance_strategy = imbalance_strategy
    st.session_state.imbalance_k_neighbors = imbalance_k_neighbors
    st.session_state.imbalance_sampling_strategy = imbalance_sampling_strategy
    
else:
    st.sidebar.warning("⚠️ imbalanced-learn no instalado")
    st.sidebar.markdown("""
    Para habilitar SMOTE/ADASYN:
    ```
    pip install imbalanced-learn
    ```
    """)
    imbalance_strategy = "none"
    st.session_state.imbalance_strategy = "none"

# Training settings
st.sidebar.markdown("---")
st.sidebar.header("⚙️ Training Configuration")



quick, imputer_mode, selected_models = sidebar_training_controls()

# Combine standard models with custom models
all_selected_models = list(selected_models) if selected_models else []
if custom_models_list:
    all_selected_models.extend(custom_models_list)

# Main content
st.subheader("Training Configuration")

col1, col2 = st.columns(2)

with col1:
    st.metric("Task", task.capitalize())
    st.metric("Imputation", imputer_mode.capitalize())

with col2:
    st.metric("Quick Mode", "Enabled" if quick else "Disabled")
    st.metric("Models Selected", len(all_selected_models))

# Display selected models
if all_selected_models:
    if selected_models:
        st.info(f"📦 Standard models: {', '.join(selected_models)}")
    if custom_models_list:
        st.success(f"🔧 Custom models: {', '.join(custom_models_list)}")
else:
    st.warning("⚠️ No models selected for training")

# Existing manual models control (model + resampling combination)
existing_model_actions = {}
if all_selected_models and not st.session_state.get('automl_enabled', False):
    existing_models_detected = False
    with st.expander("♻️ Modelos existentes detectados (modelo + resampling)", expanded=False):
        for model_name in all_selected_models:
            combination_key = get_model_combination_key(model_name, imbalance_strategy)
            existing_model_path = get_latest_model(combination_key, MODELS_DIR)
            if existing_model_path:
                existing_models_detected = True
                st.markdown(f"**{combination_key}**")
                st.caption(f"Encontrado: {existing_model_path}")
                action = st.selectbox(
                    f"Acción para {combination_key}",
                    options=["Reutilizar existente", "Reentrenar"],
                    index=0,
                    key=f"existing_action_{combination_key}",
                )
                existing_model_actions[model_name] = "reuse" if action == "Reutilizar existente" else "retrain"

        if not existing_models_detected:
            st.caption("No se encontraron combinaciones existentes para los modelos seleccionados.")

st.markdown("---")

# ==================== FEATURE TRANSFORMATION SELECTOR ====================
st.subheader("🔄 Feature Transformation")

with st.expander("ℹ️ ¿Qué transformación usar?", expanded=False):
    st.markdown("""
    **Opciones de transformación de features:**
    
    - **🔤 Original Features:** Entrena con las variables originales sin transformación.
      - ✅ Interpretabilidad directa de features
      - ✅ No requiere procesamiento adicional
      - ❌ Alta dimensionalidad si hay muchas variables
    
    - **📊 PCA (Principal Component Analysis):** Reducción de dimensionalidad maximizando varianza.
      - ✅ Reduce multicolinealidad
      - ✅ Menor dimensionalidad = entrenamiento más rápido
      - ✅ Componentes ordenados por importancia (varianza)
      - ❌ Pérdida de interpretabilidad directa
      - 💡 Mejor para datos Gaussianos / lineales
    
    - **🧬 ICA (Independent Component Analysis):** Separación de fuentes independientes.
      - ✅ Encuentra patrones no-Gaussianos
      - ✅ Componentes estadísticamente independientes
      - ✅ Útil para separar señales mezcladas
      - ❌ No ordena componentes por importancia
      - 💡 Mejor para datos no-Gaussianos con múltiples fuentes
    
    **El transformer será guardado junto con el modelo para aplicarlo automáticamente en predicciones.**
    """)

transformation_type = st.radio(
    "Selecciona tipo de features para entrenamiento:",
    ["🔤 Original Features", "📊 PCA Components", "🧬 ICA Components"],
    help="El modelo se entrenará con el tipo de features seleccionado"
)

# Initialize transformation session state
if 'transformation_applied' not in st.session_state:
    st.session_state.transformation_applied = False
    st.session_state.transformer = None
    st.session_state.transformed_df = None
    st.session_state.transformation_params = {}

# Configuration based on transformation type
if transformation_type == "📊 PCA Components" or transformation_type == "🧬 ICA Components":
    st.markdown("### ⚙️ Configuración de Transformación")
    
    col1, col2, col3 = st.columns(3)
    
    # Get numeric columns (exclude target)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)
    
    n_features = len(numeric_cols)
    
    with col1:
        if transformation_type == "📊 PCA Components":
            pca_mode = st.radio(
                "Modo de selección",
                ["Varianza", "Número fijo"],
                help="Varianza: selecciona automáticamente | Número fijo: especifica cantidad"
            )
            
            if pca_mode == "Varianza":
                variance_threshold = st.slider(
                    "Varianza acumulada deseada",
                    0.70, 0.99, 0.95, 0.01,
                    format="%.2f",
                    help="Porcentaje de varianza a capturar"
                )
                n_components = None
            else:
                n_components = st.slider(
                    "Número de componentes",
                    2, min(20, n_features), min(10, n_features),
                    help="Componentes PCA a extraer"
                )
                variance_threshold = None
        
        else:  # ICA
            n_components = st.slider(
                "Número de componentes",
                2, min(20, n_features), min(10, n_features),
                help="Componentes independientes a extraer"
            )
    
    with col2:
        if transformation_type == "🧬 ICA Components":
            ica_algorithm = st.selectbox(
                "Algoritmo ICA",
                ["parallel", "deflation"],
                help="parallel: simultáneo | deflation: secuencial"
            )
            
            ica_fun = st.selectbox(
                "Función de contraste",
                ["logcosh", "exp", "cube"],
                help="logcosh: general | exp: super-Gaussiano | cube: sub-Gaussiano"
            )
    
    with col3:
        standardize = st.checkbox(
            "Estandarizar datos",
            value=True,
            help="Recomendado para PCA/ICA (escala variables)"
        )
        
        if transformation_type == "🧬 ICA Components":
            whiten = st.checkbox(
                "Whitening",
                value=True,
                help="Pre-procesamiento para ICA (recomendado)"
            )
    
    # Apply transformation button
    if st.button("🔄 Aplicar Transformación", type="secondary", use_container_width=True):
        with st.spinner(f"Aplicando {'PCA' if transformation_type == '📊 PCA Components' else 'ICA'}..."):
            try:
                # Prepare data (only numeric columns, drop NaNs)
                df_for_transform = df[numeric_cols].dropna()
                
                if len(df_for_transform) == 0:
                    st.error("❌ No hay datos válidos después de eliminar NaNs. Aplica imputación primero.")
                    st.stop()
                
                if transformation_type == "📊 PCA Components":
                    # Apply PCA
                    from sklearn.preprocessing import StandardScaler
                    
                    # Standardize if requested
                    if standardize:
                        scaler = StandardScaler()
                        data_scaled = scaler.fit_transform(df_for_transform)
                    else:
                        data_scaled = df_for_transform.values
                        scaler = None
                    
                    # Fit PCA
                    if pca_mode == "Varianza":
                        pca = PCA(n_components=variance_threshold, random_state=42)
                    else:
                        pca = PCA(n_components=n_components, random_state=42)
                    
                    components = pca.fit_transform(data_scaled)
                    
                    # Create DataFrame
                    component_names = [f'PC{i+1}' for i in range(pca.n_components_)]
                    transformed_df = pd.DataFrame(
                        components,
                        columns=component_names,
                        index=df_for_transform.index
                    )
                    
                    # Add target back
                    transformed_df[target_col] = df.loc[transformed_df.index, target_col]
                    
                    # Store in session state
                    st.session_state.transformer = {'pca': pca, 'scaler': scaler}
                    st.session_state.transformed_df = transformed_df
                    st.session_state.transformation_applied = True
                    st.session_state.transformation_params = {
                        'type': 'pca',
                        'n_components': pca.n_components_,
                        'variance_explained': sum(pca.explained_variance_ratio_),
                        'standardize': standardize,
                        'feature_names': numeric_cols
                    }
                    
                    st.success(
                        f"✅ PCA aplicado exitosamente: {pca.n_components_} componentes | "
                        f"Varianza explicada: {sum(pca.explained_variance_ratio_)*100:.2f}%"
                    )
                
                else:  # ICA
                    # Apply ICA
                    # Convert boolean whiten to string for newer sklearn versions
                    whiten_param = 'unit-variance' if whiten else False
                    
                    ica = ICATransformer(
                        n_components=n_components,
                        algorithm=ica_algorithm,
                        fun=ica_fun,
                        whiten=whiten_param,
                        max_iter=500,
                        random_state=42
                    )
                    
                    ica.fit(df_for_transform)
                    transformed_df = ica.transform(df_for_transform)
                    
                    # Add target back
                    transformed_df[target_col] = df.loc[transformed_df.index, target_col]
                    
                    # Store in session state
                    st.session_state.transformer = ica
                    st.session_state.transformed_df = transformed_df
                    
                    st.session_state.transformation_params = {
                        'type': 'ica',
                        'n_components': n_components,
                        'algorithm': ica_algorithm,
                        'fun': ica_fun,
                        'whiten': whiten,
                        'kurtosis_mean': float(np.mean(np.abs(ica.result_.component_kurtosis))),
                        'feature_names': numeric_cols
                    }
                    st.session_state.transformation_applied = True
                    
                    st.success(
                        f"✅ ICA aplicado exitosamente: {n_components} componentes independientes | "
                        f"Kurtosis promedio: {np.mean(np.abs(ica.result_.component_kurtosis)):.3f}"
                    )
                
                # Show preview
                st.markdown("#### 📋 Preview de datos transformados")
                st.dataframe(transformed_df.head(10), width='stretch')
                st.info(f"Shape: {transformed_df.shape} | Target column: {target_col}")
                
            except Exception as e:
                st.error(f"❌ Error durante transformación: {e}")
                import traceback
                with st.expander("Ver traceback"):
                    st.code(traceback.format_exc())

# Show transformation status
if transformation_type != "🔤 Original Features":
    if st.session_state.transformation_applied:
        params = st.session_state.transformation_params
        if params['type'] == 'pca':
            st.success(
                f"✅ **PCA activo:** {params['n_components']} componentes | "
                f"Varianza: {params['variance_explained']*100:.2f}%"
            )
        else:  # ICA
            st.success(
                f"✅ **ICA activo:** {params['n_components']} componentes | "
                f"Kurtosis: {params['kurtosis_mean']:.3f}"
            )
    else:
        st.warning("⚠️ Transformación configurada pero no aplicada. Haz clic en '🔄 Aplicar Transformación'.")

# Training section
st.markdown("---")
st.subheader("Train Models")

# Show AutoML status if enabled
if st.session_state.get('automl_enabled', False):
    st.info(f"""
    🤖 **AutoML Enabled**
    - Preset: {st.session_state.get('automl_preset', 'balanced')}
    - Time Budget: {st.session_state.get('automl_time', 3600) // 60} minutes
    - AutoML will search for the best model automatically
    """)

if not all_selected_models and not st.session_state.get('automl_enabled', False):
    st.error("❌ Please select at least one model from the sidebar or enable AutoML")
else:
    # Initialize training state
    if 'is_training' not in st.session_state:
        st.session_state.is_training = False
    
    # Show button or training message
    if not st.session_state.is_training:
        # Different button text based on AutoML status
        if st.session_state.get('automl_enabled', False):
            button_text = "🤖 Start AutoML Training"
        else:
            button_text = "🚀 Start Training"
        start_button = st.button(button_text, type="primary", use_container_width=True)
    else:
        st.info("⏳ **Training in progress, please wait...**")
    
    if not st.session_state.is_training and 'start_button' in locals() and start_button:
        # Set training flag
        st.session_state.is_training = True
        
        try:
            # Determine which dataset to use based on transformation selection
            if transformation_type != "🔤 Original Features":
                if st.session_state.transformation_applied and st.session_state.transformed_df is not None:
                    df_for_training = st.session_state.transformed_df.copy()
                    params = st.session_state.transformation_params
                    transform_type = "PCA" if params['type'] == 'pca' else "ICA"
                    st.info(
                        f"ℹ️ Entrenando con **{transform_type}**: {params['n_components']} componentes "
                        f"(transformación aplicada a {len(params['feature_names'])} variables originales)"
                    )
                else:
                    st.error(
                        f"❌ Has seleccionado transformación {transformation_type} pero no la has aplicado. "
                        f"Haz clic en '🔄 Aplicar Transformación' primero."
                    )
                    st.session_state.is_training = False
                    st.stop()
            else:
                df_for_training = df.copy()
                st.info(f"ℹ️ Entrenando con **features originales**: {len(df.columns)-1} variables")
            
            # ==================== AUTOML TRAINING ====================
            if st.session_state.get('automl_enabled', False):
                st.markdown("### 🤖 AutoML Training")
                
                # Prepare data
                X = df_for_training.drop(columns=[target_col])
                y = df_for_training[target_col]
                
                # Progress display
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                def automl_progress_callback(msg: str, progress: float):
                    progress_bar.progress(progress)
                    status_text.markdown(f"**{msg}**")
                
                try:
                    from src.training import run_automl_experiment_pipeline
                    
                    # Run AutoML
                    automl_results = run_automl_experiment_pipeline(
                        X=X,
                        y=y,
                        preset=st.session_state.get('automl_preset', 'quick'),
                        time_budget=st.session_state.get('automl_time', 300),
                        metric="roc_auc",
                        include_suggestions=True,
                        compare_with_manual=len(all_selected_models) > 0,
                        progress_callback=automl_progress_callback,
                    )
                    
                    # Display results
                    st.success(f"""
                    ✅ **AutoML Completed!**
                    
                    - Best Model: {automl_results.get('best_estimator', 'Unknown')}
                    - Best Score (AUC): {automl_results.get('best_score', 0):.4f}
                    - Training Time: {automl_results.get('training_duration', 0) / 60:.1f} minutes
                    - Backend: {automl_results.get('backend', 'FLAML')}
                    """)
                    
                    # Show leaderboard
                    if 'leaderboard' in automl_results and not automl_results['leaderboard'].empty:
                        st.markdown("### 📊 Model Leaderboard")
                        st.dataframe(automl_results['leaderboard'].head(10), use_container_width=True)
                    
                    # Show suggestions
                    if 'suggestions' in automl_results and automl_results['suggestions']:
                        with st.expander("💡 Dataset Suggestions", expanded=False):
                            for s in automl_results['suggestions'][:5]:
                                priority_icon = "🔴" if s['priority'] == 'high' else "🟡" if s['priority'] == 'medium' else "🟢"
                                st.markdown(f"**{priority_icon} {s['title']}**")
                                st.markdown(f"  {s['description']}")
                                if s.get('module_link'):
                                    st.markdown(f"  → Module: `{s['module_link']}`")
                    
                    # Store results
                    st.session_state.automl_results = automl_results
                    st.session_state.training_results = automl_results
                    set_state("is_trained", True)
                    
                    # Show model path
                    if 'final_model_path' in automl_results:
                        st.info(f"📁 Model saved to: `{automl_results['final_model_path']}`")
                    
                    st.balloons()
                    
                except ImportError as e:
                    st.error(f"❌ AutoML not available: {e}")
                    st.info("Install FLAML: `pip install flaml[automl]`")
                except Exception as e:
                    st.error(f"❌ AutoML Error: {e}")
                    import traceback
                    with st.expander("Error details"):
                        st.code(traceback.format_exc())
                
                st.session_state.is_training = False
                st.stop()  # Stop here if AutoML was used
            
            # ==================== STANDARD TRAINING ====================
            # Save the DataFrame that will actually be used for training
            # This ensures metadata will reflect the correct features
            temp_dir = Path(tempfile.gettempdir())
            training_data_path = temp_dir / f"streamlit_training_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            df_for_training.to_csv(training_data_path, index=False)
            st.success(f"✅ Dataset para entrenamiento guardado: {len(df_for_training.columns)} columnas (incluyendo target)")
            
            # Save transformer if using transformation
            transformer_path = None
            if transformation_type != "🔤 Original Features" and st.session_state.transformer is not None:
                transformer_path = temp_dir / f"streamlit_transformer_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib"
                joblib.dump(st.session_state.transformer, str(transformer_path))
                st.success(f"✅ Transformer guardado temporalmente: {transformer_path.name}")
            
            # Create containers for progress display
            progress_container = st.empty()
            status_container = st.empty()
            
            # Capture stdout to show progress
            import io
            from contextlib import redirect_stdout
            
            with status_container.container():
                st.markdown("### 📊 Progreso del Entrenamiento")
                progress_area = st.empty()
                
                # Redirect stdout
                output_buffer = io.StringIO()
                
                # Get imbalance settings from session state
                imb_strategy = st.session_state.get('imbalance_strategy', 'smote')
                imb_k_neighbors = st.session_state.get('imbalance_k_neighbors', 5)
                imb_sampling = st.session_state.get('imbalance_sampling_strategy', 'auto')
                
                with redirect_stdout(output_buffer):
                    save_paths, experiment_results = train_models_with_progress(
                        data_path=str(training_data_path),  # Use the actual training dataset
                        task=task,
                        quick=quick,
                        imputer_mode=imputer_mode,
                        selected_models=all_selected_models,
                        custom_model_classes=custom_model_classes if use_custom_models else {},
                        target_column=target_col,  # Pass explicit target column
                        imbalance_strategy=imb_strategy,
                        imbalance_k_neighbors=imb_k_neighbors,
                        imbalance_sampling_strategy=imb_sampling,
                        existing_model_actions=existing_model_actions,
                    )
                
                # Get the output
                output = output_buffer.getvalue()
                
                # Display in expander
                with st.expander("📋 Ver detalles completos del entrenamiento", expanded=False):
                    st.code(output, language="text")
            
            # Clean up temporary training dataset
            try:
                if training_data_path.exists():
                    training_data_path.unlink()
            except Exception:
                pass  # Ignore cleanup errors
            
            # Update session state
            set_state("is_trained", True)
            set_state("last_train_task", task)
            set_state("last_train_models", list(save_paths.keys()))
            
            # Store training results for PDF report
            st.session_state.training_results = experiment_results
            
            # Store trained models references
            if 'trained_models' not in st.session_state:
                st.session_state.trained_models = {}
            
            st.success(f"""
            ✅ **Entrenamiento completado exitosamente**
            
            - {len(save_paths)} modelo(s) entrenado(s)
            - Validación cruzada estratificada completada
            - Curvas de aprendizaje generadas
            - Comparación estadística: {'No aplica (modelo único)' if len(experiment_results.get('cv_results', {})) <= 1 else 'Realizada'}
            - Modelos guardados en `models/`
            """)
            
            # Display saved models
            with st.expander("📁 Ver rutas de modelos guardados"):
                for name, path in save_paths.items():
                    st.code(f"{name}: {path}", language="text")
            
            # Save transformer alongside models and update metadata
            if transformer_path is not None and st.session_state.transformer is not None:
                st.markdown("---")
                st.info("💾 **Guardando transformer y actualizando metadata de modelos...**")
                
                try:
                    from pathlib import Path as PathlibPath
                    import json
                    
                    # Save transformer permanently for each model
                    for model_name, model_path in save_paths.items():
                        model_dir = PathlibPath(model_path).parent
                        transformer_save_path = model_dir / f"{model_name}_transformer.joblib"
                        
                        # Copy transformer to model directory
                        joblib.dump(st.session_state.transformer, str(transformer_save_path))
                        
                        # Update model metadata to include transformation info
                        metadata_path = model_dir / f"{model_name}_metadata.json"
                        
                        if metadata_path.exists():
                            with open(metadata_path, 'r', encoding='utf-8') as f:
                                metadata_dict = json.load(f)
                            
                            # Add transformation information
                            metadata_dict['transformation'] = {
                                'type': st.session_state.transformation_params['type'],
                                'n_components': st.session_state.transformation_params['n_components'],
                                'transformer_path': str(transformer_save_path),
                                'original_features': st.session_state.transformation_params['feature_names'],
                                'params': st.session_state.transformation_params
                            }
                            
                            with open(metadata_path, 'w', encoding='utf-8') as f:
                                json.dump(metadata_dict, f, indent=2)
                        
                        st.success(f"✅ Transformer guardado para {model_name}: `{transformer_save_path.name}`")
                    
                    st.success(f"""
                    ✅ **Transformers guardados exitosamente**
                    
                    - Tipo: {st.session_state.transformation_params['type'].upper()}
                    - Componentes: {st.session_state.transformation_params['n_components']}
                    - Variables originales transformadas: {len(st.session_state.transformation_params['feature_names'])}
                    - Metadata actualizado para todos los modelos
                    
                    **Los modelos aplicarán automáticamente esta transformación durante la predicción.**
                    """)
                    
                except Exception as e:
                    st.error(f"❌ Error guardando transformer: {e}")
                    import traceback
                    with st.expander("Ver detalles del error"):
                        st.code(traceback.format_exc())
                
                finally:
                    # Clean up temporary transformer
                    try:
                        if transformer_path.exists():
                            transformer_path.unlink()
                    except Exception:
                        pass
            
            # Display learning curves if available (INTERACTIVAS con Plotly)
            if hasattr(st.session_state, 'learning_curve_results') and st.session_state.learning_curve_results:
                st.markdown("---")
                st.subheader("📈 Curvas de Aprendizaje")
                
                with st.expander("ℹ️ ¿Cómo interpretar las curvas de aprendizaje?", expanded=False):
                    st.markdown("""
                    **Curvas de aprendizaje** muestran el rendimiento del modelo vs tamaño de entrenamiento:
                    
                    | Patrón | Diagnóstico | Solución |
                    |--------|-------------|----------|
                    | Train alto, Val bajo, Gap grande | **Overfitting** | Más datos, regularización, modelo más simple |
                    | Train bajo, Val bajo, Gap pequeño | **Underfitting** | Modelo más complejo, más features |
                    | Train≈Val, ambos altos, Gap pequeño | **Buen ajuste** | ✅ Modelo adecuado |
                    | Curvas no convergen | **No convergido** | Más datos o epochs |
                    """)
                
                lc_results = st.session_state.learning_curve_results
                
                # Import diagnosis function
                from src.training.learning_curves import plot_learning_curve, diagnose_learning_curve
                
                # Create tabs for each model
                if len(lc_results) > 0:
                    tabs = st.tabs([f"📊 {model}" for model in lc_results.keys()])
                    
                    for tab, (model_name, lc_res) in zip(tabs, lc_results.items()):
                        with tab:
                            # Generate interactive Plotly figure
                            try:
                                fig = plot_learning_curve(lc_res, title=f"Learning Curve: {model_name}")
                                st.plotly_chart(fig, use_container_width=True, key=f"lc_{model_name}")
                            except Exception as e:
                                st.warning(f"No se pudo generar gráfico interactivo: {e}")
                                # Fallback to static image
                                lc_paths = st.session_state.get('learning_curve_paths', {})
                                if model_name in lc_paths and Path(lc_paths[model_name]).exists():
                                    st.image(lc_paths[model_name], use_container_width=True)
                            
                            # Display metrics
                            col1, col2, col3 = st.columns(3)
                            
                            final_train = lc_res.train_scores_mean[-1]
                            final_val = lc_res.val_scores_mean[-1]
                            gap = abs(final_train - final_val)
                            
                            with col1:
                                st.metric("Score Final (Train)", f"{final_train:.4f}")
                            
                            with col2:
                                st.metric("Score Final (Val)", f"{final_val:.4f}")
                            
                            with col3:
                                delta_color = "normal" if gap < 0.05 else ("off" if gap < 0.10 else "inverse")
                                st.metric("Gap Train-Val", f"{gap:.4f}", delta=None)
                            
                            # Diagnosis de underfitting/overfitting
                            st.markdown("##### 🔍 Diagnóstico Automático")
                            diagnosis = diagnose_learning_curve(lc_res)
                            
                            # Show issues
                            if diagnosis['issues']:
                                for issue in diagnosis['issues']:
                                    if 'Good fit' in issue:
                                        st.success(f"✅ {issue}")
                                    elif 'underfitting' in issue.lower() or 'High bias' in issue:
                                        st.error(f"🔴 **{issue}**: El modelo es demasiado simple para capturar los patrones")
                                    elif 'overfitting' in issue.lower() or 'High variance' in issue:
                                        st.warning(f"⚠️ **{issue}**: El modelo memoriza datos en lugar de generalizar")
                                    elif 'not converged' in issue.lower():
                                        st.info(f"ℹ️ **{issue}**: El modelo podría mejorar con más datos")
                                    else:
                                        st.info(f"ℹ️ {issue}")
                            
                            # Show recommendations
                            if diagnosis['recommendations'] and 'performing well' not in ' '.join(diagnosis['recommendations']):
                                with st.expander("💡 Recomendaciones"):
                                    for rec in diagnosis['recommendations']:
                                        st.markdown(f"- {rec}")
                            
                            # Additional stats
                            with st.expander("📊 Estadísticas detalladas"):
                                st.markdown(f"""
                                - **Train mejorando**: {'Sí ✅' if diagnosis['train_improving'] else 'No ❌'}
                                - **Validación mejorando**: {'Sí ✅' if diagnosis['val_improving'] else 'No ❌'}
                                - **Convergido**: {'Sí ✅' if diagnosis['converged'] else 'No (posiblemente necesita más datos)'}
                                - **Score inicial (Train)**: {lc_res.train_scores_mean[0]:.4f}
                                - **Score inicial (Val)**: {lc_res.val_scores_mean[0]:.4f}
                                """)
            
            # 🎉 Success! Show balloons
            st.balloons()
            st.success("🎉 **¡Entrenamiento completado exitosamente!**")
            
            # Show statistical comparison if available
            st.markdown("---")
            st.subheader("📊 Comparación Estadística Rigurosa de Modelos")
            
            # Get statistical results
            stat_results = experiment_results.get('statistical_comparison', {})
            multiple_comparison = experiment_results.get('multiple_comparison', None)
            cv_results = experiment_results.get('cv_results', {})
            
            if len(cv_results) <= 1:
                st.info("ℹ️ Comparación estadística no aplicable: se entrenó un solo modelo en esta corrida.")
            elif stat_results:
                st.info("""
                **Pipeline de Análisis Estadístico Riguroso (Protocolo context.txt):**
                - 🧪 **Normalidad**: Criterio 2-de-3 (Shapiro-Wilk, D'Agostino-Pearson, Anderson-Darling)
                - 📊 **Test Global**: Friedman (no paramétrico) o ANOVA medidas repetidas (paramétrico)
                - 🔗 **Test Pareado**: Wilcoxon signed-rank (no param.) o t-test pareado (param.)
                - ⚖️ **Correcciones Múltiples**: Bonferroni, Holm-Bonferroni, Benjamini-Hochberg FDR
                - 📏 **Tamaño del Efecto**: Cohen's d + ΔAUROC
                """)
                
                # ============================================================
                # SECTION 1: DISTRIBUTION OF AUROC SCORES (Etapa 1 del protocolo)
                # ============================================================
                st.markdown("### 📈 1. Distribución de Scores AUROC por Modelo")
                st.caption(f"Validación cruzada estratificada repetida: {len(cv_results.get(list(cv_results.keys())[0], {}).get('all_scores', []))} estimaciones por modelo")
                
                # Create distribution visualization with Plotly
                import plotly.graph_objects as go
                from plotly.subplots import make_subplots
                import plotly.express as px
                
                if cv_results:
                    # Box plot + violin plot for distributions
                    fig_dist = go.Figure()
                    
                    model_names_sorted = sorted(cv_results.keys(), 
                                               key=lambda x: cv_results[x].get('mean_score', 0), 
                                               reverse=True)
                    
                    colors = px.colors.qualitative.Set2
                    
                    for idx, model_name in enumerate(model_names_sorted):
                        scores = cv_results[model_name].get('all_scores', cv_results[model_name].get('fold_scores', []))
                        color = colors[idx % len(colors)]
                        
                        # Add violin plot
                        fig_dist.add_trace(go.Violin(
                            y=scores,
                            name=model_name,
                            box_visible=True,
                            meanline_visible=True,
                            fillcolor=color,
                            opacity=0.6,
                            line_color=color,
                            points='all',
                            jitter=0.3,
                            pointpos=-0.2,
                        ))
                    
                    fig_dist.update_layout(
                        title="Distribución de AUROC por Modelo (Validación Cruzada)",
                        yaxis_title="AUROC Score",
                        xaxis_title="Modelo",
                        showlegend=False,
                        height=500,
                    )
                    
                    st.plotly_chart(fig_dist, use_container_width=True, key="dist_auroc")
                    
                    # Summary statistics table
                    st.markdown("#### 📋 Estadísticas Descriptivas")
                    desc_data = []
                    for model_name in model_names_sorted:
                        scores = cv_results[model_name].get('all_scores', cv_results[model_name].get('fold_scores', []))
                        desc_data.append({
                            "Modelo": model_name,
                            "N": len(scores),
                            "Media (μ)": f"{np.mean(scores):.4f}",
                            "Desv. Est. (σ)": f"{np.std(scores):.4f}",
                            "Mínimo": f"{np.min(scores):.4f}",
                            "Q1 (25%)": f"{np.percentile(scores, 25):.4f}",
                            "Mediana": f"{np.median(scores):.4f}",
                            "Q3 (75%)": f"{np.percentile(scores, 75):.4f}",
                            "Máximo": f"{np.max(scores):.4f}",
                            "IC 95% Inf": f"{np.percentile(scores, 2.5):.4f}",
                            "IC 95% Sup": f"{np.percentile(scores, 97.5):.4f}",
                        })
                    
                    desc_df = pd.DataFrame(desc_data)
                    st.dataframe(desc_df, use_container_width=True, hide_index=True)
                    
                    # Download descriptive statistics
                    csv_desc = desc_df.to_csv(index=False)
                    st.download_button(
                        "📥 Descargar Estadísticas Descriptivas (CSV)",
                        csv_desc,
                        "estadisticas_descriptivas_auroc.csv",
                        "text/csv",
                        key="download_desc_stats"
                    )
                
                # ============================================================
                # SECTION 2: NORMALITY TESTS (Etapa 2 del protocolo)
                # ============================================================
                st.markdown("### 🧪 2. Verificación de Normalidad (Criterio 2-de-3)")
                st.caption("Shapiro-Wilk, D'Agostino-Pearson y Anderson-Darling. Normalidad aceptada si ≥2 tests no rechazan H₀.")
                
                # Perform normality tests and show results
                from src.training.statistical_tests import test_normality_full
                
                normality_data = []
                normality_details = {}
                
                for model_name in model_names_sorted:
                    scores = np.array(cv_results[model_name].get('all_scores', cv_results[model_name].get('fold_scores', [])))
                    norm_result = test_normality_full(scores, alpha=0.05)
                    normality_details[model_name] = norm_result
                    
                    normality_data.append({
                        "Modelo": model_name,
                        "Shapiro-Wilk p": f"{norm_result.shapiro_wilk_pvalue:.4f}",
                        "Shapiro ✓": "✅" if norm_result.shapiro_wilk_normal else "❌",
                        "D'Agostino p": f"{norm_result.dagostino_pvalue:.4f}",
                        "D'Agostino ✓": "✅" if norm_result.dagostino_normal else "❌",
                        "Anderson Stat": f"{norm_result.anderson_statistic:.3f}",
                        "Anderson Crit (5%)": f"{norm_result.anderson_critical_5pct:.3f}",
                        "Anderson ✓": "✅" if norm_result.anderson_normal else "❌",
                        "Tests Pasados": f"{norm_result.normal_tests_passed}/3",
                        "Decisión": "✅ Normal" if norm_result.is_normal else "❌ No Normal",
                    })
                
                norm_df = pd.DataFrame(normality_data)
                st.dataframe(norm_df, use_container_width=True, hide_index=True)
                
                # Summary of normality
                all_normal = all(normality_details[m].is_normal for m in normality_details)
                n_normal = sum(1 for m in normality_details if normality_details[m].is_normal)
                
                if all_normal:
                    st.success(f"✅ Todas las distribuciones ({n_normal}/{len(normality_details)}) cumplen supuesto de normalidad → Se usará **test t pareado** y **ANOVA de medidas repetidas**")
                else:
                    st.warning(f"⚠️ Solo {n_normal}/{len(normality_details)} distribuciones cumplen normalidad → Se usará **Wilcoxon signed-rank** y **test de Friedman**")
                
                # Q-Q plots in expander
                with st.expander("📊 Ver Gráficos Q-Q de Normalidad"):
                    from scipy import stats as scipy_stats
                    
                    n_models = len(model_names_sorted)
                    cols_qq = st.columns(min(3, n_models))
                    
                    for idx, model_name in enumerate(model_names_sorted):
                        scores = np.array(cv_results[model_name].get('all_scores', cv_results[model_name].get('fold_scores', [])))
                        
                        # Create Q-Q plot
                        fig_qq = go.Figure()
                        
                        # Calculate theoretical quantiles
                        sorted_scores = np.sort(scores)
                        theoretical_quantiles = scipy_stats.norm.ppf(np.linspace(0.01, 0.99, len(scores)))
                        
                        # Standardize scores
                        z_scores = (sorted_scores - np.mean(scores)) / np.std(scores)
                        
                        fig_qq.add_trace(go.Scatter(
                            x=theoretical_quantiles,
                            y=z_scores,
                            mode='markers',
                            name='Datos',
                            marker=dict(color=colors[idx % len(colors)])
                        ))
                        
                        # Reference line
                        min_val = min(theoretical_quantiles.min(), z_scores.min())
                        max_val = max(theoretical_quantiles.max(), z_scores.max())
                        fig_qq.add_trace(go.Scatter(
                            x=[min_val, max_val],
                            y=[min_val, max_val],
                            mode='lines',
                            name='Referencia',
                            line=dict(color='red', dash='dash')
                        ))
                        
                        norm_status = "✅" if normality_details[model_name].is_normal else "❌"
                        fig_qq.update_layout(
                            title=f"{model_name} {norm_status}",
                            xaxis_title="Cuantiles Teóricos",
                            yaxis_title="Cuantiles Observados",
                            showlegend=False,
                            height=300,
                        )
                        
                        with cols_qq[idx % len(cols_qq)]:
                            st.plotly_chart(fig_qq, use_container_width=True, key=f"qq_{model_name}")
                
                # Download normality tests
                csv_norm = norm_df.to_csv(index=False)
                st.download_button(
                    "📥 Descargar Tests de Normalidad (CSV)",
                    csv_norm,
                    "tests_normalidad.csv",
                    "text/csv",
                    key="download_norm_tests"
                )
                
                # ============================================================
                # SECTION 3: GLOBAL TEST (Friedman or ANOVA) + MODEL RANKING
                # ============================================================
                if multiple_comparison:
                    st.markdown("### 🌐 3. Test Global de Comparación Múltiple")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        test_name_short = multiple_comparison.global_test_name.split()[0] if not pd.isna(multiple_comparison.global_test_statistic) else "N/A"
                        st.metric("Test Usado", test_name_short)
                    with col2:
                        stat_val = f"{multiple_comparison.global_test_statistic:.3f}" if not pd.isna(multiple_comparison.global_test_statistic) else "N/A"
                        st.metric("Estadístico", stat_val)
                    with col3:
                        p_val = f"{multiple_comparison.global_p_value:.4f}" if not pd.isna(multiple_comparison.global_p_value) else "N/A"
                        st.metric("P-value Global", p_val)
                    with col4:
                        if pd.isna(multiple_comparison.global_test_statistic):
                            st.info("ℹ️ Requiere ≥3 modelos")
                        elif multiple_comparison.global_significant:
                            st.success("✅ SIGNIFICATIVO")
                        else:
                            st.warning("❌ No significativo")
                    
                    if not pd.isna(multiple_comparison.global_test_statistic):
                        if multiple_comparison.global_significant:
                            st.success(f"""
                            ✅ **El test {multiple_comparison.global_test_name} rechaza H₀ (p={multiple_comparison.global_p_value:.4f})**
                            
                            Esto indica que al menos un modelo tiene rendimiento significativamente diferente del resto.
                            Procediendo con análisis post-hoc para identificar qué modelos difieren.
                            """)
                        else:
                            st.info(f"""
                            ℹ️ **El test {multiple_comparison.global_test_name} no rechaza H₀ (p={multiple_comparison.global_p_value:.4f})**
                            
                            No hay evidencia estadística de diferencias significativas entre los modelos a nivel global.
                            Las comparaciones pareadas se muestran de forma descriptiva.
                            """)
                    else:
                        st.info("ℹ️ El test global (Friedman/ANOVA) requiere al menos 3 modelos. Con 2 modelos, solo se realiza comparación pareada directa.")
                    
                    # Show multiple comparison correction info
                    n_comparisons = len(stat_results)
                    bonf_alpha = multiple_comparison.bonferroni_alpha if hasattr(multiple_comparison, 'bonferroni_alpha') else 0.05/max(n_comparisons, 1)
                    st.markdown(f"""
                    **Correcciones para {n_comparisons} comparaciones pareadas:**
                    - α Bonferroni = 0.05 / {n_comparisons} = **{bonf_alpha:.4f}**
                    - Holm-Bonferroni: Procedimiento secuencial (menos conservador)
                    - FDR (Benjamini-Hochberg): Control de tasa de falsos descubrimientos
                    """)
                    
                    # ==============================================
                    # MODEL RANKING TABLE (Clear Winner Identification)
                    # ==============================================
                    st.markdown("#### 🏆 Ranking de Modelos")
                    
                    # Calculate ranking based on mean AUROC
                    ranking_data = []
                    for model_name in cv_results.keys():
                        scores = cv_results[model_name].get('all_scores', cv_results[model_name].get('fold_scores', []))
                        ranking_data.append({
                            "Modelo": model_name,
                            "Media AUROC": np.mean(scores),
                            "Desv. Est.": np.std(scores),
                            "IC 95% Inf": np.percentile(scores, 2.5),
                            "IC 95% Sup": np.percentile(scores, 97.5),
                        })
                    
                    # Sort by mean AUROC descending
                    ranking_df = pd.DataFrame(ranking_data).sort_values("Media AUROC", ascending=False).reset_index(drop=True)
                    ranking_df.insert(0, "Posición", [f"🥇 {i+1}" if i == 0 else f"🥈 {i+1}" if i == 1 else f"🥉 {i+1}" if i == 2 else f"   {i+1}" for i in range(len(ranking_df))])
                    
                    # Get best model info
                    best_model = ranking_df.iloc[0]["Modelo"]
                    best_auroc = ranking_df.iloc[0]["Media AUROC"]
                    best_ci_low = ranking_df.iloc[0]["IC 95% Inf"]
                    best_ci_high = ranking_df.iloc[0]["IC 95% Sup"]
                    
                    # Format numeric columns
                    display_df = ranking_df.copy()
                    display_df["Media AUROC"] = display_df["Media AUROC"].apply(lambda x: f"{x:.4f}")
                    display_df["Desv. Est."] = display_df["Desv. Est."].apply(lambda x: f"{x:.4f}")
                    display_df["IC 95% Inf"] = display_df["IC 95% Inf"].apply(lambda x: f"{x:.4f}")
                    display_df["IC 95% Sup"] = display_df["IC 95% Sup"].apply(lambda x: f"{x:.4f}")
                    
                    st.dataframe(display_df, use_container_width=True, hide_index=True)
                    
                    # Determine if best model is statistically superior
                    best_is_stat_superior = False
                    inferior_models = []
                    
                    for (m1, m2), res in stat_results.items():
                        if m1 == best_model or m2 == best_model:
                            other_model = m2 if m1 == best_model else m1
                            # Check if best model wins this comparison with Bonferroni correction
                            is_sig_bonf = res.p_value < bonf_alpha
                            
                            if is_sig_bonf:
                                # Check direction of effect
                                if m1 == best_model and res.delta_auroc > 0:
                                    inferior_models.append(other_model)
                                    best_is_stat_superior = True
                                elif m2 == best_model and res.delta_auroc < 0:
                                    inferior_models.append(other_model)
                                    best_is_stat_superior = True
                    
                    # Show winner announcement
                    st.markdown("---")
                    st.markdown("#### 🏅 Conclusión del Análisis Estadístico")
                    
                    if best_is_stat_superior and inferior_models:
                        st.success(f"""
                        ## 🏆 **Mejor Modelo: {best_model}**
                        
                        **Rendimiento:** AUROC = **{best_auroc:.4f}** (IC 95%: [{best_ci_low:.4f}, {best_ci_high:.4f}])
                        
                        ✅ **Estadísticamente superior** (con corrección Bonferroni, α={bonf_alpha:.4f}) a:
                        - {', '.join(inferior_models)}
                        
                        📋 **Interpretación:** El modelo **{best_model}** no solo tiene la mayor media de AUROC, 
                        sino que esta diferencia es estadísticamente significativa tras aplicar la corrección 
                        por comparaciones múltiples, lo que proporciona **evidencia robusta** de su superioridad.
                        """)
                    elif len(cv_results) == 2:
                        # Only 2 models - direct comparison
                        other_model = [m for m in cv_results.keys() if m != best_model][0]
                        comparison_key = (best_model, other_model) if (best_model, other_model) in stat_results else (other_model, best_model)
                        res = stat_results.get(comparison_key)
                        
                        if res and res.significant:
                            st.success(f"""
                            ## 🏆 **Mejor Modelo: {best_model}**
                            
                            **Rendimiento:** AUROC = **{best_auroc:.4f}** (IC 95%: [{best_ci_low:.4f}, {best_ci_high:.4f}])
                            
                            ✅ **Estadísticamente superior** a {other_model} (p={res.p_value:.4f}, {res.test_used})
                            
                            **Tamaño del efecto:** Cohen's d = {res.effect_size:.3f} ({res.effect_size_interpretation})
                            """)
                        else:
                            st.warning(f"""
                            ## 🏆 **Modelo con Mayor AUROC: {best_model}**
                            
                            **Rendimiento:** AUROC = **{best_auroc:.4f}** (IC 95%: [{best_ci_low:.4f}, {best_ci_high:.4f}])
                            
                            ⚠️ **No hay diferencia estadísticamente significativa** con {other_model}
                            
                            **P-value:** {res.p_value:.4f if res else 'N/A'} (umbral: 0.05)
                            
                            📋 **Interpretación:** Aunque **{best_model}** tiene la mayor media de AUROC, 
                            no hay evidencia estadística suficiente para afirmar que es superior. 
                            Ambos modelos pueden considerarse equivalentes en rendimiento.
                            """)
                    else:
                        # Multiple models but best is not statistically superior
                        st.warning(f"""
                        ## 🏆 **Modelo con Mayor AUROC: {best_model}**
                        
                        **Rendimiento:** AUROC = **{best_auroc:.4f}** (IC 95%: [{best_ci_low:.4f}, {best_ci_high:.4f}])
                        
                        ⚠️ **No hay evidencia de superioridad estadística significativa** (con corrección Bonferroni)
                        
                        📋 **Interpretación:** Aunque **{best_model}** tiene la mayor media de AUROC, 
                        las diferencias con otros modelos no alcanzan significación estadística tras 
                        corregir por comparaciones múltiples. Se recomienda considerar:
                        - Complejidad/interpretabilidad del modelo
                        - Tiempo de entrenamiento e inferencia
                        - Otros criterios clínicos o de negocio
                        """)
                    
                    # Download ranking CSV
                    csv_ranking = ranking_df.to_csv(index=False)
                    st.download_button(
                        "📥 Descargar Ranking de Modelos (CSV)",
                        csv_ranking,
                        "ranking_modelos.csv",
                        "text/csv",
                        key="download_ranking"
                    )
                
                # ============================================================
                # SECTION 4: PAIRWISE COMPARISONS (Post-hoc)
                # ============================================================
                st.markdown("### 🔄 4. Comparaciones por Pares (Post-hoc)")
                
                # Display comparison matrix first
                from src.data_load import get_latest_plot
                matrix_plot = get_latest_plot(PLOTS_TRAINING_DIR, "comparison_matrix")
                
                if matrix_plot and matrix_plot.exists():
                    st.markdown("#### 🔲 Matriz de Comparaciones")
                    if matrix_plot.suffix == '.png':
                        st.image(str(matrix_plot), use_container_width=True)
                    elif matrix_plot.suffix == '.html':
                        with open(matrix_plot, 'r', encoding='utf-8') as f:
                            st.components.v1.html(f.read(), height=600, scrolling=True)
                
                st.markdown("#### 📋 Tabla de Resultados")
                
                # Create dataframe with results including corrections
                comparison_data = []
                for (m1, m2), res in stat_results.items():
                    mean_diff = res.model1_mean - res.model2_mean
                    
                    # Get corrected p-values if available
                    if multiple_comparison:
                        holm_p = multiple_comparison.holm_corrected_pvalues.get((m1, m2), res.p_value)
                        fdr_p = multiple_comparison.fdr_corrected_pvalues.get((m1, m2), res.p_value)
                        sig_bonf = res.p_value < multiple_comparison.bonferroni_alpha
                    else:
                        holm_p = res.p_value
                        fdr_p = res.p_value
                        sig_bonf = res.p_value < 0.05
                    
                    comparison_data.append({
                        "Modelo 1": m1,
                        "Modelo 2": m2,
                        "ΔAUROC": f"{res.delta_auroc:+.4f}" if hasattr(res, 'delta_auroc') else f"{mean_diff:+.4f}",
                        "Test Usado": res.test_used.replace(" test", ""),
                        "p-value": f"{res.p_value:.4f}",
                        "p Holm": f"{holm_p:.4f}",
                        "p FDR": f"{fdr_p:.4f}",
                        "Sig. (sin corr.)": "✅" if res.significant else "❌",
                        "Sig. (Bonferroni)": "✅" if sig_bonf else "❌",
                        "Cohen's d": f"{res.effect_size:.3f}",
                        "Efecto": res.effect_size_interpretation[:3],
                    })
                
                if comparison_data:
                    comparison_df = pd.DataFrame(comparison_data)
                    st.dataframe(comparison_df, use_container_width=True, hide_index=True)
                    
                    # Legend for effect sizes
                    st.caption("""
                    **Interpretación de tamaños de efecto (Cohen's d):** 
                    Neg = Negligible (|d|<0.2) | Sma = Small (0.2-0.5) | Med = Medium (0.5-0.8) | Lar = Large (>0.8)
                    """)
                    
                    # Summary of significant differences
                    sig_with_correction = sum(1 for d in comparison_data if d["Sig. (Bonferroni)"] == "✅")
                    sig_without = sum(1 for d in comparison_data if d["Sig. (sin corr.)"] == "✅")
                    
                    st.markdown(f"""
                    **📈 Resumen:**
                    - Diferencias significativas (sin corrección, p<0.05): **{sig_without}** de {len(comparison_data)}
                    - Diferencias significativas (con Bonferroni): **{sig_with_correction}** de {len(comparison_data)}
                    """)
                    
                    # Download pairwise comparisons
                    csv_pairwise = comparison_df.to_csv(index=False)
                    st.download_button(
                        "📥 Descargar Comparaciones Pareadas (CSV)",
                        csv_pairwise,
                        "comparaciones_pareadas.csv",
                        "text/csv",
                        key="download_pairwise"
                    )
                    
                    # Show individual comparison plots
                    with st.expander("📈 Ver gráficos de comparación individual"):
                        for (m1, m2), res in stat_results.items():
                            comp_plot = get_latest_plot(PLOTS_TRAINING_DIR, f"comparison_{m1}_vs_{m2}")
                            if comp_plot and comp_plot.exists():
                                st.markdown(f"**{m1} vs {m2}**")
                                if comp_plot.suffix == '.png':
                                    st.image(str(comp_plot), use_container_width=True)
                                elif comp_plot.suffix == '.html':
                                    with open(comp_plot, 'r', encoding='utf-8') as f:
                                        st.components.v1.html(f.read(), height=500, scrolling=True)
                
                # ============================================================
                # SECTION 5: COMPLETE REPORT DOWNLOAD
                # ============================================================
                st.markdown("### 📥 5. Descargar Reporte Completo")
                
                # Ensure variables are defined for the report
                if 'bonf_alpha' not in locals():
                    bonf_alpha = 0.05 / max(len(stat_results), 1) if stat_results else 0.05
                if 'sig_without' not in locals():
                    sig_without = sum(1 for res in stat_results.values() if res.significant) if stat_results else 0
                if 'sig_with_correction' not in locals():
                    sig_with_correction = sum(1 for res in stat_results.values() if res.p_value < bonf_alpha) if stat_results else 0
                
                # Create comprehensive CSV with all raw scores
                raw_scores_data = []
                for model_name in cv_results.keys():
                    scores = cv_results[model_name].get('all_scores', cv_results[model_name].get('fold_scores', []))
                    for i, score in enumerate(scores):
                        raw_scores_data.append({
                            "Modelo": model_name,
                            "Iteración": i + 1,
                            "AUROC": score
                        })
                
                raw_scores_df = pd.DataFrame(raw_scores_data)
                csv_raw = raw_scores_df.to_csv(index=False)
                
                # Create summary report
                summary_data = {
                    "Parámetro": [
                        "Fecha del análisis",
                        "Total de modelos",
                        "Estimaciones por modelo",
                        "Test global usado",
                        "P-value global",
                        "Test global significativo",
                        "Corrección Bonferroni (α)",
                        "Comparaciones con diferencia significativa (sin corrección)",
                        "Comparaciones con diferencia significativa (Bonferroni)",
                        "Mejor modelo (media AUROC)",
                    ],
                    "Valor": [
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        len(cv_results),
                        len(cv_results[list(cv_results.keys())[0]].get('all_scores', [])),
                        multiple_comparison.global_test_name if multiple_comparison and not pd.isna(multiple_comparison.global_test_statistic) else "N/A (< 3 modelos)",
                        f"{multiple_comparison.global_p_value:.6f}" if multiple_comparison and not pd.isna(multiple_comparison.global_p_value) else "N/A",
                        "Sí" if multiple_comparison and multiple_comparison.global_significant else "No",
                        f"{bonf_alpha:.6f}",
                        sig_without,
                        sig_with_correction,
                        max(cv_results.keys(), key=lambda x: cv_results[x].get('mean_score', 0)),
                    ]
                }
                summary_df = pd.DataFrame(summary_data)
                csv_summary = summary_df.to_csv(index=False)
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.download_button(
                        "📥 Scores AUROC (Datos Crudos)",
                        csv_raw,
                        "auroc_scores_raw.csv",
                        "text/csv",
                        key="download_raw_scores"
                    )
                
                with col2:
                    st.download_button(
                        "📥 Resumen del Análisis",
                        csv_summary,
                        "resumen_analisis_estadistico.csv",
                        "text/csv",
                        key="download_summary"
                    )
                
                with col3:
                    # Create complete report as JSON
                    import json
                    
                    # Helper function to convert numpy types to Python native types
                    def to_native(obj):
                        if isinstance(obj, (np.bool_, bool)):
                            return bool(obj)
                        elif isinstance(obj, (np.integer, int)):
                            return int(obj)
                        elif isinstance(obj, (np.floating, float)):
                            return float(obj)
                        elif isinstance(obj, np.ndarray):
                            return obj.tolist()
                        return obj
                    
                    complete_report = {
                        "metadata": {
                            "fecha": datetime.now().isoformat(),
                            "n_modelos": int(len(cv_results)),
                            "n_estimaciones": int(len(cv_results[list(cv_results.keys())[0]].get('all_scores', []))),
                        },
                        "estadisticas_descriptivas": {m: {
                            "mean": float(np.mean(cv_results[m].get('all_scores', []))),
                            "std": float(np.std(cv_results[m].get('all_scores', []))),
                            "min": float(np.min(cv_results[m].get('all_scores', []))),
                            "max": float(np.max(cv_results[m].get('all_scores', []))),
                            "median": float(np.median(cv_results[m].get('all_scores', []))),
                        } for m in cv_results.keys()},
                        "test_global": {
                            "nombre": str(multiple_comparison.global_test_name) if multiple_comparison else None,
                            "estadistico": float(multiple_comparison.global_test_statistic) if multiple_comparison and not pd.isna(multiple_comparison.global_test_statistic) else None,
                            "p_value": float(multiple_comparison.global_p_value) if multiple_comparison and not pd.isna(multiple_comparison.global_p_value) else None,
                            "significativo": bool(multiple_comparison.global_significant) if multiple_comparison else None,
                        },
                        "comparaciones_pareadas": [{
                            "modelo_1": str(m1),
                            "modelo_2": str(m2),
                            "delta_auroc": float(res.delta_auroc) if hasattr(res, 'delta_auroc') else None,
                            "test_usado": str(res.test_used),
                            "p_value": float(res.p_value),
                            "p_holm": float(multiple_comparison.holm_corrected_pvalues.get((m1, m2), res.p_value)) if multiple_comparison else float(res.p_value),
                            "p_fdr": float(multiple_comparison.fdr_corrected_pvalues.get((m1, m2), res.p_value)) if multiple_comparison else float(res.p_value),
                            "significativo_sin_correccion": bool(res.significant),
                            "significativo_bonferroni": bool(res.p_value < bonf_alpha) if 'bonf_alpha' in dir() else bool(res.significant),
                            "cohens_d": float(res.effect_size),
                            "interpretacion_efecto": str(res.effect_size_interpretation),
                        } for (m1, m2), res in stat_results.items()],
                    }
                    
                    st.download_button(
                        "📥 Reporte Completo (JSON)",
                        json.dumps(complete_report, indent=2, ensure_ascii=False),
                        "reporte_estadistico_completo.json",
                        "application/json",
                        key="download_json_report"
                    )
                
                st.success("✅ **Protocolo de comparación estadística completado**. Todos los resultados están disponibles para descarga.")
                
            elif len(selected_models) == 1:
                st.info("ℹ️ Selecciona al menos 2 modelos para ver la comparación estadística.")
            else:
                st.warning("⚠️ No se encontraron resultados de comparación estadística.")
        
        except FileNotFoundError as e:
            st.error(f"❌ Dataset file not found: {e}")
        except Exception as e:
            st.error(f"❌ Error during training: {e}")
            st.exception(e)
        finally:
            # Reset training flag
            st.session_state.is_training = False

st.markdown("---")

# Display learning curves from previous training if available (INTERACTIVO)
if not get_state("is_trained") and hasattr(st.session_state, 'learning_curve_results'):
    lc_results = st.session_state.learning_curve_results
    if lc_results:
        st.subheader("📈 Curvas de Aprendizaje (del último entrenamiento)")
        
        from src.training.learning_curves import plot_learning_curve, diagnose_learning_curve
        
        tabs = st.tabs([f"📊 {model}" for model in lc_results.keys()])
        
        for tab, (model_name, lc_res) in zip(tabs, lc_results.items()):
            with tab:
                try:
                    fig = plot_learning_curve(lc_res, title=f"Learning Curve: {model_name}")
                    st.plotly_chart(fig, use_container_width=True, key=f"lc_prev_{model_name}")
                    
                    # Quick diagnosis
                    diagnosis = diagnose_learning_curve(lc_res)
                    for issue in diagnosis['issues']:
                        if 'Good fit' in issue:
                            st.success(f"✅ {issue}")
                        elif 'underfitting' in issue.lower():
                            st.error(f"🔴 {issue}")
                        elif 'overfitting' in issue.lower():
                            st.warning(f"⚠️ {issue}")
                except Exception:
                    # Fallback to static
                    lc_paths = st.session_state.get('learning_curve_paths', {})
                    if model_name in lc_paths and Path(lc_paths[model_name]).exists():
                        st.image(lc_paths[model_name], use_container_width=True)
        
        st.markdown("---")

# Display saved models section
st.subheader("Saved Models")

last_task = get_state("last_train_task")
if last_task and last_task != task:
    st.info(f"ℹ️ Last training was for task: {last_task}")

display_model_list(task)

st.markdown("---")
st.subheader("📚 Comparación Histórica Multi-Modelo")

historical_models = list_saved_model_versions(task, max_per_combination=25)
if not historical_models and task not in ["mortality", "arrhythmia"]:
    historical_models = list_saved_model_versions("mortality", max_per_combination=25)

if historical_models:
    selected_historical_models = st.multiselect(
        "Selecciona modelos guardados (múltiples corridas / distintos resampling)",
        options=list(historical_models.keys()),
        help="Puedes comparar modelos de corridas diferentes y con estrategias de resampling distintas.",
    )

    historical_bootstrap_iterations = st.slider(
        "Bootstrap iteraciones para comparación histórica",
        min_value=50,
        max_value=1000,
        value=200,
        step=50,
    )

    if st.button("📊 Comparar Modelos Guardados", type="secondary", use_container_width=True):
        if not selected_historical_models:
            st.warning("⚠️ Selecciona al menos un modelo guardado para comparar.")
        else:
            testset_path_hist = get_latest_testset(task, TESTSETS_DIR)
            if not testset_path_hist:
                testset_path_hist = TESTSETS_DIR / f"testset_{task}.parquet"

            if not testset_path_hist or not Path(testset_path_hist).exists():
                st.error("❌ No se encontró test set para la comparación histórica.")
            else:
                test_df_hist = pd.read_parquet(testset_path_hist)

                candidate_targets = [target_col, CONFIG.target_column, CONFIG.arrhythmia_column, "mortality", "mortality_inhospital", "exitus", "arrhythmia"]
                target_hist = next((c for c in candidate_targets if c and c in test_df_hist.columns), None)
                if target_hist is None:
                    heuristic_cols = [c for c in test_df_hist.columns if any(x in c.lower() for x in ["mortal", "exitus", "arrhythm", "target"])]
                    target_hist = heuristic_cols[0] if heuristic_cols else None

                if target_hist is None:
                    st.error("❌ No se pudo identificar la columna objetivo en el test set.")
                else:
                    X_hist = test_df_hist.drop(columns=[target_hist])
                    y_hist = test_df_hist[target_hist]

                    rows = []
                    historical_scores = {}
                    model_errors = []

                    for model_label in selected_historical_models:
                        model_path = Path(historical_models[model_label])
                        try:
                            model_obj = joblib.load(model_path)

                            if hasattr(model_obj, "predict_proba"):
                                y_prob_hist = model_obj.predict_proba(X_hist)[:, 1]
                            elif hasattr(model_obj, "decision_function"):
                                decision = model_obj.decision_function(X_hist)
                                y_prob_hist = 1.0 / (1.0 + np.exp(-decision))
                            else:
                                raise ValueError("El modelo no soporta predict_proba ni decision_function")

                            model_metrics = compute_classification_metrics(y_hist.values, y_prob_hist)
                            boot_result = bootstrap_evaluation(
                                model=model_obj,
                                X_test=X_hist,
                                y_test=y_hist,
                                n_iterations=historical_bootstrap_iterations,
                                random_state=RANDOM_SEED,
                            )

                            if "auroc" in boot_result.metrics and len(boot_result.metrics["auroc"]) > 1:
                                historical_scores[model_label] = boot_result.metrics["auroc"]

                            rows.append({
                                "model": model_label,
                                "path": str(model_path),
                                "auroc": model_metrics.get("auroc", np.nan),
                                "auprc": model_metrics.get("auprc", np.nan),
                                "accuracy": model_metrics.get("accuracy", np.nan),
                                "precision": model_metrics.get("precision", np.nan),
                                "recall": model_metrics.get("recall", np.nan),
                                "f1": model_metrics.get("f1", np.nan),
                                "brier": model_metrics.get("brier", np.nan),
                            })
                        except Exception as e:
                            model_errors.append(f"{model_label}: {e}")

                    if rows:
                        comp_df = pd.DataFrame(rows).sort_values("auroc", ascending=False)
                        st.markdown("#### 📊 Métricas comparativas (mismo test set)")
                        st.dataframe(comp_df, width='stretch', hide_index=True)

                        if len(historical_scores) >= 2:
                            st.markdown("#### 🧪 Comparación estadística (Bootstrap AUROC)")
                            comparison_result = compare_multiple_models(historical_scores)

                            global_col1, global_col2, global_col3 = st.columns(3)
                            with global_col1:
                                st.metric("Test global", comparison_result.global_test_name)
                            with global_col2:
                                gp = comparison_result.global_p_value
                                st.metric("P-value global", f"{gp:.6f}" if not np.isnan(gp) else "N/A")
                            with global_col3:
                                st.metric("Significativo", "Sí" if comparison_result.global_significant else "No")

                            pairwise_rows = []
                            for (m1, m2), res in comparison_result.pairwise_results.items():
                                pairwise_rows.append({
                                    "model_1": m1,
                                    "model_2": m2,
                                    "test": res.test_used,
                                    "p_value": res.p_value,
                                    "delta_auroc": res.delta_auroc,
                                    "effect_size": res.effect_size,
                                    "effect_interpretation": res.effect_size_interpretation,
                                    "significant": res.significant,
                                })

                            if pairwise_rows:
                                st.dataframe(pd.DataFrame(pairwise_rows), width='stretch', hide_index=True)
                        else:
                            st.info("ℹ️ Se requiere al menos 2 modelos válidos para comparación estadística entre pares.")

                        st.session_state.historical_comparison_training = {
                            "metrics": rows,
                            "n_models": len(rows),
                            "n_valid_for_stats": len(historical_scores),
                        }

                    if model_errors:
                        with st.expander("⚠️ Modelos omitidos por error"):
                            for err in model_errors:
                                st.warning(err)
else:
    st.info("ℹ️ No hay modelos históricos disponibles para comparación en este task.")

# Training history/log
with st.expander("ℹ️ Training Notes"):
    st.markdown("""
    ### ⚙️ Configuración del Entrenamiento
    
    **Quick Mode:**
    - ✅ Búsqueda simplificada de hiperparámetros
    - ✅ Menos splits en CV (3×3 = 9 corridas en vez de 10×10 = 100)
    - ✅ Iteración rápida para depuración
    - ⚠️ Recomendado solo para exploración inicial
    
    **Estrategias de Imputación:**
    - **Iterative**: IterativeImputer de sklearn (MICE - Multiple Imputation by Chained Equations)
    - **KNN**: K-Nearest Neighbors imputation (busca valores similares)
    - **Simple**: Imputación básica (media/mediana/moda)
    
    **Tipos de Modelos Disponibles:**
    - 🌳 Decision Trees, Random Forest
    - 🚀 XGBoost (Gradient Boosting)
    - 📈 Logistic Regression
    - 🎯 Support Vector Machine (SVM)
    - 👥 K-Nearest Neighbors (KNN)
    - 📊 Naive Bayes
    
    ### 📋 Pipeline de Experimentación
    
    El **Pipeline Riguroso** implementa el proceso científico completo:
    
    1. **Validación Cruzada Estratificada Repetida**: Se entrena y evalúa cada modelo
       múltiples veces (≥30 corridas) para obtener estimaciones robustas de μ y σ.
       
    2. **Curvas de Aprendizaje**: Diagnostican sobreajuste/subajuste y la necesidad
       de más datos.
       
    3. **Comparación Estadística**: Determina si las diferencias entre modelos son
       estadísticamente significativas usando:
       - Prueba de normalidad (Shapiro-Wilk)
       - Test paramétrico (t-Student) si los datos son normales
       - Test no paramétrico (Mann-Whitney) si no lo son
       
    4. **Evaluación Final en Test Set**: Una vez seleccionado el mejor modelo:
       - Bootstrap (1000 iteraciones con reemplazo)
       - Jackknife (leave-one-out)
       - Intervalos de confianza al 95%
    
    📚 Ver documentación completa en `Tools/docs/EXPERIMENT_PIPELINE.md`
    """)

# Exportación PDF
st.markdown("---")
st.subheader("📄 Exportar Reporte de Entrenamiento")

if st.session_state.get('training_results'):
    
    def generate_training_report():
        """Generate training PDF report."""
        from pathlib import Path
        output_path = Path("reports") / "training_report.pdf"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Get training results
        training_res = st.session_state.training_results
        
        # Extract models metadata from cv_results
        models_metadata = {}
        cv_results = training_res.get('cv_results', {})
        
        for model_name in cv_results.keys():
            # Create basic metadata from CV results
            from src.models.metadata import ModelMetadata, PerformanceMetrics, TrainingMetadata, DatasetMetadata
            
            cv_data = cv_results[model_name]
            
            # Performance metrics
            perf_metrics = PerformanceMetrics(
                mean_score=cv_data['mean_score'],
                std_score=cv_data['std_score'],
                min_score=cv_data.get('min_score', 0.0),
                max_score=cv_data.get('max_score', 1.0),
                all_scores=cv_data.get('all_scores', [])
            )
            
            # Basic training metadata
            train_metadata = TrainingMetadata(
                training_date=datetime.now().isoformat(),
                training_duration_seconds=0.0,
                cv_strategy="RepeatedStratifiedKFold",
                n_cv_folds=cv_data.get('n_splits', 10),
                n_cv_repeats=cv_data.get('n_repeats', 10),
                total_cv_runs=cv_data.get('n_runs', 100),
                scoring_metric=cv_data.get('scoring', 'roc_auc'),
                preprocessing_config={},
                random_seed=42
            )
            
            # Create metadata
            models_metadata[model_name] = ModelMetadata(
                model_name=model_name,
                model_type=model_name,
                task="classification",
                model_file_path="",
                dataset=DatasetMetadata(
                    train_set_path="",
                    test_set_path="",
                    train_samples=0,
                    test_samples=0,
                    n_features=0,
                    target_column="",
                    class_distribution_train={},
                    class_distribution_test={},
                    feature_names=[]
                ),
                training=train_metadata,
                hyperparameters={},
                performance=perf_metrics
            )
        
        return generate_training_pdf(
            training_results=training_res,
            models_metadata=models_metadata,
            output_path=output_path
        )
    
    pdf_export_section(
        generate_training_report,
        section_title="Reporte de Entrenamiento",
        default_filename="training_report.pdf",
        key_prefix="training_report"
    )
else:
    st.info("ℹ️ Entrena modelos primero para generar el reporte PDF")
