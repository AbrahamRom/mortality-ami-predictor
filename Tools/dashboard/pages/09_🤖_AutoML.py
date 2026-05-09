"""AutoML Page - Automated Machine Learning.

This page provides a comprehensive AutoML interface for:
- Configuring AutoML search parameters
- Running automated model selection
- Viewing leaderboard of evaluated models
- Exporting best models
- Getting intelligent suggestions
"""

import streamlit as st
import sys
import threading
import queue
import re
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time

# Add src to path
root_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(root_path))

from app import initialize_state, get_state, set_state
from src.config import CONFIG

# Page config
st.set_page_config(
    page_title="AutoML",
    page_icon="🤖",
    layout="wide"
)

# Initialize state
initialize_state()

# Title
st.title("🤖 AutoML - Aprendizaje Automático")
st.markdown("""
**AutoML** busca automáticamente la mejor arquitectura de modelo y hiperparámetros
para tu dataset. Solo necesitas configurar el tiempo de búsqueda y el sistema
explorará múltiples algoritmos y configuraciones.
""")

st.markdown("---")

# Check data availability
cleaned_data = st.session_state.get('cleaned_data')
raw_data = st.session_state.get('raw_data')

if cleaned_data is not None:
    df = cleaned_data
    st.success("✅ Usando datos limpios del proceso de limpieza")
elif raw_data is not None:
    df = raw_data
    st.warning("⚠️ Usando datos crudos (se recomienda limpiar primero)")
else:
    st.warning("⚠️ No hay datos cargados. Por favor, carga un dataset en **🧹 Data Cleaning and EDA** primero.")
    st.stop()

# Get target column - prefer target_column_name (actual column name) over target_column (task name)
target_col = st.session_state.get('target_column_name', None)
if not target_col:
    target_col = st.session_state.get('target_column', 'mortality_inhospital')

if target_col not in df.columns:
    # Try to find it
    possible_targets = [c for c in df.columns if 'mortal' in c.lower() or 'exitus' in c.lower() or 'target' in c.lower()]
    if possible_targets:
        target_col = possible_targets[0]
    else:
        st.error(f"❌ No se encontró la columna target '{target_col}'. Columnas disponibles: {list(df.columns[:10])}")
        st.stop()
        st.stop()

# Import AutoML components
try:
    from src.automl import (
        AutoMLClassifier, FLAMLClassifier, AutoMLConfig, AutoMLPreset,
        is_autosklearn_available, is_flaml_available,
        AutoMLSuggestions, analyze_dataset, get_suggestions,
        export_best_model, create_automl_report,
        # New NAS imports
        NASClassifier, NASConfig, is_autokeras_available, is_tensorflow_available,
        # New estimator lists
        FLAML_ESTIMATORS, FLAML_ESTIMATOR_DESCRIPTIONS,
        FLAML_CLASSIFICATION_ESTIMATORS,
    )
    from src.automl.export import create_automl_report
    from src.automl.flaml_integration import FLAML_PRESETS
    
    AUTOML_AVAILABLE = is_autosklearn_available() or is_flaml_available()
    AUTOSKLEARN_AVAILABLE = is_autosklearn_available()
    FLAML_AVAILABLE = is_flaml_available()
    # Default backend - prefer auto-sklearn if available
    DEFAULT_BACKEND = "auto-sklearn" if is_autosklearn_available() else ("flaml" if is_flaml_available() else None)
    NAS_AVAILABLE = is_autokeras_available() and is_tensorflow_available()
except ImportError as e:
    AUTOML_AVAILABLE = False
    AUTOSKLEARN_AVAILABLE = False
    FLAML_AVAILABLE = False
    DEFAULT_BACKEND = None
    NAS_AVAILABLE = False
    st.error(f"❌ Módulo AutoML no disponible: {e}")
    st.info("Instala las dependencias con: `pip install flaml[automl]` o `pip install auto-sklearn` (Linux)")
    st.stop()

# Display backend info
col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    backends_available = []
    if AUTOSKLEARN_AVAILABLE:
        backends_available.append("auto-sklearn")
    if FLAML_AVAILABLE:
        backends_available.append("FLAML")
    st.info(f"🔧 **Backends disponibles:** {', '.join(backends_available) if backends_available else 'Ninguno'}")
with col2:
    if AUTOSKLEARN_AVAILABLE:
        st.metric("auto-sklearn", "✅")
    else:
        st.metric("auto-sklearn", "❌")
with col3:
    if NAS_AVAILABLE:
        st.metric("NAS", "✅ AutoKeras")
    else:
        st.metric("NAS", "❌ No disponible")

# ============================================================================
# SIDEBAR - Configuration
# ============================================================================

st.sidebar.header("⚙️ Configuración AutoML")

# Backend selector - allow choosing between available backends
available_backends = []
backend_names = {}
if AUTOSKLEARN_AVAILABLE:
    available_backends.append("auto-sklearn")
    backend_names["auto-sklearn"] = "🐧 auto-sklearn (Linux optimizado)"
if FLAML_AVAILABLE:
    available_backends.append("flaml")
    backend_names["flaml"] = "🪟 FLAML (Cross-platform, rápido)"

if len(available_backends) > 1:
    BACKEND = st.sidebar.selectbox(
        "Backend AutoML",
        available_backends,
        format_func=lambda x: backend_names.get(x, x),
        index=0,
        help="Elige el motor de AutoML a utilizar"
    )
elif len(available_backends) == 1:
    BACKEND = available_backends[0]
    st.sidebar.info(f"🔧 **Backend:** {backend_names.get(BACKEND, BACKEND)}")
else:
    BACKEND = None
    st.sidebar.error("❌ No hay backends disponibles")

# Preset selector
preset = st.sidebar.selectbox(
    "Preset de búsqueda",
    ["quick", "balanced", "high_performance", "custom"],
    index=1,
    help="Configuración predefinida del tiempo y parámetros de búsqueda"
)

preset_descriptions = {
    "quick": "🚀 **Quick** (5 min): Exploración rápida",
    "balanced": "⚖️ **Balanced** (1 hora): Balance tiempo/rendimiento", 
    "high_performance": "🏆 **High Performance** (4 horas): Máximo rendimiento",
    "custom": "🔧 **Custom**: Configuración personalizada",
}
st.sidebar.markdown(preset_descriptions.get(preset, ""))

# Custom settings
if preset == "custom":
    time_budget = st.sidebar.slider(
        "Tiempo de búsqueda (minutos)",
        min_value=1,
        max_value=480,
        value=30,
        step=5,
        help="Tiempo total para la búsqueda de modelos"
    ) * 60  # Convert to seconds
    
    ensemble_size = st.sidebar.slider(
        "Tamaño del ensemble",
        min_value=1,
        max_value=100,
        value=50,
        help="Número de modelos en el ensemble final"
    )
else:
    # Default values based on preset
    preset_configs = {
        "quick": {"time_budget": 300, "ensemble_size": 10},
        "balanced": {"time_budget": 3600, "ensemble_size": 50},
        "high_performance": {"time_budget": 14400, "ensemble_size": 100},
    }
    time_budget = preset_configs[preset]["time_budget"]
    ensemble_size = preset_configs[preset]["ensemble_size"]

# Metric selection
metric = st.sidebar.selectbox(
    "Métrica de optimización",
    ["roc_auc", "accuracy", "f1", "precision", "recall", "balanced_accuracy"],
    index=0,
    help="Métrica a optimizar durante la búsqueda"
)

# ============================================================================
# Detect available estimators (check which libraries are installed)
# ============================================================================

def check_estimator_availability():
    """Check which estimators are available based on installed libraries."""
    available = {}
    missing_libs = {}
    
    # Estimators that require sklearn (always available if FLAML works)
    sklearn_estimators = ["rf", "extra_tree", "histgb", "kneighbor", "lrl1", "lrl2", "svc", "sgd"]
    for est in sklearn_estimators:
        available[est] = True
    
    # Check LightGBM
    try:
        import lightgbm
        available["lgbm"] = True
    except ImportError:
        available["lgbm"] = False
        missing_libs["lgbm"] = "lightgbm"
    
    # Check XGBoost
    try:
        import xgboost
        available["xgboost"] = True
        available["xgb_limitdepth"] = True
    except ImportError:
        available["xgboost"] = False
        available["xgb_limitdepth"] = False
        missing_libs["xgboost"] = "xgboost"
        missing_libs["xgb_limitdepth"] = "xgboost"
    
    # Check CatBoost
    try:
        import catboost
        available["catboost"] = True
    except ImportError:
        available["catboost"] = False
        missing_libs["catboost"] = "catboost"
    
    return available, missing_libs

ESTIMATOR_AVAILABILITY, MISSING_LIBRARIES = check_estimator_availability()

# Estimator selection (for FLAML)
if BACKEND == "flaml":
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔧 Estimadores")
    
    # Show missing libraries warning
    if MISSING_LIBRARIES:
        unique_libs = set(MISSING_LIBRARIES.values())
        with st.sidebar.expander("⚠️ Librerías faltantes", expanded=False):
            st.warning(f"Algunas librerías no están instaladas:")
            for lib in unique_libs:
                st.code(f"pip install {lib}", language="bash")
    
    # All possible estimators
    all_estimators_full = [
        "lgbm", "xgboost", "xgb_limitdepth", "catboost", "rf", "extra_tree", 
        "histgb", "kneighbor", "lrl1", "lrl2", "svc", "sgd"
    ]
    
    # Only available estimators
    available_estimators = [est for est in all_estimators_full if ESTIMATOR_AVAILABILITY.get(est, False)]
    unavailable_estimators = [est for est in all_estimators_full if not ESTIMATOR_AVAILABILITY.get(est, False)]
    
    # Preset selection for estimators
    estimator_preset = st.sidebar.selectbox(
        "Preset de estimadores",
        ["balanced", "gradient_boosting", "ensemble", "linear", "all_available", "custom"],
        index=0,
        help="Configuración predefinida de estimadores a probar"
    )
    
    # Estimator descriptions
    estimator_info = {
        "lgbm": ("🚀 LightGBM", "Gradient boosting rápido"),
        "xgboost": ("⚡ XGBoost", "Gradient boosting optimizado"),
        "xgb_limitdepth": ("🌳 XGBoost Limited", "XGBoost con profundidad limitada"),
        "catboost": ("🐱 CatBoost", "Mejor manejo de categóricas"),
        "rf": ("🌲 Random Forest", "Ensemble de árboles"),
        "extra_tree": ("🌴 Extra Trees", "Árboles extremadamente aleatorios"),
        "histgb": ("📊 HistGradientBoosting", "Boosting basado en histograma"),
        "kneighbor": ("👥 K-Neighbors", "Vecinos más cercanos"),
        "lrl1": ("📏 Logistic L1", "Regresión logística Lasso"),
        "lrl2": ("📐 Logistic L2", "Regresión logística Ridge"),
        "svc": ("🎯 SVC", "Support Vector Classifier"),
        "sgd": ("🔄 SGD", "Stochastic Gradient Descent"),
    }
    
    if estimator_preset == "custom":
        # Show multiselect with only available estimators
        default_available = [e for e in ["lgbm", "xgboost", "rf", "extra_tree", "histgb"] if e in available_estimators]
        
        selected_estimators = st.sidebar.multiselect(
            "Estimadores disponibles",
            available_estimators,
            default=default_available,
            help="Solo se muestran los estimadores con librerías instaladas"
        )
        
        # Show what's available vs missing
        with st.sidebar.expander("📖 Estado de estimadores"):
            st.markdown("**✅ Disponibles:**")
            for est in available_estimators:
                name, desc = estimator_info.get(est, (est, ""))
                st.markdown(f"- {name}")
            
            if unavailable_estimators:
                st.markdown("**❌ No instalados:**")
                for est in unavailable_estimators:
                    name, desc = estimator_info.get(est, (est, ""))
                    lib = MISSING_LIBRARIES.get(est, "unknown")
                    st.markdown(f"- {name} (`pip install {lib}`)")
    else:
        # Use preset but filter to only available estimators
        preset_configs = {
            "balanced": ["lgbm", "xgboost", "catboost", "rf", "extra_tree", "histgb"],
            "gradient_boosting": ["lgbm", "xgboost", "xgb_limitdepth", "catboost", "histgb"],
            "ensemble": ["rf", "extra_tree", "lgbm", "xgboost", "catboost"],
            "linear": ["lrl1", "lrl2", "sgd"],
            "all_available": available_estimators,
        }
        preset_selection = preset_configs.get(estimator_preset, available_estimators)
        # Filter to only available
        selected_estimators = [e for e in preset_selection if e in available_estimators]
        
        # Show info about what's selected vs skipped
        skipped = [e for e in preset_selection if e not in available_estimators]
        if skipped:
            st.sidebar.warning(f"⚠️ {len(skipped)} estimador(es) omitidos por librerías faltantes")
        st.sidebar.info(f"📋 {len(selected_estimators)} estimadores disponibles seleccionados")
else:
    selected_estimators = None

# Display config summary
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Resumen AutoML")
st.sidebar.markdown(f"- **Tiempo:** {time_budget // 60} minutos")
st.sidebar.markdown(f"- **Ensemble:** {ensemble_size} modelos")
st.sidebar.markdown(f"- **Métrica:** {metric}")
if selected_estimators:
    st.sidebar.markdown(f"- **Estimadores:** {len(selected_estimators)}")

# Note about NAS
if NAS_AVAILABLE:
    st.sidebar.markdown("---")
    st.sidebar.info("🧠 **NAS disponible** - Configura en la pestaña 'Neural Architecture Search'")
else:
    st.sidebar.markdown("---")
    st.sidebar.warning("🧠 **NAS no disponible** - Instala autokeras y tensorflow")

# ============================================================================
# MAIN CONTENT - Tabs
# ============================================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🚀 Entrenar AutoML",
    "🧠 Neural Architecture Search",
    "📊 Leaderboard",
    "💡 Sugerencias",
    "📥 Exportar"
])

# ============================================================================
# TAB 1: Train AutoML
# ============================================================================

with tab1:
    st.subheader("🚀 Entrenar Modelo AutoML")
    
    # Dataset info
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Muestras", len(df))
    with col2:
        st.metric("Features", len(df.columns) - 1)
    with col3:
        st.metric("Target", target_col)
    with col4:
        if target_col in df.columns:
            class_dist = df[target_col].value_counts()
            ratio = class_dist.iloc[0] / class_dist.iloc[-1] if len(class_dist) > 1 else 1
            st.metric("Desbalance", f"{ratio:.1f}:1")
    
    st.markdown("---")
    
    # Training progress container
    if 'automl_model' not in st.session_state:
        st.session_state.automl_model = None
    if 'automl_training' not in st.session_state:
        st.session_state.automl_training = False
    
    # Start training button
    if not st.session_state.automl_training:
        col1, col2 = st.columns([1, 3])
        with col1:
            start_btn = st.button("🚀 Iniciar AutoML", type="primary", use_container_width=True)
        with col2:
            st.info(f"⏱️ Tiempo estimado: **{time_budget // 60} minutos**")
        
        if start_btn:
            st.session_state.automl_training = True
            st.session_state.automl_logs = []
            st.session_state.automl_progress = 0.0
            st.session_state.automl_status = "Iniciando..."
            st.session_state.automl_result = None
            st.session_state.automl_error = None
            st.session_state.automl_start_time = time.time()
            st.session_state.automl_thread = None  # Reset thread
            st.session_state.automl_queue = queue.Queue()  # Fresh queue
            st.session_state.automl_thread_started = False  # Flag to track if thread was started
            st.rerun()
    
    if st.session_state.automl_training:
        st.warning("⏳ **Entrenamiento en progreso...** No cierres esta página.")
        
        # Initialize session state for async training (only if not already set)
        if 'automl_queue' not in st.session_state:
            st.session_state.automl_queue = queue.Queue()
        if 'automl_thread' not in st.session_state:
            st.session_state.automl_thread = None
        if 'automl_thread_started' not in st.session_state:
            st.session_state.automl_thread_started = False
        if 'automl_logs' not in st.session_state:
            st.session_state.automl_logs = []
        if 'automl_progress' not in st.session_state:
            st.session_state.automl_progress = 0.0
        if 'automl_status' not in st.session_state:
            st.session_state.automl_status = "Iniciando..."
        if 'automl_result' not in st.session_state:
            st.session_state.automl_result = None
        if 'automl_error' not in st.session_state:
            st.session_state.automl_error = None
        
        # Display current progress
        progress_bar = st.progress(st.session_state.automl_progress)
        status_text = st.empty()
        status_text.markdown(f"**{st.session_state.automl_status}**")
        
        # Progress metrics row
        progress_metrics = st.empty()
        
        # Display logs with dark background like NAS
        log_container = st.empty()
        with log_container.container():
            st.markdown("#### 📋 Log de Entrenamiento en Tiempo Real")
            if st.session_state.automl_logs:
                # Deduplicate consecutive similar logs
                unique_logs = []
                for log in st.session_state.automl_logs[-30:]:
                    if not unique_logs or log != unique_logs[-1]:
                        unique_logs.append(log)
                
                # Create styled HTML log display with dark background
                log_html = """
                <div style='background-color: #1e1e1e; color: #d4d4d4; padding: 15px; 
                            border-radius: 8px; font-family: "Consolas", "Monaco", monospace; 
                            font-size: 13px; max-height: 400px; overflow-y: auto;
                            border: 1px solid #333;'>
                """
                for log in unique_logs[-25:]:
                    # Color code based on content
                    if '✅' in log or 'completado' in log.lower() or 'done' in log.lower():
                        color = '#4CAF50'  # Green
                    elif '❌' in log or 'error' in log.lower():
                        color = '#f44336'  # Red
                    elif '🚀' in log or '🔄' in log or 'iniciando' in log.lower():
                        color = '#2196F3'  # Blue
                    elif '⚡' in log or 'mejor' in log.lower() or 'best' in log.lower():
                        color = '#FFD700'  # Gold
                    elif '📊' in log or '🔧' in log:
                        color = '#FF9800'  # Orange
                    elif '⏱️' in log or 'tiempo' in log.lower():
                        color = '#9C27B0'  # Purple
                    elif 'lgbm' in log.lower() or 'xgboost' in log.lower() or 'rf' in log.lower():
                        color = '#00BCD4'  # Cyan for estimators
                    else:
                        color = '#e0e0e0'  # Light gray
                    log_html += f"<div style='color: {color}; margin: 3px 0; line-height: 1.4;'>{log}</div>"
                log_html += "</div>"
                st.markdown(log_html, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style='background-color: #1e1e1e; color: #888; padding: 15px; 
                            border-radius: 8px; font-family: "Consolas", "Monaco", monospace; 
                            font-size: 13px; text-align: center;
                            border: 1px solid #333;'>
                    ⏳ Esperando logs del entrenamiento...
                </div>
                """, unsafe_allow_html=True)
        
        # Prepare data
        feature_cols = [c for c in df.columns if c != target_col]
        X = df[feature_cols].copy()
        y = df[target_col].copy()
        
        # Handle missing values for AutoML
        X = X.fillna(X.median(numeric_only=True))
        for col in X.select_dtypes(include=['object', 'category']).columns:
            X[col] = X[col].fillna(X[col].mode().iloc[0] if len(X[col].mode()) > 0 else "Unknown")
        
        # Training function to run in background thread
        def train_automl_background(X_data, y_data, msg_queue, config):
            """Run AutoML training in background thread."""
            import sys
            
            try:
                backend_name = config['backend'].upper() if config['backend'] else 'AutoML'
                msg_queue.put(('progress', f'🚀 Inicializando {backend_name}...', 0.01))
                
                # Progress callback that puts messages in queue
                def queue_callback(message: str, progress: float):
                    try:
                        msg_queue.put(('progress', message, progress))
                    except Exception:
                        pass  # Ignore queue errors
                
                # Create AutoML model
                if config['backend'] == "flaml":
                    from src.automl.flaml_integration import FLAMLClassifier
                    
                    msg_queue.put(('progress', '⚙️ Configurando FLAML...', 0.02))
                    
                    automl = FLAMLClassifier(
                        time_budget=config['time_budget'],
                        metric=config['metric'],
                        estimator_list=config['estimators'],
                        ensemble=config['ensemble'],
                        verbose=2,
                        name="AutoML-Dashboard",
                        progress_callback=queue_callback,
                    )
                else:
                    from src.automl.autosklearn_integration import AutoMLClassifier
                    msg_queue.put(('progress', '⚙️ Configurando auto-sklearn...', 0.02))
                    
                    automl = AutoMLClassifier(
                        preset=AutoMLPreset(config['preset']) if config['preset'] != "custom" else AutoMLPreset.CUSTOM,
                        time_left_for_this_task=config['time_budget'],
                        ensemble_size=config['ensemble_size'],
                        metric=config['metric'],
                        name="AutoML-Dashboard",
                        progress_callback=queue_callback,
                    )
                
                msg_queue.put(('progress', f'🔄 Iniciando búsqueda con {backend_name}...', 0.05))
                
                # Fit model
                automl.fit(X_data, y_data)
                
                # Send result
                msg_queue.put(('done', automl, None))
                
            except Exception as e:
                import traceback
                error_tb = traceback.format_exc()
                try:
                    msg_queue.put(('error', str(e), error_tb))
                except Exception:
                    pass  # If queue fails, at least we tried
            finally:
                # Ensure we always send something
                try:
                    msg_queue.put(('heartbeat', 'thread_ending', 0))
                except Exception:
                    pass
        
        # FIRST: Process messages from queue (before checking thread state)
        # This ensures we get 'done' or 'error' messages before detecting thread death
        try:
            while True:
                msg = st.session_state.automl_queue.get_nowait()
                msg_type = msg[0]
                
                if msg_type == 'progress':
                    _, message, progress = msg
                    st.session_state.automl_status = message
                    st.session_state.automl_progress = progress
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    st.session_state.automl_logs.append(f"[{timestamp}] {message}")
                    
                elif msg_type == 'done':
                    _, automl, _ = msg
                    st.session_state.automl_result = automl
                    st.session_state.automl_progress = 1.0
                    st.session_state.automl_status = "✅ Entrenamiento completado!"
                    
                elif msg_type == 'error':
                    _, error_msg, tb = msg
                    st.session_state.automl_error = error_msg
                    st.session_state.automl_error_details = tb
                    st.session_state.automl_logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ Error: {error_msg}")
                
                elif msg_type == 'heartbeat':
                    st.session_state.automl_thread_ended = True
                    
        except queue.Empty:
            pass
        
        # SECOND: Start training thread if not already running
        thread_is_alive = st.session_state.automl_thread is not None and st.session_state.automl_thread.is_alive()
        
        if not st.session_state.automl_thread_started:
            # First time - start the training thread
            config = {
                'backend': BACKEND,
                'time_budget': time_budget,
                'metric': metric,
                'estimators': selected_estimators if selected_estimators else None,
                'ensemble': ensemble_size > 1,
                'ensemble_size': ensemble_size,
                'preset': preset,
            }
            
            # Add initial logs
            st.session_state.automl_logs = [
                f"[{datetime.now().strftime('%H:%M:%S')}] 🚀 Iniciando {BACKEND.upper()} AutoML...",
                f"[{datetime.now().strftime('%H:%M:%S')}] ⏱️ Time budget: {time_budget}s ({time_budget // 60} min)",
                f"[{datetime.now().strftime('%H:%M:%S')}] 📊 Métrica objetivo: {metric}",
                f"[{datetime.now().strftime('%H:%M:%S')}] 🔧 Backend: {BACKEND}",
            ]
            if selected_estimators:
                st.session_state.automl_logs.append(
                    f"[{datetime.now().strftime('%H:%M:%S')}] 🤖 Estimadores: {', '.join(selected_estimators)}"
                )
            st.session_state.automl_logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] 🔄 Preparando datos...")
            
            thread = threading.Thread(
                target=train_automl_background,
                args=(X.copy(), y.copy(), st.session_state.automl_queue, config),
                daemon=True
            )
            thread.start()
            st.session_state.automl_thread = thread
            st.session_state.automl_thread_started = True
            
        # THIRD: Check for results (after processing queue)
        elif st.session_state.automl_result is not None:
            # Training completed successfully
            elapsed = time.time() - st.session_state.get('automl_start_time', time.time())
            automl = st.session_state.automl_result
            
            # Get training logs from model
            if hasattr(automl, 'get_training_logs'):
                training_logs = automl.get_training_logs()
                for log in training_logs[-20:]:
                    if log not in st.session_state.automl_logs:
                        st.session_state.automl_logs.append(log)
            
            # Store results
            st.session_state.automl_model = automl
            st.session_state.automl_training = False
            st.session_state.automl_elapsed = elapsed
            st.session_state.automl_feature_names = feature_cols
            st.session_state.automl_result = None
            st.session_state.automl_thread = None
            st.session_state.automl_thread_started = False
            
            progress_bar.progress(1.0)
            status_text.markdown("**✅ ¡Entrenamiento completado!**")
            
            best_score = -automl.best_loss_ if automl.best_loss_ < 0 else 1 - automl.best_loss_
            st.session_state.automl_logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ Completado: {automl.best_estimator_} (score={best_score:.4f})")
            
            st.success(f"""
            ### ✅ AutoML Completado
            
            - **Tiempo:** {elapsed:.1f} segundos
            - **Backend:** {BACKEND}
            - **Mejor modelo:** {getattr(automl, 'best_estimator_', 'ensemble')}
            - **Mejor score:** {best_score:.4f}
            """)
            
            st.balloons()
            time.sleep(1)
            st.rerun()
            
        elif st.session_state.automl_error is not None:
            # Training failed with error
            st.error(f"❌ Error en el entrenamiento: {st.session_state.automl_error}")
            if st.session_state.get('automl_error_details'):
                with st.expander("🔍 Detalles del error", expanded=True):
                    st.code(st.session_state.automl_error_details, language="text")
            st.session_state.automl_training = False
            st.session_state.automl_thread_started = False
            if st.button("🔄 Reintentar"):
                st.session_state.automl_error = None
                st.session_state.automl_error_details = None
                st.rerun()
                
        elif not thread_is_alive:
            # Thread died without sending done or error
            st.error("❌ El entrenamiento terminó inesperadamente.")
            st.warning("""
            **Posibles causas:**
            - Error de memoria
            - Problema con algún estimador
            - El proceso fue terminado externamente
            
            **Sugerencias:**
            - Reduce el tiempo de búsqueda
            - Usa menos estimadores
            - Revisa la consola para más detalles
            """)
            st.session_state.automl_logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ Hilo terminó sin resultado")
            st.session_state.automl_training = False
            st.session_state.automl_thread_started = False
            if st.button("🔄 Reintentar"):
                st.rerun()
        
        # Calculate elapsed time for display
        if 'automl_start_time' in st.session_state:
            elapsed = time.time() - st.session_state.automl_start_time
            remaining = max(0, time_budget - elapsed)
            mins_elapsed = int(elapsed // 60)
            secs_elapsed = int(elapsed % 60)
            mins_remaining = int(remaining // 60)
            secs_remaining = int(remaining % 60)
            
            # Always update progress based on time
            time_progress = min(elapsed / time_budget, 0.99)
            if time_progress > st.session_state.automl_progress:
                st.session_state.automl_progress = time_progress
            
            # Update progress bar with current time-based progress
            progress_bar.progress(st.session_state.automl_progress)
            
            # Update status with time info if no recent updates
            if "Iniciando" in st.session_state.automl_status or "Inicializando" in st.session_state.automl_status:
                animation_frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
                frame_idx = int(elapsed * 2) % len(animation_frames)
                spinner = animation_frames[frame_idx]
                st.session_state.automl_status = f"{spinner} {mins_elapsed:02d}:{secs_elapsed:02d} | 🔄 Buscando mejores modelos... | ⏳ {mins_remaining:02d}:{secs_remaining:02d}"
                status_text.markdown(f"**{st.session_state.automl_status}**")
            
            # Update progress metrics row (similar to NAS)
            with progress_metrics.container():
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("⏱️ Transcurrido", f"{mins_elapsed:02d}:{secs_elapsed:02d}")
                with col2:
                    st.metric("⏳ Restante", f"{mins_remaining:02d}:{secs_remaining:02d}")
                with col3:
                    progress_pct = int(st.session_state.automl_progress * 100)
                    st.metric("📊 Progreso", f"{progress_pct}%")
                with col4:
                    st.metric("🔧 Backend", BACKEND.upper() if BACKEND else "N/A")
        
        # Auto-refresh if training is still running
        if st.session_state.automl_thread and st.session_state.automl_thread.is_alive():
            time.sleep(2)
            st.rerun()
    
    # Show results if model exists
    if st.session_state.automl_model is not None and not st.session_state.automl_training:
        st.markdown("---")
        st.subheader("📈 Resultados del Entrenamiento")
        
        automl = st.session_state.automl_model
        
        col1, col2, col3 = st.columns(3)
        with col1:
            if hasattr(automl, 'best_estimator_'):
                st.metric("Mejor Estimador", automl.best_estimator_)
        with col2:
            if hasattr(automl, 'best_loss_'):
                st.metric("Mejor Score", f"{-automl.best_loss_:.4f}")
        with col3:
            if hasattr(automl, 'fit_time_'):
                st.metric("Tiempo Total", f"{automl.fit_time_:.1f}s")
        
        # Best config
        if hasattr(automl, 'best_config_'):
            with st.expander("🔧 Mejor Configuración"):
                st.json(automl.best_config_)
        
        # Training logs
        if 'automl_logs' in st.session_state and st.session_state.automl_logs:
            with st.expander("📋 Logs de Entrenamiento", expanded=False):
                st.code("\n".join(st.session_state.automl_logs), language="text")
                
                # Download logs button
                log_text = "\n".join(st.session_state.automl_logs)
                st.download_button(
                    label="📥 Descargar Logs",
                    data=log_text,
                    file_name=f"automl_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )

# ============================================================================
# TAB 2: Neural Architecture Search (NAS)
# ============================================================================

with tab2:
    st.subheader("🧠 Neural Architecture Search")
    
    if not NAS_AVAILABLE:
        st.warning("""
        ⚠️ **AutoKeras no está disponible.** 
        
        Para usar Neural Architecture Search, instala las dependencias:
        ```bash
        pip install autokeras tensorflow
        ```
        """)
        st.info("""
        **¿Qué es NAS?**
        
        Neural Architecture Search (NAS) busca automáticamente la mejor arquitectura 
        de red neuronal para tu problema. En lugar de diseñar manualmente la red,
        NAS explora diferentes combinaciones de:
        - Número de capas
        - Neuronas por capa
        - Funciones de activación
        - Tasas de dropout
        - Y más...
        """)
    else:
        st.success("✅ AutoKeras disponible para Neural Architecture Search")
        
        # Python 3.13 compatibility warning
        import sys
        if sys.version_info >= (3, 13):
            st.warning("""
            ⚠️ **Advertencia de compatibilidad**: Estás usando Python 3.13.
            
            AutoKeras/Keras puede tener problemas de threading con Python 3.13. 
            Si encuentras errores como `'NoneType' object has no attribute 'pop'`, 
            considera usar Python 3.11 o 3.12 para mejor compatibilidad.
            """)
        
        # Initialize NAS session state values for presets
        if 'nas_ui_trials' not in st.session_state:
            st.session_state.nas_ui_trials = 10
        if 'nas_ui_epochs' not in st.session_state:
            st.session_state.nas_ui_epochs = 50
        if 'nas_ui_tuner' not in st.session_state:
            st.session_state.nas_ui_tuner = "greedy"
        if 'nas_ui_metric' not in st.session_state:
            st.session_state.nas_ui_metric = "val_auc"
        
        # Preset definitions
        nas_presets = {
            "quick": {"name": "🚀 Quick (5-10 min)", "trials": 5, "epochs": 50, "tuner": "greedy"},
            "balanced": {"name": "⚖️ Balanced (30-60 min)", "trials": 15, "epochs": 100, "tuner": "bayesian"},
            "thorough": {"name": "🔬 Thorough (2-4 horas)", "trials": 30, "epochs": 150, "tuner": "hyperband"},
            "exhaustive": {"name": "🏆 Exhaustive (4+ horas)", "trials": 50, "epochs": 200, "tuner": "bayesian"},
            "custom": {"name": "🔧 Personalizado", "trials": None, "epochs": None, "tuner": None},
        }
        
        # Preset dropdown selector
        st.markdown("### 📋 Presets de NAS")
        
        def on_preset_change():
            """Callback when preset changes."""
            selected = st.session_state.nas_preset_selector_key
            if selected != "custom":
                preset_config = nas_presets[selected]
                st.session_state.nas_ui_trials = preset_config["trials"]
                st.session_state.nas_ui_epochs = preset_config["epochs"]
                st.session_state.nas_ui_tuner = preset_config["tuner"]
        
        preset_options = list(nas_presets.keys())
        
        selected_preset = st.selectbox(
            "Selecciona un preset de configuración",
            options=preset_options,
            format_func=lambda x: nas_presets[x]["name"],
            index=0,
            key="nas_preset_selector_key",
            on_change=on_preset_change,
            help="Elige una configuración predefinida o personalizada"
        )
        
        # Show preset description
        if selected_preset != "custom":
            preset_info = nas_presets[selected_preset]
            st.info(f"📋 **{preset_info['name']}**: {preset_info['trials']} arquitecturas, {preset_info['epochs']} epochs, tuner: {preset_info['tuner']}")
        else:
            st.info("🔧 **Personalizado**: Configura los parámetros manualmente abajo")
        
        st.markdown("---")
        
        # NAS Configuration - use session state values directly
        st.markdown("### ⚙️ Configuración")
        col1, col2 = st.columns(2)
        with col1:
            nas_max_trials = st.number_input(
                "Arquitecturas a probar",
                min_value=3,
                max_value=100,
                value=st.session_state.nas_ui_trials,
                help="Número de arquitecturas de red diferentes a evaluar"
            )
        with col2:
            nas_train_epochs = st.number_input(
                "Epochs de entrenamiento",
                min_value=10,
                max_value=500,
                value=st.session_state.nas_ui_epochs,
                help="Épocas de entrenamiento por arquitectura"
            )
        
        col3, col4 = st.columns(2)
        with col3:
            tuner_options = ["greedy", "bayesian", "hyperband", "random"]
            # Find current index
            current_tuner_idx = tuner_options.index(st.session_state.nas_ui_tuner) if st.session_state.nas_ui_tuner in tuner_options else 0
            nas_algorithm = st.selectbox(
                "Algoritmo de búsqueda",
                tuner_options,
                index=current_tuner_idx,
                help="Algoritmo para explorar el espacio de arquitecturas"
            )
        with col4:
            nas_metric_options = {
                "val_auc": "AUC-ROC (Área bajo la curva) ⭐",
                "val_accuracy": "Accuracy (Precisión general)",
                "val_loss": "Loss (Pérdida - minimizar)",
                "val_precision": "Precision (Precisión positiva)",
                "val_recall": "Recall (Sensibilidad)",
            }
            metric_keys = list(nas_metric_options.keys())
            current_metric_idx = metric_keys.index(st.session_state.nas_ui_metric) if st.session_state.nas_ui_metric in metric_keys else 0
            nas_metric = st.selectbox(
                "Métrica a optimizar",
                metric_keys,
                format_func=lambda x: nas_metric_options[x],
                index=current_metric_idx,
                help="Métrica que NAS intentará optimizar durante la búsqueda"
            )
        
        # Show current config summary
        st.markdown(f"""
        **Configuración actual:**
        - 🔬 Trials: `{nas_max_trials}` | ⏱️ Epochs: `{nas_train_epochs}` | 🔧 Tuner: `{nas_algorithm}` | 🎯 Métrica: `{nas_metric}`
        """)
        
        st.markdown("---")
        
        # Initialize NAS training state
        if 'nas_model' not in st.session_state:
            st.session_state.nas_model = None
        if 'nas_logs' not in st.session_state:
            st.session_state.nas_logs = []
        
        # Start NAS button - SYNCHRONOUS execution (no threads due to Keras 3.x compatibility)
        col1, col2 = st.columns([1, 3])
        with col1:
            start_nas_btn = st.button("🧠 Iniciar NAS", type="primary", use_container_width=True)
        with col2:
            estimated_time = nas_max_trials * nas_train_epochs * 2  # rough estimate in seconds
            st.info(f"⏱️ Tiempo estimado: **{estimated_time // 60} - {estimated_time // 30} minutos**")
        
        if start_nas_btn:
            # Prepare data
            feature_cols = [c for c in df.columns if c != target_col]
            X = df[feature_cols].copy()
            y = df[target_col].copy()
            
            # Handle missing values
            X = X.fillna(X.median(numeric_only=True))
            for col in X.select_dtypes(include=['object', 'category']).columns:
                X[col] = X[col].fillna(X[col].mode().iloc[0] if len(X[col].mode()) > 0 else "Unknown")
            
            # Convert categorical to numeric
            for col in X.select_dtypes(include=['object', 'category']).columns:
                X[col] = pd.Categorical(X[col]).codes
            
            # Convert to numpy with proper dtypes (critical for AutoKeras)
            import numpy as np
            X_np = X.values.astype(np.float32)
            y_np = y.values.astype(np.int32)
            
            # Split data for final evaluation
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X_np, y_np, test_size=0.2, random_state=42, stratify=y_np
            )
            
            # Initialize logs
            st.session_state.nas_logs = [
                f"[{datetime.now().strftime('%H:%M:%S')}] 🧠 Iniciando Neural Architecture Search...",
                f"[{datetime.now().strftime('%H:%M:%S')}] 📊 Dataset: {len(X_np)} muestras, {X_np.shape[1]} features",
                f"[{datetime.now().strftime('%H:%M:%S')}] 🎯 Trials: {nas_max_trials}",
                f"[{datetime.now().strftime('%H:%M:%S')}] ⏱️ Epochs: {nas_train_epochs}",
                f"[{datetime.now().strftime('%H:%M:%S')}] 🔧 Tuner: {nas_algorithm}",
                f"[{datetime.now().strftime('%H:%M:%S')}] 📈 Métrica: {nas_metric}",
            ]
            
            start_time = time.time()
            
            try:
                # Suppress TensorFlow warnings
                import os
                import sys
                import io
                import re
                os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
                os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
                
                # Import AutoKeras directly for synchronous execution
                import autokeras as ak
                import shutil
                
                # =====================================================================
                # MONKEY PATCH: Fix Keras 3.x compatibility issue with AutoKeras
                # Keras 3.13+ has stricter validation for 'units' parameter in Dense
                # layer. AutoKeras may pass numpy integers instead of Python int,
                # causing "ValueError: Received an invalid value for `units`"
                # =====================================================================
                try:
                    import keras
                    from keras.src.layers.core import dense as dense_module
                    _original_dense_init = keras.layers.Dense.__init__
                    
                    def _patched_dense_init(self, units, *args, **kwargs):
                        # Convert units to native Python int to avoid validation error
                        if hasattr(units, 'item'):  # numpy scalar
                            units = int(units.item())
                        else:
                            units = int(units)
                        return _original_dense_init(self, units, *args, **kwargs)
                    
                    keras.layers.Dense.__init__ = _patched_dense_init
                    st.session_state.nas_logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] 🔧 Aplicado parche de compatibilidad Keras 3.x")
                except Exception as patch_err:
                    st.session_state.nas_logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] ⚠️ No se pudo aplicar parche: {patch_err}")
                # =====================================================================
                
                # Clean up previous AutoKeras directory
                ak_dir = "./autokeras_dashboard"
                if os.path.exists(ak_dir):
                    shutil.rmtree(ak_dir)
                
                st.session_state.nas_logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] 🚀 Creando clasificador AutoKeras...")
                
                # Create UI containers for real-time updates
                status_container = st.empty()
                progress_container = st.empty()
                log_container = st.empty()
                metrics_container = st.empty()
                
                status_container.info("🧠 **Ejecutando Neural Architecture Search...** Los logs aparecerán abajo.")
                
                # Custom callback to capture training progress
                class StreamlitCallback:
                    def __init__(self, target_metric="val_auc"):
                        self.trial_logs = []
                        self.current_trial = 0
                        # Initialize best_metric_value based on metric type
                        # For loss, we want to minimize (start high), for others maximize (start at 0)
                        self.best_metric_value = float('inf') if target_metric == 'val_loss' else 0.0
                        self.target_metric = target_metric
                        self.epoch_data = []
                        
                    def parse_output(self, text):
                        """Parse AutoKeras/Keras output and extract useful info."""
                        lines = text.strip().split('\n')
                        parsed = []
                        
                        for line in lines:
                            line = line.strip()
                            if not line:
                                continue
                            
                            # Trial info
                            if 'Running Trial #' in line:
                                trial_match = re.search(r'Trial #(\d+)', line)
                                if trial_match:
                                    self.current_trial = int(trial_match.group(1))
                                    parsed.append(f"🔬 **Trial #{self.current_trial}** iniciando...")
                            
                            # Epoch progress with full metrics
                            elif 'Epoch' in line and '/' in line:
                                epoch_match = re.search(r'Epoch (\d+)/(\d+)', line)
                                if epoch_match:
                                    current_epoch = int(epoch_match.group(1))
                                    total_epochs = int(epoch_match.group(2))
                                    parsed.append(f"📊 Trial {self.current_trial} - Epoch {current_epoch}/{total_epochs}")
                            
                            # Progress bar lines with metrics (e.g., "62/62 ━━━━━━━━━━━━━━━━━━━━ 1s 10ms/step - auc: 0.7377...")
                            # Only show lines that have val_ metrics (end of epoch summary)
                            elif ('val_' in line.lower()) and ('━' in line or 'step' in line):
                                # Extract all metrics using case-insensitive matching
                                acc_match = re.search(r'(?<!val_)accuracy:\s*([\d.]+)', line, re.IGNORECASE)
                                val_acc_match = re.search(r'val_accuracy:\s*([\d.]+)', line, re.IGNORECASE)
                                loss_match = re.search(r'(?<!val_)loss:\s*([\d.]+)', line)
                                val_loss_match = re.search(r'val_loss:\s*([\d.]+)', line, re.IGNORECASE)
                                # Match both val_auc and val_AUC
                                val_auc_match = re.search(r'val_auc:\s*([\d.]+)', line, re.IGNORECASE)
                                auc_match = re.search(r'(?<!val_)auc:\s*([\d.]+)', line, re.IGNORECASE)
                                
                                # Build a nice summary line
                                metrics_parts = []
                                if val_auc_match:
                                    metrics_parts.append(f"AUC: {float(val_auc_match.group(1)):.4f}")
                                if val_acc_match:
                                    metrics_parts.append(f"Acc: {float(val_acc_match.group(1)):.4f}")
                                if val_loss_match:
                                    metrics_parts.append(f"Loss: {float(val_loss_match.group(1)):.4f}")
                                
                                if metrics_parts:
                                    parsed.append(f"   📈 Val Metrics: {' | '.join(metrics_parts)}")
                                
                                epoch_info = {
                                    'trial': self.current_trial,
                                    'accuracy': float(acc_match.group(1)) if acc_match else 0,
                                    'val_accuracy': float(val_acc_match.group(1)) if val_acc_match else 0,
                                    'loss': float(loss_match.group(1)) if loss_match else 0,
                                    'val_loss': float(val_loss_match.group(1)) if val_loss_match else 0,
                                    'val_auc': float(val_auc_match.group(1)) if val_auc_match else 0,
                                    'auc': float(auc_match.group(1)) if auc_match else 0,
                                }
                                
                                # Update best metric based on target (case-insensitive comparison)
                                target_lower = self.target_metric.lower()
                                if target_lower == 'val_auc' and val_auc_match:
                                    val = float(val_auc_match.group(1))
                                    if val > self.best_metric_value:
                                        self.best_metric_value = val
                                        parsed.append(f"   🏆 **Nuevo mejor AUC: {val:.4f}**")
                                elif target_lower == 'val_accuracy' and val_acc_match:
                                    val = float(val_acc_match.group(1))
                                    if val > self.best_metric_value:
                                        self.best_metric_value = val
                                        parsed.append(f"   🏆 **Nuevo mejor Accuracy: {val:.4f}**")
                                elif target_lower == 'val_loss' and val_loss_match:
                                    val = float(val_loss_match.group(1))
                                    # For loss, lower is better
                                    if val < self.best_metric_value:
                                        self.best_metric_value = val
                                        parsed.append(f"   🏆 **Nuevo mejor Loss: {val:.4f}**")
                                
                                if epoch_info['val_accuracy'] > 0 or epoch_info.get('val_auc', 0) > 0:
                                    self.epoch_data.append(epoch_info)
                            
                            # Metrics from epoch - capture all available metrics (fallback for other format)
                            elif ('accuracy:' in line.lower() or 'val_' in line.lower() or 'loss:' in line) and 'val_' in line.lower():
                                # Extract all metrics using case-insensitive matching
                                acc_match = re.search(r'accuracy:\s*([\d.]+)', line, re.IGNORECASE)
                                val_acc_match = re.search(r'val_accuracy:\s*([\d.]+)', line, re.IGNORECASE)
                                loss_match = re.search(r'(?<!val_)loss:\s*([\d.]+)', line)
                                val_loss_match = re.search(r'val_loss:\s*([\d.]+)', line, re.IGNORECASE)
                                val_auc_match = re.search(r'val_auc:\s*([\d.]+)', line, re.IGNORECASE)
                                auc_match = re.search(r'(?<!val_)auc:\s*([\d.]+)', line, re.IGNORECASE)
                                
                                epoch_info = {
                                    'trial': self.current_trial,
                                    'accuracy': float(acc_match.group(1)) if acc_match else 0,
                                    'val_accuracy': float(val_acc_match.group(1)) if val_acc_match else 0,
                                    'loss': float(loss_match.group(1)) if loss_match else 0,
                                    'val_loss': float(val_loss_match.group(1)) if val_loss_match else 0,
                                    'val_auc': float(val_auc_match.group(1)) if val_auc_match else 0,
                                    'auc': float(auc_match.group(1)) if auc_match else 0,
                                }
                                
                                # Update best metric based on target
                                target_lower = self.target_metric.lower()
                                if target_lower == 'val_auc' and val_auc_match:
                                    val = float(val_auc_match.group(1))
                                    if val > self.best_metric_value:
                                        self.best_metric_value = val
                                elif target_lower == 'val_accuracy' and val_acc_match:
                                    val = float(val_acc_match.group(1))
                                    if val > self.best_metric_value:
                                        self.best_metric_value = val
                                elif target_lower == 'val_loss' and val_loss_match:
                                    val = float(val_loss_match.group(1))
                                    if val < self.best_metric_value:
                                        self.best_metric_value = val
                                
                                if epoch_info['val_accuracy'] > 0 or epoch_info.get('val_auc', 0) > 0:
                                    self.epoch_data.append(epoch_info)
                            
                            # Trial complete
                            elif 'Trial' in line and 'Complete' in line:
                                time_match = re.search(r'\[([\dh\sm]+)\]', line)
                                time_str = time_match.group(1) if time_match else ""
                                parsed.append(f"✅ Trial {self.current_trial} completado {time_str}")
                            
                            # Best value - handle different metrics
                            elif 'Best' in line and 'So Far:' in line:
                                best_match = re.search(r'Best\s+(\w+)\s+So Far:\s*([\d.]+)', line, re.IGNORECASE)
                                if best_match:
                                    metric_name = best_match.group(1)
                                    metric_value = float(best_match.group(2))
                                    # Only update if it's our target metric
                                    if metric_name.lower() in self.target_metric.lower():
                                        self.best_metric_value = metric_value
                                    parsed.append(f"🏆 Mejor {metric_name}: **{metric_value:.4f}**")
                        
                        return parsed
                
                callback = StreamlitCallback(target_metric=nas_metric)
                
                # Capture stdout to get training logs
                class OutputCapture:
                    def __init__(self, callback, log_container, progress_container):
                        self.callback = callback
                        self.log_container = log_container
                        self.progress_container = progress_container
                        self.buffer = ""
                        self.all_output = []
                        self.original_stdout = sys.stdout
                        self.original_stderr = sys.stderr
                        
                    def write(self, text):
                        self.original_stdout.write(text)  # Also print to console
                        self.buffer += text
                        self.all_output.append(text)
                        
                        # Parse and update UI periodically
                        if '\n' in text or len(self.buffer) > 100:
                            parsed = self.callback.parse_output(self.buffer)
                            if parsed:
                                for msg in parsed:
                                    timestamp = datetime.now().strftime("%H:%M:%S")
                                    st.session_state.nas_logs.append(f"[{timestamp}] {msg}")
                                
                                # Update log display
                                with self.log_container.container():
                                    st.markdown("#### 📋 Log de Entrenamiento en Tiempo Real")
                                    # Show last 15 logs
                                    recent_logs = st.session_state.nas_logs[-15:]
                                    log_html = "<div style='background-color: #1e1e1e; padding: 10px; border-radius: 5px; font-family: monospace; font-size: 12px; max-height: 300px; overflow-y: auto;'>"
                                    for log in recent_logs:
                                        # Color code based on content
                                        if '✅' in log or '🏆' in log:
                                            color = '#4CAF50'
                                        elif '❌' in log:
                                            color = '#f44336'
                                        elif '🔬' in log:
                                            color = '#2196F3'
                                        elif '📊' in log:
                                            color = '#FF9800'
                                        else:
                                            color = '#e0e0e0'
                                        log_html += f"<div style='color: {color}; margin: 2px 0;'>{log}</div>"
                                    log_html += "</div>"
                                    st.markdown(log_html, unsafe_allow_html=True)
                                
                                # Update progress metrics
                                with self.progress_container.container():
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.metric("🔬 Trial Actual", f"{self.callback.current_trial}/{nas_max_trials}")
                                    with col2:
                                        metric_display_name = self.callback.target_metric.replace('val_', '').upper()
                                        best_val = self.callback.best_metric_value
                                        # Handle inf for val_loss before any training
                                        display_val = "---" if best_val == float('inf') else f"{best_val:.4f}"
                                        st.metric(f"🏆 Mejor {metric_display_name}", display_val)
                                    with col3:
                                        elapsed = time.time() - start_time
                                        st.metric("⏱️ Tiempo", f"{int(elapsed//60)}m {int(elapsed%60)}s")
                            
                            self.buffer = ""
                    
                    def flush(self):
                        self.original_stdout.flush()
                
                # Create output capture
                output_capture = OutputCapture(callback, log_container, progress_container)
                
                # Create classifier with user-selected metric and tuner
                st.session_state.nas_logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] 🎯 Métrica objetivo: {nas_metric}")
                st.session_state.nas_logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] 🔧 Algoritmo de búsqueda: {nas_algorithm}")
                
                # Configure objective with proper direction
                # IMPORTANT: We need to use keras.metrics objects and match the exact name Keras generates
                from keras_tuner import Objective
                import keras
                
                # Map user-friendly metric names to actual Keras metric objects and objective names
                # The metric name in history depends on the metric class name
                metric_config = {
                    "val_auc": {
                        # Use keras.metrics.AUC() - this generates "auc" in history, so "val_auc" for validation
                        "objective": Objective("val_auc", direction="max"),
                        "metrics": [keras.metrics.AUC(name='auc')],  # Explicitly name it 'auc'
                        "display_name": "AUC"
                    },
                    "val_accuracy": {
                        "objective": Objective("val_accuracy", direction="max"),
                        "metrics": ["accuracy"],
                        "display_name": "Accuracy"
                    },
                    "val_loss": {
                        "objective": Objective("val_loss", direction="min"),
                        "metrics": ["accuracy"],
                        "display_name": "Loss"
                    },
                    "val_precision": {
                        "objective": Objective("val_precision", direction="max"),
                        "metrics": [keras.metrics.Precision(name='precision')],
                        "display_name": "Precision"
                    },
                    "val_recall": {
                        "objective": Objective("val_recall", direction="max"),
                        "metrics": [keras.metrics.Recall(name='recall')],
                        "display_name": "Recall"
                    },
                }
                
                # Get config for selected metric, default to accuracy
                selected_config = metric_config.get(nas_metric, metric_config["val_accuracy"])
                objective = selected_config["objective"]
                metrics_to_track = selected_config["metrics"]
                
                st.session_state.nas_logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] 📊 Objective: {objective.name}")
                st.session_state.nas_logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] 📈 Metrics: {[str(m) for m in metrics_to_track]}")
                
                clf = ak.StructuredDataClassifier(
                    max_trials=nas_max_trials,
                    tuner=nas_algorithm,
                    objective=objective,
                    metrics=metrics_to_track,
                    overwrite=True,
                    directory=ak_dir,
                    project_name="nas_dashboard",
                    seed=42,
                )
                
                st.session_state.nas_logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] 🔍 Buscando mejor arquitectura...")
                
                # Redirect stdout to capture training output
                old_stdout = sys.stdout
                sys.stdout = output_capture
                
                try:
                    # Run fit
                    clf.fit(
                        X_train,
                        y_train,
                        epochs=nas_train_epochs,
                        validation_split=0.2,
                        verbose=1,
                    )
                finally:
                    sys.stdout = old_stdout
                
                elapsed = time.time() - start_time
                
                # Export best model
                best_model = clf.export_model()
                
                # Get predictions for evaluation
                y_pred_proba = best_model.predict(X_test, verbose=0)
                y_pred = (y_pred_proba > 0.5).astype(int).flatten()
                
                # Calculate metrics
                from sklearn.metrics import (
                    accuracy_score, precision_score, recall_score, f1_score,
                    roc_auc_score, confusion_matrix, classification_report
                )
                
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, zero_division=0)
                recall = recall_score(y_test, y_pred, zero_division=0)
                f1 = f1_score(y_test, y_pred, zero_division=0)
                
                try:
                    roc_auc = roc_auc_score(y_test, y_pred_proba)
                except:
                    roc_auc = 0.0
                
                conf_matrix = confusion_matrix(y_test, y_pred)
                class_report = classification_report(y_test, y_pred, output_dict=True)
                
                # Get architecture summary
                stream = io.StringIO()
                best_model.summary(print_fn=lambda x: stream.write(x + '\n'))
                architecture_summary = stream.getvalue()
                
                # Store results
                st.session_state.nas_model = {
                    'classifier': clf,
                    'best_model': best_model,
                    'architecture_summary': architecture_summary,
                    'fit_time': elapsed,
                    'max_trials': nas_max_trials,
                    'epochs': nas_train_epochs,
                    'target_metric': nas_metric,
                    'metrics': {
                        'accuracy': accuracy,
                        'precision': precision,
                        'recall': recall,
                        'f1_score': f1,
                        'roc_auc': roc_auc,
                        'best_metric_value': callback.best_metric_value,
                    },
                    'confusion_matrix': conf_matrix,
                    'classification_report': class_report,
                    'epoch_data': callback.epoch_data,
                    'training_logs': output_capture.all_output,
                }
                
                st.session_state.nas_logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ NAS completado en {elapsed:.1f}s")
                st.session_state.nas_logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] 🎯 Accuracy: {accuracy:.4f}")
                st.session_state.nas_logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] 📊 ROC-AUC: {roc_auc:.4f}")
                
                # Clean up
                if os.path.exists(ak_dir):
                    shutil.rmtree(ak_dir)
                
                st.success(f"""
                ### ✅ Neural Architecture Search Completado
                
                - **Arquitecturas evaluadas:** {nas_max_trials}
                - **Epochs por arquitectura:** {nas_train_epochs}
                - **Tiempo total:** {elapsed:.1f} segundos ({elapsed/60:.1f} minutos)
                - **Mejor accuracy en test:** {accuracy:.4f}
                """)
                st.balloons()
                time.sleep(1)
                st.rerun()
                
            except Exception as e:
                import traceback
                error_tb = traceback.format_exc()
                st.session_state.nas_logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ Error: {str(e)}")
                
                st.error(f"❌ Error en NAS: {str(e)}")
                with st.expander("🔍 Detalles del error", expanded=True):
                    st.code(error_tb, language="text")
                
                # Provide helpful suggestions
                st.warning("""
                **Posibles soluciones:**
                - Reduce el número de trials (prueba con 3-5)
                - Reduce el número de epochs (prueba con 10-20)
                - Verifica que los datos no contengan valores NaN o infinitos
                """)
        
        # Show results if model exists
        if st.session_state.nas_model is not None:
            st.markdown("---")
            st.subheader("📈 Resultados de NAS")
            
            nas_result = st.session_state.nas_model
            
            # Main metrics row
            st.markdown("### 🎯 Métricas de Rendimiento")
            
            metrics = nas_result.get('metrics', {})
            
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                acc = metrics.get('accuracy', 0)
                st.metric("🎯 Accuracy", f"{acc:.4f}", delta=f"{(acc-0.5)*100:.1f}% vs random")
            with col2:
                prec = metrics.get('precision', 0)
                st.metric("📍 Precision", f"{prec:.4f}")
            with col3:
                rec = metrics.get('recall', 0)
                st.metric("🔍 Recall", f"{rec:.4f}")
            with col4:
                f1 = metrics.get('f1_score', 0)
                st.metric("⚖️ F1-Score", f"{f1:.4f}")
            with col5:
                auc = metrics.get('roc_auc', 0)
                st.metric("📊 ROC-AUC", f"{auc:.4f}")
            
            # Training info row
            st.markdown("### ⏱️ Información del Entrenamiento")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("🔬 Arquitecturas Probadas", nas_result.get('max_trials', 'N/A'))
            with col2:
                st.metric("📚 Epochs por Trial", nas_result.get('epochs', 'N/A'))
            with col3:
                fit_time = nas_result.get('fit_time', 0)
                st.metric("⏱️ Tiempo Total", f"{fit_time:.1f}s")
            with col4:
                best_val = metrics.get('best_metric_value', 0)
                target_metric = nas_result.get('target_metric', 'val_accuracy')
                metric_name = target_metric.replace('val_', '').upper()
                st.metric(f"🏆 Mejor {metric_name}", f"{best_val:.4f}")
            
            # Confusion Matrix visualization
            if 'confusion_matrix' in nas_result:
                st.markdown("### 📊 Matriz de Confusión")
                conf_matrix = nas_result['confusion_matrix']
                
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    # Create a styled confusion matrix display
                    import plotly.figure_factory as ff
                    import plotly.express as px
                    
                    fig = px.imshow(
                        conf_matrix,
                        labels=dict(x="Predicción", y="Real", color="Cantidad"),
                        x=['Negativo (0)', 'Positivo (1)'],
                        y=['Negativo (0)', 'Positivo (1)'],
                        color_continuous_scale='Blues',
                        text_auto=True,
                    )
                    fig.update_layout(
                        title="Matriz de Confusión",
                        width=400,
                        height=400,
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Classification report
                    st.markdown("#### 📋 Reporte de Clasificación")
                    class_report = nas_result.get('classification_report', {})
                    
                    if class_report:
                        report_df = pd.DataFrame(class_report).transpose()
                        # Format numeric columns
                        for col in ['precision', 'recall', 'f1-score']:
                            if col in report_df.columns:
                                report_df[col] = report_df[col].apply(lambda x: f"{x:.4f}" if isinstance(x, float) else x)
                        if 'support' in report_df.columns:
                            report_df['support'] = report_df['support'].apply(lambda x: f"{int(x)}" if isinstance(x, float) else x)
                        
                        st.dataframe(report_df, use_container_width=True)
            
            # Training history visualization
            if 'epoch_data' in nas_result and nas_result['epoch_data']:
                st.markdown("### 📈 Historial de Entrenamiento")
                
                epoch_data = nas_result['epoch_data']
                epoch_df = pd.DataFrame(epoch_data)
                
                if not epoch_df.empty:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Accuracy over epochs
                        fig_acc = px.line(
                            epoch_df,
                            y=['accuracy', 'val_accuracy'],
                            title='Accuracy por Epoch',
                            labels={'value': 'Accuracy', 'index': 'Epoch'},
                        )
                        fig_acc.update_layout(
                            legend_title="Métrica",
                            hovermode='x unified',
                        )
                        st.plotly_chart(fig_acc, use_container_width=True)
                    
                    with col2:
                        # Loss over epochs
                        fig_loss = px.line(
                            epoch_df,
                            y=['loss', 'val_loss'],
                            title='Loss por Epoch',
                            labels={'value': 'Loss', 'index': 'Epoch'},
                        )
                        fig_loss.update_layout(
                            legend_title="Métrica",
                            hovermode='x unified',
                        )
                        st.plotly_chart(fig_loss, use_container_width=True)
            
            # Architecture summary
            if 'architecture_summary' in nas_result:
                with st.expander("🏗️ Arquitectura del Mejor Modelo", expanded=False):
                    st.code(nas_result['architecture_summary'], language="text")
            
            # Show training logs from session state
            if 'nas_logs' in st.session_state and st.session_state.nas_logs:
                with st.expander("📋 Logs de Entrenamiento NAS", expanded=False):
                    # Create formatted log display
                    log_html = "<div style='background-color: #1e1e1e; padding: 15px; border-radius: 8px; font-family: monospace; font-size: 12px; max-height: 400px; overflow-y: auto;'>"
                    for log in st.session_state.nas_logs:
                        if '✅' in log or '🏆' in log:
                            color = '#4CAF50'
                        elif '❌' in log:
                            color = '#f44336'
                        elif '🔬' in log:
                            color = '#2196F3'
                        elif '📊' in log or '🎯' in log:
                            color = '#FF9800'
                        else:
                            color = '#e0e0e0'
                        log_html += f"<div style='color: {color}; margin: 3px 0;'>{log}</div>"
                    log_html += "</div>"
                    st.markdown(log_html, unsafe_allow_html=True)
                    
                    # Download logs
                    log_text = "\n".join(st.session_state.nas_logs)
                    st.download_button(
                        label="📥 Descargar Logs NAS",
                        data=log_text,
                        file_name=f"nas_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain"
                    )
            
            # Actions row
            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("🗑️ Limpiar Resultados", type="secondary", use_container_width=True):
                    st.session_state.nas_model = None
                    st.session_state.nas_logs = []
                    st.rerun()
            
            with col2:
                # Export model button
                if st.button("💾 Guardar Modelo", type="primary", use_container_width=True):
                    try:
                        import os
                        model_dir = "./models/nas"
                        os.makedirs(model_dir, exist_ok=True)
                        model_path = os.path.join(model_dir, f"nas_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
                        nas_result['best_model'].save(model_path)
                        st.success(f"✅ Modelo guardado en: `{model_path}`")
                    except Exception as e:
                        st.error(f"❌ Error guardando modelo: {e}")
            
            with col3:
                # Store in session for prediction
                if st.button("📤 Usar para Predicción", type="primary", use_container_width=True):
                    st.session_state['trained_nas_model'] = nas_result['best_model']
                    st.session_state['nas_metrics'] = nas_result.get('metrics', {})
                    st.success("✅ Modelo listo para usar en predicción")

# ============================================================================
# TAB 3: Leaderboard
# ============================================================================

with tab3:
    st.subheader("📊 Leaderboard de Modelos")
    
    if st.session_state.automl_model is None:
        st.info("🔍 Entrena un modelo AutoML primero para ver el leaderboard")
    else:
        automl = st.session_state.automl_model
        
        if hasattr(automl, 'get_leaderboard'):
            leaderboard = automl.get_leaderboard()
            if leaderboard is not None and len(leaderboard) > 0:
                st.dataframe(leaderboard, use_container_width=True)
                
                # Visualize scores
                if 'mean_score' in leaderboard.columns:
                    fig = px.bar(
                        leaderboard.head(10),
                        x=leaderboard.index[:10],
                        y='mean_score',
                        title="Top 10 Modelos por Score",
                        labels={'x': 'Modelo', 'mean_score': 'Score'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No hay datos de leaderboard disponibles")
        
        # Search history for FLAML
        if hasattr(automl, 'get_search_history'):
            history = automl.get_search_history()
            if len(history) > 0:
                st.markdown("### 🔍 Historial de Búsqueda")
                st.dataframe(history, use_container_width=True)

# ============================================================================
# TAB 4: Suggestions
# ============================================================================

with tab4:
    st.subheader("💡 Sugerencias Inteligentes")
    
    st.markdown("""
    El sistema analiza las características de tu dataset y sugiere técnicas
    que podrían mejorar el rendimiento del modelo.
    """)
    
    if st.button("🔍 Analizar Dataset", type="primary"):
        with st.spinner("Analizando dataset..."):
            # Analyze
            suggestions_engine = AutoMLSuggestions()
            analysis = suggestions_engine.analyze_dataset(df, target_col, task="classification")
            suggestions = suggestions_engine.generate_suggestions()
            
            # Store for display
            st.session_state.dataset_analysis = analysis
            st.session_state.suggestions = suggestions
    
    if 'dataset_analysis' in st.session_state:
        analysis = st.session_state.dataset_analysis
        suggestions = st.session_state.suggestions
        
        # Display analysis
        st.markdown("### 📊 Análisis del Dataset")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Muestras", analysis.n_samples)
        with col2:
            st.metric("Features Numéricos", analysis.n_numeric_features)
        with col3:
            st.metric("Features Categóricos", analysis.n_categorical_features)
        with col4:
            st.metric("% Datos Faltantes", f"{analysis.missing_percentage:.1f}%")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Desbalanceado", "Sí" if analysis.is_imbalanced else "No")
        with col2:
            st.metric("Ratio de Clases", f"{analysis.imbalance_ratio:.1f}:1")
        
        st.markdown("---")
        
        # Display suggestions
        st.markdown("### 💡 Sugerencias")
        
        if not suggestions:
            st.success("✅ No se detectaron problemas significativos en el dataset")
        else:
            # Group by priority
            from src.automl.suggestions import Priority
            
            high_priority = [s for s in suggestions if s.priority == Priority.HIGH]
            medium_priority = [s for s in suggestions if s.priority == Priority.MEDIUM]
            low_priority = [s for s in suggestions if s.priority == Priority.LOW]
            
            if high_priority:
                st.markdown("#### 🔴 Alta Prioridad")
                for s in high_priority:
                    with st.expander(f"🔴 {s.title}", expanded=True):
                        st.markdown(f"**{s.description}**")
                        st.markdown(f"**Razón:** {s.reason}")
                        st.markdown(f"**Beneficio esperado:** {s.expected_benefit}")
                        if s.module_link:
                            st.markdown(f"**Módulo:** `{s.module_link}`")
                        if s.code_example:
                            st.code(s.code_example, language="python")
            
            if medium_priority:
                st.markdown("#### 🟡 Prioridad Media")
                for s in medium_priority:
                    with st.expander(f"🟡 {s.title}"):
                        st.markdown(f"**{s.description}**")
                        st.markdown(f"**Razón:** {s.reason}")
                        st.markdown(f"**Beneficio esperado:** {s.expected_benefit}")
                        if s.module_link:
                            st.markdown(f"**Módulo:** `{s.module_link}`")
            
            if low_priority:
                st.markdown("#### 🟢 Baja Prioridad")
                for s in low_priority:
                    with st.expander(f"🟢 {s.title}"):
                        st.markdown(f"**{s.description}**")
                        st.markdown(f"**Razón:** {s.reason}")

# ============================================================================
# TAB 5: Export
# ============================================================================

with tab5:
    st.subheader("📥 Exportar Modelo")
    
    if st.session_state.automl_model is None:
        st.info("🔍 Entrena un modelo AutoML primero para exportarlo")
    else:
        automl = st.session_state.automl_model
        
        st.markdown("""
        Exporta el mejor modelo encontrado por AutoML para usarlo en producción
        o en otras partes de la aplicación.
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            model_name = st.text_input(
                "Nombre del modelo",
                value="automl_best_model",
                help="Nombre para el archivo del modelo"
            )
        
        with col2:
            export_dir = st.text_input(
                "Directorio de exportación",
                value=str(Path(CONFIG.models_dir) / "automl"),
                help="Directorio donde guardar el modelo"
            )
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("💾 Exportar Mejor Modelo", type="primary", use_container_width=True):
                try:
                    output_path = export_best_model(
                        automl_model=automl,
                        output_dir=export_dir,
                        model_name=model_name,
                        include_metadata=True,
                        training_data=df,
                        target_column=target_col,
                    )
                    st.success(f"✅ Modelo exportado: `{output_path}`")
                except Exception as e:
                    st.error(f"❌ Error al exportar: {e}")
        
        with col2:
            if st.button("📄 Generar Reporte", use_container_width=True):
                try:
                    report = create_automl_report(automl)
                    st.text_area("Reporte AutoML", report, height=400)
                    
                    # Download button
                    st.download_button(
                        "📥 Descargar Reporte",
                        report,
                        file_name="automl_report.txt",
                        mime="text/plain"
                    )
                except Exception as e:
                    st.error(f"❌ Error al generar reporte: {e}")
        
        # Feature importance
        if hasattr(automl, 'get_feature_importance'):
            importance = automl.get_feature_importance()
            if importance is not None and len(importance) > 0:
                st.markdown("---")
                st.markdown("### 📊 Importancia de Features")
                
                feature_names = st.session_state.get('automl_feature_names', [f"F{i}" for i in range(len(importance))])
                
                importance_df = pd.DataFrame({
                    'Feature': feature_names[:len(importance)],
                    'Importance': importance
                }).sort_values('Importance', ascending=False)
                
                fig = px.bar(
                    importance_df.head(20),
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    title="Top 20 Features más Importantes"
                )
                fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# Footer
# ============================================================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
<p><strong>AutoML Module</strong> | Mortality AMI Predictor</p>
<p>💡 <em>AutoML busca automáticamente la mejor configuración para tu dataset</em></p>
</div>
""", unsafe_allow_html=True)
