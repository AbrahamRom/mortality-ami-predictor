"""Predictions page."""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Add parent directories to path
root_dir = Path(__file__).parents[2]
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from app import (
    get_state,
    initialize_state,
    list_saved_models,
)
from src.config import CONFIG

# Initialize
initialize_state()

# Page config
st.title("🔮 Make Predictions")
st.markdown("---")

# Check if data has been loaded
cleaned_data = st.session_state.get('cleaned_data')
raw_data = st.session_state.get('raw_data')

if cleaned_data is not None:
    df = cleaned_data
    st.success("✅ Usando datos limpios")
elif raw_data is not None:
    df = raw_data
    st.warning("⚠️ Usando datos crudos")
else:
    st.warning("⚠️ No hay datos cargados. Por favor, carga un dataset en la página **🧹 Data Cleaning and EDA** primero.")
    st.stop()

# Get target column from session state (now stores actual column name)
target_col_name = st.session_state.get('target_column_name', None)

# Determine task for model folder organization
# This maps the target column to the folder where models are saved
if target_col_name:
    if target_col_name == CONFIG.target_column or target_col_name in ['mortality', 'mortality_inhospital', 'exitus']:
        task = 'mortality'
    elif target_col_name == CONFIG.arrhythmia_column or target_col_name in ['arrhythmia', 'ventricular_arrhythmia']:
        task = 'arrhythmia'
    else:
        task = target_col_name.lower().replace(' ', '_')[:20]
else:
    # Fallback to old behavior
    task = st.session_state.get('target_column', 'mortality')
    if task == 'exitus':
        task = 'mortality'

# Model selection
st.sidebar.markdown("---")
st.sidebar.header("🎯 Model Selection")

# Try to load models from task folder, fallback to 'mortality' if empty
saved_models = list_saved_models(task)

# If no models found for custom task, also check 'mortality' folder
if not saved_models and task not in ['mortality', 'arrhythmia']:
    st.info(f"ℹ️ No models found for task '{task}', checking 'mortality' folder...")
    saved_models = list_saved_models('mortality')

if not saved_models:
    st.error(f"❌ No trained models found for task '{task}'. Please train models first.")
    st.stop()

selected_model_name = st.sidebar.selectbox(
    "Choose Model",
    list(saved_models.keys()),
    help="Select a trained model for predictions"
)

model_path = saved_models[selected_model_name]
st.sidebar.success(f"✅ Using: {selected_model_name}")

# Load model
try:
    model = joblib.load(model_path)
    st.success(f"✅ Model loaded: {selected_model_name}")
except EOFError:
    st.error(f"❌ El archivo del modelo '{selected_model_name}' está corrupto (EOFError). Por favor, elimine el archivo y vuelva a entrenar el modelo.")
    st.info(f"📁 Ruta del archivo: `{model_path}`")
    st.stop()
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.exception(e)
    st.stop()
    
    # Load model metadata if available
    metadata_path = Path(str(model_path).replace('.joblib', '.metadata.json'))
    model_metadata = None
    if metadata_path.exists():
        try:
            with open(metadata_path, 'r') as f:
                model_metadata = json.load(f)
            st.info("ℹ️ Model metadata loaded successfully")
        except Exception as e:
            st.warning(f"⚠️ Could not load metadata: {e}")
    
    # ==================== LOAD TRANSFORMER IF EXISTS ====================
    transformer = None
    transformation_info = None
    
    if model_metadata and 'transformation' in model_metadata:
        transformation_info = model_metadata['transformation']
        transformer_path_str = transformation_info.get('transformer_path')
        
        if transformer_path_str:
            transformer_path = Path(transformer_path_str)
            
            if transformer_path.exists():
                try:
                    transformer = joblib.load(str(transformer_path))
                    
                    transform_type = transformation_info['type'].upper()
                    n_comps = transformation_info['n_components']
                    
                    st.success(f"""
                    ✅ **Transformer cargado: {transform_type}**
                    
                    - Tipo: {transform_type}
                    - Componentes: {n_comps}
                    - Variables originales: {len(transformation_info.get('original_features', []))}
                    
                    **Las predicciones aplicarán automáticamente esta transformación a tus datos.**
                    """)
                    
                    # Show transformation details
                    with st.expander("🔄 Detalles de la Transformación"):
                        st.write(f"**Tipo de transformación:** {transform_type}")
                        st.write(f"**Número de componentes:** {n_comps}")
                        
                        if transform_type == 'PCA':
                            variance = transformation_info.get('params', {}).get('variance_explained', 0)
                            st.write(f"**Varianza explicada:** {variance*100:.2f}%")
                            st.write(f"**Estandarizado:** {'Sí' if transformation_info.get('params', {}).get('standardize', False) else 'No'}")
                        elif transform_type == 'ICA':
                            kurtosis = transformation_info.get('params', {}).get('kurtosis_mean', 0)
                            st.write(f"**Kurtosis promedio:** {kurtosis:.3f}")
                            st.write(f"**Algoritmo:** {transformation_info.get('params', {}).get('algorithm', 'N/A')}")
                            st.write(f"**Función:** {transformation_info.get('params', {}).get('fun', 'N/A')}")
                        
                        st.write("**Variables originales transformadas:**")
                        original_feats = transformation_info.get('original_features', [])
                        st.write(f"{len(original_feats)} variables: {', '.join(original_feats[:10])}{'...' if len(original_feats) > 10 else ''}")
                
                except Exception as e:
                    st.error(f"❌ Error cargando transformer: {e}")
                    transformer = None
            else:
                st.warning(f"⚠️ Transformer no encontrado en: `{transformer_path}`")
    
    # Determine target column
    target_col = CONFIG.target_column if task == "mortality" else CONFIG.arrhythmia_column
    
    # Try to get expected features from multiple sources (in priority order)
    expected_features = None
    feature_source = "unknown"
    
    # 1. Try from model metadata (most reliable)
    if model_metadata:
        if model_metadata.get('dataset', {}).get('feature_names'):
            expected_features = model_metadata['dataset']['feature_names']
            feature_source = "metadata"
        elif model_metadata.get('feature_names'):
            expected_features = model_metadata['feature_names']
            feature_source = "metadata"
    
    # 2. Try from model object itself
    if not expected_features:
        if hasattr(model, 'feature_names_in_'):
            expected_features = list(model.feature_names_in_)
            feature_source = "model.feature_names_in_"
        elif hasattr(model, 'named_steps'):
            # For pipelines, check the last step
            for step_name, step in reversed(list(model.named_steps.items())):
                if hasattr(step, 'feature_names_in_'):
                    expected_features = list(step.feature_names_in_)
                    feature_source = f"pipeline.{step_name}.feature_names_in_"
                    break
    
    # Display feature information
    if expected_features:
        st.info(f"ℹ️ Model expects {len(expected_features)} features (source: {feature_source})")
        
        # Display metadata info if available
        if model_metadata:
            with st.expander("📋 Model Metadata"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Training Date", model_metadata.get('created_at', 'N/A')[:10])
                with col2:
                    n_features = model_metadata.get('dataset', {}).get('n_features', 'N/A')
                    st.metric("Features", n_features)
                with col3:
                    perf = model_metadata.get('performance', {})
                    mean_score = perf.get('mean_score', 0.0)
                    st.metric("CV Score", f"{mean_score:.4f}" if mean_score else "N/A")
                
                # Training info
                training = model_metadata.get('training', {})
                if training:
                    st.write("**Training Configuration:**")
                    st.write(f"- CV Strategy: {training.get('cv_strategy', 'N/A')}")
                    st.write(f"- Total Runs: {training.get('total_cv_runs', 'N/A')}")
                    st.write(f"- Scoring: {training.get('scoring_metric', 'N/A')}")
        
        # Check if current data has these features
        missing_features = set(expected_features) - set(df.columns)
        extra_features = set(df.columns) - set(expected_features) - {target_col}
        
        if missing_features:
            st.error(f"❌ **INCOMPATIBLE DATASET:** Current dataset is missing {len(missing_features)} feature(s) that the model expects")
            with st.expander("🔍 Ver características faltantes (primeras 20)"):
                st.write(sorted(list(missing_features))[:20])
            
            st.warning("""
            **Soluciones posibles:**
            1. Cargar el dataset original con el que se entrenó este modelo
            2. Seleccionar otro modelo compatible con el dataset actual
            3. Verificar que las variables no hayan sido renombradas
            """)
            
            # Try to find training data
            training_data_path = Path(str(model_path).replace('.joblib', '_training_data.parquet'))
            if not training_data_path.exists():
                training_data_path = Path(str(model_path).replace('.joblib', '_training_data.csv'))
            
            if training_data_path.exists():
                st.info(f"💡 Dataset de entrenamiento encontrado: `{training_data_path.name}`")
                if st.button("Cargar dataset de entrenamiento"):
                    try:
                        if training_data_path.suffix == '.parquet':
                            df = pd.read_parquet(training_data_path)
                        else:
                            df = pd.read_csv(training_data_path)
                        st.session_state.df = df
                        st.success("✅ Dataset de entrenamiento cargado")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error cargando dataset: {e}")
            
            st.stop()
        
        if extra_features:
            st.info(f"ℹ️ Current dataset has {len(extra_features)} extra feature(s) that will be ignored")
            # Filter to only use expected features
            feature_cols = [c for c in expected_features if c in df.columns]
        else:
            feature_cols = expected_features
    else:
        # Fallback: use all non-target columns
        st.warning("⚠️ Could not determine expected features from model or metadata. Using all available features.")
        feature_cols = [c for c in df.columns if c not in {CONFIG.target_column, CONFIG.arrhythmia_column}]
    
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.exception(e)
    st.stop()

st.markdown("---")

# Tabs for different prediction modes
tab1, tab2 = st.tabs(["🎯 Single Prediction", "📊 Batch Predictions"])

with tab1:
    st.subheader("Single Row Prediction")
    
    row_index = st.number_input(
        "Select Row Index",
        min_value=0,
        max_value=max(0, len(df) - 1),
        value=0,
        help="Choose a row from the dataset to predict"
    )
    
    X_row = df.loc[[row_index], feature_cols].copy()
    
    # Display original values
    with st.expander("📋 Original Row Values"):
        st.dataframe(X_row.T, width='stretch')
    
    # What-if analysis for numeric features
    st.subheader("What-If Analysis")
    st.caption("Adjust numeric features to see how predictions change")
    
    numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    
    if numeric_cols:
        with st.form("whatif_form"):
            cols_per_row = 2
            for i in range(0, len(numeric_cols), cols_per_row):
                cols = st.columns(cols_per_row)
                for j, col_name in enumerate(numeric_cols[i:i+cols_per_row]):
                    with cols[j]:
                        # Get current value
                        raw_val = df.loc[row_index, col_name]
                        val = float(raw_val) if pd.notna(raw_val) else 0.0
                        
                        # Compute robust bounds
                        s = pd.to_numeric(df[col_name], errors="coerce")
                        s = s[np.isfinite(s)]
                        
                        if len(s) == 0:
                            low, high = val - 1.0, val + 1.0
                        else:
                            try:
                                low = float(np.nanpercentile(s, 5))
                                high = float(np.nanpercentile(s, 95))
                            except Exception:
                                low, high = float(s.min()), float(s.max())
                        
                        # Ensure finite bounds
                        if not np.isfinite(low):
                            low = val - 1.0
                        if not np.isfinite(high):
                            high = val + 1.0
                        
                        # Ensure valid range
                        if not (high > low):
                            center = val if np.isfinite(val) else (float(s.median()) if len(s) else 0.0)
                            pad = abs(center) * 0.1 + 1.0
                            low, high = center - pad, center + pad
                        
                        # Clamp value
                        if val < low:
                            val = low
                        elif val > high:
                            val = high
                        
                        X_row.loc[row_index, col_name] = st.slider(
                            col_name,
                            min_value=low,
                            max_value=high,
                            value=val,
                            key=f"slider_{col_name}"
                        )
            
            predict_btn = st.form_submit_button("🎯 Predict", width='stretch')
    else:
        predict_btn = st.button("🎯 Predict", width='stretch')
    
    # Make prediction
    if predict_btn or not numeric_cols:
        st.markdown("---")
        st.subheader("Prediction Results")
        
        try:
            # ==================== APPLY TRANSFORMATION IF EXISTS ====================
            X_to_predict = X_row.copy()
            
            if transformer is not None and transformation_info is not None:
                st.info(f"🔄 Aplicando transformación {transformation_info['type'].upper()}...")
                
                # Get original features that need transformation
                original_features = transformation_info.get('original_features', [])
                
                # Check if all required features are present
                missing_feats = set(original_features) - set(X_row.columns)
                if missing_feats:
                    st.error(f"❌ Faltan variables para la transformación: {missing_feats}")
                    st.stop()
                
                # Extract only the features that were transformed
                X_original = X_row[original_features]
                
                # Apply transformation based on type
                if transformation_info['type'] == 'pca':
                    # PCA transformation
                    scaler = transformer.get('scaler')
                    pca = transformer.get('pca')
                    
                    if scaler is not None:
                        X_scaled = scaler.transform(X_original)
                    else:
                        X_scaled = X_original.values
                    
                    X_transformed = pca.transform(X_scaled)
                    
                    # Create DataFrame with component names
                    component_names = [f'PC{i+1}' for i in range(pca.n_components_)]
                    X_to_predict = pd.DataFrame(
                        X_transformed,
                        columns=component_names,
                        index=X_row.index
                    )
                    
                    st.success(f"✅ Transformación PCA aplicada: {len(original_features)} variables → {pca.n_components_} componentes")
                
                elif transformation_info['type'] == 'ica':
                    # ICA transformation
                    X_transformed = transformer.transform(X_original)
                    X_to_predict = X_transformed
                    
                    st.success(f"✅ Transformación ICA aplicada: {len(original_features)} variables → {transformation_info['n_components']} componentes independientes")
                
                # Show transformation preview
                with st.expander("🔍 Ver datos transformados"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Datos Originales (primeras 5 vars):**")
                        st.dataframe(X_original.iloc[:, :5], width='stretch')
                    with col2:
                        st.write(f"**Datos Transformados ({transformation_info['type'].upper()}):**")
                        st.dataframe(X_to_predict, width='stretch')
            
            # Make prediction with (possibly transformed) data
            if hasattr(model, "predict_proba"):
                prob = model.predict_proba(X_to_predict)[:, 1][0]
                # Convert to Python float (st.progress requires float, not numpy.float32)
                prob = float(prob)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Predicted Probability", f"{prob:.4f}")
                with col2:
                    risk_level = "High" if prob > 0.5 else "Low"
                    st.metric("Risk Level", risk_level)
                
                # Visual probability bar
                st.progress(prob)
            else:
                pred = model.predict(X_to_predict)[0]
                st.metric("Prediction", str(pred))
        
        except Exception as e:
            st.error(f"❌ Prediction error: {e}")
            st.exception(e)

with tab2:
    st.subheader("Batch Predictions")
    st.caption("Upload a CSV file for batch predictions")
    
    uploaded_file = st.file_uploader(
        "Upload CSV",
        type=["csv"],
        help="Upload a CSV file with the same features as the training data"
    )
    
    if uploaded_file is not None:
        try:
            # Try multiple encodings for CSV files
            encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'windows-1252']
            upload_df = None
            last_error = None
            
            for encoding in encodings:
                try:
                    upload_df = pd.read_csv(uploaded_file, encoding=encoding)
                    break
                except (UnicodeDecodeError, LookupError) as e:
                    last_error = e
                    continue
            
            if upload_df is None:
                raise RuntimeError(
                    f"No se pudo decodificar el CSV. Error: {last_error}"
                )
            
            st.success(f"✅ Uploaded {len(upload_df)} rows")
            
            with st.expander("📋 Preview Uploaded Data"):
                st.dataframe(upload_df.head(10), width='stretch')
            
            if st.button("🎯 Predict All Rows", type="primary"):
                with st.spinner("Making predictions..."):
                    # ==================== APPLY TRANSFORMATION IF EXISTS ====================
                    df_to_predict = upload_df.copy()
                    
                    if transformer is not None and transformation_info is not None:
                        st.info(f"🔄 Aplicando transformación {transformation_info['type'].upper()} a {len(upload_df)} filas...")
                        
                        # Get original features that need transformation
                        original_features = transformation_info.get('original_features', [])
                        
                        # Check if all required features are present
                        missing_feats = set(original_features) - set(upload_df.columns)
                        if missing_feats:
                            st.error(f"❌ Faltan variables para la transformación: {missing_feats}")
                            st.stop()
                        
                        # Extract only the features that were transformed
                        df_original = upload_df[original_features]
                        
                        # Apply transformation based on type
                        if transformation_info['type'] == 'pca':
                            # PCA transformation
                            scaler = transformer.get('scaler')
                            pca = transformer.get('pca')
                            
                            if scaler is not None:
                                data_scaled = scaler.transform(df_original)
                            else:
                                data_scaled = df_original.values
                            
                            data_transformed = pca.transform(data_scaled)
                            
                            # Create DataFrame with component names
                            component_names = [f'PC{i+1}' for i in range(pca.n_components_)]
                            df_to_predict = pd.DataFrame(
                                data_transformed,
                                columns=component_names,
                                index=upload_df.index
                            )
                            
                            st.success(f"✅ Transformación PCA aplicada: {len(original_features)} variables → {pca.n_components_} componentes")
                        
                        elif transformation_info['type'] == 'ica':
                            # ICA transformation
                            df_to_predict = transformer.transform(df_original)
                            
                            st.success(f"✅ Transformación ICA aplicada: {len(original_features)} variables → {transformation_info['n_components']} componentes independientes")
                        
                        # Show transformation summary
                        with st.expander("🔍 Ver resumen de transformación"):
                            st.write(f"**Tipo:** {transformation_info['type'].upper()}")
                            st.write(f"**Filas procesadas:** {len(upload_df)}")
                            st.write(f"**Variables originales:** {len(original_features)}")
                            st.write(f"**Componentes resultantes:** {df_to_predict.shape[1]}")
                            
                            st.write("**Primeras 3 filas transformadas:**")
                            st.dataframe(df_to_predict.head(3), width='stretch')
                    
                    # Make predictions with (possibly transformed) data
                    if hasattr(model, "predict_proba"):
                        probs = model.predict_proba(df_to_predict)[:, 1]
                        output_df = upload_df.copy()
                        output_df["prediction_probability"] = probs
                        output_df["prediction_class"] = (probs > 0.5).astype(int)
                    else:
                        preds = model.predict(df_to_predict)
                        output_df = upload_df.copy()
                        output_df["prediction"] = preds
                    
                    st.success(f"✅ Predictions completed for {len(output_df)} rows")
                    
                    # Display results
                    st.dataframe(output_df, width='stretch')
                    
                    # Download button
                    csv_data = output_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Predictions",
                        data=csv_data,
                        file_name=f"predictions_{task}_{selected_model_name}.csv",
                        mime="text/csv",
                        width='stretch'
                    )
        
        except Exception as e:
            st.error(f"❌ Error processing uploaded file: {e}")
            st.exception(e)
