"""Data Overview and Exploration page."""
from __future__ import annotations

import sys
from pathlib import Path

# Add parent directories to path
root_dir = Path(__file__).parents[2]
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import json
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from app import (
    display_data_audit,
    display_dataframe_info,
    display_dataset_preview,
    initialize_state,
)
from src.config import CONFIG
from src.features import safe_feature_columns

# Initialize
initialize_state()

# Page config
st.title("📊 Data Overview & Exploration")
st.markdown("---")

# Check if data has been loaded in Data Cleaning page
cleaned_data = st.session_state.get('cleaned_data')
raw_data = st.session_state.get('raw_data')

if cleaned_data is not None:
    df = cleaned_data
    st.success("✅ Usando datos limpios del proceso de limpieza")
elif raw_data is not None:
    df = raw_data
    st.warning("⚠️ Usando datos crudos (se recomienda limpiar primero en la página anterior)")
else:
    st.warning("⚠️ No hay datos cargados. Por favor, carga un dataset en la página **🧹 Data Cleaning and EDA** primero.")
    st.stop()

# Display basic info
display_dataframe_info(df)

st.markdown("---")

# Tabs for different views
tab1, tab2, tab3 = st.tabs(["📋 Preview", "🔍 Quality Audit", "📈 Statistics"])

with tab1:
    n_rows = st.slider("Number of rows to display", 5, 50, 10)
    display_dataset_preview(df, n_rows=n_rows)

with tab2:
    # Determine target column from session state - prefer target_column_name (actual column)
    target = st.session_state.get('target_column_name', None)
    if not target:
        target = st.session_state.get('target_column', CONFIG.target_column)
    
    # Get feature columns
    if target and target in df.columns:
        feature_cols = safe_feature_columns(df, [target])
    else:
        feature_cols = list(df.columns)
    
    st.info(f"Analyzing {len(feature_cols)} feature columns (excluding target: {target})")
    display_data_audit(df, feature_cols)

with tab3:
    st.subheader("📊 Column Statistics")
    
    # Select columns to analyze
    all_cols = df.columns.tolist()
    selected_cols = st.multiselect(
        "Select columns to view statistics",
        all_cols,
        default=all_cols[:5] if len(all_cols) > 5 else all_cols
    )
    
    if selected_cols:
        st.dataframe(
            df[selected_cols].describe(),
            width='stretch'
        )
        
        # Missing values visualization
        st.subheader("Missing Values")
        missing_df = df[selected_cols].isnull().sum().to_frame("Missing Count")
        missing_df["Missing %"] = (missing_df["Missing Count"] / len(df) * 100).round(2)
        
        st.dataframe(missing_df, width='stretch')
    else:
        st.info("Select at least one column to view statistics")

st.markdown("---")

# Variable Metadata Viewer
st.subheader("🔬 Variable Details")

# Load metadata if available
metadata_path = Path(root_dir) / CONFIG.metadata_path
metadata_dict = {}

if metadata_path.exists():
    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata_dict = json.load(f)
        
        # Remove special keys
        special_keys = [k for k in metadata_dict.keys() if k.startswith('_')]
        for key in special_keys:
            metadata_dict.pop(key, None)
        
        if metadata_dict:
            st.success(f"✅ Metadatos cargados: {len(metadata_dict)} variables")
            
            # Variable selector
            variable_names = sorted([v for v in metadata_dict.keys() if v in df.columns])
            
            if variable_names:
                selected_var = st.selectbox(
                    "Selecciona una variable para ver detalles:",
                    variable_names,
                    index=0
                )
                
                if selected_var and selected_var in metadata_dict:
                    meta = metadata_dict[selected_var]
                    
                    # Display metadata in columns
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("### 📋 Información General")
                        
                        # Description (user-editable)
                        if meta.get('description'):
                            st.info(f"**Descripción:** {meta['description']}")
                        else:
                            st.warning("⚠️ Sin descripción. Puedes editarla manualmente en el archivo metadata JSON.")
                        
                        st.write(f"**Nombre:** `{meta.get('name', selected_var)}`")
                        st.write(f"**Tipo Original:** `{meta.get('original_type', 'N/A')}`")
                        st.write(f"**Tipo Limpio:** `{meta.get('cleaned_type', 'N/A')}`")
                        st.write(f"**Numérica:** {'✅ Sí' if meta.get('is_numerical', False) else '❌ No'}")
                        st.write(f"**Categórica:** {'✅ Sí' if meta.get('is_categorical', False) else '❌ No'}")
                    
                    with col2:
                        st.markdown("### 📊 Estadísticas")
                        
                        if meta.get('is_numerical'):
                            # Numerical stats
                            original_min = meta.get('original_min')
                            original_max = meta.get('original_max')
                            cleaned_min = meta.get('cleaned_min')
                            cleaned_max = meta.get('cleaned_max')
                            
                            if original_min is not None:
                                st.metric("Rango Original", f"[{original_min:.2f}, {original_max:.2f}]")
                            if cleaned_min is not None:
                                st.metric("Rango Limpio", f"[{cleaned_min:.2f}, {cleaned_max:.2f}]")
                        else:
                            # Categorical stats
                            unique_vals = meta.get('unique_values', [])
                            if unique_vals:
                                st.write(f"**Valores únicos:** {len(unique_vals)}")
                                with st.expander("Ver valores"):
                                    st.write(unique_vals)
                    
                    # Quality metrics
                    st.markdown("### 🎯 Métricas de Calidad")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        missing_count = meta.get('missing_count_original', 0)
                        missing_pct = meta.get('missing_percent_original', 0.0)
                        st.metric("Valores Faltantes", f"{missing_count} ({missing_pct:.1f}%)")
                    
                    with col2:
                        outliers = meta.get('outliers_detected', 0)
                        st.metric("Outliers Detectados", outliers)
                    
                    with col3:
                        outliers_treated = meta.get('outliers_treated', 0)
                        st.metric("Outliers Tratados", outliers_treated)
                    
                    with col4:
                        imputation = meta.get('imputation_method', 'N/A')
                        st.metric("Método de Imputación", imputation if imputation else 'Ninguno')
                    
                    # Transformations applied
                    st.markdown("### 🔧 Transformaciones Aplicadas")
                    
                    transformations = []
                    
                    if meta.get('encoding_type'):
                        transformations.append(f"**Encoding:** {meta['encoding_type']}")
                        if meta.get('encoding_mapping'):
                            with st.expander("Ver mapeo de encoding"):
                                st.json(meta['encoding_mapping'])
                    
                    if meta.get('discretization_bins'):
                        transformations.append(f"**Discretización:** {len(meta['discretization_bins'])-1} bins")
                        if meta.get('discretization_labels'):
                            with st.expander("Ver bins y etiquetas"):
                                st.write("**Bins:**", meta['discretization_bins'])
                                st.write("**Etiquetas:**", meta['discretization_labels'])
                    
                    if transformations:
                        for t in transformations:
                            st.markdown(f"- {t}")
                    else:
                        st.info("No se aplicaron transformaciones especiales")
                    
                    # Quality flags
                    quality_flags = meta.get('quality_flags', [])
                    if quality_flags:
                        st.markdown("### ⚠️ Indicadores de Calidad")
                        for flag in quality_flags:
                            if 'no_missing' in flag.lower():
                                st.success(f"✅ {flag}")
                            elif 'outliers' in flag.lower():
                                st.warning(f"⚠️ {flag}")
                            elif 'missing' in flag.lower():
                                st.error(f"🔴 {flag}")
                            else:
                                st.info(f"ℹ️ {flag}")
                    
                    # Distribution plot
                    if selected_var in df.columns:
                        st.markdown("### 📈 Distribución")
                        
                        if meta.get('is_numerical'):
                            # Numerical distribution
                            fig = px.histogram(
                                df,
                                x=selected_var,
                                title=f"Distribución de {selected_var}",
                                marginal="box"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            # Categorical distribution
                            value_counts = df[selected_var].value_counts().head(20)
                            fig = px.bar(
                                x=value_counts.index,
                                y=value_counts.values,
                                title=f"Top 20 valores de {selected_var}",
                                labels={'x': selected_var, 'y': 'Frecuencia'}
                            )
                            st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Las variables en los metadatos no coinciden con las columnas del dataset actual.")
        else:
            st.info("El archivo de metadatos está vacío o solo contiene información general.")
    
    except Exception as e:
        st.error(f"❌ Error cargando metadatos: {e}")
        st.info("Los metadatos se generan automáticamente al limpiar datos en la página **🧹 Data Cleaning and EDA**")
else:
    st.info(f"""
    📝 **Metadatos no encontrados**
    
    Los metadatos de las variables se generan automáticamente durante el proceso de limpieza.
    
    Para generar metadatos:
    1. Ve a la página **🧹 Data Cleaning and EDA**
    2. Carga y limpia un dataset
    3. Los metadatos se guardarán automáticamente en: `{metadata_path}`
    
    Los metadatos incluyen información detallada sobre cada variable, transformaciones aplicadas,
    y métricas de calidad.
    """)
