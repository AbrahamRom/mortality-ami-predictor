"""Página de Streamlit para Preparación de Datos y Análisis Exploratorio.

Esta página proporciona una interfaz completa para:
- Limpieza de datos con opciones configurables
- Análisis exploratorio univariado, bivariado y multivariado
- Visualizaciones interactivas
- Generación de reportes de calidad
- Exportación de datos limpios y análisis
"""
from __future__ import annotations

import sys
from pathlib import Path

# Add parent directories to path
root_dir = Path(__file__).parents[2]
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import json
import warnings
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Import dashboard config for absolute paths
from dashboard.app.config import CLEANED_DATASETS_DIR, PLOTS_EDA_DIR

# Import src modules
from src.config import CONFIG
from src.cleaning import CleaningConfig, DataCleaner, quick_clean
from src.eda import EDAAnalyzer, quick_eda
from src.eda import generate_univariate_pdf, generate_bivariate_pdf, generate_multivariate_pdf
from src.features import ICATransformer, compare_pca_vs_ica
from src.reporting import pdf_export_section

warnings.filterwarnings('ignore')

# Configuración de página
st.set_page_config(
    page_title="Preparación de Datos y EDA - AMI Mortality Predictor",
    layout="wide",
    page_icon="🧹"
)


def init_session_state():
    """Inicializa el estado de la sesión."""
    if 'raw_data' not in st.session_state:
        st.session_state.raw_data = None
    if 'cleaned_data' not in st.session_state:
        st.session_state.cleaned_data = None
    if 'cleaner' not in st.session_state:
        st.session_state.cleaner = None
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = None
    if 'cleaning_config' not in st.session_state:
        st.session_state.cleaning_config = CleaningConfig()
    if 'preserved_clinical_scores' not in st.session_state:
        st.session_state.preserved_clinical_scores = {}


def load_data_page():
    """Sección de carga de datos."""
    st.header("📂 Carga de Datos")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Opción 1: Cargar desde ruta
        data_path = st.text_input(
            "Ruta del dataset",
            value=CONFIG.dataset_path,
            help="Ruta al archivo CSV o Excel del dataset"
        )
        
        if st.button("Cargar dataset desde ruta", type="primary"):
            try:
                if data_path.endswith('.csv'):
                    # Try multiple encodings for CSV files
                    encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'windows-1252']
                    df = None
                    last_error = None
                    
                    for encoding in encodings:
                        try:
                            df = pd.read_csv(data_path, encoding=encoding)
                            break
                        except (UnicodeDecodeError, LookupError) as e:
                            last_error = e
                            continue
                    
                    if df is None:
                        raise RuntimeError(
                            f"No se pudo decodificar el CSV con ninguna codificación. Error: {last_error}"
                        )
                elif data_path.endswith(('.xlsx', '.xls')):
                    df = pd.read_excel(data_path)
                else:
                    st.error("Formato no soportado. Use CSV o Excel.")
                    return
                
                st.session_state.raw_data = df
                st.session_state.data_path = data_path  # Guardar la ruta del dataset
                st.success(f"✅ Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas")
            except Exception as e:
                st.error(f"❌ Error al cargar datos: {e}")
    
    with col2:
        # Opción 2: Subir archivo
        uploaded = st.file_uploader(
            "O sube un archivo",
            type=['csv', 'xlsx', 'xls'],
            help="Arrastra y suelta un archivo CSV o Excel"
        )
        
        if uploaded is not None:
            try:
                if uploaded.name.endswith('.csv'):
                    # Try multiple encodings for CSV files
                    encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'windows-1252']
                    df = None
                    last_error = None
                    
                    for encoding in encodings:
                        try:
                            df = pd.read_csv(uploaded, encoding=encoding)
                            break
                        except (UnicodeDecodeError, LookupError) as e:
                            last_error = e
                            continue
                    
                    if df is None:
                        raise RuntimeError(
                            f"No se pudo decodificar el CSV. Error: {last_error}"
                        )
                else:
                    df = pd.read_excel(uploaded)
                
                st.session_state.raw_data = df
                
                # Guardar archivo subido en temporal para uso posterior
                import tempfile
                temp_dir = Path(tempfile.gettempdir())
                temp_path = temp_dir / f"uploaded_{uploaded.name}"
                if uploaded.name.endswith('.csv'):
                    df.to_csv(temp_path, index=False)
                else:
                    df.to_excel(temp_path, index=False)
                st.session_state.data_path = str(temp_path)
                
                st.success(f"✅ Archivo cargado: {df.shape[0]} filas, {df.shape[1]} columnas")
            except Exception as e:
                st.error(f"❌ Error al leer archivo: {e}")
    
    # Opción 3: Cargar dataset limpio existente
    st.markdown("---")
    st.subheader("Cargar dataset limpio existente")
    
    # Use absolute path from dashboard config
    cleaned_dir = CLEANED_DATASETS_DIR
    if cleaned_dir.exists():
        cleaned_files = sorted(cleaned_dir.glob("cleaned_dataset_*.csv"), 
                              key=lambda p: p.stat().st_mtime, reverse=True)
        
        if cleaned_files:
            file_options = {f.name: str(f) for f in cleaned_files[:10]}
            selected_file = st.selectbox("Datasets limpios disponibles", list(file_options.keys()))
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Cargar dataset limpio"):
                    try:
                        # Try multiple encodings for CSV files
                        encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'windows-1252']
                        df = None
                        last_error = None
                        
                        for encoding in encodings:
                            try:
                                df = pd.read_csv(file_options[selected_file], encoding=encoding)
                                break
                            except (UnicodeDecodeError, LookupError) as e:
                                last_error = e
                                continue
                        
                        if df is None:
                            raise RuntimeError(
                                f"No se pudo decodificar el CSV. Error: {last_error}"
                            )
                        
                        st.session_state.cleaned_data = df
                        st.session_state.data_path = str(file_options[selected_file])
                        st.success(f"✅ Dataset limpio cargado: {df.shape}")
                    except Exception as e:
                        st.error(f"❌ Error: {e}")
            
            with col2:
                if st.button("Cargar metadatos asociados"):
                    try:
                        metadata_file = Path(CONFIG.metadata_path)
                        if metadata_file.exists():
                            with open(metadata_file, 'r', encoding='utf-8') as f:
                                metadata = json.load(f)
                            st.success("✅ Metadatos cargados")
                            with st.expander("Ver metadatos"):
                                st.json(metadata)
                        else:
                            st.warning("No se encontró archivo de metadatos")
                    except Exception as e:
                        st.error(f"❌ Error: {e}")
    
    # Vista previa de datos
    if st.session_state.raw_data is not None:
        st.markdown("---")
        st.subheader("Vista previa de datos crudos")
        
        df = st.session_state.raw_data
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Filas", f"{df.shape[0]:,}")
        col2.metric("Columnas", f"{df.shape[1]:,}")
        col3.metric("Valores faltantes", f"{df.isna().sum().sum():,}")
        col4.metric("% Completitud", f"{(1 - df.isna().mean().mean()) * 100:.1f}%")
        
        st.dataframe(df.head(100), width='stretch', height=300)


def variable_selection_page():
    """Sección para descartar variables irrelevantes antes de la limpieza."""
    st.header("🎯 Selección de Variables")
    
    # Verificar si hay datos disponibles (crudos o limpios)
    if st.session_state.raw_data is None and st.session_state.cleaned_data is None:
        st.warning("⚠️ Primero carga un dataset en la pestaña 'Carga de Datos'")
        return
    
    # Usar datos disponibles (priorizar raw_data si existe)
    if st.session_state.raw_data is not None:
        df = st.session_state.raw_data.copy()
        data_source = "datos crudos"
        data_key = "raw_data"
    else:
        df = st.session_state.cleaned_data.copy()
        data_source = "datos limpios"
        data_key = "cleaned_data"
    
    # Mostrar información del origen de datos
    col_info1, col_info2 = st.columns([3, 1])
    
    with col_info1:
        st.info(f"📊 Trabajando con: **{data_source}** ({df.shape[0]:,} filas × {df.shape[1]:,} columnas)")
    
    with col_info2:
        # Botón para cargar selección guardada
        if st.button("📂 Cargar Selección", use_container_width=True):
            try:
                config_dir = Path(CONFIG.preprocessing_config_path).parent
                selection_path = config_dir / "variable_selection.json"
                
                if selection_path.exists():
                    with open(selection_path, 'r', encoding='utf-8') as f:
                        selection_config = json.load(f)
                    
                    # Verificar que las variables existan en el dataset actual
                    vars_to_keep = set(selection_config['variables_to_keep'])
                    available_vars = set(df.columns)
                    
                    # Solo mantener variables que existen en el dataset actual
                    valid_vars = vars_to_keep.intersection(available_vars)
                    
                    if valid_vars:
                        st.session_state.variables_to_keep = valid_vars
                        st.session_state.variables_to_drop = available_vars - valid_vars
                        
                        # Eliminar el estado del multiselect para que se reconstruya
                        if 'multiselect_vars' in st.session_state:
                            del st.session_state.multiselect_vars
                        
                        st.success(f"✅ Selección cargada: {len(valid_vars)} variables")
                        
                        if len(vars_to_keep - available_vars) > 0:
                            st.warning(f"⚠️ {len(vars_to_keep - available_vars)} variables de la selección guardada no existen en el dataset actual")
                        
                        st.rerun()
                    else:
                        st.error("❌ Ninguna variable de la selección guardada existe en el dataset actual")
                else:
                    st.warning("⚠️ No hay selección guardada previamente")
            except Exception as e:
                st.error(f"❌ Error al cargar selección: {e}")
    
    # Inicializar variables en session_state
    if 'variables_to_keep' not in st.session_state:
        st.session_state.variables_to_keep = set(df.columns.tolist())
    
    if 'variables_to_drop' not in st.session_state:
        st.session_state.variables_to_drop = set()
    
    # Información general
    st.info("""
    **👉 Instrucciones:** Selecciona las variables que deseas **mantener** para el análisis y limpieza. 
    Las variables no seleccionadas serán descartadas antes de iniciar la limpieza de datos.
    """)
    
    # Estadísticas de variables
    col1, col2, col3, col4 = st.columns(4)
    total_vars = len(df.columns)
    vars_to_keep = len(st.session_state.variables_to_keep)
    vars_to_drop = len(st.session_state.variables_to_drop)
    
    col1.metric("📊 Variables Totales", total_vars)
    col2.metric("✅ Variables Seleccionadas", vars_to_keep, delta=None)
    col3.metric("🗑️ Variables a Descartar", vars_to_drop, delta=None, delta_color="inverse")
    col4.metric("📈 % Seleccionadas", f"{(vars_to_keep/total_vars*100):.1f}%")
    
    st.markdown("---")
    
    # Tabs para diferentes métodos de selección
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎨 Selección Visual", 
        "🔍 Búsqueda y Filtrado",
        "📊 Análisis de Calidad",
        "💾 Aplicar Cambios"
    ])
    
    # Tab 1: Selección visual con multiselect mejorado
    with tab1:
        st.subheader("Selección Visual de Variables")
        
        # Mostrar información de las variables
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("**Selecciona las variables a mantener:**")
            
            # Calcular el valor por defecto del multiselect
            # Usar directamente variables_to_keep, filtrando solo las que existen en df
            current_selection = sorted([
                v for v in st.session_state.variables_to_keep 
                if v in df.columns
            ])
            
            # Multiselect con todas las variables
            # Sincronizar con session_state si la key existe
            if 'multiselect_vars' not in st.session_state:
                st.session_state.multiselect_vars = current_selection
            
            selected_vars = st.multiselect(
                "Variables disponibles",
                options=sorted(df.columns.tolist()),
                default=current_selection,
                help="Selecciona múltiples variables usando Ctrl/Cmd + Click",
                key="multiselect_vars"
            )
            
            # Actualizar selección
            if st.button("🔄 Actualizar Selección", key="update_visual"):
                st.session_state.variables_to_keep = set(selected_vars)
                st.session_state.variables_to_drop = set(df.columns) - set(selected_vars)
                st.success(f"✅ Selección actualizada: {len(selected_vars)} variables seleccionadas")
                st.rerun()
        
        with col2:
            st.markdown("**Acciones rápidas:**")
            
            if st.button("✅ Seleccionar todas", key="select_all", use_container_width=True):
                all_vars = sorted(df.columns.tolist())
                st.session_state.variables_to_keep = set(all_vars)
                st.session_state.variables_to_drop = set()
                del st.session_state.multiselect_vars  # Eliminar para que se reconstruya
                st.rerun()
            
            if st.button("❌ Deseleccionar todas", key="deselect_all", use_container_width=True):
                st.session_state.variables_to_keep = set()
                st.session_state.variables_to_drop = set(df.columns.tolist())
                del st.session_state.multiselect_vars  # Eliminar para que se reconstruya
                st.rerun()
            
            if st.button("🔄 Invertir selección", key="invert_selection", use_container_width=True):
                old_keep = st.session_state.variables_to_keep.copy()
                st.session_state.variables_to_keep = st.session_state.variables_to_drop.copy()
                st.session_state.variables_to_drop = old_keep
                del st.session_state.multiselect_vars  # Eliminar para que se reconstruya
                st.rerun()
    
    # Tab 2: Búsqueda y filtrado
    with tab2:
        st.subheader("Búsqueda y Filtrado Avanzado")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Búsqueda por nombre
            search_term = st.text_input(
                "🔍 Buscar variables por nombre",
                placeholder="Ejemplo: edad, presion, colesterol...",
                help="Busca variables que contengan el texto ingresado"
            )
            
            if search_term:
                matching_vars = [col for col in df.columns if search_term.lower() in col.lower()]
                st.info(f"📊 Se encontraron {len(matching_vars)} variables que coinciden")
                
                if matching_vars:
                    col_a, col_b = st.columns(2)
                    
                    with col_a:
                        if st.button("✅ Seleccionar coincidencias", key="select_matching"):
                            st.session_state.variables_to_keep.update(matching_vars)
                            st.session_state.variables_to_drop -= set(matching_vars)
                            if 'multiselect_vars' in st.session_state:
                                del st.session_state.multiselect_vars  # Eliminar para que se reconstruya
                            st.success(f"✅ {len(matching_vars)} variables añadidas a la selección")
                            st.rerun()
                    
                    with col_b:
                        if st.button("❌ Descartar coincidencias", key="drop_matching"):
                            st.session_state.variables_to_drop.update(matching_vars)
                            st.session_state.variables_to_keep -= set(matching_vars)
                            if 'multiselect_vars' in st.session_state:
                                del st.session_state.multiselect_vars  # Eliminar para que se reconstruya
                            st.warning(f"🗑️ {len(matching_vars)} variables marcadas para descarte")
                            st.rerun()
                    
                    # Mostrar variables encontradas
                    with st.expander("Ver variables encontradas", expanded=True):
                        for var in matching_vars:
                            status = "✅" if var in st.session_state.variables_to_keep else "❌"
                            st.text(f"{status} {var}")
        
        with col2:
            # Filtro por tipo de dato
            st.markdown("**Filtrar por tipo:**")
            
            var_types = st.multiselect(
                "Tipo de dato",
                ["Numérico", "Categórico", "Datetime", "Booleano"],
                default=[],
                key="type_filter"
            )
            
            if var_types:
                filtered_vars = []
                
                if "Numérico" in var_types:
                    filtered_vars.extend(df.select_dtypes(include=[np.number]).columns.tolist())
                if "Categórico" in var_types:
                    filtered_vars.extend(df.select_dtypes(include=['object', 'category']).columns.tolist())
                if "Datetime" in var_types:
                    filtered_vars.extend(df.select_dtypes(include=['datetime64']).columns.tolist())
                if "Booleano" in var_types:
                    filtered_vars.extend(df.select_dtypes(include=['bool']).columns.tolist())
                
                filtered_vars = list(set(filtered_vars))
                st.info(f"📊 {len(filtered_vars)} variables del tipo seleccionado")
                
                if st.button("✅ Seleccionar por tipo", key="select_by_type", use_container_width=True):
                    st.session_state.variables_to_keep.update(filtered_vars)
                    st.session_state.variables_to_drop -= set(filtered_vars)
                    if 'multiselect_vars' in st.session_state:
                        del st.session_state.multiselect_vars  # Eliminar para que se reconstruya
                    st.rerun()
                
                if st.button("❌ Descartar por tipo", key="drop_by_type", use_container_width=True):
                    st.session_state.variables_to_drop.update(filtered_vars)
                    st.session_state.variables_to_keep -= set(filtered_vars)
                    if 'multiselect_vars' in st.session_state:
                        del st.session_state.multiselect_vars  # Eliminar para que se reconstruya
                    st.rerun()
    
    # Tab 3: Análisis de calidad
    with tab3:
        st.subheader("Análisis de Calidad de Variables")
        
        # Calcular métricas de calidad
        quality_metrics = []
        
        for col in df.columns:
            missing_pct = df[col].isna().sum() / len(df) * 100
            unique_count = df[col].nunique()
            unique_pct = unique_count / len(df) * 100
            dtype = str(df[col].dtype)
            is_selected = col in st.session_state.variables_to_keep
            
            # Para numéricas, calcular varianza
            if pd.api.types.is_numeric_dtype(df[col]):
                variance = df[col].var()
                is_constant = variance == 0 or unique_count == 1
            else:
                variance = None
                is_constant = unique_count == 1
            
            quality_metrics.append({
                'Variable': col,
                'Tipo': dtype,
                'Valores Únicos': unique_count,
                '% Únicos': f"{unique_pct:.1f}",
                '% Faltantes': f"{missing_pct:.1f}",
                'Constante': '⚠️' if is_constant else '✓',
                'Estado': '✅ Seleccionada' if is_selected else '❌ Descartada'
            })
        
        quality_df = pd.DataFrame(quality_metrics)
        
        # Filtros de calidad
        st.markdown("**Filtros de calidad automática:**")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            max_missing = st.slider(
                "% Máximo de faltantes",
                0, 100, 95,
                help="Descartar variables con más de este % de valores faltantes"
            )
        
        with col2:
            include_constants = st.checkbox(
                "Mantener constantes",
                value=False,
                help="Si está desactivado, descarta variables con un único valor"
            )
        
        with col3:
            min_unique_pct = st.slider(
                "% Mínimo de valores únicos",
                0.0, 100.0, 0.0,
                help="Descartar variables con menos de este % de valores únicos"
            )
        
        if st.button("🎯 Aplicar Filtros de Calidad", key="apply_quality_filters", type="primary"):
            vars_filtered = []
            
            for _, row in quality_df.iterrows():
                var_name = row['Variable']
                missing_pct = float(row['% Faltantes'])
                unique_pct = float(row['% Únicos'])
                is_constant = row['Constante'] == '⚠️'
                
                # Aplicar filtros
                keep_var = True
                
                if missing_pct > max_missing:
                    keep_var = False
                
                if is_constant and not include_constants:
                    keep_var = False
                
                if unique_pct < min_unique_pct:
                    keep_var = False
                
                if keep_var:
                    vars_filtered.append(var_name)
            
            st.session_state.variables_to_keep = set(vars_filtered)
            st.session_state.variables_to_drop = set(df.columns) - set(vars_filtered)
            if 'multiselect_vars' in st.session_state:
                del st.session_state.multiselect_vars  # Eliminar para que se reconstruya
            
            st.success(f"✅ Filtros aplicados: {len(vars_filtered)} variables seleccionadas")
            st.rerun()
        
        # Mostrar tabla de calidad
        st.markdown("---")
        st.markdown("**Tabla de Calidad de Variables:**")
        
        # Añadir filtro de vista
        view_filter = st.radio(
            "Mostrar:",
            ["Todas", "Solo Seleccionadas", "Solo Descartadas"],
            horizontal=True,
            key="quality_view_filter"
        )
        
        if view_filter == "Solo Seleccionadas":
            display_df = quality_df[quality_df['Estado'] == '✅ Seleccionada']
        elif view_filter == "Solo Descartadas":
            display_df = quality_df[quality_df['Estado'] == '❌ Descartada']
        else:
            display_df = quality_df
        
        # Mostrar con colores
        st.dataframe(
            display_df,
            use_container_width=True,
            height=400,
            hide_index=True
        )
        
        # Descargar reporte
        csv = display_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Descargar Reporte de Calidad",
            data=csv,
            file_name=f"quality_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    # Tab 4: Aplicar cambios
    with tab4:
        st.subheader("Aplicar Cambios y Continuar")
        
        # Resumen de cambios
        st.markdown("### 📋 Resumen de Cambios")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**✅ Variables Seleccionadas:**")
            if st.session_state.variables_to_keep:
                for var in sorted(st.session_state.variables_to_keep):
                    st.text(f"✓ {var}")
            else:
                st.warning("⚠️ No hay variables seleccionadas")
        
        with col2:
            st.markdown("**❌ Variables a Descartar:**")
            if st.session_state.variables_to_drop:
                for var in sorted(st.session_state.variables_to_drop):
                    st.text(f"✗ {var}")
            else:
                st.info("ℹ️ No se descartarán variables")
        
        st.markdown("---")
        
        # Confirmación y aplicación
        if st.session_state.variables_to_drop:
            st.warning(f"⚠️ Se descartarán **{len(st.session_state.variables_to_drop)}** variables del dataset")
        else:
            st.info("ℹ️ No se realizarán cambios en las variables")
        
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            if st.button("✅ Aplicar y Continuar", type="primary", use_container_width=True):
                if len(st.session_state.variables_to_keep) == 0:
                    st.error("❌ Debes seleccionar al menos una variable")
                else:
                    # ================================================================
                    # IMPORTANTE: Preservar columnas de scores clínicos automáticamente
                    # Estas columnas son necesarias para comparación con GRACE/RECUIMA
                    # Se guardan en session_state por separado, NO en el dataset
                    # ================================================================
                    CLINICAL_SCORE_COLUMNS = [
                        'escala_grace', 'GRACE', 'grace_score',  # GRACE
                        'complicaciones',  # RECUIMA (FV/TV, BAV)
                        'indice_killip',   # RECUIMA (original, no codificado)
                    ]
                    
                    # Encontrar qué columnas de scores existen en el dataframe original
                    # y guardarlas en session_state (caché separado del dataset)
                    preserved_scores = {}
                    for col in CLINICAL_SCORE_COLUMNS:
                        if col in df.columns and col not in st.session_state.variables_to_keep:
                            preserved_scores[col] = df[col].copy()
                    
                    # Guardar en session_state para uso posterior (comparaciones con GRACE, etc.)
                    if preserved_scores:
                        st.session_state.preserved_clinical_scores = preserved_scores
                        st.info(f"ℹ️ Columnas de scores clínicos guardadas en caché (separadas del dataset): {list(preserved_scores.keys())}")
                    
                    # Aplicar los cambios al dataframe (SIN añadir columnas _score_)
                    df_filtered = df[sorted(st.session_state.variables_to_keep)].copy()
                    
                    # Actualizar el session_state correcto según el origen de datos
                    if data_key == "raw_data":
                        st.session_state.raw_data = df_filtered
                    else:
                        st.session_state.cleaned_data = df_filtered
                    
                    st.success(f"✅ Variables aplicadas a {data_source}: {df_filtered.shape[1]} columnas seleccionadas")
                    st.balloons()
                    
                    # Mostrar resultado
                    st.markdown("---")
                    st.markdown("**Vista previa del dataset filtrado:**")
                    
                    col_a, col_b = st.columns(2)
                    col_a.metric("Filas", f"{df_filtered.shape[0]:,}")
                    col_b.metric("Columnas", f"{df_filtered.shape[1]:,}")
                    
                    st.dataframe(df_filtered.head(20), use_container_width=True, height=300)
        
        with col2:
            if st.button("🔄 Restablecer Todo", use_container_width=True):
                st.session_state.variables_to_keep = set(df.columns.tolist())
                st.session_state.variables_to_drop = set()
                st.info("ℹ️ Selección restablecida a todas las variables")
                st.rerun()
        
        with col3:
            if st.button("💾 Guardar Selección", use_container_width=True):
                try:
                    # Guardar configuración de variables
                    selection_config = {
                        'variables_to_keep': sorted(list(st.session_state.variables_to_keep)),
                        'variables_to_drop': sorted(list(st.session_state.variables_to_drop)),
                        'data_source': data_source,
                        'data_key': data_key,
                        'timestamp': datetime.now().isoformat(),
                        'total_variables': len(df.columns),
                        'selected_variables': len(st.session_state.variables_to_keep),
                        'dropped_variables': len(st.session_state.variables_to_drop)
                    }
                    
                    config_dir = Path(CONFIG.preprocessing_config_path).parent
                    config_dir.mkdir(parents=True, exist_ok=True)
                    
                    selection_path = config_dir / "variable_selection.json"
                    
                    with open(selection_path, 'w', encoding='utf-8') as f:
                        json.dump(selection_config, f, indent=2, ensure_ascii=False)
                    
                    st.success(f"✅ Selección guardada en {selection_path}")
                except Exception as e:
                    st.error(f"❌ Error al guardar: {e}")


# =============================================================================
# FUNCIONES FRAGMENT PARA OPTIMIZACIÓN DE RENDIMIENTO
# Estas funciones usan @st.fragment para evitar re-renderizar toda la página
# =============================================================================

@st.fragment
def custom_imputation_fragment(df, numeric_cols, categorical_cols, vars_with_missing, missing_info, missing_pct):
    """Fragment para configuración de imputación personalizada.
    
    Solo muestra variables con valores faltantes e indica el porcentaje de missings.
    Incluye opción de eliminar variables con alto porcentaje de missings.
    """
    st.subheader("Configurar imputación por variable")
    
    # Inicializar diccionarios en session_state
    if 'custom_imputation' not in st.session_state:
        st.session_state.custom_imputation = {}
    if 'custom_constant_values' not in st.session_state:
        st.session_state.custom_constant_values = {}
    if 'columns_to_drop_missing' not in st.session_state:
        st.session_state.columns_to_drop_missing = set()
    
    # Verificar si hay variables con missings
    if not vars_with_missing:
        st.success("✅ ¡No hay valores faltantes en el dataset! No es necesario configurar imputación.")
        return
    
    # Mostrar resumen de missings
    st.info(f"📊 **{len(vars_with_missing)}** variables tienen valores faltantes")
    
    # Separar variables por nivel de missings
    high_missing_threshold = 50.0  # % para considerar alto
    high_missing_vars = [v for v in vars_with_missing if missing_pct[v] >= high_missing_threshold]
    
    if high_missing_vars:
        st.warning(f"⚠️ **{len(high_missing_vars)}** variables tienen ≥{high_missing_threshold}% de valores faltantes. "
                  "Considera eliminarlas si no son críticas.")
    
    # Crear lista de opciones con porcentaje de missings
    var_options_with_pct = {
        f"{'🔴' if missing_pct[var] >= high_missing_threshold else '🟡' if missing_pct[var] >= 20 else '🟢'} {var} ({missing_pct[var]:.1f}% missing - {missing_info[var]} valores)": var 
        for var in vars_with_missing
    }
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        selected_option = st.selectbox(
            "Selecciona variable para configurar",
            [""] + list(var_options_with_pct.keys()),
            key="impute_var_select_frag",
            help="🔴 ≥50% missing | 🟡 ≥20% missing | 🟢 <20% missing"
        )
        var_to_config = var_options_with_pct.get(selected_option, "") if selected_option else ""
    
    with col2:
        if st.button("🗑️ Limpiar todas las configuraciones", key="clear_impute_config_frag"):
            st.session_state.custom_imputation = {}
            st.session_state.custom_constant_values = {}
            st.session_state.columns_to_drop_missing = set()
            st.rerun()
    
    if var_to_config:
        is_numeric = var_to_config in numeric_cols
        var_missing_pct = missing_pct[var_to_config]
        
        # Mostrar info de la variable con indicador visual
        missing_level = "🔴 Alto" if var_missing_pct >= 50 else "🟡 Medio" if var_missing_pct >= 20 else "🟢 Bajo"
        st.markdown(f"""
        **Variable seleccionada:** `{var_to_config}`  
        **Tipo:** {'Numérica' if is_numeric else 'Categórica'}  
        **Valores faltantes:** {missing_info[var_to_config]:,} ({var_missing_pct:.2f}%) - Nivel: {missing_level}
        """)
        
        # Verificar si está marcada para eliminar
        is_marked_for_drop = var_to_config in st.session_state.columns_to_drop_missing
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Opción de eliminar la variable (especialmente útil para alto % de missings)
            drop_var = st.checkbox(
                "🗑️ **Eliminar esta variable** (no imputar)",
                value=is_marked_for_drop,
                key=f"drop_var_{var_to_config}_frag",
                help="Marca esta variable para ser eliminada del dataset durante la limpieza"
            )
            
            if var_missing_pct >= 50 and not drop_var:
                st.caption("💡 *Recomendado: considera eliminar variables con >50% de missings*")
        
        with col2:
            if drop_var:
                st.info("ℹ️ Esta variable será eliminada durante la limpieza")
        
        # Si NO está marcada para eliminar, mostrar opciones de imputación
        if not drop_var:
            st.markdown("---")
            st.markdown("**Configuración de imputación:**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Estrategias según tipo
                if is_numeric:
                    strategies = ["mean", "median", "knn", "forward", "backward", "constant_numeric"]
                    strategy_labels = {
                        "mean": "Media",
                        "median": "Mediana", 
                        "knn": "KNN (vecinos cercanos)",
                        "forward": "Relleno hacia adelante",
                        "backward": "Relleno hacia atrás",
                        "constant_numeric": "Valor constante"
                    }
                else:
                    strategies = ["mode", "forward", "backward", "constant_categorical"]
                    strategy_labels = {
                        "mode": "Moda (valor más frecuente)",
                        "forward": "Relleno hacia adelante",
                        "backward": "Relleno hacia atrás",
                        "constant_categorical": "Valor constante"
                    }
                
                current_strategy = st.session_state.custom_imputation.get(var_to_config, "")
                selected_strategy = st.selectbox(
                    f"Estrategia de imputación",
                    ["(usar global)"] + strategies,
                    index=strategies.index(current_strategy) + 1 if current_strategy in strategies else 0,
                    format_func=lambda x: strategy_labels.get(x, x) if x != "(usar global)" else "🌐 Usar configuración global",
                    key=f"strategy_{var_to_config}_frag"
                )
            
            with col2:
                constant_val = None
                # Valor constante si aplica
                if selected_strategy in ["constant_numeric", "constant_categorical"]:
                    if is_numeric:
                        constant_val = st.number_input(
                            "Valor constante",
                            value=float(st.session_state.custom_constant_values.get(var_to_config, 0.0)),
                            key=f"const_{var_to_config}_frag"
                        )
                    else:
                        constant_val = st.text_input(
                            "Valor constante",
                            value=str(st.session_state.custom_constant_values.get(var_to_config, "missing")),
                            key=f"const_{var_to_config}_frag"
                        )
        
        # Botones de acción
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("✅ Aplicar configuración", key=f"apply_{var_to_config}_frag", type="primary"):
                if drop_var:
                    # Marcar para eliminar
                    st.session_state.columns_to_drop_missing.add(var_to_config)
                    # Remover de imputación si estaba
                    if var_to_config in st.session_state.custom_imputation:
                        del st.session_state.custom_imputation[var_to_config]
                    if var_to_config in st.session_state.custom_constant_values:
                        del st.session_state.custom_constant_values[var_to_config]
                    st.success(f"✅ Variable `{var_to_config}` marcada para eliminación")
                elif selected_strategy != "(usar global)":
                    # Desmarcar de eliminación si estaba
                    st.session_state.columns_to_drop_missing.discard(var_to_config)
                    st.session_state.custom_imputation[var_to_config] = selected_strategy
                    if selected_strategy in ["constant_numeric", "constant_categorical"] and constant_val is not None:
                        st.session_state.custom_constant_values[var_to_config] = constant_val
                    st.success(f"✅ Configuración guardada para {var_to_config}")
                else:
                    # Usar global y desmarcar de eliminación
                    st.session_state.columns_to_drop_missing.discard(var_to_config)
                    if var_to_config in st.session_state.custom_imputation:
                        del st.session_state.custom_imputation[var_to_config]
                    if var_to_config in st.session_state.custom_constant_values:
                        del st.session_state.custom_constant_values[var_to_config]
                    st.info(f"ℹ️ {var_to_config} usará la configuración global")
        
        with col2:
            if st.button("🗑️ Eliminar configuración", key=f"remove_{var_to_config}_frag"):
                st.session_state.columns_to_drop_missing.discard(var_to_config)
                if var_to_config in st.session_state.custom_imputation:
                    del st.session_state.custom_imputation[var_to_config]
                if var_to_config in st.session_state.custom_constant_values:
                    del st.session_state.custom_constant_values[var_to_config]
                st.info(f"ℹ️ Configuración eliminada para {var_to_config}")
    
    # Mostrar configuraciones actuales
    st.markdown("---")
    
    # Mostrar variables marcadas para eliminar
    if st.session_state.columns_to_drop_missing:
        st.markdown("**🗑️ Variables marcadas para ELIMINACIÓN:**")
        drop_data = []
        for var in st.session_state.columns_to_drop_missing:
            if var in missing_pct:
                drop_data.append({
                    'Variable': var,
                    '% Missing': f"{missing_pct[var]:.1f}%",
                    'Acción': '🗑️ Eliminar'
                })
        if drop_data:
            st.dataframe(pd.DataFrame(drop_data), use_container_width=True, hide_index=True)
    
    # Mostrar configuraciones de imputación
    if st.session_state.custom_imputation:
        st.markdown("**📋 Configuraciones de IMPUTACIÓN activas:**")
        config_data = []
        for var, strategy in st.session_state.custom_imputation.items():
            pct = missing_pct.get(var, 0)
            config_data.append({
                'Variable': var,
                '% Missing': f"{pct:.1f}%",
                'Estrategia': strategy,
                'Valor Constante': st.session_state.custom_constant_values.get(var, '-')
            })
        config_df = pd.DataFrame(config_data)
        st.dataframe(config_df, use_container_width=True, hide_index=True)


@st.fragment
def custom_encoding_fragment(df, categorical_cols):
    """Fragment para configuración de codificación personalizada por variable categórica."""
    st.subheader("Configurar codificación por variable")
    
    # Inicializar diccionario en session_state
    if 'custom_encoding' not in st.session_state:
        st.session_state.custom_encoding = {}
    if 'custom_encoding_order' not in st.session_state:
        st.session_state.custom_encoding_order = {}
    
    # Verificar si hay variables categóricas
    if not categorical_cols:
        st.info("ℹ️ No hay variables categóricas en el dataset.")
        return
    
    # Mostrar resumen
    st.info(f"📊 **{len(categorical_cols)}** variables categóricas disponibles para codificación personalizada")
    
    # Crear info de cardinalidad para cada variable
    cardinality_info = {col: df[col].nunique() for col in categorical_cols}
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Crear opciones con cardinalidad
        var_options = {
            f"{col} ({cardinality_info[col]} categorías)": col 
            for col in categorical_cols
        }
        
        selected_option = st.selectbox(
            "Selecciona variable categórica para configurar",
            [""] + list(var_options.keys()),
            key="encoding_var_select_frag",
            help="Configura el tipo de codificación para cada variable"
        )
        var_to_config = var_options.get(selected_option, "") if selected_option else ""
    
    with col2:
        if st.button("🗑️ Limpiar todas las configuraciones", key="clear_encoding_config_frag"):
            st.session_state.custom_encoding = {}
            st.session_state.custom_encoding_order = {}
            st.rerun()
    
    if var_to_config:
        n_categories = cardinality_info[var_to_config]
        categories = sorted(df[var_to_config].dropna().unique().tolist())
        
        # Mostrar info de la variable
        st.markdown(f"""
        **Variable seleccionada:** `{var_to_config}`  
        **Número de categorías:** {n_categories}  
        **Categorías:** {', '.join(str(c) for c in categories[:10])}{'...' if len(categories) > 10 else ''}
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            encoding_options = ["label", "onehot", "ordinal", "none"]
            encoding_labels = {
                "label": "Label Encoding (números enteros)",
                "onehot": "One-Hot Encoding (columnas binarias)",
                "ordinal": "Ordinal Encoding (con orden específico)",
                "none": "Sin codificación"
            }
            
            current_encoding = st.session_state.custom_encoding.get(var_to_config, "")
            selected_encoding = st.selectbox(
                "Tipo de codificación",
                ["(usar global)"] + encoding_options,
                index=encoding_options.index(current_encoding) + 1 if current_encoding in encoding_options else 0,
                format_func=lambda x: encoding_labels.get(x, x) if x != "(usar global)" else "🌐 Usar configuración global",
                key=f"encoding_{var_to_config}_frag"
            )
        
        with col2:
            # Para One-Hot, mostrar advertencia si cardinalidad alta
            if selected_encoding == "onehot" and n_categories > 10:
                st.warning(f"⚠️ Alta cardinalidad ({n_categories}). One-Hot creará {n_categories} columnas nuevas.")
        
        # Si es ordinal, permitir especificar el orden
        if selected_encoding == "ordinal":
            st.markdown("**📋 Especifica el orden de las categorías (de menor a mayor):**")
            
            current_order = st.session_state.custom_encoding_order.get(var_to_config, categories)
            
            # Usar multiselect para ordenar
            ordered_categories = st.multiselect(
                "Arrastra para ordenar (primera = valor más bajo)",
                options=categories,
                default=current_order if set(current_order) == set(categories) else categories,
                key=f"order_{var_to_config}_frag"
            )
            
            if len(ordered_categories) != len(categories):
                st.warning(f"⚠️ Selecciona todas las {len(categories)} categorías en el orden deseado")
        
        # Botones de acción
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("✅ Aplicar configuración", key=f"apply_enc_{var_to_config}_frag", type="primary"):
                if selected_encoding != "(usar global)":
                    st.session_state.custom_encoding[var_to_config] = selected_encoding
                    if selected_encoding == "ordinal" and len(ordered_categories) == len(categories):
                        st.session_state.custom_encoding_order[var_to_config] = ordered_categories
                    st.success(f"✅ Codificación configurada para {var_to_config}")
                else:
                    # Remover configuración personalizada
                    if var_to_config in st.session_state.custom_encoding:
                        del st.session_state.custom_encoding[var_to_config]
                    if var_to_config in st.session_state.custom_encoding_order:
                        del st.session_state.custom_encoding_order[var_to_config]
                    st.info(f"ℹ️ {var_to_config} usará la configuración global")
        
        with col2:
            if st.button("🗑️ Eliminar configuración", key=f"remove_enc_{var_to_config}_frag"):
                if var_to_config in st.session_state.custom_encoding:
                    del st.session_state.custom_encoding[var_to_config]
                if var_to_config in st.session_state.custom_encoding_order:
                    del st.session_state.custom_encoding_order[var_to_config]
                st.info(f"ℹ️ Configuración eliminada para {var_to_config}")
    
    # Mostrar configuraciones actuales
    if st.session_state.custom_encoding:
        st.markdown("---")
        st.markdown("**📋 Configuraciones de codificación activas:**")
        config_data = []
        for var, encoding in st.session_state.custom_encoding.items():
            order = st.session_state.custom_encoding_order.get(var, [])
            config_data.append({
                'Variable': var,
                'Categorías': cardinality_info.get(var, '-'),
                'Codificación': encoding,
                'Orden (si ordinal)': ' → '.join(str(o) for o in order[:5]) + ('...' if len(order) > 5 else '') if order else '-'
            })
        config_df = pd.DataFrame(config_data)
        st.dataframe(config_df, use_container_width=True, hide_index=True)


@st.fragment
def custom_discretization_fragment(df, numeric_cols):
    """Fragment para configuración de discretización personalizada."""
    st.subheader("Configurar discretización por variable")
    
    # Inicializar diccionarios en session_state
    if 'custom_discretization' not in st.session_state:
        st.session_state.custom_discretization = {}
    if 'custom_discretization_bins' not in st.session_state:
        st.session_state.custom_discretization_bins = {}
    
    if not numeric_cols:
        st.info("ℹ️ No hay variables numéricas en el dataset.")
        return
    
    st.info(f"📊 **{len(numeric_cols)}** variables numéricas disponibles para discretización")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        var_to_disc = st.selectbox(
            "Selecciona variable numérica para discretizar",
            [""] + numeric_cols,
            key="disc_var_select_frag"
        )
    
    with col2:
        if st.button("🗑️ Limpiar todas las configuraciones", key="clear_disc_config_frag"):
            st.session_state.custom_discretization = {}
            st.session_state.custom_discretization_bins = {}
            st.rerun()
    
    if var_to_disc:
        # Mostrar estadísticas de la variable
        var_stats = df[var_to_disc].describe()
        st.markdown(f"""
        **Variable:** `{var_to_disc}`  
        **Rango:** [{var_stats['min']:.2f}, {var_stats['max']:.2f}]  
        **Media:** {var_stats['mean']:.2f} | **Mediana:** {var_stats['50%']:.2f}
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            strategies = ["none", "uniform", "quantile", "kmeans"]
            strategy_labels = {
                "none": "Sin discretización",
                "uniform": "Uniforme (intervalos iguales)",
                "quantile": "Cuantiles (frecuencias iguales)",
                "kmeans": "K-Means (clustering)"
            }
            
            current_strategy = st.session_state.custom_discretization.get(var_to_disc, "")
            selected_disc_strategy = st.selectbox(
                "Estrategia de discretización",
                ["(usar global)"] + strategies,
                index=strategies.index(current_strategy) + 1 if current_strategy in strategies else 0,
                format_func=lambda x: strategy_labels.get(x, x) if x != "(usar global)" else "🌐 Usar configuración global",
                key=f"disc_strategy_{var_to_disc}_frag"
            )
        
        with col2:
            n_bins = 5
            if selected_disc_strategy not in ["(usar global)", "none"]:
                n_bins = st.number_input(
                    "Número de bins",
                    min_value=2,
                    max_value=20,
                    value=st.session_state.custom_discretization_bins.get(var_to_disc, 5),
                    key=f"bins_{var_to_disc}_frag"
                )
        
        # Botones de acción
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("✅ Aplicar configuración", key=f"apply_disc_{var_to_disc}_frag", type="primary"):
                if selected_disc_strategy != "(usar global)":
                    st.session_state.custom_discretization[var_to_disc] = selected_disc_strategy
                    if selected_disc_strategy != "none":
                        st.session_state.custom_discretization_bins[var_to_disc] = n_bins
                    st.success(f"✅ Discretización configurada para {var_to_disc}")
                else:
                    if var_to_disc in st.session_state.custom_discretization:
                        del st.session_state.custom_discretization[var_to_disc]
                    if var_to_disc in st.session_state.custom_discretization_bins:
                        del st.session_state.custom_discretization_bins[var_to_disc]
                    st.info(f"ℹ️ {var_to_disc} usará la configuración global")
        
        with col2:
            if st.button("🗑️ Eliminar configuración", key=f"remove_disc_{var_to_disc}_frag"):
                if var_to_disc in st.session_state.custom_discretization:
                    del st.session_state.custom_discretization[var_to_disc]
                if var_to_disc in st.session_state.custom_discretization_bins:
                    del st.session_state.custom_discretization_bins[var_to_disc]
                st.info(f"ℹ️ Configuración eliminada para {var_to_disc}")
    
    # Mostrar configuraciones actuales
    if st.session_state.custom_discretization:
        st.markdown("---")
        st.markdown("**📋 Configuraciones de discretización activas:**")
        config_df = pd.DataFrame([
            {
                'Variable': var,
                'Estrategia': strategy,
                'Bins': st.session_state.custom_discretization_bins.get(var, '-')
            }
            for var, strategy in st.session_state.custom_discretization.items()
        ])
        st.dataframe(config_df, use_container_width=True, hide_index=True)


@st.fragment
def custom_outlier_fragment(df, numeric_cols):
    """Fragment para configuración de outliers personalizada por variable."""
    st.subheader("Configurar tratamiento de outliers por variable")
    
    # Inicializar diccionarios en session_state
    if 'custom_outlier_methods' not in st.session_state:
        st.session_state.custom_outlier_methods = {}
    if 'custom_outlier_treatments' not in st.session_state:
        st.session_state.custom_outlier_treatments = {}
    if 'custom_outlier_params' not in st.session_state:
        st.session_state.custom_outlier_params = {}
    if 'columns_skip_outliers' not in st.session_state:
        st.session_state.columns_skip_outliers = set()
    
    if not numeric_cols:
        st.info("ℹ️ No hay variables numéricas en el dataset.")
        return
    
    st.info(f"📊 **{len(numeric_cols)}** variables numéricas disponibles para tratamiento de outliers")
    
    # Calcular estadísticas de outliers para cada variable (usando IQR como referencia)
    outlier_stats = {}
    for col in numeric_cols:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            n_outliers = ((df[col] < lower) | (df[col] > upper)).sum()
            pct_outliers = (n_outliers / len(df[col].dropna())) * 100 if len(df[col].dropna()) > 0 else 0
            outlier_stats[col] = {
                'n_outliers': n_outliers,
                'pct': pct_outliers,
                'lower': lower,
                'upper': upper,
                'min': df[col].min(),
                'max': df[col].max()
            }
    
    # Crear lista de opciones con info de outliers
    var_options_with_info = {
        f"{'🔴' if outlier_stats.get(var, {}).get('pct', 0) >= 10 else '🟡' if outlier_stats.get(var, {}).get('pct', 0) >= 5 else '🟢'} {var} ({outlier_stats.get(var, {}).get('n_outliers', 0)} outliers - {outlier_stats.get(var, {}).get('pct', 0):.1f}%)": var 
        for var in numeric_cols if var in outlier_stats
    }
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        selected_option = st.selectbox(
            "Selecciona variable para configurar",
            [""] + list(var_options_with_info.keys()),
            key="outlier_var_select_frag",
            help="🔴 ≥10% outliers | 🟡 ≥5% outliers | 🟢 <5% outliers (usando IQR 1.5x)"
        )
        var_to_config = var_options_with_info.get(selected_option, "") if selected_option else ""
    
    with col2:
        if st.button("🗑️ Limpiar todas las configuraciones", key="clear_outlier_config_frag"):
            st.session_state.custom_outlier_methods = {}
            st.session_state.custom_outlier_treatments = {}
            st.session_state.custom_outlier_params = {}
            st.session_state.columns_skip_outliers = set()
            st.rerun()
    
    if var_to_config:
        stats = outlier_stats.get(var_to_config, {})
        
        # Mostrar info de la variable
        outlier_level = "🔴 Alto" if stats.get('pct', 0) >= 10 else "🟡 Medio" if stats.get('pct', 0) >= 5 else "🟢 Bajo"
        st.markdown(f"""
        **Variable:** `{var_to_config}`  
        **Outliers detectados (IQR 1.5x):** {stats.get('n_outliers', 0):,} ({stats.get('pct', 0):.2f}%) - Nivel: {outlier_level}  
        **Rango datos:** [{stats.get('min', 0):.2f}, {stats.get('max', 0):.2f}]  
        **Límites IQR:** [{stats.get('lower', 0):.2f}, {stats.get('upper', 0):.2f}]
        """)
        
        # Verificar si está marcada para omitir
        is_skipped = var_to_config in st.session_state.columns_skip_outliers
        
        col1, col2 = st.columns(2)
        
        with col1:
            skip_var = st.checkbox(
                "⏭️ **Omitir esta variable** (no tratar outliers)",
                value=is_skipped,
                key=f"skip_outlier_{var_to_config}_frag",
                help="Marca esta variable para no aplicar ningún tratamiento de outliers"
            )
        
        with col2:
            if skip_var:
                st.info("ℹ️ No se tratarán outliers en esta variable")
        
        # Si NO está marcada para omitir, mostrar opciones
        if not skip_var:
            st.markdown("---")
            st.markdown("**Configuración de detección y tratamiento:**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                methods = ["iqr", "zscore", "modified_zscore", "isolation_forest", "lof", "percentile"]
                method_labels = {
                    "iqr": "📊 IQR (Rango Intercuartílico)",
                    "zscore": "📈 Z-score",
                    "modified_zscore": "📉 Modified Z-score (MAD)",
                    "isolation_forest": "🌲 Isolation Forest",
                    "lof": "🔍 LOF",
                    "percentile": "📏 Percentil"
                }
                
                current_method = st.session_state.custom_outlier_methods.get(var_to_config, "")
                selected_method = st.selectbox(
                    "Método de detección",
                    ["(usar global)"] + methods,
                    index=methods.index(current_method) + 1 if current_method in methods else 0,
                    format_func=lambda x: method_labels.get(x, x) if x != "(usar global)" else "🌐 Usar configuración global",
                    key=f"outlier_method_{var_to_config}_frag"
                )
            
            with col2:
                treatments = ["cap", "remove", "transform"]
                treatment_labels = {
                    "cap": "🔒 Limitar (Winsorización)",
                    "remove": "🗑️ Eliminar (NaN)",
                    "transform": "🔄 Transformar (log/sqrt)"
                }
                
                current_treatment = st.session_state.custom_outlier_treatments.get(var_to_config, "")
                selected_treatment = st.selectbox(
                    "Tratamiento",
                    ["(usar global)"] + treatments,
                    index=treatments.index(current_treatment) + 1 if current_treatment in treatments else 0,
                    format_func=lambda x: treatment_labels.get(x, x) if x != "(usar global)" else "🌐 Usar configuración global",
                    key=f"outlier_treatment_{var_to_config}_frag"
                )
            
            # Parámetros específicos según el método
            custom_params = st.session_state.custom_outlier_params.get(var_to_config, {})
            new_params = {}
            
            actual_method = selected_method if selected_method != "(usar global)" else "iqr"
            
            if actual_method == "iqr":
                new_params['iqr_multiplier'] = st.slider(
                    "Multiplicador IQR",
                    1.0, 3.0, 
                    custom_params.get('iqr_multiplier', 1.5),
                    0.1,
                    key=f"iqr_mult_{var_to_config}_frag"
                )
            elif actual_method == "zscore":
                new_params['zscore_threshold'] = st.slider(
                    "Umbral Z-score",
                    2.0, 4.0,
                    custom_params.get('zscore_threshold', 3.0),
                    0.1,
                    key=f"zscore_{var_to_config}_frag"
                )
            elif actual_method == "modified_zscore":
                new_params['modified_zscore_threshold'] = st.slider(
                    "Umbral Modified Z-score",
                    2.5, 5.0,
                    custom_params.get('modified_zscore_threshold', 3.5),
                    0.1,
                    key=f"mod_zscore_{var_to_config}_frag"
                )
            elif actual_method in ["isolation_forest", "lof"]:
                new_params['contamination'] = st.slider(
                    "% Contaminación",
                    0.01, 0.3,
                    custom_params.get('contamination', 0.1),
                    0.01,
                    key=f"contam_{var_to_config}_frag"
                )
                if actual_method == "lof":
                    new_params['lof_neighbors'] = st.slider(
                        "Vecinos LOF",
                        5, 50,
                        custom_params.get('lof_neighbors', 20),
                        5,
                        key=f"lof_n_{var_to_config}_frag"
                    )
            elif actual_method == "percentile":
                col_p1, col_p2 = st.columns(2)
                with col_p1:
                    new_params['lower_percentile'] = st.number_input(
                        "Percentil inferior",
                        0.0, 25.0,
                        custom_params.get('lower_percentile', 1.0),
                        0.5,
                        key=f"low_pct_{var_to_config}_frag"
                    )
                with col_p2:
                    new_params['upper_percentile'] = st.number_input(
                        "Percentil superior",
                        75.0, 100.0,
                        custom_params.get('upper_percentile', 99.0),
                        0.5,
                        key=f"up_pct_{var_to_config}_frag"
                    )
        
        # Botones de acción
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("✅ Aplicar configuración", key=f"apply_outlier_{var_to_config}_frag", type="primary"):
                if skip_var:
                    st.session_state.columns_skip_outliers.add(var_to_config)
                    # Limpiar otras configuraciones
                    if var_to_config in st.session_state.custom_outlier_methods:
                        del st.session_state.custom_outlier_methods[var_to_config]
                    if var_to_config in st.session_state.custom_outlier_treatments:
                        del st.session_state.custom_outlier_treatments[var_to_config]
                    if var_to_config in st.session_state.custom_outlier_params:
                        del st.session_state.custom_outlier_params[var_to_config]
                    st.success(f"✅ Variable `{var_to_config}` marcada para omitir tratamiento de outliers")
                else:
                    st.session_state.columns_skip_outliers.discard(var_to_config)
                    if selected_method != "(usar global)":
                        st.session_state.custom_outlier_methods[var_to_config] = selected_method
                        st.session_state.custom_outlier_params[var_to_config] = new_params
                    else:
                        if var_to_config in st.session_state.custom_outlier_methods:
                            del st.session_state.custom_outlier_methods[var_to_config]
                        if var_to_config in st.session_state.custom_outlier_params:
                            del st.session_state.custom_outlier_params[var_to_config]
                    
                    if selected_treatment != "(usar global)":
                        st.session_state.custom_outlier_treatments[var_to_config] = selected_treatment
                    else:
                        if var_to_config in st.session_state.custom_outlier_treatments:
                            del st.session_state.custom_outlier_treatments[var_to_config]
                    
                    st.success(f"✅ Configuración de outliers guardada para {var_to_config}")
        
        with col2:
            if st.button("🗑️ Eliminar configuración", key=f"remove_outlier_{var_to_config}_frag"):
                st.session_state.columns_skip_outliers.discard(var_to_config)
                if var_to_config in st.session_state.custom_outlier_methods:
                    del st.session_state.custom_outlier_methods[var_to_config]
                if var_to_config in st.session_state.custom_outlier_treatments:
                    del st.session_state.custom_outlier_treatments[var_to_config]
                if var_to_config in st.session_state.custom_outlier_params:
                    del st.session_state.custom_outlier_params[var_to_config]
                st.info(f"ℹ️ Configuración eliminada para {var_to_config}")
    
    # Mostrar configuraciones actuales
    st.markdown("---")
    
    # Variables omitidas
    if st.session_state.columns_skip_outliers:
        st.markdown("**⏭️ Variables OMITIDAS (sin tratamiento de outliers):**")
        skip_list = list(st.session_state.columns_skip_outliers)
        st.write(", ".join([f"`{v}`" for v in skip_list]))
    
    # Configuraciones personalizadas
    if st.session_state.custom_outlier_methods or st.session_state.custom_outlier_treatments:
        st.markdown("**📋 Configuraciones de outliers personalizadas:**")
        config_data = []
        all_vars = set(st.session_state.custom_outlier_methods.keys()) | set(st.session_state.custom_outlier_treatments.keys())
        
        for var in all_vars:
            method = st.session_state.custom_outlier_methods.get(var, "(global)")
            treatment = st.session_state.custom_outlier_treatments.get(var, "(global)")
            params = st.session_state.custom_outlier_params.get(var, {})
            params_str = ", ".join([f"{k}={v}" for k, v in params.items()]) if params else "-"
            config_data.append({
                'Variable': var,
                'Método': method,
                'Tratamiento': treatment,
                'Parámetros': params_str
            })
        
        if config_data:
            config_df = pd.DataFrame(config_data)
            st.dataframe(config_df, use_container_width=True, hide_index=True)


def data_cleaning_page():
    """Sección de limpieza de datos."""
    st.header("🧹 Limpieza de Datos")
    
    # Usar datos limpios si existen, sino usar datos crudos
    if st.session_state.raw_data is not None:
        df = st.session_state.raw_data.copy()
        st.info("📊 Usando datos crudos para limpieza")
    elif st.session_state.cleaned_data is not None:
        df = st.session_state.cleaned_data.copy()
        st.info("📊 Usando datos limpios existentes (se pueden re-procesar)")
    else:
        st.warning("⚠️ Primero carga un dataset en la pestaña 'Carga de Datos'")
        return
    
    # CRÍTICO: Aplicar selección de variables ANTES de la limpieza
    if 'variables_to_keep' in st.session_state and len(st.session_state.variables_to_keep) > 0:
        # Filtrar solo las variables seleccionadas
        available_vars = set(df.columns) & st.session_state.variables_to_keep
        if available_vars:
            df = df[sorted(available_vars)].copy()
            st.success(f"✅ Usando {len(available_vars)} variables seleccionadas (de {df.shape[1]} disponibles)")
            
            # Mostrar variables descartadas si hay
            if 'variables_to_drop' in st.session_state and st.session_state.variables_to_drop:
                discarded = st.session_state.variables_to_drop & set(st.session_state.raw_data.columns if st.session_state.raw_data is not None else st.session_state.cleaned_data.columns)
                if discarded:
                    with st.expander(f"🗑️ {len(discarded)} variables descartadas (no se incluirán en la limpieza)", expanded=False):
                        for var in sorted(discarded):
                            st.text(f"✗ {var}")
        else:
            st.warning("⚠️ Ninguna de las variables seleccionadas está disponible en el dataset actual")
    else:
        st.info("ℹ️ No se ha aplicado selección de variables. Se usarán todas las columnas disponibles.")
    
    # Configuración de limpieza en sidebar expandido
    with st.expander("⚙️ Configuración de Limpieza", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Imputación de Valores Faltantes")
            
            numeric_imputation = st.selectbox(
                "Método para numéricas",
                ["none", "mean", "median", "knn", "forward", "backward", "constant"],
                index=0,
                help="Ninguno (preservar originales), media, mediana, KNN, relleno hacia adelante/atrás, constante"
            )
            
            if numeric_imputation == "knn":
                knn_neighbors = st.slider("Vecinos KNN", 1, 20, 5)
            else:
                knn_neighbors = 5
            
            if numeric_imputation == "constant":
                constant_fill_numeric = st.number_input("Valor constante (numérico)", value=0.0)
            else:
                constant_fill_numeric = 0.0
            
            categorical_imputation = st.selectbox(
                "Método para categóricas",
                ["none", "mode", "constant", "forward", "backward"],
                index=0,
                help="Ninguno (preservar originales), moda, constante, relleno hacia adelante/atrás"
            )
            
            if categorical_imputation == "constant":
                constant_fill_categorical = st.text_input("Valor constante (categórico)", "missing")
            else:
                constant_fill_categorical = "missing"
        
        with col2:
            st.subheader("Detección y Tratamiento de Outliers")
            
            outlier_method = st.selectbox(
                "Método de detección",
                ["none", "iqr", "zscore", "modified_zscore", "isolation_forest", "lof", "percentile"],
                index=0,
                format_func=lambda x: {
                    "none": "❌ Ninguno (preservar originales)",
                    "iqr": "📊 IQR (Rango Intercuartílico)",
                    "zscore": "📈 Z-score (Desviación estándar)",
                    "modified_zscore": "📉 Modified Z-score (MAD - robusto)",
                    "isolation_forest": "🌲 Isolation Forest (ML)",
                    "lof": "🔍 LOF (Local Outlier Factor)",
                    "percentile": "📏 Percentil"
                }.get(x, x),
                help="""
                • **Ninguno**: Preserva los datos originales sin modificar
                • **IQR**: Clásico, usa Q1-1.5*IQR y Q3+1.5*IQR
                • **Z-score**: Basado en desviación estándar (sensible a extremos)
                • **Modified Z-score**: Usa MAD, más robusto ante extremos
                • **Isolation Forest**: Algoritmo ML, detecta anomalías complejas
                • **LOF**: Basado en densidad local, bueno para clusters
                • **Percentil**: Simple, usa percentiles configurables
                """
            )
            
            # Parámetros según el método
            iqr_multiplier = 1.5
            zscore_threshold = 3.0
            modified_zscore_threshold = 3.5
            outlier_contamination = 0.1
            lower_percentile = 1.0
            upper_percentile = 99.0
            lof_neighbors = 20
            
            if outlier_method == "iqr":
                iqr_multiplier = st.slider(
                    "Multiplicador IQR", 1.0, 3.0, 1.5, 0.1,
                    help="1.5 = moderado, 3.0 = solo extremos"
                )
            elif outlier_method == "zscore":
                zscore_threshold = st.slider(
                    "Umbral Z-score", 2.0, 4.0, 3.0, 0.1,
                    help="2.0 = sensible, 3.0 = estándar, 4.0 = conservador"
                )
            elif outlier_method == "modified_zscore":
                modified_zscore_threshold = st.slider(
                    "Umbral Modified Z-score", 2.5, 5.0, 3.5, 0.1,
                    help="3.5 es el valor recomendado (Iglewicz & Hoaglin)"
                )
            elif outlier_method in ["isolation_forest", "lof"]:
                outlier_contamination = st.slider(
                    "% Contaminación esperada", 0.01, 0.3, 0.1, 0.01,
                    help="Proporción esperada de outliers en los datos"
                )
                if outlier_method == "lof":
                    lof_neighbors = st.slider(
                        "Vecinos LOF", 5, 50, 20, 5,
                        help="Número de vecinos para calcular densidad local"
                    )
            elif outlier_method == "percentile":
                col_p1, col_p2 = st.columns(2)
                with col_p1:
                    lower_percentile = st.number_input(
                        "Percentil inferior", 0.0, 25.0, 1.0, 0.5,
                        help="Valores por debajo son outliers"
                    )
                with col_p2:
                    upper_percentile = st.number_input(
                        "Percentil superior", 75.0, 100.0, 99.0, 0.5,
                        help="Valores por encima son outliers"
                    )
            
            outlier_treatment = st.selectbox(
                "Tratamiento de outliers",
                ["none", "cap", "remove", "transform"],
                index=0,
                format_func=lambda x: {
                    "none": "❌ Ninguno (preservar originales)",
                    "cap": "🔒 Limitar (Winsorización)",
                    "remove": "🗑️ Eliminar (marcar NaN)",
                    "transform": "🔄 Transformar (log/sqrt)"
                }.get(x, x),
                help="""
                • **Limitar**: Recorta valores a los umbrales
                • **Eliminar**: Convierte outliers a NaN (luego se imputan)
                • **Transformar**: Aplica log1p o sqrt para reducir impacto
                • **Solo detectar**: Marca pero no modifica
                """
            )
        
        col3, col4 = st.columns(2)
        
        with col3:
            st.subheader("Codificación de Categóricas")
            
            categorical_encoding = st.selectbox(
                "Tipo de codificación",
                ["label", "onehot", "ordinal", "none"],
                index=0,
                help="Label encoding, one-hot, ordinal (requiere orden), o ninguno"
            )
        
        with col4:
            st.subheader("Discretización (Opcional)")
            
            discretization_strategy = st.selectbox(
                "Estrategia de discretización",
                ["none", "quantile", "uniform", "custom"],
                index=0,
                help="Ninguna, cuantiles, uniforme, o bins personalizados"
            )
            
            if discretization_strategy in ["quantile", "uniform"]:
                discretization_bins = st.slider("Número de bins", 2, 10, 5)
            else:
                discretization_bins = 5
        
        col5, col6 = st.columns(2)
        
        with col5:
            st.subheader("Opciones Generales")
            drop_duplicates = st.checkbox("Eliminar duplicados", value=True)
            drop_fully_missing = st.checkbox("Eliminar columnas totalmente vacías", value=True)
        
        with col6:
            st.write("")  # Espaciado
            drop_constant = st.checkbox("Eliminar columnas constantes", value=True)
            constant_threshold = st.slider("Umbral constante (%)", 50, 100, 95) / 100
    
    # Configuración personalizada por variable
    st.markdown("---")
    with st.expander("🎯 Configuración Personalizada por Variable (Opcional)", expanded=False):
        st.markdown("""
        Aquí puedes configurar estrategias específicas de imputación, codificación y discretización 
        para variables individuales. Si no se especifica, se usa la configuración global.
        """)
        
        # Identificar columnas numéricas y categóricas
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
        
        # Calcular missings por variable (solo las que tienen)
        missing_info = df.isnull().sum()
        missing_pct = (df.isnull().sum() / len(df) * 100).round(2)
        vars_with_missing = missing_info[missing_info > 0].index.tolist()
        
        # Tabs para imputación, outliers, codificación y discretización
        tab_impute, tab_outliers, tab_encoding, tab_discretize = st.tabs([
            "💉 Imputación Personalizada",
            "📈 Outliers Personalizado", 
            "🏷️ Codificación Personalizada",
            "📊 Discretización Personalizada"
        ])
        
        with tab_impute:
            custom_imputation_fragment(df, numeric_cols, categorical_cols, vars_with_missing, missing_info, missing_pct)
        
        with tab_outliers:
            custom_outlier_fragment(df, numeric_cols)
        
        with tab_encoding:
            custom_encoding_fragment(df, categorical_cols)
        
        with tab_discretize:
            custom_discretization_fragment(df, numeric_cols)
    
    # Crear configuración
    # Combinar columnas a eliminar por missings con las configuradas manualmente
    columns_to_drop_list = list(st.session_state.get('columns_to_drop_missing', set()))
    columns_skip_outliers_list = list(st.session_state.get('columns_skip_outliers', set()))
    
    config = CleaningConfig(
        numeric_imputation=numeric_imputation,
        categorical_imputation=categorical_imputation,
        knn_neighbors=knn_neighbors,
        constant_fill_numeric=constant_fill_numeric,
        constant_fill_categorical=constant_fill_categorical,
        custom_imputation_strategies=st.session_state.get('custom_imputation', {}),
        custom_constant_values=st.session_state.get('custom_constant_values', {}),
        outlier_method=outlier_method,
        iqr_multiplier=iqr_multiplier,
        zscore_threshold=zscore_threshold,
        modified_zscore_threshold=modified_zscore_threshold,
        outlier_contamination=outlier_contamination,
        lower_percentile=lower_percentile,
        upper_percentile=upper_percentile,
        lof_neighbors=lof_neighbors,
        outlier_treatment=outlier_treatment,
        custom_outlier_methods=st.session_state.get('custom_outlier_methods', {}),
        custom_outlier_treatments=st.session_state.get('custom_outlier_treatments', {}),
        custom_outlier_params=st.session_state.get('custom_outlier_params', {}),
        columns_skip_outliers=columns_skip_outliers_list,
        categorical_encoding=categorical_encoding,
        custom_encoding_strategies=st.session_state.get('custom_encoding', {}),
        ordinal_categories=st.session_state.get('custom_encoding_order', {}),
        discretization_strategy=discretization_strategy,
        discretization_bins=discretization_bins,
        custom_discretization_strategies=st.session_state.get('custom_discretization', {}),
        custom_discretization_bins=st.session_state.get('custom_discretization_bins', {}),
        drop_duplicates=drop_duplicates,
        drop_fully_missing=drop_fully_missing,
        drop_constant=drop_constant,
        constant_threshold=constant_threshold,
        columns_to_drop=columns_to_drop_list,
    )
    
    st.session_state.cleaning_config = config
    
    # Botón para aplicar limpieza
    st.markdown("---")
    
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
    
    with col1:
        # Usar valor guardado en session_state si existe
        saved_target = st.session_state.get('target_column_name', '')
        default_idx = 0
        if saved_target and saved_target in df.columns.tolist():
            default_idx = df.columns.tolist().index(saved_target) + 1  # +1 por el "" inicial
        
        target_column = st.selectbox(
            "Columna objetivo (no se modifica)",
            [""] + df.columns.tolist(),
            index=default_idx,
            help="Selecciona la variable objetivo si existe. Esta será la variable a predecir."
        )
        target_column = target_column if target_column else None
        
        # Guardar en session_state para usar en otras páginas
        if target_column:
            st.session_state.target_column_name = target_column
            st.session_state.target_column = target_column  # Para compatibilidad
    
    with col2:
        if st.button("🚀 Aplicar Limpieza", type="primary", use_container_width=True):
            with st.spinner("Limpiando datos..."):
                try:
                    # ================================================================
                    # PRESERVAR columnas de scores clínicos ANTES de la limpieza
                    # Estas columnas son necesarias para comparación con GRACE/RECUIMA
                    # Se guardan en session_state por separado, NO en el dataset
                    # ================================================================
                    CLINICAL_SCORE_COLUMNS = [
                        'escala_grace', 'GRACE', 'grace_score',  # GRACE
                        'complicaciones',  # RECUIMA (FV/TV, BAV)
                    ]
                    
                    # Guardar valores originales de scores antes de cualquier transformación
                    # Se guardan en session_state (caché separado del dataset)
                    preserved_scores = {}
                    for col in CLINICAL_SCORE_COLUMNS:
                        if col in df.columns:
                            preserved_scores[col] = df[col].copy()
                    
                    cleaner = DataCleaner(config)
                    df_clean = cleaner.fit_transform(df, target_column=target_column)
                    
                    # Guardar scores preservados en session_state (alineados con las filas que quedaron)
                    if preserved_scores:
                        aligned_scores = {}
                        for col, original_values in preserved_scores.items():
                            # Obtener valores alineados con las filas que quedaron después de limpieza
                            aligned_scores[col] = original_values.loc[df_clean.index]
                        st.session_state.preserved_clinical_scores = aligned_scores
                        st.info(f"ℹ️ Columnas de scores clínicos guardadas en caché (separadas del dataset): {list(aligned_scores.keys())}")
                    
                    st.session_state.cleaned_data = df_clean
                    st.session_state.cleaner = cleaner
                    
                    st.success("✅ Limpieza completada!")
                    st.balloons()
                    
                except Exception as e:
                    st.error(f"❌ Error durante la limpieza: {e}")
                    import traceback
                    st.code(traceback.format_exc())
    
    with col3:
        if st.button("💾 Guardar Config", use_container_width=True):
            try:
                config_path = Path(CONFIG.preprocessing_config_path)
                config_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(config_path, 'w', encoding='utf-8') as f:
                    json.dump(config.to_dict(), f, indent=2, ensure_ascii=False)
                
                st.success(f"✅ Configuración guardada en {config_path}")
            except Exception as e:
                st.error(f"❌ Error: {e}")
    
    with col4:
        if st.button("📂 Cargar Config", use_container_width=True):
            try:
                config_path = Path(CONFIG.preprocessing_config_path)
                
                if not config_path.exists():
                    st.warning(f"⚠️ No se encontró configuración guardada en {config_path}")
                else:
                    with open(config_path, 'r', encoding='utf-8') as f:
                        loaded_config = json.load(f)
                    
                    # Cargar configuración en session_state
                    # Imputación personalizada
                    if 'custom_imputation_strategies' in loaded_config:
                        st.session_state.custom_imputation = loaded_config['custom_imputation_strategies']
                    if 'custom_constant_values' in loaded_config:
                        st.session_state.custom_constant_values = loaded_config['custom_constant_values']
                    
                    # Columnas a eliminar
                    if 'columns_to_drop' in loaded_config:
                        st.session_state.columns_to_drop_missing = set(loaded_config['columns_to_drop'])
                    
                    # Outliers personalizados
                    if 'custom_outlier_methods' in loaded_config:
                        st.session_state.custom_outlier_methods = loaded_config['custom_outlier_methods']
                    if 'custom_outlier_treatments' in loaded_config:
                        st.session_state.custom_outlier_treatments = loaded_config['custom_outlier_treatments']
                    if 'custom_outlier_params' in loaded_config:
                        st.session_state.custom_outlier_params = loaded_config['custom_outlier_params']
                    if 'columns_skip_outliers' in loaded_config:
                        st.session_state.columns_skip_outliers = set(loaded_config['columns_skip_outliers'])
                    
                    # Codificación personalizada
                    if 'custom_encoding_strategies' in loaded_config:
                        st.session_state.custom_encoding = loaded_config['custom_encoding_strategies']
                    if 'ordinal_categories' in loaded_config:
                        st.session_state.custom_encoding_order = loaded_config['ordinal_categories']
                    
                    # Discretización personalizada
                    if 'custom_discretization_strategies' in loaded_config:
                        st.session_state.custom_discretization = loaded_config['custom_discretization_strategies']
                    if 'custom_discretization_bins' in loaded_config:
                        st.session_state.custom_discretization_bins = loaded_config['custom_discretization_bins']
                    
                    st.success(f"✅ Configuración cargada desde {config_path}")
                    st.info("ℹ️ Recarga la página para ver los valores en los widgets")
                    st.rerun()
                    
            except json.JSONDecodeError as e:
                st.error(f"❌ Error al leer el archivo JSON: {e}")
            except Exception as e:
                st.error(f"❌ Error al cargar configuración: {e}")
    
    # Mostrar resultados
    if st.session_state.cleaned_data is not None:
        st.markdown("---")
        st.subheader("📊 Resultados de Limpieza")
        
        df_clean = st.session_state.cleaned_data
        cleaner = st.session_state.cleaner
        
        # Métricas comparativas
        col1, col2, col3, col4 = st.columns(4)
        
        col1.metric(
            "Filas",
            f"{df_clean.shape[0]:,}",
            f"{df_clean.shape[0] - df.shape[0]:+,}"
        )
        col2.metric(
            "Columnas",
            f"{df_clean.shape[1]:,}",
            f"{df_clean.shape[1] - df.shape[1]:+,}"
        )
        col3.metric(
            "Valores faltantes",
            f"{df_clean.isna().sum().sum():,}",
            f"{df_clean.isna().sum().sum() - df.isna().sum().sum():+,}"
        )
        col4.metric(
            "% Completitud",
            f"{(1 - df_clean.isna().mean().mean()) * 100:.1f}%",
            f"{((1 - df_clean.isna().mean().mean()) - (1 - df.isna().mean().mean())) * 100:+.1f}%"
        )
        
        # Tabs con detalles
        tab1, tab2, tab3 = st.tabs(["Datos Limpios", "Reporte de Calidad", "Metadatos"])
        
        with tab1:
            st.dataframe(df_clean.head(100), width='stretch', height=400)
            
            # Botones de descarga
            col1, col2 = st.columns(2)
            
            with col1:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                csv_data = df_clean.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "📥 Descargar CSV limpio",
                    data=csv_data,
                    file_name=f"cleaned_dataset_{timestamp}.csv",
                    mime="text/csv",
                    width='stretch'
                )
            
            with col2:
                if st.button("💾 Guardar en Cleaned Datasets", width='stretch'):
                    try:
                        # Use absolute path from dashboard config
                        cleaned_dir = CLEANED_DATASETS_DIR
                        cleaned_dir.mkdir(parents=True, exist_ok=True)
                        
                        save_path = cleaned_dir / f"cleaned_dataset_{timestamp}.csv"
                        df_clean.to_csv(save_path, index=False)
                        
                        st.success(f"✅ Guardado en: {save_path}")
                    except Exception as e:
                        st.error(f"❌ Error: {e}")
        
        with tab2:
            if cleaner:
                report = cleaner.get_cleaning_report()
                
                st.subheader("Resumen de Operaciones")
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Variables procesadas", report.get('variables_cleaned', 0))
                col2.metric("Variables con outliers", report.get('variables_with_outliers', 0))
                col3.metric("Variables imputadas", report.get('variables_imputed', 0))
                
                col1, col2 = st.columns(2)
                col1.metric("Variables codificadas", report.get('variables_encoded', 0))
                col2.metric("Variables discretizadas", report.get('variables_discretized', 0))
                
                # Problemas de calidad
                if report['quality_issues']:
                    st.subheader("⚠️ Alertas de Calidad")
                    
                    for var, flags in report['quality_issues'].items():
                        with st.expander(f"Variable: {var}"):
                            for flag in flags:
                                if 'vacia' in flag or 'constante' in flag:
                                    st.error(f"🔴 {flag}")
                                elif 'outliers' in flag or 'missing' in flag:
                                    st.warning(f"🟡 {flag}")
                                else:
                                    st.info(f"🟢 {flag}")
        
        with tab3:
            if cleaner and cleaner.metadata:
                st.subheader("Metadatos de Variables")
                
                # Tabla resumen
                metadata_rows = []
                for name, meta in cleaner.metadata.items():
                    metadata_rows.append({
                        'Variable': name,
                        'Tipo': meta.cleaned_type,
                        'Missing Original (%)': f"{meta.missing_percent_original:.1f}",
                        'Método Imputación': meta.imputation_method or '-',
                        'Codificación': meta.encoding_type or '-',
                        'Outliers Tratados': meta.outliers_treated
                    })
                
                df_meta = pd.DataFrame(metadata_rows)
                st.dataframe(df_meta, width='stretch', height=400)
                
                # Guardar metadatos
                if st.button("💾 Guardar Metadatos como JSON"):
                    try:
                        metadata_path = Path(CONFIG.metadata_path)
                        metadata_path.parent.mkdir(parents=True, exist_ok=True)
                        
                        cleaner.save_metadata(metadata_path)
                        st.success(f"✅ Metadatos guardados en: {metadata_path}")
                    except Exception as e:
                        st.error(f"❌ Error: {e}")


def univariate_analysis_page():
    """Sección de análisis univariado."""
    st.header("📈 Análisis Univariado")
    
    # Decidir qué datos usar
    if st.session_state.cleaned_data is not None:
        df = st.session_state.cleaned_data
        st.info("📊 Usando datos limpios")
    elif st.session_state.raw_data is not None:
        df = st.session_state.raw_data
        st.warning("⚠️ Usando datos crudos (no limpios)")
    else:
        st.warning("⚠️ Carga un dataset primero")
        return
    
    # Crear o recuperar analyzer
    if st.session_state.analyzer is None or st.session_state.analyzer.df.shape != df.shape:
        with st.spinner("Inicializando analizador..."):
            analyzer = EDAAnalyzer(df)
            analyzer.analyze_univariate()
            st.session_state.analyzer = analyzer
    else:
        analyzer = st.session_state.analyzer
    
    # Seleccionar variable
    col1, col2 = st.columns([3, 1])
    
    with col1:
        selected_var = st.selectbox(
            "Selecciona una variable para analizar",
            df.columns.tolist(),
            help="Elige una variable para ver estadísticas y visualizaciones"
        )
    
    with col2:
        if st.button("🔄 Reanalizar", width='stretch'):
            with st.spinner("Analizando..."):
                analyzer.analyze_univariate([selected_var])
                st.success("✅ Análisis actualizado")
    
    if selected_var not in analyzer.univariate_results:
        analyzer.analyze_univariate([selected_var])
    
    stats = analyzer.univariate_results[selected_var]
    
    # Mostrar estadísticas
    st.markdown("---")
    st.subheader(f"Estadísticas de: **{selected_var}**")
    
    if stats.variable_type == 'numerical':
        # Estadísticas numéricas
        col1, col2, col3, col4 = st.columns(4)
        
        col1.metric("Media", f"{stats.mean:.2f}" if stats.mean else "N/A")
        col2.metric("Mediana", f"{stats.median:.2f}" if stats.median else "N/A")
        col3.metric("Desv. Estándar", f"{stats.std:.2f}" if stats.std else "N/A")
        col4.metric("Missing (%)", f"{stats.missing_percent:.1f}%")
        
        col1, col2, col3, col4 = st.columns(4)
        
        col1.metric("Mínimo", f"{stats.min:.2f}" if stats.min else "N/A")
        col2.metric("Q1 (25%)", f"{stats.q25:.2f}" if stats.q25 else "N/A")
        col3.metric("Q3 (75%)", f"{stats.q75:.2f}" if stats.q75 else "N/A")
        col4.metric("Máximo", f"{stats.max:.2f}" if stats.max else "N/A")
        
        col1, col2, col3 = st.columns(3)
        
        col1.metric("Asimetría (Skewness)", f"{stats.skewness:.2f}" if stats.skewness else "N/A")
        col2.metric("Curtosis (Kurtosis)", f"{stats.kurtosis:.2f}" if stats.kurtosis else "N/A")
        col3.metric("Conteo", f"{stats.count:,}" if stats.count else "0")
        
        # Interpretación de skewness
        if stats.skewness is not None:
            if abs(stats.skewness) < 0.5:
                skew_msg = "✅ Distribución aproximadamente simétrica"
                skew_color = "green"
            elif abs(stats.skewness) < 1:
                skew_msg = "⚠️ Distribución moderadamente asimétrica"
                skew_color = "orange"
            else:
                skew_msg = "🔴 Distribución altamente asimétrica"
                skew_color = "red"
            
            st.markdown(f":{skew_color}[{skew_msg}]")
        
        # Visualizaciones
        st.markdown("---")
        st.subheader("Visualizaciones")
        
        tab1, tab2, tab3 = st.tabs(["Histograma + KDE", "Boxplot", "Violin Plot"])
        
        with tab1:
            fig = analyzer.plot_distribution(selected_var, plot_type='histogram')
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            fig = analyzer.plot_distribution(selected_var, plot_type='box')
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            fig = analyzer.plot_distribution(selected_var, plot_type='violin')
            st.plotly_chart(fig, use_container_width=True)
    
    else:
        # Estadísticas categóricas
        col1, col2, col3, col4 = st.columns(4)
        
        col1.metric("Categorías únicas", f"{stats.n_categories:,}" if stats.n_categories else "0")
        col2.metric("Moda", str(stats.mode) if stats.mode else "N/A")
        col3.metric("Frecuencia moda", f"{stats.mode_frequency:,}" if stats.mode_frequency else "0")
        col4.metric("Missing (%)", f"{stats.missing_percent:.1f}%")
        
        # Tabla de frecuencias
        st.markdown("---")
        st.subheader("Tabla de Frecuencias")
        
        if stats.category_counts:
            freq_df = pd.DataFrame([
                {'Categoría': k, 'Frecuencia': v, 'Porcentaje': f"{v/stats.count*100:.1f}%"}
                for k, v in sorted(stats.category_counts.items(), key=lambda x: x[1], reverse=True)
            ])
            st.dataframe(freq_df, width='stretch', height=300)
        
        # Visualizaciones
        st.markdown("---")
        st.subheader("Visualizaciones")
        
        tab1, tab2 = st.tabs(["Gráfico de Barras", "Gráfico Circular"])
        
        with tab1:
            fig = analyzer.plot_distribution(selected_var, plot_type='bar')
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            fig = analyzer.plot_distribution(selected_var, plot_type='pie')
            st.plotly_chart(fig, use_container_width=True)
    
    # Exportación PDF
    st.markdown("---")
    
    def generate_univariate_report():
        """Generate univariate analysis PDF report."""
        from pathlib import Path
        output_path = Path("reports") / "univariate_eda.pdf"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        return generate_univariate_pdf(
            df=df,
            numerical_cols=analyzer.numeric_cols,
            categorical_cols=analyzer.categorical_cols,
            output_path=output_path
        )
    
    pdf_export_section(
        generate_univariate_report,
        section_title="Análisis Univariado",
        default_filename="univariate_eda.pdf",
        key_prefix="univariate_eda"
    )


def bivariate_analysis_page():
    """Sección de análisis bivariado."""
    st.header("📊 Análisis Bivariado")
    
    # Decidir qué datos usar
    if st.session_state.cleaned_data is not None:
        df = st.session_state.cleaned_data
        st.info("📊 Usando datos limpios")
    elif st.session_state.raw_data is not None:
        df = st.session_state.raw_data
        st.warning("⚠️ Usando datos crudos (no limpios)")
    else:
        st.warning("⚠️ Carga un dataset primero")
        return
    
    # Crear o recuperar analyzer
    if st.session_state.analyzer is None or st.session_state.analyzer.df.shape != df.shape:
        with st.spinner("Inicializando analizador..."):
            analyzer = EDAAnalyzer(df)
            st.session_state.analyzer = analyzer
    else:
        analyzer = st.session_state.analyzer
        # Forzar recreación si bivariate_results es una lista (versión antigua)
        if isinstance(analyzer.bivariate_results, list):
            with st.spinner("Actualizando analizador..."):
                analyzer = EDAAnalyzer(df)
                st.session_state.analyzer = analyzer
    
    # Seleccionar variables
    col1, col2 = st.columns(2)
    
    with col1:
        var1 = st.selectbox(
            "Variable 1",
            df.columns.tolist(),
            help="Primera variable para análisis"
        )
    
    with col2:
        var2 = st.selectbox(
            "Variable 2",
            [col for col in df.columns if col != var1],
            help="Segunda variable para análisis"
        )
    
    if var1 == var2:
        st.warning("⚠️ Selecciona variables diferentes")
        return
    
    # Analizar y mostrar resultados
    if st.button("🔍 Analizar Relación", type="primary"):
        with st.spinner("Analizando relación bivariada..."):
            result = analyzer.analyze_bivariate(var1, var2)
            
            # Mostrar resultados inmediatamente después del análisis
            key = f"{var1}_vs_{var2}"
            if key in analyzer.bivariate_results:
                result = analyzer.bivariate_results[key]
                
                st.markdown("---")
                st.subheader(f"Relación: **{var1}** vs **{var2}**")
                
                # Mostrar métricas según tipo de relación
                if result.relationship_type == "num-num":
                    col1, col2, col3 = st.columns(3)
                    
                    col1.metric("Correlación de Pearson", f"{result.pearson_corr:.3f}" if result.pearson_corr else "N/A")
                    col2.metric("Correlación de Spearman", f"{result.spearman_corr:.3f}" if result.spearman_corr else "N/A")
                    col3.metric("p-value", f"{result.pearson_pvalue:.4f}" if result.pearson_pvalue else "N/A")
                    
                    # Interpretación
                    if result.pearson_corr and abs(result.pearson_corr) > 0.7:
                        st.success("✅ Correlación fuerte detectada")
                    elif result.pearson_corr and abs(result.pearson_corr) > 0.4:
                        st.info("📊 Correlación moderada")
                    else:
                        st.warning("⚠️ Correlación débil o no significativa")
                    
                    # Visualización
                    try:
                        fig = analyzer.plot_scatter(var1, var2, add_trendline=True)
                    except (ImportError, ModuleNotFoundError):
                        # Si statsmodels no está instalado, mostrar sin trendline
                        fig = analyzer.plot_scatter(var1, var2, add_trendline=False)
                    st.plotly_chart(fig, use_container_width=True)
                
                elif result.relationship_type == "cat-cat":
                    col1, col2, col3 = st.columns(3)
                    
                    col1.metric("Chi-cuadrado", f"{result.chi2_statistic:.2f}" if result.chi2_statistic else "N/A")
                    col2.metric("p-value", f"{result.chi2_pvalue:.4f}" if result.chi2_pvalue else "N/A")
                    col3.metric("Cramér's V", f"{result.cramers_v:.3f}" if result.cramers_v else "N/A")
                    
                    if result.chi2_pvalue and result.chi2_pvalue < 0.05:
                        st.success("✅ Relación estadísticamente significativa (p < 0.05)")
                    else:
                        st.info("📊 No hay evidencia de relación significativa")
                    
                    # Tabla de contingencia
                    st.subheader("Tabla de Contingencia")
                    contingency = pd.crosstab(df[var1], df[var2])
                    st.dataframe(contingency, width='stretch')
                
                else:  # num-cat (ANOVA)
                    col1, col2 = st.columns(2)
                    
                    col1.metric("F-statistic (ANOVA)", f"{result.anova_f:.3f}" if result.anova_f else "N/A")
                    col2.metric("p-value", f"{result.anova_pvalue:.4f}" if result.anova_pvalue else "N/A")
                    
                    if result.anova_pvalue and result.anova_pvalue < 0.05:
                        st.success("✅ Diferencias significativas entre grupos (p < 0.05)")
                    else:
                        st.info("📊 No hay evidencia de diferencias significativas")
                    
                    # Estadísticas por grupo
                    st.subheader("Estadísticas por Grupo")
                    # Determinar cuál es numérica y cuál categórica
                    num_var = var1 if var1 in analyzer.numeric_cols else var2
                    cat_var = var2 if var1 in analyzer.numeric_cols else var1
                    group_stats = df.groupby(cat_var)[num_var].describe()
                    st.dataframe(group_stats, width='stretch')
    
    # Exportación PDF
    st.markdown("---")
    if analyzer.bivariate_results:
        
        def generate_bivariate_report():
            """Generate bivariate analysis PDF report."""
            from pathlib import Path
            output_path = Path("reports") / "bivariate_eda.pdf"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            return generate_bivariate_pdf(
                df=df,
                numerical_cols=analyzer.numeric_cols,
                output_path=output_path,
                correlation_threshold=0.3
            )
        
        pdf_export_section(
            generate_bivariate_report,
            section_title="Análisis Bivariado",
            default_filename="bivariate_eda.pdf",
            key_prefix="bivariate_eda"
        )


def multivariate_analysis_page():
    """Sección de análisis multivariado."""
    st.header("🔬 Análisis Multivariado")
    
    # Decidir qué datos usar
    if st.session_state.cleaned_data is not None:
        df = st.session_state.cleaned_data
        st.info("📊 Usando datos limpios")
    elif st.session_state.raw_data is not None:
        df = st.session_state.raw_data
        st.warning("⚠️ Usando datos crudos (no limpios)")
    else:
        st.warning("⚠️ Carga un dataset primero")
        return
    
    # Crear o recuperar analyzer
    if st.session_state.analyzer is None or st.session_state.analyzer.df.shape != df.shape:
        with st.spinner("Inicializando analizador..."):
            analyzer = EDAAnalyzer(df)
            st.session_state.analyzer = analyzer
    else:
        analyzer = st.session_state.analyzer
    
    # Tabs para diferentes análisis
    tabs = st.tabs([
        "📊 Matriz de Correlación", 
        "🎯 PCA (Análisis de Componentes Principales)",
        "🧬 ICA (Análisis de Componentes Independientes)"
    ])
    
    with tabs[0]:
        st.subheader("Matriz de Correlación")
        
        if len(analyzer.numeric_cols) < 2:
            st.warning("⚠️ Se necesitan al menos 2 variables numéricas para análisis de correlación")
        else:
            col1, col2 = st.columns(2)
            
            with col1:
                corr_method = st.selectbox(
                    "Método de correlación",
                    ["pearson", "spearman", "kendall"],
                    help="Pearson: lineal, Spearman: monotónica, Kendall: ordinal"
                )
            
            with col2:
                min_corr = st.slider(
                    "Filtrar correlaciones < ",
                    0.0, 0.9, 0.0, 0.05,
                    help="Mostrar solo correlaciones mayores a este umbral"
                )
            
            if st.button("📊 Calcular Matriz de Correlación", type="primary"):
                with st.spinner("Calculando correlaciones..."):
                    try:
                        # Calcular matriz de correlación usando el analyzer
                        corr_matrix = analyzer.analyze_multivariate(method=corr_method)
                        
                        # Visualización
                        fig = analyzer.plot_correlation_matrix(method=corr_method)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Tabla de correlaciones más altas
                        st.subheader("Top Correlaciones")
                        
                        # Extraer pares de correlación
                        corr_pairs = []
                        for i in range(len(corr_matrix.columns)):
                            for j in range(i + 1, len(corr_matrix.columns)):
                                var1 = corr_matrix.columns[i]
                                var2 = corr_matrix.columns[j]
                                corr_val = corr_matrix.iloc[i, j]
                                
                                if not pd.isna(corr_val) and abs(corr_val) >= min_corr:
                                    corr_pairs.append({
                                        'Variable 1': var1,
                                        'Variable 2': var2,
                                        'Correlación': corr_val,
                                        'Correlación Abs': abs(corr_val)
                                    })
                        
                        if corr_pairs:
                            corr_df = pd.DataFrame(corr_pairs)
                            corr_df = corr_df.sort_values('Correlación Abs', ascending=False)
                            corr_df = corr_df.drop('Correlación Abs', axis=1)
                            
                            st.dataframe(corr_df.head(20), width='stretch')
                        else:
                            st.info("No se encontraron correlaciones significativas con el filtro aplicado")
                        
                    except Exception as e:
                        st.error(f"❌ Error: {e}")
    
    with tabs[1]:
        st.subheader("Análisis de Componentes Principales (PCA)")
        
        if len(analyzer.numeric_cols) < 2:
            st.warning("⚠️ Se necesitan al menos 2 variables numéricas para PCA")
        else:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                pca_mode = st.radio(
                    "Modo de selección de componentes",
                    ["Automático (por varianza)", "Manual"],
                    help="Automático: selecciona componentes hasta alcanzar umbral de varianza"
                )
            
            with col2:
                if pca_mode == "Automático (por varianza)":
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
                        2, min(20, len(analyzer.numeric_cols)), 5
                    )
                    variance_threshold = 0.95
            
            with col3:
                scale_data = st.checkbox(
                    "Estandarizar datos",
                    value=True,
                    help="Recomendado cuando variables tienen diferentes escalas"
                )
            
            if st.button("🚀 Ejecutar PCA", type="primary"):
                with st.spinner("Ejecutando Análisis de Componentes Principales..."):
                    try:
                        # Validar que hay suficientes variables numéricas
                        if len(analyzer.numeric_cols) < 2:
                            st.error("❌ Se requieren al menos 2 variables numéricas para PCA")
                            st.stop()
                        
                        # Verificar cuántas filas completas hay
                        df_for_pca = df[analyzer.numeric_cols].dropna()
                        if len(df_for_pca) == 0:
                            st.error(
                                "❌ **No se puede ejecutar PCA: Todas las filas tienen valores faltantes**\n\n"
                                "El Análisis de Componentes Principales requiere datos completos en las variables numéricas."
                            )
                            
                            st.info(
                                "**📋 Solución recomendada:**\n\n"
                                "1. Ve a la sección **'🧹 Limpieza de Datos'** (arriba en esta misma página)\n"
                                "2. Configura la **imputación de valores faltantes** para las variables numéricas\n"
                                "3. Haz clic en **'🚀 Aplicar Limpieza'**\n"
                                "4. Regresa a esta sección para ejecutar PCA con los datos limpios"
                            )
                            
                            # Mostrar información sobre valores faltantes
                            missing_info = df[analyzer.numeric_cols].isnull().sum()
                            missing_pct = (missing_info / len(df) * 100).round(2)
                            missing_df = pd.DataFrame({
                                'Variable': missing_info.index,
                                'Valores Faltantes': missing_info.values,
                                'Porcentaje (%)': missing_pct.values
                            })
                            missing_df = missing_df[missing_df['Valores Faltantes'] > 0].sort_values(
                                'Valores Faltantes', ascending=False
                            )
                            
                            if len(missing_df) > 0:
                                st.warning("**⚠️ Variables numéricas con valores faltantes:**")
                                st.dataframe(missing_df, width='stretch', height=300)
                            
                            st.stop()
                        
                        if len(df_for_pca) < 2:
                            st.warning(
                                f"⚠️ **Datos insuficientes para PCA**\n\n"
                                f"Solo hay {len(df_for_pca)} fila(s) sin valores faltantes. "
                                "Se requieren al menos 2 observaciones completas.\n\n"
                                "**Solución:** Aplica imputación de valores faltantes para aumentar el número de filas válidas."
                            )
                            st.stop()
                        
                        pca_results = analyzer.perform_pca(
                            n_components=n_components,
                            variance_threshold=variance_threshold,
                            scale=scale_data
                        )
                        
                        st.success(f"✅ PCA completado: {pca_results.n_components} componentes")
                        
                        # Métricas
                        st.markdown("---")
                        st.subheader("Resultados de PCA")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        col1.metric(
                            "Componentes Principales",
                            pca_results.n_components
                        )
                        col2.metric(
                            "Varianza Explicada",
                            f"{sum(pca_results.explained_variance_ratio) * 100:.1f}%"
                        )
                        col3.metric(
                            "Variables Originales",
                            len(pca_results.feature_names)
                        )
                        
                        # Tabla de varianza por componente
                        st.subheader("Varianza Explicada por Componente")
                        
                        variance_df = pd.DataFrame({
                            'Componente': [f'PC{i+1}' for i in range(pca_results.n_components)],
                            'Varianza Individual (%)': [v * 100 for v in pca_results.explained_variance_ratio],
                            'Varianza Acumulada (%)': [v * 100 for v in pca_results.cumulative_variance]
                        })
                        
                        st.dataframe(variance_df, width='stretch')
                        
                        # Visualizaciones
                        st.markdown("---")
                        
                        tab1, tab2, tab3 = st.tabs([
                            "Gráfico de Scree",
                            "Biplot",
                            "Importancia de Features"
                        ])
                        
                        with tab1:
                            st.subheader("Gráfico de Scree (Varianza Explicada)")
                            fig = analyzer.plot_pca_scree()
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with tab2:
                            st.subheader("Biplot de PCA")
                            
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                pc_x = st.selectbox("PC para eje X", range(1, pca_results.n_components + 1), index=0)
                            
                            with col2:
                                pc_y = st.selectbox("PC para eje Y", range(1, pca_results.n_components + 1), index=min(1, pca_results.n_components - 1))
                            
                            with col3:
                                n_features_show = st.slider("Features a mostrar", 5, 20, 10)
                            
                            if pc_x != pc_y:
                                fig = analyzer.plot_pca_biplot(pc_x=pc_x, pc_y=pc_y, n_features=n_features_show)
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.warning("⚠️ Selecciona diferentes componentes para X e Y")
                        
                        with tab3:
                            st.subheader("Importancia de Features en Primeros Componentes")
                            
                            n_comp_importance = st.slider(
                                "Componentes a considerar",
                                1, min(5, pca_results.n_components), 
                                min(3, pca_results.n_components)
                            )
                            
                            importance_df = analyzer.get_feature_importance_pca(n_components=n_comp_importance)
                            
                            st.dataframe(importance_df, width='stretch', height=400)
                            
                            # Gráfico de barras
                            import plotly.express as px
                            fig = px.bar(
                                importance_df.head(20).reset_index(),
                                x='importance',
                                y='index',
                                orientation='h',
                                title='Top 20 Features por Importancia',
                                labels={'index': 'Feature', 'importance': 'Importancia'}
                            )
                            fig.update_layout(yaxis={'categoryorder':'total ascending'})
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Opción de guardar resultados transformados
                        st.markdown("---")
                        if st.button("💾 Guardar Datos Transformados (PCA)"):
                            try:
                                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                # Use absolute path from dashboard config
                                pca_data_path = CLEANED_DATASETS_DIR / f"pca_transformed_{timestamp}.csv"
                                pca_data_path.parent.mkdir(parents=True, exist_ok=True)
                                
                                pca_results.transformed_data.to_csv(pca_data_path, index=False)
                                st.success(f"✅ Datos PCA guardados en: {pca_data_path}")
                            except Exception as e:
                                st.error(f"❌ Error: {e}")
                        
                    except ValueError as ve:
                        # Errores específicos de validación
                        st.error(f"❌ Error de validación: {ve}")
                        st.info(
                            "💡 **Sugerencias:**\n"
                            "- Asegúrate de haber aplicado limpieza de datos primero\n"
                            "- Verifica que las variables numéricas seleccionadas tengan datos válidos\n"
                            "- Aplica imputación de valores faltantes si es necesario"
                        )
                    except Exception as e:
                        st.error(f"❌ Error durante PCA: {e}")
                        with st.expander("Ver detalles del error"):
                            import traceback
                            st.code(traceback.format_exc())
    
    # ==================== TAB 3: ICA ====================
    with tabs[2]:
        st.subheader("Análisis de Componentes Independientes (ICA)")
        
        # Explicación de ICA vs PCA
        with st.expander("ℹ️ ¿Qué es ICA y cuándo usarlo?"):
            st.markdown("""
            **ICA (Independent Component Analysis)** busca componentes **estadísticamente independientes**, 
            no solo no correlacionados como PCA.
            
            **Diferencias clave con PCA:**
            - **PCA:** Busca componentes ortogonales que maximizan la varianza (datos Gaussianos)
            - **ICA:** Busca componentes independientes que maximizan la no-Gaussianidad (datos no-Gaussianos)
            
            **Cuándo usar ICA:**
            - ✅ Datos no-Gaussianos con múltiples fuentes mezcladas (ej: señales biomédicas)
            - ✅ Cuando se busca separación de fuentes (blind source separation)
            - ✅ Cuando la independencia estadística es más importante que la varianza
            
            **Cuándo usar PCA:**
            - ✅ Reducción de dimensionalidad general
            - ✅ Datos aproximadamente Gaussianos
            - ✅ Cuando la varianza es el criterio principal
            
            **Métrica clave: Kurtosis** (medida de no-Gaussianidad)
            - Kurtosis = 0: Distribución Gaussiana
            - Kurtosis > 0: Distribución leptocúrtica (colas pesadas)
            - Kurtosis < 0: Distribución platicúrtica (colas ligeras)
            """)
        
        if len(analyzer.numeric_cols) < 2:
            st.warning("⚠️ Se necesitan al menos 2 variables numéricas para ICA")
        else:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                n_components_ica = st.slider(
                    "Número de componentes",
                    2, min(20, len(analyzer.numeric_cols)), 
                    min(5, len(analyzer.numeric_cols)),
                    help="Número de componentes independientes a extraer"
                )
            
            with col2:
                ica_algorithm = st.selectbox(
                    "Algoritmo ICA",
                    ["parallel", "deflation"],
                    help="parallel: extrae todos simultáneamente, deflation: uno por uno"
                )
            
            with col3:
                ica_fun = st.selectbox(
                    "Función de contraste",
                    ["logcosh", "exp", "cube"],
                    help="logcosh: general, exp: super-Gaussiano, cube: sub-Gaussiano"
                )
            
            col1, col2 = st.columns(2)
            
            with col1:
                whiten_ica = st.checkbox(
                    "Blanqueamiento (whitening)",
                    value=True,
                    help="Pre-procesa datos para ortogonalizar (recomendado)"
                )
            
            with col2:
                max_iter_ica = st.number_input(
                    "Iteraciones máximas",
                    200, 1000, 500, 50,
                    help="Más iteraciones = mejor convergencia pero más lento"
                )
            
            if st.button("🚀 Ejecutar ICA", type="primary"):
                with st.spinner("Ejecutando Análisis de Componentes Independientes..."):
                    try:
                        # Validar datos
                        if len(analyzer.numeric_cols) < 2:
                            st.error("❌ Se requieren al menos 2 variables numéricas para ICA")
                            st.stop()
                        
                        df_for_ica = df[analyzer.numeric_cols].dropna()
                        
                        if len(df_for_ica) == 0:
                            st.error(
                                "❌ **No se puede ejecutar ICA: Todas las filas tienen valores faltantes**\n\n"
                                "Solución: Ve a '🧹 Limpieza de Datos' y aplica imputación."
                            )
                            st.stop()
                        
                        if len(df_for_ica) < 2:
                            st.warning(
                                f"⚠️ Datos insuficientes: solo {len(df_for_ica)} fila(s) completa(s). "
                                "Se requieren al menos 2."
                            )
                            st.stop()
                        
                        # Crear y ajustar ICA
                        # Convert boolean whiten to string for newer sklearn versions
                        whiten_param = 'unit-variance' if whiten_ica else False
                        
                        ica = ICATransformer(
                            n_components=n_components_ica,
                            algorithm=ica_algorithm,
                            fun=ica_fun,
                            whiten=whiten_param,
                            max_iter=max_iter_ica,
                            random_state=42
                        )
                        
                        ica.fit(df_for_ica)
                        transformed_data = ica.transform(df_for_ica)
                        
                        # Guardar en session_state para comparación posterior
                        st.session_state.ica_transformer = ica
                        st.session_state.ica_data = transformed_data
                        
                        st.success(f"✅ ICA completado: {n_components_ica} componentes independientes extraídos")
                        
                        # ==================== MÉTRICAS ====================
                        st.markdown("---")
                        st.subheader("📊 Resultados de ICA")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        col1.metric(
                            "Componentes Independientes",
                            ica.result.n_components
                        )
                        
                        # Varianza explicada promedio
                        avg_variance = np.mean(ica.result.variance_per_component) * 100
                        col2.metric(
                            "Varianza Promedio/Comp",
                            f"{avg_variance:.1f}%"
                        )
                        
                        # Kurtosis promedio (medida de no-Gaussianidad)
                        avg_kurtosis = np.mean(np.abs(ica.result.kurtosis))
                        col3.metric(
                            "Kurtosis Promedio (abs)",
                            f"{avg_kurtosis:.2f}",
                            help="Mayor kurtosis = mayor no-Gaussianidad (mejor para ICA)"
                        )
                        
                        col4.metric(
                            "Variables Originales",
                            len(ica.result.feature_names)
                        )
                        
                        # ==================== VISUALIZACIONES ====================
                        st.markdown("---")
                        
                        tab1, tab2, tab3, tab4, tab5 = st.tabs([
                            "📈 Kurtosis",
                            "🔥 Matriz de Mezcla",
                            "📊 Distribución Componentes",
                            "📉 Varianza Explicada",
                            "⚖️ Comparación PCA vs ICA"
                        ])
                        
                        with tab1:
                            st.subheader("Kurtosis de Componentes Independientes")
                            st.markdown("""
                            **Kurtosis** mide la no-Gaussianidad de cada componente:
                            - **Kurtosis ≈ 0:** Distribución Gaussiana (normal)
                            - **Kurtosis > 0:** Leptocúrtica (colas pesadas, picos altos)
                            - **Kurtosis < 0:** Platicúrtica (colas ligeras, picos bajos)
                            
                            ICA **maximiza la kurtosis** para encontrar fuentes independientes.
                            """)
                            
                            fig_kurtosis = ica.plot_kurtosis()
                            st.plotly_chart(fig_kurtosis, use_container_width=True)
                            
                            # Tabla de kurtosis
                            kurtosis_df = pd.DataFrame({
                                'Componente': [f'IC{i+1}' for i in range(len(ica.result.kurtosis))],
                                'Kurtosis': ica.result.kurtosis,
                                'Kurtosis (abs)': np.abs(ica.result.kurtosis)
                            })
                            kurtosis_df = kurtosis_df.sort_values('Kurtosis (abs)', ascending=False)
                            
                            st.dataframe(kurtosis_df, width='stretch')
                            
                            st.info(
                                f"💡 **Interpretación:** Componente con mayor kurtosis (abs): "
                                f"**{kurtosis_df.iloc[0]['Componente']}** con kurtosis = "
                                f"{kurtosis_df.iloc[0]['Kurtosis']:.3f} → "
                                f"{'Distribución leptocúrtica (colas pesadas)' if kurtosis_df.iloc[0]['Kurtosis'] > 0 else 'Distribución platicúrtica (colas ligeras)'}"
                            )
                        
                        with tab2:
                            st.subheader("Matriz de Mezcla (Mixing Matrix)")
                            st.markdown("""
                            Muestra **cómo cada componente independiente se mezcla** para formar las variables originales.
                            
                            - **Filas:** Variables originales
                            - **Columnas:** Componentes independientes
                            - **Valores:** Peso de cada IC en cada variable (colores intensos = mayor influencia)
                            """)
                            
                            fig_mixing = ica.plot_mixing_matrix()
                            st.plotly_chart(fig_mixing, use_container_width=True)
                            
                            # Mostrar matriz como tabla
                            with st.expander("Ver Matriz de Mezcla (valores numéricos)"):
                                mixing_df = pd.DataFrame(
                                    ica.result.mixing_matrix,
                                    columns=[f'IC{i+1}' for i in range(ica.result.n_components)],
                                    index=ica.result.feature_names
                                )
                                st.dataframe(mixing_df.style.format("{:.4f}"), width='stretch')
                        
                        with tab3:
                            st.subheader("Distribución de Componentes Independientes")
                            st.markdown("""
                            Histogramas de los primeros componentes independientes.
                            ICA busca que estas distribuciones sean **lo más no-Gaussianas posible**.
                            """)
                            
                            n_show = min(6, n_components_ica)
                            fig_dist = ica.plot_components_distribution(n_components=n_show)
                            st.plotly_chart(fig_dist, use_container_width=True)
                        
                        with tab4:
                            st.subheader("Varianza Explicada por Componente")
                            st.markdown("""
                            ⚠️ **Nota:** La varianza **NO es el objetivo principal de ICA** 
                            (eso es de PCA). ICA busca **independencia estadística**, no varianza máxima.
                            
                            Sin embargo, es útil ver cuánta varianza captura cada componente.
                            """)
                            
                            fig_variance = ica.plot_variance_explained()
                            st.plotly_chart(fig_variance, use_container_width=True)
                            
                            # Tabla de varianza
                            variance_df = pd.DataFrame({
                                'Componente': [f'IC{i+1}' for i in range(len(ica.result.variance_per_component))],
                                'Varianza Individual (%)': ica.result.variance_per_component * 100,
                                'Varianza Acumulada (%)': np.cumsum(ica.result.variance_per_component) * 100
                            })
                            
                            st.dataframe(variance_df, width='stretch')
                        
                        with tab5:
                            st.subheader("Comparación: PCA vs ICA")
                            st.markdown("""
                            Comparación directa entre PCA e ICA aplicados a los mismos datos.
                            """)
                            
                            # Verificar si hay resultados de PCA en session_state
                            if hasattr(st.session_state, 'analyzer') and st.session_state.analyzer is not None:
                                with st.spinner("Calculando comparación PCA vs ICA..."):
                                    try:
                                        # Ejecutar PCA con el mismo número de componentes
                                        pca_results = analyzer.perform_pca(
                                            n_components=n_components_ica,
                                            scale=True
                                        )
                                        
                                        comparison_fig = compare_pca_vs_ica(
                                            data=df_for_ica,
                                            n_components=n_components_ica,
                                            feature_names=ica.result.feature_names
                                        )
                                        
                                        st.plotly_chart(comparison_fig, use_container_width=True)
                                        
                                        # Tabla comparativa
                                        st.markdown("### 📋 Comparación de Métricas")
                                        
                                        comparison_df = pd.DataFrame({
                                            'Métrica': [
                                                'Varianza Total Explicada (%)',
                                                'Kurtosis Promedio (abs)',
                                                'Objetivo Principal',
                                                'Asunción de Datos'
                                            ],
                                            'PCA': [
                                                f"{sum(pca_results.explained_variance_ratio) * 100:.2f}%",
                                                "0.00 (Gaussianos)",
                                                "Maximizar varianza",
                                                "Gaussianos / Cualquiera"
                                            ],
                                            'ICA': [
                                                f"{np.sum(ica.result.variance_per_component) * 100:.2f}%",
                                                f"{avg_kurtosis:.2f}",
                                                "Maximizar independencia",
                                                "No-Gaussianos"
                                            ]
                                        })
                                        
                                        st.dataframe(comparison_df, width='stretch', hide_index=True)
                                        
                                        st.success(
                                            "✅ **Recomendación:** "
                                            f"{'Usa ICA si tus datos son claramente no-Gaussianos y buscas separación de fuentes.' if avg_kurtosis > 1 else 'Usa PCA si buscas reducción de dimensionalidad general o tus datos son aproximadamente Gaussianos.'}"
                                        )
                                        
                                    except Exception as e:
                                        st.warning(f"⚠️ No se pudo completar la comparación: {e}")
                                        st.info("Ejecuta PCA en la pestaña anterior para habilitar la comparación completa.")
                            else:
                                st.info("ℹ️ Ejecuta PCA en la pestaña anterior para comparar ambos métodos.")
                        
                        # ==================== IMPORTANCIA DE FEATURES ====================
                        st.markdown("---")
                        st.subheader("🎯 Importancia de Features en Componentes Independientes")
                        
                        ic_selected = st.selectbox(
                            "Selecciona Componente Independiente",
                            [f'IC{i+1}' for i in range(n_components_ica)]
                        )
                        
                        ic_idx = int(ic_selected.replace('IC', '')) - 1
                        importance_df = ica.get_feature_importance(ic_idx)
                        
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            import plotly.express as px
                            fig_importance = px.bar(
                                importance_df.head(15).reset_index(),
                                x='importance',
                                y='feature',
                                orientation='h',
                                title=f'Top 15 Features en {ic_selected}',
                                labels={'feature': 'Variable', 'importance': 'Peso (abs)'},
                                color='importance',
                                color_continuous_scale='Viridis'
                            )
                            fig_importance.update_layout(yaxis={'categoryorder': 'total ascending'})
                            st.plotly_chart(fig_importance, use_container_width=True)
                        
                        with col2:
                            st.markdown(f"**Top 5 Features en {ic_selected}:**")
                            top5 = importance_df.head(5)
                            for idx, (feat, imp) in enumerate(top5.items(), 1):
                                st.markdown(f"{idx}. **{feat}**: {imp:.4f}")
                        
                        # Tabla completa
                        with st.expander("Ver todas las importancias"):
                            st.dataframe(importance_df, width='stretch')
                        
                        # ==================== ERROR DE RECONSTRUCCIÓN ====================
                        st.markdown("---")
                        st.subheader("🔄 Error de Reconstrucción")
                        
                        reconstruction_error = ica.get_reconstruction_error(df_for_ica)
                        
                        col1, col2 = st.columns(2)
                        col1.metric(
                            "Error de Reconstrucción (MSE)",
                            f"{reconstruction_error:.6f}",
                            help="Menor es mejor. Diferencia entre datos originales y reconstruidos desde ICs."
                        )
                        
                        # Reconstruir datos
                        reconstructed = ica.inverse_transform(transformed_data)
                        reconstruction_quality = 1 - (reconstruction_error / df_for_ica.var().mean())
                        
                        col2.metric(
                            "Calidad de Reconstrucción",
                            f"{max(0, reconstruction_quality * 100):.2f}%",
                            help="Porcentaje de información preservada"
                        )
                        
                        # ==================== GUARDAR RESULTADOS ====================
                        st.markdown("---")
                        st.subheader("💾 Guardar Resultados")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            if st.button("💾 Guardar Datos Transformados (ICA)", use_container_width=True):
                                try:
                                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                    ica_data_path = CLEANED_DATASETS_DIR / f"ica_transformed_{timestamp}.csv"
                                    ica_data_path.parent.mkdir(parents=True, exist_ok=True)
                                    
                                    transformed_data.to_csv(ica_data_path, index=False)
                                    st.success(f"✅ Datos ICA guardados en: {ica_data_path}")
                                except Exception as e:
                                    st.error(f"❌ Error: {e}")
                        
                        with col2:
                            if st.button("💾 Guardar Transformer ICA", use_container_width=True):
                                try:
                                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                    ica_model_path = CLEANED_DATASETS_DIR.parent / "models" / f"ica_transformer_{timestamp}.joblib"
                                    ica_model_path.parent.mkdir(parents=True, exist_ok=True)
                                    
                                    ica.save(str(ica_model_path))
                                    st.success(f"✅ Transformer ICA guardado en: {ica_model_path}")
                                    st.info("Puedes cargar este transformer más tarde para aplicar la misma transformación a nuevos datos.")
                                except Exception as e:
                                    st.error(f"❌ Error: {e}")
                        
                    except ValueError as ve:
                        st.error(f"❌ Error de validación: {ve}")
                        st.info(
                            "💡 **Sugerencias:**\n"
                            "- Asegúrate de haber aplicado limpieza de datos primero\n"
                            "- Verifica que las variables numéricas tengan datos válidos\n"
                            "- Aplica imputación de valores faltantes si es necesario\n"
                            "- ICA funciona mejor con datos no-Gaussianos"
                        )
                    except Exception as e:
                        st.error(f"❌ Error durante ICA: {e}")
                        with st.expander("Ver detalles del error"):
                            import traceback
                            st.code(traceback.format_exc())
    
    # Exportación PDF
    st.markdown("---")
    
    def generate_multivariate_report():
        """Generate multivariate analysis PDF report."""
        from pathlib import Path
        output_path = Path("reports") / "multivariate_eda.pdf"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        return generate_multivariate_pdf(
            df=df,
            numerical_cols=analyzer.numeric_cols,
            output_path=output_path,
            n_components=min(5, len(analyzer.numeric_cols)) if len(analyzer.numeric_cols) > 1 else None
        )
    
    pdf_export_section(
        generate_multivariate_report,
        section_title="Análisis Multivariado",
        default_filename="multivariate_eda.pdf",
        key_prefix="multivariate_eda"
    )


def quality_report_page():
    """Página de reporte de calidad de datos."""
    st.header("📋 Reporte de Calidad de Datos")
    
    # Decidir qué datos usar
    if st.session_state.cleaned_data is not None:
        df = st.session_state.cleaned_data
        st.success("✅ Analizando datos limpios")
        is_cleaned = True
    elif st.session_state.raw_data is not None:
        df = st.session_state.raw_data
        st.info("📊 Analizando datos crudos")
        is_cleaned = False
    else:
        st.warning("⚠️ Carga un dataset primero")
        return
    
    st.markdown("---")
    
    # Resumen general
    st.subheader("📊 Resumen General del Dataset")
    
    col1, col2, col3, col4 = st.columns(4)
    
    col1.metric("Total de Filas", f"{df.shape[0]:,}")
    col2.metric("Total de Columnas", f"{df.shape[1]:,}")
    col3.metric("Celdas Totales", f"{df.shape[0] * df.shape[1]:,}")
    
    total_missing = df.isna().sum().sum()
    missing_pct = (total_missing / (df.shape[0] * df.shape[1])) * 100
    col4.metric("Valores Faltantes", f"{total_missing:,} ({missing_pct:.1f}%)")
    
    # Tipos de variables
    st.markdown("---")
    st.subheader("🏷️ Tipos de Variables")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Variables Numéricas", len(numeric_cols))
        if numeric_cols:
            with st.expander("Ver lista"):
                st.write(numeric_cols)
    
    with col2:
        st.metric("Variables Categóricas", len(categorical_cols))
        if categorical_cols:
            with st.expander("Ver lista"):
                st.write(categorical_cols)
    
    # Análisis de valores faltantes
    st.markdown("---")
    st.subheader("❓ Análisis de Valores Faltantes")
    
    missing_df = pd.DataFrame({
        'Variable': df.columns,
        'Missing Count': df.isna().sum().values,
        'Missing %': (df.isna().sum() / len(df) * 100).values
    })
    missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing %', ascending=False)
    
    if len(missing_df) > 0:
        # Gráfico de barras
        import plotly.express as px
        fig = px.bar(
            missing_df.head(20),
            x='Missing %',
            y='Variable',
            orientation='h',
            title='Top 20 Variables con Valores Faltantes',
            color='Missing %',
            color_continuous_scale='Reds'
        )
        fig.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
        
        # Tabla detallada
        st.dataframe(missing_df, width='stretch', height=300)
        
        # Alertas
        critical_missing = missing_df[missing_df['Missing %'] > 50]
        if len(critical_missing) > 0:
            st.error(f"🔴 **CRÍTICO**: {len(critical_missing)} variables tienen >50% de valores faltantes")
            with st.expander("Ver variables críticas"):
                st.dataframe(critical_missing, width='stretch')
    else:
        st.success("✅ ¡No hay valores faltantes en el dataset!")
    
    # Análisis de duplicados
    st.markdown("---")
    st.subheader("🔄 Análisis de Duplicados")
    
    n_duplicates = df.duplicated().sum()
    pct_duplicates = (n_duplicates / len(df)) * 100
    
    col1, col2 = st.columns(2)
    col1.metric("Filas Duplicadas", f"{n_duplicates:,}")
    col2.metric("Porcentaje", f"{pct_duplicates:.2f}%")
    
    if n_duplicates > 0:
        st.warning(f"⚠️ Se encontraron {n_duplicates} filas duplicadas")
    else:
        st.success("✅ No hay filas duplicadas")
    
    # Análisis de cardinalidad
    st.markdown("---")
    st.subheader("🎯 Cardinalidad de Variables")
    
    cardinality_df = pd.DataFrame({
        'Variable': df.columns,
        'Unique Values': [df[col].nunique() for col in df.columns],
        'Cardinality %': [(df[col].nunique() / len(df)) * 100 for col in df.columns]
    })
    cardinality_df = cardinality_df.sort_values('Unique Values', ascending=False)
    
    # Identificar problemas
    high_card = cardinality_df[cardinality_df['Cardinality %'] > 95]
    low_card = cardinality_df[cardinality_df['Unique Values'] == 1]
    
    col1, col2 = st.columns(2)
    
    with col1:
        if len(high_card) > 0:
            st.warning(f"⚠️ {len(high_card)} variables con alta cardinalidad (>95%)")
            with st.expander("Ver variables"):
                st.dataframe(high_card, width='stretch')
    
    with col2:
        if len(low_card) > 0:
            st.error(f"🔴 {len(low_card)} variables constantes (1 valor único)")
            with st.expander("Ver variables"):
                st.dataframe(low_card, width='stretch')
    
    st.dataframe(cardinality_df, width='stretch', height=300)
    
    # Análisis de outliers (solo numéricas)
    if len(numeric_cols) > 0:
        st.markdown("---")
        st.subheader("📊 Análisis de Outliers (Método IQR)")
        
        outlier_summary = []
        
        for col in numeric_cols:
            series = df[col].dropna()
            if len(series) > 0:
                Q1 = series.quantile(0.25)
                Q3 = series.quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                
                n_outliers = ((series < lower) | (series > upper)).sum()
                pct_outliers = (n_outliers / len(series)) * 100
                
                outlier_summary.append({
                    'Variable': col,
                    'Outliers Count': n_outliers,
                    'Outliers %': pct_outliers
                })
        
        outlier_df = pd.DataFrame(outlier_summary)
        outlier_df = outlier_df[outlier_df['Outliers Count'] > 0].sort_values('Outliers %', ascending=False)
        
        if len(outlier_df) > 0:
            st.dataframe(outlier_df, width='stretch', height=300)
            
            # Gráfico
            import plotly.express as px
            fig = px.bar(
                outlier_df.head(15),
                x='Outliers %',
                y='Variable',
                orientation='h',
                title='Variables con Mayor Porcentaje de Outliers',
                color='Outliers %',
                color_continuous_scale='Oranges'
            )
            fig.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("✅ No se detectaron outliers significativos (método IQR)")
    
    # Reporte final
    st.markdown("---")
    st.subheader("📝 Resumen de Calidad")
    
    quality_score = 100
    issues = []
    
    if missing_pct > 10:
        quality_score -= 20
        issues.append(f"🔴 Alto porcentaje de valores faltantes ({missing_pct:.1f}%)")
    elif missing_pct > 5:
        quality_score -= 10
        issues.append(f"🟡 Porcentaje moderado de valores faltantes ({missing_pct:.1f}%)")
    
    if pct_duplicates > 5:
        quality_score -= 15
        issues.append(f"🔴 Alto porcentaje de duplicados ({pct_duplicates:.1f}%)")
    elif pct_duplicates > 0:
        quality_score -= 5
        issues.append(f"🟡 Hay filas duplicadas ({pct_duplicates:.1f}%)")
    
    if len(low_card) > 0:
        quality_score -= 10
        issues.append(f"🔴 {len(low_card)} variables constantes detectadas")
    
    if len(high_card) > 5:
        quality_score -= 10
        issues.append(f"🟡 {len(high_card)} variables con alta cardinalidad")
    
    # Mostrar puntuación
    col1, col2 = st.columns([1, 2])
    
    with col1:
        if quality_score >= 90:
            st.success(f"### 🌟 Puntuación de Calidad: {quality_score}/100")
        elif quality_score >= 70:
            st.warning(f"### ⚠️ Puntuación de Calidad: {quality_score}/100")
        else:
            st.error(f"### 🔴 Puntuación de Calidad: {quality_score}/100")
    
    with col2:
        if issues:
            st.markdown("**Problemas detectados:**")
            for issue in issues:
                st.markdown(f"- {issue}")
        else:
            st.success("✅ No se detectaron problemas significativos de calidad")
    
    # Botón de descarga de reporte
    st.markdown("---")
    if st.button("📥 Descargar Reporte Completo (JSON)"):
        report = {
            'timestamp': datetime.now().isoformat(),
            'dataset_type': 'cleaned' if is_cleaned else 'raw',
            'shape': {'rows': int(df.shape[0]), 'columns': int(df.shape[1])},
            'missing_summary': missing_df.to_dict('records') if len(missing_df) > 0 else [],
            'duplicates': {'count': int(n_duplicates), 'percentage': float(pct_duplicates)},
            'cardinality': cardinality_df.to_dict('records'),
            'quality_score': quality_score,
            'issues': issues
        }
        
        report_json = json.dumps(report, indent=2, ensure_ascii=False)
        st.download_button(
            "Descargar JSON",
            data=report_json,
            file_name=f"quality_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )


def main():
    """Función principal de la aplicación."""
    init_session_state()
    
    # Título y descripción
    st.title("🧹 Preparación de Datos y Análisis Exploratorio")
    st.markdown("""
    **Sistema completo de limpieza y análisis de datos para el predictor de mortalidad por IAM**
    
    Utiliza las pestañas siguientes para:
    - Cargar datos desde múltiples fuentes
    - Seleccionar variables relevantes para el análisis
    - Limpiar y preprocesar datos con configuraciones avanzadas
    - Realizar análisis exploratorio completo
    - Generar reportes de calidad
    """)
    
    # Tabs principales
    tabs = st.tabs([
        "📂 Carga de Datos",
        "🎯 Selección de Variables",
        "🧹 Limpieza de Datos",
        "📈 Análisis Univariado",
        "📊 Análisis Bivariado",
        "🔬 Análisis Multivariado",
        "📋 Reporte de Calidad"
    ])
    
    with tabs[0]:
        load_data_page()
    
    with tabs[1]:
        variable_selection_page()
    
    with tabs[2]:
        data_cleaning_page()
    
    with tabs[3]:
        univariate_analysis_page()
    
    with tabs[4]:
        bivariate_analysis_page()
    
    with tabs[5]:
        multivariate_analysis_page()
    
    with tabs[6]:
        quality_report_page()


if __name__ == "__main__":
    main()
