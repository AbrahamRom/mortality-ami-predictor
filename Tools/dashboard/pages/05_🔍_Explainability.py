"""Explainability and SHAP Analysis page."""
from __future__ import annotations

import sys
from pathlib import Path

# Add parent directories to path
root_dir = Path(__file__).parents[2]
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import joblib
import numpy as np
import streamlit as st

from app import (
    initialize_state,
    list_saved_models,
)
from app.config import get_plotly_config
from src.config import CONFIG
from src.explainability import (
    compute_shap_values,
    plot_shap_beeswarm,
    plot_shap_bar,
    plot_shap_waterfall,
    plot_shap_force,
    get_feature_importance,
    get_sample_shap_values,
    generate_explainability_pdf,
)
from src.reporting import pdf_export_section

# Initialize
initialize_state()

# Page config
st.title("🔍 Model Explainability")
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
    help="Select a trained model for explainability analysis"
)

model_path = saved_models[selected_model_name]

# SHAP settings
st.sidebar.markdown("---")
st.sidebar.header("⚙️ SHAP Settings")

n_samples = st.sidebar.slider(
    "Number of samples",
    min_value=50,
    max_value=500,
    value=200,
    step=50,
    help="Number of samples to use for SHAP analysis"
)

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
    st.stop()

# Get feature columns
# Use the actual target column name from session state, or fallback to CONFIG
target_col = target_col_name if target_col_name else (CONFIG.target_column if task == "mortality" else CONFIG.arrhythmia_column)

# Exclude all potential target columns from features
excluded_cols = {CONFIG.target_column, CONFIG.arrhythmia_column}
if target_col:
    excluded_cols.add(target_col)
feature_cols = [c for c in df.columns if c not in excluded_cols]

# Sample data for SHAP
sample_df = df[feature_cols].iloc[:n_samples]

st.markdown("---")

# Check if SHAP is available
try:
    import shap
    shap_available = True
except ImportError:
    shap_available = False
    st.error("❌ SHAP not installed. Please install it with: `pip install shap`")
    st.stop()

# SHAP analysis
st.subheader("SHAP Analysis")
st.caption(f"Using {len(sample_df)} samples for explainability")

if st.button("🚀 Compute SHAP Values", type="primary", width='stretch'):
    try:
        with st.spinner("Computing SHAP values... This may take a moment"):
            # Compute SHAP values using the explainability module
            explainer, shap_explanation = compute_shap_values(
                model=model,
                X=sample_df,
                feature_names=feature_cols,
                max_samples=n_samples
            )
            
            # Store in session state
            st.session_state.shap_explainer = explainer
            st.session_state.shap_values = shap_explanation
            
            st.success("✅ SHAP values computed successfully!")
    
    except Exception as e:
        st.error(f"❌ Error computing SHAP values: {e}")
        st.exception(e)

# Display SHAP plots if available
if "shap_values" in st.session_state and st.session_state.shap_values is not None:
    shap_values = st.session_state.shap_values
    
    st.markdown("---")
    
    # Tabs for different visualizations
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Beeswarm Plot",
        "📈 Bar Plot",
        "🎯 Force Plot",
        "🌊 Waterfall Plot"
    ])
    
    # Get Plotly configuration
    plotly_config = get_plotly_config()
    
    with tab1:
        st.markdown("### Beeswarm Plot")
        st.caption("Shows the distribution of SHAP values for each feature across all samples")
        
        try:
            fig = plot_shap_beeswarm(shap_values, max_display=20)
            st.plotly_chart(fig, use_container_width=True, config=plotly_config)
        
        except Exception as e:
            st.error(f"Error creating beeswarm plot: {e}")
    
    with tab2:
        st.markdown("### Feature Importance (Bar Plot)")
        st.caption("Mean absolute SHAP value for each feature")
        
        try:
            fig = plot_shap_bar(shap_values, max_display=20)
            st.plotly_chart(fig, use_container_width=True, config=plotly_config)
        
        except Exception as e:
            st.error(f"Error creating bar plot: {e}")
    
    with tab3:
        st.markdown("### Force Plot")
        st.caption("Visualize predictions for individual samples")
        
        # Add controls for force plot
        col1, col2 = st.columns([3, 1])
        
        with col1:
            sample_idx = st.slider(
                "Select sample index",
                min_value=0,
                max_value=len(shap_values) - 1,
                value=0
            )
        
        with col2:
            max_features = st.slider(
                "Features to display",
                min_value=10,
                max_value=50,
                value=20,
                step=5,
                help="Number of top features to show"
            )
        
        try:
            fig = plot_shap_force(shap_values, sample_idx=sample_idx, max_display=max_features)
            st.plotly_chart(fig, use_container_width=True, config=plotly_config)
            
            # Add info about interactivity
            st.info("💡 **Tip**: Hover over bars to see feature values and SHAP contributions. Use zoom and pan tools to explore the plot. Click the camera icon to export as PNG.")
        
        except Exception as e:
            st.error(f"Error creating force plot: {e}")
            # Try alternative visualization
            try:
                st.write("SHAP values for this sample:")
                shap_df = get_sample_shap_values(shap_values, sample_idx=sample_idx)
                st.dataframe(shap_df, use_container_width=True)
            except Exception as e2:
                st.error(f"Error showing alternative view: {e2}")
    
    with tab4:
        st.markdown("### Waterfall Plot")
        st.caption("Shows how each feature contributes to push the model output from the base value")
        
        # Add controls for waterfall plot
        col1, col2 = st.columns([3, 1])
        
        with col1:
            sample_idx = st.slider(
                "Select sample index",
                min_value=0,
                max_value=len(shap_values) - 1,
                value=0,
                key="waterfall_slider"
            )
        
        with col2:
            max_features_waterfall = st.slider(
                "Features to display",
                min_value=10,
                max_value=30,
                value=20,
                step=5,
                key="waterfall_features",
                help="Number of top features to show"
            )
        
        try:
            fig = plot_shap_waterfall(shap_values, sample_idx=sample_idx, max_display=max_features_waterfall)
            st.plotly_chart(fig, use_container_width=True, config=plotly_config)
        
        except Exception as e:
            st.error(f"Error creating waterfall plot: {e}")
    
    # Additional analysis
    st.markdown("---")
    
    with st.expander("🔢 Feature Importance Rankings"):
        try:
            importance_df = get_feature_importance(shap_values)
            
            st.dataframe(
                importance_df.style.format({"Mean |SHAP|": "{:.6f}"}),
                use_container_width=True,
                hide_index=True
            )
        
        except Exception as e:
            st.error(f"Error computing feature importance: {e}")
    
    with st.expander("ℹ️ About SHAP"):
        st.markdown("""
        **SHAP (SHapley Additive exPlanations)**
        
        SHAP values explain the contribution of each feature to individual predictions:
        
        - **Positive SHAP value**: Feature pushes prediction higher
        - **Negative SHAP value**: Feature pushes prediction lower
        - **Magnitude**: Importance of the feature for that prediction
        
        **Visualizations:**
        - **Beeswarm**: Overall feature importance and value distribution
        - **Bar**: Simple feature importance ranking
        - **Force**: Individual prediction explanation (optimized for many features)
        - **Waterfall**: Step-by-step contribution breakdown
        
        **Interactive Features:**
        - 🔍 **Zoom & Pan**: Click and drag to zoom, double-click to reset
        - 🖱️ **Hover**: See detailed values for each feature
        - 📊 **Adjustable**: Use sliders to control number of features displayed
        - 💾 **Export**: Use the camera icon to download as PNG
        
        SHAP values are based on game theory and provide consistent, locally accurate explanations.
        """)
    
    # Exportación PDF
    st.markdown("---")
    st.subheader("📄 Exportar Reporte de Explicabilidad")
    
    # Check if SHAP values are available (shap_values is already checked above)
    if st.session_state.get('shap_values') is not None:
        try:
            importance_df = get_feature_importance(st.session_state.shap_values)
            top_features = importance_df.head(10).index.tolist()
            
            explainability_data = {
                'feature_importance': {
                    'builtin': importance_df
                },
                'shap_values': st.session_state.shap_values,
                'feature_names': feature_cols,
                'top_features': top_features,
                'plots': {}
            }
            
            def generate_explainability_report():
                """Generate explainability PDF report."""
                from pathlib import Path
                output_path = Path("reports") / "explainability_report.pdf"
                output_path.parent.mkdir(parents=True, exist_ok=True)
                return generate_explainability_pdf(
                    model_name=selected_model_name,
                    explainability_data=explainability_data,
                    output_path=output_path
                )
            
            pdf_export_section(
                generate_explainability_report,
                section_title="Reporte de Explicabilidad",
                default_filename="explainability_report.pdf",
                key_prefix="explainability_report"
            )
        
        except Exception as e:
            st.error(f"Error preparando datos para PDF: {e}")
    else:
        st.info("ℹ️ Calcula valores SHAP primero para generar el reporte PDF")

else:
    st.info("👆 Click 'Compute SHAP Values' to generate explainability visualizations")

