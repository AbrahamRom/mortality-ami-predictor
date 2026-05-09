"""AMI Mortality/Arrhythmia Prediction Dashboard.

A modular Streamlit application for in-hospital mortality and ventricular arrhythmia prediction.

Directory Structure:
- streamlit_app.py: Main entry point (this file)
- app/: Shared utilities and state management
- pages/: Individual dashboard pages (auto-discovered by Streamlit)
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Add parent directory to path for src imports
root_dir = Path(__file__).parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from app.config import configure_page
from app.state import initialize_state

# Configure page settings
configure_page()

# Initialize session state
initialize_state()

# Add logo and title to sidebar (only in main page)
logo_path = Path(__file__).parent / "app" / "assets" / "logo_ami_predictor.png"
if logo_path.exists():
    st.sidebar.image(str(logo_path), width='stretch')

st.sidebar.title("Mortality AMI Predictor")
st.sidebar.markdown("---")

# Main page content
st.title("🏥 AMI Mortality & Arrhythmia Predictor")

st.markdown("""
### Welcome to the Dashboard

This application provides comprehensive tools for predicting in-hospital mortality 
and ventricular arrhythmias in patients with Acute Myocardial Infarction (AMI).

#### Features:
- 📊 **Data Overview**: Explore your dataset and view statistics
- 🤖 **Model Training**: Train and evaluate multiple ML models
- 🔮 **Predictions**: Make predictions on individual patients or batches
- 📈 **Model Evaluation**: View metrics, calibration curves, and decision curves
- 🔍 **Explainability**: Understand model decisions with SHAP values
- 📋 **Clinical Scores**: Calculate GRACE and TIMI risk scores
- ⚙️ **Model Comparison**: Compare different models and configurations

#### Getting Started:
1. **Configure dataset path** in the sidebar (or set `DATASET_PATH` environment variable)
2. **Select task**: Mortality prediction or arrhythmia prediction
3. **Navigate pages** using the sidebar to explore different features
4. **Train models** on the Model Training page
5. **Make predictions** on the Prediction page

---

**⚠️ Important Note**: This tool is for research and educational purposes only. 
Clinical decisions should always be made by qualified healthcare professionals.
""")

# Quick stats in the main page
from app.state import get_state, load_data

data_dir = get_state("data_dir")
default_data = data_dir / "recuima-020425.csv"

if default_data.exists():
    try:
        df = load_data(default_data, use_cache=True)
        
        st.subheader("📊 Dataset Quick Stats")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Patients", f"{len(df):,}")
        
        with col2:
            st.metric("Total Features", len(df.columns))
        
        with col3:
            missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
            st.metric("Missing Data", f"{missing_pct:.1f}%")
        
        with col4:
            models_dir = get_state("models_dir")
            if models_dir and models_dir.exists():
                n_models = len(list(models_dir.glob("*.joblib")))
                st.metric("Saved Models", n_models)
            else:
                st.metric("Saved Models", 0)
        
        st.info("👈 Use the sidebar to navigate to different pages and explore more features!")
    
    except Exception as e:
        st.info(f"💡 Load your dataset to see quick stats here. ({e})")
else:
    st.info("👈 Please configure the dataset path in the sidebar to get started.")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray; font-size: 0.9em;'>
        <p>Mortality AMI Predictor Dashboard v2.0</p>
        <p>Built with Streamlit • Powered by scikit-learn, XGBoost, and SHAP</p>
    </div>
    """,
    unsafe_allow_html=True
)
