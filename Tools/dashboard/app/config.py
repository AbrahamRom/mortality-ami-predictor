"""Dashboard configuration and page setup."""
from __future__ import annotations

from pathlib import Path

import streamlit as st


# Path configuration
ROOT_DIR = Path(__file__).parents[2]  # Points to Tools/
PROCESSED_DIR = ROOT_DIR / "processed"
CLEANED_DATASETS_DIR = PROCESSED_DIR / "cleaned_datasets"
PLOTS_DIR = PROCESSED_DIR / "plots"
MODELS_DIR = PROCESSED_DIR / "models"
TESTSETS_DIR = PROCESSED_DIR / "models" / "testsets"
METADATA_PATH = PROCESSED_DIR / "variable_metadata.json"
PREPROCESSING_CONFIG_PATH = PROCESSED_DIR / "preprocessing_config.json"
EDA_CACHE_PATH = PROCESSED_DIR / "eda_cache.pkl"

# Plots subdirectories
PLOTS_EDA_DIR = PLOTS_DIR / "eda"
PLOTS_EVALUATION_DIR = PLOTS_DIR / "evaluation"
PLOTS_EXPLAINABILITY_DIR = PLOTS_DIR / "explainability"
PLOTS_TRAINING_DIR = PLOTS_DIR / "training"

# Model type directories
MODEL_TYPES = ["dtree", "knn", "xgb", "logistic", "random_forest", "neural_network"]


# Configure Plotly defaults
PLOTLY_CONFIG = {
    # Display mode bar
    "displayModeBar": True,
    "displaylogo": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
    "modeBarButtonsToAdd": ["drawline", "drawopenpath", "eraseshape"],
    # Export options
    "toImageButtonOptions": {
        "format": "png",
        "filename": "plot",
        "height": 800,
        "width": 1200,
        "scale": 2,
    },
    # Interaction
    "scrollZoom": True,
    "doubleClick": "reset",
    "showAxisDragHandles": True,
    "showAxisRangeEntryBoxes": True,
}

# Flag to track if plotly has been configured
_plotly_configured = False


def _configure_plotly():
    """Configure Plotly template (lazy initialization to avoid circular imports)."""
    global _plotly_configured
    if not _plotly_configured:
        import plotly.io as pio
        pio.templates.default = "plotly_white"
        _plotly_configured = True


def get_plotly_config():
    """Get Plotly configuration dictionary for Streamlit charts.
    
    Returns:
        Dictionary with Plotly display and interaction settings
    """
    _configure_plotly()
    return PLOTLY_CONFIG


def configure_page():
    """Configure Streamlit page settings."""
    # Find logo if available
    assets_dir = Path(__file__).parent / "assets"
    logo = None
    
    if assets_dir.exists():
        for ext in ("logo.png", "logo.jpg", "logo.jpeg", "logo.ico"):
            logo_path = assets_dir / ext
            if logo_path.exists():
                logo = str(logo_path)
                break
        
        if logo is None:
            # Fallback: pick first image file
            imgs = list(assets_dir.glob("*.png")) + list(assets_dir.glob("*.jpg"))
            if imgs:
                logo = str(imgs[0])
    
    st.set_page_config(
        page_title="AMI Mortality/Arrhythmia Dashboard",
        page_icon=logo if logo else "🏥",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'About': "# AMI Predictor Dashboard\n\nPredicting in-hospital mortality and ventricular arrhythmias."
        }
    )


def apply_custom_css():
    """Apply custom CSS styling to the dashboard."""
    st.markdown("""
        <style>
        /* Main content area */
        .main {
            padding: 2rem;
        }
        
        /* Metric styling */
        [data-testid="stMetricValue"] {
            font-size: 2rem;
            font-weight: bold;
        }
        
        /* Info/warning/success boxes */
        .stAlert {
            border-radius: 0.5rem;
            padding: 1rem;
        }
        
        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background-color: #f0f2f6;
        }
        
        /* Button styling */
        .stButton button {
            border-radius: 0.5rem;
            font-weight: 500;
        }
        
        /* Dataframe styling */
        .dataframe {
            font-size: 0.9rem;
        }
        
        /* Plotly chart container */
        .js-plotly-plot {
            border-radius: 0.5rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        </style>
    """, unsafe_allow_html=True)
