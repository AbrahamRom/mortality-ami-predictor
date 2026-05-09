# Dashboard Overview

The **Mortality AMI Predictor Dashboard** is an interactive web application built with Streamlit that provides a user-friendly interface for all features of the system.

## 🚀 Launching the Dashboard

```bash
cd Tools
streamlit run dashboard/Dashboard.py
```

The dashboard will open automatically in your browser at `http://localhost:8501`.

## 📊 Dashboard Pages

### 🧹 Data Cleaning and EDA

Clean your raw dataset with automated preprocessing:

- Upload CSV files
- Handle missing values
- Detect and treat outliers
- Encode categorical variables
- View cleaning statistics
- Download cleaned dataset

### 📊 Data Overview

Explore your data with interactive visualizations:

- Summary statistics
- Variable distributions
- Correlation matrix
- Missing value patterns
- Class balance analysis

### 🤖 Model Training

Train machine learning models with customizable parameters:

- Select features
- Choose model type (Logistic, RF, XGBoost, Neural Network)
- Configure hyperparameters
- Cross-validation
- View training metrics
- Save trained models

### 🔮 Predictions

Make predictions on new patient data:

- Load trained models
- Input patient data manually or batch upload
- View risk scores with confidence intervals
- Risk stratification
- Export predictions

### 📈 Model Evaluation

Comprehensive model performance assessment:

- ROC curve and AUC
- Precision-Recall curve
- Calibration plot
- Decision curve analysis
- Confusion matrix
- Bootstrap confidence intervals

### 🔍 Explainability

Understand model decisions:

- SHAP waterfall plots
- SHAP summary plots
- Partial dependence plots
- Permutation importance
- Individual patient explanations

### 📋 Clinical Scores

Calculate and compare clinical risk scores:

- GRACE Score
- TIMI Score
- Killip Classification
- Compare with ML predictions

### 🔧 Custom Models

Create and integrate your own models:

- Upload custom model code
- Train with standard pipeline
- Evaluate and compare
- Deploy seamlessly

## 🎨 Dashboard Features

### State Management

The dashboard maintains state across pages using Streamlit's session state:

```python
import streamlit as st

# Access shared state
if 'trained_model' in st.session_state:
    model = st.session_state.trained_model
```

### Interactive Widgets

- **Sliders**: Adjust hyperparameters
- **Selectboxes**: Choose options
- **File uploaders**: Upload data and models
- **Data editors**: Edit patient information
- **Charts**: Interactive Plotly visualizations

### Caching

Efficient caching for better performance:

```python
@st.cache_data
def load_large_dataset(path):
    return pd.read_csv(path)

@st.cache_resource
def load_model(path):
    return joblib.load(path)
```

## 📱 Usage Examples

### Example 1: Complete Workflow

1. **Upload data** in Data Cleaning page
2. **Clean and explore** data
3. **Train model** in Model Training page
4. **Evaluate** in Model Evaluation page
5. **Make predictions** on new patients
6. **Explain** with SHAP analysis

### Example 2: Quick Prediction

1. Load pre-trained model
2. Go to Predictions page
3. Enter patient data
4. View risk score instantly

## 🔧 Configuration

### Streamlit Settings

Create `.streamlit/config.toml`:

```toml
[server]
port = 8501
enableCORS = false

[theme]
primaryColor = "#b71c1c"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f5f5f5"
textColor = "#262730"
```

### Dashboard State

The dashboard uses a centralized state manager in `app/state.py`:

```python
from app.state import DashboardState

# Initialize state
state = DashboardState()

# Store data
state.set_data('cleaned_df', df)

# Retrieve data
df = state.get_data('cleaned_df')
```

## 💡 Tips

!!! tip "Use the Sidebar"
    Navigate between pages using the sidebar menu for seamless workflow.

!!! tip "Save Your Work"
    Always save trained models and cleaned datasets before closing the dashboard.

!!! warning "Memory Usage"
    Large datasets may consume significant memory. Use the fragment dataset for testing.

## 🐛 Troubleshooting

### Dashboard Not Loading

```bash
# Check Streamlit version
streamlit --version

# Reinstall if needed
pip install --upgrade streamlit
```

### Port Already in Use

```bash
# Use different port
streamlit run dashboard/Dashboard.py --server.port 8502
```

### Import Errors

```bash
# Ensure you're in the Tools directory
cd Tools
streamlit run dashboard/Dashboard.py
```

## See Also

- [Installation Guide](../getting-started/installation.md)
- [Quick Start](../getting-started/quickstart.md)
- [Model Training Guide](training.md)
