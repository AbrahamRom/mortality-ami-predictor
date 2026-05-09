# Configuration Guide

Complete guide to configuring the Mortality AMI Predictor system.

## Environment Variables

The system can be configured using environment variables. Create a `.env` file in the `Tools` directory:

```bash
# Dataset Configuration
DATASET_PATH=../DATA/recuima-020425.csv

# Experiment Tracking
EXPERIMENT_TRACKER=mlflow
TRACKING_URI=./mlruns

# Target Variables
TARGET_COLUMN=mortality_inhospital
ARRHYTHMIA_COLUMN=ventricular_arrhythmia
REGRESSION_TARGET=

# Directory Paths
PROCESSED_DIR=processed
CLEANED_DATASETS_DIR=processed/cleaned_datasets
PLOTS_DIR=processed/plots
MODELS_DIR=processed/models
TESTSETS_DIR=processed/models/testsets
METADATA_PATH=processed/variable_metadata.json
```

## Configuration Options

### Dataset Settings

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `DATASET_PATH` | str | `""` | Path to the main dataset CSV file |
| `TARGET_COLUMN` | str | `"mortality_inhospital"` | Name of the binary target variable |
| `ARRHYTHMIA_COLUMN` | str | `"ventricular_arrhythmia"` | Optional secondary target |
| `REGRESSION_TARGET` | str | `None` | Optional continuous target variable |

### Experiment Tracking

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `EXPERIMENT_TRACKER` | str | `"mlflow"` | Tracking system: "mlflow" or "wandb" |
| `TRACKING_URI` | str | `None` | URI for MLflow or W&B entity/project |

### Directory Structure

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `PROCESSED_DIR` | str | `"processed"` | Root directory for outputs |
| `CLEANED_DATASETS_DIR` | str | `"processed/cleaned_datasets"` | Cleaned data location |
| `PLOTS_DIR` | str | `"processed/plots"` | Visualization outputs |
| `MODELS_DIR` | str | `"processed/models"` | Trained model storage |
| `TESTSETS_DIR` | str | `"processed/models/testsets"` | Test sets location |
| `METADATA_PATH` | str | `"processed/variable_metadata.json"` | Variable metadata file |

## Setting Environment Variables

=== "Windows PowerShell"

    ```powershell
    # Set temporarily (current session)
    $env:DATASET_PATH = "C:\path\to\dataset.csv"
    $env:TARGET_COLUMN = "mortality_inhospital"
    
    # Set permanently (user level)
    [Environment]::SetEnvironmentVariable("DATASET_PATH", "C:\path\to\dataset.csv", "User")
    ```

=== "Linux/Mac"

    ```bash
    # Set temporarily (current session)
    export DATASET_PATH="/path/to/dataset.csv"
    export TARGET_COLUMN="mortality_inhospital"
    
    # Set permanently (add to ~/.bashrc or ~/.zshrc)
    echo 'export DATASET_PATH="/path/to/dataset.csv"' >> ~/.bashrc
    source ~/.bashrc
    ```

=== ".env File"

    Create a `.env` file in the `Tools` directory:
    
    ```bash
    DATASET_PATH=../DATA/recuima-020425.csv
    TARGET_COLUMN=mortality_inhospital
    EXPERIMENT_TRACKER=mlflow
    ```
    
    Then load it in Python:
    
    ```python
    from dotenv import load_dotenv
    load_dotenv()
    
    from src.config import ProjectConfig
    config = ProjectConfig()
    ```

## Advanced Configuration

### Custom Model Parameters

Configure model hyperparameters in the dashboard or via code:

```python
from src.training.trainer import ModelTrainer

trainer = ModelTrainer(
    model_type="xgboost",
    params={
        "n_estimators": 100,
        "max_depth": 10,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42
    }
)
```

### Data Cleaning Configuration

```python
from src.cleaning.cleaner import DataCleaner

cleaner = DataCleaner(
    target_column="mortality_inhospital",
    missing_threshold=0.4,
    outlier_method="isolation_forest",
    encoding_strategy="target",
    imputation_strategy="knn"
)
```

### Visualization Settings

Configure plot styles and formats:

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Configure defaults
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'
```

## MLflow Configuration

### Local MLflow

```bash
# Start MLflow UI
mlflow ui --port 5000 --backend-store-uri ./mlruns
```

### Remote MLflow

```python
import os
os.environ["TRACKING_URI"] = "http://mlflow-server:5000"
os.environ["MLFLOW_TRACKING_USERNAME"] = "user"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "password"
```

## Weights & Biases Configuration

```python
import os
os.environ["EXPERIMENT_TRACKER"] = "wandb"
os.environ["TRACKING_URI"] = "your-entity/your-project"
os.environ["WANDB_API_KEY"] = "your-api-key"
```

## Dashboard Configuration

### Streamlit Settings

Create `.streamlit/config.toml`:

```toml
[server]
port = 8501
enableCORS = false
enableXsrfProtection = true

[browser]
gatherUsageStats = false

[theme]
primaryColor = "#b71c1c"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f5f5f5"
textColor = "#262730"
font = "sans serif"
```

### Dashboard Cache

```python
import streamlit as st

# Configure cache
st.set_page_config(
    page_title="Mortality AMI Predictor",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)
```

## Best Practices

!!! tip "Use Environment Variables"
    Keep sensitive information and paths in environment variables, not in code:
    
    ```python
    # Good ✓
    from src.config import ProjectConfig
    config = ProjectConfig()
    data_path = config.dataset_path
    
    # Bad ✗
    data_path = "C:/Users/HP/data.csv"
    ```

!!! tip "Version Control"
    Add `.env` to `.gitignore` to avoid committing sensitive data:
    
    ```
    # .gitignore
    .env
    *.env
    ```

!!! warning "Path Separators"
    Use forward slashes (/) in paths for cross-platform compatibility:
    
    ```python
    # Good ✓
    path = "processed/models/model.joblib"
    
    # Bad ✗ (Windows-specific)
    path = "processed\\models\\model.joblib"
    ```

## Troubleshooting

### Configuration Not Loading

```python
# Check current configuration
from src.config import ProjectConfig
config = ProjectConfig()

print(f"Dataset path: {config.dataset_path}")
print(f"Is empty: {config.dataset_path == ''}")

# Set manually if needed
config.dataset_path = "/path/to/dataset.csv"
```

### Environment Variables Not Set

```powershell
# Windows: Check environment variables
Get-ChildItem Env: | Where-Object {$_.Name -like "*DATASET*"}

# Set if missing
$env:DATASET_PATH = "path/to/data.csv"
```

```bash
# Linux/Mac: Check environment variables
printenv | grep DATASET

# Set if missing
export DATASET_PATH="/path/to/data.csv"
```

## See Also

- [Installation Guide](installation.md) - Initial setup
- [Quick Start](quickstart.md) - Getting started
- [API Reference](../api/config.md) - Configuration module documentation
