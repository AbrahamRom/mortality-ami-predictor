# Installation Guide

This guide will help you install and set up the Mortality AMI Predictor system on your machine.

## Prerequisites

Before installing, ensure you have the following:

- **Python 3.8 or higher** ([Download Python](https://www.python.org/downloads/))
- **Git** ([Download Git](https://git-scm.com/downloads))
- **pip** (usually comes with Python)
- At least **2GB of free disk space**
- (Optional) **Anaconda/Miniconda** for environment management

## Installation Methods

=== "pip (Recommended)"

    ### 1. Clone the Repository
    
    ```bash
    git clone https://github.com/Pol4720/mortality-ami-predictor.git
    cd mortality-ami-predictor/Tools
    ```
    
    ### 2. Create Virtual Environment
    
    ```bash
    # Create virtual environment
    python -m venv venv
    
    # Activate (Windows)
    venv\Scripts\activate
    
    # Activate (macOS/Linux)
    source venv/bin/activate
    ```
    
    ### 3. Install Dependencies
    
    ```bash
    pip install -r requirements.txt
    ```
    
    ### 4. Verify Installation
    
    ```bash
    python -c "import src; print('Installation successful!')"
    ```

=== "Conda"

    ### 1. Clone the Repository
    
    ```bash
    git clone https://github.com/Pol4720/mortality-ami-predictor.git
    cd mortality-ami-predictor/Tools
    ```
    
    ### 2. Create Conda Environment
    
    ```bash
    conda env create -f environment.yml
    conda activate mortality-ami-predictor
    ```
    
    ### 3. Verify Installation
    
    ```bash
    python -c "import src; print('Installation successful!')"
    ```

=== "Development"

    For development with editable installation:
    
    ### 1. Clone and Setup
    
    ```bash
    git clone https://github.com/Pol4720/mortality-ami-predictor.git
    cd mortality-ami-predictor/Tools
    ```
    
    ### 2. Install in Editable Mode
    
    ```bash
    pip install -e .
    pip install -r requirements.txt
    ```
    
    ### 3. Install Development Dependencies
    
    ```bash
    pip install pytest pytest-cov black flake8 mypy
    ```

## Quick Test

After installation, run a quick test to ensure everything works:

```bash
# Launch the dashboard
streamlit run dashboard/Dashboard.py
```

Your browser should open automatically with the dashboard. If not, navigate to `http://localhost:8501`.

## Configuration

### Environment Variables

Create a `.env` file in the `Tools` directory:

```bash
# Dataset path
DATASET_PATH=../DATA/recuima-020425.csv

# Experiment tracking
EXPERIMENT_TRACKER=mlflow
TRACKING_URI=./mlruns

# Target variables
TARGET_COLUMN=mortality_inhospital
ARRHYTHMIA_COLUMN=ventricular_arrhythmia
```

### MLflow Setup (Optional)

If using MLflow for experiment tracking:

```bash
# Start MLflow UI
mlflow ui --port 5000
```

Navigate to `http://localhost:5000` to view experiments.

## Troubleshooting

### Common Issues

!!! warning "ModuleNotFoundError"
    If you get `ModuleNotFoundError: No module named 'src'`:
    
    ```bash
    # Make sure you're in the Tools directory
    cd Tools
    
    # Install in development mode
    pip install -e .
    ```

!!! warning "Port Already in Use"
    If port 8501 is already in use for Streamlit:
    
    ```bash
    streamlit run dashboard/Dashboard.py --server.port 8502
    ```

!!! warning "Memory Error"
    If you encounter memory errors with large datasets:
    
    - Use the fragment dataset for testing: `DATA/recuima-020425-fragment.csv`
    - Increase system swap/virtual memory
    - Use data sampling in the dashboard

### Getting Help

If you encounter issues:

1. Check the [Developer Guide](../developer/index.md)
2. Search [existing issues](https://github.com/Pol4720/mortality-ami-predictor/issues)
3. Open a [new issue](https://github.com/Pol4720/mortality-ami-predictor/issues/new)

## Next Steps

Now that you have installed the system, check out:

- [Quick Start Guide](quickstart.md) - Get started with basic usage
- [Configuration](configuration.md) - Detailed configuration options
- [Dashboard Overview](../user-guide/dashboard.md) - Explore the dashboard features

---

!!! success "Installation Complete!"
    You're all set! Continue to the [Quick Start Guide](quickstart.md) to begin using the system.
