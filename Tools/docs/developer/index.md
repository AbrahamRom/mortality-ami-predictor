# Developer Guide

Contributing to the Mortality AMI Predictor project.

## Getting Started

### Development Setup

```bash
# Clone repository
git clone https://github.com/Pol4720/mortality-ami-predictor.git
cd mortality-ami-predictor

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
cd Tools
pip install -e .

# Install dev dependencies
pip install -r requirements-dev.txt
```

### Project Structure

```
mortality-ami-predictor/
├── DATA/                      # Raw datasets
├── Report/                    # LaTeX reports
├── Tools/
│   ├── src/                   # Source code
│   ├── tests/                 # Test suite
│   ├── dashboard/             # Streamlit app
│   ├── docs/                  # Documentation
│   ├── notebooks/             # Jupyter notebooks
│   ├── processed/             # Outputs
│   ├── requirements.txt       # Dependencies
│   └── setup.py              # Package setup
└── README.md
```

## Development Workflow

### 1. Create Feature Branch

```bash
git checkout -b feature/your-feature-name
```

### 2. Make Changes

Follow the [coding standards](#coding-standards) below.

### 3. Add Tests

```python
# tests/test_your_feature.py
import pytest
from src.your_module import your_function

def test_your_function():
    """Test your function."""
    result = your_function(input_data)
    assert result == expected_output
```

### 4. Run Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src tests/

# Run specific test
pytest tests/test_your_feature.py::test_your_function
```

### 5. Commit Changes

```bash
git add .
git commit -m "feat: add your feature description"
```

### 6. Push and Create PR

```bash
git push origin feature/your-feature-name
```

Then create a Pull Request on GitHub.

## Coding Standards

### Python Style Guide

Follow PEP 8:

```python
# Good
def calculate_risk_score(age: int, heart_rate: float) -> float:
    """Calculate patient risk score.
    
    Args:
        age: Patient age in years
        heart_rate: Heart rate in bpm
        
    Returns:
        Risk score between 0 and 1
    """
    normalized_age = age / 100
    normalized_hr = heart_rate / 200
    return (normalized_age + normalized_hr) / 2

# Bad
def calc(a,hr):
    return (a/100+hr/200)/2
```

### Type Hints

Use type hints for clarity:

```python
from typing import List, Dict, Optional, Union
import pandas as pd
import numpy as np

def preprocess_data(
    df: pd.DataFrame,
    columns: List[str],
    strategy: str = "mean",
    **kwargs
) -> pd.DataFrame:
    """Preprocess dataframe."""
    pass
```

### Docstrings

Use Google-style docstrings:

```python
def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_type: str = "random_forest",
    **params
) -> object:
    """Train a classification model.
    
    Args:
        X_train: Training features
        y_train: Training labels
        model_type: Type of model to train. One of:
            - "logistic": Logistic regression
            - "random_forest": Random forest classifier
            - "xgboost": XGBoost classifier
        **params: Additional model parameters
        
    Returns:
        Trained model object
        
    Raises:
        ValueError: If model_type is not recognized
        
    Example:
        >>> model = train_model(X_train, y_train, model_type="random_forest")
        >>> predictions = model.predict(X_test)
    """
    pass
```

## Testing

### Unit Tests

Test individual functions:

```python
def test_calculate_grace_score():
    """Test GRACE score calculation."""
    score = calculate_grace_score(
        age=65,
        heart_rate=85,
        systolic_bp=140,
        creatinine=1.2,
        killip_class=1,
        cardiac_arrest=False,
        st_segment_deviation=True,
        elevated_cardiac_enzymes=True
    )
    assert 100 <= score <= 200
```

### Integration Tests

Test module interactions:

```python
def test_full_pipeline():
    """Test complete training pipeline."""
    # Load data
    df = load_dataset("test_data.csv")
    
    # Clean
    cleaner = DataCleaner()
    df_clean = cleaner.clean(df)
    
    # Split
    X_train, X_test, y_train, y_test = split_data(df_clean)
    
    # Train
    trainer = ModelTrainer(model_type="logistic")
    model = trainer.train(X_train, y_train)
    
    # Evaluate
    metrics = calculate_all_metrics(model, X_test, y_test)
    
    assert metrics['auc'] > 0.5  # Better than random
```

### Fixtures

Use pytest fixtures for reusable test data:

```python
import pytest

@pytest.fixture
def sample_data():
    """Sample patient data for testing."""
    return pd.DataFrame({
        "age": [65, 70, 55],
        "sex": [1, 0, 1],
        "heart_rate": [85, 92, 78],
        "mortality": [0, 1, 0]
    })

def test_cleaner(sample_data):
    """Test data cleaner with sample data."""
    cleaner = DataCleaner()
    result = cleaner.clean(sample_data)
    assert len(result) == len(sample_data)
```

## Documentation

### Inline Comments

```python
# Calculate normalized risk score
# Age contributes 60%, heart rate 40%
risk = (age / 100) * 0.6 + (heart_rate / 200) * 0.4
```

### Module Docstrings

```python
"""Data cleaning module.

This module provides tools for cleaning medical datasets:
- Missing value imputation
- Outlier detection and handling
- Categorical encoding
- Feature discretization

Example:
    >>> from src.cleaning import DataCleaner
    >>> cleaner = DataCleaner()
    >>> clean_df = cleaner.clean(raw_df)
"""
```

### Update Documentation

After changes:

```bash
cd Tools
python generate_api_docs.py
mkdocs serve
```

## Adding New Features

### New Model Type

1. Create model class:
```python
# src/models/classifiers.py
from sklearn.base import BaseEstimator, ClassifierMixin

class MyNewModel(BaseEstimator, ClassifierMixin):
    def __init__(self, param1=1.0):
        self.param1 = param1
    
    def fit(self, X, y):
        # Training logic
        return self
    
    def predict(self, X):
        # Prediction logic
        pass
```

2. Register:
```python
# src/models/registry.py
from .classifiers import MyNewModel

ModelRegistry.register_model("my_new_model", MyNewModel)
```

3. Add tests:
```python
# tests/test_my_new_model.py
def test_my_new_model():
    model = MyNewModel(param1=2.0)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    assert len(predictions) == len(y_test)
```

### New Dashboard Page

1. Create page:
```python
# dashboard/pages/08_🆕_My_New_Page.py
import streamlit as st

st.title("My New Feature")

# Your page logic here
```

2. Page automatically appears in dashboard!

## Debugging

### Using pdb

```python
import pdb

def problematic_function(data):
    # Set breakpoint
    pdb.set_trace()
    
    # Debug from here
    result = process(data)
    return result
```

### Logging

```python
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def train_model(X, y):
    logger.info(f"Training with {len(X)} samples")
    logger.debug(f"Feature columns: {X.columns.tolist()}")
    
    # Training logic
    
    logger.info("Training complete")
```

## Performance Profiling

### Time Profiling

```python
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

# Code to profile
train_model(X_train, y_train)

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumtime')
stats.print_stats(10)  # Top 10 slowest functions
```

### Memory Profiling

```python
from memory_profiler import profile

@profile
def memory_intensive_function():
    # Function logic
    pass
```

## Common Tasks

### Add New Metric

```python
# src/evaluation/metrics.py

def calculate_custom_metric(y_true, y_pred):
    """Calculate custom metric.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Metric value
    """
    # Calculation logic
    return metric_value

# Update calculate_all_metrics()
def calculate_all_metrics(model, X, y):
    metrics = {
        # ... existing metrics
        "custom_metric": calculate_custom_metric(y, y_pred)
    }
    return metrics
```

### Add Preprocessing Step

```python
# src/preprocessing/pipelines.py

from sklearn.preprocessing import FunctionTransformer

def custom_transform(X):
    """Custom transformation."""
    # Transform logic
    return X_transformed

# Add to pipeline
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("custom", FunctionTransformer(custom_transform)),
    ("model", model)
])
```

## Continuous Integration

Our CI/CD pipeline:

```yaml
# .github/workflows/ci.yml
name: CI

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
      - run: pip install -r requirements.txt
      - run: pytest --cov=src tests/
      
  docs:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - run: pip install -r docs-requirements.txt
      - run: mkdocs build
```

## Release Process

1. Update version in `setup.py`
2. Update `CHANGELOG.md`
3. Create git tag:
   ```bash
   git tag -a v1.0.0 -m "Release version 1.0.0"
   git push origin v1.0.0
   ```
4. GitHub Actions automatically builds and deploys

## Getting Help

- **Issues**: GitHub Issues for bugs/features
- **Discussions**: GitHub Discussions for questions
- **Email**: pol.manriquez@example.com

## See Also

- [Architecture](../architecture/)
- [Testing Guide](../../tests/TESTING_SUMMARY.md)
- [API Reference](../api/)
