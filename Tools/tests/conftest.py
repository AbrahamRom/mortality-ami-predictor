"""
Shared pytest fixtures and configuration for all tests.
"""

import pytest
import warnings
import numpy as np
import pandas as pd
import tempfile
from pathlib import Path


# Configure pytest to handle warnings properly
def pytest_configure(config):
    """Configure pytest with custom settings."""
    # Suppress specific warnings
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=PendingDeprecationWarning)


# ============================================================================
# Data fixtures
# ============================================================================

@pytest.fixture(scope='session')
def random_seed():
    """Fixed random seed for reproducible tests."""
    return 42


@pytest.fixture
def simple_dataset(random_seed):
    """Generate simple synthetic dataset for quick tests."""
    np.random.seed(random_seed)
    
    n_samples = 100
    X = pd.DataFrame({
        'age': np.random.randint(40, 90, n_samples),
        'sbp': np.random.normal(120, 15, n_samples),
        'sex': np.random.choice(['M', 'F'], n_samples),
    })
    y = np.random.randint(0, 2, n_samples)
    
    return X, y


@pytest.fixture
def medium_dataset(random_seed):
    """Generate medium-sized synthetic dataset."""
    np.random.seed(random_seed)
    
    n_samples = 500
    n_features = 10
    
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    y = np.random.randint(0, 2, n_samples)
    
    return X, y


@pytest.fixture
def clinical_like_dataset(random_seed):
    """Generate dataset mimicking clinical data structure."""
    np.random.seed(random_seed)
    
    n_samples = 300
    
    data = pd.DataFrame({
        'age': np.random.randint(40, 90, n_samples),
        'sex': np.random.choice(['M', 'F'], n_samples),
        'sbp': np.random.normal(120, 20, n_samples),
        'dbp': np.random.normal(80, 15, n_samples),
        'heart_rate': np.random.randint(60, 120, n_samples),
        'glucose': np.random.normal(100, 30, n_samples),
        'creatinine': np.random.normal(1.0, 0.5, n_samples),
        'ejection_fraction': np.random.randint(30, 70, n_samples),
        'killip_class': np.random.choice([1, 2, 3, 4], n_samples),
        'has_diabetes': np.random.choice([0, 1], n_samples),
        'has_hypertension': np.random.choice([0, 1], n_samples),
        'mortality': np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
    })
    
    # Add some missing values
    data.loc[np.random.choice(data.index, 20), 'glucose'] = np.nan
    data.loc[np.random.choice(data.index, 10), 'creatinine'] = np.nan
    
    return data


@pytest.fixture
def dataset_with_outliers(random_seed):
    """Generate dataset with known outliers."""
    np.random.seed(random_seed)
    
    n_samples = 200
    
    X = pd.DataFrame({
        'normal_feature': np.random.normal(0, 1, n_samples),
        'feature_with_outliers': np.random.normal(0, 1, n_samples),
    })
    
    # Add outliers
    X.loc[np.random.choice(X.index, 5), 'feature_with_outliers'] = 10
    X.loc[np.random.choice(X.index, 5), 'feature_with_outliers'] = -10
    
    y = np.random.randint(0, 2, n_samples)
    
    return X, y


# ============================================================================
# Model fixtures
# ============================================================================

@pytest.fixture
def trained_simple_model(simple_dataset):
    """Return a trained simple sklearn model."""
    from sklearn.ensemble import RandomForestClassifier
    
    X, y = simple_dataset
    X_numeric = X.select_dtypes(include=[np.number])
    
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        model.fit(X_numeric, y)
    
    return model


@pytest.fixture
def trained_pipeline(simple_dataset):
    """Return a trained preprocessing + model pipeline."""
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier
    
    X, y = simple_dataset
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(n_estimators=10, random_state=42))
    ])
    
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        pipeline.fit(X.select_dtypes(include=[np.number]), y)
    
    return pipeline


# ============================================================================
# File/Path fixtures
# ============================================================================

@pytest.fixture
def temp_dir():
    """Create a temporary directory that cleans up after test."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def temp_model_path(temp_dir):
    """Return a temporary path for model saving."""
    return temp_dir / 'model.joblib'


@pytest.fixture
def temp_data_path(temp_dir):
    """Return a temporary path for data saving."""
    return temp_dir / 'data.csv'


# ============================================================================
# Utility fixtures
# ============================================================================

@pytest.fixture
def suppress_warnings():
    """Context manager to suppress all warnings."""
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        yield


# ============================================================================
# Marks and parametrization helpers
# ============================================================================

def pytest_collection_modifyitems(config, items):
    """Add markers to tests based on their names."""
    for item in items:
        # Add integration marker to integration tests
        if 'integration' in item.nodeid or 'end_to_end' in item.nodeid:
            item.add_marker(pytest.mark.integration)
        
        # Add unit marker to simple unit tests
        elif 'test_data_cleaning' in item.nodeid or 'test_model_io' in item.nodeid:
            item.add_marker(pytest.mark.unit)
        
        # Add serialization marker
        if 'serialization' in item.nodeid or 'metadata' in item.nodeid:
            item.add_marker(pytest.mark.serialization)
        
        # Add pipeline marker
        if 'pipeline' in item.nodeid or 'workflow' in item.nodeid:
            item.add_marker(pytest.mark.pipeline)


# ============================================================================
# Custom assertions
# ============================================================================

class CustomAssertions:
    """Custom assertion helpers for tests."""
    
    @staticmethod
    def assert_valid_predictions(predictions, n_samples, n_classes=2):
        """Assert predictions are valid."""
        assert len(predictions) == n_samples
        assert predictions.shape[1] == n_classes if predictions.ndim > 1 else True
        if predictions.ndim > 1:
            assert np.all((predictions >= 0) & (predictions <= 1))
            assert np.allclose(predictions.sum(axis=1), 1.0)
    
    @staticmethod
    def assert_valid_metrics(metrics, required_metrics=None):
        """Assert metrics dictionary is valid."""
        if required_metrics is None:
            required_metrics = ['accuracy', 'roc_auc']
        
        for metric in required_metrics:
            assert metric in metrics, f"Missing metric: {metric}"
            assert 0 <= metrics[metric] <= 1, f"Invalid metric value: {metrics[metric]}"
    
    @staticmethod
    def assert_model_is_fitted(model):
        """Assert model is fitted."""
        assert (
            hasattr(model, 'classes_') or 
            hasattr(model, 'is_fitted_') or
            hasattr(model, '_is_fitted')
        ), "Model is not fitted"


@pytest.fixture
def custom_assertions():
    """Provide custom assertions helper."""
    return CustomAssertions()


# ============================================================================
# Performance monitoring
# ============================================================================

@pytest.fixture
def performance_monitor():
    """Monitor test performance."""
    import time
    
    start_time = time.time()
    
    yield
    
    duration = time.time() - start_time
    
    if duration > 10:
        warnings.warn(f"Test took {duration:.2f} seconds (>10s threshold)")


# ============================================================================
# Cleanup hooks
# ============================================================================

@pytest.fixture(autouse=True)
def cleanup_after_test():
    """Automatically clean up after each test."""
    yield
    
    # Clean up any remaining warnings
    warnings.resetwarnings()


# ============================================================================
# Test data generators
# ============================================================================

@pytest.fixture
def data_generator():
    """Factory fixture for generating custom datasets."""
    
    def _generate(n_samples=100, n_features=5, n_classes=2, 
                   has_missing=False, has_outliers=False, random_state=42):
        """Generate custom synthetic dataset."""
        np.random.seed(random_state)
        
        X = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        
        y = np.random.randint(0, n_classes, n_samples)
        
        if has_missing:
            for col in X.columns[:2]:
                X.loc[np.random.choice(X.index, 5), col] = np.nan
        
        if has_outliers:
            X.iloc[0, 0] = 100  # Add outlier
            X.iloc[1, 0] = -100
        
        return X, y
    
    return _generate
