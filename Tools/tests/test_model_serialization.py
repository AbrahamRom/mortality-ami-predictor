"""
Comprehensive tests for model serialization and metadata.

Tests cover:
- sklearn models serialization
- Neural networks serialization
- Custom models serialization
- Metadata creation and validation
- JSON serialization/deserialization
- Model persistence
"""

import pytest
import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import joblib

from src.models.classifiers import make_classifiers
from src.models.metadata import ModelMetadata, create_metadata_from_training
from src.preprocessing import build_preprocess_pipelines
from sklearn.pipeline import Pipeline

# Check if torch is available (optional dependency)
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Skip decorator for tests requiring torch
requires_torch = pytest.mark.skipif(
    not TORCH_AVAILABLE,
    reason="PyTorch (torch) is not installed - skipping neural network tests"
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_data():
    """Generate sample data for training."""
    np.random.seed(42)
    X = pd.DataFrame({
        'age': np.random.randint(40, 90, size=100),
        'sbp': np.random.normal(120, 15, size=100),
        'sex': np.random.choice(['M', 'F'], size=100),
    })
    y = np.random.randint(0, 2, size=100)
    return X, y


# ============================================================================
# Test sklearn models serialization
# ============================================================================

class TestSklearnModelSerialization:
    """Test serialization of sklearn-based models."""
    
    def test_all_sklearn_models_metadata_creation(self):
        """Test that all sklearn models can create metadata."""
        models = make_classifiers()
        
        for model_name, (model, param_grid) in models.items():
            # Get model parameters
            params = model.get_params() if hasattr(model, 'get_params') else {}
            
            # Create metadata
            metadata = ModelMetadata(
                model_name=model_name,
                model_type=type(model).__name__,
                task='classification',
                hyperparameters=params
            )
            
            # Verify metadata is created
            assert metadata is not None
            assert metadata.model_name == model_name
            assert metadata.task == 'classification'
    
    def test_sklearn_models_json_serialization(self):
        """Test that all sklearn models can be serialized to JSON."""
        models = make_classifiers()
        
        for model_name, (model, param_grid) in models.items():
            params = model.get_params() if hasattr(model, 'get_params') else {}
            
            metadata = ModelMetadata(
                model_name=model_name,
                model_type=type(model).__name__,
                task='classification',
                hyperparameters=params
            )
            
            # Convert to dict
            metadata_dict = metadata.to_dict()
            assert isinstance(metadata_dict, dict)
            
            # Serialize to JSON (use default=str for non-serializable values)
            json_str = json.dumps(metadata_dict, indent=2, ensure_ascii=False, default=str)
            assert len(json_str) > 0
            
            # Deserialize back
            loaded_dict = json.loads(json_str)
            
            # Basic structure checks (don't compare NaN values exactly)
            assert loaded_dict['model_name'] == model_name
            assert loaded_dict['task'] == 'classification'
            assert 'hyperparameters' in loaded_dict
    
    def test_sklearn_model_pipeline_persistence(self, sample_data, tmp_path):
        """Test saving and loading sklearn model pipelines."""
        X, y = sample_data
        
        # Build preprocessing pipeline
        pre, _ = build_preprocess_pipelines(X)
        
        # Get first classifier
        name, (clf, grid) = next(iter(make_classifiers().items()))
        
        # Create pipeline
        pipe = Pipeline(steps=[('pre', pre), ('clf', clf)])
        
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning)
            pipe.fit(X, y)
        
        # Save model
        path = tmp_path / 'model.joblib'
        joblib.dump(pipe, path)
        assert path.exists()
        
        # Load model
        loaded = joblib.load(path)
        assert loaded is not None
        
        # Verify predictions work
        preds = loaded.predict_proba(X)[:, 1]
        assert preds.shape[0] == len(y)
        assert np.all((preds >= 0) & (preds <= 1))


# ============================================================================
# Test neural networks serialization
# ============================================================================

@requires_torch
class TestNeuralNetworkSerialization:
    """Test serialization of neural network models."""
    
    def test_neural_network_import(self):
        """Test that neural network module can be imported."""
        try:
            from src.models.neural_networks import TorchTabularClassifier
            assert TorchTabularClassifier is not None
        except ImportError as e:
            pytest.fail(f"Failed to import TorchTabularClassifier: {e}")
    
    def test_neural_network_instantiation(self):
        """Test that neural networks can be instantiated."""
        from src.models.neural_networks import TorchTabularClassifier
        
        configs = [
            {},  # Default config
            {"hidden": 128, "dropout": 0.3},
            {"in_dim": 50, "hidden": 256, "dropout": 0.5, "epochs": 100},
        ]
        
        for config in configs:
            model = TorchTabularClassifier(**config)
            assert model is not None
            assert hasattr(model, 'in_dim')
            assert hasattr(model, 'hidden')
            assert hasattr(model, 'dropout')
    
    def test_neural_network_metadata_creation(self):
        """Test metadata creation for neural networks."""
        from src.models.neural_networks import TorchTabularClassifier
        
        model = TorchTabularClassifier(hidden=128, dropout=0.3, epochs=100)
        
        # Get parameters
        params = {
            'in_dim': model.in_dim,
            'hidden': model.hidden,
            'dropout': model.dropout,
            'lr': model.lr,
            'epochs': model.epochs,
            'batch_size': model.batch_size,
            'focal_loss': model.focal
        }
        
        # Create metadata
        metadata = ModelMetadata(
            model_name='nn_test',
            model_type='TorchTabularClassifier',
            task='classification',
            hyperparameters=params
        )
        
        assert metadata is not None
        assert metadata.model_type == 'TorchTabularClassifier'
    
    def test_neural_network_json_serialization(self):
        """Test JSON serialization for neural networks."""
        from src.models.neural_networks import TorchTabularClassifier
        
        configs = [
            ("nn_default", {}),
            ("nn_custom", {"hidden": 128, "dropout": 0.3, "lr": 0.001}),
            ("nn_deep", {"in_dim": 50, "hidden": 256, "dropout": 0.5}),
        ]
        
        for model_name, kwargs in configs:
            model = TorchTabularClassifier(**kwargs)
            
            params = {
                'in_dim': model.in_dim,
                'hidden': model.hidden,
                'dropout': model.dropout,
                'lr': model.lr,
                'epochs': model.epochs,
                'batch_size': model.batch_size,
                'focal_loss': model.focal
            }
            
            metadata = ModelMetadata(
                model_name=model_name,
                model_type='TorchTabularClassifier',
                task='classification',
                hyperparameters=params
            )
            
            # Convert to dict
            metadata_dict = metadata.to_dict()
            assert isinstance(metadata_dict, dict)
            
            # Serialize to JSON
            json_str = json.dumps(metadata_dict, indent=2, ensure_ascii=False)
            assert len(json_str) > 0
            
            # Deserialize back
            loaded_dict = json.loads(json_str)
            assert loaded_dict == metadata_dict
    
    def test_neural_network_training_and_persistence(self, sample_data, tmp_path):
        """Test training and saving neural networks."""
        from src.models.neural_networks import TorchTabularClassifier
        
        X, y = sample_data
        X_np = X.select_dtypes(include=[np.number]).values
        
        model = TorchTabularClassifier(in_dim=X_np.shape[1], epochs=10)
        
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            model.fit(X_np, y)
        
        # Verify model is fitted
        assert model.is_fitted_
        
        # Test predictions
        preds = model.predict_proba(X_np)
        assert preds.shape[0] == len(y)
        assert preds.shape[1] == 2
        
        # Save model
        path = tmp_path / 'nn_model.joblib'
        joblib.dump(model, path)
        assert path.exists()
        
        # Load model
        loaded = joblib.load(path)
        assert loaded is not None
        
        # Verify loaded model works
        loaded_preds = loaded.predict_proba(X_np)
        np.testing.assert_array_almost_equal(preds, loaded_preds, decimal=5)


# ============================================================================
# Test custom models serialization
# ============================================================================

class TestCustomModelSerialization:
    """Test serialization of custom models."""
    
    def test_custom_base_classes_import(self):
        """Test that custom base classes can be imported."""
        from src.models.custom_base import (
            BaseCustomModel,
            BaseCustomClassifier,
            BaseCustomRegressor
        )
        assert BaseCustomModel is not None
        assert BaseCustomClassifier is not None
        assert BaseCustomRegressor is not None
    
    def test_custom_model_creation(self):
        """Test creating a custom model."""
        from src.models.custom_base import BaseCustomClassifier
        
        class DummyCustomModel(BaseCustomClassifier):
            def __init__(self, threshold=0.5, weights=None):
                super().__init__(name="DummyModel")
                self.threshold = threshold
                self.weights = weights if weights is not None else [1.0, 2.0, 3.0]
            
            def fit(self, X, y, **kwargs):
                self._validate_input(X, training=True)
                self.classes_ = np.unique(y)
                self.is_fitted_ = True
                return self
            
            def predict_proba(self, X):
                if not self.is_fitted_:
                    raise ValueError("Model must be fitted first")
                n_samples = X.shape[0]
                return np.column_stack([
                    np.ones(n_samples) * 0.5,
                    np.ones(n_samples) * 0.5
                ])
            
            def predict(self, X):
                proba = self.predict_proba(X)
                return (proba[:, 1] > self.threshold).astype(int)
        
        model = DummyCustomModel(threshold=0.7)
        assert model is not None
        assert model.threshold == 0.7
        assert len(model.weights) == 3
    
    def test_custom_model_metadata_creation(self):
        """Test metadata creation for custom models."""
        from src.models.custom_base import BaseCustomClassifier
        
        class DummyCustomModel(BaseCustomClassifier):
            def __init__(self, threshold=0.5):
                super().__init__(name="DummyModel")
                self.threshold = threshold
            
            def fit(self, X, y, **kwargs):
                self.classes_ = np.unique(y)
                self.is_fitted_ = True
                return self
            
            def predict_proba(self, X):
                n_samples = X.shape[0] if hasattr(X, 'shape') else len(X)
                return np.column_stack([
                    np.ones(n_samples) * 0.5,
                    np.ones(n_samples) * 0.5
                ])
        
        model = DummyCustomModel(threshold=0.7)
        
        params = {'threshold': model.threshold, 'name': model.name}
        
        metadata = ModelMetadata(
            model_name='custom_dummy',
            model_type='DummyCustomModel',
            task='classification',
            hyperparameters=params
        )
        
        assert metadata is not None
        assert metadata.model_type == 'DummyCustomModel'
    
    def test_custom_model_json_serialization(self):
        """Test JSON serialization for custom models."""
        from src.models.custom_base import BaseCustomClassifier
        
        class DummyCustomModel(BaseCustomClassifier):
            def __init__(self, threshold=0.5, weights=None):
                super().__init__(name="DummyModel")
                self.threshold = threshold
                self.weights = weights if weights is not None else [1.0, 2.0]
            
            def fit(self, X, y, **kwargs):
                self.classes_ = np.unique(y)
                self.is_fitted_ = True
                return self
            
            def predict_proba(self, X):
                n_samples = X.shape[0] if hasattr(X, 'shape') else len(X)
                return np.column_stack([
                    np.ones(n_samples) * 0.5,
                    np.ones(n_samples) * 0.5
                ])
        
        configs = [
            ("custom_default", {}),
            ("custom_weighted", {"threshold": 0.7, "weights": [1.0, 2.0, 3.0]}),
        ]
        
        for model_name, kwargs in configs:
            model = DummyCustomModel(**kwargs)
            
            params = {
                'threshold': model.threshold,
                'weights': model.weights,
                'name': model.name
            }
            
            metadata = ModelMetadata(
                model_name=model_name,
                model_type='DummyCustomModel',
                task='classification',
                hyperparameters=params
            )
            
            # Convert to dict
            metadata_dict = metadata.to_dict()
            assert isinstance(metadata_dict, dict)
            
            # Serialize to JSON
            json_str = json.dumps(metadata_dict, indent=2, ensure_ascii=False)
            assert len(json_str) > 0
            
            # Deserialize back
            loaded_dict = json.loads(json_str)
            assert loaded_dict == metadata_dict
    
    def test_custom_model_persistence(self, sample_data, tmp_path):
        """Test saving and loading custom models."""
        from src.models.custom_base import SimpleMLPClassifier
        
        X, y = sample_data
        X_np = X.select_dtypes(include=[np.number]).values
        
        # Create and train model
        model = SimpleMLPClassifier(hidden_layer_sizes=(32,), learning_rate=0.01, epochs=10)
        
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            model.fit(X_np, y)
        
        # Verify model is fitted
        assert model.is_fitted_
        
        # Test predictions
        preds = model.predict_proba(X_np)
        assert preds.shape[0] == len(y)
        
        # Save model
        path = tmp_path / 'custom_model.joblib'
        joblib.dump(model, path)
        assert path.exists()
        
        # Load model
        loaded = joblib.load(path)
        assert loaded is not None
        assert loaded.is_fitted_
        
        # Verify loaded model works
        loaded_preds = loaded.predict_proba(X_np)
        np.testing.assert_array_almost_equal(preds, loaded_preds, decimal=5)


# ============================================================================
# Test metadata creation from training
# ============================================================================

class TestMetadataFromTraining:
    """Test metadata creation from training results."""
    
    def test_create_metadata_from_training_basic(self, sample_data):
        """Test basic metadata creation from training."""
        X, y = sample_data
        
        # Get a simple model
        models = make_classifiers()
        model_name, (model, _) = list(models.items())[0]
        
        # Train model
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            model.fit(X.select_dtypes(include=[np.number]), y)
        
        # Create DataFrames for the function
        X_numeric = X.select_dtypes(include=[np.number])
        train_df = X_numeric.copy()
        train_df['target'] = y
        test_df = train_df.copy()  # Using same for test
        
        # Create metadata with proper signature
        metadata = create_metadata_from_training(
            model_name=model_name,
            model_type=type(model).__name__,
            task='classification',
            model_file_path='/tmp/model.joblib',
            train_set_path='/tmp/train.csv',
            test_set_path='/tmp/test.csv',
            train_df=train_df,
            test_df=test_df,
            target_column='target',
            hyperparameters=model.get_params(),
            cv_results={'mean_score': 0.85, 'std_score': 0.05, 'all_scores': [0.8, 0.85, 0.9]},
            training_config={'cv_strategy': 'StratifiedKFold', 'n_splits': 5, 'n_repeats': 1, 'total_runs': 5}
        )
        
        assert metadata is not None
        assert metadata.model_name == model_name
        assert metadata.task == 'classification'
    
    def test_metadata_serialization_roundtrip(self, sample_data):
        """Test complete metadata serialization and deserialization."""
        X, y = sample_data
        
        models = make_classifiers()
        model_name, (model, _) = list(models.items())[0]
        
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            model.fit(X.select_dtypes(include=[np.number]), y)
        
        # Create DataFrames for the function
        X_numeric = X.select_dtypes(include=[np.number])
        train_df = X_numeric.copy()
        train_df['target'] = y
        test_df = train_df.copy()
        
        # Create metadata with proper signature
        metadata = create_metadata_from_training(
            model_name=model_name,
            model_type=type(model).__name__,
            task='classification',
            model_file_path='/tmp/model.joblib',
            train_set_path='/tmp/train.csv',
            test_set_path='/tmp/test.csv',
            train_df=train_df,
            test_df=test_df,
            target_column='target',
            hyperparameters=model.get_params(),
            cv_results={'mean_score': 0.85, 'std_score': 0.05, 'all_scores': [0.8, 0.85, 0.9]},
            training_config={'cv_strategy': 'StratifiedKFold', 'n_splits': 5, 'n_repeats': 1, 'total_runs': 5}
        )
        
        # Serialize to JSON
        json_str = json.dumps(metadata.to_dict(), indent=2)
        assert len(json_str) > 0
        
        # Deserialize
        loaded_dict = json.loads(json_str)
        assert loaded_dict['model_name'] == model_name
        assert loaded_dict['task'] == 'classification'


# ============================================================================
# Integration tests
# ============================================================================

class TestSerializationIntegration:
    """Integration tests for serialization across the pipeline."""
    
    def test_end_to_end_sklearn_pipeline(self, sample_data, tmp_path):
        """Test complete sklearn pipeline with serialization."""
        X, y = sample_data
        
        # Build preprocessing
        pre, _ = build_preprocess_pipelines(X)
        
        # Get models
        models = make_classifiers()
        
        for model_name, (model, _) in list(models.items())[:3]:  # Test first 3
            # Create pipeline
            pipe = Pipeline(steps=[('pre', pre), ('clf', model)])
            
            # Train
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                pipe.fit(X, y)
            
            # Create metadata
            params = model.get_params() if hasattr(model, 'get_params') else {}
            metadata = ModelMetadata(
                model_name=model_name,
                model_type=type(model).__name__,
                task='classification',
                hyperparameters=params
            )
            
            # Save both model and metadata
            model_path = tmp_path / f'{model_name}.joblib'
            metadata_path = tmp_path / f'{model_name}_metadata.json'
            
            joblib.dump(pipe, model_path)
            with open(metadata_path, 'w') as f:
                json.dump(metadata.to_dict(), f, indent=2)
            
            assert model_path.exists()
            assert metadata_path.exists()
            
            # Load both
            loaded_pipe = joblib.load(model_path)
            with open(metadata_path, 'r') as f:
                loaded_metadata = json.load(f)
            
            # Verify
            preds = loaded_pipe.predict_proba(X)[:, 1]
            assert preds.shape[0] == len(y)
            assert loaded_metadata['model_name'] == model_name
    
    def test_mixed_model_types_serialization(self, sample_data, tmp_path):
        """Test serialization of mixed model types."""
        X, y = sample_data
        X_np = X.select_dtypes(include=[np.number]).values
        
        models_to_test = []
        
        # Add sklearn model
        from sklearn.ensemble import RandomForestClassifier
        rf = RandomForestClassifier(n_estimators=10, random_state=42)
        models_to_test.append(('sklearn_rf', rf))
        
        # Add neural network (only if torch is available)
        if TORCH_AVAILABLE:
            try:
                from src.models.neural_networks import TorchTabularClassifier
                nn = TorchTabularClassifier(in_dim=X_np.shape[1], epochs=5)
                models_to_test.append(('neural_net', nn))
            except ImportError:
                pass
        
        # Add custom model
        try:
            from src.models.custom_base import SimpleMLPClassifier
            custom = SimpleMLPClassifier(hidden_layer_sizes=(16,), epochs=5)
            models_to_test.append(('custom_mlp', custom))
        except ImportError:
            pass
        
        # Train and save all models
        for model_name, model in models_to_test:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                model.fit(X_np, y)
            
            # Save model
            path = tmp_path / f'{model_name}.joblib'
            joblib.dump(model, path)
            assert path.exists()
            
            # Verify it can be loaded
            loaded = joblib.load(path)
            preds = loaded.predict_proba(X_np)
            assert preds.shape[0] == len(y)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
