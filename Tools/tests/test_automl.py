"""Unit tests for AutoML module.

Tests cover configuration, model wrappers, suggestions engine, and export utilities.
"""
import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_classification_data():
    """Generate sample classification data for testing."""
    X, y = make_classification(
        n_samples=200,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        n_classes=2,
        random_state=42,
        weights=[0.7, 0.3],  # Imbalanced
    )
    X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(10)])
    y = pd.Series(y, name="target")
    return X, y


@pytest.fixture
def sample_dataframe():
    """Generate sample DataFrame with various data types."""
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        'numeric1': np.random.randn(n),
        'numeric2': np.random.randn(n) * 10,
        'numeric3': np.random.exponential(2, n),  # Skewed
        'cat1': np.random.choice(['A', 'B', 'C'], n),
        'cat2': np.random.choice(['X', 'Y'], n),
        'target': np.random.choice([0, 1], n, p=[0.8, 0.2]),  # Imbalanced
    })
    # Add some missing values
    df.loc[np.random.choice(n, 5), 'numeric1'] = np.nan
    df.loc[np.random.choice(n, 3), 'cat1'] = np.nan
    return df


@pytest.fixture
def temp_output_dir():
    """Create temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


# ============================================================================
# Test AutoML Configuration
# ============================================================================

class TestAutoMLConfig:
    """Tests for AutoMLConfig class."""
    
    def test_config_creation_default(self):
        """Test default configuration creation."""
        try:
            from src.automl import AutoMLConfig
            
            config = AutoMLConfig()
            
            assert config.time_left_for_this_task == 3600
            assert config.per_run_time_limit == 360
            assert config.memory_limit == 8192
            assert config.n_jobs == -1
            assert config.metric == "roc_auc"
        except ImportError:
            pytest.skip("AutoML module not available")
    
    def test_config_from_preset_quick(self):
        """Test quick preset configuration."""
        try:
            from src.automl import AutoMLConfig, AutoMLPreset
            
            config = AutoMLConfig.from_preset(AutoMLPreset.QUICK)
            
            assert config.time_left_for_this_task == 300
            assert config.ensemble_size == 10  # Quick preset uses smaller ensemble
        except ImportError:
            pytest.skip("AutoML module not available")
    
    def test_config_from_preset_balanced(self):
        """Test balanced preset configuration."""
        try:
            from src.automl import AutoMLConfig, AutoMLPreset
            
            config = AutoMLConfig.from_preset(AutoMLPreset.BALANCED)
            
            assert config.time_left_for_this_task == 3600
            assert config.ensemble_size == 50
        except ImportError:
            pytest.skip("AutoML module not available")
    
    def test_config_from_preset_high_performance(self):
        """Test high performance preset configuration."""
        try:
            from src.automl import AutoMLConfig, AutoMLPreset
            
            config = AutoMLConfig.from_preset(AutoMLPreset.HIGH_PERFORMANCE)
            
            assert config.time_left_for_this_task == 14400
            assert config.ensemble_size == 100
        except ImportError:
            pytest.skip("AutoML module not available")
    
    def test_config_custom_values(self):
        """Test custom configuration values."""
        try:
            from src.automl import AutoMLConfig
            
            config = AutoMLConfig(
                time_left_for_this_task=600,
                per_run_time_limit=120,
                memory_limit=4096,
                ensemble_size=25,
                metric="f1",
            )
            
            assert config.time_left_for_this_task == 600
            assert config.per_run_time_limit == 120
            assert config.memory_limit == 4096
            assert config.ensemble_size == 25
            assert config.metric == "f1"
        except ImportError:
            pytest.skip("AutoML module not available")
    
    def test_config_to_flaml_kwargs(self):
        """Test conversion to FLAML kwargs."""
        try:
            from src.automl import AutoMLConfig
            
            config = AutoMLConfig(
                time_left_for_this_task=300,
                metric="roc_auc",
            )
            
            kwargs = config.to_flaml_kwargs()
            
            assert "time_budget" in kwargs
            assert kwargs["time_budget"] == 300
            assert "metric" in kwargs
            assert kwargs["metric"] == "roc_auc"
        except ImportError:
            pytest.skip("AutoML module not available")


# ============================================================================
# Test Dataset Analysis
# ============================================================================

class TestDatasetAnalysis:
    """Tests for dataset analysis functionality."""
    
    def test_analyze_dataset_basic(self, sample_dataframe):
        """Test basic dataset analysis."""
        try:
            from src.automl import analyze_dataset
            
            analysis = analyze_dataset(sample_dataframe, target_column="target")
            
            assert analysis.n_samples == 100
            assert analysis.n_features == 5  # Excluding target
            assert hasattr(analysis, 'imbalance_ratio')
            assert hasattr(analysis, 'is_imbalanced')
            assert hasattr(analysis, 'missing_percentage')
        except ImportError:
            pytest.skip("AutoML module not available")
    
    def test_analyze_imbalanced_dataset(self):
        """Test detection of imbalanced dataset."""
        try:
            from src.automl import analyze_dataset
            
            # Highly imbalanced dataset
            df = pd.DataFrame({
                'feature': np.random.randn(100),
                'target': [0] * 95 + [1] * 5,  # 95-5 split
            })
            
            analysis = analyze_dataset(df, target_column="target")
            
            assert analysis.is_imbalanced
            assert analysis.imbalance_ratio > 5
        except ImportError:
            pytest.skip("AutoML module not available")
    
    def test_analyze_missing_data(self):
        """Test detection of missing data."""
        try:
            from src.automl import analyze_dataset
            
            df = pd.DataFrame({
                'feature1': [1, 2, np.nan, 4, 5],
                'feature2': [np.nan, np.nan, 3, 4, 5],
                'target': [0, 1, 0, 1, 0],
            })
            
            analysis = analyze_dataset(df, target_column="target")
            
            assert analysis.missing_percentage > 0
            assert len(analysis.features_with_missing) > 0
        except ImportError:
            pytest.skip("AutoML module not available")
    
    def test_analyze_skewed_features(self, sample_dataframe):
        """Test detection of skewed features."""
        try:
            from src.automl import analyze_dataset
            
            analysis = analyze_dataset(sample_dataframe, target_column="target")
            
            # numeric3 is exponentially distributed (skewed)
            assert len(analysis.skewed_features) > 0
        except ImportError:
            pytest.skip("AutoML module not available")


# ============================================================================
# Test Suggestions Engine
# ============================================================================

class TestSuggestions:
    """Tests for the suggestions engine."""
    
    def test_get_suggestions_imbalanced(self):
        """Test suggestions for imbalanced data."""
        try:
            from src.automl import get_suggestions
            
            df = pd.DataFrame({
                'feature': np.random.randn(100),
                'target': [0] * 90 + [1] * 10,
            })
            
            suggestions = get_suggestions(df, target_column="target")
            
            # Should suggest SMOTE or class_weight
            imbalance_suggestions = [
                s for s in suggestions 
                if 'smote' in s.title.lower() or 'imbal' in s.title.lower() or 'class' in s.title.lower()
            ]
            assert len(imbalance_suggestions) > 0
        except ImportError:
            pytest.skip("AutoML module not available")
    
    def test_get_suggestions_missing_data(self):
        """Test suggestions for missing data."""
        try:
            from src.automl import get_suggestions
            
            # Use low missing rate to trigger imputation suggestions
            df = pd.DataFrame({
                'feature1': [1, 2, 3, np.nan, 5, 6, 7, 8, 9, 10],
                'feature2': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                'target': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            })
            
            suggestions = get_suggestions(df, target_column="target")
            
            # Should suggest imputation or missing data handling
            missing_suggestions = [
                s for s in suggestions
                if any(kw in s.title.lower() for kw in ['imput', 'missing', 'faltante'])
            ]
            assert len(missing_suggestions) > 0
        except ImportError:
            pytest.skip("AutoML module not available")
    
    def test_suggestions_have_module_links(self, sample_dataframe):
        """Test that suggestions include module links."""
        try:
            from src.automl import get_suggestions
            
            suggestions = get_suggestions(sample_dataframe, target_column="target")
            
            for s in suggestions:
                assert hasattr(s, 'module_link')
                # Module links should reference dashboard pages
                if s.module_link:
                    assert any(
                        keyword in s.module_link.lower()
                        for keyword in ['preprocess', 'training', 'evaluation', 'eda', 'automl', 'cleaning']
                    )
        except ImportError:
            pytest.skip("AutoML module not available")
    
    def test_suggestions_have_priority(self, sample_dataframe):
        """Test that suggestions have priority levels."""
        try:
            from src.automl import get_suggestions
            
            suggestions = get_suggestions(sample_dataframe, target_column="target")
            
            for s in suggestions:
                assert hasattr(s, 'priority')
                assert s.priority.value in ['high', 'medium', 'low']
        except ImportError:
            pytest.skip("AutoML module not available")


# ============================================================================
# Test FLAML Integration
# ============================================================================

class TestFLAMLClassifier:
    """Tests for FLAML classifier wrapper."""
    
    def test_flaml_availability_check(self):
        """Test FLAML availability check function."""
        try:
            from src.automl import is_flaml_available
            
            # Should return boolean
            result = is_flaml_available()
            assert isinstance(result, bool)
        except ImportError:
            pytest.skip("AutoML module not available")
    
    @pytest.mark.skipif(
        not pytest.importorskip("flaml", reason="FLAML not installed"),
        reason="FLAML not available"
    )
    def test_flaml_classifier_creation(self):
        """Test FLAML classifier creation."""
        try:
            from src.automl import FLAMLClassifier
            
            clf = FLAMLClassifier(
                time_budget=10,
                metric="roc_auc",
            )
            
            assert clf is not None
            assert hasattr(clf, 'fit')
            assert hasattr(clf, 'predict')
            assert hasattr(clf, 'predict_proba')
        except ImportError:
            pytest.skip("FLAMLClassifier not available")
    
    @pytest.mark.skipif(
        not pytest.importorskip("flaml", reason="FLAML not installed"),
        reason="FLAML not available"
    )
    def test_flaml_classifier_quick_fit(self, sample_classification_data):
        """Test quick FLAML training."""
        try:
            from src.automl import FLAMLClassifier
            
            X, y = sample_classification_data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            clf = FLAMLClassifier(
                time_budget=5,  # Very quick
                metric="roc_auc",
                estimator_list=["lrl2"],  # Only logistic regression
            )
            
            clf.fit(X_train, y_train)
            
            # Should be able to predict
            predictions = clf.predict(X_test)
            assert len(predictions) == len(X_test)
            
            # Should be able to predict probabilities
            probas = clf.predict_proba(X_test)
            assert probas.shape[0] == len(X_test)
        except ImportError:
            pytest.skip("FLAMLClassifier not available")


# ============================================================================
# Test AutoML Export
# ============================================================================

class TestAutoMLExport:
    """Tests for AutoML export functionality."""
    
    def test_export_best_model(self, temp_output_dir, sample_classification_data):
        """Test exporting best model."""
        try:
            from src.automl import export_best_model, FLAMLClassifier, is_flaml_available
            
            if not is_flaml_available():
                pytest.skip("FLAML not available")
            
            X, y = sample_classification_data
            
            # Train a quick model
            clf = FLAMLClassifier(
                time_budget=5,
                estimator_list=["lrl2"],
            )
            clf.fit(X, y)
            
            # Export
            model_path = export_best_model(
                automl_model=clf,
                output_dir=temp_output_dir,
                model_name="test_model",
                include_metadata=True,
            )
            
            assert os.path.exists(model_path)
            assert str(model_path).endswith('.joblib')
        except ImportError:
            pytest.skip("AutoML export not available")
    
    def test_convert_to_standalone(self, sample_classification_data):
        """Test converting to standalone model."""
        try:
            from src.automl import convert_to_standalone, FLAMLClassifier, is_flaml_available
            
            if not is_flaml_available():
                pytest.skip("FLAML not available")
            
            X, y = sample_classification_data
            
            clf = FLAMLClassifier(
                time_budget=5,
                estimator_list=["lrl2"],
            )
            clf.fit(X, y)
            
            standalone = convert_to_standalone(
                automl_model=clf,
                feature_names=X.columns.tolist(),
            )
            
            assert hasattr(standalone, 'predict')
            assert hasattr(standalone, 'predict_proba')
            
            # Test prediction
            preds = standalone.predict(X.head())
            assert len(preds) == len(X.head())
        except ImportError:
            pytest.skip("AutoML export not available")


# ============================================================================
# Test Training Integration
# ============================================================================

class TestTrainingIntegration:
    """Tests for AutoML integration with training module."""
    
    def test_is_automl_available(self):
        """Test automl availability check in training module."""
        try:
            from src.training import is_automl_available
            
            result = is_automl_available()
            assert isinstance(result, bool)
        except ImportError:
            pytest.skip("Training module not available")
    
    @pytest.mark.skipif(
        not pytest.importorskip("flaml", reason="FLAML not installed"),
        reason="FLAML not available"
    )
    def test_run_automl_experiment_pipeline(self, sample_classification_data, temp_output_dir):
        """Test complete AutoML experiment pipeline."""
        try:
            from src.training import run_automl_experiment_pipeline, is_automl_available
            
            if not is_automl_available():
                pytest.skip("AutoML not available")
            
            X, y = sample_classification_data
            
            results = run_automl_experiment_pipeline(
                X=X,
                y=y,
                preset="quick",
                time_budget=10,  # Very quick for testing
                output_dir=temp_output_dir,
                include_suggestions=True,
                compare_with_manual=False,
            )
            
            assert "automl_model" in results
            assert "best_estimator" in results
            assert "best_score" in results
            assert "suggestions" in results
            assert "final_model_path" in results
        except ImportError:
            pytest.skip("run_automl_experiment_pipeline not available")


# ============================================================================
# Test Model Registry Integration
# ============================================================================

class TestRegistryIntegration:
    """Tests for AutoML models in the registry."""
    
    def test_automl_classifiers_in_registry(self):
        """Test that AutoML classifiers are available in registry."""
        try:
            from src.models import list_automl_classifiers, is_automl_model
            
            classifiers = list_automl_classifiers()
            
            # Should have some AutoML classifiers registered
            assert isinstance(classifiers, list)
            
            # Check is_automl_model function
            if classifiers:
                assert is_automl_model(classifiers[0])
        except ImportError:
            pytest.skip("Model registry not available")
    
    def test_make_automl_classifiers(self):
        """Test make_automl_classifiers function."""
        try:
            from src.models.classifiers import make_automl_classifiers
            
            classifiers = make_automl_classifiers()
            
            assert isinstance(classifiers, dict)
            # Should have preset-based classifiers
            expected_keys = {'automl_quick', 'automl_balanced', 'automl_high_performance'}
            assert expected_keys.issubset(set(classifiers.keys()))
        except ImportError:
            pytest.skip("make_automl_classifiers not available")


# ============================================================================
# Test Edge Cases
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_empty_dataframe(self):
        """Test handling of empty dataframe."""
        try:
            from src.automl import analyze_dataset
            
            df = pd.DataFrame()
            
            # Should handle empty dataframe gracefully with sensible defaults
            result = analyze_dataset(df, target_column="target")
            assert result.n_samples == 0
            assert result.n_features <= 0  # No features
        except ImportError:
            pytest.skip("AutoML module not available")
    
    def test_single_sample(self):
        """Test handling of single sample."""
        try:
            from src.automl import analyze_dataset
            
            df = pd.DataFrame({
                'feature': [1.0],
                'target': [0],
            })
            
            # Should handle gracefully or raise meaningful error
            try:
                analysis = analyze_dataset(df, target_column="target")
                assert analysis.n_samples == 1
            except ValueError:
                pass  # Acceptable to raise ValueError
        except ImportError:
            pytest.skip("AutoML module not available")
    
    def test_missing_target_column(self, sample_dataframe):
        """Test handling of missing target column."""
        try:
            from src.automl import analyze_dataset
            
            # analyze_dataset handles missing target gracefully
            # It just doesn't compute class-specific stats
            result = analyze_dataset(sample_dataframe, target_column="nonexistent")
            assert result.n_samples > 0  # Should still analyze features
            assert result.n_classes == 0  # No target classes analyzed
        except ImportError:
            pytest.skip("AutoML module not available")
    
    def test_all_missing_feature(self):
        """Test handling of feature with all missing values."""
        try:
            from src.automl import analyze_dataset
            
            df = pd.DataFrame({
                'feature1': [1, 2, 3, 4, 5],
                'all_missing': [np.nan, np.nan, np.nan, np.nan, np.nan],
                'target': [0, 1, 0, 1, 0],
            })
            
            analysis = analyze_dataset(df, target_column="target")
            assert analysis.missing_percentage > 0
        except ImportError:
            pytest.skip("AutoML module not available")
