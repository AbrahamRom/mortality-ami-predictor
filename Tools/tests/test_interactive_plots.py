"""Tests for interactive plotting functions.

NOTE: Some tests are skipped as they need to be updated to match current API signatures.
TODO: Update test signatures to match current implementation.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import plotly.graph_objects as go
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Import plotting functions
from src.eda.visualizations import (
    plot_distribution,
    plot_correlation_matrix,  # Changed from plot_correlation_heatmap
    plot_scatter,
)
# Note: plot_boxplot and plot_missing_data are not implemented yet
# from src.eda.visualizations import plot_boxplot, plot_missing_data
from src.explainability.shap_analysis import (
    compute_shap_values,
    plot_shap_beeswarm,
    plot_shap_bar,
    plot_shap_waterfall,
    plot_shap_force,
)
from src.evaluation.reporters import plot_confusion_matrix, plot_roc_curve
from src.evaluation.calibration import plot_calibration_curve
from src.evaluation.decision_curves import decision_curve_analysis
from src.training.learning_curves import plot_learning_curve


class TestEDAVisualizations:
    """Test EDA visualization functions."""
    
    @pytest.fixture
    def sample_dataframe(self):
        """Create a sample DataFrame for testing."""
        np.random.seed(42)
        return pd.DataFrame({
            'age': np.random.randint(20, 80, 100),
            'weight': np.random.normal(70, 15, 100),
            'height': np.random.normal(170, 10, 100),
            'category': np.random.choice(['A', 'B', 'C'], 100),
            'target': np.random.choice([0, 1], 100),
        })
    
    def test_plot_distribution_returns_plotly_figure(self, sample_dataframe):
        """Test that plot_distribution returns a Plotly Figure."""
        # Test numeric variable
        fig = plot_distribution(sample_dataframe, 'age', is_numeric=True)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0
        
        # Test categorical variable
        fig = plot_distribution(sample_dataframe, 'category', is_numeric=False)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0
    
    def test_plot_distribution_saves_image(self, sample_dataframe):
        """Test that plot_distribution can save to file."""
        import tempfile
        from pathlib import Path
        
        fig = plot_distribution(sample_dataframe, 'age', is_numeric=True)
        
        # Save as HTML
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.html') as f:
            temp_path = Path(f.name)
        
        try:
            fig.write_html(str(temp_path))
            assert temp_path.exists()
            assert temp_path.stat().st_size > 0
        finally:
            if temp_path.exists():
                temp_path.unlink()
    
    @pytest.mark.skip(reason="plot_boxplot not implemented yet")
    def test_plot_boxplot_returns_plotly_figure(self, sample_dataframe):
        """Test that plot_boxplot returns a Plotly Figure."""
        # fig = plot_boxplot(sample_dataframe, ['age', 'weight', 'height'])
        # assert isinstance(fig, go.Figure)
        # assert len(fig.data) > 0
        pass
    
    def test_plot_correlation_matrix_returns_plotly_figure(self, sample_dataframe):
        """Test that correlation matrix returns a Plotly Figure."""
        numeric_cols = ['age', 'weight', 'height']
        fig = plot_correlation_matrix(sample_dataframe, numeric_cols)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0


class TestSHAPVisualizations:
    """Test SHAP visualization functions."""
    
    @pytest.fixture
    def shap_data(self):
        """Create sample SHAP values for testing."""
        np.random.seed(42)
        X, y = make_classification(n_samples=100, n_features=10, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Train simple model
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        
        # Compute SHAP values - corrected parameter name
        feature_names = [f"feature_{i}" for i in range(10)]
        explainer, shap_values = compute_shap_values(model, X_test, feature_names=feature_names, max_samples=50)
        
        return shap_values
    
    def test_plot_shap_beeswarm_returns_plotly_figure(self, shap_data):
        """Test that SHAP beeswarm returns a Plotly Figure."""
        fig = plot_shap_beeswarm(shap_data, max_display=10)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0
    
    def test_plot_shap_bar_returns_plotly_figure(self, shap_data):
        """Test that SHAP bar plot returns a Plotly Figure."""
        fig = plot_shap_bar(shap_data, max_display=10)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0
    
    def test_plot_shap_waterfall_returns_plotly_figure(self, shap_data):
        """Test that SHAP waterfall returns a Plotly Figure."""
        fig = plot_shap_waterfall(shap_data, sample_idx=0, max_display=10)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0
    
    def test_plot_shap_force_returns_plotly_figure(self, shap_data):
        """Test that SHAP force plot returns a Plotly Figure."""
        fig = plot_shap_force(shap_data, sample_idx=0, max_display=10)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0
    
    def test_plot_shap_force_optimized_for_many_features(self, shap_data):
        """Test that force plot handles many features correctly."""
        # Test with a larger max_display value
        fig = plot_shap_force(shap_data, sample_idx=0, max_display=15)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0


class TestEvaluationVisualizations:
    """Test evaluation visualization functions."""
    
    @pytest.fixture
    def evaluation_data(self):
        """Create sample evaluation data."""
        np.random.seed(42)
        y_true = np.random.choice([0, 1], 100)
        y_prob = np.random.rand(100)
        return y_true, y_prob
    
    def test_plot_roc_curve_returns_plotly_figure(self, evaluation_data):
        """Test that ROC curve returns a Plotly Figure."""
        y_true, y_prob = evaluation_data
        fig = plot_roc_curve(y_true, y_prob)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0
    
    def test_plot_confusion_matrix_returns_plotly_figure(self, evaluation_data):
        """Test that confusion matrix returns a Plotly Figure."""
        y_true, y_prob = evaluation_data
        fig = plot_confusion_matrix(y_true, y_prob)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0
    
    def test_plot_confusion_matrix_with_custom_threshold(self, evaluation_data):
        """Test that confusion matrix works with custom threshold."""
        y_true, y_prob = evaluation_data
        fig = plot_confusion_matrix(y_true, y_prob, threshold=0.3)
        assert isinstance(fig, go.Figure)
    
    def test_plot_calibration_curve_returns_plotly_figure(self, evaluation_data):
        """Test that calibration curve returns a Plotly Figure."""
        y_true, y_prob = evaluation_data
        fig = plot_calibration_curve(y_true, y_prob)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0
    
    def test_decision_curve_analysis_returns_plotly_figure(self, evaluation_data):
        """Test that DCA returns a Plotly Figure."""
        y_true, y_prob = evaluation_data
        fig = decision_curve_analysis(y_true, y_prob)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0


class TestLearningCurves:
    """Test learning curve visualization functions."""
    
    @pytest.fixture
    def model_and_data(self):
        """Create sample model and data."""
        X, y = make_classification(n_samples=100, n_features=10, random_state=42)
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        return model, X, y
    
    def test_plot_learning_curve_returns_plotly_figure(self, model_and_data):
        """Test that learning curve returns a Plotly Figure."""
        from src.training.learning_curves import generate_learning_curve
        
        model, X, y = model_and_data
        
        # Generate learning curve result
        result = generate_learning_curve(
            model, X, y, 
            cv=3, 
            train_sizes=np.linspace(0.3, 1.0, 5), 
            scoring='accuracy'
        )
        
        # Plot the result
        fig = plot_learning_curve(result, title="Test Learning Curve")
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0


class TestPlotlyIntegration:
    """Test Plotly configuration and integration."""
    
    def test_plotly_config_exists(self):
        """Test that Plotly config is properly defined."""
        from dashboard.app.config import get_plotly_config
        
        config = get_plotly_config()
        assert isinstance(config, dict)
        assert 'displayModeBar' in config
        assert 'toImageButtonOptions' in config
    
    def test_plotly_config_values(self):
        """Test that Plotly config has expected values."""
        from dashboard.app.config import get_plotly_config
        
        config = get_plotly_config()
        assert config['displayModeBar'] is True
        assert config['displaylogo'] is False
        assert config['scrollZoom'] is True
        assert config['doubleClick'] == 'reset'
    
    def test_plotly_image_export_config(self):
        """Test that image export is properly configured."""
        from dashboard.app.config import get_plotly_config
        
        config = get_plotly_config()
        export_config = config['toImageButtonOptions']
        
        assert export_config['format'] == 'png'
        assert export_config['width'] == 1200
        assert export_config['height'] == 800
        assert export_config['scale'] == 2


class TestDualArchitecture:
    """Test dual architecture: interactive + save."""
    
    def test_functions_return_saveable_figures(self):
        """Test that plotting functions return figures that can be saved."""
        import tempfile
        from pathlib import Path
        
        # Create sample data
        df = pd.DataFrame({
            'age': np.random.normal(50, 10, 50),
            'category': np.random.choice(['A', 'B'], 50),
        })
        
        # Test plot_distribution
        fig = plot_distribution(df, 'age', is_numeric=True)
        assert isinstance(fig, go.Figure)
        
        # Verify it can be saved
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.html') as f:
            temp_path = Path(f.name)
        
        try:
            fig.write_html(str(temp_path))
            assert temp_path.exists()
        finally:
            if temp_path.exists():
                temp_path.unlink()
    
    def test_shap_functions_with_max_samples(self):
        """Test that SHAP functions work with max_samples parameter."""
        from sklearn.ensemble import RandomForestClassifier
        from src.explainability.shap_analysis import compute_shap_values
        
        # Create sample data
        X, y = make_classification(n_samples=100, n_features=10, random_state=42)
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        # Test compute_shap_values with max_samples
        feature_names = [f"feature_{i}" for i in range(10)]
        explainer, shap_values = compute_shap_values(
            model, X[:20], 
            feature_names=feature_names, 
            max_samples=20  # Updated parameter name
        )
        
        assert shap_values is not None
        assert hasattr(shap_values, 'values')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
