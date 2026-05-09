"""Tests for inverse optimization module.

Tests the InverseOptimizer class and related functionality.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.datasets import make_classification, make_regression

from src.explainability.inverse_optimization import InverseOptimizer, find_counterfactuals


@pytest.fixture
def binary_classification_data():
    """Create binary classification dataset and model."""
    X, y = make_classification(
        n_samples=200,
        n_features=10,
        n_informative=7,
        n_redundant=2,
        random_state=42
    )
    
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feature_names)
    
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X, y)
    
    return model, df, feature_names, y


@pytest.fixture
def regression_data():
    """Create regression dataset and model."""
    X, y = make_regression(
        n_samples=200,
        n_features=10,
        n_informative=8,
        random_state=42
    )
    
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feature_names)
    
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X, y)
    
    return model, df, feature_names, y


class TestInverseOptimizer:
    """Test cases for InverseOptimizer class."""
    
    def test_initialization(self, binary_classification_data):
        """Test optimizer initialization."""
        model, df, feature_names, _ = binary_classification_data
        
        optimizer = InverseOptimizer(
            model=model,
            feature_names=feature_names
        )
        
        assert optimizer.model is model
        assert optimizer.feature_names == feature_names
        assert optimizer.n_features == len(feature_names)
        assert optimizer.is_classifier is True
    
    def test_optimization_basic(self, binary_classification_data):
        """Test basic optimization functionality."""
        model, df, feature_names, _ = binary_classification_data
        
        optimizer = InverseOptimizer(
            model=model,
            feature_names=feature_names
        )
        
        # Optimize for class 0
        result = optimizer.optimize(
            target_value=0,
            modifiable_features=feature_names[:5],
            reference_data=df,
            method='SLSQP',
            n_iterations=3,
            random_state=42
        )
        
        assert 'optimal_values' in result
        assert 'achieved_prediction' in result
        assert 'success' in result
        assert len(result['optimal_values']) == 5
    
    def test_optimization_with_fixed_features(self, binary_classification_data):
        """Test optimization with fixed features."""
        model, df, feature_names, _ = binary_classification_data
        
        optimizer = InverseOptimizer(
            model=model,
            feature_names=feature_names
        )
        
        fixed_features = {
            feature_names[0]: 1.0,
            feature_names[1]: -0.5,
        }
        
        result = optimizer.optimize(
            target_value=0,
            modifiable_features=feature_names[2:6],
            fixed_features=fixed_features,
            reference_data=df,
            method='SLSQP',
            random_state=42
        )
        
        assert result['success']
        # Result includes both modifiable and fixed features
        assert len(result['optimal_values']) >= 4
        # Check that fixed features are present with correct values
        assert result['optimal_values'][feature_names[0]] == 1.0
        assert result['optimal_values'][feature_names[1]] == -0.5
    
    def test_optimization_with_initial_values(self, binary_classification_data):
        """Test optimization with initial values."""
        model, df, feature_names, _ = binary_classification_data
        
        optimizer = InverseOptimizer(
            model=model,
            feature_names=feature_names
        )
        
        initial_values = {feat: 0.5 for feat in feature_names[:5]}
        
        result = optimizer.optimize(
            target_value=0,
            modifiable_features=feature_names[:5],
            initial_values=initial_values,
            reference_data=df,
            method='SLSQP',
            random_state=42
        )
        
        assert 'optimal_values' in result
        assert result['success']
    
    def test_optimization_different_methods(self, binary_classification_data):
        """Test different optimization methods."""
        model, df, feature_names, _ = binary_classification_data
        
        optimizer = InverseOptimizer(
            model=model,
            feature_names=feature_names
        )
        
        methods = ['SLSQP', 'COBYLA']
        
        for method in methods:
            result = optimizer.optimize(
                target_value=0,
                modifiable_features=feature_names[:3],
                reference_data=df,
                method=method,
                n_iterations=2,
                random_state=42
            )
            
            assert 'optimal_values' in result
            assert result['method'] == method
    
    def test_optimization_regressor(self, regression_data):
        """Test optimization with regressor."""
        model, df, feature_names, y = regression_data
        
        optimizer = InverseOptimizer(
            model=model,
            feature_names=feature_names
        )
        
        target_value = y.mean()
        
        result = optimizer.optimize(
            target_value=target_value,
            modifiable_features=feature_names[:5],
            reference_data=df,
            method='SLSQP',
            random_state=42
        )
        
        assert 'optimal_values' in result
        assert 'achieved_prediction' in result
        assert result['predicted_class'] is None  # No class for regressor
    
    def test_sensitivity_analysis(self, binary_classification_data):
        """Test sensitivity analysis."""
        model, df, feature_names, _ = binary_classification_data
        
        optimizer = InverseOptimizer(
            model=model,
            feature_names=feature_names
        )
        
        # First optimize
        result = optimizer.optimize(
            target_value=0,
            modifiable_features=feature_names[:3],
            reference_data=df,
            method='SLSQP',
            random_state=42
        )
        
        # Then analyze sensitivity
        sensitivity_df = optimizer.sensitivity_analysis(
            optimal_values=result['optimal_values'],
            modifiable_features=feature_names[:3],
            perturbation_percent=10.0,
            n_points=10
        )
        
        assert isinstance(sensitivity_df, pd.DataFrame)
        assert 'feature' in sensitivity_df.columns
        assert 'value' in sensitivity_df.columns
        assert 'prediction' in sensitivity_df.columns
        assert 'delta_from_optimal' in sensitivity_df.columns
        assert len(sensitivity_df) == 3 * 10  # 3 features, 10 points each
    
    def test_confidence_intervals(self, binary_classification_data):
        """Test confidence interval computation."""
        model, df, feature_names, _ = binary_classification_data
        
        optimizer = InverseOptimizer(
            model=model,
            feature_names=feature_names
        )
        
        ci_result = optimizer.compute_confidence_intervals(
            target_value=0,
            modifiable_features=feature_names[:3],
            reference_data=df,
            n_bootstrap=10,  # Small number for testing
            confidence_level=0.95,
            method='SLSQP',
            random_state=42
        )
        
        assert 'confidence_intervals' in ci_result
        assert 'n_successful' in ci_result
        assert 'n_bootstrap' in ci_result
        
        # Check CI structure
        for feat in feature_names[:3]:
            if feat in ci_result['confidence_intervals']:
                ci_data = ci_result['confidence_intervals'][feat]
                assert 'mean' in ci_data
                assert 'median' in ci_data
                assert 'std' in ci_data
                assert 'lower_ci' in ci_data
                assert 'upper_ci' in ci_data
                assert ci_data['lower_ci'] <= ci_data['upper_ci']
    
    def test_invalid_inputs(self, binary_classification_data):
        """Test error handling for invalid inputs."""
        model, df, feature_names, _ = binary_classification_data
        
        optimizer = InverseOptimizer(
            model=model,
            feature_names=feature_names
        )
        
        # Empty modifiable features
        with pytest.raises(ValueError, match="Must specify at least one modifiable feature"):
            optimizer.optimize(
                target_value=0,
                modifiable_features=[],
                reference_data=df
            )
        
        # Non-existent feature
        with pytest.raises(ValueError, match="not found in model features"):
            optimizer.optimize(
                target_value=0,
                modifiable_features=["nonexistent_feature"],
                reference_data=df
            )
    
    def test_feature_bounds(self, binary_classification_data):
        """Test feature bounds handling."""
        model, df, feature_names, _ = binary_classification_data
        
        # Create custom bounds
        feature_bounds = {
            feature_names[0]: (-2.0, 2.0),
            feature_names[1]: (-1.0, 1.0),
        }
        
        optimizer = InverseOptimizer(
            model=model,
            feature_names=feature_names,
            feature_bounds=feature_bounds
        )
        
        result = optimizer.optimize(
            target_value=0,
            modifiable_features=[feature_names[0], feature_names[1]],
            reference_data=df,
            method='SLSQP',
            random_state=42
        )
        
        # Check bounds are respected
        assert feature_bounds[feature_names[0]][0] <= result['optimal_values'][feature_names[0]] <= feature_bounds[feature_names[0]][1]
        assert feature_bounds[feature_names[1]][0] <= result['optimal_values'][feature_names[1]] <= feature_bounds[feature_names[1]][1]


class TestCounterfactuals:
    """Test counterfactual explanations."""
    
    def test_find_counterfactuals(self, binary_classification_data):
        """Test counterfactual generation."""
        model, df, feature_names, y = binary_classification_data
        
        # Get an instance with class 1
        instance_idx = np.where(y == 1)[0][0]
        instance = df.iloc[instance_idx].values
        
        # Find counterfactuals for class 0
        counterfactuals = find_counterfactuals(
            model=model,
            instance=instance,
            feature_names=feature_names,
            target_class=0,
            modifiable_features=feature_names[:5],
            reference_data=df,
            n_counterfactuals=3,
            random_state=42
        )
        
        assert isinstance(counterfactuals, list)
        assert len(counterfactuals) <= 3
        
        for cf in counterfactuals:
            assert 'optimal_values' in cf
            assert 'achieved_prediction' in cf
    
    def test_counterfactuals_with_fixed_features(self, binary_classification_data):
        """Test counterfactuals with some features fixed."""
        model, df, feature_names, y = binary_classification_data
        
        instance_idx = np.where(y == 1)[0][0]
        instance = df.iloc[instance_idx]
        
        # Only allow some features to change
        modifiable = feature_names[2:5]
        
        counterfactuals = find_counterfactuals(
            model=model,
            instance=instance,
            feature_names=feature_names,
            target_class=0,
            modifiable_features=modifiable,
            reference_data=df,
            n_counterfactuals=2,
            random_state=42
        )
        
        assert len(counterfactuals) <= 2


class TestOptimizationRobustness:
    """Test optimization robustness and edge cases."""
    
    def test_multiple_restarts(self, binary_classification_data):
        """Test that multiple restarts improve results."""
        model, df, feature_names, _ = binary_classification_data
        
        optimizer = InverseOptimizer(
            model=model,
            feature_names=feature_names
        )
        
        # Run with 1 restart
        result1 = optimizer.optimize(
            target_value=0,
            modifiable_features=feature_names[:4],
            reference_data=df,
            method='SLSQP',
            n_iterations=1,
            random_state=42
        )
        
        # Run with 5 restarts
        result5 = optimizer.optimize(
            target_value=0,
            modifiable_features=feature_names[:4],
            reference_data=df,
            method='SLSQP',
            n_iterations=5,
            random_state=42
        )
        
        # More restarts should generally give better or equal objective
        assert result5['distance_to_target'] <= result1['distance_to_target'] * 1.1  # Allow 10% tolerance
    
    def test_convergence_tolerance(self, binary_classification_data):
        """Test different convergence tolerances."""
        model, df, feature_names, _ = binary_classification_data
        
        optimizer = InverseOptimizer(
            model=model,
            feature_names=feature_names
        )
        
        tolerances = [1e-3, 1e-6, 1e-9]
        
        for tol in tolerances:
            result = optimizer.optimize(
                target_value=0,
                modifiable_features=feature_names[:3],
                reference_data=df,
                method='SLSQP',
                tolerance=tol,
                random_state=42
            )
            
            assert 'optimal_values' in result
    
    def test_differential_evolution(self, binary_classification_data):
        """Test global optimization with differential evolution."""
        model, df, feature_names, _ = binary_classification_data
        
        optimizer = InverseOptimizer(
            model=model,
            feature_names=feature_names
        )
        
        result = optimizer.optimize(
            target_value=0,
            modifiable_features=feature_names[:4],
            reference_data=df,
            method='differential_evolution',
            random_state=42
        )
        
        assert 'optimal_values' in result
        assert result['method'] == 'differential_evolution'


def test_optimizer_with_linear_model():
    """Test optimizer with simple linear model."""
    # Create simple dataset
    X = np.random.randn(100, 5)
    coef = np.array([1, -1, 0.5, -0.5, 0])
    y = (X @ coef > 0).astype(int)
    
    feature_names = [f"f{i}" for i in range(5)]
    df = pd.DataFrame(X, columns=feature_names)
    
    model = LogisticRegression()
    model.fit(X, y)
    
    optimizer = InverseOptimizer(
        model=model,
        feature_names=feature_names
    )
    
    result = optimizer.optimize(
        target_value=0,
        modifiable_features=feature_names,
        reference_data=df,
        method='SLSQP',
        random_state=42
    )
    
    assert result['success']
    assert result['achieved_prediction'] < 0.5  # Should predict class 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
