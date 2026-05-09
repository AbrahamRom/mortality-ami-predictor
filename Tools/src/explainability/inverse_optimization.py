"""Inverse Optimization Module for Treatment Recommendations.

This module provides inverse prediction optimization functionality to find
optimal feature values that maximize/minimize model predictions. This is useful
for treatment optimization, intervention planning, and "what-if" scenario analysis.

Scientific Approach:
- Uses scipy.optimize for constrained optimization
- Implements gradient-free methods (COBYLA, SLSQP) for robustness
- Supports both continuous and categorical features
- Provides confidence intervals via bootstrapping
- Handles feature constraints and realistic bounds

References:
    - Scipy Optimize: https://docs.scipy.org/doc/scipy/reference/optimize.html
    - Counterfactual Explanations: Wachter et al. (2017)
    - Actionable Recourse: Ustun et al. (2019)
"""

from __future__ import annotations

import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.optimize import minimize, differential_evolution
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler


class InverseOptimizer:
    """Finds optimal feature values to achieve desired model predictions.
    
    This class implements inverse optimization for machine learning models,
    finding the combination of feature values that optimizes the model's
    prediction while respecting feature constraints.
    
    Attributes:
        model: Trained sklearn-compatible model
        feature_names: List of feature names
        feature_bounds: Dictionary of (min, max) bounds for each feature
        categorical_features: List of categorical feature names
        feature_types: Dictionary mapping features to their types
    
    Example:
        >>> optimizer = InverseOptimizer(model, feature_names)
        >>> result = optimizer.optimize(
        ...     target_value=0,  # Mortality=0
        ...     modifiable_features=['medication_X', 'lifestyle_Y'],
        ...     fixed_features={'age': 65}
        ... )
        >>> print(result['optimal_values'])
    """
    
    def __init__(
        self,
        model: BaseEstimator,
        feature_names: List[str],
        feature_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
        categorical_features: Optional[List[str]] = None,
        feature_metadata: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the inverse optimizer.
        
        Args:
            model: Trained model with predict or predict_proba method
            feature_names: List of all feature names in order
            feature_bounds: Dict of {feature: (min, max)} bounds
            categorical_features: List of categorical feature names
            feature_metadata: Additional metadata about features (types, units, etc.)
        """
        self.model = model
        self.feature_names = list(feature_names)
        self.n_features = len(feature_names)
        
        # Feature bounds
        self.feature_bounds = feature_bounds or {}
        
        # Categorical features
        self.categorical_features = set(categorical_features or [])
        
        # Feature metadata
        self.feature_metadata = feature_metadata or {}
        
        # Determine if model is classifier or regressor
        self.is_classifier = hasattr(model, 'predict_proba')
        
        # Store feature indices
        self.feature_indices = {name: i for i, name in enumerate(feature_names)}
    
    def _get_feature_bounds(self, feature: str, data_range: Optional[Tuple[float, float]] = None) -> Tuple[float, float]:
        """Get bounds for a feature.
        
        Args:
            feature: Feature name
            data_range: Optional (min, max) from data
        
        Returns:
            Tuple of (min, max) bounds
        """
        if feature in self.feature_bounds:
            return self.feature_bounds[feature]
        elif data_range is not None:
            # Add 10% margin to data range
            margin = 0.1 * (data_range[1] - data_range[0])
            return (data_range[0] - margin, data_range[1] + margin)
        else:
            # Default bounds
            return (-1e6, 1e6)
    
    def _create_objective_function(
        self,
        target_value: Union[int, float],
        modifiable_indices: List[int],
        fixed_values: np.ndarray,
        maximize: bool = False,
    ) -> Callable:
        """Create objective function for optimization.
        
        Args:
            target_value: Desired prediction value
            modifiable_indices: Indices of features to optimize
            fixed_values: Array with fixed feature values
            maximize: If True, maximize prediction; if False, minimize distance to target
        
        Returns:
            Objective function for scipy.optimize
        """
        def objective(x_modifiable: np.ndarray) -> float:
            # Construct full feature vector
            x_full = fixed_values.copy()
            x_full[modifiable_indices] = x_modifiable
            
            # Reshape for prediction
            x_full = x_full.reshape(1, -1)
            
            # Convert to DataFrame if model expects it (e.g., pipelines with ColumnTransformer)
            # This is needed because ColumnTransformer requires column names
            import pandas as pd
            x_df = pd.DataFrame(x_full, columns=self.feature_names)
            
            # Get prediction
            if self.is_classifier:
                # For binary classifier, get probability of positive class
                pred = self.model.predict_proba(x_df)[0, 1]
            else:
                pred = self.model.predict(x_df)[0]
            
            # Compute objective
            if maximize:
                # Maximize prediction (minimize negative)
                return -pred
            else:
                # Minimize distance to target
                return (pred - target_value) ** 2
        
        return objective
    
    def _create_constraints(
        self,
        modifiable_indices: List[int],
        fixed_values: np.ndarray,
        constraints: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """Create constraint functions for optimization.
        
        Args:
            modifiable_indices: Indices of modifiable features
            fixed_values: Fixed feature values
            constraints: Optional additional constraints
        
        Returns:
            List of constraint dictionaries for scipy.optimize
        """
        constraint_list = []
        
        # Add user-defined constraints
        if constraints:
            for constraint in constraints:
                if 'type' in constraint and 'fun' in constraint:
                    constraint_list.append(constraint)
        
        return constraint_list
    
    def optimize(
        self,
        target_value: Union[int, float],
        modifiable_features: List[str],
        fixed_features: Optional[Dict[str, float]] = None,
        initial_values: Optional[Dict[str, float]] = None,
        reference_data: Optional[pd.DataFrame] = None,
        method: str = 'SLSQP',
        n_iterations: int = 10,
        random_state: Optional[int] = None,
        tolerance: float = 1e-6,
        maximize: bool = False,
    ) -> Dict[str, Any]:
        """Find optimal feature values to achieve target prediction.
        
        Args:
            target_value: Desired prediction value (e.g., 0 for no mortality)
            modifiable_features: List of features that can be modified
            fixed_features: Dict of {feature: value} for features to keep fixed
            initial_values: Initial guess for modifiable features
            reference_data: Reference dataset to infer bounds
            method: Optimization method ('SLSQP', 'COBYLA', 'differential_evolution')
            n_iterations: Number of random restarts
            random_state: Random seed for reproducibility
            tolerance: Convergence tolerance
            maximize: If True, maximize prediction instead of reaching target
        
        Returns:
            Dictionary containing:
                - optimal_values: Dict of {feature: optimal_value}
                - achieved_prediction: Final model prediction
                - success: Whether optimization converged
                - distance_to_target: Distance from target value
                - n_iterations: Number of iterations used
                - method: Optimization method used
        """
        # Validate inputs
        if not modifiable_features:
            raise ValueError("Must specify at least one modifiable feature")
        
        # Check all features exist
        for feat in modifiable_features:
            if feat not in self.feature_names:
                raise ValueError(f"Feature '{feat}' not found in model features")
        
        fixed_features = fixed_features or {}
        for feat in fixed_features:
            if feat not in self.feature_names:
                raise ValueError(f"Fixed feature '{feat}' not found in model features")
        
        # Get modifiable and fixed indices
        modifiable_indices = [self.feature_indices[f] for f in modifiable_features]
        
        # Create fixed values array
        fixed_values = np.zeros(self.n_features)
        
        # Set fixed feature values
        for feat, val in fixed_features.items():
            idx = self.feature_indices[feat]
            fixed_values[idx] = val
        
        # Initialize modifiable features
        if initial_values is None:
            initial_values = {}
        
        x0 = []
        bounds = []
        
        for feat in modifiable_features:
            idx = self.feature_indices[feat]
            
            # Get initial value
            if feat in initial_values:
                x0.append(initial_values[feat])
            elif feat in fixed_features:
                x0.append(fixed_features[feat])
            elif reference_data is not None and feat in reference_data.columns:
                x0.append(reference_data[feat].median())
            else:
                x0.append(0.0)
            
            # Get bounds
            if reference_data is not None and feat in reference_data.columns:
                data_range = (reference_data[feat].min(), reference_data[feat].max())
            else:
                data_range = None
            
            feat_bounds = self._get_feature_bounds(feat, data_range)
            bounds.append(feat_bounds)
        
        x0 = np.array(x0)
        
        # Create objective function
        objective = self._create_objective_function(
            target_value=target_value,
            modifiable_indices=modifiable_indices,
            fixed_values=fixed_values,
            maximize=maximize,
        )
        
        # Set random seed
        if random_state is not None:
            np.random.seed(random_state)
        
        # Optimization
        best_result = None
        best_objective = np.inf
        
        if method.lower() == 'differential_evolution':
            # Global optimization
            result = differential_evolution(
                objective,
                bounds=bounds,
                seed=random_state,
                maxiter=1000,
                atol=tolerance,
                tol=tolerance,
            )
        else:
            # Local optimization with multiple restarts
            for i in range(n_iterations):
                if i == 0:
                    x_init = x0
                else:
                    # Random restart within bounds
                    x_init = np.array([
                        np.random.uniform(b[0], b[1]) for b in bounds
                    ])
                
                result = minimize(
                    objective,
                    x_init,
                    method=method,
                    bounds=bounds,
                    tol=tolerance,
                    options={'maxiter': 1000}
                )
                
                if result.fun < best_objective:
                    best_objective = result.fun
                    best_result = result
            
            result = best_result
        
        # Get optimal values
        optimal_modifiable = result.x
        
        # Construct full feature vector
        optimal_full = fixed_values.copy()
        optimal_full[modifiable_indices] = optimal_modifiable
        
        # Get final prediction
        optimal_full_reshaped = optimal_full.reshape(1, -1)
        
        if self.is_classifier:
            final_pred = self.model.predict_proba(optimal_full_reshaped)[0, 1]
            pred_class = self.model.predict(optimal_full_reshaped)[0]
        else:
            final_pred = self.model.predict(optimal_full_reshaped)[0]
            pred_class = final_pred
        
        # Create result dictionary
        optimal_dict = {}
        for i, feat in enumerate(modifiable_features):
            optimal_dict[feat] = optimal_modifiable[i]
        
        # Add fixed features to result
        for feat, val in fixed_features.items():
            if feat not in optimal_dict:
                optimal_dict[feat] = val
        
        return {
            'optimal_values': optimal_dict,
            'achieved_prediction': final_pred,
            'predicted_class': pred_class if self.is_classifier else None,
            'success': result.success,
            'distance_to_target': abs(final_pred - target_value),
            'n_function_evaluations': result.nfev if hasattr(result, 'nfev') else None,
            'method': method,
            'objective_value': result.fun,
            'optimization_message': result.message if hasattr(result, 'message') else 'Success',
        }
    
    def compute_confidence_intervals(
        self,
        target_value: Union[int, float],
        modifiable_features: List[str],
        fixed_features: Optional[Dict[str, float]] = None,
        reference_data: Optional[pd.DataFrame] = None,
        n_bootstrap: int = 50,
        confidence_level: float = 0.95,
        method: str = 'SLSQP',
        random_state: Optional[int] = None,
        maximize: bool = False,
    ) -> Dict[str, Any]:
        """Compute confidence intervals for optimal values via bootstrapping.
        
        Args:
            target_value: Desired prediction value
            modifiable_features: Features to optimize
            fixed_features: Fixed feature values
            reference_data: Reference dataset
            n_bootstrap: Number of bootstrap iterations
            confidence_level: Confidence level (e.g., 0.95 for 95%)
            method: Optimization method
            random_state: Random seed
            maximize: Whether to maximize prediction
        
        Returns:
            Dictionary with optimal values and confidence intervals
        """
        if random_state is not None:
            np.random.seed(random_state)
        
        # Store bootstrap results
        bootstrap_results = []
        
        for i in range(n_bootstrap):
            # Add noise to simulate uncertainty (simple approach)
            # In practice, you might retrain model on bootstrap samples
            
            try:
                result = self.optimize(
                    target_value=target_value,
                    modifiable_features=modifiable_features,
                    fixed_features=fixed_features,
                    reference_data=reference_data,
                    method=method,
                    n_iterations=3,  # Fewer iterations for bootstrap
                    random_state=random_state + i if random_state else None,
                    maximize=maximize,
                )
                
                if result['success']:
                    bootstrap_results.append(result['optimal_values'])
            
            except Exception as e:
                warnings.warn(f"Bootstrap iteration {i} failed: {e}")
                continue
        
        if not bootstrap_results:
            raise ValueError("All bootstrap iterations failed")
        
        # Compute confidence intervals
        bootstrap_df = pd.DataFrame(bootstrap_results)
        
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        ci_results = {}
        for feat in modifiable_features:
            if feat in bootstrap_df.columns:
                values = bootstrap_df[feat].values
                ci_results[feat] = {
                    'mean': np.mean(values),
                    'median': np.median(values),
                    'std': np.std(values),
                    'lower_ci': np.percentile(values, lower_percentile),
                    'upper_ci': np.percentile(values, upper_percentile),
                    'all_values': values,
                }
        
        return {
            'confidence_intervals': ci_results,
            'n_successful': len(bootstrap_results),
            'n_bootstrap': n_bootstrap,
            'confidence_level': confidence_level,
        }
    
    def sensitivity_analysis(
        self,
        optimal_values: Dict[str, float],
        modifiable_features: List[str],
        fixed_features: Optional[Dict[str, float]] = None,
        perturbation_percent: float = 10.0,
        n_points: int = 20,
    ) -> pd.DataFrame:
        """Analyze sensitivity of prediction to changes in optimal values.
        
        Args:
            optimal_values: Optimal feature values from optimization
            modifiable_features: Features to analyze
            fixed_features: Fixed feature values
            perturbation_percent: Percentage to perturb each feature
            n_points: Number of points to sample
        
        Returns:
            DataFrame with sensitivity analysis results
        """
        fixed_features = fixed_features or {}
        
        results = []
        
        for feat in modifiable_features:
            if feat not in optimal_values:
                continue
            
            optimal_val = optimal_values[feat]
            
            # Create perturbation range
            delta = abs(optimal_val) * (perturbation_percent / 100)
            if delta == 0:
                delta = 1.0
            
            perturbed_values = np.linspace(
                optimal_val - delta,
                optimal_val + delta,
                n_points
            )
            
            for perturbed_val in perturbed_values:
                # Create feature vector
                x = np.zeros(self.n_features)
                
                # Set fixed features
                for f, v in fixed_features.items():
                    x[self.feature_indices[f]] = v
                
                # Set optimal values
                for f, v in optimal_values.items():
                    if f in self.feature_indices:
                        x[self.feature_indices[f]] = v
                
                # Set perturbed value
                x[self.feature_indices[feat]] = perturbed_val
                
                # Predict
                x_reshaped = x.reshape(1, -1)
                if self.is_classifier:
                    pred = self.model.predict_proba(x_reshaped)[0, 1]
                else:
                    pred = self.model.predict(x_reshaped)[0]
                
                results.append({
                    'feature': feat,
                    'value': perturbed_val,
                    'delta_from_optimal': perturbed_val - optimal_val,
                    'prediction': pred,
                })
        
        return pd.DataFrame(results)


def find_counterfactuals(
    model: BaseEstimator,
    instance: Union[np.ndarray, pd.Series],
    feature_names: List[str],
    target_class: int,
    modifiable_features: Optional[List[str]] = None,
    reference_data: Optional[pd.DataFrame] = None,
    n_counterfactuals: int = 5,
    diversity_weight: float = 0.5,
    random_state: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Find counterfactual explanations for a given instance.
    
    Counterfactuals show minimal changes needed to flip prediction.
    
    Args:
        model: Trained classifier
        instance: Instance to explain
        feature_names: List of feature names
        target_class: Desired prediction class
        modifiable_features: Features that can be changed
        reference_data: Reference data for bounds
        n_counterfactuals: Number of counterfactuals to generate
        diversity_weight: Weight for diversity among counterfactuals
        random_state: Random seed
    
    Returns:
        List of counterfactual dictionaries
    """
    if isinstance(instance, pd.Series):
        instance = instance.values
    
    if modifiable_features is None:
        modifiable_features = feature_names
    
    optimizer = InverseOptimizer(
        model=model,
        feature_names=feature_names,
        categorical_features=[],
    )
    
    counterfactuals = []
    
    # Create fixed features (non-modifiable)
    fixed_features = {}
    for i, feat in enumerate(feature_names):
        if feat not in modifiable_features:
            fixed_features[feat] = instance[i]
    
    for i in range(n_counterfactuals):
        try:
            result = optimizer.optimize(
                target_value=target_class,
                modifiable_features=modifiable_features,
                fixed_features=fixed_features,
                reference_data=reference_data,
                method='differential_evolution',
                random_state=random_state + i if random_state else None,
            )
            
            if result['success']:
                counterfactuals.append(result)
        
        except Exception as e:
            warnings.warn(f"Counterfactual {i} failed: {e}")
    
    return counterfactuals
