# Custom Models Guide

Create and integrate your own scikit-learn compatible models seamlessly into the Mortality AMI Predictor system.

## Overview

The custom models system allows you to:

- Create sklearn-compatible estimators
- Integrate with the training pipeline
- Use in the dashboard
- Track with MLflow
- Deploy for predictions

## Quick Start

### 1. Create Your Model

```python
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np

class MyCustomClassifier(BaseEstimator, ClassifierMixin):
    """Your custom model."""
    
    def __init__(self, param1=1.0, param2=10):
        self.param1 = param1
        self.param2 = param2
    
    def fit(self, X, y):
        """Fit the model."""
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]
        # Your training logic here
        return self
    
    def predict(self, X):
        """Predict class labels."""
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]
    
    def predict_proba(self, X):
        """Predict class probabilities."""
        # Your prediction logic here
        n_samples = X.shape[0]
        proba = np.zeros((n_samples, len(self.classes_)))
        # Calculate probabilities...
        return proba
```

### 2. Register Your Model

```python
from src.models.registry import ModelRegistry

# Register
registry = ModelRegistry()
registry.register_model("my_custom", MyCustomClassifier)

# Use in training
from src.training.trainer import ModelTrainer

trainer = ModelTrainer(
    model_type="my_custom",
    params={"param1": 2.0, "param2": 20}
)
model = trainer.train(X_train, y_train)
```

### 3. Use in Dashboard

Save your model file in `models/custom/`:

```
models/custom/
└── my_custom_classifier.py
```

It will automatically appear in the dashboard!

## Complete Example

### Advanced Custom Model

```python
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np

class EnsembleVoter(BaseEstimator, ClassifierMixin):
    """Ensemble that combines multiple models via voting."""
    
    def __init__(self, n_estimators=100, voting='soft', weights=None):
        self.n_estimators = n_estimators
        self.voting = voting
        self.weights = weights
        
    def fit(self, X, y):
        """Fit the ensemble."""
        # Validate inputs
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]
        
        # Create base models
        self.models_ = [
            RandomForestClassifier(
                n_estimators=self.n_estimators,
                random_state=i
            )
            for i in range(3)
        ]
        
        # Add logistic regression
        self.models_.append(LogisticRegression(random_state=42))
        
        # Fit all models
        for model in self.models_:
            model.fit(X, y)
        
        # Default weights
        if self.weights is None:
            self.weights = np.ones(len(self.models_)) / len(self.models_)
        
        return self
    
    def predict_proba(self, X):
        """Predict class probabilities."""
        check_is_fitted(self)
        X = check_array(X)
        
        # Collect predictions
        if self.voting == 'soft':
            # Average probabilities
            probas = np.array([model.predict_proba(X) for model in self.models_])
            weighted_probas = np.average(probas, axis=0, weights=self.weights)
            return weighted_probas
        else:
            # Majority vote
            predictions = np.array([model.predict(X) for model in self.models_])
            voted = np.apply_along_axis(
                lambda x: np.bincount(x, weights=self.weights).argmax(),
                axis=0,
                arr=predictions
            )
            proba = np.zeros((X.shape[0], len(self.classes_)))
            proba[np.arange(X.shape[0]), voted] = 1
            return proba
    
    def predict(self, X):
        """Predict class labels."""
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]
    
    def get_params(self, deep=True):
        """Get parameters."""
        return {
            "n_estimators": self.n_estimators,
            "voting": self.voting,
            "weights": self.weights
        }
    
    def set_params(self, **params):
        """Set parameters."""
        for key, value in params.items():
            setattr(self, key, value)
        return self
```

### Register and Use

```python
from src.models.registry import ModelRegistry
from src.training.trainer import ModelTrainer

# Register
registry = ModelRegistry()
registry.register_model("ensemble_voter", EnsembleVoter)

# Train
trainer = ModelTrainer(
    model_type="ensemble_voter",
    params={
        "n_estimators": 100,
        "voting": "soft",
        "weights": [0.3, 0.3, 0.3, 0.1]
    }
)

model = trainer.train(X_train, y_train)

# Evaluate
from src.evaluation.metrics import calculate_all_metrics
metrics = calculate_all_metrics(model, X_test, y_test)
print(f"Ensemble AUC: {metrics['auc']:.3f}")
```

## Requirements

Your custom model must:

1. **Inherit from BaseEstimator**:
   ```python
   from sklearn.base import BaseEstimator, ClassifierMixin
   ```

2. **Implement required methods**:
   - `fit(X, y)`: Train the model
   - `predict(X)`: Predict class labels
   - `predict_proba(X)`: Predict probabilities

3. **Set required attributes**:
   ```python
   def fit(self, X, y):
       self.classes_ = np.unique(y)
       self.n_features_in_ = X.shape[1]
       # ... training logic
       return self
   ```

4. **Follow sklearn conventions**:
   - Return `self` from `fit()`
   - Accept numpy arrays or pandas DataFrames
   - Handle binary and multiclass classification

## Testing Your Model

```python
from sklearn.utils.estimator_checks import check_estimator

# Run sklearn compatibility tests
try:
    check_estimator(MyCustomClassifier())
    print("✅ Model is sklearn-compatible!")
except Exception as e:
    print(f"❌ Error: {e}")
```

## Integration Features

### Hyperparameter Tuning

Your model works with GridSearchCV and RandomizedSearchCV:

```python
from src.training.hyperparameter_tuning import grid_search

param_grid = {
    "param1": [1.0, 2.0, 3.0],
    "param2": [10, 20, 30]
}

best_model = grid_search(
    model_type="my_custom",
    X_train=X_train,
    y_train=y_train,
    param_grid=param_grid,
    cv=5
)
```

### Cross-Validation

```python
from src.training.cross_validation import cross_validate_model

cv_results = cross_validate_model(
    model,
    X_train,
    y_train,
    cv=5,
    scoring=["roc_auc", "accuracy"]
)
```

### Explainability

```python
from src.explainability.shap_analysis import SHAPAnalyzer

analyzer = SHAPAnalyzer(model, X_train)
analyzer.plot_summary(X_test, save_path="shap_custom.png")
```

## Best Practices

!!! tip "Follow sklearn API"
    Stick closely to sklearn conventions for best compatibility.

!!! tip "Add Docstrings"
    Document your model well - it will show up in the API docs!

!!! tip "Validate Inputs"
    Use `check_X_y()` and `check_array()` from sklearn.utils.validation

!!! warning "Handle Edge Cases"
    Test with small datasets, missing values, and edge cases.

## Examples

See the `docs/CUSTOM_MODELS_QUICKSTART.md` for more examples:

- Neural Network ensemble
- Stacking classifier
- Meta-learner
- Domain-specific models

## See Also

- [CUSTOM_MODELS_QUICKSTART.md](../CUSTOM_MODELS_QUICKSTART.md)
- [CUSTOM_MODELS_GUIDE.md](../CUSTOM_MODELS_GUIDE.md)
- [CUSTOM_MODELS_ARCHITECTURE.md](../CUSTOM_MODELS_ARCHITECTURE.md)
- [API: Model Registry](../api/models/registry.md)
- [Training Guide](training.md)
