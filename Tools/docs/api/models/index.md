# Models

This module defines all machine learning models, model factories, registries, and utilities for model management, persistence, and custom model integration.

## Overview

The Models module provides:

- **Pre-configured Models**: Ready-to-use classifiers and regressors
- **Model Registry**: Centralized model management
- **Custom Model Support**: Base classes for custom implementations
- **Model Persistence**: Save/load trained models
- **Model Selection**: Tools to choose the best model
- **Neural Networks**: Deep learning implementations

## Module Components

### Core Components

- **[`ModelRegistry`](registry.md)**: Centralized model registration and retrieval
- **[`ModelSelector`](selection.md)**: Model selection and comparison tools
- **[`ModelPersistence`](persistence.md)**: Save and load models
- **[`ModelMetadata`](metadata.md)**: Track model information and versions

### Model Types

- **[Classifiers](classifiers.md)**: Binary and multi-class classification
  - Logistic Regression
  - Random Forest Classifier
  - XGBoost Classifier
  - Support Vector Classifier
  - k-Nearest Neighbors

- **[Neural Networks](neural_networks.md)**: Deep learning models
  - Multi-layer Perceptrons
  - TabNet
  - Custom architectures

- **[Regressors](regressors.md)**: Continuous outcome prediction
  - Linear Regression
  - Random Forest Regressor
  - XGBoost Regressor
  - Support Vector Regressor

### Custom Models

- **[`CustomModelBase`](custom_base.md)**: Base class for custom models
- Integration with sklearn pipelines
- Hyperparameter tuning support
- Evaluation compatibility

## Quick Start

### Using Pre-configured Models

```python
from src.models.registry import ModelRegistry

# Get a model from registry
registry = ModelRegistry()
model = registry.get_model('random_forest', task='classification')

# Train the model
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)
```

### Available Models

```python
# List all available models
available_models = registry.list_models()
print(available_models)

# Get model info
info = registry.get_model_info('xgboost')
print(f"Type: {info['type']}")
print(f"Parameters: {info['default_params']}")
```

## Model Registry

The `ModelRegistry` provides centralized model management:

```python
from src.models.registry import ModelRegistry

registry = ModelRegistry()

# Register a new model
registry.register_model(
    name='my_custom_model',
    model_class=MyCustomModel,
    default_params={'param1': 10, 'param2': 0.5},
    task='classification'
)

# Get registered model
model = registry.get_model('my_custom_model')

# List models by task
classifiers = registry.list_models(task='classification')
regressors = registry.list_models(task='regression')
```

## Classifiers

### Logistic Regression

```python
from src.models.classifiers import get_logistic_regression

model = get_logistic_regression(
    penalty='l2',
    C=1.0,
    max_iter=1000,
    class_weight='balanced'
)
```

### Random Forest

```python
from src.models.classifiers import get_random_forest

model = get_random_forest(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    class_weight='balanced',
    random_state=42
)
```

### XGBoost

```python
from src.models.classifiers import get_xgboost

model = get_xgboost(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=3.0,  # For imbalanced data
    random_state=42
)
```

## Neural Networks

```python
from src.models.neural_networks import MLPClassifier

model = MLPClassifier(
    hidden_layers=[64, 32, 16],
    activation='relu',
    dropout=0.3,
    batch_size=32,
    epochs=100,
    early_stopping=True
)

# Train with validation
model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val)
)
```

## Custom Models

Create custom models by inheriting from `CustomModelBase`:

```python
from src.models.custom_base import CustomModelBase
from sklearn.base import BaseEstimator, ClassifierMixin

class MyCustomModel(CustomModelBase, BaseEstimator, ClassifierMixin):
    def __init__(self, param1=10, param2=0.5):
        super().__init__()
        self.param1 = param1
        self.param2 = param2
    
    def fit(self, X, y):
        # Your training logic
        self.classes_ = np.unique(y)
        # ... training code ...
        return self
    
    def predict(self, X):
        # Your prediction logic
        return predictions
    
    def predict_proba(self, X):
        # Your probability prediction logic
        return probabilities

# Register and use
registry.register_model('my_model', MyCustomModel)
model = registry.get_model('my_model', param1=20)
```

See [Custom Models Guide](../../user-guide/custom-models.md) for details.

## Model Selection

Select the best model using cross-validation:

```python
from src.models.selection import ModelSelector

selector = ModelSelector(
    models=['logistic', 'random_forest', 'xgboost'],
    cv=5,
    scoring='roc_auc'
)

# Find best model
best_model = selector.fit(X_train, y_train)

# Compare all models
comparison = selector.compare_models()
print(comparison)

# Get detailed results
results = selector.get_results()
```

## Model Persistence

Save and load trained models:

```python
from src.models.persistence import save_model, load_model

# Save model
save_model(
    model=trained_model,
    filepath='models/best_model.joblib',
    metadata={
        'model_type': 'XGBoost',
        'features': feature_names,
        'train_date': '2025-01-01',
        'performance': {'roc_auc': 0.85}
    }
)

# Load model
loaded_model, metadata = load_model('models/best_model.joblib')
print(f"Model type: {metadata['model_type']}")
print(f"ROC-AUC: {metadata['performance']['roc_auc']}")
```

## Model Metadata

Track model information:

```python
from src.models.metadata import ModelMetadata

metadata = ModelMetadata(
    model_name="XGBoost_v1",
    model_type="XGBoost",
    features=feature_names,
    hyperparameters=model.get_params(),
    training_date="2025-01-01",
    training_samples=len(X_train),
    performance_metrics={'roc_auc': 0.85, 'pr_auc': 0.78}
)

# Save metadata
metadata.save('models/xgboost_v1_metadata.json')

# Load metadata
loaded_metadata = ModelMetadata.load('models/xgboost_v1_metadata.json')
```

## Model Comparison

Compare multiple models:

```python
from src.models.selection import compare_models

results = compare_models(
    models=[model1, model2, model3],
    model_names=['Logistic', 'RF', 'XGBoost'],
    X_test=X_test,
    y_test=y_test,
    metrics=['roc_auc', 'pr_auc', 'f1', 'brier_score']
)

# Visualize comparison
results.plot_comparison(save_path='model_comparison.png')

# Statistical tests
significance = results.statistical_tests(
    metric='roc_auc',
    test='delong'
)
```

## Hyperparameter Optimization

Optimize model hyperparameters:

```python
from src.models.classifiers import get_xgboost
from src.training.hyperparameter_tuning import optimize_hyperparameters

# Define parameter space
param_space = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7, 10],
    'learning_rate': [0.01, 0.1, 0.3],
    'subsample': [0.7, 0.8, 0.9, 1.0]
}

# Optimize
best_model, best_params = optimize_hyperparameters(
    model=get_xgboost(),
    param_space=param_space,
    X_train=X_train,
    y_train=y_train,
    cv=5,
    scoring='roc_auc',
    method='random_search',
    n_iter=50
)
```

## Best Practices

1. **Use Registry**: Register all models for consistent access
2. **Version Models**: Track model versions and metadata
3. **Save Everything**: Persist models with full metadata
4. **Cross-validation**: Always validate model selection
5. **Hyperparameter Tuning**: Optimize before final training
6. **Class Imbalance**: Use `class_weight` or `scale_pos_weight`
7. **Feature Names**: Track feature names with models
8. **Documentation**: Document custom model implementations

## Integration with Other Modules

- **[Training Module](../training/index.md)**: Train and tune models
- **[Evaluation Module](../evaluation/index.md)**: Evaluate model performance
- **[Explainability](../explainability/shap_analysis.md)**: Interpret model predictions
- **[Prediction](../prediction/predictor.md)**: Make predictions with trained models

## Related Documentation

- [User Guide: Model Training](../../user-guide/training.md)
- [User Guide: Custom Models](../../user-guide/custom-models.md)
- [Architecture: Model Registry](../../architecture/index.md#model-registry)
- [Dashboard: Training Page](../../user-guide/dashboard-pages-reference.md#page-02-model-training)
