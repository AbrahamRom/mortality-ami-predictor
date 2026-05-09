# Model Training Guide

Train machine learning models to predict AMI mortality with automated pipelines and hyperparameter optimization.

## Quick Start

### Dashboard Training

1. Go to **🤖 Model Training** page
2. Select features to include
3. Choose model type
4. Configure hyperparameters
5. Click **"Train Model"**
6. View metrics and plots

### Python API Training

```python
from src.training.trainer import ModelTrainer
from src.data_load.loaders import load_cleaned_dataset
from src.data_load.splitters import split_data

# Load cleaned data
df = load_cleaned_dataset("processed/cleaned_datasets/cleaned_data.csv")

# Split features and target
X = df.drop(columns=["mortality_inhospital"])
y = df["mortality_inhospital"]

# Train/test split
X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)

# Initialize trainer
trainer = ModelTrainer(model_type="random_forest")

# Train
model = trainer.train(X_train, y_train)

# Evaluate
from src.evaluation.metrics import calculate_all_metrics
metrics = calculate_all_metrics(model, X_test, y_test)
print(f"AUC: {metrics['auc']:.3f}")
```

## Available Models

### Logistic Regression

Fast, interpretable baseline:

```python
trainer = ModelTrainer(
    model_type="logistic",
    params={
        "C": 1.0,
        "penalty": "l2",
        "solver": "lbfgs",
        "max_iter": 1000
    }
)
model = trainer.train(X_train, y_train)
```

### Random Forest

Robust ensemble method:

```python
trainer = ModelTrainer(
    model_type="random_forest",
    params={
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "random_state": 42
    }
)
model = trainer.train(X_train, y_train)
```

### XGBoost

High-performance gradient boosting:

```python
trainer = ModelTrainer(
    model_type="xgboost",
    params={
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42
    }
)
model = trainer.train(X_train, y_train)
```

### Neural Network

Deep learning approach:

```python
trainer = ModelTrainer(
    model_type="neural_network",
    params={
        "hidden_layer_sizes": (100, 50),
        "activation": "relu",
        "solver": "adam",
        "alpha": 0.0001,
        "learning_rate": "adaptive",
        "max_iter": 500
    }
)
model = trainer.train(X_train, y_train)
```

## Cross-Validation

Robust performance estimation:

```python
from src.training.cross_validation import cross_validate_model

cv_results = cross_validate_model(
    model,
    X_train,
    y_train,
    cv=5,  # 5-fold CV
    scoring=["roc_auc", "accuracy", "f1"]
)

print(f"Mean AUC: {cv_results['test_roc_auc'].mean():.3f}")
print(f"Std AUC: {cv_results['test_roc_auc'].std():.3f}")
```

## Hyperparameter Tuning

### Grid Search

Exhaustive search:

```python
from src.training.hyperparameter_tuning import grid_search

param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [5, 10, 15],
    "min_samples_split": [2, 5, 10]
}

best_model = grid_search(
    model_type="random_forest",
    X_train=X_train,
    y_train=y_train,
    param_grid=param_grid,
    cv=5,
    scoring="roc_auc"
)
```

### Random Search

Efficient exploration:

```python
from src.training.hyperparameter_tuning import random_search

param_distributions = {
    "n_estimators": [50, 100, 150, 200],
    "max_depth": [5, 10, 15, 20],
    "learning_rate": [0.01, 0.05, 0.1, 0.2]
}

best_model = random_search(
    model_type="xgboost",
    X_train=X_train,
    y_train=y_train,
    param_distributions=param_distributions,
    n_iter=20,
    cv=5,
    scoring="roc_auc"
)
```

## Learning Curves

Diagnose bias/variance:

```python
from src.training.learning_curves import plot_learning_curve

plot_learning_curve(
    model,
    X_train,
    y_train,
    cv=5,
    save_path="processed/plots/training/learning_curve.png"
)
```

## Model Comparison

Compare multiple models:

```python
from src.training.trainer import ModelTrainer

models = {}
results = {}

# Train multiple models
for model_type in ["logistic", "random_forest", "xgboost"]:
    trainer = ModelTrainer(model_type=model_type)
    model = trainer.train(X_train, y_train)
    models[model_type] = model
    
    # Evaluate
    metrics = calculate_all_metrics(model, X_test, y_test)
    results[model_type] = metrics

# Compare
import pandas as pd
comparison = pd.DataFrame(results).T
print(comparison[["auc", "accuracy", "sensitivity", "specificity"]])
```

## Handling Class Imbalance

### SMOTE

Synthetic oversampling:

```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# Train on balanced data
model = trainer.train(X_train_balanced, y_train_balanced)
```

### Class Weights

Built-in balancing:

```python
trainer = ModelTrainer(
    model_type="random_forest",
    params={
        "class_weight": "balanced",  # Auto-adjust weights
        "n_estimators": 100
    }
)
```

## Experiment Tracking

### MLflow

Track experiments automatically:

```python
import mlflow

mlflow.set_experiment("ami_mortality_prediction")

with mlflow.start_run():
    # Train
    model = trainer.train(X_train, y_train)
    
    # Log parameters
    mlflow.log_params(trainer.params)
    
    # Log metrics
    metrics = calculate_all_metrics(model, X_test, y_test)
    mlflow.log_metrics(metrics)
    
    # Log model
    mlflow.sklearn.log_model(model, "model")
```

## Saving Models

```python
import joblib
from datetime import datetime

# Save with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_path = f"processed/models/random_forest/model_{timestamp}.joblib"
joblib.dump(model, model_path)

# Load later
loaded_model = joblib.load(model_path)
```

## Best Practices

!!! tip "Start Simple"
    Begin with logistic regression as a baseline, then try more complex models.

!!! tip "Use Cross-Validation"
    Always use CV to get reliable performance estimates.

!!! warning "Avoid Data Leakage"
    Never use test data for any training decisions (hyperparameter tuning, feature selection, etc.)

!!! tip "Save Everything"
    Save models, preprocessors, and metadata together for reproducible predictions.

## See Also

- [API: ModelTrainer](../api/training/trainer.md)
- [API: Cross-Validation](../api/training/cross_validation.md)
- [API: Hyperparameter Tuning](../api/training/hyperparameter_tuning.md)
- [Model Evaluation](evaluation.md)
- [Custom Models](custom-models.md)
