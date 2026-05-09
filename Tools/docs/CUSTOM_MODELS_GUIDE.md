# Custom Models Guide

**Complete guide for creating, training, and deploying custom machine learning models in the Mortality AMI Predictor system.**

---

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Creating Custom Models](#creating-custom-models)
4. [Training Integration](#training-integration)
5. [Evaluation](#evaluation)
6. [Explainability](#explainability)
7. [Persistence](#persistence)
8. [Dashboard Integration](#dashboard-integration)
9. [Best Practices](#best-practices)
10. [Troubleshooting](#troubleshooting)
11. [API Reference](#api-reference)
12. [Examples](#examples)

---

## Overview

The Custom Models system allows you to:

- ✅ Create sklearn-compatible models with custom architectures
- ✅ Integrate seamlessly with the existing training pipeline
- ✅ Evaluate alongside standard models with consistent metrics
- ✅ Generate SHAP explanations and feature importance
- ✅ Save/load models with versioning and validation
- ✅ Manage models through the Streamlit dashboard

### Architecture

```
Custom Model System
├── Base Classes (src/models/custom_base.py)
│   ├── BaseCustomModel (abstract)
│   ├── BaseCustomClassifier
│   ├── BaseCustomRegressor
│   └── CustomModelWrapper
├── Integration Modules
│   ├── Training (src/training/custom_integration.py)
│   ├── Evaluation (src/evaluation/custom_integration.py)
│   └── Explainability (src/explainability/custom_integration.py)
├── Persistence (src/models/persistence.py)
└── Dashboard UI (pages/07_🔧_Custom_Models.py)
```

---

## Quick Start

### 1. Create a Custom Classifier

```python
from src.models.custom_base import BaseCustomClassifier
import numpy as np

class MyCustomClassifier(BaseCustomClassifier):
    def __init__(self, n_layers=3, learning_rate=0.01):
        super().__init__(name="MyCustomClassifier")
        self.n_layers = n_layers
        self.learning_rate = learning_rate
        self.classes_ = None
        
    def fit(self, X, y):
        """Train the model."""
        self.classes_ = np.unique(y)
        # Your training logic here
        return self
    
    def predict(self, X):
        """Make predictions."""
        # Your prediction logic here
        return predictions
    
    def predict_proba(self, X):
        """Predict probabilities."""
        # Your probability prediction logic here
        return probabilities
```

### 2. Train Your Model

```python
from sklearn.model_selection import train_test_split
import pandas as pd

# Load data
df = pd.read_csv("data.csv")
X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Train
model = MyCustomClassifier(n_layers=5, learning_rate=0.01)
model.fit(X_train, y_train)
```

### 3. Save Your Model

```python
from src.models.persistence import save_custom_model

save_custom_model(
    model=model,
    path="models/custom/my_classifier",
    feature_names=list(X.columns),
    metadata={
        "description": "Custom neural network classifier",
        "author": "Your Name"
    }
)
```

### 4. Use in Dashboard

Upload your saved model in the **Custom Models** page and select it for training/evaluation!

---

## Creating Custom Models

### Base Classes

#### BaseCustomModel (Abstract)

The foundation for all custom models. Provides sklearn compatibility.

**Required Methods:**
- `fit(X, y)`: Train the model
- `predict(X)`: Make predictions
- `get_params(deep=True)`: Get model parameters
- `set_params(**params)`: Set model parameters

**Optional Methods:**
- `save_model(path)`: Custom save logic
- `load_model(path)`: Custom load logic

#### BaseCustomClassifier

Extends `BaseCustomModel` for classification tasks.

**Additional Required:**
- `predict_proba(X)`: Return probability estimates
- `classes_`: Array of class labels (set in `fit`)

**Example:**

```python
from src.models.custom_base import BaseCustomClassifier
from sklearn.ensemble import RandomForestClassifier

class EnhancedRandomForest(BaseCustomClassifier):
    def __init__(self, n_estimators=100, max_depth=None, custom_param=42):
        super().__init__(name="EnhancedRF")
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.custom_param = custom_param
        self._rf = None
        
    def fit(self, X, y):
        self._rf = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=42
        )
        self._rf.fit(X, y)
        self.classes_ = self._rf.classes_
        return self
    
    def predict(self, X):
        return self._rf.predict(X)
    
    def predict_proba(self, X):
        return self._rf.predict_proba(X)
```

#### BaseCustomRegressor

Extends `BaseCustomModel` for regression tasks.

**Example:**

```python
from src.models.custom_base import BaseCustomRegressor
import numpy as np

class WeightedLinearRegression(BaseCustomRegressor):
    def __init__(self, alpha=1.0):
        super().__init__(name="WeightedLinReg")
        self.alpha = alpha
        self.coef_ = None
        self.intercept_ = None
    
    def fit(self, X, y, sample_weight=None):
        # Custom weighted fitting logic
        # ...
        return self
    
    def predict(self, X):
        return X @ self.coef_ + self.intercept_
```

### CustomModelWrapper

Wraps models with preprocessing pipelines.

```python
from src.models.custom_base import CustomModelWrapper
from sklearn.preprocessing import StandardScaler

model = MyCustomClassifier()
preprocessing = StandardScaler()

wrapper = CustomModelWrapper(
    model=model,
    preprocessing=preprocessing,
    hyperparameters={'n_layers': 5, 'learning_rate': 0.01}
)

# Automatically applies preprocessing
wrapper.fit(X_train, y_train)
predictions = wrapper.predict(X_test)
```

---

## Training Integration

### Basic Training

```python
from src.training.custom_integration import train_custom_model

results = train_custom_model(
    model=MyCustomClassifier(),
    X_train=X_train,
    y_train=y_train,
    X_val=X_val,
    y_val=y_val,
    cv_folds=5
)
```

### Cross-Validation

```python
from src.training.custom_integration import cross_validate_custom_model

cv_results = cross_validate_custom_model(
    model=MyCustomClassifier(),
    X=X_train,
    y=y_train,
    cv=5,
    scoring=['accuracy', 'roc_auc', 'f1']
)

print(f"Mean Accuracy: {cv_results['test_accuracy'].mean():.3f}")
print(f"Std: {cv_results['test_accuracy'].std():.3f}")
```

### Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_layers': [3, 5, 7],
    'learning_rate': [0.001, 0.01, 0.1]
}

grid_search = GridSearchCV(
    estimator=MyCustomClassifier(),
    param_grid=param_grid,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
```

### Mixed Training (Custom + Standard)

```python
from src.training.custom_integration import train_mixed_models_with_cv

models = {
    'custom_classifier': MyCustomClassifier(),
    'random_forest': RandomForestClassifier(),
    'xgboost': XGBClassifier()
}

results = train_mixed_models_with_cv(
    models=models,
    X=X_train,
    y=y_train,
    cv=5
)
```

---

## Evaluation

### Single Model Evaluation

```python
from src.evaluation.custom_integration import evaluate_custom_classifier

metrics = evaluate_custom_classifier(
    model=trained_model,
    X_test=X_test,
    y_test=y_test,
    model_name="MyCustomClassifier"
)

print(f"Accuracy: {metrics['accuracy']:.3f}")
print(f"ROC-AUC: {metrics['auroc']:.3f}")
print(f"AUPRC: {metrics['auprc']:.3f}")
```

### Batch Evaluation

```python
from src.evaluation.custom_integration import batch_evaluate_mixed_models

models_dict = {
    'model_1': model_1,
    'model_2': model_2,
    'model_3': model_3
}

results = batch_evaluate_mixed_models(
    models=models_dict,
    X_test=X_test,
    y_test=y_test
)

# Compare results
for name, metrics in results.items():
    print(f"{name}: AUC={metrics['auroc']:.3f}")
```

### Compare with Standard Models

```python
from src.evaluation.custom_integration import compare_model_performance

comparison = compare_model_performance(
    models={
        'custom': custom_model,
        'rf': random_forest,
        'xgb': xgboost_model
    },
    X_test=X_test,
    y_test=y_test,
    metrics=['accuracy', 'auroc', 'f1']
)

# Returns DataFrame with all metrics
print(comparison)
```

---

## Explainability

### SHAP Values

```python
from src.explainability.custom_integration import compute_shap_for_custom_model

shap_values = compute_shap_for_custom_model(
    model=trained_model,
    X=X_test[:100],  # Sample for speed
    feature_names=feature_names
)

# Plot SHAP summary
import shap
shap.summary_plot(shap_values, X_test[:100])
```

### Permutation Importance

```python
from src.explainability.custom_integration import compute_permutation_importance_custom

importance_df = compute_permutation_importance_custom(
    model=trained_model,
    X=X_test,
    y=y_test,
    feature_names=feature_names,
    n_repeats=10
)

print(importance_df.head())
```

### Feature Importance (Universal)

```python
from src.explainability.custom_integration import get_feature_importance_universal

importance = get_feature_importance_universal(
    model=trained_model,
    feature_names=feature_names
)

# May return None for some models
if importance is not None:
    print(importance.head(10))
```

### Explain Single Prediction

```python
from src.explainability.custom_integration import explain_prediction_custom

explanation = explain_prediction_custom(
    model=trained_model,
    X_instance=X_test.iloc[0],
    feature_names=feature_names
)

# explanation contains SHAP waterfall for this instance
```

---

## Persistence

### Save Model

```python
from src.models.persistence import save_custom_model

save_info = save_custom_model(
    model=trained_model,
    path="models/custom/my_model",
    metadata={
        "model_name": "MyCustomClassifier",
        "model_type": "classifier",
        "description": "Custom neural network for AMI mortality prediction",
        "algorithm": "Multi-layer Perceptron",
        "author": "Research Team",
        "training_date": "2025-11-01"
    },
    preprocessing=scaler,  # Optional
    feature_names=feature_names,
    training_info={
        "cv_scores": cv_scores,
        "hyperparameters": model.get_params(),
        "training_samples": len(X_train)
    },
    overwrite=True
)

print(f"Saved version: {save_info['version']}")
```

**Directory structure created:**
```
models/custom/my_model/
├── model.pkl           # Serialized model
├── metadata.json       # All metadata
├── preprocessing.pkl   # Preprocessing pipeline (if provided)
└── manifest.json       # Version info, checksums
```

### Load Model

```python
from src.models.persistence import load_custom_model

model_data = load_custom_model(
    path="models/custom/my_model",
    validate=True  # Validate after loading
)

model = model_data['model']
metadata = model_data['metadata']
preprocessing = model_data['preprocessing']  # May be None
validation = model_data['validation']

print(f"Model loaded: {metadata['model_name']}")
print(f"Validation: {'✓' if validation['is_valid'] else '✗'}")
```

### Create Model Bundle

Package model with sample data for testing/sharing.

```python
from src.models.persistence import create_model_bundle

bundle_info = create_model_bundle(
    model=trained_model,
    X_sample=X_test[:100],
    y_sample=y_test[:100],
    path="models/bundles/my_model_bundle"
)
```

### Load and Test Bundle

```python
from src.models.persistence import load_model_bundle

bundle_data = load_model_bundle(
    path="models/bundles/my_model_bundle",
    test_model=True  # Automatically test on sample data
)

print(f"Test passed: {bundle_data['test_results']['test_passed']}")
print(f"Accuracy: {bundle_data['test_results'].get('accuracy', 'N/A')}")
```

### List Saved Models

```python
from src.models.persistence import list_saved_models

models = list_saved_models(
    base_path="models/custom",
    include_info=True
)

for model in models:
    print(f"Name: {model['name']}")
    print(f"Type: {model['metadata'].get('model_type', 'unknown')}")
    print(f"Version: {model['manifest'].get('version', 'unknown')}")
    print("---")
```

### Migrate Old Model

```python
from src.models.persistence import migrate_model

migration_result = migrate_model(
    old_path="models/custom/old_model",
    new_path="models/custom/migrated_model",
    target_version="1.0.0"
)

print(f"Migrated from {migration_result['old_version']} to {migration_result['new_version']}")
```

---

## Dashboard Integration

### Upload Model

1. Navigate to **Custom Models** page (🔧)
2. Click **Upload** tab
3. Fill in model details:
   - Upload `.pkl` file
   - Model name
   - Type (classifier/regressor)
   - Description
   - Algorithm name
   - Author
4. Click **Upload Model**

### Train with Custom Models

1. Navigate to **Model Training** page (🤖)
2. Check **Include Custom Models** in sidebar
3. Select which custom models to include
4. Configure training settings
5. Click **Start Training**

Custom models will be trained alongside standard models with the same CV strategy.

### Evaluate Custom Models

1. Navigate to **Model Evaluation** page (📈)
2. Select **Custom Models** in model source radio
3. Choose custom model from dropdown
4. Click **Run Evaluation**

Results include all standard metrics (ROC-AUC, AUPRC, confusion matrix, etc.)

### Manage Models

From the **Custom Models** page:

- **View Details**: See metadata, parameters, training info
- **Test Model**: Run predictions on loaded data
- **Delete Model**: Remove from system
- **Export Bundle**: Package for sharing

---

## Best Practices

### 1. Model Design

✅ **DO:**
- Inherit from `BaseCustomClassifier` or `BaseCustomRegressor`
- Implement all required methods (`fit`, `predict`, etc.)
- Store fitted attributes with underscore suffix (`coef_`, `classes_`)
- Use `get_params`/`set_params` for hyperparameter access
- Add docstrings to all methods

❌ **DON'T:**
- Modify training data in-place
- Store large data arrays as attributes
- Use global variables
- Skip input validation

### 2. Training

✅ **DO:**
- Use cross-validation for robust estimates
- Scale/normalize features appropriately
- Handle imbalanced classes (class_weight, SMOTE)
- Monitor for overfitting
- Save training configuration

❌ **DON'T:**
- Train on test data
- Ignore data leakage
- Skip validation
- Use random seeds inconsistently

### 3. Persistence

✅ **DO:**
- Include comprehensive metadata
- Save preprocessing pipelines
- Document feature names and order
- Version your models
- Test loaded models

❌ **DON'T:**
- Overwrite without backups
- Skip validation after loading
- Forget to save preprocessing
- Use absolute file paths

### 4. Explainability

✅ **DO:**
- Provide feature importance if possible
- Test SHAP compatibility
- Document model decisions
- Validate explanations
- Use appropriate background data

❌ **DON'T:**
- Skip explainability testing
- Use too much data for SHAP (sample first)
- Ignore model-specific requirements
- Forget to check for bias

---

## Troubleshooting

### Model Won't Load

**Problem:** `ModelValidationError` when loading model

**Solutions:**
1. Check that all required methods exist
2. Verify `classes_` attribute for classifiers
3. Ensure feature count matches training
4. Try loading without validation first: `validate=False`

```python
# Debug loading
try:
    model_data = load_custom_model(path, validate=False)
    validation = validate_loaded_model(
        model_data['model'],
        model_data['metadata']
    )
    print("Validation errors:", validation['errors'])
except Exception as e:
    print(f"Load error: {e}")
```

### SHAP Not Working

**Problem:** `compute_shap_for_custom_model` fails

**Solutions:**
1. Reduce sample size (SHAP is slow on large datasets)
2. Check model has `predict_proba` or `predict`
3. Use simpler background data
4. Fall back to permutation importance

```python
# Try permutation importance instead
try:
    shap_values = compute_shap_for_custom_model(model, X[:50], feature_names)
except Exception:
    print("SHAP failed, using permutation importance")
    importance = compute_permutation_importance_custom(
        model, X_test, y_test, feature_names
    )
```

### Training Fails

**Problem:** Model doesn't converge or errors during training

**Solutions:**
1. Check input data types (convert to numpy/pandas)
2. Handle missing values
3. Scale features
4. Adjust hyperparameters
5. Add early stopping

```python
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Preprocess pipeline
imputer = SimpleImputer(strategy='median')
scaler = StandardScaler()

X_train = imputer.fit_transform(X_train)
X_train = scaler.fit_transform(X_train)

# Train with preprocessing wrapper
wrapper = CustomModelWrapper(
    model=model,
    preprocessing=Pipeline([
        ('impute', imputer),
        ('scale', scaler)
    ])
)
wrapper.fit(X_train, y_train)
```

### Predictions Wrong Shape

**Problem:** `predict_proba` returns wrong shape

**Solution:** Ensure binary classification returns (n_samples, 2):

```python
def predict_proba(self, X):
    # Get probabilities for positive class
    proba_positive = self._compute_proba(X)
    
    # Binary classification needs (n, 2) shape
    proba_negative = 1 - proba_positive
    return np.column_stack([proba_negative, proba_positive])
```

### Dashboard Upload Fails

**Problem:** Model upload in dashboard fails

**Solutions:**
1. Check file is valid pickle
2. Verify model has sklearn interface
3. Check file size limits
4. Ensure unique model name

```python
# Test pickle before upload
import joblib

# Save locally first
joblib.dump(model, 'test_model.pkl')

# Test load
loaded = joblib.load('test_model.pkl')
print("Has fit:", hasattr(loaded, 'fit'))
print("Has predict:", hasattr(loaded, 'predict'))
```

---

## API Reference

### Base Classes

#### BaseCustomModel

```python
class BaseCustomModel(ABC):
    def __init__(self, name: str = "CustomModel")
    def fit(self, X, y) -> Self
    def predict(self, X) -> np.ndarray
    def get_params(self, deep: bool = True) -> dict
    def set_params(self, **params) -> Self
    def save_model(self, path: str) -> None
    def load_model(self, path: str) -> Self
```

#### BaseCustomClassifier

```python
class BaseCustomClassifier(BaseCustomModel):
    classes_: np.ndarray  # Set in fit()
    def predict_proba(self, X) -> np.ndarray
```

#### BaseCustomRegressor

```python
class BaseCustomRegressor(BaseCustomModel):
    # Inherits all methods from BaseCustomModel
    pass
```

### Persistence Functions

```python
def save_custom_model(
    model: Any,
    path: str | Path,
    metadata: Optional[Dict] = None,
    preprocessing: Optional[Any] = None,
    feature_names: Optional[list] = None,
    training_info: Optional[Dict] = None,
    overwrite: bool = False
) -> Dict[str, Any]

def load_custom_model(
    path: str | Path,
    validate: bool = True,
    require_preprocessing: bool = False
) -> Dict[str, Any]

def validate_loaded_model(
    model: Any,
    metadata: Dict,
    preprocessing: Optional[Any] = None
) -> Dict[str, Any]

def create_model_bundle(
    model: Any,
    X_sample: np.ndarray | pd.DataFrame,
    y_sample: Optional[np.ndarray | pd.Series] = None,
    path: Optional[str | Path] = None,
    **save_kwargs
) -> Dict[str, Any]

def list_saved_models(
    base_path: str | Path,
    include_info: bool = True
) -> list
```

### Evaluation Functions

```python
def evaluate_custom_classifier(
    model: Any,
    X_test: pd.DataFrame | np.ndarray,
    y_test: pd.Series | np.ndarray,
    threshold: float = 0.5,
    model_name: str = "custom_model"
) -> Dict[str, Any]

def batch_evaluate_mixed_models(
    models: Dict[str, Any],
    X_test: pd.DataFrame | np.ndarray,
    y_test: pd.Series | np.ndarray
) -> Dict[str, Dict]
```

### Explainability Functions

```python
def compute_shap_for_custom_model(
    model: Any,
    X: pd.DataFrame | np.ndarray,
    feature_names: Optional[List[str]] = None,
    max_samples: int = 100,
    check_additivity: bool = False
) -> Optional[shap.Explanation]

def compute_permutation_importance_custom(
    model: Any,
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
    n_repeats: int = 10,
    random_state: Optional[int] = None,
    scoring: Optional[str] = None,
    feature_names: Optional[list] = None
) -> pd.DataFrame

def get_feature_importance_universal(
    model: Any,
    feature_names: Optional[List[str]] = None
) -> Optional[pd.DataFrame]
```

---

## Examples

### Example 1: Simple Ensemble Classifier

```python
from src.models.custom_base import BaseCustomClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import numpy as np

class EnsembleClassifier(BaseCustomClassifier):
    """Ensemble of RF and GB with weighted voting."""
    
    def __init__(self, rf_weight=0.5, gb_weight=0.5):
        super().__init__(name="EnsembleClassifier")
        self.rf_weight = rf_weight
        self.gb_weight = gb_weight
        self._rf = RandomForestClassifier(n_estimators=100, random_state=42)
        self._gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
        
    def fit(self, X, y):
        self._rf.fit(X, y)
        self._gb.fit(X, y)
        self.classes_ = self._rf.classes_
        return self
    
    def predict_proba(self, X):
        rf_proba = self._rf.predict_proba(X)
        gb_proba = self._gb.predict_proba(X)
        return self.rf_weight * rf_proba + self.gb_weight * gb_proba
    
    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

# Use it
model = EnsembleClassifier(rf_weight=0.6, gb_weight=0.4)
model.fit(X_train, y_train)

# Evaluate
from src.evaluation.custom_integration import evaluate_custom_classifier
metrics = evaluate_custom_classifier(model, X_test, y_test)
print(f"Ensemble AUC: {metrics['auroc']:.3f}")
```

### Example 2: Custom Neural Network

```python
from src.models.custom_base import BaseCustomClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

class DeepClassifier(BaseCustomClassifier):
    """Deep MLP with custom architecture."""
    
    def __init__(self, layers=(100, 50, 25), dropout=0.1, learning_rate=0.001):
        super().__init__(name="DeepClassifier")
        self.layers = layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self._mlp = None
        self._scaler = StandardScaler()
        
    def fit(self, X, y):
        # Scale features
        X_scaled = self._scaler.fit_transform(X)
        
        # Create and train MLP
        self._mlp = MLPClassifier(
            hidden_layer_sizes=self.layers,
            learning_rate_init=self.learning_rate,
            alpha=self.dropout,
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1
        )
        self._mlp.fit(X_scaled, y)
        self.classes_ = self._mlp.classes_
        return self
    
    def predict(self, X):
        X_scaled = self._scaler.transform(X)
        return self._mlp.predict(X_scaled)
    
    def predict_proba(self, X):
        X_scaled = self._scaler.transform(X)
        return self._mlp.predict_proba(X_scaled)

# Train and save
model = DeepClassifier(layers=(200, 100, 50), dropout=0.2)
model.fit(X_train, y_train)

save_custom_model(
    model=model,
    path="models/custom/deep_classifier",
    feature_names=list(X.columns),
    metadata={
        "description": "Deep neural network with 3 hidden layers",
        "architecture": str(model.layers),
        "total_params": sum(w.size for w in model._mlp.coefs_)
    }
)
```

### Example 3: Threshold-Optimized Classifier

```python
from src.models.custom_base import BaseCustomClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
import numpy as np

class OptimalThresholdClassifier(BaseCustomClassifier):
    """Classifier with optimal threshold from validation set."""
    
    def __init__(self, base_model=None, threshold=None):
        super().__init__(name="OptimalThresholdClassifier")
        self.base_model = base_model or LogisticRegression()
        self.threshold = threshold
        self.optimal_threshold_ = None
        
    def fit(self, X, y, X_val=None, y_val=None):
        # Train base model
        self.base_model.fit(X, y)
        self.classes_ = self.base_model.classes_
        
        # Find optimal threshold if validation data provided
        if X_val is not None and y_val is not None:
            probas = self.base_model.predict_proba(X_val)[:, 1]
            thresholds = np.linspace(0.1, 0.9, 50)
            
            best_f1 = 0
            best_threshold = 0.5
            
            for t in thresholds:
                preds = (probas >= t).astype(int)
                f1 = f1_score(y_val, preds)
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = t
            
            self.optimal_threshold_ = best_threshold
            print(f"Optimal threshold: {best_threshold:.3f} (F1: {best_f1:.3f})")
        else:
            self.optimal_threshold_ = self.threshold or 0.5
        
        return self
    
    def predict_proba(self, X):
        return self.base_model.predict_proba(X)
    
    def predict(self, X):
        probas = self.predict_proba(X)[:, 1]
        return (probas >= self.optimal_threshold_).astype(int)

# Use with validation set
model = OptimalThresholdClassifier()
model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
```

### Example 4: Complete Workflow

```python
# 1. Create model
from src.models.custom_base import BaseCustomClassifier, CustomModelWrapper
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

class MyClassifier(BaseCustomClassifier):
    # ... implementation ...
    pass

# 2. Prepare data
from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_csv("data/recuima-020425.csv")
X = df.drop(['exitus', 'patient_id'], axis=1)
y = df['exitus']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 3. Train with preprocessing
preprocessing = StandardScaler()
model = CustomModelWrapper(
    model=MyClassifier(n_layers=5),
    preprocessing=preprocessing
)

model.fit(X_train, y_train)

# 4. Cross-validate
scores = cross_val_score(
    model, X_train, y_train,
    cv=5, scoring='roc_auc'
)
print(f"CV AUC: {scores.mean():.3f} ± {scores.std():.3f}")

# 5. Evaluate on test set
from src.evaluation.custom_integration import evaluate_custom_classifier

metrics = evaluate_custom_classifier(
    model=model,
    X_test=X_test,
    y_test=y_test,
    model_name="MyClassifier"
)

print(f"Test AUC: {metrics['auroc']:.3f}")
print(f"Test AUPRC: {metrics['auprc']:.3f}")

# 6. Explain predictions
from src.explainability.custom_integration import (
    compute_shap_for_custom_model,
    compute_permutation_importance_custom
)

# SHAP
shap_values = compute_shap_for_custom_model(
    model=model,
    X=X_test[:100],
    feature_names=list(X.columns)
)

# Permutation importance
importance = compute_permutation_importance_custom(
    model=model,
    X=X_test,
    y=y_test,
    feature_names=list(X.columns),
    scoring='accuracy'
)

print("\nTop 5 features:")
print(importance.head())

# 7. Save model
from src.models.persistence import save_custom_model

save_custom_model(
    model=model,
    path="models/custom/my_classifier_final",
    feature_names=list(X.columns),
    metadata={
        "model_name": "MyClassifier",
        "description": "Custom classifier for AMI mortality",
        "author": "Research Team",
        "cv_auc": f"{scores.mean():.3f}",
        "test_auc": f"{metrics['auroc']:.3f}"
    },
    training_info={
        "n_samples_train": len(X_train),
        "n_samples_test": len(X_test),
        "cv_folds": 5,
        "hyperparameters": model.model.get_params()
    }
)

# 8. Upload to dashboard
# Go to Custom Models page and upload the saved model directory
```

---

## Additional Resources

### Documentation
- [Main README](../README.md)
- [Experiment Pipeline Guide](EXPERIMENT_PIPELINE.md)
- [Testing Summary](../tests/TESTING_SUMMARY.md)

### Code Examples
- `src/models/custom_base.py` - Base class implementations
- `tests/test_custom_models.py` - Comprehensive test examples
- `notebooks/modeling.ipynb` - Training examples

### Support
For issues or questions:
1. Check [Troubleshooting](#troubleshooting) section
2. Review test cases in `tests/test_custom_models.py`
3. Contact the development team

---


