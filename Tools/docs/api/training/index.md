# Model Training

This module provides comprehensive tools for training machine learning models, including cross-validation, hyperparameter tuning, learning curve analysis, and experiment tracking.

## Overview

The Training module offers:

- **Model Training**: Automated training with best practices
- **Cross-Validation**: K-fold, stratified, time-series CV
- **Hyperparameter Tuning**: Grid search, random search, Bayesian optimization
- **Learning Curves**: Diagnose overfitting/underfitting
- **Statistical Tests**: Compare model performance
- **Experiment Tracking**: MLflow and W&B integration
- **PDF Reports**: Training reports with all results

## Module Components

### Core Components

- **[`ModelTrainer`](trainer.md)**: Main training orchestrator
- **[`CrossValidator`](cross_validation.md)**: Cross-validation strategies
- **[`HyperparameterTuner`](hyperparameter_tuning.md)**: Hyperparameter optimization
- **[`LearningCurveAnalyzer`](learning_curves.md)**: Learning curve analysis
- **[`StatisticalTester`](statistical_tests.md)**: Statistical model comparison
- **[`PDFReportGenerator`](pdf_reports.md)**: Generate training reports
- **[`CustomModelIntegration`](custom_integration.md)**: Integrate custom models

## Quick Start

### Basic Training

```python
from src.training.trainer import ModelTrainer

# Initialize trainer
trainer = ModelTrainer(
    model_type='xgboost',
    task='classification',
    random_state=42
)

# Train model
results = trainer.fit(
    X_train=X_train,
    y_train=y_train,
    X_val=X_val,
    y_val=y_val
)

# Access trained model
model = trainer.model
print(f"Training ROC-AUC: {results['train_roc_auc']:.3f}")
print(f"Validation ROC-AUC: {results['val_roc_auc']:.3f}")
```

### Training with Cross-Validation

```python
from src.training.trainer import ModelTrainer

trainer = ModelTrainer(
    model_type='random_forest',
    cv=5,  # 5-fold cross-validation
    cv_strategy='stratified'
)

# Train with CV
cv_results = trainer.fit_cv(X_train, y_train)

print(f"Mean CV ROC-AUC: {cv_results['mean_roc_auc']:.3f}")
print(f"Std CV ROC-AUC: {cv_results['std_roc_auc']:.3f}")
```

## Cross-Validation

Perform robust model validation:

```python
from src.training.cross_validation import CrossValidator

cv = CrossValidator(
    model=model,
    cv=5,
    strategy='stratified',  # or 'kfold', 'timeseries'
    scoring=['roc_auc', 'pr_auc', 'f1', 'brier_score']
)

# Run cross-validation
results = cv.run(X_train, y_train)

# Get detailed results
print(f"ROC-AUC: {results['roc_auc_mean']:.3f} ± {results['roc_auc_std']:.3f}")

# Get fold-wise results
fold_results = cv.get_fold_results()

# Plot CV results
cv.plot_cv_scores(save_path='cv_scores.png')
```

### Nested Cross-Validation

For unbiased hyperparameter tuning:

```python
nested_cv = CrossValidator(
    model=model,
    cv_outer=5,  # Outer CV for evaluation
    cv_inner=3,  # Inner CV for tuning
    param_grid=param_grid
)

results = nested_cv.run_nested(X_train, y_train)
print(f"Unbiased ROC-AUC: {results['outer_roc_auc_mean']:.3f}")
```

## Hyperparameter Tuning

Optimize model hyperparameters:

### Grid Search

```python
from src.training.hyperparameter_tuning import GridSearchTuner

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7, 10],
    'learning_rate': [0.01, 0.1, 0.3]
}

tuner = GridSearchTuner(
    model=model,
    param_grid=param_grid,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1
)

best_model, best_params = tuner.tune(X_train, y_train)
print(f"Best parameters: {best_params}")
print(f"Best ROC-AUC: {tuner.best_score_:.3f}")
```

### Random Search

More efficient for large parameter spaces:

```python
from src.training.hyperparameter_tuning import RandomSearchTuner

param_distributions = {
    'n_estimators': [50, 100, 200, 300],
    'max_depth': range(3, 15),
    'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0]
}

tuner = RandomSearchTuner(
    model=model,
    param_distributions=param_distributions,
    n_iter=50,  # Number of iterations
    cv=5,
    scoring='roc_auc',
    random_state=42
)

best_model, best_params = tuner.tune(X_train, y_train)
```

### Bayesian Optimization

Most efficient approach:

```python
from src.training.hyperparameter_tuning import BayesianTuner

param_space = {
    'n_estimators': (50, 300),
    'max_depth': (3, 15),
    'learning_rate': (0.01, 0.3),
    'subsample': (0.6, 1.0)
}

tuner = BayesianTuner(
    model=model,
    param_space=param_space,
    n_iter=30,
    cv=5,
    scoring='roc_auc'
)

best_model, best_params = tuner.tune(X_train, y_train)

# Plot optimization history
tuner.plot_optimization_history(save_path='bayesian_opt.png')
```

## Learning Curves

Diagnose model performance:

```python
from src.training.learning_curves import LearningCurveAnalyzer

analyzer = LearningCurveAnalyzer(model=model)

# Generate learning curves
results = analyzer.generate_curves(
    X_train, y_train,
    train_sizes=[0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
    cv=5
)

# Plot learning curves
analyzer.plot_learning_curves(
    metric='roc_auc',
    save_path='learning_curves.png'
)

# Diagnose issues
diagnosis = analyzer.diagnose()
if diagnosis['overfitting']:
    print("Model is overfitting. Consider:")
    print("- More training data")
    print("- Regularization")
    print("- Feature selection")
```

## Class Imbalance Handling

Handle imbalanced datasets:

```python
from src.training.trainer import ModelTrainer

# Option 1: Class weights
trainer = ModelTrainer(
    model_type='xgboost',
    model_params={'scale_pos_weight': 3.0}  # Ratio of negative to positive
)

# Option 2: SMOTE
trainer = ModelTrainer(
    model_type='random_forest',
    resampling='smote',
    smote_params={'k_neighbors': 5}
)

# Option 3: Class-balanced sampling
trainer = ModelTrainer(
    model_type='logistic',
    model_params={'class_weight': 'balanced'}
)

results = trainer.fit(X_train, y_train)
```

## Statistical Model Comparison

Compare models statistically:

```python
from src.training.statistical_tests import StatisticalTester

tester = StatisticalTester()

# Compare two models with paired t-test
p_value = tester.paired_t_test(
    scores1=model1_cv_scores,
    scores2=model2_cv_scores
)

if p_value < 0.05:
    print("Models are significantly different")

# Compare multiple models with ANOVA
p_value = tester.anova_test([
    model1_cv_scores,
    model2_cv_scores,
    model3_cv_scores
])

# DeLong test for ROC curves
p_value = tester.delong_test(
    y_true, y_proba1, y_proba2
)
```

## Experiment Tracking

Track experiments with MLflow:

```python
import mlflow
from src.training.trainer import ModelTrainer

# Start MLflow run
with mlflow.start_run():
    trainer = ModelTrainer(
        model_type='xgboost',
        track_experiments=True
    )
    
    results = trainer.fit(X_train, y_train, X_val, y_val)
    
    # MLflow automatically logs:
    # - Parameters
    # - Metrics
    # - Model artifact
    # - Training plots

# View in MLflow UI
# mlflow ui --port 5000
```

## Training Reports

Generate comprehensive training reports:

```python
from src.training.pdf_reports import generate_training_report

generate_training_report(
    model=trained_model,
    training_results=results,
    cv_results=cv_results,
    learning_curves=lc_results,
    output_file='training_report.pdf',
    include_sections=[
        'summary',
        'cv_results',
        'learning_curves',
        'feature_importance',
        'hyperparameters',
        'recommendations'
    ]
)
```

Reports include:

- Training summary and timeline
- Cross-validation results with plots
- Learning curves analysis
- Hyperparameter tuning results
- Feature importance
- Model diagnostics
- Recommendations for improvement

## Custom Model Integration

Integrate custom models with the training pipeline:

```python
from src.training.custom_integration import CustomModelTrainer

# Train custom model
trainer = CustomModelTrainer(
    model_class=YourCustomModel,
    model_params={'param1': 10}
)

# Full training pipeline
results = trainer.fit(
    X_train, y_train,
    X_val, y_val,
    cv=5,
    tune_hyperparameters=True,
    param_grid=param_grid
)
```

See [Custom Integration Guide](custom_integration.md) for details.

## Best Practices

1. **Always Use Validation**: Split data into train/validation/test
2. **Cross-Validation**: Use CV for robust performance estimates
3. **Stratification**: Use stratified CV for imbalanced data
4. **Hyperparameter Tuning**: Tune on validation set, not test
5. **Learning Curves**: Check for overfitting/underfitting
6. **Class Imbalance**: Handle with appropriate techniques
7. **Reproducibility**: Set random seeds
8. **Experiment Tracking**: Log all experiments
9. **Early Stopping**: Prevent overfitting in iterative algorithms
10. **Documentation**: Generate PDF reports for each model

## Integration with Other Modules

- **[Models Module](../models/index.md)**: Access pre-configured models
- **[Evaluation Module](../evaluation/index.md)**: Evaluate trained models
- **[Preprocessing](../preprocessing/pipelines.md)**: Prepare data before training
- **[Explainability](../explainability/shap_analysis.md)**: Interpret trained models

## Related Documentation

- [User Guide: Model Training](../../user-guide/training.md)
- [User Guide: Custom Models](../../user-guide/custom-models.md)
- [Dashboard: Training Page](../../user-guide/dashboard-pages-reference.md#page-02-model-training)
- [Architecture: Training Pipeline](../../architecture/index.md#training-pipeline)
