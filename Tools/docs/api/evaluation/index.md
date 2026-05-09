# Model Evaluation

This module provides comprehensive tools for evaluating machine learning models, with specialized metrics and visualizations for clinical prediction tasks.

## Overview

The Evaluation module offers:

- **Performance Metrics**: Accuracy, ROC-AUC, PR-AUC, Brier score, and more
- **Calibration Analysis**: Assess prediction reliability
- **Decision Curve Analysis**: Evaluate clinical utility
- **Bootstrap Validation**: Robust confidence intervals
- **PDF Reports**: Comprehensive evaluation reports

## Module Components

### Core Classes

- **[`ModelEvaluator`](reporters.md)**: Main evaluation orchestrator
- **[`MetricsCalculator`](metrics.md)**: Calculate all performance metrics
- **[`CalibrationAnalyzer`](calibration.md)**: Calibration plots and statistics
- **[`DecisionCurveAnalyzer`](decision_curves.md)**: Clinical utility analysis
- **[`BootstrapValidator`](resampling.md)**: Bootstrap validation with CI
- **[`PDFReportGenerator`](pdf_reports.md)**: Generate evaluation reports

## Quick Start

```python
from src.evaluation.reporters import ModelEvaluator

# Initialize evaluator
evaluator = ModelEvaluator(
    model=trained_model,
    X_test=X_test,
    y_test=y_test
)

# Generate comprehensive evaluation report
evaluator.evaluate_all(
    output_dir="plots/evaluation",
    report_name="model_evaluation.pdf"
)

# Get specific metrics
metrics = evaluator.calculate_metrics()
print(f"ROC-AUC: {metrics['roc_auc']:.3f}")
print(f"PR-AUC: {metrics['pr_auc']:.3f}")
```

## Performance Metrics

### Classification Metrics

Calculate comprehensive classification metrics:

```python
from src.evaluation.metrics import calculate_classification_metrics

metrics = calculate_classification_metrics(
    y_true=y_test,
    y_pred=predictions,
    y_proba=probabilities
)
```

Available metrics:

- **Accuracy**: Overall correctness
- **Sensitivity (Recall)**: True positive rate
- **Specificity**: True negative rate
- **Precision**: Positive predictive value
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under ROC curve
- **PR-AUC**: Area under Precision-Recall curve
- **Brier Score**: Calibration measure
- **Log Loss**: Probabilistic error

### Confusion Matrix

```python
# Plot confusion matrix
evaluator.plot_confusion_matrix(
    normalize=True,
    save_path="confusion_matrix.png"
)
```

### ROC and Precision-Recall Curves

```python
# ROC curve with optimal threshold
evaluator.plot_roc_curve(
    show_optimal_threshold=True,
    save_path="roc_curve.png"
)

# Precision-Recall curve
evaluator.plot_pr_curve(
    save_path="pr_curve.png"
)
```

## Calibration Analysis

Assess whether predicted probabilities match observed frequencies:

```python
from src.evaluation.calibration import CalibrationAnalyzer

calibrator = CalibrationAnalyzer(y_true, y_proba)

# Calibration plot
calibrator.plot_calibration_curve(n_bins=10)

# Calibration metrics
cal_metrics = calibrator.calculate_calibration_metrics()
print(f"Brier Score: {cal_metrics['brier_score']:.3f}")
print(f"ECE: {cal_metrics['expected_calibration_error']:.3f}")

# Calibration slope and intercept
slope, intercept = calibrator.calibration_belt()
```

## Decision Curve Analysis

Evaluate clinical utility across different threshold probabilities:

```python
from src.evaluation.decision_curves import DecisionCurveAnalyzer

dca = DecisionCurveAnalyzer(y_true, y_proba)

# Plot decision curves
dca.plot_decision_curve(
    threshold_range=(0.0, 0.5),
    save_path="decision_curve.png"
)

# Calculate net benefit at specific threshold
net_benefit = dca.calculate_net_benefit(threshold=0.3)
```

## Bootstrap Validation

Calculate confidence intervals using bootstrap resampling:

```python
from src.evaluation.resampling import bootstrap_metrics

# Bootstrap ROC-AUC with 95% CI
results = bootstrap_metrics(
    y_true=y_test,
    y_proba=probabilities,
    metric='roc_auc',
    n_bootstrap=1000,
    confidence_level=0.95
)

print(f"ROC-AUC: {results['mean']:.3f}")
print(f"95% CI: [{results['ci_lower']:.3f}, {results['ci_upper']:.3f}]")

# Multiple metrics
metrics = bootstrap_metrics(
    y_true, y_proba,
    metrics=['roc_auc', 'pr_auc', 'brier_score'],
    n_bootstrap=1000
)
```

## Comprehensive Evaluation Reports

Generate PDF reports with all evaluation results:

```python
from src.evaluation.pdf_reports import generate_evaluation_report

generate_evaluation_report(
    model=model,
    X_test=X_test,
    y_test=y_test,
    model_name="XGBoost Classifier",
    output_file="evaluation_report.pdf",
    include_sections=[
        'metrics',
        'confusion_matrix',
        'roc_curve',
        'pr_curve',
        'calibration',
        'decision_curve',
        'bootstrap'
    ]
)
```

Reports include:

- Summary table of all metrics
- ROC and PR curves
- Confusion matrix (normalized and raw)
- Calibration plot and metrics
- Decision curve analysis
- Bootstrap confidence intervals
- Feature importance (if available)
- Recommendations for model improvement

## Custom Model Integration

Integrate custom models with the evaluation framework:

```python
from src.evaluation.custom_integration import CustomModelEvaluator

# Wrap your custom model
custom_evaluator = CustomModelEvaluator(
    predict_fn=your_model.predict,
    predict_proba_fn=your_model.predict_proba
)

# Evaluate with standard pipeline
results = custom_evaluator.evaluate(X_test, y_test)
```

See [Custom Integration Guide](custom_integration.md) for details.

## Model Comparison

Compare multiple models side by side:

```python
from src.evaluation.reporters import compare_models

# Compare models
comparison = compare_models(
    models=[model1, model2, model3],
    model_names=["Logistic", "RF", "XGBoost"],
    X_test=X_test,
    y_test=y_test
)

# Plot comparison
comparison.plot_roc_comparison(save_path="roc_comparison.png")
comparison.plot_metrics_comparison(save_path="metrics_comparison.png")

# Statistical comparison
significance = comparison.statistical_test(
    metric='roc_auc',
    test='delong'  # DeLong test for ROC-AUC
)
```

## Best Practices

1. **Multiple Metrics**: Don't rely on a single metric
2. **Calibration**: Always check calibration for probability predictions
3. **Clinical Utility**: Use decision curve analysis for clinical context
4. **Confidence Intervals**: Report bootstrap CIs for metrics
5. **Threshold Selection**: Choose threshold based on clinical needs
6. **Test Set**: Never use training data for final evaluation
7. **Documentation**: Generate PDF reports for reproducibility

## Integration with Other Modules

- **[Training Module](../training/index.md)**: Evaluate models during/after training
- **[Models Module](../models/index.md)**: Evaluate custom models
- **[Explainability](../explainability/shap_analysis.md)**: Combine with model interpretation
- **[Reporting](../reporting/pdf_generator.md)**: Generate comprehensive reports

## Related Documentation

- [User Guide: Model Evaluation](../../user-guide/evaluation.md)
- [Dashboard: Evaluation Page](../../user-guide/dashboard-pages-reference.md#page-04-model-evaluation)
- [Training Guide](../../user-guide/training.md)
