# Model Evaluation

Comprehensive evaluation of model performance with multiple metrics and visualizations.

## Quick Start

### Dashboard Evaluation

1. Navigate to **📈 Model Evaluation** page
2. Load model and test data
3. View metrics and plots
4. Generate PDF report

### Python API

```python
from src.evaluation.metrics import calculate_all_metrics
from src.evaluation.reporters import EvaluationReporter

# Calculate metrics
metrics = calculate_all_metrics(model, X_test, y_test)

print(f"AUC: {metrics['auc']:.3f}")
print(f"Accuracy: {metrics['accuracy']:.3f}")
print(f"Sensitivity: {metrics['sensitivity']:.3f}")
print(f"Specificity: {metrics['specificity']:.3f}")

# Generate report
reporter = EvaluationReporter(model, X_test, y_test)
reporter.generate_report(save_path="evaluation_report.pdf")
```

## Performance Metrics

### ROC Curve

```python
from src.evaluation.reporters import plot_roc_curve

plot_roc_curve(
    model, X_test, y_test,
    save_path="processed/plots/evaluation/roc.png"
)
```

### Calibration

```python
from src.evaluation.calibration import plot_calibration

plot_calibration(
    model, X_test, y_test,
    save_path="processed/plots/evaluation/calibration.png"
)
```

### Decision Curve Analysis

```python
from src.evaluation.decision_curves import plot_decision_curve

plot_decision_curve(
    model, X_test, y_test,
    save_path="processed/plots/evaluation/dca.png"
)
```

## Bootstrap Validation

```python
from src.evaluation.resampling import bootstrap_confidence_intervals

ci_results = bootstrap_confidence_intervals(
    model, X_test, y_test,
    n_iterations=1000
)

print(f"AUC: {ci_results['auc']['mean']:.3f}")
print(f"95% CI: [{ci_results['auc']['ci_low']:.3f}, {ci_results['auc']['ci_high']:.3f}]")
```

## See Also

- [API: Metrics](../api/evaluation/metrics.md)
- [API: Calibration](../api/evaluation/calibration.md)
- [Training Guide](training.md)
