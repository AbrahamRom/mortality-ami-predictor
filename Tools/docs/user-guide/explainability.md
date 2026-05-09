# Model Explainability

Understand and interpret model predictions using SHAP, PDP, and permutation importance.

## Quick Start

### Dashboard Explainability

1. Go to **🔍 Explainability** page
2. Load model
3. Select patient or dataset
4. View explanations

### Python API

```python
from src.explainability.shap_analysis import SHAPAnalyzer

# Initialize
analyzer = SHAPAnalyzer(model, X_train)

# Global importance
analyzer.plot_summary(
    X_test,
    save_path="processed/plots/explainability/shap_summary.png"
)

# Individual explanation
analyzer.plot_waterfall(
    X_test.iloc[0],
    save_path="processed/plots/explainability/patient_0.png"
)
```

## SHAP Analysis

### Summary Plot

```python
analyzer.plot_summary(X_test)
```

### Dependence Plot

```python
from src.explainability.shap_analysis import plot_shap_dependence

plot_shap_dependence(
    model, X_train,
    feature="age",
    save_path="processed/plots/explainability/age_dependence.png"
)
```

## Partial Dependence Plots

```python
from src.explainability.partial_dependence import plot_pdp

plot_pdp(
    model, X_train,
    features=["age", "heart_rate"],
    save_path="processed/plots/explainability/pdp.png"
)
```

## Permutation Importance

```python
from src.explainability.permutation import calculate_permutation_importance

importance = calculate_permutation_importance(model, X_test, y_test)
print(importance)
```

## See Also

- [API: SHAP](../api/explainability/shap_analysis.md)
- [API: PDP](../api/explainability/partial_dependence.md)
- [Evaluation](evaluation.md)
