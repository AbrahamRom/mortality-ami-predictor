# 🎯 Inverse Optimization Module

## Quick Overview

The **Inverse Optimization Module** answers the question: 

> *"What feature values should I set to achieve my desired prediction?"*

This is the **reverse** of typical ML prediction:
- **Normal**: Features → Model → Prediction
- **Inverse**: Desired Prediction → **Optimizer** → Optimal Features

## 🏥 Real-World Example

**Scenario**: A 68-year-old patient has 65% predicted mortality risk after AMI.

**Question**: What treatment strategy reduces this to <20%?

**Solution**:
```python
from src.explainability import InverseOptimizer

optimizer = InverseOptimizer(model, feature_names)

result = optimizer.optimize(
    target_value=0.2,  # 20% mortality
    modifiable_features=['aspirin', 'statin', 'reperfusion_time'],
    fixed_features={'age': 68, 'sex': 1}
)

print(result['optimal_values'])
# {'aspirin': 162.5, 'statin': 40.0, 'reperfusion_time': 45.2}
```

## 🚀 Quick Start

### 1. Installation

All dependencies are already in `requirements.txt`. Key package: `scipy`

### 2. Basic Usage

```python
import joblib
import pandas as pd
from src.explainability import InverseOptimizer

# Load model and data
model = joblib.load('models/my_model.joblib')
df = pd.read_csv('data/patients.csv')

# Create optimizer
optimizer = InverseOptimizer(model, df.columns.tolist())

# Find optimal values
result = optimizer.optimize(
    target_value=0,  # Survival
    modifiable_features=['treatment_A', 'treatment_B'],
    reference_data=df
)

print("Optimal treatment:", result['optimal_values'])
print("Predicted outcome:", result['achieved_prediction'])
```

### 3. Streamlit Interface

```bash
streamlit run dashboard/Dashboard.py
```

Navigate to **🎯 Inverse Optimization** page.

## 📊 Key Features

### ✅ What It Does

- **Treatment Optimization**: Find best medication/intervention strategy
- **Counterfactuals**: Minimal changes to flip prediction
- **Uncertainty Quantification**: Bootstrap confidence intervals
- **Sensitivity Analysis**: Robustness to perturbations
- **Interactive Visualizations**: Modern Plotly charts

### 🔧 Technical Features

- **Multiple Optimizers**: SLSQP, COBYLA, Differential Evolution
- **Random Restarts**: Avoid local minima
- **Feature Bounds**: Enforce realistic constraints
- **Fixed Features**: Lock unchangeable patient characteristics
- **Parallel Analysis**: Compare multiple scenarios

## 📖 Documentation

- **Full Guide**: `docs/user-guide/inverse-optimization.md`
- **API Reference**: See docstrings in `src/explainability/inverse_optimization.py`
- **Examples**: See "Example Workflows" in documentation

## 🧪 Testing

```bash
# Run tests
pytest tests/test_inverse_optimization.py -v

# With coverage
pytest tests/test_inverse_optimization.py --cov=src.explainability.inverse_optimization
```

## 🎨 Visualizations

All plots are interactive Plotly figures:

1. **Comparison Plot**: Original vs Optimal values
2. **Confidence Intervals**: Bootstrap uncertainty
3. **Sensitivity Curves**: Prediction vs feature perturbations
4. **Sensitivity Heatmap**: Overview of all features
5. **Feature Importance**: Which features matter most
6. **Summary Dashboard**: Comprehensive overview

## ⚠️ Important Notes

### Clinical Validation Required

**This is a decision support tool, not a decision maker!**

- Always validate with medical professionals
- Check clinical feasibility
- Consider contraindications
- Respect guidelines

### Model Limitations

- Results depend on model quality
- Don't extrapolate beyond training data
- Correlation ≠ Causation
- Uncertainty matters!

### Best Practices

✅ **DO**:
```python
# Use confidence intervals
ci = optimizer.compute_confidence_intervals(...)
print(f"Dose: {ci['median']} [{ci['lower_ci']}, {ci['upper_ci']}]")

# Multiple restarts
result = optimizer.optimize(..., n_iterations=10)

# Sensitivity analysis
sensitivity = optimizer.sensitivity_analysis(...)
```

❌ **DON'T**:
```python
# Single point estimate
result = optimizer.optimize(..., n_iterations=1)
print(result['optimal_values'])  # No uncertainty!

# Too many features
optimizer.optimize(modifiable_features=all_100_features)  # Will fail

# Ignore feasibility
# Use optimal values without checking if realistic
```

## 🔬 Scientific Background

### Optimization Methods

1. **SLSQP** (Sequential Least Squares Programming)
   - Gradient-based local optimization
   - Fast and accurate for smooth objectives
   - Best for: Quick analyses

2. **COBYLA** (Constrained Optimization BY Linear Approximation)
   - Gradient-free local optimization
   - Robust to noisy objectives
   - Best for: Complex models

3. **Differential Evolution**
   - Global optimization
   - Explores entire search space
   - Best for: Finding global optimum (slower)

### References

- Wachter et al. (2017) - Counterfactual Explanations
- Ustun et al. (2019) - Actionable Recourse
- SciPy Optimize Documentation

## 🤝 Integration

### With Other Modules

```python
# With SHAP
from src.explainability import compute_shap_values, InverseOptimizer

# 1. Understand what drives predictions
shap_values = compute_shap_values(model, X)

# 2. Optimize important features
important_features = get_top_features_from_shap(shap_values)
result = optimizer.optimize(modifiable_features=important_features, ...)
```

### Export Results

```python
# CSV
import pandas as pd
pd.DataFrame([result['optimal_values']]).to_csv('optimal.csv')

# JSON
import json
with open('result.json', 'w') as f:
    json.dump(result, f, indent=2)
```

## 📈 Performance

### Speed Guidelines

| Features | Method | Time |
|----------|--------|------|
| 1-5 | SLSQP | <1s |
| 5-10 | SLSQP | 1-5s |
| 10-20 | COBYLA | 5-30s |
| Any | Differential Evolution | 30s-5min |

**Tips for Speed**:
- Use SLSQP for initial exploration
- Reduce `n_iterations` (3-5 is often enough)
- Use fewer features
- Skip CI computation for quick tests

### Memory

Very low memory footprint (~MB). Suitable for web applications.

## 🐛 Troubleshooting

### Common Issues

**1. Optimization doesn't converge**
```python
# Solution: More restarts, different method
result = optimizer.optimize(
    ...,
    method='differential_evolution',  # Global search
    n_iterations=20  # More restarts
)
```

**2. Optimal values seem unrealistic**
```python
# Solution: Set feature bounds
optimizer = InverseOptimizer(
    model,
    features,
    feature_bounds={
        'drug_dose': (0, 100),  # Realistic range
        'time': (0, 240)
    }
)
```

**3. Very slow computation**
```python
# Solution: Reduce complexity
result = optimizer.optimize(
    modifiable_features=features[:5],  # Fewer features
    method='SLSQP',  # Faster method
    n_iterations=3  # Fewer restarts
)
```

**4. Bootstrap CI fails**
```python
# Solution: Reduce iterations
ci = optimizer.compute_confidence_intervals(
    ...,
    n_bootstrap=20  # Fewer iterations
)
```

## 💡 Tips & Tricks

### 1. Start Simple

```python
# Begin with 2-3 features
result = optimizer.optimize(
    modifiable_features=['feature1', 'feature2']
)
# Then gradually add more
```

### 2. Use Reference Patients

```python
# Base optimization on existing patient
reference_patient = df[df['similar_condition'] == 1].iloc[0]

result = optimizer.optimize(
    ...,
    initial_values=reference_patient.to_dict(),
    fixed_features={k: v for k, v in reference_patient.items() 
                   if k not in modifiable}
)
```

### 3. Compare Multiple Targets

```python
# Try different target values
targets = [0.1, 0.2, 0.3, 0.4, 0.5]
results = []

for target in targets:
    result = optimizer.optimize(target_value=target, ...)
    results.append(result)

# Find sweet spot
```

### 4. Feature Selection

```python
# Use SHAP to identify important features
shap_importance = get_feature_importance(shap_values)
top_features = shap_importance.head(10).index.tolist()

# Only optimize important features
result = optimizer.optimize(modifiable_features=top_features, ...)
```

## 📞 Support

- **Documentation**: `docs/user-guide/inverse-optimization.md`
- **Issues**: GitHub Issues
- **Examples**: See documentation and tests

## 🎓 Learning Path

1. **Beginner**: Run basic optimization with 2-3 features
2. **Intermediate**: Add confidence intervals and sensitivity analysis
3. **Advanced**: Multiple scenarios, counterfactuals, custom constraints
4. **Expert**: Integration with other modules, custom optimizers

## 📝 Changelog

### Version 1.0.0 (November 2025)
- ✨ Initial release
- ✅ Three optimization methods (SLSQP, COBYLA, DE)
- ✅ Bootstrap confidence intervals
- ✅ Sensitivity analysis
- ✅ Interactive Streamlit interface
- ✅ Comprehensive visualizations
- ✅ Full test suite
- ✅ Complete documentation

---

**Made with ❤️ for better clinical decision support**

*Part of the AMI Mortality Prediction System*
