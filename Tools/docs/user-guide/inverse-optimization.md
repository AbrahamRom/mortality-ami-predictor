# Inverse Optimization Module

## 📖 Overview

The **Inverse Optimization Module** provides state-of-the-art functionality for finding optimal feature values that achieve desired model predictions. This is crucial for **treatment optimization**, **intervention planning**, and **actionable insights** in clinical decision support systems.

## 🎯 Key Capabilities

### Core Features

1. **Treatment Optimization**: Find optimal medication dosages and interventions
2. **Counterfactual Explanations**: Discover minimal changes to flip predictions
3. **Confidence Intervals**: Bootstrap-based uncertainty quantification
4. **Sensitivity Analysis**: Understand robustness of optimal solutions
5. **Interactive Visualizations**: Modern Plotly-based interactive charts

### Scientific Foundation

- **Constrained Optimization**: SciPy's state-of-the-art optimizers (SLSQP, COBYLA, Differential Evolution)
- **Multiple Random Restarts**: Avoid local optima
- **Bootstrap Confidence Intervals**: Quantify uncertainty
- **Gradient-Free Methods**: Compatible with any black-box model

## 🚀 Quick Start

### Basic Usage

```python
from src.explainability import InverseOptimizer
import joblib
import pandas as pd

# Load model and data
model = joblib.load('models/best_model.joblib')
df = pd.read_csv('data/patient_data.csv')

# Create optimizer
optimizer = InverseOptimizer(
    model=model,
    feature_names=df.columns.tolist()
)

# Find optimal treatment
result = optimizer.optimize(
    target_value=0,  # Survival
    modifiable_features=['medication_dose', 'intervention_time'],
    fixed_features={'age': 65, 'sex': 1},
    reference_data=df,
    method='SLSQP',
    n_iterations=10,
    random_state=42
)

print("Optimal values:", result['optimal_values'])
print("Achieved prediction:", result['achieved_prediction'])
```

### With Confidence Intervals

```python
# Compute uncertainty
ci_result = optimizer.compute_confidence_intervals(
    target_value=0,
    modifiable_features=['medication_dose', 'intervention_time'],
    fixed_features={'age': 65, 'sex': 1},
    reference_data=df,
    n_bootstrap=50,
    confidence_level=0.95,
    random_state=42
)

for feature, stats in ci_result['confidence_intervals'].items():
    print(f"{feature}: {stats['median']:.2f} "
          f"[{stats['lower_ci']:.2f}, {stats['upper_ci']:.2f}]")
```

### Sensitivity Analysis

```python
# Analyze robustness
sensitivity_df = optimizer.sensitivity_analysis(
    optimal_values=result['optimal_values'],
    modifiable_features=['medication_dose', 'intervention_time'],
    perturbation_percent=10.0,
    n_points=20
)

# Visualize
from src.explainability import plot_sensitivity_analysis
fig = plot_sensitivity_analysis(sensitivity_df, target_value=0)
fig.show()
```

## 📊 Visualization Examples

### Compare Original vs Optimal

```python
from src.explainability import plot_optimal_values_comparison

fig = plot_optimal_values_comparison(
    original_values={'medication': 50, 'time': 120},
    optimal_values=result['optimal_values']
)
fig.show()
```

### Confidence Intervals

```python
from src.explainability import plot_confidence_intervals

fig = plot_confidence_intervals(
    ci_results=ci_result['confidence_intervals'],
    optimal_values=result['optimal_values']
)
fig.show()
```

### Comprehensive Summary

```python
from src.explainability import create_optimization_summary_figure

fig = create_optimization_summary_figure(
    result=result,
    original_values={'medication': 50, 'time': 120}
)
fig.show()
```

## 🏥 Clinical Use Cases

### Use Case 1: Mortality Risk Reduction

**Goal**: Find treatment strategy to reduce 30-day mortality from 60% to <20%

```python
# Patient baseline
patient = {
    'age': 72,
    'sex': 1,
    'prior_mi': 1,
    # ... other fixed features
}

# Optimize treatment
result = optimizer.optimize(
    target_value=0.2,  # Target 20% mortality probability
    modifiable_features=[
        'aspirin_dose',
        'statin_dose',
        'beta_blocker',
        'reperfusion_time'
    ],
    fixed_features=patient,
    reference_data=historical_data,
    method='differential_evolution',  # Global optimization
    random_state=42
)

print("Recommended treatment:")
for drug, dose in result['optimal_values'].items():
    print(f"  {drug}: {dose:.2f}")
```

### Use Case 2: Intervention Timing

**Goal**: Determine optimal time window for intervention

```python
result = optimizer.optimize(
    target_value=0,  # Survival
    modifiable_features=['door_to_balloon_time'],
    fixed_features=patient_characteristics,
    reference_data=df,
    method='SLSQP'
)

optimal_time = result['optimal_values']['door_to_balloon_time']
print(f"Optimal intervention time: {optimal_time:.1f} minutes")
```

### Use Case 3: Personalized Medicine

**Goal**: Find patient-specific optimal treatment combination

```python
# For each patient
for patient_id, patient_data in patients.iterrows():
    result = optimizer.optimize(
        target_value=0,
        modifiable_features=modifiable_treatments,
        fixed_features=patient_data.to_dict(),
        reference_data=df,
        method='SLSQP',
        n_iterations=10
    )
    
    # Store recommendations
    recommendations[patient_id] = result['optimal_values']
```

## 🔬 Technical Details

### Optimization Methods

#### SLSQP (Sequential Least Squares Programming)
- **Best for**: Smooth, differentiable objectives
- **Speed**: Fast
- **Accuracy**: High for convex problems
- **Use when**: You have gradient information or smooth models

```python
result = optimizer.optimize(method='SLSQP')
```

#### COBYLA (Constrained Optimization BY Linear Approximation)
- **Best for**: Non-smooth objectives, many constraints
- **Speed**: Medium
- **Robustness**: Very high
- **Use when**: Model predictions are noisy or discontinuous

```python
result = optimizer.optimize(method='COBYLA')
```

#### Differential Evolution
- **Best for**: Finding global optimum
- **Speed**: Slower
- **Accuracy**: Best for multimodal objectives
- **Use when**: Concerned about local minima

```python
result = optimizer.optimize(method='differential_evolution')
```

### Feature Bounds

Specify realistic ranges for features:

```python
feature_bounds = {
    'medication_dose': (0, 100),  # mg
    'sbp': (80, 200),  # mmHg
    'heart_rate': (40, 150),  # bpm
}

optimizer = InverseOptimizer(
    model=model,
    feature_names=features,
    feature_bounds=feature_bounds
)
```

### Convergence Criteria

Control optimization precision:

```python
result = optimizer.optimize(
    target_value=0,
    modifiable_features=features,
    tolerance=1e-9,  # Tighter convergence
    n_iterations=20  # More restarts
)
```

## 📈 Advanced Features

### Counterfactual Explanations

Find minimal changes to flip prediction:

```python
from src.explainability import find_counterfactuals

# Patient with high mortality risk
high_risk_patient = df[df['mortality'] == 1].iloc[0]

# Find counterfactuals
counterfactuals = find_counterfactuals(
    model=model,
    instance=high_risk_patient,
    feature_names=feature_cols,
    target_class=0,  # Change to survival
    modifiable_features=modifiable_features,
    n_counterfactuals=5,
    random_state=42
)

for i, cf in enumerate(counterfactuals):
    print(f"Counterfactual {i+1}:")
    print(f"  Changes needed: {cf['optimal_values']}")
    print(f"  New prediction: {cf['achieved_prediction']:.3f}")
```

### Parallel Scenarios

Compare multiple optimization scenarios:

```python
from src.explainability import plot_parallel_coordinates

scenarios = []

# Try different target values
for target in [0.1, 0.2, 0.3]:
    result = optimizer.optimize(
        target_value=target,
        modifiable_features=features,
        reference_data=df
    )
    scenarios.append(result)

# Visualize
fig = plot_parallel_coordinates(
    scenarios=scenarios,
    feature_names=features
)
fig.show()
```

### Bootstrap Distribution Analysis

```python
from src.explainability import plot_bootstrap_distributions

# Compute CI
ci_result = optimizer.compute_confidence_intervals(...)

# Visualize distribution for specific feature
fig = plot_bootstrap_distributions(
    ci_results=ci_result['confidence_intervals'],
    feature='medication_dose'
)
fig.show()
```

## 🎨 Streamlit Interface

The module includes a comprehensive Streamlit interface at `pages/08_🎯_Inverse_Optimization.py`.

### Features

1. **Interactive Feature Selection**
   - Search and filter features
   - Separate modifiable and fixed features
   - Visual feedback

2. **Reference Patient Selection**
   - Load existing patient as baseline
   - Auto-populate initial values

3. **Real-time Optimization**
   - Progress indicators
   - Success/failure feedback
   - Detailed results

4. **Rich Visualizations**
   - Comparison plots
   - Confidence intervals
   - Sensitivity analysis
   - Feature importance

5. **Export Capabilities**
   - CSV export of optimal values
   - JSON export of full results
   - Report generation

### Running the Interface

```bash
streamlit run dashboard/Dashboard.py
```

Then navigate to **🎯 Inverse Optimization**.

## ⚠️ Important Considerations

### Clinical Validation

**Always validate computational recommendations with clinical experts!**

- Optimal values may not be clinically feasible
- Model predictions have uncertainty
- Real-world constraints may not be captured

### Model Limitations

- Results are only as good as the underlying model
- Extrapolation beyond training data is risky
- Correlation ≠ Causation

### Practical Feasibility

- Check if optimal values are achievable
- Consider patient-specific contraindications
- Account for drug interactions
- Respect clinical guidelines

### Uncertainty Quantification

**Always compute and report confidence intervals!**

```python
# GOOD: Include uncertainty
ci_result = optimizer.compute_confidence_intervals(
    target_value=0,
    modifiable_features=features,
    n_bootstrap=50
)

print(f"Optimal dose: {ci_result['confidence_intervals']['dose']['median']:.1f} "
      f"[{ci_result['confidence_intervals']['dose']['lower_ci']:.1f}, "
      f"{ci_result['confidence_intervals']['dose']['upper_ci']:.1f}]")

# BAD: Point estimate only
result = optimizer.optimize(target_value=0, ...)
print(f"Optimal dose: {result['optimal_values']['dose']:.1f}")
```

## 🧪 Testing

Run tests to ensure correctness:

```bash
pytest tests/test_inverse_optimization.py -v
```

### Test Coverage

- Basic optimization
- Multiple methods (SLSQP, COBYLA, Differential Evolution)
- Fixed features handling
- Initial values
- Confidence intervals
- Sensitivity analysis
- Counterfactual explanations
- Edge cases and error handling

## 📚 References

### Scientific Literature

1. **Wachter, S., Mittelstadt, B., & Russell, C. (2017)**
   - "Counterfactual Explanations without Opening the Black Box"
   - *Harvard Journal of Law & Technology*
   - Foundation for counterfactual explanations

2. **Ustun, B., Spangher, A., & Liu, Y. (2019)**
   - "Actionable Recourse in Linear Classification"
   - *ACM FAT\**
   - Theoretical framework for actionable ML

3. **Karimi, A. H., et al. (2020)**
   - "Model-Agnostic Counterfactual Explanations for Consequential Decisions"
   - *AISTATS*
   - Practical algorithms

4. **Mothilal, R. K., et al. (2020)**
   - "Explaining Machine Learning Classifiers through Diverse Counterfactual Explanations"
   - *ACM FAT\**
   - DiCE framework

### Technical Documentation

- [SciPy Optimization](https://docs.scipy.org/doc/scipy/reference/optimize.html)
- [Constrained Optimization](https://en.wikipedia.org/wiki/Constrained_optimization)
- [Bootstrap Methods](https://en.wikipedia.org/wiki/Bootstrapping_(statistics))

## 🤝 Contributing

To extend this module:

1. **Add new optimization methods**:
   ```python
   # In inverse_optimization.py
   if method == 'new_method':
       result = your_optimizer(objective, bounds, ...)
   ```

2. **Add new visualizations**:
   ```python
   # In inverse_plots.py
   def plot_new_visualization(...):
       fig = go.Figure(...)
       return fig
   ```

3. **Add new constraints**:
   ```python
   # Custom constraint functions
   def constraint_func(x):
       # Return 0 if satisfied, negative if violated
       return x[0] + x[1] - 10
   
   constraints = [{'type': 'ineq', 'fun': constraint_func}]
   ```

## 📝 Example Workflows

### Complete Clinical Workflow

```python
import pandas as pd
import joblib
from src.explainability import (
    InverseOptimizer,
    plot_optimal_values_comparison,
    plot_confidence_intervals,
    plot_sensitivity_analysis
)

# 1. Load model and data
model = joblib.load('models/mortality_model.joblib')
patients_df = pd.read_csv('data/patients.csv')

# 2. Define patient characteristics
patient = {
    'age': 68,
    'sex': 1,
    'diabetes': 1,
    'hypertension': 1,
    # ... non-modifiable features
}

# 3. Define modifiable interventions
treatments = [
    'aspirin_dose',
    'statin_dose',
    'beta_blocker',
    'ace_inhibitor',
    'door_to_balloon_time'
]

# 4. Create optimizer
optimizer = InverseOptimizer(
    model=model,
    feature_names=patients_df.columns.tolist(),
    feature_bounds={
        'aspirin_dose': (0, 325),
        'statin_dose': (0, 80),
        'door_to_balloon_time': (30, 180),
    }
)

# 5. Find optimal treatment
result = optimizer.optimize(
    target_value=0.15,  # Target 15% mortality
    modifiable_features=treatments,
    fixed_features=patient,
    reference_data=patients_df,
    method='differential_evolution',
    random_state=42
)

# 6. Compute uncertainty
ci_result = optimizer.compute_confidence_intervals(
    target_value=0.15,
    modifiable_features=treatments,
    fixed_features=patient,
    reference_data=patients_df,
    n_bootstrap=50,
    confidence_level=0.95,
    random_state=42
)

# 7. Sensitivity analysis
sensitivity_df = optimizer.sensitivity_analysis(
    optimal_values=result['optimal_values'],
    modifiable_features=treatments,
    fixed_features=patient,
    perturbation_percent=15.0,
    n_points=30
)

# 8. Generate report
print("=" * 60)
print("TREATMENT OPTIMIZATION REPORT")
print("=" * 60)
print(f"\nPatient Profile:")
for key, val in patient.items():
    print(f"  {key}: {val}")

print(f"\nOptimization Result:")
print(f"  Target mortality: {result['target_value']:.1%}")
print(f"  Achieved mortality: {result['achieved_prediction']:.1%}")
print(f"  Success: {'Yes' if result['success'] else 'No'}")

print(f"\nRecommended Treatment (with 95% CI):")
for treatment in treatments:
    if treatment in result['optimal_values']:
        optimal = result['optimal_values'][treatment]
        if treatment in ci_result['confidence_intervals']:
            ci = ci_result['confidence_intervals'][treatment]
            print(f"  {treatment}: {optimal:.2f} "
                  f"[{ci['lower_ci']:.2f}, {ci['upper_ci']:.2f}]")
        else:
            print(f"  {treatment}: {optimal:.2f}")

# 9. Create visualizations
fig1 = plot_optimal_values_comparison(
    original_values={t: patients_df[t].median() for t in treatments},
    optimal_values=result['optimal_values']
)
fig1.write_html('reports/treatment_comparison.html')

fig2 = plot_confidence_intervals(
    ci_results=ci_result['confidence_intervals'],
    optimal_values=result['optimal_values']
)
fig2.write_html('reports/treatment_confidence.html')

fig3 = plot_sensitivity_analysis(
    sensitivity_df=sensitivity_df,
    target_value=0.15
)
fig3.write_html('reports/treatment_sensitivity.html')

print("\n✅ Report generated successfully!")
print("   - reports/treatment_comparison.html")
print("   - reports/treatment_confidence.html")
print("   - reports/treatment_sensitivity.html")
```

## 🎓 Learning Resources

### Tutorials

1. **Basic Optimization**: Start with simple 2-3 features
2. **Confidence Intervals**: Add uncertainty quantification
3. **Sensitivity Analysis**: Understand robustness
4. **Multiple Scenarios**: Compare different strategies
5. **Clinical Integration**: Real-world application

### Best Practices

✅ **DO**:
- Start with few modifiable features
- Use multiple random restarts
- Compute confidence intervals
- Validate with domain experts
- Check sensitivity to perturbations
- Document assumptions

❌ **DON'T**:
- Trust point estimates without CI
- Optimize >20 features simultaneously
- Ignore clinical feasibility
- Extrapolate beyond training data
- Skip validation
- Implement without expert review

---

**Last Updated**: November 8, 2025  
**Version**: 1.0.0  
**Authors**: ML Team - AMI Mortality Prediction Project
