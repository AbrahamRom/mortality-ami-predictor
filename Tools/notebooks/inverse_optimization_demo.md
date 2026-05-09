# Inverse Optimization - Quick Demo

This notebook demonstrates the inverse optimization functionality for treatment recommendations.

## Setup

```python
import sys
from pathlib import Path

# Add src to path
root_dir = Path.cwd().parent if 'notebooks' in str(Path.cwd()) else Path.cwd()
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from src.explainability import (
    InverseOptimizer,
    plot_optimal_values_comparison,
    plot_confidence_intervals,
    plot_sensitivity_analysis,
    plot_feature_importance_for_optimization,
    create_optimization_summary_figure
)

print("✅ Setup complete!")
```

## 1. Create Sample Dataset

For this demo, we'll simulate a medical dataset with treatment features.

```python
# Set random seed
np.random.seed(42)

# Create synthetic patient data
n_patients = 500

# Patient characteristics (non-modifiable)
age = np.random.randint(45, 85, n_patients)
sex = np.random.binomial(1, 0.6, n_patients)
diabetes = np.random.binomial(1, 0.3, n_patients)
hypertension = np.random.binomial(1, 0.5, n_patients)

# Treatment variables (modifiable)
aspirin_dose = np.random.uniform(0, 325, n_patients)
statin_dose = np.random.uniform(0, 80, n_patients)
beta_blocker = np.random.binomial(1, 0.7, n_patients)
reperfusion_time = np.random.uniform(30, 180, n_patients)

# Create mortality outcome (influenced by age, treatments, etc.)
mortality_prob = (
    0.3 * (age / 100) +  # Age effect
    0.2 * diabetes +  # Diabetes effect
    0.15 * hypertension +  # Hypertension effect
    -0.001 * aspirin_dose +  # Aspirin benefit
    -0.002 * statin_dose +  # Statin benefit
    -0.15 * beta_blocker +  # Beta blocker benefit
    0.001 * reperfusion_time +  # Time is critical
    np.random.normal(0, 0.1, n_patients)  # Random noise
)

# Clip to [0, 1] and create binary outcome
mortality_prob = np.clip(mortality_prob, 0, 1)
mortality = (np.random.rand(n_patients) < mortality_prob).astype(int)

# Create DataFrame
df = pd.DataFrame({
    'age': age,
    'sex': sex,
    'diabetes': diabetes,
    'hypertension': hypertension,
    'aspirin_dose': aspirin_dose,
    'statin_dose': statin_dose,
    'beta_blocker': beta_blocker,
    'reperfusion_time': reperfusion_time,
    'mortality': mortality
})

print(f"Dataset created: {len(df)} patients")
print(f"Mortality rate: {df['mortality'].mean():.1%}")
df.head()
```

## 2. Train a Model

```python
# Prepare data
X = df.drop('mortality', axis=1)
y = df['mortality']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train model
model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train, y_train)

# Evaluate
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)

print(f"Training accuracy: {train_score:.3f}")
print(f"Test accuracy: {test_score:.3f}")
```

## 3. Create Optimizer

```python
# Initialize optimizer
optimizer = InverseOptimizer(
    model=model,
    feature_names=X.columns.tolist(),
    feature_bounds={
        'aspirin_dose': (0, 325),
        'statin_dose': (0, 80),
        'reperfusion_time': (30, 180),
    }
)

print("✅ Optimizer initialized!")
```

## 4. Optimize for a High-Risk Patient

```python
# Select a high-risk patient
high_risk_patients = df[df['mortality'] == 1]
patient_idx = high_risk_patients.index[0]
patient = df.iloc[patient_idx]

print("Original patient data:")
print(patient)
print(f"\nPredicted mortality: {model.predict_proba(patient.drop('mortality').values.reshape(1, -1))[0, 1]:.1%}")
```

```python
# Define what we can modify (treatments) and what we can't (patient characteristics)
modifiable_features = ['aspirin_dose', 'statin_dose', 'beta_blocker', 'reperfusion_time']
fixed_features = {
    'age': patient['age'],
    'sex': patient['sex'],
    'diabetes': patient['diabetes'],
    'hypertension': patient['hypertension']
}

print("Modifiable features:", modifiable_features)
print("Fixed features:", fixed_features)
```

## 5. Run Optimization

```python
# Optimize to reduce mortality to <10%
result = optimizer.optimize(
    target_value=0.1,  # Target 10% mortality
    modifiable_features=modifiable_features,
    fixed_features=fixed_features,
    reference_data=X,
    method='SLSQP',
    n_iterations=10,
    random_state=42
)

print("\n" + "="*60)
print("OPTIMIZATION RESULTS")
print("="*60)
print(f"Target mortality: {0.1:.1%}")
print(f"Achieved mortality: {result['achieved_prediction']:.1%}")
print(f"Success: {'✅ Yes' if result['success'] else '❌ No'}")
print(f"Distance to target: {result['distance_to_target']:.6f}")
print(f"\nOptimal treatment values:")
for feat, val in result['optimal_values'].items():
    if feat in modifiable_features:
        original = patient[feat]
        change = val - original
        print(f"  {feat}: {val:.2f} (original: {original:.2f}, change: {change:+.2f})")
```

## 6. Visualize Results

```python
# Get original values
original_values = {feat: patient[feat] for feat in modifiable_features}

# Comparison plot
fig = plot_optimal_values_comparison(
    original_values=original_values,
    optimal_values={k: v for k, v in result['optimal_values'].items() if k in modifiable_features},
    feature_names=modifiable_features,
    title="Original vs Optimal Treatment Strategy"
)
fig.show()
```

```python
# Summary figure
fig_summary = create_optimization_summary_figure(
    result=result,
    original_values=original_values
)
fig_summary.show()
```

## 7. Confidence Intervals

Quantify uncertainty in the optimal values.

```python
# Compute confidence intervals
ci_result = optimizer.compute_confidence_intervals(
    target_value=0.1,
    modifiable_features=modifiable_features,
    fixed_features=fixed_features,
    reference_data=X,
    n_bootstrap=30,  # Use 30 for demo (use 50+ in production)
    confidence_level=0.95,
    method='SLSQP',
    random_state=42
)

print(f"Successfully computed {ci_result['n_successful']}/{ci_result['n_bootstrap']} bootstrap samples")
print(f"\nOptimal values with 95% confidence intervals:")
for feat, stats in ci_result['confidence_intervals'].items():
    print(f"  {feat}:")
    print(f"    Median: {stats['median']:.2f}")
    print(f"    95% CI: [{stats['lower_ci']:.2f}, {stats['upper_ci']:.2f}]")
    print(f"    Std: {stats['std']:.2f}")
```

```python
# Visualize confidence intervals
fig_ci = plot_confidence_intervals(
    ci_results=ci_result['confidence_intervals'],
    optimal_values={k: v for k, v in result['optimal_values'].items() if k in modifiable_features}
)
fig_ci.show()
```

## 8. Sensitivity Analysis

Understand how sensitive the prediction is to changes in optimal values.

```python
# Compute sensitivity
sensitivity_df = optimizer.sensitivity_analysis(
    optimal_values=result['optimal_values'],
    modifiable_features=modifiable_features,
    fixed_features=fixed_features,
    perturbation_percent=15.0,
    n_points=20
)

print(f"Computed sensitivity for {len(sensitivity_df)} data points")
sensitivity_df.head(10)
```

```python
# Plot sensitivity curves
fig_sens = plot_sensitivity_analysis(
    sensitivity_df=sensitivity_df,
    target_value=0.1,
    title="How Prediction Changes with Treatment Adjustments"
)
fig_sens.show()
```

```python
# Feature importance for optimization
fig_importance = plot_feature_importance_for_optimization(
    sensitivity_df=sensitivity_df,
    title="Which Treatments Matter Most?"
)
fig_importance.show()
```

## 9. Multiple Scenarios

Compare different target mortality levels.

```python
# Try different targets
targets = [0.05, 0.10, 0.15, 0.20]
results = []

print("Optimizing for different target mortality levels...")
for target in targets:
    res = optimizer.optimize(
        target_value=target,
        modifiable_features=modifiable_features,
        fixed_features=fixed_features,
        reference_data=X,
        method='SLSQP',
        n_iterations=5,
        random_state=42
    )
    results.append(res)
    print(f"  Target {target:.1%}: Achieved {res['achieved_prediction']:.1%}, Success: {res['success']}")
```

```python
# Compare results
comparison_df = pd.DataFrame([
    {
        'Target': f"{target:.1%}",
        'Achieved': f"{res['achieved_prediction']:.1%}",
        'Aspirin': res['optimal_values'].get('aspirin_dose', 0),
        'Statin': res['optimal_values'].get('statin_dose', 0),
        'Beta Blocker': res['optimal_values'].get('beta_blocker', 0),
        'Reperfusion Time': res['optimal_values'].get('reperfusion_time', 0),
        'Success': '✅' if res['success'] else '❌'
    }
    for target, res in zip(targets, results)
])

comparison_df
```

## 10. Clinical Interpretation

```python
print("="*70)
print("CLINICAL RECOMMENDATIONS")
print("="*70)
print(f"\nPatient Profile:")
print(f"  Age: {patient['age']:.0f} years")
print(f"  Sex: {'Male' if patient['sex'] == 1 else 'Female'}")
print(f"  Diabetes: {'Yes' if patient['diabetes'] == 1 else 'No'}")
print(f"  Hypertension: {'Yes' if patient['hypertension'] == 1 else 'No'}")

print(f"\nOriginal Treatment (Predicted Mortality: {model.predict_proba(patient.drop('mortality').values.reshape(1, -1))[0, 1]:.1%}):")
print(f"  Aspirin: {patient['aspirin_dose']:.1f} mg")
print(f"  Statin: {patient['statin_dose']:.1f} mg")
print(f"  Beta Blocker: {'Yes' if patient['beta_blocker'] == 1 else 'No'}")
print(f"  Reperfusion Time: {patient['reperfusion_time']:.1f} min")

optimal = result['optimal_values']
print(f"\nOptimized Treatment (Predicted Mortality: {result['achieved_prediction']:.1%}):")
print(f"  Aspirin: {optimal['aspirin_dose']:.1f} mg ({optimal['aspirin_dose'] - patient['aspirin_dose']:+.1f})")
print(f"  Statin: {optimal['statin_dose']:.1f} mg ({optimal['statin_dose'] - patient['statin_dose']:+.1f})")
print(f"  Beta Blocker: {'Yes' if optimal['beta_blocker'] >= 0.5 else 'No'}")
print(f"  Reperfusion Time: {optimal['reperfusion_time']:.1f} min ({optimal['reperfusion_time'] - patient['reperfusion_time']:+.1f})")

print(f"\n⚠️  IMPORTANT: These are algorithmic recommendations based on the model.")
print(f"    Always validate with clinical experts and consider:")
print(f"    - Patient-specific contraindications")
print(f"    - Drug interactions")
print(f"    - Clinical guidelines")
print(f"    - Practical feasibility")
```

## Summary

This notebook demonstrated:

1. ✅ Creating an inverse optimizer
2. ✅ Finding optimal treatment values
3. ✅ Computing confidence intervals
4. ✅ Performing sensitivity analysis
5. ✅ Comparing multiple scenarios
6. ✅ Visualizing results interactively

## Next Steps

- Try with your real AMI mortality model
- Use the Streamlit interface for interactive exploration
- Integrate with SHAP for feature importance analysis
- Export results for clinical review

---

**For more information, see:**
- Documentation: `docs/user-guide/inverse-optimization.md`
- Source code: `src/explainability/inverse_optimization.py`
- Tests: `tests/test_inverse_optimization.py`
