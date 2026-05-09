# Making Predictions

Use trained models to predict mortality risk for new AMI patients.

## Quick Start

### Dashboard Predictions

1. Go to **🔮 Predictions** page
2. Load trained model
3. Input patient data (manual or batch upload)
4. View risk scores
5. Export results

### Python API

```python
from src.prediction.predictor import Predictor
import joblib

# Load model
model = joblib.load("processed/models/random_forest/model.joblib")

# Create predictor
predictor = Predictor(model)

# Single patient
patient_data = {
    "age": 65,
    "sex": 1,
    "systolic_bp": 140,
    "heart_rate": 85,
    # ... other features
}

risk = predictor.predict_proba(patient_data)
print(f"Mortality risk: {risk:.2%}")
```

## Batch Predictions

```python
import pandas as pd

# Load new patients
new_patients = pd.read_csv("new_patients.csv")

# Predict
predictions = predictor.predict_batch(new_patients)
probabilities = predictor.predict_proba_batch(new_patients)

# Add to dataframe
new_patients['predicted_mortality'] = predictions
new_patients['risk_score'] = probabilities

# Save
new_patients.to_csv("predictions.csv", index=False)
```

## Risk Stratification

```python
def stratify_risk(risk_score):
    if risk_score < 0.1:
        return "Low"
    elif risk_score < 0.3:
        return "Moderate"
    else:
        return "High"

new_patients['risk_category'] = new_patients['risk_score'].apply(stratify_risk)
```

## See Also

- [API: Predictor](../api/prediction/predictor.md)
- [Model Evaluation](evaluation.md)
