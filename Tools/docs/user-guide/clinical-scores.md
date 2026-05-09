# Clinical Scores

Calculate traditional clinical risk scores for AMI patients and compare with ML predictions.

## Available Scores

### GRACE Score

Global Registry of Acute Coronary Events risk score:

```python
from src.scoring.grace import calculate_grace_score

score = calculate_grace_score(
    age=65,
    heart_rate=85,
    systolic_bp=140,
    creatinine=1.2,
    killip_class=1,
    cardiac_arrest=False,
    st_segment_deviation=True,
    elevated_cardiac_enzymes=True
)

print(f"GRACE Score: {score}")
print(f"Risk category: {grace_risk_category(score)}")
```

### TIMI Score

Thrombolysis In Myocardial Infarction risk score:

```python
from src.scoring.timi import calculate_timi_score

score = calculate_timi_score(
    age=65,
    diabetes=True,
    hypertension=True,
    angina=False,
    prior_mi=False,
    prior_aspirin=True,
    st_deviation=True,
    num_risk_factors=3
)

print(f"TIMI Score: {score}")
```

## Comparison with ML

```python
# Get ML prediction
ml_risk = predictor.predict_proba(patient_data)

# Get GRACE score
grace_score = calculate_grace_score(**patient_data)
grace_risk = grace_to_probability(grace_score)

# Compare
print(f"ML Risk: {ml_risk:.2%}")
print(f"GRACE Risk: {grace_risk:.2%}")
```

## See Also

- [API: GRACE](../api/scoring/grace.md)
- [API: TIMI](../api/scoring/timi.md)
- [Predictions](predictions.md)
