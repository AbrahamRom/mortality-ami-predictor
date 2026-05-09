# Quick Start Guide

Get up and running with Mortality AMI Predictor in just a few minutes!

## 🎯 Overview

This guide will walk you through:

1. Loading your first dataset
2. Cleaning and preprocessing data
3. Training a model
4. Making predictions
5. Evaluating results

## Step 1: Launch the Dashboard

```bash
cd Tools
streamlit run dashboard/Dashboard.py
```

The dashboard will open in your browser at `http://localhost:8501`.

## Step 2: Load and Clean Data

### Using the Dashboard

1. Navigate to **🧹 Data Cleaning and EDA** page
2. Upload your dataset or use the default: `DATA/recuima-020425-fragment.csv`
3. Click **"Clean Dataset"**
4. Review the cleaning report

### Using Python API

```python
from src.cleaning.cleaner import DataCleaner
from src.data_load.loaders import load_dataset
import pandas as pd

# Load raw data
df = load_dataset("../DATA/recuima-020425-fragment.csv")

# Initialize cleaner
cleaner = DataCleaner(
    target_column="mortality_inhospital",
    metadata_path="../DATA/variable_metadata.json"
)

# Clean data
cleaned_df = cleaner.clean(df)

# Save cleaned data
cleaned_df.to_csv("processed/cleaned_datasets/my_cleaned_data.csv", index=False)
```

## Step 3: Explore Your Data

### Dashboard EDA

1. Go to **📊 Data Overview** page
2. View variable distributions
3. Analyze correlations
4. Generate interactive plots

### Python EDA

```python
from src.eda.analyzer import EDAAnalyzer
from src.eda.visualizations import plot_univariate, plot_correlation_matrix

# Initialize analyzer
analyzer = EDAAnalyzer(cleaned_df, target="mortality_inhospital")

# Generate summary statistics
summary = analyzer.summary_statistics()
print(summary)

# Plot distributions
plot_univariate(cleaned_df, "age", save_path="processed/plots/eda/age_dist.png")

# Correlation matrix
plot_correlation_matrix(cleaned_df, save_path="processed/plots/eda/corr_matrix.png")
```

## Step 4: Train Your First Model

### Dashboard Training

1. Navigate to **🤖 Model Training** page
2. Select features to include
3. Choose a model type (e.g., Random Forest)
4. Configure hyperparameters
5. Click **"Train Model"**
6. View training metrics and plots

### Python Training

```python
from src.training.trainer import ModelTrainer
from src.data_load.splitters import split_data

# Prepare data
X = cleaned_df.drop(columns=["mortality_inhospital"])
y = cleaned_df["mortality_inhospital"]

# Split data
X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)

# Train Random Forest
trainer = ModelTrainer(
    model_type="random_forest",
    params={
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_split": 5
    }
)

model = trainer.train(X_train, y_train)

# Save model
trainer.save_model(model, "processed/models/random_forest/my_rf_model.joblib")
```

## Step 5: Make Predictions

### Dashboard Predictions

1. Go to **🔮 Predictions** page
2. Load your trained model
3. Input patient data manually or upload batch
4. View risk predictions with confidence intervals

### Python Predictions

```python
from src.prediction.predictor import Predictor
import joblib

# Load trained model
model = joblib.load("processed/models/random_forest/my_rf_model.joblib")

# Create predictor
predictor = Predictor(model)

# Single patient prediction
patient_data = {
    "age": 65,
    "sex": 1,
    "systolic_bp": 140,
    "heart_rate": 85,
    # ... other features
}

# Get prediction
risk_score = predictor.predict_proba(patient_data)
prediction = predictor.predict(patient_data)

print(f"Mortality Risk: {risk_score:.2%}")
print(f"Prediction: {'High Risk' if prediction == 1 else 'Low Risk'}")
```

## Step 6: Evaluate Performance

### Dashboard Evaluation

1. Navigate to **📈 Model Evaluation** page
2. Load model and test data
3. View metrics:
   - ROC curve and AUC
   - Calibration plot
   - Decision curve analysis
   - Bootstrap confidence intervals

### Python Evaluation

```python
from src.evaluation.metrics import calculate_all_metrics
from src.evaluation.reporters import plot_roc_curve, plot_calibration

# Calculate metrics
metrics = calculate_all_metrics(model, X_test, y_test)

print(f"AUC: {metrics['auc']:.3f}")
print(f"Accuracy: {metrics['accuracy']:.3f}")
print(f"Sensitivity: {metrics['sensitivity']:.3f}")
print(f"Specificity: {metrics['specificity']:.3f}")

# Plot ROC curve
plot_roc_curve(model, X_test, y_test, save_path="processed/plots/evaluation/roc.png")

# Plot calibration
plot_calibration(model, X_test, y_test, save_path="processed/plots/evaluation/calib.png")
```

## Step 7: Explain Predictions

### Dashboard Explainability

1. Go to **🔍 Explainability** page
2. Select a patient or group
3. View:
   - SHAP waterfall plots
   - Feature importance
   - Partial dependence plots
   - Individual explanations

### Python Explainability

```python
from src.explainability.shap_analysis import SHAPAnalyzer

# Initialize analyzer
shap_analyzer = SHAPAnalyzer(model, X_train)

# Global feature importance
shap_analyzer.plot_summary(X_test, save_path="processed/plots/explainability/shap_summary.png")

# Individual explanation
shap_analyzer.plot_waterfall(
    X_test.iloc[0],
    save_path="processed/plots/explainability/patient_explanation.png"
)
```

## Complete Example Workflow

Here's a complete end-to-end example:

```python
from src.cleaning.cleaner import DataCleaner
from src.data_load.loaders import load_dataset
from src.training.trainer import ModelTrainer
from src.evaluation.metrics import calculate_all_metrics
from src.prediction.predictor import Predictor

# 1. Load and clean
df = load_dataset("../DATA/recuima-020425-fragment.csv")
cleaner = DataCleaner(target_column="mortality_inhospital")
cleaned_df = cleaner.clean(df)

# 2. Prepare data
X = cleaned_df.drop(columns=["mortality_inhospital"])
y = cleaned_df["mortality_inhospital"]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Train model
trainer = ModelTrainer(model_type="xgboost")
model = trainer.train(X_train, y_train)

# 4. Evaluate
metrics = calculate_all_metrics(model, X_test, y_test)
print(f"Model AUC: {metrics['auc']:.3f}")

# 5. Make predictions
predictor = Predictor(model)
new_patient = X_test.iloc[0]
risk = predictor.predict_proba(new_patient)
print(f"Patient mortality risk: {risk:.2%}")
```

## Next Steps

Now that you've completed the quick start:

- **[Data Cleaning](../user-guide/data-cleaning.md)** - Learn advanced cleaning techniques
- **[Model Training](../user-guide/training.md)** - Explore all available models
- **[Custom Models](../user-guide/custom-models.md)** - Create your own models
- **[API Reference](../api/index.md)** - Detailed API documentation

## Tips for Success

!!! tip "Use Fragment Data for Testing"
    Start with `recuima-020425-fragment.csv` for faster iteration during development.

!!! tip "Save Your Work"
    All models, plots, and datasets are automatically saved in the `processed/` directory with timestamps.

!!! tip "Experiment Tracking"
    Enable MLflow to track all your experiments: `mlflow ui --port 5000`

!!! tip "Keyboard Shortcuts"
    - `Ctrl+K` or `Cmd+K` to search documentation
    - `Ctrl+/` or `Cmd+/` for command palette in dashboard

---

!!! success "You're Ready!"
    You now know the basics of the Mortality AMI Predictor system. Explore the rest of the documentation to learn more advanced features!
