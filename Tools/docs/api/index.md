# API Reference

Welcome to the Mortality AMI Predictor API reference. This section provides detailed documentation for all modules, classes, and functions in the project.

## 📚 Module Overview

The project is organized into specialized modules, each handling a specific aspect of the ML pipeline:

### Core Modules

| Module | Description | Key Components |
|--------|-------------|----------------|
| [`data_load`](data_load/loaders.md) | Data loading and splitting utilities | Loaders, Splitters, I/O |
| [`cleaning`](cleaning/index.md) | Data cleaning and preprocessing | Cleaner, Imputation, Encoding |
| [`eda`](eda/index.md) | Exploratory data analysis | Analyzer, Visualizations, Reports |
| [`preprocessing`](preprocessing/pipelines.md) | Feature preprocessing pipelines | Scalers, Transformers |
| [`features`](features/selectors.md) | Feature engineering | Selection, Creation, Extraction |
| [`models`](models/index.md) | Model definitions and factories | Model Registry, Custom Models |
| [`training`](training/index.md) | Model training orchestration | Trainers, Cross-validation |
| [`prediction`](prediction/predictor.md) | Prediction and inference | Predictors, Batch Processing |
| [`evaluation`](evaluation/index.md) | Model evaluation and metrics | Metrics, Calibration, Validation |
| [`explainability`](explainability/shap_analysis.md) | Model interpretation | SHAP, PDP, Permutation |
| [`scoring`](scoring/grace.md) | Clinical scoring systems | GRACE, TIMI, Killip |
| [`reporting`](reporting/pdf_generator.md) | Report generation | PDF Reports, Summaries |

## 🚀 Quick Navigation

### Most Used Classes and Functions

#### Data Loading
```python
::: src.data_load.loaders.load_dataset
::: src.data_load.splitters.split_data
```

#### Data Cleaning
```python
::: src.cleaning.cleaner.DataCleaner
::: src.cleaning.imputation.ImputationStrategy
```

#### Model Training
```python
::: src.training.trainer.ModelTrainer
::: src.models.factory.ModelFactory
```

#### Evaluation
```python
::: src.evaluation.metrics.calculate_all_metrics
::: src.evaluation.reporters.EvaluationReporter
```

#### Explainability
```python
::: src.explainability.shap_analysis.SHAPAnalyzer
```

## 📖 Usage Examples

### Basic Workflow

```python
from src.data_load.loaders import load_dataset
from src.cleaning.cleaner import DataCleaner
from src.training.trainer import ModelTrainer
from src.evaluation.metrics import calculate_all_metrics

# Load data
df = load_dataset("path/to/data.csv")

# Clean data
cleaner = DataCleaner(target_column="mortality_inhospital")
cleaned_df = cleaner.clean(df)

# Train model
trainer = ModelTrainer(model_type="xgboost")
model = trainer.train(X_train, y_train)

# Evaluate
metrics = calculate_all_metrics(model, X_test, y_test)
```

### Custom Model Integration

```python
from src.models.custom_integration import CustomModelRegistry
from sklearn.base import BaseEstimator, ClassifierMixin

class MyCustomModel(BaseEstimator, ClassifierMixin):
    """Your custom model implementation."""
    pass

# Register custom model
registry = CustomModelRegistry()
registry.register_model("my_model", MyCustomModel)

# Use in training
trainer = ModelTrainer(model_type="my_model")
```

## 🔍 Search Tips

Use the search bar (press `Ctrl+K` or `Cmd+K`) to quickly find:

- **Classes**: Search by class name (e.g., "DataCleaner")
- **Functions**: Search by function name (e.g., "calculate_all_metrics")
- **Parameters**: Search by parameter name (e.g., "target_column")
- **Concepts**: Search by concept (e.g., "SHAP", "calibration", "imputation")

## 📝 Documentation Conventions

Throughout this API reference:

- **Required parameters** are marked with `*` or shown without default values
- **Optional parameters** have default values shown
- **Type hints** are provided for all parameters and return values
- **Examples** are included for most functions and classes
- **See Also** sections link to related functionality

## 🏗️ Module Structure

The API is organized hierarchically:

```
src/
├── data_load/          # Data I/O operations
├── cleaning/           # Data quality and transformation
├── eda/               # Statistical analysis and visualization
├── preprocessing/     # Feature preprocessing
├── features/          # Feature engineering
├── models/            # Model definitions
├── training/          # Training orchestration
├── prediction/        # Inference
├── evaluation/        # Performance assessment
├── explainability/    # Model interpretation
├── scoring/           # Clinical scores
└── reporting/         # Report generation
```

## 🔗 Related Resources

- [User Guide](../user-guide/dashboard.md) - High-level feature explanations
- [Architecture](../architecture/index.md) - Design patterns and structure
- [Developer Guide](../developer/index.md) - Contributing guidelines

---

## Browse by Module

<div class="grid cards" markdown>

-   :material-database:{ .lg .middle } [**Data Loading**](data_load/loaders.md)
    
    Load datasets, split data, and handle I/O operations

-   :material-broom:{ .lg .middle } [**Cleaning**](cleaning/index.md)
    
    Clean and preprocess raw data for analysis

-   :material-chart-bar:{ .lg .middle } [**EDA**](eda/index.md)
    
    Explore and visualize data patterns

-   :material-cog:{ .lg .middle } [**Preprocessing**](preprocessing/pipelines.md)
    
    Transform features for model input

-   :material-auto-fix:{ .lg .middle } [**Features**](features/selectors.md)
    
    Engineer and select features

-   :material-robot:{ .lg .middle } [**Models**](models/index.md)
    
    Define and customize ML models

-   :material-school:{ .lg .middle } [**Training**](training/index.md)
    
    Train and optimize models

-   :material-crystal-ball:{ .lg .middle } [**Prediction**](prediction/predictor.md)
    
    Make predictions on new data

-   :material-chart-line:{ .lg .middle } [**Evaluation**](evaluation/index.md)
    
    Assess model performance

-   :material-lightbulb:{ .lg .middle } [**Explainability**](explainability/shap_analysis.md)
    
    Interpret model decisions

-   :material-clipboard-pulse:{ .lg .middle } [**Scoring**](scoring/grace.md)
    
    Calculate clinical scores

-   :material-file-document:{ .lg .middle } [**Reporting**](reporting/pdf_generator.md)
    
    Generate reports and summaries

</div>
