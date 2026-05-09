# Exploratory Data Analysis

Comprehensive EDA guide for AMI patient data analysis.

## Overview

EDA helps you understand your data before building models. The system provides automated analysis and visualizations.

## Quick Start

### Dashboard EDA

1. Navigate to **📊 Data Overview** page
2. View summary statistics
3. Explore distributions
4. Analyze correlations
5. Generate PDF report

### Python API

```python
from src.eda.analyzer import EDAAnalyzer

# Initialize analyzer
analyzer = EDAAnalyzer(df, target="mortality_inhospital")

# Summary statistics
summary = analyzer.summary_statistics()
print(summary)

# Generate full report
analyzer.generate_report(save_path="processed/plots/eda/report.pdf")
```

## Univariate Analysis

### Continuous Variables

```python
from src.eda.univariate import analyze_continuous

# Analyze single variable
stats = analyze_continuous(df, "age")
print(f"Mean: {stats['mean']:.2f}")
print(f"Median: {stats['median']:.2f}")
print(f"Std: {stats['std']:.2f}")

# Plot distribution
from src.eda.visualizations import plot_distribution
plot_distribution(df, "age", save_path="processed/plots/eda/age_dist.png")
```

### Categorical Variables

```python
from src.eda.univariate import analyze_categorical

# Analyze categories
stats = analyze_categorical(df, "sex")
print(stats['value_counts'])

# Plot
from src.eda.visualizations import plot_categorical
plot_categorical(df, "sex", save_path="processed/plots/eda/sex_dist.png")
```

## Bivariate Analysis

### Continuous vs Target

```python
from src.eda.bivariate import analyze_continuous_target

# Compare distributions by target
stats = analyze_continuous_target(df, "age", "mortality_inhospital")
print(f"Mean (survived): {stats['mean_0']:.2f}")
print(f"Mean (died): {stats['mean_1']:.2f}")
print(f"p-value: {stats['p_value']:.4f}")

# Plot
from src.eda.visualizations import plot_continuous_by_target
plot_continuous_by_target(
    df, "age", "mortality_inhospital",
    save_path="processed/plots/eda/age_by_target.png"
)
```

### Categorical vs Target

```python
from src.eda.bivariate import analyze_categorical_target

# Chi-square test
stats = analyze_categorical_target(df, "sex", "mortality_inhospital")
print(f"Chi-square: {stats['chi_square']:.2f}")
print(f"p-value: {stats['p_value']:.4f}")
```

## Multivariate Analysis

### Correlation Matrix

```python
from src.eda.multivariate import calculate_correlations

# Calculate
corr_matrix = calculate_correlations(df)

# Plot
from src.eda.visualizations import plot_correlation_matrix
plot_correlation_matrix(
    df,
    save_path="processed/plots/eda/correlation_matrix.png"
)
```

### PCA

```python
from src.eda.multivariate import perform_pca

# Apply PCA
pca_result = perform_pca(df, n_components=2)

# Plot
from src.eda.visualizations import plot_pca
plot_pca(
    pca_result,
    target=df["mortality_inhospital"],
    save_path="processed/plots/eda/pca.png"
)
```

## Missing Value Analysis

```python
from src.eda.analyzer import EDAAnalyzer

analyzer = EDAAnalyzer(df)

# Missing value report
missing_report = analyzer.missing_value_analysis()
print(missing_report)

# Visualize
from src.eda.visualizations import plot_missing_values
plot_missing_values(df, save_path="processed/plots/eda/missing.png")
```

## Complete EDA Report

```python
from src.eda.analyzer import EDAAnalyzer

# Initialize
analyzer = EDAAnalyzer(df, target="mortality_inhospital")

# Generate comprehensive report
report = analyzer.generate_full_report(
    output_dir="processed/plots/eda/",
    pdf_path="processed/plots/eda/full_report.pdf"
)

print("Report generated!")
print(f"- Plots: {report['n_plots']}")
print(f"- Variables: {report['n_variables']}")
print(f"- PDF: {report['pdf_path']}")
```

## See Also

- [API: EDA Analyzer](../api/eda/analyzer.md)
- [API: Visualizations](../api/eda/visualizations.md)
- [Data Cleaning](data-cleaning.md)
