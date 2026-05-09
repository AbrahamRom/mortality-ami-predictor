# Exploratory Data Analysis (EDA)

This module provides comprehensive tools for exploratory data analysis of clinical datasets, with specialized functionality for AMI patient data.

## Overview

The EDA module helps you understand your data through:

- **Statistical Analysis**: Descriptive statistics for all variable types
- **Visual Exploration**: Interactive and static visualizations
- **Distribution Analysis**: Univariate, bivariate, and multivariate analysis
- **Report Generation**: Automated PDF reports with all findings

## Module Components

### Core Classes

- **[`EDAAnalyzer`](analyzer.md)**: Main class orchestrating all EDA functionality
- **[`PDFReportGenerator`](pdf_reports.md)**: Generate comprehensive PDF reports

### Analysis Functions

- **[Univariate Analysis](univariate.md)**: Single variable distribution analysis
  - Histograms for continuous variables
  - Bar charts for categorical variables
  - Summary statistics (mean, median, std, quartiles)
  - Missing value analysis

- **[Bivariate Analysis](bivariate.md)**: Relationships between two variables
  - Scatter plots with trend lines
  - Box plots by category
  - Correlation analysis
  - Statistical tests (t-test, chi-square, ANOVA)

- **[Multivariate Analysis](multivariate.md)**: Multiple variable interactions
  - Correlation matrices with heatmaps
  - Pair plots
  - Principal Component Analysis (PCA)
  - Feature importance visualization

- **[Visualizations](visualizations.md)**: Specialized plotting functions
  - Interactive plots with Plotly
  - Static publication-quality plots with Matplotlib/Seaborn
  - Clinical-specific visualizations

## Quick Start

```python
from src.eda.analyzer import EDAAnalyzer

# Initialize analyzer
analyzer = EDAAnalyzer(data=df, target="mortality")

# Generate comprehensive report
analyzer.generate_full_report(
    output_dir="plots/eda",
    report_name="eda_report.pdf"
)

# Individual analyses
stats = analyzer.univariate_analysis()
correlations = analyzer.correlation_analysis()
pca_results = analyzer.pca_analysis(n_components=5)
```

## Key Features

### Automatic Variable Detection

The analyzer automatically detects variable types:

- **Continuous**: Numeric variables with many unique values
- **Categorical**: String variables or numeric with few unique values
- **Binary**: Variables with only two unique values
- **Target**: The outcome variable for supervised learning

### Missing Value Analysis

```python
# Analyze missing patterns
missing_report = analyzer.analyze_missing_values()

# Visualize missing data
analyzer.plot_missing_values(method='matrix')
analyzer.plot_missing_values(method='heatmap')
```

### Distribution Analysis

```python
# Test normality of continuous variables
normality_tests = analyzer.test_normality()

# Detect outliers
outliers = analyzer.detect_outliers(method='iqr')
```

### Target-Feature Relationships

```python
# Analyze relationship between features and target
target_analysis = analyzer.analyze_target_relationships()

# Statistical tests
test_results = analyzer.statistical_tests()
```

## PDF Reports

Generate comprehensive PDF reports with all analyses:

```python
# Full report with all analyses
analyzer.generate_full_report(
    output_dir="plots/eda",
    report_name="comprehensive_eda.pdf"
)

# Custom report with selected sections
analyzer.generate_custom_report(
    sections=['univariate', 'correlations', 'target_analysis'],
    output_file="custom_report.pdf"
)
```

Reports include:

- Dataset overview and summary statistics
- Distribution plots for all variables
- Correlation analysis with heatmaps
- Target-feature relationships
- Missing value analysis
- Outlier detection results
- Statistical test results
- Recommendations for preprocessing

## Integration with Other Modules

The EDA module integrates seamlessly with:

- **[Cleaning Module](../cleaning/index.md)**: Use insights to guide data cleaning
- **[Feature Engineering](../features/selectors.md)**: Identify important features
- **[Training Module](../training/index.md)**: Select features for modeling

## Best Practices

1. **Run EDA before cleaning**: Understand raw data patterns
2. **Document findings**: Use generated reports for reproducibility
3. **Iterative analysis**: Re-run EDA after each preprocessing step
4. **Domain expertise**: Interpret results with clinical knowledge
5. **Visual inspection**: Always visualize, don't rely only on statistics

## Related Documentation

- [User Guide: EDA](../../user-guide/eda.md)
- [Data Cleaning Guide](../../user-guide/data-cleaning.md)
- [Dashboard Page: Data Overview](../../user-guide/dashboard-pages-reference.md#page-01-data-overview)
