"""Bivariate analysis utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, f_oneway, pearsonr, spearmanr


@dataclass
class BivariateStats:
    """Bivariate statistics between two variables."""
    
    var1_name: str
    var2_name: str
    relationship_type: str  # 'num-num', 'num-cat', 'cat-cat'
    
    # For num-num
    pearson_corr: Optional[float] = None
    pearson_pvalue: Optional[float] = None
    spearman_corr: Optional[float] = None
    spearman_pvalue: Optional[float] = None
    
    # For cat-cat
    chi2_statistic: Optional[float] = None
    chi2_pvalue: Optional[float] = None
    cramers_v: Optional[float] = None
    
    # For num-cat
    anova_f: Optional[float] = None
    anova_pvalue: Optional[float] = None
    
    # Interpretation
    is_significant: bool = False
    strength: str = ""  # 'weak', 'moderate', 'strong'


def analyze_numeric_numeric(df: pd.DataFrame, var1: str, var2: str) -> BivariateStats:
    """Analyze correlation between two numerical variables.
    
    Args:
        df: DataFrame
        var1: First variable name
        var2: Second variable name
        
    Returns:
        BivariateStats object
    """
    df_clean = df[[var1, var2]].dropna()
    
    if len(df_clean) < 3:
        return BivariateStats(
            var1_name=var1, var2_name=var2, relationship_type='num-num'
        )
    
    # Pearson correlation
    pearson_r, pearson_p = pearsonr(df_clean[var1], df_clean[var2])
    
    # Spearman correlation
    spearman_r, spearman_p = spearmanr(df_clean[var1], df_clean[var2])
    
    # Strength interpretation
    abs_corr = abs(pearson_r)
    if abs_corr < 0.3:
        strength = 'weak'
    elif abs_corr < 0.7:
        strength = 'moderate'
    else:
        strength = 'strong'
    
    return BivariateStats(
        var1_name=var1,
        var2_name=var2,
        relationship_type='num-num',
        pearson_corr=float(pearson_r),
        pearson_pvalue=float(pearson_p),
        spearman_corr=float(spearman_r),
        spearman_pvalue=float(spearman_p),
        is_significant=pearson_p < 0.05,
        strength=strength
    )


def analyze_numeric_categorical(
    df: pd.DataFrame, 
    num_var: str, 
    cat_var: str
) -> BivariateStats:
    """Analyze relationship between numerical and categorical variables using ANOVA.
    
    Args:
        df: DataFrame
        num_var: Numerical variable name
        cat_var: Categorical variable name
        
    Returns:
        BivariateStats object
    """
    df_clean = df[[num_var, cat_var]].dropna()
    
    if len(df_clean) < 3:
        return BivariateStats(
            var1_name=num_var, var2_name=cat_var, relationship_type='num-cat'
        )
    
    # Group by category
    groups = [group[num_var].values for name, group in df_clean.groupby(cat_var)]
    
    # Filter groups with at least 2 values
    groups = [g for g in groups if len(g) >= 2]
    
    if len(groups) < 2:
        return BivariateStats(
            var1_name=num_var, var2_name=cat_var, relationship_type='num-cat'
        )
    
    # ANOVA
    f_stat, p_value = f_oneway(*groups)
    
    return BivariateStats(
        var1_name=num_var,
        var2_name=cat_var,
        relationship_type='num-cat',
        anova_f=float(f_stat),
        anova_pvalue=float(p_value),
        is_significant=p_value < 0.05,
        strength='significant' if p_value < 0.05 else 'not significant'
    )


def analyze_categorical_categorical(
    df: pd.DataFrame, 
    var1: str, 
    var2: str
) -> BivariateStats:
    """Analyze relationship between two categorical variables using Chi-square.
    
    Args:
        df: DataFrame
        var1: First variable name
        var2: Second variable name
        
    Returns:
        BivariateStats object
    """
    df_clean = df[[var1, var2]].dropna()
    
    if len(df_clean) < 3:
        return BivariateStats(
            var1_name=var1, var2_name=var2, relationship_type='cat-cat'
        )
    
    # Contingency table
    contingency_table = pd.crosstab(df_clean[var1], df_clean[var2])
    
    # Chi-square test
    chi2, p_value, dof, expected = chi2_contingency(contingency_table)
    
    # Cramér's V
    n = contingency_table.sum().sum()
    min_dim = min(contingency_table.shape) - 1
    cramers_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0
    
    # Strength of association
    if cramers_v < 0.1:
        strength = 'weak'
    elif cramers_v < 0.3:
        strength = 'moderate'
    else:
        strength = 'strong'
    
    return BivariateStats(
        var1_name=var1,
        var2_name=var2,
        relationship_type='cat-cat',
        chi2_statistic=float(chi2),
        chi2_pvalue=float(p_value),
        cramers_v=float(cramers_v),
        is_significant=p_value < 0.05,
        strength=strength
    )
