"""Statistical tests for model comparison and validation.

This module implements rigorous statistical tests to compare different
machine learning models following the methodological protocol for
rigorous comparison of predictive models (see context.txt).

Pipeline components:
1. Normality tests: Shapiro-Wilk, D'Agostino-Pearson, Anderson-Darling
2. Parametric tests: Paired t-test, ANOVA for repeated measures
3. Non-parametric tests: Wilcoxon signed-rank (paired), Friedman test (multiple)
4. Multiple comparison corrections: Bonferroni, Holm-Bonferroni, Benjamini-Hochberg (FDR)
5. Effect size: Cohen's d, ΔAUROC

References:
- DeLong, E. R., DeLong, D. M., & Clarke-Pearson, D. L. (1988). Comparing ROC curves.
- Demšar, J. (2006). Statistical Comparisons of Classifiers over Multiple Data Sets.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import warnings

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns


@dataclass
class NormalityTestResult:
    """Results from normality tests following the protocol.
    
    Protocol: Accept normality if at least 2 of 3 tests do not reject H0.
    """
    shapiro_wilk_pvalue: float
    dagostino_pvalue: float
    anderson_statistic: float
    anderson_critical_5pct: float
    
    shapiro_wilk_normal: bool
    dagostino_normal: bool
    anderson_normal: bool
    
    is_normal: bool  # Combined decision (2 of 3 criterion)
    normal_tests_passed: int


@dataclass
class StatisticalTestResult:
    """Results from statistical comparison between models."""
    
    model1_name: str
    model2_name: str
    model1_scores: List[float]
    model2_scores: List[float]
    model1_mean: float
    model1_std: float
    model2_mean: float
    model2_std: float
    
    # Normality tests (enhanced with all 3 tests)
    model1_is_normal: bool
    model2_is_normal: bool
    model1_shapiro_pvalue: float
    model2_shapiro_pvalue: float
    
    # Comparison test
    test_used: str  # "paired t-test", "wilcoxon signed-rank", etc.
    test_statistic: float
    p_value: float
    significant: bool
    alpha: float
    
    # Effect size
    effect_size: float
    effect_size_interpretation: str
    
    # Optional fields with defaults (must come after required fields)
    model1_normality_detail: Optional[NormalityTestResult] = None
    model2_normality_detail: Optional[NormalityTestResult] = None
    delta_auroc: float = 0.0  # ΔAUROC (difference between means)
    
    def __str__(self) -> str:
        """String representation of results."""
        lines = [
            "=" * 80,
            f"Statistical Comparison: {self.model1_name} vs {self.model2_name}",
            "=" * 80,
            "",
            "Descriptive Statistics:",
            f"  {self.model1_name}: μ = {self.model1_mean:.4f}, σ = {self.model1_std:.4f}",
            f"  {self.model2_name}: μ = {self.model2_mean:.4f}, σ = {self.model2_std:.4f}",
            f"  ΔAUROC = {self.delta_auroc:.4f}",
            "",
            "Normality Tests (2-of-3 criterion: Shapiro-Wilk, D'Agostino, Anderson-Darling):",
            f"  {self.model1_name}: {'Normal' if self.model1_is_normal else 'Non-normal'} (Shapiro p={self.model1_shapiro_pvalue:.4f})",
            f"  {self.model2_name}: {'Normal' if self.model2_is_normal else 'Non-normal'} (Shapiro p={self.model2_shapiro_pvalue:.4f})",
            "",
            f"Comparison Test: {self.test_used}",
            f"  Test Statistic: {self.test_statistic:.4f}",
            f"  P-value: {self.p_value:.4f}",
            f"  Significant (α={self.alpha}): {'Yes' if self.significant else 'No'}",
            "",
            f"Effect Size (Cohen's d): {self.effect_size:.4f} ({self.effect_size_interpretation})",
            "=" * 80,
        ]
        return "\n".join(lines)


@dataclass
class MultipleComparisonResult:
    """Results from comparing multiple models simultaneously.
    
    Follows the protocol for global comparison using ANOVA or Friedman,
    followed by post-hoc tests with multiple comparison corrections.
    """
    model_names: List[str]
    model_scores: Dict[str, List[float]]
    
    # Global test
    global_test_name: str  # "ANOVA repeated measures" or "Friedman"
    global_test_statistic: float
    global_p_value: float
    global_significant: bool
    
    # Post-hoc pairwise comparisons
    pairwise_results: Dict[Tuple[str, str], StatisticalTestResult] = field(default_factory=dict)
    
    # Corrected p-values
    bonferroni_alpha: float = 0.05
    holm_corrected_pvalues: Dict[Tuple[str, str], float] = field(default_factory=dict)
    fdr_corrected_pvalues: Dict[Tuple[str, str], float] = field(default_factory=dict)
    
    def __str__(self) -> str:
        n_comparisons = len(self.pairwise_results)
        lines = [
            "=" * 80,
            f"Multiple Model Comparison ({len(self.model_names)} models)",
            "=" * 80,
            "",
            f"Global Test: {self.global_test_name}",
            f"  Test Statistic: {self.global_test_statistic:.4f}",
            f"  P-value: {self.global_p_value:.6f}",
            f"  Significant (α=0.05): {'Yes' if self.global_significant else 'No'}",
            "",
            f"Multiple Comparison Corrections ({n_comparisons} pairwise comparisons):",
            f"  Bonferroni α: {self.bonferroni_alpha:.6f}",
            "=" * 80,
        ]
        return "\n".join(lines)


def test_normality(
    scores: np.ndarray,
    alpha: float = 0.05
) -> Tuple[bool, float]:
    """Test if scores follow normal distribution using Shapiro-Wilk test.
    
    This is a simplified interface that returns only Shapiro-Wilk results.
    For the full protocol (2-of-3 criterion), use test_normality_full().
    
    Args:
        scores: Array of scores to test
        alpha: Significance level
        
    Returns:
        Tuple of (is_normal, p_value)
    """
    if len(scores) < 3:
        warnings.warn(f"Too few samples ({len(scores)}) for Shapiro-Wilk test")
        return False, 0.0
    
    # Shapiro-Wilk test
    statistic, p_value = stats.shapiro(scores)
    
    # Null hypothesis: data comes from normal distribution
    # If p > alpha, we fail to reject H0 (assume normal)
    is_normal = p_value > alpha
    
    return is_normal, float(p_value)


def test_normality_full(
    scores: np.ndarray,
    alpha: float = 0.05
) -> NormalityTestResult:
    """Test normality using 3 tests with 2-of-3 criterion (protocol compliant).
    
    According to the methodological protocol, normality is accepted when
    at least 2 of 3 formal tests do not reject H0 (p >= 0.05).
    
    Tests applied:
    1. Shapiro-Wilk: Most powerful for small to medium samples
    2. D'Agostino-Pearson: Based on skewness and kurtosis
    3. Anderson-Darling: Emphasizes tails of distribution
    
    Args:
        scores: Array of scores to test (minimum 20 samples recommended)
        alpha: Significance level (default 0.05)
        
    Returns:
        NormalityTestResult with all test results and combined decision
    """
    n = len(scores)
    
    if n < 8:
        warnings.warn(f"Too few samples ({n}) for reliable normality testing. Minimum 8 required.")
        return NormalityTestResult(
            shapiro_wilk_pvalue=0.0,
            dagostino_pvalue=0.0,
            anderson_statistic=np.inf,
            anderson_critical_5pct=0.0,
            shapiro_wilk_normal=False,
            dagostino_normal=False,
            anderson_normal=False,
            is_normal=False,
            normal_tests_passed=0,
        )
    
    # 1. Shapiro-Wilk test (best for n < 50)
    try:
        shapiro_stat, shapiro_pval = stats.shapiro(scores)
        shapiro_normal = shapiro_pval >= alpha
    except Exception:
        shapiro_pval = 0.0
        shapiro_normal = False
    
    # 2. D'Agostino-Pearson test (requires n >= 20 for reliability)
    try:
        if n >= 20:
            dagostino_stat, dagostino_pval = stats.normaltest(scores)
            dagostino_normal = dagostino_pval >= alpha
        else:
            # For small samples, use skewness/kurtosis individually
            skewness = stats.skew(scores)
            kurtosis = stats.kurtosis(scores)
            # Simple criterion: |skewness| < 1 and |kurtosis| < 3
            dagostino_pval = 0.5 if (abs(skewness) < 1 and abs(kurtosis) < 3) else 0.01
            dagostino_normal = dagostino_pval >= alpha
    except Exception:
        dagostino_pval = 0.0
        dagostino_normal = False
    
    # 3. Anderson-Darling test
    try:
        anderson_result = stats.anderson(scores, dist='norm')
        anderson_stat = anderson_result.statistic
        # Critical value at 5% significance level (index 2)
        anderson_critical = anderson_result.critical_values[2]
        anderson_normal = anderson_stat < anderson_critical
    except Exception:
        anderson_stat = np.inf
        anderson_critical = 0.0
        anderson_normal = False
    
    # Count how many tests passed (2-of-3 criterion)
    tests_passed = sum([shapiro_normal, dagostino_normal, anderson_normal])
    is_normal = tests_passed >= 2
    
    return NormalityTestResult(
        shapiro_wilk_pvalue=float(shapiro_pval),
        dagostino_pvalue=float(dagostino_pval),
        anderson_statistic=float(anderson_stat),
        anderson_critical_5pct=float(anderson_critical),
        shapiro_wilk_normal=shapiro_normal,
        dagostino_normal=dagostino_normal,
        anderson_normal=anderson_normal,
        is_normal=is_normal,
        normal_tests_passed=tests_passed,
    )


def paired_t_test(
    scores1: np.ndarray,
    scores2: np.ndarray,
    alpha: float = 0.05
) -> Tuple[float, float, bool]:
    """Perform paired t-test to compare two models.
    
    Used when both distributions are normal.
    
    Args:
        scores1: Scores from model 1
        scores2: Scores from model 2
        alpha: Significance level
        
    Returns:
        Tuple of (t_statistic, p_value, is_significant)
    """
    # Paired t-test (assumes normal distributions)
    t_stat, p_value = stats.ttest_rel(scores1, scores2)
    
    # Two-tailed test
    is_significant = p_value < alpha
    
    return float(t_stat), float(p_value), is_significant


def wilcoxon_signed_rank_test(
    scores1: np.ndarray,
    scores2: np.ndarray,
    alpha: float = 0.05
) -> Tuple[float, float, bool]:
    """Perform Wilcoxon signed-rank test for paired samples.
    
    This is the NON-PARAMETRIC alternative to the paired t-test.
    Used when distributions are not normal but samples are paired
    (i.e., same folds in cross-validation).
    
    According to the protocol: Use this test when normality is rejected.
    
    Args:
        scores1: Scores from model 1 (same fold order as scores2)
        scores2: Scores from model 2 (same fold order as scores1)
        alpha: Significance level
        
    Returns:
        Tuple of (w_statistic, p_value, is_significant)
    """
    # Check for ties and zero differences
    differences = scores1 - scores2
    non_zero_diffs = differences[differences != 0]
    
    if len(non_zero_diffs) < 10:
        # Too few non-zero differences for reliable test
        # Use exact method for small samples
        try:
            w_stat, p_value = stats.wilcoxon(scores1, scores2, alternative='two-sided', method='exact')
        except Exception:
            w_stat, p_value = stats.wilcoxon(scores1, scores2, alternative='two-sided')
    else:
        # Use asymptotic approximation for larger samples
        w_stat, p_value = stats.wilcoxon(scores1, scores2, alternative='two-sided')
    
    is_significant = p_value < alpha
    
    return float(w_stat), float(p_value), is_significant


def mann_whitney_test(
    scores1: np.ndarray,
    scores2: np.ndarray,
    alpha: float = 0.05
) -> Tuple[float, float, bool]:
    """Perform Mann-Whitney U test to compare two models.
    
    NOTE: This test is for INDEPENDENT samples. For paired samples
    (like cross-validation folds), use wilcoxon_signed_rank_test instead.
    
    This test is included for robustness analysis as mentioned in the protocol,
    treating AUROC distributions as independent samples to verify consistency.
    
    Args:
        scores1: Scores from model 1
        scores2: Scores from model 2
        alpha: Significance level
        
    Returns:
        Tuple of (u_statistic, p_value, is_significant)
    """
    # Mann-Whitney U test (non-parametric for independent samples)
    u_stat, p_value = stats.mannwhitneyu(scores1, scores2, alternative='two-sided')
    
    is_significant = p_value < alpha
    
    return float(u_stat), float(p_value), is_significant


def cohens_d(scores1: np.ndarray, scores2: np.ndarray) -> float:
    """Calculate Cohen's d effect size.
    
    Interpretation:
    - |d| < 0.2: negligible
    - 0.2 <= |d| < 0.5: small
    - 0.5 <= |d| < 0.8: medium
    - |d| >= 0.8: large
    
    Args:
        scores1: Scores from model 1
        scores2: Scores from model 2
        
    Returns:
        Cohen's d effect size
    """
    n1, n2 = len(scores1), len(scores2)
    var1, var2 = np.var(scores1, ddof=1), np.var(scores2, ddof=1)
    
    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    # Cohen's d
    d = (np.mean(scores1) - np.mean(scores2)) / pooled_std
    
    return float(d)


def interpret_effect_size(d: float) -> str:
    """Interpret Cohen's d effect size.
    
    Args:
        d: Cohen's d value
        
    Returns:
        Interpretation string
    """
    abs_d = abs(d)
    
    if abs_d < 0.2:
        return "Negligible"
    elif abs_d < 0.5:
        return "Small"
    elif abs_d < 0.8:
        return "Medium"
    else:
        return "Large"


# =============================================================================
# MULTIPLE COMPARISON CORRECTIONS (Protocol Section 4.1)
# =============================================================================

def bonferroni_correction(
    p_values: List[float],
    alpha: float = 0.05
) -> Tuple[float, List[bool]]:
    """Apply Bonferroni correction for multiple comparisons.
    
    Most conservative correction: α_adjusted = α / n_comparisons
    
    Args:
        p_values: List of p-values from pairwise comparisons
        alpha: Family-wise error rate
        
    Returns:
        Tuple of (adjusted_alpha, list of is_significant for each p-value)
    """
    n = len(p_values)
    if n == 0:
        return alpha, []
    adjusted_alpha = alpha / n
    significant = [p < adjusted_alpha for p in p_values]
    return adjusted_alpha, significant


def holm_bonferroni_correction(
    p_values: List[float],
    alpha: float = 0.05
) -> Tuple[List[float], List[bool]]:
    """Apply Holm-Bonferroni step-down correction.
    
    Less conservative than Bonferroni, maintains control of FWER.
    
    Procedure:
    1. Order p-values from smallest to largest
    2. For the i-th smallest p-value, compare with α/(n-i+1)
    3. Reject H0_i if p_i ≤ α/(n-i+1) and all smaller p-values were rejected
    
    Args:
        p_values: List of p-values from pairwise comparisons
        alpha: Family-wise error rate
        
    Returns:
        Tuple of (adjusted_p_values, list of is_significant)
    """
    n = len(p_values)
    if n == 0:
        return [], []
    sorted_indices = np.argsort(p_values)
    sorted_p = np.array(p_values)[sorted_indices]
    
    adjusted_p = np.zeros(n)
    significant = np.zeros(n, dtype=bool)
    
    # Step-down procedure
    rejected_any = True
    for i, idx in enumerate(sorted_indices):
        holm_alpha = alpha / (n - i)
        if rejected_any and sorted_p[i] <= holm_alpha:
            significant[idx] = True
            adjusted_p[idx] = sorted_p[i] * (n - i)  # Adjusted p-value
        else:
            rejected_any = False
            adjusted_p[idx] = min(1.0, sorted_p[i] * (n - i))
    
    return list(adjusted_p), list(significant)


def benjamini_hochberg_fdr(
    p_values: List[float],
    alpha: float = 0.05
) -> Tuple[List[float], List[bool]]:
    """Apply Benjamini-Hochberg False Discovery Rate correction.
    
    Controls the expected proportion of false discoveries (less conservative
    than Bonferroni/Holm, prioritizes power).
    
    Procedure:
    1. Order p-values from smallest to largest
    2. Find the largest i where p_(i) ≤ (i/n) * α
    3. Reject all H0_j for j ≤ i
    
    Args:
        p_values: List of p-values from pairwise comparisons
        alpha: False discovery rate control level
        
    Returns:
        Tuple of (adjusted_p_values, list of is_significant)
    """
    n = len(p_values)
    if n == 0:
        return [], []
    sorted_indices = np.argsort(p_values)
    sorted_p = np.array(p_values)[sorted_indices]
    
    # Calculate adjusted p-values (q-values)
    adjusted_p = np.zeros(n)
    significant = np.zeros(n, dtype=bool)
    
    # Calculate BH critical values and adjusted p-values
    for i in range(n):
        rank = i + 1
        bh_critical = (rank / n) * alpha
        # Adjusted p-value
        adjusted_p[sorted_indices[i]] = min(1.0, sorted_p[i] * n / rank)
    
    # Ensure monotonicity of adjusted p-values
    for i in range(n - 2, -1, -1):
        adjusted_p[sorted_indices[i]] = min(
            adjusted_p[sorted_indices[i]], 
            adjusted_p[sorted_indices[i + 1]]
        )
    
    # Determine significance
    for i in range(n):
        significant[i] = adjusted_p[i] < alpha
    
    return list(adjusted_p), list(significant)


# =============================================================================
# GLOBAL TESTS FOR MULTIPLE MODEL COMPARISON (Protocol Section 3.4)
# =============================================================================

def friedman_test(
    model_scores: Dict[str, List[float]]
) -> Tuple[float, float, bool]:
    """Friedman test for comparing multiple related samples.
    
    Non-parametric alternative to one-way ANOVA with repeated measures.
    Used when normality is not confirmed.
    
    Args:
        model_scores: Dictionary of {model_name: list_of_scores}
                     All models must have the same number of scores (same folds)
        
    Returns:
        Tuple of (chi2_statistic, p_value, is_significant)
    """
    # Convert to matrix format: rows = folds, columns = models
    model_names = list(model_scores.keys())
    scores_matrix = np.array([model_scores[name] for name in model_names]).T
    
    # Friedman test
    chi2_stat, p_value = stats.friedmanchisquare(*[scores_matrix[:, i] for i in range(len(model_names))])
    
    is_significant = p_value < 0.05
    
    return float(chi2_stat), float(p_value), is_significant


def repeated_measures_anova(
    model_scores: Dict[str, List[float]]
) -> Tuple[float, float, bool]:
    """One-way ANOVA for repeated measures (within-subjects).
    
    Parametric test for comparing multiple models when normality is confirmed.
    
    Note: This is a simplified version using scipy. For full repeated measures
    ANOVA with sphericity correction, consider using pingouin or statsmodels.
    
    Args:
        model_scores: Dictionary of {model_name: list_of_scores}
        
    Returns:
        Tuple of (f_statistic, p_value, is_significant)
    """
    # Simple approach: use one-way ANOVA (not strictly correct for repeated measures)
    # For proper repeated measures ANOVA, use pingouin library
    model_names = list(model_scores.keys())
    groups = [model_scores[name] for name in model_names]
    
    f_stat, p_value = stats.f_oneway(*groups)
    
    is_significant = p_value < 0.05
    
    return float(f_stat), float(p_value), is_significant


def nemenyi_post_hoc(
    model_scores: Dict[str, List[float]],
    alpha: float = 0.05
) -> Dict[Tuple[str, str], Tuple[float, bool]]:
    """Nemenyi post-hoc test after Friedman test.
    
    Tests all pairwise comparisons with control of familywise error rate.
    Critical difference based on studentized range distribution.
    
    Args:
        model_scores: Dictionary of {model_name: list_of_scores}
        alpha: Significance level
        
    Returns:
        Dictionary of {(model1, model2): (rank_difference, is_significant)}
    """
    model_names = list(model_scores.keys())
    n_models = len(model_names)
    n_folds = len(model_scores[model_names[0]])
    
    # Calculate average ranks for each model
    scores_matrix = np.array([model_scores[name] for name in model_names]).T
    ranks_matrix = np.zeros_like(scores_matrix)
    
    for i in range(n_folds):
        ranks_matrix[i, :] = stats.rankdata(-scores_matrix[i, :])  # Higher score = lower rank (better)
    
    avg_ranks = np.mean(ranks_matrix, axis=0)
    
    # Critical difference for Nemenyi test
    # CD = q_alpha * sqrt(k(k+1)/(6n)) where k=n_models, n=n_folds
    # Using approximation for q_alpha
    q_alpha = 2.569  # Approximate for alpha=0.05, k>=5
    if n_models <= 4:
        q_values = {2: 1.960, 3: 2.343, 4: 2.569}
        q_alpha = q_values.get(n_models, 2.569)
    
    cd = q_alpha * np.sqrt(n_models * (n_models + 1) / (6 * n_folds))
    
    results = {}
    for i, name1 in enumerate(model_names):
        for j, name2 in enumerate(model_names[i + 1:], i + 1):
            rank_diff = abs(avg_ranks[i] - avg_ranks[j])
            is_significant = rank_diff > cd
            results[(name1, name2)] = (float(rank_diff), is_significant)
    
    return results


def compare_multiple_models(
    model_scores: Dict[str, List[float]],
    alpha: float = 0.05
) -> MultipleComparisonResult:
    """Comprehensive comparison of multiple models following the protocol.
    
    Pipeline:
    1. Test normality of all distributions (2-of-3 criterion)
    2. If all normal: ANOVA repeated measures + paired t-test post-hoc
    3. If not all normal: Friedman test + Nemenyi/Wilcoxon post-hoc
    4. Apply multiple comparison corrections (Bonferroni, Holm, FDR)
    
    Note: Friedman test and ANOVA require at least 3 models. When comparing
    only 2 models, only pairwise comparison is performed (no global test).
    
    Args:
        model_scores: Dictionary of {model_name: list_of_scores}
        alpha: Family-wise error rate
        
    Returns:
        MultipleComparisonResult with global and pairwise tests
    """
    model_names = list(model_scores.keys())
    n_models = len(model_names)
    n_comparisons = n_models * (n_models - 1) // 2
    
    if n_models == 0:
        raise ValueError("At least one model is required for comparison")

    if n_models == 1:
        return MultipleComparisonResult(
            model_names=model_names,
            model_scores=model_scores,
            global_test_name="N/A (single model)",
            global_test_statistic=np.nan,
            global_p_value=np.nan,
            global_significant=False,
            pairwise_results={},
            bonferroni_alpha=alpha,
            holm_corrected_pvalues={},
            fdr_corrected_pvalues={},
        )
    
    # Check normality of all distributions
    all_normal = True
    for name, scores in model_scores.items():
        normality_result = test_normality_full(np.array(scores), alpha)
        if not normality_result.is_normal:
            all_normal = False
            break
    
    # Global test (requires at least 3 models)
    if n_models >= 3:
        if all_normal:
            global_stat, global_p, global_sig = repeated_measures_anova(model_scores)
            global_test_name = "ANOVA (repeated measures)"
        else:
            global_stat, global_p, global_sig = friedman_test(model_scores)
            global_test_name = "Friedman test"
    else:
        # Only 2 models: no global test possible, use pairwise comparison only
        global_stat = np.nan
        global_p = np.nan
        global_sig = False
        global_test_name = "N/A (requires ≥3 models)"
    
    # Pairwise comparisons
    pairwise_results = {}
    pairwise_pvalues = []
    pairwise_keys = []
    
    for i, name1 in enumerate(model_names):
        for name2 in model_names[i + 1:]:
            result = compare_models(name1, model_scores[name1], name2, model_scores[name2], alpha)
            pairwise_results[(name1, name2)] = result
            pairwise_pvalues.append(result.p_value)
            pairwise_keys.append((name1, name2))
    
    # Apply multiple comparison corrections
    bonferroni_alpha, bonferroni_sig = bonferroni_correction(pairwise_pvalues, alpha)
    holm_adjusted, holm_sig = holm_bonferroni_correction(pairwise_pvalues, alpha)
    fdr_adjusted, fdr_sig = benjamini_hochberg_fdr(pairwise_pvalues, alpha)
    
    holm_corrected = {key: p for key, p in zip(pairwise_keys, holm_adjusted)}
    fdr_corrected = {key: p for key, p in zip(pairwise_keys, fdr_adjusted)}
    
    return MultipleComparisonResult(
        model_names=model_names,
        model_scores=model_scores,
        global_test_name=global_test_name,
        global_test_statistic=global_stat,
        global_p_value=global_p,
        global_significant=global_sig,
        pairwise_results=pairwise_results,
        bonferroni_alpha=bonferroni_alpha,
        holm_corrected_pvalues=holm_corrected,
        fdr_corrected_pvalues=fdr_corrected,
    )


def compare_models(
    model1_name: str,
    model1_scores: List[float],
    model2_name: str,
    model2_scores: List[float],
    alpha: float = 0.05,
    use_full_normality_test: bool = True
) -> StatisticalTestResult:
    """Perform complete statistical comparison between two models.
    
    Pipeline following the methodological protocol:
    1. Test normality using 2-of-3 criterion (Shapiro-Wilk, D'Agostino, Anderson-Darling)
    2. If both normal: use paired t-test (parametric)
    3. If not normal: use Wilcoxon signed-rank test (non-parametric for paired samples)
    4. Calculate effect size (Cohen's d) and ΔAUROC
    
    Note: Wilcoxon signed-rank is used instead of Mann-Whitney U because
    the scores are PAIRED (same folds in cross-validation).
    
    Args:
        model1_name: Name of first model
        model1_scores: Scores from first model
        model2_name: Name of second model
        model2_scores: Scores from second model
        alpha: Significance level for tests
        use_full_normality_test: If True, use 2-of-3 criterion; if False, only Shapiro-Wilk
        
    Returns:
        StatisticalTestResult with complete comparison
    """
    scores1 = np.array(model1_scores)
    scores2 = np.array(model2_scores)
    
    # Descriptive statistics
    mean1, std1 = float(np.mean(scores1)), float(np.std(scores1, ddof=1))
    mean2, std2 = float(np.mean(scores2)), float(np.std(scores2, ddof=1))
    delta_auroc = mean1 - mean2
    
    # Test normality
    if use_full_normality_test and len(scores1) >= 20:
        # Use full 2-of-3 criterion for sufficient sample sizes
        normality1 = test_normality_full(scores1, alpha)
        normality2 = test_normality_full(scores2, alpha)
        is_normal1 = normality1.is_normal
        is_normal2 = normality2.is_normal
        pval1 = normality1.shapiro_wilk_pvalue
        pval2 = normality2.shapiro_wilk_pvalue
    else:
        # Fall back to simple Shapiro-Wilk for small samples
        is_normal1, pval1 = test_normality(scores1, alpha)
        is_normal2, pval2 = test_normality(scores2, alpha)
        normality1 = None
        normality2 = None
    
    # Choose appropriate test based on normality
    if is_normal1 and is_normal2:
        # Both normal: use parametric test (paired t-test)
        test_stat, p_value, significant = paired_t_test(scores1, scores2, alpha)
        test_used = "Paired t-test"
    else:
        # At least one non-normal: use Wilcoxon signed-rank (paired non-parametric)
        test_stat, p_value, significant = wilcoxon_signed_rank_test(scores1, scores2, alpha)
        test_used = "Wilcoxon signed-rank test"
    
    # Calculate effect size
    effect = cohens_d(scores1, scores2)
    effect_interp = interpret_effect_size(effect)
    
    return StatisticalTestResult(
        model1_name=model1_name,
        model2_name=model2_name,
        model1_scores=model1_scores,
        model2_scores=model2_scores,
        model1_mean=mean1,
        model1_std=std1,
        model2_mean=mean2,
        model2_std=std2,
        model1_is_normal=is_normal1,
        model2_is_normal=is_normal2,
        model1_shapiro_pvalue=pval1,
        model2_shapiro_pvalue=pval2,
        model1_normality_detail=normality1 if use_full_normality_test else None,
        model2_normality_detail=normality2 if use_full_normality_test else None,
        test_used=test_used,
        test_statistic=test_stat,
        p_value=p_value,
        significant=significant,
        alpha=alpha,
        effect_size=effect,
        effect_size_interpretation=effect_interp,
        delta_auroc=delta_auroc,
    )


def compare_all_models(
    model_scores: Dict[str, List[float]],
    alpha: float = 0.05
) -> Dict[Tuple[str, str], StatisticalTestResult]:
    """Compare all pairs of models statistically.
    
    Args:
        model_scores: Dictionary mapping model_name -> list of scores
        alpha: Significance level
        
    Returns:
        Dictionary mapping (model1, model2) -> StatisticalTestResult
    """
    model_names = list(model_scores.keys())
    results = {}
    
    for i, name1 in enumerate(model_names):
        for name2 in model_names[i + 1:]:
            result = compare_models(
                model1_name=name1,
                model1_scores=model_scores[name1],
                model2_name=name2,
                model2_scores=model_scores[name2],
                alpha=alpha
            )
            results[(name1, name2)] = result
    
    return results


def plot_model_comparison(
    result: StatisticalTestResult,
    save_path: Optional[str] = None
) -> plt.Figure:
    """Create comprehensive visualization of model comparison.
    
    Args:
        result: Statistical test result
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Model Comparison: {result.model1_name} vs {result.model2_name}', 
                 fontsize=16, fontweight='bold')
    
    # 1. Box plot comparison
    ax1 = axes[0, 0]
    data_to_plot = [result.model1_scores, result.model2_scores]
    bp = ax1.boxplot(data_to_plot, labels=[result.model1_name, result.model2_name],
                     patch_artist=True)
    for patch, color in zip(bp['boxes'], ['lightblue', 'lightgreen']):
        patch.set_facecolor(color)
    ax1.set_ylabel('Score (AUROC)')
    ax1.set_title('Score Distribution')
    ax1.grid(True, alpha=0.3)
    
    # 2. Violin plot
    ax2 = axes[0, 1]
    positions = [1, 2]
    parts1 = ax2.violinplot([result.model1_scores], positions=[1], 
                            showmeans=True, showmedians=True)
    parts2 = ax2.violinplot([result.model2_scores], positions=[2], 
                            showmeans=True, showmedians=True)
    for pc in parts1['bodies']:
        pc.set_facecolor('lightblue')
        pc.set_alpha(0.7)
    for pc in parts2['bodies']:
        pc.set_facecolor('lightgreen')
        pc.set_alpha(0.7)
    ax2.set_xticks([1, 2])
    ax2.set_xticklabels([result.model1_name, result.model2_name])
    ax2.set_ylabel('Score (AUROC)')
    ax2.set_title('Score Distribution (Violin Plot)')
    ax2.grid(True, alpha=0.3)
    
    # 3. Histogram comparison with adaptive bins
    ax3 = axes[1, 0]
    
    # Calculate optimal number of bins based on data range
    all_scores = list(result.model1_scores) + list(result.model2_scores)
    data_range = max(all_scores) - min(all_scores)
    
    # Adaptive bins: use fewer bins if data range is small
    if data_range < 0.01:
        n_bins = 5  # Very small range
    elif data_range < 0.05:
        n_bins = 8
    elif data_range < 0.1:
        n_bins = 10
    else:
        n_bins = 15  # Normal range
    
    # Use Sturges' rule as backup: k = ceil(log2(n) + 1)
    sturges_bins = int(np.ceil(np.log2(len(all_scores)) + 1))
    n_bins = min(n_bins, sturges_bins, len(set(all_scores)))  # Don't exceed unique values
    
    try:
        ax3.hist(result.model1_scores, bins=n_bins, alpha=0.5, label=result.model1_name, 
                color='blue', edgecolor='black')
        ax3.hist(result.model2_scores, bins=n_bins, alpha=0.5, label=result.model2_name, 
                color='green', edgecolor='black')
    except ValueError:
        # If histogram still fails, fall back to KDE plot
        from scipy import stats
        kde1 = stats.gaussian_kde(result.model1_scores)
        kde2 = stats.gaussian_kde(result.model2_scores)
        x_range = np.linspace(min(all_scores), max(all_scores), 100)
        ax3.plot(x_range, kde1(x_range), color='blue', label=result.model1_name, linewidth=2)
        ax3.fill_between(x_range, kde1(x_range), alpha=0.3, color='blue')
        ax3.plot(x_range, kde2(x_range), color='green', label=result.model2_name, linewidth=2)
        ax3.fill_between(x_range, kde2(x_range), alpha=0.3, color='green')
        ax3.set_ylabel('Density')
    
    ax3.axvline(result.model1_mean, color='blue', linestyle='--', linewidth=2, 
                label=f'{result.model1_name} mean')
    ax3.axvline(result.model2_mean, color='green', linestyle='--', linewidth=2, 
                label=f'{result.model2_name} mean')
    ax3.set_xlabel('Score (AUROC)')
    ax3.set_ylabel('Frequency')
    ax3.set_title(f'Score Histograms ({n_bins} bins)' if 'kde1' not in locals() else 'Score Distributions (KDE)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Statistical test results (text)
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    text_content = [
        "Statistical Test Results:",
        "",
        f"{result.model1_name}:",
        f"  μ = {result.model1_mean:.4f}, σ = {result.model1_std:.4f}",
        f"  Normal: {result.model1_is_normal} (p={result.model1_shapiro_pvalue:.4f})",
        "",
        f"{result.model2_name}:",
        f"  μ = {result.model2_mean:.4f}, σ = {result.model2_std:.4f}",
        f"  Normal: {result.model2_is_normal} (p={result.model2_shapiro_pvalue:.4f})",
        "",
        f"Test: {result.test_used}",
        f"  Statistic: {result.test_statistic:.4f}",
        f"  P-value: {result.p_value:.4f}",
        f"  Significant (α={result.alpha}): {result.significant}",
        "",
        f"Effect Size (Cohen's d): {result.effect_size:.4f}",
        f"  Interpretation: {result.effect_size_interpretation}",
        "",
        "Conclusion:",
        f"  {'Significant' if result.significant else 'No significant'} difference",
        f"  Winner: {result.model1_name if result.model1_mean > result.model2_mean else result.model2_name}",
    ]
    
    ax4.text(0.1, 0.9, '\n'.join(text_content), transform=ax4.transAxes,
            fontsize=10, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def create_comparison_matrix(
    results: Dict[Tuple[str, str], StatisticalTestResult]
) -> plt.Figure:
    """Create a matrix showing pairwise model comparisons.
    
    Args:
        results: Dictionary of pairwise comparison results
        
    Returns:
        Matplotlib figure
    """
    # Extract all unique model names
    all_names = set()
    for (name1, name2) in results.keys():
        all_names.add(name1)
        all_names.add(name2)
    model_names = sorted(list(all_names))
    n_models = len(model_names)
    
    # Create matrices for p-values and effect sizes
    p_value_matrix = np.ones((n_models, n_models))
    effect_size_matrix = np.zeros((n_models, n_models))
    
    name_to_idx = {name: i for i, name in enumerate(model_names)}
    
    for (name1, name2), result in results.items():
        i, j = name_to_idx[name1], name_to_idx[name2]
        p_value_matrix[i, j] = result.p_value
        p_value_matrix[j, i] = result.p_value
        effect_size_matrix[i, j] = result.effect_size
        effect_size_matrix[j, i] = -result.effect_size
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # P-value heatmap
    im1 = ax1.imshow(p_value_matrix, cmap='RdYlGn', vmin=0, vmax=0.1)
    ax1.set_xticks(range(n_models))
    ax1.set_yticks(range(n_models))
    ax1.set_xticklabels(model_names, rotation=45, ha='right')
    ax1.set_yticklabels(model_names)
    ax1.set_title('P-values Matrix\n(Green = Significant difference)')
    
    # Add text annotations
    for i in range(n_models):
        for j in range(n_models):
            if i != j:
                text = ax1.text(j, i, f'{p_value_matrix[i, j]:.3f}',
                               ha="center", va="center", color="black", fontsize=8)
    
    plt.colorbar(im1, ax=ax1)
    
    # Effect size heatmap
    im2 = ax2.imshow(effect_size_matrix, cmap='RdBu_r', vmin=-2, vmax=2)
    ax2.set_xticks(range(n_models))
    ax2.set_yticks(range(n_models))
    ax2.set_xticklabels(model_names, rotation=45, ha='right')
    ax2.set_yticklabels(model_names)
    ax2.set_title('Effect Size Matrix (Cohen\'s d)\n(Red = Model 1 better, Blue = Model 2 better)')
    
    # Add text annotations
    for i in range(n_models):
        for j in range(n_models):
            if i != j:
                text = ax2.text(j, i, f'{effect_size_matrix[i, j]:.2f}',
                               ha="center", va="center", color="black", fontsize=8)
    
    plt.colorbar(im2, ax=ax2)
    
    plt.tight_layout()
    
    return fig
