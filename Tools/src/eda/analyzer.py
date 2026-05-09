"""Main EDA analyzer class."""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .univariate import UnivariateStats, compute_numeric_stats, compute_categorical_stats
from .bivariate import (
    BivariateStats,
    analyze_numeric_numeric,
    analyze_numeric_categorical,
    analyze_categorical_categorical,
)
from .multivariate import PCAResults, perform_pca, get_feature_importance_pca
from .visualizations import (
    plot_distribution,
    plot_correlation_matrix,
    plot_scatter,
    plot_pairwise_scatter,
    plot_pca_scree,
    plot_pca_biplot,
)


class EDAAnalyzer:
    """Complete exploratory data analysis analyzer."""
    
    def __init__(self, df: pd.DataFrame):
        """Initialize analyzer with a DataFrame.
        
        Args:
            df: DataFrame to analyze
        """
        self.df = df.copy()
        self.numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
        
        self.univariate_results: Dict[str, UnivariateStats] = {}
        self.bivariate_results: Dict[str, BivariateStats] = {}
        self.pca_results: Optional[PCAResults] = None
    
    # ==================== UNIVARIATE ANALYSIS ====================
    
    def analyze_univariate(self, columns: Optional[List[str]] = None) -> Dict[str, UnivariateStats]:
        """Perform complete univariate analysis.
        
        Args:
            columns: List of columns to analyze. If None, analyzes all.
            
        Returns:
            Dictionary with statistics by column
        """
        if columns is None:
            columns = self.df.columns.tolist()
        
        for col in columns:
            if col not in self.df.columns:
                continue
            
            is_numeric = col in self.numeric_cols
            
            if is_numeric:
                stats_obj = compute_numeric_stats(self.df, col)
            else:
                stats_obj = compute_categorical_stats(self.df, col)
            
            self.univariate_results[col] = stats_obj
        
        return self.univariate_results
    
    def plot_distribution(self, col: str, plot_type: str = 'auto') -> go.Figure:
        """Generate distribution plot for a variable.
        
        Args:
            col: Column name
            plot_type: 'histogram', 'box', 'violin', 'bar', 'pie', 'auto'
            
        Returns:
            Plotly Figure
        """
        is_numeric = col in self.numeric_cols
        return plot_distribution(self.df, col, is_numeric, plot_type)
    
    # ==================== BIVARIATE ANALYSIS ====================
    
    def analyze_bivariate(self, var1: str, var2: str) -> BivariateStats:
        """Analyze relationship between two variables.
        
        Args:
            var1: First variable
            var2: Second variable
            
        Returns:
            BivariateStats object
        """
        is_var1_numeric = var1 in self.numeric_cols
        is_var2_numeric = var2 in self.numeric_cols
        
        if is_var1_numeric and is_var2_numeric:
            result = analyze_numeric_numeric(self.df, var1, var2)
        elif is_var1_numeric and not is_var2_numeric:
            result = analyze_numeric_categorical(self.df, var1, var2)
        elif not is_var1_numeric and is_var2_numeric:
            result = analyze_numeric_categorical(self.df, var2, var1)
        else:
            result = analyze_categorical_categorical(self.df, var1, var2)
        
        # Guardar con clave para fácil acceso
        key = f"{var1}_vs_{var2}"
        self.bivariate_results[key] = result
        return result
    
    def plot_correlation_matrix(self, method: str = 'pearson') -> go.Figure:
        """Generate correlation matrix heatmap.
        
        Args:
            method: 'pearson' or 'spearman'
            
        Returns:
            Plotly Figure
        """
        return plot_correlation_matrix(self.df, self.numeric_cols, method)
    
    def plot_scatter(
        self,
        var1: str,
        var2: str,
        color_by: Optional[str] = None,
        add_trendline: bool = True
    ) -> go.Figure:
        """Generate scatter plot between two variables.
        
        Args:
            var1: X variable
            var2: Y variable
            color_by: Variable for coloring points
            add_trendline: Whether to add trendline
            
        Returns:
            Plotly Figure
        """
        return plot_scatter(self.df, var1, var2, color_by, add_trendline)
    
    def plot_pairwise_scatter(
        self,
        variables: Optional[List[str]] = None,
        max_vars: int = 10
    ) -> go.Figure:
        """Generate pairwise scatter plot matrix.
        
        Args:
            variables: List of variables to include
            max_vars: Maximum number of variables
            
        Returns:
            Plotly Figure
        """
        if variables is None:
            variables = self.numeric_cols[:max_vars]
        else:
            variables = [v for v in variables if v in self.numeric_cols][:max_vars]
        
        return plot_pairwise_scatter(self.df, variables, max_vars)
    
    # ==================== MULTIVARIATE ANALYSIS ====================
    
    def analyze_multivariate(self, method: str = 'pearson') -> pd.DataFrame:
        """Calculate correlation matrix for multivariate analysis.
        
        Args:
            method: Correlation method ('pearson', 'spearman', or 'kendall')
            
        Returns:
            Correlation matrix DataFrame
        """
        if len(self.numeric_cols) < 2:
            raise ValueError("Need at least 2 numeric columns for correlation analysis")
        
        numeric_df = self.df[self.numeric_cols]
        corr_matrix = numeric_df.corr(method=method)
        return corr_matrix
    
    def perform_pca(
        self,
        n_components: Optional[int] = None,
        variance_threshold: float = 0.95,
        scale: bool = True,
    ) -> PCAResults:
        """Perform Principal Component Analysis (PCA).
        
        Args:
            n_components: Number of components. If None, uses variance_threshold
            variance_threshold: Desired cumulative variance (0-1)
            scale: Whether to standardize data before PCA
            
        Returns:
            PCAResults object
        """
        self.pca_results = perform_pca(
            self.df,
            self.numeric_cols,
            n_components,
            variance_threshold,
            scale
        )
        return self.pca_results
    
    def plot_pca_scree(self) -> go.Figure:
        """Generate scree plot (explained variance by component).
        
        Returns:
            Plotly Figure
        """
        if self.pca_results is None:
            raise ValueError("Must run perform_pca() first")
        
        return plot_pca_scree(self.pca_results)
    
    def plot_pca_biplot(
        self,
        pc_x: int = 1,
        pc_y: int = 2,
        n_features: int = 10
    ) -> go.Figure:
        """Generate PCA biplot.
        
        Args:
            pc_x: Principal component for X axis (1-indexed)
            pc_y: Principal component for Y axis (1-indexed)
            n_features: Number of most important features to show
            
        Returns:
            Plotly Figure
        """
        if self.pca_results is None:
            raise ValueError("Must run perform_pca() first")
        
        return plot_pca_biplot(self.pca_results, pc_x, pc_y, n_features)
    
    def get_feature_importance_pca(self, n_components: int = 3) -> pd.DataFrame:
        """Get feature importance in first N components.
        
        Args:
            n_components: Number of components to consider
            
        Returns:
            DataFrame with feature importance
        """
        if self.pca_results is None:
            raise ValueError("Must run perform_pca() first")
        
        return get_feature_importance_pca(self.pca_results, n_components)
    
    # ==================== PERSISTENCE ====================
    
    def save_results(self, filepath: Path) -> None:
        """Save analysis results to pickle file.
        
        Args:
            filepath: Path to save results
        """
        results = {
            'univariate': self.univariate_results,
            'bivariate': self.bivariate_results,
            'pca': self.pca_results
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(results, f)
    
    def load_results(self, filepath: Path) -> None:
        """Load results from pickle file.
        
        Args:
            filepath: Path to pickle file
        """
        with open(filepath, 'rb') as f:
            results = pickle.load(f)
        
        self.univariate_results = results.get('univariate', {})
        self.bivariate_results = results.get('bivariate', {})
        self.pca_results = results.get('pca', None)
    
    def generate_summary_report(self) -> Dict[str, Any]:
        """Generate summary report of EDA.
        
        Returns:
            Dictionary with complete summary
        """
        report = {
            'dataset_shape': self.df.shape,
            'n_numeric': len(self.numeric_cols),
            'n_categorical': len(self.categorical_cols),
            'total_missing': int(self.df.isna().sum().sum()),
            'missing_percent': float(self.df.isna().mean().mean() * 100),
        }
        
        if self.univariate_results:
            report['univariate_analyzed'] = len(self.univariate_results)
            report['variables_with_outliers'] = sum(
                1 for v in self.univariate_results.values()
                if v.variable_type == 'numerical' and v.skewness is not None and abs(v.skewness) > 1
            )
        
        if self.bivariate_results:
            report['bivariate_analyzed'] = len(self.bivariate_results)
            report['significant_relationships'] = sum(1 for b in self.bivariate_results.values() if b.is_significant)
        
        if self.pca_results:
            report['pca_components'] = self.pca_results.n_components
            report['pca_variance_explained'] = sum(self.pca_results.explained_variance_ratio)
        
        return report


def quick_eda(
    df: pd.DataFrame, 
    run_pca: bool = False,
    n_components: Optional[int] = None
) -> EDAAnalyzer:
    """Convenience function for quick EDA.
    
    Args:
        df: DataFrame to analyze
        run_pca: Whether to run PCA
        n_components: Number of components for PCA
        
    Returns:
        EDAAnalyzer with results
    """
    analyzer = EDAAnalyzer(df)
    analyzer.analyze_univariate()
    
    if run_pca and len(analyzer.numeric_cols) >= 2:
        try:
            analyzer.perform_pca(n_components=n_components)
        except Exception as e:
            print(f"PCA failed: {e}")
    
    return analyzer
