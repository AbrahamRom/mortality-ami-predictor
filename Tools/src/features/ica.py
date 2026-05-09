"""Independent Component Analysis (ICA) for feature transformation.

ICA is a computational method for separating a multivariate signal into 
additive subcomponents, assuming the mutual statistical independence of the 
non-Gaussian source signals. It's particularly useful for:

- Blind source separation
- Feature extraction when features are mixed
- Non-Gaussian data
- When independence is more important than orthogonality (vs PCA)

Key Differences from PCA:
- PCA: Finds orthogonal components maximizing variance (Gaussian assumption)
- ICA: Finds independent components maximizing non-Gaussianity
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from sklearn.decomposition import FastICA
from sklearn.preprocessing import StandardScaler


@dataclass
class ICAResult:
    """Results from ICA transformation."""
    
    n_components: int
    n_features_original: int
    feature_names: List[str]
    component_names: List[str]
    
    # ICA specific
    mixing_matrix: np.ndarray  # A matrix: X = A @ S
    unmixing_matrix: np.ndarray  # W matrix: S = W @ X
    mean_removed: np.ndarray
    
    # Statistical properties
    component_kurtosis: np.ndarray  # Measure of non-Gaussianity
    component_means: np.ndarray
    component_stds: np.ndarray
    
    # Variance explained (not primary goal of ICA, but useful)
    variance_per_component: np.ndarray
    cumulative_variance: np.ndarray
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'n_components': self.n_components,
            'n_features_original': self.n_features_original,
            'feature_names': self.feature_names,
            'component_names': self.component_names,
            'mixing_matrix': self.mixing_matrix.tolist(),
            'unmixing_matrix': self.unmixing_matrix.tolist(),
            'mean_removed': self.mean_removed.tolist(),
            'component_kurtosis': self.component_kurtosis.tolist(),
            'component_means': self.component_means.tolist(),
            'component_stds': self.component_stds.tolist(),
            'variance_per_component': self.variance_per_component.tolist(),
            'cumulative_variance': self.cumulative_variance.tolist(),
        }


class ICATransformer:
    """Independent Component Analysis transformer with visualization.
    
    This class provides a complete ICA pipeline including:
    - Standardization (ICA requires centered and scaled data)
    - ICA transformation
    - Component analysis and visualization
    - Inverse transformation
    - Serialization support
    
    Example:
        >>> ica = ICATransformer(n_components=10)
        >>> ica.fit(X_train)
        >>> X_transformed = ica.transform(X_train)
        >>> X_reconstructed = ica.inverse_transform(X_transformed)
    """
    
    def __init__(
        self,
        n_components: Optional[int] = None,
        algorithm: str = 'parallel',
        whiten: str = 'unit-variance',
        max_iter: int = 200,
        tol: float = 1e-4,
        random_state: int = 42,
        fun: str = 'logcosh',
    ):
        """Initialize ICA transformer.
        
        Args:
            n_components: Number of components to extract. 
                         If None, same as number of features.
            algorithm: Algorithm to use: 'parallel' or 'deflation'
            whiten: Whitening strategy: 'unit-variance' or 'arbitrary-variance'
            max_iter: Maximum number of iterations
            tol: Tolerance for convergence
            random_state: Random seed for reproducibility
            fun: Functional form of G function (non-Gaussianity measure):
                 'logcosh', 'exp', or 'cube'
        """
        self.n_components = n_components
        self.algorithm = algorithm
        self.whiten = whiten
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.fun = fun
        
        # Components to be initialized during fit
        self.scaler_ = None
        self.ica_ = None
        self.result_ = None
        self.feature_names_ = None
        self.is_fitted_ = False
    
    def fit(self, X: pd.DataFrame, y=None) -> 'ICATransformer':
        """Fit ICA to data.
        
        Args:
            X: Features DataFrame
            y: Target (ignored, for sklearn compatibility)
            
        Returns:
            self
        """
        # Store feature names
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = X.columns.tolist()
            X_array = X.values
        else:
            X_array = X
            self.feature_names_ = [f'Feature_{i}' for i in range(X_array.shape[1])]
        
        # Determine n_components
        n_samples, n_features = X_array.shape
        if self.n_components is None:
            self.n_components = min(n_samples, n_features)
        
        # Step 1: Standardize data (ICA requires centered and scaled data)
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X_array)
        
        # Step 2: Apply ICA
        self.ica_ = FastICA(
            n_components=self.n_components,
            algorithm=self.algorithm,
            whiten=self.whiten,
            max_iter=self.max_iter,
            tol=self.tol,
            random_state=self.random_state,
            fun=self.fun
        )
        
        S = self.ica_.fit_transform(X_scaled)
        
        # Step 3: Compute statistics
        component_kurtosis = stats.kurtosis(S, axis=0)
        component_means = np.mean(S, axis=0)
        component_stds = np.std(S, axis=0)
        
        # Variance explained (reconstruction perspective)
        variance_per_component = np.var(S, axis=0)
        total_variance = np.sum(variance_per_component)
        variance_per_component_normalized = variance_per_component / total_variance
        cumulative_variance = np.cumsum(variance_per_component_normalized)
        
        # Component names
        component_names = [f'IC{i+1}' for i in range(self.n_components)]
        
        # Store results
        self.result_ = ICAResult(
            n_components=self.n_components,
            n_features_original=n_features,
            feature_names=self.feature_names_,
            component_names=component_names,
            mixing_matrix=self.ica_.mixing_,
            unmixing_matrix=self.ica_.components_,
            mean_removed=self.scaler_.mean_,
            component_kurtosis=component_kurtosis,
            component_means=component_means,
            component_stds=component_stds,
            variance_per_component=variance_per_component_normalized,
            cumulative_variance=cumulative_variance,
        )
        
        self.is_fitted_ = True
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data to independent components.
        
        Args:
            X: Features DataFrame
            
        Returns:
            Transformed data (independent components)
        """
        if not self.is_fitted_:
            raise ValueError("ICATransformer must be fitted before transform")
        
        # Get array
        if isinstance(X, pd.DataFrame):
            X_array = X.values
            index = X.index
        else:
            X_array = X
            index = None
        
        # Transform
        X_scaled = self.scaler_.transform(X_array)
        S = self.ica_.transform(X_scaled)
        
        # Return as DataFrame
        return pd.DataFrame(
            S,
            columns=self.result_.component_names,
            index=index
        )
    
    def inverse_transform(self, S: pd.DataFrame) -> pd.DataFrame:
        """Reconstruct original data from independent components.
        
        Args:
            S: Independent components DataFrame
            
        Returns:
            Reconstructed data in original feature space
        """
        if not self.is_fitted_:
            raise ValueError("ICATransformer must be fitted before inverse_transform")
        
        # Get array
        if isinstance(S, pd.DataFrame):
            S_array = S.values
            index = S.index
        else:
            S_array = S
            index = None
        
        # Inverse transform
        X_scaled = self.ica_.inverse_transform(S_array)
        X_reconstructed = self.scaler_.inverse_transform(X_scaled)
        
        # Return as DataFrame
        return pd.DataFrame(
            X_reconstructed,
            columns=self.feature_names_,
            index=index
        )
    
    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """Fit and transform in one step.
        
        Args:
            X: Features DataFrame
            y: Target (ignored)
            
        Returns:
            Transformed data
        """
        return self.fit(X, y).transform(X)
    
    def get_feature_importance(self, component_idx: int = 0) -> pd.DataFrame:
        """Get feature importance for a specific independent component.
        
        Args:
            component_idx: Index of the component (0-indexed)
            
        Returns:
            DataFrame with features and their mixing weights
        """
        if not self.is_fitted_:
            raise ValueError("ICATransformer must be fitted first")
        
        # Mixing matrix: X = A @ S
        # Each column of A shows how much each IC contributes to original features
        # Each row shows the mixing coefficients for one IC
        weights = self.result_.mixing_matrix[:, component_idx]
        
        importance_df = pd.DataFrame({
            'Feature': self.feature_names_,
            'Mixing_Weight': weights,
            'Abs_Weight': np.abs(weights)
        }).sort_values('Abs_Weight', ascending=False)
        
        return importance_df
    
    def plot_kurtosis(self) -> go.Figure:
        """Plot kurtosis of independent components.
        
        Kurtosis measures non-Gaussianity:
        - Kurtosis = 0: Gaussian
        - Kurtosis > 0: Heavy-tailed (super-Gaussian)
        - Kurtosis < 0: Light-tailed (sub-Gaussian)
        
        Returns:
            Plotly Figure
        """
        if not self.is_fitted_:
            raise ValueError("ICATransformer must be fitted first")
        
        fig = go.Figure()
        
        # Bar chart of kurtosis
        colors = ['red' if k < 0 else 'green' if k > 0 else 'gray' 
                  for k in self.result_.component_kurtosis]
        
        fig.add_trace(go.Bar(
            x=self.result_.component_names,
            y=self.result_.component_kurtosis,
            marker_color=colors,
            text=[f'{k:.2f}' for k in self.result_.component_kurtosis],
            textposition='auto',
            hovertemplate='%{x}<br>Kurtosis: %{y:.3f}<extra></extra>'
        ))
        
        # Reference line at 0 (Gaussian)
        fig.add_hline(y=0, line_dash="dash", line_color="black", 
                     annotation_text="Gaussian Reference")
        
        fig.update_layout(
            title='Kurtosis of Independent Components (Non-Gaussianity Measure)',
            xaxis_title='Independent Component',
            yaxis_title='Excess Kurtosis',
            template='plotly_white',
            height=500,
            showlegend=False
        )
        
        return fig
    
    def plot_mixing_matrix(self, top_n: int = 20) -> go.Figure:
        """Plot mixing matrix heatmap.
        
        The mixing matrix A shows how independent components mix to form
        the original features: X = A @ S
        
        Args:
            top_n: Number of top features to display
            
        Returns:
            Plotly Figure
        """
        if not self.is_fitted_:
            raise ValueError("ICATransformer must be fitted first")
        
        # Select top_n features with highest variance in mixing
        feature_variance = np.var(self.result_.mixing_matrix, axis=1)
        top_indices = np.argsort(feature_variance)[-top_n:]
        
        mixing_subset = self.result_.mixing_matrix[top_indices, :]
        feature_names_subset = [self.feature_names_[i] for i in top_indices]
        
        fig = go.Figure(data=go.Heatmap(
            z=mixing_subset,
            x=self.result_.component_names,
            y=feature_names_subset,
            colorscale='RdBu',
            zmid=0,
            hovertemplate='IC: %{x}<br>Feature: %{y}<br>Weight: %{z:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=f'Mixing Matrix (Top {top_n} Most Variable Features)',
            xaxis_title='Independent Component',
            yaxis_title='Original Feature',
            template='plotly_white',
            height=max(400, top_n * 20),
            width=max(600, self.n_components * 40)
        )
        
        return fig
    
    def plot_components_distribution(self, n_components_to_plot: int = 6) -> go.Figure:
        """Plot distribution of independent components.
        
        Args:
            n_components_to_plot: Number of components to plot
            
        Returns:
            Plotly Figure with subplots
        """
        if not self.is_fitted_:
            raise ValueError("ICATransformer must be fitted first")
        
        n_plot = min(n_components_to_plot, self.n_components)
        n_cols = 3
        n_rows = (n_plot + n_cols - 1) // n_cols
        
        fig = make_subplots(
            rows=n_rows, cols=n_cols,
            subplot_titles=[f'{name}<br>κ={k:.2f}' 
                          for name, k in zip(
                              self.result_.component_names[:n_plot],
                              self.result_.component_kurtosis[:n_plot]
                          )]
        )
        
        for idx in range(n_plot):
            row = idx // n_cols + 1
            col = idx % n_cols + 1
            
            # This would need actual component data to plot
            # For now, show a placeholder
            fig.add_annotation(
                text="Fit and transform data<br>to see distributions",
                xref=f"x{idx+1}" if idx > 0 else "x",
                yref=f"y{idx+1}" if idx > 0 else "y",
                x=0.5, y=0.5,
                showarrow=False,
                row=row, col=col
            )
        
        fig.update_layout(
            title='Distribution of Independent Components',
            template='plotly_white',
            height=n_rows * 300,
            showlegend=False
        )
        
        return fig
    
    def plot_variance_explained(self) -> go.Figure:
        """Plot variance explained by components.
        
        Note: Variance is not the primary optimization goal of ICA,
        but it's useful for understanding component importance.
        
        Returns:
            Plotly Figure
        """
        if not self.is_fitted_:
            raise ValueError("ICATransformer must be fitted first")
        
        fig = go.Figure()
        
        # Individual variance
        fig.add_trace(go.Bar(
            x=self.result_.component_names,
            y=self.result_.variance_per_component * 100,
            name='Individual Variance',
            marker_color='steelblue',
            hovertemplate='%{x}<br>Variance: %{y:.2f}%<extra></extra>'
        ))
        
        # Cumulative variance line
        fig.add_trace(go.Scatter(
            x=self.result_.component_names,
            y=self.result_.cumulative_variance * 100,
            name='Cumulative Variance',
            mode='lines+markers',
            line=dict(color='red', width=3),
            marker=dict(size=8),
            yaxis='y2',
            hovertemplate='%{x}<br>Cumulative: %{y:.2f}%<extra></extra>'
        ))
        
        fig.update_layout(
            title='Variance Explained by Independent Components',
            xaxis_title='Independent Component',
            yaxis=dict(title='Individual Variance (%)', side='left'),
            yaxis2=dict(title='Cumulative Variance (%)', side='right', overlaying='y', range=[0, 105]),
            template='plotly_white',
            height=500,
            hovermode='x unified'
        )
        
        return fig
    
    def save(self, filepath: str):
        """Save ICA transformer to disk.
        
        Args:
            filepath: Path to save file (.pkl or .joblib)
        """
        if not self.is_fitted_:
            raise ValueError("Cannot save unfitted ICATransformer")
        
        save_data = {
            'scaler': self.scaler_,
            'ica': self.ica_,
            'result': self.result_,
            'feature_names': self.feature_names_,
            'n_components': self.n_components,
            'algorithm': self.algorithm,
            'whiten': self.whiten,
            'max_iter': self.max_iter,
            'tol': self.tol,
            'random_state': self.random_state,
            'fun': self.fun,
        }
        
        joblib.dump(save_data, filepath)
    
    @classmethod
    def load(cls, filepath: str) -> 'ICATransformer':
        """Load ICA transformer from disk.
        
        Args:
            filepath: Path to saved file
            
        Returns:
            Loaded ICATransformer
        """
        save_data = joblib.load(filepath)
        
        # Recreate instance
        instance = cls(
            n_components=save_data['n_components'],
            algorithm=save_data['algorithm'],
            whiten=save_data['whiten'],
            max_iter=save_data['max_iter'],
            tol=save_data['tol'],
            random_state=save_data['random_state'],
            fun=save_data['fun'],
        )
        
        # Restore fitted components
        instance.scaler_ = save_data['scaler']
        instance.ica_ = save_data['ica']
        instance.result_ = save_data['result']
        instance.feature_names_ = save_data['feature_names']
        instance.is_fitted_ = True
        
        return instance
    
    def get_reconstruction_error(self, X: pd.DataFrame) -> float:
        """Compute reconstruction error.
        
        Args:
            X: Original data
            
        Returns:
            Mean squared reconstruction error
        """
        if not self.is_fitted_:
            raise ValueError("ICATransformer must be fitted first")
        
        S = self.transform(X)
        X_reconstructed = self.inverse_transform(S)
        
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X
        
        mse = np.mean((X_array - X_reconstructed.values) ** 2)
        
        return mse


def compare_pca_vs_ica(
    X: pd.DataFrame,
    n_components: int = 10
) -> Dict[str, any]:
    """Compare PCA and ICA on the same data.
    
    Args:
        X: Input data
        n_components: Number of components
        
    Returns:
        Dictionary with comparison results
    """
    from sklearn.decomposition import PCA
    
    # Fit ICA
    ica = ICATransformer(n_components=n_components)
    ica.fit(X)
    X_ica = ica.transform(X)
    
    # Fit PCA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    
    # Compare
    results = {
        'ica': {
            'transformer': ica,
            'components': X_ica,
            'kurtosis': ica.result_.component_kurtosis,
            'reconstruction_error': ica.get_reconstruction_error(X)
        },
        'pca': {
            'transformer': pca,
            'components': X_pca,
            'variance_explained': pca.explained_variance_ratio_,
            'reconstruction_error': np.mean((X_scaled - pca.inverse_transform(X_pca)) ** 2)
        }
    }
    
    return results
