"""Multivariate analysis utilities (PCA)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


@dataclass
class PCAResults:
    """Results from Principal Component Analysis."""
    
    n_components: int
    explained_variance: List[float]
    explained_variance_ratio: List[float]
    cumulative_variance: List[float]
    components: np.ndarray
    feature_names: List[str]
    transformed_data: Optional[pd.DataFrame] = None


def perform_pca(
    df: pd.DataFrame,
    numeric_cols: List[str],
    n_components: Optional[int] = None,
    variance_threshold: float = 0.95,
    scale: bool = True,
) -> PCAResults:
    """Perform Principal Component Analysis (PCA).
    
    Args:
        df: DataFrame
        numeric_cols: List of numerical column names
        n_components: Number of components. If None, uses variance_threshold
        variance_threshold: Desired cumulative variance (0-1)
        scale: Whether to standardize data before PCA
        
    Returns:
        PCAResults object
    """
    if len(numeric_cols) < 2:
        raise ValueError("Se requieren al menos 2 variables numéricas para realizar PCA")
    
    # Prepare data - filter only numeric columns that exist and have numeric data
    valid_numeric_cols = []
    for col in numeric_cols:
        if col in df.columns:
            try:
                # Try to convert to numeric and check if there are any valid values
                col_numeric = pd.to_numeric(df[col], errors='coerce')
                if col_numeric.notna().sum() > 0:
                    valid_numeric_cols.append(col)
            except:
                continue
    
    if len(valid_numeric_cols) < 2:
        raise ValueError(
            f"Se necesitan al menos 2 variables numéricas con datos válidos para PCA. "
            f"Solo se encontraron {len(valid_numeric_cols)} variable(s) válida(s)."
        )
    
    # Prepare data - drop rows with any missing values in valid numeric columns
    df_numeric = df[valid_numeric_cols].copy()
    
    # Convert all columns to numeric (in case they weren't)
    for col in valid_numeric_cols:
        df_numeric[col] = pd.to_numeric(df_numeric[col], errors='coerce')
    
    # Drop rows with any NaN values
    df_numeric = df_numeric.dropna()
    
    # Check if we have enough samples after dropping NaN
    if len(df_numeric) == 0:
        raise ValueError(
            "Todas las filas contienen al menos un valor faltante en las variables numéricas. "
            "Es necesario aplicar imputación de valores faltantes antes de ejecutar PCA."
        )
    
    if len(df_numeric) < 2:
        raise ValueError(
            f"Se requieren al menos 2 observaciones completas para PCA, pero solo hay {len(df_numeric)}. "
            "Aplique imputación de valores faltantes para aumentar el número de filas válidas."
        )
        raise ValueError(
            f"Se requieren al menos 2 muestras para PCA. Solo se encontraron {len(df_numeric)} filas "
            "sin valores faltantes. Por favor, impute los valores faltantes primero."
        )
    
    # Scale if requested
    if scale:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df_numeric)
    else:
        X_scaled = df_numeric.values
    
    # Initial PCA to see all components
    pca_full = PCA()
    pca_full.fit(X_scaled)
    
    # Determine number of components
    if n_components is None:
        cumsum = np.cumsum(pca_full.explained_variance_ratio_)
        n_components = int(np.argmax(cumsum >= variance_threshold) + 1)
    
    # Final PCA with n_components
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    
    # Create DataFrame with components
    pc_columns = [f'PC{i+1}' for i in range(n_components)]
    df_pca = pd.DataFrame(X_pca, columns=pc_columns, index=df_numeric.index)
    
    # Results
    return PCAResults(
        n_components=n_components,
        explained_variance=pca.explained_variance_.tolist(),
        explained_variance_ratio=pca.explained_variance_ratio_.tolist(),
        cumulative_variance=np.cumsum(pca.explained_variance_ratio_).tolist(),
        components=pca.components_,
        feature_names=numeric_cols,
        transformed_data=df_pca
    )


def get_feature_importance_pca(
    pca_results: PCAResults, 
    n_components: int = 3
) -> pd.DataFrame:
    """Get feature importance in first N components.
    
    Args:
        pca_results: PCAResults object
        n_components: Number of components to consider
        
    Returns:
        DataFrame with feature importance
    """
    n_comp = min(n_components, pca_results.n_components)
    
    # Create DataFrame with loadings
    loadings_df = pd.DataFrame(
        pca_results.components[:n_comp, :],
        columns=pca_results.feature_names,
        index=[f'PC{i+1}' for i in range(n_comp)]
    ).T
    
    # Calculate total importance (sum of squared loadings)
    loadings_df['importance'] = np.sqrt((loadings_df ** 2).sum(axis=1))
    loadings_df = loadings_df.sort_values('importance', ascending=False)
    
    return loadings_df
