"""Imbalanced data handling module for class imbalance in medical datasets.

This module provides comprehensive strategies for handling class imbalance,
a common problem in medical datasets like mortality prediction where the
minority class (e.g., death) is much less frequent than the majority class.

Techniques implemented:
- SMOTE (Synthetic Minority Over-sampling Technique)
- SMOTE-NC (SMOTE for Nominal and Continuous features)
- ADASYN (Adaptive Synthetic Sampling)
- BorderlineSMOTE (Focus on borderline samples)
- SVMSMOTE (SVM-guided SMOTE)
- Class Weight Adjustment
- Random Over/Under Sampling
- Combined sampling (SMOTE + Tomek Links)

References:
- Chawla et al. (2002) "SMOTE: Synthetic Minority Over-sampling Technique"
- He et al. (2008) "ADASYN: Adaptive Synthetic Sampling"
- Han et al. (2005) "Borderline-SMOTE"
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# Check for imbalanced-learn availability
try:
    from imblearn.over_sampling import (
        SMOTE,
        SMOTENC,
        ADASYN,
        BorderlineSMOTE,
        SVMSMOTE,
        RandomOverSampler,
    )
    from imblearn.under_sampling import (
        RandomUnderSampler,
        TomekLinks,
        EditedNearestNeighbours,
    )
    from imblearn.combine import SMOTETomek, SMOTEENN
    from imblearn.pipeline import Pipeline as ImbPipeline
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False
    SMOTE = None
    SMOTENC = None
    ADASYN = None
    BorderlineSMOTE = None
    SVMSMOTE = None
    RandomOverSampler = None
    RandomUnderSampler = None
    TomekLinks = None
    SMOTETomek = None
    SMOTEENN = None
    ImbPipeline = None


class ImbalanceStrategy(Enum):
    """Available strategies for handling class imbalance."""
    
    # No resampling
    NONE = "none"
    
    # Over-sampling methods
    SMOTE = "smote"                      # Standard SMOTE
    SMOTE_NC = "smote_nc"                # SMOTE for mixed features
    ADASYN = "adasyn"                    # Adaptive synthetic sampling
    BORDERLINE_SMOTE = "borderline_smote"  # Focus on borderline samples
    SVM_SMOTE = "svm_smote"              # SVM-guided SMOTE
    RANDOM_OVERSAMPLE = "random_oversample"  # Simple random oversampling
    
    # Under-sampling methods
    RANDOM_UNDERSAMPLE = "random_undersample"  # Simple random undersampling
    TOMEK_LINKS = "tomek_links"          # Remove Tomek links
    ENN = "edited_nearest_neighbours"     # Edited nearest neighbours
    
    # Combined methods
    SMOTE_TOMEK = "smote_tomek"          # SMOTE + Tomek links cleanup
    SMOTE_ENN = "smote_enn"              # SMOTE + ENN cleanup
    
    # Algorithm-specific (no resampling, use model's class_weight)
    CLASS_WEIGHT = "class_weight"         # Adjust model's class weights


# Strategy descriptions for UI/documentation
STRATEGY_DESCRIPTIONS: Dict[ImbalanceStrategy, Dict[str, str]] = {
    ImbalanceStrategy.NONE: {
        "name": "Sin Balanceo",
        "description": "No se aplica ninguna técnica de balanceo. Útil como baseline.",
        "pros": "Rápido, sin introducción de datos sintéticos",
        "cons": "Modelo puede estar sesgado hacia la clase mayoritaria",
        "recommended_for": "Datasets balanceados o como comparación baseline",
    },
    ImbalanceStrategy.SMOTE: {
        "name": "SMOTE",
        "description": "Genera datos sintéticos interpolando entre vecinos de la clase minoritaria usando K-NN.",
        "pros": "Estándar del estado del arte, no duplica datos existentes",
        "cons": "Puede generar datos en regiones ruidosas, solo para features numéricas",
        "recommended_for": "Datasets con features numéricas y desbalance moderado (1:3 a 1:10)",
    },
    ImbalanceStrategy.SMOTE_NC: {
        "name": "SMOTE-NC",
        "description": "Versión de SMOTE para datos mixtos (numéricos y categóricos).",
        "pros": "Maneja correctamente features categóricas sin crear valores inválidos",
        "cons": "Más lento que SMOTE estándar",
        "recommended_for": "Datasets con variables categóricas (sexo, tipo de tratamiento, etc.)",
    },
    ImbalanceStrategy.ADASYN: {
        "name": "ADASYN",
        "description": "SMOTE adaptativo que genera más muestras en regiones difíciles de aprender.",
        "pros": "Focaliza en regiones donde el modelo tiene dificultad",
        "cons": "Puede amplificar ruido en outliers",
        "recommended_for": "Cuando la distribución de la clase minoritaria es muy heterogénea",
    },
    ImbalanceStrategy.BORDERLINE_SMOTE: {
        "name": "Borderline-SMOTE",
        "description": "Solo genera datos sintéticos cerca de la frontera de decisión.",
        "pros": "Más preciso que SMOTE en fronteras de decisión complejas",
        "cons": "Puede no generar suficientes muestras si hay pocas en la frontera",
        "recommended_for": "Cuando las clases están claramente separadas con frontera definida",
    },
    ImbalanceStrategy.SVM_SMOTE: {
        "name": "SVM-SMOTE",
        "description": "Usa vectores de soporte de un SVM para guiar la generación de datos.",
        "pros": "Genera datos más informativos para clasificación",
        "cons": "Computacionalmente más costoso",
        "recommended_for": "Datasets donde SVM podría ser el clasificador final",
    },
    ImbalanceStrategy.RANDOM_OVERSAMPLE: {
        "name": "Random Oversampling",
        "description": "Duplica aleatoriamente muestras de la clase minoritaria.",
        "pros": "Simple, no introduce datos sintéticos",
        "cons": "Puede causar overfitting al duplicar muestras exactas",
        "recommended_for": "Datasets muy pequeños o como baseline",
    },
    ImbalanceStrategy.RANDOM_UNDERSAMPLE: {
        "name": "Random Undersampling",
        "description": "Elimina aleatoriamente muestras de la clase mayoritaria.",
        "pros": "Reduce tiempo de entrenamiento, mantiene datos originales",
        "cons": "Pierde información valiosa de la clase mayoritaria",
        "recommended_for": "Datasets muy grandes con desbalance extremo",
    },
    ImbalanceStrategy.TOMEK_LINKS: {
        "name": "Tomek Links",
        "description": "Elimina pares de muestras muy cercanas de clases diferentes.",
        "pros": "Limpia la frontera de decisión sin eliminar muchos datos",
        "cons": "Puede no ser suficiente para desbalances severos",
        "recommended_for": "Limpieza fina después de oversampling",
    },
    ImbalanceStrategy.SMOTE_TOMEK: {
        "name": "SMOTE + Tomek Links",
        "description": "Aplica SMOTE y luego limpia con Tomek Links.",
        "pros": "Combina generación de datos con limpieza de ruido",
        "cons": "Más lento que métodos individuales",
        "recommended_for": "Cuando SMOTE genera demasiados puntos ruidosos",
    },
    ImbalanceStrategy.SMOTE_ENN: {
        "name": "SMOTE + ENN",
        "description": "Aplica SMOTE y luego limpia con Edited Nearest Neighbours.",
        "pros": "Limpieza más agresiva que Tomek Links",
        "cons": "Puede eliminar demasiadas muestras en algunos casos",
        "recommended_for": "Datos ruidosos donde se necesita limpieza agresiva",
    },
    ImbalanceStrategy.CLASS_WEIGHT: {
        "name": "Class Weight (Algoritmo)",
        "description": "Ajusta los pesos de las clases en la función de pérdida del modelo.",
        "pros": "No modifica los datos, muchos modelos lo soportan nativamente",
        "cons": "No todos los algoritmos lo soportan igual de bien",
        "recommended_for": "Cuando no se quiere modificar los datos originales",
    },
}


@dataclass
class ImbalanceConfig:
    """Configuration for imbalance handling.
    
    Attributes:
        strategy: The resampling strategy to use
        sampling_strategy: Target ratio for resampling ('auto', 'minority', 'not minority', 
                          'not majority', 'all', or float ratio)
        k_neighbors: Number of nearest neighbors for SMOTE variants
        random_state: Random seed for reproducibility
        categorical_features: List of categorical feature indices (for SMOTE-NC)
        n_jobs: Number of parallel jobs (-1 for all cores)
    """
    strategy: ImbalanceStrategy = ImbalanceStrategy.SMOTE
    sampling_strategy: Union[str, float] = "auto"
    k_neighbors: int = 5
    random_state: int = 42
    categorical_features: Optional[List[int]] = None
    n_jobs: int = -1
    
    # Advanced options
    borderline_kind: str = "borderline-1"  # For BorderlineSMOTE
    svm_estimator: Optional[Any] = None     # For SVMSMOTE
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "strategy": self.strategy.value,
            "sampling_strategy": self.sampling_strategy,
            "k_neighbors": self.k_neighbors,
            "random_state": self.random_state,
            "categorical_features": self.categorical_features,
            "n_jobs": self.n_jobs,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ImbalanceConfig":
        """Create config from dictionary."""
        strategy = d.get("strategy", "smote")
        if isinstance(strategy, str):
            strategy = ImbalanceStrategy(strategy)
        return cls(
            strategy=strategy,
            sampling_strategy=d.get("sampling_strategy", "auto"),
            k_neighbors=d.get("k_neighbors", 5),
            random_state=d.get("random_state", 42),
            categorical_features=d.get("categorical_features"),
            n_jobs=d.get("n_jobs", -1),
        )


def detect_imbalance(
    y: Union[pd.Series, np.ndarray],
    threshold: float = 3.0,
) -> Tuple[bool, float, Dict[Any, int]]:
    """Detect if a dataset has class imbalance.
    
    Args:
        y: Target variable
        threshold: Ratio threshold to consider imbalanced (default: 3:1)
        
    Returns:
        Tuple of (is_imbalanced, imbalance_ratio, class_counts)
    """
    if isinstance(y, pd.Series):
        counts = y.value_counts().to_dict()
    else:
        unique, counts_arr = np.unique(y, return_counts=True)
        counts = dict(zip(unique, counts_arr))
    
    max_count = max(counts.values())
    min_count = min(counts.values())
    
    ratio = max_count / min_count if min_count > 0 else float('inf')
    is_imbalanced = ratio >= threshold
    
    return is_imbalanced, ratio, counts


def get_recommended_strategy(
    y: Union[pd.Series, np.ndarray],
    X: Optional[pd.DataFrame] = None,
) -> ImbalanceStrategy:
    """Get recommended imbalance strategy based on dataset characteristics.
    
    Args:
        y: Target variable
        X: Feature matrix (optional, used to detect categorical features)
        
    Returns:
        Recommended ImbalanceStrategy
    """
    is_imbalanced, ratio, counts = detect_imbalance(y)
    
    if not is_imbalanced:
        return ImbalanceStrategy.NONE
    
    # Check for categorical features
    has_categorical = False
    if X is not None:
        categorical_cols = X.select_dtypes(exclude=[np.number]).columns
        has_categorical = len(categorical_cols) > 0
    
    # Recommendations based on imbalance ratio and data characteristics
    if ratio > 20:
        # Extreme imbalance: use combined methods
        if has_categorical:
            return ImbalanceStrategy.CLASS_WEIGHT  # SMOTE-NC can struggle with extreme imbalance
        return ImbalanceStrategy.SMOTE_TOMEK
    elif ratio > 10:
        # High imbalance
        if has_categorical:
            return ImbalanceStrategy.SMOTE_NC
        return ImbalanceStrategy.ADASYN
    elif ratio > 5:
        # Moderate imbalance
        if has_categorical:
            return ImbalanceStrategy.SMOTE_NC
        return ImbalanceStrategy.SMOTE
    else:
        # Mild imbalance (3-5)
        return ImbalanceStrategy.CLASS_WEIGHT


def create_sampler(
    config: ImbalanceConfig,
) -> Optional[Any]:
    """Create a sampler object based on configuration.
    
    Args:
        config: Imbalance handling configuration
        
    Returns:
        Sampler object or None if strategy is NONE or CLASS_WEIGHT
    """
    if not IMBLEARN_AVAILABLE:
        raise ImportError(
            "imbalanced-learn is required for resampling. "
            "Install with: pip install imbalanced-learn"
        )
    
    if config.strategy in (ImbalanceStrategy.NONE, ImbalanceStrategy.CLASS_WEIGHT):
        return None
    
    common_params = {
        "sampling_strategy": config.sampling_strategy,
        "random_state": config.random_state,
    }
    
    if config.strategy == ImbalanceStrategy.SMOTE:
        return SMOTE(
            k_neighbors=config.k_neighbors,
            **common_params,
        )
    
    elif config.strategy == ImbalanceStrategy.SMOTE_NC:
        if config.categorical_features is None:
            raise ValueError("categorical_features must be provided for SMOTE-NC")
        return SMOTENC(
            categorical_features=config.categorical_features,
            k_neighbors=config.k_neighbors,
            **common_params,
        )
    
    elif config.strategy == ImbalanceStrategy.ADASYN:
        return ADASYN(
            n_neighbors=config.k_neighbors,
            **common_params,
        )
    
    elif config.strategy == ImbalanceStrategy.BORDERLINE_SMOTE:
        return BorderlineSMOTE(
            k_neighbors=config.k_neighbors,
            kind=config.borderline_kind,
            **common_params,
        )
    
    elif config.strategy == ImbalanceStrategy.SVM_SMOTE:
        return SVMSMOTE(
            k_neighbors=config.k_neighbors,
            svm_estimator=config.svm_estimator,
            **common_params,
        )
    
    elif config.strategy == ImbalanceStrategy.RANDOM_OVERSAMPLE:
        return RandomOverSampler(**common_params)
    
    elif config.strategy == ImbalanceStrategy.RANDOM_UNDERSAMPLE:
        return RandomUnderSampler(**common_params)
    
    elif config.strategy == ImbalanceStrategy.TOMEK_LINKS:
        return TomekLinks(n_jobs=config.n_jobs)
    
    elif config.strategy == ImbalanceStrategy.SMOTE_TOMEK:
        return SMOTETomek(
            smote=SMOTE(
                k_neighbors=config.k_neighbors,
                **common_params,
            ),
            n_jobs=config.n_jobs,
            random_state=config.random_state,
        )
    
    elif config.strategy == ImbalanceStrategy.SMOTE_ENN:
        return SMOTEENN(
            smote=SMOTE(
                k_neighbors=config.k_neighbors,
                **common_params,
            ),
            n_jobs=config.n_jobs,
            random_state=config.random_state,
        )
    
    else:
        raise ValueError(f"Unknown strategy: {config.strategy}")


def compute_class_weights(
    y: Union[pd.Series, np.ndarray],
    strategy: str = "balanced",
) -> Dict[Any, float]:
    """Compute class weights for imbalanced data.
    
    Args:
        y: Target variable
        strategy: 'balanced' or 'sqrt_balanced'
        
    Returns:
        Dictionary mapping class labels to weights
    """
    from sklearn.utils.class_weight import compute_class_weight
    
    if isinstance(y, pd.Series):
        classes = np.array(sorted(y.unique()))
        y_arr = y.values
    else:
        classes = np.unique(y)
        y_arr = y
    
    if strategy == "balanced":
        weights = compute_class_weight("balanced", classes=classes, y=y_arr)
    elif strategy == "sqrt_balanced":
        # Square root balanced - less aggressive than full balanced
        balanced_weights = compute_class_weight("balanced", classes=classes, y=y_arr)
        weights = np.sqrt(balanced_weights)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    return dict(zip(classes, weights))


def apply_class_weight_to_model(
    model: Any,
    class_weight: Union[str, Dict[Any, float]],
) -> Any:
    """Apply class weights to a model if supported.
    
    Args:
        model: sklearn-compatible model
        class_weight: 'balanced' or dictionary of weights
        
    Returns:
        Model with class weights set (or original if not supported)
    """
    if hasattr(model, "class_weight"):
        model.set_params(class_weight=class_weight)
    elif hasattr(model, "scale_pos_weight"):
        # XGBoost specific
        if isinstance(class_weight, dict) and len(class_weight) == 2:
            # Binary classification
            weights = list(class_weight.values())
            model.set_params(scale_pos_weight=weights[1] / weights[0])
    
    return model


def resample_data(
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    config: ImbalanceConfig,
) -> Tuple[Union[pd.DataFrame, np.ndarray], Union[pd.Series, np.ndarray]]:
    """Apply resampling to balance the dataset.
    
    Args:
        X: Feature matrix
        y: Target variable
        config: Imbalance handling configuration
        
    Returns:
        Tuple of (X_resampled, y_resampled)
    """
    if config.strategy in (ImbalanceStrategy.NONE, ImbalanceStrategy.CLASS_WEIGHT):
        return X, y
    
    sampler = create_sampler(config)
    
    # Store original format
    is_dataframe = isinstance(X, pd.DataFrame)
    is_series = isinstance(y, pd.Series)
    
    if is_dataframe:
        columns = X.columns
        index_name = X.index.name
    
    if is_series:
        y_name = y.name
    
    # Apply resampling
    X_res, y_res = sampler.fit_resample(X, y)
    
    # Restore format
    if is_dataframe:
        X_res = pd.DataFrame(X_res, columns=columns)
    
    if is_series:
        y_res = pd.Series(y_res, name=y_name)
    
    return X_res, y_res


def get_imbalance_report(
    y_original: Union[pd.Series, np.ndarray],
    y_resampled: Optional[Union[pd.Series, np.ndarray]] = None,
    strategy: Optional[ImbalanceStrategy] = None,
) -> Dict[str, Any]:
    """Generate a report on class distribution before/after resampling.
    
    Args:
        y_original: Original target variable
        y_resampled: Resampled target variable (optional)
        strategy: Strategy used (optional)
        
    Returns:
        Dictionary with imbalance statistics
    """
    is_imbalanced, ratio, counts = detect_imbalance(y_original)
    
    report = {
        "original": {
            "class_counts": counts,
            "total_samples": sum(counts.values()),
            "imbalance_ratio": round(ratio, 2),
            "is_imbalanced": is_imbalanced,
        }
    }
    
    if y_resampled is not None:
        is_imb_res, ratio_res, counts_res = detect_imbalance(y_resampled)
        report["resampled"] = {
            "class_counts": counts_res,
            "total_samples": sum(counts_res.values()),
            "imbalance_ratio": round(ratio_res, 2),
            "is_imbalanced": is_imb_res,
        }
        report["samples_added"] = sum(counts_res.values()) - sum(counts.values())
    
    if strategy:
        desc = STRATEGY_DESCRIPTIONS.get(strategy, {})
        report["strategy"] = {
            "name": strategy.value,
            "display_name": desc.get("name", strategy.value),
            "description": desc.get("description", ""),
        }
    
    return report


class ImbalanceHandler(BaseEstimator, TransformerMixin):
    """Sklearn-compatible transformer for handling class imbalance.
    
    This transformer wraps the imbalanced-learn samplers to work
    seamlessly in sklearn pipelines.
    
    Example:
        >>> handler = ImbalanceHandler(strategy="smote")
        >>> X_balanced, y_balanced = handler.fit_resample(X, y)
    """
    
    def __init__(
        self,
        strategy: Union[str, ImbalanceStrategy] = "smote",
        sampling_strategy: Union[str, float] = "auto",
        k_neighbors: int = 5,
        random_state: int = 42,
        categorical_features: Optional[List[int]] = None,
        n_jobs: int = -1,
    ):
        self.strategy = strategy
        self.sampling_strategy = sampling_strategy
        self.k_neighbors = k_neighbors
        self.random_state = random_state
        self.categorical_features = categorical_features
        self.n_jobs = n_jobs
        
        self._sampler = None
        self._config = None
    
    def _get_config(self) -> ImbalanceConfig:
        """Get ImbalanceConfig from parameters."""
        strategy = self.strategy
        if isinstance(strategy, str):
            strategy = ImbalanceStrategy(strategy)
        
        return ImbalanceConfig(
            strategy=strategy,
            sampling_strategy=self.sampling_strategy,
            k_neighbors=self.k_neighbors,
            random_state=self.random_state,
            categorical_features=self.categorical_features,
            n_jobs=self.n_jobs,
        )
    
    def fit(self, X, y=None):
        """Fit the handler (creates sampler)."""
        self._config = self._get_config()
        if self._config.strategy not in (ImbalanceStrategy.NONE, ImbalanceStrategy.CLASS_WEIGHT):
            self._sampler = create_sampler(self._config)
        return self
    
    def fit_resample(self, X, y):
        """Fit and resample the data."""
        self.fit(X, y)
        
        if self._sampler is None:
            return X, y
        
        return self._sampler.fit_resample(X, y)
    
    def transform(self, X):
        """Transform is identity (resampling happens in fit_resample)."""
        return X
    
    def get_report(self, y_original, y_resampled=None):
        """Get imbalance report."""
        strategy = self._config.strategy if self._config else ImbalanceStrategy(self.strategy)
        return get_imbalance_report(y_original, y_resampled, strategy)


# Convenience functions for common configurations
def get_smote_sampler(
    k_neighbors: int = 5,
    sampling_strategy: Union[str, float] = "auto",
    random_state: int = 42,
) -> Any:
    """Get a configured SMOTE sampler."""
    if not IMBLEARN_AVAILABLE:
        raise ImportError("imbalanced-learn required. Install with: pip install imbalanced-learn")
    return SMOTE(
        k_neighbors=k_neighbors,
        sampling_strategy=sampling_strategy,
        random_state=random_state,
        n_jobs=-1,
    )


def get_adasyn_sampler(
    n_neighbors: int = 5,
    sampling_strategy: Union[str, float] = "auto",
    random_state: int = 42,
) -> Any:
    """Get a configured ADASYN sampler."""
    if not IMBLEARN_AVAILABLE:
        raise ImportError("imbalanced-learn required. Install with: pip install imbalanced-learn")
    return ADASYN(
        n_neighbors=n_neighbors,
        sampling_strategy=sampling_strategy,
        random_state=random_state,
        n_jobs=-1,
    )


def get_combined_sampler(
    method: str = "smote_tomek",
    k_neighbors: int = 5,
    random_state: int = 42,
) -> Any:
    """Get a combined over/undersampling sampler."""
    if not IMBLEARN_AVAILABLE:
        raise ImportError("imbalanced-learn required. Install with: pip install imbalanced-learn")
    
    if method == "smote_tomek":
        return SMOTETomek(
            smote=SMOTE(k_neighbors=k_neighbors, random_state=random_state, n_jobs=-1),
            random_state=random_state,
            n_jobs=-1,
        )
    elif method == "smote_enn":
        return SMOTEENN(
            smote=SMOTE(k_neighbors=k_neighbors, random_state=random_state, n_jobs=-1),
            random_state=random_state,
            n_jobs=-1,
        )
    else:
        raise ValueError(f"Unknown method: {method}")


# Check availability
def is_imblearn_available() -> bool:
    """Check if imbalanced-learn is available."""
    return IMBLEARN_AVAILABLE


__all__ = [
    # Enums and configs
    "ImbalanceStrategy",
    "ImbalanceConfig",
    "STRATEGY_DESCRIPTIONS",
    # Main functions
    "detect_imbalance",
    "get_recommended_strategy",
    "create_sampler",
    "compute_class_weights",
    "apply_class_weight_to_model",
    "resample_data",
    "get_imbalance_report",
    # Class
    "ImbalanceHandler",
    # Convenience functions
    "get_smote_sampler",
    "get_adasyn_sampler",
    "get_combined_sampler",
    "is_imblearn_available",
]
