"""AutoML Suggestions Module.

Provides intelligent recommendations for ML techniques based on
dataset characteristics. Suggestions are linked to existing modules
in the application.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


class SuggestionCategory(Enum):
    """Categories of ML technique suggestions."""
    
    PREPROCESSING = "preprocessing"
    FEATURE_ENGINEERING = "feature_engineering"
    MODEL = "model"
    HYPERPARAMETERS = "hyperparameters"
    EVALUATION = "evaluation"
    IMBALANCE = "imbalance"


class Priority(Enum):
    """Priority levels for suggestions."""
    
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class DatasetAnalysis:
    """Analysis results of a dataset.
    
    Contains computed statistics and characteristics that drive
    the suggestion engine.
    """
    
    # Basic info
    n_samples: int = 0
    n_features: int = 0
    n_numeric_features: int = 0
    n_categorical_features: int = 0
    
    # Target analysis (for classification)
    n_classes: int = 0
    class_distribution: Dict[Any, float] = field(default_factory=dict)
    imbalance_ratio: float = 1.0
    is_imbalanced: bool = False
    
    # Missing data
    missing_percentage: float = 0.0
    features_with_missing: List[str] = field(default_factory=list)
    
    # Feature characteristics
    highly_correlated_pairs: List[Tuple[str, str, float]] = field(default_factory=list)
    low_variance_features: List[str] = field(default_factory=list)
    skewed_features: List[str] = field(default_factory=list)
    outlier_features: List[str] = field(default_factory=list)
    high_cardinality_features: List[str] = field(default_factory=list)
    
    # Data types
    has_categorical: bool = False
    has_text: bool = False
    has_datetime: bool = False
    
    # Size category
    size_category: str = "medium"  # small, medium, large, very_large


@dataclass
class TechniqueSuggestion:
    """A suggestion for a ML technique.
    
    Contains the suggestion details and links to relevant
    application modules.
    """
    
    # Core info
    title: str
    description: str
    category: SuggestionCategory
    priority: Priority
    
    # Rationale
    reason: str
    expected_benefit: str
    
    # Link to application
    module_link: Optional[str] = None  # Dashboard page or code module
    code_example: Optional[str] = None
    
    # Technical details
    technique_name: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    # Actions
    auto_applicable: bool = False  # Can be applied automatically
    requires_user_input: bool = False


class AutoMLSuggestions:
    """Engine for generating ML technique suggestions.
    
    Analyzes dataset characteristics and provides targeted
    recommendations linked to specific modules.
    """
    
    def __init__(self):
        """Initialize the suggestions engine."""
        self.analysis: Optional[DatasetAnalysis] = None
        self.suggestions: List[TechniqueSuggestion] = []
    
    def analyze_dataset(
        self,
        df: pd.DataFrame,
        target_column: Optional[str] = None,
        task: str = "classification"
    ) -> DatasetAnalysis:
        """
        Analyze dataset characteristics.
        
        Args:
            df: Dataset to analyze
            target_column: Name of target column
            task: Task type (classification, regression)
            
        Returns:
            DatasetAnalysis with computed statistics
        """
        analysis = DatasetAnalysis()
        
        # Basic info
        analysis.n_samples = len(df)
        analysis.n_features = len(df.columns) - (1 if target_column else 0)
        
        # Feature types
        feature_cols = [c for c in df.columns if c != target_column]
        numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df[feature_cols].select_dtypes(include=['object', 'category']).columns.tolist()
        
        analysis.n_numeric_features = len(numeric_cols)
        analysis.n_categorical_features = len(categorical_cols)
        analysis.has_categorical = len(categorical_cols) > 0
        
        # Check for datetime
        datetime_cols = df[feature_cols].select_dtypes(include=['datetime64']).columns.tolist()
        analysis.has_datetime = len(datetime_cols) > 0
        
        # Target analysis (classification)
        if target_column and target_column in df.columns and task == "classification":
            y = df[target_column]
            analysis.n_classes = y.nunique()
            
            # Class distribution
            value_counts = y.value_counts(normalize=True)
            analysis.class_distribution = value_counts.to_dict()
            
            # Imbalance ratio
            if len(value_counts) >= 2:
                max_class = value_counts.iloc[0]
                min_class = value_counts.iloc[-1]
                analysis.imbalance_ratio = max_class / min_class if min_class > 0 else float('inf')
                analysis.is_imbalanced = analysis.imbalance_ratio > 3
        
        # Missing data
        missing_per_column = df[feature_cols].isnull().mean()
        analysis.missing_percentage = missing_per_column.mean() * 100
        analysis.features_with_missing = missing_per_column[missing_per_column > 0].index.tolist()
        
        # Correlation analysis (for numeric features)
        if len(numeric_cols) > 1:
            try:
                corr_matrix = df[numeric_cols].corr().abs()
                # Find highly correlated pairs (> 0.9)
                for i in range(len(numeric_cols)):
                    for j in range(i+1, len(numeric_cols)):
                        if corr_matrix.iloc[i, j] > 0.9:
                            analysis.highly_correlated_pairs.append(
                                (numeric_cols[i], numeric_cols[j], corr_matrix.iloc[i, j])
                            )
            except:
                pass
        
        # Low variance features
        if len(numeric_cols) > 0:
            try:
                variance = df[numeric_cols].var()
                threshold = variance.quantile(0.05)
                analysis.low_variance_features = variance[variance < threshold].index.tolist()
            except:
                pass
        
        # Skewness
        if len(numeric_cols) > 0:
            try:
                skewness = df[numeric_cols].skew().abs()
                analysis.skewed_features = skewness[skewness > 1.0].index.tolist()
            except:
                pass
        
        # Size category
        if analysis.n_samples < 500:
            analysis.size_category = "small"
        elif analysis.n_samples < 5000:
            analysis.size_category = "medium"
        elif analysis.n_samples < 50000:
            analysis.size_category = "large"
        else:
            analysis.size_category = "very_large"
        
        self.analysis = analysis
        return analysis
    
    def generate_suggestions(
        self,
        analysis: Optional[DatasetAnalysis] = None
    ) -> List[TechniqueSuggestion]:
        """
        Generate technique suggestions based on analysis.
        
        Args:
            analysis: Dataset analysis (uses stored if not provided)
            
        Returns:
            List of TechniqueSuggestion objects
        """
        if analysis is None:
            analysis = self.analysis
        
        if analysis is None:
            raise ValueError("No analysis available. Call analyze_dataset first.")
        
        suggestions = []
        
        # === IMBALANCE HANDLING ===
        if analysis.is_imbalanced:
            suggestions.extend(self._suggest_imbalance_handling(analysis))
        
        # === MISSING DATA ===
        if analysis.missing_percentage > 0:
            suggestions.extend(self._suggest_missing_handling(analysis))
        
        # === FEATURE ENGINEERING ===
        if analysis.highly_correlated_pairs:
            suggestions.extend(self._suggest_correlation_handling(analysis))
        
        if analysis.skewed_features:
            suggestions.extend(self._suggest_skewness_handling(analysis))
        
        if analysis.low_variance_features:
            suggestions.extend(self._suggest_variance_handling(analysis))
        
        # === MODEL SELECTION ===
        suggestions.extend(self._suggest_models(analysis))
        
        # === EVALUATION ===
        suggestions.extend(self._suggest_evaluation(analysis))
        
        # Sort by priority
        priority_order = {Priority.HIGH: 0, Priority.MEDIUM: 1, Priority.LOW: 2}
        suggestions.sort(key=lambda x: priority_order[x.priority])
        
        self.suggestions = suggestions
        return suggestions
    
    def _suggest_imbalance_handling(
        self,
        analysis: DatasetAnalysis
    ) -> List[TechniqueSuggestion]:
        """Generate suggestions for handling class imbalance."""
        suggestions = []
        
        # SMOTE
        suggestions.append(TechniqueSuggestion(
            title="Aplicar SMOTE",
            description="Synthetic Minority Over-sampling Technique para balancear clases",
            category=SuggestionCategory.IMBALANCE,
            priority=Priority.HIGH,
            reason=f"Dataset desbalanceado con ratio {analysis.imbalance_ratio:.1f}:1",
            expected_benefit="Mejora en recall de clase minoritaria sin perder muchos datos",
            module_link="02_🤖_Model_Training",
            code_example="from imblearn.over_sampling import SMOTE\nsmote = SMOTE(random_state=42)\nX_res, y_res = smote.fit_resample(X, y)",
            technique_name="SMOTE",
            parameters={"sampling_strategy": "auto", "k_neighbors": 5},
            auto_applicable=True,
        ))
        
        # Class weights
        suggestions.append(TechniqueSuggestion(
            title="Usar class_weight='balanced'",
            description="Ajustar pesos de clase en el modelo",
            category=SuggestionCategory.IMBALANCE,
            priority=Priority.HIGH,
            reason="Alternativa más simple a SMOTE que no requiere generar datos sintéticos",
            expected_benefit="Penaliza más los errores en la clase minoritaria",
            module_link="02_🤖_Model_Training",
            technique_name="class_weight",
            parameters={"class_weight": "balanced"},
            auto_applicable=True,
        ))
        
        # XGBoost con scale_pos_weight
        if analysis.imbalance_ratio > 5:
            suggestions.append(TechniqueSuggestion(
                title="Usar XGBoost Balanced",
                description="XGBoost con scale_pos_weight ajustado al ratio de clases",
                category=SuggestionCategory.MODEL,
                priority=Priority.HIGH,
                reason=f"XGBoost con scale_pos_weight={analysis.imbalance_ratio:.1f} para desbalance severo",
                expected_benefit="Modelo robusto con manejo nativo de desbalance",
                module_link="02_🤖_Model_Training",
                technique_name="xgb_balanced",
                parameters={"scale_pos_weight": round(analysis.imbalance_ratio, 1)},
                auto_applicable=True,
            ))
        
        return suggestions
    
    def _suggest_missing_handling(
        self,
        analysis: DatasetAnalysis
    ) -> List[TechniqueSuggestion]:
        """Generate suggestions for handling missing data."""
        suggestions = []
        
        n_missing = len(analysis.features_with_missing)
        
        if analysis.missing_percentage < 5:
            # Low missing: simple imputation
            suggestions.append(TechniqueSuggestion(
                title="Imputación Simple",
                description="Usar media/mediana para valores numéricos y moda para categóricos",
                category=SuggestionCategory.PREPROCESSING,
                priority=Priority.MEDIUM,
                reason=f"Solo {analysis.missing_percentage:.1f}% de datos faltantes",
                expected_benefit="Rápido y efectivo para pocos valores faltantes",
                module_link="00_🧹_Data_Cleaning_and_EDA",
                technique_name="simple_imputation",
                auto_applicable=True,
            ))
        elif analysis.missing_percentage < 20:
            # Moderate: KNN or iterative
            suggestions.append(TechniqueSuggestion(
                title="Imputación por KNN",
                description="Usar K-Nearest Neighbors para imputar valores basándose en muestras similares",
                category=SuggestionCategory.PREPROCESSING,
                priority=Priority.HIGH,
                reason=f"{n_missing} features con datos faltantes ({analysis.missing_percentage:.1f}% total)",
                expected_benefit="Preserva relaciones entre variables mejor que imputación simple",
                module_link="00_🧹_Data_Cleaning_and_EDA",
                technique_name="knn_imputation",
                parameters={"n_neighbors": 5},
                auto_applicable=True,
            ))
            
            suggestions.append(TechniqueSuggestion(
                title="Imputación Iterativa (MICE)",
                description="Multiple Imputation by Chained Equations",
                category=SuggestionCategory.PREPROCESSING,
                priority=Priority.MEDIUM,
                reason="Método más sofisticado para datos faltantes moderados",
                expected_benefit="Captura relaciones complejas entre variables",
                module_link="02_🤖_Model_Training",
                technique_name="iterative_imputation",
                auto_applicable=True,
            ))
        else:
            # High missing: consider indicator + imputation
            suggestions.append(TechniqueSuggestion(
                title="Indicadores de Datos Faltantes",
                description="Crear columnas indicadoras para datos faltantes y luego imputar",
                category=SuggestionCategory.FEATURE_ENGINEERING,
                priority=Priority.HIGH,
                reason=f"Alto porcentaje de datos faltantes ({analysis.missing_percentage:.1f}%)",
                expected_benefit="El patrón de datos faltantes puede ser informativo",
                module_link="00_🧹_Data_Cleaning_and_EDA",
                technique_name="missing_indicator",
                auto_applicable=True,
            ))
        
        return suggestions
    
    def _suggest_correlation_handling(
        self,
        analysis: DatasetAnalysis
    ) -> List[TechniqueSuggestion]:
        """Generate suggestions for correlated features."""
        suggestions = []
        
        n_pairs = len(analysis.highly_correlated_pairs)
        
        suggestions.append(TechniqueSuggestion(
            title="Eliminar Features Correlacionados",
            description=f"Eliminar {n_pairs} pares de features altamente correlacionados (>0.9)",
            category=SuggestionCategory.FEATURE_ENGINEERING,
            priority=Priority.MEDIUM,
            reason=f"Detectados {n_pairs} pares con correlación >0.9",
            expected_benefit="Reduce redundancia y mejora interpretabilidad",
            module_link="00_🧹_Data_Cleaning_and_EDA",
            technique_name="remove_correlated",
            parameters={"threshold": 0.9, "pairs": analysis.highly_correlated_pairs[:5]},
            requires_user_input=True,
        ))
        
        suggestions.append(TechniqueSuggestion(
            title="Aplicar PCA",
            description="Reducir dimensionalidad con Principal Component Analysis",
            category=SuggestionCategory.FEATURE_ENGINEERING,
            priority=Priority.LOW,
            reason="PCA puede manejar multicolinealidad",
            expected_benefit="Reduce dimensionalidad preservando varianza",
            module_link="02_🤖_Model_Training",
            technique_name="pca",
            parameters={"n_components": 0.95},
            auto_applicable=True,
        ))
        
        return suggestions
    
    def _suggest_skewness_handling(
        self,
        analysis: DatasetAnalysis
    ) -> List[TechniqueSuggestion]:
        """Generate suggestions for skewed features."""
        suggestions = []
        
        n_skewed = len(analysis.skewed_features)
        
        suggestions.append(TechniqueSuggestion(
            title="Transformar Features Sesgados",
            description=f"Aplicar log o Box-Cox a {n_skewed} features con alta asimetría",
            category=SuggestionCategory.PREPROCESSING,
            priority=Priority.MEDIUM,
            reason=f"{n_skewed} features con skewness > 1.0",
            expected_benefit="Mejora performance de modelos lineales y distancias",
            module_link="00_🧹_Data_Cleaning_and_EDA",
            technique_name="log_transform",
            parameters={"features": analysis.skewed_features[:5]},
            auto_applicable=True,
        ))
        
        return suggestions
    
    def _suggest_variance_handling(
        self,
        analysis: DatasetAnalysis
    ) -> List[TechniqueSuggestion]:
        """Generate suggestions for low variance features."""
        suggestions = []
        
        n_low_var = len(analysis.low_variance_features)
        
        if n_low_var > 0:
            suggestions.append(TechniqueSuggestion(
                title="Eliminar Features de Baja Varianza",
                description=f"Considerar eliminar {n_low_var} features con varianza muy baja",
                category=SuggestionCategory.FEATURE_ENGINEERING,
                priority=Priority.LOW,
                reason="Features con poca variación aportan poca información",
                expected_benefit="Reduce dimensionalidad sin perder información útil",
                module_link="00_🧹_Data_Cleaning_and_EDA",
                technique_name="variance_threshold",
                parameters={"features": analysis.low_variance_features},
                requires_user_input=True,
            ))
        
        return suggestions
    
    def _suggest_models(
        self,
        analysis: DatasetAnalysis
    ) -> List[TechniqueSuggestion]:
        """Generate model suggestions based on dataset characteristics."""
        suggestions = []
        
        # Based on dataset size
        if analysis.size_category == "small":
            suggestions.append(TechniqueSuggestion(
                title="Usar Modelos Simples",
                description="Para datasets pequeños, preferir Logistic Regression o SVM",
                category=SuggestionCategory.MODEL,
                priority=Priority.HIGH,
                reason=f"Dataset pequeño ({analysis.n_samples} muestras)",
                expected_benefit="Menos riesgo de sobreajuste",
                module_link="02_🤖_Model_Training",
                technique_name="logistic_regression",
                auto_applicable=True,
            ))
        elif analysis.size_category in ["large", "very_large"]:
            suggestions.append(TechniqueSuggestion(
                title="Usar Gradient Boosting",
                description="XGBoost o LightGBM para datasets grandes",
                category=SuggestionCategory.MODEL,
                priority=Priority.HIGH,
                reason=f"Dataset grande ({analysis.n_samples} muestras)",
                expected_benefit="Alto performance y escalabilidad",
                module_link="02_🤖_Model_Training",
                technique_name="xgboost",
                auto_applicable=True,
            ))
        
        # Based on feature types
        if analysis.has_categorical and analysis.n_categorical_features > 5:
            suggestions.append(TechniqueSuggestion(
                title="Usar CatBoost o LightGBM",
                description="Modelos con soporte nativo para categorías",
                category=SuggestionCategory.MODEL,
                priority=Priority.MEDIUM,
                reason=f"Dataset con {analysis.n_categorical_features} features categóricos",
                expected_benefit="Mejor manejo de categorías sin one-hot encoding",
                module_link="02_🤖_Model_Training",
                technique_name="catboost",
                auto_applicable=True,
            ))
        
        # AutoML suggestion
        suggestions.append(TechniqueSuggestion(
            title="Ejecutar AutoML",
            description="Búsqueda automática de mejor modelo y hiperparámetros",
            category=SuggestionCategory.MODEL,
            priority=Priority.MEDIUM,
            reason="AutoML puede encontrar configuraciones óptimas automáticamente",
            expected_benefit="Ahorra tiempo de experimentación manual",
            module_link="09_🤖_AutoML",
            technique_name="automl",
            parameters={"preset": "balanced"},
            auto_applicable=True,
        ))
        
        return suggestions
    
    def _suggest_evaluation(
        self,
        analysis: DatasetAnalysis
    ) -> List[TechniqueSuggestion]:
        """Generate evaluation strategy suggestions."""
        suggestions = []
        
        # Cross-validation strategy
        if analysis.n_samples < 1000:
            suggestions.append(TechniqueSuggestion(
                title="Usar Repeated CV",
                description="Validación cruzada repetida para estimaciones más estables",
                category=SuggestionCategory.EVALUATION,
                priority=Priority.MEDIUM,
                reason="Dataset pequeño/mediano - una sola CV puede ser inestable",
                expected_benefit="Intervalos de confianza más fiables",
                module_link="02_🤖_Model_Training",
                technique_name="repeated_cv",
                parameters={"n_splits": 5, "n_repeats": 10},
                auto_applicable=True,
            ))
        
        # Metrics for imbalanced data
        if analysis.is_imbalanced:
            suggestions.append(TechniqueSuggestion(
                title="Usar AUPRC como Métrica Principal",
                description="Area Under Precision-Recall Curve para datos desbalanceados",
                category=SuggestionCategory.EVALUATION,
                priority=Priority.HIGH,
                reason="AUROC puede ser engañoso con clases muy desbalanceadas",
                expected_benefit="Evaluación más realista del modelo",
                module_link="04_📈_Model_Evaluation",
                technique_name="auprc",
                auto_applicable=True,
            ))
        
        return suggestions
    
    def get_suggestions_by_category(
        self,
        category: SuggestionCategory
    ) -> List[TechniqueSuggestion]:
        """Filter suggestions by category."""
        return [s for s in self.suggestions if s.category == category]
    
    def get_high_priority_suggestions(self) -> List[TechniqueSuggestion]:
        """Get only high priority suggestions."""
        return [s for s in self.suggestions if s.priority == Priority.HIGH]
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert suggestions to DataFrame for display."""
        rows = []
        for s in self.suggestions:
            rows.append({
                'Priority': s.priority.value,
                'Title': s.title,
                'Category': s.category.value,
                'Reason': s.reason,
                'Benefit': s.expected_benefit,
                'Module': s.module_link or 'N/A',
            })
        return pd.DataFrame(rows)


# Convenience functions

def analyze_dataset(
    df: pd.DataFrame,
    target_column: Optional[str] = None,
    task: str = "classification"
) -> DatasetAnalysis:
    """Analyze a dataset and return analysis results.
    
    Args:
        df: Dataset to analyze
        target_column: Target column name
        task: Task type
        
    Returns:
        DatasetAnalysis object
    """
    engine = AutoMLSuggestions()
    return engine.analyze_dataset(df, target_column, task)


def get_suggestions(
    df: pd.DataFrame,
    target_column: Optional[str] = None,
    task: str = "classification"
) -> List[TechniqueSuggestion]:
    """Analyze dataset and generate suggestions.
    
    Args:
        df: Dataset to analyze
        target_column: Target column name
        task: Task type
        
    Returns:
        List of suggestions
    """
    engine = AutoMLSuggestions()
    engine.analyze_dataset(df, target_column, task)
    return engine.generate_suggestions()
