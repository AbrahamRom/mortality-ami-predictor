"""
Modelo Stacking (Apilamiento) para Predicción de Mortalidad en IAM
Meta-learning: Combina múltiples modelos base mediante un meta-modelo
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Tuple, Optional, Any
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, roc_curve, auc, 
                             confusion_matrix, classification_report,
                             precision_score, recall_score, f1_score,
                             specificity_score)
import pickle
import os
from datetime import datetime
import warnings

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StackingPredictor:
    """
    Modelo Stacking que combina predicciones de múltiples modelos base
    mediante un meta-modelo (meta-learner).
    
    Parámetros:
    -----------
    base_models : dict
        Diccionario con modelos base: {'nombre': modelo_entrenado}
        Ejemplo: {'xgboost': xgb_model, 'random_forest': rf_model, ...}
    meta_model : estimador, optional
        Modelo que aprende a combinar las predicciones base 
        (default: LogisticRegression)
    cv : int
        Número de folds para validación cruzada (default: 5)
    random_state : int
        Semilla para reproducibilidad (default: 42)
    
    Atributos:
    ----------
    meta_model_fitted : bool
        Indica si el meta-modelo ha sido entrenado
    training_timestamp : datetime
        Marca de tiempo del entrenamiento
    meta_features_shape_ : tuple
        Shape de las meta-características
    
    Ejemplo:
    --------
    >>> base_models = {
    ...     'xgboost': xgb_model,
    ...     'random_forest': rf_model,
    ...     'logistic_regression': lr_model
    ... }
    >>> stacking = StackingPredictor(base_models)
    >>> stacking.fit(X_train, y_train)
    >>> predictions = stacking.predict(X_test)
    """
    
    def __init__(self, base_models: Dict[str, Any], 
                 meta_model: Optional[Any] = None, 
                 cv: int = 5,
                 random_state: int = 42):
        """Inicializar el modelo Stacking con validación"""
        
        # Validar inputs
        if not isinstance(base_models, dict) or len(base_models) == 0:
            raise ValueError("base_models debe ser un diccionario no vacío")
        
        if len(base_models) < 2:
            warnings.warn("Se recomienda al menos 2 modelos base para Stacking", 
                         UserWarning)
        
        self.base_models = base_models
        self.meta_model = (meta_model if meta_model is not None 
                          else LogisticRegression(max_iter=1000, random_state=random_state))
        self.cv = cv
        self.random_state = random_state
        self.meta_model_fitted = False
        self.training_timestamp = None
        self.meta_features_shape_ = None
        self._base_model_names = list(base_models.keys())
        
        logger.info(f"StackingPredictor inicializado con {len(base_models)} modelos base: "
                   f"{self._base_model_names}")
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'StackingPredictor':
        """
        Entrenar el meta-modelo usando predicciones de modelos base
        
        Parámetros:
        -----------
        X : array-like de shape (n_samples, n_features)
            Datos de entrenamiento
        y : array-like de shape (n_samples,)
            Etiquetas objetivo (0 o 1 para clasificación binaria)
            
        Retorna:
        --------
        self : StackingPredictor
            Retorna el objeto para permitir encadenamiento de métodos
            
        Raises:
        -------
        ValueError : Si X e y tienen número diferente de muestras
        """
        # Validar datos
        X = np.asarray(X)
        y = np.asarray(y)
        
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X e y tienen diferente número de muestras: "
                           f"{X.shape[0]} vs {y.shape[0]}")
        
        if X.ndim != 2:
            raise ValueError(f"X debe ser 2D, se recibió shape {X.shape}")
        
        logger.info(f"[Stacking] Iniciando entrenamiento con {X.shape[0]} muestras "
                   f"y {X.shape[1]} características")
        self.training_timestamp = datetime.now()
        
        try:
            # Paso 1: Generar predicciones de modelos base usando CV
            logger.info("[Stacking] Generando predicciones base mediante validación cruzada...")
            meta_features = self._generate_meta_features(X, y)
            self.meta_features_shape_ = meta_features.shape
            
            # Paso 2: Entrenar meta-modelo con las predicciones base
            logger.info("[Stacking] Entrenando meta-modelo...")
            self.meta_model.fit(meta_features, y)
            self.meta_model_fitted = True
            
            logger.info("[Stacking] Entrenamiento completado exitosamente.")
            
        except Exception as e:
            logger.error(f"Error durante el entrenamiento: {str(e)}")
            raise
        
        return self
    
    def _generate_meta_features(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Generar características meta usando validación cruzada
        Previene overfitting del meta-modelo
        
        Parámetros:
        -----------
        X, y : datos de entrenamiento
        
        Retorna:
        --------
        meta_features : array de shape (n_samples, n_base_models)
            Matriz de predicciones de todos los modelos base
        """
        skf = StratifiedKFold(n_splits=self.cv, shuffle=True, 
                             random_state=self.random_state)
        meta_features = np.zeros((X.shape[0], len(self.base_models)))
        
        for i, model_name in enumerate(self._base_model_names):
            model = self.base_models[model_name]
            logger.info(f"  Procesando modelo base [{i+1}/{len(self.base_models)}]: {model_name}")
            
            try:
                # Usar predicciones probabilísticas si están disponibles
                if hasattr(model, 'predict_proba'):
                    predictions = cross_val_predict(
                        model, X, y, cv=skf, method='predict_proba'
                    )
                    # Tomar probabilidad de la clase positiva (clase 1)
                    meta_features[:, i] = predictions[:, 1]
                else:
                    # Si no hay predict_proba, usar predicciones directas
                    predictions = cross_val_predict(model, X, y, cv=skf)
                    meta_features[:, i] = predictions
                    
            except Exception as e:
                logger.warning(f"Error procesando {model_name}: {str(e)}")
                raise
        
        return meta_features
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Realizar predicciones en nuevos datos
        
        Parámetros:
        -----------
        X : array-like de shape (n_samples, n_features)
        
        Retorna:
        --------
        predictions : array de etiquetas predichas (0 o 1)
        """
        if not self.meta_model_fitted:
            raise ValueError("El modelo Stacking no ha sido entrenado. Use .fit() primero.")
        
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError(f"X debe ser 2D, se recibió shape {X.shape}")
        
        meta_features = self._predict_meta_features(X)
        return self.meta_model.predict(meta_features)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Obtener probabilidades predichas
        
        Parámetros:
        -----------
        X : array-like de shape (n_samples, n_features)
        
        Retorna:
        --------
        probabilities : array de shape (n_samples, 2)
            Probabilidades para cada clase
        """
        if not self.meta_model_fitted:
            raise ValueError("El modelo Stacking no ha sido entrenado. Use .fit() primero.")
        
        X = np.asarray(X)
        meta_features = self._predict_meta_features(X)
        
        if hasattr(self.meta_model, 'predict_proba'):
            return self.meta_model.predict_proba(meta_features)
        else:
            # Fallback si meta_model no tiene predict_proba
            predictions = self.meta_model.predict(meta_features)
            return np.column_stack([1 - predictions, predictions])
    
    def _predict_meta_features(self, X: np.ndarray) -> np.ndarray:
        """
        Generar características meta para datos nuevos usando todos los modelos base
        
        Parámetros:
        -----------
        X : datos nuevos
        
        Retorna:
        --------
        meta_features : array de predicciones base
        """
        meta_features = np.zeros((X.shape[0], len(self.base_models)))
        
        for i, model_name in enumerate(self._base_model_names):
            model = self.base_models[model_name]
            
            if hasattr(model, 'predict_proba'):
                predictions = model.predict_proba(X)
                meta_features[:, i] = predictions[:, 1]
            else:
                predictions = model.predict(X)
                meta_features[:, i] = predictions
        
        return meta_features
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray, 
                verbose: bool = True) -> Dict[str, Any]:
        """
        Evaluar el modelo Stacking en datos de prueba
        
        Parámetros:
        -----------
        X_test, y_test : datos de prueba
        verbose : bool, mostrar resultados detallados
        
        Retorna:
        --------
        metrics : dict con métricas de evaluación
            Incluye: auroc, accuracy, precision, recall, f1, specificity,
                     confusion_matrix, classification_report
        """
        y_test = np.asarray(y_test)
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)[:, 1]
        
        # Calcular métricas completas
        metrics = {
            'auroc': roc_auc_score(y_test, y_proba),
            'accuracy': np.mean(y_pred == y_test),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'specificity': 1 - (np.sum((y_pred == 1) & (y_test == 0)) / 
                               np.sum(y_test == 0)),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred, 
                                                          output_dict=True)
        }
        
        if verbose:
            print("\n" + "="*60)
            print("EVALUACIÓN DEL MODELO STACKING")
            print("="*60)
            print(f"AUROC:       {metrics['auroc']:.4f}")
            print(f"Accuracy:    {metrics['accuracy']:.4f}")
            print(f"Precision:   {metrics['precision']:.4f}")
            print(f"Recall:      {metrics['recall']:.4f}")
            print(f"F1-Score:    {metrics['f1']:.4f}")
            print(f"Specificity: {metrics['specificity']:.4f}")
            print("\nMatriz de Confusión:")
            print(metrics['confusion_matrix'])
            print("\nReporte de Clasificación:")
            print(classification_report(y_test, y_pred))
        
        return metrics
    
    def get_meta_model_weights(self) -> pd.DataFrame:
        """
        Obtener los pesos del meta-modelo (cómo pondera cada modelo base)
        
        Retorna:
        --------
        weights_df : DataFrame con pesos de cada modelo base
            Columnas: Modelo Base, Peso, Peso Normalizado
        """
        if not self.meta_model_fitted:
            raise ValueError("El modelo Stacking no ha sido entrenado.")
        
        if hasattr(self.meta_model, 'coef_'):
            weights = self.meta_model.coef_[0]
        else:
            weights = np.ones(len(self.base_models)) / len(self.base_models)
            logger.warning("Meta-modelo no tiene coeficientes, usando pesos uniformes")
        
        weights_df = pd.DataFrame({
            'Modelo Base': self._base_model_names,
            'Peso': weights,
            'Peso Normalizado': weights / np.sum(np.abs(weights))
        })
        
        return weights_df.sort_values('Peso', ascending=False, key=abs)
    
    def save_model(self, filepath: str) -> None:
        """
        Guardar el modelo Stacking entrenado
        
        Parámetros:
        -----------
        filepath : str
            Ruta del archivo (recomendado: terminación .pkl o .pickle)
        """
        if not self.meta_model_fitted:
            logger.warning("El modelo no ha sido entrenado, se guardará sin entrenar")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(self, f)
            logger.info(f"Modelo Stacking guardado en: {filepath}")
        except Exception as e:
            logger.error(f"Error guardando el modelo: {str(e)}")
            raise
    
    @staticmethod
    def load_model(filepath: str) -> 'StackingPredictor':
        """
        Cargar un modelo Stacking previamente entrenado
        
        Parámetros:
        -----------
        filepath : str
            Ruta del archivo guardado
        
        Retorna:
        --------
        model : StackingPredictor
            Modelo cargado listo para usar
        """
        try:
            with open(filepath, 'rb') as f:
                model = pickle.load(f)
            logger.info(f"Modelo Stacking cargado desde: {filepath}")
            return model
        except FileNotFoundError:
            logger.error(f"Archivo no encontrado: {filepath}")
            raise
        except Exception as e:
            logger.error(f"Error cargando el modelo: {str(e)}")
            raise
    
    def get_summary(self) -> str:
        """
        Obtener resumen del modelo Stacking
        
        Retorna:
        --------
        summary : str
            Resumen informativo del modelo
        """
        summary = f"""
        {'='*60}
        RESUMEN DEL MODELO STACKING
        {'='*60}
        Estado: {'Entrenado' if self.meta_model_fitted else 'No entrenado'}
        Número de modelos base: {len(self.base_models)}
        Modelos base: {', '.join(self._base_model_names)}
        Meta-modelo: {type(self.meta_model).__name__}
        Folds CV: {self.cv}
        Timestamp entrenamiento: {self.training_timestamp}
        Shape meta-características: {self.meta_features_shape_}
        {'='*60}
        """
        return summary
    
    def __repr__(self) -> str:
        """Representación en string del modelo"""
        return (f"StackingPredictor(n_base_models={len(self.base_models)}, "
                f"meta_model={type(self.meta_model).__name__}, "
                f"fitted={self.meta_model_fitted})")
    
    def __str__(self) -> str:
        """Representación amigable del modelo"""
        return self.get_summary()
