from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import optuna

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import get_scorer
from sklearn.model_selection import StratifiedKFold

from src.models.custom_base import BaseCustomClassifier


class OptunaTunedRandomForest(BaseCustomClassifier):
    """
    RandomForest con ajuste Bayesiano de hiperparámetros mediante Optuna.
    
    Esta clase implementa un clasificador tipo scikit-learn que realiza tuning
    automático de hiperparámetros mediante optimización Bayesiana (TPE) durante
    el entrenamiento. Está diseñado específicamente para datasets tabulares
    clínicos con datos desbalanceados y requisitos de interpretabilidad.
    
    El proceso de entrenamiento incluye:
    1. Validación cruzada estratificada con pruning de trials
    2. Búsqueda Bayesiana de hiperparámetros
    3. Refit del modelo final con los mejores parámetros encontrados
    
    Parameters
    ----------
    n_estimators : int, default=300
        Número de árboles en el bosque. Usado como valor inicial si se
        optimiza este parámetro.
    max_depth : Optional[int], default=None
        Profundidad máxima de los árboles. None significa sin límite.
        Usado como valor inicial si se optimiza este parámetro.
    min_samples_split : int, default=2
        Número mínimo de muestras para dividir un nodo interno.
        Usado como valor inicial si se optimiza este parámetro.
    min_samples_leaf : int, default=1
        Número mínimo de muestras en nodos hoja. Mínimo 1.
    max_features : Any, default="sqrt"
        Número de features a considerar para splits. Puede ser:
        - int: número exacto de features
        - float: fracción de features
        - str: "sqrt", "log2" o None (todas)
    bootstrap : bool, default=True
        Si se usa bootstrap sampling para entrenar árboles.
    class_weight : Any, default=None
        Ponderación de clases. Opciones: None, "balanced", 
        "balanced_subsample" o dict custom.
    n_trials : int, default=50
        Número máximo de trials para Optuna. Cada trial es una 
        configuración de hiperparámetros evaluada con CV.
    cv : int, default=5
        Número de folds para validación cruzada durante tuning.
    scoring : str, default="roc_auc"
        Métrica para optimización. Debe ser un scorer válido de sklearn
        (ej: "roc_auc", "average_precision", "f1").
    random_state : int, default=42
        Semilla para reproducibilidad en RF, Optuna y CV.
    pruner : str, default="median"
        Estrategia de pruning para Optuna. Opciones:
        - "median": MedianPruner (conservador)
        - "asha": SuccessiveHalvingPruner (más agresivo)
        - "none": NopPruner (sin pruning)
    timeout : Optional[int], default=None
        Tiempo máximo (segundos) para la optimización. None = sin límite.
    search_space : Optional[Dict[str, Dict[str, Any]]], default=None
        Espacio de búsqueda custom para hiperparámetros. Si es None,
        usa `_default_search_space()`. Formato:
        {"param_name": {"type": "int|float|categorical", 
                        "low": valor, "high": valor, "step": valor}}
    name : str, default="OptunaTunedRF"
        Nombre identificador para logging y tracking.
        
    Attributes
    ----------
    study_ : Optional[optuna.Study]
        Objeto de estudio de Optuna con historial completo de trials.
    best_params_ : Optional[Dict[str, Any]]
        Diccionario con los mejores hiperparámetros encontrados.
    best_score_ : Optional[float]
        Mejor score promedio obtenido en validación cruzada.
    classes_ : np.ndarray
        Clases únicas encontradas en el target (set after fit).
    _rf : Optional[RandomForestClassifier]
        Instancia del RandomForest entrenado con mejores parámetros.
        
    Notes
    -----
    - El tuning puede ser computacionalmente costoso. Ajusta `n_trials`
      y `cv`.
    - Para datasets muy desbalanceados, considera usar
      `class_weight="balanced"` y métricas como `average_precision`.
    - El pruning ASHA es más veloz pero puede ser más agresivo que Median.
    - Usa `study_.trials_dataframe()` para analizar el historial completo.
    - El modelo final se reentrena con todos los datos después del tuning.
    """

    def __init__(
        self,
        # --- RF "base" (valores iniciales / fallback) ---
        n_estimators: int = 300,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: Any = "sqrt",
        bootstrap: bool = True,
        class_weight: Any = None,
        # --- Optuna / tuning ---
        n_trials: int = 50,
        cv: int = 5,
        scoring: str = "roc_auc",
        random_state: int = 42,
        pruner: str = "median",   # "median" | "asha" | "none"
        timeout: Optional[int] = None,
        # Permite personalizar el espacio de búsqueda
        search_space: Optional[Dict[str, Dict[str, Any]]] = None,
        # nombre para tu framework
        name: str = "OptunaTunedRF",
    ):
        super().__init__(name=name)

        # parámetros RF
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.class_weight = class_weight

        # parámetros Optuna
        self.n_trials = n_trials
        self.cv = cv
        self.scoring = scoring
        self.random_state = random_state
        self.pruner = pruner
        self.timeout = timeout
        self.search_space = search_space

        # artefactos del entrenamiento
        self._rf: Optional[RandomForestClassifier] = None
        self.study_: Optional[optuna.Study] = None
        self.best_params_: Optional[Dict[str, Any]] = None
        self.best_score_: Optional[float] = None

    # --------- sklearn compatibility ----------
    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """
        Obtiene parámetros de inicialización para compatibilidad con sklearn.
        
        Método requerido para pipelines y GridSearchCV de scikit-learn.
        Devuelve un diccionario con todos los parámetros pasados al __init__.
        
        Parameters
        ----------
        deep : bool, default=True
            Si True, devolverá parámetros de sub-estimadores (no aplica aquí).
            
        Returns
        -------
        Dict[str, Any]
            Diccionario con nombre y valor de cada parámetro.
        """
        # Importante: devolver los args del __init__
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "max_features": self.max_features,
            "bootstrap": self.bootstrap,
            "class_weight": self.class_weight,
            "n_trials": self.n_trials,
            "cv": self.cv,
            "scoring": self.scoring,
            "random_state": self.random_state,
            "pruner": self.pruner,
            "timeout": self.timeout,
            "search_space": self.search_space,
            "name": self.name,
        }

    def set_params(self, **params) -> "OptunaTunedRandomForest":
        """
        Establece parámetros para compatibilidad con sklearn.
        
        Método requerido para pipelines y GridSearchCV de scikit-learn.
        Permite modificar parámetros después de la instanciación.
        
        Parameters
        ----------
        **params : dict
            Parámetros a establecer con sus nuevos valores.
            
        Returns
        -------
        OptunaTunedRandomForest
            La instancia misma (patrón fluent).
        """
        for k, v in params.items():
            setattr(self, k, v)
        return self

    # --------- core ML ----------
    def _default_search_space(self) -> Dict[str, Dict[str, Any]]:
        """
        Define el espacio de búsqueda por defecto para hiperparámetros.
        
        Diseñado para problemas clínicos tabulares con datasets
        moderadamente dimensionados y potencial desbalance de clases.
        
        Returns
        -------
        Dict[str, Dict[str, Any]]
            Diccionario anidado especificando tipo y rangos para cada hiperparámetro.
            Formato: {"param": {"type": "int|float|categorical", "low": val, "high": val}}
            
        Notes
        -----
        - `max_depth` incluye None para permitir crecimiento ilimitado
        - `max_features` incluye fracciones para datasets con muchas columnas
        - `min_samples` relativamente bajos para datos clínicos típicos
        """
        
        return {
            "n_estimators": {"type": "int", "low": 200, "high": 900, "step": 50},
            "max_depth": {"type": "categorical", "choices": [None] + list(range(2, 31))},
            "min_samples_split": {"type": "int", "low": 2, "high": 20},
            "min_samples_leaf": {"type": "int", "low": 1, "high": 10},
            "max_features": {"type": "categorical", "choices": ["sqrt", "log2", None, 0.5, 0.8]},
            "bootstrap": {"type": "categorical", "choices": [True, False]},
            "class_weight": {"type": "categorical", "choices": [None, "balanced", "balanced_subsample"]},
        }

    def _suggest(self, trial: optuna.Trial, name: str, spec: Dict[str, Any]) -> Any:
        """
        Sugiere un valor de hiperparámetro para un trial de Optuna.
        
        Método helper que interpreta la especificación del espacio de búsqueda
        y llama al método correspondiente de `optuna.Trial`.
        
        Parameters
        ----------
        trial : optuna.Trial
            Objeto trial activo de Optuna.
        name : str
            Nombre del hiperparámetro a suggestionar.
        spec : Dict[str, Any]
            Especificación del espacio de búsqueda para este parámetro.
            Debe contener clave "type" y rangos apropiados.
            
        Returns
        -------
        Any
            Valor suggestionado para el hiperparámetro.
            
        Raises
        ------
        ValueError
            Si `spec["type"]` no es "int", "float" o "categorical".
        """
        t = spec.get("type", "categorical")
        if t == "int":
            return trial.suggest_int(
                name,
                int(spec["low"]),
                int(spec["high"]),
                step=int(spec.get("step", 1)),
                log=bool(spec.get("log", False)),
            )
        if t == "float":
            return trial.suggest_float(
                name,
                float(spec["low"]),
                float(spec["high"]),
                step=spec.get("step", None),
                log=bool(spec.get("log", False)),
            )
        if t == "categorical":
            return trial.suggest_categorical(name, list(spec["choices"]))
        raise ValueError(f"Tipo no soportado: {t} (param={name})")

    def _make_pruner(self):
        """
        Instancia el pruner de Optuna según configuración.
        
        El pruning permite terminar trials prematuramente si no muestran
        potencial de mejora, ahorrando tiempo computacional.
        
        Returns
        -------
        optuna.pruners.BasePruner
            Instancia del pruner configurado.
            
        Notes
        -----
        - "median": Pruner conservador, ideal para datasets pequeños
        - "asha": Pruner agresivo, más rápido para datasets grandes
        - "none": Desactiva pruning (útil para debugging)
        """
        
        if self.pruner == "median":
            return optuna.pruners.MedianPruner()
        if self.pruner in ("asha", "successive_halving"):
            return optuna.pruners.SuccessiveHalvingPruner()
        return optuna.pruners.NopPruner()

    def fit(self, X, y):
        """
        Entrena el modelo con tuning Bayesiano de hiperparámetros.
        
        Ejecuta los siguientes pasos:
        1. Configura espacio de búsqueda (custom o default)
        2. Crea estudio Optuna con sampler TPE y pruner
        3. Ejecuta `n_trials` de validación cruzada con pruning
        4. Guarda mejores parámetros y score
        5. Refitea modelo final con todos los datos
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Features de entrenamiento. La matriz será indexada como array
            NumPy durante CV, así que asegúrate que sea indexable.
        y : array-like of shape (n_samples,)
            Target binario (0/1 o -1/1). Se espera desbalanceo clínico típico.
            
        Returns
        -------
        self : OptunaTunedRandomForest
            La instancia entrenada con todos los artefactos disponibles.
            
        Raises
        ------
        RuntimeError
            Si `X` o `y` tienen formatos inconsistentes o no indexables.
            
        Notes
        -----
        - La métrica de optimización está definida en `self.scoring`
        - Trials con bajo rendimiento son pruned para eficiencia
        - `self.study_` contiene el historial completo para análisis
        - El modelo final usa `n_jobs=-1` para máxima velocidad
        """
        space = self.search_space or self._default_search_space()

        scorer = get_scorer(self.scoring)

        cv = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)

        sampler = optuna.samplers.TPESampler(seed=self.random_state)  # TPE
        pruner = self._make_pruner()

        study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)

        def objective(trial: optuna.Trial) -> float:
            params = {p: self._suggest(trial, p, spec) for p, spec in space.items()}

            fold_scores = []
            for fold_idx, (tr_idx, va_idx) in enumerate(cv.split(X, y)):
                rf = RandomForestClassifier(
                    **params,
                    random_state=self.random_state,
                    n_jobs=-1,
                )
                rf.fit(X[tr_idx], y[tr_idx])

                s = scorer(rf, X[va_idx], y[va_idx])
                fold_scores.append(float(s))

                # Reportar intermedio y permitir pruning
                trial.report(float(np.mean(fold_scores)), step=fold_idx)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            return float(np.mean(fold_scores))

        study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout)

        self.study_ = study
        self.best_params_ = dict(study.best_params)
        self.best_score_ = float(study.best_value)

        # Refit final con los mejores params
        self._rf = RandomForestClassifier(
            **self.best_params_,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self._rf.fit(X, y)
        self.classes_ = self._rf.classes_
        return self

    def predict(self, X):
        """
        Predice etiquetas de clase para muestras en `X`.
        
        Usa el RandomForest entrenado con mejores parámetros.
        Requiere que `fit` haya sido llamado previamente.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Features para predicción. Debe tener mismo número de columnas
            que `X` usado en entrenamiento.
            
        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Etiquetas predichas (0/1) para cada muestra.
            
        Raises
        ------
        RuntimeError
            Si el modelo no ha sido entrenado (`fit` no llamado).
        """
        if self._rf is None:
            raise RuntimeError("El modelo no está entrenado. Llama a fit() primero.")
        return self._rf.predict(X)

    def predict_proba(self, X):
        """
        Predice probabilidades de clase para muestras en `X`.
        
        Devuelve probabilidades estimadas para cada clase (0 y 1).
        Requiere que `fit` haya sido llamado previamente.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Features para predicción. Debe tener mismo número de columnas
            que `X` usado en entrenamiento.
            
        Returns
        -------
        y_proba : ndarray of shape (n_samples, 2)
            Probabilidades predichas para clases 0 y 1. Columna 1 contiene
            probabilidad de evento (clase positiva).
            
        Raises
        ------
        RuntimeError
            Si el modelo no ha sido entrenado (`fit` no llamado).
        """
        if self._rf is None:
            raise RuntimeError("El modelo no está entrenado. Llama a fit() primero.")
        return self._rf.predict_proba(X)