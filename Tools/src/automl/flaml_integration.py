"""FLAML Integration Module.

Provides cross-platform AutoML using FLAML (Fast and Lightweight AutoML).
This is the recommended option for Windows users as auto-sklearn
requires Linux.

FLAML advantages:
- Cross-platform (Windows, Linux, macOS)
- Faster than auto-sklearn for many tasks
- Lower memory requirements
- Supports more estimators out of the box

Available Estimators:
    - lgbm: LightGBM (gradient boosting)
    - xgboost: XGBoost 
    - xgb_limitdepth: XGBoost with limited depth
    - catboost: CatBoost (gradient boosting)
    - rf: Random Forest
    - extra_tree: Extra Trees
    - histgb: Histogram-based Gradient Boosting (sklearn)
    - kneighbor: K-Nearest Neighbors
    - lrl1: L1 Regularized Logistic Regression (Lasso)
    - lrl2: L2 Regularized Logistic Regression (Ridge)
    - svc: Support Vector Classifier
    - sgd: Stochastic Gradient Descent Classifier
    - nb: Naive Bayes (Gaussian)
    - mlp: Multi-Layer Perceptron (Neural Network)
    - elasticnet: Elastic Net (for regression)
    - lassolars: Lasso with LARS (for regression)
"""
from __future__ import annotations

import logging
import sys
import time
import io
import threading
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from contextlib import redirect_stdout, redirect_stderr

import numpy as np
import pandas as pd

from ..models.custom_base import BaseCustomClassifier, BaseCustomRegressor
from ..config import RANDOM_SEED
from .config import AutoMLConfig, AutoMLPreset

logger = logging.getLogger(__name__)


# ============================================================================
# FLAML Estimator Definitions - Complete list of all available estimators
# ============================================================================

# All available estimators for classification
FLAML_CLASSIFICATION_ESTIMATORS = [
    "lgbm",           # LightGBM - fast gradient boosting
    "xgboost",        # XGBoost - gradient boosting
    "xgb_limitdepth", # XGBoost with limited depth
    "catboost",       # CatBoost - gradient boosting with categorical support
    "rf",             # Random Forest
    "extra_tree",     # Extra Trees (Extremely Randomized Trees)
    "histgb",         # Histogram-based Gradient Boosting (sklearn)
    "kneighbor",      # K-Nearest Neighbors
    "lrl1",           # L1 Logistic Regression (Lasso)
    "lrl2",           # L2 Logistic Regression (Ridge)
    "svc",            # Support Vector Classifier
    "sgd",            # Stochastic Gradient Descent
]

# All available estimators for regression
FLAML_REGRESSION_ESTIMATORS = [
    "lgbm",           # LightGBM
    "xgboost",        # XGBoost
    "xgb_limitdepth", # XGBoost with limited depth
    "catboost",       # CatBoost
    "rf",             # Random Forest
    "extra_tree",     # Extra Trees
    "histgb",         # Histogram-based Gradient Boosting
    "kneighbor",      # K-Nearest Neighbors
    "elasticnet",     # Elastic Net
    "lassolars",      # Lasso with LARS
]

# Union of all estimators
FLAML_ESTIMATORS = list(set(FLAML_CLASSIFICATION_ESTIMATORS + FLAML_REGRESSION_ESTIMATORS))

# Detailed descriptions for UI
FLAML_ESTIMATOR_DESCRIPTIONS = {
    "lgbm": {
        "name": "LightGBM",
        "type": "Gradient Boosting",
        "description": "Fast, distributed, high-performance gradient boosting framework",
        "pros": ["Very fast training", "Low memory usage", "Handles large datasets well"],
        "cons": ["May overfit on small datasets"],
        "tasks": ["classification", "regression"],
    },
    "xgboost": {
        "name": "XGBoost",
        "type": "Gradient Boosting",
        "description": "Optimized distributed gradient boosting library",
        "pros": ["State-of-the-art performance", "Regularization built-in", "Handles missing values"],
        "cons": ["Slower than LightGBM", "More memory usage"],
        "tasks": ["classification", "regression"],
    },
    "xgb_limitdepth": {
        "name": "XGBoost (Limited Depth)",
        "type": "Gradient Boosting",
        "description": "XGBoost with tree depth constraints for faster training",
        "pros": ["Faster than regular XGBoost", "Less prone to overfitting"],
        "cons": ["May underfit complex patterns"],
        "tasks": ["classification", "regression"],
    },
    "catboost": {
        "name": "CatBoost",
        "type": "Gradient Boosting",
        "description": "Gradient boosting with excellent categorical feature handling",
        "pros": ["Best categorical support", "Symmetric trees", "Robust to overfitting"],
        "cons": ["Slower training", "Larger model size"],
        "tasks": ["classification", "regression"],
    },
    "rf": {
        "name": "Random Forest",
        "type": "Ensemble (Bagging)",
        "description": "Ensemble of decision trees using bagging",
        "pros": ["Robust", "Handles non-linear relationships", "Feature importance"],
        "cons": ["Large model size", "Slower inference"],
        "tasks": ["classification", "regression"],
    },
    "extra_tree": {
        "name": "Extra Trees",
        "type": "Ensemble (Bagging)",
        "description": "Extremely randomized trees - more random than Random Forest",
        "pros": ["Faster training than RF", "Reduced variance", "Good for high-dim data"],
        "cons": ["Can be less accurate than RF"],
        "tasks": ["classification", "regression"],
    },
    "histgb": {
        "name": "Histogram Gradient Boosting",
        "type": "Gradient Boosting",
        "description": "Scikit-learn's histogram-based gradient boosting",
        "pros": ["Fast training", "Native sklearn", "Handles missing values"],
        "cons": ["Less tuned than LightGBM/XGBoost"],
        "tasks": ["classification", "regression"],
    },
    "kneighbor": {
        "name": "K-Nearest Neighbors",
        "type": "Instance-Based",
        "description": "Simple, non-parametric method based on nearest neighbors",
        "pros": ["Simple to understand", "No training phase", "Non-linear boundaries"],
        "cons": ["Slow prediction", "Sensitive to scale", "Curse of dimensionality"],
        "tasks": ["classification", "regression"],
    },
    "lrl1": {
        "name": "Logistic Regression (L1)",
        "type": "Linear Model",
        "description": "Logistic regression with L1 (Lasso) regularization",
        "pros": ["Feature selection", "Interpretable", "Fast"],
        "cons": ["Only linear boundaries", "May underfit complex data"],
        "tasks": ["classification"],
    },
    "lrl2": {
        "name": "Logistic Regression (L2)",
        "type": "Linear Model",
        "description": "Logistic regression with L2 (Ridge) regularization",
        "pros": ["Stable coefficients", "Interpretable", "Fast"],
        "cons": ["Only linear boundaries", "May underfit complex data"],
        "tasks": ["classification"],
    },
    "svc": {
        "name": "Support Vector Classifier",
        "type": "Support Vector Machine",
        "description": "Support Vector Machine with RBF kernel",
        "pros": ["Effective in high dimensions", "Memory efficient"],
        "cons": ["Slow for large datasets", "Sensitive to scaling"],
        "tasks": ["classification"],
    },
    "sgd": {
        "name": "SGD Classifier",
        "type": "Linear Model",
        "description": "Stochastic Gradient Descent classifier",
        "pros": ["Very fast", "Scales to large datasets", "Online learning"],
        "cons": ["Requires tuning", "Sensitive to scaling"],
        "tasks": ["classification"],
    },
    "elasticnet": {
        "name": "Elastic Net",
        "type": "Linear Model",
        "description": "Linear regression with combined L1 and L2 regularization",
        "pros": ["Feature selection", "Stable", "Handles correlated features"],
        "cons": ["Only linear relationships"],
        "tasks": ["regression"],
    },
    "lassolars": {
        "name": "Lasso LARS",
        "type": "Linear Model",
        "description": "Lasso using Least Angle Regression",
        "pros": ["Feature selection", "Fast for sparse solutions"],
        "cons": ["Only linear relationships"],
        "tasks": ["regression"],
    },
}

# Preset configurations for different use cases
FLAML_PRESETS = {
    "quick": {
        "estimator_list": ["lgbm", "rf", "xgboost"],
        "description": "Quick exploration with best performers",
    },
    "balanced": {
        "estimator_list": ["lgbm", "xgboost", "catboost", "rf", "extra_tree", "histgb"],
        "description": "Balanced mix of gradient boosting and ensemble methods",
    },
    "comprehensive": {
        "estimator_list": FLAML_CLASSIFICATION_ESTIMATORS,
        "description": "All available estimators for thorough search",
    },
    "gradient_boosting": {
        "estimator_list": ["lgbm", "xgboost", "xgb_limitdepth", "catboost", "histgb"],
        "description": "Focus on gradient boosting methods",
    },
    "ensemble": {
        "estimator_list": ["rf", "extra_tree", "lgbm", "xgboost", "catboost"],
        "description": "Ensemble methods (trees + boosting)",
    },
    "linear": {
        "estimator_list": ["lrl1", "lrl2", "sgd"],
        "description": "Linear models for interpretability",
    },
    "all": {
        "estimator_list": FLAML_CLASSIFICATION_ESTIMATORS,
        "description": "All estimators - maximum coverage",
    },
}


class FLAMLProgressTracker:
    """Tracks FLAML progress and provides real-time updates via callback.
    
    Uses a background thread to periodically report progress even when
    FLAML is not emitting logs. Also monitors FLAML's log file for real updates.
    """
    
    def __init__(
        self,
        callback: Optional[Callable[[str, float], None]] = None,
        time_budget: int = 3600,
        update_interval: float = 2.0,  # Update every 2 seconds
        log_file: Optional[str] = None,
    ):
        self.callback = callback
        self.time_budget = time_budget
        self.update_interval = update_interval
        self.log_file = log_file
        
        self.start_time: Optional[float] = None
        self.logs: List[str] = []
        self.lock = threading.Lock()
        
        # Progress tracking
        self.models_evaluated = 0
        self.best_loss = float('inf')
        self.best_estimator: Optional[str] = None
        self.current_estimator: Optional[str] = None
        self.is_running = False
        self._thread: Optional[threading.Thread] = None
        self._last_log_pos = 0
    
    def start(self) -> None:
        """Start the progress tracker."""
        self.start_time = time.time()
        self.is_running = True
        self._thread = threading.Thread(target=self._progress_loop, daemon=True)
        self._thread.start()
        self._log("🚀 Iniciando búsqueda AutoML...")
    
    def stop(self) -> None:
        """Stop the progress tracker."""
        self.is_running = False
        if self._thread:
            self._thread.join(timeout=1.0)
    
    def _read_flaml_log(self) -> None:
        """Read updates from FLAML's log file if available."""
        if not self.log_file:
            return
        
        try:
            from pathlib import Path
            log_path = Path(self.log_file)
            if log_path.exists():
                with open(log_path, 'r') as f:
                    f.seek(self._last_log_pos)
                    new_content = f.read()
                    self._last_log_pos = f.tell()
                    
                    if new_content.strip():
                        # Parse FLAML log entries
                        for line in new_content.strip().split('\n'):
                            self._parse_log_line(line)
        except Exception:
            pass
    
    def _parse_log_line(self, line: str) -> None:
        """Parse a FLAML log line and extract info."""
        try:
            import json
            data = json.loads(line)
            
            estimator = data.get('learner', 'unknown')
            loss = data.get('validation_loss')
            
            if loss is not None:
                with self.lock:
                    self.models_evaluated = data.get('record_id', self.models_evaluated) + 1
                    
                    if loss < self.best_loss:
                        self.best_loss = loss
                        self.best_estimator = estimator
                        score = 1 - loss if loss >= 0 else -loss
                        self._log(f"⭐ Nuevo mejor: {estimator} (score={score:.4f})")
                    else:
                        self._log(f"🧪 Evaluado: {estimator} (#{self.models_evaluated})")
        except json.JSONDecodeError:
            pass
        except Exception:
            pass
    
    def _progress_loop(self) -> None:
        """Background thread that reports progress periodically."""
        last_models = 0
        animation_frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        frame_idx = 0
        
        while self.is_running:
            time.sleep(self.update_interval)
            if not self.is_running:
                break
            
            # Try to read FLAML log file
            self._read_flaml_log()
            
            with self.lock:
                elapsed = time.time() - self.start_time if self.start_time else 0
                progress = min(elapsed / self.time_budget, 0.99)
                remaining = max(0, self.time_budget - elapsed)
                
                mins_elapsed = int(elapsed // 60)
                secs_elapsed = int(elapsed % 60)
                mins_remaining = int(remaining // 60)
                secs_remaining = int(remaining % 60)
                
                # Animation frame
                spinner = animation_frames[frame_idx % len(animation_frames)]
                frame_idx += 1
                
                # Build status message
                if self.models_evaluated > 0:
                    if self.best_estimator and self.best_loss < float('inf'):
                        score = 1 - self.best_loss if self.best_loss >= 0 else -self.best_loss
                        status = (
                            f"{spinner} {mins_elapsed:02d}:{secs_elapsed:02d} | "
                            f"🔍 {self.models_evaluated} modelos | "
                            f"🏆 {self.best_estimator}: {score:.4f}"
                        )
                    else:
                        status = (
                            f"{spinner} {mins_elapsed:02d}:{secs_elapsed:02d} | "
                            f"🔍 {self.models_evaluated} modelos evaluados"
                        )
                    
                    if self.current_estimator:
                        status += f" | 🔄 {self.current_estimator}"
                else:
                    status = (
                        f"{spinner} {mins_elapsed:02d}:{secs_elapsed:02d} | "
                        f"🔄 Inicializando y evaluando estimadores..."
                    )
                
                # Add remaining time info
                status += f" | ⏳ {mins_remaining:02d}:{secs_remaining:02d}"
                
                # Only log status changes periodically
                if self.models_evaluated != last_models or frame_idx % 5 == 0:
                    if self.models_evaluated != last_models:
                        last_models = self.models_evaluated
                
                if self.callback:
                    self.callback(status, progress)
    
    def update(
        self,
        estimator: Optional[str] = None,
        loss: Optional[float] = None,
        is_best: bool = False,
    ) -> None:
        """Update progress with new trial info."""
        with self.lock:
            self.models_evaluated += 1
            self.current_estimator = estimator
            
            if loss is not None and loss < self.best_loss:
                self.best_loss = loss
                self.best_estimator = estimator
                score = 1 - loss if loss >= 0 else -loss
                self._log(f"⭐ Nuevo mejor modelo: {estimator} (score={score:.4f})")
            elif estimator:
                self._log(f"🧪 Evaluando: {estimator} (modelo #{self.models_evaluated})")
    
    def _log(self, message: str) -> None:
        """Add a log entry."""
        timestamp = time.strftime("%H:%M:%S")
        entry = f"[{timestamp}] {message}"
        self.logs.append(entry)
    
    def get_logs(self) -> List[str]:
        """Get all log entries."""
        with self.lock:
            return self.logs.copy()



def is_flaml_available() -> bool:
    """Check if FLAML is available.
    
    Returns:
        True if FLAML can be imported
    """
    try:
        import flaml
        return True
    except ImportError:
        return False


class FLAMLClassifier(BaseCustomClassifier):
    """FLAML-based AutoML Classifier.
    
    A cross-platform AutoML classifier using Microsoft's FLAML library.
    Works on Windows, Linux, and macOS.
    
    FLAML is faster and more lightweight than auto-sklearn while
    achieving competitive accuracy.
    
    Available Estimators:
        - lgbm: LightGBM (gradient boosting)
        - xgboost: XGBoost 
        - xgb_limitdepth: XGBoost with limited depth
        - catboost: CatBoost (gradient boosting)
        - rf: Random Forest
        - extra_tree: Extra Trees
        - histgb: Histogram-based Gradient Boosting
        - kneighbor: K-Nearest Neighbors
        - lrl1: L1 Logistic Regression
        - lrl2: L2 Logistic Regression
        - svc: Support Vector Classifier
        - sgd: SGD Classifier
    
    Attributes:
        config: AutoML configuration
        automl_: Fitted FLAML AutoML object
        best_estimator_: Name of best estimator
        best_config_: Best hyperparameter configuration
        best_model_: Best fitted model
    
    Example:
        >>> clf = FLAMLClassifier(time_budget=300)  # 5 minutes
        >>> clf.fit(X_train, y_train)
        >>> proba = clf.predict_proba(X_test)
        
        >>> # Use all available estimators
        >>> clf = FLAMLClassifier(
        ...     time_budget=3600,
        ...     estimator_list=FLAML_CLASSIFICATION_ESTIMATORS
        ... )
    """
    
    def __init__(
        self,
        time_budget: int = 3600,
        metric: str = "roc_auc",
        estimator_list: Optional[List[str]] = None,
        n_jobs: int = -1,
        ensemble: bool = True,
        max_iter: Optional[int] = None,
        early_stop: bool = True,
        random_state: int = RANDOM_SEED,
        verbose: int = 2,  # Default verbose = 2 for detailed output
        log_file_name: Optional[str] = None,
        name: str = "FLAMLClassifier",
        progress_callback: Optional[Callable[[str, float], None]] = None,
        use_all_estimators: bool = False,
        **kwargs
    ):
        """
        Initialize FLAML Classifier.
        
        Args:
            time_budget: Time budget in seconds
            metric: Optimization metric (roc_auc, accuracy, f1, log_loss)
            estimator_list: List of estimators to try. If None, uses balanced set.
                Options: lgbm, xgboost, xgb_limitdepth, catboost, rf, extra_tree,
                         histgb, kneighbor, lrl1, lrl2, svc, sgd
            n_jobs: Number of parallel jobs
            ensemble: Whether to use ensemble
            max_iter: Maximum iterations (None = unlimited)
            early_stop: Enable early stopping
            random_state: Random seed
            verbose: Verbosity level (0=silent, 1=progress, 2=detailed, 3=debug)
            log_file_name: Optional log file path
            name: Model name
            progress_callback: Optional progress callback(message, progress)
            use_all_estimators: If True, uses all available estimators
            **kwargs: Additional FLAML arguments
        """
        super().__init__(name=name)
        
        self.time_budget = time_budget
        self.metric = metric
        self.n_jobs = n_jobs
        self.ensemble = ensemble
        self.max_iter = max_iter
        self.early_stop = early_stop
        self.random_state = random_state
        self.verbose = verbose
        self.log_file_name = log_file_name
        self.extra_kwargs = kwargs
        self.progress_callback = progress_callback
        
        # Set estimator list
        if use_all_estimators:
            self.estimator_list = FLAML_CLASSIFICATION_ESTIMATORS.copy()
        elif estimator_list is not None:
            self.estimator_list = estimator_list
        else:
            # Default balanced list for good performance
            self.estimator_list = ["lgbm", "xgboost", "rf", "extra_tree", "catboost", "histgb"]
        
        # Runtime attributes
        self.automl_: Optional[Any] = None
        self.best_estimator_: Optional[str] = None
        self.best_config_: Optional[Dict[str, Any]] = None
        self.best_model_: Optional[Any] = None
        self.best_loss_: Optional[float] = None
        self.fit_time_: float = 0.0
        self.feature_importances_: Optional[np.ndarray] = None
        self.training_logs_: List[str] = []
    
    def fit(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series,
        **kwargs
    ) -> "FLAMLClassifier":
        """
        Fit the FLAML classifier.
        
        Args:
            X: Training features
            y: Training labels
            **kwargs: Additional FLAML fit arguments
            
        Returns:
            Self (fitted classifier)
        """
        if not is_flaml_available():
            raise ImportError(
                "FLAML is not installed. Install with: pip install flaml[automl]"
            )
        
        from flaml import AutoML
        
        self._validate_input(X, training=True)
        y = self._validate_targets(y, training=True)
        
        # Convert to arrays/DataFrames as FLAML prefers
        if isinstance(X, np.ndarray):
            X_train = pd.DataFrame(X)
        else:
            X_train = X.copy()
        
        y_train = y if isinstance(y, np.ndarray) else y.values
        
        start_time = time.time()
        
        # Create temp log file for tracking if not specified
        import tempfile
        if self.log_file_name:
            log_file_path = self.log_file_name
        else:
            log_file_path = tempfile.mktemp(suffix='_flaml.log')
        
        # Initialize progress tracker with background thread
        progress_tracker = FLAMLProgressTracker(
            callback=self.progress_callback,
            time_budget=self.time_budget,
            update_interval=2.0,  # Update every 2 seconds
            log_file=log_file_path,
        )
        
        if self.progress_callback:
            self.progress_callback("🚀 Iniciando FLAML AutoML...", 0.0)
        
        # Build FLAML settings
        flaml_settings = {
            "time_budget": self.time_budget,
            "metric": self.metric,
            "task": "classification",
            "n_jobs": self.n_jobs,
            "ensemble": self.ensemble,
            "seed": self.random_state,
            "verbose": self.verbose,
            "early_stop": self.early_stop,
        }
        
        if self.estimator_list is not None:
            flaml_settings["estimator_list"] = self.estimator_list
        
        if self.max_iter is not None:
            flaml_settings["max_iter"] = self.max_iter
        
        # Always use log file for tracking progress
        flaml_settings["log_file_name"] = log_file_path
        
        # Add extra kwargs
        flaml_settings.update(self.extra_kwargs)
        flaml_settings.update(kwargs)
        
        # Initialize and fit with progress tracking
        self.automl_ = AutoML()
        
        # Start progress tracker thread
        progress_tracker.start()
        
        try:
            self.automl_.fit(X_train, y_train, **flaml_settings)
        finally:
            # Stop progress tracker
            progress_tracker.stop()
        
        # Store results
        self.best_estimator_ = self.automl_.best_estimator
        self.best_config_ = self.automl_.best_config
        self.best_model_ = self.automl_.model
        self.best_loss_ = self.automl_.best_loss
        self.fit_time_ = time.time() - start_time
        self.training_logs_ = progress_tracker.get_logs()
        
        # Add final summary to logs
        score = -self.best_loss_ if self.best_loss_ < 0 else 1 - self.best_loss_
        self.training_logs_.append(
            f"[{time.strftime('%H:%M:%S')}] ✅ Entrenamiento completado!"
        )
        self.training_logs_.append(
            f"[{time.strftime('%H:%M:%S')}] 🏆 Mejor modelo: {self.best_estimator_} (score={score:.4f})"
        )
        self.training_logs_.append(
            f"[{time.strftime('%H:%M:%S')}] ⏱️ Tiempo total: {self.fit_time_:.1f}s"
        )
        self.training_logs_.append(
            f"[{time.strftime('%H:%M:%S')}] 🔍 Modelos evaluados: {progress_tracker.models_evaluated}"
        )
        
        # Extract feature importances if available
        self._extract_feature_importances()
        
        self.is_fitted_ = True
        
        if self.progress_callback:
            self.progress_callback(
                f"✅ FLAML completado: {self.best_estimator_} "
                f"(score={score:.4f}, time={self.fit_time_:.1f}s)",
                1.0
            )
        
        logger.info(
            f"FLAML training completed: best={self.best_estimator_}, "
            f"loss={self.best_loss_:.4f}, time={self.fit_time_:.1f}s"
        )
        
        return self
    
    def get_training_logs(self) -> List[str]:
        """Get the training logs captured during fit.
        
        Returns:
            List of log messages
        """
        return self.training_logs_.copy()
    
    def _extract_feature_importances(self) -> None:
        """Extract feature importances from best model."""
        try:
            if hasattr(self.best_model_, 'feature_importances_'):
                self.feature_importances_ = self.best_model_.feature_importances_
            elif hasattr(self.best_model_, 'coef_'):
                self.feature_importances_ = np.abs(self.best_model_.coef_).flatten()
        except Exception as e:
            logger.debug(f"Could not extract feature importances: {e}")
    
    def predict_proba(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Predict class probabilities."""
        self._validate_input(X, training=False)
        
        if isinstance(X, np.ndarray):
            X_pred = pd.DataFrame(X)
        else:
            X_pred = X
        
        return self.automl_.predict_proba(X_pred)
    
    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Predict class labels."""
        self._validate_input(X, training=False)
        
        if isinstance(X, np.ndarray):
            X_pred = pd.DataFrame(X)
        else:
            X_pred = X
        
        return self.automl_.predict(X_pred)
    
    def get_best_model(self) -> Any:
        """Get the best fitted model."""
        return self.best_model_
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """Get feature importances if available."""
        return self.feature_importances_
    
    def get_leaderboard(self) -> pd.DataFrame:
        """Get leaderboard of evaluated models.
        
        Returns:
            DataFrame with estimator performance info
        """
        if self.automl_ is None:
            return pd.DataFrame()
        
        try:
            rows = []
            
            # Best config per estimator
            if hasattr(self.automl_, 'best_config_per_estimator'):
                for est, config in self.automl_.best_config_per_estimator.items():
                    if config is not None:
                        # Try to get loss for this estimator
                        loss = None
                        if hasattr(self.automl_, 'best_loss_per_estimator'):
                            loss = self.automl_.best_loss_per_estimator.get(est)
                        
                        rows.append({
                            'estimator': est,
                            'loss': loss if loss else 'N/A',
                            'score': -loss if loss else 'N/A',
                            'config': str(config)[:100],
                            'is_best': est == self.best_estimator_,
                        })
            
            # Sort by loss
            df = pd.DataFrame(rows)
            if len(df) > 0 and 'loss' in df.columns:
                df = df.sort_values('loss', ascending=True)
            
            return df
        except Exception as e:
            logger.warning(f"Could not generate leaderboard: {e}")
            return pd.DataFrame()
    
    def get_search_history(self) -> pd.DataFrame:
        """Get the search history as a DataFrame.
        
        Returns:
            DataFrame with columns: estimator, config, loss, time
        """
        return self.get_leaderboard()
    
    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get parameters for this estimator."""
        return {
            "time_budget": self.time_budget,
            "metric": self.metric,
            "estimator_list": self.estimator_list,
            "n_jobs": self.n_jobs,
            "ensemble": self.ensemble,
            "max_iter": self.max_iter,
            "early_stop": self.early_stop,
            "random_state": self.random_state,
            "verbose": self.verbose,
            "name": self.name,
        }
    
    def set_params(self, **params) -> "FLAMLClassifier":
        """Set parameters for this estimator."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get extended metadata."""
        metadata = super().get_metadata()
        metadata.update({
            'backend': 'flaml',
            'best_estimator': self.best_estimator_,
            'best_loss': self.best_loss_,
            'fit_time_seconds': self.fit_time_,
            'time_budget': self.time_budget,
            'metric': self.metric,
            'n_logs': len(self.training_logs_),
        })
        return metadata


class FLAMLRegressor(BaseCustomRegressor):
    """FLAML-based AutoML Regressor.
    
    Cross-platform AutoML regressor using FLAML.
    
    Available Estimators:
        - lgbm: LightGBM
        - xgboost: XGBoost
        - xgb_limitdepth: XGBoost with limited depth
        - catboost: CatBoost
        - rf: Random Forest
        - extra_tree: Extra Trees
        - histgb: Histogram-based Gradient Boosting
        - kneighbor: K-Nearest Neighbors
        - elasticnet: Elastic Net
        - lassolars: Lasso LARS
    """
    
    def __init__(
        self,
        time_budget: int = 3600,
        metric: str = "r2",
        estimator_list: Optional[List[str]] = None,
        n_jobs: int = -1,
        ensemble: bool = True,
        random_state: int = RANDOM_SEED,
        verbose: int = 2,  # Default verbose = 2
        log_file_name: Optional[str] = None,
        name: str = "FLAMLRegressor",
        progress_callback: Optional[Callable[[str, float], None]] = None,
        use_all_estimators: bool = False,
        **kwargs
    ):
        """Initialize FLAML Regressor."""
        super().__init__(name=name)
        
        self.time_budget = time_budget
        self.metric = metric
        self.n_jobs = n_jobs
        self.ensemble = ensemble
        self.random_state = random_state
        self.verbose = verbose
        self.log_file_name = log_file_name
        self.extra_kwargs = kwargs
        self.progress_callback = progress_callback
        
        # Set estimator list
        if use_all_estimators:
            self.estimator_list = FLAML_REGRESSION_ESTIMATORS.copy()
        elif estimator_list is not None:
            self.estimator_list = estimator_list
        else:
            # Default balanced list
            self.estimator_list = ["lgbm", "xgboost", "rf", "extra_tree", "catboost", "histgb"]
        
        self.automl_: Optional[Any] = None
        self.best_estimator_: Optional[str] = None
        self.best_model_: Optional[Any] = None
        self.best_loss_: Optional[float] = None
        self.fit_time_: float = 0.0
        self.training_logs_: List[str] = []
    
    def fit(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series,
        **kwargs
    ) -> "FLAMLRegressor":
        """Fit the FLAML regressor."""
        if not is_flaml_available():
            raise ImportError("FLAML is not installed")
        
        from flaml import AutoML
        
        self._validate_input(X, training=True)
        
        if isinstance(X, np.ndarray):
            X_train = pd.DataFrame(X)
        else:
            X_train = X.copy()
        
        y_train = y.values if isinstance(y, pd.Series) else y
        
        start_time = time.time()
        
        # Create temp log file for tracking if not specified
        import tempfile
        if self.log_file_name:
            log_file_path = self.log_file_name
        else:
            log_file_path = tempfile.mktemp(suffix='_flaml_reg.log')
        
        # Initialize progress tracker with background thread
        progress_tracker = FLAMLProgressTracker(
            callback=self.progress_callback,
            time_budget=self.time_budget,
            update_interval=2.0,
            log_file=log_file_path,
        )
        
        if self.progress_callback:
            self.progress_callback("🚀 Iniciando FLAML Regressor...", 0.0)
        
        settings = {
            "time_budget": self.time_budget,
            "metric": self.metric,
            "task": "regression",
            "n_jobs": self.n_jobs,
            "ensemble": self.ensemble,
            "seed": self.random_state,
            "verbose": self.verbose,
        }
        
        if self.estimator_list:
            settings["estimator_list"] = self.estimator_list
        
        # Always use log file for tracking progress
        settings["log_file_name"] = log_file_path
        
        settings.update(self.extra_kwargs)
        settings.update(kwargs)
        
        # Create FLAML's native callback for real-time updates
        def flaml_callback(config: dict, metric_name: str, score: float, 
                          time_used: float, estimator: str, best_config: dict):
            """FLAML native callback - called after each trial."""
            # For regression metrics, score is typically the negative loss
            loss = -score if self.metric in ['r2'] else score
            progress_tracker.update(estimator=estimator, loss=loss)
        
        # Add callback to settings
        settings["cb"] = flaml_callback
        
        # Initialize and fit with progress tracking
        self.automl_ = AutoML()
        
        # Start progress tracker thread
        progress_tracker.start()
        
        try:
            self.automl_.fit(X_train, y_train, **settings)
        finally:
            progress_tracker.stop()
        
        self.best_estimator_ = self.automl_.best_estimator
        self.best_model_ = self.automl_.model
        self.best_loss_ = self.automl_.best_loss
        self.fit_time_ = time.time() - start_time
        self.training_logs_ = progress_tracker.get_logs()
        
        # Add final summary to logs
        self.training_logs_.append(
            f"[{time.strftime('%H:%M:%S')}] ✅ Entrenamiento completado!"
        )
        self.training_logs_.append(
            f"[{time.strftime('%H:%M:%S')}] 🏆 Mejor modelo: {self.best_estimator_} (loss={self.best_loss_:.4f})"
        )
        self.training_logs_.append(
            f"[{time.strftime('%H:%M:%S')}] ⏱️ Tiempo total: {self.fit_time_:.1f}s"
        )
        
        self.is_fitted_ = True
        
        if self.progress_callback:
            self.progress_callback(
                f"✅ FLAML Regressor completado: {self.best_estimator_} "
                f"(loss={self.best_loss_:.4f}, time={self.fit_time_:.1f}s)",
                1.0
            )
        
        return self
    
    def get_training_logs(self) -> List[str]:
        """Get the training logs captured during fit."""
        return self.training_logs_.copy()
    
    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        self._validate_input(X, training=False)
        
        if isinstance(X, np.ndarray):
            X_pred = pd.DataFrame(X)
        else:
            X_pred = X
        
        return self.automl_.predict(X_pred)
    
    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get parameters."""
        return {
            "time_budget": self.time_budget,
            "metric": self.metric,
            "estimator_list": self.estimator_list,
            "n_jobs": self.n_jobs,
            "ensemble": self.ensemble,
            "random_state": self.random_state,
            "verbose": self.verbose,
            "name": self.name,
        }
    
    def set_params(self, **params) -> "FLAMLRegressor":
        """Set parameters."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get extended metadata."""
        metadata = super().get_metadata()
        metadata.update({
            'backend': 'flaml',
            'best_estimator': self.best_estimator_,
            'best_loss': self.best_loss_,
            'fit_time_seconds': self.fit_time_,
            'time_budget': self.time_budget,
            'metric': self.metric,
            'n_logs': len(self.training_logs_),
        })
        return metadata
