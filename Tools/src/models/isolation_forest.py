from src.models.custom_base import BaseCustomClassifier
from sklearn.ensemble import IsolationForest
import numpy as np

class IsolationForestBinaryClassifier(BaseCustomClassifier):
    """
    Clasificador binario basado en Isolation Forest.

    Convención:
        0 -> normal
        1 -> anomalía
    """

    def __init__(
        self,
        n_estimators=200,
        max_samples=256,
        contamination=0.1,
        max_features=1.0,
        bootstrap=False,
        threshold="auto",
        random_state=None,
        n_jobs=None,
        name="IsolationForestBinaryClassifier" 
    ):
        super().__init__(name=name)
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.contamination = contamination
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.threshold = threshold
        self.random_state = random_state
        self.n_jobs = n_jobs
        
        self.model_ = IsolationForest(
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            contamination=self.contamination,
            max_features=self.max_features,
            bootstrap=self.bootstrap,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )

    def get_params(self, deep=True):
        return {
            "n_estimators": self.n_estimators,
            "max_samples": self.max_samples,
            "contamination": self.contamination,
            "max_features": self.max_features,
            "bootstrap": self.bootstrap,
            "threshold": self.threshold,
            "random_state": self.random_state,
            "n_jobs": self.n_jobs,
            "name": self.name,
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self


    def fit(self, X, y):
        self._validate_input(X, training=True)
        y = self._validate_targets(y, training=True)
        
        X = self._convert_to_array(X)

        self.model_.fit(X, y)
        self.is_fitted_ = True

        scores = self.model_.decision_function(X)

        if self.threshold == "auto":
            self.threshold_ = np.percentile(
                scores, 100 * self.contamination
            )
        else:
            self.threshold_ = float(self.threshold)

        self.classes_ = np.array([0, 1])
        return self

    def predict(self, X):
        scores = self.model_.decision_function(X)
        return (scores < self.threshold_).astype(int)

    def predict_proba(self, X):
        self._validate_input(X, training=False)
        X =  self._convert_to_array(X)
        
        scores = self.model_.decision_function(X)

        probs_anomaly = 1 / (1 + np.exp(scores))
        probs_normal = 1 - probs_anomaly

        return np.column_stack([probs_normal, probs_anomaly])
