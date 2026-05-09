"""Model Selection Module.

This module provides functionality for automatic best model selection
based on evaluation metrics and configurable criteria.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Literal
from dataclasses import dataclass
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score

from src.models.metadata import ModelMetadata


@dataclass
class SelectionCriteria:
    """Criteria for model selection.
    
    Attributes:
        primary_metric: Primary metric to optimize ('auroc', 'f1', 'accuracy', etc.).
        minimize: If True, minimize the metric (e.g., for loss). If False, maximize.
        secondary_metrics: Additional metrics to consider for tie-breaking.
        weights: Optional weights for combining multiple metrics.
        min_thresholds: Minimum acceptable values for each metric.
        max_complexity_penalty: Penalty factor for model complexity (0-1).
    """
    primary_metric: str = 'auroc'
    minimize: bool = False
    secondary_metrics: Optional[List[str]] = None
    weights: Optional[Dict[str, float]] = None
    min_thresholds: Optional[Dict[str, float]] = None
    max_complexity_penalty: float = 0.1
    
    def __post_init__(self):
        """Validate and set defaults."""
        if self.secondary_metrics is None:
            self.secondary_metrics = ['f1', 'precision', 'recall']
        
        if self.weights is None:
            # Default: primary metric has 70% weight, secondaries share 30%
            n_secondary = len(self.secondary_metrics)
            self.weights = {self.primary_metric: 0.7}
            if n_secondary > 0:
                weight_per_secondary = 0.3 / n_secondary
                for metric in self.secondary_metrics:
                    self.weights[metric] = weight_per_secondary
        
        if self.min_thresholds is None:
            self.min_thresholds = {}


@dataclass
class ModelScore:
    """Score for a single model.
    
    Attributes:
        model_name: Name of the model.
        primary_score: Score on primary metric.
        weighted_score: Combined weighted score.
        metrics: Dictionary of all computed metrics.
        complexity: Model complexity measure (e.g., number of parameters).
        rank: Rank among all models (1 = best).
        meets_thresholds: Whether model meets minimum thresholds.
    """
    model_name: str
    primary_score: float
    weighted_score: float
    metrics: Dict[str, float]
    complexity: Optional[int] = None
    rank: Optional[int] = None
    meets_thresholds: bool = True
    
    def __repr__(self) -> str:
        return (f"ModelScore(model='{self.model_name}', "
                f"primary={self.primary_score:.4f}, "
                f"weighted={self.weighted_score:.4f}, "
                f"rank={self.rank})")


class BestModelSelector:
    """Automatic best model selector.
    
    This class implements various strategies for selecting the best model
    from a set of trained models based on evaluation metrics.
    """
    
    def __init__(self, criteria: Optional[SelectionCriteria] = None):
        """
        Initialize selector.
        
        Args:
            criteria: Selection criteria. Uses defaults if None.
        """
        self.criteria = criteria or SelectionCriteria()
        self.model_scores: List[ModelScore] = []
        self.best_model: Optional[ModelScore] = None
    
    def select_best_model(
        self,
        models_metrics: Dict[str, Dict[str, float]],
        models_metadata: Optional[Dict[str, ModelMetadata]] = None
    ) -> ModelScore:
        """
        Select the best model from evaluation results.
        
        Args:
            models_metrics: Dictionary mapping model names to their metrics.
                Example: {'xgb': {'auroc': 0.85, 'f1': 0.80, ...}, ...}
            models_metadata: Optional metadata for each model (for complexity).
        
        Returns:
            ModelScore for the best model.
        
        Example:
            ```python
            from src.models.selection import BestModelSelector, SelectionCriteria
            
            # Define criteria
            criteria = SelectionCriteria(
                primary_metric='auroc',
                secondary_metrics=['f1', 'precision'],
                min_thresholds={'auroc': 0.7, 'f1': 0.65}
            )
            
            # Prepare metrics
            models_metrics = {
                'xgb': {'auroc': 0.85, 'f1': 0.80, 'precision': 0.78},
                'rf': {'auroc': 0.83, 'f1': 0.82, 'precision': 0.80},
            }
            
            # Select best
            selector = BestModelSelector(criteria)
            best = selector.select_best_model(models_metrics)
            print(f"Best model: {best.model_name}")
            ```
        """
        self.model_scores = []
        
        # Calculate scores for each model
        for model_name, metrics in models_metrics.items():
            # Get complexity if available
            complexity = None
            if models_metadata and model_name in models_metadata:
                metadata = models_metadata[model_name]
                if hasattr(metadata, 'n_parameters'):
                    complexity = metadata.n_parameters
            
            # Calculate weighted score
            weighted_score = self._calculate_weighted_score(metrics)
            
            # Check thresholds
            meets_thresholds = self._check_thresholds(metrics)
            
            # Get primary metric score
            primary_score = metrics.get(self.criteria.primary_metric, 0.0)
            
            # Apply complexity penalty if requested
            if complexity and self.criteria.max_complexity_penalty > 0:
                # Normalize complexity (higher complexity = higher penalty)
                max_complexity = max(
                    [m.n_parameters for m in models_metadata.values() 
                     if hasattr(m, 'n_parameters')],
                    default=1
                )
                complexity_ratio = complexity / max_complexity
                penalty = complexity_ratio * self.criteria.max_complexity_penalty
                weighted_score *= (1 - penalty)
            
            model_score = ModelScore(
                model_name=model_name,
                primary_score=primary_score,
                weighted_score=weighted_score,
                metrics=metrics.copy(),
                complexity=complexity,
                meets_thresholds=meets_thresholds
            )
            
            self.model_scores.append(model_score)
        
        # Filter models that don't meet thresholds
        valid_models = [m for m in self.model_scores if m.meets_thresholds]
        
        if not valid_models:
            raise ValueError(
                "No models meet the minimum threshold requirements. "
                f"Thresholds: {self.criteria.min_thresholds}"
            )
        
        # Sort by weighted score (descending if maximizing, ascending if minimizing)
        valid_models.sort(
            key=lambda x: x.weighted_score,
            reverse=not self.criteria.minimize
        )
        
        # Assign ranks
        for rank, model in enumerate(valid_models, start=1):
            model.rank = rank
        
        # Also assign ranks to invalid models (but at the end)
        invalid_models = [m for m in self.model_scores if not m.meets_thresholds]
        for rank, model in enumerate(invalid_models, start=len(valid_models) + 1):
            model.rank = rank
        
        # Best model is the first valid one
        self.best_model = valid_models[0]
        
        return self.best_model
    
    def _calculate_weighted_score(self, metrics: Dict[str, float]) -> float:
        """Calculate weighted score from multiple metrics."""
        total_score = 0.0
        total_weight = 0.0
        
        for metric, weight in self.criteria.weights.items():
            if metric in metrics:
                value = metrics[metric]
                # Handle NaN/None values
                if value is None or (isinstance(value, float) and np.isnan(value)):
                    value = 0.0
                
                total_score += value * weight
                total_weight += weight
        
        # Normalize by actual weights used
        if total_weight > 0:
            return total_score / total_weight
        return 0.0
    
    def _check_thresholds(self, metrics: Dict[str, float]) -> bool:
        """Check if model meets minimum thresholds."""
        for metric, threshold in self.criteria.min_thresholds.items():
            if metric not in metrics:
                return False
            
            value = metrics[metric]
            if value is None or (isinstance(value, float) and np.isnan(value)):
                return False
            
            if value < threshold:
                return False
        
        return True
    
    def get_top_k_models(self, k: int = 3) -> List[ModelScore]:
        """
        Get top K models by score.
        
        Args:
            k: Number of top models to return.
        
        Returns:
            List of top K ModelScore objects.
        """
        if not self.model_scores:
            return []
        
        # Filter valid models and sort
        valid_models = [m for m in self.model_scores if m.meets_thresholds]
        valid_models.sort(
            key=lambda x: x.weighted_score,
            reverse=not self.criteria.minimize
        )
        
        return valid_models[:k]
    
    def get_selection_report(self) -> Dict[str, Any]:
        """
        Generate detailed selection report.
        
        Returns:
            Dictionary with selection details and rankings.
        """
        if not self.model_scores:
            return {'error': 'No models scored yet'}
        
        report = {
            'criteria': {
                'primary_metric': self.criteria.primary_metric,
                'minimize': self.criteria.minimize,
                'secondary_metrics': self.criteria.secondary_metrics,
                'weights': self.criteria.weights,
                'min_thresholds': self.criteria.min_thresholds,
            },
            'best_model': {
                'name': self.best_model.model_name if self.best_model else None,
                'primary_score': self.best_model.primary_score if self.best_model else None,
                'weighted_score': self.best_model.weighted_score if self.best_model else None,
                'metrics': self.best_model.metrics if self.best_model else {},
            },
            'rankings': []
        }
        
        # Sort all models by rank
        sorted_models = sorted(self.model_scores, key=lambda x: x.rank or 999)
        
        for model in sorted_models:
            report['rankings'].append({
                'rank': model.rank,
                'model_name': model.model_name,
                'primary_score': model.primary_score,
                'weighted_score': model.weighted_score,
                'meets_thresholds': model.meets_thresholds,
                'metrics': model.metrics,
                'complexity': model.complexity,
            })
        
        return report
    
    def save_selection_report(self, output_path: str | Path) -> Path:
        """
        Save selection report to JSON file.
        
        Args:
            output_path: Path to save the report.
        
        Returns:
            Path to saved report.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        report = self.get_selection_report()
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        return output_path
    
    def compare_models(
        self,
        model1_name: str,
        model2_name: str
    ) -> Dict[str, Any]:
        """
        Compare two models in detail.
        
        Args:
            model1_name: Name of first model.
            model2_name: Name of second model.
        
        Returns:
            Dictionary with detailed comparison.
        """
        model1 = next((m for m in self.model_scores if m.model_name == model1_name), None)
        model2 = next((m for m in self.model_scores if m.model_name == model2_name), None)
        
        if not model1 or not model2:
            raise ValueError(f"One or both models not found in scored models")
        
        comparison = {
            'models': [model1_name, model2_name],
            'winner': None,
            'differences': {},
            'summary': []
        }
        
        # Compare each metric
        all_metrics = set(model1.metrics.keys()) | set(model2.metrics.keys())
        
        for metric in all_metrics:
            val1 = model1.metrics.get(metric, 0.0)
            val2 = model2.metrics.get(metric, 0.0)
            diff = val1 - val2
            
            comparison['differences'][metric] = {
                model1_name: val1,
                model2_name: val2,
                'difference': diff,
                'percentage_diff': (diff / val2 * 100) if val2 != 0 else 0.0
            }
        
        # Determine winner
        if model1.weighted_score > model2.weighted_score:
            comparison['winner'] = model1_name
            comparison['summary'].append(
                f"{model1_name} has higher weighted score "
                f"({model1.weighted_score:.4f} vs {model2.weighted_score:.4f})"
            )
        elif model2.weighted_score > model1.weighted_score:
            comparison['winner'] = model2_name
            comparison['summary'].append(
                f"{model2_name} has higher weighted score "
                f"({model2.weighted_score:.4f} vs {model1.weighted_score:.4f})"
            )
        else:
            comparison['summary'].append("Models are tied in weighted score")
        
        return comparison


def select_best_model_simple(
    models_metrics: Dict[str, Dict[str, float]],
    metric: str = 'auroc'
) -> str:
    """
    Simple best model selection by single metric.
    
    Args:
        models_metrics: Dictionary mapping model names to metrics.
        metric: Metric to use for selection.
    
    Returns:
        Name of best model.
    
    Example:
        ```python
        models_metrics = {
            'xgb': {'auroc': 0.85, 'f1': 0.80},
            'rf': {'auroc': 0.83, 'f1': 0.82},
        }
        
        best = select_best_model_simple(models_metrics, metric='auroc')
        # Returns: 'xgb'
        ```
    """
    best_model = None
    best_score = -np.inf
    
    for model_name, metrics in models_metrics.items():
        if metric in metrics:
            score = metrics[metric]
            if score > best_score:
                best_score = score
                best_model = model_name
    
    if best_model is None:
        raise ValueError(f"Metric '{metric}' not found in any model's metrics")
    
    return best_model
