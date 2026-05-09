"""Learning curve generation for model evaluation.

This module provides functions to generate and visualize learning curves
to assess model performance as training size varies.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.base import clone
from sklearn.model_selection import learning_curve as sklearn_learning_curve

from src.models.custom_base import BaseCustomModel


@dataclass
class LearningCurveResult:
    """Results from learning curve analysis."""
    
    train_sizes: List[int]
    train_scores_mean: List[float]
    train_scores_std: List[float]
    val_scores_mean: List[float]
    val_scores_std: List[float]
    train_scores_all: np.ndarray  # Shape: (n_sizes, n_cv_folds)
    val_scores_all: np.ndarray
    
    def __str__(self) -> str:
        """String representation."""
        lines = [
            "=" * 70,
            "Learning Curve Results",
            "=" * 70,
            f"Training sizes evaluated: {len(self.train_sizes)}",
            f"Range: {min(self.train_sizes)} to {max(self.train_sizes)} samples",
            "",
            "Final performance:",
            f"  Train score: {self.train_scores_mean[-1]:.4f} ± {self.train_scores_std[-1]:.4f}",
            f"  Val score: {self.val_scores_mean[-1]:.4f} ± {self.val_scores_std[-1]:.4f}",
            f"  Gap: {abs(self.train_scores_mean[-1] - self.val_scores_mean[-1]):.4f}",
            "=" * 70,
        ]
        return "\n".join(lines)


def _has_internal_parallelism(model) -> bool:
    """Check if a model has internal parallelism (like XGBoost, LightGBM).
    
    These models should not be used with n_jobs=-1 in sklearn's parallel
    processing to avoid nested parallelism deadlocks.
    """
    model_class_name = type(model).__name__.lower()
    
    # Models with internal parallelism
    parallel_models = [
        'xgb', 'xgboost', 'lgbm', 'lightgbm', 
        'catboost', 'gradientboosting'
    ]
    
    for name in parallel_models:
        if name in model_class_name:
            return True
    
    # Check if it's a pipeline and look at the final estimator
    if hasattr(model, 'steps'):
        final_estimator = model.steps[-1][1]
        return _has_internal_parallelism(final_estimator)
    
    return False


def generate_learning_curve(
    model,
    X: np.ndarray,
    y: np.ndarray,
    cv: int = 5,
    train_sizes: Optional[np.ndarray] = None,
    scoring: str = "roc_auc",
    n_jobs: int = -1,
    random_state: Optional[int] = None,
) -> LearningCurveResult:
    """Generate learning curve for a model.
    
    Args:
        model: Model or pipeline to evaluate
        X: Features
        y: Labels
        cv: Number of cross-validation folds
        train_sizes: Array of training sizes to evaluate (default: 10 points from 10% to 100%)
        scoring: Scoring metric
        n_jobs: Number of parallel jobs (-1 for all cores, but will be overridden to 1
                for models with internal parallelism to avoid deadlocks)
        random_state: Random seed
        
    Returns:
        LearningCurveResult with learning curve data
    """
    if train_sizes is None:
        train_sizes = np.linspace(0.1, 1.0, 10)
    
    # Custom models may not be picklable, so disable parallel processing for them
    if isinstance(model, BaseCustomModel):
        n_jobs = 1
    
    # Models with internal parallelism (XGBoost, LightGBM, etc.) should not use
    # sklearn's parallel processing to avoid nested parallelism deadlocks
    if _has_internal_parallelism(model):
        n_jobs = 1
    
    # Generate learning curve
    try:
        train_sizes_abs, train_scores, val_scores = sklearn_learning_curve(
            estimator=model,
            X=X,
            y=y,
            cv=cv,
            train_sizes=train_sizes,
            scoring=scoring,
            n_jobs=n_jobs,
            random_state=random_state,
            shuffle=True,
            error_score='raise'
        )
    except Exception as e:
        # Fallback: try with error_score=np.nan to see if we can get partial results
        # and avoid crashing the whole pipeline
        print(f"Warning: Learning curve generation failed with error: {e}. Retrying with error_score=nan")
        train_sizes_abs, train_scores, val_scores = sklearn_learning_curve(
            estimator=model,
            X=X,
            y=y,
            cv=cv,
            train_sizes=train_sizes,
            scoring=scoring,
            n_jobs=n_jobs,
            random_state=random_state,
            shuffle=True,
            error_score=np.nan
        )
    
    # Calculate mean and std using nanmean/nanstd to handle potential failures
    with np.errstate(invalid='ignore'): # Suppress warnings for Mean of empty slice
        train_scores_mean = np.nanmean(train_scores, axis=1)
        train_scores_std = np.nanstd(train_scores, axis=1)
        val_scores_mean = np.nanmean(val_scores, axis=1)
        val_scores_std = np.nanstd(val_scores, axis=1)
        
    # Check if we have valid results
    if np.all(np.isnan(train_scores_mean)) or np.all(np.isnan(val_scores_mean)):
        # If all results are NaN, it might be due to small dataset or scoring issues
        # Try to provide a helpful message in the logs (or just return what we have)
        print("Warning: All learning curve scores are NaN. Check if dataset is too small or scoring metric is appropriate.")
    
    return LearningCurveResult(
        train_sizes=train_sizes_abs.tolist(),
        train_scores_mean=train_scores_mean.tolist(),
        train_scores_std=train_scores_std.tolist(),
        val_scores_mean=val_scores_mean.tolist(),
        val_scores_std=val_scores_std.tolist(),
        train_scores_all=train_scores,
        val_scores_all=val_scores,
    )


def plot_learning_curve(
    result: LearningCurveResult,
    title: str = "Learning Curve",
    save_path: Optional[str] = None,
) -> Union[go.Figure, plt.Figure]:
    """Plot learning curve using Plotly for interactivity.
    
    Args:
        result: LearningCurveResult object
        title: Plot title
        save_path: Optional path to save figure (saves as PNG)
        
    Returns:
        Plotly Figure object
    """
    train_sizes = np.array(result.train_sizes)
    train_mean = np.array(result.train_scores_mean)
    train_std = np.array(result.train_scores_std)
    val_mean = np.array(result.val_scores_mean)
    val_std = np.array(result.val_scores_std)
    
    fig = go.Figure()
    
    # Training scores with confidence band
    fig.add_trace(go.Scatter(
        x=train_sizes,
        y=train_mean,
        mode='lines+markers',
        name='Training score',
        line=dict(color='blue', width=2),
        marker=dict(size=8),
        hovertemplate=(
            'Training Size: %{x}<br>'
            'Score: %{y:.4f}<br>'
            '<extra></extra>'
        )
    ))
    
    # Training confidence band
    fig.add_trace(go.Scatter(
        x=np.concatenate([train_sizes, train_sizes[::-1]]),
        y=np.concatenate([train_mean + train_std, (train_mean - train_std)[::-1]]),
        fill='toself',
        fillcolor='rgba(0, 0, 255, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        showlegend=False,
        hoverinfo='skip',
        name='Training ± std'
    ))
    
    # Validation scores with confidence band
    fig.add_trace(go.Scatter(
        x=train_sizes,
        y=val_mean,
        mode='lines+markers',
        name='Validation score',
        line=dict(color='green', width=2),
        marker=dict(size=8),
        hovertemplate=(
            'Training Size: %{x}<br>'
            'Score: %{y:.4f}<br>'
            '<extra></extra>'
        )
    ))
    
    # Validation confidence band
    fig.add_trace(go.Scatter(
        x=np.concatenate([train_sizes, train_sizes[::-1]]),
        y=np.concatenate([val_mean + val_std, (val_mean - val_std)[::-1]]),
        fill='toself',
        fillcolor='rgba(0, 255, 0, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        showlegend=False,
        hoverinfo='skip',
        name='Validation ± std'
    ))
    
    # Calculate final scores for annotation
    final_train = train_mean[-1]
    final_val = val_mean[-1]
    gap = abs(final_train - final_val)
    
    # Add annotation box
    annotation_text = (
        f"Final Train: {final_train:.3f}<br>"
        f"Final Val: {final_val:.3f}<br>"
        f"Gap: {gap:.3f}"
    )
    
    fig.add_annotation(
        x=0.05,
        y=0.05,
        xref='paper',
        yref='paper',
        text=annotation_text,
        showarrow=False,
        bgcolor='wheat',
        bordercolor='black',
        borderwidth=1,
        font=dict(size=10),
        align='left'
    )
    
    fig.update_layout(
        title=title,
        xaxis=dict(
            title='Training Set Size',
            gridcolor='lightgray'
        ),
        yaxis=dict(
            title='Score (AUROC)',
            gridcolor='lightgray'
        ),
        template='plotly_white',
        hovermode='x unified',
        width=1000,
        height=600,
        legend=dict(x=0.7, y=0.1)
    )
    
    if save_path:
        fig.write_image(save_path, width=1000, height=600)
    
    return fig


def plot_multiple_learning_curves(
    results: dict[str, LearningCurveResult],
    title: str = "Learning Curves Comparison",
    save_path: Optional[str] = None,
) -> Union[go.Figure, plt.Figure]:
    """Plot multiple learning curves for model comparison using Plotly.
    
    Args:
        results: Dictionary mapping model_name -> LearningCurveResult
        title: Plot title
        save_path: Optional path to save figure
        
    Returns:
        Plotly Figure object
    """
    # Create subplots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Training Curves', 'Validation Curves'),
        horizontal_spacing=0.1
    )
    
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    
    # Plot training curves (left)
    for idx, (name, result) in enumerate(results.items()):
        color = colors[idx % len(colors)]
        train_sizes = np.array(result.train_sizes)
        train_mean = np.array(result.train_scores_mean)
        train_std = np.array(result.train_scores_std)
        
        # Training line
        fig.add_trace(go.Scatter(
            x=train_sizes,
            y=train_mean,
            mode='lines+markers',
            name=name,
            line=dict(color=color, width=2),
            marker=dict(size=6),
            legendgroup=name,
            showlegend=True,
            hovertemplate=(
                f'<b>{name}</b><br>'
                'Size: %{x}<br>'
                'Score: %{y:.4f}<br>'
                '<extra></extra>'
            )
        ), row=1, col=1)
        
        # Training confidence band
        fig.add_trace(go.Scatter(
            x=np.concatenate([train_sizes, train_sizes[::-1]]),
            y=np.concatenate([train_mean + train_std, (train_mean - train_std)[::-1]]),
            fill='toself',
            fillcolor=f'rgba({",".join(map(str, _hex_to_rgb(color)))}, 0.15)',
            line=dict(color='rgba(255,255,255,0)'),
            showlegend=False,
            legendgroup=name,
            hoverinfo='skip'
        ), row=1, col=1)
    
    # Plot validation curves (right)
    for idx, (name, result) in enumerate(results.items()):
        color = colors[idx % len(colors)]
        train_sizes = np.array(result.train_sizes)
        val_mean = np.array(result.val_scores_mean)
        val_std = np.array(result.val_scores_std)
        
        # Validation line
        fig.add_trace(go.Scatter(
            x=train_sizes,
            y=val_mean,
            mode='lines+markers',
            name=name,
            line=dict(color=color, width=2),
            marker=dict(size=6),
            legendgroup=name,
            showlegend=False,
            hovertemplate=(
                f'<b>{name}</b><br>'
                'Size: %{x}<br>'
                'Score: %{y:.4f}<br>'
                '<extra></extra>'
            )
        ), row=1, col=2)
        
        # Validation confidence band
        fig.add_trace(go.Scatter(
            x=np.concatenate([train_sizes, train_sizes[::-1]]),
            y=np.concatenate([val_mean + val_std, (val_mean - val_std)[::-1]]),
            fill='toself',
            fillcolor=f'rgba({",".join(map(str, _hex_to_rgb(color)))}, 0.15)',
            line=dict(color='rgba(255,255,255,0)'),
            showlegend=False,
            legendgroup=name,
            hoverinfo='skip'
        ), row=1, col=2)
    
    # Update axes
    fig.update_xaxes(title_text='Training Set Size', row=1, col=1, gridcolor='lightgray')
    fig.update_xaxes(title_text='Training Set Size', row=1, col=2, gridcolor='lightgray')
    fig.update_yaxes(title_text='Training Score (AUROC)', row=1, col=1, gridcolor='lightgray')
    fig.update_yaxes(title_text='Validation Score (AUROC)', row=1, col=2, gridcolor='lightgray')
    
    fig.update_layout(
        title=title,
        template='plotly_white',
        height=600,
        width=1600,
        hovermode='x unified',
        legend=dict(x=1.05, y=0.5)
    )
    
    if save_path:
        fig.write_image(save_path, width=1600, height=600)
    
    return fig


def _hex_to_rgb(color_name: str) -> Tuple[int, int, int]:
    """Convert color name to RGB tuple."""
    color_map = {
        'blue': (0, 0, 255),
        'red': (255, 0, 0),
        'green': (0, 255, 0),
        'orange': (255, 165, 0),
        'purple': (128, 0, 128),
        'brown': (165, 42, 42),
        'pink': (255, 192, 203),
        'gray': (128, 128, 128),
        'olive': (128, 128, 0),
        'cyan': (0, 255, 255),
    }
    return color_map.get(color_name, (0, 0, 0))


def diagnose_learning_curve(result: LearningCurveResult) -> dict:
    """Diagnose issues from learning curve.
    
    Args:
        result: LearningCurveResult to analyze
        
    Returns:
        Dictionary with diagnostic information
    """
    final_train = result.train_scores_mean[-1]
    final_val = result.val_scores_mean[-1]
    gap = abs(final_train - final_val)
    
    # Analyze trends
    train_improving = result.train_scores_mean[-1] > result.train_scores_mean[0]
    val_improving = result.val_scores_mean[-1] > result.val_scores_mean[0]
    
    # Convergence check (is validation score stabilizing?)
    if len(result.val_scores_mean) >= 3:
        recent_val_std = np.std(result.val_scores_mean[-3:])
        converged = recent_val_std < 0.01
    else:
        converged = False
    
    # Diagnose problems
    diagnosis = {
        'final_train_score': final_train,
        'final_val_score': final_val,
        'gap': gap,
        'train_improving': train_improving,
        'val_improving': val_improving,
        'converged': converged,
        'issues': [],
        'recommendations': [],
    }
    
    # High bias (underfitting)
    if final_train < 0.75 and final_val < 0.75:
        diagnosis['issues'].append('High bias (underfitting)')
        diagnosis['recommendations'].append('Use more complex model')
        diagnosis['recommendations'].append('Add more features')
        diagnosis['recommendations'].append('Reduce regularization')
    
    # High variance (overfitting)
    if gap > 0.1:
        diagnosis['issues'].append('High variance (overfitting)')
        diagnosis['recommendations'].append('Get more training data')
        diagnosis['recommendations'].append('Reduce model complexity')
        diagnosis['recommendations'].append('Increase regularization')
        diagnosis['recommendations'].append('Use dropout or other regularization techniques')
    
    # Not converged
    if not converged and val_improving:
        diagnosis['issues'].append('Model not converged')
        diagnosis['recommendations'].append('Collect more training data')
        diagnosis['recommendations'].append('Continue training')
    
    # Good fit
    if gap < 0.05 and final_val > 0.80:
        diagnosis['issues'].append('Good fit!')
        diagnosis['recommendations'].append('Model is performing well')
    
    return diagnosis


def print_diagnosis(diagnosis: dict):
    """Print learning curve diagnosis in readable format.
    
    Args:
        diagnosis: Dictionary from diagnose_learning_curve
    """
    print("=" * 70)
    print("LEARNING CURVE DIAGNOSIS")
    print("=" * 70)
    print(f"\nFinal Scores:")
    print(f"  Training:   {diagnosis['final_train_score']:.4f}")
    print(f"  Validation: {diagnosis['final_val_score']:.4f}")
    print(f"  Gap:        {diagnosis['gap']:.4f}")
    
    print(f"\nTrends:")
    print(f"  Training improving:   {diagnosis['train_improving']}")
    print(f"  Validation improving: {diagnosis['val_improving']}")
    print(f"  Converged:            {diagnosis['converged']}")
    
    if diagnosis['issues']:
        print(f"\nIssues Detected:")
        for issue in diagnosis['issues']:
            print(f"  ⚠️  {issue}")
    
    if diagnosis['recommendations']:
        print(f"\nRecommendations:")
        for rec in diagnosis['recommendations']:
            print(f"  💡 {rec}")
    
    print("=" * 70)
