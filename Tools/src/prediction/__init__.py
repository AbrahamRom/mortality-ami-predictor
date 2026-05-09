"""Prediction module for making predictions with trained models."""

from .predictor import Predictor, load_predictor, predict_mortality

__all__ = [
    "Predictor",
    "load_predictor",
    "predict_mortality",
]
