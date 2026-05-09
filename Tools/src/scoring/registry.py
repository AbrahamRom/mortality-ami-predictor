"""Score registry for clinical scoring systems."""
from __future__ import annotations

from typing import Any, Dict, Type

# Registry of available scores
_SCORE_REGISTRY: Dict[str, Any] = {}


def register_score(name: str, scorer_class: Type) -> None:
    """Register a scoring class.
    
    Args:
        name: Name identifier for the score
        scorer_class: The scorer class to register
    """
    _SCORE_REGISTRY[name.lower()] = scorer_class


def get_score(name: str) -> Any:
    """Get a scorer instance by name.
    
    Args:
        name: Name of the score (case-insensitive)
        
    Returns:
        Instance of the scorer
        
    Raises:
        KeyError: If score not found
    """
    name_lower = name.lower()
    if name_lower not in _SCORE_REGISTRY:
        raise KeyError(f"Score '{name}' not found. Available: {list(_SCORE_REGISTRY.keys())}")
    return _SCORE_REGISTRY[name_lower]()


def list_scores() -> list[str]:
    """List all available score names.
    
    Returns:
        List of registered score names
    """
    return list(_SCORE_REGISTRY.keys())


# Auto-register scores on module import
def _auto_register():
    """Automatically register all available scores."""
    try:
        from .grace import GRACEScore
        register_score("grace", GRACEScore)
    except ImportError:
        pass
    
    try:
        from .timi import TIMIScore
        register_score("timi", TIMIScore)
    except ImportError:
        pass
    
    try:
        from .recuima import RECUIMAScorer
        register_score("recuima", RECUIMAScorer)
    except ImportError:
        pass


# Run auto-registration
_auto_register()
