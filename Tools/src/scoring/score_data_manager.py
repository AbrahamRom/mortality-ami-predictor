"""Manager for preserving original data needed for clinical score calculations.

This module ensures that when preprocessing data for ML models, we also preserve
the original values needed for clinical scores (GRACE, RECUIMA, etc.) that may be
transformed or encoded during preprocessing.

The key insight is that:
- ML models may use encoded/transformed features
- Clinical scores need original raw values (e.g., 'complicaciones' as text, not numeric)
- We need to preserve original test set indices to match predictions with raw data
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any
import json

import pandas as pd
import numpy as np


# ============================================================================
# CLINICAL SCORE VARIABLE DEFINITIONS
# ============================================================================

# Variables required for GRACE score calculation
GRACE_REQUIRED_VARIABLES = [
    'edad',                         # Age in years
    'frecuencia_cardiaca',          # Heart rate (bpm)
    'presion_arterial_sistolica',   # Systolic BP (mmHg)
    'creatinina',                   # Creatinine (mg/dL)
    'indice_killip',                # Killip class (I-IV or 1-4)
    'tropnina_hs',                  # Troponin (elevated enzymes marker)
]

GRACE_OPTIONAL_VARIABLES = [
    'escala_grace',                 # Pre-computed GRACE score if available
    'depresion_st',                 # ST depression
    'paro_cardiaco',                # Cardiac arrest at admission
]

# Variables required for RECUIMA score calculation
RECUIMA_REQUIRED_VARIABLES = [
    'edad',                         # Age > 70 years
    'presion_arterial_sistolica',   # SBP < 100 mmHg
    'filtrado_glomerular',          # GFR < 60 ml/min/1.73m²
    'indice_killip',                # Killip-Kimball IV
    'complicaciones',               # Contains FV/TV and BAV (as TEXT)
]

# ECG lead variables (for counting affected leads > 7)
RECUIMA_ECG_LEADS = [
    'v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7', 'v8', 'v9',
    'd1', 'd2', 'd3',
    'avf', 'avl', 'avr',
]

# Target/outcome variables
TARGET_VARIABLES = [
    'estado_vital',                 # Vital status (vivo/fallecido)
    'exitus',                       # Death indicator
    'mortality',                    # Mortality indicator
    'mortality_inhospital',         # In-hospital mortality
]

# All variables to preserve for clinical scores
ALL_SCORE_VARIABLES = list(set(
    GRACE_REQUIRED_VARIABLES +
    GRACE_OPTIONAL_VARIABLES +
    RECUIMA_REQUIRED_VARIABLES +
    RECUIMA_ECG_LEADS +
    TARGET_VARIABLES
))


# ============================================================================
# DEFAULT ORIGINAL DATASET PATH
# ============================================================================

# Get absolute path based on project root (Tools/src/scoring -> Tools -> project root)
_PROJECT_ROOT = Path(__file__).parents[3]  # src/scoring/file.py -> src -> Tools -> project_root

# Default path to original dataset
DEFAULT_ORIGINAL_DATASET_PATH = str(_PROJECT_ROOT / "DATA" / "recuima-020425.xlsx")


@dataclass
class ScoreDataConfig:
    """Configuration for score data management."""
    
    original_dataset_path: str = DEFAULT_ORIGINAL_DATASET_PATH
    preserve_variables: List[str] = field(default_factory=lambda: ALL_SCORE_VARIABLES)
    grace_variables: List[str] = field(default_factory=lambda: GRACE_REQUIRED_VARIABLES + GRACE_OPTIONAL_VARIABLES)
    recuima_variables: List[str] = field(default_factory=lambda: RECUIMA_REQUIRED_VARIABLES + RECUIMA_ECG_LEADS)
    target_variables: List[str] = field(default_factory=lambda: TARGET_VARIABLES)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'original_dataset_path': self.original_dataset_path,
            'preserve_variables': self.preserve_variables,
            'grace_variables': self.grace_variables,
            'recuima_variables': self.recuima_variables,
            'target_variables': self.target_variables,
        }


def load_original_dataset(
    path: Optional[str] = None,
    base_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Load the original dataset with raw/unencoded values.
    
    Args:
        path: Path to original dataset. If None, uses default.
        base_dir: Base directory for relative paths. If None, uses current working directory.
        
    Returns:
        DataFrame with original data
        
    Raises:
        FileNotFoundError: If dataset not found
    """
    if path is None:
        path = DEFAULT_ORIGINAL_DATASET_PATH
    
    # Resolve path
    dataset_path = Path(path)
    if not dataset_path.is_absolute():
        if base_dir is not None:
            dataset_path = base_dir / path
        else:
            # Try relative to Tools directory
            tools_dir = Path(__file__).parents[2]  # src/scoring -> Tools
            dataset_path = tools_dir / path
    
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Original dataset not found at: {dataset_path}\n"
            f"Please ensure the dataset exists at: {DEFAULT_ORIGINAL_DATASET_PATH}"
        )
    
    # Load based on extension
    suffix = dataset_path.suffix.lower()
    
    if suffix in ['.xlsx', '.xls']:
        df = pd.read_excel(dataset_path)
    elif suffix == '.csv':
        # Try different encodings
        for encoding in ['utf-8', 'latin-1', 'cp1252']:
            try:
                df = pd.read_csv(dataset_path, encoding=encoding)
                break
            except UnicodeDecodeError:
                continue
        else:
            raise ValueError(f"Could not read CSV with any encoding: {dataset_path}")
    elif suffix == '.parquet':
        df = pd.read_parquet(dataset_path)
    else:
        raise ValueError(f"Unsupported file format: {suffix}")
    
    return df


def extract_score_variables(
    df: pd.DataFrame,
    indices: Optional[np.ndarray] = None,
    config: Optional[ScoreDataConfig] = None,
) -> pd.DataFrame:
    """Extract variables needed for clinical score calculations.
    
    Args:
        df: DataFrame (original dataset with raw values)
        indices: Row indices to extract. If None, extracts all rows.
        config: Configuration specifying which variables to extract
        
    Returns:
        DataFrame with only the score-relevant variables
    """
    if config is None:
        config = ScoreDataConfig()
    
    # Find which variables are actually present
    available_vars = [v for v in config.preserve_variables if v in df.columns]
    
    # Also include any column with 'killip' in the name (case insensitive)
    for col in df.columns:
        if 'killip' in col.lower() and col not in available_vars:
            available_vars.append(col)
    
    if not available_vars:
        raise ValueError(
            f"No score variables found in dataset. "
            f"Expected some of: {config.preserve_variables[:10]}..."
        )
    
    # Extract subset
    if indices is not None:
        result = df.loc[indices, available_vars].copy()
    else:
        result = df[available_vars].copy()
    
    return result


def save_testset_score_data(
    original_df: pd.DataFrame,
    test_indices: np.ndarray,
    output_dir: Path,
    task: str,
    timestamp: str,
    config: Optional[ScoreDataConfig] = None,
) -> Path:
    """Save original data for test set indices, for clinical score calculations.
    
    This function extracts the original (unencoded) values for the test set
    that are needed for GRACE, RECUIMA, and other clinical score comparisons.
    
    Args:
        original_df: Original dataset with raw values
        test_indices: Indices of test set rows (from original DataFrame)
        output_dir: Directory to save the file
        task: Task name (e.g., 'mortality')
        timestamp: Timestamp string for filename
        config: Configuration for which variables to preserve
        
    Returns:
        Path to saved file
    """
    if config is None:
        config = ScoreDataConfig()
    
    # Extract score variables for test indices
    score_data = extract_score_variables(original_df, test_indices, config)
    
    # Add original index as a column for matching
    score_data['_original_index'] = test_indices
    
    # Save
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / f"testset_{task}_scores_original_{timestamp}.parquet"
    score_data.to_parquet(output_path, index=False)
    
    # Also save metadata about what was preserved
    metadata = {
        'timestamp': timestamp,
        'task': task,
        'n_samples': len(test_indices),
        'preserved_variables': list(score_data.columns),
        'config': config.to_dict(),
    }
    
    metadata_path = output_dir / f"testset_{task}_scores_original_{timestamp}.json"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    return output_path


def load_testset_score_data(
    testsets_dir: Path,
    task: str = "mortality",
) -> Optional[pd.DataFrame]:
    """Load the most recent original score data for a task's test set.
    
    Args:
        testsets_dir: Directory containing testsets
        task: Task name
        
    Returns:
        DataFrame with original score data, or None if not found
    """
    testsets_dir = Path(testsets_dir)
    
    if not testsets_dir.exists():
        return None
    
    # Find most recent file
    pattern = f"testset_{task}_scores_original_*.parquet"
    files = sorted(
        testsets_dir.glob(pattern),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )
    
    if not files:
        return None
    
    return pd.read_parquet(files[0])


def get_testset_original_indices(
    testset_path: Path,
    score_data_path: Optional[Path] = None,
) -> Optional[np.ndarray]:
    """Get original dataset indices for a test set.
    
    Args:
        testset_path: Path to preprocessed testset
        score_data_path: Path to original score data (optional, will search if not provided)
        
    Returns:
        Array of original indices, or None if not available
    """
    if score_data_path is None:
        # Try to find corresponding score data file
        testset_dir = testset_path.parent
        testset_name = testset_path.stem
        
        # Extract task from filename (e.g., "testset_mortality_20240101_120000")
        parts = testset_name.split('_')
        if len(parts) >= 2:
            task = parts[1]
            score_data = load_testset_score_data(testset_dir, task)
            if score_data is not None and '_original_index' in score_data.columns:
                return score_data['_original_index'].values
    else:
        score_data = pd.read_parquet(score_data_path)
        if '_original_index' in score_data.columns:
            return score_data['_original_index'].values
    
    return None


def check_score_data_availability(
    testsets_dir: Path,
    task: str = "mortality",
) -> Dict[str, Any]:
    """Check if original score data is available for a test set.
    
    Args:
        testsets_dir: Directory containing testsets
        task: Task name
        
    Returns:
        Dictionary with availability information
    """
    result = {
        'available': False,
        'path': None,
        'n_samples': 0,
        'variables': [],
        'grace_ready': False,
        'recuima_ready': False,
        'message': '',
    }
    
    score_data = load_testset_score_data(testsets_dir, task)
    
    if score_data is None:
        result['message'] = (
            "Original score data not found. "
            "Re-train the model to generate score data, or load the original dataset manually."
        )
        return result
    
    result['available'] = True
    result['n_samples'] = len(score_data)
    result['variables'] = [c for c in score_data.columns if not c.startswith('_')]
    
    # Check GRACE readiness
    grace_vars = set(GRACE_REQUIRED_VARIABLES)
    available_vars = set(result['variables'])
    grace_missing = grace_vars - available_vars
    result['grace_ready'] = len(grace_missing) == 0
    
    # Check RECUIMA readiness
    recuima_core = {'edad', 'presion_arterial_sistolica', 'filtrado_glomerular', 
                   'indice_killip', 'complicaciones'}
    recuima_missing = recuima_core - available_vars
    result['recuima_ready'] = len(recuima_missing) == 0
    
    if result['grace_ready'] and result['recuima_ready']:
        result['message'] = "✅ All clinical score variables available"
    else:
        missing_msg = []
        if not result['grace_ready']:
            missing_msg.append(f"GRACE missing: {grace_missing}")
        if not result['recuima_ready']:
            missing_msg.append(f"RECUIMA missing: {recuima_missing}")
        result['message'] = "; ".join(missing_msg)
    
    return result
