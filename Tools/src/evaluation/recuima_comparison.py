"""Rigorous statistical comparison between ML models and RECUIMA score.

This module provides comprehensive comparison tools for the RECUIMA scale
(Registro Cubano de Infarto - escala de Mortalidad Intrahospitalaria),
developed by Dr. Maikel Santos Medina.

The RECUIMA scale uses 7 predictors:
- Age > 70 years (1 point)
- Systolic BP < 100 mmHg (1 point)
- GFR < 60 ml/min/1.73m² (3 points) - NOTE: Most important predictor with highest OR
- More than 7 ECG leads affected (1 point)
- Killip-Kimball IV (1 point)
- Ventricular fibrillation/tachycardia (2 points)
- High-grade AV block (1 point)

Based on: Tesis Doctoral - Universidad de Ciencias Médicas de Santiago de Cuba
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    brier_score_loss,
    confusion_matrix,
)
from sklearn.calibration import calibration_curve


# RECUIMA score point weights based on thesis
RECUIMA_WEIGHTS = {
    'age_gt_70': 1,
    'sbp_lt_100': 1,
    'gfr_lt_60': 3,  # Highest weight - most important predictor
    'ecg_leads_gt_7': 1,
    'killip_iv': 1,
    'vf_vt': 2,
    'high_grade_avb': 1,
}

# Maximum score
RECUIMA_MAX_SCORE = sum(RECUIMA_WEIGHTS.values())  # 10 points

# Risk categories based on thesis
RECUIMA_LOW_RISK_THRESHOLD = 3  # Score <= 3 is low risk
RECUIMA_HIGH_RISK_THRESHOLD = 4  # Score >= 4 is high risk


@dataclass
class RECUIMAComparisonResult:
    """Results from comparing ML model with RECUIMA score."""
    
    # Model names (required fields first)
    model_name: str
    
    # AUC values
    model_auc: float
    recuima_auc: float
    auc_difference: float
    auc_p_value: float
    auc_ci_lower: float
    auc_ci_upper: float
    
    # NRI (Net Reclassification Improvement)
    nri: float
    nri_p_value: float
    nri_events: float  # NRI for events (cases)
    nri_nonevents: float  # NRI for non-events (controls)
    
    # IDI (Integrated Discrimination Improvement)
    idi: float
    idi_p_value: float
    
    # Calibration metrics
    model_brier: float
    recuima_brier: float
    brier_difference: float
    
    # Additional metrics
    model_accuracy: float
    recuima_accuracy: float
    model_sensitivity: float
    recuima_sensitivity: float
    model_specificity: float
    recuima_specificity: float
    
    # Statistical conclusion
    is_model_superior: bool
    superiority_level: str  # "significant", "marginal", "none", "inferior"
    
    # Fields with defaults come last
    recuima_name: str = "RECUIMA Score"
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'model_name': self.model_name,
            'recuima_name': self.recuima_name,
            'model_auc': self.model_auc,
            'recuima_auc': self.recuima_auc,
            'auc_difference': self.auc_difference,
            'auc_p_value': self.auc_p_value,
            'auc_ci_lower': self.auc_ci_lower,
            'auc_ci_upper': self.auc_ci_upper,
            'nri': self.nri,
            'nri_p_value': self.nri_p_value,
            'nri_events': self.nri_events,
            'nri_nonevents': self.nri_nonevents,
            'idi': self.idi,
            'idi_p_value': self.idi_p_value,
            'model_brier': self.model_brier,
            'recuima_brier': self.recuima_brier,
            'brier_difference': self.brier_difference,
            'model_accuracy': self.model_accuracy,
            'recuima_accuracy': self.recuima_accuracy,
            'model_sensitivity': self.model_sensitivity,
            'recuima_sensitivity': self.recuima_sensitivity,
            'model_specificity': self.model_specificity,
            'recuima_specificity': self.recuima_specificity,
            'is_model_superior': self.is_model_superior,
            'superiority_level': self.superiority_level,
            'p_value': self.auc_p_value,
        }

    @property
    def p_value(self) -> float:
        """Backward-compatible alias for the primary p-value (AUC p-value)."""
        return float(self.auc_p_value)


# Column mappings for RECUIMA variables in the dataset
RECUIMA_COLUMN_MAPPINGS = {
    # Age
    'age': ['edad', 'age', 'años', 'annos'],
    
    # Systolic blood pressure
    'systolic_bp': [
        'presion_arterial_sistolica', 
        'presion_arterial_sistolica.1',
        'tas', 
        'systolic_bp', 
        'sbp',
        'pa_sistolica',
    ],
    
    # Glomerular filtration rate
    'gfr': [
        'filtrado_glomerular', 
        'gfr', 
        'tasa_filtrado_glomerular',
        'fg',
        'tfg',
    ],
    
    # ECG leads (individual derivations to count)
    'ecg_leads': [
        # Standard leads
        'v1', 'v2', 'v3', 'v4', 'v5', 'v6',
        'd1', 'd2', 'd3',
        'avf', 'avl', 'avc',
        # Alternative names
        'lead_v1', 'lead_v2', 'lead_v3', 'lead_v4', 'lead_v5', 'lead_v6',
        'lead_i', 'lead_ii', 'lead_iii',
        'lead_avf', 'lead_avl', 'lead_avc',
    ],
    
    # Killip class
    'killip_class': [
        'indice_killip',
        'killip',
        'killip_class',
        'killip_kimball',
        'clase_killip',
    ],
    
    # Ventricular fibrillation/tachycardia - complicaciones contains FV/TV
    'vf_vt': [
        'fv_tv',
        'fibrilacion_ventricular',
        'taquicardia_ventricular',
        'arritmia_ventricular',
        'vf',
        'vt',
        # In complicaciones field: may need to decode
    ],
    
    # High-grade AV block
    'avb': [
        'bav',
        'bloqueo_av',
        'bloqueo_auriculoventricular',
        'avb',
        'high_grade_avb',
        'bav_alto_grado',
    ],
}


def find_column(df: pd.DataFrame, column_type: str) -> Optional[str]:
    """Find a column in the dataframe based on possible column names.
    
    Args:
        df: DataFrame to search
        column_type: Type of column to find (e.g., 'age', 'systolic_bp')
        
    Returns:
        Column name if found, None otherwise
    """
    candidates = RECUIMA_COLUMN_MAPPINGS.get(column_type, [])
    
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    
    # Try case-insensitive search
    df_columns_lower = {col.lower(): col for col in df.columns}
    for candidate in candidates:
        if candidate.lower() in df_columns_lower:
            return df_columns_lower[candidate.lower()]
    
    return None


def count_ecg_leads_affected(df: pd.DataFrame) -> np.ndarray:
    """Count the number of ECG leads affected (with ST changes) per patient.
    
    Counts derivations with non-zero values. The dataset contains:
    v1-v6, v7-v9, d1-d3, avl, avf, avr (15 leads total)
    
    Args:
        df: DataFrame with ECG lead columns
        
    Returns:
        Array with count of affected leads per patient (0-15)
    """
    # All ECG leads including v7-v9 as per dataset structure
    ecg_lead_columns = ['v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7', 'v8', 'v9',
                        'd1', 'd2', 'd3', 
                        'avf', 'avl', 'avr']
    
    found_columns = [col for col in ecg_lead_columns if col in df.columns]
    
    if not found_columns:
        # Try uppercase
        found_columns = [col for col in df.columns if col.upper() in [c.upper() for c in ecg_lead_columns]]
    
    if not found_columns:
        return np.zeros(len(df))
    
    # Count leads with non-zero values (affected leads)
    # The dataset contains numeric values, not binary - count how many are != 0
    return (df[found_columns].fillna(0) != 0).sum(axis=1).values


def check_recuima_requirements(df: pd.DataFrame) -> Tuple[bool, Dict[str, str], List[str]]:
    """Check if the dataset has the required columns for RECUIMA calculation.
    
    Args:
        df: DataFrame to check
        
    Returns:
        Tuple of (can_calculate, found_columns, missing_requirements)
    """
    found_columns = {}
    missing_requirements = []
    
    # Check required columns
    required_types = ['age', 'systolic_bp', 'gfr', 'killip_class']
    
    for col_type in required_types:
        col = find_column(df, col_type)
        if col:
            found_columns[col_type] = col
        else:
            missing_requirements.append(col_type)
    
    # Check ECG leads
    ecg_leads = count_ecg_leads_affected(df)
    if ecg_leads.sum() > 0:
        found_columns['ecg_leads'] = 'multiple_columns'
    else:
        # Check if there's a pre-computed column
        for candidate in ['derivaciones_afectadas', 'ecg_leads_affected', 'num_leads']:
            if candidate in df.columns:
                found_columns['ecg_leads'] = candidate
                break
        else:
            missing_requirements.append('ecg_leads')
    
    # Check optional columns (FV/TV and AVB) - not required but add to score
    vf_col = find_column(df, 'vf_vt')
    if vf_col:
        found_columns['vf_vt'] = vf_col
    else:
        # Check if we can extract from complicaciones column
        if 'complicaciones' in df.columns:
            found_columns['vf_vt'] = 'parse_complicaciones'
    
    avb_col = find_column(df, 'avb')
    if avb_col:
        found_columns['avb'] = avb_col
    else:
        # Check if we can extract from complicaciones column
        if 'complicaciones' in df.columns:
            found_columns['avb'] = 'parse_complicaciones'
    
    can_calculate = len(missing_requirements) == 0
    
    return can_calculate, found_columns, missing_requirements


def parse_complications_for_vf_tv(df: pd.DataFrame) -> np.ndarray:
    """Extract VF/TV (ventricular fibrillation/tachycardia) from complicaciones column.
    
    Looks for patterns like 'FV', 'TV', 'fibrilacion ventricular', etc.
    """
    if 'complicaciones' not in df.columns:
        return np.zeros(len(df))
    
    comp = df['complicaciones'].fillna('').astype(str).str.upper()
    # Match FV or TV as whole words or part of compound expressions
    has_vf_tv = comp.str.contains(r'\bFV\b|\bTV\b', regex=True, na=False)
    return has_vf_tv.astype(int).values


def parse_complications_for_avb(df: pd.DataFrame) -> np.ndarray:
    """Extract high-grade AV block from complicaciones column.
    
    Looks for patterns like 'BAV3', 'BAV 2:1', 'BAV2', 'bloqueo AV', etc.
    """
    if 'complicaciones' not in df.columns:
        return np.zeros(len(df))
    
    comp = df['complicaciones'].fillna('').astype(str).str.upper()
    # Match BAV3, BAV2:1, BAV-3, etc. (high-grade blocks)
    has_avb = comp.str.contains(r'BAV\s*3|BAV\s*2\s*:\s*1|BAV\-?3', regex=True, na=False)
    return has_avb.astype(int).values


def _to_numeric_safe(series: pd.Series) -> pd.Series:
    """Safely convert a series to numeric, handling comma decimals."""
    if series.dtype in ['float64', 'int64', 'float32', 'int32']:
        return series
    # Convert to string first, replace comma with dot, then to numeric
    return pd.to_numeric(
        series.astype(str).str.replace(',', '.', regex=False),
        errors='coerce'
    )


def compute_recuima_scores(
    df: pd.DataFrame,
    column_mapping: Optional[Dict[str, str]] = None
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """Compute RECUIMA scores for all patients in the DataFrame.
    
    Args:
        df: DataFrame with patient data
        column_mapping: Optional mapping of column types to column names
        
    Returns:
        Tuple of (scores, probabilities, components_dict)
    """
    n_patients = len(df)
    scores = np.zeros(n_patients)
    components = {key: np.zeros(n_patients) for key in RECUIMA_WEIGHTS.keys()}
    
    # Auto-detect columns if not provided
    if column_mapping is None:
        _, column_mapping, _ = check_recuima_requirements(df)
    
    # Age > 70 years
    age_col = column_mapping.get('age')
    if age_col and age_col in df.columns:
        age_values = _to_numeric_safe(df[age_col])
        age_points = (age_values > 70).astype(int) * RECUIMA_WEIGHTS['age_gt_70']
        scores += age_points.fillna(0)
        components['age_gt_70'] = age_points.fillna(0).values
    
    # Systolic BP < 100 mmHg
    sbp_col = column_mapping.get('systolic_bp')
    if sbp_col and sbp_col in df.columns:
        sbp_values = _to_numeric_safe(df[sbp_col])
        sbp_points = (sbp_values < 100).astype(int) * RECUIMA_WEIGHTS['sbp_lt_100']
        scores += sbp_points.fillna(0)
        components['sbp_lt_100'] = sbp_points.fillna(0).values
    
    # GFR < 60 ml/min/1.73m² (most important - 3 points)
    gfr_col = column_mapping.get('gfr')
    if gfr_col and gfr_col in df.columns:
        gfr_values = _to_numeric_safe(df[gfr_col])
        gfr_points = (gfr_values < 60).astype(int) * RECUIMA_WEIGHTS['gfr_lt_60']
        scores += gfr_points.fillna(0)
        components['gfr_lt_60'] = gfr_points.fillna(0).values
    
    # ECG leads > 7
    ecg_col = column_mapping.get('ecg_leads')
    if ecg_col == 'multiple_columns':
        ecg_count = count_ecg_leads_affected(df)
    elif ecg_col and ecg_col in df.columns:
        ecg_count = df[ecg_col].values
    else:
        ecg_count = np.zeros(n_patients)
    
    ecg_points = (ecg_count > 7).astype(int) * RECUIMA_WEIGHTS['ecg_leads_gt_7']
    scores += ecg_points
    components['ecg_leads_gt_7'] = ecg_points
    
    # Killip-Kimball IV - handle both numeric (4) and string ('IV')
    killip_col = column_mapping.get('killip_class')
    if killip_col and killip_col in df.columns:
        killip_values = df[killip_col]
        # Check if values are strings or numeric
        if killip_values.dtype == 'object':
            # String comparison: 'IV', 'iv', etc.
            is_killip_iv = killip_values.astype(str).str.strip().str.upper() == 'IV'
        else:
            # Numeric comparison
            is_killip_iv = killip_values == 4
        killip_points = is_killip_iv.astype(int) * RECUIMA_WEIGHTS['killip_iv']
        scores += killip_points.fillna(0)
        components['killip_iv'] = killip_points.fillna(0).values
    
    # Ventricular fibrillation/tachycardia (2 points)
    vf_col = column_mapping.get('vf_vt')
    if vf_col == 'parse_complicaciones':
        # Extract from complicaciones column
        vf_values = parse_complications_for_vf_tv(df)
        vf_points = vf_values * RECUIMA_WEIGHTS['vf_vt']
        scores += vf_points
        components['vf_vt'] = vf_points
    elif vf_col and vf_col in df.columns:
        vf_values = _to_numeric_safe(df[vf_col])
        vf_points = (vf_values > 0).astype(int) * RECUIMA_WEIGHTS['vf_vt']
        scores += vf_points.fillna(0)
        components['vf_vt'] = vf_points.fillna(0).values
    
    # High-grade AV block
    avb_col = column_mapping.get('avb')
    if avb_col == 'parse_complicaciones':
        # Extract from complicaciones column
        avb_values = parse_complications_for_avb(df)
        avb_points = avb_values * RECUIMA_WEIGHTS['high_grade_avb']
        scores += avb_points
        components['high_grade_avb'] = avb_points
    elif avb_col and avb_col in df.columns:
        avb_values = _to_numeric_safe(df[avb_col])
        avb_points = (avb_values > 0).astype(int) * RECUIMA_WEIGHTS['high_grade_avb']
        scores += avb_points.fillna(0)
        components['high_grade_avb'] = avb_points.fillna(0).values
    
    # Convert scores to probabilities using logistic function
    # Based on thesis validation data
    probabilities = score_to_probability(scores)
    
    return scores, probabilities, components


def score_to_probability(scores: np.ndarray) -> np.ndarray:
    """Convert RECUIMA scores to mortality probabilities.
    
    Uses an estimated logistic function based on thesis validation data.
    
    Args:
        scores: Array of RECUIMA scores (0-10)
        
    Returns:
        Array of mortality probabilities (0-1)
    """
    # Probability mapping based on thesis data
    # These are approximate values from the validation cohorts
    probability_map = {
        0: 0.02,
        1: 0.05,
        2: 0.10,
        3: 0.20,  # Low risk threshold
        4: 0.35,  # High risk starts
        5: 0.50,
        6: 0.65,
        7: 0.78,
        8: 0.88,
        9: 0.94,
        10: 0.98,
    }
    
    probabilities = np.zeros_like(scores, dtype=float)
    for i, score in enumerate(scores):
        score_int = min(10, max(0, int(score)))
        probabilities[i] = probability_map.get(score_int, 0.98)
    
    return probabilities


def delong_test(y_true: np.ndarray, pred1: np.ndarray, pred2: np.ndarray) -> Tuple[float, float]:
    """DeLong test for comparing two correlated ROC curves.
    
    Args:
        y_true: True labels
        pred1: Predictions from model 1
        pred2: Predictions from model 2
        
    Returns:
        Tuple of (z_statistic, p_value)
    """
    # Validate paired samples requirement for DeLong test (correlated ROC assumption)
    assert len(pred1) == len(pred2), (
        f"DeLong test requires paired samples (same n). "
        f"pred1 has length {len(pred1)}, pred2 has length {len(pred2)}. "
        f"Lengths must match for valid ROC comparison."
    )
    
    auc1 = roc_auc_score(y_true, pred1)
    auc2 = roc_auc_score(y_true, pred2)
    
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]
    
    n_pos = len(pos_idx)
    n_neg = len(neg_idx)
    
    def structural_components(y, pred, pos_idx, neg_idx):
        v10 = np.zeros(len(pos_idx))
        for i, pos_i in enumerate(pos_idx):
            v10[i] = np.mean(pred[pos_i] > pred[neg_idx]) + 0.5 * np.mean(pred[pos_i] == pred[neg_idx])
        return v10
    
    v10_1 = structural_components(y_true, pred1, pos_idx, neg_idx)
    v10_2 = structural_components(y_true, pred2, pos_idx, neg_idx)
    
    v01_1 = 1 - structural_components(y_true, -pred1, neg_idx, pos_idx)
    v01_2 = 1 - structural_components(y_true, -pred2, neg_idx, pos_idx)
    
    s10_1 = np.var(v10_1, ddof=1)
    s10_2 = np.var(v10_2, ddof=1)
    s01_1 = np.var(v01_1, ddof=1)
    s01_2 = np.var(v01_2, ddof=1)
    
    cov10 = np.cov(v10_1, v10_2, ddof=1)[0, 1]
    cov01 = np.cov(v01_1, v01_2, ddof=1)[0, 1]
    
    var_diff = (s10_1 / n_pos) + (s10_2 / n_pos) + (s01_1 / n_neg) + (s01_2 / n_neg) - 2 * (cov10 / n_pos + cov01 / n_neg)
    
    if var_diff <= 0:
        return 0.0, 1.0
    
    z = (auc1 - auc2) / np.sqrt(var_diff)
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    
    return z, p_value


def compute_nri(
    y_true: np.ndarray,
    pred_model: np.ndarray,
    pred_recuima: np.ndarray,
    threshold: float = 0.5,
) -> Tuple[float, float, float, float]:
    """Compute Net Reclassification Improvement (NRI).
    
    Args:
        y_true: True labels
        pred_model: Predictions from ML model
        pred_recuima: Predictions from RECUIMA score
        threshold: Classification threshold
        
    Returns:
        Tuple of (nri_total, nri_events, nri_nonevents, p_value)
    """
    # Events (y=1)
    events_mask = y_true == 1
    model_events = pred_model[events_mask]
    recuima_events = pred_recuima[events_mask]
    
    moved_up_events = np.sum(model_events > recuima_events)
    moved_down_events = np.sum(model_events < recuima_events)
    nri_events = (moved_up_events - moved_down_events) / np.sum(events_mask)
    
    # Non-events (y=0)
    nonevents_mask = y_true == 0
    model_nonevents = pred_model[nonevents_mask]
    recuima_nonevents = pred_recuima[nonevents_mask]
    
    moved_down_nonevents = np.sum(model_nonevents < recuima_nonevents)
    moved_up_nonevents = np.sum(model_nonevents > recuima_nonevents)
    nri_nonevents = (moved_down_nonevents - moved_up_nonevents) / np.sum(nonevents_mask)
    
    nri_total = nri_events + nri_nonevents
    
    n_events = np.sum(y_true == 1)
    n_nonevents = np.sum(y_true == 0)
    se_nri = np.sqrt((nri_events ** 2 / n_events) + (nri_nonevents ** 2 / n_nonevents))
    
    if se_nri > 0:
        z_stat = nri_total / se_nri
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    else:
        p_value = 1.0
    
    return nri_total, nri_events, nri_nonevents, p_value


def compute_idi(
    y_true: np.ndarray,
    pred_model: np.ndarray,
    pred_recuima: np.ndarray
) -> Tuple[float, float]:
    """Compute Integrated Discrimination Improvement (IDI).
    
    Args:
        y_true: True labels
        pred_model: Predictions from ML model
        pred_recuima: Predictions from RECUIMA score
        
    Returns:
        Tuple of (idi, p_value)
    """
    events_mask = y_true == 1
    nonevents_mask = y_true == 0
    
    model_events_mean = np.mean(pred_model[events_mask])
    recuima_events_mean = np.mean(pred_recuima[events_mask])
    
    model_nonevents_mean = np.mean(pred_model[nonevents_mask])
    recuima_nonevents_mean = np.mean(pred_recuima[nonevents_mask])
    
    is_model = model_events_mean - model_nonevents_mean
    is_recuima = recuima_events_mean - recuima_nonevents_mean
    
    idi = is_model - is_recuima
    
    n_events = np.sum(events_mask)
    n_nonevents = np.sum(nonevents_mask)
    
    se_events = np.sqrt(np.var(pred_model[events_mask]) / n_events + np.var(pred_recuima[events_mask]) / n_events)
    se_nonevents = np.sqrt(np.var(pred_model[nonevents_mask]) / n_nonevents + np.var(pred_recuima[nonevents_mask]) / n_nonevents)
    se_idi = np.sqrt(se_events**2 + se_nonevents**2)
    
    if se_idi > 0:
        z_stat = idi / se_idi
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    else:
        p_value = 1.0
    
    return idi, p_value


def compare_with_recuima(
    y_true: np.ndarray,
    y_pred_model: np.ndarray,
    y_pred_recuima: np.ndarray,
    model_name: str = "ML Model",
    threshold: float = 0.5,
    alpha: float = 0.05
) -> RECUIMAComparisonResult:
    """Comprehensive comparison of ML model with RECUIMA score.
    
    Args:
        y_true: True labels
        y_pred_model: Predictions from ML model (probabilities)
        y_pred_recuima: Predictions from RECUIMA score (probabilities)
        model_name: Name of the ML model
        threshold: Classification threshold
        alpha: Significance level for statistical tests
        
    Returns:
        RECUIMAComparisonResult object with all metrics and test results
    """
    # AUC comparison with DeLong test
    auc_model = roc_auc_score(y_true, y_pred_model)
    auc_recuima = roc_auc_score(y_true, y_pred_recuima)
    auc_diff = auc_model - auc_recuima
    
    z_stat, p_value_auc = delong_test(y_true, y_pred_model, y_pred_recuima)
    
    se_diff = abs(auc_diff / z_stat) if z_stat != 0 else 0
    ci_lower = auc_diff - 1.96 * se_diff
    ci_upper = auc_diff + 1.96 * se_diff
    
    # NRI
    nri_total, nri_events, nri_nonevents, p_value_nri = compute_nri(
        y_true, y_pred_model, y_pred_recuima, threshold
    )
    
    # IDI
    idi, p_value_idi = compute_idi(y_true, y_pred_model, y_pred_recuima)
    
    # Calibration - Brier score
    brier_model = brier_score_loss(y_true, y_pred_model)
    brier_recuima = brier_score_loss(y_true, y_pred_recuima)
    brier_diff = brier_model - brier_recuima
    
    # Confusion matrix metrics
    y_pred_model_binary = (y_pred_model >= threshold).astype(int)
    y_pred_recuima_binary = (y_pred_recuima >= threshold).astype(int)
    
    cm_model = confusion_matrix(y_true, y_pred_model_binary)
    cm_recuima = confusion_matrix(y_true, y_pred_recuima_binary)
    
    tn_m, fp_m, fn_m, tp_m = cm_model.ravel() if cm_model.size == 4 else (0, 0, 0, 0)
    tn_r, fp_r, fn_r, tp_r = cm_recuima.ravel() if cm_recuima.size == 4 else (0, 0, 0, 0)
    
    acc_model = (tp_m + tn_m) / (tp_m + tn_m + fp_m + fn_m) if (tp_m + tn_m + fp_m + fn_m) > 0 else 0
    acc_recuima = (tp_r + tn_r) / (tp_r + tn_r + fp_r + fn_r) if (tp_r + tn_r + fp_r + fn_r) > 0 else 0
    
    sens_model = tp_m / (tp_m + fn_m) if (tp_m + fn_m) > 0 else 0
    sens_recuima = tp_r / (tp_r + fn_r) if (tp_r + fn_r) > 0 else 0
    
    spec_model = tn_m / (tn_m + fp_m) if (tn_m + fp_m) > 0 else 0
    spec_recuima = tn_r / (tn_r + fp_r) if (tn_r + fp_r) > 0 else 0
    
    # Determine superiority
    is_superior = auc_diff > 0 and p_value_auc < alpha
    
    if p_value_auc < 0.001:
        superiority_level = "highly_significant"
    elif p_value_auc < 0.01:
        superiority_level = "significant"
    elif p_value_auc < 0.05:
        superiority_level = "marginal"
    elif auc_diff > 0:
        superiority_level = "favorable_trend"
    elif auc_diff < 0:
        superiority_level = "inferior"
    else:
        superiority_level = "equivalent"
    
    return RECUIMAComparisonResult(
        model_name=model_name,
        recuima_name="RECUIMA Score",
        model_auc=auc_model,
        recuima_auc=auc_recuima,
        auc_difference=auc_diff,
        auc_p_value=p_value_auc,
        auc_ci_lower=ci_lower,
        auc_ci_upper=ci_upper,
        nri=nri_total,
        nri_p_value=p_value_nri,
        nri_events=nri_events,
        nri_nonevents=nri_nonevents,
        idi=idi,
        idi_p_value=p_value_idi,
        model_brier=brier_model,
        recuima_brier=brier_recuima,
        brier_difference=brier_diff,
        model_accuracy=acc_model,
        recuima_accuracy=acc_recuima,
        model_sensitivity=sens_model,
        recuima_sensitivity=sens_recuima,
        model_specificity=spec_model,
        recuima_specificity=spec_recuima,
        is_model_superior=is_superior,
        superiority_level=superiority_level,
    )


def plot_roc_comparison_recuima(
    y_true: np.ndarray,
    y_pred_model: np.ndarray,
    y_pred_recuima: np.ndarray,
    comparison_result: RECUIMAComparisonResult
) -> go.Figure:
    """Plot ROC curves comparison between model and RECUIMA.
    
    Args:
        y_true: True labels
        y_pred_model: Model predictions
        y_pred_recuima: RECUIMA predictions
        comparison_result: RECUIMAComparisonResult object
        
    Returns:
        Plotly Figure
    """
    fpr_model, tpr_model, _ = roc_curve(y_true, y_pred_model)
    fpr_recuima, tpr_recuima, _ = roc_curve(y_true, y_pred_recuima)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=fpr_model,
        y=tpr_model,
        mode='lines',
        name=f'{comparison_result.model_name} (AUC={comparison_result.model_auc:.3f})',
        line=dict(color='blue', width=3),
        hovertemplate='FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=fpr_recuima,
        y=tpr_recuima,
        mode='lines',
        name=f'RECUIMA Score (AUC={comparison_result.recuima_auc:.3f})',
        line=dict(color='green', width=3, dash='dash'),
        hovertemplate='FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        name='Random',
        line=dict(color='gray', width=1, dash='dot'),
        hoverinfo='skip'
    ))
    
    annotation_text = f"ΔAUC = {comparison_result.auc_difference:+.3f}<br>"
    annotation_text += f"p-value = {comparison_result.p_value:.4f}<br>"
    
    if comparison_result.is_model_superior:
        annotation_text += "✅ <b>Modelo es estadísticamente superior</b>"
        annotation_color = "green"
    elif comparison_result.superiority_level == "inferior":
        annotation_text += "⚠️ <b>Modelo es inferior a RECUIMA</b>"
        annotation_color = "red"
    else:
        annotation_text += "ℹ️ Sin diferencia significativa"
        annotation_color = "orange"
    
    fig.add_annotation(
        x=0.6, y=0.2,
        text=annotation_text,
        showarrow=False,
        bgcolor=annotation_color,
        opacity=0.8,
        font=dict(color="white", size=12),
        bordercolor="white",
        borderwidth=2
    )
    
    fig.update_layout(
        title=f'Comparación ROC: {comparison_result.model_name} vs RECUIMA Score',
        xaxis=dict(title='Tasa de Falsos Positivos', range=[0, 1]),
        yaxis=dict(title='Tasa de Verdaderos Positivos', range=[0, 1]),
        width=800,
        height=600,
        template='plotly_white',
        hovermode='closest',
        legend=dict(x=0.7, y=0.1)
    )
    
    return fig


def plot_calibration_comparison_recuima(
    y_true: np.ndarray,
    y_pred_model: np.ndarray,
    y_pred_recuima: np.ndarray,
    comparison_result: RECUIMAComparisonResult,
    n_bins: int = 10
) -> go.Figure:
    """Plot calibration curves comparison.
    
    Args:
        y_true: True labels
        y_pred_model: Model predictions
        y_pred_recuima: RECUIMA predictions
        comparison_result: RECUIMAComparisonResult object
        n_bins: Number of bins for calibration
        
    Returns:
        Plotly Figure
    """
    try:
        prob_true_model, prob_pred_model = calibration_curve(
            y_true, y_pred_model, n_bins=n_bins, strategy='uniform'
        )
        prob_true_recuima, prob_pred_recuima = calibration_curve(
            y_true, y_pred_recuima, n_bins=n_bins, strategy='uniform'
        )
    except ValueError:
        return go.Figure().add_annotation(
            text="Datos insuficientes para análisis de calibración",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        name='Calibración Perfecta',
        line=dict(color='gray', width=2, dash='dash'),
        hoverinfo='skip'
    ))
    
    fig.add_trace(go.Scatter(
        x=prob_pred_model,
        y=prob_true_model,
        mode='lines+markers',
        name=f'{comparison_result.model_name} (Brier={comparison_result.model_brier:.3f})',
        line=dict(color='blue', width=3),
        marker=dict(size=10),
        hovertemplate='Predicho: %{x:.3f}<br>Observado: %{y:.3f}<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=prob_pred_recuima,
        y=prob_true_recuima,
        mode='lines+markers',
        name=f'RECUIMA Score (Brier={comparison_result.recuima_brier:.3f})',
        line=dict(color='green', width=3, dash='dot'),
        marker=dict(size=10, symbol='square'),
        hovertemplate='Predicho: %{x:.3f}<br>Observado: %{y:.3f}<extra></extra>'
    ))
    
    fig.update_layout(
        title='Comparación de Calibración',
        xaxis=dict(title='Probabilidad Predicha', range=[0, 1]),
        yaxis=dict(title='Frecuencia Observada', range=[0, 1]),
        width=800,
        height=600,
        template='plotly_white',
        hovermode='closest',
        legend=dict(x=0.05, y=0.95)
    )
    
    return fig


def plot_metrics_comparison_recuima(comparison_result: RECUIMAComparisonResult) -> go.Figure:
    """Plot bar chart comparing all metrics.
    
    Args:
        comparison_result: RECUIMAComparisonResult object
        
    Returns:
        Plotly Figure
    """
    metrics = ['AUC', 'Accuracy', 'Sensibilidad', 'Especificidad']
    model_values = [
        comparison_result.model_auc,
        comparison_result.model_accuracy,
        comparison_result.model_sensitivity,
        comparison_result.model_specificity
    ]
    recuima_values = [
        comparison_result.recuima_auc,
        comparison_result.recuima_accuracy,
        comparison_result.recuima_sensitivity,
        comparison_result.recuima_specificity
    ]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name=comparison_result.model_name,
        x=metrics,
        y=model_values,
        marker_color='blue',
        text=[f'{v:.3f}' for v in model_values],
        textposition='auto',
        hovertemplate='%{x}: %{y:.3f}<extra></extra>'
    ))
    
    fig.add_trace(go.Bar(
        name='RECUIMA Score',
        x=metrics,
        y=recuima_values,
        marker_color='green',
        text=[f'{v:.3f}' for v in recuima_values],
        textposition='auto',
        hovertemplate='%{x}: %{y:.3f}<extra></extra>'
    ))
    
    fig.update_layout(
        title='Comparación de Métricas de Rendimiento',
        yaxis=dict(title='Score', range=[0, 1.1]),
        barmode='group',
        width=800,
        height=500,
        template='plotly_white',
        showlegend=True
    )
    
    return fig


def plot_nri_idi_recuima(comparison_result: RECUIMAComparisonResult) -> go.Figure:
    """Plot NRI and IDI results.
    
    Args:
        comparison_result: RECUIMAComparisonResult object
        
    Returns:
        Plotly Figure
    """
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Net Reclassification Improvement', 'Integrated Discrimination Improvement'),
        specs=[[{"type": "bar"}, {"type": "indicator"}]]
    )
    
    fig.add_trace(
        go.Bar(
            x=['Eventos', 'No-Eventos', 'Total'],
            y=[comparison_result.nri_events, comparison_result.nri_nonevents, comparison_result.nri],
            marker_color=['green', 'orange', 'blue'],
            text=[f'{comparison_result.nri_events:.3f}', 
                  f'{comparison_result.nri_nonevents:.3f}',
                  f'{comparison_result.nri:.3f}'],
            textposition='auto',
            hovertemplate='%{x}: %{y:.3f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Indicator(
            mode="number+delta",
            value=comparison_result.idi,
            delta={'reference': 0, 'relative': False},
            title={'text': f"IDI<br><span style='font-size:0.8em'>p={comparison_result.idi_p_value:.4f}</span>"},
            domain={'x': [0, 1], 'y': [0, 1]}
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        height=400,
        showlegend=False,
        template='plotly_white'
    )
    
    fig.update_yaxes(title_text="NRI", row=1, col=1)
    
    return fig


def generate_comparison_report_recuima(comparison_result: RECUIMAComparisonResult) -> pd.DataFrame:
    """Generate comprehensive comparison report as DataFrame.
    
    Args:
        comparison_result: RECUIMAComparisonResult object
        
    Returns:
        DataFrame with all comparison metrics
    """
    report_data = [
        {
            'Métrica': 'AUC',
            'Modelo': f'{comparison_result.model_auc:.4f}',
            'RECUIMA': f'{comparison_result.recuima_auc:.4f}',
            'Diferencia': f'{comparison_result.auc_difference:+.4f}',
            'P-value': f'{comparison_result.auc_p_value:.4f}',
            'IC 95%': f'[{comparison_result.auc_ci_lower:.4f}, {comparison_result.auc_ci_upper:.4f}]'
        },
        {
            'Métrica': 'Accuracy',
            'Modelo': f'{comparison_result.model_accuracy:.4f}',
            'RECUIMA': f'{comparison_result.recuima_accuracy:.4f}',
            'Diferencia': f'{comparison_result.model_accuracy - comparison_result.recuima_accuracy:+.4f}',
            'P-value': '-',
            'IC 95%': '-'
        },
        {
            'Métrica': 'Sensibilidad',
            'Modelo': f'{comparison_result.model_sensitivity:.4f}',
            'RECUIMA': f'{comparison_result.recuima_sensitivity:.4f}',
            'Diferencia': f'{comparison_result.model_sensitivity - comparison_result.recuima_sensitivity:+.4f}',
            'P-value': '-',
            'IC 95%': '-'
        },
        {
            'Métrica': 'Especificidad',
            'Modelo': f'{comparison_result.model_specificity:.4f}',
            'RECUIMA': f'{comparison_result.recuima_specificity:.4f}',
            'Diferencia': f'{comparison_result.model_specificity - comparison_result.recuima_specificity:+.4f}',
            'P-value': '-',
            'IC 95%': '-'
        },
        {
            'Métrica': 'Brier Score',
            'Modelo': f'{comparison_result.model_brier:.4f}',
            'RECUIMA': f'{comparison_result.recuima_brier:.4f}',
            'Diferencia': f'{comparison_result.brier_difference:+.4f}',
            'P-value': '-',
            'IC 95%': '-'
        },
        {
            'Métrica': 'NRI (Total)',
            'Modelo': f'{comparison_result.nri:.4f}',
            'RECUIMA': '-',
            'Diferencia': f'{comparison_result.nri:+.4f}',
            'P-value': f'{comparison_result.nri_p_value:.4f}',
            'IC 95%': '-'
        },
        {
            'Métrica': 'IDI',
            'Modelo': f'{comparison_result.idi:.4f}',
            'RECUIMA': '-',
            'Diferencia': f'{comparison_result.idi:+.4f}',
            'P-value': f'{comparison_result.idi_p_value:.4f}',
            'IC 95%': '-'
        },
    ]
    
    return pd.DataFrame(report_data)


def get_recuima_info() -> Dict:
    """Get information about the RECUIMA score.
    
    Returns:
        Dictionary with RECUIMA score information
    """
    return {
        "name": "RECUIMA",
        "full_name": "Registro Cubano de Infarto - Mortalidad Intrahospitalaria",
        "author": "Dr. Maikel Santos Medina",
        "institution": "Hospital General Docente Dr. Ernesto Guevara de la Serna, Las Tunas",
        "thesis": "Tesis Doctoral - Universidad de Ciencias Médicas de Santiago de Cuba",
        "variables": [
            {"name": "Edad > 70 años", "points": 1, "type": "binary"},
            {"name": "TAS < 100 mmHg", "points": 1, "type": "binary"},
            {"name": "Filtrado glomerular < 60 ml/min/1.73m²", "points": 3, "type": "binary"},
            {"name": "Más de 7 derivaciones ECG afectadas", "points": 1, "type": "binary"},
            {"name": "Killip-Kimball IV", "points": 1, "type": "binary"},
            {"name": "FV/TV (arritmias ventriculares)", "points": 2, "type": "binary"},
            {"name": "BAV alto grado", "points": 1, "type": "binary"},
        ],
        "max_score": RECUIMA_MAX_SCORE,
        "risk_categories": {
            "low": {"range": "0-3", "description": "Bajo riesgo de mortalidad"},
            "high": {"range": "4-10", "description": "Alto riesgo de mortalidad"},
        },
        "validation": {
            "auc": "0.890-0.904",
            "cohorts": 3,
            "total_patients": 2348,
            "comparison_with_grace": "Estadísticamente superior (p < 0.05)",
        },
        "advantages": [
            "No requiere troponinas (disponibles en países de bajos recursos)",
            "No requiere coronariografía",
            "Variables clínicas disponibles al ingreso",
            "Validado en población cubana/latinoamericana",
            "Mayor especificidad que GRACE (87.70% vs 47.38%)",
        ],
    }
