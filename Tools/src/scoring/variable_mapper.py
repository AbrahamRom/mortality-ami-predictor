"""Central variable mapping for clinical scores.

This module handles the mapping between dataset column names and
the standardized variable names required by clinical score calculators.

Example:
    Original dataset might have: 'EDAD', 'FC', 'TA_SISTOLICA'
    This mapper converts to: 'edad', 'frecuencia_cardiaca', 'presion_arterial_sistolica'
"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import pandas as pd
import numpy as np


@dataclass
class MappingResult:
    """Result of variable mapping operation."""
    success: bool
    mapped_df: Optional[pd.DataFrame] = None
    mapped_columns: Dict[str, str] = field(default_factory=dict)  # original -> mapped
    unmapped_required: List[str] = field(default_factory=list)
    unmapped_optional: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def report(self) -> str:
        """Generate human-readable mapping report."""
        lines = ["=" * 80, "VARIABLE MAPPING REPORT", "=" * 80, ""]
        
        lines.append(f"✓ Mapped columns ({len(self.mapped_columns)}):")
        for orig, mapped in self.mapped_columns.items():
            lines.append(f"  '{orig}' → '{mapped}'")
        
        if self.unmapped_required:
            lines.append(f"\n✗ MISSING REQUIRED ({len(self.unmapped_required)}):")
            for var in self.unmapped_required:
                lines.append(f"  - {var}")
        
        if self.unmapped_optional:
            lines.append(f"\n⚠ Missing optional ({len(self.unmapped_optional)}):")
            for var in self.unmapped_optional:
                lines.append(f"  - {var}")
        
        if self.warnings:
            lines.append(f"\n⚠ Warnings:")
            for w in self.warnings:
                lines.append(f"  - {w}")
        
        lines.extend(["", "=" * 80])
        return "\n".join(lines)


class VariableMapper:
    """Maps dataset columns to standardized score variable names."""
    
    # Column name aliases (case-insensitive)
    DEFAULT_ALIASES = {
        # Age
        "age": ["edad", "age", "años", "annos", "age_years", "patient_age"],
        
        # Heart Rate
        "heart_rate": ["frecuencia_cardiaca", "heart_rate", "hr", "fc", "freq_cardiaca", "pulse"],
        
        # Systolic BP
        "systolic_bp": [
            "presion_arterial_sistolica", "tas", "systolic_bp", "sbp", 
            "pa_sistolica", "systolic", "bp_systolic"
        ],
        
        # Creatinine
        "creatinine": ["creatinina", "creatinine", "cr", "creat", "serum_creatinine"],
        
        # Killip Class
        "killip_class": [
            "indice_killip", "killip", "killip_class", "killip_kimball", 
            "clase_killip", "killip_index"
        ],
        
        # Troponin/Elevated Enzymes
        "elevated_enzymes": [
            "tropnina_hs", "troponina", "troponin", "elevated_enzymes", 
            "troponin_i", "troponin_t", "enzimas_elevadas", "ck", "ckmb"
        ],
        
        # ST Deviation
        "st_deviation": [
            "depresion_st", "st_deviation", "supradesnivel", "infradesnivel", 
            "st_segment", "st_change"
        ],
        
        # Cardiac Arrest
        "cardiac_arrest": [
            "comp_pcr", "paro_cardiaco", "cardiac_arrest", "paro", "cpb", "cardiac_action"
        ],
        
        # GFR (Glomerular Filtration Rate)
        "gfr": [
            "filtrado_glomerular", "gfr", "tasa_filtrado_glomerular", 
            "fg", "tfg", "egfr"
        ],
        
        # Complications (text field)
        "complications": [
            "complicaciones", "complications", "comp", "adverse_events"
        ],
        
        # VF/VT (Ventricular Fibrillation/Tachycardia)
        "vf_vt": [
            "comp_fv_tv", "fv_tv", "fibrilacion_ventricular", "vf_vt", 
            "ventricular_fibrillation", "ventricular_arrhythmia"
        ],
        
        # High-grade AVB (AV Block)
        "avb_high_grade": [
            "comp_bav_alto_grado", "bav_alto_grado", "high_grade_avb", 
            "avb", "av_block", "bloqueo_av"
        ],
        
        # Diabetes
        "diabetes": ["diabetes_mellitus", "diabetes", "dm", "diabetes_type2"],
        
        # Hypertension
        "hypertension": [
            "hipertension_arterial", "hipertension", "hta", "hypertension", "htn"
        ],
        
        # Weight
        "weight": ["peso", "weight", "kg", "body_weight", "weight_kg"],
        
        # Ischemia Time
        "ischemia_time": [
            "tiempo_isquemia", "ischemia_time", "time_to_treatment", 
            "symptom_onset_time", "door_to_balloon"
        ],
        
        # Anterior MI / LBBB
        "anterior_mi": [
            "anterior_wall_mi", "anterior", "anterior_infarction", "anterior_mi",
            "anterior_mi_or_lbbb", "supradesnivel"
        ],
        
        # ECG Leads Affected
        "ecg_leads_affected": [
            "ecg_leads_affected", "num_leads_affected", "leads_affected",
            "v1", "v2", "v3", "v4", "v5", "v6", "avf", "avl", "d1", "d2", "d3"
        ],
        
        # Angina Episodes in 24h
        "angina_episodes_24h": [
            "angina_episodes_24h", "angina24h", "angina_num_episodes",
            "angina_episodes", "angina_freq_24h"
        ],
        
        # Cardiac Markers
        "cardiac_markers_elevated": [
            "cardiac_markers_elevated", "troponin_elevated", "enzimas_elevadas",
            "tropnina_hs", "troponina", "markers_elevated", "elevated_markers", "ck", "ckmb"
        ],
        
        # Cholesterol Abnormal
        "cholesterol_abnormal": [
            "hiperlipoproteinemia", "cholesterol_abnormal", "cholesterol_high", "dyslipidemia"
        ],
        
        # Known Coronary Disease
        "known_coronary_disease": [
            "enfermedad_arterias_coronarias", "known_coronary_disease", "prior_cad", "prior_stenosis",
            "coronary_stenosis_50", "conocido_cad"
        ],
        
        # Aspirin Use (7 days)
        "aspirin_use_7days": [
            "asa", "aspirin_use_7days", "aspirin_recent", "aspirin_use_7d"
        ],
        
        # Smoker
        "smoker": ["tabaquismo", "smoker", "smoking", "fumador", "fuma"],
        
        # Target (outcome)
        "target": [
            "mortality_inhospital", "exitus", "estado_vital", "mortality", 
            "death", "outcome", "muerte"
        ],
    }
    
    # Score-specific required variables
    # IMPORTANT: These must match the actual score.compute() function signatures
    SCORE_REQUIREMENTS = {
        # GRACE Score (Fox et al. 2007)
        # Required: age, heart_rate, systolic_bp, creatinine, killip_class, elevated_enzymes
        # Optional: st_deviation, cardiac_arrest
        "grace": [
            "age", "heart_rate", "systolic_bp", "creatinine",
            "killip_class", "elevated_enzymes"
        ],
        "grace_optional": [
            "st_deviation", "cardiac_arrest"
        ],
        
        # TIMI-STEMI Score (Morrow et al. 2000)
        # Required: age, systolic_bp, heart_rate, killip_class, anterior_mi_or_lbbb, weight_kg, time_to_treatment_hours
        # Optional: diabetes, hypertension, angina_prior
        "timi_stemi": [
            "age", "systolic_bp", "heart_rate", "killip_class",
            "anterior_mi", "weight", "ischemia_time"
        ],
        "timi_stemi_optional": [
            "diabetes", "hypertension"
        ],
        
        # TIMI-NSTEMI Score (Antman et al. 2000)
        # Core variables needed: age, st_deviation, cardiac_markers_elevated, aspirin_use_7days
        # Plus risk factors: diabetes, hypertension, cholesterol_abnormal, known_coronary_disease, angina_episodes_24h
        "timi_nstemi": [
            "age", "st_deviation", "cardiac_markers_elevated", "aspirin_use_7days",
            "angina_episodes_24h"
        ],
        "timi_nstemi_optional": [
            "diabetes", "hypertension", "cholesterol_abnormal", "known_coronary_disease", "smoker"
        ],
        
        # RECUIMA Score (Santos Medina - Cuban Registry)
        # Required: age, systolic_bp, gfr, ecg_leads_affected, killip_class
        # Optional: vf_vt (ventricular fibrillation/tachycardia), high_grade_avb (AV block)
        "recuima": [
            "age", "systolic_bp", "gfr", "ecg_leads_affected", "killip_class"
        ],
        "recuima_optional": [
            "complications"  # vf_vt and high_grade_avb are parsed from complications text
        ],
    }
    
    def __init__(self, aliases: Optional[Dict[str, List[str]]] = None):
        """Initialize mapper with optional custom aliases.
        
        Args:
            aliases: Dict of {standard_name: [column_aliases]}
                    If None, uses DEFAULT_ALIASES
        """
        self.aliases = aliases or self.DEFAULT_ALIASES
    
    def _get_relevant_variables(self, score_type: Optional[str] = None) -> List[str]:
        """Get all relevant variables for a score type or all variables.
        
        Args:
            score_type: One of 'grace', 'timi_stemi', 'timi_nstemi', 'recuima', or None
            
        Returns:
            List of standard variable names to search
        """
        if score_type is None:
            return list(self.aliases.keys())
        
        if score_type not in self.SCORE_REQUIREMENTS:
            return list(self.aliases.keys())
        
        required = self.SCORE_REQUIREMENTS[score_type]
        optional = self.SCORE_REQUIREMENTS.get(f"{score_type}_optional", [])
        return list(set(required) | set(optional))
        
    def find_column(
        self,
        df: pd.DataFrame,
        standard_name: str,
        score_type: Optional[str] = None
    ) -> Optional[str]:
        """Find a dataset column matching a standard variable name.
        
        Args:
            df: Input DataFrame
            standard_name: Standard variable name (e.g., 'age')
            score_type: Optional score type to constrain search to relevant variables
                       One of 'grace', 'timi_stemi', 'timi_nstemi', 'recuima', or None
            
        Returns:
            Matched column name from df, or None if not found
        """
        if standard_name not in self.aliases:
            return None
        
        # If score_type specified, check if this variable is relevant
        if score_type is not None:
            relevant_vars = self._get_relevant_variables(score_type)
            if standard_name not in relevant_vars:
                return None
        
        # Create case-insensitive column lookup
        columns_lower = {c.lower(): c for c in df.columns}
        
        # Try each alias
        for alias in self.aliases[standard_name]:
            if alias.lower() in columns_lower:
                return columns_lower[alias.lower()]
        
        return None
    
    def map_dataframe(
        self,
        df: pd.DataFrame,
        score_type: str,
        check_only: bool = False
    ) -> MappingResult:
        """Map a DataFrame to standard variable names for a score.
        
        Args:
            df: Input DataFrame with arbitrary column names
            score_type: One of 'grace', 'timi_stemi', 'timi_nstemi', 'recuima'
            check_only: If True, don't modify df, just check availability
            
        Returns:
            MappingResult with mapped DataFrame and detailed report
        """
        result = MappingResult(success=False)
        
        # Validate score_type
        if score_type not in self.SCORE_REQUIREMENTS:
            result.warnings.append(f"Unknown score type: {score_type}")
            return result
        
        required = self.SCORE_REQUIREMENTS[score_type]
        optional = self.SCORE_REQUIREMENTS.get(f"{score_type}_optional", [])
        
        # Find matching columns
        mapped_cols = {}
        missing_required = []
        missing_optional = []
        
        for std_name in required:
            found_col = self.find_column(df, std_name, score_type)
            if found_col:
                mapped_cols[found_col] = std_name
            else:
                missing_required.append(std_name)
        
        for std_name in optional:
            found_col = self.find_column(df, std_name, score_type)
            if found_col:
                mapped_cols[found_col] = std_name
            else:
                missing_optional.append(std_name)
        
        result.mapped_columns = mapped_cols
        result.unmapped_required = missing_required
        result.unmapped_optional = missing_optional
        
        # Check success
        if missing_required:
            result.success = False
            result.warnings.append(
                f"Missing {len(missing_required)} required variables for {score_type}"
            )
            return result
        
        result.success = True
        
        # If check_only, don't modify df
        if check_only:
            return result
        
        # Rename columns
        rename_map = {orig: std for orig, std in mapped_cols.items()}
        result.mapped_df = df.rename(columns=rename_map)
        
        return result


def validate_mapped_dataframe(
    df: pd.DataFrame,
    required_cols: List[str],
    optional_cols: Optional[List[str]] = None,
) -> Dict[str, str]:
    """Validate that mapped DataFrame has expected columns and types.
    
    Args:
        df: Mapped DataFrame
        required_cols: List of required column names
        optional_cols: List of optional column names
        
    Returns:
        Dict of validation results per column
    """
    results = {}
    
    for col in required_cols:
        if col not in df.columns:
            results[col] = "ERROR: Missing required column"
        elif df[col].isna().all():
            results[col] = "ERROR: All values are NaN"
        elif df[col].isna().sum() > 0:
            pct = (df[col].isna().sum() / len(df)) * 100
            results[col] = f"WARNING: {pct:.1f}% missing values"
        else:
            results[col] = "OK"
    
    if optional_cols:
        for col in optional_cols:
            if col not in df.columns:
                results[col] = "NOT PRESENT (optional)"
            elif df[col].isna().all():
                results[col] = "WARNING: All values are NaN (optional)"
            else:
                results[col] = "OK"
    
    return results
