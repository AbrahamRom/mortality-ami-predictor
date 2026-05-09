"""
RECUIMA Score Implementation.

Escala predictiva de muerte hospitalaria por infarto agudo de miocardio
desarrollada por el Dr. Maikel Santos Medina.

Based on: Tesis Doctoral - Universidad de Ciencias Médicas de Santiago de Cuba
Hospital General Docente "Dr. Ernesto Guevara de la Serna" Las Tunas

Variables (máximo 10 puntos):
- Edad > 70 años (1 pt)
- TAS < 100 mmHg (1 pt)
- Filtrado glomerular < 60 ml/min/1.73m² (3 pts) ⭐
- Más de 7 derivaciones afectadas en ECG (1 pt)
- Killip Kimball IV (choque cardiogénico) (1 pt)
- Fibrilación ventricular/Taquicardia ventricular (2 pts)
- Bloqueo auriculoventricular de alto grado (1 pt)

Categorías de riesgo:
- Bajo: ≤3 puntos
- Alto: ≥4 puntos
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Tuple, Optional

import pandas as pd

from .validators import (
    ComputeResult,
    validate_recuima_inputs,
    validate_numeric,
    validate_bool,
    validate_int,
    is_missing,
)


# =============================================================================
# COMPLICACIONES PARSING DEFINITIONS
# =============================================================================

# Tokens indicating Ventricular Fibrillation
FV_TOKENS = {'fv', 'pcr-fv'}

# Tokens indicating Ventricular Tachycardia  
TV_TOKENS = {'tv'}

# Tokens indicating High-Grade AV Block (2nd Mobitz II, 3rd degree, 2:1)
BAV_ALTO_GRADO_TOKENS = {'bav3', 'bav2:1', 'bav2', 'bav', 'pcr-bav-pericar'}


def parse_complicaciones(text: str) -> Tuple[bool, bool]:
    """
    Parse complicaciones text to extract FV/TV and BAV alto grado.
    
    Args:
        text: The complicaciones text field (e.g., "tv;bav3;shock")
        
    Returns:
        Tuple of (has_fv_tv, has_bav_alto_grado)
    """
    if pd.isna(text) or not str(text).strip():
        return False, False
    
    text_lower = str(text).lower().strip()
    tokens = set(re.split(r'[;,/|]+', text_lower))
    tokens = {t.strip() for t in tokens if t.strip()}
    
    has_fv_tv = bool(tokens & FV_TOKENS) or bool(tokens & TV_TOKENS)
    has_bav_alto_grado = bool(tokens & BAV_ALTO_GRADO_TOKENS)
    
    return has_fv_tv, has_bav_alto_grado


@dataclass
class RECUIMAResult:
    """Result from RECUIMA score calculation."""
    
    score: int
    risk_category: str
    probability: float
    components: dict[str, int]
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "score": self.score,
            "risk_category": self.risk_category,
            "probability": self.probability,
            "components": self.components,
        }


class RECUIMAScorer:
    """
    RECUIMA Score Calculator.
    
    Escala predictiva de muerte hospitalaria por infarto agudo de miocardio
    desarrollada en Cuba a partir del Registro Cubano de Infarto (RECUIMA).
    
    Categorías de riesgo:
    - Bajo: Score bajo
    - Alto: Score alto
    """
    
    def __init__(self) -> None:
        """Initialize RECUIMA scorer."""
        self.name = "RECUIMA"
        self.version = "1.0"
        self.author = "Dr. Maikel Santos Medina"
        self.institution = "Hospital General Docente Dr. Ernesto Guevara de la Serna, Las Tunas"
    
    def compute(
        self,
        age: float,
        systolic_bp: float,
        gfr: float,
        ecg_leads_affected: int,
        killip_class: int,
        vf_vt: bool = False,
        high_grade_avb: bool = False,
    ) -> dict[str, Any]:
        """
        Calculate RECUIMA score.
        
        Parameters
        ----------
        age : float
            Patient age in years
        systolic_bp : float
            Systolic blood pressure in mmHg
        gfr : float
            Glomerular filtration rate in ml/min/1.73m²
        ecg_leads_affected : int
            Number of ECG leads with ST changes (0-12)
        killip_class : int
            Killip-Kimball class (1-4)
        vf_vt : bool
            Presence of ventricular fibrillation or ventricular tachycardia
        high_grade_avb : bool
            Presence of high-grade atrioventricular block
        
        Returns
        -------
        dict
            Score result with score, risk category, and components
        """
        components = {}
        score = 0
        
        # Age > 70 years
        if age > 70:
            components["age_gt_70"] = 1
            score += 1
        else:
            components["age_gt_70"] = 0
        
        # Systolic BP < 100 mmHg
        if systolic_bp < 100:
            components["sbp_lt_100"] = 1
            score += 1
        else:
            components["sbp_lt_100"] = 0
        
        # GFR < 60 ml/min/1.73m² (3 points - highest weight per thesis)
        if gfr < 60:
            components["gfr_lt_60"] = 3
            score += 3
        else:
            components["gfr_lt_60"] = 0
        
        # More than 7 ECG leads affected
        if ecg_leads_affected > 7:
            components["ecg_leads_gt_7"] = 1
            score += 1
        else:
            components["ecg_leads_gt_7"] = 0
        
        # Killip-Kimball IV (cardiogenic shock)
        if killip_class == 4:
            components["killip_iv"] = 1
            score += 1
        else:
            components["killip_iv"] = 0
        
        # Ventricular fibrillation / Ventricular tachycardia (2 points per thesis)
        if vf_vt:
            components["vf_vt"] = 2
            score += 2
        else:
            components["vf_vt"] = 0
        
        # High-grade AV block
        if high_grade_avb:
            components["high_grade_avb"] = 1
            score += 1
        else:
            components["high_grade_avb"] = 0
        
        # Determine risk category (binary: low vs high)
        # Based on thesis: Bajo riesgo ≤3, Alto riesgo ≥4
        risk_category = "high" if score >= 4 else "low"
        
        # Estimate probability based on score (0-10 with thesis weights)
        # These are approximate values based on thesis validation data
        probability_map = {
            0: 0.02,
            1: 0.05,
            2: 0.10,
            3: 0.20,
            4: 0.35,  # High risk threshold (≥4)
            5: 0.50,
            6: 0.65,
            7: 0.78,
            8: 0.88,
            9: 0.94,
            10: 0.98,
        }
        probability = probability_map.get(min(score, 10), 0.98)
        
        return RECUIMAResult(
            score=score,
            risk_category=risk_category,
            probability=probability,
            components=components,
        ).to_dict()
    
    def compute_safe(
        self,
        age: Any = None,
        systolic_bp: Any = None,
        gfr: Any = None,
        ecg_leads_affected: Any = None,
        killip_class: Any = None,
        complications: Any = None,
        vf_vt: Any = False,
        high_grade_avb: Any = False,
        **kwargs
    ) -> Dict[str, Any]:
        """Compute RECUIMA score with input validation.
        
        Validates all inputs before computation. Returns structured result
        with error/warning information.
        
        Args:
            age: Patient age (years) - REQUIRED
            systolic_bp: Systolic blood pressure in mmHg - REQUIRED
            gfr: Glomerular filtration rate in ml/min/1.73m² - REQUIRED
            ecg_leads_affected: Number of ECG leads with ST changes - REQUIRED
            killip_class: Killip-Kimball class (1-4) - REQUIRED
            complications: Text field containing complications (e.g., "tv;bav3")
                          If provided, will parse to extract vf_vt and high_grade_avb
            vf_vt: Ventricular fibrillation or tachycardia (bool) - OPTIONAL (overridden by complications)
            high_grade_avb: High-grade AV block (bool) - OPTIONAL (overridden by complications)
            **kwargs: Unexpected variables (will trigger warning)
            
        Returns:
            Dict with keys:
            - 'success': bool (True if computation successful)
            - 'score': int or None
            - 'risk_category': str or None
            - 'errors': list of error messages
            - 'warnings': list of warning messages
            - 'data': dict with full result if successful
        """
        result = ComputeResult()
        
        # STEP 5: Check for unexpected variables (variable isolation validation)
        if kwargs:
            unexpected = list(kwargs.keys())
            result.warnings.append(
                f"RECUIMA: Received unexpected variables: {', '.join(unexpected)}. "
                f"These will be ignored. Expected only: age, systolic_bp, gfr, ecg_leads_affected, "
                f"killip_class, complications, vf_vt, high_grade_avb"
            )
        
        # Validate required variables
        validation = validate_recuima_inputs(
            age, systolic_bp, gfr, ecg_leads_affected, killip_class
        )
        result.errors.extend(validation.errors)
        result.warnings.extend(validation.warnings)
        
        if not validation.is_valid:
            result.success = False
            return result.to_dict()
        
        # Convert validated values
        age_val = float(age)
        sbp_val = float(systolic_bp)
        gfr_val = float(gfr)
        ecg_val = int(ecg_leads_affected)
        killip_val = int(killip_class)
        
        # Handle complications field - if provided, parse it (takes precedence)
        if complications is not None and not is_missing(complications):
            parsed_vf_vt, parsed_bav = parse_complicaciones(str(complications))
            vf_val = parsed_vf_vt
            bav_val = parsed_bav
            if parsed_vf_vt or parsed_bav:
                result.warnings.append(
                    f"Parsed complications field: VF/VT={vf_val}, High-grade AVB={bav_val}"
                )
        else:
            # Validate optional boolean variables when not using complications field
            is_valid_vf, vf_val = validate_bool(vf_vt, "vf_vt", validation)
            is_valid_bav, bav_val = validate_bool(high_grade_avb, "high_grade_avb", validation)
            
            result.errors.extend(validation.errors)
            result.warnings.extend(validation.warnings)
            
            if not (is_valid_vf and is_valid_bav):
                result.success = False
                return result.to_dict()
        
        # Computation is safe - call regular compute()
        try:
            compute_result = self.compute(
                age=age_val,
                systolic_bp=sbp_val,
                gfr=gfr_val,
                ecg_leads_affected=ecg_val,
                killip_class=killip_val,
                vf_vt=vf_val,
                high_grade_avb=bav_val,
            )
            
            result.success = True
            result.score = compute_result['score']
            result.risk_category = compute_result['risk_category']
            result.data = compute_result
            
        except Exception as e:
            result.success = False
            result.errors.append(f"Computation error: {str(e)}")
        
        return result.to_dict()
    
    def compute_batch(
        self,
        age: "np.ndarray",
        systolic_bp: "np.ndarray",
        gfr: "np.ndarray",
        ecg_leads_affected: "np.ndarray",
        killip_class: "np.ndarray",
        vf_vt: "np.ndarray",
        high_grade_avb: "np.ndarray",
    ) -> "np.ndarray":
        """
        Calculate RECUIMA scores for multiple patients.
        
        Parameters
        ----------
        age : np.ndarray
            Array of patient ages
        systolic_bp : np.ndarray
            Array of systolic blood pressures
        gfr : np.ndarray
            Array of GFR values
        ecg_leads_affected : np.ndarray
            Array of ECG leads affected counts
        killip_class : np.ndarray
            Array of Killip classes (1-4)
        vf_vt : np.ndarray
            Array of VF/VT indicators (bool or 0/1)
        high_grade_avb : np.ndarray
            Array of high-grade AVB indicators (bool or 0/1)
        
        Returns
        -------
        np.ndarray
            Array of RECUIMA scores
        """
        import numpy as np
        
        scores = []
        for i in range(len(age)):
            result = self.compute(
                age=float(age[i]),
                systolic_bp=float(systolic_bp[i]),
                gfr=float(gfr[i]),
                ecg_leads_affected=int(ecg_leads_affected[i]),
                killip_class=int(killip_class[i]),
                vf_vt=bool(vf_vt[i]),
                high_grade_avb=bool(high_grade_avb[i]),
            )
            scores.append(result["score"])
        
        return np.array(scores)
    
    def get_info(self) -> dict[str, Any]:
        """Get score information."""
        return {
            "name": self.name,
            "version": self.version,
            "author": self.author,
            "institution": self.institution,
            "description": (
                "Escala predictiva de muerte hospitalaria por infarto agudo de miocardio "
                "desarrollada a partir del Registro Cubano de Infarto (RECUIMA)"
            ),
            "variables": [
                "Edad > 70 años",
                "TAS < 100 mmHg",
                "Filtrado glomerular < 60 ml/min/1.73m²",
                "Más de 7 derivaciones afectadas en ECG",
                "Killip Kimball IV",
                "Fibrilación ventricular/Taquicardia ventricular",
                "Bloqueo auriculoventricular de alto grado",
            ],
            "risk_categories": ["low", "high"],
            "max_score": 10,  # With thesis weights: 1+1+3+1+1+2+1 = 10
            "risk_threshold": 4,  # Score ≥4 = Alto riesgo
        }


def compute_recuima_from_dataframe(
    df: pd.DataFrame,
    age_col: str = "edad",
    sbp_col: str = "presion_arterial_sistolica",
    gfr_col: str = "filtrado_glomerular",
    killip_col: str = "indice_killip",
    complicaciones_col: str = "complicaciones",
    ecg_lead_cols: Optional[list] = None,
    comp_fv_tv_col: Optional[str] = None,
    comp_bav_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Compute RECUIMA scores from a DataFrame.
    
    This function handles parsing of 'complicaciones' column if needed,
    and computes RECUIMA scores for all rows.
    
    Args:
        df: Input DataFrame
        age_col: Column name for age
        sbp_col: Column name for systolic BP
        gfr_col: Column name for GFR
        killip_col: Column name for Killip class
        complicaciones_col: Column for raw complicaciones text
        ecg_lead_cols: List of ECG lead columns (to count affected leads)
        comp_fv_tv_col: Pre-parsed FV/TV column (if available)
        comp_bav_col: Pre-parsed BAV alto grado column (if available)
    
    Returns:
        DataFrame with added columns:
        - recuima_score: The computed score (0-10)
        - recuima_risk: Risk category ("low" or "high")
    """
    import numpy as np
    
    result_df = df.copy()
    n = len(df)
    
    # Initialize score array
    scores = np.zeros(n, dtype=int)
    
    # 1. Age > 70 (1 pt)
    if age_col in df.columns:
        scores += (df[age_col] > 70).astype(int)
    
    # 2. SBP < 100 (1 pt)
    if sbp_col in df.columns:
        scores += (df[sbp_col] < 100).astype(int)
    
    # 3. GFR < 60 (3 pts) - Most important factor
    if gfr_col in df.columns:
        scores += (df[gfr_col] < 60).astype(int) * 3
    
    # 4. ECG leads > 7 (1 pt)
    if ecg_lead_cols:
        available_leads = [c for c in ecg_lead_cols if c in df.columns]
        if available_leads:
            leads_count = df[available_leads].notna().sum(axis=1)
            scores += (leads_count > 7).astype(int)
    
    # 5. Killip IV (1 pt)
    if killip_col in df.columns:
        killip_iv = df[killip_col].apply(
            lambda x: 1 if (str(x).upper().strip() in ('IV', '4')) else 0
        )
        scores += killip_iv
    
    # 6 & 7. FV/TV (2 pts) and BAV alto grado (1 pt)
    if comp_fv_tv_col and comp_fv_tv_col in df.columns:
        scores += df[comp_fv_tv_col].astype(int) * 2
    elif complicaciones_col in df.columns:
        parsed = df[complicaciones_col].apply(parse_complicaciones)
        fv_tv = parsed.apply(lambda x: x[0]).astype(int)
        scores += fv_tv * 2
    
    if comp_bav_col and comp_bav_col in df.columns:
        scores += df[comp_bav_col].astype(int)
    elif complicaciones_col in df.columns:
        if 'parsed' not in dir():
            parsed = df[complicaciones_col].apply(parse_complicaciones)
        bav = parsed.apply(lambda x: x[1]).astype(int)
        scores += bav
    
    # Store results
    result_df['recuima_score'] = scores
    result_df['recuima_risk'] = np.where(scores >= 4, 'high', 'low')
    
    return result_df