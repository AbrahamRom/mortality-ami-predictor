"""
TIMI Risk Score Implementation (Real, Clinical-Grade).

References:
- Antman, E. M., et al. (2000). The TIMI risk score for unstable angina/
  non-ST elevation MI: A method for prognostication and therapeutic decision
  making. JAMA, 284(7), 835-842.
- Morrow, D. A., et al. (2000). TIMI risk score for ST-elevation myocardial
  infarction. Circulation, 102(17), 2031-2037.

Implements:
1. TIMI Score for ST-Elevation MI (STEMI/IAMCEST)
2. TIMI Score for Non-ST-Elevation MI (NSTEMI/IAMSEST)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple
import warnings

import numpy as np
import pandas as pd

from .validators import (
    ComputeResult,
    validate_timi_stemi_inputs,
    validate_timi_nstemi_inputs,
    validate_numeric,
    validate_bool,
    validate_int,
    validate_required_vars,
    ValidationResult,
    is_missing,
)


# =============================================================================
# TIMI STEMI (ST-Elevation MI) - Morrow et al. 2000
# =============================================================================

class TIMISTEMIScore:
    """
    TIMI Risk Score for ST-Elevation Myocardial Infarction (STEMI).
    
    Predicts in-hospital and 30-day mortality in STEMI patients.
    Designed for rapid bedside assessment at presentation.
    
    Score range: 0-14 points
    Mortality increases exponentially with score.
    """
    
    # Risk categories from Morrow et al. 2000
    MORTALITY_RATES = {
        0: 0.008,    # <1%
        1: 0.010,
        2: 0.015,
        3: 0.020,
        4: 0.030,
        5: 0.045,
        6: 0.070,
        7: 0.110,
        8: 0.165,
        9: 0.245,
        10: 0.350,
        11: 0.480,
        12: 0.630,
        13: 0.790,
        14: 0.900,
    }
    
    def __init__(self):
        """Initialize TIMI STEMI score calculator."""
        self.name = "TIMI-STEMI"
        self.version = "2.0"
        self.description = "TIMI Risk Score for ST-Elevation MI (Morrow et al. 2000)"
        self.score_type = "STEMI"
    
    def compute(
        self,
        age: float,
        systolic_bp: float,
        heart_rate: float,
        killip_class: int,
        anterior_mi_or_lbbb: bool = False,
        weight_kg: float = 80.0,
        time_to_treatment_hours: float = 2.0,
        diabetes: bool = False,
        hypertension: bool = False,
        angina_prior: bool = False,
    ) -> Dict[str, Any]:
        """
        Calculate TIMI STEMI score.
        
        Parameters
        ----------
        age : float
            Patient age in years
        systolic_bp : float
            Systolic blood pressure in mmHg
        heart_rate : float
            Heart rate in bpm
        killip_class : int
            Killip-Kimball class (I, II, III, IV or 1, 2, 3, 4)
        anterior_mi_or_lbbb : bool
            Anterior MI location OR left bundle branch block (LBBB)
        weight_kg : float
            Body weight in kg (default 80 for estimation)
        time_to_treatment_hours : float
            Time from symptom onset to first medical treatment (hours)
        diabetes : bool
            History of diabetes mellitus
        hypertension : bool
            History of hypertension
        angina_prior : bool
            Prior angina pectoris
            
        Returns
        -------
        dict
            {
                'score': int,
                'mortality_rate': float,
                'risk_category': str,
                'components': dict of component weights
            }
        """
        components = {}
        score = 0
        
        # 1. Age
        if age >= 75:
            components['age_geq_75'] = 3
            score += 3
        elif age >= 65:
            components['age_65_74'] = 2
            score += 2
        else:
            components['age_lt_65'] = 0
        
        # 2. Diabetes, hypertension, or prior angina (any one = 1 point)
        if diabetes or hypertension or angina_prior:
            components['diabetes_htn_angina'] = 1
            score += 1
        else:
            components['diabetes_htn_angina'] = 0
        
        # 3. Systolic blood pressure < 100 mmHg
        if systolic_bp < 100:
            components['sbp_lt_100'] = 3
            score += 3
        else:
            components['sbp_lt_100'] = 0
        
        # 4. Heart rate > 100 bpm
        if heart_rate > 100:
            components['hr_gt_100'] = 2
            score += 2
        else:
            components['hr_gt_100'] = 0
        
        # 5. Killip class II-IV
        if killip_class >= 2:
            components['killip_geq_2'] = 2
            score += 2
        else:
            components['killip_geq_2'] = 0
        
        # 6. Weight < 67 kg
        if weight_kg < 67:
            components['weight_lt_67'] = 1
            score += 1
        else:
            components['weight_lt_67'] = 0
        
        # 7. Anterior MI or LBBB
        if anterior_mi_or_lbbb:
            components['anterior_or_lbbb'] = 1
            score += 1
        else:
            components['anterior_or_lbbb'] = 0
        
        # 8. Time to treatment > 4 hours
        if time_to_treatment_hours > 4:
            components['time_to_tx_gt_4h'] = 1
            score += 1
        else:
            components['time_to_tx_gt_4h'] = 0
        
        # Determine risk category
        if score <= 2:
            risk = "low"
        elif score <= 5:
            risk = "intermediate"
        else:
            risk = "high"
        
        # Get mortality rate (capped at max available)
        mortality_rate = self.MORTALITY_RATES.get(min(int(score), 14), 0.9)
        
        return {
            'score': int(score),
            'mortality_rate': mortality_rate,
            'mortality_pct': mortality_rate * 100,
            'risk_category': risk,
            'components': components,
        }
    
    def compute_safe(
        self,
        age: Any = None,
        systolic_bp: Any = None,
        heart_rate: Any = None,
        killip_class: Any = None,
        anterior_mi_or_lbbb: Any = False,
        anterior: Any = None,  # ALIAS for newer mapper compatibility
        weight_kg: Any = None,
        weight: Any = None,  # ALIAS for newer mapper compatibility
        time_to_treatment_hours: Any = None,
        time_to_treatment: Any = None,  # ALIAS for newer mapper compatibility
        ischemia_time: Any = None,  # ALIAS for newer mapper compatibility
        diabetes: Any = False,
        hypertension: Any = False,
        angina_prior: Any = False,
        **kwargs
    ) -> Dict[str, Any]:
        """Compute TIMI STEMI risk score with input validation.
        
        Validates all inputs before computation. Returns structured result
        with error/warning information.
        
        Args:
            age: Patient age (years)
            systolic_bp: Systolic blood pressure (mmHg)
            heart_rate: Heart rate (bpm)
            killip_class: Killip-Kimball class (1-4)
            anterior_mi_or_lbbb: Anterior MI or LBBB indicator (or use 'anterior' alias)
            anterior: ALIAS for anterior_mi_or_lbbb (for variable mapper compatibility)
            weight_kg: Body weight in kg (or use 'weight' alias, default 80 if not provided)
            weight: ALIAS for weight_kg (for variable mapper compatibility)
            time_to_treatment_hours: Time to treatment in hours (or use 'time_to_treatment'/'ischemia_time' aliases, default 2)
            time_to_treatment: ALIAS for time_to_treatment_hours (for variable mapper compatibility)
            ischemia_time: ALIAS for time_to_treatment_hours (for variable mapper compatibility)
            diabetes: History of diabetes
            hypertension: History of hypertension
            angina_prior: Prior angina pectoris
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
                f"TIMI-STEMI: Received unexpected variables: {', '.join(unexpected)}. "
                f"These will be ignored."
            )
        
        # Validate required numeric variables
        validation = validate_timi_stemi_inputs(age, systolic_bp, heart_rate, killip_class)
        result.errors.extend(validation.errors)
        result.warnings.extend(validation.warnings)
        
        if not validation.is_valid:
            result.success = False
            return result.to_dict()
        
        # Convert validated values
        age_val = float(age)
        sbp_val = float(systolic_bp)
        hr_val = float(heart_rate)
        killip_val = int(killip_class)
        
        # Validate optional boolean variables
        is_valid_ami, ami_val = validate_bool(
            anterior if anterior is not None else anterior_mi_or_lbbb,  # Try alias first, fall back to original
            "anterior_mi_or_lbbb",
            validation
        )
        is_valid_dia, dia_val = validate_bool(diabetes, "diabetes", validation)
        is_valid_htn, htn_val = validate_bool(hypertension, "hypertension", validation)
        is_valid_ang, ang_val = validate_bool(angina_prior, "angina_prior", validation)
        
        # Validate optional numeric variables with defaults
        # Handle weight aliases: prefer 'weight' if provided, else weight_kg, else default 80
        weight_val = 80.0
        weight_input = weight if weight is not None else weight_kg
        if weight_input is not None and not is_missing(weight_input):
            is_valid_w, weight_val = validate_numeric(weight_input, "weight", validation)
            if not is_valid_w:
                weight_val = 80.0
                validation.add_warning("weight/weight_kg: Using default 80 kg")
        
        # Handle time aliases: prefer 'time_to_treatment'/'ischemia_time' if provided, else time_to_treatment_hours, else default 2
        time_val = 2.0
        time_input = time_to_treatment if time_to_treatment is not None else (
            ischemia_time if ischemia_time is not None else time_to_treatment_hours
        )
        if time_input is not None and not is_missing(time_input):
            is_valid_t, time_val = validate_numeric(time_input, "time_to_treatment", validation)
            if not is_valid_t:
                time_val = 2.0
                validation.add_warning("time_to_treatment/ischemia_time: Using default 2 hours")
        
        result.errors.extend(validation.errors)
        result.warnings.extend(validation.warnings)
        
        if not (is_valid_ami and is_valid_dia and is_valid_htn and is_valid_ang):
            result.success = False
            return result.to_dict()
        hr_val = float(heart_rate)
        killip_val = int(killip_class)
        
        # Validate optional boolean variables
        is_valid_ami, ami_val = validate_bool(
            anterior_mi_or_lbbb, "anterior_mi_or_lbbb", validation
        )
        is_valid_dia, dia_val = validate_bool(diabetes, "diabetes", validation)
        is_valid_htn, htn_val = validate_bool(hypertension, "hypertension", validation)
        is_valid_ang, ang_val = validate_bool(angina_prior, "angina_prior", validation)
        
        # Validate optional numeric variables with defaults
        weight_val = 80.0
        if weight_kg is not None and not is_missing(weight_kg):
            is_valid_w, weight_val = validate_numeric(weight_kg, "weight_kg", validation)
            if not is_valid_w:
                weight_val = 80.0
                validation.add_warning("weight_kg: Using default 80 kg")
        
        time_val = 2.0
        if time_to_treatment_hours is not None and not is_missing(time_to_treatment_hours):
            is_valid_t, time_val = validate_numeric(
                time_to_treatment_hours, "time_to_treatment_hours", validation
            )
            if not is_valid_t:
                time_val = 2.0
                validation.add_warning("time_to_treatment_hours: Using default 2 hours")
        
        result.errors.extend(validation.errors)
        result.warnings.extend(validation.warnings)
        
        if not (is_valid_ami and is_valid_dia and is_valid_htn and is_valid_ang):
            result.success = False
            return result.to_dict()
        
        # Computation is safe - call regular compute()
        try:
            compute_result = self.compute(
                age=age_val,
                systolic_bp=sbp_val,
                heart_rate=hr_val,
                killip_class=killip_val,
                anterior_mi_or_lbbb=ami_val,
                weight_kg=weight_val,
                time_to_treatment_hours=time_val,
                diabetes=dia_val,
                hypertension=htn_val,
                angina_prior=ang_val,
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
        age: np.ndarray,
        systolic_bp: np.ndarray,
        heart_rate: np.ndarray,
        killip_class: np.ndarray,
        anterior_mi_or_lbbb: np.ndarray,
        weight_kg: Optional[np.ndarray] = None,
        time_to_treatment_hours: Optional[np.ndarray] = None,
        diabetes: Optional[np.ndarray] = None,
        hypertension: Optional[np.ndarray] = None,
        angina_prior: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Compute TIMI STEMI scores for multiple patients.
        
        Parameters
        ----------
        age : np.ndarray
            Array of ages
        systolic_bp : np.ndarray
            Array of systolic BP values
        heart_rate : np.ndarray
            Array of heart rates
        killip_class : np.ndarray
            Array of Killip classes
        anterior_mi_or_lbbb : np.ndarray
            Array of anterior MI / LBBB indicators
        weight_kg : np.ndarray, optional
            Array of weights (default 80 if None)
        time_to_treatment_hours : np.ndarray, optional
            Array of time to treatment (default 2.0 if None)
        diabetes : np.ndarray, optional
            Array of diabetes indicators
        hypertension : np.ndarray, optional
            Array of hypertension indicators
        angina_prior : np.ndarray, optional
            Array of prior angina indicators
            
        Returns
        -------
        np.ndarray
            Array of TIMI STEMI scores
        """
        n = len(age)
        
        # Fill defaults if not provided
        if weight_kg is None:
            weight_kg = np.full(n, 80.0)
        if time_to_treatment_hours is None:
            time_to_treatment_hours = np.full(n, 2.0)
        if diabetes is None:
            diabetes = np.zeros(n, dtype=bool)
        if hypertension is None:
            hypertension = np.zeros(n, dtype=bool)
        if angina_prior is None:
            angina_prior = np.zeros(n, dtype=bool)
        
        scores = []
        for i in range(n):
            result = self.compute(
                age=float(age[i]),
                systolic_bp=float(systolic_bp[i]),
                heart_rate=float(heart_rate[i]),
                killip_class=int(killip_class[i]),
                anterior_mi_or_lbbb=bool(anterior_mi_or_lbbb[i]),
                weight_kg=float(weight_kg[i]),
                time_to_treatment_hours=float(time_to_treatment_hours[i]),
                diabetes=bool(diabetes[i]),
                hypertension=bool(hypertension[i]),
                angina_prior=bool(angina_prior[i]),
            )
            scores.append(result['score'])
        
        return np.array(scores)


# =============================================================================
# TIMI NSTEMI (Non-ST-Elevation MI / Unstable Angina) - Antman et al. 2000
# =============================================================================

class TIMINSTEMIScore:
    """
    TIMI Risk Score for Non-ST-Elevation MI / Unstable Angina (NSTEMI).
    
    Predicts mortality, MI, or urgent revascularization at 14 days.
    Simple 7-variable score for bedside risk assessment.
    
    Score range: 0-7 points
    References: Antman et al. JAMA 2000, 284(7):835-842.
    """
    
    # Event rates at 14 days from Antman et al. 2000
    EVENT_RATES = {
        0: 0.050,    # 5.0% (death, MI, urgent revasc at 14 days)
        1: 0.086,    # 8.6%
        2: 0.126,    # 12.6%
        3: 0.185,    # 18.5%
        4: 0.245,    # 24.5%
        5: 0.350,    # 35.0%
        6: 0.515,    # 51.5%
        7: 0.825,    # 82.5%
    }
    
    def __init__(self):
        """Initialize TIMI NSTEMI score calculator."""
        self.name = "TIMI-NSTEMI"
        self.version = "2.0"
        self.description = "TIMI Risk Score for NSTEMI/Unstable Angina (Antman et al. 2000)"
        self.score_type = "NSTEMI"
    
    def compute(
        self,
        age: float,
        risk_factors_count: int = 0,
        diabetes: bool = False,
        hypertension: bool = False,
        cholesterol_abnormal: bool = False,
        smoker: bool = False,
        known_coronary_disease: bool = False,
        st_deviation: bool = False,
        cardiac_markers_elevated: bool = False,
        aspirin_use_7days: bool = False,
        angina_episodes_24h: int = 0,
    ) -> Dict[str, Any]:
        """
        Calculate TIMI NSTEMI score.
        
        Parameters
        ----------
        age : float
            Patient age in years
        risk_factors_count : int
            Count of risk factors (0-5): diabetes, hypertension, 
            cholesterol >240, family Hx CAD, smoking
        diabetes : bool
            History of diabetes mellitus
        hypertension : bool
            History of hypertension
        cholesterol_abnormal : bool
            Abnormal cholesterol (>240) or on lipid-lowering therapy
        known_coronary_disease : bool
            Known coronary stenosis ≥50% or prior CAD event
        st_deviation : bool
            ST segment deviation ≥0.5 mm on admission ECG
        cardiac_markers_elevated : bool
            Elevated cardiac markers (troponin, CK-MB) on admission
        aspirin_use_7days : bool
            Aspirin use within past 7 days
        angina_episodes_24h : int
            Number of angina episodes in past 24 hours (0, 1, 2+)
            
        Returns
        -------
        dict
            {
                'score': int,
                'event_rate': float,
                'event_pct': float,
                'risk_category': str,
                'components': dict
            }
        """
        components = {}
        score = 0
        
        # 1. Age ≥ 65 years
        if age >= 65:
            components['age_geq_65'] = 1
            score += 1
        else:
            components['age_geq_65'] = 0
        
        # 2. ≥ 3 risk factors for coronary artery disease
        # (diabetes, hypertension, abnormal cholesterol, family Hx, smoking)
        # Compute if boolean factors are provided, else fallback to risk_factors_count
        calculated_rf = sum([diabetes, hypertension, cholesterol_abnormal, smoker, known_coronary_disease])
        if risk_factors_count >= 3 or calculated_rf >= 3:
            components['risk_factors_geq3'] = 1
            score += 1
        else:
            components['risk_factors_geq3'] = 0
        
        # 3. Known coronary artery disease (stenosis ≥50%)
        if known_coronary_disease:
            components['known_cad'] = 1
            score += 1
        else:
            components['known_cad'] = 0
        
        # 4. Aspirin use in past 7 days
        if aspirin_use_7days:
            components['aspirin_use_7days'] = 1
            score += 1
        else:
            components['aspirin_use_7days'] = 0
        
        # 5. ≥ 2 angina episodes in past 24 hours
        if angina_episodes_24h >= 2:
            components['angina_episodes_geq2'] = 1
            score += 1
        else:
            components['angina_episodes_geq2'] = 0
        
        # 6. ST segment deviation ≥ 0.5 mm
        if st_deviation:
            components['st_deviation_geq05'] = 1
            score += 1
        else:
            components['st_deviation_geq05'] = 0
        
        # 7. Elevated cardiac markers (troponin or CK-MB)
        if cardiac_markers_elevated:
            components['markers_elevated'] = 1
            score += 1
        else:
            components['markers_elevated'] = 0
        
        # Determine risk category
        if score <= 1:
            risk = "low"
        elif score <= 3:
            risk = "intermediate"
        else:
            risk = "high"
        
        # Get event rate
        event_rate = self.EVENT_RATES.get(int(score), 0.825)
        
        return {
            'score': int(score),
            'event_rate': event_rate,
            'event_pct': event_rate * 100,
            'risk_category': risk,
            'components': components,
        }
    
    def compute_safe(
        self,
        age: Any = None,
        risk_factors_count: Any = 0,
        diabetes: Any = False,
        hypertension: Any = False,
        cholesterol_abnormal: Any = False,
        smoker: Any = False,
        known_coronary_disease: Any = False,
        st_deviation: Any = False,
        cardiac_markers_elevated: Any = False,
        aspirin_use_7days: Any = False,
        angina_episodes_24h: Any = 0,
        **kwargs
    ) -> Dict[str, Any]:
        """Compute TIMI NSTEMI risk score with input validation.
        
        Validates all inputs before computation. Returns structured result
        with error/warning information.
        
        Args:
            age: Patient age (years) - REQUIRED
            risk_factors_count: Count of risk factors (default 0)
            diabetes: History of diabetes
            hypertension: History of hypertension
            cholesterol_abnormal: Abnormal cholesterol
            known_coronary_disease: Known CAD with significant stenosis
            st_deviation: ST segment deviation ≥0.5 mm
            cardiac_markers_elevated: Elevated troponin or CK-MB
            aspirin_use_7days: Aspirin use in past 7 days
            angina_episodes_24h: Number of angina episodes in 24 hours
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
                f"TIMI-NSTEMI: Received unexpected variables: {', '.join(unexpected)}. "
                f"These will be ignored."
            )
        
        # Validate required variable (age)
        validation = validate_timi_nstemi_inputs(age)
        result.errors.extend(validation.errors)
        result.warnings.extend(validation.warnings)
        
        if not validation.is_valid:
            result.success = False
            return result.to_dict()
        
        age_val = float(age)
        
        # Validate optional integer variables with defaults
        rf_val = 0
        if risk_factors_count is not None and not is_missing(risk_factors_count):
            is_valid_rf, rf_val = validate_int(
                risk_factors_count, "risk_factors_count", validation
            )
            if not is_valid_rf:
                rf_val = 0
        
        ap_val = 0
        if angina_episodes_24h is not None and not is_missing(angina_episodes_24h):
            is_valid_ap, ap_val = validate_int(
                angina_episodes_24h, "angina_episodes_24h", validation
            )
            if not is_valid_ap:
                ap_val = 0
        
        # Validate optional boolean variables
        is_valid_d, d_val = validate_bool(diabetes, "diabetes", validation)
        is_valid_h, h_val = validate_bool(hypertension, "hypertension", validation)
        is_valid_c, c_val = validate_bool(cholesterol_abnormal, "cholesterol_abnormal", validation)
        is_valid_sm, sm_val = validate_bool(smoker, "smoker", validation)
        is_valid_k, k_val = validate_bool(known_coronary_disease, "known_coronary_disease", validation)
        is_valid_s, s_val = validate_bool(st_deviation, "st_deviation", validation)
        is_valid_m, m_val = validate_bool(cardiac_markers_elevated, "cardiac_markers_elevated", validation)
        is_valid_a, a_val = validate_bool(aspirin_use_7days, "aspirin_use_7days", validation)
        
        result.errors.extend(validation.errors)
        result.warnings.extend(validation.warnings)
        
        all_bool_valid = is_valid_d and is_valid_h and is_valid_c and is_valid_sm and is_valid_k and is_valid_s and is_valid_m and is_valid_a
        if not all_bool_valid:
            result.success = False
            return result.to_dict()
        
        # Computation is safe - call regular compute()
        try:
            compute_result = self.compute(
                age=age_val,
                risk_factors_count=rf_val,
                diabetes=d_val,
                hypertension=h_val,
                cholesterol_abnormal=c_val,
                smoker=sm_val,
                known_coronary_disease=k_val,
                st_deviation=s_val,
                cardiac_markers_elevated=m_val,
                aspirin_use_7days=a_val,
                angina_episodes_24h=ap_val,
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
        age: np.ndarray,
        risk_factors_count: Optional[np.ndarray] = None,
        diabetes: Optional[np.ndarray] = None,
        hypertension: Optional[np.ndarray] = None,
        cholesterol_abnormal: Optional[np.ndarray] = None,
        smoker: Optional[np.ndarray] = None,
        known_coronary_disease: Optional[np.ndarray] = None,
        st_deviation: Optional[np.ndarray] = None,
        cardiac_markers_elevated: Optional[np.ndarray] = None,
        aspirin_use_7days: Optional[np.ndarray] = None,
        angina_episodes_24h: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Compute TIMI NSTEMI scores for multiple patients.
        
        Parameters
        ----------
        age : np.ndarray
            Array of ages
        risk_factors_count : np.ndarray, optional
            Array of risk factor counts (default 0)
        diabetes : np.ndarray, optional
            Array of diabetes indicators
        hypertension : np.ndarray, optional
            Array of hypertension indicators
        cholesterol_abnormal : np.ndarray, optional
            Array of abnormal cholesterol indicators
        known_coronary_disease : np.ndarray, optional
            Array of known CAD indicators
        st_deviation : np.ndarray, optional
            Array of ST deviation indicators
        cardiac_markers_elevated : np.ndarray, optional
            Array of elevated marker indicators
        aspirin_use_7days : np.ndarray, optional
            Array of aspirin use indicators
        angina_episodes_24h : np.ndarray, optional
            Array of angina episode counts
            
        Returns
        -------
        np.ndarray
            Array of TIMI NSTEMI scores
        """
        n = len(age)
        
        # Fill defaults
        if risk_factors_count is None:
            risk_factors_count = np.zeros(n, dtype=int)
        if diabetes is None:
            diabetes = np.zeros(n, dtype=bool)
        if hypertension is None:
            hypertension = np.zeros(n, dtype=bool)
        if cholesterol_abnormal is None:
            cholesterol_abnormal = np.zeros(n, dtype=bool)
        if known_coronary_disease is None:
            known_coronary_disease = np.zeros(n, dtype=bool)
        if st_deviation is None:
            st_deviation = np.zeros(n, dtype=bool)
        if cardiac_markers_elevated is None:
            cardiac_markers_elevated = np.zeros(n, dtype=bool)
        if aspirin_use_7days is None:
            aspirin_use_7days = np.zeros(n, dtype=bool)
        if angina_episodes_24h is None:
            angina_episodes_24h = np.zeros(n, dtype=int)
        
        scores = []
        for i in range(n):
            result = self.compute(
                age=float(age[i]),
                risk_factors_count=int(risk_factors_count[i]),
                diabetes=bool(diabetes[i]),
                hypertension=bool(hypertension[i]),
                cholesterol_abnormal=bool(cholesterol_abnormal[i]),
                known_coronary_disease=bool(known_coronary_disease[i]),
                st_deviation=bool(st_deviation[i]),
                cardiac_markers_elevated=bool(cardiac_markers_elevated[i]),
                aspirin_use_7days=bool(aspirin_use_7days[i]),
                angina_episodes_24h=int(angina_episodes_24h[i]),
            )
            scores.append(result['score'])
        
        return np.array(scores)


# =============================================================================
# UNIFIED INTERFACE (Backward Compatibility)
# =============================================================================

class TIMIScore:
    """
    Unified TIMI Score interface that auto-detects STEMI vs NSTEMI.
    
    This class provides backward compatibility while allowing users to
    specify the MI type explicitly.
    """
    
    def __init__(self):
        """Initialize unified TIMI score calculator."""
        self.stemi_scorer = TIMISTEMIScore()
        self.nstemi_scorer = TIMINSTEMIScore()
        self.name = "TIMI"
        self.version = "2.0"
    
    def compute(
        self,
        age: float,
        mi_type: str = "STEMI",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Compute TIMI score (auto or explicit MI type).
        
        Parameters
        ----------
        age : float
            Patient age
        mi_type : str
            "STEMI" or "NSTEMI" / "STEMI" (case-insensitive)
        **kwargs
            Variables specific to the MI type
            
        Returns
        -------
        dict
            Score result
        """
        mi_type = mi_type.upper()
        
        if mi_type == "STEMI":
            return self.stemi_scorer.compute(age=age, **kwargs)
        elif mi_type in ("NSTEMI", "UNSTABLE_ANGINA"):
            return self.nstemi_scorer.compute(age=age, **kwargs)
        else:
            raise ValueError(f"Unknown MI type: {mi_type}. Use 'STEMI' or 'NSTEMI'.")


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def compute_timi_stemi(
    age: float,
    systolic_bp: float,
    heart_rate: float,
    killip_class: int,
    anterior_mi_or_lbbb: bool = False,
    weight_kg: float = 80.0,
    time_to_treatment_hours: float = 2.0,
    **kwargs
) -> Dict[str, Any]:
    """Convenience function for TIMI STEMI calculation."""
    scorer = TIMISTEMIScore()
    return scorer.compute(
        age=age,
        systolic_bp=systolic_bp,
        heart_rate=heart_rate,
        killip_class=killip_class,
        anterior_mi_or_lbbb=anterior_mi_or_lbbb,
        weight_kg=weight_kg,
        time_to_treatment_hours=time_to_treatment_hours,
        **kwargs
    )


def compute_timi_nstemi(
    age: float,
    risk_factors_count: int = 0,
    st_deviation: bool = False,
    cardiac_markers_elevated: bool = False,
    **kwargs
) -> Dict[str, Any]:
    """Convenience function for TIMI NSTEMI calculation."""
    scorer = TIMINSTEMIScore()
    return scorer.compute(
        age=age,
        risk_factors_count=risk_factors_count,
        st_deviation=st_deviation,
        cardiac_markers_elevated=cardiac_markers_elevated,
        **kwargs
    )
