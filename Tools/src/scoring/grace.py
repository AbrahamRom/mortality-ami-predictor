"""GRACE risk score implementation."""
from __future__ import annotations

from typing import Dict, Any
import numpy as np

from .validators import (
    ValidationResult,
    ComputeResult,
    validate_grace_inputs,
    validate_bool,
    validate_int,
)


class GRACEScore:
    """GRACE (Global Registry of Acute Coronary Events) risk score.
    
    Predicts in-hospital and 6-month mortality risk in ACS patients.
    """
    
    # Score points for age ranges
    AGE_POINTS = {
        (0, 39): 0,
        (40, 49): 18,
        (50, 59): 36,
        (60, 69): 55,
        (70, 79): 73,
        (80, 200): 91,
    }
    
    # Heart rate points
    HEART_RATE_POINTS = {
        (0, 69): 0,
        (70, 89): 7,
        (90, 109): 13,
        (110, 149): 23,
        (150, 199): 36,
        (200, 500): 46,
    }
    
    # Systolic BP points
    SBP_POINTS = {
        (0, 79): 63,
        (80, 99): 58,
        (100, 119): 47,
        (120, 139): 37,
        (140, 159): 26,
        (160, 199): 11,
        (200, 500): 0,
    }
    
    # Creatinine points (mg/dL)
    CREATININE_POINTS = {
        (0.0, 0.39): 2,
        (0.4, 0.79): 5,
        (0.8, 1.19): 8,
        (1.2, 1.59): 11,
        (1.6, 1.99): 14,
        (2.0, 3.99): 23,
        (4.0, 100.0): 31,
    }
    
    def __init__(self):
        """Initialize GRACE score calculator."""
        pass
    
    def _get_points(self, value: float, points_dict: Dict) -> int:
        """Get points for a value based on range dictionary."""
        for (min_val, max_val), points in points_dict.items():
            if min_val <= value < max_val:
                return points
        return 0
    
    def compute(
        self,
        age: float,
        heart_rate: float,
        systolic_bp: float,
        creatinine: float,
        cardiac_arrest: bool = False,
        st_deviation: bool = False,
        elevated_enzymes: bool = False,
        killip_class: int = 1,
    ) -> Dict[str, float]:
        """Compute GRACE risk score.
        
        Args:
            age: Patient age (years)
            heart_rate: Heart rate (bpm)
            systolic_bp: Systolic blood pressure (mmHg)
            creatinine: Serum creatinine (mg/dL)
            cardiac_arrest: Whether cardiac arrest at admission
            st_deviation: Whether ST segment deviation
            elevated_enzymes: Whether elevated cardiac enzymes
            killip_class: Killip class (1-4)
            
        Returns:
            Dictionary with 'score' and 'risk_category'
        """
        score = 0
        
        # Add points for each component
        score += self._get_points(age, self.AGE_POINTS)
        score += self._get_points(heart_rate, self.HEART_RATE_POINTS)
        score += self._get_points(systolic_bp, self.SBP_POINTS)
        score += self._get_points(creatinine, self.CREATININE_POINTS)
        
        # Binary/categorical variables
        if cardiac_arrest:
            score += 43
        if st_deviation:
            score += 30
        if elevated_enzymes:
            score += 15
        
        # Killip class
        killip_points = {1: 0, 2: 21, 3: 43, 4: 43}
        score += killip_points.get(killip_class, 0)
        
        # Determine risk category
        if score <= 108:
            risk = "low"
        elif score <= 140:
            risk = "intermediate"
        else:
            risk = "high"
        
        return {
            "score": float(score),
            "risk_category": risk,
        }
    
    def compute_safe(
        self,
        age: Any = None,
        heart_rate: Any = None,
        systolic_bp: Any = None,
        creatinine: Any = None,
        cardiac_arrest: Any = False,
        st_deviation: Any = False,
        elevated_enzymes: Any = False,
        killip_class: Any = 1,
        **kwargs
    ) -> Dict[str, Any]:
        """Compute GRACE risk score with input validation.
        
        Validates all inputs before computation. Returns structured result
        with error/warning information.
        
        Args:
            age: Patient age (years)
            heart_rate: Heart rate (bpm)
            systolic_bp: Systolic blood pressure (mmHg)
            creatinine: Serum creatinine (mg/dL)
            cardiac_arrest: Whether cardiac arrest at admission
            st_deviation: Whether ST segment deviation
            elevated_enzymes: Whether elevated cardiac enzymes
            killip_class: Killip class (1-4)
            **kwargs: Unexpected variables (will trigger warning)
            
        Returns:
            Dict with keys:
            - 'success': bool (True if computation successful)
            - 'score': float or None
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
                f"GRACE: Received unexpected variables: {', '.join(unexpected)}. "
                f"These will be ignored. Expected only: age, heart_rate, systolic_bp, creatinine, "
                f"cardiac_arrest, st_deviation, elevated_enzymes, killip_class"
            )
        
        # Validate required numeric variables
        validation = validate_grace_inputs(age, heart_rate, systolic_bp, creatinine)
        result.errors.extend(validation.errors)
        result.warnings.extend(validation.warnings)
        
        if not validation.is_valid:
            result.success = False
            return result.to_dict()
        
        # Validate optional boolean variables
        is_valid_ca, cardiac_arrest_val = validate_bool(
            cardiac_arrest, "cardiac_arrest", validation
        )
        is_valid_st, st_deviation_val = validate_bool(
            st_deviation, "st_deviation", validation
        )
        is_valid_ee, elevated_enzymes_val = validate_bool(
            elevated_enzymes, "elevated_enzymes", validation
        )
        
        # Validate killip_class
        is_valid_kc, killip_val = validate_int(killip_class, "killip_class", validation)
        if is_valid_kc and not (1 <= killip_val <= 4):
            validation.add_error(f"killip_class: Must be 1-4, got {killip_val}")
            is_valid_kc = False
        
        result.errors.extend(validation.errors)
        result.warnings.extend(validation.warnings)
        
        if not is_valid_ca or not is_valid_st or not is_valid_ee or not is_valid_kc:
            result.success = False
            return result.to_dict()
        
        # Computation is safe - call regular compute()
        try:
            compute_result = self.compute(
                age=float(age),
                heart_rate=float(heart_rate),
                systolic_bp=float(systolic_bp),
                creatinine=float(creatinine),
                cardiac_arrest=cardiac_arrest_val,
                st_deviation=st_deviation_val,
                elevated_enzymes=elevated_enzymes_val,
                killip_class=killip_val,
            )
            
            result.success = True
            result.score = compute_result["score"]
            result.risk_category = compute_result["risk_category"]
            result.data = compute_result
            
        except Exception as e:
            result.success = False
            result.errors.append(f"Computation error: {str(e)}")
        
        return result.to_dict()
    
    def compute_batch(
        self,
        age: np.ndarray,
        heart_rate: np.ndarray,
        systolic_bp: np.ndarray,
        creatinine: np.ndarray,
        cardiac_arrest: np.ndarray,
        st_deviation: np.ndarray,
        elevated_enzymes: np.ndarray,
        killip_class: np.ndarray,
    ) -> np.ndarray:
        """Compute GRACE scores for multiple patients.
        
        Args:
            age: Array of ages
            heart_rate: Array of heart rates
            systolic_bp: Array of systolic BPs
            creatinine: Array of creatinine values
            cardiac_arrest: Array of cardiac arrest indicators
            st_deviation: Array of ST deviation indicators
            elevated_enzymes: Array of elevated enzyme indicators
            killip_class: Array of Killip classes
            
        Returns:
            Array of GRACE scores
        """
        scores = []
        
        for i in range(len(age)):
            result = self.compute(
                age=age[i],
                heart_rate=heart_rate[i],
                systolic_bp=systolic_bp[i],
                creatinine=creatinine[i],
                cardiac_arrest=bool(cardiac_arrest[i]),
                st_deviation=bool(st_deviation[i]),
                elevated_enzymes=bool(elevated_enzymes[i]),
                killip_class=int(killip_class[i]),
            )
            scores.append(result["score"])
        
        return np.array(scores)
