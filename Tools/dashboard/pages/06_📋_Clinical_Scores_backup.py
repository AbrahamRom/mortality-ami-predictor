"""Clinical Scores: Statistical Comparison Pipeline.

This page implements the full comparison protocol from context.txt for
clinical risk scales (GRACE, TIMI, RECUIMA) applied to the original
RECUIMA dataset.  Unlike the ML training module, the "test set" here is
the complete dataset itself — each scale is deterministic, so we generate
AUROC distributions with stratified bootstrap (B = 1000 replicas) rather
than repeated cross-validation.
"""
from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime

# Add parent directories to path
root_dir = Path(__file__).parents[2]
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score
from scipy import stats as sp_stats

from app import initialize_state

# Re-use the statistical-testing machinery already in the project
from src.training.statistical_tests import (
    test_normality_full,
    compare_multiple_models,
    compare_models,
    cohens_d,
    interpret_effect_size,
    bonferroni_correction,
    holm_bonferroni_correction,
    benjamini_hochberg_fdr,
)

# --------------------------------------------------------------------------- #
#                           INITIALISE                                         #
# --------------------------------------------------------------------------- #
initialize_state()

st.title("📋 Comparación Estadística de Escalas Clínicas")
st.markdown("---")

st.info("""
**Pipeline de Comparación — Protocolo Metodológico (*context.txt*)**

Se ejecutan **dos análisis independientes** según el tipo de infarto:

- **IAMCEST (STEMI):** GRACE vs **TIMI-STEMI** (Morrow 2000) vs RECUIMA
- **IAMSEST (NSTEMI):** GRACE vs **TIMI-NSTEMI** (Antman 2000) vs RECUIMA

**Justificación clínica:**
- Cada variante TIMI fue diseñada y validada para un subtipo de IAM diferente, con
  variables, puntos de corte y umbrales terapéuticos distintos.
- El rendimiento discriminativo (AUROC) de cada escala depende de la distribución de
  riesgo y la prevalencia de mortalidad de cada subgrupo.
- GRACE y RECUIMA, al ser escalas universales, actúan como comparadores comunes.

**Protocolo por subgrupo:**
1. Bootstrap estratificado (B = 1 000) → distribuciones AUROC
2. Normalidad — criterio 2-de-3 (Shapiro-Wilk · D'Agostino · Anderson-Darling)
3. Test global — Friedman / ANOVA según normalidad
4. Comparaciones pareadas — Wilcoxon / t pareado + Bonferroni · Holm · FDR
5. Test de DeLong — comparación directa de curvas ROC correlacionadas
6. Tamaño de efecto — Cohen's d + ΔAUROC
""")

# =========================================================================== #
#                    SECTION 0 — LOAD THE ORIGINAL DATASET                     #
# =========================================================================== #
st.markdown("---")
st.subheader("📂 Cargar Dataset Original")

# Provide default path
PROJECT_ROOT = Path(__file__).parents[3]
DEFAULT_DATASET = PROJECT_ROOT / "recuima-020425-parsed.xlsx"

use_uploaded = st.checkbox("📤 Subir dataset manualmente", value=False)

df_original: pd.DataFrame | None = None

if use_uploaded:
    uploaded = st.file_uploader(
        "Sube el dataset original (.xlsx / .csv)",
        type=["xlsx", "csv"],
        help="Debe contener todas las columnas clínicas necesarias"
    )
    if uploaded:
        try:
            if uploaded.name.endswith(".csv"):
                df_original = pd.read_csv(uploaded, sep=None, engine="python")
            else:
                df_original = pd.read_excel(uploaded)
            st.success(f"✅ Dataset cargado: {uploaded.name} — {df_original.shape[0]} pacientes, {df_original.shape[1]} columnas")
        except Exception as e:
            st.error(f"❌ Error al cargar: {e}")
else:
    if DEFAULT_DATASET.exists():
        try:
            df_original = pd.read_excel(DEFAULT_DATASET)
            st.success(f"✅ Dataset por defecto cargado: **{DEFAULT_DATASET.name}** — {df_original.shape[0]} pacientes, {df_original.shape[1]} columnas")
        except Exception as e:
            st.error(f"❌ Error al leer {DEFAULT_DATASET.name}: {e}")
    else:
        st.warning(f"⚠️ No se encontró el dataset por defecto en `{DEFAULT_DATASET}`. Sube uno manualmente.")

if df_original is None:
    st.stop()

# =========================================================================== #
#            HELPER — fuzzy column finder (robust to different datasets)        #
# =========================================================================== #

COLUMN_ALIASES: dict[str, list[str]] = {
    "age": ["edad", "age", "años", "annos"],
    "heart_rate": ["frecuencia_cardiaca", "heart_rate", "hr", "fc"],
    "systolic_bp": ["presion_arterial_sistolica", "tas", "systolic_bp", "sbp", "pa_sistolica"],
    "creatinine": ["creatinina", "creatinine", "cr"],
    "killip": ["indice_killip", "killip", "killip_class", "killip_kimball", "clase_killip"],
    "cardiac_arrest": ["comp_pcr", "paro_cardiaco", "cardiac_arrest"],
    "st_deviation": ["depresion_st", "st_deviation", "supradesnivel", "infradesnivel"],
    "ck": ["ck", "creatine_kinase"],
    "ckmb": ["ckmb", "ck_mb", "creatine_kinase_mb"],
    "elevated_enzymes": ["tropnina_hs", "troponina", "troponin", "elevated_enzymes"],
    "grace_precalc": ["escala_grace", "GRACE", "grace_score", "grace"],
    "gfr": ["filtrado_glomerular", "gfr", "tasa_filtrado_glomerular", "fg", "tfg"],
    "vf_vt": ["comp_fv_tv", "fv_tv", "fibrilacion_ventricular", "vf_vt"],
    "avb": ["comp_bav_alto_grado", "bav_alto_grado", "high_grade_avb", "avb", "bloqueo_av"],
    "diabetes": ["diabetes_mellitus", "diabetes", "dm"],
    "hypertension": ["hipertension_arterial", "hipertension", "hta", "hypertension"],
    "dyslipidemia": ["hiperlipoproteinemia", "dislipidemia", "dyslipidemia", "hiperlipidemia"],
    "smoking": ["tabaquismo", "smoking", "fumador"],
    "prior_mi": ["infarto_miocardio_agudo", "prior_mi", "ima_previo", "prior_infarction"],
    "cad": ["enfermedad_arterias_coronarias", "cad", "coronary_artery_disease"],
    "asa_use": ["asa.1", "asa_use", "aspirin", "aspirina"],
    "angina_24h": ["angina24h", "angina_24h", "severe_angina"],
    "target": ["mortality_inhospital", "exitus", "estado_vital", "mortality"],
    "shock": ["comp_shock", "shock"],
    "complicaciones": ["complicaciones", "complications"],
    "scacest": ["scacest", "tipo_iam", "stemi_type"],
    "weight": ["peso", "weight", "kg", "body_weight"],
    "ischemia_time": ["tiempo_isquemia", "ischemia_time", "time_to_treatment"],
}

ECG_LEAD_COLS = ["v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9",
                 "d1", "d2", "d3", "avl", "avf", "avr"]


def _find_col(alias_key: str, df: pd.DataFrame) -> str | None:
    """Return the first matching column name from the alias list."""
    candidates = COLUMN_ALIASES.get(alias_key, [])
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    return None


def _safe_numeric(series: pd.Series) -> pd.Series:
    """Coerce to numeric, NaN on failure."""
    return pd.to_numeric(series, errors="coerce")


def _killip_to_int(series: pd.Series) -> pd.Series:
    """Convert Killip class to integer 1-4."""
    mapping = {"i": 1, "ii": 2, "iii": 3, "iv": 4, "1": 1, "2": 2, "3": 3, "4": 4}
    return series.astype(str).str.strip().str.lower().map(mapping).fillna(1).astype(int)


def _binary_col(series: pd.Series) -> pd.Series:
    """Convert any truthy encoding to 0/1."""
    s = series.copy()
    if s.dtype == object:
        truthy = {"si", "sí", "yes", "true", "1", "s", "y"}
        s = s.astype(str).str.strip().str.lower().isin(truthy).astype(int)
    else:
        s = pd.to_numeric(s, errors="coerce").fillna(0)
        s = (s > 0).astype(int)
    return s


# =========================================================================== #
#                       SECTION 1 — COMPUTE CLINICAL SCALES                    #
# =========================================================================== #

def compute_all_scales(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    """Compute GRACE, TIMI and RECUIMA scores and probabilities.

    For each scale, if the pre-calculated score column exists it is used
    directly; otherwise the score is computed from component variables.
    Missing component variables result in partial scores (clinically valid
    approach used in registries — analogous to imputation at the component
    level with a zero/neutral contribution).

    Returns:
        (df_out, log_dict)  where df_out has new columns and log_dict
        details what happened.
    """
    log: dict[str, str] = {}
    df = df.copy()
    n = len(df)

    # ----- TARGET -----
    target_col = _find_col("target", df)
    if target_col is None:
        raise ValueError("No se encontró columna target (mortality_inhospital / exitus / estado_vital)")
    
    # Ensure binary 0/1
    if df[target_col].dtype == object:
        df["_target"] = (df[target_col].astype(str).str.lower().isin(
            ["fallecido", "dead", "1", "yes", "si", "sí"])).astype(int)
    else:
        df["_target"] = _safe_numeric(df[target_col]).fillna(0).astype(int)
    
    log["target"] = f"Columna target: **{target_col}** → {df['_target'].sum()} eventos / {n} pacientes ({df['_target'].mean()*100:.1f}%)"

    # ================================================================
    #  GRACE SCORE
    # ================================================================
    grace_precalc_col = _find_col("grace_precalc", df)
    if grace_precalc_col and df[grace_precalc_col].notna().sum() > n * 0.5:
        # Use pre-calculated score
        df["grace_score"] = _safe_numeric(df[grace_precalc_col])
        log["grace"] = f"GRACE: usando columna precalculada **{grace_precalc_col}** ({df['grace_score'].notna().sum()}/{n} válidos)"
    else:
        # Compute from components
        age_col = _find_col("age", df)
        hr_col = _find_col("heart_rate", df)
        sbp_col = _find_col("systolic_bp", df)
        cr_col = _find_col("creatinine", df)
        killip_col = _find_col("killip", df)
        arrest_col = _find_col("cardiac_arrest", df)
        st_col = _find_col("st_deviation", df)
        enz_col = _find_col("elevated_enzymes", df)

        from src.scoring import get_score
        grace_scorer = get_score("grace")

        age_s = _safe_numeric(df[age_col]) if age_col else pd.Series(np.full(n, 65), dtype=float)
        hr_s = _safe_numeric(df[hr_col]) if hr_col else pd.Series(np.full(n, 80), dtype=float)
        sbp_s = _safe_numeric(df[sbp_col]) if sbp_col else pd.Series(np.full(n, 120), dtype=float)
        
        # Creatinine: detect µmol/L vs mg/dL
        if cr_col:
            cr_s = _safe_numeric(df[cr_col])
            if cr_s.median() > 20:  # µmol/L (range ~50-150)
                cr_s = cr_s / 88.4  # Convert to mg/dL
                log["grace_creat"] = "Creatinina convertida de µmol/L a mg/dL (÷88.4)"
        else:
            cr_s = pd.Series(np.full(n, 1.0), dtype=float)

        killip_s = _killip_to_int(df[killip_col]) if killip_col else pd.Series(np.ones(n, dtype=int))
        arrest_s = _binary_col(df[arrest_col]) if arrest_col else pd.Series(np.zeros(n, dtype=int))
        st_s = _binary_col(df[st_col]) if st_col else pd.Series(np.zeros(n, dtype=int))
        enz_s = _binary_col(df[enz_col]) if enz_col else pd.Series(np.zeros(n, dtype=int))

        scores = grace_scorer.compute_batch(
            age=age_s.fillna(65).values,
            heart_rate=hr_s.fillna(80).values,
            systolic_bp=sbp_s.fillna(120).values,
            creatinine=cr_s.fillna(1.0).values,
            cardiac_arrest=arrest_s.values,
            st_deviation=st_s.values,
            elevated_enzymes=enz_s.values,
            killip_class=killip_s.values,
        )
        df["grace_score"] = scores
        found = [c for c in [age_col, hr_col, sbp_col, cr_col, killip_col, arrest_col, st_col, enz_col] if c]
        missing = [k for k, c in [("edad", age_col), ("FC", hr_col), ("TAS", sbp_col),
                                   ("creatinina", cr_col), ("killip", killip_col),
                                   ("PCR", arrest_col), ("ST", st_col), ("enzimas", enz_col)] if not c]
        log["grace"] = f"GRACE: calculado a partir de {len(found)} variables. Faltantes (valor neutro): {missing if missing else 'ninguna'}"

    # GRACE probability — logistic transform from literature
    # P(death) ≈ 1 / (1 + exp(−(score − 133) / 25))  (approximate calibration)
    grace_valid = df["grace_score"].notna()
    df["grace_prob"] = np.nan
    df.loc[grace_valid, "grace_prob"] = 1.0 / (1.0 + np.exp(-(df.loc[grace_valid, "grace_score"] - 133) / 25))

    # ================================================================
    #  TIMI SCORES — TWO VARIANTS BY INFARCTION TYPE
    #
    #  The TIMI risk score has two separate validated instruments:
    #  • TIMI-STEMI (Morrow et al., Circulation 2000): 8 variables, 0-14 pts
    #    → Applied to IAMCEST patients (scacest == 1)
    #  • TIMI-NSTEMI (Antman et al., JAMA 2000): 7 variables, 0-7 pts
    #    → Applied to IAMSEST patients (scacest == 0)
    #
    #  Each patient receives the variant corresponding to their infarction
    #  type.  Probabilities are calibrated per-variant so they can be
    #  meaningfully combined for the global AUROC comparison.
    # ================================================================
    age_col = _find_col("age", df)
    sbp_col = _find_col("systolic_bp", df)
    hr_col = _find_col("heart_rate", df)
    killip_col = _find_col("killip", df)
    st_col = _find_col("st_deviation", df)
    scacest_col = _find_col("scacest", df)
    weight_col = _find_col("weight", df)
    isch_time_col = _find_col("ischemia_time", df)
    cad_col = _find_col("cad", df)
    asa_col = _find_col("asa_use", df)
    angina24_col = _find_col("angina_24h", df)
    dm_col = _find_col("diabetes", df)
    hta_col = _find_col("hypertension", df)
    dyslip_col = _find_col("dyslipidemia", df)
    smoking_col = _find_col("smoking", df)
    prior_mi_col = _find_col("prior_mi", df)

    # Determine STEMI / NSTEMI mask -----------------------------------
    if scacest_col:
        scacest_vals = _safe_numeric(df[scacest_col]).fillna(-1)
        is_stemi = scacest_vals == 1
        is_nstemi = scacest_vals == 0
    else:
        # Fallback: treat all as STEMI (majority class 88.5 %)
        is_stemi = pd.Series(True, index=df.index)
        is_nstemi = pd.Series(False, index=df.index)
        log["timi_warn"] = ("TIMI: columna 'scacest' no encontrada; "
                            "se asume IAMCEST para todos los pacientes.")

    n_stemi = is_stemi.sum()
    n_nstemi = is_nstemi.sum()

    # Shared helper series
    has_dm = _binary_col(df[dm_col]) if dm_col else pd.Series(0, index=df.index)
    has_hta = _binary_col(df[hta_col]) if hta_col else pd.Series(0, index=df.index)
    has_dyslip = _binary_col(df[dyslip_col]) if dyslip_col else pd.Series(0, index=df.index)
    has_smoking = _binary_col(df[smoking_col]) if smoking_col else pd.Series(0, index=df.index)
    has_prior_mi = _binary_col(df[prior_mi_col]) if prior_mi_col else pd.Series(0, index=df.index)

    # Initialize separate columns per variant
    df["timi_stemi_score"] = np.nan
    df["timi_stemi_prob"] = np.nan
    df["timi_nstemi_score"] = np.nan
    df["timi_nstemi_prob"] = np.nan
    df["_is_stemi"] = is_stemi
    df["_is_nstemi"] = is_nstemi

    # ------------------------------------------------------------------
    #  A) TIMI-STEMI (Morrow et al., Circulation 2000) — 0-14 pts
    # ------------------------------------------------------------------
    stemi_found = []
    stemi_missing = []

    # 1. Age: ≥75 → 3 pts, 65-74 → 2 pts
    if age_col:
        edad = _safe_numeric(df[age_col]).fillna(0)
        timi_s_age = np.where(edad >= 75, 3, np.where(edad >= 65, 2, 0))
        stemi_found.append("Edad (≥75→3, 65-74→2)")
    else:
        timi_s_age = np.zeros(n, dtype=int)
        stemi_missing.append("Edad")

    # 2. DM or HTA or angina → 1 pt
    timi_s_risk = ((has_dm == 1) | (has_hta == 1)).astype(int).values
    stemi_found.append("DM/HTA")

    # 3. SBP < 100 mmHg → 3 pts
    if sbp_col:
        timi_s_sbp = (_safe_numeric(df[sbp_col]).fillna(120) < 100).astype(int).values * 3
        stemi_found.append("TAS<100 (×3)")
    else:
        timi_s_sbp = np.zeros(n, dtype=int)
        stemi_missing.append("TAS")

    # 4. HR > 100 bpm → 2 pts
    if hr_col:
        timi_s_hr = (_safe_numeric(df[hr_col]).fillna(80) > 100).astype(int).values * 2
        stemi_found.append("FC>100 (×2)")
    else:
        timi_s_hr = np.zeros(n, dtype=int)
        stemi_missing.append("FC")

    # 5. Killip II–IV → 2 pts
    if killip_col:
        killip_int = _killip_to_int(df[killip_col])
        timi_s_killip = (killip_int >= 2).astype(int).values * 2
        stemi_found.append("Killip II-IV (×2)")
    else:
        timi_s_killip = np.zeros(n, dtype=int)
        stemi_missing.append("Killip")

    # 6. Weight < 67 kg → 1 pt
    if weight_col:
        peso_num = pd.to_numeric(
            df[weight_col].astype(str).str.replace(",", "."), errors="coerce"
        )
        timi_s_weight = (peso_num < 67).fillna(False).astype(int).values
        stemi_found.append("Peso<67 kg")
    else:
        timi_s_weight = np.zeros(n, dtype=int)
        stemi_missing.append("Peso")

    # 7. Anterior MI or LBBB → 1 pt (identified via V1-V4 lead elevation)
    v_leads = [c for c in ["v1", "v2", "v3", "v4"] if c in df.columns]
    if v_leads:
        has_anterior = (
            df[v_leads].apply(pd.to_numeric, errors="coerce")
            .fillna(0).sum(axis=1) > 0
        ).astype(int).values
        stemi_found.append(f"IAM anterior ({', '.join(v_leads)})")
    else:
        has_anterior = np.zeros(n, dtype=int)
        stemi_missing.append("IAM anterior")

    # 8. Time to treatment > 4 hours (240 min) → 1 pt
    if isch_time_col:
        tiempo_num = pd.to_numeric(df[isch_time_col], errors="coerce")
        timi_s_time = (tiempo_num > 240).fillna(False).astype(int).values
        stemi_found.append("T. isquemia>4h")
    else:
        timi_s_time = np.zeros(n, dtype=int)
        stemi_missing.append("T. isquemia")

    timi_stemi_total = (timi_s_age + timi_s_risk + timi_s_sbp + timi_s_hr
                        + timi_s_killip + timi_s_weight + has_anterior
                        + timi_s_time)

    # Assign STEMI scores
    df.loc[is_stemi, "timi_stemi_score"] = timi_stemi_total[is_stemi.values]

    # TIMI-STEMI probability — logistic calibration (midpoint ≈ 5, scale ≈ 3)
    # Approximation consistent with Morrow 2000 observed mortality curve
    stemi_scores = df.loc[is_stemi, "timi_stemi_score"]
    df.loc[is_stemi, "timi_stemi_prob"] = 1.0 / (1.0 + np.exp(-(stemi_scores - 5.0) / 3.0))

    log["timi_stemi"] = (
        f"TIMI-STEMI (Morrow 2000): {n_stemi} pacientes IAMCEST, "
        f"{len(stemi_found)}/8 componentes. "
        f"Disponibles: {stemi_found}. "
        f"Faltantes: {stemi_missing if stemi_missing else 'ninguno'}."
    )

    # ------------------------------------------------------------------
    #  B) TIMI-NSTEMI (Antman et al., JAMA 2000) — 0-7 pts
    # ------------------------------------------------------------------
    nstemi_found = []
    nstemi_missing = []

    # 1. Age ≥ 65 → 1 pt
    if age_col:
        timi_n_age = (_safe_numeric(df[age_col]).fillna(0) >= 65).astype(int).values
        nstemi_found.append("Edad≥65")
    else:
        timi_n_age = np.zeros(n, dtype=int)
        nstemi_missing.append("Edad")

    # 2. ≥ 3 CAD risk factors (DM, HTA, dyslipidemia, smoking)
    rf_count = has_dm + has_hta + has_dyslip + has_smoking
    rf_available = [label for col, label in [
        (dm_col, "DM"), (hta_col, "HTA"),
        (dyslip_col, "Dislipidemia"), (smoking_col, "Tabaquismo"),
    ] if col]
    rf_missing_list = [label for col, label in [
        (dm_col, "DM"), (hta_col, "HTA"),
        (dyslip_col, "Dislipidemia"), (smoking_col, "Tabaquismo"),
    ] if not col]
    timi_n_risk = (rf_count >= 3).astype(int).values
    nstemi_found.append(f"≥3 FR ({', '.join(rf_available)})")
    if rf_missing_list:
        nstemi_missing.extend([f"FR:{m}" for m in rf_missing_list])

    # 3. Known CAD (stenosis ≥50%) or prior MI → 1 pt
    has_known_cad = _binary_col(df[cad_col]) if cad_col else pd.Series(0, index=df.index)
    timi_n_cad = ((has_known_cad == 1) | (has_prior_mi == 1)).astype(int).values
    if cad_col or prior_mi_col:
        nstemi_found.append("EAC/IMA previo")
    else:
        nstemi_missing.append("EAC/IMA previo")

    # 4. ASA use in past 7 days → 1 pt
    #    NOTE: excluded in original analysis (asa.1 = hospital use, not prior)
    if asa_col:
        timi_n_asa = _binary_col(df[asa_col]).values
        nstemi_found.append(f"ASA ({asa_col})")
    else:
        timi_n_asa = np.zeros(n, dtype=int)
        nstemi_missing.append("ASA")

    # 5. Severe angina (≥2 episodes in 24h) → 1 pt
    if angina24_col:
        timi_n_angina = _binary_col(df[angina24_col]).values
        nstemi_found.append("Angina24h")
    else:
        timi_n_angina = np.zeros(n, dtype=int)
        nstemi_missing.append("Angina24h")

    # 6. ST deviation ≥ 0.5mm → 1 pt
    if st_col:
        timi_n_st = _binary_col(df[st_col]).values
        # Also check infradesnivel (secondary indicator)
        infra_col = None
        for cand in ["infradesnivel"]:
            if cand in df.columns:
                infra_col = cand
                break
        if infra_col:
            timi_n_st = np.maximum(timi_n_st, _binary_col(df[infra_col]).values)
        nstemi_found.append("Desviación ST")
    else:
        timi_n_st = np.zeros(n, dtype=int)
        nstemi_missing.append("Desviación ST")

    # 7. Elevated cardiac biomarkers → 1 pt
    #    For confirmed AMI patients, assumed = 1 (all have positive markers)
    timi_n_markers = np.ones(n, dtype=int)
    nstemi_found.append("Marcadores (asumido=1, IAM confirmado)")

    timi_nstemi_total = (timi_n_age + timi_n_risk + timi_n_cad + timi_n_asa
                         + timi_n_angina + timi_n_st + timi_n_markers)

    # Assign NSTEMI scores
    df.loc[is_nstemi, "timi_nstemi_score"] = timi_nstemi_total[is_nstemi.values]

    # TIMI-NSTEMI probability — logistic calibration (midpoint ≈ 3.5, scale ≈ 1.2)
    nstemi_scores = df.loc[is_nstemi, "timi_nstemi_score"]
    df.loc[is_nstemi, "timi_nstemi_prob"] = 1.0 / (1.0 + np.exp(-(nstemi_scores - 3.5) / 1.2))

    log["timi_nstemi"] = (
        f"TIMI-NSTEMI (Antman 2000): {n_nstemi} pacientes IAMSEST, "
        f"{len(nstemi_found)}/7 componentes. "
        f"Disponibles: {nstemi_found}. "
        f"Faltantes: {nstemi_missing if nstemi_missing else 'ninguno'}."
    )

    log["timi"] = (
        f"TIMI: se aplicaron 2 variantes según tipo de IAM — "
        f"TIMI-STEMI ({n_stemi} pac., {len(stemi_found)}/8 comp.) y "
        f"TIMI-NSTEMI ({n_nstemi} pac., {len(nstemi_found)}/7 comp.). "
        f"Cada paciente recibe la variante correspondiente a su tipo de infarto."
    )

    # ================================================================
    #  RECUIMA SCORE
    # ================================================================
    age_col = _find_col("age", df)
    sbp_col = _find_col("systolic_bp", df)
    gfr_col = _find_col("gfr", df)
    killip_col = _find_col("killip", df)
    vfvt_col = _find_col("vf_vt", df)
    avb_col = _find_col("avb", df)

    recuima_score = pd.Series(np.zeros(n, dtype=int))
    recuima_found = []
    recuima_missing = []

    # 1. Age > 70 (1 pt)
    if age_col:
        recuima_score += (_safe_numeric(df[age_col]).fillna(0) > 70).astype(int)
        recuima_found.append("Edad>70")
    else:
        recuima_missing.append("Edad")

    # 2. SBP < 100 (1 pt)
    if sbp_col:
        recuima_score += (_safe_numeric(df[sbp_col]).fillna(120) < 100).astype(int)
        recuima_found.append("TAS<100")
    else:
        recuima_missing.append("TAS")

    # 3. GFR < 60 (3 pts — most important)
    if gfr_col:
        recuima_score += (_safe_numeric(df[gfr_col]).fillna(90) < 60).astype(int) * 3
        recuima_found.append("FG<60 (×3)")
    else:
        recuima_missing.append("FG")

    # 4. ECG leads > 7 (1 pt)
    avail_ecg = [c for c in ECG_LEAD_COLS if c in df.columns]
    if avail_ecg:
        leads_affected = df[avail_ecg].apply(pd.to_numeric, errors="coerce").fillna(0).gt(0).sum(axis=1)
        recuima_score += (leads_affected > 7).astype(int)
        recuima_found.append(f"ECG>{7} ({len(avail_ecg)} deriv.)")
    else:
        recuima_missing.append("ECG leads")

    # 5. Killip IV (1 pt)
    if killip_col:
        killip_int = _killip_to_int(df[killip_col])
        recuima_score += (killip_int == 4).astype(int)
        recuima_found.append("Killip IV")
    else:
        recuima_missing.append("Killip")

    # 6. VF/VT (2 pts)
    if vfvt_col:
        recuima_score += _binary_col(df[vfvt_col]) * 2
        recuima_found.append("FV/TV (×2)")
    else:
        # Attempt parsing from complicaciones
        comp_col = _find_col("complicaciones", df)
        if comp_col:
            from src.scoring.recuima import parse_complicaciones
            parsed = df[comp_col].apply(parse_complicaciones)
            recuima_score += parsed.apply(lambda x: x[0]).astype(int) * 2
            recuima_found.append("FV/TV (complicaciones, ×2)")
        else:
            recuima_missing.append("FV/TV")

    # 7. High-grade AVB (1 pt)
    if avb_col:
        recuima_score += _binary_col(df[avb_col])
        recuima_found.append("BAV alto grado")
    else:
        comp_col = _find_col("complicaciones", df)
        if comp_col:
            from src.scoring.recuima import parse_complicaciones
            parsed = df[comp_col].apply(parse_complicaciones)
            recuima_score += parsed.apply(lambda x: x[1]).astype(int)
            recuima_found.append("BAV (complicaciones)")
        else:
            recuima_missing.append("BAV")

    df["recuima_score"] = recuima_score

    # RECUIMA probability — thesis-based probability map
    prob_map = {0: 0.02, 1: 0.05, 2: 0.10, 3: 0.20, 4: 0.35, 5: 0.50,
                6: 0.65, 7: 0.78, 8: 0.88, 9: 0.94, 10: 0.98}
    df["recuima_prob"] = df["recuima_score"].clip(0, 10).map(prob_map).fillna(0.02)

    log["recuima"] = f"RECUIMA: calculado con {len(recuima_found)} componentes. Faltantes (0/neutro): {recuima_missing if recuima_missing else 'ninguno'}"

    return df, log


# =========================================================================== #
#              SECTION 2 — BOOTSTRAP AUROC DISTRIBUTIONS                       #
# =========================================================================== #

def bootstrap_auroc(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    B: int = 1000,
    seed: int = 42,
    stratified: bool = True,
) -> np.ndarray:
    """Generate B bootstrap samples of AUROC.

    For clinical scales (deterministic scores), cross-validation is not
    applicable.  Instead, the recommended approach in biostatistics literature
    (Carpenter & Bithell 2000; Efron & Tibshirani 1993) is to use stratified
    bootstrap resampling.

    When ``stratified=True`` (default), each bootstrap resample maintains
    the same class prevalence as the original dataset — critical when the
    event rate is low (~8.8 %).

    Returns:
        1-D array of B AUROC values.
    """
    rng = np.random.RandomState(seed)
    n = len(y_true)
    idx_pos = np.where(y_true == 1)[0]
    idx_neg = np.where(y_true == 0)[0]
    aurocs = np.zeros(B)

    for b in range(B):
        if stratified:
            boot_pos = rng.choice(idx_pos, size=len(idx_pos), replace=True)
            boot_neg = rng.choice(idx_neg, size=len(idx_neg), replace=True)
            boot_idx = np.concatenate([boot_pos, boot_neg])
        else:
            boot_idx = rng.choice(n, size=n, replace=True)

        y_t = y_true[boot_idx]
        y_p = y_prob[boot_idx]

        # Guard: if bootstrap has single class, skip
        if len(np.unique(y_t)) < 2:
            aurocs[b] = np.nan
            continue

        aurocs[b] = roc_auc_score(y_t, y_p)

    return aurocs[~np.isnan(aurocs)]


# =========================================================================== #
#                       COMPUTE SCALES BUTTON                                  #
# =========================================================================== #
st.markdown("---")
st.subheader("🧮 Calcular Escalas Clínicas y Ejecutar Pipeline")

# Settings
col_s1, col_s2 = st.columns(2)
with col_s1:
    n_bootstrap = st.number_input(
        "Réplicas Bootstrap (B)", min_value=100, max_value=5000,
        value=1000, step=100,
        help="Número de réplicas para generar distribuciones de AUROC"
    )
with col_s2:
    alpha_level = st.number_input(
        "Nivel α", min_value=0.01, max_value=0.10, value=0.05, step=0.01,
        help="Nivel de significación para pruebas de hipótesis"
    )

# =========================================================================== #
#           HELPER — DeLong-style Test (used in display function)              #
# =========================================================================== #

def _delong_roc_test(
    y_true: np.ndarray, y_score1: np.ndarray, y_score2: np.ndarray,
    n_boot: int = 2000, seed: int = 42,
) -> tuple[float, float, float, float, float]:
    """DeLong-style test for comparing two AUROCs (bootstrap variance).

    Returns (auc1, auc2, delta, z_stat, p_value).
    """
    auc1 = roc_auc_score(y_true, y_score1)
    auc2 = roc_auc_score(y_true, y_score2)
    n = len(y_true)
    rng = np.random.RandomState(seed)
    diffs = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        try:
            diffs.append(
                roc_auc_score(y_true[idx], y_score1[idx])
                - roc_auc_score(y_true[idx], y_score2[idx])
            )
        except Exception:
            continue
    se_diff = np.std(diffs) if diffs else 1e-6
    z_stat = (auc1 - auc2) / max(se_diff, 1e-10)
    p_value = 2 * (1 - sp_stats.norm.cdf(abs(z_stat)))
    return auc1, auc2, auc1 - auc2, z_stat, p_value


def _to_native(obj):
    """Convert numpy types to native Python for JSON serialization."""
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


# =========================================================================== #
#   REUSABLE DISPLAY FUNCTION — Full statistical pipeline for one subgroup    #
# =========================================================================== #

def display_subgroup_analysis(
    subgroup_label: str,
    bootstrap_results: dict[str, np.ndarray],
    point_aurocs: dict[str, float],
    point_auprcs: dict[str, float],
    y_true_valid: np.ndarray,
    scale_probs: dict[str, np.ndarray],
    score_col_map: dict[str, str],
    df_scores: pd.DataFrame,
    valid_mask: pd.Series,
    alpha_level: float,
    kp: str,
):
    """Render the full statistical comparison pipeline for a single subgroup.

    Parameters
    ----------
    subgroup_label : display name (e.g. "IAMCEST (STEMI)")
    kp : key prefix for unique Streamlit widget keys ("stemi" / "nstemi")
    score_col_map : maps scale display name → DataFrame score column name
    """
    import json

    n_scales = len(bootstrap_results)
    colors_roc = {"GRACE": "#1f77b4", "TIMI-STEMI": "#ff7f0e",
                  "TIMI-NSTEMI": "#ff7f0e", "RECUIMA": "#2ca02c"}

    # ----- Section A: ROC curves -----------------------------------------
    st.markdown("#### 📈 Curvas ROC")

    fig_roc = go.Figure()
    for name, probs in scale_probs.items():
        fpr, tpr, _ = roc_curve(y_true_valid, probs)
        auc_val = point_aurocs[name]
        fig_roc.add_trace(go.Scatter(
            x=fpr, y=tpr, mode="lines",
            name=f"{name} (AUROC={auc_val:.4f})",
            line=dict(color=colors_roc.get(name, None), width=2.5),
        ))
    fig_roc.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        name="Referencia (0.5)",
        line=dict(color="gray", dash="dash", width=1),
    ))
    fig_roc.update_layout(
        title=f"Curvas ROC — {subgroup_label}",
        xaxis_title="1 - Especificidad", yaxis_title="Sensibilidad",
        height=500, legend=dict(x=0.55, y=0.05),
    )
    st.plotly_chart(fig_roc, use_container_width=True, key=f"{kp}_roc")

    metrics_rows = []
    for name in scale_probs:
        metrics_rows.append({
            "Escala": name,
            "AUROC": f"{point_aurocs[name]:.4f}",
            "AUPRC": f"{point_auprcs[name]:.4f}",
        })
    st.dataframe(pd.DataFrame(metrics_rows), use_container_width=True, hide_index=True)

    # ----- Section B: Bootstrap distributions ----------------------------
    st.markdown("---")
    st.markdown("#### 📊 Distribuciones AUROC (Bootstrap Estratificado)")
    st.caption(f"B = {len(list(bootstrap_results.values())[0])} réplicas")

    fig_dist = go.Figure()
    scale_order = sorted(
        bootstrap_results.keys(),
        key=lambda k: np.mean(bootstrap_results[k]),
        reverse=True,
    )
    colors_dist = px.colors.qualitative.Set2

    for idx, name in enumerate(scale_order):
        scores = bootstrap_results[name]
        fig_dist.add_trace(go.Violin(
            y=scores, name=name,
            box_visible=True, meanline_visible=True,
            fillcolor=colors_dist[idx % len(colors_dist)],
            opacity=0.6,
            line_color=colors_dist[idx % len(colors_dist)],
            points="all", jitter=0.25, pointpos=-0.2,
        ))
    fig_dist.update_layout(
        title=f"Distribución AUROC — {subgroup_label}",
        yaxis_title="AUROC", xaxis_title="Escala",
        showlegend=False, height=500,
    )
    st.plotly_chart(fig_dist, use_container_width=True, key=f"{kp}_dist")

    desc_rows = []
    for name in scale_order:
        s = bootstrap_results[name]
        desc_rows.append({
            "Escala": name, "N": len(s),
            "Media": f"{np.mean(s):.4f}", "DE": f"{np.std(s):.4f}",
            "Mín": f"{np.min(s):.4f}",
            "Q1": f"{np.percentile(s, 25):.4f}",
            "Mediana": f"{np.median(s):.4f}",
            "Q3": f"{np.percentile(s, 75):.4f}",
            "Máx": f"{np.max(s):.4f}",
            "IC95% Inf": f"{np.percentile(s, 2.5):.4f}",
            "IC95% Sup": f"{np.percentile(s, 97.5):.4f}",
        })
    desc_df = pd.DataFrame(desc_rows)
    st.dataframe(desc_df, use_container_width=True, hide_index=True)
    st.download_button(
        "📥 Estadísticas Descriptivas (CSV)", desc_df.to_csv(index=False),
        f"escalas_{kp}_descriptivas.csv", "text/csv", key=f"{kp}_dl_desc",
    )

    # ----- Section C: Normality ------------------------------------------
    st.markdown("---")
    st.markdown("#### 🧪 Tests de Normalidad (Criterio 2-de-3)")

    norm_rows = []
    n_normal = 0
    for name in scale_order:
        scores = bootstrap_results[name]
        nr = test_normality_full(np.array(scores), alpha=alpha_level)
        if nr.is_normal:
            n_normal += 1
        norm_rows.append({
            "Escala": name,
            "Shapiro p": f"{nr.shapiro_wilk_pvalue:.4f}",
            "SW": "✅" if nr.shapiro_wilk_normal else "❌",
            "D'Ag p": f"{nr.dagostino_pvalue:.4f}",
            "DA": "✅" if nr.dagostino_normal else "❌",
            "And stat": f"{nr.anderson_statistic:.4f}",
            "AD": "✅" if nr.anderson_normal else "❌",
            "OK": f"{nr.normal_tests_passed}/3",
            "Decisión": "✅ Normal" if nr.is_normal else "❌ No Normal",
        })
    st.dataframe(pd.DataFrame(norm_rows), use_container_width=True, hide_index=True)

    if n_normal == n_scales:
        st.success("✅ Todas normales → Tests paramétricos")
    else:
        st.warning(f"⚠️ {n_normal}/{n_scales} normales → Tests no paramétricos")

    # Q-Q plots
    with st.expander("📉 Gráficos Q-Q"):
        fig_qq = make_subplots(rows=1, cols=n_scales, subplot_titles=list(scale_order))
        for idx, name in enumerate(scale_order):
            scores = bootstrap_results[name]
            theoretical = np.sort(sp_stats.norm.ppf(np.linspace(0.01, 0.99, len(scores))))
            observed = np.sort(scores)
            step = max(1, len(observed) // 300)
            fig_qq.add_trace(
                go.Scatter(x=theoretical[::step], y=observed[::step], mode="markers",
                           marker=dict(size=3, color=colors_dist[idx % len(colors_dist)]),
                           showlegend=False),
                row=1, col=idx + 1,
            )
            mn, mx = min(theoretical), max(theoretical)
            mu, sig = np.mean(scores), np.std(scores)
            fig_qq.add_trace(
                go.Scatter(x=[mn, mx], y=[mu + sig * mn, mu + sig * mx], mode="lines",
                           line=dict(color="red", dash="dash", width=1), showlegend=False),
                row=1, col=idx + 1,
            )
        fig_qq.update_layout(height=400, title_text=f"Q-Q Plots — {subgroup_label}")
        st.plotly_chart(fig_qq, use_container_width=True, key=f"{kp}_qq")

    # ----- Section D: Global test ----------------------------------------
    st.markdown("---")
    st.markdown("#### 🌐 Test Global de Comparación Múltiple")

    model_scores_dict = {name: list(bootstrap_results[name]) for name in scale_order}
    min_len = min(len(v) for v in model_scores_dict.values())
    model_scores_dict = {k: v[:min_len] for k, v in model_scores_dict.items()}

    multiple_comparison = compare_multiple_models(model_scores_dict, alpha=alpha_level)

    col_g1, col_g2, col_g3, col_g4 = st.columns(4)
    with col_g1:
        tname = (multiple_comparison.global_test_name.split()[0]
                 if not pd.isna(multiple_comparison.global_test_statistic) else "N/A")
        st.metric("Test", tname)
    with col_g2:
        sval = (f"{multiple_comparison.global_test_statistic:.3f}"
                if not pd.isna(multiple_comparison.global_test_statistic) else "N/A")
        st.metric("Estadístico", sval)
    with col_g3:
        pstr = (f"{multiple_comparison.global_p_value:.6f}"
                if not pd.isna(multiple_comparison.global_p_value) else "N/A")
        st.metric("P-value", pstr)
    with col_g4:
        if pd.isna(multiple_comparison.global_test_statistic):
            st.info("ℹ️ Requiere ≥3 escalas")
        elif multiple_comparison.global_significant:
            st.success("✅ SIGNIFICATIVO")
        else:
            st.warning("❌ No significativo")

    if not pd.isna(multiple_comparison.global_test_statistic):
        if multiple_comparison.global_significant:
            st.success(f"✅ {multiple_comparison.global_test_name} rechaza H₀ "
                       f"(p = {multiple_comparison.global_p_value:.6f})")
        else:
            st.info(f"ℹ️ {multiple_comparison.global_test_name} no rechaza H₀ "
                    f"(p = {multiple_comparison.global_p_value:.6f})")

    n_pairs = len(multiple_comparison.pairwise_results)
    bonf_alpha = multiple_comparison.bonferroni_alpha
    st.markdown(f"**Correcciones:** α Bonferroni = {alpha_level}/{n_pairs} = **{bonf_alpha:.4f}**")

    # ----- Ranking -------------------------------------------------------
    st.markdown("#### 🏆 Ranking")

    ranking_rows = []
    for name in scale_order:
        s = bootstrap_results[name]
        ranking_rows.append({
            "Escala": name,
            "Media AUROC": np.mean(s),
            "AUROC Puntual": point_aurocs[name],
            "DE": np.std(s),
            "IC95% Inf": np.percentile(s, 2.5),
            "IC95% Sup": np.percentile(s, 97.5),
        })
    ranking_df = (
        pd.DataFrame(ranking_rows)
        .sort_values("Media AUROC", ascending=False)
        .reset_index(drop=True)
    )
    ranking_df.insert(0, "Pos.", [
        f"🥇 {i+1}" if i == 0
        else f"🥈 {i+1}" if i == 1
        else f"🥉 {i+1}"
        for i in range(len(ranking_df))
    ])

    best_scale = ranking_df.iloc[0]["Escala"]
    best_auroc = ranking_df.iloc[0]["Media AUROC"]
    best_auroc_point = ranking_df.iloc[0]["AUROC Puntual"]
    best_ci_lo = ranking_df.iloc[0]["IC95% Inf"]
    best_ci_hi = ranking_df.iloc[0]["IC95% Sup"]

    disp = ranking_df.copy()
    for c in ["Media AUROC", "AUROC Puntual", "DE", "IC95% Inf", "IC95% Sup"]:
        disp[c] = disp[c].apply(lambda x: f"{x:.4f}")
    st.dataframe(disp, use_container_width=True, hide_index=True)

    # ----- Section E: Pairwise comparisons -------------------------------
    st.markdown("---")
    st.markdown("#### 🔄 Comparaciones Pareadas (Post-hoc)")

    stat_results = multiple_comparison.pairwise_results
    comparison_rows = []
    for (m1, m2), res in stat_results.items():
        holm_p = multiple_comparison.holm_corrected_pvalues.get((m1, m2), res.p_value)
        fdr_p = multiple_comparison.fdr_corrected_pvalues.get((m1, m2), res.p_value)
        sig_bonf = res.p_value < bonf_alpha
        comparison_rows.append({
            "Escala 1": m1, "Escala 2": m2,
            "ΔAUROC": f"{res.delta_auroc:+.4f}",
            "Test": res.test_used.replace(" test", ""),
            "p-value": f"{res.p_value:.6f}",
            "p Holm": f"{holm_p:.6f}", "p FDR": f"{fdr_p:.6f}",
            "Sig. (sin corr.)": "✅" if res.significant else "❌",
            "Sig. (Bonferroni)": "✅" if sig_bonf else "❌",
            "Cohen's d": f"{res.effect_size:.3f}",
            "Efecto": res.effect_size_interpretation,
        })
    if comparison_rows:
        comp_df = pd.DataFrame(comparison_rows)
        st.dataframe(comp_df, use_container_width=True, hide_index=True)
        st.caption("**Cohen's d:** Negligible (<0.2) · Small (0.2–0.5) · Medium (0.5–0.8) · Large (>0.8)")
        st.download_button(
            "📥 Comparaciones (CSV)", comp_df.to_csv(index=False),
            f"escalas_{kp}_pareadas.csv", "text/csv", key=f"{kp}_dl_pairs",
        )

    # ----- Section E.2: DeLong test --------------------------------------
    st.markdown("---")
    st.markdown("#### 📐 Test de DeLong")

    delong_rows = []
    scale_names = list(scale_probs.keys())
    for i in range(len(scale_names)):
        for j in range(i + 1, len(scale_names)):
            s1n, s2n = scale_names[i], scale_names[j]
            a1, a2, delta, z, p = _delong_roc_test(
                y_true_valid, scale_probs[s1n], scale_probs[s2n],
            )
            sig = p < alpha_level
            sig_bfl = p < bonf_alpha
            delong_rows.append({
                "Escala 1": s1n, "Escala 2": s2n,
                "AUROC 1": f"{a1:.4f}", "AUROC 2": f"{a2:.4f}",
                "ΔAUROC": f"{delta:+.4f}", "Z": f"{z:.3f}",
                "p-value": f"{p:.6f}",
                f"Sig. (α={alpha_level})": "✅" if sig else "❌",
                "Sig. (Bonferroni)": "✅" if sig_bfl else "❌",
            })
    delong_df = pd.DataFrame(delong_rows)
    st.dataframe(delong_df, use_container_width=True, hide_index=True)
    st.download_button(
        "📥 DeLong (CSV)", delong_df.to_csv(index=False),
        f"escalas_{kp}_delong.csv", "text/csv", key=f"{kp}_dl_delong",
    )

    # ----- Histograms of ΔAUROC ------------------------------------------
    with st.expander("📊 Histogramas ΔAUROC (Bootstrap)"):
        for (m1, m2), _res in stat_results.items():
            s1 = np.array(model_scores_dict[m1])
            s2 = np.array(model_scores_dict[m2])
            diff = s1 - s2
            fig_h = go.Figure()
            fig_h.add_trace(go.Histogram(
                x=diff, nbinsx=50, name=f"ΔAUROC ({m1}−{m2})",
                marker_color="#636EFA", opacity=0.75,
            ))
            fig_h.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="0")
            fig_h.add_vline(x=np.mean(diff), line_dash="solid", line_color="green",
                            annotation_text=f"μ={np.mean(diff):.4f}")
            ci_lo_d, ci_hi_d = np.percentile(diff, [2.5, 97.5])
            fig_h.add_vrect(x0=ci_lo_d, x1=ci_hi_d, fillcolor="lightgreen",
                            opacity=0.15, annotation_text="IC95%", line_width=0)
            fig_h.update_layout(title=f"ΔAUROC: {m1}−{m2}", xaxis_title="ΔAUROC",
                                yaxis_title="Frecuencia", height=350)
            st.plotly_chart(fig_h, use_container_width=True, key=f"{kp}_hist_{m1}_{m2}")
            ci0 = ci_lo_d <= 0 <= ci_hi_d
            st.caption(
                f"IC 95% [{ci_lo_d:.4f}, {ci_hi_d:.4f}] "
                + ("**incluye 0** → no significativa" if ci0 else "**excluye 0** → significativa")
            )

    # ----- Section F: Conclusion -----------------------------------------
    st.markdown("---")
    st.markdown("#### 🏅 Conclusión")

    best_is_stat_superior = False
    inferior_scales = []
    for (m1, m2), res in stat_results.items():
        if m1 == best_scale or m2 == best_scale:
            other = m2 if m1 == best_scale else m1
            is_sig = res.p_value < bonf_alpha
            if is_sig:
                if m1 == best_scale and res.delta_auroc > 0:
                    inferior_scales.append((other, res))
                    best_is_stat_superior = True
                elif m2 == best_scale and res.delta_auroc < 0:
                    inferior_scales.append((other, res))
                    best_is_stat_superior = True

    if best_is_stat_superior and inferior_scales:
        inf_lines = "\n".join(
            f"- **{n}** (ΔAUROC={r.delta_auroc:+.4f}, p={r.p_value:.6f}, d={r.effect_size:.3f})"
            for n, r in inferior_scales
        )
        st.success(f"""
        **🏆 Mejor Escala: {best_scale}** — AUROC={best_auroc_point:.4f}
        (Bootstrap: {best_auroc:.4f}, IC95%: [{best_ci_lo:.4f}, {best_ci_hi:.4f}])

        ✅ Estadísticamente superior (Bonferroni α={bonf_alpha:.4f}) a:
        {inf_lines}
        """)
    else:
        any_sig = any(r["Sig. (sin corr.)"] == "✅" for r in comparison_rows) if comparison_rows else False
        if any_sig:
            st.warning(f"""
            **🏆 Mayor AUROC: {best_scale}** — {best_auroc_point:.4f}
            (Bootstrap: {best_auroc:.4f}, IC95%: [{best_ci_lo:.4f}, {best_ci_hi:.4f}])

            ⚠️ Diferencias significativas sin corrección pero no sobreviven Bonferroni.
            """)
        else:
            st.info(f"""
            **🏆 Mayor AUROC: {best_scale}** — {best_auroc_point:.4f}
            (Bootstrap: {best_auroc:.4f}, IC95%: [{best_ci_lo:.4f}, {best_ci_hi:.4f}])

            ℹ️ No hay diferencias estadísticamente significativas entre las escalas.
            """)

    # ----- Section G: Detail per scale -----------------------------------
    st.markdown("---")
    st.markdown("#### 📊 Detalle por Escala")

    valid_sub = df_scores[valid_mask]
    for name in scale_order:
        scol = score_col_map.get(name, f"{name.lower()}_score")
        if scol not in df_scores.columns:
            continue
        with st.expander(f"📋 {name}", expanded=False):
            col_a, col_b = st.columns(2)
            with col_a:
                fig_s = go.Figure()
                for outcome, lab, col in [(0, "Vivo", "#2ca02c"), (1, "Fallecido", "#d62728")]:
                    sub = valid_sub[valid_sub["_target"] == outcome][scol].dropna()
                    fig_s.add_trace(go.Histogram(x=sub, name=lab, marker_color=col,
                                                 opacity=0.6, nbinsx=30))
                fig_s.update_layout(title=f"Score {name} por Desenlace",
                                    xaxis_title="Score", yaxis_title="Frecuencia",
                                    barmode="overlay", height=350)
                st.plotly_chart(fig_s, use_container_width=True, key=f"{kp}_sdist_{name}")
            with col_b:
                fig_b = go.Figure()
                for outcome, lab, col in [(0, "Vivo", "#2ca02c"), (1, "Fallecido", "#d62728")]:
                    sub = valid_sub[valid_sub["_target"] == outcome][scol].dropna()
                    fig_b.add_trace(go.Box(y=sub, name=lab, marker_color=col))
                fig_b.update_layout(title=f"Box {name}", yaxis_title="Score", height=350)
                st.plotly_chart(fig_b, use_container_width=True, key=f"{kp}_box_{name}")
            sbo = valid_sub.groupby("_target")[scol].describe()
            sbo.index = sbo.index.map({0: "Vivo", 1: "Fallecido"})
            st.dataframe(sbo.round(3), use_container_width=True)

    # ----- Section H: Download -------------------------------------------
    st.markdown("---")
    st.markdown("#### 📥 Descargar Reporte")

    raw_boot_rows = []
    for name in bootstrap_results:
        for i, v in enumerate(bootstrap_results[name]):
            raw_boot_rows.append({"Escala": name, "Replica": i + 1, "AUROC": v})
    raw_boot_df = pd.DataFrame(raw_boot_rows)

    full_report = {
        "metadata": {
            "subgrupo": subgroup_label,
            "fecha": datetime.now().isoformat(),
            "n_pacientes": int(valid_mask.sum()),
            "prevalencia_pct": round(float(y_true_valid.mean() * 100), 2),
            "n_bootstrap": int(len(list(bootstrap_results.values())[0])),
            "alpha": float(alpha_level),
        },
        "auroc_puntuales": {k: round(float(v), 4) for k, v in point_aurocs.items()},
        "auprc_puntuales": {k: round(float(v), 4) for k, v in point_auprcs.items()},
        "bootstrap_stats": {
            name: {
                "mean": round(float(np.mean(s)), 4),
                "std": round(float(np.std(s)), 4),
                "ci95_lo": round(float(np.percentile(s, 2.5)), 4),
                "ci95_hi": round(float(np.percentile(s, 97.5)), 4),
            }
            for name, s in bootstrap_results.items()
        },
        "test_global": {
            "nombre": str(multiple_comparison.global_test_name),
            "estadistico": (float(multiple_comparison.global_test_statistic)
                            if not pd.isna(multiple_comparison.global_test_statistic) else None),
            "p_value": (float(multiple_comparison.global_p_value)
                        if not pd.isna(multiple_comparison.global_p_value) else None),
            "significativo": bool(multiple_comparison.global_significant),
        },
        "comparaciones_pareadas": [
            {
                "escala1": m1, "escala2": m2,
                "delta_auroc": round(float(res.delta_auroc), 4),
                "test": str(res.test_used),
                "p_value": round(float(res.p_value), 6),
                "cohens_d": round(float(res.effect_size), 3),
                "efecto": str(res.effect_size_interpretation),
                "sig_raw": bool(res.significant),
                "sig_bonferroni": bool(res.p_value < bonf_alpha),
            }
            for (m1, m2), res in stat_results.items()
        ],
        "delong_tests": [
            {
                "escala1": row["Escala 1"], "escala2": row["Escala 2"],
                "delta_auroc": row["ΔAUROC"], "z_stat": row["Z"],
                "p_value": row["p-value"],
                "sig_raw": row[f"Sig. (α={alpha_level})"] == "✅",
                "sig_bonferroni": row["Sig. (Bonferroni)"] == "✅",
            }
            for row in delong_rows
        ],
        "conclusion": {
            "mejor_escala": str(best_scale),
            "auroc_puntual": round(float(best_auroc_point), 4),
            "auroc_bootstrap": round(float(best_auroc), 4),
            "estadisticamente_superior": bool(best_is_stat_superior),
        },
    }

    col_d1, col_d2, col_d3 = st.columns(3)
    with col_d1:
        st.download_button(
            "📥 Bootstrap (CSV)", raw_boot_df.to_csv(index=False),
            f"escalas_{kp}_bootstrap.csv", "text/csv", key=f"{kp}_dl_boot",
        )
    with col_d2:
        st.download_button(
            "📥 Resumen (CSV)", desc_df.to_csv(index=False),
            f"escalas_{kp}_resumen.csv", "text/csv", key=f"{kp}_dl_sum",
        )
    with col_d3:
        st.download_button(
            "📥 Reporte (JSON)",
            json.dumps(full_report, indent=2, default=_to_native, ensure_ascii=False),
            f"escalas_{kp}_reporte.json", "application/json", key=f"{kp}_dl_json",
        )


# =========================================================================== #
#                     PIPELINE EXECUTION & DISPLAY                             #
# =========================================================================== #

if st.button("🚀 Ejecutar Pipeline de Comparación", type="primary", use_container_width=True):
    with st.spinner("Calculando escalas y ejecutando pipeline estadístico…"):

        # ---- Step 0: Compute scales ----
        try:
            df_scores, calc_log = compute_all_scales(df_original)
        except Exception as e:
            st.error(f"❌ Error al calcular escalas: {e}")
            st.exception(e)
            st.stop()

        with st.expander("📋 Log de cálculo de escalas", expanded=True):
            for key, msg in calc_log.items():
                st.markdown(f"- {msg}")

        # ---- Build two subgroups ----
        def _build_subgroup(df, prob_cols, subgroup_mask=None):
            """Build valid mask, y_true, scale_probs for a subgroup."""
            mask = df["_target"].notna()
            for pc in prob_cols:
                mask &= df[pc].notna()
            if subgroup_mask is not None:
                mask &= subgroup_mask
            yt = df.loc[mask, "_target"].values.astype(int)
            sp = {name: df.loc[mask, col].values for col, name in prob_cols.items()}
            return mask, yt, sp

        # STEMI subgroup: GRACE vs TIMI-STEMI vs RECUIMA
        stemi_prob_cols = {
            "grace_prob": "GRACE",
            "timi_stemi_prob": "TIMI-STEMI",
            "recuima_prob": "RECUIMA",
        }
        stemi_mask, stemi_yt, stemi_sp = _build_subgroup(
            df_scores, stemi_prob_cols, df_scores.get("_is_stemi"),
        )
        # NSTEMI subgroup: GRACE vs TIMI-NSTEMI vs RECUIMA
        nstemi_prob_cols = {
            "grace_prob": "GRACE",
            "timi_nstemi_prob": "TIMI-NSTEMI",
            "recuima_prob": "RECUIMA",
        }
        nstemi_mask, nstemi_yt, nstemi_sp = _build_subgroup(
            df_scores, nstemi_prob_cols, df_scores.get("_is_nstemi"),
        )

        st.info(
            f"📊 **IAMCEST**: {stemi_mask.sum()} pacientes con 3 escalas válidas | "
            f"**IAMSEST**: {nstemi_mask.sum()} pacientes con 3 escalas válidas"
        )

        # ---- Bootstrap for both subgroups ----
        progress = st.progress(0, text="Bootstrap IAMCEST…")
        stemi_boot = {}
        for i, (name, probs) in enumerate(stemi_sp.items()):
            progress.progress(i / (len(stemi_sp) + len(nstemi_sp)),
                              text=f"Bootstrap STEMI — {name}…")
            stemi_boot[name] = bootstrap_auroc(stemi_yt, probs, B=n_bootstrap, seed=42 + i)

        nstemi_boot = {}
        for i, (name, probs) in enumerate(nstemi_sp.items()):
            progress.progress((len(stemi_sp) + i) / (len(stemi_sp) + len(nstemi_sp)),
                              text=f"Bootstrap NSTEMI — {name}…")
            nstemi_boot[name] = bootstrap_auroc(nstemi_yt, probs, B=n_bootstrap, seed=142 + i)
        progress.progress(1.0, text="✅ Bootstrap completado")

        # ---- Point AUROCs ----
        stemi_aurocs = {}
        stemi_auprcs = {}
        for name, probs in stemi_sp.items():
            try:
                stemi_aurocs[name] = roc_auc_score(stemi_yt, probs)
                stemi_auprcs[name] = average_precision_score(stemi_yt, probs)
            except Exception:
                stemi_aurocs[name] = np.nan
                stemi_auprcs[name] = np.nan

        nstemi_aurocs = {}
        nstemi_auprcs = {}
        for name, probs in nstemi_sp.items():
            try:
                nstemi_aurocs[name] = roc_auc_score(nstemi_yt, probs)
                nstemi_auprcs[name] = average_precision_score(nstemi_yt, probs)
            except Exception:
                nstemi_aurocs[name] = np.nan
                nstemi_auprcs[name] = np.nan

        # ---- Store in session state ----
        st.session_state["scales_df"] = df_scores
        st.session_state["scales_alpha"] = alpha_level
        st.session_state["stemi"] = {
            "boot": stemi_boot, "aurocs": stemi_aurocs, "auprcs": stemi_auprcs,
            "yt": stemi_yt, "sp": stemi_sp, "mask": stemi_mask,
            "score_map": {"GRACE": "grace_score", "TIMI-STEMI": "timi_stemi_score", "RECUIMA": "recuima_score"},
        }
        st.session_state["nstemi"] = {
            "boot": nstemi_boot, "aurocs": nstemi_aurocs, "auprcs": nstemi_auprcs,
            "yt": nstemi_yt, "sp": nstemi_sp, "mask": nstemi_mask,
            "score_map": {"GRACE": "grace_score", "TIMI-NSTEMI": "timi_nstemi_score", "RECUIMA": "recuima_score"},
        }
        st.session_state["scales_computed"] = True

# =========================================================================== #
#                          DISPLAY RESULTS                                     #
# =========================================================================== #
if not st.session_state.get("scales_computed", False):
    st.info("👆 Presiona **Ejecutar Pipeline** para calcular escalas y comparar.")
    st.stop()

df_scores = st.session_state["scales_df"]
alpha_level = st.session_state["scales_alpha"]
d_stemi = st.session_state["stemi"]
d_nstemi = st.session_state["nstemi"]

st.markdown("---")
st.subheader("📊 Resultados por Subgrupo de Infarto")
st.markdown("""
Cada variante TIMI se evalúa **únicamente** en su población objetivo.
GRACE y RECUIMA, al ser escalas universales, sirven de comparadores comunes
en ambos subgrupos.
""")

tab_stemi, tab_nstemi = st.tabs([
    f"🫀 IAMCEST — STEMI ({d_stemi['mask'].sum()} pac.)",
    f"🫀 IAMSEST — NSTEMI ({d_nstemi['mask'].sum()} pac.)",
])

with tab_stemi:
    st.markdown(f"**Subgrupo IAMCEST** — {d_stemi['mask'].sum()} pacientes, "
                f"prevalencia mortalidad: {d_stemi['yt'].mean()*100:.1f}%")
    display_subgroup_analysis(
        subgroup_label="IAMCEST (STEMI)",
        bootstrap_results=d_stemi["boot"],
        point_aurocs=d_stemi["aurocs"],
        point_auprcs=d_stemi["auprcs"],
        y_true_valid=d_stemi["yt"],
        scale_probs=d_stemi["sp"],
        score_col_map=d_stemi["score_map"],
        df_scores=df_scores,
        valid_mask=d_stemi["mask"],
        alpha_level=alpha_level,
        kp="stemi",
    )

with tab_nstemi:
    n_deaths_nstemi = d_nstemi["yt"].sum()
    if n_deaths_nstemi < 10:
        st.warning(
            f"⚠️ Solo {n_deaths_nstemi} fallecidos en subgrupo IAMSEST. "
            f"Los resultados estadísticos pueden ser inestables."
        )
    st.markdown(f"**Subgrupo IAMSEST** — {d_nstemi['mask'].sum()} pacientes, "
                f"prevalencia mortalidad: {d_nstemi['yt'].mean()*100:.1f}%")
    display_subgroup_analysis(
        subgroup_label="IAMSEST (NSTEMI)",
        bootstrap_results=d_nstemi["boot"],
        point_aurocs=d_nstemi["aurocs"],
        point_auprcs=d_nstemi["auprcs"],
        y_true_valid=d_nstemi["yt"],
        scale_probs=d_nstemi["sp"],
        score_col_map=d_nstemi["score_map"],
        df_scores=df_scores,
        valid_mask=d_nstemi["mask"],
        alpha_level=alpha_level,
        kp="nstemi",
    )

# =========================================================================== #
#                    SECTION Z — ABOUT / REFERENCES                            #
# =========================================================================== #
st.markdown("---")
with st.expander("📚 Metodología y Referencias"):
    st.markdown("""
    ### Pipeline de Comparación de Escalas Clínicas

    **Análisis por subgrupo de infarto:**  
    Dado que TIMI-STEMI y TIMI-NSTEMI están diseñadas para poblaciones diferentes,
    el pipeline se ejecuta de forma **independiente** en cada subgrupo:
    - **IAMCEST:** GRACE vs TIMI-STEMI vs RECUIMA
    - **IAMSEST:** GRACE vs TIMI-NSTEMI vs RECUIMA  
    
    Esto permite interpretar el AUROC de cada escala en la población para la que
    fue validada, sin contaminar la evaluación con pacientes fuera de indicación.

    **Generación de distribuciones AUROC:**  
    A diferencia de los modelos de ML (donde se usa validación cruzada repetida),
    las escalas clínicas son determinísticas — cada paciente recibe un score fijo.
    Por tanto, se utiliza **bootstrap estratificado** (B = 1 000 réplicas) para
    generar distribuciones de AUROC, manteniendo la prevalencia del evento en cada
    réplica (Carpenter & Bithell, 2000).

    **Verificación de normalidad:**  
    Criterio 2-de-3 con Shapiro-Wilk, D'Agostino-Pearson y Anderson-Darling.

    **Test global:**  
    - Si todas las distribuciones son normales: ANOVA de medidas repetidas
    - Si alguna no es normal: Test de Friedman

    **Comparaciones pareadas post-hoc:**  
    - Paramétrico: t-test pareado
    - No paramétrico: Wilcoxon signed-rank
    - Correcciones: Bonferroni, Holm-Bonferroni, Benjamini-Hochberg FDR

    **Test de DeLong:**  
    Comparación directa de curvas ROC sobre los mismos pacientes. Considera la
    correlación entre predicciones y no requiere supuesto de normalidad (DeLong
    et al., 1988).

    **Escalas clínicas implementadas:**
    - **GRACE** (Fox et al., BMJ 2006): 8 variables, score continuo 0-372
    - **TIMI** — se aplican dos variantes según el tipo de infarto:
      - **TIMI-STEMI** (Morrow et al., Circulation 2000): 8 componentes, 0-14 pts —
        edad, DM/HTA, TAS<100, FC>100, Killip II-IV, peso<67 kg, IAM anterior/BCRI,
        tiempo de isquemia >4h.  Aplicada a pacientes con IAMCEST (scacest=1).
      - **TIMI-NSTEMI** (Antman et al., JAMA 2000): 7 componentes, 0-7 pts —
        edad ≥65, ≥3 FR cardiovasculares, EAC conocida, uso de ASA, angina severa
        24h, desviación ST, biomarcadores elevados.  Aplicada a pacientes con
        IAMSEST (scacest=0).
    - **RECUIMA** (Santos Medina, Tesis Doctoral 2020): 7 variables, score 0-10

    **Manejo de variables faltantes:**  
    Siguiendo la práctica estándar en registros clínicos internacionales
    (Granger et al., 2003; Fox et al., 2006), cuando una variable componente
    no está disponible en el dataset, se asigna el valor neutro (contribución 0
    al score). Esto es análogo a la imputación a nivel de componente y resulta
    en un score más conservador.

    ### Referencias
    - Antman, E.M. et al. (2000). TIMI Risk Score for UA/NSTEMI. *JAMA*,
      284(7), 835-842.
    - Carpenter, J. & Bithell, J. (2000). Bootstrap confidence intervals.
      *Statistics in Medicine*, 19(9), 1141-1164.
    - DeLong, E.R. et al. (1988). Comparing the areas under two or more
      correlated receiver operating characteristic curves. *Biometrics*,
      44(3), 837-845.
    - Demšar, J. (2006). Statistical Comparisons of Classifiers over Multiple
      Data Sets. *JMLR*, 7, 1-30.
    - Fox, K.A. et al. (2006). Prediction of risk of death and MI following
      diagnosis of UA/NSTEMI. *BMJ*, 333, 1091.
    - Morrow, D.A. et al. (2000). TIMI Risk Score for ST-Elevation Myocardial
      Infarction. *Circulation*, 102(17), 2031-2037.
    - Santos Medina, M. (2020). Escala predictiva RECUIMA. Tesis Doctoral,
      Univ. Ciencias Médicas Santiago de Cuba.
    """)

# =========================================================================== #
#               APPENDIX — INDIVIDUAL SCORE CALCULATOR                         #
# =========================================================================== #
st.markdown("---")
st.markdown("---")
st.header("🧮 Calculadora Individual de Scores")
st.caption("Calcula el score para un paciente individual (modo manual)")

from src.scoring import get_score, list_scores

available_scores = list_scores()

calc_tab = st.selectbox(
    "Seleccionar escala:",
    [s for s in ["grace", "timi", "recuima"] if s in available_scores],
    format_func=lambda x: {
        "grace": "GRACE", "timi": "TIMI", "recuima": "RECUIMA",
    }.get(x, x),
)

if calc_tab == "grace":
    col1, col2 = st.columns(2)
    with col1:
        age_g = st.number_input("Edad (años)", 0, 120, 65, key="gc_age")
        hr_g = st.number_input("Frecuencia Cardíaca (lpm)", 0, 250, 80, key="gc_hr")
        sbp_g = st.number_input("TAS (mmHg)", 50, 250, 120, key="gc_sbp")
    with col2:
        cr_g = st.number_input(
            "Creatinina (mg/dL)", 0.1, 10.0, 1.0, step=0.1, key="gc_cr",
        )
        killip_g = st.selectbox("Killip", ["I", "II", "III", "IV"], key="gc_killip")
        st_g = st.checkbox("Desviación ST", key="gc_st")
        enz_g = st.checkbox("Enzimas cardíacas elevadas", key="gc_enz")
        arrest_g = st.checkbox("Paro cardíaco al ingreso", key="gc_arrest")

    if st.button("Calcular GRACE", key="btn_grace_calc"):
        scorer = get_score("grace")
        res = scorer.compute(
            age=age_g, heart_rate=hr_g, systolic_bp=sbp_g,
            creatinine=cr_g,
            killip_class={"I": 1, "II": 2, "III": 3, "IV": 4}[killip_g],
            st_deviation=st_g, elevated_enzymes=enz_g, cardiac_arrest=arrest_g,
        )
        c1, c2 = st.columns(2)
        c1.metric("Score GRACE", f"{res['score']:.0f}")
        risk_map = {
            "low": "🟢 Bajo", "intermediate": "🟡 Intermedio", "high": "🔴 Alto",
        }
        c2.metric("Riesgo", risk_map.get(res["risk_category"], res["risk_category"]))

elif calc_tab == "recuima":
    col1, col2 = st.columns(2)
    with col1:
        age_r = st.number_input("Edad (años)", 0, 120, 65, key="rc_age")
        sbp_r = st.number_input("TAS (mmHg)", 30, 250, 120, key="rc_sbp")
        gfr_r = st.number_input(
            "Filtrado Glomerular (ml/min/1.73m²)", 0.0, 200.0, 90.0, key="rc_gfr",
        )
        ecg_r = st.number_input(
            "Derivaciones ECG afectadas", 0, 12, 2, key="rc_ecg",
        )
    with col2:
        killip_r = st.selectbox("Killip", ["I", "II", "III", "IV"], key="rc_killip")
        vfvt_r = st.checkbox("FV/TV", key="rc_vfvt")
        avb_r = st.checkbox("BAV alto grado", key="rc_avb")

    if st.button("Calcular RECUIMA", key="btn_recuima_calc"):
        scorer = get_score("recuima")
        res = scorer.compute(
            age=age_r, systolic_bp=sbp_r, gfr=gfr_r,
            ecg_leads_affected=ecg_r,
            killip_class={"I": 1, "II": 2, "III": 3, "IV": 4}[killip_r],
            vf_vt=vfvt_r, high_grade_avb=avb_r,
        )
        c1, c2, c3 = st.columns(3)
        c1.metric("Score RECUIMA", f"{res['score']}/10")
        risk_label = "🔴 Alto" if res["risk_category"] == "high" else "🟢 Bajo"
        c2.metric("Riesgo", risk_label)
        c3.metric("Probabilidad", f"{res['probability']*100:.1f}%")

elif calc_tab == "timi":
    timi_variant = st.radio(
        "Tipo de infarto:",
        ["IAMCEST (STEMI)", "IAMSEST (NSTEMI)"],
        key="tc_variant", horizontal=True,
    )

    if timi_variant == "IAMCEST (STEMI)":
        st.caption("TIMI-STEMI (Morrow et al., Circulation 2000) — 0 a 14 puntos")
        col1, col2 = st.columns(2)
        with col1:
            age_ts = st.number_input("Edad (años)", 0, 120, 65, key="ts_age")
            sbp_ts = st.number_input("TAS (mmHg)", 30, 250, 120, key="ts_sbp")
            hr_ts = st.number_input("FC (lpm)", 0, 250, 80, key="ts_hr")
            peso_ts = st.number_input("Peso (kg)", 20.0, 200.0, 70.0, step=0.5, key="ts_peso")
        with col2:
            killip_ts = st.selectbox("Killip", ["I", "II", "III", "IV"], key="ts_killip")
            dm_ts = st.checkbox("Diabetes o HTA", key="ts_dm_hta")
            ant_ts = st.checkbox("IAM anterior o BCRI", key="ts_anterior")
            time_ts = st.number_input(
                "Tiempo de isquemia (min)", 0, 2000, 180, key="ts_time",
                help="Tiempo desde inicio de síntomas hasta tratamiento"
            )

        if st.button("Calcular TIMI-STEMI", key="btn_timi_stemi_calc"):
            score_ts = 0
            # 1. Age
            if age_ts >= 75:
                score_ts += 3
            elif age_ts >= 65:
                score_ts += 2
            # 2. DM/HTA
            score_ts += int(dm_ts)
            # 3. SBP < 100
            if sbp_ts < 100:
                score_ts += 3
            # 4. HR > 100
            if hr_ts > 100:
                score_ts += 2
            # 5. Killip II-IV
            killip_val = {"I": 1, "II": 2, "III": 3, "IV": 4}[killip_ts]
            if killip_val >= 2:
                score_ts += 2
            # 6. Weight < 67
            if peso_ts < 67:
                score_ts += 1
            # 7. Anterior / LBBB
            score_ts += int(ant_ts)
            # 8. Time > 4h
            if time_ts > 240:
                score_ts += 1

            risk_ts = (
                "🟢 Bajo" if score_ts <= 3
                else ("🟡 Intermedio" if score_ts <= 6 else "🔴 Alto")
            )
            c1, c2 = st.columns(2)
            c1.metric("Score TIMI-STEMI", f"{score_ts}/14")
            c2.metric("Riesgo", risk_ts)

    else:  # NSTEMI
        st.caption("TIMI-NSTEMI (Antman et al., JAMA 2000) — 0 a 7 puntos")
        col1, col2 = st.columns(2)
        with col1:
            age_tn = st.number_input("Edad (años)", 0, 120, 65, key="tn_age")
            dm_tn = st.checkbox("Diabetes", key="tn_dm")
            hta_tn = st.checkbox("Hipertensión", key="tn_hta")
            dyslip_tn = st.checkbox("Dislipidemia", key="tn_dyslip")
            smoking_tn = st.checkbox("Tabaquismo", key="tn_smoking")
        with col2:
            cad_tn = st.checkbox("EAC conocida / IMA previo", key="tn_cad")
            asa_tn = st.checkbox("ASA últimos 7 días", key="tn_asa")
            ang_tn = st.checkbox("Angina severa (≥2 ep/24h)", key="tn_ang")
            st_tn = st.checkbox("Desviación ST ≥0.5mm", key="tn_st")
            markers_tn = st.checkbox("Biomarcadores elevados", value=True, key="tn_markers")

        if st.button("Calcular TIMI-NSTEMI", key="btn_timi_nstemi_calc"):
            score_tn = 0
            if age_tn >= 65:
                score_tn += 1
            rf = sum([dm_tn, hta_tn, dyslip_tn, smoking_tn])
            if rf >= 3:
                score_tn += 1
            score_tn += int(cad_tn) + int(asa_tn) + int(ang_tn) + int(st_tn) + int(markers_tn)
            risk_tn = (
                "🟢 Bajo" if score_tn <= 2
                else ("🟡 Intermedio" if score_tn <= 4 else "🔴 Alto")
            )
            c1, c2 = st.columns(2)
            c1.metric("Score TIMI-NSTEMI", f"{score_tn}/7")
            c2.metric("Riesgo", risk_tn)

