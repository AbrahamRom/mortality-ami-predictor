"""Clinical Scores: Statistical Comparison Pipeline.

Implements the FULL comparison protocol from the Methodological Design document
(Diseño Metodológico para la Comparación Estadística de Modelos Predictivos)
for clinical risk scales (GRACE, TIMI, RECUIMA).

Pipeline features per the protocol:
─────────────────────────────────────
• Distribution generation:
  - Bootstrap estratificado (B = 1 000)  [Carpenter & Bithell 2000]
  - Validación cruzada estratificada repetida (k=5 × r=20 = 100)  [Protocol §3.2]
• Rigorous pipeline for **AUROC** (primary metric)  [Protocol §2.1]
• Rigorous pipeline for **AUPRC** (complementary metric)  [Protocol §2.2]
• General metrics analysis (Sensitivity, Specificity, VPP, VPN, F1)  [Protocol §2 descriptive]
• Normality: Shapiro-Wilk, D'Agostino-Pearson, Anderson-Darling (2-of-3)  [Protocol §3.3]
• Global tests: Friedman / ANOVA repeated measures  [Protocol §3.4]
• Pairwise: Wilcoxon signed-rank / paired t-test  [Protocol §3.4]
• Robustness: Mann-Whitney U (independent-sample check)  [Protocol §3.4 Escenario B]
• Post-hoc: Nemenyi / Wilcoxon + Holm correction  [Protocol §3.4]
• DeLong test for correlated ROC curves (AUROC only)  [Protocol §3.4.3]
• Corrections: Bonferroni, Holm-Bonferroni, Benjamini-Hochberg FDR  [Protocol §4.1]
• Effect size: Cohen's d + Δmetric  [Protocol §4.2]
• All results downloadable to Resultados/ folders  [User requirement]
"""
from __future__ import annotations

import json
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
from sklearn.metrics import (
    roc_auc_score, roc_curve, average_precision_score,
    precision_recall_curve, confusion_matrix, f1_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold
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
    mann_whitney_test,
    nemenyi_post_hoc,
)

# Orchestrator for ML vs Clinical Scales comparison
try:
    from src.evaluation.scale_ml_orchestrator import ClinicalScaleMLOrchestrator
    from src.prediction.predictor import load_predictor
    ORCHESTRATOR_AVAILABLE = True
except ImportError as e:
    ORCHESTRATOR_AVAILABLE = False
    st.warning(f"⚠️ Orchestrator no disponible: {e}")


# =========================================================================== #
#                           CONSTANTS                                          #
# =========================================================================== #
REPO_ROOT = Path(__file__).parents[3]
RESULTADOS_BASE = REPO_ROOT / "Resultados"
RESULTS_AUROC = RESULTADOS_BASE / "Resultados_auroc"
RESULTS_AUPRC = RESULTADOS_BASE / "Resultados_auprc"
RESULTS_GENERAL = RESULTADOS_BASE / "Resultados_metricas_generales"

# --------------------------------------------------------------------------- #
#                           INITIALISE                                         #
# --------------------------------------------------------------------------- #
initialize_state()

st.title("📋 Comparación Estadística de Escalas Clínicas")
st.markdown("---")

st.info("""
**Pipeline de Comparación — Protocolo Metodológico (Diseño Metodológico)**

Se ejecutan **dos análisis independientes** según el tipo de infarto:

- **IAMCEST (STEMI):** GRACE vs **TIMI-STEMI** (Morrow 2000) vs RECUIMA
- **IAMSEST (NSTEMI):** GRACE vs **TIMI-NSTEMI** (Antman 2000) vs RECUIMA

**Protocolo por subgrupo (para AUROC *y* AUPRC):**
1. Generación de distribuciones → Bootstrap estratificado (B=1 000)
   **ó** Validación cruzada estratificada repetida (k=5 × r=20 = 100)
2. Normalidad — criterio 2-de-3 (Shapiro-Wilk · D'Agostino · Anderson-Darling)
3. Análisis visual — Histogramas + curva normal, Q-Q plots, asimetría / curtosis
4. Test global — Friedman / ANOVA según normalidad
5. Comparaciones pareadas — Wilcoxon / t pareado + correcciones múltiples
6. Robustez — Mann-Whitney U (muestras independientes)
7. Post-hoc — Nemenyi (tras Friedman) / Wilcoxon + Holm
8. Test de DeLong — comparación directa ROC correlacionadas (solo AUROC)
9. Tamaño de efecto — Cohen's d + Δmétrica
10. **Métricas generales** — Sensibilidad, Especificidad, VPP, VPN, F1 (descriptivo)

**Resultados guardados en:** `Resultados/Resultados_auroc/`, `Resultados_auprc/`,
`Resultados_metricas_generales/`
""")

# =========================================================================== #
#                    SECTION 0 — LOAD THE ORIGINAL DATASET                     #
# =========================================================================== #
st.markdown("---")
st.subheader("📂 Cargar Dataset Original")

PROJECT_ROOT = Path(__file__).parents[3]
DEFAULT_DATASET = PROJECT_ROOT / "recuima-020425-parsed.xlsx"

use_uploaded = st.checkbox("📤 Subir dataset manualmente", value=False)

df_original: pd.DataFrame | None = None

if use_uploaded:
    uploaded = st.file_uploader(
        "Sube el dataset original (.xlsx / .csv)",
        type=["xlsx", "csv"],
        help="Debe contener todas las columnas clínicas necesarias",
    )
    if uploaded:
        try:
            if uploaded.name.endswith(".csv"):
                df_original = pd.read_csv(uploaded, sep=None, engine="python")
            else:
                df_original = pd.read_excel(uploaded)
            st.success(
                f"✅ Dataset cargado: {uploaded.name} — "
                f"{df_original.shape[0]} pacientes, {df_original.shape[1]} columnas"
            )
        except Exception as e:
            st.error(f"❌ Error al cargar: {e}")
else:
    if DEFAULT_DATASET.exists():
        try:
            df_original = pd.read_excel(DEFAULT_DATASET)
            st.success(
                f"✅ Dataset por defecto cargado: **{DEFAULT_DATASET.name}** — "
                f"{df_original.shape[0]} pacientes, {df_original.shape[1]} columnas"
            )
        except Exception as e:
            st.error(f"❌ Error al leer {DEFAULT_DATASET.name}: {e}")
    else:
        st.warning(
            f"⚠️ No se encontró el dataset por defecto en `{DEFAULT_DATASET}`. "
            "Sube uno manualmente."
        )

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

ECG_LEAD_COLS = [
    "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9",
    "d1", "d2", "d3", "avl", "avf", "avr",
]


def _find_col(alias_key: str, df: pd.DataFrame) -> str | None:
    candidates = COLUMN_ALIASES.get(alias_key, [])
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    return None


def _safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _killip_to_int(series: pd.Series) -> pd.Series:
    mapping = {"i": 1, "ii": 2, "iii": 3, "iv": 4, "1": 1, "2": 2, "3": 3, "4": 4}
    return series.astype(str).str.strip().str.lower().map(mapping).fillna(1).astype(int)


def _binary_col(series: pd.Series) -> pd.Series:
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
    """
    log: dict[str, str] = {}
    df = df.copy()
    n = len(df)

    # ----- TARGET -----
    target_col = _find_col("target", df)
    if target_col is None:
        raise ValueError("No se encontró columna target (mortality_inhospital / exitus / estado_vital)")

    if df[target_col].dtype == object:
        df["_target"] = (
            df[target_col].astype(str).str.lower().isin(
                ["fallecido", "dead", "1", "yes", "si", "sí"]
            )
        ).astype(int)
    else:
        df["_target"] = _safe_numeric(df[target_col]).fillna(0).astype(int)

    log["target"] = (
        f"Columna target: **{target_col}** → {df['_target'].sum()} eventos / "
        f"{n} pacientes ({df['_target'].mean()*100:.1f}%)"
    )

    # ================================================================
    #  GRACE SCORE
    # ================================================================
    grace_precalc_col = _find_col("grace_precalc", df)
    if grace_precalc_col and df[grace_precalc_col].notna().sum() > n * 0.5:
        df["grace_score"] = _safe_numeric(df[grace_precalc_col])
        log["grace"] = (
            f"GRACE: usando columna precalculada **{grace_precalc_col}** "
            f"({df['grace_score'].notna().sum()}/{n} válidos)"
        )
    else:
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

        if cr_col:
            cr_s = _safe_numeric(df[cr_col])
            if cr_s.median() > 20:
                cr_s = cr_s / 88.4
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
        missing = [
            k for k, c in [
                ("edad", age_col), ("FC", hr_col), ("TAS", sbp_col),
                ("creatinina", cr_col), ("killip", killip_col),
                ("PCR", arrest_col), ("ST", st_col), ("enzimas", enz_col),
            ] if not c
        ]
        log["grace"] = (
            f"GRACE: calculado a partir de {len(found)} variables. "
            f"Faltantes (valor neutro): {missing if missing else 'ninguna'}"
        )

    grace_valid = df["grace_score"].notna()
    df["grace_prob"] = np.nan
    df.loc[grace_valid, "grace_prob"] = 1.0 / (
        1.0 + np.exp(-(df.loc[grace_valid, "grace_score"] - 133) / 25)
    )

    # ================================================================
    #  TIMI SCORES — TWO VARIANTS BY INFARCTION TYPE
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

    if scacest_col:
        scacest_vals = _safe_numeric(df[scacest_col]).fillna(-1)
        is_stemi = scacest_vals == 1
        is_nstemi = scacest_vals == 0
    else:
        is_stemi = pd.Series(True, index=df.index)
        is_nstemi = pd.Series(False, index=df.index)
        log["timi_warn"] = (
            "TIMI: columna 'scacest' no encontrada; se asume IAMCEST para todos."
        )

    n_stemi = is_stemi.sum()
    n_nstemi = is_nstemi.sum()

    has_dm = _binary_col(df[dm_col]) if dm_col else pd.Series(0, index=df.index)
    has_hta = _binary_col(df[hta_col]) if hta_col else pd.Series(0, index=df.index)
    has_dyslip = _binary_col(df[dyslip_col]) if dyslip_col else pd.Series(0, index=df.index)
    has_smoking = _binary_col(df[smoking_col]) if smoking_col else pd.Series(0, index=df.index)
    has_prior_mi = _binary_col(df[prior_mi_col]) if prior_mi_col else pd.Series(0, index=df.index)

    df["timi_stemi_score"] = np.nan
    df["timi_stemi_prob"] = np.nan
    df["timi_nstemi_score"] = np.nan
    df["timi_nstemi_prob"] = np.nan
    df["_is_stemi"] = is_stemi
    df["_is_nstemi"] = is_nstemi

    # -- TIMI-STEMI (Morrow 2000) -- 0-14 pts --
    stemi_found, stemi_missing = [], []

    if age_col:
        edad = _safe_numeric(df[age_col]).fillna(0)
        timi_s_age = np.where(edad >= 75, 3, np.where(edad >= 65, 2, 0))
        stemi_found.append("Edad")
    else:
        timi_s_age = np.zeros(n, dtype=int); stemi_missing.append("Edad")

    timi_s_risk = ((has_dm == 1) | (has_hta == 1)).astype(int).values
    stemi_found.append("DM/HTA")

    if sbp_col:
        timi_s_sbp = (_safe_numeric(df[sbp_col]).fillna(120) < 100).astype(int).values * 3
        stemi_found.append("TAS<100")
    else:
        timi_s_sbp = np.zeros(n, dtype=int); stemi_missing.append("TAS")

    if hr_col:
        timi_s_hr = (_safe_numeric(df[hr_col]).fillna(80) > 100).astype(int).values * 2
        stemi_found.append("FC>100")
    else:
        timi_s_hr = np.zeros(n, dtype=int); stemi_missing.append("FC")

    if killip_col:
        killip_int = _killip_to_int(df[killip_col])
        timi_s_killip = (killip_int >= 2).astype(int).values * 2
        stemi_found.append("Killip II-IV")
    else:
        timi_s_killip = np.zeros(n, dtype=int); stemi_missing.append("Killip")

    if weight_col:
        peso_num = pd.to_numeric(df[weight_col].astype(str).str.replace(",", "."), errors="coerce")
        timi_s_weight = (peso_num < 67).fillna(False).astype(int).values
        stemi_found.append("Peso<67")
    else:
        timi_s_weight = np.zeros(n, dtype=int); stemi_missing.append("Peso")

    v_leads = [c for c in ["v1", "v2", "v3", "v4"] if c in df.columns]
    if v_leads:
        has_anterior = (
            df[v_leads].apply(pd.to_numeric, errors="coerce").fillna(0).sum(axis=1) > 0
        ).astype(int).values
        stemi_found.append("IAM anterior")
    else:
        has_anterior = np.zeros(n, dtype=int); stemi_missing.append("IAM anterior")

    if isch_time_col:
        tiempo_num = pd.to_numeric(df[isch_time_col], errors="coerce")
        timi_s_time = (tiempo_num > 240).fillna(False).astype(int).values
        stemi_found.append("T. isquemia>4h")
    else:
        timi_s_time = np.zeros(n, dtype=int); stemi_missing.append("T. isquemia")

    timi_stemi_total = (
        timi_s_age + timi_s_risk + timi_s_sbp + timi_s_hr
        + timi_s_killip + timi_s_weight + has_anterior + timi_s_time
    )
    df.loc[is_stemi, "timi_stemi_score"] = timi_stemi_total[is_stemi.values]
    stemi_scores = df.loc[is_stemi, "timi_stemi_score"]
    df.loc[is_stemi, "timi_stemi_prob"] = 1.0 / (1.0 + np.exp(-(stemi_scores - 5.0) / 3.0))
    log["timi_stemi"] = (
        f"TIMI-STEMI: {n_stemi} pac., {len(stemi_found)}/8 comp. "
        f"Faltantes: {stemi_missing if stemi_missing else 'ninguno'}."
    )

    # -- TIMI-NSTEMI (Antman 2000) -- 0-7 pts --
    nstemi_found, nstemi_missing = [], []

    if age_col:
        timi_n_age = (_safe_numeric(df[age_col]).fillna(0) >= 65).astype(int).values
        nstemi_found.append("Edad≥65")
    else:
        timi_n_age = np.zeros(n, dtype=int); nstemi_missing.append("Edad")

    rf_count = has_dm + has_hta + has_dyslip + has_smoking
    timi_n_risk = (rf_count >= 3).astype(int).values
    nstemi_found.append("≥3 FR")

    has_known_cad = _binary_col(df[cad_col]) if cad_col else pd.Series(0, index=df.index)
    timi_n_cad = ((has_known_cad == 1) | (has_prior_mi == 1)).astype(int).values
    if cad_col or prior_mi_col:
        nstemi_found.append("EAC/IMA previo")
    else:
        nstemi_missing.append("EAC/IMA previo")

    if asa_col:
        timi_n_asa = _binary_col(df[asa_col]).values
        nstemi_found.append("ASA")
    else:
        timi_n_asa = np.zeros(n, dtype=int); nstemi_missing.append("ASA")

    if angina24_col:
        timi_n_angina = _binary_col(df[angina24_col]).values
        nstemi_found.append("Angina24h")
    else:
        timi_n_angina = np.zeros(n, dtype=int); nstemi_missing.append("Angina24h")

    if st_col:
        timi_n_st = _binary_col(df[st_col]).values
        nstemi_found.append("Desv. ST")
    else:
        timi_n_st = np.zeros(n, dtype=int); nstemi_missing.append("Desv. ST")

    timi_n_markers = np.ones(n, dtype=int)
    nstemi_found.append("Marcadores (=1)")

    timi_nstemi_total = (
        timi_n_age + timi_n_risk + timi_n_cad + timi_n_asa
        + timi_n_angina + timi_n_st + timi_n_markers
    )
    df.loc[is_nstemi, "timi_nstemi_score"] = timi_nstemi_total[is_nstemi.values]
    nstemi_scores_s = df.loc[is_nstemi, "timi_nstemi_score"]
    df.loc[is_nstemi, "timi_nstemi_prob"] = 1.0 / (1.0 + np.exp(-(nstemi_scores_s - 3.5) / 1.2))
    log["timi_nstemi"] = (
        f"TIMI-NSTEMI: {n_nstemi} pac., {len(nstemi_found)}/7 comp. "
        f"Faltantes: {nstemi_missing if nstemi_missing else 'ninguno'}."
    )
    log["timi"] = f"TIMI: 2 variantes — STEMI ({n_stemi} pac.) y NSTEMI ({n_nstemi} pac.)."

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
    recuima_found, recuima_missing = [], []

    if age_col:
        recuima_score += (_safe_numeric(df[age_col]).fillna(0) > 70).astype(int)
        recuima_found.append("Edad>70")
    else:
        recuima_missing.append("Edad")

    if sbp_col:
        recuima_score += (_safe_numeric(df[sbp_col]).fillna(120) < 100).astype(int)
        recuima_found.append("TAS<100")
    else:
        recuima_missing.append("TAS")

    if gfr_col:
        recuima_score += (_safe_numeric(df[gfr_col]).fillna(90) < 60).astype(int) * 3
        recuima_found.append("FG<60 (×3)")
    else:
        recuima_missing.append("FG")

    avail_ecg = [c for c in ECG_LEAD_COLS if c in df.columns]
    if avail_ecg:
        leads_affected = df[avail_ecg].apply(pd.to_numeric, errors="coerce").fillna(0).gt(0).sum(axis=1)
        recuima_score += (leads_affected > 7).astype(int)
        recuima_found.append(f"ECG>{7}")
    else:
        recuima_missing.append("ECG leads")

    if killip_col:
        killip_int = _killip_to_int(df[killip_col])
        recuima_score += (killip_int == 4).astype(int)
        recuima_found.append("Killip IV")
    else:
        recuima_missing.append("Killip")

    if vfvt_col:
        recuima_score += _binary_col(df[vfvt_col]) * 2
        recuima_found.append("FV/TV (×2)")
    else:
        comp_col = _find_col("complicaciones", df)
        if comp_col:
            from src.scoring.recuima import parse_complicaciones
            parsed = df[comp_col].apply(parse_complicaciones)
            recuima_score += parsed.apply(lambda x: x[0]).astype(int) * 2
            recuima_found.append("FV/TV (comp, ×2)")
        else:
            recuima_missing.append("FV/TV")

    if avb_col:
        recuima_score += _binary_col(df[avb_col])
        recuima_found.append("BAV")
    else:
        comp_col = _find_col("complicaciones", df)
        if comp_col:
            from src.scoring.recuima import parse_complicaciones
            parsed = df[comp_col].apply(parse_complicaciones)
            recuima_score += parsed.apply(lambda x: x[1]).astype(int)
            recuima_found.append("BAV (comp)")
        else:
            recuima_missing.append("BAV")

    df["recuima_score"] = recuima_score
    prob_map = {0: 0.02, 1: 0.05, 2: 0.10, 3: 0.20, 4: 0.35, 5: 0.50,
                6: 0.65, 7: 0.78, 8: 0.88, 9: 0.94, 10: 0.98}
    df["recuima_prob"] = df["recuima_score"].clip(0, 10).map(prob_map).fillna(0.02)

    log["recuima"] = (
        f"RECUIMA: {len(recuima_found)} componentes. "
        f"Faltantes: {recuima_missing if recuima_missing else 'ninguno'}"
    )

    return df, log


# =========================================================================== #
#         SECTION 2 — DISTRIBUTION GENERATION (BOOTSTRAP + K-FOLD)             #
# =========================================================================== #

def bootstrap_metric(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric_fn,
    B: int = 1000,
    seed: int = 42,
    stratified: bool = True,
) -> np.ndarray:
    """Stratified bootstrap for any metric (AUROC, AUPRC, etc.).

    Following Carpenter & Bithell (2000) and Efron & Tibshirani (1993).
    Stratified bootstrap maintains class prevalence in each resample.
    """
    rng = np.random.RandomState(seed)
    idx_pos = np.where(y_true == 1)[0]
    idx_neg = np.where(y_true == 0)[0]
    values = np.full(B, np.nan)

    for b in range(B):
        if stratified:
            boot_pos = rng.choice(idx_pos, size=len(idx_pos), replace=True)
            boot_neg = rng.choice(idx_neg, size=len(idx_neg), replace=True)
            boot_idx = np.concatenate([boot_pos, boot_neg])
        else:
            boot_idx = rng.choice(len(y_true), size=len(y_true), replace=True)

        yt = y_true[boot_idx]
        yp = y_prob[boot_idx]
        if len(np.unique(yt)) < 2:
            continue
        try:
            values[b] = metric_fn(yt, yp)
        except Exception:
            continue

    return values[~np.isnan(values)]


def stratified_kfold_repeated_metric(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric_fn,
    k: int = 5,
    r: int = 20,
    seed: int = 42,
) -> np.ndarray:
    """Stratified K-Fold Repeated CV for any metric.

    Protocol §3.2: k=5 folds × r=20 repetitions = 100 estimates.
    For deterministic clinical scores, no training is needed — each fold's
    test partition is used to compute the metric.

    Maintains the original class proportion in each fold via stratification.
    Uses reproducible seeds (1..r) for each repetition.
    """
    values = []
    rskf = RepeatedStratifiedKFold(n_splits=k, n_repeats=r, random_state=seed)
    X_dummy = np.zeros((len(y_true), 1))

    for _, test_idx in rskf.split(X_dummy, y_true):
        yt = y_true[test_idx]
        yp = y_prob[test_idx]
        if len(np.unique(yt)) < 2:
            continue
        try:
            values.append(metric_fn(yt, yp))
        except Exception:
            continue

    return np.array(values)


def generate_distribution(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric: str = "auroc",
    method: str = "bootstrap",
    B: int = 1000,
    k: int = 5,
    r: int = 20,
    seed: int = 42,
) -> np.ndarray:
    """Unified entry point for distribution generation."""
    metric_fn = roc_auc_score if metric == "auroc" else average_precision_score

    if method == "bootstrap":
        return bootstrap_metric(y_true, y_prob, metric_fn, B=B, seed=seed)
    else:  # kfold
        return stratified_kfold_repeated_metric(y_true, y_prob, metric_fn, k=k, r=r, seed=seed)


# =========================================================================== #
#            SECTION 3 — GENERAL METRICS (Sensitivity, Spec, etc.)             #
# =========================================================================== #

def _optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Find optimal threshold using Youden's J statistic."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    j_scores = tpr - fpr
    return float(thresholds[np.argmax(j_scores)])


def compute_threshold_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float | None = None,
) -> dict[str, float]:
    """Compute secondary metrics at a given or optimal threshold."""
    if threshold is None:
        threshold = _optimal_threshold(y_true, y_prob)

    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
    acc = (tp + tn) / (tp + tn + fp + fn)

    return {
        "Umbral": threshold, "Sensibilidad": sens, "Especificidad": spec,
        "VPP": ppv, "VPN": npv, "F1-score": f1, "Exactitud": acc,
    }


def bootstrap_threshold_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    B: int = 1000,
    seed: int = 42,
    threshold: float | None = None,
) -> pd.DataFrame:
    """Bootstrap CIs for secondary metrics."""
    if threshold is None:
        threshold = _optimal_threshold(y_true, y_prob)

    rng = np.random.RandomState(seed)
    idx_pos = np.where(y_true == 1)[0]
    idx_neg = np.where(y_true == 0)[0]
    records = []

    for _ in range(B):
        boot_pos = rng.choice(idx_pos, size=len(idx_pos), replace=True)
        boot_neg = rng.choice(idx_neg, size=len(idx_neg), replace=True)
        boot_idx = np.concatenate([boot_pos, boot_neg])
        yt, yp = y_true[boot_idx], y_prob[boot_idx]
        if len(np.unique(yt)) < 2:
            continue
        records.append(compute_threshold_metrics(yt, yp, threshold))

    return pd.DataFrame(records)


# =========================================================================== #
#            SECTION 4 — RESULTS SAVING UTILITIES                              #
# =========================================================================== #

def _ensure_dirs():
    """Create Resultados directory structure."""
    for d in [RESULTS_AUROC, RESULTS_AUPRC, RESULTS_GENERAL]:
        d.mkdir(parents=True, exist_ok=True)


def _save_fig(fig, filename: str, subfolder: Path):
    """Save Plotly figure as HTML."""
    _ensure_dirs()
    path = subfolder / filename
    fig.write_html(str(path), include_plotlyjs="cdn")
    return path


def _save_csv(df: pd.DataFrame, filename: str, subfolder: Path):
    """Save DataFrame as CSV."""
    _ensure_dirs()
    path = subfolder / filename
    df.to_csv(str(path), index=False, encoding="utf-8-sig")
    return path


def _save_json(data, filename: str, subfolder: Path):
    """Save dict as JSON."""
    _ensure_dirs()
    path = subfolder / filename
    with open(str(path), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=_to_native, ensure_ascii=False)
    return path


# =========================================================================== #
#           HELPER — DeLong-style Test                                         #
# =========================================================================== #

def _delong_roc_test(
    y_true: np.ndarray, y_score1: np.ndarray, y_score2: np.ndarray,
    n_boot: int = 2000, seed: int = 42,
) -> tuple[float, float, float, float, float]:
    """DeLong-style test (bootstrap variance) for comparing two AUROCs.

    Formula (Protocol §3.4.3):
    Z = (AUROC₁ − AUROC₂) / √(Var(AUROC₁) + Var(AUROC₂) − 2·Cov(AUROC₁,AUROC₂))

    Returns (auc1, auc2, delta, z_stat, p_value).
    """
    auc1 = roc_auc_score(y_true, y_score1)
    auc2 = roc_auc_score(y_true, y_score2)
    n = len(y_true)
    rng = np.random.RandomState(seed)
    a1_boot, a2_boot = [], []
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        try:
            a1_boot.append(roc_auc_score(y_true[idx], y_score1[idx]))
            a2_boot.append(roc_auc_score(y_true[idx], y_score2[idx]))
        except Exception:
            continue
    a1_boot = np.array(a1_boot)
    a2_boot = np.array(a2_boot)
    var1 = np.var(a1_boot)
    var2 = np.var(a2_boot)
    cov12 = np.cov(a1_boot, a2_boot)[0, 1] if len(a1_boot) > 1 else 0
    se_diff = np.sqrt(max(var1 + var2 - 2 * cov12, 1e-12))
    z_stat = (auc1 - auc2) / se_diff
    p_value = 2 * (1 - sp_stats.norm.cdf(abs(z_stat)))
    return auc1, auc2, auc1 - auc2, z_stat, p_value


def _to_native(obj):
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
#  REUSABLE DISPLAY — Full rigorous pipeline for one subgroup + one metric    #
# =========================================================================== #

def display_metric_pipeline(
    subgroup_label: str,
    distributions: dict[str, np.ndarray],
    point_values: dict[str, float],
    y_true_valid: np.ndarray,
    scale_probs: dict[str, np.ndarray],
    score_col_map: dict[str, str],
    df_scores: pd.DataFrame,
    valid_mask: pd.Series,
    alpha_level: float,
    kp: str,
    metric_name: str,        # "AUROC" or "AUPRC"
    results_folder: Path,    # RESULTS_AUROC or RESULTS_AUPRC
    dist_method: str,        # "bootstrap" or "kfold"
    save_to_disk: bool = True,
):
    """Render the full rigorous statistical pipeline for one metric.

    Implements Protocol §2–§4 completely.
    """
    n_scales = len(distributions)
    metric_lower = metric_name.lower()
    sg_prefix = kp.upper()  # e.g. "STEMI"
    colors_roc = {
        "GRACE": "#1f77b4", "TIMI-STEMI": "#ff7f0e",
        "TIMI-NSTEMI": "#ff7f0e", "RECUIMA": "#2ca02c",
    }
    saved_paths: list[str] = []

    # ── A: Curves ────────────────────────────────────────────────────────
    if metric_name == "AUROC":
        st.markdown("#### 📈 Curvas ROC")
        fig_curve = go.Figure()
        for name, probs in scale_probs.items():
            fpr, tpr, _ = roc_curve(y_true_valid, probs)
            auc_val = point_values[name]
            fig_curve.add_trace(go.Scatter(
                x=fpr, y=tpr, mode="lines",
                name=f"{name} (AUROC={auc_val:.4f})",
                line=dict(color=colors_roc.get(name), width=2.5),
            ))
        fig_curve.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines", name="Referencia (0.5)",
            line=dict(color="gray", dash="dash", width=1),
        ))
        fig_curve.update_layout(
            title=f"Curvas ROC — {subgroup_label}",
            xaxis_title="1 − Especificidad (FPR)", yaxis_title="Sensibilidad (TPR)",
            height=500, legend=dict(x=0.55, y=0.05),
        )
    else:  # AUPRC
        st.markdown("#### 📈 Curvas Precision-Recall")
        fig_curve = go.Figure()
        for name, probs in scale_probs.items():
            prec, rec, _ = precision_recall_curve(y_true_valid, probs)
            ap_val = point_values[name]
            fig_curve.add_trace(go.Scatter(
                x=rec, y=prec, mode="lines",
                name=f"{name} (AUPRC={ap_val:.4f})",
                line=dict(color=colors_roc.get(name), width=2.5),
            ))
        baseline = y_true_valid.mean()
        fig_curve.add_trace(go.Scatter(
            x=[0, 1], y=[baseline, baseline], mode="lines",
            name=f"Prevalencia ({baseline:.3f})",
            line=dict(color="gray", dash="dash", width=1),
        ))
        fig_curve.update_layout(
            title=f"Curvas Precision-Recall — {subgroup_label}",
            xaxis_title="Recall (Sensibilidad)", yaxis_title="Precision (VPP)",
            height=500, legend=dict(x=0.55, y=0.05),
        )

    st.plotly_chart(fig_curve, use_container_width=True, key=f"{kp}_{metric_lower}_curve")
    if save_to_disk:
        p = _save_fig(fig_curve, f"{sg_prefix}_curvas_{metric_lower}.html", results_folder)
        saved_paths.append(str(p))

    # Point metrics table
    metrics_rows = [{"Escala": n, metric_name: f"{v:.4f}"} for n, v in point_values.items()]
    metrics_df = pd.DataFrame(metrics_rows)
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    # ── B: Distributions ────────────────────────────────────────────────
    st.markdown("---")
    method_label = (
        f"Bootstrap Estratificado (B={len(list(distributions.values())[0])})"
        if dist_method == "bootstrap"
        else f"Validación Cruzada Estratificada Repetida (n={len(list(distributions.values())[0])})"
    )
    st.markdown(f"#### 📊 Distribuciones {metric_name} — {method_label}")

    fig_dist = go.Figure()
    scale_order = sorted(
        distributions.keys(),
        key=lambda k: np.mean(distributions[k]),
        reverse=True,
    )
    colors_dist = px.colors.qualitative.Set2

    for idx, name in enumerate(scale_order):
        scores = distributions[name]
        fig_dist.add_trace(go.Violin(
            y=scores, name=name,
            box_visible=True, meanline_visible=True,
            fillcolor=colors_dist[idx % len(colors_dist)],
            opacity=0.6,
            line_color=colors_dist[idx % len(colors_dist)],
            points="all", jitter=0.25, pointpos=-0.2,
        ))
    fig_dist.update_layout(
        title=f"Distribución {metric_name} — {subgroup_label}",
        yaxis_title=metric_name, xaxis_title="Escala",
        showlegend=False, height=500,
    )
    st.plotly_chart(fig_dist, use_container_width=True, key=f"{kp}_{metric_lower}_dist")
    if save_to_disk:
        p = _save_fig(fig_dist, f"{sg_prefix}_distribuciones_{metric_lower}.html", results_folder)
        saved_paths.append(str(p))

    # Descriptive statistics
    desc_rows = []
    for name in scale_order:
        s = distributions[name]
        sk = float(sp_stats.skew(s))
        ku = float(sp_stats.kurtosis(s))
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
            "Asimetría": f"{sk:.3f}",
            "Curtosis": f"{ku:.3f}",
        })
    desc_df = pd.DataFrame(desc_rows)
    st.dataframe(desc_df, use_container_width=True, hide_index=True)
    st.download_button(
        f"📥 Estadísticas Descriptivas {metric_name} (CSV)",
        desc_df.to_csv(index=False),
        f"escalas_{kp}_{metric_lower}_descriptivas.csv", "text/csv",
        key=f"{kp}_{metric_lower}_dl_desc",
    )
    if save_to_disk:
        _save_csv(desc_df, f"{sg_prefix}_estadisticas_descriptivas_{metric_lower}.csv", results_folder)

    # ── C: Normality Tests (Protocol §3.3) ──────────────────────────────
    st.markdown("---")
    st.markdown(f"#### 🧪 Tests de Normalidad — {metric_name} (Criterio 2-de-3)")

    norm_rows = []
    normality_results: dict[str, bool] = {}
    n_normal = 0
    for name in scale_order:
        scores = distributions[name]
        nr = test_normality_full(np.array(scores), alpha=alpha_level)
        normality_results[name] = nr.is_normal
        if nr.is_normal:
            n_normal += 1
        sk = float(sp_stats.skew(scores))
        ku = float(sp_stats.kurtosis(scores))
        norm_rows.append({
            "Escala": name,
            "Shapiro p": f"{nr.shapiro_wilk_pvalue:.4f}",
            "SW": "✅" if nr.shapiro_wilk_normal else "❌",
            "D'Ag p": f"{nr.dagostino_pvalue:.4f}",
            "DA": "✅" if nr.dagostino_normal else "❌",
            "And stat": f"{nr.anderson_statistic:.4f}",
            "AD": "✅" if nr.anderson_normal else "❌",
            "OK": f"{nr.normal_tests_passed}/3",
            "|Asim|<1": "✅" if abs(sk) < 1 else "❌",
            "|Kurt|<3": "✅" if abs(ku) < 3 else "❌",
            "Decisión": "✅ Normal" if nr.is_normal else "❌ No Normal",
        })
    norm_df = pd.DataFrame(norm_rows)
    st.dataframe(norm_df, use_container_width=True, hide_index=True)
    if save_to_disk:
        _save_csv(norm_df, f"{sg_prefix}_normalidad_{metric_lower}.csv", results_folder)

    all_normal = n_normal == n_scales
    if all_normal:
        st.success(f"✅ Todas normales → Tests paramétricos (t-test pareado, ANOVA)")
    else:
        st.warning(
            f"⚠️ {n_normal}/{n_scales} normales → Tests no paramétricos "
            "(Wilcoxon, Friedman)"
        )

    # Q-Q plots
    with st.expander(f"📉 Q-Q Plots — {metric_name}"):
        fig_qq = make_subplots(rows=1, cols=n_scales, subplot_titles=list(scale_order))
        for idx, name in enumerate(scale_order):
            scores = distributions[name]
            theoretical = np.sort(sp_stats.norm.ppf(np.linspace(0.01, 0.99, len(scores))))
            observed = np.sort(scores)
            step = max(1, len(observed) // 300)
            fig_qq.add_trace(
                go.Scatter(
                    x=theoretical[::step], y=observed[::step], mode="markers",
                    marker=dict(size=3, color=colors_dist[idx % len(colors_dist)]),
                    showlegend=False,
                ),
                row=1, col=idx + 1,
            )
            mu, sig = np.mean(scores), np.std(scores)
            mn, mx = min(theoretical), max(theoretical)
            fig_qq.add_trace(
                go.Scatter(
                    x=[mn, mx], y=[mu + sig * mn, mu + sig * mx], mode="lines",
                    line=dict(color="red", dash="dash", width=1), showlegend=False,
                ),
                row=1, col=idx + 1,
            )
        fig_qq.update_layout(height=400, title_text=f"Q-Q Plots {metric_name} — {subgroup_label}")
        st.plotly_chart(fig_qq, use_container_width=True, key=f"{kp}_{metric_lower}_qq")
        if save_to_disk:
            _save_fig(fig_qq, f"{sg_prefix}_qq_plots_{metric_lower}.html", results_folder)

    # Histograms with normal overlay
    with st.expander(f"📊 Histogramas con Curva Normal Teórica — {metric_name}"):
        fig_hist_norm = make_subplots(rows=1, cols=n_scales, subplot_titles=list(scale_order))
        for idx, name in enumerate(scale_order):
            scores = distributions[name]
            fig_hist_norm.add_trace(
                go.Histogram(
                    x=scores, nbinsx=40, name=name,
                    marker_color=colors_dist[idx % len(colors_dist)],
                    opacity=0.7, histnorm="probability density",
                    showlegend=False,
                ),
                row=1, col=idx + 1,
            )
            mu, sig = np.mean(scores), np.std(scores)
            x_range = np.linspace(mu - 4 * sig, mu + 4 * sig, 200)
            y_norm = sp_stats.norm.pdf(x_range, mu, sig)
            fig_hist_norm.add_trace(
                go.Scatter(
                    x=x_range, y=y_norm, mode="lines",
                    line=dict(color="red", width=2), showlegend=False,
                ),
                row=1, col=idx + 1,
            )
        fig_hist_norm.update_layout(
            height=400,
            title_text=f"Histogramas + Normal Teórica {metric_name} — {subgroup_label}",
        )
        st.plotly_chart(fig_hist_norm, use_container_width=True, key=f"{kp}_{metric_lower}_histnorm")
        if save_to_disk:
            _save_fig(fig_hist_norm, f"{sg_prefix}_histogramas_normal_{metric_lower}.html", results_folder)

    # ── D: Global Test (Protocol §3.4) ──────────────────────────────────
    st.markdown("---")
    st.markdown(f"#### 🌐 Test Global — {metric_name}")

    model_scores_dict = {name: list(distributions[name]) for name in scale_order}
    min_len = min(len(v) for v in model_scores_dict.values())
    model_scores_dict = {k: v[:min_len] for k, v in model_scores_dict.items()}

    multiple_comparison = compare_multiple_models(model_scores_dict, alpha=alpha_level)

    col_g1, col_g2, col_g3, col_g4 = st.columns(4)
    with col_g1:
        tname = (
            multiple_comparison.global_test_name.split()[0]
            if not pd.isna(multiple_comparison.global_test_statistic) else "N/A"
        )
        st.metric("Test", tname)
    with col_g2:
        sval = (
            f"{multiple_comparison.global_test_statistic:.3f}"
            if not pd.isna(multiple_comparison.global_test_statistic) else "N/A"
        )
        st.metric("Estadístico", sval)
    with col_g3:
        pstr = (
            f"{multiple_comparison.global_p_value:.6f}"
            if not pd.isna(multiple_comparison.global_p_value) else "N/A"
        )
        st.metric("P-value", pstr)
    with col_g4:
        if pd.isna(multiple_comparison.global_test_statistic):
            st.info("ℹ️ Requiere ≥3 escalas")
        elif multiple_comparison.global_significant:
            st.success("✅ SIGNIFICATIVO")
        else:
            st.warning("❌ No significativo")

    n_pairs = len(multiple_comparison.pairwise_results)
    bonf_alpha = multiple_comparison.bonferroni_alpha

    global_test_df = pd.DataFrame([{
        "Test": str(multiple_comparison.global_test_name),
        "Estadístico": (f"{multiple_comparison.global_test_statistic:.4f}"
                        if not pd.isna(multiple_comparison.global_test_statistic) else "N/A"),
        "P-value": (f"{multiple_comparison.global_p_value:.6f}"
                    if not pd.isna(multiple_comparison.global_p_value) else "N/A"),
        "Significativo": bool(multiple_comparison.global_significant),
        "α Bonferroni": f"{bonf_alpha:.6f}",
        "N comparaciones": n_pairs,
    }])
    if save_to_disk:
        _save_csv(global_test_df, f"{sg_prefix}_test_global_{metric_lower}.csv", results_folder)

    # Nemenyi post-hoc (after Friedman) — Protocol §3.4
    if (not all_normal and n_scales >= 3
            and not pd.isna(multiple_comparison.global_test_statistic)
            and multiple_comparison.global_significant):
        st.markdown("##### 🔬 Post-hoc Nemenyi (tras Friedman)")
        try:
            nemenyi_results = nemenyi_post_hoc(model_scores_dict, alpha=alpha_level)
            nem_rows = []
            for (m1, m2), (rank_diff, is_sig) in nemenyi_results.items():
                nem_rows.append({
                    "Escala 1": m1, "Escala 2": m2,
                    "ΔRango": f"{rank_diff:.3f}",
                    "Significativo": "✅" if is_sig else "❌",
                })
            nem_df = pd.DataFrame(nem_rows)
            st.dataframe(nem_df, use_container_width=True, hide_index=True)
            if save_to_disk:
                _save_csv(nem_df, f"{sg_prefix}_nemenyi_{metric_lower}.csv", results_folder)
        except Exception as e:
            st.warning(f"Nemenyi no disponible: {e}")

    # ── E: Ranking ───────────────────────────────────────────────────────
    st.markdown(f"#### 🏆 Ranking — {metric_name}")

    ranking_rows = []
    for name in scale_order:
        s = distributions[name]
        ranking_rows.append({
            "Escala": name,
            f"Media {metric_name}": np.mean(s),
            f"{metric_name} Puntual": point_values[name],
            "DE": np.std(s),
            "IC95% Inf": np.percentile(s, 2.5),
            "IC95% Sup": np.percentile(s, 97.5),
        })
    ranking_df = (
        pd.DataFrame(ranking_rows)
        .sort_values(f"Media {metric_name}", ascending=False)
        .reset_index(drop=True)
    )
    ranking_df.insert(0, "Pos.", [
        f"🥇 {i+1}" if i == 0 else f"🥈 {i+1}" if i == 1 else f"🥉 {i+1}"
        for i in range(len(ranking_df))
    ])

    best_scale = ranking_df.iloc[0]["Escala"]
    best_mean = ranking_df.iloc[0][f"Media {metric_name}"]
    best_point = ranking_df.iloc[0][f"{metric_name} Puntual"]
    best_ci_lo = ranking_df.iloc[0]["IC95% Inf"]
    best_ci_hi = ranking_df.iloc[0]["IC95% Sup"]

    disp = ranking_df.copy()
    for c in [f"Media {metric_name}", f"{metric_name} Puntual", "DE", "IC95% Inf", "IC95% Sup"]:
        disp[c] = disp[c].apply(lambda x: f"{x:.4f}")
    st.dataframe(disp, use_container_width=True, hide_index=True)
    if save_to_disk:
        _save_csv(ranking_df.round(4), f"{sg_prefix}_ranking_{metric_lower}.csv", results_folder)

    # ── F: Pairwise Comparisons (Protocol §3.4) ─────────────────────────
    st.markdown("---")
    st.markdown(f"#### 🔄 Comparaciones Pareadas Post-hoc — {metric_name}")

    stat_results = multiple_comparison.pairwise_results
    comparison_rows = []
    for (m1, m2), res in stat_results.items():
        holm_p = multiple_comparison.holm_corrected_pvalues.get((m1, m2), res.p_value)
        fdr_p = multiple_comparison.fdr_corrected_pvalues.get((m1, m2), res.p_value)
        sig_bonf = res.p_value < bonf_alpha
        comparison_rows.append({
            "Escala 1": m1, "Escala 2": m2,
            f"Δ{metric_name}": f"{res.delta_auroc:+.4f}",
            "Test": res.test_used.replace(" test", ""),
            "p-value": f"{res.p_value:.6f}",
            "p Holm": f"{holm_p:.6f}",
            "p FDR": f"{fdr_p:.6f}",
            "Sig. (sin corr.)": "✅" if res.significant else "❌",
            "Sig. (Bonferroni)": "✅" if sig_bonf else "❌",
            "Cohen's d": f"{res.effect_size:.3f}",
            "Efecto": res.effect_size_interpretation,
        })
    if comparison_rows:
        comp_df = pd.DataFrame(comparison_rows)
        st.dataframe(comp_df, use_container_width=True, hide_index=True)
        st.caption(
            "**Cohen's d:** Negligible (<0.2) · Small (0.2–0.5) · "
            "Medium (0.5–0.8) · Large (>0.8)"
        )
        st.download_button(
            f"📥 Comparaciones Pareadas {metric_name} (CSV)",
            comp_df.to_csv(index=False),
            f"escalas_{kp}_{metric_lower}_pareadas.csv", "text/csv",
            key=f"{kp}_{metric_lower}_dl_pairs",
        )
        if save_to_disk:
            _save_csv(comp_df, f"{sg_prefix}_comparaciones_pareadas_{metric_lower}.csv", results_folder)

    # ── G: Mann-Whitney U Robustness (Protocol §3.4 Escenario B) ────────
    st.markdown("---")
    st.markdown(f"#### 🔬 Análisis de Robustez — Mann-Whitney U — {metric_name}")
    st.caption(
        "Test de Mann-Whitney U aplicado independientemente como análisis de "
        "robustez, tratando las distribuciones como muestras independientes "
        "para verificar consistencia de resultados (Protocolo §3.4 Escenario B)."
    )

    mw_rows = []
    scale_names = list(scale_probs.keys())
    for i in range(len(scale_names)):
        for j in range(i + 1, len(scale_names)):
            s1n, s2n = scale_names[i], scale_names[j]
            if s1n not in distributions or s2n not in distributions:
                continue
            d1 = distributions[s1n]
            d2 = distributions[s2n]
            try:
                u_stat, p_val, is_sig = mann_whitney_test(d1, d2, alpha=alpha_level)
                sig_bonf_mw = p_val < bonf_alpha
                mw_rows.append({
                    "Escala 1": s1n, "Escala 2": s2n,
                    f"Media {metric_name} 1": f"{np.mean(d1):.4f}",
                    f"Media {metric_name} 2": f"{np.mean(d2):.4f}",
                    "U stat": f"{u_stat:.1f}",
                    "p-value": f"{p_val:.6f}",
                    f"Sig. (α={alpha_level})": "✅" if is_sig else "❌",
                    "Sig. (Bonferroni)": "✅" if sig_bonf_mw else "❌",
                })
            except Exception:
                continue
    if mw_rows:
        mw_df = pd.DataFrame(mw_rows)
        st.dataframe(mw_df, use_container_width=True, hide_index=True)

        # Consistency check
        if comparison_rows:
            consistent = True
            for mw_row in mw_rows:
                mw_sig = mw_row[f"Sig. (α={alpha_level})"] == "✅"
                for cr in comparison_rows:
                    if (cr["Escala 1"] == mw_row["Escala 1"]
                            and cr["Escala 2"] == mw_row["Escala 2"]):
                        paired_sig = cr["Sig. (sin corr.)"] == "✅"
                        if mw_sig != paired_sig:
                            consistent = False
                        break
            if consistent:
                st.success("✅ Consistencia confirmada: Mann-Whitney U y test pareado coinciden.")
            else:
                st.warning(
                    "⚠️ Discrepancia entre Mann-Whitney U y test pareado. "
                    "Los resultados del test pareado son más confiables (muestras dependientes)."
                )
        if save_to_disk:
            _save_csv(mw_df, f"{sg_prefix}_robustez_mannwhitney_{metric_lower}.csv", results_folder)

    # ── H: DeLong Test (Protocol §3.4.3) — AUROC only ──────────────────
    if metric_name == "AUROC":
        st.markdown("---")
        st.markdown("#### 📐 Test de DeLong (Curvas ROC Correlacionadas)")
        st.caption(
            "Compara AUROCs directamente considerando la correlación entre "
            "predicciones de diferentes modelos sobre los mismos pacientes "
            "(DeLong et al., 1988)."
        )

        delong_rows = []
        for i in range(len(scale_names)):
            for j in range(i + 1, len(scale_names)):
                s1n, s2n = scale_names[i], scale_names[j]
                try:
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
                except Exception:
                    continue
        if delong_rows:
            delong_df = pd.DataFrame(delong_rows)
            st.dataframe(delong_df, use_container_width=True, hide_index=True)
            st.download_button(
                "📥 DeLong (CSV)", delong_df.to_csv(index=False),
                f"escalas_{kp}_delong.csv", "text/csv",
                key=f"{kp}_{metric_lower}_dl_delong",
            )
            if save_to_disk:
                _save_csv(delong_df, f"{sg_prefix}_delong.csv", results_folder)

    # ── I: Δmetric Histograms ────────────────────────────────────────────
    with st.expander(f"📊 Histogramas Δ{metric_name} (Bootstrap)"):
        for (m1, m2), _res in stat_results.items():
            if m1 not in model_scores_dict or m2 not in model_scores_dict:
                continue
            s1 = np.array(model_scores_dict[m1])
            s2 = np.array(model_scores_dict[m2])
            diff = s1 - s2
            fig_h = go.Figure()
            fig_h.add_trace(go.Histogram(
                x=diff, nbinsx=50, name=f"Δ{metric_name} ({m1}−{m2})",
                marker_color="#636EFA", opacity=0.75,
            ))
            fig_h.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="0")
            fig_h.add_vline(
                x=np.mean(diff), line_dash="solid", line_color="green",
                annotation_text=f"μ={np.mean(diff):.4f}",
            )
            ci_lo_d, ci_hi_d = np.percentile(diff, [2.5, 97.5])
            fig_h.add_vrect(
                x0=ci_lo_d, x1=ci_hi_d, fillcolor="lightgreen",
                opacity=0.15, annotation_text="IC95%", line_width=0,
            )
            fig_h.update_layout(
                title=f"Δ{metric_name}: {m1} − {m2}",
                xaxis_title=f"Δ{metric_name}", yaxis_title="Frecuencia", height=350,
            )
            st.plotly_chart(fig_h, use_container_width=True, key=f"{kp}_{metric_lower}_hist_{m1}_{m2}")
            ci0 = ci_lo_d <= 0 <= ci_hi_d
            st.caption(
                f"IC 95% [{ci_lo_d:.4f}, {ci_hi_d:.4f}] "
                + ("**incluye 0** → no significativa" if ci0 else "**excluye 0** → significativa")
            )
        if save_to_disk and stat_results:
            # Save combined delta histogram figure (last pair as representative)
            _save_fig(fig_h, f"{sg_prefix}_histogramas_delta_{metric_lower}.html", results_folder)

    # ── J: Conclusion ────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(f"#### 🏅 Conclusión — {metric_name}")

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
            f"- **{n}** (Δ{metric_name}={r.delta_auroc:+.4f}, p={r.p_value:.6f}, d={r.effect_size:.3f})"
            for n, r in inferior_scales
        )
        st.success(f"""
        **🏆 Mejor Escala: {best_scale}** — {metric_name}={best_point:.4f}
        (Distr.: {best_mean:.4f}, IC95%: [{best_ci_lo:.4f}, {best_ci_hi:.4f}])

        ✅ Estadísticamente superior (Bonferroni α={bonf_alpha:.4f}) a:
        {inf_lines}
        """)
    else:
        any_sig = any(r["Sig. (sin corr.)"] == "✅" for r in comparison_rows) if comparison_rows else False
        if any_sig:
            st.warning(f"""
            **🏆 Mayor {metric_name}: {best_scale}** — {best_point:.4f}
            (Distr.: {best_mean:.4f}, IC95%: [{best_ci_lo:.4f}, {best_ci_hi:.4f}])

            ⚠️ Diferencias significativas sin corrección pero no sobreviven Bonferroni.
            """)
        else:
            st.info(f"""
            **🏆 Mayor {metric_name}: {best_scale}** — {best_point:.4f}
            (Distr.: {best_mean:.4f}, IC95%: [{best_ci_lo:.4f}, {best_ci_hi:.4f}])

            ℹ️ No hay diferencias estadísticamente significativas entre las escalas.
            """)

    # ── K: Raw distributions download ────────────────────────────────────
    raw_rows = []
    for name in distributions:
        for i, v in enumerate(distributions[name]):
            raw_rows.append({"Escala": name, "Replica": i + 1, metric_name: v})
    raw_df = pd.DataFrame(raw_rows)
    st.download_button(
        f"📥 Distribuciones Crudas {metric_name} (CSV)",
        raw_df.to_csv(index=False),
        f"escalas_{kp}_{metric_lower}_distribuciones.csv", "text/csv",
        key=f"{kp}_{metric_lower}_dl_raw",
    )
    if save_to_disk:
        _save_csv(raw_df, f"{sg_prefix}_distribuciones_crudas_{metric_lower}.csv", results_folder)

    # ── L: Full JSON Report ──────────────────────────────────────────────
    delong_data = delong_rows if metric_name == "AUROC" else []
    full_report = {
        "metadata": {
            "subgrupo": subgroup_label,
            "metrica": metric_name,
            "metodo_distribucion": dist_method,
            "fecha": datetime.now().isoformat(),
            "n_pacientes": int(valid_mask.sum()),
            "prevalencia_pct": round(float(y_true_valid.mean() * 100), 2),
            "n_muestras": int(len(list(distributions.values())[0])),
            "alpha": float(alpha_level),
        },
        "puntuales": {k: round(float(v), 4) for k, v in point_values.items()},
        "distribucion_stats": {
            name: {
                "mean": round(float(np.mean(s)), 4),
                "std": round(float(np.std(s)), 4),
                "ci95_lo": round(float(np.percentile(s, 2.5)), 4),
                "ci95_hi": round(float(np.percentile(s, 97.5)), 4),
            }
            for name, s in distributions.items()
        },
        "test_global": {
            "nombre": str(multiple_comparison.global_test_name),
            "estadistico": (
                float(multiple_comparison.global_test_statistic)
                if not pd.isna(multiple_comparison.global_test_statistic) else None
            ),
            "p_value": (
                float(multiple_comparison.global_p_value)
                if not pd.isna(multiple_comparison.global_p_value) else None
            ),
            "significativo": bool(multiple_comparison.global_significant),
        },
        "comparaciones_pareadas": [
            {
                "escala1": m1, "escala2": m2,
                f"delta_{metric_lower}": round(float(res.delta_auroc), 4),
                "test": str(res.test_used),
                "p_value": round(float(res.p_value), 6),
                "cohens_d": round(float(res.effect_size), 3),
                "efecto": str(res.effect_size_interpretation),
                "sig_raw": bool(res.significant),
                "sig_bonferroni": bool(res.p_value < bonf_alpha),
            }
            for (m1, m2), res in stat_results.items()
        ],
        "delong_tests": delong_data,
        "robustez_mannwhitney": mw_rows,
        "conclusion": {
            "mejor_escala": str(best_scale),
            f"{metric_lower}_puntual": round(float(best_point), 4),
            f"{metric_lower}_distribucion": round(float(best_mean), 4),
            "estadisticamente_superior": bool(best_is_stat_superior),
        },
    }
    st.download_button(
        f"📥 Reporte Completo {metric_name} (JSON)",
        json.dumps(full_report, indent=2, default=_to_native, ensure_ascii=False),
        f"escalas_{kp}_{metric_lower}_reporte.json", "application/json",
        key=f"{kp}_{metric_lower}_dl_json",
    )
    if save_to_disk:
        _save_json(full_report, f"{sg_prefix}_reporte_completo_{metric_lower}.json", results_folder)

    return saved_paths


# =========================================================================== #
#  DISPLAY — General Metrics (Sensitivity, Specificity, VPP, VPN, F1)         #
# =========================================================================== #

def display_general_metrics(
    subgroup_label: str,
    y_true_valid: np.ndarray,
    scale_probs: dict[str, np.ndarray],
    alpha_level: float,
    kp: str,
    B_boot: int = 1000,
    save_to_disk: bool = True,
):
    """Display secondary metrics with bootstrap CIs. Protocol §2 (descriptive)."""
    sg_prefix = kp.upper()

    st.markdown(f"#### 📋 Métricas Generales — {subgroup_label}")
    st.caption(
        "Análisis descriptivo de métricas secundarias (Sensibilidad, Especificidad, "
        "VPP, VPN, F1-score) al umbral óptimo de Youden. "
        "Intervalos de confianza al 95% por bootstrap estratificado."
    )

    # Point estimates
    point_rows = []
    thresholds = {}
    for name, probs in scale_probs.items():
        thr = _optimal_threshold(y_true_valid, probs)
        thresholds[name] = thr
        m = compute_threshold_metrics(y_true_valid, probs, thr)
        row = {"Escala": name}
        for k, v in m.items():
            row[k] = f"{v:.4f}" if isinstance(v, float) else v
        point_rows.append(row)

    point_df = pd.DataFrame(point_rows)
    st.markdown("##### Estimaciones Puntuales (Umbral Óptimo de Youden)")
    st.dataframe(point_df, use_container_width=True, hide_index=True)
    st.download_button(
        f"📥 Métricas Puntuales (CSV)", point_df.to_csv(index=False),
        f"metricas_{kp}_punto.csv", "text/csv", key=f"{kp}_gm_dl_point",
    )
    if save_to_disk:
        _save_csv(point_df, f"{sg_prefix}_metricas_puntuales.csv", RESULTS_GENERAL)

    # Bootstrap CIs
    st.markdown("##### Intervalos de Confianza 95% (Bootstrap Estratificado)")
    ci_rows = []
    for name, probs in scale_probs.items():
        boot_df = bootstrap_threshold_metrics(
            y_true_valid, probs, B=B_boot, seed=42, threshold=thresholds[name],
        )
        if boot_df.empty:
            continue
        row = {"Escala": name}
        for col in ["Sensibilidad", "Especificidad", "VPP", "VPN", "F1-score", "Exactitud"]:
            if col in boot_df.columns:
                vals = boot_df[col].dropna()
                lo, hi = np.percentile(vals, [2.5, 97.5])
                mu = vals.mean()
                row[f"{col} (Media)"] = f"{mu:.4f}"
                row[f"{col} IC95%"] = f"[{lo:.4f}, {hi:.4f}]"
        ci_rows.append(row)

    ci_df = pd.DataFrame(ci_rows)
    st.dataframe(ci_df, use_container_width=True, hide_index=True)
    st.download_button(
        f"📥 Métricas Bootstrap CI (CSV)", ci_df.to_csv(index=False),
        f"metricas_{kp}_bootstrap_ci.csv", "text/csv", key=f"{kp}_gm_dl_ci",
    )
    if save_to_disk:
        _save_csv(ci_df, f"{sg_prefix}_metricas_bootstrap_ci.csv", RESULTS_GENERAL)

    # Comparison bar chart for each metric
    st.markdown("##### Comparación Visual de Métricas")
    metric_cols = ["Sensibilidad", "Especificidad", "VPP", "VPN", "F1-score"]
    fig_bar = go.Figure()
    colors_bar = px.colors.qualitative.Set2
    for idx, name in enumerate(scale_probs.keys()):
        vals = []
        for mc in metric_cols:
            for pr in point_rows:
                if pr["Escala"] == name:
                    vals.append(float(pr[mc]))
                    break
        fig_bar.add_trace(go.Bar(
            name=name, x=metric_cols, y=vals,
            marker_color=colors_bar[idx % len(colors_bar)],
        ))
    fig_bar.update_layout(
        title=f"Comparación de Métricas Generales — {subgroup_label}",
        yaxis_title="Valor", xaxis_title="Métrica",
        barmode="group", height=450,
    )
    st.plotly_chart(fig_bar, use_container_width=True, key=f"{kp}_gm_bar")
    if save_to_disk:
        _save_fig(fig_bar, f"{sg_prefix}_comparacion_metricas.html", RESULTS_GENERAL)

    # Radar chart
    fig_radar = go.Figure()
    for idx, name in enumerate(scale_probs.keys()):
        vals = []
        for mc in metric_cols:
            for pr in point_rows:
                if pr["Escala"] == name:
                    vals.append(float(pr[mc]))
                    break
        vals.append(vals[0])  # close the polygon
        fig_radar.add_trace(go.Scatterpolar(
            r=vals, theta=metric_cols + [metric_cols[0]],
            fill="toself", name=name, opacity=0.5,
        ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title=f"Perfil de Métricas — {subgroup_label}",
        height=500,
    )
    st.plotly_chart(fig_radar, use_container_width=True, key=f"{kp}_gm_radar")
    if save_to_disk:
        _save_fig(fig_radar, f"{sg_prefix}_perfil_metricas_radar.html", RESULTS_GENERAL)

    # Descriptive comparison table per scale
    st.markdown("##### Tabla Comparativa Resumida")
    summary_rows = []
    for name in scale_probs.keys():
        row = {"Escala": name}
        for pr in point_rows:
            if pr["Escala"] == name:
                row["AUROC"] = pr.get("AUROC", "—")
                for mc in metric_cols + ["Exactitud", "Umbral"]:
                    row[mc] = pr.get(mc, "—")
                break
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)
    st.dataframe(summary_df, use_container_width=True, hide_index=True)
    if save_to_disk:
        _save_csv(summary_df, f"{sg_prefix}_tabla_comparativa_resumen.csv", RESULTS_GENERAL)

    # Full JSON report for general metrics
    gm_report = {
        "subgrupo": subgroup_label,
        "fecha": datetime.now().isoformat(),
        "n_pacientes": int(len(y_true_valid)),
        "prevalencia_pct": round(float(y_true_valid.mean() * 100), 2),
        "metricas_puntuales": point_rows,
        "bootstrap_ci": ci_rows,
    }
    if save_to_disk:
        _save_json(gm_report, f"{sg_prefix}_reporte_metricas_generales.json", RESULTS_GENERAL)


# =========================================================================== #
#  DISPLAY — Detail per Scale (score distributions by outcome)                #
# =========================================================================== #

def display_scale_details(
    scale_order: list[str],
    score_col_map: dict[str, str],
    df_scores: pd.DataFrame,
    valid_mask: pd.Series,
    kp: str,
):
    """Show score distributions by outcome for each scale."""
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
                    fig_s.add_trace(go.Histogram(
                        x=sub, name=lab, marker_color=col, opacity=0.6, nbinsx=30,
                    ))
                fig_s.update_layout(
                    title=f"Score {name} por Desenlace",
                    xaxis_title="Score", yaxis_title="Frecuencia",
                    barmode="overlay", height=350,
                )
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


# =========================================================================== #
#                       SETTINGS & EXECUTION                                   #
# =========================================================================== #
st.markdown("---")
st.subheader("🧮 Configuración y Ejecución del Pipeline")

col_s1, col_s2, col_s3 = st.columns(3)
with col_s1:
    dist_method = st.radio(
        "Método de generación de distribuciones",
        ["Bootstrap Estratificado", "Validación Cruzada Estratificada Repetida"],
        help=(
            "**Bootstrap:** B réplicas estratificadas (Carpenter & Bithell 2000). "
            "**K-Fold Repetido:** k=5 folds × r repeticiones (Protocolo §3.2)."
        ),
        key="dist_method_radio",
    )
with col_s2:
    if dist_method == "Bootstrap Estratificado":
        n_bootstrap = st.number_input(
            "Réplicas Bootstrap (B)", min_value=100, max_value=5000,
            value=1000, step=100, key="n_boot_input",
        )
        n_folds = 5
        n_repeats = 20
    else:
        n_folds = st.number_input(
            "K folds", min_value=2, max_value=10, value=5, step=1, key="k_folds_input",
        )
        n_repeats = st.number_input(
            "Repeticiones (r)", min_value=5, max_value=50, value=20, step=5, key="n_reps_input",
        )
        n_bootstrap = n_folds * n_repeats  # for display purposes
with col_s3:
    alpha_level = st.number_input(
        "Nivel α", min_value=0.01, max_value=0.10, value=0.05, step=0.01, key="alpha_input",
    )
    save_disk = st.checkbox("💾 Guardar resultados en disco", value=True, key="save_disk_cb")

method_key = "bootstrap" if dist_method == "Bootstrap Estratificado" else "kfold"

st.info(
    f"**Método:** {dist_method} | "
    + (f"**B = {n_bootstrap}**" if method_key == "bootstrap"
       else f"**k={n_folds} × r={n_repeats} = {n_folds * n_repeats} estimaciones**")
    + f" | **α = {alpha_level}** | "
    + f"**Métricas:** AUROC (riguroso) + AUPRC (riguroso) + Métricas generales (descriptivo)"
)


# =========================================================================== #
#                     PIPELINE EXECUTION                                       #
# =========================================================================== #

if st.button("🚀 Ejecutar Pipeline Completo", type="primary", use_container_width=True):
    with st.spinner("Calculando escalas y ejecutando pipeline estadístico completo…"):

        # ── Step 0: Compute scales ──
        try:
            df_scores, calc_log = compute_all_scales(df_original)
        except Exception as e:
            st.error(f"❌ Error al calcular escalas: {e}")
            st.exception(e)
            st.stop()

        with st.expander("📋 Log de cálculo de escalas", expanded=True):
            for key, msg in calc_log.items():
                st.markdown(f"- {msg}")

        # ── Step 1: Build subgroups ──
        def _build_subgroup(df, prob_cols, subgroup_mask=None):
            mask = df["_target"].notna()
            for pc in prob_cols:
                mask &= df[pc].notna()
            if subgroup_mask is not None:
                mask &= subgroup_mask
            yt = df.loc[mask, "_target"].values.astype(int)
            sp = {name: df.loc[mask, col].values for col, name in prob_cols.items()}
            return mask, yt, sp

        stemi_prob_cols = {
            "grace_prob": "GRACE", "timi_stemi_prob": "TIMI-STEMI", "recuima_prob": "RECUIMA",
        }
        stemi_mask, stemi_yt, stemi_sp = _build_subgroup(
            df_scores, stemi_prob_cols, df_scores.get("_is_stemi"),
        )
        nstemi_prob_cols = {
            "grace_prob": "GRACE", "timi_nstemi_prob": "TIMI-NSTEMI", "recuima_prob": "RECUIMA",
        }
        nstemi_mask, nstemi_yt, nstemi_sp = _build_subgroup(
            df_scores, nstemi_prob_cols, df_scores.get("_is_nstemi"),
        )

        st.info(
            f"📊 **IAMCEST**: {stemi_mask.sum()} pacientes | "
            f"**IAMSEST**: {nstemi_mask.sum()} pacientes"
        )

        # ── Step 2: Generate distributions for AUROC AND AUPRC ──
        total_steps = 2 * (len(stemi_sp) + len(nstemi_sp))  # auroc + auprc
        progress = st.progress(0, text="Generando distribuciones…")
        step_counter = [0]  # mutable container to avoid nonlocal

        def _gen_dist(yt, probs, metric, seed_offset):
            step_counter[0] += 1
            progress.progress(
                step_counter[0] / total_steps, text=f"{metric.upper()} — generando…",
            )
            return generate_distribution(
                yt, probs, metric=metric, method=method_key,
                B=n_bootstrap if method_key == "bootstrap" else 1000,
                k=n_folds, r=n_repeats,
                seed=42 + seed_offset,
            )

        # STEMI distributions
        stemi_auroc_boot, stemi_auprc_boot = {}, {}
        for i, (name, probs) in enumerate(stemi_sp.items()):
            stemi_auroc_boot[name] = _gen_dist(stemi_yt, probs, "auroc", i)
            stemi_auprc_boot[name] = _gen_dist(stemi_yt, probs, "auprc", 100 + i)

        # NSTEMI distributions
        nstemi_auroc_boot, nstemi_auprc_boot = {}, {}
        for i, (name, probs) in enumerate(nstemi_sp.items()):
            nstemi_auroc_boot[name] = _gen_dist(nstemi_yt, probs, "auroc", 200 + i)
            nstemi_auprc_boot[name] = _gen_dist(nstemi_yt, probs, "auprc", 300 + i)

        progress.progress(1.0, text="✅ Distribuciones completadas")

        # ── Step 3: Point estimates ──
        def _point_metrics(yt, sp_dict):
            aurocs, auprcs = {}, {}
            for name, probs in sp_dict.items():
                try:
                    aurocs[name] = roc_auc_score(yt, probs)
                    auprcs[name] = average_precision_score(yt, probs)
                except Exception:
                    aurocs[name] = np.nan
                    auprcs[name] = np.nan
            return aurocs, auprcs

        stemi_aurocs, stemi_auprcs = _point_metrics(stemi_yt, stemi_sp)
        nstemi_aurocs, nstemi_auprcs = _point_metrics(nstemi_yt, nstemi_sp)

        # ── Step 4: Create results directories ──
        if save_disk:
            _ensure_dirs()

        # ── Store in session state ──
        st.session_state["scales_df"] = df_scores
        st.session_state["scales_alpha"] = alpha_level
        st.session_state["dist_method"] = method_key
        st.session_state["save_disk"] = save_disk
        st.session_state["stemi"] = {
            "auroc_boot": stemi_auroc_boot, "auprc_boot": stemi_auprc_boot,
            "aurocs": stemi_aurocs, "auprcs": stemi_auprcs,
            "yt": stemi_yt, "sp": stemi_sp, "mask": stemi_mask,
            "score_map": {
                "GRACE": "grace_score", "TIMI-STEMI": "timi_stemi_score",
                "RECUIMA": "recuima_score",
            },
        }
        st.session_state["nstemi"] = {
            "auroc_boot": nstemi_auroc_boot, "auprc_boot": nstemi_auprc_boot,
            "aurocs": nstemi_aurocs, "auprcs": nstemi_auprcs,
            "yt": nstemi_yt, "sp": nstemi_sp, "mask": nstemi_mask,
            "score_map": {
                "GRACE": "grace_score", "TIMI-NSTEMI": "timi_nstemi_score",
                "RECUIMA": "recuima_score",
            },
        }
        st.session_state["scales_computed"] = True


# =========================================================================== #
#                          DISPLAY RESULTS                                     #
# =========================================================================== #
if not st.session_state.get("scales_computed", False):
    st.info("👆 Presiona **Ejecutar Pipeline Completo** para calcular escalas y comparar.")
    st.stop()

df_scores = st.session_state["scales_df"]
alpha_level = st.session_state["scales_alpha"]
dist_method_used = st.session_state.get("dist_method", "bootstrap")
save_disk_flag = st.session_state.get("save_disk", True)
d_stemi = st.session_state["stemi"]
d_nstemi = st.session_state["nstemi"]

st.markdown("---")
st.subheader("📊 Resultados por Subgrupo de Infarto")
st.markdown("""
Cada variante TIMI se evalúa **únicamente** en su población objetivo.
GRACE y RECUIMA actúan como comparadores universales en ambos subgrupos.

El pipeline riguroso se aplica a **AUROC** y **AUPRC** por separado.
Las métricas generales se reportan descriptivamente con bootstrap CIs.
""")

tab_stemi, tab_nstemi, tab_ml_comparison = st.tabs([
    f"🫀 IAMCEST — STEMI ({d_stemi['mask'].sum()} pac.)",
    f"🫀 IAMSEST — NSTEMI ({d_nstemi['mask'].sum()} pac.)",
    "🔬 ML vs Escalas Clínicas",
])


def _render_subgroup(d, label, kp_prefix):
    """Render all analyses for one subgroup."""
    st.markdown(
        f"**Subgrupo {label}** — {d['mask'].sum()} pacientes, "
        f"prevalencia mortalidad: {d['yt'].mean()*100:.1f}%"
    )

    # Three main tabs: AUROC, AUPRC, General Metrics
    tab_auroc, tab_auprc, tab_general = st.tabs([
        "📈 AUROC (Riguroso)", "📈 AUPRC (Riguroso)", "📋 Métricas Generales",
    ])

    with tab_auroc:
        st.markdown(f"### Pipeline Riguroso AUROC — {label}")
        display_metric_pipeline(
            subgroup_label=label,
            distributions=d["auroc_boot"],
            point_values=d["aurocs"],
            y_true_valid=d["yt"],
            scale_probs=d["sp"],
            score_col_map=d["score_map"],
            df_scores=df_scores,
            valid_mask=d["mask"],
            alpha_level=alpha_level,
            kp=f"{kp_prefix}_auroc",
            metric_name="AUROC",
            results_folder=RESULTS_AUROC,
            dist_method=dist_method_used,
            save_to_disk=save_disk_flag,
        )
        display_scale_details(
            scale_order=sorted(d["auroc_boot"].keys(),
                               key=lambda k: np.mean(d["auroc_boot"][k]), reverse=True),
            score_col_map=d["score_map"],
            df_scores=df_scores,
            valid_mask=d["mask"],
            kp=f"{kp_prefix}_auroc",
        )

    with tab_auprc:
        st.markdown(f"### Pipeline Riguroso AUPRC — {label}")
        display_metric_pipeline(
            subgroup_label=label,
            distributions=d["auprc_boot"],
            point_values=d["auprcs"],
            y_true_valid=d["yt"],
            scale_probs=d["sp"],
            score_col_map=d["score_map"],
            df_scores=df_scores,
            valid_mask=d["mask"],
            alpha_level=alpha_level,
            kp=f"{kp_prefix}_auprc",
            metric_name="AUPRC",
            results_folder=RESULTS_AUPRC,
            dist_method=dist_method_used,
            save_to_disk=save_disk_flag,
        )

    with tab_general:
        st.markdown(f"### Métricas Generales — {label}")
        display_general_metrics(
            subgroup_label=label,
            y_true_valid=d["yt"],
            scale_probs=d["sp"],
            alpha_level=alpha_level,
            kp=f"{kp_prefix}_gm",
            save_to_disk=save_disk_flag,
        )


with tab_stemi:
    _render_subgroup(d_stemi, "IAMCEST (STEMI)", "stemi")

with tab_nstemi:
    n_deaths_nstemi = d_nstemi["yt"].sum()
    if n_deaths_nstemi < 10:
        st.warning(
            f"⚠️ Solo {n_deaths_nstemi} fallecidos en subgrupo IAMSEST. "
            "Los resultados estadísticos pueden ser inestables."
        )
    _render_subgroup(d_nstemi, "IAMSEST (NSTEMI)", "nstemi")

with tab_ml_comparison:
    st.markdown("### 🔬 Comparación ML vs Escalas Clínicas")
    st.markdown("""
    Aquí puedes ejecutar una **comparación rigurosa** entre un modelo de Machine Learning (XGBoost)
    y las escalas clínicas (GRACE, TIMI, RECUIMA) usando el **ClinicalScaleMLOrchestrator**.
    
    El pipeline incluye:
    - Carga del modelo entrenado y datos de prueba
    - Computación sincronizada de todas las escalas
    - Tests estadísticos DeLong, NRI, IDI
    - Comparación multi-método con Friedman/correcciones Holm
    - Exportación de resultados
    """)
    
    if not ORCHESTRATOR_AVAILABLE:
        st.error("❌ El orquestador no está disponible. Instala las dependencias requeridas.")
    else:
        # ===== Load Model & Test Set =====
        st.subheader("📁 1. Cargar Modelo y Datos de Prueba")
        
        col_model, col_data = st.columns(2)
        with col_model:
            st.markdown("**Modelo ML**")
            model_options = ["best_classifier_mortality", "custom_path"]
            model_choice = st.selectbox(
                "Selecciona modelo",
                model_options,
                help="Usa el mejor modelo entrenado o carga uno personalizado"
            )
            if model_choice == "custom_path":
                model_path = st.text_input("Ruta personalizada al modelo (.joblib)")
            else:
                model_path = None
        
        with col_data:
            st.markdown("**Datos de Prueba**")
            data_source = st.radio(
                "Fuente de datos",
                ["Dataset original cargado", "Archivo personalizado"],
                horizontal=True
            )
            if data_source == "Archivo personalizado":
                uploaded_test = st.file_uploader("Sube datos de prueba", type=["parquet", "csv"])
                test_df_input = None
                if uploaded_test:
                    try:
                        if uploaded_test.name.endswith(".csv"):
                            test_df_input = pd.read_csv(uploaded_test)
                        else:
                            test_df_input = pd.read_parquet(uploaded_test)
                    except Exception as e:
                        st.error(f"Error al cargar: {e}")
            else:
                test_df_input = df_original
        
        # ===== Run Orchestrator =====
        if st.button("▶️ Ejecutar Comparación Completa", key="run_orchestrator"):
            try:
                with st.spinner("⏳ Ejecutando orquestador..."):
                    # Load model
                    if model_choice == "custom_path" and model_path:
                        try:
                            from src.prediction.predictor import load_predictor
                            predictor = load_predictor(model_path=model_path)
                            st.success(f"✅ Modelo cargado: {model_path}")
                        except Exception as e:
                            st.error(f"Error cargando modelo: {e}")
                            st.stop()
                    else:
                        try:
                            from src.prediction.predictor import load_predictor
                            predictor = load_predictor(
                                model_name="best_classifier_mortality"
                            )
                            st.success("✅ Mejor modelo cargado")
                        except Exception as e:
                            st.warning(f"⚠️ No se pudo cargar modelo de  default: {e}")
                            predictor = None
                    
                    # Use provided test data
                    if test_df_input is None:
                        st.error("❌ No hay datos de prueba disponibles")
                        st.stop()
                    
                    # Initialize orchestrator
                    orchestrator = ClinicalScaleMLOrchestrator(
                        model=predictor,
                        model_name="XGBoost Mortality Predictor",
                        test_df=test_df_input,
                        target_col="_target" if "_target" in test_df_input.columns else "muerte_inhospitalaria",
                    )
                    
                    st.info("🔄 Paso 1/4: Obteniendo predicciones del modelo...")
                    orchestrator.get_ml_predictions()
                    
                    st.info("🔄 Paso 2/4: Computando escalas clínicas...")
                    scale_statuses = orchestrator.compute_all_scales()
                    for scale_name, status in scale_statuses.items():
                        if status.success:
                            st.success(f"✅ {scale_name}: {status.n_valid} válidos")
                        else:
                            st.warning(f"⚠️ {scale_name}: {status.error_message}")
                    
                    st.info("🔄 Paso 3/4: Sincronizando predicciones...")
                    valid_mask, sync_preds = orchestrator.synchronize_predictions()
                    st.success(f"✅ {valid_mask.sum()} / {len(valid_mask)} muestras sincronizadas")
                    
                    st.info("🔄 Paso 4/4: Ejecutando comparaciones estadísticas...")
                    results = orchestrator.compare_all(skip_timi=False)
                    
                    st.success("✅ ¡Comparación completada!")
                    
                    # Display summary
                    st.subheader("📊 Resultados Resumidos")
                    summary = orchestrator.generate_summary()
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Muestras Totales", summary.test_set_size)
                    with col2:
                        st.metric("Muestras Sincronizadas", summary.synchronized_size)
                    with col3:
                        st.metric("Eventos", summary.n_events)
                    with col4:
                        st.metric("Tasa de Eventos", f"{summary.event_rate:.1%}")
                    
                    # AUC comparison table
                    st.markdown("#### 🎯 Comparación de AUCs")
                    auc_comparison = pd.DataFrame([
                        {"Método": "ML Model", "AUC": summary.ml_auc if summary.ml_auc > 0 else "N/A"},
                        {"Método": "GRACE", "AUC": round(summary.grace_auc, 4), "P-value": round(summary.grace_p_value, 6)},
                        {"Método": "RECUIMA", "AUC": round(summary.recuima_auc, 4), "P-value": round(summary.recuima_p_value, 6)},
                    ])
                    st.dataframe(auc_comparison, use_container_width=True, hide_index=True)
                    
                    # Detailed results tabs
                    st.markdown("#### 📈 Resultados Detallados")
                    res_tab1, res_tab2, res_tab3 = st.tabs(["GRACE vs ML", "RECUIMA vs ML", "Multi-Método"])
                    
                    with res_tab1:
                        if "grace_vs_ml" in results:
                            result = results["grace_vs_ml"]
                            st.metric("ΔAUC", f"{result.auc_difference:.4f}")
                            st.metric("NRI", f"{result.nri:.4f}")
                            col_a, col_b = st.columns(2)
                            with col_a:
                                st.metric("DeLong p-value", f"{result.auc_p_value:.6f}")
                            with col_b:
                                st.metric("Significativo", "✅ Sí" if result.auc_p_value < 0.05 else "❌ No")
                    
                    with res_tab2:
                        if "recuima_vs_ml" in results:
                            result = results["recuima_vs_ml"]
                            st.metric("ΔAUC", f"{result.auc_difference:.4f}")
                            st.metric("NRI", f"{result.nri:.4f}")
                            col_a, col_b = st.columns(2)
                            with col_a:
                                st.metric("DeLong p-value", f"{result.auc_p_value:.6f}")
                            with col_b:
                                st.metric("Significativo", "✅ Sí" if result.auc_p_value < 0.05 else "❌ No")
                    
                    with res_tab3:
                        if "multiple_methods" in results:
                            result = results["multiple_methods"]
                            st.metric("Friedman p-value", f"{result.global_p_value:.6f}")
                            st.metric("Global Significativo", "✅ Sí" if result.global_significant else "❌ No")
                    
                    # Export option
                    st.markdown("#### 💾 Descargar Resultados")
                    export_dir = RESULTADOS_BASE / "ML_vs_Scales"
                    export_dir.mkdir(parents=True, exist_ok=True)
                    
                    if st.button("💾 Guardar Todos los Resultados"):
                        try:
                            exported = orchestrator.export_results(output_dir=export_dir)
                            st.success(f"✅ Resultados guardados en: {export_dir}")
                            for name, path in exported.items():
                                st.write(f"  • {name}: `{path}`")
                        except Exception as e:
                            st.error(f"Error al guardar: {e}")
                    
            except Exception as e:
                st.error(f"❌ Error en orquestador: {e}")
                import traceback
                st.error(traceback.format_exc())


# =========================================================================== #
#               SAVE ALL RESULTS BUTTON                                        #
# =========================================================================== #
st.markdown("---")
if save_disk_flag:
    st.success(
        f"💾 Resultados guardados automáticamente en:\n\n"
        f"- `{RESULTS_AUROC}`\n"
        f"- `{RESULTS_AUPRC}`\n"
        f"- `{RESULTS_GENERAL}`"
    )
else:
    if st.button("💾 Guardar Todos los Resultados en Disco", key="save_all_btn"):
        _ensure_dirs()
        st.info("Para guardar automáticamente, marca la casilla '💾 Guardar resultados en disco' antes de ejecutar.")


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

    **Métrica Principal: AUROC** (Protocol §2.1)
    - Independencia del umbral, interpretabilidad probabilística
    - Invarianza a la prevalencia, estándar en literatura médica
    - Pipeline riguroso completo (normalidad → test global → pareados → DeLong)

    **Métrica Complementaria: AUPRC** (Protocol §2.2)
    - Más informativa en alto desbalance de clases
    - Penaliza más severamente los falsos positivos
    - Mismo pipeline riguroso que AUROC (sin DeLong)

    **Métricas Secundarias** (Protocol §2, descriptivo):
    Sensibilidad, Especificidad, VPP, VPN, F1-score — reportadas con bootstrap
    CIs al umbral óptimo de Youden.

    **Generación de distribuciones:**
    - **Bootstrap estratificado** (B = 1 000): Carpenter & Bithell (2000)
    - **Validación cruzada estratificada repetida** (k=5 × r=20 = 100): Protocol §3.2

    **Verificación de normalidad** (Protocol §3.3):
    Criterio 2-de-3: Shapiro-Wilk, D'Agostino-Pearson, Anderson-Darling.
    Análisis visual: histogramas + curva normal, Q-Q plots, asimetría/curtosis.

    **Tests estadísticos** (Protocol §3.4):
    - **Paramétricos:** t-test pareado, ANOVA medidas repetidas
    - **No paramétricos:** Wilcoxon signed-rank (pareado), Friedman (global)
    - **Robustez:** Mann-Whitney U (muestras independientes)
    - **Post-hoc:** Nemenyi (tras Friedman), Wilcoxon + corrección Holm
    - **DeLong:** Curvas ROC correlacionadas (solo AUROC)

    **Correcciones por comparaciones múltiples** (Protocol §4.1):
    - Bonferroni: α_ajustado = α / n_comparaciones (conservadora)
    - Holm-Bonferroni: step-down, menos conservadora
    - Benjamini-Hochberg FDR: control de tasa de falso descubrimiento

    **Tamaño del efecto** (Protocol §4.2):
    - Cohen's d: Negligible (<0.2), Small (0.2-0.5), Medium (0.5-0.8), Large (>0.8)
    - Δmétrica: diferencia absoluta entre medias

    ### Escalas clínicas
    - **GRACE** (Fox et al., BMJ 2006): 8 variables, 0-372
    - **TIMI-STEMI** (Morrow et al., Circulation 2000): 8 componentes, 0-14
    - **TIMI-NSTEMI** (Antman et al., JAMA 2000): 7 componentes, 0-7
    - **RECUIMA** (Santos Medina, Tesis Doctoral 2020): 7 variables, 0-10

    ### Referencias
    - Antman, E.M. et al. (2000). TIMI Risk Score for UA/NSTEMI. *JAMA*, 284(7).
    - Carpenter, J. & Bithell, J. (2000). Bootstrap confidence intervals. *Stat in Med*, 19(9).
    - DeLong, E.R. et al. (1988). Comparing ROC curves. *Biometrics*, 44(3).
    - Demšar, J. (2006). Statistical Comparisons of Classifiers. *JMLR*, 7.
    - Fox, K.A. et al. (2006). GRACE risk prediction. *BMJ*, 333.
    - Morrow, D.A. et al. (2000). TIMI Risk Score for STEMI. *Circulation*, 102(17).
    - Santos Medina, M. (2020). Escala RECUIMA. Tesis Doctoral, UCM Santiago de Cuba.
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
    format_func=lambda x: {"grace": "GRACE", "timi": "TIMI", "recuima": "RECUIMA"}.get(x, x),
)

if calc_tab == "grace":
    col1, col2 = st.columns(2)
    with col1:
        age_g = st.number_input("Edad (años)", 0, 120, 65, key="gc_age")
        hr_g = st.number_input("Frecuencia Cardíaca (lpm)", 0, 250, 80, key="gc_hr")
        sbp_g = st.number_input("TAS (mmHg)", 50, 250, 120, key="gc_sbp")
    with col2:
        cr_g = st.number_input("Creatinina (mg/dL)", 0.1, 10.0, 1.0, step=0.1, key="gc_cr")
        killip_g = st.selectbox("Killip", ["I", "II", "III", "IV"], key="gc_killip")
        st_g = st.checkbox("Desviación ST", key="gc_st")
        enz_g = st.checkbox("Enzimas cardíacas elevadas", key="gc_enz")
        arrest_g = st.checkbox("Paro cardíaco al ingreso", key="gc_arrest")

    if st.button("Calcular GRACE", key="btn_grace_calc"):
        scorer = get_score("grace")
        res = scorer.compute(
            age=age_g, heart_rate=hr_g, systolic_bp=sbp_g, creatinine=cr_g,
            killip_class={"I": 1, "II": 2, "III": 3, "IV": 4}[killip_g],
            st_deviation=st_g, elevated_enzymes=enz_g, cardiac_arrest=arrest_g,
        )
        c1, c2 = st.columns(2)
        c1.metric("Score GRACE", f"{res['score']:.0f}")
        risk_map = {"low": "🟢 Bajo", "intermediate": "🟡 Intermedio", "high": "🔴 Alto"}
        c2.metric("Riesgo", risk_map.get(res["risk_category"], res["risk_category"]))

elif calc_tab == "recuima":
    col1, col2 = st.columns(2)
    with col1:
        age_r = st.number_input("Edad (años)", 0, 120, 65, key="rc_age")
        sbp_r = st.number_input("TAS (mmHg)", 30, 250, 120, key="rc_sbp")
        gfr_r = st.number_input("FG (ml/min/1.73m²)", 0.0, 200.0, 90.0, key="rc_gfr")
        ecg_r = st.number_input("Derivaciones ECG afectadas", 0, 12, 2, key="rc_ecg")
    with col2:
        killip_r = st.selectbox("Killip", ["I", "II", "III", "IV"], key="rc_killip")
        vfvt_r = st.checkbox("FV/TV", key="rc_vfvt")
        avb_r = st.checkbox("BAV alto grado", key="rc_avb")

    if st.button("Calcular RECUIMA", key="btn_recuima_calc"):
        scorer = get_score("recuima")
        res = scorer.compute(
            age=age_r, systolic_bp=sbp_r, gfr=gfr_r, ecg_leads_affected=ecg_r,
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
            time_ts = st.number_input("Tiempo de isquemia (min)", 0, 2000, 180, key="ts_time")

        if st.button("Calcular TIMI-STEMI", key="btn_timi_stemi_calc"):
            score_ts = 0
            if age_ts >= 75:
                score_ts += 3
            elif age_ts >= 65:
                score_ts += 2
            score_ts += int(dm_ts)
            if sbp_ts < 100:
                score_ts += 3
            if hr_ts > 100:
                score_ts += 2
            killip_val = {"I": 1, "II": 2, "III": 3, "IV": 4}[killip_ts]
            if killip_val >= 2:
                score_ts += 2
            if peso_ts < 67:
                score_ts += 1
            score_ts += int(ant_ts)
            if time_ts > 240:
                score_ts += 1
            risk_ts = "🟢 Bajo" if score_ts <= 3 else ("🟡 Intermedio" if score_ts <= 6 else "🔴 Alto")
            c1, c2 = st.columns(2)
            c1.metric("Score TIMI-STEMI", f"{score_ts}/14")
            c2.metric("Riesgo", risk_ts)

    else:
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
            risk_tn = "🟢 Bajo" if score_tn <= 2 else ("🟡 Intermedio" if score_tn <= 4 else "🔴 Alto")
            c1, c2 = st.columns(2)
            c1.metric("Score TIMI-NSTEMI", f"{score_tn}/7")
            c2.metric("Riesgo", risk_tn)
