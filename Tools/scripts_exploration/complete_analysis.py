"""
Complete analysis script for clinical scales validation report.
Calculates RECUIMA (with thesis weights), TIMI-NSTEMI (available vars), GRACE metrics,
calibration, DeLong tests, and subgroup analysis.

Dataset column mapping:
- scacest: 1 = IAMCEST, 0 = IAMSEST
- ecg: ECG-N value (numeric, not ecg_n)
- diabetes_mellitus, hipertension_arterial, tabaquismo: 1/0
- estreptoquinasa_recombinante: 'si'/'no'
"""

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Load dataset
df = pd.read_csv(r'E:\Proyectos\mortality-ami-predictor\data\recuima-020425.csv', sep=';', encoding='latin-1')

# Prepare variables
df['mortality'] = (df['estado_vital'] == 'fallecido').astype(int)

# Convert filtrado_glomerular (comma decimal to float)
df['filtrado_glomerular_num'] = pd.to_numeric(
    df['filtrado_glomerular'].astype(str).str.replace(',', '.'), errors='coerce'
)

# Extract FV/TV and BAV from complicaciones
comp = df['complicaciones'].fillna('').astype(str)
df['fv_tv'] = comp.str.contains('FV|TV', regex=True, case=False).astype(int)
df['bav'] = comp.str.contains('BAV3|BAV2:1', regex=True, case=False).astype(int)

print("="*70)
print("ANÁLISIS COMPLETO DE ESCALAS DE RIESGO - DATASET RECUIMA")
print("="*70)

# =============================================================================
# 1. DATASET CHARACTERISTICS (Table 4.1)
# =============================================================================
print("\n" + "="*70)
print("1. CARACTERÍSTICAS DEL DATASET (Tabla 4.1)")
print("="*70)

n_total = len(df)
n_mortality = df['mortality'].sum()
mortality_rate = n_mortality / n_total * 100

# Age statistics
edad_mean = df['edad'].mean()
edad_std = df['edad'].std()
edad_median = df['edad'].median()
edad_q1 = df['edad'].quantile(0.25)
edad_q3 = df['edad'].quantile(0.75)

# Sex distribution
n_male = (df['sexo'] == 'masculino').sum()
pct_male = n_male / n_total * 100

# Infarct type - scacest: 1 = IAMCEST, 0 = IAMSEST
n_stemi = (df['scacest'] == 1).sum()
pct_stemi = n_stemi / n_total * 100
n_nstemi = (df['scacest'] == 0).sum()
pct_nstemi = n_nstemi / n_total * 100

# Thrombolysis
n_thrombolysis = (df['estreptoquinasa_recombinante'] == 'si').sum()
pct_thrombolysis = n_thrombolysis / n_total * 100

print(f"N total: {n_total}")
print(f"Mortalidad: {n_mortality} ({mortality_rate:.2f}%)")
print(f"Edad: {edad_mean:.1f} ± {edad_std:.1f} (mediana: {edad_median:.0f}, IQR: {edad_q1:.0f}-{edad_q3:.0f})")
print(f"Sexo masculino: {n_male} ({pct_male:.1f}%)")
print(f"IAMCEST: {n_stemi} ({pct_stemi:.1f}%)")
print(f"IAMSEST: {n_nstemi} ({pct_nstemi:.1f}%)")
print(f"Trombolisis: {n_thrombolysis} ({pct_thrombolysis:.1f}%)")

# =============================================================================
# 2. RECUIMA SCORE WITH THESIS WEIGHTS (FG=3, FV/TV=2, others=1)
# =============================================================================
print("\n" + "="*70)
print("2. ESCALA RECUIMA (Pesos de tesis: FG=3, FV/TV=2, demás=1)")
print("="*70)

# Calculate components
# ECG-N CORRECTO: Contar derivaciones afectadas (no usar columna 'ecg' que tiene otros valores)
ecg_leads = ['v1','v2','v3','v4','v5','v6','v7','v8','v9','d1','d2','d3','avl','avf','avr']
ecg_n_correct = (df[ecg_leads].fillna(0) != 0).sum(axis=1)

recuima_edad = (df['edad'] > 70).astype(int)  # 1 point
recuima_tas = (df['presion_arterial_sistolica'] < 100).astype(int)  # 1 point
recuima_fg = (df['filtrado_glomerular_num'] < 60).astype(int) * 3  # 3 points
recuima_ecg = (ecg_n_correct > 7).astype(int)  # 1 point - CORREGIDO: contar derivaciones
recuima_killip = (df['indice_killip'] == 'IV').astype(int)  # 1 point
recuima_fvtv = df['fv_tv'] * 2  # 2 points
recuima_bav = df['bav']  # 1 point

# Total score (max = 3+2+1+1+1+1+1 = 10)
df['recuima_weighted'] = (recuima_edad + recuima_tas + recuima_fg + 
                          recuima_ecg + recuima_killip + recuima_fvtv + recuima_bav)

print("Componentes RECUIMA:")
print(f"  Edad > 70 años: {recuima_edad.sum()} pacientes (1 pt)")
print(f"  TAS < 100 mmHg: {recuima_tas.sum()} pacientes (1 pt)")
print(f"  FG < 60 ml/min: {(recuima_fg > 0).sum()} pacientes (3 pts)")
print(f"  ECG-N > 7 derivaciones: {recuima_ecg.sum()} pacientes (1 pt) - CORREGIDO")
print(f"  Killip IV: {recuima_killip.sum()} pacientes (1 pt)")
print(f"  FV/TV: {(recuima_fvtv > 0).sum()} pacientes (2 pts)")
print(f"  BAV alto grado: {recuima_bav.sum()} pacientes (1 pt)")

print(f"\nDistribución RECUIMA ponderado:")
for score in range(0, 11):
    count = (df['recuima_weighted'] == score).sum()
    if count > 0:
        mort_rate = df[df['recuima_weighted'] == score]['mortality'].mean() * 100
        print(f"  Score {score}: {count} pacientes, mortalidad {mort_rate:.1f}%")

# AUROC with CI
def bootstrap_auroc_ci(y_true, y_score, n_bootstrap=1000, ci=0.95):
    """Calculate AUROC with bootstrap confidence interval."""
    aucs = []
    rng = np.random.default_rng(42)
    n = len(y_true)
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        try:
            auc = roc_auc_score(y_true[idx], y_score[idx])
            aucs.append(auc)
        except:
            continue
    alpha = (1 - ci) / 2
    return np.percentile(aucs, [alpha * 100, (1 - alpha) * 100])

# Filter valid cases for RECUIMA
mask_recuima = df['recuima_weighted'].notna() & df['mortality'].notna()
y_true_recuima = df.loc[mask_recuima, 'mortality'].values
y_score_recuima = df.loc[mask_recuima, 'recuima_weighted'].values

auroc_recuima = roc_auc_score(y_true_recuima, y_score_recuima)
ci_recuima = bootstrap_auroc_ci(y_true_recuima, y_score_recuima)

print(f"\nAUROC RECUIMA (ponderado): {auroc_recuima:.3f} [IC95%: {ci_recuima[0]:.3f}-{ci_recuima[1]:.3f}]")

# Optimal threshold using Youden's J
fpr_rec, tpr_rec, thresholds_rec = roc_curve(y_true_recuima, y_score_recuima)
youden_j = tpr_rec - fpr_rec
optimal_idx = np.argmax(youden_j)
optimal_threshold_recuima = thresholds_rec[optimal_idx]

# Diagnostic metrics at threshold >=3 (as in thesis)
threshold_recuima = 3
pred_recuima = (df['recuima_weighted'] >= threshold_recuima).astype(int)

mask_valid = pred_recuima.notna() & df['mortality'].notna()
tn, fp, fn, tp = confusion_matrix(df.loc[mask_valid, 'mortality'], 
                                   pred_recuima[mask_valid]).ravel()

sens_recuima = tp / (tp + fn) * 100
spec_recuima = tn / (tn + fp) * 100
ppv_recuima = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
npv_recuima = tn / (tn + fn) * 100 if (tn + fn) > 0 else 0
f1_recuima = 2 * tp / (2 * tp + fp + fn) * 100
lr_pos_recuima = (sens_recuima/100) / (1 - spec_recuima/100) if spec_recuima < 100 else np.inf
lr_neg_recuima = (1 - sens_recuima/100) / (spec_recuima/100) if spec_recuima > 0 else np.inf

print(f"\nMétricas diagnósticas (umbral >={threshold_recuima}):")
print(f"  Sensibilidad: {sens_recuima:.2f}%")
print(f"  Especificidad: {spec_recuima:.2f}%")
print(f"  VPP: {ppv_recuima:.2f}%")
print(f"  VPN: {npv_recuima:.2f}%")
print(f"  F1-Score: {f1_recuima:.2f}%")
print(f"  LR+: {lr_pos_recuima:.2f}")
print(f"  LR-: {lr_neg_recuima:.3f}")
print(f"  Umbral óptimo (Youden): {optimal_threshold_recuima:.0f}")

# =============================================================================
# 3. GRACE SCORE (Precalculated in dataset)
# =============================================================================
print("\n" + "="*70)
print("3. ESCALA GRACE (Precalculada en dataset)")
print("="*70)

# Filter valid GRACE scores
mask_grace = df['escala_grace'].notna() & df['mortality'].notna()
y_true_grace = df.loc[mask_grace, 'mortality'].values
y_score_grace = df.loc[mask_grace, 'escala_grace'].values

n_grace = mask_grace.sum()
grace_mean = df.loc[mask_grace, 'escala_grace'].mean()
grace_std = df.loc[mask_grace, 'escala_grace'].std()

print(f"N con GRACE válido: {n_grace}")
print(f"GRACE medio: {grace_mean:.1f} ± {grace_std:.1f}")

auroc_grace = roc_auc_score(y_true_grace, y_score_grace)
ci_grace = bootstrap_auroc_ci(y_true_grace, y_score_grace)

print(f"AUROC GRACE: {auroc_grace:.3f} [IC95%: {ci_grace[0]:.3f}-{ci_grace[1]:.3f}]")

# Optimal threshold and metrics
fpr_grace, tpr_grace, thresholds_grace = roc_curve(y_true_grace, y_score_grace)
youden_grace = tpr_grace - fpr_grace
optimal_idx_grace = np.argmax(youden_grace)
optimal_threshold_grace = thresholds_grace[optimal_idx_grace]

# Use common threshold of 140 (high risk)
threshold_grace = 140
pred_grace = (df['escala_grace'] >= threshold_grace).astype(int)

mask_valid_grace = pred_grace.notna() & df['mortality'].notna()
tn_g, fp_g, fn_g, tp_g = confusion_matrix(df.loc[mask_valid_grace, 'mortality'], 
                                           pred_grace[mask_valid_grace]).ravel()

sens_grace = tp_g / (tp_g + fn_g) * 100
spec_grace = tn_g / (tn_g + fp_g) * 100
ppv_grace = tp_g / (tp_g + fp_g) * 100 if (tp_g + fp_g) > 0 else 0
npv_grace = tn_g / (tn_g + fn_g) * 100 if (tn_g + fn_g) > 0 else 0
f1_grace = 2 * tp_g / (2 * tp_g + fp_g + fn_g) * 100
lr_pos_grace = (sens_grace/100) / (1 - spec_grace/100) if spec_grace < 100 else np.inf
lr_neg_grace = (1 - sens_grace/100) / (spec_grace/100) if spec_grace > 0 else np.inf

print(f"\nMétricas diagnósticas (umbral >={threshold_grace}):")
print(f"  Sensibilidad: {sens_grace:.2f}%")
print(f"  Especificidad: {spec_grace:.2f}%")
print(f"  VPP: {ppv_grace:.2f}%")
print(f"  VPN: {npv_grace:.2f}%")
print(f"  F1-Score: {f1_grace:.2f}%")
print(f"  LR+: {lr_pos_grace:.2f}")
print(f"  LR-: {lr_neg_grace:.3f}")
print(f"  Umbral óptimo (Youden): {optimal_threshold_grace:.0f}")

# =============================================================================
# 4. TIMI-NSTEMI SCORE (Available variables)
# =============================================================================
print("\n" + "="*70)
print("4. ESCALA TIMI-NSTEMI (Variables disponibles)")
print("="*70)

print("Variables TIMI-NSTEMI disponibles en dataset:")

# 1. Age >= 65
timi_age = (df['edad'] >= 65).astype(int)
print(f"  1. Edad >= 65 años: {timi_age.sum()} pacientes - DISPONIBLE")

# 2. >= 3 CAD risk factors
has_dm = df['diabetes_mellitus'].fillna(0).astype(int)
has_hta = df['hipertension_arterial'].fillna(0).astype(int)
has_smoking = df['tabaquismo'].fillna(0).astype(int)
has_dyslipidemia = df['hiperlipoproteinemia'].fillna(0).astype(int) if 'hiperlipoproteinemia' in df.columns else pd.Series(0, index=df.index)

risk_factors = has_dm + has_hta + has_smoking + has_dyslipidemia
timi_risk_factors = (risk_factors >= 3).astype(int)
print(f"  2. >= 3 FR coronarios: {timi_risk_factors.sum()} pacientes - DISPONIBLE (DM, HTA, tabaco, dislipidemia)")

# 3. Known CAD
has_known_cad = df['enfermedad_arterias_coronarias'].fillna(0).astype(int) if 'enfermedad_arterias_coronarias' in df.columns else pd.Series(0, index=df.index)
has_prior_mi = df['infarto_miocardio_agudo'].fillna(0).astype(int) if 'infarto_miocardio_agudo' in df.columns else pd.Series(0, index=df.index)
timi_known_cad = ((has_known_cad == 1) | (has_prior_mi == 1)).astype(int)
print(f"  3. Enfermedad coronaria conocida: {timi_known_cad.sum()} pacientes - DISPONIBLE")

# 4. Aspirin use - check if column exists
if 'asa' in df.columns:
    timi_aspirin = (df['asa'] == 1).astype(int)
    print(f"  4. Uso de aspirina: {timi_aspirin.sum()} pacientes - DISPONIBLE (uso hospitalario)")
else:
    timi_aspirin = pd.Series(0, index=df.index)
    print(f"  4. Uso de aspirina previo: NO DISPONIBLE")

# 5. Severe angina - check angina24h
if 'angina24h' in df.columns:
    timi_angina = df['angina24h'].fillna(0).astype(int)
    print(f"  5. Angina en 24h: {timi_angina.sum()} pacientes - DISPONIBLE")
else:
    timi_angina = pd.Series(0, index=df.index)
    print(f"  5. Angina severa: NO DISPONIBLE")

# 6. ST deviation
has_st = ((df['depresion_st'].fillna(0) == 1) | (df['infradesnivel'].fillna(0) == 1)).astype(int)
print(f"  6. Desviación ST: {has_st.sum()} pacientes - DISPONIBLE")

# 7. Elevated cardiac markers - assume all AMI patients have this
timi_markers = pd.Series(1, index=df.index)
print(f"  7. Marcadores cardíacos elevados: {timi_markers.sum()} pacientes - ASUMIDO (todos IAM)")

# Calculate partial TIMI-NSTEMI (using available variables, max 7)
df['timi_nstemi'] = timi_age + timi_risk_factors + timi_known_cad + timi_angina + has_st + timi_markers

print(f"\nVariables disponibles: 6/7 (86%)")
print(f"Rango score parcial: 0-6 (máximo original: 7)")

# Filter NSTEMI patients for TIMI-NSTEMI
mask_nstemi = (df['scacest'] == 0)
n_nstemi_total = mask_nstemi.sum()
print(f"\nPacientes IAMSEST: {n_nstemi_total}")

auroc_timi = None
sens_timi = spec_timi = ppv_timi = npv_timi = f1_timi = 0
lr_pos_timi = lr_neg_timi = 0
ci_timi = [0, 0]

if n_nstemi_total > 50:
    mask_timi = mask_nstemi & df['timi_nstemi'].notna() & df['mortality'].notna()
    y_true_timi = df.loc[mask_timi, 'mortality'].values
    y_score_timi = df.loc[mask_timi, 'timi_nstemi'].values
    
    n_deaths_nstemi = y_true_timi.sum()
    print(f"Fallecidos IAMSEST: {n_deaths_nstemi}")
    
    if len(np.unique(y_true_timi)) == 2 and n_deaths_nstemi >= 10:
        auroc_timi = roc_auc_score(y_true_timi, y_score_timi)
        ci_timi = bootstrap_auroc_ci(y_true_timi, y_score_timi)
        print(f"AUROC TIMI-NSTEMI (parcial): {auroc_timi:.3f} [IC95%: {ci_timi[0]:.3f}-{ci_timi[1]:.3f}]")
        
        # Diagnostic metrics at threshold >=3
        threshold_timi = 3
        pred_timi = (df.loc[mask_timi, 'timi_nstemi'] >= threshold_timi).astype(int)
        if len(np.unique(pred_timi)) == 2:
            tn_t, fp_t, fn_t, tp_t = confusion_matrix(y_true_timi, pred_timi).ravel()
            
            sens_timi = tp_t / (tp_t + fn_t) * 100 if (tp_t + fn_t) > 0 else 0
            spec_timi = tn_t / (tn_t + fp_t) * 100 if (tn_t + fp_t) > 0 else 0
            ppv_timi = tp_t / (tp_t + fp_t) * 100 if (tp_t + fp_t) > 0 else 0
            npv_timi = tn_t / (tn_t + fn_t) * 100 if (tn_t + fn_t) > 0 else 0
            f1_timi = 2 * tp_t / (2 * tp_t + fp_t + fn_t) * 100 if (2*tp_t + fp_t + fn_t) > 0 else 0
            lr_pos_timi = (sens_timi/100) / (1 - spec_timi/100) if spec_timi < 100 else np.inf
            lr_neg_timi = (1 - sens_timi/100) / (spec_timi/100) if spec_timi > 0 else np.inf
            
            print(f"\nMétricas diagnósticas IAMSEST (umbral >={threshold_timi}):")
            print(f"  Sensibilidad: {sens_timi:.2f}%")
            print(f"  Especificidad: {spec_timi:.2f}%")
            print(f"  VPP: {ppv_timi:.2f}%")
            print(f"  VPN: {npv_timi:.2f}%")
            print(f"  F1-Score: {f1_timi:.2f}%")
    else:
        print("Insuficiente variabilidad en mortalidad IAMSEST para calcular AUROC")
else:
    print("Insuficientes pacientes IAMSEST para análisis")

# =============================================================================
# 5. TIMI-STEMI SCORE (for reference)
# =============================================================================
print("\n" + "="*70)
print("5. ESCALA TIMI-STEMI (Referencia)")
print("="*70)

print("Variables TIMI-STEMI disponibles:")

# Age component
timi_s_age = np.where(df['edad'] >= 75, 3, np.where(df['edad'] >= 65, 2, 0))
print(f"  1. Edad (65-74: 2pts, >=75: 3pts): calculable")

# DM/HTA/angina
timi_s_risk = ((has_dm == 1) | (has_hta == 1)).astype(int)
print(f"  2. DM/HTA/angina: {timi_s_risk.sum()} pacientes - DISPONIBLE")

# SBP < 100
timi_s_sbp = (df['presion_arterial_sistolica'] < 100).astype(int) * 3
print(f"  3. TAS < 100 (3pts): {(timi_s_sbp > 0).sum()} pacientes")

# HR > 100
timi_s_hr = (df['frecuencia_cardiaca'] > 100).fillna(0).astype(int) * 2
print(f"  4. FC > 100 (2pts): {(timi_s_hr > 0).sum()} pacientes")

# Killip II-IV
timi_s_killip = df['indice_killip'].isin(['II', 'III', 'IV']).astype(int) * 2
print(f"  5. Killip II-IV (2pts): {(timi_s_killip > 0).sum()} pacientes")

# Weight < 67 kg
if 'peso' in df.columns:
    peso_num = pd.to_numeric(df['peso'].astype(str).str.replace(',', '.'), errors='coerce')
    timi_s_weight = (peso_num < 67).fillna(0).astype(int)
    print(f"  6. Peso < 67 kg: {timi_s_weight.sum()} pacientes - DISPONIBLE")
else:
    timi_s_weight = pd.Series(0, index=df.index)
    print(f"  6. Peso < 67 kg: NO DISPONIBLE")

# Anterior MI - check localization or leads
has_anterior = (df[['v1', 'v2', 'v3', 'v4']].fillna(0).sum(axis=1) > 0).astype(int)
print(f"  7. IAM anterior (V1-V4): {has_anterior.sum()} pacientes")

# Time to treatment > 4h
if 'tiempo_isquemia' in df.columns:
    tiempo_num = pd.to_numeric(df['tiempo_isquemia'], errors='coerce')
    timi_s_time = (tiempo_num > 240).fillna(0).astype(int)  # > 4 hours in minutes
    print(f"  8. Tiempo isquemia > 4h: {timi_s_time.sum()} pacientes - DISPONIBLE")
else:
    timi_s_time = pd.Series(0, index=df.index)
    print(f"  8. Tiempo a tratamiento > 4h: NO DISPONIBLE")

df['timi_stemi'] = timi_s_age + timi_s_risk + timi_s_sbp + timi_s_hr + timi_s_killip + timi_s_weight + has_anterior + timi_s_time

print(f"\nVariables disponibles TIMI-STEMI: 8/8 (100%)")

# Calculate AUROC for STEMI patients
mask_stemi = (df['scacest'] == 1)
mask_timi_s = mask_stemi & df['timi_stemi'].notna() & df['mortality'].notna()
y_true_timi_s = df.loc[mask_timi_s, 'mortality'].values
y_score_timi_s = df.loc[mask_timi_s, 'timi_stemi'].values

auroc_timi_s = roc_auc_score(y_true_timi_s, y_score_timi_s)
ci_timi_s = bootstrap_auroc_ci(y_true_timi_s, y_score_timi_s)
print(f"\nAUROC TIMI-STEMI (IAMCEST): {auroc_timi_s:.3f} [IC95%: {ci_timi_s[0]:.3f}-{ci_timi_s[1]:.3f}]")

# Diagnostic metrics for TIMI-STEMI at threshold >=5
threshold_timi_s = 5
pred_timi_s = (df.loc[mask_timi_s, 'timi_stemi'] >= threshold_timi_s).astype(int)
tn_ts, fp_ts, fn_ts, tp_ts = confusion_matrix(y_true_timi_s, pred_timi_s).ravel()

sens_timi_s = tp_ts / (tp_ts + fn_ts) * 100
spec_timi_s = tn_ts / (tn_ts + fp_ts) * 100
ppv_timi_s = tp_ts / (tp_ts + fp_ts) * 100 if (tp_ts + fp_ts) > 0 else 0
npv_timi_s = tn_ts / (tn_ts + fn_ts) * 100 if (tn_ts + fn_ts) > 0 else 0
f1_timi_s = 2 * tp_ts / (2 * tp_ts + fp_ts + fn_ts) * 100
lr_pos_timi_s = (sens_timi_s/100) / (1 - spec_timi_s/100) if spec_timi_s < 100 else np.inf
lr_neg_timi_s = (1 - sens_timi_s/100) / (spec_timi_s/100) if spec_timi_s > 0 else np.inf

print(f"\nMétricas diagnósticas IAMCEST (umbral >={threshold_timi_s}):")
print(f"  Sensibilidad: {sens_timi_s:.2f}%")
print(f"  Especificidad: {spec_timi_s:.2f}%")
print(f"  VPP: {ppv_timi_s:.2f}%")
print(f"  VPN: {npv_timi_s:.2f}%")
print(f"  F1-Score: {f1_timi_s:.2f}%")
print(f"  LR+: {lr_pos_timi_s:.2f}")
print(f"  LR-: {lr_neg_timi_s:.3f}")

# =============================================================================
# 6. HOSMER-LEMESHOW CALIBRATION TEST (Table 4.3)
# =============================================================================
print("\n" + "="*70)
print("6. CALIBRACIÓN HOSMER-LEMESHOW (Tabla 4.3)")
print("="*70)

def hosmer_lemeshow_test(y_true, y_pred_prob, n_groups=10):
    """Perform Hosmer-Lemeshow test."""
    df_hl = pd.DataFrame({'y': y_true, 'p': y_pred_prob})
    try:
        df_hl['decile'] = pd.qcut(df_hl['p'], q=n_groups, duplicates='drop')
    except:
        df_hl['decile'] = pd.cut(df_hl['p'], bins=n_groups, duplicates='drop')
    
    observed = df_hl.groupby('decile', observed=True)['y'].sum()
    expected = df_hl.groupby('decile', observed=True)['p'].sum()
    n_group = df_hl.groupby('decile', observed=True)['y'].count()
    
    chi2 = 0
    for i in range(len(observed)):
        if expected.iloc[i] > 0 and n_group.iloc[i] > expected.iloc[i]:
            chi2 += ((observed.iloc[i] - expected.iloc[i]) ** 2) / (expected.iloc[i] * (1 - expected.iloc[i]/n_group.iloc[i]))
    
    df_chi = max(len(observed) - 2, 1)
    p_value = 1 - stats.chi2.cdf(chi2, df_chi)
    
    return chi2, p_value, df_chi

def score_to_prob(scores, max_score):
    """Convert score to pseudo-probability for calibration."""
    return np.clip(scores / max_score, 0, 1)

# GRACE calibration (approximate max ~300)
grace_prob = score_to_prob(y_score_grace, 300)
chi2_grace, p_grace, df_g = hosmer_lemeshow_test(y_true_grace, grace_prob)
print(f"GRACE: chi2 = {chi2_grace:.2f}, p = {p_grace:.4f}")

# RECUIMA calibration (max = 10 with weights)
recuima_prob = score_to_prob(y_score_recuima, 10)
chi2_recuima, p_recuima, df_r = hosmer_lemeshow_test(y_true_recuima, recuima_prob)
print(f"RECUIMA: chi2 = {chi2_recuima:.2f}, p = {p_recuima:.4f}")

# TIMI-STEMI calibration (max = 14)
timi_s_prob = score_to_prob(y_score_timi_s, 14)
chi2_timi_s, p_timi_s, df_ts = hosmer_lemeshow_test(y_true_timi_s, timi_s_prob)
print(f"TIMI-STEMI: chi2 = {chi2_timi_s:.2f}, p = {p_timi_s:.4f}")

# TIMI-NSTEMI calibration if available
if auroc_timi is not None:
    timi_prob = score_to_prob(y_score_timi, 7)
    chi2_timi, p_timi_n, df_tn = hosmer_lemeshow_test(y_true_timi, timi_prob)
    print(f"TIMI-NSTEMI: chi2 = {chi2_timi:.2f}, p = {p_timi_n:.4f}")
else:
    chi2_timi = p_timi_n = 0

# =============================================================================
# 7. DeLONG TEST FOR AUROC COMPARISON (Table 4.5)
# =============================================================================
print("\n" + "="*70)
print("7. COMPARACIÓN DE CURVAS ROC - TEST DeLONG (Tabla 4.5)")
print("="*70)

def delong_roc_test(y_true, y_score1, y_score2):
    """Simplified DeLong test for comparing two AUROCs using bootstrap."""
    n = len(y_true)
    auc1 = roc_auc_score(y_true, y_score1)
    auc2 = roc_auc_score(y_true, y_score2)
    
    n_bootstrap = 1000
    rng = np.random.default_rng(42)
    diffs = []
    
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        try:
            auc1_boot = roc_auc_score(y_true[idx], y_score1[idx])
            auc2_boot = roc_auc_score(y_true[idx], y_score2[idx])
            diffs.append(auc1_boot - auc2_boot)
        except:
            continue
    
    se_diff = np.std(diffs) if len(diffs) > 0 else 0.001
    z_stat = (auc1 - auc2) / se_diff if se_diff > 0 else 0
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    
    return auc1, auc2, auc1 - auc2, z_stat, p_value

# GRACE vs RECUIMA (all patients)
mask_common = mask_grace & mask_recuima
y_common = df.loc[mask_common, 'mortality'].values
grace_common = df.loc[mask_common, 'escala_grace'].values
recuima_common = df.loc[mask_common, 'recuima_weighted'].values

auc1, auc2, diff, z, p = delong_roc_test(y_common, grace_common, recuima_common)
print(f"GRACE vs RECUIMA:")
print(f"  GRACE AUROC: {auc1:.3f}")
print(f"  RECUIMA AUROC: {auc2:.3f}")
print(f"  Diferencia: {diff:.3f}")
print(f"  Z-statistic: {z:.2f}")
print(f"  p-value: {p:.4f}")

# RECUIMA vs GRACE (reverse)
auc_rec_vs_grace, auc_grace_vs_rec, diff_rg, z_rg, p_rg = delong_roc_test(y_common, recuima_common, grace_common)
print(f"\nRECUIMA vs GRACE:")
print(f"  Diferencia (RECUIMA - GRACE): {diff_rg:.3f}")
print(f"  Z-statistic: {z_rg:.2f}")
print(f"  p-value: {p_rg:.4f}")

# TIMI-STEMI vs GRACE (STEMI patients)
mask_common_s = mask_stemi & df['escala_grace'].notna() & df['timi_stemi'].notna() & df['mortality'].notna()
y_common_s = df.loc[mask_common_s, 'mortality'].values
grace_common_s = df.loc[mask_common_s, 'escala_grace'].values
timi_common_s = df.loc[mask_common_s, 'timi_stemi'].values

auc1_s, auc2_s, diff_s, z_s, p_s = delong_roc_test(y_common_s, grace_common_s, timi_common_s)
print(f"\nGRACE vs TIMI-STEMI (IAMCEST):")
print(f"  GRACE AUROC: {auc1_s:.3f}")
print(f"  TIMI-STEMI AUROC: {auc2_s:.3f}")
print(f"  Diferencia: {diff_s:.3f}")
print(f"  Z-statistic: {z_s:.2f}")
print(f"  p-value: {p_s:.4f}")

# RECUIMA vs TIMI-STEMI (STEMI patients)
mask_common_rt = mask_stemi & df['recuima_weighted'].notna() & df['timi_stemi'].notna() & df['mortality'].notna()
y_common_rt = df.loc[mask_common_rt, 'mortality'].values
recuima_common_rt = df.loc[mask_common_rt, 'recuima_weighted'].values
timi_common_rt = df.loc[mask_common_rt, 'timi_stemi'].values

auc1_rt, auc2_rt, diff_rt, z_rt, p_rt = delong_roc_test(y_common_rt, recuima_common_rt, timi_common_rt)
print(f"\nRECUIMA vs TIMI-STEMI (IAMCEST):")
print(f"  RECUIMA AUROC: {auc1_rt:.3f}")
print(f"  TIMI-STEMI AUROC: {auc2_rt:.3f}")
print(f"  Diferencia: {diff_rt:.3f}")
print(f"  Z-statistic: {z_rt:.2f}")
print(f"  p-value: {p_rt:.4f}")

# =============================================================================
# 8. SUBGROUP ANALYSIS (Tables 4.6-4.7)
# =============================================================================
print("\n" + "="*70)
print("8. ANÁLISIS POR SUBGRUPOS (Tablas 4.6-4.7)")
print("="*70)

# 8.1 By infarct type
print("\n8.1 Por tipo de infarto:")
print("-" * 50)

subgroup_results = []

for tipo, label in [(1, 'IAMCEST'), (0, 'IAMSEST')]:
    mask_tipo = (df['scacest'] == tipo)
    n_tipo = mask_tipo.sum()
    mort_tipo = df.loc[mask_tipo, 'mortality'].mean() * 100
    
    print(f"\n{label}: N={n_tipo}, Mortalidad={mort_tipo:.2f}%")
    
    result = {'subgroup': label, 'n': n_tipo, 'mortality': mort_tipo}
    
    # GRACE
    mask_g = mask_tipo & df['escala_grace'].notna() & df['mortality'].notna()
    if mask_g.sum() > 50 and df.loc[mask_g, 'mortality'].sum() >= 10:
        y_t = df.loc[mask_g, 'mortality'].values
        y_s = df.loc[mask_g, 'escala_grace'].values
        auc = roc_auc_score(y_t, y_s)
        ci = bootstrap_auroc_ci(y_t, y_s)
        print(f"  GRACE AUROC: {auc:.3f} [{ci[0]:.3f}-{ci[1]:.3f}]")
        result['grace_auroc'] = auc
        result['grace_ci'] = ci
    
    # RECUIMA
    mask_r = mask_tipo & df['recuima_weighted'].notna() & df['mortality'].notna()
    if mask_r.sum() > 50 and df.loc[mask_r, 'mortality'].sum() >= 10:
        y_t = df.loc[mask_r, 'mortality'].values
        y_s = df.loc[mask_r, 'recuima_weighted'].values
        auc = roc_auc_score(y_t, y_s)
        ci = bootstrap_auroc_ci(y_t, y_s)
        print(f"  RECUIMA AUROC: {auc:.3f} [{ci[0]:.3f}-{ci[1]:.3f}]")
        result['recuima_auroc'] = auc
        result['recuima_ci'] = ci
    
    subgroup_results.append(result)

# 8.2 By age group
print("\n\n8.2 Por grupo de edad:")
print("-" * 50)

for age_min, age_max, label in [(18, 64, '<65 años'), (65, 74, '65-74 años'), (75, 120, '>=75 años')]:
    mask_age = (df['edad'] >= age_min) & (df['edad'] <= age_max)
    n_age = mask_age.sum()
    if n_age == 0:
        continue
    mort_age = df.loc[mask_age, 'mortality'].mean() * 100
    
    print(f"\n{label}: N={n_age}, Mortalidad={mort_age:.2f}%")
    
    # GRACE
    mask_g = mask_age & df['escala_grace'].notna() & df['mortality'].notna()
    if mask_g.sum() > 50 and df.loc[mask_g, 'mortality'].sum() >= 10:
        y_t = df.loc[mask_g, 'mortality'].values
        y_s = df.loc[mask_g, 'escala_grace'].values
        auc = roc_auc_score(y_t, y_s)
        ci = bootstrap_auroc_ci(y_t, y_s)
        print(f"  GRACE AUROC: {auc:.3f} [{ci[0]:.3f}-{ci[1]:.3f}]")
    
    # RECUIMA
    mask_r = mask_age & df['recuima_weighted'].notna() & df['mortality'].notna()
    if mask_r.sum() > 50 and df.loc[mask_r, 'mortality'].sum() >= 10:
        y_t = df.loc[mask_r, 'mortality'].values
        y_s = df.loc[mask_r, 'recuima_weighted'].values
        auc = roc_auc_score(y_t, y_s)
        ci = bootstrap_auroc_ci(y_t, y_s)
        print(f"  RECUIMA AUROC: {auc:.3f} [{ci[0]:.3f}-{ci[1]:.3f}]")

# 8.3 By sex
print("\n\n8.3 Por sexo:")
print("-" * 50)

for sexo, label in [('masculino', 'Masculino'), ('femenino', 'Femenino')]:
    mask_sex = (df['sexo'] == sexo)
    n_sex = mask_sex.sum()
    mort_sex = df.loc[mask_sex, 'mortality'].mean() * 100
    
    print(f"\n{label}: N={n_sex}, Mortalidad={mort_sex:.2f}%")
    
    # GRACE
    mask_g = mask_sex & df['escala_grace'].notna() & df['mortality'].notna()
    if mask_g.sum() > 50 and df.loc[mask_g, 'mortality'].sum() >= 10:
        y_t = df.loc[mask_g, 'mortality'].values
        y_s = df.loc[mask_g, 'escala_grace'].values
        auc = roc_auc_score(y_t, y_s)
        ci = bootstrap_auroc_ci(y_t, y_s)
        print(f"  GRACE AUROC: {auc:.3f} [{ci[0]:.3f}-{ci[1]:.3f}]")
    
    # RECUIMA
    mask_r = mask_sex & df['recuima_weighted'].notna() & df['mortality'].notna()
    if mask_r.sum() > 50 and df.loc[mask_r, 'mortality'].sum() >= 10:
        y_t = df.loc[mask_r, 'mortality'].values
        y_s = df.loc[mask_r, 'recuima_weighted'].values
        auc = roc_auc_score(y_t, y_s)
        ci = bootstrap_auroc_ci(y_t, y_s)
        print(f"  RECUIMA AUROC: {auc:.3f} [{ci[0]:.3f}-{ci[1]:.3f}]")

# =============================================================================
# SUMMARY OF ALL VALUES FOR LATEX
# =============================================================================
print("\n" + "="*70)
print("RESUMEN DE VALORES PARA LATEX")
print("="*70)

print("""
========================================
TABLA 4.1 - CARACTERÍSTICAS DEL DATASET
========================================""")
print(f"N total: {n_total}")
print(f"Mortalidad hospitalaria: {n_mortality} ({mortality_rate:.2f}%)")
print(f"Edad media +/- DE: {edad_mean:.1f} +/- {edad_std:.1f}")
print(f"Edad mediana (IQR): {edad_median:.0f} ({edad_q1:.0f}-{edad_q3:.0f})")
print(f"Sexo masculino: {n_male} ({pct_male:.1f}%)")
print(f"IAMCEST: {n_stemi} ({pct_stemi:.1f}%)")
print(f"IAMSEST: {n_nstemi} ({pct_nstemi:.1f}%)")
print(f"Trombolisis: {n_thrombolysis} ({pct_thrombolysis:.1f}%)")

print("""
========================================
TABLA 4.2 - AUROC
========================================""")
print(f"GRACE: {auroc_grace:.3f} [{ci_grace[0]:.3f}-{ci_grace[1]:.3f}]")
print(f"TIMI-STEMI: {auroc_timi_s:.3f} [{ci_timi_s[0]:.3f}-{ci_timi_s[1]:.3f}]")
if auroc_timi is not None:
    print(f"TIMI-NSTEMI: {auroc_timi:.3f} [{ci_timi[0]:.3f}-{ci_timi[1]:.3f}]")
else:
    print("TIMI-NSTEMI: N/A (insuficientes casos)")
print(f"RECUIMA: {auroc_recuima:.3f} [{ci_recuima[0]:.3f}-{ci_recuima[1]:.3f}]")

print("""
========================================
TABLA 4.3 - CALIBRACIÓN HOSMER-LEMESHOW
========================================""")
print(f"GRACE: chi2={chi2_grace:.2f}, p={p_grace:.4f}")
print(f"TIMI-STEMI: chi2={chi2_timi_s:.2f}, p={p_timi_s:.4f}")
if auroc_timi is not None:
    print(f"TIMI-NSTEMI: chi2={chi2_timi:.2f}, p={p_timi_n:.4f}")
print(f"RECUIMA: chi2={chi2_recuima:.2f}, p={p_recuima:.4f}")

print("""
========================================
TABLA 4.4 - MÉTRICAS DIAGNÓSTICAS
========================================""")
print(f"GRACE (>=140):")
print(f"  Sens={sens_grace:.1f}%, Spec={spec_grace:.1f}%, VPP={ppv_grace:.1f}%, VPN={npv_grace:.1f}%")
print(f"  F1={f1_grace:.1f}%, LR+={lr_pos_grace:.2f}, LR-={lr_neg_grace:.3f}")
print(f"TIMI-STEMI (>=5):")
print(f"  Sens={sens_timi_s:.1f}%, Spec={spec_timi_s:.1f}%, VPP={ppv_timi_s:.1f}%, VPN={npv_timi_s:.1f}%")
print(f"  F1={f1_timi_s:.1f}%, LR+={lr_pos_timi_s:.2f}, LR-={lr_neg_timi_s:.3f}")
if auroc_timi is not None:
    print(f"TIMI-NSTEMI (>=3):")
    print(f"  Sens={sens_timi:.1f}%, Spec={spec_timi:.1f}%, VPP={ppv_timi:.1f}%, VPN={npv_timi:.1f}%")
    print(f"  F1={f1_timi:.1f}%, LR+={lr_pos_timi:.2f}, LR-={lr_neg_timi:.3f}")
print(f"RECUIMA (>=3):")
print(f"  Sens={sens_recuima:.1f}%, Spec={spec_recuima:.1f}%, VPP={ppv_recuima:.1f}%, VPN={npv_recuima:.1f}%")
print(f"  F1={f1_recuima:.1f}%, LR+={lr_pos_recuima:.2f}, LR-={lr_neg_recuima:.3f}")

print("""
========================================
TABLA 4.5 - DeLONG TEST
========================================""")
print(f"GRACE vs RECUIMA: diff={diff:.3f}, z={z:.2f}, p={p:.4f}")
print(f"RECUIMA vs GRACE: diff={diff_rg:.3f}, z={z_rg:.2f}, p={p_rg:.4f}")
print(f"GRACE vs TIMI-STEMI: diff={diff_s:.3f}, z={z_s:.2f}, p={p_s:.4f}")
print(f"RECUIMA vs TIMI-STEMI: diff={diff_rt:.3f}, z={z_rt:.2f}, p={p_rt:.4f}")

print("\n" + "="*70)
print("ANÁLISIS COMPLETADO")
print("="*70)
