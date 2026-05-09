"""Script para analizar el problema de AUC bajo."""
import pandas as pd
import numpy as np
from scipy.stats import pointbiserialr
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import warnings
warnings.filterwarnings('ignore')

# Cargar datasets
df_orig = pd.read_csv('../DATA/recuima-020425.csv', encoding='latin-1')
df_clean = pd.read_csv('processed/cleaned_datasets/cleaned_dataset_20260108_102546.csv')

print('=== COMPARACION DE DATASETS ===')
print(f'Original: {df_orig.shape}')
print(f'Limpio: {df_clean.shape}')

print('\n=== COLUMNAS EN LIMPIO ===')
print(list(df_clean.columns))

print('\n=== VERIFICANDO VARIABLES IMPORTANTES ===')
important = ['escala_grace', 'fraccion_eyeccion', 'edad', 'creatinina', 'indice_killip']
for var in important:
    in_orig = var in df_orig.columns
    in_clean = var in df_clean.columns
    print(f'{var}: original={in_orig}, limpio={in_clean}')

print('\n=== CORRELACION EN LIMPIO vs ORIGINAL ===')
target = 'mortality_inhospital'

for var in ['fraccion_eyeccion', 'edad', 'creatinina', 'indice_killip']:
    if var in df_clean.columns and var in df_orig.columns:
        valid_clean = df_clean[[var, target]].dropna()
        valid_orig = df_orig[[var, target]].dropna()
        
        # Convertir a numérico
        valid_clean[var] = pd.to_numeric(valid_clean[var], errors='coerce')
        valid_orig[var] = pd.to_numeric(valid_orig[var], errors='coerce')
        valid_clean = valid_clean.dropna()
        valid_orig = valid_orig.dropna()
        
        if len(valid_clean) > 10 and len(valid_orig) > 10:
            corr_clean, _ = pointbiserialr(valid_clean[target], valid_clean[var])
            corr_orig, _ = pointbiserialr(valid_orig[target], valid_orig[var])
            print(f'{var}: orig_corr={corr_orig:.4f}, clean_corr={corr_clean:.4f}')

print('\n=== ESCALA GRACE FALTA - ESTO ES CRÍTICO ===')
if 'escala_grace' not in df_clean.columns:
    print('>>> escala_grace NO ESTA en el dataset limpio!')
    print('>>> Esta variable tiene correlacion r=0.275 con mortalidad')
    print('>>> Es la variable más predictiva (después de tratamientos al alta que son data leakage)')

# Ahora intentar con el dataset original pero sin data leakage
print('\n\n' + '='*60)
print('=== ENTRENAMIENTO CON DATASET ORIGINAL (SIN DATA LEAKAGE) ===')
print('='*60)

# Excluir variables de tratamiento al alta (data leakage)
leakage_cols = [c for c in df_orig.columns if c.endswith('.1') or c in [
    'dieta', 'consejeria', 'rehabilitacion', 'consejeria_antitabaquica',
    'estadia_uci', 'estadia_hosp', 'vam', 'avc', 'mpt', 'aminas'  # Estos son durante estancia
]]
print(f'\nExcluyendo {len(leakage_cols)} columnas de data leakage')

# Variables válidas para predicción al ingreso
valid_features = [
    'edad', 'sexo', 'peso', 'talla', 'imc',
    'diabetes_mellitus', 'hipertension_arterial', 'tabaquismo',
    'hiperlipoproteinemia', 'infarto_miocardio_agudo',
    'enfermedad_arterias_coronarias',
    'presion_arterial_sistolica', 'presion_arterial_diastolica',
    'frecuencia_cardiaca', 'indice_killip', 'escala_grace',
    'fraccion_eyeccion', 'creatinina', 'filtrado_glomerular',
    'glicemia', 'colesterol', 'trigliceridos', 'hb', 'ck', 'ckmb',
    'scacest', 'supradesnivel', 'infradesnivel', 'depresion_st'
]

# Filtrar solo las que existen
available = [c for c in valid_features if c in df_orig.columns]
print(f'\nVariables válidas disponibles: {len(available)}')
print(available)

# Crear dataset para modelo
df_model = df_orig[available + [target]].copy()
df_model = df_model.dropna(subset=[target])

# Convertir todo a numérico
for col in available:
    df_model[col] = pd.to_numeric(df_model[col], errors='coerce')

# Eliminar filas con NaN en el target
df_model = df_model.dropna(subset=[target])

# Imputar missing con mediana
for col in available:
    median_val = df_model[col].median()
    if pd.isna(median_val):
        median_val = 0
    df_model[col] = df_model[col].fillna(median_val)
    
# Verificar que no hay NaN
print(f'NaN restantes: {df_model.isna().sum().sum()}')

X = df_model[available]
y = df_model[target]

print(f'\nDataset final: {X.shape}')
print(f'Clase positiva: {y.sum()} ({100*y.mean():.2f}%)')

# Entrenar modelos
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print('\n=== RESULTADOS CON VARIABLES AL INGRESO + ESCALA GRACE ===')

models = {
    'LogReg': Pipeline([
        ('scaler', StandardScaler()), 
        ('clf', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42))
    ]),
    'RF_balanced': RandomForestClassifier(
        n_estimators=200, class_weight='balanced', random_state=42, n_jobs=-1,
        max_depth=10, min_samples_leaf=5
    ),
    'GBM': GradientBoostingClassifier(
        n_estimators=200, random_state=42, max_depth=5, min_samples_leaf=10
    ),
}

for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)
    print(f'{name}: AUC-ROC = {scores.mean():.4f} (+/- {scores.std():.4f})')

# Con SMOTE
print('\n=== CON SMOTE ===')
for name, base_model in models.items():
    smote_pipe = ImbPipeline([
        ('scaler', StandardScaler()),
        ('smote', SMOTE(random_state=42)),
        ('clf', base_model)
    ])
    try:
        scores = cross_val_score(smote_pipe, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)
        print(f'{name}+SMOTE: AUC-ROC = {scores.mean():.4f} (+/- {scores.std():.4f})')
    except:
        pass

print('\n=== FEATURE IMPORTANCE (RF) ===')
rf = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
rf.fit(X, y)
importance = pd.DataFrame({
    'feature': available,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)
print(importance.head(15).to_string(index=False))
