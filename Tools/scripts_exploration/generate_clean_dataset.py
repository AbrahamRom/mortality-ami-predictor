"""Genera un dataset limpio correctamente alineado."""
import pandas as pd
import numpy as np
from datetime import datetime
from scipy.stats import pointbiserialr
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier

# Cargar dataset original
df = pd.read_excel('../DATA/recuima-020425-parsed.xlsx')
print('Dataset original:', df.shape)

# Variables recomendadas (10 top + target)
features = [
    'filtrado_glomerular', 'fraccion_eyeccion', 'edad', 'glicemia',
    'presion_arterial_diastolica', 'creatinina', 'presion_arterial_sistolica',
    'diabetes_mellitus', 'frecuencia_cardiaca', 'betabloqueadores'
]
target = 'mortality_inhospital'

# Seleccionar columnas
cols = features + [target]
df_clean = df[cols].copy()

# Eliminar filas sin target
df_clean = df_clean.dropna(subset=[target])
print('Filas con target:', len(df_clean))

# Convertir a numérico e imputar con mediana
for col in features:
    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    median = df_clean[col].median()
    n_missing = df_clean[col].isna().sum()
    if n_missing > 0:
        print(f'  {col}: imputando {n_missing} con mediana={median:.2f}')
    df_clean[col] = df_clean[col].fillna(median)

print('\nDataset final:', df_clean.shape)
print('Mortalidad:', df_clean[target].value_counts().to_dict())

# Verificar correlaciones
print('\n=== VERIFICACIÓN DE CORRELACIONES ===')
for col in features[:5]:
    corr, _ = pointbiserialr(df_clean[target], df_clean[col])
    print(f'  {col}: r={corr:+.4f}')

# Verificar características de fallecidos
dead = df_clean[df_clean[target] == 1]
alive = df_clean[df_clean[target] == 0]
print(f'\nFallecidos: N={len(dead)}, edad={dead["edad"].mean():.1f}, FE={dead["fraccion_eyeccion"].mean():.1f}')
print(f'Vivos: N={len(alive)}, edad={alive["edad"].mean():.1f}, FE={alive["fraccion_eyeccion"].mean():.1f}')

# Guardar
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
output_path = f'processed/cleaned_datasets/cleaned_dataset_{timestamp}.csv'
df_clean.to_csv(output_path, index=False)
print(f'\n✅ Guardado: {output_path}')

# Test de AUROC
print('\n=== TEST DE AUROC ===')
X = df_clean[features]
y = df_clean[target]
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
rf = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42, max_depth=10, min_samples_leaf=5, n_jobs=-1)
scores = cross_val_score(rf, X, y, cv=cv, scoring='roc_auc')
print(f'RF AUROC: {scores.mean():.4f} (+/- {scores.std():.4f})')
