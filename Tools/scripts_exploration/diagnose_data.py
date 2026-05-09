"""Diagnóstico del problema de correlaciones destruidas."""
import pandas as pd
import numpy as np

df_orig = pd.read_csv('../DATA/recuima-020425.csv', encoding='latin-1')
df_clean = pd.read_csv('processed/cleaned_datasets/cleaned_dataset_20260108_102546.csv')

print('=== DISTRIBUCIÓN DE MORTALIDAD ===')
print('\nOriginal:')
print(df_orig['mortality_inhospital'].value_counts())
print(f"Tasa mortalidad: {df_orig['mortality_inhospital'].mean()*100:.2f}%")

print('\nLimpio:')
print(df_clean['mortality_inhospital'].value_counts())
print(f"Tasa mortalidad: {df_clean['mortality_inhospital'].mean()*100:.2f}%")

# Verificar si hay diferencia en las edades de los que murieron
print('\n=== EDAD DE LOS FALLECIDOS ===')
orig_dead = df_orig[df_orig['mortality_inhospital'] == 1]['edad']
clean_dead = df_clean[df_clean['mortality_inhospital'] == 1]['edad']

print(f"\nOriginal - Edad fallecidos:")
print(f"  Media: {orig_dead.mean():.1f}")
print(f"  Mediana: {orig_dead.median():.1f}")
print(f"  Min-Max: {orig_dead.min()}-{orig_dead.max()}")

print(f"\nLimpio - Edad fallecidos:")
print(f"  Media: {clean_dead.mean():.1f}")
print(f"  Mediana: {clean_dead.median():.1f}")
print(f"  Min-Max: {clean_dead.min()}-{clean_dead.max()}")

# Verificar distribución por rangos
print('\n=== MORTALIDAD POR RANGO DE EDAD ===')
bins = [0, 50, 60, 70, 80, 100]
labels = ['<50', '50-60', '60-70', '70-80', '>80']

df_orig['edad_cat'] = pd.cut(df_orig['edad'], bins=bins, labels=labels)
df_clean['edad_cat'] = pd.cut(df_clean['edad'], bins=bins, labels=labels)

print("\n       Original    Limpio")
for cat in labels:
    orig_rate = df_orig[df_orig['edad_cat'] == cat]['mortality_inhospital'].mean() * 100
    clean_rate = df_clean[df_clean['edad_cat'] == cat]['mortality_inhospital'].mean() * 100
    print(f"{cat:>6}: {orig_rate:>6.1f}%    {clean_rate:>6.1f}%")

# Verificar si se duplicaron filas de sobrevivientes
print('\n=== VERIFICANDO DUPLICADOS ===')
print(f"Filas únicas original: {len(df_orig.drop_duplicates())}")
print(f"Filas únicas limpio: {len(df_clean.drop_duplicates())}")

# Verificar IMC
print('\n=== VERIFICANDO IMC (puede haber imputación con valores sintéticos) ===')
if 'imc' in df_orig.columns and 'imc' in df_clean.columns:
    print(f"Original IMC: mean={df_orig['imc'].mean():.2f}, std={df_orig['imc'].std():.2f}")
    print(f"Limpio IMC: mean={df_clean['imc'].mean():.2f}, std={df_clean['imc'].std():.2f}")
