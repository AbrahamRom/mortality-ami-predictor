"""
Sistema de Exploración de Variables para Maximizar AUROC
=========================================================

Este script explora automáticamente el mejor conjunto de variables para 
predecir mortalidad intrahospitalaria, evitando data leakage.

Autor: Sistema de exploración automática
Fecha: 2026-01-08
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, mutual_info_classif, RFE
from scipy.stats import pointbiserialr
import warnings
import itertools
from typing import List, Dict, Tuple, Set
import json
from datetime import datetime

warnings.filterwarnings('ignore')

# ==============================================================================
# CONFIGURACIÓN
# ==============================================================================

# Variables que NUNCA deben usarse (data leakage - información post-evento)
LEAKAGE_VARIABLES = {
    # Tratamientos al ALTA (solo pacientes que sobrevivieron los reciben)
    'asa.1', 'estatinas.1', 'clopidogrel.1', 'ieca.1', 'betabloqueadores.1',
    'dieta', 'consejeria', 'rehabilitacion', 'rehabilitacion.1',
    'consejeria_antitabaquica', 'furosemida.1', 'nitratos.1', 'otros_diureticos.1',
    'anticalcico', 'anticoagulantes.1',
    
    # Eventos DURANTE la estancia (pueden ser consecuencia, no causa)
    'vam',  # Ventilación mecánica
    'avc',  # Accidente vascular cerebral durante estancia
    'mpt',  # Marcapasos temporal
    'aminas',  # Soporte con aminas (puede ser por el desenlace)
    
    # COMPLICACIONES durante estancia (son OUTCOMES, no predictores)
    'comp_pcr',  # Paro cardiorrespiratorio
    'comp_shock', 'shock', 'shock_cardiogenico',  # Shock
    'comp_fv', 'comp_tv', 'comp_fv_tv',  # Arritmias ventriculares
    'comp_mecanicas',  # Complicaciones mecánicas
    'comp_bav_alto_grado', 'comp_bav',  # Bloqueos
    'comp_ic',  # Insuficiencia cardíaca
    'comp_otras_arritmias',  # Otras arritmias
    'complicaciones',  # Score de complicaciones
    
    # Estadías (consecuencia del desenlace)
    'estadia_uci', 'estadia_hosp', 'estadia_intrahospitalaria',
    
    # IDs y fechas
    'numero', 'anno', 'unidad', 'fecha_ingreso', 'fecha_egreso',
    
    # EXCLUIR escala_grace - el objetivo es NO depender de ella
    'escala_grace', 'GRACE', 'grace_score',
}

# Variables disponibles AL INGRESO (válidas para predicción)
VALID_ADMISSION_VARIABLES = {
    # Demográficos
    'edad', 'sexo', 'peso', 'talla', 'imc', 'color_piel',
    
    # Antecedentes
    'diabetes_mellitus', 'hipertension_arterial', 'tabaquismo',
    'hiperlipoproteinemia', 'infarto_miocardio_agudo',
    'enfermedad_arterias_coronarias', 'enfermedad_venosa_periferica',
    'enfermedad_cerebro_vascular', 'insuficiencia_cardiaca',
    'fibrilacion_auricular', 'cardiopatia_isquemica',
    
    # Signos vitales al ingreso
    'presion_arterial_sistolica', 'presion_arterial_diastolica',
    'frecuencia_cardiaca',
    
    # Clasificación clínica al ingreso (SIN escala_grace - objetivo es no depender de ella)
    'indice_killip',
    
    # ECG al ingreso
    'scacest', 'supradesnivel', 'infradesnivel', 'depresion_st',
    'localizacion_ima',  # Localización del infarto
    
    # Laboratorio al ingreso
    'fraccion_eyeccion', 'creatinina', 'filtrado_glomerular',
    'glicemia', 'colesterol', 'trigliceridos', 'hb', 'ck', 'ckmb',
    'troponina', 'ldl', 'hdl',
    
    # Tratamiento inicial (decisiones al ingreso, no al alta)
    'estreptoquinasa_recombinante', 'tiempo_puerta_aguja',
    'reperfusion', 'coronariografia', 'tiempo_isquemia',
    
    # Tratamientos al ingreso (no .1)
    'asa', 'betabloqueadores', 'clopidogrel', 'heparina', 'estatinas',
    'furosemida', 'nitratos', 'anticoagulantes', 'ieca',
}

# ==============================================================================
# FUNCIONES DE UTILIDAD
# ==============================================================================

def load_data(path: str) -> pd.DataFrame:
    """Carga el dataset."""
    if path.endswith('.xlsx') or path.endswith('.xls'):
        return pd.read_excel(path)
    else:
        return pd.read_csv(path, encoding='latin-1')


def identify_valid_features(df: pd.DataFrame, target: str) -> List[str]:
    """Identifica variables válidas para predicción (sin data leakage)."""
    valid_features = []
    
    for col in df.columns:
        # Excluir target
        if col == target:
            continue
        
        # Excluir leakage
        if col.lower() in {v.lower() for v in LEAKAGE_VARIABLES}:
            continue
        
        # Excluir variables con .1 (tratamientos al alta)
        if col.endswith('.1'):
            continue
        
        # Verificar si es numérica o convertible
        if df[col].dtype in ['int64', 'float64']:
            valid_features.append(col)
        else:
            # Intentar convertir
            try:
                numeric = pd.to_numeric(df[col], errors='coerce')
                if numeric.notna().sum() > len(df) * 0.5:  # Al menos 50% válido
                    valid_features.append(col)
            except:
                pass
    
    return valid_features


def compute_feature_importance(df: pd.DataFrame, features: List[str], target: str) -> pd.DataFrame:
    """Calcula importancia de cada variable usando múltiples métricas."""
    results = []
    
    y = df[target].values
    
    for feat in features:
        try:
            x = pd.to_numeric(df[feat], errors='coerce').values
            valid_mask = ~np.isnan(x) & ~np.isnan(y)
            
            if valid_mask.sum() < 100:
                continue
            
            x_valid = x[valid_mask]
            y_valid = y[valid_mask]
            
            # Correlación punto-biserial
            corr, pval = pointbiserialr(y_valid, x_valid)
            
            # Missing rate
            missing_rate = 1 - (valid_mask.sum() / len(df))
            
            results.append({
                'feature': feat,
                'correlation': abs(corr),
                'corr_signed': corr,
                'pvalue': pval,
                'missing_rate': missing_rate,
                'n_valid': valid_mask.sum(),
            })
        except Exception as e:
            continue
    
    return pd.DataFrame(results).sort_values('correlation', ascending=False)


def prepare_dataset(df: pd.DataFrame, features: List[str], target: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Prepara X e y para entrenamiento."""
    # Convertir a numérico
    X = df[features].copy()
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')
    
    y = df[target].copy()
    
    # Eliminar filas con target missing
    valid_idx = ~y.isna()
    X = X[valid_idx]
    y = y[valid_idx]
    
    # Imputar con mediana (simple, no destruye relaciones)
    for col in X.columns:
        median = X[col].median()
        if pd.isna(median):
            median = 0
        X[col] = X[col].fillna(median)
    
    return X, y


def evaluate_feature_set(X: pd.DataFrame, y: pd.Series, cv: StratifiedKFold) -> Dict:
    """Evalúa un conjunto de features con múltiples modelos."""
    results = {}
    
    models = {
        'LogReg': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42))
        ]),
        'RF': RandomForestClassifier(
            n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1,
            max_depth=10, min_samples_leaf=5
        ),
        'GBM': GradientBoostingClassifier(
            n_estimators=100, random_state=42, max_depth=5
        ),
    }
    
    for name, model in models.items():
        try:
            scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)
            results[name] = {
                'mean': scores.mean(),
                'std': scores.std(),
                'scores': scores.tolist()
            }
        except Exception as e:
            results[name] = {'mean': 0, 'std': 0, 'error': str(e)}
    
    # Best overall
    best_model = max(results.keys(), key=lambda k: results[k].get('mean', 0))
    results['best_model'] = best_model
    results['best_auroc'] = results[best_model]['mean']
    
    return results


def forward_selection(X: pd.DataFrame, y: pd.Series, cv: StratifiedKFold, 
                      max_features: int = 20) -> Tuple[List[str], List[Dict]]:
    """Selección forward de variables."""
    selected = []
    remaining = list(X.columns)
    history = []
    
    print("\n=== FORWARD SELECTION ===")
    
    best_overall_score = 0
    
    while remaining and len(selected) < max_features:
        best_score = 0
        best_feature = None
        
        for feat in remaining:
            test_features = selected + [feat]
            X_test = X[test_features]
            
            # Quick evaluation with RF only for speed
            try:
                rf = RandomForestClassifier(n_estimators=50, class_weight='balanced', 
                                           random_state=42, n_jobs=-1, max_depth=8)
                scores = cross_val_score(rf, X_test, y, cv=cv, scoring='roc_auc', n_jobs=-1)
                score = scores.mean()
                
                if score > best_score:
                    best_score = score
                    best_feature = feat
            except:
                continue
        
        if best_feature is None:
            break
        
        # Only add if it improves
        if best_score > best_overall_score:
            selected.append(best_feature)
            remaining.remove(best_feature)
            best_overall_score = best_score
            
            history.append({
                'step': len(selected),
                'feature_added': best_feature,
                'auroc': best_score,
                'features': selected.copy()
            })
            
            print(f"  Step {len(selected)}: +{best_feature} -> AUROC = {best_score:.4f}")
        else:
            print(f"  No improvement, stopping at {len(selected)} features")
            break
    
    return selected, history


def explore_feature_combinations(X: pd.DataFrame, y: pd.Series, 
                                  top_features: List[str], cv: StratifiedKFold,
                                  min_size: int = 5, max_size: int = 15) -> List[Dict]:
    """Explora combinaciones de las mejores features."""
    results = []
    
    print(f"\n=== EXPLORANDO COMBINACIONES DE TOP {len(top_features)} FEATURES ===")
    
    # Test different sizes
    for size in range(min_size, min(max_size + 1, len(top_features) + 1)):
        X_test = X[top_features[:size]]
        eval_result = evaluate_feature_set(X_test, y, cv)
        
        results.append({
            'n_features': size,
            'features': top_features[:size],
            'auroc': eval_result['best_auroc'],
            'best_model': eval_result['best_model'],
            'details': eval_result
        })
        
        print(f"  {size} features: AUROC = {eval_result['best_auroc']:.4f} ({eval_result['best_model']})")
    
    return results


# ==============================================================================
# MAIN EXPLORATION
# ==============================================================================

def main():
    """Ejecuta la exploración completa."""
    print("=" * 70)
    print("SISTEMA DE EXPLORACIÓN DE VARIABLES PARA PREDICCIÓN DE MORTALIDAD")
    print("=" * 70)
    
    # Cargar datos
    data_path = Path(__file__).parent.parent / "DATA" / "recuima-020425-parsed.xlsx"
    print(f"\n📂 Cargando datos desde: {data_path}")
    
    df = load_data(str(data_path))
    target = 'mortality_inhospital'
    
    print(f"   Shape: {df.shape}")
    print(f"   Target: {target}")
    print(f"   Clase 1 (muerte): {df[target].sum()} ({100*df[target].mean():.2f}%)")
    
    # Identificar variables válidas
    print("\n🔍 Identificando variables válidas (sin data leakage)...")
    valid_features = identify_valid_features(df, target)
    print(f"   Variables válidas encontradas: {len(valid_features)}")
    
    # Calcular importancia
    print("\n📊 Calculando importancia de variables...")
    importance_df = compute_feature_importance(df, valid_features, target)
    
    print("\n   TOP 20 VARIABLES POR CORRELACIÓN:")
    print("   " + "-" * 60)
    for i, row in importance_df.head(20).iterrows():
        sig = "***" if row['pvalue'] < 0.001 else "**" if row['pvalue'] < 0.01 else "*" if row['pvalue'] < 0.05 else ""
        print(f"   {row['feature']:35} r={row['corr_signed']:+.4f} {sig} (n={row['n_valid']})")
    
    # Filtrar por significancia y missing rate
    significant_features = importance_df[
        (importance_df['pvalue'] < 0.05) & 
        (importance_df['missing_rate'] < 0.3)  # Max 30% missing
    ]['feature'].tolist()
    
    print(f"\n   Variables significativas (p<0.05, <30% missing): {len(significant_features)}")
    
    # Preparar dataset
    print("\n🔧 Preparando dataset para modelado...")
    X, y = prepare_dataset(df, significant_features, target)
    print(f"   X shape: {X.shape}")
    print(f"   y distribution: {y.value_counts().to_dict()}")
    
    # Cross-validation setup
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Forward selection
    selected_features, selection_history = forward_selection(X, y, cv, max_features=25)
    
    # Evaluate final set with all models
    print("\n🎯 EVALUACIÓN FINAL DEL MEJOR CONJUNTO:")
    print("-" * 50)
    X_final = X[selected_features]
    final_results = evaluate_feature_set(X_final, y, cv)
    
    for model, res in final_results.items():
        if model in ['best_model', 'best_auroc']:
            continue
        if 'mean' in res:
            print(f"   {model}: AUROC = {res['mean']:.4f} (+/- {res['std']:.4f})")
    
    print(f"\n   🏆 MEJOR MODELO: {final_results['best_model']}")
    print(f"   🏆 MEJOR AUROC: {final_results['best_auroc']:.4f}")
    
    # Explore different sizes
    combination_results = explore_feature_combinations(
        X, y, selected_features, cv, min_size=5, max_size=min(20, len(selected_features))
    )
    
    # Find optimal
    best_combo = max(combination_results, key=lambda x: x['auroc'])
    
    print("\n" + "=" * 70)
    print("RESULTADOS FINALES")
    print("=" * 70)
    
    print(f"\n📌 MEJOR CONFIGURACIÓN ENCONTRADA:")
    print(f"   Número de variables: {best_combo['n_features']}")
    print(f"   AUROC: {best_combo['auroc']:.4f}")
    print(f"   Modelo: {best_combo['best_model']}")
    print(f"\n   Variables seleccionadas:")
    for i, feat in enumerate(best_combo['features'], 1):
        # Get correlation
        feat_info = importance_df[importance_df['feature'] == feat]
        if not feat_info.empty:
            corr = feat_info['corr_signed'].values[0]
            print(f"      {i:2}. {feat:35} (r={corr:+.4f})")
        else:
            print(f"      {i:2}. {feat}")
    
    # Save results
    results_dir = Path(__file__).parent / "exploration_results"
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    results_summary = {
        'timestamp': timestamp,
        'dataset': str(data_path),
        'target': target,
        'n_samples': len(df),
        'mortality_rate': float(df[target].mean()),
        'best_auroc': float(best_combo['auroc']),
        'best_model': best_combo['best_model'],
        'best_n_features': best_combo['n_features'],
        'best_features': best_combo['features'],
        'all_significant_features': significant_features,
        'importance_ranking': importance_df.head(30).to_dict('records'),
        'selection_history': selection_history,
        'combination_results': combination_results,
    }
    
    results_path = results_dir / f"exploration_results_{timestamp}.json"
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Resultados guardados en: {results_path}")
    
    # Generate recommended config
    print("\n" + "=" * 70)
    print("RECOMENDACIÓN PARA LA APP")
    print("=" * 70)
    print("""
Para usar estos resultados en la aplicación:

1. En la página de Data Cleaning:
   - Carga el dataset original (recuima-020425-parsed.xlsx)
   - En "Selección de Variables", selecciona SOLO las variables recomendadas
   - NO apliques imputación (ya está en 'none' por defecto)
   - NO apliques tratamiento de outliers (ya está en 'none' por defecto)

2. En la página de Training:
   - Usa class_weight='balanced' en lugar de SMOTE
   - O usa SMOTE con sampling_strategy='auto'

3. Variables recomendadas para copiar:
""")
    print(f"   {best_combo['features']}")
    
    return results_summary


if __name__ == "__main__":
    results = main()
