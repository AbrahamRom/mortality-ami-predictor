# AutoML Guide - Aprendizaje Automático

## Introducción

El módulo **AutoML** del proyecto Mortality AMI Predictor proporciona capacidades de Machine Learning automatizado, permitiendo buscar automáticamente la mejor arquitectura de modelo y configuración de hiperparámetros para tu dataset.

### ¿Qué es AutoML?

AutoML (Automated Machine Learning) automatiza el proceso de:
- **Selección de algoritmos**: Evalúa múltiples algoritmos (Random Forest, XGBoost, LightGBM, etc.)
- **Ingeniería de features**: Aplica transformaciones automáticamente
- **Optimización de hiperparámetros**: Busca la mejor configuración
- **Ensemble**: Combina múltiples modelos para mejor rendimiento

### Backends Disponibles

| Backend | Plataforma | Ventajas | Limitaciones |
|---------|------------|----------|--------------|
| **auto-sklearn** | Linux, WSL | Más completo, meta-learning | Requiere Linux |
| **FLAML** | Windows, Linux, macOS | Más rápido, menor memoria | Menos algoritmos |

El sistema detecta automáticamente el backend disponible y usa FLAML como fallback en Windows.

---

## Instalación

### FLAML (Recomendado para Windows)

```bash
pip install flaml[automl]
```

### auto-sklearn (Linux/WSL)

```bash
# Requiere Linux o WSL
pip install auto-sklearn

# Dependencias adicionales
pip install pyrfr ConfigSpace smac
```

### Verificar instalación

```python
from src.automl import is_flaml_available, is_autosklearn_available

print(f"FLAML: {is_flaml_available()}")
print(f"auto-sklearn: {is_autosklearn_available()}")
```

---

## Uso Básico

### Desde el Dashboard

1. Ve a la página **🤖 AutoML** en el dashboard
2. Selecciona un preset (quick, balanced, high_performance)
3. Configura la métrica de optimización
4. Haz clic en "Iniciar AutoML"
5. Espera a que termine la búsqueda
6. Revisa el leaderboard y exporta el mejor modelo

### Desde Código

```python
from src.automl import AutoMLClassifier, FLAMLClassifier, AutoMLPreset
import pandas as pd

# Cargar datos
df = pd.read_csv("data.csv")
X = df.drop("target", axis=1)
y = df["target"]

# Opción 1: AutoMLClassifier (detecta backend automáticamente)
clf = AutoMLClassifier(
    preset=AutoMLPreset.BALANCED,
    metric="roc_auc",
    name="MiAutoML"
)
clf.fit(X, y)

# Opción 2: FLAML directamente (cross-platform)
clf = FLAMLClassifier(
    time_budget=3600,  # 1 hora
    metric="roc_auc",
    estimator_list=["lgbm", "xgboost", "rf"],
)
clf.fit(X, y)

# Predecir
probas = clf.predict_proba(X_test)
predictions = clf.predict(X_test)

# Ver mejor modelo
print(f"Mejor estimador: {clf.best_estimator_}")
print(f"Mejor score: {-clf.best_loss_:.4f}")
```

---

## Configuración

### Presets Disponibles

| Preset | Tiempo | Uso Recomendado |
|--------|--------|-----------------|
| `quick` | 5 min | Exploración rápida, debugging |
| `balanced` | 1 hora | Uso general, buen balance |
| `high_performance` | 4 horas | Producción, máximo rendimiento |
| `overnight` | 8 horas | Búsqueda exhaustiva |

### Configuración Personalizada

```python
from src.automl import AutoMLConfig, AutoMLPreset

config = AutoMLConfig(
    preset=AutoMLPreset.CUSTOM,
    time_left_for_this_task=1800,  # 30 minutos
    per_run_time_limit=180,        # 3 min por modelo
    memory_limit=8192,             # 8 GB
    ensemble_size=50,
    metric="roc_auc",
    include_estimators=["random_forest", "extra_trees", "gradient_boosting"],
    exclude_estimators=["mlp"],    # Excluir redes neuronales
    n_jobs=-1,
)

clf = AutoMLClassifier(config=config)
```

### Parámetros Principales

| Parámetro | Descripción | Default |
|-----------|-------------|---------|
| `time_left_for_this_task` | Tiempo total en segundos | 3600 |
| `per_run_time_limit` | Tiempo máximo por modelo | 360 |
| `memory_limit` | Límite de memoria en MB | 8192 |
| `ensemble_size` | Modelos en el ensemble | 50 |
| `metric` | Métrica a optimizar | roc_auc |
| `n_jobs` | Trabajos paralelos | -1 |

---

## Métricas Disponibles

### Clasificación

| Métrica | Descripción | Uso Recomendado |
|---------|-------------|-----------------|
| `roc_auc` | Área bajo curva ROC | Default, datos balanceados |
| `balanced_accuracy` | Accuracy balanceada | Datos desbalanceados |
| `f1` | F1-Score | Balance precision/recall |
| `precision` | Precisión | Minimizar falsos positivos |
| `recall` | Recall (Sensibilidad) | Minimizar falsos negativos |
| `log_loss` | Log Loss | Probabilidades calibradas |

### Regresión

| Métrica | Descripción |
|---------|-------------|
| `r2` | Coeficiente de determinación |
| `mse` | Error cuadrático medio |
| `mae` | Error absoluto medio |

---

## Estimadores Disponibles

### FLAML

| Estimador | Descripción |
|-----------|-------------|
| `lgbm` | LightGBM |
| `xgboost` | XGBoost |
| `xgb_limitdepth` | XGBoost con profundidad limitada |
| `catboost` | CatBoost |
| `rf` | Random Forest |
| `extra_tree` | Extra Trees |
| `kneighbor` | K-Nearest Neighbors |
| `lrl1` | Logistic Regression L1 |
| `lrl2` | Logistic Regression L2 |

### auto-sklearn

| Estimador | Descripción |
|-----------|-------------|
| `random_forest` | Random Forest |
| `extra_trees` | Extra Trees |
| `gradient_boosting` | Gradient Boosting |
| `adaboost` | AdaBoost |
| `mlp` | Multi-Layer Perceptron |
| `sgd` | SGD Classifier |
| `libsvm_svc` | Support Vector Machine |
| `k_nearest_neighbors` | K-NN |
| `decision_tree` | Decision Tree |
| `lda` | Linear Discriminant Analysis |
| `qda` | Quadratic Discriminant Analysis |

---

## Sugerencias Inteligentes

El sistema analiza tu dataset y sugiere técnicas que podrían mejorar el rendimiento:

```python
from src.automl import analyze_dataset, get_suggestions

# Analizar dataset
analysis = analyze_dataset(df, target_column="mortality")

print(f"Muestras: {analysis.n_samples}")
print(f"Desbalanceado: {analysis.is_imbalanced}")
print(f"% Datos faltantes: {analysis.missing_percentage:.1f}%")

# Obtener sugerencias
suggestions = get_suggestions(df, target_column="mortality")

for s in suggestions:
    print(f"[{s.priority.value}] {s.title}")
    print(f"  Razón: {s.reason}")
    print(f"  Módulo: {s.module_link}")
```

### Tipos de Sugerencias

- **Manejo de desbalance**: SMOTE, class_weight, XGBoost balanced
- **Datos faltantes**: Imputación simple, KNN, iterativa
- **Correlación**: Eliminar features, PCA
- **Selección de modelo**: Basado en tamaño y características
- **Evaluación**: Métrica apropiada, CV strategy

---

## Exportar Modelos

### Exportar Mejor Modelo

```python
from src.automl import export_best_model

path = export_best_model(
    automl_model=clf,
    output_dir="models/automl",
    model_name="best_mortality_model",
    include_metadata=True,
    training_data=df,
    target_column="mortality"
)

print(f"Modelo guardado en: {path}")
```

### Exportar Ensemble Completo

```python
from src.automl import export_ensemble

ensemble_path, model_paths = export_ensemble(
    automl_model=clf,
    output_dir="models/automl_ensemble",
    model_name="mortality_ensemble",
    max_models=10
)

print(f"Ensemble info: {ensemble_path}")
print(f"Modelos: {len(model_paths)}")
```

### Convertir a Modelo Standalone

```python
from src.automl import convert_to_standalone

standalone = convert_to_standalone(
    automl_model=clf,
    feature_names=X.columns.tolist(),
    classes=np.array([0, 1])
)

# Ahora funciona como cualquier modelo sklearn
probas = standalone.predict_proba(X_new)
```

---

## Troubleshooting

### Error de Memoria

```
MemoryError: Unable to allocate...
```

**Solución:**
```python
config = AutoMLConfig(
    memory_limit=4096,  # Reducir a 4GB
    per_run_time_limit=120,  # Reducir tiempo por modelo
    ensemble_size=20,  # Menos modelos en ensemble
)
```

### Tiempo Excesivo

**Problema:** El entrenamiento tarda demasiado.

**Solución:**
```python
clf = FLAMLClassifier(
    time_budget=300,  # 5 minutos
    estimator_list=["lgbm", "rf"],  # Solo 2 estimadores
    max_iter=50,  # Limitar iteraciones
)
```

### auto-sklearn no disponible en Windows

**Problema:** `ImportError: auto-sklearn requires Linux`

**Solución:**
1. Usar FLAML en su lugar (automático)
2. Instalar WSL (Windows Subsystem for Linux)
3. Usar Docker con Linux

```python
# El sistema usa FLAML automáticamente en Windows
clf = AutoMLClassifier(preset=AutoMLPreset.BALANCED)
# Backend: flaml (detectado automáticamente)
```

### SMAC Errors (auto-sklearn)

```
RuntimeError: SMAC...
```

**Solución:**
```bash
# Limpiar carpetas temporales
rm -rf /tmp/autosklearn_*

# O especificar carpeta limpia
config = AutoMLConfig(
    tmp_folder="/tmp/my_automl_run",
    delete_tmp_folder_after_terminate=True
)
```

---

## Integración con el Pipeline

### Con Model Training

El módulo AutoML se integra con la página de entrenamiento:

1. Selecciona "Incluir modelos AutoML" en el sidebar
2. Elige un preset (automl_quick, automl_balanced, etc.)
3. El modelo AutoML competirá con otros modelos

### Con Model Evaluation

Los modelos AutoML exportados funcionan con todas las herramientas de evaluación:

- Métricas (ROC-AUC, AUPRC, etc.)
- Curvas ROC y PR
- Matriz de confusión
- Comparación estadística

### Con Explainability

```python
# Los modelos AutoML soportan SHAP
import shap

# Obtener el mejor modelo del ensemble
best_model = clf.get_best_model()

# Explicar predicciones
explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test)
```

---

## Ejemplos Avanzados

### Búsqueda con Callback de Progreso

```python
def my_callback(message, progress):
    print(f"[{progress*100:.0f}%] {message}")

clf = AutoMLClassifier(
    preset=AutoMLPreset.BALANCED,
    progress_callback=my_callback
)
clf.fit(X, y)
```

### Comparar AutoML vs Modelo Manual

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# AutoML
clf_auto = FLAMLClassifier(time_budget=300)
clf_auto.fit(X_train, y_train)
score_auto = clf_auto.best_loss_ * -1

# Manual
clf_manual = RandomForestClassifier(n_estimators=100)
scores_manual = cross_val_score(clf_manual, X_train, y_train, cv=5, scoring='roc_auc')

print(f"AutoML: {score_auto:.4f}")
print(f"Manual: {scores_manual.mean():.4f} ± {scores_manual.std():.4f}")
```

### AutoML para Datasets Clínicos

```python
# Configuración optimizada para predicción clínica
clf = FLAMLClassifier(
    time_budget=1800,
    metric="roc_auc",
    # Solo modelos interpretables
    estimator_list=["lgbm", "rf", "extra_tree", "lrl2"],
    ensemble=True,
    name="Clinical-AutoML"
)

clf.fit(X_clinical, y_mortality)

# El modelo está listo para uso clínico
print(f"Mejor modelo: {clf.best_estimator_}")
print(f"AUC: {-clf.best_loss_:.4f}")
```

---

## Referencias

- [FLAML Documentation](https://microsoft.github.io/FLAML/)
- [auto-sklearn Documentation](https://automl.github.io/auto-sklearn/)
- [Auto-sklearn Paper](https://papers.nips.cc/paper/2015/hash/11d0e6287202fced83f79975ec59a3a6-Abstract.html)
- [FLAML Paper](https://www.microsoft.com/en-us/research/publication/flaml-a-fast-and-lightweight-automl-library/)
