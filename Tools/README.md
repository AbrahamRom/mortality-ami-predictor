# Mortality AMI Predictor (Tools)

**Version 2.0 - Fully Modularized Architecture**

End-to-end ML project for predicting in-hospital mortality and ventricular arrhythmias in AMI patients. Features a professional, modular Python codebase, Jupyter notebooks, automated tests, and a Streamlit dashboard.

## 🎉 What's New in v2.0

- ✅ **Complete Modularization**: Transformed from 9 monolithic files into 10 specialized modules with 38+ focused files
- ✅ **Professional Design**: Factory, Strategy, Builder, Singleton, and Adapter patterns
- ✅ **Type Safety**: 100% type hints across all modules
- ✅ **Extensibility**: Easy to add new models, strategies, and features via registry pattern
- ✅ **Better Organization**: No file > 500 lines, average ~100 lines per file
- ✅ **Comprehensive Documentation**: 4+ guides including architecture, migration, and structure docs
- ✅ **New Organized Structure**: All processed data, models, and plots centralized in `processed/` directory
- ✅ **Custom Models System**: 🆕 Create, train, and deploy custom sklearn-compatible models with full dashboard integration

### 📚 Documentation

- **[Quick Start Guide](src/README.md)** - Get started in 5 minutes
- **[Complete Modularization Summary](src/COMPLETE_MODULARIZATION_SUMMARY.md)** - Full overview of all modules
- **[Migration Guide](src/MIGRATION_GUIDE.md)** - How to migrate from v1.0 to v2.0
- **[Project Structure](src/PROJECT_STRUCTURE.md)** - Visual diagrams and architecture
- **[Modularization Guide](src/MODULARIZATION_GUIDE.md)** - Deep dive into design decisions
- **[Custom Models Guide](docs/CUSTOM_MODELS_GUIDE.md)** - 🆕 Create and integrate custom ML models

---

## 📁 New Directory Structure (v2.1)

All processed outputs are now organized in the `processed/` directory with automatic timestamp management:

```
Tools/processed/
├── cleaned_datasets/              # Cleaned datasets
│   └── cleaned_dataset_YYYYMMDD_HHMMSS.csv
├── plots/                         # All visualizations organized by type
│   ├── eda/                       # Exploratory data analysis plots
│   ├── evaluation/                # Model evaluation plots
│   ├── explainability/            # SHAP, PDP, permutation plots
│   └── training/                  # Learning curves, CV results
├── models/                        # Trained models by type
│   ├── dtree/                     # Decision tree models
│   │   └── model_dtree_YYYYMMDD_HHMMSS.joblib
│   ├── knn/                       # KNN models
│   ├── xgb/                       # XGBoost models
│   ├── logistic/                  # Logistic regression models
│   ├── random_forest/             # Random forest models
│   ├── neural_network/            # Neural network models
│   └── testsets/                  # Test and train sets
│       ├── testset_dtree_YYYYMMDD_HHMMSS.parquet
│       └── trainset_dtree_YYYYMMDD_HHMMSS.parquet
├── variable_metadata.json         # Variable metadata
└── preprocessing_config.json      # Preprocessing configuration
```

### Key Features:
- **Timestamp Management**: All files saved with `YYYYMMDD_HHMMSS` format for version tracking
- **Automatic Cleanup**: Only latest model per type is kept (old versions auto-deleted)
- **Plot Organization**: Plots organized by section (eda, evaluation, explainability, training)
- **Model Organization**: Models organized by type in separate directories
- **Overwrite Logic**: Plots of same type/model are overwritten to avoid accumulation



## Quick Start

**Important**: The only required input is `DATASET_PATH` (path to your CSV/Parquet dataset). No other paths should be hardcoded.

## Instalación rápida (Windows PowerShell)

1) Crear entorno

```powershell
# Opción Conda
conda env create -f environment.yml; conda activate mortality-ami-env

# Opción venv (Python)
python -m venv .venv; .\.venv\Scripts\Activate.ps1; pip install -r requirements.txt
```

Nota sobre PyTorch/torchvision (opcional):
- La red neuronal en `src/models.py` es opcional y está desactivada por defecto. Si no la vas a usar, no necesitas instalar `torch`/`torchvision`.
- En Windows, instala PyTorch desde el índice oficial para evitar errores como: `ERROR: No matching distribution found for torchvision`.

Instalación PyTorch en Windows (elige una):

```powershell
# CPU-only
pip install --extra-index-url https://download.pytorch.org/whl/cpu torch torchvision

# CUDA 12.1 (si cuentas con drivers/CUDA compatibles)
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision

# Extras opcionales
pip install pytorch-lightning captum
```

Para activar el modelo de red neuronal en los experimentos, define:

```powershell
$env:ENABLE_TORCH_MODEL = "1"
```

2) Definir DATASET_PATH (ejemplo)

```powershell
$env:DATASET_PATH = "C:\\data\\ami_dataset.csv"
```

3) Ejecutar experimentos (entrenamiento + evaluación)

```powershell
# Script PowerShell
.\u200brun_experiments.ps1 -Data $env:DATASET_PATH

# (Opcional en Linux/macOS)
# export DATASET_PATH="/ruta/ami_dataset.csv"
# bash run_experiments.sh
```

4) Lanzar el dashboard en Streamlit

```powershell
streamlit run streamlit_app.py -- --data $env:DATASET_PATH
```

5) Ejecutar pruebas unitarias

```powershell
pytest -q
```

## Estructura del proyecto

- `src/` código modular:
	- `config.py`: configuración global (incluye `DATASET_PATH`), semillas y nombres de columnas objetivo.
	- `data_load/`: utilidades de carga y gestión de archivos
		- `loaders.py`: carga de datos (CSV/Parquet/Feather)
		- `path_utils.py`: utilidades para gestión de rutas y archivos con timestamps
		- `splitters.py`: división de datos (estratificada o temporal)
	- `cleaning/`: módulos de limpieza de datos
		- `cleaner.py`: orquestador principal de limpieza
		- `imputation.py`, `encoding.py`, `discretization.py`, `outliers.py`: estrategias específicas
	- `eda/`: análisis exploratorio de datos con visualizaciones interactivas
	- `preprocessing/`: pipelines de preprocesamiento (imputación, escalado, selección)
	- `features.py`: utilidades para seleccionar columnas seguras (excluye identificadores/objetivos).
	- `models/`: definiciones de modelos (KNN, Logistic, Tree, XGBoost, LightGBM, Neural Network)
	- `training/`: entrenamiento con validación cruzada anidada, RandomizedSearchCV, SMOTE, logging MLflow
	- `evaluation/`: métricas (AUROC, AUPRC, etc.), curvas de calibración, Decision Curve Analysis
	- `explainability/`: SHAP, importancia por permutación, PDP
	- `prediction/`: recarga modelos y genera predicciones por lotes
	- `scoring/`: cálculo de scores clínicos (GRACE, TIMI)
- `dashboard/`: aplicación Streamlit multi-página
	- `pages/`: páginas individuales para limpieza, entrenamiento, evaluación, explicabilidad
	- `app/`: configuración, estado y utilidades UI
- `notebooks/`: análisis interactivos en Jupyter
	- `eda.ipynb`: EDA completo con visualizaciones
	- `modeling.ipynb`: entrenamiento rápido de modelos base
	- `explainability.ipynb`: análisis SHAP global
- `processed/`: **NUEVO - todos los outputs organizados**
	- `cleaned_datasets/`: datasets limpios con timestamps
	- `plots/`: visualizaciones organizadas por sección
	- `models/`: modelos entrenados organizados por tipo
	- `variable_metadata.json`: metadatos de variables
- `tests/`: pruebas con pytest (carga/preproceso, entrenamiento, guardado/carga de modelo)
- `run_experiments.ps1` y `run_experiments.sh`: scripts para ejecutar experimentos
- `.github/workflows/ci.yml`: integración continua

**Nota**: Los directorios `reports/`, `dashboard/DATA/`, `dashboard/models/` y `Tools/models/` son legacy y serán eliminados. Usa `processed/` para todos los nuevos outputs.

## Cómo entrenar y evaluar

Entrenamiento (validación cruzada anidada + búsqueda aleatoria de hiperparámetros):

```powershell
python -m src.train --data $env:DATASET_PATH --task mortality
python -m src.train --data $env:DATASET_PATH --task arrhythmia
# Si tienes una columna continua (p.ej. estancia):
python -m src.train --data $env:DATASET_PATH --task regression
```

- Usa `--quick` para una ejecución rápida de depuración.
- Los mejores modelos se guardan en `models/` y el conjunto de test en `models/testset_*.parquet`.

Evaluación (hold-out):

```powershell
python -m src.evaluate --data $env:DATASET_PATH --task mortality
```

Salidas: `reports/final_metrics_mortality.csv`, `reports/final_evaluation.pdf` y figuras en `reports/figures/`.

---

## 🔧 Custom Models System

Create and integrate your own sklearn-compatible models! The system provides full support for:
- Custom model architectures (ensembles, neural networks, etc.)
- Seamless training integration with cross-validation
- Evaluation with standard metrics (ROC-AUC, AUPRC, etc.)
- SHAP explanations and feature importance
- Dashboard UI for model management

### Quick Example

```python
from src.models.custom_base import BaseCustomClassifier
import numpy as np

class MyCustomClassifier(BaseCustomClassifier):
    def __init__(self, n_layers=3, learning_rate=0.01):
        super().__init__(name="MyCustomClassifier")
        self.n_layers = n_layers
        self.learning_rate = learning_rate
        
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        # Your training logic here
        return self
    
    def predict(self, X):
        # Your prediction logic here
        return predictions
    
    def predict_proba(self, X):
        # Your probability prediction logic here
        return probabilities

# Train it
model = MyCustomClassifier(n_layers=5)
model.fit(X_train, y_train)

# Save it
from src.models.persistence import save_custom_model
save_custom_model(
    model=model,
    path="models/custom/my_classifier",
    feature_names=list(X.columns),
    metadata={"description": "My custom neural network"}
)
```

### Dashboard Integration

1. Navigate to **Custom Models** page (🔧) in the dashboard
2. Upload your saved model or create new ones
3. Use in **Model Training** page alongside standard models
4. Evaluate in **Model Evaluation** page with full metrics

---

## 🎯 Inverse Optimization & Treatment Recommendations

**NEW in v2.1**: Find optimal feature values (treatments, interventions) to achieve desired predictions!

The Inverse Optimization module answers: *"What treatment strategy minimizes mortality risk?"*

### Key Features

- **Treatment Optimization**: Find optimal medication dosages and interventions
- **Uncertainty Quantification**: Bootstrap confidence intervals for optimal values
- **Sensitivity Analysis**: Understand robustness of recommendations
- **Interactive Visualizations**: Modern Plotly charts with zoom, pan, export
- **Multiple Optimization Methods**: SLSQP, COBYLA, Differential Evolution
- **Counterfactual Explanations**: Minimal changes to flip predictions

### Quick Example

```python
from src.explainability import InverseOptimizer

# Create optimizer
optimizer = InverseOptimizer(
    model=trained_model,
    feature_names=feature_list
)

# Find optimal treatment
result = optimizer.optimize(
    target_value=0.1,  # Target 10% mortality
    modifiable_features=['aspirin_dose', 'statin_dose', 'reperfusion_time'],
    fixed_features={'age': 68, 'sex': 1, 'diabetes': 1},
    reference_data=patient_data,
    method='SLSQP',
    n_iterations=10
)

print(f"Optimal treatment: {result['optimal_values']}")
print(f"Achieved mortality: {result['achieved_prediction']:.1%}")

# Compute confidence intervals
ci_result = optimizer.compute_confidence_intervals(
    target_value=0.1,
    modifiable_features=['aspirin_dose', 'statin_dose'],
    n_bootstrap=50,
    confidence_level=0.95
)
```

### Streamlit Interface

Navigate to **🎯 Inverse Optimization** page in the dashboard for:
- Interactive feature selection (modifiable vs fixed)
- Real-time optimization with progress tracking
- Confidence interval computation
- Sensitivity analysis with interactive plots
- Multiple export formats (CSV, JSON)
- Comprehensive help and clinical examples

### Documentation

- **[Complete Guide](docs/user-guide/inverse-optimization.md)** - Full documentation with examples
- **[Quick Reference](src/explainability/INVERSE_OPTIMIZATION_README.md)** - Fast lookup
- **[Demo Notebook](notebooks/inverse_optimization_demo.md)** - Step-by-step tutorial
- **[Implementation Summary](INVERSE_OPTIMIZATION_SUMMARY.md)** - Technical details

### Use Cases

1. **Mortality Risk Reduction**: Find treatment strategies to reduce patient mortality
2. **Intervention Planning**: Optimize timing and dosages for maximum benefit
3. **Personalized Medicine**: Patient-specific treatment recommendations
4. **What-If Analysis**: Explore counterfactual scenarios

⚠️ **Important**: Always validate computational recommendations with clinical experts!

📖 **Complete guide:** [Custom Models Guide](docs/CUSTOM_MODELS_GUIDE.md)

---

## Dashboard (Streamlit)

- Cargar o seleccionar un paciente (fila), mostrar probabilidad predicha, contribuciones globales (SHAP) y análisis "what-if" para variables numéricas.
- Ejecuta:

```powershell
streamlit run streamlit_app.py -- --data $env:DATASET_PATH
```

## Predicción por lotes

```powershell
python -m src.predict --model .\models\best_classifier_mortality.joblib --input .\mis_pacientes.csv --output .\predicciones.csv
```

## Seguimiento de experimentos (opcional, MLflow)

Define las variables para activar el logging:

```powershell
$env:EXPERIMENT_TRACKER = "mlflow"
$env:TRACKING_URI = "http://localhost:5000"   # o ruta de tracking
```

## DATASET_PATH (única fuente de datos)

- Debe apuntar al fichero CSV/Parquet del dataset.
- Configúralo vía variable de entorno o pásalo por `--data` en CLI/Streamlit.
- Ejemplo: `C:\\data\\ami_dataset.csv`.

## Consideraciones de privacidad

- No se registran ni se muestran identificadores directos de pacientes.
- Si tu dataset incluye IDs (patient_id, MRN), evita usarlos como features y no los vuelques en logs/figuras.

## Reproducibilidad

- Semilla fija
- Validación cruzada estratificada y hold-out final
- Pipelines y modelos guardados en `models/` y reutilizables

## Resolución de problemas

- Dependencias opcionales: si ves errores de importación de `imblearn`, `mlflow` o `shap`, asegúrate de instalarlas (están en `requirements.txt` y `environment.yml`).
- Columnas faltantes: el código tolera columnas opcionales; si falta la columna objetivo configurada, se devolverá un error claro.
- GPU/NN: si no hay CUDA, la red neuronal usa CPU automáticamente.
