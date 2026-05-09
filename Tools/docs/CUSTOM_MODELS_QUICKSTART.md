# Sistema de Modelos Personalizados - Guía Rápida

## 🎯 Descripción

El **Sistema de Modelos Personalizados** permite a los programadores e investigadores crear, definir y gestionar sus propias arquitecturas de Machine Learning que se integran perfectamente con el sistema de predicción de mortalidad por IAM.

## 🚀 Características Principales

- ✅ **Tutorial Interactivo**: Onboarding completo con 7 pasos guiados en español
- 💻 **Editor de Código**: Escribe y valida modelos directamente en la interfaz
- 📁 **Carga de Archivos**: Sube archivos `.py` con tus modelos
- 📚 **Gestión Completa**: Visualiza, edita y elimina modelos
- 📖 **Documentación Integrada**: API reference, ejemplos y troubleshooting
- 🔄 **Integración Automática**: Los modelos se integran con el pipeline de entrenamiento

## 📝 Acceso al Sistema

### Desde la Aplicación

1. Inicia la aplicación: `streamlit run Dashboard.py`
2. Navega a la página **🔧 Custom Models** en la barra lateral
3. Si es tu primera vez, aparecerá el tutorial interactivo

### Onboarding

El tutorial incluye:

1. **Bienvenida**: Introducción al sistema
2. **Arquitectura**: Estructura de los modelos custom
3. **Ejemplo Simple**: Clasificador básico
4. **Ejemplo Avanzado**: Red neuronal
5. **Cómo Usar**: Flujo de trabajo completo
6. **Buenas Prácticas**: Recomendaciones importantes
7. **Listo para Empezar**: Resumen y recursos

## 💻 Crear un Modelo Personalizado

### Opción 1: Editor de Código

1. Ve a la pestaña **"Editor de Código"**
2. Selecciona un template o escribe desde cero
3. Valida el código con el botón **"Validar Código"**
4. Guarda con **"Guardar Código"**

### Opción 2: Cargar Archivo

1. Ve a la pestaña **"Cargar Archivo"**
2. Sube un archivo `.py` con tu modelo
3. El sistema detecta automáticamente las clases
4. Importa al sistema

## 📚 Estructura de un Modelo

```python
from src.models.custom_base import BaseCustomClassifier
import numpy as np

class MiModeloPersonalizado(BaseCustomClassifier):
    """Descripción del modelo."""
    
    def __init__(self, param1=100, name="MiModelo"):
        super().__init__(name=name)
        self.param1 = param1
        # Inicializar componentes internos
    
    def fit(self, X, y, **kwargs):
        """Entrenar el modelo."""
        self._validate_input(X, training=True)
        y = self._validate_targets(y, training=True)
        X = self._convert_to_array(X)
        
        # Tu lógica de entrenamiento aquí
        
        self.is_fitted_ = True
        return self
    
    def predict_proba(self, X):
        """Predecir probabilidades."""
        self._validate_input(X, training=False)
        X = self._convert_to_array(X)
        
        # Tu lógica de predicción aquí
        # Debe retornar (n_samples, 2) para clasificación binaria
        
        return probabilities
    
    def get_params(self, deep=True):
        """Obtener parámetros."""
        return {'param1': self.param1, 'name': self.name}
```

## 🔄 Flujo de Trabajo Completo

### 1. Definir el Modelo

- Usa el editor o carga un archivo
- El modelo debe heredar de `BaseCustomClassifier` o `BaseCustomRegressor`
- Implementa los métodos requeridos

### 2. Validar y Guardar

- Valida la sintaxis y estructura
- Guarda en `src/models/custom/`
- El sistema genera metadata automáticamente

### 3. Entrenar

- Ve a la página **🤖 Model Training**
- Activa **"Incluir Modelos Personalizados"**
- Selecciona tu modelo
- Configura hiperparámetros
- Entrena con Cross-Validation

### 4. Evaluar

- Ve a la página **📈 Model Evaluation**
- Selecciona tu modelo entrenado
- Obtén métricas completas (ROC-AUC, AUPRC, etc.)
- Compara con modelos estándar

### 5. Explicar

- Ve a la página **🔍 Explainability**
- Genera valores SHAP
- Obtén feature importance
- Analiza predicciones individuales

### 6. Usar en Producción

- Ve a la página **🔮 Predictions**
- Carga datos de pacientes
- Usa tu modelo para predicciones

## 📖 Documentación Completa

### En la Aplicación

La pestaña **"Documentación"** incluye:

- **Inicio Rápido**: Flujo básico de trabajo
- **API Reference**: Documentación completa de la API
- **Ejemplos**: Casos comunes de uso
- **Troubleshooting**: Solución a problemas comunes

### Archivos de Documentación

- **Guía Completa**: `Tools/docs/CUSTOM_MODELS_GUIDE.md`
- **Arquitectura**: `Tools/docs/CUSTOM_MODELS_ARCHITECTURE.md`
- **Código Base**: `Tools/src/models/custom_base.py`

## 🎓 Templates Incluidos

### 1. Clasificador Simple
Random Forest personalizado con post-procesamiento

### 2. Red Neuronal
MLP con normalización automática integrada

### 3. Ensemble
Combinación de múltiples modelos con votación ponderada

## ✅ Requisitos Técnicos

### Métodos Obligatorios

Para **todos los modelos**:
- `__init__()` - Constructor
- `fit(X, y)` - Entrenamiento
- `predict(X)` - Predicción
- `get_params()` - Obtener parámetros
- `set_params()` - Establecer parámetros

Para **clasificadores** (adicional):
- `predict_proba(X)` - Probabilidades
- `classes_` - Array de clases (set en fit)

### Métodos Helper Disponibles

- `_validate_input(X, training)` - Validar entrada
- `_validate_targets(y, training)` - Validar targets
- `_convert_to_array(X)` - Convertir a numpy array

## 🔧 Gestión de Modelos

### Ver Modelos Disponibles

Pestaña **"Gestionar Modelos"**:
- Lista todos los modelos guardados
- Muestra metadata y clases detectadas
- Opciones: Ver código, Editar, Eliminar

### Editar Modelo

1. Click en **"Editar"** en el modelo deseado
2. El código se carga en el editor
3. Modifica y guarda con nuevo nombre o sobrescribe

### Eliminar Modelo

1. Click en **"Eliminar"**
2. Confirma la acción
3. El modelo se elimina del sistema

## ⚠️ Consideraciones Importantes

### ✅ Hacer

1. **Validar siempre** antes de guardar
2. **Documentar** tu código con docstrings
3. **Probar** con datos pequeños primero
4. **Versionar** tus modelos (nombres diferentes)
5. **Usar nombres descriptivos**

### ❌ Evitar

1. **No modificar** datos in-place
2. **No usar** variables globales
3. **No ignorar** validación
4. **No hardcodear** valores
5. **No skip** preprocessing

## 🆘 Soporte

### Problemas Comunes

Consulta la sección de **Troubleshooting** en la documentación de la app:

- Error: "Model must be fitted before prediction"
- Error: "X has wrong number of features"
- Error: "predict_proba returns wrong shape"
- Error: "Missing classes_ attribute"
- SHAP no funciona
- Cross-validation falla

### Recursos

- **Documentación Completa**: `Tools/docs/CUSTOM_MODELS_GUIDE.md`
- **Ejemplos**: `Tools/src/models/custom_base.py`
- **Tests**: `Tools/tests/test_custom_models.py`

## 📁 Estructura de Archivos

```
Tools/
├── src/
│   └── models/
│       ├── custom/              # Tu modelos aquí
│       │   ├── __init__.py
│       │   ├── mi_modelo.py     # Tu archivo
│       │   └── mi_modelo.json   # Metadata
│       ├── custom_base.py       # Clases base
│       └── persistence.py       # Guardar/cargar
├── dashboard/
│   └── pages/
│       └── 07_🔧_Custom_Models.py  # Interfaz
└── docs/
    ├── CUSTOM_MODELS_GUIDE.md   # Guía completa
    └── CUSTOM_MODELS_ARCHITECTURE.md
```

## 🎯 Ejemplo Rápido

```python
# 1. Definir modelo
from src.models.custom_base import BaseCustomClassifier
from sklearn.ensemble import RandomForestClassifier

class MiRF(BaseCustomClassifier):
    def __init__(self, n_estimators=100):
        super().__init__(name="MiRF")
        self.n_estimators = n_estimators
        self._model = RandomForestClassifier(n_estimators=n_estimators)
    
    def fit(self, X, y, **kwargs):
        self._validate_input(X, training=True)
        y = self._validate_targets(y, training=True)
        self._model.fit(X, y)
        self.is_fitted_ = True
        return self
    
    def predict_proba(self, X):
        self._validate_input(X, training=False)
        return self._model.predict_proba(X)
    
    def get_params(self, deep=True):
        return {'n_estimators': self.n_estimators, 'name': self.name}

# 2. Guardar en la app
# 3. Entrenar en 🤖 Model Training
# 4. Evaluar en 📈 Model Evaluation
# 5. Usar en 🔮 Predictions
```

## 🎉 ¡Listo!

Ahora puedes crear y gestionar tus propios modelos personalizados de Machine Learning integrados con el sistema completo de predicción.

**¡Feliz experimentación! 🚀**
