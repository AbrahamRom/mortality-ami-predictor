"""Página de Gestión de Modelos Personalizados.

Esta página proporciona funcionalidad para:
- Onboarding interactivo para crear modelos custom
- Editor de código para definir modelos
- Carga de archivos Python con definiciones de modelos
- Gestión de modelos definidos
- Integración con el pipeline de entrenamiento
"""

import streamlit as st
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
import tempfile
import importlib.util
import inspect
import time

# Add src to path
root_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(root_path))

from src.models.persistence import (
    save_custom_model,
    load_custom_model,
    list_saved_models,
    create_model_bundle,
    load_model_bundle,
)
from src.models.custom_base import BaseCustomModel, BaseCustomClassifier, BaseCustomRegressor

# Page config
st.set_page_config(
    page_title="Modelos Personalizados",
    page_icon="🔧",
    layout="wide"
)

# Define custom models directory
CUSTOM_MODELS_DIR = root_path / "models" / "custom"
CUSTOM_MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Define code templates directory
CODE_TEMPLATES_DIR = root_path / "src" / "models" / "custom"
CODE_TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)

# Initialize session state
if "onboarding_completed" not in st.session_state:
    st.session_state.onboarding_completed = False
if "show_onboarding" not in st.session_state:
    st.session_state.show_onboarding = True
if "current_step" not in st.session_state:
    st.session_state.current_step = 0
if "custom_model_code" not in st.session_state:
    st.session_state.custom_model_code = ""
if "loaded_model_classes" not in st.session_state:
    st.session_state.loaded_model_classes = []


# ============================================================================
# ONBOARDING CONTENT
# ============================================================================

ONBOARDING_STEPS = [
    {
        "title": "🎯 Bienvenido al Sistema de Modelos Personalizados",
        "content": """
        ### ¿Qué son los Modelos Personalizados?
        
        Los **Modelos Personalizados** te permiten crear tus propias arquitecturas de Machine Learning 
        que se integran perfectamente con el sistema de predicción de mortalidad por IAM.
        
        #### ✨ Características principales:
        
        - 🔧 **Compatibilidad total con scikit-learn**: Usa `fit()`, `predict()`, y `predict_proba()`
        - 🔄 **Integración automática**: Se integran con el pipeline de entrenamiento y evaluación
        - 📊 **Métricas completas**: Obtén ROC-AUC, AUPRC, y todas las métricas estándar
        - 🔍 **Explicabilidad**: Genera valores SHAP y feature importance
        - 💾 **Persistencia**: Guarda y carga modelos con versionado automático
        - 📈 **Dashboard**: Gestiona tus modelos desde la interfaz web
        
        #### 🎓 ¿Para quién es esto?
        
        Este sistema está diseñado para **programadores e investigadores** que quieren:
        - Experimentar con arquitecturas personalizadas
        - Implementar modelos de papers de investigación
        - Comparar nuevos enfoques con modelos estándar
        - Mantener control total sobre el proceso de entrenamiento
        """,
        "icon": "🎯"
    },
    {
        "title": "🏗️ Arquitectura del Sistema",
        "content": """
        ### Estructura de un Modelo Personalizado
        
        El sistema se basa en **clases base** que proporcionan la estructura necesaria:
        
        ```
        BaseCustomModel (Abstracta)
        ├── BaseCustomClassifier → Para clasificación
        └── BaseCustomRegressor  → Para regresión
        ```
        
        #### 📋 Métodos Requeridos:
        
        **Para todos los modelos:**
        - `__init__()` - Constructor con hiperparámetros
        - `fit(X, y)` - Entrenar el modelo
        - `predict(X)` - Hacer predicciones
        - `get_params()` / `set_params()` - Gestión de parámetros
        
        **Para clasificadores (adicional):**
        - `predict_proba(X)` - Probabilidades de clase
        - `classes_` - Array con las clases (se establece en `fit`)
        
        #### 🔄 Flujo de Trabajo:
        
        1. **Definir** tu modelo heredando de `BaseCustomClassifier`
        2. **Implementar** los métodos requeridos
        3. **Guardar** el código Python del modelo
        4. **Entrenar** usando el dashboard (página 🤖 Model Training)
        5. **Evaluar** con todas las métricas (página 📈 Model Evaluation)
        6. **Explicar** con SHAP y feature importance (página 🔍 Explainability)
        """,
        "icon": "🏗️"
    },
    {
        "title": "📝 Ejemplo: Clasificador Simple",
        "content": """
        ### Ejemplo Básico de Clasificador
        
        Aquí tienes un ejemplo completo de un clasificador personalizado:
        
        ```python
        from src.models.custom_base import BaseCustomClassifier
        from sklearn.ensemble import RandomForestClassifier
        import numpy as np
        
        class MiClasificadorRF(BaseCustomClassifier):
            '''Clasificador Random Forest personalizado con post-procesamiento.'''
            
            def __init__(self, n_estimators=100, max_depth=None, 
                         threshold=0.5, name="MiClasificadorRF"):
                super().__init__(name=name)
                self.n_estimators = n_estimators
                self.max_depth = max_depth
                self.threshold = threshold
                
                # Modelo interno
                self._rf = RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    random_state=42
                )
            
            def fit(self, X, y, **kwargs):
                '''Entrenar el modelo.'''
                # Validar entrada
                self._validate_input(X, training=True)
                y = self._validate_targets(y, training=True)
                
                # Convertir a array si es necesario
                X = self._convert_to_array(X)
                
                # Entrenar
                self._rf.fit(X, y)
                self.is_fitted_ = True
                
                return self
            
            def predict_proba(self, X):
                '''Predecir probabilidades.'''
                self._validate_input(X, training=False)
                X = self._convert_to_array(X)
                
                return self._rf.predict_proba(X)
            
            def predict(self, X):
                '''Predecir clases con threshold personalizado.'''
                proba = self.predict_proba(X)
                # Usar threshold personalizado para clase positiva
                return (proba[:, 1] >= self.threshold).astype(int)
            
            def get_params(self, deep=True):
                '''Obtener parámetros.'''
                return {
                    'n_estimators': self.n_estimators,
                    'max_depth': self.max_depth,
                    'threshold': self.threshold,
                    'name': self.name
                }
        ```
        
        #### 🔑 Puntos Clave:
        
        - Hereda de `BaseCustomClassifier`
        - Usa `_validate_input()` y `_validate_targets()`
        - Establece `self.is_fitted_ = True` después de entrenar
        - Implementa `predict_proba()` retornando (n_samples, n_classes)
        - Define todos los hiperparámetros en `__init__`
        """,
        "icon": "📝"
    },
    {
        "title": "🚀 Ejemplo Avanzado: Red Neuronal",
        "content": """
        ### Clasificador con Red Neuronal Personalizada
        
        Ejemplo más avanzado con preprocessing integrado:
        
        ```python
        from src.models.custom_base import BaseCustomClassifier
        from sklearn.neural_network import MLPClassifier
        from sklearn.preprocessing import StandardScaler
        import numpy as np
        
        class RedNeuronalProfunda(BaseCustomClassifier):
            '''Red neuronal profunda con normalización automática.'''
            
            def __init__(self, hidden_layers=(200, 100, 50), 
                         learning_rate=0.001, dropout=0.2,
                         max_iter=300, name="RedNeuronalProfunda"):
                super().__init__(name=name)
                self.hidden_layers = hidden_layers
                self.learning_rate = learning_rate
                self.dropout = dropout
                self.max_iter = max_iter
                
                # Componentes internos
                self._scaler = StandardScaler()
                self._mlp = MLPClassifier(
                    hidden_layer_sizes=hidden_layers,
                    learning_rate_init=learning_rate,
                    max_iter=max_iter,
                    early_stopping=True,
                    validation_fraction=0.1,
                    random_state=42
                )
            
            def fit(self, X, y, **kwargs):
                '''Entrenar con normalización automática.'''
                self._validate_input(X, training=True)
                y = self._validate_targets(y, training=True)
                X = self._convert_to_array(X)
                
                # Normalizar datos
                X_scaled = self._scaler.fit_transform(X)
                
                # Entrenar red neuronal
                self._mlp.fit(X_scaled, y)
                self.is_fitted_ = True
                
                return self
            
            def predict_proba(self, X):
                '''Predecir con normalización.'''
                self._validate_input(X, training=False)
                X = self._convert_to_array(X)
                
                # Normalizar con los parámetros de entrenamiento
                X_scaled = self._scaler.transform(X)
                
                return self._mlp.predict_proba(X_scaled)
            
            def get_params(self, deep=True):
                return {
                    'hidden_layers': self.hidden_layers,
                    'learning_rate': self.learning_rate,
                    'dropout': self.dropout,
                    'max_iter': self.max_iter,
                    'name': self.name
                }
        ```
        
        #### ⚡ Características Avanzadas:
        
        - Preprocessing integrado (StandardScaler)
        - Early stopping automático
        - Validación durante entrenamiento
        - Gestión de estado completa
        """,
        "icon": "🚀"
    },
    {
        "title": "🔧 Cómo Usar Tus Modelos",
        "content": """
        ### Proceso Completo de Uso
        
        #### 1️⃣ Definir tu Modelo
        
        Puedes hacerlo de dos formas:
        
        **Opción A: Editor de Código** (en esta página)
        - Escribe tu código directamente en el editor
        - Valida la sintaxis en tiempo real
        - Guarda el archivo Python
        
        **Opción B: Cargar Archivo** (en esta página)
        - Crea tu modelo en tu editor favorito
        - Sube el archivo `.py`
        - El sistema detecta automáticamente las clases
        
        #### 2️⃣ Entrenar el Modelo
        
        Ve a la página **🤖 Model Training**:
        - Activa "Incluir Modelos Personalizados"
        - Selecciona tu modelo de la lista
        - Configura hiperparámetros si es necesario
        - El modelo se entrena con Cross-Validation
        
        #### 3️⃣ Evaluar Resultados
        
        Ve a la página **📈 Model Evaluation**:
        - Selecciona tu modelo entrenado
        - Obtén métricas completas (ROC-AUC, AUPRC, etc.)
        - Compara con modelos estándar
        - Visualiza matrices de confusión y curvas ROC
        
        #### 4️⃣ Explicar Predicciones
        
        Ve a la página **🔍 Explainability**:
        - Genera valores SHAP
        - Obtén feature importance
        - Explica predicciones individuales
        - Identifica patrones del modelo
        
        #### 5️⃣ Usar en Producción
        
        Ve a la página **🔮 Predictions**:
        - Carga nuevos datos de pacientes
        - Usa tu modelo para predicciones
        - Genera reportes clínicos
        """,
        "icon": "🔧"
    },
    {
        "title": "✅ Buenas Prácticas",
        "content": """
        ### 📚 Recomendaciones Importantes
        
        #### ✅ Hacer:
        
        1. **Validación de Entrada**
           - Usa `_validate_input()` en `fit()` y `predict()`
           - Verifica dimensiones y tipos de datos
        
        2. **Gestión de Estado**
           - Marca `self.is_fitted_ = True` después de entrenar
           - Guarda atributos con underscore final (`coef_`, `classes_`)
        
        3. **Hiperparámetros**
           - Define todos en `__init__()`
           - Implementa `get_params()` y `set_params()`
           - Usa valores por defecto razonables
        
        4. **Documentación**
           - Añade docstrings a todas las funciones
           - Explica los parámetros del constructor
           - Documenta el propósito del modelo
        
        5. **Testing**
           - Prueba con datos pequeños primero
           - Verifica que `predict_proba()` retorna shape correcto
           - Comprueba compatibilidad con cross-validation
        
        #### ❌ Evitar:
        
        1. **Modificar datos en lugar (in-place)**
           - No modifiques `X` directamente
           - Usa copias si necesitas transformar
        
        2. **Variables globales**
           - Todo debe ser atributo de la clase
           - No uses estado fuera de `self`
        
        3. **Ignorar validación**
           - Siempre valida antes de predecir
           - Verifica que el modelo está entrenado
        
        4. **Hardcodear valores**
           - Usa parámetros configurables
           - Evita magic numbers
        
        5. **Skip preprocessing**
           - Si escalas en `fit()`, escala en `predict()`
           - Guarda transformadores como atributos
        
        #### 🎯 Para Clasificación Binaria:
        
        ```python
        # ✅ CORRECTO: predict_proba retorna (n_samples, 2)
        def predict_proba(self, X):
            proba_positive = self._model.predict_proba(X)
            proba_negative = 1 - proba_positive
            return np.column_stack([proba_negative, proba_positive])
        
        # ❌ INCORRECTO: retorna (n_samples,)
        def predict_proba(self, X):
            return self._model.predict_proba(X)  # Solo una columna
        ```
        """,
        "icon": "✅"
    },
    {
        "title": "🎓 ¡Listo para Empezar!",
        "content": """
        ### Ya tienes todo lo necesario
        
        #### 📋 Checklist Final:
        
        - ✅ Entiendes la estructura de `BaseCustomClassifier`
        - ✅ Conoces los métodos requeridos
        - ✅ Has visto ejemplos de código
        - ✅ Sabes cómo integrar con el pipeline
        - ✅ Conoces las buenas prácticas
        
        #### 🚀 Próximos Pasos:
        
        1. **Crea tu primer modelo** usando el editor o cargando un archivo
        2. **Guarda el código** para que esté disponible en el sistema
        3. **Ve a Model Training** para entrenar tu modelo
        4. **Evalúa y compara** con modelos estándar
        5. **Experimenta e itera** hasta obtener los mejores resultados
        
        #### 💡 Recursos Adicionales:
        
        - **Documentación completa**: `Tools/docs/CUSTOM_MODELS_GUIDE.md`
        - **Ejemplos**: `Tools/src/models/custom_base.py`
        - **Tests**: `Tools/tests/test_custom_models.py`
        
        #### 🆘 ¿Necesitas Ayuda?
        
        Si tienes problemas:
        1. Revisa los mensajes de error en el validador
        2. Consulta la sección de Troubleshooting en la guía
        3. Verifica que heredas de la clase base correcta
        4. Comprueba que todos los métodos requeridos están implementados
        
        ---
        
        ### ¡Ahora cierra este tutorial y comienza a crear!
        
        Haz clic en "Completar Tutorial" para acceder al editor de modelos.
        """,
        "icon": "🎓"
    }
]


# ============================================================================
# CODE TEMPLATES
# ============================================================================

TEMPLATE_SIMPLE_CLASSIFIER = '''from src.models.custom_base import BaseCustomClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np

class MiClasificadorPersonalizado(BaseCustomClassifier):
    """Clasificador personalizado basado en Random Forest.
    
    Este es un template básico que puedes modificar según tus necesidades.
    """
    
    def __init__(self, n_estimators=100, max_depth=None, 
                 min_samples_split=2, name="MiClasificador"):
        """
        Inicializar clasificador.
        
        Args:
            n_estimators: Número de árboles en el bosque
            max_depth: Profundidad máxima de los árboles
            min_samples_split: Mínimo de muestras para dividir un nodo
            name: Nombre del modelo
        """
        super().__init__(name=name)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        
        # Modelo interno
        self._model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=42
        )
    
    def fit(self, X, y, **kwargs):
        """Entrenar el modelo."""
        # Validar entrada
        self._validate_input(X, training=True)
        y = self._validate_targets(y, training=True)
        
        # Convertir a array
        X = self._convert_to_array(X)
        
        # Entrenar
        self._model.fit(X, y)
        self.is_fitted_ = True
        
        return self
    
    def predict_proba(self, X):
        """Predecir probabilidades de clase."""
        self._validate_input(X, training=False)
        X = self._convert_to_array(X)
        
        return self._model.predict_proba(X)
    
    def get_params(self, deep=True):
        """Obtener parámetros del modelo."""
        return {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'name': self.name
        }
    
    def set_params(self, **params):
        """Establecer parámetros del modelo."""
        for key, value in params.items():
            setattr(self, key, value)
        
        # Recrear modelo con nuevos parámetros
        self._model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            random_state=42
        )
        
        return self
'''

TEMPLATE_NEURAL_NETWORK = '''from src.models.custom_base import BaseCustomClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np

class RedNeuronalPersonalizada(BaseCustomClassifier):
    """Red neuronal personalizada con preprocessing integrado.
    
    Incluye normalización automática y configuración avanzada.
    """
    
    def __init__(self, hidden_layers=(100, 50), learning_rate=0.001,
                 max_iter=200, name="RedNeuronal"):
        """
        Inicializar red neuronal.
        
        Args:
            hidden_layers: Tuple con el número de neuronas por capa oculta
            learning_rate: Tasa de aprendizaje
            max_iter: Número máximo de iteraciones
            name: Nombre del modelo
        """
        super().__init__(name=name)
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        
        # Componentes
        self._scaler = StandardScaler()
        self._mlp = MLPClassifier(
            hidden_layer_sizes=hidden_layers,
            learning_rate_init=learning_rate,
            max_iter=max_iter,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=42
        )
    
    def fit(self, X, y, **kwargs):
        """Entrenar con normalización automática."""
        self._validate_input(X, training=True)
        y = self._validate_targets(y, training=True)
        X = self._convert_to_array(X)
        
        # Normalizar
        X_scaled = self._scaler.fit_transform(X)
        
        # Entrenar
        self._mlp.fit(X_scaled, y)
        self.is_fitted_ = True
        
        return self
    
    def predict_proba(self, X):
        """Predecir con normalización."""
        self._validate_input(X, training=False)
        X = self._convert_to_array(X)
        
        X_scaled = self._scaler.transform(X)
        return self._mlp.predict_proba(X_scaled)
    
    def get_params(self, deep=True):
        """Obtener parámetros."""
        return {
            'hidden_layers': self.hidden_layers,
            'learning_rate': self.learning_rate,
            'max_iter': self.max_iter,
            'name': self.name
        }
'''

TEMPLATE_ENSEMBLE = '''from src.models.custom_base import BaseCustomClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np

class EnsemblePersonalizado(BaseCustomClassifier):
    """Ensemble de múltiples modelos con votación ponderada.
    
    Combina Random Forest, Gradient Boosting y Regresión Logística.
    """
    
    def __init__(self, rf_weight=0.4, gb_weight=0.4, lr_weight=0.2,
                 name="Ensemble"):
        """
        Inicializar ensemble.
        
        Args:
            rf_weight: Peso para Random Forest
            gb_weight: Peso para Gradient Boosting
            lr_weight: Peso para Logistic Regression
            name: Nombre del modelo
        """
        super().__init__(name=name)
        # Guardar parámetros originales (NO modificar para sklearn clone)
        self.rf_weight = rf_weight
        self.gb_weight = gb_weight
        self.lr_weight = lr_weight
        
        # Calcular pesos normalizados (en variable interna)
        total = rf_weight + gb_weight + lr_weight
        self._rf_weight_norm = rf_weight / total
        self._gb_weight_norm = gb_weight / total
        self._lr_weight_norm = lr_weight / total
        
        # Modelos
        self._rf = RandomForestClassifier(n_estimators=100, random_state=42)
        self._gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
        self._lr = LogisticRegression(max_iter=1000, random_state=42)
    
    def fit(self, X, y, **kwargs):
        """Entrenar todos los modelos."""
        self._validate_input(X, training=True)
        y = self._validate_targets(y, training=True)
        X = self._convert_to_array(X)
        
        # Entrenar cada modelo
        self._rf.fit(X, y)
        self._gb.fit(X, y)
        self._lr.fit(X, y)
        
        self.is_fitted_ = True
        return self
    
    def predict_proba(self, X):
        """Predecir con votación ponderada."""
        self._validate_input(X, training=False)
        X = self._convert_to_array(X)
        
        # Obtener probabilidades de cada modelo
        proba_rf = self._rf.predict_proba(X)
        proba_gb = self._gb.predict_proba(X)
        proba_lr = self._lr.predict_proba(X)
        
        # Combinar con pesos normalizados
        proba_ensemble = (
            self._rf_weight_norm * proba_rf +
            self._gb_weight_norm * proba_gb +
            self._lr_weight_norm * proba_lr
        )
        
        return proba_ensemble
    
    def get_params(self, deep=True):
        """Obtener parámetros."""
        return {
            'rf_weight': self.rf_weight,
            'gb_weight': self.gb_weight,
            'lr_weight': self.lr_weight,
            'name': self.name
        }
'''


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================



def show_onboarding():
    """Mostrar tutorial interactivo de onboarding."""
    st.title("🎓 Tutorial: Sistema de Modelos Personalizados")
    
    # Progress bar
    progress = (st.session_state.current_step + 1) / len(ONBOARDING_STEPS)
    st.progress(progress)
    
    # Current step
    step = ONBOARDING_STEPS[st.session_state.current_step]
    
    # Display step content
    st.markdown(f"## {step['icon']} {step['title']}")
    st.markdown(step['content'])
    
    # Navigation buttons
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        if st.session_state.current_step > 0:
            if st.button("⬅️ Anterior", use_container_width=True):
                st.session_state.current_step -= 1
                st.rerun()
    
    with col2:
        st.markdown(f"<center>Paso {st.session_state.current_step + 1} de {len(ONBOARDING_STEPS)}</center>", 
                   unsafe_allow_html=True)
    
    with col3:
        if st.session_state.current_step < len(ONBOARDING_STEPS) - 1:
            if st.button("Siguiente ➡️", use_container_width=True, type="primary"):
                st.session_state.current_step += 1
                st.rerun()
        else:
            if st.button("✅ Completar Tutorial", use_container_width=True, type="primary"):
                st.session_state.onboarding_completed = True
                st.session_state.show_onboarding = False
                st.rerun()
    
    # Skip button
    st.markdown("---")
    if st.button("⏭️ Saltar Tutorial", key="skip_onboarding"):
        st.session_state.onboarding_completed = True
        st.session_state.show_onboarding = False
        st.rerun()


def validate_model_code(code: str) -> dict:
    """
    Validar código de modelo personalizado.
    
    Returns:
        dict con 'valid' (bool), 'errors' (list), 'warnings' (list), 'classes' (list)
    """
    result = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'classes': []
    }
    
    try:
        # 1. Verificar sintaxis Python básica
        compile(code, '<string>', 'exec')
        
        # 2. Buscar imports y clases usando análisis de texto (más seguro que exec)
        import re
        
        # Buscar clases que heredan de BaseCustom*
        class_pattern = r'class\s+(\w+)\s*\([^)]*(?:BaseCustomClassifier|BaseCustomRegressor)[^)]*\):'
        matches = re.findall(class_pattern, code)
        
        if matches:
            for class_name in matches:
                # Determinar tipo basándose en la clase base
                if re.search(rf'class\s+{class_name}\s*\([^)]*BaseCustomClassifier', code):
                    class_type = 'classifier'
                elif re.search(rf'class\s+{class_name}\s*\([^)]*BaseCustomRegressor', code):
                    class_type = 'regressor'
                else:
                    class_type = 'unknown'
                
                result['classes'].append({
                    'name': class_name,
                    'type': class_type,
                })
        
        # 3. Verificar imports necesarios
        if 'from src.models.custom_base import' not in code:
            result['warnings'].append(
                "Falta import: 'from src.models.custom_base import BaseCustomClassifier'"
            )
        
        # 4. Verificar métodos requeridos en el código
        if not result['classes']:
            result['warnings'].append(
                "No se encontraron clases que hereden de BaseCustomClassifier o BaseCustomRegressor"
            )
        else:
            # Verificar métodos básicos
            required_methods = ['def fit', 'def predict', 'def get_params']
            for method in required_methods:
                if method not in code:
                    result['warnings'].append(f"No se encontró '{method}'. Asegúrate de implementarlo.")
        
    except SyntaxError as e:
        result['valid'] = False
        result['errors'].append(f"Error de sintaxis en línea {e.lineno}: {e.msg}")
    except Exception as e:
        result['valid'] = False
        result['errors'].append(f"Error al validar código: {str(e)}")
    
    return result


def save_model_code(code: str, filename: str) -> Path:
    """
    Guardar código de modelo en el directorio de modelos custom.
    
    Returns:
        Path al archivo guardado
    """
    if not filename.endswith('.py'):
        filename += '.py'
    
    filepath = CODE_TEMPLATES_DIR / filename
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(code)
    
    return filepath


def load_model_code(filepath: Path) -> str:
    """Cargar código de modelo desde archivo."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()


def get_model_class_from_file(filepath: Path):
    """
    Importar dinámicamente una clase de modelo desde un archivo.
    
    Returns:
        dict con información de las clases encontradas
    """
    spec = importlib.util.spec_from_file_location("custom_model", filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    classes = []
    for name, obj in inspect.getmembers(module, inspect.isclass):
        if issubclass(obj, (BaseCustomClassifier, BaseCustomRegressor, BaseCustomModel)):
            if obj not in [BaseCustomModel, BaseCustomClassifier, BaseCustomRegressor]:
                classes.append({
                    'name': name,
                    'class': obj,
                    'file': filepath.name,
                    'type': 'classifier' if issubclass(obj, BaseCustomClassifier) else 'regressor'
                })
    
    return classes


def list_available_model_files() -> list:
    """Listar archivos Python en el directorio de modelos custom."""
    if not CODE_TEMPLATES_DIR.exists():
        return []
    
    return sorted([f for f in CODE_TEMPLATES_DIR.glob("*.py") if f.name != "__init__.py"])


# ============================================================================
# MAIN INTERFACE SECTIONS
# ============================================================================

def code_editor_section():
    """Sección del editor de código para definir modelos."""
    st.header("💻 Editor de Código")
    
    st.markdown("""
    Define tu modelo personalizado escribiendo código Python directamente.
    El editor valida automáticamente la sintaxis y detecta las clases de modelo.
    """)
    
    # Template selector
    col1, col2 = st.columns([3, 1])
    
    with col1:
        template_choice = st.selectbox(
            "📋 Comenzar con un template:",
            ["(Código actual)", "Clasificador Simple", "Red Neuronal", "Ensemble", "Vacío"],
            key="template_selector"
        )
    
    with col2:
        if st.button("📥 Cargar Template", use_container_width=True, key="load_template_btn"):
            if template_choice == "Clasificador Simple":
                st.session_state.custom_model_code = TEMPLATE_SIMPLE_CLASSIFIER
            elif template_choice == "Red Neuronal":
                st.session_state.custom_model_code = TEMPLATE_NEURAL_NETWORK
            elif template_choice == "Ensemble":
                st.session_state.custom_model_code = TEMPLATE_ENSEMBLE
            elif template_choice == "Vacío":
                st.session_state.custom_model_code = ""
            st.rerun()
    
    # Container desplegable con las plantillas disponibles para consulta
    with st.expander("📚 Ver Plantillas Disponibles (Referencia)", expanded=False):
        st.markdown("""
        Consulta aquí el código completo de cada plantilla. 
        Usa el selector arriba para cargar una plantilla en el editor.
        """)
        
        template_tabs = st.tabs(["🔷 Clasificador Simple", "🧠 Red Neuronal", "🔗 Ensemble"])
        
        with template_tabs[0]:
            st.markdown("**Clasificador Simple basado en Random Forest**")
            st.markdown("Plantilla básica perfecta para empezar. Incluye:")
            st.markdown("- Constructor con hiperparámetros configurables")
            st.markdown("- Métodos fit() y predict_proba() completos")
            st.markdown("- Gestión de parámetros con get_params() y set_params()")
            st.code(TEMPLATE_SIMPLE_CLASSIFIER, language="python", line_numbers=True)
        
        with template_tabs[1]:
            st.markdown("**Red Neuronal con Preprocessing Integrado**")
            st.markdown("Plantilla avanzada con normalización automática. Incluye:")
            st.markdown("- StandardScaler integrado")
            st.markdown("- MLPClassifier con early stopping")
            st.markdown("- Normalización en fit() y predict()")
            st.code(TEMPLATE_NEURAL_NETWORK, language="python", line_numbers=True)
        
        with template_tabs[2]:
            st.markdown("**Ensemble de Múltiples Modelos**")
            st.markdown("Combina 3 modelos con votación ponderada. Incluye:")
            st.markdown("- Random Forest + Gradient Boosting + Regresión Logística")
            st.markdown("- Pesos configurables para cada modelo")
            st.markdown("- Votación soft (promedio de probabilidades)")
            st.code(TEMPLATE_ENSEMBLE, language="python", line_numbers=True)
    
    # Enhanced code editor with syntax highlighting
    st.markdown("### 📝 Editor de Código con Syntax Highlighting")
    
    # Info about current template
    if st.session_state.custom_model_code:
        line_count = len(st.session_state.custom_model_code.split('\n'))
        st.info(f"📊 Código actual: **{line_count} líneas**")
    else:
        st.warning("📄 Editor vacío - selecciona un template arriba")
    
    # TWO-PANEL APPROACH: Display with syntax highlighting + Edit in text area
    col_display, col_edit = st.columns([1, 1])
    
    with col_display:
        st.markdown("**Vista con Syntax Highlighting:**")
        if st.session_state.custom_model_code:
            st.code(st.session_state.custom_model_code, language="python", line_numbers=True)
        else:
            st.info("No hay código para mostrar")
    
    with col_edit:
        st.markdown("**Editor (edita aquí):**")
        # Add custom CSS for better code display
        st.markdown("""
        <style>
        .stTextArea textarea {
            font-family: 'Courier New', monospace !important;
            font-size: 13px !important;
            line-height: 1.5 !important;
            background-color: #1e1e1e !important;
            color: #d4d4d4 !important;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Code editor - use unique key
        code = st.text_area(
            "Edita tu código:",
            value=st.session_state.custom_model_code,
            height=600,
            key="code_text_area",
            help="Escribe tu código aquí. Debe incluir una clase que herede de BaseCustomClassifier o BaseCustomRegressor.",
            label_visibility="collapsed"
        )
        
        # Update session state when user types
        if code != st.session_state.custom_model_code:
            st.session_state.custom_model_code = code
    
    # Action buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        validate_btn = st.button("🔍 Validar Código", use_container_width=True, type="primary")
    
    with col2:
        save_enabled = len(code.strip()) > 0
        save_btn = st.button("💾 Guardar Código", use_container_width=True, disabled=not save_enabled, type="primary")
    
    with col3:
        clear_btn = st.button("🗑️ Limpiar", use_container_width=True)
    
    # Handle clear button
    if clear_btn:
        st.session_state.custom_model_code = ""
        st.rerun()
    
    # Initialize save mode state
    if 'save_mode' not in st.session_state:
        st.session_state.save_mode = False
    
    # Handle save button - activate save mode
    if save_btn:
        st.session_state.save_mode = True
        st.rerun()
    
    # Show save form if in save mode
    if st.session_state.save_mode:
        st.markdown("---")
        st.markdown("### 💾 Guardar Modelo")
        
        col_name, col_desc = st.columns([1, 2])
        
        with col_name:
            filename = st.text_input(
                "Nombre del archivo:",
                value="mi_modelo_personalizado.py",
                help="El archivo se guardará en src/models/custom/",
                key="save_filename"
            )
        
        with col_desc:
            description = st.text_input(
                "Descripción breve:",
                placeholder="Ej: Clasificador RF con threshold personalizado",
                help="Describe brevemente qué hace tu modelo",
                key="save_description"
            )
        
        col_cancel, col_confirm = st.columns([1, 1])
        
        with col_cancel:
            if st.button("❌ Cancelar", use_container_width=True):
                st.session_state.save_mode = False
                st.rerun()
        
        with col_confirm:
            if st.button("✅ Confirmar Guardado", use_container_width=True, type="primary"):
                with st.spinner("💾 Guardando modelo..."):
                    try:
                        # Validar primero
                        validation = validate_model_code(code)
                        
                        if not validation['valid']:
                            st.error("❌ Código con errores de sintaxis:")
                            for error in validation['errors']:
                                st.error(f"  • {error}")
                        elif not validation['classes']:
                            st.error("❌ No se encontraron clases válidas")
                            st.warning("Debe heredar de BaseCustomClassifier o BaseCustomRegressor")
                        else:
                            # Guardar archivo
                            filepath = save_model_code(code, filename)
                            
                            # Verificar guardado
                            if not filepath.exists():
                                st.error(f"❌ Error al guardar: {filepath}")
                            else:
                                # Guardar metadata
                                metadata = {
                                    'filename': filename,
                                    'description': description,
                                    'classes': validation['classes'],
                                    'created_at': datetime.now().isoformat(),
                                }
                                
                                metadata_path = filepath.with_suffix('.json')
                                with open(metadata_path, 'w', encoding='utf-8') as f:
                                    json.dump(metadata, f, indent=2, ensure_ascii=False)
                                
                                # ¡ÉXITO!
                                st.success("✅ ¡Modelo guardado exitosamente!")
                                st.balloons()
                                st.info(f"📂 {filepath}")
                                st.info(f"📦 {len(validation['classes'])} clase(s): " + ", ".join([c['name'] for c in validation['classes']]))
                                
                                # Desactivar modo guardado y recargar
                                st.session_state.save_mode = False
                                time.sleep(1)
                                st.rerun()
                            
                    except Exception as e:
                        st.error(f"❌ Error: {e}")
                        import traceback
                        with st.expander("Ver detalles del error"):
                            st.code(traceback.format_exc())
    
    if validate_btn and code.strip():
        with st.spinner("Validando código..."):
            validation = validate_model_code(code)
            
            if validation['valid']:
                st.success("✅ Código válido!")
                
                if validation['classes']:
                    st.info(f"📦 Se encontraron {len(validation['classes'])} clase(s) de modelo:")
                    for cls in validation['classes']:
                        st.markdown(f"- **{cls['name']}** ({cls['type']})")
                
                if validation['warnings']:
                    st.warning("⚠️ Advertencias:")
                    for warning in validation['warnings']:
                        st.warning(f"• {warning}")
            else:
                st.error("❌ El código tiene errores de sintaxis:")
                
                # Show errors with context
                for error in validation['errors']:
                    st.error(f"• {error}")
                
                # If there's a line number in the error, show context
                if 'línea' in error.lower() or 'line' in error.lower():
                    with st.expander("🔍 Ver contexto del error"):
                        lines = code.split('\n')
                        # Try to extract line number from error message
                        import re
                        line_match = re.search(r'línea (\d+)|line (\d+)', error.lower())
                        if line_match:
                            error_line = int(line_match.group(1) or line_match.group(2))
                            start = max(0, error_line - 3)
                            end = min(len(lines), error_line + 2)
                            
                            context_lines = []
                            for i in range(start, end):
                                marker = " >>> " if i + 1 == error_line else "     "
                                context_lines.append(f"{marker}{i+1:4d} | {lines[i]}")
                            
                            st.code('\n'.join(context_lines), language='python')
                
                if validation['warnings']:
                    st.warning("⚠️ Advertencias adicionales:")
                    for warning in validation['warnings']:
                        st.warning(f"• {warning}")


def file_upload_section():
    """Sección para cargar archivos Python con definiciones de modelos."""
    st.header("📁 Cargar Archivo Python")
    
    st.markdown("""
    Sube un archivo `.py` que contenga la definición de tu modelo personalizado.
    El sistema detectará automáticamente las clases que heredan de `BaseCustomClassifier` o `BaseCustomRegressor`.
    """)
    
    uploaded_file = st.file_uploader(
        "Selecciona un archivo Python:",
        type=['py'],
        help="Archivo .py con la definición de tu modelo custom"
    )
    
    if uploaded_file is not None:
        # Leer contenido
        code = uploaded_file.read().decode('utf-8')
        
        # Botón para cargar en el editor
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.info("💡 Puedes importar directamente o cargar en el editor para modificarlo")
        
        with col2:
            if st.button("📝 Cargar en Editor", use_container_width=True):
                st.session_state.custom_model_code = code
                st.success("✅ Código cargado en el editor")
                st.info("Ve a la pestaña 'Editor de Código' para editarlo")
        
        # Mostrar preview con números de línea
        with st.expander("👀 Vista previa del código", expanded=True):
            # Add line numbers to code preview
            lines = code.split('\n')
            numbered_code = '\n'.join([f"{i+1:4d} | {line}" for i, line in enumerate(lines)])
            st.code(numbered_code, language='python')
            st.caption(f"📊 Total: {len(lines)} líneas")
        
        # Validar
        with st.spinner("Validando archivo..."):
            validation = validate_model_code(code)
        
        # Mostrar resultados de validación
        if validation['valid']:
            st.success("✅ Archivo válido!")
            
            if validation['classes']:
                st.info(f"📦 Se encontraron {len(validation['classes'])} clase(s) de modelo:")
                
                for cls in validation['classes']:
                    with st.container():
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.markdown(f"""
                            **{cls['name']}**
                            - Tipo: {cls['type']}
                            - Hereda de: {', '.join(cls['bases'])}
                            """)
                        with col2:
                            st.markdown("")  # spacing
                
                # Botón para importar
                if st.button("📥 Importar al Sistema", type="primary"):
                    try:
                        filename = uploaded_file.name
                        filepath = save_model_code(code, filename)
                        
                        # Guardar metadata
                        metadata = {
                            'filename': filename,
                            'classes': validation['classes'],
                            'imported_at': datetime.now().isoformat(),
                            'source': 'upload'
                        }
                        
                        metadata_path = filepath.with_suffix('.json')
                        with open(metadata_path, 'w', encoding='utf-8') as f:
                            json.dump(metadata, f, indent=2, ensure_ascii=False)
                        
                        st.success(f"✅ Modelo importado exitosamente: `{filepath.name}`")
                        st.info("Ahora puedes usar este modelo en la página de entrenamiento")
                        
                        # Cargar clases
                        st.session_state.loaded_model_classes = get_model_class_from_file(filepath)
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"❌ Error al importar: {e}")
            else:
                st.warning("⚠️ No se encontraron clases de modelo válidas en el archivo")
        
        else:
            st.error("❌ El archivo tiene errores:")
            for error in validation['errors']:
                st.error(f"• {error}")


def model_manager_section():
    """Sección para gestionar modelos definidos."""
    st.header("📚 Modelos Disponibles")
    
    st.markdown("""
    Aquí puedes ver y gestionar todos los modelos personalizados que has definido.
    Estos modelos estarán disponibles para entrenamiento en la página **🤖 Model Training**.
    """)
    
    # Listar archivos de modelos
    model_files = list_available_model_files()
    
    if not model_files:
        st.info("📭 No hay modelos definidos aún. ¡Crea uno usando el editor o sube un archivo!")
        return
    
    st.markdown(f"**Total de archivos: {len(model_files)}**")
    
    # Mostrar cada modelo
    for model_file in model_files:
        with st.expander(f"📄 {model_file.name}", expanded=False):
            try:
                # Cargar código
                code = load_model_code(model_file)
                
                # Cargar metadata si existe
                metadata_path = model_file.with_suffix('.json')
                metadata = {}
                if metadata_path.exists():
                    with open(metadata_path, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                
                # Información
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown(f"**Archivo:** `{model_file.name}`")
                    if 'description' in metadata:
                        st.markdown(f"**Descripción:** {metadata['description']}")
                    if 'created_at' in metadata:
                        created = metadata['created_at'][:10]
                        st.markdown(f"**Creado:** {created}")
                    
                    # Clases encontradas
                    classes = get_model_class_from_file(model_file)
                    if classes:
                        st.markdown("**Clases:**")
                        for cls in classes:
                            st.markdown(f"- `{cls['name']}` ({cls['type']})")
                
                with col2:
                    st.markdown("**Acciones:**")
                    
                    view_code_btn = st.button("👁️ Ver Código", key=f"view_{model_file.name}")
                    
                    if st.button("📝 Editar", key=f"edit_{model_file.name}"):
                        st.session_state.custom_model_code = code
                        st.success("✅ Código cargado en el editor")
                        st.info("💡 Ve a la pestaña 'Editor de Código' para editarlo")
                    
                    if st.button("🗑️ Eliminar", key=f"delete_{model_file.name}"):
                        try:
                            model_file.unlink()
                            if metadata_path.exists():
                                metadata_path.unlink()
                            st.success(f"✅ Eliminado: {model_file.name}")
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Error al eliminar: {e}")
                
                # Show code if view button clicked
                if view_code_btn:
                    st.markdown("---")
                    st.markdown("### 📄 Código Completo")
                    
                    # Add line numbers
                    lines = code.split('\n')
                    numbered_code = '\n'.join([f"{i+1:4d} | {line}" for i, line in enumerate(lines)])
                    st.code(numbered_code, language='python')
                    st.caption(f"📊 Total: {len(lines)} líneas")
                
            except Exception as e:
                st.error(f"Error al cargar {model_file.name}: {e}")


def documentation_section():
    """Sección de documentación y referencia rápida."""
    st.header("📖 Documentación y Referencia")
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 Inicio Rápido",
        "📚 API Reference",
        "💡 Ejemplos",
        "🆘 Troubleshooting"
    ])
    
    with tab1:
        st.markdown("""
        ### 🚀 Inicio Rápido
        
        #### Flujo de Trabajo Básico:
        
        1. **Define tu modelo** (Editor o Upload)
           ```python
           from src.models.custom_base import BaseCustomClassifier
           
           class MiModelo(BaseCustomClassifier):
               def __init__(self, param=100):
                   super().__init__(name="MiModelo")
                   self.param = param
               
               def fit(self, X, y, **kwargs):
                   # Tu lógica de entrenamiento
                   self.is_fitted_ = True
                   return self
               
               def predict_proba(self, X):
                   # Tu lógica de predicción
                   return probas
           ```
        
        2. **Guarda el código** en el sistema
        
        3. **Ve a 🤖 Model Training**
           - Activa "Incluir Modelos Personalizados"
           - Selecciona tu modelo
           - Entrena con Cross-Validation
        
        4. **Evalúa en 📈 Model Evaluation**
           - Obtén métricas completas
           - Compara con otros modelos
        
        5. **Explica en 🔍 Explainability**
           - Genera valores SHAP
           - Analiza feature importance
        """)
    
    with tab2:
        st.markdown("""
        ### 📚 API Reference
        
        #### BaseCustomClassifier
        
        **Métodos Requeridos:**
        
        ```python
        def __init__(self, **params):
            '''Inicializar con hiperparámetros.'''
            super().__init__(name="MiModelo")
            # Guarda tus parámetros
        
        def fit(self, X, y, **kwargs):
            '''Entrenar el modelo.
            
            Args:
                X: Features (n_samples, n_features)
                y: Target (n_samples,)
            
            Returns:
                self
            '''
            self._validate_input(X, training=True)
            y = self._validate_targets(y, training=True)
            # Tu código de entrenamiento
            self.is_fitted_ = True
            return self
        
        def predict_proba(self, X):
            '''Predecir probabilidades.
            
            Args:
                X: Features (n_samples, n_features)
            
            Returns:
                np.ndarray: Probabilidades (n_samples, n_classes)
            '''
            self._validate_input(X, training=False)
            # Tu código de predicción
            return probas  # Shape: (n_samples, 2) para binario
        
        def get_params(self, deep=True):
            '''Obtener parámetros.'''
            return {'param1': self.param1, ...}
        
        def set_params(self, **params):
            '''Establecer parámetros.'''
            for key, value in params.items():
                setattr(self, key, value)
            return self
        ```
        
        **Atributos Importantes:**
        
        - `self.classes_`: Array con las clases (set en fit)
        - `self.is_fitted_`: bool indicando si está entrenado
        - `self.n_features_in_`: Número de features (set automáticamente)
        - `self.feature_names_in_`: Nombres de features (set automáticamente)
        
        **Métodos Helper:**
        
        - `_validate_input(X, training=False)`: Valida formato de entrada
        - `_validate_targets(y, training=False)`: Valida targets
        - `_convert_to_array(X)`: Convierte DataFrame a array
        """)
    
    with tab3:
        st.markdown("""
        ### 💡 Ejemplos Comunes
        
        #### 1. Threshold Personalizado
        
        ```python
        class ThresholdClassifier(BaseCustomClassifier):
            def __init__(self, threshold=0.5):
                super().__init__(name="ThresholdClassifier")
                self.threshold = threshold
                self._model = LogisticRegression()
            
            def fit(self, X, y, **kwargs):
                self._validate_input(X, training=True)
                y = self._validate_targets(y, training=True)
                X = self._convert_to_array(X)
                
                self._model.fit(X, y)
                self.is_fitted_ = True
                return self
            
            def predict_proba(self, X):
                self._validate_input(X, training=False)
                X = self._convert_to_array(X)
                return self._model.predict_proba(X)
            
            def predict(self, X):
                proba = self.predict_proba(X)
                return (proba[:, 1] >= self.threshold).astype(int)
        ```
        
        #### 2. Preprocessing Integrado
        
        ```python
        class ScaledClassifier(BaseCustomClassifier):
            def __init__(self):
                super().__init__(name="ScaledClassifier")
                self._scaler = StandardScaler()
                self._model = SVC(probability=True)
            
            def fit(self, X, y, **kwargs):
                self._validate_input(X, training=True)
                y = self._validate_targets(y, training=True)
                X = self._convert_to_array(X)
                
                X_scaled = self._scaler.fit_transform(X)
                self._model.fit(X_scaled, y)
                self.is_fitted_ = True
                return self
            
            def predict_proba(self, X):
                self._validate_input(X, training=False)
                X = self._convert_to_array(X)
                X_scaled = self._scaler.transform(X)
                return self._model.predict_proba(X_scaled)
        ```
        
        #### 3. Stacking de Modelos
        
        ```python
        class StackingClassifier(BaseCustomClassifier):
            def __init__(self):
                super().__init__(name="Stacking")
                self._base_models = [
                    RandomForestClassifier(),
                    GradientBoostingClassifier(),
                ]
                self._meta_model = LogisticRegression()
            
            def fit(self, X, y, **kwargs):
                self._validate_input(X, training=True)
                y = self._validate_targets(y, training=True)
                X = self._convert_to_array(X)
                
                # Train base models
                base_predictions = []
                for model in self._base_models:
                    model.fit(X, y)
                    base_predictions.append(model.predict_proba(X))
                
                # Stack predictions
                X_meta = np.column_stack(base_predictions)
                
                # Train meta model
                self._meta_model.fit(X_meta, y)
                self.is_fitted_ = True
                return self
            
            def predict_proba(self, X):
                self._validate_input(X, training=False)
                X = self._convert_to_array(X)
                
                base_predictions = []
                for model in self._base_models:
                    base_predictions.append(model.predict_proba(X))
                
                X_meta = np.column_stack(base_predictions)
                return self._meta_model.predict_proba(X_meta)
        ```
        """)
    
    with tab4:
        st.markdown("""
        ### 🆘 Problemas Comunes y Soluciones
        
        #### ❌ Error: "Model must be fitted before prediction"
        
        **Causa:** No se estableció `self.is_fitted_ = True` en `fit()`
        
        **Solución:**
        ```python
        def fit(self, X, y, **kwargs):
            # ... tu código de entrenamiento ...
            self.is_fitted_ = True  # ¡No olvides esto!
            return self
        ```
        
        #### ❌ Error: "X has wrong number of features"
        
        **Causa:** No se está usando `_validate_input()`
        
        **Solución:**
        ```python
        def fit(self, X, y, **kwargs):
            self._validate_input(X, training=True)  # Guarda n_features
            # ...
        
        def predict_proba(self, X):
            self._validate_input(X, training=False)  # Verifica n_features
            # ...
        ```
        
        #### ❌ Error: "predict_proba returns wrong shape"
        
        **Causa:** Para clasificación binaria, debe retornar (n_samples, 2)
        
        **Solución:**
        ```python
        def predict_proba(self, X):
            # Si tu modelo solo retorna probabilidad de clase positiva:
            proba_pos = self._model.predict_proba(X)
            proba_neg = 1 - proba_pos
            return np.column_stack([proba_neg, proba_pos])
        ```
        
        #### ❌ Error: "Missing classes_ attribute"
        
        **Causa:** No se llamó `_validate_targets()` en `fit()`
        
        **Solución:**
        ```python
        def fit(self, X, y, **kwargs):
            self._validate_input(X, training=True)
            y = self._validate_targets(y, training=True)  # Establece classes_
            # ...
        ```
        
        #### ❌ SHAP no funciona
        
        **Causa:** Modelo demasiado complejo o datos muy grandes
        
        **Solución:**
        - Reduce el tamaño de muestra para SHAP
        - Usa permutation importance como alternativa
        - Simplifica la arquitectura del modelo
        
        #### ❌ Cross-validation falla
        
        **Causa:** `get_params()` o `set_params()` no implementados correctamente
        
        **Solución:**
        ```python
        def get_params(self, deep=True):
            return {
                'param1': self.param1,
                'param2': self.param2,
                # Lista TODOS tus hiperparámetros
            }
        
        def set_params(self, **params):
            for key, value in params.items():
                setattr(self, key, value)
            # Recrea modelos internos si es necesario
            return self
        ```
        """)


# ============================================================================
# MAIN PAGE LAYOUT
# ============================================================================

def main():
    """Función principal de la página."""
    
    # Check if onboarding should be shown
    if not st.session_state.onboarding_completed or st.session_state.show_onboarding:
        show_onboarding()
        return
    
    # Main interface
    st.title("🔧 Sistema de Modelos Personalizados")
    st.markdown("""
    Crea, define y gestiona tus propios modelos de Machine Learning para integrarlos 
    con el sistema de predicción de mortalidad por IAM.
    """)
    
    # Show onboarding toggle in sidebar
    with st.sidebar:
        st.markdown("---")
        if st.button("🎓 Mostrar Tutorial", use_container_width=True):
            st.session_state.show_onboarding = True
            st.session_state.current_step = 0
            st.rerun()
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "💻 Editor de Código",
        "📁 Cargar Archivo",
        "📚 Gestionar Modelos",
        "📖 Documentación"
    ])
    
    with tab1:
        code_editor_section()
    
    with tab2:
        file_upload_section()
    
    with tab3:
        model_manager_section()
    
    with tab4:
        documentation_section()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
    <p><strong>Sistema de Modelos Personalizados</strong> | Mortality AMI Predictor</p>
    <p>💡 <em>Tip: Los modelos que definas aquí estarán disponibles en la página de entrenamiento</em></p>
    </div>
    """, unsafe_allow_html=True)


# Run main
if __name__ == "__main__":
    main()
