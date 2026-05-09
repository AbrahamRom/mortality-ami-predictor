# Dockerfile para Mortality AMI Predictor
# Incluye soporte para AutoML (FLAML y opcionalmente auto-sklearn)
FROM python:3.11-slim

# Establecer variables de entorno
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # AutoML configuration
    AUTOML_BACKEND=flaml \
    AUTOML_TIME_BUDGET=3600

# Argumentos de build para configuración opcional
ARG INSTALL_AUTOSKLEARN=false
ARG AUTOML_ONLY=false

# Crear directorio de trabajo
WORKDIR /app

# Instalar dependencias del sistema
# Incluye dependencias adicionales para AutoML y compilación
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    curl \
    # Dependencias para compilación de paquetes científicos
    libgomp1 \
    libopenblas-dev \
    # Dependencias opcionales para auto-sklearn (si se instala)
    swig \
    && rm -rf /var/lib/apt/lists/*

# Copiar archivos de requisitos
COPY requirements.txt .

# Instalar dependencias de Python base
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Instalar FLAML para AutoML (cross-platform, siempre incluido)
RUN pip install "flaml[automl]" && \
    echo "✅ FLAML AutoML installed successfully"

# Instalar auto-sklearn opcionalmente (solo funciona en Linux)
# Usar: docker build --build-arg INSTALL_AUTOSKLEARN=true ...
RUN if [ "$INSTALL_AUTOSKLEARN" = "true" ]; then \
        echo "📦 Installing auto-sklearn (this may take a while)..." && \
        pip install pyrfr ConfigSpace smac auto-sklearn && \
        echo "✅ auto-sklearn installed successfully"; \
    else \
        echo "ℹ️ Skipping auto-sklearn (use --build-arg INSTALL_AUTOSKLEARN=true to install)"; \
    fi

# Copiar el código de la aplicación
COPY src/ ./src/
COPY dashboard/ ./dashboard/
COPY notebooks/ ./notebooks/
COPY config.py .
COPY __init__.py .

# Crear directorios necesarios
RUN mkdir -p processed/models processed/plots processed/cleaned_datasets \
    models mlruns logs

# Copiar y dar permisos al script de healthcheck
COPY docker/healthcheck.sh /healthcheck.sh
RUN chmod +x /healthcheck.sh

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD /healthcheck.sh

# Exponer puerto para Streamlit
EXPOSE 8501

# Comando por defecto: ejecutar dashboard
CMD ["streamlit", "run", "dashboard/Dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]
