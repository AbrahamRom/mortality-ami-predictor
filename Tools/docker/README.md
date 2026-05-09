# Docker Setup para Mortality AMI Predictor

Este directorio contiene la configuración de Docker para ejecutar la aplicación Mortality AMI Predictor en contenedores, incluyendo soporte completo para **AutoML**.

## 📋 Requisitos Previos

- **Docker Desktop** (Windows/Mac) o **Docker Engine** (Linux)
  - Windows/Mac: [Descargar Docker Desktop](https://www.docker.com/products/docker-desktop)
  - Linux: [Instalar Docker Engine](https://docs.docker.com/engine/install/)
- **Docker Compose** (incluido en Docker Desktop, puede requerir instalación separada en Linux)

## 🤖 Soporte AutoML

La imagen de Docker incluye soporte para AutoML:

| Backend | Incluido por defecto | Plataforma | Notas |
|---------|---------------------|------------|-------|
| **FLAML** | ✅ Sí | Linux, Windows, Mac | Cross-platform, recomendado |
| **auto-sklearn** | ❌ Opcional | Solo Linux | Más completo, requiere build especial |

### Instalar con auto-sklearn (opcional)

```bash
# Build con auto-sklearn (solo funciona en contenedores Linux)
docker-compose build --build-arg INSTALL_AUTOSKLEARN=true

# O usando variable de entorno
INSTALL_AUTOSKLEARN=true docker-compose build
```

### Configurar AutoML via variables de entorno

Crear archivo `.env` en el directorio `docker/`:

```env
# Backend: flaml (default) o autosklearn
AUTOML_BACKEND=flaml

# Tiempo máximo de búsqueda en segundos (default: 3600 = 1 hora)
AUTOML_TIME_BUDGET=3600

# Métrica de optimización
AUTOML_METRIC=roc_auc

# Instalar auto-sklearn durante build
INSTALL_AUTOSKLEARN=false
```

## 🚀 Inicio Rápido


```bash
# Dar permisos de ejecución a los scripts
cd scripts
chmod +x *.sh

# Ejecutar la aplicación
./run-app.sh

# Ejecutar en modo desarrollo (con Jupyter y MLflow)
./run-dev.sh

# Detener la aplicación
./stop-app.sh

# Reconstruir la imagen
./rebuild.sh
```

## 📦 Servicios Disponibles

### Modo Producción (por defecto)
- **Dashboard Streamlit**: http://localhost:8501
  - Interfaz principal de la aplicación

### Modo Desarrollo (con `--profile dev`)
- **Dashboard Streamlit**: http://localhost:8501
- **Jupyter Lab**: http://localhost:8888
  - Para desarrollo y análisis de datos
- **MLflow UI**: http://localhost:5000
  - Para tracking de experimentos

## 🛠️ Uso Manual con Docker Compose

### Construir la imagen

```bash
cd docker
docker-compose build
```

### Iniciar la aplicación (producción)

```bash
docker-compose up -d
```

### Iniciar con servicios de desarrollo

```bash
docker-compose --profile dev up -d
```

### Ver logs

```bash
# Todos los servicios
docker-compose logs -f

# Solo la aplicación principal
docker-compose logs -f app

# Solo Jupyter
docker-compose logs -f jupyter
```

### Detener los servicios

```bash
# Detener y eliminar contenedores
docker-compose down

# Detener y eliminar contenedores + volúmenes
docker-compose down -v
```

### Reconstruir desde cero

```bash
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

## 📂 Estructura de Archivos

```
docker/
├── Dockerfile              # Imagen principal (incluye FLAML AutoML)
├── Dockerfile.jupyter      # Imagen para Jupyter Lab con AutoML
├── docker-compose.yml      # Configuración de servicios
├── .env                    # Variables de entorno (crear manualmente)
└── README.md               # Esta documentación

scripts/
├── run-app.bat/.sh        # Iniciar aplicación (Windows/Linux-Mac)
├── run-dev.bat/.sh        # Iniciar modo desarrollo
├── stop-app.bat/.sh       # Detener aplicación
└── rebuild.bat/.sh        # Reconstruir imagen
```

## 💾 Volúmenes y Persistencia

Los siguientes directorios se montan como volúmenes para persistir datos:

- `DATA/` → Datos de entrada (solo lectura)
- `processed/` → Datos procesados
- `models/` → Modelos entrenados
- `models/automl/` → Modelos AutoML exportados (volumen Docker)
- `mlruns/` → Experimentos de MLflow
- `logs/` → Logs de la aplicación

## 🤖 Uso de AutoML en Docker

### Desde el Dashboard

1. Accede a http://localhost:8501
2. Ve a la página **🤖 AutoML**
3. Selecciona un preset (quick, balanced, high_performance)
4. Inicia el entrenamiento

### Desde Jupyter

```python
from src.automl import FLAMLClassifier, is_flaml_available

# Verificar disponibilidad
print(f"FLAML disponible: {is_flaml_available()}")

# Entrenar modelo AutoML
clf = FLAMLClassifier(time_budget=300, metric="roc_auc")
clf.fit(X_train, y_train)
```

### Variables de entorno para AutoML

| Variable | Descripción | Default |
|----------|-------------|---------|
| `AUTOML_BACKEND` | Backend a usar: `flaml` o `autosklearn` | `flaml` |
| `AUTOML_TIME_BUDGET` | Tiempo máximo en segundos | `3600` |
| `AUTOML_METRIC` | Métrica de optimización | `roc_auc` |

## 🔧 Personalización

### Cambiar el puerto del Dashboard

Editar `docker-compose.yml`:

```yaml
services:
  app:
    ports:
      - "8080:8501"  # Cambiar 8080 por el puerto deseado
```

### Agregar variables de entorno

Editar `docker-compose.yml`:

```yaml
services:
  app:
    environment:
      - MI_VARIABLE=valor
      - OTRA_VARIABLE=otro_valor
```

### Usar archivo de variables de entorno

Crear archivo `.env` en el directorio `docker/`:

```env
STREAMLIT_PORT=8501
JUPYTER_PORT=8888
MLFLOW_PORT=5000
```

Y referenciar en `docker-compose.yml`:

```yaml
services:
  app:
    env_file:
      - .env
```

## 🐛 Troubleshooting

### Error: "Cannot connect to Docker daemon"

**Solución**: Asegúrate de que Docker Desktop esté corriendo (Windows/Mac) o que el servicio de Docker esté activo (Linux):

```bash
# Linux
sudo systemctl start docker

# Verificar estado
docker info
```

### Error: "Port already in use"

**Solución**: Otro servicio está usando el puerto. Detén el servicio o cambia el puerto en `docker-compose.yml`.

```bash
# Ver qué está usando el puerto 8501
# Windows
netstat -ano | findstr :8501

# Linux/Mac
lsof -i :8501
```

### La aplicación no se inicia

**Solución**: Ver los logs para diagnosticar:

```bash
docker-compose logs app
```

### Reconstruir completamente

Si hay problemas persistentes:

```bash
# Detener todo
docker-compose down -v

# Limpiar imágenes
docker system prune -a

# Reconstruir
docker-compose build --no-cache
docker-compose up -d
```

## 📚 Recursos Adicionales

- [Documentación de Docker](https://docs.docker.com/)
- [Documentación de Docker Compose](https://docs.docker.com/compose/)
- [Documentación de Streamlit](https://docs.streamlit.io/)
- [Best Practices para Dockerfile](https://docs.docker.com/develop/develop-images/dockerfile_best-practices/)

## 🔒 Seguridad

⚠️ **Nota**: Esta configuración es para desarrollo/demostración. Para producción, considera:

- Usar secretos de Docker para credenciales
- Configurar HTTPS con certificados
- Implementar autenticación en Streamlit
- Usar redes Docker para aislar servicios
- Escanear imágenes para vulnerabilidades
- No exponer Jupyter sin autenticación

## 📝 Notas

- La imagen de Docker se optimiza para tamaño usando Python slim
- Los datos se montan como volúmenes para evitar reconstruir la imagen
- El modo desarrollo incluye herramientas adicionales para análisis
- Los logs se pueden ver en tiempo real con `docker-compose logs -f`
