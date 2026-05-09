# Scripts de Ejecución - Mortality AMI Predictor

Esta carpeta contiene scripts de automatización para ejecutar la aplicación en diferentes sistemas operativos.

## 📁 Contenido



### Scripts de Bash (.sh)

- **`run-app.sh`** - Construir e iniciar la aplicación en modo producción
- **`run-dev.sh`** - Iniciar la aplicación en modo desarrollo (incluye Jupyter y MLflow)
- **`stop-app.sh`** - Detener todos los contenedores
- **`rebuild.sh`** - Reconstruir la imagen Docker desde cero

## 🚀 Uso


```bash
# Navegar a la carpeta scripts
cd Tools/scripts

# Dar permisos de ejecución (solo la primera vez)
chmod +x *.sh

# Ejecutar el script deseado
./run-app.sh
```

## 📋 Descripción de Scripts

### 1. run-app (Modo Producción)

**Qué hace:**
- Verifica que Docker esté instalado y corriendo
- Construye la imagen Docker si es necesario
- Inicia el contenedor de la aplicación
- Expone el dashboard en http://localhost:8501

**Cuándo usar:**
- Para ejecutar solo la aplicación principal
- En entornos de producción o demostración
- Cuando no necesitas Jupyter o MLflow

### 2. run-dev (Modo Desarrollo)

**Qué hace:**
- Inicia todos los servicios de desarrollo
- Dashboard (puerto 8501)
- Jupyter Lab (puerto 8888)
- MLflow UI (puerto 5000)

**Cuándo usar:**
- Durante el desarrollo
- Para análisis de datos con Jupyter
- Para hacer tracking de experimentos con MLflow
- Cuando necesitas todas las herramientas

### 3. stop-app

**Qué hace:**
- Detiene todos los contenedores Docker
- Limpia los recursos

**Cuándo usar:**
- Cuando terminas de usar la aplicación
- Para liberar recursos del sistema
- Antes de reconstruir

### 4. rebuild

**Qué hace:**
- Detiene todos los contenedores
- Reconstruye la imagen Docker sin usar caché
- Útil cuando hay cambios en dependencias

**Cuándo usar:**
- Después de actualizar requirements.txt
- Cuando hay problemas con la imagen actual
- Para asegurar una construcción limpia

## 🔧 Requisitos

- **Docker Desktop** (Windows/Mac) o **Docker Engine** (Linux)
- Permisos de ejecución en scripts .sh (Linux/Mac)

## ⚙️ Configuración

Los scripts usan configuraciones por defecto del archivo `docker-compose.yml`. Para personalizar:

1. **Cambiar puertos**: Editar `docker/docker-compose.yml`
2. **Variables de entorno**: Editar `docker/docker-compose.yml` o crear `.env`
3. **Volúmenes**: Editar `docker/docker-compose.yml`

## 🐛 Solución de Problemas

### Windows: "Docker no está instalado"

Instalar Docker Desktop desde: https://www.docker.com/products/docker-desktop

### Linux/Mac: "Permission denied"

Dar permisos de ejecución:

```bash
chmod +x *.sh
```

### "Port already in use"

Otro servicio está usando el puerto. Opciones:
1. Detener el servicio que usa el puerto
2. Cambiar el puerto en `docker-compose.yml`

### Docker no está corriendo

**Windows/Mac**: Iniciar Docker Desktop
**Linux**: 
```bash
sudo systemctl start docker
```

## 📊 Logs y Monitoreo

Ver logs de la aplicación:

```bash
# Desde el directorio docker
cd ../docker

# Ver logs de todos los servicios
docker-compose logs -f

# Ver logs de un servicio específico
docker-compose logs -f app
```

## 🔄 Workflow Típico

### Desarrollo diario

```bash
# Iniciar en modo desarrollo
./run-dev.sh

# Trabajar en la aplicación...

# Detener al finalizar
./stop-app.sh
```

### Actualizar dependencias

```bash
# Editar requirements.txt
vim ../requirements.txt

# Reconstruir imagen
./rebuild.sh

# Iniciar aplicación
./run-app.sh
```

### Demo/Producción

```bash
# Iniciar solo la aplicación
./run-app.sh

# La aplicación está en http://localhost:8501
```

## 📝 Notas

- Todos los scripts validan que Docker esté disponible
- Los scripts soportan tanto Docker Compose v1 como v2

## 🔗 Enlaces Útiles

- [Docker Documentation](https://docs.docker.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Jupyter Lab Documentation](https://jupyterlab.readthedocs.io/)

## 💡 Tips

1. **Primera ejecución**: Usa `run-app` o `run-dev` - construirá todo automáticamente
2. **Cambios en código**: No necesitas reconstruir, los volúmenes reflejan cambios
3. **Cambios en dependencias**: Usa `rebuild` para reconstruir la imagen
4. **Problemas**: Revisa logs con `docker-compose logs -f`
5. **Limpieza total**: `docker-compose down -v` elimina volúmenes también
