#!/bin/bash
################################################################################
# Script para construir y ejecutar la aplicación Mortality AMI Predictor
# Sistema: Linux/Mac
################################################################################

set -e

echo "========================================"
echo "Mortality AMI Predictor - Setup"
echo "========================================"
echo ""

# Verificar si Docker está instalado
if ! command -v docker &> /dev/null; then
    echo "[ERROR] Docker no está instalado"
    echo "Por favor, instala Docker desde: https://www.docker.com/get-started"
    exit 1
fi

# Verificar si Docker Compose está disponible
if ! docker compose version &> /dev/null && ! command -v docker-compose &> /dev/null; then
    echo "[ERROR] Docker Compose no está instalado"
    echo "Por favor, instala Docker Compose"
    exit 1
fi

# Usar 'docker compose' (v2) o 'docker-compose' (v1)
DOCKER_COMPOSE="docker compose"
if ! docker compose version &> /dev/null; then
    DOCKER_COMPOSE="docker-compose"
fi

echo "[INFO] Docker está instalado y disponible"
echo ""

# Cambiar al directorio docker
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../docker"

echo "[INFO] Construyendo la imagen Docker..."
$DOCKER_COMPOSE build

echo ""
echo "[SUCCESS] Imagen construida exitosamente"
echo ""
echo "[INFO] Iniciando los contenedores..."
$DOCKER_COMPOSE up -d

echo ""
echo "========================================"
echo "[SUCCESS] Aplicación iniciada exitosamente!"
echo "========================================"
echo ""
echo "La aplicación está disponible en:"
echo "  Dashboard: http://localhost:8501"
echo ""
echo "Para ver los logs:"
echo "  $DOCKER_COMPOSE logs -f"
echo ""
echo "Para detener la aplicación:"
echo "  $DOCKER_COMPOSE down"
echo ""
