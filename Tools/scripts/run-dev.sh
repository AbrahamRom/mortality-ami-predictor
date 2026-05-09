#!/bin/bash
################################################################################
# Script para ejecutar el entorno de desarrollo completo
# Sistema: Linux/Mac
################################################################################

set -e

echo "========================================"
echo "Mortality AMI Predictor - Dev Mode"
echo "========================================"
echo ""

# Determinar comando de Docker Compose
DOCKER_COMPOSE="docker compose"
if ! docker compose version &> /dev/null; then
    DOCKER_COMPOSE="docker-compose"
fi

# Cambiar al directorio docker
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../docker"

echo "[INFO] Iniciando entorno de desarrollo..."
echo ""
echo "Esto incluye:"
echo "  - Dashboard (puerto 8501)"
echo "  - Jupyter Lab (puerto 8888)"
echo "  - MLflow UI (puerto 5000)"
echo ""

$DOCKER_COMPOSE --profile dev up -d

echo ""
echo "========================================"
echo "[SUCCESS] Entorno de desarrollo iniciado!"
echo "========================================"
echo ""
echo "Servicios disponibles:"
echo "  Dashboard:    http://localhost:8501"
echo "  Jupyter Lab:  http://localhost:8888"
echo "  MLflow UI:    http://localhost:5000"
echo ""
echo "Para detener todo:"
echo "  $DOCKER_COMPOSE --profile dev down"
echo ""
