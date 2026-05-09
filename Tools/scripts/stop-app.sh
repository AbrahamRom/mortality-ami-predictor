#!/bin/bash
################################################################################
# Script para detener la aplicación
# Sistema: Linux/Mac
################################################################################

set -e

echo "========================================"
echo "Mortality AMI Predictor - Stop"
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

echo "[INFO] Deteniendo contenedores..."
$DOCKER_COMPOSE down

echo ""
echo "[SUCCESS] Contenedores detenidos exitosamente"
echo ""
