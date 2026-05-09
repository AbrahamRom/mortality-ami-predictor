#!/bin/bash
################################################################################
# Script para reconstruir la imagen Docker
# Sistema: Linux/Mac
################################################################################

set -e

echo "========================================"
echo "Mortality AMI Predictor - Rebuild"
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

echo "[INFO] Deteniendo contenedores existentes..."
$DOCKER_COMPOSE down

echo ""
echo "[INFO] Reconstruyendo la imagen (sin caché)..."
$DOCKER_COMPOSE build --no-cache

echo ""
echo "[SUCCESS] Imagen reconstruida exitosamente"
echo ""
echo "Para iniciar la aplicación, ejecuta:"
echo "  ./run-app.sh"
echo ""
