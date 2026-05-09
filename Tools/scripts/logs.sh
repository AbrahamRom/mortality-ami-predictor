#!/bin/bash
################################################################################
# Script para ver logs de los contenedores
# Sistema: Linux/Mac
################################################################################

# Determinar comando de Docker Compose
DOCKER_COMPOSE="docker compose"
if ! docker compose version &> /dev/null; then
    DOCKER_COMPOSE="docker-compose"
fi

# Cambiar al directorio docker
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../docker"

echo "========================================"
echo "Mortality AMI Predictor - Logs"
echo "========================================"
echo ""
echo "Mostrando logs en tiempo real..."
echo "Presiona Ctrl+C para salir"
echo ""

$DOCKER_COMPOSE logs -f
