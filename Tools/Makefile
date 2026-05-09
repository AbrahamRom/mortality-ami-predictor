# Makefile para Mortality AMI Predictor
# Facilita la ejecución de comandos comunes de Docker

.PHONY: help build up down dev logs clean rebuild restart status

# Variables
DOCKER_COMPOSE = docker-compose
DOCKER_COMPOSE_FILE = docker/docker-compose.yml

# Color output
RED = \033[0;31m
GREEN = \033[0;32m
YELLOW = \033[1;33m
NC = \033[0m # No Color

help: ## Mostrar esta ayuda
	@echo "$(GREEN)Mortality AMI Predictor - Docker Commands$(NC)"
	@echo ""
	@echo "$(YELLOW)Comandos disponibles:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-15s$(NC) %s\n", $$1, $$2}'
	@echo ""

build: ## Construir las imágenes Docker
	@echo "$(GREEN)Construyendo imágenes...$(NC)"
	cd docker && $(DOCKER_COMPOSE) build

up: ## Iniciar la aplicación (modo producción)
	@echo "$(GREEN)Iniciando aplicación...$(NC)"
	cd docker && $(DOCKER_COMPOSE) up -d
	@echo "$(GREEN)✓ Aplicación iniciada en http://localhost:8501$(NC)"

down: ## Detener la aplicación
	@echo "$(YELLOW)Deteniendo aplicación...$(NC)"
	cd docker && $(DOCKER_COMPOSE) down
	@echo "$(GREEN)✓ Aplicación detenida$(NC)"

dev: ## Iniciar en modo desarrollo (con Jupyter y MLflow)
	@echo "$(GREEN)Iniciando modo desarrollo...$(NC)"
	cd docker && $(DOCKER_COMPOSE) --profile dev up -d
	@echo "$(GREEN)✓ Servicios iniciados:$(NC)"
	@echo "  Dashboard:    http://localhost:8501"
	@echo "  Jupyter Lab:  http://localhost:8888"
	@echo "  MLflow UI:    http://localhost:5000"

logs: ## Ver logs en tiempo real
	cd docker && $(DOCKER_COMPOSE) logs -f

logs-app: ## Ver logs solo de la aplicación
	cd docker && $(DOCKER_COMPOSE) logs -f app

clean: ## Limpiar contenedores, imágenes y volúmenes
	@echo "$(RED)Limpiando Docker (contenedores, imágenes, volúmenes)...$(NC)"
	cd docker && $(DOCKER_COMPOSE) down -v
	@echo "$(GREEN)✓ Limpieza completada$(NC)"

rebuild: ## Reconstruir imágenes sin caché
	@echo "$(YELLOW)Reconstruyendo imágenes (sin caché)...$(NC)"
	cd docker && $(DOCKER_COMPOSE) down
	cd docker && $(DOCKER_COMPOSE) build --no-cache
	@echo "$(GREEN)✓ Imágenes reconstruidas$(NC)"

restart: ## Reiniciar la aplicación
	@echo "$(YELLOW)Reiniciando aplicación...$(NC)"
	cd docker && $(DOCKER_COMPOSE) restart
	@echo "$(GREEN)✓ Aplicación reiniciada$(NC)"

status: ## Ver estado de los contenedores
	@echo "$(GREEN)Estado de contenedores:$(NC)"
	cd docker && $(DOCKER_COMPOSE) ps

shell: ## Abrir shell en el contenedor de la aplicación
	@echo "$(GREEN)Abriendo shell en contenedor...$(NC)"
	cd docker && $(DOCKER_COMPOSE) exec app /bin/bash

test: ## Ejecutar tests dentro del contenedor
	@echo "$(GREEN)Ejecutando tests...$(NC)"
	cd docker && $(DOCKER_COMPOSE) exec app pytest tests/

install: ## Instalar dependencias adicionales en el contenedor
	@echo "$(YELLOW)Reconstruyendo con nuevas dependencias...$(NC)"
	$(MAKE) rebuild
	$(MAKE) up

prune: ## Limpiar sistema Docker (cuidado: afecta todo Docker)
	@echo "$(RED)⚠️  Esto limpiará TODOS los recursos Docker no utilizados$(NC)"
	@read -p "¿Continuar? (y/N): " confirm && [ "$$confirm" = "y" ] || exit 1
	docker system prune -a --volumes
	@echo "$(GREEN)✓ Sistema Docker limpiado$(NC)"

# Comandos de acceso rápido
jupyter: ## Abrir Jupyter Lab
	@echo "$(GREEN)Abriendo Jupyter Lab en http://localhost:8888$(NC)"
	@command -v xdg-open > /dev/null && xdg-open http://localhost:8888 || \
	 command -v open > /dev/null && open http://localhost:8888 || \
	 echo "Abre manualmente: http://localhost:8888"

dashboard: ## Abrir Dashboard
	@echo "$(GREEN)Abriendo Dashboard en http://localhost:8501$(NC)"
	@command -v xdg-open > /dev/null && xdg-open http://localhost:8501 || \
	 command -v open > /dev/null && open http://localhost:8501 || \
	 echo "Abre manualmente: http://localhost:8501"

mlflow: ## Abrir MLflow UI
	@echo "$(GREEN)Abriendo MLflow UI en http://localhost:5000$(NC)"
	@command -v xdg-open > /dev/null && xdg-open http://localhost:5000 || \
	 command -v open > /dev/null && open http://localhost:5000 || \
	 echo "Abre manualmente: http://localhost:5000"
