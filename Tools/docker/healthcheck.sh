#!/bin/sh
################################################################################
# Health check script para el contenedor de la aplicación
################################################################################

# Verificar que Streamlit esté respondiendo
curl -f http://localhost:8501/_stcore/health || exit 1

exit 0
