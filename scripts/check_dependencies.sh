#!/bin/bash

echo "🔍 Analizando dependencias críticas del proyecto AEGIS..."
echo ""

VENV_PYTHON="/home/kasemaster/Escritorio/Proyectos/Open-A.G.I/.venv/bin/python3"
SYSTEM_PYTHON="python3"

check_package() {
    local package=$1
    local priority=$2
    
    if $SYSTEM_PYTHON -c "import $package" 2>/dev/null; then
        echo "✓ $package ($priority)"
        return 0
    else
        echo "✗ $package ($priority) - FALTANTE"
        return 1
    fi
}

echo "📦 DEPENDENCIAS CORE (Alta Prioridad)"
echo "======================================"
check_package "cryptography" "HIGH"
check_package "aiohttp" "HIGH"
check_package "websockets" "HIGH"
check_package "pydantic" "HIGH"
check_package "dotenv" "HIGH"
check_package "click" "HIGH"
check_package "rich" "MEDIUM"

echo ""
echo "🧪 DEPENDENCIAS DE TESTING (Alta Prioridad)"
echo "============================================"
check_package "pytest" "HIGH"
check_package "pytest_asyncio" "HIGH"
check_package "pytest_cov" "MEDIUM"

echo ""
echo "🗄️  DEPENDENCIAS DE ALMACENAMIENTO (Media Prioridad)"
echo "====================================================="
check_package "aiosqlite" "MEDIUM"
check_package "redis" "MEDIUM"

echo ""
echo "🧠 DEPENDENCIAS DE ML (Media-Baja Prioridad)"
echo "============================================="
check_package "numpy" "MEDIUM"
check_package "sklearn" "MEDIUM"
check_package "torch" "LOW"

echo ""
echo "📊 DEPENDENCIAS DE MONITOREO (Media Prioridad)"
echo "==============================================="
check_package "loguru" "MEDIUM"
check_package "prometheus_client" "MEDIUM"
check_package "psutil" "MEDIUM"
check_package "flask" "MEDIUM"

echo ""
echo "🌐 DEPENDENCIAS DE NETWORKING (Alta Prioridad)"
echo "==============================================="
check_package "zeroconf" "MEDIUM"
check_package "netifaces" "MEDIUM"

echo ""
echo "📝 RESUMEN"
echo "=========="
echo "Las dependencias marcadas con ✗ necesitan ser instaladas"
echo "Prioridad HIGH: Críticas para el funcionamiento"
echo "Prioridad MEDIUM: Importantes pero no bloqueantes"
echo "Prioridad LOW: Opcionales para funcionalidades avanzadas"
