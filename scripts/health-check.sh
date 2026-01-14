#!/bin/bash
# ===== HEALTH CHECK SCRIPT =====
# Verifica el estado de salud del sistema AEGIS

set -e

echo "🏥 Iniciando health check de AEGIS..."

# Verificar que Python esté disponible
if ! command -v python &> /dev/null; then
    echo "❌ Python no encontrado"
    exit 1
fi

# Verificar que las dependencias estén instaladas
echo "📦 Verificando dependencias..."
python -c "import aegis; print('✅ AEGIS importable')" || {
    echo "❌ No se puede importar AEGIS"
    exit 1
}

# Verificar que Poetry esté disponible (opcional)
if command -v poetry &> /dev/null; then
    echo "📝 Verificando configuración de Poetry..."
    poetry check || {
        echo "⚠️ Configuración de Poetry inválida"
    }
fi

# Verificar archivos de configuración
echo "⚙️ Verificando archivos de configuración..."
CONFIG_FILES=(
    "pyproject.toml"
    ".github/workflows/ci-cd.yml"
    "Dockerfile.ci"
    "docker-compose.ci.yml"
)

for config_file in "${CONFIG_FILES[@]}"; do
    if [ -f "$config_file" ]; then
        echo "✅ $config_file encontrado"
    else
        echo "⚠️ $config_file no encontrado"
    fi
done

# Verificar que los tests básicos funcionen
echo "🧪 Ejecutando tests básicos..."
python -c "
import asyncio
from crypto_framework import initialize_crypto

# Test básico de inicialización
try:
    crypto = initialize_crypto({'security_level': 'HIGH', 'node_id': 'health_check'})
    print('✅ Crypto engine inicializado correctamente')
except Exception as e:
    print(f'❌ Error inicializando crypto: {e}')
    exit(1)
"

# Verificar conectividad de red (básica)
echo "🌐 Verificando conectividad de red..."
if curl -f -s --max-time 5 http://httpbin.org/ip > /dev/null 2>&1; then
    echo "✅ Conectividad de red OK"
else
    echo "⚠️ Problemas de conectividad de red"
fi

# Verificar espacio en disco
echo "💾 Verificando espacio en disco..."
DISK_USAGE=$(df / | tail -1 | awk '{print $5}' | sed 's/%//')
if [ "$DISK_USAGE" -lt 90 ]; then
    echo "✅ Espacio en disco OK (${DISK_USAGE}%)"
else
    echo "⚠️ Espacio en disco bajo (${DISK_USAGE}%)"
fi

# Verificar memoria disponible
echo "🧠 Verificando memoria..."
MEM_AVAILABLE=$(free | grep Mem | awk '{printf "%.0f", $7/1024/1024}')
if [ "$MEM_AVAILABLE" -gt 1 ]; then
    echo "✅ Memoria disponible OK (${MEM_AVAILABLE}GB)"
else
    echo "⚠️ Memoria baja (${MEM_AVAILABLE}GB)"
fi

echo ""
echo "🎉 Health check completado exitosamente!"
echo ""
echo "📊 Resumen del sistema:"
echo "   • Python: $(python --version)"
echo "   • Sistema: $(uname -a)"
echo "   • Usuario: $(whoami)"
echo "   • Directorio: $(pwd)"
echo ""
echo "🚀 Sistema listo para despliegue"
