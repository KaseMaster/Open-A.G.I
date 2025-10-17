#!/bin/bash
# ===== ROLLBACK SCRIPT =====
# Script para rollback de deployments fallidos

set -e

ENVIRONMENT=${1:-"staging"}
ROLLBACK_VERSION=${2:-"previous"}

echo "🔄 Iniciando rollback de AEGIS en $ENVIRONMENT..."

# Verificar que el entorno existe
if [ ! -f "docker-compose.$ENVIRONMENT.yml" ]; then
    echo "❌ Configuración de entorno $ENVIRONMENT no encontrada"
    exit 1
fi

# Detener servicios actuales
echo "🛑 Deteniendo servicios actuales..."
docker-compose -f docker-compose.yml -f "docker-compose.$ENVIRONMENT.yml" down

# Buscar backup más reciente
echo "📦 Buscando backup más reciente..."
LATEST_BACKUP=$(ls -t backups/ | head -1)

if [ -z "$LATEST_BACKUP" ]; then
    echo "❌ No se encontraron backups disponibles"
    exit 1
fi

echo "📦 Usando backup: $LATEST_BACKUP"

# Restaurar configuración
if [ -d "backups/$LATEST_BACKUP/config" ]; then
    echo "⚙️ Restaurando configuración..."
    cp -r "backups/$LATEST_BACKUP/config"/* config/ 2>/dev/null || true
fi

# Restaurar base de datos si existe backup
DB_BACKUP=$(ls -t backups/db-*.sql 2>/dev/null | head -1)
if [ -n "$DB_BACKUP" ] && [ "$ENVIRONMENT" = "production" ]; then
    echo "🗄️ Restaurando base de datos..."
    # Aquí iría la lógica de restauración de DB
    echo "⚠️ Restauración de DB requiere intervención manual"
fi

# Reiniciar con versión anterior
PREVIOUS_VERSION=$(docker images kasemaster/aegis-framework --format "{{.Tag}}" | grep -v latest | head -1)

if [ -n "$PREVIOUS_VERSION" ]; then
    echo "🐳 Usando versión anterior: $PREVIOUS_VERSION"
    sed -i "s|DOCKER_IMAGE=.*|DOCKER_IMAGE=kasemaster/aegis-framework:$PREVIOUS_VERSION|" ".env.$ENVIRONMENT"
else
    echo "⚠️ No se encontró versión anterior, usando latest"
fi

# Reiniciar servicios
echo "🚀 Reiniciando servicios..."
docker-compose -f docker-compose.yml -f "docker-compose.$ENVIRONMENT.yml" up -d

# Verificar que el rollback fue exitoso
echo "🔍 Verificando rollback..."
sleep 30

if docker-compose -f docker-compose.yml -f "docker-compose.$ENVIRONMENT.yml" ps | grep -q "Up"; then
    echo "✅ Rollback completado exitosamente"
    echo ""
    echo "📊 Resumen del rollback:"
    echo "   • Backup usado: $LATEST_BACKUP"
    echo "   • Versión restaurada: ${PREVIOUS_VERSION:-latest}"
    echo "   • Timestamp: $(date)"
    echo ""
    echo "🔗 Verificar servicios:"
    echo "   docker-compose -f docker-compose.yml -f docker-compose.$ENVIRONMENT.yml ps"
    echo "   docker-compose -f docker-compose.yml -f docker-compose.$ENVIRONMENT.yml logs -f"
else
    echo "❌ Rollback falló"
    docker-compose -f docker-compose.yml -f "docker-compose.$ENVIRONMENT.yml" logs
    exit 1
fi
