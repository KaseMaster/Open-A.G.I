#!/bin/bash
# Script para iniciar el stack de monitoreo

echo "🚀 Iniciando AEGIS Monitoring Stack..."
echo ""

# Verificar Docker
if ! command -v docker &> /dev/null; then
    echo "❌ Docker no está instalado"
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose no está instalado"
    exit 1
fi

echo "✓ Docker y Docker Compose disponibles"
echo ""

# Detener contenedores existentes si existen
echo "🛑 Deteniendo contenedores existentes..."
docker-compose -f docker-compose.monitoring.yml down 2>/dev/null || true
echo ""

# Iniciar stack
echo "🔧 Iniciando servicios de monitoreo..."
docker-compose -f docker-compose.monitoring.yml up -d

# Esperar a que los servicios estén listos
echo ""
echo "⏳ Esperando a que los servicios inicien..."
sleep 10

# Verificar estado
echo ""
echo "📊 Estado de los servicios:"
docker-compose -f docker-compose.monitoring.yml ps

echo ""
echo "✅ Stack de monitoreo iniciado"
echo ""
echo "🌐 Acceso a los servicios:"
echo "   Prometheus:    http://localhost:9090"
echo "   Grafana:       http://localhost:3000 (admin/admin)"
echo "   Node Exporter: http://localhost:9100/metrics"
echo ""
echo "📝 Para detener el stack:"
echo "   docker-compose -f docker-compose.monitoring.yml down"
echo ""
echo "📊 Para ver logs:"
echo "   docker-compose -f docker-compose.monitoring.yml logs -f"
echo ""
