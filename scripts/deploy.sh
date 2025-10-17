#!/bin/bash
# ===== DEPLOYMENT SCRIPT =====
# Script automatizado para deployment de AEGIS

set -e

# Configuración
ENVIRONMENT=${1:-"staging"}
VERSION=${2:-"latest"}
DOCKER_REGISTRY=${DOCKER_REGISTRY:-"kasemaster/aegis-framework"}

echo "🚀 Iniciando deployment de AEGIS v$VERSION a $ENVIRONMENT..."

# Verificar prerrequisitos
echo "📋 Verificando prerrequisitos..."

if ! command -v docker &> /dev/null; then
    echo "❌ Docker no encontrado"
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose no encontrado"
    exit 1
fi

# Ejecutar health check
echo "🏥 Ejecutando health check..."
if [ -f "scripts/health-check.sh" ]; then
    bash scripts/health-check.sh
else
    echo "⚠️ Script de health check no encontrado, continuando..."
fi

# Login a registry si es producción
if [ "$ENVIRONMENT" = "production" ] && [ -n "$DOCKERHUB_USERNAME" ]; then
    echo "🔐 Login a Docker Hub..."
    echo "$DOCKERHUB_TOKEN" | docker login -u "$DOCKERHUB_USERNAME" --password-stdin
fi

# Crear directorios necesarios
echo "📁 Creando directorios..."
mkdir -p logs
mkdir -p data
mkdir -p backups

# Backup de configuración actual (si existe)
if [ -d "config" ]; then
    echo "💾 Creando backup de configuración..."
    cp -r config "backups/config-$(date +%Y%m%d-%H%M%S)"
fi

# Descargar nueva imagen
echo "🐳 Descargando imagen Docker..."
docker pull "$DOCKER_REGISTRY:$VERSION"

# Taggear como latest para el entorno
docker tag "$DOCKER_REGISTRY:$VERSION" "aegis-$ENVIRONMENT:latest"

# Crear archivo de configuración del entorno
echo "⚙️ Configurando entorno $ENVIRONMENT..."
cat > ".env.$ENVIRONMENT" << EOF
# AEGIS Configuration for $ENVIRONMENT
AEGIS_ENV=$ENVIRONMENT
NODE_ID=node-$ENVIRONMENT-$(hostname)
LOG_LEVEL=INFO
DATABASE_URL=postgresql://aegis:password@db:5432/aegis_$ENVIRONMENT
REDIS_URL=redis://redis:6379
HEALTH_CHECK_INTERVAL=30
MAX_PEER_CONNECTIONS=50
HEARTBEAT_INTERVAL_SEC=30

# Security settings
SECURITY_LEVEL=HIGH
KEY_ROTATION_INTERVAL_HOURS=24
MAX_MESSAGE_AGE_SECONDS=300

# Docker settings
DOCKER_IMAGE=$DOCKER_REGISTRY:$VERSION
EOF

# Crear docker-compose override para el entorno
cat > "docker-compose.$ENVIRONMENT.yml" << EOF
version: '3.8'

services:
  aegis-node:
    image: aegis-$ENVIRONMENT:latest
    env_file:
      - .env.$ENVIRONMENT
    environment:
      - AEGIS_ENV=$ENVIRONMENT
    ports:
      - "${AEGIS_PORT:-8080}:8080"
    volumes:
      - ./logs:/app/logs
      - ./data:/app/data
    depends_on:
      - db
      - redis
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
    restart: unless-stopped

  db:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: aegis_$ENVIRONMENT
      POSTGRES_USER: aegis
      POSTGRES_PASSWORD: password
    volumes:
      - db_data_$ENVIRONMENT:/var/lib/postgresql/data
    ports:
      - "${DB_PORT:-5432}:5432"
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U aegis -d aegis_$ENVIRONMENT"]
      interval: 10s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    ports:
      - "${REDIS_PORT:-6379}:6379"
    volumes:
      - redis_data_$ENVIRONMENT:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 3s
      retries: 3

volumes:
  db_data_$ENVIRONMENT:
  redis_data_$ENVIRONMENT:
EOF

# Detener servicios existentes si están corriendo
echo "🛑 Deteniendo servicios existentes..."
docker-compose -f docker-compose.yml -f "docker-compose.$ENVIRONMENT.yml" down || true

# Limpiar contenedores e imágenes no utilizadas
echo "🧹 Limpiando recursos antiguos..."
docker system prune -f

# Iniciar servicios nuevos
echo "🚀 Iniciando servicios en $ENVIRONMENT..."
docker-compose -f docker-compose.yml -f "docker-compose.$ENVIRONMENT.yml" up -d

# Esperar a que los servicios estén healthy
echo "⏳ Esperando que los servicios estén listos..."
sleep 30

# Verificar estado de los servicios
echo "🔍 Verificando estado de los servicios..."
if docker-compose -f docker-compose.yml -f "docker-compose.$ENVIRONMENT.yml" ps | grep -q "Up"; then
    echo "✅ Servicios iniciados correctamente"
else
    echo "❌ Error iniciando servicios"
    docker-compose -f docker-compose.yml -f "docker-compose.$ENVIRONMENT.yml" logs
    exit 1
fi

# Ejecutar smoke tests
echo "🧪 Ejecutando smoke tests..."
if [ -f "scripts/smoke-test.sh" ]; then
    bash scripts/smoke-test.sh "$ENVIRONMENT"
else
    # Smoke test básico
    if curl -f -s --max-time 10 "http://localhost:${AEGIS_PORT:-8080}/health" > /dev/null; then
        echo "✅ Endpoint de health check responde correctamente"
    else
        echo "⚠️ Endpoint de health check no responde"
    fi
fi

# Backup de base de datos si existe
if [ "$ENVIRONMENT" = "production" ]; then
    echo "💾 Creando backup de base de datos..."
    docker-compose -f docker-compose.yml -f "docker-compose.$ENVIRONMENT.yml" exec -T db \
        pg_dump -U aegis "aegis_$ENVIRONMENT" > "backups/db-$(date +%Y%m%d-%H%M%S).sql"
fi

# Notificar éxito
echo ""
echo "🎉 ¡Deployment completado exitosamente!"
echo ""
echo "📊 Resumen del deployment:"
echo "   • Entorno: $ENVIRONMENT"
echo "   • Versión: $VERSION"
echo "   • Timestamp: $(date)"
echo "   • Servicios activos: $(docker-compose -f docker-compose.yml -f "docker-compose.$ENVIRONMENT.yml" ps --services --filter "status=running" | wc -l)"
echo ""
echo "🔗 URLs importantes:"
echo "   • API: http://localhost:${AEGIS_PORT:-8080}"
echo "   • Health: http://localhost:${AEGIS_PORT:-8080}/health"
echo "   • Logs: docker-compose -f docker-compose.yml -f docker-compose.$ENVIRONMENT.yml logs -f"
echo ""
echo "📝 Próximos pasos:"
echo "   1. Verificar logs: docker-compose logs -f aegis-node"
echo "   2. Monitorear métricas en el dashboard"
echo "   3. Ejecutar tests de integración si es necesario"
