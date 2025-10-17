#!/bin/bash
# AEGIS Framework - Production Deployment Script
# Despliegue seguro y automatizado para entornos de producción

set -euo pipefail

# Colores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuración
PROJECT_NAME="aegis-framework"
COMPOSE_FILE="docker-compose.prod.yml"
BACKUP_DIR="./backups/$(date +%Y%m%d_%H%M%S)"
LOG_FILE="./logs/deploy_$(date +%Y%m%d_%H%M%S).log"

# Logging function
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $*" | tee -a "$LOG_FILE"
}

error() {
    echo -e "${RED}[ERROR]${NC} $*" >&2 | tee -a "$LOG_FILE"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $*" | tee -a "$LOG_FILE"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $*" | tee -a "$LOG_FILE"
}

# Función de limpieza
cleanup() {
    log "Realizando limpieza..."
    # Aquí iría la lógica de cleanup si algo falla
}

trap cleanup EXIT

# Verificar prerrequisitos
check_prerequisites() {
    log "Verificando prerrequisitos..."

    # Verificar Docker
    if ! command -v docker &> /dev/null; then
        error "Docker no está instalado"
        exit 1
    fi

    # Verificar Docker Compose
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        error "Docker Compose no está instalado"
        exit 1
    fi

    # Verificar OpenSSL
    if ! command -v openssl &> /dev/null; then
        error "OpenSSL no está instalado"
        exit 1
    fi

    # Verificar que estamos en el directorio correcto
    if [[ ! -f "docker-compose.prod.yml" ]]; then
        error "Archivo docker-compose.prod.yml no encontrado. Ejecutar desde el directorio raíz del proyecto."
        exit 1
    fi

    success "Prerrequisitos verificados"
}

# Crear directorios necesarios
create_directories() {
    log "Creando directorios necesarios..."

    directories=(
        "./data/tor"
        "./data/redis"
        "./data/postgres"
        "./logs"
        "./backups"
        "./config"
        "./secrets"
        "./certs"
    )

    for dir in "${directories[@]}"; do
        mkdir -p "$dir"
        chmod 755 "$dir"
    done

    success "Directorios creados"
}

# Generar certificados SSL
generate_ssl_certificates() {
    log "Generando certificados SSL..."

    if [[ ! -f "./certs/aegis.crt" ]]; then
        # Generar certificado autofirmado para desarrollo/producción
        openssl req -x509 -newkey rsa:4096 -keyout ./certs/aegis.key -out ./certs/aegis.crt -days 365 -nodes -subj "/C=ES/ST=Madrid/L=Madrid/O=AEGIS Framework/CN=aegis.local"

        # Establecer permisos seguros
        chmod 600 ./certs/aegis.key
        chmod 644 ./certs/aegis.crt

        success "Certificados SSL generados"
    else
        warning "Certificados SSL ya existen, omitiendo generación"
    fi
}

# Generar secrets
generate_secrets() {
    log "Generando secrets..."

    # Generar contraseña de PostgreSQL
    if [[ ! -f "./secrets/postgres_password.txt" ]]; then
        openssl rand -base64 32 > ./secrets/postgres_password.txt
        chmod 600 ./secrets/postgres_password.txt
    fi

    # Generar contraseña de Grafana
    if [[ ! -f "./secrets/grafana_password.txt" ]]; then
        openssl rand -base64 16 > ./secrets/grafana_password.txt
        chmod 600 ./secrets/grafana_password.txt
    fi

    success "Secrets generados"
}

# Backup de configuración existente
backup_existing_config() {
    log "Creando backup de configuración existente..."

    if [[ -d "./data" ]] || [[ -d "./config" ]]; then
        mkdir -p "$BACKUP_DIR"
        cp -r ./data "$BACKUP_DIR/" 2>/dev/null || true
        cp -r ./config "$BACKUP_DIR/" 2>/dev/null || true
        cp -r ./secrets "$BACKUP_DIR/" 2>/dev/null || true

        success "Backup creado en $BACKUP_DIR"
    else
        log "No hay configuración previa para respaldar"
    fi
}

# Configurar firewall
configure_firewall() {
    log "Configurando firewall..."

    # Detectar sistema operativo
    if command -v ufw &> /dev/null; then
        # Ubuntu/Debian con UFW
        sudo ufw --force enable
        sudo ufw allow 80/tcp
        sudo ufw allow 443/tcp
        sudo ufw allow 8443/tcp
        sudo ufw allow 3000/tcp
        log "Firewall UFW configurado"
    elif command -v firewall-cmd &> /dev/null; then
        # CentOS/RHEL con firewalld
        sudo firewall-cmd --permanent --add-port=80/tcp
        sudo firewall-cmd --permanent --add-port=443/tcp
        sudo firewall-cmd --permanent --add-port=8443/tcp
        sudo firewall-cmd --permanent --add-port=3000/tcp
        sudo firewall-cmd --reload
        log "Firewall firewalld configurado"
    else
        warning "No se detectó UFW ni firewalld. Configurar firewall manualmente."
    fi

    success "Firewall configurado"
}

# Desplegar servicios
deploy_services() {
    log "Desplegando servicios..."

    # Detener servicios existentes si los hay
    log "Deteniendo servicios existentes..."
    docker-compose -f "$COMPOSE_FILE" down || true

    # Limpiar contenedores no utilizados
    log "Limpiando contenedores no utilizados..."
    docker system prune -f

    # Construir imágenes
    log "Construyendo imágenes..."
    docker-compose -f "$COMPOSE_FILE" build --no-cache

    # Iniciar servicios
    log "Iniciando servicios..."
    docker-compose -f "$COMPOSE_FILE" up -d

    # Esperar a que los servicios estén listos
    log "Esperando a que los servicios estén listos..."
    sleep 60

    success "Servicios desplegados"
}

# Verificar despliegue
verify_deployment() {
    log "Verificando despliegue..."

    # Verificar que los contenedores están corriendo
    running_containers=$(docker-compose -f "$COMPOSE_FILE" ps | grep "Up" | wc -l)
    total_services=$(docker-compose -f "$COMPOSE_FILE" config --services | wc -l)

    if [[ $running_containers -lt $total_services ]]; then
        warning "$running_containers de $total_services servicios están ejecutándose"
    else
        success "Todos los servicios están ejecutándose ($running_containers/$total_services)"
    fi

    # Verificar conectividad básica
    if curl -f -k https://localhost:8443/api/health &>/dev/null; then
        success "API de AEGIS responde correctamente"
    else
        warning "API de AEGIS no responde aún (puede tardar más en inicializarse)"
    fi

    # Verificar Grafana
    if curl -f http://localhost:3000/api/health &>/dev/null; then
        success "Grafana está accesible"
    else
        warning "Grafana no está accesible aún"
    fi
}

# Mostrar información post-despliegue
show_post_deployment_info() {
    log "Despliegue completado exitosamente!"
    echo
    echo "========================================"
    echo "🎉 AEGIS Framework - Producción Lista"
    echo "========================================"
    echo
    echo "🌐 URLs de acceso:"
    echo "   • Dashboard Principal: https://localhost:8443"
    echo "   • Dashboard Web:       https://localhost:8444"
    echo "   • Grafana:             http://localhost:3000"
    echo "   • Prometheus:          http://localhost:9090"
    echo
    echo "🔐 Credenciales por defecto:"
    echo "   • Grafana Admin: admin / $(cat ./secrets/grafana_password.txt)"
    echo
    echo "📊 Monitoreo:"
    echo "   • Logs: ./logs/"
    echo "   • Backups: ./backups/"
    echo "   • Configuración: ./config/production_config.json"
    echo
    echo "🛠️ Comandos útiles:"
    echo "   • Ver logs: docker-compose -f $COMPOSE_FILE logs -f"
    echo "   • Reiniciar: docker-compose -f $COMPOSE_FILE restart"
    echo "   • Detener: docker-compose -f $COMPOSE_FILE down"
    echo "   • Backup: docker-compose -f $COMPOSE_FILE exec backup /app/backup.sh"
    echo
    echo "⚠️ IMPORTANTE: Cambiar contraseñas por defecto antes de usar en producción!"
}

# Función principal
main() {
    echo "🚀 AEGIS Framework - Despliegue de Producción"
    echo "=============================================="
    echo

    # Parsear argumentos
    FORCE_DEPLOY=false
    while [[ $# -gt 0 ]]; do
        case $1 in
            --force)
                FORCE_DEPLOY=true
                shift
                ;;
            --help)
                echo "Uso: $0 [--force] [--help]"
                echo "  --force: Forzar despliegue incluso si hay servicios corriendo"
                echo "  --help: Mostrar esta ayuda"
                exit 0
                ;;
            *)
                error "Opción desconocida: $1"
                exit 1
                ;;
        esac
    done

    # Verificar si hay servicios corriendo
    if docker-compose -f "$COMPOSE_FILE" ps | grep -q "Up" && [[ $FORCE_DEPLOY != true ]]; then
        error "Hay servicios corriendo. Usar --force para forzar el despliegue."
        exit 1
    fi

    # Ejecutar despliegue
    check_prerequisites
    create_directories
    backup_existing_config
    generate_ssl_certificates
    generate_secrets
    configure_firewall
    deploy_services
    verify_deployment
    show_post_deployment_info

    success "Despliegue de producción completado exitosamente! 🎉"
}

# Ejecutar función principal
main "$@"
