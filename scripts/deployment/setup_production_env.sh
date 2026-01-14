#!/bin/bash
# AEGIS Framework - Production Environment Setup Script
# Configuración segura de variables de entorno para producción

set -euo pipefail  # Salir en error, variables no definidas, fallos en pipe

# Colores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Función de logging
log() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

# Verificar si se ejecuta como root
if [[ $EUID -eq 0 ]]; then
   error "Este script no debe ejecutarse como root"
   exit 1
fi

# Directorio de trabajo
AEGIS_HOME="${AEGIS_HOME:-/opt/aegis}"
ENV_FILE="${AEGIS_HOME}/.env"
BACKUP_DIR="${AEGIS_HOME}/backups"

log "Iniciando configuración de entorno de producción AEGIS..."

# Crear directorios necesarios
mkdir -p "${AEGIS_HOME}" "${BACKUP_DIR}" "${AEGIS_HOME}/logs" "${AEGIS_HOME}/certs"

# Generar contraseñas seguras
generate_password() {
    local length=${1:-32}
    openssl rand -base64 $length | tr -d "=+/" | cut -c1-$length
}

generate_secret() {
    openssl rand -hex 64
}

# Verificar dependencias
check_dependencies() {
    log "Verificando dependencias..."
    
    local deps=("openssl" "pwgen" "python3" "python3-pip")
    local missing_deps=()
    
    for dep in "${deps[@]}"; do
        if ! command -v "$dep" &> /dev/null; then
            missing_deps+=("$dep")
        fi
    done
    
    if [[ ${#missing_deps[@]} -ne 0 ]]; then
        error "Dependencias faltantes: ${missing_deps[*]}"
        error "Instale con: sudo apt-get install openssl pwgen python3 python3-pip"
        exit 1
    fi
    
    success "Todas las dependencias están instaladas"
}

# Generar certificados SSL autofirmados para desarrollo
generate_ssl_certs() {
    log "Generando certificados SSL..."
    
    local cert_dir="${AEGIS_HOME}/certs"
    local key_file="${cert_dir}/server.key"
    local cert_file="${cert_dir}/server.crt"
    local csr_file="${cert_dir}/server.csr"
    
    # Crear directorio de certificados con permisos seguros
    mkdir -p "$cert_dir"
    chmod 700 "$cert_dir"
    
    # Generar clave privada
    if [[ ! -f "$key_file" ]]; then
        openssl genrsa -out "$key_file" 4096
        chmod 600 "$key_file"
        success "Clave privada generada: $key_file"
    else
        warning "La clave privada ya existe: $key_file"
    fi
    
    # Generar CSR (Certificate Signing Request)
    if [[ ! -f "$csr_file" ]]; then
        openssl req -new -key "$key_file" -out "$csr_file" \
            -subj "/C=ES/ST=Madrid/L=Madrid/O=AEGIS Framework/CN=aegis.local"
        success "CSR generado: $csr_file"
    fi
    
    # Generar certificado autofirmado válido por 10 años
    if [[ ! -f "$cert_file" ]]; then
        openssl x509 -req -days 3650 -in "$csr_file" -signkey "$key_file" -out "$cert_file"
        chmod 644 "$cert_file"
        success "Certificado SSL generado: $cert_file"
    else
        warning "El certificado SSL ya existe: $cert_file"
    fi
    
    # Verificar certificado
    openssl x509 -in "$cert_file" -text -noout | grep -E "Subject:|Issuer:|Not After"
}

# Configurar variables de entorno
setup_environment() {
    log "Configurando variables de entorno..."
    
    # Hacer backup del archivo .env actual si existe
    if [[ -f "$ENV_FILE" ]]; then
        local backup_file="${BACKUP_DIR}/.env.backup.$(date +%Y%m%d_%H%M%S)"
        cp "$ENV_FILE" "$backup_file"
        warning "Backup creado: $backup_file"
    fi
    
    # Generar contraseñas y secretos seguros
    local jwt_secret=$(generate_secret)
    local postgres_password=$(generate_password 32)
    local redis_password=$(generate_password 32)
    local grafana_password=$(generate_password 16)
    local backup_encryption_key=$(generate_secret)
    local smtp_password=$(generate_password 24)
    local aws_access_key=$(generate_password 20)
    local aws_secret_key=$(generate_password 40)
    
    # Crear archivo .env
    cat > "$ENV_FILE" << EOF
# AEGIS Framework - Production Environment Configuration
# Generado el $(date)

# =============================================================================
# CONFIGURACIÓN GENERAL
# =============================================================================
AEGIS_ENVIRONMENT=production
AEGIS_VERSION=3.0.0
AEGIS_DEBUG=false
AEGIS_SECURITY_LEVEL=high
AEGIS_LOG_LEVEL=INFO
AEGIS_SERVER_ID=aegis-prod-001
AEGIS_DATA_CENTER=primary

# =============================================================================
# CONFIGURACIÓN DE BASES DE DATOS
# =============================================================================
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DATABASE=aegis_db
POSTGRES_USERNAME=aegis
POSTGRES_PASSWORD=${postgres_password}
POSTGRES_SSL_MODE=require
POSTGRES_POOL_SIZE=20

REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DATABASE=0
REDIS_USERNAME=
REDIS_PASSWORD=${redis_password}
REDIS_SSL=false
REDIS_POOL_SIZE=10

# =============================================================================
# CONFIGURACIÓN DE SEGURIDAD
# =============================================================================
JWT_SECRET=${jwt_secret}
JWT_ALGORITHM=HS256
JWT_EXPIRATION=3600
JWT_REFRESH_EXPIRATION=604800

SSL_CERT_PATH=/etc/ssl/certs/aegis.crt
SSL_KEY_PATH=/etc/ssl/private/aegis.key
SSL_CA_PATH=/etc/ssl/certs/ca-certificates.crt

# =============================================================================
# CONFIGURACIÓN DE MONITOREO
# =============================================================================
GRAFANA_ADMIN_PASSWORD=${grafana_password}
PROMETHEUS_PORT=9090
METRICS_ENABLED=true

SMTP_SERVER=localhost
SMTP_PORT=587
SMTP_USERNAME=aegis-alerts
SMTP_PASSWORD=${smtp_password}
SMTP_USE_TLS=true
ALERT_EMAIL_RECIPIENTS=admin@aegis.local

# =============================================================================
# CONFIGURACIÓN DE BACKUPS
# =============================================================================
BACKUP_ENCRYPTION_KEY=${backup_encryption_key}
BACKUP_RETENTION_DAYS=30
BACKUP_SCHEDULE="0 2 * * *"

# =============================================================================
# CONFIGURACIÓN DE CLOUD (OPCIONAL)
# =============================================================================
AWS_ACCESS_KEY_ID=${aws_access_key}
AWS_SECRET_ACCESS_KEY=${aws_secret_key}
AWS_REGION=us-east-1
AWS_S3_BUCKET=aegis-backups

# =============================================================================
# CONFIGURACIÓN DE TOR
# =============================================================================
TOR_CONTROL_PORT=9051
TOR_SOCKS_PORT=9050
TOR_HIDDEN_SERVICE_DIR=/var/lib/tor/aegis/
TOR_COOKIE_AUTHENTICATION=true

# =============================================================================
# CONFIGURACIÓN DE API
# =============================================================================
API_HOST=127.0.0.1
API_PORT=8000
API_SSL=true
API_RATE_LIMIT=1000
API_TIMEOUT=30

WEBSOCKET_HOST=127.0.0.1
WEBSOCKET_PORT=8001
WEBSOCKET_SSL=true
WEBSOCKET_MAX_CONNECTIONS=1000

# =============================================================================
# CONFIGURACIÓN DE DASHBOARD
# =============================================================================
DASHBOARD_HOST=0.0.0.0
DASHBOARD_PORT=8080
DASHBOARD_SSL=true
DASHBOARD_AUTHENTICATION=true

MONITORING_DASHBOARD_HOST=127.0.0.1
MONITORING_DASHBOARD_PORT=8051

# =============================================================================
# CONFIGURACIÓN DE RECURSOS
# =============================================================================
MAX_MEMORY_USAGE=2GB
MAX_CPU_USAGE=80
MAX_DISK_USAGE=90
MAX_CONNECTIONS=1000
MAX_PROCESSES=100
MAX_THREADS=50

# =============================================================================
# CONFIGURACIÓN DE CONSENSO
# =============================================================================
CONSENSUS_ALGORITHM=proof_of_stake
BLOCK_TIME=15
MINIMUM_STAKE=1000
VALIDATOR_COUNT=21

# =============================================================================
# CONFIGURACIÓN DE CRIPTOGRAFÍA
# =============================================================================
WALLET_ENCRYPTION_ALGORITHM=AES-256-GCM
TRANSACTION_SIGNING_ALGORITHM=ECDSA
KEY_ROTATION_INTERVAL=2592000

# =============================================================================
# CONFIGURACIÓN DE LOGGING
# =============================================================================
LOG_FILE_PATH=/var/log/aegis
LOG_MAX_FILE_SIZE=50MB
LOG_RETENTION_DAYS=30
LOG_COMPRESSION=true
AUDIT_LOG_ENABLED=true
AUDIT_LOG_RETENTION_DAYS=90

# =============================================================================
# CONFIGURACIÓN DE MANTENIMIENTO
# =============================================================================
AUTO_UPDATE_CHECK=true
AUTO_UPDATE_SECURITY=true
MAINTENANCE_WINDOW="02:00-04:00"
CLEANUP_INTERVAL=3600

# =============================================================================
# CONFIGURACIÓN DE RED
# =============================================================================
P2P_PORT=8001
P2P_MAX_PEERS=50
P2P_MIN_PEERS=5
P2P_ENCRYPTION_REQUIRED=true
P2P_SIGNATURE_VERIFICATION=true

# =============================================================================
# CONFIGURACIÓN DE SEGURIDAD ADICIONAL
# =============================================================================
RATE_LIMIT_DEFAULT=100
RATE_LIMIT_LOGIN=5
MAX_LOGIN_ATTEMPTS=5
SESSION_TIMEOUT=1800
IP_WHITELIST="127.0.0.1/32,::1/128,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16"
EOF

    # Establecer permisos seguros
    chmod 600 "$ENV_FILE"
    success "Archivo de entorno creado: $ENV_FILE"
}

# Configurar directorios y permisos
setup_directories() {
    log "Configurando directorios y permisos..."
    
    # Crear directorios necesarios
    local dirs=(
        "/var/log/aegis"
        "/var/lib/aegis"
        "/var/backups/aegis"
        "/etc/aegis"
        "/var/run/aegis"
        "/tmp/aegis"
    )
    
    for dir in "${dirs[@]}"; do
        sudo mkdir -p "$dir"
        sudo chown "$(whoami):$(whoami)" "$dir"
        sudo chmod 755 "$dir"
    done
    
    # Directorios sensibles con permisos más restrictivos
    sudo mkdir -p "/etc/ssl/private"
    sudo chmod 700 "/etc/ssl/private"
    
    success "Directorios configurados correctamente"
}

# Validar configuración
validate_configuration() {
    log "Validando configuración..."
    
    # Verificar que el archivo .env existe y tiene permisos correctos
    if [[ ! -f "$ENV_FILE" ]]; then
        error "Archivo .env no encontrado"
        exit 1
    fi
    
    if [[ $(stat -c "%a" "$ENV_FILE") != "600" ]]; then
        warning "Permisos del archivo .env deben ser 600"
        chmod 600 "$ENV_FILE"
    fi
    
    # Verificar que los certificados SSL existen
    local cert_file="${AEGIS_HOME}/certs/server.crt"
    local key_file="${AEGIS_HOME}/certs/server.key"
    
    if [[ ! -f "$cert_file" ]] || [[ ! -f "$key_file" ]]; then
        error "Certificados SSL no encontrados"
        exit 1
    fi
    
    # Validar sintaxis del certificado
    if ! openssl x509 -in "$cert_file" -text -noout &>/dev/null; then
        error "Certificado SSL inválido"
        exit 1
    fi
    
    success "Configuración validada exitosamente"
}

# Mostrar resumen
display_summary() {
    log "Resumen de configuración:"
    echo
    echo "==========================================="
    echo "  AEGIS FRAMEWORK - CONFIGURACIÓN LISTA"
    echo "==========================================="
    echo
    echo "Directorio de instalación: ${AEGIS_HOME}"
    echo "Archivo de entorno: ${ENV_FILE}"
    echo "Certificados SSL: ${AEGIS_HOME}/certs/"
    echo "Logs: /var/log/aegis/"
    echo "Backups: /var/backups/aegis/"
    echo
    echo "Puertos configurados:"
    echo "  - API REST: 8000"
    echo "  - WebSocket: 8001"
    echo "  - Dashboard: 8080"
    echo "  - Monitoreo: 8051"
    echo "  - Prometheus: 9090"
    echo "  - Tor SOCKS: 9050"
    echo "  - Tor Control: 9051"
    echo
    echo "IMPORTANTE:"
    echo "- Revise y actualice las contraseñas generadas"
    echo "- Configure los certificados SSL reales para producción"
    echo "- Ajuste los parámetros según sus necesidades"
    echo "- Ejecute el script de despliegue para completar la instalación"
    echo
    echo "Para iniciar AEGIS:"
    echo "  python main.py start-node"
    echo "  python main.py start-dashboard"
    echo
    success "Configuración de entorno completada!"
}

# Función principal
main() {
    log "Iniciando configuración de entorno AEGIS Framework v3.0.0"
    
    check_dependencies
    generate_ssl_certs
    setup_environment
    setup_directories
    validate_configuration
    display_summary
    
    success "Configuración de entorno de producción completada exitosamente!"
    log "AEGIS Framework está listo para ser desplegado en producción."
}

# Ejecutar función principal
main "$@"