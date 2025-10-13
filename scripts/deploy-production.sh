#!/bin/bash

# ============================================================================
# AEGIS Framework - Script de Despliegue en Producción para Linux
# ============================================================================
# Descripción: Script para desplegar AEGIS Framework en entorno de producción
# Autor: AEGIS Security Team
# Versión: 2.0.0
# Fecha: 2024
# ============================================================================

set -euo pipefail

# Configuración por defecto
ENVIRONMENT="production"
DOMAIN=""
SSL_CERT_PATH=""
SSL_KEY_PATH=""
SKIP_BACKUP=false
SKIP_TESTS=false
FORCE=false
DRY_RUN=false
VERBOSE=false

# Configuración de producción
REQUIRED_MEMORY_GB=4
REQUIRED_DISK_SPACE_GB=20
REQUIRED_PORTS=(80 443 8080 3000 8545 9050)
BACKUP_RETENTION_DAYS=30
HEALTH_CHECK_TIMEOUT=60
DEPLOYMENT_TIMEOUT=300

# Colores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
WHITE='\033[1;37m'
NC='\033[0m' # No Color

# Función para output con colores
print_color() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

# Función para logging
log_message() {
    local level=$1
    local message=$2
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    # Crear directorio de logs si no existe
    mkdir -p logs
    
    echo "[$timestamp] [$level] $message" >> logs/deployment.log
    
    case $level in
        "ERROR")
            print_color "$RED" "❌ $message"
            ;;
        "WARN")
            print_color "$YELLOW" "⚠️  $message"
            ;;
        "SUCCESS")
            print_color "$GREEN" "✅ $message"
            ;;
        *)
            print_color "$WHITE" "ℹ️  $message"
            ;;
    esac
}

# Función de ayuda
show_help() {
    print_color "$CYAN" "🚀 AEGIS Framework - Despliegue en Producción"
    print_color "$CYAN" "============================================="
    echo ""
    print_color "$YELLOW" "DESCRIPCIÓN:"
    echo "  Despliega AEGIS Framework en un entorno de producción seguro"
    echo ""
    print_color "$YELLOW" "USO:"
    echo "  ./deploy-production.sh [OPCIONES]"
    echo ""
    print_color "$YELLOW" "OPCIONES:"
    echo "  -e, --environment <env>      Entorno de despliegue (default: production)"
    echo "  -d, --domain <dominio>       Dominio para el despliegue (ej: aegis.example.com)"
    echo "  -c, --ssl-cert <ruta>        Ruta al certificado SSL"
    echo "  -k, --ssl-key <ruta>         Ruta a la clave privada SSL"
    echo "  --skip-backup                Omitir backup antes del despliegue"
    echo "  --skip-tests                 Omitir pruebas antes del despliegue"
    echo "  -f, --force                  Forzar despliegue sin confirmaciones"
    echo "  --dry-run                    Simular despliegue sin ejecutar cambios"
    echo "  -v, --verbose                Output detallado"
    echo "  -h, --help                   Mostrar esta ayuda"
    echo ""
    print_color "$YELLOW" "EJEMPLOS:"
    echo "  ./deploy-production.sh -d aegis.company.com"
    echo "  ./deploy-production.sh -d aegis.company.com -c cert.pem -k key.pem"
    echo "  ./deploy-production.sh --dry-run                              # Simulación"
    echo "  ./deploy-production.sh -f --skip-tests                       # Despliegue rápido"
    echo ""
    print_color "$YELLOW" "REQUISITOS:"
    echo "  - Ubuntu 20.04+ / CentOS 8+ / Debian 11+"
    echo "  - 4GB+ RAM disponible"
    echo "  - 20GB+ espacio en disco"
    echo "  - Puertos 80, 443, 8080, 3000, 8545, 9050 disponibles"
    echo "  - Certificados SSL válidos (recomendado)"
    echo "  - Permisos sudo"
    echo ""
    exit 0
}

# Parsear argumentos
parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -e|--environment)
                ENVIRONMENT="$2"
                shift 2
                ;;
            -d|--domain)
                DOMAIN="$2"
                shift 2
                ;;
            -c|--ssl-cert)
                SSL_CERT_PATH="$2"
                shift 2
                ;;
            -k|--ssl-key)
                SSL_KEY_PATH="$2"
                shift 2
                ;;
            --skip-backup)
                SKIP_BACKUP=true
                shift
                ;;
            --skip-tests)
                SKIP_TESTS=true
                shift
                ;;
            -f|--force)
                FORCE=true
                shift
                ;;
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            -v|--verbose)
                VERBOSE=true
                shift
                ;;
            -h|--help)
                show_help
                ;;
            *)
                echo "Opción desconocida: $1"
                show_help
                ;;
        esac
    done
}

# Verificar requisitos del sistema
test_prerequisites() {
    print_color "$CYAN" "🔍 Verificando requisitos del sistema..."
    
    local issues=()
    
    # Verificar sistema operativo
    if [[ ! -f /etc/os-release ]]; then
        issues+=("No se puede determinar el sistema operativo")
    else
        source /etc/os-release
        case $ID in
            ubuntu)
                if [[ $(echo "$VERSION_ID >= 20.04" | bc -l) -eq 0 ]]; then
                    issues+=("Ubuntu 20.04+ requerido, encontrado: $VERSION_ID")
                fi
                ;;
            centos|rhel)
                if [[ $(echo "$VERSION_ID >= 8" | bc -l) -eq 0 ]]; then
                    issues+=("CentOS/RHEL 8+ requerido, encontrado: $VERSION_ID")
                fi
                ;;
            debian)
                if [[ $(echo "$VERSION_ID >= 11" | bc -l) -eq 0 ]]; then
                    issues+=("Debian 11+ requerido, encontrado: $VERSION_ID")
                fi
                ;;
            *)
                issues+=("Sistema operativo no soportado: $ID")
                ;;
        esac
    fi
    
    # Verificar memoria RAM
    local total_memory_gb=$(free -g | awk '/^Mem:/{print $2}')
    if [[ $total_memory_gb -lt $REQUIRED_MEMORY_GB ]]; then
        issues+=("Memoria insuficiente: ${total_memory_gb}GB disponible, ${REQUIRED_MEMORY_GB}GB requerido")
    fi
    
    # Verificar espacio en disco
    local free_space_gb=$(df -BG . | awk 'NR==2 {print $4}' | sed 's/G//')
    if [[ $free_space_gb -lt $REQUIRED_DISK_SPACE_GB ]]; then
        issues+=("Espacio en disco insuficiente: ${free_space_gb}GB disponible, ${REQUIRED_DISK_SPACE_GB}GB requerido")
    fi
    
    # Verificar puertos
    for port in "${REQUIRED_PORTS[@]}"; do
        if ss -tuln | grep -q ":$port "; then
            issues+=("Puerto $port está en uso")
        fi
    done
    
    # Verificar Bash
    if [[ ${BASH_VERSION%%.*} -lt 4 ]]; then
        issues+=("Bash 4.0+ requerido")
    fi
    
    # Verificar permisos sudo
    if ! sudo -n true 2>/dev/null; then
        issues+=("Se requieren permisos sudo")
    fi
    
    # Verificar herramientas básicas
    local tools=("curl" "wget" "jq" "bc" "ss")
    for tool in "${tools[@]}"; do
        if ! command -v "$tool" &> /dev/null; then
            issues+=("Herramienta faltante: $tool")
        fi
    done
    
    if [[ ${#issues[@]} -gt 0 ]]; then
        log_message "ERROR" "Requisitos del sistema no cumplidos:"
        for issue in "${issues[@]}"; do
            log_message "ERROR" "  - $issue"
        done
        return 1
    fi
    
    log_message "SUCCESS" "Todos los requisitos del sistema cumplidos"
    return 0
}

# Verificar dependencias
test_dependencies() {
    print_color "$CYAN" "🔧 Verificando dependencias..."
    
    local missing=()
    
    # Python 3.8+
    if command -v python3 &> /dev/null; then
        local python_version=$(python3 --version | cut -d' ' -f2)
        log_message "INFO" "Python: $python_version"
        
        local python_major=$(echo "$python_version" | cut -d'.' -f1)
        local python_minor=$(echo "$python_version" | cut -d'.' -f2)
        
        if [[ $python_major -lt 3 ]] || [[ $python_major -eq 3 && $python_minor -lt 8 ]]; then
            missing+=("Python 3.8+ (encontrado: $python_version)")
        fi
    else
        missing+=("Python 3.8+")
    fi
    
    # Node.js 18+
    if command -v node &> /dev/null; then
        local node_version=$(node --version | sed 's/v//')
        log_message "INFO" "Node.js: $node_version"
        
        local node_major=$(echo "$node_version" | cut -d'.' -f1)
        if [[ $node_major -lt 18 ]]; then
            missing+=("Node.js 18+ (encontrado: $node_version)")
        fi
    else
        missing+=("Node.js 18+")
    fi
    
    # npm
    if command -v npm &> /dev/null; then
        local npm_version=$(npm --version)
        log_message "INFO" "npm: $npm_version"
    else
        missing+=("npm")
    fi
    
    # Git
    if command -v git &> /dev/null; then
        local git_version=$(git --version | cut -d' ' -f3)
        log_message "INFO" "Git: $git_version"
    else
        missing+=("Git")
    fi
    
    if [[ ${#missing[@]} -gt 0 ]]; then
        log_message "ERROR" "Dependencias faltantes: ${missing[*]}"
        log_message "ERROR" "Ejecuta ./install-dependencies.sh para instalar las dependencias"
        return 1
    fi
    
    log_message "SUCCESS" "Todas las dependencias están disponibles"
    return 0
}

# Crear backup del despliegue actual
backup_current_deployment() {
    if [[ $SKIP_BACKUP == true ]]; then
        log_message "WARN" "Omitiendo backup (--skip-backup especificado)"
        return 0
    fi
    
    print_color "$CYAN" "💾 Creando backup del despliegue actual..."
    
    local backup_dir="backups/production-$(date '+%Y%m%d-%H%M%S')"
    
    if [[ $DRY_RUN == true ]]; then
        log_message "INFO" "DRY RUN: Crearía backup en $backup_dir"
        return 0
    fi
    
    mkdir -p "$backup_dir"
    
    # Backup de configuración
    local config_files=(".env" "config/app_config.json" "config/torrc")
    for file in "${config_files[@]}"; do
        if [[ -f $file ]]; then
            cp "$file" "$backup_dir/"
            log_message "INFO" "Backup creado: $file"
        fi
    done
    
    # Backup de datos
    if [[ -d "data" ]]; then
        cp -r "data" "$backup_dir/"
        log_message "INFO" "Backup de datos creado"
    fi
    
    # Backup de logs importantes
    if [[ -d "logs" ]]; then
        mkdir -p "$backup_dir/logs"
        find logs -name "*.log" -mtime -7 -exec cp {} "$backup_dir/logs/" \;
        log_message "INFO" "Backup de logs recientes creado"
    fi
    
    # Crear metadata del backup
    cat > "$backup_dir/backup_info.json" << EOF
{
    "timestamp": "$(date -Iseconds)",
    "environment": "$ENVIRONMENT",
    "domain": "$DOMAIN",
    "backup_type": "production_deployment",
    "created_by": "$(whoami)",
    "hostname": "$(hostname)"
}
EOF
    
    log_message "SUCCESS" "Backup completado en: $backup_dir"
    return 0
}

# Ejecutar pruebas
run_tests() {
    if [[ $SKIP_TESTS == true ]]; then
        log_message "WARN" "Omitiendo pruebas (--skip-tests especificado)"
        return 0
    fi
    
    print_color "$CYAN" "🧪 Ejecutando pruebas..."
    
    if [[ $DRY_RUN == true ]]; then
        log_message "INFO" "DRY RUN: Ejecutaría pruebas del sistema"
        return 0
    fi
    
    # Ejecutar health check
    if [[ -f "scripts/health-check.sh" ]]; then
        if ./scripts/health-check.sh --json > /dev/null 2>&1; then
            log_message "SUCCESS" "Health check pasó"
        else
            log_message "ERROR" "Health check falló"
            return 1
        fi
    fi
    
    # Verificar configuración
    if [[ -f ".env" ]]; then
        local required_vars=("FLASK_ENV" "SECRET_KEY" "DATABASE_URL")
        
        for var in "${required_vars[@]}"; do
            if ! grep -q "^$var=" .env; then
                log_message "ERROR" "Variable de entorno faltante: $var"
                return 1
            fi
        done
        log_message "SUCCESS" "Configuración de entorno validada"
    fi
    
    return 0
}

# Configurar entorno de producción
setup_production_config() {
    print_color "$CYAN" "⚙️  Configurando entorno de producción..."
    
    if [[ $DRY_RUN == true ]]; then
        log_message "INFO" "DRY RUN: Configuraría entorno de producción"
        return 0
    fi
    
    # Crear configuración de producción
    local secret_key=$(openssl rand -hex 32)
    local password_salt=$(openssl rand -hex 16)
    
    cat > .env << EOF
# AEGIS Framework - Configuración de Producción
FLASK_ENV=production
DEBUG=False
TESTING=False

# Seguridad
SECRET_KEY=$secret_key
SECURITY_PASSWORD_SALT=$password_salt

# Base de datos
DATABASE_URL=sqlite:///production.db

# Logging
LOG_LEVEL=INFO
LOG_TO_FILE=True

# Servicios
DASHBOARD_HOST=0.0.0.0
DASHBOARD_PORT=8080
SECURECHAT_PORT=3000
BLOCKCHAIN_PORT=8545
TOR_PORT=9050

# SSL/TLS
EOF

    if [[ -n $DOMAIN ]]; then
        echo "DOMAIN=$DOMAIN" >> .env
    fi
    
    if [[ -n $SSL_CERT_PATH && -n $SSL_KEY_PATH ]]; then
        cat >> .env << EOF
SSL_CERT_PATH=$SSL_CERT_PATH
SSL_KEY_PATH=$SSL_KEY_PATH
SSL_ENABLED=True
EOF
    fi
    
    log_message "SUCCESS" "Configuración de producción creada"
    
    # Configurar app_config.json para producción
    mkdir -p config
    cat > config/app_config.json << EOF
{
    "environment": "$ENVIRONMENT",
    "debug": false,
    "logging": {
        "level": "INFO",
        "file": "logs/aegis-production.log",
        "max_size": "100MB",
        "backup_count": 5
    },
    "security": {
        "csrf_enabled": true,
        "session_timeout": 3600,
        "max_login_attempts": 5,
        "password_policy": {
            "min_length": 12,
            "require_uppercase": true,
            "require_lowercase": true,
            "require_numbers": true,
            "require_symbols": true
        }
    },
    "performance": {
        "cache_enabled": true,
        "compression_enabled": true,
        "static_file_caching": true
    }
EOF

    if [[ -n $DOMAIN ]]; then
        echo '    ,"domain": "'$DOMAIN'"' >> config/app_config.json
    fi
    
    echo '}' >> config/app_config.json
    
    log_message "SUCCESS" "Configuración de aplicación actualizada"
    return 0
}

# Desplegar servicios
deploy_services() {
    print_color "$CYAN" "🚀 Desplegando servicios..."
    
    if [[ $DRY_RUN == true ]]; then
        log_message "INFO" "DRY RUN: Desplegaría todos los servicios"
        return 0
    fi
    
    # Detener servicios existentes
    if [[ -f "scripts/stop-all-services.sh" ]]; then
        log_message "INFO" "Deteniendo servicios existentes..."
        ./scripts/stop-all-services.sh --force || true
    fi
    
    # Instalar/actualizar dependencias Python
    log_message "INFO" "Instalando dependencias Python..."
    if [[ -d "venv" ]]; then
        source venv/bin/activate
        pip install -r requirements.txt --upgrade
    else
        python3 -m venv venv
        source venv/bin/activate
        pip install -r requirements.txt
    fi
    
    # Instalar/actualizar dependencias Node.js
    log_message "INFO" "Instalando dependencias Node.js..."
    
    # Secure Chat UI
    if [[ -d "dapps/secure-chat/ui" ]]; then
        cd dapps/secure-chat/ui
        npm ci --production
        npm run build
        cd ../../..
    fi
    
    # AEGIS Token
    if [[ -d "dapps/aegis-token" ]]; then
        cd dapps/aegis-token
        npm ci --production
        cd ../..
    fi
    
    # Configurar servicios systemd (si está disponible)
    if command -v systemctl &> /dev/null; then
        log_message "INFO" "Configurando servicios systemd..."
        
        # Crear servicio para AEGIS Dashboard
        sudo tee /etc/systemd/system/aegis-dashboard.service > /dev/null << EOF
[Unit]
Description=AEGIS Framework Dashboard
After=network.target

[Service]
Type=simple
User=$(whoami)
WorkingDirectory=$(pwd)
Environment=PATH=$(pwd)/venv/bin
ExecStart=$(pwd)/venv/bin/python main.py start-dashboard --config config/app_config.json
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF
        
        sudo systemctl daemon-reload
        sudo systemctl enable aegis-dashboard.service
        log_message "SUCCESS" "Servicio systemd configurado para Dashboard"
    fi
    
    # Iniciar servicios
    log_message "INFO" "Iniciando servicios..."
    if [[ -f "scripts/start-all-services.sh" ]]; then
        ./scripts/start-all-services.sh
    fi
    
    log_message "SUCCESS" "Servicios desplegados exitosamente"
    return 0
}

# Verificar despliegue
test_deployment() {
    print_color "$CYAN" "🔍 Verificando despliegue..."
    
    if [[ $DRY_RUN == true ]]; then
        log_message "INFO" "DRY RUN: Verificaría el despliegue"
        return 0
    fi
    
    local max_attempts=$((HEALTH_CHECK_TIMEOUT / 5))
    local attempt=0
    
    while [[ $attempt -lt $max_attempts ]]; do
        ((attempt++))
        log_message "INFO" "Intento de verificación $attempt/$max_attempts..."
        
        local all_healthy=true
        
        # Verificar Dashboard
        if curl -s -f "http://localhost:8080/health" > /dev/null 2>&1; then
            log_message "SUCCESS" "Dashboard: OK"
        else
            log_message "WARN" "Dashboard: No disponible"
            all_healthy=false
        fi
        
        # Verificar Secure Chat UI
        if curl -s -f "http://localhost:3000" > /dev/null 2>&1; then
            log_message "SUCCESS" "Secure Chat UI: OK"
        else
            log_message "WARN" "Secure Chat UI: No disponible"
            all_healthy=false
        fi
        
        # Verificar Blockchain
        if curl -s -f -X POST -H "Content-Type: application/json" \
           -d '{"jsonrpc":"2.0","method":"eth_blockNumber","params":[],"id":1}' \
           "http://localhost:8545" > /dev/null 2>&1; then
            log_message "SUCCESS" "Blockchain: OK"
        else
            log_message "WARN" "Blockchain: No disponible (normal si no está configurado)"
        fi
        
        if [[ $all_healthy == true ]]; then
            log_message "SUCCESS" "Despliegue verificado exitosamente"
            return 0
        fi
        
        if [[ $attempt -lt $max_attempts ]]; then
            sleep 5
        fi
    done
    
    log_message "ERROR" "Verificación del despliegue falló después de $max_attempts intentos"
    return 1
}

# Mostrar resumen del despliegue
show_deployment_summary() {
    print_color "$CYAN" ""
    print_color "$CYAN" "🎉 Resumen del Despliegue"
    print_color "$CYAN" "========================="
    
    print_color "$GREEN" "✅ Despliegue completado exitosamente"
    echo ""
    
    print_color "$YELLOW" "🌐 URLs de Acceso:"
    echo "  Dashboard:      http://localhost:8080"
    echo "  Secure Chat UI: http://localhost:3000"
    echo "  Blockchain RPC: http://localhost:8545"
    
    if [[ -n $DOMAIN ]]; then
        echo ""
        print_color "$YELLOW" "🌍 URLs Públicas:"
        echo "  Dashboard:      https://$DOMAIN:8080"
        echo "  Secure Chat UI: https://$DOMAIN:3000"
    fi
    
    echo ""
    print_color "$YELLOW" "📋 Próximos Pasos:"
    echo "  1. Verificar que todos los servicios estén funcionando"
    echo "  2. Configurar firewall para los puertos necesarios"
    echo "  3. Configurar certificados SSL si no se hizo"
    echo "  4. Configurar monitoreo y alertas"
    echo "  5. Realizar backup regular de la configuración"
    
    echo ""
    print_color "$YELLOW" "🔧 Comandos Útiles:"
    echo "  Verificar estado:    ./scripts/health-check.sh"
    echo "  Ver logs:           tail -f logs/aegis-production.log"
    echo "  Reiniciar servicios: ./scripts/stop-all-services.sh && ./scripts/start-all-services.sh"
    echo "  Monitorear:         ./scripts/monitor-services.sh --continuous"
    
    echo ""
    print_color "$YELLOW" "⚠️  Recordatorios de Seguridad:"
    echo "  - Cambiar contraseñas por defecto"
    echo "  - Configurar firewall apropiadamente"
    echo "  - Habilitar logging de auditoría"
    echo "  - Configurar backups automáticos"
    echo "  - Revisar configuración de Tor"
}

# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================

main() {
    # Parsear argumentos
    parse_arguments "$@"
    
    print_color "$CYAN" "🚀 AEGIS Framework - Despliegue en Producción"
    print_color "$CYAN" "============================================="
    echo ""
    
    # Verificar directorio
    if [[ ! -f "main.py" ]]; then
        log_message "ERROR" "No se encontró main.py. Ejecuta este script desde el directorio raíz del proyecto AEGIS."
        exit 1
    fi
    
    # Confirmación si no es DryRun y no es Force
    if [[ $DRY_RUN == false && $FORCE == false ]]; then
        print_color "$YELLOW" "⚠️  ADVERTENCIA: Este script desplegará AEGIS Framework en producción."
        print_color "$YELLOW" "Esto puede sobrescribir configuraciones existentes."
        echo ""
        read -p "¿Continuar con el despliegue? (y/N): " -r confirmation
        if [[ ! $confirmation =~ ^[Yy]$ ]]; then
            print_color "$YELLOW" "Despliegue cancelado por el usuario."
            exit 0
        fi
    fi
    
    local start_time=$(date +%s)
    log_message "INFO" "Iniciando despliegue en producción..."
    
    # Ejecutar pasos del despliegue
    local steps=(
        "test_prerequisites:Verificar requisitos"
        "test_dependencies:Verificar dependencias"
        "backup_current_deployment:Crear backup"
        "run_tests:Ejecutar pruebas"
        "setup_production_config:Configurar producción"
        "deploy_services:Desplegar servicios"
        "test_deployment:Verificar despliegue"
    )
    
    for step in "${steps[@]}"; do
        local func_name="${step%%:*}"
        local step_name="${step##*:}"
        
        print_color "$BLUE" ""
        print_color "$BLUE" "📋 $step_name..."
        
        if ! $func_name; then
            log_message "ERROR" "Paso falló: $step_name"
            print_color "$RED" ""
            print_color "$RED" "❌ Despliegue falló en: $step_name"
            print_color "$YELLOW" "Revisa los logs en logs/deployment.log para más detalles."
            exit 1
        fi
    done
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    local duration_min=$(echo "scale=2; $duration / 60" | bc)
    
    log_message "SUCCESS" "Despliegue completado en ${duration_min} minutos"
    
    if [[ $DRY_RUN == false ]]; then
        show_deployment_summary
    else
        print_color "$GREEN" ""
        print_color "$GREEN" "✅ Simulación de despliegue completada exitosamente"
        print_color "$YELLOW" "Ejecuta sin --dry-run para realizar el despliegue real."
    fi
}

# Ejecutar función principal
main "$@"