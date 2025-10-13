#!/bin/bash

# ============================================================================
# AEGIS Framework - Configurador de Archivos para Linux
# ============================================================================
# Descripción: Script para configurar únicamente los archivos de configuración
# Autor: AEGIS Security Team
# Versión: 2.0.0
# Fecha: 2024
# ============================================================================

set -euo pipefail

# Configuración de colores
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
WHITE='\033[1;37m'
NC='\033[0m' # No Color

# Variables globales
SKIP_TOR=false
FORCE=false
VERBOSE=false

# Funciones de utilidad
log_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

log_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

log_error() {
    echo -e "${RED}❌ $1${NC}"
}

log_header() {
    echo -e "${CYAN}$1${NC}"
}

show_help() {
    log_header "🛡️  AEGIS Framework - Configurador de Archivos"
    log_header "==============================================="
    echo ""
    echo -e "${YELLOW}DESCRIPCIÓN:${NC}"
    echo "  Configura únicamente los archivos de configuración para AEGIS Framework"
    echo ""
    echo -e "${YELLOW}USO:${NC}"
    echo "  ./setup-config.sh [OPCIONES]"
    echo ""
    echo -e "${YELLOW}OPCIONES:${NC}"
    echo "  --skip-tor     Omitir configuración de Tor"
    echo "  --force        Sobrescribir archivos existentes"
    echo "  --verbose      Mostrar información detallada"
    echo "  --help         Mostrar esta ayuda"
    echo ""
    echo -e "${YELLOW}EJEMPLOS:${NC}"
    echo "  ./setup-config.sh"
    echo "  ./setup-config.sh --skip-tor"
    echo "  ./setup-config.sh --force --verbose"
    echo ""
    exit 0
}

parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --skip-tor)
                SKIP_TOR=true
                shift
                ;;
            --force)
                FORCE=true
                shift
                ;;
            --verbose)
                VERBOSE=true
                shift
                ;;
            --help)
                show_help
                ;;
            *)
                log_error "Opción desconocida: $1"
                echo "Usa --help para ver las opciones disponibles"
                exit 1
                ;;
        esac
    done
}

test_prerequisites() {
    log_info "Verificando prerequisitos..."
    
    local all_good=true
    
    # Verificar Python
    if command -v python3 &> /dev/null; then
        local python_version=$(python3 --version 2>&1)
        if [[ $python_version =~ Python\ 3\.([8-9]|1[0-9]) ]]; then
            log_success "Python: $python_version"
        else
            log_error "Python: Versión no compatible ($python_version)"
            all_good=false
        fi
    else
        log_error "Python: No encontrado"
        all_good=false
    fi
    
    # Verificar pip3
    if command -v pip3 &> /dev/null; then
        local pip_version=$(pip3 --version 2>&1)
        log_success "pip3: $pip_version"
    else
        log_error "pip3: No encontrado"
        all_good=false
    fi
    
    # Verificar Node.js
    if command -v node &> /dev/null; then
        local node_version=$(node --version 2>&1)
        if [[ $node_version =~ v(1[8-9]|2[0-9]) ]]; then
            log_success "Node.js: $node_version"
        else
            log_error "Node.js: Versión no compatible ($node_version)"
            all_good=false
        fi
    else
        log_error "Node.js: No encontrado"
        all_good=false
    fi
    
    # Verificar npm
    if command -v npm &> /dev/null; then
        local npm_version=$(npm --version 2>&1)
        log_success "npm: v$npm_version"
    else
        log_error "npm: No encontrado"
        all_good=false
    fi
    
    if [[ "$all_good" == "true" ]]; then
        return 0
    else
        return 1
    fi
}

create_config_directory() {
    log_info "Creando directorio de configuración..."
    
    if [[ ! -d "config" ]]; then
        mkdir -p config
        log_success "Directorio 'config' creado"
    else
        log_success "Directorio 'config' ya existe"
    fi
    
    # Crear subdirectorios necesarios
    local subdirs=("tor_data" "logs")
    for subdir in "${subdirs[@]}"; do
        if [[ ! -d "$subdir" ]]; then
            mkdir -p "$subdir"
            log_success "Directorio '$subdir' creado"
        else
            log_success "Directorio '$subdir' ya existe"
        fi
    done
}

create_env_file() {
    log_info "Configurando archivo .env..."
    
    local env_path=".env"
    local env_example_path=".env.example"
    
    if [[ -f "$env_path" && "$FORCE" != "true" ]]; then
        log_success "Archivo .env ya existe (usa --force para sobrescribir)"
        return
    fi
    
    if [[ -f "$env_example_path" ]]; then
        cp "$env_example_path" "$env_path"
        log_success "Archivo .env creado desde .env.example"
    else
        # Crear .env básico
        cat > "$env_path" << 'EOF'
# ============================================================================
# AEGIS Framework - Configuración de Variables de Entorno
# ============================================================================

# Blockchain Configuration
PRIVATE_KEY=0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef
RPC_URL=http://localhost:8545
NETWORK_ID=1337

# Tor Configuration
TOR_ENABLED=true
TOR_SOCKS_PORT=9050
TOR_CONTROL_PORT=9051
TOR_PASSWORD=

# Security Configuration
ENCRYPTION_KEY=your_encryption_key_here_32_chars
JWT_SECRET=your_jwt_secret_here_at_least_32_characters_long
SESSION_SECRET=your_session_secret_here_at_least_32_characters

# Network Configuration
P2P_PORT=8888
API_PORT=8080
DASHBOARD_PORT=8080
SECURE_CHAT_PORT=5173
BLOCKCHAIN_RPC_PORT=8545

# Database Configuration (if using external DB)
DATABASE_URL=
REDIS_URL=

# Logging Configuration
LOG_LEVEL=INFO
LOG_FILE=logs/aegis.log

# Development Configuration
NODE_ENV=development
DEBUG=false

# Optional: External Services
IPFS_GATEWAY=https://ipfs.io/ipfs/
ETHEREUM_MAINNET_RPC=
POLYGON_RPC=

# Optional: Monitoring
SENTRY_DSN=
PROMETHEUS_PORT=9090
EOF
        
        log_success "Archivo .env creado con configuración básica"
    fi
    
    log_warning "IMPORTANTE: Edita el archivo .env con tus valores específicos"
}

create_app_config_file() {
    log_info "Configurando archivo app_config.json..."
    
    local config_path="config/app_config.json"
    local config_example_path="config/app_config.example.json"
    
    if [[ -f "$config_path" && "$FORCE" != "true" ]]; then
        log_success "Archivo app_config.json ya existe (usa --force para sobrescribir)"
        return
    fi
    
    if [[ -f "$config_example_path" ]]; then
        cp "$config_example_path" "$config_path"
        log_success "Archivo app_config.json creado desde ejemplo"
    else
        # Crear configuración básica
        cat > "$config_path" << 'EOF'
{
  "app": {
    "name": "AEGIS Framework",
    "version": "2.0.0",
    "environment": "development",
    "debug": true
  },
  "server": {
    "host": "localhost",
    "port": 8080,
    "cors_enabled": true,
    "cors_origins": [
      "http://localhost:5173",
      "http://localhost:3000"
    ]
  },
  "security": {
    "jwt_expiration": 3600,
    "session_timeout": 1800,
    "max_login_attempts": 5,
    "password_min_length": 8
  },
  "blockchain": {
    "network": "localhost",
    "rpc_url": "http://localhost:8545",
    "chain_id": 1337,
    "gas_limit": 6721975,
    "gas_price": "20000000000"
  },
  "p2p": {
    "enabled": true,
    "port": 8888,
    "max_peers": 50,
    "discovery_enabled": true
  },
  "tor": {
    "enabled": true,
    "socks_port": 9050,
    "control_port": 9051,
    "data_directory": "./tor_data"
  },
  "logging": {
    "level": "INFO",
    "file": "logs/dashboard.log",
    "max_size": "10MB",
    "backup_count": 5
  },
  "features": {
    "secure_chat": true,
    "blockchain_integration": true,
    "tor_integration": true,
    "p2p_networking": true
  }
}
EOF
        
        log_success "Archivo app_config.json creado con configuración básica"
    fi
}

create_torrc_file() {
    if [[ "$SKIP_TOR" == "true" ]]; then
        log_warning "Omitiendo configuración de Tor"
        return
    fi
    
    log_info "Configurando archivo torrc..."
    
    local torrc_path="config/torrc"
    
    if [[ -f "$torrc_path" && "$FORCE" != "true" ]]; then
        log_success "Archivo torrc ya existe (usa --force para sobrescribir)"
        return
    fi
    
    cat > "$torrc_path" << 'EOF'
# ============================================================================
# AEGIS Framework - Configuración de Tor
# ============================================================================

# Puerto SOCKS para conexiones de aplicaciones
SocksPort 9050

# Puerto de control para gestión programática
ControlPort 9051

# Directorio de datos de Tor
DataDirectory ./tor_data

# Autenticación por cookie (más segura que contraseña)
CookieAuthentication 1

# Archivo de cookie de autenticación
CookieAuthFile ./tor_data/control_auth_cookie

# Configuración de logging
Log notice file ./logs/tor.log

# Configuración de red
# Usar bridges si es necesario (descomenta las siguientes líneas)
# UseBridges 1
# Bridge obfs4 [IP:Puerto] [Fingerprint]

# Configuración de seguridad
# Evitar nodos de salida en ciertos países (opcional)
# ExcludeExitNodes {us},{gb},{au},{ca},{nz},{dk},{fr},{nl},{no},{be}

# Configuración de rendimiento
# Ancho de banda (opcional, en KB/s)
# BandwidthRate 1024 KB
# BandwidthBurst 2048 KB

# Configuración de circuitos
# Tiempo de vida de circuitos (en segundos)
MaxCircuitDirtiness 600

# Número de saltos en circuitos
# PathsNeededToBuildCircuits 0.95

# Configuración de servicios ocultos (si es necesario)
# HiddenServiceDir ./tor_data/hidden_service/
# HiddenServicePort 80 127.0.0.1:8080

# Configuración de cliente
# ClientOnly 1

# Configuración de DNS
# DNSPort 9053
# AutomapHostsOnResolve 1

# Configuración de transparencia (Linux)
# TransPort 9040

# Configuración de seguridad adicional
# DisableAllSwap 1
# HardwareAccel 1

# Configuración de red Tor
# FascistFirewall 1
# FirewallPorts 80,443,9001,9030

# Configuración de directorio de consenso
# DirReqStatistics 0
# EntryStatistics 0
# ExitPortStatistics 0
EOF
    
    log_success "Archivo torrc creado"
    
    # Crear directorio de datos de Tor si no existe
    if [[ ! -d "tor_data" ]]; then
        mkdir -p tor_data
        log_success "Directorio tor_data creado"
    fi
}

create_logs_directory() {
    log_info "Configurando directorio de logs..."
    
    if [[ ! -d "logs" ]]; then
        mkdir -p logs
        log_success "Directorio 'logs' creado"
    else
        log_success "Directorio 'logs' ya existe"
    fi
    
    # Crear archivos de log vacíos
    local log_files=(
        "logs/dashboard.log"
        "logs/secure-chat.log"
        "logs/blockchain.log"
        "logs/tor.log"
        "logs/error.log"
        "logs/access.log"
    )
    
    for log_file in "${log_files[@]}"; do
        if [[ ! -f "$log_file" ]]; then
            touch "$log_file"
        fi
    done
    
    log_success "Archivos de log inicializados"
}

test_configuration_files() {
    log_info "Verificando archivos de configuración..."
    
    local required_files=(
        ".env:Variables de entorno"
        "config/app_config.json:Configuración principal"
    )
    
    if [[ "$SKIP_TOR" != "true" ]]; then
        required_files+=("config/torrc:Configuración de Tor")
    fi
    
    local all_good=true
    
    for file_info in "${required_files[@]}"; do
        local file_path="${file_info%%:*}"
        local description="${file_info##*:}"
        
        if [[ -f "$file_path" ]]; then
            log_success "$description: $file_path"
        else
            log_error "$description: $file_path - No encontrado"
            all_good=false
        fi
    done
    
    # Verificar directorios
    local required_dirs=("config" "logs" "tor_data")
    for dir in "${required_dirs[@]}"; do
        if [[ -d "$dir" ]]; then
            log_success "Directorio: $dir"
        else
            log_error "Directorio: $dir - No encontrado"
            all_good=false
        fi
    done
    
    if [[ "$all_good" == "true" ]]; then
        return 0
    else
        return 1
    fi
}

show_next_steps() {
    echo ""
    log_header "📋 PRÓXIMOS PASOS:"
    echo ""
    echo "1. Edita el archivo .env con tus valores específicos:"
    echo -e "   ${CYAN}nano .env${NC}"
    echo ""
    echo "2. Revisa la configuración principal:"
    echo -e "   ${CYAN}nano config/app_config.json${NC}"
    echo ""
    echo "3. Instala las dependencias de Python:"
    echo -e "   ${CYAN}python3 -m venv venv${NC}"
    echo -e "   ${CYAN}source venv/bin/activate${NC}"
    echo -e "   ${CYAN}pip3 install -r requirements.txt${NC}"
    echo ""
    echo "4. Instala las dependencias de Node.js:"
    echo -e "   ${CYAN}cd dapps/secure-chat/ui && npm install${NC}"
    echo -e "   ${CYAN}cd ../../../dapps/aegis-token && npm install${NC}"
    echo ""
    echo "5. Inicia los servicios:"
    echo -e "   ${CYAN}./scripts/start-all-services.sh${NC}"
    echo ""
}

# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================

main() {
    # Parsear argumentos
    parse_arguments "$@"
    
    log_header "🛡️  AEGIS Framework - Configurador de Archivos"
    log_header "==============================================="
    echo ""
    
    # Verificar prerequisitos
    if ! test_prerequisites; then
        log_error "Los prerequisitos no se cumplen"
        log_warning "Ejecuta primero: ./scripts/install-dependencies.sh"
        exit 1
    fi
    
    echo ""
    log_info "Iniciando configuración de archivos..."
    echo ""
    
    # Crear estructura de directorios
    create_config_directory
    
    # Crear archivos de configuración
    create_env_file
    create_app_config_file
    create_torrc_file
    create_logs_directory
    
    echo ""
    log_info "Verificación final..."
    echo ""
    
    # Verificar configuración
    if test_configuration_files; then
        echo ""
        log_header "📊 RESUMEN DE CONFIGURACIÓN"
        log_header "============================"
        log_success "Todos los archivos de configuración creados correctamente"
        echo ""
        log_success "¡Configuración completada con éxito!"
        show_next_steps
    else
        echo ""
        log_header "📊 RESUMEN DE CONFIGURACIÓN"
        log_header "============================"
        log_warning "Algunos archivos de configuración no se crearon correctamente"
        echo ""
        log_header "💡 RECOMENDACIONES:"
        echo "1. Ejecuta el script nuevamente con --force:"
        echo -e "   ${CYAN}./scripts/setup-config.sh --force${NC}"
        echo "2. Verifica manualmente los archivos faltantes"
        echo "3. Consulta la documentación de solución de problemas"
        echo ""
    fi
    
    log_header "📚 Para más información, consulta:"
    echo "- DEPLOYMENT_GUIDE_COMPLETE.md"
    echo "- DEPENDENCIES_GUIDE.md"
    echo "- TROUBLESHOOTING_GUIDE.md"
    echo ""
}

# Ejecutar función principal
main "$@"