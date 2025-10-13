#!/bin/bash

# ========================================
# AEGIS Framework - Auto Deploy Linux
# ========================================
# Autor: AEGIS Security Team
# Versión: 2.0.0
# Fecha: Diciembre 2024
# ========================================

set -euo pipefail

# Configuración de colores
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Variables globales
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
REPO_URL="https://github.com/KaseMaster/Open-A.G.I.git"
MODE="full"
SKIP_TOR=false
VERBOSE=false
FORCE=false
DISTRO=""
PACKAGE_MANAGER=""

# Funciones de utilidad
log_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

log_error() {
    echo -e "${RED}❌ $1${NC}" >&2
}

log_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

log_info() {
    echo -e "${CYAN}ℹ️  $1${NC}"
}

log_header() {
    echo ""
    echo -e "${MAGENTA}🚀 $1${NC}"
    echo -e "${MAGENTA}$(printf '=%.0s' $(seq 1 $((${#1} + 4))))${NC}"
}

# Mostrar ayuda
show_help() {
    cat << EOF
AEGIS Framework - Auto Deploy Linux v2.0.0

Uso: $0 [OPCIONES]

OPCIONES:
    --mode MODE         Modo de instalación: full, deps, config, dev, prod (default: full)
    --skip-tor          Omitir instalación de Tor
    --verbose           Mostrar salida detallada
    --force             Continuar ante errores no críticos
    --help              Mostrar esta ayuda

MODOS:
    full                Instalación completa (default)
    deps                Solo instalar dependencias
    config              Solo configurar archivos
    dev                 Instalación para desarrollo
    prod                Instalación para producción

EJEMPLOS:
    $0                          # Instalación completa
    $0 --mode deps              # Solo dependencias
    $0 --mode dev --verbose     # Desarrollo con salida detallada
    $0 --skip-tor --force       # Sin Tor, continuar ante errores

EOF
}

# Parsear argumentos
parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --mode)
                MODE="$2"
                shift 2
                ;;
            --skip-tor)
                SKIP_TOR=true
                shift
                ;;
            --verbose)
                VERBOSE=true
                shift
                ;;
            --force)
                FORCE=true
                shift
                ;;
            --help)
                show_help
                exit 0
                ;;
            *)
                log_error "Opción desconocida: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    # Validar modo
    case $MODE in
        full|deps|config|dev|prod)
            ;;
        *)
            log_error "Modo inválido: $MODE"
            show_help
            exit 1
            ;;
    esac
}

# Detectar distribución Linux
detect_distro() {
    log_header "Detectando Distribución Linux"
    
    if [[ -f /etc/os-release ]]; then
        . /etc/os-release
        DISTRO=$ID
        
        case $DISTRO in
            ubuntu|debian)
                PACKAGE_MANAGER="apt"
                log_success "Distribución detectada: $PRETTY_NAME"
                log_success "Gestor de paquetes: APT"
                ;;
            centos|rhel|fedora)
                if command -v dnf &> /dev/null; then
                    PACKAGE_MANAGER="dnf"
                else
                    PACKAGE_MANAGER="yum"
                fi
                log_success "Distribución detectada: $PRETTY_NAME"
                log_success "Gestor de paquetes: ${PACKAGE_MANAGER^^}"
                ;;
            arch)
                PACKAGE_MANAGER="pacman"
                log_success "Distribución detectada: $PRETTY_NAME"
                log_success "Gestor de paquetes: Pacman"
                ;;
            *)
                log_warning "Distribución no reconocida: $DISTRO"
                log_warning "Intentando con APT como fallback"
                PACKAGE_MANAGER="apt"
                ;;
        esac
    else
        log_error "No se pudo detectar la distribución Linux"
        return 1
    fi
    
    return 0
}

# Verificar permisos de sudo
check_sudo() {
    if [[ $EUID -eq 0 ]]; then
        log_warning "Ejecutándose como root"
        return 0
    fi
    
    if ! sudo -n true 2>/dev/null; then
        log_info "Se requieren permisos de sudo para la instalación"
        sudo -v || {
            log_error "No se pudieron obtener permisos de sudo"
            return 1
        }
    fi
    
    log_success "Permisos de sudo verificados"
    return 0
}

# Verificar requisitos del sistema
check_system_requirements() {
    log_header "Verificando Requisitos del Sistema"
    
    # Verificar kernel Linux
    local kernel_version=$(uname -r | cut -d. -f1,2)
    local kernel_major=$(echo $kernel_version | cut -d. -f1)
    local kernel_minor=$(echo $kernel_version | cut -d. -f2)
    
    if [[ $kernel_major -lt 5 ]] || [[ $kernel_major -eq 5 && $kernel_minor -lt 4 ]]; then
        log_warning "Kernel Linux $kernel_version detectado. Se recomienda 5.4+"
    else
        log_success "Kernel Linux: $kernel_version ✓"
    fi
    
    # Verificar memoria RAM
    local total_ram=$(free -g | awk '/^Mem:/{print $2}')
    if [[ $total_ram -lt 8 ]]; then
        log_warning "RAM disponible: ${total_ram}GB. Se recomienda 8GB o más."
    else
        log_success "RAM disponible: ${total_ram}GB ✓"
    fi
    
    # Verificar espacio en disco
    local free_space=$(df -BG . | awk 'NR==2 {print $4}' | sed 's/G//')
    if [[ $free_space -lt 10 ]]; then
        log_error "Espacio libre insuficiente: ${free_space}GB. Se requieren al menos 10GB."
        return 1
    else
        log_success "Espacio libre: ${free_space}GB ✓"
    fi
    
    # Verificar conexión a internet
    if ping -c 1 google.com &> /dev/null; then
        log_success "Conexión a internet: ✓"
    else
        log_error "No hay conexión a internet"
        return 1
    fi
    
    return 0
}

# Actualizar sistema
update_system() {
    log_header "Actualizando Sistema"
    
    case $PACKAGE_MANAGER in
        apt)
            sudo apt update && sudo apt upgrade -y
            sudo apt install -y curl wget git build-essential software-properties-common
            ;;
        yum)
            sudo yum update -y
            sudo yum groupinstall -y "Development Tools"
            sudo yum install -y curl wget git
            ;;
        dnf)
            sudo dnf update -y
            sudo dnf groupinstall -y "Development Tools"
            sudo dnf install -y curl wget git
            ;;
        pacman)
            sudo pacman -Syu --noconfirm
            sudo pacman -S --noconfirm base-devel curl wget git
            ;;
    esac
    
    log_success "Sistema actualizado correctamente"
}

# Instalar Python 3.11
install_python() {
    log_header "Instalando Python 3.11"
    
    if command -v python3.11 &> /dev/null; then
        log_success "Python 3.11 ya está instalado"
        return 0
    fi
    
    case $PACKAGE_MANAGER in
        apt)
            # Agregar repositorio deadsnakes para Ubuntu/Debian
            sudo add-apt-repository ppa:deadsnakes/ppa -y
            sudo apt update
            sudo apt install -y python3.11 python3.11-pip python3.11-venv python3.11-dev
            ;;
        yum|dnf)
            # Compilar desde fuente para CentOS/RHEL
            local python_version="3.11.7"
            cd /tmp
            wget https://www.python.org/ftp/python/${python_version}/Python-${python_version}.tgz
            tar xzf Python-${python_version}.tgz
            cd Python-${python_version}
            ./configure --enable-optimizations
            make altinstall
            cd "$PROJECT_ROOT"
            ;;
        pacman)
            sudo pacman -S --noconfirm python python-pip
            ;;
    esac
    
    # Crear enlaces simbólicos
    if [[ ! -L /usr/local/bin/python3 ]]; then
        sudo ln -sf $(which python3.11) /usr/local/bin/python3
    fi
    
    if [[ ! -L /usr/local/bin/pip3 ]]; then
        sudo ln -sf $(which pip3.11) /usr/local/bin/pip3
    fi
    
    log_success "Python 3.11 instalado correctamente"
}

# Instalar Node.js 20 LTS
install_nodejs() {
    log_header "Instalando Node.js 20 LTS"
    
    if command -v node &> /dev/null; then
        local node_version=$(node --version | cut -d'v' -f2 | cut -d'.' -f1)
        if [[ $node_version -ge 18 ]]; then
            log_success "Node.js $(node --version) ya está instalado"
            return 0
        fi
    fi
    
    # Instalar usando NodeSource
    curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
    
    case $PACKAGE_MANAGER in
        apt)
            sudo apt-get install -y nodejs
            ;;
        yum|dnf)
            sudo $PACKAGE_MANAGER install -y nodejs npm
            ;;
        pacman)
            sudo pacman -S --noconfirm nodejs npm
            ;;
    esac
    
    log_success "Node.js $(node --version) instalado correctamente"
}

# Instalar Tor
install_tor() {
    if [[ $SKIP_TOR == true ]]; then
        log_info "Omitiendo instalación de Tor"
        return 0
    fi
    
    log_header "Instalando Tor"
    
    if command -v tor &> /dev/null; then
        log_success "Tor ya está instalado"
        return 0
    fi
    
    case $PACKAGE_MANAGER in
        apt)
            # Agregar repositorio oficial de Tor
            sudo apt install -y apt-transport-https
            curl -s https://deb.torproject.org/torproject.org/A3C4F0F979CAA22CDBA8F512EE8CBC9E886DDD89.asc | sudo apt-key add -
            echo "deb https://deb.torproject.org/torproject.org $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/tor.list
            sudo apt update
            sudo apt install -y tor deb.torproject.org-keyring
            ;;
        yum|dnf)
            sudo $PACKAGE_MANAGER install -y epel-release
            sudo $PACKAGE_MANAGER install -y tor
            ;;
        pacman)
            sudo pacman -S --noconfirm tor
            ;;
    esac
    
    # Configurar servicio Tor
    sudo systemctl enable tor
    
    log_success "Tor instalado correctamente"
}

# Instalar Docker (opcional)
install_docker() {
    log_header "Instalando Docker (Opcional)"
    
    if command -v docker &> /dev/null; then
        log_success "Docker ya está instalado"
        return 0
    fi
    
    # Instalar Docker usando el script oficial
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    
    # Agregar usuario al grupo docker
    sudo usermod -aG docker $USER
    
    # Instalar Docker Compose
    sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose
    
    log_success "Docker instalado correctamente"
    log_info "Reinicia la sesión para usar Docker sin sudo"
}

# Verificar instalaciones
verify_dependencies() {
    log_header "Verificando Instalaciones"
    
    local commands=(
        "python3:Python"
        "pip3:Pip"
        "node:Node.js"
        "npm:NPM"
        "git:Git"
    )
    
    if [[ $SKIP_TOR == false ]]; then
        commands+=("tor:Tor")
    fi
    
    local all_ok=true
    
    for cmd_info in "${commands[@]}"; do
        local cmd=$(echo $cmd_info | cut -d: -f1)
        local desc=$(echo $cmd_info | cut -d: -f2)
        
        if command -v $cmd &> /dev/null; then
            local version=$($cmd --version 2>/dev/null | head -n1 || echo "instalado")
            log_success "$desc: $version"
        else
            log_error "$desc no está disponible"
            all_ok=false
        fi
    done
    
    if [[ $all_ok == false ]]; then
        return 1
    fi
    
    return 0
}

# Configurar proyecto
setup_project() {
    log_header "Configurando Proyecto AEGIS"
    
    cd "$PROJECT_ROOT"
    
    # Verificar si estamos en el directorio correcto
    if [[ ! -f "main.py" ]]; then
        log_error "No se encontró main.py. Asegúrate de estar en el directorio raíz del proyecto."
        return 1
    fi
    
    # Crear entorno virtual Python
    log_info "Creando entorno virtual Python..."
    python3 -m venv venv
    
    # Activar entorno virtual
    log_info "Activando entorno virtual..."
    source venv/bin/activate
    
    # Actualizar pip
    pip install --upgrade pip
    
    # Instalar dependencias Python
    log_info "Instalando dependencias Python..."
    pip install -r requirements.txt
    
    if [[ -f "requirements-dev.txt" ]]; then
        pip install -r requirements-dev.txt
    fi
    
    log_success "Dependencias Python instaladas"
    
    # Instalar dependencias Node.js para secure-chat
    if [[ -f "dapps/secure-chat/ui/package.json" ]]; then
        log_info "Instalando dependencias de Secure Chat UI..."
        cd dapps/secure-chat/ui
        npm install
        cd "$PROJECT_ROOT"
        log_success "Dependencias de Secure Chat UI instaladas"
    fi
    
    # Instalar dependencias Node.js para aegis-token
    if [[ -f "dapps/aegis-token/package.json" ]]; then
        log_info "Instalando dependencias de AEGIS Token..."
        cd dapps/aegis-token
        npm install
        cd "$PROJECT_ROOT"
        log_success "Dependencias de AEGIS Token instaladas"
    fi
    
    return 0
}

# Configurar archivos de configuración
setup_configuration() {
    log_header "Configurando Archivos de Configuración"
    
    cd "$PROJECT_ROOT"
    
    # Crear directorio de configuración si no existe
    mkdir -p config
    
    # Copiar archivos de ejemplo
    if [[ -f ".env.example" && ! -f ".env" ]]; then
        cp .env.example .env
        log_success "Archivo .env creado desde .env.example"
    fi
    
    if [[ -f "config/config.example.yml" && ! -f "config/config.yml" ]]; then
        cp config/config.example.yml config/config.yml
        log_success "Archivo config.yml creado desde config.example.yml"
    fi
    
    # Generar configuración de Tor si no existe
    if [[ ! -f "config/torrc" ]]; then
        cat > config/torrc << 'EOF'
# Configuración Tor para AEGIS
ControlPort 9051
HashedControlPassword 16:872860B76453A77D60CA2BB8C1A7042072093276A3D701AD684053EC4C

# Servicio oculto
HiddenServiceDir ./tor_data/aegis_service/
HiddenServicePort 80 127.0.0.1:8080
HiddenServicePort 3000 127.0.0.1:3000

# Configuración de rendimiento
NumEntryGuards 8
CircuitBuildTimeout 30
LearnCircuitBuildTimeout 0
EOF
        log_success "Archivo torrc creado"
    fi
    
    # Crear directorio de datos Tor
    mkdir -p tor_data/aegis_service
    chmod 700 tor_data/aegis_service
    log_success "Directorio de datos Tor creado"
    
    # Crear directorio de logs
    mkdir -p logs
    log_success "Directorio de logs creado"
    
    # Configurar permisos
    chmod +x scripts/*.sh 2>/dev/null || true
    
    return 0
}

# Ejecutar pruebas de verificación
run_tests() {
    log_header "Ejecutando Pruebas de Verificación"
    
    cd "$PROJECT_ROOT"
    
    # Activar entorno virtual
    source venv/bin/activate
    
    # Ejecutar pruebas básicas
    if [[ -f "scripts/system_validation_test.py" ]]; then
        log_info "Ejecutando pruebas de validación del sistema..."
        python scripts/system_validation_test.py
    fi
    
    # Probar importaciones críticas
    log_info "Verificando importaciones Python críticas..."
    python -c "
import sys
import flask
import requests
import cryptography
print('✅ Todas las importaciones críticas funcionan correctamente')
"
    
    log_success "Todas las pruebas pasaron correctamente"
    return 0
}

# Mostrar información de finalización
show_completion_info() {
    log_header "Instalación Completada"
    
    log_success "AEGIS Framework ha sido instalado correctamente en Linux"
    echo ""
    
    log_info "Para iniciar los servicios, ejecuta los siguientes comandos:"
    echo ""
    echo -e "${YELLOW}# Terminal 1 - Activar entorno y dashboard:${NC}"
    echo "cd $PROJECT_ROOT"
    echo "source venv/bin/activate"
    echo "python main.py start-dashboard --config ./config/app_config.json"
    echo ""
    echo -e "${YELLOW}# Terminal 2 - Tor Service:${NC}"
    echo "tor -f ./config/torrc"
    echo ""
    echo -e "${YELLOW}# Terminal 3 - Secure Chat UI:${NC}"
    echo "cd dapps/secure-chat/ui"
    echo "npm run dev"
    echo ""
    echo -e "${YELLOW}# Terminal 4 - Blockchain Local:${NC}"
    echo "cd dapps/aegis-token"
    echo "npx hardhat node"
    echo ""
    
    log_info "URLs de acceso:"
    echo "• Dashboard: http://localhost:8080"
    echo "• Secure Chat: http://localhost:5173"
    echo "• Blockchain: http://localhost:8545"
    echo ""
    
    log_info "Documentación adicional:"
    echo "• Guía completa: docs/DEPLOYMENT_GUIDE_COMPLETE.md"
    echo "• Troubleshooting: docs/TROUBLESHOOTING.md"
    echo "• Configuración: docs/CONFIGURATION.md"
    echo ""
    
    log_info "Scripts de utilidad:"
    echo "• Verificación: ./scripts/verify-installation.sh"
    echo "• Inicio rápido: ./scripts/start-services.sh"
    echo "• Detener servicios: ./scripts/stop-services.sh"
}

# Función principal
main() {
    log_header "AEGIS Framework - Auto Deploy Linux v2.0.0"
    
    # Parsear argumentos
    parse_arguments "$@"
    
    log_info "Modo de instalación: $MODE"
    [[ $SKIP_TOR == true ]] && log_info "Tor será omitido"
    [[ $VERBOSE == true ]] && log_info "Modo verbose activado"
    [[ $FORCE == true ]] && log_info "Modo force activado - continuará ante errores"
    
    # Detectar distribución
    detect_distro || exit 1
    
    # Verificar permisos
    check_sudo || exit 1
    
    # Verificar requisitos del sistema
    check_system_requirements || exit 1
    
    # Ejecutar según el modo seleccionado
    case $MODE in
        "deps")
            update_system || exit 1
            install_python || exit 1
            install_nodejs || exit 1
            install_tor || exit 1
            verify_dependencies || exit 1
            ;;
        "config")
            setup_configuration || exit 1
            ;;
        "dev")
            update_system || exit 1
            install_python || exit 1
            install_nodejs || exit 1
            install_tor || exit 1
            verify_dependencies || exit 1
            setup_project || exit 1
            setup_configuration || exit 1
            # En modo dev, no ejecutar pruebas automáticas
            ;;
        "prod")
            update_system || exit 1
            install_python || exit 1
            install_nodejs || exit 1
            install_tor || exit 1
            install_docker || exit 1
            verify_dependencies || exit 1
            setup_project || exit 1
            setup_configuration || exit 1
            run_tests || exit 1
            ;;
        *) # "full"
            update_system || exit 1
            install_python || exit 1
            install_nodejs || exit 1
            install_tor || exit 1
            verify_dependencies || exit 1
            setup_project || exit 1
            setup_configuration || exit 1
            run_tests || exit 1
            ;;
    esac
    
    show_completion_info
    log_success "¡Instalación completada exitosamente!"
}

# Manejo de errores
trap 'log_error "Error inesperado en línea $LINENO. Para soporte, consulta: docs/TROUBLESHOOTING.md"; exit 1' ERR

# Ejecutar función principal
main "$@"