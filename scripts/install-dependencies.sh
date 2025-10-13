#!/bin/bash

# ============================================================================
# AEGIS Framework - Instalador de Dependencias para Linux
# ============================================================================
# Descripción: Script para instalar únicamente las dependencias necesarias
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
SKIP_DOCKER=false
FORCE=false
VERBOSE=false

# Funciones de utilidad
print_color() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

print_header() {
    print_color "$CYAN" "🛡️  AEGIS Framework - Instalador de Dependencias"
    print_color "$CYAN" "================================================="
    echo ""
}

show_help() {
    print_header
    print_color "$YELLOW" "DESCRIPCIÓN:"
    echo "  Instala únicamente las dependencias necesarias para AEGIS Framework"
    echo ""
    print_color "$YELLOW" "USO:"
    echo "  $0 [OPCIONES]"
    echo ""
    print_color "$YELLOW" "OPCIONES:"
    echo "  --skip-tor      Omitir instalación de Tor"
    echo "  --skip-docker   Omitir instalación de Docker"
    echo "  --force         Forzar reinstalación de dependencias existentes"
    echo "  --verbose       Mostrar salida detallada"
    echo "  --help          Mostrar esta ayuda"
    echo ""
    print_color "$YELLOW" "EJEMPLOS:"
    echo "  $0"
    echo "  $0 --skip-tor --skip-docker"
    echo "  $0 --force --verbose"
    echo ""
    exit 0
}

# Parsear argumentos
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --skip-tor)
                SKIP_TOR=true
                shift
                ;;
            --skip-docker)
                SKIP_DOCKER=true
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
                print_color "$RED" "❌ Opción desconocida: $1"
                echo "Usa --help para ver las opciones disponibles"
                exit 1
                ;;
        esac
    done
}

# Detectar distribución
detect_distro() {
    if [[ -f /etc/os-release ]]; then
        . /etc/os-release
        DISTRO=$ID
        VERSION=$VERSION_ID
    elif [[ -f /etc/redhat-release ]]; then
        DISTRO="rhel"
    elif [[ -f /etc/debian_version ]]; then
        DISTRO="debian"
    else
        print_color "$RED" "❌ No se pudo detectar la distribución de Linux"
        exit 1
    fi
    
    print_color "$GREEN" "✅ Distribución detectada: $DISTRO $VERSION"
}

# Verificar permisos sudo
check_sudo() {
    if ! sudo -n true 2>/dev/null; then
        print_color "$YELLOW" "🔐 Se requieren permisos sudo para la instalación"
        sudo -v
    fi
}

# Verificar requisitos del sistema
check_system_requirements() {
    print_color "$BLUE" "🔍 Verificando requisitos del sistema..."
    
    # Verificar kernel
    local kernel_version=$(uname -r)
    print_color "$GREEN" "✅ Kernel: $kernel_version"
    
    # Verificar RAM
    local ram_mb=$(free -m | awk 'NR==2{printf "%.0f", $2}')
    local ram_gb=$((ram_mb / 1024))
    if [[ $ram_gb -lt 4 ]]; then
        print_color "$YELLOW" "⚠️  Advertencia: RAM insuficiente (${ram_gb} GB). Se recomiendan 4 GB mínimo"
    else
        print_color "$GREEN" "✅ RAM: ${ram_gb} GB - Suficiente"
    fi
    
    # Verificar espacio en disco
    local disk_space=$(df -BG / | awk 'NR==2 {print $4}' | sed 's/G//')
    if [[ $disk_space -lt 2 ]]; then
        print_color "$RED" "❌ Error: Espacio insuficiente en disco (${disk_space} GB). Se requieren 2 GB mínimo"
        exit 1
    fi
    print_color "$GREEN" "✅ Espacio libre: ${disk_space} GB - Suficiente"
    
    # Verificar conexión a internet
    if ping -c 1 google.com &> /dev/null; then
        print_color "$GREEN" "✅ Conexión a internet - OK"
    else
        print_color "$RED" "❌ Error: No hay conexión a internet"
        exit 1
    fi
}

# Actualizar sistema
update_system() {
    print_color "$BLUE" "🔄 Actualizando sistema..."
    
    case $DISTRO in
        ubuntu|debian)
            sudo apt-get update -y
            if [[ $VERBOSE == true ]]; then
                sudo apt-get upgrade -y
            else
                sudo apt-get upgrade -y > /dev/null 2>&1
            fi
            ;;
        centos|rhel|fedora)
            if command -v dnf &> /dev/null; then
                sudo dnf update -y
            else
                sudo yum update -y
            fi
            ;;
        arch)
            sudo pacman -Syu --noconfirm
            ;;
        *)
            print_color "$YELLOW" "⚠️  Distribución no soportada para actualización automática: $DISTRO"
            ;;
    esac
    
    print_color "$GREEN" "✅ Sistema actualizado"
}

# Instalar Python 3.11
install_python() {
    print_color "$BLUE" "🐍 Instalando Python 3.11..."
    
    # Verificar si Python ya está instalado
    if command -v python3 &> /dev/null && [[ $FORCE == false ]]; then
        local python_version=$(python3 --version 2>&1 | grep -oP '\d+\.\d+')
        if [[ $(echo "$python_version >= 3.8" | bc -l) -eq 1 ]]; then
            print_color "$GREEN" "✅ Python ya está instalado: $(python3 --version)"
            return 0
        fi
    fi
    
    case $DISTRO in
        ubuntu|debian)
            sudo apt-get install -y software-properties-common
            sudo add-apt-repository ppa:deadsnakes/ppa -y
            sudo apt-get update
            sudo apt-get install -y python3.11 python3.11-venv python3.11-pip python3.11-dev
            
            # Crear enlaces simbólicos
            sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1
            sudo update-alternatives --install /usr/bin/pip3 pip3 /usr/bin/pip3.11 1
            ;;
        centos|rhel|fedora)
            if command -v dnf &> /dev/null; then
                sudo dnf install -y python3.11 python3.11-pip python3.11-devel
            else
                sudo yum install -y python3.11 python3.11-pip python3.11-devel
            fi
            ;;
        arch)
            sudo pacman -S --noconfirm python python-pip
            ;;
        *)
            print_color "$RED" "❌ Distribución no soportada para instalación automática de Python: $DISTRO"
            return 1
            ;;
    esac
    
    # Verificar instalación
    if command -v python3 &> /dev/null; then
        print_color "$GREEN" "✅ Python instalado: $(python3 --version)"
        
        # Actualizar pip
        print_color "$BLUE" "📦 Actualizando pip..."
        python3 -m pip install --upgrade pip
        print_color "$GREEN" "✅ pip actualizado: $(python3 -m pip --version)"
    else
        print_color "$RED" "❌ Error: Python no se instaló correctamente"
        return 1
    fi
}

# Instalar Node.js 20 LTS
install_nodejs() {
    print_color "$BLUE" "📦 Instalando Node.js 20 LTS..."
    
    # Verificar si Node.js ya está instalado
    if command -v node &> /dev/null && [[ $FORCE == false ]]; then
        local node_version=$(node --version | grep -oP '\d+')
        if [[ $node_version -ge 18 ]]; then
            print_color "$GREEN" "✅ Node.js ya está instalado: $(node --version)"
            return 0
        fi
    fi
    
    # Instalar Node.js usando NodeSource
    print_color "$BLUE" "📥 Descargando script de instalación de NodeSource..."
    curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
    
    case $DISTRO in
        ubuntu|debian)
            sudo apt-get install -y nodejs
            ;;
        centos|rhel|fedora)
            if command -v dnf &> /dev/null; then
                sudo dnf install -y nodejs npm
            else
                sudo yum install -y nodejs npm
            fi
            ;;
        arch)
            sudo pacman -S --noconfirm nodejs npm
            ;;
        *)
            print_color "$RED" "❌ Distribución no soportada para instalación automática de Node.js: $DISTRO"
            return 1
            ;;
    esac
    
    # Verificar instalación
    if command -v node &> /dev/null && command -v npm &> /dev/null; then
        print_color "$GREEN" "✅ Node.js instalado: $(node --version)"
        print_color "$GREEN" "✅ npm instalado: v$(npm --version)"
    else
        print_color "$RED" "❌ Error: Node.js no se instaló correctamente"
        return 1
    fi
}

# Instalar Git
install_git() {
    print_color "$BLUE" "📝 Instalando Git..."
    
    # Verificar si Git ya está instalado
    if command -v git &> /dev/null && [[ $FORCE == false ]]; then
        print_color "$GREEN" "✅ Git ya está instalado: $(git --version)"
        return 0
    fi
    
    case $DISTRO in
        ubuntu|debian)
            sudo apt-get install -y git
            ;;
        centos|rhel|fedora)
            if command -v dnf &> /dev/null; then
                sudo dnf install -y git
            else
                sudo yum install -y git
            fi
            ;;
        arch)
            sudo pacman -S --noconfirm git
            ;;
        *)
            print_color "$RED" "❌ Distribución no soportada para instalación automática de Git: $DISTRO"
            return 1
            ;;
    esac
    
    # Verificar instalación
    if command -v git &> /dev/null; then
        print_color "$GREEN" "✅ Git instalado: $(git --version)"
    else
        print_color "$RED" "❌ Error: Git no se instaló correctamente"
        return 1
    fi
}

# Instalar Tor
install_tor() {
    if [[ $SKIP_TOR == true ]]; then
        print_color "$YELLOW" "⏭️  Omitiendo instalación de Tor"
        return 0
    fi
    
    print_color "$BLUE" "🧅 Instalando Tor..."
    
    # Verificar si Tor ya está instalado
    if command -v tor &> /dev/null && [[ $FORCE == false ]]; then
        local tor_version=$(tor --version | head -n1)
        print_color "$GREEN" "✅ Tor ya está instalado: $tor_version"
        return 0
    fi
    
    case $DISTRO in
        ubuntu|debian)
            sudo apt-get install -y tor
            ;;
        centos|rhel|fedora)
            if command -v dnf &> /dev/null; then
                sudo dnf install -y tor
            else
                sudo yum install -y epel-release
                sudo yum install -y tor
            fi
            ;;
        arch)
            sudo pacman -S --noconfirm tor
            ;;
        *)
            print_color "$YELLOW" "⚠️  Distribución no soportada para instalación automática de Tor: $DISTRO"
            print_color "$CYAN" "ℹ️  Tor es opcional. El sistema funcionará sin él."
            return 0
            ;;
    esac
    
    # Verificar instalación
    if command -v tor &> /dev/null; then
        local tor_version=$(tor --version | head -n1)
        print_color "$GREEN" "✅ Tor instalado: $tor_version"
    else
        print_color "$YELLOW" "⚠️  Tor no se instaló correctamente (opcional)"
        print_color "$CYAN" "ℹ️  El sistema funcionará sin Tor."
    fi
}

# Instalar Docker
install_docker() {
    if [[ $SKIP_DOCKER == true ]]; then
        print_color "$YELLOW" "⏭️  Omitiendo instalación de Docker"
        return 0
    fi
    
    print_color "$BLUE" "🐳 Instalando Docker..."
    
    # Verificar si Docker ya está instalado
    if command -v docker &> /dev/null && [[ $FORCE == false ]]; then
        print_color "$GREEN" "✅ Docker ya está instalado: $(docker --version)"
        return 0
    fi
    
    case $DISTRO in
        ubuntu|debian)
            # Instalar dependencias
            sudo apt-get install -y apt-transport-https ca-certificates curl gnupg lsb-release
            
            # Agregar clave GPG oficial de Docker
            curl -fsSL https://download.docker.com/linux/$DISTRO/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
            
            # Agregar repositorio
            echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/$DISTRO $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
            
            # Instalar Docker
            sudo apt-get update
            sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
            ;;
        centos|rhel|fedora)
            if command -v dnf &> /dev/null; then
                sudo dnf install -y dnf-plugins-core
                sudo dnf config-manager --add-repo https://download.docker.com/linux/fedora/docker-ce.repo
                sudo dnf install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
            else
                sudo yum install -y yum-utils
                sudo yum-config-manager --add-repo https://download.docker.com/linux/centos/docker-ce.repo
                sudo yum install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
            fi
            ;;
        arch)
            sudo pacman -S --noconfirm docker docker-compose
            ;;
        *)
            print_color "$YELLOW" "⚠️  Distribución no soportada para instalación automática de Docker: $DISTRO"
            print_color "$CYAN" "ℹ️  Docker es opcional. El sistema funcionará sin él."
            return 0
            ;;
    esac
    
    # Iniciar y habilitar Docker
    sudo systemctl start docker
    sudo systemctl enable docker
    
    # Agregar usuario al grupo docker
    sudo usermod -aG docker $USER
    
    # Verificar instalación
    if command -v docker &> /dev/null; then
        print_color "$GREEN" "✅ Docker instalado: $(docker --version)"
        print_color "$CYAN" "ℹ️  Nota: Reinicia la sesión para usar Docker sin sudo"
    else
        print_color "$YELLOW" "⚠️  Docker no se instaló correctamente (opcional)"
        print_color "$CYAN" "ℹ️  El sistema funcionará sin Docker."
    fi
}

# Verificar dependencias instaladas
verify_dependencies() {
    print_color "$BLUE" "🔍 Verificando dependencias instaladas..."
    
    local all_good=true
    
    # Verificar Python
    if command -v python3 &> /dev/null; then
        local python_version=$(python3 --version)
        print_color "$GREEN" "✅ Python: $python_version"
    else
        print_color "$RED" "❌ Python: No encontrado"
        all_good=false
    fi
    
    # Verificar pip
    if python3 -m pip --version &> /dev/null; then
        local pip_version=$(python3 -m pip --version)
        print_color "$GREEN" "✅ pip: $pip_version"
    else
        print_color "$RED" "❌ pip: No encontrado"
        all_good=false
    fi
    
    # Verificar Node.js
    if command -v node &> /dev/null; then
        local node_version=$(node --version)
        print_color "$GREEN" "✅ Node.js: $node_version"
    else
        print_color "$RED" "❌ Node.js: No encontrado"
        all_good=false
    fi
    
    # Verificar npm
    if command -v npm &> /dev/null; then
        local npm_version=$(npm --version)
        print_color "$GREEN" "✅ npm: v$npm_version"
    else
        print_color "$RED" "❌ npm: No encontrado"
        all_good=false
    fi
    
    # Verificar Git
    if command -v git &> /dev/null; then
        local git_version=$(git --version)
        print_color "$GREEN" "✅ Git: $git_version"
    else
        print_color "$RED" "❌ Git: No encontrado"
        all_good=false
    fi
    
    # Verificar Tor (opcional)
    if [[ $SKIP_TOR == false ]]; then
        if command -v tor &> /dev/null; then
            local tor_version=$(tor --version | head -n1)
            print_color "$GREEN" "✅ Tor: $tor_version"
        else
            print_color "$YELLOW" "⚠️  Tor: No encontrado (opcional)"
        fi
    fi
    
    # Verificar Docker (opcional)
    if [[ $SKIP_DOCKER == false ]]; then
        if command -v docker &> /dev/null; then
            local docker_version=$(docker --version)
            print_color "$GREEN" "✅ Docker: $docker_version"
        else
            print_color "$YELLOW" "⚠️  Docker: No encontrado (opcional)"
        fi
    fi
    
    if [[ $all_good == true ]]; then
        return 0
    else
        return 1
    fi
}

# Función principal
main() {
    parse_args "$@"
    
    print_header
    
    # Verificar permisos sudo
    check_sudo
    
    # Detectar distribución
    detect_distro
    
    # Verificar requisitos del sistema
    check_system_requirements
    
    echo ""
    print_color "$BLUE" "🚀 Iniciando instalación de dependencias..."
    echo ""
    
    # Actualizar sistema
    update_system
    
    echo ""
    
    # Instalar dependencias principales
    local installation_results=()
    
    install_python && installation_results+=("python:success") || installation_results+=("python:failed")
    install_nodejs && installation_results+=("nodejs:success") || installation_results+=("nodejs:failed")
    install_git && installation_results+=("git:success") || installation_results+=("git:failed")
    install_tor && installation_results+=("tor:success") || installation_results+=("tor:failed")
    install_docker && installation_results+=("docker:success") || installation_results+=("docker:failed")
    
    echo ""
    print_color "$BLUE" "🔍 Verificación final..."
    echo ""
    
    # Verificar todas las dependencias
    if verify_dependencies; then
        echo ""
        print_color "$CYAN" "📊 RESUMEN DE INSTALACIÓN"
        print_color "$CYAN" "========================="
        print_color "$GREEN" "✅ Todas las dependencias principales instaladas correctamente"
        echo ""
        print_color "$GREEN" "🎉 ¡Instalación completada con éxito!"
        echo ""
        print_color "$YELLOW" "📋 PRÓXIMOS PASOS:"
        echo "1. Ejecuta el script de configuración:"
        print_color "$CYAN" "   ./scripts/setup-config.sh"
        echo "2. O ejecuta el despliegue completo:"
        print_color "$CYAN" "   ./scripts/auto-deploy-linux.sh"
        echo ""
    else
        echo ""
        print_color "$CYAN" "📊 RESUMEN DE INSTALACIÓN"
        print_color "$CYAN" "========================="
        print_color "$YELLOW" "⚠️  Algunas dependencias no se instalaron correctamente"
        echo ""
        print_color "$YELLOW" "💡 RECOMENDACIONES:"
        echo "1. Reinicia el sistema"
        echo "2. Ejecuta el script nuevamente con --force:"
        print_color "$CYAN" "   $0 --force"
        echo "3. Verifica manualmente las dependencias faltantes"
        echo ""
    fi
    
    print_color "$CYAN" "📚 Para más información, consulta:"
    echo "- DEPLOYMENT_GUIDE_COMPLETE.md"
    echo "- DEPENDENCIES_GUIDE.md"
    echo "- TROUBLESHOOTING_GUIDE.md"
    echo ""
}

# Ejecutar función principal
main "$@"