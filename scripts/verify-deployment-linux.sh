#!/bin/bash
# ============================================================================
# AEGIS Framework - Script de Verificación Post-Despliegue (Linux)
# ============================================================================
# Descripción: Verifica que todos los componentes del sistema AEGIS estén
#              correctamente instalados y funcionando después del despliegue
# Autor: AEGIS Security Team
# Versión: 2.0.0
# Fecha: Diciembre 2024
# ============================================================================

set -euo pipefail

# ============================================================================
# CONFIGURACIÓN Y VARIABLES GLOBALES
# ============================================================================

# Colores para output
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly PURPLE='\033[0;35m'
readonly CYAN='\033[0;36m'
readonly WHITE='\033[1;37m'
readonly NC='\033[0m' # No Color

# Configuración por defecto
DETAILED=false
SKIP_SERVICES=false
SKIP_NETWORK=false
CONFIG_PATH="./config"
LOG_PATH="./logs"

# Configuración de verificación
declare -a REQUIRED_PORTS=(8080 5173 8545 9050 9051)
declare -a REQUIRED_SERVICES=("Dashboard" "SecureChat" "Blockchain" "Tor")
declare -a REQUIRED_FILES=(
    "main.py"
    "config/app_config.json"
    "config/torrc"
    "dapps/secure-chat/ui/package.json"
    "dapps/aegis-token/package.json"
)
declare -a REQUIRED_DIRECTORIES=(
    "config"
    "logs"
    "dapps/secure-chat/ui"
    "dapps/aegis-token"
    "venv"
)

# Timeouts
readonly SERVICE_TIMEOUT=30
readonly NETWORK_TIMEOUT=10
readonly HEALTH_TIMEOUT=15

# Contadores de resultados
TESTS_TOTAL=0
TESTS_PASSED=0
TESTS_FAILED=0

# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

print_color() {
    local color=$1
    local message=$2
    local no_newline=${3:-false}
    
    if [[ "$no_newline" == "true" ]]; then
        printf "${color}%s${NC}" "$message"
    else
        printf "${color}%s${NC}\n" "$message"
    fi
}

print_header() {
    local title=$1
    echo
    print_color "$PURPLE" "$(printf '=%.0s' {1..80})"
    print_color "$PURPLE" " $title"
    print_color "$PURPLE" "$(printf '=%.0s' {1..80})"
    echo
}

print_test_result() {
    local test_name=$1
    local success=$2
    local details=${3:-""}
    
    ((TESTS_TOTAL++))
    
    if [[ "$success" == "true" ]]; then
        print_color "$GREEN" "[✅ PASS] $test_name"
        ((TESTS_PASSED++))
    else
        print_color "$RED" "[❌ FAIL] $test_name"
        ((TESTS_FAILED++))
    fi
    
    if [[ -n "$details" && ("$DETAILED" == "true" || "$success" == "false") ]]; then
        print_color "$WHITE" "    └─ $details"
    fi
}

command_exists() {
    command -v "$1" >/dev/null 2>&1
}

test_port_open() {
    local host=${1:-localhost}
    local port=$2
    local timeout=${3:-5}
    
    if command_exists nc; then
        nc -z -w"$timeout" "$host" "$port" >/dev/null 2>&1
    elif command_exists telnet; then
        timeout "$timeout" telnet "$host" "$port" >/dev/null 2>&1
    else
        # Fallback usando /dev/tcp
        timeout "$timeout" bash -c "exec 3<>/dev/tcp/$host/$port" >/dev/null 2>&1
    fi
}

test_http_endpoint() {
    local url=$1
    local timeout=${2:-10}
    
    if command_exists curl; then
        local response
        response=$(curl -s -o /dev/null -w "%{http_code}" --connect-timeout "$timeout" "$url" 2>/dev/null)
        if [[ "$response" =~ ^[2-3][0-9][0-9]$ ]]; then
            echo "true|$response"
        else
            echo "false|$response"
        fi
    elif command_exists wget; then
        if wget --timeout="$timeout" --tries=1 -q --spider "$url" >/dev/null 2>&1; then
            echo "true|200"
        else
            echo "false|000"
        fi
    else
        echo "false|no_tool"
    fi
}

get_process_by_port() {
    local port=$1
    
    if command_exists lsof; then
        lsof -ti:"$port" 2>/dev/null | head -1 | xargs -r ps -p | tail -1 | awk '{print $4}'
    elif command_exists netstat; then
        netstat -tlnp 2>/dev/null | grep ":$port " | awk '{print $7}' | cut -d'/' -f2 | head -1
    else
        echo "unknown"
    fi
}

get_system_info() {
    local info_type=$1
    
    case "$info_type" in
        "os")
            if [[ -f /etc/os-release ]]; then
                . /etc/os-release
                echo "$PRETTY_NAME"
            else
                uname -s
            fi
            ;;
        "kernel")
            uname -r
            ;;
        "arch")
            uname -m
            ;;
        "ram_gb")
            local ram_kb
            ram_kb=$(grep MemTotal /proc/meminfo | awk '{print $2}')
            echo "scale=2; $ram_kb / 1024 / 1024" | bc
            ;;
        "disk_free_gb")
            df . | tail -1 | awk '{print int($4/1024/1024)}'
            ;;
    esac
}

# ============================================================================
# TESTS DE VERIFICACIÓN
# ============================================================================

test_system_requirements() {
    print_header "Verificación de Requisitos del Sistema"
    
    # Verificar distribución Linux
    local os_info
    os_info=$(get_system_info "os")
    local is_supported=true
    
    if [[ "$os_info" =~ (Ubuntu|Debian|CentOS|RHEL|Fedora|openSUSE|Arch) ]]; then
        is_supported=true
    else
        is_supported=false
    fi
    
    print_test_result "Sistema operativo soportado" "$is_supported" "Detectado: $os_info"
    
    # Verificar kernel
    local kernel_version
    kernel_version=$(get_system_info "kernel")
    local kernel_major
    kernel_major=$(echo "$kernel_version" | cut -d'.' -f1)
    local is_kernel_ok=false
    
    if [[ "$kernel_major" -ge 4 ]]; then
        is_kernel_ok=true
    fi
    
    print_test_result "Kernel Linux 4.0+" "$is_kernel_ok" "Versión: $kernel_version"
    
    # Verificar RAM
    local total_ram
    total_ram=$(get_system_info "ram_gb")
    local has_enough_ram=false
    
    if (( $(echo "$total_ram >= 8" | bc -l) )); then
        has_enough_ram=true
    fi
    
    print_test_result "RAM (8GB mínimo)" "$has_enough_ram" "RAM disponible: ${total_ram}GB"
    
    # Verificar espacio en disco
    local free_space
    free_space=$(get_system_info "disk_free_gb")
    local has_enough_space=false
    
    if [[ "$free_space" -ge 10 ]]; then
        has_enough_space=true
    fi
    
    print_test_result "Espacio en disco (10GB mínimo)" "$has_enough_space" "Espacio libre: ${free_space}GB"
    
    # Verificar conectividad a internet
    local has_internet=false
    if ping -c 1 8.8.8.8 >/dev/null 2>&1; then
        has_internet=true
    fi
    
    print_test_result "Conectividad a internet" "$has_internet" "$(if [[ "$has_internet" == "true" ]]; then echo "Disponible"; else echo "No disponible"; fi)"
}

test_dependencies() {
    print_header "Verificación de Dependencias"
    
    # Python
    local python_exists=false
    local python_details=""
    
    if command_exists python3; then
        python_exists=true
        local python_version
        python_version=$(python3 --version 2>&1)
        python_details="$python_version"
        
        # Verificar versión mínima
        local version_check
        version_check=$(python3 -c "import sys; print(sys.version_info >= (3, 8))" 2>/dev/null)
        if [[ "$version_check" != "True" ]]; then
            python_exists=false
            python_details="$python_details (versión < 3.8)"
        fi
    fi
    
    print_test_result "Python 3.8+" "$python_exists" "$python_details"
    
    # pip
    local pip_exists=false
    local pip_details=""
    
    if command_exists pip3; then
        pip_exists=true
        local pip_version
        pip_version=$(pip3 --version 2>&1 | head -1)
        pip_details="$pip_version"
    fi
    
    print_test_result "pip3" "$pip_exists" "$pip_details"
    
    # Node.js
    local node_exists=false
    local node_details=""
    
    if command_exists node; then
        local node_version
        node_version=$(node --version 2>&1)
        local version_major
        version_major=$(echo "$node_version" | sed 's/v//' | cut -d'.' -f1)
        
        if [[ "$version_major" -ge 18 ]]; then
            node_exists=true
            node_details="Versión: $node_version"
        else
            node_exists=false
            node_details="Versión: $node_version (< 18.0)"
        fi
    fi
    
    print_test_result "Node.js 18+" "$node_exists" "$node_details"
    
    # npm
    local npm_exists=false
    local npm_details=""
    
    if command_exists npm; then
        npm_exists=true
        local npm_version
        npm_version=$(npm --version 2>&1)
        npm_details="Versión: $npm_version"
    fi
    
    print_test_result "npm" "$npm_exists" "$npm_details"
    
    # Git
    local git_exists=false
    local git_details=""
    
    if command_exists git; then
        git_exists=true
        local git_version
        git_version=$(git --version 2>&1)
        git_details="$git_version"
    fi
    
    print_test_result "Git" "$git_exists" "$git_details"
    
    # Tor
    local tor_exists=false
    local tor_details=""
    
    if command_exists tor; then
        tor_exists=true
        local tor_version
        tor_version=$(tor --version 2>&1 | head -1)
        tor_details="$tor_version"
    fi
    
    print_test_result "Tor" "$tor_exists" "$tor_details"
}

test_project_structure() {
    print_header "Verificación de Estructura del Proyecto"
    
    # Verificar archivos requeridos
    for file in "${REQUIRED_FILES[@]}"; do
        local exists=false
        if [[ -f "$file" ]]; then
            exists=true
        fi
        
        local details
        if [[ "$exists" == "true" ]]; then
            details="Encontrado"
        else
            details="No encontrado"
        fi
        
        print_test_result "Archivo: $file" "$exists" "$details"
    done
    
    # Verificar directorios requeridos
    for dir in "${REQUIRED_DIRECTORIES[@]}"; do
        local exists=false
        if [[ -d "$dir" ]]; then
            exists=true
        fi
        
        local details
        if [[ "$exists" == "true" ]]; then
            details="Encontrado"
        else
            details="No encontrado"
        fi
        
        print_test_result "Directorio: $dir" "$exists" "$details"
    done
    
    # Verificar entorno virtual Python
    local venv_exists=false
    if [[ -f "venv/bin/python" ]]; then
        venv_exists=true
    fi
    
    local venv_details
    if [[ "$venv_exists" == "true" ]]; then
        venv_details="Configurado correctamente"
    else
        venv_details="No encontrado o mal configurado"
    fi
    
    print_test_result "Entorno virtual Python" "$venv_exists" "$venv_details"
    
    # Verificar node_modules
    local node_modules_ui=false
    if [[ -d "dapps/secure-chat/ui/node_modules" ]]; then
        node_modules_ui=true
    fi
    
    local ui_details
    if [[ "$node_modules_ui" == "true" ]]; then
        ui_details="Instalados"
    else
        ui_details="No instalados"
    fi
    
    print_test_result "Node modules (Secure Chat UI)" "$node_modules_ui" "$ui_details"
    
    local node_modules_token=false
    if [[ -d "dapps/aegis-token/node_modules" ]]; then
        node_modules_token=true
    fi
    
    local token_details
    if [[ "$node_modules_token" == "true" ]]; then
        token_details="Instalados"
    else
        token_details="No instalados"
    fi
    
    print_test_result "Node modules (AEGIS Token)" "$node_modules_token" "$token_details"
}

test_configuration() {
    print_header "Verificación de Configuración"
    
    # Verificar archivo .env
    local env_exists=false
    if [[ -f ".env" ]]; then
        env_exists=true
    fi
    
    local env_details
    if [[ "$env_exists" == "true" ]]; then
        env_details="Encontrado"
    else
        env_details="No encontrado - usar .env.example como base"
    fi
    
    print_test_result "Archivo .env" "$env_exists" "$env_details"
    
    # Verificar configuración de la aplicación
    local app_config_exists=false
    local app_config_valid=false
    
    if [[ -f "$CONFIG_PATH/app_config.json" ]]; then
        app_config_exists=true
        
        # Verificar que es JSON válido y tiene estructura básica
        if jq -e '.dashboard.port' "$CONFIG_PATH/app_config.json" >/dev/null 2>&1; then
            app_config_valid=true
        fi
    fi
    
    local app_config_details
    if [[ "$app_config_valid" == "true" ]]; then
        app_config_details="Válida"
    elif [[ "$app_config_exists" == "true" ]]; then
        app_config_details="Existe pero inválida"
    else
        app_config_details="No encontrada"
    fi
    
    print_test_result "Configuración de la aplicación" "$app_config_valid" "$app_config_details"
    
    # Verificar configuración de Tor
    local torrc_exists=false
    if [[ -f "$CONFIG_PATH/torrc" ]]; then
        torrc_exists=true
    fi
    
    local torrc_details
    if [[ "$torrc_exists" == "true" ]]; then
        torrc_details="Encontrada"
    else
        torrc_details="No encontrada"
    fi
    
    print_test_result "Configuración de Tor" "$torrc_exists" "$torrc_details"
    
    # Verificar directorio de logs
    local logs_exists=false
    if [[ -d "$LOG_PATH" ]]; then
        logs_exists=true
    else
        # Intentar crear el directorio
        if mkdir -p "$LOG_PATH" 2>/dev/null; then
            logs_exists=true
        fi
    fi
    
    local logs_details
    if [[ "$logs_exists" == "true" ]]; then
        logs_details="Disponible"
    else
        logs_details="No se pudo crear"
    fi
    
    print_test_result "Directorio de logs" "$logs_exists" "$logs_details"
}

test_network_ports() {
    print_header "Verificación de Puertos de Red"
    
    if [[ "$SKIP_NETWORK" == "true" ]]; then
        print_color "$YELLOW" "⏭️  Verificación de red omitida por parámetro"
        return 0
    fi
    
    for port in "${REQUIRED_PORTS[@]}"; do
        local is_open=false
        local process=""
        
        if test_port_open "localhost" "$port" "$NETWORK_TIMEOUT"; then
            is_open=true
            process=$(get_process_by_port "$port")
        fi
        
        local details
        if [[ "$is_open" == "true" ]]; then
            if [[ -n "$process" && "$process" != "unknown" ]]; then
                details="Puerto abierto - Proceso: $process"
            else
                details="Puerto abierto"
            fi
        else
            details="Puerto cerrado o no accesible"
        fi
        
        print_test_result "Puerto $port" "$is_open" "$details"
    done
}

test_services() {
    print_header "Verificación de Servicios"
    
    if [[ "$SKIP_SERVICES" == "true" ]]; then
        print_color "$YELLOW" "⏭️  Verificación de servicios omitida por parámetro"
        return 0
    fi
    
    # Dashboard (Puerto 8080)
    local dashboard_result
    dashboard_result=$(test_http_endpoint "http://localhost:8080" "$HEALTH_TIMEOUT")
    local dashboard_success
    dashboard_success=$(echo "$dashboard_result" | cut -d'|' -f1)
    local dashboard_code
    dashboard_code=$(echo "$dashboard_result" | cut -d'|' -f2)
    
    local dashboard_details
    if [[ "$dashboard_success" == "true" ]]; then
        dashboard_details="Respondiendo (HTTP $dashboard_code)"
    else
        dashboard_details="No responde (código: $dashboard_code)"
    fi
    
    print_test_result "Dashboard AEGIS" "$dashboard_success" "$dashboard_details"
    
    # Secure Chat UI (Puerto 5173)
    local chat_result
    chat_result=$(test_http_endpoint "http://localhost:5173" "$HEALTH_TIMEOUT")
    local chat_success
    chat_success=$(echo "$chat_result" | cut -d'|' -f1)
    local chat_code
    chat_code=$(echo "$chat_result" | cut -d'|' -f2)
    
    local chat_details
    if [[ "$chat_success" == "true" ]]; then
        chat_details="Respondiendo (HTTP $chat_code)"
    else
        chat_details="No responde (código: $chat_code)"
    fi
    
    print_test_result "Secure Chat UI" "$chat_success" "$chat_details"
    
    # Blockchain Local (Puerto 8545)
    local blockchain_test=false
    if test_port_open "localhost" 8545 "$NETWORK_TIMEOUT"; then
        blockchain_test=true
    fi
    
    local blockchain_details
    if [[ "$blockchain_test" == "true" ]]; then
        blockchain_details="Activo"
    else
        blockchain_details="No activo"
    fi
    
    print_test_result "Blockchain Local (Hardhat)" "$blockchain_test" "$blockchain_details"
    
    # Tor SOCKS Proxy (Puerto 9050)
    local tor_socks_test=false
    if test_port_open "localhost" 9050 "$NETWORK_TIMEOUT"; then
        tor_socks_test=true
    fi
    
    local tor_socks_details
    if [[ "$tor_socks_test" == "true" ]]; then
        tor_socks_details="Activo"
    else
        tor_socks_details="No activo"
    fi
    
    print_test_result "Tor SOCKS Proxy" "$tor_socks_test" "$tor_socks_details"
    
    # Tor Control Port (Puerto 9051)
    local tor_control_test=false
    if test_port_open "localhost" 9051 "$NETWORK_TIMEOUT"; then
        tor_control_test=true
    fi
    
    local tor_control_details
    if [[ "$tor_control_test" == "true" ]]; then
        tor_control_details="Activo"
    else
        tor_control_details="No activo"
    fi
    
    print_test_result "Tor Control Port" "$tor_control_test" "$tor_control_details"
}

test_python_environment() {
    print_header "Verificación del Entorno Python"
    
    local venv_python="./venv/bin/python"
    local venv_pip="./venv/bin/pip"
    
    if [[ -f "$venv_python" ]]; then
        # Verificar paquetes críticos
        local critical_packages=("flask" "requests" "cryptography" "stem")
        
        for package in "${critical_packages[@]}"; do
            local is_installed=false
            local version=""
            
            if "$venv_pip" show "$package" >/dev/null 2>&1; then
                is_installed=true
                version=$("$venv_pip" show "$package" | grep Version | cut -d' ' -f2)
            fi
            
            local details
            if [[ "$is_installed" == "true" ]]; then
                details="Versión: $version"
            else
                details="No instalado"
            fi
            
            print_test_result "Paquete Python: $package" "$is_installed" "$details"
        done
        
        # Verificar que se puede importar el módulo principal
        local can_import=false
        if "$venv_python" -c "import sys; sys.path.append('.'); import main" >/dev/null 2>&1; then
            can_import=true
        fi
        
        local import_details
        if [[ "$can_import" == "true" ]]; then
            import_details="Exitosa"
        else
            import_details="Falló"
        fi
        
        print_test_result "Importación del módulo principal" "$can_import" "$import_details"
    else
        print_test_result "Entorno virtual Python" "false" "No encontrado en ./venv/bin/python"
    fi
}

test_node_environment() {
    print_header "Verificación del Entorno Node.js"
    
    # Verificar Secure Chat UI
    if [[ -f "dapps/secure-chat/ui/package.json" ]]; then
        pushd "dapps/secure-chat/ui" >/dev/null
        
        # Verificar que las dependencias están instaladas
        local node_modules_exists=false
        if [[ -d "node_modules" ]]; then
            node_modules_exists=true
        fi
        
        local ui_deps_details
        if [[ "$node_modules_exists" == "true" ]]; then
            ui_deps_details="Instaladas"
        else
            ui_deps_details="No instaladas"
        fi
        
        print_test_result "Dependencias Secure Chat UI" "$node_modules_exists" "$ui_deps_details"
        
        # Verificar que el build funciona
        if [[ "$node_modules_exists" == "true" ]]; then
            local build_success=false
            if npm run build >/dev/null 2>&1; then
                build_success=true
            fi
            
            local build_details
            if [[ "$build_success" == "true" ]]; then
                build_details="Exitoso"
            else
                build_details="Falló"
            fi
            
            print_test_result "Build Secure Chat UI" "$build_success" "$build_details"
        fi
        
        popd >/dev/null
    fi
    
    # Verificar AEGIS Token
    if [[ -f "dapps/aegis-token/package.json" ]]; then
        pushd "dapps/aegis-token" >/dev/null
        
        # Verificar que las dependencias están instaladas
        local node_modules_exists=false
        if [[ -d "node_modules" ]]; then
            node_modules_exists=true
        fi
        
        local token_deps_details
        if [[ "$node_modules_exists" == "true" ]]; then
            token_deps_details="Instaladas"
        else
            token_deps_details="No instaladas"
        fi
        
        print_test_result "Dependencias AEGIS Token" "$node_modules_exists" "$token_deps_details"
        
        # Verificar compilación de contratos
        if [[ "$node_modules_exists" == "true" ]]; then
            local compile_success=false
            if npx hardhat compile >/dev/null 2>&1; then
                compile_success=true
            fi
            
            local compile_details
            if [[ "$compile_success" == "true" ]]; then
                compile_details="Exitosa"
            else
                compile_details="Falló"
            fi
            
            print_test_result "Compilación de contratos" "$compile_success" "$compile_details"
        fi
        
        popd >/dev/null
    fi
}

# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================

start_deployment_verification() {
    print_color "$PURPLE" '
 █████╗ ███████╗ ██████╗ ██╗███████╗    ███████╗██████╗  █████╗ ███╗   ███╗███████╗██╗    ██╗ ██████╗ ██████╗ ██╗  ██╗
██╔══██╗██╔════╝██╔════╝ ██║██╔════╝    ██╔════╝██╔══██╗██╔══██╗████╗ ████║██╔════╝██║    ██║██╔═══██╗██╔══██╗██║ ██╔╝
███████║█████╗  ██║  ███╗██║███████╗    █████╗  ██████╔╝███████║██╔████╔██║█████╗  ██║ █╗ ██║██║   ██║██████╔╝█████╔╝ 
██╔══██║██╔══╝  ██║   ██║██║╚════██║    ██╔══╝  ██╔══██╗██╔══██║██║╚██╔╝██║██╔══╝  ██║███╗██║██║   ██║██╔══██╗██╔═██╗ 
██║  ██║███████╗╚██████╔╝██║███████║    ██║     ██║  ██║██║  ██║██║ ╚═╝ ██║███████╗╚███╔███╔╝╚██████╔╝██║  ██║██║  ██╗
╚═╝  ╚═╝╚══════╝ ╚═════╝ ╚═╝╚══════╝    ╚═╝     ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝     ╚═╝╚══════╝ ╚══╝╚══╝  ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝'
    
    print_color "$PURPLE" "Verificación Post-Despliegue - Linux"
    print_color "$CYAN" "Versión: 2.0.0 | $(date '+%Y-%m-%d %H:%M:%S')"
    echo
    
    local start_time
    start_time=$(date +%s)
    
    # Ejecutar todas las verificaciones
    test_system_requirements
    test_dependencies
    test_project_structure
    test_configuration
    test_network_ports
    test_services
    test_python_environment
    test_node_environment
    
    # Calcular resultados finales
    local end_time
    end_time=$(date +%s)
    local duration=$((end_time - start_time))
    local all_tests_passed=false
    
    if [[ "$TESTS_FAILED" -eq 0 ]]; then
        all_tests_passed=true
    fi
    
    # Mostrar resumen final
    print_header "Resumen de Verificación"
    
    print_color "$CYAN" "Tiempo de ejecución: ${duration} segundos"
    print_color "$CYAN" "Tests ejecutados: $TESTS_TOTAL"
    print_color "$GREEN" "Tests exitosos: $TESTS_PASSED"
    print_color "$RED" "Tests fallidos: $TESTS_FAILED"
    echo
    
    if [[ "$all_tests_passed" == "true" ]]; then
        print_color "$GREEN" "🎉 ¡VERIFICACIÓN EXITOSA!"
        print_color "$GREEN" "El sistema AEGIS está correctamente instalado y configurado."
        echo
        print_color "$CYAN" "URLs de acceso:"
        print_color "$CYAN" "  • Dashboard: http://localhost:8080"
        print_color "$CYAN" "  • Secure Chat: http://localhost:5173"
        print_color "$CYAN" "  • Blockchain: http://localhost:8545"
        echo
        print_color "$CYAN" "Para iniciar todos los servicios, ejecuta:"
        print_color "$CYAN" "  ./scripts/start-all-services.sh"
    else
        print_color "$YELLOW" "⚠️  VERIFICACIÓN INCOMPLETA"
        print_color "$YELLOW" "Algunos componentes requieren atención."
        echo
        print_color "$CYAN" "Revisa los errores anteriores y:"
        print_color "$CYAN" "  1. Consulta la documentación de troubleshooting"
        print_color "$CYAN" "  2. Ejecuta los scripts de instalación faltantes"
        print_color "$CYAN" "  3. Verifica la configuración de red y puertos"
        echo
        print_color "$CYAN" "Para más ayuda:"
        print_color "$CYAN" "  ./scripts/verify-deployment-linux.sh --detailed"
    fi
    
    echo
    print_color "$CYAN" "Logs detallados disponibles en: $LOG_PATH"
    
    # Guardar reporte de verificación
    local report_path="$LOG_PATH/deployment-verification-$(date '+%Y%m%d-%H%M%S').json"
    
    # Crear directorio de logs si no existe
    mkdir -p "$LOG_PATH"
    
    # Generar reporte JSON
    cat > "$report_path" << EOF
{
    "timestamp": "$(date '+%Y-%m-%d %H:%M:%S')",
    "version": "2.0.0",
    "duration_seconds": $duration,
    "tests_total": $TESTS_TOTAL,
    "tests_passed": $TESTS_PASSED,
    "tests_failed": $TESTS_FAILED,
    "all_tests_passed": $all_tests_passed,
    "system_info": {
        "os": "$(get_system_info "os")",
        "kernel": "$(get_system_info "kernel")",
        "architecture": "$(get_system_info "arch")",
        "hostname": "$(hostname)",
        "user": "$(whoami)"
    }
}
EOF
    
    print_color "$CYAN" "Reporte guardado en: $report_path"
    
    return $(if [[ "$all_tests_passed" == "true" ]]; then echo 0; else echo 1; fi)
}

# ============================================================================
# PROCESAMIENTO DE ARGUMENTOS
# ============================================================================

show_help() {
    cat << EOF
AEGIS Framework - Script de Verificación Post-Despliegue (Linux)

USAGE:
    $0 [OPTIONS]

OPTIONS:
    -d, --detailed          Mostrar información detallada de todos los tests
    -s, --skip-services     Omitir verificación de servicios en ejecución
    -n, --skip-network      Omitir verificación de puertos de red
    -c, --config PATH       Ruta al directorio de configuración (default: ./config)
    -l, --log PATH          Ruta al directorio de logs (default: ./logs)
    -h, --help              Mostrar esta ayuda

EXAMPLES:
    $0                      Ejecutar verificación estándar
    $0 --detailed           Ejecutar con información detallada
    $0 --skip-services      Omitir verificación de servicios
    $0 -c /opt/aegis/config Usar directorio de configuración personalizado

EXIT CODES:
    0    Todos los tests pasaron exitosamente
    1    Uno o más tests fallaron
    2    Error en argumentos o ejecución

EOF
}

# Procesar argumentos de línea de comandos
while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--detailed)
            DETAILED=true
            shift
            ;;
        -s|--skip-services)
            SKIP_SERVICES=true
            shift
            ;;
        -n|--skip-network)
            SKIP_NETWORK=true
            shift
            ;;
        -c|--config)
            CONFIG_PATH="$2"
            shift 2
            ;;
        -l|--log)
            LOG_PATH="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            print_color "$RED" "Error: Argumento desconocido '$1'"
            print_color "$CYAN" "Usa '$0 --help' para ver la ayuda"
            exit 2
            ;;
    esac
done

# ============================================================================
# EJECUCIÓN PRINCIPAL
# ============================================================================

# Verificar que estamos en el directorio correcto
if [[ ! -f "main.py" ]]; then
    print_color "$RED" "❌ Error: No se encontró main.py en el directorio actual"
    print_color "$CYAN" "Asegúrate de ejecutar este script desde el directorio raíz del proyecto AEGIS"
    exit 2
fi

# Verificar dependencias del script
if ! command_exists bc; then
    print_color "$YELLOW" "⚠️  Advertencia: 'bc' no está instalado. Algunos cálculos pueden fallar."
fi

if ! command_exists jq; then
    print_color "$YELLOW" "⚠️  Advertencia: 'jq' no está instalado. La validación de JSON será limitada."
fi

# Ejecutar verificación principal
if start_deployment_verification; then
    exit 0
else
    exit 1
fi