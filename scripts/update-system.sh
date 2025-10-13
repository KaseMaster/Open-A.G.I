#!/bin/bash

# ============================================================================
# AEGIS Framework - Script de Actualización del Sistema para Linux
# ============================================================================
# Descripción: Script para actualizar dependencias y componentes del sistema
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
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LOG_FILE="$PROJECT_ROOT/logs/update-system.log"

# Opciones por defecto
CHECK_ONLY=false
UPDATE_PYTHON=false
UPDATE_NODEJS=false
UPDATE_SYSTEM=false
UPDATE_ALL=false
FORCE_UPDATE=false
CREATE_BACKUP=false
VERBOSE=false

function print_color() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

function log_message() {
    local level=$1
    local message=$2
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    # Crear directorio de logs si no existe
    mkdir -p "$(dirname "$LOG_FILE")"
    
    echo "[$timestamp] [$level] $message" >> "$LOG_FILE"
    
    if [[ "$VERBOSE" == "true" ]]; then
        print_color "$BLUE" "[$level] $message"
    fi
}

function show_help() {
    print_color "$CYAN" "🛡️  AEGIS Framework - Actualizador del Sistema"
    print_color "$CYAN" "==============================================="
    echo ""
    print_color "$YELLOW" "DESCRIPCIÓN:"
    echo "  Actualiza dependencias y componentes del sistema AEGIS"
    echo ""
    print_color "$YELLOW" "USO:"
    echo "  ./update-system.sh [OPCIONES]"
    echo ""
    print_color "$YELLOW" "OPCIONES:"
    echo "  --check-only          Solo verificar actualizaciones disponibles"
    echo "  --python              Actualizar dependencias de Python"
    echo "  --nodejs              Actualizar dependencias de Node.js"
    echo "  --system              Actualizar herramientas del sistema"
    echo "  --all                 Actualizar todo (Python + Node.js + Sistema)"
    echo "  --force               Forzar reinstalación de dependencias"
    echo "  --backup              Crear respaldo antes de actualizar"
    echo "  --verbose             Mostrar información detallada"
    echo "  --help                Mostrar esta ayuda"
    echo ""
    print_color "$YELLOW" "EJEMPLOS:"
    echo "  ./update-system.sh --check-only                        # Solo verificar"
    echo "  ./update-system.sh --python                           # Actualizar Python"
    echo "  ./update-system.sh --nodejs                           # Actualizar Node.js"
    echo "  ./update-system.sh --all --backup                     # Actualizar todo con respaldo"
    echo "  ./update-system.sh --system --force                   # Forzar actualización del sistema"
    echo ""
    exit 0
}

function parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --check-only)
                CHECK_ONLY=true
                shift
                ;;
            --python)
                UPDATE_PYTHON=true
                shift
                ;;
            --nodejs)
                UPDATE_NODEJS=true
                shift
                ;;
            --system)
                UPDATE_SYSTEM=true
                shift
                ;;
            --all)
                UPDATE_ALL=true
                shift
                ;;
            --force)
                FORCE_UPDATE=true
                shift
                ;;
            --backup)
                CREATE_BACKUP=true
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

function test_prerequisites() {
    print_color "$BLUE" "🔍 Verificando prerrequisitos..."
    log_message "INFO" "Iniciando verificación de prerrequisitos"
    
    # Cambiar al directorio del proyecto
    cd "$PROJECT_ROOT"
    
    # Verificar si estamos en el directorio correcto
    if [[ ! -f "main.py" ]]; then
        print_color "$RED" "❌ Error: No se encontró main.py. Ejecuta este script desde el directorio raíz del proyecto AEGIS."
        log_message "ERROR" "main.py no encontrado en $PROJECT_ROOT"
        exit 1
    fi
    
    # Verificar Bash
    if [[ "${BASH_VERSION%%.*}" -lt 4 ]]; then
        print_color "$RED" "❌ Error: Se requiere Bash 4.0 o superior"
        log_message "ERROR" "Versión de Bash insuficiente: $BASH_VERSION"
        exit 1
    fi
    
    # Verificar herramientas básicas
    local tools=("curl" "wget" "jq")
    for tool in "${tools[@]}"; do
        if ! command -v "$tool" &> /dev/null; then
            print_color "$YELLOW" "⚠️  Advertencia: $tool no está instalado. Algunas funciones pueden no estar disponibles."
            log_message "WARN" "$tool no está disponible"
        fi
    done
    
    print_color "$GREEN" "✅ Prerrequisitos verificados"
    log_message "INFO" "Prerrequisitos verificados exitosamente"
}

function test_internet_connection() {
    print_color "$BLUE" "🌐 Verificando conexión a internet..."
    log_message "INFO" "Verificando conexión a internet"
    
    if curl -s --max-time 10 https://www.google.com > /dev/null 2>&1; then
        print_color "$GREEN" "✅ Conexión a internet verificada"
        log_message "INFO" "Conexión a internet verificada"
        return 0
    else
        print_color "$RED" "❌ Error: No hay conexión a internet"
        log_message "ERROR" "Sin conexión a internet"
        return 1
    fi
}

function get_python_updates() {
    print_color "$BLUE" "🐍 Verificando actualizaciones de Python..."
    log_message "INFO" "Verificando actualizaciones de Python"
    
    local updates_file="/tmp/aegis_python_updates.json"
    echo "[]" > "$updates_file"
    
    # Verificar si existe requirements.txt
    if [[ -f "requirements.txt" ]]; then
        # Activar entorno virtual si existe
        if [[ -f "venv/bin/activate" ]]; then
            source venv/bin/activate
        fi
        
        # Verificar paquetes desactualizados
        if command -v pip &> /dev/null; then
            local outdated_output
            if outdated_output=$(pip list --outdated --format=json 2>/dev/null); then
                echo "$outdated_output" > "$updates_file"
                local count=$(echo "$outdated_output" | jq length 2>/dev/null || echo "0")
                log_message "INFO" "Encontradas $count actualizaciones de Python"
            else
                log_message "WARN" "Error verificando paquetes de Python"
            fi
        else
            log_message "WARN" "pip no está disponible"
        fi
    else
        log_message "INFO" "requirements.txt no encontrado"
    fi
    
    echo "$updates_file"
}

function get_nodejs_updates() {
    print_color "$BLUE" "📦 Verificando actualizaciones de Node.js..."
    log_message "INFO" "Verificando actualizaciones de Node.js"
    
    local updates_file="/tmp/aegis_nodejs_updates.json"
    echo "{}" > "$updates_file"
    
    local projects=("dapps/secure-chat/ui" "dapps/aegis-token")
    
    for project in "${projects[@]}"; do
        if [[ -f "$project/package.json" ]]; then
            log_message "INFO" "Verificando actualizaciones en $project"
            
            pushd "$project" > /dev/null 2>&1 || continue
            
            if command -v npm &> /dev/null; then
                local outdated_output
                if outdated_output=$(npm outdated --json 2>/dev/null); then
                    # Combinar resultados
                    local temp_file="/tmp/temp_updates.json"
                    jq --arg project "$project" '. as $updates | {($project): $updates}' <<< "$outdated_output" > "$temp_file"
                    jq -s '.[0] * .[1]' "$updates_file" "$temp_file" > "${updates_file}.tmp"
                    mv "${updates_file}.tmp" "$updates_file"
                    rm -f "$temp_file"
                    
                    local count=$(echo "$outdated_output" | jq 'length' 2>/dev/null || echo "0")
                    log_message "INFO" "Encontradas $count actualizaciones en $project"
                else
                    log_message "WARN" "Error verificando paquetes en $project"
                fi
            else
                log_message "WARN" "npm no está disponible"
            fi
            
            popd > /dev/null 2>&1
        else
            log_message "INFO" "$project/package.json no encontrado"
        fi
    done
    
    echo "$updates_file"
}

function get_system_updates() {
    print_color "$BLUE" "🔧 Verificando actualizaciones del sistema..."
    log_message "INFO" "Verificando actualizaciones del sistema"
    
    local updates_file="/tmp/aegis_system_updates.json"
    echo "[]" > "$updates_file"
    
    # Detectar distribución
    if [[ -f /etc/os-release ]]; then
        source /etc/os-release
        local distro="$ID"
        
        case "$distro" in
            ubuntu|debian)
                if command -v apt &> /dev/null; then
                    log_message "INFO" "Verificando actualizaciones con apt"
                    apt list --upgradable 2>/dev/null | grep -v "WARNING" | tail -n +2 | while IFS= read -r line; do
                        if [[ -n "$line" ]]; then
                            local package=$(echo "$line" | cut -d'/' -f1)
                            local versions=$(echo "$line" | grep -o '[0-9][^[:space:]]*' | head -2)
                            echo "{\"name\": \"$package\", \"type\": \"apt\"}" >> "${updates_file}.tmp"
                        fi
                    done
                    
                    if [[ -f "${updates_file}.tmp" ]]; then
                        jq -s '.' "${updates_file}.tmp" > "$updates_file"
                        rm -f "${updates_file}.tmp"
                    fi
                fi
                ;;
            fedora|centos|rhel)
                if command -v dnf &> /dev/null; then
                    log_message "INFO" "Verificando actualizaciones con dnf"
                    dnf check-update --quiet 2>/dev/null | grep -v "^$" | while IFS= read -r line; do
                        if [[ -n "$line" && ! "$line" =~ ^[[:space:]]*$ ]]; then
                            local package=$(echo "$line" | awk '{print $1}')
                            echo "{\"name\": \"$package\", \"type\": \"dnf\"}" >> "${updates_file}.tmp"
                        fi
                    done
                    
                    if [[ -f "${updates_file}.tmp" ]]; then
                        jq -s '.' "${updates_file}.tmp" > "$updates_file"
                        rm -f "${updates_file}.tmp"
                    fi
                elif command -v yum &> /dev/null; then
                    log_message "INFO" "Verificando actualizaciones con yum"
                    yum check-update --quiet 2>/dev/null | grep -v "^$" | while IFS= read -r line; do
                        if [[ -n "$line" && ! "$line" =~ ^[[:space:]]*$ ]]; then
                            local package=$(echo "$line" | awk '{print $1}')
                            echo "{\"name\": \"$package\", \"type\": \"yum\"}" >> "${updates_file}.tmp"
                        fi
                    done
                    
                    if [[ -f "${updates_file}.tmp" ]]; then
                        jq -s '.' "${updates_file}.tmp" > "$updates_file"
                        rm -f "${updates_file}.tmp"
                    fi
                fi
                ;;
            arch)
                if command -v pacman &> /dev/null; then
                    log_message "INFO" "Verificando actualizaciones con pacman"
                    pacman -Qu 2>/dev/null | while IFS= read -r line; do
                        if [[ -n "$line" ]]; then
                            local package=$(echo "$line" | awk '{print $1}')
                            echo "{\"name\": \"$package\", \"type\": \"pacman\"}" >> "${updates_file}.tmp"
                        fi
                    done
                    
                    if [[ -f "${updates_file}.tmp" ]]; then
                        jq -s '.' "${updates_file}.tmp" > "$updates_file"
                        rm -f "${updates_file}.tmp"
                    fi
                fi
                ;;
            *)
                log_message "WARN" "Distribución no soportada para actualizaciones del sistema: $distro"
                ;;
        esac
    else
        log_message "WARN" "No se pudo detectar la distribución"
    fi
    
    echo "$updates_file"
}

function show_update_summary() {
    local python_updates_file=$1
    local nodejs_updates_file=$2
    local system_updates_file=$3
    
    local total_updates=0
    
    # Contar actualizaciones de Python
    local python_count=0
    if [[ -f "$python_updates_file" ]]; then
        python_count=$(jq length "$python_updates_file" 2>/dev/null || echo "0")
        total_updates=$((total_updates + python_count))
    fi
    
    # Contar actualizaciones de Node.js
    local nodejs_count=0
    if [[ -f "$nodejs_updates_file" ]]; then
        nodejs_count=$(jq '[.[]] | add | length' "$nodejs_updates_file" 2>/dev/null || echo "0")
        total_updates=$((total_updates + nodejs_count))
    fi
    
    # Contar actualizaciones del sistema
    local system_count=0
    if [[ -f "$system_updates_file" ]]; then
        system_count=$(jq length "$system_updates_file" 2>/dev/null || echo "0")
        total_updates=$((total_updates + system_count))
    fi
    
    if [[ $total_updates -eq 0 ]]; then
        print_color "$GREEN" "✅ No se encontraron actualizaciones disponibles"
        return
    fi
    
    print_color "$CYAN" "📋 Actualizaciones Disponibles:"
    print_color "$CYAN" "==============================="
    echo ""
    
    # Mostrar actualizaciones de Python
    if [[ $python_count -gt 0 ]]; then
        print_color "$YELLOW" "🔸 Python ($python_count paquetes):"
        jq -r '.[] | "  📦 \(.name): \(.version) → \(.latest_version)"' "$python_updates_file" 2>/dev/null || true
        echo ""
    fi
    
    # Mostrar actualizaciones de Node.js
    if [[ $nodejs_count -gt 0 ]]; then
        print_color "$YELLOW" "🔸 Node.js ($nodejs_count paquetes):"
        jq -r 'to_entries[] | "  📁 \(.key):" as $project | .value | to_entries[] | "    📦 \(.key): \(.value.current) → \(.value.latest)"' "$nodejs_updates_file" 2>/dev/null || true
        echo ""
    fi
    
    # Mostrar actualizaciones del sistema
    if [[ $system_count -gt 0 ]]; then
        print_color "$YELLOW" "🔸 Sistema ($system_count paquetes):"
        jq -r '.[] | "  📦 \(.name) (\(.type))"' "$system_updates_file" 2>/dev/null || true
        echo ""
    fi
    
    log_message "INFO" "Resumen: $python_count Python, $nodejs_count Node.js, $system_count sistema"
}

function update_python_packages() {
    local force=$1
    
    print_color "$BLUE" "🐍 Actualizando paquetes de Python..."
    log_message "INFO" "Iniciando actualización de paquetes de Python (force=$force)"
    
    if [[ ! -f "requirements.txt" ]]; then
        print_color "$YELLOW" "⚠️  No se encontró requirements.txt"
        log_message "WARN" "requirements.txt no encontrado"
        return
    fi
    
    # Activar entorno virtual si existe
    if [[ -f "venv/bin/activate" ]]; then
        source venv/bin/activate
        log_message "INFO" "Entorno virtual activado"
    fi
    
    if [[ "$force" == "true" ]]; then
        print_color "$BLUE" "🔄 Reinstalando todas las dependencias de Python..."
        log_message "INFO" "Reinstalando dependencias de Python"
        pip install --force-reinstall -r requirements.txt
    else
        print_color "$BLUE" "🔄 Actualizando dependencias de Python..."
        log_message "INFO" "Actualizando dependencias de Python"
        pip install --upgrade -r requirements.txt
    fi
    
    print_color "$GREEN" "✅ Paquetes de Python actualizados"
    log_message "INFO" "Paquetes de Python actualizados exitosamente"
}

function update_nodejs_packages() {
    local force=$1
    
    print_color "$BLUE" "📦 Actualizando paquetes de Node.js..."
    log_message "INFO" "Iniciando actualización de paquetes de Node.js (force=$force)"
    
    local projects=("dapps/secure-chat/ui" "dapps/aegis-token")
    
    for project in "${projects[@]}"; do
        if [[ -f "$project/package.json" ]]; then
            print_color "$BLUE" "🔄 Actualizando $project..."
            log_message "INFO" "Actualizando $project"
            
            pushd "$project" > /dev/null 2>&1 || continue
            
            if [[ "$force" == "true" ]]; then
                rm -rf node_modules package-lock.json 2>/dev/null || true
                npm install
                log_message "INFO" "Reinstalación forzada completada para $project"
            else
                npm update
                log_message "INFO" "Actualización completada para $project"
            fi
            
            popd > /dev/null 2>&1
            print_color "$GREEN" "✅ $project actualizado"
        else
            log_message "INFO" "$project/package.json no encontrado, saltando"
        fi
    done
}

function update_system_packages() {
    local force=$1
    
    print_color "$BLUE" "🔧 Actualizando herramientas del sistema..."
    log_message "INFO" "Iniciando actualización del sistema (force=$force)"
    
    # Detectar distribución
    if [[ -f /etc/os-release ]]; then
        source /etc/os-release
        local distro="$ID"
        
        case "$distro" in
            ubuntu|debian)
                if command -v apt &> /dev/null; then
                    print_color "$BLUE" "🔄 Actualizando paquetes con apt..."
                    log_message "INFO" "Actualizando con apt"
                    
                    sudo apt update
                    if [[ "$force" == "true" ]]; then
                        sudo apt full-upgrade -y
                    else
                        sudo apt upgrade -y
                    fi
                fi
                ;;
            fedora|centos|rhel)
                if command -v dnf &> /dev/null; then
                    print_color "$BLUE" "🔄 Actualizando paquetes con dnf..."
                    log_message "INFO" "Actualizando con dnf"
                    
                    sudo dnf update -y
                elif command -v yum &> /dev/null; then
                    print_color "$BLUE" "🔄 Actualizando paquetes con yum..."
                    log_message "INFO" "Actualizando con yum"
                    
                    sudo yum update -y
                fi
                ;;
            arch)
                if command -v pacman &> /dev/null; then
                    print_color "$BLUE" "🔄 Actualizando paquetes con pacman..."
                    log_message "INFO" "Actualizando con pacman"
                    
                    sudo pacman -Syu --noconfirm
                fi
                ;;
            *)
                print_color "$YELLOW" "⚠️  Distribución no soportada para actualizaciones del sistema: $distro"
                log_message "WARN" "Distribución no soportada: $distro"
                return
                ;;
        esac
        
        print_color "$GREEN" "✅ Herramientas del sistema actualizadas"
        log_message "INFO" "Sistema actualizado exitosamente"
    else
        print_color "$YELLOW" "⚠️  No se pudo detectar la distribución"
        log_message "WARN" "No se pudo detectar la distribución"
    fi
}

function create_backup_before_update() {
    print_color "$BLUE" "💾 Creando respaldo antes de la actualización..."
    log_message "INFO" "Creando respaldo pre-actualización"
    
    if [[ -f "scripts/backup-config.sh" ]]; then
        if bash "scripts/backup-config.sh" --backup-path "backups/pre-update"; then
            print_color "$GREEN" "✅ Respaldo creado exitosamente"
            log_message "INFO" "Respaldo creado exitosamente"
        else
            print_color "$RED" "❌ Error creando respaldo"
            log_message "ERROR" "Error creando respaldo"
            
            read -p "¿Deseas continuar sin respaldo? (s/N): " -r continue_choice
            if [[ ! "$continue_choice" =~ ^[sS]$ ]]; then
                print_color "$YELLOW" "❌ Actualización cancelada por el usuario"
                log_message "INFO" "Actualización cancelada por el usuario"
                exit 0
            fi
        fi
    else
        print_color "$YELLOW" "⚠️  Script de respaldo no encontrado. Continuando sin respaldo."
        log_message "WARN" "Script de respaldo no encontrado"
    fi
}

function test_update_success() {
    print_color "$BLUE" "🔍 Verificando éxito de la actualización..."
    log_message "INFO" "Verificando éxito de la actualización"
    
    local success=true
    
    # Verificar Python
    if [[ -f "requirements.txt" ]]; then
        if [[ -f "venv/bin/activate" ]]; then
            source venv/bin/activate
        fi
        
        if command -v pip &> /dev/null; then
            if ! pip check &>/dev/null; then
                print_color "$YELLOW" "⚠️  Advertencia: Conflictos detectados en paquetes de Python"
                log_message "WARN" "Conflictos en paquetes de Python"
                success=false
            fi
        fi
    fi
    
    # Verificar Node.js
    local projects=("dapps/secure-chat/ui" "dapps/aegis-token")
    
    for project in "${projects[@]}"; do
        if [[ -f "$project/package.json" ]]; then
            pushd "$project" > /dev/null 2>&1 || continue
            
            if command -v npm &> /dev/null; then
                if ! npm audit --audit-level=high &>/dev/null; then
                    print_color "$YELLOW" "⚠️  Advertencia: Vulnerabilidades detectadas en $project"
                    log_message "WARN" "Vulnerabilidades en $project"
                    success=false
                fi
            fi
            
            popd > /dev/null 2>&1
        fi
    done
    
    if [[ "$success" == "true" ]]; then
        print_color "$GREEN" "✅ Actualización completada exitosamente"
        log_message "INFO" "Actualización completada exitosamente"
    else
        print_color "$YELLOW" "⚠️  Actualización completada con advertencias"
        log_message "WARN" "Actualización completada con advertencias"
    fi
    
    return $([[ "$success" == "true" ]] && echo 0 || echo 1)
}

# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================

function main() {
    # Crear directorio de logs
    mkdir -p "$(dirname "$LOG_FILE")"
    
    log_message "INFO" "Iniciando script de actualización del sistema"
    log_message "INFO" "Argumentos: $*"
    
    parse_arguments "$@"
    
    test_prerequisites
    
    if ! test_internet_connection; then
        print_color "$RED" "❌ Error: Se requiere conexión a internet para actualizar"
        log_message "ERROR" "Sin conexión a internet"
        exit 1
    fi
    
    # Crear respaldo si se solicita
    if [[ "$CREATE_BACKUP" == "true" ]]; then
        create_backup_before_update
    fi
    
    # Recopilar actualizaciones disponibles
    local python_updates_file=""
    local nodejs_updates_file=""
    local system_updates_file=""
    
    if [[ "$UPDATE_PYTHON" == "true" || "$UPDATE_ALL" == "true" ]]; then
        python_updates_file=$(get_python_updates)
    fi
    
    if [[ "$UPDATE_NODEJS" == "true" || "$UPDATE_ALL" == "true" ]]; then
        nodejs_updates_file=$(get_nodejs_updates)
    fi
    
    if [[ "$UPDATE_SYSTEM" == "true" || "$UPDATE_ALL" == "true" ]]; then
        system_updates_file=$(get_system_updates)
    fi
    
    # Si no se especifica ninguna opción, verificar todo
    if [[ "$UPDATE_PYTHON" == "false" && "$UPDATE_NODEJS" == "false" && "$UPDATE_SYSTEM" == "false" && "$UPDATE_ALL" == "false" ]]; then
        python_updates_file=$(get_python_updates)
        nodejs_updates_file=$(get_nodejs_updates)
        system_updates_file=$(get_system_updates)
    fi
    
    # Mostrar resumen de actualizaciones
    show_update_summary "$python_updates_file" "$nodejs_updates_file" "$system_updates_file"
    
    # Si solo es verificación, salir
    if [[ "$CHECK_ONLY" == "true" ]]; then
        print_color "$BLUE" "🔍 Verificación completada"
        log_message "INFO" "Verificación completada"
        
        # Limpiar archivos temporales
        rm -f "$python_updates_file" "$nodejs_updates_file" "$system_updates_file" 2>/dev/null || true
        exit 0
    fi
    
    # Verificar si hay actualizaciones
    local total_updates=0
    [[ -n "$python_updates_file" ]] && total_updates=$((total_updates + $(jq length "$python_updates_file" 2>/dev/null || echo "0")))
    [[ -n "$nodejs_updates_file" ]] && total_updates=$((total_updates + $(jq '[.[]] | add | length' "$nodejs_updates_file" 2>/dev/null || echo "0")))
    [[ -n "$system_updates_file" ]] && total_updates=$((total_updates + $(jq length "$system_updates_file" 2>/dev/null || echo "0")))
    
    if [[ $total_updates -eq 0 ]]; then
        # Limpiar archivos temporales
        rm -f "$python_updates_file" "$nodejs_updates_file" "$system_updates_file" 2>/dev/null || true
        exit 0
    fi
    
    # Confirmar actualización
    if [[ "$FORCE_UPDATE" == "false" ]]; then
        read -p "¿Deseas proceder con las actualizaciones? (s/N): " -r confirm
        if [[ ! "$confirm" =~ ^[sS]$ ]]; then
            print_color "$YELLOW" "❌ Actualización cancelada por el usuario"
            log_message "INFO" "Actualización cancelada por el usuario"
            
            # Limpiar archivos temporales
            rm -f "$python_updates_file" "$nodejs_updates_file" "$system_updates_file" 2>/dev/null || true
            exit 0
        fi
    fi
    
    # Realizar actualizaciones
    print_color "$BLUE" "🚀 Iniciando proceso de actualización..."
    log_message "INFO" "Iniciando proceso de actualización"
    echo ""
    
    if [[ "$UPDATE_PYTHON" == "true" || "$UPDATE_ALL" == "true" ]] && [[ -n "$python_updates_file" ]]; then
        update_python_packages "$FORCE_UPDATE"
    fi
    
    if [[ "$UPDATE_NODEJS" == "true" || "$UPDATE_ALL" == "true" ]] && [[ -n "$nodejs_updates_file" ]]; then
        update_nodejs_packages "$FORCE_UPDATE"
    fi
    
    if [[ "$UPDATE_SYSTEM" == "true" || "$UPDATE_ALL" == "true" ]] && [[ -n "$system_updates_file" ]]; then
        update_system_packages "$FORCE_UPDATE"
    fi
    
    # Verificar éxito de la actualización
    echo ""
    test_update_success
    
    echo ""
    print_color "$YELLOW" "💡 Recomendación: Reinicia los servicios para aplicar las actualizaciones"
    print_color "$WHITE" "   Usa: ./scripts/stop-all-services.sh && ./scripts/start-all-services.sh"
    
    # Limpiar archivos temporales
    rm -f "$python_updates_file" "$nodejs_updates_file" "$system_updates_file" 2>/dev/null || true
    
    log_message "INFO" "Script de actualización completado"
}

# Verificar si el script se ejecuta directamente
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi