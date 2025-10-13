#!/bin/bash

# ============================================================================
# AEGIS Framework - Script de Respaldo de Configuración para Linux
# ============================================================================
# Descripción: Script para crear respaldos de la configuración del sistema
# Autor: AEGIS Security Team
# Versión: 2.0.0
# Fecha: 2024
# ============================================================================

set -euo pipefail

# Configuración por defecto
BACKUP_PATH="./backups"
RESTORE_MODE=false
RESTORE_FROM=""
LIST_MODE=false
CLEAN_MODE=false
KEEP_DAYS=30
VERBOSE=false

# Colores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
WHITE='\033[1;37m'
NC='\033[0m' # No Color

# Función para output con colores
print_color() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

# Función para mostrar ayuda
show_help() {
    print_color "$CYAN" "🛡️  AEGIS Framework - Gestor de Respaldos de Configuración"
    print_color "$CYAN" "========================================================="
    echo ""
    print_color "$YELLOW" "DESCRIPCIÓN:"
    echo "  Crea y gestiona respaldos de la configuración del sistema AEGIS"
    echo ""
    print_color "$YELLOW" "USO:"
    echo "  ./backup-config.sh [OPCIONES]"
    echo ""
    print_color "$YELLOW" "OPCIONES:"
    echo "  -p, --backup-path PATH    Directorio para almacenar respaldos (default: ./backups)"
    echo "  -r, --restore             Restaurar desde un respaldo"
    echo "  -f, --restore-from PATH   Ruta específica del respaldo a restaurar"
    echo "  -l, --list                Listar respaldos disponibles"
    echo "  -c, --clean               Limpiar respaldos antiguos"
    echo "  -k, --keep-days N         Días a mantener respaldos (default: 30)"
    echo "  -v, --verbose             Modo verboso"
    echo "  -h, --help                Mostrar esta ayuda"
    echo ""
    print_color "$YELLOW" "EJEMPLOS:"
    echo "  ./backup-config.sh                                    # Crear respaldo"
    echo "  ./backup-config.sh -p /opt/backups                   # Respaldo en ubicación específica"
    echo "  ./backup-config.sh -l                                # Listar respaldos"
    echo "  ./backup-config.sh -r                                # Restaurar último respaldo"
    echo "  ./backup-config.sh -f ./backups/aegis_backup_20241201_120000.tar.gz"
    echo "  ./backup-config.sh -c -k 7                           # Limpiar respaldos > 7 días"
    echo ""
    exit 0
}

# Función para logging
log_message() {
    local level=$1
    local message=$2
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    if [[ "$VERBOSE" == true ]]; then
        echo "[$timestamp] [$level] $message" >&2
    fi
}

# Parsear argumentos
parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -p|--backup-path)
                BACKUP_PATH="$2"
                shift 2
                ;;
            -r|--restore)
                RESTORE_MODE=true
                shift
                ;;
            -f|--restore-from)
                RESTORE_FROM="$2"
                RESTORE_MODE=true
                shift 2
                ;;
            -l|--list)
                LIST_MODE=true
                shift
                ;;
            -c|--clean)
                CLEAN_MODE=true
                shift
                ;;
            -k|--keep-days)
                KEEP_DAYS="$2"
                shift 2
                ;;
            -v|--verbose)
                VERBOSE=true
                shift
                ;;
            -h|--help)
                show_help
                ;;
            *)
                print_color "$RED" "❌ Opción desconocida: $1"
                echo "Usa -h o --help para ver las opciones disponibles"
                exit 1
                ;;
        esac
    done
}

# Verificar prerrequisitos
test_prerequisites() {
    print_color "$BLUE" "🔍 Verificando prerrequisitos..."
    
    # Verificar si estamos en el directorio correcto
    if [[ ! -f "main.py" ]]; then
        print_color "$RED" "❌ Error: No se encontró main.py. Ejecuta este script desde el directorio raíz del proyecto AEGIS."
        exit 1
    fi
    
    # Verificar herramientas necesarias
    local tools=("tar" "gzip" "find" "jq")
    for tool in "${tools[@]}"; do
        if ! command -v "$tool" &> /dev/null; then
            print_color "$RED" "❌ Error: $tool no está instalado"
            exit 1
        fi
    done
    
    print_color "$GREEN" "✅ Prerrequisitos verificados"
}

# Crear directorio de respaldos
create_backup_directory() {
    local path=$1
    
    if [[ ! -d "$path" ]]; then
        if mkdir -p "$path"; then
            print_color "$GREEN" "📁 Directorio de respaldos creado: $path"
        else
            print_color "$RED" "❌ Error creando directorio de respaldos: $path"
            exit 1
        fi
    fi
}

# Obtener elementos a respaldar
get_backup_items() {
    local items=()
    
    # Archivos de configuración principales
    local config_files=(
        ".env"
        "config/app_config.json"
        "config/torrc"
    )
    
    for file in "${config_files[@]}"; do
        if [[ -f "$file" ]]; then
            items+=("$file")
        fi
    done
    
    # Directorios de configuración
    local config_dirs=(
        "config"
        "logs"
        "tor_data"
    )
    
    for dir in "${config_dirs[@]}"; do
        if [[ -d "$dir" ]]; then
            items+=("$dir")
        fi
    done
    
    # Archivos de dependencias
    local dep_files=(
        "requirements.txt"
        "dapps/secure-chat/ui/package.json"
        "dapps/secure-chat/ui/package-lock.json"
        "dapps/aegis-token/package.json"
        "dapps/aegis-token/package-lock.json"
        "dapps/aegis-token/hardhat.config.js"
    )
    
    for file in "${dep_files[@]}"; do
        if [[ -f "$file" ]]; then
            items+=("$file")
        fi
    done
    
    printf '%s\n' "${items[@]}"
}

# Crear respaldo
create_backup() {
    local backup_path=$1
    
    print_color "$BLUE" "🔄 Iniciando proceso de respaldo..."
    
    # Crear directorio de respaldos
    create_backup_directory "$backup_path"
    
    # Generar nombre del respaldo
    local timestamp=$(date '+%Y%m%d_%H%M%S')
    local backup_name="aegis_backup_$timestamp"
    local temp_dir="/tmp/$backup_name"
    local tar_path="$backup_path/$backup_name.tar.gz"
    
    # Crear directorio temporal
    if ! mkdir -p "$temp_dir"; then
        print_color "$RED" "❌ Error creando directorio temporal: $temp_dir"
        exit 1
    fi
    
    # Obtener elementos a respaldar
    local items
    mapfile -t items < <(get_backup_items)
    
    if [[ ${#items[@]} -eq 0 ]]; then
        print_color "$YELLOW" "⚠️  No se encontraron elementos para respaldar"
        rm -rf "$temp_dir"
        return
    fi
    
    print_color "$BLUE" "📦 Copiando ${#items[@]} elementos..."
    
    # Copiar elementos
    for item in "${items[@]}"; do
        local dest_path="$temp_dir/$item"
        local dest_dir=$(dirname "$dest_path")
        
        # Crear directorio de destino si no existe
        mkdir -p "$dest_dir"
        
        if [[ -f "$item" ]]; then
            cp "$item" "$dest_path"
            print_color "$GREEN" "  ✅ Archivo: $item"
        elif [[ -d "$item" ]]; then
            cp -r "$item" "$dest_path"
            print_color "$GREEN" "  ✅ Directorio: $item"
        fi
        
        log_message "INFO" "Copiado: $item -> $dest_path"
    done
    
    # Crear archivo de metadatos
    local metadata_file="$temp_dir/backup_metadata.json"
    cat > "$metadata_file" << EOF
{
    "backup_date": "$(date -Iseconds)",
    "backup_version": "2.0.0",
    "system_info": {
        "os": "$(uname -s)",
        "kernel": "$(uname -r)",
        "arch": "$(uname -m)",
        "hostname": "$(hostname)",
        "user": "$(whoami)"
    },
    "items": [
$(printf '        "%s"' "${items[0]}")
$(printf ',\n        "%s"' "${items[@]:1}")
    ]
}
EOF
    
    # Crear archivo tar.gz
    print_color "$BLUE" "🗜️  Comprimiendo respaldo..."
    
    if tar -czf "$tar_path" -C "$temp_dir" .; then
        # Limpiar directorio temporal
        rm -rf "$temp_dir"
        
        # Verificar respaldo
        if [[ -f "$tar_path" ]]; then
            local backup_size=$(du -h "$tar_path" | cut -f1)
            print_color "$GREEN" "✅ Respaldo creado exitosamente:"
            print_color "$WHITE" "   📁 Archivo: $tar_path"
            print_color "$WHITE" "   📊 Tamaño: $backup_size"
            print_color "$WHITE" "   🕒 Fecha: $(date '+%Y-%m-%d %H:%M:%S')"
        else
            print_color "$RED" "❌ Error: No se pudo crear el archivo de respaldo"
            exit 1
        fi
    else
        print_color "$RED" "❌ Error durante la compresión del respaldo"
        rm -rf "$temp_dir"
        exit 1
    fi
}

# Obtener lista de respaldos
get_backup_list() {
    local backup_path=$1
    
    if [[ ! -d "$backup_path" ]]; then
        return
    fi
    
    find "$backup_path" -name "aegis_backup_*.tar.gz" -type f | sort -r
}

# Mostrar lista de respaldos
show_backup_list() {
    local backup_path=$1
    
    print_color "$CYAN" "📋 Lista de Respaldos Disponibles"
    print_color "$CYAN" "================================="
    echo ""
    
    local backups
    mapfile -t backups < <(get_backup_list "$backup_path")
    
    if [[ ${#backups[@]} -eq 0 ]]; then
        print_color "$YELLOW" "⚠️  No se encontraron respaldos en: $backup_path"
        return
    fi
    
    local index=1
    for backup in "${backups[@]}"; do
        local basename=$(basename "$backup")
        local size=$(du -h "$backup" | cut -f1)
        local mtime=$(stat -c %Y "$backup")
        local date_created=$(date -d "@$mtime" '+%Y-%m-%d %H:%M:%S')
        local age_seconds=$(($(date +%s) - mtime))
        local age_days=$((age_seconds / 86400))
        local age_hours=$(((age_seconds % 86400) / 3600))
        
        local age_text
        if [[ $age_days -gt 0 ]]; then
            age_text="$age_days días"
        else
            age_text="$age_hours horas"
        fi
        
        print_color "$WHITE" "$index. $basename"
        print_color "$BLUE" "   📊 Tamaño: $size"
        print_color "$BLUE" "   🕒 Creado: $date_created"
        print_color "$BLUE" "   ⏰ Antigüedad: $age_text"
        echo ""
        
        ((index++))
    done
}

# Restaurar respaldo
restore_backup() {
    local backup_path=$1
    local restore_from=$2
    
    local backup_file=""
    
    if [[ -n "$restore_from" ]]; then
        if [[ -f "$restore_from" ]]; then
            backup_file="$restore_from"
        else
            print_color "$RED" "❌ Error: No se encontró el archivo de respaldo: $restore_from"
            exit 1
        fi
    else
        # Usar el respaldo más reciente
        local backups
        mapfile -t backups < <(get_backup_list "$backup_path")
        
        if [[ ${#backups[@]} -eq 0 ]]; then
            print_color "$RED" "❌ Error: No se encontraron respaldos para restaurar"
            exit 1
        fi
        
        backup_file="${backups[0]}"
    fi
    
    print_color "$BLUE" "🔄 Iniciando restauración desde: $backup_file"
    
    # Confirmar restauración
    print_color "$YELLOW" "⚠️  ADVERTENCIA: Esta operación sobrescribirá la configuración actual."
    read -p "¿Deseas continuar? (s/N): " -r confirm
    
    if [[ ! "$confirm" =~ ^[sS]$ ]]; then
        print_color "$YELLOW" "❌ Restauración cancelada por el usuario"
        exit 0
    fi
    
    local temp_dir="/tmp/aegis_restore_$(date '+%Y%m%d_%H%M%S')"
    
    # Crear directorio temporal
    if ! mkdir -p "$temp_dir"; then
        print_color "$RED" "❌ Error creando directorio temporal: $temp_dir"
        exit 1
    fi
    
    # Extraer respaldo
    print_color "$BLUE" "📦 Extrayendo respaldo..."
    
    if tar -xzf "$backup_file" -C "$temp_dir"; then
        # Verificar metadatos
        local metadata_file="$temp_dir/backup_metadata.json"
        if [[ -f "$metadata_file" ]]; then
            print_color "$BLUE" "📋 Información del respaldo:"
            
            local backup_date=$(jq -r '.backup_date' "$metadata_file" 2>/dev/null || echo "N/A")
            local os_info=$(jq -r '.system_info.os' "$metadata_file" 2>/dev/null || echo "N/A")
            local items_count=$(jq -r '.items | length' "$metadata_file" 2>/dev/null || echo "N/A")
            
            print_color "$WHITE" "   🕒 Fecha: $backup_date"
            print_color "$WHITE" "   🖥️  Sistema: $os_info"
            print_color "$WHITE" "   📊 Elementos: $items_count"
        fi
        
        # Crear respaldo de seguridad de la configuración actual
        print_color "$BLUE" "💾 Creando respaldo de seguridad de la configuración actual..."
        local safety_backup_path="$backup_path/safety_backup_$(date '+%Y%m%d_%H%M%S').tar.gz"
        create_backup "$(dirname "$safety_backup_path")"
        
        # Restaurar archivos
        print_color "$BLUE" "🔄 Restaurando configuración..."
        
        local restored_items=0
        
        # Copiar archivos del respaldo al directorio actual
        find "$temp_dir" -type f ! -name "backup_metadata.json" | while read -r file; do
            local relative_path="${file#$temp_dir/}"
            local target_path="./$relative_path"
            local target_dir=$(dirname "$target_path")
            
            # Crear directorio de destino si no existe
            mkdir -p "$target_dir"
            
            # Copiar archivo
            if cp "$file" "$target_path"; then
                print_color "$GREEN" "  ✅ Restaurado: $relative_path"
                ((restored_items++))
            else
                print_color "$RED" "  ❌ Error restaurando: $relative_path"
            fi
        done
        
        # Limpiar directorio temporal
        rm -rf "$temp_dir"
        
        print_color "$GREEN" "✅ Restauración completada exitosamente"
        print_color "$WHITE" "   📊 Elementos restaurados: $restored_items"
        print_color "$WHITE" "   💾 Respaldo de seguridad: $safety_backup_path"
        echo ""
        print_color "$YELLOW" "💡 Recomendación: Reinicia los servicios para aplicar la configuración restaurada"
    else
        print_color "$RED" "❌ Error durante la extracción del respaldo"
        rm -rf "$temp_dir"
        exit 1
    fi
}

# Limpiar respaldos antiguos
clean_old_backups() {
    local backup_path=$1
    local keep_days=$2
    
    print_color "$BLUE" "🧹 Limpiando respaldos antiguos (> $keep_days días)..."
    
    if [[ ! -d "$backup_path" ]]; then
        print_color "$YELLOW" "⚠️  No se encontró el directorio de respaldos: $backup_path"
        return
    fi
    
    local cutoff_date=$(date -d "$keep_days days ago" +%s)
    local old_backups=()
    
    while IFS= read -r -d '' backup; do
        local mtime=$(stat -c %Y "$backup")
        if [[ $mtime -lt $cutoff_date ]]; then
            old_backups+=("$backup")
        fi
    done < <(find "$backup_path" -name "aegis_backup_*.tar.gz" -type f -print0)
    
    if [[ ${#old_backups[@]} -eq 0 ]]; then
        print_color "$GREEN" "✅ No se encontraron respaldos antiguos para eliminar"
        return
    fi
    
    print_color "$YELLOW" "🗑️  Eliminando ${#old_backups[@]} respaldos antiguos:"
    
    local total_size=0
    for backup in "${old_backups[@]}"; do
        local basename=$(basename "$backup")
        local size_bytes=$(stat -c %s "$backup")
        local size_mb=$((size_bytes / 1024 / 1024))
        total_size=$((total_size + size_mb))
        
        print_color "$RED" "  🗑️  $basename (${size_mb} MB)"
        rm -f "$backup"
    done
    
    print_color "$GREEN" "✅ Limpieza completada"
    print_color "$WHITE" "   📊 Archivos eliminados: ${#old_backups[@]}"
    print_color "$WHITE" "   💾 Espacio liberado: ${total_size} MB"
}

# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================

main() {
    parse_arguments "$@"
    
    test_prerequisites
    
    if [[ "$LIST_MODE" == true ]]; then
        show_backup_list "$BACKUP_PATH"
    elif [[ "$CLEAN_MODE" == true ]]; then
        clean_old_backups "$BACKUP_PATH" "$KEEP_DAYS"
    elif [[ "$RESTORE_MODE" == true ]]; then
        restore_backup "$BACKUP_PATH" "$RESTORE_FROM"
    else
        create_backup "$BACKUP_PATH"
    fi
}

# Ejecutar función principal
main "$@"