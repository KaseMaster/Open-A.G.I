#!/bin/bash
# Script de despliegue para AEGIS Framework
# Facilita la instalación y configuración del proyecto

set -e

# Colores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Función para logging
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Verificar dependencias
check_dependencies() {
    log_info "Verificando dependencias..."

    local missing_deps=()

    # Python
    if ! command -v python3 &> /dev/null; then
        missing_deps+=("python3")
    fi

    # Docker
    if ! command -v docker &> /dev/null; then
        missing_deps+=("docker")
    fi

    # Docker Compose
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        missing_deps+=("docker-compose")
    fi

    if [ ${#missing_deps[@]} -ne 0 ]; then
        log_error "Dependencias faltantes: ${missing_deps[*]}"
        log_info "Instale las dependencias y ejecute el script nuevamente."
        exit 1
    fi

    log_success "Todas las dependencias están instaladas"
}

# Configurar entorno virtual
setup_venv() {
    log_info "Configurando entorno virtual..."

    if [ ! -d "venv" ]; then
        python3 -m venv venv
        log_success "Entorno virtual creado"
    else
        log_info "Entorno virtual ya existe"
    fi

    # Activar entorno virtual
    source venv/bin/activate

    log_success "Entorno virtual activado"
}

# Instalar dependencias Python
install_python_deps() {
    log_info "Instalando dependencias Python..."

    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt

    log_success "Dependencias Python instaladas"
}

# Configurar archivos de configuración
setup_config() {
    log_info "Configurando archivos de configuración..."

    # Crear archivo .env si no existe
    if [ ! -f ".env" ]; then
        cp .env.example .env 2>/dev/null || {
            # Crear .env básico si no hay ejemplo
            cat > .env << EOF
# Configuración básica de AEGIS Framework
AEGIS_LOG_LEVEL=INFO
AEGIS_DASHBOARD_PORT=8080
TOR_SOCKS_PORT=9050
TOR_CONTROL_PORT=9051
REDIS_URL=redis://localhost:6379
AEGIS_CONFIG=config/app_config.json
EOF
        }
        log_success "Archivo .env creado"
    else
        log_info "Archivo .env ya existe"
    fi

    # Crear directorios necesarios
    mkdir -p logs data/database data/keys config scripts

    log_success "Configuración completada"
}

# Configurar TOR
setup_tor() {
    log_info "Configurando TOR..."

    if command -v tor &> /dev/null; then
        # Configurar TOR para el proyecto
        if [ ! -f "config/torrc" ]; then
            cp config/torrc_basic config/torrc 2>/dev/null || {
                log_warning "Archivo torrc no encontrado, TOR se configurará manualmente"
            }
        fi

        # Iniciar TOR si no está corriendo
        if ! pgrep -x "tor" > /dev/null; then
            log_info "Iniciando servicio TOR..."
            sudo systemctl start tor 2>/dev/null || {
                log_warning "No se pudo iniciar TOR automáticamente. Inícielo manualmente."
            }
        else
            log_info "TOR ya está corriendo"
        fi

        log_success "TOR configurado"
    else
        log_warning "TOR no está instalado. Algunas funcionalidades podrían no funcionar."
    fi
}

# Ejecutar tests
run_tests() {
    log_info "Ejecutando tests..."

    source venv/bin/activate
    if python -m pytest tests/ -v --tb=short; then
        log_success "Todos los tests pasaron"
    else
        log_warning "Algunos tests fallaron. Verifique la configuración."
    fi
}

# Health check
health_check() {
    log_info "Ejecutando health check..."

    source venv/bin/activate
    if python main.py health-check; then
        log_success "Health check completado"
    else
        log_error "Health check falló"
        exit 1
    fi
}

# Menú principal
show_menu() {
    echo -e "\n${GREEN}=== AEGIS Framework - Menú de Despliegue ===${NC}"
    echo "1. Instalación completa (recomendado)"
    echo "2. Solo configurar entorno Python"
    echo "3. Solo ejecutar tests"
    echo "4. Solo health check"
    echo "5. Iniciar servicios con Docker Compose"
    echo "6. Salir"
    echo -e "${BLUE}Seleccione una opción: ${NC}"
}

# Función principal
main() {
    local choice

    # Verificar si estamos en el directorio correcto
    if [ ! -f "main.py" ]; then
        log_error "Este script debe ejecutarse desde el directorio raíz del proyecto AEGIS"
        exit 1
    fi

    while true; do
        show_menu
        read -r choice

        case $choice in
            1)
                log_info "Iniciando instalación completa..."
                check_dependencies
                setup_venv
                install_python_deps
                setup_config
                setup_tor
                run_tests
                health_check
                log_success "Instalación completa finalizada!"
                break
                ;;
            2)
                check_dependencies
                setup_venv
                install_python_deps
                log_success "Entorno Python configurado"
                ;;
            3)
                setup_venv
                run_tests
                ;;
            4)
                setup_venv
                health_check
                ;;
            5)
                log_info "Iniciando servicios con Docker Compose..."
                if command -v docker-compose &> /dev/null || docker compose version &> /dev/null; then
                    docker-compose up -d
                    log_success "Servicios iniciados. Acceda al dashboard en http://localhost:8080"
                else
                    log_error "Docker Compose no está instalado"
                fi
                ;;
            6)
                log_info "Saliendo..."
                exit 0
                ;;
            *)
                log_error "Opción inválida: $choice"
                ;;
        esac
    done
}

# Ejecutar script
main "$@"
