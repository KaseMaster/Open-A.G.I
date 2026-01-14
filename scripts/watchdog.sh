#!/bin/sh
# Watchdog script para AEGIS Framework
# Monitorea servicios críticos y los reinicia si fallan

set -e

WATCHDOG_INTERVAL=30
MAX_RESTARTS=3
RESTART_COUNT_FILE="/tmp/aegis_restart_count"

# Función para verificar si un servicio está funcionando
check_service() {
    local service_name=$1
    local container_name=$2
    local health_check=$3

    if docker ps --filter "name=${container_name}" --filter "status=running" | grep -q "${container_name}"; then
        if [ -n "${health_check}" ]; then
            if docker exec "${container_name}" sh -c "${health_check}" > /dev/null 2>&1; then
                echo "$(date): ${service_name} (${container_name}) - OK"
                return 0
            else
                echo "$(date): ${service_name} (${container_name}) - HEALTH CHECK FAILED"
                return 1
            fi
        else
            echo "$(date): ${service_name} (${container_name}) - RUNNING"
            return 0
        fi
    else
        echo "$(date): ${service_name} (${container_name}) - NOT RUNNING"
        return 1
    fi
}

# Función para reiniciar un servicio
restart_service() {
    local container_name=$1
    local restart_count=${2:-0}

    echo "$(date): Reiniciando ${container_name} (intento ${restart_count})"

    # Detener contenedor si existe
    docker stop "${container_name}" 2>/dev/null || true
    docker rm "${container_name}" 2>/dev/null || true

    # Reiniciar con docker-compose
    docker-compose up -d "${container_name}"

    echo "$(date): ${container_name} reiniciado"
}

# Función principal de monitoreo
main() {
    echo "$(date): Iniciando watchdog AEGIS Framework..."

    # Crear archivo de conteo de reinicios si no existe
    echo "0" > "${RESTART_COUNT_FILE}"

    while true; do
        echo "$(date): === CICLO DE MONITOREO ==="

        # Verificar TOR
        if ! check_service "TOR" "aegis-tor" "curl -s http://localhost:9051/tor/status"; then
            restart_service "aegis-tor" "$(cat ${RESTART_COUNT_FILE})"
        fi

        # Verificar Redis
        if ! check_service "Redis" "aegis-redis" "redis-cli ping"; then
            restart_service "aegis-redis" "$(cat ${RESTART_COUNT_FILE})"
        fi

        # Verificar nodo principal
        if ! check_service "AEGIS Node" "aegis-node" "python -c 'import main; print(\"OK\")'"; then
            restart_service "aegis-node" "$(cat ${RESTART_COUNT_FILE})"
        fi

        # Verificar web dashboard
        if ! check_service "Web Dashboard" "aegis-web-dashboard" ""; then
            restart_service "aegis-web-dashboard" "$(cat ${RESTART_COUNT_FILE})"
        fi

        # Verificar nginx
        if ! check_service "Nginx" "aegis-nginx" ""; then
            restart_service "aegis-nginx" "$(cat ${RESTART_COUNT_FILE})"
        fi

        echo "$(date): Ciclo de monitoreo completado"
        sleep "${WATCHDOG_INTERVAL}"
    done
}

# Manejar señales de terminación
trap "echo 'Watchdog detenido por señal'; exit 0" INT TERM

# Ejecutar función principal
main
