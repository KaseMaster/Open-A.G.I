#!/bin/sh
# Script de inicialización de base de datos para AEGIS Framework

set -e

echo "Inicializando base de datos AEGIS..."

# Crear directorios necesarios
mkdir -p /data/databases /data/backups

# Inicializar base de datos principal si no existe
if [ ! -f /data/databases/aegis_main.db ]; then
    echo "Creando base de datos principal..."
    # Aquí se pueden ejecutar scripts SQL de inicialización
    touch /data/databases/aegis_main.db
    chmod 666 /data/databases/aegis_main.db
fi

# Inicializar base de datos de alertas si no existe
if [ ! -f /data/databases/aegis_alerts.db ]; then
    echo "Creando base de datos de alertas..."
    touch /data/databases/aegis_alerts.db
    chmod 666 /data/databases/aegis_alerts.db
fi

# Inicializar base de datos de dashboard si no existe
if [ ! -f /data/databases/aegis_dashboard.db ]; then
    echo "Creando base de datos de dashboard..."
    touch /data/databases/aegis_dashboard.db
    chmod 666 /data/databases/aegis_dashboard.db
fi

echo "Base de datos inicializada correctamente"
echo "Directorios creados:"
ls -la /data/
