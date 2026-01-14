#!/bin/bash
# Script de entrada para contenedor Docker AEGIS

set -e

echo "🚀 Iniciando contenedor AEGIS Framework..."

# Verificar variables de entorno requeridas
if [ -z "$AEGIS_CONFIG_PATH" ]; then
    export AEGIS_CONFIG_PATH="/app/config/production_config.json"
fi

# Verificar que el archivo de configuración existe
if [ ! -f "$AEGIS_CONFIG_PATH" ]; then
    echo "⚠️ Archivo de configuración no encontrado, usando configuración por defecto"
    cp /app/config/production_config.json.default "$AEGIS_CONFIG_PATH" 2>/dev/null || true
fi

# Crear directorios necesarios
mkdir -p /app/logs /app/certs /app/data /app/backups /app/temp

# Establecer permisos correctos
chmod 755 /app/logs /app/certs /app/data /app/backups /app/temp
chmod 644 /app/config/*.json 2>/dev/null || true

# Generar certificados SSL si no existen
if [ ! -f "/app/certs/server.crt" ] || [ ! -f "/app/certs/server.key" ]; then
    echo "🔐 Generando certificados SSL..."
    python -c "
import os
from cryptography import x509
from cryptography.x509.oid import NameOID
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization
import datetime

private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
subject = issuer = x509.Name([
    x509.NameAttribute(NameOID.COUNTRY_NAME, 'US'),
    x509.NameAttribute(NameOID.ORGANIZATION_NAME, 'AEGIS Framework'),
    x509.NameAttribute(NameOID.COMMON_NAME, 'localhost'),
])

cert = x509.CertificateBuilder().subject_name(subject).issuer_name(issuer).public_key(
    private_key.public_key()
).serial_number(x509.random_serial_number()).not_valid_before(
    datetime.datetime.utcnow()
).not_valid_after(
    datetime.datetime.utcnow() + datetime.timedelta(days=365)
).add_extension(
    x509.SubjectAlternativeName([
        x509.DNSName('localhost'),
        x509.DNSName('127.0.0.1'),
    ]), critical=False
).sign(private_key, hashes.SHA256())

os.makedirs('/app/certs', exist_ok=True)
with open('/app/certs/server.crt', 'wb') as f:
    f.write(cert.public_bytes(serialization.Encoding.PEM))
with open('/app/certs/server.key', 'wb') as f:
    f.write(private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption()
    ))
print('✅ Certificados SSL generados')
" 2>/dev/null || echo "⚠️ Error generando certificados, usando certificados de desarrollo"
fi

# Verificar conectividad de red
echo "🌐 Verificando conectividad de red..."
if ! nc -z localhost 8080 2>/dev/null; then
    echo "✅ Puerto 8080 disponible"
fi

if ! nc -z localhost 8051 2>/dev/null; then
    echo "✅ Puerto 8051 disponible"
fi

# Configurar límites de recursos
ulimit -n 4096  # Límite de archivos abiertos
ulimit -u 2048  # Límite de procesos

echo "✅ Configuración inicial completada"
echo "🎯 Ejecutando comando: $@"

# Ejecutar el comando proporcionado
exec "$@"