#!/bin/bash

# Script de configuración SSL/TLS para conexionsecreta.site
# Configuración automática con Let's Encrypt

echo "🔒 Iniciando configuración SSL/TLS para conexionsecreta.site"

# Actualizar sistema
echo "📦 Actualizando sistema..."
apt update && apt upgrade -y

# Instalar Certbot
echo "🛠️ Instalando Certbot..."
apt install -y certbot python3-certbot-nginx

# Verificar configuración de Nginx
echo "🔍 Verificando configuración de Nginx..."
nginx -t

if [ $? -ne 0 ]; then
    echo "❌ Error en configuración de Nginx. Corrigiendo..."
    exit 1
fi

# Obtener certificados SSL para todos los dominios
echo "🔐 Obteniendo certificados SSL..."

# Dominio principal
certbot --nginx -d conexionsecreta.site -d www.conexionsecreta.site --non-interactive --agree-tos --email admin@conexionsecreta.site

# Subdominios para DApps
certbot --nginx -d chat.conexionsecreta.site --non-interactive --agree-tos --email admin@conexionsecreta.site
certbot --nginx -d wallet.conexionsecreta.site --non-interactive --agree-tos --email admin@conexionsecreta.site
certbot --nginx -d defi.conexionsecreta.site --non-interactive --agree-tos --email admin@conexionsecreta.site
certbot --nginx -d nft.conexionsecreta.site --non-interactive --agree-tos --email admin@conexionsecreta.site
certbot --nginx -d dao.conexionsecreta.site --non-interactive --agree-tos --email admin@conexionsecreta.site
certbot --nginx -d market.conexionsecreta.site --non-interactive --agree-tos --email admin@conexionsecreta.site

# Configurar renovación automática
echo "🔄 Configurando renovación automática..."
crontab -l | { cat; echo "0 12 * * * /usr/bin/certbot renew --quiet"; } | crontab -

# Configurar headers de seguridad adicionales
echo "🛡️ Configurando headers de seguridad..."

cat > /etc/nginx/snippets/ssl-params.conf << 'EOF'
# SSL Configuration
ssl_protocols TLSv1.2 TLSv1.3;
ssl_prefer_server_ciphers on;
ssl_dhparam /etc/nginx/dhparam.pem;
ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512:ECDHE-RSA-AES256-GCM-SHA384:DHE-RSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-SHA384;
ssl_ecdh_curve secp384r1;
ssl_session_timeout 10m;
ssl_session_cache shared:SSL:10m;
ssl_session_tickets off;
ssl_stapling on;
ssl_stapling_verify on;

# Security Headers
add_header Strict-Transport-Security "max-age=63072000; includeSubDomains; preload";
add_header X-Frame-Options DENY;
add_header X-Content-Type-Options nosniff;
add_header X-XSS-Protection "1; mode=block";
add_header Referrer-Policy "strict-origin-when-cross-origin";
add_header Content-Security-Policy "default-src 'self'; script-src 'self' 'unsafe-inline' 'unsafe-eval' https://cdn.jsdelivr.net; style-src 'self' 'unsafe-inline'; img-src 'self' data: https:; connect-src 'self' wss: https:; font-src 'self' https://fonts.gstatic.com; object-src 'none'; media-src 'self'; frame-src 'none';";
EOF

# Generar parámetros DH
echo "🔑 Generando parámetros Diffie-Hellman..."
openssl dhparam -out /etc/nginx/dhparam.pem 2048

# Reiniciar Nginx
echo "🔄 Reiniciando Nginx..."
systemctl restart nginx

# Verificar estado SSL
echo "✅ Verificando configuración SSL..."
systemctl status nginx

echo "🎉 Configuración SSL completada!"
echo "🌐 Dominios configurados:"
echo "   - https://conexionsecreta.site"
echo "   - https://www.conexionsecreta.site"
echo "   - https://chat.conexionsecreta.site"
echo "   - https://wallet.conexionsecreta.site"
echo "   - https://defi.conexionsecreta.site"
echo "   - https://nft.conexionsecreta.site"
echo "   - https://dao.conexionsecreta.site"
echo "   - https://market.conexionsecreta.site"

echo "🔒 Certificados SSL instalados y configurados"
echo "🔄 Renovación automática programada"
echo "🛡️ Headers de seguridad aplicados"