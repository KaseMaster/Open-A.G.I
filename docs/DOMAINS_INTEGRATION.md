# 🌐 Integración de Dominios - AEGIS OpenAGI

## Dominios Principales

### 🔗 Dominio Principal: `conexionsecreta.net`
- **Propósito**: Dominio principal para todas las DApps
- **Estado**: ✅ Configurado en Nginx
- **SSL/TLS**: 🔄 Pendiente (script preparado)

### 🌍 Dominio ENS: `aegis-openagi.eth`
- **Propósito**: Acceso descentralizado vía ENS
- **Estado**: ✅ Integración JavaScript implementada
- **Gateway**: `https://aegis-openagi.eth.limo`

## Arquitectura de Subdominios

### DApps Principales
```
https://conexionsecreta.net/          → Portal principal de DApps
https://conexionsecreta.net/chat      → SecureChat+ (implementado)
https://conexionsecreta.net/wallet    → AEGIS Wallet (pendiente)
https://conexionsecreta.net/defi      → DeFi Hub (pendiente)
https://conexionsecreta.net/nft       → NFT Marketplace (pendiente)
https://conexionsecreta.net/dao       → DAO Governance (pendiente)
https://conexionsecreta.net/marketplace → P2P Marketplace (pendiente)
```

### Subdominios Dedicados
```
chat.conexionsecreta.net     → SecureChat+ dedicado
wallet.conexionsecreta.net   → Wallet independiente
defi.conexionsecreta.net     → Plataforma DeFi
nft.conexionsecreta.net      → Mercado NFT
dao.conexionsecreta.net      → Sistema DAO
market.conexionsecreta.net   → Marketplace P2P
```

## Configuración Técnica

### Nginx Configuration
```nginx
server {
    listen 80;
    server_name conexionsecreta.net www.conexionsecreta.net 
                aegis-openagi.eth www.aegis-openagi.eth 
                aegis-main.openagi.network 77.237.235.224;

    # Redirigir HTTP a HTTPS (después de SSL)
    # return 301 https://$server_name$request_uri;

    # SecureChat PHP
    location /chat {
        root /opt/openagi/web/advanced-chat-php/public;
        index index.php index.html;
        try_files $uri $uri/ /index.php?$query_string;
        
        location ~ \.php$ {
            fastcgi_pass unix:/var/run/php/php8.1-fpm.sock;
            fastcgi_index index.php;
            fastcgi_param SCRIPT_FILENAME $document_root$fastcgi_script_name;
            include fastcgi_params;
        }
    }

    # DApps Portal
    location /dapps {
        proxy_pass http://127.0.0.1:8087;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    # API OpenAGI
    location /api {
        proxy_pass http://127.0.0.1:8051;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    # Raíz por defecto
    location / {
        proxy_pass http://127.0.0.1:8051;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### JavaScript Configuration
```javascript
// Archivo: /assets/js/config.js
window.ENDPOINTS = {
  PRIMARY_DOMAIN: 'https://conexionsecreta.net',
  ENS_DOMAIN: 'aegis-openagi.eth',
  OPENAGI_API: 'https://conexionsecreta.net:8182',
  WEBSOCKET: 'wss://conexionsecreta.net:8183',
  IPFS_GATEWAY: 'https://conexionsecreta.net:8184'
};
```

## Integración ENS

### Funcionalidades Implementadas
- ✅ Resolución automática de `aegis-openagi.eth`
- ✅ Detección de MetaMask/Web3
- ✅ Gateway ENS (.limo, .link)
- ✅ Subdominios ENS configurados
- ✅ Verificación de dominio automática

### Subdominios ENS
```
chat.aegis-openagi.eth     → SecureChat+
wallet.aegis-openagi.eth   → AEGIS Wallet
defi.aegis-openagi.eth     → DeFi Hub
nft.aegis-openagi.eth      → NFT Marketplace
dao.aegis-openagi.eth      → DAO Governance
api.aegis-openagi.eth      → API Gateway
```

## Configuración SSL/TLS

### Script Automatizado
```bash
# Ejecutar en el servidor
chmod +x /root/ssl-setup.sh
./ssl-setup.sh
```

### Certificados Let's Encrypt
- ✅ Script preparado para todos los dominios
- ✅ Renovación automática configurada
- ✅ Headers de seguridad incluidos
- ✅ Configuración SSL moderna

### Dominios SSL Incluidos
- `conexionsecreta.net`
- `www.conexionsecreta.net`
- `chat.conexionsecreta.net`
- `wallet.conexionsecreta.net`
- `defi.conexionsecreta.net`
- `nft.conexionsecreta.net`
- `dao.conexionsecreta.net`
- `market.conexionsecreta.net`

## Estado de Implementación

### ✅ Completado
- [x] Configuración Nginx multi-dominio
- [x] Integración JavaScript de dominios
- [x] Módulo ENS completo
- [x] Portal DApps principal
- [x] Configuración de endpoints dinámicos
- [x] Script SSL automatizado
- [x] Documentación completa

### 🔄 En Progreso
- [ ] Configuración SSL/TLS (script listo)
- [ ] Verificación DNS de conexionsecreta.net
- [ ] Configuración ENS en blockchain

### 📋 Pendiente
- [ ] Desarrollo de DApps individuales
- [ ] Configuración de subdominios DNS
- [ ] Optimización de rendimiento
- [ ] Monitoreo y analytics

## Comandos de Verificación

### Verificar Nginx
```bash
nginx -t
systemctl status nginx
```

### Verificar SSL
```bash
certbot certificates
openssl s_client -connect conexionsecreta.net:443
```

### Verificar DNS
```bash
nslookup conexionsecreta.net
dig conexionsecreta.net
```

### Verificar ENS
```bash
# En consola del navegador
await window.ensIntegration.verifyDomain();
```

## Próximos Pasos

1. **Configurar DNS**: Apuntar conexionsecreta.net al servidor
2. **Ejecutar SSL**: Correr el script ssl-setup.sh
3. **Configurar ENS**: Registrar y configurar aegis-openagi.eth
4. **Desarrollar DApps**: Implementar las aplicaciones restantes
5. **Optimizar**: Configurar CDN y cache

## Contacto y Soporte

- **Servidor**: 77.237.235.224
- **Usuario**: root
- **Configuración**: /etc/nginx/sites-available/aegis
- **Logs**: /var/log/nginx/
- **SSL**: /etc/letsencrypt/

---

*Documentación actualizada: $(date)*
*Estado: Integración de dominios completada - SSL pendiente*