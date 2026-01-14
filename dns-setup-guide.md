# 🌐 Guía de Configuración DNS para conexionsecreta.site

## 📋 **Registros DNS Requeridos**

### **Registros A (IPv4)**
```
Tipo: A
Nombre: @
Valor: 77.237.235.224
TTL: 300 (5 minutos)

Tipo: A  
Nombre: www
Valor: 77.237.235.224
TTL: 300
```

### **Subdominios para DApps**
```
Tipo: A
Nombre: chat
Valor: 77.237.235.224
TTL: 300

Tipo: A
Nombre: wallet  
Valor: 77.237.235.224
TTL: 300

Tipo: A
Nombre: defi
Valor: 77.237.235.224
TTL: 300

Tipo: A
Nombre: nft
Valor: 77.237.235.224
TTL: 300

Tipo: A
Nombre: dao
Valor: 77.237.235.224
TTL: 300

Tipo: A
Nombre: market
Valor: 77.237.235.224
TTL: 300
```

### **Registro ENS (Opcional)**
```
Tipo: A
Nombre: aegis-openagi
Valor: 77.237.235.224
TTL: 300

Tipo: A
Nombre: www.aegis-openagi
Valor: 77.237.235.224
TTL: 300
```

## 🔧 **Pasos de Configuración**

### **1. Acceder al Panel DNS**
- Inicia sesión en tu registrador de dominios
- Busca la sección "DNS Management" o "Zone File"
- Selecciona el dominio `conexionsecreta.site`

### **2. Configurar Registros Principales**
1. **Dominio Principal**: `conexionsecreta.site` → `77.237.235.224`
2. **WWW**: `www.conexionsecreta.site` → `77.237.235.224`

### **3. Configurar Subdominios DApps**
- Agregar cada subdominio listado arriba
- Todos apuntan a la misma IP: `77.237.235.224`

### **4. Verificar Propagación**
```bash
# Verificar resolución DNS
nslookup conexionsecreta.site
nslookup www.conexionsecreta.site
nslookup chat.conexionsecreta.site
```

## ⏱️ **Tiempos de Propagación**
- **TTL 300**: 5-15 minutos
- **Propagación Global**: 1-24 horas
- **Verificación Local**: Inmediata

## 🚀 **Después de Configurar DNS**

Una vez que el DNS esté propagado:

1. **Verificar resolución**: `ping conexionsecreta.site`
2. **Ejecutar SSL automático**: El script detectará el dominio
3. **Probar HTTPS**: `https://conexionsecreta.site`

## 📞 **Contacto de Soporte**
- **IP del Servidor**: 77.237.235.224
- **Puertos Requeridos**: 80 (HTTP), 443 (HTTPS)
- **Email SSL**: admin@conexionsecreta.site

---
**Nota**: Guarda este archivo para referencia futura y configuraciones adicionales.