# AEGIS - Referencia Completa de API

## Índice
- [Introducción](#introducción)
- [Autenticación y Seguridad](#autenticación-y-seguridad)
- [API del Framework Criptográfico](#api-del-framework-criptográfico)
- [API de Red P2P](#api-de-red-p2p)
- [API de Consenso](#api-de-consenso)
- [API de Almacenamiento](#api-de-almacenamiento)
- [API de Monitoreo](#api-de-monitoreo)
- [API de Métricas](#api-de-métricas)
- [API de Alertas](#api-de-alertas)
- [API de Backup](#api-de-backup)
- [API del Dashboard Web](#api-del-dashboard-web)
- [Códigos de Error](#códigos-de-error)
- [Ejemplos de Uso](#ejemplos-de-uso)

## Introducción

Esta documentación proporciona una referencia completa de todas las APIs disponibles en el sistema AEGIS. Cada módulo expone interfaces bien definidas para interactuar con sus funcionalidades.

### Convenciones de la API

- **Formato de Respuesta**: JSON
- **Codificación**: UTF-8
- **Autenticación**: Token-based o certificados X.509
- **Versionado**: Semantic Versioning (SemVer)
- **Rate Limiting**: Configurable por endpoint

## Autenticación y Seguridad

### Tipos de Autenticación

#### 1. Token-based Authentication
```python
# Generar token de acceso
token = client_auth_manager.generate_access_token(
    client_id="aegis_client_001",
    permissions=["read", "write", "admin"]
)

# Usar token en requests
headers = {"Authorization": f"Bearer {token}"}
```

#### 2. Certificados X.509
```python
# Configurar autenticación por certificado
cert_config = {
    "cert_file": "client.crt",
    "key_file": "client.key",
    "ca_file": "ca.crt"
}
```

### Endpoints de Autenticación

#### POST /auth/token
Genera un nuevo token de acceso.

**Request:**
```json
{
    "client_id": "string",
    "client_secret": "string",
    "scope": ["read", "write", "admin"]
}
```

**Response:**
```json
{
    "access_token": "string",
    "token_type": "Bearer",
    "expires_in": 3600,
    "scope": ["read", "write"]
}
```

#### POST /auth/refresh
Renueva un token existente.

**Request:**
```json
{
    "refresh_token": "string"
}
```

#### DELETE /auth/revoke
Revoca un token.

**Request:**
```json
{
    "token": "string"
}
```

## API del Framework Criptográfico

### Clase CryptoFramework

#### Generación de Claves

##### generate_keypair(algorithm, key_size)
Genera un par de claves asimétricas.

**Parámetros:**
- `algorithm` (str): "RSA", "ECC", "Ed25519"
- `key_size` (int): Tamaño de clave (2048, 3072, 4096 para RSA)

**Retorna:**
```python
{
    "public_key": "-----BEGIN PUBLIC KEY-----...",
    "private_key": "-----BEGIN PRIVATE KEY-----...",
    "algorithm": "RSA",
    "key_size": 2048,
    "created_at": "2024-01-15T10:30:00Z"
}
```

**Ejemplo:**
```python
crypto = CryptoFramework()
keypair = crypto.generate_keypair("RSA", 2048)
```

##### generate_symmetric_key(algorithm, key_size)
Genera una clave simétrica.

**Parámetros:**
- `algorithm` (str): "AES", "ChaCha20"
- `key_size` (int): 128, 192, 256

**Retorna:**
```python
{
    "key": "base64_encoded_key",
    "algorithm": "AES",
    "key_size": 256,
    "iv": "base64_encoded_iv"
}
```

#### Cifrado y Descifrado

##### encrypt_data(data, key, algorithm)
Cifra datos usando el algoritmo especificado.

**Parámetros:**
- `data` (bytes): Datos a cifrar
- `key` (str): Clave de cifrado
- `algorithm` (str): Algoritmo de cifrado

**Retorna:**
```python
{
    "encrypted_data": "base64_encoded_data",
    "algorithm": "AES-256-GCM",
    "iv": "base64_encoded_iv",
    "tag": "base64_encoded_tag"
}
```

##### decrypt_data(encrypted_data, key, algorithm)
Descifra datos.

**Parámetros:**
- `encrypted_data` (str): Datos cifrados en base64
- `key` (str): Clave de descifrado
- `algorithm` (str): Algoritmo usado

**Retorna:**
```python
{
    "decrypted_data": bytes,
    "verified": True
}
```

#### Funciones Hash

##### hash_data(data, algorithm)
Genera hash de datos.

**Parámetros:**
- `data` (bytes): Datos a hashear
- `algorithm` (str): "SHA256", "SHA3-256", "BLAKE2b"

**Retorna:**
```python
{
    "hash": "hex_encoded_hash",
    "algorithm": "SHA256"
}
```

#### Firmas Digitales

##### sign_data(data, private_key, algorithm)
Firma datos digitalmente.

**Parámetros:**
- `data` (bytes): Datos a firmar
- `private_key` (str): Clave privada
- `algorithm` (str): Algoritmo de firma

**Retorna:**
```python
{
    "signature": "base64_encoded_signature",
    "algorithm": "RSA-PSS",
    "hash_algorithm": "SHA256"
}
```

##### verify_signature(data, signature, public_key, algorithm)
Verifica una firma digital.

**Retorna:**
```python
{
    "valid": True,
    "algorithm": "RSA-PSS",
    "verified_at": "2024-01-15T10:30:00Z"
}
```

### Endpoints HTTP del Crypto Framework

#### POST /crypto/encrypt
Cifra datos vía HTTP.

**Request:**
```json
{
    "data": "base64_encoded_data",
    "algorithm": "AES-256-GCM",
    "key_id": "key_identifier"
}
```

#### POST /crypto/decrypt
Descifra datos vía HTTP.

#### POST /crypto/sign
Firma datos vía HTTP.

#### POST /crypto/verify
Verifica firma vía HTTP.

## API de Red P2P

### Clase P2PNetwork

#### Gestión de Conexiones

##### start_network(port, max_peers)
Inicia la red P2P.

**Parámetros:**
- `port` (int): Puerto de escucha
- `max_peers` (int): Máximo número de peers

**Retorna:**
```python
{
    "status": "started",
    "listening_port": 8080,
    "node_id": "unique_node_identifier",
    "max_peers": 50
}
```

##### connect_to_peer(address, port)
Conecta a un peer específico.

**Parámetros:**
- `address` (str): Dirección IP o hostname
- `port` (int): Puerto del peer

**Retorna:**
```python
{
    "connected": True,
    "peer_id": "peer_identifier",
    "connection_time": "2024-01-15T10:30:00Z",
    "protocol_version": "1.0"
}
```

##### disconnect_from_peer(peer_id)
Desconecta de un peer.

##### get_connected_peers()
Obtiene lista de peers conectados.

**Retorna:**
```python
{
    "peers": [
        {
            "peer_id": "peer_001",
            "address": "192.168.1.100",
            "port": 8080,
            "connected_at": "2024-01-15T10:30:00Z",
            "status": "active",
            "latency_ms": 45
        }
    ],
    "total_peers": 1
}
```

#### Mensajería

##### send_message(peer_id, message_type, data)
Envía mensaje a un peer específico.

**Parámetros:**
- `peer_id` (str): ID del peer destinatario
- `message_type` (str): Tipo de mensaje
- `data` (dict): Datos del mensaje

**Retorna:**
```python
{
    "sent": True,
    "message_id": "msg_12345",
    "timestamp": "2024-01-15T10:30:00Z",
    "peer_id": "peer_001"
}
```

##### broadcast_message(message_type, data, exclude_peers)
Difunde mensaje a todos los peers.

**Parámetros:**
- `message_type` (str): Tipo de mensaje
- `data` (dict): Datos del mensaje
- `exclude_peers` (list): Peers a excluir

##### register_message_handler(message_type, handler_function)
Registra manejador para tipo de mensaje.

#### Descubrimiento de Peers

##### discover_peers(method)
Descubre nuevos peers.

**Parámetros:**
- `method` (str): "mdns", "dht", "bootstrap"

**Retorna:**
```python
{
    "discovered_peers": [
        {
            "address": "192.168.1.101",
            "port": 8080,
            "node_id": "node_002",
            "services": ["consensus", "storage"]
        }
    ],
    "method": "mdns",
    "discovery_time": "2024-01-15T10:30:00Z"
}
```

### Endpoints HTTP de P2P

#### GET /p2p/status
Obtiene estado de la red P2P.

**Response:**
```json
{
    "status": "running",
    "listening_port": 8080,
    "connected_peers": 5,
    "max_peers": 50,
    "node_id": "node_001",
    "uptime_seconds": 3600
}
```

#### GET /p2p/peers
Lista peers conectados.

#### POST /p2p/connect
Conecta a un nuevo peer.

#### DELETE /p2p/peers/{peer_id}
Desconecta de un peer.

#### POST /p2p/broadcast
Difunde mensaje a todos los peers.

## API de Consenso

### Clase ConsensusAlgorithm

#### Gestión de Consenso

##### start_consensus()
Inicia el algoritmo de consenso.

**Retorna:**
```python
{
    "status": "started",
    "algorithm": "PBFT",
    "node_role": "validator",
    "view_number": 0
}
```

##### propose_value(value, metadata)
Propone un valor para consenso.

**Parámetros:**
- `value` (any): Valor a proponer
- `metadata` (dict): Metadatos adicionales

**Retorna:**
```python
{
    "proposal_id": "prop_12345",
    "status": "proposed",
    "timestamp": "2024-01-15T10:30:00Z",
    "round": 1
}
```

##### get_consensus_state()
Obtiene estado actual del consenso.

**Retorna:**
```python
{
    "current_round": 5,
    "current_view": 2,
    "status": "in_progress",
    "participants": 7,
    "required_votes": 5,
    "current_votes": 3,
    "leader": "node_003"
}
```

#### Votación

##### cast_vote(proposal_id, vote, signature)
Emite voto sobre una propuesta.

**Parámetros:**
- `proposal_id` (str): ID de la propuesta
- `vote` (bool): True para aceptar, False para rechazar
- `signature` (str): Firma del voto

##### get_vote_results(proposal_id)
Obtiene resultados de votación.

**Retorna:**
```python
{
    "proposal_id": "prop_12345",
    "total_votes": 7,
    "accept_votes": 5,
    "reject_votes": 2,
    "status": "accepted",
    "finalized_at": "2024-01-15T10:35:00Z"
}
```

### Endpoints HTTP de Consenso

#### GET /consensus/status
Estado del consenso.

#### POST /consensus/propose
Crear nueva propuesta.

#### POST /consensus/vote
Emitir voto.

#### GET /consensus/proposals
Listar propuestas.

## API de Almacenamiento

### Gestión de Datos

#### store_data(key, value, metadata)
Almacena datos en el sistema distribuido.

**Parámetros:**
- `key` (str): Clave única
- `value` (any): Datos a almacenar
- `metadata` (dict): Metadatos

**Retorna:**
```python
{
    "stored": True,
    "key": "data_key_001",
    "hash": "sha256_hash",
    "size_bytes": 1024,
    "replicas": 3,
    "stored_at": "2024-01-15T10:30:00Z"
}
```

#### retrieve_data(key)
Recupera datos por clave.

#### delete_data(key)
Elimina datos.

#### list_keys(prefix, limit)
Lista claves disponibles.

### Endpoints HTTP de Almacenamiento

#### POST /storage/data
Almacenar datos.

#### GET /storage/data/{key}
Recuperar datos.

#### DELETE /storage/data/{key}
Eliminar datos.

#### GET /storage/keys
Listar claves.

## API de Monitoreo

### Clase MonitoringDashboard

#### Métricas del Sistema

##### get_system_metrics()
Obtiene métricas del sistema.

**Retorna:**
```python
{
    "cpu_usage": 45.2,
    "memory_usage": 67.8,
    "disk_usage": 23.1,
    "network_io": {
        "bytes_sent": 1048576,
        "bytes_received": 2097152
    },
    "timestamp": "2024-01-15T10:30:00Z"
}
```

##### get_performance_metrics()
Métricas de rendimiento.

##### get_security_metrics()
Métricas de seguridad.

### Endpoints HTTP de Monitoreo

#### GET /monitoring/system
Métricas del sistema.

#### GET /monitoring/performance
Métricas de rendimiento.

#### GET /monitoring/security
Métricas de seguridad.

#### GET /monitoring/alerts
Alertas activas.

## API de Métricas

### Clase MetricsCollector

#### Recolección de Métricas

##### collect_metrics(component, metric_type)
Recolecta métricas de un componente.

##### get_historical_data(metric_name, time_range)
Obtiene datos históricos.

##### export_metrics(format)
Exporta métricas en formato específico.

### Endpoints HTTP de Métricas

#### GET /metrics/prometheus
Métricas en formato Prometheus.

#### GET /metrics/json
Métricas en formato JSON.

#### POST /metrics/custom
Registrar métrica personalizada.

## API de Alertas

### Clase AlertSystem

#### Gestión de Alertas

##### create_alert(severity, message, component)
Crea nueva alerta.

##### get_active_alerts()
Obtiene alertas activas.

##### acknowledge_alert(alert_id)
Reconoce una alerta.

##### resolve_alert(alert_id)
Resuelve una alerta.

### Endpoints HTTP de Alertas

#### GET /alerts
Listar alertas.

#### POST /alerts
Crear alerta.

#### PUT /alerts/{alert_id}/acknowledge
Reconocer alerta.

#### PUT /alerts/{alert_id}/resolve
Resolver alerta.

## API de Backup

### Clase BackupSystem

#### Operaciones de Backup

##### create_backup(backup_type, components)
Crea backup del sistema.

**Parámetros:**
- `backup_type` (str): "full", "incremental", "differential"
- `components` (list): Componentes a respaldar

**Retorna:**
```python
{
    "backup_id": "backup_20240115_103000",
    "type": "full",
    "size_bytes": 10485760,
    "components": ["crypto", "p2p", "consensus"],
    "created_at": "2024-01-15T10:30:00Z",
    "location": "s3://aegis-backups/backup_20240115_103000.tar.gz.enc"
}
```

##### restore_backup(backup_id, components)
Restaura desde backup.

##### list_backups(limit, offset)
Lista backups disponibles.

##### verify_backup(backup_id)
Verifica integridad de backup.

### Endpoints HTTP de Backup

#### POST /backup/create
Crear backup.

#### POST /backup/restore
Restaurar backup.

#### GET /backup/list
Listar backups.

#### POST /backup/verify
Verificar backup.

## API del Dashboard Web

### Endpoints del Dashboard

#### GET /dashboard
Página principal del dashboard.

#### GET /api/dashboard/overview
Resumen del sistema.

#### GET /api/dashboard/components
Estado de componentes.

#### POST /api/dashboard/command
Ejecutar comando.

## Códigos de Error

### Códigos HTTP Estándar

- **200 OK**: Operación exitosa
- **201 Created**: Recurso creado
- **400 Bad Request**: Solicitud inválida
- **401 Unauthorized**: No autorizado
- **403 Forbidden**: Prohibido
- **404 Not Found**: Recurso no encontrado
- **429 Too Many Requests**: Límite de rate excedido
- **500 Internal Server Error**: Error interno

### Códigos de Error Específicos de AEGIS

```json
{
    "error": {
        "code": "AEGIS_CRYPTO_001",
        "message": "Invalid encryption algorithm",
        "details": "Algorithm 'INVALID_ALG' is not supported",
        "timestamp": "2024-01-15T10:30:00Z"
    }
}
```

#### Códigos de Criptografía (AEGIS_CRYPTO_XXX)
- **AEGIS_CRYPTO_001**: Algoritmo de cifrado inválido
- **AEGIS_CRYPTO_002**: Clave de cifrado inválida
- **AEGIS_CRYPTO_003**: Error de verificación de firma

#### Códigos de P2P (AEGIS_P2P_XXX)
- **AEGIS_P2P_001**: Error de conexión a peer
- **AEGIS_P2P_002**: Mensaje inválido
- **AEGIS_P2P_003**: Peer no encontrado

#### Códigos de Consenso (AEGIS_CONSENSUS_XXX)
- **AEGIS_CONSENSUS_001**: Propuesta inválida
- **AEGIS_CONSENSUS_002**: Voto duplicado
- **AEGIS_CONSENSUS_003**: Consenso no alcanzado

## Ejemplos de Uso

### Ejemplo 1: Cifrado de Datos
```python
import requests
import base64

# Datos a cifrar
data = "Información confidencial"
data_b64 = base64.b64encode(data.encode()).decode()

# Solicitud de cifrado
response = requests.post("http://localhost:8080/crypto/encrypt", 
    json={
        "data": data_b64,
        "algorithm": "AES-256-GCM",
        "key_id": "master_key_001"
    },
    headers={"Authorization": "Bearer your_token_here"}
)

encrypted_data = response.json()
print(f"Datos cifrados: {encrypted_data['encrypted_data']}")
```

### Ejemplo 2: Envío de Mensaje P2P
```python
# Enviar mensaje a peer específico
response = requests.post("http://localhost:8080/p2p/send",
    json={
        "peer_id": "peer_001",
        "message_type": "data_sync",
        "data": {
            "operation": "update",
            "payload": "datos_actualizados"
        }
    }
)

result = response.json()
print(f"Mensaje enviado: {result['sent']}")
```

### Ejemplo 3: Crear Propuesta de Consenso
```python
# Crear nueva propuesta
response = requests.post("http://localhost:8080/consensus/propose",
    json={
        "value": {
            "action": "update_config",
            "parameters": {"max_peers": 100}
        },
        "metadata": {
            "priority": "high",
            "timeout": 300
        }
    }
)

proposal = response.json()
print(f"Propuesta creada: {proposal['proposal_id']}")
```

### Ejemplo 4: Monitoreo del Sistema
```python
# Obtener métricas del sistema
response = requests.get("http://localhost:8080/monitoring/system")
metrics = response.json()

print(f"CPU: {metrics['cpu_usage']}%")
print(f"Memoria: {metrics['memory_usage']}%")
print(f"Red: {metrics['network_io']['bytes_sent']} bytes enviados")
```

### Ejemplo 5: Crear Backup
```python
# Crear backup completo
response = requests.post("http://localhost:8080/backup/create",
    json={
        "backup_type": "full",
        "components": ["crypto", "p2p", "consensus", "storage"],
        "compression": True,
        "encryption": True
    }
)

backup = response.json()
print(f"Backup creado: {backup['backup_id']}")
print(f"Ubicación: {backup['location']}")
```

## Versionado de API

AEGIS utiliza versionado semántico para sus APIs:

- **Versión Mayor**: Cambios incompatibles
- **Versión Menor**: Nueva funcionalidad compatible
- **Versión Parche**: Correcciones de bugs

### Headers de Versión
```http
API-Version: 1.2.3
Accept-Version: 1.x
```

## Rate Limiting

Límites por defecto:
- **Autenticación**: 10 requests/minuto
- **APIs generales**: 100 requests/minuto
- **APIs de datos**: 1000 requests/minuto

Headers de respuesta:
```http
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1642248600
```

## Webhooks

AEGIS soporta webhooks para notificaciones en tiempo real:

### Configuración de Webhook
```json
{
    "url": "https://your-app.com/webhook",
    "events": ["consensus.proposal", "p2p.peer_connected"],
    "secret": "webhook_secret_key"
}
```

### Eventos Disponibles
- `consensus.proposal`: Nueva propuesta de consenso
- `consensus.decision`: Decisión de consenso alcanzada
- `p2p.peer_connected`: Nuevo peer conectado
- `p2p.peer_disconnected`: Peer desconectado
- `system.alert`: Nueva alerta del sistema
- `backup.completed`: Backup completado

---

*Esta documentación está en constante evolución. Para la versión más actualizada, consulte el repositorio oficial de AEGIS.*