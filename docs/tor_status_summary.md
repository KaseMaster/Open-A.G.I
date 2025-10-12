# Resumen de Estado TOR

Este documento resume la configuración y el estado actual del servicio TOR integrado en AEGIS.

## Configuración Principal

- ControlPort: 127.0.0.1:9051
- SocksPort: 127.0.0.1:9050
- HiddenServiceDir: G:\\Open A.G.I\\onion_service
- ClientOnionAuthDir: G:\\Open A.G.I\\client_onion_auth
- HashedControlPassword: configurado (ver `config/torrc`)
- TOR_CONTROL_PASSWORD (entorno): `aegis_tor_password`

## Hostname del Servicio Onion

El hostname actual se encuentra en:

```
onion_service/hostname
```

Ejemplo:

```
dbx2uivnvaodz7zuatxms5jcuu7xkn3bhmg2tvtddncrul4ywuzq23ad.onion
```

## Autorización de Clientes

Directorio de clientes autorizados en el servidor:

```
onion_service/authorized_clients/
```

Para crear credenciales de cliente:

```
python generate_client_auth.py -n <nombre_cliente>
```

Coloca el archivo `<nombre_cliente>.auth_private` en el directorio indicado por `ClientOnionAuthDir` en el cliente.

## Estado de Red y Gateway

El módulo `tor_integration` expone:

- `start_tor_service(config)`: Inicializa y autentica el Gateway de TOR.
- `get_tor_status()`: Devuelve un dict con información de circuitos, bootstrap y nodos.
- `stop_tor_service()`: Detiene el Gateway y libera recursos.

## Problemas Comunes y Soluciones

- Error de acceso al Onion (403/timeout):
  - Verificar que el cliente tenga `.auth_private` en `ClientOnionAuthDir`.
  - Reiniciar Tor para recargar `authorized_clients`.

- Fallo de autenticación al ControlPort:
  - Asegurar que `TOR_CONTROL_PASSWORD` coincide con `HashedControlPassword` en `torrc`.
  - Si no hay contraseña, usar cookie auth asegurando permisos del `Tor Browser`/servicio.

- Dashboard no accesible a través de Onion:
  - Confirmar que el dashboard local está corriendo (por defecto en 8090/8080 según configuración).
  - Revisar el mapeo `HiddenServicePort` en `config/torrc` hacia `127.0.0.1:8090`.

## Comandos Útiles

- Arrancar nodo:

```
python main.py start-node
```

- Probar acceso Onion:

```
python scripts/test_onion_access.py
```