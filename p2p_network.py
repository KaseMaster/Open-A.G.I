#!/usr/bin/env python3
"""
Red P2P Distribuida - AEGIS Framework
Implementación de red peer-to-peer con descubrimiento automático de nodos,
gestión de topología dinámica y comunicación resiliente.

Características principales:
- Descubrimiento automático de peers
- Topología de red adaptativa
- Enrutamiento inteligente de mensajes
- Balanceado de carga distribuido
- Recuperación automática de conexiones
"""

import asyncio
import time
import json
import socket
import logging
import base64
from typing import Dict, List, Set, Any, Optional, Callable, Awaitable
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict, deque
import ipaddress

# Configuración de logging temprana para permitir avisos durante imports opcionales
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# Dependencias opcionales: algunas pueden no estar disponibles en CI/Windows
try:
    import aiohttp  # type: ignore
    HAS_AIOHTTP = True
except Exception:
    aiohttp = None  # type: ignore
    HAS_AIOHTTP = False
    logger.warning("aiohttp no disponible; funcionalidades HTTP quedarán deshabilitadas en este entorno.")

try:
    import websockets  # type: ignore
    HAS_WEBSOCKETS = True
except Exception:
    websockets = None  # type: ignore
    HAS_WEBSOCKETS = False
    logger.warning("websockets no disponible; funcionalidades WS quedarán deshabilitadas en este entorno.")

try:
    from zeroconf import ServiceInfo, Zeroconf, ServiceBrowser, ServiceListener  # type: ignore
    HAS_ZEROCONF = True
except Exception:
    ServiceInfo = Zeroconf = ServiceBrowser = ServiceListener = None  # type: ignore
    HAS_ZEROCONF = False
    logger.warning("zeroconf no disponible; descubrimiento mDNS se omitirá en este entorno.")

try:
    import netifaces  # type: ignore
    HAS_NETIFACES = True
except Exception:
    netifaces = None  # type: ignore
    HAS_NETIFACES = False
    logger.warning("netifaces no disponible; detección de IP local puede ser limitada en este entorno.")

# (logger ya configurado arriba)

# Integración criptográfica opcional
try:
    from crypto_framework import CryptoEngine, SecureMessage, initialize_crypto, create_crypto_engine  # type: ignore
except Exception:
    CryptoEngine = None  # type: ignore
    SecureMessage = None  # type: ignore
    initialize_crypto = None  # type: ignore
    create_crypto_engine = None  # type: ignore
    logger.warning("crypto_framework no disponible; los canales seguros estarán deshabilitados.")


class NodeType(Enum):
    """Tipos de nodos en la red"""
    BOOTSTRAP = "bootstrap"
    FULL = "full"
    LIGHT = "light"
    VALIDATOR = "validator"
    STORAGE = "storage"


class ConnectionStatus(Enum):
    """Estados de conexión de peers"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    AUTHENTICATED = "authenticated"
    FAILED = "failed"


class MessageType(Enum):
    """Tipos de mensajes en la red P2P"""
    DISCOVERY = "discovery"
    HANDSHAKE = "handshake"
    HEARTBEAT = "heartbeat"
    DATA = "data"
    CONSENSUS = "consensus"
    SYNC = "sync"
    BROADCAST = "broadcast"


class NetworkProtocol(Enum):
    """Protocolos de red soportados"""
    TCP = "tcp"
    UDP = "udp"
    WEBSOCKET = "websocket"
    HTTP = "http"


@dataclass
class PeerInfo:
    """Información de un peer en la red"""
    peer_id: str
    node_type: NodeType
    ip_address: str
    port: int
    public_key: str
    capabilities: List[str]
    last_seen: float
    connection_status: ConnectionStatus
    reputation_score: float
    latency: float
    bandwidth: int
    supported_protocols: List[NetworkProtocol]


@dataclass
class NetworkMessage:
    """Mensaje de red P2P"""
    message_id: str
    sender_id: str
    recipient_id: str  # "*" para broadcast
    message_type: MessageType
    payload: Dict[str, Any]
    timestamp: float
    ttl: int
    signature: str
    route_path: List[str]


@dataclass
class NetworkTopology:
    """Topología de la red"""
    total_nodes: int
    connected_nodes: int
    network_diameter: int
    clustering_coefficient: float
    average_path_length: float
    node_degrees: Dict[str, int]
    critical_nodes: List[str]


class AegisServiceListenerExt(ServiceListener):
    """Listener mDNS que programa callbacks en el loop principal de asyncio.
    Maneja add/update/remove de servicios de manera segura desde el hilo de Zeroconf.
    """
    def __init__(self, discovery_service, loop: asyncio.AbstractEventLoop):
        self.discovery_service = discovery_service
        self.loop = loop

    def add_service(self, zc: Zeroconf, type_: str, name: str):
        try:
            # Evitar programar tareas si el loop ya está cerrado
            if getattr(self.loop, "is_closed", lambda: False)():
                logger.debug("🔄 Loop de asyncio cerrado; ignorando add_service durante shutdown")
                return
            asyncio.run_coroutine_threadsafe(
                self._handle_service_added(zc, type_, name), self.loop
            )
        except Exception as e:
            logger.error(f"❌ Error programando manejo de servicio mDNS: {e!r}")

    async def _handle_service_added(self, zc: Zeroconf, type_: str, name: str):
        try:
            info = await asyncio.to_thread(zc.get_service_info, type_, name)
            if info and info.properties:
                await self.discovery_service._process_discovered_service(info)
        except Exception as e:
            logger.error(f"❌ Error procesando servicio descubierto: {e}")

    def remove_service(self, zc: Zeroconf, type_: str, name: str):
        try:
            # Evitar programar tareas si el loop ya está cerrado
            if getattr(self.loop, "is_closed", lambda: False)():
                logger.debug("🔄 Loop de asyncio cerrado; ignorando remove_service durante shutdown")
                return
            asyncio.run_coroutine_threadsafe(
                self._handle_service_removed(zc, type_, name), self.loop
            )
        except Exception as e:
            logger.error(f"❌ Error programando eliminación de servicio mDNS: {e!r}")

    async def _handle_service_removed(self, zc: Zeroconf, type_: str, name: str):
        try:
            # Durante el shutdown, Zeroconf puede estar ya cerrado. Evitamos consultar get_service_info.
            # Extraemos el node_id del nombre del servicio (formato: "<node_id>.<type>")
            node_id: Optional[str] = None
            try:
                # Ejemplo de name: "smoke_node_2._aegis._tcp.local."
                # Tomamos la primera etiqueta antes del primer punto
                node_id = name.split('.')[0] if name else None
            except Exception:
                node_id = None

            await self.discovery_service._handle_service_removed(node_id)
        except Exception as e:
            logger.error(f"❌ Error procesando eliminación de servicio mDNS: {e!r}")

    def update_service(self, zc: Zeroconf, type_: str, name: str):
        try:
            # Evitar programar tareas si el loop ya está cerrado
            if getattr(self.loop, "is_closed", lambda: False)():
                logger.debug("🔄 Loop de asyncio cerrado; ignorando update_service durante shutdown")
                return
            asyncio.run_coroutine_threadsafe(
                self._handle_service_updated(zc, type_, name), self.loop
            )
        except Exception as e:
            logger.error(f"❌ Error programando actualización de servicio mDNS: {e!r}")

    async def _handle_service_updated(self, zc: Zeroconf, type_: str, name: str):
        try:
            info = await asyncio.to_thread(zc.get_service_info, type_, name)
            if info and info.properties:
                await self.discovery_service._process_discovered_service(info)
        except Exception as e:
            logger.error(f"❌ Error procesando actualización de servicio mDNS: {e}")


class PeerDiscoveryService:
    """Servicio de descubrimiento de peers"""

    def __init__(self, node_id: str, node_type: NodeType, port: int):
        self.node_id = node_id
        self.node_type = node_type
        self.port = port
        self.discovered_peers: Dict[str, PeerInfo] = {}
        self.zeroconf = None
        self.service_info = None
        self.discovery_active = False
        # Referencias para una parada más limpia
        self._mdns_browser = None
        self._mdns_task: Optional[asyncio.Task] = None

        # Configuración de descubrimiento
        self.service_type = "_aegis._tcp.local."
        self.discovery_interval = 30  # segundos
        self.peer_timeout = 300  # 5 minutos

    async def start_discovery(self):
        """Inicia el servicio de descubrimiento"""
        try:
            self.discovery_active = True
            logger.info(f"🔍 Iniciando descubrimiento de peers para nodo {self.node_id}")

            # Configurar Zeroconf si disponible
            if HAS_ZEROCONF:
                await self._setup_zeroconf()
                # Si durante la configuración se solicitó detener, limpiar y salir
                if not self.discovery_active:
                    await self.stop_discovery()
                    return
            else:
                logger.warning("mDNS/Zeroconf no disponible; se usarán métodos alternativos.")

            # Iniciar tareas de descubrimiento
            tasks = []
            # Solo iniciar mDNS si Zeroconf fue configurado correctamente
            if HAS_ZEROCONF and self.zeroconf is not None:
                self._mdns_task = asyncio.create_task(self._mdns_discovery())
                tasks.append(self._mdns_task)
            tasks.extend([
                asyncio.create_task(self._bootstrap_discovery()),
                asyncio.create_task(self._peer_maintenance()),
                asyncio.create_task(self._network_scanning())
            ])

            await asyncio.gather(*tasks)

        except Exception as e:
            logger.error(f"❌ Error iniciando descubrimiento: {e}")

    async def _setup_zeroconf(self):
        """Configura el servicio Zeroconf/mDNS"""
        try:
            if not HAS_ZEROCONF:
                logger.warning("Saltando configuración Zeroconf: dependencias no disponibles.")
                return
            # Si se ha solicitado detener antes de configurar, salir
            if not self.discovery_active:
                logger.debug("🔄 Descubrimiento desactivado; omitiendo setup Zeroconf")
                return
            # Obtener IP local
            local_ip = self._get_local_ip()

            # Crear información del servicio
            service_name = f"{self.node_id}.{self.service_type}"

            # Propiedades del servicio
            properties = {
                'node_id': self.node_id.encode('utf-8'),
                'node_type': self.node_type.value.encode('utf-8'),
                'capabilities': json.dumps(['consensus', 'storage', 'compute']).encode('utf-8'),
                'version': '1.0.0'.encode('utf-8')
            }

            # Crear ServiceInfo
            self.service_info = ServiceInfo(
                self.service_type,
                service_name,
                addresses=[socket.inet_aton(local_ip)],
                port=self.port,
                properties=properties
            )

            # Registrar servicio (posible I/O bloqueante): ejecutar en hilo
            self.zeroconf = Zeroconf()
            # Verificar estado justo antes de registrar
            if not self.discovery_active:
                logger.debug("🔄 Descubrimiento desactivado antes de registrar; omitiendo register_service")
                return
            await asyncio.to_thread(self.zeroconf.register_service, self.service_info)

            logger.info(f"📡 Servicio mDNS registrado: {service_name}")

            # Si se desactivó el descubrimiento mientras se registraba, desregistrar inmediatamente
            if not self.discovery_active and self.zeroconf and self.service_info:
                try:
                    await asyncio.to_thread(self.zeroconf.unregister_service, self.service_info)
                except Exception as e:
                    logger.debug(f"⚠️ Error al desregistrar servicio mDNS tras cancelación: {e!r}")
                try:
                    await asyncio.to_thread(self.zeroconf.close)
                except Exception as e:
                    logger.debug(f"⚠️ Error al cerrar Zeroconf tras cancelación: {e!r}")
                finally:
                    self.service_info = None
                    self.zeroconf = None

        except Exception as e:
            logger.error(f"❌ Error configurando Zeroconf: {e}")

    def _get_local_ip(self) -> str:
        """Obtiene la IP local del nodo"""
        try:
            def _is_valid_ip(ip: str) -> bool:
                try:
                    addr = ipaddress.ip_address(ip)
                    # Excluir loopback, link-local, no especificada (0.0.0.0) y multicast
                    return not (addr.is_loopback or addr.is_link_local or addr.is_unspecified or addr.is_multicast)
                except Exception:
                    return False

            if HAS_NETIFACES:
                # Intentar obtener IP de interfaces de red
                interfaces = netifaces.interfaces()

                for interface in interfaces:
                    addresses = netifaces.ifaddresses(interface)
                    if netifaces.AF_INET in addresses:
                        for addr_info in addresses[netifaces.AF_INET]:
                            ip = addr_info['addr']
                            if _is_valid_ip(ip):
                                return ip

            # Fallback 1: resolver hostname -> IP
            try:
                hostname_ip = socket.gethostbyname(socket.gethostname())
                if hostname_ip and _is_valid_ip(hostname_ip):
                    logger.debug(f"📍 IP obtenida por hostname: {hostname_ip}")
                    return hostname_ip
            except Exception:
                pass

            # Fallback 2: enumerar direcciones con getaddrinfo
            try:
                addrs = socket.getaddrinfo(None, 0, family=socket.AF_INET)
                for addr in addrs:
                    ip = addr[4][0]
                    if ip and _is_valid_ip(ip):
                        logger.debug(f"📍 IP obtenida por getaddrinfo: {ip}")
                        return ip
            except Exception:
                pass

            # Fallback: usar socket para conectar a servidor externo
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(("8.8.8.8", 80))
                return s.getsockname()[0]

        except Exception as e:
            logger.warning(f"⚠️ Error obteniendo IP local: {e}")
            return "127.0.0.1"

    async def _mdns_discovery(self):
        """Descubrimiento usando mDNS/Zeroconf"""
        try:
            # Crear listener usando clase de módulo con add/remove/update implementados
            current_loop = asyncio.get_running_loop()
            listener = AegisServiceListenerExt(self, current_loop)

            # Iniciar browser
            browser = ServiceBrowser(self.zeroconf, self.service_type, listener)
            self._mdns_browser = browser

            # Mantener activo mientras discovery esté habilitado
            while self.discovery_active:
                await asyncio.sleep(10)

            # No cancelar aquí para evitar doble cancelación; stop_discovery se encarga
            self._mdns_browser = None

        except Exception as e:
            logger.error(f"❌ Error en descubrimiento mDNS: {e}")

    async def _process_discovered_service(self, service_info):
        """Procesa un servicio descubierto"""
        try:
            # Extraer información del peer
            properties = service_info.properties or {}

            peer_id = properties.get('node_id', b'').decode('utf-8')
            node_type_str = properties.get('node_type', b'').decode('utf-8')

            if not peer_id or peer_id == self.node_id:
                return  # Ignorar nuestro propio servicio

            # Obtener IP y puerto
            ip_address = socket.inet_ntoa(service_info.addresses[0])
            port = service_info.port

            # Crear información del peer
            peer_info = PeerInfo(
                peer_id=peer_id,
                node_type=NodeType(node_type_str) if node_type_str else NodeType.FULL,
                ip_address=ip_address,
                port=port,
                public_key="",  # Se obtendrá durante handshake
                capabilities=json.loads(properties.get('capabilities', b'[]').decode('utf-8')),
                last_seen=time.time(),
                connection_status=ConnectionStatus.DISCONNECTED,
                reputation_score=1.0,
                latency=0.0,
                bandwidth=0,
                supported_protocols=[NetworkProtocol.TCP, NetworkProtocol.WEBSOCKET]
            )

            # Agregar peer descubierto
            self.discovered_peers[peer_id] = peer_info
            logger.info(f"🔍 Peer descubierto: {peer_id} ({ip_address}:{port})")

        except Exception as e:
            logger.error(f"❌ Error procesando servicio descubierto: {e}")

    async def _handle_service_removed(self, node_id: Optional[str]):
        """Maneja la eliminación de un servicio mDNS.
        Si se conoce el node_id del servicio eliminado, se remueve de la lista de descubrimientos.
        En caso contrario, se registra el evento y el mantenimiento periódico purgará peers expirados.
        """
        try:
            if node_id and node_id in self.discovered_peers:
                del self.discovered_peers[node_id]
                logger.debug(f"🗑️ Servicio mDNS eliminado; peer removido: {node_id}")
            else:
                logger.debug("🗑️ Servicio mDNS eliminado; node_id desconocido, se purgará por expiración si aplica")
        except Exception as e:
            logger.error(f"❌ Error manejando eliminación de servicio: {e}")

    async def _bootstrap_discovery(self):
        """Descubrimiento usando nodos bootstrap conocidos"""
        # Lista de nodos bootstrap conocidos
        bootstrap_nodes = [
            ("bootstrap1.aegis.network", 8080),
            ("bootstrap2.aegis.network", 8080),
            ("127.0.0.1", 8080)  # Para testing local
        ]

        while self.discovery_active:
            try:
                for host, port in bootstrap_nodes:
                    await self._query_bootstrap_node(host, port)

                await asyncio.sleep(self.discovery_interval)

            except Exception as e:
                logger.error(f"❌ Error en descubrimiento bootstrap: {e}")
                await asyncio.sleep(10)

    async def _query_bootstrap_node(self, host: str, port: int):
        """Consulta un nodo bootstrap por peers conocidos"""
        try:
            async with aiohttp.ClientSession() as session:
                url = f"http://{host}:{port}/api/peers"

                async with session.get(url, timeout=5) as response:
                    if response.status == 200:
                        peers_data = await response.json()

                        for peer_data in peers_data.get('peers', []):
                            await self._process_bootstrap_peer(peer_data)

        except Exception as e:
            logger.debug(f"⚠️ Error consultando bootstrap {host}:{port}: {e}")

    async def _process_bootstrap_peer(self, peer_data: Dict[str, Any]):
        """Procesa información de peer desde nodo bootstrap"""
        try:
            peer_id = peer_data.get('peer_id')
            if not peer_id or peer_id == self.node_id:
                return

            peer_info = PeerInfo(
                peer_id=peer_id,
                node_type=NodeType(peer_data.get('node_type', 'full')),
                ip_address=peer_data.get('ip_address', ''),
                port=peer_data.get('port', 0),
                public_key=peer_data.get('public_key', ''),
                capabilities=peer_data.get('capabilities', []),
                last_seen=time.time(),
                connection_status=ConnectionStatus.DISCONNECTED,
                reputation_score=peer_data.get('reputation_score', 1.0),
                latency=0.0,
                bandwidth=0,
                supported_protocols=[NetworkProtocol.TCP, NetworkProtocol.WEBSOCKET]
            )

            self.discovered_peers[peer_id] = peer_info
            logger.debug(f"🔍 Peer desde bootstrap: {peer_id}")

        except Exception as e:
            logger.error(f"❌ Error procesando peer bootstrap: {e}")

    async def _network_scanning(self):
        """Escaneo de red local para descubrir peers"""
        while self.discovery_active:
            try:
                # Obtener rango de red local
                local_ip = self._get_local_ip()
                network = ipaddress.IPv4Network(f"{local_ip}/24", strict=False)

                # Escanear puertos comunes
                common_ports = [8080, 8081, 8082, 9000, 9001]

                for ip in network.hosts():
                    for port in common_ports:
                        await self._scan_peer(str(ip), port)

                await asyncio.sleep(300)  # Escanear cada 5 minutos

            except Exception as e:
                logger.error(f"❌ Error en escaneo de red: {e}")
                await asyncio.sleep(60)

    async def _scan_peer(self, ip: str, port: int):
        """Escanea un peer potencial"""
        try:
            # Intentar conexión TCP
            future = asyncio.open_connection(ip, port)
            reader, writer = await asyncio.wait_for(future, timeout=2)

            # Enviar mensaje de descubrimiento
            discovery_msg = {
                "type": "discovery",
                "node_id": self.node_id,
                "node_type": self.node_type.value
            }

            writer.write(json.dumps(discovery_msg).encode() + b'\n')
            await writer.drain()

            # Leer respuesta
            response = await asyncio.wait_for(reader.readline(), timeout=2)
            text = response.decode(errors='ignore').strip()
            if not text:
                logger.debug(f"⚠️ Respuesta vacía de descubrimiento desde {ip}:{port}")
                writer.close()
                await writer.wait_closed()
                return
            try:
                response_data = json.loads(text)
            except json.JSONDecodeError as e:
                logger.debug(f"⚠️ Respuesta de descubrimiento no-JSON desde {ip}:{port}: {e}")
                writer.close()
                await writer.wait_closed()
                return

            if response_data.get('type') == 'discovery_response':
                await self._process_scan_response(ip, port, response_data)

            writer.close()
            await writer.wait_closed()

        except Exception:
            # Ignorar errores de conexión (peer no disponible)
            pass

    async def _process_scan_response(self, ip: str, port: int, response_data: Dict[str, Any]):
        """Procesa respuesta de escaneo"""
        try:
            peer_id = response_data.get('node_id')
            if not peer_id or peer_id == self.node_id:
                return

            peer_info = PeerInfo(
                peer_id=peer_id,
                node_type=NodeType(response_data.get('node_type', 'full')),
                ip_address=ip,
                port=port,
                public_key=response_data.get('public_key', ''),
                capabilities=response_data.get('capabilities', []),
                last_seen=time.time(),
                connection_status=ConnectionStatus.DISCONNECTED,
                reputation_score=1.0,
                latency=0.0,
                bandwidth=0,
                supported_protocols=[NetworkProtocol.TCP]
            )

            self.discovered_peers[peer_id] = peer_info
            logger.info(f"🔍 Peer escaneado: {peer_id} ({ip}:{port})")

        except Exception as e:
            logger.error(f"❌ Error procesando respuesta de escaneo: {e}")

    async def _peer_maintenance(self):
        """Mantenimiento de lista de peers"""
        while self.discovery_active:
            try:
                current_time = time.time()
                expired_peers = []

                # Identificar peers expirados
                for peer_id, peer_info in self.discovered_peers.items():
                    if current_time - peer_info.last_seen > self.peer_timeout:
                        expired_peers.append(peer_id)

                # Remover peers expirados
                for peer_id in expired_peers:
                    del self.discovered_peers[peer_id]
                    logger.debug(f"🗑️ Peer expirado removido: {peer_id}")

                await asyncio.sleep(60)  # Verificar cada minuto

            except Exception as e:
                logger.error(f"❌ Error en mantenimiento de peers: {e}")
                await asyncio.sleep(30)

    def get_discovered_peers(self) -> List[PeerInfo]:
        """Obtiene lista de peers descubiertos"""
        return list(self.discovered_peers.values())

    async def stop_discovery(self):
        """Detiene el servicio de descubrimiento"""
        self.discovery_active = False

        # Apagar Zeroconf de forma segura para evitar errores del event loop
        if HAS_ZEROCONF and self.zeroconf and self.service_info:
            # Cancelar browser mDNS si sigue activo
            if self._mdns_browser is not None:
                try:
                    self._mdns_browser.cancel()
                except Exception as e:
                    logger.debug(f"⚠️ Error cancelando ServiceBrowser mDNS (stop): {e!r}")
                finally:
                    self._mdns_browser = None
            # Esperar a que la tarea mDNS finalice
            if self._mdns_task is not None:
                try:
                    await asyncio.wait_for(self._mdns_task, timeout=5)
                except Exception as e:
                    logger.debug(f"⚠️ Tarea mDNS no finalizó limpiamente: {e!r}")
                finally:
                    self._mdns_task = None
            try:
                await asyncio.to_thread(self.zeroconf.unregister_service, self.service_info)
            except Exception as e:
                logger.warning(f"⚠️ Error al desregistrar servicio Zeroconf: {e!r}")
            try:
                await asyncio.to_thread(self.zeroconf.close)
            except Exception as e:
                logger.warning(f"⚠️ Error al cerrar Zeroconf: {e!r}")
            finally:
                self.service_info = None
                self.zeroconf = None

        logger.info("🔍 Servicio de descubrimiento detenido")


class ConnectionManager:
    """Gestor de conexiones P2P"""

    def __init__(self, node_id: str, port: int, crypto_engine: Optional['CryptoEngine'] = None):
        self.node_id = node_id
        self.port = port
        self.crypto_engine: Optional['CryptoEngine'] = crypto_engine
        self.active_connections: Dict[str, Dict[str, Any]] = {}
        self.connection_pool: Dict[str, asyncio.Queue] = {}
        self.max_connections = 50
        self.connection_timeout = 30

        # Callback de aplicación para mensajes personalizados (p.ej., CONSENSUS)
        self.app_message_callback: Optional[Callable[[str, Dict[str, Any]], Awaitable[None]]] = None

        # Estadísticas de conexión
        self.connection_stats = {
            "total_connections": 0,
            "active_connections": 0,
            "failed_connections": 0,
            "bytes_sent": 0,
            "bytes_received": 0
        }

        # Métricas de latencia y seguimiento de heartbeats
        # pending_heartbeats: marca de tiempo del último heartbeat enviado por peer
        self.pending_heartbeats: Dict[str, float] = {}
        # latency_by_peer: RTT medido por último heartbeat
        self.latency_by_peer: Dict[str, float] = {}

    async def start_server(self):
        """Inicia el servidor de conexiones con fallback de puerto si está en uso"""
        candidate_ports = [self.port, 8081, 8082, 9000, 9001]
        for candidate in candidate_ports:
            try:
                server = await asyncio.start_server(
                    self._handle_incoming_connection,
                    '0.0.0.0',
                    candidate
                )

                # Actualizar puerto efectivo si cambiamos
                if candidate != self.port:
                    logger.warning(f"⚠️ Puerto {self.port} en uso; servidor P2P iniciará en puerto alternativo {candidate}")
                    self.port = candidate

                logger.info(f"🌐 Servidor P2P iniciado en puerto {self.port}")

                async with server:
                    await server.serve_forever()
                return
            except OSError as e:
                # Manejar específicamente 'address already in use'
                msg = str(e).lower()
                if getattr(e, 'winerror', None) == 10048 or 'address already in use' in msg:
                    logger.warning(f"⚠️ Puerto {candidate} en uso; intentando siguiente puerto disponible")
                    continue
                logger.error(f"❌ Error de OS iniciando servidor en puerto {candidate}: {e}")
            except Exception as e:
                logger.error(f"❌ Error iniciando servidor en puerto {candidate}: {e}")

        logger.critical("🚫 No fue posible iniciar el servidor P2P: todos los puertos candidatos están en uso o fallaron")

    async def _handle_incoming_connection(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        """Maneja conexión entrante"""
        peer_address = writer.get_extra_info('peername')
        logger.debug(f"📥 Conexión entrante desde {peer_address}")

        try:
            # Leer mensaje inicial
            data = await asyncio.wait_for(reader.readline(), timeout=10)
            text = data.decode(errors='ignore').strip()
            if not text:
                logger.warning(f"⚠️ Mensaje inicial vacío desde {peer_address}")
                try:
                    writer.write(b'{"type":"error","error":"empty_message"}\n')
                    await writer.drain()
                except Exception:
                    pass
                return
            try:
                message = json.loads(text)
            except json.JSONDecodeError as e:
                logger.warning(f"⚠️ Mensaje inicial no-JSON desde {peer_address}: {e}")
                try:
                    writer.write(b'{"type":"error","error":"invalid_json"}\n')
                    await writer.drain()
                except Exception:
                    pass
                return

            # Procesar según tipo de mensaje
            if message.get('type') == 'discovery':
                await self._handle_discovery_request(writer, message)
            elif message.get('type') == 'handshake':
                await self._handle_handshake_request(reader, writer, message)
            else:
                logger.warning(f"⚠️ Tipo de mensaje desconocido: {message.get('type')}")

        except Exception as e:
            logger.error(f"❌ Error manejando conexión entrante: {e}")
        finally:
            writer.close()
            await writer.wait_closed()

    async def _handle_discovery_request(self, writer: asyncio.StreamWriter, message: Dict[str, Any]):
        """Maneja solicitud de descubrimiento"""
        try:
            response = {
                "type": "discovery_response",
                "node_id": self.node_id,
                "node_type": "full",  # Configurar según tipo de nodo
                "capabilities": ["consensus", "storage", "compute"],
                "public_key": "placeholder_public_key"
            }

            writer.write(json.dumps(response).encode() + b'\n')
            await writer.drain()

        except Exception as e:
            logger.error(f"❌ Error respondiendo descubrimiento: {e}")

    async def _handle_handshake_request(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        message: Dict[str, Any],
    ):
        """Maneja solicitud de handshake"""
        try:
            peer_id = message.get('node_id')
            if not peer_id:
                return

            # Registrar identidad pública del peer y establecer canal seguro si se proporciona
            try:
                if self.crypto_engine and message.get('public_identity'):
                    pub_b64 = message['public_identity']
                    peer_public = {
                        'node_id': base64.b64decode(pub_b64['node_id']),
                        'signing_key': base64.b64decode(pub_b64['signing_key']),
                        'encryption_key': base64.b64decode(pub_b64['encryption_key']),
                        'created_at': base64.b64decode(pub_b64['created_at']),
                    }
                    self.crypto_engine.add_peer_identity(peer_public)
                    self.crypto_engine.establish_secure_channel(peer_id)
            except Exception as e:
                logger.warning(f"⚠️ No se pudo procesar identidad pública de {peer_id}: {e}")

            # Verificar capacidad de conexiones
            if len(self.active_connections) >= self.max_connections:
                error_response = {
                    "type": "handshake_error",
                    "error": "max_connections_reached"
                }
                writer.write(json.dumps(error_response).encode() + b'\n')
                await writer.drain()
                return

            # Crear conexión
            connection_info = {
                "peer_id": peer_id,
                "reader": reader,
                "writer": writer,
                "connected_at": time.time(),
                "last_activity": time.time(),
                "bytes_sent": 0,
                "bytes_received": 0
            }

            self.active_connections[peer_id] = connection_info
            self.connection_stats["active_connections"] += 1

            # Responder handshake
            handshake_response: Dict[str, Any] = {
                "type": "handshake_response",
                "node_id": self.node_id,
                "status": "accepted"
            }

            # Adjuntar identidad pública propia para permitir canal seguro
            try:
                if self.crypto_engine and getattr(self.crypto_engine, 'identity', None):
                    pub = self.crypto_engine.identity.export_public_identity()
                    handshake_response["public_identity"] = {
                        'node_id': base64.b64encode(pub['node_id']).decode(),
                        'signing_key': base64.b64encode(pub['signing_key']).decode(),
                        'encryption_key': base64.b64encode(pub['encryption_key']).decode(),
                        'created_at': base64.b64encode(pub['created_at']).decode(),
                    }
            except Exception as e:
                logger.warning(f"⚠️ No se pudo adjuntar identidad pública propia en respuesta de handshake: {e}")

            writer.write(json.dumps(handshake_response).encode() + b'\n')
            await writer.drain()

            logger.info(f"🤝 Handshake completado con {peer_id}")

            # Iniciar manejo de mensajes
            await self._handle_peer_messages(peer_id, reader, writer)

        except Exception as e:
            logger.error(f"❌ Error en handshake: {e}")

    async def connect_to_peer(self, peer_info: PeerInfo) -> bool:
        """Conecta a un peer específico"""
        try:
            if peer_info.peer_id in self.active_connections:
                logger.debug(f"⚠️ Ya conectado a {peer_info.peer_id}")
                return True

            logger.info(f"🔗 Conectando a peer {peer_info.peer_id} ({peer_info.ip_address}:{peer_info.port})")

            # Establecer conexión TCP
            future = asyncio.open_connection(peer_info.ip_address, peer_info.port)
            reader, writer = await asyncio.wait_for(future, timeout=self.connection_timeout)

            # Enviar handshake
            handshake_msg: Dict[str, Any] = {
                "type": "handshake",
                "node_id": self.node_id,
                "timestamp": time.time()
            }

            # Adjuntar identidad pública propia para permitir canal seguro
            try:
                if self.crypto_engine and getattr(self.crypto_engine, 'identity', None):
                    pub = self.crypto_engine.identity.export_public_identity()
                    handshake_msg["public_identity"] = {
                        'node_id': base64.b64encode(pub['node_id']).decode(),
                        'signing_key': base64.b64encode(pub['signing_key']).decode(),
                        'encryption_key': base64.b64encode(pub['encryption_key']).decode(),
                        'created_at': base64.b64encode(pub['created_at']).decode(),
                    }
            except Exception as e:
                logger.warning(f"⚠️ No se pudo adjuntar identidad pública propia en handshake: {e}")

            writer.write(json.dumps(handshake_msg).encode() + b'\n')
            await writer.drain()

            # Leer respuesta
            response = await asyncio.wait_for(reader.readline(), timeout=10)
            text = response.decode(errors='ignore').strip()
            if not text:
                logger.warning(f"⚠️ Respuesta de handshake vacía desde {peer_info.ip_address}:{peer_info.port}")
                await writer.wait_closed()
                return False
            try:
                response_data = json.loads(text)
            except json.JSONDecodeError as e:
                logger.warning(f"⚠️ Respuesta de handshake no-JSON desde {peer_info.ip_address}:{peer_info.port}: {e}")
                await writer.wait_closed()
                return False

            if response_data.get('type') == 'handshake_response' and response_data.get('status') == 'accepted':
                # Conexión exitosa
                connection_info = {
                    "peer_id": peer_info.peer_id,
                    "reader": reader,
                    "writer": writer,
                    "connected_at": time.time(),
                    "last_activity": time.time(),
                    "bytes_sent": 0,
                    "bytes_received": 0
                }

                self.active_connections[peer_info.peer_id] = connection_info
                self.connection_stats["active_connections"] += 1
                self.connection_stats["total_connections"] += 1

                logger.info(f"✅ Conectado exitosamente a {peer_info.peer_id}")

                # Procesar identidad pública de respuesta y establecer canal seguro
                try:
                    if self.crypto_engine and response_data.get('public_identity'):
                        pub_b64 = response_data['public_identity']
                        peer_public = {
                            'node_id': base64.b64decode(pub_b64['node_id']),
                            'signing_key': base64.b64decode(pub_b64['signing_key']),
                            'encryption_key': base64.b64decode(pub_b64['encryption_key']),
                            'created_at': base64.b64decode(pub_b64['created_at']),
                        }
                        self.crypto_engine.add_peer_identity(peer_public)
                        self.crypto_engine.establish_secure_channel(peer_info.peer_id)
                except Exception as e:
                    logger.warning(f"⚠️ No se pudo procesar identidad pública de {peer_info.peer_id}: {e}")

                # Iniciar manejo de mensajes
                asyncio.create_task(self._handle_peer_messages(peer_info.peer_id, reader, writer))

                return True
            else:
                logger.warning(f"❌ Handshake rechazado por {peer_info.peer_id}")
                writer.close()
                await writer.wait_closed()
                return False

        except Exception as e:
            logger.error(f"❌ Error conectando a {peer_info.peer_id}: {e}")
            self.connection_stats["failed_connections"] += 1
            return False

    async def _handle_peer_messages(self, peer_id: str, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        """Maneja mensajes de un peer conectado"""
        try:
            while peer_id in self.active_connections:
                # Leer mensaje
                data = await asyncio.wait_for(reader.readline(), timeout=60)
                if not data:
                    break

                # Actualizar estadísticas
                self.active_connections[peer_id]["bytes_received"] += len(data)
                self.active_connections[peer_id]["last_activity"] = time.time()
                self.connection_stats["bytes_received"] += len(data)

                # Procesar mensaje
                try:
                    text = data.decode(errors='ignore').strip()
                    if not text:
                        logger.debug(f"⚠️ Mensaje vacío recibido de {peer_id}")
                        continue
                    try:
                        incoming = json.loads(text)
                    except json.JSONDecodeError:
                        logger.warning(f"⚠️ Mensaje JSON inválido de {peer_id}")
                        continue
                    # Si es un mensaje seguro, descifrar
                    if incoming.get('type') == 'secure' and self.crypto_engine and SecureMessage:
                        try:
                            blob_b64 = incoming.get('payload')
                            blob = base64.b64decode(blob_b64) if isinstance(blob_b64, str) else None
                            if blob:
                                secure_msg = SecureMessage.deserialize(blob)
                                plaintext = self.crypto_engine.decrypt_message(secure_msg)
                                if plaintext:
                                    try:
                                        inner = json.loads(plaintext.decode(errors='ignore'))
                                    except json.JSONDecodeError as e:
                                        logger.warning(f"⚠️ Payload seguro no-JSON de {peer_id}: {e}")
                                    else:
                                        await self._process_peer_message(peer_id, inner, is_secure=True)
                                else:
                                    logger.warning(f"⚠️ No se pudo descifrar mensaje seguro de {peer_id}")
                            else:
                                logger.warning(f"⚠️ Payload seguro inválido de {peer_id}")
                        except Exception as e:
                            logger.warning(f"⚠️ Error procesando mensaje seguro de {peer_id}: {e}")
                    else:
                        # Verificar firma en mensajes en claro si está presente
                        if self.crypto_engine and 'signature' in incoming:
                            try:
                                # Solo intentamos verificar si conocemos la identidad del peer
                                peer_identities = getattr(self.crypto_engine, 'peer_identities', {})
                                if peer_id and peer_id not in peer_identities:
                                    # Identidad aún no conocida (p.ej., antes de terminar handshake). Evitamos verificación para no generar errores.
                                    logger.debug(f"🔐 Identidad de {peer_id} desconocida; omitiendo verificación de firma")
                                else:
                                    signature_b64 = incoming.pop('signature')
                                    canonical = json.dumps(incoming, sort_keys=True).encode()
                                    signature = base64.b64decode(signature_b64) if isinstance(signature_b64, str) else b''
                                    if signature:
                                        valid = self.crypto_engine.verify_signature(canonical, signature, signer_id=peer_id)
                                        if not valid:
                                            logger.warning(f"⚠️ Firma inválida de {peer_id}")
                                            continue
                                    else:
                                        logger.warning(f"⚠️ Firma ausente/ inválida de {peer_id}")
                            except Exception as e:
                                logger.warning(f"⚠️ Error verificando firma de {peer_id}: {e}")
                        await self._process_peer_message(peer_id, incoming, is_secure=False)
                except Exception as e:
                    logger.error(f"❌ Error procesando mensaje de {peer_id}: {e}")

        except asyncio.TimeoutError:
            logger.warning(f"⏰ Timeout en conexión con {peer_id}")
        except Exception as e:
            logger.error(f"❌ Error manejando mensajes de {peer_id}: {e}")
        finally:
            await self._safe_disconnect_peer(peer_id)

    async def _process_peer_message(self, peer_id: str, message: Dict[str, Any], is_secure: bool = False):
        """Procesa mensaje de peer"""
        message_type = message.get('type')

        if message_type == 'heartbeat':
            await self._handle_heartbeat(peer_id, message)
        elif message_type == 'heartbeat_response':
            await self._handle_heartbeat_response(peer_id, message)
        elif message_type == 'data':
            await self._handle_data_message(peer_id, message)
        elif message_type == 'broadcast':
            await self._handle_broadcast_message(peer_id, message)
        elif message_type == 'consensus':
            # Los mensajes de consenso deben recibirse por canal seguro
            if not is_secure:
                logger.warning(f"⚠️ Mensaje CONSENSUS recibido sin canal seguro desde {peer_id}; descartando")
                return
            # Entregar a la capa de aplicación si hay callback registrado
            if self.app_message_callback:
                try:
                    await self.app_message_callback(peer_id, message)
                except Exception as e:
                    logger.warning(f"⚠️ Error en app_message_callback para CONSENSUS de {peer_id}: {e!r}")
            else:
                logger.debug(f"📨 CONSENSUS de {peer_id} recibido pero no hay callback registrado")
        else:
            logger.debug(f"📨 Mensaje {message_type} de {peer_id}")
            # Entregar mensajes personalizados a la aplicación si existe callback
            if self.app_message_callback:
                try:
                    await self.app_message_callback(peer_id, message)
                except Exception as e:
                    logger.warning(f"⚠️ Error en app_message_callback para mensaje {message_type} de {peer_id}: {e!r}")

    async def _handle_heartbeat(self, peer_id: str, message: Dict[str, Any]):
        """Maneja mensaje de heartbeat"""
        # Responder heartbeat
        response = {
            "type": "heartbeat_response",
            "node_id": self.node_id,
            # timestamp local cuando respondemos
            "timestamp": time.time(),
            # eco del timestamp original para calcular RTT
            "echo_timestamp": message.get("timestamp", None)
        }

        await self.send_message(peer_id, response)

    async def _handle_heartbeat_response(self, peer_id: str, message: Dict[str, Any]):
        """Procesa respuesta de heartbeat y actualiza métricas de latencia (RTT)"""
        try:
            echo_ts = message.get('echo_timestamp')
            now = time.time()
            # Si no viene echo_timestamp, intentar usar el último enviado a ese peer
            send_ts = echo_ts if isinstance(echo_ts, (int, float)) else self.pending_heartbeats.get(peer_id)
            if isinstance(send_ts, (int, float)):
                rtt = max(0.0, now - float(send_ts))
                self.latency_by_peer[peer_id] = rtt
                # Guardar también en la estructura de conexión
                conn = self.active_connections.get(peer_id)
                if conn is not None:
                    conn["latency"] = rtt
                    conn["last_activity"] = now
                # Limpiar pending para evitar crecimiento sin control
                self.pending_heartbeats.pop(peer_id, None)
                logger.debug(f"⏱️ RTT con {peer_id}: {rtt:.3f}s")
            else:
                logger.debug(f"⚠️ No se pudo calcular RTT para {peer_id}: falta timestamp de envío")
        except Exception as e:
            logger.debug(f"⚠️ Error procesando heartbeat_response de {peer_id}: {e!r}")

    async def _handle_data_message(self, peer_id: str, message: Dict[str, Any]):
        """Maneja mensaje de datos"""
        logger.debug(f"📊 Datos recibidos de {peer_id}: {len(str(message))} bytes")
        # Procesar datos según aplicación

    async def _handle_broadcast_message(self, peer_id: str, message: Dict[str, Any]):
        """Maneja mensaje de broadcast"""
        logger.debug(f"📢 Broadcast recibido de {peer_id}")

        # Reenviar a otros peers (excepto el remitente)
        for connected_peer_id in self.active_connections:
            if connected_peer_id != peer_id:
                await self.send_message(connected_peer_id, message)

    async def send_message(self, peer_id: str, message: Dict[str, Any]) -> bool:
        """Envía mensaje a un peer específico"""
        try:
            if peer_id not in self.active_connections:
                logger.warning(f"⚠️ Peer {peer_id} no conectado")
                return False

            connection = self.active_connections[peer_id]
            writer = connection["writer"]

            # Si es un heartbeat, registrar marca de tiempo para calcular RTT al recibir respuesta
            try:
                if message.get('type') == 'heartbeat':
                    ts = message.get('timestamp')
                    if isinstance(ts, (int, float)):
                        self.pending_heartbeats[peer_id] = float(ts)
            except Exception:
                pass

            # Si hay canal seguro, cifrar mensaje; de lo contrario, firmar mensaje en claro si es posible
            if self.crypto_engine and hasattr(self.crypto_engine, 'ratchet_states') and peer_id in getattr(self.crypto_engine, 'ratchet_states', {}):
                try:
                    plaintext = json.dumps(message, sort_keys=True).encode()
                    secure_msg = self.crypto_engine.encrypt_message(plaintext, peer_id)
                    if not secure_msg:
                        raise RuntimeError("Falló el cifrado del mensaje para peer_id: {peer_id}")
                    blob = secure_msg.serialize()
                    wrapped = {
                        "type": "secure",
                        "sender_id": self.node_id,
                        "recipient_id": peer_id,
                        "payload": base64.b64encode(blob).decode(),
                        "timestamp": time.time(),
                    }
                    message_data = json.dumps(wrapped).encode() + b'\n'
                except (CryptoError, RuntimeError, ValueError) as e:
                    # Fallback a mensaje en claro firmado
                    logger.warning(f"Error en cifrado seguro para {peer_id}: {e}. Usando fallback a mensaje firmado")
                    canonical = json.dumps(message, sort_keys=True).encode()
                    signature_b64 = ""
                    try:
                        signature = self.crypto_engine.sign_data(canonical)
                        signature_b64 = base64.b64encode(signature).decode()
                    except Exception:
                        signature_b64 = ""
                    message_to_send = {**message, "signature": signature_b64}
                    message_data = json.dumps(message_to_send).encode() + b'\n'
            else:
                canonical = json.dumps(message, sort_keys=True).encode()
                signature_b64 = ""
                if self.crypto_engine:
                    try:
                        signature = self.crypto_engine.sign_data(canonical)
                        signature_b64 = base64.b64encode(signature).decode()
                    except Exception:
                        signature_b64 = ""
                message_to_send = {**message, "signature": signature_b64}
                message_data = json.dumps(message_to_send).encode() + b'\n'

            writer.write(message_data)
            await writer.drain()

            # Actualizar estadísticas
            connection["bytes_sent"] += len(message_data)
            connection["last_activity"] = time.time()
            self.connection_stats["bytes_sent"] += len(message_data)

            return True

        except Exception as e:
            logger.error(f"❌ Error enviando mensaje a {peer_id}: {e}")
            await self._safe_disconnect_peer(peer_id)
            return False

    async def broadcast_message(self, message: Dict[str, Any], exclude_peers: List[str] = None) -> int:
        """Envía mensaje broadcast a todos los peers conectados"""
        exclude_peers = exclude_peers or []
        sent_count = 0

        for peer_id in list(self.active_connections.keys()):
            if peer_id not in exclude_peers:
                if await self.send_message(peer_id, message):
                    sent_count += 1

        logger.debug(f"📢 Broadcast enviado a {sent_count} peers")
        return sent_count

    async def _disconnect_peer(self, peer_id: str):
        """Desconecta un peer"""
        if peer_id in self.active_connections:
            connection = self.active_connections[peer_id]

            try:
                writer = connection["writer"]
                writer.close()
                await writer.wait_closed()
            except Exception as e:
                logger.debug(f"⚠️ Error cerrando conexión con {peer_id}: {e}")

            del self.active_connections[peer_id]
            self.connection_stats["active_connections"] -= 1

            logger.info(f"🔌 Peer {peer_id} desconectado")

    def get_connected_peers(self) -> List[str]:
        """Obtiene lista de peers conectados"""
        return list(self.active_connections.keys())

    def get_connection_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas de conexión"""
        # Incluir métricas de latencia por peer
        stats = self.connection_stats.copy()
        stats["latency_by_peer"] = dict(self.latency_by_peer)
        return stats

    async def _safe_disconnect_peer(self, peer_id: str):
        """Desconecta un peer de forma segura evitando condiciones de carrera"""
        connection = self.active_connections.get(peer_id)
        if not connection:
            logger.debug(f"⚠️ Peer {peer_id} ya no está en active_connections")
            return

        try:
            writer = connection["writer"]
            writer.close()
            await writer.wait_closed()
        except Exception as e:
            logger.debug(f"⚠️ Error cerrando conexión con {peer_id}: {e}")

        removed = self.active_connections.pop(peer_id, None)
        if removed is not None:
            self.connection_stats["active_connections"] = max(0, self.connection_stats["active_connections"] - 1)
            # Solo loguear desconexión cuando efectivamente se removió para evitar duplicados en condiciones de carrera
            logger.info(f"🔌 Peer {peer_id} desconectado (safe)")
        else:
            logger.debug(f"⚠️ Peer {peer_id} ya estaba desconectado (race handled)")


class NetworkTopologyManager:
    """Gestor de topología de red"""

    def __init__(self, node_id: str):
        self.node_id = node_id
        self.network_graph: Dict[str, Set[str]] = defaultdict(set)
        self.node_metrics: Dict[str, Dict[str, float]] = {}
        self.topology_cache = None
        self.last_analysis = 0
        self.analysis_interval = 300  # 5 minutos

    def update_peer_connections(self, peer_connections: Dict[str, List[str]]):
        """Actualiza información de conexiones de peers"""
        self.network_graph.clear()

        for peer_id, connections in peer_connections.items():
            for connected_peer in connections:
                self.network_graph[peer_id].add(connected_peer)
                self.network_graph[connected_peer].add(peer_id)

    async def analyze_topology(self) -> NetworkTopology:
        """Analiza la topología actual de la red"""
        try:
            current_time = time.time()

            # Usar caché si es reciente
            if (
                self.topology_cache
                and current_time - self.last_analysis < self.analysis_interval
            ):
                return self.topology_cache

            logger.info("📊 Analizando topología de red")

            # Para redes vacías o muy pequeñas, devolver análisis simplificado
            if len(self.network_graph) <= 1:
                topology = NetworkTopology(
                    total_nodes=len(self.network_graph),
                    connected_nodes=0,
                    network_diameter=0,
                    clustering_coefficient=0.0,
                    average_path_length=0.0,
                    node_degrees={node: 0 for node in self.network_graph},
                    critical_nodes=[]
                )
                self.topology_cache = topology
                self.last_analysis = current_time
                logger.info(f"📊 Topología analizada: {topology.total_nodes} nodos, diámetro {topology.network_diameter}")
                return topology

            # Calcular métricas básicas
            total_nodes = len(self.network_graph)
            connected_nodes = sum(1 for connections in self.network_graph.values() if connections)

            # Calcular grados de nodos
            node_degrees = {
                node_id: len(connections)
                for node_id, connections in self.network_graph.items()
            }

            # Para análisis rápido, usar métricas simplificadas si la red es pequeña
            if total_nodes < 10:
                network_diameter = await self._calculate_network_diameter()
                clustering_coefficient = 0.0  # Simplificado para redes pequeñas
                average_path_length = 0.0     # Simplificado para redes pequeñas
                critical_nodes = []           # Simplificado para redes pequeñas
            else:
                # Análisis completo solo para redes grandes
                network_diameter = await self._calculate_network_diameter()
                clustering_coefficient = await self._calculate_clustering_coefficient()
                average_path_length = await self._calculate_average_path_length()
                critical_nodes = await self._identify_critical_nodes()

            # Crear objeto de topología
            topology = NetworkTopology(
                total_nodes=total_nodes,
                connected_nodes=connected_nodes,
                network_diameter=network_diameter,
                clustering_coefficient=clustering_coefficient,
                average_path_length=average_path_length,
                node_degrees=node_degrees,
                critical_nodes=critical_nodes
            )

            # Actualizar caché
            self.topology_cache = topology
            self.last_analysis = current_time

            logger.info(f"📊 Topología analizada: {total_nodes} nodos, diámetro {network_diameter}")
            return topology

        except Exception as e:
            logger.error(f"❌ Error analizando topología: {e}")
            return NetworkTopology(0, 0, 0, 0.0, 0.0, {}, [])

    async def _calculate_network_diameter(self) -> int:
        """Calcula el diámetro de la red (camino más largo entre cualquier par de nodos)"""
        if not self.network_graph:
            return 0

        max_distance = 0

        for start_node in self.network_graph:
            distances = await self._bfs_distances(start_node)
            if distances:
                max_distance = max(max_distance, max(distances.values()))

        return max_distance

    async def _bfs_distances(self, start_node: str) -> Dict[str, int]:
        """Calcula distancias BFS desde un nodo"""
        distances = {start_node: 0}
        queue = deque([start_node])

        while queue:
            current = queue.popleft()
            current_distance = distances[current]

            for neighbor in self.network_graph.get(current, set()):
                if neighbor not in distances:
                    distances[neighbor] = current_distance + 1
                    queue.append(neighbor)

        return distances

    async def _calculate_clustering_coefficient(self) -> float:
        """Calcula el coeficiente de clustering promedio"""
        if not self.network_graph:
            return 0.0

        total_coefficient = 0.0
        node_count = 0

        for node in self.network_graph:
            neighbors = self.network_graph[node]
            if len(neighbors) < 2:
                continue

            # Contar conexiones entre vecinos
            neighbor_connections = 0
            neighbors_list = list(neighbors)

            for i, neighbor1 in enumerate(neighbors_list):
                for neighbor2 in neighbors_list[i + 1:]:
                    if neighbor2 in self.network_graph.get(neighbor1, set()):
                        neighbor_connections += 1

            # Calcular coeficiente local
            possible_connections = len(neighbors) * (len(neighbors) - 1) // 2
            if possible_connections > 0:
                local_coefficient = neighbor_connections / possible_connections
                total_coefficient += local_coefficient
                node_count += 1

        return total_coefficient / node_count if node_count > 0 else 0.0

    async def _calculate_average_path_length(self) -> float:
        """Calcula la longitud promedio de camino"""
        if not self.network_graph:
            return 0.0

        total_distance = 0
        path_count = 0

        nodes = list(self.network_graph.keys())

        for i, start_node in enumerate(nodes):
            distances = await self._bfs_distances(start_node)

            for j, end_node in enumerate(nodes):
                if i < j and end_node in distances:
                    total_distance += distances[end_node]
                    path_count += 1

        return total_distance / path_count if path_count > 0 else 0.0

    async def _identify_critical_nodes(self) -> List[str]:
        """Identifica nodos críticos para la conectividad"""
        critical_nodes = []

        if not self.network_graph:
            return critical_nodes

        # Nodos con alto grado (hubs)
        node_degrees = {
            node_id: len(connections)
            for node_id, connections in self.network_graph.items()
        }

        if node_degrees:
            avg_degree = sum(node_degrees.values()) / len(node_degrees)
            threshold = avg_degree * 2  # Nodos con grado > 2x promedio

            for node_id, degree in node_degrees.items():
                if degree > threshold:
                    critical_nodes.append(node_id)

        # Nodos puente (cuya eliminación aumenta componentes conectados)
        for node in self.network_graph:
            if await self._is_bridge_node(node):
                if node not in critical_nodes:
                    critical_nodes.append(node)

        return critical_nodes

    async def _is_bridge_node(self, node: str) -> bool:
        """Verifica si un nodo es un puente crítico"""
        # Contar componentes conectados sin el nodo
        temp_graph = {
            n: connections - {node}
            for n, connections in self.network_graph.items()
            if n != node
        }

        # Remover nodo del grafo temporal
        if node in temp_graph:
            del temp_graph[node]

        # Contar componentes conectados
        original_components = await self._count_connected_components(self.network_graph)
        modified_components = await self._count_connected_components(temp_graph)

        return modified_components > original_components

    async def _count_connected_components(self, graph: Dict[str, Set[str]]) -> int:
        """Cuenta componentes conectados en el grafo"""
        visited = set()
        components = 0

        for node in graph:
            if node not in visited:
                await self._dfs_visit(node, graph, visited)
                components += 1

        return components

    async def _dfs_visit(self, node: str, graph: Dict[str, Set[str]], visited: Set[str]):
        """Visita nodos usando DFS"""
        visited.add(node)

        for neighbor in graph.get(node, set()):
            if neighbor not in visited:
                await self._dfs_visit(neighbor, graph, visited)

    def get_optimal_routes(self, source: str, destination: str) -> List[List[str]]:
        """Encuentra rutas óptimas entre dos nodos"""
        if source not in self.network_graph or destination not in self.network_graph:
            return []

        # Implementar algoritmo de rutas múltiples (ej: k-shortest paths)
        routes = []

        # Por simplicidad, usar BFS para encontrar camino más corto
        queue = deque([(source, [source])])
        visited = {source}

        while queue:
            current, path = queue.popleft()

            if current == destination:
                routes.append(path)
                if len(routes) >= 3:  # Limitar a 3 rutas
                    break
                continue

            for neighbor in self.network_graph.get(current, set()):
                if neighbor not in visited or len(path) < 5:  # Permitir revisitas en caminos cortos
                    new_path = path + [neighbor]
                    queue.append((neighbor, new_path))
                    visited.add(neighbor)

        return routes


class P2PNetworkManager:
    """Gestor principal de la red P2P"""

    def __init__(self, node_id: str, node_type: NodeType = NodeType.FULL, port: int = 8080):
        self.node_id = node_id
        self.node_type = node_type
        self.port = port

        # Componentes principales
        self.discovery_service = PeerDiscoveryService(node_id, node_type, port)
        # Inicializar motor criptográfico si está disponible
        self.crypto_engine: Optional['CryptoEngine'] = None
        try:
            if initialize_crypto:
                # Alinear el node_id criptográfico con el node_id de red para evitar desajustes
                self.crypto_engine = initialize_crypto({"security_level": "HIGH", "node_id": self.node_id})
            elif create_crypto_engine:
                self.crypto_engine = create_crypto_engine()
                # Si se crea sin adapter, generar identidad usando el node_id de red
                if self.crypto_engine:
                    try:
                        self.crypto_engine.generate_node_identity(self.node_id)
                    except Exception as e:
                        logger.warning(f"⚠️ No se pudo generar identidad con node_id de red: {e}")
        except Exception as e:
            logger.warning(f"⚠️ No se pudo inicializar CryptoEngine: {e}")
            self.crypto_engine = None

        self.connection_manager = ConnectionManager(node_id, port, self.crypto_engine)
        self.topology_manager = NetworkTopologyManager(node_id)

        # Estado de la red
        self.network_active = False
        self.peer_list: Dict[str, PeerInfo] = {}

        # Configuración
        self.auto_connect_peers = True
        self.max_peer_connections = 20
        self.heartbeat_interval = 30
        # NAT traversal y DHT (placeholders)
        self.nat_config: Dict[str, Any] = {
            "stun_servers": ["stun.l.google.com:19302"],
            "turn_servers": [],  # Configurable según despliegue (usuario/credenciales)
        }
        self.dht_bootstrap_seeds: List[Dict[str, Any]] = []  # e.g., [{"peer_id": "seed1", "ip": "1.2.3.4", "port": 8080}]
        self.dht_active: bool = False

        # Registro de handlers de aplicación por tipo de mensaje
        self.message_handlers: Dict[str, List[Callable[[str, Dict[str, Any]], Awaitable[None]]]] = defaultdict(list)

        # Conectar el callback de mensajes de aplicación del ConnectionManager
        self.connection_manager.app_message_callback = self._on_app_message

    def register_handler(self, message_type: MessageType, handler: Callable[[str, Dict[str, Any]], Awaitable[None]]):
        """Registra un handler de aplicación para un tipo de mensaje específico"""
        self.message_handlers[message_type.value].append(handler)

    async def _on_app_message(self, peer_id: str, message: Dict[str, Any]):
        """Despacha mensajes entrantes a los handlers registrados"""
        mtype = message.get('type')
        handlers = self.message_handlers.get(mtype, [])
        if not handlers:
            logger.debug(f"🔔 Mensaje de aplicación '{mtype}' de {peer_id} sin handlers registrados")
            return
        for h in handlers:
            try:
                await h(peer_id, message)
            except Exception as e:
                logger.warning(f"⚠️ Error ejecutando handler para '{mtype}' de {peer_id}: {e!r}")

    async def start_network(self):
        """Inicia la red P2P"""
        try:
            logger.info(f"🚀 Iniciando red P2P para nodo {self.node_id}")

            self.network_active = True

            # Iniciar servicios en background sin bloquear
            asyncio.create_task(self.discovery_service.start_discovery())
            asyncio.create_task(self.connection_manager.start_server())
            asyncio.create_task(self._network_maintenance_loop())
            asyncio.create_task(self._heartbeat_loop())
            
            # Dar tiempo para que los servicios se inicialicen
            await asyncio.sleep(0.1)
            
            logger.info(f"✅ Red P2P iniciada para nodo {self.node_id}")

        except Exception as e:
            logger.error(f"❌ Error iniciando red P2P: {e}")

    async def initialize_nat_traversal(self):
        """Inicializa NAT traversal (placeholders para STUN/TURN)
        - En entornos reales, usar librerías STUN/TURN para obtener IP/PUERTO público y negociar relays
        - Aquí, registramos configuración y realizamos pruebas mínimas de conectividad"""
        try:
            logger.info("🌐 Inicializando NAT traversal (placeholder)")
            # Intento simple: resolver dirección pública vía socket (no garantiza IP pública real)
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                s.connect(("8.8.8.8", 80))
                local_addr = s.getsockname()[0]
                s.close()
                logger.info(f"🛰️ Dirección local observada para NAT: {local_addr}")
            except Exception as e:
                logger.debug(f"⚠️ No se pudo determinar dirección local para NAT: {e!r}")

            if self.nat_config.get("stun_servers"):
                logger.info(f"🔧 STUN servers configurados: {self.nat_config['stun_servers']}")
            if self.nat_config.get("turn_servers"):
                logger.info(f"🔧 TURN servers configurados: {self.nat_config['turn_servers']}")

            logger.info("✅ NAT traversal placeholder inicializado")
        except Exception as e:
            logger.warning(f"⚠️ Error inicializando NAT traversal: {e!r}")

    async def initialize_dht_overlay(self):
        """Inicializa overlay DHT (placeholders para Kademlia)
        - En despliegue real, iniciar un nodo DHT y conectarse a bootstrap seeds
        - Aquí, registramos seeds y marcamos DHT como activo"""
        try:
            logger.info("🗺️ Inicializando overlay DHT (placeholder)")
            if not self.dht_bootstrap_seeds:
                logger.info("ℹ️ No hay seeds DHT configurados; DHT permanecerá inactivo")
                self.dht_active = False
                return
            for seed in self.dht_bootstrap_seeds:
                logger.info(f"🌱 Seed DHT: {seed}")
            self.dht_active = True
            logger.info("✅ DHT placeholder inicializado y marcado activo")
        except Exception as e:
            logger.warning(f"⚠️ Error inicializando DHT overlay: {e!r}")

    async def _network_maintenance_loop(self):
        """Bucle de mantenimiento de red"""
        while self.network_active:
            try:
                # Actualizar lista de peers
                discovered_peers = self.discovery_service.get_discovered_peers()

                for peer_info in discovered_peers:
                    self.peer_list[peer_info.peer_id] = peer_info

                # Conectar a nuevos peers si es necesario
                if self.auto_connect_peers:
                    await self._auto_connect_peers()

                # Actualizar topología
                await self._update_network_topology()

                await asyncio.sleep(30)  # Mantenimiento cada 30 segundos

            except Exception as e:
                logger.error(f"❌ Error en mantenimiento de red: {e}")
                await asyncio.sleep(10)

    async def _auto_connect_peers(self):
        """Conecta automáticamente a peers descubiertos"""
        connected_peers = set(self.connection_manager.get_connected_peers())

        # Filtrar peers no conectados
        available_peers = [
            peer
            for peer in self.peer_list.values()
            if (
                peer.peer_id not in connected_peers
                and peer.connection_status != ConnectionStatus.FAILED
            )
        ]

        # Ordenar por reputación y latencia
        available_peers.sort(key=lambda p: (-p.reputation_score, p.latency))

        # Conectar hasta el límite
        connections_needed = self.max_peer_connections - len(connected_peers)

        for peer in available_peers[:connections_needed]:
            success = await self.connection_manager.connect_to_peer(peer)
            if success:
                peer.connection_status = ConnectionStatus.CONNECTED
            else:
                peer.connection_status = ConnectionStatus.FAILED

    async def _update_network_topology(self):
        """Actualiza información de topología de red"""
        try:
            # Recopilar información de conexiones
            peer_connections = {}

            for peer_id in self.connection_manager.get_connected_peers():
                # En implementación real, esto consultaría a cada peer por sus conexiones
                peer_connections[peer_id] = []

            # Actualizar topología
            self.topology_manager.update_peer_connections(peer_connections)

        except Exception as e:
            logger.error(f"❌ Error actualizando topología: {e}")

    async def _heartbeat_loop(self):
        """Bucle de heartbeat para mantener conexiones
        - Solo peers con canal seguro establecido reciben heartbeats para evitar firmas inválidas durante el arranque"""
        while self.network_active:
            try:
                heartbeat_msg = {
                    "type": "heartbeat",
                    "node_id": self.node_id,
                    "timestamp": time.time()
                }

                # Enviar heartbeat solo a peers con canal seguro establecido
                exclude_peers: List[str] = []
                try:
                    if getattr(self.connection_manager, "crypto_engine", None):
                        ratchets = getattr(self.connection_manager.crypto_engine, "ratchet_states", {})
                        for pid in self.connection_manager.get_connected_peers():
                            if pid not in ratchets:
                                exclude_peers.append(pid)
                except Exception as e:
                    logger.debug(f"⚠️ No se pudo determinar canales seguros para heartbeat: {e}")

                await self.connection_manager.broadcast_message(heartbeat_msg, exclude_peers=exclude_peers)

                await asyncio.sleep(self.heartbeat_interval)

            except Exception as e:
                logger.error(f"❌ Error en heartbeat: {e}")
                await asyncio.sleep(10)

    async def send_message(self, peer_id: str, message_type: MessageType, payload: Dict[str, Any]) -> bool:
        """Envía mensaje a un peer específico
        - Para MessageType.CONSENSUS se exige canal seguro (Double Ratchet activo)
        - Otros tipos de mensaje siguen la política de ConnectionManager (intenta cifrar si hay canal)"""
        # Para mensajes de consenso, requerimos canal seguro establecido para evitar envío en claro
        if message_type == MessageType.CONSENSUS:
            try:
                ce = self.connection_manager.crypto_engine
                if not ce or not hasattr(ce, "ratchet_states") or peer_id not in getattr(ce, "ratchet_states", {}):
                    logger.warning(f"⚠️ Canal seguro no establecido con {peer_id}; se omite envío CONSENSUS")
                    return False
            except Exception:
                logger.warning(f"⚠️ Verificación de canal seguro falló para {peer_id}; se omite envío CONSENSUS")
                return False
        message = {
            "type": message_type.value,
            "node_id": self.node_id,
            "payload": payload,
            "timestamp": time.time()
        }

        return await self.connection_manager.send_message(peer_id, message)

    async def broadcast_message(self, message_type: MessageType, payload: Dict[str, Any]) -> int:
        """Envía mensaje broadcast a todos los peers
        - Para MessageType.CONSENSUS se difunde únicamente a peers con canal seguro establecido"""
        # Para CONSENSUS, solo enviar a peers con canal seguro establecido
        if message_type == MessageType.CONSENSUS:
            ce = self.connection_manager.crypto_engine
            ratchets = getattr(ce, "ratchet_states", {}) if ce else {}
            count = 0
            for peer_id in self.connection_manager.get_connected_peers():
                if peer_id in ratchets:
                    ok = await self.send_message(peer_id, message_type, payload)
                    if ok:
                        count += 1
                else:
                    logger.debug(f"Omitiendo CONSENSUS broadcast a {peer_id} (sin canal seguro)")
            return count
        else:
            message = {
                "type": message_type.value,
                "node_id": self.node_id,
                "payload": payload,
                "timestamp": time.time()
            }
            return await self.connection_manager.broadcast_message(message)

    async def get_network_status(self) -> Dict[str, Any]:
        """Obtiene estado actual de la red con timeout interno"""
        try:
            # Obtener datos básicos sin operaciones bloqueantes
            discovered_peers_count = len(self.peer_list) if hasattr(self, 'peer_list') else 0
            connected_peers = self.connection_manager.get_connected_peers() if hasattr(self, 'connection_manager') else []
            connected_peers_count = len(connected_peers)
            
            # Intentar obtener topología con timeout muy corto
            topology = None
            try:
                topology = await asyncio.wait_for(
                    self.topology_manager.analyze_topology(), 
                    timeout=0.5
                )
            except asyncio.TimeoutError:
                logger.debug("Timeout en analyze_topology, usando datos básicos")
                topology = NetworkTopology(
                    total_nodes=discovered_peers_count,
                    connected_nodes=connected_peers_count,
                    network_diameter=0,
                    clustering_coefficient=0.0,
                    average_path_length=0.0,
                    node_degrees={},
                    critical_nodes=[]
                )
            except Exception as e:
                logger.debug(f"Error en analyze_topology: {e}")
                topology = NetworkTopology(
                    total_nodes=discovered_peers_count,
                    connected_nodes=connected_peers_count,
                    network_diameter=0,
                    clustering_coefficient=0.0,
                    average_path_length=0.0,
                    node_degrees={},
                    critical_nodes=[]
                )

            # Obtener estadísticas de conexión de forma segura
            connection_stats = {}
            try:
                connection_stats = self.connection_manager.get_connection_stats()
            except Exception as e:
                logger.debug(f"Error obteniendo connection_stats: {e}")
                connection_stats = {"error": "connection_stats_unavailable"}

            # Crear lista de peers de forma segura y limitada
            peer_list_data = []
            try:
                if hasattr(self, 'peer_list') and len(self.peer_list) <= 50:  # Limitar a 50 peers
                    peer_list_data = [asdict(peer) for peer in list(self.peer_list.values())[:50]]
                else:
                    peer_list_data = []  # Demasiados peers, omitir para evitar bloqueo
            except Exception as e:
                logger.debug(f"Error procesando peer_list: {e}")
                peer_list_data = []

            return {
                "node_id": getattr(self, 'node_id', 'unknown'),
                "node_type": getattr(self, 'node_type', NodeType.FULL).value,
                "network_active": getattr(self, 'network_active', False),
                "discovered_peers": discovered_peers_count,
                "connected_peers": connected_peers_count,
                "topology": asdict(topology),
                "connection_stats": connection_stats,
                "nat_config": getattr(self, 'nat_config', {}),
                "dht_active": getattr(self, 'dht_active', False),
                "dht_bootstrap_seeds": getattr(self, 'dht_bootstrap_seeds', []),
                "peer_list": peer_list_data
            }
        except Exception as e:
            logger.error(f"Error crítico en get_network_status: {e}")
            # Retornar estado mínimo en caso de error
            return {
                "node_id": getattr(self, 'node_id', 'unknown'),
                "node_type": "full",
                "network_active": False,
                "discovered_peers": 0,
                "connected_peers": 0,
                "topology": asdict(NetworkTopology(0, 0, 0, 0.0, 0.0, {}, [])),
                "connection_stats": {"error": "critical_error"},
                "nat_config": {},
                "dht_active": False,
                "dht_bootstrap_seeds": [],
                "peer_list": [],
                "error": str(e)
            }

    async def stop_network(self):
        """Detiene la red P2P"""
        logger.info("🛑 Deteniendo red P2P")

        self.network_active = False

        # Detener servicios
        await self.discovery_service.stop_discovery()

        # Desconectar todos los peers
        for peer_id in list(self.connection_manager.active_connections.keys()):
            await self.connection_manager._safe_disconnect_peer(peer_id)

        logger.info("✅ Red P2P detenida")


# Función principal para testing
async def main():
    """Función principal para pruebas"""
    # Crear red P2P
    network = P2PNetworkManager("test_node_1", NodeType.FULL, 8080)

    try:
        # Iniciar red en background
        asyncio.create_task(network.start_network())

        # Esperar un poco para que se inicialice
        await asyncio.sleep(5)

        # Mostrar estado de red
        status = await network.get_network_status()
        print("🌐 Estado de la red P2P:")
        print(json.dumps(status, indent=2, default=str))

        # Simular envío de mensaje
        await network.broadcast_message(MessageType.DATA, {"test": "mensaje de prueba"})

        # Mantener activo por un tiempo
        await asyncio.sleep(30)

    except KeyboardInterrupt:
        print("\n🛑 Deteniendo red...")
    finally:
        await network.stop_network()

if __name__ == "__main__":
    asyncio.run(main())


def start_network(config: dict):
    """Adapter de arranque a nivel de módulo.
    Crea y lanza P2PNetworkManager en segundo plano para compatibilidad con main.py.
    """
    try:
        node_id = config.get("node_id", "node_local")
        port = int(config.get("port", 8080))
        node_type_str = config.get("node_type", "full").upper()
        try:
            node_type = NodeType[node_type_str]
        except Exception:
            node_type = NodeType.FULL

        manager = P2PNetworkManager(node_id=node_id, node_type=node_type, port=port)
        # Ajustes opcionales
        manager.heartbeat_interval = int(config.get("heartbeat_interval_sec", 30))
        manager.max_peer_connections = int(config.get("max_peer_connections", 20))

        # Lanzar en el loop actual sin bloquear
        loop = asyncio.get_event_loop()
        loop.create_task(manager.start_network())
        logger.info(f"🔌 P2PNetworkManager iniciado (node_id={node_id}, port={port})")
        return manager
    except Exception as e:
        logger.error(f"❌ No se pudo iniciar la red P2P: {e}")
        return None
