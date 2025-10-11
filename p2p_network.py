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
from typing import Dict, List, Set, Any
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
            if HAS_ZEROCONF and HAS_NETIFACES:
                await self._setup_zeroconf()
            else:
                logger.warning("mDNS/Zeroconf no disponible; se usarán métodos alternativos.")

            # Iniciar tareas de descubrimiento
            tasks = []
            if HAS_ZEROCONF:
                tasks.append(asyncio.create_task(self._mdns_discovery()))
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
            if not (HAS_ZEROCONF and HAS_NETIFACES):
                logger.warning("Saltando configuración Zeroconf: dependencias no disponibles.")
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

            # Registrar servicio
            self.zeroconf = Zeroconf()
            self.zeroconf.register_service(self.service_info)

            logger.info(f"📡 Servicio mDNS registrado: {service_name}")

        except Exception as e:
            logger.error(f"❌ Error configurando Zeroconf: {e}")

    def _get_local_ip(self) -> str:
        """Obtiene la IP local del nodo"""
        try:
            if HAS_NETIFACES:
                # Intentar obtener IP de interfaces de red
                interfaces = netifaces.interfaces()

                for interface in interfaces:
                    addresses = netifaces.ifaddresses(interface)
                    if netifaces.AF_INET in addresses:
                        for addr_info in addresses[netifaces.AF_INET]:
                            ip = addr_info['addr']
                            if not ip.startswith('127.') and not ip.startswith('169.254.'):
                                return ip

            # Fallback: usar socket para conectar a servidor externo
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(("8.8.8.8", 80))
                return s.getsockname()[0]

        except Exception as e:
            logger.warning(f"⚠️ Error obteniendo IP local: {e}")
            return "127.0.0.1"

    async def _mdns_discovery(self):
        """Descubrimiento usando mDNS/Zeroconf"""
        class AegisServiceListener(ServiceListener):
            def __init__(self, discovery_service):
                self.discovery_service = discovery_service

            def add_service(self, zc: Zeroconf, type_: str, name: str):
                asyncio.create_task(self._handle_service_added(zc, type_, name))

            async def _handle_service_added(self, zc: Zeroconf, type_: str, name: str):
                try:
                    info = zc.get_service_info(type_, name)
                    if info and info.properties:
                        await self.discovery_service._process_discovered_service(info)
                except Exception as e:
                    logger.error(f"❌ Error procesando servicio descubierto: {e}")

        try:
            # Crear listener
            listener = AegisServiceListener(self)

            # Iniciar browser
            browser = ServiceBrowser(self.zeroconf, self.service_type, listener)

            # Mantener activo
            while self.discovery_active:
                await asyncio.sleep(10)

            browser.cancel()

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
            response_data = json.loads(response.decode().strip())

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

        if self.zeroconf and self.service_info:
            self.zeroconf.unregister_service(self.service_info)
            self.zeroconf.close()

        logger.info("🔍 Servicio de descubrimiento detenido")


class ConnectionManager:
    """Gestor de conexiones P2P"""

    def __init__(self, node_id: str, port: int):
        self.node_id = node_id
        self.port = port
        self.active_connections: Dict[str, Dict[str, Any]] = {}
        self.connection_pool: Dict[str, asyncio.Queue] = {}
        self.max_connections = 50
        self.connection_timeout = 30

        # Estadísticas de conexión
        self.connection_stats = {
            "total_connections": 0,
            "active_connections": 0,
            "failed_connections": 0,
            "bytes_sent": 0,
            "bytes_received": 0
        }

    async def start_server(self):
        """Inicia el servidor de conexiones"""
        try:
            server = await asyncio.start_server(
                self._handle_incoming_connection,
                '0.0.0.0',
                self.port
            )

            logger.info(f"🌐 Servidor P2P iniciado en puerto {self.port}")

            async with server:
                await server.serve_forever()

        except Exception as e:
            logger.error(f"❌ Error iniciando servidor: {e}")

    async def _handle_incoming_connection(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        """Maneja conexión entrante"""
        peer_address = writer.get_extra_info('peername')
        logger.debug(f"📥 Conexión entrante desde {peer_address}")

        try:
            # Leer mensaje inicial
            data = await asyncio.wait_for(reader.readline(), timeout=10)
            message = json.loads(data.decode().strip())

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
            handshake_response = {
                "type": "handshake_response",
                "node_id": self.node_id,
                "status": "accepted"
            }

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
            handshake_msg = {
                "type": "handshake",
                "node_id": self.node_id,
                "timestamp": time.time()
            }

            writer.write(json.dumps(handshake_msg).encode() + b'\n')
            await writer.drain()

            # Leer respuesta
            response = await asyncio.wait_for(reader.readline(), timeout=10)
            response_data = json.loads(response.decode().strip())

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
                    message = json.loads(data.decode().strip())
                    await self._process_peer_message(peer_id, message)
                except json.JSONDecodeError:
                    logger.warning(f"⚠️ Mensaje JSON inválido de {peer_id}")

        except asyncio.TimeoutError:
            logger.warning(f"⏰ Timeout en conexión con {peer_id}")
        except Exception as e:
            logger.error(f"❌ Error manejando mensajes de {peer_id}: {e}")
        finally:
            await self._disconnect_peer(peer_id)

    async def _process_peer_message(self, peer_id: str, message: Dict[str, Any]):
        """Procesa mensaje de peer"""
        message_type = message.get('type')

        if message_type == 'heartbeat':
            await self._handle_heartbeat(peer_id, message)
        elif message_type == 'data':
            await self._handle_data_message(peer_id, message)
        elif message_type == 'broadcast':
            await self._handle_broadcast_message(peer_id, message)
        else:
            logger.debug(f"📨 Mensaje {message_type} de {peer_id}")

    async def _handle_heartbeat(self, peer_id: str, message: Dict[str, Any]):
        """Maneja mensaje de heartbeat"""
        # Responder heartbeat
        response = {
            "type": "heartbeat_response",
            "node_id": self.node_id,
            "timestamp": time.time()
        }

        await self.send_message(peer_id, response)

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

            # Serializar y enviar mensaje
            message_data = json.dumps(message).encode() + b'\n'
            writer.write(message_data)
            await writer.drain()

            # Actualizar estadísticas
            connection["bytes_sent"] += len(message_data)
            connection["last_activity"] = time.time()
            self.connection_stats["bytes_sent"] += len(message_data)

            return True

        except Exception as e:
            logger.error(f"❌ Error enviando mensaje a {peer_id}: {e}")
            await self._disconnect_peer(peer_id)
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
        return self.connection_stats.copy()


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

            # Calcular métricas básicas
            total_nodes = len(self.network_graph)
            connected_nodes = sum(1 for connections in self.network_graph.values() if connections)

            # Calcular grados de nodos
            node_degrees = {
                node_id: len(connections)
                for node_id, connections in self.network_graph.items()
            }

            # Calcular diámetro de red
            network_diameter = await self._calculate_network_diameter()

            # Calcular coeficiente de clustering
            clustering_coefficient = await self._calculate_clustering_coefficient()

            # Calcular longitud promedio de camino
            average_path_length = await self._calculate_average_path_length()

            # Identificar nodos críticos
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
        self.connection_manager = ConnectionManager(node_id, port)
        self.topology_manager = NetworkTopologyManager(node_id)

        # Estado de la red
        self.network_active = False
        self.peer_list: Dict[str, PeerInfo] = {}

        # Configuración
        self.auto_connect_peers = True
        self.max_peer_connections = 20
        self.heartbeat_interval = 30

    async def start_network(self):
        """Inicia la red P2P"""
        try:
            logger.info(f"🚀 Iniciando red P2P para nodo {self.node_id}")

            self.network_active = True

            # Iniciar servicios
            tasks = [
                asyncio.create_task(self.discovery_service.start_discovery()),
                asyncio.create_task(self.connection_manager.start_server()),
                asyncio.create_task(self._network_maintenance_loop()),
                asyncio.create_task(self._heartbeat_loop())
            ]

            await asyncio.gather(*tasks)

        except Exception as e:
            logger.error(f"❌ Error iniciando red P2P: {e}")

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
        """Bucle de heartbeat para mantener conexiones"""
        while self.network_active:
            try:
                heartbeat_msg = {
                    "type": "heartbeat",
                    "node_id": self.node_id,
                    "timestamp": time.time()
                }

                # Enviar heartbeat a todos los peers conectados
                await self.connection_manager.broadcast_message(heartbeat_msg)

                await asyncio.sleep(self.heartbeat_interval)

            except Exception as e:
                logger.error(f"❌ Error en heartbeat: {e}")
                await asyncio.sleep(10)

    async def send_message(self, peer_id: str, message_type: MessageType, payload: Dict[str, Any]) -> bool:
        """Envía mensaje a un peer específico"""
        message = {
            "type": message_type.value,
            "node_id": self.node_id,
            "payload": payload,
            "timestamp": time.time()
        }

        return await self.connection_manager.send_message(peer_id, message)

    async def broadcast_message(self, message_type: MessageType, payload: Dict[str, Any]) -> int:
        """Envía mensaje broadcast a todos los peers"""
        message = {
            "type": message_type.value,
            "node_id": self.node_id,
            "payload": payload,
            "timestamp": time.time()
        }

        return await self.connection_manager.broadcast_message(message)

    async def get_network_status(self) -> Dict[str, Any]:
        """Obtiene estado actual de la red"""
        topology = await self.topology_manager.analyze_topology()
        connection_stats = self.connection_manager.get_connection_stats()

        return {
            "node_id": self.node_id,
            "node_type": self.node_type.value,
            "network_active": self.network_active,
            "discovered_peers": len(self.peer_list),
            "connected_peers": len(self.connection_manager.get_connected_peers()),
            "topology": asdict(topology),
            "connection_stats": connection_stats,
            "peer_list": [asdict(peer) for peer in self.peer_list.values()]
        }

    async def stop_network(self):
        """Detiene la red P2P"""
        logger.info("🛑 Deteniendo red P2P")

        self.network_active = False

        # Detener servicios
        await self.discovery_service.stop_discovery()

        # Desconectar todos los peers
        for peer_id in list(self.connection_manager.active_connections.keys()):
            await self.connection_manager._disconnect_peer(peer_id)

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
