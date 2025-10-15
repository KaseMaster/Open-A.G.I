#!/usr/bin/env python3
"""
TOR Integration Module para IA Distribuida
Implementación segura de comunicaciones anónimas P2P

AEGIS Security Framework - Uso Ético Únicamente
"""

import asyncio
import hashlib
import secrets
import struct
import time
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple, Callable, Any
import logging

# Dependencias de terceros (instalar con: pip install aiohttp cryptography stem)
import aiohttp
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ed25519, x25519
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
import stem
from stem.control import Controller
from stem import Signal

# Configuración de logging seguro
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('tor_integration.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class CircuitState(Enum):
    """Estados del circuito TOR"""
    BUILDING = "building"
    BUILT = "built"
    FAILED = "failed"
    CLOSED = "closed"

class SecurityLevel(Enum):
    """Niveles de seguridad para comunicaciones"""
    STANDARD = 1
    HIGH = 2
    PARANOID = 3

@dataclass
class TorNode:
    """Representación de un nodo en la red TOR"""
    fingerprint: str
    nickname: str
    address: str
    or_port: int
    dir_port: int
    flags: List[str]
    bandwidth: int
    country: Optional[str] = None
    
class TorCircuit:
    """Gestión de circuitos TOR con diversidad geográfica"""
    
    def __init__(self, circuit_id: str, path: List[TorNode], security_level: SecurityLevel):
        self.circuit_id = circuit_id
        self.path = path
        self.security_level = security_level
        self.state = CircuitState.BUILDING
        self.created_at = time.time()
        self.last_used = time.time()
        self.usage_count = 0
        self.max_usage = self._calculate_max_usage()
    
    def _calculate_max_usage(self) -> int:
        """Calcula el uso máximo basado en el nivel de seguridad"""
        base_usage = {
            SecurityLevel.STANDARD: 100,
            SecurityLevel.HIGH: 50,
            SecurityLevel.PARANOID: 10
        }
        return base_usage[self.security_level]
    
    def should_rotate(self) -> bool:
        """Determina si el circuito debe rotarse"""
        age_limit = {
            SecurityLevel.STANDARD: 3600,  # 1 hora
            SecurityLevel.HIGH: 1800,      # 30 minutos
            SecurityLevel.PARANOID: 600    # 10 minutos
        }
        
        current_time = time.time()
        age_exceeded = (current_time - self.created_at) > age_limit[self.security_level]
        usage_exceeded = self.usage_count >= self.max_usage
        
        return age_exceeded or usage_exceeded

class TorGateway:
    """Gateway principal para comunicaciones TOR"""
    
    def __init__(self, control_port: int = 9051, socks_port: int = 9050):
        self.control_port = control_port
        self.socks_port = socks_port
        self.controller: Optional[Controller] = None
        self.circuits: Dict[str, TorCircuit] = {}
        self.onion_services: Dict[str, str] = {}  # service_id -> private_key
        self.security_level = SecurityLevel.HIGH
        self.node_cache: List[TorNode] = []
        self.last_node_refresh = 0
        
        # Configuración de diversidad geográfica
        self.preferred_countries = ['US', 'DE', 'NL', 'SE', 'CH']
        self.excluded_countries = ['CN', 'RU', 'IR', 'KP']  # Países con censura conocida
        
    async def initialize(self) -> bool:
        """Inicializa la conexión con TOR"""
        try:
            self.controller = Controller.from_port(port=self.control_port)
            self.controller.authenticate()
            
            # Verificar que TOR esté funcionando
            if not self.controller.is_alive():
                logger.error("TOR controller no está activo")
                return False
            
            # Configurar eventos de circuito
            self.controller.add_event_listener(self._circuit_event_handler, 
                                             stem.control.EventType.CIRC)
            
            # Refrescar lista de nodos
            await self._refresh_node_list()
            
            logger.info("TOR Gateway inicializado correctamente")
            return True
            
        except Exception as e:
            logger.error(f"Error inicializando TOR Gateway: {e}")
            return False
    
    async def _refresh_node_list(self) -> None:
        """Actualiza la lista de nodos TOR disponibles"""
        try:
            current_time = time.time()
            if current_time - self.last_node_refresh < 3600:  # Cache por 1 hora
                return
            
            descriptors = self.controller.get_network_statuses()
            self.node_cache = []
            
            for desc in descriptors:
                # Filtrar nodos por flags de seguridad
                required_flags = ['Fast', 'Stable', 'Running']
                if not all(flag in desc.flags for flag in required_flags):
                    continue
                
                # Excluir países problemáticos
                if hasattr(desc, 'country') and desc.country in self.excluded_countries:
                    continue
                
                node = TorNode(
                    fingerprint=desc.fingerprint,
                    nickname=desc.nickname,
                    address=desc.address,
                    or_port=desc.or_port,
                    dir_port=desc.dir_port,
                    flags=desc.flags,
                    bandwidth=desc.bandwidth if hasattr(desc, 'bandwidth') else 0,
                    country=getattr(desc, 'country', None)
                )
                self.node_cache.append(node)
            
            self.last_node_refresh = current_time
            logger.info(f"Lista de nodos actualizada: {len(self.node_cache)} nodos disponibles")
            
        except Exception as e:
            logger.error(f"Error actualizando lista de nodos: {e}")
    
    def _select_diverse_path(self, path_length: int = 3) -> List[TorNode]:
        """Selecciona un path diverso geográficamente"""
        if len(self.node_cache) < path_length:
            raise ValueError("No hay suficientes nodos disponibles")
        
        # Separar nodos por país
        nodes_by_country = {}
        for node in self.node_cache:
            country = node.country or 'Unknown'
            if country not in nodes_by_country:
                nodes_by_country[country] = []
            nodes_by_country[country].append(node)
        
        # Seleccionar path con diversidad geográfica
        selected_path = []
        used_countries = set()
        
        # Priorizar países preferidos
        available_countries = list(nodes_by_country.keys())
        preferred_available = [c for c in self.preferred_countries if c in available_countries]
        
        for i in range(path_length):
            # Para el primer salto, usar países preferidos si están disponibles
            if i == 0 and preferred_available:
                candidates = []
                for country in preferred_available:
                    if country not in used_countries:
                        candidates.extend(nodes_by_country[country])
            else:
                # Para saltos posteriores, evitar países ya usados
                candidates = []
                for country, nodes in nodes_by_country.items():
                    if country not in used_countries:
                        candidates.extend(nodes)
            
            if not candidates:
                # Si no hay candidatos diversos, usar cualquier nodo disponible
                candidates = [n for n in self.node_cache if n not in selected_path]
            
            if not candidates:
                raise ValueError("No se pueden encontrar suficientes nodos diversos")
            
            # Selección ponderada por bandwidth
            weights = [node.bandwidth + 1 for node in candidates]  # +1 para evitar división por 0
            selected_node = secrets.choice(candidates)  # Usar secrets para aleatoriedad criptográfica
            
            selected_path.append(selected_node)
            used_countries.add(selected_node.country or 'Unknown')
        
        return selected_path
    
    async def create_circuit(self, purpose: str = "general") -> Optional[str]:
        """Crea un nuevo circuito TOR con diversidad geográfica"""
        try:
            await self._refresh_node_list()
            
            # Seleccionar path diverso
            path = self._select_diverse_path()
            path_fingerprints = [node.fingerprint for node in path]
            
            # Crear circuito
            circuit_id = self.controller.new_circuit(path_fingerprints, await_build=True)
            
            # Almacenar información del circuito
            circuit = TorCircuit(circuit_id, path, self.security_level)
            self.circuits[circuit_id] = circuit
            
            logger.info(f"Circuito {circuit_id} creado: {' -> '.join([n.country or 'Unknown' for n in path])}")
            return circuit_id
            
        except Exception as e:
            logger.error(f"Error creando circuito: {e}")
            return None
    
    def _circuit_event_handler(self, event) -> None:
        """Maneja eventos de circuito TOR"""
        circuit_id = event.id
        if circuit_id in self.circuits:
            circuit = self.circuits[circuit_id]
            
            if event.status == 'BUILT':
                circuit.state = CircuitState.BUILT
                logger.debug(f"Circuito {circuit_id} construido exitosamente")
            elif event.status == 'FAILED':
                circuit.state = CircuitState.FAILED
                logger.warning(f"Circuito {circuit_id} falló: {event.reason}")
            elif event.status == 'CLOSED':
                circuit.state = CircuitState.CLOSED
                logger.debug(f"Circuito {circuit_id} cerrado")
    
    async def create_onion_service(self, port: int, target_port: int = None) -> Optional[str]:
        """Crea un servicio onion para recibir conexiones"""
        try:
            if target_port is None:
                target_port = port
            
            # Generar clave privada para el servicio
            private_key = ed25519.Ed25519PrivateKey.generate()
            private_key_pem = private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            )
            
            # Crear servicio onion
            response = self.controller.create_ephemeral_hidden_service(
                {port: target_port},
                key_type='ED25519-V3',
                key_content=private_key_pem.decode()
            )
            
            service_id = response.service_id
            self.onion_services[service_id] = private_key_pem.decode()
            
            logger.info(f"Servicio onion creado: {service_id}.onion:{port}")
            return f"{service_id}.onion"
            
        except Exception as e:
            logger.error(f"Error creando servicio onion: {e}")
            return None
    
    async def send_message(self, target_onion: str, port: int, message: bytes, 
                          circuit_id: str = None) -> bool:
        """Envía un mensaje a través de TOR"""
        try:
            # Seleccionar o crear circuito
            if circuit_id and circuit_id in self.circuits:
                circuit = self.circuits[circuit_id]
                if circuit.should_rotate():
                    await self._rotate_circuit(circuit_id)
                    circuit_id = await self.create_circuit()
            else:
                circuit_id = await self.create_circuit()
            
            if not circuit_id:
                logger.error("No se pudo crear circuito para envío")
                return False
            
            # Configurar proxy SOCKS5 para usar el circuito específico
            proxy_url = f"socks5://127.0.0.1:{self.socks_port}"
            
            # Crear sesión HTTP con proxy TOR
            connector = aiohttp.TCPConnector()
            timeout = aiohttp.ClientTimeout(total=30)
            
            async with aiohttp.ClientSession(
                connector=connector,
                timeout=timeout
            ) as session:
                
                # Configurar headers para evitar fingerprinting
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; rv:91.0) Gecko/20100101 Firefox/91.0',
                    'Accept': 'application/octet-stream',
                    'Accept-Language': 'en-US,en;q=0.5',
                    'Accept-Encoding': 'gzip, deflate',
                    'Connection': 'keep-alive',
                    'Upgrade-Insecure-Requests': '1'
                }
                
                url = f"http://{target_onion}:{port}/message"
                
                async with session.post(
                    url,
                    data=message,
                    headers=headers,
                    proxy=proxy_url
                ) as response:
                    
                    if response.status == 200:
                        # Actualizar estadísticas del circuito
                        circuit = self.circuits[circuit_id]
                        circuit.usage_count += 1
                        circuit.last_used = time.time()
                        
                        logger.debug(f"Mensaje enviado exitosamente a {target_onion}")
                        return True
                    else:
                        logger.error(f"Error enviando mensaje: HTTP {response.status}")
                        return False
            
        except Exception as e:
            logger.error(f"Error enviando mensaje a {target_onion}: {e}")
            return False
    
    async def _rotate_circuit(self, circuit_id: str) -> None:
        """Rota un circuito por seguridad"""
        try:
            if circuit_id in self.circuits:
                self.controller.close_circuit(circuit_id)
                del self.circuits[circuit_id]
                logger.info(f"Circuito {circuit_id} rotado por seguridad")
        except Exception as e:
            logger.error(f"Error rotando circuito {circuit_id}: {e}")
    
    async def cleanup_old_circuits(self) -> None:
        """Limpia circuitos antiguos automáticamente"""
        circuits_to_remove = []
        
        for circuit_id, circuit in self.circuits.items():
            if circuit.should_rotate():
                circuits_to_remove.append(circuit_id)
        
        for circuit_id in circuits_to_remove:
            await self._rotate_circuit(circuit_id)
    
    async def get_network_status(self) -> Dict:
        """Obtiene el estado actual de la red TOR"""
        try:
            info = self.controller.get_info(['status/circuit-established', 
                                           'status/enough-dir-info',
                                           'status/bootstrap-phase'])
            
            return {
                'circuit_established': info.get('status/circuit-established') == '1',
                'enough_dir_info': info.get('status/enough-dir-info') == '1',
                'bootstrap_phase': info.get('status/bootstrap-phase', 'Unknown'),
                'active_circuits': len([c for c in self.circuits.values() 
                                      if c.state == CircuitState.BUILT]),
                'total_circuits': len(self.circuits),
                'available_nodes': len(self.node_cache)
            }
            
        except Exception as e:
            logger.error(f"Error obteniendo estado de red: {e}")
            return {}
    
    async def shutdown(self) -> None:
        """Cierra todas las conexiones y limpia recursos"""
        try:
            # Cerrar todos los circuitos
            for circuit_id in list(self.circuits.keys()):
                await self._rotate_circuit(circuit_id)
            
            # Remover servicios onion
            for service_id in list(self.onion_services.keys()):
                try:
                    self.controller.remove_ephemeral_hidden_service(service_id)
                except:
                    pass
            
            # Cerrar controlador
            if self.controller:
                self.controller.close()
            
            logger.info("TOR Gateway cerrado correctamente")
            
        except Exception as e:
            logger.error(f"Error cerrando TOR Gateway: {e}")

# Funciones de utilidad para integración
async def create_secure_tor_gateway(security_level: SecurityLevel = SecurityLevel.HIGH) -> TorGateway:
    """Crea y configura un gateway TOR seguro"""
    gateway = TorGateway()
    gateway.security_level = security_level
    
    if await gateway.initialize():
        return gateway
    else:
        raise RuntimeError("No se pudo inicializar TOR Gateway")

def generate_node_identity() -> Tuple[ed25519.Ed25519PrivateKey, str]:
    """Genera una identidad única para el nodo"""
    private_key = ed25519.Ed25519PrivateKey.generate()
    public_key = private_key.public_key()
    
    # Crear identificador único basado en la clave pública
    public_bytes = public_key.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw
    )
    
    node_id = hashlib.blake2b(public_bytes, digest_size=16).hexdigest()
    
    return private_key, node_id

# Ejemplo de uso
async def main():
    """Ejemplo de uso del TOR Gateway"""
    try:
        # Crear gateway
        gateway = await create_secure_tor_gateway(SecurityLevel.HIGH)
        
        # Crear servicio onion para recibir conexiones
        onion_address = await gateway.create_onion_service(8080)
        if onion_address:
            print(f"Servicio onion disponible en: {onion_address}:8080")
        
        # Obtener estado de la red
        status = await gateway.get_network_status()
        print(f"Estado de la red TOR: {status}")
        
        # Mantener el servicio activo
        print("Presiona Ctrl+C para salir...")
        try:
            while True:
                await asyncio.sleep(60)
                await gateway.cleanup_old_circuits()
        except KeyboardInterrupt:
            pass
        
        # Limpieza
        await gateway.shutdown()
        
    except Exception as e:
        logger.error(f"Error en ejemplo principal: {e}")

def start_gateway(config: Dict[str, Any]):
    """Adapter de arranque para inicializar el TOR Gateway en segundo plano.
    Compatible con main.py (no bloquea el event loop principal).
    """
    async def _run():
        try:
            level_str = str(config.get("security_level", "HIGH")).upper()
            level = SecurityLevel[level_str] if level_str in SecurityLevel.__members__ else SecurityLevel.HIGH
            gateway = await create_secure_tor_gateway(level)
            # Crear servicio onion opcional si se proporciona puerto
            svc_port = int(config.get("service_port", 0))
            if svc_port:
                onion = await gateway.create_onion_service(svc_port)
                if onion:
                    logger.info(f"🧅 Servicio onion disponible: {onion}:{svc_port}")
            logger.info("🔐 TOR Gateway iniciado")
            return gateway
        except Exception as e:
            logger.error(f"❌ No se pudo iniciar TOR Gateway: {e}")
            return None

    try:
        loop = asyncio.get_event_loop()
        return loop.create_task(_run())
    except Exception as e:
        logger.error(f"❌ Error programando inicio de TOR Gateway: {e}")
        return None

if __name__ == "__main__":
    asyncio.run(main())