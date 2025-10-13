#!/usr/bin/env python3
"""
Integración TOR-P2P para AEGIS Framework
Combina optimizaciones de TOR y P2P para máxima seguridad y rendimiento
"""

import asyncio
import logging
import json
import time
from typing import Dict, List, Optional, Set, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib
import secrets

from tor_optimization_config import TorOptimizationConfig, TorOptimizationLevel, create_optimized_config
from p2p_optimization_manager import P2POptimizationManager, P2POptimizationConfig, P2POptimizationLevel

logger = logging.getLogger(__name__)

class IntegrationMode(Enum):
    """Modos de integración TOR-P2P"""
    TOR_ONLY = "tor_only"           # Solo TOR para todas las conexiones
    P2P_ONLY = "p2p_only"           # Solo P2P directo
    HYBRID = "hybrid"               # TOR para sensible, P2P para rendimiento
    ADAPTIVE = "adaptive"           # Cambio dinámico según condiciones

class SecurityLevel(Enum):
    """Niveles de seguridad de la integración"""
    MINIMAL = "minimal"     # Rendimiento máximo, seguridad básica
    STANDARD = "standard"   # Balance seguridad/rendimiento
    HIGH = "high"          # Alta seguridad, rendimiento reducido
    PARANOID = "paranoid"  # Máxima seguridad, rendimiento mínimo

@dataclass
class ConnectionProfile:
    """Perfil de conexión para diferentes tipos de comunicación"""
    name: str
    use_tor: bool
    use_p2p: bool
    encryption_required: bool
    anonymity_required: bool
    latency_priority: bool
    bandwidth_priority: bool
    
    # Configuraciones específicas
    tor_circuit_timeout: float = 30.0
    p2p_connection_timeout: float = 10.0
    max_hops: int = 3
    geographic_diversity: bool = True

@dataclass
class IntegrationConfig:
    """Configuración de integración TOR-P2P"""
    
    # Configuración general
    integration_mode: IntegrationMode = IntegrationMode.HYBRID
    security_level: SecurityLevel = SecurityLevel.STANDARD
    
    # Configuraciones de componentes
    tor_config: TorOptimizationConfig = None
    p2p_config: P2POptimizationConfig = None
    
    # Perfiles de conexión
    connection_profiles: Dict[str, ConnectionProfile] = None
    
    # Configuración de failover
    enable_failover: bool = True
    failover_timeout: float = 15.0
    max_failover_attempts: int = 3
    
    # Configuración de monitoreo
    health_check_interval: float = 60.0
    performance_monitoring: bool = True
    
    # Configuración de balanceado
    load_balancing: bool = True
    tor_weight: float = 0.3  # Peso para conexiones TOR (0-1)
    p2p_weight: float = 0.7  # Peso para conexiones P2P (0-1)
    
    def __post_init__(self):
        if self.connection_profiles is None:
            self.connection_profiles = self._create_default_profiles()
    
    def _create_default_profiles(self) -> Dict[str, ConnectionProfile]:
        """Crea perfiles de conexión por defecto"""
        return {
            'anonymous': ConnectionProfile(
                name='anonymous',
                use_tor=True,
                use_p2p=False,
                encryption_required=True,
                anonymity_required=True,
                latency_priority=False,
                bandwidth_priority=False,
                max_hops=3,
                geographic_diversity=True
            ),
            'fast': ConnectionProfile(
                name='fast',
                use_tor=False,
                use_p2p=True,
                encryption_required=True,
                anonymity_required=False,
                latency_priority=True,
                bandwidth_priority=True,
                p2p_connection_timeout=5.0
            ),
            'secure': ConnectionProfile(
                name='secure',
                use_tor=True,
                use_p2p=True,
                encryption_required=True,
                anonymity_required=True,
                latency_priority=False,
                bandwidth_priority=False,
                tor_circuit_timeout=45.0,
                max_hops=4
            ),
            'balanced': ConnectionProfile(
                name='balanced',
                use_tor=True,
                use_p2p=True,
                encryption_required=True,
                anonymity_required=False,
                latency_priority=True,
                bandwidth_priority=True,
                tor_circuit_timeout=20.0,
                p2p_connection_timeout=8.0
            )
        }

class TorP2PIntegrationManager:
    """Gestor de integración TOR-P2P"""
    
    def __init__(self, config: IntegrationConfig = None):
        self.config = config or IntegrationConfig()
        
        # Gestores de componentes
        self.tor_manager = None  # Se inicializará con tor_integration.py
        self.p2p_manager = P2POptimizationManager(self.config.p2p_config)
        
        # Estado de la integración
        self.active_connections: Dict[str, Dict] = {}
        self.connection_stats: Dict[str, Dict] = {}
        self.failover_history: List[Dict] = []
        
        # Métricas de rendimiento
        self.performance_metrics = {
            'tor_latency': [],
            'p2p_latency': [],
            'tor_bandwidth': [],
            'p2p_bandwidth': [],
            'failover_count': 0,
            'connection_success_rate': 0.0
        }
        
        # Tareas de monitoreo
        self._monitoring_tasks: List[asyncio.Task] = []
        self._running = False
    
    async def start_integration(self):
        """Inicia el sistema de integración"""
        if self._running:
            return
        
        self._running = True
        logger.info("Iniciando integración TOR-P2P")
        
        try:
            # Inicializar gestores
            if self.config.integration_mode in [IntegrationMode.TOR_ONLY, IntegrationMode.HYBRID, IntegrationMode.ADAPTIVE]:
                await self._initialize_tor_manager()
            
            if self.config.integration_mode in [IntegrationMode.P2P_ONLY, IntegrationMode.HYBRID, IntegrationMode.ADAPTIVE]:
                await self.p2p_manager.start_optimization()
            
            # Iniciar tareas de monitoreo
            self._monitoring_tasks = [
                asyncio.create_task(self._health_monitor()),
                asyncio.create_task(self._performance_monitor()),
                asyncio.create_task(self._adaptive_optimizer())
            ]
            
            logger.info("Integración TOR-P2P iniciada correctamente")
            
        except Exception as e:
            logger.error(f"Error iniciando integración: {e}")
            await self.stop_integration()
            raise
    
    async def stop_integration(self):
        """Detiene el sistema de integración"""
        if not self._running:
            return
        
        self._running = False
        logger.info("Deteniendo integración TOR-P2P")
        
        # Cancelar tareas de monitoreo
        for task in self._monitoring_tasks:
            task.cancel()
        
        await asyncio.gather(*self._monitoring_tasks, return_exceptions=True)
        self._monitoring_tasks.clear()
        
        # Detener gestores
        if self.p2p_manager:
            await self.p2p_manager.stop_optimization()
        
        # Cerrar conexiones activas
        await self._close_all_connections()
        
        logger.info("Integración TOR-P2P detenida")
    
    async def create_connection(self, 
                             target: str, 
                             profile_name: str = 'balanced',
                             metadata: Dict = None) -> Optional[str]:
        """Crea una conexión usando el perfil especificado"""
        
        if profile_name not in self.config.connection_profiles:
            logger.error(f"Perfil de conexión desconocido: {profile_name}")
            return None
        
        profile = self.config.connection_profiles[profile_name]
        connection_id = self._generate_connection_id(target, profile_name)
        
        logger.info(f"Creando conexión {connection_id} con perfil {profile_name}")
        
        try:
            # Determinar método de conexión
            connection_method = await self._select_connection_method(profile, target)
            
            # Crear conexión
            connection_info = await self._establish_connection(
                connection_id, target, profile, connection_method, metadata
            )
            
            if connection_info:
                self.active_connections[connection_id] = connection_info
                logger.info(f"Conexión {connection_id} establecida exitosamente")
                return connection_id
            else:
                logger.error(f"Falló la creación de conexión {connection_id}")
                return None
                
        except Exception as e:
            logger.error(f"Error creando conexión {connection_id}: {e}")
            return None
    
    async def send_message(self, 
                          connection_id: str, 
                          message: Union[str, bytes],
                          priority: str = 'normal') -> bool:
        """Envía un mensaje a través de una conexión"""
        
        if connection_id not in self.active_connections:
            logger.error(f"Conexión {connection_id} no encontrada")
            return False
        
        connection_info = self.active_connections[connection_id]
        
        try:
            # Preparar mensaje
            prepared_message = await self._prepare_message(message, connection_info)
            
            # Enviar según el método de conexión
            success = await self._send_via_connection(connection_info, prepared_message)
            
            # Actualizar estadísticas
            self._update_connection_stats(connection_id, 'message_sent', success)
            
            return success
            
        except Exception as e:
            logger.error(f"Error enviando mensaje por {connection_id}: {e}")
            self._update_connection_stats(connection_id, 'message_error', False)
            return False
    
    async def close_connection(self, connection_id: str):
        """Cierra una conexión específica"""
        if connection_id in self.active_connections:
            connection_info = self.active_connections[connection_id]
            
            try:
                await self._close_connection(connection_info)
                del self.active_connections[connection_id]
                logger.info(f"Conexión {connection_id} cerrada")
                
            except Exception as e:
                logger.error(f"Error cerrando conexión {connection_id}: {e}")
    
    async def _initialize_tor_manager(self):
        """Inicializa el gestor TOR"""
        try:
            # Aquí se integraría con tor_integration.py
            # Por ahora, simulamos la inicialización
            logger.info("Inicializando gestor TOR")
            
            # Crear configuración TOR optimizada
            if not self.config.tor_config:
                tor_level = self._map_security_to_tor_level(self.config.security_level)
                self.config.tor_config = create_optimized_config(tor_level)
            
            # TODO: Integrar con TorGateway de tor_integration.py
            logger.info("Gestor TOR inicializado")
            
        except Exception as e:
            logger.error(f"Error inicializando TOR: {e}")
            raise
    
    def _map_security_to_tor_level(self, security_level: SecurityLevel) -> TorOptimizationLevel:
        """Mapea nivel de seguridad a nivel de optimización TOR"""
        mapping = {
            SecurityLevel.MINIMAL: TorOptimizationLevel.PERFORMANCE,
            SecurityLevel.STANDARD: TorOptimizationLevel.BALANCED,
            SecurityLevel.HIGH: TorOptimizationLevel.SECURITY,
            SecurityLevel.PARANOID: TorOptimizationLevel.SECURITY
        }
        return mapping.get(security_level, TorOptimizationLevel.BALANCED)
    
    def _map_security_to_p2p_level(self, security_level: SecurityLevel) -> P2POptimizationLevel:
        """Mapea nivel de seguridad a nivel de optimización P2P"""
        mapping = {
            SecurityLevel.MINIMAL: P2POptimizationLevel.AGGRESSIVE,
            SecurityLevel.STANDARD: P2POptimizationLevel.BALANCED,
            SecurityLevel.HIGH: P2POptimizationLevel.CONSERVATIVE,
            SecurityLevel.PARANOID: P2POptimizationLevel.CONSERVATIVE
        }
        return mapping.get(security_level, P2POptimizationLevel.BALANCED)
    
    async def _select_connection_method(self, profile: ConnectionProfile, target: str) -> str:
        """Selecciona el método de conexión óptimo"""
        
        if self.config.integration_mode == IntegrationMode.TOR_ONLY:
            return 'tor'
        elif self.config.integration_mode == IntegrationMode.P2P_ONLY:
            return 'p2p'
        elif self.config.integration_mode == IntegrationMode.HYBRID:
            # Decidir basado en el perfil
            if profile.anonymity_required:
                return 'tor'
            elif profile.latency_priority:
                return 'p2p'
            else:
                return 'balanced'  # Usar ambos
        else:  # ADAPTIVE
            return await self._adaptive_method_selection(profile, target)
    
    async def _adaptive_method_selection(self, profile: ConnectionProfile, target: str) -> str:
        """Selección adaptiva del método de conexión"""
        
        # Factores para la decisión
        factors = {
            'tor_available': True,  # TODO: Verificar disponibilidad real
            'p2p_available': True,  # TODO: Verificar disponibilidad real
            'tor_latency': self._get_average_tor_latency(),
            'p2p_latency': self._get_average_p2p_latency(),
            'network_load': self._get_network_load(),
            'security_requirement': profile.anonymity_required
        }
        
        # Lógica de decisión adaptiva
        if factors['security_requirement']:
            return 'tor'
        
        if factors['tor_latency'] > 500 and factors['p2p_latency'] < 100:
            return 'p2p'
        
        if factors['network_load'] > 0.8:
            return 'tor'  # TOR puede ser más estable bajo alta carga
        
        return 'balanced'
    
    def _get_average_tor_latency(self) -> float:
        """Obtiene la latencia promedio de TOR"""
        if self.performance_metrics['tor_latency']:
            return sum(self.performance_metrics['tor_latency'][-10:]) / len(self.performance_metrics['tor_latency'][-10:])
        return 200.0  # Valor por defecto
    
    def _get_average_p2p_latency(self) -> float:
        """Obtiene la latencia promedio de P2P"""
        if self.performance_metrics['p2p_latency']:
            return sum(self.performance_metrics['p2p_latency'][-10:]) / len(self.performance_metrics['p2p_latency'][-10:])
        return 50.0  # Valor por defecto
    
    def _get_network_load(self) -> float:
        """Obtiene la carga actual de la red (0-1)"""
        # TODO: Implementar cálculo real de carga
        return 0.3  # Valor simulado
    
    async def _establish_connection(self, 
                                  connection_id: str,
                                  target: str,
                                  profile: ConnectionProfile,
                                  method: str,
                                  metadata: Dict = None) -> Optional[Dict]:
        """Establece la conexión usando el método especificado"""
        
        connection_info = {
            'id': connection_id,
            'target': target,
            'profile': profile.name,
            'method': method,
            'created_at': time.time(),
            'metadata': metadata or {},
            'stats': {
                'messages_sent': 0,
                'messages_received': 0,
                'bytes_sent': 0,
                'bytes_received': 0,
                'errors': 0
            }
        }
        
        try:
            if method == 'tor':
                connection_info.update(await self._establish_tor_connection(target, profile))
            elif method == 'p2p':
                connection_info.update(await self._establish_p2p_connection(target, profile))
            elif method == 'balanced':
                connection_info.update(await self._establish_balanced_connection(target, profile))
            
            return connection_info
            
        except Exception as e:
            logger.error(f"Error estableciendo conexión {method}: {e}")
            
            # Intentar failover si está habilitado
            if self.config.enable_failover and method != 'failover':
                logger.info(f"Intentando failover para conexión {connection_id}")
                return await self._attempt_failover(connection_id, target, profile, method)
            
            return None
    
    async def _establish_tor_connection(self, target: str, profile: ConnectionProfile) -> Dict:
        """Establece conexión TOR"""
        # TODO: Integrar con TorGateway
        logger.info(f"Estableciendo conexión TOR a {target}")
        
        # Simulación por ahora
        await asyncio.sleep(profile.tor_circuit_timeout / 10)  # Simular tiempo de establecimiento
        
        return {
            'tor_circuit_id': f"tor_circuit_{secrets.token_hex(8)}",
            'tor_nodes': ['node1', 'node2', 'node3'],  # Simulado
            'established_at': time.time()
        }
    
    async def _establish_p2p_connection(self, target: str, profile: ConnectionProfile) -> Dict:
        """Establece conexión P2P"""
        logger.info(f"Estableciendo conexión P2P a {target}")
        
        # Simulación por ahora
        await asyncio.sleep(profile.p2p_connection_timeout / 10)
        
        return {
            'p2p_peer_id': f"p2p_peer_{secrets.token_hex(8)}",
            'direct_connection': True,
            'established_at': time.time()
        }
    
    async def _establish_balanced_connection(self, target: str, profile: ConnectionProfile) -> Dict:
        """Establece conexión balanceada (TOR + P2P)"""
        logger.info(f"Estableciendo conexión balanceada a {target}")
        
        # Intentar ambos métodos en paralelo
        tor_task = asyncio.create_task(self._establish_tor_connection(target, profile))
        p2p_task = asyncio.create_task(self._establish_p2p_connection(target, profile))
        
        try:
            tor_info, p2p_info = await asyncio.gather(tor_task, p2p_task, return_exceptions=True)
            
            result = {'balanced_mode': True}
            
            if not isinstance(tor_info, Exception):
                result.update(tor_info)
                result['tor_available'] = True
            else:
                result['tor_available'] = False
                logger.warning(f"TOR no disponible: {tor_info}")
            
            if not isinstance(p2p_info, Exception):
                result.update(p2p_info)
                result['p2p_available'] = True
            else:
                result['p2p_available'] = False
                logger.warning(f"P2P no disponible: {p2p_info}")
            
            if not result.get('tor_available') and not result.get('p2p_available'):
                raise Exception("Ni TOR ni P2P están disponibles")
            
            return result
            
        except Exception as e:
            logger.error(f"Error en conexión balanceada: {e}")
            raise
    
    async def _attempt_failover(self, 
                              connection_id: str,
                              target: str,
                              profile: ConnectionProfile,
                              failed_method: str) -> Optional[Dict]:
        """Intenta failover a método alternativo"""
        
        failover_methods = {
            'tor': 'p2p',
            'p2p': 'tor',
            'balanced': 'tor' if profile.anonymity_required else 'p2p'
        }
        
        failover_method = failover_methods.get(failed_method)
        if not failover_method:
            return None
        
        logger.info(f"Failover de {failed_method} a {failover_method} para {connection_id}")
        
        # Registrar failover
        self.failover_history.append({
            'connection_id': connection_id,
            'from_method': failed_method,
            'to_method': failover_method,
            'timestamp': time.time(),
            'target': target
        })
        
        self.performance_metrics['failover_count'] += 1
        
        try:
            return await self._establish_connection(
                connection_id, target, profile, failover_method + '_failover'
            )
        except Exception as e:
            logger.error(f"Failover también falló: {e}")
            return None
    
    def _generate_connection_id(self, target: str, profile: str) -> str:
        """Genera un ID único para la conexión"""
        data = f"{target}:{profile}:{time.time()}:{secrets.token_hex(4)}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]
    
    async def _prepare_message(self, message: Union[str, bytes], connection_info: Dict) -> bytes:
        """Prepara un mensaje para envío"""
        if isinstance(message, str):
            message = message.encode('utf-8')
        
        # TODO: Añadir encriptación si es requerida
        # TODO: Añadir headers de protocolo
        
        return message
    
    async def _send_via_connection(self, connection_info: Dict, message: bytes) -> bool:
        """Envía mensaje a través de la conexión"""
        method = connection_info['method']
        
        try:
            if 'tor' in method and connection_info.get('tor_available', True):
                return await self._send_via_tor(connection_info, message)
            elif 'p2p' in method and connection_info.get('p2p_available', True):
                return await self._send_via_p2p(connection_info, message)
            elif method == 'balanced':
                # Intentar P2P primero por velocidad, luego TOR
                if connection_info.get('p2p_available'):
                    success = await self._send_via_p2p(connection_info, message)
                    if success:
                        return True
                
                if connection_info.get('tor_available'):
                    return await self._send_via_tor(connection_info, message)
            
            return False
            
        except Exception as e:
            logger.error(f"Error enviando mensaje: {e}")
            return False
    
    async def _send_via_tor(self, connection_info: Dict, message: bytes) -> bool:
        """Envía mensaje vía TOR"""
        # TODO: Integrar con TorGateway
        logger.debug(f"Enviando mensaje vía TOR: {len(message)} bytes")
        await asyncio.sleep(0.1)  # Simular latencia TOR
        return True
    
    async def _send_via_p2p(self, connection_info: Dict, message: bytes) -> bool:
        """Envía mensaje vía P2P"""
        # TODO: Integrar con P2P network
        logger.debug(f"Enviando mensaje vía P2P: {len(message)} bytes")
        await asyncio.sleep(0.02)  # Simular latencia P2P
        return True
    
    def _update_connection_stats(self, connection_id: str, event: str, success: bool):
        """Actualiza estadísticas de conexión"""
        if connection_id in self.active_connections:
            stats = self.active_connections[connection_id]['stats']
            
            if event == 'message_sent':
                stats['messages_sent'] += 1
                if not success:
                    stats['errors'] += 1
            elif event == 'message_received':
                stats['messages_received'] += 1
            elif event == 'message_error':
                stats['errors'] += 1
    
    async def _close_connection(self, connection_info: Dict):
        """Cierra una conexión específica"""
        method = connection_info['method']
        
        if 'tor' in method:
            # TODO: Cerrar circuito TOR
            pass
        
        if 'p2p' in method:
            # TODO: Cerrar conexión P2P
            pass
    
    async def _close_all_connections(self):
        """Cierra todas las conexiones activas"""
        for connection_id in list(self.active_connections.keys()):
            await self.close_connection(connection_id)
    
    async def _health_monitor(self):
        """Monitor de salud del sistema"""
        while self._running:
            try:
                # Verificar salud de conexiones
                current_time = time.time()
                stale_connections = []
                
                for conn_id, conn_info in self.active_connections.items():
                    age = current_time - conn_info['created_at']
                    if age > 3600:  # 1 hora
                        stale_connections.append(conn_id)
                
                # Cerrar conexiones obsoletas
                for conn_id in stale_connections:
                    logger.info(f"Cerrando conexión obsoleta: {conn_id}")
                    await self.close_connection(conn_id)
                
                await asyncio.sleep(self.config.health_check_interval)
                
            except Exception as e:
                logger.error(f"Error en monitor de salud: {e}")
                await asyncio.sleep(30)
    
    async def _performance_monitor(self):
        """Monitor de rendimiento"""
        while self._running:
            try:
                if self.config.performance_monitoring:
                    # Recopilar métricas de rendimiento
                    await self._collect_performance_metrics()
                
                await asyncio.sleep(60)  # Cada minuto
                
            except Exception as e:
                logger.error(f"Error en monitor de rendimiento: {e}")
                await asyncio.sleep(30)
    
    async def _adaptive_optimizer(self):
        """Optimizador adaptivo"""
        while self._running:
            try:
                if self.config.integration_mode == IntegrationMode.ADAPTIVE:
                    await self._optimize_adaptive_behavior()
                
                await asyncio.sleep(300)  # Cada 5 minutos
                
            except Exception as e:
                logger.error(f"Error en optimizador adaptivo: {e}")
                await asyncio.sleep(60)
    
    async def _collect_performance_metrics(self):
        """Recopila métricas de rendimiento"""
        # TODO: Implementar recopilación real de métricas
        pass
    
    async def _optimize_adaptive_behavior(self):
        """Optimiza el comportamiento adaptivo"""
        # TODO: Implementar lógica de optimización adaptiva
        pass
    
    def get_integration_status(self) -> Dict:
        """Obtiene el estado de la integración"""
        return {
            'running': self._running,
            'integration_mode': self.config.integration_mode.value,
            'security_level': self.config.security_level.value,
            'active_connections': len(self.active_connections),
            'performance_metrics': self.performance_metrics.copy(),
            'failover_count': len(self.failover_history),
            'connection_profiles': list(self.config.connection_profiles.keys())
        }

def create_integration_config(
    mode: IntegrationMode = IntegrationMode.HYBRID,
    security: SecurityLevel = SecurityLevel.STANDARD
) -> IntegrationConfig:
    """Crea una configuración de integración optimizada"""
    
    config = IntegrationConfig(
        integration_mode=mode,
        security_level=security
    )
    
    # Configurar TOR según el nivel de seguridad
    tor_level_map = {
        SecurityLevel.MINIMAL: TorOptimizationLevel.PERFORMANCE,
        SecurityLevel.STANDARD: TorOptimizationLevel.BALANCED,
        SecurityLevel.HIGH: TorOptimizationLevel.SECURITY,
        SecurityLevel.PARANOID: TorOptimizationLevel.SECURITY
    }
    
    config.tor_config = create_optimized_config(tor_level_map[security])
    
    # Configurar P2P según el nivel de seguridad
    p2p_level_map = {
        SecurityLevel.MINIMAL: P2POptimizationLevel.AGGRESSIVE,
        SecurityLevel.STANDARD: P2POptimizationLevel.BALANCED,
        SecurityLevel.HIGH: P2POptimizationLevel.CONSERVATIVE,
        SecurityLevel.PARANOID: P2POptimizationLevel.CONSERVATIVE
    }
    
    from p2p_optimization_manager import create_optimized_p2p_config
    config.p2p_config = create_optimized_p2p_config(p2p_level_map[security])
    
    return config

if __name__ == "__main__":
    # Ejemplo de uso
    async def main():
        config = create_integration_config(
            IntegrationMode.HYBRID,
            SecurityLevel.STANDARD
        )
        
        manager = TorP2PIntegrationManager(config)
        
        try:
            await manager.start_integration()
            
            # Crear conexión de prueba
            conn_id = await manager.create_connection(
                "test_target",
                "balanced",
                {"test": True}
            )
            
            if conn_id:
                # Enviar mensaje de prueba
                success = await manager.send_message(conn_id, "Mensaje de prueba")
                print(f"Mensaje enviado: {success}")
                
                # Cerrar conexión
                await manager.close_connection(conn_id)
            
            # Mostrar estado
            status = manager.get_integration_status()
            print(json.dumps(status, indent=2))
            
        finally:
            await manager.stop_integration()
    
    asyncio.run(main())