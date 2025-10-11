#!/usr/bin/env python3
"""
Sistema de Tolerancia a Fallos y Recuperación de Nodos - AEGIS Framework
Implementación robusta para detección, recuperación y mantenimiento de nodos en red P2P distribuida.

Características principales:
- Heartbeat distribuido con detección inteligente
- Recuperación automática de nodos caídos
- Replicación de datos críticos
- Balanceador de carga resiliente
- Monitoreo de salud en tiempo real
"""

import asyncio
import time
import json
import hashlib
import logging
from typing import Dict, List, Set, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict, deque
import aiohttp
import socket
from datetime import datetime, timedelta

# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NodeStatus(Enum):
    """Estados posibles de un nodo en la red"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    SUSPECTED = "suspected"
    FAILED = "failed"
    RECOVERING = "recovering"
    OFFLINE = "offline"

class FailureType(Enum):
    """Tipos de fallos detectables"""
    NETWORK_PARTITION = "network_partition"
    NODE_CRASH = "node_crash"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    BYZANTINE_BEHAVIOR = "byzantine_behavior"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    TIMEOUT = "timeout"

@dataclass
class NodeInfo:
    """Información completa de un nodo"""
    node_id: str
    address: str
    port: int
    status: NodeStatus
    last_heartbeat: float
    capabilities: Dict[str, Any]
    load_metrics: Dict[str, float]
    failure_count: int = 0
    recovery_attempts: int = 0
    join_time: float = 0.0
    reputation_score: float = 1.0

@dataclass
class FailureEvent:
    """Evento de fallo detectado"""
    node_id: str
    failure_type: FailureType
    timestamp: float
    details: Dict[str, Any]
    severity: str  # "low", "medium", "high", "critical"

class HeartbeatManager:
    """Gestor de heartbeats distribuido con detección inteligente"""
    
    def __init__(self, node_id: str, heartbeat_interval: float = 5.0):
        self.node_id = node_id
        self.heartbeat_interval = heartbeat_interval
        self.nodes: Dict[str, NodeInfo] = {}
        self.failure_detector_threshold = 3  # Fallos consecutivos para marcar como sospechoso
        self.recovery_timeout = 30.0  # Tiempo para intentar recuperación
        self.running = False
        
    async def start_heartbeat_service(self):
        """Inicia el servicio de heartbeat"""
        self.running = True
        logger.info(f"🫀 Iniciando servicio de heartbeat para nodo {self.node_id}")
        
        # Tareas concurrentes
        tasks = [
            asyncio.create_task(self._send_heartbeats()),
            asyncio.create_task(self._monitor_nodes()),
            asyncio.create_task(self._cleanup_stale_nodes())
        ]
        
        await asyncio.gather(*tasks)
    
    async def _send_heartbeats(self):
        """Envía heartbeats periódicos a todos los nodos conocidos"""
        while self.running:
            try:
                heartbeat_data = {
                    "node_id": self.node_id,
                    "timestamp": time.time(),
                    "status": "healthy",
                    "load_metrics": await self._get_load_metrics(),
                    "sequence": int(time.time())
                }
                
                # Enviar a todos los nodos activos
                for node_id, node_info in self.nodes.items():
                    if node_info.status in [NodeStatus.HEALTHY, NodeStatus.DEGRADED]:
                        await self._send_heartbeat_to_node(node_info, heartbeat_data)
                
                await asyncio.sleep(self.heartbeat_interval)
                
            except Exception as e:
                logger.error(f"❌ Error enviando heartbeats: {e}")
                await asyncio.sleep(1)
    
    async def _send_heartbeat_to_node(self, node_info: NodeInfo, heartbeat_data: Dict):
        """Envía heartbeat a un nodo específico"""
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=3)) as session:
                url = f"http://{node_info.address}:{node_info.port}/heartbeat"
                async with session.post(url, json=heartbeat_data) as response:
                    if response.status == 200:
                        # Heartbeat exitoso
                        node_info.failure_count = 0
                        if node_info.status == NodeStatus.SUSPECTED:
                            node_info.status = NodeStatus.HEALTHY
                            logger.info(f"✅ Nodo {node_info.node_id} recuperado")
                    else:
                        await self._handle_heartbeat_failure(node_info)
                        
        except Exception as e:
            await self._handle_heartbeat_failure(node_info)
    
    async def _handle_heartbeat_failure(self, node_info: NodeInfo):
        """Maneja fallos de heartbeat"""
        node_info.failure_count += 1
        
        if node_info.failure_count >= self.failure_detector_threshold:
            if node_info.status == NodeStatus.HEALTHY:
                node_info.status = NodeStatus.SUSPECTED
                logger.warning(f"⚠️ Nodo {node_info.node_id} marcado como sospechoso")
                
                # Iniciar proceso de recuperación
                await self._initiate_node_recovery(node_info)
            
            elif node_info.status == NodeStatus.SUSPECTED:
                node_info.status = NodeStatus.FAILED
                logger.error(f"💀 Nodo {node_info.node_id} marcado como fallido")
                
                # Notificar fallo crítico
                await self._handle_node_failure(node_info)
    
    async def _monitor_nodes(self):
        """Monitorea el estado de todos los nodos"""
        while self.running:
            try:
                current_time = time.time()
                
                for node_id, node_info in list(self.nodes.items()):
                    # Verificar timeout de heartbeat
                    if current_time - node_info.last_heartbeat > self.heartbeat_interval * 3:
                        if node_info.status == NodeStatus.HEALTHY:
                            node_info.status = NodeStatus.SUSPECTED
                            logger.warning(f"⏰ Timeout de heartbeat para nodo {node_id}")
                    
                    # Verificar métricas de rendimiento
                    await self._check_performance_degradation(node_info)
                
                await asyncio.sleep(self.heartbeat_interval / 2)
                
            except Exception as e:
                logger.error(f"❌ Error monitoreando nodos: {e}")
                await asyncio.sleep(1)
    
    async def _check_performance_degradation(self, node_info: NodeInfo):
        """Verifica degradación de rendimiento"""
        if not node_info.load_metrics:
            return
        
        # Umbrales de degradación
        cpu_threshold = 0.9
        memory_threshold = 0.9
        response_time_threshold = 5.0
        
        cpu_usage = node_info.load_metrics.get("cpu_usage", 0)
        memory_usage = node_info.load_metrics.get("memory_usage", 0)
        response_time = node_info.load_metrics.get("avg_response_time", 0)
        
        if (cpu_usage > cpu_threshold or 
            memory_usage > memory_threshold or 
            response_time > response_time_threshold):
            
            if node_info.status == NodeStatus.HEALTHY:
                node_info.status = NodeStatus.DEGRADED
                logger.warning(f"📉 Nodo {node_info.node_id} degradado por rendimiento")
    
    async def _get_load_metrics(self) -> Dict[str, float]:
        """Obtiene métricas de carga del nodo actual"""
        try:
            import psutil
            return {
                "cpu_usage": psutil.cpu_percent(interval=0.1),
                "memory_usage": psutil.virtual_memory().percent / 100.0,
                "disk_usage": psutil.disk_usage('/').percent / 100.0,
                "network_io": psutil.net_io_counters().bytes_sent + psutil.net_io_counters().bytes_recv,
                "avg_response_time": 0.1,  # Placeholder
                "active_connections": len(psutil.net_connections())
            }
        except ImportError:
            # Métricas simuladas si psutil no está disponible
            return {
                "cpu_usage": 0.3,
                "memory_usage": 0.4,
                "disk_usage": 0.2,
                "network_io": 1000000,
                "avg_response_time": 0.1,
                "active_connections": 10
            }

class DataReplicationManager:
    """Gestor de replicación de datos críticos"""
    
    def __init__(self, node_id: str, replication_factor: int = 3):
        self.node_id = node_id
        self.replication_factor = replication_factor
        self.data_store: Dict[str, Any] = {}
        self.replica_locations: Dict[str, List[str]] = {}  # data_key -> [node_ids]
        self.consistency_level = "quorum"  # "one", "quorum", "all"
        
    async def store_data(self, key: str, data: Any, critical: bool = False) -> bool:
        """Almacena datos con replicación automática"""
        try:
            # Almacenar localmente
            self.data_store[key] = {
                "data": data,
                "timestamp": time.time(),
                "version": self._generate_version(key),
                "critical": critical,
                "checksum": self._calculate_checksum(data)
            }
            
            # Seleccionar nodos para replicación
            replica_nodes = await self._select_replica_nodes(key, critical)
            self.replica_locations[key] = replica_nodes
            
            # Replicar a nodos seleccionados
            replication_tasks = []
            for node_id in replica_nodes:
                task = asyncio.create_task(
                    self._replicate_to_node(node_id, key, self.data_store[key])
                )
                replication_tasks.append(task)
            
            # Esperar replicación según nivel de consistencia
            if self.consistency_level == "all":
                results = await asyncio.gather(*replication_tasks, return_exceptions=True)
                success_count = sum(1 for r in results if r is True)
                return success_count == len(replica_nodes)
            
            elif self.consistency_level == "quorum":
                results = await asyncio.gather(*replication_tasks, return_exceptions=True)
                success_count = sum(1 for r in results if r is True)
                return success_count >= (len(replica_nodes) // 2 + 1)
            
            else:  # "one"
                # Al menos una replicación exitosa
                for task in asyncio.as_completed(replication_tasks):
                    result = await task
                    if result is True:
                        return True
                return False
                
        except Exception as e:
            logger.error(f"❌ Error almacenando datos {key}: {e}")
            return False
    
    async def _select_replica_nodes(self, key: str, critical: bool) -> List[str]:
        """Selecciona nodos óptimos para replicación"""
        # Implementación simplificada - en producción usaría consistent hashing
        available_nodes = [
            node_id for node_id, node_info in self.heartbeat_manager.nodes.items()
            if node_info.status in [NodeStatus.HEALTHY, NodeStatus.DEGRADED]
        ]
        
        # Para datos críticos, preferir nodos con mejor reputación
        if critical:
            available_nodes.sort(
                key=lambda nid: self.heartbeat_manager.nodes[nid].reputation_score,
                reverse=True
            )
        
        # Seleccionar hasta replication_factor nodos
        return available_nodes[:min(self.replication_factor, len(available_nodes))]
    
    async def _replicate_to_node(self, node_id: str, key: str, data_entry: Dict) -> bool:
        """Replica datos a un nodo específico"""
        try:
            node_info = self.heartbeat_manager.nodes.get(node_id)
            if not node_info:
                return False
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
                url = f"http://{node_info.address}:{node_info.port}/replicate"
                payload = {
                    "key": key,
                    "data_entry": data_entry,
                    "source_node": self.node_id
                }
                
                async with session.post(url, json=payload) as response:
                    return response.status == 200
                    
        except Exception as e:
            logger.error(f"❌ Error replicando a nodo {node_id}: {e}")
            return False
    
    def _generate_version(self, key: str) -> str:
        """Genera versión única para el dato"""
        return f"{self.node_id}_{int(time.time() * 1000000)}"
    
    def _calculate_checksum(self, data: Any) -> str:
        """Calcula checksum de los datos"""
        data_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(data_str.encode()).hexdigest()

class NodeRecoveryManager:
    """Gestor de recuperación automática de nodos"""
    
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.recovery_strategies = {
            FailureType.NODE_CRASH: self._recover_from_crash,
            FailureType.NETWORK_PARTITION: self._recover_from_partition,
            FailureType.PERFORMANCE_DEGRADATION: self._recover_from_degradation,
            FailureType.RESOURCE_EXHAUSTION: self._recover_from_resource_exhaustion,
            FailureType.TIMEOUT: self._recover_from_timeout
        }
        self.recovery_history: Dict[str, List[FailureEvent]] = defaultdict(list)
        self.max_recovery_attempts = 3
        
    async def initiate_recovery(self, node_info: NodeInfo, failure_type: FailureType) -> bool:
        """Inicia proceso de recuperación para un nodo"""
        try:
            logger.info(f"🔧 Iniciando recuperación de nodo {node_info.node_id} por {failure_type.value}")
            
            # Verificar límite de intentos
            if node_info.recovery_attempts >= self.max_recovery_attempts:
                logger.error(f"❌ Máximo de intentos de recuperación alcanzado para {node_info.node_id}")
                return False
            
            node_info.recovery_attempts += 1
            node_info.status = NodeStatus.RECOVERING
            
            # Registrar evento de fallo
            failure_event = FailureEvent(
                node_id=node_info.node_id,
                failure_type=failure_type,
                timestamp=time.time(),
                details={"recovery_attempt": node_info.recovery_attempts},
                severity="high"
            )
            self.recovery_history[node_info.node_id].append(failure_event)
            
            # Ejecutar estrategia de recuperación específica
            recovery_strategy = self.recovery_strategies.get(failure_type)
            if recovery_strategy:
                success = await recovery_strategy(node_info, failure_event)
                
                if success:
                    node_info.status = NodeStatus.HEALTHY
                    node_info.recovery_attempts = 0
                    logger.info(f"✅ Recuperación exitosa de nodo {node_info.node_id}")
                    return True
                else:
                    logger.warning(f"⚠️ Fallo en recuperación de nodo {node_info.node_id}")
                    return False
            else:
                logger.error(f"❌ No hay estrategia de recuperación para {failure_type.value}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error en recuperación de nodo {node_info.node_id}: {e}")
            return False
    
    async def _recover_from_crash(self, node_info: NodeInfo, failure_event: FailureEvent) -> bool:
        """Recuperación de crash de nodo"""
        try:
            # Intentar reconectar
            await asyncio.sleep(2)  # Esperar antes de reintentar
            
            # Verificar si el nodo responde
            if await self._ping_node(node_info):
                # Sincronizar datos perdidos
                await self._synchronize_node_data(node_info)
                return True
            
            # Si no responde, intentar reinicio remoto
            return await self._attempt_remote_restart(node_info)
            
        except Exception as e:
            logger.error(f"❌ Error recuperando de crash: {e}")
            return False
    
    async def _recover_from_partition(self, node_info: NodeInfo, failure_event: FailureEvent) -> bool:
        """Recuperación de partición de red"""
        try:
            # Esperar a que se resuelva la partición
            await asyncio.sleep(5)
            
            # Verificar conectividad
            if await self._ping_node(node_info):
                # Resolver conflictos de datos
                await self._resolve_partition_conflicts(node_info)
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"❌ Error recuperando de partición: {e}")
            return False
    
    async def _recover_from_degradation(self, node_info: NodeInfo, failure_event: FailureEvent) -> bool:
        """Recuperación de degradación de rendimiento"""
        try:
            # Reducir carga del nodo
            await self._redistribute_load(node_info)
            
            # Esperar mejora
            await asyncio.sleep(10)
            
            # Verificar mejora en métricas
            current_metrics = await self._get_node_metrics(node_info)
            if self._is_performance_acceptable(current_metrics):
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"❌ Error recuperando de degradación: {e}")
            return False
    
    async def _recover_from_resource_exhaustion(self, node_info: NodeInfo, failure_event: FailureEvent) -> bool:
        """Recuperación de agotamiento de recursos"""
        try:
            # Liberar recursos
            await self._cleanup_node_resources(node_info)
            
            # Redistribuir carga
            await self._redistribute_load(node_info)
            
            await asyncio.sleep(5)
            return await self._ping_node(node_info)
            
        except Exception as e:
            logger.error(f"❌ Error recuperando de agotamiento de recursos: {e}")
            return False
    
    async def _recover_from_timeout(self, node_info: NodeInfo, failure_event: FailureEvent) -> bool:
        """Recuperación de timeouts"""
        try:
            # Aumentar timeouts temporalmente
            await asyncio.sleep(3)
            
            # Verificar conectividad con timeout extendido
            return await self._ping_node(node_info, timeout=10)
            
        except Exception as e:
            logger.error(f"❌ Error recuperando de timeout: {e}")
            return False
    
    async def _ping_node(self, node_info: NodeInfo, timeout: int = 5) -> bool:
        """Verifica si un nodo responde"""
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
                url = f"http://{node_info.address}:{node_info.port}/ping"
                async with session.get(url) as response:
                    return response.status == 200
        except:
            return False

class FaultToleranceOrchestrator:
    """Orquestador principal del sistema de tolerancia a fallos"""
    
    def __init__(self, node_id: str, config: Dict[str, Any] = None):
        self.node_id = node_id
        self.config = config or {}
        
        # Componentes principales
        self.heartbeat_manager = HeartbeatManager(node_id)
        self.replication_manager = DataReplicationManager(node_id)
        self.recovery_manager = NodeRecoveryManager(node_id)
        
        # Referencias cruzadas
        self.replication_manager.heartbeat_manager = self.heartbeat_manager
        
        # Estado del sistema
        self.system_health = "healthy"
        self.active_failures: Dict[str, FailureEvent] = {}
        self.performance_metrics = {}
        
    async def start(self):
        """Inicia el sistema de tolerancia a fallos"""
        logger.info(f"🚀 Iniciando sistema de tolerancia a fallos para nodo {self.node_id}")
        
        # Iniciar componentes
        tasks = [
            asyncio.create_task(self.heartbeat_manager.start_heartbeat_service()),
            asyncio.create_task(self._monitor_system_health()),
            asyncio.create_task(self._periodic_maintenance())
        ]
        
        await asyncio.gather(*tasks)
    
    async def _monitor_system_health(self):
        """Monitorea la salud general del sistema"""
        while True:
            try:
                # Calcular métricas de salud
                total_nodes = len(self.heartbeat_manager.nodes)
                healthy_nodes = sum(
                    1 for node in self.heartbeat_manager.nodes.values()
                    if node.status == NodeStatus.HEALTHY
                )
                
                if total_nodes > 0:
                    health_ratio = healthy_nodes / total_nodes
                    
                    if health_ratio >= 0.8:
                        self.system_health = "healthy"
                    elif health_ratio >= 0.6:
                        self.system_health = "degraded"
                    else:
                        self.system_health = "critical"
                        logger.error(f"🚨 Sistema en estado crítico: {health_ratio:.2%} nodos saludables")
                
                await asyncio.sleep(30)  # Verificar cada 30 segundos
                
            except Exception as e:
                logger.error(f"❌ Error monitoreando salud del sistema: {e}")
                await asyncio.sleep(5)
    
    async def _periodic_maintenance(self):
        """Mantenimiento periódico del sistema"""
        while True:
            try:
                # Limpiar historial antiguo
                await self._cleanup_old_events()
                
                # Optimizar replicación
                await self._optimize_replication()
                
                # Generar reporte de salud
                await self._generate_health_report()
                
                await asyncio.sleep(300)  # Cada 5 minutos
                
            except Exception as e:
                logger.error(f"❌ Error en mantenimiento periódico: {e}")
                await asyncio.sleep(60)
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Obtiene estado completo del sistema"""
        return {
            "node_id": self.node_id,
            "system_health": self.system_health,
            "timestamp": time.time(),
            "nodes": {
                node_id: {
                    "status": node.status.value,
                    "last_heartbeat": node.last_heartbeat,
                    "failure_count": node.failure_count,
                    "load_metrics": node.load_metrics
                }
                for node_id, node in self.heartbeat_manager.nodes.items()
            },
            "active_failures": len(self.active_failures),
            "replication_factor": self.replication_manager.replication_factor,
            "data_entries": len(self.replication_manager.data_store)
        }

# Función principal para testing
async def main():
    """Función principal para pruebas"""
    node_id = f"node_{int(time.time())}"
    
    # Crear y iniciar sistema de tolerancia a fallos
    fault_tolerance = FaultToleranceOrchestrator(node_id)
    
    # Simular algunos nodos
    fault_tolerance.heartbeat_manager.nodes["node_1"] = NodeInfo(
        node_id="node_1",
        address="127.0.0.1",
        port=8001,
        status=NodeStatus.HEALTHY,
        last_heartbeat=time.time(),
        capabilities={"cpu_cores": 4, "memory_gb": 8},
        load_metrics={"cpu_usage": 0.3, "memory_usage": 0.4}
    )
    
    fault_tolerance.heartbeat_manager.nodes["node_2"] = NodeInfo(
        node_id="node_2",
        address="127.0.0.1",
        port=8002,
        status=NodeStatus.HEALTHY,
        last_heartbeat=time.time(),
        capabilities={"cpu_cores": 8, "memory_gb": 16},
        load_metrics={"cpu_usage": 0.2, "memory_usage": 0.3}
    )
    
    # Mostrar estado inicial
    status = await fault_tolerance.get_system_status()
    print("🔍 Estado inicial del sistema:")
    print(json.dumps(status, indent=2))
    
    # Simular almacenamiento de datos críticos
    await fault_tolerance.replication_manager.store_data(
        "critical_config",
        {"encryption_key": "abc123", "consensus_threshold": 0.67},
        critical=True
    )
    
    print("✅ Sistema de tolerancia a fallos inicializado correctamente")

if __name__ == "__main__":
    asyncio.run(main())