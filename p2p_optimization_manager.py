#!/usr/bin/env python3
"""
Gestor de Optimización P2P para AEGIS Framework
Mejoras de rendimiento, conectividad y estabilidad de la red P2P
"""

import asyncio
import logging
import time
import json
import statistics
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

class P2POptimizationLevel(Enum):
    """Niveles de optimización P2P"""
    CONSERVATIVE = "conservative"  # Configuración conservadora
    BALANCED = "balanced"         # Balance rendimiento/estabilidad
    AGGRESSIVE = "aggressive"     # Máximo rendimiento

class ConnectionQuality(Enum):
    """Calidad de conexión P2P"""
    EXCELLENT = "excellent"  # < 50ms latencia, > 95% uptime
    GOOD = "good"           # < 100ms latencia, > 90% uptime
    FAIR = "fair"           # < 200ms latencia, > 80% uptime
    POOR = "poor"           # > 200ms latencia, < 80% uptime

@dataclass
class PeerMetrics:
    """Métricas de rendimiento de un peer"""
    peer_id: str
    ip_address: str
    port: int
    
    # Métricas de conectividad
    connection_attempts: int = 0
    successful_connections: int = 0
    failed_connections: int = 0
    last_connection_time: float = 0
    
    # Métricas de rendimiento
    average_latency: float = 0.0
    latency_samples: List[float] = None
    bandwidth_up: float = 0.0  # KB/s
    bandwidth_down: float = 0.0  # KB/s
    
    # Métricas de estabilidad
    uptime_percentage: float = 0.0
    disconnection_count: int = 0
    last_seen: float = 0
    
    # Métricas de calidad
    message_success_rate: float = 0.0
    data_integrity_score: float = 1.0
    
    def __post_init__(self):
        if self.latency_samples is None:
            self.latency_samples = deque(maxlen=100)  # Últimas 100 muestras
    
    @property
    def connection_success_rate(self) -> float:
        """Tasa de éxito de conexiones"""
        if self.connection_attempts == 0:
            return 0.0
        return self.successful_connections / self.connection_attempts
    
    @property
    def quality_score(self) -> float:
        """Puntuación de calidad del peer (0-1)"""
        factors = [
            self.connection_success_rate,
            min(1.0, 200.0 / max(self.average_latency, 1.0)),  # Latencia inversa
            self.uptime_percentage / 100.0,
            self.message_success_rate,
            self.data_integrity_score
        ]
        return statistics.mean(factors)
    
    @property
    def connection_quality(self) -> ConnectionQuality:
        """Determina la calidad de conexión"""
        if (self.average_latency < 50 and self.uptime_percentage > 95 and 
            self.connection_success_rate > 0.95):
            return ConnectionQuality.EXCELLENT
        elif (self.average_latency < 100 and self.uptime_percentage > 90 and 
              self.connection_success_rate > 0.90):
            return ConnectionQuality.GOOD
        elif (self.average_latency < 200 and self.uptime_percentage > 80 and 
              self.connection_success_rate > 0.80):
            return ConnectionQuality.FAIR
        else:
            return ConnectionQuality.POOR

@dataclass
class P2POptimizationConfig:
    """Configuración de optimización P2P"""
    
    # Configuración general
    optimization_level: P2POptimizationLevel = P2POptimizationLevel.BALANCED
    max_peers: int = 50
    min_peers: int = 5
    target_peers: int = 20
    
    # Configuración de conexión
    connection_timeout: float = 10.0
    handshake_timeout: float = 5.0
    keep_alive_interval: float = 30.0
    max_connection_attempts: int = 3
    
    # Configuración de descubrimiento
    discovery_interval: float = 60.0
    mdns_enabled: bool = True
    dht_enabled: bool = True
    bootstrap_nodes: List[str] = None
    
    # Configuración de calidad
    min_quality_score: float = 0.3
    peer_evaluation_interval: float = 300.0  # 5 minutos
    quality_check_samples: int = 10
    
    # Configuración de rendimiento
    message_queue_size: int = 1000
    concurrent_connections: int = 10
    bandwidth_limit_kb: Optional[int] = None
    
    # Configuración de recuperación
    auto_reconnect: bool = True
    reconnect_delay: float = 5.0
    max_reconnect_attempts: int = 5
    
    def __post_init__(self):
        if self.bootstrap_nodes is None:
            self.bootstrap_nodes = []

class P2POptimizationManager:
    """Gestor de optimización para la red P2P"""
    
    def __init__(self, config: P2POptimizationConfig = None):
        self.config = config or P2POptimizationConfig()
        self.peer_metrics: Dict[str, PeerMetrics] = {}
        self.connection_pool: Set[str] = set()
        self.blacklisted_peers: Set[str] = set()
        self.preferred_peers: Set[str] = set()
        
        # Estadísticas de red
        self.network_stats = {
            'total_connections': 0,
            'successful_connections': 0,
            'failed_connections': 0,
            'average_network_latency': 0.0,
            'network_uptime': 0.0,
            'start_time': time.time()
        }
        
        # Tareas de optimización
        self._optimization_tasks: List[asyncio.Task] = []
        self._running = False
    
    async def start_optimization(self):
        """Inicia el sistema de optimización"""
        if self._running:
            return
        
        self._running = True
        logger.info("Iniciando sistema de optimización P2P")
        
        # Iniciar tareas de optimización
        self._optimization_tasks = [
            asyncio.create_task(self._peer_quality_monitor()),
            asyncio.create_task(self._connection_optimizer()),
            asyncio.create_task(self._network_health_monitor()),
            asyncio.create_task(self._peer_discovery_optimizer())
        ]
        
        logger.info("Sistema de optimización P2P iniciado")
    
    async def stop_optimization(self):
        """Detiene el sistema de optimización"""
        if not self._running:
            return
        
        self._running = False
        logger.info("Deteniendo sistema de optimización P2P")
        
        # Cancelar tareas
        for task in self._optimization_tasks:
            task.cancel()
        
        # Esperar a que terminen
        await asyncio.gather(*self._optimization_tasks, return_exceptions=True)
        self._optimization_tasks.clear()
        
        logger.info("Sistema de optimización P2P detenido")
    
    def record_connection_attempt(self, peer_id: str, ip_address: str, port: int, success: bool):
        """Registra un intento de conexión"""
        if peer_id not in self.peer_metrics:
            self.peer_metrics[peer_id] = PeerMetrics(peer_id, ip_address, port)
        
        metrics = self.peer_metrics[peer_id]
        metrics.connection_attempts += 1
        
        if success:
            metrics.successful_connections += 1
            metrics.last_connection_time = time.time()
            self.network_stats['successful_connections'] += 1
        else:
            metrics.failed_connections += 1
            self.network_stats['failed_connections'] += 1
        
        self.network_stats['total_connections'] += 1
    
    def record_latency(self, peer_id: str, latency_ms: float):
        """Registra la latencia de un peer"""
        if peer_id in self.peer_metrics:
            metrics = self.peer_metrics[peer_id]
            metrics.latency_samples.append(latency_ms)
            metrics.average_latency = statistics.mean(metrics.latency_samples)
    
    def record_disconnection(self, peer_id: str):
        """Registra una desconexión"""
        if peer_id in self.peer_metrics:
            self.peer_metrics[peer_id].disconnection_count += 1
    
    def update_peer_uptime(self, peer_id: str, uptime_percentage: float):
        """Actualiza el uptime de un peer"""
        if peer_id in self.peer_metrics:
            self.peer_metrics[peer_id].uptime_percentage = uptime_percentage
            self.peer_metrics[peer_id].last_seen = time.time()
    
    def get_best_peers(self, count: int = None) -> List[PeerMetrics]:
        """Obtiene los mejores peers según su calidad"""
        if count is None:
            count = self.config.target_peers
        
        # Filtrar peers activos y de buena calidad
        active_peers = [
            metrics for metrics in self.peer_metrics.values()
            if (metrics.quality_score >= self.config.min_quality_score and
                metrics.peer_id not in self.blacklisted_peers and
                time.time() - metrics.last_seen < 300)  # Visto en últimos 5 min
        ]
        
        # Ordenar por calidad
        active_peers.sort(key=lambda p: p.quality_score, reverse=True)
        
        return active_peers[:count]
    
    def should_connect_to_peer(self, peer_id: str) -> bool:
        """Determina si se debe conectar a un peer"""
        if peer_id in self.blacklisted_peers:
            return False
        
        if peer_id in self.preferred_peers:
            return True
        
        if peer_id not in self.peer_metrics:
            return True  # Nuevo peer, intentar conexión
        
        metrics = self.peer_metrics[peer_id]
        return metrics.quality_score >= self.config.min_quality_score
    
    def blacklist_peer(self, peer_id: str, reason: str = ""):
        """Añade un peer a la lista negra"""
        self.blacklisted_peers.add(peer_id)
        logger.warning(f"Peer {peer_id} añadido a lista negra: {reason}")
    
    def whitelist_peer(self, peer_id: str):
        """Añade un peer a la lista de preferidos"""
        self.preferred_peers.add(peer_id)
        if peer_id in self.blacklisted_peers:
            self.blacklisted_peers.remove(peer_id)
        logger.info(f"Peer {peer_id} añadido a lista de preferidos")
    
    async def _peer_quality_monitor(self):
        """Monitorea la calidad de los peers"""
        while self._running:
            try:
                current_time = time.time()
                
                # Evaluar calidad de peers
                for peer_id, metrics in list(self.peer_metrics.items()):
                    # Verificar si el peer está obsoleto
                    if current_time - metrics.last_seen > 1800:  # 30 minutos
                        logger.debug(f"Removiendo peer obsoleto: {peer_id}")
                        del self.peer_metrics[peer_id]
                        continue
                    
                    # Verificar calidad
                    if metrics.quality_score < self.config.min_quality_score:
                        if metrics.connection_attempts > 5:  # Dar oportunidades
                            self.blacklist_peer(peer_id, "Calidad insuficiente")
                
                await asyncio.sleep(self.config.peer_evaluation_interval)
                
            except Exception as e:
                logger.error(f"Error en monitor de calidad: {e}")
                await asyncio.sleep(30)
    
    async def _connection_optimizer(self):
        """Optimiza las conexiones P2P"""
        while self._running:
            try:
                # Obtener peers activos
                active_peers = len([
                    p for p in self.peer_metrics.values()
                    if time.time() - p.last_seen < 60
                ])
                
                # Ajustar conexiones según configuración
                if active_peers < self.config.min_peers:
                    logger.info(f"Pocos peers activos ({active_peers}), iniciando descubrimiento")
                    # Aquí se podría triggear descubrimiento adicional
                
                elif active_peers > self.config.max_peers:
                    logger.info(f"Demasiados peers ({active_peers}), optimizando conexiones")
                    # Desconectar peers de menor calidad
                    worst_peers = sorted(
                        self.peer_metrics.values(),
                        key=lambda p: p.quality_score
                    )[:active_peers - self.config.target_peers]
                    
                    for peer in worst_peers:
                        if peer.quality_score < 0.5:  # Solo desconectar peers realmente malos
                            logger.debug(f"Desconectando peer de baja calidad: {peer.peer_id}")
                
                await asyncio.sleep(60)  # Revisar cada minuto
                
            except Exception as e:
                logger.error(f"Error en optimizador de conexiones: {e}")
                await asyncio.sleep(30)
    
    async def _network_health_monitor(self):
        """Monitorea la salud general de la red"""
        while self._running:
            try:
                # Calcular estadísticas de red
                if self.peer_metrics:
                    latencies = [p.average_latency for p in self.peer_metrics.values() if p.average_latency > 0]
                    if latencies:
                        self.network_stats['average_network_latency'] = statistics.mean(latencies)
                
                # Calcular uptime de red
                uptime = time.time() - self.network_stats['start_time']
                self.network_stats['network_uptime'] = uptime
                
                # Log estadísticas periódicamente
                if int(uptime) % 300 == 0:  # Cada 5 minutos
                    self._log_network_stats()
                
                await asyncio.sleep(30)
                
            except Exception as e:
                logger.error(f"Error en monitor de salud: {e}")
                await asyncio.sleep(30)
    
    async def _peer_discovery_optimizer(self):
        """Optimiza el descubrimiento de peers"""
        while self._running:
            try:
                # Verificar si necesitamos más peers
                active_count = len([
                    p for p in self.peer_metrics.values()
                    if time.time() - p.last_seen < 120
                ])
                
                if active_count < self.config.target_peers:
                    logger.debug("Iniciando descubrimiento optimizado de peers")
                    # Aquí se implementaría lógica de descubrimiento inteligente
                
                await asyncio.sleep(self.config.discovery_interval)
                
            except Exception as e:
                logger.error(f"Error en optimizador de descubrimiento: {e}")
                await asyncio.sleep(60)
    
    def _log_network_stats(self):
        """Log de estadísticas de red"""
        stats = self.network_stats.copy()
        stats['active_peers'] = len([
            p for p in self.peer_metrics.values()
            if time.time() - p.last_seen < 60
        ])
        stats['blacklisted_peers'] = len(self.blacklisted_peers)
        stats['preferred_peers'] = len(self.preferred_peers)
        
        logger.info(f"Estadísticas de red P2P: {json.dumps(stats, indent=2)}")
    
    def get_optimization_report(self) -> Dict:
        """Genera un reporte de optimización"""
        current_time = time.time()
        
        # Peers por calidad
        quality_distribution = defaultdict(int)
        for metrics in self.peer_metrics.values():
            quality_distribution[metrics.connection_quality.value] += 1
        
        # Peers activos
        active_peers = [
            p for p in self.peer_metrics.values()
            if current_time - p.last_seen < 60
        ]
        
        # Convertir configuración a formato serializable
        config_dict = asdict(self.config)
        config_dict['optimization_level'] = self.config.optimization_level.value
        
        return {
            'timestamp': current_time,
            'network_stats': self.network_stats,
            'peer_count': {
                'total': len(self.peer_metrics),
                'active': len(active_peers),
                'blacklisted': len(self.blacklisted_peers),
                'preferred': len(self.preferred_peers)
            },
            'quality_distribution': dict(quality_distribution),
            'best_peers': [
                {
                    'peer_id': p.peer_id,
                    'quality_score': p.quality_score,
                    'latency': p.average_latency,
                    'uptime': p.uptime_percentage
                }
                for p in self.get_best_peers(5)
            ],
            'optimization_config': config_dict
        }

def create_optimized_p2p_config(level: P2POptimizationLevel = P2POptimizationLevel.BALANCED) -> P2POptimizationConfig:
    """Crea una configuración P2P optimizada"""
    
    config = P2POptimizationConfig(optimization_level=level)
    
    if level == P2POptimizationLevel.CONSERVATIVE:
        config.max_peers = 30
        config.target_peers = 15
        config.connection_timeout = 15.0
        config.max_connection_attempts = 2
        config.min_quality_score = 0.5
        
    elif level == P2POptimizationLevel.AGGRESSIVE:
        config.max_peers = 100
        config.target_peers = 50
        config.connection_timeout = 5.0
        config.max_connection_attempts = 5
        config.min_quality_score = 0.2
        config.concurrent_connections = 20
    
    return config

if __name__ == "__main__":
    # Ejemplo de uso
    async def main():
        config = create_optimized_p2p_config(P2POptimizationLevel.BALANCED)
        manager = P2POptimizationManager(config)
        
        await manager.start_optimization()
        
        # Simular algunas métricas
        manager.record_connection_attempt("peer1", "192.168.1.100", 8080, True)
        manager.record_latency("peer1", 45.0)
        manager.update_peer_uptime("peer1", 98.5)
        
        # Generar reporte
        report = manager.get_optimization_report()
        print(json.dumps(report, indent=2))
        
        await manager.stop_optimization()
    
    asyncio.run(main())