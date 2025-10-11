#!/usr/bin/env python3
"""
Sistema de Asignación Dinámica de Recursos Computacionales
AEGIS Security Framework - Uso Ético Únicamente

Este módulo implementa un sistema inteligente para:
- Gestión dinámica de recursos computacionales en la red P2P
- Balanceado de carga basado en capacidades y disponibilidad
- Asignación de tareas según especialización de nodos
- Monitoreo en tiempo real de rendimiento y utilización
- Algoritmos de optimización para eficiencia energética

ADVERTENCIA: Este código es para investigación y desarrollo ético únicamente.
El uso malicioso está estrictamente prohibido.
"""

import os
import time
import psutil
import asyncio
import threading
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import json
import logging
from collections import defaultdict, deque
import statistics
import hashlib

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ResourceType(Enum):
    """Tipos de recursos computacionales"""
    CPU = "cpu"
    MEMORY = "memory"
    STORAGE = "storage"
    NETWORK = "network"
    GPU = "gpu"
    SPECIALIZED = "specialized"  # Para tareas específicas como ML, criptografía

class TaskPriority(Enum):
    """Prioridades de tareas"""
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4
    BACKGROUND = 5

class NodeStatus(Enum):
    """Estados de los nodos"""
    ACTIVE = "active"
    BUSY = "busy"
    OVERLOADED = "overloaded"
    MAINTENANCE = "maintenance"
    OFFLINE = "offline"

class TaskType(Enum):
    """Tipos de tareas distribuidas"""
    COMPUTATION = "computation"
    STORAGE = "storage"
    VALIDATION = "validation"
    CONSENSUS = "consensus"
    MACHINE_LEARNING = "ml"
    CRYPTOGRAPHIC = "crypto"
    DATA_PROCESSING = "data_processing"

@dataclass
class ResourceCapacity:
    """Capacidades de recursos de un nodo"""
    cpu_cores: int
    cpu_frequency: float  # GHz
    memory_total: int  # MB
    storage_total: int  # MB
    network_bandwidth: float  # Mbps
    gpu_count: int = 0
    gpu_memory: int = 0  # MB
    specialized_units: Dict[str, int] = field(default_factory=dict)
    
    def calculate_power_score(self) -> float:
        """Calcular puntuación de poder computacional"""
        base_score = (
            self.cpu_cores * self.cpu_frequency * 0.3 +
            self.memory_total / 1024 * 0.2 +
            self.storage_total / 1024 * 0.1 +
            self.network_bandwidth * 0.2 +
            self.gpu_count * self.gpu_memory / 1024 * 0.2
        )
        
        # Bonificación por unidades especializadas
        specialized_bonus = sum(self.specialized_units.values()) * 0.1
        
        return base_score + specialized_bonus

@dataclass
class ResourceUtilization:
    """Utilización actual de recursos"""
    cpu_percent: float
    memory_percent: float
    storage_percent: float
    network_io: float  # MB/s
    gpu_percent: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def get_availability_score(self) -> float:
        """Calcular puntuación de disponibilidad (0-1)"""
        cpu_avail = max(0, 100 - self.cpu_percent) / 100
        mem_avail = max(0, 100 - self.memory_percent) / 100
        storage_avail = max(0, 100 - self.storage_percent) / 100
        gpu_avail = max(0, 100 - self.gpu_percent) / 100 if self.gpu_percent > 0 else 1.0
        
        return (cpu_avail * 0.4 + mem_avail * 0.3 + 
                storage_avail * 0.2 + gpu_avail * 0.1)

@dataclass
class ComputeTask:
    """Tarea computacional distribuida"""
    task_id: str
    task_type: TaskType
    priority: TaskPriority
    resource_requirements: Dict[ResourceType, float]
    estimated_duration: float  # segundos
    deadline: Optional[datetime] = None
    data_size: int = 0  # MB
    created_at: datetime = field(default_factory=datetime.utcnow)
    assigned_node: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result_data: Optional[bytes] = None
    
    def is_expired(self) -> bool:
        """Verificar si la tarea ha expirado"""
        if self.deadline is None:
            return False
        return datetime.utcnow() > self.deadline
    
    def get_urgency_score(self) -> float:
        """Calcular puntuación de urgencia"""
        base_urgency = 6 - self.priority.value  # Invertir prioridad
        
        if self.deadline:
            time_left = (self.deadline - datetime.utcnow()).total_seconds()
            if time_left <= 0:
                return 10.0  # Máxima urgencia
            
            # Aumentar urgencia conforme se acerca el deadline
            urgency_multiplier = max(1.0, 3600 / max(time_left, 60))
            return base_urgency * urgency_multiplier
        
        return base_urgency

@dataclass
class NodeProfile:
    """Perfil completo de un nodo"""
    node_id: str
    capacity: ResourceCapacity
    current_utilization: ResourceUtilization
    status: NodeStatus
    specializations: List[TaskType] = field(default_factory=list)
    reputation_score: float = 1.0
    last_seen: datetime = field(default_factory=datetime.utcnow)
    performance_history: deque = field(default_factory=lambda: deque(maxlen=100))
    task_completion_rate: float = 1.0
    average_response_time: float = 0.0
    
    def update_utilization(self, utilization: ResourceUtilization):
        """Actualizar utilización de recursos"""
        self.current_utilization = utilization
        self.last_seen = datetime.utcnow()
        
        # Actualizar estado basado en utilización
        availability = utilization.get_availability_score()
        if availability > 0.7:
            self.status = NodeStatus.ACTIVE
        elif availability > 0.3:
            self.status = NodeStatus.BUSY
        else:
            self.status = NodeStatus.OVERLOADED
    
    def calculate_suitability_score(self, task: ComputeTask) -> float:
        """Calcular qué tan adecuado es el nodo para una tarea"""
        if self.status == NodeStatus.OFFLINE:
            return 0.0
        
        # Puntuación base por disponibilidad
        availability_score = self.current_utilization.get_availability_score()
        
        # Bonificación por especialización
        specialization_bonus = 1.5 if task.task_type in self.specializations else 1.0
        
        # Penalización por sobrecarga
        overload_penalty = 0.5 if self.status == NodeStatus.OVERLOADED else 1.0
        
        # Bonificación por reputación
        reputation_bonus = self.reputation_score
        
        # Verificar requisitos mínimos
        requirements_met = self._check_requirements(task)
        if not requirements_met:
            return 0.0
        
        return (availability_score * specialization_bonus * 
                overload_penalty * reputation_bonus)
    
    def _check_requirements(self, task: ComputeTask) -> bool:
        """Verificar si el nodo cumple los requisitos de la tarea"""
        for resource_type, required_amount in task.resource_requirements.items():
            if resource_type == ResourceType.CPU:
                available = (100 - self.current_utilization.cpu_percent) / 100 * self.capacity.cpu_cores
                if available < required_amount:
                    return False
            elif resource_type == ResourceType.MEMORY:
                available = (100 - self.current_utilization.memory_percent) / 100 * self.capacity.memory_total
                if available < required_amount:
                    return False
            elif resource_type == ResourceType.STORAGE:
                available = (100 - self.current_utilization.storage_percent) / 100 * self.capacity.storage_total
                if available < required_amount:
                    return False
            elif resource_type == ResourceType.GPU:
                if self.capacity.gpu_count == 0 and required_amount > 0:
                    return False
        
        return True

class ResourceMonitor:
    """Monitor de recursos del sistema local"""
    
    def __init__(self, update_interval: float = 5.0):
        self.update_interval = update_interval
        self.running = False
        self.current_utilization: Optional[ResourceUtilization] = None
        self.capacity: Optional[ResourceCapacity] = None
        self._monitor_thread: Optional[threading.Thread] = None
        
    def start_monitoring(self):
        """Iniciar monitoreo de recursos"""
        if self.running:
            return
        
        self.running = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        
        logger.info("Monitor de recursos iniciado")
    
    def stop_monitoring(self):
        """Detener monitoreo de recursos"""
        self.running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
        
        logger.info("Monitor de recursos detenido")
    
    def get_system_capacity(self) -> ResourceCapacity:
        """Obtener capacidades del sistema"""
        if self.capacity is None:
            self.capacity = self._detect_capacity()
        return self.capacity
    
    def get_current_utilization(self) -> Optional[ResourceUtilization]:
        """Obtener utilización actual"""
        return self.current_utilization
    
    def _detect_capacity(self) -> ResourceCapacity:
        """Detectar capacidades del sistema"""
        cpu_count = psutil.cpu_count(logical=True)
        cpu_freq = psutil.cpu_freq()
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Detectar GPU (simplificado)
        gpu_count = 0
        gpu_memory = 0
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            gpu_count = len(gpus)
            gpu_memory = sum(gpu.memoryTotal for gpu in gpus) if gpus else 0
        except ImportError:
            pass
        
        return ResourceCapacity(
            cpu_cores=cpu_count,
            cpu_frequency=cpu_freq.current / 1000 if cpu_freq else 2.0,  # GHz
            memory_total=memory.total // (1024 * 1024),  # MB
            storage_total=disk.total // (1024 * 1024),  # MB
            network_bandwidth=100.0,  # Estimación por defecto
            gpu_count=gpu_count,
            gpu_memory=gpu_memory
        )
    
    def _monitor_loop(self):
        """Bucle principal de monitoreo"""
        while self.running:
            try:
                # CPU
                cpu_percent = psutil.cpu_percent(interval=1)
                
                # Memoria
                memory = psutil.virtual_memory()
                memory_percent = memory.percent
                
                # Almacenamiento
                disk = psutil.disk_usage('/')
                storage_percent = (disk.used / disk.total) * 100
                
                # Red
                net_io = psutil.net_io_counters()
                network_io = (net_io.bytes_sent + net_io.bytes_recv) / (1024 * 1024)  # MB
                
                # GPU (si está disponible)
                gpu_percent = 0.0
                try:
                    import GPUtil
                    gpus = GPUtil.getGPUs()
                    if gpus:
                        gpu_percent = statistics.mean(gpu.load * 100 for gpu in gpus)
                except ImportError:
                    pass
                
                self.current_utilization = ResourceUtilization(
                    cpu_percent=cpu_percent,
                    memory_percent=memory_percent,
                    storage_percent=storage_percent,
                    network_io=network_io,
                    gpu_percent=gpu_percent
                )
                
                time.sleep(self.update_interval)
                
            except Exception as e:
                logger.error(f"Error en monitoreo de recursos: {e}")
                time.sleep(self.update_interval)

class LoadBalancer:
    """Balanceador de carga inteligente"""
    
    def __init__(self):
        self.node_profiles: Dict[str, NodeProfile] = {}
        self.task_queue: List[ComputeTask] = []
        self.assignment_history: deque = deque(maxlen=1000)
        self.load_balancing_strategies = {
            'round_robin': self._round_robin_assignment,
            'least_loaded': self._least_loaded_assignment,
            'best_fit': self._best_fit_assignment,
            'weighted_random': self._weighted_random_assignment
        }
        self.current_strategy = 'best_fit'
    
    def register_node(self, node_profile: NodeProfile):
        """Registrar nuevo nodo en el balanceador"""
        self.node_profiles[node_profile.node_id] = node_profile
        logger.info(f"Nodo {node_profile.node_id} registrado en balanceador")
    
    def update_node_utilization(self, node_id: str, utilization: ResourceUtilization):
        """Actualizar utilización de un nodo"""
        if node_id in self.node_profiles:
            self.node_profiles[node_id].update_utilization(utilization)
    
    def assign_task(self, task: ComputeTask) -> Optional[str]:
        """Asignar tarea al nodo más adecuado"""
        strategy_func = self.load_balancing_strategies.get(self.current_strategy)
        if not strategy_func:
            logger.error(f"Estrategia desconocida: {self.current_strategy}")
            return None
        
        assigned_node = strategy_func(task)
        
        if assigned_node:
            task.assigned_node = assigned_node
            task.started_at = datetime.utcnow()
            
            # Registrar asignación
            self.assignment_history.append({
                'task_id': task.task_id,
                'node_id': assigned_node,
                'timestamp': datetime.utcnow(),
                'strategy': self.current_strategy
            })
            
            logger.info(f"Tarea {task.task_id} asignada a nodo {assigned_node}")
        
        return assigned_node
    
    def _round_robin_assignment(self, task: ComputeTask) -> Optional[str]:
        """Asignación por round robin"""
        active_nodes = [
            node for node in self.node_profiles.values()
            if node.status in [NodeStatus.ACTIVE, NodeStatus.BUSY]
        ]
        
        if not active_nodes:
            return None
        
        # Obtener último nodo usado
        last_assignment = self.assignment_history[-1] if self.assignment_history else None
        last_node_id = last_assignment['node_id'] if last_assignment else None
        
        # Encontrar siguiente nodo
        if last_node_id:
            try:
                last_index = next(i for i, node in enumerate(active_nodes) 
                                if node.node_id == last_node_id)
                next_index = (last_index + 1) % len(active_nodes)
            except StopIteration:
                next_index = 0
        else:
            next_index = 0
        
        return active_nodes[next_index].node_id
    
    def _least_loaded_assignment(self, task: ComputeTask) -> Optional[str]:
        """Asignación al nodo menos cargado"""
        suitable_nodes = [
            (node.node_id, node.current_utilization.get_availability_score())
            for node in self.node_profiles.values()
            if node.status in [NodeStatus.ACTIVE, NodeStatus.BUSY] and
               node._check_requirements(task)
        ]
        
        if not suitable_nodes:
            return None
        
        # Ordenar por disponibilidad (mayor disponibilidad primero)
        suitable_nodes.sort(key=lambda x: x[1], reverse=True)
        return suitable_nodes[0][0]
    
    def _best_fit_assignment(self, task: ComputeTask) -> Optional[str]:
        """Asignación por mejor ajuste"""
        node_scores = [
            (node.node_id, node.calculate_suitability_score(task))
            for node in self.node_profiles.values()
            if node.status != NodeStatus.OFFLINE
        ]
        
        # Filtrar nodos con puntuación > 0
        suitable_nodes = [(node_id, score) for node_id, score in node_scores if score > 0]
        
        if not suitable_nodes:
            return None
        
        # Ordenar por puntuación (mayor puntuación primero)
        suitable_nodes.sort(key=lambda x: x[1], reverse=True)
        return suitable_nodes[0][0]
    
    def _weighted_random_assignment(self, task: ComputeTask) -> Optional[str]:
        """Asignación aleatoria ponderada por capacidad"""
        import random
        
        node_weights = [
            (node.node_id, node.calculate_suitability_score(task))
            for node in self.node_profiles.values()
            if node.status != NodeStatus.OFFLINE
        ]
        
        # Filtrar nodos con peso > 0
        suitable_nodes = [(node_id, weight) for node_id, weight in node_weights if weight > 0]
        
        if not suitable_nodes:
            return None
        
        # Selección aleatoria ponderada
        total_weight = sum(weight for _, weight in suitable_nodes)
        random_value = random.uniform(0, total_weight)
        
        cumulative_weight = 0
        for node_id, weight in suitable_nodes:
            cumulative_weight += weight
            if random_value <= cumulative_weight:
                return node_id
        
        return suitable_nodes[-1][0]  # Fallback
    
    def get_load_statistics(self) -> Dict[str, Any]:
        """Obtener estadísticas de carga"""
        if not self.node_profiles:
            return {}
        
        active_nodes = [
            node for node in self.node_profiles.values()
            if node.status != NodeStatus.OFFLINE
        ]
        
        if not active_nodes:
            return {}
        
        cpu_utilizations = [node.current_utilization.cpu_percent for node in active_nodes]
        memory_utilizations = [node.current_utilization.memory_percent for node in active_nodes]
        
        return {
            'total_nodes': len(self.node_profiles),
            'active_nodes': len(active_nodes),
            'average_cpu_utilization': statistics.mean(cpu_utilizations),
            'max_cpu_utilization': max(cpu_utilizations),
            'average_memory_utilization': statistics.mean(memory_utilizations),
            'max_memory_utilization': max(memory_utilizations),
            'current_strategy': self.current_strategy,
            'total_assignments': len(self.assignment_history)
        }

class ResourceManager:
    """Gestor principal de recursos distribuidos"""
    
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.monitor = ResourceMonitor()
        self.load_balancer = LoadBalancer()
        self.task_queue: asyncio.Queue = asyncio.Queue()
        self.active_tasks: Dict[str, ComputeTask] = {}
        self.completed_tasks: deque = deque(maxlen=1000)
        self.running = False
        
        # Callbacks para comunicación con otros componentes
        self.task_executor: Optional[Callable] = None
        self.network_communicator: Optional[Callable] = None
        
    async def start(self):
        """Iniciar gestor de recursos"""
        if self.running:
            return
        
        self.running = True
        self.monitor.start_monitoring()
        
        # Registrar nodo local
        capacity = self.monitor.get_system_capacity()
        utilization = self.monitor.get_current_utilization() or ResourceUtilization(0, 0, 0, 0)
        
        local_profile = NodeProfile(
            node_id=self.node_id,
            capacity=capacity,
            current_utilization=utilization,
            status=NodeStatus.ACTIVE
        )
        
        self.load_balancer.register_node(local_profile)
        
        # Iniciar tareas de gestión
        asyncio.create_task(self._resource_update_loop())
        asyncio.create_task(self._task_processing_loop())
        asyncio.create_task(self._cleanup_loop())
        
        logger.info(f"ResourceManager iniciado para nodo {self.node_id}")
    
    async def stop(self):
        """Detener gestor de recursos"""
        self.running = False
        self.monitor.stop_monitoring()
        
        # Completar tareas activas
        for task in self.active_tasks.values():
            task.completed_at = datetime.utcnow()
            self.completed_tasks.append(task)
        
        self.active_tasks.clear()
        
        logger.info("ResourceManager detenido")
    
    async def submit_task(self, task: ComputeTask) -> bool:
        """Enviar tarea para procesamiento"""
        if task.is_expired():
            logger.warning(f"Tarea {task.task_id} expirada, rechazada")
            return False
        
        await self.task_queue.put(task)
        logger.info(f"Tarea {task.task_id} enviada a cola")
        return True
    
    async def register_remote_node(self, node_profile: NodeProfile):
        """Registrar nodo remoto"""
        self.load_balancer.register_node(node_profile)
        logger.info(f"Nodo remoto {node_profile.node_id} registrado")
    
    async def update_remote_node(self, node_id: str, utilization: ResourceUtilization):
        """Actualizar estado de nodo remoto"""
        self.load_balancer.update_node_utilization(node_id, utilization)
    
    def set_task_executor(self, executor: Callable):
        """Establecer función ejecutora de tareas"""
        self.task_executor = executor
    
    def set_network_communicator(self, communicator: Callable):
        """Establecer función de comunicación de red"""
        self.network_communicator = communicator
    
    async def _resource_update_loop(self):
        """Bucle de actualización de recursos"""
        while self.running:
            try:
                utilization = self.monitor.get_current_utilization()
                if utilization:
                    self.load_balancer.update_node_utilization(self.node_id, utilization)
                
                await asyncio.sleep(5.0)
                
            except Exception as e:
                logger.error(f"Error en actualización de recursos: {e}")
                await asyncio.sleep(5.0)
    
    async def _task_processing_loop(self):
        """Bucle de procesamiento de tareas"""
        while self.running:
            try:
                # Obtener tarea de la cola
                task = await asyncio.wait_for(self.task_queue.get(), timeout=1.0)
                
                # Asignar tarea
                assigned_node = self.load_balancer.assign_task(task)
                
                if assigned_node == self.node_id:
                    # Ejecutar localmente
                    await self._execute_local_task(task)
                elif assigned_node and self.network_communicator:
                    # Enviar a nodo remoto
                    await self.network_communicator(assigned_node, task)
                else:
                    # No se pudo asignar
                    logger.warning(f"No se pudo asignar tarea {task.task_id}")
                    await asyncio.sleep(1.0)
                    await self.task_queue.put(task)  # Reintentarlo
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error en procesamiento de tareas: {e}")
                await asyncio.sleep(1.0)
    
    async def _execute_local_task(self, task: ComputeTask):
        """Ejecutar tarea localmente"""
        self.active_tasks[task.task_id] = task
        
        try:
            if self.task_executor:
                result = await self.task_executor(task)
                task.result_data = result
            
            task.completed_at = datetime.utcnow()
            self.completed_tasks.append(task)
            
            logger.info(f"Tarea {task.task_id} completada localmente")
            
        except Exception as e:
            logger.error(f"Error ejecutando tarea {task.task_id}: {e}")
            task.completed_at = datetime.utcnow()
            
        finally:
            self.active_tasks.pop(task.task_id, None)
    
    async def _cleanup_loop(self):
        """Bucle de limpieza de recursos"""
        while self.running:
            try:
                # Limpiar tareas expiradas
                current_time = datetime.utcnow()
                expired_tasks = [
                    task_id for task_id, task in self.active_tasks.items()
                    if task.is_expired()
                ]
                
                for task_id in expired_tasks:
                    task = self.active_tasks.pop(task_id)
                    task.completed_at = current_time
                    self.completed_tasks.append(task)
                    logger.warning(f"Tarea {task_id} expirada y removida")
                
                await asyncio.sleep(60.0)  # Limpiar cada minuto
                
            except Exception as e:
                logger.error(f"Error en limpieza: {e}")
                await asyncio.sleep(60.0)
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Obtener métricas del sistema"""
        utilization = self.monitor.get_current_utilization()
        capacity = self.monitor.get_system_capacity()
        load_stats = self.load_balancer.get_load_statistics()
        
        return {
            'node_id': self.node_id,
            'capacity': {
                'cpu_cores': capacity.cpu_cores,
                'memory_total_mb': capacity.memory_total,
                'storage_total_mb': capacity.storage_total,
                'power_score': capacity.calculate_power_score()
            },
            'utilization': {
                'cpu_percent': utilization.cpu_percent if utilization else 0,
                'memory_percent': utilization.memory_percent if utilization else 0,
                'storage_percent': utilization.storage_percent if utilization else 0,
                'availability_score': utilization.get_availability_score() if utilization else 1.0
            },
            'tasks': {
                'active_count': len(self.active_tasks),
                'completed_count': len(self.completed_tasks),
                'queue_size': self.task_queue.qsize()
            },
            'load_balancing': load_stats
        }

# Funciones de utilidad

def create_compute_task(task_type: TaskType, priority: TaskPriority,
                       cpu_cores: float = 1.0, memory_mb: float = 512.0,
                       duration_seconds: float = 60.0) -> ComputeTask:
    """Crear tarea computacional con parámetros básicos"""
    task_id = hashlib.sha256(f"{time.time()}{task_type.value}".encode()).hexdigest()[:16]
    
    return ComputeTask(
        task_id=task_id,
        task_type=task_type,
        priority=priority,
        resource_requirements={
            ResourceType.CPU: cpu_cores,
            ResourceType.MEMORY: memory_mb
        },
        estimated_duration=duration_seconds
    )

async def example_task_executor(task: ComputeTask) -> bytes:
    """Ejemplo de ejecutor de tareas"""
    logger.info(f"Ejecutando tarea {task.task_id} de tipo {task.task_type.value}")
    
    # Simular trabajo computacional
    await asyncio.sleep(min(task.estimated_duration, 5.0))
    
    # Retornar resultado simulado
    result = f"Resultado de {task.task_id}".encode()
    return result

# Demostración del sistema
async def demo_resource_management():
    """Demostración del sistema de gestión de recursos"""
    print("🔧 Demo del Sistema de Gestión de Recursos")
    print("=" * 50)
    
    # Crear gestor de recursos
    manager = ResourceManager("demo_node_001")
    manager.set_task_executor(example_task_executor)
    
    await manager.start()
    
    # Crear algunas tareas de ejemplo
    tasks = [
        create_compute_task(TaskType.COMPUTATION, TaskPriority.HIGH, 2.0, 1024.0, 30.0),
        create_compute_task(TaskType.MACHINE_LEARNING, TaskPriority.MEDIUM, 4.0, 2048.0, 120.0),
        create_compute_task(TaskType.DATA_PROCESSING, TaskPriority.LOW, 1.0, 512.0, 60.0)
    ]
    
    # Enviar tareas
    for task in tasks:
        await manager.submit_task(task)
        print(f"✅ Tarea {task.task_id} enviada")
    
    # Esperar procesamiento
    await asyncio.sleep(10.0)
    
    # Mostrar métricas
    metrics = manager.get_system_metrics()
    print(f"\n📊 Métricas del Sistema:")
    print(f"CPU: {metrics['utilization']['cpu_percent']:.1f}%")
    print(f"Memoria: {metrics['utilization']['memory_percent']:.1f}%")
    print(f"Tareas activas: {metrics['tasks']['active_count']}")
    print(f"Tareas completadas: {metrics['tasks']['completed_count']}")
    
    await manager.stop()

def init_pool(config: dict):
    """Adapter a nivel de módulo para inicializar el pool de recursos.
    Crea ResourceManager y lanza su servicio de fondo.
    """
    try:
        node_id = config.get("node_id", "node_local")
        manager = ResourceManager(node_id=node_id)
        loop = asyncio.get_event_loop()
        loop.create_task(manager.start())
        logger.info(f"🧩 ResourceManager iniciado para node_id={node_id}")
        return manager
    except Exception as e:
        logger.error(f"❌ No se pudo iniciar ResourceManager: {e}")
        return None

if __name__ == "__main__":
    asyncio.run(demo_resource_management())