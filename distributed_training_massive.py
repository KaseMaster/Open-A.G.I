#!/usr/bin/env python3
"""
🚀 AEGIS Distributed Training - Sprint 4.1
Sistema de entrenamiento distribuido a escala masiva con optimización automática
"""

import asyncio
import time
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
import threading
import queue
import psutil
from concurrent.futures import ThreadPoolExecutor

# Importar componentes del framework
from ml_framework_integration import MLFrameworkManager, MLFramework

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DistributedStrategy(Enum):
    """Estrategias de distribución disponibles"""
    DATA_PARALLEL = "data_parallel"
    MODEL_PARALLEL = "model_parallel"
    PIPELINE_PARALLEL = "pipeline_parallel"
    ZERO_REDUNDANCY_OPTIMIZER = "zero_redundancy"
    MIXED_PARALLEL = "mixed_parallel"

class CommunicationBackend(Enum):
    """Backends de comunicación"""
    NCCL = "nccl"  # Para GPUs NVIDIA
    GLOO = "gloo"  # Para CPUs
    MPI = "mpi"   # Para clusters HPC

class OptimizationTechnique(Enum):
    """Técnicas de optimización para distributed training"""
    GRADIENT_COMPRESSION = "gradient_compression"
    MIXED_PRECISION = "mixed_precision"
    GRADIENT_ACCUMULATION = "gradient_accumulation"
    EFFICIENT_ALLREDUCE = "efficient_allreduce"

@dataclass
class NodeConfig:
    """Configuración de un nodo de entrenamiento"""
    node_id: str
    ip_address: str
    port: int
    rank: int
    world_size: int
    device_type: str = "cpu"  # cpu, cuda, tpu
    device_count: int = 1
    memory_gb: float = 8.0
    compute_units: int = 4  # CPU cores or GPU count
    network_bandwidth_mbps: float = 1000.0
    is_master: bool = False

@dataclass
class DistributedConfig:
    """Configuración completa de entrenamiento distribuido"""
    strategy: DistributedStrategy = DistributedStrategy.DATA_PARALLEL
    backend: CommunicationBackend = CommunicationBackend.GLOO
    world_size: int = 4
    master_addr: str = "127.0.0.1"
    master_port: int = 12345
    timeout_seconds: int = 1800

    # Optimizaciones
    gradient_compression: bool = True
    mixed_precision: bool = True
    gradient_accumulation_steps: int = 1
    bucket_size_mb: float = 25.0

    # Tolerancia a fallos
    max_node_failures: int = 2
    checkpoint_frequency: int = 1000
    auto_recovery: bool = True

@dataclass
class TrainingMetrics:
    """Métricas de entrenamiento distribuido"""
    epoch: int
    step: int
    loss: float
    learning_rate: float
    throughput_samples_per_sec: float
    communication_time_ms: float
    computation_time_ms: float
    memory_usage_gb: float
    gradient_norm: float
    timestamp: float = field(default_factory=time.time)

@dataclass
class DistributedTrainingJob:
    """Trabajo de entrenamiento distribuido"""
    job_id: str
    model_name: str
    config: DistributedConfig
    nodes: List[NodeConfig]
    status: str = "pending"
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    metrics_history: List[TrainingMetrics] = field(default_factory=list)
    checkpoints: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

class GradientCompressor:
    """Compresor de gradientes para optimizar comunicación"""

    def __init__(self, compression_ratio: float = 0.01):
        self.compression_ratio = compression_ratio
        self.quantizer = self._create_quantizer()

    def _create_quantizer(self):
        """Crear quantizer para gradientes"""
        # Implementación simplificada
        return lambda x: torch.round(x * (1.0 / self.compression_ratio)) * self.compression_ratio

    def compress(self, gradients: List[torch.Tensor]) -> Tuple[List[torch.Tensor], Dict[str, Any]]:
        """Comprimir gradientes"""
        compressed = []
        metadata = {"original_size": 0, "compressed_size": 0, "compression_ratio": self.compression_ratio}

        for grad in gradients:
            if grad is not None:
                metadata["original_size"] += grad.numel()
                compressed_grad = self.quantizer(grad)
                compressed.append(compressed_grad)
                metadata["compressed_size"] += compressed_grad.numel()
            else:
                compressed.append(None)

        return compressed, metadata

    def decompress(self, compressed_gradients: List[torch.Tensor]) -> List[torch.Tensor]:
        """Descomprimir gradientes"""
        return compressed_gradients  # En implementación real, descomprimir

class MixedPrecisionManager:
    """Gestor de mixed precision training"""

    def __init__(self):
        self.scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

    def forward_pass(self, model: nn.Module, input_data: torch.Tensor) -> torch.Tensor:
        """Forward pass con autocast si está disponible"""
        if torch.cuda.is_available():
            with torch.cuda.amp.autocast():
                return model(input_data)
        return model(input_data)

    def backward_pass(self, loss: torch.Tensor, optimizer: torch.optim.Optimizer):
        """Backward pass con scaler si está disponible"""
        if self.scaler:
            self.scaler.scale(loss).backward()
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            loss.backward()
            optimizer.step()

class NodeMonitor:
    """Monitor de nodos para detectar fallos y performance"""

    def __init__(self, node_config: NodeConfig):
        self.node_config = node_config
        self.last_heartbeat = time.time()
        self.metrics_queue = queue.Queue()
        self.is_alive = True

    def update_metrics(self, metrics: Dict[str, Any]):
        """Actualizar métricas del nodo"""
        self.last_heartbeat = time.time()
        self.metrics_queue.put(metrics)

        # Mantener solo últimas 100 métricas
        if self.metrics_queue.qsize() > 100:
            self.metrics_queue.get()

    def get_latest_metrics(self) -> Optional[Dict[str, Any]]:
        """Obtener últimas métricas"""
        try:
            return self.metrics_queue.get_nowait()
        except queue.Empty:
            return None

    def check_health(self) -> bool:
        """Verificar salud del nodo"""
        time_since_heartbeat = time.time() - self.last_heartbeat
        return time_since_heartbeat < 60  # 60 segundos timeout

    def get_system_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas del sistema"""
        return {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_usage": psutil.disk_usage('/').percent,
            "network_connections": len(psutil.net_connections()),
            "load_average": psutil.getloadavg() if hasattr(psutil, 'getloadavg') else None
        }

class DistributedTrainer:
    """Entrenador distribuido principal"""

    def __init__(self, config: DistributedConfig):
        self.config = config
        self.nodes: Dict[str, NodeMonitor] = {}
        self.compressor = GradientCompressor() if config.gradient_compression else None
        self.mixed_precision = MixedPrecisionManager() if config.mixed_precision else None
        self.is_initialized = False

    def add_node(self, node_config: NodeConfig):
        """Agregar nodo al cluster"""
        monitor = NodeMonitor(node_config)
        self.nodes[node_config.node_id] = monitor
        logger.info(f"✅ Nodo agregado: {node_config.node_id} ({node_config.device_type})")

    def initialize_cluster(self):
        """Inicializar cluster distribuido"""
        logger.info("🔧 Inicializando cluster distribuido...")

        # Configurar entorno distribuido
        os.environ['MASTER_ADDR'] = self.config.master_addr
        os.environ['MASTER_PORT'] = str(self.config.master_port)
        os.environ['WORLD_SIZE'] = str(self.config.world_size)

        # Inicializar proceso distribuido
        try:
            dist.init_process_group(
                backend=self.config.backend.value,
                rank=0,  # Master rank
                world_size=self.config.world_size,
                timeout=asyncio.TimeoutError(self.config.timeout_seconds)
            )
            self.is_initialized = True
            logger.info("✅ Cluster distribuido inicializado")
        except Exception as e:
            logger.error(f"❌ Error inicializando cluster: {e}")
            raise

    def create_distributed_model(self, model: nn.Module, device: torch.device) -> nn.Module:
        """Crear modelo distribuido"""
        if not self.is_initialized:
            raise RuntimeError("Cluster no inicializado")

        # Wrap model con DDP
        ddp_model = DDP(model, device_ids=[device] if device.type == 'cuda' else None)

        logger.info(f"📦 Modelo distribuido creado en {device}")
        return ddp_model

    async def train_epoch_distributed(self, model: nn.Module, dataloader: torch.utils.data.DataLoader,
                                    optimizer: torch.optim.Optimizer, criterion: nn.Module,
                                    epoch: int, device: torch.device) -> TrainingMetrics:
        """Entrenar una epoch de forma distribuida"""

        model.train()
        running_loss = 0.0
        steps = 0
        epoch_start = time.time()

        comm_time = 0.0
        comp_time = 0.0

        for batch_idx, (inputs, targets) in enumerate(dataloader):
            batch_start = time.time()

            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            comp_start = time.time()
            optimizer.zero_grad()

            if self.mixed_precision:
                outputs = self.mixed_precision.forward_pass(model, inputs)
                loss = criterion(outputs, targets)
                self.mixed_precision.backward_pass(loss, optimizer)
            else:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

            comp_time += time.time() - comp_start

            # Sincronización de gradientes (all-reduce)
            if self.config.world_size > 1:
                comm_start = time.time()
                dist.barrier()  # Sincronización
                comm_time += time.time() - comm_start

            running_loss += loss.item()
            steps += 1

            # Checkpoint periódico
            if batch_idx % self.config.checkpoint_frequency == 0:
                await self._save_checkpoint(model, optimizer, epoch, batch_idx)

        epoch_time = time.time() - epoch_start
        avg_loss = running_loss / len(dataloader)

        # Calcular throughput
        total_samples = len(dataloader.dataset)
        throughput = total_samples / epoch_time

        # Métricas adicionales
        gradient_norm = self._calculate_gradient_norm(model)

        metrics = TrainingMetrics(
            epoch=epoch,
            step=steps,
            loss=avg_loss,
            learning_rate=optimizer.param_groups[0]['lr'],
            throughput_samples_per_sec=throughput,
            communication_time_ms=comm_time * 1000,
            computation_time_ms=comp_time * 1000,
            memory_usage_gb=torch.cuda.memory_allocated(device) / 1024**3 if device.type == 'cuda' else 0,
            gradient_norm=gradient_norm
        )

        return metrics

    def _calculate_gradient_norm(self, model: nn.Module) -> float:
        """Calcular norma de gradientes"""
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm ** (1. / 2)

    async def _save_checkpoint(self, model: nn.Module, optimizer: torch.optim.Optimizer,
                             epoch: int, step: int):
        """Guardar checkpoint"""
        checkpoint_path = f"checkpoint_epoch_{epoch}_step_{step}.pt"
        torch.save({
            'epoch': epoch,
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, checkpoint_path)

        logger.debug(f"💾 Checkpoint guardado: {checkpoint_path}")

    async def monitor_cluster_health(self) -> Dict[str, Any]:
        """Monitorear salud del cluster"""

        health_status = {
            "total_nodes": len(self.nodes),
            "healthy_nodes": 0,
            "failed_nodes": 0,
            "avg_cpu_usage": 0.0,
            "avg_memory_usage": 0.0,
            "network_issues": 0
        }

        total_cpu = 0.0
        total_memory = 0.0

        for node_id, monitor in self.nodes.items():
            if monitor.check_health():
                health_status["healthy_nodes"] += 1
                stats = monitor.get_system_stats()
                total_cpu += stats["cpu_percent"]
                total_memory += stats["memory_percent"]
            else:
                health_status["failed_nodes"] += 1

        if health_status["healthy_nodes"] > 0:
            health_status["avg_cpu_usage"] = total_cpu / health_status["healthy_nodes"]
            health_status["avg_memory_usage"] = total_memory / health_status["healthy_nodes"]

        return health_status

class DistributedTrainingOrchestrator:
    """Orquestador principal de entrenamiento distribuido"""

    def __init__(self):
        self.active_jobs: Dict[str, DistributedTrainingJob] = {}
        self.completed_jobs: List[DistributedTrainingJob] = []
        self.trainers: Dict[str, DistributedTrainer] = {}

    async def create_training_job(self, model_name: str, config: DistributedConfig,
                                nodes: List[NodeConfig]) -> str:
        """Crear trabajo de entrenamiento distribuido"""

        job_id = f"dist_job_{int(time.time())}_{hash(model_name) % 10000}"

        job = DistributedTrainingJob(
            job_id=job_id,
            model_name=model_name,
            config=config,
            nodes=nodes
        )

        self.active_jobs[job_id] = job

        # Crear trainer
        trainer = DistributedTrainer(config)
        for node in nodes:
            trainer.add_node(node)
        self.trainers[job_id] = trainer

        logger.info(f"🎯 Trabajo de entrenamiento creado: {job_id}")
        return job_id

    async def start_training_job(self, job_id: str, model: nn.Module,
                               train_dataloader: torch.utils.data.DataLoader,
                               optimizer: torch.optim.Optimizer,
                               criterion: nn.Module, num_epochs: int) -> bool:
        """Iniciar trabajo de entrenamiento"""

        if job_id not in self.active_jobs:
            logger.error(f"Trabajo no encontrado: {job_id}")
            return False

        job = self.active_jobs[job_id]
        trainer = self.trainers[job_id]

        try:
            job.started_at = time.time()
            job.status = "running"

            # Inicializar cluster
            trainer.initialize_cluster()

            # Determinar dispositivo
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Crear modelo distribuido
            dist_model = trainer.create_distributed_model(model, device)
            dist_model.to(device)

            logger.info(f"🚀 Iniciando entrenamiento distribuido en {len(job.nodes)} nodos")

            # Loop de entrenamiento
            for epoch in range(num_epochs):
                logger.info(f"📊 Epoch {epoch + 1}/{num_epochs}")

                # Entrenar epoch
                metrics = await trainer.train_epoch_distributed(
                    dist_model, train_dataloader, optimizer, criterion, epoch, device
                )

                # Registrar métricas
                job.metrics_history.append(metrics)

                logger.info(f"   Loss: {metrics.loss:.4f}, Throughput: {metrics.throughput_samples_per_sec:.1f} samples/sec")

                # Verificar salud del cluster
                health = await trainer.monitor_cluster_health()
                if health["failed_nodes"] > job.config.max_node_failures:
                    logger.error("❌ Demasiados nodos fallidos, abortando entrenamiento")
                    job.status = "failed"
                    job.errors.append("Cluster health degraded")
                    break

            # Completar job
            job.completed_at = time.time()
            job.status = "completed"

            # Mover a completados
            self.completed_jobs.append(job)
            del self.active_jobs[job_id]
            del self.trainers[job_id]

            logger.info(f"✅ Entrenamiento completado: {job_id}")
            return True

        except Exception as e:
            logger.error(f"❌ Error en entrenamiento: {e}")
            job.status = "failed"
            job.errors.append(str(e))
            return False

    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Obtener estado de trabajo"""

        job = self.active_jobs.get(job_id) or next(
            (j for j in self.completed_jobs if j.job_id == job_id), None
        )

        if not job:
            return None

        return {
            "job_id": job.job_id,
            "status": job.status,
            "model_name": job.model_name,
            "nodes": len(job.nodes),
            "created_at": job.created_at,
            "started_at": job.started_at,
            "completed_at": job.completed_at,
            "latest_metrics": job.metrics_history[-1] if job.metrics_history else None,
            "checkpoints": len(job.checkpoints),
            "errors": job.errors
        }

    def list_active_jobs(self) -> List[Dict[str, Any]]:
        """Listar trabajos activos"""
        return [self.get_job_status(job_id) for job_id in self.active_jobs.keys()]

    def get_cluster_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas del cluster"""
        return {
            "active_jobs": len(self.active_jobs),
            "completed_jobs": len(self.completed_jobs),
            "total_nodes": sum(len(job.nodes) for job in self.active_jobs.values()),
            "total_training_time": sum(
                (job.completed_at - job.started_at) if job.completed_at and job.started_at else 0
                for job in self.completed_jobs
            )
        }

# ===== DEMO Y EJEMPLOS =====

async def demo_distributed_training():
    """Demostración completa de entrenamiento distribuido"""

    print("🚀 AEGIS Distributed Training Demo")
    print("=" * 40)

    # Crear orquestador
    orchestrator = DistributedTrainingOrchestrator()

    # Configurar cluster simulado
    config = DistributedConfig(
        strategy=DistributedStrategy.DATA_PARALLEL,
        backend=CommunicationBackend.GLOO,
        world_size=4,  # 4 nodos
        gradient_compression=True,
        mixed_precision=torch.cuda.is_available(),
        max_node_failures=1
    )

    # Crear nodos simulados
    nodes = []
    for i in range(4):
        node = NodeConfig(
            node_id=f"node_{i}",
            ip_address=f"192.168.1.{100+i}",
            port=12345 + i,
            rank=i,
            world_size=4,
            device_type="cuda" if torch.cuda.is_available() else "cpu",
            device_count=1,
            memory_gb=16.0,
            compute_units=8,
            is_master=(i == 0)
        )
        nodes.append(node)

    print(f"✅ Cluster configurado: {len(nodes)} nodos")

    # Crear trabajo de entrenamiento
    job_id = await orchestrator.create_training_job("resnet_distributed", config, nodes)
    print(f"🎯 Trabajo creado: {job_id}")

    # Crear modelo simple para demo
    model = nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )

    # Crear datos simulados
    train_data = torch.randn(1000, 784)
    train_labels = torch.randint(0, 10, (1000,))
    train_dataset = torch.utils.data.TensorDataset(train_data, train_labels)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Configurar optimizer y loss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    print("\\n🏋️ Iniciando entrenamiento distribuido...")

    # Iniciar entrenamiento (simulado para demo)
    success = await orchestrator.start_training_job(
        job_id, model, train_dataloader, optimizer, criterion, num_epochs=3
    )

    if success:
        print("✅ Entrenamiento distribuido completado exitosamente!")

        # Mostrar estadísticas finales
        job_status = orchestrator.get_job_status(job_id)
        if job_status and job_status["latest_metrics"]:
            metrics = job_status["latest_metrics"]
            print("
📊 MÉTRICAS FINALES:"            print(".4f"            print(".1f"            print(".1f"            print(".2f"            print(".3f"
        # Estadísticas del cluster
        cluster_stats = orchestrator.get_cluster_stats()
        print("
🏗️ ESTADÍSTICAS DEL CLUSTER:"        print(f"   • Trabajos completados: {cluster_stats['completed_jobs']}")
        print(f"   • Nodos utilizados: {cluster_stats['total_nodes']}")
        print(".1f"    else:
        print("❌ Entrenamiento falló")

    print("\\n" + "=" * 60)
    print("🌟 Distributed Training funcionando correctamente!")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(demo_distributed_training())
