#!/usr/bin/env python3
"""
📤 AEGIS Model Distribution System
Sistema para distribución eficiente de modelos entre nodos
con replicación, versionado y sincronización automática
"""

import asyncio
import hashlib
import json
import time
import secrets
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
import logging

from ml_framework_integration import MLFrameworkManager

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DistributionStatus(Enum):
    """Estados de distribución"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class ReplicationStrategy(Enum):
    """Estrategias de replicación"""
    PUSH = "push"  # El origen empuja a destinos
    PULL = "pull"  # Los destinos obtienen del origen
    HYBRID = "hybrid"  # Combinación de ambas

@dataclass
class ModelChunk:
    """Chunk de modelo para distribución eficiente"""
    chunk_id: str
    model_id: str
    chunk_index: int
    total_chunks: int
    data: bytes
    checksum: str
    size: int
    timestamp: float

@dataclass
class DistributionTask:
    """Tarea de distribución de modelo"""
    task_id: str
    model_id: str
    source_node: str
    target_nodes: Set[str]
    strategy: ReplicationStrategy
    status: DistributionStatus
    priority: int
    created_at: float
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    progress: Dict[str, float] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    bandwidth_used: int = 0
    chunks_total: int = 0
    chunks_transferred: int = 0

@dataclass
class ModelVersion:
    """Versión de modelo con metadatos"""
    version_id: str
    model_id: str
    version_number: str
    parent_version: Optional[str]
    created_at: float
    size_bytes: int
    checksum: str
    metadata: Dict[str, Any]
    is_active: bool = True

class ModelDistributionService:
    """Servicio de distribución de modelos"""

    def __init__(self, ml_manager: MLFrameworkManager, node_id: str,
                 chunk_size: int = 1048576):  # 1MB chunks
        self.ml_manager = ml_manager
        self.node_id = node_id
        self.chunk_size = chunk_size

        # Almacenamiento local
        self.local_models: Dict[str, bytes] = {}  # model_id -> serialized_model
        self.model_versions: Dict[str, List[ModelVersion]] = {}  # model_id -> versions
        self.distribution_tasks: Dict[str, DistributionTask] = {}

        # Cache de chunks para distribución eficiente
        self.chunk_cache: Dict[str, ModelChunk] = {}

        # Configuración de red
        self.max_concurrent_transfers = 5
        self.transfer_semaphore = asyncio.Semaphore(self.max_concurrent_transfers)

    async def register_model_for_distribution(self, model_id: str,
                                           initial_version: str = "1.0.0") -> bool:
        """Registrar modelo para distribución"""

        if model_id not in self.ml_manager.models:
            logger.error(f"Modelo {model_id} no encontrado en ML manager")
            return False

        # Crear versión inicial
        version = ModelVersion(
            version_id=f"{model_id}_v_{initial_version}",
            model_id=model_id,
            version_number=initial_version,
            parent_version=None,
            created_at=time.time(),
            size_bytes=0,  # Se calcula después
            checksum="",
            metadata={"framework": self.ml_manager.models[model_id].framework.value}
        )

        if model_id not in self.model_versions:
            self.model_versions[model_id] = []
        self.model_versions[model_id].append(version)

        logger.info(f"✅ Modelo {model_id} registrado para distribución (v{initial_version})")
        return True

    async def create_model_version(self, model_id: str, changes_description: str = "") -> Optional[str]:
        """Crear nueva versión de modelo"""

        if model_id not in self.model_versions:
            logger.error(f"Modelo {model_id} no registrado para distribución")
            return None

        versions = self.model_versions[model_id]
        if not versions:
            return None

        # Calcular nueva versión
        current_version = versions[-1].version_number
        version_parts = current_version.split('.')
        version_parts[-1] = str(int(version_parts[-1]) + 1)
        new_version_num = '.'.join(version_parts)

        # Crear nueva versión
        new_version = ModelVersion(
            version_id=f"{model_id}_v_{new_version_num}",
            model_id=model_id,
            version_number=new_version_num,
            parent_version=versions[-1].version_id,
            created_at=time.time(),
            size_bytes=0,
            checksum="",
            metadata={
                "changes": changes_description,
                "created_by": self.node_id,
                "framework": versions[-1].metadata.get("framework")
            }
        )

        versions.append(new_version)
        logger.info(f"✅ Nueva versión creada: {model_id} v{new_version_num}")

        return new_version.version_id

    async def distribute_model(self, model_id: str, target_nodes: List[str],
                             strategy: ReplicationStrategy = ReplicationStrategy.PUSH,
                             priority: int = 1) -> Optional[str]:
        """Iniciar distribución de modelo"""

        if model_id not in self.local_models and model_id not in self.ml_manager.models:
            logger.error(f"Modelo {model_id} no disponible localmente")
            return None

        # Crear tarea de distribución
        task_id = f"dist_{secrets.token_hex(4)}"
        task = DistributionTask(
            task_id=task_id,
            model_id=model_id,
            source_node=self.node_id,
            target_nodes=set(target_nodes),
            strategy=strategy,
            status=DistributionStatus.PENDING,
            priority=priority,
            created_at=time.time()
        )

        self.distribution_tasks[task_id] = task

        # Iniciar distribución en background
        asyncio.create_task(self._execute_distribution(task))

        logger.info(f"🚀 Distribución iniciada: {model_id} -> {len(target_nodes)} nodos")
        return task_id

    async def _execute_distribution(self, task: DistributionTask):
        """Ejecutar distribución de modelo"""

        try:
            task.status = DistributionStatus.IN_PROGRESS
            task.started_at = time.time()

            # Obtener modelo serializado
            model_data = await self._get_model_data(task.model_id)
            if not model_data:
                raise ValueError(f"No se pudo obtener datos del modelo {task.model_id}")

            # Dividir en chunks
            chunks = self._split_into_chunks(task.model_id, model_data)
            task.chunks_total = len(chunks)

            logger.info(f"📦 Modelo dividido en {len(chunks)} chunks")

            # Distribuir según estrategia
            if task.strategy == ReplicationStrategy.PUSH:
                await self._push_distribution(task, chunks)
            elif task.strategy == ReplicationStrategy.PULL:
                await self._setup_pull_distribution(task, chunks)
            else:  # HYBRID
                await self._hybrid_distribution(task, chunks)

            task.status = DistributionStatus.COMPLETED
            task.completed_at = time.time()

            duration = task.completed_at - task.started_at
            logger.info(f"✅ Distribución completada: {task.model_id} en {duration:.1f}s")
        except Exception as e:
            task.status = DistributionStatus.FAILED
            task.errors.append(str(e))
            logger.error(f"❌ Distribución fallida {task.task_id}: {e}")

    async def _push_distribution(self, task: DistributionTask, chunks: List[ModelChunk]):
        """Distribución push: origen empuja a destinos"""

        # Para cada destino
        for target_node in task.target_nodes:
            try:
                # Transferir chunks concurrentemente con límite
                transfer_tasks = []
                for chunk in chunks:
                    task_coro = self._transfer_chunk_to_node(chunk, target_node)
                    transfer_tasks.append(task_coro)

                # Ejecutar transferencias en lotes para no sobrecargar
                batch_size = 3
                for i in range(0, len(transfer_tasks), batch_size):
                    batch = transfer_tasks[i:i+batch_size]
                    await asyncio.gather(*batch, return_exceptions=True)

                    # Actualizar progreso
                    task.chunks_transferred += len(batch)
                    task.progress[target_node] = (task.chunks_transferred / task.chunks_total) * 100

                logger.info(f"✅ Modelo enviado a {target_node}")

            except Exception as e:
                task.errors.append(f"Error enviando a {target_node}: {e}")
                logger.error(f"❌ Error enviando a {target_node}: {e}")

    async def _setup_pull_distribution(self, task: DistributionTask, chunks: List[ModelChunk]):
        """Configurar distribución pull: destinos obtienen del origen"""

        # Cachear chunks para que los nodos puedan obtenerlos
        for chunk in chunks:
            self.chunk_cache[chunk.chunk_id] = chunk

        # Notificar a nodos destino (simulado - en producción usaría P2P)
        for target_node in task.target_nodes:
            logger.info(f"📥 Configurado pull para {target_node} - {len(chunks)} chunks disponibles")

        # En escenario real, aquí se enviaría notificación P2P
        task.status = DistributionStatus.COMPLETED

    async def _hybrid_distribution(self, task: DistributionTask, chunks: List[ModelChunk]):
        """Distribución híbrida"""

        # Combinar push y pull basado en prioridad y disponibilidad
        high_priority_nodes = list(task.target_nodes)[:len(task.target_nodes)//2]
        low_priority_nodes = list(task.target_nodes)[len(task.target_nodes)//2:]

        # Push para nodos de alta prioridad
        if high_priority_nodes:
            push_task = DistributionTask(
                task_id=f"{task.task_id}_push",
                model_id=task.model_id,
                source_node=task.source_node,
                target_nodes=set(high_priority_nodes),
                strategy=ReplicationStrategy.PUSH,
                status=DistributionStatus.IN_PROGRESS,
                priority=task.priority,
                created_at=time.time()
            )
            await self._push_distribution(push_task, chunks)

        # Pull para nodos de baja prioridad
        if low_priority_nodes:
            pull_task = DistributionTask(
                task_id=f"{task.task_id}_pull",
                model_id=task.model_id,
                source_node=task.source_node,
                target_nodes=set(low_priority_nodes),
                strategy=ReplicationStrategy.PULL,
                status=DistributionStatus.IN_PROGRESS,
                priority=task.priority - 1,
                created_at=time.time()
            )
            await self._setup_pull_distribution(pull_task, chunks)

    async def _transfer_chunk_to_node(self, chunk: ModelChunk, target_node: str) -> bool:
        """Transferir chunk a nodo específico"""

        async with self.transfer_semaphore:
            try:
                # Simular transferencia de red (en producción usaría P2P)
                await asyncio.sleep(0.01)  # Simular latencia

                # Verificar integridad
                received_checksum = hashlib.sha256(chunk.data).hexdigest()
                if received_checksum != chunk.checksum:
                    raise ValueError(f"Checksum mismatch para chunk {chunk.chunk_id}")

                # Actualizar métricas
                self.distribution_tasks[chunk.model_id.split('_')[0]].bandwidth_used += chunk.size

                logger.debug(f"📤 Chunk {chunk.chunk_index+1}/{chunk.total_chunks} enviado a {target_node}")
                return True

            except Exception as e:
                logger.error(f"❌ Error transfiriendo chunk {chunk.chunk_id} a {target_node}: {e}")
                return False

    async def _get_model_data(self, model_id: str) -> Optional[bytes]:
        """Obtener datos serializados del modelo"""

        # Primero verificar si ya está en cache local
        if model_id in self.local_models:
            return self.local_models[model_id]

        # Intentar obtener del ML manager
        try:
            # En escenario real, aquí se serializaría el modelo
            # Por simplicidad, creamos datos dummy
            dummy_size = 1024 * 1024  # 1MB
            model_data = secrets.token_bytes(dummy_size)

            # Cachear localmente
            self.local_models[model_id] = model_data
            return model_data

        except Exception as e:
            logger.error(f"Error obteniendo modelo {model_id}: {e}")
            return None

    def _split_into_chunks(self, model_id: str, model_data: bytes) -> List[ModelChunk]:
        """Dividir modelo en chunks para distribución eficiente"""

        chunks = []
        total_size = len(model_data)

        for i in range(0, total_size, self.chunk_size):
            chunk_data = model_data[i:i + self.chunk_size]
            chunk_id = f"{model_id}_chunk_{i // self.chunk_size}"

            chunk = ModelChunk(
                chunk_id=chunk_id,
                model_id=model_id,
                chunk_index=i // self.chunk_size,
                total_chunks=(total_size + self.chunk_size - 1) // self.chunk_size,
                data=chunk_data,
                checksum=hashlib.sha256(chunk_data).hexdigest(),
                size=len(chunk_data),
                timestamp=time.time()
            )

            chunks.append(chunk)

        return chunks

    def get_distribution_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Obtener estado de tarea de distribución"""

        if task_id not in self.distribution_tasks:
            return None

        task = self.distribution_tasks[task_id]
        return {
            "task_id": task.task_id,
            "model_id": task.model_id,
            "status": task.status.value,
            "progress": task.progress,
            "target_nodes": list(task.target_nodes),
            "chunks_total": task.chunks_total,
            "chunks_transferred": task.chunks_transferred,
            "bandwidth_used": task.bandwidth_used,
            "errors": task.errors,
            "created_at": task.created_at,
            "started_at": task.started_at,
            "completed_at": task.completed_at
        }

    def get_model_versions(self, model_id: str) -> List[Dict[str, Any]]:
        """Obtener versiones de un modelo"""

        if model_id not in self.model_versions:
            return []

        return [
            {
                "version_id": v.version_id,
                "version_number": v.version_number,
                "created_at": v.created_at,
                "size_bytes": v.size_bytes,
                "is_active": v.is_active,
                "metadata": v.metadata
            }
            for v in self.model_versions[model_id]
        ]

    async def cleanup_old_versions(self, model_id: str, keep_versions: int = 5):
        """Limpiar versiones antiguas de modelo"""

        if model_id not in self.model_versions:
            return

        versions = self.model_versions[model_id]
        if len(versions) <= keep_versions:
            return

        # Mantener solo las versiones más recientes
        versions_to_remove = versions[:-keep_versions]
        for version in versions_to_remove:
            version.is_active = False

        logger.info(f"🧹 Limpiadas {len(versions_to_remove)} versiones antiguas de {model_id}")

async def demo_model_distribution():
    """Demostración del sistema de distribución de modelos"""

    print("📤 DEMO - AEGIS Model Distribution System")
    print("=" * 50)

    # Inicializar componentes
    ml_manager = MLFrameworkManager()
    distribution_service = ModelDistributionService(ml_manager, "demo_node")

    print("✅ Sistema de distribución inicializado")

    # Registrar modelos para distribución
    print("\n📝 Registrando modelos para distribución...")

    test_models = ["model_1", "model_2", "model_3"]
    for model_id in test_models:
        # Simular registro en ML manager
        from ml_framework_integration import ModelMetadata, MLFramework, ModelType
        metadata = ModelMetadata(
            model_id=model_id,
            framework=MLFramework.PYTORCH,
            model_type=ModelType.CLASSIFICATION,
            architecture="Test Model",
            input_shape=[784],
            output_shape=[10],
            parameters=100000,
            created_at=time.time(),
            updated_at=time.time(),
            version="1.0.0"
        )
        ml_manager.models[model_id] = metadata

        # Registrar para distribución
        await distribution_service.register_model_for_distribution(model_id)
        print(f"✅ Modelo {model_id} registrado para distribución")

    # Simular nodos destino
    target_nodes = ["node_alpha", "node_beta", "node_gamma", "node_delta"]

    print("\n🎯 Iniciando distribuciones...")
    # Distribuir modelos con diferentes estrategias
    distribution_tasks = []

    for i, model_id in enumerate(test_models):
        # Alternar estrategias
        strategies = [ReplicationStrategy.PUSH, ReplicationStrategy.PULL, ReplicationStrategy.HYBRID]
        strategy = strategies[i % len(strategies)]

        # Distribuir a subconjunto de nodos
        nodes_for_model = target_nodes[i:i+3] if i < 2 else target_nodes

        task_id = await distribution_service.distribute_model(
            model_id=model_id,
            target_nodes=nodes_for_model,
            strategy=strategy,
            priority=3-i  # Diferentes prioridades
        )

        if task_id:
            distribution_tasks.append((task_id, model_id, strategy))
            print(f"🚀 Distribución iniciada: {model_id} -> {nodes_for_model} ({strategy.value})")

    # Esperar completación
    print("\n⏳ Esperando completación de distribuciones...")
    await asyncio.sleep(3)  # Dar tiempo a las tareas

    # Mostrar resultados
    print("\n📊 RESULTADOS DE DISTRIBUCIÓN:")
    for task_id, model_id, strategy in distribution_tasks:
        status = distribution_service.get_distribution_status(task_id)
        if status:
            progress_avg = sum(status['progress'].values()) / len(status['progress']) if status['progress'] else 0
            print(f"   • {model_id} ({strategy.value}): {status['status']} - {progress_avg:.1f}% completado")
            if status['errors']:
                print(f"     ❌ Errores: {len(status['errors'])}")

    # Mostrar versiones de modelos
    print("\n📋 VERSIONES DE MODELOS:")
    for model_id in test_models:
        versions = distribution_service.get_model_versions(model_id)
        print(f"   • {model_id}: {len(versions)} versiones")
        for version in versions[-2:]:  # Mostrar últimas 2 versiones
            print(f"     - v{version['version_number']} ({time.strftime('%H:%M:%S', time.localtime(version['created_at']))})")

    # Simular creación de nuevas versiones
    print("\n🔄 Creando nuevas versiones...")
    for model_id in test_models[:2]:  # Solo para algunos modelos
        new_version = await distribution_service.create_model_version(
            model_id=model_id,
            changes_description="Mejora en accuracy"
        )
        if new_version:
            print(f"✅ Nueva versión creada: {model_id} -> {new_version}")

    print("\n📈 ESTADÍSTICAS FINALES:")
    total_tasks = len(distribution_service.distribution_tasks)
    completed_tasks = sum(1 for t in distribution_service.distribution_tasks.values()
                         if t.status == DistributionStatus.COMPLETED)
    total_bandwidth = sum(t.bandwidth_used for t in distribution_service.distribution_tasks.values())

    print(f"   • Tareas de distribución: {total_tasks}")
    print(f"   • Tareas completadas: {completed_tasks}")
    print(f"   • Bandwidth total usado: {total_bandwidth:,} bytes")
    print(f"   • Modelos versionados: {len(distribution_service.model_versions)}")

    print("\n🎉 Demo de distribución de modelos completada exitosamente!")

if __name__ == "__main__":
    asyncio.run(demo_model_distribution())
