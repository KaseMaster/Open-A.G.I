#!/usr/bin/env python3
"""
🗄️ SISTEMA DE PERSISTENCIA DE ESTADO DISTRIBUIDO - AEGIS Framework
Módulo para replicación y recuperación de estado de consenso entre nodos.

Características principales:
- Replicación de estado de consenso entre nodos
- Checkpointing distribuido para recuperación de fallos
- Recuperación automática de estado
- Consistencia eventual del estado global
- Compresión de datos para eficiencia
"""

import asyncio
import json
import time
import hashlib
import secrets
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import logging

# Dependencias opcionales
try:
    import aiofiles  # type: ignore
    HAS_AIOFILES = True
except Exception:
    aiofiles = None  # type: ignore
    HAS_AIOFILES = False

try:
    import lz4.frame as lz4  # type: ignore
    HAS_LZ4 = True
except Exception:
    lz4 = None  # type: ignore
    HAS_LZ4 = False

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StateType(Enum):
    """Tipos de estado que se pueden persistir"""
    CONSENSUS_STATE = "consensus_state"
    NODE_REPUTATION = "node_reputation"
    NETWORK_TOPOLOGY = "network_topology"
    KNOWLEDGE_BASE = "knowledge_base"
    BLOCKCHAIN_STATE = "blockchain_state"

class ReplicationStatus(Enum):
    """Estados de replicación"""
    PENDING = "pending"
    REPLICATING = "replicating"
    COMPLETED = "completed"
    FAILED = "failed"
    CONFLICT = "conflict"

@dataclass
class StateChunk:
    """Fragmento de estado replicable"""
    chunk_id: str
    state_type: StateType
    node_id: str
    sequence_number: int
    data: Dict[str, Any]
    checksum: str
    timestamp: float
    compressed: bool = False
    size_bytes: int = 0

    def __post_init__(self):
        """Calcular tamaño y checksum si no están definidos"""
        if self.size_bytes == 0:
            self.size_bytes = len(json.dumps(self.data).encode())

        if not self.checksum:
            data_str = json.dumps(self.data, sort_keys=True)
            self.checksum = hashlib.sha256(data_str.encode()).hexdigest()

@dataclass
class StateCheckpoint:
    """Checkpoint completo del estado del sistema"""
    checkpoint_id: str
    node_id: str
    timestamp: float
    sequence_number: int
    state_chunks: Dict[str, StateChunk] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    status: ReplicationStatus = ReplicationStatus.PENDING

    def add_chunk(self, chunk: StateChunk) -> None:
        """Agrega un chunk al checkpoint"""
        self.state_chunks[chunk.chunk_id] = chunk

    def get_total_size(self) -> int:
        """Calcula el tamaño total del checkpoint"""
        return sum(chunk.size_bytes for chunk in self.state_chunks.values())

    def is_complete(self) -> bool:
        """Verifica si el checkpoint está completo"""
        return len(self.state_chunks) > 0 and self.status == ReplicationStatus.COMPLETED

@dataclass
class ReplicationRequest:
    """Solicitud de replicación de estado"""
    request_id: str
    source_node: str
    target_nodes: List[str]
    checkpoint_id: str
    priority: int = 1  # 1-10, 10 es máxima prioridad
    deadline: Optional[float] = None
    status: ReplicationStatus = ReplicationStatus.PENDING

class DistributedStateManager:
    """Gestor principal de persistencia de estado distribuido"""

    def __init__(self, node_id: str, storage_path: str = "./state_storage"):
        self.node_id = node_id
        self.storage_path = storage_path
        self.checkpoints: Dict[str, StateCheckpoint] = {}
        self.active_replications: Dict[str, ReplicationRequest] = {}
        self.state_versions: Dict[str, int] = {}  # state_type -> version

        # Configuración
        self.max_chunk_size = 1024 * 1024  # 1MB por chunk
        self.replication_factor = 3  # Número mínimo de réplicas
        self.checkpoint_interval = 300  # 5 minutos entre checkpoints
        self.cleanup_interval = 3600  # 1 hora para limpieza

        # Estadísticas
        self.stats = {
            "checkpoints_created": 0,
            "chunks_replicated": 0,
            "recovery_operations": 0,
            "conflicts_resolved": 0,
            "data_compressed": 0
        }

        logger.info(f"🗄️ DistributedStateManager inicializado para nodo {node_id}")

    async def create_checkpoint(self, state_type: StateType, state_data: Dict[str, Any]) -> StateCheckpoint:
        """Crea un checkpoint del estado actual"""
        try:
            checkpoint_id = f"{state_type.value}_{self.node_id}_{int(time.time())}"
            sequence_number = self.state_versions.get(state_type.value, 0) + 1

            checkpoint = StateCheckpoint(
                checkpoint_id=checkpoint_id,
                node_id=self.node_id,
                timestamp=time.time(),
                sequence_number=sequence_number,
                metadata={
                    "state_type": state_type.value,
                    "compression_enabled": HAS_LZ4,
                    "node_version": "1.0.0"
                }
            )

            # Dividir datos en chunks si es necesario
            chunks = await self._split_state_into_chunks(state_type, state_data, sequence_number)
            for chunk in chunks:
                checkpoint.add_chunk(chunk)

            # Guardar checkpoint localmente
            await self._save_checkpoint(checkpoint)

            # Actualizar versión del estado
            self.state_versions[state_type.value] = sequence_number
            self.checkpoints[checkpoint_id] = checkpoint
            self.stats["checkpoints_created"] += 1

            logger.info(f"✅ Checkpoint creado: {checkpoint_id} ({len(chunks)} chunks)")
            return checkpoint

        except Exception as e:
            logger.error(f"❌ Error creando checkpoint: {e}")
            raise

    async def replicate_checkpoint(self, checkpoint: StateCheckpoint, target_nodes: List[str]) -> bool:
        """Replica un checkpoint a nodos objetivo"""
        try:
            request_id = secrets.token_hex(16)
            replication_request = ReplicationRequest(
                request_id=request_id,
                source_node=self.node_id,
                target_nodes=target_nodes,
                checkpoint_id=checkpoint.checkpoint_id,
                priority=5  # Prioridad media
            )

            self.active_replications[request_id] = replication_request

            # Iniciar replicación paralela
            tasks = []
            for target_node in target_nodes[:self.replication_factor]:
                task = asyncio.create_task(
                    self._replicate_to_node(checkpoint, target_node, request_id)
                )
                tasks.append(task)

            # Esperar a que al menos N réplicas se completen
            results = await asyncio.gather(*tasks, return_exceptions=True)
            successful_replicas = sum(1 for r in results if not isinstance(r, Exception))

            if successful_replicas >= self.replication_factor:
                checkpoint.status = ReplicationStatus.COMPLETED
                logger.info(f"✅ Checkpoint replicado exitosamente: {checkpoint.checkpoint_id}")
                return True
            else:
                checkpoint.status = ReplicationStatus.FAILED
                logger.warning(f"⚠️ Replicación fallida para {checkpoint.checkpoint_id}")
                return False

        except Exception as e:
            logger.error(f"❌ Error replicando checkpoint: {e}")
            return False
        finally:
            # Limpiar solicitud de replicación
            if request_id in self.active_replications:
                del self.active_replications[request_id]

    async def recover_state(self, state_type: StateType, target_version: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """Recupera estado desde checkpoints replicados"""
        try:
            # Buscar el checkpoint más reciente para el tipo de estado
            candidates = [
                cp for cp in self.checkpoints.values()
                if cp.metadata.get("state_type") == state_type.value
            ]

            if not candidates:
                logger.warning(f"⚠️ No hay checkpoints disponibles para {state_type.value}")
                return None

            # Seleccionar checkpoint (más reciente o versión específica)
            if target_version:
                checkpoint = next((cp for cp in candidates if cp.sequence_number == target_version), None)
            else:
                checkpoint = max(candidates, key=lambda cp: cp.sequence_number)

            if not checkpoint:
                logger.warning(f"⚠️ Checkpoint no encontrado para {state_type.value} v{target_version}")
                return None

            # Reconstruir estado desde chunks
            state_data = await self._reconstruct_state_from_chunks(checkpoint)
            self.stats["recovery_operations"] += 1

            logger.info(f"🔄 Estado recuperado: {state_type.value} v{checkpoint.sequence_number}")
            return state_data

        except Exception as e:
            logger.error(f"❌ Error recuperando estado: {e}")
            return None

    async def _split_state_into_chunks(self, state_type: StateType, state_data: Dict[str, Any],
                                     sequence_number: int) -> List[StateChunk]:
        """Divide el estado en chunks replicables"""
        chunks = []
        data_str = json.dumps(state_data, sort_keys=True)
        data_bytes = data_str.encode()

        # Comprimir si está disponible
        if HAS_LZ4:
            compressed_data = lz4.compress(data_bytes)
            if len(compressed_data) < len(data_bytes):
                data_bytes = compressed_data
                compressed = True
                self.stats["data_compressed"] += 1
            else:
                compressed = False
        else:
            compressed = False

        # Crear chunk único por ahora (implementación simplificada)
        # En producción, dividir en chunks más pequeños
        chunk_id = f"{state_type.value}_{sequence_number}_{secrets.token_hex(8)}"
        chunk = StateChunk(
            chunk_id=chunk_id,
            state_type=state_type,
            node_id=self.node_id,
            sequence_number=sequence_number,
            data=state_data,
            checksum="",
            timestamp=time.time(),
            compressed=compressed,
            size_bytes=len(data_bytes)
        )

        # Calcular checksum
        chunk.checksum = hashlib.sha256(data_str.encode()).hexdigest()
        chunks.append(chunk)

        return chunks

    async def _save_checkpoint(self, checkpoint: StateCheckpoint) -> None:
        """Guarda checkpoint localmente"""
        if not HAS_AIOFILES:
            logger.warning("⚠️ aiofiles no disponible; checkpoint no guardado localmente")
            return

        try:
            checkpoint_file = f"{self.storage_path}/{checkpoint.checkpoint_id}.json"

            # Crear directorio si no existe
            import os
            os.makedirs(self.storage_path, exist_ok=True)

            # Serializar checkpoint
            checkpoint_data = {
                "checkpoint_id": checkpoint.checkpoint_id,
                "node_id": checkpoint.node_id,
                "timestamp": checkpoint.timestamp,
                "sequence_number": checkpoint.sequence_number,
                "metadata": checkpoint.metadata,
                "chunks": {
                    chunk_id: {
                        "chunk_id": chunk.chunk_id,
                        "state_type": chunk.state_type.value,
                        "node_id": chunk.node_id,
                        "sequence_number": chunk.sequence_number,
                        "data": chunk.data,
                        "checksum": chunk.checksum,
                        "timestamp": chunk.timestamp,
                        "compressed": chunk.compressed,
                        "size_bytes": chunk.size_bytes
                    }
                    for chunk_id, chunk in checkpoint.state_chunks.items()
                }
            }

            async with aiofiles.open(checkpoint_file, 'w') as f:
                await f.write(json.dumps(checkpoint_data, indent=2, default=str))

            logger.debug(f"💾 Checkpoint guardado: {checkpoint_file}")

        except Exception as e:
            logger.error(f"❌ Error guardando checkpoint: {e}")

    async def _replicate_to_node(self, checkpoint: StateCheckpoint, target_node: str, request_id: str) -> bool:
        """Replica checkpoint a un nodo específico"""
        try:
            # En implementación real, esto enviaría el checkpoint vía P2P
            # Por ahora, simulamos replicación exitosa
            logger.debug(f"📤 Replicando {checkpoint.checkpoint_id} a {target_node}")

            # Simular latencia de red
            await asyncio.sleep(0.1)

            # Marcar como replicado exitosamente
            self.stats["chunks_replicated"] += len(checkpoint.state_chunks)
            return True

        except Exception as e:
            logger.error(f"❌ Error replicando a {target_node}: {e}")
            return False

    async def _reconstruct_state_from_chunks(self, checkpoint: StateCheckpoint) -> Dict[str, Any]:
        """Reconstruye estado desde chunks"""
        # Para implementación simplificada, tomar el primer chunk
        if not checkpoint.state_chunks:
            raise ValueError("Checkpoint no tiene chunks")

        chunk = next(iter(checkpoint.state_chunks.values()))

        # Verificar integridad
        data_str = json.dumps(chunk.data, sort_keys=True)
        calculated_checksum = hashlib.sha256(data_str.encode()).hexdigest()

        if calculated_checksum != chunk.checksum:
            raise ValueError(f"Checksum inválido para chunk {chunk.chunk_id}")

        # Descomprimir si es necesario
        if chunk.compressed and HAS_LZ4:
            # En implementación real, descomprimir datos
            pass

        return chunk.data

    def get_state_info(self, state_type: StateType) -> Dict[str, Any]:
        """Obtiene información sobre el estado actual"""
        checkpoints = [
            cp for cp in self.checkpoints.values()
            if cp.metadata.get("state_type") == state_type.value
        ]

        if not checkpoints:
            return {"status": "no_checkpoints"}

        latest_checkpoint = max(checkpoints, key=lambda cp: cp.sequence_number)

        return {
            "state_type": state_type.value,
            "current_version": self.state_versions.get(state_type.value, 0),
            "latest_checkpoint": latest_checkpoint.checkpoint_id,
            "checkpoint_count": len(checkpoints),
            "total_size_bytes": sum(cp.get_total_size() for cp in checkpoints),
            "last_checkpoint_time": latest_checkpoint.timestamp
        }

    async def cleanup_old_checkpoints(self, max_age_seconds: int = 86400 * 7) -> int:
        """Limpia checkpoints antiguos"""
        current_time = time.time()
        to_remove = []

        for checkpoint_id, checkpoint in self.checkpoints.items():
            if current_time - checkpoint.timestamp > max_age_seconds:
                to_remove.append(checkpoint_id)

        for checkpoint_id in to_remove:
            del self.checkpoints[checkpoint_id]

        if to_remove:
            logger.info(f"🧹 {len(to_remove)} checkpoints antiguos limpiados")

        return len(to_remove)

    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del sistema de persistencia"""
        return {
            "node_id": self.node_id,
            "total_checkpoints": len(self.checkpoints),
            "active_replications": len(self.active_replications),
            "state_types_tracked": len(self.state_versions),
            **self.stats
        }

async def demo_distributed_state_persistence():
    """Demostración del sistema de persistencia de estado distribuido"""
    print("🗄️ DEMO - SISTEMA DE PERSISTENCIA DE ESTADO DISTRIBUIDO")
    print("=" * 60)

    # Inicializar gestor
    state_manager = DistributedStateManager("demo_node_1")

    try:
        # 1. Crear checkpoint de estado de consenso simulado
        print("\n📦 Creando checkpoint de estado de consenso...")
        consensus_state = {
            "view_number": 5,
            "sequence_number": 42,
            "leader_node": "node_leader",
            "active_nodes": ["node_1", "node_2", "node_3", "node_4"],
            "last_commit_timestamp": time.time(),
            "pending_changes": [
                {"type": "knowledge_update", "id": "change_123"},
                {"type": "reputation_update", "id": "change_124"}
            ]
        }

        checkpoint = await state_manager.create_checkpoint(
            StateType.CONSENSUS_STATE,
            consensus_state
        )
        print(f"✅ Checkpoint creado: {checkpoint.checkpoint_id}")

        # 2. Simular replicación
        print("\n📤 Simulando replicación a otros nodos...")
        target_nodes = ["node_2", "node_3", "node_4"]
        success = await state_manager.replicate_checkpoint(checkpoint, target_nodes)
        print(f"✅ Replicación: {'Exitosa' if success else 'Fallida'}")

        # 3. Simular recuperación de estado
        print("\n🔄 Simulando recuperación de estado...")
        recovered_state = await state_manager.recover_state(StateType.CONSENSUS_STATE)

        if recovered_state:
            print("✅ Estado recuperado exitosamente")
            print(f"   View number: {recovered_state['view_number']}")
            print(f"   Sequence number: {recovered_state['sequence_number']}")
            print(f"   Active nodes: {len(recovered_state['active_nodes'])}")
        else:
            print("❌ Error recuperando estado")

        # 4. Mostrar estadísticas
        print("\n📊 Estadísticas del sistema:")
        stats = state_manager.get_stats()
        print(f"   Checkpoints totales: {stats['total_checkpoints']}")
        print(f"   Chunks replicados: {stats['chunks_replicated']}")
        print(f"   Operaciones de recuperación: {stats['recovery_operations']}")
        print(f"   Datos comprimidos: {stats['data_compressed']}")

        # 5. Información de estado
        print("\n📋 Información de estado de consenso:")
        state_info = state_manager.get_state_info(StateType.CONSENSUS_STATE)
        for key, value in state_info.items():
            print(f"   {key}: {value}")

        print("\n🎉 Demo completada exitosamente!")

    except Exception as e:
        print(f"❌ Error en demo: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(demo_distributed_state_persistence())
