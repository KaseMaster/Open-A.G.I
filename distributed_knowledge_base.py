#!/usr/bin/env python3
"""
Knowledge Base Distribuida - AEGIS Framework
Sistema de almacenamiento y versionado distribuido para conocimiento colaborativo.

Características:
- Almacenamiento direccionado por contenido (IPFS-like)
- Versionado ligero con merge automático
- Sincronización P2P con Merkle trees
- Compresión y deduplicación automática
- API REST para integración

Programador Principal: Jose Gómez alias KaseMaster
Contacto: kasemaster@aegis-framework.com
Versión: 2.0.0
Licencia: MIT
"""

import asyncio
import json
import hashlib
import time
import logging
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field, asdict
from enum import Enum
import sqlite3
import threading
from datetime import datetime, timezone
import os
import tempfile
import shutil

# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class KnowledgeFormat(Enum):
    """Formatos de conocimiento soportados"""
    JSON = "json"
    TEXT = "text"
    BINARY = "binary"
    ML_MODEL = "ml_model"
    CONSENSUS_PROPOSAL = "consensus_proposal"


class SyncStatus(Enum):
    """Estados de sincronización"""
    LOCAL = "local"
    SYNCING = "syncing"
    SYNCED = "synced"
    CONFLICT = "conflict"
    DELETED = "deleted"


@dataclass
class KnowledgeEntry:
    """Entrada de conocimiento en la base distribuida"""
    content_id: str  # Hash del contenido
    content: Any  # Contenido real
    format: KnowledgeFormat
    metadata: Dict[str, Any]
    version: int
    parent_versions: List[str]  # Versiones padre para merge
    author: str
    timestamp: float
    signature: str  # Firma del autor
    sync_status: SyncStatus = SyncStatus.LOCAL
    last_sync: float = 0.0
    size_bytes: int = 0

    def __post_init__(self):
        """Calcular tamaño si no se proporciona"""
        if self.size_bytes == 0:
            if isinstance(self.content, str):
                self.size_bytes = len(self.content.encode('utf-8'))
            elif isinstance(self.content, (dict, list)):
                self.size_bytes = len(json.dumps(self.content).encode('utf-8'))
            else:
                self.size_bytes = len(str(self.content).encode('utf-8'))


@dataclass
class KnowledgeBranch:
    """Rama de conocimiento (similar a Git)"""
    branch_id: str
    name: str
    head_version: str  # Content_id de la versión actual
    created_at: float
    created_by: str
    is_default: bool = False
    merge_base: Optional[str] = None  # Para merges


@dataclass
class KnowledgeSync:
    """Estado de sincronización con peers"""
    peer_id: str
    last_sync_time: float
    known_versions: Set[str]
    missing_versions: Set[str]
    conflicts: List[str]
    sync_in_progress: bool = False


class MerkleTree:
    """Árbol de Merkle para verificación de integridad"""

    def __init__(self, entries: List[KnowledgeEntry]):
        self.entries = entries
        self.root_hash = self._build_tree()

    def _build_tree(self) -> str:
        """Construye árbol de Merkle y retorna hash raíz"""
        if not self.entries:
            return hashlib.sha256(b"empty").hexdigest()

        # Crear hashes de hojas
        hashes = []
        for entry in self.entries:
            entry_hash = hashlib.sha256(
                f"{entry.content_id}:{entry.version}:{entry.author}:{entry.timestamp}".encode()
            ).hexdigest()
            hashes.append(entry_hash)

        # Construir árbol hacia arriba
        while len(hashes) > 1:
            new_hashes = []
            for i in range(0, len(hashes), 2):
                if i + 1 < len(hashes):
                    combined = hashes[i] + hashes[i + 1]
                else:
                    combined = hashes[i] + hashes[i]  # Duplicar si es impar
                new_hashes.append(hashlib.sha256(combined.encode()).hexdigest())
            hashes = new_hashes

        return hashes[0] if hashes else hashlib.sha256(b"empty").hexdigest()

    def get_proof(self, content_id: str) -> List[str]:
        """Genera prueba de Merkle para una entrada específica"""
        entry_index = None
        for i, entry in enumerate(self.entries):
            if entry.content_id == content_id:
                entry_index = i
                break

        if entry_index is None:
            return []

        # Implementación simplificada de prueba de Merkle
        return [self.entries[entry_index].content_id, self.root_hash]


class DistributedKnowledgeBase:
    """Base de conocimiento distribuida principal"""

    def __init__(self, node_id: str, storage_path: str = None):
        self.node_id = node_id
        self.storage_path = storage_path or os.path.join(tempfile.gettempdir(), f"aegis_kb_{node_id}")

        # Crear directorio de almacenamiento
        os.makedirs(self.storage_path, exist_ok=True)

        # Base de datos local
        self.db_path = os.path.join(self.storage_path, "knowledge.db")
        self._init_database()

        # Estado de la base de conocimiento
        self.entries: Dict[str, KnowledgeEntry] = {}  # content_id -> entry
        self.branches: Dict[str, KnowledgeBranch] = {}  # branch_id -> branch
        self.sync_states: Dict[str, KnowledgeSync] = {}  # peer_id -> sync_state
        self.current_branch: str = "main"  # Rama actual

        # Configuración
        self.compression_enabled = True
        self.max_versions_per_content = 100
        self.sync_interval = 300  # 5 minutos

        # Crear rama por defecto
        self._create_default_branch()

    def _init_database(self):
        """Inicializa base de datos SQLite local"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Tabla de entradas
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS knowledge_entries (
                content_id TEXT PRIMARY KEY,
                content TEXT,
                format TEXT,
                metadata TEXT,
                version INTEGER,
                parent_versions TEXT,
                author TEXT,
                timestamp REAL,
                signature TEXT,
                sync_status TEXT,
                last_sync REAL,
                size_bytes INTEGER
            )
        ''')

        # Tabla de ramas
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS branches (
                branch_id TEXT PRIMARY KEY,
                name TEXT,
                head_version TEXT,
                created_at REAL,
                created_by TEXT,
                is_default BOOLEAN,
                merge_base TEXT
            )
        ''')

        # Tabla de sincronización
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sync_states (
                peer_id TEXT PRIMARY KEY,
                last_sync_time REAL,
                known_versions TEXT,
                missing_versions TEXT,
                conflicts TEXT,
                sync_in_progress BOOLEAN
            )
        ''')

        conn.commit()
        conn.close()

    def _create_default_branch(self):
        """Crea rama por defecto (main)"""
        branch = KnowledgeBranch(
            branch_id="main",
            name="main",
            head_version="",
            created_at=time.time(),
            created_by=self.node_id,
            is_default=True
        )

        self.branches["main"] = branch
        self._save_branch(branch)

    def _save_entry(self, entry: KnowledgeEntry):
        """Guarda entrada en base de datos"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT OR REPLACE INTO knowledge_entries
            (content_id, content, format, metadata, version, parent_versions,
             author, timestamp, signature, sync_status, last_sync, size_bytes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            entry.content_id,
            json.dumps(entry.content) if isinstance(entry.content, (dict, list)) else str(entry.content),
            entry.format.value,
            json.dumps(entry.metadata),
            entry.version,
            json.dumps(entry.parent_versions),
            entry.author,
            entry.timestamp,
            entry.signature,
            entry.sync_status.value,
            entry.last_sync,
            entry.size_bytes
        ))

        conn.commit()
        conn.close()

    def _save_branch(self, branch: KnowledgeBranch):
        """Guarda rama en base de datos"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT OR REPLACE INTO branches
            (branch_id, name, head_version, created_at, created_by, is_default, merge_base)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            branch.branch_id,
            branch.name,
            branch.head_version,
            branch.created_at,
            branch.created_by,
            branch.is_default,
            branch.merge_base
        ))

        conn.commit()
        conn.close()

    def _load_entries(self) -> Dict[str, KnowledgeEntry]:
        """Carga entradas desde base de datos"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('SELECT * FROM knowledge_entries')
        rows = cursor.fetchall()

        entries = {}
        for row in rows:
            (
                content_id, content_str, format_str, metadata_str, version,
                parent_versions_str, author, timestamp, signature,
                sync_status_str, last_sync, size_bytes
            ) = row

            # Parsear contenido
            try:
                if format_str in ['json', 'ml_model', 'consensus_proposal']:
                    content = json.loads(content_str)
                else:
                    content = content_str
            except:
                content = content_str

            # Parsear metadata y parent_versions
            metadata = json.loads(metadata_str) if metadata_str else {}
            parent_versions = json.loads(parent_versions_str) if parent_versions_str else []

            entry = KnowledgeEntry(
                content_id=content_id,
                content=content,
                format=KnowledgeFormat(format_str),
                metadata=metadata,
                version=version,
                parent_versions=parent_versions,
                author=author,
                timestamp=timestamp,
                signature=signature,
                sync_status=SyncStatus(sync_status_str),
                last_sync=last_sync,
                size_bytes=size_bytes
            )

            entries[content_id] = entry

        conn.close()
        return entries

    def _load_branches(self) -> Dict[str, KnowledgeBranch]:
        """Carga ramas desde base de datos"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('SELECT * FROM branches')
        rows = cursor.fetchall()

        branches = {}
        for row in rows:
            (branch_id, name, head_version, created_at, created_by, is_default, merge_base) = row

            branch = KnowledgeBranch(
                branch_id=branch_id,
                name=name,
                head_version=head_version,
                created_at=created_at,
                created_by=created_by,
                is_default=bool(is_default),
                merge_base=merge_base
            )

            branches[branch_id] = branch

        conn.close()
        return branches

    def load_state(self):
        """Carga estado completo desde base de datos"""
        self.entries = self._load_entries()
        self.branches = self._load_branches()

        # Configurar rama actual (main por defecto)
        if "main" in self.branches:
            self.current_branch = "main"
        else:
            self.current_branch = list(self.branches.keys())[0] if self.branches else None

        logger.info(f"📚 Estado cargado: {len(self.entries)} entradas, {len(self.branches)} ramas")

    def save_state(self):
        """Guarda estado completo en base de datos"""
        for entry in self.entries.values():
            self._save_entry(entry)

        for branch in self.branches.values():
            self._save_branch(branch)

    def add_knowledge(self, content: Any, format: KnowledgeFormat,
                     metadata: Dict[str, Any] = None, parent_versions: List[str] = None) -> str:
        """Agrega nuevo conocimiento a la base"""
        metadata = metadata or {}
        parent_versions = parent_versions or []

        # Determinar versiones padre
        if not parent_versions and self.current_branch:
            current_head = self.branches[self.current_branch].head_version
            if current_head in self.entries:
                parent_versions = [current_head]

        # Crear hash del contenido
        if isinstance(content, str):
            content_bytes = content.encode('utf-8')
        elif isinstance(content, (dict, list)):
            content_bytes = json.dumps(content, sort_keys=True).encode('utf-8')
        else:
            content_bytes = str(content).encode('utf-8')

        content_id = hashlib.sha256(content_bytes).hexdigest()

        # Verificar si el contenido ya existe
        if content_id in self.entries:
            existing_entry = self.entries[content_id]
            # Crear nueva versión si hay cambios
            if existing_entry.version < self.max_versions_per_content:
                new_version = existing_entry.version + 1
                content_id = f"{content_id}_v{new_version}"
            else:
                # Reutilizar versión más antigua
                oldest_version = min([
                    e.content_id for e in self.entries.values()
                    if e.content_id.startswith(content_id.split('_v')[0])
                ])
                content_id = oldest_version

        # Crear entrada de conocimiento
        entry = KnowledgeEntry(
            content_id=content_id,
            content=content,
            format=format,
            metadata=metadata,
            version=1,
            parent_versions=parent_versions,
            author=self.node_id,
            timestamp=time.time(),
            signature="",  # Se firmará después
            sync_status=SyncStatus.LOCAL
        )

        # Firmar entrada (simplificado)
        entry.signature = hashlib.sha256(
            f"{content_id}:{entry.author}:{entry.timestamp}".encode()
        ).hexdigest()

        # Guardar entrada
        self.entries[content_id] = entry
        self._save_entry(entry)

        # Actualizar rama actual
        if self.current_branch:
            self.branches[self.current_branch].head_version = content_id
            self._save_branch(self.branches[self.current_branch])

        logger.info(f"➕ Conocimiento agregado: {content_id[:16]}...")
        return content_id

    def get_knowledge(self, content_id: str) -> Optional[KnowledgeEntry]:
        """Obtiene entrada de conocimiento por ID"""
        return self.entries.get(content_id)

    def search_knowledge(self, query: str, format: KnowledgeFormat = None,
                        author: str = None, limit: int = 50) -> List[KnowledgeEntry]:
        """Busca conocimiento por criterios"""
        results = []

        for entry in self.entries.values():
            # Filtrar por formato
            if format and entry.format != format:
                continue

            # Filtrar por autor
            if author and entry.author != author:
                continue

            # Buscar en contenido y metadata
            search_text = ""
            if isinstance(entry.content, str):
                search_text = entry.content.lower()
            elif isinstance(entry.content, dict):
                search_text = json.dumps(entry.content).lower()

            search_text += " " + " ".join(entry.metadata.values()).lower()

            if query.lower() in search_text:
                results.append(entry)

        # Ordenar por timestamp descendente
        results.sort(key=lambda x: x.timestamp, reverse=True)
        return results[:limit]

    def get_branch_history(self, branch_id: str = None) -> List[KnowledgeEntry]:
        """Obtiene historial de versiones de una rama"""
        branch_id = branch_id or self.current_branch
        if branch_id not in self.branches:
            return []

        branch = self.branches[branch_id]
        history = []
        current_id = branch.head_version

        # Seguir cadena de versiones hacia atrás
        while current_id and current_id in self.entries:
            entry = self.entries[current_id]
            history.append(entry)

            # Encontrar versión padre
            if entry.parent_versions:
                current_id = entry.parent_versions[0]
            else:
                break

        return history

    def create_branch(self, name: str, from_branch: str = None) -> str:
        """Crea nueva rama de conocimiento"""
        from_branch = from_branch or self.current_branch
        if from_branch not in self.branches:
            raise ValueError(f"Rama {from_branch} no existe")

        branch_id = hashlib.sha256(f"{name}:{time.time()}".encode()).hexdigest()[:16]

        branch = KnowledgeBranch(
            branch_id=branch_id,
            name=name,
            head_version=self.branches[from_branch].head_version,
            created_at=time.time(),
            created_by=self.node_id,
            is_default=False
        )

        self.branches[branch_id] = branch
        self._save_branch(branch)

        logger.info(f"🌿 Rama creada: {name} ({branch_id})")
        return branch_id

    def merge_branches(self, source_branch: str, target_branch: str = None) -> Dict[str, Any]:
        """Fusiona dos ramas de conocimiento"""
        target_branch = target_branch or self.current_branch
        if source_branch not in self.branches or target_branch not in self.branches:
            return {"success": False, "error": "Rama no existe"}

        source = self.branches[source_branch]
        target = self.branches[target_branch]

        # Encontrar base común
        base_versions = set()
        current = target.head_version
        while current and current in self.entries:
            base_versions.add(current)
            entry = self.entries[current]
            if entry.parent_versions:
                current = entry.parent_versions[0]
            else:
                break

        # Encontrar cambios en source que no están en base
        changes_to_merge = []
        current = source.head_version
        while current and current in self.entries:
            if current not in base_versions:
                changes_to_merge.append(current)
            entry = self.entries[current]
            if entry.parent_versions:
                current = entry.parent_versions[0]
            else:
                break

        # Aplicar merge (simplificado)
        for change_id in changes_to_merge:
            change_entry = self.entries[change_id]

            # Crear nueva versión en target
            new_content_id = self.add_knowledge(
                content=change_entry.content,
                format=change_entry.format,
                metadata={**change_entry.metadata, "merged_from": source_branch},
                parent_versions=[target.head_version]
            )

            # Actualizar head de target
            target.head_version = new_content_id

        self._save_branch(target)

        return {
            "success": True,
            "merged_changes": len(changes_to_merge),
            "new_head": target.head_version
        }

    def get_sync_status(self, peer_id: str) -> Dict[str, Any]:
        """Obtiene estado de sincronización con peer"""
        if peer_id not in self.sync_states:
            self.sync_states[peer_id] = KnowledgeSync(
                peer_id=peer_id,
                last_sync_time=0.0,
                known_versions=set(),
                missing_versions=set(),
                conflicts=[]
            )

        sync_state = self.sync_states[peer_id]

        return {
            "peer_id": peer_id,
            "last_sync": sync_state.last_sync_time,
            "local_versions": len(self.entries),
            "known_by_peer": len(sync_state.known_versions),
            "missing_by_peer": len(sync_state.missing_versions),
            "conflicts": len(sync_state.conflicts),
            "sync_in_progress": sync_state.sync_in_progress
        }

    def sync_with_peer(self, peer_id: str, peer_entries: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Sincroniza conocimiento con peer"""
        sync_state = self.sync_states.get(peer_id)
        if not sync_state:
            sync_state = KnowledgeSync(
                peer_id=peer_id,
                last_sync_time=0.0,
                known_versions=set(),
                missing_versions=set(),
                conflicts=[]
            )
            self.sync_states[peer_id] = sync_state

        sync_state.sync_in_progress = True

        try:
            # Convertir entradas de peer a objetos KnowledgeEntry
            peer_entry_objects = []
            for entry_data in peer_entries:
                entry = KnowledgeEntry(
                    content_id=entry_data["content_id"],
                    content=entry_data["content"],
                    format=KnowledgeFormat(entry_data["format"]),
                    metadata=entry_data["metadata"],
                    version=entry_data["version"],
                    parent_versions=entry_data["parent_versions"],
                    author=entry_data["author"],
                    timestamp=entry_data["timestamp"],
                    signature=entry_data["signature"],
                    sync_status=SyncStatus.SYNCED,
                    last_sync=time.time()
                )
                peer_entry_objects.append(entry)

            # Encontrar versiones faltantes
            peer_version_ids = {e.content_id for e in peer_entry_objects}
            local_version_ids = set(self.entries.keys())

            missing_versions = peer_version_ids - local_version_ids
            new_versions = local_version_ids - peer_version_ids

            # Agregar versiones faltantes
            for entry in peer_entry_objects:
                if entry.content_id in missing_versions:
                    self.entries[entry.content_id] = entry
                    self._save_entry(entry)

            # Resolver conflictos (simplificado)
            conflicts_resolved = 0
            for entry in peer_entry_objects:
                if entry.content_id in self.entries:
                    local_entry = self.entries[entry.content_id]

                    # Si hay diferencia en contenido pero mismo hash, hay conflicto
                    if (local_entry.content != entry.content and
                        local_entry.version == entry.version):
                        # Resolver automáticamente: mantener versión más reciente
                        if entry.timestamp > local_entry.timestamp:
                            self.entries[entry.content_id] = entry
                            conflicts_resolved += 1

            # Actualizar estado de sincronización
            sync_state.last_sync_time = time.time()
            sync_state.known_versions = peer_version_ids
            sync_state.missing_versions = missing_versions
            sync_state.conflicts = []  # Limpiar conflictos resueltos

            result = {
                "success": True,
                "local_versions": len(local_version_ids),
                "peer_versions": len(peer_version_ids),
                "missing_added": len(missing_versions),
                "new_for_peer": len(new_versions),
                "conflicts_resolved": conflicts_resolved,
                "sync_time": sync_state.last_sync_time
            }

            logger.info(f"🔄 Sincronización completada con {peer_id}: {result}")
            return result

        finally:
            sync_state.sync_in_progress = False

    def get_api_status(self) -> Dict[str, Any]:
        """Obtiene estado para API REST"""
        return {
            "node_id": self.node_id,
            "current_branch": self.current_branch,
            "total_entries": len(self.entries),
            "total_branches": len(self.branches),
            "storage_path": self.storage_path,
            "db_size_bytes": os.path.getsize(self.db_path) if os.path.exists(self.db_path) else 0,
            "oldest_entry": min([e.timestamp for e in self.entries.values()]) if self.entries else 0,
            "newest_entry": max([e.timestamp for e in self.entries.values()]) if self.entries else 0,
            "formats_used": list(set([e.format.value for e in self.entries.values()])),
            "sync_peers": len(self.sync_states)
        }

    def cleanup_old_versions(self):
        """Limpia versiones antiguas manteniendo historia"""
        try:
            # Mantener solo las últimas N versiones por contenido
            content_groups = {}
            for entry in self.entries.values():
                base_id = entry.content_id.split('_v')[0]
                if base_id not in content_groups:
                    content_groups[base_id] = []
                content_groups[base_id].append(entry)

            # Para cada grupo, mantener solo las versiones más recientes
            for base_id, entries in content_groups.items():
                entries.sort(key=lambda x: x.version, reverse=True)

                # Mantener las últimas max_versions_per_content versiones
                for entry in entries[self.max_versions_per_content:]:
                    # No eliminar completamente, marcar como archivado
                    entry.sync_status = SyncStatus.DELETED
                    self._save_entry(entry)

            logger.info(f"🧹 Limpieza completada: {len(content_groups)} grupos procesados")

        except Exception as e:
            logger.error(f"❌ Error en limpieza: {e}")


# Funciones de utilidad para integración
def create_knowledge_base(node_id: str, storage_path: str = None) -> DistributedKnowledgeBase:
    """Crea nueva base de conocimiento distribuida"""
    return DistributedKnowledgeBase(node_id, storage_path)


def initialize_knowledge_base(config: Dict[str, Any]) -> DistributedKnowledgeBase:
    """Inicializa Knowledge Base desde configuración"""
    node_id = config.get("node_id", "node_local")
    storage_path = config.get("storage_path", None)

    kb = create_knowledge_base(node_id, storage_path)
    kb.load_state()

    logger.info(f"📚 Knowledge Base inicializada para nodo {node_id}")
    return kb


if __name__ == "__main__":
    # Demo del sistema de Knowledge Base
    async def demo_knowledge_base():
        print("📚 Demo de Knowledge Base Distribuida")
        print("=" * 50)

        # Crear base de conocimiento
        kb = create_knowledge_base("demo_node")

        # Agregar conocimiento de diferentes tipos
        print("\n➕ Agregando conocimiento...")

        # Conocimiento JSON
        model_data = {
            "model_name": "neural_network_v1",
            "layers": 3,
            "parameters": 1000,
            "accuracy": 0.95
        }

        model_id = kb.add_knowledge(
            content=model_data,
            format=KnowledgeFormat.ML_MODEL,
            metadata={"type": "model", "version": "1.0"}
        )

        print(f"✅ Modelo ML agregado: {model_id[:16]}...")

        # Conocimiento de texto
        knowledge_text = "Machine learning es un subcampo de la inteligencia artificial..."

        text_id = kb.add_knowledge(
            content=knowledge_text,
            format=KnowledgeFormat.TEXT,
            metadata={"category": "tutorial", "language": "es"}
        )

        print(f"✅ Texto agregado: {text_id[:16]}...")

        # Buscar conocimiento
        print("\n🔍 Buscando conocimiento...")
        results = kb.search_knowledge("machine learning", limit=5)

        for entry in results:
            print(f"  - {entry.content_id[:16]}...: {entry.format.value} por {entry.author}")

        # Crear rama
        print("\n🌿 Creando rama...")
        feature_branch = kb.create_branch("feature_ml_improvements")
        print(f"✅ Rama creada: {feature_branch}")

        # Mostrar estado
        print("\n📊 Estado de la base:")
        status = kb.get_api_status()
        print(f"  - Entradas totales: {status['total_entries']}")
        print(f"  - Ramas: {status['total_branches']}")
        print(f"  - Tamaño DB: {status['db_size_bytes']} bytes")

        # Cleanup
        kb.cleanup_old_versions()

    asyncio.run(demo_knowledge_base())
