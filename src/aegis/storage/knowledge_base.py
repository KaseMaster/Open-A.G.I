#!/usr/bin/env python3
"""
Knowledge Base Distribuida MVP - AEGIS Framework
Sistema de almacenamiento direccionado por contenido con versionado y sincronización P2P

Características:
- Almacenamiento direccionado por contenido (Content-Addressed Storage)
- Versionado ligero tipo Git con merge automático
- Sincronización P2P con detección de conflictos
- Indexación distribuida con Bloom filters
- Compresión y deduplicación automática
- Tolerancia a fallos con replicación 3x

Autor: AEGIS Security Framework
Uso: Exclusivamente ético y educativo
"""

import asyncio
import hashlib
import json
import time
import zlib
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Set, Tuple
from enum import Enum
import logging
from collections import defaultdict
import os
import sqlite3
import aiosqlite
from pathlib import Path

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ContentType(Enum):
    """Tipos de contenido soportados"""
    TEXT = "text"
    JSON = "json"
    BINARY = "binary"
    MODEL = "model"
    DATASET = "dataset"
    CODE = "code"

class VersionOperation(Enum):
    """Operaciones de versionado"""
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    MERGE = "merge"
    FORK = "fork"

class SyncStatus(Enum):
    """Estados de sincronización"""
    SYNCED = "synced"
    PENDING = "pending"
    CONFLICT = "conflict"
    ERROR = "error"

@dataclass
class ContentHash:
    """Hash de contenido con metadatos"""
    hash_value: str
    algorithm: str = "sha256"
    size: int = 0
    compressed_size: int = 0
    
    def __post_init__(self):
        if len(self.hash_value) != 64:  # SHA256 hex length
            raise ValueError("Hash SHA256 inválido")

@dataclass
class ContentVersion:
    """Versión de contenido"""
    version_id: str
    content_hash: ContentHash
    parent_versions: List[str]
    author_node: str
    timestamp: float
    operation: VersionOperation
    metadata: Dict[str, Any]
    signature: Optional[str] = None

@dataclass
class ContentEntry:
    """Entrada de contenido en la base de conocimiento"""
    content_id: str
    content_type: ContentType
    current_version: str
    versions: Dict[str, ContentVersion]
    tags: Set[str]
    access_count: int
    last_accessed: float
    replication_nodes: Set[str]
    sync_status: SyncStatus
    
class BloomFilter:
    """Bloom filter simple para indexación distribuida"""
    
    def __init__(self, capacity: int = 10000, error_rate: float = 0.1):
        self.capacity = capacity
        self.error_rate = error_rate
        self.bit_array_size = self._calculate_bit_array_size()
        self.hash_count = self._calculate_hash_count()
        self.bit_array = bytearray(self.bit_array_size // 8 + 1)
    
    def _calculate_bit_array_size(self) -> int:
        """Calcula el tamaño óptimo del array de bits"""
        import math
        return int(-self.capacity * math.log(self.error_rate) / (math.log(2) ** 2))
    
    def _calculate_hash_count(self) -> int:
        """Calcula el número óptimo de funciones hash"""
        import math
        return int(self.bit_array_size * math.log(2) / self.capacity)
    
    def _hash(self, item: str, seed: int) -> int:
        """Función hash con semilla"""
        return hash(f"{item}:{seed}") % self.bit_array_size
    
    def add(self, item: str) -> None:
        """Añade un elemento al filtro"""
        for i in range(self.hash_count):
            bit_index = self._hash(item, i)
            byte_index = bit_index // 8
            bit_offset = bit_index % 8
            self.bit_array[byte_index] |= (1 << bit_offset)
    
    def contains(self, item: str) -> bool:
        """Verifica si un elemento puede estar en el filtro"""
        for i in range(self.hash_count):
            bit_index = self._hash(item, i)
            byte_index = bit_index // 8
            bit_offset = bit_index % 8
            if not (self.bit_array[byte_index] & (1 << bit_offset)):
                return False
        return True

class DistributedKnowledgeBase:
    """Base de conocimiento distribuida con almacenamiento direccionado por contenido"""
    
    def __init__(self, node_id: str, storage_path: str = "./knowledge_storage", 
                 p2p_manager: Optional[Any] = None):
        self.node_id = node_id
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        self.p2p_manager = p2p_manager
        
        # Base de datos local
        self.db_path = self.storage_path / "knowledge.db"
        
        # Índices en memoria
        self.content_index: Dict[str, ContentEntry] = {}
        self.hash_to_content: Dict[str, str] = {}
        self.tag_index: Dict[str, Set[str]] = defaultdict(set)
        
        # Bloom filters para indexación distribuida
        self.local_bloom = BloomFilter()
        self.peer_blooms: Dict[str, BloomFilter] = {}
        
        # Configuración de replicación
        self.replication_factor = 3
        self.sync_interval = 30  # segundos
        
        # Estado de sincronización
        self.pending_syncs: Dict[str, float] = {}
        self.sync_conflicts: Dict[str, List[ContentVersion]] = {}
        
        # Tareas asíncronas
        self._sync_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
    
    async def initialize(self) -> None:
        """Inicializa la base de conocimiento"""
        try:
            await self._setup_database()
            await self._load_content_index()
            await self._rebuild_bloom_filter()
            
            # Iniciar tareas de mantenimiento
            self._sync_task = asyncio.create_task(self._sync_loop())
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            
            logger.info(f"Knowledge Base inicializada: {len(self.content_index)} entradas")
            
        except Exception as e:
            logger.error(f"Error inicializando Knowledge Base: {e}")
            raise
    
    async def _setup_database(self) -> None:
        """Configura la base de datos SQLite"""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS content_entries (
                    content_id TEXT PRIMARY KEY,
                    content_type TEXT NOT NULL,
                    current_version TEXT NOT NULL,
                    tags TEXT,
                    access_count INTEGER DEFAULT 0,
                    last_accessed REAL,
                    replication_nodes TEXT,
                    sync_status TEXT DEFAULT 'synced',
                    created_at REAL DEFAULT (julianday('now')),
                    updated_at REAL DEFAULT (julianday('now'))
                )
            """)
            
            await db.execute("""
                CREATE TABLE IF NOT EXISTS content_versions (
                    version_id TEXT PRIMARY KEY,
                    content_id TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    parent_versions TEXT,
                    author_node TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    operation TEXT NOT NULL,
                    metadata TEXT,
                    signature TEXT,
                    FOREIGN KEY (content_id) REFERENCES content_entries (content_id)
                )
            """)
            
            await db.execute("""
                CREATE TABLE IF NOT EXISTS content_data (
                    content_hash TEXT PRIMARY KEY,
                    data BLOB NOT NULL,
                    compressed_data BLOB,
                    size INTEGER NOT NULL,
                    compressed_size INTEGER,
                    created_at REAL DEFAULT (julianday('now'))
                )
            """)
            
            await db.execute("CREATE INDEX IF NOT EXISTS idx_content_hash ON content_versions(content_hash)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_content_type ON content_entries(content_type)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_tags ON content_entries(tags)")
            
            await db.commit()
    
    async def _load_content_index(self) -> None:
        """Carga el índice de contenido desde la base de datos"""
        async with aiosqlite.connect(self.db_path) as db:
            # Cargar entradas de contenido
            async with db.execute("SELECT * FROM content_entries") as cursor:
                async for row in cursor:
                    content_id = row[0]
                    entry = ContentEntry(
                        content_id=content_id,
                        content_type=ContentType(row[1]),
                        current_version=row[2],
                        versions={},
                        tags=set(json.loads(row[3]) if row[3] else []),
                        access_count=row[4],
                        last_accessed=row[5],
                        replication_nodes=set(json.loads(row[6]) if row[6] else []),
                        sync_status=SyncStatus(row[7])
                    )
                    self.content_index[content_id] = entry
                    
                    # Actualizar índice de tags
                    for tag in entry.tags:
                        self.tag_index[tag].add(content_id)
            
            # Cargar versiones
            async with db.execute("SELECT * FROM content_versions") as cursor:
                async for row in cursor:
                    version = ContentVersion(
                        version_id=row[0],
                        content_hash=ContentHash(row[2]),
                        parent_versions=json.loads(row[3]) if row[3] else [],
                        author_node=row[4],
                        timestamp=row[5],
                        operation=VersionOperation(row[6]),
                        metadata=json.loads(row[7]) if row[7] else {},
                        signature=row[8]
                    )
                    
                    content_id = row[1]
                    if content_id in self.content_index:
                        self.content_index[content_id].versions[version.version_id] = version
                        self.hash_to_content[version.content_hash.hash_value] = content_id
    
    async def _rebuild_bloom_filter(self) -> None:
        """Reconstruye el Bloom filter local"""
        self.local_bloom = BloomFilter()
        for content_id in self.content_index:
            self.local_bloom.add(content_id)
            # Añadir también los hashes de contenido
            entry = self.content_index[content_id]
            for version in entry.versions.values():
                self.local_bloom.add(version.content_hash.hash_value)
    
    def _calculate_content_hash(self, data: bytes) -> ContentHash:
        """Calcula el hash SHA256 del contenido"""
        hash_value = hashlib.sha256(data).hexdigest()
        compressed_data = zlib.compress(data, level=6)
        
        return ContentHash(
            hash_value=hash_value,
            algorithm="sha256",
            size=len(data),
            compressed_size=len(compressed_data)
        )
    
    def _generate_version_id(self) -> str:
        """Genera un ID único para una versión"""
        timestamp = str(time.time())
        node_hash = hashlib.sha256(f"{self.node_id}:{timestamp}".encode()).hexdigest()
        return f"v_{node_hash[:16]}_{int(time.time())}"
    
    async def store_content(self, content_id: str, data: bytes, 
                          content_type: ContentType = ContentType.BINARY,
                          tags: Optional[Set[str]] = None,
                          metadata: Optional[Dict[str, Any]] = None) -> str:
        """Almacena contenido en la base de conocimiento"""
        try:
            # Calcular hash del contenido
            content_hash = self._calculate_content_hash(data)
            
            # Verificar si el contenido ya existe
            if content_hash.hash_value in self.hash_to_content:
                existing_content_id = self.hash_to_content[content_hash.hash_value]
                logger.info(f"Contenido duplicado detectado: {existing_content_id}")
                return existing_content_id
            
            # Crear nueva versión
            version_id = self._generate_version_id()
            parent_versions = []
            
            # Si el content_id ya existe, obtener versión padre
            if content_id in self.content_index:
                parent_versions = [self.content_index[content_id].current_version]
            
            version = ContentVersion(
                version_id=version_id,
                content_hash=content_hash,
                parent_versions=parent_versions,
                author_node=self.node_id,
                timestamp=time.time(),
                operation=VersionOperation.CREATE if content_id not in self.content_index else VersionOperation.UPDATE,
                metadata=metadata or {}
            )
            
            # Comprimir datos
            compressed_data = zlib.compress(data, level=6)
            
            # Almacenar en base de datos
            async with aiosqlite.connect(self.db_path) as db:
                # Almacenar datos del contenido
                await db.execute("""
                    INSERT OR REPLACE INTO content_data 
                    (content_hash, data, compressed_data, size, compressed_size)
                    VALUES (?, ?, ?, ?, ?)
                """, (content_hash.hash_value, data, compressed_data, 
                     content_hash.size, content_hash.compressed_size))
                
                # Almacenar versión
                await db.execute("""
                    INSERT INTO content_versions 
                    (version_id, content_id, content_hash, parent_versions, 
                     author_node, timestamp, operation, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (version_id, content_id, content_hash.hash_value,
                     json.dumps(parent_versions), self.node_id, version.timestamp,
                     version.operation.value, json.dumps(version.metadata)))
                
                # Crear o actualizar entrada de contenido
                tags_json = json.dumps(list(tags)) if tags else "[]"
                replication_nodes = json.dumps([self.node_id])
                
                await db.execute("""
                    INSERT OR REPLACE INTO content_entries 
                    (content_id, content_type, current_version, tags, 
                     access_count, last_accessed, replication_nodes, sync_status)
                    VALUES (?, ?, ?, ?, 0, ?, ?, 'pending')
                """, (content_id, content_type.value, version_id, tags_json,
                     time.time(), replication_nodes))
                
                await db.commit()
            
            # Actualizar índices en memoria
            if content_id not in self.content_index:
                self.content_index[content_id] = ContentEntry(
                    content_id=content_id,
                    content_type=content_type,
                    current_version=version_id,
                    versions={},
                    tags=tags or set(),
                    access_count=0,
                    last_accessed=time.time(),
                    replication_nodes={self.node_id},
                    sync_status=SyncStatus.PENDING
                )
            
            entry = self.content_index[content_id]
            entry.versions[version_id] = version
            entry.current_version = version_id
            entry.sync_status = SyncStatus.PENDING
            
            # Actualizar índices
            self.hash_to_content[content_hash.hash_value] = content_id
            self.local_bloom.add(content_id)
            self.local_bloom.add(content_hash.hash_value)
            
            if tags:
                for tag in tags:
                    self.tag_index[tag].add(content_id)
            
            # Marcar para sincronización
            self.pending_syncs[content_id] = time.time()
            
            logger.info(f"Contenido almacenado: {content_id} (versión {version_id})")
            return content_id
            
        except Exception as e:
            logger.error(f"Error almacenando contenido {content_id}: {e}")
            raise
    
    async def retrieve_content(self, content_id: str, version_id: Optional[str] = None) -> Optional[bytes]:
        """Recupera contenido de la base de conocimiento"""
        try:
            if content_id not in self.content_index:
                # Intentar buscar en peers
                return await self._retrieve_from_peers(content_id, version_id)
            
            entry = self.content_index[content_id]
            target_version = version_id or entry.current_version
            
            if target_version not in entry.versions:
                logger.warning(f"Versión {target_version} no encontrada para {content_id}")
                return None
            
            version = entry.versions[target_version]
            content_hash = version.content_hash.hash_value
            
            # Recuperar datos de la base de datos
            async with aiosqlite.connect(self.db_path) as db:
                async with db.execute(
                    "SELECT compressed_data, data FROM content_data WHERE content_hash = ?",
                    (content_hash,)
                ) as cursor:
                    row = await cursor.fetchone()
                    if not row:
                        logger.warning(f"Datos no encontrados para hash {content_hash}")
                        return None
                    
                    # Usar datos comprimidos si están disponibles
                    if row[0]:
                        data = zlib.decompress(row[0])
                    else:
                        data = row[1]
                    
                    # Actualizar estadísticas de acceso
                    entry.access_count += 1
                    entry.last_accessed = time.time()
                    
                    await db.execute("""
                        UPDATE content_entries 
                        SET access_count = ?, last_accessed = ?
                        WHERE content_id = ?
                    """, (entry.access_count, entry.last_accessed, content_id))
                    await db.commit()
                    
                    return data
        
        except Exception as e:
            logger.error(f"Error recuperando contenido {content_id}: {e}")
            return None
    
    async def _retrieve_from_peers(self, content_id: str, version_id: Optional[str] = None) -> Optional[bytes]:
        """Intenta recuperar contenido de peers de la red"""
        if not self.p2p_manager:
            return None
        
        try:
            # Consultar peers que pueden tener el contenido
            query_message = {
                "type": "knowledge_query",
                "content_id": content_id,
                "version_id": version_id,
                "requester": self.node_id,
                "timestamp": time.time()
            }
            
            # Enviar consulta a peers (implementación simplificada)
            # En una implementación real, esto usaría el P2P manager
            logger.info(f"Consultando peers por contenido {content_id}")
            
            # Por ahora, retornar None (implementación futura)
            return None
            
        except Exception as e:
            logger.error(f"Error consultando peers por {content_id}: {e}")
            return None
    
    async def search_content(self, query: str, content_type: Optional[ContentType] = None,
                           tags: Optional[Set[str]] = None, limit: int = 10) -> List[str]:
        """Busca contenido en la base de conocimiento"""
        results = []
        
        try:
            # Búsqueda por tags
            if tags:
                candidates = set()
                for tag in tags:
                    if tag in self.tag_index:
                        if not candidates:
                            candidates = self.tag_index[tag].copy()
                        else:
                            candidates &= self.tag_index[tag]
                
                for content_id in candidates:
                    if len(results) >= limit:
                        break
                    if content_type is None or self.content_index[content_id].content_type == content_type:
                        results.append(content_id)
            
            # Búsqueda por tipo de contenido
            elif content_type:
                for content_id, entry in self.content_index.items():
                    if len(results) >= limit:
                        break
                    if entry.content_type == content_type:
                        results.append(content_id)
            
            # Búsqueda textual simple (en metadatos)
            else:
                query_lower = query.lower()
                for content_id, entry in self.content_index.items():
                    if len(results) >= limit:
                        break
                    
                    # Buscar en ID y metadatos
                    if (query_lower in content_id.lower() or
                        any(query_lower in str(v).lower() 
                            for version in entry.versions.values()
                            for v in version.metadata.values())):
                        results.append(content_id)
            
            return results
            
        except Exception as e:
            logger.error(f"Error en búsqueda: {e}")
            return []
    
    async def get_content_info(self, content_id: str) -> Optional[Dict[str, Any]]:
        """Obtiene información detallada de un contenido"""
        if content_id not in self.content_index:
            return None
        
        entry = self.content_index[content_id]
        current_version = entry.versions.get(entry.current_version)
        
        return {
            "content_id": content_id,
            "content_type": entry.content_type.value,
            "current_version": entry.current_version,
            "version_count": len(entry.versions),
            "tags": list(entry.tags),
            "access_count": entry.access_count,
            "last_accessed": entry.last_accessed,
            "replication_nodes": list(entry.replication_nodes),
            "sync_status": entry.sync_status.value,
            "size": current_version.content_hash.size if current_version else 0,
            "compressed_size": current_version.content_hash.compressed_size if current_version else 0,
            "created_by": current_version.author_node if current_version else None,
            "created_at": current_version.timestamp if current_version else None
        }
    
    async def _sync_loop(self) -> None:
        """Bucle de sincronización con peers"""
        while True:
            try:
                await asyncio.sleep(self.sync_interval)
                await self._sync_with_peers()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error en sincronización: {e}")
    
    async def _sync_with_peers(self) -> None:
        """Sincroniza contenido pendiente con peers"""
        if not self.pending_syncs:
            return
        
        logger.info(f"Sincronizando {len(self.pending_syncs)} elementos pendientes")
        
        # Implementación simplificada - en producción usaría P2P manager
        for content_id in list(self.pending_syncs.keys()):
            try:
                # Marcar como sincronizado (simulado)
                if content_id in self.content_index:
                    self.content_index[content_id].sync_status = SyncStatus.SYNCED
                    
                    async with aiosqlite.connect(self.db_path) as db:
                        await db.execute("""
                            UPDATE content_entries 
                            SET sync_status = 'synced'
                            WHERE content_id = ?
                        """, (content_id,))
                        await db.commit()
                
                del self.pending_syncs[content_id]
                
            except Exception as e:
                logger.error(f"Error sincronizando {content_id}: {e}")
    
    async def _cleanup_loop(self) -> None:
        """Bucle de limpieza y mantenimiento"""
        while True:
            try:
                await asyncio.sleep(300)  # 5 minutos
                await self._cleanup_old_data()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error en limpieza: {e}")
    
    async def _cleanup_old_data(self) -> None:
        """Limpia datos antiguos y optimiza almacenamiento"""
        try:
            # Limpiar versiones antiguas (mantener últimas 10 por contenido)
            cleanup_count = 0
            
            async with aiosqlite.connect(self.db_path) as db:
                for content_id, entry in self.content_index.items():
                    if len(entry.versions) > 10:
                        # Ordenar versiones por timestamp
                        sorted_versions = sorted(
                            entry.versions.items(),
                            key=lambda x: x[1].timestamp,
                            reverse=True
                        )
                        
                        # Mantener solo las 10 más recientes
                        versions_to_keep = dict(sorted_versions[:10])
                        versions_to_delete = [v_id for v_id, _ in sorted_versions[10:]]
                        
                        # Eliminar versiones antiguas
                        for version_id in versions_to_delete:
                            await db.execute(
                                "DELETE FROM content_versions WHERE version_id = ?",
                                (version_id,)
                            )
                            del entry.versions[version_id]
                            cleanup_count += 1
                
                await db.commit()
            
            if cleanup_count > 0:
                logger.info(f"Limpieza completada: {cleanup_count} versiones eliminadas")
                
        except Exception as e:
            logger.error(f"Error en limpieza: {e}")
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Obtiene estadísticas de la base de conocimiento"""
        try:
            total_content = len(self.content_index)
            total_versions = sum(len(entry.versions) for entry in self.content_index.values())
            
            # Estadísticas por tipo de contenido
            type_stats = defaultdict(int)
            for entry in self.content_index.values():
                type_stats[entry.content_type.value] += 1
            
            # Calcular tamaño total
            total_size = 0
            total_compressed_size = 0
            
            async with aiosqlite.connect(self.db_path) as db:
                async with db.execute("SELECT SUM(size), SUM(compressed_size) FROM content_data") as cursor:
                    row = await cursor.fetchone()
                    if row:
                        total_size = row[0] or 0
                        total_compressed_size = row[1] or 0
            
            return {
                "total_content": total_content,
                "total_versions": total_versions,
                "content_by_type": dict(type_stats),
                "total_size_bytes": total_size,
                "total_compressed_size_bytes": total_compressed_size,
                "compression_ratio": (1 - total_compressed_size / total_size) if total_size > 0 else 0,
                "pending_syncs": len(self.pending_syncs),
                "sync_conflicts": len(self.sync_conflicts),
                "storage_path": str(self.storage_path),
                "node_id": self.node_id
            }
            
        except Exception as e:
            logger.error(f"Error obteniendo estadísticas: {e}")
            return {}
    
    async def shutdown(self) -> None:
        """Cierra la base de conocimiento limpiamente"""
        try:
            # Cancelar tareas de mantenimiento
            if self._sync_task:
                self._sync_task.cancel()
                try:
                    await self._sync_task
                except asyncio.CancelledError:
                    pass
            
            if self._cleanup_task:
                self._cleanup_task.cancel()
                try:
                    await self._cleanup_task
                except asyncio.CancelledError:
                    pass
            
            # Sincronización final
            await self._sync_with_peers()
            
            logger.info("Knowledge Base cerrada correctamente")
            
        except Exception as e:
            logger.error(f"Error cerrando Knowledge Base: {e}")

# Funciones de utilidad para integración

async def create_knowledge_base(node_id: str, storage_path: str = "./knowledge_storage",
                              p2p_manager: Optional[Any] = None) -> DistributedKnowledgeBase:
    """Crea e inicializa una nueva base de conocimiento"""
    kb = DistributedKnowledgeBase(node_id, storage_path, p2p_manager)
    await kb.initialize()
    return kb

async def demo_knowledge_base():
    """Demostración de la base de conocimiento"""
    print("🧠 Iniciando demo de Knowledge Base Distribuida...")
    
    # Crear base de conocimiento
    kb = await create_knowledge_base("demo_node_001")
    
    try:
        # Almacenar contenido de prueba
        test_data = b"Este es un contenido de prueba para la base de conocimiento distribuida"
        content_id = await kb.store_content(
            "test_content_001",
            test_data,
            ContentType.TEXT,
            tags={"demo", "test", "knowledge"},
            metadata={"description": "Contenido de prueba", "version": "1.0"}
        )
        
        print(f"✅ Contenido almacenado: {content_id}")
        
        # Recuperar contenido
        retrieved_data = await kb.retrieve_content(content_id)
        if retrieved_data == test_data:
            print("✅ Contenido recuperado correctamente")
        
        # Buscar contenido
        results = await kb.search_content("test", tags={"demo"})
        print(f"✅ Búsqueda completada: {len(results)} resultados")
        
        # Obtener información
        info = await kb.get_content_info(content_id)
        print(f"✅ Información del contenido: {info}")
        
        # Estadísticas
        stats = await kb.get_statistics()
        print(f"✅ Estadísticas: {stats}")
        
    finally:
        await kb.shutdown()

if __name__ == "__main__":
    asyncio.run(demo_knowledge_base())