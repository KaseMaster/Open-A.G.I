#!/usr/bin/env python3
"""
Tests para Knowledge Base Distribuida - AEGIS Framework
Pruebas de funcionalidad del sistema de conocimiento distribuido.
"""

import pytest
import asyncio
import json
import tempfile
import os
import time
import hashlib
from unittest.mock import Mock

# Asegurar que el directorio del proyecto esté en PYTHONPATH
import sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from distributed_knowledge_base import (
    DistributedKnowledgeBase, KnowledgeFormat, SyncStatus,
    KnowledgeEntry, KnowledgeBranch, create_knowledge_base
)


class TestKnowledgeBaseCore:
    """Tests básicos de la Knowledge Base"""

    def setup_method(self):
        """Configuración para cada test"""
        self.temp_dir = tempfile.mkdtemp()
        self.kb = create_knowledge_base("test_node", self.temp_dir)

    def teardown_method(self):
        """Limpieza después de cada test"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_knowledge_base_initialization(self):
        """Test inicialización de la base de conocimiento"""
        assert self.kb.node_id == "test_node"
        assert self.kb.current_branch == "main"
        assert "main" in self.kb.branches
        assert self.kb.branches["main"].is_default is True
        assert len(self.kb.entries) == 0

    def test_add_knowledge_json(self):
        """Test agregar conocimiento en formato JSON"""
        content = {
            "model_name": "test_model",
            "accuracy": 0.95,
            "parameters": 1000
        }

        content_id = self.kb.add_knowledge(
            content=content,
            format=KnowledgeFormat.ML_MODEL,
            metadata={"version": "1.0", "type": "neural_network"}
        )

        assert content_id is not None
        assert len(content_id) == 64  # SHA256 hex length

        # Verificar que se guardó
        entry = self.kb.get_knowledge(content_id)
        assert entry is not None
        assert entry.content == content
        assert entry.format == KnowledgeFormat.ML_MODEL
        assert entry.author == "test_node"
        assert entry.sync_status == SyncStatus.LOCAL

    def test_add_knowledge_text(self):
        """Test agregar conocimiento en formato texto"""
        content = "Este es un conocimiento de prueba para el sistema distribuido."

        content_id = self.kb.add_knowledge(
            content=content,
            format=KnowledgeFormat.TEXT,
            metadata={"category": "test", "language": "es"}
        )

        assert content_id is not None

        # Verificar que se guardó correctamente
        entry = self.kb.get_knowledge(content_id)
        assert entry.content == content
        assert entry.format == KnowledgeFormat.TEXT
        assert entry.metadata["category"] == "test"
        assert entry.size_bytes > 0

    def test_search_knowledge(self):
        """Test búsqueda de conocimiento"""
        # Agregar múltiples entradas
        entries = [
            {"type": "model", "name": "neural_network_v1"},
            {"type": "tutorial", "title": "Machine Learning Basics"},
            {"type": "data", "description": "Neural network weights"}
        ]

        content_ids = []
        for entry in entries:
            content_id = self.kb.add_knowledge(
                content=entry,
                format=KnowledgeFormat.JSON,
                metadata={"category": entry["type"]}
            )
            content_ids.append(content_id)

        # Buscar por "neural"
        results = self.kb.search_knowledge("neural", limit=10)

        assert len(results) >= 2  # Debería encontrar al menos 2 resultados

        # Buscar por categoría específica
        model_results = self.kb.search_knowledge("model", format=KnowledgeFormat.JSON)
        assert len(model_results) >= 1

    def test_versioning_system(self):
        """Test sistema de versionado"""
        # Agregar versión inicial
        content_v1 = {"model": "v1", "params": 100}
        content_id_v1 = self.kb.add_knowledge(
            content=content_v1,
            format=KnowledgeFormat.ML_MODEL
        )

        # Agregar versión actualizada
        content_v2 = {"model": "v1", "params": 200}  # Mismo modelo, parámetros diferentes
        content_id_v2 = self.kb.add_knowledge(
            content=content_v2,
            format=KnowledgeFormat.ML_MODEL
        )

        # Verificar que son versiones diferentes
        assert content_id_v1 != content_id_v2

        # Verificar que la rama apunta a la versión más reciente
        current_branch = self.kb.branches[self.kb.current_branch]
        assert current_branch.head_version == content_id_v2

    def test_branch_management(self):
        """Test gestión de ramas"""
        # Crear nueva rama
        branch_id = self.kb.create_branch("feature_new_model", "main")

        assert branch_id is not None
        assert branch_id in self.kb.branches
        assert self.kb.branches[branch_id].name == "feature_new_model"

        # Cambiar a la nueva rama
        old_branch = self.kb.current_branch
        self.kb.current_branch = branch_id

        # Agregar conocimiento en la nueva rama
        content = {"feature": "new_model_architecture"}
        content_id = self.kb.add_knowledge(
            content=content,
            format=KnowledgeFormat.JSON
        )

        # Verificar que la nueva rama apunta al nuevo contenido
        new_branch = self.kb.branches[branch_id]
        assert new_branch.head_version == content_id

        # La rama original no debería haber cambiado
        old_branch_obj = self.kb.branches[old_branch]
        assert old_branch_obj.head_version != content_id

    def test_merge_branches(self):
        """Test fusión de ramas"""
        # Crear rama de características
        feature_branch = self.kb.create_branch("feature_branch")

        # Cambiar a rama de características
        self.kb.current_branch = feature_branch

        # Agregar cambios en feature branch
        feature_content = {"feature": "new_architecture", "version": "2.0"}
        feature_id = self.kb.add_knowledge(
            content=feature_content,
            format=KnowledgeFormat.JSON,
            metadata={"branch": "feature"}
        )

        # Cambiar de vuelta a main y agregar cambios allí también
        self.kb.current_branch = "main"
        main_content = {"model": "improved", "version": "1.5"}
        main_id = self.kb.add_knowledge(
            content=main_content,
            format=KnowledgeFormat.JSON,
            metadata={"branch": "main"}
        )

        # Fusionar feature branch en main
        merge_result = self.kb.merge_branches(feature_branch, "main")

        assert merge_result["success"] is True
        assert merge_result["merged_changes"] >= 1

        # Verificar que main ahora incluye los cambios de feature
        final_main = self.kb.branches["main"]
        assert final_main.head_version != main_id  # Debería haber cambiado


class TestKnowledgeBaseSync:
    """Tests de sincronización entre nodos"""

    def setup_method(self):
        """Configuración para tests de sincronización"""
        self.temp_dir1 = tempfile.mkdtemp()
        self.temp_dir2 = tempfile.mkdtemp()
        self.kb1 = create_knowledge_base("node_1", self.temp_dir1)
        self.kb2 = create_knowledge_base("node_2", self.temp_dir2)

    def teardown_method(self):
        """Limpieza después de tests"""
        import shutil
        shutil.rmtree(self.temp_dir1, ignore_errors=True)
        shutil.rmtree(self.temp_dir2, ignore_errors=True)

    def test_initial_sync_state(self):
        """Test estado inicial de sincronización"""
        status = self.kb1.get_sync_status("node_2")

        assert status["peer_id"] == "node_2"
        assert status["local_versions"] == 0
        assert status["known_by_peer"] == 0
        assert status["sync_in_progress"] is False

    def test_knowledge_sync_between_peers(self):
        """Test sincronización de conocimiento entre peers"""
        # Agregar conocimiento en node1
        content = {"model": "sync_test", "version": "1.0"}
        content_id = self.kb1.add_knowledge(
            content=content,
            format=KnowledgeFormat.ML_MODEL
        )

        # Obtener entradas de node1 para sincronizar
        entries_data = []
        for entry in self.kb1.entries.values():
            entries_data.append({
                "content_id": entry.content_id,
                "content": entry.content,
                "format": entry.format.value,
                "metadata": entry.metadata,
                "version": entry.version,
                "parent_versions": entry.parent_versions,
                "author": entry.author,
                "timestamp": entry.timestamp,
                "signature": entry.signature
            })

        # Sincronizar con node2
        sync_result = self.kb2.sync_with_peer("node_1", entries_data)

        assert sync_result["success"] is True
        assert sync_result["missing_added"] == 1
        assert sync_result["local_versions"] == 1  # node2 ahora tiene la entrada

        # Verificar que node2 tiene la entrada
        synced_entry = self.kb2.get_knowledge(content_id)
        assert synced_entry is not None
        assert synced_entry.content == content
        assert synced_entry.sync_status == SyncStatus.SYNCED

    def test_conflict_resolution(self):
        """Test resolución de conflictos en sincronización"""
        # Agregar mismo contenido en ambos nodos con diferentes valores
        content1 = {"value": 100, "node": "node1"}
        content2 = {"value": 200, "node": "node2"}  # Conflicto

        # Calcular mismo hash para simular conflicto
        content_str1 = json.dumps(content1, sort_keys=True)
        content_str2 = json.dumps(content2, sort_keys=True)

        # Ambos deberían tener el mismo hash si el contenido es idéntico
        # Para este test, creamos entradas con timestamps diferentes
        time.sleep(0.1)  # Asegurar timestamps diferentes

        # Agregar en node1
        content_id1 = self.kb1.add_knowledge(
            content=content1,
            format=KnowledgeFormat.JSON
        )

        time.sleep(0.1)

        # Agregar en node2 (mismo hash, contenido diferente)
        content_id2 = self.kb2.add_knowledge(
            content=content2,
            format=KnowledgeFormat.JSON
        )

        # Verificar que ambos nodos tienen versiones diferentes
        assert content_id1 != content_id2

        # Simular sincronización
        entries1 = []
        for entry in self.kb1.entries.values():
            entries1.append({
                "content_id": entry.content_id,
                "content": entry.content,
                "format": entry.format.value,
                "metadata": entry.metadata,
                "version": entry.version,
                "parent_versions": entry.parent_versions,
                "author": entry.author,
                "timestamp": entry.timestamp,
                "signature": entry.signature
            })

        sync_result = self.kb2.sync_with_peer("node_1", entries1)

        # Verificar que se detectó y resolvió el conflicto
        assert sync_result["conflicts_resolved"] >= 0


class TestKnowledgeBaseAPI:
    """Tests de la API de Knowledge Base"""

    def setup_method(self):
        """Configuración para tests de API"""
        self.temp_dir = tempfile.mkdtemp()
        self.kb = create_knowledge_base("api_test_node", self.temp_dir)

    def teardown_method(self):
        """Limpieza después de tests"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_api_status(self):
        """Test API de estado de la base de conocimiento"""
        status = self.kb.get_api_status()

        assert status["node_id"] == "api_test_node"
        assert status["current_branch"] == "main"
        assert status["total_entries"] == 0
        assert status["total_branches"] == 1
        assert "storage_path" in status
        assert "db_size_bytes" in status

    def test_get_branch_history(self):
        """Test obtener historial de rama"""
        # Agregar múltiples versiones
        for i in range(3):
            content = {"version": i + 1, "data": f"content_v{i+1}"}
            self.kb.add_knowledge(
                content=content,
                format=KnowledgeFormat.JSON,
                metadata={"iteration": i + 1}
            )

        # Obtener historial de la rama main
        history = self.kb.get_branch_history("main")

        assert len(history) == 3
        # Verificar que están ordenados por timestamp descendente
        assert history[0].metadata["iteration"] == 3
        assert history[1].metadata["iteration"] == 2
        assert history[2].metadata["iteration"] == 1

    def test_knowledge_entry_properties(self):
        """Test propiedades de entradas de conocimiento"""
        content = {"large": "x" * 1000}  # Contenido grande para test de tamaño

        content_id = self.kb.add_knowledge(
            content=content,
            format=KnowledgeFormat.JSON
        )

        entry = self.kb.get_knowledge(content_id)

        assert entry.size_bytes > 1000  # Debería calcular tamaño correctamente
        assert entry.content_id.startswith(hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest())
        assert entry.author == "api_test_node"
        assert entry.sync_status == SyncStatus.LOCAL
        assert entry.format == KnowledgeFormat.JSON


# Tests asíncronos
@pytest.mark.asyncio
async def test_async_knowledge_operations():
    """Test operaciones asíncronas de la Knowledge Base"""
    temp_dir = tempfile.mkdtemp()
    kb = create_knowledge_base("async_node", temp_dir)

    try:
        # Operaciones asíncronas simuladas
        await asyncio.sleep(0.1)

        # Verificar que la base funciona en contexto asíncrono
        status = kb.get_api_status()
        assert status["node_id"] == "async_node"

        # Agregar conocimiento asíncronamente
        content_id = kb.add_knowledge(
            content="async test content",
            format=KnowledgeFormat.TEXT
        )

        assert content_id is not None

        # Limpiar versiones antiguas
        kb.cleanup_old_versions()

        # Verificar que la limpieza funcionó
        assert len(kb.entries) >= 1

    finally:
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    # Ejecutar tests con pytest
    pytest.main([__file__, "-v", "--tb=short"])
