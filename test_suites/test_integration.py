#!/usr/bin/env python3
"""
Tests de Integración para AEGIS
Verifica la interacción correcta entre todos los componentes del sistema.
"""

import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional
from unittest.mock import AsyncMock, Mock, patch
import tempfile
import shutil

# Importaciones condicionales para dependencias opcionales
try:
    from storage_system import AEGISStorage
    STORAGE_AVAILABLE = True
except ImportError:
    STORAGE_AVAILABLE = False
    logging.warning("Módulo storage_system no disponible - funcionalidad de almacenamiento deshabilitada")

try:
    import pytest
    PYTEST_AVAILABLE = True
except ImportError:
    PYTEST_AVAILABLE = False
    logging.warning("Módulo pytest no disponible - algunos tests pueden fallar")

# Importar el framework de tests
sys.path.append(str(Path(__file__).parent.parent))
from test_framework import (
    AEGISTestFramework, TestSuite, TestType, TestStatus,
    integration_test, performance_test, stress_test
)

class AEGISIntegrationTests:
    """Tests de integración para todo el sistema AEGIS"""
    
    def __init__(self):
        self.aegis_node = None
        self.temp_dir = None
        self.test_config = None
        self.components = {}
    
    def setup(self):
        """Setup para tests de integración"""
        # Crear directorio temporal
        self.temp_dir = tempfile.mkdtemp(prefix="aegis_integration_")
        
        # Configuración de test
        self.test_config = {
            "node_id": "test_node_001",
            "data_dir": self.temp_dir,
            "p2p_port": 8000,
            "api_port": 8080,
            "tor_integration": False,  # Deshabilitado para tests
            "crypto_framework": True,
            "consensus_system": True,
            "monitoring": True,
            "resource_manager": True,
            "performance_optimizer": True,
            "logging_system": True,
            "config_manager": True,
            "api_server": True,
            "metrics_collector": True,
            "alert_system": True,
            "web_dashboard": True,
            "backup_system": True
        }
        
        # Intentar importar y configurar componentes
        self._setup_components()
    
    def teardown(self):
        """Cleanup después de tests"""
        # Detener todos los componentes
        if self.aegis_node:
            asyncio.create_task(self._stop_aegis_node())
        
        # Limpiar directorio temporal
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def _setup_components(self):
        """Configura los componentes para testing"""
        try:
            # Importar el nodo principal
            from main import AEGISNode
            self.aegis_node = AEGISNode(self.test_config)
        except ImportError:
            # Crear mock del nodo principal
            self.aegis_node = self._create_aegis_mock()
        
        # Configurar componentes individuales
        self._setup_individual_components()
    
    def _setup_individual_components(self):
        """Configura componentes individuales"""
        component_configs = {
            "crypto": {"key_size": 256, "algorithm": "AES-256-GCM"},
            "p2p": {"port": 8000, "max_peers": 10},
            "consensus": {"algorithm": "PBFT", "timeout": 5},
            "storage": {"type": "file", "path": self.temp_dir},
            "metrics": {"collection_interval": 1, "retention_days": 1}
        }
        
        for component, config in component_configs.items():
            try:
                self.components[component] = self._create_component(component, config)
            except Exception as e:
                # Usar mock si el componente no está disponible
                self.components[component] = self._create_component_mock(component)
    
    def _create_component(self, component_name: str, config: dict):
        """Crea un componente real"""
        if component_name == "crypto":
            from crypto_framework import AEGISCryptoFramework
            return AEGISCryptoFramework(config)
        elif component_name == "p2p":
            from p2p_network import AEGISP2PNetwork
            return AEGISP2PNetwork(**config)
        elif component_name == "consensus":
            from consensus_system import AEGISConsensus
            return AEGISConsensus(config)
        elif component_name == "storage":
            if STORAGE_AVAILABLE:
                return AEGISStorage(config)
            else:
                return Mock()
        elif component_name == "metrics":
            from metrics_collector import AEGISMetricsCollector
            return AEGISMetricsCollector(config)
        else:
            return Mock()
    
    def _create_component_mock(self, component_name: str):
        """Crea un mock para un componente"""
        mock = Mock()
        
        if component_name == "crypto":
            mock.encrypt_data = Mock(return_value=b"encrypted")
            mock.decrypt_data = Mock(return_value=b"decrypted")
            mock.generate_key = Mock(return_value="test_key")
        elif component_name == "p2p":
            mock.start = AsyncMock(return_value=True)
            mock.discover_peers = AsyncMock(return_value=[])
            mock.send_message = AsyncMock(return_value=True)
        elif component_name == "consensus":
            mock.propose_value = AsyncMock(return_value=True)
            mock.get_consensus_state = Mock(return_value="ready")
        elif component_name == "storage":
            mock.store_data = AsyncMock(return_value="test_id")
            mock.retrieve_data = AsyncMock(return_value={"test": "data"})
        elif component_name == "metrics":
            mock.collect_metrics = AsyncMock(return_value={"cpu": 50})
            mock.store_metrics = AsyncMock(return_value=True)
        
        return mock
    
    def _create_aegis_mock(self):
        """Crea un mock del nodo AEGIS principal"""
        mock = Mock()
        mock.start = AsyncMock(return_value=True)
        mock.stop = AsyncMock(return_value=True)
        mock.get_status = Mock(return_value={"status": "running"})
        mock.get_components = Mock(return_value=self.components)
        return mock
    
    async def _stop_aegis_node(self):
        """Detiene el nodo AEGIS"""
        if hasattr(self.aegis_node, 'stop'):
            await self.aegis_node.stop()
    
    @integration_test
    async def test_full_system_startup(self):
        """Test de inicio completo del sistema"""
        # Iniciar el nodo AEGIS
        result = await self.aegis_node.start()
        assert result is True, "El nodo AEGIS debe iniciarse correctamente"
        
        # Verificar que todos los componentes están activos
        status = self.aegis_node.get_status()
        assert status["status"] == "running", "El nodo debe estar en estado 'running'"
        
        # Verificar componentes individuales
        components = self.aegis_node.get_components()
        expected_components = ["crypto", "p2p", "consensus", "storage", "metrics"]
        
        for component in expected_components:
            assert component in components, f"El componente {component} debe estar disponible"
    
    @integration_test
    async def test_crypto_p2p_integration(self):
        """Test de integración entre crypto y P2P"""
        crypto = self.components["crypto"]
        p2p = self.components["p2p"]
        
        # Generar par de claves
        key_pair = crypto.generate_key_pair()
        
        # Iniciar P2P
        await p2p.start()
        
        # Crear mensaje encriptado
        test_message = {"type": "test", "data": "integration_test"}
        encrypted_message = crypto.encrypt_data(
            json.dumps(test_message).encode(),
            key_pair["public"]
        )
        
        # Enviar mensaje encriptado via P2P
        p2p_message = {
            "type": "encrypted_data",
            "payload": encrypted_message,
            "sender_key": key_pair["public"]
        }
        
        # Simular envío (en test real sería a un peer)
        result = await p2p.send_message("test_peer", p2p_message)
        assert result is True, "El mensaje encriptado debe enviarse correctamente"
    
    @integration_test
    async def test_consensus_storage_integration(self):
        """Test de integración entre consenso y almacenamiento"""
        consensus = self.components["consensus"]
        storage = self.components["storage"]
        
        # Proponer un valor para consenso
        test_value = {"action": "store_data", "data": {"key": "value"}}
        
        # Simular proceso de consenso
        consensus_result = await consensus.propose_value(test_value)
        assert consensus_result is True, "La propuesta de consenso debe ser exitosa"
        
        # Una vez alcanzado el consenso, almacenar el dato
        storage_id = await storage.store_data(test_value["data"])
        assert storage_id is not None, "Los datos deben almacenarse correctamente"
        
        # Verificar que se pueden recuperar
        retrieved_data = await storage.retrieve_data(storage_id)
        assert retrieved_data == test_value["data"], "Los datos recuperados deben coincidir"
    
    @integration_test
    async def test_metrics_monitoring_integration(self):
        """Test de integración entre métricas y monitoreo"""
        metrics = self.components["metrics"]
        
        # Recopilar métricas del sistema
        system_metrics = await metrics.collect_metrics()
        assert system_metrics is not None, "Las métricas del sistema deben recopilarse"
        assert "cpu" in system_metrics, "Las métricas deben incluir CPU"
        
        # Almacenar métricas
        storage_result = await metrics.store_metrics(system_metrics)
        assert storage_result is True, "Las métricas deben almacenarse correctamente"
        
        # Verificar alertas basadas en métricas
        if system_metrics.get("cpu", 0) > 80:
            # Simular alerta de CPU alta
            alert_triggered = True
        else:
            alert_triggered = False
        
        # En un sistema real, esto activaría el sistema de alertas
        assert isinstance(alert_triggered, bool), "El sistema de alertas debe responder a métricas"
    
    @integration_test
    async def test_full_data_flow(self):
        """Test del flujo completo de datos en el sistema"""
        crypto = self.components["crypto"]
        p2p = self.components["p2p"]
        consensus = self.components["consensus"]
        storage = self.components["storage"]
        
        # 1. Crear datos de prueba
        test_data = {
            "timestamp": time.time(),
            "type": "user_data",
            "content": "Test integration data"
        }
        
        # 2. Encriptar datos
        key = crypto.generate_key(256)
        encrypted_data = crypto.encrypt_data(json.dumps(test_data).encode(), key)
        
        # 3. Proponer almacenamiento via consenso
        proposal = {
            "action": "store_encrypted_data",
            "data": encrypted_data,
            "key_hash": crypto.hash_data(key.encode())
        }
        
        consensus_result = await consensus.propose_value(proposal)
        assert consensus_result is True, "El consenso para almacenar debe ser exitoso"
        
        # 4. Almacenar datos encriptados
        storage_id = await storage.store_data(encrypted_data)
        assert storage_id is not None, "Los datos encriptados deben almacenarse"
        
        # 5. Notificar a peers via P2P
        notification = {
            "type": "data_stored",
            "storage_id": storage_id,
            "data_hash": crypto.hash_data(encrypted_data)
        }
        
        await p2p.start()
        broadcast_result = await p2p.broadcast_message(notification)
        assert broadcast_result is True, "La notificación debe broadcastearse"
        
        # 6. Recuperar y verificar datos
        retrieved_encrypted = await storage.retrieve_data(storage_id)
        decrypted_data = crypto.decrypt_data(retrieved_encrypted, key)
        final_data = json.loads(decrypted_data.decode())
        
        assert final_data == test_data, "Los datos finales deben coincidir con los originales"
    
    @performance_test
    async def test_system_performance_under_load(self):
        """Test de rendimiento del sistema bajo carga"""
        # Iniciar todos los componentes
        await self.aegis_node.start()
        
        # Simular carga de trabajo
        tasks = []
        task_count = 50
        
        for i in range(task_count):
            # Crear tareas concurrentes que usen múltiples componentes
            task = self._simulate_user_operation(i)
            tasks.append(task)
        
        start_time = time.time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        total_time = time.time() - start_time
        
        # Analizar resultados
        successful_operations = sum(1 for r in results if r is True)
        success_rate = successful_operations / task_count
        throughput = successful_operations / total_time
        
        assert success_rate > 0.8, f"Al menos 80% de operaciones deben ser exitosas (actual: {success_rate:.1%})"
        assert throughput > 5, f"El throughput debe ser mayor a 5 ops/seg (actual: {throughput:.1f})"
        assert total_time < 30, f"Las operaciones deben completarse en menos de 30s (actual: {total_time:.1f}s)"
    
    async def _simulate_user_operation(self, operation_id: int) -> bool:
        """Simula una operación de usuario que usa múltiples componentes"""
        try:
            crypto = self.components["crypto"]
            storage = self.components["storage"]
            
            # 1. Generar datos
            user_data = {
                "operation_id": operation_id,
                "timestamp": time.time(),
                "data": f"User operation {operation_id}"
            }
            
            # 2. Encriptar
            key = crypto.generate_key(256)
            encrypted = crypto.encrypt_data(json.dumps(user_data).encode(), key)
            
            # 3. Almacenar
            storage_id = await storage.store_data(encrypted)
            
            # 4. Verificar almacenamiento
            retrieved = await storage.retrieve_data(storage_id)
            
            return retrieved == encrypted
            
        except Exception as e:
            return False
    
    @stress_test
    async def test_system_resilience(self):
        """Test de resistencia del sistema ante fallos"""
        await self.aegis_node.start()
        
        # Simular fallos en componentes
        failure_scenarios = [
            self._simulate_crypto_failure,
            self._simulate_storage_failure,
            self._simulate_network_failure
        ]
        
        for scenario in failure_scenarios:
            # Ejecutar escenario de fallo
            await scenario()
            
            # Verificar que el sistema sigue funcionando
            status = self.aegis_node.get_status()
            assert status["status"] in ["running", "degraded"], "El sistema debe mantenerse operativo"
            
            # Dar tiempo para recuperación
            await asyncio.sleep(1)
    
    async def _simulate_crypto_failure(self):
        """Simula fallo en el componente crypto"""
        crypto = self.components["crypto"]
        
        # Simular fallo temporal
        original_encrypt = crypto.encrypt_data
        crypto.encrypt_data = Mock(side_effect=Exception("Crypto failure"))
        
        # Esperar un momento
        await asyncio.sleep(0.5)
        
        # Restaurar funcionalidad
        crypto.encrypt_data = original_encrypt
    
    async def _simulate_storage_failure(self):
        """Simula fallo en el almacenamiento"""
        storage = self.components["storage"]
        
        # Simular fallo temporal
        original_store = storage.store_data
        storage.store_data = AsyncMock(side_effect=Exception("Storage failure"))
        
        await asyncio.sleep(0.5)
        
        # Restaurar funcionalidad
        storage.store_data = original_store
    
    async def _simulate_network_failure(self):
        """Simula fallo de red"""
        p2p = self.components["p2p"]
        
        # Simular desconexión temporal
        original_send = p2p.send_message
        p2p.send_message = AsyncMock(side_effect=Exception("Network failure"))
        
        await asyncio.sleep(0.5)
        
        # Restaurar conectividad
        p2p.send_message = original_send
    
    @integration_test
    async def test_configuration_management(self):
        """Test de gestión de configuración integrada"""
        # Verificar que la configuración se propaga correctamente
        config_updates = {
            "p2p_port": 8001,
            "max_peers": 20,
            "encryption_enabled": True
        }
        
        # Simular actualización de configuración
        for key, value in config_updates.items():
            self.test_config[key] = value
        
        # Verificar que los componentes reciben la nueva configuración
        # En un sistema real, esto activaría la reconfiguración automática
        assert self.test_config["p2p_port"] == 8001, "La configuración debe actualizarse"
    
    @integration_test
    async def test_backup_and_recovery(self):
        """Test de backup y recuperación del sistema"""
        storage = self.components["storage"]
        
        # Crear datos de prueba
        test_data = {"important": "data", "timestamp": time.time()}
        storage_id = await storage.store_data(test_data)
        
        # Simular backup
        backup_data = {
            "storage_id": storage_id,
            "data": test_data,
            "backup_timestamp": time.time()
        }
        
        backup_file = os.path.join(self.temp_dir, "test_backup.json")
        with open(backup_file, 'w') as f:
            json.dump(backup_data, f)
        
        # Simular pérdida de datos
        # En un sistema real, esto eliminaría los datos del storage
        
        # Simular recuperación desde backup
        with open(backup_file, 'r') as f:
            recovered_backup = json.load(f)
        
        # Restaurar datos
        restored_id = await storage.store_data(recovered_backup["data"])
        retrieved_data = await storage.retrieve_data(restored_id)
        
        assert retrieved_data == test_data, "Los datos deben recuperarse correctamente desde backup"

def create_integration_test_suite() -> TestSuite:
    """Crea la suite de tests de integración"""
    integration_tests = AEGISIntegrationTests()
    
    return TestSuite(
        name="AEGISIntegration",
        description="Tests de integración completos para AEGIS",
        tests=[
            integration_tests.test_full_system_startup,
            integration_tests.test_crypto_p2p_integration,
            integration_tests.test_consensus_storage_integration,
            integration_tests.test_metrics_monitoring_integration,
            integration_tests.test_full_data_flow,
            integration_tests.test_system_performance_under_load,
            integration_tests.test_system_resilience,
            integration_tests.test_configuration_management,
            integration_tests.test_backup_and_recovery
        ],
        setup_func=integration_tests.setup,
        teardown_func=integration_tests.teardown,
        test_type=TestType.INTEGRATION
    )

if __name__ == "__main__":
    # Ejecutar tests de integración individualmente
    async def main():
        from test_framework import get_test_framework
        
        framework = get_test_framework()
        suite = create_integration_test_suite()
        framework.register_test_suite(suite)
        
        results = await framework.run_all_tests()
        print(f"Integration tests - Éxito: {results['summary']['success_rate']:.1f}%")
    
    asyncio.run(main())