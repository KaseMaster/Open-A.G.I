#!/usr/bin/env python3
"""
Tests de Rendimiento para AEGIS
Evalúa el desempeño del sistema bajo diferentes condiciones de carga.
"""

import asyncio
import unittest
import pytest
import time
import psutil
import threading
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import Mock, patch, AsyncMock
import sys
import os
import json
import tempfile
import shutil

# Agregar el directorio padre al path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from test_framework import (
    AEGISTestFramework, TestSuite, TestType, TestStatus,
    performance_test, stress_test, PerformanceMonitor
)

class AEGISPerformanceTests:
    """Tests de rendimiento para AEGIS"""
    
    def __init__(self):
        self.performance_monitor = PerformanceMonitor()
        self.temp_dir = None
        self.components = {}
        self.baseline_metrics = {}
    
    def setup(self):
        """Setup para tests de rendimiento"""
        self.temp_dir = tempfile.mkdtemp(prefix="aegis_perf_")
        self._setup_components()
        self._establish_baseline()
    
    def teardown(self):
        """Cleanup después de tests"""
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def _setup_components(self):
        """Configura componentes para testing de rendimiento"""
        # Configuraciones optimizadas para rendimiento
        configs = {
            "crypto": {
                "key_size": 256,
                "algorithm": "AES-256-GCM",
                "cache_size": 1000,
                "parallel_operations": True
            },
            "p2p": {
                "port": 8000,
                "max_peers": 100,
                "message_buffer_size": 10000,
                "connection_pool_size": 50
            },
            "storage": {
                "type": "memory",  # Más rápido para tests
                "cache_size": 10000,
                "batch_size": 100
            },
            "consensus": {
                "algorithm": "PBFT",
                "timeout": 1,  # Timeout reducido para tests
                "batch_size": 50
            }
        }
        
        for component, config in configs.items():
            self.components[component] = self._create_performance_component(component, config)
    
    def _create_performance_component(self, component_name: str, config: dict):
        """Crea componente optimizado para rendimiento"""
        mock = Mock()
        
        if component_name == "crypto":
            # Simular operaciones crypto con tiempos realistas
            mock.encrypt_data = Mock(side_effect=self._simulate_crypto_encrypt)
            mock.decrypt_data = Mock(side_effect=self._simulate_crypto_decrypt)
            mock.generate_key = Mock(side_effect=self._simulate_key_generation)
            mock.hash_data = Mock(side_effect=self._simulate_hashing)
            
        elif component_name == "p2p":
            mock.send_message = AsyncMock(side_effect=self._simulate_p2p_send)
            mock.broadcast_message = AsyncMock(side_effect=self._simulate_p2p_broadcast)
            mock.discover_peers = AsyncMock(side_effect=self._simulate_peer_discovery)
            
        elif component_name == "storage":
            mock.store_data = AsyncMock(side_effect=self._simulate_storage_write)
            mock.retrieve_data = AsyncMock(side_effect=self._simulate_storage_read)
            mock.batch_store = AsyncMock(side_effect=self._simulate_batch_storage)
            
        elif component_name == "consensus":
            mock.propose_value = AsyncMock(side_effect=self._simulate_consensus_proposal)
            mock.reach_consensus = AsyncMock(side_effect=self._simulate_consensus_completion)
        
        return mock
    
    def _simulate_crypto_encrypt(self, data: bytes, key: str) -> bytes:
        """Simula encriptación con tiempo realista"""
        # Simular tiempo de procesamiento basado en tamaño de datos
        processing_time = len(data) * 0.000001  # 1 microsegundo por byte
        time.sleep(processing_time)
        return b"encrypted_" + data
    
    def _simulate_crypto_decrypt(self, data: bytes, key: str) -> bytes:
        """Simula desencriptación con tiempo realista"""
        processing_time = len(data) * 0.000001
        time.sleep(processing_time)
        return data[10:]  # Remover prefijo "encrypted_"
    
    def _simulate_key_generation(self, key_size: int) -> str:
        """Simula generación de claves"""
        # Tiempo basado en tamaño de clave
        processing_time = key_size * 0.00001
        time.sleep(processing_time)
        return f"key_{key_size}_{time.time()}"
    
    def _simulate_hashing(self, data: bytes) -> str:
        """Simula hashing"""
        processing_time = len(data) * 0.0000005
        time.sleep(processing_time)
        return f"hash_{hash(data)}"
    
    async def _simulate_p2p_send(self, peer_id: str, message: dict) -> bool:
        """Simula envío P2P"""
        # Simular latencia de red
        await asyncio.sleep(0.001)  # 1ms de latencia
        return True
    
    async def _simulate_p2p_broadcast(self, message: dict) -> bool:
        """Simula broadcast P2P"""
        # Tiempo proporcional al número de peers
        peer_count = 10  # Simular 10 peers
        await asyncio.sleep(0.001 * peer_count)
        return True
    
    async def _simulate_peer_discovery(self) -> list:
        """Simula descubrimiento de peers"""
        await asyncio.sleep(0.1)  # 100ms para discovery
        return [f"peer_{i}" for i in range(10)]
    
    async def _simulate_storage_write(self, data: dict) -> str:
        """Simula escritura en storage"""
        # Tiempo basado en tamaño de datos
        data_size = len(json.dumps(data))
        processing_time = data_size * 0.000001
        await asyncio.sleep(processing_time)
        return f"storage_id_{time.time()}"
    
    async def _simulate_storage_read(self, storage_id: str) -> dict:
        """Simula lectura de storage"""
        await asyncio.sleep(0.0005)  # 0.5ms para lectura
        return {"data": f"retrieved_for_{storage_id}"}
    
    async def _simulate_batch_storage(self, data_list: list) -> list:
        """Simula almacenamiento en lote"""
        # Más eficiente que operaciones individuales
        batch_time = len(data_list) * 0.0001
        await asyncio.sleep(batch_time)
        return [f"batch_id_{i}" for i in range(len(data_list))]
    
    async def _simulate_consensus_proposal(self, value: dict) -> bool:
        """Simula propuesta de consenso"""
        # Tiempo variable según complejidad
        await asyncio.sleep(0.01)  # 10ms para propuesta
        return True
    
    async def _simulate_consensus_completion(self, proposal_id: str) -> dict:
        """Simula completar consenso"""
        await asyncio.sleep(0.05)  # 50ms para consenso completo
        return {"status": "accepted", "proposal_id": proposal_id}
    
    def _establish_baseline(self):
        """Establece métricas baseline del sistema"""
        self.baseline_metrics = {
            "cpu_usage": psutil.cpu_percent(interval=1),
            "memory_usage": psutil.virtual_memory().percent,
            "disk_io": psutil.disk_io_counters()._asdict() if psutil.disk_io_counters() else {},
            "network_io": psutil.net_io_counters()._asdict() if psutil.net_io_counters() else {}
        }
    
    @performance_test
    async def test_crypto_operations_throughput(self):
        """Test de throughput de operaciones criptográficas"""
        crypto = self.components["crypto"]
        
        # Test de encriptación
        test_data = b"Test data for encryption performance" * 100  # ~3.7KB
        iterations = 1000
        
        start_time = time.time()
        for i in range(iterations):
            key = crypto.generate_key(256)
            encrypted = crypto.encrypt_data(test_data, key)
            decrypted = crypto.decrypt_data(encrypted, key)
        
        total_time = time.time() - start_time
        throughput = iterations / total_time
        
        # Métricas esperadas
        assert throughput > 500, f"Throughput crypto debe ser > 500 ops/seg (actual: {throughput:.1f})"
        assert total_time < 5, f"Operaciones crypto deben completarse en < 5s (actual: {total_time:.2f}s)"
        
        return {
            "throughput_ops_per_sec": throughput,
            "total_time_seconds": total_time,
            "operations_count": iterations
        }
    
    @performance_test
    async def test_p2p_message_throughput(self):
        """Test de throughput de mensajes P2P"""
        p2p = self.components["p2p"]
        
        message_count = 1000
        message_size = 1024  # 1KB por mensaje
        
        test_message = {
            "type": "performance_test",
            "data": "x" * message_size,
            "timestamp": time.time()
        }
        
        # Test de envío individual
        start_time = time.time()
        tasks = []
        for i in range(message_count):
            task = p2p.send_message(f"peer_{i % 10}", test_message)
            tasks.append(task)
        
        await asyncio.gather(*tasks)
        total_time = time.time() - start_time
        
        throughput = message_count / total_time
        bandwidth = (message_count * message_size) / total_time / 1024  # KB/s
        
        assert throughput > 100, f"Throughput P2P debe ser > 100 msg/seg (actual: {throughput:.1f})"
        assert bandwidth > 100, f"Bandwidth debe ser > 100 KB/s (actual: {bandwidth:.1f})"
        
        return {
            "message_throughput": throughput,
            "bandwidth_kbps": bandwidth,
            "total_messages": message_count,
            "total_time": total_time
        }
    
    @performance_test
    async def test_storage_operations_performance(self):
        """Test de rendimiento de operaciones de almacenamiento"""
        storage = self.components["storage"]
        
        # Test de escritura
        write_count = 1000
        test_data = {"performance_test": True, "data": "x" * 512}  # ~512 bytes
        
        write_start = time.time()
        storage_ids = []
        for i in range(write_count):
            storage_id = await storage.store_data({**test_data, "id": i})
            storage_ids.append(storage_id)
        
        write_time = time.time() - write_start
        write_throughput = write_count / write_time
        
        # Test de lectura
        read_start = time.time()
        read_tasks = [storage.retrieve_data(sid) for sid in storage_ids]
        retrieved_data = await asyncio.gather(*read_tasks)
        read_time = time.time() - read_start
        read_throughput = len(retrieved_data) / read_time
        
        # Test de escritura en lote
        batch_data = [{"batch_test": True, "id": i} for i in range(100)]
        batch_start = time.time()
        batch_ids = await storage.batch_store(batch_data)
        batch_time = time.time() - batch_start
        batch_throughput = len(batch_data) / batch_time
        
        assert write_throughput > 200, f"Write throughput debe ser > 200 ops/seg (actual: {write_throughput:.1f})"
        assert read_throughput > 500, f"Read throughput debe ser > 500 ops/seg (actual: {read_throughput:.1f})"
        assert batch_throughput > write_throughput, "Batch operations deben ser más eficientes"
        
        return {
            "write_throughput": write_throughput,
            "read_throughput": read_throughput,
            "batch_throughput": batch_throughput,
            "write_time": write_time,
            "read_time": read_time,
            "batch_time": batch_time
        }
    
    @performance_test
    async def test_consensus_performance(self):
        """Test de rendimiento del sistema de consenso"""
        consensus = self.components["consensus"]
        
        # Test de propuestas individuales
        proposal_count = 100
        proposals = [{"proposal_id": i, "data": f"proposal_{i}"} for i in range(proposal_count)]
        
        start_time = time.time()
        proposal_tasks = [consensus.propose_value(proposal) for proposal in proposals]
        proposal_results = await asyncio.gather(*proposal_tasks)
        proposal_time = time.time() - start_time
        
        successful_proposals = sum(1 for result in proposal_results if result)
        proposal_throughput = successful_proposals / proposal_time
        
        # Test de consenso completo
        consensus_start = time.time()
        consensus_tasks = [consensus.reach_consensus(f"proposal_{i}") for i in range(50)]
        consensus_results = await asyncio.gather(*consensus_tasks)
        consensus_time = time.time() - consensus_start
        
        consensus_throughput = len(consensus_results) / consensus_time
        
        assert proposal_throughput > 10, f"Proposal throughput debe ser > 10 ops/seg (actual: {proposal_throughput:.1f})"
        assert consensus_throughput > 5, f"Consensus throughput debe ser > 5 ops/seg (actual: {consensus_throughput:.1f})"
        
        return {
            "proposal_throughput": proposal_throughput,
            "consensus_throughput": consensus_throughput,
            "successful_proposals": successful_proposals,
            "total_proposals": proposal_count
        }
    
    @stress_test
    async def test_concurrent_operations_stress(self):
        """Test de estrés con operaciones concurrentes"""
        crypto = self.components["crypto"]
        p2p = self.components["p2p"]
        storage = self.components["storage"]
        
        # Configurar carga de trabajo mixta
        concurrent_operations = 500
        operation_types = ["crypto", "p2p", "storage"]
        
        async def mixed_operation(operation_id: int):
            """Operación mixta que usa múltiples componentes"""
            op_type = operation_types[operation_id % len(operation_types)]
            
            try:
                if op_type == "crypto":
                    data = f"stress_test_data_{operation_id}".encode()
                    key = crypto.generate_key(256)
                    encrypted = crypto.encrypt_data(data, key)
                    decrypted = crypto.decrypt_data(encrypted, key)
                    return decrypted == data
                
                elif op_type == "p2p":
                    message = {"stress_test": True, "id": operation_id}
                    result = await p2p.send_message(f"peer_{operation_id % 10}", message)
                    return result
                
                elif op_type == "storage":
                    data = {"stress_test": True, "operation_id": operation_id}
                    storage_id = await storage.store_data(data)
                    retrieved = await storage.retrieve_data(storage_id)
                    return retrieved is not None
                
            except Exception as e:
                return False
        
        # Ejecutar operaciones concurrentes
        start_time = time.time()
        start_cpu = psutil.cpu_percent()
        start_memory = psutil.virtual_memory().percent
        
        tasks = [mixed_operation(i) for i in range(concurrent_operations)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        end_time = time.time()
        end_cpu = psutil.cpu_percent()
        end_memory = psutil.virtual_memory().percent
        
        # Analizar resultados
        successful_ops = sum(1 for r in results if r is True)
        failed_ops = sum(1 for r in results if isinstance(r, Exception))
        success_rate = successful_ops / concurrent_operations
        total_time = end_time - start_time
        throughput = successful_ops / total_time
        
        # Métricas de recursos
        cpu_increase = end_cpu - start_cpu
        memory_increase = end_memory - start_memory
        
        assert success_rate > 0.95, f"Success rate debe ser > 95% (actual: {success_rate:.1%})"
        assert throughput > 50, f"Throughput bajo estrés debe ser > 50 ops/seg (actual: {throughput:.1f})"
        assert cpu_increase < 50, f"Incremento de CPU debe ser < 50% (actual: {cpu_increase:.1f}%)"
        assert memory_increase < 20, f"Incremento de memoria debe ser < 20% (actual: {memory_increase:.1f}%)"
        
        return {
            "success_rate": success_rate,
            "throughput": throughput,
            "total_operations": concurrent_operations,
            "successful_operations": successful_ops,
            "failed_operations": failed_ops,
            "cpu_increase": cpu_increase,
            "memory_increase": memory_increase,
            "total_time": total_time
        }
    
    @performance_test
    async def test_memory_usage_efficiency(self):
        """Test de eficiencia en uso de memoria"""
        import gc
        
        # Medición inicial
        gc.collect()
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        # Crear carga de trabajo que use memoria
        crypto = self.components["crypto"]
        storage = self.components["storage"]
        
        # Operaciones que consumen memoria
        large_data_operations = 1000
        large_data = b"x" * 10240  # 10KB por operación
        
        stored_ids = []
        for i in range(large_data_operations):
            # Encriptar datos grandes
            key = crypto.generate_key(256)
            encrypted = crypto.encrypt_data(large_data, key)
            
            # Almacenar
            storage_id = await storage.store_data({
                "encrypted_data": encrypted,
                "key": key,
                "operation_id": i
            })
            stored_ids.append(storage_id)
            
            # Forzar garbage collection cada 100 operaciones
            if i % 100 == 0:
                gc.collect()
        
        # Medición después de operaciones
        peak_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        # Limpiar datos
        for storage_id in stored_ids:
            await storage.retrieve_data(storage_id)
        
        gc.collect()
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        # Calcular métricas
        memory_increase = peak_memory - initial_memory
        memory_per_operation = memory_increase / large_data_operations
        memory_cleanup_efficiency = (peak_memory - final_memory) / memory_increase
        
        assert memory_per_operation < 0.1, f"Memoria por operación debe ser < 0.1MB (actual: {memory_per_operation:.3f}MB)"
        assert memory_cleanup_efficiency > 0.8, f"Eficiencia de limpieza debe ser > 80% (actual: {memory_cleanup_efficiency:.1%})"
        
        return {
            "initial_memory_mb": initial_memory,
            "peak_memory_mb": peak_memory,
            "final_memory_mb": final_memory,
            "memory_increase_mb": memory_increase,
            "memory_per_operation_mb": memory_per_operation,
            "cleanup_efficiency": memory_cleanup_efficiency
        }
    
    @performance_test
    async def test_latency_distribution(self):
        """Test de distribución de latencia"""
        p2p = self.components["p2p"]
        
        # Medir latencias de operaciones individuales
        latencies = []
        operation_count = 1000
        
        for i in range(operation_count):
            start = time.time()
            await p2p.send_message(f"peer_{i % 10}", {"latency_test": i})
            end = time.time()
            latencies.append((end - start) * 1000)  # Convertir a ms
        
        # Calcular estadísticas de latencia
        avg_latency = statistics.mean(latencies)
        median_latency = statistics.median(latencies)
        p95_latency = sorted(latencies)[int(0.95 * len(latencies))]
        p99_latency = sorted(latencies)[int(0.99 * len(latencies))]
        max_latency = max(latencies)
        min_latency = min(latencies)
        
        # Verificar que las latencias están dentro de rangos aceptables
        assert avg_latency < 10, f"Latencia promedio debe ser < 10ms (actual: {avg_latency:.2f}ms)"
        assert p95_latency < 20, f"P95 latencia debe ser < 20ms (actual: {p95_latency:.2f}ms)"
        assert p99_latency < 50, f"P99 latencia debe ser < 50ms (actual: {p99_latency:.2f}ms)"
        
        return {
            "avg_latency_ms": avg_latency,
            "median_latency_ms": median_latency,
            "p95_latency_ms": p95_latency,
            "p99_latency_ms": p99_latency,
            "max_latency_ms": max_latency,
            "min_latency_ms": min_latency,
            "total_operations": operation_count
        }
    
    @stress_test
    async def test_long_running_stability(self):
        """Test de estabilidad a largo plazo"""
        # Simular operaciones continuas durante un período extendido
        duration_seconds = 30  # 30 segundos para el test
        operation_interval = 0.1  # Una operación cada 100ms
        
        crypto = self.components["crypto"]
        storage = self.components["storage"]
        
        start_time = time.time()
        operations_completed = 0
        errors_encountered = 0
        
        # Métricas de recursos iniciales
        initial_cpu = psutil.cpu_percent()
        initial_memory = psutil.virtual_memory().percent
        
        while (time.time() - start_time) < duration_seconds:
            try:
                # Operación mixta
                data = f"stability_test_{operations_completed}".encode()
                key = crypto.generate_key(256)
                encrypted = crypto.encrypt_data(data, key)
                
                storage_data = {
                    "encrypted": encrypted,
                    "timestamp": time.time(),
                    "operation_id": operations_completed
                }
                
                storage_id = await storage.store_data(storage_data)
                retrieved = await storage.retrieve_data(storage_id)
                
                operations_completed += 1
                
            except Exception as e:
                errors_encountered += 1
            
            await asyncio.sleep(operation_interval)
        
        # Métricas finales
        final_cpu = psutil.cpu_percent()
        final_memory = psutil.virtual_memory().percent
        
        total_time = time.time() - start_time
        avg_throughput = operations_completed / total_time
        error_rate = errors_encountered / (operations_completed + errors_encountered)
        
        cpu_stability = abs(final_cpu - initial_cpu)
        memory_stability = abs(final_memory - initial_memory)
        
        assert error_rate < 0.01, f"Error rate debe ser < 1% (actual: {error_rate:.2%})"
        assert avg_throughput > 5, f"Throughput promedio debe ser > 5 ops/seg (actual: {avg_throughput:.1f})"
        assert cpu_stability < 20, f"CPU debe mantenerse estable ±20% (variación: {cpu_stability:.1f}%)"
        assert memory_stability < 10, f"Memoria debe mantenerse estable ±10% (variación: {memory_stability:.1f}%)"
        
        return {
            "duration_seconds": total_time,
            "operations_completed": operations_completed,
            "errors_encountered": errors_encountered,
            "error_rate": error_rate,
            "avg_throughput": avg_throughput,
            "cpu_stability": cpu_stability,
            "memory_stability": memory_stability
        }

def create_performance_test_suite() -> TestSuite:
    """Crea la suite de tests de rendimiento"""
    performance_tests = AEGISPerformanceTests()
    
    return TestSuite(
        name="AEGISPerformance",
        description="Tests de rendimiento y estrés para AEGIS",
        tests=[
            performance_tests.test_crypto_operations_throughput,
            performance_tests.test_p2p_message_throughput,
            performance_tests.test_storage_operations_performance,
            performance_tests.test_consensus_performance,
            performance_tests.test_concurrent_operations_stress,
            performance_tests.test_memory_usage_efficiency,
            performance_tests.test_latency_distribution,
            performance_tests.test_long_running_stability
        ],
        setup_func=performance_tests.setup,
        teardown_func=performance_tests.teardown,
        test_type=TestType.PERFORMANCE
    )

if __name__ == "__main__":
    # Ejecutar tests de rendimiento individualmente
    async def main():
        from test_framework import get_test_framework
        
        framework = get_test_framework()
        suite = create_performance_test_suite()
        framework.register_test_suite(suite)
        
        results = await framework.run_all_tests()
        print(f"Performance tests - Éxito: {results['summary']['success_rate']:.1f}%")
        
        # Mostrar métricas de rendimiento
        for test_name, result in results['results'].items():
            if result.metrics:
                print(f"\n{test_name} - Métricas:")
                for metric, value in result.metrics.items():
                    print(f"  {metric}: {value}")
    
    asyncio.run(main())