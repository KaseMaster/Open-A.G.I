#!/usr/bin/env python3
"""
AEGIS Test Framework - Sistema de Testing Completo
Implementa tests unitarios, de integración y de rendimiento para todos los componentes de AEGIS.
"""

import asyncio
import unittest
import time
import json
import os
import tempfile
import shutil
from typing import Dict, List, Any, Optional, Callable
from unittest.mock import Mock, patch, AsyncMock
from dataclasses import dataclass
from enum import Enum
import threading
import psutil
import logging
from pathlib import Path

# Importación condicional de pytest para manejar excepciones Skipped
try:
    import pytest
    from _pytest.outcomes import Skipped
    PYTEST_AVAILABLE = True
except ImportError:
    PYTEST_AVAILABLE = False
    # Crear una clase dummy para Skipped si pytest no está disponible
    class Skipped(Exception):
        pass

# Configurar logging para tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestType(Enum):
    """Tipos de tests disponibles"""
    UNIT = "unit"
    INTEGRATION = "integration"
    PERFORMANCE = "performance"
    SECURITY = "security"
    STRESS = "stress"

class TestStatus(Enum):
    """Estados de los tests"""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"

@dataclass
class TestResult:
    """Resultado de un test individual"""
    name: str
    test_type: TestType
    status: TestStatus
    duration: float
    error_message: Optional[str] = None
    performance_metrics: Optional[Dict[str, Any]] = None
    coverage_percentage: Optional[float] = None

@dataclass
class TestSuite:
    """Suite de tests"""
    name: str
    description: str
    tests: List[Callable]
    setup_func: Optional[Callable] = None
    teardown_func: Optional[Callable] = None
    test_type: TestType = TestType.UNIT

class PerformanceMonitor:
    """Monitor de rendimiento para tests"""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.cpu_usage = []
        self.memory_usage = []
        self.monitoring = False
        self.monitor_thread = None
    
    def start_monitoring(self):
        """Inicia el monitoreo de rendimiento"""
        self.start_time = time.time()
        self.monitoring = True
        self.cpu_usage = []
        self.memory_usage = []
        
        def monitor():
            while self.monitoring:
                self.cpu_usage.append(psutil.cpu_percent())
                self.memory_usage.append(psutil.virtual_memory().percent)
                time.sleep(0.1)
        
        self.monitor_thread = threading.Thread(target=monitor)
        self.monitor_thread.start()
    
    def stop_monitoring(self) -> Dict[str, Any]:
        """Detiene el monitoreo y retorna métricas"""
        self.end_time = time.time()
        self.monitoring = False
        
        if self.monitor_thread:
            self.monitor_thread.join()
        
        return {
            "duration": self.end_time - self.start_time,
            "avg_cpu_usage": sum(self.cpu_usage) / len(self.cpu_usage) if self.cpu_usage else 0,
            "max_cpu_usage": max(self.cpu_usage) if self.cpu_usage else 0,
            "avg_memory_usage": sum(self.memory_usage) / len(self.memory_usage) if self.memory_usage else 0,
            "max_memory_usage": max(self.memory_usage) if self.memory_usage else 0
        }

class MockAEGISComponents:
    """Mocks para componentes de AEGIS"""
    
    @staticmethod
    def mock_tor_integration():
        """Mock del módulo TOR"""
        mock = Mock()
        mock.start_tor = AsyncMock(return_value=True)
        mock.stop_tor = AsyncMock(return_value=True)
        mock.get_tor_status = Mock(return_value={"status": "running", "circuits": 3})
        return mock
    
    @staticmethod
    def mock_p2p_network():
        """Mock de la red P2P"""
        mock = Mock()
        mock.start_p2p = AsyncMock(return_value=True)
        mock.discover_peers = AsyncMock(return_value=["peer1", "peer2"])
        mock.send_message = AsyncMock(return_value=True)
        return mock
    
    @staticmethod
    def mock_crypto_framework():
        """Mock del framework criptográfico"""
        mock = Mock()
        mock.encrypt_data = Mock(return_value=b"encrypted_data")
        mock.decrypt_data = Mock(return_value=b"decrypted_data")
        mock.generate_key = Mock(return_value="test_key")
        return mock
    
    @staticmethod
    def mock_metrics_collector():
        """Mock del colector de métricas"""
        mock = Mock()
        mock.collect_metrics = AsyncMock(return_value={"cpu": 50, "memory": 60})
        mock.store_metrics = AsyncMock(return_value=True)
        return mock

class AEGISTestFramework:
    """Framework principal de testing para AEGIS"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._default_config()
        self.test_suites: List[TestSuite] = []
        self.results: List[TestResult] = []
        self.temp_dir = None
        self.performance_monitor = PerformanceMonitor()
        
    def _default_config(self) -> Dict[str, Any]:
        """Configuración por defecto para tests"""
        return {
            "test_timeout": 30,
            "performance_threshold": {
                "max_cpu_usage": 80,
                "max_memory_usage": 70,
                "max_response_time": 1.0
            },
            "coverage_threshold": 80,
            "parallel_execution": True,
            "generate_reports": True,
            "report_format": ["json", "html"],
            "mock_external_services": True
        }
    
    def setup_test_environment(self):
        """Configura el entorno de testing"""
        # Crear directorio temporal
        self.temp_dir = tempfile.mkdtemp(prefix="aegis_test_")
        
        # Configurar variables de entorno para tests
        os.environ["AEGIS_TEST_MODE"] = "true"
        os.environ["AEGIS_TEST_DIR"] = self.temp_dir
        
        logger.info(f"Entorno de test configurado en: {self.temp_dir}")
    
    def cleanup_test_environment(self):
        """Limpia el entorno de testing"""
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        
        # Limpiar variables de entorno
        os.environ.pop("AEGIS_TEST_MODE", None)
        os.environ.pop("AEGIS_TEST_DIR", None)
        
        logger.info("Entorno de test limpiado")
    
    def register_test_suite(self, suite: TestSuite):
        """Registra una suite de tests"""
        self.test_suites.append(suite)
        logger.info(f"Suite registrada: {suite.name} ({len(suite.tests)} tests)")
    
    async def run_single_test(self, test_func: Callable, test_name: str, test_type: TestType) -> TestResult:
        """Ejecuta un test individual"""
        logger.info(f"Ejecutando test: {test_name}")
        
        # Iniciar monitoreo de rendimiento
        self.performance_monitor.start_monitoring()
        
        start_time = time.time()
        status = TestStatus.PENDING
        error_message = None
        
        try:
            # Ejecutar el test
            if asyncio.iscoroutinefunction(test_func):
                await asyncio.wait_for(test_func(), timeout=self.config["test_timeout"])
            else:
                test_func()
            
            status = TestStatus.PASSED
            
        except AssertionError as e:
            status = TestStatus.FAILED
            error_message = str(e)
            
        except asyncio.TimeoutError:
            status = TestStatus.ERROR
            error_message = f"Test timeout después de {self.config['test_timeout']} segundos"
        
        # Capturar específicamente la excepción Skipped de pytest
        except Skipped as e:
            status = TestStatus.SKIPPED
            error_message = str(e)
        
        except Exception as e:
            status = TestStatus.ERROR
            error_message = f"Error inesperado: {str(e)}"
        
        # Detener monitoreo
        performance_metrics = self.performance_monitor.stop_monitoring()
        duration = time.time() - start_time
        
        result = TestResult(
            name=test_name,
            test_type=test_type,
            status=status,
            duration=duration,
            error_message=error_message,
            performance_metrics=performance_metrics
        )
        
        self.results.append(result)
        return result
    
    async def run_test_suite(self, suite: TestSuite) -> List[TestResult]:
        """Ejecuta una suite completa de tests"""
        logger.info(f"Ejecutando suite: {suite.name}")
        suite_results = []
        
        # Ejecutar setup si existe
        if suite.setup_func:
            try:
                if asyncio.iscoroutinefunction(suite.setup_func):
                    await suite.setup_func()
                else:
                    suite.setup_func()
            except Exception as e:
                logger.error(f"Error en setup de suite {suite.name}: {e}")
                return suite_results
        
        # Ejecutar tests
        for i, test_func in enumerate(suite.tests):
            test_name = f"{suite.name}.{test_func.__name__}"
            result = await self.run_single_test(test_func, test_name, suite.test_type)
            suite_results.append(result)
        
        # Ejecutar teardown si existe
        if suite.teardown_func:
            try:
                if asyncio.iscoroutinefunction(suite.teardown_func):
                    await suite.teardown_func()
                else:
                    suite.teardown_func()
            except Exception as e:
                logger.error(f"Error en teardown de suite {suite.name}: {e}")
        
        return suite_results
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Ejecuta todos los tests registrados"""
        logger.info("Iniciando ejecución de todos los tests")
        
        self.setup_test_environment()
        
        try:
            start_time = time.time()
            
            # Ejecutar todas las suites
            for suite in self.test_suites:
                await self.run_test_suite(suite)
            
            total_duration = time.time() - start_time
            
            # Generar estadísticas
            stats = self._generate_statistics(total_duration)
            
            # Generar reportes si está habilitado
            if self.config["generate_reports"]:
                self._generate_reports(stats)
            
            return stats
            
        finally:
            self.cleanup_test_environment()
    
    def _generate_statistics(self, total_duration: float) -> Dict[str, Any]:
        """Genera estadísticas de los tests ejecutados"""
        total_tests = len(self.results)
        passed = len([r for r in self.results if r.status == TestStatus.PASSED])
        failed = len([r for r in self.results if r.status == TestStatus.FAILED])
        errors = len([r for r in self.results if r.status == TestStatus.ERROR])
        
        # Estadísticas de rendimiento
        avg_duration = sum(r.duration for r in self.results) / total_tests if total_tests > 0 else 0
        max_duration = max((r.duration for r in self.results), default=0)
        
        # Estadísticas por tipo de test
        by_type = {}
        for test_type in TestType:
            type_results = [r for r in self.results if r.test_type == test_type]
            by_type[test_type.value] = {
                "total": len(type_results),
                "passed": len([r for r in type_results if r.status == TestStatus.PASSED]),
                "failed": len([r for r in type_results if r.status == TestStatus.FAILED])
            }
        
        return {
            "summary": {
                "total_tests": total_tests,
                "passed": passed,
                "failed": failed,
                "errors": errors,
                "success_rate": (passed / total_tests * 100) if total_tests > 0 else 0,
                "total_duration": total_duration,
                "avg_test_duration": avg_duration,
                "max_test_duration": max_duration
            },
            "by_type": by_type,
            "failed_tests": [
                {
                    "name": r.name,
                    "error": r.error_message,
                    "duration": r.duration
                }
                for r in self.results if r.status in [TestStatus.FAILED, TestStatus.ERROR]
            ],
            "performance_issues": [
                {
                    "name": r.name,
                    "duration": r.duration,
                    "cpu_usage": r.performance_metrics.get("max_cpu_usage", 0) if r.performance_metrics else 0
                }
                for r in self.results 
                if r.performance_metrics and (
                    r.duration > self.config["performance_threshold"]["max_response_time"] or
                    r.performance_metrics.get("max_cpu_usage", 0) > self.config["performance_threshold"]["max_cpu_usage"]
                )
            ]
        }
    
    def _generate_reports(self, stats: Dict[str, Any]):
        """Genera reportes de testing"""
        report_dir = Path(self.temp_dir) / "reports"
        report_dir.mkdir(exist_ok=True)
        
        # Reporte JSON
        if "json" in self.config["report_format"]:
            json_report = report_dir / "test_report.json"
            with open(json_report, 'w') as f:
                json.dump({
                    "statistics": stats,
                    "detailed_results": [
                        {
                            "name": r.name,
                            "type": r.test_type.value,
                            "status": r.status.value,
                            "duration": r.duration,
                            "error": r.error_message,
                            "performance": r.performance_metrics
                        }
                        for r in self.results
                    ]
                }, f, indent=2)
        
        # Reporte HTML
        if "html" in self.config["report_format"]:
            html_report = report_dir / "test_report.html"
            self._generate_html_report(html_report, stats)
        
        logger.info(f"Reportes generados en: {report_dir}")
    
    def _generate_html_report(self, file_path: Path, stats: Dict[str, Any]):
        """Genera reporte HTML"""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>AEGIS Test Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .summary {{ background: #f5f5f5; padding: 15px; border-radius: 5px; }}
                .passed {{ color: green; }}
                .failed {{ color: red; }}
                .error {{ color: orange; }}
                table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>AEGIS Test Report</h1>
            
            <div class="summary">
                <h2>Resumen</h2>
                <p>Total de Tests: {stats['summary']['total_tests']}</p>
                <p class="passed">Exitosos: {stats['summary']['passed']}</p>
                <p class="failed">Fallidos: {stats['summary']['failed']}</p>
                <p class="error">Errores: {stats['summary']['errors']}</p>
                <p>Tasa de Éxito: {stats['summary']['success_rate']:.1f}%</p>
                <p>Duración Total: {stats['summary']['total_duration']:.2f}s</p>
            </div>
            
            <h2>Tests Fallidos</h2>
            <table>
                <tr><th>Test</th><th>Error</th><th>Duración</th></tr>
        """
        
        for failed_test in stats['failed_tests']:
            html_content += f"""
                <tr>
                    <td>{failed_test['name']}</td>
                    <td>{failed_test['error']}</td>
                    <td>{failed_test['duration']:.2f}s</td>
                </tr>
            """
        
        html_content += """
            </table>
        </body>
        </html>
        """
        
        with open(file_path, 'w') as f:
            f.write(html_content)

# Instancia global del framework
_test_framework = None

def get_test_framework(config: Optional[Dict[str, Any]] = None) -> AEGISTestFramework:
    """Obtiene la instancia del framework de testing"""
    global _test_framework
    if _test_framework is None:
        _test_framework = AEGISTestFramework(config)
    return _test_framework

async def start_test_framework(config: Optional[Dict[str, Any]] = None):
    """Inicia el framework de testing"""
    framework = get_test_framework(config)
    logger.info("Framework de testing AEGIS iniciado")
    return framework

async def stop_test_framework():
    """Detiene el framework de testing"""
    global _test_framework
    if _test_framework:
        _test_framework.cleanup_test_environment()
        _test_framework = None
    logger.info("Framework de testing AEGIS detenido")

def test_decorator(test_type: TestType = TestType.UNIT):
    """Decorador para marcar funciones como tests"""
    def decorator(func):
        func._aegis_test = True
        func._aegis_test_type = test_type
        return func
    return decorator

# Decoradores específicos para tipos de test
unit_test = test_decorator(TestType.UNIT)
integration_test = test_decorator(TestType.INTEGRATION)
performance_test = test_decorator(TestType.PERFORMANCE)
security_test = test_decorator(TestType.SECURITY)
stress_test = test_decorator(TestType.STRESS)

if __name__ == "__main__":
    # Ejemplo de uso
    async def main():
        framework = await start_test_framework()
        
        # Registrar suites de test (se implementarán en archivos separados)
        # framework.register_test_suite(crypto_test_suite)
        # framework.register_test_suite(p2p_test_suite)
        
        # Ejecutar todos los tests
        results = await framework.run_all_tests()
        
        print(f"Tests ejecutados: {results['summary']['total_tests']}")
        print(f"Tasa de éxito: {results['summary']['success_rate']:.1f}%")
        
        await stop_test_framework()
    
    asyncio.run(main())