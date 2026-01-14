#!/usr/bin/env python3
"""
AEGIS Framework - Load Testing Suite
Pruebas de carga completas para el sistema integrado
"""

import asyncio
import time
import statistics
import json
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import aiohttp
import websockets
import threading
import queue

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class LoadTestResult:
    """Resultado de una prueba de carga"""
    test_name: str
    duration: float
    total_requests: int
    successful_requests: int
    failed_requests: int
    avg_response_time: float
    min_response_time: float
    max_response_time: float
    p95_response_time: float
    p99_response_time: float
    requests_per_second: float
    error_rate: float
    response_times: List[float]
    errors: List[str]

class LoadTester:
    """Probador de carga para AEGIS Framework"""

    def __init__(self, base_url: str = "http://localhost:8080"):
        self.base_url = base_url
        self.results: List[LoadTestResult] = []
        self.executor = ThreadPoolExecutor(max_workers=50)

    async def run_all_tests(self) -> Dict[str, Any]:
        """Ejecutar todas las pruebas de carga"""
        print("🔥 AEGIS Framework - Suite de Pruebas de Carga")
        print("=" * 60)

        test_configs = [
            # API REST tests
            {"name": "API Health Check", "endpoint": "/api/health", "method": "GET",
             "concurrency": 50, "duration": 30, "target_rps": 100},

            {"name": "API Metrics", "endpoint": "/api/metrics", "method": "GET",
             "concurrency": 30, "duration": 20, "target_rps": 50},

            {"name": "API Nodes", "endpoint": "/api/nodes", "method": "GET",
             "concurrency": 20, "duration": 15, "target_rps": 30},

            # Consensus simulation
            {"name": "Consensus Proposals", "endpoint": "/api/consensus/propose", "method": "POST",
             "concurrency": 10, "duration": 60, "target_rps": 5, "payload": {"type": "test_proposal"}},

            # P2P message simulation
            {"name": "P2P Messages", "endpoint": "/api/p2p/broadcast", "method": "POST",
             "concurrency": 25, "duration": 45, "target_rps": 20, "payload": {"message": "load_test"}},

            # State persistence tests
            {"name": "State Checkpoints", "endpoint": "/api/state/checkpoint", "method": "POST",
             "concurrency": 5, "duration": 30, "target_rps": 2, "payload": {"state_type": "test"}},
        ]

        # Ejecutar pruebas secuencialmente para evitar interferencia
        for config in test_configs:
            print(f"\n🚀 Ejecutando prueba: {config['name']}")
            result = await self.run_load_test(**config)
            self.results.append(result)
            self._print_test_result(result)

        # Prueba de estrés del sistema completo
        print("
💥 Ejecutando prueba de estrés del sistema completo..."        stress_result = await self.run_stress_test()
        self.results.append(stress_result)
        self._print_test_result(stress_result)

        # Generar reporte final
        return self._generate_final_report()

    async def run_load_test(self, name: str, endpoint: str, method: str = "GET",
                          concurrency: int = 10, duration: int = 30, target_rps: int = 10,
                          payload: Optional[Dict] = None) -> LoadTestResult:
        """Ejecutar una prueba de carga específica"""

        response_times = []
        errors = []
        successful_requests = 0
        failed_requests = 0

        start_time = time.time()

        # Crear tareas concurrentes
        tasks = []
        for i in range(concurrency):
            task = asyncio.create_task(
                self._run_worker(name, endpoint, method, duration, target_rps,
                               payload, response_times, errors)
            )
            tasks.append(task)

        # Ejecutar workers
        worker_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Contabilizar resultados
        for result in worker_results:
            if isinstance(result, dict):
                successful_requests += result.get('successful', 0)
                failed_requests += result.get('failed', 0)
            else:
                failed_requests += 1
                errors.append(f"Worker error: {str(result)}")

        total_requests = successful_requests + failed_requests
        test_duration = time.time() - start_time

        # Calcular métricas
        avg_response_time = statistics.mean(response_times) if response_times else 0
        min_response_time = min(response_times) if response_times else 0
        max_response_time = max(response_times) if response_times else 0

        # Calcular percentiles
        sorted_times = sorted(response_times)
        p95_response_time = sorted_times[int(len(sorted_times) * 0.95)] if sorted_times else 0
        p99_response_time = sorted_times[int(len(sorted_times) * 0.99)] if sorted_times else 0

        requests_per_second = total_requests / test_duration if test_duration > 0 else 0
        error_rate = (failed_requests / total_requests) * 100 if total_requests > 0 else 0

        return LoadTestResult(
            test_name=name,
            duration=test_duration,
            total_requests=total_requests,
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            avg_response_time=avg_response_time,
            min_response_time=min_response_time,
            max_response_time=max_response_time,
            p95_response_time=p95_response_time,
            p99_response_time=p99_response_time,
            requests_per_second=requests_per_second,
            error_rate=error_rate,
            response_times=response_times,
            errors=errors[:10]  # Limitar errores reportados
        )

    async def _run_worker(self, test_name: str, endpoint: str, method: str,
                         duration: int, target_rps: int, payload: Optional[Dict],
                         response_times: List[float], errors: List[str]) -> Dict[str, int]:
        """Worker para ejecutar requests"""

        successful = 0
        failed = 0
        end_time = time.time() + duration

        # Calcular delay entre requests para alcanzar target RPS
        delay = 1.0 / (target_rps / 10) if target_rps > 0 else 0.1  # 10 workers

        async with aiohttp.ClientSession() as session:
            while time.time() < end_time:
                try:
                    start_time = time.time()

                    url = f"{self.base_url}{endpoint}"
                    headers = {'Content-Type': 'application/json'}

                    if method.upper() == 'GET':
                        async with session.get(url, headers=headers) as response:
                            await response.text()
                    elif method.upper() == 'POST':
                        data = json.dumps(payload) if payload else '{}'
                        async with session.post(url, data=data, headers=headers) as response:
                            await response.text()

                    response_time = (time.time() - start_time) * 1000  # ms
                    response_times.append(response_time)
                    successful += 1

                except Exception as e:
                    failed += 1
                    errors.append(f"{test_name}: {str(e)}")

                # Control de rate limiting
                await asyncio.sleep(delay)

        return {'successful': successful, 'failed': failed}

    async def run_stress_test(self) -> LoadTestResult:
        """Ejecutar prueba de estrés del sistema completo"""

        print("   Iniciando prueba de estrés con 100 usuarios concurrentes...")

        # Combinar múltiples tipos de requests
        stress_endpoints = [
            ("/api/health", "GET", None),
            ("/api/metrics", "GET", None),
            ("/api/nodes", "GET", None),
            ("/api/p2p/broadcast", "POST", {"message": "stress_test"}),
        ]

        response_times = []
        errors = []
        successful_requests = 0
        failed_requests = 0

        start_time = time.time()
        duration = 120  # 2 minutos de estrés

        # Crear 100 workers
        tasks = []
        for i in range(100):
            endpoint, method, payload = stress_endpoints[i % len(stress_endpoints)]
            task = asyncio.create_task(
                self._run_stress_worker(endpoint, method, payload, duration,
                                      response_times, errors)
            )
            tasks.append(task)

        # Ejecutar workers de estrés
        worker_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Contabilizar resultados
        for result in worker_results:
            if isinstance(result, dict):
                successful_requests += result.get('successful', 0)
                failed_requests += result.get('failed', 0)
            else:
                failed_requests += 1
                errors.append(f"Stress worker error: {str(result)}")

        total_requests = successful_requests + failed_requests
        test_duration = time.time() - start_time

        # Calcular métricas
        avg_response_time = statistics.mean(response_times) if response_times else 0
        min_response_time = min(response_times) if response_times else 0
        max_response_time = max(response_times) if response_times else 0

        sorted_times = sorted(response_times)
        p95_response_time = sorted_times[int(len(sorted_times) * 0.95)] if sorted_times else 0
        p99_response_time = sorted_times[int(len(sorted_times) * 0.99)] if sorted_times else 0

        requests_per_second = total_requests / test_duration if test_duration > 0 else 0
        error_rate = (failed_requests / total_requests) * 100 if total_requests > 0 else 0

        return LoadTestResult(
            test_name="Sistema Completo - Stress Test",
            duration=test_duration,
            total_requests=total_requests,
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            avg_response_time=avg_response_time,
            min_response_time=min_response_time,
            max_response_time=max_response_time,
            p95_response_time=p95_response_time,
            p99_response_time=p99_response_time,
            requests_per_second=requests_per_second,
            error_rate=error_rate,
            response_times=response_times,
            errors=errors[:20]  # Más errores para stress test
        )

    async def _run_stress_worker(self, endpoint: str, method: str, payload: Optional[Dict],
                                duration: int, response_times: List[float], errors: List[str]) -> Dict[str, int]:
        """Worker para prueba de estrés"""

        successful = 0
        failed = 0
        end_time = time.time() + duration

        async with aiohttp.ClientSession() as session:
            while time.time() < end_time:
                try:
                    start_time = time.time()

                    url = f"{self.base_url}{endpoint}"
                    headers = {'Content-Type': 'application/json'}

                    if method.upper() == 'GET':
                        async with session.get(url, headers=headers) as response:
                            await response.text()
                    elif method.upper() == 'POST':
                        data = json.dumps(payload) if payload else '{}'
                        async with session.post(url, data=data, headers=headers) as response:
                            await response.text()

                    response_time = (time.time() - start_time) * 1000  # ms
                    response_times.append(response_time)
                    successful += 1

                    # Pequeño delay para evitar sobrecarga total
                    await asyncio.sleep(0.01)

                except Exception as e:
                    failed += 1
                    errors.append(f"Stress test: {str(e)}")
                    await asyncio.sleep(0.1)  # Backoff en caso de error

        return {'successful': successful, 'failed': failed}

    def _print_test_result(self, result: LoadTestResult):
        """Imprimir resultado de una prueba"""

        print(f"   ✅ Completada en {result.duration:.1f}s")
        print(f"   📊 Requests totales: {result.total_requests}")
        print(f"   🎯 RPS: {result.requests_per_second:.1f}")
        print(f"   📈 Latencia promedio: {result.avg_response_time:.1f}ms")
        print(f"   📉 Latencia P95: {result.p95_response_time:.1f}ms")
        print(f"   📉 Latencia P99: {result.p99_response_time:.1f}ms")
        print(f"   ⚠️ Tasa de error: {result.error_rate:.1f}%")

        if result.errors:
            print(f"   ❌ Errores encontrados: {len(result.errors)}")
            for error in result.errors[:3]:  # Mostrar primeros 3 errores
                print(f"      • {error[:100]}...")

    def _generate_final_report(self) -> Dict[str, Any]:
        """Generar reporte final de todas las pruebas"""

        # Calcular métricas agregadas
        total_requests = sum(r.total_requests for r in self.results)
        total_successful = sum(r.successful_requests for r in self.results)
        total_failed = sum(r.failed_requests for r in self.results)
        overall_error_rate = (total_failed / total_requests) * 100 if total_requests > 0 else 0

        # Métricas de performance
        all_response_times = []
        for result in self.results:
            all_response_times.extend(result.response_times)

        if all_response_times:
            avg_response_time = statistics.mean(all_response_times)
            p95_response_time = sorted(all_response_times)[int(len(all_response_times) * 0.95)]
            p99_response_time = sorted(all_response_times)[int(len(all_response_times) * 0.99)]
        else:
            avg_response_time = p95_response_time = p99_response_time = 0

        # Evaluar rendimiento
        performance_rating = "❌ CRÍTICO"
        if overall_error_rate < 1 and avg_response_time < 100:
            performance_rating = "✅ EXCELENTE"
        elif overall_error_rate < 5 and avg_response_time < 500:
            performance_rating = "⚠️ BUENO"
        elif overall_error_rate < 10 and avg_response_time < 1000:
            performance_rating = "🟡 ACEPTABLE"

        report = {
            "summary": {
                "total_tests": len(self.results),
                "total_requests": total_requests,
                "successful_requests": total_successful,
                "failed_requests": total_failed,
                "overall_error_rate": round(overall_error_rate, 2),
                "avg_response_time_ms": round(avg_response_time, 2),
                "p95_response_time_ms": round(p95_response_time, 2),
                "p99_response_time_ms": round(p99_response_time, 2),
                "performance_rating": performance_rating
            },
            "test_results": [
                {
                    "name": r.test_name,
                    "requests_per_second": round(r.requests_per_second, 1),
                    "avg_response_time": round(r.avg_response_time, 1),
                    "error_rate": round(r.error_rate, 2),
                    "status": "✅ PASSED" if r.error_rate < 5 and r.avg_response_time < 1000 else "❌ FAILED"
                }
                for r in self.results
            ],
            "recommendations": self._generate_recommendations()
        }

        # Guardar reporte en archivo
        with open("load_test_report.json", "w") as f:
            json.dump(report, f, indent=2, default=str)

        return report

    def _generate_recommendations(self) -> List[str]:
        """Generar recomendaciones basadas en los resultados"""

        recommendations = []

        # Analizar resultados para recomendaciones
        high_error_tests = [r for r in self.results if r.error_rate > 5]
        if high_error_tests:
            recommendations.append(f"📈 Mejorar estabilidad en {len(high_error_tests)} pruebas con alta tasa de error")

        slow_tests = [r for r in self.results if r.avg_response_time > 500]
        if slow_tests:
            recommendations.append(f"⚡ Optimizar performance en {len(slow_tests)} endpoints lentos")

        # Recomendaciones generales
        recommendations.extend([
            "🔧 Considerar implementar rate limiting más granular",
            "📊 Configurar monitoring detallado para endpoints críticos",
            "🛡️ Implementar circuit breakers para protección contra fallos",
            "📈 Evaluar escalado horizontal para manejar mayor carga"
        ])

        return recommendations

async def run_load_tests(base_url: str = "http://localhost:8080"):
    """Ejecutar suite completa de pruebas de carga"""

    tester = LoadTester(base_url)

    try:
        report = await tester.run_all_tests()

        print("
📊 REPORTE FINAL DE PRUEBAS DE CARGA"        print("=" * 50)
        print(f"🎯 Tests ejecutados: {report['summary']['total_tests']}")
        print(f"📊 Requests totales: {report['summary']['total_requests']:,}")
        print(f"✅ Requests exitosos: {report['summary']['successful_requests']:,}")
        print(f"❌ Requests fallidos: {report['summary']['failed_requests']:,}")
        print(f"⚠️ Tasa de error general: {report['summary']['overall_error_rate']}%")
        print(f"📈 Latencia promedio: {report['summary']['avg_response_time_ms']:.1f}ms")
        print(f"📉 Latencia P95: {report['summary']['p95_response_time_ms']:.1f}ms")
        print(f"📉 Latencia P99: {report['summary']['p99_response_time_ms']:.1f}ms")
        print(f"🏆 Rating de performance: {report['summary']['performance_rating']}")

        print("
📋 RESULTADOS POR PRUEBA:"        for test in report['test_results']:
            print(f"   • {test['name']}: {test['status']}")
            print(f"     RPS: {test['requests_per_second']}, Latencia: {test['avg_response_time']}ms, Error: {test['error_rate']}%")

        print("
💡 RECOMENDACIONES:"        for rec in report['recommendations']:
            print(f"   • {rec}")

        print("
💾 Reporte detallado guardado en: load_test_report.json"        return report

    except Exception as e:
        print(f"❌ Error ejecutando pruebas de carga: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    import sys

    # Permitir configuración de URL base
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8080"

    print(f"🔥 Iniciando pruebas de carga contra: {base_url}")
    asyncio.run(run_load_tests(base_url))
