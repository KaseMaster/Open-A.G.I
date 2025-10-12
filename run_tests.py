#!/usr/bin/env python3
"""
Script Principal para Ejecutar Tests de AEGIS
Ejecuta todas las suites de tests y genera reportes detallados.
"""

import asyncio
import sys
import os
import json
import time
import argparse
from datetime import datetime
from pathlib import Path

# Agregar directorios al path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_suites"))

from test_framework import get_test_framework, TestType, TestStatus
from test_suites.test_crypto import create_crypto_test_suite
from test_suites.test_p2p import create_p2p_test_suite
from test_suites.test_integration import create_integration_test_suite
from test_suites.test_performance import create_performance_test_suite

class AEGISTestRunner:
    """Runner principal para todos los tests de AEGIS"""
    
    def __init__(self):
        self.framework = get_test_framework()
        self.results_dir = Path("test_results")
        self.results_dir.mkdir(exist_ok=True)
        
    def register_all_suites(self):
        """Registra todas las suites de tests"""
        print("🔧 Registrando suites de tests...")
        
        suites = [
            ("Crypto", create_crypto_test_suite),
            ("P2P", create_p2p_test_suite),
            ("Integration", create_integration_test_suite),
            ("Performance", create_performance_test_suite)
        ]
        
        for suite_name, suite_creator in suites:
            try:
                suite = suite_creator()
                self.framework.register_test_suite(suite)
                print(f"  ✅ {suite_name}: {len(suite.tests)} tests registrados")
            except Exception as e:
                print(f"  ❌ {suite_name}: Error al registrar - {e}")
    
    async def run_specific_suite(self, suite_name: str):
        """Ejecuta una suite específica"""
        print(f"\n🚀 Ejecutando suite: {suite_name}")
        print("=" * 60)
        
        suite_results = await self.framework.run_test_suite(suite_name)
        
        if suite_results:
            self._print_suite_results(suite_name, suite_results)
            return suite_results
        else:
            print(f"❌ No se encontró la suite: {suite_name}")
            return None
    
    async def run_all_tests(self, test_types: list = None):
        """Ejecuta todos los tests o tipos específicos"""
        print("\n🚀 Iniciando ejecución completa de tests AEGIS")
        print("=" * 60)
        
        start_time = time.time()
        
        # Filtrar por tipos si se especifica
        if test_types:
            print(f"📋 Ejecutando solo tests de tipo: {', '.join(test_types)}")
        
        results = await self.framework.run_all_tests()
        
        total_time = time.time() - start_time
        
        # Generar reportes
        await self._generate_reports(results, total_time)
        
        return results
    
    def _print_suite_results(self, suite_name: str, results: dict):
        """Imprime resultados de una suite"""
        summary = results.get('summary', {})
        
        print(f"\n📊 Resultados de {suite_name}:")
        print(f"  Total: {summary.get('total_tests', 0)}")
        print(f"  Exitosos: {summary.get('passed_tests', 0)} ✅")
        print(f"  Fallidos: {summary.get('failed_tests', 0)} ❌")
        print(f"  Omitidos: {summary.get('skipped_tests', 0)} ⏭️")
        print(f"  Tasa de éxito: {summary.get('success_rate', 0):.1f}%")
        print(f"  Tiempo total: {summary.get('total_time', 0):.2f}s")
        
        # Mostrar tests fallidos
        failed_tests = [name for name, result in results.get('results', {}).items() 
                       if result.status == TestStatus.FAILED]
        
        if failed_tests:
            print(f"\n❌ Tests fallidos en {suite_name}:")
            for test_name in failed_tests:
                result = results['results'][test_name]
                print(f"  • {test_name}: {result.error_message}")
    
    async def _generate_reports(self, results: dict, total_time: float):
        """Genera reportes detallados"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Reporte JSON detallado
        json_report = self._create_json_report(results, total_time, timestamp)
        json_file = self.results_dir / f"aegis_test_report_{timestamp}.json"
        
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(json_report, f, indent=2, ensure_ascii=False, default=str)
        
        # Reporte HTML
        html_report = self._create_html_report(results, total_time, timestamp)
        html_file = self.results_dir / f"aegis_test_report_{timestamp}.html"
        
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_report)
        
        # Reporte de consola
        self._print_final_summary(results, total_time)
        
        print(f"\n📄 Reportes generados:")
        print(f"  JSON: {json_file}")
        print(f"  HTML: {html_file}")
    
    def _create_json_report(self, results: dict, total_time: float, timestamp: str) -> dict:
        """Crea reporte JSON detallado"""
        return {
            "aegis_test_report": {
                "timestamp": timestamp,
                "execution_time": total_time,
                "summary": results.get('summary', {}),
                "suites": {},
                "environment": {
                    "python_version": sys.version,
                    "platform": sys.platform,
                    "working_directory": os.getcwd()
                }
            }
        }
    
    def _create_html_report(self, results: dict, total_time: float, timestamp: str) -> str:
        """Crea reporte HTML"""
        summary = results.get('summary', {})
        
        html = f"""
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AEGIS Test Report - {timestamp}</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        .header {{ text-align: center; margin-bottom: 30px; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 8px; }}
        .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-bottom: 30px; }}
        .metric {{ background: #f8f9fa; padding: 20px; border-radius: 8px; text-align: center; border-left: 4px solid #007bff; }}
        .metric h3 {{ margin: 0 0 10px 0; color: #495057; }}
        .metric .value {{ font-size: 2em; font-weight: bold; color: #007bff; }}
        .success {{ border-left-color: #28a745; }}
        .success .value {{ color: #28a745; }}
        .failure {{ border-left-color: #dc3545; }}
        .failure .value {{ color: #dc3545; }}
        .warning {{ border-left-color: #ffc107; }}
        .warning .value {{ color: #ffc107; }}
        .suite {{ margin-bottom: 30px; border: 1px solid #dee2e6; border-radius: 8px; overflow: hidden; }}
        .suite-header {{ background: #e9ecef; padding: 15px; font-weight: bold; }}
        .suite-content {{ padding: 15px; }}
        .test-item {{ padding: 10px; border-bottom: 1px solid #f1f3f4; display: flex; justify-content: space-between; align-items: center; }}
        .test-item:last-child {{ border-bottom: none; }}
        .status {{ padding: 4px 8px; border-radius: 4px; font-size: 0.8em; font-weight: bold; }}
        .status.passed {{ background: #d4edda; color: #155724; }}
        .status.failed {{ background: #f8d7da; color: #721c24; }}
        .status.skipped {{ background: #fff3cd; color: #856404; }}
        .footer {{ text-align: center; margin-top: 30px; padding-top: 20px; border-top: 1px solid #dee2e6; color: #6c757d; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🛡️ AEGIS Test Report</h1>
            <p>Generado el {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}</p>
        </div>
        
        <div class="summary">
            <div class="metric">
                <h3>Total Tests</h3>
                <div class="value">{summary.get('total_tests', 0)}</div>
            </div>
            <div class="metric success">
                <h3>Exitosos</h3>
                <div class="value">{summary.get('passed_tests', 0)}</div>
            </div>
            <div class="metric failure">
                <h3>Fallidos</h3>
                <div class="value">{summary.get('failed_tests', 0)}</div>
            </div>
            <div class="metric warning">
                <h3>Omitidos</h3>
                <div class="value">{summary.get('skipped_tests', 0)}</div>
            </div>
            <div class="metric">
                <h3>Tasa de Éxito</h3>
                <div class="value">{summary.get('success_rate', 0):.1f}%</div>
            </div>
            <div class="metric">
                <h3>Tiempo Total</h3>
                <div class="value">{total_time:.1f}s</div>
            </div>
        </div>
        
        <div class="suites">
            <!-- Las suites se agregarían aquí dinámicamente -->
        </div>
        
        <div class="footer">
            <p>AEGIS - Sistema de Seguridad Avanzado | Test Framework v1.0</p>
        </div>
    </div>
</body>
</html>
        """
        
        return html
    
    def _print_final_summary(self, results: dict, total_time: float):
        """Imprime resumen final en consola"""
        summary = results.get('summary', {})
        
        print("\n" + "=" * 60)
        print("🏁 RESUMEN FINAL DE TESTS AEGIS")
        print("=" * 60)
        
        print(f"📊 Estadísticas Generales:")
        print(f"  Total de tests: {summary.get('total_tests', 0)}")
        print(f"  Exitosos: {summary.get('passed_tests', 0)} ✅")
        print(f"  Fallidos: {summary.get('failed_tests', 0)} ❌")
        print(f"  Omitidos: {summary.get('skipped_tests', 0)} ⏭️")
        print(f"  Tasa de éxito: {summary.get('success_rate', 0):.1f}%")
        print(f"  Tiempo total: {total_time:.2f} segundos")
        
        # Estado general
        if summary.get('success_rate', 0) >= 95:
            print(f"\n🎉 EXCELENTE: Sistema AEGIS en óptimas condiciones")
        elif summary.get('success_rate', 0) >= 80:
            print(f"\n✅ BUENO: Sistema AEGIS funcionando correctamente")
        elif summary.get('success_rate', 0) >= 60:
            print(f"\n⚠️  ADVERTENCIA: Sistema AEGIS requiere atención")
        else:
            print(f"\n🚨 CRÍTICO: Sistema AEGIS requiere revisión inmediata")
        
        # Recomendaciones
        failed_count = summary.get('failed_tests', 0)
        if failed_count > 0:
            print(f"\n🔧 Recomendaciones:")
            print(f"  • Revisar {failed_count} tests fallidos")
            print(f"  • Verificar configuración del sistema")
            print(f"  • Ejecutar tests individuales para diagnóstico detallado")

async def main():
    """Función principal"""
    parser = argparse.ArgumentParser(description="Ejecutor de Tests AEGIS")
    parser.add_argument("--suite", "-s", help="Ejecutar suite específica (crypto, p2p, integration, performance)")
    parser.add_argument("--type", "-t", action="append", help="Tipos de test a ejecutar (unit, integration, performance, security)")
    parser.add_argument("--list", "-l", action="store_true", help="Listar suites disponibles")
    parser.add_argument("--quick", "-q", action="store_true", help="Ejecutar solo tests rápidos")
    
    args = parser.parse_args()
    
    runner = AEGISTestRunner()
    
    if args.list:
        print("📋 Suites de tests disponibles:")
        print("  • crypto - Tests del framework criptográfico")
        print("  • p2p - Tests de la red P2P")
        print("  • integration - Tests de integración")
        print("  • performance - Tests de rendimiento")
        return
    
    # Registrar todas las suites
    runner.register_all_suites()
    
    if args.suite:
        # Ejecutar suite específica
        await runner.run_specific_suite(args.suite)
    else:
        # Ejecutar todos los tests
        test_types = args.type if args.type else None
        
        if args.quick:
            # Solo tests unitarios y de integración básica
            test_types = ["unit", "integration"]
        
        await runner.run_all_tests(test_types)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⏹️  Ejecución interrumpida por el usuario")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Error fatal: {e}")
        sys.exit(1)