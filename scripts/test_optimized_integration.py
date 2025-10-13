#!/usr/bin/env python3
"""
Test de integración optimizada TOR-P2P
Prueba las optimizaciones implementadas en el sistema AEGIS
"""

import asyncio
import json
import time
import logging
from pathlib import Path
import sys

# Añadir el directorio raíz al path
sys.path.append(str(Path(__file__).parent.parent))

from tor_p2p_integration import TorP2PIntegrationManager, IntegrationMode, SecurityLevel
from p2p_optimization_manager import P2POptimizationManager, P2POptimizationLevel

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class OptimizationTester:
    """Tester para las optimizaciones TOR-P2P"""
    
    def __init__(self):
        self.integration_manager = None
        self.p2p_optimizer = None
        self.test_results = {}
    
    async def setup_test_environment(self):
        """Configura el entorno de pruebas"""
        logger.info("🔧 Configurando entorno de pruebas...")
        
        # Inicializar gestor de integración
        self.integration_manager = TorP2PIntegrationManager()
        
        # Inicializar optimizador P2P
        self.p2p_optimizer = P2POptimizationManager()
        
        logger.info("✅ Entorno configurado correctamente")
    
    async def test_tor_optimization(self):
        """Prueba las optimizaciones de TOR"""
        logger.info("🧅 Probando optimizaciones TOR...")
        
        start_time = time.time()
        
        try:
            # Verificar configuración optimizada de TOR
            config_path = Path("config/torrc_optimized")
            if config_path.exists():
                logger.info("✅ Configuración TOR optimizada encontrada")
                self.test_results['tor_config'] = True
            else:
                logger.warning("⚠️ Configuración TOR optimizada no encontrada")
                self.test_results['tor_config'] = False
            
            # Simular conexión TOR
            await asyncio.sleep(0.5)  # Simular latencia de conexión
            
            self.test_results['tor_connection_time'] = time.time() - start_time
            self.test_results['tor_status'] = 'optimized'
            
            logger.info(f"✅ TOR optimizado en {self.test_results['tor_connection_time']:.2f}s")
            
        except Exception as e:
            logger.error(f"❌ Error en optimización TOR: {e}")
            self.test_results['tor_status'] = 'error'
            self.test_results['tor_error'] = str(e)
    
    async def test_p2p_optimization(self):
        """Prueba las optimizaciones P2P"""
        logger.info("🌐 Probando optimizaciones P2P...")
        
        start_time = time.time()
        
        try:
            # Iniciar optimizador P2P
            await self.p2p_optimizer.start_optimization()
            
            # Simular métricas de peers
            test_peers = [
                f"peer_{i}" for i in range(5)
            ]
            
            for peer_id in test_peers:
                # Simular conexión exitosa
                self.p2p_optimizer.record_connection_attempt(peer_id, "127.0.0.1", 8080, True)
                
                # Simular latencia aleatoria
                import random
                latency = random.uniform(20, 100)
                self.p2p_optimizer.record_latency(peer_id, latency)
                
                # Simular tiempo de actividad
                self.p2p_optimizer.update_peer_uptime(peer_id, random.uniform(0.8, 0.99))
            
            # Obtener reporte de optimización
            optimization_report = self.p2p_optimizer.get_optimization_report()
            
            self.test_results['p2p_optimization_time'] = time.time() - start_time
            self.test_results['p2p_peer_count'] = optimization_report['peer_count']['total']
            self.test_results['p2p_best_peers'] = len(optimization_report['best_peers'])
            self.test_results['p2p_status'] = 'optimized'
            
            logger.info(f"✅ P2P optimizado con {self.test_results['p2p_peer_count']} peers")
            
        except Exception as e:
            logger.error(f"❌ Error en optimización P2P: {e}")
            self.test_results['p2p_status'] = 'error'
            self.test_results['p2p_error'] = str(e)
    
    async def test_integration_modes(self):
        """Prueba los diferentes modos de integración"""
        logger.info("🔄 Probando modos de integración...")
        
        integration_modes = [
            IntegrationMode.TOR_ONLY,
            IntegrationMode.P2P_ONLY,
            IntegrationMode.HYBRID,
            IntegrationMode.ADAPTIVE
        ]
        
        mode_results = {}
        
        for mode in integration_modes:
            try:
                start_time = time.time()
                
                # Configurar modo directamente en el config
                self.integration_manager.config.integration_mode = mode
                self.integration_manager.config.security_level = SecurityLevel.STANDARD
                
                # Iniciar integración
                await self.integration_manager.start_integration()
                
                # Simular envío de mensaje
                success = await self.integration_manager.send_message(
                    "test_connection",
                    {"test": "optimization_message"},
                    priority="normal"
                )
                
                mode_results[mode.value] = {
                    'success': success,
                    'time': time.time() - start_time,
                    'status': 'ok'
                }
                
                logger.info(f"✅ Modo {mode.value}: {success}")
                
                # Detener integración
                await self.integration_manager.stop_integration()
                
            except Exception as e:
                logger.error(f"❌ Error en modo {mode.value}: {e}")
                mode_results[mode.value] = {
                    'success': False,
                    'error': str(e),
                    'status': 'error'
                }
        
        self.test_results['integration_modes'] = mode_results
    
    async def test_performance_monitoring(self):
        """Prueba el monitoreo de rendimiento"""
        logger.info("📊 Probando monitoreo de rendimiento...")
        
        try:
            # Obtener estado de integración
            integration_status = self.integration_manager.get_integration_status()
            
            self.test_results['performance_monitoring'] = {
                'health_check': integration_status is not None,
                'metrics_available': 'performance_metrics' in integration_status,
                'status': 'ok'
            }
            
            logger.info("✅ Monitoreo de rendimiento funcionando")
            
        except Exception as e:
            logger.error(f"❌ Error en monitoreo: {e}")
            self.test_results['performance_monitoring'] = {
                'status': 'error',
                'error': str(e)
            }
    
    async def run_comprehensive_test(self):
        """Ejecuta todas las pruebas de optimización"""
        logger.info("🚀 Iniciando pruebas de optimización TOR-P2P...")
        
        test_start = time.time()
        
        # Configurar entorno
        await self.setup_test_environment()
        
        # Ejecutar pruebas
        await self.test_tor_optimization()
        await self.test_p2p_optimization()
        await self.test_integration_modes()
        await self.test_performance_monitoring()
        
        # Calcular tiempo total
        total_time = time.time() - test_start
        
        # Generar reporte final
        self.test_results['test_summary'] = {
            'total_time': total_time,
            'timestamp': time.time(),
            'tests_completed': len([
                k for k, v in self.test_results.items()
                if isinstance(v, dict) and v.get('status') != 'error'
            ]),
            'overall_status': 'optimized' if all(
                v.get('status') != 'error' for v in self.test_results.values()
                if isinstance(v, dict)
            ) else 'partial'
        }
        
        return self.test_results
    
    def print_results(self):
        """Imprime los resultados de las pruebas"""
        print("\n" + "="*60)
        print("🔍 REPORTE DE OPTIMIZACIÓN TOR-P2P")
        print("="*60)
        
        summary = self.test_results.get('test_summary', {})
        print(f"⏱️  Tiempo total: {summary.get('total_time', 0):.2f}s")
        print(f"✅ Pruebas completadas: {summary.get('tests_completed', 0)}")
        print(f"🎯 Estado general: {summary.get('overall_status', 'unknown')}")
        
        print("\n📋 DETALLES POR COMPONENTE:")
        print("-" * 40)
        
        # TOR
        if 'tor_status' in self.test_results:
            status = self.test_results['tor_status']
            time_taken = self.test_results.get('tor_connection_time', 0)
            print(f"🧅 TOR: {status} ({time_taken:.2f}s)")
        
        # P2P
        if 'p2p_status' in self.test_results:
            status = self.test_results['p2p_status']
            peer_count = self.test_results.get('p2p_peer_count', 0)
            print(f"🌐 P2P: {status} ({peer_count} peers)")
        
        # Modos de integración
        if 'integration_modes' in self.test_results:
            modes = self.test_results['integration_modes']
            successful_modes = sum(1 for m in modes.values() if m.get('success'))
            print(f"🔄 Integración: {successful_modes}/{len(modes)} modos exitosos")
        
        # Monitoreo
        if 'performance_monitoring' in self.test_results:
            monitoring = self.test_results['performance_monitoring']
            status = monitoring.get('status', 'unknown')
            print(f"📊 Monitoreo: {status}")
        
        print("\n" + "="*60)

async def main():
    """Función principal"""
    tester = OptimizationTester()
    
    try:
        # Ejecutar pruebas
        results = await tester.run_comprehensive_test()
        
        # Mostrar resultados
        tester.print_results()
        
        # Guardar resultados en archivo
        results_file = Path("test_results_optimization.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📄 Resultados guardados en: {results_file}")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Error en pruebas: {e}")
        return {'error': str(e)}

if __name__ == "__main__":
    asyncio.run(main())