#!/usr/bin/env python3
"""
🔍 AEGIS Framework - Sistema de Validación Completa
Validación integral de todos los componentes del sistema AEGIS
"""

import asyncio
import json
import logging
import time
import requests
import socket
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SystemValidator:
    """Validador completo del sistema AEGIS"""
    
    def __init__(self):
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'validation_results': {},
            'service_status': {},
            'performance_metrics': {},
            'recommendations': [],
            'overall_status': 'unknown'
        }
        
        # Configuración de servicios
        self.services = {
            'dashboard': {'url': 'http://localhost:5000', 'name': 'Dashboard Principal'},
            'tor': {'port': 9050, 'name': 'Servicio TOR'},
            'tor_control': {'port': 9051, 'name': 'Control TOR'},
            'secure_chat': {'url': 'http://localhost:3000', 'name': 'Chat Seguro UI'},
            'blockchain': {'url': 'http://localhost:8545', 'name': 'Nodo Blockchain'}
        }
    
    def check_port_availability(self, port: int, host: str = 'localhost') -> bool:
        """Verifica si un puerto está disponible/en uso"""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(3)
                result = sock.connect_ex((host, port))
                return result == 0  # 0 significa que la conexión fue exitosa
        except Exception as e:
            logger.error(f"Error verificando puerto {port}: {e}")
            return False
    
    def check_http_service(self, url: str, timeout: int = 5) -> Dict[str, Any]:
        """Verifica el estado de un servicio HTTP"""
        try:
            response = requests.get(url, timeout=timeout, allow_redirects=False)
            return {
                'status': 'running',
                'status_code': response.status_code,
                'response_time': response.elapsed.total_seconds(),
                'accessible': True
            }
        except requests.exceptions.ConnectionError:
            return {
                'status': 'connection_refused',
                'accessible': False,
                'error': 'Conexión rechazada'
            }
        except requests.exceptions.Timeout:
            return {
                'status': 'timeout',
                'accessible': False,
                'error': 'Timeout de conexión'
            }
        except Exception as e:
            return {
                'status': 'error',
                'accessible': False,
                'error': str(e)
            }
    
    def validate_tor_service(self) -> Dict[str, Any]:
        """Valida el servicio TOR"""
        logger.info("🧅 Validando servicio TOR...")
        
        # Verificar puerto SOCKS
        socks_available = self.check_port_availability(9050)
        
        # Verificar puerto de control
        control_available = self.check_port_availability(9051)
        
        # Verificar archivos de configuración
        torrc_exists = Path('config/torrc').exists()
        
        status = {
            'socks_port': socks_available,
            'control_port': control_available,
            'config_exists': torrc_exists,
            'status': 'running' if (socks_available and control_available) else 'stopped'
        }
        
        if status['status'] == 'running':
            logger.info("✅ Servicio TOR funcionando correctamente")
        else:
            logger.warning("⚠️ Problemas detectados en servicio TOR")
        
        return status
    
    def validate_dashboard(self) -> Dict[str, Any]:
        """Valida el dashboard principal"""
        logger.info("📊 Validando dashboard principal...")
        
        result = self.check_http_service('http://localhost:5000')
        
        if result.get('accessible'):
            logger.info("✅ Dashboard accesible")
        else:
            logger.warning("⚠️ Dashboard no accesible")
        
        return result
    
    def validate_secure_chat(self) -> Dict[str, Any]:
        """Valida la UI del chat seguro"""
        logger.info("💬 Validando chat seguro...")
        
        result = self.check_http_service('http://localhost:3000')
        
        if result.get('accessible'):
            logger.info("✅ Chat seguro accesible")
        else:
            logger.warning("⚠️ Chat seguro no accesible")
        
        return result
    
    def validate_blockchain(self) -> Dict[str, Any]:
        """Valida el nodo blockchain"""
        logger.info("⛓️ Validando nodo blockchain...")
        
        try:
            # Verificar puerto
            port_available = self.check_port_availability(8545)
            
            # Intentar llamada RPC
            rpc_data = {
                "jsonrpc": "2.0",
                "method": "eth_blockNumber",
                "params": [],
                "id": 1
            }
            
            response = requests.post(
                'http://localhost:8545',
                json=rpc_data,
                timeout=5
            )
            
            if response.status_code == 200:
                block_data = response.json()
                return {
                    'status': 'running',
                    'port_available': port_available,
                    'rpc_accessible': True,
                    'latest_block': block_data.get('result', 'unknown')
                }
            else:
                return {
                    'status': 'error',
                    'port_available': port_available,
                    'rpc_accessible': False,
                    'error': f'HTTP {response.status_code}'
                }
                
        except Exception as e:
            return {
                'status': 'error',
                'port_available': False,
                'rpc_accessible': False,
                'error': str(e)
            }
    
    def validate_file_structure(self) -> Dict[str, Any]:
        """Valida la estructura de archivos críticos"""
        logger.info("📁 Validando estructura de archivos...")
        
        critical_files = [
            'main.py',
            'config/torrc',
            'p2p_optimization_manager.py',
            'tor_p2p_integration.py',
            'security_protocols.py',
            'requirements.txt'
        ]
        
        critical_dirs = [
            'config',
            'scripts',
            'reports',
            'dapps/secure-chat',
            'dapps/aegis-token'
        ]
        
        file_status = {}
        for file_path in critical_files:
            file_status[file_path] = Path(file_path).exists()
        
        dir_status = {}
        for dir_path in critical_dirs:
            dir_status[dir_path] = Path(dir_path).exists()
        
        return {
            'files': file_status,
            'directories': dir_status,
            'all_files_present': all(file_status.values()),
            'all_dirs_present': all(dir_status.values())
        }
    
    def validate_p2p_optimization(self) -> Dict[str, Any]:
        """Valida los componentes de optimización P2P"""
        logger.info("🌐 Validando optimización P2P...")
        
        try:
            # Importar y probar el manager P2P
            sys.path.append('.')
            from p2p_optimization_manager import P2POptimizationManager
            
            # Crear instancia de prueba
            manager = P2POptimizationManager()
            
            # Verificar métodos principales
            methods_available = {
                'start_optimization': hasattr(manager, 'start_optimization'),
                'stop_optimization': hasattr(manager, 'stop_optimization'),
                'record_connection_attempt': hasattr(manager, 'record_connection_attempt'),
                'get_optimization_report': hasattr(manager, 'get_optimization_report')
            }
            
            return {
                'status': 'available',
                'methods': methods_available,
                'all_methods_present': all(methods_available.values())
            }
            
        except ImportError as e:
            return {
                'status': 'import_error',
                'error': str(e)
            }
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def validate_tor_integration(self) -> Dict[str, Any]:
        """Valida la integración TOR-P2P"""
        logger.info("🔗 Validando integración TOR-P2P...")
        
        try:
            from tor_p2p_integration import TorP2PIntegrationManager
            
            # Crear instancia de prueba
            manager = TorP2PIntegrationManager()
            
            # Verificar métodos principales
            methods_available = {
                'start_integration': hasattr(manager, 'start_integration'),
                'stop_integration': hasattr(manager, 'stop_integration'),
                'create_connection': hasattr(manager, 'create_connection'),
                'send_message': hasattr(manager, 'send_message')
            }
            
            return {
                'status': 'available',
                'methods': methods_available,
                'all_methods_present': all(methods_available.values())
            }
            
        except ImportError as e:
            return {
                'status': 'import_error',
                'error': str(e)
            }
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def generate_recommendations(self) -> List[str]:
        """Genera recomendaciones basadas en los resultados"""
        recommendations = []
        
        # Verificar servicios
        if not self.results['service_status'].get('tor', {}).get('status') == 'running':
            recommendations.append("🧅 Iniciar servicio TOR para comunicación anónima")
        
        if not self.results['service_status'].get('dashboard', {}).get('accessible'):
            recommendations.append("📊 Verificar configuración del dashboard principal")
        
        if not self.results['service_status'].get('blockchain', {}).get('status') == 'running':
            recommendations.append("⛓️ Iniciar nodo blockchain para funcionalidad completa")
        
        # Verificar componentes
        if not self.results['validation_results'].get('p2p_optimization', {}).get('all_methods_present'):
            recommendations.append("🌐 Revisar implementación de optimización P2P")
        
        if not self.results['validation_results'].get('tor_integration', {}).get('all_methods_present'):
            recommendations.append("🔗 Verificar integración TOR-P2P")
        
        # Recomendaciones generales
        recommendations.extend([
            "🔒 Revisar logs de seguridad regularmente",
            "📈 Monitorear métricas de rendimiento",
            "🔄 Actualizar dependencias periódicamente",
            "💾 Realizar respaldos de configuración"
        ])
        
        return recommendations
    
    def calculate_overall_status(self) -> str:
        """Calcula el estado general del sistema"""
        critical_services = ['tor', 'dashboard']
        critical_components = ['p2p_optimization', 'tor_integration']
        
        # Verificar servicios críticos
        services_ok = 0
        for service in critical_services:
            service_data = self.results['service_status'].get(service, {})
            if service_data.get('status') == 'running' or service_data.get('accessible'):
                services_ok += 1
        
        # Verificar componentes críticos
        components_ok = 0
        for component in critical_components:
            component_data = self.results['validation_results'].get(component, {})
            if component_data.get('status') == 'available':
                components_ok += 1
        
        # Calcular porcentaje de salud
        total_checks = len(critical_services) + len(critical_components)
        passed_checks = services_ok + components_ok
        health_percentage = (passed_checks / total_checks) * 100
        
        if health_percentage >= 80:
            return 'excellent'
        elif health_percentage >= 60:
            return 'good'
        elif health_percentage >= 40:
            return 'fair'
        else:
            return 'poor'
    
    async def run_validation(self) -> Dict[str, Any]:
        """Ejecuta la validación completa del sistema"""
        logger.info("🚀 Iniciando validación completa del sistema AEGIS...")
        start_time = time.time()
        
        # Validar servicios
        self.results['service_status'] = {
            'tor': self.validate_tor_service(),
            'dashboard': self.validate_dashboard(),
            'secure_chat': self.validate_secure_chat(),
            'blockchain': self.validate_blockchain()
        }
        
        # Validar componentes
        self.results['validation_results'] = {
            'file_structure': self.validate_file_structure(),
            'p2p_optimization': self.validate_p2p_optimization(),
            'tor_integration': self.validate_tor_integration()
        }
        
        # Métricas de rendimiento
        validation_time = time.time() - start_time
        self.results['performance_metrics'] = {
            'validation_time': validation_time,
            'total_checks': 7,
            'timestamp': datetime.now().isoformat()
        }
        
        # Generar recomendaciones
        self.results['recommendations'] = self.generate_recommendations()
        
        # Calcular estado general
        self.results['overall_status'] = self.calculate_overall_status()
        
        return self.results
    
    def print_summary(self):
        """Imprime un resumen de la validación"""
        print("\n" + "="*80)
        print("🔍 RESUMEN DE VALIDACIÓN DEL SISTEMA AEGIS")
        print("="*80)
        
        # Estado general
        status_emoji = {
            'excellent': '🟢',
            'good': '🟡',
            'fair': '🟠',
            'poor': '🔴'
        }
        
        overall = self.results['overall_status']
        print(f"📊 Estado General: {status_emoji.get(overall, '⚪')} {overall.upper()}")
        print(f"⏱️ Tiempo de Validación: {self.results['performance_metrics']['validation_time']:.2f}s")
        
        # Servicios
        print("\n🔧 SERVICIOS:")
        for service, data in self.results['service_status'].items():
            status = data.get('status', 'unknown')
            accessible = data.get('accessible', False)
            
            if status == 'running' or accessible:
                print(f"  ✅ {service.upper()}: Funcionando")
            else:
                print(f"  ❌ {service.upper()}: No disponible")
        
        # Componentes
        print("\n🧩 COMPONENTES:")
        for component, data in self.results['validation_results'].items():
            status = data.get('status', 'unknown')
            
            if status == 'available':
                print(f"  ✅ {component.replace('_', ' ').title()}: Disponible")
            else:
                print(f"  ❌ {component.replace('_', ' ').title()}: No disponible")
        
        # Recomendaciones principales
        print("\n💡 RECOMENDACIONES PRINCIPALES:")
        for i, rec in enumerate(self.results['recommendations'][:5], 1):
            print(f"  {i}. {rec}")
        
        print("\n" + "="*80)

async def main():
    """Función principal"""
    validator = SystemValidator()
    
    try:
        # Ejecutar validación
        results = await validator.run_validation()
        
        # Mostrar resumen
        validator.print_summary()
        
        # Guardar reporte
        timestamp = int(time.time())
        report_path = Path(f'reports/system_validation_{timestamp}.json')
        report_path.parent.mkdir(exist_ok=True)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📄 Reporte guardado en: {report_path}")
        print(f"\n📄 Reporte completo guardado en: {report_path}")
        
        # Determinar código de salida
        if results['overall_status'] in ['excellent', 'good']:
            print("🎉 Validación completada exitosamente!")
            return 0
        else:
            print("⚠️ Se encontraron problemas en la validación")
            return 1
            
    except Exception as e:
        logger.error(f"❌ Error durante la validación: {e}")
        print(f"❌ Error durante la validación: {e}")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)