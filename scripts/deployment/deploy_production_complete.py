#!/usr/bin/env python3
"""
AEGIS Framework - Despliegue de Producción Completo
Script maestro para desplegar todos los componentes en producción
"""

import os
import sys
import subprocess
import logging
import time
import json
import signal
from pathlib import Path
from typing import Dict, List, Optional

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('aegis_production.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AEGISProductionDeployment:
    """Despliegue completo de producción para AEGIS Framework"""
    
    def __init__(self):
        self.processes: Dict[str, subprocess.Popen] = {}
        self.config = self.load_config()
        self.setup_signal_handlers()
        logger.info("🚀 Inicializando despliegue de producción AEGIS")
    
    def load_config(self) -> Dict:
        """Cargar configuración de producción"""
        config_path = Path("production_config_v3.json")
        
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"⚠️ Error cargando configuración: {e}")
        
        # Configuración por defecto
        return {
            "node": {"port": 8080, "workers": 4},
            "api": {"port": 8000, "workers": 4},
            "dashboard": {"port": 3000, "workers": 2},
            "admin": {"port": 8081, "workers": 2}
        }
    
    def setup_signal_handlers(self):
        """Configurar manejadores de señales"""
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
        logger.info("✅ Manejadores de señales configurados")
    
    def _signal_handler(self, signum, frame):
        """Manejador de señales"""
        logger.info(f"📡 Señal {signum} recibida, apagando...")
        self.shutdown()
        sys.exit(0)
    
    def pre_deployment_checks(self) -> bool:
        """Verificaciones previas al despliegue"""
        logger.info("🔍 Realizando verificaciones previas...")
        
        checks = [
            ("Python 3.8+", self.check_python_version),
            ("Módulos requeridos", self.check_required_modules),
            ("Espacio en disco", self.check_disk_space),
            ("Puertos disponibles", self.check_ports_available),
            ("Permisos", self.check_permissions)
        ]
        
        all_passed = True
        for check_name, check_func in checks:
            try:
                result = check_func()
                status = "✅" if result else "❌"
                logger.info(f"  {status} {check_name}")
                all_passed = all_passed and result
            except Exception as e:
                logger.error(f"❌ Error en {check_name}: {e}")
                all_passed = False
        
        return all_passed
    
    def check_python_version(self) -> bool:
        """Verificar versión de Python"""
        version = sys.version_info
        if version < (3, 8):
            logger.error(f"❌ Python 3.8+ requerido, versión actual: {version.major}.{version.minor}")
            return False
        logger.info(f"✅ Python {version.major}.{version.minor}.{version.micro}")
        return True
    
    def check_required_modules(self) -> bool:
        """Verificar módulos requeridos"""
        required = ["gunicorn", "uvicorn", "flask", "fastapi", "redis", "psycopg2"]
        missing = []
        
        for module in required:
            try:
                __import__(module)
            except ImportError:
                missing.append(module)
        
        if missing:
            logger.error(f"❌ Módulos faltantes: {', '.join(missing)}")
            return False
        
        logger.info(f"✅ Todos los módulos requeridos disponibles")
        return True
    
    def check_disk_space(self) -> bool:
        """Verificar espacio en disco"""
        try:
            import shutil
            total, used, free = shutil.disk_usage("/")
            free_gb = free // (2**30)
            
            if free_gb < 1:  # Mínimo 1GB libre
                logger.error(f"❌ Espacio en disco insuficiente: {free_gb}GB libres")
                return False
            
            logger.info(f"✅ Espacio en disco: {free_gb}GB libres")
            return True
            
        except Exception as e:
            logger.warning(f"⚠️ No se pudo verificar espacio en disco: {e}")
            return True
    
    def check_ports_available(self) -> bool:
        """Verificar disponibilidad de puertos"""
        ports = [8080, 8000, 3000, 8081]
        available = []
        
        for port in ports:
            try:
                import socket
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    result = s.connect_ex(('127.0.0.1', port))
                    if result == 0:  # Puerto en uso
                        available.append(False)
                        logger.warning(f"⚠️ Puerto {port} en uso")
                    else:
                        available.append(True)
                        logger.info(f"✅ Puerto {port} disponible")
            except Exception as e:
                logger.warning(f"⚠️ Error verificando puerto {port}: {e}")
                available.append(True)
        
        return all(available)
    
    def check_permissions(self) -> bool:
        """Verificar permisos necesarios"""
        try:
            # Verificar escritura en directorio actual
            test_file = Path(".aegis_test")
            test_file.write_text("test")
            test_file.unlink()
            
            logger.info("✅ Permisos de escritura OK")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error de permisos: {e}")
            return False
    
    def deploy_services(self) -> bool:
        """Desplegar todos los servicios"""
        logger.info("🚀 Desplegando servicios AEGIS...")
        
        services = [
            ("aegis-node", self.deploy_node_service),
            ("aegis-api", self.deploy_api_service),
            ("aegis-dashboard", self.deploy_dashboard_service),
            ("aegis-admin", self.deploy_admin_service)
        ]
        
        results = {}
        for name, deploy_func in services:
            try:
                logger.info(f"🚀 Desplegando {name}...")
                results[name] = deploy_func()
                
                if results[name]:
                    logger.info(f"✅ {name} desplegado exitosamente")
                else:
                    logger.error(f"❌ {name} falló al desplegar")
                
                time.sleep(2)  # Esperar entre servicios
                
            except Exception as e:
                logger.error(f"❌ Error desplegando {name}: {e}")
                results[name] = False
        
        success_count = sum(1 for r in results.values() if r)
        total_count = len(results)
        
        logger.info(f"📊 Resultados del despliegue: {success_count}/{total_count} servicios")
        
        for name, success in results.items():
            status = "✅ OK" if success else "❌ FALLÓ"
            logger.info(f"  {status} {name}")
        
        return success_count == total_count
    
    def deploy_node_service(self) -> bool:
        """Desplegar servicio de nodo"""
        config = self.config.get("node", {"port": 8080, "workers": 4})
        return self.start_wsgi_server("aegis-node", "gunicorn", config["port"], config["workers"], "node:app")
    
    def deploy_api_service(self) -> bool:
        """Desplegar servicio API"""
        config = self.config.get("api", {"port": 8000, "workers": 4})
        return self.start_wsgi_server("aegis-api", "gunicorn", config["port"], config["workers"], "api:app")
    
    def deploy_dashboard_service(self) -> bool:
        """Desplegar servicio de dashboard"""
        config = self.config.get("dashboard", {"port": 3000, "workers": 2})
        return self.start_wsgi_server("aegis-dashboard", "uvicorn", config["port"], config["workers"], "dashboard:app")
    
    def deploy_admin_service(self) -> bool:
        """Desplegar servicio de administración"""
        config = self.config.get("admin", {"port": 8081, "workers": 2})
        return self.start_wsgi_server("aegis-admin", "uvicorn", config["port"], config["workers"], "admin:app")
    
    def start_wsgi_server(self, name: str, server_type: str, port: int, workers: int, app_module: str) -> bool:
        """Iniciar servidor WSGI/ASGI"""
        logger.info(f"🚀 Iniciando {name} ({server_type}) en puerto {port}")
        
        try:
            # Construir comando
            if server_type == "gunicorn":
                cmd = [
                    sys.executable, "-m", "gunicorn",
                    "--bind", f"127.0.0.1:{port}",
                    "--workers", str(workers),
                    "--worker-class", "sync",
                    "--max-requests", "1000",
                    "--timeout", "30",
                    "--keepalive", "2",
                    "--log-level", "info",
                    "--preload",
                    app_module
                ]
            else:  # uvicorn
                cmd = [
                    sys.executable, "-m", "uvicorn",
                    app_module,
                    "--host", "127.0.0.1",
                    "--port", str(port),
                    "--workers", str(workers),
                    "--log-level", "info"
                ]
            
            # Configurar entorno
            env = os.environ.copy()
            env.update({
                "AEGIS_SERVER_NAME": name,
                "AEGIS_SERVER_PORT": str(port),
                "AEGIS_ENVIRONMENT": "production",
                "AEGIS_LOG_LEVEL": "info"
            })
            
            # Iniciar proceso
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env
            )
            
            self.processes[name] = process
            
            # Verificar si se inició correctamente
            time.sleep(3)
            if process.poll() is None:
                logger.info(f"✅ {name} iniciado exitosamente (PID: {process.pid})")
                return True
            else:
                stdout, stderr = process.communicate()
                error_msg = stderr if stderr else stdout
                logger.error(f"❌ {name} falló al iniciar: {error_msg}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error iniciando {name}: {e}")
            return False
    
    def monitor_services(self):
        """Monitorear servicios"""
        logger.info("📊 Iniciando monitoreo de servicios...")
        
        while True:
            try:
                for name, process in list(self.processes.items()):
                    if process.poll() is not None:
                        logger.warning(f"⚠️ Servicio {name} murió (exit code: {process.returncode})")
                        # Intentar reiniciar
                        self.restart_service(name)
                    
                    # Verificar salud periódicamente
                    if int(time.time()) % 60 == 0:  # Cada minuto
                        self.check_service_health(name)
                
                time.sleep(30)  # Verificar cada 30 segundos
                
            except KeyboardInterrupt:
                logger.info("⏹️ Monitoreo interrumpido")
                break
            except Exception as e:
                logger.error(f"❌ Error en monitoreo: {e}")
                time.sleep(10)
    
    def check_service_health(self, name: str):
        """Verificar salud de un servicio"""
        try:
            import requests
            
            # Mapear servicios a puertos
            port_map = {
                "aegis-node": 8080,
                "aegis-api": 8000,
                "aegis-dashboard": 3000,
                "aegis-admin": 8081
            }
            
            port = port_map.get(name)
            if port:
                response = requests.get(f"http://127.0.0.1:{port}/health", timeout=5)
                if response.status_code == 200:
                    logger.info(f"✅ {name}: Salud OK")
                else:
                    logger.warning(f"⚠️ {name}: HTTP {response.status_code}")
                    
        except Exception as e:
            logger.warning(f"⚠️ {name}: No responde - {e}")
    
    def restart_service(self, name: str) -> bool:
        """Reiniciar un servicio"""
        logger.info(f"🔄 Reiniciando {name}...")
        
        # Detener servicio
        self.stop_service(name)
        time.sleep(2)
        
        # Reiniciar según tipo
        service_map = {
            "aegis-node": self.deploy_node_service,
            "aegis-api": self.deploy_api_service,
            "aegis-dashboard": self.deploy_dashboard_service,
            "aegis-admin": self.deploy_admin_service
        }
        
        deploy_func = service_map.get(name)
        if deploy_func:
            return deploy_func()
        
        return False
    
    def stop_service(self, name: str):
        """Detener un servicio"""
        if name in self.processes:
            process = self.processes[name]
            logger.info(f"⏹️ Deteniendo {name} (PID: {process.pid})...")
            
            try:
                process.terminate()
                process.wait(timeout=10)
                logger.info(f"✅ {name} detenido")
            except subprocess.TimeoutExpired:
                logger.warning(f"⏰ Timeout deteniendo {name}, forzando...")
                process.kill()
                process.wait()
            
            del self.processes[name]
    
    def shutdown(self):
        """Apagar todos los servicios"""
        logger.info("⏹️ Apagando todos los servicios...")
        
        for name in list(self.processes.keys()):
            self.stop_service(name)
            time.sleep(1)
        
        logger.info("✅ Todos los servicios apagados")
    
    def generate_deployment_report(self):
        """Generar reporte de despliegue"""
        logger.info("📊 Generando reporte de despliegue...")
        
        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "status": "success",
            "services": {},
            "configuration": self.config
        }
        
        for name, process in self.processes.items():
            is_running = process.poll() is None
            report["services"][name] = {
                "running": is_running,
                "pid": process.pid if is_running else None,
                "exit_code": process.returncode if not is_running else None
            }
        
        # Guardar reporte
        report_path = Path("deployment_report.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"📄 Reporte guardado en {report_path}")
        return report

def main():
    """Función principal"""
    logger.info("🚀 AEGIS Framework - Despliegue de Producción Completo")
    
    deployer = AEGISProductionDeployment()
    
    try:
        # Verificaciones previas
        if not deployer.pre_deployment_checks():
            logger.error("❌ Verificaciones previas fallaron")
            sys.exit(1)
        
        # Desplegar servicios
        if deployer.deploy_services():
            logger.info("🎉 Despliegue completado exitosamente")
            
            # Generar reporte
            report = deployer.generate_deployment_report()
            
            # Mostrar estado final
            logger.info("📊 Estado de servicios:")
            for name, info in report["services"].items():
                status = "🟢 Corriendo" if info["running"] else "🔴 Detenido"
                logger.info(f"  {status} {name}")
            
            # URLs de acceso
            logger.info("🌐 URLs de acceso:")
            logger.info("  • AEGIS Node: http://127.0.0.1:8080")
            logger.info("  • AEGIS API: http://127.0.0.1:8000")
            logger.info("  • AEGIS Dashboard: http://127.0.0.1:3000")
            logger.info("  • AEGIS Admin: http://127.0.0.1:8081")
            
            # Iniciar monitoreo
            logger.info("📊 Iniciando monitoreo...")
            deployer.monitor_services()
            
        else:
            logger.error("❌ Despliegue falló")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("⏹️ Interrupción detectada, apagando...")
        deployer.shutdown()
        sys.exit(0)
    except Exception as e:
        logger.error(f"❌ Error crítico: {e}")
        deployer.shutdown()
        sys.exit(1)

if __name__ == "__main__":
    main()