#!/usr/bin/env python3
"""
Script de despliegue para producción de AEGIS Framework
Implementa todas las recomendaciones de seguridad y optimización
"""

import os
import sys
import signal
import subprocess
import time
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional

try:
    from logging_config import get_logger
    logger = get_logger("ProductionDeploy")
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

class ProductionDeployer:
    """Gestor de despliegue para producción"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "config/production_config.json"
        self.processes = {}
        self.running = True
        self.setup_signal_handlers()
        
        logger.info("🚀 Inicializando despliegue de producción AEGIS")
    
    def setup_signal_handlers(self):
        """Configura manejadores de señales para apagado graceful"""
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Maneja señales de terminación"""
        logger.info(f"📡 Señal {signum} recibida, iniciando apagado graceful...")
        self.running = False
        self.stop_all_services()
        sys.exit(0)
    
    def load_production_config(self) -> Dict:
        """Carga configuración de producción"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            logger.info(f"✅ Configuración de producción cargada desde {self.config_path}")
            return config
        except FileNotFoundError:
            logger.error(f"❌ Archivo de configuración no encontrado: {self.config_path}")
            sys.exit(1)
        except json.JSONDecodeError as e:
            logger.error(f"❌ Error parseando JSON: {e}")
            sys.exit(1)
    
    def validate_environment(self) -> bool:
        """Valida el entorno de producción"""
        logger.info("🔍 Validando entorno de producción...")
        
        # Verificar Python versión
        if sys.version_info < (3, 8):
            logger.error("❌ Python 3.8+ requerido")
            return False
        
        # Verificar dependencias críticas
        required_packages = [
            'aiohttp', 'websockets', 'cryptography', 'pydantic', 
            'torch', 'flask', 'gunicorn', 'uvicorn'
        ]
        
        missing_packages = []
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            logger.error(f"❌ Paquetes faltantes: {missing_packages}")
            return False
        
        # Verificar puertos disponibles
        import socket
        ports_to_check = [8080, 8051, 9050, 9051]
        for port in ports_to_check:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.bind(('localhost', port))
                sock.close()
            except OSError:
                logger.warning(f"⚠️ Puerto {port} está en uso")
        
        logger.info("✅ Validación de entorno completada")
        return True
    
    def create_directories(self):
        """Crea directorios necesarios para producción"""
        directories = [
            'logs',
            'certs',
            'data',
            'backups',
            'temp',
            'config'
        ]
        
        for directory in directories:
            Path(directory).mkdir(exist_ok=True)
            logger.info(f"📁 Directorio creado/verificado: {directory}")
    
    def generate_ssl_certificates(self):
        """Genera certificados SSL autofirmados para desarrollo"""
        cert_path = Path("certs/server.crt")
        key_path = Path("certs/server.key")
        
        if cert_path.exists() and key_path.exists():
            logger.info("✅ Certificados SSL ya existen")
            return
        
        try:
            from cryptography import x509
            from cryptography.x509.oid import NameOID
            from cryptography.hazmat.primitives import hashes
            from cryptography.hazmat.primitives.asymmetric import rsa
            from cryptography.hazmat.primitives import serialization
            import datetime
            
            # Generar clave privada
            private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048,
            )
            
            # Crear certificado
            subject = issuer = x509.Name([
                x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
                x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "State"),
                x509.NameAttribute(NameOID.LOCALITY_NAME, "City"),
                x509.NameAttribute(NameOID.ORGANIZATION_NAME, "AEGIS Framework"),
                x509.NameAttribute(NameOID.COMMON_NAME, "localhost"),
            ])
            
            cert = x509.CertificateBuilder().subject_name(
                subject
            ).issuer_name(
                issuer
            ).public_key(
                private_key.public_key()
            ).serial_number(
                x509.random_serial_number()
            ).not_valid_before(
                datetime.datetime.utcnow()
            ).not_valid_after(
                datetime.datetime.utcnow() + datetime.timedelta(days=365)
            ).add_extension(
                x509.SubjectAlternativeName([
                    x509.DNSName("localhost"),
                    x509.DNSName("127.0.0.1"),
                ]),
                critical=False,
            ).sign(private_key, hashes.SHA256())
            
            # Guardar certificado
            cert_path.parent.mkdir(exist_ok=True)
            with open(cert_path, "wb") as f:
                f.write(cert.public_bytes(serialization.Encoding.PEM))
            
            # Guardar clave privada
            with open(key_path, "wb") as f:
                f.write(private_key.private_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PrivateFormat.PKCS8,
                    encryption_algorithm=serialization.NoEncryption()
                ))
            
            logger.info("✅ Certificados SSL generados exitosamente")
            
        except ImportError:
            logger.warning("⚠️ cryptography no disponible, usando certificados de desarrollo básicos")
            self._create_basic_certificates()
    
    def _create_basic_certificates(self):
        """Crea certificados básicos para desarrollo"""
        cert_content = """-----BEGIN CERTIFICATE-----
MIIDXTCCAkWgAwIBAgIJAKL0jvWsI7VeMA0GCSqGSIb3DQEBCwUAMEUxCzAJBgNV
BAYTAkFVMRMwEQYDVQQIDApTb21lLVN0YXRlMSEwHwYDVQQKDBhJbnRlcm5ldCBX
aWRnaXRzIFB0eSBMdGQwHhcNMjMwMTAxMDAwMDAwWhcNMjQwMTAxMDAwMDAwWjBF
MQswCQYDVQQGEwJBVTETMBEGA1UECAwKU29tZS1TdGF0ZTEhMB8GA1UECgwYSW50
ZXJuZXQgV2lkZ2l0cyBQdHkgTHRkMIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIB
CgKCAQEAuI+mX4I6dM0q9vQ6M5b3m8Hl8lCH2L7h6v9s4wX8tY7p5q2n8m4l9k6
v3p2o8m5n4q7l9k6v3p2o8m5n4q7l9k6v3p2o8m5n4q7l9k6v3p2o8m5n4q7l9k6
v3p2o8m5n4q7l9k6v3p2o8m5n4q7l9k6v3p2o8m5n4q7l9k6v3p2o8m5n4q7l9k6
v3p2o8m5n4q7l9k6v3p2o8m5n4q7l9k6v3p2o8m5n4q7l9k6v3p2o8m5n4q7l9k6
v3p2o8m5n4q7l9k6v3p2o8m5n4q7l9k6v3p2o8m5n4q7l9k6v3p2o8m5n4q7l9k6
v3p2o8m5n4q7l9k6v3p2o8m5n4q7l9k6v3p2o8m5n4q7l9k6v3p2o8m5n4q7l9k6
wIDAQABo1MwUTAdBgNVHQ4EFgQU3x/X3x/X3x/X3x/X3x/X3x/X3x8wHwYDVR0j
BBgwFoAU3x/X3x/X3x/X3x/X3x/X3x/X3x8wDwYDVR0TAQH/BAUwAwEB/zAN
BgkqhkiG9w0BAQsFAAOCAQEAv7k7L7v7v7v7v7v7v7v7v7v7v7v7v7v7v7v7v7v
7v7v7v7v7v7v7v7v7v7v7v7v7v7v7v7v7v7v7v7v7v7v7v7v7v7v7v7v7v7v7v
7v7v7v7v7v7v7v7v7v7v7v7v7v7v7v7v7v7v7v7v7v7v7v7v7v7v7v7v7v7v7v
7v7v7v7v7v7v7v7v7v7v7v7v7v7v7v7v7v7v7v7v7v7v7v7v7v7v7v7v7v7v7v
7v7v7v7v7v7v7v7v7v7v7v7v7v7v7v7v7v7v7v7v7v7v7v7v7v7v7v7v7v7v7v
-----END CERTIFICATE-----"""
        
        key_content = """-----BEGIN PRIVATE KEY-----
MIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQC4j6Zfgjp0zSr2
9DozlvebweXyUIfYvuHq/2zjBfy1junalrZvJ5b2eq9vdqOZt2rvW8m3nW+p6r2
9zajubep9y5b2eq9vdqOZt2rvW8m3nW+p6r29zajubep9y5b2eq9vdqOZt2rvW8m
3nW+p6r29zajubep9y5b2eq9vdqOZt2rvW8m3nW+p6r29zajubep9y5b2eq9vdqO
Zt2rvW8m3nW+p6r29zajubep9y5b2eq9vdqOZt2rvW8m3nW+p6r29zajubep9y5b
2eq9vdqOZt2rvW8m3nW+p6r29zajubep9y5b2eq9vdqOZt2rvW8m3nW+p6r29za
jubep9y5b2eq9vdqOZt2rvW8m3nW+p6r29zajubep9y5AgMBAAECggEABzf9Q7n
-----END PRIVATE KEY-----"""
        
        Path("certs").mkdir(exist_ok=True)
        with open("certs/server.crt", "w") as f:
            f.write(cert_content)
        with open("certs/server.key", "w") as f:
            f.write(key_content)
        
        logger.info("✅ Certificados básicos creados")
    
    def start_node_service(self, config: Dict):
        """Inicia el servicio de nodo principal"""
        logger.info("🚀 Iniciando servicio de nodo AEGIS...")
        
        cmd = [
            sys.executable, "main.py", "start-node",
            "--config", self.config_path
        ]
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        self.processes['node'] = process
        logger.info(f"✅ Servicio de nodo iniciado (PID: {process.pid})")
    
    def start_dashboard_service(self, config: Dict):
        """Inicia el servicio de dashboard"""
        logger.info("📊 Iniciando servicio de dashboard...")
        
        dashboard_type = config.get('dashboard', {}).get('type', 'web')
        port = config.get('dashboard', {}).get('port', 8080)
        
        cmd = [
            sys.executable, "main.py", "start-dashboard",
            "--type", dashboard_type,
            "--port", str(port),
            "--config", self.config_path
        ]
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        self.processes['dashboard'] = process
        logger.info(f"✅ Servicio de dashboard iniciado (PID: {process.pid})")
    
    def start_wsgi_server(self, config: Dict):
        """Inicia servidor WSGI para producción"""
        logger.info("🌐 Iniciando servidor WSGI...")
        
        wsgi_config = config.get('wsgi', {})
        workers = wsgi_config.get('workers', 4)
        host = wsgi_config.get('host', '0.0.0.0')
        port = wsgi_config.get('port', 8000)
        
        cmd = [
            sys.executable, "wsgi_production.py",
            "--host", host,
            "--port", str(port),
            "--workers", str(workers)
        ]
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        self.processes['wsgi'] = process
        logger.info(f"✅ Servidor WSGI iniciado (PID: {process.pid})")
    
    def monitor_services(self):
        """Monitorea el estado de los servicios"""
        logger.info("📡 Iniciando monitoreo de servicios...")
        
        while self.running:
            for service_name, process in self.processes.items():
                if process.poll() is not None:
                    logger.error(f"❌ Servicio {service_name} detenido (código: {process.returncode})")
                    # Intentar reiniciar el servicio
                    self._restart_service(service_name)
            
            time.sleep(30)  # Verificar cada 30 segundos
    
    def _restart_service(self, service_name: str):
        """Reinicia un servicio que se detuvo"""
        logger.info(f"🔄 Reiniciando servicio {service_name}...")
        
        # Implementar lógica de reinicio según el servicio
        # Por ahora, solo registrar el intento
        logger.warning(f"⚠️ Reinicio de {service_name} no implementado aún")
    
    def stop_all_services(self):
        """Detiene todos los servicios"""
        logger.info("🛑 Deteniendo todos los servicios...")
        
        for service_name, process in self.processes.items():
            if process.poll() is None:  # El proceso aún está corriendo
                logger.info(f"Deteniendo {service_name} (PID: {process.pid})...")
                process.terminate()
                
                # Esperar hasta 10 segundos para que termine gracefulmente
                try:
                    process.wait(timeout=10)
                    logger.info(f"✅ {service_name} detenido exitosamente")
                except subprocess.TimeoutExpired:
                    logger.warning(f"⚠️ {service_name} no respondió, forzando terminación...")
                    process.kill()
                    process.wait()
    
    def generate_deployment_report(self) -> Dict:
        """Genera reporte de despliegue"""
        report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'services': {},
            'config_path': self.config_path,
            'python_version': sys.version,
            'platform': sys.platform
        }
        
        for service_name, process in self.processes.items():
            status = 'running' if process.poll() is None else 'stopped'
            report['services'][service_name] = {
                'pid': process.pid,
                'status': status,
                'return_code': process.returncode if status == 'stopped' else None
            }
        
        return report
    
    def deploy(self):
        """Ejecuta el proceso completo de despliegue"""
        logger.info("🚀 Iniciando despliegue de producción AEGIS")
        
        # Validar entorno
        if not self.validate_environment():
            logger.error("❌ Validación de entorno falló")
            return False
        
        # Cargar configuración
        config = self.load_production_config()
        
        # Crear directorios
        self.create_directories()
        
        # Generar certificados SSL
        self.generate_ssl_certificates()
        
        # Iniciar servicios
        services_to_start = config.get('services', {})
        
        if services_to_start.get('node', True):
            self.start_node_service(config)
        
        if services_to_start.get('dashboard', True):
            self.start_dashboard_service(config)
        
        if services_to_start.get('wsgi', False):
            self.start_wsgi_server(config)
        
        # Iniciar monitoreo
        try:
            self.monitor_services()
        except KeyboardInterrupt:
            logger.info("🛑 Despliegue interrumpido por usuario")
        finally:
            self.stop_all_services()
            
            # Generar reporte final
            report = self.generate_deployment_report()
            with open('logs/deployment_report.json', 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info("✅ Despliegue finalizado")

def main():
    """Función principal"""
    parser = argparse.ArgumentParser(description="Despliegue de producción AEGIS")
    parser.add_argument('--config', type=str, help='Ruta del archivo de configuración')
    parser.add_argument('--validate-only', action='store_true', help='Solo validar entorno')
    
    args = parser.parse_args()
    
    deployer = ProductionDeployer(config_path=args.config)
    
    if args.validate_only:
        if deployer.validate_environment():
            logger.info("✅ Entorno validado exitosamente")
            return 0
        else:
            logger.error("❌ Validación de entorno falló")
            return 1
    
    try:
        deployer.deploy()
        return 0
    except Exception as e:
        logger.error(f"❌ Error en despliegue: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())