#!/usr/bin/env python3
"""
Script de arranque para AEGIS Open AGI en producción
Inicia el servidor con configuración optimizada
"""

import os
import sys
import subprocess
import signal
import time
import argparse
from pathlib import Path

try:
    from logging_config import get_logger
    logger = get_logger("Production")
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

class AEGISProductionServer:
    """Gestor del servidor de producción AEGIS"""
    
    def __init__(self):
        self.processes = []
        self.running = False
        self.project_root = Path(__file__).parent
        
    def start_production_server(self, server_type='gunicorn', port=8080, workers=None):
        """Inicia el servidor de producción"""
        
        logger.info(f"🚀 Iniciando servidor AEGIS en producción ({server_type})")
        
        # Verificar dependencias
        self._check_dependencies()
        
        # Configurar directorios
        self._setup_directories()
        
        # Iniciar según el tipo de servidor
        if server_type == 'gunicorn':
            return self._start_gunicorn(port, workers)
        elif server_type == 'uvicorn':
            return self._start_uvicorn(port, workers)
        else:
            logger.error(f"❌ Tipo de servidor no soportado: {server_type}")
            return False
    
    def _check_dependencies(self):
        """Verifica que todas las dependencias estén instaladas"""
        required_packages = ['gunicorn', 'psutil']
        missing = []
        
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing.append(package)
        
        if missing:
            logger.warning(f"📦 Paquetes faltantes: {missing}")
            logger.info("💡 Instalando dependencias...")
            
            for package in missing:
                try:
                    subprocess.run([sys.executable, '-m', 'pip', 'install', package], 
                                 check=True, capture_output=True)
                    logger.info(f"✅ {package} instalado")
                except subprocess.CalledProcessError as e:
                    logger.error(f"❌ Error instalando {package}: {e}")
                    raise
    
    def _setup_directories(self):
        """Crea directorios necesarios"""
        directories = ['logs', 'tmp', 'data']
        
        for directory in directories:
            dir_path = self.project_root / directory
            dir_path.mkdir(exist_ok=True)
            logger.debug(f"📁 Directorio verificado: {dir_path}")
    
    def _start_gunicorn(self, port, workers):
        """Inicia servidor con Gunicorn"""
        
        # Detectar número óptimo de workers
        if workers is None:
            import multiprocessing
            workers = min(multiprocessing.cpu_count() * 2 + 1, 8)
        
        logger.info(f"🔧 Configurando Gunicorn con {workers} workers")
        
        # Comando Gunicorn
        cmd = [
            sys.executable, '-m', 'gunicorn',
            '--bind', f'0.0.0.0:{port}',
            '--workers', str(workers),
            '--worker-class', 'sync',
            '--worker-connections', '1000',
            '--max-requests', '1000',
            '--max-requests-jitter', '50',
            '--timeout', '30',
            '--keepalive', '2',
            '--preload',
            '--access-logfile', 'logs/gunicorn_access.log',
            '--error-logfile', 'logs/gunicorn_error.log',
            '--log-level', 'info',
            '--proc-name', 'aegis-server',
            '--pythonpath', str(self.project_root),
            'wsgi_production:app'
        ]
        
        try:
            logger.info(f"🚀 Iniciando Gunicorn en puerto {port}")
            process = subprocess.Popen(cmd, cwd=self.project_root)
            self.processes.append(process)
            
            # Esperar un momento para verificar que inicie correctamente
            time.sleep(3)
            
            if process.poll() is None:
                logger.info(f"✅ Gunicorn iniciado exitosamente (PID: {process.pid})")
                logger.info(f"🔗 Servidor disponible en: http://0.0.0.0:{port}")
                return True
            else:
                logger.error("❌ Gunicorn falló al iniciar")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error iniciando Gunicorn: {e}")
            return False
    
    def _start_uvicorn(self, port, workers):
        """Inicia servidor con Uvicorn"""
        
        # Detectar número óptimo de workers
        if workers is None:
            import multiprocessing
            workers = min(multiprocessing.cpu_count(), 4)
        
        logger.info(f"🔧 Configurando Uvicorn con {workers} workers")
        
        # Comando Uvicorn
        cmd = [
            sys.executable, '-m', 'uvicorn',
            '--host', '0.0.0.0',
            '--port', str(port),
            '--workers', str(workers),
            '--loop', 'uvloop',
            '--http', 'httptools',
            '--limit-concurrency', '100',
            '--limit-max-requests', '1000',
            '--timeout-keep-alive', '5',
            '--access-log',
            '--log-level', 'info',
            'wsgi_production:app'
        ]
        
        try:
            logger.info(f"🚀 Iniciando Uvicorn en puerto {port}")
            process = subprocess.Popen(cmd, cwd=self.project_root)
            self.processes.append(process)
            
            # Esperar un momento para verificar que inicie correctamente
            time.sleep(3)
            
            if process.poll() is None:
                logger.info(f"✅ Uvicorn iniciado exitosamente (PID: {process.pid})")
                logger.info(f"🔗 Servidor disponible en: http://0.0.0.0:{port}")
                return True
            else:
                logger.error("❌ Uvicorn falló al iniciar")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error iniciando Uvicorn: {e}")
            return False
    
    def stop_server(self):
        """Detiene el servidor de producción"""
        logger.info("🛑 Deteniendo servidor de producción")
        
        for process in self.processes:
            try:
                # Enviar SIGTERM
                process.terminate()
                process.wait(timeout=10)
                logger.info(f"✅ Proceso {process.pid} terminado")
            except subprocess.TimeoutExpired:
                # Si no termina, enviar SIGKILL
                process.kill()
                logger.warning(f"⚠️ Proceso {process.pid} terminado forzadamente")
            except Exception as e:
                logger.error(f"❌ Error terminando proceso {process.pid}: {e}")
        
        self.processes.clear()
        logger.info("🛑 Servidor detenido")
    
    def restart_server(self, server_type='gunicorn', port=8080, workers=None):
        """Reinicia el servidor"""
        logger.info("🔄 Reiniciando servidor")
        self.stop_server()
        time.sleep(2)
        return self.start_production_server(server_type, port, workers)
    
    def status(self):
        """Muestra el estado del servidor"""
        if not self.processes:
            logger.info("📊 Servidor no está ejecutándose")
            return False
        
        for process in self.processes:
            try:
                if process.poll() is None:
                    logger.info(f"✅ Proceso activo: PID {process.pid}")
                else:
                    logger.warning(f"⚠️ Proceso terminado: PID {process.pid}")
            except Exception as e:
                logger.error(f"❌ Error verificando proceso: {e}")
        
        return True
    
    def _signal_handler(self, signum, frame):
        """Manejador de señales"""
        logger.info(f"📡 Señal {signum} recibida")
        if signum == signal.SIGTERM:
            self.stop_server()
            sys.exit(0)
        elif signum == signal.SIGHUP:
            self.restart_server()

def main():
    """Función principal"""
    parser = argparse.ArgumentParser(description='AEGIS Production Server')
    parser.add_argument('action', choices=['start', 'stop', 'restart', 'status'],
                       help='Acción a ejecutar')
    parser.add_argument('--server', choices=['gunicorn', 'uvicorn'], default='gunicorn',
                       help='Tipo de servidor WSGI')
    parser.add_argument('--port', type=int, default=8080,
                       help='Puerto del servidor')
    parser.add_argument('--workers', type=int, default=None,
                       help='Número de workers')
    parser.add_argument('--daemon', action='store_true',
                       help='Ejecutar como demonio')
    
    args = parser.parse_args()
    
    server = AEGISProductionServer()
    
    # Configurar manejadores de señales
    signal.signal(signal.SIGTERM, server._signal_handler)
    signal.signal(signal.SIGHUP, server._signal_handler)
    
    if args.action == 'start':
        success = server.start_production_server(args.server, args.port, args.workers)
        if success:
            logger.info("🎉 Servidor de producción iniciado exitosamente")
            if not args.daemon:
                try:
                    # Mantener el proceso activo
                    while True:
                        time.sleep(1)
                except KeyboardInterrupt:
                    logger.info("⏹️ Interrupción detectada")
                    server.stop_server()
        else:
            logger.error("❌ Falló el inicio del servidor")
            sys.exit(1)
    
    elif args.action == 'stop':
        server.stop_server()
    
    elif args.action == 'restart':
        success = server.restart_server(args.server, args.port, args.workers)
        if success:
            logger.info("🔄 Servidor reiniciado exitosamente")
        else:
            logger.error("❌ Falló el reinicio del servidor")
            sys.exit(1)
    
    elif args.action == 'status':
        server.status()

if __name__ == "__main__":
    main()