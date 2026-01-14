#!/usr/bin/env python3
"""
Gestor avanzado de servidores WSGI/ASGI para AEGIS Open AGI
Soporte para Gunicorn y Uvicorn con monitoreo y recuperación automática
"""

import os
import sys
import time
import signal
import subprocess
import threading
import psutil
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum

try:
    from logging_config import get_logger
    logger = get_logger("WSGI_Manager")
except ImportError:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

class ServerType(Enum):
    """Tipos de servidores soportados"""
    GUNICORN = "gunicorn"
    UVICORN = "uvicorn"

@dataclass
class ServerConfig:
    """Configuración del servidor"""
    server_type: ServerType
    host: str = "127.0.0.1"
    port: int = 8000
    workers: int = 4
    worker_class: str = "sync"
    max_requests: int = 1000
    timeout: int = 30
    keepalive: int = 2
    ssl_enabled: bool = False
    ssl_certfile: Optional[str] = None
    ssl_keyfile: Optional[str] = None
    user: Optional[str] = None
    group: Optional[str] = None
    daemon: bool = False
    log_level: str = "info"
    config_file: Optional[str] = None
    app_module: str = "wsgi_production:app"

class WSGIServerManager:
    """Gestor de servidores WSGI/ASGI con monitoreo avanzado"""
    
    def __init__(self):
        self.processes: Dict[str, subprocess.Popen] = {}
        self.monitoring_thread: Optional[threading.Thread] = None
        self.monitoring_active = False
        self.health_check_interval = 30
        self.max_restart_attempts = 3
        self.restart_delays = [1, 5, 15]  # Segundos entre reinicios
        self.server_configs: Dict[str, ServerConfig] = {}
        self.server_stats: Dict[str, Dict[str, Any]] = {}
        self.setup_signal_handlers()
        
    def setup_signal_handlers(self):
        """Configurar manejadores de señales"""
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGHUP, self._signal_handler)
        logger.info("✅ Manejadores de señales configurados")
    
    def _signal_handler(self, signum, frame):
        """Manejador de señales del sistema"""
        logger.info(f"📡 Señal {signum} recibida")
        if signum in [signal.SIGTERM, signal.SIGINT]:
            self.stop_all_servers()
            sys.exit(0)
        elif signum == signal.SIGHUP:
            self.reload_all_servers()
    
    def add_server(self, name: str, config: ServerConfig) -> bool:
        """Agregar un servidor a gestionar"""
        try:
            self.server_configs[name] = config
            self.server_stats[name] = {
                'status': 'stopped',
                'pid': None,
                'start_time': None,
                'restart_count': 0,
                'last_restart': None,
                'total_requests': 0,
                'errors': 0,
                'memory_usage': 0,
                'cpu_usage': 0
            }
            logger.info(f"✅ Servidor '{name}' agregado: {config.server_type.value}")
            return True
        except Exception as e:
            logger.error(f"❌ Error agregando servidor '{name}': {e}")
            return False
    
    def start_server(self, name: str) -> bool:
        """Iniciar un servidor específico"""
        if name not in self.server_configs:
            logger.error(f"❌ Servidor '{name}' no encontrado")
            return False
        
        config = self.server_configs[name]
        
        # Verificar si ya está corriendo
        if name in self.processes and self.processes[name].poll() is None:
            logger.warning(f"⚠️ Servidor '{name}' ya está corriendo (PID: {self.processes[name].pid})")
            return True
        
        try:
            # Construir comando según el tipo de servidor
            if config.server_type == ServerType.GUNICORN:
                cmd = self._build_gunicorn_command(config)
            elif config.server_type == ServerType.UVICORN:
                cmd = self._build_uvicorn_command(config)
            else:
                logger.error(f"❌ Tipo de servidor no soportado: {config.server_type}")
                return False
            
            logger.info(f"🚀 Iniciando servidor '{name}': {' '.join(cmd)}")
            
            # Configurar entorno
            env = os.environ.copy()
            env.update(self._get_server_environment(config))
            
            # Iniciar proceso
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True,
                env=env,
                preexec_fn=os.setsid if os.name != 'nt' else None
            )
            
            self.processes[name] = process
            self.server_stats[name].update({
                'status': 'starting',
                'pid': process.pid,
                'start_time': time.time(),
                'restart_count': self.server_stats[name]['restart_count'] + 1,
                'last_restart': time.time()
            })
            
            # Esperar un momento para verificar si se inició correctamente
            time.sleep(2)
            
            if process.poll() is None:
                self.server_stats[name]['status'] = 'running'
                logger.info(f"✅ Servidor '{name}' iniciado exitosamente (PID: {process.pid})")
                
                # Iniciar monitoreo si no está activo
                if not self.monitoring_active:
                    self.start_monitoring()
                
                return True
            else:
                # Leer error
                stdout, stderr = process.communicate()
                error_msg = stderr if stderr else stdout
                logger.error(f"❌ Servidor '{name}' falló al iniciar: {error_msg}")
                self.server_stats[name]['status'] = 'failed'
                self.server_stats[name]['errors'] += 1
                return False
                
        except Exception as e:
            logger.error(f"❌ Error iniciando servidor '{name}': {e}")
            self.server_stats[name]['status'] = 'error'
            self.server_stats[name]['errors'] += 1
            return False
    
    def _build_gunicorn_command(self, config: ServerConfig) -> List[str]:
        """Construir comando para Gunicorn"""
        cmd = [
            sys.executable, "-m", "gunicorn",
            "--bind", f"{config.host}:{config.port}",
            "--workers", str(config.workers),
            "--worker-class", config.worker_class,
            "--max-requests", str(config.max_requests),
            "--timeout", str(config.timeout),
            "--keepalive", str(config.keepalive),
            "--log-level", config.log_level,
            "--preload"
        ]
        
        if config.ssl_enabled and config.ssl_certfile and config.ssl_keyfile:
            cmd.extend([
                "--certfile", config.ssl_certfile,
                "--keyfile", config.ssl_keyfile
            ])
        
        if config.user:
            cmd.extend(["--user", config.user])
        
        if config.group:
            cmd.extend(["--group", config.group])
        
        if config.daemon:
            cmd.append("--daemon")
        
        if config.config_file:
            cmd.extend(["--config", config.config_file])
        
        # Agregar configuración personalizada
        cmd.extend(["--config", "gunicorn_config.py"])
        
        # Módulo de aplicación
        cmd.append(config.app_module)
        
        return cmd
    
    def _build_uvicorn_command(self, config: ServerConfig) -> List[str]:
        """Construir comando para Uvicorn"""
        cmd = [
            sys.executable, "-m", "uvicorn",
            config.app_module,
            "--host", config.host,
            "--port", str(config.port),
            "--workers", str(config.workers),
            "--log-level", config.log_level
        ]
        
        if config.ssl_enabled and config.ssl_certfile and config.ssl_keyfile:
            cmd.extend([
                "--ssl-keyfile", config.ssl_keyfile,
                "--ssl-certfile", config.ssl_certfile
            ])
        
        if config.daemon:
            cmd.append("--daemon")
        
        # Agregar configuración personalizada
        cmd.extend(["--config", "uvicorn_config.py"])
        
        return cmd
    
    def _get_server_environment(self, config: ServerConfig) -> Dict[str, str]:
        """Obtener variables de entorno específicas del servidor"""
        env = {}
        
        if config.server_type == ServerType.GUNICORN:
            env.update({
                "GUNICORN_BIND": f"{config.host}:{config.port}",
                "GUNICORN_WORKERS": str(config.workers),
                "GUNICORN_WORKER_CLASS": config.worker_class,
                "GUNICORN_MAX_REQUESTS": str(config.max_requests),
                "GUNICORN_TIMEOUT": str(config.timeout),
                "GUNICORN_KEEPALIVE": str(config.keepalive),
                "GUNICORN_LOG_LEVEL": config.log_level,
                "GUNICORN_DAEMON": str(config.daemon).lower(),
                "GUNICORN_SSL_ENABLED": str(config.ssl_enabled).lower()
            })
            
            if config.ssl_certfile:
                env["GUNICORN_SSL_CERTFILE"] = config.ssl_certfile
            if config.ssl_keyfile:
                env["GUNICORN_SSL_KEYFILE"] = config.ssl_keyfile
            if config.user:
                env["GUNICORN_USER"] = config.user
            if config.group:
                env["GUNICORN_GROUP"] = config.group
        
        elif config.server_type == ServerType.UVICORN:
            env.update({
                "UVICORN_HOST": config.host,
                "UVICORN_PORT": str(config.port),
                "UVICORN_WORKERS": str(config.workers),
                "UVICORN_LOG_LEVEL": config.log_level,
                "UVICORN_DAEMON": str(config.daemon).lower(),
                "UVICORN_SSL_ENABLED": str(config.ssl_enabled).lower()
            })
            
            if config.ssl_certfile:
                env["UVICORN_SSL_CERTFILE"] = config.ssl_certfile
            if config.ssl_keyfile:
                env["UVICORN_SSL_KEYFILE"] = config.ssl_keyfile
        
        return env
    
    def stop_server(self, name: str, timeout: int = 30) -> bool:
        """Detener un servidor específico"""
        if name not in self.processes:
            logger.warning(f"⚠️ Servidor '{name}' no está corriendo")
            return True
        
        try:
            process = self.processes[name]
            logger.info(f"⏹️ Deteniendo servidor '{name}' (PID: {process.pid})...")
            
            # Enviar SIGTERM
            if os.name != 'nt':
                try:
                    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                except ProcessLookupError:
                    pass
            
            # Esperar a que termine
            try:
                process.terminate()
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                logger.warning(f"⏰ Timeout al detener '{name}', forzando terminación...")
                process.kill()
                process.wait(timeout=5)
            
            del self.processes[name]
            self.server_stats[name]['status'] = 'stopped'
            self.server_stats[name]['pid'] = None
            
            logger.info(f"✅ Servidor '{name}' detenido exitosamente")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error deteniendo servidor '{name}': {e}")
            return False
    
    def restart_server(self, name: str) -> bool:
        """Reiniciar un servidor específico"""
        logger.info(f"🔄 Reiniciando servidor '{name}'...")
        
        # Detener servidor
        self.stop_server(name)
        time.sleep(2)  # Esperar antes de reiniciar
        
        # Iniciar servidor
        return self.start_server(name)
    
    def start_all_servers(self) -> Dict[str, bool]:
        """Iniciar todos los servidores configurados"""
        results = {}
        
        for name in self.server_configs:
            results[name] = self.start_server(name)
            time.sleep(1)  # Esperar entre servidores
        
        return results
    
    def stop_all_servers(self) -> Dict[str, bool]:
        """Detener todos los servidores"""
        results = {}
        
        for name in list(self.processes.keys()):
            results[name] = self.stop_server(name)
            time.sleep(0.5)
        
        # Detener monitoreo
        self.stop_monitoring()
        
        return results
    
    def reload_all_servers(self) -> Dict[str, bool]:
        """Recargar configuración de todos los servidores"""
        results = {}
        
        for name in self.server_configs:
            results[name] = self.restart_server(name)
            time.sleep(2)
        
        return results
    
    def get_server_status(self, name: str) -> Dict[str, Any]:
        """Obtener estado de un servidor"""
        if name not in self.server_stats:
            return {}
        
        stats = self.server_stats[name].copy()
        
        # Actualizar información del proceso si está corriendo
        if name in self.processes and stats['pid']:
            try:
                process = psutil.Process(stats['pid'])
                stats['memory_usage'] = process.memory_info().rss / 1024 / 1024  # MB
                stats['cpu_usage'] = process.cpu_percent()
                stats['status'] = process.status()
                stats['uptime'] = time.time() - stats['start_time'] if stats['start_time'] else 0
            except psutil.NoSuchProcess:
                stats['status'] = 'stopped'
                stats['pid'] = None
        
        return stats
    
    def get_all_status(self) -> Dict[str, Dict[str, Any]]:
        """Obtener estado de todos los servidores"""
        status = {}
        
        for name in self.server_configs:
            status[name] = self.get_server_status(name)
        
        return status
    
    def start_monitoring(self):
        """Iniciar monitoreo de servidores"""
        if self.monitoring_active:
            logger.info("📊 Monitoreo ya está activo")
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        logger.info("📊 Monitoreo de servidores iniciado")
    
    def stop_monitoring(self):
        """Detener monitoreo de servidores"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        logger.info("📊 Monitoreo de servidores detenido")
    
    def _monitoring_loop(self):
        """Bucle de monitoreo"""
        while self.monitoring_active:
            try:
                # Verificar cada servidor
                for name in list(self.processes.keys()):
                    if name not in self.processes:
                        continue
                    
                    process = self.processes[name]
                    
                    # Verificar si el proceso sigue vivo
                    if process.poll() is not None:
                        logger.warning(f"⚠️ Servidor '{name}' murió inesperadamente (exit code: {process.returncode})")
                        self.server_stats[name]['status'] = 'crashed'
                        
                        # Intentar reiniciar si no excedió el límite
                        if self.server_stats[name]['restart_count'] < self.max_restart_attempts:
                            delay = self.restart_delays[
                                min(self.server_stats[name]['restart_count'] - 1, len(self.restart_delays) - 1)
                            ]
                            logger.info(f"🔄 Reiniciando servidor '{name}' en {delay} segundos...")
                            time.sleep(delay)
                            self.start_server(name)
                        else:
                            logger.error(f"❌ Servidor '{name}' excedió el límite de reinicios")
                            self.server_stats[name]['status'] = 'failed'
                    
                    # Actualizar estadísticas
                    self.update_server_stats(name)
                
                time.sleep(self.health_check_interval)
                
            except Exception as e:
                logger.error(f"❌ Error en monitoreo: {e}")
                time.sleep(5)
    
    def update_server_stats(self, name: str):
        """Actualizar estadísticas de un servidor"""
        try:
            stats = self.get_server_status(name)
            
            # Registrar información de rendimiento
            if stats.get('memory_usage', 0) > 0:
                logger.debug(f"📊 {name}: Memory={stats['memory_usage']:.1f}MB, CPU={stats.get('cpu_usage', 0):.1f}%")
            
            # Verificar umbrales de recursos
            memory_usage = stats.get('memory_usage', 0)
            cpu_usage = stats.get('cpu_usage', 0)
            
            if memory_usage > 1000:  # 1GB
                logger.warning(f"⚠️ {name}: Alto uso de memoria ({memory_usage:.1f}MB)")
            
            if cpu_usage > 80:
                logger.warning(f"⚠️ {name}: Alto uso de CPU ({cpu_usage:.1f}%)")
                
        except Exception as e:
            logger.error(f"❌ Error actualizando estadísticas de '{name}': {e}")

def main():
    """Función principal para pruebas"""
    manager = WSGIServerManager()
    
    # Configurar servidores de ejemplo
    gunicorn_config = ServerConfig(
        server_type=ServerType.GUNICORN,
        host="127.0.0.1",
        port=8000,
        workers=4,
        log_level="info"
    )
    
    uvicorn_config = ServerConfig(
        server_type=ServerType.UVICORN,
        host="127.0.0.1",
        port=8001,
        workers=2,
        log_level="info"
    )
    
    manager.add_server("api-gunicorn", gunicorn_config)
    manager.add_server("api-uvicorn", uvicorn_config)
    
    # Iniciar servidores
    logger.info("🚀 Iniciando servidores WSGI/ASGI...")
    results = manager.start_all_servers()
    
    for name, success in results.items():
        logger.info(f"{'✅' if success else '❌'} {name}: {'Iniciado' if success else 'Falló'}")
    
    try:
        # Mantener activo
        while True:
            time.sleep(10)
            status = manager.get_all_status()
            for name, stats in status.items():
                logger.info(f"📊 {name}: {stats.get('status', 'unknown')}")
    except KeyboardInterrupt:
        logger.info("⏹️ Deteniendo servidores...")
        manager.stop_all_servers()

if __name__ == "__main__":
    main()