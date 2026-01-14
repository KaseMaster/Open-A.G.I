#!/usr/bin/env python3
"""
Despliegue WSGI de Producción - AEGIS Framework
Script principal para iniciar servidores WSGI en producción
"""

import os
import sys
import subprocess
import logging
import time
from pathlib import Path

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def deploy_production_servers():
    """Desplegar servidores WSGI de producción"""
    logger.info("🚀 Iniciando despliegue de servidores WSGI")
    
    # Configuración de servidores
    servers = [
        {
            "name": "aegis-node",
            "type": "gunicorn",
            "port": 8080,
            "workers": 4,
            "module": "node:app"
        },
        {
            "name": "aegis-api", 
            "type": "gunicorn",
            "port": 8000,
            "workers": 4,
            "module": "api:app"
        },
        {
            "name": "aegis-dashboard",
            "type": "uvicorn", 
            "port": 3000,
            "workers": 2,
            "module": "dashboard:app"
        }
    ]
    
    processes = []
    
    for server in servers:
        logger.info(f"🚀 Iniciando {server['name']} en puerto {server['port']}")
        
        try:
            if server["type"] == "gunicorn":
                cmd = [
                    sys.executable, "-m", "gunicorn",
                    "--bind", f"127.0.0.1:{server['port']}",
                    "--workers", str(server['workers']),
                    "--worker-class", "sync",
                    "--max-requests", "1000",
                    "--timeout", "30",
                    "--keepalive", "2",
                    "--log-level", "info",
                    "--preload",
                    server["module"]
                ]
            else:  # uvicorn
                cmd = [
                    sys.executable, "-m", "uvicorn",
                    server["module"],
                    "--host", "127.0.0.1",
                    "--port", str(server["port"]),
                    "--workers", str(server["workers"]),
                    "--log-level", "info"
                ]
            
            # Iniciar proceso
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            processes.append((server["name"], process))
            
            logger.info(f"✅ {server['name']} iniciado (PID: {process.pid})")
            time.sleep(2)  # Esperar entre servidores
            
        except Exception as e:
            logger.error(f"❌ Error iniciando {server['name']}: {e}")
            return False
    
    logger.info("🎉 Todos los servidores WSGI iniciados exitosamente")
    return True

def check_requirements():
    """Verificar requisitos del sistema"""
    logger.info("🔍 Verificando requisitos...")
    
    # Verificar Python
    if sys.version_info < (3, 8):
        logger.error("❌ Python 3.8+ requerido")
        return False
    
    # Verificar módulos
    required_modules = ["gunicorn", "uvicorn"]
    missing = []
    
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing.append(module)
    
    if missing:
        logger.error(f"❌ Módulos faltantes: {', '.join(missing)}")
        return False
    
    logger.info("✅ Requisitos verificados")
    return True

def main():
    """Función principal"""
    logger.info("🚀 AEGIS Framework - Despliegue WSGI de Producción")
    
    try:
        # Verificar requisitos
        if not check_requirements():
            sys.exit(1)
        
        # Desplegar servidores
        if deploy_production_servers():
            logger.info("🎉 Despliegue completado exitosamente")
            logger.info("📊 Servidores corriendo:")
            logger.info("  • AEGIS Node: http://127.0.0.1:8080")
            logger.info("  • AEGIS API: http://127.0.0.1:8000") 
            logger.info("  • AEGIS Dashboard: http://127.0.0.1:3000")
            
            # Mantener script corriendo
            logger.info("📊 Monitoreo activo. Presiona Ctrl+C para detener.")
            while True:
                time.sleep(30)
                logger.info("🟢 Sistema operativo")
        else:
            logger.error("❌ Despliegue falló")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("⏹️ Interrupción detectada, cerrando...")
        sys.exit(0)
    except Exception as e:
        logger.error(f"❌ Error crítico: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()