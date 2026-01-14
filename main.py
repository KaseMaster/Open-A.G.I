#!/usr/bin/env python3
"""
AEGIS Open AGI - Sistema de IA Distribuida y Seguridad Avanzada
Desarrollador: José María Gómez García
Contacto: kasemaster@protonmail.com
Versión: 2.1.0
Licencia: MIT
"""

import sys
import importlib
import asyncio
import json
import os
import time
from typing import Optional, Tuple, Dict, Any

# Añadir src al path para importar módulos locales
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# Importar configuración de logging mejorada
try:
    from aegis_core.logging_config import logger, get_logger
except ImportError:
    # Fallback básico si no está disponible logging_config
    try:
        from loguru import logger
    except Exception:
        # Fallback mínimo si loguru no está disponible
        class _L:
            def info(self, *a, **k): print(*a)
            def warning(self, *a, **k): print(*a)
            def error(self, *a, **k): print(*a)
            def success(self, *a, **k): print(*a)
        logger = _L()

import click
from dotenv import load_dotenv


def safe_import(module_name: str) -> Tuple[Optional[object], Optional[Exception]]:
    """Importa un módulo de forma segura, devolviendo (módulo, error)."""
    try:
        mod = importlib.import_module(module_name)
        return mod, None
    except Exception as e:
        return None, e


def module_call(mod: object, func_name: str, *args, **kwargs):
    """Llama a una función de un módulo si existe y es callable."""
    if not mod:
        return False
    fn = getattr(mod, func_name, None)
    if callable(fn):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error ejecutando {mod.__name__}.{func_name}: {e}")
            return False
    return False


DEFAULT_CONFIG: Dict[str, Any] = {
    "app": {
        "log_level": "INFO",
        "enable": {
            "tor": True,
            "p2p": True,
            "crypto": True,
            "consensus": True,
            "monitoring": True,
            "resource_manager": True,
        },
    },
    "tor": {
        "enabled": True,
        "socks_port": 9050,
        "control_port": 9051,
        "onion_routing": True,
    },
    "p2p": {
        "discovery": "zeroconf",
        "heartbeat_interval_sec": 30,
    },
    "crypto": {
        "rotate_interval_hours": 24,
        "hash": "blake3",
        "symmetric": "chacha20-poly1305",
    },
    "consensus": {
        "algorithm": "PoC+PBFT",
    },
    "monitoring": {
        "dashboard_port": 8080,
        "enable_socketio": True,
    },
    "security": {
        "rate_limit_per_minute": 120,
        "validate_peer_input": True,
    },
}


def load_config(config_path: Optional[str]) -> Dict[str, Any]:
    """Carga configuración desde JSON y aplica overrides desde .env."""
    load_dotenv()
    cfg: Dict[str, Any] = DEFAULT_CONFIG.copy()
    # Resolver ruta de configuración: prioridad -> parámetro, env, config/app_config.json, app_config.json
    if config_path:
        path = config_path
    else:
        env_path = os.environ.get("AEGIS_CONFIG")
        if env_path:
            path = env_path
        else:
            candidates = [os.path.join("config", "app_config.json"), "app_config.json"]
            path = next((p for p in candidates if os.path.exists(p)), candidates[0])

    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                user_cfg = json.load(f)
            # merge superficial (solo primer nivel y 'app.enable')
            for k, v in user_cfg.items():
                if k == "app" and isinstance(v, dict):
                    cfg["app"].update(v)
                    if "enable" in v and isinstance(v["enable"], dict):
                        cfg["app"]["enable"].update(v["enable"])
                else:
                    cfg[k] = v
        except Exception as e:
            logger.warning(f"No se pudo cargar {path}, usando valores por defecto: {e}")
    else:
        logger.info(f"Config por defecto: no se encontró {path}")

    # Overrides por entorno
    log_level = os.environ.get("AEGIS_LOG_LEVEL")
    if log_level:
        cfg["app"]["log_level"] = log_level

    dash_port = os.environ.get("AEGIS_DASHBOARD_PORT")
    if dash_port and dash_port.isdigit():
        cfg["monitoring"]["dashboard_port"] = int(dash_port)

    return cfg


def health_summary() -> dict:
    summary = {
        "python": sys.version,
        "modules": {},
    }

    for m in [
        "aiohttp",
        "websockets",
        "asyncio",
        "cryptography",
        "pydantic",
        "torch",
    ]:
        mod, err = safe_import(m)
        summary["modules"][m] = {
            "available": mod is not None,
            "error": str(err) if err else None,
        }

    # Comprobación rápida de CUDA (si torch está disponible)
    try:
        import torch  # type: ignore
        summary["cuda"] = {
            "available": bool(torch.cuda.is_available()),
            "device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        }
    except Exception:
        summary["cuda"] = {"available": False, "device_count": 0}

    return summary


async def start_node(dry_run: bool = False, config_path: Optional[str] = None):
    cfg = load_config(config_path)
    logger.info("Iniciando nodo AEGIS Open AGI distribuido...")
    logger.info(f"Config aplicada: enable={cfg['app']['enable']}")

    # Inicializar seguridad si está habilitada
    if cfg.get('security', {}).get('enabled', False):
        try:
            from aegis_core.security_production import get_security_manager, SecurityConfig
            security_config = SecurityConfig(**cfg.get('security', {}))
            security_manager = get_security_manager(security_config)
            logger.info("🔐 Seguridad de producción inicializada")
        except ImportError as e:
            logger.warning(f"⚠️ No se pudo cargar seguridad de producción: {e}")

    tor_mod, tor_err = safe_import("aegis_core.tor_integration")
    p2p_mod, p2p_err = safe_import("aegis_core.p2p_network")
    crypto_mod, crypto_err = safe_import("aegis_core.crypto_framework")
    consensus_mod, cons_err = safe_import("aegis_core.consensus_algorithm")
    monitor_mod, mon_err = safe_import("aegis_core.monitoring_dashboard")
    resource_mod, res_err = safe_import("aegis_core.resource_manager")

    if tor_err:
        logger.warning(f"TOR no disponible: {tor_err}")
    if p2p_err:
        logger.warning(f"P2P no disponible: {p2p_err}")
    if crypto_err:
        logger.warning(f"Crypto no disponible: {crypto_err}")
    if cons_err:
        logger.warning(f"Consenso no disponible: {cons_err}")
    if mon_err:
        logger.warning(f"Monitoreo no disponible: {mon_err}")
    if res_err:
        logger.warning(f"Gestión de recursos no disponible: {res_err}")

    if dry_run:
        logger.info("Dry-run activado: verificando módulos y saliendo.")
        return

    # Inicializaciones seguras (solo si existen)
    if cfg["app"]["enable"].get("tor"):
        module_call(tor_mod, "start_gateway", cfg.get("tor", {}))
    if cfg["app"]["enable"].get("resource_manager"):
        module_call(resource_mod, "init_pool", cfg.get("p2p", {}))
    if cfg["app"]["enable"].get("crypto"):
        module_call(crypto_mod, "initialize_crypto", cfg.get("crypto", {}))
    if cfg["app"]["enable"].get("p2p"):
        module_call(p2p_mod, "start_network", cfg.get("p2p", {}))
    if cfg["app"]["enable"].get("consensus"):
        module_call(consensus_mod, "start_consensus_loop", cfg.get("consensus", {}))
    if cfg["app"]["enable"].get("monitoring"):
        module_call(monitor_mod, "start_dashboard", cfg.get("monitoring", {}))

    logger.success("Nodo inicializado. Procesos en ejecución (si están disponibles).")
    logger.info("✅ AEGIS Open AGI iniciado exitosamente")
    logger.info(f"📊 Dashboard de monitoreo disponible en: http://127.0.0.1:{cfg['monitoring']['dashboard_port']}")

    # Mostrar estado de seguridad si está habilitada
    if cfg.get('security', {}).get('enabled', False):
        try:
            from aegis_core.security_production import get_security_status
            security_status = get_security_status()
            logger.info(f"🔒 Estado de seguridad: {security_status}")
        except ImportError:
            pass

    # Mantener el proceso activo
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("🛑 Deteniendo AEGIS Open AGI...")


@click.group()
def cli():
    """CLI AEGIS - IA Distribuida y Colaborativa."""
    pass


@cli.command()
@click.option("--dry-run", is_flag=True, help="No ejecutar procesos, solo validar módulos.")
@click.option("--config", type=click.Path(exists=False), help="Ruta del archivo de configuración JSON.")
def start_node_cmd(dry_run: bool, config: Optional[str]):
    """Inicia el nodo distribuido (TOR, P2P, Crypto, Consenso, Monitoreo)."""
    try:
        asyncio.run(start_node(dry_run=dry_run, config_path=config))
    except Exception as e:
        logger.error(f"Fallo al iniciar el nodo: {e}")
        sys.exit(1)


@cli.command()
@click.option("--config", type=click.Path(exists=False), help="Ruta del archivo de configuración JSON.")
def health_check(config: Optional[str]):
    """Muestra un resumen de salud del entorno y módulos clave."""
    summary = health_summary()
    logger.info("Resumen de salud:")
    for k, v in summary.items():
        logger.info(f"- {k}: {v}")


@cli.command()
def list_modules():
    """Lista el estado de importación de módulos principales."""
    mods = [
        "aegis_core.tor_integration",
        "aegis_core.p2p_network",
        "aegis_core.crypto_framework",
        "aegis_core.consensus_algorithm",
        "aegis_core.monitoring_dashboard",
        "aegis_core.resource_manager",
    ]
    for m in mods:
        mod, err = safe_import(m)
        if mod:
            logger.success(f"{m}: disponible")
        else:
            logger.warning(f"{m}: no disponible ({err})")


@cli.command(name="start-dashboard")
@click.option("--type", "dashboard_type", type=click.Choice(["monitoring", "web"], case_sensitive=False), default="monitoring", help="Tipo de dashboard a iniciar")
@click.option("--host", type=str, help="Host para el dashboard (por defecto 0.0.0.0)")
@click.option("--port", type=int, help="Puerto para el dashboard")
@click.option("--config", type=click.Path(exists=False), help="Ruta del archivo de configuración JSON.")
def start_dashboard_cmd(dashboard_type: str, host: Optional[str], port: Optional[int], config: Optional[str]):
    """Inicia únicamente el dashboard (monitoring o web)."""
    cfg = load_config(config)

    dashboard_type = (dashboard_type or "monitoring").lower()
    target_host = host or cfg.get("monitoring", {}).get("host", "0.0.0.0")
    target_port = port or int(cfg.get("monitoring", {}).get("dashboard_port", 8080))

    if dashboard_type == "monitoring":
        mod, err = safe_import("aegis_core.monitoring_dashboard")
        if err or not mod:
            logger.error(f"No se pudo importar monitoring_dashboard: {err}")
            sys.exit(1)
        # Construir config para monitoring_dashboard
        mon_cfg = cfg.get("monitoring", {}).copy()
        mon_cfg["host"] = target_host
        mon_cfg["dashboard_port"] = target_port
        logger.info(f"Iniciando Monitoring Dashboard en http://{target_host}:{target_port}")
        res = module_call(mod, "start_dashboard", mon_cfg)
        if res is False:
            logger.error("Fallo al iniciar Monitoring Dashboard")
            sys.exit(1)
        logger.success("Monitoring Dashboard iniciado")
        # Mantener proceso activo para que el servidor siga vivo
        try:
            logger.info("Manteniendo proceso activo. Presiona Ctrl+C para detener.")
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            logger.info("Detención solicitada por usuario")
    else:
        mod, err = safe_import("aegis_core.web_dashboard")
        if err or not mod:
            logger.error(f"No se pudo importar web_dashboard: {err}")
            sys.exit(1)
        # Config para web dashboard
        web_cfg = {
            "host": target_host or "0.0.0.0",
            "port": target_port or 8080,
            "debug": os.environ.get("AEGIS_DEBUG", "0") == "1",
        }
        logger.info(f"Iniciando Web Dashboard en http://{web_cfg['host']}:{web_cfg['port']}")
        try:
            # start_web_dashboard es async, debemos esperar a que inicie
            start_fn = getattr(mod, "start_web_dashboard", None)
            if not callable(start_fn):
                logger.error("web_dashboard.start_web_dashboard no encontrado")
                sys.exit(1)
            asyncio.run(start_fn(web_cfg))
            logger.success("Web Dashboard iniciado")
            # Mantener proceso activo para que el servidor siga vivo
            try:
                logger.info("Manteniendo proceso activo. Presiona Ctrl+C para detener.")
                while True:
                    time.sleep(3600)
            except KeyboardInterrupt:
                logger.info("Detención solicitada por usuario")
        except Exception as e:
            logger.error(f"Error iniciando Web Dashboard: {e}")
            sys.exit(1)


if __name__ == "__main__":
    cli()