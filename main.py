#!/usr/bin/env python3
"""
AEGIS Framework - Sistema de IA Distribuida y Seguridad Avanzada
Programador Principal: Jose Gómez alias KaseMaster
Contacto: kasemaster@aegis-framework.com
Versión: 2.0.0
Licencia: MIT
"""

import sys
import importlib
import asyncio
import json
import os
import time
from typing import Optional, Tuple, Dict, Any

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
        "node_id": "aegis_node",
        "enable": {
            "tor": True,
            "p2p": True,
            "crypto": True,
            "identity": True,
            "consensus": True,
            "state_persistence": True,
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
        "security_level": "HIGH",
    },
    "identity": {
        "trust_threshold": 0.7,
        "max_failed_attempts": 5,
    },
    "consensus": {
        "algorithm": "PoC+PBFT",
        "poc_interval_sec": 300,
        "min_computation_score": 50.0,
    },
    "state_persistence": {
        "max_chunk_size": 1048576,  # 1MB
        "replication_factor": 3,
        "checkpoint_interval_sec": 300,
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
    logger.info("Iniciando nodo AEGIS distribuido...")
    logger.info(f"Config aplicada: enable={cfg['app']['enable']}")

    tor_mod, tor_err = safe_import("tor_integration")
    p2p_mod, p2p_err = safe_import("p2p_network")
    crypto_mod, crypto_err = safe_import("crypto_framework")
    identity_mod, identity_err = safe_import("ed25519_identity_manager")
    consensus_mod, cons_err = safe_import("consensus_protocol")  # Updated to new consensus protocol
    state_mod, state_err = safe_import("distributed_state_persistence")
    monitor_mod, mon_err = safe_import("monitoring_dashboard")
    integrated_mod, int_err = safe_import("integrated_dashboard")
    resource_mod, res_err = safe_import("resource_manager")

    if tor_err:
        logger.warning(f"TOR no disponible: {tor_err}")
    if p2p_err:
        logger.warning(f"P2P no disponible: {p2p_err}")
    if crypto_err:
        logger.warning(f"Crypto no disponible: {crypto_err}")
    if identity_err:
        logger.warning(f"Identity Manager no disponible: {identity_err}")
    if cons_err:
        logger.warning(f"Consenso Protocol no disponible: {cons_err}")
    if state_err:
        logger.warning(f"State Persistence no disponible: {state_err}")
    if mon_err:
        logger.warning(f"Monitoreo no disponible: {mon_err}")
    if int_err:
        logger.warning(f"Dashboard Integrado no disponible: {int_err}")
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
    if cfg["app"]["enable"].get("identity", True):  # Nuevo: Identity Manager
        module_call(identity_mod, "Ed25519IdentityManager", cfg.get("app", {}).get("node_id", "aegis_node"))
    if cfg["app"]["enable"].get("p2p"):
        module_call(p2p_mod, "start_network", cfg.get("p2p", {}))
    if cfg["app"]["enable"].get("consensus"):
        module_call(consensus_mod, "HybridConsensus", cfg.get("consensus", {}))
    if cfg["app"]["enable"].get("state_persistence", True):  # Nuevo: State Persistence
        module_call(state_mod, "DistributedStateManager", cfg.get("app", {}).get("node_id", "aegis_node"))
    if cfg["app"]["enable"].get("monitoring"):
        module_call(monitor_mod, "start_dashboard", cfg.get("monitoring", {}))

    logger.success("Nodo inicializado. Procesos en ejecución (si están disponibles).")


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
        "tor_integration",
        "p2p_network",
        "crypto_framework",
        "ed25519_identity_manager",
        "consensus_protocol",
        "distributed_state_persistence",
        "monitoring_dashboard",
        "integrated_dashboard",
        "resource_manager",
    ]
    for m in mods:
        mod, err = safe_import(m)
        if mod:
            logger.success(f"{m}: disponible")
        else:
            logger.warning(f"{m}: no disponible ({err})")


@cli.command(name="start-dashboard")
@click.option("--type", "dashboard_type", type=click.Choice(["monitoring", "web", "integrated"], case_sensitive=False), default="integrated", help="Tipo de dashboard a iniciar")
@click.option("--host", type=str, help="Host para el dashboard (por defecto 0.0.0.0)")
@click.option("--port", type=int, help="Puerto para el dashboard")
@click.option("--config", type=click.Path(exists=False), help="Ruta del archivo de configuración JSON.")
def start_dashboard_cmd(dashboard_type: str, host: Optional[str], port: Optional[int], config: Optional[str]):
    """Inicia únicamente el dashboard (monitoring, web o integrado)."""
    cfg = load_config(config)

    dashboard_type = (dashboard_type or "integrated").lower()
    target_host = host or cfg.get("monitoring", {}).get("host", "0.0.0.0")
    target_port = port or int(cfg.get("monitoring", {}).get("dashboard_port", 8080))

    if dashboard_type == "integrated":
        # Importar y usar dashboard integrado
        try:
            from integrated_dashboard import start_integrated_dashboard

            # Configuración para dashboard integrado
            dashboard_config = {
                "host": target_host,
                "dashboard_port": target_port,
                "node_id": cfg.get("app", {}).get("node_id", "aegis_node"),
                "enable_p2p": cfg["app"]["enable"].get("p2p", True),
                "enable_knowledge_base": cfg["app"]["enable"].get("resource_manager", True),
                "enable_heartbeat": cfg["app"]["enable"].get("monitoring", True),
                "enable_crypto": cfg["app"]["enable"].get("crypto", True),
                "p2p": cfg.get("p2p", {}),
                "knowledge_base": {"node_id": cfg.get("app", {}).get("node_id", "aegis_node")},
                "heartbeat": {
                    "node_id": cfg.get("app", {}).get("node_id", "aegis_node"),
                    "heartbeat_interval_sec": cfg.get("p2p", {}).get("heartbeat_interval_sec", 30),
                    "heartbeat_timeout_sec": 10
                },
                "crypto": {
                    "node_id": cfg.get("app", {}).get("node_id", "aegis_node"),
                    "security_level": cfg.get("crypto", {}).get("security_level", "HIGH")
                }
            }

            logger.info(f"Iniciando Dashboard Integrado en http://{target_host}:{target_port}")
            dashboard = start_integrated_dashboard(dashboard_config)

            # Mantener proceso activo
            try:
                logger.info("Manteniendo dashboard activo. Presiona Ctrl+C para detener.")
                while True:
                    time.sleep(3600)
            except KeyboardInterrupt:
                logger.info("Detención solicitada por usuario")

        except ImportError as e:
            logger.error(f"Error importando dashboard integrado: {e}")
            logger.info("Usando dashboard de respaldo...")
            # Fallback al dashboard original
            dashboard_type = "monitoring"

    if dashboard_type == "monitoring":
        mod, err = safe_import("monitoring_dashboard")
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
        mod, err = safe_import("web_dashboard")
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