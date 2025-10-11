import sys
import importlib
import asyncio
import json
import os
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
    path = config_path or os.environ.get("AEGIS_CONFIG", "app_config.json")
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
    consensus_mod, cons_err = safe_import("consensus_algorithm")
    monitor_mod, mon_err = safe_import("monitoring_dashboard")
    resource_mod, res_err = safe_import("resource_manager")

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
        "consensus_algorithm",
        "monitoring_dashboard",
        "resource_manager",
    ]
    for m in mods:
        mod, err = safe_import(m)
        if mod:
            logger.success(f"{m}: disponible")
        else:
            logger.warning(f"{m}: no disponible ({err})")


if __name__ == "__main__":
    cli()