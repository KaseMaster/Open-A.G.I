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
            "performance_optimizer": True,
            "logging_system": True,
            "config_manager": True,
            "api_server": True,
            "metrics_collector": True,
            "alert_system": True,
            "web_dashboard": True,
            "backup_system": True,
            "test_framework": True,
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
    "performance": {
        "enable_batching": True,
        "enable_compression": True,
        "batch_size": 100,
        "compression_level": 6,
        "compression_threshold": 1024,
        "analysis_interval": 300,
        "auto_optimize": True,
    },
    "logging": {
        "log_path": "aegis_logs",
        "rotation": {
            "max_file_size": 104857600,
            "max_files": 10,
            "retention_days": 30,
            "compress_old_files": True
        },
        "aggregation_rules": {
            "critical_errors": {
                "time_window": 300,
                "threshold": 5
            },
            "security_attempts": {
                "time_window": 600,
                "threshold": 10
            }
        }
    },
    "config_manager": {
        "base_path": "aegis_config",
        "backup_interval": 3600,
        "auto_backup": True,
        "encryption_enabled": True,
        "watch_files": True
    },
    "api_server": {
        "host": "0.0.0.0",
        "port": 8000,
        "cors_origins": ["*"],
        "trusted_hosts": ["localhost", "127.0.0.1"],
        "rate_limiting": {
            "default": {"requests_per_minute": 60, "requests_per_hour": 1000},
            "admin": {"requests_per_minute": 120, "requests_per_hour": 2000},
            "service": {"requests_per_minute": 300, "requests_per_hour": 5000}
        },
        "jwt": {
            "expire_minutes": 30,
            "algorithm": "HS256"
        },
        "websocket": {
            "enabled": True,
            "max_connections": 100
        }
    },
    "metrics": {
        "collection_interval": 30,
        "storage_path": "aegis_metrics",
        "retention_hours": 24,
        "enable_system_metrics": True,
        "enable_application_metrics": True,
        "alert_rules": {
            "cpu_warning_threshold": 80,
            "cpu_critical_threshold": 90,
            "memory_warning_threshold": 85,
            "memory_critical_threshold": 95,
            "disk_critical_threshold": 95
        },
        "reports": {
            "auto_generate": True,
            "interval_minutes": 60,
            "types": ["system_health", "performance", "alerts"]
        }
    },
    "alerts": {
        "enabled": True,
        "storage_path": "aegis_alerts",
        "retention_days": 30,
        "notification_channels": {
            "email": {
                "enabled": False,
                "smtp_server": "smtp.gmail.com",
                "smtp_port": 587,
                "username": "",
                "password": "",
                "from_address": "aegis@example.com",
                "to_addresses": []
            },
            "webhook": {
                "enabled": False,
                "url": "",
                "headers": {},
                "timeout": 30
            },
            "slack": {
                "enabled": False,
                "webhook_url": "",
                "channel": "#alerts",
                "username": "AEGIS"
            }
        },
        "rules": {
            "cpu_high": {
                "enabled": True,
                "threshold": 85,
                "duration": 300,
                "severity": "warning"
            },
            "memory_high": {
                "enabled": True,
                "threshold": 90,
                "duration": 300,
                "severity": "warning"
            },
            "disk_full": {
                "enabled": True,
                "threshold": 95,
                "duration": 60,
                "severity": "critical"
            },
            "service_down": {
                "enabled": True,
                "severity": "critical"
            }
        },
        "ai_analysis": {
            "enabled": True,
            "pattern_detection": True,
            "predictive_alerts": True,
            "learning_period_days": 7
        }
    },
    "web_dashboard": {
        "enabled": True,
        "host": "0.0.0.0",
        "port": 8080,
        "debug": False,
        "auto_refresh_interval": 5,
        "max_history_points": 100,
        "enable_websockets": True,
        "enable_real_time_charts": True,
        "theme": "dark",
        "authentication": {
            "enabled": False,
            "username": "admin",
            "password": "aegis123"
        },
        "features": {
            "system_monitoring": True,
            "service_status": True,
            "real_time_charts": True,
            "alerts_panel": True,
            "metrics_history": True,
            "log_viewer": True
        }
    },
    "backup_system": {
        "enabled": True,
        "base_path": "aegis_backups",
        "encryption": {
            "enabled": True,
            "key_rotation_days": 90
        },
        "cloud_storage": {
            "enabled": False,
            "provider": "aws_s3",
            "bucket": "aegis-backups",
            "auto_upload": True,
            "aws_access_key": "",
            "aws_secret_key": "",
            "aws_region": "us-east-1"
        },
        "default_jobs": {
            "config_backup": {
                "enabled": True,
                "schedule": "6h",
                "compression": "tar.gz",
                "encryption": True,
                "retention_days": 30
            },
            "logs_backup": {
                "enabled": True,
                "schedule": "daily",
                "compression": "tar.bz2",
                "encryption": False,
                "retention_days": 7
            },
            "data_backup": {
                "enabled": True,
                "schedule": "12h",
                "compression": "zip",
                "encryption": True,
                "retention_days": 60
            }
        },
        "cleanup": {
            "auto_cleanup": True,
            "cleanup_interval": "daily",
            "max_backup_size_gb": 10
        }
    },
    "test_framework": {
        "enabled": True,
        "auto_run_on_start": False,
        "test_types": ["unit", "integration", "performance", "security"],
        "parallel_execution": True,
        "max_workers": 4,
        "timeout_seconds": 300,
        "reports": {
            "enabled": True,
            "output_dir": "test_results",
            "formats": ["json", "html", "console"],
            "detailed_errors": True,
            "performance_metrics": True
        },
        "suites": {
            "crypto": {
                "enabled": True,
                "priority": 1,
                "timeout": 60
            },
            "p2p": {
                "enabled": True,
                "priority": 2,
                "timeout": 120
            },
            "integration": {
                "enabled": True,
                "priority": 3,
                "timeout": 300
            },
            "performance": {
                "enabled": True,
                "priority": 4,
                "timeout": 600
            }
        },
        "continuous_testing": {
            "enabled": False,
            "interval_minutes": 60,
            "on_code_change": True,
            "failure_threshold": 10
        }
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
    consensus_mod, cons_err = safe_import("consensus_algorithm")
    monitor_mod, mon_err = safe_import("monitoring_dashboard")
    resource_mod, res_err = safe_import("resource_manager")
    perf_mod, perf_err = safe_import("performance_optimizer")
    logging_mod, logging_err = safe_import("logging_system")
    config_mod, config_err = safe_import("config_manager")
    api_mod, api_err = safe_import("api_server")
    metrics_mod, metrics_err = safe_import("metrics_collector")
    alert_mod, alert_err = safe_import("alert_system")
    dashboard_mod, dashboard_err = safe_import("web_dashboard")
    backup_mod, backup_err = safe_import("backup_system")
    test_mod, test_err = safe_import("test_framework")

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
    if perf_err:
        logger.warning(f"Optimizador de rendimiento no disponible: {perf_err}")
    if logging_err:
        logger.warning(f"Sistema de logging no disponible: {logging_err}")
    if config_err:
        logger.warning(f"Gestor de configuración no disponible: {config_err}")
    if api_err:
        logger.warning(f"Servidor API no disponible: {api_err}")
    if metrics_err:
        logger.warning(f"Sistema de métricas no disponible: {metrics_err}")
    if alert_err:
        logger.warning(f"Sistema de alertas no disponible: {alert_err}")
    if dashboard_err:
        logger.warning(f"Dashboard web no disponible: {dashboard_err}")
    if backup_err:
        logger.warning(f"Sistema de backup no disponible: {backup_err}")
    if test_err:
        logger.warning(f"Framework de tests no disponible: {test_err}")

    if dry_run:
        logger.info("Modo dry-run: no se ejecutarán procesos.")
        return

    # Inicializar gestor de configuración primero
    if cfg["app"]["enable"].get("config_manager", False) and config_mod:
        try:
            logger.info("🚀 Iniciando gestor de configuración avanzada...")
            await module_call(config_mod, "start_config_system", cfg.get("config_manager", {}))
            logger.info("✅ Gestor de configuración iniciado")
        except Exception as e:
            logger.error(f"❌ Error iniciando gestor de configuración: {e}")

    # Inicializar sistema de logging distribuido
    if cfg["app"]["enable"].get("logging_system", False) and logging_mod:
        try:
            logger.info("🚀 Iniciando sistema de logging distribuido...")
            await module_call(logging_mod, "initialize_logging", 
                            node_id=f"aegis_node_{os.getpid()}", 
                            config=cfg.get("logging", {}))
            logger.info("✅ Sistema de logging distribuido iniciado")
        except Exception as e:
            logger.error(f"❌ Error iniciando sistema de logging: {e}")

    # Inicializar optimizador de rendimiento
    if cfg["app"]["enable"].get("performance_optimizer", False) and perf_mod:
        try:
            logger.info("🚀 Iniciando optimizador de rendimiento...")
            await module_call(perf_mod, "start_optimizer", cfg.get("performance", {}))
            logger.info("✅ Optimizador de rendimiento iniciado")
        except Exception as e:
            logger.error(f"❌ Error iniciando optimizador de rendimiento: {e}")

    # Inicializar TOR
    if cfg["app"]["enable"].get("tor", False) and tor_mod:
        try:
            logger.info("🚀 Iniciando integración TOR...")
            await module_call(tor_mod, "start_tor_service", cfg["tor"])
            logger.info("✅ TOR iniciado")
        except Exception as e:
            logger.error(f"❌ Error iniciando TOR: {e}")

    # Inicializar P2P
    if cfg["app"]["enable"].get("p2p", False) and p2p_mod:
        try:
            logger.info("🚀 Iniciando red P2P...")
            await module_call(p2p_mod, "start_p2p_network", cfg["p2p"])
            logger.info("✅ Red P2P iniciada")
        except Exception as e:
            logger.error(f"❌ Error iniciando P2P: {e}")

    # Inicializar Crypto
    if cfg["app"]["enable"].get("crypto", False) and crypto_mod:
        try:
            logger.info("🚀 Iniciando framework criptográfico...")
            await module_call(crypto_mod, "start_crypto_framework", cfg["crypto"])
            logger.info("✅ Framework criptográfico iniciado")
        except Exception as e:
            logger.error(f"❌ Error iniciando crypto: {e}")

    # Inicializar Consenso
    if cfg["app"]["enable"].get("consensus", False) and consensus_mod:
        try:
            logger.info("🚀 Iniciando algoritmo de consenso...")
            await module_call(consensus_mod, "start_consensus", cfg["consensus"])
            logger.info("✅ Algoritmo de consenso iniciado")
        except Exception as e:
            logger.error(f"❌ Error iniciando consenso: {e}")

    # Inicializar Monitoreo
    if cfg["app"]["enable"].get("monitoring", False) and monitor_mod:
        try:
            logger.info("🚀 Iniciando dashboard de monitoreo...")
            await module_call(monitor_mod, "start_monitoring", cfg["monitoring"])
            logger.info("✅ Dashboard de monitoreo iniciado")
        except Exception as e:
            logger.error(f"❌ Error iniciando monitoreo: {e}")

    # Inicializar Gestión de Recursos
    if cfg["app"]["enable"].get("resource_manager", False) and resource_mod:
        try:
            logger.info("🚀 Iniciando gestión de recursos...")
            await module_call(resource_mod, "start_resource_manager")
            logger.info("✅ Gestión de recursos iniciada")
        except Exception as e:
            logger.error(f"❌ Error iniciando gestión de recursos: {e}")

    # Inicializar Servidor API
    if cfg["app"]["enable"].get("api_server", False) and api_mod:
        try:
            logger.info("🚀 Iniciando servidor API REST...")
            await module_call(api_mod, "start_api_server", cfg.get("api_server", {}))
            logger.info("✅ Servidor API REST iniciado")
        except Exception as e:
            logger.error(f"❌ Error iniciando servidor API: {e}")

    # Inicializar Sistema de Métricas
    if cfg["app"]["enable"].get("metrics_collector", False) and metrics_mod:
        try:
            logger.info("🚀 Iniciando sistema de métricas avanzadas...")
            await module_call(metrics_mod, "start_metrics_collector", cfg.get("metrics", {}))
            logger.info("✅ Sistema de métricas avanzadas iniciado")
        except Exception as e:
            logger.error(f"❌ Error iniciando sistema de métricas: {e}")

    # Inicializar Sistema de Alertas Inteligentes
    if cfg["app"]["enable"].get("alert_system", False) and alert_mod:
        try:
            logger.info("🚀 Iniciando sistema de alertas inteligentes...")
            await module_call(alert_mod, "start_alert_system", cfg.get("alerts", {}))
            logger.info("✅ Sistema de alertas inteligentes iniciado")
        except Exception as e:
            logger.error(f"❌ Error iniciando sistema de alertas: {e}")

    # Inicializar Dashboard Web Interactivo
    if cfg["app"]["enable"].get("web_dashboard", False) and dashboard_mod:
        try:
            logger.info("🚀 Iniciando dashboard web interactivo...")
            await module_call(dashboard_mod, "start_web_dashboard", cfg.get("web_dashboard", {}))
            logger.info("✅ Dashboard web interactivo iniciado")
        except Exception as e:
            logger.error(f"❌ Error iniciando dashboard web: {e}")

    # Inicializar Sistema de Backup Automático
    if cfg["app"]["enable"].get("backup_system", False) and backup_mod:
        try:
            logger.info("🚀 Iniciando sistema de backup automático...")
            await module_call(backup_mod, "start_backup_system", cfg.get("backup_system", {}))
            logger.info("✅ Sistema de backup automático iniciado")
        except Exception as e:
            logger.error(f"❌ Error iniciando sistema de backup: {e}")

    # Inicializar Framework de Tests
    if cfg["app"]["enable"].get("test_framework", False) and test_mod:
        try:
            logger.info("🚀 Iniciando framework de tests...")
            await module_call(test_mod, "start_test_framework", cfg.get("test_framework", {}))
            logger.info("✅ Framework de tests iniciado")
        except Exception as e:
            logger.error(f"❌ Error iniciando framework de tests: {e}")

    logger.info("🎯 Nodo AEGIS completamente iniciado")
    logger.info("Presiona Ctrl+C para detener el nodo")

    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        logger.info("🛑 Deteniendo nodo AEGIS...")
        
        # Detener sistema de logging al final
        if cfg["app"]["enable"].get("logging_system", False) and logging_mod:
            try:
                await module_call(logging_mod, "shutdown_logging")
                logger.info("✅ Sistema de logging detenido")
            except Exception as e:
                logger.error(f"❌ Error deteniendo sistema de logging: {e}")
        
        logger.info("✅ Nodo AEGIS detenido correctamente")


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