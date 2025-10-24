#!/usr/bin/env python3
"""
AEGIS Configuration Manager
Sistema avanzado de gestión de configuraciones dinámicas

Características:
- Configuración dinámica en tiempo real
- Validación de esquemas
- Respaldo automático
- Notificaciones de cambios
- Configuración por entornos
- Encriptación de valores sensibles
"""

import json
import os
import asyncio
import hashlib
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Callable, Union
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum
import threading
from cryptography.fernet import Fernet
import yaml

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


class ConfigLevel(Enum):
    """Niveles de configuración por prioridad"""
    SYSTEM = "system"
    USER = "user"
    ENVIRONMENT = "environment"
    RUNTIME = "runtime"


class ConfigType(Enum):
    """Tipos de configuración"""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    LIST = "list"
    DICT = "dict"
    ENCRYPTED = "encrypted"


@dataclass
class ConfigSchema:
    """Esquema de validación para configuraciones"""
    key: str
    config_type: ConfigType
    required: bool = False
    default: Any = None
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    allowed_values: Optional[List[Any]] = None
    description: str = ""
    sensitive: bool = False


@dataclass
class ConfigChange:
    """Registro de cambio de configuración"""
    timestamp: datetime
    key: str
    old_value: Any
    new_value: Any
    level: ConfigLevel
    user: str = "system"
    reason: str = ""


class ConfigValidator:
    """Validador de configuraciones"""
    
    def __init__(self):
        self.schemas: Dict[str, ConfigSchema] = {}
    
    def register_schema(self, schema: ConfigSchema):
        """Registra un esquema de validación"""
        self.schemas[schema.key] = schema
        logger.debug(f"📋 Esquema registrado: {schema.key}")
    
    def validate(self, key: str, value: Any) -> tuple[bool, str]:
        """Valida un valor contra su esquema"""
        if key not in self.schemas:
            return True, "No schema defined"
        
        schema = self.schemas[key]
        
        # Verificar tipo
        if not self._validate_type(value, schema.config_type):
            return False, f"Invalid type for {key}. Expected {schema.config_type.value}"
        
        # Verificar rango
        if schema.min_value is not None and value < schema.min_value:
            return False, f"Value {value} below minimum {schema.min_value}"
        
        if schema.max_value is not None and value > schema.max_value:
            return False, f"Value {value} above maximum {schema.max_value}"
        
        # Verificar valores permitidos
        if schema.allowed_values and value not in schema.allowed_values:
            return False, f"Value {value} not in allowed values: {schema.allowed_values}"
        
        return True, "Valid"
    
    def _validate_type(self, value: Any, expected_type: ConfigType) -> bool:
        """Valida el tipo de un valor"""
        type_map = {
            ConfigType.STRING: str,
            ConfigType.INTEGER: int,
            ConfigType.FLOAT: (int, float),
            ConfigType.BOOLEAN: bool,
            ConfigType.LIST: list,
            ConfigType.DICT: dict,
            ConfigType.ENCRYPTED: str
        }
        
        expected = type_map.get(expected_type)
        if expected is None:
            return True
        
        return isinstance(value, expected)


class ConfigEncryption:
    """Manejo de encriptación para valores sensibles"""
    
    def __init__(self, key_file: str = "config.key"):
        self.key_file = Path(key_file)
        self.cipher = self._load_or_create_key()
    
    def _load_or_create_key(self) -> Fernet:
        """Carga o crea una clave de encriptación"""
        if self.key_file.exists():
            with open(self.key_file, 'rb') as f:
                key = f.read()
        else:
            key = Fernet.generate_key()
            with open(self.key_file, 'wb') as f:
                f.write(key)
            logger.info(f"🔐 Nueva clave de encriptación creada: {self.key_file}")
        
        return Fernet(key)
    
    def encrypt(self, value: str) -> str:
        """Encripta un valor"""
        return self.cipher.encrypt(value.encode()).decode()
    
    def decrypt(self, encrypted_value: str) -> str:
        """Desencripta un valor"""
        return self.cipher.decrypt(encrypted_value.encode()).decode()


class ConfigWatcher:
    """Observador de cambios en archivos de configuración"""
    
    def __init__(self, config_manager):
        self.config_manager = config_manager
        self.watching = False
        self.watch_thread = None
        self.file_hashes: Dict[str, str] = {}
    
    def start_watching(self):
        """Inicia el monitoreo de archivos"""
        if self.watching:
            return
        
        self.watching = True
        self.watch_thread = threading.Thread(target=self._watch_loop, daemon=True)
        self.watch_thread.start()
        logger.info("👁️ Iniciado monitoreo de archivos de configuración")
    
    def stop_watching(self):
        """Detiene el monitoreo"""
        self.watching = False
        if self.watch_thread:
            self.watch_thread.join(timeout=1)
    
    def _watch_loop(self):
        """Bucle principal de monitoreo"""
        while self.watching:
            try:
                for config_file in self.config_manager.config_files:
                    if os.path.exists(config_file):
                        current_hash = self._get_file_hash(config_file)
                        if config_file in self.file_hashes:
                            if self.file_hashes[config_file] != current_hash:
                                logger.info(f"📁 Cambio detectado en {config_file}")
                                self.config_manager._reload_file(config_file)
                        self.file_hashes[config_file] = current_hash
                
                time.sleep(1)  # Verificar cada segundo
            except Exception as e:
                logger.error(f"Error en monitoreo de archivos: {e}")
                time.sleep(5)
    
    def _get_file_hash(self, filepath: str) -> str:
        """Calcula hash MD5 de un archivo"""
        hash_md5 = hashlib.md5()
        try:
            with open(filepath, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
        except Exception:
            return ""
        return hash_md5.hexdigest()


class ConfigManager:
    """Gestor principal de configuraciones AEGIS"""
    
    def __init__(self, base_path: str = "config"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True)
        
        # Configuraciones por nivel
        self.configs: Dict[ConfigLevel, Dict[str, Any]] = {
            level: {} for level in ConfigLevel
        }
        
        # Componentes
        self.validator = ConfigValidator()
        self.encryption = ConfigEncryption(self.base_path / "config.key")
        self.watcher = ConfigWatcher(self)
        
        # Archivos de configuración
        self.config_files = [
            str(self.base_path / "system.json"),
            str(self.base_path / "user.json"),
            str(self.base_path / "environment.json")
        ]
        
        # Callbacks y estado
        self.change_callbacks: List[Callable] = []
        self.change_history: List[ConfigChange] = []
        self.backup_interval = 3600  # 1 hora
        self.last_backup = time.time()
        
        # Inicialización
        self._load_all_configs()
        self._register_default_schemas()
        self.watcher.start_watching()
        
        logger.info("⚙️ ConfigManager inicializado correctamente")
    
    def _register_default_schemas(self):
        """Registra esquemas por defecto para AEGIS"""
        schemas = [
            ConfigSchema("app.log_level", ConfigType.STRING, True, "INFO", 
                        allowed_values=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]),
            ConfigSchema("tor.socks_port", ConfigType.INTEGER, True, 9050, 1024, 65535),
            ConfigSchema("tor.control_port", ConfigType.INTEGER, True, 9051, 1024, 65535),
            ConfigSchema("p2p.heartbeat_interval_sec", ConfigType.INTEGER, True, 30, 5, 300),
            ConfigSchema("monitoring.dashboard_port", ConfigType.INTEGER, True, 8080, 1024, 65535),
            ConfigSchema("performance.batch_size", ConfigType.INTEGER, True, 100, 1, 10000),
            ConfigSchema("performance.compression_level", ConfigType.INTEGER, True, 6, 1, 9),
            ConfigSchema("crypto.api_key", ConfigType.ENCRYPTED, False, sensitive=True),
            ConfigSchema("database.password", ConfigType.ENCRYPTED, False, sensitive=True),
        ]
        
        for schema in schemas:
            self.validator.register_schema(schema)
    
    def _load_all_configs(self):
        """Carga todas las configuraciones"""
        level_files = {
            ConfigLevel.SYSTEM: self.base_path / "system.json",
            ConfigLevel.USER: self.base_path / "user.json",
            ConfigLevel.ENVIRONMENT: self.base_path / "environment.json"
        }
        
        for level, file_path in level_files.items():
            if file_path.exists():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        self.configs[level] = json.load(f)
                    logger.debug(f"📖 Configuración {level.value} cargada desde {file_path}")
                except Exception as e:
                    logger.error(f"Error cargando {file_path}: {e}")
                    self.configs[level] = {}
    
    def _reload_file(self, filepath: str):
        """Recarga un archivo de configuración específico"""
        try:
            # Determinar nivel por nombre de archivo
            filename = Path(filepath).name
            level_map = {
                "system.json": ConfigLevel.SYSTEM,
                "user.json": ConfigLevel.USER,
                "environment.json": ConfigLevel.ENVIRONMENT
            }
            
            level = level_map.get(filename)
            if not level:
                return
            
            with open(filepath, 'r', encoding='utf-8') as f:
                new_config = json.load(f)
            
            # Detectar cambios
            old_config = self.configs[level].copy()
            self.configs[level] = new_config
            
            # Notificar cambios
            self._notify_changes(old_config, new_config, level)
            
        except Exception as e:
            logger.error(f"Error recargando {filepath}: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Obtiene un valor de configuración con prioridad por niveles"""
        # Orden de prioridad: RUNTIME > ENVIRONMENT > USER > SYSTEM
        priority_order = [ConfigLevel.RUNTIME, ConfigLevel.ENVIRONMENT, 
                         ConfigLevel.USER, ConfigLevel.SYSTEM]
        
        for level in priority_order:
            value = self._get_nested_value(self.configs[level], key)
            if value is not None:
                # Desencriptar si es necesario
                if self._is_encrypted_key(key):
                    try:
                        return self.encryption.decrypt(value)
                    except Exception:
                        logger.warning(f"No se pudo desencriptar {key}")
                        return value
                return value
        
        return default
    
    def set(self, key: str, value: Any, level: ConfigLevel = ConfigLevel.RUNTIME, 
            user: str = "system", reason: str = "") -> bool:
        """Establece un valor de configuración"""
        # Validar valor
        is_valid, error_msg = self.validator.validate(key, value)
        if not is_valid:
            logger.error(f"Validación fallida para {key}: {error_msg}")
            return False
        
        # Encriptar si es necesario
        if self._is_encrypted_key(key) and isinstance(value, str):
            value = self.encryption.encrypt(value)
        
        # Obtener valor anterior
        old_value = self.get(key)
        
        # Establecer valor
        self._set_nested_value(self.configs[level], key, value)
        
        # Registrar cambio
        change = ConfigChange(
            timestamp=datetime.now(),
            key=key,
            old_value=old_value,
            new_value=value,
            level=level,
            user=user,
            reason=reason
        )
        self.change_history.append(change)
        
        # Notificar callbacks
        self._notify_callbacks(change)
        
        # Guardar si no es runtime
        if level != ConfigLevel.RUNTIME:
            self._save_level(level)
        
        logger.info(f"⚙️ Configuración actualizada: {key} = {value} ({level.value})")
        return True
    
    def _get_nested_value(self, config: Dict, key: str) -> Any:
        """Obtiene un valor anidado usando notación de puntos"""
        keys = key.split('.')
        current = config
        
        for k in keys:
            if isinstance(current, dict) and k in current:
                current = current[k]
            else:
                return None
        
        return current
    
    def _set_nested_value(self, config: Dict, key: str, value: Any):
        """Establece un valor anidado usando notación de puntos"""
        keys = key.split('.')
        current = config
        
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        
        current[keys[-1]] = value
    
    def _is_encrypted_key(self, key: str) -> bool:
        """Verifica si una clave debe ser encriptada"""
        schema = self.validator.schemas.get(key)
        return schema and (schema.config_type == ConfigType.ENCRYPTED or schema.sensitive)
    
    def _save_level(self, level: ConfigLevel):
        """Guarda un nivel de configuración a archivo"""
        if level == ConfigLevel.RUNTIME:
            return
        
        filename_map = {
            ConfigLevel.SYSTEM: "system.json",
            ConfigLevel.USER: "user.json",
            ConfigLevel.ENVIRONMENT: "environment.json"
        }
        
        filename = filename_map.get(level)
        if not filename:
            return
        
        filepath = self.base_path / filename
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.configs[level], f, indent=2, ensure_ascii=False)
            logger.debug(f"💾 Configuración {level.value} guardada en {filepath}")
        except Exception as e:
            logger.error(f"Error guardando {filepath}: {e}")
    
    def _notify_callbacks(self, change: ConfigChange):
        """Notifica a los callbacks registrados"""
        for callback in self.change_callbacks:
            try:
                callback(change)
            except Exception as e:
                logger.error(f"Error en callback de configuración: {e}")
    
    def _notify_changes(self, old_config: Dict, new_config: Dict, level: ConfigLevel):
        """Notifica cambios detectados en archivos"""
        # Implementación simplificada - detecta cambios básicos
        for key, value in new_config.items():
            if key not in old_config or old_config[key] != value:
                change = ConfigChange(
                    timestamp=datetime.now(),
                    key=key,
                    old_value=old_config.get(key),
                    new_value=value,
                    level=level,
                    user="file_watcher",
                    reason="File change detected"
                )
                self.change_history.append(change)
                self._notify_callbacks(change)
    
    def register_callback(self, callback: Callable[[ConfigChange], None]):
        """Registra un callback para cambios de configuración"""
        self.change_callbacks.append(callback)
        logger.debug("📞 Callback de configuración registrado")
    
    def get_all(self, level: Optional[ConfigLevel] = None) -> Dict[str, Any]:
        """Obtiene todas las configuraciones de un nivel o combinadas"""
        if level:
            return self.configs[level].copy()
        
        # Combinar todos los niveles con prioridad
        combined = {}
        for level in [ConfigLevel.SYSTEM, ConfigLevel.USER, 
                     ConfigLevel.ENVIRONMENT, ConfigLevel.RUNTIME]:
            self._merge_configs(combined, self.configs[level])
        
        return combined
    
    def _merge_configs(self, target: Dict, source: Dict):
        """Fusiona configuraciones recursivamente"""
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._merge_configs(target[key], value)
            else:
                target[key] = value
    
    def backup_configs(self):
        """Crea respaldo de todas las configuraciones"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = self.base_path / "backups" / timestamp
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        for level in [ConfigLevel.SYSTEM, ConfigLevel.USER, ConfigLevel.ENVIRONMENT]:
            if self.configs[level]:
                filename = f"{level.value}.json"
                backup_file = backup_dir / filename
                
                try:
                    with open(backup_file, 'w', encoding='utf-8') as f:
                        json.dump(self.configs[level], f, indent=2, ensure_ascii=False)
                except Exception as e:
                    logger.error(f"Error creando respaldo {backup_file}: {e}")
        
        self.last_backup = time.time()
        logger.info(f"💾 Respaldo de configuraciones creado: {backup_dir}")
    
    def restore_backup(self, backup_timestamp: str):
        """Restaura configuraciones desde un respaldo"""
        backup_dir = self.base_path / "backups" / backup_timestamp
        
        if not backup_dir.exists():
            logger.error(f"Respaldo no encontrado: {backup_dir}")
            return False
        
        try:
            for level in [ConfigLevel.SYSTEM, ConfigLevel.USER, ConfigLevel.ENVIRONMENT]:
                backup_file = backup_dir / f"{level.value}.json"
                if backup_file.exists():
                    with open(backup_file, 'r', encoding='utf-8') as f:
                        self.configs[level] = json.load(f)
                    self._save_level(level)
            
            logger.info(f"🔄 Configuraciones restauradas desde: {backup_timestamp}")
            return True
        except Exception as e:
            logger.error(f"Error restaurando respaldo: {e}")
            return False
    
    def get_change_history(self, key: Optional[str] = None, 
                          since: Optional[datetime] = None) -> List[ConfigChange]:
        """Obtiene historial de cambios"""
        history = self.change_history
        
        if key:
            history = [c for c in history if c.key == key]
        
        if since:
            history = [c for c in history if c.timestamp >= since]
        
        return sorted(history, key=lambda x: x.timestamp, reverse=True)
    
    def export_config(self, filepath: str, level: Optional[ConfigLevel] = None):
        """Exporta configuración a archivo"""
        config = self.get_all(level)
        
        try:
            if filepath.endswith('.yaml') or filepath.endswith('.yml'):
                with open(filepath, 'w', encoding='utf-8') as f:
                    yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
            else:
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(config, f, indent=2, ensure_ascii=False)
            
            logger.info(f"📤 Configuración exportada a: {filepath}")
        except Exception as e:
            logger.error(f"Error exportando configuración: {e}")
    
    def import_config(self, filepath: str, level: ConfigLevel = ConfigLevel.USER):
        """Importa configuración desde archivo"""
        try:
            if filepath.endswith('.yaml') or filepath.endswith('.yml'):
                with open(filepath, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
            else:
                with open(filepath, 'r', encoding='utf-8') as f:
                    config = json.load(f)
            
            # Validar y establecer cada configuración
            for key, value in self._flatten_dict(config).items():
                self.set(key, value, level, user="import", reason=f"Imported from {filepath}")
            
            logger.info(f"📥 Configuración importada desde: {filepath}")
        except Exception as e:
            logger.error(f"Error importando configuración: {e}")
    
    def _flatten_dict(self, d: Dict, parent_key: str = '', sep: str = '.') -> Dict:
        """Aplana un diccionario anidado"""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
    
    async def start_auto_backup(self):
        """Inicia respaldos automáticos"""
        while True:
            try:
                if time.time() - self.last_backup >= self.backup_interval:
                    self.backup_configs()
                await asyncio.sleep(300)  # Verificar cada 5 minutos
            except Exception as e:
                logger.error(f"Error en respaldo automático: {e}")
                await asyncio.sleep(60)
    
    def shutdown(self):
        """Cierra el gestor de configuraciones"""
        self.watcher.stop_watching()
        self.backup_configs()
        logger.info("⚙️ ConfigManager cerrado correctamente")


# Instancia global
config_manager = None


def get_config_manager() -> ConfigManager:
    """Obtiene la instancia global del gestor de configuraciones"""
    global config_manager
    if config_manager is None:
        config_manager = ConfigManager()
    return config_manager


def init_config_manager(base_path: str = "config") -> ConfigManager:
    """Inicializa el gestor de configuraciones"""
    global config_manager
    config_manager = ConfigManager(base_path)
    return config_manager


async def start_config_system(config: Dict[str, Any] = None):
    """Inicia el sistema de configuración AEGIS"""
    try:
        manager = get_config_manager()
        
        # Configurar desde parámetros si se proporcionan
        if config:
            for key, value in manager._flatten_dict(config).items():
                manager.set(key, value, ConfigLevel.RUNTIME, user="system", 
                           reason="System initialization")
        
        # Iniciar respaldos automáticos
        asyncio.create_task(manager.start_auto_backup())
        
        logger.success("⚙️ Sistema de configuración AEGIS iniciado correctamente")
        
    except Exception as e:
        logger.error(f"Error iniciando sistema de configuración: {e}")
        raise


if __name__ == "__main__":
    # Demostración del sistema de configuración
    async def demo():
        print("🔧 Demostración del Sistema de Configuración AEGIS")
        
        # Inicializar
        manager = init_config_manager("demo_config")
        
        # Establecer algunas configuraciones
        manager.set("app.name", "AEGIS Demo", ConfigLevel.USER)
        manager.set("app.version", "2.0.0", ConfigLevel.SYSTEM)
        manager.set("database.host", "localhost", ConfigLevel.ENVIRONMENT)
        manager.set("database.password", "secret123", ConfigLevel.USER)
        
        # Leer configuraciones
        print(f"App Name: {manager.get('app.name')}")
        print(f"App Version: {manager.get('app.version')}")
        print(f"DB Host: {manager.get('database.host')}")
        print(f"DB Password: {manager.get('database.password')}")
        
        # Mostrar historial
        print("\n📋 Historial de cambios:")
        for change in manager.get_change_history()[:5]:
            print(f"  {change.timestamp}: {change.key} = {change.new_value}")
        
        # Exportar configuración
        manager.export_config("demo_export.json")
        
        # Crear respaldo
        manager.backup_configs()
        
        print("\n✅ Demostración completada")
        manager.shutdown()
    
    asyncio.run(demo())