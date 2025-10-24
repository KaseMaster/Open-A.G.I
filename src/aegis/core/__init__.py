"""
AEGIS Core Module

Funcionalidad central del framework AEGIS incluyendo gestión de configuración
y sistema de logging.
"""

# Lazy imports para evitar cargar módulos pesados al inicio
__all__ = [
    "config_manager",
    "logging_system",
]


def __getattr__(name):
    """Importación lazy de módulos."""
    import importlib
    
    if name == "config_manager":
        module = importlib.import_module("aegis.core.config_manager")
        globals()[name] = module
        return module
    elif name == "logging_system":
        module = importlib.import_module("aegis.core.logging_system")
        globals()[name] = module
        return module
    
    raise AttributeError(f"module 'aegis.core' has no attribute '{name}'")
