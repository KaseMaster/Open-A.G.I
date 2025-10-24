"""
AEGIS - Advanced Encrypted Governance and Intelligence System

Un framework distribuido resistente a computación cuántica para gobernanza
descentralizada e inteligencia artificial.
"""

__version__ = "1.0.0"
__author__ = "AEGIS Development Team"
__license__ = "MIT"

# Los submódulos se importan de forma lazy para evitar dependencias circulares
# y permitir que los módulos individuales se carguen según sea necesario.

__all__ = [
    # Core
    "core",
    "config_manager",
    "logging_system",
    # Networking
    "networking",
    "p2p_network",
    "tor_integration",
    # Security
    "security",
    "crypto_framework",
    "security_protocols",
    # Blockchain
    "blockchain",
    "blockchain_integration",
    "consensus_algorithm",
    "consensus_protocol",
    # Storage
    "storage",
    "backup_system",
    "knowledge_base",
    # Monitoring
    "monitoring",
    "metrics_collector",
    "alert_system",
    "monitoring_dashboard",
    # Optimization
    "optimization",
    "performance_optimizer",
    "resource_manager",
    # API
    "api",
    "api_server",
    "web_dashboard",
    # Deployment
    "deployment",
    "deployment_orchestrator",
    "fault_tolerance",
]


def __getattr__(name):
    """Importación lazy de submódulos para evitar cargar todo al inicio."""
    import importlib
    
    # Mapeo de nombres a rutas de módulos
    module_map = {
        # Paquetes
        "core": "aegis.core",
        "networking": "aegis.networking",
        "security": "aegis.security",
        "blockchain": "aegis.blockchain",
        "storage": "aegis.storage",
        "monitoring": "aegis.monitoring",
        "optimization": "aegis.optimization",
        "api": "aegis.api",
        "deployment": "aegis.deployment",
        # Módulos específicos
        "config_manager": "aegis.core.config_manager",
        "logging_system": "aegis.core.logging_system",
        "p2p_network": "aegis.networking.p2p_network",
        "tor_integration": "aegis.networking.tor_integration",
        "crypto_framework": "aegis.security.crypto_framework",
        "security_protocols": "aegis.security.security_protocols",
        "blockchain_integration": "aegis.blockchain.blockchain_integration",
        "consensus_algorithm": "aegis.blockchain.consensus_algorithm",
        "consensus_protocol": "aegis.blockchain.consensus_protocol",
        "backup_system": "aegis.storage.backup_system",
        "knowledge_base": "aegis.storage.knowledge_base",
        "metrics_collector": "aegis.monitoring.metrics_collector",
        "alert_system": "aegis.monitoring.alert_system",
        "monitoring_dashboard": "aegis.monitoring.monitoring_dashboard",
        "performance_optimizer": "aegis.optimization.performance_optimizer",
        "resource_manager": "aegis.optimization.resource_manager",
        "api_server": "aegis.api.api_server",
        "web_dashboard": "aegis.api.web_dashboard",
        "deployment_orchestrator": "aegis.deployment.deployment_orchestrator",
        "fault_tolerance": "aegis.deployment.fault_tolerance",
    }
    
    if name in module_map:
        module = importlib.import_module(module_map[name])
        globals()[name] = module
        return module
    
    raise AttributeError(f"module 'aegis' has no attribute '{name}'")
