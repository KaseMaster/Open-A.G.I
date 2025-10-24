#!/usr/bin/env python3
"""
Script de compatibilidad para imports durante la migración a la estructura de paquetes aegis.

Este módulo proporciona importaciones compatibles hacia atrás para que el código existente
siga funcionando mientras se migra a la nueva estructura de paquetes src/aegis/.

Se puede usar de dos formas:
1. Importar este módulo al inicio: `import aegis_compat`
2. Los módulos individuales mantienen sus nombres originales como alias
"""

import sys
import os
from pathlib import Path

# Añadir src/ al path de Python para encontrar el paquete aegis
project_root = Path(__file__).parent
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Importar desde la nueva estructura y registrar en sys.modules con los nombres antiguos
try:
    # Core modules
    from aegis.core import config_manager
    from aegis.core import logging_system
    
    # Networking modules
    from aegis.networking import p2p_network
    from aegis.networking import tor_integration
    
    # Security modules
    from aegis.security import crypto_framework
    from aegis.security import security_protocols
    
    # Blockchain modules
    from aegis.blockchain import blockchain_integration
    from aegis.blockchain import consensus_algorithm
    from aegis.blockchain import consensus_protocol
    
    # Storage modules
    from aegis.storage import backup_system
    from aegis.storage import knowledge_base
    
    # Monitoring modules
    from aegis.monitoring import metrics_collector
    from aegis.monitoring import alert_system
    from aegis.monitoring import monitoring_dashboard
    
    # Optimization modules
    from aegis.optimization import performance_optimizer
    from aegis.optimization import resource_manager
    
    # API modules
    from aegis.api import api_server
    from aegis.api import web_dashboard
    
    # Deployment modules
    from aegis.deployment import deployment_orchestrator
    from aegis.deployment import fault_tolerance
    
    # Registrar módulos en sys.modules para compatibilidad hacia atrás
    sys.modules['config_manager'] = config_manager
    sys.modules['logging_system'] = logging_system
    sys.modules['p2p_network'] = p2p_network
    sys.modules['tor_integration'] = tor_integration
    sys.modules['crypto_framework'] = crypto_framework
    sys.modules['security_protocols'] = security_protocols
    sys.modules['blockchain_integration'] = blockchain_integration
    sys.modules['consensus_algorithm'] = consensus_algorithm
    sys.modules['consensus_protocol'] = consensus_protocol
    sys.modules['backup_system'] = backup_system
    sys.modules['knowledge_base'] = knowledge_base
    sys.modules['metrics_collector'] = metrics_collector
    sys.modules['alert_system'] = alert_system
    sys.modules['monitoring_dashboard'] = monitoring_dashboard
    sys.modules['performance_optimizer'] = performance_optimizer
    sys.modules['resource_manager'] = resource_manager
    sys.modules['api_server'] = api_server
    sys.modules['web_dashboard'] = web_dashboard
    sys.modules['deployment_orchestrator'] = deployment_orchestrator
    sys.modules['fault_tolerance'] = fault_tolerance
    
    print("✅ AEGIS: Módulos cargados exitosamente desde src/aegis/")
    
except ImportError as e:
    print(f"⚠️ AEGIS: Error cargando módulos desde src/aegis/: {e}")
    print("   Intentando cargar desde el directorio raíz como respaldo...")
    # Si falla, los imports antiguos desde el root seguirán funcionando

__all__ = [
    'config_manager',
    'logging_system',
    'p2p_network',
    'tor_integration',
    'crypto_framework',
    'security_protocols',
    'blockchain_integration',
    'consensus_algorithm',
    'consensus_protocol',
    'backup_system',
    'knowledge_base',
    'metrics_collector',
    'alert_system',
    'monitoring_dashboard',
    'performance_optimizer',
    'resource_manager',
    'api_server',
    'web_dashboard',
    'deployment_orchestrator',
    'fault_tolerance',
]
