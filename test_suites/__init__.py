"""
Test Suites para AEGIS
Paquete que contiene todas las suites de tests del sistema.
"""

__version__ = "1.0.0"
__author__ = "AEGIS Security Team"

# Importaciones principales
from .test_crypto import create_crypto_test_suite
from .test_p2p import create_p2p_test_suite
from .test_integration import create_integration_test_suite
from .test_performance import create_performance_test_suite

__all__ = [
    'create_crypto_test_suite',
    'create_p2p_test_suite', 
    'create_integration_test_suite',
    'create_performance_test_suite'
]