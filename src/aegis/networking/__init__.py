"""
AEGIS Networking Module

Componentes de red incluyendo P2P e integración con Tor para comunicaciones
seguras y anónimas.
"""

from aegis.networking import p2p_network
from aegis.networking import tor_integration

__all__ = [
    "p2p_network",
    "tor_integration",
]
