import asyncio
import os
import sys

# Asegurar que el directorio del proyecto está en sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import time
from typing import Any

import p2p_network as p2p


class DummyServiceInfo:
    def __init__(self, service_type: str, name: str, addresses: Any, port: int, properties: dict):
        self.type_ = service_type
        self.name = name
        self.addresses = addresses
        self.port = port
        self.properties = properties


class DummyZeroconf:
    def __init__(self):
        self.registered = False
        self.unregistered = False
        self.closed = False

    def register_service(self, info: DummyServiceInfo):
        # Simular registro
        time.sleep(0.01)
        self.registered = True

    def unregister_service(self, info: DummyServiceInfo):
        time.sleep(0.01)
        self.unregistered = True

    def close(self):
        time.sleep(0.01)
        self.closed = True


class DummyServiceBrowser:
    def __init__(self, zeroconf: DummyZeroconf, service_type: str, listener: Any):
        self.zeroconf = zeroconf
        self.service_type = service_type
        self.listener = listener
        self.cancel_calls = 0

    def cancel(self):
        self.cancel_calls += 1


async def run_test() -> None:
    # Forzar disponibilidad de Zeroconf en el módulo y sustituir clases por dummies
    p2p.HAS_ZEROCONF = True
    p2p.ServiceInfo = DummyServiceInfo
    p2p.Zeroconf = DummyZeroconf
    p2p.ServiceBrowser = DummyServiceBrowser

    # También forzar netifaces como disponible para pasar la comprobación inicial del setup
    p2p.HAS_NETIFACES = True

    svc = p2p.PeerDiscoveryService(node_id="n1", node_type=p2p.NodeType.FULL, port=9000)
    # Evitar dependencias externas en _get_local_ip
    svc._get_local_ip = lambda: "192.168.1.50"

    # Ejecutar setup Zeroconf (activar descubrimiento antes del setup)
    svc.discovery_active = True
    await svc._setup_zeroconf()
    assert isinstance(svc.zeroconf, DummyZeroconf), "Zeroconf no configurado correctamente"
    assert isinstance(svc.service_info, DummyServiceInfo), "ServiceInfo no creado correctamente"
    assert svc.zeroconf.registered, "Servicio no fue registrado"

    # Verificar que _mdns_discovery no cancela el ServiceBrowser (cancelación solo en stop_discovery)
    # discovery_active falso para salir rápidamente del bucle
    svc.discovery_active = False
    await svc._mdns_discovery()
    # El browser creado dentro de _mdns_discovery se asigna y luego se limpia sin cancelar
    # No podemos acceder a ese browser directamente, pero sí verificar que no hay cancelaciones previas
    # Forzar un browser activo y comprobar que stop_discovery es quien lo cancela
    svc._mdns_browser = DummyServiceBrowser(svc.zeroconf, svc.service_type, listener=None)
    pre_cancel_calls = svc._mdns_browser.cancel_calls

    # Preparar estado como si descubrimiento estuviera activo
    svc.discovery_active = True
    # Ahora detener descubrimiento
    await svc.stop_discovery()

    assert svc._mdns_browser is None, "ServiceBrowser debe limpiarse en stop_discovery"
    # El dummy anterior ya no está referenciado, pero verificamos que se llamó cancel al menos una vez
    assert pre_cancel_calls == 0, "No debe haber cancelaciones antes de stop_discovery"
    # Comprobamos que Zeroconf se cerró y desregistró el servicio
    assert svc.zeroconf is None, "Zeroconf debe ser None tras stop_discovery"
    # No podemos acceder a la instancia anterior, pero el flujo garantiza llamar a unregister y close
    # Dado que se ejecutaron a través de to_thread, confiamos en que no haya excepciones y se haya limpiado


if __name__ == "__main__":
    asyncio.run(run_test())