#!/usr/bin/env python3
"""
Script de prueba para validar la integración TOR en AEGIS
"""

import asyncio
import socket
import socks
import time
from stem import Signal
from stem.control import Controller

# Guardar referencia del socket original para evitar efectos globales
ORIGINAL_SOCKET = socket.socket

def test_tor_connectivity():
    """Prueba la conectividad básica de TOR"""
    print("🔍 Probando conectividad TOR...")
    
    # Configurar proxy SOCKS5 para TOR SOLO durante esta prueba, con resolución remota (rdns)
    socks.set_default_proxy(socks.SOCKS5, '127.0.0.1', 9050, rdns=True)
    socket.socket = socks.socksocket
    
    try:
        import urllib.request
        response = urllib.request.urlopen('http://httpbin.org/ip', timeout=30)
        ip_info = response.read().decode()
        print(f"✅ Conectividad TOR exitosa: {ip_info}")
        return True
    except Exception as e:
        print(f"❌ Error de conectividad TOR: {e}")
        return False
    finally:
        # Restaurar socket original para no afectar otras pruebas (ej. controlador Stem)
        socket.socket = ORIGINAL_SOCKET

def test_tor_control():
    """Prueba el control de TOR via stem"""
    print("🔍 Probando control de TOR...")
    
    try:
        with Controller.from_port(port=9051) as controller:
            # Usar autenticación automática: intentará cookie y/o contraseña
            # No modificar socket global para que la conexión al puerto 9051 sea directa
            controller.authenticate()
            print("✅ Autenticación con controlador TOR exitosa")
            
            # Obtener información del circuito
            circuits = controller.get_circuits()
            print(f"📊 Circuitos activos: {len(circuits)}")
            
            # Crear nuevo circuito
            controller.signal(Signal.NEWNYM)
            print("🔄 Nuevo circuito solicitado")
            
            return True
    except Exception as e:
        print(f"❌ Error de control TOR: {e}")
        return False

def test_onion_service():
    """Prueba el servicio onion"""
    print("🔍 Probando servicio onion...")
    
    try:
        # Leer dirección onion
        with open('onion_service/hostname', 'r') as f:
            onion_address = f.read().strip()
        
        print(f"🧅 Dirección onion: {onion_address}")
        
        # Probar acceso al servicio onion (requiere que el dashboard esté ejecutándose)
        # Usar requests con socks5h para resolución remota via Tor
        import requests
        proxies = {
            'http': 'socks5h://127.0.0.1:9050',
            'https': 'socks5h://127.0.0.1:9050',
        }
        url = f"http://{onion_address}"
        try:
            resp = requests.get(url, proxies=proxies, timeout=45)
            print(f"✅ Servicio onion accesible: {resp.status_code}")
            return True
        except Exception as e:
            print(f"⚠️ Servicio onion no accesible (dashboard puede no estar ejecutándose): {e}")
            return False
            
    except Exception as e:
        print(f"❌ Error probando servicio onion: {e}")
        return False

async def test_tor_integration_module():
    """Prueba el módulo tor_integration.py"""
    print("🔍 Probando módulo tor_integration...")
    
    try:
        from tor_integration import TorGateway
        
        # Crear instancia del gateway
        gateway = TorGateway()
        
        # Intentar inicializar
        success = await gateway.initialize()
        
        if success:
            print("✅ TorGateway inicializado correctamente")
            
            # Probar obtener estado de red
            network_status = await gateway.get_network_status()
            print(f"📊 Estado de red TOR: {network_status}")
            
            # Cerrar correctamente el gateway
            await gateway.shutdown()
            return True
        else:
            print("❌ Error inicializando TorGateway")
            return False
            
    except Exception as e:
        print(f"❌ Error en módulo tor_integration: {e}")
        return False

def main():
    """Función principal de pruebas"""
    print("🛡️ AEGIS TOR Integration Test Suite")
    print("=" * 50)
    
    results = []
    
    # Pruebas síncronas
    results.append(("Conectividad TOR", test_tor_connectivity()))
    results.append(("Control TOR", test_tor_control()))
    results.append(("Servicio Onion", test_onion_service()))
    
    # Pruebas asíncronas
    try:
        loop = asyncio.get_event_loop()
        integration_result = loop.run_until_complete(test_tor_integration_module())
        results.append(("Módulo Integration", integration_result))
    except Exception as e:
        print(f"❌ Error ejecutando pruebas asíncronas: {e}")
        results.append(("Módulo Integration", False))
    
    # Resumen
    print("\n📊 RESUMEN DE PRUEBAS:")
    print("=" * 30)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nResultado: {passed}/{total} pruebas exitosas ({(passed/total)*100:.1f}%)")
    
    if passed == total:
        print("🎉 ¡Todas las pruebas TOR exitosas!")
    else:
        print("⚠️ Algunas pruebas fallaron. Revisar configuración.")

if __name__ == "__main__":
    main()