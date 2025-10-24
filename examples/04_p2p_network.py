#!/usr/bin/env python3
"""
AEGIS Framework - P2P Network Example
Ejemplo de tipos de mensajes y componentes de red P2P
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

def main():
    print("🌐 AEGIS P2P Network Example")
    print("="*60)
    print()
    
    from aegis.networking.p2p_network import MessageType
    
    # 1. Tipos de mensajes P2P
    print("1️⃣  Tipos de Mensajes P2P")
    print()
    print("   AEGIS soporta los siguientes tipos de mensajes:")
    print()
    
    for msg_type in MessageType:
        # Descripción de cada tipo
        descriptions = {
            'discovery': 'Descubrimiento de nuevos nodos',
            'handshake': 'Establecimiento de conexión',
            'heartbeat': 'Verificación de estado (keep-alive)',
            'data': 'Transferencia de datos',
            'consensus': 'Mensajes de consenso PBFT',
            'broadcast': 'Difusión a toda la red',
            'request': 'Solicitud de información'
        }
        
        desc = descriptions.get(msg_type.value, 'Mensaje de propósito general')
        print(f"   📨 {msg_type.name:15s} - {desc}")
    
    print()
    
    # 2. Arquitectura de red
    print("2️⃣  Arquitectura de Red P2P")
    print()
    print("   Componentes principales:")
    print("   ├─ DHT (Distributed Hash Table)")
    print("   │  └─ Descubrimiento descentralizado de nodos")
    print("   ├─ Bootstrap Nodes")
    print("   │  └─ Puntos de entrada a la red")
    print("   ├─ Connection Pool")
    print("   │  └─ Gestión eficiente de conexiones")
    print("   ├─ Routing Table")
    print("   │  └─ Enrutamiento óptimo de mensajes")
    print("   └─ mDNS Discovery")
    print("      └─ Descubrimiento en red local")
    print()
    
    # 3. Características de la red
    print("3️⃣  Características de la Red")
    print()
    print("   ✓ Descentralizada - Sin punto único de fallo")
    print("   ✓ Auto-organizada - Topología dinámica")
    print("   ✓ Tolerante a fallos - Reconexión automática")
    print("   ✓ Escalable - Soporta 100+ nodos")
    print("   ✓ Segura - Opcional: canales encriptados + TOR")
    print()
    
    # 4. Flujo de conexión
    print("4️⃣  Flujo de Conexión a la Red")
    print()
    print("   Paso 1: Inicializar nodo local")
    print("   Paso 2: Conectar a bootstrap nodes")
    print("   Paso 3: Intercambiar handshake")
    print("   Paso 4: Sincronizar DHT")
    print("   Paso 5: Comenzar heartbeat (cada 30s)")
    print("   Paso 6: Descubrir peers adicionales")
    print("   Paso 7: Participar en consenso")
    print()
    
    # 5. Ejemplo de estructura de mensaje
    print("5️⃣  Estructura de Mensaje P2P")
    print()
    print("   {")
    print("     'type': 'discovery',           # Tipo de mensaje")
    print("     'sender': 'node_abc123',       # ID del remitente")
    print("     'timestamp': 1698765432.123,   # Timestamp Unix")
    print("     'payload': {                   # Datos del mensaje")
    print("       'node_id': 'node_abc123',")
    print("       'address': '192.168.1.100:8000',")
    print("       'capabilities': ['consensus', 'storage']")
    print("     },")
    print("     'signature': 'abcd1234...'     # Firma digital")
    print("   }")
    print()
    
    # 6. Métricas de red
    print("6️⃣  Métricas de Red (Ejemplo)")
    print()
    print("   📊 Nodos activos: 42")
    print("   📊 Conexiones: 128")
    print("   📊 Mensajes/s: 1,234")
    print("   📊 Latencia promedio: 45ms")
    print("   📊 Uptime: 99.5%")
    print()
    
    print("="*60)
    print("✅ Red P2P AEGIS (84 KB de código)")
    print("   Ver implementación completa en:")
    print("   src/aegis/networking/p2p_network.py")
    print()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
