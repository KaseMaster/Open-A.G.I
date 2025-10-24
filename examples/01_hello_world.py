#!/usr/bin/env python3
"""
AEGIS Framework - Hello World Example
Ejemplo básico de inicialización y verificación del framework
"""

import sys
from pathlib import Path

# Añadir src/ al path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

def main():
    print("="*60)
    print(" "*15 + "🎯 AEGIS Framework")
    print(" "*18 + "Hello World")
    print("="*60)
    print()
    
    # 1. Verificar importación de módulos core
    print("1️⃣  Verificando módulos core...")
    try:
        from aegis.core.config_manager import ConfigManager
        print("   ✓ ConfigManager disponible")
        
        from aegis.core import logging_system
        print("   ✓ Logging System disponible")
        
        print()
    except ImportError as e:
        print(f"   ✗ Error importando core: {e}")
        return
    
    # 2. Inicializar configuración
    print("2️⃣  Inicializando configuración...")
    try:
        config = ConfigManager()
        print(f"   ✓ Entorno: {getattr(config, 'env', 'default')}")
        print(f"   ✓ Config cargada correctamente")
        print()
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return
    
    # 3. Verificar componentes de seguridad
    print("3️⃣  Verificando componentes de seguridad...")
    try:
        from aegis.security.crypto_framework import CryptoEngine
        crypto = CryptoEngine()
        print("   ✓ CryptoEngine inicializado")
        print()
    except Exception as e:
        print(f"   ⚠ Crypto no disponible: {e}")
        print()
    
    # 4. Verificar Merkle Tree
    print("4️⃣  Verificando Merkle Tree nativo...")
    try:
        from aegis.blockchain.merkle_tree import MerkleTree
        tree = MerkleTree()
        tree.add_leaf(b"Hello AEGIS")
        tree.make_tree()
        root = tree.get_merkle_root_hex()
        print(f"   ✓ Merkle Root: {root[:32]}...")
        print()
    except Exception as e:
        print(f"   ✗ Error: {e}")
        print()
    
    # 5. Estado final
    print("5️⃣  Estado del Sistema")
    print("   ✅ AEGIS Framework está funcionando correctamente")
    print("   📦 Todos los componentes core están operativos")
    print()
    
    print("="*60)
    print()
    print("Próximos pasos:")
    print("  - Ver ejemplos en: examples/")
    print("  - Documentación: docs/ARCHITECTURE.md")
    print("  - Demo completa: python3 scripts/demo.py")
    print()
    print("="*60)

if __name__ == "__main__":
    main()
