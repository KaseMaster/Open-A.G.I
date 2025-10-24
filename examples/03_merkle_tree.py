#!/usr/bin/env python3
"""
AEGIS Framework - Merkle Tree Example
Ejemplo de creación y verificación de Merkle Trees
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

def main():
    print("🌳 AEGIS Merkle Tree Example")
    print("="*60)
    print()
    
    from aegis.blockchain.merkle_tree import MerkleTree, create_merkle_tree
    
    # 1. Crear árbol con transacciones
    print("1️⃣  Creando Merkle Tree con transacciones")
    
    transactions = [
        b"tx1: Alice -> Bob 10 coins",
        b"tx2: Bob -> Charlie 5 coins",
        b"tx3: Charlie -> David 3 coins",
        b"tx4: David -> Eve 2 coins",
        b"tx5: Eve -> Frank 1 coin"
    ]
    
    tree = create_merkle_tree(transactions, hash_type='sha256')
    print(f"   ✓ Árbol creado con {len(transactions)} transacciones")
    print()
    
    # 2. Obtener raíz Merkle
    print("2️⃣  Raíz del Merkle Tree")
    root = tree.get_merkle_root_hex()
    print(f"   Raíz: {root}")
    print()
    
    # 3. Generar prueba de Merkle
    print("3️⃣  Generando prueba de Merkle para tx1")
    proof = tree.get_proof(0)  # Prueba para primera transacción
    
    print(f"   ✓ Prueba generada con {len(proof)} pasos:")
    for i, step in enumerate(proof, 1):
        position = "derecha" if step['right'] else "izquierda"
        print(f"      Paso {i}: Hash en {position}")
        print(f"               {step['hash'][:32]}...")
    print()
    
    # 4. Validar prueba de Merkle
    print("4️⃣  Validando prueba de Merkle")
    
    # Hash de la transacción original
    tx_hash = tree.leaves[0]
    merkle_root = tree.get_merkle_root()
    
    is_valid = tree.validate_proof(proof, tx_hash, merkle_root)
    print(f"   Transacción: {transactions[0].decode()}")
    print(f"   Prueba válida: {'✅ SÍ' if is_valid else '❌ NO'}")
    print()
    
    # 5. Intentar validar con dato incorrecto
    print("5️⃣  Validando con dato incorrecto")
    
    wrong_tx = b"tx1: Alice -> Bob 100 coins"  # Cantidad incorrecta
    tree_wrong = MerkleTree(hash_type='sha256')
    tree_wrong.add_leaf(wrong_tx, do_hash=True)
    wrong_hash = tree_wrong.leaves[0]
    
    is_valid_wrong = tree.validate_proof(proof, wrong_hash, merkle_root)
    print(f"   Transacción incorrecta: {wrong_tx.decode()}")
    print(f"   Prueba válida: {'✅ SÍ' if is_valid_wrong else '❌ NO (esperado)'}")
    print()
    
    # 6. Demostrar diferentes algoritmos de hash
    print("6️⃣  Probando diferentes algoritmos de hash")
    
    algorithms = ['sha256', 'sha3_256', 'sha512', 'blake2b']
    
    for algo in algorithms:
        tree_algo = create_merkle_tree([b"test"], hash_type=algo)
        root_algo = tree_algo.get_merkle_root_hex()
        print(f"   {algo:12s}: {root_algo[:32]}...")
    
    print()
    print("="*60)
    print("✅ Merkle Tree implementado nativamente")
    print("   - Sin dependencias externas")
    print("   - 4 algoritmos de hash soportados")
    print("   - Generación y validación de pruebas")
    print()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
