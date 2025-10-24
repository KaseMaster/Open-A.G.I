"""
Implementación nativa de Merkle Tree para AEGIS Framework
Reemplaza la dependencia merkletools
"""

import hashlib
from typing import List, Optional


class MerkleTree:
    """
    Implementación de Merkle Tree para verificación de integridad de datos
    """
    
    def __init__(self, hash_type: str = 'sha256'):
        """
        Inicializa el Merkle Tree
        
        Args:
            hash_type: Algoritmo de hash ('sha256', 'sha3_256', etc.)
        """
        self.hash_type = hash_type
        self.leaves: List[bytes] = []
        self.levels: List[List[bytes]] = []
        self.is_ready = False
        
        self._hash_function = self._get_hash_function(hash_type)
    
    def _get_hash_function(self, hash_type: str):
        """Obtiene la función de hash según el tipo"""
        hash_functions = {
            'sha256': hashlib.sha256,
            'sha3_256': hashlib.sha3_256,
            'sha512': hashlib.sha512,
            'blake2b': hashlib.blake2b,
        }
        
        if hash_type not in hash_functions:
            raise ValueError(f"Hash type '{hash_type}' not supported")
        
        return hash_functions[hash_type]
    
    def _hash(self, data: bytes) -> bytes:
        """Calcula el hash de los datos"""
        return self._hash_function(data).digest()
    
    def add_leaf(self, value: bytes, do_hash: bool = True) -> None:
        """
        Añade una hoja al árbol
        
        Args:
            value: Valor a añadir
            do_hash: Si True, calcula el hash del valor
        """
        self.is_ready = False
        
        if do_hash:
            value = self._hash(value)
        
        self.leaves.append(value)
    
    def make_tree(self) -> None:
        """Construye el árbol Merkle"""
        if len(self.leaves) == 0:
            raise ValueError("No se pueden construir árbol sin hojas")
        
        self.levels = [self.leaves[:]]
        
        while len(self.levels[-1]) > 1:
            current_level = self.levels[-1]
            next_level = []
            
            for i in range(0, len(current_level), 2):
                left = current_level[i]
                
                if i + 1 < len(current_level):
                    right = current_level[i + 1]
                else:
                    right = left
                
                combined = left + right
                parent_hash = self._hash(combined)
                next_level.append(parent_hash)
            
            self.levels.append(next_level)
        
        self.is_ready = True
    
    def get_merkle_root(self) -> Optional[bytes]:
        """
        Obtiene la raíz del árbol Merkle
        
        Returns:
            Hash raíz del árbol o None si no está construido
        """
        if not self.is_ready:
            self.make_tree()
        
        if len(self.levels) > 0 and len(self.levels[-1]) > 0:
            return self.levels[-1][0]
        
        return None
    
    def get_merkle_root_hex(self) -> Optional[str]:
        """
        Obtiene la raíz del árbol en formato hexadecimal
        
        Returns:
            Hash raíz en hex o None
        """
        root = self.get_merkle_root()
        if root:
            return root.hex()
        return None
    
    def get_proof(self, index: int) -> List[dict]:
        """
        Obtiene la prueba de Merkle para una hoja específica
        
        Args:
            index: Índice de la hoja
            
        Returns:
            Lista de diccionarios con la prueba
        """
        if not self.is_ready:
            self.make_tree()
        
        if index < 0 or index >= len(self.leaves):
            raise IndexError(f"Índice {index} fuera de rango")
        
        proof = []
        
        for level_idx in range(len(self.levels) - 1):
            level = self.levels[level_idx]
            
            is_right = index % 2 == 1
            sibling_index = index - 1 if is_right else index + 1
            
            if sibling_index < len(level):
                sibling_hash = level[sibling_index]
                proof.append({
                    'right': is_right,
                    'hash': sibling_hash.hex()
                })
            
            index = index // 2
        
        return proof
    
    def validate_proof(self, proof: List[dict], target_hash: bytes, merkle_root: bytes) -> bool:
        """
        Valida una prueba de Merkle
        
        Args:
            proof: Prueba generada por get_proof
            target_hash: Hash de la hoja objetivo
            merkle_root: Raíz del árbol para verificar
            
        Returns:
            True si la prueba es válida
        """
        if isinstance(target_hash, str):
            target_hash = bytes.fromhex(target_hash)
        
        if isinstance(merkle_root, str):
            merkle_root = bytes.fromhex(merkle_root)
        
        current_hash = target_hash
        
        for step in proof:
            sibling_hash = bytes.fromhex(step['hash'])
            
            if step['right']:
                combined = sibling_hash + current_hash
            else:
                combined = current_hash + sibling_hash
            
            current_hash = self._hash(combined)
        
        return current_hash == merkle_root
    
    def reset_tree(self) -> None:
        """Resetea el árbol"""
        self.leaves = []
        self.levels = []
        self.is_ready = False


def create_merkle_tree(data_list: List[bytes], hash_type: str = 'sha256') -> MerkleTree:
    """
    Crea y construye un árbol Merkle a partir de una lista de datos
    
    Args:
        data_list: Lista de datos a incluir en el árbol
        hash_type: Algoritmo de hash a usar
        
    Returns:
        MerkleTree construido
    """
    tree = MerkleTree(hash_type=hash_type)
    
    for data in data_list:
        if isinstance(data, str):
            data = data.encode('utf-8')
        tree.add_leaf(data, do_hash=True)
    
    tree.make_tree()
    return tree


if __name__ == "__main__":
    print("🌳 Probando implementación de Merkle Tree...")
    
    data = [b"tx1", b"tx2", b"tx3", b"tx4"]
    
    tree = create_merkle_tree(data)
    root = tree.get_merkle_root_hex()
    
    print(f"✓ Raíz del árbol: {root}")
    
    proof = tree.get_proof(0)
    print(f"✓ Prueba para tx1: {proof}")
    
    is_valid = tree.validate_proof(proof, tree.leaves[0], tree.get_merkle_root())
    print(f"✓ Validación: {'OK' if is_valid else 'FALLÓ'}")
    
    print("\n✅ Merkle Tree implementado correctamente")
