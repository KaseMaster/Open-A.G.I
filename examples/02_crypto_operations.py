#!/usr/bin/env python3
"""
AEGIS Framework - Cryptography Example
Ejemplo de operaciones criptográficas básicas
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

def main():
    print("🔐 AEGIS Cryptography Example")
    print("="*60)
    print()
    
    from aegis.security.crypto_framework import CryptoEngine
    
    # Inicializar engine
    crypto = CryptoEngine()
    print("✓ CryptoEngine inicializado")
    print()
    
    # 1. Hashing
    print("1️⃣  Hashing")
    data = b"AEGIS Framework - Distributed AI"
    
    hash_sha256 = crypto.hash_data(data, algorithm='sha256')
    print(f"   SHA-256: {hash_sha256.hex()}")
    
    hash_sha3 = crypto.hash_data(data, algorithm='sha3_256')
    print(f"   SHA3-256: {hash_sha3.hex()[:64]}...")
    print()
    
    # 2. Firma Digital
    print("2️⃣  Firma Digital (Ed25519)")
    
    # Generar par de claves
    private_key = crypto.generate_keypair()
    print("   ✓ Par de claves generado")
    
    # Firmar datos
    signature = crypto.sign_data(data, private_key)
    print(f"   ✓ Datos firmados ({len(signature)} bytes)")
    
    # Verificar firma
    public_key = private_key.public_key()
    is_valid = crypto.verify_signature(data, signature, public_key)
    print(f"   ✓ Firma válida: {is_valid}")
    print()
    
    # 3. Encriptación Simétrica (AES-256)
    print("3️⃣  Encriptación Simétrica (AES-256-GCM)")
    
    # Generar clave
    aes_key = crypto.generate_aes_key()
    print("   ✓ Clave AES-256 generada")
    
    # Encriptar
    plaintext = b"Mensaje secreto para AEGIS"
    encrypted = crypto.encrypt_symmetric(plaintext, aes_key)
    print(f"   ✓ Encriptado ({len(encrypted)} bytes)")
    
    # Desencriptar
    decrypted = crypto.decrypt_symmetric(encrypted, aes_key)
    print(f"   ✓ Desencriptado: {decrypted.decode()}")
    print()
    
    # 4. Encriptación Asimétrica (RSA)
    print("4️⃣  Encriptación Asimétrica (RSA-4096)")
    
    # Generar claves RSA
    rsa_private, rsa_public = crypto.generate_rsa_keypair(key_size=2048)
    print("   ✓ Par de claves RSA generado")
    
    # Encriptar con clave pública
    message = b"Secreto"
    encrypted_rsa = crypto.encrypt_asymmetric(message, rsa_public)
    print(f"   ✓ Encriptado con RSA ({len(encrypted_rsa)} bytes)")
    
    # Desencriptar con clave privada
    decrypted_rsa = crypto.decrypt_asymmetric(encrypted_rsa, rsa_private)
    print(f"   ✓ Desencriptado: {decrypted_rsa.decode()}")
    print()
    
    print("="*60)
    print("✅ Todas las operaciones criptográficas completadas")
    print()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
