#!/usr/bin/env python3
"""
Tests Unitarios para el Framework Criptográfico de AEGIS
"""

import asyncio
import unittest
import pytest
from unittest.mock import Mock, patch
import sys
import os

# Agregar el directorio padre al path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from test_framework import (
    AEGISTestFramework, TestSuite, TestType, TestStatus,
    unit_test, integration_test, performance_test, security_test
)

class CryptoFrameworkTests:
    """Tests para el framework criptográfico"""
    
    def __init__(self):
        self.crypto_module = None
        self.test_data = {
            "plaintext": b"Test data for encryption",
            "key_size": 256,
            "test_message": "Hello, AEGIS!"
        }
    
    def setup(self):
        """Setup para tests de crypto"""
        try:
            # Intentar importar el módulo crypto real
            from crypto_framework import AEGISCryptoFramework
            self.crypto_module = AEGISCryptoFramework()
        except ImportError:
            # Usar mock si no está disponible
            self.crypto_module = self._create_crypto_mock()
    
    def teardown(self):
        """Cleanup después de tests"""
        if hasattr(self.crypto_module, 'cleanup'):
            self.crypto_module.cleanup()
    
    def _create_crypto_mock(self):
        """Crea un mock del módulo crypto"""
        mock = Mock()
        mock.generate_key = Mock(return_value="test_key_256_bits")
        mock.encrypt_data = Mock(return_value=b"encrypted_test_data")
        mock.decrypt_data = Mock(return_value=self.test_data["plaintext"])
        mock.hash_data = Mock(return_value="test_hash_sha256")
        mock.sign_data = Mock(return_value="test_signature")
        mock.verify_signature = Mock(return_value=True)
        return mock
    
    @unit_test
    def test_key_generation(self):
        """Test de generación de claves"""
        key = self.crypto_module.generate_key(self.test_data["key_size"])
        
        assert key is not None, "La clave no debe ser None"
        assert len(key) > 0, "La clave debe tener longitud mayor a 0"
        
        # Test de unicidad
        key2 = self.crypto_module.generate_key(self.test_data["key_size"])
        assert key != key2, "Las claves generadas deben ser únicas"
    
    @unit_test
    def test_encryption_decryption(self):
        """Test de encriptación y desencriptación"""
        key = self.crypto_module.generate_key(256)
        plaintext = self.test_data["plaintext"]
        
        # Encriptar
        encrypted = self.crypto_module.encrypt_data(plaintext, key)
        assert encrypted is not None, "Los datos encriptados no deben ser None"
        assert encrypted != plaintext, "Los datos encriptados deben ser diferentes al texto plano"
        
        # Desencriptar
        decrypted = self.crypto_module.decrypt_data(encrypted, key)
        assert decrypted == plaintext, "Los datos desencriptados deben coincidir con el original"
    
    @unit_test
    def test_hashing(self):
        """Test de funciones hash"""
        data = self.test_data["plaintext"]
        
        # Test SHA-256
        hash1 = self.crypto_module.hash_data(data, algorithm="sha256")
        hash2 = self.crypto_module.hash_data(data, algorithm="sha256")
        
        assert hash1 == hash2, "El hash debe ser determinístico"
        assert len(hash1) > 0, "El hash no debe estar vacío"
        
        # Test con datos diferentes
        different_data = b"Different test data"
        hash3 = self.crypto_module.hash_data(different_data, algorithm="sha256")
        assert hash1 != hash3, "Datos diferentes deben producir hashes diferentes"
    
    @unit_test
    def test_digital_signatures(self):
        """Test de firmas digitales"""
        # Generar par de claves
        private_key = self.crypto_module.generate_private_key()
        public_key = self.crypto_module.get_public_key(private_key)
        
        data = self.test_data["plaintext"]
        
        # Firmar datos
        signature = self.crypto_module.sign_data(data, private_key)
        assert signature is not None, "La firma no debe ser None"
        
        # Verificar firma
        is_valid = self.crypto_module.verify_signature(data, signature, public_key)
        assert is_valid, "La firma debe ser válida"
        
        # Test con datos modificados
        modified_data = b"Modified test data"
        is_valid_modified = self.crypto_module.verify_signature(modified_data, signature, public_key)
        assert not is_valid_modified, "La firma no debe ser válida para datos modificados"
    
    @security_test
    def test_key_security(self):
        """Test de seguridad de claves"""
        # Test de entropía de claves
        keys = [self.crypto_module.generate_key(256) for _ in range(10)]
        
        # Verificar que todas las claves son diferentes
        unique_keys = set(keys)
        assert len(unique_keys) == len(keys), "Todas las claves deben ser únicas"
        
        # Test de longitud mínima
        for key in keys:
            assert len(key) >= 32, "Las claves deben tener al menos 256 bits (32 bytes)"
    
    @security_test
    def test_encryption_security(self):
        """Test de seguridad de encriptación"""
        key = self.crypto_module.generate_key(256)
        plaintext = self.test_data["plaintext"]
        
        # Test de no determinismo (mismo texto, diferentes resultados)
        encrypted1 = self.crypto_module.encrypt_data(plaintext, key)
        encrypted2 = self.crypto_module.encrypt_data(plaintext, key)
        
        # Si usa IV/nonce, los resultados deben ser diferentes
        if hasattr(self.crypto_module, 'uses_iv') and self.crypto_module.uses_iv:
            assert encrypted1 != encrypted2, "La encriptación debe usar IV/nonce para no ser determinística"
        
        # Ambos deben desencriptar al mismo texto
        decrypted1 = self.crypto_module.decrypt_data(encrypted1, key)
        decrypted2 = self.crypto_module.decrypt_data(encrypted2, key)
        
        assert decrypted1 == plaintext, "Primera desencriptación debe ser correcta"
        assert decrypted2 == plaintext, "Segunda desencriptación debe ser correcta"
    
    @performance_test
    async def test_encryption_performance(self):
        """Test de rendimiento de encriptación"""
        key = self.crypto_module.generate_key(256)
        
        # Test con diferentes tamaños de datos
        test_sizes = [1024, 10240, 102400]  # 1KB, 10KB, 100KB
        
        for size in test_sizes:
            data = b"x" * size
            
            start_time = asyncio.get_event_loop().time()
            encrypted = self.crypto_module.encrypt_data(data, key)
            encryption_time = asyncio.get_event_loop().time() - start_time
            
            start_time = asyncio.get_event_loop().time()
            decrypted = self.crypto_module.decrypt_data(encrypted, key)
            decryption_time = asyncio.get_event_loop().time() - start_time
            
            # Verificar que la operación es razonablemente rápida
            assert encryption_time < 1.0, f"Encriptación de {size} bytes debe tomar menos de 1 segundo"
            assert decryption_time < 1.0, f"Desencriptación de {size} bytes debe tomar menos de 1 segundo"
            assert decrypted == data, "Los datos desencriptados deben coincidir"
    
    @performance_test
    def test_key_generation_performance(self):
        """Test de rendimiento de generación de claves"""
        import time
        
        start_time = time.time()
        keys = [self.crypto_module.generate_key(256) for _ in range(100)]
        total_time = time.time() - start_time
        
        # Debe generar 100 claves en menos de 5 segundos
        assert total_time < 5.0, "Generación de 100 claves debe tomar menos de 5 segundos"
        
        # Verificar que todas son únicas
        assert len(set(keys)) == 100, "Todas las claves deben ser únicas"
    
    @integration_test
    async def test_crypto_with_storage(self):
        """Test de integración con sistema de almacenamiento"""
        # Este test requiere el módulo de almacenamiento
        try:
            from storage_system import AEGISStorage
            storage = AEGISStorage()
        except ImportError:
            # Skip si no está disponible
            pytest.skip("Módulo de almacenamiento no disponible")
        
        key = self.crypto_module.generate_key(256)
        data = self.test_data["plaintext"]
        
        # Encriptar y almacenar
        encrypted = self.crypto_module.encrypt_data(data, key)
        storage_id = await storage.store_data(encrypted, metadata={"encrypted": True})
        
        # Recuperar y desencriptar
        retrieved = await storage.retrieve_data(storage_id)
        decrypted = self.crypto_module.decrypt_data(retrieved, key)
        
        assert decrypted == data, "Los datos deben ser correctos después del ciclo completo"
    
    @integration_test
    async def test_crypto_with_network(self):
        """Test de integración con módulo de red"""
        try:
            from p2p_network import AEGISP2PNetwork
            network = AEGISP2PNetwork()
        except ImportError:
            pytest.skip("Módulo de red no disponible")
        
        # Test de intercambio seguro de claves
        key_pair = self.crypto_module.generate_key_pair()
        public_key = key_pair["public"]
        
        # Simular envío de clave pública
        message = {
            "type": "key_exchange",
            "public_key": public_key,
            "timestamp": asyncio.get_event_loop().time()
        }
        
        # Firmar mensaje
        signature = self.crypto_module.sign_data(
            str(message).encode(), 
            key_pair["private"]
        )
        message["signature"] = signature
        
        # Verificar que el mensaje se puede validar
        is_valid = self.crypto_module.verify_signature(
            str({k: v for k, v in message.items() if k != "signature"}).encode(),
            signature,
            public_key
        )
        
        assert is_valid, "La firma del mensaje debe ser válida"

def create_crypto_test_suite() -> TestSuite:
    """Crea la suite de tests para crypto"""
    crypto_tests = CryptoFrameworkTests()
    
    return TestSuite(
        name="CryptoFramework",
        description="Tests para el framework criptográfico de AEGIS",
        tests=[
            crypto_tests.test_key_generation,
            crypto_tests.test_encryption_decryption,
            crypto_tests.test_hashing,
            crypto_tests.test_digital_signatures,
            crypto_tests.test_key_security,
            crypto_tests.test_encryption_security,
            crypto_tests.test_encryption_performance,
            crypto_tests.test_key_generation_performance,
            crypto_tests.test_crypto_with_storage,
            crypto_tests.test_crypto_with_network
        ],
        setup_func=crypto_tests.setup,
        teardown_func=crypto_tests.teardown,
        test_type=TestType.UNIT
    )

if __name__ == "__main__":
    # Ejecutar tests de crypto individualmente
    async def main():
        from test_framework import get_test_framework
        
        framework = get_test_framework()
        suite = create_crypto_test_suite()
        framework.register_test_suite(suite)
        
        results = await framework.run_all_tests()
        print(f"Crypto tests - Éxito: {results['summary']['success_rate']:.1f}%")
    
    asyncio.run(main())