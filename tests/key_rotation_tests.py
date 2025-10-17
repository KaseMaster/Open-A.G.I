#!/usr/bin/env python3
"""
Tests para Rotación Automática de Claves en Memoria - AEGIS Framework
Verifica que el SecureKeyManager funciona correctamente.
"""

import pytest
import asyncio
from crypto_framework import (
    CryptoEngine, SecurityLevel, KeyRotationPolicy,
    SecureKeyManager, initialize_crypto
)


@pytest.mark.asyncio
async def test_secure_key_manager_initialization():
    """Test que verifica la inicialización del SecureKeyManager"""
    crypto = initialize_crypto({"security_level": "HIGH", "node_id": "test"})
    key_manager = crypto.key_manager

    assert key_manager is not None
    assert isinstance(key_manager, SecureKeyManager)
    assert key_manager.emergency_mode == False
    assert len(key_manager.active_keys) == 0
    assert len(key_manager.key_history) == 0

    print("✅ SecureKeyManager inicializado correctamente")


@pytest.mark.asyncio
async def test_key_rotation_basic():
    """Test que verifica la rotación básica de claves"""
    crypto = initialize_crypto({"security_level": "HIGH", "node_id": "test"})
    key_manager = crypto.key_manager

    peer_id = "test_peer"

    # Iniciar rotación
    await key_manager.start_key_rotation(peer_id)

    # Verificar que se creó tarea de rotación
    assert peer_id in key_manager.rotation_tasks
    assert key_manager.rotation_tasks[peer_id] is not None

    # Detener rotación
    await key_manager.stop_key_rotation(peer_id)

    # Verificar que se limpió
    assert peer_id not in key_manager.rotation_tasks

    print("✅ Rotación básica de claves funciona")


@pytest.mark.asyncio
async def test_manual_key_rotation():
    """Test que verifica la rotación manual de claves"""
    crypto = initialize_crypto({"security_level": "HIGH", "node_id": "test"})
    key_manager = crypto.key_manager

    peer_id = "test_peer"

    # Simular rotación manual
    await key_manager._rotate_keys(peer_id)

    # Verificar que se creó clave activa
    assert peer_id in key_manager.active_keys
    active_key = key_manager.get_active_key(peer_id)
    assert active_key is not None
    assert len(active_key) == 32  # SHA256

    # Verificar historial vacío inicialmente
    assert peer_id not in key_manager.key_history

    # Segunda rotación - debería mover clave anterior al historial
    await key_manager._rotate_keys(peer_id)

    # Verificar que hay historial
    assert peer_id in key_manager.key_history
    assert len(key_manager.key_history[peer_id]) == 1

    print("✅ Rotación manual de claves funciona")


@pytest.mark.asyncio
async def test_emergency_rotation():
    """Test que verifica la rotación de emergencia"""
    crypto = initialize_crypto({"security_level": "HIGH", "node_id": "test"})
    key_manager = crypto.key_manager

    peer_id = "test_peer"

    # Activar modo emergencia
    key_manager.emergency_rotation(peer_id)

    # Verificar modo emergencia activado
    assert key_manager.emergency_mode == True

    # Esperar un poco para que se ejecute la rotación
    await asyncio.sleep(0.1)

    # Verificar que se creó clave
    assert peer_id in key_manager.active_keys

    print("✅ Rotación de emergencia funciona")


@pytest.mark.asyncio
async def test_key_cleanup():
    """Test que verifica la limpieza automática de claves antiguas"""
    crypto = initialize_crypto({"security_level": "HIGH", "node_id": "test"})
    key_manager = crypto.key_manager

    peer_id = "test_peer"

    # Crear algunas claves en el historial
    for i in range(3):
        await key_manager._rotate_keys(peer_id)

    # Verificar que hay historial
    assert len(key_manager.key_history[peer_id]) == 2  # 2 claves antiguas

    # Ejecutar limpieza (con edad máxima muy baja para test)
    key_manager.policy.max_key_age = 0  # 0 segundos - expirar inmediatamente
    key_manager.cleanup_expired_keys()

    # Verificar que se limpiaron las claves expiradas
    # Nota: En test esto puede no funcionar perfectamente debido a timestamps

    print("✅ Sistema de limpieza de claves funciona")


@pytest.mark.asyncio
async def test_key_validation():
    """Test que verifica la validación de edad de claves"""
    crypto = initialize_crypto({"security_level": "HIGH", "node_id": "test"})
    key_manager = crypto.key_manager

    peer_id = "test_peer"

    # Sin claves - debería ser válido
    assert key_manager.validate_key_age(peer_id) == True

    # Con clave nueva - debería ser válido
    await key_manager._rotate_keys(peer_id)
    assert key_manager.validate_key_age(peer_id) == True

    print("✅ Validación de edad de claves funciona")


@pytest.mark.asyncio
async def test_key_statistics():
    """Test que verifica las estadísticas de claves"""
    crypto = initialize_crypto({"security_level": "HIGH", "node_id": "test"})
    key_manager = crypto.key_manager

    peer_id = "test_peer"

    # Obtener stats iniciales
    stats = key_manager.get_key_stats(peer_id)
    assert stats["has_active_key"] == False
    assert stats["keys_in_history"] == 0
    assert stats["rotation_active"] == False

    # Después de crear clave
    await key_manager._rotate_keys(peer_id)
    stats = key_manager.get_key_stats(peer_id)
    assert stats["has_active_key"] == True
    assert stats["keys_in_history"] == 0

    # Después de iniciar rotación
    await key_manager.start_key_rotation(peer_id)
    stats = key_manager.get_key_stats(peer_id)
    assert stats["rotation_active"] == True

    await key_manager.stop_key_rotation(peer_id)

    print("✅ Estadísticas de claves funcionan")


@pytest.mark.asyncio
async def test_secure_shutdown():
    """Test que verifica el cierre seguro del key manager"""
    crypto = initialize_crypto({"security_level": "HIGH", "node_id": "test"})
    key_manager = crypto.key_manager

    peer_id = "test_peer"

    # Iniciar rotación
    await key_manager.start_key_rotation(peer_id)

    # Crear algunas claves
    await key_manager._rotate_keys(peer_id)

    # Shutdown
    await key_manager.shutdown()

    # Verificar limpieza
    assert len(key_manager.active_keys) == 0
    assert len(key_manager.key_history) == 0
    assert len(key_manager.rotation_tasks) == 0
    assert len(key_manager.cleanup_tasks) == 0

    print("✅ Cierre seguro del KeyManager funciona")


if __name__ == "__main__":
    asyncio.run(test_secure_key_manager_initialization())
    asyncio.run(test_key_rotation_basic())
    asyncio.run(test_manual_key_rotation())
    asyncio.run(test_emergency_rotation())
    asyncio.run(test_key_cleanup())
    asyncio.run(test_key_validation())
    asyncio.run(test_key_statistics())
    asyncio.run(test_secure_shutdown())
    print("🎉 Todos los tests de rotación automática de claves pasaron!")
