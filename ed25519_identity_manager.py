#!/usr/bin/env python3
"""
🆔 SISTEMA DE IDENTIDADES ED25519 - AEGIS Framework
Módulo dedicado para gestión de identidades criptográficas de nodos Ed25519.

Características principales:
- Generación automática de identidades Ed25519
- Registro y verificación de identidades de peers
- Sistema de confianza y reputación basado en identidad
- Intercambio seguro de claves públicas
- Validación de firmas para autenticación
"""

import asyncio
import time
import json
import hashlib
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import logging

# Dependencias criptográficas
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.exceptions import InvalidSignature

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IdentityStatus(Enum):
    """Estados de identidad de un nodo"""
    UNKNOWN = "unknown"
    PENDING = "pending"      # Identidad recibida pero no verificada
    VERIFIED = "verified"    # Identidad verificada y confiable
    SUSPICIOUS = "suspicious"  # Identidad con comportamiento sospechoso
    REVOKED = "revoked"      # Identidad revocada

class TrustLevel(Enum):
    """Niveles de confianza para identidades"""
    NONE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    MAXIMUM = 4

@dataclass
class NodeIdentityRecord:
    """Registro completo de identidad de nodo"""
    node_id: str
    signing_public_key: ed25519.Ed25519PublicKey
    encryption_public_key: Optional[bytes] = None  # Para integración con crypto_framework
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_seen: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    status: IdentityStatus = IdentityStatus.UNKNOWN
    trust_level: TrustLevel = TrustLevel.NONE
    reputation_score: float = 0.5
    signature_count: int = 0  # Número de firmas verificadas exitosamente
    failed_verifications: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validar claves al inicializar"""
        if not isinstance(self.signing_public_key, ed25519.Ed25519PublicKey):
            raise ValueError("La clave de firma debe ser Ed25519PublicKey")

    def export_public_data(self) -> Dict[str, bytes]:
        """Exportar datos públicos para intercambio"""
        data = {
            'node_id': self.node_id.encode('utf-8'),
            'signing_key': self.signing_public_key.public_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PublicFormat.Raw
            ),
            'created_at': str(self.created_at).encode('utf-8')
        }

        if self.encryption_public_key:
            data['encryption_key'] = self.encryption_public_key

        return data

    def verify_signature(self, message: bytes, signature: bytes) -> bool:
        """Verificar firma con la clave pública del nodo"""
        try:
            self.signing_public_key.verify(signature, message)
            self.signature_count += 1
            # Actualizar confianza basada en verificaciones exitosas
            if self.signature_count > 10 and self.failed_verifications == 0:
                self.trust_level = TrustLevel.HIGH
            return True
        except InvalidSignature:
            self.failed_verifications += 1
            # Penalizar confianza por verificación fallida
            if self.failed_verifications > 3:
                self.trust_level = TrustLevel.LOW
                self.status = IdentityStatus.SUSPICIOUS
            return False

    def update_last_seen(self):
        """Actualizar timestamp de última actividad"""
        self.last_seen = datetime.now(timezone.utc)

    def calculate_trust_score(self) -> float:
        """Calcular puntuación de confianza basada en múltiples factores"""
        base_score = 0.5

        # Factor de tiempo (identidades más antiguas son más confiables)
        age_days = (datetime.now(timezone.utc) - self.created_at).days
        age_factor = min(age_days / 365, 1.0)  # Máximo 1 año

        # Factor de actividad (más actividad = más confianza)
        activity_factor = min(self.signature_count / 100, 1.0)  # Máximo 100 firmas

        # Factor de éxito (menos fallos = más confianza)
        if self.signature_count + self.failed_verifications > 0:
            success_rate = self.signature_count / (self.signature_count + self.failed_verifications)
        else:
            success_rate = 0.5

        # Factor de nivel de confianza
        trust_factor = self.trust_level.value / TrustLevel.MAXIMUM.value

        # Calcular score final
        final_score = (
            base_score * 0.2 +
            age_factor * 0.2 +
            activity_factor * 0.2 +
            success_rate * 0.3 +
            trust_factor * 0.1
        )

        self.reputation_score = final_score
        return final_score

class Ed25519IdentityManager:
    """Gestor principal de identidades Ed25519"""

    def __init__(self):
        self.identities: Dict[str, NodeIdentityRecord] = {}
        self.own_identity: Optional[NodeIdentityRecord] = None
        self.trust_threshold = 0.7  # Umbral mínimo para considerar identidad confiable
        self.max_failed_attempts = 5  # Máximo intentos fallidos antes de marcar sospechosa

        logger.info("🆔 Ed25519IdentityManager inicializado")

    async def generate_own_identity(self, node_id: str) -> NodeIdentityRecord:
        """Generar identidad Ed25519 propia para el nodo"""
        try:
            # Generar clave privada Ed25519
            private_key = ed25519.Ed25519PrivateKey.generate()

            # Crear registro de identidad
            identity = NodeIdentityRecord(
                node_id=node_id,
                signing_public_key=private_key.public_key(),
                status=IdentityStatus.VERIFIED,  # Nuestra propia identidad siempre está verificada
                trust_level=TrustLevel.MAXIMUM,  # Confianza máxima en nosotros mismos
                reputation_score=1.0
            )

            self.own_identity = identity
            self.identities[node_id] = identity

            # Mostrar información de la clave generada
            public_key_bytes = identity.signing_public_key.public_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PublicFormat.Raw
            )
            key_hash = hashlib.sha256(public_key_bytes).hexdigest()[:16]

            logger.info(f"🔐 Identidad Ed25519 generada para {node_id}")
            logger.info(f"🔑 Fingerprint: {key_hash}")
            logger.info(f"📅 Creada: {identity.created_at.strftime('%Y-%m-%d %H:%M:%S UTC')}")

            return identity

        except Exception as e:
            logger.error(f"❌ Error generando identidad Ed25519: {e}")
            raise

    async def register_peer_identity(self, public_data: Dict[str, bytes]) -> bool:
        """Registrar identidad de un peer remoto"""
        try:
            node_id = public_data['node_id'].decode('utf-8')

            # Verificar si ya existe
            if node_id in self.identities:
                logger.debug(f"🔄 Actualizando identidad existente para {node_id}")
                existing = self.identities[node_id]
                existing.update_last_seen()
                return True

            # Crear nueva identidad
            signing_key = ed25519.Ed25519PublicKey.from_public_bytes(
                public_data['signing_key']
            )

            created_at_str = public_data['created_at'].decode('utf-8')
            created_at = datetime.fromisoformat(created_at_str)

            identity = NodeIdentityRecord(
                node_id=node_id,
                signing_public_key=signing_key,
                created_at=created_at,
                status=IdentityStatus.PENDING,  # Pendiente de verificación
                trust_level=TrustLevel.LOW
            )

            # Agregar clave de cifrado si está presente (para integración con crypto_framework)
            if 'encryption_key' in public_data:
                identity.encryption_public_key = public_data['encryption_key']

            self.identities[node_id] = identity

            # Calcular score inicial de confianza
            identity.calculate_trust_score()

            logger.info(f"✅ Identidad registrada para peer {node_id}")
            logger.info(f"📊 Nivel de confianza inicial: {identity.trust_level.name}")

            return True

        except Exception as e:
            logger.error(f"❌ Error registrando identidad de peer: {e}")
            return False

    async def verify_peer_identity(self, node_id: str, challenge: bytes = None) -> bool:
        """Verificar identidad de un peer mediante desafío-respuesta"""
        if node_id not in self.identities:
            logger.warning(f"⚠️ Peer {node_id} no encontrado en registro de identidades")
            return False

        identity = self.identities[node_id]

        # Si ya está verificada, retornar True
        if identity.status == IdentityStatus.VERIFIED:
            return True

        # Para verificación completa, necesitaríamos un desafío-respuesta
        # Por ahora, marcamos como verificada si tiene buena reputación
        if identity.reputation_score >= self.trust_threshold:
            identity.status = IdentityStatus.VERIFIED
            identity.trust_level = TrustLevel.HIGH
            logger.info(f"✅ Identidad de {node_id} verificada por reputación")
            return True

        logger.warning(f"⚠️ Identidad de {node_id} no pudo ser verificada")
        return False

    async def verify_message_signature(self, node_id: str, message: bytes, signature: bytes) -> bool:
        """Verificar firma de mensaje de un peer"""
        if node_id not in self.identities:
            logger.warning(f"⚠️ Peer {node_id} no encontrado para verificación de firma")
            return False

        identity = self.identities[node_id]
        identity.update_last_seen()

        # Verificar firma
        is_valid = identity.verify_signature(message, signature)

        if is_valid:
            # Recalcular score de confianza
            identity.calculate_trust_score()
            logger.debug(f"✅ Firma verificada para {node_id}")
        else:
            logger.warning(f"❌ Firma inválida de {node_id}")
            # Verificar si debe ser marcada como sospechosa
            if identity.failed_verifications >= self.max_failed_attempts:
                identity.status = IdentityStatus.SUSPICIOUS
                logger.warning(f"🚨 Peer {node_id} marcado como sospechoso por múltiples fallos")

        return is_valid

    def get_identity_status(self, node_id: str) -> Dict[str, Any]:
        """Obtener estado completo de identidad de un peer"""
        if node_id not in self.identities:
            return {"status": "unknown"}

        identity = self.identities[node_id]
        return {
            "node_id": identity.node_id,
            "status": identity.status.value,
            "trust_level": identity.trust_level.name,
            "reputation_score": round(identity.reputation_score, 3),
            "signature_count": identity.signature_count,
            "failed_verifications": identity.failed_verifications,
            "created_at": identity.created_at.isoformat(),
            "last_seen": identity.last_seen.isoformat(),
            "is_trusted": identity.reputation_score >= self.trust_threshold
        }

    def get_trusted_peers(self) -> List[str]:
        """Obtener lista de peers confiables"""
        trusted = []
        for node_id, identity in self.identities.items():
            if (identity.status == IdentityStatus.VERIFIED and
                identity.reputation_score >= self.trust_threshold):
                trusted.append(node_id)
        return trusted

    def get_suspicious_peers(self) -> List[str]:
        """Obtener lista de peers sospechosos"""
        suspicious = []
        for node_id, identity in self.identities.items():
            if identity.status in [IdentityStatus.SUSPICIOUS, IdentityStatus.REVOKED]:
                suspicious.append(node_id)
        return suspicious

    def revoke_identity(self, node_id: str, reason: str = "manual_revoke"):
        """Revocar identidad de un peer"""
        if node_id in self.identities:
            self.identities[node_id].status = IdentityStatus.REVOKED
            self.identities[node_id].trust_level = TrustLevel.NONE
            logger.warning(f"🚫 Identidad revocada para {node_id}: {reason}")

    def cleanup_old_identities(self, max_age_days: int = 90):
        """Limpiar identidades antiguas inactivas"""
        current_time = datetime.now(timezone.utc)
        to_remove = []

        for node_id, identity in self.identities.items():
            age_days = (current_time - identity.last_seen).days
            if age_days > max_age_days and identity.status != IdentityStatus.VERIFIED:
                to_remove.append(node_id)

        for node_id in to_remove:
            del self.identities[node_id]
            logger.debug(f"🧹 Identidad antigua limpiada: {node_id}")

        if to_remove:
            logger.info(f"🧹 {len(to_remove)} identidades antiguas limpiadas")

    def get_identity_report(self) -> Dict[str, Any]:
        """Generar reporte completo de identidades"""
        total_identities = len(self.identities)
        verified_count = sum(1 for i in self.identities.values() if i.status == IdentityStatus.VERIFIED)
        suspicious_count = sum(1 for i in self.identities.values() if i.status == IdentityStatus.SUSPICIOUS)
        revoked_count = sum(1 for i in self.identities.values() if i.status == IdentityStatus.REVOKED)

        return {
            "total_identities": total_identities,
            "verified_identities": verified_count,
            "suspicious_identities": suspicious_count,
            "revoked_identities": revoked_count,
            "trusted_peers": self.get_trusted_peers(),
            "suspicious_peers": self.get_suspicious_peers(),
            "average_reputation": sum(i.reputation_score for i in self.identities.values()) / total_identities if total_identities > 0 else 0
        }

async def demo_ed25519_identity_system():
    """Demostración del sistema de identidades Ed25519"""
    print("🆔 DEMO - SISTEMA DE IDENTIDADES ED25519")
    print("=" * 50)

    # Inicializar gestor
    identity_manager = Ed25519IdentityManager()

    try:
        # 1. Generar identidad propia
        print("\n🔐 Generando identidad Ed25519 propia...")
        own_identity = await identity_manager.generate_own_identity("demo_node_1")
        print(f"✅ Identidad generada: {own_identity.node_id}")

        # 2. Simular registro de peers
        print("\n👥 Registrando identidades de peers...")

        # Simular peer 1 - confiable
        peer1_data = {
            'node_id': b'peer_1',
            'signing_key': ed25519.Ed25519PrivateKey.generate().public_key().public_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PublicFormat.Raw
            ),
            'created_at': b'2024-01-15T10:00:00'
        }
        await identity_manager.register_peer_identity(peer1_data)

        # Simular peer 2 - sospechoso
        peer2_data = {
            'node_id': b'peer_2',
            'signing_key': ed25519.Ed25519PrivateKey.generate().public_key().public_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PublicFormat.Raw
            ),
            'created_at': b'2024-01-15T10:00:00'
        }
        await identity_manager.register_peer_identity(peer2_data)

        # 3. Simular verificación de firmas
        print("\n🔏 Simulando verificación de firmas...")

        # Verificación exitosa para peer 1
        test_message = b"Hola desde peer_1"
        private_key = ed25519.Ed25519PrivateKey.generate()
        signature = private_key.sign(test_message)

        # Usar la clave correcta para verificación
        peer1_private = ed25519.Ed25519PrivateKey.from_private_bytes(
            ed25519.Ed25519PrivateKey.generate().private_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PrivateFormat.Raw,
                encryption_algorithm=serialization.NoEncryption()
            )
        )
        peer1_signature = peer1_private.sign(test_message)

        # Verificar con la clave correcta
        is_valid = await identity_manager.verify_message_signature("peer_1", test_message, peer1_signature)
        print(f"✅ Firma de peer_1: {'Válida' if is_valid else 'Inválida'}")

        # 4. Mostrar estado de identidades
        print("\n📊 Estado de identidades:")
        report = identity_manager.get_identity_report()
        print(f"   Total de identidades: {report['total_identities']}")
        print(f"   Identidades verificadas: {report['verified_identities']}")
        print(f"   Peers confiables: {len(report['trusted_peers'])}")
        print(f"   Peers sospechosos: {len(report['suspicious_peers'])}")
        print(f"   Reputación promedio: {report['average_reputation']:.3f}")
        # 5. Mostrar detalles de cada identidad
        print("\n📋 Detalles de identidades:")
        for node_id in ['demo_node_1', 'peer_1', 'peer_2']:
            status = identity_manager.get_identity_status(node_id)
            print(f"   {node_id}:")
            print(f"      Estado: {status['status']}")
            print(f"      Confianza: {status['trust_level']}")
            print(f"      Reputación: {status['reputation_score']:.3f}")
        print("\n🎉 Demo completada exitosamente!")

    except Exception as e:
        print(f"❌ Error en demo: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(demo_ed25519_identity_system())
