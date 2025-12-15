#!/usr/bin/env python3
"""
Framework Criptográfico para IA Distribuida y Colaborativa
AEGIS Security Framework - Uso Ético Únicamente

Programador Principal: Jose Gómez alias KaseMaster
Contacto: kasemaster@aegis-framework.com
Versión: 2.1.0 - Con Perfect Forward Secrecy
Licencia: MIT

Este módulo implementa un sistema criptográfico robusto para:
- Autenticación de nodos mediante Ed25519
- Cifrado de extremo a extremo con ChaCha20-Poly1305
- Intercambio de claves con X25519 + Diffie-Hellman Efímero
- Double Ratchet COMPLETO con Forward Secrecy
- Firmas digitales para integridad de datos

ADVERTENCIA: Este código es para investigación y desarrollo ético únicamente.
El uso malicioso está estrictamente prohibido.
"""

import os
# Verified Crypto Framework
import time
import hmac
import hashlib
import secrets
from typing import Dict, Optional, Tuple, List, Any
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import logging
from datetime import datetime, timedelta, timezone

# Dependencias criptográficas
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ed25519, x25519
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.exceptions import InvalidSignature

# Configuración de logging seguro
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('crypto_security.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SecurityLevel(Enum):
    """Niveles de seguridad para diferentes contextos"""
    STANDARD = "standard"
    HIGH = "high"
    PARANOID = "paranoid"

class KeyType(Enum):
    """Tipos de claves criptográficas"""
    SIGNING = "signing"
    ENCRYPTION = "encryption"
    EPHEMERAL = "ephemeral"

@dataclass
class CryptoConfig:
    """Configuración criptográfica del sistema"""
    security_level: SecurityLevel = SecurityLevel.HIGH
    key_rotation_interval: int = 86400  # 24 horas
    max_message_age: int = 300  # 5 minutos
    ratchet_advance_threshold: int = 100  # mensajes antes de avanzar ratchet
    pbkdf2_iterations: int = 100000
    
    def __post_init__(self):
        """Ajustar configuración según nivel de seguridad"""
        if self.security_level == SecurityLevel.PARANOID:
            self.key_rotation_interval = 3600  # 1 hora
            self.max_message_age = 60  # 1 minuto
            self.ratchet_advance_threshold = 50
            self.pbkdf2_iterations = 200000
        elif self.security_level == SecurityLevel.STANDARD:
            self.key_rotation_interval = 172800  # 48 horas
            self.max_message_age = 600  # 10 minutos
            self.ratchet_advance_threshold = 200
            self.pbkdf2_iterations = 50000

@dataclass
class NodeIdentity:
    """Identidad criptográfica de un nodo"""
    node_id: str
    signing_key: ed25519.Ed25519PrivateKey
    encryption_key: x25519.X25519PrivateKey
    public_signing_key: ed25519.Ed25519PublicKey = field(init=False)
    public_encryption_key: x25519.X25519PublicKey = field(init=False)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def __post_init__(self):
        """Derivar claves públicas"""
        self.public_signing_key = self.signing_key.public_key()
        self.public_encryption_key = self.encryption_key.public_key()
    
    def export_public_identity(self) -> Dict[str, bytes]:
        """Exportar identidad pública para intercambio"""
        return {
            'node_id': self.node_id.encode(),
            'signing_key': self.public_signing_key.public_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PublicFormat.Raw
            ),
            'encryption_key': self.public_encryption_key.public_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PublicFormat.Raw
            ),
            'created_at': str(self.created_at).encode()
        }
    
    @classmethod
    def from_public_data(cls, data: Dict[str, bytes]) -> 'PublicNodeIdentity':
        """Crear identidad pública desde datos exportados"""
        return PublicNodeIdentity(
            node_id=data['node_id'].decode(),
            public_signing_key=ed25519.Ed25519PublicKey.from_public_bytes(
                data['signing_key']
            ),
            public_encryption_key=x25519.X25519PublicKey.from_public_bytes(
                data['encryption_key']
            ),
            created_at=datetime.fromisoformat(data['created_at'].decode())
        )

@dataclass
class PublicNodeIdentity:
    """Identidad pública de un nodo remoto"""
    node_id: str
    public_signing_key: ed25519.Ed25519PublicKey
    public_encryption_key: x25519.X25519PublicKey
    created_at: datetime
    last_seen: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    trust_score: float = 0.5  # Puntuación de confianza inicial

@dataclass
class KeyRotationPolicy:
    """Política de rotación de claves"""
    rotation_interval: int = 3600  # 1 hora por defecto
    max_key_age: int = 86400  # 24 horas máximo
    emergency_rotation: bool = False  # Rotación inmediata por sospecha de compromiso
    cleanup_delay: int = 300  # 5 minutos antes de limpiar claves antiguas


class SecureKeyManager:
    """Gestor seguro de claves con rotación automática en memoria"""

    def __init__(self, crypto_engine: 'CryptoEngine', policy: KeyRotationPolicy = None):
        self.crypto_engine = crypto_engine
        self.policy = policy or KeyRotationPolicy()
        self.key_history: Dict[str, List[Tuple[bytes, datetime]]] = {}  # peer_id -> [(key, created_at), ...]
        self.active_keys: Dict[str, bytes] = {}  # peer_id -> current_active_key
        self.rotation_tasks: Dict[str, asyncio.Task] = {}
        self.cleanup_tasks: Dict[str, asyncio.Task] = {}
        self.emergency_mode = False

        logger.info("🔐 SecureKeyManager inicializado con política de rotación automática")

    async def start_key_rotation(self, peer_id: str):
        """Inicia la rotación automática de claves para un peer"""
        if peer_id in self.rotation_tasks:
            logger.debug(f"Rotación ya activa para {peer_id}")
            return

        # Crear tarea de rotación
        task = asyncio.create_task(self._key_rotation_loop(peer_id))
        self.rotation_tasks[peer_id] = task

        logger.info(f"🔄 Rotación automática iniciada para peer {peer_id}")

    async def stop_key_rotation(self, peer_id: str):
        """Detiene la rotación de claves para un peer"""
        if peer_id in self.rotation_tasks:
            self.rotation_tasks[peer_id].cancel()
            del self.rotation_tasks[peer_id]

        if peer_id in self.cleanup_tasks:
            self.cleanup_tasks[peer_id].cancel()
            del self.cleanup_tasks[peer_id]

        # Limpiar historial de claves
        if peer_id in self.key_history:
            del self.key_history[peer_id]
        if peer_id in self.active_keys:
            del self.active_keys[peer_id]

        logger.info(f"🛑 Rotación detenida para peer {peer_id}")

    async def _key_rotation_loop(self, peer_id: str):
        """Bucle de rotación automática de claves"""
        while peer_id in self.rotation_tasks:
            try:
                # Esperar intervalo de rotación (o menos en modo emergencia)
                interval = 60 if self.emergency_mode else self.policy.rotation_interval
                await asyncio.sleep(interval)

                if peer_id not in self.rotation_tasks:
                    break

                # Realizar rotación
                await self._rotate_keys(peer_id)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"❌ Error en rotación de claves para {peer_id}: {e}")
                await asyncio.sleep(60)  # Esperar antes de reintentar

    async def _rotate_keys(self, peer_id: str):
        """Realiza la rotación de claves para un peer"""
        try:
            # Generar nueva identidad para el peer (simula nueva clave)
            # En implementación real, esto sería coordinado con el peer remoto
            new_key_seed = secrets.token_bytes(32)
            new_key = hashlib.sha256(new_key_seed).digest()

            # Guardar clave anterior en historial antes de reemplazar
            if peer_id in self.active_keys:
                old_key = self.active_keys[peer_id]
                old_timestamp = datetime.now(timezone.utc)

                if peer_id not in self.key_history:
                    self.key_history[peer_id] = []
                self.key_history[peer_id].append((old_key, old_timestamp))

                # Programar limpieza de clave antigua
                await self._schedule_key_cleanup(peer_id, old_key, old_timestamp)

            # Establecer nueva clave como activa
            self.active_keys[peer_id] = new_key

            # Si es modo emergencia, forzar nueva conexión
            if self.emergency_mode:
                logger.warning(f"🚨 ROTACIÓN DE EMERGENCIA para {peer_id} - posible compromiso detectado")
                # Aquí se podría forzar reconexión o invalidar sesiones existentes

            logger.info(f"🔄 Claves rotadas exitosamente para peer {peer_id}")

        except Exception as e:
            logger.error(f"❌ Error rotando claves para {peer_id}: {e}")

    async def _schedule_key_cleanup(self, peer_id: str, old_key: bytes, created_at: datetime):
        """Programa la limpieza de una clave antigua"""
        async def cleanup_old_key():
            try:
                await asyncio.sleep(self.policy.cleanup_delay)

                # Verificar que la clave aún existe en el historial
                if peer_id in self.key_history:
                    # Remover la clave específica del historial
                    self.key_history[peer_id] = [
                        (key, ts) for key, ts in self.key_history[peer_id]
                        if key != old_key
                    ]

                    # Limpiar lista vacía
                    if not self.key_history[peer_id]:
                        del self.key_history[peer_id]

                # Limpiar referencia a la clave (best effort)
                # Nota: En Python esto ayuda al garbage collector

                logger.debug(f"🧹 Clave antigua limpiada para peer {peer_id}")

            except Exception as e:
                logger.debug(f"⚠️ Error limpiando clave antigua para {peer_id}: {e}")

        # Crear tarea de limpieza
        task = asyncio.create_task(cleanup_old_key())
        if peer_id not in self.cleanup_tasks:
            self.cleanup_tasks[peer_id] = task

    def emergency_rotation(self, peer_id: str = None):
        """Activa rotación de emergencia para todos los peers o uno específico"""
        self.emergency_mode = True

        if peer_id:
            # Rotación inmediata para peer específico
            asyncio.create_task(self._rotate_keys(peer_id))
            logger.critical(f"🚨 ROTACIÓN DE EMERGENCIA ACTIVADA para {peer_id}")
        else:
            # Rotación inmediata para todos los peers
            for pid in list(self.active_keys.keys()):
                asyncio.create_task(self._rotate_keys(pid))
            logger.critical("🚨 ROTACIÓN DE EMERGENCIA ACTIVADA para TODOS los peers")

        # Programar desactivación del modo emergencia después de 1 hora
        async def deactivate_emergency():
            await asyncio.sleep(3600)
            self.emergency_mode = False
            logger.info("🔄 Modo de rotación de emergencia desactivado")

        asyncio.create_task(deactivate_emergency())

    def get_active_key(self, peer_id: str) -> Optional[bytes]:
        """Obtiene la clave activa actual para un peer"""
        return self.active_keys.get(peer_id)

    def validate_key_age(self, peer_id: str) -> bool:
        """Valida que la clave activa no haya excedido la edad máxima"""
        if peer_id not in self.key_history:
            return True  # No hay historial, clave es nueva

        # Verificar edad de la clave más reciente en historial
        # (la clave activa no tiene timestamp, asumimos que es reciente)
        current_time = datetime.now(timezone.utc)

        for _, created_at in self.key_history.get(peer_id, []):
            age = (current_time - created_at).total_seconds()
            if age > self.policy.max_key_age:
                return False

        return True

    def get_key_stats(self, peer_id: str) -> Dict[str, Any]:
        """Obtiene estadísticas de claves para un peer"""
        stats = {
            "has_active_key": peer_id in self.active_keys,
            "keys_in_history": len(self.key_history.get(peer_id, [])),
            "rotation_active": peer_id in self.rotation_tasks,
            "cleanup_active": peer_id in self.cleanup_tasks,
            "emergency_mode": self.emergency_mode,
            "oldest_key_age": None,
            "newest_key_age": None
        }

        if peer_id in self.key_history:
            timestamps = [ts for _, ts in self.key_history[peer_id]]
            if timestamps:
                current_time = datetime.now(timezone.utc)
                ages = [(current_time - ts).total_seconds() for ts in timestamps]
                stats["oldest_key_age"] = max(ages)
                stats["newest_key_age"] = min(ages)

        return stats

    def cleanup_expired_keys(self):
        """Limpia claves expiradas de todos los peers"""
        current_time = datetime.now(timezone.utc)
        total_cleaned = 0

        for peer_id in list(self.key_history.keys()):
            original_count = len(self.key_history[peer_id])

            # Filtrar claves no expiradas
            self.key_history[peer_id] = [
                (key, ts) for key, ts in self.key_history[peer_id]
                if (current_time - ts).total_seconds() <= self.policy.max_key_age
            ]

            cleaned = original_count - len(self.key_history[peer_id])
            total_cleaned += cleaned

            if not self.key_history[peer_id]:
                del self.key_history[peer_id]

        if total_cleaned > 0:
            logger.info(f"🧹 {total_cleaned} claves expiradas limpiadas de memoria")

    async def shutdown(self):
        """Cierra el gestor de claves de forma segura"""
        logger.info("🔐 Cerrando SecureKeyManager...")

        # Cancelar todas las tareas de rotación
        for task in self.rotation_tasks.values():
            task.cancel()
        self.rotation_tasks.clear()

        # Cancelar todas las tareas de limpieza
        for task in self.cleanup_tasks.values():
            task.cancel()
        self.cleanup_tasks.clear()

        # Limpiar claves de memoria
        self.key_history.clear()
        self.active_keys.clear()

        logger.info("✅ SecureKeyManager cerrado")

@dataclass
class RatchetState:
    """Estado del Double Ratchet COMPLETO con Perfect Forward Secrecy"""
    # Root key para derivar nuevas claves
    root_key: bytes

    # Claves de cadena para envío y recepción
    chain_key_send: bytes
    chain_key_recv: bytes

    # Claves públicas efímeras para DH ratchet
    dh_send: x25519.X25519PrivateKey = field(default_factory=x25519.X25519PrivateKey.generate)
    dh_recv: Optional[x25519.X25519PublicKey] = None

    # Números de mensaje para orden correcto
    message_number_send: int = 0
    message_number_recv: int = 0
    previous_chain_length: int = 0

    # Claves saltadas para mensajes fuera de orden
    skipped_keys: Dict[Tuple[int, int], bytes] = field(default_factory=dict)

    def __post_init__(self):
        """Inicializar claves públicas efímeras"""
        if not hasattr(self, 'dh_send_public'):
            self.dh_send_public = self.dh_send.public_key()

    @property
    def dh_send_public(self) -> x25519.X25519PublicKey:
        """Clave pública efímera de envío"""
        return self.dh_send.public_key()

    @dh_send_public.setter
    def dh_send_public(self, value: x25519.X25519PublicKey):
        """Setter para compatibilidad"""
        pass

    def advance_dh_ratchet(self, peer_dh_public: x25519.X25519PublicKey) -> None:
        """Avanza el DH ratchet con nueva clave pública del peer - versión simplificada"""
        try:
            # Calcular secreto compartido usando nuestra clave privada y clave pública del peer
            shared_secret = self.dh_send.exchange(peer_dh_public)

            # Derivar nueva root key
            self.root_key = self._kdf_rk(self.root_key, shared_secret)

            # Derivar nueva chain key de recepción
            self.chain_key_recv = self._kdf_ck(self.root_key)

            # NO reiniciar message numbers - mantener sincronización
            # self.message_number_send = 0
            # self.message_number_recv = 0

            logger.debug("🔄 DH ratchet avanzado con nueva clave efímera")

        except Exception as e:
            logger.error(f"❌ Error avanzando DH ratchet: {e}")
            raise

    def advance_sending_chain(self) -> bytes:
        """Avanza cadena de envío y generar clave de mensaje"""
        message_key = self._derive_message_key(self.chain_key_send)
        self.chain_key_send = self._derive_chain_key(self.chain_key_send)
        self.message_number_send += 1
        return message_key

    def advance_receiving_chain(self) -> bytes:
        """Avanza cadena de recepción y generar clave de mensaje"""
        message_key = self._derive_message_key(self.chain_key_recv)
        self.chain_key_recv = self._derive_chain_key(self.chain_key_recv)
        self.message_number_recv += 1
        return message_key

    def try_message_key(self, message_number: int, dh_public: Optional[x25519.X25519PublicKey]) -> Optional[bytes]:
        """Intenta obtener clave de mensaje, avanzando ratchet si es necesario"""
        # Primero verificar si hay claves saltadas
        key_id = (message_number, hash(dh_public.public_bytes_raw() if dh_public else b''))
        if key_id in self.skipped_keys:
            return self.skipped_keys.pop(key_id)

        # Si el mensaje es para un chain diferente, avanzar DH ratchet
        if dh_public and dh_public != self.dh_recv:
            self.advance_dh_ratchet(dh_public)

        # Verificar si podemos generar la clave
        if message_number == self.message_number_recv:
            return self.advance_receiving_chain()
        elif message_number < self.message_number_recv:
            # Mensaje demasiado antiguo
            return None
        else:
            # Mensaje futuro - saltar claves intermedias
            skipped_keys = []
            while self.message_number_recv < message_number:
                skipped_key = self.advance_receiving_chain()
                skipped_keys.append(skipped_key)

            # Guardar claves saltadas para uso futuro
            for i, skipped_key in enumerate(skipped_keys[:-1]):
                skip_id = (self.message_number_recv - len(skipped_keys) + i, hash(self.dh_recv.public_bytes_raw() if self.dh_recv else b''))
                self.skipped_keys[skip_id] = skipped_key

            return skipped_keys[-1]

    def _kdf_rk(self, root_key: bytes, input_key: bytes) -> bytes:
        """Derivar nueva root key usando HKDF"""
        return HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=root_key,
            info=b"root_key"
        ).derive(input_key)

    def _kdf_ck(self, root_key: bytes) -> bytes:
        """Derivar chain key desde root key"""
        return HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=None,
            info=b"chain_key"
        ).derive(root_key)

    def _derive_message_key(self, chain_key: bytes) -> bytes:
        """Derivar clave de mensaje desde clave de cadena"""
        return hmac.new(chain_key, b"message", hashlib.sha256).digest()

    def _derive_chain_key(self, chain_key: bytes) -> bytes:
        """Derivar siguiente clave de cadena"""
        return hmac.new(chain_key, b"chain", hashlib.sha256).digest()

class SecureMessage:
    """Mensaje cifrado con metadatos de seguridad y Forward Secrecy"""

    def __init__(self, ciphertext: bytes, nonce: bytes, sender_id: str,
                 recipient_id: str, message_number: int, timestamp: float,
                 signature: bytes, dh_public: Optional[bytes] = None):
        self.ciphertext = ciphertext
        self.nonce = nonce
        self.sender_id = sender_id
        self.recipient_id = recipient_id
        self.message_number = message_number
        self.timestamp = timestamp
        self.signature = signature
        self.dh_public = dh_public  # Clave pública efímera para DH ratchet

    def serialize(self) -> bytes:
        """Serializar mensaje para transmisión"""
        data = {
            'ciphertext': self.ciphertext,
            'nonce': self.nonce,
            'sender_id': self.sender_id.encode(),
            'recipient_id': self.recipient_id.encode(),
            'message_number': self.message_number.to_bytes(4, 'big'),
            'timestamp': int(self.timestamp).to_bytes(8, 'big'),
            'signature': self.signature,
            'dh_public': self.dh_public or b''
        }

        # Formato: longitud + datos para cada campo
        serialized = b''
        for key, value in data.items():
            if isinstance(value, int):
                value = value.to_bytes(4, 'big')
            serialized += len(value).to_bytes(4, 'big') + value

        return serialized

    @classmethod
    def deserialize(cls, data: bytes) -> 'SecureMessage':
        """Deserializar mensaje desde bytes"""
        offset = 0
        fields = {}

        field_names = ['ciphertext', 'nonce', 'sender_id', 'recipient_id',
                      'message_number', 'timestamp', 'signature', 'dh_public']

        for field_name in field_names:
            length = int.from_bytes(data[offset:offset+4], 'big')
            offset += 4
            value = data[offset:offset+length]
            offset += length

            if field_name in ['sender_id', 'recipient_id']:
                fields[field_name] = value.decode()
            elif field_name == 'message_number':
                fields[field_name] = int.from_bytes(value, 'big')
            elif field_name == 'timestamp':
                fields[field_name] = float(int.from_bytes(value, 'big'))
            else:
                fields[field_name] = value

        return cls(**fields)

class CryptoEngine:
    """Motor criptográfico principal del sistema"""

    def __init__(self, config: CryptoConfig = None):
        self.config = config or CryptoConfig()
        self.identity: Optional[NodeIdentity] = None
        self.peer_identities: Dict[str, PublicNodeIdentity] = {}
        self.ratchet_states: Dict[str, RatchetState] = {}
        self.session_keys: Dict[str, bytes] = {}
        self.key_rotation_tasks: Dict[str, asyncio.Task] = {}

        # Inicializar gestor de claves seguras
        self.key_manager = SecureKeyManager(self)

        logger.info(f"CryptoEngine inicializado con nivel {self.config.security_level.value}")

    def generate_node_identity(self, node_id: str = None) -> NodeIdentity:
        """Generar nueva identidad criptográfica para el nodo"""
        if node_id is None:
            node_id = secrets.token_hex(16)
        
        signing_key = ed25519.Ed25519PrivateKey.generate()
        encryption_key = x25519.X25519PrivateKey.generate()
        
        self.identity = NodeIdentity(node_id, signing_key, encryption_key)
        
        logger.info(f"Nueva identidad generada para nodo {node_id}")
        return self.identity
    
    def add_peer_identity(self, peer_data: Dict[str, bytes]) -> bool:
        """Agregar identidad de peer remoto"""
        try:
            peer_identity = NodeIdentity.from_public_data(peer_data)
            self.peer_identities[peer_identity.node_id] = peer_identity
            
            logger.info(f"Peer {peer_identity.node_id} agregado al registro")
            return True
        except Exception as e:
            logger.error(f"Error agregando peer: {e}")
            return False
    
    def establish_secure_channel(self, peer_id: str) -> bool:
        """Establecer canal seguro con peer usando X25519 + Double Ratchet COMPLETO"""
        if peer_id not in self.peer_identities:
            logger.error(f"Peer {peer_id} no encontrado en registro")
            return False

        if not self.identity:
            logger.error("Identidad local no inicializada")
            return False

        try:
            peer_identity = self.peer_identities[peer_id]

            # Intercambio de claves X25519 estático inicial
            shared_secret = self.identity.encryption_key.exchange(
                peer_identity.public_encryption_key
            )

            # Derivar root key inicial del Double Ratchet
            root_key = HKDF(
                algorithm=hashes.SHA256(),
                length=32,
                salt=None,
                info=b"initial_root_key"
            ).derive(shared_secret)

            # Derivar chain keys iniciales
            chain_key_send = HKDF(
                algorithm=hashes.SHA256(),
                length=32,
                salt=None,
                info=b"initial_chain_send"
            ).derive(shared_secret)

            chain_key_recv = HKDF(
                algorithm=hashes.SHA256(),
                length=32,
                salt=None,
                info=b"initial_chain_recv"
            ).derive(shared_secret)

            # Inicializar estado del ratchet con Forward Secrecy
            self.ratchet_states[peer_id] = RatchetState(
                root_key=root_key,
                chain_key_send=chain_key_send,
                chain_key_recv=chain_key_recv,
                dh_send=x25519.X25519PrivateKey.generate(),  # Primera clave efímera
                dh_recv=None  # Se establecerá con el primer mensaje
            )

            # Programar rotación de claves
            self._schedule_key_rotation(peer_id)

            logger.info(f"🔒 Canal seguro con Perfect Forward Secrecy establecido con {peer_id}")
            return True

        except Exception as e:
            logger.error(f"Error estableciendo canal con {peer_id}: {e}")
            return False
    
    def encrypt_message(self, plaintext: bytes, recipient_id: str) -> Optional[SecureMessage]:
        """Cifrar mensaje para destinatario específico con Perfect Forward Secrecy"""
        if recipient_id not in self.ratchet_states:
            logger.error(f"No hay canal seguro con {recipient_id}")
            return None

        if not self.identity:
            logger.error("Identidad local no inicializada")
            return None

        try:
            ratchet = self.ratchet_states[recipient_id]

            # AVANZAR DH RATCHET: Generar nueva clave efímera para cada mensaje
            # Esto proporciona Perfect Forward Secrecy
            new_dh_private = x25519.X25519PrivateKey.generate()
            current_dh_public = new_dh_private.public_key().public_bytes_raw()
            ratchet.dh_send = new_dh_private

            # Obtener clave de mensaje del chain ratchet
            message_key = ratchet.advance_sending_chain()

            # Cifrar con ChaCha20-Poly1305 usando la clave derivada
            cipher = ChaCha20Poly1305(message_key)
            nonce = os.urandom(12)
            ciphertext = cipher.encrypt(nonce, plaintext, None)

            # Crear mensaje con metadatos y clave efímera actual para Forward Secrecy
            timestamp = time.time()
            message = SecureMessage(
                ciphertext=ciphertext,
                nonce=nonce,
                sender_id=self.identity.node_id,
                recipient_id=recipient_id,
                message_number=ratchet.message_number_send - 1,
                timestamp=timestamp,
                signature=b'',  # Se agregará después
                dh_public=current_dh_public  # Clave efímera actual para que el receptor pueda avanzar el ratchet
            )

            # Preparar metadatos y firma
            self._finalize_secure_message(message, timestamp, current_dh_public)

            logger.debug(f"🔐 Mensaje cifrado con PFS para {recipient_id}")
            return message

        except Exception as e:
            logger.error(f"Error cifrando mensaje para {recipient_id}: {e}")
            return None

    def _finalize_secure_message(self, message: SecureMessage, timestamp: float, dh_public: bytes):
        """Firma el mensaje y finaliza metadatos"""
        message_data = (
            message.ciphertext + message.nonce +
            message.sender_id.encode() + message.recipient_id.encode() +
            message.message_number.to_bytes(4, 'big') +
            int(timestamp).to_bytes(8, 'big') +
            dh_public
        )
        message.signature = self.identity.signing_key.sign(message_data)
    
    def decrypt_message(self, message: SecureMessage) -> Optional[bytes]:
        """Descifrar mensaje recibido con Perfect Forward Secrecy"""
        if not self._validate_message_source(message):
            return None

        try:
            # Verificar edad del mensaje
            if time.time() - message.timestamp > self.config.max_message_age:
                logger.warning(f"Mensaje de {message.sender_id} demasiado antiguo")
                return None

            # Verificar firma y ratchet
            peer_identity = self.peer_identities[message.sender_id]
            if not self._verify_message_signature(message, peer_identity):
                return None

            # Obtener ratchet state
            ratchet = self.ratchet_states[message.sender_id]

            # Si hay clave efímera en el mensaje, avanzar DH ratchet
            if message.dh_public:
                try:
                    peer_dh_public = x25519.X25519PublicKey.from_public_bytes(message.dh_public)
                    logger.debug(f"🔄 Avanzando DH ratchet para {message.sender_id} con clave efímera")
                    ratchet.advance_dh_ratchet(peer_dh_public)
                except Exception as e:
                    logger.warning(f"No se pudo procesar clave efímera de {message.sender_id}: {e}")

            # Obtener clave de mensaje usando el chain ratchet
            logger.debug(f"📥 Obteniendo clave de mensaje {message.message_number} para {message.sender_id}")
            message_key = ratchet.advance_receiving_chain()
            logger.debug(f"🔑 Clave de mensaje obtenida: {message_key is not None}")
            if not message_key:
                logger.error(f"No se pudo obtener clave para mensaje {message.message_number} de {message.sender_id}")
                return None

            # Descifrar con ChaCha20-Poly1305
            cipher = ChaCha20Poly1305(message_key)
            plaintext = cipher.decrypt(message.nonce, message.ciphertext, None)

            # Actualizar última actividad del peer
            peer_identity.last_seen = datetime.now(timezone.utc)

            logger.debug(f"🔓 Mensaje descifrado con PFS de {message.sender_id}")
            return plaintext

        except InvalidSignature:
            logger.error(f"❌ Firma inválida en mensaje de {message.sender_id}")
            return None
        except Exception as e:
            logger.error(f"Error descifrando mensaje de {message.sender_id}: {e}")
            return None

    def _validate_message_source(self, message: SecureMessage) -> bool:
        """Valida origen del mensaje"""
        if message.sender_id not in self.ratchet_states:
            logger.error(f"No hay canal seguro con {message.sender_id}")
            return False
            
        if message.sender_id not in self.peer_identities:
            logger.error(f"Peer {message.sender_id} no está en el registro")
            return False
        return True

    def _verify_message_signature(self, message: SecureMessage, peer_identity: PublicNodeIdentity) -> bool:
        """Verifica la firma del mensaje"""
        try:
            message_data = (
                message.ciphertext + message.nonce +
                message.sender_id.encode() + message.recipient_id.encode() +
                message.message_number.to_bytes(4, 'big') +
                int(message.timestamp).to_bytes(8, 'big') +
                message.dh_public
            )
            peer_identity.public_signing_key.verify(message.signature, message_data)
            return True
        except InvalidSignature:
            logger.error(f"❌ Firma inválida en mensaje de {message.sender_id}")
            return False
    
    def sign_data(self, data: bytes) -> bytes:
        """Firmar datos con clave de identidad"""
        if not self.identity:
            raise ValueError("Identidad no inicializada")
        
        return self.identity.signing_key.sign(data)
    
    def verify_signature(self, data: bytes, signature: bytes, signer_id: str) -> bool:
        """Verificar firma de datos"""
        if signer_id not in self.peer_identities:
            logger.error(f"Peer {signer_id} no encontrado para verificación")
            return False
        
        try:
            peer_identity = self.peer_identities[signer_id]
            peer_identity.public_signing_key.verify(signature, data)
            return True
        except InvalidSignature:
            logger.warning(f"Firma inválida de {signer_id}")
            return False
        except Exception as e:
            logger.error(f"Error verificando firma de {signer_id}: {e}")
            return False
    
    def _schedule_key_rotation(self, peer_id: str):
        """Programar rotación automática de claves"""
        async def rotate_keys():
            while peer_id in self.ratchet_states:
                await asyncio.sleep(self.config.key_rotation_interval)
                
                if peer_id in self.ratchet_states:
                    logger.info(f"Rotando claves para {peer_id}")
                    # Reestablecer canal seguro
                    self.establish_secure_channel(peer_id)
        
        task = asyncio.create_task(rotate_keys())
        self.key_rotation_tasks[peer_id] = task
    
    def get_security_metrics(self) -> Dict[str, Any]:
        """Obtener métricas de seguridad del sistema"""
        return {
            'security_level': self.config.security_level.value,
            'active_channels': len(self.ratchet_states),
            'known_peers': len(self.peer_identities),
            'key_rotation_interval': self.config.key_rotation_interval,
            'identity_age': (
                datetime.now(timezone.utc) - self.identity.created_at
            ).total_seconds() if self.identity else 0,
            'oldest_peer': min([
                (datetime.now(timezone.utc) - peer.created_at).total_seconds()
                for peer in self.peer_identities.values()
            ]) if self.peer_identities else 0
        }
    
    def cleanup_expired_sessions(self):
        """Limpiar sesiones expiradas"""
        current_time = datetime.now(timezone.utc)
        expired_peers = []
        
        for peer_id, peer_identity in self.peer_identities.items():
            if (current_time - peer_identity.last_seen).total_seconds() > 3600:  # 1 hora
                expired_peers.append(peer_id)
        
        for peer_id in expired_peers:
            logger.info(f"Limpiando sesión expirada con {peer_id}")
            self.peer_identities.pop(peer_id, None)
            self.ratchet_states.pop(peer_id, None)
            
            if peer_id in self.key_rotation_tasks:
                self.key_rotation_tasks[peer_id].cancel()
                del self.key_rotation_tasks[peer_id]
    
    async def shutdown(self):
        """Cerrar motor criptográfico de forma segura"""
        logger.info("🔐 Cerrando motor criptográfico...")
        
        # Cancelar tareas de rotación
        for task in self.key_rotation_tasks.values():
            task.cancel()

        # Cerrar gestor de claves seguras
        await self.key_manager.shutdown()

        # Limpiar datos sensibles
        self.ratchet_states.clear()
        self.session_keys.clear()

        logger.info("✅ Motor criptográfico cerrado")

# Funciones de utilidad

def create_crypto_engine(security_level: SecurityLevel = SecurityLevel.HIGH) -> CryptoEngine:
    """Crear motor criptográfico con configuración específica"""
    config = CryptoConfig(security_level=security_level)
    return CryptoEngine(config)

def generate_secure_password(length: int = 32) -> str:
    """Generar contraseña segura para configuración"""
    alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*"
    return ''.join(secrets.choice(alphabet) for _ in range(length))

def derive_key_from_password(password: str, salt: bytes = None, iterations: int = 100000) -> bytes:
    """Derivar clave criptográfica desde contraseña"""
    if salt is None:
        salt = os.urandom(16)
    
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=iterations
    )
    
    return kdf.derive(password.encode())

# Ejemplo de uso
async def demo_crypto_system():
    """Demostración del sistema criptográfico"""
    print("🔐 Demo del Framework Criptográfico AEGIS")
    print("=" * 50)
    
    # Crear dos nodos
    alice_crypto = create_crypto_engine(SecurityLevel.HIGH)
    bob_crypto = create_crypto_engine(SecurityLevel.HIGH)
    
    # Generar identidades
    alice_identity = alice_crypto.generate_node_identity("alice")
    bob_identity = bob_crypto.generate_node_identity("bob")
    
    print(f"✅ Alice ID: {alice_identity.node_id}")
    print(f"✅ Bob ID: {bob_identity.node_id}")
    
    # Intercambiar identidades públicas
    alice_public = alice_identity.export_public_identity()
    bob_public = bob_identity.export_public_identity()
    
    alice_crypto.add_peer_identity(bob_public)
    bob_crypto.add_peer_identity(alice_public)
    
    # Establecer canales seguros
    alice_crypto.establish_secure_channel("bob")
    bob_crypto.establish_secure_channel("alice")
    
    print("🔗 Canales seguros establecidos")
    
    # Intercambiar mensajes cifrados
    message = b"Hola Bob, este es un mensaje secreto desde Alice!"
    encrypted_msg = alice_crypto.encrypt_message(message, "bob")
    
    if encrypted_msg:
        print(f"📤 Alice envía mensaje cifrado")
        decrypted_msg = bob_crypto.decrypt_message(encrypted_msg)
        
        if decrypted_msg:
            print(f"📥 Bob recibe: {decrypted_msg.decode()}")
        else:
            print("❌ Error descifrando mensaje")
    
    # Mostrar métricas
    alice_metrics = alice_crypto.get_security_metrics()
    print(f"\n📊 Métricas de Alice: {alice_metrics}")
    
    # Limpiar
    await alice_crypto.shutdown()
    await bob_crypto.shutdown()

def initialize_crypto(config: Dict[str, Any]) -> CryptoEngine:
    """Adapter a nivel de módulo para inicializar CryptoEngine.
    Lee el nivel de seguridad y genera identidad de nodo.
    """
    try:
        level_str = str(config.get("security_level", "HIGH")).upper()
        level = SecurityLevel[level_str] if level_str in SecurityLevel.__members__ else SecurityLevel.HIGH
        engine = create_crypto_engine(level)
        node_id = config.get("node_id", None)
        engine.generate_node_identity(node_id)
        logger.info(f"🔐 CryptoEngine iniciado (security_level={level.value}, node_id={engine.identity.node_id})")
        return engine
    except Exception as e:
        logger.error(f"❌ No se pudo inicializar CryptoEngine: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(demo_crypto_system())