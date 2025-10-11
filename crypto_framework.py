#!/usr/bin/env python3
"""
Framework Criptográfico para IA Distribuida y Colaborativa
AEGIS Security Framework - Uso Ético Únicamente

Este módulo implementa un sistema criptográfico robusto para:
- Autenticación de nodos mediante Ed25519
- Cifrado de extremo a extremo con ChaCha20-Poly1305
- Intercambio de claves con X25519
- Double Ratchet para forward secrecy
- Firmas digitales para integridad de datos

ADVERTENCIA: Este código es para investigación y desarrollo ético únicamente.
El uso malicioso está estrictamente prohibido.
"""

import os
import time
import hmac
import hashlib
import secrets
from typing import Dict, Optional, Tuple, List, Any
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import logging
from datetime import datetime, timedelta

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
    created_at: datetime = field(default_factory=datetime.utcnow)
    
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
    last_seen: datetime = field(default_factory=datetime.utcnow)
    trust_score: float = 0.5  # Puntuación de confianza inicial

@dataclass
class RatchetState:
    """Estado del Double Ratchet para forward secrecy"""
    root_key: bytes
    chain_key_send: bytes
    chain_key_recv: bytes
    message_number_send: int = 0
    message_number_recv: int = 0
    previous_chain_length: int = 0
    skipped_keys: Dict[Tuple[int, int], bytes] = field(default_factory=dict)
    
    def advance_sending_chain(self) -> bytes:
        """Avanzar cadena de envío y generar clave de mensaje"""
        message_key = self._derive_message_key(self.chain_key_send)
        self.chain_key_send = self._derive_chain_key(self.chain_key_send)
        self.message_number_send += 1
        return message_key
    
    def advance_receiving_chain(self) -> bytes:
        """Avanzar cadena de recepción y generar clave de mensaje"""
        message_key = self._derive_message_key(self.chain_key_recv)
        self.chain_key_recv = self._derive_chain_key(self.chain_key_recv)
        self.message_number_recv += 1
        return message_key
    
    def _derive_message_key(self, chain_key: bytes) -> bytes:
        """Derivar clave de mensaje desde clave de cadena"""
        return hmac.new(chain_key, b"message", hashlib.sha256).digest()
    
    def _derive_chain_key(self, chain_key: bytes) -> bytes:
        """Derivar siguiente clave de cadena"""
        return hmac.new(chain_key, b"chain", hashlib.sha256).digest()

class SecureMessage:
    """Mensaje cifrado con metadatos de seguridad"""
    
    def __init__(self, ciphertext: bytes, nonce: bytes, sender_id: str,
                 recipient_id: str, message_number: int, timestamp: float,
                 signature: bytes):
        self.ciphertext = ciphertext
        self.nonce = nonce
        self.sender_id = sender_id
        self.recipient_id = recipient_id
        self.message_number = message_number
        self.timestamp = timestamp
        self.signature = signature
    
    def serialize(self) -> bytes:
        """Serializar mensaje para transmisión"""
        data = {
            'ciphertext': self.ciphertext,
            'nonce': self.nonce,
            'sender_id': self.sender_id.encode(),
            'recipient_id': self.recipient_id.encode(),
            'message_number': self.message_number.to_bytes(4, 'big'),
            'timestamp': int(self.timestamp).to_bytes(8, 'big'),
            'signature': self.signature
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
                      'message_number', 'timestamp', 'signature']
        
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
        """Establecer canal seguro con peer usando X25519 + Double Ratchet"""
        if peer_id not in self.peer_identities:
            logger.error(f"Peer {peer_id} no encontrado en registro")
            return False
        
        if not self.identity:
            logger.error("Identidad local no inicializada")
            return False
        
        try:
            peer_identity = self.peer_identities[peer_id]
            
            # Intercambio de claves X25519
            shared_secret = self.identity.encryption_key.exchange(
                peer_identity.public_encryption_key
            )
            
            # Derivar claves del Double Ratchet
            root_key = HKDF(
                algorithm=hashes.SHA256(),
                length=32,
                salt=None,
                info=b"root_key"
            ).derive(shared_secret)
            
            chain_key_send = HKDF(
                algorithm=hashes.SHA256(),
                length=32,
                salt=None,
                info=b"chain_send"
            ).derive(shared_secret)
            
            chain_key_recv = HKDF(
                algorithm=hashes.SHA256(),
                length=32,
                salt=None,
                info=b"chain_recv"
            ).derive(shared_secret)
            
            # Inicializar estado del ratchet
            self.ratchet_states[peer_id] = RatchetState(
                root_key=root_key,
                chain_key_send=chain_key_send,
                chain_key_recv=chain_key_recv
            )
            
            # Programar rotación de claves
            self._schedule_key_rotation(peer_id)
            
            logger.info(f"Canal seguro establecido con {peer_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error estableciendo canal con {peer_id}: {e}")
            return False
    
    def encrypt_message(self, plaintext: bytes, recipient_id: str) -> Optional[SecureMessage]:
        """Cifrar mensaje para destinatario específico"""
        if recipient_id not in self.ratchet_states:
            logger.error(f"No hay canal seguro con {recipient_id}")
            return None
        
        if not self.identity:
            logger.error("Identidad local no inicializada")
            return None
        
        try:
            ratchet = self.ratchet_states[recipient_id]
            
            # Obtener clave de mensaje del ratchet
            message_key = ratchet.advance_sending_chain()
            
            # Cifrar con ChaCha20-Poly1305
            cipher = ChaCha20Poly1305(message_key)
            nonce = os.urandom(12)
            ciphertext = cipher.encrypt(nonce, plaintext, None)
            
            # Crear mensaje con metadatos
            timestamp = time.time()
            message = SecureMessage(
                ciphertext=ciphertext,
                nonce=nonce,
                sender_id=self.identity.node_id,
                recipient_id=recipient_id,
                message_number=ratchet.message_number_send - 1,
                timestamp=timestamp,
                signature=b''  # Se agregará después
            )
            
            # Firmar mensaje
            message_data = (
                message.ciphertext + message.nonce + 
                message.sender_id.encode() + message.recipient_id.encode() +
                message.message_number.to_bytes(4, 'big') +
                int(timestamp).to_bytes(8, 'big')
            )
            
            message.signature = self.identity.signing_key.sign(message_data)
            
            logger.debug(f"Mensaje cifrado para {recipient_id}")
            return message
            
        except Exception as e:
            logger.error(f"Error cifrando mensaje para {recipient_id}: {e}")
            return None
    
    def decrypt_message(self, message: SecureMessage) -> Optional[bytes]:
        """Descifrar mensaje recibido"""
        if message.sender_id not in self.ratchet_states:
            logger.error(f"No hay canal seguro con {message.sender_id}")
            return None
        
        if message.sender_id not in self.peer_identities:
            logger.error(f"Peer {message.sender_id} no está en el registro")
            return None
        
        try:
            # Verificar edad del mensaje
            if time.time() - message.timestamp > self.config.max_message_age:
                logger.warning(f"Mensaje de {message.sender_id} demasiado antiguo")
                return None
            
            # Verificar firma
            peer_identity = self.peer_identities[message.sender_id]
            message_data = (
                message.ciphertext + message.nonce + 
                message.sender_id.encode() + message.recipient_id.encode() +
                message.message_number.to_bytes(4, 'big') +
                int(message.timestamp).to_bytes(8, 'big')
            )
            
            peer_identity.public_signing_key.verify(message.signature, message_data)
            
            # Obtener clave de mensaje del ratchet
            ratchet = self.ratchet_states[message.sender_id]
            message_key = ratchet.advance_receiving_chain()
            
            # Descifrar
            cipher = ChaCha20Poly1305(message_key)
            plaintext = cipher.decrypt(message.nonce, message.ciphertext, None)
            
            # Actualizar última actividad del peer
            peer_identity.last_seen = datetime.utcnow()
            
            logger.debug(f"Mensaje descifrado de {message.sender_id}")
            return plaintext
            
        except InvalidSignature:
            logger.error(f"Firma inválida en mensaje de {message.sender_id}")
            return None
        except Exception as e:
            logger.error(f"Error descifrando mensaje de {message.sender_id}: {e}")
            return None
    
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
                datetime.utcnow() - self.identity.created_at
            ).total_seconds() if self.identity else 0,
            'oldest_peer': min([
                (datetime.utcnow() - peer.created_at).total_seconds()
                for peer in self.peer_identities.values()
            ]) if self.peer_identities else 0
        }
    
    def cleanup_expired_sessions(self):
        """Limpiar sesiones expiradas"""
        current_time = datetime.utcnow()
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
        logger.info("Cerrando motor criptográfico...")
        
        # Cancelar tareas de rotación
        for task in self.key_rotation_tasks.values():
            task.cancel()
        
        # Limpiar datos sensibles
        self.ratchet_states.clear()
        self.session_keys.clear()
        
        logger.info("Motor criptográfico cerrado")

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