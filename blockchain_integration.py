#!/usr/bin/env python3
"""
Integración Blockchain - AEGIS Framework
Sistema blockchain distribuido para garantizar inmutabilidad de datos,
contratos inteligentes y consenso descentralizado.

Características principales:
- Blockchain personalizado con PoS (Proof of Stake)
- Contratos inteligentes para IA distribuida
- Inmutabilidad de modelos y datos de entrenamiento
- Tokenización de recursos computacionales
- Auditoría transparente de decisiones de IA
"""

import asyncio
import time
import json
import hashlib
import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
from collections import deque
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
import merkletools
import sqlite3
import pickle
import base64

# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TransactionType(Enum):
    """Tipos de transacciones en la blockchain"""
    TRANSFER = "transfer"
    SMART_CONTRACT = "smart_contract"
    MODEL_REGISTRATION = "model_registration"
    DATA_VALIDATION = "data_validation"
    CONSENSUS_VOTE = "consensus_vote"
    RESOURCE_ALLOCATION = "resource_allocation"
    AI_DECISION = "ai_decision"


class BlockStatus(Enum):
    """Estados de un bloque"""
    PENDING = "pending"
    MINING = "mining"
    VALIDATED = "validated"
    CONFIRMED = "confirmed"
    REJECTED = "rejected"


class ContractStatus(Enum):
    """Estados de contratos inteligentes"""
    DEPLOYED = "deployed"
    ACTIVE = "active"
    PAUSED = "paused"
    TERMINATED = "terminated"
    ERROR = "error"


class StakeStatus(Enum):
    """Estados de stake"""
    ACTIVE = "active"
    LOCKED = "locked"
    SLASHED = "slashed"
    WITHDRAWN = "withdrawn"


@dataclass
class Transaction:
    """Transacción en la blockchain"""
    tx_id: str
    tx_type: TransactionType
    sender: str
    recipient: str
    amount: float
    data: Dict[str, Any]
    timestamp: float
    nonce: int
    gas_limit: int
    gas_price: float
    signature: str
    public_key: str


@dataclass
class Block:
    """Bloque en la blockchain"""
    block_id: str
    index: int
    previous_hash: str
    merkle_root: str
    timestamp: float
    transactions: List[Transaction]
    validator: str
    stake_weight: float
    nonce: int
    difficulty: int
    block_hash: str
    status: BlockStatus
    confirmations: int


@dataclass
class SmartContract:
    """Contrato inteligente"""
    contract_id: str
    name: str
    version: str
    creator: str
    code: str
    abi: Dict[str, Any]
    state: Dict[str, Any]
    balance: float
    gas_used: int
    status: ContractStatus
    created_at: float
    updated_at: float


@dataclass
class ValidatorStake:
    """Stake de un validador"""
    validator_id: str
    stake_amount: float
    locked_until: float
    status: StakeStatus
    rewards_earned: float
    penalties: float
    delegation_count: int
    performance_score: float


@dataclass
class AIModel:
    """Modelo de IA registrado en blockchain"""
    model_id: str
    name: str
    version: str
    creator: str
    model_hash: str
    metadata: Dict[str, Any]
    training_data_hash: str
    performance_metrics: Dict[str, float]
    validation_results: Dict[str, Any]
    deployment_status: str
    created_at: float


class CryptographicManager:
    """Gestor de operaciones criptográficas"""

    def __init__(self):
        self.key_pairs: Dict[str, Dict[str, Any]] = {}

    def generate_key_pair(self, node_id: str) -> Dict[str, str]:
        """Genera par de claves para un nodo"""
        try:
            # Generar clave privada RSA
            private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048
            )

            # Obtener clave pública
            public_key = private_key.public_key()

            # Serializar claves
            private_pem = private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            )

            public_pem = public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            )

            # Almacenar claves
            self.key_pairs[node_id] = {
                'private_key': private_key,
                'public_key': public_key,
                'private_pem': private_pem.decode('utf-8'),
                'public_pem': public_pem.decode('utf-8')
            }

            logger.info(f"🔐 Par de claves generado para {node_id}")

            return {
                'private_key': private_pem.decode('utf-8'),
                'public_key': public_pem.decode('utf-8')
            }

        except Exception as e:
            logger.error(f"❌ Error generando claves para {node_id}: {e}")
            return {}

    def sign_data(self, node_id: str, data: bytes) -> str:
        """Firma datos con clave privada"""
        try:
            if node_id not in self.key_pairs:
                raise ValueError(f"No hay claves para {node_id}")

            private_key = self.key_pairs[node_id]['private_key']

            signature = private_key.sign(
                data,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )

            return base64.b64encode(signature).decode('utf-8')

        except Exception as e:
            logger.error(f"❌ Error firmando datos: {e}")
            return ""

    def verify_signature(self, public_key_pem: str, data: bytes, signature: str) -> bool:
        """Verifica firma digital"""
        try:
            # Deserializar clave pública
            public_key = serialization.load_pem_public_key(public_key_pem.encode('utf-8'))

            # Decodificar firma
            signature_bytes = base64.b64decode(signature)

            # Verificar firma
            public_key.verify(
                signature_bytes,
                data,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )

            return True

        except Exception as e:
            logger.debug(f"⚠️ Firma inválida: {e}")
            return False

    def hash_data(self, data: Union[str, bytes, Dict[str, Any]]) -> str:
        """Calcula hash SHA-256 de datos"""
        try:
            if isinstance(data, dict):
                data = json.dumps(data, sort_keys=True)

            if isinstance(data, str):
                data = data.encode('utf-8')

            return hashlib.sha256(data).hexdigest()

        except Exception as e:
            logger.error(f"❌ Error calculando hash: {e}")
            return ""

    def create_merkle_tree(self, transactions: List[Transaction]) -> str:
        """Crea árbol de Merkle para transacciones"""
        try:
            mt = merkletools.MerkleTools(hash_type="sha256")

            # Agregar hashes de transacciones
            for tx in transactions:
                tx_data = json.dumps(asdict(tx), sort_keys=True)
                mt.add_leaf(tx_data)

            # Construir árbol
            mt.make_tree()

            # Obtener raíz de Merkle
            merkle_root = mt.get_merkle_root()

            return merkle_root.hex() if merkle_root else ""

        except Exception as e:
            logger.error(f"❌ Error creando árbol de Merkle: {e}")
            return ""


class TransactionPool:
    """Pool de transacciones pendientes"""

    def __init__(self, max_size: int = 10000):
        self.pending_transactions: Dict[str, Transaction] = {}
        self.transaction_queue = deque()
        self.max_size = max_size
        self.crypto_manager = CryptographicManager()

    def add_transaction(self, transaction: Transaction) -> bool:
        """Agrega transacción al pool"""
        try:
            # Verificar si el pool está lleno
            if len(self.pending_transactions) >= self.max_size:
                logger.warning("⚠️ Pool de transacciones lleno")
                return False

            # Validar transacción
            if not self._validate_transaction(transaction):
                logger.warning(f"⚠️ Transacción inválida: {transaction.tx_id}")
                return False

            # Agregar al pool
            self.pending_transactions[transaction.tx_id] = transaction
            self.transaction_queue.append(transaction.tx_id)

            logger.debug(f"📝 Transacción agregada al pool: {transaction.tx_id}")
            return True

        except Exception as e:
            logger.error(f"❌ Error agregando transacción: {e}")
            return False

    def _validate_transaction(self, transaction: Transaction) -> bool:
        """Valida una transacción"""
        try:
            # Verificar campos obligatorios
            if not all([transaction.tx_id, transaction.sender, transaction.signature]):
                return False

            # Verificar que no sea duplicada
            if transaction.tx_id in self.pending_transactions:
                return False

            # Verificar firma digital
            tx_data = self._get_transaction_data_for_signing(transaction)
            if not self.crypto_manager.verify_signature(
                transaction.public_key,
                tx_data.encode('utf-8'),
                transaction.signature
            ):
                return False

            # Verificar timestamp (no muy antigua ni futura)
            current_time = time.time()
            if abs(current_time - transaction.timestamp) > 3600:  # 1 hora
                return False

            # Validaciones específicas por tipo
            if transaction.tx_type == TransactionType.TRANSFER:
                if transaction.amount <= 0:
                    return False

            return True

        except Exception as e:
            logger.error(f"❌ Error validando transacción: {e}")
            return False

    def _get_transaction_data_for_signing(self, transaction: Transaction) -> str:
        """Obtiene datos de transacción para firma"""
        data = {
            'tx_id': transaction.tx_id,
            'tx_type': transaction.tx_type.value,
            'sender': transaction.sender,
            'recipient': transaction.recipient,
            'amount': transaction.amount,
            'data': transaction.data,
            'timestamp': transaction.timestamp,
            'nonce': transaction.nonce
        }
        return json.dumps(data, sort_keys=True)

    def get_transactions_for_block(self, max_count: int = 100) -> List[Transaction]:
        """Obtiene transacciones para crear un bloque"""
        transactions = []

        while len(transactions) < max_count and self.transaction_queue:
            tx_id = self.transaction_queue.popleft()
            if tx_id in self.pending_transactions:
                transactions.append(self.pending_transactions[tx_id])
                del self.pending_transactions[tx_id]

        return transactions

    def remove_transaction(self, tx_id: str):
        """Remueve transacción del pool"""
        if tx_id in self.pending_transactions:
            del self.pending_transactions[tx_id]

    def get_pending_count(self) -> int:
        """Obtiene número de transacciones pendientes"""
        return len(self.pending_transactions)


class ProofOfStakeValidator:
    """Validador Proof of Stake"""

    def __init__(self, node_id: str):
        self.node_id = node_id
        self.validators: Dict[str, ValidatorStake] = {}
        self.total_stake = 0.0
        self.min_stake = 1000.0  # Stake mínimo para ser validador
        self.reward_rate = 0.05  # 5% anual
        self.slash_rate = 0.1    # 10% de penalización

    def register_validator(self, validator_id: str, stake_amount: float) -> bool:
        """Registra un nuevo validador"""
        try:
            if stake_amount < self.min_stake:
                logger.warning(f"⚠️ Stake insuficiente para {validator_id}: {stake_amount}")
                return False

            validator_stake = ValidatorStake(
                validator_id=validator_id,
                stake_amount=stake_amount,
                locked_until=time.time() + (30 * 24 * 3600),  # 30 días
                status=StakeStatus.ACTIVE,
                rewards_earned=0.0,
                penalties=0.0,
                delegation_count=0,
                performance_score=1.0
            )

            self.validators[validator_id] = validator_stake
            self.total_stake += stake_amount

            logger.info(f"✅ Validador registrado: {validator_id} con stake {stake_amount}")
            return True

        except Exception as e:
            logger.error(f"❌ Error registrando validador: {e}")
            return False

    def select_validator(self, block_height: int) -> Optional[str]:
        """Selecciona validador usando algoritmo PoS"""
        try:
            if not self.validators:
                return None

            # Filtrar validadores activos
            active_validators = {
                v_id: stake for v_id, stake in self.validators.items()
                if stake.status == StakeStatus.ACTIVE and stake.stake_amount > 0
            }

            if not active_validators:
                return None

            # Calcular probabilidades basadas en stake
            total_active_stake = sum(stake.stake_amount for stake in active_validators.values())

            # Usar hash del bloque anterior como semilla
            seed = hashlib.sha256(f"{block_height}{time.time()}".encode()).hexdigest()
            random_value = int(seed[:8], 16) / 0xFFFFFFFF

            # Selección ponderada por stake
            cumulative_probability = 0.0

            for validator_id, stake in active_validators.items():
                probability = stake.stake_amount / total_active_stake
                cumulative_probability += probability

                if random_value <= cumulative_probability:
                    logger.debug(f"🎯 Validador seleccionado: {validator_id}")
                    return validator_id

            # Fallback: seleccionar el último
            return list(active_validators.keys())[-1]

        except Exception as e:
            logger.error(f"❌ Error seleccionando validador: {e}")
            return None

    def validate_block(self, validator_id: str, block: Block) -> bool:
        """Valida un bloque propuesto"""
        try:
            if validator_id not in self.validators:
                return False

            validator = self.validators[validator_id]

            # Verificar que el validador esté activo
            if validator.status != StakeStatus.ACTIVE:
                return False

            # Verificar estructura del bloque
            if not self._validate_block_structure(block):
                return False

            # Verificar transacciones
            if not self._validate_block_transactions(block):
                return False

            # Verificar hash del bloque
            if not self._validate_block_hash(block):
                return False

            return True

        except Exception as e:
            logger.error(f"❌ Error validando bloque: {e}")
            return False

    def _validate_block_structure(self, block: Block) -> bool:
        """Valida estructura del bloque"""
        required_fields = [
            'block_id',
            'index',
            'previous_hash',
            'merkle_root',
            'timestamp',
            'transactions',
            'validator',
        ]

        for req_field in required_fields:
            if not hasattr(block, req_field) or getattr(block, req_field) is None:
                return False

        return True

    def _validate_block_transactions(self, block: Block) -> bool:
        """Valida transacciones del bloque"""
        # Verificar que no esté vacío (excepto bloque génesis)
        if block.index > 0 and not block.transactions:
            return False

        # Verificar límite de transacciones
        if len(block.transactions) > 1000:
            return False

        # Verificar duplicados
        tx_ids = [tx.tx_id for tx in block.transactions]
        if len(tx_ids) != len(set(tx_ids)):
            return False

        return True

    def _validate_block_hash(self, block: Block) -> bool:
        """Valida hash del bloque"""
        try:
            # Recalcular hash
            block_data = {
                'index': block.index,
                'previous_hash': block.previous_hash,
                'merkle_root': block.merkle_root,
                'timestamp': block.timestamp,
                'validator': block.validator,
                'nonce': block.nonce
            }

            calculated_hash = hashlib.sha256(
                json.dumps(block_data, sort_keys=True).encode()
            ).hexdigest()

            return calculated_hash == block.block_hash

        except Exception as e:
            logger.error(f"❌ Error validando hash de bloque: {e}")
            return False

    def reward_validator(self, validator_id: str, reward_amount: float):
        """Otorga recompensa a validador"""
        if validator_id in self.validators:
            self.validators[validator_id].rewards_earned += reward_amount
            logger.debug(f"💰 Recompensa otorgada a {validator_id}: {reward_amount}")

    def slash_validator(self, validator_id: str, slash_amount: float):
        """Penaliza validador por mal comportamiento"""
        if validator_id in self.validators:
            validator = self.validators[validator_id]
            validator.penalties += slash_amount
            validator.stake_amount = max(0, validator.stake_amount - slash_amount)

            if validator.stake_amount < self.min_stake:
                validator.status = StakeStatus.SLASHED

            logger.warning(f"⚡ Validador penalizado {validator_id}: -{slash_amount}")


class SmartContractEngine:
    """Motor de contratos inteligentes"""

    def __init__(self):
        self.contracts: Dict[str, SmartContract] = {}
        self.contract_storage: Dict[str, Dict[str, Any]] = {}
        self.gas_price = 0.001  # Precio del gas
        self.max_gas_per_contract = 1000000

    def deploy_contract(
        self,
        creator: str,
        name: str,
        code: str,
        initial_balance: float = 0.0,
    ) -> Optional[str]:
        """Despliega un nuevo contrato inteligente"""
        try:
            contract_id = hashlib.sha256(f"{creator}{name}{time.time()}".encode()).hexdigest()

            # Validar código del contrato
            if not self._validate_contract_code(code):
                logger.error("❌ Código de contrato inválido")
                return None

            # Crear contrato
            contract = SmartContract(
                contract_id=contract_id,
                name=name,
                version="1.0.0",
                creator=creator,
                code=code,
                abi=self._generate_abi(code),
                state={},
                balance=initial_balance,
                gas_used=0,
                status=ContractStatus.DEPLOYED,
                created_at=time.time(),
                updated_at=time.time()
            )

            # Almacenar contrato
            self.contracts[contract_id] = contract
            self.contract_storage[contract_id] = {}

            logger.info(f"📜 Contrato desplegado: {name} ({contract_id[:8]}...)")
            return contract_id

        except Exception as e:
            logger.error(f"❌ Error desplegando contrato: {e}")
            return None

    def _validate_contract_code(self, code: str) -> bool:
        """Valida código del contrato"""
        try:
            # Verificaciones básicas de seguridad
            forbidden_imports = ['os', 'sys', 'subprocess', 'eval', 'exec']

            for forbidden in forbidden_imports:
                if forbidden in code:
                    logger.warning(f"⚠️ Importación prohibida detectada: {forbidden}")
                    return False

            # Verificar que sea código Python válido
            compile(code, '<contract>', 'exec')

            return True

        except SyntaxError as e:
            logger.error(f"❌ Error de sintaxis en contrato: {e}")
            return False
        except Exception as e:
            logger.error(f"❌ Error validando contrato: {e}")
            return False

    def _generate_abi(self, code: str) -> Dict[str, Any]:
        """Genera ABI (Application Binary Interface) del contrato"""
        # Implementación simplificada - en producción sería más compleja
        abi = {
            "functions": [],
            "events": [],
            "constructor": {}
        }

        # Analizar código para extraer funciones públicas
        lines = code.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('def ') and not line.startswith('def _'):
                func_name = line.split('(')[0].replace('def ', '')
                abi["functions"].append({
                    "name": func_name,
                    "type": "function",
                    "inputs": [],
                    "outputs": []
                })

        return abi

    def execute_contract(
        self,
        contract_id: str,
        function_name: str,
        parameters: Dict[str, Any],
        caller: str,
        gas_limit: int = 100000,
    ) -> Dict[str, Any]:
        """Ejecuta función de contrato inteligente"""
        try:
            if contract_id not in self.contracts:
                return {"error": "Contract not found"}

            contract = self.contracts[contract_id]

            if contract.status != ContractStatus.ACTIVE and contract.status != ContractStatus.DEPLOYED:
                return {"error": "Contract not active"}

            # Verificar gas disponible
            gas_cost = self._estimate_gas_cost(function_name, parameters)
            if gas_cost > gas_limit:
                return {"error": "Gas limit exceeded"}

            # Crear contexto de ejecución
            execution_context = {
                'contract_id': contract_id,
                'caller': caller,
                'block_timestamp': time.time(),
                'gas_remaining': gas_limit,
                'storage': self.contract_storage[contract_id],
                'balance': contract.balance
            }

            # Ejecutar función
            result = self._execute_contract_function(
                contract, function_name, parameters, execution_context
            )

            # Actualizar estado del contrato
            contract.gas_used += gas_cost
            contract.updated_at = time.time()

            # Actualizar storage si cambió
            self.contract_storage[contract_id] = execution_context['storage']

            logger.debug(f"⚙️ Contrato ejecutado: {function_name} en {contract_id[:8]}...")

            return {
                "success": True,
                "result": result,
                "gas_used": gas_cost,
                "new_state": contract.state
            }

        except Exception as e:
            logger.error(f"❌ Error ejecutando contrato: {e}")
            return {"error": str(e)}

    def _estimate_gas_cost(self, function_name: str, parameters: Dict[str, Any]) -> int:
        """Estima costo de gas para ejecución"""
        base_cost = 1000
        param_cost = len(str(parameters)) * 10
        return base_cost + param_cost

    def _execute_contract_function(
        self,
        contract: SmartContract,
        function_name: str,
        parameters: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Any:
        """Ejecuta función específica del contrato"""
        try:
            # Crear namespace seguro para ejecución
            safe_globals = {
                '__builtins__': {
                    'len': len,
                    'str': str,
                    'int': int,
                    'float': float,
                    'bool': bool,
                    'list': list,
                    'dict': dict,
                    'max': max,
                    'min': min,
                    'sum': sum,
                    'abs': abs,
                    'round': round
                },
                'context': context,
                'storage': context['storage'],
                'caller': context['caller'],
                'timestamp': context['block_timestamp']
            }

            # Ejecutar código del contrato
            exec(contract.code, safe_globals)

            # Llamar función específica
            if function_name in safe_globals:
                function = safe_globals[function_name]
                return function(**parameters)
            else:
                raise ValueError(f"Function {function_name} not found")

        except Exception as e:
            logger.error(f"❌ Error en ejecución de contrato: {e}")
            raise

    def get_contract_state(self, contract_id: str) -> Optional[Dict[str, Any]]:
        """Obtiene estado actual del contrato"""
        if contract_id in self.contracts:
            contract = self.contracts[contract_id]
            return {
                "contract_id": contract_id,
                "name": contract.name,
                "status": contract.status.value,
                "balance": contract.balance,
                "gas_used": contract.gas_used,
                "state": contract.state,
                "storage": self.contract_storage.get(contract_id, {})
            }
        return None


class BlockchainCore:
    """Núcleo de la blockchain"""

    def __init__(self, node_id: str, genesis_validator: str = None):
        self.node_id = node_id
        self.chain: List[Block] = []
        self.transaction_pool = TransactionPool()
        self.pos_validator = ProofOfStakeValidator(node_id)
        self.smart_contract_engine = SmartContractEngine()
        self.crypto_manager = CryptographicManager()

        # Estado de la blockchain
        self.current_difficulty = 4
        self.block_time_target = 10  # segundos
        self.max_block_size = 1000000  # bytes

        # Base de datos
        self.db_path = f"blockchain_{node_id}.db"
        self._init_database()

        # Crear bloque génesis
        if not self.chain:
            self._create_genesis_block(genesis_validator or node_id)

    def _init_database(self):
        """Inicializa base de datos SQLite"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Tabla de bloques
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS blocks (
                    block_id TEXT PRIMARY KEY,
                    index_num INTEGER,
                    previous_hash TEXT,
                    merkle_root TEXT,
                    timestamp REAL,
                    validator TEXT,
                    block_hash TEXT,
                    block_data BLOB
                )
            ''')

            # Tabla de transacciones
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS transactions (
                    tx_id TEXT PRIMARY KEY,
                    block_id TEXT,
                    tx_type TEXT,
                    sender TEXT,
                    recipient TEXT,
                    amount REAL,
                    timestamp REAL,
                    tx_data BLOB
                )
            ''')

            # Tabla de contratos
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS contracts (
                    contract_id TEXT PRIMARY KEY,
                    name TEXT,
                    creator TEXT,
                    status TEXT,
                    created_at REAL,
                    contract_data BLOB
                )
            ''')

            conn.commit()
            conn.close()

            logger.info(f"💾 Base de datos inicializada: {self.db_path}")

        except Exception as e:
            logger.error(f"❌ Error inicializando base de datos: {e}")

    def _create_genesis_block(self, genesis_validator: str):
        """Crea el bloque génesis"""
        try:
            genesis_transaction = Transaction(
                tx_id="genesis_tx",
                tx_type=TransactionType.TRANSFER,
                sender="genesis",
                recipient=genesis_validator,
                amount=1000000.0,  # Tokens iniciales
                data={"type": "genesis", "message": "AEGIS Genesis Block"},
                timestamp=time.time(),
                nonce=0,
                gas_limit=0,
                gas_price=0.0,
                signature="genesis_signature",
                public_key="genesis_public_key"
            )

            merkle_root = self.crypto_manager.create_merkle_tree([genesis_transaction])

            genesis_block = Block(
                block_id="genesis_block",
                index=0,
                previous_hash="0" * 64,
                merkle_root=merkle_root,
                timestamp=time.time(),
                transactions=[genesis_transaction],
                validator=genesis_validator,
                stake_weight=1.0,
                nonce=0,
                difficulty=0,
                block_hash="",
                status=BlockStatus.CONFIRMED,
                confirmations=1
            )

            # Calcular hash del bloque
            genesis_block.block_hash = self._calculate_block_hash(genesis_block)

            # Agregar a la cadena
            self.chain.append(genesis_block)

            # Guardar en base de datos
            self._save_block_to_db(genesis_block)

            logger.info("🌱 Bloque génesis creado")

        except Exception as e:
            logger.error(f"❌ Error creando bloque génesis: {e}")

    def add_transaction(self, transaction: Transaction) -> bool:
        """Agrega transacción al pool"""
        return self.transaction_pool.add_transaction(transaction)

    def create_transaction(
        self,
        sender: str,
        recipient: str,
        amount: float,
        tx_type: TransactionType = TransactionType.TRANSFER,
        data: Dict[str, Any] = None,
    ) -> Optional[Transaction]:
        """Crea una nueva transacción"""
        try:
            # Generar ID único
            tx_id = hashlib.sha256(f"{sender}{recipient}{amount}{time.time()}".encode()).hexdigest()

            transaction = Transaction(
                tx_id=tx_id,
                tx_type=tx_type,
                sender=sender,
                recipient=recipient,
                amount=amount,
                data=data or {},
                timestamp=time.time(),
                nonce=self._get_next_nonce(sender),
                gas_limit=100000,
                gas_price=0.001,
                signature="",
                public_key=""
            )

            # Firmar transacción (requiere claves del sender)
            if sender in self.crypto_manager.key_pairs:
                tx_data = self.transaction_pool._get_transaction_data_for_signing(transaction)
                transaction.signature = self.crypto_manager.sign_data(sender, tx_data.encode('utf-8'))
                transaction.public_key = self.crypto_manager.key_pairs[sender]['public_pem']

            return transaction

        except Exception as e:
            logger.error(f"❌ Error creando transacción: {e}")
            return None

    def _get_next_nonce(self, sender: str) -> int:
        """Obtiene siguiente nonce para un sender"""
        # Contar transacciones del sender en la blockchain
        nonce = 0
        for block in self.chain:
            for tx in block.transactions:
                if tx.sender == sender:
                    nonce = max(nonce, tx.nonce + 1)

        return nonce

    async def mine_block(self) -> Optional[Block]:
        """Mina un nuevo bloque"""
        try:
            if not self.chain:
                logger.error("❌ No hay bloque génesis")
                return None

            # Seleccionar validador
            validator_id = self.pos_validator.select_validator(len(self.chain))
            if not validator_id:
                logger.warning("⚠️ No hay validadores disponibles")
                return None

            # Obtener transacciones del pool
            transactions = self.transaction_pool.get_transactions_for_block(100)

            if not transactions:
                logger.debug("📭 No hay transacciones para minar")
                return None

            # Crear nuevo bloque
            previous_block = self.chain[-1]
            merkle_root = self.crypto_manager.create_merkle_tree(transactions)

            new_block = Block(
                block_id=hashlib.sha256(f"{len(self.chain)}{time.time()}".encode()).hexdigest(),
                index=len(self.chain),
                previous_hash=previous_block.block_hash,
                merkle_root=merkle_root,
                timestamp=time.time(),
                transactions=transactions,
                validator=validator_id,
                stake_weight=self.pos_validator.validators[validator_id].stake_amount,
                nonce=0,
                difficulty=self.current_difficulty,
                block_hash="",
                status=BlockStatus.MINING,
                confirmations=0
            )

            # Calcular hash del bloque
            new_block.block_hash = self._calculate_block_hash(new_block)

            # Validar bloque
            if self.pos_validator.validate_block(validator_id, new_block):
                new_block.status = BlockStatus.VALIDATED

                # Agregar a la cadena
                self.chain.append(new_block)

                # Guardar en base de datos
                self._save_block_to_db(new_block)

                # Recompensar validador
                reward = self._calculate_block_reward(new_block)
                self.pos_validator.reward_validator(validator_id, reward)

                logger.info(f"⛏️ Bloque minado: {new_block.index} por {validator_id}")
                return new_block
            else:
                logger.warning("❌ Bloque inválido")
                return None

        except Exception as e:
            logger.error(f"❌ Error minando bloque: {e}")
            return None

    def _calculate_block_hash(self, block: Block) -> str:
        """Calcula hash del bloque"""
        block_data = {
            'index': block.index,
            'previous_hash': block.previous_hash,
            'merkle_root': block.merkle_root,
            'timestamp': block.timestamp,
            'validator': block.validator,
            'nonce': block.nonce
        }

        return hashlib.sha256(json.dumps(block_data, sort_keys=True).encode()).hexdigest()

    def _calculate_block_reward(self, block: Block) -> float:
        """Calcula recompensa por bloque"""
        base_reward = 10.0
        tx_fees = sum(tx.gas_price * tx.gas_limit for tx in block.transactions)
        return base_reward + tx_fees

    def _save_block_to_db(self, block: Block):
        """Guarda bloque en base de datos"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Serializar bloque
            block_data = pickle.dumps(block)

            cursor.execute('''
                INSERT OR REPLACE INTO blocks
                (block_id, index_num, previous_hash, merkle_root, timestamp, validator, block_hash, block_data)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                block.block_id, block.index, block.previous_hash, block.merkle_root,
                block.timestamp, block.validator, block.block_hash, block_data
            ))

            # Guardar transacciones
            for tx in block.transactions:
                tx_data = pickle.dumps(tx)
                cursor.execute('''
                    INSERT OR REPLACE INTO transactions
                    (tx_id, block_id, tx_type, sender, recipient, amount, timestamp, tx_data)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    tx.tx_id, block.block_id, tx.tx_type.value, tx.sender,
                    tx.recipient, tx.amount, tx.timestamp, tx_data
                ))

            conn.commit()
            conn.close()

        except Exception as e:
            logger.error(f"❌ Error guardando bloque en DB: {e}")

    def get_balance(self, address: str) -> float:
        """Obtiene balance de una dirección"""
        balance = 0.0

        for block in self.chain:
            for tx in block.transactions:
                if tx.recipient == address:
                    balance += tx.amount
                if tx.sender == address:
                    balance -= tx.amount

        return balance

    def get_blockchain_info(self) -> Dict[str, Any]:
        """Obtiene información de la blockchain"""
        return {
            "node_id": self.node_id,
            "chain_length": len(self.chain),
            "latest_block": self.chain[-1].block_id if self.chain else None,
            "pending_transactions": self.transaction_pool.get_pending_count(),
            "total_validators": len(self.pos_validator.validators),
            "active_contracts": len(self.smart_contract_engine.contracts),
            "current_difficulty": self.current_difficulty
        }

    def register_ai_model(
        self,
        creator: str,
        model_name: str,
        model_hash: str,
        metadata: Dict[str, Any],
    ) -> Optional[str]:
        """Registra modelo de IA en la blockchain"""
        try:
            model_id = hashlib.sha256(f"{creator}{model_name}{time.time()}".encode()).hexdigest()

            ai_model = AIModel(
                model_id=model_id,
                name=model_name,
                version="1.0.0",
                creator=creator,
                model_hash=model_hash,
                metadata=metadata,
                training_data_hash=metadata.get('training_data_hash', ''),
                performance_metrics=metadata.get('performance_metrics', {}),
                validation_results=metadata.get('validation_results', {}),
                deployment_status='registered',
                created_at=time.time()
            )

            # Crear transacción de registro
            tx_data = {
                "type": "ai_model_registration",
                "model": asdict(ai_model)
            }

            transaction = self.create_transaction(
                sender=creator,
                recipient="ai_registry",
                amount=0.0,
                tx_type=TransactionType.MODEL_REGISTRATION,
                data=tx_data
            )

            if transaction and self.add_transaction(transaction):
                logger.info(f"🤖 Modelo IA registrado: {model_name} ({model_id[:8]}...)")
                return model_id

            return None

        except Exception as e:
            logger.error(f"❌ Error registrando modelo IA: {e}")
            return None

# Función principal para testing


async def main():
    """Función principal para pruebas"""
    # Crear blockchain
    blockchain = BlockchainCore("test_node", "validator_1")

    # Generar claves para nodos
    blockchain.crypto_manager.generate_key_pair("validator_1")
    blockchain.crypto_manager.generate_key_pair("user_1")

    # Registrar validador
    blockchain.pos_validator.register_validator("validator_1", 10000.0)

    try:
        # Crear algunas transacciones
        tx1 = blockchain.create_transaction("validator_1", "user_1", 100.0)
        if tx1:
            blockchain.add_transaction(tx1)

        tx2 = blockchain.create_transaction("validator_1", "user_1", 50.0)
        if tx2:
            blockchain.add_transaction(tx2)

        # Minar bloque
        block = await blockchain.mine_block()
        if block:
            print(f"✅ Bloque minado: {block.index}")

        # Desplegar contrato inteligente
        contract_code = '''
def transfer(sender, recipient, amount):
    if storage.get(sender, 0) >= amount:
        storage[sender] = storage.get(sender, 0) - amount
        storage[recipient] = storage.get(recipient, 0) + amount
        return True
    return False

def get_balance(address):
    return storage.get(address, 0)
'''

        contract_id = blockchain.smart_contract_engine.deploy_contract(
            "validator_1", "SimpleToken", contract_code, 1000.0
        )

        if contract_id:
            print(f"📜 Contrato desplegado: {contract_id[:8]}...")

        # Registrar modelo IA
        model_id = blockchain.register_ai_model(
            "validator_1",
            "AEGIS_Model_v1",
            "model_hash_123",
            {
                "description": "Modelo de consenso distribuido",
                "performance_metrics": {"accuracy": 0.95, "f1_score": 0.92}
            }
        )

        if model_id:
            print(f"🤖 Modelo IA registrado: {model_id[:8]}...")

        # Mostrar información de blockchain
        info = blockchain.get_blockchain_info()
        print("\n🔗 Información de Blockchain:")
        print(json.dumps(info, indent=2))

    except Exception as e:
        logger.error(f"❌ Error en prueba: {e}")

if __name__ == "__main__":
    asyncio.run(main())
