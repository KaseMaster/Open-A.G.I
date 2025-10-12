#!/usr/bin/env python3
"""
Protocolo de Consenso Híbrido para IA Distribuida
Implementación de Proof of Computation + Practical Byzantine Fault Tolerance

AEGIS Security Framework - Uso Ético Únicamente
"""

import asyncio
import hashlib
import json
import time
import secrets
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Dict, List, Optional, Any, Callable
import logging
from collections import defaultdict
from cryptography.hazmat.primitives.asymmetric import ed25519

# Configuración de logging temprana
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# numpy es opcional para permitir pruebas en entornos mínimos
try:
    import numpy as np  # type: ignore
    HAS_NUMPY = True
except Exception:
    np = None  # type: ignore
    HAS_NUMPY = False
    logger.warning("numpy no disponible; algunas funciones de Proof of Computation quedarán deshabilitadas.")

# (logger ya configurado arriba)


class ConsensusState(Enum):
    """Estados del proceso de consenso"""
    IDLE = "idle"
    PROPOSING = "proposing"
    PREPARING = "preparing"
    COMMITTING = "committing"
    FINALIZING = "finalizing"
    COMPLETED = "completed"
    FAILED = "failed"


class MessageType(Enum):
    """Tipos de mensajes en el protocolo"""
    PROPOSAL = "proposal"
    PREPARE = "prepare"
    COMMIT = "commit"
    VIEW_CHANGE = "view_change"
    POC_CHALLENGE = "poc_challenge"
    POC_RESPONSE = "poc_response"
    HEARTBEAT = "heartbeat"


class NodeRole(Enum):
    """Roles de los nodos en el consenso"""
    LEADER = "leader"
    VALIDATOR = "validator"
    OBSERVER = "observer"


@dataclass
class ComputationChallenge:
    """Desafío computacional para Proof of Computation"""
    challenge_id: str
    problem_type: str
    parameters: Dict[str, Any]
    difficulty: int
    timeout: float
    expected_operations: int


@dataclass
class ComputationProof:
    """Prueba de computación realizada"""
    challenge_id: str
    node_id: str
    solution: Any
    computation_time: float
    operations_performed: int
    verification_hash: str
    timestamp: float


@dataclass
class ConsensusMessage:
    """Mensaje del protocolo de consenso"""
    message_type: MessageType
    sender_id: str
    view_number: int
    sequence_number: int
    payload: Dict[str, Any]
    timestamp: float
    signature: Optional[bytes] = None


@dataclass
class NodeReputation:
    """Reputación de un nodo en la red"""
    node_id: str
    computation_score: float
    reliability_score: float
    response_time_avg: float
    successful_validations: int
    failed_validations: int
    last_activity: float
    total_contributions: int


class ProofOfComputation:
    """Implementación de Proof of Computation"""

    def __init__(self):
        self.active_challenges: Dict[str, ComputationChallenge] = {}
        self.completed_proofs: Dict[str, List[ComputationProof]] = {}
        self.difficulty_adjustment_factor = 1.0
        self.target_solve_time = 30.0  # segundos

    def generate_ml_challenge(self, difficulty: int) -> ComputationChallenge:
        """Genera un desafío de Machine Learning verificable"""
        challenge_id = secrets.token_hex(16)

        if not HAS_NUMPY:
            # Fallback mínimo cuando numpy no está disponible: desafío trivial
            challenge = ComputationChallenge(
                challenge_id=challenge_id,
                problem_type="noop",
                parameters={},
                difficulty=1,
                timeout=5.0,
                expected_operations=1,
            )
            self.active_challenges[challenge_id] = challenge
            return challenge

        # Generar problema de regresión lineal con datos sintéticos
        n_samples = 100 + (difficulty * 50)
        n_features = 5 + (difficulty * 2)

        # Semilla determinística basada en el challenge_id para reproducibilidad
        seed = int(hashlib.sha256(challenge_id.encode()).hexdigest()[:8], 16)
        np.random.seed(seed)

        # Generar datos sintéticos
        X = np.random.randn(n_samples, n_features)
        true_coefficients = np.random.randn(n_features)
        noise = np.random.randn(n_samples) * 0.1
        y = X @ true_coefficients + noise

        challenge = ComputationChallenge(
            challenge_id=challenge_id,
            problem_type="linear_regression",
            parameters={
                "X_hash": hashlib.sha256(X.tobytes()).hexdigest(),
                "y_hash": hashlib.sha256(y.tobytes()).hexdigest(),
                "n_samples": n_samples,
                "n_features": n_features,
                "seed": seed
            },
            difficulty=difficulty,
            timeout=60.0 + (difficulty * 30),
            expected_operations=n_samples * n_features * 10
        )

        self.active_challenges[challenge_id] = challenge
        return challenge

    def solve_challenge(self, challenge: ComputationChallenge, node_id: str) -> Optional[ComputationProof]:
        """Resuelve un desafío computacional"""
        start_time = time.time()

        try:
            if challenge.problem_type == "noop":
                solution = {"result": "ok"}
                computation_time = time.time() - start_time
                verification_hash = hashlib.sha256(
                    f"{challenge.challenge_id}{node_id}noop".encode()
                ).hexdigest()
                proof = ComputationProof(
                    challenge_id=challenge.challenge_id,
                    node_id=node_id,
                    solution=solution,
                    computation_time=computation_time,
                    operations_performed=1,
                    verification_hash=verification_hash,
                    timestamp=time.time(),
                )
                return proof
            if challenge.problem_type == "linear_regression":
                # Recrear los datos usando la misma semilla
                seed = challenge.parameters["seed"]
                np.random.seed(seed)

                n_samples = challenge.parameters["n_samples"]
                n_features = challenge.parameters["n_features"]

                X = np.random.randn(n_samples, n_features)
                true_coefficients = np.random.randn(n_features)
                noise = np.random.randn(n_samples) * 0.1
                y = X @ true_coefficients + noise

                # Verificar integridad de datos
                X_hash = hashlib.sha256(X.tobytes()).hexdigest()
                y_hash = hashlib.sha256(y.tobytes()).hexdigest()

                if (
                    X_hash != challenge.parameters["X_hash"]
                    or y_hash != challenge.parameters["y_hash"]
                ):
                    logger.error("Datos del desafío no coinciden")
                    return None

                # Resolver regresión lineal usando mínimos cuadrados
                # β = (X^T X)^(-1) X^T y
                XtX = X.T @ X
                Xty = X.T @ y
                coefficients = np.linalg.solve(XtX, Xty)

                # Calcular métricas de calidad
                predictions = X @ coefficients
                mse = np.mean((y - predictions) ** 2)

                solution = {
                    "coefficients": coefficients.tolist(),
                    "mse": float(mse),
                    "method": "least_squares"
                }

                computation_time = time.time() - start_time
                operations_performed = n_samples * n_features * 10  # Estimación

                # Crear hash de verificación
                solution_str = json.dumps(solution, sort_keys=True)
                verification_hash = hashlib.sha256(
                    f"{challenge.challenge_id}{node_id}{solution_str}".encode()
                ).hexdigest()

                proof = ComputationProof(
                    challenge_id=challenge.challenge_id,
                    node_id=node_id,
                    solution=solution,
                    computation_time=computation_time,
                    operations_performed=operations_performed,
                    verification_hash=verification_hash,
                    timestamp=time.time()
                )

                return proof

        except Exception as e:
            logger.error(f"Error resolviendo desafío {challenge.challenge_id}: {e}")
            return None

    def verify_proof(self, proof: ComputationProof, challenge: ComputationChallenge) -> bool:
        """Verifica una prueba de computación"""
        try:
            if challenge.problem_type == "linear_regression":
                # Recrear datos y verificar solución
                seed = challenge.parameters["seed"]
                np.random.seed(seed)

                n_samples = challenge.parameters["n_samples"]
                n_features = challenge.parameters["n_features"]

                X = np.random.randn(n_samples, n_features)
                true_coefficients = np.random.randn(n_features)
                noise = np.random.randn(n_samples) * 0.1
                y = X @ true_coefficients + noise

                # Verificar solución
                coefficients = np.array(proof.solution["coefficients"])
                predictions = X @ coefficients
                calculated_mse = np.mean((y - predictions) ** 2)

                # Tolerancia para errores de punto flotante
                mse_diff = abs(calculated_mse - proof.solution["mse"])
                if mse_diff > 1e-10:
                    logger.warning(f"MSE no coincide: {calculated_mse} vs {proof.solution['mse']}")
                    return False

                # Verificar hash
                solution_str = json.dumps(proof.solution, sort_keys=True)
                expected_hash = hashlib.sha256(
                    f"{proof.challenge_id}{proof.node_id}{solution_str}".encode()
                ).hexdigest()

                if expected_hash != proof.verification_hash:
                    logger.warning("Hash de verificación no coincide")
                    return False

                # Verificar tiempo de computación razonable
                if proof.computation_time > challenge.timeout:
                    logger.warning("Tiempo de computación excede el límite")
                    return False

                return True

        except Exception as e:
            logger.error(f"Error verificando prueba: {e}")
            return False

    def calculate_computation_score(self, proof: ComputationProof, challenge: ComputationChallenge) -> float:
        """Calcula el puntaje de computación basado en la prueba"""
        base_score = 100.0

        # Penalizar por tiempo excesivo
        time_factor = min(1.0, challenge.timeout / max(proof.computation_time, 0.1))

        # Bonificar por eficiencia en operaciones
        ops_factor = min(1.0, proof.operations_performed / challenge.expected_operations)

        # Bonificar por dificultad
        difficulty_bonus = 1.0 + (challenge.difficulty * 0.1)

        score = base_score * time_factor * ops_factor * difficulty_bonus
        return max(0.0, min(1000.0, score))  # Limitar entre 0 y 1000


class PBFTConsensus:
    """Implementación de Practical Byzantine Fault Tolerance"""

    def __init__(self, node_id: str, private_key: ed25519.Ed25519PrivateKey):
        self.node_id = node_id
        self.private_key = private_key
        self.public_key = private_key.public_key()

        # Estado del consenso
        self.view_number = 0
        self.sequence_number = 0
        self.state = ConsensusState.IDLE
        self.current_proposal = None

        # Nodos conocidos
        self.known_nodes: Dict[str, ed25519.Ed25519PublicKey] = {}
        self.node_reputations: Dict[str, NodeReputation] = {}
        self.byzantine_threshold = 0  # f < n/3

        # Mensajes recibidos
        self.prepare_messages: Dict[int, Dict[str, ConsensusMessage]] = defaultdict(dict)
        self.commit_messages: Dict[int, Dict[str, ConsensusMessage]] = defaultdict(dict)

        # Callbacks
        self.message_handlers: Dict[MessageType, Callable] = {
            MessageType.PROPOSAL: self._handle_proposal,
            MessageType.PREPARE: self._handle_prepare,
            MessageType.COMMIT: self._handle_commit,
            MessageType.VIEW_CHANGE: self._handle_view_change
        }

    def add_node(self, node_id: str, public_key: ed25519.Ed25519PublicKey) -> None:
        """Añade un nodo conocido a la red"""
        self.known_nodes[node_id] = public_key

        # Inicializar reputación
        if node_id not in self.node_reputations:
            self.node_reputations[node_id] = NodeReputation(
                node_id=node_id,
                computation_score=100.0,
                reliability_score=100.0,
                response_time_avg=1.0,
                successful_validations=0,
                failed_validations=0,
                last_activity=time.time(),
                total_contributions=0
            )

        # Actualizar umbral bizantino
        n = len(self.known_nodes)
        self.byzantine_threshold = (n - 1) // 3

    def sign_message(self, message: ConsensusMessage) -> bytes:
        """Firma un mensaje con la clave privada del nodo"""
        message_dict = asdict(message)
        message_dict.pop('signature', None)  # Remover firma existente
        message_bytes = json.dumps(message_dict, sort_keys=True).encode()
        signature = self.private_key.sign(message_bytes)
        return signature

    def verify_message(self, message: ConsensusMessage) -> bool:
        """Verifica la firma de un mensaje"""
        if message.sender_id not in self.known_nodes:
            logger.warning(f"Nodo desconocido: {message.sender_id}")
            return False

        if not message.signature:
            logger.warning("Mensaje sin firma")
            return False

        try:
            public_key = self.known_nodes[message.sender_id]
            message_dict = asdict(message)
            message_dict.pop('signature')
            message_bytes = json.dumps(message_dict, sort_keys=True).encode()

            public_key.verify(message.signature, message_bytes)
            return True

        except Exception as e:
            logger.warning(f"Error verificando firma: {e}")
            return False

    def is_leader(self, view_number: int) -> bool:
        """Determina si este nodo es el líder para la vista actual"""
        if not self.known_nodes:
            return False

        # Selección determinística del líder basada en reputación y vista
        nodes_by_reputation = sorted(
            self.node_reputations.items(),
            key=lambda x: (x[1].computation_score + x[1].reliability_score, x[0])
        )

        leader_index = view_number % len(nodes_by_reputation)
        leader_id = nodes_by_reputation[leader_index][0]

        return leader_id == self.node_id

    async def propose_change(self, change_data: Dict[str, Any]) -> bool:
        """Propone un cambio en el estado del sistema"""
        if not self.is_leader(self.view_number):
            logger.warning("Solo el líder puede proponer cambios")
            return False

        if self.state != ConsensusState.IDLE:
            logger.warning("Consenso ya en progreso")
            return False

        self.state = ConsensusState.PROPOSING
        self.sequence_number += 1

        proposal_message = ConsensusMessage(
            message_type=MessageType.PROPOSAL,
            sender_id=self.node_id,
            view_number=self.view_number,
            sequence_number=self.sequence_number,
            payload={
                "change_data": change_data,
                "timestamp": time.time()
            },
            timestamp=time.time()
        )

        proposal_message.signature = self.sign_message(proposal_message)
        self.current_proposal = proposal_message

        # Broadcast a todos los nodos
        await self._broadcast_message(proposal_message)

        logger.info(f"Propuesta enviada: seq={self.sequence_number}")
        return True

    async def _handle_proposal(self, message: ConsensusMessage) -> None:
        """Maneja una propuesta recibida"""
        if not self.verify_message(message):
            logger.warning("Propuesta con firma inválida")
            return

        if message.view_number != self.view_number:
            logger.warning(f"Propuesta de vista incorrecta: {message.view_number} vs {self.view_number}")
            return

        if self.state != ConsensusState.IDLE:
            logger.warning("Ya hay consenso en progreso")
            return

        # Validar la propuesta
        if await self._validate_proposal(message):
            self.state = ConsensusState.PREPARING
            self.current_proposal = message

            # Enviar mensaje PREPARE
            prepare_message = ConsensusMessage(
                message_type=MessageType.PREPARE,
                sender_id=self.node_id,
                view_number=self.view_number,
                sequence_number=message.sequence_number,
                payload={"proposal_hash": self._hash_message(message)},
                timestamp=time.time()
            )

            prepare_message.signature = self.sign_message(prepare_message)
            await self._broadcast_message(prepare_message)

            logger.info(f"PREPARE enviado para seq={message.sequence_number}")

    async def _handle_prepare(self, message: ConsensusMessage) -> None:
        """Maneja un mensaje PREPARE"""
        if not self.verify_message(message):
            return

        if message.view_number != self.view_number:
            return

        seq_num = message.sequence_number
        self.prepare_messages[seq_num][message.sender_id] = message

        # Verificar si tenemos suficientes PREPAREs (2f + 1)
        required_prepares = 2 * self.byzantine_threshold + 1
        if len(self.prepare_messages[seq_num]) >= required_prepares:

            if self.state == ConsensusState.PREPARING:
                self.state = ConsensusState.COMMITTING

                # Enviar mensaje COMMIT
                commit_message = ConsensusMessage(
                    message_type=MessageType.COMMIT,
                    sender_id=self.node_id,
                    view_number=self.view_number,
                    sequence_number=seq_num,
                    payload={"proposal_hash": message.payload["proposal_hash"]},
                    timestamp=time.time()
                )

                commit_message.signature = self.sign_message(commit_message)
                await self._broadcast_message(commit_message)

                logger.info(f"COMMIT enviado para seq={seq_num}")

    async def _handle_commit(self, message: ConsensusMessage) -> None:
        """Maneja un mensaje COMMIT"""
        if not self.verify_message(message):
            return

        if message.view_number != self.view_number:
            return

        seq_num = message.sequence_number
        self.commit_messages[seq_num][message.sender_id] = message

        # Verificar si tenemos suficientes COMMITs (2f + 1)
        required_commits = 2 * self.byzantine_threshold + 1
        if len(self.commit_messages[seq_num]) >= required_commits:

            if self.state == ConsensusState.COMMITTING:
                self.state = ConsensusState.FINALIZING

                # Aplicar el cambio
                if self.current_proposal:
                    await self._apply_change(self.current_proposal.payload["change_data"])

                # Limpiar estado
                self._cleanup_consensus_state(seq_num)
                self.state = ConsensusState.COMPLETED

                logger.info(f"Consenso completado para seq={seq_num}")

                # Volver a estado IDLE después de un breve delay
                await asyncio.sleep(0.1)
                self.state = ConsensusState.IDLE

    async def _handle_view_change(self, message: ConsensusMessage) -> None:
        """Maneja un cambio de vista (para tolerancia a fallos del líder)"""
        # Implementación simplificada del cambio de vista
        if not self.verify_message(message):
            return

        new_view = message.payload.get("new_view", self.view_number + 1)
        if new_view > self.view_number:
            logger.info(f"Cambiando a vista {new_view}")
            self.view_number = new_view
            self.state = ConsensusState.IDLE
            self._cleanup_all_consensus_state()

    async def _validate_proposal(self, proposal: ConsensusMessage) -> bool:
        """Valida una propuesta antes de aceptarla"""
        try:
            change_data = proposal.payload["change_data"]

            # Validaciones básicas
            if not isinstance(change_data, dict):
                return False

            # Validar timestamp (no muy antiguo ni futuro)
            timestamp = proposal.payload.get("timestamp", 0)
            current_time = time.time()
            if abs(current_time - timestamp) > 300:  # 5 minutos
                logger.warning("Propuesta con timestamp inválido")
                return False

            # Validaciones específicas del dominio
            change_type = change_data.get("type")
            if change_type == "knowledge_update":
                return await self._validate_knowledge_update(change_data)
            elif change_type == "node_reputation_update":
                return await self._validate_reputation_update(change_data)

            return True

        except Exception as e:
            logger.error(f"Error validando propuesta: {e}")
            return False

    async def _validate_knowledge_update(self, change_data: Dict[str, Any]) -> bool:
        """Valida una actualización de conocimiento"""
        required_fields = ["content_hash", "source_node", "timestamp"]
        return all(field in change_data for field in required_fields)

    async def _validate_reputation_update(self, change_data: Dict[str, Any]) -> bool:
        """Valida una actualización de reputación"""
        required_fields = ["node_id", "score_delta", "reason"]
        return all(field in change_data for field in required_fields)

    async def _apply_change(self, change_data: Dict[str, Any]) -> None:
        """Aplica un cambio consensuado al estado del sistema"""
        try:
            change_type = change_data.get("type")

            if change_type == "knowledge_update":
                await self._apply_knowledge_update(change_data)
            elif change_type == "node_reputation_update":
                await self._apply_reputation_update(change_data)

            logger.info(f"Cambio aplicado: {change_type}")

        except Exception as e:
            logger.error(f"Error aplicando cambio: {e}")

    async def _apply_knowledge_update(self, change_data: Dict[str, Any]) -> None:
        """Aplica una actualización de conocimiento"""
        # Implementación específica para actualizar la base de conocimiento
        source_node = change_data["source_node"]

        # Actualizar reputación del nodo contribuyente
        if source_node in self.node_reputations:
            self.node_reputations[source_node].total_contributions += 1
            self.node_reputations[source_node].last_activity = time.time()

    async def _apply_reputation_update(self, change_data: Dict[str, Any]) -> None:
        """Aplica una actualización de reputación"""
        node_id = change_data["node_id"]
        score_delta = change_data["score_delta"]

        if node_id in self.node_reputations:
            reputation = self.node_reputations[node_id]
            reputation.computation_score = max(0, reputation.computation_score + score_delta)

    def _hash_message(self, message: ConsensusMessage) -> str:
        """Calcula el hash de un mensaje"""
        message_dict = asdict(message)
        message_dict.pop('signature', None)
        message_str = json.dumps(message_dict, sort_keys=True)
        return hashlib.sha256(message_str.encode()).hexdigest()

    def _cleanup_consensus_state(self, sequence_number: int) -> None:
        """Limpia el estado de consenso para un número de secuencia"""
        self.prepare_messages.pop(sequence_number, None)
        self.commit_messages.pop(sequence_number, None)

    def _cleanup_all_consensus_state(self) -> None:
        """Limpia todo el estado de consenso"""
        self.prepare_messages.clear()
        self.commit_messages.clear()
        self.current_proposal = None

    async def _broadcast_message(self, message: ConsensusMessage) -> None:
        """Broadcast de un mensaje a todos los nodos conocidos"""
        # Esta función debe ser implementada por la capa de red
        # Por ahora es un placeholder
        logger.debug(f"Broadcasting {message.message_type.value} to {len(self.known_nodes)} nodes")


class HybridConsensus:
    """Consenso híbrido que combina PoC y PBFT"""

    def __init__(self, node_id: str, private_key: ed25519.Ed25519PrivateKey):
        self.node_id = node_id
        self.poc = ProofOfComputation()
        self.pbft = PBFTConsensus(node_id, private_key)

        # Configuración del consenso híbrido
        self.poc_interval = 300  # 5 minutos entre desafíos PoC
        self.last_poc_challenge = 0
        self.min_computation_score = 50.0  # Puntaje mínimo para participar en PBFT

    async def start_consensus_round(self) -> None:
        """Inicia una ronda de consenso híbrido"""
        current_time = time.time()

        # Verificar si es tiempo de un desafío PoC
        if current_time - self.last_poc_challenge > self.poc_interval:
            await self._run_poc_round()
            self.last_poc_challenge = current_time

        # Ejecutar PBFT para cambios pendientes
        await self._run_pbft_round()

    async def _run_poc_round(self) -> None:
        """Ejecuta una ronda de Proof of Computation"""
        logger.info("Iniciando ronda de Proof of Computation")

        # Generar desafío
        difficulty = self._calculate_difficulty()
        challenge = self.poc.generate_ml_challenge(difficulty)

        # Resolver desafío localmente
        proof = self.poc.solve_challenge(challenge, self.node_id)

        if proof and self.poc.verify_proof(proof, challenge):
            # Calcular puntaje
            score = self.poc.calculate_computation_score(proof, challenge)

            # Actualizar reputación local
            if self.node_id in self.pbft.node_reputations:
                reputation = self.pbft.node_reputations[self.node_id]
                reputation.computation_score = (reputation.computation_score * 0.9) + (score * 0.1)

            logger.info(f"PoC completado: score={score:.2f}")
        else:
            logger.warning("Falló la prueba de computación")

    async def _run_pbft_round(self) -> None:
        """Ejecuta una ronda de PBFT si hay cambios pendientes"""
        # Verificar si este nodo puede participar en PBFT
        if self.node_id in self.pbft.node_reputations:
            reputation = self.pbft.node_reputations[self.node_id]
            if reputation.computation_score < self.min_computation_score:
                logger.info("Puntaje de computación insuficiente para PBFT")
                return

        # Simular propuesta de cambio (en implementación real vendría de la aplicación)
        if self.pbft.is_leader(self.pbft.view_number) and self.pbft.state == ConsensusState.IDLE:
            sample_change = {
                "type": "knowledge_update",
                "content_hash": secrets.token_hex(32),
                "source_node": self.node_id,
                "timestamp": time.time()
            }

            await self.pbft.propose_change(sample_change)

    def _calculate_difficulty(self) -> int:
        """Calcula la dificultad del próximo desafío PoC"""
        # Ajustar dificultad basado en el rendimiento de la red
        base_difficulty = 3

        # Considerar el número de nodos activos
        active_nodes = len([r for r in self.pbft.node_reputations.values()
                           if time.time() - r.last_activity < 600])

        if active_nodes > 10:
            return base_difficulty + 1
        elif active_nodes < 5:
            return max(1, base_difficulty - 1)

        return base_difficulty

    def get_network_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas de la red de consenso"""
        current_time = time.time()
        active_nodes = [
            r for r in self.pbft.node_reputations.values()
            if current_time - r.last_activity < 600
        ]

        return {
            "total_nodes": len(self.pbft.node_reputations),
            "active_nodes": len(active_nodes),
            "byzantine_threshold": self.pbft.byzantine_threshold,
            "current_view": self.pbft.view_number,
            "consensus_state": self.pbft.state.value,
            "avg_computation_score": np.mean([r.computation_score for r in active_nodes]) if active_nodes else 0,
            "avg_reliability_score": np.mean([r.reliability_score for r in active_nodes]) if active_nodes else 0,
            "is_leader": self.pbft.is_leader(self.pbft.view_number)
        }

# Ejemplo de uso


async def main():
    """Ejemplo de uso del consenso híbrido"""
    # Generar identidad del nodo
    private_key = ed25519.Ed25519PrivateKey.generate()
    node_id = secrets.token_hex(16)

    # Crear consenso híbrido
    consensus = HybridConsensus(node_id, private_key)

    # Simular otros nodos
    for i in range(5):
        other_key = ed25519.Ed25519PrivateKey.generate()
        other_id = f"node_{i}"
        consensus.pbft.add_node(other_id, other_key.public_key())

    # Ejecutar rondas de consenso
    for round_num in range(3):
        logger.info(f"=== Ronda de Consenso {round_num + 1} ===")
        await consensus.start_consensus_round()

        # Mostrar estadísticas
        stats = consensus.get_network_stats()
        logger.info(f"Estadísticas: {stats}")

        await asyncio.sleep(2)

if __name__ == "__main__":
    asyncio.run(main())
