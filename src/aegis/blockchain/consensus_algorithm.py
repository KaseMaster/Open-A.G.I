#!/usr/bin/env python3
"""
Algoritmo de Consenso Distribuido - AEGIS Framework
Implementación híbrida de consenso bizantino tolerante a fallos con optimizaciones para IA distribuida.

Características principales:
- Consenso PBFT (Practical Byzantine Fault Tolerance) optimizado
- Mecanismo de votación ponderada por reputación
- Consenso rápido para operaciones no críticas
- Detección y mitigación de comportamiento bizantino
- Sincronización de estado distribuido
"""

import asyncio
import time
import json
import hashlib
import logging
from typing import Dict, List, Set, Optional, Tuple, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict, Counter

# Configuración de logging temprana para usar logger en imports opcionales
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# aiohttp es opcional para permitir pruebas en entornos mínimos (p.ej. Windows CI)
try:
    import aiohttp  # type: ignore
    HAS_AIOHTTP = True
except Exception:
    aiohttp = None  # type: ignore
    HAS_AIOHTTP = False
    logger.warning("aiohttp no disponible; algunas características de red se deshabilitarán en este entorno.")
from datetime import datetime, timedelta
import random

# (ya configurado arriba)

class ConsensusPhase(Enum):
    """Fases del algoritmo de consenso"""
    PREPARE = "prepare"
    PROMISE = "promise"
    PROPOSE = "propose"
    ACCEPT = "accept"
    COMMIT = "commit"
    ABORT = "abort"

class MessageType(Enum):
    """Tipos de mensajes de consenso"""
    PROPOSAL = "proposal"
    PREPARE = "prepare"
    PROMISE = "promise"
    ACCEPT = "accept"
    COMMIT = "commit"
    ABORT = "abort"
    HEARTBEAT = "heartbeat"
    VIEW_CHANGE = "view_change"

class ProposalStatus(Enum):
    """Estados de una propuesta"""
    PENDING = "pending"
    PREPARING = "preparing"
    PROMISED = "promised"
    ACCEPTED = "accepted"
    COMMITTED = "committed"
    ABORTED = "aborted"

@dataclass
class ConsensusMessage:
    """Mensaje de consenso"""
    message_id: str
    message_type: MessageType
    sender_id: str
    proposal_id: str
    view_number: int
    sequence_number: int
    data: Dict[str, Any]
    timestamp: float
    signature: str = ""

@dataclass
class Proposal:
    """Propuesta para consenso"""
    proposal_id: str
    proposer_id: str
    data: Dict[str, Any]
    timestamp: float
    view_number: int
    sequence_number: int
    status: ProposalStatus
    votes: Dict[str, bool] = None  # node_id -> vote
    signatures: List[str] = None
    priority: int = 1  # 1=low, 2=medium, 3=high, 4=critical
    
    def __post_init__(self):
        if self.votes is None:
            self.votes = {}
        if self.signatures is None:
            self.signatures = []

@dataclass
class NodeState:
    """Estado de un nodo en el consenso"""
    node_id: str
    view_number: int
    sequence_number: int
    is_leader: bool
    reputation_score: float
    last_activity: float
    byzantine_score: float = 0.0  # Puntuación de comportamiento sospechoso
    vote_weight: float = 1.0

class ByzantineDetector:
    """Detector de comportamiento bizantino"""
    
    def __init__(self, threshold: float = 0.7):
        self.threshold = threshold
        self.behavior_history: Dict[str, List[Dict]] = defaultdict(list)
        self.reputation_scores: Dict[str, float] = defaultdict(lambda: 1.0)
        
    def record_behavior(self, node_id: str, behavior_type: str, details: Dict[str, Any]):
        """Registra comportamiento de un nodo"""
        behavior_record = {
            "type": behavior_type,
            "timestamp": time.time(),
            "details": details
        }
        
        self.behavior_history[node_id].append(behavior_record)
        
        # Mantener solo los últimos 100 registros
        if len(self.behavior_history[node_id]) > 100:
            self.behavior_history[node_id] = self.behavior_history[node_id][-100:]
        
        # Actualizar puntuación de reputación
        self._update_reputation_score(node_id)
    
    def _update_reputation_score(self, node_id: str):
        """Actualiza la puntuación de reputación de un nodo"""
        recent_behaviors = [
            b for b in self.behavior_history[node_id]
            if time.time() - b["timestamp"] < 3600  # Última hora
        ]
        
        if not recent_behaviors:
            return
        
        # Contar comportamientos positivos y negativos
        positive_count = sum(1 for b in recent_behaviors if b["type"] in ["correct_vote", "timely_response", "valid_proposal"])
        negative_count = sum(1 for b in recent_behaviors if b["type"] in ["incorrect_vote", "timeout", "invalid_proposal", "conflicting_messages"])
        
        total_count = len(recent_behaviors)
        if total_count > 0:
            positive_ratio = positive_count / total_count
            self.reputation_scores[node_id] = max(0.1, min(1.0, positive_ratio))
    
    def is_byzantine(self, node_id: str) -> bool:
        """Determina si un nodo muestra comportamiento bizantino"""
        return self.reputation_scores[node_id] < self.threshold
    
    def get_vote_weight(self, node_id: str) -> float:
        """Obtiene el peso de voto basado en reputación"""
        return self.reputation_scores[node_id]

class ConsensusEngine:
    """Motor principal de consenso distribuido"""
    
    def __init__(self, node_id: str, nodes: List[str], f: int = None):
        self.node_id = node_id
        self.nodes = set(nodes)
        self.f = f or (len(nodes) - 1) // 3  # Máximo número de nodos bizantinos tolerables
        
        # Estado del consenso
        self.view_number = 0
        self.sequence_number = 0
        self.current_leader = self._select_leader()
        self.node_states: Dict[str, NodeState] = {}
        
        # Propuestas y mensajes
        self.active_proposals: Dict[str, Proposal] = {}
        self.message_log: List[ConsensusMessage] = []
        self.committed_proposals: List[Proposal] = []
        
        # Componentes auxiliares
        self.byzantine_detector = ByzantineDetector()
        self.message_handlers = {
            MessageType.PROPOSAL: self._handle_proposal,
            MessageType.PREPARE: self._handle_prepare,
            MessageType.PROMISE: self._handle_promise,
            MessageType.ACCEPT: self._handle_accept,
            MessageType.COMMIT: self._handle_commit,
            MessageType.VIEW_CHANGE: self._handle_view_change
        }
        
        # Configuración
        self.timeout_duration = 10.0  # Timeout para fases de consenso
        self.heartbeat_interval = 2.0
        self.running = False
        
        # Inicializar estados de nodos
        for node in self.nodes:
            self.node_states[node] = NodeState(
                node_id=node,
                view_number=0,
                sequence_number=0,
                is_leader=(node == self.current_leader),
                reputation_score=1.0,
                last_activity=time.time()
            )
    
    def _select_leader(self) -> str:
        """Selecciona el líder actual basado en el número de vista"""
        sorted_nodes = sorted(self.nodes)
        return sorted_nodes[self.view_number % len(sorted_nodes)]
    
    async def start_consensus_service(self):
        """Inicia el servicio de consenso"""
        self.running = True
        logger.info(f"🏛️ Iniciando servicio de consenso para nodo {self.node_id}")
        logger.info(f"👑 Líder actual: {self.current_leader}")
        
        # Tareas concurrentes
        tasks = [
            asyncio.create_task(self._heartbeat_loop()),
            asyncio.create_task(self._timeout_monitor()),
            asyncio.create_task(self._process_message_queue())
        ]
        
        await asyncio.gather(*tasks)
    
    async def propose(self, data: Dict[str, Any], priority: int = 1) -> str:
        """Propone un nuevo valor para consenso"""
        if not self.running:
            raise RuntimeError("Servicio de consenso no está ejecutándose")
        
        # Solo el líder puede proponer
        if self.node_id != self.current_leader:
            logger.warning(f"⚠️ Solo el líder {self.current_leader} puede proponer")
            return None
        
        # Crear propuesta
        proposal_id = self._generate_proposal_id()
        proposal = Proposal(
            proposal_id=proposal_id,
            proposer_id=self.node_id,
            data=data,
            timestamp=time.time(),
            view_number=self.view_number,
            sequence_number=self.sequence_number,
            status=ProposalStatus.PENDING,
            priority=priority
        )
        
        self.active_proposals[proposal_id] = proposal
        logger.info(f"📝 Nueva propuesta creada: {proposal_id}")
        
        # Iniciar proceso de consenso
        await self._start_consensus_round(proposal)
        return proposal_id
    
    async def _start_consensus_round(self, proposal: Proposal):
        """Inicia una ronda de consenso para una propuesta"""
        try:
            proposal.status = ProposalStatus.PREPARING
            
            # Fase 1: Prepare
            prepare_message = ConsensusMessage(
                message_id=self._generate_message_id(),
                message_type=MessageType.PREPARE,
                sender_id=self.node_id,
                proposal_id=proposal.proposal_id,
                view_number=self.view_number,
                sequence_number=self.sequence_number,
                data={"proposal_data": proposal.data, "priority": proposal.priority},
                timestamp=time.time()
            )
            
            # Enviar prepare a todos los nodos
            await self._broadcast_message(prepare_message)
            
            # Esperar promises
            await self._wait_for_promises(proposal)
            
        except Exception as e:
            logger.error(f"❌ Error en ronda de consenso para {proposal.proposal_id}: {e}")
            proposal.status = ProposalStatus.ABORTED
    
    async def _wait_for_promises(self, proposal: Proposal):
        """Espera por mensajes promise de los nodos"""
        promises_received = 0
        required_promises = len(self.nodes) - self.f  # Mayoría bizantina
        
        start_time = time.time()
        while promises_received < required_promises and time.time() - start_time < self.timeout_duration:
            await asyncio.sleep(0.1)
            
            # Contar promises recibidos para esta propuesta
            promises_received = sum(
                1 for msg in self.message_log
                if (msg.message_type == MessageType.PROMISE and 
                    msg.proposal_id == proposal.proposal_id and
                    msg.view_number == self.view_number)
            )
        
        if promises_received >= required_promises:
            proposal.status = ProposalStatus.PROMISED
            await self._send_accept_phase(proposal)
        else:
            logger.warning(f"⏰ Timeout esperando promises para {proposal.proposal_id}")
            proposal.status = ProposalStatus.ABORTED
    
    async def _send_accept_phase(self, proposal: Proposal):
        """Envía fase de accept"""
        accept_message = ConsensusMessage(
            message_id=self._generate_message_id(),
            message_type=MessageType.ACCEPT,
            sender_id=self.node_id,
            proposal_id=proposal.proposal_id,
            view_number=self.view_number,
            sequence_number=self.sequence_number,
            data={"proposal_data": proposal.data},
            timestamp=time.time()
        )
        
        await self._broadcast_message(accept_message)
        await self._wait_for_accepts(proposal)
    
    async def _wait_for_accepts(self, proposal: Proposal):
        """Espera por mensajes accept de los nodos"""
        accepts_received = 0
        required_accepts = len(self.nodes) - self.f
        
        start_time = time.time()
        while accepts_received < required_accepts and time.time() - start_time < self.timeout_duration:
            await asyncio.sleep(0.1)
            
            # Contar accepts recibidos (ponderados por reputación)
            weighted_accepts = 0
            for msg in self.message_log:
                if (msg.message_type == MessageType.ACCEPT and 
                    msg.proposal_id == proposal.proposal_id and
                    msg.view_number == self.view_number):
                    
                    vote_weight = self.byzantine_detector.get_vote_weight(msg.sender_id)
                    weighted_accepts += vote_weight
            
            accepts_received = int(weighted_accepts)
        
        if accepts_received >= required_accepts:
            proposal.status = ProposalStatus.ACCEPTED
            await self._send_commit_phase(proposal)
        else:
            logger.warning(f"⏰ Timeout esperando accepts para {proposal.proposal_id}")
            proposal.status = ProposalStatus.ABORTED
    
    async def _send_commit_phase(self, proposal: Proposal):
        """Envía fase de commit"""
        commit_message = ConsensusMessage(
            message_id=self._generate_message_id(),
            message_type=MessageType.COMMIT,
            sender_id=self.node_id,
            proposal_id=proposal.proposal_id,
            view_number=self.view_number,
            sequence_number=self.sequence_number,
            data={"proposal_data": proposal.data},
            timestamp=time.time()
        )
        
        await self._broadcast_message(commit_message)
        
        # Marcar como comprometida
        proposal.status = ProposalStatus.COMMITTED
        self.committed_proposals.append(proposal)
        self.sequence_number += 1
        
        logger.info(f"✅ Propuesta {proposal.proposal_id} comprometida exitosamente")
        
        # Ejecutar callback si existe
        await self._execute_committed_proposal(proposal)
    
    async def _execute_committed_proposal(self, proposal: Proposal):
        """Ejecuta una propuesta comprometida"""
        try:
            # Aquí se ejecutaría la lógica específica de la aplicación
            logger.info(f"🚀 Ejecutando propuesta comprometida: {proposal.proposal_id}")
            
            # Ejemplo: actualizar estado de la aplicación
            if "state_update" in proposal.data:
                await self._apply_state_update(proposal.data["state_update"])
            
            # Registrar comportamiento positivo del proposer
            self.byzantine_detector.record_behavior(
                proposal.proposer_id,
                "correct_proposal",
                {"proposal_id": proposal.proposal_id}
            )
            
        except Exception as e:
            logger.error(f"❌ Error ejecutando propuesta {proposal.proposal_id}: {e}")
    
    async def _apply_state_update(self, state_update: Dict[str, Any]):
        """Aplica actualización de estado"""
        # Implementación específica de la aplicación
        logger.info(f"🔄 Aplicando actualización de estado: {state_update}")
    
    async def _handle_proposal(self, message: ConsensusMessage):
        """Maneja mensaje de propuesta"""
        # Solo procesar si viene del líder actual
        if message.sender_id != self.current_leader:
            logger.warning(f"⚠️ Propuesta de nodo no líder: {message.sender_id}")
            self.byzantine_detector.record_behavior(
                message.sender_id,
                "invalid_proposal",
                {"reason": "not_leader"}
            )
            return
        
        # Validar propuesta
        if self._validate_proposal(message):
            # Registrar comportamiento positivo
            self.byzantine_detector.record_behavior(
                message.sender_id,
                "valid_proposal",
                {"proposal_id": message.proposal_id}
            )
        else:
            # Registrar comportamiento sospechoso
            self.byzantine_detector.record_behavior(
                message.sender_id,
                "invalid_proposal",
                {"proposal_id": message.proposal_id}
            )
    
    async def _handle_prepare(self, message: ConsensusMessage):
        """Maneja mensaje prepare"""
        # Enviar promise si es válido
        if self._validate_prepare(message):
            promise_message = ConsensusMessage(
                message_id=self._generate_message_id(),
                message_type=MessageType.PROMISE,
                sender_id=self.node_id,
                proposal_id=message.proposal_id,
                view_number=message.view_number,
                sequence_number=message.sequence_number,
                data={"accepted": True},
                timestamp=time.time()
            )
            
            await self._send_message_to_node(message.sender_id, promise_message)
            
            # Registrar comportamiento positivo
            self.byzantine_detector.record_behavior(
                message.sender_id,
                "correct_prepare",
                {"proposal_id": message.proposal_id}
            )
    
    async def _handle_promise(self, message: ConsensusMessage):
        """Maneja mensaje promise"""
        # Solo el líder procesa promises
        if self.node_id == self.current_leader:
            logger.debug(f"📨 Promise recibido de {message.sender_id} para {message.proposal_id}")
            
            # Registrar comportamiento positivo
            self.byzantine_detector.record_behavior(
                message.sender_id,
                "timely_response",
                {"message_type": "promise"}
            )
    
    async def _handle_accept(self, message: ConsensusMessage):
        """Maneja mensaje accept"""
        logger.debug(f"✅ Accept recibido de {message.sender_id} para {message.proposal_id}")
        
        # Registrar comportamiento positivo
        self.byzantine_detector.record_behavior(
            message.sender_id,
            "correct_vote",
            {"proposal_id": message.proposal_id}
        )
    
    async def _handle_commit(self, message: ConsensusMessage):
        """Maneja mensaje commit"""
        if message.sender_id == self.current_leader:
            # Aplicar la propuesta comprometida
            proposal_data = message.data.get("proposal_data", {})
            logger.info(f"📋 Commit recibido para propuesta: {message.proposal_id}")
            
            # Crear propuesta local si no existe
            if message.proposal_id not in self.active_proposals:
                proposal = Proposal(
                    proposal_id=message.proposal_id,
                    proposer_id=message.sender_id,
                    data=proposal_data,
                    timestamp=message.timestamp,
                    view_number=message.view_number,
                    sequence_number=message.sequence_number,
                    status=ProposalStatus.COMMITTED
                )
                self.committed_proposals.append(proposal)
                await self._execute_committed_proposal(proposal)
    
    async def _handle_view_change(self, message: ConsensusMessage):
        """Maneja cambio de vista"""
        logger.info(f"🔄 Cambio de vista solicitado por {message.sender_id}")
        
        # Incrementar número de vista
        self.view_number += 1
        self.current_leader = self._select_leader()
        
        # Actualizar estado de nodos
        for node_state in self.node_states.values():
            node_state.view_number = self.view_number
            node_state.is_leader = (node_state.node_id == self.current_leader)
        
        logger.info(f"👑 Nuevo líder: {self.current_leader}")
    
    def _validate_proposal(self, message: ConsensusMessage) -> bool:
        """Valida una propuesta"""
        # Verificaciones básicas
        if not message.data or "proposal_data" not in message.data:
            return False
        
        # Verificar secuencia
        if message.sequence_number <= self.sequence_number:
            return False
        
        # Verificar vista
        if message.view_number < self.view_number:
            return False
        
        return True
    
    def _validate_prepare(self, message: ConsensusMessage) -> bool:
        """Valida un mensaje prepare"""
        return (message.view_number >= self.view_number and
                message.sender_id == self.current_leader)
    
    async def _broadcast_message(self, message: ConsensusMessage):
        """Difunde un mensaje a todos los nodos"""
        self.message_log.append(message)
        
        for node_id in self.nodes:
            if node_id != self.node_id:
                await self._send_message_to_node(node_id, message)
    
    async def _send_message_to_node(self, node_id: str, message: ConsensusMessage):
        """Envía un mensaje a un nodo específico"""
        try:
            # En una implementación real, esto sería una llamada HTTP/gRPC
            # Por ahora, simulamos el envío
            logger.debug(f"📤 Enviando {message.message_type.value} a {node_id}")
            
            # Simular latencia de red
            await asyncio.sleep(random.uniform(0.01, 0.1))
            
        except Exception as e:
            logger.error(f"❌ Error enviando mensaje a {node_id}: {e}")
            
            # Registrar comportamiento sospechoso si hay muchos fallos
            self.byzantine_detector.record_behavior(
                node_id,
                "timeout",
                {"message_type": message.message_type.value}
            )
    
    async def _heartbeat_loop(self):
        """Loop de heartbeat para detectar nodos caídos"""
        while self.running:
            try:
                # Enviar heartbeat a todos los nodos
                heartbeat_message = ConsensusMessage(
                    message_id=self._generate_message_id(),
                    message_type=MessageType.HEARTBEAT,
                    sender_id=self.node_id,
                    proposal_id="",
                    view_number=self.view_number,
                    sequence_number=self.sequence_number,
                    data={"status": "alive"},
                    timestamp=time.time()
                )
                
                await self._broadcast_message(heartbeat_message)
                await asyncio.sleep(self.heartbeat_interval)
                
            except Exception as e:
                logger.error(f"❌ Error en heartbeat loop: {e}")
                await asyncio.sleep(1)
    
    async def _timeout_monitor(self):
        """Monitor de timeouts para detectar líder caído"""
        while self.running:
            try:
                current_time = time.time()
                
                # Verificar si el líder está respondiendo
                leader_state = self.node_states.get(self.current_leader)
                if leader_state and current_time - leader_state.last_activity > self.timeout_duration * 2:
                    logger.warning(f"⏰ Líder {self.current_leader} no responde, iniciando cambio de vista")
                    await self._initiate_view_change()
                
                await asyncio.sleep(self.timeout_duration / 2)
                
            except Exception as e:
                logger.error(f"❌ Error en monitor de timeouts: {e}")
                await asyncio.sleep(1)
    
    async def _initiate_view_change(self):
        """Inicia un cambio de vista"""
        view_change_message = ConsensusMessage(
            message_id=self._generate_message_id(),
            message_type=MessageType.VIEW_CHANGE,
            sender_id=self.node_id,
            proposal_id="",
            view_number=self.view_number + 1,
            sequence_number=self.sequence_number,
            data={"reason": "leader_timeout"},
            timestamp=time.time()
        )
        
        await self._broadcast_message(view_change_message)
    
    async def _process_message_queue(self):
        """Procesa cola de mensajes entrantes"""
        while self.running:
            try:
                # En una implementación real, esto procesaría mensajes de una cola
                # Por ahora, simplemente mantenemos el loop activo
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"❌ Error procesando cola de mensajes: {e}")
                await asyncio.sleep(1)
    
    def _generate_proposal_id(self) -> str:
        """Genera ID único para propuesta"""
        return f"{self.node_id}_{self.view_number}_{self.sequence_number}_{int(time.time() * 1000)}"
    
    def _generate_message_id(self) -> str:
        """Genera ID único para mensaje"""
        return f"msg_{self.node_id}_{int(time.time() * 1000000)}"
    
    async def get_consensus_status(self) -> Dict[str, Any]:
        """Obtiene estado actual del consenso"""
        return {
            "node_id": self.node_id,
            "view_number": self.view_number,
            "sequence_number": self.sequence_number,
            "current_leader": self.current_leader,
            "is_leader": self.node_id == self.current_leader,
            "active_proposals": len(self.active_proposals),
            "committed_proposals": len(self.committed_proposals),
            "byzantine_tolerance": self.f,
            "node_count": len(self.nodes),
            "reputation_scores": dict(self.byzantine_detector.reputation_scores),
            "running": self.running
        }

# Función principal para testing
async def main():
    """Función principal para pruebas"""
    # Simular red de 4 nodos (tolera 1 bizantino)
    nodes = ["node_1", "node_2", "node_3", "node_4"]
    
    # Crear motor de consenso para node_1
    consensus = ConsensusEngine("node_1", nodes)
    
    # Mostrar estado inicial
    status = await consensus.get_consensus_status()
    print("🏛️ Estado inicial del consenso:")
    print(json.dumps(status, indent=2))
    
    # Simular propuesta si es líder
    if consensus.node_id == consensus.current_leader:
        proposal_data = {
            "operation": "update_model_weights",
            "model_id": "gpt_distributed_v1",
            "weights_hash": "abc123def456",
            "timestamp": time.time()
        }
        
        print(f"📝 Proponiendo actualización de modelo...")
        proposal_id = await consensus.propose(proposal_data, priority=3)
        print(f"✅ Propuesta creada: {proposal_id}")
    
    print("🚀 Motor de consenso inicializado correctamente")

def start_consensus_loop(config: Dict[str, Any]):
    """Adapter de arranque para ConsensusEngine compatible con main.py.
    Crea el motor y lanza el servicio asincrónico sin bloquear.
    """
    try:
        node_id = config.get("node_id", "node_local")
        nodes = config.get("nodes") or [node_id]
        consensus = ConsensusEngine(node_id=node_id, nodes=nodes)
        loop = asyncio.get_event_loop()
        loop.create_task(consensus.start_consensus_service())
        logger.info(f"🏛️ ConsensusEngine iniciado (node_id={node_id}, nodes={len(nodes)})")
        return consensus
    except Exception as e:
        logger.error(f"❌ No se pudo iniciar el bucle de consenso: {e}")
        return None

if __name__ == "__main__":
    asyncio.run(main())