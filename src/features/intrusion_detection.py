#!/usr/bin/env python3
"""
Sistema de Detección de Intrusiones (IDS) - AEGIS Framework
Detecta ataques de red y comportamiento anómalo en tiempo real.
"""

import asyncio
import time
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import statistics

logger = logging.getLogger(__name__)


class AttackType(Enum):
    """Tipos de ataques detectables"""
    FLOODING = "flooding"
    SPOOFING = "spoofing"
    REPLAY = "replay"
    MITM = "man_in_the_middle"
    ANOMALOUS_BEHAVIOR = "anomalous_behavior"
    INVALID_SIGNATURE = "invalid_signature"
    MALFORMED_MESSAGE = "malformed_message"
    CONSENSUS_ATTACK = "consensus_attack"
    SPAM = "spam"
    IDENTITY_FRAUD = "identity_fraud"


class AlertSeverity(Enum):
    """Severidad de las alertas"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class IntrusionAlert:
    """Alerta de intrusión detectada"""
    alert_id: str
    timestamp: float
    attack_type: AttackType
    severity: AlertSeverity
    source_peer: str
    target_peer: Optional[str]
    description: str
    evidence: Dict[str, Any]
    confidence_score: float  # 0.0 a 1.0
    mitigated: bool = False


@dataclass
class DetectionRule:
    """Regla de detección configurable"""
    rule_id: str
    attack_type: AttackType
    severity: AlertSeverity
    enabled: bool = True
    threshold: float = 0.0
    time_window: int = 60  # segundos
    description: str = ""


@dataclass
class PeerBehaviorProfile:
    """Perfil de comportamiento de un peer"""
    peer_id: str
    message_rate: float = 0.0  # mensajes por segundo
    connection_attempts: int = 0
    failed_authentications: int = 0
    signature_failures: int = 0
    consensus_disagreements: int = 0
    last_activity: float = 0.0
    reputation_score: float = 0.5
    is_suspicious: bool = False

    # Estadísticas históricas
    message_history: deque = field(default_factory=lambda: deque(maxlen=100))
    latency_history: deque = field(default_factory=lambda: deque(maxlen=50))


class IntrusionDetectionSystem:
    """Sistema de Detección de Intrusiones para AEGIS"""

    def __init__(self):
        self.alerts: List[IntrusionAlert] = []
        self.active_alerts: Dict[str, IntrusionAlert] = {}
        self.detection_rules: Dict[str, DetectionRule] = {}
        self.peer_profiles: Dict[str, PeerBehaviorProfile] = {}
        self.message_buffer: deque = deque(maxlen=1000)  # Buffer de mensajes recientes
        self.anomaly_thresholds = {
            'message_rate': 10.0,  # mensajes/segundo
            'connection_rate': 5.0,  # conexiones/minuto
            'signature_failure_rate': 0.1,  # 10% de fallos
            'latency_spike': 2.0,  # multiplicador de latencia normal
        }

        # Estadísticas del sistema
        self.stats = {
            'total_alerts': 0,
            'active_alerts': 0,
            'mitigated_attacks': 0,
            'false_positives': 0
        }

        self._initialize_default_rules()
        logger.info("🛡️ Sistema de Detección de Intrusiones inicializado")

    def _initialize_default_rules(self):
        """Inicializar reglas de detección por defecto"""
        default_rules = [
            DetectionRule(
                rule_id="flooding_detection",
                attack_type=AttackType.FLOODING,
                severity=AlertSeverity.HIGH,
                threshold=50.0,  # 50 mensajes en ventana de tiempo
                time_window=60,
                description="Detección de flooding: demasiados mensajes desde un peer"
            ),
            DetectionRule(
                rule_id="spoofing_detection",
                attack_type=AttackType.SPOOFING,
                severity=AlertSeverity.CRITICAL,
                threshold=1.0,  # Cualquier intento de spoofing
                description="Detección de spoofing de identidad"
            ),
            DetectionRule(
                rule_id="replay_detection",
                attack_type=AttackType.REPLAY,
                severity=AlertSeverity.MEDIUM,
                threshold=3.0,  # 3 mensajes duplicados
                time_window=300,
                description="Detección de ataques de replay"
            ),
            DetectionRule(
                rule_id="signature_failure_detection",
                attack_type=AttackType.INVALID_SIGNATURE,
                severity=AlertSeverity.HIGH,
                threshold=5.0,  # 5 fallos de firma
                time_window=300,
                description="Detección de fallos repetidos de firma"
            ),
            DetectionRule(
                rule_id="consensus_attack_detection",
                attack_type=AttackType.CONSENSUS_ATTACK,
                severity=AlertSeverity.CRITICAL,
                threshold=10.0,  # 10 desacuerdos en consenso
                time_window=600,
                description="Detección de ataques al consenso"
            ),
            DetectionRule(
                rule_id="anomalous_behavior_detection",
                attack_type=AttackType.ANOMALOUS_BEHAVIOR,
                severity=AlertSeverity.MEDIUM,
                threshold=2.0,  # 2 desviaciones estándar
                description="Detección de comportamiento anómalo"
            )
        ]

        for rule in default_rules:
            self.detection_rules[rule.rule_id] = rule

    async def monitor_message(self, message: Dict[str, Any], source_peer: str, target_peer: Optional[str] = None):
        """Monitorear un mensaje en busca de patrones sospechosos"""
        try:
            # Agregar mensaje al buffer
            self.message_buffer.append({
                'message': message,
                'source': source_peer,
                'target': target_peer,
                'timestamp': time.time()
            })

            # Actualizar perfil del peer
            self._update_peer_profile(source_peer, message)

            # Ejecutar detecciones
            await self._run_detections(source_peer, message, target_peer)

        except Exception as e:
            logger.error(f"❌ Error monitoreando mensaje de {source_peer}: {e}")

    def _update_peer_profile(self, peer_id: str, message: Dict[str, Any]):
        """Actualizar perfil de comportamiento del peer"""
        if peer_id not in self.peer_profiles:
            self.peer_profiles[peer_id] = PeerBehaviorProfile(peer_id=peer_id)

        profile = self.peer_profiles[peer_id]
        current_time = time.time()

        # Actualizar estadísticas
        profile.last_activity = current_time
        profile.message_history.append(current_time)

        # Calcular tasa de mensajes (mensajes por segundo en última ventana)
        recent_messages = [t for t in profile.message_history if current_time - t < 60]
        profile.message_rate = len(recent_messages) / 60.0

    async def _run_detections(self, source_peer: str, message: Dict[str, Any], target_peer: Optional[str]):
        """Ejecutar todas las reglas de detección activas"""
        for rule in self.detection_rules.values():
            if not rule.enabled:
                continue

            try:
                alert = await self._evaluate_rule(rule, source_peer, message, target_peer)
                if alert:
                    await self._raise_alert(alert)
            except Exception as e:
                logger.error(f"❌ Error evaluando regla {rule.rule_id}: {e}")

    async def _evaluate_rule(self, rule: DetectionRule, source_peer: str, message: Dict[str, Any], target_peer: Optional[str]) -> Optional[IntrusionAlert]:
        """Evaluar una regla específica de detección"""
        current_time = time.time()

        if rule.attack_type == AttackType.FLOODING:
            return await self._detect_flooding(rule, source_peer, current_time)

        elif rule.attack_type == AttackType.SPOOFING:
            return self._detect_spoofing(rule, source_peer, message)

        elif rule.attack_type == AttackType.REPLAY:
            return self._detect_replay(rule, message)

        elif rule.attack_type == AttackType.INVALID_SIGNATURE:
            return self._detect_invalid_signature(rule, source_peer)

        elif rule.attack_type == AttackType.CONSENSUS_ATTACK:
            return self._detect_consensus_attack(rule, source_peer)

        elif rule.attack_type == AttackType.ANOMALOUS_BEHAVIOR:
            return self._detect_anomalous_behavior(rule, source_peer)

        return None

    async def _detect_flooding(self, rule: DetectionRule, source_peer: str, current_time: float) -> Optional[IntrusionAlert]:
        """Detectar ataques de flooding"""
        if source_peer not in self.peer_profiles:
            return None

        profile = self.peer_profiles[source_peer]

        # Contar mensajes en la ventana de tiempo
        window_start = current_time - rule.time_window
        recent_messages = [t for t in profile.message_history if t > window_start]

        if len(recent_messages) > rule.threshold:
            confidence = min(1.0, len(recent_messages) / (rule.threshold * 2))

            return IntrusionAlert(
                alert_id=f"flood_{source_peer}_{int(current_time)}",
                timestamp=current_time,
                attack_type=AttackType.FLOODING,
                severity=rule.severity,
                source_peer=source_peer,
                target_peer=None,
                description=f"Flooding detectado: {len(recent_messages)} mensajes en {rule.time_window}s",
                evidence={
                    'message_count': len(recent_messages),
                    'time_window': rule.time_window,
                    'threshold': rule.threshold,
                    'message_rate': profile.message_rate
                },
                confidence_score=confidence
            )

        return None

    def _detect_spoofing(self, rule: DetectionRule, source_peer: str, message: Dict[str, Any]) -> Optional[IntrusionAlert]:
        """Detectar spoofing de identidad"""
        # Verificar inconsistencias en la identidad del mensaje
        message_sender = message.get('sender_id') or message.get('node_id')

        if message_sender and message_sender != source_peer:
            return IntrusionAlert(
                alert_id=f"spoof_{source_peer}_{int(time.time())}",
                timestamp=time.time(),
                attack_type=AttackType.SPOOFING,
                severity=rule.severity,
                source_peer=source_peer,
                target_peer=None,
                description=f"Spoofing detectado: mensaje de {message_sender} enviado por {source_peer}",
                evidence={
                    'claimed_sender': message_sender,
                    'actual_sender': source_peer,
                    'message_type': message.get('type')
                },
                confidence_score=1.0
            )

        return None

    def _detect_replay(self, rule: DetectionRule, message: Dict[str, Any]) -> Optional[IntrusionAlert]:
        """Detectar ataques de replay"""
        # Buscar mensajes duplicados en el buffer
        current_time = time.time()
        window_start = current_time - rule.time_window

        duplicate_count = 0
        for buffered_msg in self.message_buffer:
            if buffered_msg['timestamp'] > window_start:
                # Comparar contenido relevante del mensaje (simplificado)
                if self._messages_similar(message, buffered_msg['message']):
                    duplicate_count += 1

        if duplicate_count >= rule.threshold:
            return IntrusionAlert(
                alert_id=f"replay_{int(current_time)}",
                timestamp=current_time,
                attack_type=AttackType.REPLAY,
                severity=rule.severity,
                source_peer=message.get('sender_id', 'unknown'),
                target_peer=message.get('recipient_id'),
                description=f"Ataque de replay detectado: {duplicate_count} mensajes duplicados",
                evidence={
                    'duplicate_count': duplicate_count,
                    'time_window': rule.time_window,
                    'message_hash': hash(str(message))
                },
                confidence_score=min(1.0, duplicate_count / (rule.threshold * 2))
            )

        return None

    def _detect_invalid_signature(self, rule: DetectionRule, source_peer: str) -> Optional[IntrusionAlert]:
        """Detectar fallos repetidos de firma"""
        if source_peer not in self.peer_profiles:
            return None

        profile = self.peer_profiles[source_peer]
        current_time = time.time()
        window_start = current_time - rule.time_window

        # En implementación real, esto se alimentaría desde el crypto engine
        # Por ahora, simulamos basado en perfil
        recent_failures = getattr(profile, 'recent_signature_failures', 0)

        if recent_failures > rule.threshold:
            return IntrusionAlert(
                alert_id=f"sigfail_{source_peer}_{int(current_time)}",
                timestamp=current_time,
                attack_type=AttackType.INVALID_SIGNATURE,
                severity=rule.severity,
                source_peer=source_peer,
                target_peer=None,
                description=f"Fallos repetidos de firma: {recent_failures} en {rule.time_window}s",
                evidence={
                    'failure_count': recent_failures,
                    'time_window': rule.time_window
                },
                confidence_score=min(1.0, recent_failures / (rule.threshold * 2))
            )

        return None

    def _detect_consensus_attack(self, rule: DetectionRule, source_peer: str) -> Optional[IntrusionAlert]:
        """Detectar ataques al consenso"""
        if source_peer not in self.peer_profiles:
            return None

        profile = self.peer_profiles[source_peer]
        current_time = time.time()
        window_start = current_time - rule.time_window

        # Contar desacuerdos recientes de consenso
        recent_disagreements = profile.consensus_disagreements

        if recent_disagreements > rule.threshold:
            return IntrusionAlert(
                alert_id=f"consensus_{source_peer}_{int(current_time)}",
                timestamp=current_time,
                attack_type=AttackType.CONSENSUS_ATTACK,
                severity=rule.severity,
                source_peer=source_peer,
                target_peer=None,
                description=f"Ataque al consenso detectado: {recent_disagreements} desacuerdos",
                evidence={
                    'disagreement_count': recent_disagreements,
                    'time_window': rule.time_window
                },
                confidence_score=min(1.0, recent_disagreements / (rule.threshold * 2))
            )

        return None

    def _detect_anomalous_behavior(self, rule: DetectionRule, source_peer: str) -> Optional[IntrusionAlert]:
        """Detectar comportamiento anómalo usando análisis estadístico"""
        if source_peer not in self.peer_profiles:
            return None

        profile = self.peer_profiles[source_peer]

        # Calcular estadísticas de latencia si hay suficientes datos
        if len(profile.latency_history) >= 10:
            try:
                mean_latency = statistics.mean(profile.latency_history)
                stdev_latency = statistics.stdev(profile.latency_history)

                # Detectar picos de latencia
                if profile.latency_history:
                    latest_latency = profile.latency_history[-1]
                    if stdev_latency > 0 and (latest_latency - mean_latency) / stdev_latency > rule.threshold:
                        return IntrusionAlert(
                            alert_id=f"anomaly_{source_peer}_{int(time.time())}",
                            timestamp=time.time(),
                            attack_type=AttackType.ANOMALOUS_BEHAVIOR,
                            severity=rule.severity,
                            source_peer=source_peer,
                            target_peer=None,
                            description=f"Comportamiento anómalo: latencia {latest_latency:.2f}ms (normal: {mean_latency:.2f}±{stdev_latency:.2f})",
                            evidence={
                                'latest_latency': latest_latency,
                                'mean_latency': mean_latency,
                                'stdev_latency': stdev_latency,
                                'deviation': (latest_latency - mean_latency) / stdev_latency
                            },
                            confidence_score=min(1.0, abs(latest_latency - mean_latency) / (stdev_latency * 2))
                        )

            except statistics.StatisticsError:
                pass  # No hay suficientes datos

        return None

    def _messages_similar(self, msg1: Dict[str, Any], msg2: Dict[str, Any]) -> bool:
        """Verificar si dos mensajes son similares (para detección de replay)"""
        # Comparación simplificada - en producción usar hashes criptográficos
        key_fields = ['type', 'payload', 'sender_id', 'recipient_id']
        return all(msg1.get(field) == msg2.get(field) for field in key_fields)

    async def _raise_alert(self, alert: IntrusionAlert):
        """Generar y procesar una alerta de intrusión"""
        # Agregar a listas
        self.alerts.append(alert)
        self.active_alerts[alert.alert_id] = alert
        self.stats['total_alerts'] += 1
        self.stats['active_alerts'] += 1

        # Log de seguridad
        logger.warning(f"🚨 ALERTA DE SEGURIDAD: {alert.attack_type.value.upper()} - {alert.description}")
        logger.warning(f"   Severidad: {alert.severity.value.upper()} | Confianza: {alert.confidence_score:.2f}")
        logger.warning(f"   Peer: {alert.source_peer} | Evidencia: {alert.evidence}")

        # En implementación real, aquí se integrarían acciones de mitigación:
        # - Bloquear peer temporalmente
        # - Notificar administradores
        # - Activar rotación de emergencia de claves
        # - Aislar el peer de la red

        # Marcar como mitigado automáticamente para alertas críticas
        if alert.severity in [AlertSeverity.CRITICAL]:
            alert.mitigated = True
            self.stats['mitigated_attacks'] += 1
            logger.info(f"🛡️ Ataque mitigado automáticamente: {alert.alert_id}")

    def report_incident(self, peer_id: str, incident_type: str, evidence: Dict[str, Any] = None):
        """Reportar un incidente manualmente (desde otros componentes)"""
        current_time = time.time()

        # Mapear tipos de incidente a AttackType
        attack_type_map = {
            'invalid_signature': AttackType.INVALID_SIGNATURE,
            'malformed_message': AttackType.MALFORMED_MESSAGE,
            'consensus_attack': AttackType.CONSENSUS_ATTACK,
            'spam': AttackType.SPAM,
            'identity_fraud': AttackType.IDENTITY_FRAUD,
        }

        attack_type = attack_type_map.get(incident_type, AttackType.ANOMALOUS_BEHAVIOR)
        severity_map = {
            AttackType.INVALID_SIGNATURE: AlertSeverity.MEDIUM,
            AttackType.MALFORMED_MESSAGE: AlertSeverity.MEDIUM,
            AttackType.CONSENSUS_ATTACK: AlertSeverity.HIGH,
            AttackType.SPAM: AlertSeverity.LOW,
            AttackType.IDENTITY_FRAUD: AlertSeverity.CRITICAL,
        }

        severity = severity_map.get(attack_type, AlertSeverity.MEDIUM)

        alert = IntrusionAlert(
            alert_id=f"incident_{peer_id}_{incident_type}_{int(current_time)}",
            timestamp=current_time,
            attack_type=attack_type,
            severity=severity,
            source_peer=peer_id,
            target_peer=None,
            description=f"Incidente reportado: {incident_type}",
            evidence=evidence or {},
            confidence_score=0.8  # Alta confianza para reportes manuales
        )

        asyncio.create_task(self._raise_alert(alert))

    def update_peer_consensus_behavior(self, peer_id: str, agreed: bool):
        """Actualizar métricas de comportamiento en consenso"""
        if peer_id not in self.peer_profiles:
            self.peer_profiles[peer_id] = PeerBehaviorProfile(peer_id=peer_id)

        profile = self.peer_profiles[peer_id]
        if not agreed:
            profile.consensus_disagreements += 1

    def update_peer_latency(self, peer_id: str, latency: float):
        """Actualizar métricas de latencia del peer"""
        if peer_id not in self.peer_profiles:
            self.peer_profiles[peer_id] = PeerBehaviorProfile(peer_id=peer_id)

        profile = self.peer_profiles[peer_id]
        profile.latency_history.append(latency)

    def get_peer_risk_score(self, peer_id: str) -> float:
        """Obtener puntuación de riesgo de un peer (0.0 = seguro, 1.0 = muy riesgoso)"""
        if peer_id not in self.peer_profiles:
            return 0.0

        profile = self.peer_profiles[peer_id]

        # Calcular riesgo basado en múltiples factores
        risk_factors = [
            min(1.0, profile.message_rate / self.anomaly_thresholds['message_rate']) * 0.3,
            min(1.0, profile.failed_authentications / 10.0) * 0.2,
            min(1.0, profile.signature_failures / 5.0) * 0.2,
            min(1.0, profile.consensus_disagreements / 20.0) * 0.3,
        ]

        return sum(risk_factors)

    def get_active_alerts(self, min_severity: AlertSeverity = AlertSeverity.LOW) -> List[IntrusionAlert]:
        """Obtener alertas activas filtradas por severidad mínima"""
        severity_order = {AlertSeverity.LOW: 0, AlertSeverity.MEDIUM: 1, AlertSeverity.HIGH: 2, AlertSeverity.CRITICAL: 3}
        min_level = severity_order[min_severity]

        return [
            alert for alert in self.active_alerts.values()
            if severity_order[alert.severity] >= min_level
        ]

    def get_system_status(self) -> Dict[str, Any]:
        """Obtener estado general del sistema IDS"""
        return {
            'total_alerts': self.stats['total_alerts'],
            'active_alerts': len(self.active_alerts),
            'mitigated_attacks': self.stats['mitigated_attacks'],
            'monitored_peers': len(self.peer_profiles),
            'enabled_rules': sum(1 for rule in self.detection_rules.values() if rule.enabled),
            'system_health': 'operational' if len(self.active_alerts) < 10 else 'degraded'
        }

    def cleanup_old_alerts(self, max_age: int = 3600):
        """Limpiar alertas antiguas"""
        current_time = time.time()
        to_remove = []

        for alert_id, alert in self.active_alerts.items():
            if current_time - alert.timestamp > max_age:
                to_remove.append(alert_id)

        for alert_id in to_remove:
            del self.active_alerts[alert_id]

        self.stats['active_alerts'] = len(self.active_alerts)

    def reset_stats(self):
        """Reiniciar estadísticas del sistema"""
        self.stats = {
            'total_alerts': 0,
            'active_alerts': 0,
            'mitigated_attacks': 0,
            'false_positives': 0
        }
