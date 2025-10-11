#!/usr/bin/env python3
"""
Sistema de Aprendizaje Distribuido - AEGIS Framework
Implementación avanzada de federated learning con optimizaciones para IA colaborativa.

Características principales:
- Federated Learning con agregación segura
- Aprendizaje diferencial privado
- Optimización de gradientes distribuidos
- Detección de ataques de envenenamiento
- Sincronización adaptativa de modelos
"""

import asyncio
import time
import json
import hashlib
import logging
import numpy as np
from typing import Dict, List, Set, Optional, Tuple, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict, deque
import pickle
import base64
from datetime import datetime, timedelta
import random
import math

# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LearningPhase(Enum):
    """Fases del aprendizaje distribuido"""
    INITIALIZATION = "initialization"
    LOCAL_TRAINING = "local_training"
    GRADIENT_SHARING = "gradient_sharing"
    AGGREGATION = "aggregation"
    MODEL_UPDATE = "model_update"
    VALIDATION = "validation"
    SYNCHRONIZATION = "synchronization"

class AggregationMethod(Enum):
    """Métodos de agregación de modelos"""
    FEDERATED_AVERAGING = "federated_averaging"
    WEIGHTED_AVERAGING = "weighted_averaging"
    BYZANTINE_ROBUST = "byzantine_robust"
    DIFFERENTIAL_PRIVATE = "differential_private"
    ADAPTIVE_AGGREGATION = "adaptive_aggregation"

class AttackType(Enum):
    """Tipos de ataques detectables"""
    POISONING = "poisoning"
    BYZANTINE = "byzantine"
    INFERENCE = "inference"
    BACKDOOR = "backdoor"
    GRADIENT_INVERSION = "gradient_inversion"

@dataclass
class ModelUpdate:
    """Actualización de modelo de un nodo"""
    node_id: str
    model_id: str
    update_id: str
    gradients: Dict[str, np.ndarray]
    weights: Dict[str, np.ndarray]
    metadata: Dict[str, Any]
    timestamp: float
    local_epochs: int
    data_size: int
    loss: float
    accuracy: float
    signature: str = ""

@dataclass
class TrainingRound:
    """Ronda de entrenamiento distribuido"""
    round_id: str
    global_model_version: str
    participating_nodes: Set[str]
    start_time: float
    end_time: Optional[float]
    phase: LearningPhase
    aggregation_method: AggregationMethod
    updates_received: Dict[str, ModelUpdate]
    aggregated_model: Optional[Dict[str, np.ndarray]]
    performance_metrics: Dict[str, float]

@dataclass
class NodeCapabilities:
    """Capacidades de un nodo para aprendizaje"""
    node_id: str
    compute_power: float  # FLOPS disponibles
    memory_capacity: int  # MB disponibles
    bandwidth: float  # Mbps
    data_size: int  # Tamaño del dataset local
    specialization: List[str]  # Tipos de datos/modelos especializados
    privacy_level: str  # "low", "medium", "high"
    reliability_score: float  # 0.0 - 1.0

class PrivacyPreservingAggregator:
    """Agregador con preservación de privacidad"""
    
    def __init__(self, epsilon: float = 1.0, delta: float = 1e-5):
        self.epsilon = epsilon  # Parámetro de privacidad diferencial
        self.delta = delta
        self.noise_scale = self._calculate_noise_scale()
        
    def _calculate_noise_scale(self) -> float:
        """Calcula la escala de ruido para privacidad diferencial"""
        # Mecanismo Gaussiano para privacidad diferencial
        sensitivity = 1.0  # Sensibilidad L2 del mecanismo
        return sensitivity * math.sqrt(2 * math.log(1.25 / self.delta)) / self.epsilon
    
    def add_differential_privacy_noise(self, gradients: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Añade ruido para privacidad diferencial"""
        noisy_gradients = {}
        
        for layer_name, gradient in gradients.items():
            # Añadir ruido Gaussiano
            noise = np.random.normal(0, self.noise_scale, gradient.shape)
            noisy_gradients[layer_name] = gradient + noise
            
        return noisy_gradients
    
    def secure_aggregation(self, updates: List[ModelUpdate]) -> Dict[str, np.ndarray]:
        """Agregación segura con preservación de privacidad"""
        if not updates:
            return {}
        
        # Obtener estructura del modelo
        layer_names = list(updates[0].gradients.keys())
        aggregated_gradients = {}
        
        for layer_name in layer_names:
            # Recopilar gradientes de todos los nodos
            layer_gradients = []
            weights = []
            
            for update in updates:
                if layer_name in update.gradients:
                    # Añadir ruido para privacidad
                    noisy_gradient = self.add_differential_privacy_noise(
                        {layer_name: update.gradients[layer_name]}
                    )[layer_name]
                    
                    layer_gradients.append(noisy_gradient)
                    # Peso basado en tamaño de datos y confiabilidad
                    weight = update.data_size * self._get_node_reliability(update.node_id)
                    weights.append(weight)
            
            if layer_gradients:
                # Agregación ponderada
                weights = np.array(weights)
                weights = weights / np.sum(weights)  # Normalizar
                
                aggregated_gradient = np.zeros_like(layer_gradients[0])
                for gradient, weight in zip(layer_gradients, weights):
                    aggregated_gradient += weight * gradient
                
                aggregated_gradients[layer_name] = aggregated_gradient
        
        return aggregated_gradients
    
    def _get_node_reliability(self, node_id: str) -> float:
        """Obtiene la confiabilidad de un nodo"""
        # En una implementación real, esto vendría del sistema de reputación
        return 1.0

class ByzantineRobustAggregator:
    """Agregador robusto contra ataques bizantinos"""
    
    def __init__(self, byzantine_ratio: float = 0.3):
        self.byzantine_ratio = byzantine_ratio
        self.attack_detector = AttackDetector()
        
    def robust_aggregation(self, updates: List[ModelUpdate]) -> Dict[str, np.ndarray]:
        """Agregación robusta contra nodos bizantinos"""
        if not updates:
            return {}
        
        # Filtrar actualizaciones sospechosas
        clean_updates = self._filter_byzantine_updates(updates)
        
        if not clean_updates:
            logger.warning("⚠️ Todas las actualizaciones fueron filtradas como bizantinas")
            return {}
        
        # Usar mediana geométrica para robustez
        return self._geometric_median_aggregation(clean_updates)
    
    def _filter_byzantine_updates(self, updates: List[ModelUpdate]) -> List[ModelUpdate]:
        """Filtra actualizaciones bizantinas"""
        clean_updates = []
        
        for update in updates:
            # Detectar anomalías en gradientes
            if not self._is_gradient_anomalous(update):
                # Verificar consistencia con historial
                if self._is_update_consistent(update):
                    clean_updates.append(update)
                else:
                    logger.warning(f"⚠️ Actualización inconsistente de {update.node_id}")
            else:
                logger.warning(f"⚠️ Gradientes anómalos detectados en {update.node_id}")
        
        return clean_updates
    
    def _is_gradient_anomalous(self, update: ModelUpdate) -> bool:
        """Detecta si los gradientes son anómalos"""
        for layer_name, gradient in update.gradients.items():
            # Verificar magnitud de gradientes
            gradient_norm = np.linalg.norm(gradient)
            if gradient_norm > 100.0:  # Umbral configurable
                return True
            
            # Verificar distribución de gradientes
            if np.std(gradient) > 10.0:  # Umbral configurable
                return True
        
        return False
    
    def _is_update_consistent(self, update: ModelUpdate) -> bool:
        """Verifica consistencia de la actualización"""
        # Verificar métricas de rendimiento
        if update.loss < 0 or update.accuracy < 0 or update.accuracy > 1:
            return False
        
        # Verificar coherencia entre loss y accuracy
        if update.loss < 0.1 and update.accuracy < 0.5:
            return False
        
        return True
    
    def _geometric_median_aggregation(self, updates: List[ModelUpdate]) -> Dict[str, np.ndarray]:
        """Agregación usando mediana geométrica"""
        if not updates:
            return {}
        
        layer_names = list(updates[0].gradients.keys())
        aggregated_gradients = {}
        
        for layer_name in layer_names:
            gradients = [update.gradients[layer_name] for update in updates if layer_name in update.gradients]
            
            if gradients:
                # Calcular mediana geométrica (aproximación iterativa)
                aggregated_gradients[layer_name] = self._compute_geometric_median(gradients)
        
        return aggregated_gradients
    
    def _compute_geometric_median(self, gradients: List[np.ndarray], max_iterations: int = 100) -> np.ndarray:
        """Calcula la mediana geométrica de un conjunto de gradientes"""
        if len(gradients) == 1:
            return gradients[0]
        
        # Inicializar con la media aritmética
        median = np.mean(gradients, axis=0)
        
        for _ in range(max_iterations):
            # Calcular pesos basados en distancias
            distances = [np.linalg.norm(gradient - median) for gradient in gradients]
            
            # Evitar división por cero
            weights = [1.0 / max(d, 1e-8) for d in distances]
            total_weight = sum(weights)
            
            if total_weight == 0:
                break
            
            # Actualizar mediana
            new_median = np.zeros_like(median)
            for gradient, weight in zip(gradients, weights):
                new_median += (weight / total_weight) * gradient
            
            # Verificar convergencia
            if np.linalg.norm(new_median - median) < 1e-6:
                break
            
            median = new_median
        
        return median

class AttackDetector:
    """Detector de ataques en aprendizaje distribuido"""
    
    def __init__(self):
        self.update_history: Dict[str, List[ModelUpdate]] = defaultdict(list)
        self.baseline_metrics: Dict[str, Dict[str, float]] = {}
        
    def detect_attack(self, update: ModelUpdate) -> Optional[AttackType]:
        """Detecta posibles ataques en una actualización"""
        # Detectar envenenamiento por gradientes anómalos
        if self._detect_poisoning_attack(update):
            return AttackType.POISONING
        
        # Detectar comportamiento bizantino
        if self._detect_byzantine_behavior(update):
            return AttackType.BYZANTINE
        
        # Detectar ataques de backdoor
        if self._detect_backdoor_attack(update):
            return AttackType.BACKDOOR
        
        return None
    
    def _detect_poisoning_attack(self, update: ModelUpdate) -> bool:
        """Detecta ataques de envenenamiento"""
        # Verificar magnitud de gradientes
        for layer_name, gradient in update.gradients.items():
            gradient_norm = np.linalg.norm(gradient)
            
            # Comparar con historial del nodo
            node_history = self.update_history.get(update.node_id, [])
            if node_history:
                historical_norms = [np.linalg.norm(u.gradients.get(layer_name, np.array([0]))) 
                                  for u in node_history[-10:]]  # Últimas 10 actualizaciones
                avg_norm = np.mean(historical_norms)
                
                # Detectar desviación significativa
                if gradient_norm > avg_norm * 5:  # Umbral configurable
                    return True
        
        return False
    
    def _detect_byzantine_behavior(self, update: ModelUpdate) -> bool:
        """Detecta comportamiento bizantino"""
        # Verificar inconsistencias en métricas
        if update.loss < 0 or update.accuracy > 1.0:
            return True
        
        # Verificar coherencia temporal
        node_history = self.update_history.get(update.node_id, [])
        if len(node_history) >= 2:
            last_update = node_history[-1]
            
            # Verificar cambios drásticos en rendimiento
            loss_change = abs(update.loss - last_update.loss)
            accuracy_change = abs(update.accuracy - last_update.accuracy)
            
            if loss_change > 10.0 or accuracy_change > 0.5:
                return True
        
        return False
    
    def _detect_backdoor_attack(self, update: ModelUpdate) -> bool:
        """Detecta ataques de backdoor"""
        # Verificar patrones sospechosos en gradientes
        for layer_name, gradient in update.gradients.items():
            # Detectar patrones regulares que podrían indicar backdoors
            if self._has_suspicious_patterns(gradient):
                return True
        
        return False
    
    def _has_suspicious_patterns(self, gradient: np.ndarray) -> bool:
        """Detecta patrones sospechosos en gradientes"""
        # Verificar periodicidad sospechosa
        flat_gradient = gradient.flatten()
        
        # Calcular autocorrelación
        if len(flat_gradient) > 100:
            autocorr = np.correlate(flat_gradient, flat_gradient, mode='full')
            autocorr = autocorr[autocorr.size // 2:]
            
            # Detectar picos periódicos
            if len(autocorr) > 10 and np.max(autocorr[1:10]) > 0.8 * autocorr[0]:
                return True
        
        return False

class DistributedLearningCoordinator:
    """Coordinador principal del aprendizaje distribuido"""
    
    def __init__(self, node_id: str, model_architecture: Dict[str, Any]):
        self.node_id = node_id
        self.model_architecture = model_architecture
        
        # Estado del aprendizaje
        self.current_round: Optional[TrainingRound] = None
        self.global_model_version = "v1.0.0"
        self.round_counter = 0
        
        # Nodos participantes
        self.registered_nodes: Dict[str, NodeCapabilities] = {}
        self.active_nodes: Set[str] = set()
        
        # Componentes especializados
        self.privacy_aggregator = PrivacyPreservingAggregator()
        self.byzantine_aggregator = ByzantineRobustAggregator()
        self.attack_detector = AttackDetector()
        
        # Configuración
        self.min_participants = 3
        self.max_participants = 20
        self.round_timeout = 300.0  # 5 minutos
        self.target_accuracy = 0.95
        
        # Historial
        self.training_history: List[TrainingRound] = []
        self.performance_metrics: Dict[str, List[float]] = defaultdict(list)
        
    async def start_learning_service(self):
        """Inicia el servicio de aprendizaje distribuido"""
        logger.info(f"🧠 Iniciando servicio de aprendizaje distribuido para nodo {self.node_id}")
        
        # Tareas concurrentes
        tasks = [
            asyncio.create_task(self._coordination_loop()),
            asyncio.create_task(self._monitor_participants()),
            asyncio.create_task(self._performance_tracking())
        ]
        
        await asyncio.gather(*tasks)
    
    async def register_node(self, capabilities: NodeCapabilities) -> bool:
        """Registra un nodo para participar en el aprendizaje"""
        try:
            self.registered_nodes[capabilities.node_id] = capabilities
            logger.info(f"📝 Nodo {capabilities.node_id} registrado para aprendizaje")
            
            # Añadir a nodos activos si cumple requisitos
            if self._meets_participation_requirements(capabilities):
                self.active_nodes.add(capabilities.node_id)
                logger.info(f"✅ Nodo {capabilities.node_id} añadido a participantes activos")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error registrando nodo {capabilities.node_id}: {e}")
            return False
    
    def _meets_participation_requirements(self, capabilities: NodeCapabilities) -> bool:
        """Verifica si un nodo cumple los requisitos para participar"""
        return (capabilities.compute_power >= 1.0 and  # Mínimo 1 GFLOPS
                capabilities.memory_capacity >= 1024 and  # Mínimo 1GB
                capabilities.reliability_score >= 0.7)  # Mínimo 70% confiabilidad
    
    async def start_training_round(self, aggregation_method: AggregationMethod = AggregationMethod.FEDERATED_AVERAGING) -> str:
        """Inicia una nueva ronda de entrenamiento"""
        if self.current_round and self.current_round.end_time is None:
            logger.warning("⚠️ Ya hay una ronda de entrenamiento en progreso")
            return None
        
        # Seleccionar participantes
        participants = self._select_participants()
        if len(participants) < self.min_participants:
            logger.error(f"❌ Insuficientes participantes: {len(participants)} < {self.min_participants}")
            return None
        
        # Crear nueva ronda
        round_id = f"round_{self.round_counter}_{int(time.time())}"
        self.current_round = TrainingRound(
            round_id=round_id,
            global_model_version=self.global_model_version,
            participating_nodes=participants,
            start_time=time.time(),
            end_time=None,
            phase=LearningPhase.INITIALIZATION,
            aggregation_method=aggregation_method,
            updates_received={},
            aggregated_model=None,
            performance_metrics={}
        )
        
        self.round_counter += 1
        logger.info(f"🚀 Iniciando ronda de entrenamiento {round_id} con {len(participants)} participantes")
        
        # Notificar a participantes
        await self._notify_training_start(participants)
        
        return round_id
    
    def _select_participants(self) -> Set[str]:
        """Selecciona nodos para participar en la ronda"""
        # Filtrar nodos elegibles
        eligible_nodes = [
            node_id for node_id in self.active_nodes
            if self.registered_nodes[node_id].reliability_score >= 0.7
        ]
        
        # Ordenar por capacidades y confiabilidad
        eligible_nodes.sort(
            key=lambda nid: (
                self.registered_nodes[nid].compute_power * 
                self.registered_nodes[nid].reliability_score
            ),
            reverse=True
        )
        
        # Seleccionar hasta max_participants
        selected = set(eligible_nodes[:min(self.max_participants, len(eligible_nodes))])
        
        # Asegurar diversidad (diferentes especializaciones)
        if len(selected) > self.min_participants:
            selected = self._ensure_diversity(selected)
        
        return selected
    
    def _ensure_diversity(self, candidates: Set[str]) -> Set[str]:
        """Asegura diversidad en especializaciones de participantes"""
        specializations = defaultdict(list)
        
        # Agrupar por especialización
        for node_id in candidates:
            node_caps = self.registered_nodes[node_id]
            for spec in node_caps.specialization:
                specializations[spec].append(node_id)
        
        # Seleccionar representantes de cada especialización
        diverse_selection = set()
        for spec_nodes in specializations.values():
            if diverse_selection:
                # Añadir el mejor de cada especialización
                best_node = max(spec_nodes, 
                              key=lambda nid: self.registered_nodes[nid].reliability_score)
                diverse_selection.add(best_node)
            else:
                # Añadir todos los de la primera especialización
                diverse_selection.update(spec_nodes[:3])  # Máximo 3 por especialización
        
        return diverse_selection
    
    async def _notify_training_start(self, participants: Set[str]):
        """Notifica a los participantes el inicio del entrenamiento"""
        notification = {
            "round_id": self.current_round.round_id,
            "global_model_version": self.global_model_version,
            "model_architecture": self.model_architecture,
            "training_config": {
                "local_epochs": 5,
                "batch_size": 32,
                "learning_rate": 0.001
            },
            "deadline": time.time() + self.round_timeout
        }
        
        for node_id in participants:
            await self._send_training_notification(node_id, notification)
    
    async def _send_training_notification(self, node_id: str, notification: Dict[str, Any]):
        """Envía notificación de entrenamiento a un nodo"""
        try:
            # En una implementación real, esto sería una llamada HTTP/gRPC
            logger.debug(f"📤 Enviando notificación de entrenamiento a {node_id}")
            
            # Simular latencia
            await asyncio.sleep(0.1)
            
        except Exception as e:
            logger.error(f"❌ Error enviando notificación a {node_id}: {e}")
    
    async def receive_model_update(self, update: ModelUpdate) -> bool:
        """Recibe actualización de modelo de un nodo"""
        if not self.current_round:
            logger.warning("⚠️ No hay ronda activa para recibir actualizaciones")
            return False
        
        if update.node_id not in self.current_round.participating_nodes:
            logger.warning(f"⚠️ Nodo {update.node_id} no está participando en la ronda actual")
            return False
        
        # Detectar ataques
        attack_type = self.attack_detector.detect_attack(update)
        if attack_type:
            logger.warning(f"🚨 Ataque detectado de {update.node_id}: {attack_type.value}")
            return False
        
        # Almacenar actualización
        self.current_round.updates_received[update.node_id] = update
        logger.info(f"📥 Actualización recibida de {update.node_id} ({len(self.current_round.updates_received)}/{len(self.current_round.participating_nodes)})")
        
        # Verificar si se han recibido todas las actualizaciones
        if len(self.current_round.updates_received) >= len(self.current_round.participating_nodes):
            await self._start_aggregation()
        
        return True
    
    async def _start_aggregation(self):
        """Inicia el proceso de agregación de modelos"""
        if not self.current_round:
            return
        
        logger.info(f"🔄 Iniciando agregación para ronda {self.current_round.round_id}")
        self.current_round.phase = LearningPhase.AGGREGATION
        
        updates = list(self.current_round.updates_received.values())
        
        # Seleccionar método de agregación
        if self.current_round.aggregation_method == AggregationMethod.DIFFERENTIAL_PRIVATE:
            aggregated_model = self.privacy_aggregator.secure_aggregation(updates)
        elif self.current_round.aggregation_method == AggregationMethod.BYZANTINE_ROBUST:
            aggregated_model = self.byzantine_aggregator.robust_aggregation(updates)
        else:
            aggregated_model = self._federated_averaging(updates)
        
        self.current_round.aggregated_model = aggregated_model
        self.current_round.phase = LearningPhase.MODEL_UPDATE
        
        # Calcular métricas de rendimiento
        await self._calculate_round_metrics()
        
        # Finalizar ronda
        await self._finalize_training_round()
    
    def _federated_averaging(self, updates: List[ModelUpdate]) -> Dict[str, np.ndarray]:
        """Agregación por promedio federado estándar"""
        if not updates:
            return {}
        
        # Calcular pesos basados en tamaño de datos
        total_data_size = sum(update.data_size for update in updates)
        weights = [update.data_size / total_data_size for update in updates]
        
        # Obtener estructura del modelo
        layer_names = list(updates[0].gradients.keys())
        aggregated_gradients = {}
        
        for layer_name in layer_names:
            # Promedio ponderado de gradientes
            weighted_gradient = np.zeros_like(updates[0].gradients[layer_name])
            
            for update, weight in zip(updates, weights):
                if layer_name in update.gradients:
                    weighted_gradient += weight * update.gradients[layer_name]
            
            aggregated_gradients[layer_name] = weighted_gradient
        
        return aggregated_gradients
    
    async def _calculate_round_metrics(self):
        """Calcula métricas de rendimiento de la ronda"""
        if not self.current_round or not self.current_round.updates_received:
            return
        
        updates = list(self.current_round.updates_received.values())
        
        # Métricas agregadas
        avg_loss = np.mean([update.loss for update in updates])
        avg_accuracy = np.mean([update.accuracy for update in updates])
        total_data_size = sum(update.data_size for update in updates)
        
        self.current_round.performance_metrics = {
            "average_loss": avg_loss,
            "average_accuracy": avg_accuracy,
            "total_data_size": total_data_size,
            "participation_rate": len(updates) / len(self.current_round.participating_nodes),
            "round_duration": time.time() - self.current_round.start_time
        }
        
        # Actualizar historial global
        self.performance_metrics["loss"].append(avg_loss)
        self.performance_metrics["accuracy"].append(avg_accuracy)
        
        logger.info(f"📊 Métricas de ronda - Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.4f}")
    
    async def _finalize_training_round(self):
        """Finaliza la ronda de entrenamiento"""
        if not self.current_round:
            return
        
        self.current_round.end_time = time.time()
        self.current_round.phase = LearningPhase.SYNCHRONIZATION
        
        # Añadir al historial
        self.training_history.append(self.current_round)
        
        # Actualizar versión del modelo global
        self.global_model_version = f"v{self.round_counter}.0.0"
        
        # Notificar finalización a participantes
        await self._notify_round_completion()
        
        logger.info(f"✅ Ronda {self.current_round.round_id} completada exitosamente")
        
        # Verificar si se alcanzó el objetivo
        if self._check_convergence():
            logger.info("🎯 Objetivo de entrenamiento alcanzado!")
        else:
            # Programar siguiente ronda
            await asyncio.sleep(5)  # Pausa entre rondas
            await self.start_training_round(self.current_round.aggregation_method)
        
        self.current_round = None
    
    def _check_convergence(self) -> bool:
        """Verifica si el modelo ha convergido"""
        if len(self.performance_metrics["accuracy"]) < 2:
            return False
        
        current_accuracy = self.performance_metrics["accuracy"][-1]
        
        # Verificar si se alcanzó la precisión objetivo
        if current_accuracy >= self.target_accuracy:
            return True
        
        # Verificar estabilidad (cambio mínimo en últimas 3 rondas)
        if len(self.performance_metrics["accuracy"]) >= 3:
            recent_accuracies = self.performance_metrics["accuracy"][-3:]
            accuracy_variance = np.var(recent_accuracies)
            
            if accuracy_variance < 0.001:  # Muy poca variación
                return True
        
        return False
    
    async def _notify_round_completion(self):
        """Notifica la finalización de la ronda a los participantes"""
        if not self.current_round:
            return
        
        completion_data = {
            "round_id": self.current_round.round_id,
            "new_model_version": self.global_model_version,
            "aggregated_model": self._serialize_model(self.current_round.aggregated_model),
            "performance_metrics": self.current_round.performance_metrics
        }
        
        for node_id in self.current_round.participating_nodes:
            await self._send_completion_notification(node_id, completion_data)
    
    def _serialize_model(self, model: Dict[str, np.ndarray]) -> str:
        """Serializa el modelo para transmisión"""
        if not model:
            return ""
        
        try:
            # Convertir numpy arrays a listas para JSON
            serializable_model = {
                layer_name: weights.tolist()
                for layer_name, weights in model.items()
            }
            return base64.b64encode(pickle.dumps(serializable_model)).decode('utf-8')
        except Exception as e:
            logger.error(f"❌ Error serializando modelo: {e}")
            return ""
    
    async def _send_completion_notification(self, node_id: str, completion_data: Dict[str, Any]):
        """Envía notificación de finalización a un nodo"""
        try:
            logger.debug(f"📤 Enviando notificación de finalización a {node_id}")
            await asyncio.sleep(0.1)  # Simular latencia
        except Exception as e:
            logger.error(f"❌ Error enviando notificación de finalización a {node_id}: {e}")
    
    async def _coordination_loop(self):
        """Loop principal de coordinación"""
        while True:
            try:
                # Verificar estado de rondas activas
                if self.current_round:
                    # Verificar timeout
                    if time.time() - self.current_round.start_time > self.round_timeout:
                        logger.warning(f"⏰ Timeout en ronda {self.current_round.round_id}")
                        await self._handle_round_timeout()
                
                await asyncio.sleep(10)  # Verificar cada 10 segundos
                
            except Exception as e:
                logger.error(f"❌ Error en loop de coordinación: {e}")
                await asyncio.sleep(5)
    
    async def _handle_round_timeout(self):
        """Maneja timeout de ronda de entrenamiento"""
        if not self.current_round:
            return
        
        # Proceder con las actualizaciones recibidas
        if len(self.current_round.updates_received) >= self.min_participants:
            logger.info(f"⚡ Procediendo con {len(self.current_round.updates_received)} actualizaciones")
            await self._start_aggregation()
        else:
            logger.error(f"❌ Insuficientes actualizaciones recibidas, abortando ronda")
            self.current_round.phase = LearningPhase.INITIALIZATION
            self.current_round = None
    
    async def _monitor_participants(self):
        """Monitorea el estado de los participantes"""
        while True:
            try:
                # Verificar nodos activos
                inactive_nodes = []
                for node_id in list(self.active_nodes):
                    node_caps = self.registered_nodes.get(node_id)
                    if not node_caps or node_caps.reliability_score < 0.5:
                        inactive_nodes.append(node_id)
                
                # Remover nodos inactivos
                for node_id in inactive_nodes:
                    self.active_nodes.discard(node_id)
                    logger.warning(f"⚠️ Nodo {node_id} removido por baja confiabilidad")
                
                await asyncio.sleep(60)  # Verificar cada minuto
                
            except Exception as e:
                logger.error(f"❌ Error monitoreando participantes: {e}")
                await asyncio.sleep(30)
    
    async def _performance_tracking(self):
        """Rastrea métricas de rendimiento del sistema"""
        while True:
            try:
                # Generar reporte de rendimiento
                if len(self.training_history) > 0:
                    await self._generate_performance_report()
                
                await asyncio.sleep(300)  # Cada 5 minutos
                
            except Exception as e:
                logger.error(f"❌ Error en seguimiento de rendimiento: {e}")
                await asyncio.sleep(60)
    
    async def _generate_performance_report(self):
        """Genera reporte de rendimiento del sistema"""
        if not self.performance_metrics["accuracy"]:
            return
        
        current_accuracy = self.performance_metrics["accuracy"][-1]
        current_loss = self.performance_metrics["loss"][-1]
        total_rounds = len(self.training_history)
        
        logger.info(f"📈 Reporte de rendimiento - Rondas: {total_rounds}, "
                   f"Accuracy: {current_accuracy:.4f}, Loss: {current_loss:.4f}")
    
    async def get_learning_status(self) -> Dict[str, Any]:
        """Obtiene estado actual del aprendizaje distribuido"""
        current_round_info = None
        if self.current_round:
            current_round_info = {
                "round_id": self.current_round.round_id,
                "phase": self.current_round.phase.value,
                "participants": len(self.current_round.participating_nodes),
                "updates_received": len(self.current_round.updates_received),
                "progress": len(self.current_round.updates_received) / len(self.current_round.participating_nodes)
            }
        
        return {
            "node_id": self.node_id,
            "global_model_version": self.global_model_version,
            "registered_nodes": len(self.registered_nodes),
            "active_nodes": len(self.active_nodes),
            "total_rounds": len(self.training_history),
            "current_round": current_round_info,
            "performance_metrics": {
                "latest_accuracy": self.performance_metrics["accuracy"][-1] if self.performance_metrics["accuracy"] else 0,
                "latest_loss": self.performance_metrics["loss"][-1] if self.performance_metrics["loss"] else 0,
                "convergence_progress": min(1.0, (self.performance_metrics["accuracy"][-1] if self.performance_metrics["accuracy"] else 0) / self.target_accuracy)
            }
        }

# Función principal para testing
async def main():
    """Función principal para pruebas"""
    # Arquitectura de modelo simple
    model_architecture = {
        "layers": [
            {"type": "dense", "units": 128, "activation": "relu"},
            {"type": "dense", "units": 64, "activation": "relu"},
            {"type": "dense", "units": 10, "activation": "softmax"}
        ],
        "optimizer": "adam",
        "loss": "categorical_crossentropy"
    }
    
    # Crear coordinador
    coordinator = DistributedLearningCoordinator("coordinator_1", model_architecture)
    
    # Registrar nodos simulados
    for i in range(5):
        capabilities = NodeCapabilities(
            node_id=f"node_{i+1}",
            compute_power=random.uniform(1.0, 10.0),
            memory_capacity=random.randint(2048, 8192),
            bandwidth=random.uniform(10.0, 100.0),
            data_size=random.randint(1000, 10000),
            specialization=["image_classification", "nlp"][i % 2:i % 2 + 1],
            privacy_level="medium",
            reliability_score=random.uniform(0.7, 1.0)
        )
        await coordinator.register_node(capabilities)
    
    # Mostrar estado inicial
    status = await coordinator.get_learning_status()
    print("🧠 Estado inicial del aprendizaje distribuido:")
    print(json.dumps(status, indent=2))
    
    # Iniciar ronda de entrenamiento
    round_id = await coordinator.start_training_round(AggregationMethod.FEDERATED_AVERAGING)
    if round_id:
        print(f"🚀 Ronda de entrenamiento iniciada: {round_id}")
    
    print("✅ Sistema de aprendizaje distribuido inicializado correctamente")

if __name__ == "__main__":
    asyncio.run(main())