#!/usr/bin/env python3
"""
🤝 AEGIS Federated Learning System
Sistema de aprendizaje federado para entrenamiento distribuido
y preservación de privacidad de datos
"""

import asyncio
import json
import time
import secrets
import hashlib
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
import logging

from ml_framework_integration import MLFramework, FederatedUpdate, MLFrameworkManager

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FederatedPhase(Enum):
    """Fases del proceso federado"""
    WAITING_FOR_CLIENTS = "waiting_for_clients"
    SENDING_MODEL = "sending_model"
    TRAINING = "training"
    COLLECTING_UPDATES = "collecting_updates"
    AGGREGATING = "aggregating"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class FederatedClient:
    """Cliente federado"""
    client_id: str
    node_id: str
    status: str = "disconnected"
    last_seen: float = 0
    current_round: int = 0
    samples_count: int = 0
    training_progress: float = 0.0
    metrics: Dict[str, float] = field(default_factory=dict)

@dataclass
class FederatedRound:
    """Ronda de entrenamiento federado"""
    round_number: int
    status: FederatedPhase
    start_time: float
    end_time: Optional[float] = None
    participating_clients: Set[str] = field(default_factory=set)
    collected_updates: int = 0
    required_updates: int = 0
    global_model_version: str = ""
    aggregated_metrics: Dict[str, float] = field(default_factory=dict)

class FederatedLearningCoordinator:
    """Coordinador del aprendizaje federado"""

    def __init__(self, ml_manager: MLFrameworkManager, min_clients: int = 3,
                 max_rounds: int = 10, aggregation_timeout: int = 300):
        self.ml_manager = ml_manager
        self.min_clients = min_clients
        self.max_rounds = max_rounds
        self.aggregation_timeout = aggregation_timeout

        # Estado del sistema
        self.clients: Dict[str, FederatedClient] = {}
        self.active_round: Optional[FederatedRound] = None
        self.completed_rounds: List[FederatedRound] = []
        self.global_model_id: Optional[str] = None

        # Callbacks para comunicación con nodos
        self.on_round_started = None
        self.on_round_completed = None
        self.on_client_joined = None

    async def start_federated_training(self, initial_model_id: str,
                                     dataset_config: Dict[str, Any]) -> str:
        """Iniciar entrenamiento federado"""

        if len(self.clients) < self.min_clients:
            raise ValueError(f"Insuficientes clientes conectados. Necesarios: {self.min_clients}, Disponibles: {len(self.clients)}")

        self.global_model_id = initial_model_id

        # Crear primera ronda
        round_obj = FederatedRound(
            round_number=1,
            status=FederatedPhase.WAITING_FOR_CLIENTS,
            start_time=time.time(),
            required_updates=len(self.clients)
        )

        self.active_round = round_obj

        logger.info(f"🚀 Entrenamiento federado iniciado con {len(self.clients)} clientes")
        logger.info(f"📊 Modelo global: {initial_model_id}")

        # Notificar inicio
        if self.on_round_started:
            await self.on_round_started(round_obj)

        return f"federated_{secrets.token_hex(8)}"

    async def register_client(self, node_id: str, client_info: Dict[str, Any]) -> str:
        """Registrar un nuevo cliente federado"""

        client_id = f"client_{secrets.token_hex(4)}"

        client = FederatedClient(
            client_id=client_id,
            node_id=node_id,
            status="connected",
            last_seen=time.time(),
            samples_count=client_info.get('samples_count', 0)
        )

        self.clients[client_id] = client

        logger.info(f"✅ Cliente federado registrado: {client_id} ({node_id})")

        if self.on_client_joined:
            await self.on_client_joined(client)

        return client_id

    async def start_round(self, round_number: int) -> Dict[str, Any]:
        """Iniciar una nueva ronda de entrenamiento"""

        if not self.active_round:
            raise ValueError("No hay entrenamiento federado activo")

        # Actualizar estado de la ronda anterior
        if round_number > 1:
            await self._complete_previous_round()

        # Crear nueva ronda
        self.active_round = FederatedRound(
            round_number=round_number,
            status=FederatedPhase.SENDING_MODEL,
            start_time=time.time(),
            required_updates=len(self.clients)
        )

        # Preparar modelo para distribución
        model_info = self.ml_manager.get_model_info(self.global_model_id)
        if not model_info:
            raise ValueError(f"Modelo global {self.global_model_id} no encontrado")

        # Enviar modelo a clientes
        self.active_round.status = FederatedPhase.TRAINING

        round_config = {
            "round_number": round_number,
            "global_model_id": self.global_model_id,
            "model_info": model_info,
            "training_config": {
                "epochs": 1,
                "batch_size": 32,
                "learning_rate": 0.01
            },
            "deadline": time.time() + self.aggregation_timeout
        }

        logger.info(f"🎯 Ronda {round_number} iniciada - Esperando {len(self.clients)} actualizaciones")

        return round_config

    async def submit_client_update(self, client_id: str, update: FederatedUpdate) -> bool:
        """Recibir actualización de un cliente"""

        if not self.active_round:
            return False

        if client_id not in self.clients:
            logger.warning(f"Cliente desconocido: {client_id}")
            return False

        if update.round_number != self.active_round.round_number:
            logger.warning(f"Ronda incorrecta. Esperada: {self.active_round.round_number}, Recibida: {update.round_number}")
            return False

        # Actualizar cliente
        self.clients[client_id].current_round = update.round_number
        self.clients[client_id].training_progress = 1.0
        self.clients[client_id].metrics = update.metrics
        self.clients[client_id].last_seen = time.time()

        # Enviar actualización al ML manager
        await self.ml_manager.submit_federated_update(update)

        self.active_round.collected_updates += 1
        self.active_round.participating_clients.add(client_id)

        logger.info(f"📥 Actualización recibida de {client_id} (ronda {update.round_number})")
        logger.info(f"📊 Progreso: {self.active_round.collected_updates}/{self.active_round.required_updates}")

        # Verificar si tenemos suficientes actualizaciones
        if self.active_round.collected_updates >= self.active_round.required_updates:
            await self._aggregate_round()

        return True

    async def _aggregate_round(self):
        """Agregar actualizaciones de la ronda actual"""

        if not self.active_round:
            return

        self.active_round.status = FederatedPhase.AGGREGATING
        logger.info(f"🔄 Agregando actualizaciones de ronda {self.active_round.round_number}")

        try:
            # Agregar pesos
            aggregated_weights = await self.ml_manager.aggregate_federated_updates(
                self.global_model_id, self.active_round.round_number
            )

            if aggregated_weights:
                # Calcular métricas agregadas
                total_samples = 0
                weighted_metrics = {}

                for client_id in self.active_round.participating_clients:
                    client = self.clients[client_id]
                    total_samples += client.samples_count

                    for metric_name, metric_value in client.metrics.items():
                        if metric_name not in weighted_metrics:
                            weighted_metrics[metric_name] = 0
                        weighted_metrics[metric_name] += metric_value * client.samples_count

                # Promediar métricas
                for metric_name in weighted_metrics:
                    weighted_metrics[metric_name] /= total_samples

                self.active_round.aggregated_metrics = weighted_metrics

                logger.info(f"✅ Agregación completada - Métricas: {weighted_metrics}")

                # Completar ronda
                await self._complete_round()

            else:
                logger.error("❌ Error en agregación de pesos")
                self.active_round.status = FederatedPhase.FAILED

        except Exception as e:
            logger.error(f"❌ Error en agregación: {e}")
            self.active_round.status = FederatedPhase.FAILED

    async def _complete_round(self):
        """Completar la ronda actual"""

        if not self.active_round:
            return

        self.active_round.end_time = time.time()
        self.active_round.status = FederatedPhase.COMPLETED

        # Agregar a rondas completadas
        self.completed_rounds.append(self.active_round)

        logger.info(f"🏁 Ronda {self.active_round.round_number} completada en {self.active_round.end_time - self.active_round.start_time:.1f}s")

        # Notificar completación
        if self.on_round_completed:
            await self.on_round_completed(self.active_round)

        # Verificar si continuar con siguiente ronda
        if self.active_round.round_number < self.max_rounds:
            # Crear siguiente ronda automáticamente
            await asyncio.sleep(1)  # Pequeño delay
            await self.start_round(self.active_round.round_number + 1)
        else:
            logger.info("🎉 Entrenamiento federado completado!")

    async def _complete_previous_round(self):
        """Completar la ronda anterior si existe"""

        if self.active_round and self.active_round.status != FederatedPhase.COMPLETED:
            self.active_round.end_time = time.time()
            self.active_round.status = FederatedPhase.COMPLETED
            self.completed_rounds.append(self.active_round)

    def get_federated_status(self) -> Dict[str, Any]:
        """Obtener estado del entrenamiento federado"""

        return {
            "active_round": self.active_round.round_number if self.active_round else None,
            "round_status": self.active_round.status.value if self.active_round else None,
            "connected_clients": len(self.clients),
            "participating_clients": len(self.active_round.participating_clients) if self.active_round else 0,
            "collected_updates": self.active_round.collected_updates if self.active_round else 0,
            "required_updates": self.active_round.required_updates if self.active_round else 0,
            "completed_rounds": len(self.completed_rounds),
            "global_model_id": self.global_model_id
        }

    def get_client_status(self, client_id: str) -> Optional[Dict[str, Any]]:
        """Obtener estado de un cliente"""

        if client_id not in self.clients:
            return None

        client = self.clients[client_id]
        return {
            "client_id": client.client_id,
            "node_id": client.node_id,
            "status": client.status,
            "current_round": client.current_round,
            "training_progress": client.training_progress,
            "samples_count": client.samples_count,
            "metrics": client.metrics,
            "last_seen": client.last_seen
        }

class FederatedLearningClient:
    """Cliente para aprendizaje federado"""

    def __init__(self, client_id: str, node_id: str, ml_manager: MLFrameworkManager):
        self.client_id = client_id
        self.node_id = node_id
        self.ml_manager = ml_manager
        self.local_model: Optional[Any] = None
        self.current_round = 0
        self.training_data = None
        self.is_training = False

    async def receive_global_model(self, model_id: str, round_number: int):
        """Recibir modelo global para entrenamiento local"""

        self.current_round = round_number

        # En un escenario real, aquí se descargaría el modelo
        # Por simplicidad, asumimos que ya está disponible
        logger.info(f"📥 Modelo global recibido: {model_id} (ronda {round_number})")

        # Preparar datos de entrenamiento locales (simulado)
        self.training_data = {
            "samples": 1000,  # Simulado
            "features": 784,
            "labels": 10
        }

    async def train_local_model(self, training_config: Dict[str, Any]) -> FederatedUpdate:
        """Entrenar modelo localmente"""

        if not self.training_data:
            raise ValueError("No hay datos de entrenamiento disponibles")

        self.is_training = True
        start_time = time.time()

        try:
            logger.info(f"🏋️ Entrenamiento local iniciado (cliente {self.client_id})")

            # Simular entrenamiento local
            # En un escenario real, aquí se haría el entrenamiento real
            await asyncio.sleep(2)  # Simular tiempo de entrenamiento

            # Simular métricas de entrenamiento
            metrics = {
                "loss": 0.5 + (secrets.randbelow(50) / 100),  # 0.5-1.0
                "accuracy": 0.8 + (secrets.randbelow(20) / 100),  # 0.8-1.0
                "val_loss": 0.6 + (secrets.randbelow(40) / 100),
                "val_accuracy": 0.75 + (secrets.randbelow(25) / 100)
            }

            # Simular cambios en pesos (delta weights)
            # En un escenario real, se calcularía la diferencia con el modelo global
            weights_delta = {
                "layer_0_weights": [[secrets.randbelow(100)/1000 for _ in range(10)] for _ in range(784)],
                "layer_0_bias": [secrets.randbelow(100)/1000 for _ in range(10)],
                "layer_1_weights": [[secrets.randbelow(100)/1000 for _ in range(10)] for _ in range(128)],
                "layer_1_bias": [secrets.randbelow(100)/1000 for _ in range(10)]
            }

            training_time = time.time() - start_time

            # Crear actualización
            update = FederatedUpdate(
                client_id=self.client_id,
                model_id="global_model",  # En escenario real, vendría del coordinator
                round_number=self.current_round,
                weights_delta=weights_delta,
                sample_count=self.training_data["samples"],
                metrics=metrics,
                timestamp=time.time(),
                checksum=hashlib.sha256(json.dumps(weights_delta, sort_keys=True).encode()).hexdigest()
            )

            logger.info(f"🏋️ Entrenamiento local completado en {training_time:.1f}s")
            return update

        finally:
            self.is_training = False

async def demo_federated_learning():
    """Demostración del sistema de aprendizaje federado"""

    print("🤝 DEMO - AEGIS Federated Learning System")
    print("=" * 50)

    # Inicializar componentes
    ml_manager = MLFrameworkManager()
    coordinator = FederatedLearningCoordinator(ml_manager, min_clients=2, max_rounds=3)

    print("✅ Sistema federado inicializado")

    # Simular registro de clientes
    print("\n👥 Registrando clientes federados...")

    clients = []
    for i in range(3):
        node_id = f"node_{i}"
        client_info = {"samples_count": 1000 + (i * 500)}

        client_id = await coordinator.register_client(node_id, client_info)
        client = FederatedLearningClient(client_id, node_id, ml_manager)
        clients.append(client)

        print(f"✅ Cliente {i+1} registrado: {client_id}")

    # Mostrar estado inicial
    status = coordinator.get_federated_status()
    print("\n📊 Estado inicial:")
    print(f"   Clientes conectados: {status['connected_clients']}")
    print(f"   Ronda activa: {status['active_round']}")

    # Simular entrenamiento federado
    print("\n🚀 Iniciando simulación de entrenamiento federado...")
    # Para demo, creamos un modelo dummy
    dummy_model_id = "demo_global_model"

    # Simular registro de modelo en ML manager para demo
    print("📝 Registrando modelo global para demo...")
    # Crear metadatos mock para el modelo
    from ml_framework_integration import ModelMetadata, MLFramework, ModelType
    mock_metadata = ModelMetadata(
        model_id=dummy_model_id,
        framework=MLFramework.PYTORCH,
        model_type=ModelType.CLASSIFICATION,
        architecture="Mock Neural Network",
        input_shape=[784],
        output_shape=[10],
        parameters=100000,
        created_at=time.time(),
        updated_at=time.time(),
        version="1.0.0",
        owner_node="coordinator"
    )
    ml_manager.models[dummy_model_id] = mock_metadata
    print(f"✅ Modelo mock registrado: {dummy_model_id}")

    try:
        # Iniciar entrenamiento
        training_id = await coordinator.start_federated_training(dummy_model_id, {})

        # Simular varias rondas
        for round_num in range(1, 4):  # 3 rondas
            print(f"\n🎯 RONDA {round_num}")
            print("-" * 20)

            # Iniciar ronda
            round_config = await coordinator.start_round(round_num)
            print(f"📤 Modelo enviado a {len(clients)} clientes")

            # Simular entrenamiento de clientes
            update_tasks = []
            for client in clients:
                await client.receive_global_model(dummy_model_id, round_num)
                task = asyncio.create_task(client.train_local_model(round_config["training_config"]))
                update_tasks.append(task)

            # Recolectar actualizaciones
            updates = await asyncio.gather(*update_tasks)

            # Enviar actualizaciones al coordinador
            for update in updates:
                await coordinator.submit_client_update(update.client_id, update)

            # Esperar agregación
            await asyncio.sleep(2)

            # Mostrar estado
            status = coordinator.get_federated_status()
            print(f"✅ Ronda {round_num} completada")
            print(f"   Actualizaciones recolectadas: {status['collected_updates']}")
            print(f"   Rondas completadas: {status['completed_rounds']}")

        print("\n🏆 SIMULACIÓN COMPLETADA")
        print(f"   Rondas completadas: {len(coordinator.completed_rounds)}")
        print(f"   Clientes participantes: {status['connected_clients']}")

        # Mostrar métricas finales
        print("\n📊 MÉTRICAS FINALES:")
        for i, round_obj in enumerate(coordinator.completed_rounds, 1):
            if round_obj.aggregated_metrics:
                metrics = round_obj.aggregated_metrics
                print(f"   Ronda {i}: Loss={metrics.get('loss', 'N/A'):.3f}, "
                      f"Accuracy={metrics.get('accuracy', 'N/A'):.3f}")

        print("\n🎉 Demo de aprendizaje federado completada exitosamente!")

    except Exception as e:
        print(f"❌ Error en simulación: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(demo_federated_learning())
