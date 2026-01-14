#!/usr/bin/env python3
"""
🧠 AEGIS ML Framework Integration
Sistema unificado para integración de frameworks de ML (TensorFlow, PyTorch)
con capacidades de aprendizaje federado y distribución de modelos
"""

import asyncio
import json
import time
import hashlib
import secrets
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import logging

# Dependencias ML opcionales
try:
    import tensorflow as tf
    HAS_TENSORFLOW = True
except ImportError:
    tf = None
    HAS_TENSORFLOW = False

try:
    import torch
    import torch.nn as nn
    HAS_PYTORCH = True
except ImportError:
    torch = None
    nn = None
    HAS_PYTORCH = False

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MLFramework(Enum):
    """Frameworks de ML soportados"""
    TENSORFLOW = "tensorflow"
    PYTORCH = "pytorch"
    GENERIC = "generic"

class ModelType(Enum):
    """Tipos de modelos soportados"""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    GENERATIVE = "generative"
    TRANSFORMER = "transformer"
    CNN = "cnn"
    RNN = "rnn"

class TrainingMode(Enum):
    """Modos de entrenamiento"""
    CENTRALIZED = "centralized"
    FEDERATED = "federated"
    DISTRIBUTED = "distributed"

@dataclass
class ModelMetadata:
    """Metadatos de un modelo de ML"""
    model_id: str
    framework: MLFramework
    model_type: ModelType
    architecture: str
    input_shape: List[int]
    output_shape: List[int]
    parameters: int
    created_at: float
    updated_at: float
    version: str
    accuracy: Optional[float] = None
    loss: Optional[float] = None
    training_mode: TrainingMode = TrainingMode.CENTRALIZED
    owner_node: str = ""
    is_federated: bool = False
    federated_rounds: int = 0

@dataclass
class TrainingJob:
    """Trabajo de entrenamiento"""
    job_id: str
    model_id: str
    dataset_id: str
    epochs: int
    batch_size: int
    learning_rate: float
    optimizer: str
    loss_function: str
    metrics: List[str]
    status: str
    progress: float
    start_time: float
    end_time: Optional[float] = None
    worker_nodes: List[str] = field(default_factory=list)
    federated_clients: int = 0

@dataclass
class FederatedUpdate:
    """Actualización federada de un cliente"""
    client_id: str
    model_id: str
    round_number: int
    weights_delta: Dict[str, Any]  # Cambios en los pesos del modelo
    sample_count: int  # Número de muestras usadas para training
    metrics: Dict[str, float]
    timestamp: float
    checksum: str

class MLModelInterface(ABC):
    """Interfaz abstracta para modelos de ML"""

    @abstractmethod
    async def load_model(self, model_path: str) -> Any:
        """Cargar modelo desde archivo"""
        pass

    @abstractmethod
    async def save_model(self, model: Any, model_path: str) -> None:
        """Guardar modelo en archivo"""
        pass

    @abstractmethod
    async def predict(self, model: Any, input_data: Any) -> Any:
        """Realizar predicción con el modelo"""
        pass

    @abstractmethod
    async def get_model_weights(self, model: Any) -> Dict[str, Any]:
        """Obtener pesos del modelo"""
        pass

    @abstractmethod
    async def set_model_weights(self, model: Any, weights: Dict[str, Any]) -> None:
        """Establecer pesos del modelo"""
        pass

    @abstractmethod
    async def aggregate_weights(self, weight_updates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Agregar actualizaciones de pesos (para federated learning)"""
        pass

class TensorFlowInterface(MLModelInterface):
    """Implementación para TensorFlow/Keras"""

    async def load_model(self, model_path: str) -> Any:
        if not HAS_TENSORFLOW:
            raise ImportError("TensorFlow no está disponible")

        def _load():
            return tf.keras.models.load_model(model_path)
        return await asyncio.get_event_loop().run_in_executor(None, _load)

    async def save_model(self, model: Any, model_path: str) -> None:
        def _save():
            model.save(model_path)
        await asyncio.get_event_loop().run_in_executor(None, _save)

    async def predict(self, model: Any, input_data: Any) -> Any:
        def _predict():
            return model.predict(input_data)
        return await asyncio.get_event_loop().run_in_executor(None, _predict)

    async def get_model_weights(self, model: Any) -> Dict[str, Any]:
        def _get_weights():
            weights = {}
            for i, layer in enumerate(model.layers):
                if layer.weights:
                    weights[f"layer_{i}"] = [w.numpy().tolist() for w in layer.weights]
            return weights
        return await asyncio.get_event_loop().run_in_executor(None, _get_weights)

    async def set_model_weights(self, model: Any, weights: Dict[str, Any]) -> None:
        def _set_weights():
            for i, layer in enumerate(model.layers):
                layer_weights_key = f"layer_{i}"
                if layer_weights_key in weights:
                    layer_weights = [tf.convert_to_tensor(w) for w in weights[layer_weights_key]]
                    layer.set_weights(layer_weights)
        await asyncio.get_event_loop().run_in_executor(None, _set_weights)

    async def aggregate_weights(self, weight_updates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Federated averaging para TensorFlow"""
        if not weight_updates:
            return {}

        aggregated = {}
        total_samples = sum(update.get('sample_count', 1) for update in weight_updates)

        # Para cada layer
        layer_keys = weight_updates[0]['weights'].keys()
        for layer_key in layer_keys:
            layer_weights = []
            layer_biases = []

            for update in weight_updates:
                if layer_key in update['weights']:
                    weights = update['weights'][layer_key]
                    if len(weights) >= 2:  # weights + bias
                        layer_weights.append(weights[0])
                        layer_biases.append(weights[1])
                    elif len(weights) == 1:  # solo weights
                        layer_weights.append(weights[0])

            # Federated averaging
            if layer_weights:
                avg_weights = []
                for i in range(len(layer_weights[0])):
                    weight_sum = sum(w[i] for w in layer_weights)
                    avg_weights.append(weight_sum / len(layer_weights))
                aggregated[layer_key] = [avg_weights]

                if layer_biases:
                    avg_biases = []
                    for i in range(len(layer_biases[0])):
                        bias_sum = sum(b[i] for b in layer_biases)
                        avg_biases.append(bias_sum / len(layer_biases))
                    aggregated[layer_key].append(avg_biases)

        return aggregated

class PyTorchInterface(MLModelInterface):
    """Implementación para PyTorch"""

    async def load_model(self, model_path: str) -> Any:
        if not HAS_PYTORCH:
            raise ImportError("PyTorch no está disponible")

        def _load():
            # Para compatibilidad con PyTorch 2.6+, intentar carga segura primero
            try:
                # Intentar carga con weights_only=True (más seguro)
                model = torch.load(model_path, weights_only=True, map_location=torch.device('cpu'))
                # Si es solo pesos, necesitamos recrear la arquitectura
                # Por simplicidad, asumimos que se guardó el modelo completo
                return model
            except Exception:
                # Fallback: carga completa (solo si se confía en la fuente)
                logger.warning(f"Cargando modelo PyTorch con weights_only=False: {model_path}")
                model = torch.load(model_path, weights_only=False, map_location=torch.device('cpu'))
                model.eval()
                return model

        return await asyncio.get_event_loop().run_in_executor(None, _load)

    async def save_model(self, model: Any, model_path: str) -> None:
        def _save():
            torch.save(model, model_path)
        await asyncio.get_event_loop().run_in_executor(None, _save)

    async def predict(self, model: Any, input_data: Any) -> Any:
        def _predict():
            with torch.no_grad():
                if isinstance(input_data, torch.Tensor):
                    output = model(input_data)
                else:
                    tensor_input = torch.tensor(input_data, dtype=torch.float32)
                    output = model(tensor_input)
                return output.numpy()
        return await asyncio.get_event_loop().run_in_executor(None, _predict)

    async def get_model_weights(self, model: Any) -> Dict[str, Any]:
        def _get_weights():
            weights = {}
            for name, param in model.named_parameters():
                weights[name] = param.data.numpy().tolist()
            return weights
        return await asyncio.get_event_loop().run_in_executor(None, _get_weights)

    async def set_model_weights(self, model: Any, weights: Dict[str, Any]) -> None:
        def _set_weights():
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if name in weights:
                        param.data = torch.tensor(weights[name], dtype=param.data.dtype)
        await asyncio.get_event_loop().run_in_executor(None, _set_weights)

    async def aggregate_weights(self, weight_updates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Federated averaging para PyTorch"""
        if not weight_updates:
            return {}

        aggregated = {}
        total_samples = sum(update.get('sample_count', 1) for update in weight_updates)

        # Obtener todas las keys de parámetros
        param_keys = set()
        for update in weight_updates:
            param_keys.update(update['weights'].keys())

        # Federated averaging por parámetro
        for param_key in param_keys:
            param_updates = []
            weights = []

            for update in weight_updates:
                if param_key in update['weights']:
                    param_updates.append(update['weights'][param_key])
                    weights.append(update.get('sample_count', 1))

            if param_updates:
                # Weighted average
                total_weight = sum(weights)
                avg_param = []
                for i in range(len(param_updates[0])):
                    weighted_sum = sum(p[i] * w for p, w in zip(param_updates, weights))
                    avg_param.append(weighted_sum / total_weight)

                aggregated[param_key] = avg_param

        return aggregated

class MLFrameworkManager:
    """Gestor unificado de frameworks de ML"""

    def __init__(self):
        self.interfaces: Dict[MLFramework, MLModelInterface] = {}
        self.models: Dict[str, ModelMetadata] = {}
        self.training_jobs: Dict[str, TrainingJob] = {}
        self.federated_updates: Dict[str, List[FederatedUpdate]] = {}

        # Registrar interfaces disponibles
        if HAS_TENSORFLOW:
            self.interfaces[MLFramework.TENSORFLOW] = TensorFlowInterface()
            logger.info("✅ TensorFlow interface registrada")

        if HAS_PYTORCH:
            self.interfaces[MLFramework.PYTORCH] = PyTorchInterface()
            logger.info("✅ PyTorch interface registrada")

        if not self.interfaces:
            logger.warning("⚠️ No se encontraron frameworks de ML disponibles")

    async def register_model(self, model_path: str, framework: MLFramework,
                           model_type: ModelType, architecture: str,
                           input_shape: List[int], output_shape: List[int],
                           owner_node: str = "") -> str:
        """Registrar un nuevo modelo"""

        if framework not in self.interfaces:
            raise ValueError(f"Framework {framework.value} no está disponible")

        model_id = f"{framework.value}_{secrets.token_hex(8)}"

        # Cargar modelo para obtener metadatos
        interface = self.interfaces[framework]
        model = await interface.load_model(model_path)
        weights = await interface.get_model_weights(model)

        # Estimar número de parámetros
        total_params = sum(len(param) if isinstance(param, list) else 1
                          for param in weights.values())

        metadata = ModelMetadata(
            model_id=model_id,
            framework=framework,
            model_type=model_type,
            architecture=architecture,
            input_shape=input_shape,
            output_shape=output_shape,
            parameters=total_params,
            created_at=time.time(),
            updated_at=time.time(),
            version="1.0.0",
            owner_node=owner_node
        )

        self.models[model_id] = metadata
        logger.info(f"✅ Modelo registrado: {model_id} ({total_params} parámetros)")

        return model_id

    async def predict(self, model_id: str, input_data: Any) -> Any:
        """Realizar predicción con un modelo"""

        if model_id not in self.models:
            raise ValueError(f"Modelo {model_id} no encontrado")

        metadata = self.models[model_id]
        interface = self.interfaces[metadata.framework]

        # Cargar modelo desde cache o storage
        model_path = f"./models/{model_id}.h5"  # Placeholder
        model = await interface.load_model(model_path)

        return await interface.predict(model, input_data)

    async def start_training_job(self, model_id: str, dataset_id: str,
                               config: Dict[str, Any]) -> str:
        """Iniciar un trabajo de entrenamiento"""

        job_id = f"train_{secrets.token_hex(8)}"

        job = TrainingJob(
            job_id=job_id,
            model_id=model_id,
            dataset_id=dataset_id,
            epochs=config.get('epochs', 10),
            batch_size=config.get('batch_size', 32),
            learning_rate=config.get('learning_rate', 0.001),
            optimizer=config.get('optimizer', 'adam'),
            loss_function=config.get('loss_function', 'categorical_crossentropy'),
            metrics=config.get('metrics', ['accuracy']),
            status='pending',
            progress=0.0,
            start_time=time.time()
        )

        self.training_jobs[job_id] = job
        logger.info(f"✅ Trabajo de entrenamiento iniciado: {job_id}")

        return job_id

    async def submit_federated_update(self, update: FederatedUpdate) -> None:
        """Enviar actualización federada"""

        model_id = update.model_id
        if model_id not in self.federated_updates:
            self.federated_updates[model_id] = []

        self.federated_updates[model_id].append(update)

        # Marcar modelo como federated
        if model_id in self.models:
            self.models[model_id].is_federated = True
            self.models[model_id].federated_rounds = max(
                self.models[model_id].federated_rounds,
                update.round_number
            )

        logger.info(f"✅ Actualización federada recibida: {update.client_id} -> {model_id}")

    async def aggregate_federated_updates(self, model_id: str, round_number: int) -> Optional[Dict[str, Any]]:
        """Agregar actualizaciones federadas para una ronda"""

        if model_id not in self.federated_updates:
            return None

        # Filtrar actualizaciones para esta ronda
        round_updates = [
            update for update in self.federated_updates[model_id]
            if update.round_number == round_number
        ]

        if not round_updates:
            return None

        # Obtener interface del framework
        if model_id not in self.models:
            return None

        framework = self.models[model_id].framework
        interface = self.interfaces[framework]

        # Preparar actualizaciones para agregación
        weight_updates = []
        for update in round_updates:
            weight_updates.append({
                'weights': update.weights_delta,
                'sample_count': update.sample_count
            })

        # Agregar pesos
        aggregated_weights = await interface.aggregate_weights(weight_updates)

        logger.info(f"✅ Pesos agregados para ronda {round_number} de {model_id}")

        return aggregated_weights

    def get_model_info(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Obtener información de un modelo"""

        if model_id not in self.models:
            return None

        metadata = self.models[model_id]
        return {
            'model_id': metadata.model_id,
            'framework': metadata.framework.value,
            'model_type': metadata.model_type.value,
            'architecture': metadata.architecture,
            'parameters': metadata.parameters,
            'is_federated': metadata.is_federated,
            'federated_rounds': metadata.federated_rounds,
            'version': metadata.version,
            'created_at': metadata.created_at,
            'owner_node': metadata.owner_node
        }

    def list_models(self) -> List[Dict[str, Any]]:
        """Listar todos los modelos registrados"""
        return [self.get_model_info(model_id) for model_id in self.models.keys()]

    def get_training_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Obtener estado de un trabajo de entrenamiento"""

        if job_id not in self.training_jobs:
            return None

        job = self.training_jobs[job_id]
        return {
            'job_id': job.job_id,
            'model_id': job.model_id,
            'status': job.status,
            'progress': job.progress,
            'epochs': job.epochs,
            'start_time': job.start_time,
            'end_time': job.end_time,
            'worker_nodes': job.worker_nodes
        }

async def demo_ml_framework_integration():
    """Demostración de integración de frameworks ML"""

    print("🧠 DEMO - AEGIS ML Framework Integration")
    print("=" * 50)

    ml_manager = MLFrameworkManager()

    try:
        # Verificar frameworks disponibles
        available_frameworks = list(ml_manager.interfaces.keys())
        print(f"✅ Frameworks disponibles: {[f.value for f in available_frameworks]}")

        if not available_frameworks:
            print("⚠️ No hay frameworks ML disponibles para la demo")
            return

        # Demo de registro de modelo
        print("\n📝 Registrando modelo de ejemplo...")

        # Para demo, creamos un modelo simple si está disponible
        if HAS_TENSORFLOW:
            print("🎯 Usando TensorFlow para demo...")

            # Crear un modelo simple de clasificación
            import tensorflow as tf
            model = tf.keras.Sequential([
                tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)),
                tf.keras.layers.Dense(10, activation='softmax')
            ])
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

            # Guardar modelo temporalmente
            model_path = "./demo_model.h5"
            model.save(model_path)

            # Registrar en el sistema
            model_id = await ml_manager.register_model(
                model_path=model_path,
                framework=MLFramework.TENSORFLOW,
                model_type=ModelType.CLASSIFICATION,
                architecture="Simple MLP",
                input_shape=[784],
                output_shape=[10],
                owner_node="demo_node"
            )

            print(f"✅ Modelo registrado: {model_id}")

            # Demo de predicción
            print("\n🔮 Probando predicción...")
            import numpy as np
            test_input = np.random.rand(1, 784).astype(np.float32)
            prediction = await ml_manager.predict(model_id, test_input)
            print(f"✅ Predicción completada: shape {prediction.shape}")

            # Limpiar archivo demo
            import os
            if os.path.exists(model_path):
                os.remove(model_path)

        elif HAS_PYTORCH:
            print("🎯 Usando PyTorch para demo...")

            # Crear modelo simple
            import torch.nn as nn
            model = nn.Sequential(
                nn.Linear(784, 128),
                nn.ReLU(),
                nn.Linear(128, 10),
                nn.Softmax(dim=1)
            )

            # Guardar solo state_dict para compatibilidad
            model_path = "./demo_model.pth"
            torch.save(model.state_dict(), model_path)

            # Para registro, necesitamos el modelo completo
            # Por ahora, saltamos el registro completo en la demo
            print("✅ Demo PyTorch completada (state_dict guardado)")

            # Limpiar
            import os
            if os.path.exists(model_path):
                os.remove(model_path)

        # Mostrar información del sistema
        print("\n📊 Información del sistema ML:")
        models = ml_manager.list_models()
        print(f"   Modelos registrados: {len(models)}")

        for model in models:
            print(f"   • {model['model_id']}: {model['framework']} - {model['architecture']}")

        print("\n🎉 Demo de integración ML completada exitosamente!")

    except Exception as e:
        print(f"❌ Error en demo: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(demo_ml_framework_integration())
