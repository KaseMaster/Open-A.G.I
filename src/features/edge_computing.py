#!/usr/bin/env python3
"""
🛠️ AEGIS Edge Computing System
Sistema de computación en el borde para dispositivos IoT
con optimización de modelos ML y aprendizaje federado distribuido
"""

import asyncio
import json
import time
import secrets
import hashlib
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DeviceType(Enum):
    """Tipos de dispositivos edge"""
    RASPBERRY_PI = "raspberry_pi"
    JETSON_NANO = "jetson_nano"
    CORAL_DEV_BOARD = "coral_dev_board"
    ESP32 = "esp32"
    ARDUINO = "arduino"
    MOBILE_PHONE = "mobile_phone"
    SMART_CAMERA = "smart_camera"
    INDUSTRIAL_SENSOR = "industrial_sensor"

class EdgeCapability(Enum):
    """Capacidades de dispositivos edge"""
    INFERENCE_ONLY = "inference_only"
    TRAINING_MINI_BATCH = "training_mini_batch"
    FEDERATED_CLIENT = "federated_client"
    DATA_COLLECTION = "data_collection"
    REAL_TIME_PROCESSING = "real_time_processing"

class ModelOptimization(Enum):
    """Técnicas de optimización de modelos"""
    QUANTIZATION = "quantization"
    PRUNING = "pruning"
    DISTILLATION = "distillation"
    TENSORRT = "tensorrt"
    OPENVINO = "openvino"
    TFLITE = "tflite"
    ONNX = "onnx"

@dataclass
class EdgeDevice:
    """Dispositivo edge"""
    device_id: str
    device_type: DeviceType
    capabilities: Set[EdgeCapability]
    hardware_specs: Dict[str, Any]
    location: Dict[str, float]  # lat, lon
    status: str = "offline"
    last_seen: float = 0
    battery_level: Optional[float] = None
    temperature: Optional[float] = None
    network_quality: Optional[float] = None
    deployed_models: List[str] = field(default_factory=list)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)

@dataclass
class EdgeModel:
    """Modelo optimizado para edge"""
    model_id: str
    original_model_id: str
    optimization: ModelOptimization
    target_device: DeviceType
    model_size_bytes: int
    inference_time_ms: float
    accuracy_drop: float
    power_consumption: float
    created_at: float
    checksum: str

@dataclass
class EdgeDeployment:
    """Despliegue en dispositivo edge"""
    deployment_id: str
    device_id: str
    model_id: str
    status: str
    deployed_at: float
    last_active: float
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    error_logs: List[str] = field(default_factory=list)

@dataclass
class EdgeFederatedRound:
    """Ronda federada en edge"""
    round_id: str
    participating_devices: Set[str]
    status: str
    start_time: float
    end_time: Optional[float] = None
    collected_updates: int = 0
    required_updates: int = 0
    aggregated_metrics: Dict[str, Any] = field(default_factory=dict)

class EdgeModelOptimizer:
    """Optimizador de modelos para edge"""

    def __init__(self):
        self.optimization_profiles: Dict[Tuple[DeviceType, ModelOptimization], Dict[str, Any]] = {
            (DeviceType.RASPBERRY_PI, ModelOptimization.QUANTIZATION): {
                "accuracy_drop": 0.02,
                "size_reduction": 0.75,
                "speed_improvement": 2.5
            },
            (DeviceType.JETSON_NANO, ModelOptimization.TENSORRT): {
                "accuracy_drop": 0.01,
                "size_reduction": 0.9,
                "speed_improvement": 5.0
            },
            (DeviceType.CORAL_DEV_BOARD, ModelOptimization.TFLITE): {
                "accuracy_drop": 0.005,
                "size_reduction": 0.8,
                "speed_improvement": 4.0
            },
            (DeviceType.ESP32, ModelOptimization.QUANTIZATION): {
                "accuracy_drop": 0.05,
                "size_reduction": 0.5,
                "speed_improvement": 3.0
            }
        }

    async def optimize_model(self, original_model_id: str, device_type: DeviceType,
                           optimization: ModelOptimization) -> Optional[EdgeModel]:
        """Optimizar modelo para dispositivo edge"""

        # Simular proceso de optimización
        await asyncio.sleep(2)

        profile = self.optimization_profiles.get((device_type, optimization))
        if not profile:
            logger.warning(f"No optimization profile for {device_type.value} + {optimization.value}")
            return None

        # Calcular métricas simuladas
        model_size_bytes = 50000000  # 50MB base
        optimized_size = int(model_size_bytes * profile["size_reduction"])
        inference_time = 100 / profile["speed_improvement"]  # ms
        accuracy_drop = profile["accuracy_drop"]
        power_consumption = 0.5 + (optimization == ModelOptimization.TENSORRT) * 1.5

        # Crear modelo edge
        edge_model = EdgeModel(
            model_id=f"edge_{original_model_id}_{device_type.value}_{optimization.value}",
            original_model_id=original_model_id,
            optimization=optimization,
            target_device=device_type,
            model_size_bytes=optimized_size,
            inference_time_ms=inference_time,
            accuracy_drop=accuracy_drop,
            power_consumption=power_consumption,
            created_at=time.time(),
            checksum=hashlib.sha256(f"{original_model_id}{device_type.value}{optimization.value}".encode()).hexdigest()
        )

        logger.info(f"✅ Modelo optimizado: {edge_model.model_id} ({optimized_size} bytes, {inference_time:.1f}ms)")
        return edge_model

class EdgeDeviceManager:
    """Gestor de dispositivos edge"""

    def __init__(self):
        self.devices: Dict[str, EdgeDevice] = {}
        self.deployments: Dict[str, EdgeDeployment] = {}
        self.device_groups: Dict[str, Set[str]] = {}

    async def register_device(self, device_info: Dict[str, Any]) -> Optional[str]:
        """Registrar nuevo dispositivo edge"""

        device_id = device_info.get("device_id") or f"edge_{secrets.token_hex(4)}"

        device = EdgeDevice(
            device_id=device_id,
            device_type=DeviceType(device_info["device_type"]),
            capabilities=set(EdgeCapability(cap) for cap in device_info.get("capabilities", [])),
            hardware_specs=device_info.get("hardware_specs", {}),
            location=device_info.get("location", {"lat": 0.0, "lon": 0.0}),
            status="online",
            last_seen=time.time()
        )

        self.devices[device_id] = device

        # Asignar a grupos basados en capacidades
        for capability in device.capabilities:
            group_name = f"group_{capability.value}"
            if group_name not in self.device_groups:
                self.device_groups[group_name] = set()
            self.device_groups[group_name].add(device_id)

        logger.info(f"✅ Dispositivo edge registrado: {device_id} ({device.device_type.value})")
        return device_id

    async def update_device_status(self, device_id: str, status: Dict[str, Any]):
        """Actualizar estado de dispositivo"""

        if device_id not in self.devices:
            return

        device = self.devices[device_id]
        device.status = status.get("status", device.status)
        device.last_seen = time.time()
        device.battery_level = status.get("battery_level")
        device.temperature = status.get("temperature")
        device.network_quality = status.get("network_quality")

        # Actualizar métricas de rendimiento
        if "performance" in status:
            device.performance_metrics.update(status["performance"])

    async def deploy_model_to_device(self, device_id: str, model_id: str) -> Optional[str]:
        """Desplegar modelo en dispositivo edge"""

        if device_id not in self.devices:
            logger.error(f"Dispositivo {device_id} no encontrado")
            return None

        device = self.devices[device_id]

        # Verificar compatibilidad
        if EdgeCapability.INFERENCE_ONLY not in device.capabilities:
            logger.error(f"Dispositivo {device_id} no soporta inferencia")
            return None

        # Simular despliegue
        await asyncio.sleep(1)

        deployment = EdgeDeployment(
            deployment_id=f"deploy_{secrets.token_hex(4)}",
            device_id=device_id,
            model_id=model_id,
            status="deployed",
            deployed_at=time.time(),
            last_active=time.time()
        )

        self.deployments[deployment.deployment_id] = deployment
        device.deployed_models.append(model_id)

        logger.info(f"✅ Modelo desplegado: {model_id} -> {device_id}")
        return deployment.deployment_id

    def get_device_status(self, device_id: str) -> Optional[Dict[str, Any]]:
        """Obtener estado de dispositivo"""

        if device_id not in self.devices:
            return None

        device = self.devices[device_id]
        return {
            "device_id": device.device_id,
            "device_type": device.device_type.value,
            "capabilities": [cap.value for cap in device.capabilities],
            "status": device.status,
            "last_seen": device.last_seen,
            "battery_level": device.battery_level,
            "temperature": device.temperature,
            "network_quality": device.network_quality,
            "deployed_models": device.deployed_models,
            "performance_metrics": device.performance_metrics
        }

    def get_devices_by_capability(self, capability: EdgeCapability) -> List[str]:
        """Obtener dispositivos por capacidad"""

        group_name = f"group_{capability.value}"
        return list(self.device_groups.get(group_name, set()))

class EdgeFederatedCoordinator:
    """Coordinador de aprendizaje federado en edge"""

    def __init__(self, device_manager: EdgeDeviceManager):
        self.device_manager = device_manager
        self.active_rounds: Dict[str, EdgeFederatedRound] = {}
        self.completed_rounds: List[EdgeFederatedRound] = []

    async def start_edge_federated_round(self, model_id: str,
                                       participating_devices: List[str]) -> Optional[str]:
        """Iniciar ronda federada en dispositivos edge"""

        round_id = f"edge_round_{secrets.token_hex(4)}"

        # Verificar que dispositivos tienen capacidad federada
        capable_devices = []
        for device_id in participating_devices:
            device = self.device_manager.devices.get(device_id)
            if device and EdgeCapability.FEDERATED_CLIENT in device.capabilities:
                capable_devices.append(device_id)

        if len(capable_devices) < 2:
            logger.error("Insuficientes dispositivos con capacidad federada")
            return None

        round_obj = EdgeFederatedRound(
            round_id=round_id,
            participating_devices=set(capable_devices),
            status="active",
            start_time=time.time(),
            required_updates=len(capable_devices)
        )

        self.active_rounds[round_id] = round_obj

        logger.info(f"🚀 Ronda federada edge iniciada: {round_id} con {len(capable_devices)} dispositivos")
        return round_id

    async def submit_edge_update(self, round_id: str, device_id: str,
                               update_data: Dict[str, Any]) -> bool:
        """Recibir actualización de dispositivo edge"""

        if round_id not in self.active_rounds:
            return False

        round_obj = self.active_rounds[round_id]

        if device_id not in round_obj.participating_devices:
            return False

        # Procesar actualización (simulado)
        round_obj.collected_updates += 1

        # Actualizar métricas del dispositivo
        device = self.device_manager.devices.get(device_id)
        if device:
            device.performance_metrics.update({
                "federated_updates": device.performance_metrics.get("federated_updates", 0) + 1,
                "last_federated_round": round_id,
                "update_timestamp": time.time()
            })

        logger.info(f"📥 Actualización edge recibida: {device_id} en ronda {round_id}")

        # Verificar si ronda está completa
        if round_obj.collected_updates >= round_obj.required_updates:
            await self._complete_edge_round(round_obj)

        return True

    async def _complete_edge_round(self, round_obj: EdgeFederatedRound):
        """Completar ronda federada edge"""

        round_obj.end_time = time.time()
        round_obj.status = "completed"

        # Calcular métricas agregadas
        total_devices = len(round_obj.participating_devices)
        round_obj.aggregated_metrics = {
            "total_devices": total_devices,
            "participation_rate": round_obj.collected_updates / round_obj.required_updates,
            "round_duration": round_obj.end_time - round_obj.start_time,
            "updates_per_second": round_obj.collected_updates / (round_obj.end_time - round_obj.start_time)
        }

        # Mover a rondas completadas
        self.completed_rounds.append(round_obj)
        del self.active_rounds[round_obj.round_id]

        logger.info(f"🏁 Ronda federada edge completada: {round_obj.round_id}")

class EdgeComputingSystem:
    """Sistema completo de computación edge"""

    def __init__(self):
        self.device_manager = EdgeDeviceManager()
        self.model_optimizer = EdgeModelOptimizer()
        self.federated_coordinator = EdgeFederatedCoordinator(self.device_manager)
        self.edge_models: Dict[str, EdgeModel] = {}

    async def register_edge_device(self, device_info: Dict[str, Any]) -> Optional[str]:
        """Registrar dispositivo edge en el sistema"""

        return await self.device_manager.register_device(device_info)

    async def optimize_and_deploy_model(self, original_model_id: str,
                                      device_type: DeviceType,
                                      target_devices: List[str],
                                      optimization: ModelOptimization = ModelOptimization.QUANTIZATION) -> List[str]:
        """Optimizar modelo y desplegar en dispositivos edge"""

        # Optimizar modelo
        edge_model = await self.model_optimizer.optimize_model(
            original_model_id, device_type, optimization
        )

        if not edge_model:
            return []

        self.edge_models[edge_model.model_id] = edge_model

        # Desplegar en dispositivos
        deployment_ids = []
        for device_id in target_devices:
            deployment_id = await self.device_manager.deploy_model_to_device(
                device_id, edge_model.model_id
            )
            if deployment_id:
                deployment_ids.append(deployment_id)

        logger.info(f"✅ Modelo optimizado y desplegado: {edge_model.model_id} en {len(deployment_ids)} dispositivos")
        return deployment_ids

    async def start_edge_federated_learning(self, model_id: str,
                                          device_group: List[str]) -> Optional[str]:
        """Iniciar aprendizaje federado en dispositivos edge"""

        return await self.federated_coordinator.start_edge_federated_round(
            model_id, device_group
        )

    def get_system_status(self) -> Dict[str, Any]:
        """Obtener estado general del sistema edge"""

        total_devices = len(self.device_manager.devices)
        online_devices = sum(1 for d in self.device_manager.devices.values() if d.status == "online")
        active_deployments = sum(1 for d in self.device_manager.deployments.values() if d.status == "deployed")
        active_rounds = len(self.federated_coordinator.active_rounds)

        return {
            "total_devices": total_devices,
            "online_devices": online_devices,
            "offline_devices": total_devices - online_devices,
            "active_deployments": active_deployments,
            "optimized_models": len(self.edge_models),
            "active_federated_rounds": active_rounds,
            "completed_federated_rounds": len(self.federated_coordinator.completed_rounds),
            "device_groups": len(self.device_manager.device_groups)
        }

    def get_edge_models(self) -> List[Dict[str, Any]]:
        """Obtener lista de modelos edge"""

        return [
            {
                "model_id": model.model_id,
                "original_model_id": model.original_model_id,
                "optimization": model.optimization.value,
                "target_device": model.target_device.value,
                "model_size_bytes": model.model_size_bytes,
                "inference_time_ms": model.inference_time_ms,
                "accuracy_drop": model.accuracy_drop,
                "power_consumption": model.power_consumption
            }
            for model in self.edge_models.values()
        ]
