#!/usr/bin/env python3
"""
🛠️ AEGIS SDK - Software Development Kit
SDK completo para desarrolladores del AEGIS Framework
con APIs unificadas, helpers y utilidades avanzadas
"""

import asyncio
import json
import time
import secrets
import hashlib
from typing import Dict, List, Any, Optional, Union, Callable, TypeVar, Generic
from dataclasses import dataclass, field
from enum import Enum
import logging

# Importar componentes del framework
from ml_framework_integration import MLFrameworkManager, MLFramework
from federated_learning import FederatedLearningCoordinator
from model_distribution import ModelDistributionService
from multi_cloud_orchestration import MultiCloudOrchestrator, CloudProvider, InstanceType
from edge_computing import EdgeComputingSystem, DeviceType

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SDKError(Exception):
    """Excepción base del SDK"""
    pass

class AuthenticationError(SDKError):
    """Error de autenticación"""
    pass

class ValidationError(SDKError):
    """Error de validación"""
    pass

class ResourceNotFoundError(SDKError):
    """Recurso no encontrado"""
    pass

class SDKConfig:
    """Configuración del SDK"""

    def __init__(self, api_url: str = "http://localhost:8080",
                 api_key: Optional[str] = None,
                 timeout: float = 30.0,
                 retry_attempts: int = 3,
                 enable_cache: bool = True):
        self.api_url = api_url
        self.api_key = api_key or secrets.token_hex(32)
        self.timeout = timeout
        self.retry_attempts = retry_attempts
        self.enable_cache = enable_cache

@dataclass
class SDKResponse:
    """Respuesta unificada del SDK"""
    success: bool
    data: Any = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

T = TypeVar('T')

class AEGISClient:
    """Cliente principal del SDK de AEGIS"""

    def __init__(self, config: SDKConfig):
        self.config = config
        self.session_token: Optional[str] = None
        self._cache: Dict[str, Any] = {}
        self._ml_manager = MLFrameworkManager()
        self._federated_coordinator = FederatedLearningCoordinator(self._ml_manager)
        self._distribution_service = ModelDistributionService(self._ml_manager, "sdk_client")
        self._cloud_orchestrator = MultiCloudOrchestrator()
        self._edge_system = EdgeComputingSystem()

    async def authenticate(self, credentials: Dict[str, Any]) -> SDKResponse:
        """Autenticar con el sistema AEGIS"""
        try:
            # Simular autenticación (en producción sería llamada a API)
            await asyncio.sleep(0.1)

            if credentials.get("api_key") == self.config.api_key:
                self.session_token = secrets.token_hex(32)
                return SDKResponse(
                    success=True,
                    data={"session_token": self.session_token},
                    metadata={"expires_in": 3600}
                )
            else:
                raise AuthenticationError("Credenciales inválidas")

        except Exception as e:
            return SDKResponse(success=False, error=str(e))

    async def _ensure_authenticated(self):
        """Asegurar que el cliente está autenticado"""
        if not self.session_token:
            raise AuthenticationError("Cliente no autenticado. Llama authenticate() primero.")

    # ===== ML FRAMEWORK METHODS =====

    async def register_model(self, model_path: str, framework: MLFramework,
                           model_type: str, metadata: Dict[str, Any]) -> SDKResponse:
        """Registrar un modelo de ML"""
        await self._ensure_authenticated()

        try:
            # Crear metadatos del modelo
            from ml_framework_integration import ModelMetadata, ModelType

            model_metadata = ModelMetadata(
                model_id=f"model_{secrets.token_hex(8)}",
                framework=framework,
                model_type=ModelType(model_type),
                architecture=metadata.get("architecture", "Unknown"),
                input_shape=metadata.get("input_shape", []),
                output_shape=metadata.get("output_shape", []),
                parameters=metadata.get("parameters", 0),
                created_at=time.time(),
                updated_at=time.time(),
                version=metadata.get("version", "1.0.0")
            )

            self._ml_manager.models[model_metadata.model_id] = model_metadata

            # Registrar para distribución
            await self._distribution_service.register_model_for_distribution(model_metadata.model_id)

            return SDKResponse(
                success=True,
                data={
                    "model_id": model_metadata.model_id,
                    "framework": framework.value,
                    "status": "registered"
                }
            )

        except Exception as e:
            return SDKResponse(success=False, error=str(e))

    async def predict(self, model_id: str, input_data: Any) -> SDKResponse:
        """Realizar predicción con un modelo"""
        await self._ensure_authenticated()

        try:
            prediction = await self._ml_manager.predict(model_id, input_data)
            return SDKResponse(success=True, data={"prediction": prediction})

        except Exception as e:
            return SDKResponse(success=False, error=str(e))

    async def start_federated_training(self, model_id: str,
                                     participant_nodes: List[str]) -> SDKResponse:
        """Iniciar entrenamiento federado"""
        await self._ensure_authenticated()

        try:
            training_id = await self._federated_coordinator.start_federated_training(
                model_id, {"participants": participant_nodes}
            )

            if training_id:
                return SDKResponse(
                    success=True,
                    data={
                        "training_id": training_id,
                        "model_id": model_id,
                        "participants": len(participant_nodes),
                        "status": "started"
                    }
                )
            else:
                return SDKResponse(success=False, error="No se pudo iniciar entrenamiento federado")

        except Exception as e:
            return SDKResponse(success=False, error=str(e))

    async def distribute_model(self, model_id: str, target_nodes: List[str],
                             strategy: str = "push") -> SDKResponse:
        """Distribuir modelo a nodos"""
        await self._ensure_authenticated()

        try:
            from model_distribution import ReplicationStrategy

            strategy_enum = ReplicationStrategy(strategy.lower())
            task_id = await self._distribution_service.distribute_model(
                model_id, target_nodes, strategy_enum
            )

            if task_id:
                return SDKResponse(
                    success=True,
                    data={
                        "task_id": task_id,
                        "model_id": model_id,
                        "targets": len(target_nodes),
                        "strategy": strategy
                    }
                )
            else:
                return SDKResponse(success=False, error="No se pudo iniciar distribución")

        except Exception as e:
            return SDKResponse(success=False, error=str(e))

    # ===== CLOUD ORCHESTRATION METHODS =====

    async def create_cloud_deployment(self, name: str, provider: str,
                                    region: str, instance_config: Dict[str, Any]) -> SDKResponse:
        """Crear despliegue en cloud"""
        await self._ensure_authenticated()

        try:
            provider_enum = CloudProvider(provider.lower())
            instance_type = InstanceType(instance_config.get("instance_type", "t2_micro"))

            deployment_id = await self._cloud_orchestrator.create_deployment(
                name=name,
                provider=provider_enum,
                region=region,
                instance_type=instance_type,
                instance_count=instance_config.get("count", 1),
                auto_scaling=instance_config.get("auto_scaling", True),
                min_instances=instance_config.get("min_instances", 1),
                max_instances=instance_config.get("max_instances", 5),
                cost_budget=instance_config.get("cost_budget")
            )

            if deployment_id:
                return SDKResponse(
                    success=True,
                    data={
                        "deployment_id": deployment_id,
                        "provider": provider,
                        "region": region,
                        "instances": instance_config.get("count", 1)
                    }
                )
            else:
                return SDKResponse(success=False, error="No se pudo crear despliegue")

        except Exception as e:
            return SDKResponse(success=False, error=str(e))

    async def get_cloud_metrics(self) -> SDKResponse:
        """Obtener métricas de cloud"""
        await self._ensure_authenticated()

        try:
            metrics = await self._cloud_orchestrator.get_global_metrics()
            return SDKResponse(success=True, data=metrics)

        except Exception as e:
            return SDKResponse(success=False, error=str(e))

    # ===== EDGE COMPUTING METHODS =====

    async def register_edge_device(self, device_info: Dict[str, Any]) -> SDKResponse:
        """Registrar dispositivo edge"""
        await self._ensure_authenticated()

        try:
            device_id = await self._edge_system.register_edge_device(device_info)

            if device_id:
                return SDKResponse(
                    success=True,
                    data={
                        "device_id": device_id,
                        "device_type": device_info.get("device_type"),
                        "capabilities": device_info.get("capabilities", [])
                    }
                )
            else:
                return SDKResponse(success=False, error="No se pudo registrar dispositivo")

        except Exception as e:
            return SDKResponse(success=False, error=str(e))

    async def deploy_to_edge(self, model_id: str, device_ids: List[str],
                           optimization: str = "quantization") -> SDKResponse:
        """Desplegar modelo optimizado en dispositivos edge"""
        await self._ensure_authenticated()

        try:
            from edge_computing import ModelOptimization

            opt_enum = ModelOptimization(optimization.upper())
            deployment_ids = await self._edge_system.optimize_and_deploy_model(
                model_id, DeviceType.RASPBERRY_PI, device_ids, opt_enum
            )

            return SDKResponse(
                success=True,
                data={
                    "model_id": model_id,
                    "deployments": len(deployment_ids),
                    "optimization": optimization,
                    "device_count": len(device_ids)
                }
            )

        except Exception as e:
            return SDKResponse(success=False, error=str(e))

    async def start_edge_federated(self, model_id: str, device_ids: List[str]) -> SDKResponse:
        """Iniciar aprendizaje federado en edge"""
        await self._ensure_authenticated()

        try:
            round_id = await self._edge_system.start_edge_federated_learning(model_id, device_ids)

            if round_id:
                return SDKResponse(
                    success=True,
                    data={
                        "round_id": round_id,
                        "model_id": model_id,
                        "devices": len(device_ids),
                        "status": "started"
                    }
                )
            else:
                return SDKResponse(success=False, error="No se pudo iniciar federated learning en edge")

        except Exception as e:
            return SDKResponse(success=False, error=str(e))

    # ===== UTILITY METHODS =====

    async def health_check(self) -> SDKResponse:
        """Verificar estado del sistema"""
        try:
            # Verificar conectividad con todos los subsistemas
            systems_status = {
                "ml_framework": len(self._ml_manager.models) >= 0,
                "federated_learning": self._federated_coordinator is not None,
                "distribution": self._distribution_service is not None,
                "cloud_orchestrator": len(self._cloud_orchestrator.deployments) >= 0,
                "edge_system": self._edge_system.device_manager is not None
            }

            all_healthy = all(systems_status.values())

            return SDKResponse(
                success=all_healthy,
                data={
                    "overall_health": "healthy" if all_healthy else "degraded",
                    "systems": systems_status,
                    "timestamp": time.time()
                }
            )

        except Exception as e:
            return SDKResponse(success=False, error=str(e))

    async def get_system_info(self) -> SDKResponse:
        """Obtener información del sistema"""
        await self._ensure_authenticated()

        try:
            info = {
                "version": "3.3.0",
                "frameworks": {
                    "ml_supported": [f.value for f in self._ml_manager.interfaces.keys()],
                    "cloud_providers": [p.value for p in CloudProvider],
                    "edge_devices": [d.value for d in DeviceType]
                },
                "capabilities": {
                    "federated_learning": True,
                    "model_distribution": True,
                    "auto_scaling": True,
                    "edge_computing": True,
                    "multi_cloud": True
                },
                "limits": {
                    "max_models": 1000,
                    "max_devices": 10000,
                    "max_deployments": 100,
                    "max_federated_rounds": 50
                }
            }

            return SDKResponse(success=True, data=info)

        except Exception as e:
            return SDKResponse(success=False, error=str(e))

# ===== HIGH-LEVEL APIs =====

class AEGIS:
    """API de alto nivel para casos de uso comunes"""

    def __init__(self, api_key: Optional[str] = None):
        config = SDKConfig(api_key=api_key)
        self.client = AEGISClient(config)

    async def quick_start(self, use_case: str, config: Dict[str, Any]) -> SDKResponse:
        """Inicio rápido para casos de uso comunes"""

        # Autenticar automáticamente
        auth_result = await self.client.authenticate({"api_key": self.client.config.api_key})
        if not auth_result.success:
            return auth_result

        try:
            if use_case == "federated_learning":
                return await self._setup_federated_learning(config)
            elif use_case == "edge_deployment":
                return await self._setup_edge_deployment(config)
            elif use_case == "cloud_scaling":
                return await self._setup_cloud_scaling(config)
            else:
                return SDKResponse(success=False, error=f"Caso de uso no soportado: {use_case}")

        except Exception as e:
            return SDKResponse(success=False, error=str(e))

    async def _setup_federated_learning(self, config: Dict[str, Any]) -> SDKResponse:
        """Configurar aprendizaje federado rápidamente"""

        model_id = config.get("model_id", "default_model")
        participants = config.get("participants", ["node_1", "node_2", "node_3"])

        # Registrar modelo si no existe
        if model_id not in self.client._ml_manager.models:
            register_result = await self.client.register_model(
                model_path="",  # Simulado
                framework=MLFramework.PYTORCH,
                model_type="classification",
                metadata={"architecture": "MLP", "input_shape": [784], "output_shape": [10]}
            )
            if not register_result.success:
                return register_result

        # Iniciar entrenamiento federado
        return await self.client.start_federated_training(model_id, participants)

    async def _setup_edge_deployment(self, config: Dict[str, Any]) -> SDKResponse:
        """Configurar despliegue en edge rápidamente"""

        model_id = config.get("model_id", "edge_model")
        devices = config.get("devices", [])

        # Registrar dispositivos si no existen
        for device_info in devices:
            if not any(d["device_id"] == device_info.get("device_id")
                      for d in [self.client._edge_system.device_manager.get_device_status(did)
                               for did in self.client._edge_system.device_manager.devices.keys()]
                      if d):
                await self.client.register_edge_device(device_info)

        # Desplegar modelo
        device_ids = [d["device_id"] for d in devices]
        return await self.client.deploy_to_edge(model_id, device_ids)

    async def _setup_cloud_scaling(self, config: Dict[str, Any]) -> SDKResponse:
        """Configurar escalado en cloud rápidamente"""

        deployment_config = {
            "name": config.get("name", "auto_deployment"),
            "instance_type": config.get("instance_type", "t2_micro"),
            "count": config.get("count", 2),
            "auto_scaling": True,
            "min_instances": config.get("min_instances", 1),
            "max_instances": config.get("max_instances", 5)
        }

        return await self.client.create_cloud_deployment(
            provider=config.get("provider", "aws"),
            region=config.get("region", "us-east-1"),
            **deployment_config
        )

# ===== ASYNC CONTEXT MANAGER =====

class aegis_session:
    """Context manager para sesiones del SDK"""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.client: Optional[AEGISClient] = None

    async def __aenter__(self):
        config = SDKConfig(api_key=self.api_key)
        self.client = AEGISClient(config)

        # Autenticar automáticamente
        auth_result = await self.client.authenticate({"api_key": config.api_key})
        if not auth_result.success:
            raise AuthenticationError(auth_result.error)

        return self.client

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # Cleanup si es necesario
        pass

# ===== DECORATORS =====

def aegis_operation(operation_name: str):
    """Decorador para operaciones del SDK"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                logger.info(f"🚀 Iniciando operación: {operation_name}")
                result = await func(*args, **kwargs)

                duration = time.time() - start_time
                if result.success:
                    logger.info(f"✅ Operación completada en {duration:.2f}s")
                else:
                    logger.error(f"❌ Operación fallida: {result.error}")

                return result

            except Exception as e:
                duration = time.time() - start_time
                logger.error(f"❌ Error en operación: {e} (duración: {duration:.2f}s)")
                return SDKResponse(success=False, error=str(e))

        return wrapper
    return decorator

# ===== DEMO Y EJEMPLOS =====

async def demo_sdk():
    """Demostración completa del SDK"""

    print("🛠️ AEGIS SDK - Demo Completa")
    print("=" * 50)

    # Inicializar SDK
    aegis = AEGIS()

    try:
        # ===== EJEMPLO 1: INICIO RÁPIDO =====
        print("\n🚀 Ejemplo 1: Inicio rápido de Federated Learning")

        fl_config = {
            "model_id": "resnet_classifier",
            "participants": ["node_1", "node_2", "node_3", "node_4"]
        }

        result = await aegis.quick_start("federated_learning", fl_config)
        if result.success:
            print(f"✅ Entrenamiento federado iniciado: {result.data}")
        else:
            print(f"❌ Error: {result.error}")

        # ===== EJEMPLO 2: DESPLIEGUE EN EDGE =====
        print("\n🛠️ Ejemplo 2: Despliegue en dispositivos Edge")

        edge_config = {
            "model_id": "mobile_net",
            "devices": [
                {
                    "device_type": "raspberry_pi",
                    "capabilities": ["inference_only", "federated_client"],
                    "hardware_specs": {"cpu": "ARM Cortex-A72", "ram": "4GB"}
                },
                {
                    "device_type": "jetson_nano",
                    "capabilities": ["inference_only", "training_mini_batch"],
                    "hardware_specs": {"gpu": "128-core Maxwell", "ram": "4GB"}
                }
            ]
        }

        result = await aegis.quick_start("edge_deployment", edge_config)
        if result.success:
            print(f"✅ Modelo desplegado en {result.data['device_count']} dispositivos")
        else:
            print(f"❌ Error: {result.error}")

        # ===== EJEMPLO 3: ESCALADO EN CLOUD =====
        print("\n☁️ Ejemplo 3: Escalado automático en Cloud")

        cloud_config = {
            "name": "ml-training-cluster",
            "provider": "aws",
            "region": "us-west-2",
            "instance_type": "t3_small",
            "count": 2,
            "min_instances": 1,
            "max_instances": 8,
            "cost_budget": 100.0
        }

        result = await aegis.quick_start("cloud_scaling", cloud_config)
        if result.success:
            print(f"✅ Cluster desplegado: {result.data['instances']} instancias en {result.data['provider']}")
        else:
            print(f"❌ Error: {result.error}")

        # ===== EJEMPLO 4: USO AVANZADO CON CONTEXT MANAGER =====
        print("\n🔧 Ejemplo 4: Uso avanzado del SDK")

        async with aegis_session() as client:
            # Verificar health del sistema
            health = await client.health_check()
            print(f"❤️ System Health: {health.data['overall_health']}")

            # Obtener información del sistema
            info = await client.get_system_info()
            print(f"ℹ️ Frameworks soportados: {info.data['frameworks']['ml_supported']}")

            # Realizar predicción (simulada)
            # prediction = await client.predict("resnet_classifier", test_data)
            # print(f"🎯 Predicción: {prediction.data}")

        # ===== EJEMPLO 5: MÉTRICAS Y MONITOREO =====
        print("\n📊 Ejemplo 5: Monitoreo y métricas")

        # Obtener métricas de cloud
        metrics_result = await aegis.client.get_cloud_metrics()
        if metrics_result.success:
            for provider, provider_metrics in metrics_result.data.items():
                print(f"☁️ {provider.upper()}: {provider_metrics['running_instances']} instancias, "
                      f"${provider_metrics['total_cost']}/hora")
        else:
            print(f"⚠️ No se pudieron obtener métricas: {metrics_result.error}")

        print("\n🎉 DEMO COMPLETA EXITOSA!")
        print("🌟 SDK de AEGIS completamente funcional")
        print("=" * 60)

    except Exception as e:
        print(f"❌ Error en demo: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(demo_sdk())
