#!/usr/bin/env python3
"""
🎯 AEGIS Templates - Boilerplates para casos de uso comunes
Plantillas pre-configuradas para acelerar el desarrollo con AEGIS
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, List
import asyncio

# Importar SDK
from aegis_sdk import AEGIS, aegis_session

class AEGISTemplates:
    """Gestor de plantillas para casos de uso comunes"""

    def __init__(self):
        self.templates_dir = Path("./templates")
        self.templates_dir.mkdir(exist_ok=True)
        self.templates = self._load_templates()

    def _load_templates(self) -> Dict[str, Dict[str, Any]]:
        """Cargar plantillas disponibles"""

        return {
            "federated_learning": {
                "name": "Aprendizaje Federado Básico",
                "description": "Template para implementar aprendizaje federado con múltiples participantes",
                "category": "ml",
                "files": {
                    "federated_client.py": self._get_federated_client_template(),
                    "federated_coordinator.py": self._get_federated_coordinator_template(),
                    "config.json": self._get_federated_config_template()
                },
                "requirements": ["pytorch", "numpy"],
                "estimated_time": "2-3 horas"
            },

            "edge_computing": {
                "name": "Computación en Edge",
                "description": "Template para desplegar modelos en dispositivos IoT",
                "category": "edge",
                "files": {
                    "edge_device.py": self._get_edge_device_template(),
                    "model_optimizer.py": self._get_model_optimizer_template(),
                    "deployment_config.json": self._get_edge_deployment_config()
                },
                "requirements": ["tensorflow", "tflite"],
                "estimated_time": "1-2 horas"
            },

            "cloud_deployment": {
                "name": "Despliegue en Cloud",
                "description": "Template para desplegar aplicaciones en múltiples proveedores cloud",
                "category": "cloud",
                "files": {
                    "cloud_deployment.py": self._get_cloud_deployment_template(),
                    "scaling_policy.py": self._get_scaling_policy_template(),
                    "monitoring.py": self._get_monitoring_template()
                },
                "requirements": ["boto3", "google-cloud-compute"],
                "estimated_time": "3-4 horas"
            },

            "hybrid_system": {
                "name": "Sistema Híbrido Completo",
                "description": "Template completo que combina cloud, edge y federated learning",
                "category": "fullstack",
                "files": {
                    "main.py": self._get_hybrid_main_template(),
                    "orchestrator.py": self._get_hybrid_orchestrator_template(),
                    "monitoring.py": self._get_hybrid_monitoring_template(),
                    "config.json": self._get_hybrid_config_template()
                },
                "requirements": ["fastapi", "uvicorn", "pytorch", "tensorflow"],
                "estimated_time": "5-7 horas"
            },

            "ml_pipeline": {
                "name": "Pipeline de ML Distribuido",
                "description": "Template para pipelines de ML con preprocesamiento distribuido",
                "category": "ml",
                "files": {
                    "pipeline.py": self._get_ml_pipeline_template(),
                    "data_processor.py": self._get_data_processor_template(),
                    "model_trainer.py": self._get_model_trainer_template()
                },
                "requirements": ["scikit-learn", "pandas", "pytorch"],
                "estimated_time": "4-5 horas"
            },

            "iot_sensor_network": {
                "name": "Red de Sensores IoT",
                "description": "Template para redes de sensores con procesamiento distribuido",
                "category": "iot",
                "files": {
                    "sensor_node.py": self._get_sensor_node_template(),
                    "data_aggregator.py": self._get_data_aggregator_template(),
                    "analytics.py": self._get_iot_analytics_template()
                },
                "requirements": ["paho-mqtt", "numpy", "matplotlib"],
                "estimated_time": "3-4 horas"
            }
        }

    def list_templates(self, category: str = None) -> List[Dict[str, Any]]:
        """Listar plantillas disponibles"""

        templates = []
        for template_id, template in self.templates.items():
            if category is None or template["category"] == category:
                templates.append({
                    "id": template_id,
                    "name": template["name"],
                    "description": template["description"],
                    "category": template["category"],
                    "files_count": len(template["files"]),
                    "requirements": template["requirements"],
                    "estimated_time": template["estimated_time"]
                })

        return templates

    def get_template_info(self, template_id: str) -> Dict[str, Any]:
        """Obtener información detallada de una plantilla"""

        if template_id not in self.templates:
            raise ValueError(f"Template '{template_id}' no encontrado")

        template = self.templates[template_id]
        return {
            "id": template_id,
            "name": template["name"],
            "description": template["description"],
            "category": template["category"],
            "files": list(template["files"].keys()),
            "requirements": template["requirements"],
            "estimated_time": template["estimated_time"],
            "readme": self._get_template_readme(template_id)
        }

    def generate_project(self, template_id: str, project_name: str,
                        output_dir: str = "./generated_projects") -> str:
        """Generar proyecto desde plantilla"""

        if template_id not in self.templates:
            raise ValueError(f"Template '{template_id}' no encontrado")

        template = self.templates[template_id]

        # Crear directorio del proyecto
        project_dir = Path(output_dir) / project_name
        project_dir.mkdir(parents=True, exist_ok=True)

        # Generar archivos
        for filename, content in template["files"].items():
            file_path = project_dir / filename
            file_path.parent.mkdir(parents=True, exist_ok=True)

            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)

        # Generar archivos adicionales
        self._generate_project_files(template_id, project_dir, project_name)

        return str(project_dir)

    def _generate_project_files(self, template_id: str, project_dir: Path, project_name: str):
        """Generar archivos adicionales del proyecto"""

        # README.md
        readme_content = self._get_template_readme(template_id)
        readme_path = project_dir / "README.md"
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(readme_content)

        # requirements.txt
        template = self.templates[template_id]
        requirements = template["requirements"] + ["aegis-sdk", "rich", "click"]

        req_content = "\n".join(requirements)
        req_path = project_dir / "requirements.txt"
        with open(req_path, 'w', encoding='utf-8') as f:
            f.write(req_content)

        # .gitignore
        gitignore_content = """
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# AEGIS specific
models/
data/
logs/
config/secrets/
*.key
*.pem

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db
        """.strip()

        gitignore_path = project_dir / ".gitignore"
        with open(gitignore_path, 'w', encoding='utf-8') as f:
            f.write(gitignore_content)

    def _get_template_readme(self, template_id: str) -> str:
        """Generar README para una plantilla"""

        template = self.templates[template_id]

        readme = f"""# {template["name"]}

{template["description"]}

## 🚀 Inicio Rápido

```bash
# Instalar dependencias
pip install -r requirements.txt

# Configurar AEGIS
export AEGIS_API_KEY="your-api-key-here"

# Ejecutar
python main.py
```

## 📋 Requisitos

- Python 3.8+
- {", ".join(template["requirements"])}
- AEGIS Framework SDK

## 🏗️ Arquitectura

Este template incluye:
{chr(10).join(f"- {file}" for file in template["files"].keys())}

## 📊 Características

- ✅ Configuración optimizada para producción
- ✅ Monitoreo integrado
- ✅ Manejo de errores robusto
- ✅ Documentación completa

## 🔧 Personalización

1. Modifica `config.json` según tus necesidades
2. Ajusta los parámetros de modelo en los archivos de configuración
3. Personaliza las métricas y logging según tu caso de uso

## 📈 Próximos Pasos

- Configurar credenciales de cloud providers
- Ajustar parámetros de escalado automático
- Implementar métricas personalizadas
- Configurar alertas y monitoreo

## 🆘 Soporte

Para soporte técnico, consulta la [documentación de AEGIS](https://docs.aegis-framework.com) o crea un issue en el repositorio.

---
*Template generado automáticamente por AEGIS Framework*
*Tiempo estimado de implementación: {template["estimated_time"]}*
"""

        return readme

    # ===== TEMPLATE CONTENT METHODS =====

    def _get_federated_client_template(self) -> str:
        """Template para cliente federado"""
        return '''#!/usr/bin/env python3
"""
Cliente Federado - AEGIS Federated Learning
Implementación de cliente para aprendizaje federado distribuido
"""

import asyncio
import json
import time
from typing import Dict, Any, Optional
import logging

from aegis_sdk import aegis_session, SDKResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FederatedClient:
    """Cliente para aprendizaje federado"""

    def __init__(self, client_id: str, node_id: str):
        self.client_id = client_id
        self.node_id = node_id
        self.current_round: Optional[str] = None
        self.local_model = None
        self.training_data = []
        self.is_training = False

    async def connect_to_coordinator(self):
        """Conectar con el coordinador federado"""
        async with aegis_session() as client:
            logger.info(f"🔗 Conectando cliente {self.client_id} al coordinador")

            # Simular datos de entrenamiento locales
            self.training_data = self._load_local_data()

            logger.info(f"📊 Datos locales cargados: {len(self.training_data)} muestras")

    async def participate_in_round(self, round_id: str, model_config: Dict[str, Any]):
        """Participar en una ronda de entrenamiento federado"""
        self.current_round = round_id

        async with aegis_session() as client:
            logger.info(f"🎯 Participando en ronda {round_id}")

            try:
                # Recibir modelo global
                await self._download_global_model(client, model_config)

                # Entrenar modelo localmente
                update_data = await self._train_local_model()

                # Enviar actualización
                success = await self._send_update_to_coordinator(client, update_data)

                if success:
                    logger.info(f"✅ Actualización enviada para ronda {round_id}")
                else:
                    logger.error(f"❌ Error enviando actualización para ronda {round_id}")

            except Exception as e:
                logger.error(f"❌ Error en ronda {round_id}: {e}")

    async def _download_global_model(self, client, model_config: Dict[str, Any]):
        """Descargar modelo global del coordinador"""
        # En una implementación real, aquí se descargaría el modelo
        logger.info("📥 Descargando modelo global...")
        await asyncio.sleep(1)  # Simular descarga
        self.local_model = {"type": "neural_network", "config": model_config}

    async def _train_local_model(self) -> Dict[str, Any]:
        """Entrenar modelo localmente"""
        logger.info("🏋️ Entrenando modelo localmente...")

        self.is_training = True

        try:
            # Simular entrenamiento
            epochs = 5
            for epoch in range(epochs):
                logger.info(f"   Epoch {epoch + 1}/{epochs}")
                await asyncio.sleep(0.5)  # Simular tiempo de entrenamiento

            # Generar actualización simulada
            update_data = {
                "client_id": self.client_id,
                "round_id": self.current_round,
                "weights_delta": {
                    "layer_1": [[0.01 * (i + j) for j in range(10)] for i in range(20)],
                    "layer_2": [[0.005 * (i + j) for j in range(10)] for i in range(10)]
                },
                "sample_count": len(self.training_data),
                "metrics": {
                    "loss": 0.3 + (time.time() % 0.2),
                    "accuracy": 0.85 + (time.time() % 0.1),
                    "val_loss": 0.35 + (time.time() % 0.15),
                    "val_accuracy": 0.82 + (time.time() % 0.08)
                },
                "training_time": 15.5,
                "timestamp": time.time()
            }

            return update_data

        finally:
            self.is_training = False

    async def _send_update_to_coordinator(self, client, update_data: Dict[str, Any]) -> bool:
        """Enviar actualización al coordinador"""
        # En una implementación real, esto enviaría la actualización
        logger.info("📤 Enviando actualización al coordinador...")
        await asyncio.sleep(0.5)  # Simular envío
        return True

    def _load_local_data(self) -> list:
        """Cargar datos de entrenamiento locales"""
        # Simular carga de datos
        return [{"features": [0.1 * i for i in range(784)], "label": i % 10}
                for i in range(1000)]

async def main():
    """Función principal"""
    client = FederatedClient("client_001", "node_alpha")

    logger.info("🚀 Iniciando cliente federado...")

    # Conectar al coordinador
    await client.connect_to_coordinator()

    # Mantener cliente ejecutándose
    logger.info("✅ Cliente federado listo para recibir rondas")

    # En una implementación real, aquí habría un loop que escucha
    # por nuevas rondas del coordinador
    try:
        while True:
            await asyncio.sleep(60)  # Esperar nuevas rondas
    except KeyboardInterrupt:
        logger.info("👋 Cliente federado detenido")

if __name__ == "__main__":
    asyncio.run(main())
'''

    def _get_federated_coordinator_template(self) -> str:
        """Template para coordinador federado"""
        return '''#!/usr/bin/env python3
"""
Coordinador Federado - AEGIS Federated Learning
Orquestador central para aprendizaje federado distribuido
"""

import asyncio
import json
import time
from typing import Dict, List, Any, Set
import logging

from aegis_sdk import AEGIS, SDKResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FederatedCoordinator:
    """Coordinador de aprendizaje federado"""

    def __init__(self, model_id: str, min_clients: int = 3, max_rounds: int = 10):
        self.model_id = model_id
        self.min_clients = min_clients
        self.max_rounds = max_rounds
        self.aegis = AEGIS()

        # Estado
        self.active_round: Optional[Dict[str, Any]] = None
        self.completed_rounds: List[Dict[str, Any]] = []
        self.registered_clients: Set[str] = set()
        self.current_round_number = 0

    async def initialize(self):
        """Inicializar coordinador"""
        logger.info("🎯 Inicializando coordinador federado")

        # Verificar modelo disponible
        async with aegis_session() as client:
            # En una implementación real, verificar que el modelo existe
            logger.info(f"📋 Modelo configurado: {self.model_id}")

    async def register_client(self, client_id: str, client_info: Dict[str, Any]) -> bool:
        """Registrar un nuevo cliente"""
        if client_id in self.registered_clients:
            logger.warning(f"Cliente {client_id} ya registrado")
            return False

        self.registered_clients.add(client_id)
        logger.info(f"✅ Cliente registrado: {client_id} ({client_info.get('samples', 0)} muestras)")

        # Verificar si podemos iniciar primera ronda
        if len(self.registered_clients) >= self.min_clients and not self.active_round:
            await self._start_new_round()

        return True

    async def _start_new_round(self):
        """Iniciar nueva ronda de entrenamiento"""
        if self.current_round_number >= self.max_rounds:
            logger.info("🎉 Todas las rondas completadas")
            return

        self.current_round_number += 1
        round_id = f"round_{self.current_round_number:03d}"

        logger.info(f"🚀 Iniciando ronda {self.current_round_number}")

        # Crear ronda usando SDK
        result = await self.aegis.client.start_federated_training(
            self.model_id,
            list(self.registered_clients)
        )

        if result.success:
            self.active_round = {
                "round_id": round_id,
                "number": self.current_round_number,
                "start_time": time.time(),
                "participants": list(self.registered_clients),
                "collected_updates": 0,
                "required_updates": len(self.registered_clients)
            }

            logger.info(f"✅ Ronda {round_id} iniciada con {len(self.registered_clients)} participantes")
        else:
            logger.error(f"❌ Error iniciando ronda: {result.error}")

    async def receive_client_update(self, client_id: str, update_data: Dict[str, Any]) -> bool:
        """Recibir actualización de un cliente"""
        if not self.active_round:
            logger.warning(f"No hay ronda activa para actualización de {client_id}")
            return False

        if client_id not in self.active_round["participants"]:
            logger.warning(f"Cliente {client_id} no autorizado para esta ronda")
            return False

        # Procesar actualización
        self.active_round["collected_updates"] += 1

        logger.info(f"📥 Actualización recibida de {client_id} "
                   f"({self.active_round['collected_updates']}/{self.active_round['required_updates']})")

        # Verificar si ronda está completa
        if self.active_round["collected_updates"] >= self.active_round["required_updates"]:
            await self._complete_round()

        return True

    async def _complete_round(self):
        """Completar ronda actual"""
        if not self.active_round:
            return

        self.active_round["end_time"] = time.time()
        self.active_round["duration"] = self.active_round["end_time"] - self.active_round["start_time"]

        # Agregar métricas simuladas
        self.active_round["aggregated_metrics"] = {
            "avg_loss": 0.25,
            "avg_accuracy": 0.87,
            "total_samples": sum(range(800, 1200, 100))  # Simulado
        }

        # Mover a rondas completadas
        self.completed_rounds.append(self.active_round)
        completed_round = self.active_round
        self.active_round = None

        logger.info(f"🏁 Ronda {completed_round['round_id']} completada en "
                   f"{completed_round['duration']:.1f}s")

        # Iniciar siguiente ronda después de un delay
        await asyncio.sleep(5)
        await self._start_new_round()

    def get_status(self) -> Dict[str, Any]:
        """Obtener estado del coordinador"""
        return {
            "model_id": self.model_id,
            "active_round": self.active_round["round_id"] if self.active_round else None,
            "current_round_number": self.current_round_number,
            "registered_clients": len(self.registered_clients),
            "completed_rounds": len(self.completed_rounds),
            "max_rounds": self.max_rounds,
            "min_clients": self.min_clients
        }

async def main():
    """Función principal"""
    coordinator = FederatedCoordinator(
        model_id="resnet_classifier",
        min_clients=3,
        max_rounds=5
    )

    logger.info("🎯 Iniciando coordinador federado...")

    await coordinator.initialize()

    # Simular registro de clientes
    clients_info = [
        {"client_id": "client_001", "samples": 1000},
        {"client_id": "client_002", "samples": 1200},
        {"client_id": "client_003", "samples": 800},
        {"client_id": "client_004", "samples": 1500}
    ]

    for client_info in clients_info:
        await coordinator.register_client(
            client_info["client_id"],
            {"samples": client_info["samples"]}
        )
        await asyncio.sleep(1)  # Simular llegada de clientes

    # Mantener coordinador ejecutándose
    logger.info("✅ Coordinador federado operativo")

    try:
        while True:
            status = coordinator.get_status()
            logger.info(f"📊 Estado: Ronda {status['current_round_number']}, "
                       f"Clientes: {status['registered_clients']}, "
                       f"Rondas completadas: {status['completed_rounds']}")

            await asyncio.sleep(30)  # Reporte cada 30 segundos
    except KeyboardInterrupt:
        logger.info("👋 Coordinador federado detenido")

if __name__ == "__main__":
    asyncio.run(main())
'''

    def _get_federated_config_template(self) -> str:
        """Template de configuración para federated learning"""
        return '''{
  "federated_learning": {
    "model_id": "resnet_classifier",
    "coordinator": {
      "host": "coordinator.aegis.local",
      "port": 8443,
      "min_clients": 3,
      "max_rounds": 10,
      "aggregation_timeout": 300
    },
    "clients": [
      {
        "client_id": "client_001",
        "node_id": "node_alpha",
        "data_samples": 1000,
        "hardware": "GPU"
      },
      {
        "client_id": "client_002",
        "node_id": "node_beta",
        "data_samples": 1200,
        "hardware": "CPU"
      },
      {
        "client_id": "client_003",
        "node_id": "node_gamma",
        "data_samples": 800,
        "hardware": "TPU"
      }
    ],
    "training": {
      "epochs_per_round": 1,
      "batch_size": 32,
      "learning_rate": 0.01,
      "optimizer": "adam",
      "loss_function": "categorical_crossentropy",
      "metrics": ["accuracy", "precision", "recall"]
    },
    "privacy": {
      "differential_privacy": true,
      "noise_multiplier": 0.1,
      "max_grad_norm": 1.0
    },
    "network": {
      "heartbeat_interval": 30,
      "connection_timeout": 60,
      "max_retries": 3
    }
  }
}'''

    def _get_edge_device_template(self) -> str:
        """Template para dispositivo edge"""
        return '''#!/usr/bin/env python3
"""
Dispositivo Edge - AEGIS Edge Computing
Implementación de dispositivo IoT con capacidades de inferencia
"""

import asyncio
import json
import time
import psutil
from typing import Dict, Any, Optional
import logging

from aegis_sdk import aegis_session, SDKResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EdgeDevice:
    """Dispositivo IoT con capacidades de ML"""

    def __init__(self, device_id: str, device_type: str, capabilities: list):
        self.device_id = device_id
        self.device_type = device_type
        self.capabilities = capabilities
        self.deployed_models: Dict[str, Any] = {}
        self.is_running = False
        self.last_heartbeat = time.time()

    async def initialize(self):
        """Inicializar dispositivo edge"""
        logger.info(f"🔧 Inicializando dispositivo edge: {self.device_id}")

        async with aegis_session() as client:
            # Registrar dispositivo en la plataforma
            device_info = {
                "device_type": self.device_type,
                "capabilities": self.capabilities,
                "hardware_specs": self._get_hardware_specs(),
                "location": {"lat": 40.7128, "lon": -74.0060},  # Simulado
                "firmware_version": "1.0.0"
            }

            result = await client.register_edge_device(device_info)

            if result.success:
                logger.info(f"✅ Dispositivo registrado: {result.data['device_id']}")
                self.is_running = True
            else:
                logger.error(f"❌ Error registrando dispositivo: {result.error}")

    async def run_device_loop(self):
        """Loop principal del dispositivo"""
        logger.info("🚀 Iniciando loop del dispositivo edge")

        while self.is_running:
            try:
                await self._heartbeat()
                await self._process_sensor_data()
                await self._run_inference_tasks()

                # Esperar antes del siguiente ciclo
                await asyncio.sleep(30)  # Ciclo cada 30 segundos

            except Exception as e:
                logger.error(f"❌ Error en loop del dispositivo: {e}")
                await asyncio.sleep(10)

    async def _heartbeat(self):
        """Enviar heartbeat a la plataforma"""
        async with aegis_session() as client:
            status = self._get_device_status()
            # En una implementación real, enviar heartbeat
            logger.debug(f"💓 Heartbeat enviado: {self.device_id}")

    async def _process_sensor_data(self):
        """Procesar datos de sensores locales"""
        # Simular lectura de sensores
        sensor_data = {
            "temperature": 25.5 + (time.time() % 5),
            "humidity": 60.2 + (time.time() % 10),
            "motion_detected": (time.time() % 60) < 5,  # Movimiento cada minuto
            "light_level": 800 + int(time.time() % 200)
        }

        # Procesar con modelos desplegados si hay datos relevantes
        for model_id, model_info in self.deployed_models.items():
            if "inference_only" in self.capabilities:
                await self._run_inference(model_id, sensor_data)

    async def _run_inference(self, model_id: str, input_data: Dict[str, Any]):
        """Ejecutar inferencia con un modelo"""
        async with aegis_session() as client:
            # Preparar datos de entrada
            # En una implementación real, aquí se haría preprocesamiento
            processed_data = [list(input_data.values())]

            # Ejecutar predicción
            result = await client.predict(model_id, processed_data)

            if result.success:
                prediction = result.data.get("prediction", [])
                logger.info(f"🎯 Inferencia {model_id}: {prediction}")

                # Enviar resultados a la nube si es necesario
                await self._send_results_to_cloud(prediction)
            else:
                logger.error(f"❌ Error en inferencia: {result.error}")

    async def _send_results_to_cloud(self, results: Any):
        """Enviar resultados a la nube"""
        # En una implementación real, enviar vía MQTT, HTTP, etc.
        logger.debug(f"☁️ Enviando resultados a la nube: {results}")

    async def _run_inference_tasks(self):
        """Ejecutar tareas programadas de inferencia"""
        # Verificar si hay tareas federadas pendientes
        if "federated_client" in self.capabilities:
            await self._check_federated_tasks()

    async def _check_federated_tasks(self):
        """Verificar tareas de aprendizaje federado"""
        # En una implementación real, consultar con el coordinador
        logger.debug("🤝 Verificando tareas federadas...")

    def _get_hardware_specs(self) -> Dict[str, Any]:
        """Obtener especificaciones de hardware"""
        return {
            "cpu_cores": psutil.cpu_count(),
            "memory_gb": psutil.virtual_memory().total / (1024**3),
            "storage_gb": psutil.disk_usage('/').total / (1024**3),
            "cpu_freq_mhz": psutil.cpu_freq().max if psutil.cpu_freq() else 0,
            "platform": "edge_device"
        }

    def _get_device_status(self) -> Dict[str, Any]:
        """Obtener estado actual del dispositivo"""
        return {
            "device_id": self.device_id,
            "status": "online" if self.is_running else "offline",
            "battery_level": 85.5,  # Simulado
            "temperature": 35.2,   # Simulado
            "network_quality": 0.9,  # Simulado
            "uptime_seconds": time.time() - self.last_heartbeat,
            "deployed_models_count": len(self.deployed_models)
        }

    async def shutdown(self):
        """Apagar dispositivo"""
        logger.info(f"🛑 Apagando dispositivo: {self.device_id}")
        self.is_running = False

async def main():
    """Función principal"""
    device = EdgeDevice(
        device_id="edge_sensor_001",
        device_type="raspberry_pi",
        capabilities=["inference_only", "data_collection", "federated_client"]
    )

    logger.info("🚀 Iniciando dispositivo edge...")

    try:
        # Inicializar
        await device.initialize()

        # Ejecutar loop principal
        await device.run_device_loop()

    except KeyboardInterrupt:
        logger.info("👋 Dispositivo edge detenido por usuario")
    except Exception as e:
        logger.error(f"❌ Error fatal en dispositivo: {e}")
    finally:
        await device.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
'''

    def _get_model_optimizer_template(self) -> str:
        """Template para optimización de modelos"""
        return '''#!/usr/bin/env python3
"""
Optimizador de Modelos - AEGIS Edge Computing
Herramientas para optimizar modelos de ML para dispositivos edge
"""

import asyncio
import time
from typing import Dict, Any, Optional
import logging

from aegis_sdk import aegis_session, SDKResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelOptimizer:
    """Optimizador de modelos para edge"""

    def __init__(self):
        self.optimization_profiles = {
            "quantization": {
                "description": "Quantization-aware training",
                "accuracy_drop": 0.02,
                "size_reduction": 0.75,
                "speed_improvement": 2.5
            },
            "pruning": {
                "description": "Weight pruning",
                "accuracy_drop": 0.01,
                "size_reduction": 0.6,
                "speed_improvement": 1.8
            },
            "distillation": {
                "description": "Knowledge distillation",
                "accuracy_drop": 0.03,
                "size_reduction": 0.8,
                "speed_improvement": 2.2
            }
        }

    async def optimize_model(self, model_id: str, target_device: str,
                           optimization_type: str = "quantization") -> Dict[str, Any]:
        """Optimizar modelo para dispositivo edge"""

        logger.info(f"🔧 Optimizando modelo {model_id} para {target_device} usando {optimization_type}")

        if optimization_type not in self.optimization_profiles:
            raise ValueError(f"Tipo de optimización no soportado: {optimization_type}")

        profile = self.optimization_profiles[optimization_type]

        # Simular proceso de optimización
        await asyncio.sleep(5)  # Simular tiempo de optimización

        optimized_model = {
            "original_model_id": model_id,
            "optimized_model_id": f"{model_id}_{target_device}_{optimization_type}",
            "target_device": target_device,
            "optimization_type": optimization_type,
            "metrics": {
                "accuracy_drop": profile["accuracy_drop"],
                "size_reduction": profile["size_reduction"],
                "speed_improvement": profile["speed_improvement"],
                "power_savings": 0.3  # Simulado
            },
            "created_at": time.time(),
            "status": "optimized"
        }

        logger.info(f"✅ Modelo optimizado: {optimized_model['optimized_model_id']}")
        logger.info(f"   📊 Reducción de tamaño: {profile['size_reduction']*100:.1f}%")
        logger.info(f"   ⚡ Mejora de velocidad: {profile['speed_improvement']:.1f}x")

        return optimized_model

    async def compare_optimizations(self, model_id: str, target_device: str) -> Dict[str, Any]:
        """Comparar diferentes técnicas de optimización"""

        logger.info(f"📊 Comparando optimizaciones para {model_id} en {target_device}")

        results = {}
        for opt_type in self.optimization_profiles.keys():
            result = await self.optimize_model(model_id, target_device, opt_type)
            results[opt_type] = result["metrics"]

        # Recomendar mejor opción (trade-off entre accuracy y performance)
        recommendations = self._analyze_tradeoffs(results)

        return {
            "model_id": model_id,
            "target_device": target_device,
            "optimization_results": results,
            "recommendations": recommendations
        }

    def _analyze_tradeoffs(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analizar trade-offs entre optimizaciones"""

        # Calcular scores (simplificado)
        scores = {}
        for opt_type, metrics in results.items():
            # Score = (size_reduction * 0.4) + (speed_improvement * 0.4) - (accuracy_drop * 0.2)
            score = (metrics["size_reduction"] * 0.4 +
                    metrics["speed_improvement"] * 0.4 -
                    metrics["accuracy_drop"] * 0.2)
            scores[opt_type] = score

        # Encontrar mejor opción
        best_option = max(scores, key=scores.get)

        return {
            "best_option": best_option,
            "best_score": scores[best_option],
            "all_scores": scores,
            "recommendation": f"Usar {best_option} para el mejor balance entre performance y accuracy"
        }

    async def deploy_optimized_model(self, optimized_model: Dict[str, Any],
                                   device_ids: list) -> Dict[str, Any]:
        """Desplegar modelo optimizado en dispositivos"""

        logger.info(f"🚀 Desplegando modelo optimizado en {len(device_ids)} dispositivos")

        deployment_results = {}

        async with aegis_session() as client:
            for device_id in device_ids:
                try:
                    result = await client.deploy_to_edge(
                        optimized_model["optimized_model_id"],
                        [device_id]
                    )

                    deployment_results[device_id] = {
                        "success": result.success,
                        "error": result.error if not result.success else None
                    }

                    if result.success:
                        logger.info(f"✅ Desplegado en {device_id}")
                    else:
                        logger.error(f"❌ Error en {device_id}: {result.error}")

                except Exception as e:
                    deployment_results[device_id] = {
                        "success": False,
                        "error": str(e)
                    }

        successful_deployments = sum(1 for r in deployment_results.values() if r["success"])

        return {
            "optimized_model_id": optimized_model["optimized_model_id"],
            "total_devices": len(device_ids),
            "successful_deployments": successful_deployments,
            "failed_deployments": len(device_ids) - successful_deployments,
            "deployment_results": deployment_results
        }

async def main():
    """Función principal"""
    optimizer = ModelOptimizer()

    logger.info("🔧 Iniciando optimizador de modelos...")

    # Ejemplo de optimización
    model_id = "resnet_classifier"
    target_device = "raspberry_pi"

    try:
        # Comparar optimizaciones
        logger.info("📊 Comparando técnicas de optimización...")
        comparison = await optimizer.compare_optimizations(model_id, target_device)

        print("\n🏆 RESULTADOS DE COMPARACIÓN:")
        for opt_type, metrics in comparison["optimization_results"].items():
            print(f"   • {opt_type.upper()}:")
            print(f"     - Reducción tamaño: {metrics['size_reduction']*100:.1f}%")
            print(f"     - Mejora velocidad: {metrics['speed_improvement']:.1f}x")
            print(f"     - Caída accuracy: {metrics['accuracy_drop']*100:.1f}%")

        print(f"\n💡 RECOMENDACIÓN: {comparison['recommendations']['recommendation']}")

        # Optimizar con la mejor opción
        best_option = comparison["recommendations"]["best_option"]
        logger.info(f"🎯 Optimizando con {best_option}...")

        optimized_model = await optimizer.optimize_model(model_id, target_device, best_option)

        # Desplegar en dispositivos de ejemplo
        device_ids = ["edge_rpi_001", "edge_rpi_002", "edge_rpi_003"]
        deployment_result = await optimizer.deploy_optimized_model(optimized_model, device_ids)

        print(f"\n🚀 RESULTADOS DE DESPLIEGUE:")
        print(f"   • Modelo: {deployment_result['optimized_model_id']}")
        print(f"   • Dispositivos objetivo: {deployment_result['total_devices']}")
        print(f"   • Despliegues exitosos: {deployment_result['successful_deployments']}")
        print(f"   • Despliegues fallidos: {deployment_result['failed_deployments']}")

        logger.info("✅ Optimización y despliegue completados")

    except Exception as e:
        logger.error(f"❌ Error en optimización: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
'''

    def _get_edge_deployment_config(self) -> str:
        """Configuración de despliegue para edge"""
        return '''{
  "edge_deployment": {
    "target_devices": [
      {
        "device_type": "raspberry_pi",
        "count": 5,
        "capabilities": ["inference_only", "data_collection"]
      },
      {
        "device_type": "jetson_nano",
        "count": 2,
        "capabilities": ["inference_only", "training_mini_batch", "federated_client"]
      },
      {
        "device_type": "coral_dev_board",
        "count": 3,
        "capabilities": ["inference_only", "real_time_processing"]
      }
    ],
    "models": [
      {
        "model_id": "mobilenet_v2",
        "optimization": "quantization",
        "target_devices": ["raspberry_pi", "mobile_phone"]
      },
      {
        "model_id": "efficientnet_lite",
        "optimization": "tflite",
        "target_devices": ["coral_dev_board"]
      },
      {
        "model_id": "yolo_tiny",
        "optimization": "tensorrt",
        "target_devices": ["jetson_nano"]
      }
    ],
    "deployment_strategy": {
      "batch_size": 10,
      "timeout_seconds": 300,
      "retry_attempts": 3,
      "rollback_on_failure": true
    },
    "monitoring": {
      "metrics_interval_seconds": 60,
      "health_check_interval_seconds": 30,
      "alert_on_failure": true
    },
    "network": {
      "protocol": "mqtt",
      "broker_host": "edge.aegis.local",
      "broker_port": 1883,
      "keep_alive_seconds": 60
    }
  }
}'''

    def _get_cloud_deployment_template(self) -> str:
        """Template para despliegue en cloud"""
        return '''#!/usr/bin/env python3
"""
Despliegue en Cloud - AEGIS Multi-Cloud Orchestration
Automatización de despliegues en múltiples proveedores cloud
"""

import asyncio
import json
import time
from typing import Dict, List, Any, Optional
import logging

from aegis_sdk import AEGIS, SDKResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CloudDeploymentManager:
    """Gestor de despliegues en cloud"""

    def __init__(self):
        self.aegis = AEGIS()
        self.deployments: Dict[str, Dict[str, Any]] = {}

    async def create_multi_cloud_deployment(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Crear despliegue multi-cloud"""

        deployment_name = config["name"]
        logger.info(f"☁️ Creando despliegue multi-cloud: {deployment_name}")

        results = {}

        # Desplegar en cada provider configurado
        for provider_config in config["providers"]:
            provider_name = provider_config["name"]
            region = provider_config["region"]
            instance_config = provider_config["instances"]

            logger.info(f"🚀 Desplegando en {provider_name} ({region})...")

            result = await self.aegis.client.create_cloud_deployment(
                name=f"{deployment_name}_{provider_name}",
                provider=provider_name,
                region=region,
                instance_config=instance_config
            )

            results[provider_name] = {
                "success": result.success,
                "deployment_id": result.data.get("deployment_id") if result.success else None,
                "error": result.error if not result.success else None
            }

            if result.success:
                logger.info(f"✅ Despliegue exitoso en {provider_name}")
            else:
                logger.error(f"❌ Error en {provider_name}: {result.error}")

        # Resumen final
        successful_deployments = sum(1 for r in results.values() if r["success"])
        total_deployments = len(results)

        return {
            "deployment_name": deployment_name,
            "total_providers": total_deployments,
            "successful_deployments": successful_deployments,
            "failed_deployments": total_deployments - successful_deployments,
            "results": results
        }

    async def setup_load_balancer(self, deployment_ids: List[str],
                                load_balancer_config: Dict[str, Any]) -> Dict[str, Any]:
        """Configurar load balancer para múltiples despliegues"""

        logger.info("⚖️ Configurando load balancer...")

        # En una implementación real, esto crearía un load balancer
        # que distribuya tráfico entre los despliegues

        lb_config = {
            "name": load_balancer_config.get("name", "aegis_lb"),
            "type": load_balancer_config.get("type", "application"),
            "backends": deployment_ids,
            "health_check": {
                "path": "/health",
                "interval": 30,
                "timeout": 5,
                "healthy_threshold": 2,
                "unhealthy_threshold": 3
            },
            "ssl_config": load_balancer_config.get("ssl", {}),
            "routing_rules": load_balancer_config.get("rules", [])
        }

        logger.info(f"✅ Load balancer configurado: {lb_config['name']}")

        return {
            "load_balancer_id": f"lb_{int(time.time())}",
            "config": lb_config,
            "status": "active"
        }

    async def setup_auto_scaling(self, deployment_id: str,
                               scaling_config: Dict[str, Any]) -> Dict[str, Any]:
        """Configurar auto-scaling para un despliegue"""

        logger.info(f"📈 Configurando auto-scaling para {deployment_id}")

        # Configuración de auto-scaling
        asg_config = {
            "min_instances": scaling_config.get("min_instances", 1),
            "max_instances": scaling_config.get("max_instances", 10),
            "desired_capacity": scaling_config.get("desired_capacity", 3),
            "scaling_policies": [
                {
                    "name": "cpu_based_scale_out",
                    "metric": "CPUUtilization",
                    "target_value": scaling_config.get("cpu_target", 70.0),
                    "scale_out_cooldown": 300,
                    "scale_in_cooldown": 300
                },
                {
                    "name": "request_based_scale_out",
                    "metric": "RequestCountPerTarget",
                    "target_value": scaling_config.get("request_target", 1000),
                    "scale_out_cooldown": 180,
                    "scale_in_cooldown": 180
                }
            ]
        }

        logger.info(f"✅ Auto-scaling configurado: {asg_config['min_instances']}-{asg_config['max_instances']} instancias")

        return {
            "auto_scaling_group_id": f"asg_{deployment_id}",
            "config": asg_config,
            "status": "active"
        }

    async def setup_monitoring(self, deployment_ids: List[str],
                             monitoring_config: Dict[str, Any]) -> Dict[str, Any]:
        """Configurar monitoreo para despliegues"""

        logger.info("📊 Configurando monitoreo multi-cloud...")

        monitoring_setup = {
            "dashboards": [],
            "alerts": [],
            "metrics": []
        }

        # Crear dashboard por provider
        for deployment_id in deployment_ids:
            dashboard = {
                "name": f"deployment_{deployment_id}",
                "metrics": [
                    "cpu_utilization",
                    "memory_utilization",
                    "network_in",
                    "network_out",
                    "request_count",
                    "error_rate"
                ],
                "charts": [
                    {"type": "line", "metrics": ["cpu_utilization", "memory_utilization"]},
                    {"type": "area", "metrics": ["network_in", "network_out"]},
                    {"type": "bar", "metrics": ["request_count"]},
                    {"type": "gauge", "metrics": ["error_rate"]}
                ]
            }
            monitoring_setup["dashboards"].append(dashboard)

        # Configurar alertas
        alerts = [
            {
                "name": "high_cpu_alert",
                "metric": "cpu_utilization",
                "condition": ">",
                "threshold": monitoring_config.get("cpu_alert_threshold", 80.0),
                "duration": "5m",
                "channels": ["email", "slack"]
            },
            {
                "name": "high_error_rate_alert",
                "metric": "error_rate",
                "condition": ">",
                "threshold": monitoring_config.get("error_alert_threshold", 5.0),
                "duration": "2m",
                "channels": ["email", "pagerduty"]
            }
        ]
        monitoring_setup["alerts"] = alerts

        logger.info(f"✅ Monitoreo configurado: {len(monitoring_setup['dashboards'])} dashboards, {len(alerts)} alertas")

        return monitoring_setup

    async def deploy_application_stack(self, stack_config: Dict[str, Any]) -> Dict[str, Any]:
        """Desplegar stack completo de aplicación"""

        logger.info("🏗️ Desplegando stack completo de aplicación...")

        stack_name = stack_config["name"]

        try:
            # 1. Crear despliegues en cloud
            deployment_results = await self.create_multi_cloud_deployment(stack_config["cloud"])

            # 2. Configurar load balancer
            deployment_ids = [r["deployment_id"] for r in deployment_results["results"].values()
                            if r.get("deployment_id")]
            lb_result = await self.setup_load_balancer(deployment_ids, stack_config["load_balancer"])

            # 3. Configurar auto-scaling
            scaling_results = []
            for provider, result in deployment_results["results"].items():
                if result["success"]:
                    asg_result = await self.setup_auto_scaling(
                        result["deployment_id"],
                        stack_config["auto_scaling"]
                    )
                    scaling_results.append(asg_result)

            # 4. Configurar monitoreo
            monitoring_result = await self.setup_monitoring(
                deployment_ids,
                stack_config["monitoring"]
            )

            # Resultado final
            return {
                "stack_name": stack_name,
                "status": "deployed",
                "components": {
                    "cloud_deployments": deployment_results,
                    "load_balancer": lb_result,
                    "auto_scaling": scaling_results,
                    "monitoring": monitoring_result
                },
                "endpoints": {
                    "load_balancer_url": f"https://{stack_name}.aegis.cloud",
                    "api_url": f"https://api.{stack_name}.aegis.cloud",
                    "monitoring_url": f"https://monitoring.{stack_name}.aegis.cloud"
                }
            }

        except Exception as e:
            logger.error(f"❌ Error desplegando stack: {e}")
            return {
                "stack_name": stack_name,
                "status": "failed",
                "error": str(e)
            }

async def main():
    """Función principal"""
    manager = CloudDeploymentManager()

    logger.info("☁️ Iniciando gestor de despliegues cloud...")

    # Configuración de ejemplo para stack completo
    stack_config = {
        "name": "ml-serving-platform",
        "cloud": {
            "providers": [
                {
                    "name": "aws",
                    "region": "us-east-1",
                    "instances": {
                        "instance_type": "t3.medium",
                        "count": 3,
                        "auto_scaling": True,
                        "min_instances": 2,
                        "max_instances": 8
                    }
                },
                {
                    "name": "gcp",
                    "region": "us-central1",
                    "instances": {
                        "instance_type": "e2-standard-2",
                        "count": 2,
                        "auto_scaling": True,
                        "min_instances": 1,
                        "max_instances": 5
                    }
                }
            ]
        },
        "load_balancer": {
            "name": "ml-serving-lb",
            "type": "application",
            "ssl": {"certificate": "letsencrypt"}
        },
        "auto_scaling": {
            "min_instances": 2,
            "max_instances": 10,
            "cpu_target": 70.0,
            "request_target": 1000
        },
        "monitoring": {
            "cpu_alert_threshold": 80.0,
            "error_alert_threshold": 5.0
        }
    }

    try:
        # Desplegar stack completo
        result = await manager.deploy_application_stack(stack_config)

        if result["status"] == "deployed":
            print("
🎉 STACK DESPLEGADO EXITOSAMENTE!"            print(f"   📦 Nombre: {result['stack_name']}")
            print(f"   🌐 Load Balancer: {result['endpoints']['load_balancer_url']}")
            print(f"   🔗 API: {result['endpoints']['api_url']}")
            print(f"   📊 Monitoreo: {result['endpoints']['monitoring_url']}")

            # Detalles de despliegues
            cloud_deploy = result['components']['cloud_deployments']
            print("
🏗️ DETALLES DE DESPLIEGUES:"            print(f"   • Proveedores: {cloud_deploy['total_providers']}")
            print(f"   • Despliegues exitosos: {cloud_deploy['successful_deployments']}")
            print(f"   • Despliegues fallidos: {cloud_deploy['failed_deployments']}")

            logger.info("✅ Despliegue multi-cloud completado")
        else:
            print(f"❌ Error en despliegue: {result.get('error', 'Unknown error')}")

    except Exception as e:
        logger.error(f"❌ Error en despliegue cloud: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
'''

    def _get_scaling_policy_template(self) -> str:
        """Template para políticas de escalado"""
        return '''#!/usr/bin/env python3
"""
Políticas de Escalado - AEGIS Cloud Orchestration
Implementación de estrategias de auto-scaling inteligentes
"""

import asyncio
import time
from typing import Dict, List, Any, Optional
from enum import Enum
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ScalingMetric(Enum):
    CPU_UTILIZATION = "cpu_utilization"
    MEMORY_UTILIZATION = "memory_utilization"
    NETWORK_IN = "network_in"
    NETWORK_OUT = "network_out"
    REQUEST_COUNT = "request_count"
    RESPONSE_TIME = "response_time"
    ERROR_RATE = "error_rate"
    CUSTOM_METRIC = "custom_metric"

class ScalingAction(Enum):
    SCALE_OUT = "scale_out"
    SCALE_IN = "scale_in"
    NO_ACTION = "no_action"

class IntelligentAutoScaler:
    """Auto-scaler inteligente con múltiples estrategias"""

    def __init__(self, deployment_id: str):
        self.deployment_id = deployment_id
        self.policies: List[Dict[str, Any]] = []
        self.metrics_history: List[Dict[str, Any]] = []
        self.last_scaling_action = 0
        self.cooldown_period = 300  # 5 minutos

    def add_policy(self, metric: ScalingMetric, threshold: float,
                  operator: str, action: ScalingAction,
                  cooldown: int = 300, duration: int = 300) -> None:
        """Agregar política de escalado"""

        policy = {
            "id": f"policy_{len(self.policies)}",
            "metric": metric.value,
            "threshold": threshold,
            "operator": operator,
            "action": action.value,
            "cooldown": cooldown,
            "duration": duration,
            "last_triggered": 0,
            "trigger_count": 0
        }

        self.policies.append(policy)
        logger.info(f"✅ Política agregada: {metric.value} {operator} {threshold} -> {action.value}")

    async def evaluate_scaling(self, current_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluar si es necesario escalar"""

        # Agregar métricas al historial
        self.metrics_history.append({
            "timestamp": time.time(),
            "metrics": current_metrics
        })

        # Mantener solo últimas 100 entradas
        if len(self.metrics_history) > 100:
            self.metrics_history = self.metrics_history[-100:]

        # Evaluar cada política
        recommended_actions = []

        for policy in self.policies:
            action = await self._evaluate_policy(policy, current_metrics)
            if action:
                recommended_actions.append({
                    "policy_id": policy["id"],
                    "action": action,
                    "reason": f"{policy['metric']} {policy['operator']} {policy['threshold']}"
                })

        # Decidir acción final (priorizar scale_out sobre scale_in)
        final_action = self._consolidate_actions(recommended_actions)

        result = {
            "deployment_id": self.deployment_id,
            "current_metrics": current_metrics,
            "evaluated_policies": len(self.policies),
            "recommended_actions": recommended_actions,
            "final_action": final_action,
            "cooldown_remaining": max(0, self.cooldown_period - (time.time() - self.last_scaling_action))
        }

        if final_action != ScalingAction.NO_ACTION.value:
            await self._execute_scaling(final_action, result)

        return result

    async def _evaluate_policy(self, policy: Dict[str, Any],
                             metrics: Dict[str, Any]) -> Optional[str]:
        """Evaluar una política individual"""

        metric_value = metrics.get(policy["metric"])
        if metric_value is None:
            return None

        threshold = policy["threshold"]
        operator = policy["operator"]

        # Verificar condición
        condition_met = False
        if operator == ">":
            condition_met = metric_value > threshold
        elif operator == "<":
            condition_met = metric_value < threshold
        elif operator == ">=":
            condition_met = metric_value >= threshold
        elif operator == "<=":
            condition_met = metric_value <= threshold

        if condition_met:
            # Verificar cooldown
            time_since_last_trigger = time.time() - policy["last_triggered"]
            if time_since_last_trigger >= policy["cooldown"]:
                policy["last_triggered"] = time.time()
                policy["trigger_count"] += 1
                return policy["action"]

        return None

    def _consolidate_actions(self, actions: List[Dict[str, Any]]) -> str:
        """Consolidar múltiples acciones recomendadas"""

        if not actions:
            return ScalingAction.NO_ACTION.value

        # Contar acciones por tipo
        action_counts = {}
        for action in actions:
            action_type = action["action"]
            action_counts[action_type] = action_counts.get(action_type, 0) + 1

        # Priorizar: scale_out > no_action > scale_in
        if ScalingAction.SCALE_OUT.value in action_counts:
            return ScalingAction.SCALE_OUT.value
        elif ScalingAction.SCALE_IN.value in action_counts:
            return ScalingAction.SCALE_IN.value
        else:
            return ScalingAction.NO_ACTION.value

    async def _execute_scaling(self, action: str, evaluation_result: Dict[str, Any]) -> None:
        """Ejecutar acción de escalado"""

        # Verificar cooldown global
        time_since_last_action = time.time() - self.last_scaling_action
        if time_since_last_action < self.cooldown_period:
            logger.info(f"⏳ Cooldown activo, saltando escalado (restan {self.cooldown_period - time_since_last_action:.0f}s)")
            return

        logger.info(f"🔄 Ejecutando escalado: {action} para {self.deployment_id}")

        # En una implementación real, aquí se llamaría a la API de cloud
        if action == ScalingAction.SCALE_OUT.value:
            await self._scale_out()
        elif action == ScalingAction.SCALE_IN.value:
            await self._scale_in()

        self.last_scaling_action = time.time()

    async def _scale_out(self) -> None:
        """Escalar hacia afuera (agregar instancias)"""
        logger.info("📈 Escalando hacia afuera...")
        # Simular escalado
        await asyncio.sleep(2)
        logger.info("✅ Nueva instancia agregada")

    async def _scale_in(self) -> None:
        """Escalar hacia adentro (remover instancias)"""
        logger.info("📉 Escalando hacia adentro...")
        # Simular reducción
        await asyncio.sleep(1)
        logger.info("✅ Instancia removida")

    def get_scaling_statistics(self) -> Dict[str, Any]:
        """Obtener estadísticas de escalado"""

        total_triggers = sum(p["trigger_count"] for p in self.policies)
        active_policies = sum(1 for p in self.policies if p["trigger_count"] > 0)

        return {
            "deployment_id": self.deployment_id,
            "total_policies": len(self.policies),
            "active_policies": active_policies,
            "total_triggers": total_triggers,
            "last_scaling_action": self.last_scaling_action,
            "metrics_history_size": len(self.metrics_history),
            "cooldown_period": self.cooldown_period
        }

class PredictiveAutoScaler:
    """Auto-scaler predictivo usando machine learning"""

    def __init__(self, deployment_id: str):
        self.deployment_id = deployment_id
        self.historical_patterns: List[Dict[str, Any]] = []
        self.prediction_model = None  # En producción, sería un modelo ML

    async def analyze_patterns(self, metrics_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analizar patrones históricos para predicción"""

        logger.info("🔮 Analizando patrones para escalado predictivo...")

        # Análisis simplificado (en producción usaría ML real)
        peak_hours = []
        low_hours = []

        for entry in metrics_history[-168:]:  # Últimos 7 días (asumiendo datos por hora)
            hour = time.localtime(entry["timestamp"]).tm_hour
            cpu = entry["metrics"].get("cpu_utilization", 0)

            if cpu > 80:
                peak_hours.append(hour)
            elif cpu < 30:
                low_hours.append(hour)

        # Encontrar patrones
        peak_pattern = max(set(peak_hours), key=peak_hours.count) if peak_hours else None
        low_pattern = max(set(low_hours), key=low_hours.count) if low_hours else None

        return {
            "peak_hour": peak_pattern,
            "low_hour": low_pattern,
            "confidence": min(len(metrics_history) / 168, 1.0),  # Confianza basada en datos disponibles
            "recommendations": [
                f"Pre-escalar a las {peak_pattern}:00 si patrones históricos lo indican" if peak_pattern else "No hay patrones de peak claros",
                f"Reducir capacidad a las {low_pattern}:00 para optimizar costos" if low_pattern else "No hay patrones de bajo uso claros"
            ]
        }

    async def predict_and_scale(self, current_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Predecir demanda y escalar proactivamente"""

        # Obtener análisis de patrones
        patterns = await self.analyze_patterns(self.historical_patterns)

        current_hour = time.localtime().tm_hour
        prediction = {
            "current_hour": current_hour,
            "predicted_load": "medium",
            "recommended_action": ScalingAction.NO_ACTION.value,
            "confidence": patterns["confidence"],
            "reasoning": []
        }

        # Lógica predictiva simplificada
        if patterns["peak_hour"] is not None:
            hours_until_peak = (patterns["peak_hour"] - current_hour) % 24
            if hours_until_peak <= 2:  # 2 horas antes del peak
                prediction["predicted_load"] = "high"
                prediction["recommended_action"] = ScalingAction.SCALE_OUT.value
                prediction["reasoning"].append(f"Peak esperado en {hours_until_peak} horas")

        if patterns["low_hour"] is not None:
            hours_until_low = (patterns["low_hour"] - current_hour) % 24
            if hours_until_low <= 1:  # 1 hora antes del bajo uso
                prediction["predicted_load"] = "low"
                prediction["recommended_action"] = ScalingAction.SCALE_IN.value
                prediction["reasoning"].append(f"Bajo uso esperado en {hours_until_low} horas")

        # Agregar a patrones históricos
        self.historical_patterns.append({
            "timestamp": time.time(),
            "metrics": current_metrics,
            "prediction": prediction
        })

        # Mantener solo últimos 1000 registros
        if len(self.historical_patterns) > 1000:
            self.historical_patterns = self.historical_patterns[-1000:]

        return prediction

async def demo_auto_scaling():
    """Demostración de auto-scaling inteligente"""

    print("📈 DEMO - AEGIS Intelligent Auto-Scaling")
    print("=" * 50)

    # Crear auto-scaler inteligente
    scaler = IntelligentAutoScaler("demo_deployment")

    # Agregar políticas de escalado
    scaler.add_policy(ScalingMetric.CPU_UTILIZATION, 75.0, ">", ScalingAction.SCALE_OUT)
    scaler.add_policy(ScalingMetric.CPU_UTILIZATION, 25.0, "<", ScalingAction.SCALE_IN)
    scaler.add_policy(ScalingMetric.MEMORY_UTILIZATION, 80.0, ">", ScalingAction.SCALE_OUT)
    scaler.add_policy(ScalingMetric.REQUEST_COUNT, 1000, ">", ScalingAction.SCALE_OUT, cooldown=180)

    print("✅ Políticas de escalado configuradas:")
    for policy in scaler.policies:
        print(f"   • {policy['metric']} {policy['operator']} {policy['threshold']} -> {policy['action']}")

    # Simular métricas y evaluar escalado
    print("
🔄 Simulando evaluación de escalado..."    test_metrics = [
        {"cpu_utilization": 85.5, "memory_utilization": 72.3, "request_count": 850, "error_rate": 2.1},
        {"cpu_utilization": 45.2, "memory_utilization": 58.7, "request_count": 620, "error_rate": 1.8},
        {"cpu_utilization": 92.1, "memory_utilization": 88.4, "request_count": 1250, "error_rate": 3.2},
        {"cpu_utilization": 23.4, "memory_utilization": 34.5, "request_count": 280, "error_rate": 0.5},
        {"cpu_utilization": 78.9, "memory_utilization": 81.2, "request_count": 980, "error_rate": 2.8}
    ]

    for i, metrics in enumerate(test_metrics):
        print(f"\n📊 Evaluación {i+1}: CPU={metrics['cpu_utilization']}%, "
              f"Mem={metrics['memory_utilization']}%, Requests={metrics['request_count']}")

        result = await scaler.evaluate_scaling(metrics)

        print(f"   🎯 Acción recomendada: {result['final_action']}")
        if result['recommended_actions']:
            print(f"   📋 Políticas activadas: {len(result['recommended_actions'])}")

        # Simular tiempo entre evaluaciones
        await asyncio.sleep(1)

    # Mostrar estadísticas finales
    stats = scaler.get_scaling_statistics()
    print("
📈 ESTADÍSTICAS FINALES:"    print(f"   • Políticas configuradas: {stats['total_policies']}")
    print(f"   • Políticas activas: {stats['active_policies']}")
    print(f"   • Triggers totales: {stats['total_triggers']}")
    print(f"   • Historial de métricas: {stats['metrics_history_size']}")

    # Demo de escalado predictivo
    print("
🔮 DEMO - ESCALADO PREDICTIVO"    predictor = PredictiveAutoScaler("predictive_demo")

    # Simular datos históricos
    historical_data = []
    for i in range(24):  # 24 horas
        hour = i % 24
        # Simular patrón: alto uso durante horas de trabajo (9-17)
        base_cpu = 90 if 9 <= hour <= 17 else 30
        cpu = base_cpu + (time.time() % 20)  # Variación

        historical_data.append({
            "timestamp": time.time() - (23 - i) * 3600,  # Últimas 24 horas
            "metrics": {"cpu_utilization": cpu, "memory_utilization": cpu * 0.8}
        })

    predictor.historical_patterns = historical_data

    # Análisis de patrones
    patterns = await predictor.analyze_patterns(historical_data)
    print(f"🔍 Patrones detectados:")
    print(f"   • Hora de peak: {patterns['peak_hour']}:00")
    print(f"   • Hora de bajo uso: {patterns['low_hour']}:00")
    print(f"   • Confianza: {patterns['confidence']:.1f}")

    for rec in patterns['recommendations']:
        print(f"   💡 {rec}")

    # Predicción actual
    current_metrics = {"cpu_utilization": 65.0, "memory_utilization": 52.0}
    prediction = await predictor.predict_and_scale(current_metrics)

    print(f"\n🎯 Predicción actual (hora {prediction['current_hour']}:00):")
    print(f"   • Carga predicha: {prediction['predicted_load']}")
    print(f"   • Acción recomendada: {prediction['recommended_action']}")
    print(f"   • Confianza: {prediction['confidence']:.1f}")

    if prediction['reasoning']:
        for reason in prediction['reasoning']:
            print(f"   📋 {reason}")

    print("
🎉 DEMO COMPLETA EXITOSA!"    print("🌟 Auto-scaling inteligente funcionando correctamente")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(demo_auto_scaling())
'''

    def _get_monitoring_template(self) -> str:
        """Template para sistema de monitoreo"""
        return '''#!/usr/bin/env python3
"""
Sistema de Monitoreo - AEGIS Cloud Operations
Monitoreo completo con dashboards, alertas y análisis
"""

import asyncio
import json
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Alert:
    """Alerta del sistema de monitoreo"""
    alert_id: str
    name: str
    severity: str
    message: str
    metric: str
    value: float
    threshold: float
    condition: str
    target: str
    timestamp: float
    status: str = "active"

@dataclass
class Dashboard:
    """Dashboard de métricas"""
    dashboard_id: str
    name: str
    description: str
    charts: List[Dict[str, Any]]
    metrics: List[str]
    refresh_interval: int
    created_at: float

class AEGISMonitoring:
    """Sistema completo de monitoreo para AEGIS"""

    def __init__(self):
        self.alerts: List[Alert] = []
        self.dashboards: Dict[str, Dashboard] = {}
        self.metrics_buffer: List[Dict[str, Any]] = []
        self.alert_rules: List[Dict[str, Any]] = []

    async def setup_monitoring(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Configurar sistema de monitoreo completo"""

        logger.info("📊 Configurando sistema de monitoreo AEGIS...")

        # Crear dashboards
        dashboards_created = await self._create_dashboards(config.get("dashboards", []))

        # Configurar alertas
        alerts_configured = await self._configure_alerts(config.get("alerts", []))

        # Configurar métricas
        metrics_setup = await self._setup_metrics_collection(config.get("metrics", {}))

        return {
            "dashboards_created": dashboards_created,
            "alerts_configured": alerts_configured,
            "metrics_setup": metrics_setup,
            "status": "active"
        }

    async def _create_dashboards(self, dashboard_configs: List[Dict[str, Any]]) -> int:
        """Crear dashboards configurados"""

        default_dashboards = [
            {
                "name": "System Overview",
                "description": "Vista general del sistema AEGIS",
                "charts": [
                    {"type": "line", "metrics": ["cpu_utilization", "memory_utilization"], "title": "System Resources"},
                    {"type": "area", "metrics": ["network_in", "network_out"], "title": "Network Traffic"},
                    {"type": "bar", "metrics": ["request_count", "error_count"], "title": "Request Statistics"},
                    {"type": "gauge", "metrics": ["health_score"], "title": "System Health"}
                ],
                "metrics": ["cpu_utilization", "memory_utilization", "network_in", "network_out",
                           "request_count", "error_count", "health_score"],
                "refresh_interval": 30
            },
            {
                "name": "Cloud Performance",
                "description": "Rendimiento de despliegues cloud",
                "charts": [
                    {"type": "line", "metrics": ["cloud_cpu", "cloud_memory"], "title": "Cloud Resources"},
                    {"type": "heatmap", "metrics": ["instance_utilization"], "title": "Instance Utilization"},
                    {"type": "bar", "metrics": ["scaling_events"], "title": "Auto-scaling Events"}
                ],
                "metrics": ["cloud_cpu", "cloud_memory", "instance_utilization", "scaling_events"],
                "refresh_interval": 60
            },
            {
                "name": "Edge Computing",
                "description": "Monitoreo de dispositivos edge",
                "charts": [
                    {"type": "scatter", "metrics": ["device_locations"], "title": "Device Locations"},
                    {"type": "line", "metrics": ["edge_inference_time"], "title": "Inference Performance"},
                    {"type": "bar", "metrics": ["federated_rounds"], "title": "Federated Learning"}
                ],
                "metrics": ["device_locations", "edge_inference_time", "federated_rounds"],
                "refresh_interval": 120
            }
        ]

        # Combinar con configuraciones personalizadas
        all_dashboards = default_dashboards + dashboard_configs

        created_count = 0
        for dash_config in all_dashboards:
            dashboard_id = f"dash_{int(time.time())}_{created_count}"

            dashboard = Dashboard(
                dashboard_id=dashboard_id,
                name=dash_config["name"],
                description=dash_config["description"],
                charts=dash_config["charts"],
                metrics=dash_config["metrics"],
                refresh_interval=dash_config["refresh_interval"],
                created_at=time.time()
            )

            self.dashboards[dashboard_id] = dashboard
            created_count += 1

            logger.info(f"✅ Dashboard creado: {dashboard.name}")

        return created_count

    async def _configure_alerts(self, alert_configs: List[Dict[str, Any]]) -> int:
        """Configurar reglas de alertas"""

        default_alerts = [
            {
                "name": "High CPU Usage",
                "metric": "cpu_utilization",
                "condition": ">",
                "threshold": 80.0,
                "severity": "warning",
                "description": "CPU utilization above 80%",
                "channels": ["email", "slack"],
                "cooldown": 300
            },
            {
                "name": "High Memory Usage",
                "metric": "memory_utilization",
                "condition": ">",
                "threshold": 85.0,
                "severity": "warning",
                "description": "Memory utilization above 85%",
                "channels": ["email"],
                "cooldown": 600
            },
            {
                "name": "Service Down",
                "metric": "health_status",
                "condition": "==",
                "threshold": 0,
                "severity": "critical",
                "description": "Service is down",
                "channels": ["email", "sms", "pagerduty"],
                "cooldown": 60
            },
            {
                "name": "High Error Rate",
                "metric": "error_rate",
                "condition": ">",
                "threshold": 5.0,
                "severity": "error",
                "description": "Error rate above 5%",
                "channels": ["email", "slack"],
                "cooldown": 180
            }
        ]

        # Combinar con alertas personalizadas
        all_alerts = default_alerts + alert_configs

        for alert_config in all_alerts:
            rule = {
                "rule_id": f"rule_{int(time.time())}_{len(self.alert_rules)}",
                "name": alert_config["name"],
                "metric": alert_config["metric"],
                "condition": alert_config["condition"],
                "threshold": alert_config["threshold"],
                "severity": alert_config["severity"],
                "description": alert_config["description"],
                "channels": alert_config["channels"],
                "cooldown": alert_config["cooldown"],
                "last_triggered": 0,
                "enabled": True
            }

            self.alert_rules.append(rule)
            logger.info(f"✅ Regla de alerta configurada: {rule['name']}")

        return len(all_alerts)

    async def _setup_metrics_collection(self, metrics_config: Dict[str, Any]) -> Dict[str, Any]:
        """Configurar colección de métricas"""

        collection_config = {
            "enabled_metrics": metrics_config.get("enabled", [
                "cpu_utilization", "memory_utilization", "network_in", "network_out",
                "request_count", "error_count", "response_time", "health_status"
            ]),
            "collection_interval": metrics_config.get("interval", 30),
            "retention_days": metrics_config.get("retention", 30),
            "storage_backend": metrics_config.get("storage", "timescaledb"),
            "aggregation": metrics_config.get("aggregation", ["1m", "5m", "1h", "1d"])
        }

        logger.info(f"✅ Colección de métricas configurada: {len(collection_config['enabled_metrics'])} métricas")

        return collection_config

    async def collect_metrics(self, source: str, metrics: Dict[str, Any]) -> None:
        """Recopilar métricas de una fuente"""

        metric_entry = {
            "timestamp": time.time(),
            "source": source,
            "metrics": metrics
        }

        self.metrics_buffer.append(metric_entry)

        # Procesar alertas
        await self._process_alerts(metrics, source)

        # Mantener buffer limitado
        if len(self.metrics_buffer) > 1000:
            # En producción, persistir en base de datos
            self.metrics_buffer = self.metrics_buffer[-500:]

    async def _process_alerts(self, metrics: Dict[str, Any], source: str) -> None:
        """Procesar reglas de alertas"""

        current_time = time.time()

        for rule in self.alert_rules:
            if not rule["enabled"]:
                continue

            # Verificar cooldown
            if current_time - rule["last_triggered"] < rule["cooldown"]:
                continue

            metric_value = metrics.get(rule["metric"])
            if metric_value is None:
                continue

            # Evaluar condición
            condition_met = self._evaluate_condition(
                metric_value, rule["condition"], rule["threshold"]
            )

            if condition_met:
                # Crear alerta
                alert = Alert(
                    alert_id=f"alert_{int(current_time)}_{len(self.alerts)}",
                    name=rule["name"],
                    severity=rule["severity"],
                    message=f"{rule['description']} - Value: {metric_value}, Threshold: {rule['threshold']}",
                    metric=rule["metric"],
                    value=metric_value,
                    threshold=rule["threshold"],
                    condition=rule["condition"],
                    target=source,
                    timestamp=current_time
                )

                self.alerts.append(alert)
                rule["last_triggered"] = current_time

                # Notificar (en producción, enviar a canales configurados)
                await self._send_alert_notification(alert, rule["channels"])

                logger.warning(f"🚨 ALERTA: {alert.name} - {alert.message}")

    def _evaluate_condition(self, value: float, condition: str, threshold: float) -> bool:
        """Evaluar condición de alerta"""
        if condition == ">":
            return value > threshold
        elif condition == "<":
            return value < threshold
        elif condition == ">=":
            return value >= threshold
        elif condition == "<=":
            return value <= threshold
        elif condition == "==":
            return value == threshold
        elif condition == "!=":
            return value != threshold
        return False

    async def _send_alert_notification(self, alert: Alert, channels: List[str]) -> None:
        """Enviar notificación de alerta"""
        # En producción, implementar envío real a Slack, email, etc.
        logger.info(f"📤 Enviando alerta a canales: {', '.join(channels)}")

    def get_dashboard_data(self, dashboard_id: str) -> Optional[Dict[str, Any]]:
        """Obtener datos de un dashboard"""

        if dashboard_id not in self.dashboards:
            return None

        dashboard = self.dashboards[dashboard_id]

        # Simular datos de métricas recientes
        recent_metrics = self.metrics_buffer[-50:] if self.metrics_buffer else []

        return {
            "dashboard_id": dashboard.dashboard_id,
            "name": dashboard.name,
            "description": dashboard.description,
            "charts": dashboard.charts,
            "metrics": dashboard.metrics,
            "refresh_interval": dashboard.refresh_interval,
            "data": recent_metrics
        }

    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Obtener alertas activas"""

        return [
            {
                "alert_id": alert.alert_id,
                "name": alert.name,
                "severity": alert.severity,
                "message": alert.message,
                "metric": alert.metric,
                "value": alert.value,
                "threshold": alert.threshold,
                "target": alert.target,
                "timestamp": alert.timestamp,
                "status": alert.status
            }
            for alert in self.alerts[-50:]  # Últimas 50 alertas
            if alert.status == "active"
        ]

    def get_system_health_score(self) -> Dict[str, Any]:
        """Calcular score de salud del sistema"""

        if not self.metrics_buffer:
            return {"score": 50, "status": "unknown", "issues": ["No metrics available"]}

        # Analizar métricas recientes
        recent_metrics = self.metrics_buffer[-10:]  # Últimas 10 entradas

        scores = []
        issues = []

        for entry in recent_metrics:
            metrics = entry["metrics"]

            # CPU score
            cpu = metrics.get("cpu_utilization", 50)
            if cpu > 90:
                scores.append(20)
                issues.append("High CPU usage")
            elif cpu > 70:
                scores.append(60)
                issues.append("Elevated CPU usage")
            else:
                scores.append(100)

            # Memory score
            memory = metrics.get("memory_utilization", 50)
            if memory > 95:
                scores.append(10)
                issues.append("Critical memory usage")
            elif memory > 80:
                scores.append(50)
                issues.append("High memory usage")
            else:
                scores.append(100)

            # Error rate score
            error_rate = metrics.get("error_rate", 0)
            if error_rate > 10:
                scores.append(20)
                issues.append("High error rate")
            elif error_rate > 5:
                scores.append(60)
                issues.append("Elevated error rate")
            else:
                scores.append(100)

        # Calcular score promedio
        avg_score = sum(scores) / len(scores) if scores else 50

        # Determinar estado
        if avg_score >= 90:
            status = "excellent"
        elif avg_score >= 75:
            status = "good"
        elif avg_score >= 50:
            status = "warning"
        else:
            status = "critical"

        return {
            "score": round(avg_score, 1),
            "status": status,
            "issues": list(set(issues)),  # Remover duplicados
            "metrics_analyzed": len(recent_metrics),
            "timestamp": time.time()
        }

async def demo_monitoring():
    """Demostración del sistema de monitoreo"""

    print("📊 DEMO - AEGIS Monitoring System")
    print("=" * 50)

    monitoring = AEGISMonitoring()

    # Configurar monitoreo
    config = {
        "dashboards": [
            {
                "name": "Custom ML Metrics",
                "description": "Métricas específicas de modelos ML",
                "charts": [
                    {"type": "line", "metrics": ["model_accuracy", "inference_time"], "title": "Model Performance"}
                ],
                "metrics": ["model_accuracy", "inference_time", "training_loss"],
                "refresh_interval": 60
            }
        ],
        "alerts": [
            {
                "name": "Model Accuracy Drop",
                "metric": "model_accuracy",
                "condition": "<",
                "threshold": 85.0,
                "severity": "error",
                "description": "Model accuracy dropped below 85%",
                "channels": ["email", "slack"],
                "cooldown": 3600
            }
        ],
        "metrics": {
            "enabled": ["cpu_utilization", "memory_utilization", "model_accuracy", "inference_time"],
            "interval": 30,
            "retention": 7
        }
    }

    # Configurar sistema
    setup_result = await monitoring.setup_monitoring(config)
    print(f"✅ Sistema de monitoreo configurado: {setup_result['status']}")

    # Simular recopilación de métricas
    print("
📈 Simulando recopilación de métricas..."    sample_metrics = [
        {"cpu_utilization": 65.5, "memory_utilization": 72.3, "error_rate": 1.2, "model_accuracy": 92.1, "inference_time": 45.2},
        {"cpu_utilization": 78.2, "memory_utilization": 85.1, "error_rate": 0.8, "model_accuracy": 91.8, "inference_time": 47.1},
        {"cpu_utilization": 89.5, "memory_utilization": 91.2, "error_rate": 2.1, "model_accuracy": 88.5, "inference_time": 52.3},
        {"cpu_utilization": 45.2, "memory_utilization": 58.7, "error_rate": 0.5, "model_accuracy": 93.2, "inference_time": 43.8},
        {"cpu_utilization": 92.1, "memory_utilization": 94.5, "error_rate": 4.2, "model_accuracy": 85.1, "inference_time": 58.9}
    ]

    for i, metrics in enumerate(sample_metrics):
        await monitoring.collect_metrics(f"server_{i+1}", metrics)
        await asyncio.sleep(0.5)

        # Mostrar score de salud
        health = monitoring.get_system_health_score()
        print(f"💓 Health Score {i+1}: {health['score']}/100 ({health['status']})")

        if health['issues']:
            print(f"   ⚠️ Issues: {', '.join(health['issues'])}")

    # Mostrar alertas generadas
    active_alerts = monitoring.get_active_alerts()
    print("
🚨 ALERTAS GENERADAS:"    if active_alerts:
        for alert in active_alerts:
            print(f"   • {alert['name']} ({alert['severity']}): {alert['message']}")
    else:
        print("   ✅ No hay alertas activas")

    # Mostrar datos de dashboard
    if monitoring.dashboards:
        dashboard_id = list(monitoring.dashboards.keys())[0]
        dashboard_data = monitoring.get_dashboard_data(dashboard_id)

        if dashboard_data:
            print("
📊 DATOS DE DASHBOARD:"            print(f"   📈 Dashboard: {dashboard_data['name']}")
            print(f"   📋 Métricas: {', '.join(dashboard_data['metrics'])}")
            print(f"   🔄 Intervalo: {dashboard_data['refresh_interval']}s")
            print(f"   📊 Puntos de datos: {len(dashboard_data['data'])}")

    # Análisis final
    final_health = monitoring.get_system_health_score()
    print("
🏆 ANÁLISIS FINAL:"    print(f"   📊 Score promedio: {final_health['score']}/100")
    print(f"   🎯 Estado general: {final_health['status'].upper()}")
    print(f"   📈 Alertas totales: {len(monitoring.alerts)}")
    print(f"   📋 Dashboards: {len(monitoring.dashboards)}")
    print(f"   📊 Métricas recolectadas: {len(monitoring.metrics_buffer)}")

    if final_health['issues']:
        print("
💡 RECOMENDACIONES:"        for issue in final_health['issues']:
            print(f"   • Revisar: {issue}")

    print("
🎉 DEMO COMPLETA EXITOSA!"    print("🌟 Sistema de monitoreo completamente operativo")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(demo_monitoring())
'''

    def _get_hybrid_main_template(self) -> str:
        """Template principal para sistema híbrido"""
        return '''#!/usr/bin/env python3
"""
AEGIS Hybrid System - Main Application
Sistema completo que integra cloud, edge y ML
"""

import asyncio
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import logging

# Importar componentes del sistema
from aegis_sdk import AEGIS
from ml_framework_integration import MLFrameworkManager
from federated_learning import FederatedLearningCoordinator
from multi_cloud_orchestration import MultiCloudOrchestrator
from edge_computing import EdgeComputingSystem

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AEGISHybridSystem:
    """Sistema híbrido completo AEGIS"""

    def __init__(self):
        self.app = FastAPI(title="AEGIS Hybrid System", version="3.3.0")

        # Configurar CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # Inicializar componentes
        self.aegis_sdk = AEGIS()
        self.ml_manager = MLFrameworkManager()
        self.federated_coordinator = FederatedLearningCoordinator(self.ml_manager)
        self.cloud_orchestrator = MultiCloudOrchestrator()
        self.edge_system = EdgeComputingSystem()

        # Configurar rutas
        self._setup_routes()

        logger.info("🎯 Sistema híbrido AEGIS inicializado")

    def _setup_routes(self):
        """Configurar rutas de la API"""

        @self.app.get("/")
        async def root():
            """Endpoint raíz"""
            return {
                "message": "AEGIS Hybrid System API",
                "version": "3.3.0",
                "status": "operational",
                "components": ["ml", "federated", "cloud", "edge"]
            }

        @self.app.get("/health")
        async def health_check():
            """Verificación de salud del sistema"""
            try:
                # Verificar todos los componentes
                health_status = {
                    "ml_framework": True,
                    "federated_learning": True,
                    "cloud_orchestration": True,
                    "edge_system": True,
                    "overall": True
                }

                return {
                    "status": "healthy" if all(health_status.values()) else "degraded",
                    "components": health_status,
                    "timestamp": asyncio.get_event_loop().time()
                }

            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/ml/models")
        async def register_model(model_data: dict):
            """Registrar un modelo de ML"""
            try:
                result = await self.aegis_sdk.client.register_model(
                    model_path=model_data.get("path", ""),
                    framework=model_data.get("framework", "tensorflow"),
                    model_type=model_data.get("type", "classification"),
                    metadata=model_data.get("metadata", {})
                )

                if result.success:
                    return {"status": "success", "model_id": result.data["model_id"]}
                else:
                    raise HTTPException(status_code=400, detail=result.error)

            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/ml/predict/{model_id}")
        async def predict(model_id: str, input_data: dict):
            """Realizar predicción"""
            try:
                result = await self.aegis_sdk.client.predict(model_id, input_data["data"])

                if result.success:
                    return {"prediction": result.data["prediction"]}
                else:
                    raise HTTPException(status_code=400, detail=result.error)

            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/federated/training")
        async def start_federated_training(training_config: dict):
            """Iniciar entrenamiento federado"""
            try:
                result = await self.aegis_sdk.client.start_federated_training(
                    training_config["model_id"],
                    training_config["participants"]
                )

                if result.success:
                    return {
                        "training_id": result.data["training_id"],
                        "status": "started",
                        "participants": result.data["participants"]
                    }
                else:
                    raise HTTPException(status_code=400, detail=result.error)

            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/cloud/deploy")
        async def deploy_to_cloud(deployment_config: dict):
            """Desplegar en cloud"""
            try:
                result = await self.aegis_sdk.client.create_cloud_deployment(
                    name=deployment_config["name"],
                    provider=deployment_config["provider"],
                    region=deployment_config["region"],
                    instance_config=deployment_config["instances"]
                )

                if result.success:
                    return {
                        "deployment_id": result.data["deployment_id"],
                        "provider": result.data["provider"],
                        "instances": result.data["instances"]
                    }
                else:
                    raise HTTPException(status_code=400, detail=result.error)

            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/edge/devices")
        async def register_edge_device(device_config: dict):
            """Registrar dispositivo edge"""
            try:
                device_id = await self.edge_system.register_edge_device(device_config)

                if device_id:
                    return {
                        "device_id": device_id,
                        "status": "registered",
                        "capabilities": device_config.get("capabilities", [])
                    }
                else:
                    raise HTTPException(status_code=400, detail="Error registering device")

            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/edge/deploy/{model_id}")
        async def deploy_to_edge(model_id: str, deployment_config: dict):
            """Desplegar modelo en dispositivos edge"""
            try:
                device_ids = deployment_config["device_ids"]
                result = await self.aegis_sdk.client.deploy_to_edge(
                    model_id, device_ids,
                    deployment_config.get("optimization", "quantization")
                )

                if result.success:
                    return {
                        "model_id": model_id,
                        "deployments": result.data["deployments"],
                        "device_count": result.data["device_count"]
                    }
                else:
                    raise HTTPException(status_code=400, detail=result.error)

            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/metrics")
        async def get_metrics():
            """Obtener métricas del sistema"""
            try:
                # Obtener métricas de cloud
                cloud_metrics = await self.aegis.client.get_cloud_metrics()

                # Obtener estado del sistema edge
                edge_status = self.edge_system.get_system_status()

                return {
                    "cloud": cloud_metrics.data if cloud_metrics.success else {},
                    "edge": edge_status,
                    "timestamp": asyncio.get_event_loop().time()
                }

            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/dashboard")
        async def get_dashboard_data():
            """Obtener datos para dashboard"""
            try:
                # Compilar datos de todos los componentes
                dashboard_data = {
                    "system_health": await self._get_system_health(),
                    "ml_models": len(self.ml_manager.models),
                    "federated_rounds": len(self.federated_coordinator.completed_rounds),
                    "cloud_deployments": len(self.cloud_orchestrator.deployments),
                    "edge_devices": self.edge_system.get_system_status()["total_devices"],
                    "active_services": {
                        "ml": True,
                        "federated": True,
                        "cloud": True,
                        "edge": True
                    }
                }

                return dashboard_data

            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

    async def _get_system_health(self) -> dict:
        """Obtener estado de salud general del sistema"""
        try:
            health_result = await self.aegis_sdk.client.health_check()

            if health_result.success:
                return health_result.data
            else:
                return {"status": "error", "message": health_result.error}

        except Exception as e:
            return {"status": "error", "message": str(e)}

    async def start_services(self):
        """Iniciar servicios del sistema híbrido"""
        logger.info("🚀 Iniciando servicios del sistema híbrido...")

        # Aquí se iniciarían servicios adicionales como:
        # - Monitor de recursos
        # - Recolector de métricas
        # - Gestor de sesiones federadas
        # - Balanceador de carga

        logger.info("✅ Servicios del sistema híbrido iniciados")

    def run(self, host: str = "0.0.0.0", port: int = 8000):
        """Ejecutar el servidor"""
        logger.info(f"🌐 Iniciando servidor en {host}:{port}")

        uvicorn.run(
            self.app,
            host=host,
            port=port,
            log_level="info"
        )

# Instancia global del sistema
hybrid_system = AEGISHybridSystem()

if __name__ == "__main__":
    # Ejecutar servidor
    hybrid_system.run()
'''

    def _get_hybrid_orchestrator_template(self) -> str:
        """Template para orquestador híbrido"""
        return '''#!/usr/bin/env python3
"""
Orquestador Híbrido - AEGIS Hybrid System
Orquestador inteligente que coordina recursos entre cloud y edge
"""

import asyncio
import time
from typing import Dict, List, Any, Optional
import logging

from aegis_sdk import AEGIS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HybridOrchestrator:
    """Orquestador que coordina recursos híbridos"""

    def __init__(self):
        self.aegis = AEGIS()
        self.workload_policies: Dict[str, Dict[str, Any]] = {}
        self.resource_allocation: Dict[str, Any] = {}
        self.performance_targets: Dict[str, float] = {}

    async def initialize(self):
        """Inicializar orquestador"""
        logger.info("🎼 Inicializando orquestador híbrido")

        # Políticas de workload por defecto
        self.workload_policies = {
            "inference": {
                "preferred_target": "edge",
                "fallback_targets": ["cloud"],
                "performance_target": 50,  # ms
                "cost_priority": "low"
            },
            "training": {
                "preferred_target": "cloud",
                "fallback_targets": ["edge"],
                "performance_target": 3600,  # 1 hora
                "cost_priority": "medium"
            },
            "federated_learning": {
                "preferred_target": "distributed",
                "fallback_targets": ["cloud"],
                "performance_target": 1800,  # 30 min por ronda
                "cost_priority": "high"
            }
        }

        # Targets de performance
        self.performance_targets = {
            "inference_latency": 100,  # ms
            "training_throughput": 1000,  # samples/second
            "federated_round_time": 600,  # seconds
            "cost_efficiency": 0.8  # 80% de eficiencia objetivo
        }

        logger.info("✅ Orquestador híbrido inicializado")

    async def orchestrate_workload(self, workload_type: str,
                                 workload_config: Dict[str, Any]) -> Dict[str, Any]:
        """Orquestrar un workload específico"""

        logger.info(f"🎯 Orquestando workload: {workload_type}")

        if workload_type not in self.workload_policies:
            raise ValueError(f"Tipo de workload no soportado: {workload_type}")

        policy = self.workload_policies[workload_type]

        # Evaluar objetivos y restricciones
        targets = await self._evaluate_targets(workload_config, policy)

        # Seleccionar mejor target
        selected_target = await self._select_optimal_target(targets, policy)

        # Ejecutar workload
        execution_result = await self._execute_on_target(
            workload_type, workload_config, selected_target
        )

        # Monitorear y optimizar
        await self._monitor_and_optimize(execution_result)

        return {
            "workload_type": workload_type,
            "selected_target": selected_target,
            "execution_result": execution_result,
            "timestamp": time.time()
        }

    async def _evaluate_targets(self, workload_config: Dict[str, Any],
                              policy: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
        """Evaluar targets disponibles"""

        targets = {
            "cloud": [],
            "edge": [],
            "distributed": []
        }

        # Evaluar capacidad de cloud
        cloud_capacity = await self._assess_cloud_capacity(workload_config)
        if cloud_capacity["available"]:
            targets["cloud"].append({
                "provider": cloud_capacity["recommended_provider"],
                "capacity_score": cloud_capacity["capacity_score"],
                "cost_estimate": cloud_capacity["cost_estimate"],
                "performance_estimate": cloud_capacity["performance_estimate"]
            })

        # Evaluar capacidad de edge
        edge_capacity = await self._assess_edge_capacity(workload_config)
        if edge_capacity["available"]:
            targets["edge"].extend(edge_capacity["devices"])

        # Evaluar capacidad distribuida (federated)
        distributed_capacity = await self._assess_distributed_capacity(workload_config)
        if distributed_capacity["available"]:
            targets["distributed"].append({
                "participants": distributed_capacity["participant_count"],
                "coordination_overhead": distributed_capacity["coordination_overhead"],
                "privacy_score": distributed_capacity["privacy_score"]
            })

        return targets

    async def _assess_cloud_capacity(self, workload_config: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluar capacidad de cloud"""

        # Obtener métricas de cloud
        metrics_result = await self.aegis.client.get_cloud_metrics()

        if not metrics_result.success:
            return {"available": False, "reason": "No cloud metrics available"}

        # Evaluar proveedores
        best_provider = None
        best_score = 0
        best_cost = float('inf')

        for provider, provider_metrics in metrics_result.data.items():
            # Calcular score de capacidad
            cpu_available = 100 - provider_metrics.get("avg_cpu_utilization", 0)
            memory_available = 100 - provider_metrics.get("avg_memory_utilization", 0)
            capacity_score = (cpu_available + memory_available) / 2

            # Estimar costo
            instance_count = provider_metrics.get("running_instances", 1)
            hourly_rate = provider_metrics.get("total_cost", 1.0) / max(instance_count, 1)
            cost_estimate = hourly_rate * workload_config.get("estimated_duration_hours", 1)

            # Seleccionar mejor opción
            if capacity_score > best_score or (capacity_score == best_score and cost_estimate < best_cost):
                best_provider = provider
                best_score = capacity_score
                best_cost = cost_estimate

        return {
            "available": best_provider is not None,
            "recommended_provider": best_provider,
            "capacity_score": best_score,
            "cost_estimate": best_cost,
            "performance_estimate": best_score * 10  # Score simplificado
        }

    async def _assess_edge_capacity(self, workload_config: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluar capacidad de edge"""

        # Simular evaluación de dispositivos edge
        # En producción, consultar el sistema edge

        available_devices = [
            {
                "device_id": "edge_001",
                "device_type": "raspberry_pi",
                "capabilities": ["inference_only"],
                "performance_score": 7.5,
                "battery_level": 85,
                "network_quality": 0.9
            },
            {
                "device_id": "edge_002",
                "device_type": "jetson_nano",
                "capabilities": ["inference_only", "federated_client"],
                "performance_score": 9.2,
                "battery_level": 92,
                "network_quality": 0.8
            }
        ]

        # Filtrar por requisitos del workload
        suitable_devices = []
        for device in available_devices:
            if workload_config.get("requires_federated", False):
                if "federated_client" not in device["capabilities"]:
                    continue

            if device["battery_level"] < workload_config.get("min_battery", 20):
                continue

            if device["network_quality"] < workload_config.get("min_network_quality", 0.5):
                continue

            suitable_devices.append(device)

        return {
            "available": len(suitable_devices) > 0,
            "devices": suitable_devices,
            "total_capacity": sum(d["performance_score"] for d in suitable_devices)
        }

    async def _assess_distributed_capacity(self, workload_config: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluar capacidad distribuida"""

        # Simular evaluación de capacidad federada
        # En producción, consultar coordinador federado

        federated_status = {
            "participant_count": 5,
            "avg_participant_performance": 8.0,
            "coordination_overhead": 0.15,  # 15% overhead
            "privacy_score": 9.5,  # Alta privacidad
            "network_stability": 0.85
        }

        # Evaluar viabilidad
        min_participants = workload_config.get("min_participants", 3)
        max_coordination_overhead = workload_config.get("max_coordination_overhead", 0.3)

        viable = (
            federated_status["participant_count"] >= min_participants and
            federated_status["coordination_overhead"] <= max_coordination_overhead
        )

        return {
            "available": viable,
            "participant_count": federated_status["participant_count"],
            "coordination_overhead": federated_status["coordination_overhead"],
            "privacy_score": federated_status["privacy_score"],
            "estimated_performance": federated_status["avg_participant_performance"] * 0.8  # Con overhead
        }

    async def _select_optimal_target(self, targets: Dict[str, List[Dict[str, Any]]],
                                   policy: Dict[str, Any]) -> Dict[str, Any]:
        """Seleccionar el target óptimo"""

        preferred_target = policy["preferred_target"]
        fallback_targets = policy["fallback_targets"]

        # Intentar target preferido
        if preferred_target in targets and targets[preferred_target]:
            return {
                "type": preferred_target,
                "options": targets[preferred_target][0],  # Seleccionar mejor opción
                "reason": f"Target preferido disponible: {preferred_target}"
            }

        # Intentar fallbacks
        for fallback in fallback_targets:
            if fallback in targets and targets[fallback]:
                return {
                    "type": fallback,
                    "options": targets[fallback][0],
                    "reason": f"Fallback seleccionado: {fallback}"
                }

        # Si no hay opciones, usar cloud como último recurso
        if targets.get("cloud"):
            return {
                "type": "cloud",
                "options": targets["cloud"][0],
                "reason": "Último recurso: cloud"
            }

        raise ValueError("No hay targets disponibles para el workload")

    async def _execute_on_target(self, workload_type: str, workload_config: Dict[str, Any],
                               target: Dict[str, Any]) -> Dict[str, Any]:
        """Ejecutar workload en el target seleccionado"""

        target_type = target["type"]
        target_options = target["options"]

        logger.info(f"🚀 Ejecutando {workload_type} en {target_type}")

        if target_type == "cloud":
            return await self._execute_cloud_workload(workload_type, workload_config, target_options)
        elif target_type == "edge":
            return await self._execute_edge_workload(workload_type, workload_config, target_options)
        elif target_type == "distributed":
            return await self._execute_distributed_workload(workload_type, workload_config, target_options)
        else:
            raise ValueError(f"Tipo de target no soportado: {target_type}")

    async def _execute_cloud_workload(self, workload_type: str, workload_config: Dict[str, Any],
                                    target_options: Dict[str, Any]) -> Dict[str, Any]:
        """Ejecutar workload en cloud"""

        # Crear despliegue cloud
        deployment_result = await self.aegis.client.create_cloud_deployment(
            name=f"{workload_type}_deployment",
            provider=target_options["provider"],
            region="us-east-1",  # Default
            instance_config={
                "instance_type": "t3.medium",
                "count": 2,
                "auto_scaling": True
            }
        )

        if not deployment_result.success:
            raise RuntimeError(f"Error creando despliegue cloud: {deployment_result.error}")

        return {
            "target": "cloud",
            "deployment_id": deployment_result.data["deployment_id"],
            "provider": target_options["provider"],
            "estimated_cost": target_options.get("cost_estimate", 0),
            "status": "executing"
        }

    async def _execute_edge_workload(self, workload_type: str, workload_config: Dict[str, Any],
                                   target_options: Dict[str, Any]) -> Dict[str, Any]:
        """Ejecutar workload en edge"""

        # Simular despliegue en dispositivo edge
        # En producción, usar el sistema edge real

        return {
            "target": "edge",
            "device_id": target_options["device_id"],
            "device_type": target_options["device_type"],
            "performance_score": target_options["performance_score"],
            "status": "executing"
        }

    async def _execute_distributed_workload(self, workload_type: str, workload_config: Dict[str, Any],
                                          target_options: Dict[str, Any]) -> Dict[str, Any]:
        """Ejecutar workload distribuido"""

        # Iniciar entrenamiento federado
        federated_result = await self.aegis.client.start_federated_training(
            workload_config.get("model_id", "default_model"),
            [f"participant_{i}" for i in range(target_options["participants"])]
        )

        if not federated_result.success:
            raise RuntimeError(f"Error iniciando federated learning: {federated_result.error}")

        return {
            "target": "distributed",
            "training_id": federated_result.data["training_id"],
            "participants": target_options["participants"],
            "coordination_overhead": target_options["coordination_overhead"],
            "status": "executing"
        }

    async def _monitor_and_optimize(self, execution_result: Dict[str, Any]) -> None:
        """Monitorear y optimizar ejecución"""

        # Registrar resultado para aprendizaje futuro
        self.resource_allocation[execution_result.get("deployment_id", "unknown")] = {
            "result": execution_result,
            "timestamp": time.time(),
            "performance_metrics": {}  # En producción, recopilar métricas reales
        }

        logger.info(f"📊 Resultado registrado para optimización futura")

    async def get_orchestration_status(self) -> Dict[str, Any]:
        """Obtener estado del orquestador"""

        return {
            "workload_policies": len(self.workload_policies),
            "active_allocations": len(self.resource_allocation),
            "performance_targets": self.performance_targets,
            "supported_workloads": list(self.workload_policies.keys())
        }

async def demo_hybrid_orchestrator():
    """Demostración del orquestador híbrido"""

    print("🎼 DEMO - AEGIS Hybrid Orchestrator")
    print("=" * 50)

    orchestrator = HybridOrchestrator()
    await orchestrator.initialize()

    # Definir workloads de ejemplo
    workloads = [
        {
            "type": "inference",
            "config": {
                "model_id": "resnet_classifier",
                "estimated_duration_hours": 0.1,
                "latency_requirement": 50  # ms
            }
        },
        {
            "type": "training",
            "config": {
                "model_id": "bert_model",
                "dataset_size": 10000,
                "estimated_duration_hours": 2,
                "accuracy_target": 0.95
            }
        },
        {
            "type": "federated_learning",
            "config": {
                "model_id": "federated_model",
                "min_participants": 3,
                "estimated_duration_hours": 1,
                "privacy_requirement": "high"
            }
        }
    ]

    for workload in workloads:
        print(f"\n🎯 Orquestando workload: {workload['type']}")

        try:
            result = await orchestrator.orchestrate_workload(
                workload["type"], workload["config"]
            )

            print(f"✅ Workload orquestado exitosamente")
            print(f"   🎯 Target seleccionado: {result['selected_target']['type']}")
            print(f"   📋 Razón: {result['selected_target']['reason']}")
            print(f"   🚀 Estado: {result['execution_result']['status']}")

        except Exception as e:
            print(f"❌ Error orquestando workload: {e}")

    # Mostrar estado final
    status = await orchestrator.get_orchestration_status()
    print("
📊 ESTADO FINAL DEL ORQUESTADOR:"    print(f"   📋 Políticas de workload: {status['workload_policies']}")
    print(f"   🎯 Workloads soportados: {', '.join(status['supported_workloads'])}")
    print(f"   📈 Allocaciones activas: {status['active_allocations']}")

    print("
🎉 DEMO COMPLETA EXITOSA!"    print("🌟 Orquestador híbrido completamente operativo")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(demo_hybrid_orchestrator())
'''

    def _get_hybrid_monitoring_template(self) -> str:
        """Template para monitoreo del sistema híbrido"""
        return '''#!/usr/bin/env python3
"""
Monitoreo Híbrido - AEGIS Hybrid System
Sistema de monitoreo unificado para cloud, edge y ML
"""

import asyncio
import time
import json
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime, timedelta

from aegis_sdk import AEGIS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HybridMonitoringSystem:
    """Sistema de monitoreo para infraestructura híbrida"""

    def __init__(self):
        self.aegis = AEGIS()
        self.metrics_history: Dict[str, List[Dict[str, Any]]] = {}
        self.alerts: List[Dict[str, Any]] = []
        self.anomaly_detectors: Dict[str, Any] = {}

    async def initialize_monitoring(self):
        """Inicializar sistema de monitoreo"""

        logger.info("📊 Inicializando monitoreo híbrido")

        # Inicializar históricos de métricas
        self.metrics_history = {
            "cloud": [],
            "edge": [],
            "ml": [],
            "federated": []
        }

        # Configurar detectores de anomalías
        self.anomaly_detectors = {
            "cpu_spike": {"threshold": 85, "window": 300},  # 5 minutos
            "memory_leak": {"threshold": 90, "trend_window": 3600},  # 1 hora
            "latency_spike": {"threshold": 200, "baseline": 50},  # ms
            "error_rate_spike": {"threshold": 5, "baseline": 1}  # %
        }

        logger.info("✅ Sistema de monitoreo híbrido inicializado")

    async def collect_hybrid_metrics(self) -> Dict[str, Any]:
        """Recopilar métricas de todos los componentes"""

        metrics = {
            "timestamp": time.time(),
            "cloud": {},
            "edge": {},
            "ml": {},
            "federated": {},
            "system": {}
        }

        try:
            # Métricas de cloud
            cloud_result = await self.aegis.client.get_cloud_metrics()
            if cloud_result.success:
                metrics["cloud"] = cloud_result.data

            # Métricas de edge (simuladas)
            metrics["edge"] = await self._collect_edge_metrics()

            # Métricas de ML
            metrics["ml"] = await self._collect_ml_metrics()

            # Métricas federadas
            metrics["federated"] = await self._collect_federated_metrics()

            # Métricas del sistema
            metrics["system"] = await self._collect_system_metrics()

            # Almacenar en historial
            for category, category_metrics in metrics.items():
                if category != "timestamp" and isinstance(category_metrics, dict):
                    if category not in self.metrics_history:
                        self.metrics_history[category] = []
                    self.metrics_history[category].append({
                        "timestamp": metrics["timestamp"],
                        "metrics": category_metrics
                    })

                    # Mantener solo últimas 1000 entradas por categoría
                    if len(self.metrics_history[category]) > 1000:
                        self.metrics_history[category] = self.metrics_history[category][-1000:]

            # Detectar anomalías
            await self._detect_anomalies(metrics)

            return metrics

        except Exception as e:
            logger.error(f"❌ Error recopilando métricas híbridas: {e}")
            return metrics

    async def _collect_edge_metrics(self) -> Dict[str, Any]:
        """Recopilar métricas de dispositivos edge"""

        # Simular métricas de edge
        return {
            "total_devices": 5,
            "online_devices": 4,
            "offline_devices": 1,
            "avg_battery_level": 78.5,
            "avg_temperature": 42.3,
            "avg_network_quality": 0.85,
            "inference_requests": 1250,
            "avg_inference_time": 45.2,  # ms
            "federated_updates": 89
        }

    async def _collect_ml_metrics(self) -> Dict[str, Any]:
        """Recopilar métricas de ML"""

        # Simular métricas de ML
        return {
            "models_loaded": 3,
            "total_predictions": 5670,
            "avg_prediction_time": 23.4,  # ms
            "model_accuracy": {
                "resnet_classifier": 0.921,
                "bert_sentiment": 0.895,
                "cnn_detector": 0.887
            },
            "cache_hit_rate": 0.78,
            "memory_usage": 0.65  # %
        }

    async def _collect_federated_metrics(self) -> Dict[str, Any]:
        """Recopilar métricas de federated learning"""

        # Simular métricas federadas
        return {
            "active_rounds": 1,
            "completed_rounds": 5,
            "total_participants": 4,
            "avg_participants_per_round": 3.8,
            "current_round_progress": 0.65,
            "federated_accuracy_trend": [0.82, 0.85, 0.87, 0.89, 0.91],
            "communication_overhead": 0.12,  # 12%
            "privacy_score": 9.2
        }

    async def _collect_system_metrics(self) -> Dict[str, Any]:
        """Recopilar métricas generales del sistema"""

        # Verificar salud del sistema
        health_result = await self.aegis.client.health_check()

        return {
            "overall_health": health_result.data.get("overall_health", "unknown") if health_result.success else "error",
            "uptime": time.time() - asyncio.get_event_loop().time(),  # Simplificado
            "active_components": 4,  # cloud, edge, ml, federated
            "total_alerts": len(self.alerts),
            "active_anomalies": len([a for a in self.alerts if a.get("status") == "active"])
        }

    async def _detect_anomalies(self, current_metrics: Dict[str, Any]) -> None:
        """Detectar anomalías en las métricas"""

        for detector_name, detector_config in self.anomaly_detectors.items():
            anomaly = await self._check_anomaly(detector_name, detector_config, current_metrics)

            if anomaly:
                alert = {
                    "alert_id": f"alert_{int(time.time())}_{len(self.alerts)}",
                    "type": "anomaly",
                    "detector": detector_name,
                    "severity": anomaly["severity"],
                    "message": anomaly["message"],
                    "metric_value": anomaly["value"],
                    "threshold": anomaly["threshold"],
                    "timestamp": time.time(),
                    "status": "active"
                }

                self.alerts.append(alert)

                logger.warning(f"🚨 ANOMALÍA DETECTADA: {alert['message']}")

    async def _check_anomaly(self, detector_name: str, config: Dict[str, Any],
                           metrics: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Verificar anomalía específica"""

        if detector_name == "cpu_spike":
            # Verificar spike de CPU en cloud
            cloud_metrics = metrics.get("cloud", {})
            for provider, provider_data in cloud_metrics.items():
                cpu_util = provider_data.get("avg_cpu_utilization", 0)
                if cpu_util > config["threshold"]:
                    return {
                        "severity": "warning",
                        "message": f"CPU utilization spike in {provider}: {cpu_util}% (threshold: {config['threshold']}%)",
                        "value": cpu_util,
                        "threshold": config["threshold"]
                    }

        elif detector_name == "memory_leak":
            # Verificar posible memory leak
            edge_metrics = metrics.get("edge", {})
            memory_usage = edge_metrics.get("avg_memory_utilization", 0)
            if memory_usage > config["threshold"]:
                return {
                    "severity": "error",
                    "message": f"Potential memory leak detected: {memory_usage}% utilization",
                    "value": memory_usage,
                    "threshold": config["threshold"]
                }

        elif detector_name == "latency_spike":
            # Verificar latencia elevada
            ml_metrics = metrics.get("ml", {})
            avg_latency = ml_metrics.get("avg_prediction_time", 0)
            if avg_latency > config["threshold"]:
                return {
                    "severity": "warning",
                    "message": f"Inference latency spike: {avg_latency}ms (threshold: {config['threshold']}ms)",
                    "value": avg_latency,
                    "threshold": config["threshold"]
                }

        elif detector_name == "error_rate_spike":
            # Verificar tasa de error elevada
            system_metrics = metrics.get("system", {})
            # Simular cálculo de tasa de error
            error_rate = 2.1  # Simulado
            if error_rate > config["threshold"]:
                return {
                    "severity": "error",
                    "message": f"Error rate spike: {error_rate}% (threshold: {config['threshold']}%)",
                    "value": error_rate,
                    "threshold": config["threshold"]
                }

        return None

    def get_hybrid_dashboard_data(self) -> Dict[str, Any]:
        """Obtener datos para dashboard híbrido"""

        # Datos recientes (últimas 24 horas)
        recent_cloud = self.metrics_history.get("cloud", [])[-24:]  # 1 por hora
        recent_edge = self.metrics_history.get("edge", [])[-1440:]  # 1 por minuto
        recent_ml = self.metrics_history.get("ml", [])[-1440:]
        recent_federated = self.metrics_history.get("federated", [])[-24:]

        return {
            "cloud_overview": self._aggregate_cloud_data(recent_cloud),
            "edge_overview": self._aggregate_edge_data(recent_edge),
            "ml_performance": self._aggregate_ml_data(recent_ml),
            "federated_progress": self._aggregate_federated_data(recent_federated),
            "active_alerts": [a for a in self.alerts[-10:] if a.get("status") == "active"],
            "system_health": self._calculate_system_health()
        }

    def _aggregate_cloud_data(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Agregar datos de cloud"""

        if not data:
            return {"total_cost": 0, "avg_cpu": 0, "avg_memory": 0, "instance_trend": []}

        total_cost = 0
        cpu_values = []
        memory_values = []

        for entry in data:
            for provider, metrics in entry["metrics"].items():
                total_cost += metrics.get("total_cost", 0)
                cpu_values.append(metrics.get("avg_cpu_utilization", 0))
                memory_values.append(metrics.get("avg_memory_utilization", 0))

        return {
            "total_cost": round(total_cost, 2),
            "avg_cpu": round(sum(cpu_values) / len(cpu_values), 1) if cpu_values else 0,
            "avg_memory": round(sum(memory_values) / len(memory_values), 1) if memory_values else 0,
            "cost_trend": [round(sum([e["metrics"][p].get("total_cost", 0) for p in e["metrics"]]), 2) for e in data[-7:]]
        }

    def _aggregate_edge_data(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Agregar datos de edge"""

        if not data:
            return {"online_devices": 0, "avg_battery": 0, "inference_requests": 0}

        total_inference = sum(d["metrics"].get("inference_requests", 0) for d in data)
        avg_battery = sum(d["metrics"].get("avg_battery_level", 0) for d in data) / len(data)
        online_devices = data[-1]["metrics"].get("online_devices", 0) if data else 0

        return {
            "online_devices": online_devices,
            "avg_battery": round(avg_battery, 1),
            "total_inference_requests": total_inference,
            "inference_trend": [d["metrics"].get("inference_requests", 0) for d in data[-10:]]
        }

    def _aggregate_ml_data(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Agregar datos de ML"""

        if not data:
            return {"models_loaded": 0, "avg_latency": 0, "total_predictions": 0}

        total_predictions = sum(d["metrics"].get("total_predictions", 0) for d in data)
        avg_latency = sum(d["metrics"].get("avg_prediction_time", 0) for d in data) / len(data)
        models_loaded = data[-1]["metrics"].get("models_loaded", 0) if data else 0

        return {
            "models_loaded": models_loaded,
            "avg_prediction_latency": round(avg_latency, 1),
            "total_predictions": total_predictions,
            "latency_trend": [d["metrics"].get("avg_prediction_time", 0) for d in data[-10:]]
        }

    def _aggregate_federated_data(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Agregar datos federados"""

        if not data:
            return {"active_rounds": 0, "completed_rounds": 0, "avg_participants": 0}

        completed_rounds = max((d["metrics"].get("completed_rounds", 0) for d in data), default=0)
        active_rounds = data[-1]["metrics"].get("active_rounds", 0) if data else 0
        avg_participants = sum(d["metrics"].get("avg_participants_per_round", 0) for d in data) / len(data)

        return {
            "active_rounds": active_rounds,
            "completed_rounds": completed_rounds,
            "avg_participants": round(avg_participants, 1),
            "accuracy_trend": data[-1]["metrics"].get("federated_accuracy_trend", []) if data else []
        }

    def _calculate_system_health(self) -> Dict[str, Any]:
        """Calcular salud general del sistema"""

        # Lógica simplificada de health score
        active_alerts = len([a for a in self.alerts if a.get("status") == "active"])

        if active_alerts == 0:
            health_score = 100
            status = "excellent"
        elif active_alerts <= 2:
            health_score = 85
            status = "good"
        elif active_alerts <= 5:
            health_score = 70
            status = "warning"
        else:
            health_score = 50
            status = "critical"

        return {
            "score": health_score,
            "status": status,
            "active_alerts": active_alerts,
            "components": {
                "cloud": True,
                "edge": True,
                "ml": True,
                "federated": True
            }
        }

    def generate_hybrid_report(self, time_range: str = "24h") -> Dict[str, Any]:
        """Generar reporte completo del sistema híbrido"""

        # Calcular rango de tiempo
        if time_range == "24h":
            hours_back = 24
        elif time_range == "7d":
            hours_back = 168
        else:
            hours_back = 24

        cutoff_time = time.time() - (hours_back * 3600)

        # Filtrar datos por tiempo
        filtered_data = {}
        for category, data in self.metrics_history.items():
            filtered_data[category] = [d for d in data if d["timestamp"] > cutoff_time]

        # Calcular estadísticas
        report = {
            "time_range": time_range,
            "generated_at": time.time(),
            "summary": {
                "total_data_points": sum(len(data) for data in filtered_data.values()),
                "active_alerts": len([a for a in self.alerts if a.get("timestamp", 0) > cutoff_time]),
                "system_health": self._calculate_system_health()
            },
            "metrics_summary": {},
            "alerts_summary": self._summarize_alerts(cutoff_time),
            "recommendations": self._generate_recommendations(filtered_data)
        }

        # Resumir métricas por categoría
        for category, data in filtered_data.items():
            if data:
                report["metrics_summary"][category] = self._calculate_category_stats(data)

        return report

    def _summarize_alerts(self, cutoff_time: float) -> Dict[str, Any]:
        """Resumir alertas en el período"""

        recent_alerts = [a for a in self.alerts if a.get("timestamp", 0) > cutoff_time]

        alert_counts = {}
        for alert in recent_alerts:
            alert_type = alert.get("type", "unknown")
            severity = alert.get("severity", "unknown")
            key = f"{alert_type}_{severity}"
            alert_counts[key] = alert_counts.get(key, 0) + 1

        return {
            "total_alerts": len(recent_alerts),
            "alerts_by_type": alert_counts,
            "most_common": max(alert_counts.items(), key=lambda x: x[1]) if alert_counts else None
        }

    def _calculate_category_stats(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calcular estadísticas para una categoría"""

        if not data:
            return {}

        # Estadísticas básicas
        timestamps = [d["timestamp"] for d in data]
        duration = max(timestamps) - min(timestamps) if timestamps else 0

        return {
            "data_points": len(data),
            "time_span_hours": round(duration / 3600, 1),
            "avg_interval_seconds": round(duration / len(data), 1) if data else 0
        }

    def _generate_recommendations(self, data: Dict[str, List[Dict[str, Any]]]) -> List[str]:
        """Generar recomendaciones basadas en los datos"""

        recommendations = []

        # Analizar datos de cloud
        cloud_data = data.get("cloud", [])
        if cloud_data:
            latest_cloud = cloud_data[-1]["metrics"]
            total_cost = sum(m.get("total_cost", 0) for m in latest_cloud.values())
            if total_cost > 20:
                recommendations.append("Considerar reserved instances para reducción de costos en cloud")

        # Analizar datos de edge
        edge_data = data.get("edge", [])
        if edge_data:
            latest_edge = edge_data[-1]["metrics"]
            offline_devices = latest_edge.get("offline_devices", 0)
            if offline_devices > 0:
                recommendations.append("Implementar reconexión automática para dispositivos edge offline")

        # Analizar datos de ML
        ml_data = data.get("ml", [])
        if ml_data:
            avg_latencies = [d["metrics"].get("avg_prediction_time", 0) for d in ml_data]
            if avg_latencies and sum(avg_latencies) / len(avg_latencies) > 100:
                recommendations.append("Optimizar modelos ML para reducir latencia de inferencia")

        # Recomendaciones generales
        recommendations.extend([
            "Implementar monitoreo distribuido con alertas automáticas",
            "Configurar políticas de auto-scaling basadas en métricas",
            "Establecer backups automáticos y disaster recovery",
            "Implementar rotación de logs y retención de métricas"
        ])

        return recommendations

async def demo_hybrid_monitoring():
    """Demostración del sistema de monitoreo híbrido"""

    print("📊 DEMO - AEGIS Hybrid Monitoring System")
    print("=" * 50)

    monitoring = HybridMonitoringSystem()
    await monitoring.initialize_monitoring()

    print("✅ Sistema de monitoreo híbrido inicializado")

    # Simular recopilación de métricas durante un período
    print("
📈 Simulando recopilación de métricas..."    for i in range(10):
        metrics = await monitoring.collect_hybrid_metrics()
        print(f"✅ Recopiladas métricas #{i+1} - Componentes: {len(metrics) - 1}")

        # Simular algunos alerts
        if i == 3:
            # Simular alerta manual
            monitoring.alerts.append({
                "alert_id": f"manual_alert_{i}",
                "type": "manual",
                "severity": "info",
                "message": "Simulación de alerta manual",
                "timestamp": time.time(),
                "status": "active"
            })

        await asyncio.sleep(1)

    # Obtener datos del dashboard
    dashboard_data = monitoring.get_hybrid_dashboard_data()
    print("
📊 DATOS DEL DASHBOARD:"    print(f"   ☁️ Cloud Cost: ${dashboard_data['cloud_overview']['total_cost']}/hora")
    print(f"   🛠️ Edge Devices Online: {dashboard_data['edge_overview']['online_devices']}")
    print(f"   🧠 ML Models: {dashboard_data['ml_performance']['models_loaded']}")
    print(f"   🤝 Federated Rounds: {dashboard_data['federated_progress']['completed_rounds']}")
    print(f"   ❤️ System Health: {dashboard_data['system_health']['status']} ({dashboard_data['system_health']['score']}%)")

    # Mostrar alertas activas
    active_alerts = dashboard_data['active_alerts']
    print("
🚨 ALERTAS ACTIVAS:"    if active_alerts:
        for alert in active_alerts:
            print(f"   • {alert['type']} ({alert['severity']}): {alert['message']}")
    else:
        print("   ✅ No hay alertas activas")

    # Generar reporte
    report = monitoring.generate_hybrid_report("24h")
    print("
📋 REPORTE DE 24 HORAS:"    print(f"   📊 Puntos de datos: {report['summary']['total_data_points']}")
    print(f"   🚨 Alertas: {report['alerts_summary']['total_alerts']}")
    print(f"   ❤️ Salud del sistema: {report['summary']['system_health']['status']}")

    # Mostrar recomendaciones
    recommendations = report['recommendations']
    print("
💡 RECOMENDACIONES:"    for rec in recommendations[:3]:  # Mostrar primeras 3
        print(f"   • {rec}")

    print("
🎉 DEMO COMPLETA EXITOSA!"    print("🌟 Sistema de monitoreo híbrido completamente operativo")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(demo_hybrid_monitoring())
'''

    def _get_hybrid_config_template(self) -> str:
        """Configuración completa del sistema híbrido"""
        return '''{
  "hybrid_system": {
    "name": "aegis_production_system",
    "version": "3.3.0",
    "environment": "production",

    "cloud_config": {
      "providers": [
        {
          "name": "aws",
          "regions": ["us-east-1", "us-west-2"],
          "default_instance_type": "t3.medium",
          "auto_scaling": {
            "min_instances": 2,
            "max_instances": 10,
            "cpu_target": 70.0,
            "memory_target": 75.0
          },
          "cost_limits": {
            "daily_budget": 100.0,
            "monthly_budget": 3000.0
          }
        },
        {
          "name": "gcp",
          "regions": ["us-central1", "us-west1"],
          "default_instance_type": "e2-standard-2",
          "auto_scaling": {
            "min_instances": 1,
            "max_instances": 8,
            "cpu_target": 75.0,
            "memory_target": 80.0
          },
          "cost_limits": {
            "daily_budget": 80.0,
            "monthly_budget": 2400.0
          }
        }
      ],
      "load_balancer": {
        "type": "application",
        "ssl_enabled": true,
        "health_check_interval": 30,
        "session_stickiness": false
      }
    },

    "edge_config": {
      "device_types": [
        {
          "type": "raspberry_pi",
          "capabilities": ["inference_only", "data_collection"],
          "optimization": "quantization",
          "max_devices": 50
        },
        {
          "type": "jetson_nano",
          "capabilities": ["inference_only", "training_mini_batch", "federated_client"],
          "optimization": "tensorrt",
          "max_devices": 20
        },
        {
          "type": "coral_dev_board",
          "capabilities": ["inference_only", "real_time_processing"],
          "optimization": "tflite",
          "max_devices": 30
        }
      ],
      "federated_learning": {
        "min_participants": 3,
        "max_rounds": 50,
        "aggregation_timeout": 600,
        "privacy_level": "high",
        "differential_privacy": {
          "enabled": true,
          "noise_multiplier": 0.1,
          "max_grad_norm": 1.0
        }
      },
      "network": {
        "protocols": ["mqtt", "websocket", "http"],
        "compression": true,
        "encryption": "tls_1_3"
      }
    },

    "ml_config": {
      "frameworks": {
        "tensorflow": {
          "versions": ["2.11", "2.12"],
          "optimizations": ["quantization", "pruning", "distillation"]
        },
        "pytorch": {
          "versions": ["1.13", "2.0"],
          "optimizations": ["quantization", "pruning", "torchscript"]
        }
      },
      "model_registry": {
        "storage_backend": "minio",
        "versioning": true,
        "access_control": true
      },
      "inference": {
        "batch_size": 32,
        "timeout_ms": 5000,
        "caching": true,
        "load_balancing": true
      }
    },

    "monitoring_config": {
      "enabled": true,
      "collection_interval": 30,
      "metrics_retention_days": 30,
      "alerts": {
        "cpu_threshold": 80,
        "memory_threshold": 85,
        "latency_threshold": 200,
        "error_rate_threshold": 5
      },
      "dashboards": {
        "cloud_performance": true,
        "edge_monitoring": true,
        "ml_metrics": true,
        "federated_progress": true,
        "system_health": true
      }
    },

    "security_config": {
      "encryption": {
        "data_at_rest": "aes_256_gcm",
        "data_in_transit": "tls_1_3",
        "key_rotation_days": 30
      },
      "authentication": {
        "method": "oauth2_jwt",
        "mfa_required": true,
        "session_timeout": 3600
      },
      "network_security": {
        "firewall_rules": true,
        "ddos_protection": true,
        "intrusion_detection": true
      }
    },

    "orchestration_config": {
      "workload_policies": {
        "inference": {
          "preferred_target": "edge",
          "fallback_targets": ["cloud"],
          "performance_target": 50,
          "cost_priority": "low"
        },
        "training": {
          "preferred_target": "cloud",
          "fallback_targets": ["distributed"],
          "performance_target": 3600,
          "cost_priority": "medium"
        },
        "federated_learning": {
          "preferred_target": "distributed",
          "fallback_targets": ["cloud"],
          "performance_target": 1800,
          "cost_priority": "high"
        }
      },
      "resource_limits": {
        "max_cpu_percent": 90,
        "max_memory_percent": 85,
        "max_network_bandwidth": 1000,
        "max_storage_percent": 80
      }
    },

    "backup_config": {
      "enabled": true,
      "schedule": "daily",
      "retention_days": 30,
      "storage_backend": "s3",
      "encryption": true,
      "cross_region_replication": true
    },

    "logging_config": {
      "level": "INFO",
      "format": "json",
      "centralized": true,
      "retention_days": 90,
      "alert_on_errors": true
    }
  }
}'''
