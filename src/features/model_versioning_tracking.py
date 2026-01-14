#!/usr/bin/env python3
"""
📊 AEGIS Model Versioning & Experiment Tracking - Sprint 4.1
Sistema completo de MLOps para versionado, tracking y gestión de modelos
"""

import asyncio
import time
import json
import hashlib
import pickle
import copy
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
import logging
import uuid

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelStage(Enum):
    """Estados del modelo en el lifecycle"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"
    DEPRECATED = "deprecated"

class ExperimentStatus(Enum):
    """Estados de experimento"""
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"

@dataclass
class ModelVersion:
    """Versión específica de un modelo"""
    model_name: str
    version: str
    model_id: str
    framework: str
    architecture: str
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    # Metadata del modelo
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)

    # Información de entrenamiento
    training_data_hash: str = ""
    training_config: Dict[str, Any] = field(default_factory=dict)
    training_duration: float = 0.0

    # Información de dataset
    dataset_info: Dict[str, Any] = field(default_factory=dict)

    # Lineage
    parent_versions: List[str] = field(default_factory=list)
    child_versions: List[str] = field(default_factory=list)

    # Estado y staging
    stage: ModelStage = ModelStage.DEVELOPMENT
    stage_changed_at: Optional[datetime] = None

    # Archivos y artifacts
    model_path: Optional[str] = None
    artifacts: Dict[str, str] = field(default_factory=dict)  # path -> hash

    # Metadata adicional
    description: str = ""
    author: str = ""
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario para serialización"""
        data = asdict(self)
        # Convertir enums a strings
        data["stage"] = self.stage.value
        # Convertir datetimes a strings
        data["created_at"] = self.created_at.isoformat()
        data["updated_at"] = self.updated_at.isoformat()
        if self.stage_changed_at:
            data["stage_changed_at"] = self.stage_changed_at.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelVersion':
        """Crear instancia desde diccionario"""
        # Convertir strings a enums
        data["stage"] = ModelStage(data["stage"])
        # Convertir strings a datetimes
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        data["updated_at"] = datetime.fromisoformat(data["updated_at"])
        if data.get("stage_changed_at"):
            data["stage_changed_at"] = datetime.fromisoformat(data["stage_changed_at"])
        return cls(**data)

    def calculate_hash(self) -> str:
        """Calcular hash único de la versión"""
        content = f"{self.model_name}{self.version}{self.model_id}"
        content += json.dumps(self.hyperparameters, sort_keys=True)
        content += json.dumps(self.metrics, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def promote_to_stage(self, new_stage: ModelStage, notes: str = ""):
        """Promover modelo a un nuevo stage"""
        if new_stage == self.stage:
            return

        self.stage = new_stage
        self.stage_changed_at = datetime.now()
        self.updated_at = datetime.now()

        if notes:
            self.notes += f"\\n[{datetime.now().isoformat()}] Promoted to {new_stage.value}: {notes}"

        logger.info(f"📈 Model {self.model_name} v{self.version} promoted to {new_stage.value}")

@dataclass
class ExperimentRun:
    """Ejecución individual de experimento"""
    experiment_id: str
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""

    # Estado
    status: ExperimentStatus = ExperimentStatus.RUNNING
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Configuración del experimento
    config: Dict[str, Any] = field(default_factory=dict)

    # Métricas y resultados
    metrics: Dict[str, Any] = field(default_factory=dict)
    artifacts: Dict[str, str] = field(default_factory=dict)  # name -> path

    # Modelos generados
    model_versions: List[str] = field(default_factory=list)  # version IDs

    # Información del sistema
    system_info: Dict[str, Any] = field(default_factory=dict)
    environment: Dict[str, Any] = field(default_factory=dict)

    # Logs y notas
    logs: List[Dict[str, Any]] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario"""
        data = asdict(self)
        data["status"] = self.status.value
        data["created_at"] = self.created_at.isoformat()
        if self.started_at:
            data["started_at"] = self.started_at.isoformat()
        if self.completed_at:
            data["completed_at"] = self.completed_at.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExperimentRun':
        """Crear instancia desde diccionario"""
        data["status"] = ExperimentStatus(data["status"])
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        if data.get("started_at"):
            data["started_at"] = datetime.fromisoformat(data["started_at"])
        if data.get("completed_at"):
            data["completed_at"] = datetime.fromisoformat(data["completed_at"])
        return cls(**data)

    def start(self):
        """Marcar experimento como iniciado"""
        self.started_at = datetime.now()
        self.status = ExperimentStatus.RUNNING

    def complete(self, success: bool = True):
        """Marcar experimento como completado"""
        self.completed_at = datetime.now()
        self.status = ExperimentStatus.COMPLETED if success else ExperimentStatus.FAILED

    def log_metric(self, key: str, value: Any, step: Optional[int] = None):
        """Registrar métrica"""
        if "metrics" not in self.metrics:
            self.metrics["metrics"] = {}

        if step is not None:
            if key not in self.metrics["metrics"]:
                self.metrics["metrics"][key] = []
            self.metrics["metrics"][key].append({"step": step, "value": value})
        else:
            self.metrics["metrics"][key] = value

    def log_param(self, key: str, value: Any):
        """Registrar parámetro"""
        if "params" not in self.metrics:
            self.metrics["params"] = {}
        self.metrics["params"][key] = value

    def add_log(self, level: str, message: str, **kwargs):
        """Agregar log entry"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "level": level,
            "message": message,
            **kwargs
        }
        self.logs.append(log_entry)

@dataclass
class Experiment:
    """Experimento que contiene múltiples runs"""
    experiment_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    # Tags y metadata
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Runs del experimento
    runs: List[ExperimentRun] = field(default_factory=list)

    # Métricas agregadas
    best_run_id: Optional[str] = None
    best_metric: Optional[str] = None
    best_value: Optional[float] = None

    def add_run(self, run: ExperimentRun):
        """Agregar run al experimento"""
        self.runs.append(run)
        self.updated_at = datetime.now()

    def get_runs_by_status(self, status: ExperimentStatus) -> List[ExperimentRun]:
        """Obtener runs por estado"""
        return [run for run in self.runs if run.status == status]

    def get_best_run(self, metric: str, higher_is_better: bool = True) -> Optional[ExperimentRun]:
        """Obtener mejor run basado en métrica"""
        completed_runs = [run for run in self.runs if run.status == ExperimentStatus.COMPLETED]

        if not completed_runs:
            return None

        def get_metric_value(run: ExperimentRun) -> float:
            metrics = run.metrics.get("metrics", {})
            if isinstance(metrics.get(metric), list):
                # Si es una lista (por steps), tomar el último valor
                return metrics[metric][-1]["value"] if metrics[metric] else float('-inf')
            return metrics.get(metric, float('-inf'))

        best_run = max(completed_runs, key=get_metric_value) if higher_is_better else min(completed_runs, key=get_metric_value)

        # Actualizar best run del experimento
        if get_metric_value(best_run) != float('-inf'):
            self.best_run_id = best_run.run_id
            self.best_metric = metric
            self.best_value = get_metric_value(best_run)

        return best_run

class ModelRegistry:
    """Registro centralizado de modelos"""

    def __init__(self):
        self.models: Dict[str, Dict[str, ModelVersion]] = {}  # model_name -> version -> ModelVersion
        self.storage_path: Optional[Path] = None

    def set_storage_path(self, path: Union[str, Path]):
        """Configurar path de almacenamiento"""
        self.storage_path = Path(path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

    def register_model(self, model_version: ModelVersion) -> bool:
        """Registrar nueva versión de modelo"""
        if model_version.model_name not in self.models:
            self.models[model_version.model_name] = {}

        # Verificar que la versión no existe
        if model_version.version in self.models[model_version.model_name]:
            logger.warning(f"Version {model_version.version} already exists for model {model_version.model_name}")
            return False

        self.models[model_version.model_name][model_version.version] = model_version

        # Guardar en disco si hay storage configurado
        if self.storage_path:
            self._save_model_version(model_version)

        logger.info(f"✅ Model {model_version.model_name} v{model_version.version} registered")
        return True

    def get_model_version(self, model_name: str, version: str) -> Optional[ModelVersion]:
        """Obtener versión específica de modelo"""
        return self.models.get(model_name, {}).get(version)

    def get_latest_version(self, model_name: str, stage: Optional[ModelStage] = None) -> Optional[ModelVersion]:
        """Obtener última versión de un modelo"""
        model_versions = self.models.get(model_name, {})

        if not model_versions:
            return None

        versions = list(model_versions.values())

        if stage:
            versions = [v for v in versions if v.stage == stage]

        if not versions:
            return None

        return max(versions, key=lambda v: v.created_at)

    def list_models(self, stage: Optional[ModelStage] = None) -> List[ModelVersion]:
        """Listar modelos"""
        all_versions = []
        for model_versions in self.models.values():
            for version in model_versions.values():
                if stage is None or version.stage == stage:
                    all_versions.append(version)

        return sorted(all_versions, key=lambda v: v.updated_at, reverse=True)

    def promote_model(self, model_name: str, version: str, new_stage: ModelStage,
                     notes: str = "") -> bool:
        """Promover modelo a nuevo stage"""
        model_version = self.get_model_version(model_name, version)

        if not model_version:
            logger.error(f"Model {model_name} v{version} not found")
            return False

        model_version.promote_to_stage(new_stage, notes)

        # Guardar cambios
        if self.storage_path:
            self._save_model_version(model_version)

        return True

    def delete_model_version(self, model_name: str, version: str) -> bool:
        """Eliminar versión de modelo"""
        if model_name in self.models and version in self.models[model_name]:
            del self.models[model_name][version]

            # Eliminar archivos si existen
            if self.storage_path:
                model_file = self.storage_path / f"{model_name}_{version}.json"
                if model_file.exists():
                    model_file.unlink()

            logger.info(f"🗑️ Model {model_name} v{version} deleted")
            return True

        return False

    def _save_model_version(self, model_version: ModelVersion):
        """Guardar versión de modelo en disco"""
        if not self.storage_path:
            return

        filename = f"{model_version.model_name}_{model_version.version}.json"
        filepath = self.storage_path / filename

        with open(filepath, 'w') as f:
            json.dump(model_version.to_dict(), f, indent=2)

    def _load_model_version(self, filepath: Path) -> Optional[ModelVersion]:
        """Cargar versión de modelo desde disco"""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            return ModelVersion.from_dict(data)
        except Exception as e:
            logger.error(f"Error loading model version from {filepath}: {e}")
            return None

    def load_from_disk(self):
        """Cargar todos los modelos desde disco"""
        if not self.storage_path:
            return

        for json_file in self.storage_path.glob("*.json"):
            model_version = self._load_model_version(json_file)
            if model_version:
                if model_version.model_name not in self.models:
                    self.models[model_version.model_name] = {}
                self.models[model_version.model_name][model_version.version] = model_version

class ExperimentTracker:
    """Tracker de experimentos"""

    def __init__(self):
        self.experiments: Dict[str, Experiment] = {}
        self.active_runs: Dict[str, ExperimentRun] = {}  # run_id -> run
        self.storage_path: Optional[Path] = None

    def set_storage_path(self, path: Union[str, Path]):
        """Configurar path de almacenamiento"""
        self.storage_path = Path(path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

    def create_experiment(self, name: str, description: str = "",
                         tags: List[str] = None) -> str:
        """Crear nuevo experimento"""
        experiment = Experiment(
            name=name,
            description=description,
            tags=tags or []
        )

        self.experiments[experiment.experiment_id] = experiment

        # Guardar en disco
        if self.storage_path:
            self._save_experiment(experiment)

        logger.info(f"🧪 Experiment '{name}' created with ID: {experiment.experiment_id}")
        return experiment.experiment_id

    def start_run(self, experiment_id: str, name: str = "", config: Dict[str, Any] = None) -> Optional[str]:
        """Iniciar nueva run de experimento"""
        if experiment_id not in self.experiments:
            logger.error(f"Experiment {experiment_id} not found")
            return None

        run = ExperimentRun(
            experiment_id=experiment_id,
            name=name or f"Run {len(self.experiments[experiment_id].runs) + 1}",
            config=config or {}
        )

        run.start()

        # Agregar al experimento
        self.experiments[experiment_id].add_run(run)
        self.active_runs[run.run_id] = run

        # Guardar
        if self.storage_path:
            self._save_experiment(self.experiments[experiment_id])

        logger.info(f"▶️ Started run {run.run_id} in experiment {experiment_id}")
        return run.run_id

    def log_metric(self, run_id: str, key: str, value: Any, step: Optional[int] = None):
        """Log metric to active run"""
        if run_id in self.active_runs:
            self.active_runs[run_id].log_metric(key, value, step)

    def log_param(self, run_id: str, key: str, value: Any):
        """Log parameter to active run"""
        if run_id in self.active_runs:
            self.active_runs[run_id].log_param(key, value)

    def log_artifact(self, run_id: str, name: str, path: str):
        """Log artifact to run"""
        if run_id in self.active_runs:
            self.active_runs[run_id].artifacts[name] = path

    def end_run(self, run_id: str, success: bool = True):
        """Terminar run"""
        if run_id in self.active_runs:
            run = self.active_runs[run_id]
            run.complete(success)

            # Actualizar experimento
            experiment = self.experiments[run.experiment_id]
            experiment.updated_at = datetime.now()

            # Guardar
            if self.storage_path:
                self._save_experiment(experiment)

            del self.active_runs[run_id]

            logger.info(f"⏹️ Run {run_id} {'completed' if success else 'failed'}")

    def get_experiment(self, experiment_id: str) -> Optional[Experiment]:
        """Obtener experimento"""
        return self.experiments.get(experiment_id)

    def list_experiments(self, tags: List[str] = None) -> List[Experiment]:
        """Listar experimentos"""
        experiments = list(self.experiments.values())

        if tags:
            experiments = [exp for exp in experiments if any(tag in exp.tags for tag in tags)]

        return sorted(experiments, key=lambda e: e.updated_at, reverse=True)

    def compare_runs(self, run_ids: List[str], metrics: List[str] = None) -> Dict[str, Any]:
        """Comparar múltiples runs"""
        comparison = {}

        for run_id in run_ids:
            for exp in self.experiments.values():
                for run in exp.runs:
                    if run.run_id == run_id:
                        run_data = {
                            "run_id": run.run_id,
                            "experiment": exp.name,
                            "status": run.status.value,
                            "metrics": run.metrics.get("metrics", {}),
                            "params": run.metrics.get("params", {})
                        }

                        if metrics:
                            run_data["metrics"] = {k: v for k, v in run_data["metrics"].items() if k in metrics}

                        comparison[run_id] = run_data
                        break

        return comparison

    def _save_experiment(self, experiment: Experiment):
        """Guardar experimento en disco"""
        if not self.storage_path:
            return

        filename = f"experiment_{experiment.experiment_id}.json"
        filepath = self.storage_path / filename

        with open(filepath, 'w') as f:
            json.dump({
                "experiment": asdict(experiment),
                "runs": [run.to_dict() for run in experiment.runs]
            }, f, indent=2, default=str)

class AEGISModelOps:
    """Sistema completo de MLOps para AEGIS"""

    def __init__(self):
        self.model_registry = ModelRegistry()
        self.experiment_tracker = ExperimentTracker()
        self.storage_initialized = False

    def initialize_storage(self, base_path: Union[str, Path]):
        """Inicializar almacenamiento persistente"""
        base_path = Path(base_path)
        base_path.mkdir(parents=True, exist_ok=True)

        # Configurar storage para registry y tracker
        self.model_registry.set_storage_path(base_path / "models")
        self.experiment_tracker.set_storage_path(base_path / "experiments")

        # Cargar datos existentes
        self.model_registry.load_from_disk()

        self.storage_initialized = True
        logger.info(f"💾 MLOps storage initialized at {base_path}")

    async def create_experiment(self, name: str, description: str = "",
                               tags: List[str] = None) -> str:
        """Crear experimento"""
        return self.experiment_tracker.create_experiment(name, description, tags)

    async def start_run(self, experiment_id: str, name: str = "",
                       config: Dict[str, Any] = None) -> Optional[str]:
        """Iniciar run"""
        return self.experiment_tracker.start_run(experiment_id, name, config)

    async def log_to_run(self, run_id: str, **kwargs):
        """Log información a run activa"""
        for key, value in kwargs.items():
            if key.startswith("metric_"):
                metric_name = key[7:]  # Remove "metric_" prefix
                self.experiment_tracker.log_metric(run_id, metric_name, value)
            elif key.startswith("param_"):
                param_name = key[6:]  # Remove "param_" prefix
                self.experiment_tracker.log_param(run_id, param_name, value)
            elif key.startswith("artifact_"):
                artifact_name = key[9:]  # Remove "artifact_" prefix
                self.experiment_tracker.log_artifact(run_id, artifact_name, str(value))

    async def end_run(self, run_id: str, success: bool = True):
        """Terminar run"""
        self.experiment_tracker.end_run(run_id, success)

    async def register_model_version(self, model_version: ModelVersion) -> bool:
        """Registrar versión de modelo"""
        return self.model_registry.register_model(model_version)

    async def promote_model(self, model_name: str, version: str,
                           stage: ModelStage, notes: str = "") -> bool:
        """Promover modelo"""
        return self.model_registry.promote_model(model_name, version, stage, notes)

    def get_model_info(self, model_name: str, version: Optional[str] = None) -> Optional[ModelVersion]:
        """Obtener información de modelo"""
        if version:
            return self.model_registry.get_model_version(model_name, version)
        else:
            return self.model_registry.get_latest_version(model_name)

    def list_models(self, stage: Optional[ModelStage] = None) -> List[ModelVersion]:
        """Listar modelos"""
        return self.model_registry.list_models(stage)

    def get_experiment_info(self, experiment_id: str) -> Optional[Experiment]:
        """Obtener información de experimento"""
        return self.experiment_tracker.get_experiment(experiment_id)

    def list_experiments(self, tags: List[str] = None) -> List[Experiment]:
        """Listar experimentos"""
        return self.experiment_tracker.list_experiments(tags)

    def compare_experiments(self, experiment_ids: List[str]) -> Dict[str, Any]:
        """Comparar experimentos"""
        comparison = {}

        for exp_id in experiment_ids:
            exp = self.experiment_tracker.get_experiment(exp_id)
            if exp:
                comparison[exp_id] = {
                    "name": exp.name,
                    "runs": len(exp.runs),
                    "completed_runs": len(exp.get_runs_by_status(ExperimentStatus.COMPLETED)),
                    "best_run": exp.best_run_id,
                    "best_metric": exp.best_metric,
                    "best_value": exp.best_value
                }

        return comparison

# ===== DEMO Y EJEMPLOS =====

async def demo_model_versioning_tracking():
    """Demostración completa del sistema de versioning y tracking"""

    print("📊 AEGIS Model Versioning & Experiment Tracking Demo")
    print("=" * 55)

    # Inicializar sistema MLOps
    mlops = AEGISModelOps()
    mlops.initialize_storage("./mlops_storage")

    print("✅ Sistema MLOps inicializado")

    # ===== DEMO 1: EXPERIMENT TRACKING =====
    print("\\n🧪 DEMO 1: Experiment Tracking")
    print("-" * 35)

    # Crear experimento
    exp_id = await mlops.create_experiment(
        name="Image Classification Optimization",
        description="Experimentando con diferentes arquitecturas para clasificación de imágenes",
        tags=["classification", "computer_vision", "optimization"]
    )

    print(f"✅ Experimento creado: {exp_id}")

    # Ejecutar múltiples runs
    architectures = ["resnet50", "efficientnet", "mobilenet"]
    runs_data = []

    for arch in architectures:
        print(f"\\n🏃 Ejecutando run con {arch}...")

        # Iniciar run
        run_id = await mlops.start_run(
            exp_id,
            name=f"Training {arch}",
            config={
                "architecture": arch,
                "batch_size": 32,
                "learning_rate": 0.001,
                "epochs": 10
            }
        )

        if not run_id:
            print(f"❌ Error iniciando run para {arch}")
            continue

        # Simular entrenamiento y logging
        for epoch in range(1, 6):  # 5 epochs para demo
            # Simular métricas
            accuracy = 0.5 + (epoch * 0.1) + (architectures.index(arch) * 0.05) + np.random.normal(0, 0.02)
            loss = 2.0 - (epoch * 0.2) - (architectures.index(arch) * 0.1) + np.random.normal(0, 0.1)

            # Log metrics
            await mlops.log_to_run(run_id,
                                  metric_accuracy=accuracy,
                                  metric_loss=loss,
                                  metric_epoch=epoch)

            await asyncio.sleep(0.1)  # Simular tiempo de entrenamiento

        # Log final parameters
        await mlops.log_to_run(run_id,
                              param_final_accuracy=accuracy,
                              param_architecture=arch,
                              param_total_epochs=5)

        # Terminar run
        success = accuracy > 0.7  # Simular éxito basado en accuracy
        await mlops.end_run(run_id, success)

        runs_data.append({
            "run_id": run_id,
            "architecture": arch,
            "final_accuracy": accuracy,
            "success": success
        })

        print(".3f"
    # Obtener información del experimento
    experiment = mlops.get_experiment_info(exp_id)
    if experiment:
        print(f"\\n📈 Experimento '{experiment.name}':")
        print(f"   • Runs totales: {len(experiment.runs)}")
        print(f"   • Runs completados: {len(experiment.get_runs_by_status(ExperimentStatus.COMPLETED))}")

        # Obtener mejor run
        best_run = experiment.get_best_run("accuracy")
        if best_run:
            print(f"   • Mejor run: {best_run.name}")
            print(".3f"
    # ===== DEMO 2: MODEL VERSIONING =====
    print("\\n\\n📦 DEMO 2: Model Versioning & Registry")
    print("-" * 40)

    # Registrar versiones de modelo
    for i, run_data in enumerate(runs_data):
        if run_data["success"]:
            # Crear versión de modelo
            from ml_framework_integration import MLFramework

            model_version = ModelVersion(
                model_name="image_classifier",
                version=f"v1.{i}",
                model_id=f"model_{run_data['run_id']}",
                framework=MLFramework.PYTORCH.value,
                architecture=run_data["architecture"],
                hyperparameters={
                    "learning_rate": 0.001,
                    "batch_size": 32,
                    "architecture": run_data["architecture"]
                },
                metrics={
                    "accuracy": run_data["final_accuracy"],
                    "loss": 0.5  # Simulado
                },
                training_data_hash="dataset_v1_hash",
                training_config={
                    "epochs": 5,
                    "optimizer": "adam",
                    "scheduler": "step"
                },
                tags=["computer_vision", "classification"],
                description=f"Image classifier trained with {run_data['architecture']}",
                author="AEGIS AutoML"
            )

            # Registrar modelo
            success = await mlops.register_model_version(model_version)
            if success:
                print(f"✅ Modelo registrado: {model_version.model_name} {model_version.version}")
            else:
                print(f"❌ Error registrando modelo {model_version.version}")

    # Listar modelos
    all_models = mlops.list_models()
    print(f"\\n📋 Modelos registrados: {len(all_models)}")

    for model in all_models[:3]:  # Mostrar primeros 3
        print(f"   • {model.model_name} {model.version} - {model.architecture} "
              ".3f"              f"({model.stage.value})")

    # Promover mejor modelo a staging
    if all_models:
        best_model = max(all_models, key=lambda m: m.metrics.get("accuracy", 0))
        promoted = await mlops.promote_model(
            best_model.model_name,
            best_model.version,
            ModelStage.STAGING,
            "Promovido automáticamente por demo - mejor accuracy"
        )

        if promoted:
            print(f"\\n📈 Modelo promovido a STAGING: {best_model.model_name} {best_model.version}")

    # ===== DEMO 3: COMPARACIÓN Y ANÁLISIS =====
    print("\\n\\n📊 DEMO 3: Comparación y Análisis")
    print("-" * 35)

    # Comparar experimentos
    experiments = mlops.list_experiments()
    if len(experiments) > 0:
        exp_comparison = mlops.compare_experiments([exp.experiment_id for exp in experiments])

        print("🔍 Comparación de experimentos:")
        for exp_id, data in exp_comparison.items():
            print(f"   • {data['name']}: {data['runs']} runs, "
                  f"{data['completed_runs']} completados")
            if data['best_value']:
                print(".3f"
    # Comparar runs dentro del experimento
    if experiment and len(experiment.runs) > 1:
        run_ids = [run.run_id for run in experiment.runs if run.status == ExperimentStatus.COMPLETED][:3]
        if run_ids:
            runs_comparison = mlops.experiment_tracker.compare_runs(run_ids, ["accuracy", "loss"])

            print("\\n⚖️ Comparación de runs:")
            for run_id, run_data in runs_comparison.items():
                accuracy = run_data["metrics"].get("accuracy", "N/A")
                loss = run_data["metrics"].get("loss", "N/A")
                print(f"   • {run_data['experiment']} - {run_id[:8]}...: acc={accuracy}, loss={loss}")

    # ===== DEMO 4: MODEL LIFECYCLE =====
    print("\\n\\n🔄 DEMO 4: Model Lifecycle Management")
    print("-" * 40)

    # Simular lifecycle completo
    staging_models = mlops.list_models(ModelStage.STAGING)
    if staging_models:
        model_to_promote = staging_models[0]

        print(f"🎯 Gestionando lifecycle de: {model_to_promote.model_name} {model_to_promote.version}")

        # Simular testing en staging
        print("🧪 Testing en STAGING...")
        await asyncio.sleep(1)

        # Promover a producción
        promoted = await mlops.promote_model(
            model_to_promote.model_name,
            model_to_promote.version,
            ModelStage.PRODUCTION,
            "Aprobado por testing automático - métricas satisfactorias"
        )

        if promoted:
            print("✅ Modelo promovido a PRODUCCIÓN")

        # Simular uso en producción
        await asyncio.sleep(0.5)

        # Crear nueva versión (simulando update)
        new_version = ModelVersion(
            model_name=model_to_promote.model_name,
            version=f"v1.{int(model_to_promote.version.split('.')[1]) + 1}",
            model_id=f"updated_{model_to_promote.model_id}",
            framework=model_to_promote.framework,
            architecture=model_to_promote.architecture,
            hyperparameters=model_to_promote.hyperparameters,
            metrics={"accuracy": model_to_promote.metrics["accuracy"] + 0.02},  # Mejorado
            parent_versions=[model_to_promote.version],
            description="Versión actualizada con mejores métricas"
        )

        await mlops.register_model_version(new_version)
        print(f"🆕 Nueva versión creada: {new_version.version}")

        # Archivar versión anterior
        archived = await mlops.promote_model(
            model_to_promote.model_name,
            model_to_promote.version,
            ModelStage.ARCHIVED,
            "Archivado por nueva versión superior"
        )

        if archived:
            print(f"📦 Versión anterior archivada: {model_to_promote.version}")

    # ===== ESTADÍSTICAS FINALES =====
    print("\\n\\n🎉 DEMO COMPLETA - ESTADÍSTICAS FINALES")
    print("=" * 50)

    # Estadísticas del sistema
    total_experiments = len(mlops.list_experiments())
    total_models = len(mlops.list_models())
    production_models = len(mlops.list_models(ModelStage.PRODUCTION))

    print("📊 SISTEMA MLOPS:")
    print(f"   • Experimentos totales: {total_experiments}")
    print(f"   • Modelos registrados: {total_models}")
    print(f"   • Modelos en producción: {production_models}")
    print(f"   • Stages utilizados: Development, Staging, Production, Archived")

    print("\\n🏆 FUNCIONALIDADES DEMOSTRADAS:")
    print("   ✅ Experiment tracking completo")
    print("   ✅ Model versioning automático")
    print("   ✅ Model registry con staging")
    print("   ✅ Comparación de experimentos")
    print("   ✅ Model lifecycle management")
    print("   ✅ Persistent storage")
    print("   ✅ Metrics y parameters logging")

    print("\\n💡 INSIGHTS OBTENIDOS:")
    print("   • El tracking de experimentos es crucial para reproducibilidad")
    print("   • El versioning permite evolución controlada de modelos")
    print("   • Los stages facilitan deployment seguro (dev->staging->prod)")
    print("   • La comparación automática acelera la optimización")

    print("\\n🚀 PRÓXIMOS PASOS PARA MLOps:")
    print("   • Integrar con MLflow o similar para estándares")
    print("   • Agregar model serving automático")
    print("   • Implementar A/B testing de modelos")
    print("   • Crear dashboards de monitoreo")
    print("   • Automatizar retraining basado en data drift")

    print("\\n" + "=" * 60)
    print("🌟 AEGIS MLOps - ¡Experiment Tracking y Versioning completos!")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(demo_model_versioning_tracking())
