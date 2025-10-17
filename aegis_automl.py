#!/usr/bin/env python3
"""
🤖 AEGIS AutoML - Automated Machine Learning - Sprint 4.1
Sistema automático de generación, entrenamiento y optimización de modelos de ML
"""

import asyncio
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
import json
import hashlib
from pathlib import Path
import random

# Importar componentes del framework
from ml_framework_integration import MLFrameworkManager, MLFramework
from advanced_model_optimization import AdvancedModelOptimizer, OptimizationConfig, OptimizationTechnique

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TaskType(Enum):
    """Tipos de tarea de ML soportados"""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    TIME_SERIES = "time_series"
    NLP_CLASSIFICATION = "nlp_classification"
    IMAGE_CLASSIFICATION = "image_classification"
    OBJECT_DETECTION = "object_detection"
    SEGMENTATION = "segmentation"

class DataType(Enum):
    """Tipos de datos soportados"""
    TABULAR = "tabular"
    IMAGE = "image"
    TEXT = "text"
    TIME_SERIES = "time_series"
    MIXED = "mixed"

class ModelArchitecture(Enum):
    """Arquitecturas de modelo disponibles"""
    LINEAR = "linear"
    MLP = "mlp"
    CNN = "cnn"
    RNN = "rnn"
    TRANSFORMER = "transformer"
    RESNET = "resnet"
    EFFICIENTNET = "efficientnet"
    BERT = "bert"
    LSTM = "lstm"

@dataclass
class AutoMLConfig:
    """Configuración para AutoML"""
    task_type: TaskType
    data_type: DataType
    target_metric: str = "accuracy"
    max_time_minutes: int = 60
    max_models: int = 10
    ensemble_size: int = 3
    cv_folds: int = 5
    optimization_budget: float = 0.3  # Porcentaje del tiempo para optimización
    early_stopping: bool = True
    feature_engineering: bool = True
    hyperparameter_tuning: bool = True

@dataclass
class DataAnalysis:
    """Análisis de datos"""
    num_samples: int
    num_features: int
    data_type: DataType
    task_type: TaskType
    missing_values: float
    class_distribution: Optional[Dict[str, int]] = None
    feature_types: Dict[str, str] = field(default_factory=dict)
    correlations: Dict[str, float] = field(default_factory=dict)
    recommended_preprocessing: List[str] = field(default_factory=list)

@dataclass
class ModelCandidate:
    """Candidato de modelo generado por AutoML"""
    model_id: str
    architecture: ModelArchitecture
    framework: MLFramework
    hyperparameters: Dict[str, Any]
    estimated_complexity: float
    expected_accuracy: float
    training_time_estimate: float
    created_at: float = field(default_factory=time.time)

@dataclass
class AutoMLResult:
    """Resultado de AutoML"""
    task_id: str
    best_model_id: str
    models_evaluated: int
    best_score: float
    training_time: float
    models_generated: List[ModelCandidate]
    ensemble_models: List[str]
    optimization_applied: bool
    final_model_id: str
    status: str = "pending"
    created_at: float = field(default_factory=time.time)

class DataAnalyzer:
    """Analizador automático de datos"""

    def __init__(self):
        self.analysis_cache: Dict[str, DataAnalysis] = {}

    async def analyze_dataset(self, data: Any, task_hint: Optional[TaskType] = None) -> DataAnalysis:
        """Analizar dataset automáticamente"""

        logger.info("🔍 Analizando dataset...")

        # Calcular hash para cache
        data_hash = hashlib.md5(str(data).encode()).hexdigest()[:8]

        if data_hash in self.analysis_cache:
            logger.info("✅ Usando análisis cacheado")
            return self.analysis_cache[data_hash]

        # Análisis básico
        if isinstance(data, pd.DataFrame):
            analysis = await self._analyze_tabular_data(data, task_hint)
        elif isinstance(data, dict) and "images" in data:
            analysis = await self._analyze_image_data(data, task_hint)
        elif isinstance(data, dict) and "text" in data:
            analysis = await self._analyze_text_data(data, task_hint)
        else:
            analysis = await self._analyze_generic_data(data, task_hint)

        # Cachear resultado
        self.analysis_cache[data_hash] = analysis

        logger.info("✅ Análisis completado")
        return analysis

    async def _analyze_tabular_data(self, df: pd.DataFrame, task_hint: Optional[TaskType]) -> DataAnalysis:
        """Analizar datos tabulares"""

        await asyncio.sleep(0.5)  # Simular análisis

        num_samples, num_features = df.shape

        # Detectar tipos de feature
        feature_types = {}
        for col in df.columns:
            if df[col].dtype in ['int64', 'float64']:
                if df[col].nunique() < 20:
                    feature_types[col] = "categorical"
                else:
                    feature_types[col] = "numerical"
            else:
                feature_types[col] = "categorical"

        # Calcular valores faltantes
        missing_values = df.isnull().sum().sum() / (num_samples * num_features)

        # Intentar detectar tipo de tarea
        if task_hint:
            task_type = task_hint
        else:
            # Lógica simple para detectar clasificación vs regresión
            target_col = df.columns[-1]
            if df[target_col].dtype == 'object' or df[target_col].nunique() < 20:
                task_type = TaskType.CLASSIFICATION
            else:
                task_type = TaskType.REGRESSION

        # Distribución de clases (si es clasificación)
        class_distribution = None
        if task_type == TaskType.CLASSIFICATION:
            class_distribution = df[target_col].value_counts().to_dict()

        # Correlaciones básicas
        correlations = {}
        if num_features < 50:  # Solo para datasets pequeños
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 1:
                corr_matrix = df[numeric_cols].corr()
                for i, col1 in enumerate(numeric_cols):
                    for col2 in numeric_cols[i+1:]:
                        correlations[f"{col1}_{col2}"] = corr_matrix.loc[col1, col2]

        # Recomendaciones de preprocesamiento
        preprocessing = []
        if missing_values > 0.1:
            preprocessing.append("imputation")
        if any(ft == "categorical" for ft in feature_types.values()):
            preprocessing.append("encoding")
        if task_type == TaskType.CLASSIFICATION and class_distribution:
            minority_class_ratio = min(class_distribution.values()) / sum(class_distribution.values())
            if minority_class_ratio < 0.1:
                preprocessing.append("balancing")

        return DataAnalysis(
            num_samples=num_samples,
            num_features=num_features,
            data_type=DataType.TABULAR,
            task_type=task_type,
            missing_values=missing_values,
            class_distribution=class_distribution,
            feature_types=feature_types,
            correlations=correlations,
            recommended_preprocessing=preprocessing
        )

    async def _analyze_image_data(self, data: Dict[str, Any], task_hint: Optional[TaskType]) -> DataAnalysis:
        """Analizar datos de imágenes"""

        await asyncio.sleep(0.5)

        images = data.get("images", [])
        labels = data.get("labels", [])

        return DataAnalysis(
            num_samples=len(images),
            num_features=3,  # RGB channels
            data_type=DataType.IMAGE,
            task_type=task_hint or TaskType.IMAGE_CLASSIFICATION,
            missing_values=0.0,
            recommended_preprocessing=["normalization", "augmentation"]
        )

    async def _analyze_text_data(self, data: Dict[str, Any], task_hint: Optional[TaskType]) -> DataAnalysis:
        """Analizar datos de texto"""

        await asyncio.sleep(0.5)

        texts = data.get("text", [])
        labels = data.get("labels", [])

        return DataAnalysis(
            num_samples=len(texts),
            num_features=1,  # Text feature
            data_type=DataType.TEXT,
            task_type=task_hint or TaskType.NLP_CLASSIFICATION,
            missing_values=0.0,
            recommended_preprocessing=["tokenization", "embedding"]
        )

    async def _analyze_generic_data(self, data: Any, task_hint: Optional[TaskType]) -> DataAnalysis:
        """Análisis genérico de datos"""

        await asyncio.sleep(0.2)

        return DataAnalysis(
            num_samples=len(data) if hasattr(data, '__len__') else 1,
            num_features=1,
            data_type=DataType.MIXED,
            task_type=task_hint or TaskType.CLASSIFICATION,
            missing_values=0.0
        )

class ArchitectureGenerator:
    """Generador automático de arquitecturas de modelo"""

    def __init__(self):
        self.templates = self._load_architecture_templates()

    def _load_architecture_templates(self) -> Dict[TaskType, List[Dict[str, Any]]]:
        """Cargar templates de arquitectura"""

        return {
            TaskType.CLASSIFICATION: [
                {
                    "architecture": ModelArchitecture.MLP,
                    "framework": MLFramework.PYTORCH,
                    "complexity": 0.3,
                    "hyperparameter_ranges": {
                        "hidden_dims": [[64, 32], [128, 64], [256, 128, 64]],
                        "dropout": [0.1, 0.2, 0.3],
                        "learning_rate": [0.001, 0.01, 0.1]
                    }
                },
                {
                    "architecture": ModelArchitecture.LINEAR,
                    "framework": MLFramework.SCIKIT_LEARN,
                    "complexity": 0.1,
                    "hyperparameter_ranges": {
                        "C": [0.1, 1.0, 10.0],
                        "max_iter": [1000, 2000]
                    }
                }
            ],

            TaskType.IMAGE_CLASSIFICATION: [
                {
                    "architecture": ModelArchitecture.CNN,
                    "framework": MLFramework.PYTORCH,
                    "complexity": 0.5,
                    "hyperparameter_ranges": {
                        "num_conv_layers": [2, 3, 4],
                        "channels": [[32, 64], [64, 128], [32, 64, 128]],
                        "kernel_size": [3, 5],
                        "learning_rate": [0.001, 0.01]
                    }
                },
                {
                    "architecture": ModelArchitecture.RESNET,
                    "framework": MLFramework.PYTORCH,
                    "complexity": 0.8,
                    "hyperparameter_ranges": {
                        "num_layers": [18, 34, 50],
                        "pretrained": [True, False]
                    }
                }
            ],

            TaskType.NLP_CLASSIFICATION: [
                {
                    "architecture": ModelArchitecture.TRANSFORMER,
                    "framework": MLFramework.PYTORCH,
                    "complexity": 0.9,
                    "hyperparameter_ranges": {
                        "model_name": ["bert-base-uncased", "distilbert-base-uncased"],
                        "max_length": [128, 256, 512],
                        "learning_rate": [2e-5, 3e-5, 5e-5]
                    }
                },
                {
                    "architecture": ModelArchitecture.LSTM,
                    "framework": MLFramework.PYTORCH,
                    "complexity": 0.6,
                    "hyperparameter_ranges": {
                        "hidden_dim": [64, 128, 256],
                        "num_layers": [1, 2],
                        "dropout": [0.1, 0.2]
                    }
                }
            ]
        }

    async def generate_candidates(self, analysis: DataAnalysis,
                                config: AutoMLConfig) -> List[ModelCandidate]:
        """Generar candidatos de modelo"""

        logger.info("🏗️ Generando candidatos de modelo...")

        if analysis.task_type not in self.templates:
            raise ValueError(f"No hay templates para el tipo de tarea: {analysis.task_type}")

        templates = self.templates[analysis.task_type]
        candidates = []

        # Generar múltiples candidatos por template
        for template in templates:
            for i in range(min(3, config.max_models // len(templates))):
                candidate = await self._generate_candidate_from_template(
                    template, analysis, i
                )
                candidates.append(candidate)

                if len(candidates) >= config.max_models:
                    break

            if len(candidates) >= config.max_models:
                break

        logger.info(f"✅ Generados {len(candidates)} candidatos de modelo")

        return candidates

    async def _generate_candidate_from_template(self, template: Dict[str, Any],
                                              analysis: DataAnalysis, index: int) -> ModelCandidate:
        """Generar candidato desde template"""

        await asyncio.sleep(0.1)  # Simular generación

        # Generar hyperparámetros aleatorios del rango disponible
        hyperparameters = {}
        for param, values in template["hyperparameter_ranges"].items():
            hyperparameters[param] = random.choice(values)

        # Ajustar hyperparámetros basado en análisis de datos
        hyperparameters = self._adjust_hyperparameters_for_data(
            hyperparameters, analysis, template["architecture"]
        )

        # Estimar complejidad y performance
        complexity = self._estimate_complexity(template, hyperparameters, analysis)
        expected_accuracy = self._estimate_accuracy(template, hyperparameters, analysis)
        training_time = self._estimate_training_time(template, hyperparameters, analysis)

        return ModelCandidate(
            model_id=f"automl_{template['architecture'].value}_{index}_{int(time.time())}",
            architecture=template["architecture"],
            framework=template["framework"],
            hyperparameters=hyperparameters,
            estimated_complexity=complexity,
            expected_accuracy=expected_accuracy,
            training_time_estimate=training_time
        )

    def _adjust_hyperparameters_for_data(self, params: Dict[str, Any],
                                       analysis: DataAnalysis,
                                       architecture: ModelArchitecture) -> Dict[str, Any]:
        """Ajustar hyperparámetros basado en datos"""

        adjusted = params.copy()

        # Ajustes específicos por arquitectura
        if architecture == ModelArchitecture.MLP:
            # Ajustar hidden_dims basado en número de features
            if analysis.num_features > 100:
                adjusted["hidden_dims"] = [256, 128, 64]
            elif analysis.num_features > 50:
                adjusted["hidden_dims"] = [128, 64]

        elif architecture == ModelArchitecture.CNN:
            # Ajustar arquitectura CNN basado en tamaño de imagen
            # (simplificado)
            pass

        return adjusted

    def _estimate_complexity(self, template: Dict[str, Any], params: Dict[str, Any],
                           analysis: DataAnalysis) -> float:
        """Estimar complejidad del modelo"""

        base_complexity = template["complexity"]

        # Ajustar por tamaño de datos
        data_factor = min(1.0, analysis.num_samples / 10000)

        # Ajustar por número de parámetros
        param_factor = 1.0
        if "hidden_dims" in params:
            total_params = sum(params["hidden_dims"]) * analysis.num_features
            param_factor = min(1.0, total_params / 100000)

        return base_complexity * data_factor * param_factor

    def _estimate_accuracy(self, template: Dict[str, Any], params: Dict[str, Any],
                         analysis: DataAnalysis) -> float:
        """Estimar accuracy esperada"""

        # Estimaciones simplificadas basadas en arquitectura
        base_accuracy = {
            ModelArchitecture.LINEAR: 0.75,
            ModelArchitecture.MLP: 0.82,
            ModelArchitecture.CNN: 0.88,
            ModelArchitecture.RESNET: 0.92,
            ModelArchitecture.TRANSFORMER: 0.90,
            ModelArchitecture.LSTM: 0.85
        }.get(template["architecture"], 0.8)

        # Ajustar por calidad de datos
        data_quality_factor = 1.0 - analysis.missing_values

        # Ajustar por complejidad de tarea
        task_complexity = 1.0
        if analysis.task_type == TaskType.IMAGE_CLASSIFICATION:
            task_complexity = 1.2
        elif analysis.task_type == TaskType.NLP_CLASSIFICATION:
            task_complexity = 1.1

        return min(0.95, base_accuracy * data_quality_factor / task_complexity)

    def _estimate_training_time(self, template: Dict[str, Any], params: Dict[str, Any],
                              analysis: DataAnalysis) -> float:
        """Estimar tiempo de entrenamiento en minutos"""

        base_time = 10  # minutos base

        # Factor por arquitectura
        arch_factor = {
            ModelArchitecture.LINEAR: 0.5,
            ModelArchitecture.MLP: 1.0,
            ModelArchitecture.CNN: 2.0,
            ModelArchitecture.RESNET: 3.0,
            ModelArchitecture.TRANSFORMER: 4.0,
            ModelArchitecture.LSTM: 2.5
        }.get(template["architecture"], 1.0)

        # Factor por datos
        data_factor = analysis.num_samples / 1000

        # Factor por complejidad de hyperparámetros
        param_factor = 1.0
        if "num_layers" in params and params["num_layers"] > 2:
            param_factor = 1.5

        return base_time * arch_factor * data_factor * param_factor

class AutoMLTrainer:
    """Entrenador automático para AutoML"""

    def __init__(self, ml_manager: MLFrameworkManager):
        self.ml_manager = ml_manager
        self.training_results: Dict[str, Dict[str, Any]] = {}

    async def train_and_evaluate(self, candidate: ModelCandidate,
                               train_data: Any, val_data: Any,
                               config: AutoMLConfig) -> Dict[str, Any]:
        """Entrenar y evaluar un candidato"""

        logger.info(f"🏋️ Entrenando modelo {candidate.model_id}")

        start_time = time.time()

        try:
            # Simular entrenamiento
            await asyncio.sleep(candidate.training_time_estimate * 6)  # Simular tiempo real

            # Generar métricas simuladas
            score = candidate.expected_accuracy + random.uniform(-0.05, 0.05)
            score = max(0.1, min(0.99, score))  # Clamp to reasonable range

            # Métricas adicionales
            if config.task_type == TaskType.CLASSIFICATION:
                metrics = {
                    "accuracy": score,
                    "precision": score - 0.02,
                    "recall": score + 0.01,
                    "f1": score,
                    "training_time": time.time() - start_time
                }
            else:
                metrics = {
                    "mse": 1.0 - score,
                    "mae": 1.0 - score,
                    "r2": score,
                    "training_time": time.time() - start_time
                }

            self.training_results[candidate.model_id] = {
                "metrics": metrics,
                "status": "completed",
                "candidate": candidate
            }

            logger.info(".3f"            return metrics

        except Exception as e:
            logger.error(f"❌ Error entrenando {candidate.model_id}: {e}")
            self.training_results[candidate.model_id] = {
                "metrics": {},
                "status": "failed",
                "error": str(e)
            }
            return {}

class AEGISAutoML:
    """Sistema completo de AutoML"""

    def __init__(self):
        self.ml_manager = MLFrameworkManager()
        self.data_analyzer = DataAnalyzer()
        self.architecture_generator = ArchitectureGenerator()
        self.trainer = AutoMLTrainer(self.ml_manager)
        self.optimizer = AdvancedModelOptimizer(self.ml_manager)
        self.completed_runs: List[AutoMLResult] = []

    async def run_automl(self, train_data: Any, config: AutoMLConfig,
                        test_data: Optional[Any] = None) -> AutoMLResult:
        """Ejecutar AutoML completo"""

        task_id = f"automl_{int(time.time())}_{hash(str(train_data)) % 10000}"
        logger.info(f"🚀 Iniciando AutoML: {task_id}")

        start_time = time.time()

        try:
            # Paso 1: Análisis de datos
            logger.info("📊 Analizando datos...")
            analysis = await self.data_analyzer.analyze_dataset(train_data, config.task_type)

            # Paso 2: Generar candidatos
            logger.info("🏗️ Generando arquitecturas...")
            candidates = await self.architecture_generator.generate_candidates(analysis, config)

            # Paso 3: Entrenar y evaluar
            logger.info("🏋️ Entrenando modelos...")
            evaluation_results = []
            for candidate in candidates:
                if time.time() - start_time > config.max_time_minutes * 60 * 0.7:  # 70% del tiempo
                    logger.warning("⏰ Tiempo límite alcanzado, deteniendo entrenamiento")
                    break

                metrics = await self.trainer.train_and_evaluate(
                    candidate, train_data, test_data, config
                )

                if metrics:
                    evaluation_results.append((candidate, metrics))

            if not evaluation_results:
                raise RuntimeError("No se pudieron entrenar modelos exitosamente")

            # Paso 4: Seleccionar mejor modelo
            best_candidate, best_metrics = max(
                evaluation_results,
                key=lambda x: x[1][config.target_metric]
            )

            logger.info(f"🏆 Mejor modelo: {best_candidate.model_id} "
                       ".3f"
            # Paso 5: Crear ensemble (opcional)
            ensemble_models = []
            if config.ensemble_size > 1 and len(evaluation_results) >= config.ensemble_size:
                top_candidates = sorted(
                    evaluation_results,
                    key=lambda x: x[1][config.target_metric],
                    reverse=True
                )[:config.ensemble_size]

                ensemble_models = [c.model_id for c, _ in top_candidates]
                logger.info(f"🤝 Ensemble creado con {len(ensemble_models)} modelos")

            # Paso 6: Optimizar mejor modelo (opcional)
            optimization_budget_seconds = config.max_time_minutes * 60 * config.optimization_budget
            time_remaining = config.max_time_minutes * 60 - (time.time() - start_time)

            final_model_id = best_candidate.model_id
            optimization_applied = False

            if time_remaining > optimization_budget_seconds and optimization_budget_seconds > 60:
                logger.info("🔧 Aplicando optimización al mejor modelo...")

                opt_config = OptimizationConfig(
                    technique=OptimizationTechnique.QUANTIZATION,
                    target_platform="cpu",
                    compression_ratio=0.5
                )

                opt_result = await self.optimizer.optimize_model(best_candidate.model_id, opt_config)

                if opt_result.status == "completed":
                    final_model_id = opt_result.optimized_model_id
                    optimization_applied = True
                    logger.info("✅ Optimización aplicada exitosamente")
                else:
                    logger.warning("⚠️ Optimización fallida, usando modelo original")

            # Crear resultado final
            result = AutoMLResult(
                task_id=task_id,
                best_model_id=best_candidate.model_id,
                models_evaluated=len(evaluation_results),
                best_score=best_metrics[config.target_metric],
                training_time=time.time() - start_time,
                models_generated=candidates,
                ensemble_models=ensemble_models,
                optimization_applied=optimization_applied,
                final_model_id=final_model_id,
                status="completed"
            )

            self.completed_runs.append(result)

            logger.info("✅ AutoML completado exitosamente"
            return result

        except Exception as e:
            logger.error(f"❌ Error en AutoML: {e}")
            return AutoMLResult(
                task_id=task_id,
                best_model_id="",
                models_evaluated=0,
                best_score=0.0,
                training_time=time.time() - start_time,
                models_generated=[],
                ensemble_models=[],
                optimization_applied=False,
                final_model_id="",
                status="failed"
            )

    def get_automl_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas de AutoML"""

        if not self.completed_runs:
            return {"total_runs": 0}

        completed = [r for r in self.completed_runs if r.status == "completed"]

        return {
            "total_runs": len(self.completed_runs),
            "completed_runs": len(completed),
            "avg_score": np.mean([r.best_score for r in completed]) if completed else 0,
            "avg_training_time": np.mean([r.training_time for r in completed]) if completed else 0,
            "total_models_generated": sum(len(r.models_generated) for r in completed),
            "optimization_applied": sum(1 for r in completed if r.optimization_applied)
        }

# ===== DEMO Y EJEMPLOS =====

async def demo_automl():
    """Demostración completa de AutoML"""

    print("🤖 AEGIS AutoML Demo")
    print("=" * 40)

    automl = AEGISAutoML()

    # Crear datos de ejemplo
    print("\\n📊 Preparando datos de ejemplo...")

    # Dataset de clasificación simple
    np.random.seed(42)
    n_samples = 1000
    n_features = 20

    X = np.random.randn(n_samples, n_features)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)  # Clasificación binaria simple

    # Convertir a DataFrame para análisis
    import pandas as pd
    feature_names = [f"feature_{i}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    df["target"] = y

    print(f"✅ Dataset creado: {n_samples} muestras, {n_features} features")

    # Configuración de AutoML
    config = AutoMLConfig(
        task_type=TaskType.CLASSIFICATION,
        data_type=DataType.TABULAR,
        target_metric="accuracy",
        max_time_minutes=5,  # Demo rápida
        max_models=6,
        ensemble_size=3,
        cv_folds=3,
        optimization_budget=0.2
    )

    print("\\n⚙️ Configuración AutoML:")
    print(f"   • Tipo de tarea: {config.task_type.value}")
    print(f"   • Métrica objetivo: {config.target_metric}")
    print(f"   • Máximo modelos: {config.max_models}")
    print(f"   • Tiempo límite: {config.max_time_minutes} minutos")

    # Ejecutar AutoML
    print("\\n🚀 Ejecutando AutoML...")
    start_time = time.time()

    result = await automl.run_automl(df, config)

    total_time = time.time() - start_time

    if result.status == "completed":
        print("\\n🎉 AutoML completado exitosamente!")
        print(".1f"        print(f"   📊 Modelos evaluados: {result.models_evaluated}")
        print(".3f"        print(f"   🏆 Mejor modelo: {result.best_model_id}")

        if result.ensemble_models:
            print(f"   🤝 Ensemble creado: {len(result.ensemble_models)} modelos")

        if result.optimization_applied:
            print(f"   🔧 Modelo optimizado: {result.final_model_id}")
        else:
            print("   📝 Modelo sin optimización adicional")

        # Mostrar detalles de modelos generados
        print("\\n🏗️ Modelos generados:")
        for i, candidate in enumerate(result.models_generated[:5]):  # Primeros 5
            print(f"   {i+1}. {candidate.architecture.value} - "
                  ".3f"                  ".1f")

    else:
        print(f"❌ AutoML falló: {result.status}")

    # Estadísticas finales
    stats = automl.get_automl_stats()
    print("\\n📈 Estadísticas de AutoML:")
    print(f"   • Ejecuciones totales: {stats['total_runs']}")
    print(f"   • Ejecuciones exitosas: {stats['completed_runs']}")
    print(f"   • Score promedio: {stats['avg_score']:.3f}")
    print(".1f"    print(f"   • Modelos generados: {stats['total_models_generated']}")
    print(f"   • Optimizaciones aplicadas: {stats['optimization_applied']}")

    print("\\n" + "=" * 60)
    print("🌟 AEGIS AutoML - ¡Automatización de ML completada!")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(demo_automl())
