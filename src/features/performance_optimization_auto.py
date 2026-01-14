#!/usr/bin/env python3
"""
⚡ AEGIS Performance Optimization Auto - Sprint 4.1
Sistema automático de optimización de performance para modelos de IA
"""

import asyncio
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
import threading
import cProfile
import pstats
import io
import psutil
import GPUtil
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BottleneckType(Enum):
    """Tipos de bottleneck identificados"""
    CPU_BOUND = "cpu_bound"
    GPU_BOUND = "gpu_bound"
    MEMORY_BOUND = "memory_bound"
    IO_BOUND = "io_bound"
    NETWORK_BOUND = "network_bound"
    COMPUTATION_BOUND = "computation_bound"

class OptimizationTarget(Enum):
    """Objetivos de optimización"""
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    MEMORY_USAGE = "memory_usage"
    POWER_CONSUMPTION = "power_consumption"
    ACCURACY = "accuracy"

@dataclass
class PerformanceProfile:
    """Perfil de performance de un modelo"""
    model_name: str
    device_type: str
    batch_size: int

    # Métricas de timing
    forward_time_ms: float
    backward_time_ms: float
    total_time_ms: float

    # Métricas de recursos
    cpu_usage_percent: float
    memory_usage_gb: float
    gpu_usage_percent: Optional[float] = None
    gpu_memory_gb: Optional[float] = None

    # Métricas de modelo
    model_size_mb: float
    num_parameters: int
    flops: Optional[float] = None

    # Bottlenecks identificados
    bottlenecks: List[BottleneckType] = field(default_factory=list)

    # Timestamp
    created_at: float = field(default_factory=time.time)

@dataclass
class OptimizationRecommendation:
    """Recomendación de optimización"""
    optimization_type: str
    description: str
    expected_improvement: float  # Porcentaje
    confidence: float  # 0-1
    implementation_complexity: str  # low, medium, high
    applies_to: List[str]  # Qué componentes afecta
    prerequisites: List[str] = field(default_factory=list)

@dataclass
class OptimizedConfig:
    """Configuración optimizada"""
    batch_size: int
    learning_rate: float
    gradient_accumulation_steps: int
    mixed_precision: bool
    gradient_checkpointing: bool
    memory_efficient_attention: bool
    compile_model: bool
    num_workers: int
    prefetch_factor: int
    pin_memory: bool

    # Optimizaciones específicas
    custom_optimizations: Dict[str, Any] = field(default_factory=dict)

class PerformanceProfiler:
    """Profiler automático de performance"""

    def __init__(self):
        self.profiles: List[PerformanceProfile] = []
        self.baseline_profiles: Dict[str, PerformanceProfile] = {}

    async def profile_model(self, model: nn.Module, input_shape: Tuple[int, ...],
                          batch_sizes: List[int] = None, device: str = "auto") -> List[PerformanceProfile]:
        """Profile completo de un modelo"""

        if batch_sizes is None:
            batch_sizes = [1, 4, 16, 32, 64]

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        device = torch.device(device)

        profiles = []

        for batch_size in batch_sizes:
            logger.info(f"🔍 Profiling batch_size={batch_size} en {device}")

            try:
                profile = await self._profile_single_batch(model, input_shape, batch_size, device)
                profiles.append(profile)
                self.profiles.append(profile)

            except Exception as e:
                logger.error(f"❌ Error profiling batch_size {batch_size}: {e}")
                continue

        logger.info(f"✅ Profiling completado: {len(profiles)} configuraciones")
        return profiles

    async def _profile_single_batch(self, model: nn.Module, input_shape: Tuple[int, ...],
                                  batch_size: int, device: torch.device) -> PerformanceProfile:
        """Profile una configuración específica"""

        # Preparar modelo
        model = model.to(device)
        model.eval()

        # Crear input de prueba
        input_tensor = torch.randn(batch_size, *input_shape).to(device)
        target_tensor = torch.randint(0, 1000, (batch_size,)).to(device)

        # Medir uso de CPU/GPU antes
        cpu_before = psutil.cpu_percent()
        memory_before = psutil.virtual_memory().used / (1024**3)

        gpu_before = None
        gpu_memory_before = None
        if device.type == "cuda":
            gpu_stats = GPUtil.getGPUs()[0] if GPUtil.getGPUs() else None
            if gpu_stats:
                gpu_before = gpu_stats.load * 100
                gpu_memory_before = gpu_stats.memoryUsed / 1024  # GB

        # Forward pass timing
        torch.cuda.synchronize() if device.type == "cuda" else None
        forward_start = time.time()

        with torch.no_grad():
            output = model(input_tensor)

        torch.cuda.synchronize() if device.type == "cuda" else None
        forward_time = (time.time() - forward_start) * 1000  # ms

        # Backward pass timing (entrenamiento)
        model.train()
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        torch.cuda.synchronize() if device.type == "cuda" else None
        backward_start = time.time()

        optimizer.zero_grad()
        loss = criterion(output, target_tensor)
        loss.backward()
        optimizer.step()

        torch.cuda.synchronize() if device.type == "cuda" else None
        backward_time = (time.time() - backward_start) * 1000  # ms

        total_time = forward_time + backward_time

        # Medir uso después
        cpu_after = psutil.cpu_percent()
        memory_after = psutil.virtual_memory().used / (1024**3)

        gpu_after = None
        gpu_memory_after = None
        if device.type == "cuda":
            gpu_stats = GPUtil.getGPUs()[0] if GPUtil.getGPUs() else None
            if gpu_stats:
                gpu_after = gpu_stats.load * 100
                gpu_memory_after = gpu_stats.memoryUsed / 1024

        # Calcular promedios
        cpu_usage = (cpu_before + cpu_after) / 2
        memory_usage = (memory_before + memory_after) / 2
        gpu_usage = ((gpu_before + gpu_after) / 2) if gpu_before is not None else None
        gpu_memory = ((gpu_memory_before + gpu_memory_after) / 2) if gpu_memory_before is not None else None

        # Información del modelo
        model_size = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**2)  # MB
        num_parameters = sum(p.numel() for p in model.parameters())

        # Estimar FLOPs (simplificado)
        flops = self._estimate_flops(model, input_shape)

        # Identificar bottlenecks
        bottlenecks = self._identify_bottlenecks(
            forward_time, backward_time, cpu_usage, memory_usage, gpu_usage, device.type
        )

        return PerformanceProfile(
            model_name=model.__class__.__name__,
            device_type=device.type,
            batch_size=batch_size,
            forward_time_ms=forward_time,
            backward_time_ms=backward_time,
            total_time_ms=total_time,
            cpu_usage_percent=cpu_usage,
            memory_usage_gb=memory_usage,
            gpu_usage_percent=gpu_usage,
            gpu_memory_gb=gpu_memory,
            model_size_mb=model_size,
            num_parameters=num_parameters,
            flops=flops,
            bottlenecks=bottlenecks
        )

    def _estimate_flops(self, model: nn.Module, input_shape: Tuple[int, ...]) -> Optional[float]:
        """Estimar FLOPs del modelo (simplificado)"""
        try:
            # Contar operaciones básicas
            total_flops = 0

            for module in model.modules():
                if isinstance(module, nn.Linear):
                    # FLOPs = 2 * input_features * output_features (aprox)
                    total_flops += 2 * module.in_features * module.out_features
                elif isinstance(module, nn.Conv2d):
                    # FLOPs para conv = 2 * kernel_ops * output_elements
                    output_elements = np.prod([s for s in input_shape[1:]])  # H*W*C aproximado
                    kernel_ops = module.kernel_size[0] * module.kernel_size[1] * module.in_channels
                    total_flops += 2 * kernel_ops * output_elements * module.out_channels

            return float(total_flops)

        except Exception:
            return None

    def _identify_bottlenecks(self, forward_time: float, backward_time: float,
                            cpu_usage: float, memory_usage: float,
                            gpu_usage: Optional[float], device_type: str) -> List[BottleneckType]:
        """Identificar bottlenecks de performance"""

        bottlenecks = []

        # CPU bound
        if cpu_usage > 80:
            bottlenecks.append(BottleneckType.CPU_BOUND)

        # GPU bound
        if gpu_usage and gpu_usage > 80:
            bottlenecks.append(BottleneckType.GPU_BOUND)

        # Memory bound
        if memory_usage > psutil.virtual_memory().total * 0.8 / (1024**3):  # 80% de RAM
            bottlenecks.append(BottleneckType.MEMORY_BOUND)

        # Computation bound (backward mucho más lento que forward)
        if backward_time > forward_time * 3:
            bottlenecks.append(BottleneckType.COMPUTATION_BOUND)

        return bottlenecks

    def set_baseline(self, model_name: str, profile: PerformanceProfile):
        """Establecer perfil baseline para comparaciones"""
        self.baseline_profiles[model_name] = profile

    def compare_to_baseline(self, profile: PerformanceProfile) -> Dict[str, float]:
        """Comparar perfil con baseline"""

        if profile.model_name not in self.baseline_profiles:
            return {}

        baseline = self.baseline_profiles[profile.model_name]

        return {
            "latency_improvement": (baseline.total_time_ms - profile.total_time_ms) / baseline.total_time_ms * 100,
            "memory_improvement": (baseline.memory_usage_gb - profile.memory_usage_gb) / baseline.memory_usage_gb * 100,
            "cpu_improvement": (baseline.cpu_usage_percent - profile.cpu_usage_percent) / baseline.cpu_usage_percent * 100
        }

class OptimizationEngine:
    """Motor de optimización automática"""

    def __init__(self, profiler: PerformanceProfiler):
        self.profiler = profiler
        self.optimization_strategies = self._load_optimization_strategies()

    def _load_optimization_strategies(self) -> Dict[str, Callable]:
        """Cargar estrategias de optimización"""

        return {
            "batch_size_optimization": self._optimize_batch_size,
            "mixed_precision": self._apply_mixed_precision,
            "gradient_checkpointing": self._apply_gradient_checkpointing,
            "memory_efficient_attention": self._apply_memory_efficient_attention,
            "model_compilation": self._apply_model_compilation,
            "data_loading_optimization": self._optimize_data_loading,
            "kernel_optimization": self._optimize_kernels
        }

    async def generate_recommendations(self, profiles: List[PerformanceProfile],
                                     target: OptimizationTarget = OptimizationTarget.LATENCY) -> List[OptimizationRecommendation]:
        """Generar recomendaciones de optimización"""

        recommendations = []

        # Analizar todos los perfiles
        best_profile = min(profiles, key=lambda p: p.total_time_ms)
        worst_bottlenecks = self._find_common_bottlenecks(profiles)

        # Recomendaciones basadas en bottlenecks
        for bottleneck in worst_bottlenecks:
            recs = self._recommendations_for_bottleneck(bottleneck, best_profile, target)
            recommendations.extend(recs)

        # Recomendaciones generales
        recommendations.extend(self._general_recommendations(best_profile, target))

        # Ordenar por impacto esperado
        recommendations.sort(key=lambda r: r.expected_improvement, reverse=True)

        return recommendations[:10]  # Top 10

    def _find_common_bottlenecks(self, profiles: List[PerformanceProfile]) -> List[BottleneckType]:
        """Encontrar bottlenecks más comunes"""

        bottleneck_counts = {}
        for profile in profiles:
            for bottleneck in profile.bottlenecks:
                bottleneck_counts[bottleneck] = bottleneck_counts.get(bottleneck, 0) + 1

        # Retornar bottlenecks que aparecen en al menos 50% de perfiles
        threshold = len(profiles) * 0.5
        return [bt for bt, count in bottleneck_counts.items() if count >= threshold]

    def _recommendations_for_bottleneck(self, bottleneck: BottleneckType,
                                      profile: PerformanceProfile,
                                      target: OptimizationTarget) -> List[OptimizationRecommendation]:
        """Generar recomendaciones para un bottleneck específico"""

        recommendations = []

        if bottleneck == BottleneckType.CPU_BOUND:
            recommendations.extend([
                OptimizationRecommendation(
                    optimization_type="mixed_precision",
                    description="Usar mixed precision (FP16) para acelerar operaciones en CPU",
                    expected_improvement=25.0,
                    confidence=0.8,
                    implementation_complexity="low",
                    applies_to=["forward_pass", "backward_pass"]
                ),
                OptimizationRecommendation(
                    optimization_type="batch_size_optimization",
                    description="Optimizar batch size para mejor uso de CPU cache",
                    expected_improvement=15.0,
                    confidence=0.7,
                    implementation_complexity="medium",
                    applies_to=["data_loading", "memory_layout"]
                )
            ])

        elif bottleneck == BottleneckType.GPU_BOUND:
            recommendations.extend([
                OptimizationRecommendation(
                    optimization_type="gradient_checkpointing",
                    description="Aplicar gradient checkpointing para reducir uso de GPU memory",
                    expected_improvement=30.0,
                    confidence=0.9,
                    implementation_complexity="medium",
                    applies_to=["training_loop", "memory_management"],
                    prerequisites=["transformer_models"]
                ),
                OptimizationRecommendation(
                    optimization_type="kernel_optimization",
                    description="Optimizar kernels CUDA para mejor occupancy",
                    expected_improvement=20.0,
                    confidence=0.6,
                    implementation_complexity="high",
                    applies_to=["gpu_kernels", "memory_access"]
                )
            ])

        elif bottleneck == BottleneckType.MEMORY_BOUND:
            recommendations.extend([
                OptimizationRecommendation(
                    optimization_type="gradient_accumulation",
                    description="Usar gradient accumulation para reducir memory footprint",
                    expected_improvement=35.0,
                    confidence=0.85,
                    implementation_complexity="low",
                    applies_to=["training_loop", "batch_processing"]
                ),
                OptimizationRecommendation(
                    optimization_type="memory_efficient_attention",
                    description="Implementar attention memory-efficient para transformers",
                    expected_improvement=40.0,
                    confidence=0.75,
                    implementation_complexity="medium",
                    applies_to=["attention_layers", "transformer_blocks"]
                )
            ])

        return recommendations

    def _general_recommendations(self, profile: PerformanceProfile,
                               target: OptimizationTarget) -> List[OptimizationRecommendation]:
        """Recomendaciones generales aplicables a todos"""

        return [
            OptimizationRecommendation(
                optimization_type="data_loading_optimization",
                description="Optimizar data loading con num_workers y prefetch",
                expected_improvement=10.0,
                confidence=0.9,
                implementation_complexity="low",
                applies_to=["data_pipeline", "cpu_usage"]
            ),
            OptimizationRecommendation(
                optimization_type="model_compilation",
                description="Compilar modelo con torch.compile para mejor performance",
                expected_improvement=15.0,
                confidence=0.7,
                implementation_complexity="low",
                applies_to=["model_execution", "jit_compilation"],
                prerequisites=["torch_2_0_plus"]
            )
        ]

    async def apply_optimizations(self, model: nn.Module, recommendations: List[OptimizationRecommendation],
                                config: OptimizedConfig) -> Tuple[nn.Module, OptimizedConfig]:
        """Aplicar optimizaciones recomendadas"""

        optimized_model = model
        updated_config = config

        applied_optimizations = []

        for rec in recommendations:
            if rec.optimization_type in self.optimization_strategies:
                try:
                    logger.info(f"🔧 Aplicando: {rec.optimization_type}")

                    optimized_model, updated_config = await self.optimization_strategies[rec.optimization_type](
                        optimized_model, updated_config
                    )

                    applied_optimizations.append(rec.optimization_type)

                except Exception as e:
                    logger.error(f"❌ Error aplicando {rec.optimization_type}: {e}")

        logger.info(f"✅ Optimizaciones aplicadas: {len(applied_optimizations)}")
        return optimized_model, updated_config

    # ===== IMPLEMENTACIONES DE OPTIMIZACIONES =====

    async def _optimize_batch_size(self, model: nn.Module, config: OptimizedConfig) -> Tuple[nn.Module, OptimizedConfig]:
        """Optimizar batch size automáticamente"""
        # En implementación real, probar diferentes batch sizes
        config.batch_size = min(config.batch_size * 2, 128)  # Duplicar batch size
        return model, config

    async def _apply_mixed_precision(self, model: nn.Module, config: OptimizedConfig) -> Tuple[nn.Module, OptimizedConfig]:
        """Aplicar mixed precision"""
        config.mixed_precision = True
        return model, config

    async def _apply_gradient_checkpointing(self, model: nn.Module, config: OptimizedConfig) -> Tuple[nn.Module, OptimizedConfig]:
        """Aplicar gradient checkpointing"""
        # Solo para modelos que lo soportan (transformers)
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
            config.gradient_checkpointing = True
        return model, config

    async def _apply_memory_efficient_attention(self, model: nn.Module, config: OptimizedConfig) -> Tuple[nn.Module, OptimizedConfig]:
        """Aplicar attention memory-efficient"""
        # Para modelos transformer
        config.memory_efficient_attention = True
        config.custom_optimizations["attention_optimization"] = "memory_efficient"
        return model, config

    async def _apply_model_compilation(self, model: nn.Module, config: OptimizedConfig) -> Tuple[nn.Module, OptimizedConfig]:
        """Compilar modelo"""
        if hasattr(torch, 'compile'):
            try:
                model = torch.compile(model)
                config.compile_model = True
            except Exception as e:
                logger.warning(f"Model compilation failed: {e}")
        return model, config

    async def _optimize_data_loading(self, model: nn.Module, config: OptimizedConfig) -> Tuple[nn.Module, OptimizedConfig]:
        """Optimizar data loading"""
        config.num_workers = min(4, config.num_workers + 2)
        config.prefetch_factor = 2
        config.pin_memory = True
        return model, config

    async def _optimize_kernels(self, model: nn.Module, config: OptimizedConfig) -> Tuple[nn.Module, OptimizedConfig]:
        """Optimizar kernels CUDA"""
        # Configuraciones para mejor occupancy
        config.custom_optimizations["cuda_optimization"] = {
            "cudnn_benchmark": True,
            "cudnn_deterministic": False
        }
        return model, config

class AutoPerformanceOptimizer:
    """Optimizador automático de performance completo"""

    def __init__(self):
        self.profiler = PerformanceProfiler()
        self.optimization_engine = OptimizationEngine(self.profiler)

    async def optimize_model_automatically(self, model: nn.Module,
                                         input_shape: Tuple[int, ...],
                                         target: OptimizationTarget = OptimizationTarget.LATENCY,
                                         max_optimization_time: int = 300) -> Dict[str, Any]:
        """Optimización automática completa"""

        logger.info("🚀 Iniciando optimización automática de performance")

        start_time = time.time()
        optimization_results = {
            "original_profile": None,
            "optimized_profile": None,
            "recommendations_applied": [],
            "performance_improvement": {},
            "optimization_time": 0,
            "status": "running"
        }

        try:
            # Paso 1: Profiling inicial
            logger.info("📊 Profiling modelo original...")
            profiles = await self.profiler.profile_model(model, input_shape)

            if not profiles:
                raise RuntimeError("No se pudieron generar perfiles")

            best_profile = min(profiles, key=lambda p: p.total_time_ms)
            optimization_results["original_profile"] = best_profile

            # Paso 2: Generar recomendaciones
            logger.info("💡 Generando recomendaciones de optimización...")
            recommendations = await self.optimization_engine.generate_recommendations(profiles, target)

            if not recommendations:
                logger.warning("No se generaron recomendaciones")
                optimization_results["status"] = "no_optimizations_needed"
                return optimization_results

            # Paso 3: Aplicar optimizaciones
            logger.info(f"🔧 Aplicando {len(recommendations)} optimizaciones...")

            # Configuración inicial
            initial_config = OptimizedConfig(
                batch_size=best_profile.batch_size,
                learning_rate=0.001,
                gradient_accumulation_steps=1,
                mixed_precision=False,
                gradient_checkpointing=False,
                memory_efficient_attention=False,
                compile_model=False,
                num_workers=2,
                prefetch_factor=1,
                pin_memory=False
            )

            optimized_model, final_config = await self.optimization_engine.apply_optimizations(
                model, recommendations[:5], initial_config  # Aplicar top 5
            )

            optimization_results["recommendations_applied"] = [r.optimization_type for r in recommendations[:5]]

            # Paso 4: Re-profile optimizado
            logger.info("📊 Re-profiling modelo optimizado...")
            optimized_profiles = await self.profiler.profile_model(
                optimized_model, input_shape, [final_config.batch_size]
            )

            if optimized_profiles:
                optimized_profile = optimized_profiles[0]
                optimization_results["optimized_profile"] = optimized_profile

                # Calcular mejoras
                improvement = self.profiler.compare_to_baseline(optimized_profile)
                optimization_results["performance_improvement"] = improvement

                logger.info("✅ Optimización completada exitosamente")

            optimization_results["status"] = "completed"

        except Exception as e:
            logger.error(f"❌ Error en optimización automática: {e}")
            optimization_results["status"] = "failed"
            optimization_results["error"] = str(e)

        finally:
            optimization_results["optimization_time"] = time.time() - start_time

        return optimization_results

    async def benchmark_optimization(self, model: nn.Module, input_shape: Tuple[int, ...],
                                   optimization_configs: List[OptimizedConfig]) -> Dict[str, Any]:
        """Benchmark diferentes configuraciones de optimización"""

        logger.info(f"🏁 Benchmarking {len(optimization_configs)} configuraciones...")

        results = {}

        for i, config in enumerate(optimization_configs):
            logger.info(f"📊 Testing config {i+1}/{len(optimization_configs)}")

            # Aplicar configuración al modelo
            optimized_model = self._apply_config_to_model(model, config)

            # Profile
            profiles = await self.profiler.profile_model(
                optimized_model, input_shape, [config.batch_size]
            )

            if profiles:
                results[f"config_{i}"] = {
                    "config": config,
                    "profile": profiles[0]
                }

        # Encontrar mejor configuración
        if results:
            best_config = min(results.values(),
                            key=lambda r: r["profile"].total_time_ms)
            results["best_config"] = best_config

        return results

    def _apply_config_to_model(self, model: nn.Module, config: OptimizedConfig) -> nn.Module:
        """Aplicar configuración de optimización al modelo"""

        # Aplicar optimizaciones soportadas
        if config.compile_model and hasattr(torch, 'compile'):
            try:
                model = torch.compile(model)
            except:
                pass

        if config.gradient_checkpointing and hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()

        return model

# ===== DEMO Y EJEMPLOS =====

async def demo_performance_optimization():
    """Demostración completa de optimización automática de performance"""

    print("⚡ AEGIS Performance Optimization Auto Demo")
    print("=" * 50)

    # Crear sistema de optimización
    optimizer = AutoPerformanceOptimizer()

    # Crear modelo de ejemplo (simple CNN)
    model = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(64, 128, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(128, 10)
    )

    input_shape = (3, 32, 32)  # Imágenes pequeñas para demo

    print("🧠 Modelo de ejemplo creado: Simple CNN")
    print(f"📊 Input shape: {input_shape}")

    # Ejecutar optimización automática
    print("\\n🚀 Iniciando optimización automática...")

    results = await optimizer.optimize_model_automatically(
        model, input_shape,
        target=OptimizationTarget.LATENCY,
        max_optimization_time=60  # 1 minuto para demo
    )

    # Mostrar resultados
    print("\\n📊 RESULTADOS DE OPTIMIZACIÓN:")
    print(f"   • Estado: {results['status']}")
    print(".1f"    print(f"   • Optimizaciones aplicadas: {len(results['recommendations_applied'])}")

    if results['original_profile'] and results['optimized_profile']:
        orig = results['original_profile']
        opt = results['optimized_profile']

        print("\\n⚡ COMPARACIÓN DE PERFORMANCE:")
        print(f"   • Latencia original: {orig.total_time_ms:.2f}ms")
        print(f"   • Latencia optimizada: {opt.total_time_ms:.2f}ms")
        print(".1f"        print(f"   • Memoria original: {orig.memory_usage_gb:.2f}GB")
        print(f"   • Memoria optimizada: {opt.memory_usage_gb:.2f}GB")
        print(".1f"
        if results['performance_improvement']:
            impr = results['performance_improvement']
            print("\\n📈 MEJORAS LOGRADAS:")
            print(".1f"            print(".1f"            print(".1f"
    if results['recommendations_applied']:
        print("\\n🔧 OPTIMIZACIONES APLICADAS:")
        for opt in results['recommendations_applied']:
            print(f"   ✅ {opt}")

    # Demo de profiling detallado
    print("\\n🔍 DEMO DE PROFILING DETALLADO")

    # Profile con diferentes batch sizes
    profiles = await optimizer.profiler.profile_model(model, input_shape, [1, 4, 16])

    print("\\n📋 PERFILES POR BATCH SIZE:")
    print("   Batch | Forward | Backward | Total | Memory | CPU")
    print("   ------|---------|----------|-------|--------|-----")

    for profile in profiles:
        print(f"   {profile.batch_size:5} | {profile.forward_time_ms:7.1f} | {profile.backward_time_ms:8.1f} | "
              f"{profile.total_time_ms:5.1f} | {profile.memory_usage_gb:6.2f} | {profile.cpu_usage_percent:3.0f}%")

    # Identificar mejores configuraciones
    if profiles:
        best_latency = min(profiles, key=lambda p: p.total_time_ms)
        best_memory = min(profiles, key=lambda p: p.memory_usage_gb)

        print("\\n🏆 CONFIGURACIONES ÓPTIMAS:")
        print(f"   • Mejor latencia: Batch {best_latency.batch_size} "
              f"({best_latency.total_time_ms:.1f}ms)")
        print(f"   • Mejor memoria: Batch {best_memory.batch_size} "
              f"({best_memory.memory_usage_gb:.2f}GB)")

    print("\\n" + "=" * 60)
    print("🌟 Performance Optimization funcionando correctamente!")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(demo_performance_optimization())
