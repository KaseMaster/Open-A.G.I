#!/usr/bin/env python3
"""
Optimizador de Rendimiento - AEGIS Framework
Sistema inteligente para análisis, monitoreo y optimización automática
del rendimiento en el framework distribuido.

Características principales:
- Análisis de rendimiento en tiempo real
- Optimización automática de recursos
- Predicción de cuellos de botella
- Balanceador de carga inteligente
- Optimización de algoritmos ML
- Gestión dinámica de memoria
- Análisis de patrones de tráfico
- Recomendaciones de escalado
"""

import asyncio
import time
import json
import logging
import statistics
import numpy as np
import pandas as pd
from typing import Dict, List, Set, Optional, Any, Tuple, Callable
from dataclasses import dataclass, asdict, field
from enum import Enum
import threading
import queue
import psutil
import GPUtil
import networkx as nx
from collections import defaultdict, deque
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import joblib
import warnings
warnings.filterwarnings('ignore')

# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MetricType(Enum):
    """Tipos de métricas"""
    CPU_USAGE = "cpu_usage"
    MEMORY_USAGE = "memory_usage"
    DISK_IO = "disk_io"
    NETWORK_IO = "network_io"
    GPU_USAGE = "gpu_usage"
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"
    QUEUE_SIZE = "queue_size"
    CONNECTION_COUNT = "connection_count"

class OptimizationType(Enum):
    """Tipos de optimización"""
    RESOURCE_ALLOCATION = "resource_allocation"
    LOAD_BALANCING = "load_balancing"
    CACHING = "caching"
    ALGORITHM_TUNING = "algorithm_tuning"
    SCALING = "scaling"
    MEMORY_MANAGEMENT = "memory_management"
    NETWORK_OPTIMIZATION = "network_optimization"

class PerformanceLevel(Enum):
    """Niveles de rendimiento"""
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    CRITICAL = "critical"

class OptimizationPriority(Enum):
    """Prioridades de optimización"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

@dataclass
class PerformanceMetric:
    """Métrica de rendimiento"""
    metric_type: MetricType
    value: float
    timestamp: float
    node_id: str
    service_name: str = ""
    unit: str = ""
    threshold_warning: float = 0.0
    threshold_critical: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PerformanceAnomaly:
    """Anomalía de rendimiento"""
    anomaly_id: str
    metric_type: MetricType
    severity: str
    description: str
    detected_at: float
    node_id: str
    service_name: str
    current_value: float
    expected_range: Tuple[float, float]
    confidence: float
    suggested_actions: List[str] = field(default_factory=list)

@dataclass
class OptimizationRecommendation:
    """Recomendación de optimización"""
    recommendation_id: str
    optimization_type: OptimizationType
    priority: OptimizationPriority
    description: str
    expected_improvement: float
    implementation_cost: float
    risk_level: str
    target_nodes: List[str]
    target_services: List[str]
    parameters: Dict[str, Any] = field(default_factory=dict)
    estimated_duration: int = 0  # minutos

@dataclass
class ResourceProfile:
    """Perfil de recursos de nodo"""
    node_id: str
    cpu_cores: int
    cpu_usage: float
    memory_total: float
    memory_used: float
    disk_total: float
    disk_used: float
    network_bandwidth: float
    network_usage: float
    gpu_count: int = 0
    gpu_usage: float = 0.0
    load_average: float = 0.0
    active_connections: int = 0

@dataclass
class PerformanceBaseline:
    """Línea base de rendimiento"""
    service_name: str
    node_id: str
    baseline_metrics: Dict[MetricType, Dict[str, float]]  # mean, std, min, max
    created_at: float
    sample_count: int
    confidence_interval: float = 0.95

class MetricsCollector:
    """Recolector de métricas de rendimiento"""
    
    def __init__(self, collection_interval: int = 10):
        self.collection_interval = collection_interval
        self.metrics_queue = queue.Queue()
        self.is_collecting = False
        self.collection_thread = None
        
        # Historial de métricas
        self.metrics_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        logger.info("📊 Recolector de métricas inicializado")
    
    def start_collection(self):
        """Inicia recolección de métricas"""
        if not self.is_collecting:
            self.is_collecting = True
            self.collection_thread = threading.Thread(target=self._collect_metrics_loop)
            self.collection_thread.daemon = True
            self.collection_thread.start()
            logger.info("🚀 Recolección de métricas iniciada")
    
    def stop_collection(self):
        """Detiene recolección de métricas"""
        self.is_collecting = False
        if self.collection_thread:
            self.collection_thread.join(timeout=5)
        logger.info("🛑 Recolección de métricas detenida")
    
    def _collect_metrics_loop(self):
        """Loop principal de recolección"""
        while self.is_collecting:
            try:
                # Recolectar métricas del sistema
                system_metrics = self._collect_system_metrics()
                
                # Recolectar métricas de red
                network_metrics = self._collect_network_metrics()
                
                # Recolectar métricas de GPU si están disponibles
                gpu_metrics = self._collect_gpu_metrics()
                
                # Combinar todas las métricas
                all_metrics = system_metrics + network_metrics + gpu_metrics
                
                # Almacenar métricas
                for metric in all_metrics:
                    self.metrics_queue.put(metric)
                    self._store_metric_in_history(metric)
                
                time.sleep(self.collection_interval)
                
            except Exception as e:
                logger.error(f"❌ Error recolectando métricas: {e}")
                time.sleep(self.collection_interval)
    
    def _collect_system_metrics(self) -> List[PerformanceMetric]:
        """Recolecta métricas del sistema"""
        metrics = []
        timestamp = time.time()
        node_id = "local"  # En un sistema distribuido, esto sería dinámico
        
        try:
            # CPU
            cpu_percent = psutil.cpu_percent(interval=1)
            metrics.append(PerformanceMetric(
                metric_type=MetricType.CPU_USAGE,
                value=cpu_percent,
                timestamp=timestamp,
                node_id=node_id,
                unit="%",
                threshold_warning=70.0,
                threshold_critical=90.0
            ))
            
            # Memoria
            memory = psutil.virtual_memory()
            metrics.append(PerformanceMetric(
                metric_type=MetricType.MEMORY_USAGE,
                value=memory.percent,
                timestamp=timestamp,
                node_id=node_id,
                unit="%",
                threshold_warning=80.0,
                threshold_critical=95.0,
                metadata={"total": memory.total, "available": memory.available}
            ))
            
            # Disco I/O
            disk_io = psutil.disk_io_counters()
            if disk_io:
                metrics.append(PerformanceMetric(
                    metric_type=MetricType.DISK_IO,
                    value=disk_io.read_bytes + disk_io.write_bytes,
                    timestamp=timestamp,
                    node_id=node_id,
                    unit="bytes",
                    metadata={
                        "read_bytes": disk_io.read_bytes,
                        "write_bytes": disk_io.write_bytes,
                        "read_count": disk_io.read_count,
                        "write_count": disk_io.write_count
                    }
                ))
            
            # Red I/O
            net_io = psutil.net_io_counters()
            if net_io:
                metrics.append(PerformanceMetric(
                    metric_type=MetricType.NETWORK_IO,
                    value=net_io.bytes_sent + net_io.bytes_recv,
                    timestamp=timestamp,
                    node_id=node_id,
                    unit="bytes",
                    metadata={
                        "bytes_sent": net_io.bytes_sent,
                        "bytes_recv": net_io.bytes_recv,
                        "packets_sent": net_io.packets_sent,
                        "packets_recv": net_io.packets_recv
                    }
                ))
            
        except Exception as e:
            logger.error(f"❌ Error recolectando métricas del sistema: {e}")
        
        return metrics
    
    def _collect_network_metrics(self) -> List[PerformanceMetric]:
        """Recolecta métricas de red"""
        metrics = []
        timestamp = time.time()
        node_id = "local"
        
        try:
            # Conexiones activas
            connections = psutil.net_connections()
            active_connections = len([conn for conn in connections if conn.status == 'ESTABLISHED'])
            
            metrics.append(PerformanceMetric(
                metric_type=MetricType.CONNECTION_COUNT,
                value=active_connections,
                timestamp=timestamp,
                node_id=node_id,
                unit="connections",
                threshold_warning=1000,
                threshold_critical=5000
            ))
            
        except Exception as e:
            logger.error(f"❌ Error recolectando métricas de red: {e}")
        
        return metrics
    
    def _collect_gpu_metrics(self) -> List[PerformanceMetric]:
        """Recolecta métricas de GPU"""
        metrics = []
        timestamp = time.time()
        node_id = "local"
        
        try:
            gpus = GPUtil.getGPUs()
            for i, gpu in enumerate(gpus):
                metrics.append(PerformanceMetric(
                    metric_type=MetricType.GPU_USAGE,
                    value=gpu.load * 100,
                    timestamp=timestamp,
                    node_id=node_id,
                    service_name=f"gpu_{i}",
                    unit="%",
                    threshold_warning=80.0,
                    threshold_critical=95.0,
                    metadata={
                        "memory_used": gpu.memoryUsed,
                        "memory_total": gpu.memoryTotal,
                        "temperature": gpu.temperature
                    }
                ))
                
        except Exception as e:
            logger.debug(f"GPU no disponible o error: {e}")
        
        return metrics
    
    def _store_metric_in_history(self, metric: PerformanceMetric):
        """Almacena métrica en historial"""
        key = f"{metric.node_id}_{metric.service_name}_{metric.metric_type.value}"
        self.metrics_history[key].append(metric)
    
    def get_recent_metrics(self, metric_type: MetricType = None, node_id: str = None, 
                          service_name: str = None, limit: int = 100) -> List[PerformanceMetric]:
        """Obtiene métricas recientes"""
        all_metrics = []
        
        for key, metrics_deque in self.metrics_history.items():
            for metric in list(metrics_deque)[-limit:]:
                # Filtrar por criterios
                if metric_type and metric.metric_type != metric_type:
                    continue
                if node_id and metric.node_id != node_id:
                    continue
                if service_name and metric.service_name != service_name:
                    continue
                
                all_metrics.append(metric)
        
        # Ordenar por timestamp
        all_metrics.sort(key=lambda m: m.timestamp, reverse=True)
        return all_metrics[:limit]
    
    def get_metric_statistics(self, metric_type: MetricType, node_id: str = None, 
                            service_name: str = None, time_window: int = 3600) -> Dict[str, float]:
        """Obtiene estadísticas de métricas"""
        current_time = time.time()
        cutoff_time = current_time - time_window
        
        # Obtener métricas en ventana de tiempo
        metrics = self.get_recent_metrics(metric_type, node_id, service_name, limit=1000)
        filtered_metrics = [m for m in metrics if m.timestamp >= cutoff_time]
        
        if not filtered_metrics:
            return {}
        
        values = [m.value for m in filtered_metrics]
        
        return {
            "count": len(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0,
            "min": min(values),
            "max": max(values),
            "p95": np.percentile(values, 95),
            "p99": np.percentile(values, 99)
        }

class AnomalyDetector:
    """Detector de anomalías de rendimiento"""
    
    def __init__(self, sensitivity: float = 0.1):
        self.sensitivity = sensitivity
        self.models: Dict[str, IsolationForest] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.baselines: Dict[str, PerformanceBaseline] = {}
        
        # Configurar modelos
        self.isolation_forest_params = {
            'contamination': sensitivity,
            'random_state': 42,
            'n_estimators': 100
        }
        
        logger.info("🔍 Detector de anomalías inicializado")
    
    def train_baseline(self, metrics: List[PerformanceMetric], service_name: str, node_id: str):
        """Entrena línea base de rendimiento"""
        try:
            # Agrupar métricas por tipo
            metrics_by_type = defaultdict(list)
            for metric in metrics:
                if metric.service_name == service_name and metric.node_id == node_id:
                    metrics_by_type[metric.metric_type].append(metric.value)
            
            # Calcular estadísticas base
            baseline_metrics = {}
            for metric_type, values in metrics_by_type.items():
                if len(values) >= 10:  # Mínimo de muestras
                    baseline_metrics[metric_type] = {
                        "mean": statistics.mean(values),
                        "std": statistics.stdev(values) if len(values) > 1 else 0,
                        "min": min(values),
                        "max": max(values),
                        "p25": np.percentile(values, 25),
                        "p75": np.percentile(values, 75)
                    }
            
            # Crear línea base
            baseline = PerformanceBaseline(
                service_name=service_name,
                node_id=node_id,
                baseline_metrics=baseline_metrics,
                created_at=time.time(),
                sample_count=len(metrics)
            )
            
            baseline_key = f"{node_id}_{service_name}"
            self.baselines[baseline_key] = baseline
            
            # Entrenar modelo de detección de anomalías
            self._train_anomaly_model(baseline_key, metrics)
            
            logger.info(f"✅ Línea base entrenada: {baseline_key}")
            
        except Exception as e:
            logger.error(f"❌ Error entrenando línea base: {e}")
    
    def _train_anomaly_model(self, baseline_key: str, metrics: List[PerformanceMetric]):
        """Entrena modelo de detección de anomalías"""
        try:
            # Preparar datos de entrenamiento
            features = []
            for metric in metrics:
                feature_vector = [
                    metric.value,
                    metric.timestamp % 86400,  # Hora del día
                    metric.timestamp % 604800,  # Día de la semana
                ]
                features.append(feature_vector)
            
            if len(features) < 10:
                return
            
            # Normalizar datos
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            
            # Entrenar modelo
            model = IsolationForest(**self.isolation_forest_params)
            model.fit(features_scaled)
            
            # Almacenar modelo y scaler
            self.models[baseline_key] = model
            self.scalers[baseline_key] = scaler
            
            logger.info(f"🤖 Modelo de anomalías entrenado: {baseline_key}")
            
        except Exception as e:
            logger.error(f"❌ Error entrenando modelo de anomalías: {e}")
    
    def detect_anomalies(self, metrics: List[PerformanceMetric]) -> List[PerformanceAnomaly]:
        """Detecta anomalías en métricas"""
        anomalies = []
        
        try:
            # Agrupar métricas por servicio y nodo
            grouped_metrics = defaultdict(list)
            for metric in metrics:
                key = f"{metric.node_id}_{metric.service_name}"
                grouped_metrics[key].append(metric)
            
            # Detectar anomalías para cada grupo
            for key, group_metrics in grouped_metrics.items():
                group_anomalies = self._detect_group_anomalies(key, group_metrics)
                anomalies.extend(group_anomalies)
            
        except Exception as e:
            logger.error(f"❌ Error detectando anomalías: {e}")
        
        return anomalies
    
    def _detect_group_anomalies(self, baseline_key: str, metrics: List[PerformanceMetric]) -> List[PerformanceAnomaly]:
        """Detecta anomalías para un grupo de métricas"""
        anomalies = []
        
        try:
            # Verificar si tenemos línea base
            if baseline_key not in self.baselines:
                return anomalies
            
            baseline = self.baselines[baseline_key]
            
            # Detectar anomalías estadísticas
            for metric in metrics:
                if metric.metric_type in baseline.baseline_metrics:
                    baseline_stats = baseline.baseline_metrics[metric.metric_type]
                    anomaly = self._check_statistical_anomaly(metric, baseline_stats)
                    if anomaly:
                        anomalies.append(anomaly)
            
            # Detectar anomalías con ML si tenemos modelo
            if baseline_key in self.models:
                ml_anomalies = self._detect_ml_anomalies(baseline_key, metrics)
                anomalies.extend(ml_anomalies)
            
        except Exception as e:
            logger.error(f"❌ Error detectando anomalías de grupo: {e}")
        
        return anomalies
    
    def _check_statistical_anomaly(self, metric: PerformanceMetric, 
                                 baseline_stats: Dict[str, float]) -> Optional[PerformanceAnomaly]:
        """Verifica anomalía estadística"""
        try:
            mean = baseline_stats["mean"]
            std = baseline_stats["std"]
            
            # Calcular z-score
            if std > 0:
                z_score = abs(metric.value - mean) / std
                
                # Detectar anomalía (más de 3 desviaciones estándar)
                if z_score > 3:
                    severity = "critical" if z_score > 5 else "high"
                    
                    return PerformanceAnomaly(
                        anomaly_id=f"stat_{metric.node_id}_{metric.service_name}_{int(metric.timestamp)}",
                        metric_type=metric.metric_type,
                        severity=severity,
                        description=f"Valor anómalo detectado: {metric.value:.2f} (z-score: {z_score:.2f})",
                        detected_at=time.time(),
                        node_id=metric.node_id,
                        service_name=metric.service_name,
                        current_value=metric.value,
                        expected_range=(mean - 2*std, mean + 2*std),
                        confidence=min(z_score / 5, 1.0),
                        suggested_actions=[
                            "Verificar carga del sistema",
                            "Revisar logs de aplicación",
                            "Considerar escalado de recursos"
                        ]
                    )
            
        except Exception as e:
            logger.error(f"❌ Error verificando anomalía estadística: {e}")
        
        return None
    
    def _detect_ml_anomalies(self, baseline_key: str, metrics: List[PerformanceMetric]) -> List[PerformanceAnomaly]:
        """Detecta anomalías usando ML"""
        anomalies = []
        
        try:
            model = self.models[baseline_key]
            scaler = self.scalers[baseline_key]
            
            # Preparar características
            features = []
            for metric in metrics:
                feature_vector = [
                    metric.value,
                    metric.timestamp % 86400,
                    metric.timestamp % 604800,
                ]
                features.append(feature_vector)
            
            if not features:
                return anomalies
            
            # Normalizar y predecir
            features_scaled = scaler.transform(features)
            predictions = model.predict(features_scaled)
            anomaly_scores = model.decision_function(features_scaled)
            
            # Crear anomalías para predicciones negativas
            for i, (prediction, score, metric) in enumerate(zip(predictions, anomaly_scores, metrics)):
                if prediction == -1:  # Anomalía detectada
                    confidence = abs(score)
                    severity = "critical" if confidence > 0.5 else "medium"
                    
                    anomaly = PerformanceAnomaly(
                        anomaly_id=f"ml_{metric.node_id}_{metric.service_name}_{int(metric.timestamp)}",
                        metric_type=metric.metric_type,
                        severity=severity,
                        description=f"Anomalía ML detectada (score: {score:.3f})",
                        detected_at=time.time(),
                        node_id=metric.node_id,
                        service_name=metric.service_name,
                        current_value=metric.value,
                        expected_range=(0, 0),  # Se calcularía basado en el modelo
                        confidence=confidence,
                        suggested_actions=[
                            "Investigar causa raíz",
                            "Verificar configuración",
                            "Monitorear tendencia"
                        ]
                    )
                    
                    anomalies.append(anomaly)
            
        except Exception as e:
            logger.error(f"❌ Error detectando anomalías ML: {e}")
        
        return anomalies

class PerformanceOptimizer:
    """Optimizador de rendimiento"""
    
    def __init__(self):
        self.optimization_history: List[OptimizationRecommendation] = []
        self.active_optimizations: Dict[str, OptimizationRecommendation] = {}
        
        # Configurar optimizadores específicos
        self.resource_optimizer = ResourceOptimizer()
        self.load_balancer_optimizer = LoadBalancerOptimizer()
        self.cache_optimizer = CacheOptimizer()
        self.algorithm_optimizer = AlgorithmOptimizer()
        
        logger.info("⚡ Optimizador de rendimiento inicializado")
    
    def analyze_performance(self, metrics: List[PerformanceMetric], 
                          anomalies: List[PerformanceAnomaly]) -> List[OptimizationRecommendation]:
        """Analiza rendimiento y genera recomendaciones"""
        recommendations = []
        
        try:
            # Analizar métricas por tipo
            metrics_by_type = defaultdict(list)
            for metric in metrics:
                metrics_by_type[metric.metric_type].append(metric)
            
            # Generar recomendaciones por tipo de optimización
            resource_recs = self.resource_optimizer.analyze(metrics_by_type, anomalies)
            recommendations.extend(resource_recs)
            
            load_balance_recs = self.load_balancer_optimizer.analyze(metrics_by_type, anomalies)
            recommendations.extend(load_balance_recs)
            
            cache_recs = self.cache_optimizer.analyze(metrics_by_type, anomalies)
            recommendations.extend(cache_recs)
            
            algorithm_recs = self.algorithm_optimizer.analyze(metrics_by_type, anomalies)
            recommendations.extend(algorithm_recs)
            
            # Priorizar recomendaciones
            recommendations = self._prioritize_recommendations(recommendations)
            
            # Almacenar en historial
            self.optimization_history.extend(recommendations)
            
            logger.info(f"📋 Generadas {len(recommendations)} recomendaciones de optimización")
            
        except Exception as e:
            logger.error(f"❌ Error analizando rendimiento: {e}")
        
        return recommendations
    
    def _prioritize_recommendations(self, recommendations: List[OptimizationRecommendation]) -> List[OptimizationRecommendation]:
        """Prioriza recomendaciones por impacto y costo"""
        try:
            # Calcular score de prioridad
            for rec in recommendations:
                priority_score = 0
                
                # Factor de prioridad
                priority_weights = {
                    OptimizationPriority.CRITICAL: 100,
                    OptimizationPriority.HIGH: 75,
                    OptimizationPriority.MEDIUM: 50,
                    OptimizationPriority.LOW: 25
                }
                priority_score += priority_weights.get(rec.priority, 0)
                
                # Factor de mejora esperada
                priority_score += rec.expected_improvement * 10
                
                # Factor de costo (inverso)
                priority_score -= rec.implementation_cost * 5
                
                # Factor de riesgo (inverso)
                risk_weights = {"low": 0, "medium": -10, "high": -25, "critical": -50}
                priority_score += risk_weights.get(rec.risk_level, 0)
                
                rec.parameters["priority_score"] = priority_score
            
            # Ordenar por score
            recommendations.sort(key=lambda r: r.parameters.get("priority_score", 0), reverse=True)
            
        except Exception as e:
            logger.error(f"❌ Error priorizando recomendaciones: {e}")
        
        return recommendations
    
    async def implement_recommendation(self, recommendation: OptimizationRecommendation) -> bool:
        """Implementa recomendación de optimización"""
        try:
            logger.info(f"🔧 Implementando optimización: {recommendation.description}")
            
            # Marcar como activa
            self.active_optimizations[recommendation.recommendation_id] = recommendation
            
            # Implementar según tipo
            success = False
            if recommendation.optimization_type == OptimizationType.RESOURCE_ALLOCATION:
                success = await self.resource_optimizer.implement(recommendation)
            elif recommendation.optimization_type == OptimizationType.LOAD_BALANCING:
                success = await self.load_balancer_optimizer.implement(recommendation)
            elif recommendation.optimization_type == OptimizationType.CACHING:
                success = await self.cache_optimizer.implement(recommendation)
            elif recommendation.optimization_type == OptimizationType.ALGORITHM_TUNING:
                success = await self.algorithm_optimizer.implement(recommendation)
            
            # Actualizar estado
            if success:
                logger.info(f"✅ Optimización implementada: {recommendation.recommendation_id}")
            else:
                logger.error(f"❌ Fallo implementando optimización: {recommendation.recommendation_id}")
                del self.active_optimizations[recommendation.recommendation_id]
            
            return success
            
        except Exception as e:
            logger.error(f"❌ Error implementando recomendación: {e}")
            return False
    
    def get_optimization_status(self) -> Dict[str, Any]:
        """Obtiene estado de optimizaciones"""
        return {
            "active_optimizations": len(self.active_optimizations),
            "total_recommendations": len(self.optimization_history),
            "optimization_types": {
                opt_type.value: len([r for r in self.optimization_history 
                                   if r.optimization_type == opt_type])
                for opt_type in OptimizationType
            },
            "priority_distribution": {
                priority.value: len([r for r in self.optimization_history 
                                   if r.priority == priority])
                for priority in OptimizationPriority
            }
        }

class ResourceOptimizer:
    """Optimizador de recursos"""
    
    def analyze(self, metrics_by_type: Dict[MetricType, List[PerformanceMetric]], 
                anomalies: List[PerformanceAnomaly]) -> List[OptimizationRecommendation]:
        """Analiza uso de recursos y genera recomendaciones"""
        recommendations = []
        
        try:
            # Analizar CPU
            if MetricType.CPU_USAGE in metrics_by_type:
                cpu_recs = self._analyze_cpu_usage(metrics_by_type[MetricType.CPU_USAGE])
                recommendations.extend(cpu_recs)
            
            # Analizar memoria
            if MetricType.MEMORY_USAGE in metrics_by_type:
                memory_recs = self._analyze_memory_usage(metrics_by_type[MetricType.MEMORY_USAGE])
                recommendations.extend(memory_recs)
            
            # Analizar disco
            if MetricType.DISK_IO in metrics_by_type:
                disk_recs = self._analyze_disk_usage(metrics_by_type[MetricType.DISK_IO])
                recommendations.extend(disk_recs)
            
        except Exception as e:
            logger.error(f"❌ Error analizando recursos: {e}")
        
        return recommendations
    
    def _analyze_cpu_usage(self, cpu_metrics: List[PerformanceMetric]) -> List[OptimizationRecommendation]:
        """Analiza uso de CPU"""
        recommendations = []
        
        try:
            if not cpu_metrics:
                return recommendations
            
            # Calcular estadísticas
            cpu_values = [m.value for m in cpu_metrics]
            avg_cpu = statistics.mean(cpu_values)
            max_cpu = max(cpu_values)
            
            # CPU alta sostenida
            if avg_cpu > 80:
                recommendations.append(OptimizationRecommendation(
                    recommendation_id=f"cpu_high_{int(time.time())}",
                    optimization_type=OptimizationType.RESOURCE_ALLOCATION,
                    priority=OptimizationPriority.HIGH,
                    description=f"CPU alta sostenida ({avg_cpu:.1f}%). Considerar escalado vertical.",
                    expected_improvement=30.0,
                    implementation_cost=7.0,
                    risk_level="medium",
                    target_nodes=[m.node_id for m in cpu_metrics],
                    target_services=list(set(m.service_name for m in cpu_metrics if m.service_name)),
                    parameters={
                        "current_avg": avg_cpu,
                        "recommended_action": "scale_up_cpu",
                        "suggested_increase": "50%"
                    }
                ))
            
            # CPU baja sostenida (posible sobre-provisioning)
            elif avg_cpu < 20 and max_cpu < 40:
                recommendations.append(OptimizationRecommendation(
                    recommendation_id=f"cpu_low_{int(time.time())}",
                    optimization_type=OptimizationType.RESOURCE_ALLOCATION,
                    priority=OptimizationPriority.LOW,
                    description=f"CPU subutilizada ({avg_cpu:.1f}%). Considerar escalado hacia abajo.",
                    expected_improvement=15.0,
                    implementation_cost=3.0,
                    risk_level="low",
                    target_nodes=[m.node_id for m in cpu_metrics],
                    target_services=list(set(m.service_name for m in cpu_metrics if m.service_name)),
                    parameters={
                        "current_avg": avg_cpu,
                        "recommended_action": "scale_down_cpu",
                        "suggested_decrease": "25%"
                    }
                ))
            
        except Exception as e:
            logger.error(f"❌ Error analizando CPU: {e}")
        
        return recommendations
    
    def _analyze_memory_usage(self, memory_metrics: List[PerformanceMetric]) -> List[OptimizationRecommendation]:
        """Analiza uso de memoria"""
        recommendations = []
        
        try:
            if not memory_metrics:
                return recommendations
            
            memory_values = [m.value for m in memory_metrics]
            avg_memory = statistics.mean(memory_values)
            max_memory = max(memory_values)
            
            # Memoria alta
            if avg_memory > 85:
                recommendations.append(OptimizationRecommendation(
                    recommendation_id=f"memory_high_{int(time.time())}",
                    optimization_type=OptimizationType.RESOURCE_ALLOCATION,
                    priority=OptimizationPriority.CRITICAL if avg_memory > 95 else OptimizationPriority.HIGH,
                    description=f"Memoria alta ({avg_memory:.1f}%). Riesgo de OOM.",
                    expected_improvement=40.0,
                    implementation_cost=8.0,
                    risk_level="high" if avg_memory > 95 else "medium",
                    target_nodes=[m.node_id for m in memory_metrics],
                    target_services=list(set(m.service_name for m in memory_metrics if m.service_name)),
                    parameters={
                        "current_avg": avg_memory,
                        "recommended_action": "increase_memory",
                        "suggested_increase": "100%" if avg_memory > 95 else "50%"
                    }
                ))
            
        except Exception as e:
            logger.error(f"❌ Error analizando memoria: {e}")
        
        return recommendations
    
    def _analyze_disk_usage(self, disk_metrics: List[PerformanceMetric]) -> List[OptimizationRecommendation]:
        """Analiza uso de disco"""
        recommendations = []
        
        try:
            if not disk_metrics:
                return recommendations
            
            # Analizar patrones de I/O
            io_values = [m.value for m in disk_metrics]
            avg_io = statistics.mean(io_values)
            max_io = max(io_values)
            
            # I/O alto sostenido
            if avg_io > 1e9:  # 1GB/s promedio
                recommendations.append(OptimizationRecommendation(
                    recommendation_id=f"disk_io_high_{int(time.time())}",
                    optimization_type=OptimizationType.RESOURCE_ALLOCATION,
                    priority=OptimizationPriority.MEDIUM,
                    description="I/O de disco alto. Considerar SSD o optimización de acceso.",
                    expected_improvement=25.0,
                    implementation_cost=6.0,
                    risk_level="low",
                    target_nodes=[m.node_id for m in disk_metrics],
                    target_services=list(set(m.service_name for m in disk_metrics if m.service_name)),
                    parameters={
                        "current_avg_io": avg_io,
                        "recommended_action": "optimize_storage",
                        "suggestions": ["upgrade_to_ssd", "implement_caching", "optimize_queries"]
                    }
                ))
            
        except Exception as e:
            logger.error(f"❌ Error analizando disco: {e}")
        
        return recommendations
    
    async def implement(self, recommendation: OptimizationRecommendation) -> bool:
        """Implementa optimización de recursos"""
        try:
            action = recommendation.parameters.get("recommended_action")
            
            if action == "scale_up_cpu":
                return await self._scale_cpu_up(recommendation)
            elif action == "scale_down_cpu":
                return await self._scale_cpu_down(recommendation)
            elif action == "increase_memory":
                return await self._increase_memory(recommendation)
            elif action == "optimize_storage":
                return await self._optimize_storage(recommendation)
            
            return False
            
        except Exception as e:
            logger.error(f"❌ Error implementando optimización de recursos: {e}")
            return False
    
    async def _scale_cpu_up(self, recommendation: OptimizationRecommendation) -> bool:
        """Escala CPU hacia arriba"""
        logger.info("🔧 Simulando escalado de CPU hacia arriba")
        # En implementación real, esto interactuaría con orquestador (K8s, Docker, etc.)
        await asyncio.sleep(1)
        return True
    
    async def _scale_cpu_down(self, recommendation: OptimizationRecommendation) -> bool:
        """Escala CPU hacia abajo"""
        logger.info("🔧 Simulando escalado de CPU hacia abajo")
        await asyncio.sleep(1)
        return True
    
    async def _increase_memory(self, recommendation: OptimizationRecommendation) -> bool:
        """Aumenta memoria"""
        logger.info("🔧 Simulando aumento de memoria")
        await asyncio.sleep(1)
        return True
    
    async def _optimize_storage(self, recommendation: OptimizationRecommendation) -> bool:
        """Optimiza almacenamiento"""
        logger.info("🔧 Simulando optimización de almacenamiento")
        await asyncio.sleep(1)
        return True

class LoadBalancerOptimizer:
    """Optimizador de balanceador de carga"""
    
    def analyze(self, metrics_by_type: Dict[MetricType, List[PerformanceMetric]], 
                anomalies: List[PerformanceAnomaly]) -> List[OptimizationRecommendation]:
        """Analiza balanceador de carga"""
        recommendations = []
        
        try:
            # Analizar distribución de carga
            if MetricType.CONNECTION_COUNT in metrics_by_type:
                conn_recs = self._analyze_connection_distribution(metrics_by_type[MetricType.CONNECTION_COUNT])
                recommendations.extend(conn_recs)
            
            # Analizar latencia
            if MetricType.LATENCY in metrics_by_type:
                latency_recs = self._analyze_latency_patterns(metrics_by_type[MetricType.LATENCY])
                recommendations.extend(latency_recs)
            
        except Exception as e:
            logger.error(f"❌ Error analizando balanceador de carga: {e}")
        
        return recommendations
    
    def _analyze_connection_distribution(self, conn_metrics: List[PerformanceMetric]) -> List[OptimizationRecommendation]:
        """Analiza distribución de conexiones"""
        recommendations = []
        
        try:
            # Agrupar por nodo
            connections_by_node = defaultdict(list)
            for metric in conn_metrics:
                connections_by_node[metric.node_id].append(metric.value)
            
            if len(connections_by_node) > 1:
                # Calcular desbalance
                node_averages = {node: statistics.mean(values) for node, values in connections_by_node.items()}
                avg_connections = statistics.mean(node_averages.values())
                max_connections = max(node_averages.values())
                min_connections = min(node_averages.values())
                
                # Detectar desbalance significativo
                if max_connections > min_connections * 2:
                    recommendations.append(OptimizationRecommendation(
                        recommendation_id=f"load_balance_{int(time.time())}",
                        optimization_type=OptimizationType.LOAD_BALANCING,
                        priority=OptimizationPriority.MEDIUM,
                        description=f"Desbalance de carga detectado. Ratio: {max_connections/min_connections:.2f}",
                        expected_improvement=20.0,
                        implementation_cost=4.0,
                        risk_level="low",
                        target_nodes=list(connections_by_node.keys()),
                        target_services=[],
                        parameters={
                            "node_averages": node_averages,
                            "recommended_action": "rebalance_load",
                            "algorithm": "weighted_round_robin"
                        }
                    ))
            
        except Exception as e:
            logger.error(f"❌ Error analizando distribución de conexiones: {e}")
        
        return recommendations
    
    def _analyze_latency_patterns(self, latency_metrics: List[PerformanceMetric]) -> List[OptimizationRecommendation]:
        """Analiza patrones de latencia"""
        recommendations = []
        
        try:
            if not latency_metrics:
                return recommendations
            
            latency_values = [m.value for m in latency_metrics]
            avg_latency = statistics.mean(latency_values)
            p95_latency = np.percentile(latency_values, 95)
            
            # Latencia alta
            if avg_latency > 1000:  # 1 segundo
                recommendations.append(OptimizationRecommendation(
                    recommendation_id=f"latency_high_{int(time.time())}",
                    optimization_type=OptimizationType.LOAD_BALANCING,
                    priority=OptimizationPriority.HIGH,
                    description=f"Latencia alta detectada ({avg_latency:.0f}ms promedio)",
                    expected_improvement=35.0,
                    implementation_cost=5.0,
                    risk_level="medium",
                    target_nodes=[m.node_id for m in latency_metrics],
                    target_services=list(set(m.service_name for m in latency_metrics if m.service_name)),
                    parameters={
                        "avg_latency": avg_latency,
                        "p95_latency": p95_latency,
                        "recommended_action": "optimize_routing",
                        "suggestions": ["geographic_routing", "connection_pooling", "caching"]
                    }
                ))
            
        except Exception as e:
            logger.error(f"❌ Error analizando patrones de latencia: {e}")
        
        return recommendations
    
    async def implement(self, recommendation: OptimizationRecommendation) -> bool:
        """Implementa optimización de balanceador"""
        try:
            action = recommendation.parameters.get("recommended_action")
            
            if action == "rebalance_load":
                return await self._rebalance_load(recommendation)
            elif action == "optimize_routing":
                return await self._optimize_routing(recommendation)
            
            return False
            
        except Exception as e:
            logger.error(f"❌ Error implementando optimización de balanceador: {e}")
            return False
    
    async def _rebalance_load(self, recommendation: OptimizationRecommendation) -> bool:
        """Rebalancea carga"""
        logger.info("⚖️ Simulando rebalanceo de carga")
        await asyncio.sleep(1)
        return True
    
    async def _optimize_routing(self, recommendation: OptimizationRecommendation) -> bool:
        """Optimiza enrutamiento"""
        logger.info("🛣️ Simulando optimización de enrutamiento")
        await asyncio.sleep(1)
        return True

class CacheOptimizer:
    """Optimizador de caché"""
    
    def analyze(self, metrics_by_type: Dict[MetricType, List[PerformanceMetric]], 
                anomalies: List[PerformanceAnomaly]) -> List[OptimizationRecommendation]:
        """Analiza uso de caché"""
        recommendations = []
        
        try:
            # Analizar patrones de acceso
            if MetricType.LATENCY in metrics_by_type and MetricType.THROUGHPUT in metrics_by_type:
                cache_recs = self._analyze_cache_patterns(
                    metrics_by_type[MetricType.LATENCY],
                    metrics_by_type[MetricType.THROUGHPUT]
                )
                recommendations.extend(cache_recs)
            
        except Exception as e:
            logger.error(f"❌ Error analizando caché: {e}")
        
        return recommendations
    
    def _analyze_cache_patterns(self, latency_metrics: List[PerformanceMetric], 
                               throughput_metrics: List[PerformanceMetric]) -> List[OptimizationRecommendation]:
        """Analiza patrones de caché"""
        recommendations = []
        
        try:
            # Calcular métricas combinadas
            avg_latency = statistics.mean([m.value for m in latency_metrics])
            avg_throughput = statistics.mean([m.value for m in throughput_metrics])
            
            # Detectar oportunidades de caché
            if avg_latency > 500 and avg_throughput < 1000:  # Latencia alta, throughput bajo
                recommendations.append(OptimizationRecommendation(
                    recommendation_id=f"cache_opportunity_{int(time.time())}",
                    optimization_type=OptimizationType.CACHING,
                    priority=OptimizationPriority.MEDIUM,
                    description="Oportunidad de caché detectada. Latencia alta con throughput bajo.",
                    expected_improvement=45.0,
                    implementation_cost=6.0,
                    risk_level="low",
                    target_nodes=list(set(m.node_id for m in latency_metrics + throughput_metrics)),
                    target_services=list(set(m.service_name for m in latency_metrics + throughput_metrics if m.service_name)),
                    parameters={
                        "avg_latency": avg_latency,
                        "avg_throughput": avg_throughput,
                        "recommended_action": "implement_caching",
                        "cache_types": ["redis", "memcached", "application_cache"]
                    }
                ))
            
        except Exception as e:
            logger.error(f"❌ Error analizando patrones de caché: {e}")
        
        return recommendations
    
    async def implement(self, recommendation: OptimizationRecommendation) -> bool:
        """Implementa optimización de caché"""
        try:
            action = recommendation.parameters.get("recommended_action")
            
            if action == "implement_caching":
                return await self._implement_caching(recommendation)
            
            return False
            
        except Exception as e:
            logger.error(f"❌ Error implementando optimización de caché: {e}")
            return False
    
    async def _implement_caching(self, recommendation: OptimizationRecommendation) -> bool:
        """Implementa caché"""
        logger.info("🗄️ Simulando implementación de caché")
        await asyncio.sleep(1)
        return True

class AlgorithmOptimizer:
    """Optimizador de algoritmos"""
    
    def analyze(self, metrics_by_type: Dict[MetricType, List[PerformanceMetric]], 
                anomalies: List[PerformanceAnomaly]) -> List[OptimizationRecommendation]:
        """Analiza algoritmos"""
        recommendations = []
        
        try:
            # Analizar eficiencia algorítmica
            if MetricType.CPU_USAGE in metrics_by_type and MetricType.THROUGHPUT in metrics_by_type:
                algo_recs = self._analyze_algorithm_efficiency(
                    metrics_by_type[MetricType.CPU_USAGE],
                    metrics_by_type[MetricType.THROUGHPUT]
                )
                recommendations.extend(algo_recs)
            
        except Exception as e:
            logger.error(f"❌ Error analizando algoritmos: {e}")
        
        return recommendations
    
    def _analyze_algorithm_efficiency(self, cpu_metrics: List[PerformanceMetric], 
                                    throughput_metrics: List[PerformanceMetric]) -> List[OptimizationRecommendation]:
        """Analiza eficiencia algorítmica"""
        recommendations = []
        
        try:
            avg_cpu = statistics.mean([m.value for m in cpu_metrics])
            avg_throughput = statistics.mean([m.value for m in throughput_metrics])
            
            # Detectar ineficiencia (CPU alto, throughput bajo)
            if avg_cpu > 70 and avg_throughput < 500:
                recommendations.append(OptimizationRecommendation(
                    recommendation_id=f"algorithm_inefficient_{int(time.time())}",
                    optimization_type=OptimizationType.ALGORITHM_TUNING,
                    priority=OptimizationPriority.MEDIUM,
                    description="Posible ineficiencia algorítmica. CPU alto con throughput bajo.",
                    expected_improvement=30.0,
                    implementation_cost=8.0,
                    risk_level="medium",
                    target_nodes=list(set(m.node_id for m in cpu_metrics + throughput_metrics)),
                    target_services=list(set(m.service_name for m in cpu_metrics + throughput_metrics if m.service_name)),
                    parameters={
                        "avg_cpu": avg_cpu,
                        "avg_throughput": avg_throughput,
                        "recommended_action": "optimize_algorithms",
                        "suggestions": ["profile_code", "optimize_loops", "parallel_processing"]
                    }
                ))
            
        except Exception as e:
            logger.error(f"❌ Error analizando eficiencia algorítmica: {e}")
        
        return recommendations
    
    async def implement(self, recommendation: OptimizationRecommendation) -> bool:
        """Implementa optimización de algoritmos"""
        try:
            action = recommendation.parameters.get("recommended_action")
            
            if action == "optimize_algorithms":
                return await self._optimize_algorithms(recommendation)
            
            return False
            
        except Exception as e:
            logger.error(f"❌ Error implementando optimización de algoritmos: {e}")
            return False
    
    async def _optimize_algorithms(self, recommendation: OptimizationRecommendation) -> bool:
        """Optimiza algoritmos"""
        logger.info("🧮 Simulando optimización de algoritmos")
        await asyncio.sleep(1)
        return True

class PerformanceReporter:
    """Generador de reportes de rendimiento"""
    
    def __init__(self):
        self.report_history: List[Dict[str, Any]] = []
        logger.info("📊 Generador de reportes inicializado")
    
    def generate_performance_report(self, metrics: List[PerformanceMetric], 
                                  anomalies: List[PerformanceAnomaly],
                                  recommendations: List[OptimizationRecommendation]) -> Dict[str, Any]:
        """Genera reporte de rendimiento"""
        try:
            report = {
                "timestamp": time.time(),
                "summary": self._generate_summary(metrics, anomalies, recommendations),
                "metrics_analysis": self._analyze_metrics(metrics),
                "anomalies_analysis": self._analyze_anomalies(anomalies),
                "recommendations_analysis": self._analyze_recommendations(recommendations),
                "performance_score": self._calculate_performance_score(metrics, anomalies),
                "trends": self._analyze_trends(metrics),
                "alerts": self._generate_alerts(anomalies, recommendations)
            }
            
            self.report_history.append(report)
            logger.info("📋 Reporte de rendimiento generado")
            
            return report
            
        except Exception as e:
            logger.error(f"❌ Error generando reporte: {e}")
            return {}
    
    def _generate_summary(self, metrics: List[PerformanceMetric], 
                         anomalies: List[PerformanceAnomaly],
                         recommendations: List[OptimizationRecommendation]) -> Dict[str, Any]:
        """Genera resumen ejecutivo"""
        return {
            "total_metrics": len(metrics),
            "total_anomalies": len(anomalies),
            "critical_anomalies": len([a for a in anomalies if a.severity == "critical"]),
            "total_recommendations": len(recommendations),
            "high_priority_recommendations": len([r for r in recommendations if r.priority == OptimizationPriority.HIGH]),
            "nodes_monitored": len(set(m.node_id for m in metrics)),
            "services_monitored": len(set(m.service_name for m in metrics if m.service_name))
        }
    
    def _analyze_metrics(self, metrics: List[PerformanceMetric]) -> Dict[str, Any]:
        """Analiza métricas"""
        analysis = {}
        
        # Agrupar por tipo
        metrics_by_type = defaultdict(list)
        for metric in metrics:
            metrics_by_type[metric.metric_type].append(metric.value)
        
        # Calcular estadísticas por tipo
        for metric_type, values in metrics_by_type.items():
            if values:
                analysis[metric_type.value] = {
                    "count": len(values),
                    "mean": statistics.mean(values),
                    "median": statistics.median(values),
                    "std": statistics.stdev(values) if len(values) > 1 else 0,
                    "min": min(values),
                    "max": max(values),
                    "p95": np.percentile(values, 95),
                    "p99": np.percentile(values, 99)
                }
        
        return analysis
    
    def _analyze_anomalies(self, anomalies: List[PerformanceAnomaly]) -> Dict[str, Any]:
        """Analiza anomalías"""
        if not anomalies:
            return {"total": 0}
        
        return {
            "total": len(anomalies),
            "by_severity": {
                severity: len([a for a in anomalies if a.severity == severity])
                for severity in set(a.severity for a in anomalies)
            },
            "by_metric_type": {
                metric_type.value: len([a for a in anomalies if a.metric_type == metric_type])
                for metric_type in set(a.metric_type for a in anomalies)
            },
            "by_node": {
                node: len([a for a in anomalies if a.node_id == node])
                for node in set(a.node_id for a in anomalies)
            },
            "average_confidence": statistics.mean([a.confidence for a in anomalies])
        }
    
    def _analyze_recommendations(self, recommendations: List[OptimizationRecommendation]) -> Dict[str, Any]:
        """Analiza recomendaciones"""
        if not recommendations:
            return {"total": 0}
        
        return {
            "total": len(recommendations),
            "by_priority": {
                priority.value: len([r for r in recommendations if r.priority == priority])
                for priority in set(r.priority for r in recommendations)
            },
            "by_type": {
                opt_type.value: len([r for r in recommendations if r.optimization_type == opt_type])
                for opt_type in set(r.optimization_type for r in recommendations)
            },
            "total_expected_improvement": sum(r.expected_improvement for r in recommendations),
            "total_implementation_cost": sum(r.implementation_cost for r in recommendations),
            "average_risk_level": statistics.mode([r.risk_level for r in recommendations]) if recommendations else "unknown"
        }
    
    def _calculate_performance_score(self, metrics: List[PerformanceMetric], 
                                   anomalies: List[PerformanceAnomaly]) -> Dict[str, Any]:
        """Calcula score de rendimiento"""
        try:
            base_score = 100.0
            
            # Penalizar por anomalías
            for anomaly in anomalies:
                if anomaly.severity == "critical":
                    base_score -= 20
                elif anomaly.severity == "high":
                    base_score -= 10
                elif anomaly.severity == "medium":
                    base_score -= 5
            
            # Penalizar por métricas fuera de umbral
            for metric in metrics:
                if metric.threshold_critical > 0 and metric.value > metric.threshold_critical:
                    base_score -= 15
                elif metric.threshold_warning > 0 and metric.value > metric.threshold_warning:
                    base_score -= 5
            
            # Normalizar score
            final_score = max(0, min(100, base_score))
            
            # Determinar nivel
            if final_score >= 90:
                level = PerformanceLevel.EXCELLENT
            elif final_score >= 75:
                level = PerformanceLevel.GOOD
            elif final_score >= 60:
                level = PerformanceLevel.ACCEPTABLE
            elif final_score >= 40:
                level = PerformanceLevel.POOR
            else:
                level = PerformanceLevel.CRITICAL
            
            return {
                "score": final_score,
                "level": level.value,
                "factors": {
                    "anomalies_impact": len(anomalies) * -5,
                    "threshold_violations": len([m for m in metrics if m.threshold_warning > 0 and m.value > m.threshold_warning])
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Error calculando score de rendimiento: {e}")
            return {"score": 0, "level": "unknown"}
    
    def _analyze_trends(self, metrics: List[PerformanceMetric]) -> Dict[str, Any]:
        """Analiza tendencias"""
        trends = {}
        
        try:
            # Agrupar métricas por tipo y ordenar por tiempo
            metrics_by_type = defaultdict(list)
            for metric in sorted(metrics, key=lambda m: m.timestamp):
                metrics_by_type[metric.metric_type].append(metric)
            
            # Calcular tendencias para cada tipo
            for metric_type, type_metrics in metrics_by_type.items():
                if len(type_metrics) >= 3:
                    values = [m.value for m in type_metrics]
                    timestamps = [m.timestamp for m in type_metrics]
                    
                    # Calcular tendencia lineal simple
                    if len(values) > 1:
                        x = np.array(range(len(values)))
                        y = np.array(values)
                        slope = np.polyfit(x, y, 1)[0]
                        
                        trend_direction = "increasing" if slope > 0 else "decreasing" if slope < 0 else "stable"
                        trend_strength = abs(slope) / (max(values) - min(values)) if max(values) != min(values) else 0
                        
                        trends[metric_type.value] = {
                            "direction": trend_direction,
                            "strength": trend_strength,
                            "slope": slope,
                            "recent_value": values[-1],
                            "change_percent": ((values[-1] - values[0]) / values[0] * 100) if values[0] != 0 else 0
                        }
            
        except Exception as e:
            logger.error(f"❌ Error analizando tendencias: {e}")
        
        return trends
    
    def _generate_alerts(self, anomalies: List[PerformanceAnomaly],
                        recommendations: List[OptimizationRecommendation]) -> List[Dict[str, Any]]:
        """Genera alertas"""
        alerts = []
        
        try:
            # Alertas por anomalías críticas
            critical_anomalies = [a for a in anomalies if a.severity == "critical"]
            for anomaly in critical_anomalies:
                alerts.append({
                    "type": "anomaly",
                    "severity": "critical",
                    "title": f"Anomalía crítica en {anomaly.metric_type.value}",
                    "description": anomaly.description,
                    "node_id": anomaly.node_id,
                    "service_name": anomaly.service_name,
                    "timestamp": anomaly.detected_at
                })
            
            # Alertas por recomendaciones críticas
            critical_recommendations = [r for r in recommendations if r.priority == OptimizationPriority.CRITICAL]
            for rec in critical_recommendations:
                alerts.append({
                    "type": "recommendation",
                    "severity": "critical",
                    "title": f"Optimización crítica requerida",
                    "description": rec.description,
                    "expected_improvement": rec.expected_improvement,
                    "timestamp": time.time()
                })
            
        except Exception as e:
            logger.error(f"❌ Error generando alertas: {e}")
        
        return alerts
    
    def export_report(self, report: Dict[str, Any], format: str = "json") -> str:
        """Exporta reporte"""
        try:
            timestamp = int(report.get("timestamp", time.time()))
            filename = f"performance_report_{timestamp}.{format}"
            
            if format == "json":
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(report, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"📄 Reporte exportado: {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"❌ Error exportando reporte: {e}")
            return ""

class PerformanceOptimizerOrchestrator:
    """Orquestador principal del optimizador de rendimiento"""
    
    def __init__(self, collection_interval: int = 30, analysis_interval: int = 300):
        self.collection_interval = collection_interval
        self.analysis_interval = analysis_interval
        
        # Componentes principales
        self.metrics_collector = MetricsCollector(collection_interval)
        self.anomaly_detector = AnomalyDetector()
        self.performance_optimizer = PerformanceOptimizer()
        self.performance_reporter = PerformanceReporter()
        
        # Estado del sistema
        self.is_running = False
        self.analysis_thread = None
        self.last_analysis = 0
        
        # Configuración
        self.config = {
            "enable_auto_optimization": True,
            "max_concurrent_optimizations": 3,
            "optimization_cooldown": 1800,  # 30 minutos
            "baseline_training_period": 3600,  # 1 hora
            "anomaly_sensitivity": 0.1
        }
        
        logger.info("🎯 Orquestador de optimización de rendimiento inicializado")
    
    async def start(self):
        """Inicia el sistema de optimización"""
        try:
            if self.is_running:
                logger.warning("⚠️ Sistema ya está ejecutándose")
                return
            
            self.is_running = True
            
            # Iniciar recolección de métricas
            self.metrics_collector.start_collection()
            
            # Iniciar análisis periódico
            self.analysis_thread = threading.Thread(target=self._analysis_loop)
            self.analysis_thread.daemon = True
            self.analysis_thread.start()
            
            logger.info("🚀 Sistema de optimización iniciado")
            
            # Período inicial de entrenamiento
            await self._initial_training_period()
            
        except Exception as e:
            logger.error(f"❌ Error iniciando sistema: {e}")
            await self.stop()
    
    async def stop(self):
        """Detiene el sistema de optimización"""
        try:
            self.is_running = False
            
            # Detener recolección
            self.metrics_collector.stop_collection()
            
            # Esperar thread de análisis
            if self.analysis_thread:
                self.analysis_thread.join(timeout=10)
            
            logger.info("🛑 Sistema de optimización detenido")
            
        except Exception as e:
            logger.error(f"❌ Error deteniendo sistema: {e}")
    
    async def _initial_training_period(self):
        """Período inicial de entrenamiento"""
        try:
            logger.info("🎓 Iniciando período de entrenamiento inicial...")
            
            # Esperar acumulación de métricas
            await asyncio.sleep(60)  # 1 minuto
            
            # Obtener métricas para entrenamiento
            training_metrics = self.metrics_collector.get_recent_metrics(limit=500)
            
            if len(training_metrics) >= 50:
                # Entrenar líneas base por servicio/nodo
                services_nodes = set((m.service_name, m.node_id) for m in training_metrics)
                
                for service_name, node_id in services_nodes:
                    service_metrics = [m for m in training_metrics 
                                     if m.service_name == service_name and m.node_id == node_id]
                    
                    if len(service_metrics) >= 10:
                        self.anomaly_detector.train_baseline(service_metrics, service_name, node_id)
                
                logger.info("✅ Entrenamiento inicial completado")
            else:
                logger.warning("⚠️ Insuficientes métricas para entrenamiento inicial")
            
        except Exception as e:
            logger.error(f"❌ Error en entrenamiento inicial: {e}")
    
    def _analysis_loop(self):
        """Loop principal de análisis"""
        while self.is_running:
            try:
                current_time = time.time()
                
                # Verificar si es tiempo de análisis
                if current_time - self.last_analysis >= self.analysis_interval:
                    asyncio.run(self._perform_analysis())
                    self.last_analysis = current_time
                
                time.sleep(30)  # Verificar cada 30 segundos
                
            except Exception as e:
                logger.error(f"❌ Error en loop de análisis: {e}")
                time.sleep(60)
    
    async def _perform_analysis(self):
        """Realiza análisis completo"""
        try:
            logger.info("🔍 Iniciando análisis de rendimiento...")
            
            # Obtener métricas recientes
            recent_metrics = self.metrics_collector.get_recent_metrics(limit=1000)
            
            if not recent_metrics:
                logger.warning("⚠️ No hay métricas disponibles para análisis")
                return
            
            # Detectar anomalías
            anomalies = self.anomaly_detector.detect_anomalies(recent_metrics)
            
            # Generar recomendaciones
            recommendations = self.performance_optimizer.analyze_performance(recent_metrics, anomalies)
            
            # Generar reporte
            report = self.performance_reporter.generate_performance_report(
                recent_metrics, anomalies, recommendations
            )
            
            # Log resumen
            logger.info(f"📊 Análisis completado:")
            logger.info(f"   - Métricas analizadas: {len(recent_metrics)}")
            logger.info(f"   - Anomalías detectadas: {len(anomalies)}")
            logger.info(f"   - Recomendaciones generadas: {len(recommendations)}")
            logger.info(f"   - Score de rendimiento: {report.get('performance_score', {}).get('score', 'N/A')}")
            
            # Implementar optimizaciones automáticas si está habilitado
            if self.config["enable_auto_optimization"]:
                await self._auto_implement_optimizations(recommendations)
            
            # Exportar reporte
            self.performance_reporter.export_report(report)
            
        except Exception as e:
            logger.error(f"❌ Error realizando análisis: {e}")
    
    async def _auto_implement_optimizations(self, recommendations: List[OptimizationRecommendation]):
        """Implementa optimizaciones automáticamente"""
        try:
            # Filtrar recomendaciones para auto-implementación
            auto_recommendations = [
                r for r in recommendations
                if r.priority in [OptimizationPriority.CRITICAL, OptimizationPriority.HIGH]
                and r.risk_level in ["low", "medium"]
                and r.implementation_cost <= 5.0
            ]
            
            # Limitar número de optimizaciones concurrentes
            max_concurrent = self.config["max_concurrent_optimizations"]
            current_active = len(self.performance_optimizer.active_optimizations)
            
            available_slots = max_concurrent - current_active
            auto_recommendations = auto_recommendations[:available_slots]
            
            # Implementar recomendaciones
            for recommendation in auto_recommendations:
                logger.info(f"🤖 Auto-implementando: {recommendation.description}")
                success = await self.performance_optimizer.implement_recommendation(recommendation)
                
                if success:
                    logger.info(f"✅ Auto-optimización exitosa: {recommendation.recommendation_id}")
                else:
                    logger.error(f"❌ Fallo auto-optimización: {recommendation.recommendation_id}")
                
                # Pequeña pausa entre implementaciones
                await asyncio.sleep(5)
            
        except Exception as e:
            logger.error(f"❌ Error en auto-implementación: {e}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Obtiene estado del sistema"""
        try:
            return {
                "is_running": self.is_running,
                "last_analysis": self.last_analysis,
                "next_analysis": self.last_analysis + self.analysis_interval,
                "metrics_collected": len(self.metrics_collector.metrics_history),
                "baselines_trained": len(self.anomaly_detector.baselines),
                "optimization_status": self.performance_optimizer.get_optimization_status(),
                "reports_generated": len(self.performance_reporter.report_history),
                "configuration": self.config
            }
            
        except Exception as e:
            logger.error(f"❌ Error obteniendo estado del sistema: {e}")
            return {}
    
    def update_configuration(self, new_config: Dict[str, Any]):
        """Actualiza configuración"""
        try:
            self.config.update(new_config)
            logger.info(f"⚙️ Configuración actualizada: {new_config}")
            
        except Exception as e:
            logger.error(f"❌ Error actualizando configuración: {e}")

# Función principal para demostración
async def main():
    """Función principal de demostración"""
    try:
        logger.info("🎯 Iniciando demostración del Optimizador de Rendimiento AEGIS")
        
        # Crear orquestador
        orchestrator = PerformanceOptimizerOrchestrator(
            collection_interval=10,  # Recolectar cada 10 segundos
            analysis_interval=60     # Analizar cada minuto
        )
        
        # Iniciar sistema
        await orchestrator.start()
        
        # Ejecutar por un período de demostración
        logger.info("🔄 Ejecutando optimización por 5 minutos...")
        await asyncio.sleep(300)  # 5 minutos
        
        # Mostrar estado final
        status = orchestrator.get_system_status()
        logger.info(f"📊 Estado final del sistema:")
        logger.info(f"   - Métricas recolectadas: {status.get('metrics_collected', 0)}")
        logger.info(f"   - Líneas base entrenadas: {status.get('baselines_trained', 0)}")
        logger.info(f"   - Reportes generados: {status.get('reports_generated', 0)}")
        
        # Detener sistema
        await orchestrator.stop()
        
        logger.info("✅ Demostración completada exitosamente")
        
    except Exception as e:
        logger.error(f"❌ Error en demostración: {e}")

if __name__ == "__main__":
    asyncio.run(main())