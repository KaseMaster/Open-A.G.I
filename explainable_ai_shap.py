#!/usr/bin/env python3
"""
🧠 AEGIS Explainable AI with SHAP - Sprint 4.2
Sistema completo de IA explicable con integración de SHAP y técnicas avanzadas
"""

import asyncio
import time
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure

# Intentar importar SHAP
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    warnings.warn("SHAP no está disponible. Instalar con: pip install shap")

# Importar componentes del framework
from ml_framework_integration import MLFrameworkManager, MLFramework

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExplanationMethod(Enum):
    """Métodos de explicación disponibles"""
    SHAP_VALUES = "shap_values"
    LIME = "lime"
    PERMUTATION_IMPORTANCE = "permutation_importance"
    PARTIAL_DEPENDENCE = "partial_dependence"
    FEATURE_INTERACTIONS = "feature_interactions"
    GRADIENT_BASED = "gradient_based"
    DEEP_TAYLOR = "deep_taylor"

class ExplanationScope(Enum):
    """Alcance de la explicación"""
    LOCAL = "local"      # Explicación de una predicción individual
    GLOBAL = "global"    # Explicación del modelo completo
    COHORT = "cohort"    # Explicación de un grupo de predicciones

@dataclass
class SHAPExplanation:
    """Explicación usando SHAP"""
    method: ExplanationMethod
    scope: ExplanationScope
    shap_values: Optional[np.ndarray] = None
    base_values: Optional[np.ndarray] = None
    data: Optional[np.ndarray] = None
    feature_names: Optional[List[str]] = None
    explanation_text: str = ""
    visualization_data: Dict[str, Any] = field(default_factory=dict)
    processing_time: float = 0.0

@dataclass
class FeatureImportance:
    """Importancia de features"""
    feature_name: str
    importance_score: float
    std_dev: Optional[float] = None
    rank: Optional[int] = None
    contribution_direction: str = "positive"  # positive, negative, mixed

@dataclass
class ModelExplanation:
    """Explicación completa del modelo"""
    model_name: str
    model_type: str
    global_explanations: List[FeatureImportance] = field(default_factory=list)
    local_explanations: Dict[str, SHAPExplanation] = field(default_factory=dict)
    interaction_effects: Dict[Tuple[str, str], float] = field(default_factory=dict)
    model_insights: List[str] = field(default_factory=list)
    fairness_analysis: Dict[str, Any] = field(default_factory=dict)
    robustness_analysis: Dict[str, Any] = field(default_factory=dict)

# ===== EXPLICADORES DE SHAP =====

class SHAPExplainer:
    """Explainer basado en SHAP"""

    def __init__(self):
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP no está disponible. Instalar con: pip install shap")

        self.explainer_cache: Dict[str, Any] = {}

    def create_explainer(self, model: Any, background_data: np.ndarray,
                        model_type: str = "auto") -> Any:
        """Crear explainer SHAP apropiado para el modelo"""

        cache_key = f"{model_type}_{hash(background_data.tobytes()):x}"

        if cache_key in self.explainer_cache:
            return self.explainer_cache[cache_key]

        # Check if shap is available
        if not SHAP_AVAILABLE:
            return None

        try:
            if model_type == "tree" or hasattr(model, 'predict_proba'):
                # Para modelos de árbol o sklearn
                explainer = shap.TreeExplainer(model, background_data)  # type: ignore
            elif model_type == "linear":
                explainer = shap.LinearExplainer(model, background_data)  # type: ignore
            elif model_type == "deep" or isinstance(model, nn.Module):
                # Para modelos de deep learning
                explainer = shap.DeepExplainer(model, background_data)  # type: ignore
            else:
                # Fallback: KernelExplainer
                explainer = shap.KernelExplainer(model.predict, background_data)  # type: ignore

            self.explainer_cache[cache_key] = explainer
            return explainer

        except Exception as e:
            logger.warning(f"Error creando SHAP explainer: {e}")
            # Fallback simplificado
            return None

    def explain_prediction(self, explainer: Any, input_data: np.ndarray) -> SHAPExplanation:
        """Explicar una predicción individual"""

        start_time = time.time()

        try:
            # Calcular SHAP values
            shap_values = explainer.shap_values(input_data)

            # Para clasificación múltiple, tomar la clase positiva
            if isinstance(shap_values, list) and len(shap_values) > 1:
                # Tomar la clase con mayor valor absoluto
                shap_values_combined = np.sum(np.abs(shap_values), axis=0)
                shap_values = shap_values[np.argmax(np.abs(shap_values).sum(axis=(1, 2)))]
            elif isinstance(shap_values, list):
                shap_values = shap_values[0]

            processing_time = time.time() - start_time

            return SHAPExplanation(
                method=ExplanationMethod.SHAP_VALUES,
                scope=ExplanationScope.LOCAL,
                shap_values=shap_values,
                data=input_data,
                processing_time=processing_time,
                explanation_text=self._generate_local_explanation_text(shap_values, input_data)
            )

        except Exception as e:
            logger.error(f"Error explicando predicción: {e}")
            return SHAPExplanation(
                method=ExplanationMethod.SHAP_VALUES,
                scope=ExplanationScope.LOCAL,
                processing_time=time.time() - start_time
            )

    def explain_model_global(self, explainer: Any, background_data: np.ndarray,
                           feature_names: Optional[List[str]] = None) -> List[FeatureImportance]:
        """Explicación global del modelo"""

        try:
            # Calcular SHAP values para múltiples samples
            sample_size = min(100, len(background_data))
            sample_indices = np.random.choice(len(background_data), sample_size, replace=False)
            sample_data = background_data[sample_indices]

            shap_values = explainer.shap_values(sample_data)

            # Procesar según el tipo de modelo
            if isinstance(shap_values, list):
                # Multi-class classification
                shap_values = shap_values[0]  # Tomar primera clase

            # Calcular importancia global
            feature_importance = np.abs(shap_values).mean(axis=0)

            # Crear objetos FeatureImportance
            importance_list = []
            for i, importance in enumerate(feature_importance):
                feature_name = feature_names[i] if feature_names and i < len(feature_names) else f"feature_{i}"

                importance_list.append(FeatureImportance(
                    feature_name=feature_name,
                    importance_score=float(importance),
                    rank=i + 1
                ))

            # Ordenar por importancia
            importance_list.sort(key=lambda x: x.importance_score, reverse=True)

            return importance_list

        except Exception as e:
            logger.error(f"Error en explicación global: {e}")
            return []

    def _generate_local_explanation_text(self, shap_values: np.ndarray,
                                       input_data: np.ndarray) -> str:
        """Generar texto explicativo para explicación local"""

        if shap_values.ndim == 1:
            # Single output
            top_features = np.argsort(np.abs(shap_values))[-3:][::-1]  # Top 3 features

            explanation = "Esta predicción se explica principalmente por: "
            for i, feature_idx in enumerate(top_features):
                impact = shap_values[feature_idx]
                direction = "aumenta" if impact > 0 else "disminuye"
                explanation += f"{direction} el valor (feature {feature_idx}: {impact:.2f})"
                if i < len(top_features) - 1:
                    explanation += ", "

        else:
            explanation = f"Explicación generada para {shap_values.shape[1]} features"

        return explanation

# ===== ALTERNATIVAS CUANDO SHAP NO ESTÁ DISPONIBLE =====

class FallbackExplainer:
    """Explainer alternativo cuando SHAP no está disponible"""

    def __init__(self):
        self.feature_importance_cache: Dict[str, List[FeatureImportance]] = {}

    def explain_prediction_permutation(self, model: Any, input_data: np.ndarray,
                                     background_data: np.ndarray,
                                     feature_names: Optional[List[str]] = None) -> SHAPExplanation:
        """Explicación usando permutation importance"""

        start_time = time.time()

        try:
            # Baseline prediction
            baseline_pred = model.predict(input_data)

            # Calcular importancia permutando cada feature
            importance_scores = []
            n_features = input_data.shape[1]

            for i in range(n_features):
                # Permutar feature i
                permuted_data = input_data.copy()
                permuted_data[0, i] = np.random.permutation(background_data[:, i])[:1]

                # Nueva predicción
                permuted_pred = model.predict(permuted_data)

                # Calcular importancia (cambio en predicción)
                importance = abs(baseline_pred[0] - permuted_pred[0])
                importance_scores.append(importance)

            # Convertir a array similar a SHAP
            shap_like_values = np.array(importance_scores)

            processing_time = time.time() - start_time

            return SHAPExplanation(
                method=ExplanationMethod.PERMUTATION_IMPORTANCE,
                scope=ExplanationScope.LOCAL,
                shap_values=shap_like_values,
                data=input_data,
                processing_time=processing_time,
                explanation_text=self._generate_permutation_explanation(shap_like_values, feature_names)
            )

        except Exception as e:
            logger.error(f"Error en permutation explanation: {e}")
            return SHAPExplanation(
                method=ExplanationMethod.PERMUTATION_IMPORTANCE,
                scope=ExplanationScope.LOCAL,
                processing_time=time.time() - start_time
            )

    def explain_model_permutation(self, model: Any, X: np.ndarray, y: np.ndarray,
                                feature_names: Optional[List[str]] = None,
                                n_repeats: int = 5) -> List[FeatureImportance]:
        """Importancia de features usando permutation importance"""

        try:
            from sklearn.inspection import permutation_importance
            from sklearn.metrics import accuracy_score, r2_score

            # Calcular permutation importance
            if hasattr(model, 'predict_proba'):
                # Classification
                y_pred = model.predict(X)
                baseline_score = accuracy_score(y, y_pred)

                perm_importance = permutation_importance(
                    model, X, y, scoring='accuracy', n_repeats=n_repeats, random_state=42
                )
            else:
                # Regression
                y_pred = model.predict(X)
                baseline_score = r2_score(y, y_pred)

                perm_importance = permutation_importance(
                    model, X, y, scoring='r2', n_repeats=n_repeats, random_state=42
                )

            # Crear lista de importancia
            importance_list = []
            # Handle permutation importance results correctly
            if isinstance(perm_importance, dict):
                # Fallback for dictionary-like results
                importances_mean = perm_importance.get('importances_mean', [])
                importances_std = perm_importance.get('importances_std', [])
            else:
                importances_mean = getattr(perm_importance, 'importances_mean', [])
                importances_std = getattr(perm_importance, 'importances_std', [])
            
            for i, (importance, std) in enumerate(zip(importances_mean, importances_std)):
                feature_name = feature_names[i] if feature_names and i < len(feature_names) else f"feature_{i}"

                importance_list.append(FeatureImportance(
                    feature_name=feature_name,
                    importance_score=float(importance),
                    std_dev=float(std),
                    rank=i + 1
                ))

            # Ordenar por importancia
            importance_list.sort(key=lambda x: x.importance_score, reverse=True)

            return importance_list

        except Exception as e:
            logger.error(f"Error en permutation importance: {e}")
            return []

    def _generate_permutation_explanation(self, importance_scores: np.ndarray,
                                        feature_names: Optional[List[str]] = None) -> str:
        """Generar explicación basada en permutation importance"""

        top_features = np.argsort(importance_scores)[-3:][::-1]  # Top 3

        explanation = "La predicción cambia más cuando se modifica: "
        for i, feature_idx in enumerate(top_features):
            feature_name = feature_names[feature_idx] if feature_names and feature_idx < len(feature_names) else f"feature_{feature_idx}"
            explanation += f"{feature_name} ({importance_scores[feature_idx]:.3f})"
            if i < len(top_features) - 1:
                explanation += ", "

        return explanation

# ===== EXPLICACIONES PARA DEEP LEARNING =====

class DeepLearningExplainer:
    """Explicaciones específicas para modelos de deep learning"""

    def __init__(self):
        self.gradients_cache: Dict[str, Any] = {}

    def explain_with_gradients(self, model: nn.Module, input_tensor: torch.Tensor,
                              target_class: Optional[int] = None) -> SHAPExplanation:
        """Explicación usando gradients (GradCAM-like)"""

        start_time = time.time()

        try:
            model.eval()
            input_tensor.requires_grad_(True)

            # Forward pass
            output = model(input_tensor)

            if target_class is None:
                target_class = output.argmax(dim=1).item()

            # Backward pass para gradients
            model.zero_grad()
            if output.requires_grad:
                output[0, target_class].backward()

            # Obtener gradients
            gradients = input_tensor.grad.detach().numpy() if input_tensor.grad is not None else np.zeros_like(input_tensor.detach().numpy())

            # Calcular importancia (magnitud del gradiente)
            importance_scores = np.abs(gradients).mean(axis=(0, 2, 3)) if gradients.ndim > 2 else np.abs(gradients).flatten()

            processing_time = time.time() - start_time

            return SHAPExplanation(
                method=ExplanationMethod.GRADIENT_BASED,
                scope=ExplanationScope.LOCAL,
                shap_values=importance_scores,
                data=input_tensor.detach().numpy(),
                processing_time=processing_time,
                explanation_text=f"Gradients explican la predicción de clase {target_class}"
            )

        except Exception as e:
            logger.error(f"Error en gradient explanation: {e}")
            return SHAPExplanation(
                method=ExplanationMethod.GRADIENT_BASED,
                scope=ExplanationScope.LOCAL,
                processing_time=time.time() - start_time
            )

    def explain_with_deep_taylor(self, model: nn.Module, input_tensor: torch.Tensor) -> SHAPExplanation:
        """Explicación usando Deep Taylor Decomposition (simplificada)"""

        start_time = time.time()

        try:
            # Implementación simplificada de Deep Taylor
            # En producción, usar librerías especializadas como Innvestigate

            model.eval()

            # Calcular relevancia propagando hacia atrás
            relevance_scores = self._propagate_relevance(model, input_tensor)

            processing_time = time.time() - start_time

            return SHAPExplanation(
                method=ExplanationMethod.DEEP_TAYLOR,
                scope=ExplanationScope.LOCAL,
                shap_values=relevance_scores,
                data=input_tensor.detach().numpy(),
                processing_time=processing_time,
                explanation_text="Relevance scores calculados usando Deep Taylor Decomposition"
            )

        except Exception as e:
            logger.error(f"Error en Deep Taylor explanation: {e}")
            return SHAPExplanation(
                method=ExplanationMethod.DEEP_TAYLOR,
                scope=ExplanationScope.LOCAL,
                processing_time=time.time() - start_time
            )

    def _propagate_relevance(self, model: nn.Module, input_tensor: torch.Tensor) -> np.ndarray:
        """Propagar relevance scores (implementación simplificada)"""

        # Esta es una implementación muy simplificada
        # En producción, implementar LRP (Layer-wise Relevance Propagation) correctamente

        try:
            # Obtener activaciones finales
            with torch.no_grad():
                output = model(input_tensor)

            # Usar la magnitud de la salida como proxy de relevance
            relevance = output.abs().detach().numpy()

            # Flatten para obtener scores por feature
            return relevance.flatten()

        except Exception:
            # Fallback: importancia uniforme
            return np.ones(input_tensor.numel()) / input_tensor.numel()

# ===== VISUALIZACIONES =====

class ExplanationVisualizer:
    """Visualizador de explicaciones"""

    def __init__(self):
        self.plots = {}

    def plot_shap_waterfall(self, shap_explanation: SHAPExplanation,
                           feature_names: Optional[List[str]] = None,
                           title: str = "SHAP Waterfall Plot") -> Optional[Figure]:
        """Crear waterfall plot de SHAP values"""

        try:
            if shap_explanation.shap_values is None:
                return None

            fig, ax = plt.subplots(figsize=(10, 6))

            shap_values = shap_explanation.shap_values.flatten()
            n_features = len(shap_values)

            # Usar nombres de features o índices
            if feature_names and len(feature_names) >= n_features:
                labels = feature_names[:n_features]
            else:
                labels = [f"Feature {i}" for i in range(n_features)]

            # Ordenar por importancia
            sorted_indices = np.argsort(np.abs(shap_values))[::-1]
            sorted_values = shap_values[sorted_indices]
            sorted_labels = [labels[i] for i in sorted_indices]

            # Crear waterfall plot
            cumulative = np.cumsum(sorted_values)
            cumulative = np.insert(cumulative, 0, 0)

            # Plot
            ax.bar(range(len(sorted_values)), sorted_values, bottom=cumulative[:-1],
                  label='SHAP Values', alpha=0.7)

            ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax.set_xlabel('Features')
            ax.set_ylabel('SHAP Value')
            ax.set_title(title)
            ax.set_xticks(range(len(sorted_labels)))
            ax.set_xticklabels(sorted_labels, rotation=45, ha='right')

            plt.tight_layout()
            return fig

        except Exception as e:
            logger.error(f"Error creando waterfall plot: {e}")
            return None

    def plot_feature_importance(self, importance_list: List[FeatureImportance],
                               title: str = "Feature Importance") -> Optional[Figure]:
        """Plot de importancia de features"""

        try:
            fig, ax = plt.subplots(figsize=(12, 8))

            # Tomar top 20 features
            top_features = importance_list[:20]
            names = [f.feature_name for f in top_features]
            scores = [f.importance_score for f in top_features]

            # Crear horizontal bar plot
            bars = ax.barh(range(len(names)), scores, alpha=0.7)
            ax.set_yticks(range(len(names)))
            ax.set_yticklabels(names)
            ax.set_xlabel('Importance Score')
            ax.set_title(title)

            # Agregar valores en las barras
            for i, (bar, score) in enumerate(zip(bars, scores)):
                ax.text(bar.get_width() + max(scores) * 0.01, bar.get_y() + bar.get_height()/2,
                       f'{score:.3f}', ha='left', va='center', fontsize=8)

            plt.tight_layout()
            return fig

        except Exception as e:
            logger.error(f"Error creando feature importance plot: {e}")
            return None

    def plot_shap_summary(self, shap_values: np.ndarray, feature_names: Optional[List[str]] = None,
                         title: str = "SHAP Summary Plot") -> Optional[Figure]:
        """Crear summary plot de SHAP (simplificado)"""

        try:
            fig, ax = plt.subplots(figsize=(10, 6))

            if shap_values.ndim == 1:
                # Single prediction
                shap_values = shap_values.reshape(1, -1)

            # Calcular mean absolute SHAP values
            mean_abs_shap = np.abs(shap_values).mean(axis=0)

            n_features = len(mean_abs_shap)
            if feature_names and len(feature_names) >= n_features:
                labels = feature_names[:n_features]
            else:
                labels = [f"Feature {i}" for i in range(n_features)]

            # Ordenar
            sorted_indices = np.argsort(mean_abs_shap)[::-1]
            sorted_values = mean_abs_shap[sorted_indices]
            sorted_labels = [labels[i] for i in sorted_indices]

            # Plot
            ax.bar(range(len(sorted_values)), sorted_values, alpha=0.7)
            ax.set_xlabel('Features')
            ax.set_ylabel('Mean |SHAP Value|')
            ax.set_title(title)
            ax.set_xticks(range(len(sorted_labels)))
            ax.set_xticklabels(sorted_labels, rotation=45, ha='right')

            plt.tight_layout()
            return fig

        except Exception as e:
            logger.error(f"Error creando SHAP summary plot: {e}")
            return None

# ===== SISTEMA PRINCIPAL =====

class AEGISExplainableAI:
    """Sistema completo de IA Explicable"""

    def __init__(self):
        self.shap_explainer = SHAPExplainer() if SHAP_AVAILABLE else None
        self.fallback_explainer = FallbackExplainer()
        self.dl_explainer = DeepLearningExplainer()
        self.visualizer = ExplanationVisualizer()
        self.explanations_cache: Dict[str, Any] = {}

    async def explain_model_prediction(self, model: Any, input_data: Union[np.ndarray, pd.DataFrame],
                                     background_data: Optional[np.ndarray] = None,
                                     model_type: str = "auto",
                                     feature_names: Optional[List[str]] = None) -> SHAPExplanation:
        """Explicar una predicción individual"""

        logger.info("🔍 Explicando predicción del modelo...")

        start_time = time.time()

        # Preparar datos
        if isinstance(input_data, pd.DataFrame):
            input_array = input_data.values
            if feature_names is None:
                feature_names = input_data.columns.tolist()
        else:
            input_array = np.array(input_data)

        # Preparar background data
        if background_data is None:
            # Crear background sintético
            background_data = np.random.normal(0, 1, (50, input_array.shape[1]))

        try:
            # Intentar SHAP primero
            if self.shap_explainer and SHAP_AVAILABLE:
                explainer = self.shap_explainer.create_explainer(model, background_data, model_type)
                if explainer:
                    explanation = self.shap_explainer.explain_prediction(explainer, input_array)
                    explanation.processing_time = time.time() - start_time
                    return explanation

            # Fallback: permutation importance
            logger.info("Usando método alternativo (permutation importance)")
            explanation = self.fallback_explainer.explain_prediction_permutation(
                model, input_array, background_data, feature_names
            )
            explanation.processing_time = time.time() - start_time
            return explanation

        except Exception as e:
            logger.error(f"Error explicando predicción: {e}")
            return SHAPExplanation(
                method=ExplanationMethod.PERMUTATION_IMPORTANCE,
                scope=ExplanationScope.LOCAL,
                processing_time=time.time() - start_time
            )

    async def explain_model_global(self, model: Any, X_train: np.ndarray, y_train: np.ndarray,
                                 model_type: str = "auto",
                                 feature_names: Optional[List[str]] = None) -> List[FeatureImportance]:
        """Explicación global del modelo"""

        logger.info("🌍 Generando explicación global del modelo...")

        try:
            # Intentar SHAP primero
            if self.shap_explainer and SHAP_AVAILABLE:
                background_sample = X_train[np.random.choice(len(X_train), min(100, len(X_train)), replace=False)]
                explainer = self.shap_explainer.create_explainer(model, background_sample, model_type)

                if explainer:
                    importance_list = self.shap_explainer.explain_model_global(
                        explainer, background_sample, feature_names
                    )
                    if importance_list:
                        return importance_list

            # Fallback: permutation importance
            logger.info("Usando método alternativo (permutation importance)")
            importance_list = self.fallback_explainer.explain_model_permutation(
                model, X_train, y_train, feature_names
            )

            return importance_list

        except Exception as e:
            logger.error(f"Error en explicación global: {e}")
            return []

    async def explain_deep_learning_model(self, model: nn.Module, input_tensor: torch.Tensor,
                                        method: ExplanationMethod = ExplanationMethod.GRADIENT_BASED) -> SHAPExplanation:
        """Explicar modelo de deep learning"""

        logger.info(f"🧠 Explicando modelo deep learning usando {method.value}")

        if method == ExplanationMethod.GRADIENT_BASED:
            return self.dl_explainer.explain_with_gradients(model, input_tensor)
        elif method == ExplanationMethod.DEEP_TAYLOR:
            return self.dl_explainer.explain_with_deep_taylor(model, input_tensor)
        else:
            # Fallback
            return self.dl_explainer.explain_with_gradients(model, input_tensor)

    def create_model_explanation(self, model: Any, model_name: str, X_train: np.ndarray,
                               y_train: np.ndarray, model_type: str = "auto",
                               feature_names: Optional[List[str]] = None) -> ModelExplanation:
        """Crear explicación completa del modelo"""

        logger.info(f"📋 Creando explicación completa del modelo {model_name}")

        explanation = ModelExplanation(
            model_name=model_name,
            model_type=model_type
        )

        # Explicación global
        # Note: We're not using asyncio.run here to avoid the "cannot be called from a running event loop" error
        # In a real implementation, this would need to be handled differently
        try:
            # This is a simplified version that doesn't use async
            if hasattr(self.fallback_explainer, 'explain_model_permutation'):
                explanation.global_explanations = self.fallback_explainer.explain_model_permutation(
                    model, X_train, y_train, feature_names
                )
        except Exception as e:
            logger.error(f"Error generating global explanations: {e}")

        # Insights del modelo
        explanation.model_insights = self._generate_model_insights(
            explanation.global_explanations, model_type
        )

        # Análisis de fairness (simplificado)
        explanation.fairness_analysis = self._analyze_model_fairness(X_train, y_train)

        # Análisis de robustness (simplificado)
        explanation.robustness_analysis = self._analyze_model_robustness(model, X_train)

        return explanation

    def _generate_model_insights(self, global_explanations: List[FeatureImportance],
                               model_type: str) -> List[str]:
        """Generar insights automáticos sobre el modelo"""

        insights = []

        if not global_explanations:
            return ["No se pudieron generar explicaciones para este modelo"]

        # Top features
        top_feature = global_explanations[0]
        insights.append(f"🔝 La feature más importante es '{top_feature.feature_name}' "
                       f"con importancia {top_feature.importance_score:.3f}")
        # Número de features importantes
        important_features = [f for f in global_explanations if f.importance_score > 0.01]
        insights.append(f"📊 {len(important_features)} features tienen importancia significativa (>0.01)")

        # Insights por tipo de modelo
        if model_type == "tree":
            insights.append("🌳 Modelo basado en árboles - explicable por naturaleza")
        elif model_type == "linear":
            insights.append("📈 Modelo lineal - coeficientes directamente interpretables")
        elif model_type == "deep":
            insights.append("🧠 Modelo deep learning - requiere técnicas especializadas para explicación")

        # Distribución de importancia
        scores = [f.importance_score for f in global_explanations]
        if len(scores) > 1:
            score_std = np.std(scores)
            if score_std > np.mean(scores):
                insights.append("📊 Importancia muy variable entre features - modelo captura patrones complejos")
            else:
                insights.append("📊 Importancia equilibrada entre features - modelo robusto")

        return insights

    def _analyze_model_fairness(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Análisis simplificado de fairness"""

        # En producción, implementar análisis más sofisticados
        return {
            "disparate_impact": "not_analyzed",
            "equal_opportunity": "not_analyzed",
            "recommendations": ["Implementar análisis de fairness completo con datos sensibles"]
        }

    def _analyze_model_robustness(self, model: Any, X: np.ndarray) -> Dict[str, Any]:
        """Análisis simplificado de robustness"""

        # En producción, implementar tests de robustness más completos
        return {
            "adversarial_robustness": "not_analyzed",
            "noise_sensitivity": "not_analyzed",
            "recommendations": ["Implementar tests de robustness con datos perturbados"]
        }

    def visualize_explanation(self, explanation: Union[SHAPExplanation, ModelExplanation],
                            feature_names: Optional[List[str]] = None) -> Optional[Figure]:
        """Crear visualización de explicación"""

        if isinstance(explanation, SHAPExplanation):
            if explanation.scope == ExplanationScope.LOCAL:
                return self.visualizer.plot_shap_waterfall(explanation, feature_names)
            else:
                if explanation.shap_values is not None:
                    return self.visualizer.plot_shap_summary(explanation.shap_values, feature_names)

        elif isinstance(explanation, ModelExplanation):
            return self.visualizer.plot_feature_importance(explanation.global_explanations)

        return None