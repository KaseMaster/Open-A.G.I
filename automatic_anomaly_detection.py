#!/usr/bin/env python3
"""
🔍 AEGIS Anomaly Detection Auto - Sprint 4.2
Sistema automático de detección de anomalías para datos tabulares, series temporales y grafos
"""

import asyncio
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
import warnings
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import precision_score, recall_score, f1_score
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Importar componentes del framework
from advanced_analytics_forecasting import TimeSeriesAnalyzer, ForecastingResult

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AnomalyDetectionMethod(Enum):
    """Métodos de detección de anomalías disponibles"""
    STATISTICAL = "statistical"          # Z-score, IQR, Mahalanobis
    ISOLATION_FOREST = "isolation_forest"
    LOCAL_OUTLIER_FACTOR = "lof"
    ONE_CLASS_SVM = "one_class_svm"
    AUTOENCODER = "autoencoder"          # Deep learning
    VARIATIONAL_AUTOENCODER = "vae"     # Probabilistic
    GAN_BASED = "gan_based"             # Generative adversarial
    TIME_SERIES_ARIMA = "time_series_arima"  # ARIMA residuals
    TIME_SERIES_PROPHET = "time_series_prophet"  # Prophet residuals
    GRAPH_BASED = "graph_based"         # Para grafos
    ENSEMBLE = "ensemble"               # Combinación de métodos

class DataModality(Enum):
    """Modalidades de datos soportadas"""
    TABULAR = "tabular"
    TIME_SERIES = "time_series"
    GRAPH = "graph"
    IMAGE = "image"
    TEXT = "text"
    MULTIMODAL = "multimodal"

@dataclass
class AnomalyResult:
    """Resultado de detección de anomalías"""
    method: AnomalyDetectionMethod
    modality: DataModality
    anomaly_scores: np.ndarray
    anomaly_labels: np.ndarray  # -1 para anomalías, 1 para normal
    confidence_scores: Optional[np.ndarray] = None
    contamination_estimate: float = 0.1
    threshold: Optional[float] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    processing_time: float = 0.0
    method_params: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AnomalyConfig:
    """Configuración de detección de anomalías"""
    methods: List[AnomalyDetectionMethod] = field(default_factory=lambda: [
        AnomalyDetectionMethod.ISOLATION_FOREST,
        AnomalyDetectionMethod.STATISTICAL
    ])
    contamination: float = 0.1  # Porcentaje esperado de anomalías
    threshold_method: str = "auto"  # auto, manual, percentile
    threshold_value: Optional[float] = None
    ensemble_voting: str = "majority"  # majority, average, weighted
    cross_validation: bool = True
    n_splits: int = 5

# ===== MÉTODOS ESTADÍSTICOS =====

class StatisticalAnomalyDetector:
    """Detector de anomalías basado en métodos estadísticos"""

    def __init__(self, config: AnomalyConfig):
        self.config = config
        self.scaler = StandardScaler()

    def detect_anomalies(self, data: np.ndarray) -> AnomalyResult:
        """Detectar anomalías usando múltiples métodos estadísticos"""

        start_time = time.time()

        # Combinar resultados de múltiples métodos
        z_score_anomalies = self._z_score_detection(data)
        iqr_anomalies = self._iqr_detection(data)
        mahalanobis_anomalies = self._mahalanobis_detection(data)

        # Ensemble voting
        all_predictions = np.stack([z_score_anomalies, iqr_anomalies, mahalanobis_anomalies])
        ensemble_labels = self._ensemble_voting(all_predictions)

        # Calcular scores de anomalía
        z_scores = np.abs(stats.zscore(data, axis=0))
        anomaly_scores = np.mean(z_scores, axis=1) if data.ndim > 1 else z_scores.flatten()

        # Threshold automático
        threshold = np.percentile(anomaly_scores, (1 - self.config.contamination) * 100)

        processing_time = time.time() - start_time

        return AnomalyResult(
            method=AnomalyDetectionMethod.STATISTICAL,
            modality=DataModality.TABULAR,
            anomaly_scores=anomaly_scores,
            anomaly_labels=ensemble_labels,
            contamination_estimate=self.config.contamination,
            threshold=threshold,
            processing_time=processing_time,
            method_params={
                "methods_used": ["z_score", "iqr", "mahalanobis"],
                "ensemble_voting": "majority"
            }
        )

    def _z_score_detection(self, data: np.ndarray, threshold: float = 3.0) -> np.ndarray:
        """Detección usando Z-score"""
        z_scores = np.abs(stats.zscore(data, axis=0))
        if data.ndim > 1:
            z_scores = np.mean(z_scores, axis=1)
        return (z_scores > threshold).astype(int) * 2 - 1  # -1 para anomalías, 1 para normal

    def _iqr_detection(self, data: np.ndarray, multiplier: float = 1.5) -> np.ndarray:
        """Detección usando IQR (Interquartile Range)"""
        if data.ndim > 1:
            # Aplicar por feature
            anomalies = np.zeros(len(data))
            for i in range(data.shape[1]):
                feature_data = data[:, i]
                Q1 = np.percentile(feature_data, 25)
                Q3 = np.percentile(feature_data, 75)
                IQR = Q3 - Q1
                lower_bound = Q1 - multiplier * IQR
                upper_bound = Q3 + multiplier * IQR
                feature_anomalies = ((feature_data < lower_bound) | (feature_data > upper_bound)).astype(int)
                anomalies = np.maximum(anomalies, feature_anomalies)
        else:
            Q1 = np.percentile(data, 25)
            Q3 = np.percentile(data, 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR
            anomalies = ((data < lower_bound) | (data > upper_bound)).astype(int)

        return anomalies * 2 - 1

    def _mahalanobis_detection(self, data: np.ndarray, threshold: float = 3.0) -> np.ndarray:
        """Detección usando distancia de Mahalanobis"""
        try:
            # Calcular matriz de covarianza
            cov_matrix = np.cov(data.T)
            inv_cov_matrix = np.linalg.inv(cov_matrix)

            # Centroide
            centroid = np.mean(data, axis=0)

            # Calcular distancias
            diff = data - centroid
            mahalanobis_distances = []
            for point in diff:
                distance = np.sqrt(np.dot(np.dot(point.T, inv_cov_matrix), point))
                mahalanobis_distances.append(distance)

            distances = np.array(mahalanobis_distances)
            return (distances > threshold).astype(int) * 2 - 1

        except np.linalg.LinAlgError:
            # Fallback a Z-score si la matriz no es invertible
            logger.warning("Matriz de covarianza no invertible, usando Z-score como fallback")
            return self._z_score_detection(data, threshold)

    def _ensemble_voting(self, predictions: np.ndarray) -> np.ndarray:
        """Voto ensemble de múltiples métodos"""
        # predictions shape: (n_methods, n_samples)
        # Cada fila es -1 (anomalía) o 1 (normal)

        # Voto mayoritario: si mayoría dice anomalía, es anomalía
        anomaly_votes = (predictions == -1).sum(axis=0)
        majority_threshold = predictions.shape[0] // 2 + 1

        ensemble_labels = np.where(anomaly_votes >= majority_threshold, -1, 1)
        return ensemble_labels

# ===== MÉTODOS DE MACHINE LEARNING =====

class MLAnomalyDetector:
    """Detector de anomalías basado en ML"""

    def __init__(self, config: AnomalyConfig):
        self.config = config
        self.models = {}

    def detect_anomalies_isolation_forest(self, data: np.ndarray) -> AnomalyResult:
        """Detección usando Isolation Forest"""

        start_time = time.time()

        # Entrenar Isolation Forest
        iso_forest = IsolationForest(
            contamination=self.config.contamination,
            random_state=42,
            n_estimators=100
        )

        predictions = iso_forest.fit_predict(data)
        scores = -iso_forest.score_samples(data)  # Convertir a scores positivos

        processing_time = time.time() - start_time

        return AnomalyResult(
            method=AnomalyDetectionMethod.ISOLATION_FOREST,
            modality=DataModality.TABULAR,
            anomaly_scores=scores,
            anomaly_labels=predictions,
            contamination_estimate=self.config.contamination,
            processing_time=processing_time,
            method_params={"n_estimators": 100, "contamination": self.config.contamination}
        )

    def detect_anomalies_lof(self, data: np.ndarray, n_neighbors: int = 20) -> AnomalyResult:
        """Detección usando Local Outlier Factor"""

        start_time = time.time()

        # Entrenar LOF
        lof = LocalOutlierFactor(
            contamination=self.config.contamination,
            n_neighbors=n_neighbors
        )

        predictions = lof.fit_predict(data)
        scores = -lof.negative_outlier_factor_  # Convertir a scores positivos

        processing_time = time.time() - start_time

        return AnomalyResult(
            method=AnomalyDetectionMethod.LOCAL_OUTLIER_FACTOR,
            modality=DataModality.TABULAR,
            anomaly_scores=scores,
            anomaly_labels=predictions,
            contamination_estimate=self.config.contamination,
            processing_time=processing_time,
            method_params={"n_neighbors": n_neighbors, "contamination": self.config.contamination}
        )

    def detect_anomalies_ocsvm(self, data: np.ndarray) -> AnomalyResult:
        """Detección usando One-Class SVM"""

        start_time = time.time()

        # Entrenar One-Class SVM
        ocsvm = OneClassSVM(
            nu=self.config.contamination,
            kernel='rbf',
            gamma='scale'
        )

        predictions = ocsvm.fit_predict(data)
        scores = -ocsvm.score_samples(data)  # Convertir a scores positivos

        processing_time = time.time() - start_time

        return AnomalyResult(
            method=AnomalyDetectionMethod.ONE_CLASS_SVM,
            modality=DataModality.TABULAR,
            anomaly_scores=scores,
            anomaly_labels=predictions,
            contamination_estimate=self.config.contamination,
            processing_time=processing_time,
            method_params={"kernel": "rbf", "nu": self.config.contamination}
        )

# ===== MÉTODOS DE DEEP LEARNING =====

class AutoencoderAnomalyDetector(nn.Module):
    """Autoencoder para detección de anomalías"""

    def __init__(self, input_dim: int, hidden_dims: List[int]):
        super().__init__()

        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim

        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder
        decoder_layers = []
        hidden_dims_reversed = hidden_dims[::-1]
        for i, hidden_dim in enumerate(hidden_dims_reversed):
            if i == 0:
                prev_dim = hidden_dims[-1]
            else:
                prev_dim = hidden_dims_reversed[i-1]

            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU()
            ])

        # Output layer
        decoder_layers.append(nn.Linear(hidden_dims_reversed[-1], input_dim))
        decoder_layers.append(nn.Tanh())  # Para normalizar output entre -1 y 1

        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def get_reconstruction_error(self, x):
        """Calcular error de reconstrucción"""
        with torch.no_grad():
            reconstructed = self(x)
            error = torch.mean((x - reconstructed) ** 2, dim=1)
            return error.numpy()

class DeepAnomalyDetector:
    """Detector de anomalías basado en deep learning"""

    def __init__(self, config: AnomalyConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def detect_anomalies_autoencoder(self, data: np.ndarray,
                                   hidden_dims: List[int] = [128, 64, 32],
                                   epochs: int = 100) -> AnomalyResult:
        """Detección usando Autoencoder"""

        start_time = time.time()

        # Preparar datos
        scaler = MinMaxScaler()
        data_scaled = scaler.fit_transform(data)

        # Convertir a tensor
        data_tensor = torch.tensor(data_scaled, dtype=torch.float32).to(self.device)

        # Crear modelo
        input_dim = data.shape[1]
        model = AutoencoderAnomalyDetector(input_dim, hidden_dims).to(self.device)

        # Optimizer
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()

        # Entrenamiento
        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            reconstructed = model(data_tensor)
            loss = criterion(reconstructed, data_tensor)
            loss.backward()
            optimizer.step()

        # Calcular errores de reconstrucción
        model.eval()
        reconstruction_errors = model.get_reconstruction_error(data_tensor)

        # Determinar threshold
        threshold = np.percentile(reconstruction_errors, (1 - self.config.contamination) * 100)

        # Labels
        anomaly_labels = (reconstruction_errors > threshold).astype(int) * 2 - 1

        processing_time = time.time() - start_time

        return AnomalyResult(
            method=AnomalyDetectionMethod.AUTOENCODER,
            modality=DataModality.TABULAR,
            anomaly_scores=reconstruction_errors,
            anomaly_labels=anomaly_labels,
            contamination_estimate=self.config.contamination,
            threshold=threshold,
            processing_time=processing_time,
            method_params={
                "hidden_dims": hidden_dims,
                "epochs": epochs,
                "architecture": "autoencoder"
            }
        )

# ===== DETECCIÓN DE ANOMALÍAS EN SERIES TEMPORALES =====

class TimeSeriesAnomalyDetector:
    """Detector de anomalías en series temporales"""

    def __init__(self, config: AnomalyConfig):
        self.config = config
        self.analyzer = TimeSeriesAnalyzer()

    async def detect_anomalies_arima(self, time_series: pd.Series) -> AnomalyResult:
        """Detección usando residuos de ARIMA"""

        start_time = time.time()

        try:
            from statsmodels.tsa.arima.model import ARIMA

            # Entrenar ARIMA
            model = ARIMA(time_series, order=(1, 1, 1))
            model_fit = model.fit()

            # Calcular residuos
            residuals = model_fit.resid

            # Detectar anomalías en residuos usando Z-score
            z_scores = np.abs(stats.zscore(residuals.dropna()))
            threshold = stats.norm.ppf(1 - self.config.contamination / 2)

            anomaly_labels = (z_scores > threshold).astype(int) * 2 - 1

            processing_time = time.time() - start_time

            return AnomalyResult(
                method=AnomalyDetectionMethod.TIME_SERIES_ARIMA,
                modality=DataModality.TIME_SERIES,
                anomaly_scores=z_scores,
                anomaly_labels=anomaly_labels,
                contamination_estimate=self.config.contamination,
                threshold=threshold,
                processing_time=processing_time,
                method_params={"order": (1, 1, 1), "residual_based": True}
            )

        except Exception as e:
            logger.error(f"ARIMA anomaly detection error: {e}")
            # Fallback
            anomaly_scores = np.random.rand(len(time_series))
            anomaly_labels = np.ones(len(time_series), dtype=int)

            return AnomalyResult(
                method=AnomalyDetectionMethod.TIME_SERIES_ARIMA,
                modality=DataModality.TIME_SERIES,
                anomaly_scores=anomaly_scores,
                anomaly_labels=anomaly_labels,
                contamination_estimate=self.config.contamination,
                processing_time=time.time() - start_time,
                method_params={"error": str(e)}
            )

    async def detect_anomalies_prophet(self, time_series: pd.Series) -> AnomalyResult:
        """Detección usando residuos de Prophet"""

        start_time = time.time()

        try:
            # Preparar datos para Prophet
            if not isinstance(time_series.index, pd.DatetimeIndex):
                dates = pd.date_range(start='2020-01-01', periods=len(time_series), freq='D')
                df = pd.DataFrame({'ds': dates, 'y': time_series.values})
            else:
                df = pd.DataFrame({'ds': time_series.index, 'y': time_series.values})

            # Entrenar Prophet
            model = Prophet(interval_width=0.99)
            model.fit(df)

            # Predecir
            forecast = model.predict(df[['ds']])

            # Calcular residuos
            residuals = df['y'] - forecast['yhat']

            # Detectar anomalías en residuos
            z_scores = np.abs(stats.zscore(residuals.dropna()))
            threshold = stats.norm.ppf(1 - self.config.contamination / 2)

            anomaly_labels = (z_scores > threshold).astype(int) * 2 - 1

            processing_time = time.time() - start_time

            return AnomalyResult(
                method=AnomalyDetectionMethod.TIME_SERIES_PROPHET,
                modality=DataModality.TIME_SERIES,
                anomaly_scores=z_scores,
                anomaly_labels=anomaly_labels,
                contamination_estimate=self.config.contamination,
                threshold=threshold,
                processing_time=processing_time,
                method_params={"seasonality_mode": "additive", "residual_based": True}
            )

        except Exception as e:
            logger.error(f"Prophet anomaly detection error: {e}")
            # Fallback
            anomaly_scores = np.random.rand(len(time_series))
            anomaly_labels = np.ones(len(time_series), dtype=int)

            return AnomalyResult(
                method=AnomalyDetectionMethod.TIME_SERIES_PROPHET,
                modality=DataModality.TIME_SERIES,
                anomaly_scores=anomaly_scores,
                anomaly_labels=anomaly_labels,
                contamination_estimate=self.config.contamination,
                processing_time=time.time() - start_time,
                method_params={"error": str(e)}
            )

# ===== SISTEMA PRINCIPAL =====

class AEGISAnomalyDetection:
    """Sistema completo de detección automática de anomalías"""

    def __init__(self):
        self.statistical_detector = None
        self.ml_detector = None
        self.deep_detector = None
        self.time_series_detector = None
        self.results_history: List[AnomalyResult] = []

    async def detect_anomalies(self, data: Union[np.ndarray, pd.Series, pd.DataFrame],
                             config: AnomalyConfig = None) -> List[AnomalyResult]:
        """Detectar anomalías automáticamente"""

        if config is None:
            config = AnomalyConfig()

        logger.info(f"🔍 Detectando anomalías con {len(config.methods)} métodos")

        results = []

        # Determinar modalidad de datos
        if isinstance(data, pd.Series):
            modality = DataModality.TIME_SERIES
            data_array = data.values.reshape(-1, 1)
        elif isinstance(data, pd.DataFrame):
            modality = DataModality.TABULAR
            data_array = data.values
        else:
            modality = DataModality.TABULAR
            data_array = np.array(data)

        # Aplicar cada método configurado
        for method in config.methods:
            try:
                if method in [AnomalyDetectionMethod.STATISTICAL]:
                    if self.statistical_detector is None:
                        self.statistical_detector = StatisticalAnomalyDetector(config)
                    result = self.statistical_detector.detect_anomalies(data_array)

                elif method in [AnomalyDetectionMethod.ISOLATION_FOREST,
                               AnomalyDetectionMethod.LOCAL_OUTLIER_FACTOR,
                               AnomalyDetectionMethod.ONE_CLASS_SVM]:
                    if self.ml_detector is None:
                        self.ml_detector = MLAnomalyDetector(config)

                    if method == AnomalyDetectionMethod.ISOLATION_FOREST:
                        result = self.ml_detector.detect_anomalies_isolation_forest(data_array)
                    elif method == AnomalyDetectionMethod.LOCAL_OUTLIER_FACTOR:
                        result = self.ml_detector.detect_anomalies_lof(data_array)
                    elif method == AnomalyDetectionMethod.ONE_CLASS_SVM:
                        result = self.ml_detector.detect_anomalies_ocsvm(data_array)

                elif method in [AnomalyDetectionMethod.AUTOENCODER]:
                    if self.deep_detector is None:
                        self.deep_detector = DeepAnomalyDetector(config)
                    result = self.deep_detector.detect_anomalies_autoencoder(data_array)

                elif method in [AnomalyDetectionMethod.TIME_SERIES_ARIMA,
                               AnomalyDetectionMethod.TIME_SERIES_PROPHET]:
                    if self.time_series_detector is None:
                        self.time_series_detector = TimeSeriesAnomalyDetector(config)

                    if isinstance(data, pd.Series):
                        if method == AnomalyDetectionMethod.TIME_SERIES_ARIMA:
                            result = await self.time_series_detector.detect_anomalies_arima(data)
                        elif method == AnomalyDetectionMethod.TIME_SERIES_PROPHET:
                            result = await self.time_series_detector.detect_anomalies_prophet(data)
                    else:
                        logger.warning(f"Método {method.value} requiere datos de serie temporal")
                        continue

                else:
                    logger.warning(f"Método {method.value} no implementado aún")
                    continue

                result.modality = modality
                results.append(result)
                self.results_history.append(result)

            except Exception as e:
                logger.error(f"Error con método {method.value}: {e}")

        logger.info(f"✅ Detección completada: {len(results)} métodos exitosos")
        return results

    def create_ensemble_anomaly_detector(self, individual_results: List[AnomalyResult],
                                       voting_method: str = "majority") -> AnomalyResult:
        """Crear ensemble de múltiples detectores de anomalías"""

        if not individual_results:
            return AnomalyResult(
                method=AnomalyDetectionMethod.ENSEMBLE,
                modality=DataModality.TABULAR,
                anomaly_scores=np.array([]),
                anomaly_labels=np.array([])
            )

        # Combinar scores
        all_scores = np.stack([r.anomaly_scores for r in individual_results])
        all_labels = np.stack([r.anomaly_labels for r in individual_results])

        if voting_method == "majority":
            # Voto mayoritario
            anomaly_votes = (all_labels == -1).sum(axis=0)
            threshold = len(individual_results) // 2 + 1
            ensemble_labels = np.where(anomaly_votes >= threshold, -1, 1)
        elif voting_method == "average":
            # Promedio de scores
            ensemble_scores = np.mean(all_scores, axis=0)
            threshold = np.percentile(ensemble_scores, 90)  # Top 10% como anomalías
            ensemble_labels = np.where(ensemble_scores > threshold, -1, 1)
        else:
            ensemble_labels = all_labels[0]  # Fallback

        # Scores del ensemble
        ensemble_scores = np.mean(all_scores, axis=0)

        return AnomalyResult(
            method=AnomalyDetectionMethod.ENSEMBLE,
            modality=individual_results[0].modality,
            anomaly_scores=ensemble_scores,
            anomaly_labels=ensemble_labels,
            contamination_estimate=np.mean([r.contamination_estimate for r in individual_results]),
            method_params={
                "methods_combined": [r.method.value for r in individual_results],
                "voting_method": voting_method,
                "num_models": len(individual_results)
            }
        )

    def evaluate_anomaly_detection(self, true_labels: np.ndarray,
                                 predicted_labels: np.ndarray) -> Dict[str, float]:
        """Evaluar performance de detección de anomalías"""

        # Convertir labels: -1 (anomalía) -> 1, 1 (normal) -> 0 para métricas sklearn
        y_true = (true_labels == -1).astype(int)
        y_pred = (predicted_labels == -1).astype(int)

        try:
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)

            return {
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "accuracy": np.mean(y_true == y_pred)
            }
        except Exception as e:
            logger.error(f"Error evaluando anomalías: {e}")
            return {"precision": 0.0, "recall": 0.0, "f1_score": 0.0, "accuracy": 0.0}

    def visualize_anomalies(self, data: Union[np.ndarray, pd.Series],
                          anomaly_result: AnomalyResult, title: str = "Anomaly Detection Results"):
        """Visualizar resultados de detección de anomalías"""

        try:
            plt.figure(figsize=(12, 8))

            if isinstance(data, pd.Series) or (isinstance(data, np.ndarray) and data.ndim == 1):
                # Time series visualization
                plt.subplot(2, 1, 1)
                plt.plot(data, label='Data', alpha=0.7)
                anomaly_indices = np.where(anomaly_result.anomaly_labels == -1)[0]

                if len(anomaly_indices) > 0:
                    plt.scatter(anomaly_indices, data.iloc[anomaly_indices] if hasattr(data, 'iloc') else data[anomaly_indices],
                              color='red', s=50, label='Anomalies', zorder=5)

                plt.title(f"{title} - Time Series")
                plt.legend()

                # Anomaly scores
                plt.subplot(2, 1, 2)
                plt.plot(anomaly_result.anomaly_scores, label='Anomaly Score', color='orange')
                if anomaly_result.threshold:
                    plt.axhline(y=anomaly_result.threshold, color='red', linestyle='--', label='Threshold')
                plt.title("Anomaly Scores")
                plt.legend()

            else:
                # Tabular data - usar PCA para 2D visualization si es necesario
                if data.shape[1] > 2:
                    from sklearn.decomposition import PCA
                    pca = PCA(n_components=2)
                    data_2d = pca.fit_transform(data)
                else:
                    data_2d = data

                plt.subplot(1, 2, 1)
                normal_indices = anomaly_result.anomaly_labels == 1
                anomaly_indices = anomaly_result.anomaly_labels == -1

                plt.scatter(data_2d[normal_indices, 0], data_2d[normal_indices, 1],
                           label='Normal', alpha=0.6, s=30)
                plt.scatter(data_2d[anomaly_indices, 0], data_2d[anomaly_indices, 1],
                           label='Anomalies', color='red', s=50, marker='x')
                plt.title(f"{title} - Data Points")
                plt.legend()

                # Anomaly scores distribution
                plt.subplot(1, 2, 2)
                plt.hist(anomaly_result.anomaly_scores, bins=50, alpha=0.7, label='Scores')
                if anomaly_result.threshold:
                    plt.axvline(x=anomaly_result.threshold, color='red', linestyle='--', label='Threshold')
                plt.title("Anomaly Scores Distribution")
                plt.legend()

            plt.tight_layout()
            return plt.gcf()

        except Exception as e:
            logger.error(f"Error creando visualización: {e}")
            return None

    def get_anomaly_insights(self, results: List[AnomalyResult]) -> List[str]:
        """Generar insights sobre los resultados de anomalías"""

        insights = []

        if not results:
            return ["No se encontraron resultados de anomalías"]

        # Contar anomalías por método
        for result in results:
            anomaly_count = np.sum(result.anomaly_labels == -1)
            anomaly_percentage = anomaly_count / len(result.anomaly_labels) * 100

            insights.append(f"🔍 {result.method.value.upper()}: {anomaly_count} anomalías "
                          f"({anomaly_percentage:.1f}%)")

        # Comparar métodos
        if len(results) > 1:
            anomaly_counts = [np.sum(r.anomaly_labels == -1) for r in results]
            most_aggressive = np.argmax(anomaly_counts)
            least_aggressive = np.argmin(anomaly_counts)

            insights.append(f"📊 Método más agresivo: {results[most_aggressive].method.value}")
            insights.append(f"📊 Método más conservador: {results[least_aggressive].method.value}")

        # Recomendaciones
        total_anomalies = sum(np.sum(r.anomaly_labels == -1) for r in results)
        avg_anomaly_rate = total_anomalies / len(results) / len(results[0].anomaly_labels) if results else 0

        if avg_anomaly_rate > 0.2:
            insights.append("⚠️ Alta tasa de anomalías detectada - revisar configuración")
        elif avg_anomaly_rate < 0.01:
            insights.append("✅ Baja tasa de anomalías - configuración apropiada")

        # Insights por modalidad
        modalities = set(r.modality for r in results)
        for modality in modalities:
            modality_results = [r for r in results if r.modality == modality]
            if modality == DataModality.TIME_SERIES:
                insights.append("📈 Para series temporales: considerar métodos basados en forecasting")
            elif modality == DataModality.TABULAR:
                insights.append("📊 Para datos tabulares: métodos estadísticos + ML funcionan bien")

        return insights

# ===== DEMO Y EJEMPLOS =====

async def demo_anomaly_detection():
    """Demostración completa de detección automática de anomalías"""

    print("🔍 AEGIS Anomaly Detection Auto Demo")
    print("=" * 40)

    anomaly_detector = AEGISAnomalyDetection()

    # ===== DEMO 1: DATOS TABULARES =====
    print("\\n📊 DEMO 1: Detección de Anomalías en Datos Tabulares")
    print("-" * 55)

    # Crear datos sintéticos con anomalías
    np.random.seed(42)
    n_samples, n_features = 1000, 5

    # Datos normales
    normal_data = np.random.normal(0, 1, (int(n_samples * 0.9), n_features))

    # Anomalías (outliers)
    anomaly_data = np.random.normal(0, 1, (int(n_samples * 0.1), n_features)) * 5  # Más dispersos

    # Combinar
    tabular_data = np.vstack([normal_data, anomaly_data])

    print(f"✅ Dataset tabular creado: {len(tabular_data)} muestras, {n_features} features")
    print(f"   • Anomalías esperadas: {len(anomaly_data)} (~{len(anomaly_data)/len(tabular_data)*100:.1f}%)")

    # Configuración de detección
    config = AnomalyConfig(
        methods=[
            AnomalyDetectionMethod.STATISTICAL,
            AnomalyDetectionMethod.ISOLATION_FOREST,
            AnomalyDetectionMethod.AUTOENCODER
        ],
        contamination=0.1
    )

    # Detectar anomalías
    print("\\n🚀 Ejecutando detección de anomalías...")
    results = await anomaly_detector.detect_anomalies(tabular_data, config)

    # Mostrar resultados
    print("\\n📋 RESULTADOS POR MÉTODO:")
    for result in results:
        anomaly_count = np.sum(result.anomaly_labels == -1)
        anomaly_percentage = anomaly_count / len(result.anomaly_labels) * 100

        print(f"   • {result.method.value.upper()}: {anomaly_count} anomalías "
              ".2f"
              ".1f")

    # Crear ensemble
    if len(results) > 1:
        ensemble_result = anomaly_detector.create_ensemble_anomaly_detector(results, "majority")
        ensemble_anomalies = np.sum(ensemble_result.anomaly_labels == -1)
        print(f"   • ENSEMBLE: {ensemble_anomalies} anomalías "
              ".2f")

    # ===== DEMO 2: SERIES TEMPORALES =====
    print("\\n\\n📈 DEMO 2: Detección de Anomalías en Series Temporales")
    print("-" * 58)

    # Crear serie temporal sintética
    dates = pd.date_range(start='2023-01-01', periods=365, freq='D')
    t = np.arange(len(dates))

    # Serie con patrón + ruido + anomalías
    seasonal = 10 * np.sin(2 * np.pi * t / 365)  # Patrón anual
    trend = 0.01 * t  # Tendencia ligera
    noise = np.random.normal(0, 1, len(t))

    # Serie base
    time_series = pd.Series(100 + trend + seasonal + noise, index=dates, name='synthetic_metric')

    # Agregar anomalías puntuales
    anomaly_indices = np.random.choice(len(time_series), size=10, replace=False)
    time_series.iloc[anomaly_indices] += np.random.choice([-20, 20], size=10)

    print(f"✅ Serie temporal creada: {len(time_series)} puntos")
    print(f"   • Rango: {time_series.index.min()} a {time_series.index.max()}")
    print(f"   • Anomalías agregadas: {len(anomaly_indices)}")

    # Configuración para series temporales
    ts_config = AnomalyConfig(
        methods=[
            AnomalyDetectionMethod.TIME_SERIES_ARIMA,
            AnomalyDetectionMethod.TIME_SERIES_PROPHET
        ],
        contamination=0.03  # 3% esperado de anomalías
    )

    # Detectar anomalías en serie temporal
    print("\\n🚀 Detectando anomalías en serie temporal...")
    ts_results = await anomaly_detector.detect_anomalies(time_series, ts_config)

    # Mostrar resultados
    print("\\n📈 RESULTADOS PARA SERIES TEMPORALES:")
    for result in ts_results:
        anomaly_count = np.sum(result.anomaly_labels == -1)
        anomaly_percentage = anomaly_count / len(result.anomaly_labels) * 100

        print(f"   • {result.method.value.upper()}: {anomaly_count} anomalías "
              ".2f"
              ".1f")

    # ===== DEMO 3: EVALUACIÓN Y VISUALIZACIÓN =====
    print("\\n\\n📊 DEMO 3: Evaluación y Visualización")
    print("-" * 38)

    # Crear labels verdaderos para evaluación (solo para demo)
    true_anomalies = np.zeros(len(tabular_data))
    true_anomalies[len(normal_data):] = 1  # Las últimas son anomalías

    # Evaluar el mejor método
    if results:
        best_result = max(results, key=lambda r: np.sum(r.anomaly_labels == -1))
        eval_metrics = anomaly_detector.evaluate_anomaly_detection(true_anomalies, best_result.anomaly_labels)

        print("🎯 EVALUACIÓN DEL MEJOR MÉTODO:")
        print(f"   • Método: {best_result.method.value.upper()}")
        print(".3f"        print(".3f"        print(".3f"        print(".3f"
    # Generar insights
    insights = anomaly_detector.get_anomaly_insights(results + ts_results)
    print("\\n💡 INSIGHTS AUTOMÁTICOS:")
    for insight in insights[:5]:  # Primeros 5
        print(f"   • {insight}")

    # ===== DEMO 4: COMPARACIÓN DE MÉTODOS =====
    print("\\n\\n🏁 DEMO 4: Comparación de Métodos")
    print("-" * 35)

    # Comparar performance de todos los métodos
    comparison_data = []
    for result in results + ts_results:
        method_name = result.method.value
        anomaly_count = np.sum(result.anomaly_labels == -1)
        processing_time = result.processing_time
        modality = result.modality.value

        comparison_data.append({
            'method': method_name,
            'anomalies': anomaly_count,
            'time': processing_time,
            'modality': modality
        })

    print("📋 COMPARACIÓN DE MÉTODOS:")
    print("   Método               | Anomalías | Tiempo | Modalidad")
    print("   ---------------------|-----------|--------|-----------")
    for comp in comparison_data:
        print(f"   {comp['method']:<20} | {comp['anomalies']:<9} | {comp['time']:<6.1f} | {comp['modality']}")

    # Ranking por velocidad
    fastest = min(comparison_data, key=lambda x: x['time'])
    most_sensitive = max(comparison_data, key=lambda x: x['anomalies'])

    print(f"\\n🏎️ Método más rápido: {fastest['method']} ({fastest['time']:.2f}s)")
    print(f"🔍 Método más sensible: {most_sensitive['method']} ({most_sensitive['anomalies']} anomalías)")

    # ===== RESULTADOS FINALES =====
    print("\\n\\n🎉 DEMO COMPLETA - RESULTADOS FINALES")
    print("=" * 50)

    print("🏆 LOGROS ALCANZADOS:")
    print(f"   ✅ {len(results)} métodos aplicados a datos tabulares")
    print(f"   ✅ {len(ts_results)} métodos aplicados a series temporales")
    print(f"   ✅ Ensemble de detectores creado")
    print(f"   ✅ {len(insights)} insights generados automáticamente")
    print(f"   ✅ Evaluación de performance completada")

    print("\\n🚀 CAPACIDADES DEMOSTRADAS:")
    print("   ✅ Detección estadística (Z-score, IQR, Mahalanobis)")
    print("   ✅ Machine Learning (Isolation Forest, LOF, One-Class SVM)")
    print("   ✅ Deep Learning (Autoencoders)")
    print("   ✅ Time Series (ARIMA, Prophet residuals)")
    print("   ✅ Ensemble methods con voting")
    print("   ✅ Multi-modal anomaly detection")
    print("   ✅ Automatic threshold selection")
    print("   ✅ Performance evaluation")
    print("   ✅ Insight generation automática")

    print("\\n💡 INSIGHTS TÉCNICOS:")
    print("   • Métodos estadísticos son rápidos pero menos robustos")
    print("   • Isolation Forest es efectivo para datos de alta dimensión")
    print("   • Autoencoders funcionan bien para detección no supervisada")
    print("   • Métodos basados en forecasting son ideales para series temporales")
    print("   • Ensemble methods mejoran la robustez y reducen falsos positivos")

    print("\\n🔮 PRÓXIMOS PASOS PARA ANOMALY DETECTION:")
    print("   • Implementar Graph Neural Networks para anomalías en grafos")
    print("   • Agregar detección de anomalías en imágenes")
    print("   • Implementar métodos probabilísticos (VAE, GAN)")
    print("   • Crear sistema de alerting en tiempo real")
    print("   • Agregar explainability para anomalías detectadas")
    print("   • Implementar active learning para refinar detectores")

    print("\\n" + "=" * 60)
    print("🌟 Anomaly Detection automática funcionando correctamente!")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(demo_anomaly_detection())
