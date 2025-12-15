#!/usr/bin/env python3
"""
PAF Engine (Predictive Anomaly Forecasting)
Enables transition from reactive to predictive governance
"""

import json
import time
import logging
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from dataclasses import dataclass, asdict

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ForecastResult:
    """Represents a forecasting result"""
    metric_name: str
    forecast_values: List[float]
    confidence_intervals: List[Tuple[float, float]]
    timestamp: float
    horizon: int

class PAF_Engine:
    """Predictive Anomaly Forecasting Engine"""
    
    def __init__(self):
        self.historical_data: Dict[str, List[float]] = {}
        self.forecast_models: Dict[str, Any] = {}
        self.is_trained = False
        self.oracle_data: List[Dict[str, Any]] = []
        
    def train_on_oracle_data(self, oracle_data: Optional[List[Dict[str, Any]]] = None):
        """
        Train the PAF engine on historical oracle data
        
        Args:
            oracle_data: Optional list of historical data records
        """
        if oracle_data is None:
            # Generate synthetic training data if none provided
            oracle_data = self._generate_synthetic_training_data()
        
        self.oracle_data = oracle_data
        
        # Extract metrics from oracle data
        metrics = {
            "I_eff": [],
            "g_vector_magnitude": [],
            "coherence_score": [],
            "action_efficiency": []
        }
        
        for record in oracle_data:
            metrics["I_eff"].append(record.get("I_eff", 0.01))
            metrics["g_vector_magnitude"].append(record.get("g_vector_magnitude", 0.1))
            metrics["coherence_score"].append(record.get("coherence_score", 0.95))
            metrics["action_efficiency"].append(record.get("action_efficiency", 0.8))
        
        # Store historical data
        self.historical_data = metrics
        
        # Train models for each metric
        for metric_name, values in metrics.items():
            if len(values) > 10:  # Need minimum data to train
                self._train_model(metric_name, values)
        
        self.is_trained = True
        logger.info("PAF Engine trained on oracle data")
    
    def _generate_synthetic_training_data(self, num_samples: int = 1000) -> List[Dict[str, Any]]:
        """
        Generate synthetic training data for initial training
        
        Args:
            num_samples: Number of samples to generate
            
        Returns:
            List of synthetic data records
        """
        data = []
        for i in range(num_samples):
            # Simulate realistic patterns in the data
            time_factor = i / num_samples
            noise = np.random.normal(0, 0.01)
            
            record = {
                "timestamp": time.time() - (num_samples - i) * 3600,  # Hours ago
                "I_eff": max(0.001, 0.01 + 0.02 * np.sin(2 * np.pi * time_factor * 10) + noise),
                "g_vector_magnitude": max(0.0, 0.1 + 0.2 * np.sin(2 * np.pi * time_factor * 5) + noise),
                "coherence_score": max(0.8, min(1.0, 0.95 + 0.05 * np.sin(2 * np.pi * time_factor * 3) + noise)),
                "action_efficiency": max(0.5, min(1.0, 0.8 + 0.2 * np.sin(2 * np.pi * time_factor * 7) + noise))
            }
            data.append(record)
        return data
    
    def _train_model(self, metric_name: str, values: List[float]):
        """
        Train a forecasting model for a specific metric
        
        Args:
            metric_name: Name of the metric to train model for
            values: Historical values of the metric
        """
        if len(values) < 10:
            return
            
        # Convert to numpy array
        values = np.array(values)
        
        # Create features (use lagged values as features)
        window_size = min(10, len(values) // 2)
        X, y = [], []
        
        for i in range(window_size, len(values)):
            X.append(values[i-window_size:i])
            y.append(values[i])
        
        if len(X) == 0:
            return
            
        X = np.array(X)
        y = np.array(y)
        
        # Train linear regression model
        model = LinearRegression()
        model.fit(X, y)
        
        # Store model
        self.forecast_models[metric_name] = {
            "model": model,
            "window_size": window_size,
            "last_values": values[-window_size:].tolist()
        }
        
        logger.debug(f"Trained model for {metric_name} with {len(X)} samples")
    
    def forecast_I_eff(self, horizon: int = 10) -> ForecastResult:
        """
        Forecast future I_eff values
        
        Args:
            horizon: Number of time steps to forecast
            
        Returns:
            ForecastResult with predicted values
        """
        return self._forecast_metric("I_eff", horizon)
    
    def forecast_g_vector(self, horizon: int = 10) -> ForecastResult:
        """
        Forecast future g-vector magnitude values
        
        Args:
            horizon: Number of time steps to forecast
            
        Returns:
            ForecastResult with predicted values
        """
        return self._forecast_metric("g_vector_magnitude", horizon)
    
    def forecast_coherence(self, horizon: int = 10) -> ForecastResult:
        """
        Forecast future coherence score values
        
        Args:
            horizon: Number of time steps to forecast
            
        Returns:
            ForecastResult with predicted values
        """
        return self._forecast_metric("coherence_score", horizon)
    
    def _forecast_metric(self, metric_name: str, horizon: int) -> ForecastResult:
        """
        Forecast future values for a specific metric
        
        Args:
            metric_name: Name of the metric to forecast
            horizon: Number of time steps to forecast
            
        Returns:
            ForecastResult with predicted values
        """
        if not self.is_trained or metric_name not in self.forecast_models:
            # Return dummy forecast if not trained
            return ForecastResult(
                metric_name=metric_name,
                forecast_values=[0.01] * horizon,
                confidence_intervals=[(0.005, 0.015)] * horizon,
                timestamp=time.time(),
                horizon=horizon
            )
        
        model_info = self.forecast_models[metric_name]
        model = model_info["model"]
        window_size = model_info["window_size"]
        last_values = model_info["last_values"][:]
        
        forecast_values = []
        confidence_intervals = []
        
        # Generate forecasts for the specified horizon
        for _ in range(horizon):
            # Prepare input features
            X = np.array(last_values[-window_size:]).reshape(1, -1)
            
            # Make prediction
            prediction = model.predict(X)[0]
            
            # Add to forecast
            forecast_values.append(float(prediction))
            
            # Simple confidence interval (in a real implementation, this would be more sophisticated)
            confidence_interval = (float(prediction * 0.95), float(prediction * 1.05))
            confidence_intervals.append(confidence_interval)
            
            # Update last values for next prediction
            last_values.append(prediction)
        
        return ForecastResult(
            metric_name=metric_name,
            forecast_values=forecast_values,
            confidence_intervals=confidence_intervals,
            timestamp=time.time(),
            horizon=horizon
        )
    
    def detect_anomalies(self, metric_name: str, current_value: float) -> Dict[str, Any]:
        """
        Detect anomalies in a metric value
        
        Args:
            metric_name: Name of the metric to check
            current_value: Current value of the metric
            
        Returns:
            Dictionary with anomaly detection results
        """
        if metric_name not in self.historical_data or len(self.historical_data[metric_name]) < 10:
            return {
                "is_anomaly": False,
                "anomaly_score": 0.0,
                "threshold": 0.0,
                "message": "Insufficient data for anomaly detection"
            }
        
        # Get historical data for this metric
        historical_values = np.array(self.historical_data[metric_name])
        
        # Use Isolation Forest for anomaly detection
        scaler = StandardScaler()
        scaled_values = scaler.fit_transform(historical_values.reshape(-1, 1))
        
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        iso_forest.fit(scaled_values)
        
        # Check if current value is anomalous
        scaled_current = scaler.transform([[current_value]])
        anomaly_prediction = iso_forest.predict(scaled_current)[0]
        anomaly_score = -iso_forest.score_samples(scaled_current)[0]
        
        is_anomaly = anomaly_prediction == -1
        threshold = iso_forest.contamination
        
        return {
            "is_anomaly": is_anomaly,
            "anomaly_score": float(anomaly_score),
            "threshold": threshold,
            "message": f"Value {'is' if is_anomaly else 'is not'} anomalous"
        }
    
    def get_critical_I_eff_threshold(self) -> float:
        """
        Get the critical I_eff threshold for anomaly detection
        
        Returns:
            Critical threshold value
        """
        # In a real implementation, this would be dynamically calculated
        # For now, we'll use a fixed threshold
        return 0.05

# Example usage
if __name__ == "__main__":
    # Create PAF engine
    paf_engine = PAF_Engine()
    
    # Train on oracle data
    paf_engine.train_on_oracle_data()
    
    # Forecast I_eff
    I_eff_forecast = paf_engine.forecast_I_eff(horizon=5)
    print(f"I_eff forecast: {I_eff_forecast.forecast_values}")
    
    # Forecast g-vector
    g_forecast = paf_engine.forecast_g_vector(horizon=5)
    print(f"g-vector forecast: {g_forecast.forecast_values}")
    
    # Check if a value is anomalous
    anomaly_result = paf_engine.detect_anomalies("I_eff", 0.08)
    print(f"Anomaly detection: {anomaly_result}")