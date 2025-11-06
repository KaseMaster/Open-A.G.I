#!/usr/bin/env python3
"""
🔮 Dimensional Observer Agent
Real-time telemetry and anomaly detection for Ω-Ψ states in Quantum Currency network.

This module implements continuous monitoring of coherence metrics with anomaly detection
on semantic_shift and sentiment_energy fields, and streams data to dashboard for visualization.
"""

import asyncio
import time
import json
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import logging
from datetime import datetime, timedelta
from scipy import stats

# Import Quantum Currency components
# Handle relative imports properly
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import required classes
from dataclasses import dataclass

# Try to import from core, fallback to mock if not available
try:
    from core.cal_engine import CALEngine
    from models.coherence_attunement_layer import CoherenceAttunementLayer
except ImportError:
    # Create mock classes for testing
    class CALEngine:
        def __init__(self, network_id="test"):
            self.network_id = network_id
            self.omega_history = []
            
        def compute_omega_state(self, token_data, sentiment_data, semantic_data, attention_data):
            # Mock implementation
            class MockOmegaState:
                def __init__(self, token_rate, sentiment_energy, semantic_shift, meta_attention_spectrum, coherence_score):
                    self.token_rate = token_rate
                    self.sentiment_energy = sentiment_energy
                    self.semantic_shift = semantic_shift
                    self.meta_attention_spectrum = meta_attention_spectrum
                    self.coherence_score = coherence_score
                    self.modulator = 1.0
                    self.time_delay = 0.0
                    self.integrated_feedback = 0.0
                    self.timestamp = time.time()
                
            return MockOmegaState(
                token_rate=token_data.get("rate", 0.0),
                sentiment_energy=sentiment_data.get("energy", 0.0),
                semantic_shift=semantic_data.get("shift", 0.0),
                meta_attention_spectrum=attention_data,
                coherence_score=0.5
            )
            
        def get_coherence_health_indicator(self):
            return "green"

    class CoherenceAttunementLayer:
        def __init__(self, network_id="test"):
            self.network_id = network_id

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TelemetryData:
    """Telemetry data point for Ω-Ψ monitoring"""
    timestamp: float
    omega_state: Dict[str, Any]
    psi_score: float
    modulator: float
    time_delay: float
    integrated_feedback: float
    network_health: str

@dataclass
class AnomalyDetectionResult:
    """Result of anomaly detection on Ω-Ψ fields"""
    timestamp: float
    field_name: str
    value: float
    z_score: float
    is_anomaly: bool
    severity: str  # "low", "medium", "high"
    description: str

class DimensionalObserverAgent:
    """
    Dimensional Observer Agent for Quantum Currency v0.4.0
    
    This agent provides real-time telemetry and anomaly detection for Ω-Ψ states:
    1. Streams Ω and Ψ data live to the dashboard for visualization
    2. Detects anomalies in semantic_shift and sentiment_energy fields
    3. Monitors network coherence health indicators
    4. Provides alerts for dimensional inconsistencies
    """

    def __init__(self, network_id: str = "quantum-network-001"):
        self.network_id = network_id
        
        # Initialize monitoring components
        self.cal_engine = CALEngine(network_id)
        self.coherence_layer = CoherenceAttunementLayer(network_id)
        
        # Telemetry storage
        self.telemetry_history: List[TelemetryData] = []
        self.anomaly_history: List[AnomalyDetectionResult] = []
        
        # Monitoring configuration
        self.monitoring_config = {
            "telemetry_frequency": 5.0,  # seconds
            "anomaly_detection_window": 100,  # data points
            "z_score_threshold": 3.0,  # for anomaly detection
            "health_check_frequency": 30.0  # seconds
        }
        
        # Anomaly detection statistics
        self.field_stats: Dict[str, Dict[str, float]] = {
            "semantic_shift": {"mean": 0.0, "std": 1.0, "count": 0},
            "sentiment_energy": {"mean": 0.0, "std": 1.0, "count": 0}
        }
        
        logger.info(f"🔮 Dimensional Observer Agent initialized for network: {network_id}")

    async def start_monitoring(self):
        """Start continuous monitoring of Ω-Ψ states"""
        logger.info("🔮 Starting dimensional observer monitoring...")
        
        while True:
            try:
                # Collect telemetry data
                telemetry_data = await self._collect_telemetry()
                
                # Store telemetry
                self.telemetry_history.append(telemetry_data)
                
                # Keep only recent history
                if len(self.telemetry_history) > 1000:
                    self.telemetry_history = self.telemetry_history[-1000:]
                
                # Perform anomaly detection
                anomalies = await self._detect_anomalies(telemetry_data)
                
                # Store anomalies
                self.anomaly_history.extend(anomalies)
                
                # Keep only recent anomalies
                if len(self.anomaly_history) > 100:
                    self.anomaly_history = self.anomaly_history[-100:]
                
                # Check network health
                health_status = await self._check_network_health()
                
                # Log significant events
                if anomalies:
                    for anomaly in anomalies:
                        if anomaly.is_anomaly and anomaly.severity in ["high", "medium"]:
                            logger.warning(f"⚠️ Anomaly detected: {anomaly.field_name} = {anomaly.value:.4f} "
                                         f"(z-score: {anomaly.z_score:.2f}) - {anomaly.description}")
                
                if health_status == "critical":
                    logger.critical(f"🚨 Critical network health detected: {health_status}")
                
                # Wait for next monitoring cycle
                await asyncio.sleep(self.monitoring_config["telemetry_frequency"])
                
            except Exception as e:
                logger.error(f"❌ Error in monitoring cycle: {e}")
                await asyncio.sleep(10)  # Wait longer on error

    async def _collect_telemetry(self) -> TelemetryData:
        """Collect current Ω-Ψ telemetry data"""
        # In a real implementation, this would get data from the actual network
        # For now, we'll simulate data collection
        
        # Get latest Ω-state (in real implementation, this would come from network nodes)
        if self.cal_engine.omega_history:
            latest_omega = self.cal_engine.omega_history[-1]
        else:
            # Create a sample Ω-state for demonstration
            latest_omega = self.cal_engine.compute_omega_state(
                token_data={"rate": 5.0},
                sentiment_data={"energy": 0.7},
                semantic_data={"shift": 0.3},
                attention_data=[0.1, 0.2, 0.3, 0.4, 0.5]
            )
        
        # Convert Ω-state to dictionary for telemetry
        omega_dict = {
            "token_rate": latest_omega.token_rate,
            "sentiment_energy": latest_omega.sentiment_energy,
            "semantic_shift": latest_omega.semantic_shift,
            "meta_attention_spectrum": latest_omega.meta_attention_spectrum,
            "coherence_score": latest_omega.coherence_score,
            "modulator": latest_omega.modulator,
            "time_delay": latest_omega.time_delay,
            "integrated_feedback": latest_omega.integrated_feedback
        }
        
        # Get network health indicator
        health_indicator = self.cal_engine.get_coherence_health_indicator()
        
        telemetry = TelemetryData(
            timestamp=time.time(),
            omega_state=omega_dict,
            psi_score=latest_omega.coherence_score,
            modulator=latest_omega.modulator,
            time_delay=latest_omega.time_delay,
            integrated_feedback=latest_omega.integrated_feedback,
            network_health=health_indicator
        )
        
        return telemetry

    async def _detect_anomalies(self, telemetry_data: TelemetryData) -> List[AnomalyDetectionResult]:
        """Detect anomalies in semantic_shift and sentiment_energy fields"""
        anomalies = []
        
        # Extract fields to monitor
        fields_to_monitor = {
            "semantic_shift": telemetry_data.omega_state["semantic_shift"],
            "sentiment_energy": telemetry_data.omega_state["sentiment_energy"]
        }
        
        # Update statistics and detect anomalies for each field
        for field_name, value in fields_to_monitor.items():
            # Update running statistics
            self._update_field_statistics(field_name, value)
            
            # Calculate z-score
            stats = self.field_stats[field_name]
            if stats["std"] > 0:
                z_score = abs((value - stats["mean"]) / stats["std"])
            else:
                z_score = 0.0
            
            # Determine if anomaly
            is_anomaly = z_score > self.monitoring_config["z_score_threshold"]
            
            # Determine severity
            if z_score > self.monitoring_config["z_score_threshold"] * 2:
                severity = "high"
            elif z_score > self.monitoring_config["z_score_threshold"]:
                severity = "medium"
            else:
                severity = "low"
            
            # Create description
            if is_anomaly:
                description = f"Value {value:.4f} deviates significantly from mean {stats['mean']:.4f} (z={z_score:.2f})"
            else:
                description = f"Value {value:.4f} within normal range (z={z_score:.2f})"
            
            # Create anomaly result
            anomaly_result = AnomalyDetectionResult(
                timestamp=telemetry_data.timestamp,
                field_name=field_name,
                value=value,
                z_score=z_score,
                is_anomaly=is_anomaly,
                severity=severity,
                description=description
            )
            
            anomalies.append(anomaly_result)
        
        return anomalies

    def _update_field_statistics(self, field_name: str, value: float):
        """Update running statistics for a field"""
        if field_name not in self.field_stats:
            self.field_stats[field_name] = {"mean": 0.0, "std": 1.0, "count": 0}
        
        stats = self.field_stats[field_name]
        count = stats["count"]
        
        # Update mean using online algorithm
        if count == 0:
            stats["mean"] = value
            stats["std"] = 1.0
        else:
            # Update mean
            new_mean = stats["mean"] + (value - stats["mean"]) / (count + 1)
            
            # Update std (simplified)
            if count > 1:
                variance = ((count - 1) * (stats["std"] ** 2) + (value - stats["mean"]) * (value - new_mean)) / count
                stats["std"] = np.sqrt(max(0, variance))
            
            stats["mean"] = new_mean
        
        stats["count"] = count + 1

    async def _check_network_health(self) -> str:
        """Check overall network health based on recent telemetry"""
        if not self.telemetry_history:
            return "unknown"
        
        # Get recent health indicators
        recent_health = [data.network_health for data in self.telemetry_history[-10:]]
        
        # Count health statuses
        health_counts = {}
        for health in recent_health:
            health_counts[health] = health_counts.get(health, 0) + 1
        
        # Determine dominant health status
        if health_counts.get("critical", 0) > 3:
            return "critical"
        elif health_counts.get("red", 0) > 5:
            return "unstable"
        elif health_counts.get("yellow", 0) > 5:
            return "degraded"
        elif health_counts.get("green", 0) > 7:
            return "healthy"
        else:
            return "degraded"

    def get_recent_telemetry(self, count: int = 10) -> List[TelemetryData]:
        """Get recent telemetry data points"""
        return self.telemetry_history[-count:] if self.telemetry_history else []

    def get_recent_anomalies(self, count: int = 10) -> List[AnomalyDetectionResult]:
        """Get recent anomaly detection results"""
        return self.anomaly_history[-count:] if self.anomaly_history else []

    def get_network_health_summary(self) -> Dict[str, Any]:
        """Get network health summary statistics"""
        if not self.telemetry_history:
            return {"status": "no_data", "psi_stats": {}, "anomaly_count": 0}
        
        # Calculate Ψ statistics
        recent_psi = [data.psi_score for data in self.telemetry_history[-50:]]
        psi_stats = {
            "mean": float(np.mean(recent_psi)),
            "std": float(np.std(recent_psi)),
            "min": float(np.min(recent_psi)),
            "max": float(np.max(recent_psi)),
            "trend": "stable" if len(recent_psi) > 1 else "unknown"
        }
        
        # Determine trend
        if len(recent_psi) > 10:
            recent_values = recent_psi[-10:]
            older_values = recent_psi[-20:-10]
            recent_mean = np.mean(recent_values)
            older_mean = np.mean(older_values)
            
            if recent_mean > older_mean + 0.1:
                psi_stats["trend"] = "improving"
            elif recent_mean < older_mean - 0.1:
                psi_stats["trend"] = "declining"
            else:
                psi_stats["trend"] = "stable"
        
        # Count anomalies
        anomaly_count = len([a for a in self.anomaly_history[-50:] if a.is_anomaly])
        
        # Get current health
        current_health = self.telemetry_history[-1].network_health if self.telemetry_history else "unknown"
        
        return {
            "status": "operational",
            "current_health": current_health,
            "psi_stats": psi_stats,
            "anomaly_count": anomaly_count,
            "total_telemetry_points": len(self.telemetry_history)
        }

    async def stream_to_dashboard(self):
        """Stream telemetry data to dashboard (placeholder for actual implementation)"""
        # In a real implementation, this would:
        # 1. Connect to dashboard WebSocket endpoint
        # 2. Stream telemetry data in real-time
        # 3. Handle reconnection logic
        # 4. Format data for dashboard consumption
        
        logger.info("📡 Streaming telemetry to dashboard...")
        
        # For now, just log that streaming would occur
        if self.telemetry_history:
            latest = self.telemetry_history[-1]
            logger.info(f"📡 Streaming: Ψ={latest.psi_score:.4f}, health={latest.network_health}")

# Demo function
async def demo_observer_agent():
    """Demonstrate the Dimensional Observer Agent"""
    print("🔮 Dimensional Observer Agent Demo")
    print("=" * 50)
    
    # Initialize observer agent
    observer = DimensionalObserverAgent("demo-network-001")
    
    # Collect some sample telemetry
    for i in range(5):
        telemetry = await observer._collect_telemetry()
        anomalies = await observer._detect_anomalies(telemetry)
        
        print(f"📊 Telemetry Point {i+1}:")
        print(f"   Ψ Score: {telemetry.psi_score:.4f}")
        print(f"   Network Health: {telemetry.network_health}")
        print(f"   Anomalies Detected: {len([a for a in anomalies if a.is_anomaly])}")
        
        # Simulate some data variation
        # This would normally come from the network
        observer.cal_engine.compute_omega_state(
            token_data={"rate": 5.0 + i * 0.1},
            sentiment_data={"energy": 0.7 + i * 0.01},
            semantic_data={"shift": 0.3 + i * 0.005},
            attention_data=[0.1 + i*0.01, 0.2 + i*0.01, 0.3 + i*0.01, 0.4 + i*0.01, 0.5 + i*0.01]
        )
    
    # Show health summary
    health_summary = observer.get_network_health_summary()
    print(f"\n📈 Network Health Summary:")
    print(f"   Status: {health_summary['status']}")
    if 'current_health' in health_summary:
        print(f"   Current Health: {health_summary['current_health']}")
    if 'psi_stats' in health_summary:
        psi_stats = health_summary['psi_stats']
        print(f"   Ψ Mean: {psi_stats.get('mean', 'N/A')}")
        print(f"   Ψ Trend: {psi_stats.get('trend', 'N/A')}")
    print(f"   Anomalies: {health_summary.get('anomaly_count', 'N/A')}")
    
    print("\n✅ Demo completed!")

if __name__ == "__main__":
    asyncio.run(demo_observer_agent())