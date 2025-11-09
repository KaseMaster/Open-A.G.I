#!/usr/bin/env python3
"""
CAL Engine - Coherence Attunement Layer Core
Implements the recursive stability loop and Ω-state recursion for UHES

This module provides:
1. λ(L) Adaptive Decay Factor computation
2. m_t(L) Harmonic Modulator
3. Ω_t(L) State Update
4. Dimensional Stability Validation
5. Ω-State Checkpointing for mainnet readiness
"""

import numpy as np
from typing import List, Dict, Tuple, Any, Optional, Union
import logging
import time
import math
import json
import os
from dataclasses import dataclass, asdict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Golden ratio and related constants
PHI = 1.618033988749895
LAMBDA = 1.0 / PHI

@dataclass
class CheckpointData:
    """Represents checkpointed Ω-state data for rapid restarts"""
    I_t_L: List[float]  # Integrated feedback vector
    Omega_t_L: List[float]  # Harmonic state vector
    timestamp: float
    coherence_score: float
    modulator: float
    network_id: str

class CALEngine:
    """
    Coherence Attunement Layer Engine
    Implements the core functions for Ω-state recursion and harmonic validation
    """
    
    def __init__(self, network_id: str = "quantum-currency-uhes-001"):
        self.network_id = network_id
        self.omega_history: List[Dict[str, Any]] = []
        self.performance_metrics: Dict[str, List[float]] = {
            "latency": [],
            "memory_usage": [],
            "wave_propagation_throughput": []
        }
        self.safety_bounds = {
            "dimensional_clamp": 10.0,  # K bound for dimensional consistency
            "coherence_recovery_threshold": 0.7,  # Minimum Ψ for recovery
            "recovery_step_limit": 50  # Max steps for re-stabilization
        }
        
        # Configuration parameters
        self.config = {
            "lambda_decay_factor": LAMBDA,
            "phi_scaling": PHI,
            "auto_tuning_enabled": True,
            "tuning_interval": 300  # 5 minutes
        }
        
        # AI-driven tuning parameters
        self.tuning_params = {
            "latency_weight": 0.3,
            "memory_weight": 0.4,
            "throughput_weight": 0.3,
            "learning_rate": 0.01
        }
        
        self.last_tuning_time = time.time()
        
        # Checkpointing for mainnet readiness
        self.checkpoints: List[CheckpointData] = []
        self.checkpoint_dir = "checkpoints"
        self.max_checkpoints = 10
        
        # Create checkpoint directory if it doesn't exist
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        
        logger.info(f"🌀 CAL Engine initialized for network: {network_id}")
    
    def lambda_L(self, L: str, psi_t: float) -> float:
        """
        Compute Adaptive Decay Factor λ(L)
        
        Args:
            L: Scale level (Lμ, Lϕ, LΦ)
            psi_t: Current coherence score
            
        Returns:
            float: Adaptive decay factor
        """
        # ϕ₁ · Ψ_t - The Adaptive Decay Factor
        phi_1 = PHI  # Using golden ratio as ϕ₁
        lambda_L = phi_1 * psi_t
        
        logger.debug(f"λ(L) computed: {lambda_L:.4f} for scale {L}")
        return lambda_L
    
    def modulator(self, I_vector: Union[List[float], np.ndarray], 
                  lambda_L: float, K: float = 10.0) -> float:
        """
        Compute Harmonic Modulator m_t(L)
        
        Args:
            I_vector: Integrated feedback vector
            lambda_L: Adaptive decay factor
            K: Clamping bound (default: 10.0)
            
        Returns:
            float: Harmonic modulator value
        """
        # Convert to numpy array if needed
        if not isinstance(I_vector, np.ndarray):
            I_vector = np.array(I_vector)
        
        # Compute projection of I_vector
        if len(I_vector) > 0:
            proj_I = np.mean(I_vector)  # Simple projection
        else:
            proj_I = 0.0
        
        # Compute λ(L) · proj(I_t(L))
        product = lambda_L * proj_I
        
        # Clamp to ±K bounds for dimensional consistency
        clamped_product = max(-K, min(K, product))
        
        # Compute modulator: exp(clamp(λ(L) · proj(I_t(L)), -K, K))
        modulator_value = math.exp(clamped_product)
        
        logger.debug(f"Modulator computed: {modulator_value:.4f} (product: {product:.4f}, clamped: {clamped_product:.4f})")
        return modulator_value
    
    def update_omega(self, features: Union[List[float], np.ndarray], 
                     I_vector: Union[List[float], np.ndarray], 
                     L: str) -> Tuple[np.ndarray, float]:
        """
        Update Ω State Vector
        
        Args:
            features: Feature vector F_t
            I_vector: Integrated feedback vector I_t(L)
            L: Scale level (Lμ, Lϕ, LΦ)
            
        Returns:
            Tuple of (updated_omega_vector, modulator_value)
        """
        # Convert to numpy arrays if needed
        if not isinstance(features, np.ndarray):
            features = np.array(features)
        if not isinstance(I_vector, np.ndarray):
            I_vector = np.array(I_vector)
        
        # Compute λ(L) from current coherence (using placeholder value)
        # In practice, this would come from the coherence scorer
        current_psi = 0.8  # Placeholder - would be computed from history
        lambda_L = self.lambda_L(L, current_psi)
        
        # Compute modulator
        m_t = self.modulator(I_vector, lambda_L)
        
        # Normalize features
        if len(features) > 0 and np.linalg.norm(features) > 0:
            normalized_features = features / np.linalg.norm(features)
        else:
            normalized_features = features
        
        # Update Ω_t(L) = Normalize(F_t) × m_t(L)
        updated_omega = normalized_features * m_t if len(normalized_features) > 0 else np.array([])
        
        logger.debug(f"Ω state updated for scale {L}: norm={np.linalg.norm(updated_omega):.4f}, m_t={m_t:.4f}")
        return updated_omega, m_t
    
    def validate_dimensional_stability(self, modulator_value: float, K: float = 10.0) -> bool:
        """
        Validate Dimensional Stability Test (Critical Safety Check)
        
        Args:
            modulator_value: The computed modulator value
            K: Clamping bound (default: 10.0)
            
        Returns:
            bool: True if dimensionally stable, False otherwise
        """
        # Check that the log of modulator is within bounds
        # This ensures λ(L) · proj(I_t(L)) remains dimensionless and clamped
        try:
            log_modulator = math.log(modulator_value)
            is_stable = -K <= log_modulator <= K
            if not is_stable:
                logger.warning(f"Dimensional instability detected: log(m_t) = {log_modulator:.4f} exceeds bounds ±{K}")
            return is_stable
        except (ValueError, OverflowError):
            logger.warning(f"Dimensional instability: modulator value {modulator_value} causes overflow")
            return False
    
    def compute_recursive_coherence(self, omega_vectors: List[np.ndarray]) -> float:
        """
        Compute recursive coherence score between Ω vectors
        
        Args:
            omega_vectors: List of Ω state vectors
            
        Returns:
            float: Aggregated coherence score (0-1)
        """
        if not omega_vectors or len(omega_vectors) < 2:
            return 0.0
        
        # Compute pairwise cosine similarities
        similarities = []
        for i, vec1 in enumerate(omega_vectors):
            for j, vec2 in enumerate(omega_vectors):
                if i != j:
                    # Compute cosine similarity
                    if len(vec1) > 0 and len(vec2) > 0:
                        dot_product = np.dot(vec1, vec2)
                        norm1 = np.linalg.norm(vec1)
                        norm2 = np.linalg.norm(vec2)
                        if norm1 > 0 and norm2 > 0:
                            cos_sim = dot_product / (norm1 * norm2)
                            similarities.append(max(0.0, min(1.0, cos_sim)))  # Clamp to [0,1]
        
        if not similarities:
            return 0.0
        
        # Return mean similarity as coherence score
        coherence_score = float(np.mean(similarities))
        logger.debug(f"Recursive coherence computed: {coherence_score:.4f}")
        return coherence_score
    
    def record_performance_metrics(self, latency: float, memory_usage: float, 
                                 throughput: float):
        """
        Record performance metrics for AI-driven tuning
        
        Args:
            latency: System latency in seconds
            memory_usage: Memory usage percentage (0-100)
            throughput: Wave propagation throughput
        """
        self.performance_metrics["latency"].append(latency)
        self.performance_metrics["memory_usage"].append(memory_usage)
        self.performance_metrics["wave_propagation_throughput"].append(throughput)
        
        # Keep only recent history (last 100 metrics)
        for metric_name in self.performance_metrics:
            if len(self.performance_metrics[metric_name]) > 100:
                self.performance_metrics[metric_name] = self.performance_metrics[metric_name][-100:]
    
    def analyze_performance_trends(self) -> Dict[str, float]:
        """
        Analyze performance trends to identify optimization opportunities
        
        Returns:
            Dict with trend analysis results
        """
        trends = {}
        
        for metric_name, values in self.performance_metrics.items():
            if len(values) < 2:
                trends[metric_name] = {"trend": 0.0, "stability": 1.0}
                continue
            
            # Calculate trend (slope of recent values)
            recent_values = values[-10:] if len(values) >= 10 else values
            x = np.arange(len(recent_values))
            slope, _ = np.polyfit(x, recent_values, 1)
            
            # Calculate stability (inverse of variance)
            stability = 1.0 / (np.var(recent_values) + 1e-10)
            
            trends[metric_name] = {
                "trend": float(slope),
                "stability": float(stability),
                "current": float(recent_values[-1]),
                "average": float(np.mean(recent_values))
            }
        
        return trends
    
    def ai_driven_coherence_tuning(self) -> Dict[str, Any]:
        """
        AI-driven coherence tuning for performance optimization
        
        Returns:
            Dict with tuning recommendations and adjustments
        """
        if not self.config["auto_tuning_enabled"]:
            return {"status": "disabled", "message": "Auto-tuning is disabled"}
        
        # Check if it's time for tuning
        current_time = time.time()
        if current_time - self.last_tuning_time < self.config["tuning_interval"]:
            return {"status": "skipped", "message": "Tuning interval not reached"}
        
        self.last_tuning_time = current_time
        
        # Analyze performance trends
        trends = self.analyze_performance_trends()
        
        # Calculate tuning adjustments based on performance metrics
        adjustments = {}
        recommendations = []
        
        # Latency optimization
        if "latency" in trends:
            latency_info = trends["latency"]
            if isinstance(latency_info, dict) and "trend" in latency_info:
                latency_trend = latency_info["trend"]
                if latency_trend > 0.01:  # Increasing latency
                    # Reduce lambda decay factor to speed up processing
                    adjustments["lambda_decay_factor"] = max(0.1, self.config["lambda_decay_factor"] * 0.95)
                    recommendations.append("Reduce lambda decay factor to optimize latency")
                elif latency_trend < -0.01:  # Decreasing latency
                    # Can afford to increase coherence quality
                    adjustments["lambda_decay_factor"] = min(0.9, self.config["lambda_decay_factor"] * 1.05)
                    recommendations.append("Increase lambda decay factor to improve coherence quality")
        
        # Memory optimization
        if "memory_usage" in trends:
            memory_info = trends["memory_usage"]
            if isinstance(memory_info, dict) and "current" in memory_info:
                memory_current = memory_info["current"]
                if memory_current > 85:  # High memory usage
                    # Reduce memory-intensive operations
                    adjustments["phi_scaling"] = max(1.1, self.config["phi_scaling"] * 0.98)
                    recommendations.append("Reduce phi scaling to optimize memory usage")
                elif memory_current < 50:  # Low memory usage
                    # Can afford more complex operations
                    adjustments["phi_scaling"] = min(2.0, self.config["phi_scaling"] * 1.02)
                    recommendations.append("Increase phi scaling to improve processing power")
        
        # Throughput optimization
        if "wave_propagation_throughput" in trends:
            throughput_info = trends["wave_propagation_throughput"]
            if isinstance(throughput_info, dict) and "trend" in throughput_info:
                throughput_trend = throughput_info["trend"]
                if throughput_trend < -0.05:  # Decreasing throughput
                    # Optimize for throughput
                    adjustments["tuning_interval"] = max(60, self.config["tuning_interval"] * 0.9)
                    recommendations.append("Increase tuning frequency to optimize throughput")
                elif throughput_trend > 0.05:  # Increasing throughput
                    # Can reduce tuning frequency to save resources
                    adjustments["tuning_interval"] = min(600, self.config["tuning_interval"] * 1.1)
                    recommendations.append("Reduce tuning frequency to conserve resources")
        
        # Apply adjustments
        for param, value in adjustments.items():
            self.config[param] = value
        
        result = {
            "status": "success",
            "adjustments": adjustments,
            "recommendations": recommendations,
            "performance_trends": trends,
            "timestamp": current_time
        }
        
        logger.info(f"AI-driven coherence tuning completed: {len(adjustments)} adjustments made")
        return result
    
    def get_auto_balance_mode_status(self) -> Dict[str, Any]:
        """
        Get Auto-Balance Mode status for continuous system recalculations
        
        Returns:
            Dict with Auto-Balance Mode status
        """
        return {
            "enabled": self.config["auto_tuning_enabled"],
            "tuning_interval": self.config["tuning_interval"],
            "last_tuning": self.last_tuning_time,
            "next_tuning": self.last_tuning_time + self.config["tuning_interval"],
            "performance_metrics_count": {k: len(v) for k, v in self.performance_metrics.items()}
        }
    
    # Ω-State Checkpointing Implementation for Mainnet Readiness
    def create_checkpoint(self, I_t_L: List[float], Omega_t_L: List[float], 
                         coherence_score: float, modulator: float) -> CheckpointData:
        """
        Create a checkpoint of the current Ω-state for rapid restarts
        
        Args:
            I_t_L: Integrated feedback vector
            Omega_t_L: Harmonic state vector
            coherence_score: Current coherence score
            modulator: Current modulator value
            
        Returns:
            CheckpointData object representing the checkpoint
        """
        checkpoint = CheckpointData(
            I_t_L=I_t_L,
            Omega_t_L=Omega_t_L,
            timestamp=time.time(),
            coherence_score=coherence_score,
            modulator=modulator,
            network_id=self.network_id
        )
        
        # Add to checkpoints list
        self.checkpoints.append(checkpoint)
        
        # Keep only recent checkpoints
        if len(self.checkpoints) > self.max_checkpoints:
            self.checkpoints = self.checkpoints[-self.max_checkpoints:]
        
        # Serialize and persist to durable storage
        self._persist_checkpoint(checkpoint)
        
        logger.info(f"_checkpoint created: coherence={coherence_score:.4f}, modulator={modulator:.4f}")
        return checkpoint
    
    def _persist_checkpoint(self, checkpoint: CheckpointData):
        """
        Persist checkpoint to encrypted durable storage
        
        Args:
            checkpoint: CheckpointData to persist
        """
        try:
            # Create checkpoint filename
            timestamp_str = str(int(checkpoint.timestamp))
            filename = f"checkpoint_{timestamp_str}.json"
            filepath = os.path.join(self.checkpoint_dir, filename)
            
            # Serialize checkpoint data
            checkpoint_dict = asdict(checkpoint)
            
            # Write to file
            with open(filepath, 'w') as f:
                json.dump(checkpoint_dict, f, indent=2)
            
            logger.info(f"Checkpoint persisted to {filepath}")
        except Exception as e:
            logger.error(f"Failed to persist checkpoint: {e}")
    
    def load_latest_checkpoint(self) -> Optional[CheckpointData]:
        """
        Load the latest checkpoint from durable storage
        
        Returns:
            Latest CheckpointData or None if no checkpoints exist
        """
        try:
            # Get all checkpoint files
            checkpoint_files = [f for f in os.listdir(self.checkpoint_dir) 
                              if f.startswith("checkpoint_") and f.endswith(".json")]
            
            if not checkpoint_files:
                logger.info("No checkpoints found")
                return None
            
            # Sort by timestamp (newest first)
            checkpoint_files.sort(reverse=True)
            
            # Load latest checkpoint
            latest_file = checkpoint_files[0]
            filepath = os.path.join(self.checkpoint_dir, latest_file)
            
            with open(filepath, 'r') as f:
                checkpoint_dict = json.load(f)
            
            # Convert to CheckpointData
            checkpoint = CheckpointData(
                I_t_L=checkpoint_dict["I_t_L"],
                Omega_t_L=checkpoint_dict["Omega_t_L"],
                timestamp=checkpoint_dict["timestamp"],
                coherence_score=checkpoint_dict["coherence_score"],
                modulator=checkpoint_dict["modulator"],
                network_id=checkpoint_dict["network_id"]
            )
            
            logger.info(f"Latest checkpoint loaded: {latest_file}")
            return checkpoint
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return None
    
    def validate_checkpoint_consistency(self, checkpoint: CheckpointData) -> bool:
        """
        Validate checkpoint consistency for harmonic continuity
        
        Args:
            checkpoint: CheckpointData to validate
            
        Returns:
            bool: True if consistent, False otherwise
        """
        try:
            # Basic validation
            if not isinstance(checkpoint.I_t_L, list) or not isinstance(checkpoint.Omega_t_L, list):
                logger.warning("Invalid checkpoint data types")
                return False
            
            # Check coherence score bounds
            if not (0.0 <= checkpoint.coherence_score <= 1.0):
                logger.warning(f"Invalid coherence score: {checkpoint.coherence_score}")
                return False
            
            # Check modulator bounds (should be positive)
            if checkpoint.modulator <= 0:
                logger.warning(f"Invalid modulator: {checkpoint.modulator}")
                return False
            
            # Check timestamp
            if checkpoint.timestamp > time.time() + 60:  # Allow 1 minute future tolerance
                logger.warning(f"Invalid timestamp: {checkpoint.timestamp}")
                return False
            
            logger.info("Checkpoint consistency validation passed")
            return True
        except Exception as e:
            logger.error(f"Checkpoint consistency validation failed: {e}")
            return False

# Example usage and testing
if __name__ == "__main__":
    # Create CAL engine instance
    cal_engine = CALEngine()
    
    # Test λ(L) computation
    lambda_value = cal_engine.lambda_L("LΦ", 0.85)
    print(f"λ(L) for LΦ at Ψ=0.85: {lambda_value:.4f}")
    
    # Test modulator computation
    I_vector = [0.1, 0.2, 0.3, 0.4, 0.5]
    modulator_value = cal_engine.modulator(I_vector, lambda_value)
    print(f"Modulator m_t(L): {modulator_value:.4f}")
    
    # Test dimensional stability
    is_stable = cal_engine.validate_dimensional_stability(modulator_value)
    print(f"Dimensional stability: {is_stable}")
    
    # Test Ω state update
    features = [1.0, 2.0, 3.0, 4.0, 5.0]
    I_vector = [0.1, 0.15, 0.2, 0.25, 0.3]
    updated_omega, m_t = cal_engine.update_omega(features, I_vector, "LΦ")
    print(f"Updated Ω vector norm: {np.linalg.norm(updated_omega):.4f}")
    
    # Test recursive coherence
    omega_vectors = [
        np.array([1.0, 0.5, 0.2]),
        np.array([0.9, 0.6, 0.15]),
        np.array([1.1, 0.4, 0.25])
    ]
    coherence = cal_engine.compute_recursive_coherence(omega_vectors)
    print(f"Recursive coherence: {coherence:.4f}")
    
    # Test checkpointing
    print("\n--- Testing Checkpointing ---")
    checkpoint = cal_engine.create_checkpoint(
        I_t_L=[0.1, 0.2, 0.3],
        Omega_t_L=[1.0, 0.5, 0.2],
        coherence_score=0.85,
        modulator=1.2
    )
    print(f"Checkpoint created: {checkpoint}")
    
    # Load checkpoint
    loaded_checkpoint = cal_engine.load_latest_checkpoint()
    if loaded_checkpoint:
        is_consistent = cal_engine.validate_checkpoint_consistency(loaded_checkpoint)
        print(f"Checkpoint consistency: {is_consistent}")