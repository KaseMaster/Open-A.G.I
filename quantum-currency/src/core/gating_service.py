#!/usr/bin/env python3
"""
Gating Service for Global Curvature Resonance System
Implements GAS calculation, memory locking, and safe mode triggers
"""

import numpy as np
import logging
import time
from typing import Dict, Any, Optional
import json
import os
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class GatingMetrics:
    """Metrics for gating service"""
    gas: float  # Geometric Alignment Score
    cs: float   # Coherence Stability
    rsi: float  # Recursive Stability Index
    safe_mode_active: bool
    timestamp: float

class GatingService:
    """
    Gating Service for Global Curvature Resonance System
    
    Implements:
    - Geometric Alignment Score (GAS) calculation
    - Memory write locking based on dual thresholds
    - Safe Mode triggering for critical conditions
    """
    
    def __init__(self, log_file: str = "logs/safe_mode.log"):
        self.log_file = log_file
        self.safe_mode_active = False
        self.memory_locked = False
        self.last_metrics = GatingMetrics(0.0, 0.0, 0.0, False, 0.0)
        
        # Create logs directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        logger.info("Gating Service initialized")
    
    def calculate_GAS(self, R_Omega_tensor: np.ndarray, geometric_eigenvalues: Dict[str, Any]) -> float:
        """
        Calculate Geometric Alignment Score (GAS) from R_Ω tensor and geometric eigenvalues
        
        Args:
            R_Omega_tensor: Resonant Curvature Tensor
            geometric_eigenvalues: Dictionary with quantum numbers Q(n, ℓ, m, s)
            
        Returns:
            float: Geometric Alignment Score (0.0 to 1.0)
        """
        try:
            # Calculate tensor norm as base alignment score
            tensor_norm = np.linalg.norm(R_Omega_tensor)
            
            # Extract quantum numbers
            n = geometric_eigenvalues.get('n', 1)
            l = geometric_eigenvalues.get('l', 0)
            m = geometric_eigenvalues.get('m', 0)
            s = geometric_eigenvalues.get('s', 0.5)
            
            # Calculate alignment based on quantum number harmonics
            # Higher quantum numbers should align better with curvature
            quantum_alignment = np.sqrt(n**2 + l**2 + abs(m)**2 + s**2) / 10.0
            
            # Normalize to 0-1 range - ensure we don't divide by zero
            # Use a more appropriate normalization factor based on expected tensor magnitudes
            normalization_factor = 1e-62  # Expected typical magnitude
            if tensor_norm > 0:
                gas = min(1.0, (tensor_norm * quantum_alignment) / normalization_factor)
            else:
                gas = 0.0
            
            logger.debug(f"GAS calculated: {gas:.4f}")
            return gas
        except Exception as e:
            logger.error(f"Error calculating GAS: {e}")
            return 0.0
    
    def apply_memory_lock(self, coherence_stability: float, gas: float) -> bool:
        """
        Apply memory write lock based on dual thresholds
        
        Args:
            coherence_stability: Current coherence stability (CS)
            gas: Geometric Alignment Score
            
        Returns:
            bool: True if memory locked, False otherwise
        """
        # Check dual thresholds: BOTH CS ≥ 0.80 AND GAS ≥ 0.99 must be met for unlocked state
        # Lock if either condition is not met
        should_lock = coherence_stability < 0.80 or gas < 0.99
        
        if should_lock and not self.memory_locked:
            self.memory_locked = True
            self._log_lock_event("MEMORY_LOCK_ACTIVATED", coherence_stability, gas)
            logger.warning("Memory write lock activated")
        elif not should_lock and self.memory_locked:
            self.memory_locked = False
            self._log_lock_event("MEMORY_LOCK_RELEASED", coherence_stability, gas)
            logger.info("Memory write lock released")
            
        return self.memory_locked
    
    def check_safe_mode_trigger(self, rsi: float, gas: float) -> bool:
        """
        Check if Safe Mode should be triggered
        
        Args:
            rsi: Recursive Stability Index
            gas: Geometric Alignment Score
            
        Returns:
            bool: True if Safe Mode activated, False otherwise
        """
        # Safe Mode triggers: RSI < 0.65 or GAS < 0.90
        should_trigger = rsi < 0.65 or gas < 0.90
        
        if should_trigger and not self.safe_mode_active:
            self.safe_mode_active = True
            self._log_lock_event("SAFE_MODE_ACTIVATED", rsi, gas)
            logger.critical("Safe Mode activated - recursion clamped")
        elif not should_trigger and self.safe_mode_active:
            self.safe_mode_active = False
            self._log_lock_event("SAFE_MODE_DEACTIVATED", rsi, gas)
            logger.info("Safe Mode deactivated")
            
        return self.safe_mode_active
    
    def _log_lock_event(self, event_type: str, metric1: float, metric2: float):
        """
        Log lock events to safe mode log file
        
        Args:
            event_type: Type of event
            metric1: First metric value
            metric2: Second metric value
        """
        try:
            log_entry = {
                "timestamp": time.time(),
                "event_type": event_type,
                "metric1": metric1,
                "metric2": metric2,
                "safe_mode": self.safe_mode_active,
                "memory_locked": self.memory_locked
            }
            
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
        except Exception as e:
            logger.error(f"Error logging lock event: {e}")
    
    def get_current_metrics(self, R_Omega_tensor: np.ndarray, 
                          geometric_eigenvalues: Dict[str, Any],
                          coherence_stability: float,
                          rsi: float) -> GatingMetrics:
        """
        Get current gating metrics
        
        Args:
            R_Omega_tensor: Resonant Curvature Tensor
            geometric_eigenvalues: Dictionary with quantum numbers
            coherence_stability: Current coherence stability
            rsi: Recursive Stability Index
            
        Returns:
            GatingMetrics: Current metrics
        """
        gas = self.calculate_GAS(R_Omega_tensor, geometric_eigenvalues)
        self.apply_memory_lock(coherence_stability, gas)
        self.check_safe_mode_trigger(rsi, gas)
        
        metrics = GatingMetrics(
            gas=gas,
            cs=coherence_stability,
            rsi=rsi,
            safe_mode_active=self.safe_mode_active,
            timestamp=time.time()
        )
        
        self.last_metrics = metrics
        return metrics
    
    def get_safe_mode_status(self) -> bool:
        """Get current Safe Mode status"""
        return self.safe_mode_active
    
    def get_memory_lock_status(self) -> bool:
        """Get current memory lock status"""
        return self.memory_locked

# Example usage
if __name__ == "__main__":
    # Create gating service
    gating_service = GatingService()
    
    # Example R_Ω tensor (4x4)
    R_Omega = np.array([
        [1.0e-62, 0, 0, 0],
        [0, 1.0e-62, 0, 0],
        [0, 0, 1.0e-62, 0],
        [0, 0, 0, 1.0e-62]
    ])
    
    # Example geometric eigenvalues
    eigenvalues = {
        'n': 3,
        'l': 2,
        'm': 1,
        's': 0.5
    }
    
    # Calculate metrics
    metrics = gating_service.get_current_metrics(
        R_Omega_tensor=R_Omega,
        geometric_eigenvalues=eigenvalues,
        coherence_stability=0.95,
        rsi=0.85
    )
    
    print(f"GAS: {metrics.gas:.4f}")
    print(f"CS: {metrics.cs:.4f}")
    print(f"RSI: {metrics.rsi:.4f}")
    print(f"Safe Mode: {metrics.safe_mode_active}")