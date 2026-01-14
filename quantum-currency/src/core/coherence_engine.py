#!/usr/bin/env python3
"""
Coherence Engine with Modulator Clamp Response
Implements recursive stabilization and telemetry synchronization
"""

import numpy as np
import logging
import time
from typing import List, Dict, Any, Optional
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from core.gating_service import GatingService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CoherenceEngine:
    """
    Coherence Engine with Modulator Clamp Response
    
    Implements:
    - Modulator clamp response when Safe Mode is active
    - Telemetry synchronization
    - Recursive stabilization
    """
    
    def __init__(self, gating_service: GatingService):
        self.gating_service = gating_service
        self.omega_history: List[np.ndarray] = []
        self.modulator_history: List[float] = []
        self.telemetry_feed: Dict[str, Any] = {}
        self.safe_mode_active = False
        
        logger.info("Coherence Engine initialized")
    
    def calculate_modulator(self, I_vector: np.ndarray, lambda_L: float, K: float = 10.0) -> float:
        """
        Calculate modulator with potential clamping in Safe Mode
        
        Args:
            I_vector: Integrated feedback vector
            lambda_L: Adaptive decay factor
            K: Clamping bound
            
        Returns:
            float: Modulator value
        """
        try:
            # Check if Safe Mode is active
            if self.gating_service.get_safe_mode_status():
                logger.warning("Safe Mode active - clamping modulator to 0")
                return 0.0
            
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
            modulator_value = np.exp(clamped_product)
            
            # Store in history
            self.modulator_history.append(modulator_value)
            if len(self.modulator_history) > 100:
                self.modulator_history = self.modulator_history[-100:]
            
            logger.debug(f"Modulator calculated: {modulator_value:.4f}")
            return float(modulator_value)
        except Exception as e:
            logger.error(f"Error calculating modulator: {e}")
            return 1.0  # Return neutral value on error
    
    def update_omega(self, features: np.ndarray, I_vector: np.ndarray, L: str) -> np.ndarray:
        """
        Update Ω state vector
        
        Args:
            features: Feature vector
            I_vector: Integrated feedback vector
            L: Scale level
            
        Returns:
            np.ndarray: Updated Ω vector
        """
        try:
            # Normalize features
            if len(features) > 0 and np.linalg.norm(features) > 0:
                normalized_features = features / np.linalg.norm(features)
            else:
                normalized_features = features
            
            # Calculate lambda_L (simplified for demonstration)
            lambda_L = 0.618  # Golden ratio reciprocal
            
            # Calculate modulator with potential clamping
            modulator = self.calculate_modulator(I_vector, lambda_L)
            
            # Update Ω_t(L) = Normalize(F_t) × m_t(L)
            updated_omega = normalized_features * modulator if len(normalized_features) > 0 else np.array([])
            
            # Store in history
            self.omega_history.append(updated_omega)
            if len(self.omega_history) > 10:
                self.omega_history = self.omega_history[-10:]
            
            logger.debug(f"Ω state updated for scale {L}: norm={np.linalg.norm(updated_omega):.4f}")
            return updated_omega
        except Exception as e:
            logger.error(f"Error updating Ω state: {e}")
            return np.array([])
    
    def synchronize_telemetry(self, metrics: Dict[str, Any]):
        """
        Synchronize with telemetry feed
        
        Args:
            metrics: Dictionary of telemetry metrics
        """
        try:
            self.telemetry_feed.update(metrics)
            
            # Check Safe Mode status and broadcast
            safe_mode_status = self.gating_service.get_safe_mode_status()
            if safe_mode_status != self.safe_mode_active:
                self.safe_mode_active = safe_mode_status
                self._broadcast_clamp_state(safe_mode_status)
                
            logger.debug(f"Telemetry synchronized - Safe Mode: {safe_mode_status}")
        except Exception as e:
            logger.error(f"Error synchronizing telemetry: {e}")
    
    def _broadcast_clamp_state(self, safe_mode_active: bool):
        """
        Broadcast clamp state to all subsystems
        
        Args:
            safe_mode_active: Current Safe Mode status
        """
        try:
            broadcast_data = {
                "timestamp": time.time(),
                "safe_mode": safe_mode_active,
                "clamp_state": "ACTIVE" if safe_mode_active else "RELEASED",
                "lambda_clamped": safe_mode_active
            }
            
            # In a real implementation, this would broadcast to all connected subsystems
            logger.info(f"Clamp state broadcast: {broadcast_data}")
        except Exception as e:
            logger.error(f"Error broadcasting clamp state: {e}")
    
    def get_coherence_status(self) -> Dict[str, Any]:
        """
        Get current coherence status
        
        Returns:
            Dict with coherence status information
        """
        return {
            "omega_history_length": len(self.omega_history),
            "modulator_history_length": len(self.modulator_history),
            "safe_mode_active": self.safe_mode_active,
            "last_modulator": self.modulator_history[-1] if self.modulator_history else 1.0,
            "omega_norm": float(np.linalg.norm(self.omega_history[-1])) if self.omega_history else 0.0,
            "timestamp": time.time()
        }

# Example usage
if __name__ == "__main__":
    # Create gating service
    gating_service = GatingService()
    
    # Create coherence engine
    coherence_engine = CoherenceEngine(gating_service)
    
    # Example vectors
    features = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    I_vector = np.array([0.1, 0.15, 0.2, 0.25, 0.3])
    
    # Update Ω state
    updated_omega = coherence_engine.update_omega(features, I_vector, "L_Phi")
    print(f"Updated Ω norm: {np.linalg.norm(updated_omega):.4f}")
    
    # Test Safe Mode clamping
    gating_service.check_safe_mode_trigger(0.5, 0.8)  # This should activate Safe Mode
    modulator_clamped = coherence_engine.calculate_modulator(I_vector, 0.618)
    print(f"Modulator with Safe Mode: {modulator_clamped:.4f}")