#!/usr/bin/env python3
"""
LLM Adapter with Systemic Resonance Dampening
Down-weights external temperature via R_Ω proportional filter
"""

import numpy as np
import logging
from typing import Dict, Any, List
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMAdapter:
    """
    LLM Adapter with Systemic Resonance Dampening
    
    Implements:
    - Down-weighting of external temperature via R_Ω proportional filter
    - Temporal smoothing for stability
    """
    
    def __init__(self, smoothing_factor: float = 0.5):
        self.smoothing_factor = smoothing_factor
        self.temperature_history: List[float] = []
        self.dampened_temperature_history: List[float] = []
        self.R_Omega_history: List[np.ndarray] = []
        
        logger.info("LLM Adapter initialized with resonance dampening")
    
    def apply_resonance_dampening(self, T_external: float, R_Omega_tensor: np.ndarray) -> float:
        """
        Apply resonance dampening to external temperature using R_Ω proportional filter
        
        Args:
            T_external: External temperature from LLM
            R_Omega_tensor: Resonant Curvature Tensor
            
        Returns:
            float: Dampened temperature
        """
        try:
            # Calculate R_Ω magnitude as dampening factor
            R_Omega_magnitude = np.linalg.norm(R_Omega_tensor)
            
            # Normalize magnitude to dampening factor (0.1 to 1.0)
            # Higher curvature = more dampening
            dampening_factor = max(0.1, min(1.0, R_Omega_magnitude / 1e-60))
            
            # Apply proportional filter
            dampened_temperature = T_external * dampening_factor
            
            # Apply temporal smoothing
            smoothed_temperature = self._apply_temporal_smoothing(float(dampened_temperature))
            
            # Store in history
            self.temperature_history.append(T_external)
            self.dampened_temperature_history.append(smoothed_temperature)
            self.R_Omega_history.append(R_Omega_tensor.copy())
            
            # Keep only recent history
            if len(self.temperature_history) > 50:
                self.temperature_history = self.temperature_history[-50:]
                self.dampened_temperature_history = self.dampened_temperature_history[-50:]
                self.R_Omega_history = self.R_Omega_history[-50:]
            
            logger.debug(f"T_external: {T_external:.4f} -> T_dampened: {smoothed_temperature:.4f} "
                        f"(dampening factor: {dampening_factor:.4f})")
            
            return smoothed_temperature
        except Exception as e:
            logger.error(f"Error applying resonance dampening: {e}")
            # Return original temperature on error
            return T_external
    
    def _apply_temporal_smoothing(self, temperature: float) -> float:
        """
        Apply temporal smoothing for stability
        
        Args:
            temperature: Current temperature value
            
        Returns:
            float: Smoothed temperature
        """
        if not self.dampened_temperature_history:
            return float(temperature)
        
        # Exponential moving average
        last_smoothed = self.dampened_temperature_history[-1]
        smoothed = self.smoothing_factor * float(temperature) + (1 - self.smoothing_factor) * last_smoothed
        
        return float(smoothed)
    
    def get_dampening_statistics(self) -> Dict[str, Any]:
        """
        Get dampening statistics
        
        Returns:
            Dict with dampening statistics
        """
        if not self.dampened_temperature_history:
            return {
                "samples": 0,
                "avg_dampening_factor": 1.0,
                "min_temperature": 0.0,
                "max_temperature": 0.0
            }
        
        # Calculate average dampening factor
        if len(self.temperature_history) == len(self.dampened_temperature_history):
            dampening_factors = [
                dampened / original if original > 0 else 1.0
                for dampened, original in zip(self.dampened_temperature_history, self.temperature_history)
            ]
            avg_dampening = np.mean(dampening_factors)
        else:
            avg_dampening = 1.0
        
        return {
            "samples": len(self.dampened_temperature_history),
            "avg_dampening_factor": float(avg_dampening),
            "min_temperature": float(min(self.dampened_temperature_history)),
            "max_temperature": float(max(self.dampened_temperature_history)),
            "current_temperature": float(self.dampened_temperature_history[-1]) if self.dampened_temperature_history else 0.0
        }

# Example usage
if __name__ == "__main__":
    # Create LLM adapter
    llm_adapter = LLMAdapter(smoothing_factor=0.3)
    
    # Example R_Ω tensor
    R_Omega = np.array([
        [2.0e-62, 0, 0, 0],
        [0, 2.0e-62, 0, 0],
        [0, 0, 2.0e-62, 0],
        [0, 0, 0, 2.0e-62]
    ])
    
    # Test with various external temperatures
    external_temperatures = [0.8, 0.9, 1.0, 1.1, 1.2]
    
    for T_ext in external_temperatures:
        T_dampened = llm_adapter.apply_resonance_dampening(T_ext, R_Omega)
        print(f"T_external: {T_ext:.3f} -> T_dampened: {T_dampened:.3f}")
    
    # Print statistics
    stats = llm_adapter.get_dampening_statistics()
    print(f"\nDampening Statistics: {stats}")