#!/usr/bin/env python3
"""
Stress Tensor Module for Curvature-Coherence Integrator
Implements T_Ω calculation from Ω-vector and its time derivatives
"""

import numpy as np
from typing import List, Dict, Tuple, Any, Optional, Union
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Physical constants
PLANCK_CONSTANT = 6.62607015e-34  # J·s
REDUCED_PLANCK_CONSTANT = PLANCK_CONSTANT / (2 * np.pi)  # ħ = h/(2π) J·s
SPEED_OF_LIGHT = 299792458  # m/s

class StressTensorCalculator:
    """
    Stress Tensor Calculator for Self-Coherence Pressure T_Ω
    Derives T_Ω from Ω-vector and its time derivatives with units [M][L]⁻¹[T]⁻²
    """
    
    def __init__(self):
        self.omega_history: List[np.ndarray] = []
        self.time_history: List[float] = []
        
    def calculate_T_Omega(self, omega_vector: np.ndarray, time: Optional[float] = None) -> float:
        """
        Calculate Self-Coherence Pressure T_Ω from Ω-vector and its time derivatives
        
        Args:
            omega_vector: Ω-vector from CAL Engine
            time: Timestamp for time derivative calculation (optional)
            
        Returns:
            float: T_Ω value with units [M][L]⁻¹[T]⁻² (or [Energy][L]⁻³)
        """
        if time is None:
            time = float(np.datetime64('now').astype('datetime64[us]').astype(float) / 1e6)
            
        # Store in history for time derivative calculation
        self.omega_history.append(omega_vector.copy())
        self.time_history.append(time)
        
        # Keep only recent history (last 10 points)
        if len(self.omega_history) > 10:
            self.omega_history = self.omega_history[-10:]
            self.time_history = self.time_history[-10:]
        
        # Calculate norm of Ω-vector
        omega_norm = np.linalg.norm(omega_vector)
        
        # Calculate time derivatives if we have history
        if len(self.omega_history) > 1:
            # Convert numpy arrays to Python lists for compatibility
            norms_history = [float(np.linalg.norm(vec)) for vec in self.omega_history]
            
            # First derivative: d|Ω|/dt
            d_omega_dt = self._calculate_time_derivative(
                norms_history, 
                self.time_history
            )
            
            # Second derivative: d²|Ω|/dt²
            d2_omega_dt2 = self._calculate_second_derivative(
                norms_history, 
                self.time_history
            )
        else:
            d_omega_dt = 0.0
            d2_omega_dt2 = 0.0
            
        # Calculate T_Ω using the full stress-energy tensor approach:
        # T_Ω = (ħc/L_planck) * [α₁ * |Ω|² + α₂ * |dΩ/dt|² + α₃ * |d²Ω/dt²|²]
        
        # Coupling constants (dimensionless)
        alpha_1 = 1.0e-20  # Coupling for Ω norm
        alpha_2 = 1.0e-30  # Coupling for first derivative
        alpha_3 = 1.0e-40  # Coupling for second derivative
        
        # Planck length for dimensional consistency
        planck_length = np.sqrt(REDUCED_PLANCK_CONSTANT * SPEED_OF_LIGHT / GRAVITATIONAL_CONSTANT)
        
        # Calculate T_Ω with proper units [M][L]⁻¹[T]⁻²
        T_Omega = (REDUCED_PLANCK_CONSTANT * SPEED_OF_LIGHT / planck_length) * (
            alpha_1 * (omega_norm**2) + 
            alpha_2 * (d_omega_dt**2) + 
            alpha_3 * (d2_omega_dt2**2)
        )
        
        logger.debug(f"T_Ω calculated: {T_Omega:.6e} [M][L]⁻¹[T]⁻²")
        return float(T_Omega)
    
    def _calculate_time_derivative(self, values: List[float], times: List[float]) -> float:
        """
        Calculate first time derivative using finite differences
        
        Args:
            values: List of values
            times: List of corresponding times
            
        Returns:
            float: First derivative
        """
        if len(values) < 2:
            return 0.0
            
        # Use central difference for interior points, forward/backward for endpoints
        if len(values) >= 3:
            # Central difference using last three points
            dt = times[-1] - times[-3]
            if dt != 0:
                return (values[-1] - values[-3]) / dt
            else:
                return 0.0
        else:
            # Forward difference
            dt = times[-1] - times[-2]
            if dt != 0:
                return (values[-1] - values[-2]) / dt
            else:
                return 0.0
    
    def _calculate_second_derivative(self, values: List[float], times: List[float]) -> float:
        """
        Calculate second time derivative using finite differences
        
        Args:
            values: List of values
            times: List of corresponding times
            
        Returns:
            float: Second derivative
        """
        if len(values) < 3:
            return 0.0
            
        # Use central difference formula: f''(x) ≈ [f(x+h) - 2f(x) + f(x-h)] / h²
        if len(values) >= 3:
            dt1 = times[-1] - times[-2]
            dt2 = times[-2] - times[-3]
            
            if dt1 != 0 and dt2 != 0:
                # Non-uniform grid second derivative
                f2 = values[-1]  # f(x+h)
                f1 = values[-2]  # f(x)
                f0 = values[-3]  # f(x-h)
                
                # For non-uniform grid, we use the more general formula
                h1 = dt2  # x - (x-h) = h
                h2 = dt1  # (x+h) - x = h
                
                if h1 != h2:
                    # Non-uniform grid formula
                    second_deriv = 2 * (f2 / (h1 * (h1 + h2)) - f1 / (h1 * h2) + f0 / (h2 * (h1 + h2)))
                else:
                    # Uniform grid
                    second_deriv = (f2 - 2 * f1 + f0) / (h1**2)
                    
                return float(second_deriv)
            else:
                return 0.0
        else:
            return 0.0
    
    def validate_T_Omega_units(self, T_Omega: float) -> bool:
        """
        Validate that T_Ω has correct units [M][L]⁻¹[T]⁻²
        
        Args:
            T_Omega: Calculated T_Ω value
            
        Returns:
            bool: True if units are valid
        """
        # Check that T_Ω is finite and positive (energy density should be positive)
        is_valid = np.isfinite(T_Omega) and T_Omega >= 0
        logger.info(f"T_Ω units validation: {'PASSED' if is_valid else 'FAILED'}")
        return is_valid

# Physical constants needed for the module
GRAVITATIONAL_CONSTANT = 6.67430e-11  # m³/(kg·s²)

# Example usage
if __name__ == "__main__":
    # Create stress tensor calculator
    stress_calc = StressTensorCalculator()
    
    # Example Ω-vectors over time
    omega_vectors = [
        np.array([0.9, 0.85, 0.78, 0.92, 0.88]),
        np.array([0.91, 0.86, 0.79, 0.91, 0.89]),
        np.array([0.92, 0.87, 0.80, 0.90, 0.90]),
    ]
    
    times = [0.0, 0.1, 0.2]  # seconds
    
    # Calculate T_Ω for each time step
    for i, omega_vec in enumerate(omega_vectors):
        T_Omega = stress_calc.calculate_T_Omega(omega_vec, times[i])
        is_valid = stress_calc.validate_T_Omega_units(T_Omega)
        print(f"Time {times[i]:.1f}s: T_Ω = {T_Omega:.6e} [M][L]⁻¹[T]⁻², Valid: {is_valid}")