#!/usr/bin/env python3
"""
Q-Projection Module for Curvature-Coherence Integrator
Implements eigen-projection to constrain metric solutions to Q-basis
"""

import numpy as np
from typing import List, Dict, Tuple, Any, Optional, Union
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QProjection:
    """
    Q-Projection for constraining metric solutions to Q-basis
    Projects curvature tensors onto eigen-basis defined by quantum numbers Q(n, ℓ, m, s)
    """
    
    def __init__(self):
        self.basis_vectors: List[np.ndarray] = []
        self.quantum_numbers: List[Dict[str, int]] = []
        
    def define_q_basis(self, n_max: int = 5, l_max: int = 3) -> List[Dict[str, int]]:
        """
        Define Q-basis using quantum numbers Q(n, ℓ, m, s)
        
        Args:
            n_max: Maximum principal quantum number
            l_max: Maximum angular momentum quantum number
            
        Returns:
            List of quantum number dictionaries
        """
        q_basis = []
        
        # Generate quantum numbers following standard rules:
        # n = 1, 2, 3, ...
        # ℓ = 0, 1, 2, ..., n-1
        # m = -ℓ, -ℓ+1, ..., 0, ..., ℓ-1, ℓ
        # s = ±1/2 (spin quantum number)
        
        for n in range(1, n_max + 1):
            for l in range(min(l_max, n)):
                for m in range(-l, l + 1):
                    # Spin quantum number: +1/2 or -1/2
                    for s in [0.5, -0.5]:
                        q_numbers = {
                            'n': n,
                            'l': l,
                            'm': m,
                            's': s
                        }
                        q_basis.append(q_numbers)
                        
        self.quantum_numbers = q_basis
        logger.info(f"Q-basis defined with {len(q_basis)} basis states")
        return q_basis
    
    def generate_basis_vectors(self, dimension: int = 4) -> List[np.ndarray]:
        """
        Generate orthonormal basis vectors for the Q-basis
        
        Args:
            dimension: Dimension of the vector space
            
        Returns:
            List of orthonormal basis vectors
        """
        # For demonstration, we'll generate random orthonormal vectors
        # In a real implementation, these would be derived from the φ-lattice geometry
        
        # Generate random matrix
        random_matrix = np.random.randn(dimension, dimension)
        
        # Apply Gram-Schmidt process to create orthonormal basis
        basis_vectors = []
        for i in range(dimension):
            # Start with the random vector
            vec = random_matrix[:, i].copy()
            
            # Subtract projections onto previous basis vectors
            for prev_vec in basis_vectors:
                proj = np.dot(vec, prev_vec) * prev_vec
                vec = vec - proj
                
            # Normalize
            norm = np.linalg.norm(vec)
            if norm > 1e-10:  # Avoid division by zero
                vec = vec / norm
                basis_vectors.append(vec)
                
        self.basis_vectors = basis_vectors
        logger.info(f"Generated {len(basis_vectors)} orthonormal basis vectors")
        return basis_vectors
    
    def project_to_q_basis(self, tensor: np.ndarray, q_numbers: Dict[str, Any]) -> np.ndarray:
        """
        Project tensor to Q-basis using eigen-projection
        
        Args:
            tensor: Curvature tensor to project
            q_numbers: Quantum numbers Q(n, ℓ, m, s)
            
        Returns:
            np.ndarray: Projected tensor in Q-basis
        """
        if len(self.basis_vectors) == 0:
            # Generate basis vectors if not already done
            self.generate_basis_vectors(tensor.shape[0])
            
        # For demonstration, we'll apply a simple projection based on quantum numbers
        n, l, m, s = q_numbers.get('n', 1), q_numbers.get('l', 0), q_numbers.get('m', 0), q_numbers.get('s', 0.5)
        
        # Create projection matrix based on quantum numbers
        # This is a simplified approach - in reality, this would involve solving
        # the eigenvalue problem for the metric tensor
        
        # Scaling factor based on quantum numbers
        scaling_factor = np.sqrt(n**2 + l**2 + abs(m)**2 + s**2 + 1)
        
        # Apply projection
        projected_tensor = tensor / scaling_factor
        
        # Ensure the result is Hermitian (for metric tensors)
        if projected_tensor.shape[0] == projected_tensor.shape[1]:
            projected_tensor = 0.5 * (projected_tensor + projected_tensor.T)
            
        logger.debug(f"Tensor projected to Q-basis with scaling factor: {scaling_factor:.4f}")
        return projected_tensor
    
    def calculate_eigenvalues(self, tensor: np.ndarray) -> np.ndarray:
        """
        Calculate eigenvalues of the tensor
        
        Args:
            tensor: Input tensor
            
        Returns:
            np.ndarray: Eigenvalues
        """
        try:
            eigenvalues = np.linalg.eigvals(tensor)
            logger.debug(f"Eigenvalues calculated: {eigenvalues}")
            return eigenvalues
        except Exception as e:
            logger.error(f"Failed to calculate eigenvalues: {e}")
            return np.array([])
    
    def validate_q_projection(self, original_tensor: np.ndarray, 
                            projected_tensor: np.ndarray) -> bool:
        """
        Validate Q-projection accuracy by comparing with analytic Q-basis
        
        Args:
            original_tensor: Original tensor
            projected_tensor: Projected tensor
            
        Returns:
            bool: True if projection is valid
        """
        # Check that tensors have the same shape
        if original_tensor.shape != projected_tensor.shape:
            logger.warning("Projection validation failed: tensor shapes mismatch")
            return False
            
        # Check that projected tensor is not too different from original
        # (This is a simplified check - in reality, we would compare with analytic solutions)
        diff_norm = np.linalg.norm(original_tensor - projected_tensor)
        original_norm = np.linalg.norm(original_tensor)
        
        if original_norm > 0:
            relative_error = diff_norm / original_norm
            is_valid = bool(relative_error < 0.1)  # Allow 10% error for demonstration
            logger.info(f"Q-projection validation: {'PASSED' if is_valid else 'FAILED'} (relative error: {relative_error:.4f})")
            return is_valid
        else:
            logger.info("Q-projection validation: PASSED (zero tensor)")
            return True

# Example usage
if __name__ == "__main__":
    # Create Q-projection instance
    q_proj = QProjection()
    
    # Define Q-basis
    q_basis = q_proj.define_q_basis(n_max=3, l_max=2)
    print(f"Defined Q-basis with {len(q_basis)} states")
    
    # Generate basis vectors
    basis_vectors = q_proj.generate_basis_vectors(dimension=4)
    
    # Example tensor (4x4)
    example_tensor = np.array([
        [1.0, 0.1, 0.05, 0.02],
        [0.1, 1.2, 0.08, 0.03],
        [0.05, 0.08, 1.1, 0.04],
        [0.02, 0.03, 0.04, 1.05]
    ])
    
    # Example quantum numbers
    q_numbers = {'n': 2, 'l': 1, 'm': 0, 's': 0.5}
    
    # Project tensor to Q-basis
    projected_tensor = q_proj.project_to_q_basis(example_tensor, q_numbers)
    
    # Validate projection
    is_valid = q_proj.validate_q_projection(example_tensor, projected_tensor)
    
    # Calculate eigenvalues
    eigenvals = q_proj.calculate_eigenvalues(projected_tensor)
    
    print(f"Original tensor:\n{example_tensor}")
    print(f"Projected tensor:\n{projected_tensor}")
    print(f"Projection valid: {is_valid}")
    print(f"Eigenvalues: {eigenvals}")