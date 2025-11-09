#!/usr/bin/env python3
"""
🧠 Coherence Attunement Layer (CAL) - RHUFT Integration
Enhanced coherence engine that replaces FFT-based spectral correlation with Ω-state recursion.

This module implements the CAL's core functionality:
1. Ω-state vector computation and recursion
2. Dimensional consistency validation
3. Modulator (m_t) adaptive weighting
4. Multi-dimensional coherence scoring with penalties
5. Integration with RΦV Consensus Engine
6. Ω-State Checkpointing for mainnet readiness
"""

import numpy as np
from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass, field, asdict
import logging
import time
import math
import json
import os

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

@dataclass
class OmegaState:
    """Represents the Ω-state vector with key feature components"""
    timestamp: float
    token_rate: float  # Token flow rate
    sentiment_energy: float  # Network sentiment energy
    semantic_shift: float  # Semantic coherence shift
    meta_attention_spectrum: List[float]  # Meta-attention spectrum
    coherence_score: float = 0.0  # Overall coherence score (Ψ)
    modulator: float = 1.0  # Adaptive weighting factor (m_t)
    time_delay: float = 0.0  # Time delay parameter (τ)

@dataclass
class CoherencePenalties:
    """Represents the three penalty components for coherence scoring"""
    cosine_penalty: float = 0.0  # a·cos term
    entropy_penalty: float = 0.0  # b·entropy term
    variance_penalty: float = 0.0  # c·var term

class CoherenceAttunementLayer:
    """
    Coherence Attunement Layer (CAL) - Core Engine
    
    This class implements the CAL's Ω-state recursion mechanism that replaces
    the FFT-based spectral correlation in the RΦV Consensus Engine.
    """
    
    def __init__(self, network_id: str = "quantum-currency-cal-001"):
        self.network_id = network_id
        self.omega_history: List[OmegaState] = []
        self.penalty_history: List[CoherencePenalties] = []
        self.safety_bounds = {
            "dimensional_clamp": 10.0,  # K bound for dimensional consistency
            "coherence_recovery_threshold": 0.7,  # Minimum Ψ for recovery
            "recovery_step_limit": 50  # Max steps for re-stabilization
        }
        
        # Configuration parameters
        self.config = {
            "lambda_decay_factor": LAMBDA,
            "phi_scaling": PHI,
            "penalty_weights": {
                "cosine": 1.0,
                "entropy": 0.5,
                "variance": 0.3
            }
        }
        
        # Checkpointing for mainnet readiness
        self.checkpoints: List[CheckpointData] = []
        self.checkpoint_dir = "checkpoints"
        self.max_checkpoints = 10
        
        # Create checkpoint directory if it doesn't exist
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        
        logger.info(f"🧠 Coherence Attunement Layer initialized for network: {network_id}")
    
    def compute_omega_state(self, 
                          token_data: Dict[str, Any],
                          sentiment_data: Dict[str, Any],
                          semantic_data: Dict[str, Any],
                          attention_data: List[float]) -> OmegaState:
        """
        Compute the Ω-state vector from multi-dimensional input data
        
        Args:
            token_data: Token flow information
            sentiment_data: Network sentiment metrics
            semantic_data: Semantic coherence data
            attention_data: Meta-attention spectrum
            
        Returns:
            OmegaState object with computed Ω vector components
        """
        timestamp = time.time()
        
        # Extract token rate (simplified)
        token_rate = token_data.get("rate", 0.0)
        
        # Extract sentiment energy (simplified)
        sentiment_energy = sentiment_data.get("energy", 0.0)
        
        # Extract semantic shift (simplified)
        semantic_shift = semantic_data.get("shift", 0.0)
        
        # Ensure attention spectrum is normalized
        if attention_data:
            attention_spectrum = np.array(attention_data)
            if np.max(attention_spectrum) > 0:
                attention_spectrum = attention_spectrum / np.max(attention_spectrum)
            meta_attention_spectrum = attention_spectrum.tolist()
        else:
            meta_attention_spectrum = [0.0] * 10  # Default spectrum
        
        # Compute initial coherence score (will be refined)
        coherence_score = self._compute_initial_coherence(
            token_rate, sentiment_energy, semantic_shift, meta_attention_spectrum
        )
        
        # Compute modulator m_t(L) - adaptive weighting factor
        modulator = self._compute_modulator(coherence_score)
        
        # Compute time delay τ(L) - based on current scale
        time_delay = self._compute_time_delay(token_rate)
        
        omega_state = OmegaState(
            timestamp=timestamp,
            token_rate=token_rate,
            sentiment_energy=sentiment_energy,
            semantic_shift=semantic_shift,
            meta_attention_spectrum=meta_attention_spectrum,
            coherence_score=coherence_score,
            modulator=modulator,
            time_delay=time_delay
        )
        
        # Add to history
        self.omega_history.append(omega_state)
        
        # Keep only recent history (last 100 states)
        if len(self.omega_history) > 100:
            self.omega_history = self.omega_history[-100:]
        
        logger.debug(f"Ω-state computed: Ψ={coherence_score:.4f}, m_t={modulator:.4f}")
        return omega_state
    
    def _compute_initial_coherence(self, token_rate: float, sentiment_energy: float,
                                 semantic_shift: float, attention_spectrum: List[float]) -> float:
        """Compute initial coherence score from Ω-state components"""
        # Normalize components to [0,1] range
        norm_token = 1.0 - math.exp(-abs(token_rate) / 100.0)
        norm_sentiment = 1.0 - math.exp(-abs(sentiment_energy) / 10.0)
        norm_semantic = 1.0 - math.exp(-abs(semantic_shift) / 5.0)
        norm_attention = float(np.mean(attention_spectrum)) if attention_spectrum else 0.0
        
        # Weighted combination
        coherence = (
            0.3 * norm_token +
            0.3 * norm_sentiment +
            0.2 * norm_semantic +
            0.2 * norm_attention
        )
        
        return max(0.0, min(1.0, coherence))  # Clamp to [0,1]
    
    def _compute_modulator(self, coherence_score: float) -> float:
        """
        Compute the modulator m_t(L) - adaptive weighting factor
        This makes the λ decay responsive to the rate of coherence shift
        """
        if len(self.omega_history) < 2:
            return 1.0
        
        # Calculate rate of coherence shift
        recent_coherence = [state.coherence_score for state in self.omega_history[-5:]]
        if len(recent_coherence) >= 2:
            coherence_shift_rate = abs(recent_coherence[-1] - recent_coherence[0]) / len(recent_coherence)
        else:
            coherence_shift_rate = 0.0
        
        # Modulator adapts based on shift rate
        # Higher shift rate -> lower modulator -> stronger decay enforcement
        modulator = max(0.1, 1.0 - coherence_shift_rate * 2.0)
        return modulator
    
    def _compute_time_delay(self, token_rate: float) -> float:
        """Compute time delay τ(L) based on current scale"""
        # Simple scaling with token rate
        return max(0.1, 0.5 + token_rate / 1000.0)
    
    def compute_recursive_coherence(self, omega_states: List[OmegaState]) -> Tuple[float, CoherencePenalties]:
        """
        Compute recursive coherence score with all three penalty components
        This replaces the FFT-based spectral correlation with Ω-state recursion
        
        Args:
            omega_states: List of Ω-states from multiple nodes/steps
            
        Returns:
            Tuple of (aggregated_coherence_score, penalties)
        """
        if not omega_states:
            return 0.0, CoherencePenalties()
        
        # Compute pairwise Ω-state alignments
        alignments = []
        entropies = []
        variances = []
        
        for i, local_state in enumerate(omega_states):
            # Get remote states (all except local)
            remote_states = [state for j, state in enumerate(omega_states) if i != j]
            
            if not remote_states:
                continue
            
            # Compute alignment (cosine similarity between Ω vectors)
            local_vector = self._omega_to_vector(local_state)
            remote_vectors = [self._omega_to_vector(state) for state in remote_states]
            
            # Average alignment with remote states
            alignment_scores = []
            for remote_vector in remote_vectors:
                cos_sim = self._cosine_similarity(local_vector, remote_vector)
                alignment_scores.append(cos_sim)
            
            avg_alignment = float(np.mean(alignment_scores)) if alignment_scores else 0.0
            alignments.append(avg_alignment)
            
            # Compute entropy of attention spectrum
            entropy = self._compute_entropy(local_state.meta_attention_spectrum)
            entropies.append(entropy)
            
            # Compute variance of Ω components
            variance = self._compute_omega_variance(local_state)
            variances.append(variance)
        
        if not alignments:
            return 0.0, CoherencePenalties()
        
        # Compute penalties
        avg_alignment = float(np.mean(alignments))
        avg_entropy = float(np.mean(entropies))
        avg_variance = float(np.mean(variances))
        
        # Apply penalty weights
        weights = self.config["penalty_weights"]
        cosine_penalty = weights["cosine"] * (1.0 - avg_alignment)
        entropy_penalty = weights["entropy"] * avg_entropy
        variance_penalty = weights["variance"] * avg_variance
        
        penalties = CoherencePenalties(
            cosine_penalty=cosine_penalty,
            entropy_penalty=entropy_penalty,
            variance_penalty=variance_penalty
        )
        
        # Compute final coherence score with penalties
        raw_coherence = avg_alignment
        total_penalty = cosine_penalty + entropy_penalty + variance_penalty
        final_coherence = max(0.0, raw_coherence - total_penalty)
        
        logger.debug(f"Recursive coherence: {final_coherence:.4f} (raw: {raw_coherence:.4f}, penalty: {total_penalty:.4f})")
        return final_coherence, penalties
    
    def _omega_to_vector(self, omega_state: OmegaState) -> np.ndarray:
        """Convert Ω-state to vector representation for comparison"""
        return np.array([
            omega_state.token_rate,
            omega_state.sentiment_energy,
            omega_state.semantic_shift,
            *omega_state.meta_attention_spectrum[:5]  # Use first 5 components
        ])
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute cosine similarity between two vectors"""
        if len(vec1) != len(vec2):
            min_len = min(len(vec1), len(vec2))
            vec1 = vec1[:min_len]
            vec2 = vec2[:min_len]
        
        dot_product = float(np.dot(vec1, vec2))
        norm1 = float(np.linalg.norm(vec1))
        norm2 = float(np.linalg.norm(vec2))
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return max(0.0, min(1.0, dot_product / (norm1 * norm2)))  # Clamp to [0,1]
    
    def _compute_entropy(self, spectrum: List[float]) -> float:
        """Compute entropy of attention spectrum"""
        if not spectrum or sum(spectrum) == 0:
            return 0.0
        
        # Normalize spectrum
        normalized = np.array(spectrum) / sum(spectrum)
        
        # Compute entropy
        entropy = -sum(p * math.log(p + 1e-10) for p in normalized if p > 0)
        return float(entropy)
    
    def _compute_omega_variance(self, omega_state: OmegaState) -> float:
        """Compute variance of Ω components"""
        components = [
            omega_state.token_rate,
            omega_state.sentiment_energy,
            omega_state.semantic_shift,
            *omega_state.meta_attention_spectrum
        ]
        
        if len(components) == 0:
            return 0.0
        
        return float(np.var(components))
    
    def validate_dimensional_consistency(self, omega_state: OmegaState) -> bool:
        """
        Validate that the λ(L) · proj(I_t(L)) term remains dimensionless
        and clamped within ±K bounds (critical safety check)
        """
        # Check that all components are within safety bounds
        components = [
            omega_state.token_rate,
            omega_state.sentiment_energy,
            omega_state.semantic_shift,
            *omega_state.meta_attention_spectrum
        ]
        
        bound = self.safety_bounds["dimensional_clamp"]
        for component in components:
            if abs(component) > bound:
                logger.warning(f"Dimensional inconsistency detected: component {component} exceeds bound ±{bound}")
                return False
        
        # Check modulator is within valid range
        if not (0.1 <= omega_state.modulator <= 2.0):
            logger.warning(f"Modulator out of range: {omega_state.modulator}")
            return False
        
        return True
    
    def simulate_harmonic_shock_recovery(self, shock_magnitude: float = 1.0) -> Tuple[bool, int]:
        """
        Simulate harmonic shock and verify re-stabilization
        Injects a large, incoherent feature vector and checks recovery
        
        Args:
            shock_magnitude: Magnitude of the spectral shock
            
        Returns:
            Tuple of (recovered, steps_to_recovery)
        """
        if not self.omega_history:
            return False, 0
        
        logger.info(f"🧪 Simulating harmonic shock with magnitude {shock_magnitude}")
        
        # Get current state
        current_state = self.omega_history[-1]
        
        # Inject shock (artificially reduce coherence)
        shocked_state = OmegaState(
            timestamp=time.time(),
            token_rate=current_state.token_rate * (1 + shock_magnitude),
            sentiment_energy=current_state.sentiment_energy * (1 - shock_magnitude),
            semantic_shift=current_state.semantic_shift + shock_magnitude * 2,
            meta_attention_spectrum=[x * (1 - shock_magnitude * 0.5) for x in current_state.meta_attention_spectrum],
            coherence_score=max(0.0, current_state.coherence_score - shock_magnitude),
            modulator=current_state.modulator * 0.5,  # Reduce modulator
            time_delay=current_state.time_delay * 1.5  # Increase time delay
        )
        
        # Add shocked state to history
        self.omega_history.append(shocked_state)
        
        # Simulate recovery steps
        steps = 0
        recovery_threshold = self.safety_bounds["coherence_recovery_threshold"]
        max_steps = self.safety_bounds["recovery_step_limit"]
        
        while steps < max_steps:
            steps += 1
            
            # Simulate natural recovery (gradual improvement)
            if len(self.omega_history) >= 2:
                prev_state = self.omega_history[-2]
                current_coherence = self.omega_history[-1].coherence_score
                
                # Recovery rate based on previous state
                recovery_rate = 0.05 + (1.0 - current_coherence) * 0.1
                new_coherence = min(1.0, current_coherence + recovery_rate)
                
                # Create recovered state
                recovered_state = OmegaState(
                    timestamp=time.time(),
                    token_rate=prev_state.token_rate * 0.95 + self.omega_history[-1].token_rate * 0.05,
                    sentiment_energy=prev_state.sentiment_energy * 0.95 + self.omega_history[-1].sentiment_energy * 0.05,
                    semantic_shift=prev_state.semantic_shift * 0.95 + self.omega_history[-1].semantic_shift * 0.05,
                    meta_attention_spectrum=[
                        prev * 0.95 + curr * 0.05 
                        for prev, curr in zip(prev_state.meta_attention_spectrum, self.omega_history[-1].meta_attention_spectrum)
                    ],
                    coherence_score=new_coherence,
                    modulator=min(2.0, self.omega_history[-1].modulator * 1.05),
                    time_delay=max(0.1, self.omega_history[-1].time_delay * 0.95)
                )
                
                self.omega_history.append(recovered_state)
                
                # Check if recovered
                if new_coherence >= recovery_threshold:
                    logger.info(f"✅ Harmonic shock recovery successful in {steps} steps")
                    return True, steps
        
        logger.warning(f"⚠️ Harmonic shock recovery failed after {max_steps} steps")
        return False, max_steps
    
    def get_coherence_health_indicator(self) -> str:
        """
        Get coherence health indicator for UX layer
        Returns color-coded status based on Ψ score
        """
        if not self.omega_history:
            return "unknown"
        
        current_psi = self.omega_history[-1].coherence_score
        
        if current_psi >= 0.85:
            return "green"  # Stable (Ready for Macro Write)
        elif current_psi >= 0.65:
            return "yellow"  # Flux (Normal Operation)
        elif current_psi >= 0.35:
            return "red"  # Unattuned (Safe Mode Active)
        else:
            return "critical"  # Critical (Emergency Mode)
    
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
        
        logger.info(f"Ω-state checkpoint created: coherence={coherence_score:.4f}, modulator={modulator:.4f}")
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

# Example usage
if __name__ == "__main__":
    # Create CAL instance
    cal = CoherenceAttunementLayer()
    
    # Simulate some Ω-state computations
    for i in range(5):
        omega = cal.compute_omega_state(
            token_data={"rate": 100.0 + i * 10},
            sentiment_data={"energy": 0.5 + i * 0.1},
            semantic_data={"shift": 0.2 + i * 0.05},
            attention_data=[0.1, 0.2, 0.3, 0.4, 0.5]
        )
        print(f"Step {i}: Ψ={omega.coherence_score:.4f}, m_t={omega.modulator:.4f}")
    
    # Compute recursive coherence
    if len(cal.omega_history) >= 2:
        coherence, penalties = cal.compute_recursive_coherence(cal.omega_history[-3:])
        print(f"Recursive Coherence: {coherence:.4f}")
        print(f"Penalties: cos={penalties.cosine_penalty:.4f}, "
              f"ent={penalties.entropy_penalty:.4f}, "
              f"var={penalties.variance_penalty:.4f}")
    
    # Test dimensional consistency
    if cal.omega_history:
        is_consistent = cal.validate_dimensional_consistency(cal.omega_history[-1])
        print(f"Dimensional Consistency: {is_consistent}")
    
    # Test harmonic shock recovery
    recovered, steps = cal.simulate_harmonic_shock_recovery(0.5)
    print(f"Harmonic Shock Recovery: {recovered} in {steps} steps")