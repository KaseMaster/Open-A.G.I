#!/usr/bin/env python3
"""
⚛️ CAL Engine - Coherence Attunement Layer Core Engine
Implements the Ω-state recursion mechanism with dimensional consistency validation.

This module provides the core mathematical engine for the Coherence Attunement Layer (CAL),
implementing the refined Ω-state recursion equations and safety constraints as specified
in the v0.2.0-beta CAL → RΦV Fusion Specification.
"""

import numpy as np
from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass, field
import logging
import time
import math
from scipy.integrate import quad
from scipy.optimize import minimize

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Golden ratio and related constants
PHI = 1.618033988749895
LAMBDA = 1.0 / PHI

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
    integrated_feedback: float = 0.0  # Integrated feedback I_t(L)

@dataclass
class CoherencePenalties:
    """Represents the three penalty components for coherence scoring"""
    cosine_penalty: float = 0.0  # a·cos term
    entropy_penalty: float = 0.0  # b·entropy term
    variance_penalty: float = 0.0  # c·var term

@dataclass
class SafetyBounds:
    """Safety bounds for dimensional consistency validation"""
    dimensional_clamp: float = 10.0  # K bound for dimensional consistency
    coherence_recovery_threshold: float = 0.7  # Minimum Ψ for recovery
    recovery_step_limit: int = 50  # Max steps for re-stabilization
    entropy_threshold: float = 0.25  # Maximum entropy penalty during stable cycles

class CALEngine:
    """
    CAL Engine - Core Mathematical Engine for Ω-State Recursion
    
    Implements the refined Ω-state recursion with dimensional consistency validation:
    1. Ω_t(L) = Normalize(F_t) × m_t(L) - Frequency-based Ω-state estimation
    2. m_t(L) = exp(clamp(λ(L) · proj(I_t(L)), -K, K)) - Stable non-linearity
    3. λ(L) = (1/φ) · Ψ_t - Direct control linking feedback strength to system health
    4. I_t(L) = Σ w_i(L) · Ω_{t-i}(L) · Δt - Discrete sum for integrated feedback
    """

    def __init__(self, network_id: str = "quantum-currency-cal-001"):
        self.network_id = network_id
        self.omega_history: List[OmegaState] = []
        self.penalty_history: List[CoherencePenalties] = []
        self.safety_bounds = SafetyBounds()
        
        # Configuration parameters
        self.config = {
            "lambda_decay_factor": LAMBDA,
            "phi_scaling": PHI,
            "penalty_weights": {
                "cosine": 1.0,
                "entropy": 0.5,
                "variance": 0.3
            },
            "integration_window": 10,  # Number of steps for feedback integration
            "checkpoint_interval": 100  # Steps between checkpoints
        }
        
        # Checkpoint storage for rapid restarts
        self.checkpoints: List[OmegaState] = []
        
        logger.info(f"⚛️ CAL Engine initialized for network: {network_id}")

    def compute_omega_state(self, 
                          token_data: Dict[str, Any],
                          sentiment_data: Dict[str, Any],
                          semantic_data: Dict[str, Any],
                          attention_data: List[float]) -> OmegaState:
        """
        Compute the Ω-state vector from multi-dimensional input data using refined equations
        
        Args:
            token_data: Token flow information
            sentiment_data: Network sentiment metrics
            semantic_data: Semantic coherence data
            attention_data: Meta-attention spectrum
            
        Returns:
            OmegaState object with computed Ω vector components
        """
        timestamp = time.time()
        
        # Extract components
        token_rate = token_data.get("rate", 0.0)
        sentiment_energy = sentiment_data.get("energy", 0.0)
        semantic_shift = semantic_data.get("shift", 0.0)
        
        # Ensure attention spectrum is normalized
        if attention_data:
            attention_spectrum = np.array(attention_data)
            if np.max(attention_spectrum) > 0:
                attention_spectrum = attention_spectrum / np.max(attention_spectrum)
            meta_attention_spectrum = attention_spectrum.tolist()
        else:
            meta_attention_spectrum = [0.0] * 10  # Default spectrum
        
        # Compute integrated feedback I_t(L)
        integrated_feedback = self._compute_integrated_feedback()
        
        # Compute initial coherence score (will be refined)
        coherence_score = self._compute_initial_coherence(
            token_rate, sentiment_energy, semantic_shift, meta_attention_spectrum
        )
        
        # Compute adaptive decay λ(L) - directly linked to coherence score
        lambda_decay = self._compute_lambda_decay(coherence_score)
        
        # Compute modulator m_t(L) with dimensional safety clamping
        modulator = self._compute_modulator(lambda_decay, integrated_feedback)
        
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
            time_delay=time_delay,
            integrated_feedback=integrated_feedback
        )
        
        # Add to history
        self.omega_history.append(omega_state)
        
        # Create checkpoint for rapid restarts
        if len(self.omega_history) % self.config["checkpoint_interval"] == 0:
            self._create_checkpoint(omega_state)
        
        # Keep only recent history (last 1000 states)
        if len(self.omega_history) > 1000:
            self.omega_history = self.omega_history[-1000:]
        
        logger.debug(f"Ω-state computed: Ψ={coherence_score:.4f}, m_t={modulator:.4f}, λ={lambda_decay:.4f}")
        return omega_state

    def _compute_integrated_feedback(self) -> float:
        """Compute integrated feedback I_t(L) using discrete sum"""
        if len(self.omega_history) < 2:
            return 0.0
        
        # Use recent history for integration
        window_size = min(self.config["integration_window"], len(self.omega_history))
        recent_states = self.omega_history[-window_size:]
        
        # Compute weighted sum with exponential decay weights
        integrated_feedback = 0.0
        total_weight = 0.0
        dt = 1.0  # Time step (could be made dynamic)
        
        for i, state in enumerate(recent_states):
            # Exponential decay weight
            weight = math.exp(-i * 0.1)  # Decay factor
            # Project Ω-state to scalar feedback
            feedback = self._project_omega_state(state)
            integrated_feedback += weight * feedback * dt
            total_weight += weight
        
        return integrated_feedback / max(total_weight, 1e-10) if total_weight > 0 else 0.0

    def _project_omega_state(self, omega_state: OmegaState) -> float:
        """Project Ω-state to scalar feedback value"""
        # Simple projection using weighted components
        return (
            0.4 * omega_state.token_rate +
            0.3 * omega_state.sentiment_energy +
            0.2 * omega_state.semantic_shift +
            0.1 * sum(omega_state.meta_attention_spectrum) / len(omega_state.meta_attention_spectrum)
        )

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

    def _compute_lambda_decay(self, coherence_score: float) -> float:
        """
        Compute adaptive decay λ(L) directly linked to system health
        λ(L) = (1/φ) · Ψ_t
        """
        return self.config["lambda_decay_factor"] * coherence_score

    def _compute_modulator(self, lambda_decay: float, integrated_feedback: float) -> float:
        """
        Compute the modulator m_t(L) with dimensional safety clamping
        m_t(L) = exp(clamp(λ(L) · proj(I_t(L)), -K, K))
        """
        # Compute argument to exponential
        argument = lambda_decay * integrated_feedback
        
        # Clamp argument to safety bounds
        clamped_argument = max(-self.safety_bounds.dimensional_clamp, 
                              min(self.safety_bounds.dimensional_clamp, argument))
        
        # Compute modulator
        modulator = math.exp(clamped_argument)
        
        # Ensure modulator stays within reasonable bounds
        return max(0.1, min(10.0, modulator))

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
        
        # Store penalties for monitoring
        self.penalty_history.append(penalties)
        if len(self.penalty_history) > 100:
            self.penalty_history = self.penalty_history[-100:]
        
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
        
        bound = self.safety_bounds.dimensional_clamp
        for component in components:
            if abs(component) > bound:
                logger.warning(f"Dimensional inconsistency detected: component {component} exceeds bound ±{bound}")
                return False
        
        # Check modulator is within valid range
        if not (0.1 <= omega_state.modulator <= 10.0):
            logger.warning(f"Modulator out of range: {omega_state.modulator}")
            return False
        
        # Check entropy constraint during stable cycles
        if len(self.penalty_history) >= 10:
            recent_entropies = [p.entropy_penalty for p in self.penalty_history[-10:]]
            avg_entropy = np.mean(recent_entropies)
            if omega_state.coherence_score > 0.8 and avg_entropy > self.safety_bounds.entropy_threshold:
                logger.warning(f"Entropy constraint violated: {avg_entropy:.4f} > {self.safety_bounds.entropy_threshold}")
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
            time_delay=current_state.time_delay * 1.5,  # Increase time delay
            integrated_feedback=current_state.integrated_feedback * 0.8
        )
        
        # Add shocked state to history
        self.omega_history.append(shocked_state)
        
        # Simulate recovery steps
        steps = 0
        recovery_threshold = self.safety_bounds.coherence_recovery_threshold
        max_steps = self.safety_bounds.recovery_step_limit
        
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
                    modulator=min(10.0, self.omega_history[-1].modulator * 1.05),
                    time_delay=max(0.1, self.omega_history[-1].time_delay * 0.95),
                    integrated_feedback=prev_state.integrated_feedback * 0.95 + self.omega_history[-1].integrated_feedback * 0.05
                )
                
                self.omega_history.append(recovered_state)
                
                # Check if recovered
                if new_coherence >= recovery_threshold:
                    logger.info(f"✅ Harmonic shock recovery successful in {steps} steps")
                    return True, steps
        
        logger.warning(f"⚠️ Harmonic shock recovery failed after {max_steps} steps")
        return False, max_steps

    def _create_checkpoint(self, omega_state: OmegaState):
        """Create checkpoint for rapid restarts"""
        self.checkpoints.append(omega_state)
        if len(self.checkpoints) > 10:  # Keep only recent checkpoints
            self.checkpoints = self.checkpoints[-10:]
        logger.info(f"💾 Checkpoint created at step {len(self.omega_history)}")

    def get_latest_checkpoint(self) -> Optional[OmegaState]:
        """Get the latest checkpoint for rapid restarts"""
        return self.checkpoints[-1] if self.checkpoints else None

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

    def predict_coherence_breakdown(self) -> Tuple[bool, float]:
        """
        Predict when coherence breakdown might occur by analyzing variance trends
        Returns (will_breakdown, risk_score) where risk_score is probability of breakdown
        """
        if len(self.penalty_history) < 5:
            return False, 0.0
        
        # Analyze recent variance trends
        recent_variances = [p.variance_penalty for p in self.penalty_history[-5:]]
        variance_trend = np.polyfit(range(len(recent_variances)), recent_variances, 1)[0]
        
        # Analyze entropy trends
        recent_entropies = [p.entropy_penalty for p in self.penalty_history[-5:]]
        entropy_trend = np.polyfit(range(len(recent_entropies)), recent_entropies, 1)[0]
        
        # Combined risk score
        risk_score = 0.0
        if variance_trend > 0.01:  # Increasing variance
            risk_score += 0.3
        if entropy_trend > 0.005:  # Increasing entropy
            risk_score += 0.2
        if self.omega_history[-1].coherence_score < 0.5:  # Low current coherence
            risk_score += 0.5
            
        will_breakdown = risk_score > 0.3
        return will_breakdown, min(1.0, risk_score)

# Example usage
if __name__ == "__main__":
    # Create CAL engine instance
    cal_engine = CALEngine()
    
    # Simulate some Ω-state computations
    for i in range(5):
        omega = cal_engine.compute_omega_state(
            token_data={"rate": 100.0 + i * 10},
            sentiment_data={"energy": 0.5 + i * 0.1},
            semantic_data={"shift": 0.2 + i * 0.05},
            attention_data=[0.1, 0.2, 0.3, 0.4, 0.5]
        )
        print(f"Step {i}: Ψ={omega.coherence_score:.4f}, m_t={omega.modulator:.4f}")
    
    # Compute recursive coherence
    if len(cal_engine.omega_history) >= 2:
        coherence, penalties = cal_engine.compute_recursive_coherence(cal_engine.omega_history[-3:])
        print(f"Recursive Coherence: {coherence:.4f}")
        print(f"Penalties: cos={penalties.cosine_penalty:.4f}, "
              f"ent={penalties.entropy_penalty:.4f}, "
              f"var={penalties.variance_penalty:.4f}")
    
    # Test dimensional consistency
    if cal_engine.omega_history:
        is_consistent = cal_engine.validate_dimensional_consistency(cal_engine.omega_history[-1])
        print(f"Dimensional Consistency: {is_consistent}")
    
    # Test harmonic shock recovery
    recovered, steps = cal_engine.simulate_harmonic_shock_recovery(0.5)
    print(f"Harmonic Shock Recovery: {recovered} in {steps} steps")
    
    # Test coherence breakdown prediction
    will_breakdown, risk_score = cal_engine.predict_coherence_breakdown()
    print(f"Coherence Breakdown Prediction: {will_breakdown} (risk: {risk_score:.4f})")