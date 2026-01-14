#!/usr/bin/env python3
"""
🎛️ The Meta-Regulator - Autonomous Systemic Tuning
Formalizes the AI as an autonomous system tuner with Reinforcement Learning capabilities.

This module implements:
1. Reinforcement Learning Meta-Regulator for fine-tuning system physics
2. Action space for adjusting core mathematical parameters
3. Reward function for maximizing stable throughput
"""

import numpy as np
import time
import logging
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import sys
import os

# Add the parent directory to the path to import reinforcement_learning_integration
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# Import required components
from ..core.cal_engine import CALEngine
from ..models.coherence_attunement_layer import CoherenceAttunementLayer
from ..models.quantum_memory import UnifiedFieldMemory

# Try to import RL components
try:
    # Try absolute import
    from reinforcement_learning_integration import AEGISReinforcementLearning, RLConfig, RLAlgorithm
    RL_AVAILABLE = True
except ImportError:
    # Skip import - use mock
    AEGISReinforcementLearning = None
    RLConfig = None
    RLAlgorithm = None
    RL_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TuningParameter(Enum):
    """Parameters that can be tuned by the Meta-Regulator"""
    PSI_WEIGHT_ALIGNMENT = "psi_weight_alignment"      # a weight in Ψ equation
    PSI_WEIGHT_ENTROPY = "psi_weight_entropy"          # b weight in Ψ equation
    PSI_WEIGHT_VARIANCE = "psi_weight_variance"        # c weight in Ψ equation
    DIMENSIONAL_CLAMP = "dimensional_clamp"            # K clamping constant
    TEMPORAL_DELAY_MICRO = "temporal_delay_micro"      # τ(L_μ)
    TEMPORAL_DELAY_PHASE = "temporal_delay_phase"      # τ(L_ϕ)
    TEMPORAL_DELAY_MACRO = "temporal_delay_macro"      # τ(L_Φ)

@dataclass
class TuningAction:
    """Represents an action taken by the Meta-Regulator"""
    parameter: TuningParameter
    delta: float  # Change amount
    timestamp: float
    confidence: float  # Confidence in the action

@dataclass
class SystemState:
    """Represents the current system state for the Meta-Regulator"""
    internal_coherence: float           # H_internal
    psi_variance: float                 # Variance(Ψ)
    resource_cost: float                # Resource usage per transaction
    a_weight: float                     # Current a weight
    b_weight: float                     # Current b weight
    c_weight: float                     # Current c weight
    dimensional_clamp: float            # Current K value
    temporal_delays: Dict[str, float]   # Current τ(L) values

@dataclass
class MetaRegulatorConfig:
    """Configuration for the Meta-Regulator"""
    learning_rate: float = 0.01
    exploration_rate: float = 0.1
    discount_factor: float = 0.95
    update_interval: int = 300  # seconds
    simulation_enabled: bool = True
    safety_bounds: Optional[Dict[str, Tuple[float, float]]] = None
    
    def __post_init__(self):
        if self.safety_bounds is None:
            self.safety_bounds = {}

class MetaRegulator:
    """
    The Meta-Regulator - Autonomous Systemic Tuning
    Formalizes the AI as an autonomous system tuner with Reinforcement Learning capabilities
    """
    
    def __init__(self, network_id: str = "quantum-currency-meta-001"):
        self.network_id = network_id
        
        # Core components
        self.cal_engine = CALEngine(network_id=f"{network_id}-cal")
        self.coherence_layer = CoherenceAttunementLayer(network_id=f"{network_id}-coherence")
        self.ufm = UnifiedFieldMemory(network_id=f"{network_id}-ufm")
        
        # RL system (if available)
        self.rl_system = None
        if RL_AVAILABLE:
            try:
                self.rl_system = AEGISReinforcementLearning() if AEGISReinforcementLearning else None
            except Exception as e:
                logger.warning(f"Failed to initialize RL system: {e}")
        
        # Tuning history
        self.tuning_history: List[Tuple[SystemState, TuningAction, float]] = []
        self.current_params = self._get_current_parameters()
        self.last_update_time = time.time()
        
        # Configuration
        self.config = MetaRegulatorConfig(
            learning_rate=0.01,
            exploration_rate=0.1,
            discount_factor=0.95,
            update_interval=300,  # 5 minutes
            simulation_enabled=True,
            safety_bounds={
                "psi_weights": (0.0, 1.0),
                "dimensional_clamp": (8.0, 12.0),
                "temporal_delays": (0.1, 10.0)
            }
        )
        
        logger.info(f"🎛️ Meta-Regulator initialized for network: {network_id}")
    
    def _get_current_parameters(self) -> Dict[str, float]:
        """Get current system parameters"""
        # Get parameters from CAL engine
        cal_config = self.cal_engine.config
        
        # Get parameters from coherence layer
        coherence_config = self.coherence_layer.config
        
        return {
            "a_weight": coherence_config["penalty_weights"]["cosine"],
            "b_weight": coherence_config["penalty_weights"]["entropy"],
            "c_weight": coherence_config["penalty_weights"]["variance"],
            "dimensional_clamp": self.coherence_layer.safety_bounds["dimensional_clamp"],
            "temporal_delay_micro": 1.0,  # Placeholder
            "temporal_delay_phase": 2.0,  # Placeholder
            "temporal_delay_macro": 5.0   # Placeholder
        }
    
    def _get_system_state(self) -> SystemState:
        """
        Get current system state for the Meta-Regulator
        
        Returns:
            SystemState object with current metrics
        """
        # In a real implementation, these would be fetched from actual system metrics
        # For now, we'll use placeholder values with some variation
        
        # Simulate system metrics
        internal_coherence = 0.85 + np.random.normal(0, 0.05)  # 0.80-0.90
        psi_variance = 0.02 + abs(np.random.normal(0, 0.01))   # 0.01-0.05
        resource_cost = 0.65 + np.random.normal(0, 0.1)        # 0.40-0.90
        
        # Clamp values to reasonable ranges
        internal_coherence = max(0.0, min(1.0, internal_coherence))
        psi_variance = max(0.0, psi_variance)
        resource_cost = max(0.0, min(1.0, resource_cost))
        
        return SystemState(
            internal_coherence=internal_coherence,
            psi_variance=psi_variance,
            resource_cost=resource_cost,
            a_weight=self.current_params["a_weight"],
            b_weight=self.current_params["b_weight"],
            c_weight=self.current_params["c_weight"],
            dimensional_clamp=self.current_params["dimensional_clamp"],
            temporal_delays={
                "micro": self.current_params["temporal_delay_micro"],
                "phase": self.current_params["temporal_delay_phase"],
                "macro": self.current_params["temporal_delay_macro"]
            }
        )
    
    def _calculate_reward(self, state: SystemState, next_state: SystemState) -> float:
        """
        Calculate reward for the Meta-Regulator
        
        Reward_Meta = α·H_internal - β·Variance(Ψ) - γ·ResourceCost
        
        Args:
            state: Previous system state
            next_state: Current system state
            
        Returns:
            float: Calculated reward
        """
        # Reward weights
        alpha = 0.5  # Weight for internal coherence
        beta = 0.3   # Weight for psi variance penalty
        gamma = 0.2  # Weight for resource cost penalty
        
        # Calculate reward components
        coherence_reward = next_state.internal_coherence - state.internal_coherence
        variance_penalty = next_state.psi_variance
        resource_penalty = next_state.resource_cost
        
        # Calculate total reward
        reward = alpha * coherence_reward - beta * variance_penalty - gamma * resource_penalty
        
        logger.debug(f"Reward calculated: {reward:.4f} "
                    f"(coherence: {coherence_reward:.4f}, "
                    f"variance: {-beta * variance_penalty:.4f}, "
                    f"resource: {-gamma * resource_penalty:.4f})")
        
        return reward
    
    def _select_action(self, state: SystemState) -> TuningAction:
        """
        Select action using RL policy or simple heuristic
        
        Action space:
        - Δa, Δb, Δc (Ψ-Weight Shift)
        - ΔK (Clamping Constant)
        - Δτ(L_μ), Δτ(L_ϕ), Δτ(L_Φ) (Temporal Delay)
        
        Args:
            state: Current system state
            
        Returns:
            TuningAction to take
        """
        # If RL system is available, use it
        if self.rl_system and hasattr(self.rl_system, 'select_action'):
            try:
                # In a full implementation, this would use the trained RL model
                # For now, we'll use a simplified approach
                pass
            except Exception as e:
                logger.warning(f"RL action selection failed: {e}")
        
        # Simple heuristic-based action selection
        # This is the fallback when RL is not available or fails
        
        # Determine which parameter to adjust based on system state
        parameter = None
        delta = 0.0
        confidence = 0.8  # Base confidence
        
        if state.psi_variance > 0.03:
            # High variance - adjust entropy weight
            parameter = TuningParameter.PSI_WEIGHT_ENTROPY
            delta = -0.01 * (state.psi_variance - 0.03)  # Reduce entropy weight
        elif state.internal_coherence < 0.8:
            # Low coherence - adjust alignment weight
            parameter = TuningParameter.PSI_WEIGHT_ALIGNMENT
            delta = 0.01 * (0.8 - state.internal_coherence)  # Increase alignment weight
        elif state.resource_cost > 0.7:
            # High resource cost - adjust dimensional clamp
            parameter = TuningParameter.DIMENSIONAL_CLAMP
            delta = -0.1 * (state.resource_cost - 0.7)  # Reduce clamp to save resources
        else:
            # Random exploration
            if np.random.random() < self.config.exploration_rate:
                parameter_list = [p for p in TuningParameter]
                parameter = parameter_list[np.random.randint(0, len(parameter_list))]
                delta = np.random.normal(0, 0.01)
                confidence = 0.5
            else:
                # Small maintenance adjustment
                parameter = TuningParameter.PSI_WEIGHT_VARIANCE
                delta = np.random.normal(0, 0.001)
                confidence = 0.9
        
        # Clamp delta to reasonable range
        delta = max(-0.05, min(0.05, delta))
        
        action = TuningAction(
            parameter=parameter,
            delta=delta,
            timestamp=time.time(),
            confidence=confidence
        )
        
        logger.debug(f"Action selected: {parameter.value} += {delta:.4f} (confidence: {confidence:.2f})")
        return action
    
    def _apply_action(self, action: TuningAction, state: SystemState) -> bool:
        """
        Apply tuning action to the system
        
        Args:
            action: TuningAction to apply
            state: Current system state (for safety checks)
            
        Returns:
            bool: True if action applied successfully
        """
        # Safety check with micro-simulation
        if self.config.simulation_enabled:
            if not self._micro_simulate_action(action, state):
                logger.warning(f"Action rejected by micro-simulation: {action.parameter.value}")
                return False
        
        # Apply the action
        try:
            if action.parameter == TuningParameter.PSI_WEIGHT_ALIGNMENT:
                new_value = max(0.0, min(1.0, self.current_params["a_weight"] + action.delta))
                self.coherence_layer.config["penalty_weights"]["cosine"] = new_value
                self.current_params["a_weight"] = new_value
                
            elif action.parameter == TuningParameter.PSI_WEIGHT_ENTROPY:
                new_value = max(0.0, min(1.0, self.current_params["b_weight"] + action.delta))
                self.coherence_layer.config["penalty_weights"]["entropy"] = new_value
                self.current_params["b_weight"] = new_value
                
            elif action.parameter == TuningParameter.PSI_WEIGHT_VARIANCE:
                new_value = max(0.0, min(1.0, self.current_params["c_weight"] + action.delta))
                self.coherence_layer.config["penalty_weights"]["variance"] = new_value
                self.current_params["c_weight"] = new_value
                
            elif action.parameter == TuningParameter.DIMENSIONAL_CLAMP:
                new_value = max(8.0, min(12.0, self.current_params["dimensional_clamp"] + action.delta))
                self.coherence_layer.safety_bounds["dimensional_clamp"] = new_value
                self.current_params["dimensional_clamp"] = new_value
                
            elif action.parameter == TuningParameter.TEMPORAL_DELAY_MICRO:
                new_value = max(0.1, min(10.0, self.current_params["temporal_delay_micro"] + action.delta))
                self.current_params["temporal_delay_micro"] = new_value
                
            elif action.parameter == TuningParameter.TEMPORAL_DELAY_PHASE:
                new_value = max(0.1, min(10.0, self.current_params["temporal_delay_phase"] + action.delta))
                self.current_params["temporal_delay_phase"] = new_value
                
            elif action.parameter == TuningParameter.TEMPORAL_DELAY_MACRO:
                new_value = max(0.1, min(10.0, self.current_params["temporal_delay_macro"] + action.delta))
                self.current_params["temporal_delay_macro"] = new_value
            
            logger.info(f"✅ Action applied: {action.parameter.value} += {action.delta:.4f}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to apply action {action.parameter.value}: {e}")
            return False
    
    def _micro_simulate_action(self, action: TuningAction, state: SystemState) -> bool:
        """
        Micro-simulate action to predict Ψ score before applying to staging
        
        Args:
            action: TuningAction to simulate
            state: Current system state
            
        Returns:
            bool: True if simulation predicts safe outcome
        """
        # In a full implementation, this would run a micro-simulation
        # For now, we'll use a simple heuristic
        
        # Predict new Ψ score based on action
        predicted_psi = state.internal_coherence
        
        # Adjust prediction based on action type
        if action.parameter in [TuningParameter.PSI_WEIGHT_ALIGNMENT, 
                               TuningParameter.PSI_WEIGHT_ENTROPY, 
                               TuningParameter.PSI_WEIGHT_VARIANCE]:
            # Weight adjustments affect coherence directly
            predicted_psi += action.delta * 0.1
            
        elif action.parameter == TuningParameter.DIMENSIONAL_CLAMP:
            # Clamp adjustments affect stability
            if abs(action.delta) > 1.0:
                # Large changes might be unsafe
                return predicted_psi > 0.96  # Only allow if already high coherence
        
        # Safety check
        return predicted_psi >= 0.75  # Minimum safe coherence threshold
    
    def run_meta_regulator_cycle(self) -> Dict[str, Any]:
        """
        Run one cycle of the Meta-Regulator RL Loop
        
        Returns:
            Dict with cycle results
        """
        logger.info("🔄 Running Meta-Regulator cycle")
        
        # 1. Observe State (S_t)
        state_t = self._get_system_state()
        
        # 2. Agent Decides Action (A_t)
        action_t = self._select_action(state_t)
        
        # 3. Apply Action to System
        action_success = self._apply_action(action_t, state_t)
        
        # 4. Observe Next State (S_t+1) & Calculate Reward (R_t)
        state_t_plus_1 = self._get_system_state()
        reward_t = self._calculate_reward(state_t, state_t_plus_1)
        
        # 5. Store experience
        self.tuning_history.append((state_t, action_t, reward_t))
        
        # 6. Train Model (if RL system available)
        if self.rl_system and hasattr(self.rl_system, 'update_policy'):
            try:
                # In a full implementation, this would train the RL model
                # For now, we'll skip this step
                pass
            except Exception as e:
                logger.warning(f"RL training failed: {e}")
        
        # Update last update time
        self.last_update_time = time.time()
        
        result = {
            "status": "success" if action_success else "failed",
            "state_t": asdict(state_t),
            "action_t": {
                "parameter": action_t.parameter.value,
                "delta": action_t.delta,
                "confidence": action_t.confidence
            },
            "state_t_plus_1": asdict(state_t_plus_1),
            "reward_t": reward_t,
            "timestamp": time.time()
        }
        
        logger.info(f"Meta-Regulator cycle completed: reward={reward_t:.4f}")
        return result
    
    def get_tuning_report(self) -> Dict[str, Any]:
        """
        Get tuning report for the Meta-Regulator
        
        Returns:
            Dict with tuning information
        """
        if not self.tuning_history:
            recent_actions = []
        else:
            # Get last 10 actions
            recent_actions = []
            for state, action, reward in self.tuning_history[-10:]:
                recent_actions.append({
                    "parameter": action.parameter.value,
                    "delta": action.delta,
                    "reward": reward,
                    "timestamp": action.timestamp
                })
        
        return {
            "network_id": self.network_id,
            "cycles_completed": len(self.tuning_history),
            "current_parameters": self.current_params,
            "recent_actions": recent_actions,
            "config": asdict(self.config),
            "rl_available": RL_AVAILABLE,
            "last_update": self.last_update_time,
            "time_since_last_update": time.time() - self.last_update_time
        }

# Example usage and testing
if __name__ == "__main__":
    # Create Meta-Regulator instance
    meta_regulator = MetaRegulator()
    
    # Run a few cycles
    print("🎛️ Running Meta-Regulator cycles...")
    for i in range(3):
        result = meta_regulator.run_meta_regulator_cycle()
        print(f"Cycle {i+1}: Reward = {result['reward_t']:.4f}")
    
    # Get tuning report
    report = meta_regulator.get_tuning_report()
    print(f"Tuning report: {report['cycles_completed']} cycles completed")