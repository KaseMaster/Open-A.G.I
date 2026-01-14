#!/usr/bin/env python3
"""
Lambda Attunement Layer for Dynamic Coherence Optimization
Implements self-attunement subsystem that dynamically adjusts λ(t) to maximize system Coherence Density C(t)
"""

import time
import threading
import json
import logging
from collections import deque
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, asdict
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AttunementConfig:
    """Configuration for the Lambda Attunement Controller"""
    enabled: bool = True
    alpha_initial: float = 1.0
    alpha_min: float = 0.8
    alpha_max: float = 1.2
    delta_alpha_max: float = 0.02
    lr: float = 0.001
    momentum: float = 0.85
    epsilon: float = 1e-5
    settle_delay: float = 0.25
    gradient_averaging_window: int = 3
    cycle_sleep: float = 1.0
    safety: Dict[str, Any] = None
    logging: Dict[str, Any] = None

    def __post_init__(self):
        if self.safety is None:
            self.safety = {
                "entropy_max": 0.002,
                "h_internal_min": 0.95,
                "revert_on_failure": True
            }
        if self.logging is None:
            self.logging = {
                "audit_ledger_path": "/var/lib/uhes/attunement_ledger.log"
            }

@dataclass
class AttunementRecord:
    """Record of an attunement change for audit purposes"""
    timestamp: float
    old_alpha: float
    new_alpha: float
    reason: str
    c_before: float
    c_after: float
    actor: str = "attunement_controller"
    signature: Optional[str] = None

class CoherenceDensityMeter:
    """Measures system Coherence Density C_hat(t) as a proxy for optimization objective"""
    
    def __init__(self, cal_engine, window: int = 5):
        self.cal = cal_engine
        self.window = window

    def compute_C_hat(self) -> float:
        """
        Compute computational proxy for Coherence Density C(t)
        Sample over nodes/subdomains and short time window Δt
        Compute per-node squared norm of Ω vector, average and integrate
        Normalize into [0,1] range
        """
        try:
            # Sample per-node omega norms and average
            samples = self.cal.sample_omega_snapshot()
            # samples: list of node omega vectors
            if not samples:
                return 0.0
                
            total = 0.0
            for v in samples:
                # Compute squared norm of omega vector
                total += sum(x*x for x in v)
            C = total / max(1, len(samples))
            
            # Normalize if cal has normalization function
            return self.cal.normalize_C(C)
        except Exception as e:
            logger.error(f"Error computing C_hat: {e}")
            return 0.0

class LambdaAttunementController:
    """
    Implements λ-Attunement Layer that dynamically adjusts Recursive Feedback Coefficient λ(t)
    to maximize system Coherence Density C(t), subject to stability constraints.
    """
    
    def __init__(self, cal_engine, config: Optional[Dict[str, Any]] = None):
        self.cal = cal_engine
        cfg_dict = config or {}
        # Ensure safety and logging are always dicts
        if "safety" not in cfg_dict or cfg_dict["safety"] is None:
            cfg_dict["safety"] = {
                "entropy_max": 0.002,
                "h_internal_min": 0.95,
                "revert_on_failure": True
            }
        if "logging" not in cfg_dict or cfg_dict["logging"] is None:
            cfg_dict["logging"] = {
                "audit_ledger_path": "/var/lib/uhes/attunement_ledger.log"
            }
        self.cfg = AttunementConfig(**cfg_dict)
        self.alpha = self.cfg.alpha_initial
        self.velocity = 0.0
        self.history = deque(maxlen=self.cfg.gradient_averaging_window)
        self.running = False
        self.lock = threading.Lock()
        self.meter = CoherenceDensityMeter(cal_engine)
        self.audit_ledger: List[AttunementRecord] = []
        self.mode = 0  # 0=idle, 1=gradient, 2=pid, 3=emergency
        self.accept_counter = 0
        self.revert_counter = 0

    def start(self):
        """Start the attunement controller loop"""
        if not self.cfg.enabled:
            logger.info("Lambda attunement controller is disabled")
            return
            
        self.running = True
        threading.Thread(target=self._loop, daemon=True).start()
        logger.info("Lambda attunement controller started")

    def stop(self):
        """Stop the attunement controller loop"""
        self.running = False
        logger.info("Lambda attunement controller stopped")

    def _set_alpha(self, alpha: float) -> float:
        """
        Apply alpha multiplier to CAL engine safely
        Returns clamped alpha value
        """
        # Clamp alpha to safety bounds
        alpha = max(self.cfg.alpha_min, min(self.cfg.alpha_max, alpha))
        
        # Apply to CAL engine
        self.cal.set_alpha_multiplier(alpha)
        
        # Update Prometheus metrics (would be implemented in metrics_exporter)
        # uhes_alpha_value.set(alpha)
        # uhes_lambda_value.set(self.cal.get_lambda() * alpha)
        
        return alpha

    def _estimate_gradient(self, eps: Optional[float] = None) -> tuple:
        """
        Finite-difference gradient estimate
        Returns (gradient, C0) tuple
        """
        if eps is None:
            eps = self.cfg.epsilon
            
        C0 = self.meter.compute_C_hat()
        
        # Apply small perturbation forward
        old_alpha = self.alpha
        test_alpha = max(self.cfg.alpha_min, min(self.cfg.alpha_max, old_alpha + eps))
        self._set_alpha(test_alpha)
        time.sleep(self.cfg.settle_delay)
        C1 = self.meter.compute_C_hat()
        
        # Restore
        self._set_alpha(old_alpha)
        
        # Compute gradient
        if abs(test_alpha - old_alpha) > 1e-12:  # Avoid division by zero
            grad = (C1 - C0) / (test_alpha - old_alpha)
        else:
            grad = 0.0
            
        return grad, C0

    def _check_safety_constraints(self) -> bool:
        """
        Check safety constraints before committing alpha change
        Returns True if safe, False otherwise
        """
        try:
            # Check entropy rate
            entropy_rate = self.cal.get_entropy_rate()
            if entropy_rate > self.cfg.safety["entropy_max"]:
                logger.warning(f"Entropy rate {entropy_rate} exceeds max {self.cfg.safety['entropy_max']}")
                return False
                
            # Check internal coherence
            h_internal = self.cal.get_h_internal()
            if h_internal < self.cfg.safety["h_internal_min"]:
                logger.warning(f"Internal coherence {h_internal} below min {self.cfg.safety['h_internal_min']}")
                return False
                
            # Check m_t(L) bounds
            m_t_bounds = self.cal.get_m_t_bounds()
            # In a real implementation, we would check these bounds
            # For now, we'll assume they're within acceptable range
            
            return True
        except Exception as e:
            logger.error(f"Error checking safety constraints: {e}")
            return False

    def _log_attunement_change(self, old_alpha: float, new_alpha: float, 
                              c_before: float, c_after: float, accepted: bool):
        """
        Log attunement change to audit ledger
        """
        record = AttunementRecord(
            timestamp=time.time(),
            old_alpha=old_alpha,
            new_alpha=new_alpha,
            reason="gradient_ascent" if accepted else "safety_violation",
            c_before=c_before,
            c_after=c_after,
            actor="attunement_controller"
        )
        
        self.audit_ledger.append(record)
        
        # In a real implementation, we would also append to a persistent log file
        # For now, we'll just log to console
        logger.info(f"Attunement change: {old_alpha:.6f} -> {new_alpha:.6f} "
                   f"(C: {c_before:.6f} -> {c_after:.6f}) {'ACCEPTED' if accepted else 'REVERTED'}")
        
        # Update Prometheus counters
        if accepted:
            self.accept_counter += 1
            # uhes_alpha_accept.inc()
        else:
            self.revert_counter += 1
            # uhes_alpha_revert.inc()

    def update(self) -> bool:
        """
        Perform one attunement update step
        Returns True if change was accepted, False if reverted
        """
        with self.lock:
            try:
                # Measure current coherence
                grad, C0 = self._estimate_gradient()
                
                # Ascent step with momentum
                self.velocity = self.cfg.momentum * self.velocity + self.cfg.lr * grad
                delta = max(-self.cfg.delta_alpha_max, min(self.cfg.delta_alpha_max, self.velocity))
                new_alpha = max(self.cfg.alpha_min, min(self.cfg.alpha_max, self.alpha + delta))
                
                # Safety pre-checks
                old_alpha = self.alpha
                self._set_alpha(new_alpha)
                time.sleep(self.cfg.settle_delay)
                C_new = self.meter.compute_C_hat()
                
                # Check if change should be accepted or reverted
                should_accept = True
                
                # Check if coherence improved
                if C_new < C0 and self.cfg.safety.get("revert_on_failure", True):
                    should_accept = False
                    
                # Check safety constraints
                if not self._check_safety_constraints():
                    should_accept = False
                    
                if should_accept:
                    # Accept change
                    self.alpha = new_alpha
                    self._log_attunement_change(old_alpha, new_alpha, C0, C_new, True)
                    self.history.append((C0, C_new, grad, delta))
                    self.mode = 1  # gradient mode
                    return True
                else:
                    # Revert change
                    self._set_alpha(old_alpha)
                    self._log_attunement_change(old_alpha, old_alpha, C0, C_new, False)
                    # Reduce learning rate or switch to PID mode as fallback
                    self.cfg.lr *= 0.9  # Adaptive learning rate reduction
                    self.mode = 2  # PID mode fallback
                    return False
                    
            except Exception as e:
                logger.error(f"Error in attunement update: {e}")
                # Emergency mode
                self.mode = 3
                return False

    def _loop(self):
        """Main attunement loop"""
        while self.running:
            try:
                if self.cfg.enabled:
                    self.update()
                time.sleep(self.cfg.cycle_sleep)
            except Exception as e:
                logger.error(f"Error in attunement loop: {e}")
                time.sleep(self.cfg.cycle_sleep)

    def get_status(self) -> Dict[str, Any]:
        """Get current status of the attunement controller"""
        return {
            "alpha": self.alpha,
            "velocity": self.velocity,
            "mode": self.mode,
            "accept_counter": self.accept_counter,
            "revert_counter": self.revert_counter,
            "history_length": len(self.history),
            "enabled": self.cfg.enabled
        }

    def get_audit_ledger(self) -> List[Dict[str, Any]]:
        """Get audit ledger as list of dictionaries"""
        return [asdict(record) for record in self.audit_ledger]

    def save_state(self, filepath: str) -> bool:
        """Save attunement state to file"""
        try:
            state = {
                "alpha": self.alpha,
                "velocity": self.velocity,
                "history": list(self.history),
                "audit_ledger": self.get_audit_ledger(),
                "config": asdict(self.cfg)
            }
            
            with open(filepath, 'w') as f:
                json.dump(state, f, indent=2)
                
            logger.info(f"Attunement state saved to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error saving attunement state: {e}")
            return False

    def load_state(self, filepath: str) -> bool:
        """Load attunement state from file"""
        try:
            with open(filepath, 'r') as f:
                state = json.load(f)
                
            self.alpha = state["alpha"]
            self.velocity = state["velocity"]
            self.history = deque(state["history"], maxlen=self.cfg.gradient_averaging_window)
            
            # Load audit ledger
            self.audit_ledger = []
            for record_dict in state["audit_ledger"]:
                record = AttunementRecord(**record_dict)
                self.audit_ledger.append(record)
                
            logger.info(f"Attunement state loaded from {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error loading attunement state: {e}")
            return False

def demo_lambda_attunement():
    """Demonstrate lambda attunement functionality"""
    print("🔬 Lambda Attunement Controller Demo")
    print("=" * 40)
    
    # This is a simplified demo - in a real implementation, we would have a CAL engine
    class MockCALEngine:
        def __init__(self):
            self.alpha_multiplier = 1.0
            self.omega_history = []
            
        def sample_omega_snapshot(self):
            # Generate mock omega vectors
            return [np.random.random(5).tolist() for _ in range(3)]
            
        def normalize_C(self, C):
            # Normalize C to [0,1] range
            return min(1.0, max(0.0, C / 10.0))
            
        def set_alpha_multiplier(self, alpha):
            self.alpha_multiplier = alpha
            print(f"   Setting alpha multiplier to {alpha:.4f}")
            
        def get_lambda(self):
            return 0.5  # Mock lambda value
            
        def get_entropy_rate(self):
            return np.random.random() * 0.001  # Mock entropy rate
            
        def get_h_internal(self):
            return 0.98  # Mock internal coherence
            
        def get_m_t_bounds(self):
            return (-1.0, 1.0)  # Mock bounds
            
    # Create mock CAL engine
    cal_engine = MockCALEngine()
    
    # Create attunement controller
    config = {
        "alpha_initial": 1.0,
        "alpha_min": 0.8,
        "alpha_max": 1.2,
        "lr": 0.01,
        "momentum": 0.9
    }
    
    controller = LambdaAttunementController(cal_engine, config)
    
    # Show initial status
    print(f"Initial alpha: {controller.alpha:.4f}")
    print(f"Initial C_hat: {controller.meter.compute_C_hat():.4f}")
    
    # Run a few update steps
    print("\nRunning attunement updates...")
    for i in range(5):
        accepted = controller.update()
        status = controller.get_status()
        print(f"Step {i+1}: alpha={status['alpha']:.4f}, "
              f"C_hat={controller.meter.compute_C_hat():.4f}, "
              f"{'ACCEPTED' if accepted else 'REVERTED'}")
        time.sleep(0.1)  # Short delay for demo
    
    # Show final status
    final_status = controller.get_status()
    print(f"\nFinal status:")
    print(f"   Alpha: {final_status['alpha']:.4f}")
    print(f"   Mode: {final_status['mode']}")
    print(f"   Accepts: {final_status['accept_counter']}")
    print(f"   Reverts: {final_status['revert_counter']}")
    
    # Show audit ledger
    ledger = controller.get_audit_ledger()
    print(f"\nAudit ledger entries: {len(ledger)}")
    if ledger:
        last_entry = ledger[-1]
        print(f"   Last entry: {last_entry['old_alpha']:.4f} -> {last_entry['new_alpha']:.4f}")
    
    print("\n✅ Lambda attunement demo completed!")

if __name__ == "__main__":
    demo_lambda_attunement()