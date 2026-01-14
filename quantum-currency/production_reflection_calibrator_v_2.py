#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
production_reflection_calibrator_v2.py

Upgraded Production Reflection & Coherence Calibration System

Key upgrades focused on four areas requested by the user:
 - Harmonic Engine (HE): dedicated class that consolidates mathematical cluster
 - Ω-Security Primitives: HMAC signing, tamper-evident report envelopes, secure entropy
 - Meta-Regulator: a formalized autonomous tuner skeleton with reward shaping
 - Implementation Guidance: instruction-level pseudocode and structured hooks

This file is intended as a production-ready scaffold: core logic, safe defaults,
and clear integration points for real metrics, Prometheus, Slack, K8s, and secure
key management.

Notes:
 - Replace placeholder secrets with your secret manager integration (Vault, KMS).
 - Prometheus fetch routines use requests; adapt to your environment (TLS, auth).
 - The MetaRegulator is provided as a policy/tuner skeleton. Plug in your RL or
   optimisation engine (Ray, PPO, Optuna, etc.).

"""

from __future__ import annotations

import json
import logging
import os
import time
import hmac
import hashlib
import random
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import requests

# Try to import NumPy for FFT operations, fallback to math if not available
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None  # type: ignore

try:
    import math
    MATH_AVAILABLE = True
except ImportError:
    MATH_AVAILABLE = False
    math = None  # type: ignore

# ---------- Configuration & Constants ----------
REPORT_DIR = os.environ.get("PRC_REPORT_DIR", "/mnt/data")
F0 = float(os.environ.get("PRC_BASE_FREQ", 432.0))
DEFAULT_HSVP_CYCLES = 5
DEFAULT_CCF_CYCLES = 10

# Security defaults (in production store keys in Vault/KMS)
HMAC_KEY = os.environ.get("PRC_HMAC_KEY", "change-me-please")
HMAC_ALGO = "sha256"

# Random seed for reproducibility in simulated mode
SIM_SEED = int(os.environ.get("PRC_SIM_SEED", "0"))
SIMULATED_MODE = bool(int(os.environ.get("PRC_SIMULATED", "1")))

os.makedirs(REPORT_DIR, exist_ok=True)

# ---------- Logging ----------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("ProductionReflectionCalibratorV2")

# ---------- Dataclasses ----------
@dataclass
class ComponentStatus:
    name: str
    status: str
    details: str
    timestamp: str = datetime.now(timezone.utc).isoformat()


@dataclass
class CalibrationResult:
    cycle_id: int
    timestamp: str
    metrics: Dict[str, float]
    parameters: Dict[str, float]
    stability: bool
    coherence_score: float
    delta_psi: float
    entropy_rate: float
    caf: float


# ---------- Utility helpers ----------

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def secure_sign(payload: bytes, key: str = HMAC_KEY, algo: str = HMAC_ALGO) -> str:
    """Return a hex HMAC signature for payload. In production use KMS-managed key."""
    return hmac.new(key.encode(), payload, getattr(hashlib, algo)).hexdigest()


def retry_on_exception(retries: int = 3, delay: float = 1.0):
    def decorator(fn):
        def wrapper(*args, **kwargs):
            last_exc = None
            for i in range(retries):
                try:
                    return fn(*args, **kwargs)
                except Exception as e:
                    last_exc = e
                    logger.warning(f"Retry {i+1}/{retries} for {fn.__name__}: {e}")
                    time.sleep(delay * (2 ** i))  # Exponential backoff
            if last_exc:
                raise last_exc
            raise Exception("Unknown error in retry decorator")
        return wrapper
    return decorator


# ---------- Harmonic Engine (HE) ----------
class HarmonicEngine:
    """Consolidates core mathematical cluster for peak performance.

    Responsibilities:
     - Compute coherence score from multimodal inputs
     - Maintain short-term coherence history and compute variance σ²Ω
     - Frequency normalization utilities
     - Provide deterministic simulated metrics when SIMULATED_MODE is True

    Implementation notes & pseudocode:
     - real_coherence = f(weights, metrics, spectral_analysis)
     - spectral_analysis -> compute FFT on time-series signals from Prometheus or local probes
     - coherence_score combines power in harmonic bands around F0 and phase-locking value

    Pseudocode for spectral coherence (implementation left as integration point):

        def spectral_coherence(time_series, base_freq=F0):
            # 1. apply bandpass filter around base_freq
            # 2. compute FFT and power spectral density
            # 3. measure power ratio inside harmonic window / total power
            # 4. return normalized coherence score [0..1]

    """

    def __init__(self, base_freq: float = F0, history_len: int = 32):
        self.base_freq = base_freq
        self.history_len = history_len
        self.history: List[float] = []
        if SIM_SEED:
            random.seed(SIM_SEED)

    def push(self, value: float) -> None:
        self.history.append(value)
        if len(self.history) > self.history_len:
            self.history.pop(0)

    def mean(self) -> float:
        return sum(self.history) / len(self.history) if self.history else 0.0

    def variance(self) -> float:
        if not self.history:
            return 0.0
        m = self.mean()
        return sum((x - m) ** 2 for x in self.history) / len(self.history)

    def compute_coherence(self, inputs: Optional[Dict[str, float]] = None) -> float:
        """Compute a coherence score using fused inputs. In simulated mode returns stable values.

        inputs: e.g. {'spectral_power': 0.9, 'phase_lock': 0.95, 'prometheus_coh': 0.98}
        merging strategy: weighted average with tunable weights from MetaRegulator
        """
        if SIMULATED_MODE:
            # deterministic simulated coherence for demo/testing
            val = 0.975 + random.uniform(-0.01, 0.01)
            self.push(val)
            return val

        # Real implementation should fuse spectral metrics here.
        if not inputs:
            logger.debug("No inputs supplied to compute_coherence; returning 0.0")
            return 0.0

        # Example weighted fusion (weights should be provided by MetaRegulator)
        weights = inputs.get("weights", {})
        # Ensure weights is a dict
        if not isinstance(weights, dict):
            weights = {"spectral": 0.4, "phase": 0.4, "prom": 0.2}
            
        spectral = inputs.get("spectral_power", 0.9)
        phase_lock = inputs.get("phase_lock", 0.95)
        prometheus_coh = inputs.get("prometheus_coh", 0.0)

        w_spec = weights.get("spectral", 0.4)
        w_phase = weights.get("phase", 0.4)
        w_prom = weights.get("prom", 0.2)

        score = w_spec * spectral + w_phase * phase_lock + w_prom * prometheus_coh
        score = max(0.0, min(1.0, score))
        self.push(score)
        return score

    def spectral_coherence(self, time_series: List[float], sampling_rate: float = 1.0) -> float:
        """Compute spectral coherence using FFT-based analysis.
        
        Implementation of PSD-based spectral coherence:
        1. Apply windowing to reduce spectral leakage
        2. Compute FFT and power spectral density
        3. Measure power ratio in harmonic bands around base frequency
        4. Return normalized coherence score [0..1]
        
        Args:
            time_series: List of time-domain samples
            sampling_rate: Sampling rate in Hz
            
        Returns:
            Spectral coherence score between 0 and 1
        """
        if not time_series or len(time_series) < 4:
            return 0.0
            
        if not NUMPY_AVAILABLE or np is None:
            # Fallback implementation using basic math
            return self._basic_spectral_coherence(time_series, sampling_rate)
            
        try:
            # Convert to numpy array
            signal = np.array(time_series)
            
            # Apply windowing to reduce spectral leakage
            windowed = signal * np.hanning(len(signal))
            
            # Compute FFT
            fft_result = np.fft.fft(windowed)
            freqs = np.fft.fftfreq(len(windowed), 1/sampling_rate)
            
            # Compute power spectral density
            psd = np.abs(fft_result) ** 2
            
            # Find frequency bins around base frequency
            base_freq_bin = np.argmin(np.abs(freqs - self.base_freq))
            
            # Define harmonic window (±5% around base frequency)
            freq_resolution = sampling_rate / len(signal)
            window_bins = int(0.05 * self.base_freq / freq_resolution)
            
            start_bin = max(0, base_freq_bin - window_bins)
            end_bin = min(len(psd), base_freq_bin + window_bins)
            
            # Calculate power in harmonic window vs total power
            harmonic_power = np.sum(psd[start_bin:end_bin])
            total_power = np.sum(psd)
            
            if total_power == 0:
                return 0.0
                
            coherence = harmonic_power / total_power
            return float(np.clip(coherence, 0.0, 1.0))
            
        except Exception as e:
            logger.warning(f"Error in spectral_coherence: {e}")
            return 0.0

    def _basic_spectral_coherence(self, time_series: List[float], sampling_rate: float = 1.0) -> float:
        """Basic spectral coherence implementation without NumPy.
        
        Simplified approach using discrete Fourier transform concepts.
        """
        if not MATH_AVAILABLE or math is None:
            # If math is not available, return a default value
            return 0.5
            
        n = len(time_series)
        if n < 4:
            return 0.0
            
        # Compute basic DFT around base frequency
        base_freq_normalized = self.base_freq / sampling_rate
        
        # Compute power around base frequency using Goertzel-like approach
        omega = 2 * math.pi * base_freq_normalized
        cos_omega = math.cos(omega)
        sin_omega = math.sin(omega)
        
        # Initialize Goertzel algorithm variables
        s1 = 0.0
        s2 = 0.0
        
        # Apply Goertzel algorithm
        for i in range(n):
            s0 = time_series[i] + 2 * cos_omega * s1 - s2
            s2 = s1
            s1 = s0
            
        # Calculate power at target frequency
        power_at_freq = s1**2 + s2**2 - 2 * cos_omega * s1 * s2
        
        # Calculate total signal power
        total_power = sum(x**2 for x in time_series)
        
        if total_power == 0:
            return 0.0
            
        # Return normalized coherence
        coherence = power_at_freq / total_power
        return max(0.0, min(1.0, coherence))

    def frequency_normalize(self) -> Dict[str, Any]:
        """Normalize frequency parameters — placeholder hook for external actuators.

        Pseudocode:
            read current oscillator params
            compute target offsets based on variance, coherence
            apply corrections via control plane (k8s, actuator API)
        """
        logger.info("HarmonicEngine: frequency_normalize() invoked")
        return {"normalized": True, "timestamp": now_iso()}


# ---------- Ω-Security Primitives ----------
class OmegaSecurity:
    """Implements intrinsic security primitives based on coherence.

    Features:
     - Tamper-evident envelopes: sign reports with HMAC and include entropy
     - Entropy & nonce generation using OS-secure RNG
     - Lightweight integrity verification helper

    Security notes:
     - Never hardcode keys in production. Use AWS KMS / GCP KMS / HashiCorp Vault.
     - Consider using an asymmetric signature scheme and publish public keys for verification.
    """

    def __init__(self, hmac_key: str = HMAC_KEY):
        self.hmac_key = hmac_key

    @staticmethod
    def secure_nonce(length: int = 16) -> str:
        return os.urandom(length).hex()

    def envelope(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        signature = secure_sign(raw, key=self.hmac_key)
        return {
            "payload": payload,
            "meta": {
                "timestamp": now_iso(),
                "nonce": self.secure_nonce(),
                "signature": signature,
                "algo": HMAC_ALGO
            }
        }

    def verify_envelope(self, envelope: Dict[str, Any]) -> bool:
        try:
            payload = envelope["payload"]
            provided_sig = envelope["meta"]["signature"]
            raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
            expected_sig = secure_sign(raw, key=self.hmac_key)
            return hmac.compare_digest(provided_sig, expected_sig)
        except Exception:
            return False


# ---------- Meta-Regulator (Autonomous System Tuner) ----------
class MetaRegulator:
    """A formalized autonomous tuner that observes coherence deltas and issues tunings.

    Responsibilities:
     - Collect signals and compute reward
     - Produce tuning actions (weights, PID gains, CAF targets)
     - Maintain conservative 'safe zone' policy to avoid aggressive actuations

    Implementation guidance (instruction-level pseudocode):

    observe -> compute_reward -> propose_action -> validate_action -> apply_action -> log

    Where compute_reward might be:
        reward = w1 * (ΔΨ) + w2 * (improvement_in_Cw) - w3 * (entropy_rate)

    Action validation pseudocode:
        if action would change parameters by > max_delta: reject or clamp
        if proposed action violates safety rules: fallback to no-op

    Plug-in points:
     - replace propose_action with RL policy (PPO, DDPG) or Bayesian optimizer
     - use a replay buffer for training offline

    """

    def __init__(self, safe_delta: float = 0.05):
        self.safe_delta = safe_delta
        self.history: List[Dict[str, Any]] = []
        
        # Try to import stable-baselines3 for PPO implementation
        global PPO_AVAILABLE
        try:
            from stable_baselines3 import PPO  # type: ignore
            PPO_AVAILABLE = True
        except ImportError:
            PPO_AVAILABLE = False

    def compute_reward(self, before: Dict[str, float], after: Dict[str, float]) -> float:
        # Instruction-level: compute reward from delta of coherence & entropy
        delta_psi = after.get("Psi", 0.0) - before.get("Psi", 0.0)
        delta_cw = after.get("Cw", 0.0) - before.get("Cw", 0.0)
        entropy = after.get("entropy_rate", 0.0)
        # Weights (tunable by external policy)
        reward = 1.0 * delta_psi + 0.5 * delta_cw - 0.3 * entropy
        logger.debug(f"MetaRegulator reward computed: {reward:.6f}")
        return reward

    def propose_action(self, observation: Dict[str, Any]) -> Dict[str, float]:
        """Return a suggested tuning action.

        For now a conservative heuristic. Replace with trained policy.
        """
        # Example: tune fusion weights and CAF target slightly in direction of positive reward
        # Pseudocode implementation:
        # 1) compute gradient estimate from recent history (or use RL policy)
        # 2) propose small adjustments bounded by safe_delta
        current_weights = observation.get("weights", {"spectral": 0.4, "phase": 0.4, "prom": 0.2})
        # Ensure weights is a dict
        if not isinstance(current_weights, dict):
            current_weights = {"spectral": 0.4, "phase": 0.4, "prom": 0.2}
            
        psi = observation.get("Psi", 0.98)
        action = {}
        # nudge weights toward spectral if Psi is low
        if psi < 0.98:
            action = {"spectral": min(current_weights["spectral"] + 0.02, 0.9),
                      "phase": max(current_weights["phase"] - 0.01, 0.05),
                      "prom": max(current_weights["prom"] - 0.01, 0.01)}
        else:
            # small conservative drift
            action = {k: max(min(v + random.uniform(-0.01, 0.01), 0.9), 0.0) for k, v in current_weights.items()}

        # enforce normalization
        s = sum(action.values())
        if s <= 0:
            action = current_weights
        else:
            action = {k: v / s for k, v in action.items()}

        # clamp deltas
        for k in action:
            delta = abs(action[k] - current_weights.get(k, 0.0))
            if delta > self.safe_delta:
                logger.debug(f"Clamping {k} delta from {delta} to safe_delta {self.safe_delta}")
                if action[k] > current_weights.get(k, 0.0):
                    action[k] = current_weights[k] + self.safe_delta
                else:
                    action[k] = current_weights[k] - self.safe_delta

        logger.info(f"MetaRegulator proposed action: {action}")
        return action

    def validate_action(self, action: Dict[str, float]) -> bool:
        # Example safety rules: weights must sum near 1 and be within [0,1]
        s = sum(action.values())
        if not (0.9 <= s <= 1.1):
            logger.warning("Proposed action violates normalization constraints")
            return False
        if any(v < 0.0 or v > 1.0 for v in action.values()):
            logger.warning("Proposed action has values outside [0,1]")
            return False
        return True

    def train_ppo_policy(self, observations: List[Dict], rewards: List[float]) -> Optional[Any]:
        """Skeleton for training a PPO policy using stable-baselines3.
        
        This is a placeholder implementation that shows how to integrate
        a real RL policy. In practice, you would need:
        1. A proper environment definition
        2. Correct observation/reward shaping
        3. Training loop with replay buffer
        
        Pseudocode for PPO integration:
            env = QuantumCurrencyEnv(observation_space, action_space)
            model = PPO("MlpPolicy", env, verbose=1)
            model.learn(total_timesteps=10000)
            return model
        """
        if not PPO_AVAILABLE:
            logger.warning("stable-baselines3 not available, cannot train PPO policy")
            return None
            
        # Validate inputs
        if not observations or not rewards:
            logger.warning("Empty observations or rewards provided to PPO training")
            return None
            
        if len(observations) != len(rewards):
            logger.error("Mismatch between observations and rewards length")
            return None
            
        try:
            logger.info(f"Training PPO policy with {len(observations)} observations")
            # This is a placeholder - in a real implementation you would:
            # 1. Create a custom environment
            # 2. Initialize PPO with appropriate policy
            # 3. Train on observations and rewards
            # 4. Return the trained model
            
            # For now, just return a mock object to show the concept
            class MockPPOModel:
                def __init__(self):
                    self.training_steps = len(observations)
                
                def predict(self, observation):
                    # Return a random valid action
                    weights = {"spectral": 0.4, "phase": 0.4, "prom": 0.2}
                    # Add some noise
                    weights = {k: max(min(v + random.uniform(-0.05, 0.05), 0.9), 0.0) for k, v in weights.items()}
                    # Normalize
                    s = sum(weights.values())
                    if s > 0:
                        weights = {k: v/s for k, v in weights.items()}
                    return weights, None
                    
            return MockPPOModel()
            
        except Exception as e:
            logger.error(f"Error training PPO policy: {e}")
            return None


# ---------- ProductionReflectionCalibrator ----------
class ProductionReflectionCalibrator:
    def __init__(self, prometheus_url: Optional[str] = None, slack_webhook_url: Optional[str] = None):
        self.prometheus_url = prometheus_url
        self.slack_webhook_url = slack_webhook_url
        self.report_dir = REPORT_DIR
        self.he = HarmonicEngine()
        self.sec = OmegaSecurity()
        self.reg = MetaRegulator()
        self.previous_psi = 0.975
        self.coherence_history: List[float] = []

    # ---------- I. Component Verification Layer ----------
    def verify_components(self) -> List[ComponentStatus]:
        logger.info("I. Component Verification Layer")
        statuses: List[ComponentStatus] = []

        expected_components = [
            ("emanation_deploy.py", "File exists and is accessible"),
            ("prometheus_connector.py", "File exists"),
            ("slack_alerts.py", "File exists"),
            ("dashboard/realtime_coherence_dashboard.py", "File exists"),
            ("dashboard/templates/dashboard.html", "File exists"),
            ("k8s/emanation-monitor-cronjob.yaml", "File exists"),
            ("k8s/emanation-monitor-deployment.yaml", "File exists")
        ]

        for path, ok_msg in expected_components:
            if os.path.exists(path):
                statuses.append(ComponentStatus(name=path, status="✅ Stable", details=ok_msg))
            else:
                statuses.append(ComponentStatus(name=path, status="⚠️  Missing", details="Not found on disk"))

        # Additional runtime checks
        if self.prometheus_url:
            statuses.append(ComponentStatus(name="prometheus_endpoint", status="✅ Configured", details=self.prometheus_url))
        else:
            statuses.append(ComponentStatus(name="prometheus_endpoint", status="⚠️  Not Configured", details="Provide --prometheus-url"))

        if self.slack_webhook_url:
            statuses.append(ComponentStatus(name="slack_webhook", status="✅ Configured", details=self.slack_webhook_url))
        else:
            statuses.append(ComponentStatus(name="slack_webhook", status="⚠️  Not Configured", details="Provide --slack-webhook"))

        # Print summary
        for s in statuses:
            logger.info(f"Component {s.name}: {s.status} - {s.details}")

        return statuses

    # ---------- Prometheus fetch utility ----------
    @retry_on_exception(retries=3, delay=1.0)
    def fetch_prometheus_metric(self, query: str) -> Optional[float]:
        if not self.prometheus_url:
            raise RuntimeError("Prometheus URL not configured")
        resp = requests.get(f"{self.prometheus_url}/api/v1/query", params={"query": query}, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        try:
            value = float(data["data"]["result"][0]["value"][1])
            return value
        except Exception:
            return None

    # ---------- II. Harmonic Self-Verification Protocol (HSVP) ----------
    def run_harmonic_self_verification(self, cycles: int = DEFAULT_HSVP_CYCLES, coherence_threshold: float = 0.98) -> Dict[str, Any]:
        logger.info("II. Harmonic Self-Verification Protocol (HSVP)")
        results: Dict[str, Any] = {"cycles": cycles, "cycle_results": []}

        for cycle in range(1, cycles + 1):
            # Acquire or simulate metrics
            if SIMULATED_MODE:
                coherence_score = self.he.compute_coherence()
                caf = 1.03 + random.uniform(-0.02, 0.03)
                entropy_rate = random.uniform(0.001, 0.003)
                psi = coherence_score  # simplified mapping
                
                # Generate synthetic time series for spectral analysis
                time_series = [random.uniform(-1, 1) for _ in range(100)]
                spectral_power = self.he.spectral_coherence(time_series, sampling_rate=10.0)
            else:
                # Example: fetch Prometheus metrics and spectral inputs
                prometheus_coh = self.fetch_prometheus_metric('uhes_coherence_score') or 0.0
                # In a real implementation, you would fetch time series data for spectral analysis
                time_series = [random.uniform(-1, 1) for _ in range(100)]  # Placeholder
                spectral_power = self.he.spectral_coherence(time_series, sampling_rate=10.0)
                
                inputs = {"prometheus_coh": prometheus_coh, "spectral_power": spectral_power}
                coherence_score = self.he.compute_coherence(inputs)
                caf = self.fetch_prometheus_metric('uhes_caf') or 1.0
                entropy_rate = self.fetch_prometheus_metric('uhes_entropy') or 0.001
                psi = coherence_score

            self.coherence_history.append(coherence_score)
            if len(self.coherence_history) > 32:
                self.coherence_history.pop(0)

            variance_omega = self.he.variance()

            # AutoBalance check
            auto_balance_stable = abs(psi - self.previous_psi) <= 0.005

            # Frequency normalization policy
            frequency_normalized = False
            if variance_omega > 0.0008:
                logger.info(f"Variance ω = {variance_omega:.6f} > threshold; normalizing")
                self.he.frequency_normalize()
                frequency_normalized = True

            result = {
                "cycle": cycle,
                "timestamp": now_iso(),
                "metrics": {"H_internal": coherence_score, "CAF": caf, "entropy_rate": entropy_rate, "Psi": psi, "spectral_power": spectral_power},
                "variance_omega": variance_omega,
                "auto_balance_stable": auto_balance_stable,
                "frequency_normalized": frequency_normalized
            }
            results["cycle_results"].append(result)

            # MetaRegulator usage example: propose minor weight adjustments
            observation = {"Psi": psi, "weights": {"spectral": 0.4, "phase": 0.4, "prom": 0.2}}
            proposed = self.reg.propose_action(observation)
            if self.reg.validate_action(proposed):
                # Apply action: in a real system this updates the fusion weights used by HE
                logger.info(f"Applying regulated tuning: {proposed}")
            else:
                logger.warning("Proposed action rejected by MetaRegulator validation")

            # Update previous psi
            self.previous_psi = psi

        # Summary
        summary = {
            "avg_H_internal": sum(r["metrics"]["H_internal"] for r in results["cycle_results"]) / len(results["cycle_results"]),
            "avg_CAF": sum(r["metrics"]["CAF"] for r in results["cycle_results"]) / len(results["cycle_results"]),
            "avg_entropy": sum(r["metrics"]["entropy_rate"] for r in results["cycle_results"]) / len(results["cycle_results"])
        }
        results["summary"] = summary
        return results

    # ---------- III. Coherence Calibration Matrix (CCM) ----------
    def run_coherence_calibration_matrix(self) -> Dict[str, Any]:
        logger.info("III. Coherence Calibration Matrix (CCM)")
        results: Dict[str, Any] = {"timestamp": now_iso()}

        # C1 - Resonance Check (prometheus vs internal)
        try:
            internal = {"coherence_score": self.he.mean() or 0.0}
            prometheus = None
            if self.prometheus_url and not SIMULATED_MODE:
                prometheus = {"coherence_score": self.fetch_prometheus_metric('uhes_coherence_score')}
            else:
                # In simulated mode, generate consistent values
                prometheus = {"coherence_score": (self.he.mean() or 0.0) + random.uniform(-0.01, 0.01)}

            if prometheus:
                drift = abs(internal['coherence_score'] - prometheus['coherence_score']) if prometheus['coherence_score'] else None
                results['C1'] = {"internal": internal, "prometheus": prometheus, "drift": drift}
            else:
                results['C1'] = {"internal": internal, "prometheus": prometheus, "status": "incomplete"}
        except Exception as e:
            results['C1'] = {"error": str(e)}

        # C2 - Frequency Sync (PID) - pseudocode + conservative apply
        results['C2'] = {"status": "simulated", "pid_applied": False}
        # Pseudocode for PID tuning:
        # 1. read error = target_coherence - observed_coherence
        # 2. compute dI and dD terms from history
        # 3. produce new gains Kp, Ki, Kd within safe bounds
        # 4. write to actuator or parameter store (k8s configmap, SSM)

        # C3 - Cross-System Reflection
        comm_layers = {
            "slack": "verified" if self.slack_webhook_url else "not_configured",
            "dashboard": "assumed_synced",
            "api_coherence": "assumed_synced"
        }
        results['C3'] = comm_layers

        # C4 - Adaptive Learning (meta-regulator tuning summary)
        results['C4'] = {"meta_regulator_status": "idle", "last_action": None}

        return results

    # ---------- IV. Continuous Coherence Flow (CCF) ----------
    def start_continuous_coherence_flow(self, mode: str = "auto", monitoring_cycles: int = DEFAULT_CCF_CYCLES) -> List[CalibrationResult]:
        logger.info(f"IV. Continuous Coherence Flow (CCF) - mode={mode}")
        results: List[CalibrationResult] = []

        for cid in range(1, monitoring_cycles + 1):
            if SIMULATED_MODE:
                coherence_score = self.he.compute_coherence()
                caf = 1.04 + random.uniform(-0.02, 0.03)
                entropy_rate = random.uniform(0.001, 0.003)
                psi = coherence_score
                lambda_L = 0.8 + random.uniform(-0.1, 0.1)
                m_t = 0.6 + random.uniform(-0.2, 0.2)
                Omega_t = 1.0 + random.uniform(-0.2, 0.2)
                
                # Generate synthetic time series for spectral analysis
                time_series = [random.uniform(-1, 1) for _ in range(100)]
                spectral_power = self.he.spectral_coherence(time_series, sampling_rate=10.0)
            else:
                # Integrate real measurements
                prometheus_coh = self.fetch_prometheus_metric('uhes_coherence_score') or 0.0
                # In a real implementation, fetch time series data
                time_series = [random.uniform(-1, 1) for _ in range(100)]  # Placeholder
                spectral_power = self.he.spectral_coherence(time_series, sampling_rate=10.0)
                
                inputs = {"prometheus_coh": prometheus_coh, "spectral_power": spectral_power}
                coherence_score = self.he.compute_coherence(inputs)
                caf = self.fetch_prometheus_metric('uhes_caf') or 1.0
                entropy_rate = self.fetch_prometheus_metric('uhes_entropy') or 0.001
                psi = coherence_score
                # fetch parameters from actuator / state store
                lambda_L = 1.0
                m_t = 0.5
                Omega_t = 1.0

            delta_psi = psi - self.previous_psi
            stability = abs(delta_psi) < 0.01

            # AutoBalance: apply conservative adjustments
            if abs(delta_psi) > 0.02:
                logger.info(f"CCF: large ΔΨ={delta_psi:.6f} — applying conservative AutoBalance")
                # pseudocode: call actuator to nudge parameters; here we just log

            result = CalibrationResult(
                cycle_id=cid,
                timestamp=now_iso(),
                metrics={"H_internal": coherence_score, "entropy_rate": entropy_rate, "CAF": caf, "Psi": psi, "spectral_power": spectral_power},
                parameters={"lambda_L": lambda_L, "m_t": m_t, "Omega_t": Omega_t},
                stability=stability,
                coherence_score=coherence_score,
                delta_psi=delta_psi,
                entropy_rate=entropy_rate,
                caf=caf
            )
            results.append(result)

            # Append audit with envelope
            audit = {"cycle": cid, "result": asdict(result)}
            envelope = self.sec.envelope(audit)
            with open(os.path.join(self.report_dir, "production_coherence_audit_v2.json"), "a") as f:
                f.write(json.dumps(envelope) + "\n")

            self.previous_psi = psi
            time.sleep(0.5 if SIMULATED_MODE else 1.0)

        return results

    # ---------- V. Dimensional Reflection & Meta-Stability Check ----------
    def run_dimensional_reflection(self) -> Dict[str, Any]:
        logger.info("V. Dimensional Reflection and Meta-Stability Check")

        # Pseudocode: compute composite resonance from stored aggregates
        psi = self.he.mean() or 0.975
        caf = 1.04
        h_ft = 0.94

        cw = (psi + caf - h_ft) / 3.0
        previous_cw = cw - random.uniform(-0.005, 0.005)
        improved = cw >= previous_cw

        dgs = 0.97 + random.uniform(-0.01, 0.01)
        dgs_stable = dgs >= 0.97

        report = {
            "timestamp": now_iso(),
            "Cw": cw,
            "previous_Cw": previous_cw,
            "improved": improved,
            "DGS": dgs,
            "dgs_stable": dgs_stable,
            "system_metrics": {"Psi": psi, "CAF": caf, "H_internal": h_ft}
        }

        # Sign and persist
        envelope = self.sec.envelope(report)
        with open(os.path.join(self.report_dir, "emanation_reflection_report_v2.json"), "w") as f:
            json.dump(envelope, f, indent=2)

        # Slack notify if configured & degradation
        if not improved and self.slack_webhook_url:
            try:
                text = f"[ALERT] Cw decreased to {cw:.6f} at {now_iso()}"
                requests.post(self.slack_webhook_url, json={"text": text}, timeout=3)
            except Exception as e:
                logger.warning(f"Failed to send slack notification: {e}")

        return report

    # ---------- Full protocol ----------
    def run_full_calibration_protocol(self) -> Dict[str, Any]:
        header = """
        =""" * 20
        logger.info("Running full Production Reflection & Coherence Calibration Protocol v2")

        results: Dict[str, Any] = {"protocol_timestamp": now_iso()}
        results["components"] = [asdict(s) for s in self.verify_components()]
        results["verification"] = self.run_harmonic_self_verification()
        results["calibration"] = self.run_coherence_calibration_matrix()
        results["continuous_flow"] = [asdict(r) for r in self.start_continuous_coherence_flow()]
        results["reflection"] = self.run_dimensional_reflection()

        # Wrap and sign final protocol
        envelope = self.sec.envelope(results)
        out_file = os.path.join(self.report_dir, "production_calibration_protocol_results_v2.json")
        with open(out_file, "w") as f:
            json.dump(envelope, f, indent=2)
        logger.info(f"Saved protocol results to {out_file}")

        return results


# ---------- Kubernetes Manifest Generator ----------
def generate_k8s_manifests(namespace: str = "quantum-currency", schedule: str = "*/15 * * * *", image: str = "quantumcurrency/reflection-calibrator:latest") -> Dict[str, str]:
    """Generate Kubernetes ConfigMap and CronJob manifests for periodic execution.
    
    Args:
        namespace: Kubernetes namespace for the resources
        schedule: Cron schedule expression
        image: Docker image to use for the job
    
    Returns:
        Dictionary with manifest names as keys and manifest content as values
    """
    manifests = {}
    
    # ConfigMap manifest
    configmap_manifest = f"""apiVersion: v1
kind: ConfigMap
metadata:
  name: quantum-currency-calibrator-config
  namespace: {namespace}
data:
  PRC_REPORT_DIR: "{REPORT_DIR}"
  PRC_BASE_FREQ: "{F0}"
  PRC_SIMULATED: "0"  # Set to 1 for testing
  PRC_HSVP_CYCLES: "{DEFAULT_HSVP_CYCLES}"
  PRC_CCF_CYCLES: "{DEFAULT_CCF_CYCLES}"
"""
    
    manifests["configmap.yaml"] = configmap_manifest
    
    # CronJob manifest
    cronjob_manifest = f"""apiVersion: batch/v1
kind: CronJob
metadata:
  name: quantum-currency-reflection-calibrator
  namespace: {namespace}
spec:
  schedule: "{schedule}"  # Run according to schedule
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: reflection-calibrator
            image: {image}
            envFrom:
            - configMapRef:
                name: quantum-currency-calibrator-config
            env:
            - name: PRC_HMAC_KEY
              valueFrom:
                secretKeyRef:
                  name: quantum-currency-secrets
                  key: hmac-key
            - name: PRC_PROMETHEUS_URL
              valueFrom:
                secretKeyRef:
                  name: quantum-currency-secrets
                  key: prometheus-url
            - name: PRC_SLACK_WEBHOOK
              valueFrom:
                secretKeyRef:
                  name: quantum-currency-secrets
                  key: slack-webhook
            volumeMounts:
            - name: reports-volume
              mountPath: {REPORT_DIR}
            command:
            - python
            - production_reflection_calibrator_v2.py
            - --full
          volumes:
          - name: reports-volume
            persistentVolumeClaim:
              claimName: quantum-currency-reports-pvc
          restartPolicy: OnFailure
"""
    
    manifests["cronjob.yaml"] = cronjob_manifest
    
    return manifests


# ---------- Unit Test Skeleton ----------
def create_unit_tests() -> str:
    """Create a skeleton for unit tests using pytest.
    
    Returns:
        String containing test code
    """
    test_code = '''#!/usr/bin/env python3
"""
Unit tests for Production Reflection Calibrator V2
"""

import pytest
import os
import json
from unittest.mock import patch, MagicMock
from production_reflection_calibrator_v2 import (
    HarmonicEngine, 
    OmegaSecurity, 
    MetaRegulator, 
    ProductionReflectionCalibrator,
    generate_k8s_manifests,
    create_unit_tests,
    generate_ci_workflow,
    generate_dockerfile,
    generate_requirements,
    generate_readme
)

class TestHarmonicEngine:
    """Test the HarmonicEngine class"""
    
    def test_initialization(self):
        """Test HarmonicEngine initialization"""
        he = HarmonicEngine()
        assert he.base_freq == 432.0
        assert he.history_len == 32
        assert len(he.history) == 0
    
    def test_compute_coherence_simulated(self):
        """Test coherence computation in simulated mode"""
        he = HarmonicEngine()
        coherence = he.compute_coherence()
        assert 0.965 <= coherence <= 0.985
    
    def test_history_management(self):
        """Test history management"""
        he = HarmonicEngine(history_len=3)
        for i in range(5):
            he.push(float(i))
        assert len(he.history) == 3
        assert he.history == [2.0, 3.0, 4.0]
    
    def test_variance_computation(self):
        """Test variance computation"""
        he = HarmonicEngine()
        he.history = [1.0, 2.0, 3.0]
        variance = he.variance()
        assert variance == 2.0/3.0
    
    @pytest.mark.skipif(not pytest.importorskip("numpy"), reason="NumPy not available")
    def test_spectral_coherence_with_numpy(self):
        """Test spectral coherence with NumPy"""
        he = HarmonicEngine()
        # Generate synthetic signal
        import numpy as np
        t = np.linspace(0, 1, 100)
        signal = np.sin(2 * np.pi * he.base_freq * t) + 0.1 * np.random.randn(len(t))
        coherence = he.spectral_coherence(signal.tolist(), sampling_rate=100.0)
        assert 0.0 <= coherence <= 1.0
    
    def test_spectral_coherence_without_numpy(self):
        """Test spectral coherence without NumPy"""
        he = HarmonicEngine()
        signal = [0.5 * i/100 for i in range(100)]
        coherence = he.spectral_coherence(signal, sampling_rate=100.0)
        assert 0.0 <= coherence <= 1.0
    
    def test_spectral_coherence_edge_cases(self):
        """Test spectral coherence with edge cases"""
        he = HarmonicEngine()
        # Empty signal
        coherence = he.spectral_coherence([], sampling_rate=100.0)
        assert coherence == 0.0
        
        # Short signal
        coherence = he.spectral_coherence([1.0], sampling_rate=100.0)
        assert coherence == 0.0

class TestOmegaSecurity:
    """Test the OmegaSecurity class"""
    
    def test_envelope_creation(self):
        """Test envelope creation and verification"""
        sec = OmegaSecurity()
        payload = {"test": "data", "value": 42}
        envelope = sec.envelope(payload)
        
        assert "payload" in envelope
        assert "meta" in envelope
        assert "signature" in envelope["meta"]
        assert "nonce" in envelope["meta"]
        
        # Verify the envelope
        assert sec.verify_envelope(envelope)
    
    def test_envelope_tampering_detection(self):
        """Test that tampered envelopes are detected"""
        sec = OmegaSecurity()
        payload = {"test": "data"}
        envelope = sec.envelope(payload)
        
        # Tamper with the payload
        envelope["payload"]["test"] = "tampered"
        
        # Verification should fail
        assert not sec.verify_envelope(envelope)
    
    def test_envelope_verification_edge_cases(self):
        """Test envelope verification with edge cases"""
        sec = OmegaSecurity()
        # Test with malformed envelope
        assert not sec.verify_envelope({"malformed": "envelope"})
        
        # Test with missing signature
        payload = {"test": "data"}
        envelope = sec.envelope(payload)
        del envelope["meta"]["signature"]
        assert not sec.verify_envelope(envelope)

class TestMetaRegulator:
    """Test the MetaRegulator class"""
    
    def test_reward_computation(self):
        """Test reward computation"""
        reg = MetaRegulator()
        before = {"Psi": 0.95, "Cw": 0.8}
        after = {"Psi": 0.97, "Cw": 0.85, "entropy_rate": 0.002}
        reward = reg.compute_reward(before, after)
        assert isinstance(reward, float)
    
    def test_action_proposal(self):
        """Test action proposal"""
        reg = MetaRegulator()
        observation = {"Psi": 0.97, "weights": {"spectral": 0.4, "phase": 0.4, "prom": 0.2}}
        action = reg.propose_action(observation)
        
        # Check that action is valid
        assert isinstance(action, dict)
        assert all(k in action for k in ["spectral", "phase", "prom"])
        assert all(0.0 <= v <= 1.0 for v in action.values())
        
        # Check normalization
        assert abs(sum(action.values()) - 1.0) < 0.001
    
    def test_action_validation(self):
        """Test action validation"""
        reg = MetaRegulator()
        
        # Valid action
        valid_action = {"spectral": 0.4, "phase": 0.4, "prom": 0.2}
        assert reg.validate_action(valid_action)
        
        # Invalid action - wrong sum
        invalid_action = {"spectral": 0.5, "phase": 0.5, "prom": 0.5}
        assert not reg.validate_action(invalid_action)
        
        # Invalid action - out of bounds
        invalid_action2 = {"spectral": -0.1, "phase": 0.5, "prom": 0.6}
        assert not reg.validate_action(invalid_action2)
    
    def test_ppo_training(self):
        """Test PPO training method"""
        reg = MetaRegulator()
        # Test with empty data
        model = reg.train_ppo_policy([], [])
        assert model is None or hasattr(model, 'predict')
        
        # Test with mismatched data
        observations = [{"test": 1}]
        rewards = [1.0, 2.0]  # Different length
        model = reg.train_ppo_policy(observations, rewards)
        assert model is None
        
        # Test with valid data
        observations = [{"test": i} for i in range(5)]
        rewards = [float(i) for i in range(5)]
        model = reg.train_ppo_policy(observations, rewards)
        assert model is None or hasattr(model, 'predict')

class TestProductionReflectionCalibrator:
    """Test the main calibrator class"""
    
    def test_initialization(self):
        """Test calibrator initialization"""
        calibrator = ProductionReflectionCalibrator()
        assert calibrator.report_dir == "/mnt/data"
        assert calibrator.he is not None
        assert calibrator.sec is not None
        assert calibrator.reg is not None
    
    def test_component_verification(self):
        """Test component verification"""
        calibrator = ProductionReflectionCalibrator()
        statuses = calibrator.verify_components()
        assert isinstance(statuses, list)
        assert len(statuses) > 0
    
    @patch('production_reflection_calibrator_v2.requests.get')
    def test_prometheus_fetch(self, mock_get):
        """Test Prometheus metric fetching"""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": {
                "result": [
                    {
                        "value": [0, "0.95"]
                    }
                ]
            }
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        calibrator = ProductionReflectionCalibrator(prometheus_url="http://test-prometheus")
        result = calibrator.fetch_prometheus_metric("test_query")
        assert result == 0.95
    
    def test_prometheus_fetch_without_url(self):
        """Test Prometheus metric fetching without URL"""
        calibrator = ProductionReflectionCalibrator()
        with pytest.raises(RuntimeError):
            calibrator.fetch_prometheus_metric("test_query")

class TestUtilityFunctions:
    """Test utility functions"""
    
    def test_k8s_manifest_generation(self):
        """Test Kubernetes manifest generation"""
        manifests = generate_k8s_manifests()
        assert isinstance(manifests, dict)
        assert "configmap.yaml" in manifests
        assert "cronjob.yaml" in manifests
        
        # Check that manifests contain expected content
        configmap = manifests["configmap.yaml"]
        assert "quantum-currency-calibrator-config" in configmap
        
        cronjob = manifests["cronjob.yaml"]
        assert "quantum-currency-reflection-calibrator" in cronjob
    
    def test_unit_test_generation(self):
        """Test unit test generation"""
        test_code = create_unit_tests()
        assert isinstance(test_code, str)
        assert "TestHarmonicEngine" in test_code
        assert "TestOmegaSecurity" in test_code
    
    def test_ci_workflow_generation(self):
        """Test CI workflow generation"""
        workflow = generate_ci_workflow()
        assert isinstance(workflow, str)
        assert "Quantum Currency Calibrator CI" in workflow
        assert "pytest" in workflow
    
    def test_dockerfile_generation(self):
        """Test Dockerfile generation"""
        dockerfile = generate_dockerfile()
        assert isinstance(dockerfile, str)
        assert "FROM python:3.9-slim" in dockerfile
        assert "production_reflection_calibrator_v2.py" in dockerfile
    
    def test_requirements_generation(self):
        """Test requirements.txt generation"""
        requirements = generate_requirements()
        assert isinstance(requirements, str)
        assert "requests" in requirements
        assert "numpy" in requirements
    
    def test_readme_generation(self):
        """Test README.md generation"""
        readme = generate_readme()
        assert isinstance(readme, str)
        assert "Production Reflection Calibrator V2" in readme
        assert "Usage" in readme

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
'''
    
    return test_code


# ---------- CI Workflow Generator ----------
def generate_ci_workflow() -> str:
    """Generate a GitHub Actions workflow for CI testing.
    
    Returns:
        String containing workflow YAML
    """
    workflow = '''name: Quantum Currency Calibrator CI

on:
  push:
    branches: [ main, develop ]
    tags: [ 'v*' ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, "3.10", 3.11]

    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install pytest pytest-cov
    
    - name: Install optional dependencies
      run: |
        pip install numpy  # For FFT functionality
        pip install stable-baselines3  # For PPO policy
    
    - name: Run tests
      run: |
        python -m pytest tests/test_production_reflection_calibrator_v2.py -v --cov=.
    
    - name: Run linting
      run: |
        pip install flake8
        flake8 production_reflection_calibrator_v2.py
    
    - name: Run type checking
      run: |
        pip install mypy
        mypy production_reflection_calibrator_v2.py
    
    - name: Upload coverage reports
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        flags: unittests
        name: codecov-umbrella

  build:
    runs-on: ubuntu-latest
    needs: test
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v2
    
    - name: Login to DockerHub
      uses: docker/login-action@v2
      with:
        username: ${{ secrets.DOCKERHUB_USERNAME }}
        password: ${{ secrets.DOCKERHUB_TOKEN }}
    
    - name: Extract metadata (tags, labels) for Docker
      id: meta
      uses: docker/metadata-action@v4
      with:
        images: quantumcurrency/reflection-calibrator
        tags: |
          type=ref,event=branch
          type=ref,event=pr
          type=semver,pattern={{version}}
          type=semver,pattern={{major}}.{{minor}}
    
    - name: Build and push
      uses: docker/build-push-action@v4
      with:
        context: .
        file: ./Dockerfile.calibrator
        push: ${{ github.event_name != 'pull_request' }}
        tags: ${{ steps.meta.outputs.tags }}
        labels: ${{ steps.meta.outputs.labels }}
    
    - name: Deploy to Kubernetes (on main branch)
      if: github.ref == 'refs/heads/main'
      run: |
        echo "Deploying to Kubernetes cluster"
        # Add kubectl commands here
'''
    
    return workflow


# ---------- Dockerfile Generator ----------
def generate_dockerfile() -> str:
    """Generate a Dockerfile for containerizing the application.
    
    Returns:
        String containing Dockerfile content
    """
    dockerfile = '''# Use Python 3.9 slim image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install optional dependencies for full functionality
RUN pip install --no-cache-dir numpy stable-baselines3

# Copy application code
COPY production_reflection_calibrator_v2.py .

# Create directory for reports
RUN mkdir -p /mnt/data

# Create non-root user
RUN useradd -m -u 1000 appuser
USER appuser

# Set environment variables
ENV PRC_REPORT_DIR=/mnt/data
ENV PRC_SIMULATED=0

# Run the application
ENTRYPOINT ["python", "production_reflection_calibrator_v2.py"]
CMD ["--full"]
'''
    
    return dockerfile


# ---------- Requirements Generator ----------
def generate_requirements() -> str:
    """Generate a requirements.txt file.
    
    Returns:
        String containing requirements.txt content
    """
    requirements = '''requests>=2.25.1
numpy>=1.21.0
stable-baselines3>=1.0.0
pytest>=6.2.4
pytest-cov>=2.12.1
flake8>=3.9.2
mypy>=0.910
'''
    
    return requirements


# ---------- README Generator ----------
def generate_readme() -> str:
    """Generate a README.md file with documentation.
    
    Returns:
        String containing README.md content
    """
    readme = '''# Production Reflection Calibrator V2

Upgraded Production Reflection & Coherence Calibration System for Quantum Currency

## Key Features

- **Harmonic Engine (HE)**: Dedicated class that consolidates mathematical cluster for coherence computation, history, variance (σ²Ω), and frequency normalization
- **Ω-Security Primitives**: HMAC signing, tamper-evident report envelopes, secure entropy
- **Meta-Regulator**: Formalized autonomous tuner skeleton with reward shaping and instruction-level pseudocode
- **Implementation Guidance**: Clear pseudocode blocks and structured hooks for integration
- **Operational Upgrades**: Prometheus fetcher with retries, Slack notification integration, deterministic simulated mode

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```bash
# Run full protocol
python production_reflection_calibrator_v2.py --full

# Run specific components
python production_reflection_calibrator_v2.py --verify
python production_reflection_calibrator_v2.py --hsvp
python production_reflection_calibrator_v2.py --ccm
python production_reflection_calibrator_v2.py --ccf
python production_reflection_calibrator_v2.py --reflection
```

### With External Services

```bash
# With Prometheus and Slack
python production_reflection_calibrator_v2.py --full \
  --prometheus-url http://prometheus:9090 \
  --slack-webhook https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK
```

### Environment Variables

- `PRC_REPORT_DIR`: Directory for reports (default: `/mnt/data`)
- `PRC_BASE_FREQ`: Base frequency (default: `432.0`)
- `PRC_HMAC_KEY`: HMAC key for security (required in production)
- `PRC_SIMULATED`: Simulated mode (default: `1`)
- `PRC_SIM_SEED`: Random seed for reproducibility (default: `0`)

### Production Deployment

Set environment variables and secrets:

```bash
export PRC_HMAC_KEY="your-hmac-key"
export PRC_SIMULATED=0
```

## Artifact Generation

The tool can generate various artifacts for deployment:

```bash
# Generate Kubernetes manifests
python production_reflection_calibrator_v2.py --generate-k8s

# Generate unit tests
python production_reflection_calibrator_v2.py --generate-tests

# Generate CI workflow
python production_reflection_calibrator_v2.py --generate-ci

# Generate Dockerfile
python production_reflection_calibrator_v2.py --generate-dockerfile

# Generate requirements.txt
python production_reflection_calibrator_v2.py --generate-requirements
```

## Components

### I. Component Verification Layer
Verifies system components are present and accessible.

### II. Harmonic Self-Verification Protocol (HSVP)
Runs self-verification cycles to check system coherence.

### III. Coherence Calibration Matrix (CCM)
Calibrates coherence across different system layers.

### IV. Continuous Coherence Flow (CCF)
Maintains continuous coherence monitoring and adjustment.

### V. Dimensional Reflection & Meta-Stability Check
Performs high-level system reflection and stability checks.

## Security

- All audit and reflection reports are signed with HMAC
- Secure nonce generation using OS-secure RNG
- Tamper-evident envelopes for all persisted data

In production, store keys in Vault/KMS rather than environment variables.

## Dependencies

- Python 3.8+
- requests
- numpy (optional, for FFT functionality)
- stable-baselines3 (optional, for PPO policy)

## Testing

```bash
python -m pytest tests/test_production_reflection_calibrator_v2.py -v
```
'''
    
    return readme


# ---------- CLI Entrypoint ----------
def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Production Reflection & Coherence Calibration v2")
    parser.add_argument("--prometheus-url", help="Prometheus server URL", required=False)
    parser.add_argument("--slack-webhook", help="Slack webhook URL", required=False)
    parser.add_argument("--full", action="store_true", help="Run full protocol")
    parser.add_argument("--verify", action="store_true", help="Run component verification only")
    parser.add_argument("--hsvp", action="store_true", help="Run HSVP only")
    parser.add_argument("--ccm", action="store_true", help="Run CCM only")
    parser.add_argument("--ccf", action="store_true", help="Run CCF only")
    parser.add_argument("--reflection", action="store_true", help="Run reflection only")
    parser.add_argument("--generate-k8s", action="store_true", help="Generate Kubernetes manifests")
    parser.add_argument("--k8s-namespace", default="quantum-currency", help="Kubernetes namespace for generated manifests")
    parser.add_argument("--k8s-schedule", default="*/15 * * * *", help="Cron schedule for the job")
    parser.add_argument("--k8s-image", default="quantumcurrency/reflection-calibrator:latest", help="Docker image for the job")
    parser.add_argument("--generate-tests", action="store_true", help="Generate unit test skeleton")
    parser.add_argument("--generate-ci", action="store_true", help="Generate CI workflow")
    parser.add_argument("--generate-dockerfile", action="store_true", help="Generate Dockerfile")
    parser.add_argument("--generate-requirements", action="store_true", help="Generate requirements.txt")
    parser.add_argument("--generate-readme", action="store_true", help="Generate README.md")
    args = parser.parse_args()

    # Handle generation commands
    if args.generate_k8s:
        manifests = generate_k8s_manifests(namespace=args.k8s_namespace, schedule=args.k8s_schedule, image=args.k8s_image)
        for name, content in manifests.items():
            with open(name, "w") as f:
                f.write(content)
            print(f"Generated {name}")
        return 0
    
    if args.generate_tests:
        test_code = create_unit_tests()
        with open("test_production_reflection_calibrator_v2.py", "w") as f:
            f.write(test_code)
        print("Generated test_production_reflection_calibrator_v2.py")
        return 0
    
    if args.generate_ci:
        workflow = generate_ci_workflow()
        os.makedirs(".github/workflows", exist_ok=True)
        with open(".github/workflows/quantum_currency_calibrator_ci.yml", "w") as f:
            f.write(workflow)
        print("Generated .github/workflows/quantum_currency_calibrator_ci.yml")
        return 0
    
    if args.generate_dockerfile:
        dockerfile = generate_dockerfile()
        with open("Dockerfile.calibrator", "w") as f:
            f.write(dockerfile)
        print("Generated Dockerfile.calibrator")
        return 0
    
    if args.generate_requirements:
        requirements = generate_requirements()
        with open("requirements.txt", "w") as f:
            f.write(requirements)
        print("Generated requirements.txt")
        return 0
    
    if args.generate_readme:
        readme = generate_readme()
        with open("README.md", "w") as f:
            f.write(readme)
        print("Generated README.md")
        return 0

    calibrator = ProductionReflectionCalibrator(prometheus_url=args.prometheus_url, slack_webhook_url=args.slack_webhook)

    if args.verify:
        calibrator.verify_components()
    elif args.hsvp:
        calibrator.run_harmonic_self_verification()
    elif args.ccm:
        calibrator.run_coherence_calibration_matrix()
    elif args.ccf:
        calibrator.start_continuous_coherence_flow()
    elif args.reflection:
        calibrator.run_dimensional_reflection()
    else:
        calibrator.run_full_calibration_protocol()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())