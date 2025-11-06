#!/usr/bin/env python3
"""
Harmonic Validation Module for Quantum Currency System
Implements Recursive Φ-Resonance Validation (RΦV) for coherence-based consensus
This module provides functions to:
1. Compute frequency spectra from time series data
2. Calculate coherence scores between nodes
3. Perform recursive validation with φ-scaling and λ-decay
4. Generate harmonic snapshots and proof bundles
5. Integrate with token economy based on coherence scores

Part of the Open-A.G.I framework for quantum-harmonic currency
"""

import numpy as np
from typing import List, Dict, Tuple, Any, Optional
import json
import hashlib
import hmac
import time
from dataclasses import dataclass, asdict
from scipy.signal import csd

# Golden ratio and related constants
PHI = 1.618033988749895
LAMBDA = 1.0 / PHI


@dataclass
class HarmonicSnapshot:
    """Represents a snapshot of harmonic data from a node"""
    node_id: str
    timestamp: float
    times: List[float]
    values: List[float]
    spectrum: List[Tuple[float, float]]  # [(frequency, amplitude), ...]
    spectrum_hash: str
    CS: float  # Coherence Score
    phi_params: Dict[str, float]
    signature: Optional[str] = None


@dataclass
class HarmonicProofBundle:
    """Bundle of harmonic snapshots with aggregated coherence score"""
    snapshots: List[HarmonicSnapshot]
    aggregated_CS: float
    aggregator_signature: Optional[str] = None
    timestamp: float = 0.0

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


def compute_spectrum(times: np.ndarray, values: np.ndarray) -> List[Tuple[float, float]]:
    """
    Compute frequency spectrum using FFT
    
    Args:
        times: Time series timestamps
        values: Time series values
        
    Returns:
        List of (frequency, amplitude) tuples
    """
    # Apply windowing to reduce spectral leakage
    windowed_values = values * np.hanning(len(values))
    
    # Compute FFT
    freqs = np.fft.rfftfreq(len(times), d=(times[1] - times[0]))
    amps = np.abs(np.fft.rfft(windowed_values))
    
    # Normalize amplitudes
    if np.max(amps) > 0:
        amps = amps / np.max(amps)
    
    # Return as list of tuples
    return list(zip(freqs, amps))


def pairwise_coherence(x1: np.ndarray, x2: np.ndarray, fs: float) -> float:
    """
    Compute pairwise coherence between two time series using cross-spectral density
    
    Args:
        x1: First time series
        x2: Second time series
        fs: Sampling frequency
        
    Returns:
        Coherence score between 0 and 1
    """
    # Ensure same length
    min_len = min(len(x1), len(x2))
    x1 = x1[:min_len]
    x2 = x2[:min_len]
    
    # Compute cross-spectral density
    f, Pxy = csd(x1, x2, fs=fs)
    f, Pxx = csd(x1, x1, fs=fs)
    f, Pyy = csd(x2, x2, fs=fs)
    
    # Compute coherence
    coherence = np.abs(Pxy)**2 / (Pxx * Pyy + 1e-18)
    
    # Return mean coherence (normalized)
    return float(np.mean(coherence.real))


def compute_coherence_score(
    local_snapshot: HarmonicSnapshot, 
    remote_snapshots: List[HarmonicSnapshot]
) -> float:
    """
    Compute aggregated coherence score between local node and remote nodes
    
    Args:
        local_snapshot: Local node's harmonic snapshot
        remote_snapshots: List of remote nodes' snapshots
        
    Returns:
        Aggregated coherence score (0-1)
    """
    if not remote_snapshots:
        return 0.0
    
    # Convert local data to numpy arrays
    local_times = np.array(local_snapshot.times)
    local_values = np.array(local_snapshot.values)
    fs = 1.0 / (local_times[1] - local_times[0]) if len(local_times) > 1 else 1.0
    
    coherence_scores = []
    
    for remote in remote_snapshots:
        # Convert remote data to numpy arrays
        remote_times = np.array(remote.times)
        remote_values = np.array(remote.values)
        
        # Resample to match lengths if needed
        min_len = min(len(local_values), len(remote_values))
        if min_len == 0:
            continue
            
        local_resampled = local_values[:min_len]
        remote_resampled = remote_values[:min_len]
        
        # Compute pairwise coherence
        coh = pairwise_coherence(local_resampled, remote_resampled, fs)
        coherence_scores.append(coh)
    
    if not coherence_scores:
        return 0.0
        
    # Return mean coherence score
    return float(np.mean(coherence_scores))


def apply_recursive_decay(
    coherence_series: List[float], 
    tau_steps: int = 3
) -> float:
    """
    Apply recursive λ-weighted sum to coherence samples
    
    Args:
        coherence_series: List of coherence values at successive delay steps
        tau_steps: Number of delay steps to consider
        
    Returns:
        Recursively weighted coherence score
    """
    if not coherence_series:
        return 0.0
        
    total = 0.0
    weight_sum = 0.0
    
    for n, val in enumerate(coherence_series[:tau_steps]):
        # Weight = (1/φ^n) * (λ^n) = (1/φ^n) * (1/φ)^n = (1/φ^2)^n
        weight = (1.0 / (PHI ** (2 * n)))
        total += weight * val
        weight_sum += weight
    
    if weight_sum == 0:
        return 0.0
        
    return total / weight_sum


def recursive_validate(
    snapshot_bundle: List[HarmonicSnapshot],
    threshold: float = 0.75
) -> Tuple[bool, Optional[HarmonicProofBundle]]:
    """
    Perform recursive validation on a bundle of harmonic snapshots
    
    Args:
        snapshot_bundle: List of harmonic snapshots from multiple nodes
        threshold: Minimum coherence score required for validation
        
    Returns:
        Tuple of (is_valid, proof_bundle)
    """
    if len(snapshot_bundle) < 2:
        return False, None
    
    # Compute pairwise coherences
    coherences = []
    
    for i, local_snapshot in enumerate(snapshot_bundle):
        # Get remote snapshots (all except local)
        remote_snapshots = [s for j, s in enumerate(snapshot_bundle) if i != j]
        
        # Compute coherence score for this node against all others
        cs = compute_coherence_score(local_snapshot, remote_snapshots)
        coherences.append(cs)
    
    # Apply recursive decay to get aggregated score
    aggregated_CS = apply_recursive_decay(coherences)
    
    # Create proof bundle
    proof_bundle = HarmonicProofBundle(
        snapshots=snapshot_bundle,
        aggregated_CS=aggregated_CS
    )
    
    # Validate against threshold
    is_valid = aggregated_CS >= threshold
    
    return is_valid, proof_bundle


def make_snapshot(
    node_id: str,
    times: List[float],
    values: List[float],
    secret_key: Optional[str] = None
) -> HarmonicSnapshot:
    """
    Create a harmonic snapshot with optional signature
    
    Args:
        node_id: ID of the node creating the snapshot
        times: Time series timestamps
        values: Time series values
        secret_key: Optional secret key for signing
        
    Returns:
        HarmonicSnapshot object
    """
    # Convert to numpy arrays for processing
    times_array = np.array(times)
    values_array = np.array(values)
    
    # Compute spectrum
    spectrum = compute_spectrum(times_array, values_array)
    
    # Compute coherence score (placeholder - would be computed during validation)
    CS = 0.0
    
    # Create hash of spectrum data
    spectrum_str = json.dumps(spectrum, sort_keys=True)
    spectrum_hash = hashlib.sha256(spectrum_str.encode()).hexdigest()
    
    # Create snapshot
    snapshot = HarmonicSnapshot(
        node_id=node_id,
        timestamp=time.time(),
        times=times,
        values=values,
        spectrum=spectrum,
        spectrum_hash=spectrum_hash,
        CS=CS,
        phi_params={
            "phi": PHI,
            "lambda": LAMBDA,
            "tau": 1.0  # Placeholder value
        }
    )
    
    # Add signature if secret key provided
    if secret_key:
        snapshot_data = f"{node_id}{snapshot.timestamp}{spectrum_hash}".encode()
        snapshot.signature = hmac.new(
            secret_key.encode(), 
            snapshot_data, 
            hashlib.sha256
        ).hexdigest()
    
    return snapshot


def update_snapshot_coherence(snapshot: HarmonicSnapshot, CS: float) -> HarmonicSnapshot:
    """
    Update a snapshot with its computed coherence score
    
    Args:
        snapshot: The snapshot to update
        CS: The coherence score to set
        
    Returns:
        Updated snapshot
    """
    snapshot.CS = CS
    return snapshot


def calculate_token_rewards(coherence_score: float, validator_chr_score: float) -> Dict[str, float]:
    """
    Calculate token rewards based on coherence score and validator CHR reputation
    
    Args:
        coherence_score: The coherence score (0-1)
        validator_chr_score: The validator's CHR reputation score (0-1)
        
    Returns:
        Dictionary with token rewards {token_type: amount}
    """
    # Base rewards
    base_flx_reward = 100.0
    base_chr_reward = 50.0
    base_atr_reward = 25.0
    base_psy_reward = 10.0
    base_res_reward = 5.0
    
    # Multipliers based on coherence score
    coherence_multiplier = 0.5 + coherence_score * 0.5  # 0.5-1.0x
    
    # Multipliers based on validator CHR score
    chr_multiplier = 0.8 + validator_chr_score * 0.4  # 0.8-1.2x
    
    # Combined multiplier
    combined_multiplier = coherence_multiplier * chr_multiplier
    
    rewards = {
        "FLX": base_flx_reward * combined_multiplier,
        "CHR": base_chr_reward * combined_multiplier,
        "ATR": base_atr_reward * combined_multiplier,
        "PSY": base_psy_reward * combined_multiplier,
        "RES": base_res_reward * combined_multiplier
    }
    
    return rewards


def validate_harmonic_transaction(tx: Dict[str, Any], config: Dict[str, float]) -> bool:
    """
    Validate a harmonic transaction based on coherence score and CHR reputation
    
    Args:
        tx: Transaction dictionary with coherence and CHR data
        config: Configuration with thresholds
        
    Returns:
        True if transaction is valid, False otherwise
    """
    cs = tx.get("aggregated_cs", 0)
    chr_score = tx.get("sender_chr", 0)
    mint_th = config.get("mint_threshold", 0.75)
    chr_th = config.get("min_chr", 0.6)

    # Both coherence and reputation must be high enough
    if cs >= mint_th and chr_score >= chr_th:
        return True
    return False


def generate_test_signal(freq: float, phase: float, duration: float, sample_rate: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a test sinusoidal signal
    
    Args:
        freq: Frequency in Hz
        phase: Phase in radians
        duration: Duration in seconds
        sample_rate: Samples per second
        
    Returns:
        Tuple of (times, values) arrays
    """
    t = np.linspace(0, duration, int(duration * sample_rate))
    x = np.sin(2 * np.pi * freq * t + phase)
    return t, x


if __name__ == "__main__":
    # Example usage
    print("Harmonic Validation Module - Test")
    
    # Generate test signals
    t1, x1 = generate_test_signal(10, 0, 1.0, 1000)  # Node A: 10 Hz sine
    t2, x2 = generate_test_signal(10, 0, 1.0, 1000)  # Node B: 10 Hz sine (coherent)
    t3, x3 = generate_test_signal(15, 0, 1.0, 1000)  # Node C: 15 Hz sine (less coherent)
    
    # Add some noise
    x1 += 0.1 * np.random.randn(len(x1))
    x2 += 0.1 * np.random.randn(len(x2))
    x3 += 0.2 * np.random.randn(len(x3))
    
    # Create snapshots
    snapshot_a = make_snapshot("node-A", t1.tolist(), x1.tolist(), "secretA")
    snapshot_b = make_snapshot("node-B", t2.tolist(), x2.tolist(), "secretB")
    snapshot_c = make_snapshot("node-C", t3.tolist(), x3.tolist(), "secretC")
    
    # Test coherence computation
    coherence_ab = pairwise_coherence(x1, x2, 1000)
    coherence_ac = pairwise_coherence(x1, x3, 1000)
    
    print(f"Coherence A-B (should be high): {coherence_ab:.3f}")
    print(f"Coherence A-C (should be lower): {coherence_ac:.3f}")
    
    # Test recursive validation
    bundle = [snapshot_a, snapshot_b, snapshot_c]
    is_valid, proof = recursive_validate(bundle, threshold=0.5)
    
    if proof:
        print(f"Aggregated Coherence Score: {proof.aggregated_CS:.3f}")
        print(f"Validation result: {'PASS' if is_valid else 'FAIL'}")
        
        # Test token rewards calculation
        rewards = calculate_token_rewards(proof.aggregated_CS, 0.85)
        print(f"Token rewards for coherence {proof.aggregated_CS:.3f}:")
        for token, amount in rewards.items():
            print(f"  {token}: {amount:.2f}")