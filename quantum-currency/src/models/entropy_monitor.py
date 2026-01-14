#!/usr/bin/env python3
"""
Entropy Monitor - Self-Healing & Entropic Decay System
Implements entropy monitoring, self-healing merge, and decay as evolution

This module provides:
1. Entropy monitoring for Quantum Packets
2. Self-healing merge protocols
3. Transmutation and evolution mechanisms
"""

import numpy as np
from typing import List, Dict, Tuple, Any, Optional, Union
import logging
import time
import math
from dataclasses import dataclass
from .quantum_memory import QuantumPacket, UnifiedFieldMemory, ScaleLevel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class EntropyMetrics:
    """Entropy metrics for a Quantum Packet"""
    spectral_entropy: float          # H(F_t) term from Coherence Scorer
    temporal_variance: float        # Variance over time
    dimensional_dispersion: float   # Dispersion across dimensions
    coherence_drift: float          # Rate of coherence change
    timestamp: float               # When metrics were computed

class EntropyMonitor:
    """
    Entropy Monitor - Tracks and manages entropy in the UFM system
    """
    
    def __init__(self, ufm: UnifiedFieldMemory, network_id: str = "quantum-currency-uhes-entropy-001"):
        self.network_id = network_id
        self.ufm = ufm  # Reference to Unified Field Memory
        self.entropy_history: Dict[str, List[EntropyMetrics]] = {}  # packet_id -> metrics history
        self.high_entropy_packets: Dict[str, float] = {}  # packet_id -> entropy value
        self.safety_thresholds = {
            "spectral_entropy_max": 2.0,      # Maximum allowed spectral entropy
            "coherence_drift_max": 0.1,       # Maximum allowed coherence drift per second
            "rebalancing_threshold": 1.5,     # Threshold to trigger rebalancing
            "transmutation_threshold": 3.0    # Threshold to trigger transmutation
        }
        
        logger.info(f"🔥 Entropy Monitor initialized for network: {network_id}")
    
    def compute_spectral_entropy(self, attention_spectrum: List[float]) -> float:
        """
        Compute spectral entropy H(F_t) from attention spectrum
        
        Args:
            attention_spectrum: Meta-attention spectrum
            
        Returns:
            float: Spectral entropy value
        """
        if not attention_spectrum or sum(attention_spectrum) == 0:
            return 0.0
        
        # Normalize spectrum
        normalized = np.array(attention_spectrum) / sum(attention_spectrum)
        
        # Compute entropy: H(F_t) = -Σ(p_i * log(p_i))
        entropy = -sum(p * math.log(p + 1e-10) for p in normalized if p > 0)
        return float(max(0.0, entropy))  # Ensure non-negative
    
    def compute_temporal_variance(self, packet_id: str, window_size: int = 5) -> float:
        """
        Compute temporal variance of a packet's metrics over time
        
        Args:
            packet_id: ID of packet to analyze
            window_size: Number of recent metrics to consider
            
        Returns:
            float: Temporal variance
        """
        if packet_id not in self.entropy_history:
            return 0.0
        
        history = self.entropy_history[packet_id][-window_size:]
        if len(history) < 2:
            return 0.0
        
        # Compute variance of spectral entropy over time
        entropy_values = [m.spectral_entropy for m in history]
        variance = float(np.var(entropy_values))
        return variance
    
    def compute_dimensional_dispersion(self, omega_vector: List[float]) -> float:
        """
        Compute dimensional dispersion of Ω vector
        
        Args:
            omega_vector: Ω state vector
            
        Returns:
            float: Dimensional dispersion
        """
        if not omega_vector:
            return 0.0
        
        # Compute variance across dimensions
        dispersion = float(np.var(omega_vector))
        return dispersion
    
    def compute_coherence_drift(self, packet_id: str, window_size: int = 3) -> float:
        """
        Compute rate of coherence change (drift)
        
        Args:
            packet_id: ID of packet to analyze
            window_size: Number of recent metrics to consider
            
        Returns:
            float: Coherence drift rate (per second)
        """
        if packet_id not in self.entropy_history:
            return 0.0
        
        history = self.entropy_history[packet_id][-window_size:]
        if len(history) < 2:
            return 0.0
        
        # Compute rate of change
        first = history[0]
        last = history[-1]
        time_diff = last.timestamp - first.timestamp
        
        if time_diff <= 0:
            return 0.0
        
        coherence_diff = last.spectral_entropy - first.spectral_entropy
        drift_rate = abs(coherence_diff) / time_diff
        return float(drift_rate)
    
    def compute_entropy_metrics(self, packet: QuantumPacket, 
                              attention_spectrum: Optional[List[float]] = None) -> EntropyMetrics:
        """
        Compute comprehensive entropy metrics for a Quantum Packet
        
        Args:
            packet: QuantumPacket to analyze
            attention_spectrum: Optional attention spectrum (from metadata if not provided)
            
        Returns:
            EntropyMetrics: Computed metrics
        """
        # Get attention spectrum from packet metadata if not provided
        if attention_spectrum is None and packet.metadata:
            attention_spectrum = packet.metadata.get("attention_spectrum", [])
        
        # Compute all metrics
        spectral_entropy = self.compute_spectral_entropy(attention_spectrum or [])
        temporal_variance = self.compute_temporal_variance(packet.id or "")
        dimensional_dispersion = self.compute_dimensional_dispersion(packet.omega_vector)
        coherence_drift = self.compute_coherence_drift(packet.id or "")
        
        metrics = EntropyMetrics(
            spectral_entropy=spectral_entropy,
            temporal_variance=temporal_variance,
            dimensional_dispersion=dimensional_dispersion,
            coherence_drift=coherence_drift,
            timestamp=time.time()
        )
        
        # Store in history
        if packet.id is not None:
            if packet.id not in self.entropy_history:
                self.entropy_history[packet.id] = []
            self.entropy_history[packet.id].append(metrics)
            
            # Keep only recent history (last 50 metrics)
            if len(self.entropy_history[packet.id]) > 50:
                self.entropy_history[packet.id] = self.entropy_history[packet.id][-50:]
        
        logger.debug(f"Entropy metrics computed for packet {packet.id}: {metrics}")
        return metrics
    
    def detect_high_entropy(self, packet_id: str, metrics: EntropyMetrics) -> bool:
        """
        Detect if a packet has high entropy requiring intervention
        
        Args:
            packet_id: ID of packet to check
            metrics: Current entropy metrics
            
        Returns:
            bool: True if high entropy detected
        """
        # Compute composite entropy score
        composite_entropy = (
            metrics.spectral_entropy +
            metrics.temporal_variance * 0.5 +
            metrics.dimensional_dispersion * 0.3 +
            metrics.coherence_drift * 2.0
        )
        
        # Check against thresholds
        is_high_entropy = (
            metrics.spectral_entropy > self.safety_thresholds["spectral_entropy_max"] or
            metrics.coherence_drift > self.safety_thresholds["coherence_drift_max"] or
            composite_entropy > self.safety_thresholds["rebalancing_threshold"]
        )
        
        if is_high_entropy and packet_id is not None:
            self.high_entropy_packets[packet_id] = composite_entropy
            logger.warning(f"High entropy detected in packet {packet_id}: {composite_entropy:.4f}")
        
        return is_high_entropy
    
    def trigger_rebalancing_protocol(self, packet_id: str) -> bool:
        """
        Trigger Rebalancing Protocol for high-entropy packets
        Uses m_t(L) modulator to synchronize Ω vectors with neighboring high-Ψ packets
        
        Args:
            packet_id: ID of packet requiring rebalancing
            
        Returns:
            bool: True if rebalancing successful
        """
        logger.info(f"🔄 Triggering rebalancing protocol for packet {packet_id}")
        
        # Retrieve the problematic packet
        packet = self.ufm.retrieve_packet(packet_id)
        if not packet:
            logger.error(f"Packet {packet_id} not found for rebalancing")
            return False
        
        # Find neighboring high-Ψ packets for synchronization
        # In a full implementation, this would use graph traversal
        # For now, we'll get high-coherence packets from the same layer
        scale_level = ScaleLevel(packet.scale_level)
        layer = self.ufm.layers.get(scale_level)
        
        if not layer:
            logger.error(f"Layer {scale_level} not found")
            return False
        
        # Get high-coherence neighbors
        neighbors = layer.get_high_coherence_packets(threshold=0.8)
        
        # Exclude the problematic packet itself
        neighbors = [p for p in neighbors if p.id != packet_id][:5]  # Limit to 5 neighbors
        
        if not neighbors:
            logger.warning(f"No high-coherence neighbors found for packet {packet_id}")
            return False
        
        # Synchronize Ω vectors using weighted average
        # Weight by coherence scores
        total_weight = sum(p.psi_score for p in neighbors)
        if total_weight <= 0:
            logger.warning(f"Invalid weights for neighbors of packet {packet_id}")
            return False
        
        # Compute weighted average of neighbor Ω vectors
        weighted_sum = np.zeros(len(packet.omega_vector)) if packet.omega_vector else np.array([])
        
        for neighbor in neighbors:
            if neighbor.omega_vector:
                weight = neighbor.psi_score / total_weight
                neighbor_vector = np.array(neighbor.omega_vector)
                if len(weighted_sum) == 0:
                    weighted_sum = np.zeros_like(neighbor_vector)
                # Ensure vectors have same length
                if len(weighted_sum) == len(neighbor_vector):
                    weighted_sum += neighbor_vector * weight
        
        # Update packet's Ω vector
        if len(weighted_sum) > 0:
            packet.omega_vector = weighted_sum.tolist()
            logger.info(f"✅ Rebalanced packet {packet_id} with {len(neighbors)} neighbors")
            return True
        else:
            logger.error(f"Failed to rebalance packet {packet_id} - vector dimension mismatch")
            return False
    
    def transmute_packet(self, packet_id: str) -> bool:
        """
        Transmute a packet - remove data but integrate Ω signature into latent potential
        When τ(L) expires or entropy exceeds threshold, the packet is Transmuted
        
        Args:
            packet_id: ID of packet to transmute
            
        Returns:
            bool: True if transmutation successful
        """
        logger.info(f"⚡ Transmuting packet {packet_id}")
        
        # Retrieve the packet
        packet = self.ufm.retrieve_packet(packet_id)
        if not packet:
            logger.error(f"Packet {packet_id} not found for transmutation")
            return False
        
        # Check if entropy exceeds transmutation threshold
        if packet_id in self.high_entropy_packets:
            entropy_value = self.high_entropy_packets[packet_id]
            if entropy_value < self.safety_thresholds["transmutation_threshold"]:
                logger.info(f"Packet {packet_id} entropy {entropy_value:.4f} below transmutation threshold")
                return False
        
        # Integrate Ω signature into I_t(L) as latent potential
        # In a full implementation, this would update the integrated feedback vector
        omega_signature = packet.omega_vector
        psi_score = packet.psi_score
        timestamp = packet.timestamp
        
        logger.info(f"Integrated Ω signature from packet {packet_id} into latent potential (Ψ={psi_score:.4f})")
        
        # Remove packet data but preserve metadata about transmutation
        if packet.metadata is None:
            packet.metadata = {}
        packet.metadata["transmuted"] = True
        packet.metadata["transmuted_at"] = time.time()
        packet.metadata["original_psi"] = psi_score
        packet.data_payload = None  # Remove data payload
        
        logger.info(f"✅ Transmuted packet {packet_id} - data removed, Ω signature preserved")
        return True
    
    def monitor_all_packets(self) -> Dict[str, List[str]]:
        """
        Monitor all packets in UFM for entropy issues
        
        Returns:
            Dict mapping action types to lists of affected packet IDs
        """
        actions_taken = {
            "rebalanced": [],
            "transmuted": [],
            "monitored": []
        }
        
        # Iterate through all layers and packets
        for scale_level, layer in self.ufm.layers.items():
            for packet_id, packet in layer.packets.items():
                # Compute entropy metrics
                metrics = self.compute_entropy_metrics(packet)
                
                # Check for high entropy
                if self.detect_high_entropy(packet_id, metrics):
                    # Determine appropriate action based on entropy level
                    composite_entropy = (
                        metrics.spectral_entropy +
                        metrics.temporal_variance * 0.5 +
                        metrics.dimensional_dispersion * 0.3 +
                        metrics.coherence_drift * 2.0
                    )
                    
                    if composite_entropy > self.safety_thresholds["transmutation_threshold"]:
                        # High entropy - trigger transmutation
                        if self.transmute_packet(packet_id):
                            actions_taken["transmuted"].append(packet_id)
                    elif composite_entropy > self.safety_thresholds["rebalancing_threshold"]:
                        # Moderate entropy - trigger rebalancing
                        if self.trigger_rebalancing_protocol(packet_id):
                            actions_taken["rebalanced"].append(packet_id)
                
                actions_taken["monitored"].append(packet_id)
        
        # Clean up resolved high-entropy packets
        resolved_packets = []
        for packet_id in self.high_entropy_packets:
            packet = self.ufm.retrieve_packet(packet_id)
            if not packet or packet_id not in actions_taken["monitored"]:
                resolved_packets.append(packet_id)
        
        for packet_id in resolved_packets:
            del self.high_entropy_packets[packet_id]
        
        logger.info(f"Monitored {len(actions_taken['monitored'])} packets: "
                   f"{len(actions_taken['rebalanced'])} rebalanced, "
                   f"{len(actions_taken['transmuted'])} transmuted")
        
        return actions_taken

# Example usage and testing
if __name__ == "__main__":
    # Create UFM and Entropy Monitor
    ufm = UnifiedFieldMemory()
    entropy_monitor = EntropyMonitor(ufm)
    
    # Create test packet with high entropy characteristics
    packet = ufm.create_quantum_packet(
        omega_vector=[1.0, 0.5, 0.2],
        psi_score=0.85,
        scale_level="LΦ",
        data_payload="High entropy test data",
        attention_spectrum=[0.3, 0.4, 0.3]  # High dispersion spectrum
    )
    
    # Store packet
    if ufm.store_packet(packet):
        print(f"Stored packet {packet.id}")
        
        # Compute entropy metrics
        metrics = entropy_monitor.compute_entropy_metrics(packet)
        print(f"Entropy metrics: {metrics}")
        
        # Check for high entropy
        is_high = entropy_monitor.detect_high_entropy(packet.id or "", metrics)
        print(f"High entropy detected: {is_high}")
        
        # Monitor all packets
        actions = entropy_monitor.monitor_all_packets()
        print(f"Actions taken: {actions}")