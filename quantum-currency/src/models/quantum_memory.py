#!/usr/bin/env python3
"""
Quantum Memory System - Unified Field Memory (UFM)
Implements the ϕ-Lattice Store with Micro, Phase, and Macro layers

This module provides:
1. Quantum Packet (QP) data structure
2. Three harmonic memory layers (Lμ, Lϕ, LΦ)
3. Coherent Database (CDB) interface
4. Wave propagation queries
"""

import numpy as np
from typing import List, Dict, Tuple, Any, Optional, Union
import logging
import time
import json
from dataclasses import dataclass, asdict
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ScaleLevel(Enum):
    """Scale levels in the ϕ-Lattice Store"""
    MICRO = "Lμ"      # Micro layer - Short-term memory
    PHASE = "Lϕ"      # Phase layer - Working memory
    MACRO = "LΦ"      # Macro layer - Long-term memory

@dataclass
class QuantumPacket:
    """
    Quantum Packet - Fundamental unit of data/memory in UFM
    Contains compressed state with coherence signature
    """
    # Core Ω-state components
    omega_vector: List[float]          # Ω_t(L) state vector
    psi_score: float                   # Coherence score Ψ
    timestamp: float                   # Creation timestamp
    scale_level: str                   # Scale level (Lμ, Lϕ, LΦ)
    
    # Data payload and compression
    data_payload: Optional[str] = None # Compressed data payload
    compression_ratio: float = 1.0     # γ compression ratio (∝ Ψ)
    
    # Metadata
    id: Optional[str] = None           # Unique identifier
    metadata: Optional[Dict[str, Any]] = None  # Additional metadata
    
    def __post_init__(self):
        """Initialize ID if not provided"""
        if self.id is None:
            self.id = f"qp_{int(self.timestamp * 1000000)}_{hash(str(self.omega_vector)) % 10000}"
        
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'QuantumPacket':
        """Create QuantumPacket from dictionary"""
        return cls(**data)

class CoherenceRelation:
    """
    Coherence Relation (CR) between Quantum Packets
    Weighted edge in the Coherent Database graph
    """
    
    def __init__(self, source_id: str, target_id: str, weight: float = 0.0):
        self.source_id = source_id
        self.target_id = target_id
        self.weight = weight  # CR weight based on cosine similarity
    
    def compute_weight(self, source_omega: List[float], target_omega: List[float]) -> float:
        """
        Compute CR weight as cosine similarity between Ω vectors
        Weight = cos(Ω_source, Ω_target) × Scale Alignment(L)
        
        Args:
            source_omega: Source packet Ω vector
            target_omega: Target packet Ω vector
            
        Returns:
            float: Computed weight
        """
        if not source_omega or not target_omega:
            return 0.0
        
        # Convert to numpy arrays
        vec1 = np.array(source_omega)
        vec2 = np.array(target_omega)
        
        # Compute cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        cos_similarity = dot_product / (norm1 * norm2)
        
        # Apply scale alignment factor (simplified)
        scale_alignment = 1.0  # Would depend on scale levels in practice
        
        weight = cos_similarity * scale_alignment
        self.weight = max(0.0, min(1.0, weight))  # Clamp to [0,1]
        return self.weight

class QuantumMemoryLayer:
    """
    Quantum Memory Layer - Implements one of the ϕ-Lattice layers
    """
    
    def __init__(self, scale_level: ScaleLevel):
        self.scale_level = scale_level
        self.packets: Dict[str, QuantumPacket] = {}  # Packet ID -> Packet
        self.relations: Dict[Tuple[str, str], CoherenceRelation] = {}  # (source, target) -> CR
        logger.info(f"-initialized for scale {scale_level.value}")

    def store_packet(self, packet: QuantumPacket) -> bool:
        """
        Store a Quantum Packet in this layer
        
        Args:
            packet: QuantumPacket to store
            
        Returns:
            bool: True if stored successfully
        """
        if packet.scale_level != self.scale_level.value:
            logger.warning(f"Packet scale {packet.scale_level} doesn't match layer scale {self.scale_level.value}")
            return False
        
        # Fix: Check that packet.id is not None before using as dictionary key
        if packet.id is not None:
            self.packets[packet.id] = packet
            logger.debug(f"Stored packet {packet.id} in {self.scale_level.value}")
            return True
        else:
            logger.error("Cannot store packet with None ID")
            return False
    
    def retrieve_packet(self, packet_id: str) -> Optional[QuantumPacket]:
        """
        Retrieve a Quantum Packet by ID
        
        Args:
            packet_id: ID of packet to retrieve
            
        Returns:
            QuantumPacket or None if not found
        """
        return self.packets.get(packet_id)
    
    def update_coherence_relations(self) -> int:
        """
        Update all coherence relations between packets in this layer
        
        Returns:
            int: Number of relations updated
        """
        count = 0
        packet_list = list(self.packets.values())
        
        for i, source_packet in enumerate(packet_list):
            # Fix: Check that source_packet.id is not None
            if source_packet.id is None:
                continue
                
            for j, target_packet in enumerate(packet_list):
                # Fix: Check that target_packet.id is not None
                if i != j and target_packet.id is not None:
                    # Create or update coherence relation
                    relation_key = (source_packet.id, target_packet.id)
                    if relation_key not in self.relations:
                        self.relations[relation_key] = CoherenceRelation(source_packet.id, target_packet.id)
                    
                    # Compute weight based on Ω vectors
                    weight = self.relations[relation_key].compute_weight(
                        source_packet.omega_vector, 
                        target_packet.omega_vector
                    )
                    count += 1
        
        logger.debug(f"Updated {count} coherence relations in {self.scale_level.value}")
        return count
    
    def get_high_coherence_packets(self, threshold: float = 0.8) -> List[QuantumPacket]:
        """
        Get packets with high coherence scores
        
        Args:
            threshold: Minimum Ψ score threshold
            
        Returns:
            List of high-coherence packets
        """
        return [p for p in self.packets.values() if p.psi_score >= threshold]

class UnifiedFieldMemory:
    """
    Unified Field Memory (UFM) - ϕ-Lattice Store
    Manages all three harmonic memory layers
    """
    
    def __init__(self, network_id: str = "quantum-currency-uhes-ufm-001"):
        self.network_id = network_id
        self.layers: Dict[ScaleLevel, QuantumMemoryLayer] = {
            ScaleLevel.MICRO: QuantumMemoryLayer(ScaleLevel.MICRO),
            ScaleLevel.PHASE: QuantumMemoryLayer(ScaleLevel.PHASE),
            ScaleLevel.MACRO: QuantumMemoryLayer(ScaleLevel.MACRO)
        }
        
        logger.info(f"🧠 Unified Field Memory initialized for network: {network_id}")
    
    def store_packet(self, packet: QuantumPacket) -> bool:
        """
        Store a Quantum Packet in the appropriate layer
        
        Args:
            packet: QuantumPacket to store
            
        Returns:
            bool: True if stored successfully
        """
        scale_level = ScaleLevel(packet.scale_level)
        if scale_level in self.layers:
            return self.layers[scale_level].store_packet(packet)
        else:
            logger.error(f"Invalid scale level: {packet.scale_level}")
            return False
    
    def retrieve_packet(self, packet_id: str) -> Optional[QuantumPacket]:
        """
        Retrieve a Quantum Packet by ID from any layer
        
        Args:
            packet_id: ID of packet to retrieve
            
        Returns:
            QuantumPacket or None if not found
        """
        for layer in self.layers.values():
            packet = layer.retrieve_packet(packet_id)
            if packet:
                return packet
        return None
    
    def create_quantum_packet(self, 
                            omega_vector: List[float],
                            psi_score: float,
                            scale_level: str,
                            data_payload: Optional[str] = None,
                            **metadata) -> QuantumPacket:
        """
        Create a new Quantum Packet with fractal compression
        
        Args:
            omega_vector: Ω state vector
            psi_score: Coherence score Ψ
            scale_level: Scale level (Lμ, Lϕ, LΦ)
            data_payload: Data to compress
            **metadata: Additional metadata
            
        Returns:
            QuantumPacket: Created packet
        """
        # Compute compression ratio γ ∝ Ψ
        compression_ratio = 1.0 + (psi_score * 2.0)  # γ = 1 + 2*Ψ
        
        packet = QuantumPacket(
            omega_vector=omega_vector,
            psi_score=psi_score,
            timestamp=time.time(),
            scale_level=scale_level,
            data_payload=data_payload,
            compression_ratio=compression_ratio,
            metadata=metadata
        )
        
        logger.debug(f"Created Quantum Packet with γ={compression_ratio:.2f} at Ψ={psi_score:.4f}")
        return packet
    
    def update_all_coherence_relations(self) -> Dict[ScaleLevel, int]:
        """
        Update coherence relations in all layers
        
        Returns:
            Dict mapping ScaleLevel to number of relations updated
        """
        results = {}
        for scale_level, layer in self.layers.items():
            count = layer.update_coherence_relations()
            results[scale_level] = count
        return results
    
    def wave_propagation_query(self, 
                             start_omega: List[float], 
                             max_depth: int = 5,
                             threshold: float = 0.7) -> List[QuantumPacket]:
        """
        Wave Propagation Query - Resonance-based recall
        Traverses graph from current Ω_t(L) state, finding highest CR-weighted clusters
        
        Args:
            start_omega: Starting Ω vector
            max_depth: Maximum traversal depth
            threshold: Minimum CR weight threshold
            
        Returns:
            List of most resonant packets
        """
        # For now, we'll search all layers for high-coherence packets
        # In a full implementation, this would do actual graph traversal
        resonant_packets = []
        
        for layer in self.layers.values():
            high_coherence_packets = layer.get_high_coherence_packets(threshold)
            # Sort by coherence score
            high_coherence_packets.sort(key=lambda p: p.psi_score, reverse=True)
            resonant_packets.extend(high_coherence_packets[:10])  # Top 10 from each layer
        
        # Sort all packets by coherence score
        resonant_packets.sort(key=lambda p: p.psi_score, reverse=True)
        return resonant_packets[:20]  # Return top 20 overall
    
    def get_layer_statistics(self) -> Dict[str, Dict[str, Any]]:
        """
        Get statistics for all memory layers
        
        Returns:
            Dict with statistics for each layer
        """
        stats = {}
        for scale_level, layer in self.layers.items():
            stats[scale_level.value] = {
                "packet_count": len(layer.packets),
                "relation_count": len(layer.relations),
                "avg_psi_score": np.mean([p.psi_score for p in layer.packets.values()]) if layer.packets else 0.0,
                "high_coherence_count": len(layer.get_high_coherence_packets())
            }
        return stats

# Example usage and testing
if __name__ == "__main__":
    # Create UFM instance
    ufm = UnifiedFieldMemory()
    
    # Create some test packets
    packet1 = ufm.create_quantum_packet(
        omega_vector=[1.0, 0.5, 0.2],
        psi_score=0.85,
        scale_level="LΦ",
        data_payload="Test data 1",
        source="test_node_1"
    )
    
    packet2 = ufm.create_quantum_packet(
        omega_vector=[0.9, 0.6, 0.15],
        psi_score=0.92,
        scale_level="LΦ",
        data_payload="Test data 2",
        source="test_node_2"
    )
    
    packet3 = ufm.create_quantum_packet(
        omega_vector=[1.1, 0.4, 0.25],
        psi_score=0.78,
        scale_level="Lϕ",
        data_payload="Test data 3",
        source="test_node_3"
    )
    
    # Store packets (fix: check return values)
    success1 = ufm.store_packet(packet1)
    success2 = ufm.store_packet(packet2)
    success3 = ufm.store_packet(packet3)
    
    print(f"Storage results: {success1}, {success2}, {success3}")
    
    # Update coherence relations
    relation_counts = ufm.update_all_coherence_relations()
    print(f"Updated relations: {relation_counts}")
    
    # Test retrieval (fix: check that packet1.id is not None)
    if packet1.id is not None:
        retrieved = ufm.retrieve_packet(packet1.id)
        print(f"Retrieved packet: {retrieved.id if retrieved else 'None'}")
    
    # Test wave propagation query
    resonant_packets = ufm.wave_propagation_query([1.0, 0.5, 0.2])
    print(f"Found {len(resonant_packets)} resonant packets")
    
    # Get statistics
    stats = ufm.get_layer_statistics()
    print(f"Layer statistics: {stats}")