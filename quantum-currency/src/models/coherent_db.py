#!/usr/bin/env python3
"""
Coherent Database (CDB) - Graph Database Implementation
Implements the graph database structure for Quantum Packets and Coherence Relations

This module provides:
1. Graph database structure (Nodes=QP, Edges=CR)
2. Wave propagation query algorithm
3. Coherence-based indexing and retrieval
"""

import numpy as np
from typing import List, Dict, Tuple, Any, Optional, Union
import logging
import time
import heapq
from dataclasses import dataclass
from collections import defaultdict
from .quantum_memory import QuantumPacket, UnifiedFieldMemory, ScaleLevel, CoherenceRelation

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class GraphNode:
    """Node in the Coherent Database graph"""
    packet_id: str
    omega_vector: List[float]
    psi_score: float
    scale_level: str
    timestamp: float
    metadata: Dict[str, Any]

@dataclass
class GraphEdge:
    """Edge in the Coherent Database graph"""
    source_id: str
    target_id: str
    weight: float  # Coherence Relation weight
    timestamp: float

class CoherentDatabase:
    """
    Coherent Database (CDB) - Graph Database Implementation
    Implements the graph structure where Nodes are Quantum Packets and Edges are Coherence Relations
    """
    
    def __init__(self, ufm: UnifiedFieldMemory, network_id: str = "quantum-currency-uhes-cdb-001"):
        self.network_id = network_id
        self.ufm = ufm  # Reference to Unified Field Memory
        self.nodes: Dict[str, GraphNode] = {}  # packet_id -> GraphNode
        self.edges: Dict[Tuple[str, str], GraphEdge] = {}  # (source, target) -> GraphEdge
        self.adjacency_list: Dict[str, List[str]] = defaultdict(list)  # source_id -> [target_ids]
        
        logger.info(f"📊 Coherent Database initialized for network: {network_id}")
    
    def sync_with_ufm(self) -> Tuple[int, int]:
        """
        Synchronize CDB with Unified Field Memory
        
        Returns:
            Tuple of (nodes_added, edges_added)
        """
        nodes_added = 0
        edges_added = 0
        
        # Sync all packets from UFM layers
        for scale_level, layer in self.ufm.layers.items():
            for packet_id, packet in layer.packets.items():
                # Add/update node
                if packet.id is not None:
                    node = GraphNode(
                        packet_id=packet.id,
                        omega_vector=packet.omega_vector,
                        psi_score=packet.psi_score,
                        scale_level=packet.scale_level,
                        timestamp=packet.timestamp,
                        metadata=packet.metadata or {}
                    )
                    self.nodes[packet.id] = node
                    nodes_added += 1
                    
                    # Add/update edges based on coherence relations in the layer
                    for (source_id, target_id), cr in layer.relations.items():
                        if source_id is not None and target_id is not None:
                            edge_key = (source_id, target_id)
                            edge = GraphEdge(
                                source_id=source_id,
                                target_id=target_id,
                                weight=cr.weight,
                                timestamp=time.time()
                            )
                            self.edges[edge_key] = edge
                            self.adjacency_list[source_id].append(target_id)
                            edges_added += 1
        
        logger.info(f"🔄 CDB synchronized: {nodes_added} nodes, {edges_added} edges")
        return nodes_added, edges_added
    
    def compute_coherence_relation_weight(self, source_omega: List[float], 
                                        target_omega: List[float],
                                        source_scale: str,
                                        target_scale: str) -> float:
        """
        Compute Coherence Relation (CR) weight
        CR(edge) = cos(Ω_source, Ω_target) × ScaleAlignment(L)
        
        Args:
            source_omega: Source packet Ω vector
            target_omega: Target packet Ω vector
            source_scale: Source scale level
            target_scale: Target scale level
            
        Returns:
            float: Computed CR weight
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
        
        # Compute Scale Alignment factor
        # This depends on the relationship between scale levels
        scale_alignment = self._compute_scale_alignment(source_scale, target_scale)
        
        weight = cos_similarity * scale_alignment
        return max(0.0, min(1.0, weight))  # Clamp to [0,1]
    
    def _compute_scale_alignment(self, source_scale: str, target_scale: str) -> float:
        """
        Compute Scale Alignment factor based on scale level relationship
        
        Args:
            source_scale: Source scale level
            target_scale: Target scale level
            
        Returns:
            float: Scale alignment factor
        """
        # Define scale hierarchy: Lμ < Lϕ < LΦ
        scale_order = {"Lμ": 0, "Lϕ": 1, "LΦ": 2}
        
        source_order = scale_order.get(source_scale, 1)
        target_order = scale_order.get(target_scale, 1)
        
        # Alignment is higher when scales are closer
        order_diff = abs(source_order - target_order)
        alignment = max(0.0, 1.0 - (order_diff * 0.3))  # 1.0 for same scale, 0.7 for adjacent, 0.4 for distant
        return alignment
    
    def add_quantum_packet(self, packet: QuantumPacket) -> bool:
        """
        Add a Quantum Packet as a node in the CDB
        
        Args:
            packet: QuantumPacket to add
            
        Returns:
            bool: True if added successfully
        """
        if packet.id is None:
            logger.error("Cannot add packet with None ID to CDB")
            return False
        
        # Create graph node
        node = GraphNode(
            packet_id=packet.id,
            omega_vector=packet.omega_vector,
            psi_score=packet.psi_score,
            scale_level=packet.scale_level,
            timestamp=packet.timestamp,
            metadata=packet.metadata or {}
        )
        
        self.nodes[packet.id] = node
        logger.debug(f"Added node {packet.id} to CDB")
        return True
    
    def add_coherence_relation(self, source_id: str, target_id: str, 
                             source_omega: List[float], target_omega: List[float],
                             source_scale: str, target_scale: str) -> bool:
        """
        Add a Coherence Relation as an edge in the CDB
        
        Args:
            source_id: Source packet ID
            target_id: Target packet ID
            source_omega: Source Ω vector
            target_omega: Target Ω vector
            source_scale: Source scale level
            target_scale: Target scale level
            
        Returns:
            bool: True if added successfully
        """
        # Compute CR weight
        weight = self.compute_coherence_relation_weight(
            source_omega, target_omega, source_scale, target_scale
        )
        
        # Create graph edge
        edge_key = (source_id, target_id)
        edge = GraphEdge(
            source_id=source_id,
            target_id=target_id,
            weight=weight,
            timestamp=time.time()
        )
        
        self.edges[edge_key] = edge
        self.adjacency_list[source_id].append(target_id)
        
        logger.debug(f"Added edge {source_id}->{target_id} with weight {weight:.4f}")
        return True
    
    def wave_propagation_query(self, start_omega: List[float], 
                             start_scale: str = "LΦ",
                             max_depth: int = 5,
                             beam_width: int = 10) -> List[Tuple[QuantumPacket, float]]:
        """
        Wave Propagation Query - Graph traversal beginning from current Ω_t(L) state
        Finds the most resonant (highest CR-weighted) clusters, not nearest neighbors
        
        Args:
            start_omega: Starting Ω vector
            start_scale: Starting scale level
            max_depth: Maximum traversal depth
            beam_width: Number of best paths to keep at each level (beam search)
            
        Returns:
            List of (QuantumPacket, accumulated_weight) tuples, sorted by weight
        """
        logger.info(f"🌊 Starting wave propagation query from scale {start_scale}")
        
        # Find the most similar starting node
        start_node = self._find_most_similar_node(start_omega, start_scale)
        if not start_node:
            logger.warning("No suitable starting node found for wave propagation")
            return []
        
        # Use beam search to find resonant paths
        # Each item in the beam: (node_id, accumulated_weight, path_length, path)
        beam = [(start_node.packet_id, 1.0, 0, [start_node.packet_id])]
        results = []
        
        for depth in range(max_depth):
            next_beam = []
            
            for node_id, acc_weight, path_length, path in beam:
                # Get neighbors
                neighbors = self.adjacency_list.get(node_id, [])
                
                # Score each neighbor
                for neighbor_id in neighbors:
                    if neighbor_id in path:  # Avoid cycles
                        continue
                    
                    edge_key = (node_id, neighbor_id)
                    edge = self.edges.get(edge_key)
                    if not edge:
                        continue
                    
                    # Accumulate weight
                    new_weight = acc_weight * edge.weight
                    new_path = path + [neighbor_id]
                    
                    # Add to next beam
                    next_beam.append((neighbor_id, new_weight, path_length + 1, new_path))
                    
                    # Also collect results
                    neighbor_node = self.nodes.get(neighbor_id)
                    if neighbor_node:
                        # Retrieve the actual QuantumPacket from UFM
                        packet = self.ufm.retrieve_packet(neighbor_id)
                        if packet:
                            results.append((packet, new_weight))
            
            # Sort and keep only top beam_width paths
            next_beam.sort(key=lambda x: x[1], reverse=True)
            beam = next_beam[:beam_width]
            
            logger.debug(f"Depth {depth}: {len(beam)} paths in beam, {len(results)} results collected")
        
        # Sort results by accumulated weight (descending)
        results.sort(key=lambda x: x[1], reverse=True)
        
        logger.info(f"Wave propagation completed: {len(results)} resonant packets found")
        return results
    
    def _find_most_similar_node(self, target_omega: List[float], target_scale: str) -> Optional[GraphNode]:
        """
        Find the most similar node to the target Ω vector and scale
        
        Args:
            target_omega: Target Ω vector
            target_scale: Target scale level
            
        Returns:
            GraphNode or None if no suitable node found
        """
        if not target_omega or not self.nodes:
            return None
        
        target_vec = np.array(target_omega)
        best_similarity = -1.0
        best_node = None
        
        for node in self.nodes.values():
            # Compute similarity based on Ω vectors and scale alignment
            if node.omega_vector:
                node_vec = np.array(node.omega_vector)
                if len(node_vec) == len(target_vec):
                    # Cosine similarity
                    dot_product = np.dot(target_vec, node_vec)
                    norm_target = np.linalg.norm(target_vec)
                    norm_node = np.linalg.norm(node_vec)
                    
                    if norm_target > 0 and norm_node > 0:
                        cos_similarity = dot_product / (norm_target * norm_node)
                        
                        # Scale alignment factor
                        scale_alignment = self._compute_scale_alignment(node.scale_level, target_scale)
                        
                        # Combined score
                        combined_score = cos_similarity * scale_alignment
                        
                        if combined_score > best_similarity:
                            best_similarity = combined_score
                            best_node = node
        
        return best_node
    
    def get_priority_index(self, packet_id: str) -> float:
        """
        Compute Retrieval Priority Function for a packet
        Priority(QP) = Ψ(QP) × (τ(L)/λ(L)) × Resonance(CR)
        
        Args:
            packet_id: ID of packet to score
            
        Returns:
            float: Priority score
        """
        node = self.nodes.get(packet_id)
        if not node:
            return 0.0
        
        # Ψ(QP) - Coherence score
        psi_score = node.psi_score
        
        # (τ(L)/λ(L)) - Time/scale factor (simplified)
        # In a full implementation, τ(L) would be the time delay and λ(L) the decay factor
        tau_lambda_ratio = 1.0  # Placeholder
        
        # Resonance(CR) - Average coherence relation weight
        outgoing_edges = [edge for (src, _), edge in self.edges.items() if src == packet_id]
        if outgoing_edges:
            avg_resonance = float(np.mean([edge.weight for edge in outgoing_edges]))
        else:
            avg_resonance = 0.5  # Default value
        
        priority = float(psi_score * tau_lambda_ratio * avg_resonance)
        return priority
    
    def get_high_priority_packets(self, limit: int = 20) -> List[Tuple[QuantumPacket, float]]:
        """
        Get packets with highest priority scores
        
        Args:
            limit: Maximum number of packets to return
            
        Returns:
            List of (QuantumPacket, priority_score) tuples
        """
        # Score all nodes
        scored_packets = []
        
        for packet_id, node in self.nodes.items():
            priority = self.get_priority_index(packet_id)
            packet = self.ufm.retrieve_packet(packet_id)
            if packet:
                scored_packets.append((packet, priority))
        
        # Sort by priority (descending) and limit results
        scored_packets.sort(key=lambda x: x[1], reverse=True)
        return scored_packets[:limit]
    
    def get_database_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the Coherent Database
        
        Returns:
            Dict with database statistics
        """
        stats = {
            "node_count": len(self.nodes),
            "edge_count": len(self.edges),
            "average_node_psi": np.mean([node.psi_score for node in self.nodes.values()]) if self.nodes else 0.0,
            "average_edge_weight": np.mean([edge.weight for edge in self.edges.values()]) if self.edges else 0.0,
            "scale_distribution": self._get_scale_distribution(),
            "coherence_distribution": self._get_coherence_distribution()
        }
        return stats
    
    def _get_scale_distribution(self) -> Dict[str, int]:
        """Get distribution of packets across scale levels"""
        distribution = {"Lμ": 0, "Lϕ": 0, "LΦ": 0}
        for node in self.nodes.values():
            if node.scale_level in distribution:
                distribution[node.scale_level] += 1
        return distribution
    
    def _get_coherence_distribution(self) -> Dict[str, int]:
        """Get distribution of packets by coherence score ranges"""
        distribution = {
            "0.0-0.2": 0,
            "0.2-0.4": 0,
            "0.4-0.6": 0,
            "0.6-0.8": 0,
            "0.8-1.0": 0
        }
        
        for node in self.nodes.values():
            psi = node.psi_score
            if psi < 0.2:
                distribution["0.0-0.2"] += 1
            elif psi < 0.4:
                distribution["0.2-0.4"] += 1
            elif psi < 0.6:
                distribution["0.4-0.6"] += 1
            elif psi < 0.8:
                distribution["0.6-0.8"] += 1
            else:
                distribution["0.8-1.0"] += 1
        
        return distribution

# Example usage and testing
if __name__ == "__main__":
    # Create UFM and CDB
    ufm = UnifiedFieldMemory()
    cdb = CoherentDatabase(ufm)
    
    # Create test packets
    packet1 = ufm.create_quantum_packet(
        omega_vector=[1.0, 0.5, 0.2],
        psi_score=0.85,
        scale_level="LΦ",
        data_payload="Test data 1"
    )
    
    packet2 = ufm.create_quantum_packet(
        omega_vector=[0.9, 0.6, 0.15],
        psi_score=0.92,
        scale_level="LΦ",
        data_payload="Test data 2"
    )
    
    packet3 = ufm.create_quantum_packet(
        omega_vector=[1.1, 0.4, 0.25],
        psi_score=0.78,
        scale_level="Lϕ",
        data_payload="Test data 3"
    )
    
    # Store packets
    ufm.store_packet(packet1)
    ufm.store_packet(packet2)
    ufm.store_packet(packet3)
    
    # Synchronize CDB with UFM
    nodes_added, edges_added = cdb.sync_with_ufm()
    print(f"Synchronized: {nodes_added} nodes, {edges_added} edges")
    
    # Add coherence relations
    cdb.add_coherence_relation(
        packet1.id or "", packet2.id or "",
        packet1.omega_vector, packet2.omega_vector,
        packet1.scale_level, packet2.scale_level
    )
    
    cdb.add_coherence_relation(
        packet2.id or "", packet3.id or "",
        packet2.omega_vector, packet3.omega_vector,
        packet2.scale_level, packet3.scale_level
    )
    
    # Test wave propagation query
    results = cdb.wave_propagation_query([1.0, 0.5, 0.2])
    print(f"Wave propagation found {len(results)} resonant packets")
    for packet, weight in results[:5]:  # Show top 5
        print(f"  Packet {packet.id}: weight={weight:.4f}, Ψ={packet.psi_score:.4f}")
    
    # Test priority indexing
    if packet1.id is not None:
        priority = cdb.get_priority_index(packet1.id)
        print(f"Priority score for packet1: {priority:.4f}")
    
    # Get high priority packets
    high_priority = cdb.get_high_priority_packets(5)
    print(f"Top {len(high_priority)} high priority packets:")
    for packet, score in high_priority:
        print(f"  Packet {packet.id}: priority={score:.4f}")
    
    # Get database statistics
    stats = cdb.get_database_statistics()
    print(f"Database statistics: {stats}")