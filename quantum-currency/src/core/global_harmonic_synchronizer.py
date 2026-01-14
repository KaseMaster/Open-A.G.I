#!/usr/bin/env python3
"""
Global Harmonic Synchronizer Module
Implements the Global Harmonic Synchronizer for aggregating coherence metrics 
and redistributing stabilizing feedback across connected nodes
"""

import sys
import os
import json
import time
import hashlib
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field

# Add parent directory to path to import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# Import our existing modules
from core.harmonic_validation import compute_spectrum, compute_coherence_score, HarmonicSnapshot
from models.external_network import ExternalNetworkConnector, ResonanceData

@dataclass
class GlobalCoherenceMetrics:
    """Represents global coherence metrics"""
    timestamp: float
    global_coherence: float
    regional_entropy_hotspots: List[Dict[str, Any]]
    systemic_equilibrium: float
    connected_nodes: int
    coherence_history: List[float] = field(default_factory=list)

@dataclass
class FieldAdjustment:
    """Represents a field adjustment for stabilizing feedback"""
    node_id: str
    adjustment_type: str  # coherence, entropy, flow
    adjustment_value: float
    timestamp: float
    applied: bool = False

class GlobalHarmonicSynchronizer:
    """
    Implements the Global Harmonic Synchronizer module
    Aggregates coherence metrics from all connected nodes and redistributes stabilizing feedback
    """
    
    def __init__(self, network_id: str = "global-harmonic-network-001"):
        self.network_id = network_id
        self.external_network = ExternalNetworkConnector(network_id)
        self.global_metrics: List[GlobalCoherenceMetrics] = []
        self.field_adjustments: List[FieldAdjustment] = []
        self.learning_patterns: List[Dict[str, Any]] = []  # Store learning patterns
        self.synchronizer_config = {
            "aggregation_interval": 60,  # 1 minute
            "feedback_distribution_frequency": 300,  # 5 minutes
            "coherence_threshold": 0.85,  # Minimum coherence for stability
            "entropy_threshold": 0.3,  # Maximum entropy for stability
            "max_adjustments": 1000
        }
        self.last_aggregation = 0.0
        self.last_feedback_distribution = 0.0
        self.last_learning_update = 0.0
    
    def aggregate_coherence_metrics(self) -> Optional[GlobalCoherenceMetrics]:
        """
        Aggregate coherence metrics from all connected nodes
        
        Returns:
            GlobalCoherenceMetrics if successful, None otherwise
        """
        current_time = time.time()
        
        # Check if it's time to aggregate
        if current_time - self.last_aggregation < self.synchronizer_config["aggregation_interval"]:
            # Return last metrics if available
            if self.global_metrics:
                return self.global_metrics[-1]
            return None
        
        self.last_aggregation = current_time
        
        # Get network topology
        topology = self.external_network.get_network_topology()
        connected_nodes = topology.get("active_systems", 0)
        
        # Get real-time feedback from all systems
        feedback_data = self.external_network.get_real_time_feedback()
        
        if not feedback_data:
            return None
        
        # Extract metrics from feedback
        coherence_values = []
        entropy_values = []
        flow_values = []
        entropy_hotspots = []
        
        for feedback in feedback_data:
            if "coherence" in feedback:
                coherence_values.append(feedback["coherence"])
            if "entropy" in feedback:
                entropy_values.append(feedback["entropy"])
                # Identify entropy hotspots
                if feedback["entropy"] > self.synchronizer_config["entropy_threshold"]:
                    entropy_hotspots.append({
                        "system_id": feedback.get("system_id", "unknown"),
                        "entropy": feedback["entropy"],
                        "timestamp": feedback["timestamp"]
                    })
            if "flow" in feedback:
                flow_values.append(feedback["flow"])
        
        # Calculate global metrics
        global_coherence = float(np.mean(coherence_values)) if coherence_values else 0.0
        systemic_equilibrium = float(np.mean(flow_values)) if flow_values else 0.0
        
        # Create global metrics object
        metrics = GlobalCoherenceMetrics(
            timestamp=current_time,
            global_coherence=global_coherence,
            regional_entropy_hotspots=entropy_hotspots,
            systemic_equilibrium=systemic_equilibrium,
            connected_nodes=connected_nodes,
            coherence_history=[global_coherence]
        )
        
        # Add to history
        self.global_metrics.append(metrics)
        
        # Keep only recent history (last 100 entries)
        if len(self.global_metrics) > 100:
            self.global_metrics = self.global_metrics[-100:]
        
        return metrics
    
    def identify_disharmony_areas(self, metrics: GlobalCoherenceMetrics) -> List[Dict[str, Any]]:
        """
        Identify areas of disharmony that need corrective field adjustments
        
        Args:
            metrics: Global coherence metrics
            
        Returns:
            List of disharmony areas requiring adjustments
        """
        disharmony_areas = []
        
        # Check global coherence
        if metrics.global_coherence < self.synchronizer_config["coherence_threshold"]:
            disharmony_areas.append({
                "type": "low_global_coherence",
                "severity": "high" if metrics.global_coherence < 0.7 else "medium",
                "current_value": metrics.global_coherence,
                "threshold": self.synchronizer_config["coherence_threshold"],
                "recommended_adjustment": self.synchronizer_config["coherence_threshold"] - metrics.global_coherence
            })
        
        # Check entropy hotspots
        if metrics.regional_entropy_hotspots:
            disharmony_areas.append({
                "type": "entropy_hotspots",
                "severity": "high" if len(metrics.regional_entropy_hotspots) > 5 else "medium",
                "count": len(metrics.regional_entropy_hotspots),
                "hotspots": metrics.regional_entropy_hotspots
            })
        
        # Check systemic equilibrium
        if abs(metrics.systemic_equilibrium) > 0.5:  # High flow imbalance
            disharmony_areas.append({
                "type": "flow_imbalance",
                "severity": "high",
                "current_value": metrics.systemic_equilibrium,
                "recommended_adjustment": -metrics.systemic_equilibrium * 0.1  # 10% correction
            })
        
        return disharmony_areas
    
    def propose_corrective_adjustments(self, disharmony_areas: List[Dict[str, Any]]) -> List[FieldAdjustment]:
        """
        Propose corrective field adjustments for identified disharmony areas
        
        Args:
            disharmony_areas: List of identified disharmony areas
            
        Returns:
            List of proposed field adjustments
        """
        adjustments = []
        current_time = time.time()
        
        for area in disharmony_areas:
            area_type = area["type"]
            
            if area_type == "low_global_coherence":
                # Propose coherence enhancement adjustments
                adjustment_value = area["recommended_adjustment"]
                # Apply to all connected systems
                topology = self.external_network.get_network_topology()
                active_systems = topology.get("active_systems", 0)
                
                for i in range(active_systems):
                    system_id = f"system-{i}"
                    adjustment = FieldAdjustment(
                        node_id=system_id,
                        adjustment_type="coherence",
                        adjustment_value=adjustment_value / max(active_systems, 1),
                        timestamp=current_time
                    )
                    adjustments.append(adjustment)
            
            elif area_type == "entropy_hotspots":
                # Propose entropy reduction adjustments for hotspots
                hotspots = area.get("hotspots", [])
                for hotspot in hotspots:
                    system_id = hotspot.get("system_id", "unknown")
                    entropy_value = hotspot.get("entropy", 0.0)
                    adjustment_value = -(entropy_value - self.synchronizer_config["entropy_threshold"]) * 0.5
                    
                    adjustment = FieldAdjustment(
                        node_id=system_id,
                        adjustment_type="entropy",
                        adjustment_value=adjustment_value,
                        timestamp=current_time
                    )
                    adjustments.append(adjustment)
            
            elif area_type == "flow_imbalance":
                # Propose flow balancing adjustments
                adjustment_value = area["recommended_adjustment"]
                # Apply to all connected systems
                topology = self.external_network.get_network_topology()
                active_systems = topology.get("active_systems", 0)
                
                for i in range(active_systems):
                    system_id = f"system-{i}"
                    adjustment = FieldAdjustment(
                        node_id=system_id,
                        adjustment_type="flow",
                        adjustment_value=adjustment_value / max(active_systems, 1),
                        timestamp=current_time
                    )
                    adjustments.append(adjustment)
        
        # Limit adjustments to max_adjustments
        if len(adjustments) > self.synchronizer_config["max_adjustments"]:
            adjustments = adjustments[:self.synchronizer_config["max_adjustments"]]
        
        return adjustments
    
    def distribute_stabilizing_feedback(self) -> Dict[str, Any]:
        """
        Distribute stabilizing feedback to connected nodes
        
        Returns:
            Dictionary with distribution results
        """
        current_time = time.time()
        
        # Check if it's time to distribute feedback
        if current_time - self.last_feedback_distribution < self.synchronizer_config["feedback_distribution_frequency"]:
            return {"status": "skipped", "message": "Feedback distribution interval not reached"}
        
        self.last_feedback_distribution = current_time
        
        # Aggregate current metrics
        metrics = self.aggregate_coherence_metrics()
        if not metrics:
            return {"status": "error", "message": "Failed to aggregate coherence metrics"}
        
        # Identify disharmony areas
        disharmony_areas = self.identify_disharmony_areas(metrics)
        
        # Apply learned patterns to improve adjustments
        adjustments = self.apply_learned_patterns(disharmony_areas)
        
        # Store adjustments
        self.field_adjustments.extend(adjustments)
        
        # Keep only recent adjustments (last max_adjustments)
        if len(self.field_adjustments) > self.synchronizer_config["max_adjustments"]:
            self.field_adjustments = self.field_adjustments[-self.synchronizer_config["max_adjustments"]:]
        
        # Learn from adjustments
        learning_result = self.learn_from_adjustments()
        
        # Identify disharmony patterns
        patterns = self.identify_disharmony_patterns()
        
        # In a real implementation, this would send adjustments to connected nodes
        # For demo purposes, we'll just simulate the distribution
        distributed_count = len(adjustments)
        
        return {
            "status": "success",
            "message": f"Distributed stabilizing feedback to {distributed_count} nodes",
            "adjustments_count": distributed_count,
            "disharmony_areas_identified": len(disharmony_areas),
            "patterns_identified": len(patterns),
            "learning_result": learning_result,
            "global_coherence": metrics.global_coherence,
            "timestamp": current_time
        }
    
    def get_global_coherence_map(self) -> Dict[str, Any]:
        """
        Get global coherence map for visualization
        
        Returns:
            Dictionary with global coherence map data
        """
        # Aggregate current metrics
        metrics = self.aggregate_coherence_metrics()
        if not metrics:
            return {"status": "error", "message": "Failed to aggregate coherence metrics"}
        
        # Get network topology
        topology = self.external_network.get_network_topology()
        
        # Create coherence map data
        coherence_map = {
            "status": "success",
            "network_id": self.network_id,
            "timestamp": metrics.timestamp,
            "global_coherence": metrics.global_coherence,
            "connected_nodes": metrics.connected_nodes,
            "regional_entropy_hotspots": metrics.regional_entropy_hotspots,
            "systemic_equilibrium": metrics.systemic_equilibrium,
            "topology": topology
        }
        
        return coherence_map
    
    def get_systemic_equilibrium_analytics(self) -> Dict[str, Any]:
        """
        Get systemic equilibrium analytics
        
        Returns:
            Dictionary with systemic equilibrium analytics
        """
        # Aggregate current metrics
        metrics = self.aggregate_coherence_metrics()
        if not metrics:
            return {"status": "error", "message": "Failed to aggregate coherence metrics"}
        
        # Calculate trends from history
        coherence_history = [m.global_coherence for m in self.global_metrics[-10:]] if self.global_metrics else []
        equilibrium_history = [m.systemic_equilibrium for m in self.global_metrics[-10:]] if self.global_metrics else []
        
        coherence_trend = 0.0
        equilibrium_trend = 0.0
        
        if len(coherence_history) > 1:
            coherence_trend = np.polyfit(range(len(coherence_history)), coherence_history, 1)[0]
        
        if len(equilibrium_history) > 1:
            equilibrium_trend = np.polyfit(range(len(equilibrium_history)), equilibrium_history, 1)[0]
        
        return {
            "status": "success",
            "global_coherence": metrics.global_coherence,
            "systemic_equilibrium": metrics.systemic_equilibrium,
            "coherence_trend": coherence_trend,
            "equilibrium_trend": equilibrium_trend,
            "entropy_hotspots_count": len(metrics.regional_entropy_hotspots),
            "connected_nodes": metrics.connected_nodes,
            "timestamp": metrics.timestamp
        }
    
    def calculate_coherence_amplification_factor(self) -> Dict[str, Any]:
        """
        Calculate Coherence Amplification Factor (CAF)
        
        Returns:
            Dictionary with CAF calculation results
        """
        # Aggregate current metrics
        metrics = self.aggregate_coherence_metrics()
        if not metrics:
            return {"status": "error", "message": "Failed to aggregate coherence metrics"}
        
        # For CAF calculation, we need:
        # CAF = (H_external - H_baseline) / H_internal
        
        # Simulate baseline coherence (system without external influence)
        baseline_coherence = 0.75
        
        # Current global coherence (with external influence)
        external_coherence = metrics.global_coherence
        
        # Internal coherence would be lower than external
        internal_coherence = external_coherence * 0.9  # Simulated internal coherence
        
        # Calculate CAF
        if internal_coherence > 0:
            caf = (external_coherence - baseline_coherence) / internal_coherence
        else:
            caf = 0.0
        
        return {
            "status": "success",
            "caf": caf,
            "external_coherence": external_coherence,
            "internal_coherence": internal_coherence,
            "baseline_coherence": baseline_coherence,
            "message": "CAF >= 1.03" if caf >= 1.03 else "CAF < 1.03",
            "timestamp": metrics.timestamp
        }
    
    def identify_disharmony_patterns(self) -> List[Dict[str, Any]]:
        """
        Identify patterns in disharmony areas using adaptive learning algorithms
        
        Returns:
            List of identified disharmony patterns
        """
        patterns = []
        
        # Look for recurring disharmony types
        disharmony_counts = {}
        recent_metrics = self.global_metrics[-20:] if len(self.global_metrics) >= 20 else self.global_metrics
        
        for metrics in recent_metrics:
            disharmony_areas = self.identify_disharmony_areas(metrics)
            for area in disharmony_areas:
                area_type = area["type"]
                if area_type not in disharmony_counts:
                    disharmony_counts[area_type] = 0
                disharmony_counts[area_type] += 1
        
        # Identify frequently occurring disharmony types
        for area_type, count in disharmony_counts.items():
            if count >= 3:  # Appears in at least 3 of the last 20 metrics
                patterns.append({
                    "type": "frequent_disharmony",
                    "disharmony_type": area_type,
                    "frequency": count,
                    "percentage": (count / len(recent_metrics)) * 100 if recent_metrics else 0
                })
        
        # Look for correlation patterns between metrics
        if len(recent_metrics) >= 5:
            coherence_values = [m.global_coherence for m in recent_metrics]
            entropy_values = []
            flow_values = []
            
            for metrics in recent_metrics:
                # Get average entropy and flow from feedback data
                feedback_data = self.external_network.get_real_time_feedback(len(recent_metrics))
                if feedback_data:
                    entropies = [f.get("entropy", 0) for f in feedback_data if "entropy" in f]
                    flows = [f.get("flow", 0) for f in feedback_data if "flow" in f]
                    if entropies:
                        entropy_values.append(np.mean(entropies))
                    if flows:
                        flow_values.append(np.mean(flows))
            
            # Calculate correlations
            if len(coherence_values) > 1 and len(entropy_values) == len(coherence_values):
                coherence_entropy_corr = np.corrcoef(coherence_values, entropy_values)[0, 1]
                if abs(coherence_entropy_corr) > 0.7:  # Strong correlation
                    patterns.append({
                        "type": "correlation_pattern",
                        "variables": ["coherence", "entropy"],
                        "correlation": coherence_entropy_corr,
                        "strength": "strong" if abs(coherence_entropy_corr) > 0.9 else "moderate"
                    })
            
            if len(coherence_values) > 1 and len(flow_values) == len(coherence_values):
                coherence_flow_corr = np.corrcoef(coherence_values, flow_values)[0, 1]
                if abs(coherence_flow_corr) > 0.7:  # Strong correlation
                    patterns.append({
                        "type": "correlation_pattern",
                        "variables": ["coherence", "flow"],
                        "correlation": coherence_flow_corr,
                        "strength": "strong" if abs(coherence_flow_corr) > 0.9 else "moderate"
                    })
        
        return patterns
    
    def learn_from_adjustments(self) -> Dict[str, Any]:
        """
        Learn from past field adjustments to improve future corrections
        
        Returns:
            Dictionary with learning results
        """
        current_time = time.time()
        
        # Check if it's time to update learning (every 10 minutes)
        if current_time - self.last_learning_update < 600:
            return {"status": "skipped", "message": "Learning update interval not reached"}
        
        self.last_learning_update = current_time
        
        # Analyze recent adjustments and their effectiveness
        recent_adjustments = [adj for adj in self.field_adjustments if current_time - adj.timestamp < 3600]  # Last hour
        if not recent_adjustments:
            return {"status": "no_data", "message": "No recent adjustments to analyze"}
        
        # Group adjustments by type and node
        adjustment_groups = {}
        for adj in recent_adjustments:
            key = f"{adj.node_id}_{adj.adjustment_type}"
            if key not in adjustment_groups:
                adjustment_groups[key] = []
            adjustment_groups[key].append(adj)
        
        # Analyze effectiveness of each group
        learning_results = []
        for key, adjustments in adjustment_groups.items():
            if len(adjustments) >= 3:  # Need at least 3 adjustments to analyze
                # Calculate average adjustment value
                avg_adjustment = np.mean([adj.adjustment_value for adj in adjustments])
                
                # Store learning pattern
                pattern = {
                    "node_id": adjustments[0].node_id,
                    "adjustment_type": adjustments[0].adjustment_type,
                    "avg_adjustment": float(avg_adjustment),
                    "count": len(adjustments),
                    "timestamp": current_time
                }
                
                self.learning_patterns.append(pattern)
                learning_results.append(pattern)
        
        # Keep only recent learning patterns (last 100)
        if len(self.learning_patterns) > 100:
            self.learning_patterns = self.learning_patterns[-100:]
        
        return {
            "status": "success",
            "patterns_learned": len(learning_results),
            "total_patterns": len(self.learning_patterns),
            "timestamp": current_time
        }
    
    def apply_learned_patterns(self, disharmony_areas: List[Dict[str, Any]]) -> List[FieldAdjustment]:
        """
        Apply learned patterns to improve corrective adjustments
        
        Args:
            disharmony_areas: List of identified disharmony areas
            
        Returns:
            List of improved field adjustments
        """
        # First get standard adjustments
        adjustments = self.propose_corrective_adjustments(disharmony_areas)
        
        # Apply learned patterns to improve adjustments
        for adjustment in adjustments:
            # Look for matching learning patterns
            matching_patterns = [p for p in self.learning_patterns 
                               if p["node_id"] == adjustment.node_id and 
                                  p["adjustment_type"] == adjustment.adjustment_type]
            
            if matching_patterns:
                # Use the most recent pattern
                latest_pattern = matching_patterns[-1]
                # Adjust the value based on learned patterns
                adjustment.adjustment_value = latest_pattern["avg_adjustment"] * 1.1  # Slightly increase based on learning
        
        return adjustments

def demo_global_harmonic_synchronizer():
    """Demonstrate Global Harmonic Synchronizer capabilities"""
    print("🌐 Global Harmonic Synchronizer Demo")
    print("=" * 40)
    
    # Create synchronizer instance
    synchronizer = GlobalHarmonicSynchronizer("demo-network")
    
    # Connect to some external systems
    print("\n🔗 Connecting to External Systems:")
    system1_id = synchronizer.external_network.connect_external_system(
        name="Ethereum Bridge",
        url="https://eth-bridge.example.com",
        api_key="eth-api-key-123"
    )
    
    system2_id = synchronizer.external_network.connect_external_system(
        name="Solana Gateway",
        url="https://sol-gateway.example.com",
        api_key="sol-api-key-456"
    )
    
    if system1_id and system2_id:
        print(f"   Connected to Ethereum Bridge: {system1_id[:16]}...")
        print(f"   Connected to Solana Gateway: {system2_id[:16]}...")
    else:
        print("   Failed to connect to external systems")
        return
    
    # Collect some feedback data
    print("\n📊 Collecting Feedback Data:")
    feedback1 = {
        "coherence": 0.87,
        "entropy": 0.15,
        "flow": 0.05
    }
    
    feedback2 = {
        "coherence": 0.92,
        "entropy": 0.08,
        "flow": -0.02
    }
    
    synchronizer.external_network.collect_real_time_feedback(system1_id, feedback1)
    synchronizer.external_network.collect_real_time_feedback(system2_id, feedback2)
    print("   Feedback data collected")
    
    # Aggregate coherence metrics
    print("\n📈 Aggregating Coherence Metrics:")
    metrics = synchronizer.aggregate_coherence_metrics()
    if metrics:
        print(f"   Global Coherence: {metrics.global_coherence:.3f}")
        print(f"   Connected Nodes: {metrics.connected_nodes}")
        print(f"   Systemic Equilibrium: {metrics.systemic_equilibrium:.3f}")
        print(f"   Entropy Hotspots: {len(metrics.regional_entropy_hotspots)}")
    else:
        print("   Failed to aggregate metrics")
    
    # Get global coherence map
    print("\n🗺️  Global Coherence Map:")
    coherence_map = synchronizer.get_global_coherence_map()
    if coherence_map["status"] == "success":
        print(f"   Network: {coherence_map['network_id']}")
        print(f"   Global Coherence: {coherence_map['global_coherence']:.3f}")
        print(f"   Connected Nodes: {coherence_map['connected_nodes']}")
    
    # Get systemic equilibrium analytics
    print("\n📊 Systemic Equilibrium Analytics:")
    analytics = synchronizer.get_systemic_equilibrium_analytics()
    if analytics["status"] == "success":
        print(f"   Global Coherence: {analytics['global_coherence']:.3f}")
        print(f"   Systemic Equilibrium: {analytics['systemic_equilibrium']:.3f}")
        print(f"   Entropy Hotspots: {analytics['entropy_hotspots_count']}")
        print(f"   Connected Nodes: {analytics['connected_nodes']}")
    
    # Calculate CAF
    print("\n🔬 Coherence Amplification Factor:")
    caf_result = synchronizer.calculate_coherence_amplification_factor()
    if caf_result["status"] == "success":
        print(f"   CAF: {caf_result['caf']:.3f}")
        print(f"   External Coherence: {caf_result['external_coherence']:.3f}")
        print(f"   Internal Coherence: {caf_result['internal_coherence']:.3f}")
        print(f"   Baseline Coherence: {caf_result['baseline_coherence']:.3f}")
        print(f"   Status: {caf_result['message']}")
    
    # Distribute stabilizing feedback
    print("\n🔄 Distributing Stabilizing Feedback:")
    feedback_result = synchronizer.distribute_stabilizing_feedback()
    if feedback_result["status"] == "success":
        print(f"   {feedback_result['message']}")
        print(f"   Adjustments: {feedback_result['adjustments_count']}")
        print(f"   Disharmony Areas: {feedback_result['disharmony_areas_identified']}")
    else:
        print(f"   {feedback_result['message']}")
    
    print("\n✅ Global Harmonic Synchronizer demo completed!")

if __name__ == "__main__":
    demo_global_harmonic_synchronizer()