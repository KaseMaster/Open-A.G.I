#!/usr/bin/env python3
"""
Multi-Node Simulator for Quantum Currency v0.2.0
Tests the CAL-RΦV Fusion at scale with multiple nodes

This module implements:
1. Multi-node network simulation
2. Coherence propagation across nodes
3. Stress testing under various network conditions
4. Performance benchmarks and optimization
"""

import sys
import os
import time
import threading
import json
import logging
from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass, asdict
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import the modules to be tested
from src.models.coherence_attunement_layer import CoherenceAttunementLayer, OmegaState
from src.core.harmonic_validation import make_snapshot, compute_coherence_score, recursive_validate, HarmonicSnapshot
from src.core.token_rules import apply_token_effects

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class NodeState:
    """Represents the state of a node in the simulation"""
    node_id: str
    cal: CoherenceAttunementLayer
    snapshot: Optional[HarmonicSnapshot]
    omega_state: Optional[OmegaState]
    coherence_history: List[float]
    token_balances: Dict[str, float]
    chr_score: float
    is_active: bool = True


@dataclass
class SimulationMetrics:
    """Represents metrics collected during simulation"""
    avg_network_coherence: float
    coherence_variance: float
    validation_success_rate: float
    avg_token_distribution: Dict[str, float]
    simulation_duration: float
    nodes_processed: int
    timestamp: float = 0.0

    def __post_init__(self):
        if self.timestamp == 0:
            self.timestamp = time.time()


class MultiNodeSimulator:
    """
    Multi-Node Simulator for Quantum Currency v0.2.0
    Tests the CAL-RΦV Fusion at scale with multiple nodes
    """

    def __init__(self, num_nodes: int = 10, network_id: str = "quantum-currency-sim-001"):
        self.num_nodes = num_nodes
        self.network_id = network_id
        self.nodes: List[NodeState] = []
        self.simulation_round = 0
        self.metrics_history: List[SimulationMetrics] = []
        
        # Configuration parameters
        self.config = {
            "validation_threshold": 0.7,
            "coherence_recovery_steps": 50,
            "max_concurrent_threads": 10,
            "simulation_rounds": 100
        }
        
        # Initialize nodes
        self._initialize_nodes()
        
        logger.info(f"Intialized Multi-Node Simulator with {num_nodes} nodes for network: {network_id}")

    def _initialize_nodes(self):
        """Initialize nodes with Coherence Attunement Layers"""
        for i in range(self.num_nodes):
            node_id = f"sim-node-{i:03d}"
            cal = CoherenceAttunementLayer(network_id=f"{self.network_id}-{node_id}")
            
            node_state = NodeState(
                node_id=node_id,
                cal=cal,
                snapshot=None,
                omega_state=None,
                coherence_history=[],
                token_balances={"FLX": 1000.0, "CHR": 500.0, "PSY": 200.0, "ATR": 300.0, "RES": 50.0},
                chr_score=0.7 + (i * 0.01),  # Vary CHR scores slightly
                is_active=True
            )
            
            self.nodes.append(node_state)
        
        logger.info(f"Initialized {len(self.nodes)} nodes")

    def _generate_test_signal(self, freq: float, phase: float, duration: float = 1.0, 
                             sample_rate: float = 1000, noise_level: float = 0.1) -> Tuple[List[float], List[float]]:
        """Generate a test sinusoidal signal with noise"""
        t = np.linspace(0, duration, int(duration * sample_rate))
        x = np.sin(2 * np.pi * freq * t + phase)
        
        # Add noise
        noise = np.random.normal(0, noise_level, len(x))
        x_noisy = x + noise
        
        return t.tolist(), x_noisy.tolist()

    def _compute_node_omega_state(self, node: NodeState, round_num: int) -> OmegaState:
        """Compute Ω-state for a node based on its current state"""
        # Generate varying parameters based on round and node
        base_freq = 50.0 + (round_num * 0.1) + (int(node.node_id.split('-')[-1]) * 0.05)
        phase = (round_num * 0.01) + (int(node.node_id.split('-')[-1]) * 0.1)
        
        # Generate time series data
        times, values = self._generate_test_signal(
            freq=base_freq, 
            phase=phase, 
            duration=0.5,
            noise_level=0.05 + (round_num * 0.001)
        )
        
        # Create snapshot
        snapshot = make_snapshot(
            node_id=node.node_id,
            times=times,
            values=values,
            secret_key=f"secret-{node.node_id}"
        )
        node.snapshot = snapshot
        
        # Compute Ω-state using CAL
        omega = node.cal.compute_omega_state(
            token_data={"rate": 5.0 + (round_num * 0.05) + (int(node.node_id.split('-')[-1]) * 0.1)},
            sentiment_data={"energy": 0.7 + (round_num * 0.01) + (int(node.node_id.split('-')[-1]) * 0.005)},
            semantic_data={"shift": 0.3 + (round_num * 0.005) + (int(node.node_id.split('-')[-1]) * 0.002)},
            attention_data=[
                0.1 + (round_num * 0.001) + (int(node.node_id.split('-')[-1]) * 0.01),
                0.2 + (round_num * 0.002) + (int(node.node_id.split('-')[-1]) * 0.01),
                0.3 + (round_num * 0.003) + (int(node.node_id.split('-')[-1]) * 0.01),
                0.4 + (round_num * 0.004) + (int(node.node_id.split('-')[-1]) * 0.01),
                0.5 + (round_num * 0.005) + (int(node.node_id.split('-')[-1]) * 0.01)
            ]
        )
        
        node.omega_state = omega
        return omega

    def _process_node_round(self, node: NodeState, round_num: int) -> Tuple[str, float, Dict[str, Any]]:
        """Process a single round for a node"""
        try:
            # Compute Ω-state
            omega = self._compute_node_omega_state(node, round_num)
            
            # Store coherence in history
            node.coherence_history.append(omega.coherence_score)
            
            # Return results
            return node.node_id, omega.coherence_score, {
                "token_rate": omega.token_rate,
                "sentiment_energy": omega.sentiment_energy,
                "semantic_shift": omega.semantic_shift,
                "modulator": omega.modulator
            }
        except Exception as e:
            logger.error(f"Error processing node {node.node_id}: {e}")
            return node.node_id, 0.0, {}

    def _compute_network_coherence(self) -> float:
        """Compute overall network coherence"""
        if not self.nodes:
            return 0.0
            
        active_coherences = [
            node.omega_state.coherence_score 
            for node in self.nodes 
            if node.omega_state is not None and node.is_active
        ]
        
        if not active_coherences:
            return 0.0
            
        return float(np.mean(active_coherences))

    def _validate_network_consensus(self) -> Tuple[bool, float]:
        """Validate consensus across the network"""
        if len(self.nodes) < 2:
            return False, 0.0
            
        # Get snapshots from active nodes
        active_snapshots = [
            node.snapshot 
            for node in self.nodes 
            if node.snapshot is not None and node.is_active
        ]
        
        if len(active_snapshots) < 2:
            return False, 0.0
            
        # Perform recursive validation
        is_valid, proof_bundle = recursive_validate(
            active_snapshots, 
            threshold=self.config["validation_threshold"]
        )
        
        aggregated_cs = proof_bundle.aggregated_CS if proof_bundle else 0.0
        return is_valid, aggregated_cs

    def _collect_simulation_metrics(self) -> SimulationMetrics:
        """Collect metrics from the current simulation state"""
        # Network coherence metrics
        network_coherences = [
            node.omega_state.coherence_score 
            for node in self.nodes 
            if node.omega_state is not None and node.is_active
        ]
        
        avg_coherence = float(np.mean(network_coherences)) if network_coherences else 0.0
        coherence_variance = float(np.var(network_coherences)) if network_coherences else 0.0
        
        # Token distribution
        token_totals = {"FLX": 0.0, "CHR": 0.0, "PSY": 0.0, "ATR": 0.0, "RES": 0.0}
        active_nodes = [node for node in self.nodes if node.is_active]
        
        for node in active_nodes:
            for token, balance in node.token_balances.items():
                token_totals[token] += balance
                
        avg_token_distribution = {
            token: total / len(active_nodes) if active_nodes else 0.0 
            for token, total in token_totals.items()
        }
        
        # Validation success rate (simplified)
        validation_success_rate = 1.0 if avg_coherence > self.config["validation_threshold"] else 0.0
        
        metrics = SimulationMetrics(
            avg_network_coherence=avg_coherence,
            coherence_variance=coherence_variance,
            validation_success_rate=validation_success_rate,
            avg_token_distribution=avg_token_distribution,
            simulation_duration=0.0,  # Will be set by caller
            nodes_processed=len(active_nodes)
        )
        
        return metrics

    def run_simulation_round(self, round_num: int) -> Dict[str, Any]:
        """Run a single round of simulation"""
        start_time = time.time()
        logger.info(f"Starting simulation round {round_num} with {len(self.nodes)} nodes")
        
        # Process all nodes in parallel
        results = {}
        with ThreadPoolExecutor(max_workers=self.config["max_concurrent_threads"]) as executor:
            # Submit tasks for all active nodes
            future_to_node = {
                executor.submit(self._process_node_round, node, round_num): node 
                for node in self.nodes 
                if node.is_active
            }
            
            # Collect results
            for future in as_completed(future_to_node):
                node_id, coherence, details = future.result()
                results[node_id] = {
                    "coherence": coherence,
                    "details": details
                }
        
        # Compute network metrics
        network_coherence = self._compute_network_coherence()
        is_valid, aggregated_cs = self._validate_network_consensus()
        
        # Collect simulation metrics
        metrics = self._collect_simulation_metrics()
        metrics.simulation_duration = time.time() - start_time
        self.metrics_history.append(metrics)
        
        # Log round results
        round_duration = time.time() - start_time
        logger.info(f"Round {round_num} completed in {round_duration:.2f}s - "
                   f"Network Coherence: {network_coherence:.4f}, Validation: {'PASS' if is_valid else 'FAIL'}")
        
        return {
            "round": round_num,
            "network_coherence": network_coherence,
            "aggregated_cs": aggregated_cs,
            "is_valid": is_valid,
            "node_results": results,
            "metrics": asdict(metrics),
            "duration": round_duration
        }

    def simulate_network_shock(self, shock_magnitude: float = 1.0, affected_nodes: float = 0.3):
        """Simulate a network shock affecting a percentage of nodes"""
        num_affected = int(len(self.nodes) * affected_nodes)
        affected_indices = np.random.choice(len(self.nodes), num_affected, replace=False)
        
        logger.info(f"Simulating network shock (magnitude: {shock_magnitude}) affecting {num_affected} nodes")
        
        for idx in affected_indices:
            node = self.nodes[idx]
            if node.omega_state:
                # Apply shock by degrading coherence
                shocked_coherence = max(0.0, node.omega_state.coherence_score - shock_magnitude)
                node.omega_state.coherence_score = shocked_coherence
                node.coherence_history.append(shocked_coherence)
                
                logger.debug(f"Node {node.node_id} shocked - Coherence: {shocked_coherence:.4f}")

    def simulate_node_failure(self, failure_rate: float = 0.1):
        """Simulate random node failures"""
        num_failures = int(len(self.nodes) * failure_rate)
        failure_indices = np.random.choice(len(self.nodes), num_failures, replace=False)
        
        logger.info(f"Simulating {num_failures} node failures")
        
        for idx in failure_indices:
            node = self.nodes[idx]
            if node.is_active:
                node.is_active = False
                logger.debug(f"Node {node.node_id} failed")

    def run_full_simulation(self) -> List[Dict[str, Any]]:
        """Run the complete multi-node simulation"""
        logger.info(f"Starting full simulation with {self.config['simulation_rounds']} rounds")
        
        results = []
        start_time = time.time()
        
        for round_num in range(self.config["simulation_rounds"]):
            # Run simulation round
            round_result = self.run_simulation_round(round_num)
            results.append(round_result)
            
            # Every 10 rounds, simulate some network events
            if round_num > 0 and round_num % 10 == 0:
                # 30% chance of network shock
                if np.random.random() < 0.3:
                    self.simulate_network_shock(shock_magnitude=np.random.uniform(0.1, 0.5))
                
                # 10% chance of node failures
                if np.random.random() < 0.1:
                    self.simulate_node_failure(failure_rate=np.random.uniform(0.05, 0.15))
            
            # Log progress every 20 rounds
            if round_num > 0 and round_num % 20 == 0:
                elapsed = time.time() - start_time
                logger.info(f"Progress: {round_num}/{self.config['simulation_rounds']} rounds "
                           f"({elapsed:.2f}s elapsed)")
        
        total_duration = time.time() - start_time
        logger.info(f"Simulation completed in {total_duration:.2f}s")
        
        return results

    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate a performance report from the simulation"""
        if not self.metrics_history:
            return {}
        
        # Aggregate metrics
        avg_network_coherences = [m.avg_network_coherence for m in self.metrics_history]
        validation_rates = [m.validation_success_rate for m in self.metrics_history]
        coherences_variances = [m.coherence_variance for m in self.metrics_history]
        
        report = {
            "simulation_summary": {
                "total_rounds": len(self.metrics_history),
                "total_nodes": len(self.nodes),
                "active_nodes": len([n for n in self.nodes if n.is_active]),
                "failed_nodes": len([n for n in self.nodes if not n.is_active])
            },
            "coherence_metrics": {
                "final_network_coherence": avg_network_coherences[-1] if avg_network_coherences else 0.0,
                "avg_network_coherence": float(np.mean(avg_network_coherences)),
                "max_network_coherence": float(np.max(avg_network_coherences)),
                "min_network_coherence": float(np.min(avg_network_coherences)),
                "coherence_stability": float(np.std(avg_network_coherences))
            },
            "validation_metrics": {
                "success_rate": float(np.mean(validation_rates)),
                "total_validations": len(validation_rates),
                "successful_validations": sum(validation_rates)
            },
            "network_stability": {
                "avg_variance": float(np.mean(coherences_variances)),
                "max_variance": float(np.max(coherences_variances)),
                "stability_score": 1.0 - float(np.mean(coherences_variances))  # Higher is better
            },
            "token_distribution": self.metrics_history[-1].avg_token_distribution if self.metrics_history else {}
        }
        
        return report

    def plot_simulation_results(self, results: List[Dict[str, Any]], save_path: Optional[str] = None):
        """Plot simulation results"""
        if not results:
            logger.warning("No results to plot")
            return
        
        # Extract data for plotting
        rounds = [r["round"] for r in results]
        network_coherences = [r["network_coherence"] for r in results]
        aggregated_cs = [r["aggregated_cs"] for r in results]
        durations = [r["duration"] for r in results]
        
        # Create plots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Multi-Node Simulation Results - CAL-RΦV Fusion', fontsize=16)
        
        # Network Coherence over time
        ax1.plot(rounds, network_coherences, 'b-', linewidth=2, label='Network Coherence')
        ax1.set_xlabel('Simulation Round')
        ax1.set_ylabel('Coherence Score')
        ax1.set_title('Network Coherence Over Time')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Aggregated Coherence Score
        ax2.plot(rounds, aggregated_cs, 'g-', linewidth=2, label='Aggregated CS')
        ax2.set_xlabel('Simulation Round')
        ax2.set_ylabel('Coherence Score')
        ax2.set_title('Aggregated Coherence Score')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Round Duration
        ax3.plot(rounds, durations, 'r-', linewidth=2, label='Round Duration')
        ax3.set_xlabel('Simulation Round')
        ax3.set_ylabel('Duration (seconds)')
        ax3.set_title('Processing Time Per Round')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # Validation Success (if we have that data)
        valid_rounds = [1 if r["is_valid"] else 0 for r in results]
        ax4.plot(rounds, valid_rounds, 'm-', linewidth=2, label='Validation Success')
        ax4.set_xlabel('Simulation Round')
        ax4.set_ylabel('Success (1=Pass, 0=Fail)')
        ax4.set_title('Consensus Validation Success')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        
        plt.show()

    def save_simulation_data(self, results: List[Dict[str, Any]], filepath: str):
        """Save simulation data to file"""
        data = {
            "simulation_config": self.config,
            "network_id": self.network_id,
            "total_nodes": self.num_nodes,
            "results": results,
            "metrics_history": [asdict(m) for m in self.metrics_history],
            "final_report": self.generate_performance_report()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Simulation data saved to {filepath}")


def main():
    """Main function to run the multi-node simulation"""
    logger.info("Starting Multi-Node Simulation for Quantum Currency v0.2.0")
    
    # Create simulator
    simulator = MultiNodeSimulator(num_nodes=20, network_id="qc-sim-beta-001")
    
    # Run simulation
    results = simulator.run_full_simulation()
    
    # Generate report
    report = simulator.generate_performance_report()
    
    # Print summary
    print("\n" + "="*60)
    print("MULTI-NODE SIMULATION SUMMARY")
    print("="*60)
    print(f"Network Coherence: {report['coherence_metrics']['avg_network_coherence']:.4f}")
    print(f"Validation Success: {report['validation_metrics']['success_rate']:.2%}")
    print(f"Stability Score: {report['network_stability']['stability_score']:.4f}")
    print(f"Active Nodes: {report['simulation_summary']['active_nodes']}")
    print(f"Failed Nodes: {report['simulation_summary']['failed_nodes']}")
    
    # Save results
    timestamp = int(time.time())
    simulator.save_simulation_data(results, f"simulation_results_{timestamp}.json")
    simulator.plot_simulation_results(results, f"simulation_plot_{timestamp}.png")
    
    logger.info("Multi-node simulation completed successfully")


if __name__ == "__main__":
    main()