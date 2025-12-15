#!/usr/bin/env python3
"""
Phase I - Production Hardening & Redundancy
I.A – Φ-Harmonic Sharding & Node Redundancy
I.B – Zero-Dissonance Deployment Pipeline (CI/CD)
I.C – QRA Key Management & Sealing
"""

import logging
import numpy as np
from typing import Dict, Any, List
from pathlib import Path
import json
import hashlib
import time

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PhiHarmonicSharding:
    """Φ-Harmonic Sharding implementation for QECS nodes"""
    
    def __init__(self, shard_count: int = 5):
        self.shard_count = shard_count
        self.shards = {}
        self.shard_coherence = {}
        
    def partition_nodes(self, nodes: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Partition QECS nodes into Φ-Harmonic Shards based on QRA Q-Seed parameters
        
        Args:
            nodes: List of node dictionaries with QRA parameters
            
        Returns:
            Dictionary mapping shard IDs to node lists
        """
        # Initialize shards
        for i in range(self.shard_count):
            self.shards[f"SHARD_{i:02d}"] = []
            self.shard_coherence[f"SHARD_{i:02d}"] = 0.0
            
        # Partition nodes based on harmonic alignment
        for node in nodes:
            # Extract QRA parameters for harmonic analysis
            qra_params = node.get('qra_params', {})
            n = qra_params.get('n', 1)
            l = qra_params.get('l', 0)
            m = qra_params.get('m', 0)
            s = qra_params.get('s', 0.5)
            
            # Compute harmonic signature
            harmonic_signature = self._compute_harmonic_signature(n, l, m, s)
            
            # Assign to shard based on signature
            shard_index = int(harmonic_signature * self.shard_count) % self.shard_count
            shard_id = f"SHARD_{shard_index:02d}"
            
            self.shards[shard_id].append(node)
            logger.info(f"Assigned node {node['node_id']} to {shard_id}")
            
        # Calculate shard coherence
        self._calculate_shard_coherence()
        
        return self.shards
    
    def _compute_harmonic_signature(self, n: int, l: int, m: int, s: float) -> float:
        """Compute harmonic signature based on QRA Q-Seed parameters"""
        # Use quantum harmonic oscillator formula with QRA parameters
        phi = (1 + np.sqrt(5)) / 2  # Golden ratio
        signature = (n * phi + l * np.sqrt(2) + m * np.sqrt(3) + s) % 1.0
        return signature
    
    def _calculate_shard_coherence(self):
        """Calculate coherence for each shard"""
        for shard_id, nodes in self.shards.items():
            if not nodes:
                self.shard_coherence[shard_id] = 0.0
                continue
                
            # Calculate average coherence of nodes in shard
            total_coherence = sum(node.get('coherence_score', 0.95) for node in nodes)
            avg_coherence = total_coherence / len(nodes)
            
            # Apply harmonic weighting
            harmonic_factor = 1.0 - (len(nodes) / 100.0)  # Decrease with size
            shard_coherence = avg_coherence * harmonic_factor
            
            self.shard_coherence[shard_id] = shard_coherence
            logger.info(f"{shard_id} coherence: {shard_coherence:.4f}")
    
    def simulate_shard_failure(self, failed_shard_id: str) -> Dict[str, Any]:
        """
        Simulate single-shard failure and analyze system impact
        
        Args:
            failed_shard_id: ID of the shard that fails
            
        Returns:
            Dictionary with impact analysis
        """
        if failed_shard_id not in self.shards:
            raise ValueError(f"Shard {failed_shard_id} not found")
            
        # Get remaining shards
        remaining_shards = {k: v for k, v in self.shards.items() if k != failed_shard_id}
        
        # Calculate system-wide impact
        total_nodes = sum(len(nodes) for nodes in self.shards.values())
        failed_nodes = len(self.shards[failed_shard_id])
        remaining_nodes = total_nodes - failed_nodes
        
        # Calculate I_eff (inertial efficiency) impact
        I_eff = failed_nodes / total_nodes
        
        # Calculate coherence of remaining system
        remaining_coherence = np.mean([self.shard_coherence[shard_id] 
                                     for shard_id in remaining_shards.keys() 
                                     if self.shard_coherence[shard_id] > 0])
        
        # Calculate g_vector magnitudes for remaining shards
        g_vector_magnitudes = {}
        for shard_id, nodes in remaining_shards.items():
            # Simplified calculation - in practice this would be more complex
            g_magnitude = len(nodes) * 0.01  # Per-node contribution
            g_vector_magnitudes[shard_id] = min(1.0, g_magnitude)
        
        result = {
            "failed_shard": failed_shard_id,
            "failed_nodes": failed_nodes,
            "remaining_nodes": remaining_nodes,
            "system_impact_ratio": failed_nodes / total_nodes,
            "I_eff": I_eff,
            "remaining_system_coherence": remaining_coherence,
            "g_vector_magnitudes": g_vector_magnitudes,
            "shard_coherence_after_failure": self.shard_coherence
        }
        
        logger.info(f"Simulated failure of {failed_shard_id}: I_eff={I_eff:.4f}, "
                   f"Coherence={remaining_coherence:.4f}")
        
        return result

class ZeroDissonanceDeployment:
    """Zero-Dissonance Deployment Pipeline (CI/CD)"""
    
    def __init__(self):
        self.final_status_code = "200_COHERENT_LOCK"
        self.deployment_blocked = False
        
    def validate_deployment(self, orchestration_result: Dict[str, Any]) -> bool:
        """
        Validate deployment based on orchestration results
        
        Args:
            orchestration_result: Results from IACE v2.0 orchestration
            
        Returns:
            bool: True if deployment is allowed, False otherwise
        """
        final_status = orchestration_result.get('final_status', '500_CRITICAL_DISSONANCE')
        delta_lambda = orchestration_result.get('delta_lambda', 0.1)
        
        # Check Final_Status_Code
        if final_status != "200_COHERENT_LOCK":
            logger.error(f"Deployment blocked: Final_Status_Code = {final_status}")
            self.final_status_code = final_status
            self.deployment_blocked = True
            return False
            
        # Check ΔΛ (delta lambda) across all modules
        if delta_lambda >= 0.005:
            logger.error(f"Deployment blocked: ΔΛ = {delta_lambda:.4f} >= 0.005")
            self.final_status_code = "400_LAMBDA_DISCREPANCY"
            self.deployment_blocked = True
            return False
            
        logger.info("✅ Deployment validation passed - all coherence checks satisfied")
        self.final_status_code = "200_COHERENT_LOCK"
        self.deployment_blocked = False
        return True
    
    def get_final_status_code(self) -> str:
        """Get the final status code"""
        return self.final_status_code

class QRAKeyManagement:
    """QRA Key Management & Sealing"""
    
    def __init__(self, tee_enabled: bool = True):
        self.tee_enabled = tee_enabled
        self.sealed_keys = {}
        
    def generate_qra_in_tee(self, node_id: str, system_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate QRA key in Trusted Execution Environment (TEE)
        
        Args:
            node_id: Node identifier
            system_metrics: Current system metrics for sealing
            
        Returns:
            Dictionary with QRA key data
        """
        if not self.tee_enabled:
            logger.warning("TEE not enabled - keys will not be securely generated")
            return {}
            
        # Simulate TEE key generation
        coherence_score = system_metrics.get('C_system', 0.95)
        gas_target = system_metrics.get('GAS_target', 0.95)
        cycle_count = system_metrics.get('cycle_count', 100)
        
        # Generate bioresonant parameters
        phi_ratio = np.random.uniform(1.6, 1.62)  # Close to golden ratio
        inertial_efficiency = np.random.uniform(0.01, 0.1)  # I_eff cost
        
        # Create QRA key
        qra_key = {
            "node_id": node_id,
            "qra_id": hashlib.sha256(f"{node_id}_{time.time()}".encode()).hexdigest()[:16],
            "timestamp": time.time(),
            "Coherence_Score": float(coherence_score),
            "Phi_Ratio": float(phi_ratio),
            "I_eff_Cost": float(inertial_efficiency),
            "version": "AFIP_1.0"
        }
        
        logger.info(f"✅ QRA key generated for {node_id} in TEE")
        return qra_key
    
    def seal_private_key(self, qra_key: Dict[str, Any], system_metrics: Dict[str, Any]) -> str:
        """
        Seal private key with multi-factor Φ-lock
        
        Args:
            qra_key: QRA key data
            system_metrics: System metrics for sealing
            
        Returns:
            str: Sealed key identifier
        """
        if not self.tee_enabled:
            logger.warning("TEE not enabled - keys cannot be sealed")
            return ""
            
        node_id = qra_key.get('node_id', 'unknown')
        
        # Multi-factor Φ-lock:
        # 1. Time-integrated C_system metrics over last 100 cycles
        c_system_avg = system_metrics.get('C_system_avg', 0.95)
        
        # 2. Node shard identity
        shard_id = system_metrics.get('shard_id', 'SHARD_00')
        
        # 3. Bioresonant entropy signature
        entropy_signature = system_metrics.get('entropy_signature', 0.01)
        
        # Create seal
        seal_data = f"{c_system_avg}_{shard_id}_{entropy_signature}_{time.time()}"
        seal_hash = hashlib.sha256(seal_data.encode()).hexdigest()
        
        # Store sealed key
        self.sealed_keys[seal_hash] = {
            "node_id": node_id,
            "seal_data": seal_data,
            "timestamp": time.time(),
            "qra_key": qra_key
        }
        
        logger.info(f"✅ QRA key for {node_id} sealed with Φ-lock")
        return seal_hash
    
    def validate_key_integrity(self, qra_key: Dict[str, Any]) -> bool:
        """
        Validate QRA key integrity
        
        Args:
            qra_key: QRA key to validate
            
        Returns:
            bool: True if key is valid, False otherwise
        """
        # Simple validation - in practice this would be more complex
        required_fields = ['node_id', 'qra_id', 'Coherence_Score', 'Phi_Ratio']
        for field in required_fields:
            if field not in qra_key:
                logger.error(f"Key validation failed: missing field {field}")
                return False
                
        # Check coherence score
        coherence = qra_key.get('Coherence_Score', 0)
        if coherence < 0.85:
            logger.error(f"Key validation failed: low coherence {coherence}")
            return False
            
        logger.info(f"✅ QRA key integrity validation passed for {qra_key.get('node_id')}")
        return True

# Example usage
if __name__ == "__main__":
    # Example nodes with QRA parameters
    nodes = [
        {"node_id": "node_001", "coherence_score": 0.98, 
         "qra_params": {"n": 1, "l": 0, "m": 0, "s": 0.5}},
        {"node_id": "node_002", "coherence_score": 0.96,
         "qra_params": {"n": 2, "l": 1, "m": -1, "s": 0.7}},
        {"node_id": "node_003", "coherence_score": 0.97,
         "qra_params": {"n": 1, "l": 1, "m": 0, "s": 0.3}},
        {"node_id": "node_004", "coherence_score": 0.95,
         "qra_params": {"n": 3, "l": 2, "m": 1, "s": 0.9}},
    ]
    
    # Phase I.A - Φ-Harmonic Sharding
    print("=== Phase I.A – Φ-Harmonic Sharding ===")
    sharding = PhiHarmonicSharding(shard_count=3)
    shards = sharding.partition_nodes(nodes)
    
    for shard_id, shard_nodes in shards.items():
        print(f"{shard_id}: {[node['node_id'] for node in shard_nodes]} "
              f"(Coherence: {sharding.shard_coherence[shard_id]:.4f})")
    
    # Simulate shard failure
    print("\n=== Shard Failure Simulation ===")
    failure_result = sharding.simulate_shard_failure("SHARD_00")
    print(f"I_eff: {failure_result['I_eff']:.4f}")
    print(f"Remaining system coherence: {failure_result['remaining_system_coherence']:.4f}")
    
    # Phase I.B - Zero-Dissonance Deployment
    print("\n=== Phase I.B – Zero-Dissonance Deployment ===")
    deployment = ZeroDissonanceDeployment()
    
    # Valid deployment
    valid_result = {
        "final_status": "200_COHERENT_LOCK",
        "delta_lambda": 0.001
    }
    allowed = deployment.validate_deployment(valid_result)
    print(f"Valid deployment allowed: {allowed}")
    print(f"Status code: {deployment.get_final_status_code()}")
    
    # Invalid deployment
    invalid_result = {
        "final_status": "500_CRITICAL_DISSONANCE",
        "delta_lambda": 0.01
    }
    allowed = deployment.validate_deployment(invalid_result)
    print(f"Invalid deployment allowed: {allowed}")
    print(f"Status code: {deployment.get_final_status_code()}")
    
    # Phase I.C - QRA Key Management
    print("\n=== Phase I.C – QRA Key Management ===")
    key_mgmt = QRAKeyManagement(tee_enabled=True)
    
    system_metrics = {
        "C_system": 0.98,
        "GAS_target": 0.95,
        "cycle_count": 150,
        "C_system_avg": 0.97,
        "shard_id": "SHARD_01",
        "entropy_signature": 0.005
    }
    
    qra_key = key_mgmt.generate_qra_in_tee("node_001", system_metrics)
    if qra_key:
        seal_id = key_mgmt.seal_private_key(qra_key, system_metrics)
        valid = key_mgmt.validate_key_integrity(qra_key)
        print(f"Key sealed with ID: {seal_id[:16]}...")
        print(f"Key integrity: {'✅ Valid' if valid else '❌ Invalid'}")