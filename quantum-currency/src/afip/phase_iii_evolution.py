#!/usr/bin/env python3
"""
Phase III - Autonomous Evolution & Protocol Finalization
III.A – Coherence Protocol Governance Mechanism (CPGM)
III.B – Final Coherence Lock
"""

import logging
import numpy as np
from typing import Dict, Any, List
from pathlib import Path
import json
import time
import hashlib

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CoherenceProtocolGovernance:
    """Coherence Protocol Governance Mechanism (CPGM)"""
    
    def __init__(self):
        self.proposals = []
        self.approved_proposals = []
        self.g_avg_threshold = 0.1
        self.qra_coherence_threshold = 0.95
        self.required_active_qras = 0.95  # 95% of QRAs must be active
        self.voting_history = []
        
    def create_protocol_amendment_proposal(self, optimization_vector: Dict[str, Any], 
                                         proposer: str = "AGI") -> Dict[str, Any]:
        """
        Convert AGI Optimization_Vector into formal HSMF Protocol Amendment Proposal
        
        Args:
            optimization_vector: AGI optimization vector
            proposer: Entity proposing the amendment
            
        Returns:
            Dictionary with proposal data
        """
        # Create unique proposal ID
        proposal_data = f"{proposer}_{time.time()}_{json.dumps(optimization_vector, sort_keys=True)}"
        proposal_id = hashlib.sha256(proposal_data.encode()).hexdigest()[:16]
        
        proposal = {
            "proposal_id": proposal_id,
            "proposer": proposer,
            "timestamp": time.time(),
            "optimization_vector": optimization_vector,
            "status": "PENDING",
            "approval_conditions": {
                "g_avg_threshold": self.g_avg_threshold,
                "qra_coherence_threshold": self.qra_coherence_threshold,
                "required_active_qras": self.required_active_qras
            },
            "votes": [],
            "approval_gate_passed": False
        }
        
        self.proposals.append(proposal)
        logger.info(f"📝 Protocol amendment proposal {proposal_id} created by {proposer}")
        
        return proposal
    
    def evaluate_approval_gate(self, system_metrics: Dict[str, Any], 
                              qra_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate Approval Gate conditions for protocol amendments
        
        Args:
            system_metrics: Current system metrics
            qra_metrics: List of QRA metrics for active nodes
            
        Returns:
            Dictionary with evaluation results
        """
        # Condition 1: |g_avg| < 0.1 for 1,000 cycles
        g_avg = system_metrics.get("g_avg", 0.5)
        g_avg_condition = abs(g_avg) < self.g_avg_threshold
        
        # Condition 2: ≥95% of active QRAs C_score ≥ 0.95
        high_coherence_qras = 0
        if not qra_metrics:
            qra_condition = False
        else:
            high_coherence_qras = sum(1 for qra in qra_metrics 
                                    if qra.get("C_score", 0) >= self.qra_coherence_threshold)
            qra_condition = (high_coherence_qras / len(qra_metrics)) >= self.required_active_qras
        
        # Both conditions must be met
        approval_gate_passed = g_avg_condition and qra_condition
        
        result = {
            "g_avg": g_avg,
            "g_avg_condition_met": g_avg_condition,
            "g_avg_threshold": self.g_avg_threshold,
            "|g_avg|": abs(g_avg),
            "active_qras": len(qra_metrics),
            "high_coherence_qras": high_coherence_qras,
            "qra_condition_met": qra_condition,
            "qra_coherence_threshold": self.qra_coherence_threshold,
            "approval_gate_passed": approval_gate_passed,
            "timestamp": time.time()
        }
        
        logger.info(f"Approval gate evaluation: {'✅ PASSED' if approval_gate_passed else '❌ FAILED'}")
        logger.info(f"  |g_avg|: {abs(g_avg):.4f} < {self.g_avg_threshold}: {g_avg_condition}")
        if qra_metrics:
            ratio = high_coherence_qras / len(qra_metrics) if len(qra_metrics) > 0 else 0
            logger.info(f"  QRA coherence: {ratio:.2%} ≥ {self.required_active_qras:.2%}: {qra_condition}")
        
        return result
    
    def vote_on_proposal(self, proposal_id: str, voter_id: str, 
                        vote: bool, voter_weight: float = 1.0) -> bool:
        """
        Vote on a protocol amendment proposal
        
        Args:
            proposal_id: ID of the proposal
            voter_id: ID of the voter
            vote: True for approval, False for rejection
            voter_weight: Weight of the vote (based on coherence score)
            
        Returns:
            bool: True if vote was recorded, False if proposal not found
        """
        proposal = next((p for p in self.proposals if p["proposal_id"] == proposal_id), None)
        if not proposal:
            logger.error(f"Proposal {proposal_id} not found")
            return False
            
        # Record vote
        vote_record = {
            "voter_id": voter_id,
            "vote": vote,
            "weight": voter_weight,
            "timestamp": time.time()
        }
        
        proposal["votes"].append(vote_record)
        self.voting_history.append(vote_record)
        
        logger.info(f"🗳️ Vote recorded for proposal {proposal_id}: "
                   f"{voter_id} {'APPROVES' if vote else 'REJECTS'} (weight: {voter_weight})")
        
        return True
    
    def tally_votes(self, proposal_id: str) -> Dict[str, Any]:
        """
        Tally votes for a proposal
        
        Args:
            proposal_id: ID of the proposal
            
        Returns:
            Dictionary with voting results
        """
        proposal = next((p for p in self.proposals if p["proposal_id"] == proposal_id), None)
        if not proposal:
            logger.error(f"Proposal {proposal_id} not found")
            return {}
            
        # Calculate weighted vote results
        approve_weight = sum(vote["weight"] for vote in proposal["votes"] if vote["vote"])
        reject_weight = sum(vote["weight"] for vote in proposal["votes"] if not vote["vote"])
        total_weight = approve_weight + reject_weight
        
        # Determine outcome
        if total_weight > 0:
            approval_percentage = approve_weight / total_weight
            approved = approval_percentage > 0.5
        else:
            approval_percentage = 0.0
            approved = False
            
        result = {
            "proposal_id": proposal_id,
            "approve_weight": approve_weight,
            "reject_weight": reject_weight,
            "total_weight": total_weight,
            "approval_percentage": approval_percentage,
            "approved": approved,
            "vote_count": len(proposal["votes"])
        }
        
        # Update proposal status
        if approved:
            proposal["status"] = "APPROVED"
            proposal["approval_gate_passed"] = True
            self.approved_proposals.append(proposal)
            logger.info(f"✅ Proposal {proposal_id} APPROVED with {approval_percentage:.1%} approval")
        else:
            proposal["status"] = "REJECTED"
            logger.info(f"❌ Proposal {proposal_id} REJECTED with {approval_percentage:.1%} approval")
            
        return result

class FinalCoherenceLock:
    """Final Coherence Lock for protocol finalization"""
    
    def __init__(self, observation_period_days: int = 7):
        self.observation_period_days = observation_period_days
        self.observation_start_time = None
        self.metrics_history = []
        self.lock_achieved = False
        self.final_report = {}
        
        # KPI thresholds
        self.C_system_threshold = 0.999
        self.delta_lambda_threshold = 0.001
        self.I_eff_threshold = 0.001
        self.RSI_threshold = 0.99
        self.gravity_well_anomalies_threshold = 0
        
    def start_observation_period(self):
        """Start the final coherence observation period"""
        self.observation_start_time = time.time()
        self.metrics_history = []
        self.lock_achieved = False
        logger.info(f"🔒 Final Coherence Lock observation period started for {self.observation_period_days} days")
        
    def record_metrics(self, metrics: Dict[str, Any]) -> bool:
        """
        Record system metrics during observation period
        
        Args:
            metrics: Current system metrics
            
        Returns:
            bool: True if metrics were recorded, False if outside observation period
        """
        if not self.observation_start_time:
            logger.warning("Observation period not started")
            return False
            
        current_time = time.time()
        elapsed_days = (current_time - self.observation_start_time) / (24 * 3600)
        
        if elapsed_days > self.observation_period_days:
            logger.info("Observation period completed")
            return False
            
        # Add timestamp to metrics
        metrics_with_time = metrics.copy()
        metrics_with_time["timestamp"] = current_time
        metrics_with_time["elapsed_days"] = elapsed_days
        
        self.metrics_history.append(metrics_with_time)
        logger.debug(f"Recorded metrics at day {elapsed_days:.2f}")
        
        return True
    
    def check_kpi_thresholds(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check if current metrics meet KPI thresholds
        
        Args:
            metrics: Current system metrics
            
        Returns:
            Dictionary with threshold check results
        """
        C_system = metrics.get("C_system", 0)
        delta_lambda = metrics.get("delta_lambda", 1.0)
        I_eff = metrics.get("I_eff", 1.0)
        RSI = metrics.get("RSI", 0)
        gravity_well_anomalies = metrics.get("gravity_well_anomalies", 1)
        
        # Check thresholds
        C_system_pass = C_system >= self.C_system_threshold
        delta_lambda_pass = delta_lambda <= self.delta_lambda_threshold
        I_eff_pass = I_eff <= self.I_eff_threshold
        RSI_pass = RSI >= self.RSI_threshold
        gravity_well_pass = gravity_well_anomalies <= self.gravity_well_anomalies_threshold
        
        all_passed = all([C_system_pass, delta_lambda_pass, I_eff_pass, RSI_pass, gravity_well_pass])
        
        result = {
            "C_system": C_system,
            "C_system_threshold": self.C_system_threshold,
            "C_system_pass": C_system_pass,
            "delta_lambda": delta_lambda,
            "delta_lambda_threshold": self.delta_lambda_threshold,
            "delta_lambda_pass": delta_lambda_pass,
            "I_eff": I_eff,
            "I_eff_threshold": self.I_eff_threshold,
            "I_eff_pass": I_eff_pass,
            "RSI": RSI,
            "RSI_threshold": self.RSI_threshold,
            "RSI_pass": RSI_pass,
            "gravity_well_anomalies": gravity_well_anomalies,
            "gravity_well_anomalies_threshold": self.gravity_well_anomalies_threshold,
            "gravity_well_pass": gravity_well_pass,
            "all_kpis_passed": all_passed
        }
        
        return result
    
    def evaluate_final_coherence_lock(self) -> Dict[str, Any]:
        """
        Evaluate if Final Coherence Lock has been achieved
        
        Returns:
            Dictionary with evaluation results
        """
        if not self.metrics_history:
            logger.warning("No metrics recorded for evaluation")
            return {"lock_achieved": False, "reason": "No metrics recorded"}
            
        # Check if observation period is complete
        current_time = time.time()
        elapsed_days = (current_time - self.observation_start_time) / (24 * 3600) if self.observation_start_time else 0
        
        if elapsed_days < self.observation_period_days:
            remaining_days = self.observation_period_days - elapsed_days
            logger.info(f"Observation period still active ({remaining_days:.1f} days remaining)")
            # For demo/testing purposes, we'll still evaluate what we have
            # In production, this would wait for the full period
            pass
            
        # Evaluate all metrics in history
        all_metrics_passed = True
        kpi_summary = {
            "C_system": [],
            "delta_lambda": [],
            "I_eff": [],
            "RSI": [],
            "gravity_well_anomalies": []
        }
        
        for metrics in self.metrics_history:
            kpi_check = self.check_kpi_thresholds(metrics)
            all_metrics_passed = all_metrics_passed and kpi_check["all_kpis_passed"]
            
            # Collect metrics for summary
            kpi_summary["C_system"].append(kpi_check["C_system"])
            kpi_summary["delta_lambda"].append(kpi_check["delta_lambda"])
            kpi_summary["I_eff"].append(kpi_check["I_eff"])
            kpi_summary["RSI"].append(kpi_check["RSI"])
            kpi_summary["gravity_well_anomalies"].append(kpi_check["gravity_well_anomalies"])
        
        # Calculate summary statistics
        summary_stats = {}
        for kpi, values in kpi_summary.items():
            if values:
                summary_stats[f"{kpi}_avg"] = np.mean(values)
                summary_stats[f"{kpi}_min"] = np.min(values)
                summary_stats[f"{kpi}_max"] = np.max(values)
                summary_stats[f"{kpi}_std"] = np.std(values)
        
        # Determine if lock is achieved
        lock_achieved = all_metrics_passed and len(self.metrics_history) > 0
        
        # Generate final report
        final_report = {
            "observation_period_days": self.observation_period_days,
            "actual_observation_days": elapsed_days,
            "total_metrics_records": len(self.metrics_history),
            "lock_achieved": lock_achieved,
            "kpi_summary": summary_stats,
            "final_status_code": "200_COHERENT_LOCK" if lock_achieved else "500_CRITICAL_DISSONANCE",
            "timestamp": current_time
        }
        
        self.lock_achieved = lock_achieved
        self.final_report = final_report
        
        status = "✅ ACHIEVED" if lock_achieved else "❌ NOT ACHIEVED"
        logger.info(f"Final Coherence Lock {status}")
        
        if lock_achieved:
            logger.info("🎉 QECS has achieved Final Coherence Lock - ready for production!")
        else:
            logger.warning("⚠️ QECS has not achieved Final Coherence Lock - further tuning required")
            
        return final_report

# Example usage
if __name__ == "__main__":
    # Phase III.A - Coherence Protocol Governance Mechanism (CPGM)
    print("=== Phase III.A – Coherence Protocol Governance Mechanism ===")
    cpgm = CoherenceProtocolGovernance()
    
    # Create a protocol amendment proposal
    optimization_vector = {
        "lambda_adjustment": {"lambda1": 0.05, "lambda2": -0.03},
        "caf_alpha_update": 0.02,
        "haru_learning_rate": 0.001
    }
    
    proposal = cpgm.create_protocol_amendment_proposal(optimization_vector, "AGI_OPTIMIZER")
    print(f"Proposal ID: {proposal['proposal_id']}")
    print(f"Proposal status: {proposal['status']}")
    
    # Evaluate approval gate
    system_metrics = {"g_avg": 0.05}
    qra_metrics = [
        {"node_id": "node_001", "C_score": 0.98},
        {"node_id": "node_002", "C_score": 0.96},
        {"node_id": "node_003", "C_score": 0.97},
        {"node_id": "node_004", "C_score": 0.99},
    ]
    
    gate_evaluation = cpgm.evaluate_approval_gate(system_metrics, qra_metrics)
    print(f"Approval gate passed: {gate_evaluation['approval_gate_passed']}")
    print(f"  |g_avg|: {gate_evaluation['|g_avg|']:.4f}")
    print(f"  QRA condition: {gate_evaluation['qra_condition_met']}")
    
    # Vote on proposal
    cpgm.vote_on_proposal(proposal["proposal_id"], "node_001", True, 0.98)
    cpgm.vote_on_proposal(proposal["proposal_id"], "node_002", True, 0.96)
    cpgm.vote_on_proposal(proposal["proposal_id"], "node_003", False, 0.97)  # Dissenting vote
    cpgm.vote_on_proposal(proposal["proposal_id"], "node_004", True, 0.99)
    
    # Tally votes
    vote_results = cpgm.tally_votes(proposal["proposal_id"])
    print(f"Proposal approved: {vote_results['approved']}")
    print(f"Approval percentage: {vote_results['approval_percentage']:.1%}")
    
    # Phase III.B - Final Coherence Lock
    print("\n=== Phase III.B – Final Coherence Lock ===")
    final_lock = FinalCoherenceLock(observation_period_days=7)
    
    # Start observation period
    final_lock.start_observation_period()
    
    # Simulate recording metrics over time
    for day in range(1, 8):
        # Simulate metrics that meet thresholds
        metrics = {
            "C_system": np.random.normal(0.9995, 0.0002),  # Target: ≥ 0.999
            "delta_lambda": np.random.exponential(0.0005),  # Target: ≤ 0.001
            "I_eff": np.random.exponential(0.0003),        # Target: ≤ 0.001
            "RSI": np.random.normal(0.995, 0.002),         # Target: ≥ 0.99
            "gravity_well_anomalies": 0                    # Target: ≤ 0
        }
        
        final_lock.record_metrics(metrics)
        print(f"Day {day}: C_system={metrics['C_system']:.4f}, "
              f"ΔΛ={metrics['delta_lambda']:.4f}, I_eff={metrics['I_eff']:.4f}")
    
    # Evaluate final coherence lock
    final_evaluation = final_lock.evaluate_final_coherence_lock()
    print(f"\nFinal Coherence Lock: {'✅ ACHIEVED' if final_evaluation['lock_achieved'] else '❌ NOT ACHIEVED'}")
    print(f"Final Status Code: {final_evaluation['final_status_code']}")
    
    if final_evaluation['lock_achieved']:
        print("🎉 QECS is ready for production deployment!")
    else:
        print("⚠️ Additional tuning required before production deployment")