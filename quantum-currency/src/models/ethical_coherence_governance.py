#!/usr/bin/env python3
"""
Ethical Coherence Governance (ECG) Engine
Dynamic rule engine ensuring expansion remains benevolent, balanced, and transparent

This module implements:
1. Ethical validation of governance proposals
2. Coherence-based ethical scoring
3. Transparency reporting
4. Benevolence monitoring
5. Balance enforcement mechanisms
"""

import numpy as np
from typing import List, Dict, Tuple, Any, Optional, Union
import logging
import time
import math
from dataclasses import dataclass, field
from .ai_governance import AIGovernance, GovernanceProposal, Validator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class EthicalRule:
    """Represents an ethical rule in the ECG system"""
    rule_id: str
    name: str
    description: str
    category: str  # "benevolence", "transparency", "balance", "consent"
    weight: float  # Importance weight (0.0 to 1.0)
    threshold: float  # Minimum score required (0.0 to 1.0)
    formula: str  # Formula for calculating rule compliance
    active: bool = True
    last_updated: float = 0.0

@dataclass
class EthicalAssessment:
    """Represents an ethical assessment of a proposal or action"""
    assessment_id: str
    target_id: str  # Proposal ID or action identifier
    timestamp: float
    overall_score: float  # Overall ethical score (0.0 to 1.0)
    category_scores: Dict[str, float]  # Scores by category
    rule_compliance: Dict[str, float]  # Compliance by rule
    violations: List[str]  # List of violated rule IDs
    recommendations: List[str]  # Recommendations for improvement
    approved: bool = False  # Whether ethically approved

@dataclass
class TransparencyReport:
    """Represents a transparency report for governance activities"""
    report_id: str
    period_start: float
    period_end: float
    generated_at: float
    activities: List[Dict[str, Any]]  # List of governance activities
    ethical_scores: Dict[str, float]  # Ethical scores for activities
    violations_summary: Dict[str, int]  # Count of violations by type
    recommendations: List[str]  # System-wide recommendations

class EthicalCoherenceGovernance:
    """
    Ethical Coherence Governance (ECG) Engine
    Ensures expansion remains benevolent, balanced, and transparent
    """
    
    def __init__(self, ai_governance: AIGovernance):
        self.ai_governance = ai_governance
        self.rules: Dict[str, EthicalRule] = {}
        self.assessments: Dict[str, EthicalAssessment] = {}
        self.transparency_reports: Dict[str, TransparencyReport] = {}
        self.violation_history: List[Dict[str, Any]] = []
        
        # Initialize default ethical rules
        self._initialize_default_rules()
        
        logger.info("⚖️ Ethical Coherence Governance (ECG) Engine initialized")
    
    def _initialize_default_rules(self):
        """Initialize default ethical rules for the ECG system"""
        default_rules = [
            EthicalRule(
                rule_id="benevolence_001",
                name="Benefit to Collective",
                description="Proposals must demonstrate clear benefit to the collective system",
                category="benevolence",
                weight=0.25,
                threshold=0.7,
                formula="collective_benefit_score >= 0.7"
            ),
            EthicalRule(
                rule_id="benevolence_002",
                name="No Harm Principle",
                description="Proposals must not cause demonstrable harm to participants",
                category="benevolence",
                weight=0.3,
                threshold=0.9,
                formula="harm_risk_score <= 0.1"
            ),
            EthicalRule(
                rule_id="transparency_001",
                name="Full Disclosure",
                description="All proposal impacts must be fully disclosed",
                category="transparency",
                weight=0.2,
                threshold=0.95,
                formula="disclosure_completeness >= 0.95"
            ),
            EthicalRule(
                rule_id="transparency_002",
                name="Audit Trail",
                description="All governance actions must maintain complete audit trails",
                category="transparency",
                weight=0.15,
                threshold=1.0,
                formula="audit_trail_integrity == 1.0"
            ),
            EthicalRule(
                rule_id="balance_001",
                name="Equitable Distribution",
                description="Benefits and burdens must be distributed equitably",
                category="balance",
                weight=0.2,
                threshold=0.8,
                formula="distribution_equity_score >= 0.8"
            ),
            EthicalRule(
                rule_id="balance_002",
                name="Power Concentration Limit",
                description="No single entity should gain excessive control",
                category="balance",
                weight=0.25,
                threshold=0.9,
                formula="power_concentration_score <= 0.1"
            ),
            EthicalRule(
                rule_id="consent_001",
                name="Explicit Consent",
                description="Affected parties must provide explicit consent for significant changes",
                category="consent",
                weight=0.2,
                threshold=0.95,
                formula="consent_obtained == True"
            ),
            EthicalRule(
                rule_id="consent_002",
                name="Opt-out Availability",
                description="Participants must have clear opt-out mechanisms",
                category="consent",
                weight=0.15,
                threshold=1.0,
                formula="opt_out_mechanism_available == True"
            )
        ]
        
        for rule in default_rules:
            self.rules[rule.rule_id] = rule
        
        logger.info(f"Intialized {len(default_rules)} default ethical rules")
    
    def add_ethical_rule(self, rule: EthicalRule) -> bool:
        """
        Add a new ethical rule to the ECG system
        
        Args:
            rule: EthicalRule to add
            
        Returns:
            bool: True if rule added successfully
        """
        if rule.rule_id in self.rules:
            logger.warning(f"Rule {rule.rule_id} already exists")
            return False
        
        self.rules[rule.rule_id] = rule
        rule.last_updated = time.time()
        logger.info(f"Added ethical rule: {rule.name}")
        return True
    
    def update_ethical_rule(self, rule_id: str, **kwargs) -> bool:
        """
        Update an existing ethical rule
        
        Args:
            rule_id: ID of rule to update
            **kwargs: Fields to update
            
        Returns:
            bool: True if rule updated successfully
        """
        rule = self.rules.get(rule_id)
        if not rule:
            logger.warning(f"Rule {rule_id} not found")
            return False
        
        # Update fields
        for key, value in kwargs.items():
            if hasattr(rule, key):
                setattr(rule, key, value)
        
        rule.last_updated = time.time()
        logger.info(f"Updated ethical rule: {rule.name}")
        return True
    
    def remove_ethical_rule(self, rule_id: str) -> bool:
        """
        Remove an ethical rule from the ECG system
        
        Args:
            rule_id: ID of rule to remove
            
        Returns:
            bool: True if rule removed successfully
        """
        if rule_id not in self.rules:
            logger.warning(f"Rule {rule_id} not found")
            return False
        
        rule_name = self.rules[rule_id].name
        del self.rules[rule_id]
        logger.info(f"Removed ethical rule: {rule_name}")
        return True
    
    def assess_proposal_ethics(self, proposal: GovernanceProposal) -> EthicalAssessment:
        """
        Assess the ethical compliance of a governance proposal
        
        Args:
            proposal: GovernanceProposal to assess
            
        Returns:
            EthicalAssessment: Assessment results
        """
        assessment_id = f"assess_{int(time.time() * 1000000)}_{hash(proposal.id) % 10000}"
        
        # Calculate category scores
        category_scores = self._calculate_category_scores(proposal)
        
        # Calculate rule compliance
        rule_compliance = self._calculate_rule_compliance(proposal)
        
        # Identify violations
        violations = []
        for rule_id, compliance_score in rule_compliance.items():
            rule = self.rules.get(rule_id)
            if rule and compliance_score < rule.threshold:
                violations.append(rule_id)
        
        # Calculate overall score
        overall_score = self._calculate_overall_ethical_score(category_scores, rule_compliance)
        
        # Generate recommendations
        recommendations = self._generate_ethical_recommendations(proposal, violations)
        
        # Determine approval
        approved = len(violations) == 0 and overall_score >= 0.8
        
        assessment = EthicalAssessment(
            assessment_id=assessment_id,
            target_id=proposal.id,
            timestamp=time.time(),
            overall_score=overall_score,
            category_scores=category_scores,
            rule_compliance=rule_compliance,
            violations=violations,
            recommendations=recommendations,
            approved=approved
        )
        
        self.assessments[assessment_id] = assessment
        
        # Log violations
        if violations:
            violation_record = {
                "timestamp": time.time(),
                "proposal_id": proposal.id,
                "violations": violations,
                "overall_score": overall_score
            }
            self.violation_history.append(violation_record)
        
        logger.info(f"Ethical assessment for proposal {proposal.id}: "
                   f"score={overall_score:.4f}, approved={approved}")
        
        return assessment
    
    def _calculate_category_scores(self, proposal: GovernanceProposal) -> Dict[str, float]:
        """
        Calculate ethical scores by category for a proposal
        
        Args:
            proposal: GovernanceProposal to assess
            
        Returns:
            Dict[str, float]: Scores by category
        """
        # This is a simplified implementation - in a real system, this would involve
        # complex analysis of the proposal content, target changes, and potential impacts
        
        categories = ["benevolence", "transparency", "balance", "consent"]
        scores = {}
        
        # Simulate scoring based on proposal characteristics
        # In a real implementation, this would analyze the actual proposal content
        base_score = 0.7 + (hash(proposal.title) % 30) / 100.0  # 0.7 to 0.99
        
        for category in categories:
            # Add some randomness while maintaining base score influence
            category_modifier = (hash(f"{proposal.id}{category}") % 20 - 10) / 100.0  # -0.1 to +0.1
            scores[category] = max(0.0, min(1.0, base_score + category_modifier))
        
        return scores
    
    def _calculate_rule_compliance(self, proposal: GovernanceProposal) -> Dict[str, float]:
        """
        Calculate compliance scores for each ethical rule
        
        Args:
            proposal: GovernanceProposal to assess
            
        Returns:
            Dict[str, float]: Compliance scores by rule
        """
        compliance_scores = {}
        
        # Simulate compliance calculation
        # In a real implementation, this would evaluate the proposal against each rule
        for rule_id, rule in self.rules.items():
            # Base compliance with some randomness
            base_compliance = 0.6 + (hash(f"{proposal.id}{rule_id}") % 40) / 100.0  # 0.6 to 0.99
            compliance_scores[rule_id] = max(0.0, min(1.0, base_compliance))
        
        return compliance_scores
    
    def _calculate_overall_ethical_score(self, category_scores: Dict[str, float], 
                                       rule_compliance: Dict[str, float]) -> float:
        """
        Calculate overall ethical score based on category scores and rule compliance
        
        Args:
            category_scores: Scores by category
            rule_compliance: Compliance scores by rule
            
        Returns:
            float: Overall ethical score (0.0 to 1.0)
        """
        # Weighted average of category scores
        category_weight = 0.4
        categories_score = np.mean(list(category_scores.values())) if category_scores else 0.5
        
        # Weighted average of rule compliance
        rules_weight = 0.6
        rules_score = np.mean(list(rule_compliance.values())) if rule_compliance else 0.5
        
        overall_score = float((category_weight * categories_score) + (rules_weight * rules_score))
        return max(0.0, min(1.0, overall_score))
    
    def _generate_ethical_recommendations(self, proposal: GovernanceProposal, 
                                        violations: List[str]) -> List[str]:
        """
        Generate ethical recommendations for improving a proposal
        
        Args:
            proposal: GovernanceProposal to assess
            violations: List of violated rule IDs
            
        Returns:
            List[str]: Recommendations for improvement
        """
        recommendations = []
        
        # Generic recommendations based on violations
        if violations:
            recommendations.append("Review and address all ethical violations before resubmission")
            recommendations.append("Ensure full transparency in proposal documentation")
            recommendations.append("Conduct impact assessment on affected stakeholders")
        
        # Category-specific recommendations
        recommendations.append("Consider broader systemic implications of proposed changes")
        recommendations.append("Ensure equitable distribution of benefits and responsibilities")
        recommendations.append("Maintain clear audit trails for all governance actions")
        
        return recommendations
    
    def generate_transparency_report(self, period_days: int = 30) -> TransparencyReport:
        """
        Generate a transparency report for governance activities
        
        Args:
            period_days: Number of days to include in report
            
        Returns:
            TransparencyReport: Generated transparency report
        """
        report_id = f"trans_{int(time.time() * 1000000)}"
        period_end = time.time()
        period_start = period_end - (period_days * 24 * 60 * 60)
        
        # Collect activities (simplified implementation)
        activities = []
        ethical_scores = {}
        violations_summary = {}
        
        # Include recent assessments
        for assessment_id, assessment in self.assessments.items():
            if period_start <= assessment.timestamp <= period_end:
                activities.append({
                    "type": "ethical_assessment",
                    "target_id": assessment.target_id,
                    "timestamp": assessment.timestamp,
                    "result": "approved" if assessment.approved else "rejected"
                })
                ethical_scores[assessment.target_id] = assessment.overall_score
                
                # Summarize violations
                for violation in assessment.violations:
                    violations_summary[violation] = violations_summary.get(violation, 0) + 1
        
        # Include recent violations
        for violation_record in self.violation_history:
            if period_start <= violation_record["timestamp"] <= period_end:
                activities.append({
                    "type": "ethical_violation",
                    "target_id": violation_record["proposal_id"],
                    "timestamp": violation_record["timestamp"],
                    "violations": violation_record["violations"]
                })
        
        # Generate system-wide recommendations
        recommendations = [
            "Continue monitoring ethical compliance of all governance proposals",
            "Review and update ethical rules as needed based on emerging patterns",
            "Enhance transparency reporting with more detailed impact assessments"
        ]
        
        report = TransparencyReport(
            report_id=report_id,
            period_start=period_start,
            period_end=period_end,
            generated_at=time.time(),
            activities=activities,
            ethical_scores=ethical_scores,
            violations_summary=violations_summary,
            recommendations=recommendations
        )
        
        self.transparency_reports[report_id] = report
        logger.info(f"Generated transparency report: {report_id}")
        return report
    
    def get_ethical_compliance_summary(self) -> Dict[str, Any]:
        """
        Get a summary of ethical compliance across the system
        
        Returns:
            Dict[str, Any]: Compliance summary
        """
        total_assessments = len(self.assessments)
        approved_assessments = len([a for a in self.assessments.values() if a.approved])
        total_violations = len(self.violation_history)
        
        # Category compliance rates
        category_compliance = {}
        if self.assessments:
            for category in ["benevolence", "transparency", "balance", "consent"]:
                scores = []
                for assessment in self.assessments.values():
                    if category in assessment.category_scores:
                        scores.append(assessment.category_scores[category])
                if scores:
                    category_compliance[category] = float(np.mean(scores))
        
        # Rule compliance rates
        rule_compliance = {}
        if self.assessments:
            for rule_id in self.rules:
                scores = []
                for assessment in self.assessments.values():
                    if rule_id in assessment.rule_compliance:
                        scores.append(assessment.rule_compliance[rule_id])
                if scores:
                    rule_compliance[rule_id] = float(np.mean(scores))
        
        return {
            "total_assessments": total_assessments,
            "approved_rate": approved_assessments / max(1, total_assessments),
            "total_violations": total_violations,
            "category_compliance": category_compliance,
            "rule_compliance": rule_compliance,
            "timestamp": time.time()
        }
    
    def enforce_ethical_governance(self, proposal: GovernanceProposal) -> Tuple[bool, str]:
        """
        Enforce ethical governance by assessing and potentially blocking proposals
        
        Args:
            proposal: GovernanceProposal to assess
            
        Returns:
            Tuple[bool, str]: (approved, message)
        """
        # Perform ethical assessment
        assessment = self.assess_proposal_ethics(proposal)
        
        if assessment.approved:
            return True, f"Proposal {proposal.id} ethically approved"
        else:
            violation_details = ", ".join(assessment.violations)
            return False, f"Proposal {proposal.id} ethically rejected due to violations: {violation_details}"

# Example usage and testing
if __name__ == "__main__":
    # This would normally integrate with the existing AI governance system
    from unittest.mock import Mock
    from .ai_governance import GovernanceProposal
    
    # Create mock AI governance system
    mock_governance = Mock(spec=AIGovernance)
    
    # Create ECG engine
    ecg = EthicalCoherenceGovernance(mock_governance)
    
    print(f"Initialized ECG with {len(ecg.rules)} ethical rules")
    
    # Create a mock proposal using the actual GovernanceProposal class
    mock_proposal = GovernanceProposal(
        id="prop_12345",
        proposer_id="validator_1",
        title="Network Parameter Update",
        description="Update network parameters for better performance",
        target_omega={"token_rate": 1.2},
        timestamp=time.time()
    )
    
    # Assess proposal ethics
    assessment = ecg.assess_proposal_ethics(mock_proposal)
    
    print(f"Ethical assessment for '{mock_proposal.title}':")
    print(f"  Overall score: {assessment.overall_score:.4f}")
    print(f"  Approved: {assessment.approved}")
    print(f"  Violations: {len(assessment.violations)}")
    print(f"  Recommendations: {len(assessment.recommendations)}")
    
    # Generate transparency report
    report = ecg.generate_transparency_report(7)  # 7-day report
    print(f"\nGenerated transparency report:")
    print(f"  Activities: {len(report.activities)}")
    print(f"  Violations: {len(report.violations_summary)}")
    
    # Get compliance summary
    summary = ecg.get_ethical_compliance_summary()
    print(f"\nEthical compliance summary:")
    print(f"  Total assessments: {summary['total_assessments']}")
    print(f"  Approval rate: {summary['approved_rate']:.2%}")
    print(f"  Total violations: {summary['total_violations']}")