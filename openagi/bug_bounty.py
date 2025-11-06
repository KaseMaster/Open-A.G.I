#!/usr/bin/env python3
"""
Bug Bounty and Security Disclosure Framework for Quantum Currency System
Implements community-driven bug bounty and security disclosure policies

This module provides a framework for managing security vulnerability reporting,
bug bounty programs, and responsible disclosure practices for the quantum currency system.
"""

import sys
import os
import json
import time
import hashlib
import hmac
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from enum import Enum

# Add parent directory to path to import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import our existing modules
from openagi.compliance_framework import ComplianceFramework
from openagi.onchain_governance import OnChainGovernanceSystem

class VulnerabilitySeverity(Enum):
    """Severity levels for security vulnerabilities"""
    LOW = "low"          # Minor issues with minimal impact
    MEDIUM = "medium"    # Moderate issues that could affect some users
    HIGH = "high"        # Serious issues that could affect many users
    CRITICAL = "critical" # Critical issues that could compromise the system

class BountyStatus(Enum):
    """Status of bug bounty submissions"""
    SUBMITTED = "submitted"     # Initial submission
    UNDER_REVIEW = "under_review" # Being reviewed by security team
    VALID = "valid"            # Valid vulnerability confirmed
    INVALID = "invalid"        # Invalid submission or not a real vulnerability
    DUPLICATE = "duplicate"    # Already reported vulnerability
    RESOLVED = "resolved"      # Issue has been fixed
    REWARDED = "rewarded"      # Bounty has been paid

@dataclass
class VulnerabilityReport:
    """Represents a security vulnerability report"""
    report_id: str
    reporter: str
    title: str
    description: str
    severity: VulnerabilitySeverity
    affected_components: List[str]
    steps_to_reproduce: List[str]
    submitted_at: float
    status: BountyStatus
    cvss_score: Optional[float] = None  # Common Vulnerability Scoring System score
    poc_evidence: Optional[str] = None  # Proof of concept evidence
    assigned_to: Optional[str] = None
    resolved_at: Optional[float] = None
    resolution_notes: Optional[str] = None
    reward_amount: Optional[float] = None
    reward_token: Optional[str] = None  # Token type for reward (e.g., "FLX")

@dataclass
class SecurityAdvisory:
    """Represents a security advisory for public disclosure"""
    advisory_id: str
    title: str
    description: str
    severity: VulnerabilitySeverity
    affected_versions: List[str]
    fixed_in_version: Optional[str]
    published_at: float
    disclosure_date: Optional[float]  # When it will be publicly disclosed
    cve_id: Optional[str] = None  # Common Vulnerabilities and Exposures ID
    references: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if self.references is None:
            self.references = []

@dataclass
class BountyProgram:
    """Represents a bug bounty program"""
    program_id: str
    name: str
    description: str
    start_date: float
    end_date: Optional[float]
    budget: float
    token_type: str  # "FLX" or other token
    rules: List[str]
    scope: List[str]  # Components covered by the bounty
    out_of_scope: List[str]  # Components not covered
    rewards: Dict[str, float]  # Severity -> reward amount mapping

class BugBountyManager:
    """
    Manages community-driven bug bounty and security disclosure policies
    """
    
    def __init__(self, program_key: str = "default-bounty-key"):
        self.program_key = program_key
        self.compliance_framework = ComplianceFramework()
        self.governance_system = OnChainGovernanceSystem()
        self.vulnerability_reports: Dict[str, VulnerabilityReport] = {}
        self.security_advisories: Dict[str, SecurityAdvisory] = {}
        self.bounty_programs: Dict[str, BountyProgram] = {}
        self.bounty_config = {
            "default_rewards": {
                "low": 100.0,
                "medium": 500.0,
                "high": 1000.0,
                "critical": 5000.0
            },
            "disclosure_policy": "responsible",  # "responsible", "full", "limited"
            "review_timeout": 7 * 24 * 60 * 60,  # 7 days in seconds
            "payment_timeout": 30 * 24 * 60 * 60,  # 30 days in seconds
            "anonymous_reporting": True,
            "triage_team": ["security-lead", "core-dev-1", "core-dev-2"]
        }
        self._initialize_default_program()
    
    def _initialize_default_program(self):
        """Initialize the default bug bounty program"""
        program = BountyProgram(
            program_id="bounty-program-001",
            name="Quantum Currency Bug Bounty Program",
            description="Community-driven bug bounty program for the Quantum Currency system",
            start_date=time.time(),
            end_date=None,  # Ongoing program
            budget=100000.0,  # 100,000 FLX budget
            token_type="FLX",
            rules=[
                "Do not violate privacy or destroy data",
                "Do not access accounts you don't own",
                "Do not perform social engineering attacks",
                "Report vulnerabilities privately first",
                "Give us reasonable time to fix before public disclosure"
            ],
            scope=[
                "Core consensus engine",
                "Harmonic validation system",
                "Token economy simulation",
                "Governance mechanisms",
                "Network protocols",
                "Smart contract implementations",
                "API endpoints",
                "Wallet applications"
            ],
            out_of_scope=[
                "DDoS attacks",
                "Spam attacks",
                "Physical security issues",
                "Social engineering",
                "Third-party services"
            ],
            rewards=self.bounty_config["default_rewards"]
        )
        
        self.bounty_programs[program.program_id] = program
    
    def submit_vulnerability_report(self, reporter: str, title: str, description: str,
                                  affected_components: List[str], steps_to_reproduce: List[str],
                                  severity: Optional[VulnerabilitySeverity] = None,
                                  poc_evidence: Optional[str] = None,
                                  cvss_score: Optional[float] = None) -> str:
        """
        Submit a vulnerability report
        
        Args:
            reporter: Name/ID of the reporter
            title: Title of the vulnerability
            description: Detailed description
            affected_components: List of affected system components
            steps_to_reproduce: Steps to reproduce the vulnerability
            severity: Severity level (if known)
            poc_evidence: Proof of concept evidence
            cvss_score: CVSS score (if calculated)
            
        Returns:
            Report ID
        """
        report_id = f"vuln-{int(time.time())}-{hashlib.md5(title.encode()).hexdigest()[:8]}"
        
        # If severity not provided, estimate based on description and CVSS score
        if severity is None:
            severity = self._estimate_severity(description, cvss_score)
        
        # Create vulnerability report
        report = VulnerabilityReport(
            report_id=report_id,
            reporter=reporter,
            title=title,
            description=description,
            severity=severity,
            affected_components=affected_components,
            steps_to_reproduce=steps_to_reproduce,
            submitted_at=time.time(),
            status=BountyStatus.SUBMITTED,
            cvss_score=cvss_score,
            poc_evidence=poc_evidence
        )
        
        self.vulnerability_reports[report_id] = report
        
        # Log the submission
        print(f"New vulnerability report submitted: {report_id}")
        print(f"  Title: {title}")
        print(f"  Severity: {severity.value}")
        print(f"  Reporter: {reporter}")
        
        return report_id
    
    def _estimate_severity(self, description: str, cvss_score: Optional[float]) -> VulnerabilitySeverity:
        """
        Estimate vulnerability severity based on description and CVSS score
        
        Args:
            description: Vulnerability description
            cvss_score: CVSS score if available
            
        Returns:
            Estimated severity level
        """
        # If CVSS score is provided, use it
        if cvss_score is not None:
            if cvss_score >= 9.0:
                return VulnerabilitySeverity.CRITICAL
            elif cvss_score >= 7.0:
                return VulnerabilitySeverity.HIGH
            elif cvss_score >= 4.0:
                return VulnerabilitySeverity.MEDIUM
            else:
                return VulnerabilitySeverity.LOW
        
        # Otherwise, estimate based on keywords in description
        description_lower = description.lower()
        
        # Critical keywords
        critical_keywords = ["remote code execution", "privilege escalation", "bypass authentication", 
                           "chain split", "consensus failure", "fund theft"]
        
        # High keywords
        high_keywords = ["denial of service", "memory corruption", "buffer overflow", 
                        "signature forgery", "double spend", "replay attack"]
        
        # Medium keywords
        medium_keywords = ["information disclosure", "timing attack", "weak randomness", 
                          "race condition", "insecure configuration"]
        
        # Check for critical keywords
        if any(keyword in description_lower for keyword in critical_keywords):
            return VulnerabilitySeverity.CRITICAL
        
        # Check for high keywords
        if any(keyword in description_lower for keyword in high_keywords):
            return VulnerabilitySeverity.HIGH
        
        # Check for medium keywords
        if any(keyword in description_lower for keyword in medium_keywords):
            return VulnerabilitySeverity.MEDIUM
        
        # Default to low severity
        return VulnerabilitySeverity.LOW
    
    def review_vulnerability_report(self, report_id: str, reviewer: str,
                                  is_valid: bool, notes: Optional[str] = None) -> bool:
        """
        Review a vulnerability report
        
        Args:
            report_id: ID of the report to review
            reviewer: Name/ID of the reviewer
            is_valid: Whether the report is valid
            notes: Review notes
            
        Returns:
            True if review was successful, False otherwise
        """
        if report_id not in self.vulnerability_reports:
            print(f"Report {report_id} not found")
            return False
        
        report = self.vulnerability_reports[report_id]
        
        # Check if report is already resolved or rewarded
        if report.status in [BountyStatus.RESOLVED, BountyStatus.REWARDED]:
            print(f"Report {report_id} is already resolved")
            return False
        
        # Update report status
        if is_valid:
            report.status = BountyStatus.VALID
            report.assigned_to = reviewer
            
            # Calculate reward amount based on severity
            program = list(self.bounty_programs.values())[0]  # Get first program
            reward_amount = program.rewards.get(report.severity.value, 0.0)
            report.reward_amount = reward_amount
            report.reward_token = program.token_type
            
            print(f"Report {report_id} validated. Reward: {reward_amount} {program.token_type}")
        else:
            report.status = BountyStatus.INVALID
            print(f"Report {report_id} marked as invalid")
        
        if notes:
            report.resolution_notes = notes
        
        return True
    
    def resolve_vulnerability(self, report_id: str, resolution_notes: str,
                            fixed_in_version: Optional[str] = None) -> bool:
        """
        Mark a vulnerability as resolved
        
        Args:
            report_id: ID of the report to resolve
            resolution_notes: Notes about the resolution
            fixed_in_version: Version where the fix was implemented
            
        Returns:
            True if resolution was successful, False otherwise
        """
        if report_id not in self.vulnerability_reports:
            print(f"Report {report_id} not found")
            return False
        
        report = self.vulnerability_reports[report_id]
        
        # Check if report is valid
        if report.status != BountyStatus.VALID:
            print(f"Report {report_id} is not in valid status")
            return False
        
        # Update report
        report.status = BountyStatus.RESOLVED
        report.resolved_at = time.time()
        report.resolution_notes = resolution_notes
        
        # Create security advisory if this is a public vulnerability
        if report.severity in [VulnerabilitySeverity.HIGH, VulnerabilitySeverity.CRITICAL]:
            self._create_security_advisory(report, fixed_in_version)
        
        print(f"Report {report_id} marked as resolved")
        return True
    
    def _create_security_advisory(self, report: VulnerabilityReport, 
                                fixed_in_version: Optional[str]) -> str:
        """
        Create a security advisory for a resolved critical/high severity vulnerability
        
        Args:
            report: The resolved vulnerability report
            fixed_in_version: Version where the fix was implemented
            
        Returns:
            Advisory ID
        """
        advisory_id = f"advisory-{int(time.time())}-{hashlib.md5(report.title.encode()).hexdigest()[:8]}"
        
        # Determine disclosure date based on policy
        disclosure_delay = 30 * 24 * 60 * 60  # 30 days
        if self.bounty_config["disclosure_policy"] == "responsible":
            disclosure_date = time.time() + disclosure_delay
        elif self.bounty_config["disclosure_policy"] == "limited":
            disclosure_date = time.time() + (disclosure_delay // 2)  # 15 days
        else:  # full disclosure
            disclosure_date = time.time()
        
        # Create advisory
        advisory = SecurityAdvisory(
            advisory_id=advisory_id,
            title=report.title,
            description=report.description,
            severity=report.severity,
            affected_versions=["all"],  # Simplified for demo
            fixed_in_version=fixed_in_version,
            published_at=time.time(),
            disclosure_date=disclosure_date,
            references=[f"Report ID: {report.report_id}"]
        )
        
        self.security_advisories[advisory_id] = advisory
        
        print(f"Security advisory {advisory_id} created for {report.report_id}")
        return advisory_id
    
    def pay_bounty(self, report_id: str) -> bool:
        """
        Pay bounty for a resolved vulnerability report
        
        Args:
            report_id: ID of the report to pay bounty for
            
        Returns:
            True if payment was successful, False otherwise
        """
        if report_id not in self.vulnerability_reports:
            print(f"Report {report_id} not found")
            return False
        
        report = self.vulnerability_reports[report_id]
        
        # Check if report is resolved
        if report.status != BountyStatus.RESOLVED:
            print(f"Report {report_id} is not resolved yet")
            return False
        
        # Check if already paid
        if report.status == BountyStatus.REWARDED:
            print(f"Report {report_id} bounty already paid")
            return False
        
        # Check if payment is past due
        if report.resolved_at and (time.time() - report.resolved_at > self.bounty_config["payment_timeout"]):
            print(f"Report {report_id} payment is past due")
            return False
        
        # Pay the bounty (in a real implementation, this would transfer tokens)
        report.status = BountyStatus.REWARDED
        print(f"Bounty of {report.reward_amount} {report.reward_token} paid for report {report_id}")
        
        return True
    
    def get_active_reports(self) -> List[VulnerabilityReport]:
        """
        Get list of active (non-resolved) vulnerability reports
        
        Returns:
            List of active vulnerability reports
        """
        return [
            report for report in self.vulnerability_reports.values()
            if report.status not in [BountyStatus.RESOLVED, BountyStatus.REWARDED, BountyStatus.INVALID]
        ]
    
    def get_resolved_reports(self) -> List[VulnerabilityReport]:
        """
        Get list of resolved vulnerability reports
        
        Returns:
            List of resolved vulnerability reports
        """
        return [
            report for report in self.vulnerability_reports.values()
            if report.status in [BountyStatus.RESOLVED, BountyStatus.REWARDED]
        ]
    
    def get_program_stats(self) -> Dict:
        """
        Get bug bounty program statistics
        
        Returns:
            Dictionary with program statistics
        """
        total_reports = len(self.vulnerability_reports)
        valid_reports = len([
            report for report in self.vulnerability_reports.values()
            if report.status == BountyStatus.VALID
        ])
        resolved_reports = len([
            report for report in self.vulnerability_reports.values()
            if report.status in [BountyStatus.RESOLVED, BountyStatus.REWARDED]
        ])
        rewarded_reports = len([
            report for report in self.vulnerability_reports.values()
            if report.status == BountyStatus.REWARDED
        ])
        
        # Calculate total rewards paid
        total_rewards = sum([
            report.reward_amount or 0.0
            for report in self.vulnerability_reports.values()
            if report.status == BountyStatus.REWARDED
        ])
        
        # Get severity distribution
        severity_counts = {}
        for report in self.vulnerability_reports.values():
            severity = report.severity.value
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        return {
            "total_reports": total_reports,
            "valid_reports": valid_reports,
            "resolved_reports": resolved_reports,
            "rewarded_reports": rewarded_reports,
            "total_rewards_paid": total_rewards,
            "severity_distribution": severity_counts,
            "active_reports": len(self.get_active_reports())
        }
    
    def generate_security_report(self) -> Dict:
        """
        Generate a comprehensive security report
        
        Returns:
            Dictionary with security report
        """
        stats = self.get_program_stats()
        
        # Calculate security health score
        # Base score on resolved reports vs total reports
        if stats["total_reports"] > 0:
            resolution_rate = stats["resolved_reports"] / stats["total_reports"]
        else:
            resolution_rate = 1.0
        
        # Adjust for severity distribution
        critical_count = stats["severity_distribution"].get("critical", 0)
        high_count = stats["severity_distribution"].get("high", 0)
        
        # Higher score if fewer critical/high vulnerabilities
        severity_penalty = (critical_count * 0.2 + high_count * 0.1)
        
        security_score = max(0.0, min(1.0, resolution_rate - severity_penalty))
        
        health_status = "excellent" if security_score > 0.9 else \
                       "good" if security_score > 0.7 else \
                       "fair" if security_score > 0.5 else "poor"
        
        return {
            "report_timestamp": time.time(),
            "security_score": security_score,
            "health_status": health_status,
            "stats": stats,
            "active_reports": [
                {
                    "id": report.report_id,
                    "title": report.title,
                    "severity": report.severity.value,
                    "submitted_at": report.submitted_at,
                    "status": report.status.value
                }
                for report in self.get_active_reports()
            ],
            "recent_advisories": [
                {
                    "id": advisory.advisory_id,
                    "title": advisory.title,
                    "severity": advisory.severity.value,
                    "published_at": advisory.published_at
                }
                for advisory in list(self.security_advisories.values())[-5:]  # Last 5 advisories
            ]
        }

def demo_bug_bounty_system():
    """Demonstrate bug bounty and security disclosure system"""
    print("🛡️  Bug Bounty and Security Disclosure Demo")
    print("=" * 45)
    
    # Create bug bounty manager
    bounty_manager = BugBountyManager("quantum-bounty-key")
    
    # Show program details
    print("\n📋 Bug Bounty Program:")
    program = list(bounty_manager.bounty_programs.values())[0]
    print(f"   Name: {program.name}")
    print(f"   Budget: {program.budget} {program.token_type}")
    print(f"   Scope: {len(program.scope)} components covered")
    print(f"   Rules: {len(program.rules)} rules")
    
    # Show reward structure
    print("\n💰 Reward Structure:")
    for severity, reward in program.rewards.items():
        print(f"   {severity.capitalize()}: {reward} {program.token_type}")
    
    # Submit vulnerability reports
    print("\n📥 Submitting Vulnerability Reports:")
    
    # Critical vulnerability
    critical_report_id = bounty_manager.submit_vulnerability_report(
        reporter="community-member-001",
        title="Remote Code Execution in Consensus Engine",
        description="A vulnerability in the consensus engine allows remote code execution through crafted harmonic snapshots.",
        affected_components=["Core consensus engine", "Harmonic validation system"],
        steps_to_reproduce=[
            "1. Generate malicious harmonic snapshot with embedded shellcode",
            "2. Submit snapshot to validator node",
            "3. Shellcode executes with validator privileges"
        ],
        severity=VulnerabilitySeverity.CRITICAL,
        poc_evidence="POC code available upon request",
        cvss_score=9.8
    )
    print(f"   Critical vulnerability report: {critical_report_id}")
    
    # High vulnerability
    high_report_id = bounty_manager.submit_vulnerability_report(
        reporter="security-researcher-001",
        title="Denial of Service in Network Protocol",
        description="Malformed network packets can cause validator nodes to crash.",
        affected_components=["Network protocols", "Validator nodes"],
        steps_to_reproduce=[
            "1. Connect to validator node",
            "2. Send specially crafted packet with invalid harmonic data",
            "3. Node crashes with out-of-bounds memory access"
        ],
        severity=VulnerabilitySeverity.HIGH,
        cvss_score=7.5
    )
    print(f"   High vulnerability report: {high_report_id}")
    
    # Medium vulnerability
    medium_report_id = bounty_manager.submit_vulnerability_report(
        reporter="community-member-002",
        title="Information Disclosure in API",
        description="API endpoint exposes internal system information to unauthorized users.",
        affected_components=["API endpoints", "REST API layer"],
        steps_to_reproduce=[
            "1. Send request to /api/debug endpoint without authentication",
            "2. Receive detailed system information including node versions"
        ],
        severity=VulnerabilitySeverity.MEDIUM,
        cvss_score=5.3
    )
    print(f"   Medium vulnerability report: {medium_report_id}")
    
    # Low vulnerability
    low_report_id = bounty_manager.submit_vulnerability_report(
        reporter="new-contributor-001",
        title="Weak Randomness in Token Generation",
        description="Token generation uses predictable random number generator.",
        affected_components=["Token economy simulation"],
        steps_to_reproduce=[
            "1. Observe multiple token generation events",
            "2. Predict next token values based on previous patterns"
        ],
        severity=VulnerabilitySeverity.LOW,
        cvss_score=2.1
    )
    print(f"   Low vulnerability report: {low_report_id}")
    
    # Show active reports
    print("\n🔍 Active Reports:")
    active_reports = bounty_manager.get_active_reports()
    for report in active_reports:
        print(f"   {report.report_id}: {report.title}")
        print(f"      Severity: {report.severity.value}")
        print(f"      Status: {report.status.value}")
        print(f"      Reporter: {report.reporter}")
    
    # Review reports
    print("\n✅ Reviewing Reports:")
    
    # Review critical report as valid
    if bounty_manager.review_vulnerability_report(
        critical_report_id, 
        "security-lead", 
        True, 
        "Confirmed RCE vulnerability. Immediate patch required."
    ):
        critical_report = bounty_manager.vulnerability_reports[critical_report_id]
        print(f"   {critical_report_id}: VALID - Reward: {critical_report.reward_amount} {critical_report.reward_token}")
    
    # Review high report as valid
    if bounty_manager.review_vulnerability_report(
        high_report_id, 
        "core-dev-1", 
        True, 
        "Confirmed DoS vulnerability. Patch in progress."
    ):
        high_report = bounty_manager.vulnerability_reports[high_report_id]
        print(f"   {high_report_id}: VALID - Reward: {high_report.reward_amount} {high_report.reward_token}")
    
    # Review medium report as valid
    if bounty_manager.review_vulnerability_report(
        medium_report_id, 
        "core-dev-2", 
        True, 
        "Confirmed information disclosure. Fix ready for deployment."
    ):
        medium_report = bounty_manager.vulnerability_reports[medium_report_id]
        print(f"   {medium_report_id}: VALID - Reward: {medium_report.reward_amount} {medium_report.reward_token}")
    
    # Review low report as invalid (duplicate)
    if bounty_manager.review_vulnerability_report(
        low_report_id, 
        "security-lead", 
        False, 
        "Duplicate of previously reported issue #SEC-1234"
    ):
        print(f"   {low_report_id}: INVALID - Duplicate report")
    
    # Resolve vulnerabilities
    print("\n🔧 Resolving Vulnerabilities:")
    
    # Resolve critical vulnerability
    if bounty_manager.resolve_vulnerability(
        critical_report_id,
        "Fixed RCE vulnerability by implementing proper input validation and sandboxing.",
        "v2.1.0"
    ):
        print(f"   {critical_report_id}: RESOLVED in v2.1.0")
    
    # Resolve high vulnerability
    if bounty_manager.resolve_vulnerability(
        high_report_id,
        "Fixed DoS vulnerability by adding packet size limits and validation.",
        "v2.1.0"
    ):
        print(f"   {high_report_id}: RESOLVED in v2.1.0")
    
    # Resolve medium vulnerability
    if bounty_manager.resolve_vulnerability(
        medium_report_id,
        "Fixed information disclosure by removing debug endpoint from production builds.",
        "v2.0.1"
    ):
        print(f"   {medium_report_id}: RESOLVED in v2.0.1")
    
    # Pay bounties
    print("\n💸 Paying Bounties:")
    
    # Pay critical vulnerability bounty
    if bounty_manager.pay_bounty(critical_report_id):
        critical_report = bounty_manager.vulnerability_reports[critical_report_id]
        print(f"   {critical_report_id}: Paid {critical_report.reward_amount} {critical_report.reward_token}")
    
    # Pay high vulnerability bounty
    if bounty_manager.pay_bounty(high_report_id):
        high_report = bounty_manager.vulnerability_reports[high_report_id]
        print(f"   {high_report_id}: Paid {high_report.reward_amount} {high_report.reward_token}")
    
    # Pay medium vulnerability bounty
    if bounty_manager.pay_bounty(medium_report_id):
        medium_report = bounty_manager.vulnerability_reports[medium_report_id]
        print(f"   {medium_report_id}: Paid {medium_report.reward_amount} {medium_report.reward_token}")
    
    # Show program statistics
    print("\n📊 Program Statistics:")
    stats = bounty_manager.get_program_stats()
    print(f"   Total reports: {stats['total_reports']}")
    print(f"   Valid reports: {stats['valid_reports']}")
    print(f"   Resolved reports: {stats['resolved_reports']}")
    print(f"   Rewarded reports: {stats['rewarded_reports']}")
    print(f"   Total rewards paid: {stats['total_rewards_paid']} FLX")
    
    print("\n📈 Severity Distribution:")
    for severity, count in stats['severity_distribution'].items():
        print(f"   {severity.capitalize()}: {count}")
    
    # Generate security report
    print("\n🛡️  Security Report:")
    security_report = bounty_manager.generate_security_report()
    
    print(f"   Security score: {security_report['security_score']:.2f}")
    print(f"   Health status: {security_report['health_status']}")
    print(f"   Active reports: {len(security_report['active_reports'])}")
    print(f"   Recent advisories: {len(security_report['recent_advisories'])}")
    
    # Show recent advisories
    if security_report['recent_advisories']:
        print("\n📢 Recent Security Advisories:")
        for advisory in security_report['recent_advisories']:
            print(f"   {advisory['title']} ({advisory['severity']})")
    
    print("\n✅ Bug bounty and security disclosure demo completed!")

if __name__ == "__main__":
    demo_bug_bounty_system()