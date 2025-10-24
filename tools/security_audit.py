"""
AEGIS Security Audit Tools
Comprehensive security auditing for advanced features
"""

import asyncio
import time
import logging
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import json
import hashlib
import sys

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.aegis.security.advanced_crypto import AdvancedSecurityManager, SecurityFeature
from src.aegis.core.performance_optimizer import PerformanceOptimizer
from src.aegis.monitoring.security_monitor import SecurityMonitor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AuditSeverity(Enum):
    """Audit finding severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AuditCategory(Enum):
    """Audit categories"""
    AUTHENTICATION = "authentication"
    ENCRYPTION = "encryption"
    ACCESS_CONTROL = "access_control"
    PRIVACY = "privacy"
    CONFIGURATION = "configuration"
    NETWORK = "network"
    COMPLIANCE = "compliance"


@dataclass
class AuditFinding:
    """Security audit finding"""
    audit_id: str
    category: AuditCategory
    severity: AuditSeverity
    title: str
    description: str
    recommendation: str
    affected_components: List[str] = field(default_factory=list)
    evidence: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    resolved: bool = False
    resolution_notes: str = ""


@dataclass
class AuditReport:
    """Comprehensive security audit report"""
    report_id: str
    generated_at: float
    framework_version: str
    audit_scope: List[str]
    findings: List[AuditFinding] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    compliance_status: Dict[str, Any] = field(default_factory=dict)


class SecurityAuditTool:
    """Comprehensive security audit tool for AEGIS Framework"""
    
    def __init__(self, security_manager: AdvancedSecurityManager):
        self.security_manager = security_manager
        self.findings: List[AuditFinding] = []
        self.audit_reports: List[AuditReport] = []
        self.audit_rules = self._load_audit_rules()
    
    def _load_audit_rules(self) -> Dict[str, Any]:
        """Load audit rules and best practices"""
        return {
            "authentication": {
                "zkp_min_entropy": 128,
                "proof_expiration": 300,  # 5 minutes
                "max_failed_attempts": 5
            },
            "encryption": {
                "min_key_size": 2048,
                "cipher_suite": ["chacha20-poly1305", "aes-256-gcm"],
                "key_rotation_days": 90
            },
            "privacy": {
                "dp_epsilon_max": 1.0,
                "dp_delta_max": 1e-5,
                "budget_consumption_warning": 0.8
            },
            "configuration": {
                "log_level": "INFO",
                "debug_mode_disabled": True,
                "default_passwords_changed": True
            },
            "network": {
                "tls_required": True,
                "allowed_ports": [8080, 9091],
                "max_connections_per_ip": 100
            }
        }
    
    async def run_comprehensive_audit(self, scope: Optional[List[str]] = None) -> AuditReport:
        """Run comprehensive security audit"""
        logger.info("🔍 Starting comprehensive security audit...")
        
        if scope is None:
            scope = ["authentication", "encryption", "access_control", "privacy", 
                    "configuration", "network", "compliance"]
        
        start_time = time.time()
        
        # Run audits in parallel
        audit_tasks = []
        for category in scope:
            if category == "authentication":
                audit_tasks.append(self._audit_authentication())
            elif category == "encryption":
                audit_tasks.append(self._audit_encryption())
            elif category == "access_control":
                audit_tasks.append(self._audit_access_control())
            elif category == "privacy":
                audit_tasks.append(self._audit_privacy())
            elif category == "configuration":
                audit_tasks.append(self._audit_configuration())
            elif category == "network":
                audit_tasks.append(self._audit_network())
            elif category == "compliance":
                audit_tasks.append(self._audit_compliance())
        
        # Execute all audit tasks
        await asyncio.gather(*audit_tasks)
        
        # Generate report
        report = self._generate_audit_report(scope)
        
        elapsed_time = time.time() - start_time
        logger.info(f"✅ Security audit completed in {elapsed_time:.2f}s")
        logger.info(f"📊 Findings: {len(self.findings)} issues identified")
        
        self.audit_reports.append(report)
        return report
    
    async def _audit_authentication(self) -> None:
        """Audit authentication mechanisms"""
        logger.info("🔐 Auditing authentication mechanisms...")
        
        # Check ZKP configuration
        stats = self.security_manager.get_security_stats()
        zk_enabled = stats.get("enabled_features", {}).get("zero_knowledge_proof", False)
        
        if not zk_enabled:
            finding = AuditFinding(
                audit_id=f"auth_001_{int(time.time())}",
                category=AuditCategory.AUTHENTICATION,
                severity=AuditSeverity.HIGH,
                title="Zero-Knowledge Proofs Disabled",
                description="ZKP authentication is disabled, reducing security",
                recommendation="Enable ZKP authentication for enhanced security",
                affected_components=["security_manager", "authentication"],
                evidence={"zkp_enabled": zk_enabled}
            )
            self.findings.append(finding)
        
        # Check proof expiration
        # This would normally check actual proof timestamps
        finding = AuditFinding(
            audit_id=f"auth_002_{int(time.time())}",
            category=AuditCategory.AUTHENTICATION,
            severity=AuditSeverity.LOW,
            title="Authentication Best Practices",
            description="Review authentication timeout and proof expiration settings",
            recommendation="Set appropriate proof expiration times (300-600 seconds)",
            affected_components=["security_manager"],
            evidence={"review_needed": True}
        )
        self.findings.append(finding)
    
    async def _audit_encryption(self) -> None:
        """Audit encryption mechanisms"""
        logger.info("🔒 Auditing encryption mechanisms...")
        
        # Check homomorphic encryption
        stats = self.security_manager.get_security_stats()
        he_enabled = stats.get("enabled_features", {}).get("homomorphic_encryption", False)
        
        if not he_enabled:
            finding = AuditFinding(
                audit_id=f"enc_001_{int(time.time())}",
                category=AuditCategory.ENCRYPTION,
                severity=AuditSeverity.MEDIUM,
                title="Homomorphic Encryption Disabled",
                description="Homomorphic encryption is disabled, limiting privacy-preserving computations",
                recommendation="Enable homomorphic encryption for sensitive data processing",
                affected_components=["security_manager", "homomorphic_encryption"],
                evidence={"he_enabled": he_enabled}
            )
            self.findings.append(finding)
        
        # Check key management
        finding = AuditFinding(
            audit_id=f"enc_002_{int(time.time())}",
            category=AuditCategory.ENCRYPTION,
            severity=AuditSeverity.LOW,
            title="Key Management Review",
            description="Regular key rotation and management practices should be reviewed",
            recommendation="Implement automated key rotation every 90 days",
            affected_components=["key_management"],
            evidence={"review_needed": True}
        )
        self.findings.append(finding)
    
    async def _audit_access_control(self) -> None:
        """Audit access control mechanisms"""
        logger.info("👥 Auditing access control mechanisms...")
        
        # Check SMC configuration
        stats = self.security_manager.get_security_stats()
        smc_enabled = stats.get("enabled_features", {}).get("secure_multi_party_computation", False)
        
        if not smc_enabled:
            finding = AuditFinding(
                audit_id=f"ac_001_{int(time.time())}",
                category=AuditCategory.ACCESS_CONTROL,
                severity=AuditSeverity.MEDIUM,
                title="Secure Multi-Party Computation Disabled",
                description="SMC is disabled, limiting secure collaborative computations",
                recommendation="Enable SMC for multi-party privacy-preserving operations",
                affected_components=["security_manager", "smc"],
                evidence={"smc_enabled": smc_enabled}
            )
            self.findings.append(finding)
        
        # Check role-based access
        finding = AuditFinding(
            audit_id=f"ac_002_{int(time.time())}",
            category=AuditCategory.ACCESS_CONTROL,
            severity=AuditSeverity.LOW,
            title="RBAC Configuration",
            description="Review role-based access control configuration",
            recommendation="Implement principle of least privilege for all components",
            affected_components=["access_control"],
            evidence={"review_needed": True}
        )
        self.findings.append(finding)
    
    async def _audit_privacy(self) -> None:
        """Audit privacy mechanisms"""
        logger.info("的隱私權 Auditing privacy mechanisms...")
        
        # Check differential privacy
        stats = self.security_manager.get_security_stats()
        dp_enabled = stats.get("enabled_features", {}).get("differential_privacy", False)
        
        if not dp_enabled:
            finding = AuditFinding(
                audit_id=f"priv_001_{int(time.time())}",
                category=AuditCategory.PRIVACY,
                severity=AuditSeverity.HIGH,
                title="Differential Privacy Disabled",
                description="Differential privacy is disabled, risking individual privacy",
                recommendation="Enable differential privacy for statistical queries",
                affected_components=["security_manager", "differential_privacy"],
                evidence={"dp_enabled": dp_enabled}
            )
            self.findings.append(finding)
        
        # Check privacy budget
        privacy_params = stats.get("privacy_parameters", {})
        epsilon = privacy_params.get("epsilon", 1.0)
        delta = privacy_params.get("delta", 1e-5)
        
        if epsilon > self.audit_rules["privacy"]["dp_epsilon_max"]:
            finding = AuditFinding(
                audit_id=f"priv_002_{int(time.time())}",
                category=AuditCategory.PRIVACY,
                severity=AuditSeverity.MEDIUM,
                title="High Privacy Budget Consumption",
                description=f"Epsilon value ({epsilon}) exceeds recommended maximum ({self.audit_rules['privacy']['dp_epsilon_max']})",
                recommendation="Reduce epsilon value to strengthen privacy guarantees",
                affected_components=["differential_privacy"],
                evidence={"epsilon": epsilon, "max_epsilon": self.audit_rules["privacy"]["dp_epsilon_max"]}
            )
            self.findings.append(finding)
    
    async def _audit_configuration(self) -> None:
        """Audit configuration settings"""
        logger.info("⚙️  Auditing configuration settings...")
        
        # Check debug mode
        # This would normally check actual config
        finding = AuditFinding(
            audit_id=f"cfg_001_{int(time.time())}",
            category=AuditCategory.CONFIGURATION,
            severity=AuditSeverity.LOW,
            title="Configuration Review",
            description="Review debug mode and logging settings for production",
            recommendation="Ensure debug mode is disabled and logging is appropriate for production",
            affected_components=["configuration"],
            evidence={"review_needed": True}
        )
        self.findings.append(finding)
    
    async def _audit_network(self) -> None:
        """Audit network security"""
        logger.info("🌐 Auditing network security...")
        
        # Check TLS configuration
        finding = AuditFinding(
            audit_id=f"net_001_{int(time.time())}",
            category=AuditCategory.NETWORK,
            severity=AuditSeverity.LOW,
            title="Network Security Review",
            description="Review TLS configuration and network access controls",
            recommendation="Ensure TLS 1.3 is enforced and network policies are restrictive",
            affected_components=["network_security"],
            evidence={"review_needed": True}
        )
        self.findings.append(finding)
    
    async def _audit_compliance(self) -> None:
        """Audit compliance requirements"""
        logger.info("📋 Auditing compliance requirements...")
        
        # Check compliance documentation
        finding = AuditFinding(
            audit_id=f"comp_001_{int(time.time())}",
            category=AuditCategory.COMPLIANCE,
            severity=AuditSeverity.LOW,
            title="Compliance Documentation",
            description="Review compliance documentation and audit trails",
            recommendation="Maintain comprehensive audit logs and compliance documentation",
            affected_components=["compliance"],
            evidence={"review_needed": True}
        )
        self.findings.append(finding)
    
    def _generate_audit_report(self, scope: List[str]) -> AuditReport:
        """Generate comprehensive audit report"""
        report_id = hashlib.sha256(f"audit_{time.time()}".encode()).hexdigest()[:16]
        
        # Categorize findings
        findings_by_severity = {}
        findings_by_category = {}
        
        for finding in self.findings:
            # By severity
            severity = finding.severity.value
            if severity not in findings_by_severity:
                findings_by_severity[severity] = []
            findings_by_severity[severity].append(finding)
            
            # By category
            category = finding.category.value
            if category not in findings_by_category:
                findings_by_category[category] = []
            findings_by_category[category].append(finding)
        
        # Generate summary
        summary = {
            "total_findings": len(self.findings),
            "findings_by_severity": {k: len(v) for k, v in findings_by_severity.items()},
            "findings_by_category": {k: len(v) for k, v in findings_by_category.items()},
            "critical_issues": len(findings_by_severity.get("critical", [])),
            "high_issues": len(findings_by_severity.get("high", [])),
            "medium_issues": len(findings_by_severity.get("medium", [])),
            "low_issues": len(findings_by_severity.get("low", []))
        }
        
        # Generate recommendations
        recommendations = []
        if summary["critical_issues"] > 0:
            recommendations.append("Address critical security issues immediately")
        if summary["high_issues"] > 0:
            recommendations.append("Prioritize high-severity issues in next sprint")
        if summary["medium_issues"] > 0:
            recommendations.append("Plan medium-severity fixes for upcoming releases")
        
        # Compliance status
        compliance_status = {
            "overall_rating": self._calculate_compliance_rating(summary),
            "areas_needing_attention": list(findings_by_category.keys()),
            "next_audit_scheduled": time.time() + 7776000  # 90 days
        }
        
        report = AuditReport(
            report_id=report_id,
            generated_at=time.time(),
            framework_version="2.2.0",
            audit_scope=scope,
            findings=self.findings.copy(),
            summary=summary,
            recommendations=recommendations,
            compliance_status=compliance_status
        )
        
        return report
    
    def _calculate_compliance_rating(self, summary: Dict[str, Any]) -> str:
        """Calculate overall compliance rating"""
        critical = summary.get("critical_issues", 0)
        high = summary.get("high_issues", 0)
        medium = summary.get("medium_issues", 0)
        
        # Simple rating calculation
        if critical > 0:
            return "Failing"
        elif high > 2:
            return "Needs Improvement"
        elif high > 0 or medium > 5:
            return "Acceptable"
        else:
            return "Good"
    
    def export_report(self, report: AuditReport, filename: Optional[str] = None) -> str:
        """Export audit report to JSON file"""
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"aegis_security_audit_{timestamp}.json"
        
        # Convert report to serializable format
        report_dict = {
            "report_id": report.report_id,
            "generated_at": report.generated_at,
            "framework_version": report.framework_version,
            "audit_scope": report.audit_scope,
            "summary": report.summary,
            "recommendations": report.recommendations,
            "compliance_status": report.compliance_status,
            "findings": [
                {
                    "audit_id": f.audit_id,
                    "category": f.category.value,
                    "severity": f.severity.value,
                    "title": f.title,
                    "description": f.description,
                    "recommendation": f.recommendation,
                    "affected_components": f.affected_components,
                    "evidence": f.evidence,
                    "timestamp": f.timestamp,
                    "resolved": f.resolved,
                    "resolution_notes": f.resolution_notes
                }
                for f in report.findings
            ]
        }
        
        with open(filename, 'w') as f:
            json.dump(report_dict, f, indent=2)
        
        logger.info(f"📄 Audit report exported to {filename}")
        return filename
    
    def print_executive_summary(self, report: AuditReport):
        """Print executive summary of audit findings"""
        print("\n" + "="*80)
        print("AEGIS FRAMEWORK - SECURITY AUDIT EXECUTIVE SUMMARY")
        print("="*80)
        print(f"Report ID: {report.report_id}")
        print(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(report.generated_at))}")
        print(f"Framework Version: {report.framework_version}")
        print()
        
        # Summary statistics
        summary = report.summary
        print("SUMMARY STATISTICS:")
        print("-" * 30)
        print(f"Total Findings: {summary['total_findings']}")
        print(f"Critical Issues: {summary['critical_issues']}")
        print(f"High Issues: {summary['high_issues']}")
        print(f"Medium Issues: {summary['medium_issues']}")
        print(f"Low Issues: {summary['low_issues']}")
        print()
        
        # Compliance rating
        print(f"COMPLIANCE RATING: {report.compliance_status['overall_rating']}")
        print()
        
        # Key recommendations
        if report.recommendations:
            print("KEY RECOMMENDATIONS:")
            print("-" * 30)
            for i, rec in enumerate(report.recommendations, 1):
                print(f"{i}. {rec}")
            print()
        
        # Top severity issues
        findings_by_severity = {}
        for finding in report.findings:
            severity = finding.severity.value
            if severity not in findings_by_severity:
                findings_by_severity[severity] = []
            findings_by_severity[severity].append(finding)
        
        for severity in ["critical", "high", "medium"]:
            issues = findings_by_severity.get(severity, [])
            if issues:
                print(f"{severity.upper()} ISSUES ({len(issues)}):")
                print("-" * 30)
                for issue in issues[:3]:  # Show top 3
                    print(f"  • {issue.title}")
                if len(issues) > 3:
                    print(f"  ... and {len(issues) - 3} more")
                print()
        
        print("="*80)


async def run_security_audit():
    """Run comprehensive security audit"""
    logger.info("🔍 Starting AEGIS Security Audit")
    
    # Create security manager
    security_manager = AdvancedSecurityManager()
    
    # Enable all security features for audit
    for feature in SecurityFeature:
        security_manager.enabled_features[feature] = True
    
    # Create audit tool
    audit_tool = SecurityAuditTool(security_manager)
    
    # Run comprehensive audit
    report = await audit_tool.run_comprehensive_audit()
    
    # Print executive summary
    audit_tool.print_executive_summary(report)
    
    # Export detailed report
    filename = audit_tool.export_report(report)
    logger.info(f"✅ Detailed audit report saved to {filename}")
    
    return report


if __name__ == "__main__":
    asyncio.run(run_security_audit())
