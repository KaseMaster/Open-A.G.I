#!/usr/bin/env python3
"""
🔍 AUDITORÍAS DE SEGURIDAD EXHAUSTIVAS - AEGIS Framework
Módulo completo de auditoría de seguridad para validación de vulnerabilidades,
análisis de riesgos y cumplimiento normativo.

Características principales:
- Análisis estático de código para vulnerabilidades
- Análisis dinámico de seguridad en tiempo de ejecución
- Pruebas de penetración simuladas
- Validación de cumplimiento SOC 2
- Análisis de riesgos y amenazas
- Monitoreo continuo de seguridad
- Generación de reportes de auditoría
"""

import asyncio
import time
import hashlib
import hmac
import secrets
import json
import logging
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import re
import ast
import inspect
import threading
from concurrent.futures import ThreadPoolExecutor
import os

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AuditSeverity(Enum):
    """Severidad de hallazgos de auditoría"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

class AuditCategory(Enum):
    """Categorías de auditoría de seguridad"""
    CRYPTOGRAPHY = "cryptography"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    INPUT_VALIDATION = "input_validation"
    SESSION_MANAGEMENT = "session_management"
    ERROR_HANDLING = "error_handling"
    LOGGING = "logging"
    CONFIGURATION = "configuration"
    DEPENDENCIES = "dependencies"
    NETWORK_SECURITY = "network_security"

class ComplianceStandard(Enum):
    """Estándares de cumplimiento"""
    SOC_2 = "soc_2"
    ISO_27001 = "iso_27001"
    NIST = "nist"
    GDPR = "gdpr"
    HIPAA = "hipaa"

@dataclass
class SecurityFinding:
    """Hallazgo de seguridad identificado"""
    finding_id: str
    title: str
    description: str
    severity: AuditSeverity
    category: AuditCategory
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    code_snippet: Optional[str] = None
    cwe_id: Optional[str] = None  # Common Weakness Enumeration
    remediation: str = ""
    evidence: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    status: str = "open"  # open, fixed, accepted, false_positive

@dataclass
class AuditReport:
    """Reporte completo de auditoría"""
    audit_id: str
    target_system: str
    audit_type: str
    start_time: datetime
    end_time: Optional[datetime] = None
    findings: List[SecurityFinding] = field(default_factory=list)
    compliance_score: Dict[str, float] = field(default_factory=dict)
    risk_assessment: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    executive_summary: str = ""

class StaticCodeAnalyzer:
    """Analizador estático de código para vulnerabilidades"""

    def __init__(self):
        self.vulnerability_patterns = {
            # SQL Injection
            "sql_injection": [
                re.compile(r"execute\s*\(\s*.*\+.*\)"),
                re.compile(r"cursor\.execute\s*\(\s*.*%.*\)"),
                re.compile(r"SELECT.*WHERE.*\+.*"),
            ],
            # XSS
            "xss": [
                re.compile(r"innerHTML\s*=.*\+.*"),
                re.compile(r"document\.write\s*\(.*\+.*\)"),
            ],
            # Hardcoded secrets
            "hardcoded_secrets": [
                re.compile(r"(?i)(password|secret|key)\s*=\s*['\"][^'\"]*['\"]"),
                re.compile(r"api_key\s*=\s*['\"][^'\"]*['\"]"),
            ],
            # Weak crypto
            "weak_crypto": [
                re.compile(r"MD5\s*\("),
                re.compile(r"SHA-1\s*\("),
                re.compile(r"DES\s*\("),
            ],
            # Command injection
            "command_injection": [
                re.compile(r"os\.system\s*\(.*\+.*\)"),
                re.compile(r"subprocess\.call\s*\(.*\+.*\)"),
            ],
        }

    async def analyze_file(self, file_path: str) -> List[SecurityFinding]:
        """Analiza un archivo en busca de vulnerabilidades"""
        findings = []

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')

            # Análisis basado en patrones
            for vuln_type, patterns in self.vulnerability_patterns.items():
                for pattern in patterns:
                    for match in pattern.finditer(content):
                        line_number = content[:match.start()].count('\n') + 1
                        code_snippet = lines[line_number - 1].strip() if line_number <= len(lines) else ""

                        finding = SecurityFinding(
                            finding_id=f"static_{vuln_type}_{hashlib.md5(f'{file_path}:{line_number}'.encode()).hexdigest()[:8]}",
                            title=f"Potencial {vuln_type.replace('_', ' ').title()}",
                            description=f"Patrón sospechoso detectado que podría indicar {vuln_type}",
                            severity=self._get_severity_for_vuln(vuln_type),
                            category=self._get_category_for_vuln(vuln_type),
                            file_path=file_path,
                            line_number=line_number,
                            code_snippet=code_snippet,
                            remediation=self._get_remediation_for_vuln(vuln_type)
                        )
                        findings.append(finding)

            # Análisis de imports peligrosos
            dangerous_imports = ["pickle", "eval", "exec", "yaml.unsafe_load"]
            for import_name in dangerous_imports:
                if f"import {import_name}" in content or f"from {import_name}" in content:
                    finding = SecurityFinding(
                        finding_id=f"import_{import_name}_{hashlib.md5(file_path.encode()).hexdigest()[:8]}",
                        title=f"Uso de import peligroso: {import_name}",
                        description=f"El módulo {import_name} puede ser inseguro si no se usa correctamente",
                        severity=AuditSeverity.MEDIUM,
                        category=AuditCategory.INPUT_VALIDATION,
                        file_path=file_path,
                        remediation=f"Revisar el uso de {import_name} y considerar alternativas seguras"
                    )
                    findings.append(finding)

        except Exception as e:
            logger.error(f"Error analizando {file_path}: {e}")

        return findings

    def _get_severity_for_vuln(self, vuln_type: str) -> AuditSeverity:
        severity_map = {
            "sql_injection": AuditSeverity.CRITICAL,
            "command_injection": AuditSeverity.CRITICAL,
            "hardcoded_secrets": AuditSeverity.HIGH,
            "weak_crypto": AuditSeverity.HIGH,
            "xss": AuditSeverity.MEDIUM,
        }
        return severity_map.get(vuln_type, AuditSeverity.MEDIUM)

    def _get_category_for_vuln(self, vuln_type: str) -> AuditCategory:
        category_map = {
            "sql_injection": AuditCategory.INPUT_VALIDATION,
            "xss": AuditCategory.INPUT_VALIDATION,
            "command_injection": AuditCategory.INPUT_VALIDATION,
            "hardcoded_secrets": AuditCategory.CONFIGURATION,
            "weak_crypto": AuditCategory.CRYPTOGRAPHY,
        }
        return category_map.get(vuln_type, AuditCategory.CONFIGURATION)

    def _get_remediation_for_vuln(self, vuln_type: str) -> str:
        remediation_map = {
            "sql_injection": "Usar consultas parametrizadas o ORM seguro",
            "command_injection": "Usar subprocess con lista de argumentos en lugar de strings",
            "hardcoded_secrets": "Mover credenciales a variables de entorno o sistema de gestión de secrets",
            "weak_crypto": "Usar algoritmos modernos como SHA-256, AES-GCM",
            "xss": "Implementar sanitización de entrada y Content Security Policy",
        }
        return remediation_map.get(vuln_type, "Revisar y corregir el código inseguro")

class DynamicSecurityAnalyzer:
    """Analizador dinámico de seguridad en tiempo de ejecución"""

    def __init__(self):
        self.runtime_findings: List[SecurityFinding] = []
        self.monitoring_active = False
        self._lock = threading.Lock()

    async def start_monitoring(self):
        """Inicia monitoreo dinámico de seguridad"""
        self.monitoring_active = True
        logger.info("🔍 Iniciando monitoreo dinámico de seguridad")

        # Monitorear en thread separado
        def monitor_thread():
            while self.monitoring_active:
                asyncio.run(self._check_runtime_security())
                time.sleep(5)  # Chequear cada 5 segundos

        thread = threading.Thread(target=monitor_thread, daemon=True)
        thread.start()

    def stop_monitoring(self):
        """Detiene el monitoreo dinámico"""
        self.monitoring_active = False
        logger.info("🛑 Monitoreo dinámico detenido")

    async def _check_runtime_security(self):
        """Verificaciones de seguridad en tiempo real"""
        # Verificar memoria para data leakage
        # Verificar conexiones de red activas
        # Verificar uso de CPU/memoria anómalo
        # Simular algunas verificaciones

        # Verificación de entropía
        entropy_sample = secrets.token_bytes(32)
        entropy_quality = len(set(entropy_sample)) / len(entropy_sample)

        if entropy_quality < 0.8:
            finding = SecurityFinding(
                finding_id=f"entropy_{int(time.time())}",
                title="Calidad de entropía baja",
                description=f"Calidad de entropía detectada: {entropy_quality:.2f}",
                severity=AuditSeverity.MEDIUM,
                category=AuditCategory.CRYPTOGRAPHY,
                remediation="Revisar fuente de entropía del sistema"
            )
            with self._lock:
                self.runtime_findings.append(finding)

class PenetrationTester:
    """Simulador de pruebas de penetración"""

    def __init__(self):
        self.test_results: List[SecurityFinding] = []

    async def run_penetration_tests(self, target_system: str) -> List[SecurityFinding]:
        """Ejecuta batería de pruebas de penetración simuladas"""
        logger.info(f"🎯 Iniciando pruebas de penetración en {target_system}")

        findings = []

        # Pruebas de autenticación
        auth_findings = await self._test_authentication_weaknesses()
        findings.extend(auth_findings)

        # Pruebas de autorización
        authz_findings = await self._test_authorization_flaws()
        findings.extend(authz_findings)

        # Pruebas de inyección
        injection_findings = await self._test_injection_vulnerabilities()
        findings.extend(injection_findings)

        # Pruebas de configuración
        config_findings = await self._test_configuration_issues()
        findings.extend(config_findings)

        return findings

    async def _test_authentication_weaknesses(self) -> List[SecurityFinding]:
        """Pruebas de debilidades en autenticación"""
        findings = []

        # Simular prueba de fuerza bruta
        # En implementación real, esto sería más sofisticado
        weak_passwords = ["password", "123456", "admin", "root"]

        for password in weak_passwords:
            # Simular verificación (en realidad no haría login)
            if len(password) < 8:
                finding = SecurityFinding(
                    finding_id=f"auth_weak_{hashlib.md5(password.encode()).hexdigest()[:8]}",
                    title="Contraseña potencialmente débil detectada",
                    description=f"Contraseña '{password}' es demasiado corta o común",
                    severity=AuditSeverity.HIGH,
                    category=AuditCategory.AUTHENTICATION,
                    remediation="Implementar política de contraseñas fuerte (mínimo 12 caracteres, complejidad)"
                )
                findings.append(finding)

        return findings

    async def _test_authorization_flaws(self) -> List[SecurityFinding]:
        """Pruebas de fallos de autorización"""
        findings = []

        # Simular prueba de IDOR (Insecure Direct Object Reference)
        # Simular acceso a recursos sin autorización

        finding = SecurityFinding(
            finding_id="authz_idor_sim",
            title="Potencial vulnerabilidad IDOR",
            description="Acceso directo a objetos sin verificación de propiedad",
            severity=AuditSeverity.HIGH,
            category=AuditCategory.AUTHORIZATION,
            remediation="Implementar verificación de propiedad en todas las operaciones de recursos"
        )
        findings.append(finding)

        return findings

    async def _test_injection_vulnerabilities(self) -> List[SecurityFinding]:
        """Pruebas de vulnerabilidades de inyección"""
        findings = []

        # Simular pruebas de SQL injection
        sql_payloads = ["' OR '1'='1", "'; DROP TABLE users;--"]

        for payload in sql_payloads:
            # Simular prueba (no ejecutar realmente)
            finding = SecurityFinding(
                finding_id=f"injection_sql_{hashlib.md5(payload.encode()).hexdigest()[:8]}",
                title="Payload SQL injection detectado",
                description=f"Payload potencialmente peligroso: {payload}",
                severity=AuditSeverity.CRITICAL,
                category=AuditCategory.INPUT_VALIDATION,
                remediation="Implementar sanitización de entrada y consultas parametrizadas"
            )
            findings.append(finding)

        return findings

    async def _test_configuration_issues(self) -> List[SecurityFinding]:
        """Pruebas de problemas de configuración"""
        findings = []

        # Verificar configuración insegura
        config_issues = [
            "DEBUG=True en producción",
            "CORS configurado para *",
            "Headers de seguridad faltantes"
        ]

        for issue in config_issues:
            finding = SecurityFinding(
                finding_id=f"config_{hashlib.md5(issue.encode()).hexdigest()[:8]}",
                title=f"Problema de configuración: {issue}",
                description="Configuración insegura detectada",
                severity=AuditSeverity.MEDIUM,
                category=AuditCategory.CONFIGURATION,
                remediation="Revisar y corregir configuración de seguridad"
            )
            findings.append(finding)

        return findings

class ComplianceValidator:
    """Validador de cumplimiento normativo"""

    def __init__(self):
        self.compliance_checks = {
            ComplianceStandard.SOC_2: self._check_soc2_compliance,
            ComplianceStandard.ISO_27001: self._check_iso27001_compliance,
        }

    async def validate_compliance(self, standard: ComplianceStandard,
                                system_components: Dict[str, Any]) -> Dict[str, Any]:
        """Valida cumplimiento con un estándar específico"""
        logger.info(f"📋 Validando cumplimiento {standard.value}")

        if standard in self.compliance_checks:
            return await self.compliance_checks[standard](system_components)
        else:
            return {"compliant": False, "score": 0, "issues": ["Estándar no soportado"]}

    async def _check_soc2_compliance(self, components: Dict[str, Any]) -> Dict[str, Any]:
        """Valida cumplimiento SOC 2"""
        score = 0
        max_score = 100
        issues = []

        # Verificar controles de seguridad
        security_controls = components.get("security_controls", {})

        # Confidencialidad
        if security_controls.get("encryption_at_rest", False):
            score += 15
        else:
            issues.append("Falta cifrado en reposo")

        if security_controls.get("encryption_in_transit", False):
            score += 15
        else:
            issues.append("Falta cifrado en tránsito")

        # Integridad
        if security_controls.get("data_validation", False):
            score += 10
        else:
            issues.append("Falta validación de datos")

        # Disponibilidad
        if components.get("monitoring", {}).get("active", False):
            score += 10
        else:
            issues.append("Falta monitoreo activo")

        # Control de acceso
        if security_controls.get("multi_factor_auth", False):
            score += 15
        else:
            issues.append("Falta autenticación multifactor")

        # Logging y monitoreo
        if components.get("logging", {}).get("comprehensive", False):
            score += 15
        else:
            issues.append("Logging insuficiente")

        # Gestión de incidentes
        if components.get("incident_response", {}).get("plan_exists", False):
            score += 10
        else:
            issues.append("Falta plan de respuesta a incidentes")

        # Auditoría
        if components.get("audit", {}).get("enabled", False):
            score += 10
        else:
            issues.append("Falta auditoría habilitada")

        compliance_percentage = (score / max_score) * 100

        return {
            "compliant": compliance_percentage >= 80,
            "score": compliance_percentage,
            "issues": issues,
            "recommendations": [
                "Implementar controles faltantes",
                "Realizar auditoría independiente",
                "Documentar procedimientos de seguridad"
            ]
        }

    async def _check_iso27001_compliance(self, components: Dict[str, Any]) -> Dict[str, Any]:
        """Valida cumplimiento ISO 27001 (simplificado)"""
        # Implementación simplificada
        return {
            "compliant": True,
            "score": 85.0,
            "issues": ["Requiere auditoría formal"],
            "recommendations": ["Obtener certificación formal"]
        }

class SecurityAuditEngine:
    """Motor principal de auditorías de seguridad"""

    def __init__(self):
        self.static_analyzer = StaticCodeAnalyzer()
        self.dynamic_analyzer = DynamicSecurityAnalyzer()
        self.penetration_tester = PenetrationTester()
        self.compliance_validator = ComplianceValidator()
        self.audit_reports: Dict[str, AuditReport] = {}

    async def perform_comprehensive_audit(self, target_system: str,
                                        code_paths: List[str] = None) -> AuditReport:
        """Realiza auditoría completa de seguridad"""
        audit_id = f"audit_{int(time.time())}_{hashlib.md5(target_system.encode()).hexdigest()[:8]}"

        logger.info(f"🔍 Iniciando auditoría completa: {audit_id}")

        report = AuditReport(
            audit_id=audit_id,
            target_system=target_system,
            audit_type="comprehensive",
            start_time=datetime.now()
        )

        try:
            # 1. Análisis estático de código
            logger.info("📝 Ejecutando análisis estático...")
            static_findings = []
            if code_paths:
                for path in code_paths:
                    if os.path.isfile(path):
                        findings = await self.static_analyzer.analyze_file(path)
                        static_findings.extend(findings)
                    elif os.path.isdir(path):
                        # Recursivamente analizar directorio
                        for root, dirs, files in os.walk(path):
                            for file in files:
                                if file.endswith(('.py', '.js', '.ts', '.java')):
                                    file_path = os.path.join(root, file)
                                    findings = await self.static_analyzer.analyze_file(file_path)
                                    static_findings.extend(findings)

            report.findings.extend(static_findings)

            # 2. Pruebas de penetración
            logger.info("🎯 Ejecutando pruebas de penetración...")
            pentest_findings = await self.penetration_tester.run_penetration_tests(target_system)
            report.findings.extend(pentest_findings)

            # 3. Análisis dinámico (simulado)
            logger.info("⚡ Iniciando análisis dinámico...")
            await self.dynamic_analyzer.start_monitoring()
            await asyncio.sleep(2)  # Permitir que el monitoreo recolecte datos
            dynamic_findings = self.dynamic_analyzer.runtime_findings.copy()
            report.findings.extend(dynamic_findings)
            self.dynamic_analyzer.stop_monitoring()

            # 4. Validación de cumplimiento
            logger.info("📋 Validando cumplimiento...")
            system_components = self._gather_system_components(target_system)
            soc2_compliance = await self.compliance_validator.validate_compliance(
                ComplianceStandard.SOC_2, system_components
            )
            report.compliance_score["SOC_2"] = soc2_compliance["score"]

            # 5. Evaluación de riesgos
            risk_assessment = self._perform_risk_assessment(report.findings)
            report.risk_assessment = risk_assessment

            # 6. Generar recomendaciones
            recommendations = self._generate_recommendations(report.findings, soc2_compliance)
            report.recommendations = recommendations

            # 7. Resumen ejecutivo
            report.executive_summary = self._generate_executive_summary(report)

        except Exception as e:
            logger.error(f"❌ Error en auditoría: {e}")
            report.findings.append(SecurityFinding(
                finding_id=f"audit_error_{audit_id}",
                title="Error durante auditoría",
                description=f"Error inesperado durante la auditoría: {str(e)}",
                severity=AuditSeverity.INFO,
                category=AuditCategory.LOGGING
            ))

        finally:
            report.end_time = datetime.now()
            self.audit_reports[audit_id] = report

        logger.info(f"✅ Auditoría completada: {len(report.findings)} hallazgos")
        return report

    def _gather_system_components(self, target_system: str) -> Dict[str, Any]:
        """Reúne información sobre componentes del sistema"""
        # En implementación real, esto analizaría el sistema actual
        return {
            "security_controls": {
                "encryption_at_rest": True,
                "encryption_in_transit": True,
                "data_validation": True,
                "multi_factor_auth": False
            },
            "monitoring": {"active": True},
            "logging": {"comprehensive": True},
            "incident_response": {"plan_exists": True},
            "audit": {"enabled": True}
        }

    def _perform_risk_assessment(self, findings: List[SecurityFinding]) -> Dict[str, Any]:
        """Realiza evaluación de riesgos basada en hallazgos"""
        risk_scores = {
            AuditSeverity.CRITICAL: 10,
            AuditSeverity.HIGH: 7,
            AuditSeverity.MEDIUM: 4,
            AuditSeverity.LOW: 2,
            AuditSeverity.INFO: 1
        }

        total_risk = sum(risk_scores.get(finding.severity, 0) for finding in findings)
        risk_level = "Bajo" if total_risk < 10 else "Medio" if total_risk < 25 else "Alto" if total_risk < 50 else "Crítico"

        return {
            "total_risk_score": total_risk,
            "risk_level": risk_level,
            "critical_findings": len([f for f in findings if f.severity == AuditSeverity.CRITICAL]),
            "high_findings": len([f for f in findings if f.severity == AuditSeverity.HIGH]),
            "categories_affected": list(set(f.category.value for f in findings))
        }

    def _generate_recommendations(self, findings: List[SecurityFinding],
                                compliance: Dict[str, Any]) -> List[str]:
        """Genera recomendaciones basadas en hallazgos"""
        recommendations = []

        # Recomendaciones basadas en severidad
        if any(f.severity == AuditSeverity.CRITICAL for f in findings):
            recommendations.append("🔴 Abordar inmediatamente hallazgos críticos - pueden comprometer la seguridad del sistema")

        if any(f.severity == AuditSeverity.HIGH for f in findings):
            recommendations.append("🟠 Priorizar corrección de hallazgos de alta severidad en las próximas 2 semanas")

        # Recomendaciones por categoría
        categories = set(f.category for f in findings)
        if AuditCategory.CRYPTOGRAPHY in categories:
            recommendations.append("🔐 Revisar y actualizar implementación criptográfica")

        if AuditCategory.AUTHENTICATION in categories:
            recommendations.append("🔑 Mejorar controles de autenticación")

        # Recomendaciones de cumplimiento
        if not compliance.get("compliant", False):
            recommendations.append("📋 Abordar problemas de cumplimiento identificados")
            recommendations.extend(compliance.get("recommendations", []))

        return recommendations

    def _generate_executive_summary(self, report: AuditReport) -> str:
        """Genera resumen ejecutivo de la auditoría"""
        duration = report.end_time - report.start_time if report.end_time else timedelta(0)

        summary = f"""
Auditoría de Seguridad Completa - {report.target_system}

Período de auditoría: {report.start_time.strftime('%Y-%m-%d %H:%M')} - {report.end_time.strftime('%Y-%m-%d %H:%M') if report.end_time else 'En progreso'}
Duración: {duration.total_seconds():.1f} segundos

📊 HALLAZGOS:
- Total de hallazgos: {len(report.findings)}
- Críticos: {len([f for f in report.findings if f.severity == AuditSeverity.CRITICAL])}
- Altos: {len([f for f in report.findings if f.severity == AuditSeverity.HIGH])}
- Medios: {len([f for f in report.findings if f.severity == AuditSeverity.MEDIUM])}

🎯 EVALUACIÓN DE RIESGOS:
- Nivel de riesgo: {report.risk_assessment.get('risk_level', 'Desconocido')}
- Puntaje total: {report.risk_assessment.get('total_risk_score', 0)}

📋 CUMPLIMIENTO:
- SOC 2: {report.compliance_score.get('SOC_2', 0):.1f}%

{chr(10).join(f"• {rec}" for rec in report.recommendations[:3])}
        """.strip()

        return summary

    def generate_audit_report(self, audit_id: str) -> str:
        """Genera reporte completo en formato texto"""
        if audit_id not in self.audit_reports:
            return "Auditoría no encontrada"

        report = self.audit_reports[audit_id]

        output = []
        output.append("=" * 80)
        output.append("REPORTE DE AUDITORÍA DE SEGURIDAD")
        output.append("=" * 80)
        output.append("")

        output.append(report.executive_summary)
        output.append("")

        output.append("DETALLE DE HALLAZGOS:")
        output.append("-" * 40)

        for finding in sorted(report.findings, key=lambda x: x.severity.value, reverse=True):
            output.append(f"🔍 {finding.title}")
            output.append(f"   Severidad: {finding.severity.value.upper()}")
            output.append(f"   Categoría: {finding.category.value}")
            if finding.file_path:
                output.append(f"   Archivo: {finding.file_path}:{finding.line_number or 'N/A'}")
            output.append(f"   Descripción: {finding.description}")
            if finding.remediation:
                output.append(f"   Remediation: {finding.remediation}")
            output.append("")

        output.append("RECOMENDACIONES COMPLETAS:")
        output.append("-" * 40)
        for rec in report.recommendations:
            output.append(f"• {rec}")

        return "\n".join(output)

async def main():
    """Función principal de demostración de auditorías"""
    print("🔍 DEMO DE AUDITORÍAS DE SEGURIDAD EXHAUSTIVAS - AEGIS Framework")
    print("=" * 70)

    # Inicializar motor de auditorías
    audit_engine = SecurityAuditEngine()

    try:
        # Ejecutar auditoría completa
        print("\n🚀 Iniciando auditoría completa del sistema AEGIS...")

        # Definir rutas de código para análisis
        code_paths = ["p2p_network.py", "crypto_engine.py", "performance_optimizer.py"]

        audit_report = await audit_engine.perform_comprehensive_audit(
            target_system="AEGIS_Framework",
            code_paths=code_paths
        )

        print("\n📊 RESULTADOS DE LA AUDITORÍA:")
        print(f"   🔍 ID de auditoría: {audit_report.audit_id}")
        print(f"   📅 Inicio: {audit_report.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   📅 Fin: {audit_report.end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   🎯 Hallazgos totales: {len(audit_report.findings)}")

        # Conteo por severidad
        severity_count = {}
        for finding in audit_report.findings:
            severity_count[finding.severity] = severity_count.get(finding.severity, 0) + 1

        print("   📈 Por severidad:")
        for severity, count in severity_count.items():
            print(f"      {severity.value.upper()}: {count}")

        # Puntajes de cumplimiento
        print("   📋 Cumplimiento:")
        for standard, score in audit_report.compliance_score.items():
            status = "✅" if score >= 80 else "⚠️"
            print(f"      {standard}: {status} {score:.1f}%")

        # Evaluación de riesgos
        risk = audit_report.risk_assessment
        print(f"   🎯 Evaluación de riesgos: {risk.get('risk_level', 'Desconocido')}")
        print(f"   📊 Puntaje de riesgo: {risk.get('total_risk_score', 0)}")

        print("\n📋 RESUMEN EJECUTIVO:")
        print(audit_report.executive_summary)

        # Generar reporte completo
        print("\n📄 GENERANDO REPORTE COMPLETO...")
        full_report = audit_engine.generate_audit_report(audit_report.audit_id)

        # Guardar reporte a archivo
        report_filename = f"security_audit_report_{audit_report.audit_id}.txt"
        with open(report_filename, 'w', encoding='utf-8') as f:
            f.write(full_report)

        print(f"   ✅ Reporte guardado: {report_filename}")

        print("\n🎉 ¡Auditoría de seguridad completada exitosamente!")
        print("   🔒 Vulnerabilidades identificadas y documentadas")
        print("   📊 Evaluación de riesgos realizada")
        print("   📋 Validación de cumplimiento completada")
        print("   📄 Reporte de auditoría generado")

        return {
            "audit_report": audit_report,
            "findings_count": len(audit_report.findings),
            "risk_assessment": audit_report.risk_assessment,
            "compliance_scores": audit_report.compliance_score,
            "recommendations": audit_report.recommendations
        }

    except Exception as e:
        print(f"❌ Error en auditoría: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    asyncio.run(main())
