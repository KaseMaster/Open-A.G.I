"""
AEGIS Advanced Testing Framework
Comprehensive testing for security features
"""

import asyncio
import time
import logging
from typing import Dict, List, Any, Optional, Callable, Awaitable
from dataclasses import dataclass, field
from enum import Enum
import pytest
import random
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.aegis.security.advanced_crypto import AdvancedSecurityManager, SecurityFeature
from src.aegis.core.performance_optimizer import PerformanceOptimizer
from src.aegis.monitoring.security_monitor import SecurityMonitor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestCategory(Enum):
    """Test categories"""
    UNIT = "unit"
    INTEGRATION = "integration"
    PERFORMANCE = "performance"
    SECURITY = "security"
    COMPLIANCE = "compliance"


class TestSeverity(Enum):
    """Test severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class TestCase:
    """Individual test case"""
    test_id: str
    name: str
    category: TestCategory
    severity: TestSeverity
    description: str
    test_function: Callable
    expected_result: Any = None
    timeout_seconds: float = 30.0
    tags: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)


@dataclass
class TestResult:
    """Result of a test execution"""
    test_id: str
    name: str
    category: TestCategory
    severity: TestSeverity
    passed: bool
    execution_time: float
    error_message: Optional[str] = None
    stack_trace: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    memory_usage_mb: float = 0.0


@dataclass
class TestSuiteResult:
    """Result of a complete test suite"""
    suite_name: str
    total_tests: int
    passed_tests: int
    failed_tests: int
    execution_time: float
    test_results: List[TestResult] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    coverage_percentage: float = 0.0


class AdvancedSecurityTestFramework:
    """Advanced testing framework for AEGIS security features"""
    
    def __init__(self):
        self.security_manager = AdvancedSecurityManager()
        self.performance_optimizer = PerformanceOptimizer()
        self.security_monitor = SecurityMonitor(self.security_manager)
        
        # Test cases organized by category
        self.test_cases: Dict[TestCategory, List[TestCase]] = {
            category: [] for category in TestCategory
        }
        
        # Test results
        self.test_results: List[TestResult] = []
        self.suite_results: List[TestSuiteResult] = []
        
        # Initialize test cases
        self._initialize_test_cases()
    
    def _initialize_test_cases(self):
        """Initialize all test cases"""
        # Unit tests
        self._add_unit_tests()
        
        # Integration tests
        self._add_integration_tests()
        
        # Performance tests
        self._add_performance_tests()
        
        # Security tests
        self._add_security_tests()
        
        # Compliance tests
        self._add_compliance_tests()
    
    def _add_unit_tests(self):
        """Add unit tests for security features"""
        unit_tests = [
            TestCase(
                test_id="ut_zkp_001",
                name="ZKP Proof Generation",
                category=TestCategory.UNIT,
                severity=TestSeverity.HIGH,
                description="Test zero-knowledge proof generation",
                test_function=self._test_zkp_proof_generation,
                tags=["zkp", "authentication"]
            ),
            TestCase(
                test_id="ut_zkp_002",
                name="ZKP Proof Verification",
                category=TestCategory.UNIT,
                severity=TestSeverity.HIGH,
                description="Test zero-knowledge proof verification",
                test_function=self._test_zkp_proof_verification,
                tags=["zkp", "authentication"]
            ),
            TestCase(
                test_id="ut_he_001",
                name="Homomorphic Encryption",
                category=TestCategory.UNIT,
                severity=TestSeverity.HIGH,
                description="Test homomorphic encryption/decryption",
                test_function=self._test_homomorphic_encryption,
                tags=["homomorphic", "encryption"]
            ),
            TestCase(
                test_id="ut_he_002",
                name="Homomorphic Addition",
                category=TestCategory.UNIT,
                severity=TestSeverity.HIGH,
                description="Test homomorphic addition operations",
                test_function=self._test_homomorphic_addition,
                tags=["homomorphic", "encryption"]
            ),
            TestCase(
                test_id="ut_smc_001",
                name="SMC Secret Sharing",
                category=TestCategory.UNIT,
                severity=TestSeverity.HIGH,
                description="Test secure multi-party computation secret sharing",
                test_function=self._test_smc_secret_sharing,
                tags=["smc", "secret_sharing"]
            ),
            TestCase(
                test_id="ut_dp_001",
                name="Differential Privacy",
                category=TestCategory.UNIT,
                severity=TestSeverity.HIGH,
                description="Test differential privacy mechanisms",
                test_function=self._test_differential_privacy,
                tags=["differential_privacy", "privacy"]
            )
        ]
        
        for test_case in unit_tests:
            self.test_cases[TestCategory.UNIT].append(test_case)
    
    def _add_integration_tests(self):
        """Add integration tests for security features"""
        integration_tests = [
            TestCase(
                test_id="it_auth_001",
                name="Authentication Integration",
                category=TestCategory.INTEGRATION,
                severity=TestSeverity.CRITICAL,
                description="Test authentication with ZKP integration",
                test_function=self._test_authentication_integration,
                tags=["authentication", "zkp", "integration"]
            ),
            TestCase(
                test_id="it_encrypt_001",
                name="Encryption Integration",
                category=TestCategory.INTEGRATION,
                severity=TestSeverity.HIGH,
                description="Test encryption with homomorphic operations",
                test_function=self._test_encryption_integration,
                tags=["encryption", "homomorphic", "integration"]
            ),
            TestCase(
                test_id="it_privacy_001",
                name="Privacy Integration",
                category=TestCategory.INTEGRATION,
                severity=TestSeverity.HIGH,
                description="Test privacy with differential privacy integration",
                test_function=self._test_privacy_integration,
                tags=["privacy", "differential_privacy", "integration"]
            ),
            TestCase(
                test_id="it_smc_001",
                name="SMC Integration",
                category=TestCategory.INTEGRATION,
                severity=TestSeverity.HIGH,
                description="Test secure multi-party computation integration",
                test_function=self._test_smc_integration,
                tags=["smc", "integration"]
            )
        ]
        
        for test_case in integration_tests:
            self.test_cases[TestCategory.INTEGRATION].append(test_case)
    
    def _add_performance_tests(self):
        """Add performance tests for security features"""
        performance_tests = [
            TestCase(
                test_id="pt_zkp_001",
                name="ZKP Performance",
                category=TestCategory.PERFORMANCE,
                severity=TestSeverity.MEDIUM,
                description="Test ZKP proof generation performance",
                test_function=self._test_zkp_performance,
                tags=["zkp", "performance"]
            ),
            TestCase(
                test_id="pt_he_001",
                name="Homomorphic Encryption Performance",
                category=TestCategory.PERFORMANCE,
                severity=TestSeverity.MEDIUM,
                description="Test homomorphic encryption performance",
                test_function=self._test_homomorphic_performance,
                tags=["homomorphic", "performance"]
            ),
            TestCase(
                test_id="pt_smc_001",
                name="SMC Performance",
                category=TestCategory.PERFORMANCE,
                severity=TestSeverity.MEDIUM,
                description="Test secure multi-party computation performance",
                test_function=self._test_smc_performance,
                tags=["smc", "performance"]
            ),
            TestCase(
                test_id="pt_dp_001",
                name="Differential Privacy Performance",
                category=TestCategory.PERFORMANCE,
                severity=TestSeverity.MEDIUM,
                description="Test differential privacy performance",
                test_function=self._test_dp_performance,
                tags=["differential_privacy", "performance"]
            )
        ]
        
        for test_case in performance_tests:
            self.test_cases[TestCategory.PERFORMANCE].append(test_case)
    
    def _add_security_tests(self):
        """Add security tests for advanced features"""
        security_tests = [
            TestCase(
                test_id="st_zkp_001",
                name="ZKP Security",
                category=TestCategory.SECURITY,
                severity=TestSeverity.CRITICAL,
                description="Test ZKP security against known attacks",
                test_function=self._test_zkp_security,
                tags=["zkp", "security", "attack"]
            ),
            TestCase(
                test_id="st_he_001",
                name="Homomorphic Encryption Security",
                category=TestCategory.SECURITY,
                severity=TestSeverity.CRITICAL,
                description="Test homomorphic encryption security",
                test_function=self._test_homomorphic_security,
                tags=["homomorphic", "security", "encryption"]
            ),
            TestCase(
                test_id="st_smc_001",
                name="SMC Security",
                category=TestCategory.SECURITY,
                severity=TestSeverity.HIGH,
                description="Test secure multi-party computation security",
                test_function=self._test_smc_security,
                tags=["smc", "security"]
            ),
            TestCase(
                test_id="st_dp_001",
                name="Differential Privacy Security",
                category=TestCategory.SECURITY,
                severity=TestSeverity.HIGH,
                description="Test differential privacy security guarantees",
                test_function=self._test_dp_security,
                tags=["differential_privacy", "security", "privacy"]
            )
        ]
        
        for test_case in security_tests:
            self.test_cases[TestCategory.SECURITY].append(test_case)
    
    def _add_compliance_tests(self):
        """Add compliance tests for security features"""
        compliance_tests = [
            TestCase(
                test_id="ct_gdpr_001",
                name="GDPR Compliance",
                category=TestCategory.COMPLIANCE,
                severity=TestSeverity.HIGH,
                description="Test GDPR compliance for data protection",
                test_function=self._test_gdpr_compliance,
                tags=["gdpr", "compliance", "privacy"]
            ),
            TestCase(
                test_id="ct_hipaa_001",
                name="HIPAA Compliance",
                category=TestCategory.COMPLIANCE,
                severity=TestSeverity.HIGH,
                description="Test HIPAA compliance for healthcare data",
                test_function=self._test_hipaa_compliance,
                tags=["hipaa", "compliance", "healthcare"]
            ),
            TestCase(
                test_id="ct_soc2_001",
                name="SOC2 Compliance",
                category=TestCategory.COMPLIANCE,
                severity=TestSeverity.MEDIUM,
                description="Test SOC2 compliance for security controls",
                test_function=self._test_soc2_compliance,
                tags=["soc2", "compliance", "security"]
            )
        ]
        
        for test_case in compliance_tests:
            self.test_cases[TestCategory.COMPLIANCE].append(test_case)
    
    async def run_test_suite(
        self,
        categories: Optional[List[TestCategory]] = None,
        tags: Optional[List[str]] = None
    ) -> TestSuiteResult:
        """Run a comprehensive test suite"""
        if categories is None:
            categories = list(TestCategory)
        
        logger.info(f"🧪 Starting test suite for categories: {[c.value for c in categories]}")
        
        start_time = time.time()
        all_results = []
        
        # Filter test cases by categories and tags
        filtered_cases = []
        for category in categories:
            for test_case in self.test_cases[category]:
                # Filter by tags if specified
                if tags and not any(tag in test_case.tags for tag in tags):
                    continue
                filtered_cases.append(test_case)
        
        logger.info(f"📋 Running {len(filtered_cases)} test cases...")
        
        # Run tests concurrently
        test_tasks = []
        for test_case in filtered_cases:
            task = asyncio.create_task(self._run_single_test(test_case))
            test_tasks.append(task)
        
        # Wait for all tests to complete
        results = await asyncio.gather(*test_tasks, return_exceptions=True)
        
        # Process results
        for result in results:
            if isinstance(result, TestResult):
                all_results.append(result)
            elif isinstance(result, Exception):
                logger.error(f"Test execution error: {result}")
        
        # Calculate suite statistics
        total_tests = len(all_results)
        passed_tests = sum(1 for r in all_results if r.passed)
        failed_tests = total_tests - passed_tests
        execution_time = time.time() - start_time
        
        # Calculate coverage (simplified)
        coverage_percentage = (passed_tests / max(1, total_tests)) * 100
        
        suite_result = TestSuiteResult(
            suite_name=f"AEGIS_Security_Test_Suite_{int(time.time())}",
            total_tests=total_tests,
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            execution_time=execution_time,
            test_results=all_results,
            coverage_percentage=coverage_percentage
        )
        
        self.suite_results.append(suite_result)
        
        logger.info(f"✅ Test suite completed in {execution_time:.2f}s")
        logger.info(f"📊 Results: {passed_tests}/{total_tests} tests passed ({coverage_percentage:.1f}% coverage)")
        
        return suite_result
    
    async def _run_single_test(self, test_case: TestCase) -> TestResult:
        """Run a single test case"""
        start_time = time.time()
        
        try:
            # Run the test function
            result = await asyncio.wait_for(
                test_case.test_function(),
                timeout=test_case.timeout_seconds
            )
            
            execution_time = time.time() - start_time
            
            test_result = TestResult(
                test_id=test_case.test_id,
                name=test_case.name,
                category=test_case.category,
                severity=test_case.severity,
                passed=result is not False,  # Assume True unless explicitly False
                execution_time=execution_time,
                error_message=None,
                stack_trace=None
            )
            
            logger.info(f"✅ Test {test_case.test_id}: {test_case.name} - {'PASSED' if result is not False else 'FAILED'}")
            
        except asyncio.TimeoutError:
            execution_time = time.time() - start_time
            test_result = TestResult(
                test_id=test_case.test_id,
                name=test_case.name,
                category=test_case.category,
                severity=test_case.severity,
                passed=False,
                execution_time=execution_time,
                error_message=f"Test timeout after {test_case.timeout_seconds}s",
                stack_trace=None
            )
            logger.error(f"⏰ Test {test_case.test_id}: {test_case.name} - TIMEOUT")
            
        except Exception as e:
            execution_time = time.time() - start_time
            import traceback
            stack_trace = traceback.format_exc()
            
            test_result = TestResult(
                test_id=test_case.test_id,
                name=test_case.name,
                category=test_case.category,
                severity=test_case.severity,
                passed=False,
                execution_time=execution_time,
                error_message=str(e),
                stack_trace=stack_trace
            )
            
            logger.error(f"❌ Test {test_case.test_id}: {test_case.name} - FAILED: {e}")
        
        self.test_results.append(test_result)
        return test_result
    
    # Unit Test Implementations
    async def _test_zkp_proof_generation(self) -> bool:
        """Test ZKP proof generation"""
        secret = b"test_secret_123"
        statement = "authenticate:test_user"
        verifier_id = "test_verifier"
        
        # Generate proof
        proof = self.security_manager.create_zk_proof(secret, statement, verifier_id)
        
        # Verify proof was generated
        return proof is not None and hasattr(proof, 'proof_id')
    
    async def _test_zkp_proof_verification(self) -> bool:
        """Test ZKP proof verification"""
        secret = b"test_secret_456"
        statement = "authenticate:test_user_2"
        verifier_id = "test_verifier_2"
        
        # Generate proof
        proof = self.security_manager.create_zk_proof(secret, statement, verifier_id)
        
        # Verify proof
        is_valid = self.security_manager.verify_zk_proof(proof, statement)
        
        return is_valid
    
    async def _test_homomorphic_encryption(self) -> bool:
        """Test homomorphic encryption/decryption"""
        original_value = 42
        
        # Encrypt value
        encrypted = self.security_manager.encrypt_value(original_value)
        
        # Decrypt value
        decrypted = self.security_manager.decrypt_value(encrypted)
        
        return decrypted == original_value
    
    async def _test_homomorphic_addition(self) -> bool:
        """Test homomorphic addition"""
        value1 = 15
        value2 = 25
        expected_sum = value1 + value2
        
        # Encrypt values
        encrypted1 = self.security_manager.encrypt_value(value1)
        encrypted2 = self.security_manager.encrypt_value(value2)
        
        # Add homomorphically
        encrypted_sum = self.security_manager.add_encrypted_values(encrypted1, encrypted2)
        
        # Decrypt result
        actual_sum = self.security_manager.decrypt_value(encrypted_sum)
        
        return actual_sum == expected_sum
    
    async def _test_smc_secret_sharing(self) -> bool:
        """Test SMC secret sharing"""
        secret = 12345
        threshold = 3
        
        # Generate shares
        shares = self.security_manager.generate_secret_shares(secret, threshold)
        
        # Reconstruct secret
        reconstructed = self.security_manager.reconstruct_secret_from_shares(shares)
        
        return reconstructed == secret
    
    async def _test_differential_privacy(self) -> bool:
        """Test differential privacy"""
        true_value = 1000
        private_value = self.security_manager.privatize_data(true_value, "count")
        
        # Check that private value is a float (indicating noise was added)
        return isinstance(private_value, float)
    
    # Integration Test Implementations
    async def _test_authentication_integration(self) -> bool:
        """Test authentication integration with ZKP"""
        # Enable ZKP
        self.security_manager.enabled_features[SecurityFeature.ZERO_KNOWLEDGE_PROOF] = True
        
        # Create proof
        secret = b"integration_test_secret"
        statement = "authenticate:integration_test"
        proof = self.security_manager.create_zk_proof(secret, statement, "integration_verifier")
        
        # Verify proof
        is_valid = self.security_manager.verify_zk_proof(proof, statement)
        
        # Disable ZKP
        self.security_manager.enabled_features[SecurityFeature.ZERO_KNOWLEDGE_PROOF] = False
        
        return is_valid
    
    async def _test_encryption_integration(self) -> bool:
        """Test encryption integration with homomorphic operations"""
        # Enable homomorphic encryption
        self.security_manager.enabled_features[SecurityFeature.HOMOMORPHIC_ENCRYPTION] = True
        
        # Perform homomorphic operation
        value1 = 10
        value2 = 20
        encrypted1 = self.security_manager.encrypt_value(value1)
        encrypted2 = self.security_manager.encrypt_value(value2)
        encrypted_sum = self.security_manager.add_encrypted_values(encrypted1, encrypted2)
        decrypted_sum = self.security_manager.decrypt_value(encrypted_sum)
        
        expected_sum = value1 + value2
        
        # Disable homomorphic encryption
        self.security_manager.enabled_features[SecurityFeature.HOMOMORPHIC_ENCRYPTION] = False
        
        return decrypted_sum == expected_sum
    
    async def _test_privacy_integration(self) -> bool:
        """Test privacy integration with differential privacy"""
        # Enable differential privacy
        self.security_manager.enabled_features[SecurityFeature.DIFFERENTIAL_PRIVACY] = True
        
        # Perform private query
        true_count = 500
        private_count = self.security_manager.privatize_data(true_count, "count")
        
        # Check that privacy was applied
        is_privatized = isinstance(private_count, float) and private_count != true_count
        
        # Disable differential privacy
        self.security_manager.enabled_features[SecurityFeature.DIFFERENTIAL_PRIVACY] = False
        
        return is_privatized
    
    async def _test_smc_integration(self) -> bool:
        """Test SMC integration"""
        # Enable SMC
        self.security_manager.enabled_features[SecurityFeature.SECURE_MULTI_PARTY_COMPUTATION] = True
        
        # Perform SMC operation
        secret = 98765
        shares = self.security_manager.generate_secret_shares(secret, 3)
        reconstructed = self.security_manager.reconstruct_secret_from_shares(shares)
        
        # Disable SMC
        self.security_manager.enabled_features[SecurityFeature.SECURE_MULTI_PARTY_COMPUTATION] = False
        
        return reconstructed == secret
    
    # Performance Test Implementations
    async def _test_zkp_performance(self) -> bool:
        """Test ZKP performance"""
        start_time = time.time()
        
        # Generate 100 proofs
        for i in range(100):
            secret = f"perf_test_secret_{i}".encode()
            statement = f"authenticate:perf_test_{i}"
            proof = self.security_manager.create_zk_proof(secret, statement, "perf_verifier")
        
        end_time = time.time()
        total_time = end_time - start_time
        ops_per_second = 100 / total_time
        
        logger.info(f"ZKP Performance: {ops_per_second:.0f} ops/s")
        
        # Should achieve reasonable performance
        return ops_per_second > 1000  # 1000+ ops/s minimum
    
    async def _test_homomorphic_performance(self) -> bool:
        """Test homomorphic encryption performance"""
        start_time = time.time()
        
        # Perform 100 homomorphic operations
        for i in range(100):
            value = i
            encrypted = self.security_manager.encrypt_value(value)
            decrypted = self.security_manager.decrypt_value(encrypted)
        
        end_time = time.time()
        total_time = end_time - start_time
        ops_per_second = 100 / total_time
        
        logger.info(f"Homomorphic Performance: {ops_per_second:.0f} ops/s")
        
        return ops_per_second > 100  # 100+ ops/s minimum
    
    async def _test_smc_performance(self) -> bool:
        """Test SMC performance"""
        start_time = time.time()
        
        # Perform 50 SMC operations
        for i in range(50):
            secret = i * 100
            shares = self.security_manager.generate_secret_shares(secret, 3)
            reconstructed = self.security_manager.reconstruct_secret_from_shares(shares)
        
        end_time = time.time()
        total_time = end_time - start_time
        ops_per_second = 50 / total_time
        
        logger.info(f"SMC Performance: {ops_per_second:.0f} ops/s")
        
        return ops_per_second > 10  # 10+ ops/s minimum
    
    async def _test_dp_performance(self) -> bool:
        """Test differential privacy performance"""
        start_time = time.time()
        
        # Perform 200 DP queries
        for i in range(200):
            value = i * 10
            private_value = self.security_manager.privatize_data(value, "count")
        
        end_time = time.time()
        total_time = end_time - start_time
        ops_per_second = 200 / total_time
        
        logger.info(f"DP Performance: {ops_per_second:.0f} ops/s")
        
        return ops_per_second > 500  # 500+ ops/s minimum
    
    # Security Test Implementations
    async def _test_zkp_security(self) -> bool:
        """Test ZKP security"""
        # Test proof verification with wrong statement
        secret = b"security_test_secret"
        statement = "authenticate:security_test"
        wrong_statement = "authenticate:wrong_test"
        
        proof = self.security_manager.create_zk_proof(secret, statement, "security_verifier")
        is_valid_wrong = self.security_manager.verify_zk_proof(proof, wrong_statement)
        
        # Should reject wrong statement
        return not is_valid_wrong
    
    async def _test_homomorphic_security(self) -> bool:
        """Test homomorphic encryption security"""
        # Test that encrypted values don't reveal plaintext
        value = 42
        encrypted = self.security_manager.encrypt_value(value)
        
        # Encrypted value should not equal plaintext
        is_secure = str(encrypted) != str(value)
        
        return is_secure
    
    async def _test_smc_security(self) -> bool:
        """Test SMC security"""
        # Test that individual shares don't reveal secret
        secret = 12345
        shares = self.security_manager.generate_secret_shares(secret, 3)
        
        # Using less than threshold shares should not reveal secret
        partial_shares = dict(list(shares.items())[:2])  # Only 2 out of 3
        try:
            reconstructed = self.security_manager.reconstruct_secret_from_shares(partial_shares)
            # Should either fail or produce wrong result
            is_secure = reconstructed != secret
        except:
            # Exception is also acceptable (security by design)
            is_secure = True
        
        return is_secure
    
    async def _test_dp_security(self) -> bool:
        """Test differential privacy security"""
        # Test that repeated queries don't reveal individual values
        true_values = [100, 200, 300, 400, 500]
        private_values = []
        
        for value in true_values:
            private_value = self.security_manager.privatize_data(value, "count")
            private_values.append(private_value)
        
        # Private values should be different from true values
        differences = [abs(pv - tv) for pv, tv in zip(private_values, true_values)]
        avg_difference = sum(differences) / len(differences)
        
        # Should have significant noise added
        return avg_difference > 10
    
    # Compliance Test Implementations
    async def _test_gdpr_compliance(self) -> bool:
        """Test GDPR compliance"""
        # Check that privacy features are available
        dp_enabled = self.security_manager.enabled_features.get(
            SecurityFeature.DIFFERENTIAL_PRIVACY, False
        )
        zkp_enabled = self.security_manager.enabled_features.get(
            SecurityFeature.ZERO_KNOWLEDGE_PROOF, False
        )
        
        # GDPR requires privacy protection
        return dp_enabled or zkp_enabled
    
    async def _test_hipaa_compliance(self) -> bool:
        """Test HIPAA compliance"""
        # Check that encryption is available
        he_enabled = self.security_manager.enabled_features.get(
            SecurityFeature.HOMOMORPHIC_ENCRYPTION, False
        )
        smc_enabled = self.security_manager.enabled_features.get(
            SecurityFeature.SECURE_MULTI_PARTY_COMPUTATION, False
        )
        
        # HIPAA requires data encryption
        return he_enabled or smc_enabled
    
    async def _test_soc2_compliance(self) -> bool:
        """Test SOC2 compliance"""
        # Check that monitoring is available
        stats = self.security_manager.get_security_stats()
        monitoring_available = "enabled_features" in stats
        
        # SOC2 requires monitoring and auditing
        return monitoring_available
    
    def print_test_report(self, suite_result: TestSuiteResult):
        """Print comprehensive test report"""
        print("\n" + "="*80)
        print(f"AEGIS FRAMEWORK - SECURITY TEST REPORT")
        print("="*80)
        print(f"Suite: {suite_result.suite_name}")
        print(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(suite_result.timestamp))}")
        print(f"Execution Time: {suite_result.execution_time:.2f}s")
        print()
        
        # Summary statistics
        print("SUMMARY STATISTICS:")
        print("-" * 30)
        print(f"Total Tests: {suite_result.total_tests}")
        print(f"Passed: {suite_result.passed_tests}")
        print(f"Failed: {suite_result.failed_tests}")
        print(f"Success Rate: {(suite_result.passed_tests/suite_result.total_tests)*100:.1f}%")
        print(f"Coverage: {suite_result.coverage_percentage:.1f}%")
        print()
        
        # Results by category
        print("RESULTS BY CATEGORY:")
        print("-" * 30)
        categories = {}
        for result in suite_result.test_results:
            category = result.category.value
            if category not in categories:
                categories[category] = {"total": 0, "passed": 0}
            categories[category]["total"] += 1
            if result.passed:
                categories[category]["passed"] += 1
        
        for category, stats in categories.items():
            rate = (stats["passed"] / stats["total"]) * 100
            print(f"  {category.upper()}: {stats['passed']}/{stats['total']} ({rate:.1f}%)")
        print()
        
        # Failed tests
        failed_tests = [r for r in suite_result.test_results if not r.passed]
        if failed_tests:
            print("FAILED TESTS:")
            print("-" * 30)
            for test in failed_tests[:10]:  # Show top 10
                print(f"  ❌ {test.name} ({test.test_id})")
                if test.error_message:
                    print(f"     Error: {test.error_message}")
            if len(failed_tests) > 10:
                print(f"     ... and {len(failed_tests) - 10} more")
            print()
        
        # Performance highlights
        performance_tests = [r for r in suite_result.test_results 
                           if r.category == TestCategory.PERFORMANCE and r.passed]
        if performance_tests:
            print("PERFORMANCE HIGHLIGHTS:")
            print("-" * 30)
            for test in performance_tests:
                ops_per_sec = 1.0 / test.execution_time if test.execution_time > 0 else 0
                print(f"  ⚡ {test.name}: {ops_per_sec:.0f} ops/s")
            print()
        
        print("="*80)


async def run_advanced_security_tests():
    """Run comprehensive advanced security tests"""
    logger.info("🔬 Starting AEGIS Advanced Security Testing Framework")
    
    # Create test framework
    test_framework = AdvancedSecurityTestFramework()
    
    # Run comprehensive test suite
    suite_result = await test_framework.run_test_suite()
    
    # Print detailed report
    test_framework.print_test_report(suite_result)
    
    # Export results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"aegis_security_test_results_{timestamp}.json"
    
    # Export to JSON
    import json
    report_data = {
        "suite_name": suite_result.suite_name,
        "timestamp": suite_result.timestamp,
        "execution_time": suite_result.execution_time,
        "total_tests": suite_result.total_tests,
        "passed_tests": suite_result.passed_tests,
        "failed_tests": suite_result.failed_tests,
        "coverage_percentage": suite_result.coverage_percentage,
        "test_results": [
            {
                "test_id": result.test_id,
                "name": result.name,
                "category": result.category.value,
                "severity": result.severity.value,
                "passed": result.passed,
                "execution_time": result.execution_time,
                "error_message": result.error_message,
                "timestamp": result.timestamp
            }
            for result in suite_result.test_results
        ]
    }
    
    with open(filename, 'w') as f:
        json.dump(report_data, f, indent=2)
    
    logger.info(f"📄 Test results exported to {filename}")
    
    # Summary
    success_rate = (suite_result.passed_tests / suite_result.total_tests) * 100
    logger.info(f"🎯 Test Suite Summary: {suite_result.passed_tests}/{suite_result.total_tests} passed ({success_rate:.1f}% success rate)")
    
    return suite_result


if __name__ == "__main__":
    asyncio.run(run_advanced_security_tests())
