"""
AEGIS Comprehensive Tutorial Series
Step-by-step tutorials for advanced security features
"""

import asyncio
import time
import random
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.aegis.security.advanced_crypto import AdvancedSecurityManager, SecurityFeature
from src.aegis.core.performance_optimizer import PerformanceOptimizer
from src.aegis.ml.federated_learning import FederatedClient, FederatedServer, FederatedConfig
from src.aegis.blockchain.consensus_protocol import HybridConsensus
from cryptography.hazmat.primitives.asymmetric import ed25519

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AEGISTutorial:
    """Base class for AEGIS tutorials"""
    
    def __init__(self, title: str, description: str):
        self.title = title
        self.description = description
        self.steps_completed = 0
        self.total_steps = 0
        self.start_time = time.time()
    
    def log_step(self, step_description: str):
        """Log tutorial step completion"""
        self.steps_completed += 1
        logger.info(f"Step {self.steps_completed}/{self.total_steps}: {step_description}")
    
    def start_tutorial(self):
        """Start tutorial execution"""
        logger.info(f"🎓 Starting Tutorial: {self.title}")
        logger.info(f"📝 Description: {self.description}")
        logger.info("=" * 60)
    
    def end_tutorial(self):
        """End tutorial execution"""
        elapsed_time = time.time() - self.start_time
        logger.info("=" * 60)
        logger.info(f"✅ Tutorial Completed: {self.title}")
        logger.info(f"⏱️  Total Time: {elapsed_time:.2f}s")
        logger.info(f"📊 Progress: {self.steps_completed}/{self.total_steps} steps")
        logger.info("=" * 60)


class ZeroKnowledgeProofsTutorial(AEGISTutorial):
    """Tutorial: Zero-Knowledge Proofs for Privacy-Preserving Authentication"""
    
    def __init__(self):
        super().__init__(
            "Zero-Knowledge Proofs for Privacy-Preserving Authentication",
            "Learn how to implement ZKPs for secure node authentication without revealing secrets"
        )
        self.total_steps = 7
        self.security_manager = AdvancedSecurityManager()
    
    async def run_tutorial(self):
        """Run the ZKP tutorial"""
        self.start_tutorial()
        
        # Step 1: Introduction to ZKPs
        self.log_step("Understanding Zero-Knowledge Proofs")
        logger.info("ZKPs allow proving knowledge of a secret without revealing the secret itself.")
        logger.info("They're perfect for authentication where you want to prove identity")
        logger.info("without transmitting passwords or secrets over the network.")
        await asyncio.sleep(1)
        
        # Step 2: Setting up security manager
        self.log_step("Setting up Advanced Security Manager")
        logger.info("Creating security manager with ZKP features enabled...")
        self.security_manager.enabled_features[SecurityFeature.ZERO_KNOWLEDGE_PROOF] = True
        logger.info("✓ Security manager ready")
        await asyncio.sleep(0.5)
        
        # Step 3: Creating a secret
        self.log_step("Creating authentication secret")
        secret = b"my_secure_password_123"
        logger.info(f"Secret created: {secret.decode()}")
        logger.info("This secret will never be transmitted over the network!")
        await asyncio.sleep(0.5)
        
        # Step 4: Generating ZK proof
        self.log_step("Generating Zero-Knowledge Proof")
        logger.info("Creating ZK proof for authentication...")
        proof = self.security_manager.create_zk_proof(
            secret=secret,
            statement="authenticate:user_123",
            verifier_id="auth_server"
        )
        logger.info(f"✓ Proof generated (ID: {proof.proof_id[:8]}...)")
        logger.info(f"  Commitment: {proof.commitment.hex()[:16]}...")
        logger.info(f"  Challenge: {proof.challenge.hex()[:16]}...")
        await asyncio.sleep(1)
        
        # Step 5: Verifying the proof
        self.log_step("Verifying Zero-Knowledge Proof")
        logger.info("Verifying proof with authentication server...")
        is_valid = self.security_manager.verify_zk_proof(
            proof, "authenticate:user_123"
        )
        logger.info(f"✓ Proof verification: {'SUCCESS' if is_valid else 'FAILED'}")
        if is_valid:
            logger.info("🎉 Authentication successful without revealing the secret!")
        await asyncio.sleep(1)
        
        # Step 6: Range proofs
        self.log_step("Creating Range Proofs")
        logger.info("Creating range proof to prove numeric values are within bounds...")
        value = 42
        range_proof = self.security_manager.zk_prover.create_range_proof(
            value=value,
            min_val=0,
            max_val=100,
            verifier_id="range_verifier"
        )
        logger.info(f"✓ Range proof created for value {value}")
        
        # Verify range proof
        is_range_valid = self.security_manager.zk_prover.verify_range_proof(
            range_proof, 0, 100
        )
        logger.info(f"✓ Range proof verification: {'VALID' if is_range_valid else 'INVALID'}")
        await asyncio.sleep(1)
        
        # Step 7: Security considerations
        self.log_step("Security Best Practices")
        logger.info("🔑 ZKP Security Best Practices:")
        logger.info("  1. Use strong, unique secrets for each authentication")
        logger.info("  2. Implement proof expiration (timestamps)")
        logger.info("  3. Use secure random number generation")
        logger.info("  4. Regularly rotate authentication secrets")
        logger.info("  5. Monitor proof verification logs for anomalies")
        await asyncio.sleep(1)
        
        self.end_tutorial()


class HomomorphicEncryptionTutorial(AEGISTutorial):
    """Tutorial: Homomorphic Encryption for Privacy-Preserving Computations"""
    
    def __init__(self):
        super().__init__(
            "Homomorphic Encryption for Privacy-Preserving Computations",
            "Learn how to perform computations on encrypted data without decryption"
        )
        self.total_steps = 6
        self.security_manager = AdvancedSecurityManager()
    
    async def run_tutorial(self):
        """Run the homomorphic encryption tutorial"""
        self.start_tutorial()
        
        # Step 1: Introduction to homomorphic encryption
        self.log_step("Understanding Homomorphic Encryption")
        logger.info("Homomorphic encryption allows computations on encrypted data")
        logger.info("without ever decrypting it. Perfect for privacy-preserving analytics!")
        logger.info("Types: Additive (can add encrypted values) and Multiplicative (can multiply by scalars)")
        await asyncio.sleep(1)
        
        # Step 2: Setting up security manager
        self.log_step("Setting up Homomorphic Encryption")
        logger.info("Enabling homomorphic encryption features...")
        self.security_manager.enabled_features[SecurityFeature.HOMOMORPHIC_ENCRYPTION] = True
        logger.info("✓ Homomorphic encryption enabled")
        await asyncio.sleep(0.5)
        
        # Step 3: Basic encryption/decryption
        self.log_step("Basic Encryption and Decryption")
        logger.info("Encrypting sensitive value...")
        original_value = 123
        encrypted = self.security_manager.encrypt_value(
            original_value, 
            {"type": "salary", "employee": "emp_001"}
        )
        logger.info(f"✓ Value {original_value} encrypted")
        
        # Decrypt
        decrypted = self.security_manager.decrypt_value(encrypted)
        logger.info(f"✓ Decrypted value: {decrypted}")
        logger.info(f"  Match: {'✓' if decrypted == original_value else '✗'}")
        await asyncio.sleep(1)
        
        # Step 4: Homomorphic addition
        self.log_step("Homomorphic Addition")
        logger.info("Performing addition on encrypted values...")
        value1 = 15
        value2 = 25
        
        encrypted1 = self.security_manager.encrypt_value(value1)
        encrypted2 = self.security_manager.encrypt_value(value2)
        
        # Add homomorphically
        encrypted_sum = self.security_manager.add_encrypted_values(encrypted1, encrypted2)
        decrypted_sum = self.security_manager.decrypt_value(encrypted_sum)
        
        logger.info(f"✓ Encrypted addition: {value1} + {value2} = {decrypted_sum}")
        logger.info(f"  Match: {'✓' if decrypted_sum == (value1 + value2) else '✗'}")
        await asyncio.sleep(1)
        
        # Step 5: Homomorphic multiplication
        self.log_step("Homomorphic Multiplication by Scalar")
        logger.info("Multiplying encrypted value by scalar...")
        encrypted_value = self.security_manager.encrypt_value(10)
        scalar = 3
        
        encrypted_product = self.security_manager.multiply_encrypted_by_scalar(
            encrypted_value, scalar
        )
        decrypted_product = self.security_manager.decrypt_value(encrypted_product)
        
        logger.info(f"✓ Encrypted multiplication: {10} × {scalar} = {decrypted_product}")
        logger.info(f"  Match: {'✓' if decrypted_product == (10 * scalar) else '✗'}")
        await asyncio.sleep(1)
        
        # Step 6: Privacy-preserving analytics
        self.log_step("Privacy-Preserving Analytics Example")
        logger.info("Simulating privacy-preserving salary analysis...")
        
        # Simulate encrypted salaries from multiple employees
        employee_salaries = [50000, 60000, 70000, 80000, 90000]
        encrypted_salaries = []
        
        for salary in employee_salaries:
            encrypted_salary = self.security_manager.encrypt_value(salary)
            encrypted_salaries.append(encrypted_salary)
        
        # Homomorphically compute total
        total_encrypted = encrypted_salaries[0]
        for encrypted_salary in encrypted_salaries[1:]:
            total_encrypted = self.security_manager.add_encrypted_values(
                total_encrypted, encrypted_salary
            )
        
        total_salary = self.security_manager.decrypt_value(total_encrypted)
        expected_total = sum(employee_salaries)
        
        logger.info(f"✓ Encrypted salary total: ${total_salary:,}")
        logger.info(f"  Expected total: ${expected_total:,}")
        logger.info(f"  Match: {'✓' if total_salary == expected_total else '✗'}")
        
        # Compute average without revealing individual salaries
        average_salary = total_salary / len(employee_salaries)
        logger.info(f"✓ Average salary (computed on encrypted data): ${average_salary:,.0f}")
        await asyncio.sleep(1)
        
        self.end_tutorial()


class SecureMPCTutorial(AEGISTutorial):
    """Tutorial: Secure Multi-Party Computation"""
    
    def __init__(self):
        super().__init__(
            "Secure Multi-Party Computation",
            "Learn how to perform secure computations with multiple parties without revealing inputs"
        )
        self.total_steps = 5
        self.security_manager = AdvancedSecurityManager()
    
    async def run_tutorial(self):
        """Run the SMC tutorial"""
        self.start_tutorial()
        
        # Step 1: Introduction to SMC
        self.log_step("Understanding Secure Multi-Party Computation")
        logger.info("SMC enables multiple parties to jointly compute a function")
        logger.info("without revealing their private inputs to each other.")
        logger.info("Perfect for collaborative analytics while preserving privacy!")
        await asyncio.sleep(1)
        
        # Step 2: Setting up SMC
        self.log_step("Setting up Secure Multi-Party Computation")
        logger.info("Enabling SMC features...")
        self.security_manager.enabled_features[SecurityFeature.SECURE_MULTI_PARTY_COMPUTATION] = True
        logger.info("✓ SMC enabled")
        
        # Add parties
        parties = ["bank_a", "bank_b", "bank_c", "bank_d", "bank_e"]
        for party in parties:
            self.security_manager.add_party_to_smc(party)
            logger.info(f"  Added party: {party}")
        await asyncio.sleep(1)
        
        # Step 3: Secret sharing
        self.log_step("Secret Sharing with Shamir's Scheme")
        logger.info("Splitting secret among parties using Shamir's Secret Sharing...")
        secret = 12345
        threshold = 3  # Need 3 out of 5 parties to reconstruct
        
        shares = self.security_manager.generate_secret_shares(secret, threshold)
        logger.info(f"✓ Secret {secret} split into {len(shares)} shares")
        logger.info(f"  Threshold: {threshold} parties required for reconstruction")
        
        # Display some shares
        for i, (party_id, share) in enumerate(list(shares.items())[:3]):
            logger.info(f"  Share {i+1}: Party {party_id} -> ({share[0]}, {share[1]})")
        await asyncio.sleep(1)
        
        # Step 4: Secret reconstruction
        self.log_step("Reconstructing Secret from Shares")
        logger.info("Reconstructing secret from subset of shares...")
        
        # Use only threshold number of shares
        subset_shares = dict(list(shares.items())[:threshold])
        reconstructed = self.security_manager.reconstruct_secret_from_shares(subset_shares)
        
        logger.info(f"✓ Reconstructed secret: {reconstructed}")
        logger.info(f"  Original secret: {secret}")
        logger.info(f"  Match: {'✓' if reconstructed == secret else '✗'}")
        await asyncio.sleep(1)
        
        # Step 5: Secure aggregation example
        self.log_step("Secure Aggregation Example")
        logger.info("Simulating secure salary aggregation across banks...")
        
        # Each bank has private salary data
        bank_data = {
            "bank_a": 5000000,  # $5M total salaries
            "bank_b": 7500000,  # $7.5M total salaries
            "bank_c": 3200000,  # $3.2M total salaries
            "bank_d": 9800000,  # $9.8M total salaries
            "bank_e": 4500000   # $4.5M total salaries
        }
        
        # Each bank encrypts their data
        encrypted_bank_data = {}
        for bank_id, total_salaries in bank_data.items():
            encrypted_data = self.security_manager.encrypt_value(
                total_salaries,
                {"bank": bank_id, "data_type": "total_salaries"}
            )
            encrypted_bank_data[bank_id] = encrypted_data
            logger.info(f"  {bank_id} encrypted ${total_salaries:,}")
        
        # Banks generate shares of their encrypted data
        bank_shares = {}
        for bank_id, encrypted_data in encrypted_bank_data.items():
            decrypted_data = self.security_manager.decrypt_value(encrypted_data)
            shares = self.security_manager.generate_secret_shares(decrypted_data, threshold=3)
            bank_shares[bank_id] = shares
            logger.info(f"  {bank_id} shared encrypted data")
        
        # Secure aggregation - each party combines their shares
        aggregated_shares = {}
        for i in range(threshold):
            combined_share_x = i + 1
            combined_share_y = 0
            
            # Sum all shares at position i+1
            for bank_id, shares in bank_shares.items():
                if bank_id in shares:
                    share_x, share_y = shares[bank_id]
                    if share_x == combined_share_x:
                        combined_share_y += share_y
            
            aggregated_shares[f"aggregator_{i+1}"] = (combined_share_x, combined_share_y)
        
        # Reconstruct aggregated result
        total_aggregated = self.security_manager.reconstruct_secret_from_shares(aggregated_shares)
        expected_total = sum(bank_data.values())
        
        logger.info(f"✓ Secure aggregation result: ${total_aggregated:,}")
        logger.info(f"  Expected total: ${expected_total:,}")
        logger.info(f"  Match: {'✓' if total_aggregated == expected_total else '✗'}")
        
        logger.info("🎉 No individual bank revealed their salary data!")
        await asyncio.sleep(1)
        
        self.end_tutorial()


class DifferentialPrivacyTutorial(AEGISTutorial):
    """Tutorial: Differential Privacy for Statistical Data Protection"""
    
    def __init__(self):
        super().__init__(
            "Differential Privacy for Statistical Data Protection",
            "Learn how to protect individual privacy in statistical queries"
        )
        self.total_steps = 6
        self.security_manager = AdvancedSecurityManager()
    
    async def run_tutorial(self):
        """Run the differential privacy tutorial"""
        self.start_tutorial()
        
        # Step 1: Introduction to differential privacy
        self.log_step("Understanding Differential Privacy")
        logger.info("Differential privacy adds mathematical guarantees to protect")
        logger.info("individual privacy in statistical databases and queries.")
        logger.info("Perfect for releasing useful statistics without compromising privacy!")
        await asyncio.sleep(1)
        
        # Step 2: Setting up differential privacy
        self.log_step("Setting up Differential Privacy")
        logger.info("Enabling differential privacy features...")
        self.security_manager.enabled_features[SecurityFeature.DIFFERENTIAL_PRIVACY] = True
        logger.info("✓ Differential privacy enabled")
        await asyncio.sleep(0.5)
        
        # Step 3: Count queries with privacy
        self.log_step("Private Count Queries")
        logger.info("Performing count query with differential privacy...")
        true_count = 1000
        private_count = self.security_manager.privatize_data(true_count, "count")
        
        logger.info(f"✓ True count: {true_count}")
        logger.info(f"✓ Private count: {private_count:.0f}")
        logger.info(f"  Difference: {abs(true_count - private_count):.0f}")
        await asyncio.sleep(1)
        
        # Step 4: Sum queries with privacy
        self.log_step("Private Sum Queries")
        logger.info("Performing sum query with differential privacy...")
        true_sum = 5000000.0  # $5M total
        max_value = 100000.0  # Max salary $100K
        
        private_sum = self.security_manager.privatize_data(
            true_sum, "sum", max_value=max_value
        )
        
        logger.info(f"✓ True sum: ${true_sum:,.0f}")
        logger.info(f"✓ Private sum: ${private_sum:,.0f}")
        logger.info(f"  Difference: ${abs(true_sum - private_sum):,.0f}")
        await asyncio.sleep(1)
        
        # Step 5: Mean queries with privacy
        self.log_step("Private Mean Queries")
        logger.info("Performing mean query with differential privacy...")
        
        # Generate sample data
        salaries = [random.uniform(30000, 150000) for _ in range(1000)]
        true_mean = sum(salaries) / len(salaries)
        
        private_mean = self.security_manager.privatize_data(salaries, "mean")
        
        logger.info(f"✓ True mean salary: ${true_mean:,.0f}")
        logger.info(f"✓ Private mean salary: ${private_mean:,.0f}")
        logger.info(f"  Difference: ${abs(true_mean - private_mean):,.0f}")
        await asyncio.sleep(1)
        
        # Step 6: Privacy budget management
        self.log_step("Privacy Budget Management")
        logger.info("Managing privacy budget for multiple queries...")
        
        # Simulate multiple queries with budget tracking
        epsilon_total = 1.0
        queries_remaining = 5
        epsilon_per_query = epsilon_total / queries_remaining
        
        logger.info(f"📊 Total privacy budget: ε = {epsilon_total}")
        logger.info(f"  Queries remaining: {queries_remaining}")
        logger.info(f"  Epsilon per query: {epsilon_per_query:.2f}")
        
        # Perform queries
        for i in range(queries_remaining):
            query_value = random.randint(100, 1000)
            private_value = self.security_manager.privatize_data(query_value, "count")
            logger.info(f"  Query {i+1}: {query_value} → {private_value:.0f}")
        
        logger.info("🔒 Privacy budget consumed appropriately")
        logger.info("💡 Always track privacy budget to prevent over-exposure!")
        await asyncio.sleep(1)
        
        self.end_tutorial()


class ComprehensiveSecurityTutorial(AEGISTutorial):
    """Tutorial: Comprehensive Security Integration"""
    
    def __init__(self):
        super().__init__(
            "Comprehensive Security Integration",
            "Learn how to combine all advanced security features for maximum protection"
        )
        self.total_steps = 8
        self.security_manager = AdvancedSecurityManager()
    
    async def run_tutorial(self):
        """Run the comprehensive security tutorial"""
        self.start_tutorial()
        
        # Step 1: Enabling all security features
        self.log_step("Enabling All Advanced Security Features")
        logger.info("Activating zero-knowledge proofs, homomorphic encryption,")
        logger.info("secure multi-party computation, and differential privacy...")
        
        for feature in SecurityFeature:
            self.security_manager.enabled_features[feature] = True
            logger.info(f"  ✓ {feature.value} enabled")
        await asyncio.sleep(1)
        
        # Step 2: Privacy-preserving federated learning
        self.log_step("Privacy-Preserving Federated Learning")
        logger.info("Setting up federated learning with all security features...")
        
        # Create federated learning config
        config = FederatedConfig()
        logger.info("✓ Federated learning configuration created")
        
        # Simulate secure client authentication
        client_secret = b"fl_client_secret_123"
        client_proof = self.security_manager.create_zk_proof(
            secret=client_secret,
            statement="authenticate:fl_client_001",
            verifier_id="fl_coordinator"
        )
        logger.info("✓ Client authenticated with zero-knowledge proof")
        
        # Simulate encrypted model updates
        model_update = 0.12345  # Gradient update
        encrypted_update = self.security_manager.encrypt_value(
            int(model_update * 100000),  # Scale for integer encryption
            {"type": "gradient", "client": "fl_client_001"}
        )
        logger.info("✓ Model update encrypted for privacy")
        
        # Add differential privacy to updates
        noisy_update = self.security_manager.privatize_data(model_update, "mean")
        logger.info(f"✓ Model update privatized: {model_update:.5f} → {noisy_update:.5f}")
        await asyncio.sleep(1)
        
        # Step 3: Secure consensus with authentication
        self.log_step("Secure Consensus with Zero-Knowledge Authentication")
        logger.info("Setting up blockchain consensus with ZK authentication...")
        
        # Create consensus node with ZK authentication
        private_key = ed25519.Ed25519PrivateKey.generate()
        consensus = HybridConsensus(
            node_id="consensus_node_001",
            private_key=private_key
        )
        logger.info("✓ Consensus node created")
        
        # Node authentication with ZK proof
        node_secret = b"consensus_node_secret_456"
        node_proof = self.security_manager.create_zk_proof(
            secret=node_secret,
            statement="authenticate:consensus_node_001",
            verifier_id="consensus_coordinator"
        )
        logger.info("✓ Node authenticated for consensus participation")
        await asyncio.sleep(1)
        
        # Step 4: Secure multi-party model aggregation
        self.log_step("Secure Multi-Party Model Aggregation")
        logger.info("Aggregating model updates from multiple parties securely...")
        
        # Simulate multiple parties with encrypted updates
        parties = ["party_a", "party_b", "party_c", "party_d"]
        encrypted_updates = {}
        
        for party in parties:
            update_value = random.uniform(-0.1, 0.1)
            encrypted_update = self.security_manager.encrypt_value(
                int(update_value * 100000),
                {"party": party, "update_type": "gradient"}
            )
            encrypted_updates[party] = encrypted_update
            logger.info(f"  {party} encrypted update: {update_value:.5f}")
        
        # Homomorphically add all updates
        if encrypted_updates:
            aggregated_encrypted = list(encrypted_updates.values())[0]
            for encrypted_update in list(encrypted_updates.values())[1:]:
                aggregated_encrypted = self.security_manager.add_encrypted_values(
                    aggregated_encrypted, encrypted_update
                )
            
            # Decrypt aggregated result
            aggregated_result = self.security_manager.decrypt_value(aggregated_encrypted)
            aggregated_decimal = aggregated_result / 100000.0
            
            logger.info(f"✓ Secure aggregation result: {aggregated_decimal:.5f}")
        await asyncio.sleep(1)
        
        # Step 5: Privacy-preserving analytics
        self.log_step("Privacy-Preserving Analytics")
        logger.info("Performing analytics on sensitive data with privacy guarantees...")
        
        # Simulate sensitive employee data
        employee_data = [
            {"id": f"emp_{i:04d}", "salary": random.randint(30000, 150000), "department": random.choice(["eng", "sales", "marketing", "hr"])}
            for i in range(1000)
        ]
        
        # Compute private statistics
        salaries = [emp["salary"] for emp in employee_data]
        true_mean_salary = sum(salaries) / len(salaries)
        private_mean_salary = self.security_manager.privatize_data(salaries, "mean")
        
        # Department-wise analysis
        departments = {}
        for emp in employee_data:
            dept = emp["department"]
            if dept not in departments:
                departments[dept] = []
            departments[dept].append(emp["salary"])
        
        logger.info("📊 Private Department Salary Analysis:")
        for dept, dept_salaries in departments.items():
            true_dept_mean = sum(dept_salaries) / len(dept_salaries)
            private_dept_mean = self.security_manager.privatize_data(dept_salaries, "mean")
            logger.info(f"  {dept.upper()}: ${true_dept_mean:,.0f} → ${private_dept_mean:,.0f}")
        
        logger.info(f"  Overall: ${true_mean_salary:,.0f} → ${private_mean_salary:,.0f}")
        await asyncio.sleep(1)
        
        # Step 6: Secure data sharing
        self.log_step("Secure Data Sharing Between Organizations")
        logger.info("Sharing data between organizations without revealing raw data...")
        
        # Organization A has dataset A
        org_a_data = [random.randint(1, 100) for _ in range(500)]
        org_a_secret = 12345
        
        # Organization B has dataset B
        org_b_data = [random.randint(1, 100) for _ in range(500)]
        org_b_secret = 67890
        
        # Each organization encrypts their data
        encrypted_a = self.security_manager.encrypt_value(org_a_secret)
        encrypted_b = self.security_manager.encrypt_value(org_b_secret)
        
        # Homomorphically compute sum
        encrypted_sum = self.security_manager.add_encrypted_values(encrypted_a, encrypted_b)
        decrypted_sum = self.security_manager.decrypt_value(encrypted_sum)
        
        logger.info(f"✓ Organization A data: {org_a_secret}")
        logger.info(f"✓ Organization B data: {org_b_secret}")
        logger.info(f"✓ Secure sum: {decrypted_sum}")
        logger.info("🎉 Neither organization saw the other's raw data!")
        await asyncio.sleep(1)
        
        # Step 7: Compliance and auditing
        self.log_step("Compliance and Security Auditing")
        logger.info("Ensuring compliance with privacy regulations...")
        
        # Get security statistics
        stats = self.security_manager.get_security_stats()
        logger.info("🛡️  Security Statistics:")
        logger.info(f"  ZK Proofs Generated: {stats['zk_proofs_generated']}")
        logger.info(f"  SMC Parties: {stats['parties_in_smc']}")
        logger.info(f"  Privacy Epsilon: {stats['privacy_parameters']['epsilon']}")
        
        # Verify all features are enabled
        enabled_features = stats["enabled_features"]
        logger.info("✅ Enabled Security Features:")
        for feature, enabled in enabled_features.items():
            status = "✓" if enabled else "✗"
            logger.info(f"  {status} {feature}")
        await asyncio.sleep(1)
        
        # Step 8: Best practices summary
        self.log_step("Security Best Practices Summary")
        logger.info("🔐 Comprehensive Security Best Practices:")
        logger.info("  1. Layer multiple security mechanisms (defense in depth)")
        logger.info("  2. Regularly rotate secrets and keys")
        logger.info("  3. Monitor security metrics and alerts")
        logger.info("  4. Implement proper access controls")
        logger.info("  5. Maintain audit trails for compliance")
        logger.info("  6. Educate users on security practices")
        logger.info("  7. Stay updated with security patches")
        logger.info("  8. Conduct regular security assessments")
        await asyncio.sleep(1)
        
        self.end_tutorial()


async def run_all_tutorials():
    """Run all AEGIS security tutorials"""
    logger.info("🚀 Starting AEGIS Comprehensive Security Tutorials")
    logger.info("=" * 80)
    
    tutorials = [
        ZeroKnowledgeProofsTutorial(),
        HomomorphicEncryptionTutorial(),
        SecureMPCTutorial(),
        DifferentialPrivacyTutorial(),
        ComprehensiveSecurityTutorial()
    ]
    
    for tutorial in tutorials:
        try:
            await tutorial.run_tutorial()
            logger.info("\n" + "="*80 + "\n")
            await asyncio.sleep(2)  # Pause between tutorials
        except Exception as e:
            logger.error(f"Tutorial failed: {e}")
    
    logger.info("🎓 All AEGIS Security Tutorials Completed!")
    logger.info("🔒 You are now ready to implement advanced security features!")


if __name__ == "__main__":
    asyncio.run(run_all_tutorials())
