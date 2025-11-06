"""
Integration Tests for Quantum Currency System
"""
import sys
import os
import unittest

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

# Use relative imports for the modules
from core.harmonic_validation import (
    HarmonicSnapshot, 
    make_snapshot, 
    compute_coherence_score
)
from core.token_rules import validate_harmonic_tx, apply_token_effects


class TestQuantumCurrencyIntegration(unittest.TestCase):
    """Test suite for quantum currency integration"""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.test_times = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        self.test_values = [1.0, 1.1, 1.2, 1.1, 1.0, 0.9]
        
        # Create test validator IDs and keys
        self.validator_ids = ["validator-001", "validator-002", "validator-003"]
        self.validator_keys = ["key-001", "key-002", "key-003"]
        
        # Initialize test ledger
        self.test_ledger = {
            "balances": {
                "validator-001": {"FLX": 1000.0, "CHR": 500.0},
                "validator-002": {"FLX": 1000.0, "CHR": 500.0},
                "validator-003": {"FLX": 1000.0, "CHR": 500.0}
            },
            "chr": {}
        }
        
    def test_full_harmonic_validation_cycle(self):
        """Test a full harmonic validation cycle with multiple nodes"""
        # Step 1: Generate snapshots from all validators
        print("Step 1: Generating snapshots...")
        snapshots = []
        for i, (validator_id, secret_key) in enumerate(zip(self.validator_ids, self.validator_keys)):
            # Slightly modify values for each validator to simulate real-world differences
            modified_values = [v + i * 0.01 for v in self.test_values]
            
            snapshot = make_snapshot(
                node_id=validator_id,
                times=self.test_times,
                values=modified_values,
                secret_key=secret_key
            )
            snapshots.append(snapshot)
            
        # Step 2: Compute coherence scores
        print("Step 2: Computing coherence scores...")
        for i, local_snapshot in enumerate(snapshots):
            # Use other snapshots as remote snapshots
            remote_snapshots = [snapshots[j] for j in range(len(snapshots)) if j != i]
            
            coherence_score = compute_coherence_score(local_snapshot, remote_snapshots)
            print(f"   {local_snapshot.node_id} coherence score: {coherence_score:.4f}")
            
            # Store coherence score in ledger
            self.test_ledger["chr"][local_snapshot.node_id] = coherence_score
            
        # Step 3: Validate and apply token effects
        print("Step 3: Validating transactions and applying token effects...")
        for i, local_snapshot in enumerate(snapshots):
            # Create transaction data
            remote_snapshots = [snapshots[j].__dict__ for j in range(len(snapshots)) if j != i]
            
            # Test FLX mint transaction
            tx_mint = {
                "local_snapshot": local_snapshot.__dict__,
                "snapshot_bundle": remote_snapshots,
                "sender": local_snapshot.node_id,
                "receiver": local_snapshot.node_id,
                "amount": 100.0,
                "token": "FLX",
                "action": "mint",
                "aggregated_cs": self.test_ledger["chr"][local_snapshot.node_id],
                "sender_chr": self.test_ledger["chr"][local_snapshot.node_id]
            }
            
            # Validate transaction
            is_valid = validate_harmonic_tx(tx_mint)
            
            print(f"   FLX mint transaction from {local_snapshot.node_id}: {'Valid' if is_valid else 'Invalid'}")
            
            # Apply token effects if valid
            if is_valid:
                initial_flx = self.test_ledger["balances"][local_snapshot.node_id]["FLX"]
                apply_token_effects(self.test_ledger, tx_mint)
                final_flx = self.test_ledger["balances"][local_snapshot.node_id]["FLX"]
                self.assertGreater(final_flx, initial_flx)
                
        # Step 4: Check final ledger state
        print("Step 4: Checking final ledger state...")
        for validator_id in self.validator_ids:
            chr_score = self.test_ledger["chr"].get(validator_id, 0.0)
            flx_balance = self.test_ledger["balances"].get(validator_id, {}).get("FLX", 0.0)
            chr_balance = self.test_ledger["balances"].get(validator_id, {}).get("CHR", 0.0)
            
            print(f"   {validator_id}: CHR={chr_score:.4f}, FLX={flx_balance:.2f}, CHR={chr_balance:.2f}")
            
        # Verify that all validators have positive balances
        for validator_id in self.validator_ids:
            flx_balance = self.test_ledger["balances"].get(validator_id, {}).get("FLX", 0.0)
            chr_balance = self.test_ledger["balances"].get(validator_id, {}).get("CHR", 0.0)
            self.assertGreater(flx_balance, 0.0)
            self.assertGreater(chr_balance, 0.0)

    def test_coherence_score_calculation(self):
        """Test coherence score calculation between snapshots"""
        # Create two snapshots with similar data
        snapshot1 = make_snapshot(
            node_id="validator-1",
            times=self.test_times,
            values=self.test_values,
            secret_key="key-1"
        )
        
        # Slightly modified values for the second snapshot
        modified_values = [v + 0.05 for v in self.test_values]
        snapshot2 = make_snapshot(
            node_id="validator-2",
            times=self.test_times,
            values=modified_values,
            secret_key="key-2"
        )
        
        # Calculate coherence score
        coherence_score = compute_coherence_score(snapshot1, [snapshot2])
        
        # Verify coherence score is within expected range (0.0 to 1.0)
        self.assertGreaterEqual(coherence_score, 0.0)
        self.assertLessEqual(coherence_score, 1.0)
        
        print(f"Coherence score between similar snapshots: {coherence_score:.4f}")


if __name__ == '__main__':
    unittest.main()