# test_quantum_tokenomics.py
# Production-ready test suite for Quantum Tokenomics (T0/T1), CAL, EMA Ψ, anti-gaming, and ZKP

import pytest
import math
from datetime import datetime, timedelta

# Assuming these modules exist in your codebase
from quantum_currency.tokenomics import MiningAgent, TokenLedger
from quantum_currency.cal import CAL
from quantum_currency.metrics import MetricsExporter
from quantum_currency.zk import ZKVerifier

# -------------------------
# 1️⃣ Unit Tests: Minting, EMA Ψ, Decay Factors
# -------------------------

def test_c_hat_decay_factor():
    """Verify decay factor incentivizes low-Ĉ(t) recovery more than high-Ĉ(t)."""
    low_prev, low_current = 0.50, 0.51
    high_prev, high_current = 0.95, 0.96
    delta_low = low_current - low_prev
    delta_high = high_current - high_prev

    reward_low = MiningAgent.calculate_t0_reward(delta_low, low_current, stake=100)
    reward_high = MiningAgent.calculate_t0_reward(delta_high, high_current, stake=100)

    assert reward_low > reward_high, "Reward should be higher for low Ĉ(t) scenario"

def test_stake_weighting():
    """Verify quadratic stake weighting for MintT0."""
    delta_c = 0.01
    validator_a_stake = 100
    validator_b_stake = 10000
    c_current = 0.9

    reward_a = MiningAgent.calculate_t0_reward(delta_c, c_current, validator_a_stake)
    reward_b = MiningAgent.calculate_t0_reward(delta_c, c_current, validator_b_stake)

    expected_ratio = math.sqrt(validator_b_stake) / math.sqrt(validator_a_stake)
    actual_ratio = reward_b / reward_a
    assert math.isclose(actual_ratio, expected_ratio, rel_tol=1e-5), "Quadratic stake weighting failed"

def test_ema_psi_dampening():
    """Verify EMA Ψ dampens sudden high-to-low spike attempts."""
    psi_values = [0.95, 0.95, 0.0]  # Adversarial spike then drop
    ema_result = MiningAgent.calculate_ema_psi(psi_values, smoothing=0.5)
    assert ema_result < 0.7, "EMA Ψ did not dampen adversarial spike correctly"

# -------------------------
# 2️⃣ Integration & Adversarial Tests
# -------------------------

def test_memory_node_spam_attack():
    """Verify T1 mint only occurs when HE successfully uses memory nodes."""
    nodes = [{"RΦV": 0.1, "used": False} for _ in range(1000)]
    t1_reward = MiningAgent.process_memory_nodes(nodes)
    assert t1_reward == 0, "T1 should be zero if nodes are unused"

def test_governance_slashing_effect():
    """Verify stake reduction affects next epoch minting proportionally."""
    validator = {"stake": 1000, "psi_ema": 0.9}
    MiningAgent.record_stake_change(validator, new_stake=500)  # 50% slashed
    reward = MiningAgent.calculate_t0_reward(delta_c=0.01, c_current=0.9, stake=validator["stake"])
    expected_ratio = 500 / 1000
    assert math.isclose(reward / 100, expected_ratio, rel_tol=1e-5), "T0 mint did not adjust after slashing"

def test_emergency_cal_state_mint_block():
    """Verify no minting occurs when CAL is in emergency state."""
    cal = CAL()
    cal.set_emergency_state(True)
    reward = MiningAgent.calculate_t0_reward(delta_c=0.01, c_current=0.5, stake=100, cal=cal)
    assert reward == 0, "Minting should be blocked during emergency CAL state"

# -------------------------
# 3️⃣ ZKP Verification Tests
# -------------------------

def test_zkp_generation_and_verification():
    """Verify ZKP can be generated and verified for a validator reward."""
    ledger = TokenLedger()
    validator_id = "v1"
    epoch_id = 10
    proof = ledger.generate_coherence_mint_proof(validator_id, epoch_id)
    assert ZKVerifier.verify(proof, public_inputs={"validator_id": validator_id, "epoch_id": epoch_id}), \
        "ZKP verification failed"

# -------------------------
# 4️⃣ Metrics & Observability Tests
# -------------------------

def test_metrics_exporter_updates():
    """Verify observability metrics reflect system actions."""
    exporter = MetricsExporter()
    exporter.record_reward_prevented(10)
    exporter.record_lockup_balance(500)
    assert exporter.metrics["qc_gaming_reward_prevented"] == 10
    assert exporter.metrics["qc_lockup_balance"] == 500

# -------------------------
# 5️⃣ Stress Test Simulation
# -------------------------

def test_adversarial_validator_pool():
    """Simulate 50+ validators with mixed behavior under ΔĈ(t) changes."""
    validators = [{"stake": 100 + i*10, "psi_ema": 0.9 if i % 2 == 0 else 0.5} for i in range(50)]
    delta_c = 0.02
    c_current = 0.88

    rewards = [MiningAgent.calculate_t0_reward(delta_c, c_current, v["stake"], psi_ema=v["psi_ema"]) for v in validators]
    # Ensure no reward exceeds expected maximum
    assert max(rewards) < 1000, "Reward exceeded sanity check under adversarial conditions"

# -------------------------
# Run with pytest
# -------------------------
if __name__ == "__main__":
    pytest.main(["-v", "test_quantum_tokenomics.py"])
