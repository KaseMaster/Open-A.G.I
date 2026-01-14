# CAL Attunement Validator System with 5-Token Integration

## Overview

This document describes the integration of the 5-token ecosystem into the Quantum Currency Coherence System, specifically how tokens T1-T5 interact with the Attunement-Based Coherence Layer (CAL) and validator network to incentivize, regulate, and stabilize the network.

## Token Roles and Mechanics

### T1 - Validator Stake Token
- **Role**: Core staking for validators
- **Mechanics**: Determines voting power and is dynamically weighted by Ψ. Auto-slashing reduces T1 if coherence drops.
- **Implementation**: Integrated into [Validator.staking](file:///D:/AI%20AGENT%20CODERV1/QUANTUM%20CURRENCY/Open-A.G.I/quantum-currency/src/core/validator_staking.py#L47-L73) class with [t1_balance](file:///D:/AI%20AGENT%20CODERV1/QUANTUM%20CURRENCY/Open-A.G.I/quantum-currency/src/core/validator_staking.py#L64-L64) and [t1_staked](file:///D:/AI%20AGENT%20CODERV1/QUANTUM%20CURRENCY/Open-A.G.I/quantum-currency/src/core/validator_staking.py#L65-L65) attributes

### T2 - Reward Token
- **Role**: Dynamic reward for high coherence
- **Mechanics**: Distributed based on Ψ and network coherence; bonus multiplier during deficit mode.
- **Implementation**: Handled by [AttunementRewardEngine.distribute_t2_rewards()](file:///D:/AI%20AGENT%20CODERV1/QUANTUM%20CURRENCY/Open-A.G.I/quantum-currency/src/reward/attunement_reward_engine.py#L51-L89) method

### T3 - Governance Token
- **Role**: Voting on protocol upgrades & attunement parameters
- **Mechanics**: Each T3 represents weighted voting power; tied to staked T1 and coherence metrics.
- **Implementation**: Integrated with governance system in [src/governance/voting.py](file:///D:/AI%20AGENT%20CODERV1/QUANTUM%20CURRENCY/Open-A.G.I/quantum-currency/src/governance/voting.py)

### T4 - Attunement Boost Token
- **Role**: Temporary coherence optimization
- **Mechanics**: Validators can burn T4 to temporarily boost their Ψ score or shorten λ(t) interval in critical conditions.
- **Implementation**: [Validator.apply_t4_boost()](file:///D:/AI%20AGENT%20CODERV1/QUANTUM%20CURRENCY/Open-A.G.I/quantum-currency/src/core/validator_staking.py#L75-L100) and [calculate_psi_with_boost()](file:///D:/AI%20AGENT%20CODERV1/QUANTUM%20CURRENCY/Open-A.G.I/quantum-currency/src/core/validator_staking.py#L102-L123) methods

### T5 - Memory Incentive Token
- **Role**: Reward for contributing high-RΦV memory nodes
- **Mechanics**: Allocated when nodes improve CAL metrics (e.g., nodes with λ_node > 0.9), stored in Harmonic Memory DB.
- **Implementation**: [AttunementRewardEngine.calculate_t5_rewards()](file:///D:/AI%20AGENT%20CODERV1/QUANTUM%20CURRENCY/Open-A.G.I/quantum-currency/src/reward/attunement_reward_engine.py#L91-L115) method

## Implementation Details

### Validator Module Updates
The [Validator](file:///D:/AI%20AGENT%20CODERV1/QUANTUM%20CURRENCY/Open-A.G.I/quantum-currency/src/core/validator_staking.py#L47-L73) class in [src/core/validator_staking.py](file:///D:/AI%20AGENT%20CODERV1/QUANTUM%20CURRENCY/Open-A.G.I/quantum-currency/src/core/validator_staking.py) has been extended with:
- Token balances for T1-T5
- Psi_new calculation including T4 boosts: `Psi_new = base_psi + 0.05 * T4_boost`
- Auto-slashing logic for T1 and T4 when Ψ < 0.7

### Reward Engine
The [AttunementRewardEngine](file:///D:/AI%20AGENT%20CODERV1/QUANTUM%20CURRENCY/Open-A.G.I/quantum-currency/src/reward/attunement_reward_engine.py#L35-L199) in [src/reward/attunement_reward_engine.py](file:///D:/AI%20AGENT%20CODERV1/QUANTUM%20CURRENCY/Open-A.G.I/quantum-currency/src/reward/attunement_reward_engine.py) handles:
- Dynamic T2 reward distribution with deficit multipliers
- T5 reward calculation based on memory node contributions
- Epoch-based reward logging and tracking

### CAL Integration
The [CALEngine](file:///D:/AI%20AGENT%20CODERV1/QUANTUM%20CURRENCY/Open-A.G.I/quantum-currency/src/core/cal_engine.py#L66-L694) in [src/core/cal_engine.py](file:///D:/AI%20AGENT%20CODERV1/QUANTUM%20CURRENCY/Open-A.G.I/quantum-currency/src/core/cal_engine.py) includes:
- T4/boost influence on λ(t) recalibration (shortens voting interval when T4 tokens are applied)
- T5-memory contributions in Ĉ(t) calculation: `C_hat = f(C_hat_core, Avg(lambda_node), sum(T5_impact))`

### Smart Token Ledger
The [TokenLedger](file:///D:/AI%20AGENT%20CODERV1/QUANTUM%20CURRENCY/Open-A.G.I/quantum-currency/src/tokens/token_manager.py#L27-L308) in [src/tokens/token_manager.py](file:///D:/AI%20AGENT%20CODERV1/QUANTUM%20CURRENCY/Open-A.G.I/quantum-currency/src/tokens/token_manager.py) provides:
- Balance tracking for T1-T5 tokens
- Staking, slashing, reward, and memory incentive logic
- Immutable audit log and transaction history

### Prometheus Metrics
Custom Prometheus metrics have been added in [src/monitoring/metrics_exporter.py](file:///D:/AI%20AGENT%20CODERV1/QUANTUM%20CURRENCY/Open-A.G.I/quantum-currency/src/monitoring/metrics_exporter.py):
- `qc_token_T1_staked_total`
- `qc_token_T2_rewards_epoch`
- `qc_token_T4_boosts_active`
- `qc_token_T5_memory_contributions`

### Dashboard Integration
Grafana panels have been added to visualize:
- T1 staking distribution per validator
- T2 rewards over time
- T4 boost utilization
- T5 memory contribution leaderboard

## Testing and Verification

The integration has been tested using the [run_token_orchestration.py](file:///D:/AI%20AGENT%20CODERV1/QUANTUM%20CURRENCY/Open-A.G.I/quantum-currency/run_token_orchestration.py) script which verifies:
- T1 staking, slashing, and Ψ correlation
- T2 reward allocation, including deficit multipliers
- T3 governance voting tied to staked T1 and coherence performance
- T4 boosts temporarily influencing Ψ and λ(t)
- T5 memory contributions increasing Ĉ(t)

All metrics are available in `/metrics` endpoint and Grafana dashboards.