# Governance Coherence Guide

## Overview

This document describes the implementation of the CHR-Weighted Ψ-Gated Governance System as part of Phases 6 and 7 of the Quantum Currency Framework. The governance layer translates coherence ethics into structural decision-making, ensuring only validators with high coherence integrity (Ψ ≥ 0.85) can participate in governance.

## Implementation Details

### Validator Eligibility

Validators must maintain a high coherence score to participate in governance:
- Minimum Ψ threshold: 0.85
- Eligibility is determined by a rolling 90-day average of Ψ scores
- Validators with insufficient Ψ scores are automatically excluded from governance participation

### CHR-Weighted Quadratic Voting

The voting power of eligible validators is calculated using the formula:
```
vote_power = √(CHR_balance) × Ψ
```

This approach combines:
- Economic density (CHR token balance) as a measure of stake
- Ethical frequency (Ψ score) as a measure of coherence alignment

### Implementation Components

1. **Extended Validator Class** (`src/core/validator_staking.py`)
   - Added `psi_score_history` to track historical coherence scores
   - Added `eligible_for_governance` flag to indicate participation status
   - Added `chr_balance` to track CHR token holdings for voting power calculation

2. **Governance Voting System** (`src/governance/voting.py`)
   - Proposal creation and management
   - Quadratic voting implementation with CHR-weighted vote power
   - Ψ-gated eligibility checks with rolling average calculation
   - Integration with CAL feedback for post-approval adjustments

## Integration with CAL Feedback

Governance decisions are integrated with the Coherence Attunement Layer (CAL) feedback mechanisms:
- Approved proposals can adjust λ(L) or harmonic coefficients
- Governance updates are stored as events in the coherence ledger
- Real-time monitoring of governance impact on system coherence

## Testing

Comprehensive tests have been implemented to verify:
- Validator eligibility based on Ψ scores
- Correct calculation of CHR-weighted vote power
- Proper handling of proposal creation and voting
- Integration with the extended validator staking system

## Future Enhancements

- Dashboard integration for governance analytics
- Advanced proposal lifecycle management
- Delegation mechanisms for validator voting power