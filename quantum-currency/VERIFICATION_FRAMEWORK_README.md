# Quantum Currency Verification Framework

This framework provides a full, production-ready verification and AGI integration system for the Quantum Currency network. It combines all hardened Quantum Tokenomics checks with real-time Open AGI monitoring, adaptive control, and reporting.

## Architecture Overview

```
[Quantum Network]
       |
       v
[Metrics Collector] --> Prometheus API --> [Open AGI Agent] --> Recommendations
       |                                       |
       v                                       v
  [MiningAgent Epoch Loop] <---------------- Apply dynamic adjustments
       |
       v
  [TokenLedger / ZKP] --> [AGI audits proofs in real-time]
       |
       v
  [Governance DB] --> [AGI simulates stake-power relationships]
       |
       v
[Alerts / Dashboard / Auto-Healing] <-- AGI can trigger emergency actions
```

## Key Components

1. **Open AGI Agent** - Dynamically monitors coherence metrics and adjusts MiningAgent parameters
2. **MiningAgent Epoch Loop** - Core verification loop that validates tokenomics
3. **TokenLedger / ZKP** - Tracks T0/T1/T4/T5 tokens and verifies ZKP proofs
4. **Governance DB** - Simulates stake-power relationships
5. **Alerts / Dashboard / Auto-Healing** - Emergency action triggers

## Features Implemented

✅ Full T0/T1/T4/T5 reconciliation
✅ Governance power validation
✅ Coherence monitoring (Ĉ(t) ≥ 0.9)
✅ Memory node utilization check
✅ Security & adversarial simulation (EMA Ψ damping)
✅ ZKP minting verification
✅ Automatic JSON report generation
✅ Open AGI integration for dynamic adjustments and predictive control

## Running the Framework

### Option 1: Direct Execution

```bash
python run_full_verification.py
```

### Option 2: Windows Batch Script

```cmd
run_full_verification.bat
```

### Option 3: Docker Compose (Recommended)

```bash
docker-compose up -d
```

## Docker Services

- **mining-agent** - Main verification loop
- **cal-engine** - Coherence Attunement Layer engine
- **governance** - Governance simulation service
- **zkp-ledger** - Zero-Knowledge Proof ledger
- **agi-coordinator** - Open AGI policy coordinator
- **verification-loop** - Continuous verification service
- **prometheus** - Metrics collection
- **grafana** - Visualization dashboard
- **alertmanager** - Alert handling

## Configuration

The framework can be configured through environment variables:

- `EPOCH_DURATION` - Duration of each verification epoch (default: 60 seconds)
- `METRICS_COLLECTION_INTERVAL` - How often to collect metrics (default: 10 seconds)
- `LOG_DIRECTORY` - Directory for log files (default: /var/log/quantum)

## Reports

Verification reports are generated in JSON format and saved to:
- Linux/Mac: `/var/log/quantum/`
- Windows: `C:\opt\quantum-currency\logs\`

## Next Steps

1. Deploy in staging environment first to ensure Open AGI actions do not destabilize live network
2. Configure AGI policy files to enforce safe parameter boundaries
3. Connect Prometheus / Grafana for live visualization of AGI decisions, coherence, and tokenomics
4. Test full emergency handling including force_emergency_state.py

## Troubleshooting

If you encounter issues:

1. Check the logs in `/var/log/quantum/` or `C:\opt\quantum-currency\logs\`
2. Ensure all Python dependencies are installed
3. Verify Docker services are running correctly
4. Check network connectivity between services