# Quantum Currency v0.3.0 - Unified Self-Healing Verification Pipeline

## Overview
This update introduces a fully autonomous, coherence-aware test and recovery loop for the Quantum Currency system, replacing the previous fragile PowerShell orchestration with a unified Python-based process manager.

## New Components

### 1. Coherence Attunement Agent (`src/coherence_attunement_agent.py`)
A new Python module that provides:
- API server lifecycle management (start/stop/restart)
- Coherence health validation using λ(t) and Ĉ(t) metrics
- Automated healing and recovery mechanisms
- Comprehensive logging capabilities

### 2. Unified Orchestrator (`scripts/orchestrator.py`)
A master Python script that replaces the PowerShell orchestration with:
- Sequential execution of all 16 menu options from `run_quantum_currency.bat`
- Automated patching and retry logic for failed demos (especially Mint Transaction)
- Dynamic API health validation between critical stages
- Integration with the Coherence Attunement Agent for λ(t) and Ĉ(t) monitoring

### 3. Updated PowerShell Script (`full_system_verification.ps1`)
The PowerShell script has been simplified to:
- Handle dependency installation and module integrity checks
- Launch the new Python orchestrator
- Provide summary reporting

## Key Features

### Autonomous Execution
- Runs all 16 menu options automatically without user intervention
- Skips or retries intelligently depending on success

### Intelligent Self-Healing
- Automatically patches import issues in Mint Transaction demo
- Retries failed demos with exponential backoff
- Detects and resolves coherence anomalies

### Coherence-Aware Monitoring
- Continuously monitors λ(t) (Lambda Attunement) and Ĉ(t) (Coherence Density)
- Restarts server or recalibrates when coherence fails
- Provides real-time health scoring

### Comprehensive Logging
- Consolidates all console outputs and coherence logs
- Produces a single verifiable audit log
- PowerShell summary shows only essential results for clarity

## Integration Benefits

1. **Eliminates Manual Log Parsing** - All logs are automatically consolidated
2. **Full Coherence Validation** - Runtime monitoring of λ(t) and Ĉ(t) metrics
3. **Automated Patching** - No manual intervention needed for common import issues
4. **CI/CD Ready** - Designed for automated testing pipelines
5. **True Coherence-Driven Resilience** - Aligns technology with λ(t) stability principles

## File Structure
```
QUANTUM_CURRENCY/
│
├── src/
│   ├── core/
│   ├── dashboard/
│   ├── api/
│   │   └── main_api.py
│   └── coherence_attunement_agent.py   <-- NEW
│
├── scripts/
│   ├── run_quantum_currency.bat
│   ├── self_healing_verification.ps1
│   └── orchestrator.py                 <-- NEW MASTER SCRIPT
│
└── logs/
    └── full_self_healing_log.txt
```

## Expected Output
```
[AGENT] Server started (PID: 2438)
[ORCH] ▶ Running Option 1...
[ORCH] ✅ Option 4 passed all unit tests.
[AGENT] ✅ Attunement OK: λ(t)=1.002, Ĉ(t)=0.981
[ORCH] ⚙️ Attempting to fix Mint Transaction import issue...
[ORCH] ✅ Option 15 passed on retry 1.
[ORCH] 🌐 Checking system coherence via API...
[AGENT] ✅ Attunement OK: λ(t)=1.010, Ĉ(t)=0.997
[ORCH] ✅ Full Phase 6–7 verification sequence completed successfully.
```