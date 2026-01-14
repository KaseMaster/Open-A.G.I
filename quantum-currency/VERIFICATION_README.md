# HMN Production Coherence Verification & Stabilization Test

## Overview
This document provides instructions for running the comprehensive verification and stabilization testing for the Harmonic Mesh Network (HMN) production environment.

## Prerequisites
- Python 3.7 or higher
- Virtual environment with required dependencies installed
- Waitress WSGI server (will be installed automatically if missing)

## Verification Components

### 1. Python Scripts
- `run_final_hmn_verification.py` - Main verification script
- `run_simplified_hmn_verification.py` - Simplified verification script
- `run_full_hmn_verification.py` - Full verification with WSGI server startup (not recommended)
- `run_mass_emergence_verification.py` - Mass Emergence validation with auto-tuning

### 2. Batch Files
- `run_complete_hmn_verification.bat` - Windows batch file to run the complete verification
- `deploy_production_windows.bat` - Production deployment script
- `run_mass_emergence_verification.bat` - Windows batch file for Mass Emergence verification

### 3. Reports
- All verification reports are generated in the `reports/` directory
- Reports include detailed results of coherence scores, token transactions, and system performance
- Mass Emergence reports: `mass_emergence_report_YYYYMMDD_HHMMSS.md`
- Mass Emergence verification reports: `mass_emergence_verification_report_YYYYMMDD_HHMMSS.md`

## Running the Verification

### Option 1: Using the Batch File (Recommended)
```cmd
run_complete_hmn_verification.bat
```

### Option 2: Mass Emergence Verification
```cmd
run_mass_emergence_verification.bat
```

### Option 3: Direct Python Execution
```cmd
cd D:\AI AGENT CODERV1\QUANTUM CURRENCY\Open-A.G.I\quantum-currency
python run_final_hmn_verification.py
```

### Option 4: Mass Emergence Direct Execution
```cmd
cd D:\AI AGENT CODERV1\QUANTUM CURRENCY\Open-A.G.I\quantum-currency
python run_mass_emergence_verification.py
```

### Option 5: Manual Execution with Server
1. Start the server in one terminal:
   ```cmd
   cd D:\AI AGENT CODERV1\QUANTUM CURRENCY\Open-A.G.I\quantum-currency\src\api
   python -m waitress --host=127.0.0.1 --port=5000 --threads=4 main:app
   ```

2. Run the verification in another terminal:
   ```cmd
   cd D:\AI AGENT CODERV1\QUANTUM CURRENCY\Open-A.G.I\quantum-currency
   python run_final_hmn_verification.py
   ```

## Verification Process

The verification script performs the following checks:

1. **Production Environment Setup**
   - Ensures HMN nodes are running on production WSGI server
   - Starts all HMN services automatically

2. **Coherence Attunement Validation**
   - Monitors all validator nodes' coherence scores
   - Confirms all nodes are above threshold (≥ 0.750)
   - Identifies any nodes below threshold
   - Tests Auto-Balance feature to trigger stabilizing adjustments
   - Verifies coherence scores increase to meet or exceed thresholds
   - Confirms Global Resonance Dashboard reflects real-time coherence and feedback

3. **CAL Engine Functional Verification**
   - Ensures CAL Engine starts automatically with the system
   - Validates predictive and auto-tuning capabilities
   - Confirms real-time coherence flow visualization
   - Tests distribute stabilizing feedback operation

4. **Biometric & Feedback Stream**
   - Connects HRV, GSR, EEG sensors
   - Verifies energetic state analysis updates in real-time
   - Tests feedback submission functionality for coherence adjustments

5. **Dashboard Functionality Check**
   - Validates all dashboard features are fully operational

6. **5-Token System Verification**
   - Executes transactions with all tokens (FLX, CHR, PSY, ATR, RES)
   - Confirms proper ledger recording and balances
   - Validates reputation, attention, resonance, and psychological metrics update accordingly

7. **Stress & Performance Tests**
   - Measures throughput (ops/sec) and latency (ms)
   - Ensures network stability under load
   - Confirms coherence stability, no regressions

### Mass Emergence Verification Process

8. **Mass Emergence Calculation**
   - Derives dimensional constant C_mass = [M][L]⁻⁴
   - Calculates mass density ρ_mass using quantum field integrals
   - Validates dimensional consistency across all field equations

9. **Auto-Tuning Engine**
   - AI-driven coherence optimization
   - Real-time performance metrics analysis
   - Dynamic parameter adjustment for optimal system performance

10. **Mass Emergence Reporting**
    - Automated Mass Emergence Reports
    - Detailed validation results
    - Performance metrics and recommendations

## Generated Reports

After successful verification, the following reports are generated:

- `verification_report_YYYYMMDD_HHMMSS.md` - Detailed verification results
- `FINAL_HMN_VERIFICATION_SUMMARY.md` - Summary of all verification activities
- `mass_emergence_report_YYYYMMDD_HHMMSS.md` - Mass Emergence calculation and validation report
- `mass_emergence_verification_report_YYYYMMDD_HHMMSS.md` - Comprehensive Mass Emergence verification summary

## Mass Emergence Acceptance Criteria

✅ **Dimensional stability maintained**  
✅ **C_mass successfully derived**  
✅ **CAL Engine coherent with expanded Unified Field**  
✅ **Ω-field remains stable under mass-coupled feedback**  
✅ **Recursive coherence ≥ 0.90 sustained**

## Troubleshooting

### Common Issues

1. **Python not found**: Ensure Python is installed and added to PATH
2. **Import errors**: Ensure all dependencies are installed via `pip install -r requirements.txt`
3. **Server connection issues**: Ensure the API server is running before executing verification
4. **Permission errors**: Run the script as administrator if file access issues occur

### Server Management

To start the production server:
```cmd
cd D:\AI AGENT CODERV1\QUANTUM CURRENCY\Open-A.G.I\quantum-currency\src\api
python -m waitress --host=0.0.0.0 --port=5000 --threads=4 main:app
```

To stop the server:
```cmd
taskkill /f /im python.exe
```

## System Requirements

### Minimum
- Windows 10 or higher
- 4 GB RAM
- 2 CPU cores
- 1 GB free disk space

### Recommended
- Windows 11
- 8 GB RAM
- 4 CPU cores
- 5 GB free disk space

## Support

For issues with the verification process, please check:
1. All Python dependencies are installed
2. The Waitress WSGI server is available
3. The API server is running on port 5000
4. Firewall settings allow local network connections