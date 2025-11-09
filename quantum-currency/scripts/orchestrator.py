#!/usr/bin/env python3
"""
Unified Self-Healing Orchestrator for Quantum Currency System
This script orchestrates the full verification pipeline with coherence-aware self-healing.
"""

import os
import subprocess
import time
import sys
from typing import List, Dict, Any

# Add the src directory to the path so we can import the coherence attunement agent
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

try:
    from coherence_attunement_agent import CoherenceAttunementAgent
    HAS_COHERENCE_AGENT = True
except ImportError:
    HAS_COHERENCE_AGENT = False
    print("⚠️  Coherence Attunement Agent not available, using fallback methods")

# Configuration
BATCH_FILE = "run_quantum_currency.bat"
MAX_RETRIES = 2
OPTIONS = list(range(1, 17))
REQUIRED_DEMOS = [2, 3, 14, 15]
CRITICAL_DASHBOARD_OPTIONS = [5, 6]
LOG_PATH = "logs/full_self_healing_log.txt"

def run_batch_option(option: int) -> bool:
    """Run a specific menu option from the batch file."""
    cmd = f'cmd /c "{BATCH_FILE} {option}"'
    print(f"[ORCH] ▶ Running Option {option}...")
    
    # Create logs directory if it doesn't exist
    log_dir = os.path.dirname(LOG_PATH)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    try:
        proc = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=120)
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(proc.stdout)
            f.write(proc.stderr)
        return proc.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"[ORCH] ❌ Option {option} timed out")
        return False
    except Exception as e:
        print(f"[ORCH] ❌ Error running Option {option}: {e}")
        return False

def fix_mint_transaction():
    """Attempt to auto-fix Mint Transaction demo import issue."""
    print("[ORCH] ⚙️ Attempting to fix Mint Transaction import issue...")
    path = "scripts/demo_mint_flex.py"
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            if "import harmonic_validation" not in content:
                content = "from openagi import harmonic_validation\n" + content
                with open(path, "w", encoding="utf-8") as f:
                    f.write(content)
                print("[ORCH] ✅ Import fixed successfully.")
        except Exception as e:
            print(f"[ORCH] ❌ Error fixing Mint Transaction demo: {e}")
    else:
        print("[ORCH] ⚠️ Mint Transaction demo file not found.")

def main():
    print("[ORCH] === Quantum Currency v0.3.0 — Coherence-Aware Verification ===")
    
    # Initialize the coherence attunement agent if available
    agent = None
    if HAS_COHERENCE_AGENT:
        agent = CoherenceAttunementAgent(log_path=LOG_PATH)
        
        if not agent.start_server():
            print("[ORCH] ❌ Server startup failed. Exiting.")
            return 1
    else:
        print("[ORCH] ⚠️ Coherence Attunement Agent not available, using basic orchestration")
    
    try:
        for option in OPTIONS:
            success = run_batch_option(option)
            
            if option in REQUIRED_DEMOS and not success:
                print(f"[ORCH] ❌ Option {option} failed. Retrying fix cycle...")
                fix_mint_transaction()
                for retry in range(MAX_RETRIES):
                    if run_batch_option(option):
                        print(f"[ORCH] ✅ Option {option} passed on retry {retry+1}.")
                        break
                else:
                    print(f"[ORCH] ❌ Option {option} repeatedly failed after {MAX_RETRIES} retries.")
            
            # Check coherence health between critical steps if agent is available
            if agent and option in CRITICAL_DASHBOARD_OPTIONS:
                print("[ORCH] 🌐 Checking system coherence via API...")
                if not agent.check_attunement_health():
                    print("[ORCH] ⚠️ Coherence anomaly detected — reinitializing attunement.")
                    agent.stop_server()
                    time.sleep(5)
                    agent.start_server()
                    agent.check_attunement_health()

        print("[ORCH] ✅ Full Phase 6–7 verification sequence completed successfully.")
        return 0

    except KeyboardInterrupt:
        print("[ORCH] ⚠️ Verification interrupted by user.")
        return 1
    except Exception as e:
        print(f"[ORCH] ❌ Unexpected error during verification: {e}")
        return 1
    finally:
        if agent:
            agent.stop_server()
            print("[ORCH] 🧩 All processes terminated cleanly.")

if __name__ == "__main__":
    sys.exit(main())