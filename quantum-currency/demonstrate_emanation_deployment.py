#!/usr/bin/env python3
"""
Demonstrate the complete Emanation Phase deployment workflow
"""

import sys
import os
import subprocess
import time
import json
from datetime import datetime

def run_command(command, description):
    """Run a shell command and return the result"""
    print(f"🔧 {description}")
    print(f"   Command: {command}")
    
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print("   ✅ Success")
            if result.stdout:
                print(f"   Output: {result.stdout.strip()}")
        else:
            print("   ❌ Failed")
            if result.stderr:
                print(f"   Error: {result.stderr.strip()}")
        print()
        return result.returncode == 0
    except Exception as e:
        print(f"   ❌ Exception: {e}")
        print()
        return False

def main():
    print("=" * 80)
    print("🌌 Quantum Currency Emanation Phase Deployment Demonstration")
    print("=" * 80)
    print()
    
    # Check if we're in the right directory
    if not os.path.exists("emanation_deploy.py"):
        print("❌ Error: This script must be run from the quantum-currency directory")
        return 1
    
    print("📍 Current directory:", os.getcwd())
    print()
    
    # Step 1: Generate sample reports
    print("📋 STEP 1: Generate Sample Reports")
    print("-" * 40)
    if not run_command("python3 generate_sample_reports.py", "Generating sample reports..."):
        print("❌ Failed to generate sample reports")
        return 1
    
    # Wait a moment for files to be written
    time.sleep(2)
    
    # Step 2: Verify reports
    print("🔍 STEP 2: Verify Generated Reports")
    print("-" * 40)
    if not run_command("python3 verify_reports.py", "Verifying reports..."):
        print("❌ Failed to verify reports")
        return 1
    
    # Step 3: Run staging verification (single cycle for demo)
    print("🧪 STEP 3: Run Staging Verification")
    print("-" * 40)
    if not run_command("python3 run_staging_verification.py --cycles 1 --interval 5", "Running staging verification..."):
        print("❌ Failed to run staging verification")
        return 1
    
    # Wait for report to be generated
    time.sleep(2)
    
    # Step 4: Run emanation deployment monitor (single cycle for demo)
    print("📡 STEP 4: Run Emanation Deployment Monitor")
    print("-" * 40)
    
    # Create data directory for demo
    os.makedirs("/mnt/data", exist_ok=True)
    
    if not run_command("python3 emanation_deploy.py --cycles 1 --interval 5 --report-dir /mnt/data", "Running emanation monitor..."):
        print("❌ Failed to run emanation monitor")
        return 1
    
    # Wait for report to be generated
    time.sleep(2)
    
    # Step 5: Show generated files
    print("📂 STEP 5: Show Generated Files")
    print("-" * 40)
    
    # Show sample reports
    print("Sample Reports:")
    run_command("ls -la emanation_cycle_*.json", "Listing sample cycle reports...")
    run_command("ls -la emanation_deployment_summary.json", "Listing sample summary report...")
    
    # Show verification reports
    print("\nVerification Reports:")
    run_command("ls -la reports/staging/", "Listing staging verification reports...")
    
    # Show monitoring reports
    print("\nMonitoring Reports:")
    run_command("ls -la /mnt/data/", "Listing monitoring reports...")
    
    # Step 6: Show report contents
    print("📄 STEP 6: Show Sample Report Contents")
    print("-" * 40)
    
    # Show a sample cycle report
    try:
        cycle_report_files = [f for f in os.listdir(".") if f.startswith("emanation_cycle_") and f.endswith(".json")]
        if cycle_report_files:
            with open(cycle_report_files[0], 'r') as f:
                cycle_report = json.load(f)
            print(f"Sample Cycle Report ({cycle_report_files[0]}):")
            print(json.dumps(cycle_report, indent=2)[:500] + "..." if len(json.dumps(cycle_report, indent=2)) > 500 else json.dumps(cycle_report, indent=2))
            print()
    except Exception as e:
        print(f"❌ Failed to read sample cycle report: {e}")
    
    # Show summary report
    try:
        if os.path.exists("emanation_deployment_summary.json"):
            with open("emanation_deployment_summary.json", 'r') as f:
                summary_report = json.load(f)
            print("Summary Report:")
            print(json.dumps(summary_report, indent=2))
            print()
    except Exception as e:
        print(f"❌ Failed to read summary report: {e}")
    
    # Final summary
    print("=" * 80)
    print("🎉 Emanation Phase Deployment Demonstration Complete!")
    print("=" * 80)
    print()
    print("✅ What we accomplished:")
    print("   • Generated sample monitoring reports")
    print("   • Verified report structure and content")
    print("   • Ran staging verification process")
    print("   • Executed emanation deployment monitoring")
    print("   • Demonstrated full workflow")
    print()
    print("🚀 Next steps for production deployment:")
    print("   1. Replace simulated metrics with real Prometheus data")
    print("   2. Configure alerting integrations (Slack/PagerDuty)")
    print("   3. Deploy Kubernetes manifests")
    print("   4. Set up continuous monitoring")
    print("   5. Integrate with Grafana dashboards")
    print()
    print("📚 Key scripts created:")
    print("   • emanation_deploy.py - Main monitoring controller")
    print("   • run_staging_verification.py - Staging verification")
    print("   • generate_sample_reports.py - Sample data generator")
    print("   • verify_reports.py - Report verification tool")
    print("   • Kubernetes manifests for production deployment")
    print()
    print("The Quantum Currency Emanation Phase is ready for deployment! 🌟")
    print()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())