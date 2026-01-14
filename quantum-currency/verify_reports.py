#!/usr/bin/env python3
"""
Verify generated reports from the Emanation Deployment Monitor
"""

import sys
import os
import json
import glob
from datetime import datetime
from typing import Dict, Any, List, Optional

def load_report(filepath: str) -> Optional[Dict[str, Any]]:
    """Load a JSON report file"""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Failed to load report {filepath}: {e}")
        return None

def verify_cycle_report(report: Dict[str, Any], cycle_num: int) -> bool:
    """Verify a cycle report has the expected structure"""
    required_fields = [
        "cycle", "timestamp", "metrics", "control_parameters", 
        "adjustments_made", "alerts", "status"
    ]
    
    # Check required fields exist
    for field in required_fields:
        if field not in report:
            print(f"❌ Missing field in cycle {cycle_num}: {field}")
            return False
    
    # Check cycle number
    if report["cycle"] != cycle_num:
        print(f"❌ Incorrect cycle number in cycle {cycle_num}: expected {cycle_num}, got {report['cycle']}")
        return False
    
    # Check timestamp format
    try:
        datetime.fromisoformat(report["timestamp"].rstrip('Z'))
    except Exception:
        print(f"❌ Invalid timestamp format in cycle {cycle_num}: {report['timestamp']}")
        return False
    
    # Check metrics structure
    metrics = report["metrics"]
    expected_metrics = ["h_internal", "caf", "entropy_rate", "connected_systems", "coherence_score"]
    for metric in expected_metrics:
        if metric not in metrics:
            print(f"❌ Missing metric in cycle {cycle_num}: {metric}")
            return False
    
    # Check control parameters structure
    control_params = report["control_parameters"]
    expected_params = ["lambda_L", "m_t", "Omega_t", "Psi"]
    for param in expected_params:
        if param not in control_params:
            print(f"❌ Missing control parameter in cycle {cycle_num}: {param}")
            return False
    
    # Check adjustments structure
    adjustments = report["adjustments_made"]
    for param in expected_params:
        if param not in adjustments:
            print(f"❌ Missing adjustment parameter in cycle {cycle_num}: {param}")
            return False
    
    # Check status is valid
    valid_statuses = ["stable", "warning", "critical"]
    if report["status"] not in valid_statuses:
        print(f"❌ Invalid status in cycle {cycle_num}: {report['status']}")
        return False
    
    print(f"✅ Cycle {cycle_num} report verified successfully")
    return True

def verify_summary_report(report: Dict[str, Any]) -> bool:
    """Verify a summary report has the expected structure"""
    required_fields = [
        "summary", "timestamp", "total_cycles", "average_metrics",
        "total_alerts", "critical_alerts", "final_control_parameters", "status"
    ]
    
    # Check required fields exist
    for field in required_fields:
        if field not in report:
            print(f"❌ Missing field in summary report: {field}")
            return False
    
    # Check summary flag
    if not report["summary"]:
        print("❌ Summary flag not set to true")
        return False
    
    # Check timestamp format
    try:
        datetime.fromisoformat(report["timestamp"].rstrip('Z'))
    except Exception:
        print(f"❌ Invalid timestamp format in summary: {report['timestamp']}")
        return False
    
    # Check average metrics structure
    avg_metrics = report["average_metrics"]
    expected_metrics = ["h_internal", "caf", "entropy_rate", "connected_systems", "coherence_score"]
    for metric in expected_metrics:
        if metric not in avg_metrics:
            print(f"❌ Missing average metric in summary: {metric}")
            return False
    
    # Check final control parameters structure
    final_params = report["final_control_parameters"]
    expected_params = ["lambda_L", "m_t", "Omega_t", "Psi"]
    for param in expected_params:
        if param not in final_params:
            print(f"❌ Missing final control parameter in summary: {param}")
            return False
    
    # Check status is valid
    valid_statuses = ["stable", "issues_detected"]
    if report["status"] not in valid_statuses:
        print(f"❌ Invalid status in summary: {report['status']}")
        return False
    
    print("✅ Summary report verified successfully")
    return True

def run_verification() -> bool:
    """Run the full verification process"""
    print("=" * 80)
    print("🔍 Verifying Emanation Deployment Monitor Reports")
    print("=" * 80)
    print()
    
    # Check if /mnt/data directory exists
    if not os.path.exists("/mnt/data"):
        print("❌ /mnt/data directory not found")
        return False
    
    # Find all cycle reports
    cycle_files = glob.glob("/mnt/data/emanation_cycle_*.json")
    cycle_files.sort()
    
    if not cycle_files:
        print("❌ No cycle reports found in /mnt/data")
        return False
    
    print(f"📊 Found {len(cycle_files)} cycle reports")
    
    # Verify each cycle report
    cycle_reports = []
    all_cycles_valid = True
    
    for i, filepath in enumerate(cycle_files, 1):
        print(f"\n📄 Verifying {os.path.basename(filepath)}...")
        report = load_report(filepath)
        if report is None:
            all_cycles_valid = False
            continue
        
        cycle_num = int(os.path.basename(filepath).split('_')[2].split('.')[0])
        if verify_cycle_report(report, cycle_num):
            cycle_reports.append(report)
        else:
            all_cycles_valid = False
    
    # Find summary report
    summary_files = glob.glob("/mnt/data/emanation_deployment_summary.json")
    
    if not summary_files:
        print("\n❌ Summary report not found")
        return False
    
    print(f"\n📄 Verifying {os.path.basename(summary_files[0])}...")
    summary_report = load_report(summary_files[0])
    if summary_report is None:
        return False
    
    summary_valid = verify_summary_report(summary_report)
    
    # Print final results
    print("\n" + "=" * 80)
    print("📋 VERIFICATION RESULTS")
    print("=" * 80)
    
    if all_cycles_valid and summary_valid:
        print("✅ All reports verified successfully!")
        print(f"   • {len(cycle_reports)} cycle reports")
        print(f"   • 1 summary report")
        
        # Print some statistics
        if cycle_reports:
            first_metrics = cycle_reports[0]["metrics"]
            last_metrics = cycle_reports[-1]["metrics"]
            
            print("\n📈 Metrics Improvement:")
            print(f"   H_internal: {first_metrics['h_internal']:.4f} → {last_metrics['h_internal']:.4f}")
            print(f"   CAF: {first_metrics['caf']:.4f} → {last_metrics['caf']:.4f}")
            print(f"   Entropy Rate: {first_metrics['entropy_rate']:.6f} → {last_metrics['entropy_rate']:.6f}")
        
        return True
    else:
        print("❌ Some reports failed verification")
        if not all_cycles_valid:
            print("   • Cycle reports had issues")
        if not summary_valid:
            print("   • Summary report had issues")
        return False

def main() -> int:
    success = run_verification()
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())