#!/usr/bin/env python3
"""
Generate sample reports for testing the Emanation Deployment Monitor
"""

import sys
import os
import json
import random
from datetime import datetime, timedelta
from typing import Dict, Any

def generate_sample_cycle_report(cycle: int) -> Dict[str, Any]:
    """Generate a sample cycle report"""
    # Generate timestamp for this cycle
    timestamp = (datetime.now() - timedelta(minutes=(5-cycle)*10)).isoformat() + "Z"
    
    # Generate realistic metrics with some variation
    base_h_internal = 0.95 + (cycle * 0.005)  # Gradually improve
    base_caf = 1.0 + (cycle * 0.005)  # Gradually improve
    base_entropy = 0.003 - (cycle * 0.0002)  # Gradually decrease
    
    return {
        "cycle": cycle,
        "timestamp": timestamp,
        "metrics": {
            "h_internal": round(base_h_internal + random.uniform(-0.01, 0.01), 4),
            "caf": round(base_caf + random.uniform(-0.01, 0.01), 4),
            "entropy_rate": round(base_entropy + random.uniform(-0.0005, 0.0005), 6),
            "connected_systems": random.randint(8, 12),
            "coherence_score": round(0.92 + (cycle * 0.003) + random.uniform(-0.01, 0.01), 4)
        },
        "control_parameters": {
            "lambda_L": round(0.5 + (cycle * 0.02), 3),
            "m_t": round(1.0 + (cycle * 0.01), 3),
            "Omega_t": round(0.8 + (cycle * 0.015), 3),
            "Psi": round(0.7 + (cycle * 0.01), 3)
        },
        "adjustments_made": {
            "lambda_L": round(random.uniform(-0.05, 0.05), 3),
            "m_t": round(random.uniform(-0.03, 0.03), 3),
            "Omega_t": round(random.uniform(-0.04, 0.04), 3),
            "Psi": round(random.uniform(-0.02, 0.02), 3)
        },
        "alerts": [] if cycle > 2 else [f"WARNING: Initial stabilization in progress (cycle {cycle})"],
        "status": "stable" if cycle > 2 else "warning"
    }

def generate_sample_summary_report(reports: list) -> Dict[str, Any]:
    """Generate a sample summary report"""
    # Calculate averages
    metrics_sum = {
        "h_internal": 0,
        "caf": 0,
        "entropy_rate": 0,
        "connected_systems": 0,
        "coherence_score": 0
    }
    
    alert_count = 0
    critical_alerts = 0
    
    for report in reports:
        for metric in metrics_sum:
            metrics_sum[metric] += report["metrics"][metric]
        alert_count += len(report["alerts"])
        critical_alerts += len([a for a in report["alerts"] if "CRITICAL" in a])
    
    metrics_avg = {k: round(v / len(reports), 4) for k, v in metrics_sum.items()}
    
    return {
        "summary": True,
        "timestamp": datetime.now().isoformat() + "Z",
        "total_cycles": len(reports),
        "average_metrics": metrics_avg,
        "total_alerts": alert_count,
        "critical_alerts": critical_alerts,
        "final_control_parameters": reports[-1]["control_parameters"] if reports else {},
        "status": "stable" if critical_alerts == 0 else "issues_detected"
    }

def save_report(report: Dict[str, Any], filename: str, directory: str = "/mnt/data"):
    """Save report to file"""
    # Ensure directory exists
    os.makedirs(directory, exist_ok=True)
    
    filepath = os.path.join(directory, filename)
    with open(filepath, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"📄 Generated sample report: {filepath}")

def main():
    print("=" * 80)
    print("📄 Generating Sample Reports for Emanation Deployment Monitor")
    print("=" * 80)
    print()
    
    # Generate 5 cycle reports
    reports = []
    for i in range(1, 6):
        report = generate_sample_cycle_report(i)
        filename = f"emanation_cycle_{i:03d}.json"
        save_report(report, filename)
        reports.append(report)
        print(f"✅ Generated cycle {i} report")
    
    # Generate summary report
    summary = generate_sample_summary_report(reports)
    save_report(summary, "emanation_deployment_summary.json")
    print("✅ Generated summary report")
    
    print()
    print("=" * 80)
    print("✅ Sample Reports Generation Complete!")
    print("=" * 80)
    print("Generated files:")
    print("  • emanation_cycle_001.json")
    print("  • emanation_cycle_002.json")
    print("  • emanation_cycle_003.json")
    print("  • emanation_cycle_004.json")
    print("  • emanation_cycle_005.json")
    print("  • emanation_deployment_summary.json")
    print()
    print("These sample reports can be used for testing the dashboard")
    print("and verification systems before running actual monitoring.")

if __name__ == "__main__":
    sys.exit(main())