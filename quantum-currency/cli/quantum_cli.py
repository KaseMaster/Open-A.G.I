#!/usr/bin/env python3
"""
Quantum Currency CLI Tool v0.2.0
Command-line interface for Quantum Currency system with CAL-RΦV Fusion

This tool provides:
1. System status monitoring
2. Network coherence checking
3. Token balance queries
4. Simulation execution
5. Dashboard launching
"""

import sys
import os
import argparse
import json
import time
from datetime import datetime

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import the modules
from src.models.coherence_attunement_layer import CoherenceAttunementLayer
from src.simulation.multi_node_simulator import MultiNodeSimulator
from src.dashboard.dashboard_app import QuantumCurrencyDashboard


def print_header():
    """Print CLI header"""
    print("""
    ╔══════════════════════════════════════════════════════════════════════════════╗
    ║                    QUANTUM CURRENCY CLI v0.2.0                             ║
    ║              CAL-RΦV Fusion Implementation Complete                      ║
    ║==========================================================================║
    ║  ✓ Phase 1: Core Refactor (CAL Implementation)                          ║
    ║  ✓ Phase 2: Multi-node Simulation                                       ║
    ║  ✓ Phase 3: Dashboard Integration                                       ║
    ║  🚧 Phase 4: OpenAGI Policy Feedback Loop                               ║
    ╚══════════════════════════════════════════════════════════════════════════════╝
    """)


def check_system_status():
    """Check and display system status"""
    print("Checking Quantum Currency System Status...")
    print("-" * 50)
    
    # Initialize CAL
    cal = CoherenceAttunementLayer(network_id="qc-cli-status-check")
    
    # Compute sample Ω-state
    omega = cal.compute_omega_state(
        token_data={"rate": 5.0},
        sentiment_data={"energy": 0.7},
        semantic_data={"shift": 0.3},
        attention_data=[0.1, 0.2, 0.3, 0.4, 0.5]
    )
    
    # Get health status
    health = cal.get_coherence_health_indicator()
    health_emoji = {
        "green": "✅",
        "yellow": "⚠️",
        "red": "❌",
        "critical": "🔥"
    }.get(health, "❓")
    
    print(f"Network Coherence: {omega.coherence_score:.4f}")
    print(f"System Health: {health_emoji} {health.upper()}")
    print(f"Ω-State Components:")
    print(f"  • Token Rate: {omega.token_rate:.4f}")
    print(f"  • Sentiment Energy: {omega.sentiment_energy:.4f}")
    print(f"  • Semantic Shift: {omega.semantic_shift:.4f}")
    print(f"  • Modulator: {omega.modulator:.4f}")
    print(f"  • Time Delay: {omega.time_delay:.4f}")


def run_quick_simulation():
    """Run a quick simulation"""
    print("Running Quick Multi-Node Simulation...")
    print("-" * 40)
    
    # Create simulator with fewer nodes for quick test
    simulator = MultiNodeSimulator(num_nodes=5, network_id="qc-cli-sim-quick")
    
    # Run 3 rounds
    results = []
    for i in range(3):
        result = simulator.run_simulation_round(i)
        results.append(result)
        print(f"Round {i+1}: Coherence = {result['network_coherence']:.4f}, "
              f"Validation = {'PASS' if result['is_valid'] else 'FAIL'}")
        time.sleep(0.5)  # Small delay for better UX
    
    # Generate report
    report = simulator.generate_performance_report()
    print(f"\nSimulation Summary:")
    print(f"  • Average Coherence: {report['coherence_metrics']['avg_network_coherence']:.4f}")
    print(f"  • Success Rate: {report['validation_metrics']['success_rate']:.1%}")
    print(f"  • Stability Score: {report['network_stability']['stability_score']:.4f}")


def launch_dashboard():
    """Launch the dashboard"""
    print("Launching Quantum Currency Dashboard...")
    print("-" * 40)
    
    # Create dashboard
    dashboard = QuantumCurrencyDashboard(network_id="qc-cli-dashboard")
    
    # Show initial view
    dashboard.render_text_dashboard()
    
    # Update a few times to show dynamics
    print("\nUpdating dashboard...")
    for i in range(3):
        dashboard.update_metrics()
        time.sleep(1)
        dashboard.render_text_dashboard()
    
    # Show health summary
    summary = dashboard.get_health_summary()
    print(f"\nHealth Summary:")
    print(f"  • Current Coherence: {summary['current_coherence']:.4f}")
    print(f"  • Trend: {summary['coherence_trend']}")
    print(f"  • Status: {summary['health_status']}")


def show_roadmap():
    """Display the current roadmap status"""
    print("Quantum Currency v0.2.0 Roadmap Status")
    print("=" * 50)
    print("""
VERSION 0.2.0 - MAINNET RELEASE
------------------------------
✅ CAL-RΦV Fusion Implementation
✅ Coherence Attunement Layer
✅ Multi-node Simulation Testing
✅ Dashboard Integration
🚧 AI Co-governance (OpenAGI)
🚧 Harmonic Gating Mechanisms

COMPLETED PHASES:
✅ Phase 1: Core Refactor (v0.2.0-alpha)
✅ Phase 2: Multi-node Simulation (v0.2.0-beta)
✅ Phase 3: Dashboard Integration

UPCOMING:
🚧 Phase 4: OpenAGI Policy Feedback Loop
    """)


def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(
        description="Quantum Currency CLI Tool v0.2.0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  status      Check system status
  simulate    Run quick simulation
  dashboard   Launch dashboard
  roadmap     Show roadmap status
  all         Run all checks
        """
    )
    
    parser.add_argument(
        'command',
        choices=['status', 'simulate', 'dashboard', 'roadmap', 'all'],
        help='Command to execute'
    )
    
    parser.add_argument(
        '--nodes',
        type=int,
        default=5,
        help='Number of nodes for simulation (default: 5)'
    )
    
    args = parser.parse_args()
    
    # Print header
    print_header()
    
    # Execute command
    if args.command == 'status':
        check_system_status()
    elif args.command == 'simulate':
        run_quick_simulation()
    elif args.command == 'dashboard':
        launch_dashboard()
    elif args.command == 'roadmap':
        show_roadmap()
    elif args.command == 'all':
        check_system_status()
        print("\n")
        run_quick_simulation()
        print("\n")
        launch_dashboard()
        print("\n")
        show_roadmap()


if __name__ == "__main__":
    main()