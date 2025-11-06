#!/usr/bin/env python3
"""
Quantum Currency Dashboard v0.2.0
Real-time visualization of CAL-RΦV Fusion metrics

This module implements:
1. Real-time Ω-state visualization
2. Coherence health indicators
3. Network monitoring
4. Token economy dashboard
"""

import sys
import os
import json
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import time
import threading
from datetime import datetime

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import the modules
from src.models.coherence_attunement_layer import CoherenceAttunementLayer, OmegaState
from src.core.harmonic_validation import HarmonicSnapshot

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DashboardMetrics:
    """Represents metrics displayed on the dashboard"""
    timestamp: float
    network_coherence: float
    omega_state_metrics: Dict[str, float]
    token_balances: Dict[str, float]
    active_validators: int
    recent_events: List[str]
    health_status: str  # green, yellow, red, critical


class QuantumCurrencyDashboard:
    """
    Quantum Currency Dashboard for v0.2.0
    Real-time visualization of CAL-RΦV Fusion metrics
    """

    def __init__(self, network_id: str = "quantum-currency-dashboard-001"):
        self.network_id = network_id
        self.cal = CoherenceAttunementLayer(network_id=network_id)
        self.metrics_history: List[DashboardMetrics] = []
        self.is_running = False
        self.update_interval = 1.0  # seconds
        self.max_history = 100  # Maximum number of historical points
        
        # Initialize with sample data
        self._initialize_sample_data()
        
        logger.info(f"Initialized Quantum Currency Dashboard for network: {network_id}")

    def _initialize_sample_data(self):
        """Initialize dashboard with sample data"""
        # Create sample Ω-state
        sample_omega = self.cal.compute_omega_state(
            token_data={"rate": 5.0},
            sentiment_data={"energy": 0.7},
            semantic_data={"shift": 0.3},
            attention_data=[0.1, 0.2, 0.3, 0.4, 0.5]
        )
        
        # Create initial metrics
        initial_metrics = DashboardMetrics(
            timestamp=time.time(),
            network_coherence=sample_omega.coherence_score,
            omega_state_metrics={
                "token_rate": sample_omega.token_rate,
                "sentiment_energy": sample_omega.sentiment_energy,
                "semantic_shift": sample_omega.semantic_shift,
                "modulator": sample_omega.modulator,
                "time_delay": sample_omega.time_delay
            },
            token_balances={"FLX": 1000.0, "CHR": 500.0, "PSY": 200.0, "ATR": 300.0, "RES": 50.0},
            active_validators=5,
            recent_events=["Dashboard initialized", "CAL engine started"],
            health_status=self._get_health_status(sample_omega.coherence_score)
        )
        
        self.metrics_history.append(initial_metrics)

    def _get_health_status(self, coherence_score: float) -> str:
        """Get health status based on coherence score"""
        if coherence_score >= 0.85:
            return "green"  # Stable (Ready for Macro Write)
        elif coherence_score >= 0.65:
            return "yellow"  # Flux (Normal Operation)
        elif coherence_score >= 0.35:
            return "red"  # Unattuned (Safe Mode Active)
        else:
            return "critical"  # Critical (Emergency Mode)

    def _collect_current_metrics(self) -> DashboardMetrics:
        """Collect current metrics for display"""
        # In a real implementation, this would collect data from the network
        # For now, we'll simulate changing data
        
        # Get the latest Ω-state or create a new one
        if self.cal.omega_history:
            latest_omega = self.cal.omega_history[-1]
            # Add some variation to simulate real-time changes
            variation = (len(self.metrics_history) * 0.001) % 0.1
            new_coherence = max(0.0, min(1.0, latest_omega.coherence_score + variation - 0.05))
        else:
            new_coherence = 0.85
        
        # Create new Ω-state with variation
        new_omega = self.cal.compute_omega_state(
            token_data={"rate": 5.0 + (len(self.metrics_history) * 0.01)},
            sentiment_data={"energy": 0.7 + (len(self.metrics_history) * 0.001)},
            semantic_data={"shift": 0.3 + (len(self.metrics_history) * 0.0005)},
            attention_data=[
                0.1 + (len(self.metrics_history) * 0.001),
                0.2 + (len(self.metrics_history) * 0.001),
                0.3 + (len(self.metrics_history) * 0.001),
                0.4 + (len(self.metrics_history) * 0.001),
                0.5 + (len(self.metrics_history) * 0.001)
            ]
        )
        
        # Update token balances with some variation
        base_balances = {"FLX": 1000.0, "CHR": 500.0, "PSY": 200.0, "ATR": 300.0, "RES": 50.0}
        if self.metrics_history:
            last_balances = self.metrics_history[-1].token_balances
            for token in base_balances:
                variation = (len(self.metrics_history) * 0.1) % 10
                base_balances[token] = last_balances[token] + variation
        
        # Generate events based on changes
        events = []
        if self.metrics_history:
            last_metrics = self.metrics_history[-1]
            if abs(new_coherence - last_metrics.network_coherence) > 0.05:
                direction = "increased" if new_coherence > last_metrics.network_coherence else "decreased"
                events.append(f"Network coherence {direction} to {new_coherence:.3f}")
            
            if len(self.metrics_history) % 20 == 0:
                events.append("Periodic system check completed")
        
        metrics = DashboardMetrics(
            timestamp=time.time(),
            network_coherence=new_coherence,
            omega_state_metrics={
                "token_rate": new_omega.token_rate,
                "sentiment_energy": new_omega.sentiment_energy,
                "semantic_shift": new_omega.semantic_shift,
                "modulator": new_omega.modulator,
                "time_delay": new_omega.time_delay
            },
            token_balances=base_balances,
            active_validators=5,
            recent_events=events if events else ["System operating normally"],
            health_status=self._get_health_status(new_coherence)
        )
        
        return metrics

    def update_metrics(self):
        """Update dashboard metrics"""
        metrics = self._collect_current_metrics()
        self.metrics_history.append(metrics)
        
        # Keep only recent history
        if len(self.metrics_history) > self.max_history:
            self.metrics_history = self.metrics_history[-self.max_history:]
        
        logger.debug(f"Dashboard updated - Coherence: {metrics.network_coherence:.4f}, "
                    f"Health: {metrics.health_status}")

    def get_current_view(self) -> Dict[str, Any]:
        """Get current dashboard view data"""
        if not self.metrics_history:
            return {}
        
        current_metrics = self.metrics_history[-1]
        
        # Prepare data for visualization
        view_data = {
            "timestamp": datetime.fromtimestamp(current_metrics.timestamp).isoformat(),
            "network_coherence": current_metrics.network_coherence,
            "health_status": current_metrics.health_status,
            "omega_metrics": current_metrics.omega_state_metrics,
            "token_balances": current_metrics.token_balances,
            "active_validators": current_metrics.active_validators,
            "recent_events": current_metrics.recent_events,
            "historical_data": {
                "timestamps": [datetime.fromtimestamp(m.timestamp).isoformat() 
                              for m in self.metrics_history[-20:]],  # Last 20 points
                "coherences": [m.network_coherence for m in self.metrics_history[-20:]],
                "health_statuses": [m.health_status for m in self.metrics_history[-20:]]
            }
        }
        
        return view_data

    def start_dashboard(self):
        """Start the dashboard update loop"""
        if self.is_running:
            logger.warning("Dashboard is already running")
            return
        
        self.is_running = True
        logger.info("Dashboard started")
        
        def update_loop():
            while self.is_running:
                try:
                    self.update_metrics()
                    time.sleep(self.update_interval)
                except Exception as e:
                    logger.error(f"Error in dashboard update loop: {e}")
                    time.sleep(self.update_interval)
        
        # Start update thread
        self.update_thread = threading.Thread(target=update_loop, daemon=True)
        self.update_thread.start()
        
        logger.info("Dashboard update loop started")

    def stop_dashboard(self):
        """Stop the dashboard update loop"""
        self.is_running = False
        logger.info("Dashboard stopped")

    def export_metrics(self, filepath: str):
        """Export metrics history to file"""
        export_data = {
            "network_id": self.network_id,
            "export_timestamp": time.time(),
            "metrics_history": [asdict(m) for m in self.metrics_history]
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Metrics exported to {filepath}")

    def get_health_summary(self) -> Dict[str, Any]:
        """Get a summary of system health"""
        if not self.metrics_history:
            return {}
        
        current = self.metrics_history[-1]
        recent = self.metrics_history[-10:] if len(self.metrics_history) >= 10 else self.metrics_history
        
        avg_coherence = sum(m.network_coherence for m in recent) / len(recent)
        coherence_trend = "stable"
        if len(recent) > 1:
            if recent[-1].network_coherence > recent[0].network_coherence:
                coherence_trend = "improving"
            elif recent[-1].network_coherence < recent[0].network_coherence:
                coherence_trend = "declining"
        
        summary = {
            "current_coherence": current.network_coherence,
            "health_status": current.health_status,
            "avg_recent_coherence": avg_coherence,
            "coherence_trend": coherence_trend,
            "active_validators": current.active_validators,
            "timestamp": datetime.fromtimestamp(current.timestamp).isoformat()
        }
        
        return summary

    def render_text_dashboard(self):
        """Render a simple text-based dashboard"""
        if not self.metrics_history:
            print("No data available")
            return
        
        current = self.metrics_history[-1]
        health_indicator = {
            "green": "✅",
            "yellow": "⚠️",
            "red": "❌",
            "critical": "🔥"
        }.get(current.health_status, "❓")
        
        print("\n" + "="*60)
        print(f"QUANTUM CURRENCY DASHBOARD - {self.network_id}")
        print("="*60)
        print(f"Last Update: {datetime.fromtimestamp(current.timestamp).strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Network Health: {health_indicator} {current.health_status.upper()}")
        print(f"Network Coherence: {current.network_coherence:.4f}")
        print(f"Active Validators: {current.active_validators}")
        print()
        
        print("Ω-STATE METRICS:")
        print(f"  Token Rate: {current.omega_state_metrics['token_rate']:.4f}")
        print(f"  Sentiment Energy: {current.omega_state_metrics['sentiment_energy']:.4f}")
        print(f"  Semantic Shift: {current.omega_state_metrics['semantic_shift']:.4f}")
        print(f"  Modulator: {current.omega_state_metrics['modulator']:.4f}")
        print(f"  Time Delay: {current.omega_state_metrics['time_delay']:.4f}")
        print()
        
        print("TOKEN BALANCES:")
        for token, balance in current.token_balances.items():
            print(f"  {token}: {balance:.2f}")
        print()
        
        print("RECENT EVENTS:")
        for event in current.recent_events[-3:]:  # Show last 3 events
            print(f"  • {event}")
        print("="*60)


def main():
    """Main function to run the dashboard"""
    logger.info("Starting Quantum Currency Dashboard v0.2.0")
    
    # Create dashboard
    dashboard = QuantumCurrencyDashboard(network_id="qc-dashboard-mainnet-001")
    
    # Start dashboard
    dashboard.start_dashboard()
    
    # Run for a while to show updates
    print("Quantum Currency Dashboard v0.2.0")
    print("Press Ctrl+C to stop")
    
    try:
        for i in range(30):  # Run for 30 seconds
            dashboard.render_text_dashboard()
            time.sleep(2)
    except KeyboardInterrupt:
        print("\nStopping dashboard...")
    
    # Stop dashboard
    dashboard.stop_dashboard()
    
    # Show final summary
    summary = dashboard.get_health_summary()
    print("\nFINAL HEALTH SUMMARY:")
    print(f"Current Coherence: {summary.get('current_coherence', 0):.4f}")
    print(f"Health Status: {summary.get('health_status', 'unknown')}")
    print(f"Trend: {summary.get('coherence_trend', 'unknown')}")
    
    # Export data
    timestamp = int(time.time())
    dashboard.export_metrics(f"dashboard_metrics_{timestamp}.json")
    print(f"Metrics exported to dashboard_metrics_{timestamp}.json")


if __name__ == "__main__":
    main()