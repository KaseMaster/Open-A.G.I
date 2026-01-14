#!/usr/bin/env python3
"""
HMN CLI Tool
Command-line interface for managing and monitoring Harmonic Mesh Network nodes
"""

import argparse
import json
import requests
import sys
from typing import Dict, Any
from datetime import datetime

class HMNCLI:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
    
    def get_node_health(self) -> Dict[str, Any]:
        """Get node health status"""
        try:
            response = requests.get(f"{self.base_url}/metrics", timeout=5)
            if response.status_code == 200:
                # Parse Prometheus metrics
                metrics = self._parse_prometheus_metrics(response.text)
                return {
                    "status": "healthy",
                    "timestamp": datetime.now().isoformat(),
                    "metrics": metrics
                }
            else:
                return {
                    "status": "unhealthy",
                    "error": f"HTTP {response.status_code}",
                    "timestamp": datetime.now().isoformat()
                }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _parse_prometheus_metrics(self, metrics_text: str) -> Dict[str, Any]:
        """Parse Prometheus metrics text into a dictionary"""
        metrics = {}
        for line in metrics_text.split('\n'):
            if line and not line.startswith('#') and ' ' in line:
                parts = line.split(' ')
                if len(parts) >= 2:
                    key = parts[0]
                    value = parts[1]
                    # Try to convert to number
                    try:
                        if '.' in value:
                            metrics[key] = float(value)
                        else:
                            metrics[key] = int(value)
                    except ValueError:
                        metrics[key] = value
        return metrics
    
    def get_ledger_info(self) -> Dict[str, Any]:
        """Get ledger information"""
        # In a real implementation, this would query the ledger service
        return {
            "status": "mock",
            "transaction_count": 1250,
            "pending_transactions": 5,
            "last_block": "0xabc123",
            "timestamp": datetime.now().isoformat()
        }
    
    def get_consensus_stats(self) -> Dict[str, Any]:
        """Get consensus statistics"""
        # In a real implementation, this would query the consensus service
        return {
            "status": "mock",
            "validators": 5,
            "rounds_completed": 42,
            "emergency_rounds": 2,
            "avg_round_duration": 2.3,
            "slashing_events": 1,
            "boosting_events": 3,
            "timestamp": datetime.now().isoformat()
        }
    
    def format_health_output(self, health_data: Dict[str, Any]) -> str:
        """Format health data for display"""
        if health_data["status"] == "healthy":
            output = "✅ Node Health: Healthy\n"
            output += f"🕒 Last Check: {health_data['timestamp']}\n\n"
            output += "📈 Key Metrics:\n"
            
            metrics = health_data.get("metrics", {})
            key_metrics = [
                ("hmn_node_health_status", "Health Status"),
                ("hmn_node_lambda_t", "λ(t)"),
                ("hmn_node_coherence_density", "Ĉ(t)"),
                ("hmn_node_psi_score", "Ψ"),
                ("hmn_node_service_calls_total", "Service Calls")
            ]
            
            for metric_key, display_name in key_metrics:
                if metric_key in metrics:
                    value = metrics[metric_key]
                    output += f"  {display_name}: {value}\n"
        else:
            output = "❌ Node Health: Unhealthy\n"
            output += f"🕒 Last Check: {health_data['timestamp']}\n"
            output += f"💥 Error: {health_data.get('error', 'Unknown error')}\n"
        
        return output
    
    def format_ledger_output(self, ledger_data: Dict[str, Any]) -> str:
        """Format ledger data for display"""
        output = "📚 Ledger Information\n"
        output += "=" * 20 + "\n"
        output += f"Transaction Count: {ledger_data['transaction_count']}\n"
        output += f"Pending Transactions: {ledger_data['pending_transactions']}\n"
        output += f"Last Block: {ledger_data['last_block']}\n"
        output += f"Timestamp: {ledger_data['timestamp']}\n"
        return output
    
    def format_consensus_output(self, consensus_data: Dict[str, Any]) -> str:
        """Format consensus data for display"""
        output = "⚖️ Consensus Statistics\n"
        output += "=" * 22 + "\n"
        output += f"Validators: {consensus_data['validators']}\n"
        output += f"Rounds Completed: {consensus_data['rounds_completed']}\n"
        output += f"Emergency Rounds: {consensus_data['emergency_rounds']}\n"
        output += f"Avg Round Duration: {consensus_data['avg_round_duration']:.2f}s\n"
        output += f"Slashing Events: {consensus_data['slashing_events']}\n"
        output += f"Boosting Events: {consensus_data['boosting_events']}\n"
        output += f"Timestamp: {consensus_data['timestamp']}\n"
        return output

def main():
    parser = argparse.ArgumentParser(description="HMN CLI Tool")
    parser.add_argument("command", choices=["health", "ledger", "consensus", "stats"],
                       help="Command to execute")
    parser.add_argument("--url", "-u", default="http://localhost:8000",
                       help="Base URL of the HMN node (default: http://localhost:8000)")
    parser.add_argument("--format", "-f", choices=["json", "text"], default="text",
                       help="Output format (default: text)")
    
    args = parser.parse_args()
    
    cli = HMNCLI(args.url)
    
    try:
        if args.command == "health":
            health_data = cli.get_node_health()
            if args.format == "json":
                print(json.dumps(health_data, indent=2))
            else:
                print(cli.format_health_output(health_data))
        
        elif args.command == "ledger":
            ledger_data = cli.get_ledger_info()
            if args.format == "json":
                print(json.dumps(ledger_data, indent=2))
            else:
                print(cli.format_ledger_output(ledger_data))
        
        elif args.command == "consensus":
            consensus_data = cli.get_consensus_stats()
            if args.format == "json":
                print(json.dumps(consensus_data, indent=2))
            else:
                print(cli.format_consensus_output(consensus_data))
        
        elif args.command == "stats":
            # Show all statistics
            health_data = cli.get_node_health()
            ledger_data = cli.get_ledger_info()
            consensus_data = cli.get_consensus_stats()
            
            if args.format == "json":
                stats = {
                    "health": health_data,
                    "ledger": ledger_data,
                    "consensus": consensus_data
                }
                print(json.dumps(stats, indent=2))
            else:
                print(cli.format_health_output(health_data))
                print()
                print(cli.format_ledger_output(ledger_data))
                print()
                print(cli.format_consensus_output(consensus_data))
    
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()