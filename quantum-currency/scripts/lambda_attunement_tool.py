#!/usr/bin/env python3
"""
CLI Tool for Lambda Attunement Layer
Provides command-line interface for managing and testing the λ-Attunement Layer
"""

import argparse
import json
import sys
import os
import time
from typing import Optional, Dict, Any

# Add the src directory to the path
src_path = os.path.join(os.path.dirname(__file__), '..', 'src')
sys.path.insert(0, src_path)

# Import the modules dynamically
lambda_attunement_path = os.path.join(src_path, 'core', 'lambda_attunement.py')
if os.path.exists(lambda_attunement_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("lambda_attunement", lambda_attunement_path)
    if spec is not None and spec.loader is not None:
        lambda_attunement = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(lambda_attunement)
        
        LambdaAttunementController = lambda_attunement.LambdaAttunementController
        AttunementConfig = lambda_attunement.AttunementConfig
    else:
        raise ImportError("Could not import lambda_attunement module")
else:
    raise ImportError(f"Lambda attunement module not found at {lambda_attunement_path}")

class MockCALEngine:
    """Mock CAL engine for CLI tool"""
    def __init__(self):
        self.alpha_multiplier = 1.0
        self.omega_history = []
        
    def sample_omega_snapshot(self):
        # Generate mock omega vectors
        import numpy as np
        return [np.random.random(5).tolist() for _ in range(3)]
        
    def normalize_C(self, C):
        # Normalize C to [0,1] range
        return min(1.0, max(0.0, C / 10.0))
        
    def set_alpha_multiplier(self, alpha):
        self.alpha_multiplier = alpha
        print(f"Setting alpha multiplier to {alpha:.4f}")
        
    def get_lambda(self):
        return 0.5  # Mock lambda value
        
    def get_entropy_rate(self):
        import numpy as np
        return np.random.random() * 0.001  # Mock entropy rate
        
    def get_h_internal(self):
        return 0.98  # Mock internal coherence
        
    def get_m_t_bounds(self):
        return (-1.0, 1.0)  # Mock bounds

def load_config(config_file: str) -> Dict[str, Any]:
    """Load configuration from JSON file"""
    try:
        with open(config_file, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Configuration file {config_file} not found, using defaults")
        return {}
    except json.JSONDecodeError as e:
        print(f"Error parsing configuration file {config_file}: {e}")
        return {}

def save_config(config: Dict[str, Any], config_file: str) -> bool:
    """Save configuration to JSON file"""
    try:
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        return True
    except Exception as e:
        print(f"Error saving configuration to {config_file}: {e}")
        return False

def run_dry_run(config: Dict[str, Any], cycles: int = 5) -> None:
    """Run a dry-run of the attunement controller"""
    print("🔍 Running Lambda Attunement Dry-Run")
    print("=" * 40)
    
    # Create mock CAL engine
    cal_engine = MockCALEngine()
    
    # Create attunement controller
    controller = LambdaAttunementController(cal_engine, config)
    
    # Show initial status
    print(f"Initial alpha: {controller.alpha:.4f}")
    print(f"Initial C_hat: {controller.meter.compute_C_hat():.4f}")
    
    # Run update cycles
    print(f"\nRunning {cycles} attunement cycles...")
    for i in range(cycles):
        accepted = controller.update()
        status = controller.get_status()
        print(f"Cycle {i+1}: alpha={status['alpha']:.4f}, "
              f"C_hat={controller.meter.compute_C_hat():.4f}, "
              f"{'ACCEPTED' if accepted else 'REVERTED'}")
        time.sleep(0.1)  # Short delay for demo
    
    # Show final status
    final_status = controller.get_status()
    print(f"\nFinal status:")
    print(f"   Alpha: {final_status['alpha']:.4f}")
    print(f"   Mode: {final_status['mode']}")
    print(f"   Accepts: {final_status['accept_counter']}")
    print(f"   Reverts: {final_status['revert_counter']}")
    
    # Show audit ledger summary
    ledger = controller.get_audit_ledger()
    print(f"\nAudit ledger entries: {len(ledger)}")
    if ledger:
        last_entry = ledger[-1]
        print(f"   Last entry: {last_entry['old_alpha']:.4f} -> {last_entry['new_alpha']:.4f}")

def show_status(config_file: str) -> None:
    """Show current attunement status"""
    print("📊 Lambda Attunement Status")
    print("=" * 30)
    
    # Load configuration
    config = load_config(config_file)
    if not config:
        print("No configuration found, using defaults")
        config = {}
    
    # Show configuration
    print("Configuration:")
    for key, value in config.items():
        print(f"   {key}: {value}")
    
    # In a real implementation, we would connect to a running controller
    # For now, we'll just show that the tool is working
    print("\n✅ Status tool is ready")

def save_state(state_file: str, config_file: str) -> None:
    """Save current attunement state"""
    print("💾 Saving Attunement State")
    print("=" * 25)
    
    # Load configuration
    config = load_config(config_file)
    
    # Create mock controller to save state
    cal_engine = MockCALEngine()
    controller = LambdaAttunementController(cal_engine, config)
    
    # Save state
    if controller.save_state(state_file):
        print(f"✅ State saved to {state_file}")
    else:
        print(f"❌ Failed to save state to {state_file}")

def load_state(state_file: str, config_file: str) -> None:
    """Load attunement state"""
    print("📂 Loading Attunement State")
    print("=" * 25)
    
    # Create mock controller to load state
    cal_engine = MockCALEngine()
    controller = LambdaAttunementController(cal_engine, {})
    
    # Load state
    if controller.load_state(state_file):
        print(f"✅ State loaded from {state_file}")
        
        # Save the loaded configuration
        config = {
            "alpha_initial": controller.alpha,
            "alpha_min": controller.cfg.alpha_min,
            "alpha_max": controller.cfg.alpha_max,
            "lr": controller.cfg.lr,
            "momentum": controller.cfg.momentum
        }
        if save_config(config, config_file):
            print(f"✅ Configuration saved to {config_file}")
    else:
        print(f"❌ Failed to load state from {state_file}")

def main():
    """Main entry point for the CLI tool"""
    parser = argparse.ArgumentParser(description="Lambda Attunement CLI Tool")
    parser.add_argument("--config", default="attunement_config.json", 
                        help="Configuration file (default: attunement_config.json)")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Dry-run command
    dry_run_parser = subparsers.add_parser("dry-run", help="Run a dry-run of the attunement controller")
    dry_run_parser.add_argument("--cycles", type=int, default=5, 
                                help="Number of cycles to run (default: 5)")
    
    # Status command
    subparsers.add_parser("status", help="Show current attunement status")
    
    # Save state command
    save_parser = subparsers.add_parser("save-state", help="Save current attunement state")
    save_parser.add_argument("--state-file", default="attunement_state.json",
                             help="State file to save to (default: attunement_state.json)")
    
    # Load state command
    load_parser = subparsers.add_parser("load-state", help="Load attunement state")
    load_parser.add_argument("--state-file", default="attunement_state.json",
                             help="State file to load from (default: attunement_state.json)")
    
    # Config command
    config_parser = subparsers.add_parser("config", help="Manage configuration")
    config_parser.add_argument("--set", nargs=2, metavar=("KEY", "VALUE"),
                               help="Set a configuration value")
    config_parser.add_argument("--get", metavar="KEY",
                               help="Get a configuration value")
    config_parser.add_argument("--list", action="store_true",
                               help="List all configuration values")
    
    args = parser.parse_args()
    
    if args.command == "dry-run":
        config = load_config(args.config)
        run_dry_run(config, args.cycles)
    elif args.command == "status":
        show_status(args.config)
    elif args.command == "save-state":
        save_state(args.state_file, args.config)
    elif args.command == "load-state":
        load_state(args.state_file, args.config)
    elif args.command == "config":
        config = load_config(args.config)
        if args.set:
            key, value = args.set
            # Try to convert value to appropriate type
            try:
                if '.' in value:
                    value = float(value)
                elif value.lower() in ['true', 'false']:
                    value = value.lower() == 'true'
                else:
                    value = int(value)
            except ValueError:
                # Keep as string
                pass
            config[key] = value
            if save_config(config, args.config):
                print(f"✅ Set {key} = {value}")
        elif args.get:
            if args.get in config:
                print(f"{args.get} = {config[args.get]}")
            else:
                print(f"❌ Configuration key '{args.get}' not found")
        elif args.list:
            print("Configuration:")
            for key, value in config.items():
                print(f"   {key}: {value}")
        else:
            config_parser.print_help()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()