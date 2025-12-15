#!/usr/bin/env python3
"""
Test script for IACE v2.0 Telemetry System
"""

import sys
import os
import time
import json

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_telemetry_system():
    print("🧪 Testing IACE v2.0 Telemetry System")
    print("=" * 40)
    
    try:
        # Import telemetry streamer
        from src.monitoring.telemetry_streamer import telemetry_streamer
        
        # Test data callback function
        def get_test_kpi_data():
            import numpy as np
            return {
                "coherence": np.random.uniform(0.90, 1.00),
                "gas": np.random.uniform(0.95, 1.00),
                "rsi": np.random.uniform(0.85, 0.99),
                "lambda_opt": np.random.uniform(0.5, 1.0),
                "caf_emission": np.random.uniform(0.0, 10.0),
                "gravity_well_count": np.random.randint(0, 5),
                "stable_clusters": np.random.randint(10, 50),
                "transaction_rate": np.random.uniform(0.1, 10.0),
                "system_health": "STABLE" if np.random.random() > 0.1 else "WARNING"
            }
        
        # Subscribe to telemetry updates
        def on_telemetry_update(data):
            print(f"📡 Telemetry update received: Coherence={data['coherence']:.3f}, GAS={data['gas']:.3f}")
        
        telemetry_streamer.subscribe(on_telemetry_update)
        
        # Test pushing telemetry data
        print(" Pushing test telemetry data...")
        for i in range(3):
            test_data = get_test_kpi_data()
            test_data["test_run"] = i + 1
            telemetry_streamer.push_telemetry(test_data)
            time.sleep(1)
        
        # Test getting current KPIs
        current_kpis = telemetry_streamer.get_current_kpis()
        print(f" Current KPIs: {current_kpis}")
        
        # Test getting historical trends
        trends = telemetry_streamer.get_historical_trends("coherence", hours=1)
        print(f" Historical coherence data points: {len(trends)}")
        
        # Test getting available metrics
        if current_kpis:
            metric_names = list(current_kpis.keys())
            print(f" Available metrics: {metric_names}")
        
        # Test continuous streaming
        print(" Starting continuous telemetry streaming for 5 seconds...")
        telemetry_streamer.start_streaming(get_test_kpi_data, interval=1.0)
        time.sleep(5)
        telemetry_streamer.stop_streaming()
        
        print("✅ Telemetry system test completed successfully!")
        return True
        
    except ImportError as e:
        print(f"❌ Error importing telemetry modules: {e}")
        return False
    except Exception as e:
        print(f"❌ Error testing telemetry system: {e}")
        return False

if __name__ == "__main__":
    success = test_telemetry_system()
    sys.exit(0 if success else 1)