#!/usr/bin/env python3
"""
Telemetry Streamer for IACE v2.0
Provides real-time KPI streaming, historical trend plotting, and dashboard integration
"""

import json
import time
import threading
from typing import Dict, Any, List
from pathlib import Path
import numpy as np

class TelemetryStreamer:
    """Streams telemetry data to dashboard and maintains historical KPI trends"""
    
    def __init__(self, history_file: str = "kpi_history.json"):
        self.history_file = Path(history_file)
        self.kpi_history: List[Dict[str, Any]] = []
        self.streaming = False
        self.stream_thread = None
        self.subscribers = []
        self.load_history()
        
    def load_history(self):
        """Load historical KPI data from file"""
        if self.history_file.exists():
            try:
                with open(self.history_file, 'r') as f:
                    self.kpi_history = json.load(f)
                print(f"Loaded {len(self.kpi_history)} historical KPI records")
            except Exception as e:
                print(f"Error loading KPI history: {e}")
                self.kpi_history = []
    
    def save_history(self):
        """Save KPI history to file"""
        try:
            # Keep only last 1000 records to prevent file from growing too large
            if len(self.kpi_history) > 1000:
                self.kpi_history = self.kpi_history[-1000:]
                
            with open(self.history_file, 'w') as f:
                json.dump(self.kpi_history, f, indent=2)
        except Exception as e:
            print(f"Error saving KPI history: {e}")
    
    def subscribe(self, callback):
        """Subscribe to telemetry updates"""
        if callback not in self.subscribers:
            self.subscribers.append(callback)
    
    def unsubscribe(self, callback):
        """Unsubscribe from telemetry updates"""
        if callback in self.subscribers:
            self.subscribers.remove(callback)
    
    def push_telemetry(self, kpi_data: Dict[str, Any]):
        """Push telemetry data to subscribers and store in history"""
        # Add timestamp
        kpi_data["timestamp"] = time.time()
        
        # Store in history
        self.kpi_history.append(kpi_data)
        self.save_history()
        
        # Notify subscribers
        for callback in self.subscribers:
            try:
                callback(kpi_data)
            except Exception as e:
                print(f"Error notifying subscriber: {e}")
    
    def get_historical_trends(self, metric_name: str, hours: int = 24) -> List[Dict[str, Any]]:
        """Get historical trend data for a specific metric"""
        if not self.kpi_history:
            return []
        
        # Calculate cutoff time
        cutoff_time = time.time() - (hours * 3600)
        
        # Filter data for the specified metric and time range
        trend_data = []
        for record in self.kpi_history:
            if record.get("timestamp", 0) >= cutoff_time:
                if metric_name in record:
                    trend_data.append({
                        "timestamp": record["timestamp"],
                        "value": record[metric_name],
                        "coherence": record.get("coherence", 0),
                        "gas": record.get("gas", 0),
                        "rsi": record.get("rsi", 0)
                    })
        
        return trend_data
    
    def get_current_kpis(self) -> Dict[str, Any]:
        """Get the most recent KPI data"""
        if self.kpi_history:
            return self.kpi_history[-1]
        return {}
    
    def start_streaming(self, kpi_source_callback, interval: float = 5.0):
        """Start continuous telemetry streaming"""
        if self.streaming:
            print("Telemetry streamer is already running")
            return
            
        self.streaming = True
        self.stream_thread = threading.Thread(
            target=self._stream_loop, 
            args=(kpi_source_callback, interval),
            daemon=True
        )
        self.stream_thread.start()
        print("Telemetry streamer started")
    
    def stop_streaming(self):
        """Stop continuous telemetry streaming"""
        self.streaming = False
        if self.stream_thread:
            self.stream_thread.join()
        print("Telemetry streamer stopped")
    
    def _stream_loop(self, kpi_source_callback, interval: float):
        """Internal loop for continuous telemetry streaming"""
        while self.streaming:
            try:
                # Get current KPIs from source
                kpi_data = kpi_source_callback()
                if kpi_data:
                    self.push_telemetry(kpi_data)
                time.sleep(interval)
            except Exception as e:
                print(f"Error in telemetry stream loop: {e}")
                time.sleep(interval)

# Global telemetry streamer instance
telemetry_streamer = TelemetryStreamer()

def get_kpi_data_from_system():
    """
    Example function to get KPI data from the QECS system
    In a real implementation, this would interface with the actual system components
    """
    # Simulate KPI data
    return {
        "coherence": np.random.uniform(0.90, 1.00),
        "gas": np.random.uniform(0.95, 1.00),
        "rsi": np.random.uniform(0.85, 0.99),
        "lambda_opt": np.random.uniform(0.5, 1.0),
        "delta_lambda": np.random.uniform(0.0, 0.01),
        "caf_emission": np.random.uniform(0.0, 10.0),
        "gravity_well_count": np.random.randint(0, 5),
        "stable_clusters": np.random.randint(10, 50),
        "transaction_rate": np.random.uniform(0.1, 10.0),
        "system_health": "STABLE" if np.random.random() > 0.1 else "WARNING"
    }

# Example usage
if __name__ == "__main__":
    print("Telemetry Streamer Demo")
    print("=" * 30)
    
    # Start streaming
    telemetry_streamer.start_streaming(get_kpi_data_from_system, interval=2.0)
    
    # Subscribe to updates
    def on_telemetry_update(data):
        print(f"Telemetry update: Coherence={data['coherence']:.3f}, GAS={data['gas']:.3f}")
    
    telemetry_streamer.subscribe(on_telemetry_update)
    
    # Let it run for a bit
    try:
        time.sleep(10)
    except KeyboardInterrupt:
        pass
    
    # Stop streaming
    telemetry_streamer.stop_streaming()
    print("Demo completed")