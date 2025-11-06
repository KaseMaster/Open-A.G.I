#!/usr/bin/env python3
"""
IoT Sensor Data Ingestion for Quantum Currency System
Simulates ingestion of real-world harmonic data from IoT sensors
"""

import sys
import os
import time
import json
import random
import numpy as np
from typing import List, Tuple
from datetime import datetime

# Add parent directory to path to import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from openagi.harmonic_validation import make_snapshot, compute_spectrum

class IoTDataIngestor:
    """Simulates IoT sensor data ingestion for harmonic validation"""
    
    def __init__(self, sensor_id: str):
        self.sensor_id = sensor_id
        self.data_buffer = []
        
    def generate_sensor_data(self, duration: float = 0.5, sample_rate: int = 2048) -> Tuple[List[float], List[float]]:
        """
        Generate simulated sensor data with harmonic components
        
        Args:
            duration: Duration of the signal in seconds
            sample_rate: Sampling rate in Hz
            
        Returns:
            Tuple of (times, values) lists
        """
        t = np.linspace(0, duration, int(duration * sample_rate))
        
        # Base signal with multiple harmonic components
        signal = (
            np.sin(2 * np.pi * 50 * t) +  # 50 Hz fundamental
            0.3 * np.sin(2 * np.pi * 100 * t) +  # 100 Hz harmonic
            0.2 * np.sin(2 * np.pi * 150 * t) +  # 150 Hz harmonic
            0.1 * np.sin(2 * np.pi * 200 * t) +  # 200 Hz harmonic
            0.05 * np.random.normal(0, 0.1, len(t))  # Add some noise
        )
        
        return t.tolist(), signal.tolist()
    
    def ingest_data(self) -> dict:
        """
        Ingest sensor data and create a harmonic snapshot
        
        Returns:
            Dictionary containing the snapshot data
        """
        times, values = self.generate_sensor_data()
        
        # Create a snapshot from the sensor data
        snapshot = make_snapshot(
            node_id=f"sensor-{self.sensor_id}",
            times=times,
            values=values
        )
        
        # Convert to dictionary for JSON serialization
        snapshot_data = {
            "node_id": snapshot.node_id,
            "timestamp": snapshot.timestamp,
            "times": snapshot.times,
            "values": snapshot.values,
            "spectrum": snapshot.spectrum,
            "spectrum_hash": snapshot.spectrum_hash,
            "CS": snapshot.CS,
            "phi_params": snapshot.phi_params
        }
        
        # Add to buffer
        self.data_buffer.append(snapshot_data)
        
        # Keep only the last 10 snapshots in buffer
        if len(self.data_buffer) > 10:
            self.data_buffer.pop(0)
            
        return snapshot_data
    
    def get_buffer_data(self) -> List[dict]:
        """Get all buffered sensor data"""
        return self.data_buffer.copy()
    
    def clear_buffer(self):
        """Clear the data buffer"""
        self.data_buffer.clear()

def simulate_iot_network(num_sensors: int = 3):
    """
    Simulate a network of IoT sensors ingesting harmonic data
    
    Args:
        num_sensors: Number of sensors to simulate
    """
    print("📡 IoT Sensor Data Ingestion Simulation")
    print("=" * 50)
    
    # Create sensor ingestors
    sensors = [IoTDataIngestor(f"sensor-{i+1}") for i in range(num_sensors)]
    
    print(f"Intialized {num_sensors} IoT sensors")
    print("Starting data ingestion...")
    
    # Simulate data ingestion over time
    for cycle in range(5):
        print(f"\n🔄 Ingestion Cycle {cycle + 1}")
        
        # Ingest data from each sensor
        for i, sensor in enumerate(sensors):
            try:
                snapshot_data = sensor.ingest_data()
                print(f"   Sensor {i+1}: Ingested snapshot at {datetime.fromtimestamp(snapshot_data['timestamp']).strftime('%H:%M:%S')}")
                print(f"      Spectrum hash: {snapshot_data['spectrum_hash'][:16]}...")
                print(f"      Coherence score: {snapshot_data['CS']:.4f}")
            except Exception as e:
                print(f"   Sensor {i+1}: Error - {e}")
        
        # Wait before next cycle
        time.sleep(1)
    
    # Display summary
    print(f"\n📊 Ingestion Summary:")
    for i, sensor in enumerate(sensors):
        buffer_data = sensor.get_buffer_data()
        print(f"   Sensor {i+1}: {len(buffer_data)} snapshots in buffer")
    
    print("\n✅ IoT sensor data ingestion simulation completed!")

if __name__ == "__main__":
    simulate_iot_network(3)