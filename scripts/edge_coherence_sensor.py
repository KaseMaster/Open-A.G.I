#!/usr/bin/env python3
"""
Edge Computation Module for Real-Time Coherence Sensing
Processes harmonic data on edge devices for real-time coherence measurement
"""

import sys
import os
import time
import json
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

# Add parent directory to path to import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from openagi.harmonic_validation import (
    compute_spectrum, 
    pairwise_coherence, 
    compute_coherence_score,
    HarmonicSnapshot
)

@dataclass
class EdgeSensorReading:
    """Represents a sensor reading from an edge device"""
    sensor_id: str
    timestamp: float
    times: List[float]
    values: List[float]
    coherence_score: float = 0.0
    spectrum: Optional[List[Tuple[float, float]]] = None

class EdgeCoherenceSensor:
    """Edge computation module for real-time coherence sensing"""
    
    def __init__(self, sensor_id: str, sampling_rate: int = 2048):
        self.sensor_id = sensor_id
        self.sampling_rate = sampling_rate
        self.readings_buffer = []
        self.reference_signal = None
        
    def generate_reference_signal(self, duration: float = 0.5) -> Tuple[List[float], List[float]]:
        """
        Generate a reference signal for coherence comparison
        
        Args:
            duration: Duration of the signal in seconds
            
        Returns:
            Tuple of (times, values) lists
        """
        t = np.linspace(0, duration, int(duration * self.sampling_rate))
        # Reference signal with known harmonic components
        signal = (
            np.sin(2 * np.pi * 50 * t) +  # 50 Hz fundamental
            0.5 * np.sin(2 * np.pi * 100 * t)  # 100 Hz harmonic
        )
        return t.tolist(), signal.tolist()
    
    def generate_sensor_signal(self, 
                              base_freq: float = 50.0, 
                              noise_level: float = 0.1,
                              harmonic_distortion: float = 0.2,
                              duration: float = 0.5) -> Tuple[List[float], List[float]]:
        """
        Generate a sensor signal with configurable parameters
        
        Args:
            base_freq: Base frequency in Hz
            noise_level: Level of random noise to add
            harmonic_distortion: Level of harmonic distortion
            duration: Duration of the signal in seconds
            
        Returns:
            Tuple of (times, values) lists
        """
        t = np.linspace(0, duration, int(duration * self.sampling_rate))
        
        # Base signal with some variation
        signal = (
            np.sin(2 * np.pi * base_freq * t) +  # Fundamental
            harmonic_distortion * np.sin(2 * np.pi * base_freq * 2 * t) +  # 2nd harmonic
            harmonic_distortion * 0.5 * np.sin(2 * np.pi * base_freq * 3 * t) +  # 3rd harmonic
            np.random.normal(0, noise_level, len(t))  # Random noise
        )
        
        return t.tolist(), signal.tolist()
    
    def compute_realtime_coherence(self, 
                                  local_times: List[float], 
                                  local_values: List[float],
                                  reference_times: Optional[List[float]] = None,
                                  reference_values: Optional[List[float]] = None) -> float:
        """
        Compute real-time coherence between local signal and reference
        
        Args:
            local_times: Time series timestamps for local signal
            local_values: Time series values for local signal
            reference_times: Time series timestamps for reference signal (optional)
            reference_values: Time series values for reference signal (optional)
            
        Returns:
            Coherence score between 0 and 1
        """
        # If no reference provided, use internal reference
        if reference_times is None or reference_values is None:
            if self.reference_signal is None:
                reference_times, reference_values = self.generate_reference_signal()
                self.reference_signal = (reference_times, reference_values)
            else:
                reference_times, reference_values = self.reference_signal
        
        # Convert to numpy arrays
        local_array = np.array(local_values)
        reference_array = np.array(reference_values)
        fs = self.sampling_rate
        
        # Compute coherence
        try:
            coherence = pairwise_coherence(local_array, reference_array, fs)
            return float(coherence)
        except Exception as e:
            print(f"Warning: Error computing coherence - {e}")
            return 0.0
    
    def take_reading(self, 
                    base_freq: float = 50.0, 
                    noise_level: float = 0.1,
                    harmonic_distortion: float = 0.2) -> EdgeSensorReading:
        """
        Take a sensor reading and compute coherence
        
        Args:
            base_freq: Base frequency in Hz
            noise_level: Level of random noise to add
            harmonic_distortion: Level of harmonic distortion
            
        Returns:
            EdgeSensorReading object with coherence score
        """
        # Generate sensor signal
        times, values = self.generate_sensor_signal(
            base_freq=base_freq,
            noise_level=noise_level,
            harmonic_distortion=harmonic_distortion
        )
        
        # Compute spectrum
        spectrum = compute_spectrum(np.array(times), np.array(values))
        
        # Compute coherence
        coherence_score = self.compute_realtime_coherence(times, values)
        
        # Create reading
        reading = EdgeSensorReading(
            sensor_id=self.sensor_id,
            timestamp=time.time(),
            times=times,
            values=values,
            coherence_score=coherence_score,
            spectrum=spectrum
        )
        
        # Add to buffer
        self.readings_buffer.append(reading)
        
        # Keep only the last 20 readings
        if len(self.readings_buffer) > 20:
            self.readings_buffer.pop(0)
            
        return reading
    
    def get_average_coherence(self) -> float:
        """Get average coherence score from recent readings"""
        if not self.readings_buffer:
            return 0.0
        
        scores = [reading.coherence_score for reading in self.readings_buffer]
        return float(np.mean(scores))
    
    def get_readings_buffer(self) -> List[EdgeSensorReading]:
        """Get all buffered readings"""
        return self.readings_buffer.copy()

def simulate_edge_network(num_sensors: int = 3):
    """
    Simulate a network of edge coherence sensors
    
    Args:
        num_sensors: Number of edge sensors to simulate
    """
    print("📡 Edge Coherence Sensor Network Simulation")
    print("=" * 50)
    
    # Create edge sensors
    sensors = []
    for i in range(num_sensors):
        # Vary parameters for each sensor to simulate different conditions
        sensor = EdgeCoherenceSensor(
            sensor_id=f"edge-{i+1}",
            sampling_rate=2048
        )
        sensors.append(sensor)
    
    print(f"Intialized {num_sensors} edge coherence sensors")
    print("Starting real-time coherence sensing...")
    
    # Simulate sensing over time
    for cycle in range(5):
        print(f"\n🔄 Sensing Cycle {cycle + 1}")
        
        # Take readings from each sensor
        cycle_readings = []
        for i, sensor in enumerate(sensors):
            # Vary parameters for each cycle to simulate changing conditions
            base_freq = 50.0 + np.random.normal(0, 0.5)  # Slight frequency variation
            noise_level = 0.05 + np.random.exponential(0.05)  # Variable noise
            harmonic_distortion = 0.1 + np.random.exponential(0.1)  # Variable distortion
            
            try:
                reading = sensor.take_reading(
                    base_freq=base_freq,
                    noise_level=noise_level,
                    harmonic_distortion=harmonic_distortion
                )
                cycle_readings.append(reading)
                
                print(f"   Sensor {i+1}: Coherence = {reading.coherence_score:.4f}")
                print(f"      Base freq: {base_freq:.2f} Hz")
                print(f"      Noise level: {noise_level:.4f}")
            except Exception as e:
                print(f"   Sensor {i+1}: Error - {e}")
        
        # Compute network coherence statistics
        if cycle_readings:
            coherence_scores = [r.coherence_score for r in cycle_readings]
            avg_coherence = np.mean(coherence_scores)
            std_coherence = np.std(coherence_scores)
            
            print(f"   Network Stats: Avg = {avg_coherence:.4f}, Std = {std_coherence:.4f}")
        
        # Wait before next cycle
        time.sleep(1)
    
    # Display summary
    print(f"\n📊 Edge Network Summary:")
    for i, sensor in enumerate(sensors):
        avg_coherence = sensor.get_average_coherence()
        buffer_size = len(sensor.get_readings_buffer())
        print(f"   Sensor {i+1}: Avg coherence = {avg_coherence:.4f} ({buffer_size} readings)")
    
    print("\n✅ Edge coherence sensor network simulation completed!")

if __name__ == "__main__":
    simulate_edge_network(3)