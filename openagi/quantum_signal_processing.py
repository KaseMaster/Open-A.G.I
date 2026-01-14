#!/usr/bin/env python3
"""
Quantum Signal Processing for Quantum Currency System
Implements quantum-compatible signal processing experiments
"""

import sys
import os
import json
import time
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, fftfreq

# Add parent directory to path to import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

@dataclass
class QuantumSignal:
    """Represents a quantum signal for processing"""
    signal_id: str
    times: List[float]
    values: List[float]
    frequency_domain: Optional[List[complex]] = None
    processed: bool = False
    quantum_features: Optional[Dict] = None

class QuantumSignalProcessor:
    """
    Implements quantum-compatible signal processing experiments
    """
    
    def __init__(self):
        self.processed_signals: Dict[str, QuantumSignal] = {}
    
    def generate_quantum_noise(self, duration: float = 1.0, sample_rate: int = 1000) -> QuantumSignal:
        """
        Generate quantum-like noise signal for testing
        
        Args:
            duration: Duration of the signal in seconds
            sample_rate: Sample rate in Hz
            
        Returns:
            QuantumSignal object with quantum-like noise
        """
        # Generate time vector
        t = np.linspace(0, duration, int(duration * sample_rate))
        
        # Generate quantum-like noise with specific characteristics
        # This simulates the probabilistic nature of quantum measurements
        np.random.seed(42)  # For reproducible results
        
        # Combine multiple noise sources to simulate quantum behavior
        # 1. Thermal noise (Gaussian)
        thermal_noise = np.random.normal(0, 0.1, len(t))
        
        # 2. Shot noise (Poisson-like)
        shot_noise = np.random.poisson(1, len(t)) - 1
        shot_noise = shot_noise.astype(float) * 0.05
        
        # 3. Flicker noise (1/f noise)
        flicker_noise = self._generate_flicker_noise(len(t), sample_rate) * 0.02
        
        # 4. Quantum tunneling-like effects
        tunneling_effects = self._generate_tunneling_effects(len(t)) * 0.03
        
        # Combine all noise sources
        combined_signal = thermal_noise + shot_noise + flicker_noise + tunneling_effects
        
        # Normalize the signal
        combined_signal = combined_signal / np.max(np.abs(combined_signal))
        
        signal_obj = QuantumSignal(
            signal_id=f"quantum-noise-{int(time.time())}",
            times=t.tolist(),
            values=combined_signal.tolist()
        )
        
        self.processed_signals[signal_obj.signal_id] = signal_obj
        return signal_obj
    
    def _generate_flicker_noise(self, length: int, sample_rate: int) -> np.ndarray:
        """
        Generate flicker noise (1/f noise) approximation
        
        Args:
            length: Length of the signal
            sample_rate: Sample rate in Hz
            
        Returns:
            Flicker noise signal
        """
        # Simple approximation of 1/f noise
        freqs = fftfreq(length, 1/sample_rate)
        # Avoid division by zero at DC
        freqs[0] = 1e-10
        
        # Generate random phases
        phases = np.random.uniform(0, 2*np.pi, length)
        
        # Create 1/f spectrum
        spectrum = np.zeros(length, dtype=complex)
        spectrum[1:length//2] = (1/np.abs(freqs[1:length//2])) * np.exp(1j * phases[1:length//2])
        spectrum[length//2+1:] = np.conj(spectrum[1:length//2][::-1])
        
        # Convert back to time domain
        flicker_signal = np.real(np.fft.ifft(spectrum))
        
        return flicker_signal
    
    def _generate_tunneling_effects(self, length: int) -> np.ndarray:
        """
        Generate quantum tunneling-like effects
        
        Args:
            length: Length of the signal
            
        Returns:
            Tunneling effects signal
        """
        # Simulate quantum tunneling as sudden amplitude changes
        tunneling_signal = np.zeros(length)
        
        # Random number of tunneling events
        num_events = np.random.poisson(5)
        
        for _ in range(num_events):
            # Random position and amplitude for tunneling event
            pos = np.random.randint(0, length)
            amplitude = np.random.normal(0, 0.5)
            width = np.random.randint(10, 50)
            
            # Create a Gaussian pulse to represent tunneling event
            x = np.arange(length)
            pulse = amplitude * np.exp(-((x - pos)**2) / (2 * width**2))
            tunneling_signal += pulse
        
        return tunneling_signal
    
    def apply_quantum_filter(self, quantum_signal: QuantumSignal) -> QuantumSignal:
        """
        Apply quantum-compatible filtering to a signal
        
        Args:
            quantum_signal: Input quantum signal
            
        Returns:
            Filtered quantum signal
        """
        # Convert to numpy arrays for processing
        times = np.array(quantum_signal.times)
        values = np.array(quantum_signal.values)
        
        # Apply quantum-compatible filters
        # 1. Quantum decoherence filter (exponential decay envelope)
        decoherence_filter = np.exp(-times * 0.5)
        decoherence_filtered = values * decoherence_filter
        
        # 2. Quantum interference filter (comb filter to simulate interference patterns)
        comb_filter = 1 + 0.3 * np.cos(2 * np.pi * 10 * times)  # 10 Hz interference
        interference_filtered = decoherence_filtered * comb_filter
        
        # 3. Quantum measurement projection filter (thresholding)
        threshold = 0.1
        projected_signal = np.where(np.abs(interference_filtered) > threshold, 
                                   interference_filtered, 0)
        
        # Create filtered signal object
        filtered_signal = QuantumSignal(
            signal_id=f"{quantum_signal.signal_id}-filtered",
            times=times.tolist(),
            values=projected_signal.tolist(),
            processed=True
        )
        
        self.processed_signals[filtered_signal.signal_id] = filtered_signal
        return filtered_signal
    
    def extract_quantum_features(self, quantum_signal: QuantumSignal) -> Dict:
        """
        Extract quantum-specific features from a signal
        
        Args:
            quantum_signal: Input quantum signal
            
        Returns:
            Dictionary of quantum features
        """
        # Convert to numpy arrays for processing
        times = np.array(quantum_signal.times)
        values = np.array(quantum_signal.values)
        
        # Compute frequency domain representation
        freq_domain = fft(values)
        freqs = fftfreq(len(values), times[1] - times[0])
        
        # Store frequency domain in signal object
        quantum_signal.frequency_domain = freq_domain.tolist()
        
        # Extract quantum features
        features = {
            "signal_energy": np.sum(np.abs(values)**2),
            "peak_frequency": freqs[np.argmax(np.abs(freq_domain))],
            "spectral_entropy": self._compute_spectral_entropy(freq_domain),
            "quantum_coherence": self._compute_quantum_coherence(values),
            "entanglement_measure": self._compute_entanglement_measure(values),
            "superposition_index": self._compute_superposition_index(values),
            "decoherence_rate": self._compute_decoherence_rate(values, times)
        }
        
        quantum_signal.quantum_features = features
        return features
    
    def _compute_spectral_entropy(self, freq_domain: np.ndarray) -> float:
        """
        Compute spectral entropy as a measure of signal complexity
        
        Args:
            freq_domain: Frequency domain representation
            
        Returns:
            Spectral entropy value
        """
        # Normalize power spectrum
        power_spectrum = np.abs(freq_domain)**2
        normalized_power = power_spectrum / np.sum(power_spectrum)
        
        # Avoid log(0) by replacing zeros with a small value
        normalized_power = np.where(normalized_power == 0, 1e-12, normalized_power)
        
        # Compute entropy
        entropy = -np.sum(normalized_power * np.log2(normalized_power))
        return entropy
    
    def _compute_quantum_coherence(self, signal_values: np.ndarray) -> float:
        """
        Compute a measure of quantum coherence in the signal
        
        Args:
            signal_values: Signal values
            
        Returns:
            Quantum coherence measure
        """
        # Quantum coherence can be measured as the degree of phase correlation
        # For simplicity, we'll use the ratio of signal energy to noise energy
        signal_energy = np.sum(np.abs(signal_values)**2)
        noise_energy = np.sum(np.abs(np.diff(signal_values))**2)
        
        if noise_energy == 0:
            return 1.0
        
        coherence = signal_energy / (signal_energy + noise_energy)
        return coherence
    
    def _compute_entanglement_measure(self, signal_values: np.ndarray) -> float:
        """
        Compute a measure of entanglement-like correlations in the signal
        
        Args:
            signal_values: Signal values
            
        Returns:
            Entanglement measure
        """
        # For classical signals, we can measure correlations as a proxy for entanglement
        # Compute autocorrelation as a measure of self-correlation
        autocorr = np.correlate(signal_values, signal_values, mode='full')
        autocorr = autocorr[autocorr.size // 2:]
        
        # Normalize
        if autocorr[0] != 0:
            autocorr = autocorr / autocorr[0]
        
        # Measure of entanglement as the sum of correlations beyond a threshold
        threshold = 0.1
        entanglement = np.sum(np.abs(autocorr[1:][np.abs(autocorr[1:]) > threshold]))
        
        return min(entanglement, 1.0)  # Normalize to [0, 1]
    
    def _compute_superposition_index(self, signal_values: np.ndarray) -> float:
        """
        Compute a measure of superposition-like behavior in the signal
        
        Args:
            signal_values: Signal values
            
        Returns:
            Superposition index
        """
        # Superposition can be measured as the diversity of signal states
        # We'll use the number of distinct amplitude levels as a proxy
        unique_values = np.unique(np.round(signal_values, decimals=2))
        max_possible_states = len(signal_values)
        
        if max_possible_states == 0:
            return 0.0
        
        superposition_index = len(unique_values) / max_possible_states
        return superposition_index
    
    def _compute_decoherence_rate(self, signal_values: np.ndarray, times: np.ndarray) -> float:
        """
        Compute decoherence rate of the signal
        
        Args:
            signal_values: Signal values
            times: Time points
            
        Returns:
            Decoherence rate
        """
        # Decoherence rate can be measured as the rate of amplitude decay
        if len(signal_values) < 2:
            return 0.0
        
        # Fit exponential decay to signal envelope
        envelope = np.abs(signal_values)
        
        # Simple linear fit to log of envelope to estimate decay rate
        log_envelope = np.log(envelope + 1e-10)  # Add small value to avoid log(0)
        
        if len(times) > 1:
            # Compute slope of log envelope
            slope = np.polyfit(times, log_envelope, 1)[0]
            decoherence_rate = -slope
            return max(0, decoherence_rate)  # Ensure non-negative rate
        
        return 0.0
    
    def visualize_signal(self, quantum_signal: QuantumSignal, title: str = "Quantum Signal"):
        """
        Visualize a quantum signal (time domain and frequency domain)
        
        Args:
            quantum_signal: Quantum signal to visualize
            title: Plot title
        """
        # Convert to numpy arrays
        times = np.array(quantum_signal.times)
        values = np.array(quantum_signal.values)
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Time domain plot
        ax1.plot(times, values, 'b-', linewidth=1)
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Amplitude')
        ax1.set_title(f'{title} - Time Domain')
        ax1.grid(True, alpha=0.3)
        
        # Frequency domain plot (if available)
        if quantum_signal.frequency_domain is not None:
            freq_domain = np.array(quantum_signal.frequency_domain)
            freqs = fftfreq(len(values), times[1] - times[0])
            
            # Plot positive frequencies only
            positive_freqs = freqs[:len(freqs)//2]
            positive_spectrum = np.abs(freq_domain[:len(freq_domain)//2])
            
            ax2.plot(positive_freqs, positive_spectrum, 'r-', linewidth=1)
            ax2.set_xlabel('Frequency (Hz)')
            ax2.set_ylabel('Magnitude')
            ax2.set_title(f'{title} - Frequency Domain')
            ax2.grid(True, alpha=0.3)
        else:
            # Compute FFT if not already done
            freq_domain = fft(values)
            freqs = fftfreq(len(values), times[1] - times[0])
            
            # Plot positive frequencies only
            positive_freqs = freqs[:len(freqs)//2]
            positive_spectrum = np.abs(freq_domain[:len(freq_domain)//2])
            
            ax2.plot(positive_freqs, positive_spectrum, 'r-', linewidth=1)
            ax2.set_xlabel('Frequency (Hz)')
            ax2.set_ylabel('Magnitude')
            ax2.set_title(f'{title} - Frequency Domain')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

def demo_quantum_signal_processing():
    """Demonstrate quantum signal processing capabilities"""
    print("🔬 Quantum Signal Processing Demo")
    print("=" * 40)
    
    # Create signal processor
    processor = QuantumSignalProcessor()
    
    # Generate quantum noise signal
    print("\n📡 Generating Quantum Noise Signal:")
    quantum_noise = processor.generate_quantum_noise(duration=1.0, sample_rate=1000)
    print(f"   Signal ID: {quantum_noise.signal_id}")
    print(f"   Signal length: {len(quantum_noise.values)} samples")
    print(f"   Duration: 1.0 seconds")
    print(f"   Sample rate: 1000 Hz")
    
    # Apply quantum filtering
    print("\n⚙️  Applying Quantum-Compatible Filtering:")
    filtered_signal = processor.apply_quantum_filter(quantum_noise)
    print(f"   Filtered signal ID: {filtered_signal.signal_id}")
    print(f"   Processing completed: {filtered_signal.processed}")
    
    # Extract quantum features
    print("\n🧬 Extracting Quantum Features:")
    features = processor.extract_quantum_features(quantum_noise)
    for feature, value in features.items():
        print(f"   {feature}: {value:.4f}")
    
    # Process another signal with different characteristics
    print("\n📡 Generating Second Quantum Signal:")
    quantum_signal2 = processor.generate_quantum_noise(duration=0.5, sample_rate=2000)
    print(f"   Signal ID: {quantum_signal2.signal_id}")
    
    # Extract features for second signal
    features2 = processor.extract_quantum_features(quantum_signal2)
    print(f"   Quantum coherence: {features2['quantum_coherence']:.4f}")
    print(f"   Spectral entropy: {features2['spectral_entropy']:.4f}")
    
    print("\n✅ Quantum signal processing demo completed!")

if __name__ == "__main__":
    demo_quantum_signal_processing()