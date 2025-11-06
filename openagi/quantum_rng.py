#!/usr/bin/env python3
"""
Quantum Random Number Generation for Quantum Currency System
Implements photonic or QRNG-based entropy streams for true randomness
"""

import sys
import os
import json
import time
import hashlib
import secrets
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
import numpy as np

# Add parent directory to path to import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

@dataclass
class QuantumEntropySource:
    """Represents a quantum entropy source"""
    source_id: str
    source_type: str  # "photonic", "thermal", "radioactive", "atmospheric"
    last_entropy: Optional[str] = None
    entropy_rate: float = 0.0  # bits per second
    quality_score: float = 0.0  # 0.0 to 1.0
    active: bool = True

@dataclass
class EntropySample:
    """Represents a sample of entropy data"""
    timestamp: float
    source_id: str
    raw_data: str
    processed_entropy: str
    quality_metric: float

class QuantumRNG:
    """
    Implements quantum random number generation using various entropy sources
    """
    
    def __init__(self):
        self.entropy_sources: Dict[str, QuantumEntropySource] = {}
        self.entropy_samples: List[EntropySample] = []
        self.combined_entropy_pool: str = ""
        self._initialize_entropy_sources()
    
    def _initialize_entropy_sources(self):
        """Initialize simulated entropy sources"""
        # In a real implementation, these would connect to actual quantum devices
        sources = [
            QuantumEntropySource(
                source_id="photonic-qe-01",
                source_type="photonic",
                entropy_rate=1000.0,  # 1000 bits per second
                quality_score=0.95
            ),
            QuantumEntropySource(
                source_id="thermal-qe-01",
                source_type="thermal",
                entropy_rate=500.0,  # 500 bits per second
                quality_score=0.85
            ),
            QuantumEntropySource(
                source_id="atmospheric-qe-01",
                source_type="atmospheric",
                entropy_rate=100.0,  # 100 bits per second
                quality_score=0.75
            )
        ]
        
        for source in sources:
            self.entropy_sources[source.source_id] = source
    
    def _generate_photonic_entropy(self, length: int = 32) -> str:
        """
        Simulate photonic entropy generation
        In a real implementation, this would interface with actual photonic quantum devices
        
        Args:
            length: Length of entropy string to generate
            
        Returns:
            Hexadecimal string representing photonic entropy
        """
        # Simulate quantum randomness from photon detection
        # In reality, this would come from quantum processes like photon polarization
        np.random.seed(int(time.time() * 1000000) % (2**32))  # Use time as seed
        
        # Generate random bits based on quantum-like processes
        bits = []
        for _ in range(length * 4):  # 4 bits per hex digit
            # Simulate quantum measurement randomness
            # This could represent photon detection events
            quantum_event = np.random.random()
            
            # Quantum randomness threshold
            if quantum_event > 0.5:
                bits.append('1')
            else:
                bits.append('0')
        
        # Convert bits to hex string
        bit_string = ''.join(bits)
        hex_entropy = ''.join([hex(int(bit_string[i:i+4], 2))[2:] 
                              for i in range(0, len(bit_string), 4)])
        
        return hex_entropy[:length]
    
    def _generate_thermal_entropy(self, length: int = 32) -> str:
        """
        Simulate thermal entropy generation
        In a real implementation, this would measure thermal noise
        
        Args:
            length: Length of entropy string to generate
            
        Returns:
            Hexadecimal string representing thermal entropy
        """
        # Simulate thermal noise entropy
        np.random.seed(int(time.time() * 1000000 + 1) % (2**32))  # Different seed
        
        # Generate entropy from thermal noise simulation
        thermal_values = np.random.normal(0, 1, length * 2)
        
        # Convert to hex string
        hex_entropy = ''.join([format(int(abs(val) * 255) % 16, 'x') 
                              for val in thermal_values])
        
        return hex_entropy[:length]
    
    def _generate_atmospheric_entropy(self, length: int = 32) -> str:
        """
        Simulate atmospheric entropy generation
        In a real implementation, this would measure atmospheric noise
        
        Args:
            length: Length of entropy string to generate
            
        Returns:
            Hexadecimal string representing atmospheric entropy
        """
        # Simulate atmospheric noise entropy
        np.random.seed(int(time.time() * 1000000 + 2) % (2**32))  # Different seed
        
        # Generate entropy from atmospheric noise simulation
        # This could represent radio frequency noise from the atmosphere
        atmospheric_noise = np.random.exponential(1.0, length * 2)
        
        # Convert to hex string
        hex_entropy = ''.join([format(int(noise * 100) % 16, 'x') 
                              for noise in atmospheric_noise])
        
        return hex_entropy[:length]
    
    def collect_entropy(self, source_id: str) -> Optional[EntropySample]:
        """
        Collect entropy from a specific source
        
        Args:
            source_id: Identifier of the entropy source
            
        Returns:
            EntropySample object or None if source not found
        """
        if source_id not in self.entropy_sources:
            return None
        
        source = self.entropy_sources[source_id]
        timestamp = time.time()
        
        # Generate raw entropy based on source type
        if source.source_type == "photonic":
            raw_entropy = self._generate_photonic_entropy(64)
        elif source.source_type == "thermal":
            raw_entropy = self._generate_thermal_entropy(64)
        elif source.source_type == "atmospheric":
            raw_entropy = self._generate_atmospheric_entropy(64)
        else:
            # Fallback to OS-provided entropy
            raw_entropy = secrets.token_hex(32)
        
        # Process entropy (in a real implementation, this would include bias correction)
        processed_entropy = self._process_entropy(raw_entropy)
        
        # Create entropy sample
        sample = EntropySample(
            timestamp=timestamp,
            source_id=source_id,
            raw_data=raw_entropy,
            processed_entropy=processed_entropy,
            quality_metric=source.quality_score
        )
        
        # Store the sample
        self.entropy_samples.append(sample)
        
        # Update source's last entropy
        source.last_entropy = processed_entropy
        
        return sample
    
    def _process_entropy(self, raw_entropy: str) -> str:
        """
        Process raw entropy to improve quality
        In a real implementation, this would include bias correction and whitening
        
        Args:
            raw_entropy: Raw entropy string
            
        Returns:
            Processed entropy string
        """
        # Simple processing: hash to ensure good distribution
        # In a real implementation, this would be more sophisticated
        processed = hashlib.sha256(raw_entropy.encode()).hexdigest()
        return processed
    
    def get_combined_entropy(self, length: int = 32) -> str:
        """
        Get combined entropy from all active sources
        
        Args:
            length: Desired length of entropy string
            
        Returns:
            Combined entropy string
        """
        # Collect entropy from all active sources
        active_samples = []
        for source_id, source in self.entropy_sources.items():
            if source.active:
                sample = self.collect_entropy(source_id)
                if sample:
                    active_samples.append(sample)
        
        if not active_samples:
            # Fallback to OS-provided entropy
            return secrets.token_hex(length)
        
        # Combine entropy from all sources
        combined_raw = ''.join([sample.processed_entropy for sample in active_samples])
        
        # Hash to combine and ensure good distribution
        combined_entropy = hashlib.sha256(combined_raw.encode()).hexdigest()
        
        # Update combined entropy pool
        self.combined_entropy_pool = combined_entropy
        
        return combined_entropy[:length * 2]  # Hex string, 2 chars per byte
    
    def generate_random_bytes(self, length: int = 32) -> bytes:
        """
        Generate cryptographically secure random bytes
        
        Args:
            length: Number of bytes to generate
            
        Returns:
            Random bytes
        """
        # Get combined entropy
        hex_entropy = self.get_combined_entropy(length)
        
        # Convert hex to bytes
        try:
            random_bytes = bytes.fromhex(hex_entropy[:length * 2])
        except ValueError:
            # Fallback if hex conversion fails
            random_bytes = secrets.token_bytes(length)
        
        return random_bytes
    
    def generate_random_int(self, min_val: int = 0, max_val: int = 2**32) -> int:
        """
        Generate a random integer within a range
        
        Args:
            min_val: Minimum value (inclusive)
            max_val: Maximum value (exclusive)
            
        Returns:
            Random integer
        """
        # Calculate range
        range_size = max_val - min_val
        if range_size <= 0:
            return min_val
        
        # Generate enough random bytes for the range
        byte_count = (range_size.bit_length() + 7) // 8
        if byte_count == 0:
            byte_count = 1
        
        random_bytes = self.generate_random_bytes(byte_count)
        random_int = int.from_bytes(random_bytes, byteorder='big')
        
        # Map to range
        return min_val + (random_int % range_size)
    
    def get_entropy_quality_report(self) -> Dict:
        """
        Get a report on the quality of entropy sources
        
        Returns:
            Dictionary with entropy quality metrics
        """
        report = {
            "total_sources": len(self.entropy_sources),
            "active_sources": len([s for s in self.entropy_sources.values() if s.active]),
            "combined_pool_size": len(self.combined_entropy_pool),
            "sources": {}
        }
        
        for source_id, source in self.entropy_sources.items():
            report["sources"][source_id] = {
                "type": source.source_type,
                "quality_score": source.quality_score,
                "entropy_rate": source.entropy_rate,
                "active": source.active,
                "last_entropy": source.last_entropy[:16] if source.last_entropy else None
            }
        
        return report
    
    def add_entropy_source(self, source: QuantumEntropySource):
        """
        Add a new entropy source
        
        Args:
            source: QuantumEntropySource to add
        """
        self.entropy_sources[source.source_id] = source
    
    def deactivate_source(self, source_id: str):
        """
        Deactivate an entropy source
        
        Args:
            source_id: Identifier of the source to deactivate
        """
        if source_id in self.entropy_sources:
            self.entropy_sources[source_id].active = False
    
    def activate_source(self, source_id: str):
        """
        Activate an entropy source
        
        Args:
            source_id: Identifier of the source to activate
        """
        if source_id in self.entropy_sources:
            self.entropy_sources[source_id].active = True

def demo_quantum_rng():
    """Demonstrate quantum random number generation"""
    print("🎲 Quantum Random Number Generation Demo")
    print("=" * 45)
    
    # Create QRNG instance
    qrng = QuantumRNG()
    
    # Show entropy sources
    print("\n📡 Quantum Entropy Sources:")
    for source_id, source in qrng.entropy_sources.items():
        print(f"   {source_id}: {source.source_type} (Quality: {source.quality_score:.2f})")
    
    # Collect entropy from each source
    print("\n🔐 Collecting Entropy:")
    for source_id in qrng.entropy_sources.keys():
        sample = qrng.collect_entropy(source_id)
        if sample:
            print(f"   {source_id}: {sample.processed_entropy[:16]}...")
    
    # Get combined entropy
    print("\n🔗 Combined Entropy:")
    combined = qrng.get_combined_entropy(32)
    print(f"   Combined: {combined}")
    
    # Generate random bytes
    print("\n🔢 Random Byte Generation:")
    for i in range(3):
        random_bytes = qrng.generate_random_bytes(16)
        print(f"   Sample {i+1}: {random_bytes.hex()}")
    
    # Generate random integers
    print("\n🔢 Random Integer Generation:")
    for i in range(3):
        random_int = qrng.generate_random_int(1, 1000)
        print(f"   Sample {i+1}: {random_int}")
    
    # Show quality report
    print("\n📊 Entropy Quality Report:")
    report = qrng.get_entropy_quality_report()
    print(f"   Total sources: {report['total_sources']}")
    print(f"   Active sources: {report['active_sources']}")
    print(f"   Combined pool size: {report['combined_pool_size']} chars")
    
    # Test consensus randomness
    print("\n🏛️  Consensus Round Randomness:")
    consensus_seeds = []
    for round_num in range(5):
        seed = qrng.get_combined_entropy(16)
        consensus_seeds.append(seed)
        print(f"   Round {round_num+1}: {seed}")
    
    print("\n✅ Quantum RNG demo completed!")

# Integration with consensus system
class ConsensusRandomnessProvider:
    """
    Provides quantum randomness for consensus rounds
    """
    
    def __init__(self):
        self.qrng = QuantumRNG()
    
    def get_consensus_seed(self) -> str:
        """
        Get a quantum-random seed for a consensus round
        
        Returns:
            Hexadecimal string representing the seed
        """
        return self.qrng.get_combined_entropy(32)
    
    def get_validator_selection_seed(self) -> str:
        """
        Get a quantum-random seed for validator selection
        
        Returns:
            Hexadecimal string representing the seed
        """
        return self.qrng.get_combined_entropy(16)
    
    def get_transaction_nonce(self) -> int:
        """
        Get a quantum-random nonce for a transaction
        
        Returns:
            Random integer nonce
        """
        return self.qrng.generate_random_int(0, 2**64)

def demo_consensus_integration():
    """Demonstrate integration with consensus system"""
    print("\n🏛️  Consensus System Integration Demo")
    print("=" * 40)
    
    # Create consensus randomness provider
    provider = ConsensusRandomnessProvider()
    
    # Simulate consensus rounds
    print("\n🔄 Consensus Rounds:")
    for round_num in range(3):
        seed = provider.get_consensus_seed()
        print(f"   Round {round_num+1} seed: {seed[:16]}...")
    
    # Simulate validator selection
    print("\n👥 Validator Selection:")
    for i in range(5):
        seed = provider.get_validator_selection_seed()
        validator_id = int(seed, 16) % 100  # Simulate validator ID
        print(f"   Validator {validator_id:02d} selected (seed: {seed[:8]}...)")
    
    # Simulate transaction nonces
    print("\n🧾 Transaction Nonces:")
    for i in range(5):
        nonce = provider.get_transaction_nonce()
        print(f"   Transaction {i+1}: nonce = {nonce}")

if __name__ == "__main__":
    demo_quantum_rng()
    demo_consensus_integration()