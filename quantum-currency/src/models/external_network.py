#!/usr/bin/env python3
"""
External Network Integration for Quantum Currency
Implements connectors for inter-system connectivity and coherence bridges
"""

import sys
import os
import json
import time
import hashlib
import secrets
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, asdict, field
import numpy as np

# Add parent directory to path to import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# Import our existing modules
from core.harmonic_validation import compute_spectrum, compute_coherence_score, HarmonicSnapshot
from models.harmonic_wallet import WalletAccount

@dataclass
class ExternalSystem:
    """Represents an external system connected to the Quantum Currency network"""
    system_id: str
    name: str
    url: str
    api_key: str
    connected_at: float
    last_sync: float = 0.0
    coherence_score: float = 0.0
    status: str = "active"  # active, disconnected, syncing

@dataclass
class CoherenceBridge:
    """Represents a coherence bridge between systems"""
    bridge_id: str
    source_system: str
    target_system: str
    created_at: float
    last_validation: float = 0.0
    parity_checksum: float = 0.0
    status: str = "active"  # active, inactive, validating
    data_translation_map: Dict[str, str] = field(default_factory=dict)  # Map for data translation
    last_sync: float = 0.0
    sync_frequency: int = 300  # 5 minutes
    coherence_history: List[float] = field(default_factory=list)

@dataclass
class ResonanceData:
    """Represents resonance data from external systems"""
    system_id: str
    timestamp: float
    coherence: float
    entropy: float
    flow: float
    checksum: str

class ExternalNetworkConnector:
    """
    Implements external network integration with secure APIs and harmonic synchronization
    """
    
    def __init__(self, network_name: str = "quantum-currency-external"):
        self.network_name = network_name
        self.systems: Dict[str, ExternalSystem] = {}
        self.bridges: Dict[str, CoherenceBridge] = {}
        self.resonance_history: List[ResonanceData] = []
        self.feedback_mechanisms = []  # Store real-time feedback
        self.entropy_listeners = {}  # Store entropy listeners
        self.network_config = {
            "sync_interval": 300,  # 5 minutes
            "validation_threshold": 0.01,  # Maximum coherence delta
            "max_connections": 50,
            "encryption_required": True
        }
        self.synchronization_protocols = {
            "harmonic_sync_enabled": True,
            "sync_frequency": 60,  # seconds
            "data_compression": True,
            "checksum_validation": True
        }
    
    def connect_external_system(self, name: str, url: str, api_key: str) -> Optional[str]:
        """
        Connect to an external system
        
        Args:
            name: Name of the external system
            url: URL of the external system
            api_key: API key for authentication
            
        Returns:
            System ID if successful, None otherwise
        """
        # Validate inputs
        if not name or not url or not api_key:
            print("Invalid connection parameters")
            return None
        
        # Check connection limit
        if len(self.systems) >= self.network_config["max_connections"]:
            print("Maximum connections reached")
            return None
        
        # Create system ID
        system_id = hashlib.sha256(f"{name}{url}{time.time()}".encode()).hexdigest()[:32]
        
        # Create external system
        system = ExternalSystem(
            system_id=system_id,
            name=name,
            url=url,
            api_key=api_key,
            connected_at=time.time(),
            last_sync=time.time()
        )
        
        self.systems[system_id] = system
        return system_id
    
    def disconnect_external_system(self, system_id: str) -> bool:
        """
        Disconnect from an external system
        
        Args:
            system_id: ID of the system to disconnect
            
        Returns:
            True if successful, False otherwise
        """
        if system_id not in self.systems:
            print("System not found")
            return False
        
        # Remove system
        del self.systems[system_id]
        
        # Remove associated bridges
        bridges_to_remove = [bridge_id for bridge_id, bridge in self.bridges.items() 
                            if bridge.source_system == system_id or bridge.target_system == system_id]
        for bridge_id in bridges_to_remove:
            del self.bridges[bridge_id]
        
        return True
    
    def create_coherence_bridge(self, source_system: str, target_system: str) -> Optional[str]:
        """
        Create a coherence bridge between two systems
        
        Args:
            source_system: ID of the source system
            target_system: ID of the target system
            
        Returns:
            Bridge ID if successful, None otherwise
        """
        # Validate systems
        if source_system not in self.systems or target_system not in self.systems:
            print("Invalid system IDs")
            return None
        
        # Check if bridge already exists
        for bridge in self.bridges.values():
            if (bridge.source_system == source_system and bridge.target_system == target_system) or \
               (bridge.source_system == target_system and bridge.target_system == source_system):
                print("Bridge already exists between these systems")
                return bridge.bridge_id
        
        # Create bridge ID
        bridge_id = hashlib.sha256(f"{source_system}{target_system}{time.time()}".encode()).hexdigest()[:32]
        
        # Create coherence bridge
        bridge = CoherenceBridge(
            bridge_id=bridge_id,
            source_system=source_system,
            target_system=target_system,
            created_at=time.time()
        )
        
        self.bridges[bridge_id] = bridge
        return bridge_id
    
    def configure_coherence_bridge(self, bridge_id: str, translation_map: Dict[str, str], 
                                 sync_frequency: int = 300) -> bool:
        """
        Configure a coherence bridge with data translation and sync settings
        
        Args:
            bridge_id: ID of the bridge to configure
            translation_map: Dictionary mapping source data fields to target data fields
            sync_frequency: Sync frequency in seconds
            
        Returns:
            True if configuration was successful, False otherwise
        """
        if bridge_id not in self.bridges:
            print("Bridge not found")
            return False
        
        bridge = self.bridges[bridge_id]
        bridge.data_translation_map = translation_map
        bridge.sync_frequency = sync_frequency
        
        return True
    
    def translate_data_for_bridge(self, bridge_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Translate data according to bridge configuration
        
        Args:
            bridge_id: ID of the bridge
            data: Data to translate
            
        Returns:
            Translated data
        """
        if bridge_id not in self.bridges:
            print("Bridge not found")
            return data
        
        bridge = self.bridges[bridge_id]
        translated_data = {}
        
        # Apply translation map
        for source_field, target_field in bridge.data_translation_map.items():
            if source_field in data:
                translated_data[target_field] = data[source_field]
        
        # Add any unmapped fields
        for key, value in data.items():
            if key not in bridge.data_translation_map:
                translated_data[key] = value
        
        return translated_data
    
    def sync_coherence_bridge(self, bridge_id: str) -> Dict[str, Any]:
        """
        Synchronize data across a coherence bridge
        
        Args:
            bridge_id: ID of the bridge to synchronize
            
        Returns:
            Dictionary with synchronization results
        """
        if bridge_id not in self.bridges:
            return {"status": "error", "message": "Bridge not found"}
        
        bridge = self.bridges[bridge_id]
        source_system = self.systems.get(bridge.source_system)
        target_system = self.systems.get(bridge.target_system)
        
        if not source_system or not target_system:
            return {"status": "error", "message": "Connected systems not found"}
        
        # Check if it's time to sync
        current_time = time.time()
        if current_time - bridge.last_sync < bridge.sync_frequency:
            return {"status": "skipped", "message": "Sync frequency not reached"}
        
        # Simulate data synchronization
        # In a real implementation, this would:
        # 1. Fetch data from source system
        # 2. Translate data according to bridge configuration
        # 3. Send data to target system
        # 4. Validate coherence
        
        # For demo purposes, we'll simulate synchronization
        bridge.last_sync = current_time
        coherence_score = np.random.random() * 0.2 + 0.8  # Between 0.8 and 1.0
        bridge.coherence_history.append(coherence_score)
        
        # Keep only recent history (last 100 entries)
        if len(bridge.coherence_history) > 100:
            bridge.coherence_history = bridge.coherence_history[-100:]
        
        return {
            "status": "success",
            "message": f"Synchronized bridge {bridge_id[:8]}...",
            "coherence_score": coherence_score,
            "timestamp": current_time
        }
    
    def validate_harmonic_parity(self, bridge_id: str) -> bool:
        """
        Validate harmonic parity between connected systems
        
        Args:
            bridge_id: ID of the bridge to validate
            
        Returns:
            True if parity is maintained, False otherwise
        """
        if bridge_id not in self.bridges:
            print("Bridge not found")
            return False
        
        bridge = self.bridges[bridge_id]
        source_system = self.systems.get(bridge.source_system)
        target_system = self.systems.get(bridge.target_system)
        
        if not source_system or not target_system:
            print("Connected systems not found")
            return False
        
        # In a real implementation, this would involve:
        # 1. Fetching coherence data from both systems
        # 2. Computing parity checksum
        # 3. Comparing checksums
        
        # For demo purposes, we'll simulate validation
        current_time = time.time()
        checksum = np.random.random()  # Simulated checksum
        
        bridge.parity_checksum = checksum
        bridge.last_validation = current_time
        
        # Simulate validation result
        is_valid = np.random.random() > 0.1  # 90% chance of being valid
        bridge.status = "active" if is_valid else "inactive"
        
        return is_valid
    
    def calculate_real_time_checksum(self, system_id: str) -> Optional[str]:
        """
        Calculate real-time checksum for harmonic parity validation
        
        Args:
            system_id: ID of the system to calculate checksum for
            
        Returns:
            Checksum string if successful, None otherwise
        """
        if system_id not in self.systems:
            print("System not found")
            return None
        
        system = self.systems[system_id]
        
        # Generate time series data for checksum calculation
        duration = 0.5  # 500ms
        sample_rate = 2048  # 2048 Hz
        t = np.linspace(0, duration, int(duration * sample_rate))
        
        # Generate signal for checksum calculation
        freq = np.random.uniform(40, 60)  # Random frequency between 40-60 Hz
        signal = np.sin(2 * np.pi * freq * t)
        
        # Add noise
        noise_level = 0.01
        signal += np.random.normal(0, noise_level, len(signal))
        
        # Compute spectrum
        spectrum = compute_spectrum(t, signal)
        
        # Create checksum from spectrum and timestamp
        checksum_data = f"{system_id}{time.time()}{str(spectrum)[:100]}"
        checksum = hashlib.sha256(checksum_data.encode()).hexdigest()[:32]
        
        return checksum
    
    def validate_real_time_parity(self, bridge_id: str) -> Dict[str, Any]:
        """
        Validate harmonic parity through real-time checksums
        
        Args:
            bridge_id: ID of the bridge to validate
            
        Returns:
            Dictionary with validation results
        """
        if bridge_id not in self.bridges:
            return {"status": "error", "message": "Bridge not found"}
        
        bridge = self.bridges[bridge_id]
        source_system = self.systems.get(bridge.source_system)
        target_system = self.systems.get(bridge.target_system)
        
        if not source_system or not target_system:
            return {"status": "error", "message": "Connected systems not found"}
        
        # Calculate real-time checksums
        source_checksum = self.calculate_real_time_checksum(bridge.source_system)
        target_checksum = self.calculate_real_time_checksum(bridge.target_system)
        
        if not source_checksum or not target_checksum:
            return {"status": "error", "message": "Failed to calculate checksums"}
        
        # Compare checksums
        checksum_delta = abs(hash(source_checksum) - hash(target_checksum))
        is_valid = checksum_delta <= 1e-10  # Very small delta for equality
        
        # Update bridge status
        bridge.parity_checksum = hash(source_checksum)  # Store source checksum
        bridge.last_validation = time.time()
        bridge.status = "active" if is_valid else "inactive"
        
        return {
            "status": "success",
            "valid": is_valid,
            "source_checksum": source_checksum,
            "target_checksum": target_checksum,
            "checksum_delta": checksum_delta,
            "message": "Real-time parity validation completed",
            "timestamp": bridge.last_validation
        }
    
    def get_external_resonance_data(self, system_id: str) -> Optional[ResonanceData]:
        """
        Get resonance data from an external system
        
        Args:
            system_id: ID of the system
            
        Returns:
            ResonanceData if successful, None otherwise
        """
        if system_id not in self.systems:
            print("System not found")
            return None
        
        # In a real implementation, this would fetch data from the external system
        # For demo purposes, we'll simulate the data
        resonance_data = ResonanceData(
            system_id=system_id,
            timestamp=time.time(),
            coherence=np.random.random(),
            entropy=np.random.random() * 0.5,
            flow=np.random.random() * 2 - 1,  # Between -1 and 1
            checksum=hashlib.sha256(f"{system_id}{time.time()}".encode()).hexdigest()[:16]
        )
        
        self.resonance_history.append(resonance_data)
        
        # Keep only recent history
        if len(self.resonance_history) > 1000:
            self.resonance_history = self.resonance_history[-1000:]
        
        return resonance_data
    
    def synchronize_with_external_system(self, system_id: str) -> Dict[str, Any]:
        """
        Synchronize with an external system
        
        Args:
            system_id: ID of the system to synchronize with
            
        Returns:
            Dictionary with synchronization results
        """
        if system_id not in self.systems:
            return {"status": "error", "message": "System not found"}
        
        system = self.systems[system_id]
        
        # In a real implementation, this would:
        # 1. Fetch latest data from the external system
        # 2. Validate coherence
        # 3. Update local state
        # 4. Send updates to the external system if needed
        
        # For demo purposes, we'll simulate synchronization
        system.last_sync = time.time()
        system.coherence_score = np.random.random() * 0.2 + 0.8  # Between 0.8 and 1.0
        
        return {
            "status": "success",
            "message": f"Synchronized with {system.name}",
            "coherence_score": system.coherence_score,
            "timestamp": system.last_sync
        }
    
    def harmonic_synchronization_protocol(self, system_id: str) -> Dict[str, Any]:
        """
        Implement harmonic synchronization protocol for cross-platform data exchange
        
        Args:
            system_id: ID of the system to synchronize with
            
        Returns:
            Dictionary with synchronization results
        """
        if system_id not in self.systems:
            return {"status": "error", "message": "System not found"}
        
        system = self.systems[system_id]
        
        # Apply harmonic synchronization protocol
        # 1. Generate harmonic validation snapshots
        duration = 0.5  # 500ms
        sample_rate = 2048  # 2048 Hz
        t = np.linspace(0, duration, int(duration * sample_rate))
        
        # Generate coherent signals for validation
        freq = np.random.uniform(40, 60)  # Random frequency between 40-60 Hz
        signal1 = np.sin(2 * np.pi * freq * t)
        signal2 = np.sin(2 * np.pi * freq * t)  # Identical signal
        
        # Add small noise
        noise_level = 0.01
        signal1 += np.random.normal(0, noise_level, len(signal1))
        signal2 += np.random.normal(0, noise_level, len(signal2))
        
        # Compute spectra for the snapshots
        spectrum1 = compute_spectrum(t, signal1)
        spectrum2 = compute_spectrum(t, signal2)
        
        # Create spectrum hashes
        spectrum_hash1 = hashlib.sha256(str(spectrum1).encode()).hexdigest()[:32]
        spectrum_hash2 = hashlib.sha256(str(spectrum2).encode()).hexdigest()[:32]
        
        # Create phi parameters
        phi_params = {"phi": 1.618033988749895, "lambda": 0.618033988749895}
        
        # Create HarmonicSnapshot objects for coherence validation
        snapshot1 = HarmonicSnapshot(
            node_id=f"{system_id}-sync-1",
            timestamp=time.time(),
            times=t.tolist(),
            values=signal1.tolist(),
            spectrum=spectrum1,
            spectrum_hash=spectrum_hash1,
            CS=0.0,  # Will be computed later
            phi_params=phi_params
        )
        
        snapshot2 = HarmonicSnapshot(
            node_id=f"{system_id}-sync-2",
            timestamp=time.time(),
            times=t.tolist(),
            values=signal2.tolist(),
            spectrum=spectrum2,
            spectrum_hash=spectrum_hash2,
            CS=0.0,  # Will be computed later
            phi_params=phi_params
        )
        
        # Compute coherence score between the two snapshots
        coherence_score = compute_coherence_score(snapshot1, [snapshot2])
        
        # Update system coherence score
        system.coherence_score = coherence_score
        system.last_sync = time.time()
        
        return {
            "status": "success",
            "message": f"Harmonic synchronization completed with {system.name}",
            "coherence_score": coherence_score,
            "timestamp": system.last_sync
        }
    
    def two_way_resonance_exchange(self, system_id: str) -> Dict[str, Any]:
        """
        Enable two-way resonance exchange with other coherent economies
        
        Args:
            system_id: ID of the system to exchange with
            
        Returns:
            Dictionary with exchange results
        """
        if system_id not in self.systems:
            return {"status": "error", "message": "System not found"}
        
        system = self.systems[system_id]
        
        # Simulate two-way resonance exchange
        # 1. Send local resonance data
        local_resonance = {
            "coherence": np.random.random(),
            "entropy": np.random.random() * 0.5,
            "flow": np.random.random() * 2 - 1
        }
        
        # 2. Receive external resonance data
        external_resonance = self.get_external_resonance_data(system_id)
        
        # 3. Calculate resonance delta
        if external_resonance:
            delta_coherence = abs(local_resonance["coherence"] - external_resonance.coherence)
            delta_entropy = abs(local_resonance["entropy"] - external_resonance.entropy)
            delta_flow = abs(local_resonance["flow"] - external_resonance.flow)
            
            # 4. Validate resonance parity
            is_in_resonance = (delta_coherence <= self.network_config["validation_threshold"] and
                             delta_entropy <= self.network_config["validation_threshold"] and
                             delta_flow <= self.network_config["validation_threshold"])
            
            # 5. Calculate harmonic adjustment if needed
            harmonic_adjustment = 0.0
            if not is_in_resonance:
                # Calculate adjustment to bring systems into resonance
                harmonic_adjustment = min(delta_coherence, delta_entropy, abs(delta_flow))
            
            return {
                "status": "success",
                "message": f"Two-way resonance exchange with {system.name}",
                "local_resonance": local_resonance,
                "external_resonance": {
                    "coherence": external_resonance.coherence,
                    "entropy": external_resonance.entropy,
                    "flow": external_resonance.flow
                },
                "resonance_deltas": {
                    "coherence": delta_coherence,
                    "entropy": delta_entropy,
                    "flow": delta_flow
                },
                "in_resonance": is_in_resonance,
                "harmonic_adjustment": harmonic_adjustment,
                "timestamp": time.time()
            }
        
        return {
            "status": "partial_success",
            "message": f"Sent resonance data to {system.name} but failed to receive response",
            "local_resonance": local_resonance,
            "timestamp": time.time()
        }
    
    def continuous_resonance_exchange(self, system_ids: List[str]) -> Dict[str, Any]:
        """
        Enable continuous two-way resonance exchange with multiple coherent economies
        
        Args:
            system_ids: List of system IDs to exchange with
            
        Returns:
            Dictionary with exchange results
        """
        results = []
        in_resonance_count = 0
        
        for system_id in system_ids:
            if system_id in self.systems:
                # Perform two-way resonance exchange
                result = self.two_way_resonance_exchange(system_id)
                results.append(result)
                
                # Count systems in resonance
                if result.get("in_resonance", False):
                    in_resonance_count += 1
        
        # Calculate overall resonance status
        total_systems = len(system_ids)
        resonance_ratio = in_resonance_count / total_systems if total_systems > 0 else 0
        overall_in_resonance = resonance_ratio >= 0.8  # 80% of systems in resonance
        
        return {
            "status": "success",
            "message": f"Continuous resonance exchange with {len(results)} systems",
            "results": results,
            "overall_in_resonance": overall_in_resonance,
            "resonance_ratio": resonance_ratio,
            "in_resonance_count": in_resonance_count,
            "total_systems": total_systems,
            "timestamp": time.time()
        }
    
    def collect_real_time_feedback(self, system_id: str, feedback_data: Dict[str, Any]) -> bool:
        """
        Collect real-time feedback from connected systems
        
        Args:
            system_id: ID of the system providing feedback
            feedback_data: Dictionary containing feedback data
            
        Returns:
            True if feedback was collected successfully, False otherwise
        """
        if system_id not in self.systems:
            print("System not found")
            return False
        
        # Add timestamp to feedback
        feedback_data["timestamp"] = time.time()
        feedback_data["system_id"] = system_id
        
        # Store feedback
        self.feedback_mechanisms.append(feedback_data)
        
        # Keep only recent feedback (last 1000 entries)
        if len(self.feedback_mechanisms) > 1000:
            self.feedback_mechanisms = self.feedback_mechanisms[-1000:]
        
        return True
    
    def get_real_time_feedback(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get recent real-time feedback from connected systems
        
        Args:
            limit: Maximum number of feedback entries to return
            
        Returns:
            List of feedback entries
        """
        return self.feedback_mechanisms[-limit:] if self.feedback_mechanisms else []
    
    def analyze_feedback_trends(self) -> Dict[str, Any]:
        """
        Analyze trends in real-time feedback from connected systems
        
        Returns:
            Dictionary with trend analysis
        """
        if not self.feedback_mechanisms:
            return {"status": "no_data", "message": "No feedback data available"}
        
        # Extract metrics from feedback
        coherence_values = []
        entropy_values = []
        flow_values = []
        
        for feedback in self.feedback_mechanisms[-100:]:  # Last 100 entries
            if "coherence" in feedback:
                coherence_values.append(feedback["coherence"])
            if "entropy" in feedback:
                entropy_values.append(feedback["entropy"])
            if "flow" in feedback:
                flow_values.append(feedback["flow"])
        
        # Calculate trends
        trends = {}
        
        if coherence_values:
            trends["coherence"] = {
                "current": coherence_values[-1] if coherence_values else 0,
                "average": np.mean(coherence_values),
                "trend": np.polyfit(range(len(coherence_values)), coherence_values, 1)[0] if len(coherence_values) > 1 else 0,
                "stability": 1.0 / (np.var(coherence_values) + 1e-10)
            }
        
        if entropy_values:
            trends["entropy"] = {
                "current": entropy_values[-1] if entropy_values else 0,
                "average": np.mean(entropy_values),
                "trend": np.polyfit(range(len(entropy_values)), entropy_values, 1)[0] if len(entropy_values) > 1 else 0,
                "stability": 1.0 / (np.var(entropy_values) + 1e-10)
            }
        
        if flow_values:
            trends["flow"] = {
                "current": flow_values[-1] if flow_values else 0,
                "average": np.mean(flow_values),
                "trend": np.polyfit(range(len(flow_values)), flow_values, 1)[0] if len(flow_values) > 1 else 0,
                "stability": 1.0 / (np.var(flow_values) + 1e-10)
            }
        
        return {
            "status": "success",
            "trends": trends,
            "total_feedback": len(self.feedback_mechanisms),
            "recent_feedback": len(coherence_values)
        }
    
    def register_entropy_listener(self, listener_id: str, system_id: str) -> bool:
        """
        Register an external resonance listener for interaction entropy monitoring
        
        Args:
            listener_id: Unique identifier for the listener
            system_id: ID of the system to monitor
            
        Returns:
            True if listener was registered successfully, False otherwise
        """
        if system_id not in self.systems:
            print("System not found")
            return False
        
        # Register listener
        self.entropy_listeners[listener_id] = {
            "system_id": system_id,
            "registered_at": time.time(),
            "last_check": time.time(),
            "entropy_history": []
        }
        
        return True
    
    def unregister_entropy_listener(self, listener_id: str) -> bool:
        """
        Unregister an external resonance listener
        
        Args:
            listener_id: Unique identifier for the listener
            
        Returns:
            True if listener was unregistered successfully, False otherwise
        """
        if listener_id not in self.entropy_listeners:
            print("Listener not found")
            return False
        
        # Unregister listener
        del self.entropy_listeners[listener_id]
        
        return True
    
    def monitor_interaction_entropy(self, listener_id: str) -> Dict[str, Any]:
        """
        Monitor interaction entropy for a registered listener
        
        Args:
            listener_id: Unique identifier for the listener
            
        Returns:
            Dictionary with entropy monitoring results
        """
        if listener_id not in self.entropy_listeners:
            return {"status": "error", "message": "Listener not found"}
        
        listener = self.entropy_listeners[listener_id]
        system_id = listener["system_id"]
        
        if system_id not in self.systems:
            return {"status": "error", "message": "System not found"}
        
        # Get resonance data
        resonance_data = self.get_external_resonance_data(system_id)
        
        if not resonance_data:
            return {"status": "error", "message": "Failed to get resonance data"}
        
        # Calculate entropy metrics
        entropy_metrics = {
            "timestamp": time.time(),
            "coherence": resonance_data.coherence,
            "entropy": resonance_data.entropy,
            "flow": resonance_data.flow,
            "checksum": resonance_data.checksum
        }
        
        # Store entropy history
        listener["entropy_history"].append(entropy_metrics)
        listener["last_check"] = time.time()
        
        # Keep only recent history (last 100 entries)
        if len(listener["entropy_history"]) > 100:
            listener["entropy_history"] = listener["entropy_history"][-100:]
        
        # Calculate entropy trends
        entropy_values = [entry["entropy"] for entry in listener["entropy_history"][-10:]]
        if len(entropy_values) > 1:
            entropy_trend = np.polyfit(range(len(entropy_values)), entropy_values, 1)[0]
        else:
            entropy_trend = 0
        
        # Check for high entropy
        is_high_entropy = resonance_data.entropy > 0.3  # Threshold for high entropy
        
        return {
            "status": "success",
            "entropy_metrics": entropy_metrics,
            "entropy_trend": entropy_trend,
            "is_high_entropy": is_high_entropy,
            "history_length": len(listener["entropy_history"])
        }
    
    def get_entropy_monitoring_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all entropy monitoring activities
        
        Returns:
            Dictionary with entropy monitoring summary
        """
        total_listeners = len(self.entropy_listeners)
        high_entropy_count = 0
        
        # Check each listener for high entropy
        for listener_id, listener in self.entropy_listeners.items():
            if listener["entropy_history"]:
                latest_entropy = listener["entropy_history"][-1]["entropy"]
                if latest_entropy > 0.3:  # Threshold for high entropy
                    high_entropy_count += 1
        
        return {
            "status": "success",
            "total_listeners": total_listeners,
            "active_listeners": total_listeners,
            "high_entropy_listeners": high_entropy_count,
            "timestamp": time.time()
        }
    
    def get_network_topology(self) -> Dict[str, Any]:
        """
        Get network topology information
        
        Returns:
            Dictionary with network topology information
        """
        active_systems = len([s for s in self.systems.values() if s.status == "active"])
        active_bridges = len([b for b in self.bridges.values() if b.status == "active"])
        
        return {
            "total_systems": len(self.systems),
            "active_systems": active_systems,
            "total_bridges": len(self.bridges),
            "active_bridges": active_bridges,
            "resonance_history_count": len(self.resonance_history)
        }

def demo_external_network():
    """Demonstrate external network capabilities"""
    print("🌐 External Network Integration Demo")
    print("=" * 35)
    
    # Create external network connector
    network = ExternalNetworkConnector("demo-network")
    
    # Connect to external systems
    print("\n🔗 Connecting to External Systems:")
    system1_id = network.connect_external_system(
        name="Ethereum Bridge",
        url="https://eth-bridge.example.com",
        api_key="eth-api-key-123"
    )
    
    system2_id = network.connect_external_system(
        name="Solana Gateway",
        url="https://sol-gateway.example.com",
        api_key="sol-api-key-456"
    )
    
    if system1_id and system2_id:
        print(f"   Connected to Ethereum Bridge: {system1_id[:16]}...")
        print(f"   Connected to Solana Gateway: {system2_id[:16]}...")
    else:
        print("   Failed to connect to external systems")
        return
    
    # Create coherence bridge
    print("\n🌉 Creating Coherence Bridge:")
    bridge_id = network.create_coherence_bridge(system1_id, system2_id)
    if bridge_id:
        print(f"   Bridge created: {bridge_id[:16]}...")
    else:
        print("   Failed to create coherence bridge")
    
    # Validate harmonic parity
    print("\n⚖️  Validating Harmonic Parity:")
    if bridge_id:
        is_valid = network.validate_harmonic_parity(bridge_id)
        print(f"   Parity validation: {'✅ Valid' if is_valid else '❌ Invalid'}")
    
    # Get resonance data
    print("\n📊 Getting External Resonance Data:")
    resonance_data = network.get_external_resonance_data(system1_id)
    if resonance_data:
        print(f"   Coherence: {resonance_data.coherence:.3f}")
        print(f"   Entropy: {resonance_data.entropy:.3f}")
        print(f"   Flow: {resonance_data.flow:.3f}")
    
    # Synchronize with external system
    print("\n🔄 Synchronizing with External System:")
    sync_result = network.synchronize_with_external_system(system1_id)
    if sync_result["status"] == "success":
        print(f"   {sync_result['message']}")
        print(f"   Coherence Score: {sync_result['coherence_score']:.3f}")
    
    # Show network topology
    print("\n🌐 Network Topology:")
    topology = network.get_network_topology()
    print(f"   Total Systems: {topology['total_systems']}")
    print(f"   Active Systems: {topology['active_systems']}")
    print(f"   Total Bridges: {topology['total_bridges']}")
    print(f"   Active Bridges: {topology['active_bridges']}")
    
    print("\n✅ External network demo completed!")

if __name__ == "__main__":
    demo_external_network()