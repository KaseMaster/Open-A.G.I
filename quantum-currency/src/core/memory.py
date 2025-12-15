#!/usr/bin/env python3
"""
Memory Module with Gating Hooks for Global Curvature Resonance System
Implements write phase and macro gating with logging
"""

import logging
import time
import json
import os
from typing import Any, Dict
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from core.gating_service import GatingService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MemoryManager:
    """
    Memory Manager with Gating Hooks
    
    Implements:
    - Gating hooks for write_phase and write_macro
    - Logging for lock events
    - Integration with Gating Service
    """
    
    def __init__(self, gating_service: GatingService, log_file: str = "logs/safe_mode.log"):
        self.gating_service = gating_service
        self.log_file = log_file
        self.memory_store: Dict[str, Any] = {}
        
        # Create logs directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        logger.info("Memory Manager initialized with gating hooks")
    
    def write_phase(self, key: str, value: Any, scale_level: str = "L_phi") -> bool:
        """
        Write phase data with gating hook
        
        Args:
            key: Memory key
            value: Data to write
            scale_level: Scale level (L_phi or L_Phi)
            
        Returns:
            bool: True if write successful, False if gated
        """
        # Check if memory is locked
        if self.gating_service.get_memory_lock_status():
            self._log_write_attempt("WRITE_PHASE_BLOCKED", key, scale_level)
            logger.warning(f"Phase write blocked for {key} at {scale_level}")
            return False
        
        # Perform write operation
        self.memory_store[f"phase_{key}"] = value
        self._log_write_attempt("WRITE_PHASE_SUCCESS", key, scale_level)
        logger.debug(f"Phase written: {key} at {scale_level}")
        return True
    
    def write_macro(self, key: str, value: Any, scale_level: str = "L_Phi") -> bool:
        """
        Write macro data with gating hook
        
        Args:
            key: Memory key
            value: Data to write
            scale_level: Scale level (L_phi or L_Phi)
            
        Returns:
            bool: True if write successful, False if gated
        """
        # Check if memory is locked
        if self.gating_service.get_memory_lock_status():
            self._log_write_attempt("WRITE_MACRO_BLOCKED", key, scale_level)
            logger.warning(f"Macro write blocked for {key} at {scale_level}")
            return False
        
        # Perform write operation
        self.memory_store[f"macro_{key}"] = value
        self._log_write_attempt("WRITE_MACRO_SUCCESS", key, scale_level)
        logger.debug(f"Macro written: {key} at {scale_level}")
        return True
    
    def _log_write_attempt(self, event_type: str, key: str, scale_level: str):
        """
        Log write attempts to safe mode log
        
        Args:
            event_type: Type of write event
            key: Memory key
            scale_level: Scale level
        """
        try:
            log_entry = {
                "timestamp": time.time(),
                "event_type": event_type,
                "key": key,
                "scale_level": scale_level,
                "safe_mode": self.gating_service.get_safe_mode_status(),
                "memory_locked": self.gating_service.get_memory_lock_status()
            }
            
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
        except Exception as e:
            logger.error(f"Error logging write attempt: {e}")
    
    def read(self, key: str) -> Any:
        """
        Read data from memory
        
        Args:
            key: Memory key to read
            
        Returns:
            Any: Data stored at key, or None if not found
        """
        return self.memory_store.get(key, None)
    
    def get_memory_status(self) -> Dict[str, Any]:
        """
        Get current memory status
        
        Returns:
            Dict with memory status information
        """
        return {
            "total_entries": len(self.memory_store),
            "memory_locked": self.gating_service.get_memory_lock_status(),
            "safe_mode": self.gating_service.get_safe_mode_status(),
            "timestamp": time.time()
        }

# Example usage
if __name__ == "__main__":
    # Create gating service
    gating_service = GatingService()
    
    # Create memory manager
    memory_manager = MemoryManager(gating_service)
    
    # Test write operations
    success1 = memory_manager.write_phase("test_phase", {"data": "phase_data"}, "L_phi")
    success2 = memory_manager.write_macro("test_macro", {"data": "macro_data"}, "L_Phi")
    
    print(f"Phase write: {'SUCCESS' if success1 else 'BLOCKED'}")
    print(f"Macro write: {'SUCCESS' if success2 else 'BLOCKED'}")
    
    # Test memory lock
    gating_service.apply_memory_lock(0.75, 0.95)  # This should lock memory
    success3 = memory_manager.write_phase("test_phase_2", {"data": "phase_data_2"}, "L_phi")
    print(f"Phase write with lock: {'SUCCESS' if success3 else 'BLOCKED'}")