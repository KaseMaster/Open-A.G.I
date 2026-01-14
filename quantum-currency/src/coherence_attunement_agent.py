#!/usr/bin/env python3
"""
🧠 Coherence Attunement Agent (CAA) - Unified Self-Healing Layer
This module implements the Coherence Attunement Agent that provides:
1. API server lifecycle management
2. Coherence health validation using λ(t) and Ĉ(t) metrics
3. Automated healing and recovery mechanisms
"""

import requests
import time
import subprocess
import sys
import os
import psutil
import threading
from typing import Optional, Tuple, Dict, Any, List
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='[AGENT] %(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class CoherenceAttunementAgent:
    """Coherence Attunement Agent for Quantum Currency System"""
    
    def __init__(self, host: str = "127.0.0.1", port: int = 5000, log_path: str = "logs/full_self_healing_log.txt"):
        self.host = host
        self.port = port
        self.base_url = f"http://{host}:{port}"
        self.log_path = log_path
        self.server_process: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()
        
        # Create logs directory if it doesn't exist
        log_dir = os.path.dirname(log_path)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
    
    def _log(self, message: str):
        """Log message to both console and file"""
        logger.info(message)
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(f"[AGENT] {time.strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")
        except Exception as e:
            logger.warning(f"Failed to write to log file: {e}")
    
    def is_server_running(self) -> bool:
        """Check if the API server is running by testing the health endpoint"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except:
            try:
                # Fallback to root endpoint
                response = requests.get(f"{self.base_url}/", timeout=5)
                return response.status_code == 200
            except:
                return False
    
    def kill_existing_processes(self) -> None:
        """Kill any existing Python processes that might be using the port"""
        self._log("Checking for existing Python processes...")
        
        # Kill any processes using the port
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                # Check if it's a Python process
                if proc.info['name'] == 'python.exe' or proc.info['name'] == 'python':
                    # Check if it's using our port
                    for conn in proc.connections():
                        if conn.laddr.port == self.port:
                            self._log(f"Killing process {proc.info['pid']} using port {self.port}")
                            proc.kill()
                            proc.wait(timeout=5)
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass
        
        # Additional cleanup using taskkill
        try:
            subprocess.run(["taskkill", "/f", "/im", "python.exe"], 
                          capture_output=True, timeout=10)
        except:
            pass
            
        time.sleep(2)  # Wait for processes to terminate
    
    def start_server(self, timeout: int = 30) -> bool:
        """Start the API server and wait for it to be ready"""
        with self._lock:
            # Check if server is already running
            if self.is_server_running():
                self._log("API server is already running")
                return True
            
            # Kill any existing processes
            self.kill_existing_processes()
            
            # Start the server
            self._log("Starting API server...")
            try:
                # Change to the correct directory
                api_script_path = os.path.join("src", "api", "main.py")
                
                # Start the server process
                self.server_process = subprocess.Popen([
                    sys.executable, api_script_path
                ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=".")
                
                # Wait for server to start
                start_time = time.time()
                while time.time() - start_time < timeout:
                    if self.is_server_running():
                        pid = self.server_process.pid if self.server_process else "Unknown"
                        self._log(f"API server started successfully (PID: {pid})")
                        return True
                    time.sleep(1)
                    self._log(f"Waiting for API server to start... ({int(time.time() - start_time)}s)")
                
                self._log("API server failed to start within timeout period")
                return False
                
            except Exception as e:
                self._log(f"Error starting API server: {e}")
                return False
    
    def stop_server(self) -> bool:
        """Stop the API server"""
        with self._lock:
            if self.server_process:
                try:
                    self._log("Stopping API server...")
                    self.server_process.terminate()
                    self.server_process.wait(timeout=10)
                    self._log("API server stopped successfully")
                    self.server_process = None
                    return True
                except subprocess.TimeoutExpired:
                    self._log("Force killing API server...")
                    if self.server_process:
                        self.server_process.kill()
                        self.server_process.wait()
                    self.server_process = None
                    return True
                except Exception as e:
                    self._log(f"Error stopping API server: {e}")
                    return False
            else:
                # Kill any remaining processes
                self.kill_existing_processes()
                return True
    
    def restart_server(self, timeout: int = 30) -> bool:
        """Restart the API server"""
        self._log("Restarting API server...")
        self.stop_server()
        time.sleep(2)
        return self.start_server(timeout)
    
    def check_attunement_health(self) -> bool:
        """
        Check system coherence health via API endpoints
        Returns True if system is coherent, False otherwise
        """
        self._log("Checking system coherence health...")
        
        try:
            # Check health endpoint for λ(t) and Ĉ(t) metrics
            response = requests.get(f"{self.base_url}/health", timeout=10)
            
            if response.status_code == 200:
                health_data = response.json()
                
                # Extract coherence metrics
                lambda_t = health_data.get("lambda_t", 0)
                c_t = health_data.get("c_t", 0)
                
                self._log(f"Attunement OK: λ(t)={lambda_t:.3f}, Ĉ(t)={c_t:.3f}")
                
                # Check if coherence is within acceptable range
                # λ(t) should be close to 1.0 and Ĉ(t) should be > 0.8 for healthy system
                if 0.95 <= lambda_t <= 1.05 and c_t >= 0.8:
                    return True
                else:
                    self._log(f"Coherence metrics outside normal range: λ(t)={lambda_t:.3f}, Ĉ(t)={c_t:.3f}")
                    return False
            else:
                self._log(f"Health endpoint returned status {response.status_code}")
                return False
                
        except requests.exceptions.ConnectionError:
            self._log("Connection error - API server may not be running")
            return False
        except requests.exceptions.Timeout:
            self._log("Health check timeout")
            return False
        except Exception as e:
            self._log(f"Error during health check: {e}")
            return False

if __name__ == "__main__":
    # Example usage
    agent = CoherenceAttunementAgent()
    
    if agent.start_server():
        print("✅ Server started successfully")
        
        # Check health
        if agent.check_attunement_health():
            print("✅ System is coherent")
        else:
            print("⚠️ System coherence issues detected")
        
        # Stop server
        agent.stop_server()
        print("🛑 Server stopped")
    else:
        print("❌ Failed to start server")