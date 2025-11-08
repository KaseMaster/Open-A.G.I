#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Real-time Coherence Dashboard for Quantum Currency Emanation Monitor
"""

import json
import logging
import argparse
from datetime import datetime
from typing import Dict, Any, List
from flask import Flask, render_template, jsonify
from flask_socketio import SocketIO, emit
import threading
import time

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CoherenceDashboard")

app = Flask(__name__)
app.config['SECRET_KEY'] = 'quantum-currency-secret-key'
socketio = SocketIO(app, cors_allowed_origins="*")

# In-memory storage for metrics (in production, this would be a database)
metrics_history = []
alerts_history = []

class CoherenceDashboard:
    """
    Real-time coherence dashboard for the Quantum Currency Emanation Monitor.
    """
    
    def __init__(self, data_directory: str = "/mnt/data"):
        """
        Initialize the dashboard.
        
        Args:
            data_directory: Directory where monitoring reports are stored
        """
        self.data_directory = data_directory
        self.running = False
        self.monitoring_thread = None
    
    def load_latest_metrics(self) -> Dict[str, Any]:
        """
        Load the latest metrics from reports.
        
        Returns:
            Dictionary of latest metrics
        """
        try:
            # In a real implementation, this would read from the latest report files
            # For now, we'll simulate with random data
            import random
            
            metrics = {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "coherence_score": round(random.uniform(0.95, 0.99), 6),
                "entropy_rate": round(random.uniform(0.001, 0.003), 6),
                "CAF": round(random.uniform(1.0, 1.1), 6),
                "lambda_L": round(random.uniform(0.5, 1.5), 4),
                "m_t": round(random.uniform(0.3, 1.2), 4),
                "Omega_t": round(random.uniform(0.8, 1.3), 4),
                "Psi": round(random.uniform(0.8, 1.0), 4),
                "stable": random.random() > 0.1  # 90% chance of stability
            }
            
            # Store in history
            metrics_history.append(metrics)
            if len(metrics_history) > 100:  # Keep only last 100 entries
                metrics_history.pop(0)
            
            return metrics
        except Exception as e:
            logger.error(f"Failed to load metrics: {e}")
            return {}
    
    def check_alerts(self, metrics: Dict[str, Any]) -> List[str]:
        """
        Check for alerts based on metrics.
        
        Args:
            metrics: Current metrics
            
        Returns:
            List of alert messages
        """
        alerts = []
        
        if metrics.get("coherence_score", 0) < 0.95:
            alerts.append("CRITICAL: Coherence score below 0.95")
        
        if metrics.get("entropy_rate", 0) > 0.003:
            alerts.append("WARNING: High entropy rate")
        
        if metrics.get("CAF", 0) < 1.0:
            alerts.append("WARNING: Low CAF")
        
        # Store alerts
        for alert in alerts:
            alerts_history.append({
                "timestamp": metrics.get("timestamp", datetime.utcnow().isoformat() + "Z"),
                "message": alert
            })
            if len(alerts_history) > 50:  # Keep only last 50 alerts
                alerts_history.pop(0)
        
        return alerts
    
    def start_monitoring(self):
        """
        Start the background monitoring thread.
        """
        if not self.running:
            self.running = True
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
            self.monitoring_thread.daemon = True
            self.monitoring_thread.start()
            logger.info("Monitoring started")
    
    def stop_monitoring(self):
        """
        Stop the background monitoring thread.
        """
        self.running = False
        if self.monitoring_thread:
            self.monitoring_thread.join()
        logger.info("Monitoring stopped")
    
    def _monitoring_loop(self):
        """
        Background monitoring loop.
        """
        while self.running:
            try:
                # Load latest metrics
                metrics = self.load_latest_metrics()
                
                # Check for alerts
                alerts = self.check_alerts(metrics)
                
                # Emit updates via WebSocket
                socketio.emit('metrics_update', metrics)
                if alerts:
                    socketio.emit('alerts_update', alerts)
                
                # Wait before next update
                time.sleep(5)  # Update every 5 seconds
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(5)  # Continue even if there's an error

# Create dashboard instance
dashboard = CoherenceDashboard()

# Flask routes
@app.route('/')
def index():
    """
    Serve the main dashboard page.
    """
    return render_template('dashboard.html')

@app.route('/api/metrics')
def get_metrics():
    """
    Get current metrics.
    """
    metrics = dashboard.load_latest_metrics()
    return jsonify(metrics)

@app.route('/api/history')
def get_history():
    """
    Get metrics history.
    """
    return jsonify(metrics_history[-20:])  # Last 20 entries

@app.route('/api/alerts')
def get_alerts():
    """
    Get recent alerts.
    """
    return jsonify(alerts_history[-10:])  # Last 10 alerts

@app.route('/api/status')
def get_status():
    """
    Get system status.
    """
    return jsonify({
        "running": dashboard.running,
        "timestamp": datetime.utcnow().isoformat() + "Z"
    })

@socketio.on('connect')
def handle_connect():
    """
    Handle WebSocket connection.
    """
    logger.info("Client connected")
    emit('status', {'connected': True})

@socketio.on('disconnect')
def handle_disconnect():
    """
    Handle WebSocket disconnection.
    """
    logger.info("Client disconnected")

@socketio.on('start_monitoring')
def handle_start_monitoring():
    """
    Handle start monitoring request.
    """
    dashboard.start_monitoring()
    emit('monitoring_status', {'running': True})

@socketio.on('stop_monitoring')
def handle_stop_monitoring():
    """
    Handle stop monitoring request.
    """
    dashboard.stop_monitoring()
    emit('monitoring_status', {'running': False})

def main():
    """
    Main entry point for the dashboard.
    """
    parser = argparse.ArgumentParser(description="Quantum Currency Real-time Coherence Dashboard")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=5000, help="Port to bind to")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--data-dir", default="/mnt/data", help="Data directory")
    
    args = parser.parse_args()
    
    # Update dashboard data directory
    dashboard.data_directory = args.data_dir
    
    # Start monitoring
    dashboard.start_monitoring()
    
    # Run the Flask app
    logger.info(f"Starting dashboard on {args.host}:{args.port}")
    socketio.run(app, host=args.host, port=args.port, debug=args.debug)

if __name__ == "__main__":
    exit(main())