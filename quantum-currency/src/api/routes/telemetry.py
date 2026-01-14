#!/usr/bin/env python3
"""
Telemetry API for IACE v2.0 Dashboard Integration
Exposes real-time KPI streaming and historical trend data
"""

from flask import Blueprint, jsonify
import json
from typing import Dict, Any
from src.monitoring.telemetry_streamer import telemetry_streamer

# Create blueprint
telemetry_bp = Blueprint('telemetry', __name__)

@telemetry_bp.route('/telemetry/current', methods=['GET'])
def get_current_telemetry():
    """
    Get current telemetry data
    """
    try:
        current_kpis = telemetry_streamer.get_current_kpis()
        return jsonify({
            "status": "success",
            "data": current_kpis
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Error fetching current telemetry: {str(e)}"
        }), 500

@telemetry_bp.route('/telemetry/history/<metric_name>', methods=['GET'])
def get_telemetry_history(metric_name: str):
    """
    Get historical telemetry data for a specific metric
    """
    try:
        # Get last 24 hours of data
        history = telemetry_streamer.get_historical_trends(metric_name, hours=24)
        return jsonify({
            "status": "success",
            "metric": metric_name,
            "data": history
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Error fetching telemetry history: {str(e)}"
        }), 500

@telemetry_bp.route('/telemetry/metrics', methods=['GET'])
def get_available_metrics():
    """
    Get list of available metrics
    """
    try:
        current_kpis = telemetry_streamer.get_current_kpis()
        metric_names = list(current_kpis.keys()) if current_kpis else []
        
        return jsonify({
            "status": "success",
            "metrics": metric_names
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Error fetching metrics list: {str(e)}"
        }), 500

@telemetry_bp.route('/telemetry/status', methods=['GET'])
def get_telemetry_status():
    """
    Get telemetry system status
    """
    try:
        current_kpis = telemetry_streamer.get_current_kpis()
        history_count = len(telemetry_streamer.kpi_history)
        
        return jsonify({
            "status": "success",
            "system_status": "active",
            "latest_update": current_kpis.get("timestamp", 0) if current_kpis else 0,
            "history_records": history_count,
            "subscribers": len(telemetry_streamer.subscribers)
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Error fetching telemetry status: {str(e)}"
        }), 500

# Example usage
if __name__ == "__main__":
    print("Telemetry API Module")
    print("This module should be integrated with the main Flask application")
    print("Endpoints:")
    print("  GET /telemetry/current - Current telemetry data")
    print("  GET /telemetry/history/<metric_name> - Historical data for metric")
    print("  GET /telemetry/metrics - List of available metrics")
    print("  GET /telemetry/status - Telemetry system status")