#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Prometheus API Connector for Quantum Currency Emanation Monitor
"""

import requests
import json
import logging
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger("PrometheusConnector")

class PrometheusConnector:
    """
    Connector to fetch real metrics from Prometheus server.
    """
    
    def __init__(self, prometheus_url: str, timeout: int = 10):
        """
        Initialize the Prometheus connector.
        
        Args:
            prometheus_url: URL of the Prometheus server
            timeout: Request timeout in seconds
        """
        self.prometheus_url = prometheus_url.rstrip('/')
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'QuantumCurrency-EmanationMonitor/1.0'})
        
    def query(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Execute a PromQL query.
        
        Args:
            query: PromQL query string
            
        Returns:
            Query result as dictionary or None if failed
        """
        try:
            url = f"{self.prometheus_url}/api/v1/query"
            params = {'query': query}
            
            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            
            result = response.json()
            if result.get('status') == 'success':
                return result.get('data', {})
            else:
                logger.error(f"Prometheus query failed: {result.get('error', 'Unknown error')}")
                return None
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to connect to Prometheus: {e}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Prometheus response: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error during Prometheus query: {e}")
            return None
    
    def get_quantum_metrics(self) -> Optional[Dict[str, float]]:
        """
        Fetch core Quantum Currency metrics from Prometheus.
        
        Returns:
            Dictionary of metrics or None if failed
        """
        metrics = {}
        
        # Define the metrics we want to fetch
        metric_queries = {
            'coherence_score': 'quantum_coherence_score',
            'entropy_rate': 'quantum_entropy_rate',
            'CAF': 'quantum_caf',
            'H_internal': 'quantum_h_internal',
            'connected_systems': 'quantum_connected_systems'
        }
        
        # Fetch each metric
        for metric_name, query in metric_queries.items():
            try:
                result = self.query(query)
                if result and 'result' in result and result['result']:
                    # Get the latest value
                    value = float(result['result'][0]['value'][1])
                    metrics[metric_name] = value
                    logger.debug(f"Fetched {metric_name}: {value}")
                else:
                    logger.warning(f"No data returned for metric {metric_name}")
            except (IndexError, ValueError, KeyError) as e:
                logger.warning(f"Failed to parse {metric_name}: {e}")
            except Exception as e:
                logger.error(f"Error fetching {metric_name}: {e}")
        
        return metrics if metrics else None
    
    def get_system_health(self) -> Optional[Dict[str, Any]]:
        """
        Get system health metrics.
        
        Returns:
            Dictionary of health metrics or None if failed
        """
        health_metrics = {}
        
        # Check if Prometheus is healthy
        try:
            response = self.session.get(f"{self.prometheus_url}/-/healthy", timeout=self.timeout)
            health_metrics['prometheus_healthy'] = response.status_code == 200
        except Exception as e:
            logger.error(f"Failed to check Prometheus health: {e}")
            health_metrics['prometheus_healthy'] = False
        
        # Check if Prometheus is ready
        try:
            response = self.session.get(f"{self.prometheus_url}/-/ready", timeout=self.timeout)
            health_metrics['prometheus_ready'] = response.status_code == 200
        except Exception as e:
            logger.error(f"Failed to check Prometheus readiness: {e}")
            health_metrics['prometheus_ready'] = False
        
        return health_metrics
    
    def test_connection(self) -> bool:
        """
        Test connection to Prometheus server.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            # Test with a simple query
            result = self.query('up')
            return result is not None
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False

def main():
    """
    Test the Prometheus connector.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Prometheus Connector")
    parser.add_argument("--prometheus-url", default="http://localhost:9090", 
                       help="Prometheus server URL")
    parser.add_argument("--test-query", default="quantum_coherence_score", 
                       help="Test query to execute")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    print(f"Testing Prometheus connector with URL: {args.prometheus_url}")
    
    # Create connector
    connector = PrometheusConnector(args.prometheus_url)
    
    # Test connection
    print("\n1. Testing connection...")
    if connector.test_connection():
        print("✅ Connection successful")
    else:
        print("❌ Connection failed")
        return 1
    
    # Get system health
    print("\n2. Checking system health...")
    health = connector.get_system_health()
    if health:
        for key, value in health.items():
            status = "✅" if value else "❌"
            print(f"   {status} {key}: {value}")
    else:
        print("❌ Failed to get health metrics")
    
    # Execute test query
    print(f"\n3. Executing test query: {args.test_query}")
    result = connector.query(args.test_query)
    if result:
        print("✅ Query successful")
        print(f"   Result type: {result.get('resultType', 'unknown')}")
        print(f"   Result count: {len(result.get('result', []))}")
    else:
        print("❌ Query failed")
    
    # Get quantum metrics
    print("\n4. Fetching Quantum Currency metrics...")
    metrics = connector.get_quantum_metrics()
    if metrics:
        print("✅ Metrics fetched successfully")
        for key, value in metrics.items():
            print(f"   {key}: {value}")
    else:
        print("⚠️  No metrics available (this is expected in test environment)")
    
    print("\n🎉 Prometheus connector test completed!")
    return 0

if __name__ == "__main__":
    exit(main())