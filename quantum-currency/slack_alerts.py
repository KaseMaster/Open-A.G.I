#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Slack Alert Module for Quantum Currency Emanation Monitor
"""

import requests
import json
import logging
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger("SlackAlerts")

class SlackAlerts:
    """
    Send alerts to Slack channels via webhook.
    """
    
    def __init__(self, webhook_url: str, timeout: int = 10):
        """
        Initialize the Slack alerts module.
        
        Args:
            webhook_url: Slack webhook URL
            timeout: Request timeout in seconds
        """
        self.webhook_url = webhook_url
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'QuantumCurrency-EmanationMonitor/1.0',
            'Content-Type': 'application/json'
        })
    
    def send_alert(self, message: str, alert_type: str = "info", 
                   metrics: Optional[Dict[str, Any]] = None) -> bool:
        """
        Send an alert to Slack.
        
        Args:
            message: Alert message
            alert_type: Type of alert (info, warning, critical)
            metrics: Optional metrics data to include
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Determine emoji based on alert type
            emoji_map = {
                "info": ":information_source:",
                "warning": ":warning:",
                "critical": ":rotating_light:",
                "success": ":white_check_mark:"
            }
            emoji = emoji_map.get(alert_type.lower(), ":information_source:")
            
            # Build the message payload
            payload = {
                "text": f"{emoji} Quantum Currency Alert",
                "attachments": [
                    {
                        "color": self._get_color_for_alert_type(alert_type),
                        "fields": [
                            {
                                "title": "Alert",
                                "value": message,
                                "short": False
                            },
                            {
                                "title": "Timestamp",
                                "value": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
                                "short": True
                            },
                            {
                                "title": "Type",
                                "value": alert_type.title(),
                                "short": True
                            }
                        ]
                    }
                ]
            }
            
            # Add metrics if provided
            if metrics:
                metrics_fields = []
                for key, value in metrics.items():
                    metrics_fields.append({
                        "title": key.replace('_', ' ').title(),
                        "value": str(value),
                        "short": True
                    })
                if metrics_fields:
                    payload["attachments"][0]["fields"].extend(metrics_fields)
            
            # Send the webhook request
            response = self.session.post(
                self.webhook_url,
                data=json.dumps(payload),
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                logger.info(f"Slack alert sent successfully: {message}")
                return True
            else:
                logger.error(f"Failed to send Slack alert: {response.status_code} - {response.text}")
                return False
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to connect to Slack: {e}")
            return False
        except json.JSONDecodeError as e:
            logger.error(f"Failed to encode Slack payload: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error sending Slack alert: {e}")
            return False
    
    def _get_color_for_alert_type(self, alert_type: str) -> str:
        """
        Get color code for alert type.
        
        Args:
            alert_type: Type of alert
            
        Returns:
            Color code for Slack attachment
        """
        color_map = {
            "info": "good",
            "warning": "warning",
            "critical": "danger",
            "success": "good"
        }
        return color_map.get(alert_type.lower(), "good")
    
    def send_coherence_alert(self, coherence_score: float, target: float) -> bool:
        """
        Send a specific coherence alert.
        
        Args:
            coherence_score: Current coherence score
            target: Target coherence score
            
        Returns:
            True if successful, False otherwise
        """
        if coherence_score < target * 0.95:
            alert_type = "critical"
            message = f"CRITICAL: Coherence score {coherence_score:.4f} below 95% of target {target}"
        elif coherence_score < target:
            alert_type = "warning"
            message = f"WARNING: Coherence score {coherence_score:.4f} below target {target}"
        else:
            alert_type = "success"
            message = f"RESOLVED: Coherence score {coherence_score:.4f} restored above target {target}"
        
        metrics = {
            "coherence_score": coherence_score,
            "target": target,
            "delta": coherence_score - target
        }
        
        return self.send_alert(message, alert_type, metrics)
    
    def send_entropy_alert(self, entropy_rate: float, threshold: float) -> bool:
        """
        Send a specific entropy alert.
        
        Args:
            entropy_rate: Current entropy rate
            threshold: Entropy threshold
            
        Returns:
            True if successful, False otherwise
        """
        if entropy_rate > threshold * 1.5:
            alert_type = "critical"
            message = f"CRITICAL: Entropy rate {entropy_rate:.6f} above 150% of threshold {threshold}"
        elif entropy_rate > threshold:
            alert_type = "warning"
            message = f"WARNING: Entropy rate {entropy_rate:.6f} above threshold {threshold}"
        else:
            alert_type = "success"
            message = f"RESOLVED: Entropy rate {entropy_rate:.6f} restored below threshold {threshold}"
        
        metrics = {
            "entropy_rate": entropy_rate,
            "threshold": threshold,
            "ratio": entropy_rate / threshold
        }
        
        return self.send_alert(message, alert_type, metrics)
    
    def send_caf_alert(self, caf: float, target: float) -> bool:
        """
        Send a specific CAF alert.
        
        Args:
            caf: Current CAF value
            target: Target CAF value
            
        Returns:
            True if successful, False otherwise
        """
        if caf < target * 0.9:
            alert_type = "critical"
            message = f"CRITICAL: CAF {caf:.4f} below 90% of target {target}"
        elif caf < target:
            alert_type = "warning"
            message = f"WARNING: CAF {caf:.4f} below target {target}"
        else:
            alert_type = "success"
            message = f"RESOLVED: CAF {caf:.4f} restored above target {target}"
        
        metrics = {
            "CAF": caf,
            "target": target,
            "delta": caf - target
        }
        
        return self.send_alert(message, alert_type, metrics)
    
    def test_connection(self) -> bool:
        """
        Test connection to Slack webhook.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            # Send a test message
            test_payload = {
                "text": ":test_tube: Quantum Currency Emanation Monitor - Connection Test",
                "attachments": [
                    {
                        "color": "good",
                        "fields": [
                            {
                                "title": "Status",
                                "value": "Connection successful",
                                "short": False
                            },
                            {
                                "title": "Timestamp",
                                "value": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
                                "short": True
                            }
                        ]
                    }
                ]
            }
            
            response = self.session.post(
                self.webhook_url,
                data=json.dumps(test_payload),
                timeout=self.timeout
            )
            
            success = response.status_code == 200
            if success:
                logger.info("Slack connection test successful")
            else:
                logger.error(f"Slack connection test failed: {response.status_code} - {response.text}")
            
            return success
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to connect to Slack for test: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error during Slack connection test: {e}")
            return False

def main():
    """
    Test the Slack alerts module.
    """
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description="Test Slack Alerts Module")
    parser.add_argument("--webhook-url", 
                       default=os.environ.get("SLACK_WEBHOOK_URL"),
                       help="Slack webhook URL (or set SLACK_WEBHOOK_URL env var)")
    parser.add_argument("--test-alert", action="store_true",
                       help="Send a test alert")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    if not args.webhook_url:
        print("❌ No webhook URL provided. Set --webhook-url or SLACK_WEBHOOK_URL environment variable.")
        return 1
    
    print(f"Testing Slack alerts with webhook URL: {args.webhook_url[:30]}...")
    
    # Create alerts module
    alerts = SlackAlerts(args.webhook_url)
    
    # Test connection
    print("\n1. Testing connection...")
    if alerts.test_connection():
        print("✅ Connection successful")
    else:
        print("❌ Connection failed")
        return 1
    
    # Send test alert if requested
    if args.test_alert:
        print("\n2. Sending test alert...")
        if alerts.send_alert("This is a test alert from Quantum Currency Emanation Monitor", "info"):
            print("✅ Test alert sent successfully")
        else:
            print("❌ Failed to send test alert")
    
    print("\n🎉 Slack alerts module test completed!")
    return 0

if __name__ == "__main__":
    exit(main())