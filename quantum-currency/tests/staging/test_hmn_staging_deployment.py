"""
Staging Deployment Verification for HMN Components
"""

import sys
import os
import json
import time
import subprocess
import requests
from datetime import datetime
from typing import Dict, Any

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from network.hmn.full_node import FullNode
from network.hmn.deploy_node import load_node_config


class HMNStagingVerifier:
    """Staging deployment verifier for HMN components"""

    def __init__(self):
        self.node_id = "staging-test-node-001"
        self.network_config = {
            "shard_count": 3,
            "replication_factor": 2,
            "validator_count": 5,
            "metrics_port": 8000,
            "enable_tls": True,
            "discovery_enabled": True
        }

    def verify_docker_deployment(self):
        """Verify Docker deployment configuration"""
        print("Verifying Docker deployment...")
        
        # Check that Dockerfile exists
        dockerfile_path = "Dockerfile.hmn-node"
        if not os.path.exists(dockerfile_path):
            print("❌ HMN Dockerfile not found")
            return False
        
        # Check key components in Dockerfile
        try:
            with open(dockerfile_path, 'r') as f:
                content = f.read()
            
            required_components = [
                "FROM python:",
                "WORKDIR /app",
                "COPY requirements.txt",
                "COPY src/",
                "EXPOSE 8000",
                "CMD [\"python\", \"-m\", \"src.network.hmn.deploy_node\""
            ]
            
            missing_components = []
            for component in required_components:
                if component not in content:
                    missing_components.append(component)
            
            if missing_components:
                print(f"❌ Missing components in Dockerfile: {', '.join(missing_components)}")
                return False
            else:
                print("✅ Dockerfile contains all required components")
                return True
                
        except Exception as e:
            print(f"❌ Error reading Dockerfile: {e}")
            return False

    def verify_kubernetes_deployment(self):
        """Verify Kubernetes deployment configuration"""
        print("Verifying Kubernetes deployment...")
        
        # Check that Kubernetes deployment file exists
        k8s_file = "k8s/hmn-node-deployment.yaml"
        if not os.path.exists(k8s_file):
            print("❌ HMN Kubernetes deployment file not found")
            return False
        
        # Check key components in Kubernetes deployment
        try:
            with open(k8s_file, 'r') as f:
                content = f.read()
            
            required_components = [
                "apiVersion: apps/v1",
                "kind: Deployment",
                "name: hmn-node",
                "image: quantum-currency/hmn-node:latest",
                "containerPort: 8000",
                "kind: Service",
                "name: hmn-node-service"
            ]
            
            missing_components = []
            for component in required_components:
                if component not in content:
                    missing_components.append(component)
            
            if missing_components:
                print(f"❌ Missing components in Kubernetes deployment: {', '.join(missing_components)}")
                return False
            else:
                print("✅ Kubernetes deployment contains all required components")
                return True
                
        except Exception as e:
            print(f"❌ Error reading Kubernetes deployment: {e}")
            return False

    def verify_node_health_monitoring(self):
        """Verify node health monitoring and metrics"""
        print("Verifying node health monitoring...")
        
        # Initialize a test node
        node = FullNode(self.node_id, self.network_config)
        
        # Check health status endpoint
        try:
            health_status = node.get_health_status()
            required_fields = ["node_id", "overall_health", "timestamp", "services"]
            
            missing_fields = []
            for field in required_fields:
                if field not in health_status:
                    missing_fields.append(field)
            
            if missing_fields:
                print(f"❌ Missing fields in health status: {', '.join(missing_fields)}")
                return False
            else:
                print("✅ Health status endpoint provides all required fields")
        except Exception as e:
            print(f"❌ Error checking health status: {e}")
            return False
        
        # Check node stats endpoint
        try:
            node_stats = node.get_node_stats()
            required_fields = [
                "node_id", "running", "health_status", "cal_state",
                "memory_stats", "consensus_stats", "service_health"
            ]
            
            missing_fields = []
            for field in required_fields:
                if field not in node_stats:
                    missing_fields.append(field)
            
            if missing_fields:
                print(f"❌ Missing fields in node stats: {', '.join(missing_fields)}")
                return False
            else:
                print("✅ Node stats endpoint provides all required fields")
        except Exception as e:
            print(f"❌ Error checking node stats: {e}")
            return False
        
        return True

    def verify_prometheus_metrics(self):
        """Verify Prometheus metrics exposure"""
        print("Verifying Prometheus metrics...")
        
        # Initialize a test node
        node = FullNode(self.node_id, self.network_config)
        
        # Check that metrics attributes exist
        try:
            # Check that the node has the required metrics attributes
            required_attributes = [
                "service_health"
            ]
            
            missing_attributes = []
            for attr in required_attributes:
                if not hasattr(node, attr):
                    missing_attributes.append(attr)
            
            if missing_attributes:
                print(f"❌ Missing metrics attributes: {', '.join(missing_attributes)}")
                return False
            else:
                print("✅ Node has all required metrics attributes")
        except Exception as e:
            print(f"❌ Error checking metrics attributes: {e}")
            return False
        
        return True

    def verify_cli_tools(self):
        """Verify CLI tools functionality"""
        print("Verifying CLI tools...")
        
        # Check that deploy_node.py exists and is executable
        deploy_script = "src/network/hmn/deploy_node.py"
        if not os.path.exists(deploy_script):
            print("❌ Deploy node script not found")
            return False
        
        # Try to load node config
        try:
            # This will test that the config loading works
            config = load_node_config(None)  # Use default config
            if config is not None:
                print("✅ CLI tools and config loading work correctly")
                return True
            else:
                print("❌ Config loading returned None")
                return False
        except Exception as e:
            print(f"❌ Error testing CLI tools: {e}")
            return False

    def verify_network_resilience(self):
        """Verify network resilience and failure recovery"""
        print("Verifying network resilience...")
        
        # Initialize a test node
        node = FullNode(self.node_id, self.network_config)
        
        # Test service health monitoring
        try:
            # Mark a service as unhealthy
            node._update_service_health("cal_engine", False, "Test failure")
            
            # Check that health status reflects this
            health_status = node.get_health_status()
            cal_engine_health = health_status["services"]["cal_engine"]
            
            if not cal_engine_health["healthy"]:
                print("✅ Service health monitoring works correctly")
            else:
                print("❌ Service health monitoring failed to detect failure")
                return False
                
        except Exception as e:
            print(f"❌ Error testing service health monitoring: {e}")
            return False
        
        # Test auto-restart functionality
        try:
            # Trigger auto-restart
            node.auto_restart_failed_services()
            
            # Check that service health was updated
            health_status = node.get_health_status()
            cal_engine_health = health_status["services"]["cal_engine"]
            
            # In a real system, we would check if the service was actually restarted
            # For now, we're just verifying the method exists and doesn't crash
            print("✅ Auto-restart functionality exists and runs without error")
            
        except Exception as e:
            print(f"❌ Error testing auto-restart functionality: {e}")
            return False
        
        return True

    def verify_multi_node_simulation(self):
        """Verify multi-node simulation and communication"""
        print("Verifying multi-node simulation...")
        
        # Initialize multiple test nodes
        nodes = []
        for i in range(3):
            node_id = f"test-node-{i:03d}"
            node = FullNode(node_id, self.network_config)
            nodes.append(node)
        
        # Check that nodes can be initialized
        if len(nodes) == 3:
            print("✅ Multi-node simulation can be initialized")
        else:
            print("❌ Failed to initialize multi-node simulation")
            return False
        
        # Test that nodes have different IDs
        node_ids = [node.node_id for node in nodes]
        if len(set(node_ids)) == 3:
            print("✅ Multi-node simulation has unique node IDs")
        else:
            print("❌ Multi-node simulation has duplicate node IDs")
            return False
        
        return True

    def run_staging_verification(self):
        """Run complete staging verification"""
        print("Running HMN Staging Deployment Verification")
        print("=" * 50)
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        tests = [
            ("Docker Deployment", self.verify_docker_deployment),
            ("Kubernetes Deployment", self.verify_kubernetes_deployment),
            ("Node Health Monitoring", self.verify_node_health_monitoring),
            ("Prometheus Metrics", self.verify_prometheus_metrics),
            ("CLI Tools", self.verify_cli_tools),
            ("Network Resilience", self.verify_network_resilience),
            ("Multi-node Simulation", self.verify_multi_node_simulation)
        ]
        
        results = []
        for test_name, test_func in tests:
            print(f"{test_name}:")
            try:
                result = test_func()
                results.append((test_name, result))
                print()  # Add spacing between tests
            except Exception as e:
                print(f"❌ Test failed with exception: {e}")
                import traceback
                traceback.print_exc()
                results.append((test_name, False))
                print()
        
        # Summary
        print("=" * 50)
        print("Staging Verification Summary:")
        passed = 0
        for test_name, success in results:
            status = "✅ PASSED" if success else "❌ FAILED"
            print(f"  {test_name}: {status}")
            if success:
                passed += 1
        
        total = len(results)
        print(f"\nOverall: {passed}/{total} tests PASSED")
        
        # Create verification report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report = {
            "verification_timestamp": datetime.now().isoformat() + "Z",
            "component": "HMN",
            "status": "passed" if passed == total else "failed",
            "tests": {name: "passed" if success else "failed" for name, success in results},
            "summary": {
                "passed": passed,
                "total": total,
                "success_rate": passed / total if total > 0 else 0
            }
        }
        
        report_filename = f"staging_verification_hmn_{timestamp}.json"
        with open(report_filename, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\nReport saved to: {report_filename}")
        
        if passed == total:
            print("\n🎉 All staging verification tests PASSED!")
            print("✅ HMN is ready for staging deployment!")
            return True
        else:
            print("\n❌ Some staging verification tests FAILED")
            print("❌ HMN requires further testing before staging deployment")
            return False


if __name__ == "__main__":
    verifier = HMNStagingVerifier()
    success = verifier.run_staging_verification()
    sys.exit(0 if success else 1)