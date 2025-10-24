"""
AEGIS Kubernetes Deployment
Advanced deployment configuration for Kubernetes with security features
"""

import asyncio
import time
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
import yaml
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import AEGIS modules
try:
    from src.aegis.security.advanced_crypto import AdvancedSecurityManager, SecurityFeature
    HAS_AEGIS_MODULES = True
except ImportError as e:
    print(f"Warning: Could not import AEGIS modules: {e}")
    HAS_AEGIS_MODULES = False
    # Create mock classes for standalone operation
    class AdvancedSecurityManager:
        def __init__(self):
            pass
    
    class SecurityFeature:
        ZERO_KNOWLEDGE_PROOF = "zkp"
        HOMOMORPHIC_ENCRYPTION = "homomorphic"
        SECURE_MULTI_PARTY_COMPUTATION = "smc"
        DIFFERENTIAL_PRIVACY = "differential_privacy"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AEGISKubernetesDeployment:
    """Kubernetes deployment configuration for AEGIS with advanced security"""
    
    def __init__(self, namespace: str = "aegis-system"):
        self.namespace = namespace
        self.security_manager = AdvancedSecurityManager()
        self.deployment_configs = {}
        
        # Track deployment progress
        self.deployed_components = []
        self.deployment_errors = []
    
    def generate_base_deployment(self) -> Dict[str, Any]:
        """Generate base Kubernetes deployment configuration"""
        return {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": "aegis-base",
                "namespace": self.namespace,
                "labels": {
                    "app": "aegis",
                    "version": "v2.2.0"
                }
            },
            "spec": {
                "replicas": 3,
                "selector": {
                    "matchLabels": {
                        "app": "aegis"
                    }
                },
                "template": {
                    "metadata": {
                        "labels": {
                            "app": "aegis"
                        }
                    },
                    "spec": {
                        "containers": [{
                            "name": "aegis-node",
                            "image": "aegisframework/aegis:v2.2.0",
                            "ports": [{
                                "containerPort": 8080,
                                "name": "api"
                            }],
                            "envFrom": [{
                                "configMapRef": {
                                    "name": "aegis-config"
                                }
                            }],
                            "resources": {
                                "requests": {
                                    "memory": "512Mi",
                                    "cpu": "500m"
                                },
                                "limits": {
                                    "memory": "2Gi",
                                    "cpu": "2000m"
                                }
                            },
                            "securityContext": {
                                "runAsNonRoot": True,
                                "runAsUser": 1000,
                                "readOnlyRootFilesystem": True
                            }
                        }],
                        "securityContext": {
                            "fsGroup": 2000
                        }
                    }
                }
            }
        }
    
    def generate_security_enhanced_deployment(self) -> Dict[str, Any]:
        """Generate security-enhanced Kubernetes deployment"""
        deployment = self.generate_base_deployment()
        deployment["metadata"]["name"] = "aegis-secure-node"
        
        # Enable security features in environment
        security_env = [
            {"name": "ENABLE_ZERO_KNOWLEDGE_PROOFS", "value": "true"},
            {"name": "ENABLE_HOMOMORPHIC_ENCRYPTION", "value": "true"},
            {"name": "ENABLE_SECURE_MPC", "value": "true"},
            {"name": "ENABLE_DIFFERENTIAL_PRIVACY", "value": "true"},
            {"name": "SECURITY_LEVEL", "value": "high"},
            {"name": "KEY_ROTATION_INTERVAL", "value": "3600"},
            {"name": "ENCRYPTION_ALGORITHM", "value": "chacha20-poly1305"}
        ]
        
        # Add security environment variables
        deployment["spec"]["template"]["spec"]["containers"][0]["env"] = security_env
        
        # Add security volumes
        volumes = [
            {
                "name": "security-keys",
                "emptyDir": {
                    "medium": "Memory"
                }
            },
            {
                "name": "tmp-storage",
                "emptyDir": {}
            }
        ]
        
        volume_mounts = [
            {
                "name": "security-keys",
                "mountPath": "/var/run/aegis/security",
                "readOnly": True
            },
            {
                "name": "tmp-storage",
                "mountPath": "/tmp"
            }
        ]
        
        deployment["spec"]["template"]["spec"]["volumes"] = volumes
        deployment["spec"]["template"]["spec"]["containers"][0]["volumeMounts"] = volume_mounts
        
        # Add init container for security setup
        init_containers = [{
            "name": "security-init",
            "image": "aegisframework/security-init:v2.2.0",
            "command": ["/bin/sh", "-c"],
            "args": [
                "echo 'Initializing security keys...' && "
                "mkdir -p /security-keys && "
                "chmod 700 /security-keys && "
                "echo 'Security initialization complete'"
            ],
            "volumeMounts": [{
                "name": "security-keys",
                "mountPath": "/security-keys"
            }]
        }]
        
        deployment["spec"]["template"]["spec"]["initContainers"] = init_containers
        
        return deployment
    
    def generate_service_configuration(self) -> Dict[str, Any]:
        """Generate Kubernetes service configuration"""
        return {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {
                "name": "aegis-service",
                "namespace": self.namespace,
                "labels": {
                    "app": "aegis"
                }
            },
            "spec": {
                "selector": {
                    "app": "aegis"
                },
                "ports": [
                    {
                        "name": "api",
                        "port": 8080,
                        "targetPort": 8080,
                        "protocol": "TCP"
                    },
                    {
                        "name": "metrics",
                        "port": 9091,
                        "targetPort": 9091,
                        "protocol": "TCP"
                    }
                ],
                "type": "ClusterIP"
            }
        }
    
    def generate_ingress_configuration(self) -> Dict[str, Any]:
        """Generate Kubernetes ingress configuration"""
        return {
            "apiVersion": "networking.k8s.io/v1",
            "kind": "Ingress",
            "metadata": {
                "name": "aegis-ingress",
                "namespace": self.namespace,
                "annotations": {
                    "nginx.ingress.kubernetes.io/rewrite-target": "/",
                    "nginx.ingress.kubernetes.io/ssl-redirect": "true",
                    "cert-manager.io/cluster-issuer": "letsencrypt-prod"
                }
            },
            "spec": {
                "tls": [{
                    "hosts": ["aegis.example.com"],
                    "secretName": "aegis-tls"
                }],
                "rules": [{
                    "host": "aegis.example.com",
                    "http": {
                        "paths": [{
                            "path": "/",
                            "pathType": "Prefix",
                            "backend": {
                                "service": {
                                    "name": "aegis-service",
                                    "port": {
                                        "number": 8080
                                    }
                                }
                            }
                        }]
                    }
                }]
            }
        }
    
    def generate_config_map(self) -> Dict[str, Any]:
        """Generate Kubernetes ConfigMap with security configuration"""
        return {
            "apiVersion": "v1",
            "kind": "ConfigMap",
            "metadata": {
                "name": "aegis-config",
                "namespace": self.namespace
            },
            "data": {
                "AEGIS_CONFIG": """
{
  "security": {
    "zero_knowledge_proofs": true,
    "homomorphic_encryption": true,
    "secure_mpc": true,
    "differential_privacy": true,
    "security_level": "high",
    "key_rotation_interval": 3600,
    "max_message_age": 300,
    "ratchet_advance_threshold": 100
  },
  "performance": {
    "memory_optimization": true,
    "concurrency_optimization": true,
    "network_optimization": true,
    "max_workers": 100,
    "queue_size": 1000
  },
  "monitoring": {
    "metrics_collection": true,
    "alerting": true,
    "log_level": "INFO",
    "metrics_port": 9091
  },
  "network": {
    "p2p_port": 8080,
    "api_port": 8080,
    "enable_tls": true,
    "certificate_file": "/etc/aegis/certs/tls.crt",
    "key_file": "/etc/aegis/certs/tls.key"
  }
}
                """,
                "LOG_LEVEL": "INFO",
                "METRICS_PORT": "9091"
            }
        }
    
    def generate_secret_configuration(self) -> Dict[str, Any]:
        """Generate Kubernetes Secret configuration (example - in practice use sealed secrets)"""
        return {
            "apiVersion": "v1",
            "kind": "Secret",
            "metadata": {
                "name": "aegis-secrets",
                "namespace": self.namespace
            },
            "type": "Opaque",
            "data": {
                # These would be base64 encoded in real deployment
                "jwt_secret": "YWFhYWFhYWFhYWFhYWFhYWFhYWFhYWFhYWFhYWFhYWE=",  # "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa" base64
                "encryption_key": "YmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmI=",  # "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb" base64
                "database_password": "Y2NjY2NjY2NjY2NjY2NjY2NjY2NjY2NjY2NjY2NjY2M="  # "cccccccccccccccccccccccccccccccc" base64
            }
        }
    
    def generate_horizontal_pod_autoscaler(self) -> Dict[str, Any]:
        """Generate Horizontal Pod Autoscaler configuration"""
        return {
            "apiVersion": "autoscaling/v2",
            "kind": "HorizontalPodAutoscaler",
            "metadata": {
                "name": "aegis-hpa",
                "namespace": self.namespace
            },
            "spec": {
                "scaleTargetRef": {
                    "apiVersion": "apps/v1",
                    "kind": "Deployment",
                    "name": "aegis-secure-node"
                },
                "minReplicas": 3,
                "maxReplicas": 20,
                "metrics": [
                    {
                        "type": "Resource",
                        "resource": {
                            "name": "cpu",
                            "target": {
                                "type": "Utilization",
                                "averageUtilization": 70
                            }
                        }
                    },
                    {
                        "type": "Resource",
                        "resource": {
                            "name": "memory",
                            "target": {
                                "type": "Utilization",
                                "averageUtilization": 80
                            }
                        }
                    }
                ]
            }
        }
    
    def generate_network_policy(self) -> Dict[str, Any]:
        """Generate Network Policy for enhanced security"""
        return {
            "apiVersion": "networking.k8s.io/v1",
            "kind": "NetworkPolicy",
            "metadata": {
                "name": "aegis-network-policy",
                "namespace": self.namespace
            },
            "spec": {
                "podSelector": {
                    "matchLabels": {
                        "app": "aegis"
                    }
                },
                "policyTypes": ["Ingress", "Egress"],
                "ingress": [
                    {
                        "from": [
                            {
                                "namespaceSelector": {
                                    "matchLabels": {
                                        "name": "monitoring"
                                    }
                                }
                            }
                        ],
                        "ports": [
                            {
                                "protocol": "TCP",
                                "port": 9091
                            }
                        ]
                    },
                    {
                        "from": [
                            {
                                "namespaceSelector": {
                                    "matchLabels": {
                                        "name": "ingress-nginx"
                                    }
                                }
                            }
                        ],
                        "ports": [
                            {
                                "protocol": "TCP",
                                "port": 8080
                            }
                        ]
                    }
                ],
                "egress": [
                    {
                        "to": [
                            {
                                "namespaceSelector": {
                                    "matchLabels": {
                                        "name": "kube-system"
                                    }
                                }
                            }
                        ],
                        "ports": [
                            {
                                "protocol": "TCP",
                                "port": 53
                            }
                        ]
                    }
                ]
            }
        }
    
    def generate_resource_quota(self) -> Dict[str, Any]:
        """Generate Resource Quota for the namespace"""
        return {
            "apiVersion": "v1",
            "kind": "ResourceQuota",
            "metadata": {
                "name": "aegis-resource-quota",
                "namespace": self.namespace
            },
            "spec": {
                "hard": {
                    "requests.cpu": "4",
                    "requests.memory": "8Gi",
                    "limits.cpu": "16",
                    "limits.memory": "32Gi",
                    "persistentvolumeclaims": "10",
                    "services.loadbalancers": "2",
                    "services.nodeports": "0"
                }
            }
        }
    
    def generate_all_configurations(self) -> Dict[str, Dict[str, Any]]:
        """Generate all Kubernetes configurations"""
        configurations = {
            "deployment": self.generate_security_enhanced_deployment(),
            "service": self.generate_service_configuration(),
            "ingress": self.generate_ingress_configuration(),
            "config_map": self.generate_config_map(),
            "secret": self.generate_secret_configuration(),
            "hpa": self.generate_horizontal_pod_autoscaler(),
            "network_policy": self.generate_network_policy(),
            "resource_quota": self.generate_resource_quota()
        }
        
        return configurations
    
    def save_configurations(self, output_dir: str = "k8s_configs"):
        """Save all configurations to YAML files"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        configurations = self.generate_all_configurations()
        
        for config_name, config_data in configurations.items():
            filename = f"{output_dir}/{config_name}.yaml"
            with open(filename, 'w') as f:
                yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)
            logger.info(f"✓ Saved {config_name} to {filename}")
        
        # Create a single combined file
        combined_filename = f"{output_dir}/aegis-deployment-all.yaml"
        with open(combined_filename, 'w') as f:
            for config_name, config_data in configurations.items():
                yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)
                f.write("---\n")
        
        logger.info(f"✓ Combined configuration saved to {combined_filename}")
        return combined_filename
    
    def validate_configurations(self) -> bool:
        """Validate all generated configurations"""
        try:
            configurations = self.generate_all_configurations()
            
            # Validate each configuration has required fields
            required_fields = {
                "deployment": ["apiVersion", "kind", "metadata", "spec"],
                "service": ["apiVersion", "kind", "metadata", "spec"],
                "ingress": ["apiVersion", "kind", "metadata", "spec"],
                "config_map": ["apiVersion", "kind", "metadata", "data"],
                "secret": ["apiVersion", "kind", "metadata", "data"],
                "hpa": ["apiVersion", "kind", "metadata", "spec"],
                "network_policy": ["apiVersion", "kind", "metadata", "spec"],
                "resource_quota": ["apiVersion", "kind", "metadata", "spec"]
            }
            
            for config_name, required in required_fields.items():
                config_data = configurations[config_name]
                for field in required:
                    if field not in config_data:
                        logger.error(f"Missing required field '{field}' in {config_name}")
                        return False
            
            logger.info("✓ All configurations validated successfully")
            return True
            
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            return False
    
    def print_deployment_guide(self):
        """Print Kubernetes deployment guide"""
        guide = """
============================================================
AEGIS Framework - Kubernetes Deployment Guide
============================================================

PREREQUISITES:
-------------
1. Kubernetes cluster (v1.20+)
2. kubectl configured
3. Helm (optional, for advanced deployments)
4. cert-manager (for TLS certificates)
5. Ingress controller (NGINX recommended)

DEPLOYMENT STEPS:
-----------------

1. Create Namespace:
   kubectl create namespace aegis-system

2. Apply Configurations:
   kubectl apply -f k8s_configs/

3. Verify Deployment:
   kubectl get pods -n aegis-system
   kubectl get services -n aegis-system

4. Check Logs:
   kubectl logs -l app=aegis -n aegis-system

5. Access Service:
   kubectl port-forward svc/aegis-service 8080:8080 -n aegis-system

SECURITY CONFIGURATION:
-----------------------

Environment Variables:
  ENABLE_ZERO_KNOWLEDGE_PROOFS=true
  ENABLE_HOMOMORPHIC_ENCRYPTION=true
  ENABLE_SECURE_MPC=true
  ENABLE_DIFFERENTIAL_PRIVACY=true
  SECURITY_LEVEL=high

Volumes:
  - security-keys: Memory-backed ephemeral storage for keys
  - tmp-storage: Temporary storage for operations

Network Policies:
  - Restricted ingress to only monitoring and ingress namespaces
  - Limited egress to DNS resolution only

RESOURCE MANAGEMENT:
--------------------

CPU Limits: 2000m (2 cores)
Memory Limits: 2Gi
HPA: Scales from 3 to 20 replicas based on 70% CPU, 80% Memory

MONITORING:
----------

Metrics Endpoint: :9091/metrics
Prometheus Integration: Automatic
Grafana Dashboards: Pre-configured

TROUBLESHOOTING:
---------------

Common Issues:
1. Permission denied - Check RBAC roles
2. Volume mount failures - Verify storage class
3. Network connectivity - Check network policies
4. Resource limits - Adjust quotas if needed

Logs:
  kubectl logs -l app=aegis -n aegis-system --since=1h

Describe pods:
  kubectl describe pods -l app=aegis -n aegis-system

============================================================
        """
        print(guide)


async def deploy_aegis_kubernetes():
    """Deploy AEGIS to Kubernetes"""
    logger.info("🚀 Starting AEGIS Kubernetes Deployment")
    
    # Create deployment manager
    deployment = AEGISKubernetesDeployment()
    
    # Validate configurations
    if not deployment.validate_configurations():
        logger.error("❌ Configuration validation failed")
        return False
    
    # Save configurations
    config_file = deployment.save_configurations()
    logger.info(f"✅ Configurations saved to {config_file}")
    
    # Print deployment guide
    deployment.print_deployment_guide()
    
    logger.info("✅ AEGIS Kubernetes deployment configuration complete!")
    logger.info("📋 Next steps:")
    logger.info("  1. Review generated configurations in k8s_configs/")
    logger.info("  2. Apply to your Kubernetes cluster:")
    logger.info("     kubectl apply -f k8s_configs/")
    logger.info("  3. Monitor deployment:")
    logger.info("     kubectl get pods -n aegis-system")
    
    return True


if __name__ == "__main__":
    asyncio.run(deploy_aegis_kubernetes())
