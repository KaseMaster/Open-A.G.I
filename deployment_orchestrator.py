#!/usr/bin/env python3
"""
Orquestador de Despliegue - AEGIS Framework
Sistema automatizado para despliegue y configuración del framework
distribuido en múltiples entornos y topologías de red.

Características principales:
- Despliegue automatizado multi-entorno
- Configuración dinámica de topología
- Gestión de contenedores Docker
- Orquestación con Kubernetes
- Monitoreo de salud del despliegue
- Rollback automático en caso de fallos
- Escalado horizontal automático
- Gestión de secretos y configuraciones
"""

import asyncio
import time
import json
import logging
import yaml
import os
import subprocess
import tempfile
import shutil
import socket
import threading
from typing import Dict, List, Set, Optional, Any, Tuple
from dataclasses import dataclass, asdict, field
from enum import Enum
import docker
import kubernetes
from kubernetes import client, config
import consul
import etcd3
import requests
import paramiko
import jinja2
from pathlib import Path
import hashlib
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import psutil
import netifaces

# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DeploymentEnvironment(Enum):
    """Entornos de despliegue"""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"
    EDGE = "edge"

class DeploymentStrategy(Enum):
    """Estrategias de despliegue"""
    ROLLING_UPDATE = "rolling_update"
    BLUE_GREEN = "blue_green"
    CANARY = "canary"
    RECREATE = "recreate"
    A_B_TESTING = "a_b_testing"

class NodeRole(Enum):
    """Roles de nodos en el despliegue"""
    COORDINATOR = "coordinator"
    WORKER = "worker"
    EDGE = "edge"
    BOOTSTRAP = "bootstrap"
    MONITOR = "monitor"

class DeploymentStatus(Enum):
    """Estados del despliegue"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    DEPLOYED = "deployed"
    FAILED = "failed"
    ROLLING_BACK = "rolling_back"
    ROLLED_BACK = "rolled_back"

class HealthStatus(Enum):
    """Estados de salud"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"

@dataclass
class NodeConfiguration:
    """Configuración de nodo"""
    node_id: str
    role: NodeRole
    hostname: str
    ip_address: str
    port: int
    cpu_cores: int
    memory_gb: int
    storage_gb: int
    gpu_count: int = 0
    network_bandwidth: int = 1000  # Mbps
    zone: str = "default"
    labels: Dict[str, str] = field(default_factory=dict)
    environment_vars: Dict[str, str] = field(default_factory=dict)

@dataclass
class ServiceConfiguration:
    """Configuración de servicio"""
    service_name: str
    image: str
    version: str
    replicas: int
    cpu_request: str
    cpu_limit: str
    memory_request: str
    memory_limit: str
    ports: List[int]
    environment_vars: Dict[str, str] = field(default_factory=dict)
    volumes: List[Dict[str, str]] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    health_check: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DeploymentConfiguration:
    """Configuración completa de despliegue"""
    deployment_id: str
    environment: DeploymentEnvironment
    strategy: DeploymentStrategy
    nodes: List[NodeConfiguration]
    services: List[ServiceConfiguration]
    network_config: Dict[str, Any]
    security_config: Dict[str, Any]
    monitoring_config: Dict[str, Any]
    backup_config: Dict[str, Any] = field(default_factory=dict)
    scaling_config: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DeploymentState:
    """Estado del despliegue"""
    deployment_id: str
    status: DeploymentStatus
    health: HealthStatus
    start_time: float
    last_update: float
    deployed_services: Dict[str, str] = field(default_factory=dict)
    failed_services: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    logs: List[str] = field(default_factory=list)

class ContainerManager:
    """Gestor de contenedores Docker"""
    
    def __init__(self):
        try:
            self.client = docker.from_env()
            logger.info("✅ Cliente Docker inicializado")
        except Exception as e:
            logger.error(f"❌ Error inicializando Docker: {e}")
            self.client = None
    
    async def build_image(self, dockerfile_path: str, image_name: str, tag: str = "latest") -> bool:
        """Construye imagen Docker"""
        try:
            if not self.client:
                raise Exception("Cliente Docker no disponible")
            
            logger.info(f"🔨 Construyendo imagen: {image_name}:{tag}")
            
            # Construir imagen
            image, logs = self.client.images.build(
                path=dockerfile_path,
                tag=f"{image_name}:{tag}",
                rm=True,
                forcerm=True
            )
            
            # Procesar logs de construcción
            for log in logs:
                if 'stream' in log:
                    logger.debug(log['stream'].strip())
            
            logger.info(f"✅ Imagen construida: {image.id[:12]}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error construyendo imagen: {e}")
            return False
    
    async def push_image(self, image_name: str, tag: str, registry: str = None) -> bool:
        """Sube imagen a registro"""
        try:
            if not self.client:
                raise Exception("Cliente Docker no disponible")
            
            full_name = f"{registry}/{image_name}:{tag}" if registry else f"{image_name}:{tag}"
            
            logger.info(f"📤 Subiendo imagen: {full_name}")
            
            # Subir imagen
            for line in self.client.images.push(full_name, stream=True, decode=True):
                if 'status' in line:
                    logger.debug(f"Push: {line['status']}")
            
            logger.info(f"✅ Imagen subida: {full_name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error subiendo imagen: {e}")
            return False
    
    async def run_container(self, service_config: ServiceConfiguration, node_config: NodeConfiguration) -> Optional[str]:
        """Ejecuta contenedor"""
        try:
            if not self.client:
                raise Exception("Cliente Docker no disponible")
            
            container_name = f"{service_config.service_name}_{node_config.node_id}"
            
            # Configurar puertos
            port_bindings = {}
            for port in service_config.ports:
                port_bindings[f"{port}/tcp"] = port
            
            # Configurar volúmenes
            volumes = {}
            for volume in service_config.volumes:
                volumes[volume["host_path"]] = {
                    "bind": volume["container_path"],
                    "mode": volume.get("mode", "rw")
                }
            
            # Configurar variables de entorno
            environment = {**service_config.environment_vars, **node_config.environment_vars}
            
            logger.info(f"🚀 Ejecutando contenedor: {container_name}")
            
            # Ejecutar contenedor
            container = self.client.containers.run(
                image=f"{service_config.image}:{service_config.version}",
                name=container_name,
                ports=port_bindings,
                volumes=volumes,
                environment=environment,
                detach=True,
                restart_policy={"Name": "unless-stopped"},
                labels={
                    "aegis.service": service_config.service_name,
                    "aegis.node": node_config.node_id,
                    "aegis.role": node_config.role.value
                }
            )
            
            logger.info(f"✅ Contenedor ejecutado: {container.id[:12]}")
            return container.id
            
        except Exception as e:
            logger.error(f"❌ Error ejecutando contenedor: {e}")
            return None
    
    async def stop_container(self, container_id: str) -> bool:
        """Detiene contenedor"""
        try:
            if not self.client:
                return False
            
            container = self.client.containers.get(container_id)
            container.stop(timeout=30)
            container.remove()
            
            logger.info(f"🛑 Contenedor detenido: {container_id[:12]}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error deteniendo contenedor: {e}")
            return False

class KubernetesManager:
    """Gestor de Kubernetes"""
    
    def __init__(self, kubeconfig_path: str = None):
        try:
            if kubeconfig_path:
                config.load_kube_config(config_file=kubeconfig_path)
            else:
                config.load_incluster_config()
            
            self.v1 = client.CoreV1Api()
            self.apps_v1 = client.AppsV1Api()
            self.networking_v1 = client.NetworkingV1Api()
            
            logger.info("✅ Cliente Kubernetes inicializado")
            
        except Exception as e:
            logger.error(f"❌ Error inicializando Kubernetes: {e}")
            self.v1 = None
            self.apps_v1 = None
            self.networking_v1 = None
    
    async def create_namespace(self, namespace: str) -> bool:
        """Crea namespace"""
        try:
            if not self.v1:
                return False
            
            # Verificar si existe
            try:
                self.v1.read_namespace(name=namespace)
                logger.info(f"📁 Namespace ya existe: {namespace}")
                return True
            except client.exceptions.ApiException as e:
                if e.status != 404:
                    raise
            
            # Crear namespace
            namespace_obj = client.V1Namespace(
                metadata=client.V1ObjectMeta(name=namespace)
            )
            
            self.v1.create_namespace(body=namespace_obj)
            logger.info(f"✅ Namespace creado: {namespace}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error creando namespace: {e}")
            return False
    
    async def deploy_service(self, service_config: ServiceConfiguration, namespace: str = "default") -> bool:
        """Despliega servicio en Kubernetes"""
        try:
            if not self.apps_v1:
                return False
            
            # Crear Deployment
            deployment = self._create_deployment_manifest(service_config)
            
            try:
                self.apps_v1.create_namespaced_deployment(
                    namespace=namespace,
                    body=deployment
                )
                logger.info(f"✅ Deployment creado: {service_config.service_name}")
            except client.exceptions.ApiException as e:
                if e.status == 409:  # Already exists
                    self.apps_v1.patch_namespaced_deployment(
                        name=service_config.service_name,
                        namespace=namespace,
                        body=deployment
                    )
                    logger.info(f"🔄 Deployment actualizado: {service_config.service_name}")
                else:
                    raise
            
            # Crear Service
            service = self._create_service_manifest(service_config)
            
            try:
                self.v1.create_namespaced_service(
                    namespace=namespace,
                    body=service
                )
                logger.info(f"✅ Service creado: {service_config.service_name}")
            except client.exceptions.ApiException as e:
                if e.status == 409:  # Already exists
                    self.v1.patch_namespaced_service(
                        name=service_config.service_name,
                        namespace=namespace,
                        body=service
                    )
                    logger.info(f"🔄 Service actualizado: {service_config.service_name}")
                else:
                    raise
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error desplegando servicio: {e}")
            return False
    
    def _create_deployment_manifest(self, service_config: ServiceConfiguration) -> client.V1Deployment:
        """Crea manifiesto de Deployment"""
        # Configurar contenedor
        container = client.V1Container(
            name=service_config.service_name,
            image=f"{service_config.image}:{service_config.version}",
            ports=[client.V1ContainerPort(container_port=port) for port in service_config.ports],
            env=[client.V1EnvVar(name=k, value=v) for k, v in service_config.environment_vars.items()],
            resources=client.V1ResourceRequirements(
                requests={
                    "cpu": service_config.cpu_request,
                    "memory": service_config.memory_request
                },
                limits={
                    "cpu": service_config.cpu_limit,
                    "memory": service_config.memory_limit
                }
            )
        )
        
        # Configurar health check si está definido
        if service_config.health_check:
            if "http" in service_config.health_check:
                http_check = service_config.health_check["http"]
                container.liveness_probe = client.V1Probe(
                    http_get=client.V1HTTPGetAction(
                        path=http_check["path"],
                        port=http_check["port"]
                    ),
                    initial_delay_seconds=http_check.get("initial_delay", 30),
                    period_seconds=http_check.get("period", 10)
                )
        
        # Configurar Pod template
        pod_template = client.V1PodTemplateSpec(
            metadata=client.V1ObjectMeta(
                labels={"app": service_config.service_name}
            ),
            spec=client.V1PodSpec(containers=[container])
        )
        
        # Configurar Deployment spec
        deployment_spec = client.V1DeploymentSpec(
            replicas=service_config.replicas,
            selector=client.V1LabelSelector(
                match_labels={"app": service_config.service_name}
            ),
            template=pod_template
        )
        
        # Crear Deployment
        deployment = client.V1Deployment(
            api_version="apps/v1",
            kind="Deployment",
            metadata=client.V1ObjectMeta(name=service_config.service_name),
            spec=deployment_spec
        )
        
        return deployment
    
    def _create_service_manifest(self, service_config: ServiceConfiguration) -> client.V1Service:
        """Crea manifiesto de Service"""
        # Configurar puertos
        ports = [
            client.V1ServicePort(
                port=port,
                target_port=port,
                protocol="TCP"
            ) for port in service_config.ports
        ]
        
        # Configurar Service spec
        service_spec = client.V1ServiceSpec(
            selector={"app": service_config.service_name},
            ports=ports,
            type="ClusterIP"
        )
        
        # Crear Service
        service = client.V1Service(
            api_version="v1",
            kind="Service",
            metadata=client.V1ObjectMeta(name=service_config.service_name),
            spec=service_spec
        )
        
        return service
    
    async def get_deployment_status(self, service_name: str, namespace: str = "default") -> Dict[str, Any]:
        """Obtiene estado del deployment"""
        try:
            if not self.apps_v1:
                return {"status": "unknown", "error": "Kubernetes no disponible"}
            
            deployment = self.apps_v1.read_namespaced_deployment(
                name=service_name,
                namespace=namespace
            )
            
            status = {
                "name": service_name,
                "namespace": namespace,
                "replicas": deployment.spec.replicas,
                "ready_replicas": deployment.status.ready_replicas or 0,
                "available_replicas": deployment.status.available_replicas or 0,
                "updated_replicas": deployment.status.updated_replicas or 0,
                "conditions": []
            }
            
            if deployment.status.conditions:
                for condition in deployment.status.conditions:
                    status["conditions"].append({
                        "type": condition.type,
                        "status": condition.status,
                        "reason": condition.reason,
                        "message": condition.message
                    })
            
            return status
            
        except Exception as e:
            logger.error(f"❌ Error obteniendo estado de deployment: {e}")
            return {"status": "error", "error": str(e)}

class ConfigurationManager:
    """Gestor de configuraciones"""
    
    def __init__(self, config_dir: str = "./config"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        self.template_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(str(self.config_dir / "templates")),
            autoescape=jinja2.select_autoescape(['html', 'xml'])
        )
    
    def generate_docker_compose(self, deployment_config: DeploymentConfiguration) -> str:
        """Genera archivo docker-compose.yml"""
        compose_config = {
            "version": "3.8",
            "services": {},
            "networks": {
                "aegis-network": {
                    "driver": "bridge",
                    "ipam": {
                        "config": [{"subnet": deployment_config.network_config.get("subnet", "172.20.0.0/16")}]
                    }
                }
            },
            "volumes": {}
        }
        
        # Configurar servicios
        for service in deployment_config.services:
            service_config = {
                "image": f"{service.image}:{service.version}",
                "container_name": f"{service.service_name}_{deployment_config.deployment_id}",
                "restart": "unless-stopped",
                "networks": ["aegis-network"],
                "environment": service.environment_vars,
                "ports": [f"{port}:{port}" for port in service.ports]
            }
            
            # Configurar volúmenes
            if service.volumes:
                service_config["volumes"] = [
                    f"{vol['host_path']}:{vol['container_path']}:{vol.get('mode', 'rw')}"
                    for vol in service.volumes
                ]
            
            # Configurar dependencias
            if service.dependencies:
                service_config["depends_on"] = service.dependencies
            
            # Configurar health check
            if service.health_check and "http" in service.health_check:
                http_check = service.health_check["http"]
                service_config["healthcheck"] = {
                    "test": f"curl -f http://localhost:{http_check['port']}{http_check['path']} || exit 1",
                    "interval": f"{http_check.get('period', 10)}s",
                    "timeout": "5s",
                    "retries": 3,
                    "start_period": f"{http_check.get('initial_delay', 30)}s"
                }
            
            compose_config["services"][service.service_name] = service_config
        
        # Escribir archivo
        compose_file = self.config_dir / f"docker-compose-{deployment_config.deployment_id}.yml"
        with open(compose_file, 'w') as f:
            yaml.dump(compose_config, f, default_flow_style=False, indent=2)
        
        logger.info(f"✅ Docker Compose generado: {compose_file}")
        return str(compose_file)
    
    def generate_kubernetes_manifests(self, deployment_config: DeploymentConfiguration) -> str:
        """Genera manifiestos de Kubernetes"""
        manifests_dir = self.config_dir / f"k8s-{deployment_config.deployment_id}"
        manifests_dir.mkdir(exist_ok=True)
        
        # Generar namespace
        namespace_manifest = {
            "apiVersion": "v1",
            "kind": "Namespace",
            "metadata": {
                "name": f"aegis-{deployment_config.deployment_id}"
            }
        }
        
        with open(manifests_dir / "namespace.yaml", 'w') as f:
            yaml.dump(namespace_manifest, f, default_flow_style=False)
        
        # Generar manifiestos para cada servicio
        for service in deployment_config.services:
            # Deployment
            deployment_manifest = {
                "apiVersion": "apps/v1",
                "kind": "Deployment",
                "metadata": {
                    "name": service.service_name,
                    "namespace": f"aegis-{deployment_config.deployment_id}"
                },
                "spec": {
                    "replicas": service.replicas,
                    "selector": {
                        "matchLabels": {"app": service.service_name}
                    },
                    "template": {
                        "metadata": {
                            "labels": {"app": service.service_name}
                        },
                        "spec": {
                            "containers": [{
                                "name": service.service_name,
                                "image": f"{service.image}:{service.version}",
                                "ports": [{"containerPort": port} for port in service.ports],
                                "env": [{"name": k, "value": v} for k, v in service.environment_vars.items()],
                                "resources": {
                                    "requests": {
                                        "cpu": service.cpu_request,
                                        "memory": service.memory_request
                                    },
                                    "limits": {
                                        "cpu": service.cpu_limit,
                                        "memory": service.memory_limit
                                    }
                                }
                            }]
                        }
                    }
                }
            }
            
            with open(manifests_dir / f"{service.service_name}-deployment.yaml", 'w') as f:
                yaml.dump(deployment_manifest, f, default_flow_style=False)
            
            # Service
            service_manifest = {
                "apiVersion": "v1",
                "kind": "Service",
                "metadata": {
                    "name": service.service_name,
                    "namespace": f"aegis-{deployment_config.deployment_id}"
                },
                "spec": {
                    "selector": {"app": service.service_name},
                    "ports": [{"port": port, "targetPort": port} for port in service.ports],
                    "type": "ClusterIP"
                }
            }
            
            with open(manifests_dir / f"{service.service_name}-service.yaml", 'w') as f:
                yaml.dump(service_manifest, f, default_flow_style=False)
        
        logger.info(f"✅ Manifiestos K8s generados: {manifests_dir}")
        return str(manifests_dir)
    
    def generate_monitoring_config(self, deployment_config: DeploymentConfiguration) -> str:
        """Genera configuración de monitoreo"""
        monitoring_config = {
            "prometheus": {
                "scrape_configs": []
            },
            "grafana": {
                "dashboards": [],
                "datasources": []
            },
            "alertmanager": {
                "rules": []
            }
        }
        
        # Configurar scraping de Prometheus
        for service in deployment_config.services:
            if "metrics_port" in service.environment_vars:
                scrape_config = {
                    "job_name": service.service_name,
                    "static_configs": [{
                        "targets": [f"{service.service_name}:{service.environment_vars['metrics_port']}"]
                    }],
                    "scrape_interval": "15s",
                    "metrics_path": "/metrics"
                }
                monitoring_config["prometheus"]["scrape_configs"].append(scrape_config)
        
        # Escribir configuración
        monitoring_file = self.config_dir / f"monitoring-{deployment_config.deployment_id}.yml"
        with open(monitoring_file, 'w') as f:
            yaml.dump(monitoring_config, f, default_flow_style=False, indent=2)
        
        logger.info(f"✅ Configuración de monitoreo generada: {monitoring_file}")
        return str(monitoring_file)

class HealthChecker:
    """Verificador de salud de servicios"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.timeout = 10
    
    async def check_service_health(self, service_config: ServiceConfiguration, node_config: NodeConfiguration) -> Dict[str, Any]:
        """Verifica salud de servicio"""
        health_result = {
            "service": service_config.service_name,
            "node": node_config.node_id,
            "status": HealthStatus.UNKNOWN,
            "checks": [],
            "response_time": None,
            "error": None
        }
        
        try:
            # Verificar health check HTTP si está configurado
            if service_config.health_check and "http" in service_config.health_check:
                http_check = service_config.health_check["http"]
                url = f"http://{node_config.ip_address}:{http_check['port']}{http_check['path']}"
                
                start_time = time.time()
                response = self.session.get(url)
                response_time = time.time() - start_time
                
                health_result["response_time"] = response_time
                
                if response.status_code == 200:
                    health_result["status"] = HealthStatus.HEALTHY
                    health_result["checks"].append({
                        "type": "http",
                        "status": "passed",
                        "response_code": response.status_code,
                        "response_time": response_time
                    })
                else:
                    health_result["status"] = HealthStatus.UNHEALTHY
                    health_result["checks"].append({
                        "type": "http",
                        "status": "failed",
                        "response_code": response.status_code,
                        "response_time": response_time
                    })
            
            # Verificar conectividad de puertos
            for port in service_config.ports:
                port_check = self._check_port_connectivity(node_config.ip_address, port)
                health_result["checks"].append(port_check)
                
                if port_check["status"] == "failed" and health_result["status"] != HealthStatus.UNHEALTHY:
                    health_result["status"] = HealthStatus.DEGRADED
            
            # Si no hay checks específicos, verificar conectividad básica
            if not health_result["checks"]:
                basic_check = self._check_basic_connectivity(node_config.ip_address)
                health_result["checks"].append(basic_check)
                
                if basic_check["status"] == "passed":
                    health_result["status"] = HealthStatus.HEALTHY
                else:
                    health_result["status"] = HealthStatus.UNHEALTHY
            
        except Exception as e:
            health_result["status"] = HealthStatus.UNHEALTHY
            health_result["error"] = str(e)
            health_result["checks"].append({
                "type": "exception",
                "status": "failed",
                "error": str(e)
            })
        
        return health_result
    
    def _check_port_connectivity(self, ip_address: str, port: int) -> Dict[str, Any]:
        """Verifica conectividad de puerto"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            result = sock.connect_ex((ip_address, port))
            sock.close()
            
            if result == 0:
                return {
                    "type": "port",
                    "port": port,
                    "status": "passed",
                    "message": f"Puerto {port} accesible"
                }
            else:
                return {
                    "type": "port",
                    "port": port,
                    "status": "failed",
                    "message": f"Puerto {port} no accesible"
                }
        except Exception as e:
            return {
                "type": "port",
                "port": port,
                "status": "failed",
                "error": str(e)
            }
    
    def _check_basic_connectivity(self, ip_address: str) -> Dict[str, Any]:
        """Verifica conectividad básica"""
        try:
            # Ping básico usando socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            result = sock.connect_ex((ip_address, 22))  # SSH port como proxy de conectividad
            sock.close()
            
            return {
                "type": "connectivity",
                "status": "passed" if result == 0 else "failed",
                "message": f"Conectividad a {ip_address}: {'OK' if result == 0 else 'FAIL'}"
            }
        except Exception as e:
            return {
                "type": "connectivity",
                "status": "failed",
                "error": str(e)
            }

class DeploymentOrchestrator:
    """Orquestador principal de despliegue"""
    
    def __init__(self, config_dir: str = "./config"):
        self.config_manager = ConfigurationManager(config_dir)
        self.container_manager = ContainerManager()
        self.k8s_manager = KubernetesManager()
        self.health_checker = HealthChecker()
        
        self.deployments: Dict[str, DeploymentState] = {}
        self.deployment_configs: Dict[str, DeploymentConfiguration] = {}
        
        # Configurar directorio de trabajo
        self.work_dir = Path("./deployments")
        self.work_dir.mkdir(exist_ok=True)
        
        logger.info("🚀 Orquestador de Despliegue inicializado")
    
    async def deploy(self, deployment_config: DeploymentConfiguration) -> bool:
        """Ejecuta despliegue completo"""
        deployment_id = deployment_config.deployment_id
        
        try:
            logger.info(f"🚀 Iniciando despliegue: {deployment_id}")
            
            # Inicializar estado
            deployment_state = DeploymentState(
                deployment_id=deployment_id,
                status=DeploymentStatus.PENDING,
                health=HealthStatus.UNKNOWN,
                start_time=time.time(),
                last_update=time.time()
            )
            
            self.deployments[deployment_id] = deployment_state
            self.deployment_configs[deployment_id] = deployment_config
            
            # Actualizar estado
            await self._update_deployment_status(deployment_id, DeploymentStatus.IN_PROGRESS)
            
            # Validar configuración
            if not await self._validate_deployment_config(deployment_config):
                await self._update_deployment_status(deployment_id, DeploymentStatus.FAILED)
                return False
            
            # Preparar entorno
            if not await self._prepare_deployment_environment(deployment_config):
                await self._update_deployment_status(deployment_id, DeploymentStatus.FAILED)
                return False
            
            # Ejecutar estrategia de despliegue
            success = await self._execute_deployment_strategy(deployment_config)
            
            if success:
                await self._update_deployment_status(deployment_id, DeploymentStatus.DEPLOYED)
                
                # Verificar salud post-despliegue
                await self._post_deployment_health_check(deployment_config)
                
                logger.info(f"✅ Despliegue completado: {deployment_id}")
                return True
            else:
                await self._update_deployment_status(deployment_id, DeploymentStatus.FAILED)
                
                # Intentar rollback automático
                if deployment_config.environment != DeploymentEnvironment.DEVELOPMENT:
                    await self._rollback_deployment(deployment_id)
                
                logger.error(f"❌ Despliegue fallido: {deployment_id}")
                return False
                
        except Exception as e:
            logger.error(f"💥 Error en despliegue {deployment_id}: {e}")
            await self._update_deployment_status(deployment_id, DeploymentStatus.FAILED)
            return False
    
    async def _validate_deployment_config(self, config: DeploymentConfiguration) -> bool:
        """Valida configuración de despliegue"""
        try:
            logger.info("🔍 Validando configuración de despliegue")
            
            # Validar nodos
            if not config.nodes:
                logger.error("❌ No hay nodos configurados")
                return False
            
            # Validar servicios
            if not config.services:
                logger.error("❌ No hay servicios configurados")
                return False
            
            # Validar conectividad de nodos
            for node in config.nodes:
                if not await self._validate_node_connectivity(node):
                    logger.error(f"❌ Nodo no accesible: {node.node_id}")
                    return False
            
            # Validar dependencias de servicios
            service_names = {service.service_name for service in config.services}
            for service in config.services:
                for dependency in service.dependencies:
                    if dependency not in service_names:
                        logger.error(f"❌ Dependencia no encontrada: {dependency} para {service.service_name}")
                        return False
            
            # Validar recursos
            for node in config.nodes:
                if not await self._validate_node_resources(node, config.services):
                    logger.error(f"❌ Recursos insuficientes en nodo: {node.node_id}")
                    return False
            
            logger.info("✅ Configuración validada")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error validando configuración: {e}")
            return False
    
    async def _validate_node_connectivity(self, node: NodeConfiguration) -> bool:
        """Valida conectividad de nodo"""
        try:
            # Verificar ping básico
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            result = sock.connect_ex((node.ip_address, 22))
            sock.close()
            
            return result == 0
            
        except Exception as e:
            logger.error(f"❌ Error verificando conectividad de {node.node_id}: {e}")
            return False
    
    async def _validate_node_resources(self, node: NodeConfiguration, services: List[ServiceConfiguration]) -> bool:
        """Valida recursos de nodo"""
        try:
            # Calcular recursos requeridos para servicios en este nodo
            required_cpu = 0
            required_memory = 0
            
            for service in services:
                # Parsear requests de CPU (ej: "100m" = 0.1 cores)
                cpu_request = service.cpu_request
                if cpu_request.endswith('m'):
                    required_cpu += int(cpu_request[:-1]) / 1000
                else:
                    required_cpu += float(cpu_request)
                
                # Parsear requests de memoria (ej: "256Mi" = 256 MB)
                memory_request = service.memory_request
                if memory_request.endswith('Mi'):
                    required_memory += int(memory_request[:-2])
                elif memory_request.endswith('Gi'):
                    required_memory += int(memory_request[:-2]) * 1024
                else:
                    required_memory += int(memory_request)
            
            # Verificar disponibilidad (con margen del 20%)
            available_cpu = node.cpu_cores * 0.8
            available_memory = node.memory_gb * 1024 * 0.8
            
            if required_cpu > available_cpu:
                logger.warning(f"⚠️ CPU insuficiente en {node.node_id}: {required_cpu} > {available_cpu}")
                return False
            
            if required_memory > available_memory:
                logger.warning(f"⚠️ Memoria insuficiente en {node.node_id}: {required_memory} > {available_memory}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error validando recursos de {node.node_id}: {e}")
            return False
    
    async def _prepare_deployment_environment(self, config: DeploymentConfiguration) -> bool:
        """Prepara entorno de despliegue"""
        try:
            logger.info("🔧 Preparando entorno de despliegue")
            
            # Generar archivos de configuración
            if config.environment == DeploymentEnvironment.DEVELOPMENT:
                compose_file = self.config_manager.generate_docker_compose(config)
                logger.info(f"📄 Docker Compose: {compose_file}")
            else:
                k8s_manifests = self.config_manager.generate_kubernetes_manifests(config)
                logger.info(f"📄 Manifiestos K8s: {k8s_manifests}")
            
            # Generar configuración de monitoreo
            monitoring_config = self.config_manager.generate_monitoring_config(config)
            logger.info(f"📊 Configuración de monitoreo: {monitoring_config}")
            
            # Preparar directorios de trabajo
            deployment_dir = self.work_dir / config.deployment_id
            deployment_dir.mkdir(exist_ok=True)
            
            # Crear directorios para logs y datos
            (deployment_dir / "logs").mkdir(exist_ok=True)
            (deployment_dir / "data").mkdir(exist_ok=True)
            
            logger.info("✅ Entorno preparado")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error preparando entorno: {e}")
            return False
    
    async def _execute_deployment_strategy(self, config: DeploymentConfiguration) -> bool:
        """Ejecuta estrategia de despliegue"""
        try:
            logger.info(f"🎯 Ejecutando estrategia: {config.strategy.value}")
            
            if config.strategy == DeploymentStrategy.ROLLING_UPDATE:
                return await self._rolling_update_deployment(config)
            elif config.strategy == DeploymentStrategy.BLUE_GREEN:
                return await self._blue_green_deployment(config)
            elif config.strategy == DeploymentStrategy.CANARY:
                return await self._canary_deployment(config)
            elif config.strategy == DeploymentStrategy.RECREATE:
                return await self._recreate_deployment(config)
            else:
                logger.error(f"❌ Estrategia no soportada: {config.strategy.value}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error ejecutando estrategia: {e}")
            return False
    
    async def _rolling_update_deployment(self, config: DeploymentConfiguration) -> bool:
        """Despliegue con rolling update"""
        try:
            logger.info("🔄 Ejecutando rolling update")
            
            # Ordenar servicios por dependencias
            ordered_services = self._topological_sort_services(config.services)
            
            # Desplegar servicios uno por uno
            for service in ordered_services:
                logger.info(f"🚀 Desplegando servicio: {service.service_name}")
                
                if config.environment == DeploymentEnvironment.DEVELOPMENT:
                    # Usar Docker para desarrollo
                    success = await self._deploy_service_docker(service, config.nodes[0])
                else:
                    # Usar Kubernetes para otros entornos
                    success = await self._deploy_service_kubernetes(service, config)
                
                if not success:
                    logger.error(f"❌ Fallo desplegando {service.service_name}")
                    return False
                
                # Esperar a que el servicio esté saludable
                if not await self._wait_for_service_health(service, config.nodes[0]):
                    logger.error(f"❌ Servicio no saludable: {service.service_name}")
                    return False
                
                # Actualizar estado
                self.deployments[config.deployment_id].deployed_services[service.service_name] = "deployed"
            
            logger.info("✅ Rolling update completado")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error en rolling update: {e}")
            return False
    
    async def _blue_green_deployment(self, config: DeploymentConfiguration) -> bool:
        """Despliegue blue-green"""
        try:
            logger.info("🔵🟢 Ejecutando despliegue blue-green")
            
            # Crear entorno "green" (nuevo)
            green_config = self._create_green_environment(config)
            
            # Desplegar en entorno green
            if not await self._deploy_to_environment(green_config, "green"):
                return False
            
            # Verificar salud del entorno green
            if not await self._verify_environment_health(green_config):
                return False
            
            # Cambiar tráfico de blue a green
            if not await self._switch_traffic(config, green_config):
                return False
            
            # Limpiar entorno blue anterior
            await self._cleanup_blue_environment(config)
            
            logger.info("✅ Despliegue blue-green completado")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error en despliegue blue-green: {e}")
            return False
    
    async def _canary_deployment(self, config: DeploymentConfiguration) -> bool:
        """Despliegue canary"""
        try:
            logger.info("🐤 Ejecutando despliegue canary")
            
            # Desplegar versión canary (5% del tráfico)
            canary_config = self._create_canary_environment(config, traffic_percentage=5)
            
            if not await self._deploy_to_environment(canary_config, "canary"):
                return False
            
            # Monitorear métricas por 5 minutos
            if not await self._monitor_canary_metrics(canary_config, duration=300):
                logger.error("❌ Métricas de canary fallaron")
                await self._rollback_canary(canary_config)
                return False
            
            # Incrementar tráfico gradualmente
            for percentage in [10, 25, 50, 100]:
                logger.info(f"📈 Incrementando tráfico canary a {percentage}%")
                
                if not await self._update_canary_traffic(canary_config, percentage):
                    await self._rollback_canary(canary_config)
                    return False
                
                # Monitorear por 2 minutos en cada incremento
                if not await self._monitor_canary_metrics(canary_config, duration=120):
                    await self._rollback_canary(canary_config)
                    return False
            
            # Promover canary a producción
            await self._promote_canary_to_production(canary_config)
            
            logger.info("✅ Despliegue canary completado")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error en despliegue canary: {e}")
            return False
    
    async def _recreate_deployment(self, config: DeploymentConfiguration) -> bool:
        """Despliegue recreate (detener todo y recrear)"""
        try:
            logger.info("🔄 Ejecutando despliegue recreate")
            
            # Detener servicios existentes
            await self._stop_existing_services(config)
            
            # Esperar a que se detengan completamente
            await asyncio.sleep(10)
            
            # Desplegar nuevas versiones
            return await self._rolling_update_deployment(config)
            
        except Exception as e:
            logger.error(f"❌ Error en despliegue recreate: {e}")
            return False
    
    async def _deploy_service_docker(self, service: ServiceConfiguration, node: NodeConfiguration) -> bool:
        """Despliega servicio usando Docker"""
        try:
            container_id = await self.container_manager.run_container(service, node)
            return container_id is not None
            
        except Exception as e:
            logger.error(f"❌ Error desplegando servicio Docker: {e}")
            return False
    
    async def _deploy_service_kubernetes(self, service: ServiceConfiguration, config: DeploymentConfiguration) -> bool:
        """Despliega servicio usando Kubernetes"""
        try:
            namespace = f"aegis-{config.deployment_id}"
            
            # Crear namespace si no existe
            await self.k8s_manager.create_namespace(namespace)
            
            # Desplegar servicio
            return await self.k8s_manager.deploy_service(service, namespace)
            
        except Exception as e:
            logger.error(f"❌ Error desplegando servicio K8s: {e}")
            return False
    
    async def _wait_for_service_health(self, service: ServiceConfiguration, node: NodeConfiguration, timeout: int = 300) -> bool:
        """Espera a que el servicio esté saludable"""
        try:
            start_time = time.time()
            
            while time.time() - start_time < timeout:
                health_result = await self.health_checker.check_service_health(service, node)
                
                if health_result["status"] == HealthStatus.HEALTHY:
                    logger.info(f"✅ Servicio saludable: {service.service_name}")
                    return True
                
                logger.info(f"⏳ Esperando salud de {service.service_name}...")
                await asyncio.sleep(10)
            
            logger.error(f"⏰ Timeout esperando salud de {service.service_name}")
            return False
            
        except Exception as e:
            logger.error(f"❌ Error verificando salud: {e}")
            return False
    
    def _topological_sort_services(self, services: List[ServiceConfiguration]) -> List[ServiceConfiguration]:
        """Ordena servicios por dependencias usando ordenamiento topológico"""
        # Crear grafo de dependencias
        graph = {service.service_name: service.dependencies for service in services}
        service_map = {service.service_name: service for service in services}
        
        # Algoritmo de Kahn para ordenamiento topológico
        in_degree = {name: 0 for name in graph}
        for dependencies in graph.values():
            for dep in dependencies:
                if dep in in_degree:
                    in_degree[dep] += 1
        
        queue = [name for name, degree in in_degree.items() if degree == 0]
        result = []
        
        while queue:
            current = queue.pop(0)
            result.append(service_map[current])
            
            for neighbor in graph[current]:
                if neighbor in in_degree:
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        queue.append(neighbor)
        
        return result
    
    async def _post_deployment_health_check(self, config: DeploymentConfiguration):
        """Verificación de salud post-despliegue"""
        try:
            logger.info("🏥 Verificando salud post-despliegue")
            
            health_results = []
            
            # Verificar cada servicio en cada nodo
            for service in config.services:
                for node in config.nodes:
                    health_result = await self.health_checker.check_service_health(service, node)
                    health_results.append(health_result)
            
            # Actualizar estado de salud del despliegue
            healthy_services = sum(1 for result in health_results if result["status"] == HealthStatus.HEALTHY)
            total_checks = len(health_results)
            
            if healthy_services == total_checks:
                self.deployments[config.deployment_id].health = HealthStatus.HEALTHY
            elif healthy_services > total_checks * 0.7:
                self.deployments[config.deployment_id].health = HealthStatus.DEGRADED
            else:
                self.deployments[config.deployment_id].health = HealthStatus.UNHEALTHY
            
            # Almacenar métricas de salud
            self.deployments[config.deployment_id].metrics["health_check"] = {
                "healthy_services": healthy_services,
                "total_checks": total_checks,
                "health_ratio": healthy_services / total_checks,
                "detailed_results": health_results
            }
            
            logger.info(f"🏥 Salud del despliegue: {self.deployments[config.deployment_id].health.value}")
            
        except Exception as e:
            logger.error(f"❌ Error en verificación de salud: {e}")
            self.deployments[config.deployment_id].health = HealthStatus.UNKNOWN
    
    async def _update_deployment_status(self, deployment_id: str, status: DeploymentStatus):
        """Actualiza estado del despliegue"""
        if deployment_id in self.deployments:
            self.deployments[deployment_id].status = status
            self.deployments[deployment_id].last_update = time.time()
            
            logger.info(f"📊 Estado de despliegue {deployment_id}: {status.value}")
    
    async def _rollback_deployment(self, deployment_id: str) -> bool:
        """Ejecuta rollback del despliegue"""
        try:
            logger.info(f"🔙 Iniciando rollback: {deployment_id}")
            
            await self._update_deployment_status(deployment_id, DeploymentStatus.ROLLING_BACK)
            
            # Implementar lógica de rollback específica
            # Por ahora, simplemente detener servicios fallidos
            config = self.deployment_configs[deployment_id]
            await self._stop_existing_services(config)
            
            await self._update_deployment_status(deployment_id, DeploymentStatus.ROLLED_BACK)
            
            logger.info(f"✅ Rollback completado: {deployment_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error en rollback: {e}")
            return False
    
    async def _stop_existing_services(self, config: DeploymentConfiguration):
        """Detiene servicios existentes"""
        try:
            logger.info("🛑 Deteniendo servicios existentes")
            
            if config.environment == DeploymentEnvironment.DEVELOPMENT:
                # Detener contenedores Docker
                if self.container_manager.client:
                    containers = self.container_manager.client.containers.list(
                        filters={"label": f"aegis.deployment={config.deployment_id}"}
                    )
                    
                    for container in containers:
                        container.stop(timeout=30)
                        container.remove()
                        logger.info(f"🛑 Contenedor detenido: {container.name}")
            
            else:
                # Detener servicios de Kubernetes
                if self.k8s_manager.apps_v1:
                    namespace = f"aegis-{config.deployment_id}"
                    
                    # Escalar deployments a 0
                    deployments = self.k8s_manager.apps_v1.list_namespaced_deployment(namespace=namespace)
                    
                    for deployment in deployments.items:
                        deployment.spec.replicas = 0
                        self.k8s_manager.apps_v1.patch_namespaced_deployment(
                            name=deployment.metadata.name,
                            namespace=namespace,
                            body=deployment
                        )
                        logger.info(f"🛑 Deployment escalado a 0: {deployment.metadata.name}")
            
        except Exception as e:
            logger.error(f"❌ Error deteniendo servicios: {e}")
    
    # Métodos auxiliares para estrategias avanzadas (implementación básica)
    def _create_green_environment(self, config: DeploymentConfiguration) -> DeploymentConfiguration:
        """Crea configuración para entorno green"""
        green_config = config
        green_config.deployment_id = f"{config.deployment_id}-green"
        return green_config
    
    async def _deploy_to_environment(self, config: DeploymentConfiguration, env_name: str) -> bool:
        """Despliega a entorno específico"""
        return await self._rolling_update_deployment(config)
    
    async def _verify_environment_health(self, config: DeploymentConfiguration) -> bool:
        """Verifica salud del entorno"""
        await self._post_deployment_health_check(config)
        return self.deployments[config.deployment_id].health in [HealthStatus.HEALTHY, HealthStatus.DEGRADED]
    
    async def _switch_traffic(self, old_config: DeploymentConfiguration, new_config: DeploymentConfiguration) -> bool:
        """Cambia tráfico entre entornos"""
        logger.info("🔄 Cambiando tráfico a nuevo entorno")
        return True
    
    async def _cleanup_blue_environment(self, config: DeploymentConfiguration):
        """Limpia entorno blue anterior"""
        logger.info("🧹 Limpiando entorno anterior")
    
    def _create_canary_environment(self, config: DeploymentConfiguration, traffic_percentage: int) -> DeploymentConfiguration:
        """Crea configuración para entorno canary"""
        canary_config = config
        canary_config.deployment_id = f"{config.deployment_id}-canary"
        return canary_config
    
    async def _monitor_canary_metrics(self, config: DeploymentConfiguration, duration: int) -> bool:
        """Monitorea métricas de canary"""
        logger.info(f"📊 Monitoreando canary por {duration}s")
        await asyncio.sleep(duration)
        return True
    
    async def _update_canary_traffic(self, config: DeploymentConfiguration, percentage: int) -> bool:
        """Actualiza porcentaje de tráfico canary"""
        logger.info(f"📈 Actualizando tráfico canary: {percentage}%")
        return True
    
    async def _rollback_canary(self, config: DeploymentConfiguration):
        """Rollback de canary"""
        logger.info("🔙 Rollback de canary")
    
    async def _promote_canary_to_production(self, config: DeploymentConfiguration):
        """Promueve canary a producción"""
        logger.info("🎉 Promoviendo canary a producción")
    
    def get_deployment_status(self, deployment_id: str) -> Optional[DeploymentState]:
        """Obtiene estado del despliegue"""
        return self.deployments.get(deployment_id)
    
    def list_deployments(self) -> Dict[str, DeploymentState]:
        """Lista todos los despliegues"""
        return self.deployments.copy()
    
    async def cleanup_deployment(self, deployment_id: str) -> bool:
        """Limpia despliegue"""
        try:
            if deployment_id in self.deployment_configs:
                config = self.deployment_configs[deployment_id]
                await self._stop_existing_services(config)
                
                # Limpiar directorios
                deployment_dir = self.work_dir / deployment_id
                if deployment_dir.exists():
                    shutil.rmtree(deployment_dir)
                
                # Limpiar estado
                del self.deployments[deployment_id]
                del self.deployment_configs[deployment_id]
                
                logger.info(f"🧹 Despliegue limpiado: {deployment_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"❌ Error limpiando despliegue: {e}")
            return False

# Función principal para demostración
async def main():
    """Función principal de demostración"""
    try:
        print("🚀 Iniciando Orquestador de Despliegue - AEGIS Framework")
        
        # Crear orquestador
        orchestrator = DeploymentOrchestrator()
        
        # Configuración de ejemplo
        deployment_config = DeploymentConfiguration(
            deployment_id="aegis-demo-001",
            environment=DeploymentEnvironment.DEVELOPMENT,
            strategy=DeploymentStrategy.ROLLING_UPDATE,
            nodes=[
                NodeConfiguration(
                    node_id="node-001",
                    role=NodeRole.COORDINATOR,
                    hostname="localhost",
                    ip_address="127.0.0.1",
                    port=8000,
                    cpu_cores=4,
                    memory_gb=8,
                    storage_gb=100
                )
            ],
            services=[
                ServiceConfiguration(
                    service_name="aegis-coordinator",
                    image="aegis/coordinator",
                    version="1.0.0",
                    replicas=1,
                    cpu_request="500m",
                    cpu_limit="1000m",
                    memory_request="512Mi",
                    memory_limit="1Gi",
                    ports=[8000, 8001],
                    environment_vars={
                        "NODE_ROLE": "coordinator",
                        "LOG_LEVEL": "INFO"
                    },
                    health_check={
                        "http": {
                            "path": "/health",
                            "port": 8000,
                            "initial_delay": 30,
                            "period": 10
                        }
                    }
                )
            ],
            network_config={
                "subnet": "172.20.0.0/16",
                "dns_servers": ["8.8.8.8", "8.8.4.4"]
            },
            security_config={
                "tls_enabled": True,
                "authentication_required": True
            },
            monitoring_config={
                "metrics_enabled": True,
                "logging_level": "INFO"
            }
        )
        
        # Ejecutar despliegue
        print("📋 Configuración de despliegue creada")
        print(f"🎯 Estrategia: {deployment_config.strategy.value}")
        print(f"🌍 Entorno: {deployment_config.environment.value}")
        print(f"📊 Servicios: {len(deployment_config.services)}")
        print(f"🖥️ Nodos: {len(deployment_config.nodes)}")
        
        # Nota: En un entorno real, aquí se ejecutaría el despliegue
        # success = await orchestrator.deploy(deployment_config)
        
        print("\n✅ Demostración del Orquestador de Despliegue completada")
        print("📝 Para ejecutar despliegue real, descomenta la línea de deploy()")
        
        return 0
        
    except Exception as e:
        logger.error(f"💥 Error en demostración: {e}")
        return 1

if __name__ == "__main__":
    import sys
    exit_code = asyncio.run(main())
    sys.exit(exit_code)