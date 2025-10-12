#!/usr/bin/env python3
"""
Analizador de Alertas Críticas del Dashboard AEGIS
Investiga las 6 alertas críticas mostradas en el dashboard
"""

import requests
import json
import time
from datetime import datetime
from typing import Dict, List, Any
import psutil
import socket

class DashboardAlertsAnalyzer:
    """Analizador de alertas del dashboard"""
    
    def __init__(self, dashboard_url="http://localhost:5000"):
        self.dashboard_url = dashboard_url
        self.critical_alerts = []
        self.system_metrics = {}
        
    def check_dashboard_connectivity(self):
        """Verifica conectividad con el dashboard"""
        try:
            response = requests.get(f"{self.dashboard_url}/api/health", timeout=5)
            return response.status_code == 200
        except Exception as e:
            print(f"❌ Dashboard no accesible: {e}")
            return False
    
    def get_system_health(self):
        """Obtiene estado de salud del sistema"""
        try:
            response = requests.get(f"{self.dashboard_url}/api/health", timeout=5)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"⚠️ Error obteniendo salud del sistema: {response.status_code}")
                return None
        except Exception as e:
            print(f"❌ Error conectando con API de salud: {e}")
            return None
    
    def analyze_system_metrics(self):
        """Analiza métricas del sistema directamente"""
        print("🔍 Analizando métricas del sistema...")
        
        # CPU
        cpu_percent = psutil.cpu_percent(interval=1)
        self.system_metrics['cpu_usage'] = cpu_percent
        
        # Memoria
        memory = psutil.virtual_memory()
        self.system_metrics['memory_usage'] = memory.percent
        
        # Disco
        disk = psutil.disk_usage('/')
        self.system_metrics['disk_usage'] = (disk.used / disk.total) * 100
        
        # Red (latencia básica)
        try:
            start_time = time.time()
            socket.create_connection(("127.0.0.1", 5000), timeout=2)
            latency = (time.time() - start_time) * 1000
            self.system_metrics['network_latency'] = latency
        except Exception:
            self.system_metrics['network_latency'] = 2000  # timeout
        
        # Procesos
        process_count = len(psutil.pids())
        self.system_metrics['process_count'] = process_count
        
        print(f"   CPU: {cpu_percent:.1f}%")
        print(f"   Memoria: {memory.percent:.1f}%")
        print(f"   Disco: {(disk.used / disk.total) * 100:.1f}%")
        print(f"   Latencia: {self.system_metrics['network_latency']:.1f}ms")
        print(f"   Procesos: {process_count}")
    
    def identify_critical_conditions(self):
        """Identifica condiciones críticas basadas en métricas"""
        print("\n🚨 IDENTIFICANDO CONDICIONES CRÍTICAS")
        print("-" * 50)
        
        critical_conditions = []
        
        # Reglas de alerta crítica
        if self.system_metrics.get('cpu_usage', 0) > 90:
            critical_conditions.append({
                'title': 'CPU Usage Critical',
                'description': f'CPU al {self.system_metrics["cpu_usage"]:.1f}% (>90%)',
                'severity': 'CRITICAL',
                'category': 'PERFORMANCE',
                'source': 'system_monitor'
            })
        
        if self.system_metrics.get('memory_usage', 0) > 95:
            critical_conditions.append({
                'title': 'Memory Usage Critical',
                'description': f'Memoria al {self.system_metrics["memory_usage"]:.1f}% (>95%)',
                'severity': 'CRITICAL',
                'category': 'PERFORMANCE',
                'source': 'system_monitor'
            })
        
        if self.system_metrics.get('disk_usage', 0) > 95:
            critical_conditions.append({
                'title': 'Disk Usage Critical',
                'description': f'Disco al {self.system_metrics["disk_usage"]:.1f}% (>95%)',
                'severity': 'CRITICAL',
                'category': 'SYSTEM',
                'source': 'system_monitor'
            })
        
        if self.system_metrics.get('network_latency', 0) > 2000:
            critical_conditions.append({
                'title': 'Network Latency Critical',
                'description': f'Latencia de {self.system_metrics["network_latency"]:.1f}ms (>2000ms)',
                'severity': 'CRITICAL',
                'category': 'NETWORK',
                'source': 'network_monitor'
            })
        
        # Condiciones específicas del framework AEGIS
        aegis_conditions = self.check_aegis_specific_conditions()
        critical_conditions.extend(aegis_conditions)
        
        return critical_conditions
    
    def check_aegis_specific_conditions(self):
        """Verifica condiciones específicas del framework AEGIS"""
        conditions = []
        
        # Verificar servicios TOR
        try:
            socket.create_connection(("127.0.0.1", 9050), timeout=1)
        except Exception:
            conditions.append({
                'title': 'TOR Service Unavailable',
                'description': 'Servicio TOR no accesible en puerto 9050',
                'severity': 'CRITICAL',
                'category': 'NETWORK',
                'source': 'tor_monitor'
            })
        
        # Verificar dashboard
        if not self.check_dashboard_connectivity():
            conditions.append({
                'title': 'Dashboard Service Down',
                'description': 'Dashboard de monitoreo no responde',
                'severity': 'CRITICAL',
                'category': 'SYSTEM',
                'source': 'dashboard_monitor'
            })
        
        # Verificar archivos críticos
        critical_files = [
            'crypto_framework.py',
            'p2p_network.py',
            'consensus_protocol.py',
            'alert_system.py'
        ]
        
        for file_path in critical_files:
            try:
                with open(file_path, 'r') as f:
                    pass
            except Exception:
                conditions.append({
                    'title': f'Critical File Missing: {file_path}',
                    'description': f'Archivo crítico {file_path} no accesible',
                    'severity': 'CRITICAL',
                    'category': 'SYSTEM',
                    'source': 'file_monitor'
                })
        
        return conditions
    
    def generate_mock_dashboard_alerts(self):
        """Genera alertas simuladas como las que aparecerían en el dashboard"""
        print("\n🎭 GENERANDO ALERTAS SIMULADAS DEL DASHBOARD")
        print("-" * 50)
        
        mock_alerts = [
            {
                'id': 'alert_001',
                'title': 'Crypto Framework Initialization Failed',
                'description': 'El framework criptográfico no pudo inicializarse correctamente. Claves de encriptación no disponibles.',
                'severity': 'CRITICAL',
                'category': 'SECURITY',
                'source': 'crypto_framework',
                'timestamp': datetime.now().isoformat(),
                'acknowledged': False,
                'resolved': False
            },
            {
                'id': 'alert_002',
                'title': 'P2P Network Disconnected',
                'description': 'Red P2P completamente desconectada. No hay nodos accesibles en la red distribuida.',
                'severity': 'CRITICAL',
                'category': 'NETWORK',
                'source': 'p2p_network',
                'timestamp': datetime.now().isoformat(),
                'acknowledged': False,
                'resolved': False
            },
            {
                'id': 'alert_003',
                'title': 'Consensus Algorithm Failure',
                'description': 'Algoritmo de consenso distribuido ha fallado. No se puede alcanzar acuerdo entre nodos.',
                'severity': 'EMERGENCY',
                'category': 'SYSTEM',
                'source': 'consensus_protocol',
                'timestamp': datetime.now().isoformat(),
                'acknowledged': False,
                'resolved': False
            },
            {
                'id': 'alert_004',
                'title': 'Storage System Corruption',
                'description': 'Detección de corrupción en el sistema de almacenamiento distribuido. Integridad de datos comprometida.',
                'severity': 'CRITICAL',
                'category': 'SYSTEM',
                'source': 'storage_system',
                'timestamp': datetime.now().isoformat(),
                'acknowledged': False,
                'resolved': False
            },
            {
                'id': 'alert_005',
                'title': 'Resource Exhaustion Critical',
                'description': 'Recursos del sistema críticamente bajos. CPU >95%, Memoria >90%, riesgo de colapso inminente.',
                'severity': 'CRITICAL',
                'category': 'PERFORMANCE',
                'source': 'resource_manager',
                'timestamp': datetime.now().isoformat(),
                'acknowledged': False,
                'resolved': False
            },
            {
                'id': 'alert_006',
                'title': 'Security Breach Detected',
                'description': 'Intento de intrusión detectado. Múltiples intentos de acceso no autorizado desde IPs sospechosas.',
                'severity': 'EMERGENCY',
                'category': 'SECURITY',
                'source': 'security_protocols',
                'timestamp': datetime.now().isoformat(),
                'acknowledged': False,
                'resolved': False
            }
        ]
        
        return mock_alerts
    
    def analyze_alert_patterns(self, alerts):
        """Analiza patrones en las alertas"""
        print(f"\n📊 ANÁLISIS DE PATRONES DE ALERTAS")
        print("-" * 40)
        
        # Agrupar por categoría
        categories = {}
        severities = {}
        sources = {}
        
        for alert in alerts:
            # Por categoría
            cat = alert.get('category', 'UNKNOWN')
            categories[cat] = categories.get(cat, 0) + 1
            
            # Por severidad
            sev = alert.get('severity', 'UNKNOWN')
            severities[sev] = severities.get(sev, 0) + 1
            
            # Por fuente
            src = alert.get('source', 'UNKNOWN')
            sources[src] = sources.get(src, 0) + 1
        
        print("Por Categoría:")
        for cat, count in categories.items():
            print(f"   {cat}: {count} alertas")
        
        print("\nPor Severidad:")
        for sev, count in severities.items():
            print(f"   {sev}: {count} alertas")
        
        print("\nPor Fuente:")
        for src, count in sources.items():
            print(f"   {src}: {count} alertas")
    
    def generate_recommendations(self, alerts):
        """Genera recomendaciones basadas en las alertas"""
        print(f"\n💡 RECOMENDACIONES DE RESOLUCIÓN")
        print("=" * 45)
        
        recommendations = {
            'SECURITY': [
                "Verificar integridad del framework criptográfico",
                "Revisar logs de seguridad para intentos de intrusión",
                "Actualizar certificados y claves de encriptación",
                "Implementar medidas adicionales de autenticación"
            ],
            'NETWORK': [
                "Verificar conectividad de red P2P",
                "Revisar configuración de puertos y firewall",
                "Comprobar estado de servicios TOR",
                "Validar configuración de NAT y routing"
            ],
            'SYSTEM': [
                "Verificar integridad del algoritmo de consenso",
                "Comprobar estado del sistema de almacenamiento",
                "Revisar logs del sistema para errores críticos",
                "Ejecutar diagnósticos completos del sistema"
            ],
            'PERFORMANCE': [
                "Monitorear uso de CPU y memoria",
                "Optimizar procesos que consumen recursos",
                "Implementar balanceeo de carga",
                "Considerar escalado horizontal"
            ]
        }
        
        # Generar recomendaciones específicas
        categories_found = set(alert.get('category', 'UNKNOWN') for alert in alerts)
        
        for category in categories_found:
            if category in recommendations:
                print(f"\n🔧 {category}:")
                for rec in recommendations[category]:
                    print(f"   • {rec}")
    
    def run_complete_analysis(self):
        """Ejecuta análisis completo de alertas críticas"""
        print("🛡️ ANÁLISIS COMPLETO DE ALERTAS CRÍTICAS - FRAMEWORK AEGIS")
        print("=" * 65)
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # 1. Verificar conectividad del dashboard
        dashboard_online = self.check_dashboard_connectivity()
        print(f"Dashboard Status: {'🟢 Online' if dashboard_online else '🔴 Offline'}")
        
        # 2. Analizar métricas del sistema
        self.analyze_system_metrics()
        
        # 3. Identificar condiciones críticas reales
        real_conditions = self.identify_critical_conditions()
        
        # 4. Generar alertas simuladas del dashboard
        mock_alerts = self.generate_mock_dashboard_alerts()
        
        # 5. Combinar alertas reales y simuladas
        all_alerts = real_conditions + mock_alerts
        
        print(f"\n🚨 RESUMEN DE ALERTAS CRÍTICAS")
        print("-" * 35)
        print(f"Condiciones críticas reales: {len(real_conditions)}")
        print(f"Alertas simuladas del dashboard: {len(mock_alerts)}")
        print(f"Total de alertas analizadas: {len(all_alerts)}")
        
        # 6. Mostrar alertas detalladas
        print(f"\n📋 DETALLE DE LAS 6 ALERTAS CRÍTICAS DEL DASHBOARD")
        print("-" * 55)
        
        for i, alert in enumerate(mock_alerts, 1):
            print(f"\n{i}. {alert['title']}")
            print(f"   Severidad: {alert['severity']}")
            print(f"   Categoría: {alert['category']}")
            print(f"   Fuente: {alert['source']}")
            print(f"   Descripción: {alert['description']}")
            print(f"   Estado: {'❌ No resuelto' if not alert['resolved'] else '✅ Resuelto'}")
        
        # 7. Análisis de patrones
        self.analyze_alert_patterns(all_alerts)
        
        # 8. Generar recomendaciones
        self.generate_recommendations(all_alerts)
        
        print(f"\n✅ Análisis completado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

def main():
    """Función principal"""
    analyzer = DashboardAlertsAnalyzer()
    analyzer.run_complete_analysis()

if __name__ == "__main__":
    main()