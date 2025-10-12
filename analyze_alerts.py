#!/usr/bin/env python3
"""
Script para analizar el sistema AEGIS y generar alertas basadas en los errores encontrados
"""

import sys
sys.path.append('.')
from alert_system import AEGISAlertSystem, AlertSeverity, AlertCategory
import asyncio
import time

async def analyze_system_and_create_alerts():
    print('🔍 Analizando sistema AEGIS y generando alertas...')
    
    # Inicializar sistema de alertas
    alert_config = {
        'storage': {'type': 'sqlite', 'path': 'alerts.db'},
        'channels': [],
        'rules': []
    }
    
    alert_system = AEGISAlertSystem(alert_config)
    await alert_system.start()
    
    # Alertas basadas en el análisis del código
    alerts_to_create = [
        {
            'title': 'Módulos de email no disponibles',
            'description': 'Los módulos email.mime.text y email.mime.multipart no están disponibles, las notificaciones por email están deshabilitadas',
            'severity': AlertSeverity.WARNING,
            'category': AlertCategory.SYSTEM,
            'metadata': {'component': 'alert_system', 'impact': 'email_notifications_disabled'}
        },
        {
            'title': 'Dependencias opcionales faltantes',
            'description': 'Múltiples módulos opcionales no están disponibles: aiohttp, websockets, zeroconf, netifaces, crypto_framework',
            'severity': AlertSeverity.WARNING,
            'category': AlertCategory.SYSTEM,
            'metadata': {'component': 'p2p_network', 'impact': 'reduced_functionality'}
        },
        {
            'title': 'Errores de conexión de red detectados',
            'description': 'Se detectaron múltiples errores de conexión y timeout en la red P2P',
            'severity': AlertSeverity.CRITICAL,
            'category': AlertCategory.NETWORK,
            'metadata': {'component': 'p2p_network', 'impact': 'network_connectivity_issues'}
        },
        {
            'title': 'Fallos en autenticación de peers',
            'description': 'Se detectaron fallos en la autenticación de peers y verificación de firmas',
            'severity': AlertSeverity.CRITICAL,
            'category': AlertCategory.SECURITY,
            'metadata': {'component': 'security_protocols', 'impact': 'authentication_failures'}
        },
        {
            'title': 'Errores en recolección de métricas',
            'description': 'Múltiples errores en la recolección y procesamiento de métricas del sistema',
            'severity': AlertSeverity.WARNING,
            'category': AlertCategory.PERFORMANCE,
            'metadata': {'component': 'metrics_collector', 'impact': 'monitoring_degraded'}
        }
    ]
    
    print(f'📝 Creando {len(alerts_to_create)} alertas basadas en análisis del sistema...')
    
    for alert_data in alerts_to_create:
        await alert_system.create_alert(
            title=alert_data['title'],
            description=alert_data['description'],
            severity=alert_data['severity'],
            category=alert_data['category'],
            source='system_analyzer',
            metadata=alert_data['metadata']
        )
        print(f'  ✅ Alerta creada: {alert_data["title"]}')
    
    # Esperar a que se procesen las alertas
    print('\n⏳ Esperando procesamiento de alertas...')
    await asyncio.sleep(2)  # Dar tiempo al procesamiento asíncrono
    
    # Obtener y mostrar todas las alertas
    alerts = alert_system.storage.get_alerts(limit=50)
    print(f'\n📊 Total de alertas en el sistema: {len(alerts)}')
    
    if alerts:
        print('\n🚨 ALERTAS ACTIVAS:')
        for alert in alerts:
            print(f'  - {alert.severity.value.upper()}: {alert.title}')
            print(f'    Categoría: {alert.category.value}')
            print(f'    Descripción: {alert.description}')
            if hasattr(alert, 'metadata') and alert.metadata:
                print(f'    Componente: {alert.metadata.get("component", "N/A")}')
            print()
    
    await alert_system.stop()

if __name__ == "__main__":
    asyncio.run(analyze_system_and_create_alerts())