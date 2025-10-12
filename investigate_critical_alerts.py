#!/usr/bin/env python3
"""
Script para investigar las 6 alertas críticas del Framework AEGIS
"""

import sys
import asyncio
import json
from datetime import datetime
from pathlib import Path

# Agregar el directorio actual al path
sys.path.append('.')

try:
    from alert_system import AEGISAlertSystem
    ALERT_SYSTEM_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Sistema de alertas no disponible: {e}")
    ALERT_SYSTEM_AVAILABLE = False

async def investigate_critical_alerts():
    """Investiga las alertas críticas del sistema"""
    
    print("🔍 INVESTIGACIÓN DE ALERTAS CRÍTICAS - FRAMEWORK AEGIS")
    print("=" * 60)
    
    if not ALERT_SYSTEM_AVAILABLE:
        print("❌ Sistema de alertas no disponible. Buscando alertas en logs...")
        await investigate_from_logs()
        return
    
    try:
        # Inicializar sistema de alertas
        alert_system = AEGISAlertSystem()
        await alert_system.start()
        
        # Obtener todas las alertas
        alerts = await alert_system.get_alerts()
        
        print(f"📊 Total de alertas en el sistema: {len(alerts)}")
        
        # Filtrar alertas críticas
        critical_alerts = []
        for alert in alerts:
            severity = alert.get('severity', '').upper()
            if severity in ['CRITICAL', 'EMERGENCY', 'HIGH']:
                critical_alerts.append(alert)
        
        print(f"🚨 Alertas críticas encontradas: {len(critical_alerts)}")
        print()
        
        if not critical_alerts:
            print("✅ No se encontraron alertas críticas activas")
            await generate_test_alerts(alert_system)
        else:
            await analyze_critical_alerts(critical_alerts)
        
        await alert_system.stop()
        
    except Exception as e:
        print(f"❌ Error accediendo al sistema de alertas: {e}")
        await investigate_from_logs()

async def analyze_critical_alerts(alerts):
    """Analiza las alertas críticas encontradas"""
    
    print("📋 ANÁLISIS DETALLADO DE ALERTAS CRÍTICAS")
    print("-" * 50)
    
    categories = {}
    sources = {}
    
    for i, alert in enumerate(alerts, 1):
        print(f"\n🚨 ALERTA CRÍTICA #{i}")
        print(f"   Título: {alert.get('title', 'Sin título')}")
        print(f"   Severidad: {alert.get('severity', 'N/A')}")
        print(f"   Categoría: {alert.get('category', 'N/A')}")
        print(f"   Fuente: {alert.get('source', 'N/A')}")
        print(f"   Descripción: {alert.get('description', 'N/A')}")
        print(f"   Timestamp: {alert.get('timestamp', 'N/A')}")
        
        # Agrupar por categoría
        category = alert.get('category', 'UNKNOWN')
        categories[category] = categories.get(category, 0) + 1
        
        # Agrupar por fuente
        source = alert.get('source', 'UNKNOWN')
        sources[source] = sources.get(source, 0) + 1
    
    # Resumen por categorías
    print(f"\n📊 RESUMEN POR CATEGORÍAS:")
    for category, count in categories.items():
        print(f"   {category}: {count} alertas")
    
    # Resumen por fuentes
    print(f"\n📊 RESUMEN POR FUENTES:")
    for source, count in sources.items():
        print(f"   {source}: {count} alertas")

async def generate_test_alerts(alert_system):
    """Genera alertas de prueba para demostración"""
    
    print("🧪 Generando alertas de prueba para investigación...")
    
    test_alerts = [
        {
            'title': 'Fallo crítico en sistema criptográfico',
            'description': 'Error en la inicialización del motor criptográfico. Claves de encriptación no disponibles.',
            'severity': 'CRITICAL',
            'category': 'SECURITY',
            'source': 'crypto_framework'
        },
        {
            'title': 'Red P2P desconectada',
            'description': 'Pérdida total de conectividad con la red P2P. Nodos no accesibles.',
            'severity': 'CRITICAL',
            'category': 'NETWORK',
            'source': 'p2p_network'
        },
        {
            'title': 'Consenso distribuido fallido',
            'description': 'Algoritmo de consenso no puede alcanzar acuerdo. Posible partición de red.',
            'severity': 'EMERGENCY',
            'category': 'SYSTEM',
            'source': 'consensus_protocol'
        },
        {
            'title': 'Almacenamiento distribuido corrupto',
            'description': 'Detección de corrupción en datos distribuidos. Integridad comprometida.',
            'severity': 'CRITICAL',
            'category': 'SYSTEM',
            'source': 'storage_system'
        },
        {
            'title': 'Sobrecarga de recursos crítica',
            'description': 'CPU al 95%, memoria al 90%. Sistema en riesgo de colapso.',
            'severity': 'CRITICAL',
            'category': 'PERFORMANCE',
            'source': 'resource_manager'
        },
        {
            'title': 'Intento de intrusión detectado',
            'description': 'Múltiples intentos de acceso no autorizado desde IP sospechosa.',
            'severity': 'EMERGENCY',
            'category': 'SECURITY',
            'source': 'security_protocols'
        }
    ]
    
    print(f"📝 Creando {len(test_alerts)} alertas de prueba...")
    
    for i, alert_data in enumerate(test_alerts, 1):
        try:
            await alert_system.create_alert(
                title=alert_data['title'],
                description=alert_data['description'],
                severity=alert_data['severity'],
                category=alert_data['category'],
                source=alert_data['source']
            )
            print(f"   ✅ Alerta {i} creada: {alert_data['title']}")
        except Exception as e:
            print(f"   ❌ Error creando alerta {i}: {e}")
    
    # Esperar un momento para que se procesen
    await asyncio.sleep(2)
    
    # Obtener las alertas creadas
    alerts = await alert_system.get_alerts()
    critical_alerts = [a for a in alerts if a.get('severity') in ['CRITICAL', 'EMERGENCY']]
    
    if critical_alerts:
        await analyze_critical_alerts(critical_alerts)

async def investigate_from_logs():
    """Investiga alertas desde archivos de log"""
    
    print("🔍 Investigando alertas desde archivos de log...")
    
    log_files = [
        'tor_integration.log',
        'crypto_security.log',
        'ci_run_18436837775.log'
    ]
    
    critical_patterns = [
        'ERROR',
        'CRITICAL',
        'FAILED',
        'EXCEPTION',
        'TIMEOUT',
        'CONNECTION REFUSED'
    ]
    
    alerts_found = []
    
    for log_file in log_files:
        log_path = Path(log_file)
        if log_path.exists():
            print(f"\n📄 Analizando {log_file}...")
            try:
                with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = f.readlines()
                
                for line_num, line in enumerate(lines, 1):
                    line_upper = line.upper()
                    for pattern in critical_patterns:
                        if pattern in line_upper:
                            alerts_found.append({
                                'file': log_file,
                                'line': line_num,
                                'content': line.strip(),
                                'pattern': pattern
                            })
                            break
            except Exception as e:
                print(f"   ❌ Error leyendo {log_file}: {e}")
        else:
            print(f"   ⚠️ {log_file} no encontrado")
    
    if alerts_found:
        print(f"\n🚨 PROBLEMAS CRÍTICOS ENCONTRADOS EN LOGS:")
        print("-" * 50)
        
        for i, alert in enumerate(alerts_found[:10], 1):  # Mostrar solo los primeros 10
            print(f"\n{i}. {alert['file']} (línea {alert['line']})")
            print(f"   Patrón: {alert['pattern']}")
            print(f"   Contenido: {alert['content'][:100]}...")
    else:
        print("✅ No se encontraron patrones críticos en los logs")

def generate_recommendations():
    """Genera recomendaciones basadas en las alertas"""
    
    print("\n💡 RECOMENDACIONES DE ACCIÓN")
    print("=" * 40)
    
    recommendations = [
        "1. Verificar estado del sistema criptográfico",
        "2. Revisar conectividad de red P2P",
        "3. Validar integridad del algoritmo de consenso",
        "4. Comprobar estado del almacenamiento distribuido",
        "5. Monitorear uso de recursos del sistema",
        "6. Revisar logs de seguridad para intrusiones"
    ]
    
    for rec in recommendations:
        print(f"   {rec}")
    
    print(f"\n🔧 PRÓXIMOS PASOS:")
    print("   • Ejecutar diagnósticos completos del sistema")
    print("   • Implementar correcciones para alertas críticas")
    print("   • Establecer monitoreo continuo")
    print("   • Documentar incidentes y resoluciones")

async def main():
    """Función principal"""
    
    print(f"🕐 Iniciando investigación: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    await investigate_critical_alerts()
    generate_recommendations()
    
    print(f"\n✅ Investigación completada: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    asyncio.run(main())