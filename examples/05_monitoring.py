#!/usr/bin/env python3
"""
AEGIS Framework - System Monitoring Example
Ejemplo de recolección de métricas del sistema
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

def main():
    print("📊 AEGIS System Monitoring Example")
    print("="*60)
    print()
    
    import psutil
    import time
    from datetime import datetime
    
    # 1. Métricas de CPU
    print("1️⃣  Métricas de CPU")
    print()
    cpu_percent = psutil.cpu_percent(interval=1)
    cpu_count = psutil.cpu_count()
    cpu_freq = psutil.cpu_freq()
    
    print(f"   CPU Usage: {cpu_percent}%")
    print(f"   CPU Cores: {cpu_count}")
    if cpu_freq:
        print(f"   CPU Freq: {cpu_freq.current:.0f} MHz")
    print()
    
    # 2. Métricas de Memoria
    print("2️⃣  Métricas de Memoria")
    print()
    memory = psutil.virtual_memory()
    
    print(f"   Total: {memory.total / (1024**3):.2f} GB")
    print(f"   Usado: {memory.used / (1024**3):.2f} GB ({memory.percent}%)")
    print(f"   Disponible: {memory.available / (1024**3):.2f} GB")
    print()
    
    # 3. Métricas de Disco
    print("3️⃣  Métricas de Disco")
    print()
    disk = psutil.disk_usage('/')
    
    print(f"   Total: {disk.total / (1024**3):.2f} GB")
    print(f"   Usado: {disk.used / (1024**3):.2f} GB ({disk.percent}%)")
    print(f"   Libre: {disk.free / (1024**3):.2f} GB")
    print()
    
    # 4. Métricas de Red
    print("4️⃣  Métricas de Red")
    print()
    net_io = psutil.net_io_counters()
    
    print(f"   Bytes enviados: {net_io.bytes_sent / (1024**2):.2f} MB")
    print(f"   Bytes recibidos: {net_io.bytes_recv / (1024**2):.2f} MB")
    print(f"   Paquetes enviados: {net_io.packets_sent:,}")
    print(f"   Paquetes recibidos: {net_io.packets_recv:,}")
    print()
    
    # 5. Procesos activos
    print("5️⃣  Procesos del Sistema")
    print()
    process_count = len(psutil.pids())
    print(f"   Procesos activos: {process_count}")
    print()
    
    # 6. Monitoreo en tiempo real (5 segundos)
    print("6️⃣  Monitoreo en Tiempo Real (5 seg)")
    print()
    print("   Timestamp         CPU%    RAM%    Disco%")
    print("   " + "-"*50)
    
    for i in range(5):
        timestamp = datetime.now().strftime("%H:%M:%S")
        cpu = psutil.cpu_percent(interval=1)
        ram = psutil.virtual_memory().percent
        disk = psutil.disk_usage('/').percent
        
        print(f"   {timestamp}       {cpu:5.1f}   {ram:5.1f}   {disk:5.1f}")
    
    print()
    
    # 7. Alertas simuladas
    print("7️⃣  Sistema de Alertas")
    print()
    
    # Umbrales
    CPU_THRESHOLD = 80.0
    RAM_THRESHOLD = 90.0
    DISK_THRESHOLD = 85.0
    
    alerts = []
    
    if cpu_percent > CPU_THRESHOLD:
        alerts.append(f"⚠️  CPU alta: {cpu_percent}% (umbral: {CPU_THRESHOLD}%)")
    
    if memory.percent > RAM_THRESHOLD:
        alerts.append(f"⚠️  RAM alta: {memory.percent}% (umbral: {RAM_THRESHOLD}%)")
    
    if disk.percent > DISK_THRESHOLD:
        alerts.append(f"⚠️  Disco lleno: {disk.percent}% (umbral: {DISK_THRESHOLD}%)")
    
    if alerts:
        for alert in alerts:
            print(f"   {alert}")
    else:
        print("   ✅ Todos los recursos dentro de límites normales")
    
    print()
    
    # 8. Integración con AEGIS
    print("8️⃣  Integración con AEGIS Framework")
    print()
    print("   El sistema de monitoreo AEGIS incluye:")
    print("   ✓ Metrics Collector - Recolección continua")
    print("   ✓ Dashboard Web - Visualización en tiempo real")
    print("   ✓ Alert System - Notificaciones automáticas")
    print("   ✓ Tracing Distribuido - OpenTelemetry")
    print()
    print("   Ver implementación en:")
    print("   - src/aegis/monitoring/metrics_collector.py")
    print("   - src/aegis/monitoring/monitoring_dashboard.py")
    print("   - src/aegis/monitoring/alert_system.py")
    print()
    
    print("="*60)
    print("✅ Monitoreo del sistema completado")
    print()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
