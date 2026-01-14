#!/usr/bin/env python3
"""
🎯 AEGIS Monitoring Demo - Sprint 5.1
Demostración rápida del sistema de monitoring enterprise
"""

import asyncio
import time
from enterprise_monitoring import AEGISMonitoringSystem

async def quick_monitoring_demo():
    """Demostración rápida de monitoring"""

    print("🎯 AEGIS Enterprise Monitoring Quick Demo")
    print("=" * 40)

    monitoring = AEGISMonitoringSystem()

    print("🚀 Iniciando monitoring...")
    await monitoring.start_monitoring()

    # Simular algunas requests
    for i in range(5):
        monitoring.record_api_request(
            f"/api/test/{i}",
            "GET",
            0.3 + (i * 0.1),
            200 if i < 4 else 500
        )
        time.sleep(0.1)

    print("\\n📊 Estado del sistema:")
    dashboard_data = monitoring.get_dashboard_data()
    print(f"   • CPU: {dashboard_data['system_metrics']['cpu'].get('mean', 0):.1f}%")
    print(f"   • Memoria: {dashboard_data['system_metrics']['memory'].get('mean', 0):.1f}%")
    print(f"   • Alertas activas: {dashboard_data['active_alerts']}")

    # Health check
    health = await monitoring.health_checker.check_all_components()
    healthy_count = sum(1 for h in health.values() if h.status == 'healthy')
    print(f"   • Componentes saludables: {healthy_count}/{len(health)}")

    # Performance report
    report = monitoring.performance_analyzer.generate_performance_report(1)
    print(".3f"    print(f"   • Error rate: {report.error_rate:.1f}%")

    monitoring.stop_monitoring()

    print("\\n🎉 Monitoring funcionando!")

if __name__ == "__main__":
    asyncio.run(quick_monitoring_demo())
