#!/usr/bin/env python3
"""
🎯 AEGIS Production Cases Demo - Sprint 5.1
Demostración rápida de casos de uso de producción
"""

import asyncio
import numpy as np
from production_use_cases import ProductionDemosManager

async def quick_production_demo():
    """Demostración rápida de casos de uso"""

    print("🎯 AEGIS Production Use Cases Quick Demo")
    print("=" * 40)

    demos = ProductionDemosManager()
    await demos.initialize_all_demos()

    # Customer service demo
    print("\n🤖 Customer Service Chatbot...")
    customer_result = await demos.run_customer_service_demo()
    print(f"   • Respuesta: {customer_result.results.get('generated_response', '')[:60]}...")
    print(f"   • Confianza: {customer_result.metrics.get('confidence', 0):.3f}")

    # Content moderation demo
    print("\n🛡️ Content Moderation...")
    moderation_result = await demos.run_content_moderation_demo()
    print(f"   • Decisión: {moderation_result.results.get('moderation_decision', 'N/A')}")
    print(f"   • Score: {moderation_result.metrics.get('moderation_score', 0):.3f}")

    # Medical assistant demo
    print("\n🏥 Medical Diagnosis Assistant...")
    medical_result = await demos.run_medical_assistant_demo()
    print(f"   • Riesgo: {medical_result.metrics.get('risk_level', 'N/A')}")
    print(f"   • Precisión: {medical_result.metrics.get('diagnosis_accuracy', 0):.3f}")

    # Statistics
    stats = demos.get_demo_statistics()
    print("\n📊 Estadísticas:")
    print(f"   • Total ejecuciones: {stats['total_demo_runs']}")
    print(f"   • Ejecuciones exitosas: {stats['successful_runs']:.1f}%")
    print(f"   • Tiempo promedio: {stats['avg_execution_time']:.1f}s")
    print(f"   • Recursos usados: {stats['avg_resource_usage']:.1f}%")

    print("\n🎉 Production use cases funcionando!")

if __name__ == "__main__":
    asyncio.run(quick_production_demo())
