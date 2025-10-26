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
    print("\\n🤖 Customer Service Chatbot...")
    customer_result = await demos.run_customer_service_demo()
    print(f"   • Respuesta: {customer_result.results.get('generated_response', '')[:60]}...")
    print(".3f"
    # Content moderation demo
    print("\\n🛡️ Content Moderation...")
    moderation_result = await demos.run_content_moderation_demo()
    print(f"   • Decisión: {moderation_result.results.get('moderation_decision', 'N/A')}")
    print(".3f"
    # Medical assistant demo
    print("\\n🏥 Medical Diagnosis Assistant...")
    medical_result = await demos.run_medical_assistant_demo()
    print(f"   • Riesgo: {medical_result.metrics.get('risk_level', 'N/A')}")
    print(".3f"
    # Statistics
    stats = demos.get_demo_statistics()
    print("\\n📊 Estadísticas:")
    print(f"   • Total ejecuciones: {stats['total_demo_runs']}")
    print(".1f"    print(".1f"    print(".1f"
    print("\\n🎉 Production use cases funcionando!")

if __name__ == "__main__":
    asyncio.run(quick_production_demo())
