#!/usr/bin/env python3
"""
🎯 AEGIS Federated Analytics Demo - Sprint 4.2
Demostración rápida del sistema de analytics federados
"""

import asyncio
from federated_analytics_privacy import AEGISFederatedAnalytics, QueryType, PrivacyLevel

async def quick_federated_demo():
    """Demostración rápida de federated analytics"""

    print("🎯 AEGIS Federated Analytics Quick Demo")
    print("=" * 42)

    analytics = AEGISFederatedAnalytics(num_participants=3)

    print("✅ Sistema inicializado con 3 participantes")

    # Ejecutar query federada
    result = await analytics.execute_federated_query(
        QueryType.COUNT, {}, PrivacyLevel.BASIC
    )

    print("\\n🔍 Query ejecutada:")
    print(".3f"    print(f"   • Participantes: {result.participant_count}")
    print(f"   • Muestras: {result.total_samples}")
    print(f"   • Privacidad: {result.privacy_guarantees['privacy_level']}")

    # Ejecutar queries en paralelo
    queries = [
        (QueryType.MEAN, {"column": "value"}, PrivacyLevel.BASIC),
        (QueryType.SUM, {"column": "score"}, PrivacyLevel.ENHANCED)
    ]

    results = await analytics.execute_multiple_queries(queries)

    print("\\n⚡ Queries paralelas:")
    for i, res in enumerate(results):
        query_name = queries[i][0].value
        print(".3f"
    # Reporte de privacidad
    privacy = analytics.get_privacy_report()
    dp = privacy['differential_privacy']
    print(".2f"
    print("\\n🎉 Federated Analytics funcionando!")

if __name__ == "__main__":
    asyncio.run(quick_federated_demo())
