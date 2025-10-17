#!/usr/bin/env python3
"""
🎯 AEGIS Anomaly Detection Demo - Sprint 4.2
Demostración rápida del sistema de detección automática de anomalías
"""

import asyncio
import numpy as np
import pandas as pd
from automatic_anomaly_detection import AEGISAnomalyDetection, AnomalyConfig, AnomalyDetectionMethod

async def quick_anomaly_demo():
    """Demostración rápida de detección de anomalías"""

    print("🎯 AEGIS Anomaly Detection Quick Demo")
    print("=" * 40)

    detector = AEGISAnomalyDetection()

    # Crear datos con anomalías
    np.random.seed(42)
    normal_data = np.random.normal(0, 1, (900, 3))
    anomaly_data = np.random.normal(0, 1, (100, 3)) * 3  # Anomalías más dispersas
    data = np.vstack([normal_data, anomaly_data])

    print(f"✅ Datos creados: {len(data)} muestras, {data.shape[1]} features")
    print(f"   • Anomalías esperadas: {len(anomaly_data)} (~{len(anomaly_data)/len(data)*100:.1f}%)")

    # Configuración
    config = AnomalyConfig(
        methods=[AnomalyDetectionMethod.STATISTICAL, AnomalyDetectionMethod.ISOLATION_FOREST],
        contamination=0.1
    )

    # Detectar anomalías
    print("\\n🚀 Detectando anomalías...")
    results = await detector.detect_anomalies(data, config)

    # Mostrar resultados
    print("\\n📋 RESULTADOS:")
    for result in results:
        anomalies = np.sum(result.anomaly_labels == -1)
        percentage = anomalies / len(result.anomaly_labels) * 100
        print(".2f"
    # Insights
    insights = detector.get_anomaly_insights(results)
    print("\\n💡 Insight:", insights[0] if insights else "Detección completada")

    print("\\n🎉 Anomaly Detection funcionando!")

if __name__ == "__main__":
    asyncio.run(quick_anomaly_demo())
