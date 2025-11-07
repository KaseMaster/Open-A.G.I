#!/usr/bin/env python3
"""
🎯 AEGIS Advanced Analytics Demo - Sprint 4.2
Demostración rápida del sistema de forecasting y analytics avanzados
"""

import asyncio
import pandas as pd
import numpy as np
from advanced_analytics_forecasting import AEGISAdvancedAnalytics, ForecastingConfig, ForecastingModel

async def quick_analytics_demo():
    """Demostración rápida de analytics avanzados"""

    print("🎯 AEGIS Advanced Analytics Quick Demo")
    print("=" * 35)

    analytics = AEGISAdvancedAnalytics()

    # Crear datos de ejemplo simples
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    # Serie con tendencia creciente y estacionalidad semanal
    t = np.arange(100)
    values = 100 + 0.5 * t + 10 * np.sin(2 * np.pi * t / 7) + np.random.normal(0, 2, 100)

    time_series = pd.Series(values, index=dates, name='demo_metric')

    print(f"✅ Serie temporal creada: {len(time_series)} puntos")

    # Configuración simple
    config = ForecastingConfig(
        horizon=10,
        models_to_try=[ForecastingModel.ARIMA, ForecastingModel.PROPHET]
    )

    # Ejecutar analytics
    print("\\n🚀 Ejecutando analytics...")
    results = await analytics.analyze_and_forecast(time_series, config, "quick_demo")

    # Mostrar resultados
    analysis = results['analysis']
    forecasts = results['forecasts']

    print("\\n📊 ANÁLISIS:")
    print(f"   • Estacionaria: {'✅ Sí' if analysis.is_stationary else '❌ No'}")
    if analysis.seasonality_type:
        print(f"   • Estacionalidad: {analysis.seasonality_type.value}")

    print("\\n🔮 FORECASTS:")
    for forecast in forecasts:
        mae = forecast.metrics.get('mae', 'N/A')
        print(f"   📊 {forecast.model_name.value}: MAE={mae:.3f}")

    # Insights
    insights = await analytics.generate_insights(analysis, forecasts)
    print("\\n💡 INSIGHTS:")
    for insight in insights[:2]:  # Primeros 2
        print(f"   • {insight}")

    print("\\n🎉 Analytics funcionando!")

if __name__ == "__main__":
    asyncio.run(quick_analytics_demo())
