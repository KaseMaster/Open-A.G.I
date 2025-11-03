#!/usr/bin/env python3
"""
🎯 AEGIS XAI Demo - Sprint 4.2
Demostración rápida del sistema de Explainable AI
"""

import asyncio
import numpy as np
from explainable_ai_shap import AEGISExplainableAI

async def quick_xai_demo():
    """Demostración rápida de XAI"""

    print("🎯 AEGIS Explainable AI Quick Demo")
    print("=" * 35)

    xai = AEGISExplainableAI()

    # Crear datos simples
    np.random.seed(42)
    X = np.random.randn(100, 3)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    feature_names = ['feature_A', 'feature_B', 'feature_C']

    # Modelo simple
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)

    print("✅ Modelo entrenado")

    # Explicación local
    sample = X[0:1]
    explanation = await xai.explain_model_prediction(
        model, sample, X[:20], "tree", feature_names
    )

    print("\n🔍 Explicación local:")
    print(f"   • Método: {explanation.method.value}")
    print(f"   • Tiempo: {explanation.processing_time:.3f}s")
    print(f"   • Texto: {explanation.explanation_text}")

    # Explicación global
    global_exp = await xai.explain_model_global(model, X, y, "tree", feature_names)

    print("\n🌍 Importancia global:")
    for i, feat in enumerate(global_exp[:3]):
        print(f"   • {feat.feature_name}: {feat.importance_score:.3f}")
    # Insights
    full_exp = xai.create_model_explanation(model, "Demo Model", X, y, "tree", feature_names)
    insights = full_exp.model_insights

    print("\n💡 Insights:")
    for insight in insights[:2]:
        print(f"   • {insight}")

    print("\n🎉 XAI funcionando!")

if __name__ == "__main__":
    asyncio.run(quick_xai_demo())