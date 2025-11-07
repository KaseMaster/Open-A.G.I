#!/usr/bin/env python3
"""
🎯 AEGIS Multimodal Fusion Demo - Sprint 4.3
Demostración rápida del sistema de fusión multimodal
"""

import asyncio
import numpy as np
from multimodal_fusion import AEGISMultimodalFusion, MultimodalInput, MultimodalTask

async def quick_multimodal_demo():
    """Demostración rápida de multimodal fusion"""

    print("🎯 AEGIS Multimodal Fusion Quick Demo")
    print("=" * 37)

    fusion = AEGISMultimodalFusion()

    # Entrada multimodal
    multimodal_input = MultimodalInput(
        text="This product is amazing and wonderful!",
        image=np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8),
        audio=np.random.randn(500)
    )

    print("✅ Entrada multimodal creada")

    # Sentiment analysis
    result = await fusion.process_multimodal_input(
        multimodal_input, MultimodalTask.MULTIMODAL_SENTIMENT
    )

    print("\n😊 Multimodal Sentiment:")
    print(f"   • Sentimiento: {result.prediction}")
    print(f"   • Confianza: {result.confidence:.3f}")
    print(f"   • Tiempo: {result.processing_time:.3f}s")

    # Feature extraction
    features = fusion.extract_multimodal_features(multimodal_input)
    print("\n📊 Features:")
    print(f"   • Texto: {len(features.text_features) if features.text_features is not None else 0} dims")
    print(f"   • Imagen: {len(features.image_features) if features.image_features is not None else 0} dims")
    print(f"   • Audio: {len(features.audio_features) if features.audio_features is not None else 0} dims")
    print(f"   • Fusionadas: {len(features.fused_features) if features.fused_features is not None else 0} dims")

    print("\n🎉 Multimodal Fusion funcionando!")

if __name__ == "__main__":
    asyncio.run(quick_multimodal_demo())
