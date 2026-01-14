#!/usr/bin/env python3
"""
🎯 AEGIS Computer Vision Demo - Sprint 4.3
Demostración rápida del sistema de computer vision
"""

import asyncio
import numpy as np
from advanced_computer_vision import AEGISAdvancedComputerVision, VisionConfig

async def quick_computer_vision_demo():
    """Demostración rápida de computer vision"""

    print("🎯 AEGIS Computer Vision Quick Demo")
    print("=" * 37)

    vision = AEGISAdvancedComputerVision()

    # Crear imagen sintética
    image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

    print("✅ Imagen sintética creada")

    # Procesar imagen
    print("\\n🚀 Procesando imagen...")
    results = await vision.process_image(image)

    print("\\n📊 RESULTADOS:")
    for task, result in results.items():
        if hasattr(result, 'processing_time'):
            print(".3f")
        else:
            print(f"   • {task}: procesado")

    print("\\n🎉 Computer Vision funcionando!")

if __name__ == "__main__":
    asyncio.run(quick_computer_vision_demo())
