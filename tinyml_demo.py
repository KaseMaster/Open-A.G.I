#!/usr/bin/env python3
"""
🎯 AEGIS TinyML Demo - Sprint 4.3
Demostración rápida de TinyML y Edge AI
"""

import torch
from tinyml_edge_ai import AEGISTinyML, TinyModelConfig, EdgePlatform

async def quick_tinyml_demo():
    """Demostración rápida de TinyML"""

    print("🎯 AEGIS TinyML & Edge AI Quick Demo")
    print("=" * 38)

    tinyml = AEGISTinyML()

    # Crear modelo tiny
    model = tinyml.create_tiny_model("mobilenet", TinyModelConfig(num_classes=10, width_multiplier=0.5))

    print(f"✅ Modelo tiny creado: {tinyml._get_model_params(model)} parámetros")

    # Benchmarking simulado
    benchmarks = tinyml.benchmark_platforms(model)

    print("\\n📊 Benchmarks por plataforma:")
    for platform, benchmark in benchmarks.items():
        print(".1f"
    print("\\n🎉 TinyML funcionando!")

if __name__ == "__main__":
    asyncio.run(quick_tinyml_demo())
