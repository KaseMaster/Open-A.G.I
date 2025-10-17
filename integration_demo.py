#!/usr/bin/env python3
"""
🎯 AEGIS Integration Demo - Sprint 5.1
Demostración rápida del sistema de integración end-to-end
"""

import asyncio
import numpy as np
from integration_pipeline import AEGISIntegrationPipeline, PipelineInput, PipelineType

async def quick_integration_demo():
    """Demostración rápida de integration pipeline"""

    print("🎯 AEGIS Integration Pipeline Quick Demo")
    print("=" * 41)

    pipeline = AEGISIntegrationPipeline()
    await pipeline.initialize_pipeline()

    # Test multimodal
    multimodal_data = {
        'text': 'Beautiful sunset over mountains',
        'image': np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8),
        'audio': np.random.randn(500)
    }

    input_data = PipelineInput(
        data=multimodal_data,
        pipeline_type=PipelineType.MULTIMODAL_PIPELINE
    )

    print("🔄 Procesando pipeline multimodal...")
    result = await pipeline.process_pipeline(input_data)

    print("\\n📊 Resultado:")
    print(f"   • Tiempo: {result.processing_time:.3f}s")
    print(f"   • Etapas: {len(result.stages_completed)}")
    print(f"   • Resultados: {len(result.results)}")

    # Test generative
    gen_input = PipelineInput(
        data={'prompt': 'Write about AI future'},
        pipeline_type=PipelineType.GENERATIVE_PIPELINE
    )

    print("\\n🎨 Procesando pipeline generativo...")
    gen_result = await pipeline.process_pipeline(gen_input)

    print("\\n📝 Resultado generativo:")
    print(f"   • Tiempo: {gen_result.processing_time:.3f}s")
    print(f"   • Contenido generado: {len(str(gen_result.results))} chars")

    print("\\n🎉 Integration funcionando!")

if __name__ == "__main__":
    asyncio.run(quick_integration_demo())
