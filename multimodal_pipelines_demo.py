#!/usr/bin/env python3
"""
🎯 AEGIS Multimodal Pipelines Demo - Sprint 5.1
Demostración rápida de pipelines multimodales especializados
"""

import asyncio
import numpy as np
from multimodal_pipelines import MultimodalPipelineManager, MultimodalPipelineConfig, MultimodalPipelineInput, MultimodalPipelineType

async def quick_multimodal_pipelines_demo():
    """Demostración rápida de multimodal pipelines"""

    print("🎯 AEGIS Multimodal Pipelines Quick Demo")
    print("=" * 41)

    manager = MultimodalPipelineManager()

    # VQA Pipeline
    vqa_config = MultimodalPipelineConfig(pipeline_type=MultimodalPipelineType.VISUAL_QUESTION_ANSWERING)
    vqa_input = MultimodalPipelineInput(
        text="What do you see in this image?",
        image=np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8),
        config=vqa_config
    )

    print("❓ Procesando VQA...")
    vqa_result = await manager.process_multimodal_pipeline(vqa_input)
    print(f"   • Respuesta: {vqa_result.primary_output[:50]}...")

    # Image Captioning
    caption_config = MultimodalPipelineConfig(pipeline_type=MultimodalPipelineType.IMAGE_CAPTIONING)
    caption_input = MultimodalPipelineInput(
        image=np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8),
        config=caption_config
    )

    print("\\n📝 Generando caption...")
    caption_result = await manager.process_multimodal_pipeline(caption_input)
    print(f"   • Caption: {caption_result.primary_output[:50]}...")

    # Multimodal Sentiment
    sentiment_config = MultimodalPipelineConfig(pipeline_type=MultimodalPipelineType.MULTIMODAL_SENTIMENT_ANALYSIS)
    sentiment_input = MultimodalPipelineInput(
        text="I love this product!",
        audio=np.random.randn(500),
        config=sentiment_config
    )

    print("\\n😊 Analizando sentimiento...")
    sentiment_result = await manager.process_multimodal_pipeline(sentiment_input)
    print(f"   • Sentimiento: {sentiment_result.primary_output}")

    # Cross-Modal Retrieval
    retrieval_config = MultimodalPipelineConfig(pipeline_type=MultimodalPipelineType.CROSS_MODAL_RETRIEVAL)
    retrieval_input = MultimodalPipelineInput(
        text="bright scene",
        config=retrieval_config
    )

    print("\\n🔍 Retrieval cross-modal...")
    retrieval_result = await manager.process_multimodal_pipeline(retrieval_input)
    print(f"   • Items encontrados: {len(retrieval_result.primary_output)}")

    print("\\n🎉 Multimodal Pipelines funcionando!")

if __name__ == "__main__":
    asyncio.run(quick_multimodal_pipelines_demo())
