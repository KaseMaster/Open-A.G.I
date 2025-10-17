#!/usr/bin/env python3
"""
🎯 AEGIS Generative AI Demo - Sprint 4.3
Demostración rápida del sistema de IA Generativa
"""

import asyncio
from generative_ai import AEGISGenerativeAI

async def quick_generative_demo():
    """Demostración rápida de Generative AI"""

    print("🎯 AEGIS Generative AI Quick Demo")
    print("=" * 33)

    gen_ai = AEGISGenerativeAI()

    # Text generation
    prompt = "The future of AI is"
    text_result = await gen_ai.generate_text(prompt)

    print("\\n✍️ Text Generation:")
    print(f"   • Prompt: {prompt}")
    print(f"   • Generated: {text_result.generated_text[:80]}...")

    # Simulated image generation
    image_result = await gen_ai.generate_image("A beautiful landscape")
    print("\\n🎨 Image Generation:")
    print(f"   • Prompt: A beautiful landscape")
    print(f"   • Shape: {image_result.image.shape}")
    print(".3f"
    # Story with images
    story = await gen_ai.generate_story_with_images("space exploration", 2)
    print("\\n📖 Story Generation:")
    print(f"   • Theme: space exploration")
    print(f"   • Scenes: {len(story['scenes'])}")
    print(f"   • Story: {story['full_story'][:60]}...")

    print("\\n🎉 Generative AI funcionando!")

if __name__ == "__main__":
    asyncio.run(quick_generative_demo())
