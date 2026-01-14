#!/usr/bin/env python3
"""
🎯 AEGIS NLP Demo - Sprint 4.3
Demostración rápida del sistema de Natural Language Processing
"""

import asyncio
from natural_language_processing import AEGISNaturalLanguageProcessing

async def quick_nlp_demo():
    """Demostración rápida de NLP"""

    print("🎯 AEGIS Natural Language Processing Quick Demo")
    print("=" * 47)

    nlp = AEGISNaturalLanguageProcessing()

    # Texto de ejemplo
    text = "Apple Inc. is a technology company founded by Steve Jobs."

    print(f"✅ Texto preparado: {text[:50]}...")

    # Procesar texto
    print("\\n🚀 Procesando texto...")
    results = await nlp.process_text(text)

    print("\\n📊 RESULTADOS:")
    for task, result in results.items():
        if hasattr(result, 'processing_time'):
            print(".3f")
        else:
            print(f"   • {task}: procesado")

    # Question answering
    question = "Who founded Apple?"
    qa_result = nlp.answer_question(question, text)
    print(f"\\n❓ QA: '{qa_result.answer}'")

    # Text generation
    generation = nlp.generate_text("Apple is known for", max_length=20)
    print(f"\\n✍️ Generation: {generation.generated_text}")

    print("\\n🎉 NLP funcionando!")

if __name__ == "__main__":
    asyncio.run(quick_nlp_demo())
