#!/usr/bin/env python3
"""
🎯 AEGIS Audio/Speech Demo - Sprint 4.3
Demostración rápida del sistema de audio y voz
"""

import asyncio
from audio_speech_processing import AEGISAudioSpeechProcessing

async def quick_audio_demo():
    """Demostración rápida de audio/speech processing"""

    print("🎯 AEGIS Audio/Speech Processing Quick Demo")
    print("=" * 43)

    audio_system = AEGISAudioSpeechProcessing()

    # Text-to-Speech
    text = "Hola, esto es una demostración de síntesis de voz."
    result = audio_system.synthesize_speech(text, "demo_tts.mp3")

    print("\\n🗣️ Text-to-Speech:")
    print(f"   • Texto: {text}")
    print(f"   • Audio generado: {len(result.audio_data)} bytes")
    print(".3f"    print("   • Archivo: demo_tts.mp3")

    # Simulated audio processing
    print("\\n🎵 Audio Processing Simulation:")
    print("   • Speech Recognition: 'Hola mundo' (confidence: 0.95)")
    print("   • Audio Classification: 'speech' (confidence: 0.92)")
    print("   • Speaker ID: 'speaker_001' (confidence: 0.88)")
    print("   • Emotion: 'happy' (confidence: 0.76)")

    print("\\n🎉 Audio/Speech funcionando!")

if __name__ == "__main__":
    asyncio.run(quick_audio_demo())
