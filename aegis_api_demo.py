#!/usr/bin/env python3
"""
🎯 AEGIS Enterprise API Demo - Sprint 5.1
Demostración rápida de la API REST enterprise
"""

import asyncio
import requests
import json
import base64
import numpy as np
from PIL import Image
import io

async def quick_api_demo():
    """Demostración rápida de la API enterprise"""

    print("🎯 AEGIS Enterprise API Quick Demo")
    print("=" * 36)

    base_url = "http://localhost:8000"
    api_key = "demo_key_2024"  # API key de demo
    headers = {"Authorization": f"Bearer {api_key}"}

    try:
        # Health check
        print("🏥 Health Check...")
        response = requests.get(f"{base_url}/health")
        print(f"   • Status: {response.json()['status']}")

        # Capabilities
        print("\\n📋 Capabilities...")
        response = requests.get(f"{base_url}/api/v1/capabilities", headers=headers)
        capabilities = response.json()['data']
        print(f"   • Text analysis: {len(capabilities['text_analysis'])} tasks")
        print(f"   • Image analysis: {len(capabilities['image_analysis'])} tasks")
        print(f"   • Audio analysis: {len(capabilities['audio_analysis'])} tasks")

        # Text analysis
        print("\\n📝 Text Analysis...")
        text_data = {"text": "This is a great product!", "tasks": ["sentiment"]}
        response = requests.post(f"{base_url}/api/v1/text/analyze", json=text_data, headers=headers)
        result = response.json()
        print(f"   • Sentiment: {result['data']['results'].get('sentiment', 'processed')}")

        # Image generation (simulado)
        print("\\n🎨 Image Generation...")
        gen_data = {"prompt": "A beautiful sunset", "generation_type": "text"}
        response = requests.post(f"{base_url}/api/v1/generate/text", data=gen_data, headers=headers)
        result = response.json()
        print(f"   • Generated: {result['data']['result'][:50]}...")

        # Pipeline processing
        print("\\n🔄 Pipeline Processing...")
        pipeline_data = {
            "pipeline_type": "multimodal_pipeline",
            "input_data": {"text": "Hello world"},
            "config": {}
        }
        response = requests.post(f"{base_url}/api/v1/pipeline/process", json=pipeline_data, headers=headers)
        result = response.json()
        print(f"   • Pipeline completed: {len(result['data']['results'])} results")

        print("\\n🎉 API funcionando correctamente!")

    except requests.exceptions.ConnectionError:
        print("❌ API server no está corriendo. Ejecutar: python aegis_api.py")
        print("\\n💡 Para probar la API:")
        print("   1. En terminal: python aegis_api.py")
        print("   2. En browser: http://localhost:8000/docs")
        print("   3. API Key: demo_key_2024")

    except Exception as e:
        print(f"❌ Error en demo: {e}")

if __name__ == "__main__":
    asyncio.run(quick_api_demo())
