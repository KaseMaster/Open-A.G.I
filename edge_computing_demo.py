#!/usr/bin/env python3
"""
🎯 AEGIS Edge Computing - Demo Completa
Demostración del sistema de computación en el borde
con optimización de modelos y aprendizaje federado distribuido
"""

import asyncio
import time
from edge_computing import (
    EdgeComputingSystem, DeviceType, EdgeCapability,
    ModelOptimization
)

async def demo_edge_computing():
    """Demo completa del sistema edge computing"""

    print("🛠️ AEGIS Edge Computing - Demo Completa")
    print("=" * 50)
    print()

    # ===== FASE 1: INICIALIZACIÓN DEL SISTEMA EDGE =====
    print("📡 FASE 1: Inicialización del sistema edge...")

    edge_system = EdgeComputingSystem()

    # Definir dispositivos edge de ejemplo
    edge_devices = [
        {
            "device_type": "raspberry_pi",
            "capabilities": ["inference_only", "federated_client", "data_collection"],
            "hardware_specs": {"cpu": "ARM Cortex-A72", "ram": "4GB", "storage": "32GB"},
            "location": {"lat": 40.7128, "lon": -74.0060}  # NYC
        },
        {
            "device_type": "jetson_nano",
            "capabilities": ["inference_only", "training_mini_batch", "federated_client"],
            "hardware_specs": {"gpu": "128-core Maxwell", "cpu": "Quad-core ARM", "ram": "4GB"},
            "location": {"lat": 34.0522, "lon": -118.2437}  # LA
        },
        {
            "device_type": "coral_dev_board",
            "capabilities": ["inference_only", "real_time_processing"],
            "hardware_specs": {"tpu": "Edge TPU", "cpu": "NXP i.MX 8M", "ram": "2GB"},
            "location": {"lat": 41.8781, "lon": -87.6298}  # Chicago
        },
        {
            "device_type": "esp32",
            "capabilities": ["inference_only", "data_collection"],
            "hardware_specs": {"cpu": "Xtensa dual-core", "ram": "520KB", "storage": "4MB"},
            "location": {"lat": 37.7749, "lon": -122.4194}  # SF
        },
        {
            "device_type": "mobile_phone",
            "capabilities": ["inference_only", "federated_client", "real_time_processing"],
            "hardware_specs": {"cpu": "ARMv8", "ram": "8GB", "gpu": "Adreno"},
            "location": {"lat": 47.6062, "lon": -122.3321}  # Seattle
        }
    ]

    registered_devices = []

    # Registrar dispositivos
    for device_info in edge_devices:
        device_id = await edge_system.register_edge_device(device_info)
        if device_id:
            registered_devices.append(device_id)
            print(f"✅ Dispositivo registrado: {device_id} ({device_info['device_type']})")

    print(f"✅ Registrados {len(registered_devices)} dispositivos edge")
    print("✅ FASE 1 COMPLETADA: Sistema edge inicializado")
    print()

    # ===== FASE 2: OPTIMIZACIÓN DE MODELOS =====
    print("🔧 FASE 2: Optimización de modelos para edge...")

    # Modelos a optimizar
    optimization_tasks = [
        ("resnet_classifier", DeviceType.RASPBERRY_PI, ModelOptimization.QUANTIZATION),
        ("bert_sentiment", DeviceType.JETSON_NANO, ModelOptimization.TENSORRT),
        ("cnn_detector", DeviceType.CORAL_DEV_BOARD, ModelOptimization.TFLITE),
        ("mobile_net", DeviceType.MOBILE_PHONE, ModelOptimization.QUANTIZATION),
        ("tiny_model", DeviceType.ESP32, ModelOptimization.QUANTIZATION)
    ]

    optimized_models = []

    for model_name, device_type, optimization in optimization_tasks:
        # Encontrar dispositivos compatibles
        compatible_devices = []
        if device_type == DeviceType.RASPBERRY_PI:
            compatible_devices = registered_devices[:2]  # Primeros 2 dispositivos
        elif device_type == DeviceType.JETSON_NANO:
            compatible_devices = registered_devices[1:3]  # Dispositivos 2-3
        elif device_type == DeviceType.CORAL_DEV_BOARD:
            compatible_devices = registered_devices[2:4]  # Dispositivos 3-4
        elif device_type == DeviceType.MOBILE_PHONE:
            compatible_devices = registered_devices[4:]  # Último dispositivo
        elif device_type == DeviceType.ESP32:
            compatible_devices = registered_devices[3:5]  # Dispositivos 4-5

        if compatible_devices:
            deployment_ids = await edge_system.optimize_and_deploy_model(
                model_name, device_type, compatible_devices, optimization
            )

            if deployment_ids:
                optimized_models.append((model_name, device_type, optimization, len(deployment_ids)))
                print(f"✅ Optimizado y desplegado: {model_name} -> {len(deployment_ids)} dispositivos")

    print(f"✅ Optimizados {len(optimized_models)} modelos para edge")
    print("✅ FASE 2 COMPLETADA: Modelos optimizados y desplegados")
    print()

    # ===== FASE 3: APRENDIZAJE FEDERADO EN EDGE =====
    print("🤝 FASE 3: Aprendizaje federado distribuido en edge...")

    # Seleccionar dispositivos con capacidad federada
    federated_devices = []
    for device_id in registered_devices:
        status = edge_system.device_manager.get_device_status(device_id)
        if status and "federated_client" in status["capabilities"]:
            federated_devices.append(device_id)

    print(f"📱 Encontrados {len(federated_devices)} dispositivos con capacidad federada")

    if len(federated_devices) >= 3:
        # Iniciar ronda federada
        model_for_federated = "federated_model_v1"
        round_id = await edge_system.start_edge_federated_learning(
            model_for_federated, federated_devices
        )

        if round_id:
            print(f"🚀 Ronda federada iniciada: {round_id} con {len(federated_devices)} dispositivos")

            # Simular envío de actualizaciones desde dispositivos
            update_tasks = []
            for i, device_id in enumerate(federated_devices):
                # Simular actualización con datos del dispositivo
                update_data = {
                    "weights_delta": f"mock_weights_device_{i}",
                    "sample_count": 100 + (i * 50),
                    "metrics": {
                        "loss": 0.5 - (i * 0.05),
                        "accuracy": 0.8 + (i * 0.03),
                        "latency": 50 + (i * 10)
                    }
                }

                # Enviar actualización
                success = await edge_system.federated_coordinator.submit_edge_update(
                    round_id, device_id, update_data
                )

                if success:
                    print(f"📤 Actualización enviada desde {device_id}")

                # Pequeño delay entre actualizaciones
                await asyncio.sleep(0.5)

            print(f"✅ Todas las actualizaciones enviadas para ronda {round_id}")

        else:
            print("⚠️ No se pudo iniciar ronda federada (mínimo 3 dispositivos requeridos)")

    print("✅ FASE 3 COMPLETADA: Aprendizaje federado ejecutado")
    print()

    # ===== FASE 4: MONITOREO Y MÉTRICAS =====
    print("📊 FASE 4: Monitoreo del sistema edge...")

    # Obtener estado del sistema
    system_status = edge_system.get_system_status()
    print("🌐 ESTADO DEL SISTEMA EDGE:")
    print(f"   • Dispositivos totales: {system_status['total_devices']}")
    print(f"   • Dispositivos online: {system_status['online_devices']}")
    print(f"   • Dispositivos offline: {system_status['offline_devices']}")
    print(f"   • Despliegues activos: {system_status['active_deployments']}")
    print(f"   • Modelos optimizados: {system_status['optimized_models']}")
    print(f"   • Rondas federadas activas: {system_status['active_federated_rounds']}")
    print(f"   • Rondas federadas completadas: {system_status['completed_federated_rounds']}")

    # Mostrar modelos edge
    edge_models = edge_system.get_edge_models()
    print("\n🧠 MODELOS EDGE OPTIMIZADOS:")
    for model in edge_models:
        size_mb = model['model_size_bytes'] / (1024 * 1024)
        print(f"   • {model['model_id']}")
        print(f"     - Original: {model['original_model_id']}")
        print(f"     - Optimización: {model['optimization']}")
        print(f"     - Tamaño: {size_mb:.1f} MB")
        print(f"     - Tiempo inferencia: {model['inference_time_ms']:.1f} ms")
        print(f"     - Caída accuracy: {model['accuracy_drop']:.1f}%")

    # Mostrar estado detallado de dispositivos
    print("\n📱 ESTADO DE DISPOSITIVOS EDGE:")
    for device_id in registered_devices:
        status = edge_system.device_manager.get_device_status(device_id)
        if status:
            print(f"   • {device_id} ({status['device_type']})")
            print(f"     - Estado: {status['status']}")
            print(f"     - Modelos desplegados: {len(status['deployed_models'])}")
            if status['battery_level'] is not None:
                print(f"     - Batería: {status['battery_level']:.1f}%")
            if status['temperature'] is not None:
                print(f"     - Temperatura: {status['temperature']:.1f}°C")

    print("✅ FASE 4 COMPLETADA: Sistema completamente monitoreado")
    print()

    # ===== FASE 5: REPORTES FINALES Y OPTIMIZACIONES =====
    print("📋 FASE 5: Reportes finales y recomendaciones...")

    # Calcular métricas de eficiencia
    total_models_size = sum(m['model_size_bytes'] for m in edge_models)
    avg_inference_time = sum(m['inference_time_ms'] for m in edge_models) / len(edge_models) if edge_models else 0
    avg_accuracy_drop = sum(m['accuracy_drop'] for m in edge_models) / len(edge_models) if edge_models else 0

    print("⚡ MÉTRICAS DE EFICIENCIA:")
    print(f"   • Tamaño total modelos: {total_models_size / (1024*1024):.1f} MB")
    print(f"   • Tiempo inferencia promedio: {avg_inference_time:.1f} ms")
    print(f"   • Caída accuracy promedio: {avg_accuracy_drop:.1f}%")

    # Análisis de distribución de capacidades
    capability_distribution = {}
    for device_id in registered_devices:
        status = edge_system.device_manager.get_device_status(device_id)
        if status:
            for cap in status['capabilities']:
                capability_distribution[cap] = capability_distribution.get(cap, 0) + 1

    print("\n🎯 DISTRIBUCIÓN DE CAPACIDADES:")
    for capability, count in capability_distribution.items():
        percentage = (count / len(registered_devices)) * 100
        print(f"   • {capability}: {count} dispositivos ({percentage:.1f}%)")

    # Recomendaciones de optimización
    print("\n💡 RECOMENDACIONES DE OPTIMIZACIÓN:")
    if avg_accuracy_drop > 0.03:
        print("   • Considerar técnicas de cuantización más avanzadas")
    if avg_inference_time > 100:
        print("   • Evaluar optimizaciones adicionales (pruning, distillation)")
    if system_status['offline_devices'] > 0:
        print("   • Implementar reconexión automática para dispositivos offline")
    if len(federated_devices) < len(registered_devices) * 0.6:
        print("   • Expandir capacidad federada a más dispositivos")

    print("✅ FASE 5 COMPLETADA: Optimizaciones analizadas")
    print()

    # ===== RESULTADOS FINALES =====
    print("🏆 RESULTADOS FINALES - SISTEMA EDGE COMPUTING")
    print("=" * 55)

    success_metrics = {
        "dispositivos_registrados": len(registered_devices),
        "modelos_optimizados": len(edge_models),
        "despliegues_realizados": system_status['active_deployments'],
        "rondas_federadas": len(edge_system.federated_coordinator.completed_rounds),
        "capacidades_soportadas": len(capability_distribution)
    }

    print("🎯 LOGROS ALCANZADOS:")
    for metric, value in success_metrics.items():
        print(f"   • {metric.replace('_', ' ').title()}: {value}")

    # Calcular score de éxito
    max_score = 100
    actual_score = min(max_score, (
        (len(registered_devices) / 5 * 20) +      # 20% por dispositivos
        (len(edge_models) / 5 * 20) +             # 20% por modelos optimizados
        (system_status['active_deployments'] / 10 * 20) +  # 20% por despliegues
        (len(edge_system.federated_coordinator.completed_rounds) / 1 * 20) +  # 20% por FL
        (len(capability_distribution) / 5 * 20)   # 20% por capacidades
    ))

    print("\n📊 PUNTUACIÓN DE ÉXITO:")
    print(".1f"    if actual_score >= 80:
        print("   Estado: ✅ EXCELENTE - Sistema completamente operativo")
    elif actual_score >= 60:
        print("   Estado: ⚠️ BUENO - Funcional pero con oportunidades de mejora")
    else:
        print("   Estado: ❌ REQUIERE ATENCIÓN - Revisar configuración")

    # Próximos pasos
    print("\n🚀 PRÓXIMOS PASOS RECOMENDADOS:")
    print("   • Implementar actualización OTA para dispositivos edge")
    print("   • Agregar soporte para más tipos de dispositivos IoT")
    print("   • Implementar compresión de datos para redes limitadas")
    print("   • Desarrollar dashboard de monitoreo en tiempo real")
    print("   • Integrar con sistemas de gestión de energía")

    print("\n🎉 DEMO COMPLETA EXITOSA!")
    print("🌟 Sistema de edge computing completamente operativo")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(demo_edge_computing())
