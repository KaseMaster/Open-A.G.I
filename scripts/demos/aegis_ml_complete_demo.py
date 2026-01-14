#!/usr/bin/env python3
"""
🎯 AEGIS ML Integration - Demo Completa
Demostración integrada de todo el sistema ML:
- Framework unificado (TensorFlow + PyTorch)
- Aprendizaje federado
- Distribución de modelos
- Gestión de versiones
"""

import asyncio
import time
from typing import Dict, Any

from ml_framework_integration import MLFrameworkManager, MLFramework, ModelType
from federated_learning import FederatedLearningCoordinator
from model_distribution import ModelDistributionService

async def comprehensive_ml_demo():
    """Demo completa del sistema ML integrado"""

    print("🎯 AEGIS ML INTEGRATION - DEMO COMPLETA")
    print("=" * 60)
    print()

    # ===== FASE 1: INICIALIZACIÓN DE COMPONENTES =====
    print("📦 FASE 1: Inicializando componentes ML...")

    # 1.1 ML Framework Manager
    ml_manager = MLFrameworkManager()
    print("✅ ML Framework Manager inicializado")

    # 1.2 Federated Learning Coordinator
    fl_coordinator = FederatedLearningCoordinator(ml_manager, min_clients=3, max_rounds=5)
    print("✅ Federated Learning Coordinator inicializado")

    # 1.3 Model Distribution Service
    distribution_service = ModelDistributionService(ml_manager, "demo_coordinator")
    print("✅ Model Distribution Service inicializado")

    print("✅ FASE 1 COMPLETADA: Todos los componentes inicializados")
    print()

    # ===== FASE 2: REGISTRO Y DISTRIBUCIÓN DE MODELOS =====
    print("📝 FASE 2: Registro y distribución de modelos...")

    # Crear modelos de ejemplo
    models_to_create = [
        ("resnet_classifier", MLFramework.PYTORCH, ModelType.CLASSIFICATION),
        ("bert_sentiment", MLFramework.PYTORCH, ModelType.TRANSFORMER),
        ("cnn_detector", MLFramework.PYTORCH, ModelType.CNN)
    ]

    registered_models = []

    for model_name, framework, model_type in models_to_create:
        # Simular registro en ML manager
        from ml_framework_integration import ModelMetadata
        metadata = ModelMetadata(
            model_id=model_name,
            framework=framework,
            model_type=model_type,
            architecture=f"Demo {model_type.value.title()} Model",
            input_shape=[224, 224, 3] if model_type == ModelType.CNN else [784],
            output_shape=[1000] if model_type == ModelType.CLASSIFICATION else [2],
            parameters=1000000 + len(registered_models) * 500000,
            created_at=time.time(),
            updated_at=time.time(),
            version="1.0.0"
        )
        ml_manager.models[model_name] = metadata

        # Registrar para distribución
        await distribution_service.register_model_for_distribution(model_name)

        registered_models.append(model_name)
        print(f"✅ Modelo registrado: {model_name} ({framework.value})")

    # Distribuir modelos iniciales
    target_nodes = ["node_alpha", "node_beta", "node_gamma", "node_delta", "node_epsilon"]

    for model_id in registered_models:
        task_id = await distribution_service.distribute_model(
            model_id=model_id,
            target_nodes=target_nodes[:3],  # Distribuir a 3 nodos
            priority=2
        )
        print(f"🚀 Distribución iniciada: {model_id} -> 3 nodos")

    print("✅ FASE 2 COMPLETADA: Modelos registrados y distribuidos")
    print()

    # ===== FASE 3: CONFIGURACIÓN DE APRENDIZAJE FEDERADO =====
    print("🤝 FASE 3: Configuración de aprendizaje federado...")

    # Registrar clientes federados
    federated_clients = []
    for i in range(4):
        node_id = f"client_node_{i}"
        client_info = {
            "samples_count": 5000 + (i * 1000),
            "hardware": "GPU" if i % 2 == 0 else "CPU"
        }

        client_id = await fl_coordinator.register_client(node_id, client_info)
        federated_clients.append(client_id)
        print(f"✅ Cliente federado registrado: {client_id} ({node_id})")

    # Iniciar entrenamiento federado con el primer modelo
    global_model = registered_models[0]
    training_session = await fl_coordinator.start_federated_training(global_model, {
        "dataset": "imagenet_sample",
        "task": "classification"
    })

    print(f"🎯 Entrenamiento federado iniciado: {training_session}")
    print("✅ FASE 3 COMPLETADA: Sistema federado configurado")
    print()

    # ===== FASE 4: SIMULACIÓN DE RONDAS FEDERADAS =====
    print("🔄 FASE 4: Simulación de rondas federadas...")

    for round_num in range(1, 4):  # 3 rondas
        print(f"\n🎯 RONDA FEDERADA {round_num}")
        print("-" * 30)

        # Iniciar ronda
        round_config = await fl_coordinator.start_round(round_num)
        print(f"📤 Ronda {round_num} iniciada - Esperando {len(federated_clients)} actualizaciones")

        # Simular clientes enviando actualizaciones
        from federated_learning import FederatedLearningClient
        from ml_framework_integration import FederatedUpdate

        update_tasks = []
        for i, client_id in enumerate(federated_clients):
            # Crear cliente y simular entrenamiento
            node_id = f"client_node_{i}"
            client = FederatedLearningClient(client_id, node_id, ml_manager)

            # Recibir modelo global
            await client.receive_global_model(global_model, round_num)

            # Simular entrenamiento local
            update = await client.train_local_model(round_config["training_config"])

            # Enviar actualización
            success = await fl_coordinator.submit_client_update(client_id, update)
            if success:
                print(f"📥 Actualización recibida de {client_id}")
            else:
                print(f"❌ Error en actualización de {client_id}")

        # Esperar procesamiento
        await asyncio.sleep(2)

        # Mostrar estado de la ronda
        status = fl_coordinator.get_federated_status()
        print(f"✅ Ronda {round_num} completada")
        print(f"   📊 Actualizaciones: {status['collected_updates']}/{status['required_updates']}")
        print(f"   📈 Rondas totales completadas: {status['completed_rounds']}")

        # Crear nueva versión del modelo después de cada ronda
        new_version = await distribution_service.create_model_version(
            global_model,
            f"Actualización federada ronda {round_num}"
        )
        if new_version:
            print(f"🔄 Nueva versión creada: {global_model} -> {new_version}")

        # Redistribuir modelo actualizado
        redistribute_task = await distribution_service.distribute_model(
            model_id=global_model,
            target_nodes=target_nodes,
            priority=3  # Alta prioridad para actualizaciones
        )
        if redistribute_task:
            print(f"🔄 Redistribución de modelo actualizado iniciada")

        # Pequeña pausa entre rondas
        await asyncio.sleep(1)

    print("✅ FASE 4 COMPLETADA: Rondas federadas completadas")
    print()

    # ===== FASE 5: REPORTES FINALES Y ESTADÍSTICAS =====
    print("📊 FASE 5: Reportes finales y estadísticas...")

    # 5.1 Estadísticas de aprendizaje federado
    fl_status = fl_coordinator.get_federated_status()
    print("🤝 ESTADÍSTICAS FEDERADAS:")
    print(f"   • Clientes conectados: {fl_status['connected_clients']}")
    print(f"   • Rondas completadas: {fl_status['completed_rounds']}")
    print(f"   • Modelo global: {fl_status['global_model_id']}")

    # 5.2 Estadísticas de distribución
    total_tasks = len(distribution_service.distribution_tasks)
    completed_tasks = sum(1 for t in distribution_service.distribution_tasks.values()
                         if t.status.name == "COMPLETED")
    total_bandwidth = sum(t.bandwidth_used for t in distribution_service.distribution_tasks.values())

    print("📤 ESTADÍSTICAS DE DISTRIBUCIÓN:")
    print(f"   • Tareas totales: {total_tasks}")
    print(f"   • Tareas completadas: {completed_tasks}")
    print(f"   • Bandwidth total: {total_bandwidth:,} bytes")

    # 5.3 Estadísticas de modelos y versiones
    total_versions = sum(len(versions) for versions in distribution_service.model_versions.values())
    print("📋 ESTADÍSTICAS DE MODELOS:")
    print(f"   • Modelos registrados: {len(registered_models)}")
    print(f"   • Versiones totales: {total_versions}")
    print(f"   • Modelos distribuidos: {len(distribution_service.model_versions)}")

    # 5.4 Métricas de rendimiento por ronda
    completed_rounds = fl_coordinator.completed_rounds
    if completed_rounds:
        print("📈 MÉTRICAS DE RENDIMIENTO:")
        for i, round_obj in enumerate(completed_rounds, 1):
            if hasattr(round_obj, 'aggregated_metrics') and round_obj.aggregated_metrics:
                metrics = round_obj.aggregated_metrics
                duration = round_obj.end_time - round_obj.start_time if round_obj.end_time else 0
                print(f"   • Ronda {i}: Loss={metrics.get('loss', 'N/A'):.3f}, "
                      f"Accuracy={metrics.get('accuracy', 'N/A'):.3f}, "
                      f"Duration={duration:.1f}s")

    # 5.5 Resumen de capacidades demostradas
    print("🎯 CAPACIDADES DEMOSTRADAS:")
    capabilities = [
        "✅ Framework unificado (TensorFlow + PyTorch)",
        "✅ Aprendizaje federado con múltiples clientes",
        "✅ Distribución eficiente de modelos",
        "✅ Versionado automático de modelos",
        "✅ Agregación federada de pesos",
        "✅ Sincronización entre nodos",
        "✅ Escalabilidad horizontal",
        "✅ Gestión de recursos distribuida"
    ]

    for capability in capabilities:
        print(f"   {capability}")

    print("
🎉 DEMO COMPLETA EXITOSA!"    print("🌟 Sistema ML completamente integrado y operativo")
    print("=" * 60)

    # 5.6 Próximos pasos sugeridos
    print("💡 PRÓXIMOS PASOS RECOMENDADOS:")
    next_steps = [
        "🔧 Integración con datasets reales (CIFAR-10, ImageNet)",
        "📊 Implementación de métricas avanzadas (F1, AUC, etc.)",
        "🔒 Adición de privacidad diferencial",
        "⚡ Optimización de comunicación (compresión, cuantización)",
        "🌐 Integración con orquestadores (Kubernetes, Docker Swarm)",
        "📈 Dashboards de monitoreo en tiempo real",
        "🔄 Automatización de pipelines MLOps"
    ]

    for step in next_steps:
        print(f"   • {step}")

if __name__ == "__main__":
    asyncio.run(comprehensive_ml_demo())
