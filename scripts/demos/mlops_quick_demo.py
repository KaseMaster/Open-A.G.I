#!/usr/bin/env python3
"""
🎯 AEGIS MLOps Demo - Sprint 4.1
Demostración rápida del sistema de Model Versioning & Experiment Tracking
"""

import asyncio
import time
from model_versioning_tracking import AEGISModelOps, ModelVersion, ModelStage
from ml_framework_integration import MLFramework

async def quick_mlops_demo():
    """Demostración rápida del sistema MLOps"""

    print("🎯 AEGIS MLOps Quick Demo")
    print("=" * 30)

    # Inicializar MLOps
    mlops = AEGISModelOps()
    mlops.initialize_storage("./demo_mlops")

    print("✅ MLOps inicializado")

    # Crear experimento
    exp_id = await mlops.create_experiment(
        "Quick MLOps Demo",
        "Demostración rápida del sistema MLOps",
        ["demo", "quickstart"]
    )

    print(f"✅ Experimento creado: {exp_id[:8]}...")

    # Ejecutar 3 runs rápidas
    configs = [
        {"lr": 0.01, "batch_size": 32},
        {"lr": 0.001, "batch_size": 64},
        {"lr": 0.0001, "batch_size": 128}
    ]

    run_results = []

    for i, config in enumerate(configs):
        print(f"\\n🏃 Run {i+1}: lr={config['lr']}, batch={config['batch_size']}")

        # Iniciar run
        run_id = await mlops.start_run(exp_id, f"Config {i+1}", config)

        # Simular métricas
        final_acc = 0.8 + (i * 0.05) + (time.time() % 0.1)

        await mlops.log_to_run(run_id,
                              metric_accuracy=final_acc,
                              metric_loss=0.3 - (i * 0.05),
                              param_learning_rate=config["lr"],
                              param_batch_size=config["batch_size"])

        await mlops.end_run(run_id, True)

        run_results.append((run_id, final_acc))

        print(".3f"
    # Registrar mejor modelo
    best_run = max(run_results, key=lambda x: x[1])
    best_run_id, best_acc = best_run

    model_version = ModelVersion(
        model_name="demo_classifier",
        version="v1.0",
        model_id=f"model_{best_run_id}",
        framework=MLFramework.PYTORCH.value,
        architecture="MLP",
        hyperparameters={"learning_rate": 0.001, "batch_size": 64},
        metrics={"accuracy": best_acc, "precision": best_acc - 0.02},
        tags=["demo", "classification"],
        description="Modelo generado por demo MLOps"
    )

    await mlops.register_model_version(model_version)
    print(f"\\n✅ Mejor modelo registrado: {model_version.model_name} {model_version.version}")

    # Promover a producción
    await mlops.promote_model("demo_classifier", "v1.0", ModelStage.PRODUCTION)
    print("📈 Modelo promovido a PRODUCCIÓN")

    # Mostrar resumen
    experiments = mlops.list_experiments()
    models = mlops.list_models()

    print("
📊 RESUMEN:"    print(f"   • Experimentos: {len(experiments)}")
    print(f"   • Modelos: {len(models)}")
    print(f"   • En producción: {len(mlops.list_models(ModelStage.PRODUCTION))}")

    print("\\n🎉 ¡MLOps funcionando perfectamente!")

if __name__ == "__main__":
    asyncio.run(quick_mlops_demo())
