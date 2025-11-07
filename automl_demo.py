#!/usr/bin/env python3
"""
🎯 AEGIS AutoML Demo - Sprint 4.1
Demostración completa del sistema de Automated Machine Learning
"""

import asyncio
import time
import json
import pandas as pd
import numpy as np
from aegis_automl import AEGISAutoML, AutoMLConfig, TaskType, DataType

async def comprehensive_automl_demo():
    """Demostración completa y avanzada de AutoML"""

    print("🚀 AEGIS AutoML - Comprehensive Demo")
    print("=" * 50)
    print("🤖 Automated Machine Learning Showcase")
    print("=" * 50)

    automl = AEGISAutoML()

    # ===== DEMO 1: CLASIFICACIÓN TABULAR =====
    print("\\n📊 DEMO 1: AutoML para Clasificación Tabular")
    print("-" * 45)

    # Crear dataset sintético de clasificación
    np.random.seed(42)
    n_samples, n_features = 2000, 15

    # Generar features correlacionados
    X = np.random.randn(n_samples, n_features)
    # Hacer algunas features más importantes
    X[:, 0] = X[:, 0] * 2 + np.random.randn(n_samples) * 0.1  # Feature importante
    X[:, 1] = X[:, 1] * 1.5 + np.random.randn(n_samples) * 0.1  # Feature importante

    # Target basado en combinación de features
    linear_combo = X[:, 0] * 2.0 + X[:, 1] * -1.5 + X[:, 2] * 0.8 + np.random.randn(n_samples) * 0.5
    y = (linear_combo > linear_combo.mean()).astype(int)

    # Crear DataFrame
    feature_names = [f"feature_{i}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=pd.Index(feature_names))
    df["target"] = y

    print(f"✅ Dataset creado: {len(df)} muestras")
    print(f"   • Features: {n_features}")
    print(f"   • Clases: {df['target'].nunique()} (distribución: {dict(df['target'].value_counts())})")

    # Configuración AutoML
    config = AutoMLConfig(
        task_type=TaskType.CLASSIFICATION,
        data_type=DataType.TABULAR,
        target_metric="accuracy",
        max_time_minutes=8,
        max_models=8,
        ensemble_size=3,
        cv_folds=3,
        optimization_budget=0.25,
        feature_engineering=True,
        hyperparameter_tuning=True
    )

    print("\\n⚙️ Configuración AutoML:")
    print(f"   • Tiempo máximo: {config.max_time_minutes} minutos")
    print(f"   • Máximo modelos: {config.max_models}")
    print(f"   • Ensemble size: {config.ensemble_size}")
    print(f"   • CV folds: {config.cv_folds}")

    # Ejecutar AutoML
    print("\\n🚀 Ejecutando AutoML para clasificación...")
    start_time = time.time()

    result = await automl.run_automl(df, config)

    demo1_time = time.time() - start_time

    if result.status == "completed":
        print("\n🎉 ¡AutoML completado exitosamente!")
        print(f"   ⏱️ Tiempo total: {demo1_time:.1f}s")
        print(f"   📊 Modelos evaluados: {result.models_evaluated}")
        print(f"   🏆 Mejor arquitectura: {result.best_model_id.split('_')[1]}")

        if result.ensemble_models:
            print(f"   🤝 Ensemble creado con {len(result.ensemble_models)} modelos")

        if result.optimization_applied:
            print("   🔧 Optimización aplicada al mejor modelo")
    else:
        print(f"❌ AutoML falló: revisar configuración")
        return

    # ===== DEMO 2: COMPARACIÓN DE TAREAS =====
    print("\\n\\n📈 DEMO 2: Comparación de Diferentes Tareas de ML")
    print("-" * 50)

    tasks_to_compare = [
        (TaskType.CLASSIFICATION, "Clasificación Binaria"),
        (TaskType.REGRESSION, "Regresión"),
        (TaskType.IMAGE_CLASSIFICATION, "Clasificación de Imágenes"),
    ]

    comparison_results = []

    for task_type, task_name in tasks_to_compare:
        print(f"\\n🎯 Probando {task_name}...")

        # Crear datos apropiados para cada tarea
        if task_type == TaskType.CLASSIFICATION:
            # Usar datos del demo 1
            task_data = df.copy()
        elif task_type == TaskType.REGRESSION:
            # Crear datos de regresión
            X_reg = np.random.randn(1500, 10)
            y_reg = X_reg[:, 0] * 2.3 + X_reg[:, 1] * -1.7 + X_reg[:, 2] * 0.5 + np.random.randn(1500) * 0.3
            df_reg = pd.DataFrame(X_reg, columns=pd.Index([f"feat_{i}" for i in range(10)]))
            df_reg["target"] = y_reg
            task_data = df_reg
        else:  # Image classification
            # Simular datos de imágenes
            task_data = {
                "images": [np.random.rand(224, 224, 3) for _ in range(500)],
                "labels": np.random.randint(0, 10, 500)
            }

        # Configuración simplificada para comparación rápida
        compare_config = AutoMLConfig(
            task_type=task_type,
            data_type=DataType.TABULAR if isinstance(task_data, pd.DataFrame) else DataType.IMAGE,
            target_metric="accuracy" if task_type != TaskType.REGRESSION else "r2",
            max_time_minutes=3,
            max_models=4
        )

        task_start = time.time()
        task_result = await automl.run_automl(task_data, compare_config)
        task_time = time.time() - task_start

        if task_result.status == "completed":
            comparison_results.append({
                "task": task_name,
                "score": task_result.best_score,
                "models": task_result.models_evaluated,
                "time": task_time,
                "ensemble": len(task_result.ensemble_models) > 0
            })

            print(f"   ⏱️ Tiempo: {task_time:.1f}s")
            print(f"   📊 Score: {task_result.best_score:.3f}")
        else:
            print("❌ Falló")

    # Mostrar comparación
    print("\\n📊 COMPARACIÓN DE TAREAS:")
    print("-" * 40)
    for comp in comparison_results:
        ensemble_icon = "🤝" if comp["ensemble"] else "🏆"
        print(f"   {ensemble_icon} {comp['task']}: {comp['score']:.3f} "
              f"({comp['models']} modelos, {comp['time']:.1f}s)")

    # ===== DEMO 3: OPTIMIZACIÓN AUTOMÁTICA =====
    print("\\n\\n🔧 DEMO 3: Optimización Automática Integrada")
    print("-" * 45)

    # Usar el mejor modelo del demo 1 para optimización
    if result.status == "completed":
        print("🎯 Optimizando el mejor modelo encontrado...")

        from advanced_model_optimization import (
            OptimizationTechnique, OptimizationConfig as OptConfig
        )

        # Aplicar múltiples técnicas de optimización
        optimization_scenarios = [
            ("Quantization", OptimizationTechnique.QUANTIZATION, {"compression_ratio": 0.5}),
            ("Pruning", OptimizationTechnique.PRUNING, {"compression_ratio": 0.4}),
            ("Compresión Completa", OptimizationTechnique.COMPRESSION, {"compression_ratio": 0.7})
        ]

        optimization_results = []

        for opt_name, opt_technique, opt_params in optimization_scenarios:
            print(f"\\n🔧 Aplicando {opt_name}...")

            opt_config = OptConfig(
                technique=opt_technique,
                target_platform="cpu",
                compression_ratio=opt_params["compression_ratio"]
            )

            opt_start = time.time()
            opt_result = await automl.optimizer.optimize_model(
                result.final_model_id, opt_config
            )
            opt_time = time.time() - opt_start

            if opt_result.status == "completed":
                optimization_results.append({
                    "technique": opt_name,
                    "compression": opt_result.compression_ratio,
                    "performance_gain": opt_result.performance_gain,
                    "time": opt_time
                })

                print(f"   ⏱️ Tiempo: {opt_time:.2f}s")
                print(f"   📊 Compresión: {opt_result.compression_ratio:.3f}")
                print(f"   🚀 Ganancia: {opt_result.performance_gain:.1f}x")
            else:
                print(f"❌ {opt_name} falló")

        # Resumen de optimización
        if optimization_results:
            print("\\n📊 RESULTADOS DE OPTIMIZACIÓN:")
            best_opt = max(optimization_results, key=lambda x: x["compression"])
            print(f"   💾 Ahorro máximo: {(1 - best_opt['compression']) * 100:.1f}%")

    # ===== DEMO 4: ANÁLISIS DE EFICIENCIA =====
    print("\\n\\n📈 DEMO 4: Análisis de Eficiencia AutoML")
    print("-" * 40)

    # Estadísticas del sistema AutoML
    stats = automl.get_automl_stats()

    print("📊 ESTADÍSTICAS GENERALES:")
    print(f"   • Ejecuciones totales: {stats['total_runs']}")
    print(f"   • Tasa de éxito: {stats['completed_runs']}/{stats['total_runs']} "
          f"({stats['completed_runs'] / max(stats['total_runs'], 1) * 100:.1f}%)")
    print(f"   • Modelos generados: {stats['total_models_generated']}")
    print(f"   • Optimizaciones aplicadas: {stats['optimization_applied']}")

    # Análisis de eficiencia
    total_time_all_demos = demo1_time + sum(c["time"] for c in comparison_results)
    total_models_all_demos = result.models_evaluated + sum(c["models"] for c in comparison_results)

    print("\\n⚡ EFICIENCIA:")
    print(f"   • Tiempo total: {total_time_all_demos:.1f}s")
    print(f"   • Modelos totales: {total_models_all_demos}")
    print(f"   🎯 Modelos por minuto: {total_models_all_demos / (total_time_all_demos/60):.1f}")

    # ===== RESULTADOS FINALES =====
    print("\\n\\n🎉 DEMO COMPLETA - RESULTADOS FINALES")
    print("=" * 50)

    print("🏆 LOGROS ALCANZADOS:")
    print(f"   ✅ AutoML ejecutado exitosamente")
    print(f"   ✅ {stats['total_runs']} tareas de ML automatizadas")
    print(f"   ✅ {stats['total_models_generated']} arquitecturas generadas automáticamente")
    print(f"   ✅ {len(comparison_results)} tipos de tarea diferentes probados")
    print(f"   ✅ Optimización integrada funcionando")

    print("\\n🚀 CAPACIDADES DEMOSTRADAS:")
    print("   • 🤖 Generación automática de arquitecturas de modelo")
    print("   • 📊 Análisis inteligente de datos")
    print("   • 🏋️ Entrenamiento y evaluación automatizados")
    print("   • 🤝 Creación de ensembles de modelos")
    print("   • 🔧 Optimización integrada (quantization, pruning)")
    print("   • 📈 Comparación automática de performance")

    print("\\n💡 INSIGHTS OBTENIDOS:")
    print("   • AutoML puede encontrar modelos competitivos en minutos")
    print("   • La optimización integrada mejora significativamente el despliegue")
    print("   • Los ensembles ofrecen mejor robustez que modelos individuales")
    print("   • Diferentes tareas requieren diferentes enfoques de optimización")

    print("\\n🔮 PRÓXIMOS PASOS PARA AutoML:")
    print("   • Implementar búsqueda de arquitectura neuronal (NAS)")
    print("   • Agregar soporte para más tipos de datos (time series, graphs)")
    print("   • Integrar técnicas de few-shot learning")
    print("   • Optimizar para edge devices automáticamente")
    print("   • Implementar AutoML distribuido")

    print("\\n" + "=" * 60)
    print("🌟 AEGIS AutoML - ¡El futuro del Machine Learning automatizado!")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(comprehensive_automl_demo())
