#!/usr/bin/env python3
"""
🎯 AEGIS GNN Demo - Sprint 4.2
Demostración rápida del sistema de Graph Neural Networks
"""

import asyncio
from graph_neural_networks import AEGISGraphNeuralNetworks, GNNConfig, GNNType, TaskType

async def quick_gnn_demo():
    """Demostración rápida de GNN"""

    print("🎯 AEGIS Graph Neural Networks Quick Demo")
    print("=" * 42)

    gnn = AEGISGraphNeuralNetworks()

    # Crear grafo pequeño
    graph_data = gnn.data_loader.create_synthetic_graph(
        num_nodes=50, num_features=8, num_classes=2, edge_prob=0.1
    )

    print(f"✅ Grafo sintético: {graph_data.num_nodes} nodos, {graph_data.num_edges} aristas")

    # Configuración GNN simple
    config = GNNConfig(
        model_type=GNNType.GCN,
        task_type=TaskType.NODE_CLASSIFICATION,
        hidden_dims=[32, 16],
        epochs=20,
        patience=5
    )

    # Entrenar modelo
    print("\\n🚀 Entrenando GCN...")
    result = await gnn.train_gnn_model(graph_data, config)

    print(".3f"    print(".3f"    print(".1f"
    # Análisis rápido
    analysis = await gnn.analyze_graph_properties(graph_data)
    print(f"\\n📊 Grafo: densidad {analysis['basic_stats']['density']:.3f}, "
          f"{analysis['num_communities']} comunidades")

    print("\\n🎉 GNN funcionando correctamente!")

if __name__ == "__main__":
    asyncio.run(quick_gnn_demo())
