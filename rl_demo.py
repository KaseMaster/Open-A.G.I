#!/usr/bin/env python3
"""
🎯 AEGIS RL Demo - Sprint 4.2
Demostración rápida del sistema de Reinforcement Learning
"""

import asyncio
from reinforcement_learning_integration import AEGISReinforcementLearning, RLAlgorithm, RLConfig

async def quick_rl_demo():
    """Demostración rápida de RL"""

    print("🎯 AEGIS Reinforcement Learning Quick Demo")
    print("=" * 43)

    rl = AEGISReinforcementLearning()

    # Configurar CartPole
    await rl.setup_environment("cartpole", {"max_episode_steps": 200})

    # Configuración simple
    config = RLConfig(
        algorithm=RLAlgorithm.DQN,
        num_episodes=50,  # Muy reducido para demo
        max_steps_per_episode=100
    )

    print("\n🚀 Entrenando agente RL...")
    results = await rl.train_rl_agent("cartpole", RLAlgorithm.DQN, config)

    print(f"   ⏱️ Tiempo: {results['training_time']:.1f}s")
    print(f"   🏆 Recompensa: {results['final_reward']:.2f}")
    print(f"   📊 Éxito: {results['success_rate']:.1f}%")

    insights = rl.get_training_insights(results)
    print("\n💡 Insight:", insights[0] if insights else "Entrenamiento completado")

    print("\n🎉 RL funcionando!")

if __name__ == "__main__":
    asyncio.run(quick_rl_demo())
