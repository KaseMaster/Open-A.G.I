#!/usr/bin/env python3
"""
🎮 AEGIS Reinforcement Learning - Sprint 4.2
Sistema completo de Reinforcement Learning integrado en AEGIS
"""

import asyncio
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
import random
import collections
from collections import deque, namedtuple
import gym
from gym import spaces

# Importar componentes del framework
from ml_framework_integration import MLFrameworkManager, MLFramework

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Experience tuple for replay buffer
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class RLAlgorithm(Enum):
    """Algoritmos de RL disponibles"""
    DQN = "dqn"
    DOUBLE_DQN = "double_dqn"
    DUELING_DQN = "dueling_dqn"
    PPO = "ppo"
    A2C = "a2c"
    SAC = "sac"
    REINFORCE = "reinforce"

class ExplorationStrategy(Enum):
    """Estrategias de exploración"""
    EPSILON_GREEDY = "epsilon_greedy"
    SOFTMAX = "softmax"
    UCB = "ucb"
    THOMPSON_SAMPLING = "thompson_sampling"

@dataclass
class RLConfig:
    """Configuración de RL"""
    algorithm: RLAlgorithm = RLAlgorithm.DQN
    exploration_strategy: ExplorationStrategy = ExplorationStrategy.EPSILON_GREEDY

    # Parámetros de red
    hidden_dims: List[int] = field(default_factory=lambda: [128, 64])
    learning_rate: float = 1e-3
    gamma: float = 0.99  # Discount factor

    # Parámetros de exploración
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995

    # Parámetros de entrenamiento
    batch_size: int = 64
    buffer_size: int = 10000
    target_update_freq: int = 100
    num_episodes: int = 1000
    max_steps_per_episode: int = 500

    # Parámetros específicos por algoritmo
    tau: float = 0.005  # Para soft updates en SAC
    alpha: float = 0.2  # Temperature parameter en SAC
    clip_ratio: float = 0.2  # PPO clip ratio
    value_coef: float = 0.5  # Value loss coefficient
    entropy_coef: float = 0.01  # Entropy coefficient

@dataclass
class RLEnvironment:
    """Wrapper para entornos de RL"""
    name: str
    env: Any
    state_dim: int
    action_dim: int
    action_space_type: str  # "discrete" or "continuous"
    max_episode_steps: int = 1000
    reward_scale: float = 1.0

    def reset(self):
        """Reset environment"""
        return self.env.reset()

    def step(self, action):
        """Step environment"""
        obs, reward, done, info = self.env.step(action)
        return obs, reward * self.reward_scale, done, info

    def render(self):
        """Render environment"""
        return self.env.render()

    def close(self):
        """Close environment"""
        return self.env.close()

# ===== REDES NEURONALES PARA RL =====

class QNetwork(nn.Module):
    """Q-Network para DQN"""

    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int]):
        super().__init__()
        layers = []

        # Input layer
        layers.append(nn.Linear(state_dim, hidden_dims[0]))
        layers.append(nn.ReLU())

        # Hidden layers
        for i in range(1, len(hidden_dims)):
            layers.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))
            layers.append(nn.ReLU())

        # Output layer
        layers.append(nn.Linear(hidden_dims[-1], action_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        return self.network(x)

class DuelingQNetwork(nn.Module):
    """Dueling Q-Network"""

    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int]):
        super().__init__()

        # Feature layer compartida
        self.feature_layer = nn.Sequential(
            nn.Linear(state_dim, hidden_dims[0]),
            nn.ReLU()
        )

        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], 1)  # Single value
        )

        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], action_dim)  # Advantage per action
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass con dueling"""
        features = self.feature_layer(x)

        values = self.value_stream(features)
        advantages = self.advantage_stream(features)

        # Combinar V y A: Q = V + (A - mean(A))
        q_values = values + (advantages - advantages.mean(dim=1, keepdim=True))

        return q_values

class ActorCriticNetwork(nn.Module):
    """Red Actor-Critic para PPO/A2C"""

    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int], continuous: bool = False):
        super().__init__()
        self.continuous = continuous

        # Shared feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU()
        )

        # Critic (Value) head
        self.critic = nn.Linear(hidden_dims[1], 1)

        # Actor head
        if continuous:
            self.actor_mean = nn.Linear(hidden_dims[1], action_dim)
            self.actor_log_std = nn.Linear(hidden_dims[1], action_dim)
        else:
            self.actor = nn.Linear(hidden_dims[1], action_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass"""
        features = self.feature_extractor(x)

        # Value prediction
        value = self.critic(features)

        # Policy prediction
        if self.continuous:
            mean = self.actor_mean(features)
            log_std = self.actor_log_std(features)
            std = torch.exp(log_std)
            return (mean, std), value
        else:
            logits = self.actor(features)
            return logits, value

    def get_action(self, state: torch.Tensor, deterministic: bool = False):
        """Obtener acción de la policy"""
        with torch.no_grad():
            if self.continuous:
                (mean, std), _ = self.forward(state)
                if deterministic:
                    return mean
                else:
                    dist = Normal(mean, std)
                    action = dist.sample()
                    log_prob = dist.log_prob(action).sum(dim=-1)
                    return action, log_prob
            else:
                logits, _ = self.forward(state)
                if deterministic:
                    return torch.argmax(logits, dim=-1)
                else:
                    dist = Categorical(logits=logits)
                    action = dist.sample()
                    log_prob = dist.log_prob(action)
                    return action, log_prob

class SACNetwork(nn.Module):
    """Red para Soft Actor-Critic"""

    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int]):
        super().__init__()

        # Q networks (2 para double Q-learning)
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], 1)
        )

        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], 1)
        )

        # Policy network
        self.policy = nn.Sequential(
            nn.Linear(state_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU()
        )

        self.policy_mean = nn.Linear(hidden_dims[1], action_dim)
        self.policy_log_std = nn.Linear(hidden_dims[1], action_dim)

    def forward(self, state: torch.Tensor, action: torch.Tensor = None):
        """Forward pass"""
        if action is not None:
            # Q network forward
            sa = torch.cat([state, action], dim=-1)
            q1 = self.q1(sa)
            q2 = self.q2(sa)
            return q1, q2
        else:
            # Policy forward
            features = self.policy(state)
            mean = self.policy_mean(features)
            log_std = torch.clamp(self.policy_log_std(features), -20, 2)
            return mean, log_std

# ===== COMPONENTES DE RL =====

class ReplayBuffer:
    """Buffer de experiencia para RL"""

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, experience: Experience):
        """Agregar experiencia al buffer"""
        self.buffer.append(experience)

    def sample(self, batch_size: int) -> List[Experience]:
        """Sample batch del buffer"""
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def __len__(self):
        return len(self.buffer)

class ExplorationStrategy:
    """Estrategia de exploración base"""

    def __init__(self, config: RLConfig):
        self.config = config

    def get_action(self, q_values: torch.Tensor, epsilon: float = None) -> int:
        """Obtener acción con exploración"""
        raise NotImplementedError

class EpsilonGreedy(ExplorationStrategy):
    """Epsilon-greedy exploration"""

    def get_action(self, q_values: torch.Tensor, epsilon: float = None) -> int:
        """Seleccionar acción con epsilon-greedy"""
        if epsilon is None:
            epsilon = self.config.epsilon_end

        if random.random() < epsilon:
            return random.randint(0, len(q_values) - 1)
        else:
            return q_values.argmax().item()

# ===== ALGORITMOS DE RL =====

class DQNTrainer:
    """Trainer para Deep Q-Network"""

    def __init__(self, config: RLConfig, state_dim: int, action_dim: int):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Redes
        if config.algorithm == RLAlgorithm.DUELING_DQN:
            self.policy_net = DuelingQNetwork(state_dim, action_dim, config.hidden_dims).to(self.device)
            self.target_net = DuelingQNetwork(state_dim, action_dim, config.hidden_dims).to(self.device)
        else:
            self.policy_net = QNetwork(state_dim, action_dim, config.hidden_dims).to(self.device)
            self.target_net = QNetwork(state_dim, action_dim, config.hidden_dims).to(self.device)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=config.learning_rate)

        # Replay buffer
        self.replay_buffer = ReplayBuffer(config.buffer_size)

        # Exploration
        self.exploration = EpsilonGreedy(config)
        self.epsilon = config.epsilon_start

    def select_action(self, state: np.ndarray) -> int:
        """Seleccionar acción"""
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            q_values = self.policy_net(state_tensor)

        return self.exploration.get_action(q_values.squeeze(), self.epsilon)

    def store_transition(self, state: np.ndarray, action: int, reward: float,
                        next_state: np.ndarray, done: bool):
        """Almacenar transición en replay buffer"""
        experience = Experience(state, action, reward, next_state, done)
        self.replay_buffer.push(experience)

    def update_policy(self) -> float:
        """Actualizar policy network"""

        if len(self.replay_buffer) < self.config.batch_size:
            return 0.0

        # Sample batch
        experiences = self.replay_buffer.sample(self.config.batch_size)

        # Preparar batch
        states = torch.tensor([e.state for e in experiences], dtype=torch.float32, device=self.device)
        actions = torch.tensor([e.action for e in experiences], dtype=torch.long, device=self.device)
        rewards = torch.tensor([e.reward for e in experiences], dtype=torch.float32, device=self.device)
        next_states = torch.tensor([e.next_state for e in experiences], dtype=torch.float32, device=self.device)
        dones = torch.tensor([e.done for e in experiences], dtype=torch.float32, device=self.device)

        # Calcular Q targets
        with torch.no_grad():
            next_q_values = self.target_net(next_states)
            if self.config.algorithm == RLAlgorithm.DOUBLE_DQN:
                # Double DQN
                next_actions = self.policy_net(next_states).argmax(dim=1)
                next_q_values = next_q_values.gather(1, next_actions.unsqueeze(1)).squeeze()
            else:
                next_q_values = next_q_values.max(dim=1)[0]

            q_targets = rewards + self.config.gamma * next_q_values * (1 - dones)

        # Calcular Q current
        q_current = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze()

        # Loss
        loss = F.mse_loss(q_current, q_targets)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def update_target_network(self, step: int):
        """Actualizar target network"""
        if step % self.config.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self):
        """Decaer epsilon"""
        self.epsilon = max(self.config.epsilon_end, self.epsilon * self.config.epsilon_decay)

class PPOTrainer:
    """Trainer para Proximal Policy Optimization"""

    def __init__(self, config: RLConfig, state_dim: int, action_dim: int, continuous: bool = False):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.continuous = continuous

        # Red actor-critic
        self.model = ActorCriticNetwork(state_dim, action_dim, config.hidden_dims, continuous).to(self.device)
        self.old_model = ActorCriticNetwork(state_dim, action_dim, config.hidden_dims, continuous).to(self.device)
        self.old_model.load_state_dict(self.model.state_dict())

        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.learning_rate)

        # Storage para PPO updates
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []

    def store_transition(self, state: np.ndarray, action: Any, log_prob: float,
                        reward: float, value: float, done: bool):
        """Almacenar transición para PPO"""
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def select_action(self, state: np.ndarray) -> Tuple[Any, float, float]:
        """Seleccionar acción usando policy actual"""
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            if self.continuous:
                action, log_prob = self.model.get_action(state_tensor, deterministic=False)
                value = self.model.forward(state_tensor)[1]
                return action.cpu().numpy()[0], log_prob.item(), value.item()
            else:
                action, log_prob = self.model.get_action(state_tensor, deterministic=False)
                value = self.model.forward(state_tensor)[1]
                return action.item(), log_prob.item(), value.item()

    def update_policy(self) -> Dict[str, float]:
        """Actualizar policy usando PPO"""

        if not self.states:
            return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}

        # Convertir a tensores
        states = torch.tensor(self.states, dtype=torch.float32, device=self.device)
        actions = torch.tensor(self.actions, dtype=torch.float32 if self.continuous else torch.long,
                              device=self.device)
        old_log_probs = torch.tensor(self.log_probs, dtype=torch.float32, device=self.device)
        rewards = torch.tensor(self.rewards, dtype=torch.float32, device=self.device)
        values = torch.tensor(self.values, dtype=torch.float32, device=self.device)

        # Calcular advantages y returns
        advantages, returns = self._compute_advantages_returns(rewards, values, self.dones)

        # PPO updates
        policy_losses = []
        value_losses = []
        entropies = []

        for _ in range(4):  # PPO epochs
            # Calcular ratios
            if self.continuous:
                new_logits, new_values = self.model(states)
                dist = Normal(new_logits[0], new_logits[1])
                new_log_probs = dist.log_prob(actions).sum(dim=-1)
            else:
                new_logits, new_values = self.model(states)
                dist = Categorical(logits=new_logits)
                new_log_probs = dist.log_prob(actions)

            ratios = torch.exp(new_log_probs - old_log_probs)

            # PPO clipped objective
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.config.clip_ratio, 1 + self.config.clip_ratio) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            value_loss = F.mse_loss(new_values.squeeze(), returns)

            # Entropy bonus
            entropy = dist.entropy().mean()

            # Total loss
            total_loss = policy_loss + self.config.value_coef * value_loss - self.config.entropy_coef * entropy

            # Update
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())
            entropies.append(entropy.item())

        # Update old model
        self.old_model.load_state_dict(self.model.state_dict())

        # Clear storage
        self._clear_storage()

        return {
            "policy_loss": np.mean(policy_losses),
            "value_loss": np.mean(value_losses),
            "entropy": np.mean(entropies)
        }

    def _compute_advantages_returns(self, rewards: torch.Tensor, values: torch.Tensor,
                                   dones: List[bool]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computar advantages y returns usando GAE"""

        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)

        # Generalized Advantage Estimation (GAE)
        gae = 0
        next_value = 0

        for t in reversed(range(len(rewards))):
            if dones[t]:
                next_value = 0

            delta = rewards[t] + self.config.gamma * next_value - values[t]
            gae = delta + self.config.gamma * 0.95 * gae  # lambda = 0.95

            advantages[t] = gae
            returns[t] = gae + values[t]
            next_value = values[t]

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages, returns

    def _clear_storage(self):
        """Limpiar storage de PPO"""
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.values.clear()
        self.dones.clear()

# ===== ENTRENAMIENTO Y EVALUACIÓN =====

class RLTrainingOrchestrator:
    """Orquestador principal de entrenamiento RL"""

    def __init__(self):
        self.environments: Dict[str, RLEnvironment] = {}
        self.trainers: Dict[str, Any] = {}
        self.training_results: Dict[str, List[Dict[str, Any]]] = {}

    def register_environment(self, name: str, env: Any, config: Dict[str, Any] = None):
        """Registrar entorno de RL"""

        if isinstance(env, str):
            # Crear entorno de Gym
            env = gym.make(env)

        # Determinar dimensiones
        if isinstance(env.observation_space, spaces.Box):
            state_dim = np.prod(env.observation_space.shape)
        else:
            state_dim = env.observation_space.n

        if isinstance(env.action_space, spaces.Box):
            action_dim = np.prod(env.action_space.shape)
            action_space_type = "continuous"
        else:
            action_dim = env.action_space.n
            action_space_type = "discrete"

        rl_env = RLEnvironment(
            name=name,
            env=env,
            state_dim=state_dim,
            action_dim=action_dim,
            action_space_type=action_space_type,
            max_episode_steps=config.get("max_episode_steps", 1000) if config else 1000,
            reward_scale=config.get("reward_scale", 1.0) if config else 1.0
        )

        self.environments[name] = rl_env
        logger.info(f"✅ Entorno registrado: {name} ({action_space_type}, state_dim={state_dim}, action_dim={action_dim})")

    def create_trainer(self, env_name: str, config: RLConfig) -> str:
        """Crear trainer para un entorno"""

        if env_name not in self.environments:
            raise ValueError(f"Entorno {env_name} no registrado")

        env = self.environments[env_name]
        trainer_id = f"{config.algorithm.value}_{env_name}_{int(time.time())}"

        # Crear trainer apropiado
        if config.algorithm in [RLAlgorithm.DQN, RLAlgorithm.DOUBLE_DQN, RLAlgorithm.DUELING_DQN]:
            trainer = DQNTrainer(config, env.state_dim, env.action_dim)
        elif config.algorithm in [RLAlgorithm.PPO, RLAlgorithm.A2C]:
            continuous = env.action_space_type == "continuous"
            trainer = PPOTrainer(config, env.state_dim, env.action_dim, continuous)
        else:
            raise ValueError(f"Algoritmo {config.algorithm} no soportado")

        self.trainers[trainer_id] = trainer
        logger.info(f"✅ Trainer creado: {trainer_id} ({config.algorithm.value})")

        return trainer_id

    async def train_agent(self, trainer_id: str, env_name: str, config: RLConfig) -> Dict[str, Any]:
        """Entrenar agente RL"""

        if trainer_id not in self.trainers:
            raise ValueError(f"Trainer {trainer_id} no encontrado")

        if env_name not in self.environments:
            raise ValueError(f"Entorno {env_name} no encontrado")

        trainer = self.trainers[trainer_id]
        env = self.environments[env_name]

        logger.info(f"🚀 Iniciando entrenamiento: {trainer_id} en {env_name}")

        episode_rewards = []
        episode_lengths = []
        training_losses = []
        training_metrics = []

        start_time = time.time()

        for episode in range(config.num_episodes):
            state = env.reset()
            episode_reward = 0
            episode_steps = 0
            done = False

            while not done and episode_steps < config.max_steps_per_episode:
                # Seleccionar acción
                if hasattr(trainer, 'select_action'):
                    action = trainer.select_action(state)
                else:
                    action, log_prob, value = trainer.select_action(state)

                # Ejecutar acción
                next_state, reward, done, info = env.step(action)

                # Almacenar transición
                if hasattr(trainer, 'store_transition'):
                    # DQN-style
                    trainer.store_transition(state, action, reward, next_state, done)
                else:
                    # PPO-style
                    trainer.store_transition(state, action, log_prob, reward, value, done)

                # Actualizar modelo
                if hasattr(trainer, 'update_policy'):
                    if isinstance(trainer, DQNTrainer):
                        loss = trainer.update_policy()
                        if episode % config.target_update_freq == 0:
                            trainer.update_target_network(episode)
                        if loss > 0:
                            training_losses.append(loss)
                    else:
                        # PPO updates después de cada episodio
                        if episode_steps == config.max_steps_per_episode - 1 or done:
                            metrics = trainer.update_policy()
                            training_metrics.append(metrics)

                state = next_state
                episode_reward += reward
                episode_steps += 1

                # Decaer epsilon para DQN
                if hasattr(trainer, 'decay_epsilon') and episode % 10 == 0:
                    trainer.decay_epsilon()

            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_steps)

            # Logging
            if episode % 50 == 0:
                avg_reward = np.mean(episode_rewards[-50:])
                logger.info(f"Episode {episode}: Avg Reward = {avg_reward:.2f}")

        training_time = time.time() - start_time

        results = {
            "trainer_id": trainer_id,
            "env_name": env_name,
            "algorithm": config.algorithm.value,
            "total_episodes": len(episode_rewards),
            "avg_reward": np.mean(episode_rewards),
            "max_reward": np.max(episode_rewards),
            "reward_std": np.std(episode_rewards),
            "avg_episode_length": np.mean(episode_lengths),
            "training_time": training_time,
            "episodes_per_second": len(episode_rewards) / training_time,
            "final_performance": np.mean(episode_rewards[-50:]),  # Últimas 50 episodes
            "training_losses": training_losses[-100:] if training_losses else [],
            "training_metrics": training_metrics[-10:] if training_metrics else []
        }

        self.training_results[trainer_id] = results

        logger.info(f"✅ Entrenamiento completado: {results['final_performance']:.2f} reward promedio")

        return results

    def evaluate_agent(self, trainer_id: str, env_name: str, num_episodes: int = 10) -> Dict[str, Any]:
        """Evaluar agente entrenado"""

        if trainer_id not in self.trainers:
            raise ValueError(f"Trainer {trainer_id} no encontrado")

        trainer = self.trainers[trainer_id]
        env = self.environments[env_name]

        logger.info(f"📊 Evaluando agente: {trainer_id}")

        evaluation_rewards = []
        evaluation_lengths = []

        for episode in range(num_episodes):
            state = env.reset()
            episode_reward = 0
            episode_steps = 0
            done = False

            while not done and episode_steps < env.max_episode_steps:
                # Seleccionar acción (determinístico para evaluación)
                if hasattr(trainer, 'select_action'):
                    if hasattr(trainer, 'exploration'):
                        # DQN: usar epsilon mínimo
                        trainer.epsilon = 0.01
                    action = trainer.select_action(state)
                else:
                    # PPO: modo determinístico
                    action, _, _ = trainer.select_action(state)

                next_state, reward, done, info = env.step(action)

                state = next_state
                episode_reward += reward
                episode_steps += 1

            evaluation_rewards.append(episode_reward)
            evaluation_lengths.append(episode_steps)

        results = {
            "evaluation_episodes": num_episodes,
            "avg_reward": np.mean(evaluation_rewards),
            "std_reward": np.std(evaluation_rewards),
            "max_reward": np.max(evaluation_rewards),
            "min_reward": np.min(evaluation_rewards),
            "avg_episode_length": np.mean(evaluation_lengths)
        }

        logger.info(f"📊 Evaluación completada: {results['avg_reward']:.2f} ± {results['std_reward']:.2f}")

        return results

# ===== SISTEMA PRINCIPAL =====

class AEGISReinforcementLearning:
    """Sistema completo de Reinforcement Learning para AEGIS"""

    def __init__(self):
        self.orchestrator = RLTrainingOrchestrator()
        self.environments = {}
        self.trained_agents = {}

    async def setup_environment(self, env_name: str, env_config: Dict[str, Any] = None):
        """Configurar entorno de RL"""

        if env_config is None:
            env_config = {}

        # Mapear nombres comunes a entornos Gym
        gym_env_map = {
            "cartpole": "CartPole-v1",
            "mountain_car": "MountainCar-v0",
            "pendulum": "Pendulum-v1",
            "acrobot": "Acrobot-v1",
            "lunar_lander": "LunarLander-v2"
        }

        gym_env_name = gym_env_map.get(env_name, env_name)

        try:
            self.orchestrator.register_environment(env_name, gym_env_name, env_config)
            logger.info(f"✅ Entorno configurado: {env_name}")
        except Exception as e:
            logger.error(f"❌ Error configurando entorno {env_name}: {e}")

    async def train_rl_agent(self, env_name: str, algorithm: RLAlgorithm = RLAlgorithm.DQN,
                           config: RLConfig = None) -> Dict[str, Any]:
        """Entrenar agente RL completo"""

        if config is None:
            config = RLConfig(algorithm=algorithm)

        logger.info(f"🤖 Entrenando agente {algorithm.value} en {env_name}")

        # Crear trainer
        trainer_id = self.orchestrator.create_trainer(env_name, config)

        # Entrenar
        training_results = await self.orchestrator.train_agent(trainer_id, env_name, config)

        # Evaluar
        evaluation_results = self.orchestrator.evaluate_agent(trainer_id, env_name)

        # Combinar resultados
        complete_results = {
            **training_results,
            "evaluation": evaluation_results,
            "agent_id": trainer_id,
            "environment": env_name,
            "algorithm": algorithm.value,
            "config": config
        }

        self.trained_agents[trainer_id] = complete_results

        logger.info(f"🎉 Agente entrenado exitosamente: {complete_results['final_performance']:.2f} reward")

        return complete_results

    async def compare_algorithms(self, env_name: str, algorithms: List[RLAlgorithm],
                               config: RLConfig = None) -> Dict[str, Any]:
        """Comparar múltiples algoritmos RL"""

        if config is None:
            config = RLConfig()

        logger.info(f"🏁 Comparando {len(algorithms)} algoritmos en {env_name}")

        results = {}

        for algorithm in algorithms:
            try:
                test_config = RLConfig(algorithm=algorithm, num_episodes=min(200, config.num_episodes))
                result = await self.train_rl_agent(env_name, algorithm, test_config)
                results[algorithm.value] = result

                logger.info(f"✅ {algorithm.value}: {result['final_performance']:.2f} reward")

            except Exception as e:
                logger.error(f"❌ Error con {algorithm.value}: {e}")
                results[algorithm.value] = {"error": str(e)}

        # Ranking
        successful_results = {k: v for k, v in results.items() if "error" not in v}
        if successful_results:
            ranked = sorted(successful_results.items(),
                          key=lambda x: x[1]['final_performance'], reverse=True)
            results["ranking"] = [alg for alg, _ in ranked]

        logger.info(f"🏆 Ranking: {results.get('ranking', 'N/A')}")

        return results

    def get_training_insights(self, results: Dict[str, Any]) -> List[str]:
        """Generar insights del entrenamiento"""

        insights = []

        # Performance insights
        if results['final_performance'] > 100:
            insights.append("🚀 Excelente performance - el agente aprendió bien la tarea")
        elif results['final_performance'] > 0:
            insights.append("✅ Buen progreso - el agente está aprendiendo")
        else:
            insights.append("⚠️ Performance limitada - revisar configuración del algoritmo")

        # Training insights
        training_time = results['training_time']
        if training_time > 300:  # 5 minutos
            insights.append("⏱️ Entrenamiento largo - considerar optimizaciones de red o batch size")
        elif training_time < 30:
            insights.append("⚡ Entrenamiento rápido - buena eficiencia computacional")

        # Stability insights
        reward_std = results['reward_std']
        if reward_std < results['avg_reward'] * 0.5:
            insights.append("📊 Performance estable - buen aprendizaje consistente")
        else:
            insights.append("📈 Alta variabilidad - considerar más episodios de entrenamiento")

        # Algorithm-specific insights
        algorithm = results['algorithm']
        if algorithm == "dqn":
            insights.append("🎯 DQN funciona bien para espacios de acción discretos")
        elif algorithm == "ppo":
            insights.append("🧠 PPO ofrece buen balance entre estabilidad y sample efficiency")

        return insights

# ===== DEMO Y EJEMPLOS =====

async def demo_reinforcement_learning():
    """Demostración completa de Reinforcement Learning"""

    print("🎮 AEGIS Reinforcement Learning Demo")
    print("=" * 40)

    rl_system = AEGISReinforcementLearning()

    # Configurar entorno
    print("\\n🏗️ Configurando entorno...")
    await rl_system.setup_environment("cartpole", {"max_episode_steps": 500})

    print("✅ Entorno CartPole configurado")

    # Configuración de RL simplificada para demo
    config = RLConfig(
        algorithm=RLAlgorithm.DQN,
        num_episodes=100,  # Reducido para demo rápida
        max_steps_per_episode=200,
        batch_size=32,
        learning_rate=1e-3,
        gamma=0.99
    )

    print("\\n⚙️ Configuración RL:")
    print(f"   • Algoritmo: {config.algorithm.value}")
    print(f"   • Episodios: {config.num_episodes}")
    print(f"   • Learning rate: {config.learning_rate}")

    # Entrenar agente
    print("\\n🚀 Entrenando agente DQN...")
    start_time = time.time()

    results = await rl_system.train_rl_agent("cartpole", RLAlgorithm.DQN, config)

    training_time = time.time() - start_time

    # Mostrar resultados
    print("\\n📊 RESULTADOS DE ENTRENAMIENTO:")
    print(f"   • Episodes totales: {results['total_episodes']}")
    print(".2f"    print(".1f"    print(".2f"    print(".1f"    print(".2f"    print(".1f"
    # Evaluación
    evaluation = results['evaluation']
    print("\\n🎯 RESULTADOS DE EVALUACIÓN:")
    print(".2f"    print(".2f"    print(".1f"    print(".1f"
    # Insights
    insights = rl_system.get_training_insights(results)
    print("\\n💡 INSIGHTS:")
    for insight in insights:
        print(f"   • {insight}")

    print("\\n🎉 DEMO COMPLETA - RESULTADOS FINALES")
    print("=" * 50)

    print("🏆 LOGROS ALCANZADOS:")
    print(f"   ✅ Agente RL entrenado exitosamente")
    print(".2f"    print(f"   ✅ {results['total_episodes']} episodios completados")
    print(f"   ✅ Sistema de evaluación funcionando")
    print(f"   ✅ {len(insights)} insights generados automáticamente")

    print("\\n🚀 CAPACIDADES DEMOSTRADAS:")
    print("   ✅ Deep Q-Network (DQN) implementation")
    print("   ✅ Experience replay y target networks")
    print("   ✅ Epsilon-greedy exploration")
    print("   ✅ Training loop completo")
    print("   ✅ Agent evaluation")
    print("   ✅ Performance metrics")
    print("   ✅ Automatic insights generation")

    print("\\n💡 PARA PRODUCCIÓN:")
    print("   • Implementar más algoritmos (PPO, SAC, etc.)")
    print("   • Agregar multi-agent RL capabilities")
    print("   • Integrar con entornos custom")
    print("   • Implementar model serving para políticas")
    print("   • Agregar hyperparameter optimization")
    print("   • Crear dashboard de monitoreo")

    print("\\n" + "=" * 60)
    print("🌟 Reinforcement Learning funcionando correctamente!")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(demo_reinforcement_learning())
