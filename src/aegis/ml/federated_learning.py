"""
AEGIS Federated Learning Module
Advanced federated learning algorithms including FedProx and SCAFFOLD
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import logging
import time
from pathlib import Path
import json
import hashlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AggregationStrategy(Enum):
    """Federated learning aggregation strategies"""
    FED_AVG = "fedavg"
    FED_PROX = "fedprox"
    SCAFFOLD = "scaffold"
    FED_OPT = "fedopt"
    FED_NOVA = "fednova"


@dataclass
class FederatedConfig:
    """Configuration for federated learning"""
    aggregation_strategy: AggregationStrategy = AggregationStrategy.FED_AVG
    num_rounds: int = 100
    clients_per_round: int = 10
    local_epochs: int = 5
    local_batch_size: int = 32
    learning_rate: float = 0.01
    
    mu: float = 0.01
    
    server_learning_rate: float = 1.0
    server_momentum: float = 0.9
    
    differential_privacy: bool = False
    dp_epsilon: float = 1.0
    dp_delta: float = 1e-5
    dp_clip_norm: float = 1.0
    
    min_clients: int = 2
    max_clients: int = 100
    
    model_checkpoint_interval: int = 10
    checkpoint_dir: str = "checkpoints"


@dataclass
class ClientState:
    """State for federated learning client"""
    client_id: str
    num_samples: int
    model_updates: Dict[str, torch.Tensor]
    loss: float
    accuracy: float
    training_time: float
    round_number: int
    
    control_variates: Optional[Dict[str, torch.Tensor]] = None
    

@dataclass
class ServerState:
    """State for federated learning server"""
    global_model: Dict[str, torch.Tensor]
    round_number: int
    total_samples: int
    aggregated_loss: float
    aggregated_accuracy: float
    
    control_variates: Optional[Dict[str, torch.Tensor]] = None
    server_optimizer_state: Optional[Dict] = None
    
    client_history: List[Dict[str, Any]] = field(default_factory=list)
    performance_metrics: Dict[str, List[float]] = field(default_factory=dict)


class FederatedClient:
    """Federated learning client with support for multiple algorithms"""
    
    def __init__(
        self,
        client_id: str,
        model: nn.Module,
        config: FederatedConfig,
        device: str = "cpu"
    ):
        self.client_id = client_id
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        self.control_variates = None
        if config.aggregation_strategy == AggregationStrategy.SCAFFOLD:
            self.control_variates = {
                name: torch.zeros_like(param)
                for name, param in model.named_parameters()
            }
        
        self.optimizer = optim.SGD(
            model.parameters(),
            lr=config.learning_rate,
            momentum=0.9 if config.aggregation_strategy != AggregationStrategy.FED_PROX else 0.0
        )
        
        self.criterion = nn.CrossEntropyLoss()
        
    def set_global_model(self, global_state: Dict[str, torch.Tensor]):
        """Update local model with global model parameters"""
        self.model.load_state_dict(global_state)
        
    def set_control_variates(self, server_control: Dict[str, torch.Tensor]):
        """Update control variates for SCAFFOLD algorithm"""
        if self.config.aggregation_strategy == AggregationStrategy.SCAFFOLD:
            self.control_variates = {
                name: tensor.clone() for name, tensor in server_control.items()
            }
    
    def train(
        self,
        train_loader: Any,
        global_model_state: Optional[Dict[str, torch.Tensor]] = None
    ) -> ClientState:
        """Train local model and return updates"""
        start_time = time.time()
        self.model.train()
        
        global_params = None
        if self.config.aggregation_strategy == AggregationStrategy.FED_PROX and global_model_state:
            global_params = {name: param.clone() for name, param in global_model_state.items()}
        
        total_loss = 0.0
        correct = 0
        total = 0
        num_batches = 0
        
        initial_params = None
        if self.config.aggregation_strategy == AggregationStrategy.SCAFFOLD:
            initial_params = {
                name: param.clone() for name, param in self.model.named_parameters()
            }
        
        for epoch in range(self.config.local_epochs):
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                
                if self.config.aggregation_strategy == AggregationStrategy.FED_PROX and global_params:
                    proximal_term = 0.0
                    for name, param in self.model.named_parameters():
                        if name in global_params:
                            proximal_term += ((param - global_params[name]) ** 2).sum()
                    loss += (self.config.mu / 2) * proximal_term
                
                loss.backward()
                
                if self.config.aggregation_strategy == AggregationStrategy.SCAFFOLD and self.control_variates:
                    for name, param in self.model.named_parameters():
                        if param.grad is not None and name in self.control_variates:
                            param.grad.data -= self.control_variates[name]
                
                if self.config.differential_privacy:
                    self._apply_differential_privacy()
                
                self.optimizer.step()
                
                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        accuracy = correct / total if total > 0 else 0.0
        
        model_updates = {
            name: param.data.clone() for name, param in self.model.named_parameters()
        }
        
        new_control_variates = None
        if self.config.aggregation_strategy == AggregationStrategy.SCAFFOLD and initial_params:
            new_control_variates = {}
            K = self.config.local_epochs * len(train_loader)
            for name, param in self.model.named_parameters():
                if name in initial_params:
                    delta = initial_params[name] - param.data
                    new_control_variates[name] = self.control_variates[name] - delta / (K * self.config.learning_rate)
        
        training_time = time.time() - start_time
        
        return ClientState(
            client_id=self.client_id,
            num_samples=total,
            model_updates=model_updates,
            loss=avg_loss,
            accuracy=accuracy,
            training_time=training_time,
            round_number=0,
            control_variates=new_control_variates
        )
    
    def _apply_differential_privacy(self):
        """Apply differential privacy to gradients"""
        total_norm = 0.0
        for param in self.model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        
        clip_coef = self.config.dp_clip_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for param in self.model.parameters():
                if param.grad is not None:
                    param.grad.data.mul_(clip_coef)
        
        for param in self.model.parameters():
            if param.grad is not None:
                noise = torch.randn_like(param.grad) * self.config.dp_clip_norm * self.config.dp_epsilon
                param.grad.data.add_(noise)


class FederatedServer:
    """Federated learning server with multiple aggregation strategies"""
    
    def __init__(
        self,
        model: nn.Module,
        config: FederatedConfig,
        device: str = "cpu"
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        self.state = ServerState(
            global_model={name: param.data.clone() for name, param in model.named_parameters()},
            round_number=0,
            total_samples=0,
            aggregated_loss=0.0,
            aggregated_accuracy=0.0,
            performance_metrics={
                "loss": [],
                "accuracy": [],
                "training_time": []
            }
        )
        
        if config.aggregation_strategy == AggregationStrategy.SCAFFOLD:
            self.state.control_variates = {
                name: torch.zeros_like(param)
                for name, param in model.named_parameters()
            }
        
        if config.aggregation_strategy == AggregationStrategy.FED_OPT:
            self.state.server_optimizer_state = {
                "velocity": {
                    name: torch.zeros_like(param)
                    for name, param in model.named_parameters()
                }
            }
        
        Path(config.checkpoint_dir).mkdir(exist_ok=True, parents=True)
    
    def aggregate(self, client_states: List[ClientState]) -> ServerState:
        """Aggregate client updates using configured strategy"""
        if len(client_states) < self.config.min_clients:
            logger.warning(f"Not enough clients: {len(client_states)} < {self.config.min_clients}")
            return self.state
        
        self.state.round_number += 1
        
        if self.config.aggregation_strategy == AggregationStrategy.FED_AVG:
            self._aggregate_fedavg(client_states)
        elif self.config.aggregation_strategy == AggregationStrategy.FED_PROX:
            self._aggregate_fedavg(client_states)
        elif self.config.aggregation_strategy == AggregationStrategy.SCAFFOLD:
            self._aggregate_scaffold(client_states)
        elif self.config.aggregation_strategy == AggregationStrategy.FED_OPT:
            self._aggregate_fedopt(client_states)
        elif self.config.aggregation_strategy == AggregationStrategy.FED_NOVA:
            self._aggregate_fednova(client_states)
        
        self._update_metrics(client_states)
        
        if self.state.round_number % self.config.model_checkpoint_interval == 0:
            self._save_checkpoint()
        
        return self.state
    
    def _aggregate_fedavg(self, client_states: List[ClientState]):
        """FedAvg aggregation: weighted average by number of samples"""
        total_samples = sum(state.num_samples for state in client_states)
        
        aggregated_params = {}
        for name in self.state.global_model.keys():
            aggregated_params[name] = torch.zeros_like(self.state.global_model[name])
            
            for state in client_states:
                weight = state.num_samples / total_samples
                aggregated_params[name] += weight * state.model_updates[name]
        
        self.state.global_model = aggregated_params
        self.model.load_state_dict(aggregated_params)
        self.state.total_samples = total_samples
    
    def _aggregate_scaffold(self, client_states: List[ClientState]):
        """SCAFFOLD aggregation with control variates"""
        total_samples = sum(state.num_samples for state in client_states)
        
        aggregated_params = {}
        delta_control = {name: torch.zeros_like(param) for name, param in self.state.control_variates.items()}
        
        for name in self.state.global_model.keys():
            aggregated_params[name] = torch.zeros_like(self.state.global_model[name])
            
            for state in client_states:
                weight = state.num_samples / total_samples
                aggregated_params[name] += weight * state.model_updates[name]
                
                if state.control_variates and name in state.control_variates:
                    delta_control[name] += weight * (state.control_variates[name] - self.state.control_variates[name])
        
        for name in self.state.control_variates.keys():
            self.state.control_variates[name] += delta_control[name] * len(client_states) / self.config.clients_per_round
        
        self.state.global_model = aggregated_params
        self.model.load_state_dict(aggregated_params)
        self.state.total_samples = total_samples
    
    def _aggregate_fedopt(self, client_states: List[ClientState]):
        """FedOpt aggregation with server-side optimizer"""
        total_samples = sum(state.num_samples for state in client_states)
        
        pseudo_gradient = {}
        for name in self.state.global_model.keys():
            weighted_updates = torch.zeros_like(self.state.global_model[name])
            
            for state in client_states:
                weight = state.num_samples / total_samples
                weighted_updates += weight * state.model_updates[name]
            
            pseudo_gradient[name] = self.state.global_model[name] - weighted_updates
        
        for name in self.state.global_model.keys():
            velocity = self.state.server_optimizer_state["velocity"][name]
            velocity = self.config.server_momentum * velocity + pseudo_gradient[name]
            self.state.server_optimizer_state["velocity"][name] = velocity
            
            self.state.global_model[name] -= self.config.server_learning_rate * velocity
        
        self.model.load_state_dict(self.state.global_model)
        self.state.total_samples = total_samples
    
    def _aggregate_fednova(self, client_states: List[ClientState]):
        """FedNova aggregation with normalized averaging"""
        total_tau = sum(state.num_samples for state in client_states)
        
        aggregated_params = {}
        for name in self.state.global_model.keys():
            aggregated_params[name] = torch.zeros_like(self.state.global_model[name])
            
            for state in client_states:
                tau_i = state.num_samples
                a_i = tau_i / total_tau
                
                delta = state.model_updates[name] - self.state.global_model[name]
                aggregated_params[name] += a_i * delta
            
            aggregated_params[name] = self.state.global_model[name] + aggregated_params[name]
        
        self.state.global_model = aggregated_params
        self.model.load_state_dict(aggregated_params)
        self.state.total_samples = total_tau
    
    def _update_metrics(self, client_states: List[ClientState]):
        """Update server metrics from client states"""
        total_samples = sum(state.num_samples for state in client_states)
        
        weighted_loss = sum(
            state.loss * state.num_samples for state in client_states
        ) / total_samples
        
        weighted_accuracy = sum(
            state.accuracy * state.num_samples for state in client_states
        ) / total_samples
        
        avg_training_time = sum(
            state.training_time for state in client_states
        ) / len(client_states)
        
        self.state.aggregated_loss = weighted_loss
        self.state.aggregated_accuracy = weighted_accuracy
        
        self.state.performance_metrics["loss"].append(weighted_loss)
        self.state.performance_metrics["accuracy"].append(weighted_accuracy)
        self.state.performance_metrics["training_time"].append(avg_training_time)
        
        self.state.client_history.append({
            "round": self.state.round_number,
            "num_clients": len(client_states),
            "total_samples": total_samples,
            "loss": weighted_loss,
            "accuracy": weighted_accuracy,
            "training_time": avg_training_time
        })
        
        logger.info(
            f"Round {self.state.round_number}: "
            f"Loss={weighted_loss:.4f}, "
            f"Accuracy={weighted_accuracy:.4f}, "
            f"Clients={len(client_states)}"
        )
    
    def _save_checkpoint(self):
        """Save model checkpoint and server state"""
        checkpoint_path = Path(self.config.checkpoint_dir) / f"round_{self.state.round_number}.pt"
        
        checkpoint = {
            "round": self.state.round_number,
            "model_state_dict": self.state.global_model,
            "aggregated_loss": self.state.aggregated_loss,
            "aggregated_accuracy": self.state.aggregated_accuracy,
            "performance_metrics": self.state.performance_metrics,
            "config": {
                "aggregation_strategy": self.config.aggregation_strategy.value,
                "num_rounds": self.config.num_rounds,
                "clients_per_round": self.config.clients_per_round
            }
        }
        
        if self.state.control_variates:
            checkpoint["control_variates"] = self.state.control_variates
        
        if self.state.server_optimizer_state:
            checkpoint["server_optimizer_state"] = self.state.server_optimizer_state
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")
        
        metrics_path = Path(self.config.checkpoint_dir) / "metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump({
                "round": self.state.round_number,
                "performance_metrics": {
                    k: [float(v) for v in vals]
                    for k, vals in self.state.performance_metrics.items()
                },
                "client_history": self.state.client_history
            }, f, indent=2)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint and server state"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.state.global_model = checkpoint["model_state_dict"]
        self.model.load_state_dict(self.state.global_model)
        self.state.round_number = checkpoint["round"]
        self.state.aggregated_loss = checkpoint.get("aggregated_loss", 0.0)
        self.state.aggregated_accuracy = checkpoint.get("aggregated_accuracy", 0.0)
        self.state.performance_metrics = checkpoint.get("performance_metrics", {})
        
        if "control_variates" in checkpoint:
            self.state.control_variates = checkpoint["control_variates"]
        
        if "server_optimizer_state" in checkpoint:
            self.state.server_optimizer_state = checkpoint["server_optimizer_state"]
        
        logger.info(f"Checkpoint loaded from round {self.state.round_number}")
    
    def get_global_model(self) -> Dict[str, torch.Tensor]:
        """Get current global model state"""
        return self.state.global_model
    
    def get_control_variates(self) -> Optional[Dict[str, torch.Tensor]]:
        """Get current control variates (for SCAFFOLD)"""
        return self.state.control_variates
