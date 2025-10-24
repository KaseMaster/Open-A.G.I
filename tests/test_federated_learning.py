import pytest
import sys
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.aegis.ml.federated_learning import (
    FederatedClient,
    FederatedServer,
    FederatedConfig,
    AggregationStrategy,
    ClientState,
    ServerState
)


class SimpleModel(nn.Module):
    """Simple neural network for testing"""
    def __init__(self, input_dim=10, hidden_dim=20, output_dim=2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


@pytest.fixture
def simple_model():
    return SimpleModel()


@pytest.fixture
def federated_config():
    return FederatedConfig(
        aggregation_strategy=AggregationStrategy.FED_AVG,
        num_rounds=5,
        clients_per_round=3,
        local_epochs=2,
        local_batch_size=16,
        learning_rate=0.01
    )


@pytest.fixture
def dummy_dataloader():
    """Create dummy data loader for testing"""
    X = torch.randn(100, 10)
    y = torch.randint(0, 2, (100,))
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=16, shuffle=True)


class TestFederatedLearning:
    
    def test_federated_config_initialization(self, federated_config):
        assert federated_config.aggregation_strategy == AggregationStrategy.FED_AVG
        assert federated_config.num_rounds == 5
        assert federated_config.clients_per_round == 3
        assert federated_config.local_epochs == 2
    
    def test_aggregation_strategy_enum(self):
        assert AggregationStrategy.FED_AVG.value == "fedavg"
        assert AggregationStrategy.FED_PROX.value == "fedprox"
        assert AggregationStrategy.SCAFFOLD.value == "scaffold"
        assert AggregationStrategy.FED_OPT.value == "fedopt"
        assert AggregationStrategy.FED_NOVA.value == "fednova"
    
    def test_federated_client_initialization(self, simple_model, federated_config):
        client = FederatedClient(
            client_id="client_001",
            model=simple_model,
            config=federated_config,
            device="cpu"
        )
        
        assert client.client_id == "client_001"
        assert client.device == "cpu"
        assert isinstance(client.model, nn.Module)
        assert client.optimizer is not None
    
    def test_federated_server_initialization(self, simple_model, federated_config):
        server = FederatedServer(
            model=simple_model,
            config=federated_config,
            device="cpu"
        )
        
        assert server.state.round_number == 0
        assert server.state.total_samples == 0
        assert len(server.state.global_model) > 0
    
    def test_client_training(self, simple_model, federated_config, dummy_dataloader):
        client = FederatedClient(
            client_id="client_001",
            model=simple_model,
            config=federated_config,
            device="cpu"
        )
        
        client_state = client.train(dummy_dataloader)
        
        assert isinstance(client_state, ClientState)
        assert client_state.client_id == "client_001"
        assert client_state.num_samples > 0
        assert client_state.loss >= 0
        assert 0 <= client_state.accuracy <= 1
        assert client_state.training_time > 0
        assert len(client_state.model_updates) > 0
    
    def test_fedavg_aggregation(self, simple_model, federated_config, dummy_dataloader):
        server = FederatedServer(
            model=simple_model,
            config=federated_config,
            device="cpu"
        )
        
        client_states = []
        for i in range(3):
            client = FederatedClient(
                client_id=f"client_{i:03d}",
                model=SimpleModel(),
                config=federated_config,
                device="cpu"
            )
            client.set_global_model(server.get_global_model())
            state = client.train(dummy_dataloader)
            client_states.append(state)
        
        server_state = server.aggregate(client_states)
        
        assert server_state.round_number == 1
        assert server_state.total_samples > 0
        assert server_state.aggregated_loss >= 0
        assert 0 <= server_state.aggregated_accuracy <= 1
        assert len(server_state.client_history) == 1
    
    def test_fedprox_initialization(self, simple_model):
        config = FederatedConfig(
            aggregation_strategy=AggregationStrategy.FED_PROX,
            mu=0.01
        )
        
        client = FederatedClient(
            client_id="client_001",
            model=simple_model,
            config=config,
            device="cpu"
        )
        
        assert config.mu == 0.01
        assert isinstance(client, FederatedClient)
    
    def test_scaffold_initialization(self, simple_model):
        config = FederatedConfig(
            aggregation_strategy=AggregationStrategy.SCAFFOLD
        )
        
        client = FederatedClient(
            client_id="client_001",
            model=simple_model,
            config=config,
            device="cpu"
        )
        
        assert client.control_variates is not None
        assert len(client.control_variates) > 0
    
    def test_scaffold_aggregation(self, simple_model, dummy_dataloader):
        config = FederatedConfig(
            aggregation_strategy=AggregationStrategy.SCAFFOLD,
            local_epochs=2
        )
        
        server = FederatedServer(
            model=simple_model,
            config=config,
            device="cpu"
        )
        
        assert server.state.control_variates is not None
        
        client_states = []
        for i in range(2):
            client = FederatedClient(
                client_id=f"client_{i:03d}",
                model=SimpleModel(),
                config=config,
                device="cpu"
            )
            client.set_global_model(server.get_global_model())
            if server.state.control_variates:
                client.set_control_variates(server.get_control_variates())
            
            state = client.train(dummy_dataloader)
            client_states.append(state)
        
        server_state = server.aggregate(client_states)
        
        assert server_state.round_number == 1
        assert server_state.control_variates is not None
    
    def test_fedopt_aggregation(self, simple_model, dummy_dataloader):
        config = FederatedConfig(
            aggregation_strategy=AggregationStrategy.FED_OPT,
            server_learning_rate=0.5,
            server_momentum=0.9
        )
        
        server = FederatedServer(
            model=simple_model,
            config=config,
            device="cpu"
        )
        
        assert server.state.server_optimizer_state is not None
        assert "velocity" in server.state.server_optimizer_state
        
        client_states = []
        for i in range(2):
            client = FederatedClient(
                client_id=f"client_{i:03d}",
                model=SimpleModel(),
                config=config,
                device="cpu"
            )
            client.set_global_model(server.get_global_model())
            state = client.train(dummy_dataloader)
            client_states.append(state)
        
        server_state = server.aggregate(client_states)
        
        assert server_state.round_number == 1
    
    def test_fednova_aggregation(self, simple_model, dummy_dataloader):
        config = FederatedConfig(
            aggregation_strategy=AggregationStrategy.FED_NOVA
        )
        
        server = FederatedServer(
            model=simple_model,
            config=config,
            device="cpu"
        )
        
        client_states = []
        for i in range(2):
            client = FederatedClient(
                client_id=f"client_{i:03d}",
                model=SimpleModel(),
                config=config,
                device="cpu"
            )
            client.set_global_model(server.get_global_model())
            state = client.train(dummy_dataloader)
            client_states.append(state)
        
        server_state = server.aggregate(client_states)
        
        assert server_state.round_number == 1
        assert server_state.total_samples > 0
    
    def test_multiple_rounds(self, simple_model, federated_config, dummy_dataloader):
        server = FederatedServer(
            model=simple_model,
            config=federated_config,
            device="cpu"
        )
        
        for round_num in range(3):
            client_states = []
            for i in range(2):
                client = FederatedClient(
                    client_id=f"client_{i:03d}",
                    model=SimpleModel(),
                    config=federated_config,
                    device="cpu"
                )
                client.set_global_model(server.get_global_model())
                state = client.train(dummy_dataloader)
                client_states.append(state)
            
            server.aggregate(client_states)
        
        assert server.state.round_number == 3
        assert len(server.state.client_history) == 3
        assert len(server.state.performance_metrics["loss"]) == 3
    
    def test_checkpoint_saving(self, simple_model, federated_config, dummy_dataloader, tmp_path):
        config = FederatedConfig(
            aggregation_strategy=AggregationStrategy.FED_AVG,
            model_checkpoint_interval=1,
            checkpoint_dir=str(tmp_path)
        )
        
        server = FederatedServer(
            model=simple_model,
            config=config,
            device="cpu"
        )
        
        client_states = []
        for i in range(2):
            client = FederatedClient(
                client_id=f"client_{i:03d}",
                model=SimpleModel(),
                config=config,
                device="cpu"
            )
            client.set_global_model(server.get_global_model())
            state = client.train(dummy_dataloader)
            client_states.append(state)
        
        server.aggregate(client_states)
        
        checkpoint_file = tmp_path / "round_1.pt"
        assert checkpoint_file.exists()
        
        metrics_file = tmp_path / "metrics.json"
        assert metrics_file.exists()
    
    def test_checkpoint_loading(self, simple_model, dummy_dataloader, tmp_path):
        config = FederatedConfig(
            aggregation_strategy=AggregationStrategy.FED_AVG,
            model_checkpoint_interval=1,
            checkpoint_dir=str(tmp_path)
        )
        
        server1 = FederatedServer(
            model=simple_model,
            config=config,
            device="cpu"
        )
        
        client_states = []
        for i in range(2):
            client = FederatedClient(
                client_id=f"client_{i:03d}",
                model=SimpleModel(),
                config=config,
                device="cpu"
            )
            client.set_global_model(server1.get_global_model())
            state = client.train(dummy_dataloader)
            client_states.append(state)
        
        server1.aggregate(client_states)
        checkpoint_path = str(tmp_path / "round_1.pt")
        
        server2 = FederatedServer(
            model=SimpleModel(),
            config=config,
            device="cpu"
        )
        server2.load_checkpoint(checkpoint_path)
        
        assert server2.state.round_number == 1
        assert server2.state.aggregated_loss >= 0
    
    def test_performance_metrics_tracking(self, simple_model, federated_config, dummy_dataloader):
        server = FederatedServer(
            model=simple_model,
            config=federated_config,
            device="cpu"
        )
        
        for _ in range(3):
            client_states = []
            for i in range(2):
                client = FederatedClient(
                    client_id=f"client_{i:03d}",
                    model=SimpleModel(),
                    config=federated_config,
                    device="cpu"
                )
                client.set_global_model(server.get_global_model())
                state = client.train(dummy_dataloader)
                client_states.append(state)
            
            server.aggregate(client_states)
        
        assert "loss" in server.state.performance_metrics
        assert "accuracy" in server.state.performance_metrics
        assert "training_time" in server.state.performance_metrics
        assert len(server.state.performance_metrics["loss"]) == 3
    
    def test_minimum_clients_requirement(self, simple_model):
        config = FederatedConfig(
            min_clients=5
        )
        
        server = FederatedServer(
            model=simple_model,
            config=config,
            device="cpu"
        )
        
        client_states = [
            ClientState(
                client_id=f"client_{i:03d}",
                num_samples=100,
                model_updates={},
                loss=1.0,
                accuracy=0.5,
                training_time=1.0,
                round_number=0
            )
            for i in range(3)
        ]
        
        server_state = server.aggregate(client_states)
        
        assert server_state.round_number == 0
