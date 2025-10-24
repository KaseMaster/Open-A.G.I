"""
AEGIS Federated Learning Demo
Demonstrates FedAvg, FedProx, and SCAFFOLD algorithms with MNIST-like synthetic data
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.aegis.ml.federated_learning import (
    FederatedClient,
    FederatedServer,
    FederatedConfig,
    AggregationStrategy
)


class MNISTModel(nn.Module):
    """Simple CNN model for image classification"""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def generate_synthetic_data(num_clients=5, samples_per_client=1000):
    """Generate synthetic image data for federated learning"""
    print("Generating synthetic data for federated learning...")
    
    client_data = []
    
    for i in range(num_clients):
        X = torch.randn(samples_per_client, 1, 28, 28)
        y = torch.randint(0, 10, (samples_per_client,))
        
        dataset = TensorDataset(X, y)
        train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        client_data.append({
            'client_id': f'client_{i:03d}',
            'data_loader': train_loader,
            'num_samples': samples_per_client
        })
        
        print(f"  Client {i}: {samples_per_client} samples")
    
    return client_data


def run_federated_training(
    strategy: AggregationStrategy,
    num_rounds: int = 10,
    num_clients: int = 5,
    clients_per_round: int = 3
):
    """Run federated learning with specified strategy"""
    
    print(f"\n{'='*80}")
    print(f"Running Federated Learning: {strategy.value.upper()}")
    print(f"{'='*80}\n")
    
    config = FederatedConfig(
        aggregation_strategy=strategy,
        num_rounds=num_rounds,
        clients_per_round=clients_per_round,
        local_epochs=3,
        local_batch_size=32,
        learning_rate=0.01,
        mu=0.01 if strategy == AggregationStrategy.FED_PROX else 0.0,
        model_checkpoint_interval=5
    )
    
    print(f"Configuration:")
    print(f"  Aggregation: {config.aggregation_strategy.value}")
    print(f"  Rounds: {config.num_rounds}")
    print(f"  Clients per round: {config.clients_per_round}")
    print(f"  Local epochs: {config.local_epochs}")
    print(f"  Learning rate: {config.learning_rate}")
    if strategy == AggregationStrategy.FED_PROX:
        print(f"  Proximal term (mu): {config.mu}")
    print()
    
    global_model = MNISTModel()
    
    server = FederatedServer(
        model=global_model,
        config=config,
        device="cpu"
    )
    
    client_data = generate_synthetic_data(
        num_clients=num_clients,
        samples_per_client=1000
    )
    
    print(f"\nStarting federated training with {num_clients} clients...")
    print(f"{'='*80}\n")
    
    for round_num in range(config.num_rounds):
        print(f"Round {round_num + 1}/{config.num_rounds}")
        print(f"{'-'*80}")
        
        import random
        selected_clients = random.sample(client_data, config.clients_per_round)
        
        client_states = []
        
        for client_info in selected_clients:
            client = FederatedClient(
                client_id=client_info['client_id'],
                model=MNISTModel(),
                config=config,
                device="cpu"
            )
            
            client.set_global_model(server.get_global_model())
            
            if strategy == AggregationStrategy.SCAFFOLD and server.state.control_variates:
                client.set_control_variates(server.get_control_variates())
            
            state = client.train(
                train_loader=client_info['data_loader'],
                global_model_state=server.get_global_model() if strategy == AggregationStrategy.FED_PROX else None
            )
            
            client_states.append(state)
            
            print(f"  {client_info['client_id']}: "
                  f"Loss={state.loss:.4f}, "
                  f"Accuracy={state.accuracy:.4f}, "
                  f"Time={state.training_time:.2f}s")
        
        server_state = server.aggregate(client_states)
        
        print(f"\n  [AGGREGATED] "
              f"Loss={server_state.aggregated_loss:.4f}, "
              f"Accuracy={server_state.aggregated_accuracy:.4f}")
        print(f"{'-'*80}\n")
    
    print(f"\n{'='*80}")
    print(f"Training Complete!")
    print(f"{'='*80}\n")
    
    print("Final Performance Metrics:")
    print(f"  Final Loss: {server.state.aggregated_loss:.4f}")
    print(f"  Final Accuracy: {server.state.aggregated_accuracy:.4f}")
    print(f"  Total Rounds: {server.state.round_number}")
    print(f"  Total Samples: {server.state.total_samples}")
    
    if len(server.state.performance_metrics.get("loss", [])) > 0:
        losses = server.state.performance_metrics["loss"]
        accuracies = server.state.performance_metrics["accuracy"]
        
        print(f"\n  Loss trend: {losses[0]:.4f} → {losses[-1]:.4f} "
              f"(Δ {losses[-1] - losses[0]:+.4f})")
        print(f"  Accuracy trend: {accuracies[0]:.4f} → {accuracies[-1]:.4f} "
              f"(Δ {accuracies[-1] - accuracies[0]:+.4f})")
    
    return server


def compare_strategies():
    """Compare different federated learning strategies"""
    
    print("\n" + "="*80)
    print("AEGIS Federated Learning - Strategy Comparison")
    print("="*80)
    
    strategies = [
        AggregationStrategy.FED_AVG,
        AggregationStrategy.FED_PROX,
        AggregationStrategy.SCAFFOLD
    ]
    
    results = {}
    
    for strategy in strategies:
        server = run_federated_training(
            strategy=strategy,
            num_rounds=5,
            num_clients=5,
            clients_per_round=3
        )
        
        results[strategy.value] = {
            'final_loss': server.state.aggregated_loss,
            'final_accuracy': server.state.aggregated_accuracy,
            'loss_history': server.state.performance_metrics.get("loss", []),
            'accuracy_history': server.state.performance_metrics.get("accuracy", [])
        }
    
    print(f"\n\n{'='*80}")
    print("COMPARISON SUMMARY")
    print(f"{'='*80}\n")
    
    print(f"{'Strategy':<20} {'Final Loss':<15} {'Final Accuracy':<15} {'Improvement':<15}")
    print(f"{'-'*80}")
    
    for strategy_name, metrics in results.items():
        loss_history = metrics['loss_history']
        accuracy_history = metrics['accuracy_history']
        
        if len(loss_history) > 1 and len(accuracy_history) > 1:
            loss_improvement = loss_history[0] - loss_history[-1]
            accuracy_improvement = accuracy_history[-1] - accuracy_history[0]
            improvement_str = f"↓{loss_improvement:.4f} / ↑{accuracy_improvement:.4f}"
        else:
            improvement_str = "N/A"
        
        print(f"{strategy_name.upper():<20} "
              f"{metrics['final_loss']:<15.4f} "
              f"{metrics['final_accuracy']:<15.4f} "
              f"{improvement_str:<15}")
    
    print(f"{'-'*80}\n")
    
    best_strategy = min(results.items(), key=lambda x: x[1]['final_loss'])
    print(f"Best Strategy (by loss): {best_strategy[0].upper()}")
    
    best_strategy_acc = max(results.items(), key=lambda x: x[1]['final_accuracy'])
    print(f"Best Strategy (by accuracy): {best_strategy_acc[0].upper()}\n")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="AEGIS Federated Learning Demo")
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["fedavg", "fedprox", "scaffold", "fedopt", "fednova", "compare"],
        default="fedavg",
        help="Aggregation strategy to use (default: fedavg, use 'compare' to compare all)"
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=10,
        help="Number of training rounds (default: 10)"
    )
    parser.add_argument(
        "--clients",
        type=int,
        default=5,
        help="Total number of clients (default: 5)"
    )
    parser.add_argument(
        "--clients-per-round",
        type=int,
        default=3,
        help="Clients per round (default: 3)"
    )
    
    args = parser.parse_args()
    
    if args.strategy == "compare":
        compare_strategies()
    else:
        strategy = AggregationStrategy(args.strategy)
        run_federated_training(
            strategy=strategy,
            num_rounds=args.rounds,
            num_clients=args.clients,
            clients_per_round=args.clients_per_round
        )


if __name__ == "__main__":
    main()
