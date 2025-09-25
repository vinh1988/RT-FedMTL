"""
Flower server implementation for federated BERT learning.
Handles server-side orchestration of federated training.
"""

import torch
from typing import Dict, List, Optional, Tuple, Any
from flwr.server import Server, ClientManager, ServerConfig
from flwr.server.strategy import Strategy, FedAvg
from flwr.common import (
    Parameters, 
    FitRes, 
    EvaluateRes, 
    parameters_to_ndarrays,
    ndarrays_to_parameters
)
import numpy as np
from dataclasses import dataclass
import logging

from ..models.federated_bert import FederatedBERTServer, FederatedBERTConfig
from ..models.knowledge_transfer import (
    AdaptiveKnowledgeTransfer,
    ProgressiveTransferConfig,
    DynamicAlignmentConfig
)
from ..aggregation.fedavg import LoRAFedAvgAggregator, AggregationConfig

logger = logging.getLogger(__name__)


@dataclass
class FlowerServerConfig:
    """Configuration for Flower federated server"""
    num_rounds: int = 50
    min_fit_clients: int = 2
    min_evaluate_clients: int = 2
    min_available_clients: int = 2
    fraction_fit: float = 1.0
    fraction_evaluate: float = 1.0
    
    # Server model configuration
    server_model_config: FederatedBERTConfig = None
    
    # Knowledge transfer configuration
    enable_knowledge_transfer: bool = True
    progressive_config: ProgressiveTransferConfig = None
    alignment_config: DynamicAlignmentConfig = None
    
    # Aggregation configuration
    aggregation_config: AggregationConfig = None
    
    def __post_init__(self):
        if self.server_model_config is None:
            self.server_model_config = FederatedBERTConfig()
        
        if self.progressive_config is None:
            self.progressive_config = ProgressiveTransferConfig()
        
        if self.alignment_config is None:
            self.alignment_config = DynamicAlignmentConfig()
        
        if self.aggregation_config is None:
            self.aggregation_config = AggregationConfig()


class FederatedBERTStrategy(FedAvg):
    """
    Custom Flower strategy for federated BERT learning.
    Implements LoRA parameter aggregation and knowledge transfer.
    """
    
    def __init__(self, config: FlowerServerConfig):
        super().__init__(
            fraction_fit=config.fraction_fit,
            fraction_evaluate=config.fraction_evaluate,
            min_fit_clients=config.min_fit_clients,
            min_evaluate_clients=config.min_evaluate_clients,
            min_available_clients=config.min_available_clients,
        )
        
        self.config = config
        self.current_round = 0
        
        # Initialize server model
        self.server_model = FederatedBERTServer(config.server_model_config)
        self.server_model.eval()
        
        # Initialize aggregator
        self.aggregator = LoRAFedAvgAggregator(config.aggregation_config)
        
        # Initialize knowledge transfer
        if config.enable_knowledge_transfer:
            self.knowledge_transfer = AdaptiveKnowledgeTransfer(
                config.progressive_config,
                config.alignment_config
            )
        else:
            self.knowledge_transfer = None
        
        logger.info("FederatedBERTStrategy initialized")
    
    def initialize_parameters(self, client_manager: ClientManager) -> Optional[Parameters]:
        """Initialize global model parameters"""
        
        # Get server LoRA parameters
        server_params = self.server_model.get_lora_parameters()
        
        # Convert to numpy arrays
        param_arrays = [param.cpu().numpy() for param in server_params.values()]
        param_names = list(server_params.keys())
        
        logger.info(f"Initialized {len(param_arrays)} LoRA parameters")
        logger.info(f"Parameter names: {param_names}")
        
        # Store parameter names for later use
        self.param_names = param_names
        
        return ndarrays_to_parameters(param_arrays)
    
    def aggregate_fit(self, server_round: int, results: List[Tuple[Any, FitRes]], 
                     failures: List[Tuple[Any, Exception]]) -> Tuple[Optional[Parameters], Dict[str, Any]]:
        """Aggregate client parameters using LoRA-aware FedAvg"""
        
        self.current_round = server_round
        
        if not results:
            logger.warning(f"No results to aggregate in round {server_round}")
            return None, {}
        
        logger.info(f"Round {server_round}: Aggregating {len(results)} client results")
        
        # Extract client parameters and info
        client_parameters_list = []
        client_info_list = []
        
        for client_proxy, fit_res in results:
            # Convert parameters back to tensors
            param_arrays = parameters_to_ndarrays(fit_res.parameters)
            client_params = {}
            
            for i, param_name in enumerate(self.param_names):
                if i < len(param_arrays):
                    client_params[param_name] = torch.from_numpy(param_arrays[i])
            
            client_parameters_list.append(client_params)
            
            # Extract client info from metrics
            client_info = {
                "data_size": fit_res.num_examples,
                "loss": fit_res.metrics.get("train_loss", 1.0),
                "accuracy": fit_res.metrics.get("train_accuracy", 0.0)
            }
            client_info_list.append(client_info)
        
        # Aggregate using LoRA aggregator
        aggregated_params = self.aggregator.aggregate(client_parameters_list, client_info_list)
        
        # Update server model with aggregated parameters
        self.server_model.set_lora_parameters(aggregated_params)
        
        # Convert back to Flower parameters
        param_arrays = [param.cpu().numpy() for param in aggregated_params.values()]
        aggregated_parameters = ndarrays_to_parameters(param_arrays)
        
        # Get aggregation statistics
        agg_stats = self.aggregator.get_aggregation_stats(client_parameters_list)
        
        # Compute knowledge transfer weight for next round
        transfer_weight = 0.0
        if self.knowledge_transfer:
            transfer_weight = self.knowledge_transfer.get_transfer_weight(server_round + 1)
        
        metrics = {
            "aggregated_clients": len(results),
            "failed_clients": len(failures),
            "transfer_weight": transfer_weight,
            **agg_stats
        }
        
        logger.info(f"Round {server_round} aggregation completed. Stats: {metrics}")
        
        return aggregated_parameters, metrics
    
    def aggregate_evaluate(self, server_round: int, results: List[Tuple[Any, EvaluateRes]], 
                          failures: List[Tuple[Any, Exception]]) -> Tuple[Optional[float], Dict[str, Any]]:
        """Aggregate evaluation results"""
        
        if not results:
            return None, {}
        
        # Aggregate evaluation metrics
        total_examples = sum([res.num_examples for _, res in results])
        
        # Weighted average of losses and accuracies
        weighted_loss = sum([res.loss * res.num_examples for _, res in results]) / total_examples
        
        accuracies = [res.metrics.get("accuracy", 0.0) * res.num_examples for _, res in results]
        weighted_accuracy = sum(accuracies) / total_examples if accuracies else 0.0
        
        metrics = {
            "eval_loss": weighted_loss,
            "eval_accuracy": weighted_accuracy,
            "num_clients": len(results),
            "total_examples": total_examples,
            "round": server_round
        }
        
        logger.info(f"Round {server_round} evaluation: Loss={weighted_loss:.4f}, Accuracy={weighted_accuracy:.4f}")
        
        return weighted_loss, metrics
    
    def configure_fit(self, server_round: int, parameters: Parameters, client_manager: ClientManager):
        """Configure the next round of training"""
        
        # Get standard configuration
        config = super().configure_fit(server_round, parameters, client_manager)
        
        # Add knowledge transfer configuration
        if self.knowledge_transfer:
            transfer_weight = self.knowledge_transfer.get_transfer_weight(server_round)
            
            # Add transfer configuration to client configs
            for client_proxy, fit_config in config:
                fit_config.config["transfer_weight"] = transfer_weight
                fit_config.config["round"] = server_round
                
                # Add server knowledge if available
                if hasattr(self, 'server_knowledge'):
                    fit_config.config["server_knowledge"] = self.server_knowledge
        
        return config
    
    def get_server_knowledge(self, sample_inputs: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, Any]:
        """Extract server knowledge for client transfer"""
        
        if sample_inputs is None:
            return {}
        
        self.server_model.eval()
        with torch.no_grad():
            server_outputs = self.server_model(
                input_ids=sample_inputs["input_ids"],
                attention_mask=sample_inputs["attention_mask"],
                return_hidden_states=True
            )
        
        return {
            "logits": server_outputs["logits"].cpu(),
            "projected_hidden_states": server_outputs["projected_hidden_states"].cpu()
        }


class FlowerFederatedServer:
    """
    Main federated server using Flower framework.
    Orchestrates the entire federated learning process.
    """
    
    def __init__(self, config: FlowerServerConfig):
        self.config = config
        self.strategy = FederatedBERTStrategy(config)
        
    def start_server(self, server_address: str = "localhost:8080"):
        """Start the federated server"""
        
        from flwr.server import start_server
        
        logger.info(f"Starting federated server on {server_address}")
        logger.info(f"Configuration: {self.config}")
        
        # Configure server
        server_config = ServerConfig(num_rounds=self.config.num_rounds)
        
        # Start server
        start_server(
            server_address=server_address,
            config=server_config,
            strategy=self.strategy,
        )
    
    def simulate_federated_learning(self, client_fn, num_clients: int = 10):
        """Simulate federated learning in a single process"""
        
        from flwr.simulation import start_simulation
        
        logger.info(f"Starting federated simulation with {num_clients} clients")
        
        # Start simulation
        history = start_simulation(
            client_fn=client_fn,
            num_clients=num_clients,
            config=ServerConfig(num_rounds=self.config.num_rounds),
            strategy=self.strategy,
            client_resources={"num_cpus": 1, "num_gpus": 0.0}  # Adjust based on available resources
        )
        
        logger.info("Federated simulation completed")
        return history


def create_flower_server(num_rounds: int = 50,
                        num_clients: int = 10,
                        enable_knowledge_transfer: bool = True) -> FlowerFederatedServer:
    """Factory function to create Flower federated server"""
    
    config = FlowerServerConfig(
        num_rounds=num_rounds,
        min_fit_clients=max(2, num_clients // 2),
        min_evaluate_clients=max(2, num_clients // 2),
        min_available_clients=max(2, num_clients // 2),
        enable_knowledge_transfer=enable_knowledge_transfer
    )
    
    return FlowerFederatedServer(config)


if __name__ == "__main__":
    # Test server creation
    server = create_flower_server(num_rounds=10, num_clients=5)
    print("Flower federated server created successfully")
    print(f"Strategy: {type(server.strategy).__name__}")
    print(f"Configuration: {server.config}")
