#!/usr/bin/env python3
"""
Main script for running federated BERT learning with LoRA.
Single-terminal simulation using Flower framework.
"""

import argparse
import logging
from typing import Dict, Any
import hydra
from omegaconf import DictConfig, OmegaConf

import sys
import os

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.server.flower_server import create_flower_server, FlowerServerConfig
from src.clients.flower_client import client_fn
from src.models.federated_bert import FederatedBERTConfig
from src.models.knowledge_transfer import ProgressiveTransferConfig, DynamicAlignmentConfig
from src.aggregation.fedavg import AggregationConfig
from src.utils.training_utils import setup_logging, set_seed, get_device
from src.utils.data_utils import create_federated_datasets

logger = logging.getLogger(__name__)


def create_client_fn(config: DictConfig):
    """Create client function with proper data partitioning"""
    
    # Load and partition federated datasets
    train_partitions, test_partitions, num_labels = create_federated_datasets(
        task_name=config.data.task_name,
        num_clients=config.federated.num_clients,
        partition_strategy="iid",  # Can be made configurable
        test_split=0.2
    )
    
    # Update model config with correct number of labels
    config.server_model.num_labels = num_labels
    config.client_model.num_labels = num_labels
    
    def client_fn_with_data(cid: str):
        """Client function with actual data"""
        from src.clients.flower_client import FlowerFederatedClient, FlowerClientConfig
        from src.models.federated_bert import FederatedBERTConfig
        
        client_id = int(cid)
        
        # Get client data
        train_data = train_partitions[client_id % len(train_partitions)]
        test_data = test_partitions[client_id % len(test_partitions)]
        
        # Create client config
        client_config = FlowerClientConfig(
            client_id=client_id,
            local_epochs=config.training.local_epochs,
            batch_size=config.data.train_batch_size,
            learning_rate=config.training.learning_rate,
            client_model_config=FederatedBERTConfig(
                server_model_name=config.server_model.name,
                client_model_name=config.client_model.name,
                num_labels=num_labels,
                lora_r=config.lora.r,
                lora_alpha=config.lora.alpha,
                lora_dropout=config.lora.dropout,
                lora_target_modules=config.lora.target_modules
            ),
            dataset_name=config.data.dataset_name,
            task_name=config.data.task_name,
            max_length=config.data.max_length,
            device=str(config.experiment.device),
            enable_knowledge_transfer=config.knowledge_transfer.progressive_transfer.enabled
        )
        
        return FlowerFederatedClient(client_config, train_data, test_data)
    
    return client_fn_with_data


def run_federated_simulation(config: DictConfig):
    """Run federated learning simulation"""
    
    logger.info("Starting federated BERT learning simulation")
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(config)}")
    
    # Setup
    set_seed(config.experiment.seed)
    device = get_device(config.experiment.device)
    
    # Create server configuration
    server_config = FlowerServerConfig(
        num_rounds=config.federated.num_rounds,
        min_fit_clients=max(2, config.federated.clients_per_round),
        min_evaluate_clients=max(2, config.federated.clients_per_round),
        min_available_clients=max(2, config.federated.clients_per_round),
        fraction_fit=config.federated.clients_per_round / config.federated.num_clients,
        fraction_evaluate=config.federated.clients_per_round / config.federated.num_clients,
        
        # Model configuration
        server_model_config=FederatedBERTConfig(
            server_model_name=config.server_model.name,
            client_model_name=config.client_model.name,
            num_labels=2,  # Will be updated after data loading
            lora_r=config.lora.r,
            lora_alpha=config.lora.alpha,
            lora_dropout=config.lora.dropout,
            lora_target_modules=config.lora.target_modules,
            projection_hidden_size=config.knowledge_transfer.projection_hidden_size
        ),
        
        # Knowledge transfer configuration
        enable_knowledge_transfer=config.knowledge_transfer.progressive_transfer.enabled,
        progressive_config=ProgressiveTransferConfig(
            warmup_rounds=config.knowledge_transfer.progressive_transfer.warmup_rounds,
            max_transfer_weight=config.knowledge_transfer.progressive_transfer.transfer_weight
        ),
        alignment_config=DynamicAlignmentConfig(
            logits_weight=config.knowledge_transfer.dynamic_alignment.logits_weight,
            hidden_weight=config.knowledge_transfer.dynamic_alignment.hidden_weight,
            temperature=config.knowledge_transfer.dynamic_alignment.temperature
        ),
        
        # Aggregation configuration
        aggregation_config=AggregationConfig(
            weighting_strategy="data_size",
            momentum=0.0
        )
    )
    
    # Create server
    server = create_flower_server(
        num_rounds=config.federated.num_rounds,
        num_clients=config.federated.num_clients,
        enable_knowledge_transfer=config.knowledge_transfer.progressive_transfer.enabled
    )
    
    # Update server with our custom configuration
    server.config = server_config
    
    # Create client function
    client_fn_with_data = create_client_fn(config)
    
    # Run simulation
    logger.info(f"Starting simulation with {config.federated.num_clients} clients")
    logger.info(f"Clients per round: {config.federated.clients_per_round}")
    logger.info(f"Total rounds: {config.federated.num_rounds}")
    
    history = server.simulate_federated_learning(
        client_fn=client_fn_with_data,
        num_clients=config.federated.num_clients
    )
    
    logger.info("Federated learning simulation completed")
    
    # Print final results
    if history and hasattr(history, 'metrics_distributed'):
        final_metrics = history.metrics_distributed.get("eval_accuracy", [])
        if final_metrics:
            final_accuracy = final_metrics[-1][1]  # (round, accuracy)
            logger.info(f"Final accuracy: {final_accuracy:.4f}")
    
    return history


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(config: DictConfig) -> None:
    """Main function"""
    
    # Setup logging
    setup_logging(
        log_level=config.experiment.log_level,
        log_file=f"{config.logging.log_dir}/{config.experiment.name}.log"
    )
    
    try:
        # Run federated simulation
        history = run_federated_simulation(config)
        
        logger.info("Experiment completed successfully")
        
    except Exception as e:
        logger.error(f"Experiment failed: {str(e)}", exc_info=True)
        raise


def parse_args():
    """Parse command line arguments for non-Hydra usage"""
    parser = argparse.ArgumentParser(description="Federated BERT Learning with LoRA")
    
    parser.add_argument("--num_clients", type=int, default=10, 
                       help="Number of federated clients")
    parser.add_argument("--clients_per_round", type=int, default=5,
                       help="Number of clients participating per round")
    parser.add_argument("--num_rounds", type=int, default=20,
                       help="Number of federated learning rounds")
    parser.add_argument("--local_epochs", type=int, default=3,
                       help="Number of local epochs per client")
    parser.add_argument("--batch_size", type=int, default=16,
                       help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                       help="Learning rate")
    parser.add_argument("--task_name", type=str, default="sst2",
                       help="GLUE task name")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use (auto, cpu, cuda)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--enable_knowledge_transfer", action="store_true",
                       help="Enable knowledge transfer")
    
    return parser.parse_args()


def run_simple_simulation():
    """Run simulation with simple command line arguments"""
    
    args = parse_args()
    
    # Setup logging
    setup_logging("INFO")
    
    logger.info("Starting simple federated simulation")
    logger.info(f"Arguments: {args}")
    
    # Set seed
    set_seed(args.seed)
    
    # Create simple server
    server = create_flower_server(
        num_rounds=args.num_rounds,
        num_clients=args.num_clients,
        enable_knowledge_transfer=args.enable_knowledge_transfer
    )
    
    # Use simple client function
    history = server.simulate_federated_learning(
        client_fn=client_fn,
        num_clients=args.num_clients
    )
    
    logger.info("Simple federated simulation completed")
    return history


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--simple":
        # Run simple simulation without Hydra
        run_simple_simulation()
    else:
        # Run with Hydra configuration
        main()
