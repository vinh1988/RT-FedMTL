#!/usr/bin/env python3
"""
Example script to run federated BERT learning on GLUE tasks.
This demonstrates how to run experiments on real datasets.
"""

import sys
import os
import logging
import argparse
import torch # Import torch for CUDA check

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.server.flower_server import create_flower_server, FlowerServerConfig
from src.clients.flower_client import FlowerFederatedClient, FlowerClientConfig
from src.models.federated_bert import FederatedBERTConfig
from src.models.knowledge_transfer import ProgressiveTransferConfig, DynamicAlignmentConfig
from src.aggregation.fedavg import AggregationConfig
from src.utils.training_utils import setup_logging, set_seed
from src.utils.data_utils import create_federated_datasets


def create_client_fn_with_glue_data(task_name: str, num_clients: int, model_name: str = "prajjwal1/bert-tiny", max_length: int = 128):
    """Create client function with GLUE data"""

    from transformers import AutoTokenizer

    # Determine device (CPU or CUDA if available)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load and partition data
    train_partitions, test_partitions, num_labels = create_federated_datasets(
        task_name=task_name,
        num_clients=num_clients,
        partition_strategy="iid",
        test_split=0.2
    )

    # Tokenize datasets
    def tokenize_function(examples):
        # Use task-specific columns
        if "sentence1" in examples and "sentence2" in examples:
            return tokenizer(examples["sentence1"], examples["sentence2"], truncation=True, padding="max_length", max_length=max_length)
        elif "sentence" in examples:
            return tokenizer(examples["sentence"], truncation=True, padding="max_length", max_length=max_length)
        else:
            # Fallback for other formats (e.g., if 'text' column is present)
            return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=max_length)

    # Apply tokenization to all partitions
    # Ensure to remove original columns and rename 'label' to 'labels' if necessary
    tokenized_train_partitions = []
    for partition in train_partitions:
        tokenized_partition = partition.map(tokenize_function, batched=True)
        tokenized_partition = tokenized_partition.remove_columns([col for col in partition.column_names if col not in ["input_ids", "attention_mask", "token_type_ids", "label"]])
        if "label" in tokenized_partition.column_names:
            tokenized_partition = tokenized_partition.rename_column("label", "labels")
        tokenized_train_partitions.append(tokenized_partition)

    tokenized_test_partitions = []
    for partition in test_partitions:
        tokenized_partition = partition.map(tokenize_function, batched=True)
        tokenized_partition = tokenized_partition.remove_columns([col for col in partition.column_names if col not in ["input_ids", "attention_mask", "token_type_ids", "label"]])
        if "label" in tokenized_partition.column_names:
            tokenized_partition = tokenized_partition.rename_column("label", "labels")
        tokenized_test_partitions.append(tokenized_partition)

    def client_fn(cid: str):
        client_id = int(cid)

        # Log CUDA availability within Ray worker
        client_device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"Ray worker client {client_id} using device: {client_device}")
        if client_device == "cuda":
            logging.info(f"Ray worker client {client_id} CUDA device count: {torch.cuda.device_count()}")

        # Get client data
        train_data = tokenized_train_partitions[client_id % len(tokenized_train_partitions)]
        test_data = tokenized_test_partitions[client_id % len(tokenized_test_partitions)]

        # Create client config
        config = FlowerClientConfig(
            client_id=client_id,
            local_epochs=3,
            batch_size=16,
            learning_rate=2e-5,
            client_model_config=FederatedBERTConfig(
                num_labels=num_labels
            ),
            task_name=task_name,
            device=client_device  # Use detected device
        )

        return FlowerFederatedClient(config, train_data, test_data)

    return client_fn, num_labels


def run_glue_experiment(task_name: str, num_clients: int = 10, num_rounds: int = 20):
    """Run federated learning experiment on GLUE task"""
    
    logger = logging.getLogger(__name__)
    logger.info(f"Starting federated BERT experiment on {task_name.upper()}")
    
    # Create client function with data
    client_fn, num_labels = create_client_fn_with_glue_data(task_name, num_clients)
    
    # Create server configuration
    server_config = FlowerServerConfig(
        num_rounds=num_rounds,
        min_fit_clients=max(2, num_clients // 2),
        min_evaluate_clients=max(2, num_clients // 2),
        min_available_clients=max(2, num_clients // 2),
        fraction_fit=0.5,
        fraction_evaluate=0.5,
        
        server_model_config=FederatedBERTConfig(num_labels=num_labels),
        enable_knowledge_transfer=True,
        progressive_config=ProgressiveTransferConfig(start_round=5),
        alignment_config=DynamicAlignmentConfig(),
        aggregation_config=AggregationConfig(weighting_strategy="data_size")
    )

    # Add the project root to PYTHONPATH for Ray workers
    project_root = os.path.join(os.path.dirname(__file__), "..")
    if "PYTHONPATH" in os.environ:
        os.environ["PYTHONPATH"] = f"{project_root}:{os.environ["PYTHONPATH"]}"
    else:
        os.environ["PYTHONPATH"] = project_root
    
    # Create and run server
    from src.server.flower_server import FlowerFederatedServer
    server = FlowerFederatedServer(server_config)
    
    logger.info(f"Configuration: {num_clients} clients, {num_rounds} rounds")
    logger.info(f"Task: {task_name}, Labels: {num_labels}")
    
    # Run simulation
    history = server.simulate_federated_learning(
        client_fn=client_fn,
        num_clients=num_clients
    )
    
    logger.info(f"Experiment on {task_name.upper()} completed!")
    return history


def main():
    """Main function with command line arguments"""
    
    parser = argparse.ArgumentParser(description="Run federated BERT on GLUE tasks")
    parser.add_argument("--task", type=str, default="sst2",
                       choices=["sst2", "cola", "mrpc", "qqp", "rte"],
                       help="GLUE task name")
    parser.add_argument("--num_clients", type=int, default=10,
                       help="Number of federated clients")
    parser.add_argument("--num_rounds", type=int, default=20,
                       help="Number of federated rounds")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--log_level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    
    args = parser.parse_args()
    
    # Setup
    setup_logging(args.log_level)
    set_seed(args.seed)
    
    try:
        # Run experiment
        history = run_glue_experiment(
            task_name=args.task,
            num_clients=args.num_clients,
            num_rounds=args.num_rounds
        )
        
        print(f"\n{'='*50}")
        print(f"Experiment completed successfully!")
        print(f"Task: {args.task.upper()}")
        print(f"Clients: {args.num_clients}")
        print(f"Rounds: {args.num_rounds}")
        print(f"{'='*50}\n")
        
        return 0
        
    except Exception as e:
        logging.error(f"Experiment failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())
