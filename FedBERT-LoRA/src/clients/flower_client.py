"""
Flower client implementation for federated BERT learning.
Handles client-side training with TinyBERT and LoRA.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple, Any
from flwr.client import Client, NumPyClient
from flwr.common import (
    Parameters,
    FitIns,
    FitRes,
    EvaluateIns, 
    EvaluateRes,
    parameters_to_ndarrays,
    ndarrays_to_parameters
)
import numpy as np
from dataclasses import dataclass
import logging
from transformers import AutoTokenizer
from datasets import Dataset

from ..models.federated_bert import FederatedBERTClient, FederatedBERTConfig
from ..models.knowledge_transfer import (
    AdaptiveKnowledgeTransfer,
    FederatedDistillationLoss,
    ProgressiveTransferConfig,
    DynamicAlignmentConfig
)
from ..utils.data_utils import create_data_loader

logger = logging.getLogger(__name__)


@dataclass
class FlowerClientConfig:
    """Configuration for Flower federated client"""
    client_id: int
    local_epochs: int = 3
    batch_size: int = 16
    learning_rate: float = 2e-5
    
    # Model configuration
    client_model_config: FederatedBERTConfig = None
    
    # Data configuration
    dataset_name: str = "glue"
    task_name: str = "sst2"
    max_length: int = 128
    
    # Knowledge transfer configuration
    enable_knowledge_transfer: bool = True
    progressive_config: ProgressiveTransferConfig = None
    alignment_config: DynamicAlignmentConfig = None
    
    # Training configuration
    device: str = "cpu"
    gradient_clipping: float = 1.0
    warmup_steps: int = 100
    
    def __post_init__(self):
        if self.client_model_config is None:
            self.client_model_config = FederatedBERTConfig()
        
        if self.progressive_config is None:
            self.progressive_config = ProgressiveTransferConfig()
        
        if self.alignment_config is None:
            self.alignment_config = DynamicAlignmentConfig()


class FlowerFederatedClient(NumPyClient):
    """
    Flower client for federated BERT learning with TinyBERT.
    """
    
    def __init__(self, config: FlowerClientConfig, train_data: Dataset, test_data: Dataset):
        super().__init__()
        
        self.config = config
        self.device = torch.device(config.device)
        
        # Initialize client model
        self.model = FederatedBERTClient(config.client_model_config)
        self.model.to(self.device)
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.client_model_config.client_model_name
        )
        
        # Prepare data loaders - simplified for now
        from torch.utils.data import DataLoader, Dataset as TorchDataset
        
        class SimpleDataset(TorchDataset):
            def __init__(self, data):
                self.data = data
            def __len__(self):
                return len(self.data['input_ids'])
            def __getitem__(self, idx):
                return {
                    'input_ids': torch.tensor(self.data['input_ids'][idx], dtype=torch.long),
                    'attention_mask': torch.tensor(self.data['attention_mask'][idx], dtype=torch.long),
                    'labels': torch.tensor(self.data['labels'][idx], dtype=torch.long)
                }
        
        self.train_loader = DataLoader(
            SimpleDataset(train_data), 
            batch_size=config.batch_size,
            shuffle=True
        )
        
        self.test_loader = DataLoader(
            SimpleDataset(test_data),
            batch_size=config.batch_size,
            shuffle=False
        )
        
        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=0.01
        )
        
        # Initialize knowledge transfer components
        if config.enable_knowledge_transfer:
            self.knowledge_transfer = AdaptiveKnowledgeTransfer(
                config.progressive_config,
                config.alignment_config
            )
            self.distillation_loss = FederatedDistillationLoss()
        else:
            self.knowledge_transfer = None
            self.distillation_loss = None
        
        # Store parameter names for consistency
        self.param_names = list(self.model.get_lora_parameters().keys())
        
        logger.info(f"Client {config.client_id} initialized with {len(train_data)} training samples")
    
    def get_parameters(self, config: Dict[str, Any]) -> List[np.ndarray]:
        """Get model parameters as numpy arrays"""
        
        lora_params = self.model.get_lora_parameters()
        param_arrays = []
        
        for param_name in self.param_names:
            if param_name in lora_params:
                param_arrays.append(lora_params[param_name].cpu().detach().numpy())
            else:
                logger.warning(f"Parameter {param_name} not found in client model")
        
        return param_arrays
    
    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """Set model parameters from numpy arrays"""
        
        param_dict = {}
        for i, param_name in enumerate(self.param_names):
            if i < len(parameters):
                param_dict[param_name] = torch.from_numpy(parameters[i]).to(self.device)
        
        self.model.set_lora_parameters(param_dict)
    
    def fit(self, parameters: List[np.ndarray], config: Dict[str, Any]) -> Tuple[List[np.ndarray], int, Dict[str, Any]]:
        """Train the local model"""
        
        # Set global parameters
        self.set_parameters(parameters)
        
        # Extract training configuration
        round_num = config.get("round", 0)
        transfer_weight = config.get("transfer_weight", 0.0)
        server_knowledge = config.get("server_knowledge", None)
        
        logger.info(f"Client {self.config.client_id} starting training round {round_num}")
        logger.info(f"Transfer weight: {transfer_weight:.4f}")
        
        # Train the model
        train_loss, train_accuracy = self._train_epoch(
            round_num, 
            transfer_weight, 
            server_knowledge
        )
        
        # Get updated parameters
        updated_parameters = self.get_parameters(config)
        
        # Training metrics
        metrics = {
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "transfer_weight": transfer_weight,
            "round": round_num
        }
        
        logger.info(f"Client {self.config.client_id} completed training: Loss={train_loss:.4f}, Acc={train_accuracy:.4f}")
        
        return updated_parameters, len(self.train_loader.dataset), metrics
    
    def evaluate(self, parameters: List[np.ndarray], config: Dict[str, Any]) -> Tuple[float, int, Dict[str, Any]]:
        """Evaluate the local model"""
        
        # Set global parameters
        self.set_parameters(parameters)
        
        # Evaluate the model
        eval_loss, eval_accuracy = self._evaluate()
        
        metrics = {
            "accuracy": eval_accuracy,
            "loss": eval_loss
        }
        
        logger.info(f"Client {self.config.client_id} evaluation: Loss={eval_loss:.4f}, Acc={eval_accuracy:.4f}")
        
        return eval_loss, len(self.test_loader.dataset), metrics
    
    def _train_epoch(self, round_num: int, transfer_weight: float, 
                    server_knowledge: Optional[Dict[str, Any]] = None) -> Tuple[float, float]:
        """Train for one epoch"""
        
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Forward pass
            outputs = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
                return_hidden_states=self.knowledge_transfer is not None
            )
            
            task_loss = outputs["loss"]
            total_loss += task_loss.item()
            
            # Compute accuracy
            predictions = torch.argmax(outputs["logits"], dim=-1)
            total_correct += (predictions == batch["labels"]).sum().item()
            total_samples += batch["labels"].size(0)
            
            # Knowledge transfer loss
            if self.knowledge_transfer and server_knowledge and transfer_weight > 0:
                # Create dummy server outputs for demonstration
                # In practice, server knowledge would be provided
                server_outputs = {
                    "logits": torch.randn_like(outputs["logits"]),
                    "projected_hidden_states": torch.randn_like(outputs["hidden_states"][:, 0])
                }
                
                transfer_losses = self.knowledge_transfer.compute_transfer_loss(
                    round_num, outputs, server_outputs, batch["attention_mask"]
                )
                
                # Combine task and transfer losses
                combined_losses = self.distillation_loss.compute_loss(task_loss, transfer_losses)
                loss = combined_losses["total_loss"]
            else:
                loss = task_loss
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            if self.config.gradient_clipping > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config.gradient_clipping
                )
            
            self.optimizer.step()
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = total_correct / total_samples
        
        return avg_loss, accuracy
    
    def _evaluate(self) -> Tuple[float, float]:
        """Evaluate the model"""
        
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in self.test_loader:
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"]
                )
                
                total_loss += outputs["loss"].item()
                
                # Compute accuracy
                predictions = torch.argmax(outputs["logits"], dim=-1)
                total_correct += (predictions == batch["labels"]).sum().item()
                total_samples += batch["labels"].size(0)
        
        avg_loss = total_loss / len(self.test_loader)
        accuracy = total_correct / total_samples
        
        return avg_loss, accuracy


def create_flower_client(client_id: int, 
                        train_data: Dataset,
                        test_data: Dataset,
                        local_epochs: int = 3,
                        batch_size: int = 16,
                        learning_rate: float = 2e-5,
                        device: str = "cpu") -> FlowerFederatedClient:
    """Factory function to create Flower federated client"""
    
    config = FlowerClientConfig(
        client_id=client_id,
        local_epochs=local_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        device=device
    )
    
    return FlowerFederatedClient(config, train_data, test_data)


def client_fn(cid: str) -> FlowerFederatedClient:
    """Create a client function for Flower simulation"""
    
    client_id = int(cid)

    
    try:
        # Load and partition data (this is a placeholder)
        # In practice, you would load actual data here
        from datasets import Dataset
        
        # Create dummy data for demonstration
        dummy_data = {
            "input_ids": [[101, 2023, 2003, 1037, 3231, 102] + [0] * 122] * 50,  # Smaller dataset
            "attention_mask": [[1, 1, 1, 1, 1, 1] + [0] * 122] * 50,
            "labels": [0, 1] * 25  # Binary classification
        }
        
        train_data = Dataset.from_dict(dummy_data)
        test_data = Dataset.from_dict({k: v[:10] for k, v in dummy_data.items()})  # Smaller test set
        
        return create_flower_client(
            client_id=client_id,
            train_data=train_data,
            test_data=test_data,
            device="cpu"  # Use CPU for simulation
        )
    except Exception as e:
        print(f"Error creating client {cid}: {e}")
        # Return a minimal client if data loading fails
        from datasets import Dataset
        minimal_data = {
            "input_ids": [[101, 102] + [0] * 126] * 10,
            "attention_mask": [[1, 1] + [0] * 126] * 10,
            "labels": [0] * 10
        }
        train_data = Dataset.from_dict(minimal_data)
        test_data = Dataset.from_dict({k: v[:5] for k, v in minimal_data.items()})
        
        return create_flower_client(
            client_id=client_id,
            train_data=train_data,
            test_data=test_data,
            device="cpu"
        )


if __name__ == "__main__":
    # Test client creation
    client = client_fn("0")
    print(f"Flower federated client created successfully")
    print(f"Client ID: {client.config.client_id}")
    print(f"Training samples: {len(client.train_loader.dataset)}")
    print(f"Test samples: {len(client.test_loader.dataset)}")
