#!/usr/bin/env python3
"""
Standard Federated Learning Client Model
Single task model for federated learning (no MTL)
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import Dict, List, Optional, Tuple
import logging
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

class FederatedClientModel(nn.Module):
    """
    Standard Federated Learning Client Model
    
    Architecture:
    - Single MiniLM encoder
    - Single task-specific head
    - Standard federated learning (no multi-task sharing)
    """
    
    def __init__(self, base_model_name: str, task: str, learning_rate: float = 2e-5):
        super().__init__()
        self.base_model_name = base_model_name
        self.task = task
        self.learning_rate = learning_rate
        
        # Load base MiniLM model
        logger.info(f"Initializing Federated Client Model with base: {base_model_name}")
        self.bert = AutoModel.from_pretrained(base_model_name)
        self.hidden_size = self.bert.config.hidden_size
        
        # Task-specific head
        if task == 'stsb':
            # Regression head for semantic similarity
            self.classifier = nn.Linear(self.hidden_size, 1)
        else:
            # Binary classification heads for SST-2 and QQP
            self.classifier = nn.Linear(self.hidden_size, 2)
        
        # Setup tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        
        # Log model info
        total_params = sum(p.numel() for p in self.parameters())
        logger.info(f"Federated Client Model initialized for task: {task}")
        logger.info(f"Total parameters: {total_params:,}")
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            
        Returns:
            Logits for the task
        """
        # Get BERT embeddings
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Use [CLS] token representation
        pooled_output = outputs.last_hidden_state[:, 0, :]
        
        # Apply task-specific classifier
        logits = self.classifier(pooled_output)
        
        return logits
    
    def get_parameters(self) -> Dict[str, torch.Tensor]:
        """
        Get all model parameters for federated aggregation
        
        Returns:
            Dictionary of parameter names and tensors
        """
        params = {}
        for name, param in self.named_parameters():
            params[name] = param.data.clone()
        return params
    
    def set_parameters(self, parameters: Dict[str, torch.Tensor]) -> None:
        """
        Set model parameters from federated aggregation
        
        Args:
            parameters: Dictionary of parameter names and tensors
        """
        with torch.no_grad():
            for name, param in self.named_parameters():
                if name in parameters:
                    param.data.copy_(parameters[name])
                else:
                    logger.warning(f"Parameter {name} not found in received parameters")
    
    def get_task_dataloader(self, task: str, batch_size: int, dataset_data: Dict = None) -> DataLoader:
        """
        Create a DataLoader for the specific task
        
        Args:
            task: Task name
            batch_size: Batch size
            dataset_data: Optional pre-loaded dataset data
            
        Returns:
            DataLoader for the task
        """
        from src.datasets.federated_datasets import FederatedDataset
        
        if dataset_data:
            # Use provided dataset data
            dataset = FederatedDataset(
                texts=dataset_data.get('texts', []),
                labels=dataset_data.get('labels', []),
                tokenizer=self.tokenizer,
                max_length=128,
                task=task
            )
        else:
            # Load dataset from files
            dataset = FederatedDataset(
                task=task,
                tokenizer=self.tokenizer,
                max_length=128
            )
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0
        )
    
    def get_loss_function(self, task: str):
        """
        Get the appropriate loss function for the task
        
        Args:
            task: Task name
            
        Returns:
            Loss function
        """
        if task == 'stsb':
            # Mean Squared Error for regression
            return nn.MSELoss()
        else:
            # Cross Entropy for classification
            return nn.CrossEntropyLoss()
    
    def get_metrics(self, task: str, predictions: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
        """
        Calculate metrics for the task
        
        Args:
            task: Task name
            predictions: Model predictions
            labels: Ground truth labels
            
        Returns:
            Dictionary of metrics
        """
        if task == 'stsb':
            # Regression metrics
            mse = torch.nn.functional.mse_loss(predictions.squeeze(), labels.float())
            rmse = torch.sqrt(mse)
            
            # Convert to accuracy-like metric (1 - normalized MSE)
            accuracy_like = 1.0 - (mse / (labels.max() - labels.min()).pow(2))
            
            return {
                'mse': mse.item(),
                'rmse': rmse.item(),
                'accuracy': accuracy_like.item()
            }
        else:
            # Classification metrics
            pred_classes = torch.argmax(predictions, dim=1)
            correct = (pred_classes == labels).sum().item()
            total = labels.size(0)
            accuracy = correct / total
            
            return {
                'accuracy': accuracy
            }
