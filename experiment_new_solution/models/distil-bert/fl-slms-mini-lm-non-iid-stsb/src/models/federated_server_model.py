#!/usr/bin/env python3
"""
Standard Federated Learning Server Model
Single task model for federated learning (no MTL)
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

class FederatedServerModel(nn.Module):
    """
    Standard Federated Learning Server Model
    
    Architecture:
    - Single MiniLM encoder (shared across all clients)
    - Single task-specific head (based on client task)
    
    This is a standard FL model where each client trains on its specific task
    and the server aggregates the same model architecture.
    """
    
    def __init__(self, base_model_name: str, task: str):
        super().__init__()
        self.base_model_name = base_model_name
        self.task = task
        
        # Load base MiniLM model
        logger.info(f"Initializing Federated Server Model with base: {base_model_name}")
        self.bert = AutoModel.from_pretrained(base_model_name)
        self.hidden_size = self.bert.config.hidden_size
        
        # Task-specific head
        if task == 'stsb':
            # Regression head for semantic similarity
            self.classifier = nn.Linear(self.hidden_size, 1)
        else:
            # Binary classification heads for SST-2 and QQP
            self.classifier = nn.Linear(self.hidden_size, 2)
        
        # Log model info
        total_params = sum(p.numel() for p in self.parameters())
        bert_params = sum(p.numel() for p in self.bert.parameters())
        classifier_params = sum(p.numel() for p in self.classifier.parameters())
        
        logger.info(f"Federated Server Model initialized for task: {task}")
        logger.info(f"Base model parameters: {bert_params:,}")
        logger.info(f"Classifier parameters: {classifier_params:,}")
        logger.info(f"Total parameters: {total_params:,}")
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, task: str = None) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            task: Task name (optional, uses self.task if not provided)
            
        Returns:
            Logits for the task
        """
        if task is None:
            task = self.task
            
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
    
    def get_model_info(self) -> Dict[str, any]:
        """
        Get model information for logging
        
        Returns:
            Dictionary with model details
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_name': self.base_model_name,
            'task': self.task,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'hidden_size': self.hidden_size,
            'architecture': 'MiniLM-L6-H384'
        }
