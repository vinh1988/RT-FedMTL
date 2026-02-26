#!/usr/bin/env python3
"""
Standard BERT Model for Client-Side Training
Compatible with MTL LoRA Server Model
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import List, Dict, Optional
from torch.utils.data import DataLoader, TensorDataset
import logging

logger = logging.getLogger(__name__)

class StandardBERTModel(nn.Module):
    """
    Standard BERT model for client-side training
    Compatible with MTL LoRA server architecture
    
    This model receives LoRA parameters + task heads from server
    and trains them locally
    """
    
    def __init__(self, base_model_name: str, tasks: List[str]):
        super().__init__()
        self.base_model_name = base_model_name
        self.tasks = tasks
        
        # Load BERT encoder
        logger.info(f"Initializing Standard BERT Model with base: {base_model_name}")
        self.bert = AutoModel.from_pretrained(base_model_name)
        self.hidden_size = self.bert.config.hidden_size
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        
        # Freeze BERT parameters to match server (will only train LoRA + heads)
        for param in self.bert.parameters():
            param.requires_grad = False
        
        # Task-specific heads (same as server)
        self.task_heads = nn.ModuleDict()
        for task in tasks:
            if task == 'stsb':
                self.task_heads[task] = nn.Linear(self.hidden_size, 1)
            else:
                self.task_heads[task] = nn.Linear(self.hidden_size, 2)
        
        # LoRA parameters will be received from server
        self.lora_params = {}
        
        logger.info(f"Client model initialized with {len(tasks)} tasks: {tasks}")
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, task: str) -> torch.Tensor:
        """
        Forward pass through BERT + task head
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            task: Task name
            
        Returns:
            Logits for the specified task
        """
        # Forward through BERT (frozen)
        with torch.no_grad():
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # Get [CLS] token representation
        cls_output = outputs.last_hidden_state[:, 0, :]
        
        # Apply task-specific head
        if task not in self.task_heads:
            raise ValueError(f"Unknown task: {task}. Available tasks: {list(self.task_heads.keys())}")
        
        logits = self.task_heads[task](cls_output)
        
        return logits
    
    def load_model_slice(self, model_slice: Dict[str, torch.Tensor]):
        """
        Load model slice from server (LoRA + task head parameters)
        
        Args:
            model_slice: Dictionary of parameters from server
        """
        logger.info(f"Loading model slice with {len(model_slice)} parameters")
        
        # Store LoRA parameters (we don't use them in forward pass for simplicity)
        # In a full implementation, these would be integrated into attention layers
        self.lora_params = {k: v for k, v in model_slice.items() if 'lora' in k.lower()}
        
        # Load task head parameters
        with torch.no_grad():
            for name, param_value in model_slice.items():
                if 'task_heads' in name:
                    # Extract task name and parameter name
                    parts = name.split('.')
                    if len(parts) >= 3:
                        task = parts[1]
                        param_name = '.'.join(parts[2:])
                        
                        if task in self.task_heads:
                            # Load parameter into task head
                            for pn, p in self.task_heads[task].named_parameters():
                                if pn == param_name:
                                    if isinstance(param_value, torch.Tensor):
                                        p.data.copy_(param_value)
                                    else:
                                        p.data.copy_(torch.tensor(param_value))
                                    logger.debug(f"Loaded parameter: {name}")
        
        logger.info(f"Model slice loaded: {len(self.lora_params)} LoRA params, "
                   f"{len([k for k in model_slice.keys() if 'task_heads' in k])} task head params")
    
    def get_all_parameters(self) -> Dict[str, torch.Tensor]:
        """
        Get all trainable parameters (LoRA + task heads)
        
        Returns:
            Dictionary of parameter names to tensors
        """
        params = {}
        
        # Add LoRA parameters (unchanged, just return what we received)
        params.update(self.lora_params)
        
        # Add task head parameters
        for task, head in self.task_heads.items():
            for name, param in head.named_parameters():
                params[f"task_heads.{task}.{name}"] = param.data.clone()
        
        return params
    
    def get_task_dataloader(self, task: str, batch_size: int, dataset_data: Dict) -> DataLoader:
        """
        Create dataloader for a specific task
        
        Args:
            task: Task name
            batch_size: Batch size
            dataset_data: Dictionary with 'texts' and 'labels'
            
        Returns:
            DataLoader for the task
        """
        texts = dataset_data.get('texts', [])
        labels = dataset_data.get('labels', [])
        
        if not texts or not labels:
            logger.warning(f"No data provided for task {task}")
            return DataLoader(TensorDataset(
                torch.empty(0, dtype=torch.long),
                torch.empty(0, dtype=torch.long),
                torch.empty(0, dtype=torch.long)
            ), batch_size=batch_size)
        
        # Tokenize texts
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors='pt'
        )
        
        # Convert labels to tensors
        if task == 'stsb':
            # Regression - float labels
            labels_tensor = torch.tensor(labels, dtype=torch.float)
        else:
            # Classification - long labels
            labels_tensor = torch.tensor(labels, dtype=torch.long)
        
        # Create dataset
        dataset = TensorDataset(
            encodings['input_ids'],
            encodings['attention_mask'],
            labels_tensor
        )
        
        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True
        )
        
        logger.info(f"Created dataloader for task {task}: {len(dataset)} samples, "
                   f"{len(dataloader)} batches")
        
        return dataloader
