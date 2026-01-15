#!/usr/bin/env python3
"""
PEFT LoRA Model Implementation for Federated Multi-Task Learning
Uses Hugging Face PEFT library for parameter-efficient fine-tuning
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoConfig
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class PEFTLoRAMTLModel(nn.Module):
    """
    Multi-Task Learning model with PEFT LoRA adapters
    Uses separate LoRA adapters for each task with a shared base model
    """
    
    def __init__(
        self, 
        base_model_name: str, 
        tasks: List[str],
        lora_rank: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        target_modules: List[str] = None,
        unfreeze_layers: int = 2  # Number of last transformer layers to unfreeze
    ):
        super().__init__()
        self.base_model_name = base_model_name
        self.tasks = tasks
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.unfreeze_layers = unfreeze_layers
        
        # Default target modules for BERT (Q, V projections in attention)
        if target_modules is None:
            target_modules = ["query", "value"]  # Can also add "key", "dense"
        
        self.target_modules = target_modules
        
        logger.info(f"Initializing PEFT LoRA MTL Model with {len(tasks)} tasks")
        logger.info(f"LoRA config: rank={lora_rank}, alpha={lora_alpha}, dropout={lora_dropout}")
        logger.info(f"Target modules: {target_modules}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        
        # Load shared base BERT model (encoder only)
        # Load base config to get model info (we'll create task-specific models later)
        config = AutoConfig.from_pretrained(base_model_name)
        self.hidden_size = config.hidden_size
        self.num_layers = config.num_hidden_layers
        
        logger.info(f"Initializing PEFT LoRA MTL Model:")
        logger.info(f"  Base model: {base_model_name} ({self.num_layers} layers)")
        logger.info(f"  LoRA rank: {lora_rank}, alpha: {lora_alpha}")
        logger.info(f"  Target modules: {target_modules}")
        logger.info(f"  Will unfreeze last {self.unfreeze_layers} layers in each task model")
        
        # Task-specific LoRA adapters and classification heads
        self.task_adapters = nn.ModuleDict()
        self.task_heads = nn.ModuleDict()
        
        for task in tasks:
            # Create LoRA config for this task
            peft_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,  # We'll add our own heads
                r=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=target_modules,
                bias="none",  # Don't adapt bias
                inference_mode=False
            )
            
            # Create a copy of base model with LoRA for this task
            task_model = AutoModel.from_pretrained(base_model_name)
            task_peft_model = get_peft_model(task_model, peft_config)
            
            # CRITICAL FIX: Unfreeze layers in THIS task model
            # The unfreezing on self.bert doesn't carry over to these new models!
            if hasattr(task_peft_model.base_model.model, 'encoder'):
                base_bert = task_peft_model.base_model.model  # Access the actual BERT inside PEFT wrapper
                total_layers = len(base_bert.encoder.layer)
                start_unfreeze = max(0, total_layers - self.unfreeze_layers)
                
                logger.info(f"  Task '{task}': Unfreezing last {self.unfreeze_layers} layers (layers {start_unfreeze}-{total_layers-1})")
                for layer_idx in range(start_unfreeze, total_layers):
                    for param in base_bert.encoder.layer[layer_idx].parameters():
                        param.requires_grad = True
                
                # Unfreeze pooler
                if hasattr(base_bert, 'pooler') and base_bert.pooler is not None:
                    for param in base_bert.pooler.parameters():
                        param.requires_grad = True
            
            self.task_adapters[task] = task_peft_model
            
            # Task-specific classification/regression head
            if task == 'stsb':
                self.task_heads[task] = nn.Linear(self.hidden_size, 1)  # Regression
            else:
                self.task_heads[task] = nn.Linear(self.hidden_size, 2)  # Binary classification
        
        logger.info(f"[SUCCESS] PEFT LoRA MTL Model initialized successfully")
        self._print_trainable_parameters()
    
    def forward(self, input_ids, attention_mask, task_name: str):
        """
        Forward pass for a specific task using its LoRA adapter
        """
        if task_name not in self.task_adapters:
            raise ValueError(f"Task '{task_name}' not found in model's tasks: {self.tasks}")
        
        # Get the task-specific LoRA model
        task_model = self.task_adapters[task_name]
        
        # Forward through BERT with LoRA adapter
        outputs = task_model(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output  # [CLS] token representation
        
        # Task-specific head
        logits = self.task_heads[task_name](pooled_output)
        
        return logits
    
    def get_lora_parameters(self, task: str = None) -> Dict[str, torch.Tensor]:
        """
        Extract trainable parameters (LoRA adapters + unfrozen layers) for federated aggregation
        If task is specified, return only that task's parameters
        Otherwise, return all tasks' parameters
        """
        lora_params = {}
        
        tasks_to_extract = [task] if task else self.tasks
        
        for task_name in tasks_to_extract:
            if task_name not in self.task_adapters:
                continue
            
            task_model = self.task_adapters[task_name]
            
            # Get ALL trainable parameters (LoRA adapters + unfrozen base layers)
            for name, param in task_model.named_parameters():
                if param.requires_grad:  # Extract ALL trainable parameters
                    full_name = f"{task_name}.{name}"
                    lora_params[full_name] = param.data.clone()
            
            # Also include task head parameters
            for name, param in self.task_heads[task_name].named_parameters():
                full_name = f"{task_name}.head.{name}"
                lora_params[full_name] = param.data.clone()
        
        logger.debug(f"Extracted {len(lora_params)} trainable parameters (LoRA + unfrozen layers) for task(s): {tasks_to_extract}")
        return lora_params
    
    def set_lora_parameters(self, lora_params: Dict[str, torch.Tensor], task: str = None):
        """
        Set trainable parameters (LoRA + unfrozen layers) from federated aggregation
        """
        tasks_to_update = [task] if task else self.tasks
        
        for task_name in tasks_to_update:
            if task_name not in self.task_adapters:
                continue
            
            task_model = self.task_adapters[task_name]
            
            # Update ALL trainable parameters (LoRA adapters + unfrozen base layers)
            for name, param in task_model.named_parameters():
                full_name = f"{task_name}.{name}"
                if full_name in lora_params and param.requires_grad:
                    param.data = lora_params[full_name].to(param.device)
            
            # Update task head parameters
            for name, param in self.task_heads[task_name].named_parameters():
                full_name = f"{task_name}.head.{name}"
                if full_name in lora_params:
                    param.data = lora_params[full_name].to(param.device)
        
        logger.debug(f"Updated trainable parameters (LoRA + unfrozen layers) for task(s): {tasks_to_update}")
    
    def get_task_dataloader(self, task: str, batch_size: int, dataset_data: Dict):
        """
        Create a DataLoader for a specific task
        Compatible with existing federated training code
        """
        from torch.utils.data import DataLoader, TensorDataset
        
        texts = dataset_data['texts']
        labels = dataset_data['labels']
        
        # Tokenize
        encodings = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )
        
        # Create dataset
        dataset = TensorDataset(
            encodings['input_ids'],
            encodings['attention_mask'],
            torch.tensor(labels, dtype=torch.float32 if task == 'stsb' else torch.long)
        )
        
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    def _print_trainable_parameters(self):
        """Print the number of trainable parameters"""
        total_params = 0
        trainable_params = 0
        
        for task in self.tasks:
            task_model = self.task_adapters[task]
            for param in task_model.parameters():
                total_params += param.numel()
                if param.requires_grad:
                    trainable_params += param.numel()
            
            # Add head parameters
            for param in self.task_heads[task].parameters():
                total_params += param.numel()
                trainable_params += param.numel()
        
        percentage = 100 * trainable_params / total_params if total_params > 0 else 0
        
        logger.info(f"="*60)
        logger.info(f"PEFT LoRA Model Parameters:")
        logger.info(f"  Total parameters: {total_params:,}")
        logger.info(f"  Trainable parameters: {trainable_params:,}")
        logger.info(f"  Percentage trainable: {percentage:.2f}%")
        logger.info(f"="*60)
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get model summary for logging"""
        total_params = sum(p.numel() for task in self.tasks 
                          for p in self.task_adapters[task].parameters())
        trainable_params = sum(p.numel() for task in self.tasks 
                              for p in self.task_adapters[task].parameters() if p.requires_grad)
        
        return {
            "base_model": self.base_model_name,
            "tasks": self.tasks,
            "lora_rank": self.lora_rank,
            "lora_alpha": self.lora_alpha,
            "target_modules": self.target_modules,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "trainable_percentage": 100 * trainable_params / total_params if total_params > 0 else 0
        }


class PEFTLoRAServerModel(nn.Module):
    """
    Server-side PEFT LoRA MTL model for federated learning
    Maintains global LoRA adapters and aggregates client updates
    """
    
    def __init__(
        self,
        base_model_name: str,
        tasks: List[str],
        lora_rank: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        target_modules: List[str] = None,
        unfreeze_layers: int = 2
    ):
        super().__init__()
        
        # Use the same architecture as client model
        self.mtl_model = PEFTLoRAMTLModel(
            base_model_name=base_model_name,
            tasks=tasks,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            unfreeze_layers=unfreeze_layers
        )
        
        self.base_model_name = base_model_name
        self.tasks = tasks
        self.lora_rank = lora_rank
        
        logger.info(f"[SUCCESS] PEFT LoRA Server Model initialized with {len(tasks)} tasks")
    
    def forward(self, input_ids, attention_mask, task_name: str):
        """Forward pass through the MTL model"""
        return self.mtl_model(input_ids, attention_mask, task_name)
    
    def get_global_lora_state(self) -> Dict[str, torch.Tensor]:
        """Get global LoRA state for broadcasting to clients"""
        return self.mtl_model.get_lora_parameters()
    
    def update_from_aggregation(self, aggregated_params: Dict[str, torch.Tensor]):
        """Update model from aggregated client LoRA parameters"""
        self.mtl_model.set_lora_parameters(aggregated_params)
        logger.info(f"Updated server model with {len(aggregated_params)} aggregated LoRA parameters")
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get detailed model summary"""
        return self.mtl_model.get_model_summary()

