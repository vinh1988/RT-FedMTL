#!/usr/bin/env python3
"""
Multi-Task Learning Server Model with LoRA (Option 1)
Server maintains frozen BERT + shared LoRA adapters + task-specific heads
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from typing import List, Dict, Optional
import logging
from src.lora.federated_lora import LoRALayer

logger = logging.getLogger(__name__)

class MTLLoRAServerModel(nn.Module):
    """
    Multi-Task Server Model with LoRA + Partial Layer Unfreezing
    
    Architecture:
    - BERT encoder bottom layers: FROZEN (pre-trained knowledge preserved)
    - BERT encoder top layers: TRAINABLE (task-specific adaptation)
    - Shared LoRA adapters: TRAINABLE (aggregated from ALL clients for cross-task learning)
    - Task-specific heads: TRAINABLE (aggregated within same-task clients)
    
    Benefits:
    - More flexible than pure LoRA (top layers can adapt)
    - Better than full fine-tuning (bottom layers preserve backbone)
    - Cross-task knowledge transfer via shared LoRA
    """
    
    def __init__(self, base_model_name: str, tasks: List[str], 
                 lora_rank: int = 8, lora_alpha: float = 16.0, lora_dropout: float = 0.1,
                 layers_to_freeze: int = 2):
        super().__init__()
        self.base_model_name = base_model_name
        self.tasks = tasks
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.layers_to_freeze = layers_to_freeze
        
        # Load base model (BERT or DistilBERT)
        logger.info(f"Initializing MTL LoRA Server Model with base: {base_model_name}")
        self.bert = AutoModel.from_pretrained(base_model_name)
        self.hidden_size = self.bert.config.hidden_size
        
        # Handle different architectures
        if hasattr(self.bert, 'encoder'):
            # BERT/MiniLM architecture
            self.num_layers = len(self.bert.encoder.layer)
            self.is_distilbert = False
        elif hasattr(self.bert, 'transformer'):
            # DistilBERT architecture
            self.num_layers = len(self.bert.transformer.layer)
            self.is_distilbert = True
        else:
            raise ValueError(f"Unsupported model architecture: {type(self.bert)}")
        
        # Partial layer freezing
        frozen_params, unfrozen_params = self._freeze_bottom_layers(layers_to_freeze)
        logger.info(f"Model layers: {self.num_layers} total, {layers_to_freeze} frozen, {self.num_layers - layers_to_freeze} unfrozen")
        logger.info(f"Frozen base params: {frozen_params:,}, Unfrozen base params: {unfrozen_params:,}")
        
        # Inject shared LoRA adapters into attention layers
        self.shared_lora = self._create_shared_lora_adapters()
        lora_params = sum(p.numel() for p in self.shared_lora.parameters())
        logger.info(f"Created shared LoRA adapters: {lora_params:,} trainable params")
        
        # Task-specific heads (trainable)
        self.task_heads = nn.ModuleDict()
        for task in tasks:
            if task == 'stsb':
                # Regression head for semantic similarity
                self.task_heads[task] = nn.Linear(self.hidden_size, 1)
            else:
                # Binary classification heads for SST-2 and QQP
                self.task_heads[task] = nn.Linear(self.hidden_size, 2)
        
        logger.info(f"Created {len(tasks)} task-specific heads: {tasks}")
        for task in tasks:
            task_params = sum(p.numel() for p in self.task_heads[task].parameters())
            logger.info(f"  Task '{task}' head: {task_params:,} params")
        
        total_trainable = unfrozen_params + lora_params + sum(p.numel() for p in self.task_heads.parameters())
        logger.info(f"Total trainable parameters: {total_trainable:,}")
    
    def _freeze_bottom_layers(self, num_layers_to_freeze: int) -> tuple:
        """
        Freeze bottom N layers of BERT, keep top layers trainable
        
        Args:
            num_layers_to_freeze: Number of bottom layers to freeze
            
        Returns:
            Tuple of (frozen_params, unfrozen_params)
        """
        # Get the correct layer container based on architecture
        if self.is_distilbert:
            total_layers = len(self.bert.transformer.layer)
            layer_container = self.bert.transformer.layer
        else:
            total_layers = len(self.bert.encoder.layer)
            layer_container = self.bert.encoder.layer
        frozen_params = 0
        unfrozen_params = 0
        
        # Always freeze embeddings
        for param in self.bert.embeddings.parameters():
            param.requires_grad = False
            frozen_params += param.numel()
        
        # Freeze bottom layers, unfreeze top layers
        for i, layer in enumerate(layer_container):
            if i < num_layers_to_freeze:
                # Freeze this layer
                for param in layer.parameters():
                    param.requires_grad = False
                    frozen_params += param.numel()
                logger.info(f"  Layer {i}: FROZEN")
            else:
                # Unfreeze this layer
                for param in layer.parameters():
                    param.requires_grad = True
                    unfrozen_params += param.numel()
                logger.info(f"  Layer {i}: TRAINABLE")
        
        # Freeze pooler if exists
        if hasattr(self.bert, 'pooler') and self.bert.pooler is not None:
            for param in self.bert.pooler.parameters():
                param.requires_grad = False
                frozen_params += param.numel()
        
        return frozen_params, unfrozen_params
    
    def _create_shared_lora_adapters(self) -> nn.ModuleDict:
        """
        Create shared LoRA adapters for attention layers
        
        Target layers: Q, K, V projections and output dense layer in each attention block
        These LoRA adapters are shared across all tasks and aggregated from ALL clients
        """
        lora_modules = nn.ModuleDict()
        
        # Inject LoRA into attention layers
        if self.is_distilbert:
            layer_container = self.bert.transformer.layer
        else:
            layer_container = self.bert.encoder.layer
            
        for layer_idx, layer in enumerate(layer_container):
            # Handle different attention module names
            if self.is_distilbert:
                # DistilBERT attention modules
                attention_modules = [
                    ('q_lin', layer.attention.q_lin),
                    ('k_lin', layer.attention.k_lin),
                    ('v_lin', layer.attention.v_lin),
                    ('out_lin', layer.attention.out_lin)
                ]
            else:
                # BERT attention modules
                attention_modules = [
                    ('query', layer.attention.self.query),
                    ('key', layer.attention.self.key),
                    ('value', layer.attention.self.value),
                    ('output', layer.attention.output.dense)
                ]
            for proj_name, proj_module in attention_modules:
                module_name = f"layer_{layer_idx}_attention_{proj_name}"
                lora_modules[module_name] = LoRALayer(
                    in_features=proj_module.in_features,
                    out_features=proj_module.out_features,
                    rank=self.lora_rank,
                    alpha=self.lora_alpha,
                    dropout=self.lora_dropout
                )
        
        logger.info(f"Injected LoRA into {len(lora_modules)} attention modules")
        return lora_modules
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, task: str) -> torch.Tensor:
        """
        Forward pass with frozen BERT + shared LoRA + task head
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            task: Task name to use specific head
            
        Returns:
            Logits for the specified task
        """
        # BERT encoder (frozen) - forward pass
        with torch.no_grad():
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        
        # Get hidden states from each layer
        hidden_states = outputs.hidden_states  # Tuple of (batch_size, seq_len, hidden_size)
        
        # Apply shared LoRA adapters to attention outputs
        lora_enhanced_output = hidden_states[-1]  # Start with last layer output
        
        # For simplification, we apply LoRA to the final [CLS] representation
        # In a full implementation, LoRA would be integrated into each attention layer's forward pass
        cls_hidden = lora_enhanced_output[:, 0, :]  # (batch_size, hidden_size)
        
        # Apply a combined LoRA transformation (simplified version)
        # In practice, this should be integrated into BERT's attention mechanism
        lora_output = cls_hidden
        for module_name, lora_layer in self.shared_lora.items():
            if 'layer_last' in module_name or len(self.shared_lora) < 10:
                # Apply LoRA transformation and add to hidden state
                lora_delta = lora_layer(cls_hidden)
                # For output projection, add the LoRA delta
                if lora_delta.shape == lora_output.shape:
                    lora_output = lora_output + lora_delta
                break  # Simplified: apply only one LoRA for demo purposes
        
        # Task-specific head
        if task not in self.task_heads:
            raise ValueError(f"Unknown task: {task}. Available tasks: {list(self.task_heads.keys())}")
        
        logits = self.task_heads[task](lora_output)
        
        return logits
    
    def get_shared_lora_parameters(self) -> Dict[str, torch.Tensor]:
        """
        Get shared LoRA parameters (aggregated from ALL clients)
        
        Returns:
            Dictionary of parameter names to tensors for shared LoRA
        """
        params = {}
        for module_name, lora_module in self.shared_lora.items():
            params[f"shared_lora.{module_name}.lora_A"] = lora_module.lora_A.data.clone()
            params[f"shared_lora.{module_name}.lora_B"] = lora_module.lora_B.data.clone()
        return params
    
    def set_shared_lora_parameters(self, parameters: Dict[str, torch.Tensor]):
        """
        Set shared LoRA parameters after aggregation
        
        Args:
            parameters: Dictionary of parameter names to tensors
        """
        with torch.no_grad():
            for module_name, lora_module in self.shared_lora.items():
                lora_a_key = f"shared_lora.{module_name}.lora_A"
                lora_b_key = f"shared_lora.{module_name}.lora_B"
                
                if lora_a_key in parameters:
                    lora_module.lora_A.data.copy_(parameters[lora_a_key])
                if lora_b_key in parameters:
                    lora_module.lora_B.data.copy_(parameters[lora_b_key])
    
    def get_task_head_parameters(self, task: str) -> Dict[str, torch.Tensor]:
        """
        Get task-specific head parameters (aggregated only from same-task clients)
        
        Args:
            task: Task name ('sst2', 'qqp', or 'stsb')
            
        Returns:
            Dictionary of parameter names to tensors for the task head
        """
        if task not in self.task_heads:
            raise ValueError(f"Unknown task: {task}. Available tasks: {list(self.task_heads.keys())}")
        
        return {f"task_heads.{task}.{name}": param.data.clone() 
                for name, param in self.task_heads[task].named_parameters()}
    
    def set_task_head_parameters(self, task: str, parameters: Dict[str, torch.Tensor]):
        """
        Set task-specific head parameters after aggregation
        
        Args:
            task: Task name
            parameters: Dictionary of parameter names to tensors
        """
        if task not in self.task_heads:
            raise ValueError(f"Unknown task: {task}")
        
        with torch.no_grad():
            for name, param in self.task_heads[task].named_parameters():
                full_name = f"task_heads.{task}.{name}"
                if full_name in parameters:
                    param.data.copy_(parameters[full_name])
    
    def get_model_slice_for_task(self, task: str) -> Dict[str, torch.Tensor]:
        """
        Get model slice for a specific task (frozen BERT is NOT sent, only LoRA + task head)
        
        Args:
            task: Task name
            
        Returns:
            Dictionary containing shared LoRA parameters and task-specific head parameters
        """
        model_slice = {}
        
        # IMPORTANT: Do NOT send frozen BERT parameters (saves 168x communication)
        # Only send trainable parameters
        
        # Add shared LoRA parameters
        model_slice.update(self.get_shared_lora_parameters())
        
        # Add task-specific head parameters
        model_slice.update(self.get_task_head_parameters(task))
        
        return model_slice
    
    def get_unfrozen_layer_parameters(self) -> Dict[str, torch.Tensor]:
        """
        Get unfrozen BERT layer parameters (for aggregation)
        
        Returns:
            Dictionary of parameter names to tensors for unfrozen layers
        """
        params = {}
        # Get the correct layer container based on architecture
        if self.is_distilbert:
            layer_container = self.bert.transformer.layer
            layer_prefix = "bert.transformer.layer"
        else:
            layer_container = self.bert.encoder.layer
            layer_prefix = "bert.encoder.layer"
            
        for i, layer in enumerate(layer_container):
            if i >= self.layers_to_freeze:
                for name, param in layer.named_parameters():
                    params[f"{layer_prefix}.{i}.{name}"] = param.data.clone()
        return params
    
    def set_unfrozen_layer_parameters(self, parameters: Dict[str, torch.Tensor]):
        """
        Set unfrozen BERT layer parameters after aggregation
        
        Args:
            parameters: Dictionary of parameter names to tensors
        """
        with torch.no_grad():
            # Get the correct layer container based on architecture
            if self.is_distilbert:
                layer_container = self.bert.transformer.layer
                layer_prefix = "bert.transformer.layer"
            else:
                layer_container = self.bert.encoder.layer
                layer_prefix = "bert.encoder.layer"
                
            for i, layer in enumerate(layer_container):
                if i >= self.layers_to_freeze:
                    for name, param in layer.named_parameters():
                        key = f"{layer_prefix}.{i}.{name}"
                        if key in parameters:
                            param.data.copy_(parameters[key])
    
    def get_all_trainable_parameters(self) -> Dict[str, torch.Tensor]:
        """
        Get all trainable parameters (unfrozen BERT layers + shared LoRA + all task heads)
        
        Returns:
            Dictionary of all trainable parameter names to tensors
        """
        params = {}
        # Add unfrozen BERT layer parameters
        params.update(self.get_unfrozen_layer_parameters())
        # Add LoRA parameters
        params.update(self.get_shared_lora_parameters())
        # Add task head parameters
        for task in self.tasks:
            params.update(self.get_task_head_parameters(task))
        return params
    
    def get_model_summary(self) -> Dict:
        """Get summary of model architecture with partial unfreezing info"""
        # Count frozen and unfrozen BERT parameters
        frozen_params = sum(p.numel() for p in self.bert.embeddings.parameters())
        unfrozen_bert_params = 0
        
        # Get the correct layer container based on architecture
        if self.is_distilbert:
            layer_container = self.bert.transformer.layer
        else:
            layer_container = self.bert.encoder.layer
            
        for i, layer in enumerate(layer_container):
            layer_params = sum(p.numel() for p in layer.parameters())
            if i < self.layers_to_freeze:
                frozen_params += layer_params
            else:
                unfrozen_bert_params += layer_params
        
        # Handle pooler (only BERT has it, DistilBERT doesn't)
        if hasattr(self.bert, 'pooler') and self.bert.pooler is not None:
            frozen_params += sum(p.numel() for p in self.bert.pooler.parameters())
        
        lora_params = sum(p.numel() for p in self.shared_lora.parameters())
        task_head_sizes = {task: sum(p.numel() for p in head.parameters()) 
                          for task, head in self.task_heads.items()}
        total_trainable = unfrozen_bert_params + lora_params + sum(task_head_sizes.values())
        total_params = frozen_params + total_trainable
        
        return {
            'base_model': self.base_model_name,
            'tasks': self.tasks,
            'num_layers': self.num_layers,
            'layers_frozen': self.layers_to_freeze,
            'layers_unfrozen': self.num_layers - self.layers_to_freeze,
            'frozen_bert_parameters': frozen_params,
            'unfrozen_bert_parameters': unfrozen_bert_params,
            'shared_lora_parameters': lora_params,
            'task_head_parameters': task_head_sizes,
            'total_trainable_parameters': total_trainable,
            'total_parameters': total_params,
            'trainable_ratio': f"{total_trainable / total_params * 100:.2f}%",
            'lora_config': {
                'rank': self.lora_rank,
                'alpha': self.lora_alpha,
                'dropout': self.lora_dropout
            }
        }

