#!/usr/bin/env python3
"""
LoRA (Low-Rank Adaptation) Implementation
Parameter-efficient fine-tuning for federated learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Optional, Union
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class LoRALayer(nn.Module):
    """LoRA layer implementation for parameter-efficient adaptation"""

    def __init__(self, in_features: int, out_features: int, rank: int = 8, alpha: float = 16.0, dropout: float = 0.1):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout

        # Low-rank matrices A and B
        self.lora_A = nn.Parameter(torch.randn(rank, in_features))
        self.lora_B = nn.Parameter(torch.randn(out_features, rank))

        # Dropout for regularization
        self.dropout_layer = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Scaling factor
        self.scaling = alpha / rank

        # Initialize parameters
        nn.init.normal_(self.lora_A, mean=0, std=0.02)
        nn.init.normal_(self.lora_B, mean=0, std=0.02)

    def forward(self, x):
        """Forward pass with LoRA adaptation"""
        # Apply dropout
        x = self.dropout_layer(x)

        # LoRA computation: (B @ A) * scaling
        lora_output = (self.lora_B @ self.lora_A) * self.scaling

        return x + lora_output

    def get_lora_params(self) -> Dict[str, torch.Tensor]:
        """Get LoRA parameters for serialization"""
        return {
            'lora_A': self.lora_A.data.clone(),
            'lora_B': self.lora_B.data.clone(),
            'rank': self.rank,
            'alpha': self.alpha
        }

    def load_lora_params(self, params: Dict[str, torch.Tensor]):
        """Load LoRA parameters from serialized data"""
        self.lora_A.data = params['lora_A'].clone()
        self.lora_B.data = params['lora_B'].clone()
        self.rank = params['rank']
        self.alpha = params['alpha']

class LoRAFederatedModel(nn.Module):
    """Federated model with LoRA adapters for multi-task learning"""

    def __init__(self, base_model_name: str, tasks: List[str], lora_rank: int = 8, lora_alpha: float = 16.0):
        super().__init__()

        # Load base model (parameters will be frozen)
        self.base_model = AutoModelForSequenceClassification.from_pretrained(
            base_model_name, num_labels=1  # For KD compatibility
        )

        # Freeze base model parameters
        for param in self.base_model.parameters():
            param.requires_grad = False

        # Task-specific LoRA adapters
        self.task_adapters = nn.ModuleDict({
            task: LoRALayer(
                in_features=self.base_model.config.hidden_size,
                out_features=self.get_num_labels(task),
                rank=lora_rank,
                alpha=lora_alpha
            ) for task in tasks
        })

        self.tasks = tasks
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha

    def get_num_labels(self, task: str) -> int:
        """Get number of labels for each task"""
        task_labels = {
            'sst2': 2,  # Binary classification
            'qqp': 2,   # Binary classification
            'stsb': 1   # Regression
        }
        return task_labels.get(task, 2)

    def forward(self, input_ids, attention_mask, task_name):
        """Forward pass with task-specific LoRA"""
        # Base model forward pass
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )

        # Get hidden states from last layer
        hidden_states = outputs.hidden_states[-1]

        # Apply task-specific LoRA adaptation
        if task_name in self.task_adapters:
            lora_output = self.task_adapters[task_name](hidden_states)
            # Combine base logits with LoRA adaptation
            combined_logits = outputs.logits + lora_output
        else:
            combined_logits = outputs.logits

        return combined_logits

    def get_all_lora_params(self) -> Dict[str, Dict]:
        """Get LoRA parameters for all tasks"""
        return {
            task: adapter.get_lora_params()
            for task, adapter in self.task_adapters.items()
        }

    def load_all_lora_params(self, all_params: Dict[str, Dict]):
        """Load LoRA parameters for all tasks"""
        for task, params in all_params.items():
            if task in self.task_adapters:
                self.task_adapters[task].load_lora_params(params)

    def get_task_dataloader(self, task: str, batch_size: int = 8):
        """Get DataLoader for a specific task"""
        # This is a placeholder - in practice, you'd use the actual dataset handlers
        # For now, return a dummy dataloader
        from torch.utils.data import DataLoader, TensorDataset

        # Create dummy data for demonstration
        dummy_input_ids = torch.randint(0, 1000, (10, 128))
        dummy_attention_mask = torch.ones(10, 128)
        dummy_labels = torch.randint(0, 2, (10,))

        dataset = TensorDataset(dummy_input_ids, dummy_attention_mask, dummy_labels)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)

class LoRAAggregator:
    """Aggregates LoRA parameters from multiple clients"""

    def __init__(self):
        self.aggregation_history = []

    def aggregate_lora_updates(self, client_updates: List[Dict], client_weights: List[float] = None) -> Dict[str, Dict]:
        """Aggregate LoRA parameters using federated averaging"""
        if not client_updates:
            return {}

        if client_weights is None:
            # Equal weighting if no weights provided
            client_weights = [1.0 / len(client_updates)] * len(client_updates)

        aggregated_params = {}

        # Get all unique tasks across clients
        all_tasks = set()
        for update in client_updates:
            all_tasks.update(update['lora_updates'].keys())

        # Aggregate parameters for each task
        for task in all_tasks:
            task_params = {}

            # Get parameters for this task from all clients that have it
            task_updates = []
            task_weights = []

            for i, update in enumerate(client_updates):
                if task in update['lora_updates']:
                    task_updates.append(update['lora_updates'][task])
                    task_weights.append(client_weights[i])

            if task_updates:
                # Aggregate each parameter type
                for param_name in task_updates[0].keys():
                    if param_name in ['lora_A', 'lora_B']:
                        # Weighted average of parameter matrices
                        weighted_sum = sum(
                            update[param_name] * weight
                            for update, weight in zip(task_updates, task_weights)
                        )
                        task_params[param_name] = weighted_sum

                # Preserve metadata
                task_params['rank'] = task_updates[0]['rank']
                task_params['alpha'] = task_updates[0]['alpha']

                aggregated_params[task] = task_params

        # Record aggregation
        self.aggregation_history.append({
            'timestamp': torch.tensor([0.0]),  # Placeholder for timestamp
            'num_clients': len(client_updates),
            'tasks_aggregated': list(aggregated_params.keys()),
            'aggregation_weights': client_weights
        })

        return aggregated_params

    def get_aggregation_summary(self) -> Dict:
        """Get summary of aggregation history"""
        return {
            'total_aggregations': len(self.aggregation_history),
            'average_clients_per_aggregation': sum(
                agg['num_clients'] for agg in self.aggregation_history
            ) / len(self.aggregation_history) if self.aggregation_history else 0,
            'unique_tasks_aggregated': list(set(
                task for agg in self.aggregation_history
                for task in agg['tasks_aggregated']
            ))
        }
