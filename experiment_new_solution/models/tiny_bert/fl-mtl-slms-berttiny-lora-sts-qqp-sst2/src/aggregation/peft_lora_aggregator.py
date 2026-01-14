#!/usr/bin/env python3
"""
PEFT LoRA Aggregator for Federated Multi-Task Learning
Aggregates LoRA adapter parameters across clients with task-aware logic
"""

import torch
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class PEFTLoRAAggregator:
    """
    Aggregates PEFT LoRA parameters in a federated multi-task learning setting
    - Shared LoRA adapters: Aggregated across ALL clients
    - Task-specific heads: Aggregated within same-task clients only
    """
    
    def __init__(self):
        self.aggregation_count = 0
    
    def aggregate_lora_updates(self, client_updates: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Aggregate LoRA parameters from multiple clients
        
        Args:
            client_updates: List of dicts containing:
                - 'client_id': str
                - 'task': str (e.g., 'sst2', 'qqp', 'stsb')
                - 'lora_parameters': Dict[str, torch.Tensor]
                - 'metrics': Dict (optional)
        
        Returns:
            Aggregated LoRA parameters
        """
        if not client_updates:
            logger.warning("No client updates received for aggregation")
            return {}
        
        self.aggregation_count += 1
        logger.info(f"="*60)
        logger.info(f"Aggregation #{self.aggregation_count}: Processing {len(client_updates)} clients")
        
        # Group updates by task
        task_groups: Dict[str, List[Dict]] = {}
        for update in client_updates:
            task = update.get('task')
            if task:
                if task not in task_groups:
                    task_groups[task] = []
                task_groups[task].append(update)
            else:
                logger.warning(f"Client {update.get('client_id', 'unknown')} update missing task label")
        
        # Log task distribution
        for task, updates in task_groups.items():
            logger.info(f"  Task '{task}': {len(updates)} clients")
        
        # Strategy for PEFT LoRA in MTL:
        # 1. Each task has its own LoRA adapter in the model
        # 2. We aggregate each task's LoRA parameters separately
        # 3. Only clients working on the same task contribute to that task's adapter
        
        aggregated_params = {}
        
        for task, task_updates in task_groups.items():
            logger.info(f"Aggregating task '{task}' LoRA parameters from {len(task_updates)} clients")
            
            # Extract all LoRA parameters for this task
            task_params_list = []
            for update in task_updates:
                lora_params = update.get('lora_parameters', {})
                
                # Filter parameters belonging to this task
                task_lora_params = {
                    k: v for k, v in lora_params.items()
                    if k.startswith(f"{task}.")
                }
                
                if task_lora_params:
                    task_params_list.append(task_lora_params)
            
            if not task_params_list:
                logger.warning(f"No LoRA parameters found for task '{task}'")
                continue
            
            # Perform FedAvg on this task's LoRA parameters
            task_aggregated = self._fedavg(task_params_list)
            
            # Merge into global aggregated parameters
            aggregated_params.update(task_aggregated)
            
            logger.info(f"  ✓ Aggregated {len(task_aggregated)} parameters for task '{task}'")
        
        logger.info(f"Total aggregated parameters: {len(aggregated_params)}")
        logger.info(f"="*60)
        
        return aggregated_params
    
    def _fedavg(self, client_param_lists: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Federated Averaging: Average parameters across clients
        """
        if not client_param_lists:
            return {}
        
        # Initialize with zeros based on first client's structure
        first_client_params = client_param_lists[0]
        aggregated_params = {
            name: torch.zeros_like(self._to_tensor(param))
            for name, param in first_client_params.items()
        }
        
        # Sum parameters from all clients
        for client_params in client_param_lists:
            for name, param in client_params.items():
                if name in aggregated_params:
                    aggregated_params[name] += self._to_tensor(param)
                else:
                    logger.warning(f"Parameter '{name}' not found in aggregation structure")
        
        # Average by number of clients
        num_clients = len(client_param_lists)
        for name in aggregated_params:
            aggregated_params[name] /= num_clients
        
        return aggregated_params
    
    def aggregate_weighted(
        self, 
        client_updates: List[Dict],
        weights: List[float] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Weighted aggregation based on client dataset sizes or custom weights
        
        Args:
            client_updates: List of client updates
            weights: Optional list of weights (if None, uses equal weights)
        
        Returns:
            Aggregated LoRA parameters
        """
        if not client_updates:
            return {}
        
        # Use equal weights if not provided
        if weights is None:
            weights = [1.0 / len(client_updates)] * len(client_updates)
        else:
            # Normalize weights to sum to 1
            total_weight = sum(weights)
            weights = [w / total_weight for w in weights]
        
        logger.info(f"Performing weighted aggregation with weights: {weights}")
        
        # Group by task and aggregate with weights
        task_groups: Dict[str, List[tuple]] = {}
        for i, update in enumerate(client_updates):
            task = update.get('task')
            if task:
                if task not in task_groups:
                    task_groups[task] = []
                task_groups[task].append((update, weights[i]))
        
        aggregated_params = {}
        
        for task, task_updates_weights in task_groups.items():
            # Extract task-specific parameters with weights
            task_params_list = []
            task_weights = []
            
            for update, weight in task_updates_weights:
                lora_params = update.get('lora_parameters', {})
                task_lora_params = {
                    k: v for k, v in lora_params.items()
                    if k.startswith(f"{task}.")
                }
                
                if task_lora_params:
                    task_params_list.append(task_lora_params)
                    task_weights.append(weight)
            
            if not task_params_list:
                continue
            
            # Normalize task weights
            total_task_weight = sum(task_weights)
            normalized_task_weights = [w / total_task_weight for w in task_weights]
            
            # Weighted average
            task_aggregated = self._weighted_fedavg(task_params_list, normalized_task_weights)
            aggregated_params.update(task_aggregated)
            
            logger.info(f"  ✓ Weighted aggregation for task '{task}': {len(task_aggregated)} parameters")
        
        return aggregated_params
    
    def _weighted_fedavg(
        self, 
        client_param_lists: List[Dict[str, torch.Tensor]],
        weights: List[float]
    ) -> Dict[str, torch.Tensor]:
        """
        Weighted Federated Averaging
        """
        if not client_param_lists:
            return {}
        
        first_client_params = client_param_lists[0]
        aggregated_params = {
            name: torch.zeros_like(self._to_tensor(param))
            for name, param in first_client_params.items()
        }
        
        # Weighted sum
        for client_params, weight in zip(client_param_lists, weights):
            for name, param in client_params.items():
                if name in aggregated_params:
                    aggregated_params[name] += self._to_tensor(param) * weight
        
        return aggregated_params
    
    def _to_tensor(self, value: Any) -> torch.Tensor:
        """Convert value to tensor, handling lists and existing tensors"""
        if isinstance(value, list):
            return torch.tensor(value, dtype=torch.float32)
        elif isinstance(value, torch.Tensor):
            return value
        else:
            return torch.tensor(value, dtype=torch.float32)
    
    def get_aggregation_stats(self) -> Dict[str, int]:
        """Get aggregation statistics"""
        return {
            "total_aggregations": self.aggregation_count
        }

