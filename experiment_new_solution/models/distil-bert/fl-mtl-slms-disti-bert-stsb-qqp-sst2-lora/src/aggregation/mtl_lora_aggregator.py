#!/usr/bin/env python3
"""
Multi-Task Learning LoRA Aggregator for Federated Learning
Task-aware aggregation: shared LoRA from all clients, task heads from same-task clients
"""

import torch
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

class MTLLoRAAggregator:
    """
    MTL-aware LoRA Aggregator for Federated Learning (Option 1)
    
    Aggregation Strategy:
    1. Shared LoRA Adapters: FedAvg across ALL clients (cross-task knowledge transfer)
    2. Task-Specific Heads: FedAvg within same-task clients only (task specialization)
    
    This maintains the benefits of MTL while adding parameter efficiency from LoRA.
    """
    
    def __init__(self):
        self.aggregation_history = []
    
    def aggregate_mtl_lora_updates(self, client_updates: List[Dict]) -> Dict:
        """
        Perform MTL-aware LoRA aggregation
        
        Args:
            client_updates: List of dictionaries containing:
                - 'client_id': Client identifier
                - 'task': Task name ('sst2', 'qqp', or 'stsb')
                - 'lora_updates': Model parameters (LoRA + task heads)
                - 'metrics': Training metrics
        
        Returns:
            Dictionary with:
                - 'shared_lora': Aggregated shared LoRA parameters (from ALL clients)
                - 'task_heads': Dict of aggregated task-specific head parameters per task
        """
        if not client_updates:
            logger.warning("No client updates to aggregate")
            return {'shared_lora': {}, 'task_heads': {}}
        
        # Group updates by task
        task_groups = {}
        for update in client_updates:
            task = self._extract_task_from_update(update)
            if task not in task_groups:
                task_groups[task] = []
            task_groups[task].append(update)
        
        logger.info(f"MTL LoRA Aggregation: {len(client_updates)} clients across {len(task_groups)} tasks")
        for task, updates in task_groups.items():
            logger.info(f"  Task '{task}': {len(updates)} clients")
        
        # Extract all parameters from client updates
        all_client_params = []
        for update in client_updates:
            params = update.get('lora_updates', {})
            if params:
                all_client_params.append(params)
        
        if not all_client_params:
            logger.warning("No valid parameters found in client updates")
            return {'shared_lora': {}, 'task_heads': {}}
        
        # Separate shared LoRA parameters from task-specific parameters
        shared_lora_list = []
        task_head_params = {task: [] for task in task_groups.keys()}
        
        for i, update in enumerate(client_updates):
            task = self._extract_task_from_update(update)
            params = all_client_params[i]
            
            # Separate shared LoRA and task-specific parameters
            shared_lora = {}
            task_specific = {}
            
            for param_name, param_value in params.items():
                if 'shared_lora' in param_name:
                    # Shared LoRA parameter (aggregated from ALL clients)
                    shared_lora[param_name] = param_value
                elif 'task_heads.' in param_name:
                    # Task-specific head parameter (aggregated within same-task clients)
                    task_specific[param_name] = param_value
                else:
                    # Default: treat as shared if unclear
                    logger.warning(f"Ambiguous parameter name: {param_name}, treating as shared")
                    shared_lora[param_name] = param_value
            
            shared_lora_list.append(shared_lora)
            if task_specific:
                task_head_params[task].append(task_specific)
        
        # Aggregate shared LoRA parameters using FedAvg across ALL clients
        aggregated_shared_lora = self._fedavg(shared_lora_list)
        logger.info(f"Aggregated {len(aggregated_shared_lora)} shared LoRA parameters from ALL {len(client_updates)} clients")
        
        # Aggregate task-specific heads using FedAvg within same-task clients
        aggregated_task_heads = {}
        for task, task_params_list in task_head_params.items():
            if task_params_list:
                aggregated_task_heads[task] = self._fedavg(task_params_list)
                logger.info(f"Aggregated {len(aggregated_task_heads[task])} parameters for task '{task}' head from {len(task_params_list)} clients")
            else:
                logger.warning(f"No task-specific parameters found for task '{task}'")
                aggregated_task_heads[task] = {}
        
        # Record aggregation event
        self.aggregation_history.append({
            'total_clients': len(client_updates),
            'task_distribution': {task: len(updates) for task, updates in task_groups.items()},
            'shared_lora_params_count': len(aggregated_shared_lora),
            'task_heads_count': {task: len(params) for task, params in aggregated_task_heads.items()}
        })
        
        return {
            'shared_lora': aggregated_shared_lora,
            'task_heads': aggregated_task_heads
        }
    
    def _fedavg(self, params_list: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Federated Averaging (FedAvg) algorithm
        
        Args:
            params_list: List of parameter dictionaries from different clients
            
        Returns:
            Averaged parameters
        """
        if not params_list:
            return {}
        
        # Get all parameter names
        param_names = set()
        for params in params_list:
            param_names.update(params.keys())
        
        averaged_params = {}
        
        for param_name in param_names:
            # Collect this parameter from all clients that have it
            param_tensors = []
            for params in params_list:
                if param_name in params:
                    param_value = params[param_name]
                    
                    # Convert to tensor if needed
                    if isinstance(param_value, torch.Tensor):
                        param_tensors.append(param_value.cpu())
                    elif isinstance(param_value, (list, tuple)):
                        param_tensors.append(torch.tensor(param_value))
                    else:
                        param_tensors.append(torch.tensor([param_value]))
            
            # Average across clients
            if param_tensors:
                try:
                    stacked = torch.stack(param_tensors)
                    averaged_params[param_name] = torch.mean(stacked, dim=0)
                except Exception as e:
                    logger.error(f"Error averaging parameter {param_name}: {e}")
                    # Use first client's value as fallback
                    averaged_params[param_name] = param_tensors[0]
        
        return averaged_params
    
    def _extract_task_from_update(self, update: Dict) -> str:
        """
        Extract task name from client update
        
        Args:
            update: Client update dictionary
            
        Returns:
            Task name ('sst2', 'qqp', or 'stsb')
        """
        # Try to get task from update directly
        if 'task' in update:
            return update['task']
        
        # Try to infer from client_id
        client_id = update.get('client_id', '')
        if 'sst2' in client_id.lower():
            return 'sst2'
        elif 'qqp' in client_id.lower():
            return 'qqp'
        elif 'stsb' in client_id.lower():
            return 'stsb'
        
        # Try to infer from metrics
        metrics = update.get('metrics', {})
        if metrics:
            # Get the first task from metrics
            for key in metrics.keys():
                if key in ['sst2', 'qqp', 'stsb']:
                    return key
        
        # Default fallback
        logger.warning(f"Could not determine task for client {client_id}, defaulting to 'sst2'")
        return 'sst2'
    
    def get_aggregation_summary(self) -> Dict:
        """Get summary of aggregation history"""
        if not self.aggregation_history:
            return {'total_aggregations': 0}
        
        recent = self.aggregation_history[-1]
        return {
            'total_aggregations': len(self.aggregation_history),
            'recent_clients': recent['total_clients'],
            'recent_task_distribution': recent['task_distribution'],
            'recent_shared_lora_params': recent['shared_lora_params_count'],
            'recent_task_heads': recent['task_heads_count']
        }
    
    def print_aggregation_stats(self):
        """Print detailed aggregation statistics"""
        if not self.aggregation_history:
            logger.info("No aggregation history available")
            return
        
        logger.info("=" * 70)
        logger.info("MTL LoRA Aggregation Statistics")
        logger.info("=" * 70)
        
        summary = self.get_aggregation_summary()
        logger.info(f"Total Aggregation Rounds: {summary['total_aggregations']}")
        logger.info(f"Recent Round:")
        logger.info(f"  - Total Clients: {summary['recent_clients']}")
        logger.info(f"  - Task Distribution: {summary['recent_task_distribution']}")
        logger.info(f"  - Shared LoRA Params: {summary['recent_shared_lora_params']}")
        logger.info(f"  - Task Heads: {summary['recent_task_heads']}")
        logger.info("=" * 70)
