"""
FedAvg aggregation for federated BERT learning with LoRA.
Implements weighted averaging of LoRA parameters.
"""

import torch
from typing import Dict, List, Optional, Tuple, Any
from collections import OrderedDict
import numpy as np
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class AggregationConfig:
    """Configuration for federated aggregation"""
    weighting_strategy: str = "uniform"  # uniform, data_size, loss_based
    clip_norm: Optional[float] = None  # Gradient clipping
    momentum: float = 0.0  # Server momentum
    adaptive_weighting: bool = False  # Adaptive client weighting
    min_weight: float = 0.01  # Minimum client weight
    max_weight: float = 1.0   # Maximum client weight


class FedAvgAggregator:
    """
    Base FedAvg aggregator for federated learning.
    Implements weighted averaging of model parameters.
    """
    
    def __init__(self, config: AggregationConfig):
        self.config = config
        self.server_momentum = None
        self.round_num = 0
        
    def compute_client_weights(self, client_info: List[Dict[str, Any]]) -> List[float]:
        """Compute aggregation weights for clients"""
        
        if self.config.weighting_strategy == "uniform":
            # Equal weights for all clients
            num_clients = len(client_info)
            return [1.0 / num_clients] * num_clients
            
        elif self.config.weighting_strategy == "data_size":
            # Weight by local dataset size
            data_sizes = [info.get("data_size", 1) for info in client_info]
            total_size = sum(data_sizes)
            return [size / total_size for size in data_sizes]
            
        elif self.config.weighting_strategy == "loss_based":
            # Weight inversely by local loss (lower loss = higher weight)
            losses = [info.get("loss", 1.0) for info in client_info]
            # Invert losses and normalize
            inv_losses = [1.0 / (loss + 1e-8) for loss in losses]
            total_inv_loss = sum(inv_losses)
            weights = [inv_loss / total_inv_loss for inv_loss in inv_losses]
            
            # Apply min/max constraints
            weights = [max(min(w, self.config.max_weight), self.config.min_weight) 
                      for w in weights]
            # Renormalize
            total_weight = sum(weights)
            return [w / total_weight for w in weights]
        
        else:
            raise ValueError(f"Unknown weighting strategy: {self.config.weighting_strategy}")
    
    def aggregate_parameters(self, client_parameters: List[Dict[str, torch.Tensor]],
                           client_weights: List[float]) -> Dict[str, torch.Tensor]:
        """Aggregate client parameters using weighted averaging"""
        
        if not client_parameters:
            raise ValueError("No client parameters to aggregate")
        
        # Initialize aggregated parameters
        aggregated = {}
        
        # Get parameter names from first client
        param_names = client_parameters[0].keys()
        
        for param_name in param_names:
            # Collect parameters from all clients
            param_list = []
            weight_list = []
            
            for client_params, weight in zip(client_parameters, client_weights):
                if param_name in client_params:
                    param_list.append(client_params[param_name])
                    weight_list.append(weight)
            
            if not param_list:
                logger.warning(f"Parameter {param_name} not found in any client")
                continue
            
            # Weighted average
            weighted_sum = torch.zeros_like(param_list[0])
            total_weight = 0.0
            
            for param, weight in zip(param_list, weight_list):
                weighted_sum += weight * param
                total_weight += weight
            
            aggregated[param_name] = weighted_sum / total_weight if total_weight > 0 else weighted_sum
        
        return aggregated
    
    def apply_clipping(self, parameters: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply gradient clipping if configured"""
        if self.config.clip_norm is None:
            return parameters
        
        # Compute total norm
        total_norm = 0.0
        for param in parameters.values():
            total_norm += param.norm().item() ** 2
        total_norm = total_norm ** 0.5
        
        # Apply clipping
        if total_norm > self.config.clip_norm:
            clip_factor = self.config.clip_norm / total_norm
            parameters = {name: param * clip_factor for name, param in parameters.items()}
        
        return parameters
    
    def apply_momentum(self, current_params: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply server-side momentum"""
        if self.config.momentum == 0.0:
            self.server_momentum = current_params
            return current_params
        
        if self.server_momentum is None:
            self.server_momentum = {name: torch.zeros_like(param) 
                                  for name, param in current_params.items()}
        
        # Update momentum
        updated_momentum = {}
        updated_params = {}
        
        for name, param in current_params.items():
            if name in self.server_momentum:
                # momentum = β * momentum + (1-β) * current_grad
                updated_momentum[name] = (self.config.momentum * self.server_momentum[name] + 
                                        (1 - self.config.momentum) * param)
                updated_params[name] = updated_momentum[name]
            else:
                updated_momentum[name] = param
                updated_params[name] = param
        
        self.server_momentum = updated_momentum
        return updated_params
    
    def aggregate(self, client_parameters: List[Dict[str, torch.Tensor]],
                 client_info: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Main aggregation method"""
        
        # Compute client weights
        client_weights = self.compute_client_weights(client_info)
        
        logger.info(f"Round {self.round_num}: Aggregating {len(client_parameters)} clients with weights {client_weights}")
        
        # Aggregate parameters
        aggregated_params = self.aggregate_parameters(client_parameters, client_weights)
        
        # Apply clipping
        aggregated_params = self.apply_clipping(aggregated_params)
        
        # Apply momentum
        aggregated_params = self.apply_momentum(aggregated_params)
        
        self.round_num += 1
        return aggregated_params


class LoRAFedAvgAggregator(FedAvgAggregator):
    """
    Specialized FedAvg aggregator for LoRA parameters.
    Handles LoRA-specific parameter aggregation.
    """
    
    def __init__(self, config: AggregationConfig):
        super().__init__(config)
        self.lora_param_names = set()
        
    def identify_lora_parameters(self, parameters: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Separate LoRA parameters from other parameters"""
        lora_params = {}
        other_params = {}
        
        for name, param in parameters.items():
            if "lora" in name.lower():
                lora_params[name] = param
                self.lora_param_names.add(name)
            else:
                other_params[name] = param
        
        return lora_params, other_params
    
    def aggregate_lora_parameters(self, client_parameters: List[Dict[str, torch.Tensor]],
                                client_weights: List[float]) -> Dict[str, torch.Tensor]:
        """Aggregate only LoRA parameters"""
        
        # Extract LoRA parameters from each client
        lora_params_list = []
        for client_params in client_parameters:
            lora_params, _ = self.identify_lora_parameters(client_params)
            lora_params_list.append(lora_params)
        
        # Aggregate LoRA parameters
        return self.aggregate_parameters(lora_params_list, client_weights)
    
    def aggregate(self, client_parameters: List[Dict[str, torch.Tensor]],
                 client_info: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Aggregate LoRA parameters using FedAvg"""
        
        # Compute client weights
        client_weights = self.compute_client_weights(client_info)
        
        logger.info(f"Round {self.round_num}: Aggregating LoRA parameters from {len(client_parameters)} clients")
        
        # Aggregate only LoRA parameters
        aggregated_lora = self.aggregate_lora_parameters(client_parameters, client_weights)
        
        # Apply clipping and momentum to LoRA parameters
        aggregated_lora = self.apply_clipping(aggregated_lora)
        aggregated_lora = self.apply_momentum(aggregated_lora)
        
        self.round_num += 1
        return aggregated_lora
    
    def get_aggregation_stats(self, client_parameters: List[Dict[str, torch.Tensor]]) -> Dict[str, Any]:
        """Get statistics about the aggregation"""
        
        if not client_parameters:
            return {}
        
        # Count LoRA parameters
        lora_param_count = 0
        total_param_count = 0
        
        for name, param in client_parameters[0].items():
            total_param_count += param.numel()
            if "lora" in name.lower():
                lora_param_count += param.numel()
        
        return {
            "total_parameters": total_param_count,
            "lora_parameters": lora_param_count,
            "lora_percentage": (lora_param_count / total_param_count) * 100 if total_param_count > 0 else 0,
            "num_clients": len(client_parameters),
            "lora_param_names": list(self.lora_param_names)
        }


def create_fedavg_aggregator(weighting_strategy: str = "uniform",
                           clip_norm: Optional[float] = None,
                           momentum: float = 0.0,
                           lora_only: bool = True) -> FedAvgAggregator:
    """Factory function to create FedAvg aggregator"""
    
    config = AggregationConfig(
        weighting_strategy=weighting_strategy,
        clip_norm=clip_norm,
        momentum=momentum
    )
    
    if lora_only:
        return LoRAFedAvgAggregator(config)
    else:
        return FedAvgAggregator(config)


if __name__ == "__main__":
    # Test the aggregator
    config = AggregationConfig(weighting_strategy="data_size")
    aggregator = LoRAFedAvgAggregator(config)
    
    # Create dummy client parameters
    client1_params = {
        "bert.encoder.layer.0.attention.self.query.lora_A": torch.randn(16, 768),
        "bert.encoder.layer.0.attention.self.query.lora_B": torch.randn(768, 16),
        "classifier.weight": torch.randn(2, 768)
    }
    
    client2_params = {
        "bert.encoder.layer.0.attention.self.query.lora_A": torch.randn(16, 768),
        "bert.encoder.layer.0.attention.self.query.lora_B": torch.randn(768, 16),
        "classifier.weight": torch.randn(2, 768)
    }
    
    client_params_list = [client1_params, client2_params]
    client_info = [{"data_size": 100}, {"data_size": 150}]
    
    # Test aggregation
    aggregated = aggregator.aggregate(client_params_list, client_info)
    stats = aggregator.get_aggregation_stats(client_params_list)
    
    print("Aggregation completed successfully")
    print(f"Aggregated parameters: {list(aggregated.keys())}")
    print(f"Statistics: {stats}")
