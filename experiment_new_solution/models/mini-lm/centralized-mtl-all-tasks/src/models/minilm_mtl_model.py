"""
MiniLM Multi-Task Learning Model for Centralized Training
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import Dict, List, Optional, Union

class MiniLMMTLModel(nn.Module):
    """MiniLM model for multi-task learning"""
    
    def __init__(self, model_name: str, task_configs: Dict, max_length: int = 128):
        super().__init__()
        self.model_name = model_name
        self.task_configs = task_configs
        self.max_length = max_length
        
        # Load pretrained model
        self.backbone = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.hidden_size = self.backbone.config.hidden_size
        
        # Create task-specific heads
        self.task_heads = nn.ModuleDict()
        for task_name, task_config in task_configs.items():
            num_labels = task_config['num_labels']
            task_type = task_config['task_type']
            
            if task_type == 'classification':
                # Classification head
                self.task_heads[task_name] = nn.Sequential(
                    nn.Linear(self.hidden_size, self.hidden_size // 2),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(self.hidden_size // 2, num_labels)
                )
            elif task_type == 'regression':
                # Regression head
                self.task_heads[task_name] = nn.Sequential(
                    nn.Linear(self.hidden_size, self.hidden_size // 2),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(self.hidden_size // 2, num_labels)
                )
            else:
                raise ValueError(f"Unsupported task type: {task_type}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, 
                task_name: str, labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Forward pass for specific task"""
        # Get backbone features
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        
        # Get task-specific predictions
        if task_name not in self.task_heads:
            raise ValueError(f"Task {task_name} not found in task heads")
        
        logits = self.task_heads[task_name](pooled_output)
        
        result = {"logits": logits}
        
        if labels is not None:
            task_config = self.task_configs[task_name]
            task_type = task_config['task_type']
            
            if task_type == 'classification':
                loss_fn = nn.CrossEntropyLoss()
                result["loss"] = loss_fn(logits, labels)
            elif task_type == 'regression':
                loss_fn = nn.MSELoss()
                result["loss"] = loss_fn(logits.squeeze(), labels.squeeze())
            else:
                raise ValueError(f"Unsupported task type: {task_type}")
            
        return result
    
    def tokenize(self, texts: Union[str, List[str]], pair_texts: Optional[Union[str, List[str]]] = None) -> Dict[str, torch.Tensor]:
        """Tokenize input texts"""
        if pair_texts is not None:
            # For sentence pair tasks (QQP, STSB)
            return self.tokenizer(
                texts,
                pair_texts,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
        else:
            # For single sentence tasks (SST2)
            return self.tokenizer(
                texts,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
