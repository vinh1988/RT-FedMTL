"""
MiniLM Model for Centralized Training
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import Dict, List, Optional

class MiniLMModel(nn.Module):
    """MiniLM model for centralized training"""
    
    def __init__(self, model_name: str, num_labels: int, max_length: int = 128):
        super().__init__()
        self.model_name = model_name
        self.num_labels = num_labels
        self.max_length = max_length
        
        # Load pretrained model
        self.backbone = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.backbone.config.hidden_size, num_labels)
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, 
                labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Forward pass"""
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        result = {"logits": logits}
        
        if labels is not None:
            # STSB is a regression task - use MSE loss
            # For regression, we need to squeeze logits to match label shape
            if logits.dim() > 1 and logits.size(-1) == 1:
                logits = logits.squeeze(-1)  # Remove last dimension for regression
            
            loss_fn = nn.MSELoss()
            loss = loss_fn(logits, labels.float())
            result["loss"] = loss
            
        return result
    
    def tokenize(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """Tokenize input texts"""
        return self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
