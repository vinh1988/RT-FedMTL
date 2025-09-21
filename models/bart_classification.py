#!/usr/bin/env python3
"""
BART Models for Classification with LoRA Support
Implements DistilBART and MobileBART models for federated learning classification tasks.
"""

import torch
import torch.nn as nn
from transformers import (
    BartForConditionalGeneration,
    AutoConfig,
    AutoModel,
    BartTokenizer
)
from peft import LoraConfig, get_peft_model, TaskType
from typing import Dict, List, Optional, Union
import logging

logger = logging.getLogger(__name__)


class BARTClassificationHead(nn.Module):
    """Classification head for BART models"""
    
    def __init__(self, hidden_size: int, num_labels: int, dropout: float = 0.1):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hidden_size, num_labels)
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = torch.tanh(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return self.out_proj(hidden_states)


class DistilBARTClassifier(nn.Module):
    """DistilBART model for classification"""
    
    def __init__(self, model_name: str = "facebook/distilbart-cnn-12-6", 
                 num_labels: int = 20, max_length: int = 512):
        super().__init__()
        self.model_name = model_name
        self.num_labels = num_labels
        self.max_length = max_length
        
        # Load base model (using regular BART for DistilBART-like functionality)
        self.base_model = BartForConditionalGeneration.from_pretrained(model_name)
        self.config = self.base_model.config
        
        # Add classification head
        self.classifier = BARTClassificationHead(
            hidden_size=self.config.d_model,
            num_labels=num_labels
        )
        
        # Initialize weights
        self._init_classifier_weights()
        
    def _init_classifier_weights(self):
        """Initialize classification head weights"""
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=self.config.init_std)
                if module.bias is not None:
                    module.bias.data.zero_()
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, 
                labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Forward pass for classification"""
        
        # Get encoder outputs
        encoder_outputs = self.base_model.get_encoder()(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Use [CLS] token (first token) for classification
        pooled_output = encoder_outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_size]
        
        # Classification
        logits = self.classifier(pooled_output)
        
        outputs = {"logits": logits}
        
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs["loss"] = loss
        
        return outputs
    
    def generate_logits(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Generate logits for knowledge distillation"""
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask)
            return outputs["logits"]


class MobileBARTClassifier(nn.Module):
    """MobileBART model for classification"""
    
    def __init__(self, model_name: str = "valhalla/mobile-bart", 
                 num_labels: int = 20, max_length: int = 256):
        super().__init__()
        self.model_name = model_name
        self.num_labels = num_labels
        self.max_length = max_length
        
        try:
            # Try to load MobileBART
            self.base_model = AutoModel.from_pretrained(model_name)
            self.config = self.base_model.config
        except:
            # Fallback to regular BART if MobileBART not available
            logger.warning(f"MobileBART not found, using regular BART: {model_name}")
            self.base_model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")
            self.config = self.base_model.config
        
        # Add classification head
        self.classifier = BARTClassificationHead(
            hidden_size=self.config.d_model,
            num_labels=num_labels
        )
        
        # Initialize weights
        self._init_classifier_weights()
        
    def _init_classifier_weights(self):
        """Initialize classification head weights"""
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=self.config.init_std)
                if module.bias is not None:
                    module.bias.data.zero_()
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, 
                labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Forward pass for classification"""
        
        # Get encoder outputs
        encoder_outputs = self.base_model.get_encoder()(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Use [CLS] token (first token) for classification
        pooled_output = encoder_outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_size]
        
        # Classification
        logits = self.classifier(pooled_output)
        
        outputs = {"logits": logits}
        
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs["loss"] = loss
        
        return outputs
    
    def generate_logits(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Generate logits for knowledge distillation"""
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask)
            return outputs["logits"]


class LoRABARTModel:
    """LoRA-enabled BART model wrapper"""
    
    def __init__(self, model: nn.Module, lora_config: LoraConfig):
        self.base_model = model
        self.lora_config = lora_config
        self.model = get_peft_model(model, lora_config)
        
    def get_model(self):
        return self.model
    
    def print_trainable_parameters(self):
        """Print trainable parameters"""
        trainable_params = 0
        all_param = 0
        for _, param in self.model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(f"trainable params: {trainable_params:,} || all params: {all_param:,} || trainable%: {100 * trainable_params / all_param:.2f}")


def create_distilbart_model(model_name: str = "facebook/distilbart-cnn-12-6",
                          num_labels: int = 20,
                          max_length: int = 512,
                          lora_config: Optional[LoraConfig] = None) -> nn.Module:
    """Create DistilBART model with optional LoRA"""
    
    # Create base model
    model = DistilBARTClassifier(model_name, num_labels, max_length)
    
    # Apply LoRA if provided
    if lora_config is not None:
        lora_model = LoRABARTModel(model, lora_config)
        return lora_model.get_model()
    
    return model


def create_mobilebart_model(model_name: str = "valhalla/mobile-bart",
                          num_labels: int = 20,
                          max_length: int = 256,
                          lora_config: Optional[LoraConfig] = None) -> nn.Module:
    """Create MobileBART model with optional LoRA"""
    
    # Create base model
    model = MobileBARTClassifier(model_name, num_labels, max_length)
    
    # Apply LoRA if provided
    if lora_config is not None:
        lora_model = LoRABARTModel(model, lora_config)
        return lora_model.get_model()
    
    return model


def create_lora_config(r: int = 8, lora_alpha: int = 16, lora_dropout: float = 0.1,
                      target_modules: List[str] = None) -> LoraConfig:
    """Create LoRA configuration"""
    
    if target_modules is None:
        target_modules = ["q_proj", "v_proj"]
    
    return LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
    )


def test_models():
    """Test function for the models"""
    print("Testing BART Classification Models...")
    
    # Test DistilBART
    print("\n1. Testing DistilBART...")
    distilbart = DistilBARTClassifier(num_labels=20)
    print(f"DistilBART parameters: {sum(p.numel() for p in distilbart.parameters()):,}")
    
    # Test MobileBART
    print("\n2. Testing MobileBART...")
    mobilebart = MobileBARTClassifier(num_labels=20)
    print(f"MobileBART parameters: {sum(p.numel() for p in mobilebart.parameters()):,}")
    
    # Test with LoRA
    print("\n3. Testing with LoRA...")
    lora_config = create_lora_config(r=8, lora_alpha=16)
    
    distilbart_lora = create_distilbart_model(lora_config=lora_config)
    mobilebart_lora = create_mobilebart_model(lora_config=lora_config)
    
    print("LoRA models created successfully!")
    
    # Test forward pass
    print("\n4. Testing forward pass...")
    batch_size = 2
    seq_length = 128
    
    input_ids = torch.randint(0, 1000, (batch_size, seq_length))
    attention_mask = torch.ones(batch_size, seq_length)
    labels = torch.randint(0, 20, (batch_size,))
    
    # Test DistilBART
    outputs = distilbart(input_ids, attention_mask, labels)
    print(f"DistilBART output shape: {outputs['logits'].shape}")
    print(f"DistilBART loss: {outputs['loss'].item():.4f}")
    
    # Test MobileBART
    outputs = mobilebart(input_ids, attention_mask, labels)
    print(f"MobileBART output shape: {outputs['logits'].shape}")
    print(f"MobileBART loss: {outputs['loss'].item():.4f}")
    
    print("\nAll tests passed successfully!")


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Test the models
    test_models()
