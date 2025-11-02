from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoConfig
from peft import LoraConfig, get_peft_model, PeftModel


@dataclass
class FederatedBERTConfig:
    model_name: str = "prajjwal1/bert-tiny"
    num_labels: int = 2
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    
    # Model configuration
    freeze_client_model: bool = True
    apply_lora_to_teacher: bool = True
    apply_lora_to_student: bool = True
    lora_target_modules: list = field(default_factory=lambda: ["query", "value"])
    
    # For client-side only (might differ from server model)
    client_model_name: str = "prajjwal1/bert-tiny"


class FederatedBERTBase(nn.Module):
    def __init__(self, config: FederatedBERTConfig):
        super().__init__()
        self.config = config
        
        # Load base model
        logger.info(f"Loading base model from {config.model_name}")
        base_config = AutoConfig.from_pretrained(config.model_name, num_labels=config.num_labels)
        self.model = AutoModelForSequenceClassification.from_pretrained(config.model_name, config=base_config)
        
        # Log initial parameter status
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Model loaded with {total_params:,} total parameters")
        
        # Apply LoRA if specified
        if getattr(config, 'apply_lora_to_teacher', True) or getattr(config, 'apply_lora_to_student', True):
            logger.info(f"Applying LoRA with r={config.lora_r}, alpha={config.lora_alpha}, "
                      f"dropout={config.lora_dropout} to modules: {config.lora_target_modules}")
            
            lora_config = LoraConfig(
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                bias="none",
                task_type="SEQ_CLS",
                target_modules=config.lora_target_modules
            )
            self.model = get_peft_model(self.model, lora_config)
            
            # Log LoRA parameters
            lora_params = sum(p.numel() for name, p in self.model.named_parameters() if 'lora_' in name)
            logger.info(f"Added {lora_params:,} LoRA parameters")
            
            # Freeze base model parameters if specified
            if getattr(config, 'freeze_client_model', False):
                logger.info("Freezing base model parameters")
                for name, param in self.model.named_parameters():
                    if 'lora_' not in name:  # Only freeze non-LoRA parameters
                        param.requires_grad = False
                        logger.debug(f"Frozen parameter: {name}")
                logger.info("Base model parameters frozen, only LoRA parameters are trainable")
            else:
                logger.info("Base model parameters remain trainable")
        else:
            logger.info("LoRA not applied to this model instance")
        
        # Print trainable parameters
        self.model.print_trainable_parameters()
        
        # Log detailed parameter information
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        logger.info(f"Parameter status - Trainable: {trainable_params:,}, Frozen: {frozen_params:,}, "
                   f"Total: {total_params:,}")
        
        # Log first few parameter names and their status
        logger.debug("Sample parameters and their training status:")
        for name, param in list(self.model.named_parameters())[:5]:  # First 5 params as sample
            logger.debug(f"  {name}: {'Trainable' if param.requires_grad else 'Frozen'}")
        if len(list(self.model.named_parameters())) > 5:
            logger.debug("  ... and more parameters ...")
        
    def get_lora_parameters(self) -> Dict[str, torch.Tensor]:
        return {name: param for name, param in self.model.named_parameters() if "lora" in name}

    def set_lora_parameters(self, lora_params: Dict[str, torch.Tensor]) -> None:
        for name, param in self.model.named_parameters():
            if "lora" in name and name in lora_params:
                param.data.copy_(lora_params[name].data)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: Optional[torch.Tensor] = None, return_hidden_states: bool = False):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=return_hidden_states
        )
        
        # For knowledge transfer, we might need projected hidden states
        # This is a placeholder, actual projection might be needed based on KT method
        projected_hidden_states = None
        if return_hidden_states and outputs.hidden_states:
            # Assuming we take the last hidden state of the [CLS] token as representation
            hidden_states = outputs.hidden_states[-1][:, 0, :]
            # A simple projection if needed. For now, just return it.
            projected_hidden_states = hidden_states

        return {"logits": outputs.logits, "loss": outputs.loss, "hidden_states": projected_hidden_states}


class FederatedBERTServer(FederatedBERTBase):
    def __init__(self, config: FederatedBERTConfig):
        super().__init__(config)


class FederatedBERTClient(FederatedBERTBase):
    def __init__(self, config: FederatedBERTConfig):
        super().__init__(config)
